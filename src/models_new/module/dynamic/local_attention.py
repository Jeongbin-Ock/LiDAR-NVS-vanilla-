"""Radius local attention: the block V15 runs after the temporal stack."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.experiment_config.definitions import (
    LOCAL_ATTENTION_MAX_QUERY_CHUNK_SIZE,
)

from ...utils.attention import Rotary3D
from .common import _cfg_get

try:
    import flash_attn
except ImportError:  # CPU/unit-test fallback remains available.
    flash_attn = None

# CUDA caps grid dimensions y and z at 65535, and a varlen FlashAttention call
# launches over its sequence count; see ``query_chunk_size`` for the crash this
# guards against. Config validation reads the same limit.
MAX_QUERY_CHUNK_SIZE = LOCAL_ATTENTION_MAX_QUERY_CHUNK_SIZE


def _number_option(cfg, name, default, *, integer=False):
    """``cfg.name`` as a finite float, or as an int when ``integer`` is set."""
    value = _cfg_get(cfg, name, default)
    try:
        number = float(value)
        valid = math.isfinite(number) and not isinstance(value, bool)
    except (TypeError, ValueError):
        valid = False
    if not valid:
        raise ValueError(
            f"local attention {name} must be a finite number, got {value!r}"
        )
    if not integer:
        return number
    if number != int(number):
        raise ValueError(f"local attention {name} must be an integer, got {value!r}")
    return int(number)


class RadiusLocalAttentionBlock(nn.Module):
    """One pre-LN block whose attention neighborhood is a metric ball.

    Every token attends to every token of its own window -- both sweeps, itself
    included -- that lies within ``radius_m`` of it. Positions are the tokens'
    coordinates in the window's reference frame, with no velocity correction,
    and ``sample_index`` names the window: tokens of different windows never
    see each other, and the rows of one window need not be contiguous.

    The tokens are Utonia stage-3 cells on a 0.4 m grid, so the default 0.8 m
    radius is two voxels: a token sees its own cell, the cells around it and
    the ring beyond them, on both sweeps. Measured on real windows (r = 0.8 m,
    about 14k tokens per window) that is 12 neighbors on average -- median 10,
    p95 29, p99 38, at most 80.

    ``x + out_proj(Attn(input_norm(x)))`` is followed by
    ``x + ffn(ffn_norm(x))``. There is no LayerScale, no QK norm, no dropout
    and no learnable position parameter; the FFN's last layer starts at zero,
    so at initialization the block adds only its attention branch.

    **Position encoding.** Q and K carry Utonia's 3D RoPE at their own
    positions, with ``rope_base = 10**1.2`` and
    ``rope_position_scale = 2*pi/0.4``. The per-axis wavelengths
    ``2*pi/(scale*inv_freq)`` are 0.40, 0.63, 1.00, 1.59, 2.52 and 4.00 m for a
    36-wide head. The shortest is one voxel. The longest band's half
    wavelength (2 m) is wider than the radius, so that band never wraps its
    phase inside a neighborhood and still orders every neighbor along each
    axis. The temporal heads use 5-232 m instead: they resolve motion between
    sweeps, not the layout of a surface within two voxels.

    **No key cap, packed keys.** Every neighbor is kept: a cap would drop keys
    exactly where the scene is densest (vehicles, poles). Padding every query
    to the 80-key maximum would spend about 6.6x the work of a 12-key mean, so
    the CUDA path packs the neighbors instead. Each query is one sequence of
    length one in a varlen FlashAttention call (one call per query chunk), and
    its key sequence is that query's neighbor list, gathered from the CSR
    layout that :meth:`radius_neighbors` builds. The fallback path pads and
    masks instead; it is the reference the packed path must equal, and it is
    what CPU tensors always use.
    """

    def __init__(self, cfg, dim: int):
        super().__init__()
        self.dim = int(dim)
        self.radius_m = _number_option(cfg, "radius_m", 0.8)
        self.num_heads = _number_option(cfg, "num_heads", 12, integer=True)
        self.mlp_ratio = _number_option(cfg, "mlp_ratio", 4.0)
        rope_base = _number_option(cfg, "rope_base", 10 ** 1.2)
        rope_position_scale = _number_option(
            cfg, "rope_position_scale", 2 * math.pi / 0.4
        )
        self.implementation = str(
            _cfg_get(cfg, "implementation", "flash_varlen")
        ).lower()
        # Queries per neighbor-search chunk AND per FlashAttention call. An
        # earlier module in this repo passed every token as its own length-1
        # varlen sequence in one FlashAttention-2 call and crashed at ~58k
        # sequences with ``CUDA error: invalid configuration argument``
        # (suspected: the 65535 cap on a CUDA grid dimension); batch size two,
        # about 29k sequences, worked. Chunking keeps every call far below
        # that, and the constructor refuses anything past the cap itself.
        self.query_chunk_size = _number_option(
            cfg, "query_chunk_size", 4096, integer=True
        )

        if self.dim <= 0 or self.num_heads <= 0:
            raise ValueError("local attention dimensions must be positive")
        if self.radius_m <= 0.0:
            raise ValueError("local attention radius_m must be positive")
        if self.mlp_ratio <= 0.0 or int(self.dim * self.mlp_ratio) < 1:
            raise ValueError(
                "local attention mlp_ratio must be positive and leave a "
                "non-empty hidden layer"
            )
        if rope_base <= 0.0:
            raise ValueError("local attention rope_base must be positive")
        if rope_position_scale <= 0.0:
            raise ValueError("local attention rope_position_scale must be positive")
        if self.implementation not in ("flash_varlen", "sdpa", "auto"):
            raise ValueError(
                "local attention implementation must be 'flash_varlen', "
                f"'sdpa' or 'auto', got {self.implementation!r}"
            )
        if not 0 < self.query_chunk_size <= MAX_QUERY_CHUNK_SIZE:
            raise ValueError(
                "local attention query_chunk_size must be in "
                f"[1, {MAX_QUERY_CHUNK_SIZE}], got {self.query_chunk_size}"
            )
        if self.dim % self.num_heads != 0:
            raise ValueError("local attention dim must be divisible by num_heads")
        self.head_dim = self.dim // self.num_heads
        if self.head_dim % 6 != 0:
            raise ValueError(
                "local attention head dim must be divisible by 6 for 3D RoPE"
            )
        self.scale = self.head_dim ** -0.5
        hidden = int(self.dim * self.mlp_ratio)

        # The registration order below is the checkpoint contract.
        self.input_norm = nn.LayerNorm(self.dim)
        self.q_proj = nn.Linear(self.dim, self.dim)
        self.k_proj = nn.Linear(self.dim, self.dim)
        self.v_proj = nn.Linear(self.dim, self.dim)
        self.out_proj = nn.Linear(self.dim, self.dim)
        self.ffn_norm = nn.LayerNorm(self.dim)
        self.ffn = nn.Sequential(
            nn.Linear(self.dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, self.dim),
        )
        # Start the position-wise refinement as an identity, as in the
        # temporal layers.
        nn.init.zeros_(self.ffn[-1].weight)
        nn.init.zeros_(self.ffn[-1].bias)
        self.rope = Rotary3D(
            self.head_dim, base=rope_base, position_scale=rope_position_scale
        )

    def extra_repr(self):
        return (
            f"radius_m={self.radius_m}, num_heads={self.num_heads}, "
            f"head_dim={self.head_dim}, implementation={self.implementation}, "
            f"query_chunk_size={self.query_chunk_size}"
        )

    def _use_flash(self, x):
        """Whether the varlen kernel is available for this batch."""
        if (
            x.is_cuda
            and self.implementation == "flash_varlen"
            and flash_attn is None
        ):
            raise RuntimeError(
                "V15 local attention requested flash_varlen, but "
                "flash_attn is unavailable"
            )
        return (
            x.is_cuda
            and flash_attn is not None
            and self.implementation in ("flash_varlen", "auto")
        )

    @torch.no_grad()
    def radius_neighbors(self, position, sample_index):
        """CSR neighbor lists ``(offsets, index)`` for every token.

        ``offsets`` is ``(N+1,)`` int64 over the queries in global row order
        and ``index`` is ``(E,)`` int64 global key rows, so query ``i`` reads
        ``index[offsets[i]:offsets[i+1]]``, ascending. The pair ``(i, j)`` is
        present iff both rows belong to the same sample and
        ``||p_i - p_j|| <= radius_m``; the bound is inclusive, so a token is
        always its own neighbor.

        Distances are the norms of the coordinate differences, in float32.
        ``torch.cdist`` was not used, for two measured reasons. Its default
        Gram-matrix mode (``|a|^2 + |b|^2 - 2 a.b``) cancels: on a scene
        centered on its mean and 100 m across it was off by up to 0.03 m, even
        for a point against itself, which flips pairs at the radius. Its exact
        mode is right but took 0.6 s per 14k-token window against about 10 ms
        here on an A100. A difference of nearby coordinates carries a rounding
        error relative to itself rather than to the coordinates' magnitude, so
        no centering is needed either. Queries run ``query_chunk_size`` at a
        time to bound the ``(3, chunk, N)`` block.
        """
        n_tokens = int(position.shape[0])
        if position.shape != (n_tokens, 3) or not position.is_floating_point():
            raise ValueError("local attention position must be a float (N,3) tensor")
        if sample_index.shape != (n_tokens,) or sample_index.dtype not in (
            torch.int32, torch.int64
        ):
            raise ValueError(
                "local attention sample_index must be an (N,) integer tensor"
            )
        if n_tokens == 0:
            return (
                torch.zeros(1, dtype=torch.long, device=position.device),
                torch.zeros(0, dtype=torch.long, device=position.device),
            )
        if not bool(torch.isfinite(position).all()):
            raise ValueError("local attention positions must be finite")

        coords = position.float()
        _, sample = torch.unique(sample_index, return_inverse=True)
        sizes = torch.bincount(sample).tolist()
        # Stable, so the rows of each sample stay in ascending global order.
        grouped_rows = torch.argsort(sample, stable=True).split(sizes)

        query_rows = []
        key_rows = []
        for rows in grouped_rows:
            xyz = coords[rows].t().contiguous()
            for start in range(0, xyz.shape[1], self.query_chunk_size):
                stop = min(start + self.query_chunk_size, xyz.shape[1])
                distance = torch.linalg.vector_norm(
                    xyz[:, start:stop, None] - xyz[:, None, :], dim=0
                )
                query, key = (distance <= self.radius_m).nonzero(as_tuple=True)
                query_rows.append(rows[query + start])
                key_rows.append(rows[key])

        query_rows = torch.cat(query_rows)
        key_rows = torch.cat(key_rows)
        # ``nonzero`` is row-major, so each sample's pairs are already ordered
        # by query and then by key; the stable sort only interleaves samples.
        by_query = torch.argsort(query_rows, stable=True)
        counts = torch.bincount(query_rows, minlength=n_tokens)
        offsets = F.pad(torch.cumsum(counts, dim=0), (1, 0))
        return offsets, key_rows[by_query]

    def _query_chunks(self, offsets):
        """``(q0, q1, e0, e1, max_keys)`` per query chunk, resolved on the host.

        The neighbor counts are data dependent, so the chunk table is the one
        place the forward reads them back: one transfer of ``offsets``, after
        which no chunk costs another synchronization.
        """
        host = offsets.cpu()
        counts = host[1:] - host[:-1]
        n_tokens = int(counts.shape[0])
        chunks = []
        for q0 in range(0, n_tokens, self.query_chunk_size):
            q1 = min(q0 + self.query_chunk_size, n_tokens)
            chunks.append((
                q0, q1, int(host[q0]), int(host[q1]), int(counts[q0:q1].max()),
            ))
        return chunks

    def _flash_attention(self, q, k, v, offsets, index, chunks):
        """Packed neighbors through varlen FlashAttention, chunk by chunk.

        Query ``i`` of a chunk is sequence ``i`` of length one; its key
        sequence is the slice of ``index`` between two CSR offsets. K and V
        were projected and rotated once for the N tokens, and are only now
        gathered. The gather stays float32 (its backward is an ``index_add``,
        which should not accumulate a dozen neighbors in bf16) and is cast
        just before the kernel.
        """
        major, _minor = torch.cuda.get_device_capability(q.device)
        flash_dtype = torch.bfloat16 if major >= 8 else torch.float16
        n_tokens = int(q.shape[0])
        k_rows = k.reshape(n_tokens, self.dim)
        v_rows = v.reshape(n_tokens, self.dim)
        outputs = []
        for q0, q1, e0, e1, max_keys in chunks:
            neighbors = index[e0:e1]
            cu_q = torch.arange(q1 - q0 + 1, dtype=torch.int32, device=q.device)
            cu_k = (offsets[q0:q1 + 1] - e0).to(torch.int32)
            attended = flash_attn.flash_attn_varlen_func(
                q=q[q0:q1].to(flash_dtype),
                k=k_rows.index_select(0, neighbors).reshape(
                    -1, self.num_heads, self.head_dim
                ).to(flash_dtype),
                v=v_rows.index_select(0, neighbors).reshape(
                    -1, self.num_heads, self.head_dim
                ).to(flash_dtype),
                cu_seqlens_q=cu_q,
                cu_seqlens_k=cu_k,
                max_seqlen_q=1,
                max_seqlen_k=max_keys,
                dropout_p=0.0,
                softmax_scale=self.scale,
                causal=False,
            )
            outputs.append(attended.to(q.dtype))
        return torch.cat(outputs, dim=0)

    def _sdpa_attention(self, q, k, v, offsets, index, chunks):
        """Reference/fallback path: padded neighbors plus a boolean mask.

        The neighbor sets are exactly Flash's. A chunk is padded to its own
        longest list; padding slots re-read the query's first neighbor and are
        masked out, so no row is ever fully masked.
        """
        n_tokens = int(q.shape[0])
        k_rows = k.reshape(n_tokens, self.dim)
        v_rows = v.reshape(n_tokens, self.dim)
        outputs = []
        for q0, q1, _e0, _e1, max_keys in chunks:
            count = offsets[q0 + 1:q1 + 1] - offsets[q0:q1]
            first = offsets[q0:q1, None]
            slot = torch.arange(max_keys, device=q.device)
            valid = slot[None, :] < count[:, None]
            neighbors = index[torch.where(valid, first + slot[None, :], first)]
            shape = (q1 - q0, max_keys, self.num_heads, self.head_dim)
            k_c = k_rows.index_select(0, neighbors.reshape(-1)).reshape(shape)
            v_c = v_rows.index_select(0, neighbors.reshape(-1)).reshape(shape)
            attended = F.scaled_dot_product_attention(
                q[q0:q1].unsqueeze(2),
                k_c.transpose(1, 2),
                v_c.transpose(1, 2),
                attn_mask=valid[:, None, None, :],
                dropout_p=0.0,
                scale=self.scale,
            )
            outputs.append(attended.squeeze(2))
        return torch.cat(outputs, dim=0)

    def forward(self, feature, position, sample_index):
        """``(N, dim)`` features -> ``(N, dim)``, in ``feature.dtype``.

        ``position`` is ``(N, 3)`` metres in the window's reference frame and
        is treated as input geometry (no gradient); ``sample_index`` is ``(N,)``
        integer window ids.
        """
        n_tokens = int(feature.shape[0])
        if feature.ndim != 2 or feature.shape[1] != self.dim:
            raise ValueError(f"local attention feature must be (N,{self.dim})")
        if position.shape != (n_tokens, 3):
            raise ValueError("position must align with local attention features")
        if sample_index.shape != (n_tokens,):
            raise ValueError("sample_index must give one id per local attention token")
        if position.device != feature.device or sample_index.device != feature.device:
            raise ValueError("position and sample_index must be on the feature's device")
        use_flash = self._use_flash(feature)

        offsets, index = self.radius_neighbors(position, sample_index)
        if n_tokens == 0:
            return feature

        conditioned = self.input_norm(feature)
        q = self.q_proj(conditioned).reshape(n_tokens, self.num_heads, self.head_dim)
        k = self.k_proj(conditioned).reshape(n_tokens, self.num_heads, self.head_dim)
        v = self.v_proj(conditioned).reshape(n_tokens, self.num_heads, self.head_dim)
        # Each token is rotated at its own position, before any gather: a key
        # keeps the same rotation no matter which query reads it. The rotation
        # and the softmax run in float32 (float64 for a float64 model).
        compute_dtype = torch.promote_types(feature.dtype, torch.float32)
        angles = self.rope.angles(position.detach())
        q = self.rope.rotate(q.to(compute_dtype), angles[:, None, :, :])
        k = self.rope.rotate(k.to(compute_dtype), angles[:, None, :, :])
        v = v.to(compute_dtype)

        chunks = self._query_chunks(offsets)
        if use_flash:
            attended = self._flash_attention(q, k, v, offsets, index, chunks)
        else:
            attended = self._sdpa_attention(q, k, v, offsets, index, chunks)

        x = feature + self.out_proj(
            attended.reshape(n_tokens, self.dim).to(feature.dtype)
        )
        x = x + self.ffn(self.ffn_norm(x))
        return x.to(feature.dtype)


__all__ = ["MAX_QUERY_CHUNK_SIZE", "RadiusLocalAttentionBlock"]
