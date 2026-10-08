"""Temporal cross-attention and correspondence matchers."""
from __future__ import annotations

import math
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from ...utils.attention import Rotary3D
from .common import SinusoidalScalarEncoder, _cfg_get

try:
    import flash_attn
except ImportError:  # CPU/unit-test fallback remains available.
    flash_attn = None

class BidirectionalCrossAttentionBase(nn.Module):
    """Layer stack and endpoint pairing shared by every temporal variant.

    Each layer forms Q/K/V for *both* directions from one immutable input
    snapshot, executes the ragged endpoint pairs, and only then commits the
    residual/FFN update. Subclasses own the constructor and ``forward`` and
    differ only in how the attention scores are formed and which head's
    probabilities are read as correspondence.
    """

    def _build_qk_norm(self, enabled: bool) -> None:
        """Per-layer RMSNorm on Q and K, or no normalization at all.

        Nothing else bounds ``|q|`` or ``|k|``: ``input_norm`` sits before the
        projection, so the projection gain passes through unchecked and the
        logit grows as its square. RMSNorm rather than LayerNorm because
        LayerNorm's mean subtraction would mix Rotary3D's x/y/z chunks.
        """
        self.qk_norm = bool(enabled)
        if not self.qk_norm:
            self.q_norm = None
            self.k_norm = None
            return
        self.q_norm = nn.ModuleList([
            nn.RMSNorm(self.head_dim) for _ in range(self.n_layers)
        ])
        self.k_norm = nn.ModuleList([
            nn.RMSNorm(self.head_dim) for _ in range(self.n_layers)
        ])

    def _build_transformer_layers(self, cfg):
        """The per-layer stack every temporal variant shares.

        Input norm, the four attention projections, the LayerScale gate, and a
        zero-initialized FFN, all built in the one order the saved checkpoints
        expect. Variants differ only in how the scores are formed, never in
        which parameters exist here. ``layer_scale: false`` (V15) is the one
        structural switch: it builds no gate at all, so the out_proj branch is
        added to the residual stream as is.
        """
        hidden = int(self.dim * self.mlp_ratio)
        self.input_norm = nn.ModuleList([
            nn.LayerNorm(self.dim) for _ in range(self.n_layers)
        ])
        self.q_proj = nn.ModuleList([
            nn.Linear(self.dim, self.dim) for _ in range(self.n_layers)
        ])
        self.k_proj = nn.ModuleList([
            nn.Linear(self.dim, self.dim) for _ in range(self.n_layers)
        ])
        self.v_proj = nn.ModuleList([
            nn.Linear(self.dim, self.dim) for _ in range(self.n_layers)
        ])
        self.out_proj = nn.ModuleList([
            nn.Linear(self.dim, self.dim) for _ in range(self.n_layers)
        ])
        layer_scale = _cfg_get(cfg, "layer_scale", True)
        if not isinstance(layer_scale, bool):
            raise ValueError("dynamic temporal layer_scale must be true or false")
        if layer_scale:
            layer_scale_init = float(_cfg_get(cfg, "layer_scale_init", 0.1))
            if not 0.0 < layer_scale_init <= 1.0:
                raise ValueError(
                    "dynamic temporal layer_scale_init must be in (0,1]"
                )
            self.cross_layer_scale = nn.ParameterList([
                nn.Parameter(torch.full((self.dim,), layer_scale_init))
                for _ in range(self.n_layers)
            ])
        else:
            if _cfg_get(cfg, "layer_scale_init", None) is not None:
                raise ValueError(
                    "dynamic temporal layer_scale=false has no LayerScale to "
                    "initialize; drop layer_scale_init"
                )
            self.cross_layer_scale = None
        self.ffn_norm = nn.ModuleList([
            nn.LayerNorm(self.dim) for _ in range(self.n_layers)
        ])
        self.ffn = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.dim, hidden),
                nn.SiLU(),
                nn.Linear(hidden, self.dim),
            )
            for _ in range(self.n_layers)
        ])
        # Start the position-wise refinement as an identity without suppressing
        # the cross-frame branch itself.
        for ffn in self.ffn:
            nn.init.zeros_(ffn[-1].weight)
            nn.init.zeros_(ffn[-1].bias)

    def _cross_branch(self, layer, branch):
        """The cross-attention branch as layer ``layer`` adds it to the residual.

        Gated by the layer's LayerScale ``gamma_l``, or as is when the stack
        was built without LayerScale.
        """
        if self.cross_layer_scale is None:
            return branch
        return self.cross_layer_scale[layer] * branch

    @staticmethod
    def _frame_layout(token_offset, frame_batch_idx):
        counts = torch.diff(
            token_offset,
            prepend=token_offset.new_zeros(1),
        ).long()
        batch_frames = defaultdict(list)
        for frame, batch_id in enumerate(frame_batch_idx.tolist()):
            batch_frames[int(batch_id)].append(frame)
        for batch_id, frames in batch_frames.items():
            if len(frames) != 2:
                raise ValueError(
                    "Dynamic 2DGS currently requires exactly two input frames "
                    f"per sample; batch {batch_id} has {len(frames)}"
                )
        opposite = torch.empty_like(frame_batch_idx)
        for frames in batch_frames.values():
            opposite[frames[0]] = frames[1]
            opposite[frames[1]] = frames[0]

        starts = torch.cumsum(counts, dim=0) - counts
        frame_rows = [
            torch.arange(
                int(starts[f]), int(starts[f] + counts[f]),
                device=token_offset.device,
            )
            for f in range(counts.numel())
        ]
        memory_rows = torch.cat([
            frame_rows[int(opposite[f])] for f in range(counts.numel())
        ])
        return counts, counts[opposite], memory_rows

    def _flash_attention(
        self,
        q,
        k,
        v,
        q_counts,
        k_counts,
        *,
        flash_dtype=None,
    ):
        cu_q = F.pad(torch.cumsum(q_counts, dim=0, dtype=torch.int32), (1, 0))
        cu_k = F.pad(torch.cumsum(k_counts, dim=0, dtype=torch.int32), (1, 0))
        if flash_dtype is None:
            major, _minor = torch.cuda.get_device_capability(q.device)
            flash_dtype = torch.bfloat16 if major >= 8 else torch.float16
        attended = flash_attn.flash_attn_varlen_func(
            q=q.to(flash_dtype),
            k=k.to(flash_dtype),
            v=v.to(flash_dtype),
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            max_seqlen_q=int(q_counts.max().item()),
            max_seqlen_k=int(k_counts.max().item()),
            dropout_p=0.0,
            softmax_scale=self.scale,
            causal=False,
        )
        return attended.to(q.dtype)

    def _sdpa_attention(self, q, k, v, q_counts, k_counts):
        """Reference/fallback path; sequence pairing is identical to Flash."""
        q_starts = torch.cumsum(q_counts, dim=0) - q_counts
        k_starts = torch.cumsum(k_counts, dim=0) - k_counts
        chunks = []
        for qs, qn, ks, kn in zip(q_starts, q_counts, k_starts, k_counts):
            qs, qn, ks, kn = map(int, (qs, qn, ks, kn))
            q_i = q[qs:qs + qn].transpose(0, 1).unsqueeze(0)
            k_i = k[ks:ks + kn].transpose(0, 1).unsqueeze(0)
            v_i = v[ks:ks + kn].transpose(0, 1).unsqueeze(0)
            out_i = F.scaled_dot_product_attention(
                q_i, k_i, v_i, dropout_p=0.0, scale=self.scale
            )
            chunks.append(out_i.squeeze(0).transpose(0, 1))
        return torch.cat(chunks, dim=0)


class MaxSpeedBarrierLayerWeightedCrossAttention(
    BidirectionalCrossAttentionBase
):
    """V11 split position encoding with a parameter-only layer mixture.

    Head 0 of every layer is the correspondence head, and is the only head whose
    probabilities ever touch coordinates.  It carries no 3D RoPE.  Its position
    encoding is instead one additive physical barrier on the logit,

    ``bias_ij = -a * relu(||x_i - x_j|| / (v_max * |dt|) - 1)^p``,

    which is exactly zero inside the radius a ``v_max`` object could cover over
    the endpoint interval and grows outside it -- quadratically at ``p=2``
    (V11's shipped form), linearly at ``p=1``.  Unlike a ``-(d/R)^2`` prior
    the hinge is not a bilinear form, so it cannot be folded into extra Q/K
    channels; this head therefore runs as an explicit softmax,
    chunked over queries and recomputed in backward.  Its V carries the ordinary
    feature channels plus opposite-frame xyz, so one pass returns both the
    feature update and ``W_l @ xyz``.

    Heads ``1..H-1`` keep ordinary metric 3D RoPE and stay on varlen
    FlashAttention.

    ``v_init`` mixes the ``L`` head-0 coordinate expectations under
    ``softmax(a_0..a_{L-1})`` over ``L`` plain learned scalars -- no global
    token and no per-sample scorer -- so the mixture asks only which *depth*
    resolves correspondence, starting from an exact uniform ``1/L``.
    """

    _MATCH_HEAD = 0

    @staticmethod
    def _frame_first(value, counts):
        """The first token's value in every frame, one entry per frame."""
        starts = torch.cumsum(counts, dim=0) - counts
        return value[starts.long()]

    def __init__(self, cfg, dim: int):
        super().__init__()
        self.dim = int(dim)
        self.num_heads = int(_cfg_get(cfg, "num_heads", 8))
        self.n_layers = int(_cfg_get(cfg, "layers", 1))
        self.mlp_ratio = float(_cfg_get(cfg, "mlp_ratio", 4.0))
        self.implementation = str(
            _cfg_get(cfg, "implementation", "flash_varlen")
        ).lower()
        self.use_time_embedding = bool(
            _cfg_get(cfg, "use_time_embedding", True)
        )
        self.position_encoding = str(
            _cfg_get(cfg, "position_encoding", "barrier_rope_split")
        ).lower()
        self.barrier_speed_mps = float(_cfg_get(cfg, "barrier_speed_mps", 30.0))
        self.barrier_weight = float(_cfg_get(cfg, "barrier_weight", 4.0))
        # p=2 is V11's quadratic hinge; p=1 makes the penalty grow linearly
        # once past the radius, so its slope is the same everywhere outside.
        self.barrier_exponent = int(_cfg_get(cfg, "barrier_exponent", 2))
        self.match_chunk_size = int(_cfg_get(cfg, "match_chunk_size", 1024))

        if self.dim <= 0 or self.num_heads <= 0:
            raise ValueError("V11 attention dimensions must be positive")
        if self.dim % self.num_heads != 0:
            raise ValueError("dynamic token dim must be divisible by num_heads")
        if self.num_heads < 2:
            raise ValueError(
                "V11 needs one barrier match head plus at least one RoPE head"
            )
        if self.n_layers < 1:
            raise ValueError("dynamic temporal layers must be at least one")
        if self.mlp_ratio <= 0.0:
            raise ValueError("dynamic temporal mlp_ratio must be positive")
        if not self.use_time_embedding:
            raise ValueError("V11 temporal attention requires time embedding")
        if self.position_encoding != "barrier_rope_split":
            raise ValueError(
                "V11 temporal position_encoding must be 'barrier_rope_split'"
            )
        if self.barrier_speed_mps <= 0.0:
            raise ValueError("V11 barrier_speed_mps must be positive")
        if self.barrier_weight < 0.0:
            raise ValueError("V11 barrier_weight must be non-negative")
        if self.barrier_exponent not in (1, 2):
            raise ValueError("barrier_exponent must be 1 or 2")
        if self.match_chunk_size <= 0:
            raise ValueError("V11 match_chunk_size must be positive")

        self.head_dim = self.dim // self.num_heads
        if self.head_dim % 6 != 0:
            raise ValueError(
                "dynamic attention head dim must be divisible by 6 for 3D RoPE"
            )
        self.scale = self.head_dim ** -0.5
        self.rope_head_count = self.num_heads - 1

        rope_position_scale = _cfg_get(cfg, "rope_position_scale", None)
        if rope_position_scale is None:
            range_m = float(_cfg_get(cfg, "rope_range_m", 110.0))
            rope_position_scale = math.pi / range_m
        # Heads 1..H-1 only; head 0 reads geometry from the barrier instead.
        self.rope = Rotary3D(
            self.head_dim,
            base=float(_cfg_get(cfg, "rope_base", 100.0)),
            position_scale=float(rope_position_scale),
        )

        # Endpoint time: Fourier -> MLP(..., time_dim) -> bias-free
        # Linear(time_dim, dim), added once to the fused token before layer one.
        # Every temporal variant shares this parameterization.
        time_dim = int(_cfg_get(cfg, "time_embedding_dim", 64))
        time_frequencies = int(_cfg_get(cfg, "time_frequencies", 8))
        self.time_encoder = SinusoidalScalarEncoder(time_dim, time_frequencies)
        self.time_to_feature = nn.Linear(time_dim, self.dim, bias=False)

        if not bool(_cfg_get(cfg, "qk_norm", True)):
            raise ValueError("V11 requires qk_norm=true")
        self._build_qk_norm(True)
        self._build_transformer_layers(cfg)

        # One scalar per layer, shared by every token and every sample. Zero
        # init is an exact uniform 1/L mixture, and each logit's only gradient
        # is the render loss' preference among the layers' coordinate readouts.
        self.layer_logits = nn.Parameter(torch.zeros(self.n_layers))

    @torch.no_grad()
    def attention_temperature_stats(self):
        """QK-Norm gain diagnostics next to the fixed barrier geometry.

        The content logit is bounded by ``gamma_q * gamma_k * head_dim * scale``
        while the barrier is a fixed constant, so watching the bound is also how
        one sees the barrier being outgrown.
        """
        stats = {
            "qk_motion_head_count": 1.0,
            "barrier_speed_mps": self.barrier_speed_mps,
            "barrier_weight": self.barrier_weight,
            "barrier_exponent": float(self.barrier_exponent),
        }
        bound = 0.0
        for layer in range(self.n_layers):
            gq = self.q_norm[layer].weight.float()
            gk = self.k_norm[layer].weight.float()
            rms_q = float(gq.square().mean().sqrt())
            rms_k = float(gk.square().mean().sqrt())
            bound = max(bound, rms_q * rms_k * self.head_dim * self.scale)
            stats[f"qk_gamma_q_layer{layer}"] = rms_q
            stats[f"qk_gamma_k_layer{layer}"] = rms_k
        stats["qk_max_logit_bound"] = bound
        return stats

    def _barrier_bias(self, query_position, key_position, radius):
        """Reference form ``a * relu(d/R - 1)^p`` for one chunk of queries.

        This is what the barrier *means*; :meth:`_barrier_chunk` evaluates the
        algebraically identical ``(a/R^p) * relu(d - R)^p`` because that folds
        into the score GEMM. Positions are input geometry rather than
        parameters, so the barrier is a constant of the graph either way.
        """
        with torch.no_grad():
            excess = torch.cdist(query_position, key_position)
            excess = excess.div_(float(radius)).sub_(1.0).clamp_min_(0.0)
            if self.barrier_exponent == 2:
                excess = excess.square_()
            return excess.mul_(self.barrier_weight)

    def _barrier_chunk(
        self, query, key, value, query_position, key_position, radius
    ):
        """One head-0 softmax over every opposite-frame key for this chunk.

        ``radius`` is one Python scalar: a chunk never spans an endpoint pair,
        and the barrier radius is that sample's physical interval. At these
        sizes the ``N_q x N_k`` block is bandwidth-bound rather than
        FLOP-bound, so the hinge is written as ``relu(d - R)^p`` and folded
        into the score GEMM's ``beta * C`` accumulate. That leaves three
        elementwise passes over the block instead of nine. Scaling by ``R``
        moves outside the hinge either way, since ``relu`` is positively
        homogeneous: ``a * relu(d/R - 1)^p == (a/R^p) * relu(d - R)^p``.
        """
        radius = float(radius)
        with torch.no_grad():
            excess = torch.cdist(
                query_position, key_position
            ).sub_(radius).relu_()
            if self.barrier_exponent == 2:
                excess = excess.square_()
        score = torch.addmm(
            excess,
            query,
            key.transpose(0, 1),
            beta=-self.barrier_weight / (radius ** self.barrier_exponent),
            alpha=self.scale,
        )
        probability = torch.softmax(score, dim=-1)
        attended = probability @ value
        return attended[..., :self.head_dim], attended[..., self.head_dim:]

    def _barrier_attention(
        self,
        query,
        key,
        value,
        query_position,
        key_position,
        q_count_values,
        k_count_values,
        frame_radius_values,
    ):
        """Head-0 feature update and ``W_l @ xyz``, chunked over queries.

        Retaining every chunk's ``N_q x N_k`` probabilities across twelve layers
        would dominate activation memory, so each chunk is recomputed in
        backward. The ragged layout and the
        per-frame radius arrive as Python scalars the caller resolved once, so
        no layer costs a host synchronization.
        """
        recompute = (
            self.training
            and torch.is_grad_enabled()
            and (
                query.requires_grad or key.requires_grad or value.requires_grad
            )
        )
        feature_chunks = []
        position_chunks = []
        query_start = 0
        key_start = 0
        for q_count, k_count, radius in zip(
            q_count_values, k_count_values, frame_radius_values
        ):
            if q_count <= 0 or k_count <= 0:
                raise ValueError("V11 requires tokens in both endpoint frames")
            keys = slice(key_start, key_start + k_count)
            key_xyz = key_position[keys]
            # Centering only improves the accuracy of the pairwise distance in
            # global reference coordinates. It changes no distance and no
            # displacement, because the readout adds the same origin back.
            origin = 0.5 * (
                query_position[query_start:query_start + q_count].mean(dim=0)
                + key_xyz.mean(dim=0)
            )
            key_centered = key_xyz - origin
            payload = torch.cat([value[keys], key_centered], dim=-1)
            for start in range(0, q_count, self.match_chunk_size):
                end = min(start + self.match_chunk_size, q_count)
                rows = slice(query_start + start, query_start + end)
                inputs = (
                    query[rows],
                    key[keys],
                    payload,
                    query_position[rows] - origin,
                    key_centered,
                    radius,
                )
                if recompute:
                    chunk = checkpoint(
                        self._barrier_chunk, *inputs, use_reentrant=False
                    )
                else:
                    chunk = self._barrier_chunk(*inputs)
                feature_chunks.append(chunk[0])
                position_chunks.append(chunk[1] + origin)
            query_start += q_count
            key_start += k_count
        return (
            torch.cat(feature_chunks, dim=0),
            torch.cat(position_chunks, dim=0),
        )

    def _combine_layer_matches(self, layer_matched_positions, position_ref):
        """Mix the per-layer coordinate expectations under the layer softmax."""
        if layer_matched_positions.ndim != 3:
            raise ValueError("V11 layer matches must have shape (N,L,3)")
        if layer_matched_positions.shape[1] != self.n_layers:
            raise ValueError("V11 layer matches must cover every layer")
        if layer_matched_positions.shape[-1] != 3:
            raise ValueError("V11 layer matches must end in xyz")
        if position_ref.shape != (layer_matched_positions.shape[0], 3):
            raise ValueError("V11 positions must align with layer matches")
        weights = torch.softmax(self.layer_logits.float(), dim=-1)
        matched_position = (
            weights.reshape(1, -1, 1) * layer_matched_positions.float()
        ).sum(dim=1)
        displacement = matched_position - position_ref.float()
        token_count = int(layer_matched_positions.shape[0])
        return (
            matched_position,
            displacement,
            self.layer_logits.detach().expand(token_count, -1),
            weights.detach().expand(token_count, -1),
        )

    def _prepare_forward(
        self,
        feature,
        position_ref,
        token_offset,
        frame_batch_idx,
        token_duration_sec,
    ):
        """Validate the batch and resolve the ragged layout and barrier radii.

        Every barrier variant needs exactly this, and needs it before any
        attention runs. Resolving the counts and the radius to Python scalars
        once matters: re-reading CUDA counts inside the layer loop would cost
        several synchronizing host round-trips per temporal layer.
        """
        n_tokens = int(feature.shape[0])
        if feature.shape != (n_tokens, self.dim):
            raise ValueError(f"V11 feature must be (N,{self.dim})")
        if position_ref.shape != (n_tokens, 3):
            raise ValueError("position_ref must align with V11 token features")
        if token_duration_sec.shape != (n_tokens,):
            raise ValueError("V11 duration must provide one scalar per token")
        if not bool(torch.all(
            torch.isfinite(token_duration_sec) & (token_duration_sec > 0.0)
        )):
            raise ValueError("V11 requires positive finite durations")

        q_counts, k_counts, memory_rows = self._frame_layout(
            token_offset, frame_batch_idx
        )
        if int(q_counts.sum()) != n_tokens:
            raise ValueError("token_offset does not cover all V11 tokens")
        if bool(torch.any(q_counts <= 0)) or bool(torch.any(k_counts <= 0)):
            raise ValueError("V11 requires tokens in both endpoint frames")

        frame_duration = self._frame_first(token_duration_sec, q_counts)
        if not torch.equal(
            torch.repeat_interleave(frame_duration, q_counts),
            token_duration_sec,
        ):
            raise ValueError(
                "V11 requires one endpoint interval per frame; the barrier "
                "radius is a per-sample physical duration"
            )
        return {
            "n_tokens": n_tokens,
            "q_counts": q_counts,
            "k_counts": k_counts,
            "memory_rows": memory_rows,
            "q_count_values": [int(value) for value in q_counts.tolist()],
            "k_count_values": [int(value) for value in k_counts.tolist()],
            # What an object at the admissible top speed could cover over this
            # sample's endpoint interval.
            "frame_radius_values": [
                max(self.barrier_speed_mps * float(value), 1.0e-4)
                for value in frame_duration.tolist()
            ],
        }

    def _layer_frame_radius_values(self, layout, layer):
        """The per-frame barrier radii that layer ``layer``'s head 0 uses.

        Every layer shares one radius per frame here. V12.2 overrides this to
        give each layer its own learned speed.
        """
        del layer
        return layout["frame_radius_values"]

    def _use_flash(self, x):
        """Whether the varlen kernel is available for this batch."""
        if (
            x.is_cuda
            and self.implementation == "flash_varlen"
            and flash_attn is None
        ):
            raise RuntimeError(
                "V11 temporal attention requested flash_varlen, but "
                "flash_attn is unavailable"
            )
        return (
            x.is_cuda
            and flash_attn is not None
            and self.implementation in ("flash_varlen", "auto")
        )

    def forward(
        self,
        feature,
        position_ref,
        token_offset,
        frame_batch_idx,
        token_time_coordinate,
        token_duration_sec,
    ):
        if token_time_coordinate.shape != (int(feature.shape[0]),):
            raise ValueError("V11 token time must provide one scalar per token")
        layout = self._prepare_forward(
            feature, position_ref, token_offset, frame_batch_idx,
            token_duration_sec,
        )
        n_tokens = layout["n_tokens"]
        q_counts = layout["q_counts"]
        k_counts = layout["k_counts"]
        memory_rows = layout["memory_rows"]
        q_count_values = layout["q_count_values"]
        k_count_values = layout["k_count_values"]

        x = feature + self.time_to_feature(
            self.time_encoder(token_time_coordinate)
        ).to(feature.dtype)

        query_position = position_ref.float()
        key_position = position_ref[memory_rows].float()
        q_angles = self.rope.angles(query_position)
        k_angles = self.rope.angles(key_position)

        use_flash = self._use_flash(x)

        layer_matched_positions = []
        for layer in range(self.n_layers):
            snapshot = x
            conditioned = self.input_norm[layer](snapshot)
            q = self.q_proj[layer](conditioned).reshape(
                n_tokens, self.num_heads, self.head_dim
            )
            k = self.k_proj[layer](conditioned)[memory_rows].reshape(
                n_tokens, self.num_heads, self.head_dim
            )
            v = self.v_proj[layer](conditioned)[memory_rows].reshape(
                n_tokens, self.num_heads, self.head_dim
            )
            q = self.q_norm[layer](q).float()
            k = self.k_norm[layer](k).float()
            v = v.float()

            match_attended, matched_position = self._barrier_attention(
                q[:, self._MATCH_HEAD],
                k[:, self._MATCH_HEAD],
                v[:, self._MATCH_HEAD],
                query_position,
                key_position,
                q_count_values,
                k_count_values,
                self._layer_frame_radius_values(layout, layer),
            )
            layer_matched_positions.append(matched_position)

            rope_head = slice(self._MATCH_HEAD + 1, None)
            rope_attended_inputs = (
                self.rope.rotate(q[:, rope_head], q_angles[:, None, :, :]),
                self.rope.rotate(k[:, rope_head], k_angles[:, None, :, :]),
                v[:, rope_head],
                q_counts,
                k_counts,
            )
            if use_flash:
                rope_attended = self._flash_attention(*rope_attended_inputs)
            else:
                rope_attended = self._sdpa_attention(*rope_attended_inputs)

            attended = torch.cat(
                [match_attended.unsqueeze(1), rope_attended], dim=1
            ).reshape(n_tokens, self.dim).to(snapshot.dtype)
            x = snapshot + self._cross_branch(
                layer, self.out_proj[layer](attended)
            )
            x = x + self.ffn[layer](self.ffn_norm[layer](x))

        (
            matched_position,
            displacement,
            token_logits,
            token_weights,
        ) = self._combine_layer_matches(
            torch.stack(layer_matched_positions, dim=1), position_ref
        )
        return x, {
            "delta_p_match": displacement,
            "delta_p_init": displacement,
            "matched_position": matched_position,
            "motion_layer_logits": token_logits,
            "motion_layer_weights": token_weights,
        }


class _LearnedRadiusBarrierCrossAttention(
    MaxSpeedBarrierLayerWeightedCrossAttention
):
    """V11.4's barrier arithmetic with the radius left in the autograd graph.

    V11.4 resolves its fixed radius to a Python float and computes the whole
    barrier under ``no_grad``.  The learned-speed variants (V12, V12.1, V12.2)
    instead keep each radius as a 0-D tensor and detach only the pairwise
    input-geometry distance.  Reconstruction gradients can therefore reach the
    learned speed without opening an unintended path into the source
    coordinates.  How many speeds exist and how they are parameterized is up to
    the subclass, which also supplies the radius tensors.
    """

    @staticmethod
    def _as_radius_tensor(radius, reference):
        if torch.is_tensor(radius):
            return radius.to(device=reference.device, dtype=reference.dtype)
        return reference.new_tensor(float(radius))

    def _barrier_bias(self, query_position, key_position, radius):
        """Differentiable-in-radius form of the V11.4 barrier penalty."""
        radius = self._as_radius_tensor(radius, query_position)
        with torch.no_grad():
            distance = torch.cdist(query_position, key_position)
        excess = torch.relu(distance - radius)
        if self.barrier_exponent == 2:
            excess = excess.square()
        return excess * (
            self.barrier_weight / radius.pow(self.barrier_exponent)
        )

    def _barrier_chunk(
        self, query, key, value, query_position, key_position, radius
    ):
        """Head-0 softmax whose radius retains its gradient to the speed."""
        penalty = self._barrier_bias(
            query_position, key_position, radius
        )
        score = torch.addmm(
            penalty,
            query,
            key.transpose(0, 1),
            beta=-1.0,
            alpha=self.scale,
        )
        probability = torch.softmax(score, dim=-1)
        attended = probability @ value
        return attended[..., :self.head_dim], attended[..., self.head_dim:]


def _softplus_inverse(value: float) -> float:
    """``softplus^-1(y) = y + log(1 - exp(-y))``, accurate for small ``y``."""
    return value + math.log(-math.expm1(-value))


class _GlobalLearnedSpeedBarrierCrossAttention(
    _LearnedRadiusBarrierCrossAttention
):
    """One learned max speed shared by every layer, direction and sample.

    A subclass registers the speed parameter and exposes the positive speed as
    ``v_max``; this class turns it into each frame's radius
    ``v_max * |dt|`` and keeps that radius in the autograd graph.
    """

    def _prepare_forward(
        self,
        feature,
        position_ref,
        token_offset,
        frame_batch_idx,
        token_duration_sec,
    ):
        layout = super()._prepare_forward(
            feature,
            position_ref,
            token_offset,
            frame_batch_idx,
            token_duration_sec,
        )
        frame_duration = self._frame_first(
            token_duration_sec, layout["q_counts"]
        ).float()
        frame_radius = (self.v_max * frame_duration).clamp_min(1.0e-4)
        # Keep 0-D tensors all the way into the checkpointed chunks. Converting
        # these to Python floats would silently detach the learned speed.
        layout["frame_radius_values"] = tuple(frame_radius.unbind())
        return layout


class LearnableMaxSpeedBarrierLayerWeightedCrossAttention(
    _GlobalLearnedSpeedBarrierCrossAttention
):
    """V12: V11.4's linear barrier with one learned global ``v_max``.

    The only new parameter is ``v_0 = log(v_max)``.  It starts at the
    configured 30 m/s value and the physical speed used by every layer, head-0
    readout, direction, and sample is ``exp(v_0)``.  Keeping the parameter in
    log space makes the speed strictly positive; it deliberately does not add
    an upper bound or a barrier-specific regularizer.

    QK-Norm is inherited unchanged from V11.4.  Each layer still owns one Q
    gain vector and one K gain vector, and each vector is broadcast across all
    heads.  In particular, this class does not use V11.1's match/RoPE gain
    split.
    """

    def __init__(self, cfg, dim: int):
        super().__init__(cfg, dim)
        self.initial_barrier_speed_mps = self.barrier_speed_mps
        self.v_0 = nn.Parameter(torch.tensor(
            math.log(self.initial_barrier_speed_mps), dtype=torch.float32
        ))

    @property
    def v_max(self):
        """Current positive max speed in metres per second."""
        return self.v_0.exp()

    @torch.no_grad()
    def attention_temperature_stats(self):
        """V11.4 temperature diagnostics plus the learned barrier state."""
        stats = super().attention_temperature_stats()
        stats["barrier_speed_initial_mps"] = self.initial_barrier_speed_mps
        stats["barrier_v_0"] = float(self.v_0)
        # Preserve the established metric name, but report the current learned
        # speed rather than the initialization stored in the config.
        stats["barrier_speed_mps"] = float(self.v_max)
        return stats


class GlobalSoftplusMaxSpeedBarrierCrossAttention(
    _GlobalLearnedSpeedBarrierCrossAttention
):
    """V14.1: V12's one global learned speed through V12.2's softplus.

    ``v_max = softplus(v_raw)`` with a single scalar ``v_raw`` initialized to
    ``softplus^-1(barrier_speed_mps)``, so every layer, both endpoint
    directions and every sample read correspondence under one radius
    ``v_max * |dt|``, and a fresh module reproduces V11.4's fixed 30 m/s
    barrier exactly. As in V12.2 an optimizer step moves the speed by about
    the same number of m/s at any speed (softplus slope ~1 near 30 m/s); as in
    V12 there is one speed rather than one per layer. Nothing clamps the speed
    or pulls it back toward its initial value.
    """

    def __init__(self, cfg, dim: int):
        super().__init__(cfg, dim)
        self.initial_barrier_speed_mps = self.barrier_speed_mps
        self.v_raw = nn.Parameter(torch.tensor(
            _softplus_inverse(self.initial_barrier_speed_mps),
            dtype=torch.float32,
        ))

    @property
    def v_max(self):
        """Current positive max speed in metres per second."""
        return F.softplus(self.v_raw)

    @torch.no_grad()
    def attention_temperature_stats(self):
        """V11.4 temperature diagnostics plus the learned global speed."""
        stats = super().attention_temperature_stats()
        stats["barrier_speed_initial_mps"] = self.initial_barrier_speed_mps
        stats["barrier_v_raw"] = float(self.v_raw)
        stats["barrier_speed_mps"] = float(self.v_max)
        return stats


class FinalFeatureBarrierCrossAttention(
    MaxSpeedBarrierLayerWeightedCrossAttention
):
    """V11.2: uniform-RoPE flash refinement with one barrier readout at the end.

    V11 splits its heads because head 0 has to double as the correspondence
    head: it drops 3D RoPE, carries the max-speed barrier instead, and reads
    opposite-frame coordinates at every layer.  That barrier is a hinge on
    Euclidean distance rather than a bilinear form, so it cannot be folded into
    Q/K channels and head 0 has to run an explicit chunked softmax while heads
    1..H-1 stay on varlen FlashAttention.

    V11.2 forms ``v_init`` *after* the stack, so nothing inside it needs to
    read coordinates.  Head 0 therefore carries ordinary 3D RoPE like every
    other head, the barrier leaves the layer loop entirely, and all ``H`` heads
    go through one FlashAttention call per layer.  On an A100 at 7.5k tokens
    per frame this is about 40% off the module's forward and backward.

    Correspondence is then read once, from the complete final feature ``f'``,
    by one dedicated head:

    ``match_input_norm(f') -> match_q_proj/match_k_proj -> qk RMSNorm``.

    That head is a structural copy of a single V11 readout head: the same
    ``LayerNorm(dim) -> Linear(dim, head_dim) -> RMSNorm(head_dim)`` chain at
    the same width.  PyTorch initializes ``Linear`` from ``fan_in``, which is
    ``dim`` for both, so it starts from the identical distribution as head 0's
    rows of V11's ``Linear(dim, dim)``.  Its dense softmax keeps V11's exact
    barrier, chunking and backward recomputation; this is the one place in the
    variant where geometry enters a probability.

    Against V11 this moves two things at once, which the experiment intends:
    where correspondence is read, and whether the layer stack still needs a
    geometric prior of its own once it no longer produces correspondence.
    """

    def __init__(self, cfg, dim: int):
        super().__init__(cfg, dim)
        # Every head is a RoPE head here. V11's layer mixture had one scalar
        # per layer to weight per-layer coordinate readouts; there are no
        # per-layer readouts left to weight, so the parameter goes rather than
        # sitting in the checkpoint collecting no gradient.
        del self.layer_logits
        self.rope_head_count = self.num_heads
        # The one readout head, appended after the layer stack.
        self.match_input_norm = nn.LayerNorm(self.dim)
        self.match_q_proj = nn.Linear(self.dim, self.head_dim)
        self.match_k_proj = nn.Linear(self.dim, self.head_dim)
        self.match_q_norm = nn.RMSNorm(self.head_dim)
        self.match_k_norm = nn.RMSNorm(self.head_dim)

    @torch.no_grad()
    def attention_temperature_stats(self):
        """Per-layer QK gains, plus the readout head that faces the barrier.

        The inherited per-layer keys stay comparable to a V11 run even though
        head 0 now carries RoPE, because they describe the same projections.
        ``qk_motion_head_count`` is corrected to zero: no layer head reads
        coordinates any more. The readout's own bound is what races the fixed
        barrier, so it is reported separately.
        """
        stats = super().attention_temperature_stats()
        stats["qk_motion_head_count"] = 0.0
        gamma_q = float(self.match_q_norm.weight.float().square().mean().sqrt())
        gamma_k = float(self.match_k_norm.weight.float().square().mean().sqrt())
        stats["qk_match_gamma_q"] = gamma_q
        stats["qk_match_gamma_k"] = gamma_k
        stats["qk_match_logit_bound"] = (
            gamma_q * gamma_k * self.head_dim * self.scale
        )
        return stats

    def _final_match_descriptors(self, refined, memory_rows):
        """Project ``f'`` through this variant's own head-width readout."""
        conditioned = self.match_input_norm(refined)
        query = self.match_q_norm(self.match_q_proj(conditioned))
        # Gather before the norm exactly as the layer stack does; RMSNorm is
        # per row, so the order is a readability choice, not a numerical one.
        key = self.match_k_norm(self.match_k_proj(conditioned)[memory_rows])
        return query.float(), key.float()

    def forward(
        self,
        feature,
        position_ref,
        token_offset,
        frame_batch_idx,
        token_time_coordinate,
        token_duration_sec,
    ):
        n_tokens = int(feature.shape[0])
        if token_time_coordinate.shape != (n_tokens,):
            raise ValueError("V11 token time must provide one scalar per token")
        layout = self._prepare_forward(
            feature, position_ref, token_offset, frame_batch_idx,
            token_duration_sec,
        )
        q_counts = layout["q_counts"]
        k_counts = layout["k_counts"]
        memory_rows = layout["memory_rows"]

        x = feature + self.time_to_feature(
            self.time_encoder(token_time_coordinate)
        ).to(feature.dtype)

        query_position = position_ref.float()
        key_position = position_ref[memory_rows].float()
        q_angles = self.rope.angles(query_position)
        k_angles = self.rope.angles(key_position)
        use_flash = self._use_flash(x)

        for layer in range(self.n_layers):
            snapshot = x
            conditioned = self.input_norm[layer](snapshot)
            q = self.q_proj[layer](conditioned).reshape(
                n_tokens, self.num_heads, self.head_dim
            )
            k = self.k_proj[layer](conditioned)[memory_rows].reshape(
                n_tokens, self.num_heads, self.head_dim
            )
            v = self.v_proj[layer](conditioned)[memory_rows].reshape(
                n_tokens, self.num_heads, self.head_dim
            )
            # One call over every head: no head is held out of the kernel.
            attended_inputs = (
                self.rope.rotate(
                    self.q_norm[layer](q).float(), q_angles[:, None, :, :]
                ),
                self.rope.rotate(
                    self.k_norm[layer](k).float(), k_angles[:, None, :, :]
                ),
                v.float(),
                q_counts,
                k_counts,
            )
            if use_flash:
                attended = self._flash_attention(*attended_inputs)
            else:
                attended = self._sdpa_attention(*attended_inputs)

            attended = attended.reshape(n_tokens, self.dim).to(snapshot.dtype)
            x = snapshot + self._cross_branch(
                layer, self.out_proj[layer](attended)
            )
            x = x + self.ffn[layer](self.ffn_norm[layer](x))

        query, key = self._final_match_descriptors(x, memory_rows)
        # The barrier softmax is V11's, chunked and recomputed in backward. The
        # feature payload is zero because this attention is a coordinate
        # readout only; probability still multiplies the real opposite-frame
        # key coordinates inside ``_barrier_attention``.
        _unused_feature, matched_position = self._barrier_attention(
            query,
            key,
            key.new_zeros(key.shape),
            query_position,
            key_position,
            layout["q_count_values"],
            layout["k_count_values"],
            layout["frame_radius_values"],
        )
        displacement = matched_position.float() - query_position
        return x, {
            "delta_p_match": displacement,
            "delta_p_init": displacement,
            "matched_position": matched_position,
        }


class GroupedHeadRMSNorm(nn.Module):
    """RMSNorm over the head dimension with one learned gain per head *group*.

    Under RMSNorm ``|q| = gamma_rms * sqrt(head_dim)`` holds exactly for every
    head, so a single gain broadcast over all heads pins them to one logit
    temperature by construction.  That is right when the heads share a job, and
    wrong when one of them does not: V11.1's head 0 must reach a logit gap near
    seven to make its coordinate readout metric, while the RoPE feature heads
    want to keep averaging broadly.  Splitting the gain on exactly the boundary
    the position encoding already uses lets head 0 sharpen on its own.

    ``group_of_head[h]`` names the gain row head ``h`` reads.
    """

    def __init__(self, head_dim: int, group_of_head, eps: float = 1.0e-6):
        super().__init__()
        index = torch.as_tensor(list(group_of_head), dtype=torch.long)
        if index.ndim != 1 or index.numel() < 1:
            raise ValueError("group_of_head must list one group per head")
        if int(index.min()) < 0:
            raise ValueError("group_of_head entries must be non-negative")
        self.head_dim = int(head_dim)
        self.eps = float(eps)
        self.num_groups = int(index.max()) + 1
        # Ones is the RMSNorm identity, matching nn.RMSNorm's initialization.
        self.weight = nn.Parameter(torch.ones(self.num_groups, self.head_dim))
        self.register_buffer("group_of_head", index, persistent=False)

    def forward(self, x):
        if x.ndim < 2 or x.shape[-1] != self.head_dim:
            raise ValueError(f"grouped RMSNorm expects (..., H, {self.head_dim})")
        if x.shape[-2] != self.group_of_head.numel():
            raise ValueError("grouped RMSNorm head count does not match config")
        value = x.float()
        normalized = value * torch.rsqrt(
            value.square().mean(dim=-1, keepdim=True) + self.eps
        )
        return (normalized * self.weight[self.group_of_head]).to(x.dtype)

    @torch.no_grad()
    def group_gain_rms(self):
        return [
            float(self.weight[g].float().square().mean().sqrt())
            for g in range(self.num_groups)
        ]


class FixedMotionHeadRMSNorm(nn.Module):
    """Two-group RMSNorm with a fixed motion-head gain.

    Head ``match_head`` is normalized with an implicit, non-trainable gain of
    one. Every other head shares the single learned ``rope_weight`` vector.
    Consequently no parameter exists that can change the motion head's output
    norm, while the RoPE feature heads retain the usual learnable RMSNorm gain.
    """

    def __init__(
        self,
        head_dim: int,
        num_heads: int,
        *,
        match_head: int = 0,
        eps: float | None = None,
    ):
        super().__init__()
        self.head_dim = int(head_dim)
        self.num_heads = int(num_heads)
        self.match_head = int(match_head)
        self.eps = None if eps is None else float(eps)
        if self.head_dim <= 0 or self.num_heads < 2:
            raise ValueError(
                "fixed-motion RMSNorm needs a positive width and at least two heads"
            )
        if not 0 <= self.match_head < self.num_heads:
            raise ValueError("match_head must index one configured head")

        # This is the only affine parameter: heads 1..H-1 share it. Head 0's
        # gain is synthesized as a constant one in forward and is never placed
        # in the optimizer or checkpoint.
        self.rope_weight = nn.Parameter(torch.ones(self.head_dim))
        is_match = torch.arange(self.num_heads) == self.match_head
        self.register_buffer("is_match_head", is_match, persistent=False)

    def forward(self, x):
        expected = (self.num_heads, self.head_dim)
        if x.ndim < 2 or tuple(x.shape[-2:]) != expected:
            raise ValueError(
                "fixed-motion RMSNorm expects (..., "
                f"{self.num_heads}, {self.head_dim})"
            )
        value = x.float()
        normalized = F.rms_norm(
            value,
            (self.head_dim,),
            weight=None,
            eps=self.eps,
        )
        learned = self.rope_weight.reshape(1, -1).expand(
            self.num_heads, -1
        )
        gain = torch.where(
            self.is_match_head.reshape(-1, 1),
            torch.ones_like(learned),
            learned,
        )
        return (normalized * gain).to(x.dtype)

    @torch.no_grad()
    def gain_rms(self):
        return 1.0, float(
            self.rope_weight.float().square().mean().sqrt()
        )


class FixedMotionGainLearnableMaxSpeedBarrierCrossAttention(
    LearnableMaxSpeedBarrierLayerWeightedCrossAttention
):
    """V12.1: V12 with fixed-gain head 0 and learned-gain RoPE heads.

    Q and K keep separate gain parameters at every layer, but within either
    projection heads 1..H-1 share one vector. Head 0 has no affine parameter:
    its RMSNorm gain is always one, so its Q/K logit scale cannot grow during
    training. V12's learned global ``v_max`` is inherited unchanged. A second
    global positive scalar learns the barrier strength,

    ``lambda = exp(log_barrier_lambda)``,

    and scales ``relu(d / (v_max * |dt|) - 1)``. It starts at one, so fresh
    V12.1 behavior still matches V12 before either scalar receives an update.
    """

    def __init__(self, cfg, dim: int):
        super().__init__(cfg, dim)
        self.initial_barrier_lambda = float(
            _cfg_get(cfg, "barrier_lambda_init", 1.0)
        )
        if (
            not math.isfinite(self.initial_barrier_lambda)
            or self.initial_barrier_lambda <= 0.0
        ):
            raise ValueError("V12.1 barrier_lambda_init must be positive and finite")
        self.log_barrier_lambda = nn.Parameter(torch.tensor(
            math.log(self.initial_barrier_lambda), dtype=torch.float32
        ))
        self.q_norm = nn.ModuleList([
            FixedMotionHeadRMSNorm(
                self.head_dim,
                self.num_heads,
                match_head=self._MATCH_HEAD,
            )
            for _ in range(self.n_layers)
        ])
        self.k_norm = nn.ModuleList([
            FixedMotionHeadRMSNorm(
                self.head_dim,
                self.num_heads,
                match_head=self._MATCH_HEAD,
            )
            for _ in range(self.n_layers)
        ])

    @property
    def barrier_lambda(self):
        """Current positive multiplier on the soft-barrier penalty."""
        return self.log_barrier_lambda.exp()

    def _barrier_bias(self, query_position, key_position, radius):
        penalty = super()._barrier_bias(
            query_position, key_position, radius
        )
        return self.barrier_lambda * penalty

    @torch.no_grad()
    def attention_temperature_stats(self):
        """Report the fixed match gain and learned RoPE gain separately."""
        stats = {
            "qk_motion_head_count": 1.0,
            "barrier_speed_initial_mps": self.initial_barrier_speed_mps,
            "barrier_v_0": float(self.v_0),
            "barrier_speed_mps": float(self.v_max),
            "barrier_lambda_initial": self.initial_barrier_lambda,
            "barrier_log_lambda": float(self.log_barrier_lambda),
            "barrier_lambda": float(self.barrier_lambda),
            "barrier_weight": self.barrier_weight,
            "barrier_exponent": float(self.barrier_exponent),
        }
        bounds = {"match": 0.0, "rope": 0.0}
        for layer in range(self.n_layers):
            q_match, q_rope = self.q_norm[layer].gain_rms()
            k_match, k_rope = self.k_norm[layer].gain_rms()
            for tag, rms_q, rms_k in (
                ("match", q_match, k_match),
                ("rope", q_rope, k_rope),
            ):
                stats[f"qk_gamma_q_{tag}_layer{layer}"] = rms_q
                stats[f"qk_gamma_k_{tag}_layer{layer}"] = rms_k
                bounds[tag] = max(
                    bounds[tag],
                    rms_q * rms_k * self.head_dim * self.scale,
                )
        for tag, bound in bounds.items():
            stats[f"qk_max_logit_bound_{tag}"] = bound
        stats["qk_max_logit_bound"] = max(bounds.values())
        return stats


class LayerwiseSoftplusMaxSpeedBarrierCrossAttention(
    _LearnedRadiusBarrierCrossAttention
):
    """V12.2: V12's learned barrier speed, one per layer, through softplus.

    V12 learns one global ``v_max = exp(v_0)``. V12.2 gives every layer its own
    speed and replaces the exponential with a softplus:

    ``v_max_l = softplus(v_raw_l)``, ``R_{l,i} = v_max_l * |dt_i|``

    for layers ``l = 0..L-1``. Head 0 of layer ``l`` reads opposite-frame
    coordinates under that layer's radius, so the layer mixture weighs readouts
    taken at different reach. Every ``v_raw_l`` starts at
    ``softplus^-1(barrier_speed_mps)``: each layer begins at the configured
    30 m/s, and a fresh V12.2 reproduces V12 and V11.4 exactly.

    Softplus keeps every speed strictly positive, as ``exp`` did, but its slope
    ``sigmoid(v_raw)`` is one at 30 m/s (to float precision) and still 0.95 at
    3 m/s. An optimizer step therefore moves a speed by about the same number
    of metres per second at any speed, whereas a step on V12's ``v_0`` scales
    the speed by a constant factor. Nothing clamps a speed or pulls it back
    toward its initial value.

    QK-Norm, the layer mixture and the linear hinge are V12's, unchanged.
    """

    def __init__(self, cfg, dim: int):
        super().__init__(cfg, dim)
        self.initial_barrier_speed_mps = self.barrier_speed_mps
        speed = self.initial_barrier_speed_mps
        # expm1 keeps softplus^-1 accurate for small speeds. At 30 m/s the
        # correction is 1e-13, below float32 spacing.
        self.v_raw = nn.Parameter(torch.full(
            (self.n_layers,), _softplus_inverse(speed), dtype=torch.float32
        ))

    @property
    def v_max(self):
        """Current positive max speed of every layer in m/s, shape ``(L,)``."""
        return F.softplus(self.v_raw)

    @torch.no_grad()
    def attention_temperature_stats(self):
        """V11.4 temperature diagnostics plus every layer's learned speed."""
        stats = super().attention_temperature_stats()
        # V11.4's scalar entry is the configured initial speed, and no single
        # current speed exists any more; per-layer values and their range
        # replace it rather than leave it reporting the initialization.
        del stats["barrier_speed_mps"]
        stats["barrier_speed_initial_mps"] = self.initial_barrier_speed_mps
        raw = self.v_raw.float().cpu()
        speed = self.v_max.float().cpu()
        for layer in range(self.n_layers):
            stats[f"barrier_v_raw_layer{layer}"] = float(raw[layer])
            stats[f"barrier_speed_mps_layer{layer}"] = float(speed[layer])
        stats["barrier_speed_mps_min"] = float(speed.min())
        stats["barrier_speed_mps_mean"] = float(speed.mean())
        stats["barrier_speed_mps_max"] = float(speed.max())
        return stats

    def _prepare_forward(
        self,
        feature,
        position_ref,
        token_offset,
        frame_batch_idx,
        token_duration_sec,
    ):
        layout = super()._prepare_forward(
            feature,
            position_ref,
            token_offset,
            frame_batch_idx,
            token_duration_sec,
        )
        frame_duration = self._frame_first(
            token_duration_sec, layout["q_counts"]
        ).float()
        # Row l holds layer l's radius for every frame. As in V12, 0-D tensors
        # carry each radius into the checkpointed chunks; Python floats would
        # silently detach the learned speeds.
        radius = (
            self.v_max[:, None] * frame_duration[None, :]
        ).clamp_min(1.0e-4)
        # Drop V11.4's shared fixed-speed floats so nothing can read them.
        del layout["frame_radius_values"]
        layout["layer_frame_radius_values"] = tuple(
            tuple(row.unbind()) for row in radius.unbind()
        )
        return layout

    def _layer_frame_radius_values(self, layout, layer):
        return layout["layer_frame_radius_values"][layer]


class CosineRoPELayerWeightedCrossAttention(
    BidirectionalCrossAttentionBase
):
    """V13 all-head RoPE refinement with a cosine motion head.

    Every layer sends every head through one varlen FlashAttention call. All
    heads receive the same metric 3D RoPE, and no head has an RMSNorm module.
    Heads 1..H-1 retain the raw projected Q/K magnitude and the transformer's
    ordinary ``1/sqrt(head_dim)`` score scale. Head 0 alone is L2-normalized and
    uses ``cosine / motion_temperature``.

    FlashAttention accepts one softmax scale for the complete tensor. To give
    head 0 its own temperature without splitting the kernel, its normalized Q
    and K are each multiplied by

    ``sqrt(1 / (motion_temperature * global_softmax_scale))``.

    Their dot product therefore becomes exactly cosine/temperature after the
    shared scale is applied, while every other head is untouched. Q/K/V are
    padded to an aligned width and head 0's V carries opposite-frame xyz in the
    extra channels. The same Flash call consequently returns the feature update
    and head 0's coordinate expectation. Plain learned layer logits mix those
    per-layer expectations exactly as in V11-V12.1.
    """

    _MATCH_HEAD = 0
    _XYZ_CHANNELS = 3
    _FLASH_HEAD_ALIGNMENT = 8
    _USES_COSINE_MOTION_SCORE = True
    _REQUIRES_QK_NORM = False
    _REQUIRED_POSITION_ENCODING = "rope_all_heads"
    _VARIANT_LABEL = "V13"

    def __init__(self, cfg, dim: int):
        super().__init__()
        self.dim = int(dim)
        self.num_heads = int(_cfg_get(cfg, "num_heads", 8))
        self.n_layers = int(_cfg_get(cfg, "layers", 1))
        self.mlp_ratio = float(_cfg_get(cfg, "mlp_ratio", 4.0))
        self.implementation = str(
            _cfg_get(cfg, "implementation", "flash_varlen")
        ).lower()
        self.use_time_embedding = bool(
            _cfg_get(cfg, "use_time_embedding", True)
        )
        self.position_encoding = str(
            _cfg_get(
                cfg, "position_encoding", self._REQUIRED_POSITION_ENCODING
            )
        ).lower()
        self.qk_norm = bool(_cfg_get(cfg, "qk_norm", False))
        if self._USES_COSINE_MOTION_SCORE:
            self.motion_temperature = float(
                _cfg_get(cfg, "motion_temperature", 0.03)
            )

        if self.dim <= 0 or self.num_heads <= 0:
            raise ValueError(
                f"{self._VARIANT_LABEL} attention dimensions must be positive"
            )
        if self.dim % self.num_heads != 0:
            raise ValueError("dynamic token dim must be divisible by num_heads")
        if self.n_layers < 1:
            raise ValueError("dynamic temporal layers must be at least one")
        if self.mlp_ratio <= 0.0:
            raise ValueError("dynamic temporal mlp_ratio must be positive")
        if not self.use_time_embedding:
            raise ValueError(
                f"{self._VARIANT_LABEL} temporal attention requires time "
                "embedding"
            )
        if self.position_encoding != self._REQUIRED_POSITION_ENCODING:
            raise ValueError(
                f"{self._VARIANT_LABEL} temporal position_encoding must be "
                f"'{self._REQUIRED_POSITION_ENCODING}'"
            )
        if self.qk_norm != self._REQUIRES_QK_NORM:
            raise ValueError(
                f"{self._VARIANT_LABEL} requires qk_norm="
                f"{str(self._REQUIRES_QK_NORM).lower()}"
            )
        if self._USES_COSINE_MOTION_SCORE and (
            not math.isfinite(self.motion_temperature)
            or self.motion_temperature <= 0.0
        ):
            raise ValueError("V13 motion_temperature must be positive and finite")

        self.head_dim = self.dim // self.num_heads
        if self.head_dim % 6 != 0:
            raise ValueError(
                "dynamic attention head dim must be divisible by 6 for 3D RoPE"
            )
        self.scale = self.head_dim ** -0.5
        if self._USES_COSINE_MOTION_SCORE:
            self.motion_qk_scale = math.sqrt(
                1.0 / (self.motion_temperature * self.scale)
            )
        unpadded_head_dim = self.head_dim + self._XYZ_CHANNELS
        self.attention_head_dim = (
            (unpadded_head_dim + self._FLASH_HEAD_ALIGNMENT - 1)
            // self._FLASH_HEAD_ALIGNMENT
            * self._FLASH_HEAD_ALIGNMENT
        )
        if self.attention_head_dim > 256:
            raise ValueError(
                f"{self._VARIANT_LABEL} padded attention head width must not "
                "exceed 256"
            )

        rope_position_scale = _cfg_get(cfg, "rope_position_scale", None)
        if rope_position_scale is None:
            range_m = float(_cfg_get(cfg, "rope_range_m", 110.0))
            rope_position_scale = math.pi / range_m
        self.rope = Rotary3D(
            self.head_dim,
            base=float(_cfg_get(cfg, "rope_base", 100.0)),
            position_scale=float(rope_position_scale),
        )

        time_dim = int(_cfg_get(cfg, "time_embedding_dim", 64))
        time_frequencies = int(_cfg_get(cfg, "time_frequencies", 8))
        self.time_encoder = SinusoidalScalarEncoder(time_dim, time_frequencies)
        self.time_to_feature = nn.Linear(time_dim, self.dim, bias=False)

        if self.qk_norm:
            self._build_qk_norm(True)
        # V13/V13.1/V13.2 intentionally register no q_norm/k_norm modules.
        # V13.3/V13.4 instead build the same per-layer affine RMSNorm pair as
        # V11.4.
        self._build_transformer_layers(cfg)
        self.layer_logits = nn.Parameter(torch.zeros(self.n_layers))

    @torch.no_grad()
    def attention_temperature_stats(self):
        """Report the fixed cosine scale; feature-head logits are unbounded."""
        return {
            "qk_motion_head_count": 1.0,
            "qk_norm_enabled": 0.0,
            "motion_temperature": self.motion_temperature,
            "motion_max_abs_content_logit": 1.0 / self.motion_temperature,
        }

    def _encode_qk(self, query, key, query_angles, key_angles):
        """Apply head-0 cosine scaling and all-head 3D RoPE."""
        expected_query = (query.shape[0], self.num_heads, self.head_dim)
        expected_key = (key.shape[0], self.num_heads, self.head_dim)
        if tuple(query.shape) != expected_query or tuple(key.shape) != expected_key:
            raise ValueError(
                f"{self._VARIANT_LABEL} Q/K must have shape "
                "(tokens, num_heads, head_dim)"
            )
        query = query.float()
        key = key.float()
        motion_query = F.normalize(
            query[:, self._MATCH_HEAD], p=2.0, dim=-1, eps=1.0e-6
        ) * self.motion_qk_scale
        motion_key = F.normalize(
            key[:, self._MATCH_HEAD], p=2.0, dim=-1, eps=1.0e-6
        ) * self.motion_qk_scale
        query = torch.cat([
            motion_query.unsqueeze(1), query[:, self._MATCH_HEAD + 1:],
        ], dim=1)
        key = torch.cat([
            motion_key.unsqueeze(1), key[:, self._MATCH_HEAD + 1:],
        ], dim=1)
        return (
            self.rope.rotate(query, query_angles[:, None, :, :]),
            self.rope.rotate(key, key_angles[:, None, :, :]),
        )

    def _pad_attention_tensors(self, query, key, value, key_position):
        """Align the Flash width and append xyz only to head 0's V payload."""
        if key_position.shape != (key.shape[0], 3):
            raise ValueError(
                f"{self._VARIANT_LABEL} key positions must align with K/V rows"
            )
        padded_query = query.new_zeros(
            query.shape[0], self.num_heads, self.attention_head_dim,
            dtype=torch.float32,
        )
        padded_key = key.new_zeros(
            key.shape[0], self.num_heads, self.attention_head_dim,
            dtype=torch.float32,
        )
        padded_value = value.new_zeros(
            value.shape[0], self.num_heads, self.attention_head_dim,
            dtype=torch.float32,
        )
        padded_query[..., :self.head_dim] = query.float()
        padded_key[..., :self.head_dim] = key.float()
        padded_value[..., :self.head_dim] = value.float()
        padded_value[
            :, self._MATCH_HEAD,
            self.head_dim:self.head_dim + self._XYZ_CHANNELS,
        ] = key_position.float()
        return padded_query, padded_key, padded_value

    @staticmethod
    def _frame_first(value, counts):
        starts = torch.cumsum(counts, dim=0) - counts
        return value[starts.long()]

    def _prepare_forward(
        self,
        feature,
        position_ref,
        token_offset,
        frame_batch_idx,
        token_duration_sec,
    ):
        n_tokens = int(feature.shape[0])
        if feature.shape != (n_tokens, self.dim):
            raise ValueError(
                f"{self._VARIANT_LABEL} feature must be (N,{self.dim})"
            )
        if position_ref.shape != (n_tokens, 3):
            raise ValueError("position_ref must align with V13 token features")
        if token_duration_sec.shape != (n_tokens,):
            raise ValueError(
                f"{self._VARIANT_LABEL} duration must provide one scalar per "
                "token"
            )
        if not bool(torch.all(
            torch.isfinite(token_duration_sec) & (token_duration_sec > 0.0)
        )):
            raise ValueError(
                f"{self._VARIANT_LABEL} requires positive finite durations"
            )

        q_counts, k_counts, memory_rows = self._frame_layout(
            token_offset, frame_batch_idx
        )
        if int(q_counts.sum()) != n_tokens:
            raise ValueError("token_offset does not cover all V13 tokens")
        if bool(torch.any(q_counts <= 0)) or bool(torch.any(k_counts <= 0)):
            raise ValueError(
                f"{self._VARIANT_LABEL} requires tokens in both endpoint frames"
            )
        frame_duration = self._frame_first(token_duration_sec, q_counts)
        if not torch.equal(
            torch.repeat_interleave(frame_duration, q_counts),
            token_duration_sec,
        ):
            raise ValueError(
                f"{self._VARIANT_LABEL} requires one physical endpoint "
                "interval per frame"
            )
        return q_counts, k_counts, memory_rows

    def _use_flash(self, x):
        if (
            x.is_cuda
            and self.implementation == "flash_varlen"
            and flash_attn is None
        ):
            raise RuntimeError(
                f"{self._VARIANT_LABEL} temporal attention requested "
                "flash_varlen, but "
                "flash_attn is unavailable"
            )
        return (
            x.is_cuda
            and flash_attn is not None
            and self.implementation in ("flash_varlen", "auto")
        )

    def _combine_layer_matches(self, layer_matched_positions, position_ref):
        if layer_matched_positions.shape != (
            position_ref.shape[0], self.n_layers, 3
        ):
            raise ValueError(
                f"{self._VARIANT_LABEL} layer matches must have shape (N,L,3)"
            )
        weights = torch.softmax(self.layer_logits.float(), dim=-1)
        matched_position = (
            weights.reshape(1, -1, 1) * layer_matched_positions.float()
        ).sum(dim=1)
        displacement = matched_position - position_ref.float()
        token_count = int(position_ref.shape[0])
        return (
            matched_position,
            displacement,
            self.layer_logits.detach().expand(token_count, -1),
            weights.detach().expand(token_count, -1),
        )

    def forward(
        self,
        feature,
        position_ref,
        token_offset,
        frame_batch_idx,
        token_time_coordinate,
        token_duration_sec,
    ):
        n_tokens = int(feature.shape[0])
        if token_time_coordinate.shape != (n_tokens,):
            raise ValueError(
                f"{self._VARIANT_LABEL} token time must provide one scalar "
                "per token"
            )
        q_counts, k_counts, memory_rows = self._prepare_forward(
            feature, position_ref, token_offset, frame_batch_idx,
            token_duration_sec,
        )

        x = feature + self.time_to_feature(
            self.time_encoder(token_time_coordinate)
        ).to(feature.dtype)
        query_position = position_ref.float()
        key_position = position_ref[memory_rows].float()
        query_angles = self.rope.angles(query_position)
        key_angles = self.rope.angles(key_position)
        use_flash = self._use_flash(x)

        layer_matched_positions = []
        xyz_slice = slice(
            self.head_dim, self.head_dim + self._XYZ_CHANNELS
        )
        for layer in range(self.n_layers):
            snapshot = x
            conditioned = self.input_norm[layer](snapshot)
            query = self.q_proj[layer](conditioned).reshape(
                n_tokens, self.num_heads, self.head_dim
            )
            key = self.k_proj[layer](conditioned)[memory_rows].reshape(
                n_tokens, self.num_heads, self.head_dim
            )
            value = self.v_proj[layer](conditioned)[memory_rows].reshape(
                n_tokens, self.num_heads, self.head_dim
            )
            if self.qk_norm:
                query = self.q_norm[layer](query)
                key = self.k_norm[layer](key)
            query, key = self._encode_qk(
                query, key, query_angles, key_angles
            )
            query, key, value = self._pad_attention_tensors(
                query, key, value, key_position
            )
            if use_flash:
                # The V payload contains metric xyz; fp16 retains materially
                # finer driving-range coordinates than bf16.
                attended = self._flash_attention(
                    query, key, value, q_counts, k_counts,
                    flash_dtype=torch.float16,
                )
            else:
                attended = self._sdpa_attention(
                    query, key, value, q_counts, k_counts
                )

            layer_matched_positions.append(
                attended[:, self._MATCH_HEAD, xyz_slice]
            )
            attended_feature = attended[..., :self.head_dim].reshape(
                n_tokens, self.dim
            ).to(snapshot.dtype)
            x = snapshot + self._cross_branch(
                layer, self.out_proj[layer](attended_feature)
            )
            x = x + self.ffn[layer](self.ffn_norm[layer](x))

        (
            matched_position,
            displacement,
            token_logits,
            token_weights,
        ) = self._combine_layer_matches(
            torch.stack(layer_matched_positions, dim=1), position_ref
        )
        return x, {
            "delta_p_match": displacement,
            "delta_p_init": displacement,
            "matched_position": matched_position,
            "motion_layer_logits": token_logits,
            "motion_layer_weights": token_weights,
        }


class RawQKRoPELayerWeightedCrossAttention(
    CosineRoPELayerWeightedCrossAttention
):
    """V13.2: V13 with ordinary raw-QK scoring on head 0.

    No head uses RMSNorm or L2 normalization. Head 0 therefore follows exactly
    the same score contract as heads 1..H-1,

    ``dot(RoPE(q), RoPE(k)) / sqrt(head_dim)``.

    The inherited padding keeps opposite-frame xyz in head 0's V payload, so
    all heads still run in one FlashAttention call and the layer-wise coordinate
    readout is unchanged apart from the score itself.
    """

    _USES_COSINE_MOTION_SCORE = False
    _VARIANT_LABEL = "V13.2"

    @torch.no_grad()
    def attention_temperature_stats(self):
        """Raw Q/K norms are data-dependent, so no finite bound is reported."""
        return {
            "qk_motion_head_count": 1.0,
            "qk_norm_enabled": 0.0,
            "motion_score_scale": self.scale,
        }

    def _encode_qk(self, query, key, query_angles, key_angles):
        """Apply 3D RoPE without normalizing or rescaling any head."""
        expected_query = (query.shape[0], self.num_heads, self.head_dim)
        expected_key = (key.shape[0], self.num_heads, self.head_dim)
        if tuple(query.shape) != expected_query or tuple(key.shape) != expected_key:
            raise ValueError(
                f"{self._VARIANT_LABEL} Q/K must have shape "
                "(tokens, num_heads, head_dim)"
            )
        return (
            self.rope.rotate(query.float(), query_angles[:, None, :, :]),
            self.rope.rotate(key.float(), key_angles[:, None, :, :]),
        )


class RMSNormRoPELayerWeightedCrossAttention(
    RawQKRoPELayerWeightedCrossAttention
):
    """V13.3: V11.4 Q/K RMSNorm with all-head 3D RoPE/FlashAttention.

    Every layer RMS-normalizes each projected Q/K head with one affine gain
    vector shared across heads, exactly as V11.4 did. Unlike V11.4, head 0 has
    no distance-dependent barrier and receives the same metric 3D RoPE as the
    remaining heads. The inherited packed V payload still carries key xyz only
    on head 0, so one all-head FlashAttention call returns both refined
    features and the coordinate expectation used by the learned layer mixture.
    """

    _REQUIRES_QK_NORM = True
    _VARIANT_LABEL = "V13.3"

    @torch.no_grad()
    def attention_temperature_stats(self):
        """Report the all-head RMS gains and finite content-logit bound."""
        stats = {
            "qk_motion_head_count": 1.0,
            "qk_norm_enabled": 1.0,
            "motion_score_scale": self.scale,
        }
        bound = 0.0
        for layer in range(self.n_layers):
            q_gain = self.q_norm[layer].weight.float()
            k_gain = self.k_norm[layer].weight.float()
            rms_q = float(q_gain.square().mean().sqrt())
            rms_k = float(k_gain.square().mean().sqrt())
            bound = max(
                bound, rms_q * rms_k * self.head_dim * self.scale
            )
            stats[f"qk_gamma_q_layer{layer}"] = rms_q
            stats[f"qk_gamma_k_layer{layer}"] = rms_k
        stats["qk_max_logit_bound"] = bound
        return stats


class RMSNormFeatureRoPELayerWeightedCrossAttention(
    RMSNormRoPELayerWeightedCrossAttention
):
    """V13.4: V13.3 with 3D RoPE removed from match head 0 only.

    Q and K are still RMS-normalized on every head. Head 0 then enters the
    shared FlashAttention call unchanged, while heads 1..H-1 receive the same
    metric 3D RoPE as V13.3. The packed head-0 V payload and learned layer
    mixture are inherited unchanged, and no additive distance term is added.
    """

    _REQUIRED_POSITION_ENCODING = "rope_except_head0"
    _VARIANT_LABEL = "V13.4"

    def __init__(self, cfg, dim: int):
        super().__init__(cfg, dim)
        if self.num_heads < 2:
            raise ValueError(
                "V13.4 requires one unrotated match head and at least one "
                "3D-RoPE feature head"
            )

    def _encode_qk(self, query, key, query_angles, key_angles):
        """Leave head 0 unchanged and rotate every remaining head."""
        expected_query = (query.shape[0], self.num_heads, self.head_dim)
        expected_key = (key.shape[0], self.num_heads, self.head_dim)
        if tuple(query.shape) != expected_query or tuple(key.shape) != expected_key:
            raise ValueError(
                "V13.4 Q/K must have shape (tokens, num_heads, head_dim)"
            )
        query = query.float()
        key = key.float()
        return (
            torch.cat([
                query[:, :1],
                self.rope.rotate(
                    query[:, 1:], query_angles[:, None, :, :]
                ),
            ], dim=1),
            torch.cat([
                key[:, :1],
                self.rope.rotate(
                    key[:, 1:], key_angles[:, None, :, :]
                ),
            ], dim=1),
        )


class GroupedGainBarrierCrossAttention(
    MaxSpeedBarrierLayerWeightedCrossAttention
):
    """V11.1: V11's stack with QK-Norm's gain split on the head-0 boundary.

    This is the only change to the temporal module.  Everything V11 does --
    the max-speed barrier on head 0, the chunked explicit softmax, RoPE on
    heads 1..H-1, and the parameter-only layer mixture -- is inherited
    unmodified, so the two variants differ by one tensor shape.

    Under RMSNorm ``|q| = gamma_rms * sqrt(head_dim)`` holds exactly for every
    head, so a single gain broadcast over all heads pins them to one logit
    temperature by construction.  Head 0 has a different job from the rest: its
    probabilities are read as a coordinate, and a soft-argmax is only metric
    when the mass is concentrated, which needs a logit gap near seven (measured
    on V11: ~37.7 tokens inside a 2 m ball against ~3735 admitted candidates).
    The RoPE feature heads have no such requirement and are free to keep
    averaging broadly.  Splitting the gain on exactly the boundary the position
    encoding already uses lets each side find its own temperature.

    Measured on V11.1's first three epochs, the two groups do separate and the
    direction depends on depth: head 0 runs up to 25% hotter than the RoPE
    heads in layers 1-5 and up to 7% cooler in layers 8-11, while V11's single
    shared gain sits above both.
    """

    def __init__(self, cfg, dim: int):
        super().__init__(cfg, dim)
        # Head 0 carries the barrier and the readout; heads 1..H-1 carry RoPE.
        # The gain split follows that same boundary.
        group_of_head = [1] * self.num_heads
        group_of_head[self._MATCH_HEAD] = 0
        self.q_norm = nn.ModuleList([
            GroupedHeadRMSNorm(self.head_dim, group_of_head)
            for _ in range(self.n_layers)
        ])
        self.k_norm = nn.ModuleList([
            GroupedHeadRMSNorm(self.head_dim, group_of_head)
            for _ in range(self.n_layers)
        ])

    @torch.no_grad()
    def attention_temperature_stats(self):
        """Per-group QK gains, so the match head can be watched on its own.

        V11 could only report one bound for all twelve heads, which is precisely
        the coupling this variant removes.
        """
        stats = {
            "qk_motion_head_count": 1.0,
            "barrier_speed_mps": self.barrier_speed_mps,
            "barrier_weight": self.barrier_weight,
            "barrier_exponent": float(self.barrier_exponent),
        }
        names = {0: "match", 1: "rope"}
        bounds = {}
        for layer in range(self.n_layers):
            gq = self.q_norm[layer].group_gain_rms()
            gk = self.k_norm[layer].group_gain_rms()
            for group, (rms_q, rms_k) in enumerate(zip(gq, gk)):
                tag = names.get(group, str(group))
                stats[f"qk_gamma_q_{tag}_layer{layer}"] = rms_q
                stats[f"qk_gamma_k_{tag}_layer{layer}"] = rms_k
                bounds[tag] = max(
                    bounds.get(tag, 0.0),
                    rms_q * rms_k * self.head_dim * self.scale,
                )
        for tag, bound in bounds.items():
            stats[f"qk_max_logit_bound_{tag}"] = bound
        stats["qk_max_logit_bound"] = max(bounds.values()) if bounds else 0.0
        return stats

__all__ = [
    "BidirectionalCrossAttentionBase",
    "MaxSpeedBarrierLayerWeightedCrossAttention",
    "FinalFeatureBarrierCrossAttention",
    "CosineRoPELayerWeightedCrossAttention",
    "RawQKRoPELayerWeightedCrossAttention",
    "RMSNormRoPELayerWeightedCrossAttention",
    "RMSNormFeatureRoPELayerWeightedCrossAttention",
    "FixedMotionGainLearnableMaxSpeedBarrierCrossAttention",
    "FixedMotionHeadRMSNorm",
    "GroupedHeadRMSNorm",
    "GroupedGainBarrierCrossAttention",
    "GlobalSoftplusMaxSpeedBarrierCrossAttention",
    "LayerwiseSoftplusMaxSpeedBarrierCrossAttention",
    "LearnableMaxSpeedBarrierLayerWeightedCrossAttention",
]
