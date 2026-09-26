"""Learned Gaussian-count router of the router-capable Dynamic 2DGS backends.

Each occupied Utonia token arrives with its refined feature and a precomputed
seed-delta bank, one candidate row per K. A post-temporal MLP predicts K
logits, hard Gumbel-Softmax straight-through routing selects one K per token,
and only the selected K-specific joint head runs. Range-quantile rows give
every slot its own seed, so the trunk reads the selected row; token anchors
give every slot the same seed, so the decoder reads the feature alone.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gaussian_assembly import gradient_scale_identity, opacity_gate_st


class GridSlotHead(nn.Module):
    """Predict K={1..K_max} Gaussians per token with a learned count router.

    ``count_mode="learned_gumbel"`` predicts logits from the temporally refined
    token feature and executes only the selected K-specific head for each
    token. The selected hard-ST scalar gates the activated opacity of that
    expert's K outputs: forward rendering is unchanged, while backward connects
    the rendering loss to all count logits through the softmax Jacobian.

    ``seed_conditioned`` concatenates the selected row's ``3 * K_max`` seed
    deltas to the trunk input. Shared-anchor seed modes turn it off, because
    their rows repeat one seed and carry nothing slot-specific.
    """

    def __init__(self, cfg, gs_params, dim, seed_conditioned=True):
        super().__init__()
        self.dim = int(dim)
        self.seed_conditioned = bool(seed_conditioned)
        self.count_mode = str(getattr(cfg, "count_mode", "legacy")).lower()
        if self.count_mode != "learned_gumbel":
            raise ValueError(
                "the Gaussian-count router requires "
                "p2g.grid_query.count_mode='learned_gumbel'"
            )
        learned_count = getattr(cfg, "learned_count", None)
        if learned_count is None:
            raise ValueError(
                "p2g.grid_query.learned_count is required for "
                f"count_mode={self.count_mode!r}"
            )
        k_max = getattr(learned_count, "K_max", None)
        if k_max is None:
            raise ValueError("p2g.grid_query.learned_count.K_max is required")
        self.gumbel_tau = float(getattr(learned_count, "tau", 1.0))
        if not self.gumbel_tau > 0.0:
            raise ValueError("p2g.grid_query.learned_count.tau must be positive")
        self.k_max = int(k_max)
        self.grad_balance = str(getattr(cfg, "grad_balance", "sqrt_k"))
        # A missing scope keeps the historical output-level backward; the
        # shipped configs nest ``token`` under learned_count.
        self.grad_balance_scope = str(getattr(
            learned_count,
            "grad_balance_scope",
            getattr(cfg, "grad_balance_scope", "output"),
        )).lower()
        if self.k_max <= 0:
            raise ValueError("grid_query.K_max must be positive")
        if self.grad_balance not in ("sqrt_k", "none"):
            raise ValueError("grid_query.grad_balance must be 'sqrt_k' or 'none'")
        if self.grad_balance_scope not in ("output", "token"):
            raise ValueError(
                "grid_query.grad_balance_scope must be 'output' or 'token'"
            )

        param_names = ["shs", "opacity", "scaling", "rotation"]
        param_sizes = []
        for name in param_names:
            size = getattr(gs_params, name, None)
            if size is None:
                raise ValueError(f"p2g.gs_params.{name} is required for grid mode")
            param_sizes.append(int(size))
        offset_size = int(getattr(gs_params, "offset", 0) or 0)
        if offset_size > 0:
            param_sizes.append(offset_size)
        self.param_dim = sum(param_sizes)
        if self.param_dim <= 0:
            raise ValueError("p2g.gs_params must define a positive output width")
        self.opacity_start = param_sizes[0]
        self.opacity_end = self.opacity_start + param_sizes[1]

        # Module names, shapes and construction order are the checkpoint
        # contract of every router variant.
        self.seed_dim = 3 * self.k_max if self.seed_conditioned else 0
        self.trunk = nn.Sequential(
            nn.Linear(self.dim + self.seed_dim, self.dim),
            nn.SiLU(),
        )
        self.k_heads = nn.ModuleList([
            nn.Linear(self.dim, k * self.param_dim)
            for k in range(1, self.k_max + 1)
        ])
        self.count_predictor = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.SiLU(),
            nn.Linear(self.dim, self.k_max),
        )
        # Uniform initial categorical probabilities avoid imposing an
        # arbitrary K preference before the rendering losses provide signal.
        nn.init.zeros_(self.count_predictor[-1].weight)
        nn.init.zeros_(self.count_predictor[-1].bias)

    def _pack_selected_prefix(self, selected_k, anchor_offset):
        num_anchors = selected_k.shape[0]
        anchor_index = torch.repeat_interleave(
            torch.arange(num_anchors, device=selected_k.device),
            selected_k,
        )
        gaussian_offset = self._gaussian_offset(selected_k, anchor_offset)
        if anchor_index.numel() == 0:
            empty_long = torch.zeros(
                0, dtype=torch.long, device=selected_k.device
            )
            return anchor_index, empty_long, empty_long, gaussian_offset

        slot_start = torch.cumsum(selected_k, dim=0) - selected_k
        slot_index = (
            torch.arange(anchor_index.numel(), device=selected_k.device)
            - torch.repeat_interleave(slot_start, selected_k)
        )
        slot_k = selected_k[anchor_index]
        return anchor_index, slot_index, slot_k, gaussian_offset

    def _opacity_gate_st(self, raw_params, gate):
        """Forward-identical raw opacity with activated-opacity ST backward.

        The renderer applies sigmoid internally.  We therefore build the exact
        activated opacity ``sigmoid(raw) * gate``, map it back to a raw logit,
        and use a straight-through value replacement so the selected expert's
        opacities are bit-identical in the forward. Only packed selected slots
        enter this function, hence their hard gate value is one and the renderer
        never receives zero-opacity placeholders.
        """
        if raw_params.shape[0] == 0:
            return raw_params
        raw_opacity = raw_params[:, self.opacity_start:self.opacity_end]
        opacity_st = opacity_gate_st(raw_opacity, gate)
        return torch.cat([
            raw_params[:, :self.opacity_start],
            opacity_st,
            raw_params[:, self.opacity_end:],
        ], dim=-1)

    def _gaussian_offset(self, anchor_k, anchor_offset):
        frame_gaussian_counts = []
        start = 0
        for end_tensor in anchor_offset:
            end = int(end_tensor)
            frame_gaussian_counts.append(anchor_k[start:end].sum())
            start = end
        if not frame_gaussian_counts:
            return torch.zeros(0, dtype=torch.long, device=anchor_k.device)
        return torch.cumsum(torch.stack(frame_gaussian_counts), dim=0).long()

    def _balance_token_feature(self, feature, k):
        """Scale only the gradient returning to the shared token/trunk path."""
        if (
            self.grad_balance != "sqrt_k"
            or self.grad_balance_scope != "token"
        ):
            return feature
        return gradient_scale_identity(feature, float(k) ** -0.5)

    def _gumbel_selection(self, logits):
        """Hard one-hot forward; soft categorical derivative backward."""
        if self.training:
            return F.gumbel_softmax(
                logits.float(),
                tau=self.gumbel_tau,
                hard=True,
                dim=-1,
            ).to(dtype=logits.dtype)
        selected = logits.argmax(dim=-1)
        return F.one_hot(selected, num_classes=self.k_max).to(dtype=logits.dtype)

    def forward(self, anchor_feature, delta_p, anchor_offset):
        """Route every token to one K and decode only that K's Gaussians."""
        expected_delta_shape = (
            anchor_feature.shape[0], self.k_max, self.k_max, 3,
        )
        if tuple(delta_p.shape) != expected_delta_shape:
            raise ValueError(
                "learned_gumbel delta_p must have shape "
                f"{expected_delta_shape}, got {tuple(delta_p.shape)}"
            )

        logits = self.count_predictor(anchor_feature)
        selection = self._gumbel_selection(logits)
        selected_index = selection.argmax(dim=-1).to(torch.long)
        selected_k = selected_index + 1
        num_anchors = anchor_feature.shape[0]

        # K is a hard routing decision in the forward: each token contributes
        # only its selected candidate geometry and runs only its selected
        # K-specific output head. The discrete gather deliberately carries no
        # count-router gradient; that gradient is supplied below by the selected
        # ST scalar applied to activated opacity.
        anchor_rows = torch.arange(
            num_anchors, device=anchor_feature.device
        )
        if self.seed_conditioned:
            selected_delta = delta_p[anchor_rows, selected_index]
            trunk_input = torch.cat([
                anchor_feature,
                selected_delta.to(dtype=anchor_feature.dtype).reshape(
                    num_anchors, self.seed_dim
                ),
            ], dim=-1)
        else:
            trunk_input = anchor_feature
        trunk_feature = self.trunk(trunk_input)

        (
            anchor_index,
            slot_index,
            slot_k,
            gaussian_offset,
        ) = self._pack_selected_prefix(
            selected_k, anchor_offset
        )
        raw_params = anchor_feature.new_zeros(
            (anchor_index.numel(), self.param_dim)
        )
        anchor_start = torch.cumsum(selected_k, dim=0) - selected_k
        for k, head in enumerate(self.k_heads, start=1):
            selected_rows = (selected_k == k).nonzero(as_tuple=True)[0]
            if selected_rows.numel() == 0:
                continue
            head_input = self._balance_token_feature(
                trunk_feature[selected_rows], k
            )
            predicted = head(head_input).reshape(
                -1, k, self.param_dim
            )
            output_rows = anchor_start[selected_rows, None] + torch.arange(
                k, device=anchor_feature.device
            ).view(1, -1)
            raw_params = raw_params.index_copy(
                0,
                output_rows.reshape(-1),
                predicted.reshape(-1, self.param_dim),
            )

        # For token n, gather only y_ST[n, k*_n]. Its forward value is exactly
        # one. Since that single probability still depends on every logit via
        # softmax normalization, the opacity gate sends gradients to all K logits
        # without executing or padding the unselected expert outputs.
        selected_gate = selection.gather(
            1, selected_index.unsqueeze(-1)
        ).squeeze(-1)
        packed_gate = selected_gate[anchor_index]
        raw_params = self._opacity_gate_st(raw_params, packed_gate)

        return raw_params, {
            "anchor_index": anchor_index,
            "slot_index": slot_index,
            "slot_k": slot_k,
            "anchor_k": selected_k,
            "gaussian_offset": gaussian_offset,
            "k_logits": logits,
            "k_selection": selection,
            "packed_selected_gate": packed_gate,
            "selected_only": True,
        }

    def gradient_weight(self, slot_k, dtype):
        if (
            self.grad_balance == "none"
            or self.grad_balance_scope == "token"
        ):
            return torch.ones_like(slot_k, dtype=dtype)
        return slot_k.to(dtype=dtype).rsqrt()


__all__ = ["GridSlotHead"]
