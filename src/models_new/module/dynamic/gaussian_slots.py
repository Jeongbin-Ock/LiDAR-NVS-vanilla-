"""Token-anchored Gaussian slots of the V15 family, and V15.3's dense head.

A slot is one Gaussian's raw parameters, ``P`` channels in ``SLOT_PARAM_NAMES``
order (shs, opacity, scaling, rotation, offset). Every V15-family head decodes
slots around the token's one anchor, its Utonia position in the window reference
frame: a slot sits at ``anchor + offset_bound * tanh(offset)``. The heads differ
only in which slots a token renders -- a routed prefix of G1..G5 (V15/V15.1,
``nested_count.py``), one routed per-option expert (V15.2,
``option_experts.py``), or all of them (V15.3, :class:`DenseCountGaussianHead`
here) -- so the slot layout, its initialization and its packing live here.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from .common import _cfg_get

SLOT_PARAM_NAMES = ("shs", "opacity", "scaling", "rotation", "offset")


def slot_param_sizes(gs_params) -> list[int]:
    """Channel widths of one slot, in ``SLOT_PARAM_NAMES`` order."""
    sizes = [int(getattr(gs_params, name)) for name in SLOT_PARAM_NAMES]
    if sizes[-2:] != [4, 3]:
        raise ValueError(
            "token-anchored Gaussian slots require rotation=4 and offset=3"
        )
    return sizes


def initialize_slot_projection(projection: nn.Linear, head_cfg, param_sizes,
                               slots: int) -> None:
    """Start every slot of ``projection``'s output from the dynamic-head priors.

    The output holds ``slots`` consecutive slots. Weights are N(0, 0.01) except
    the offset rows, which are zero, and every slot's bias is the prior:
    ``initial_opacity`` as a logit, ``initial_scale_m`` through the inverse
    softplus, the identity quaternion and a zero offset. Every slot therefore
    starts on the token anchor with the same shape, and the weight noise breaks
    the symmetry between them.
    """
    opacity = float(_cfg_get(head_cfg, "initial_opacity", 0.2))
    opacity = min(max(opacity, 1.0e-4), 1.0 - 1.0e-4)
    scale = float(_cfg_get(head_cfg, "initial_scale_m", 0.3))
    sh_size, opacity_size, scale_size, rotation_size, offset_size = param_sizes
    param_dim = sum(param_sizes)
    if projection.out_features != slots * param_dim:
        raise ValueError(
            f"a {slots}-slot projection must have {slots * param_dim} outputs, "
            f"got {projection.out_features}"
        )
    opacity_start = sh_size
    scale_start = opacity_start + opacity_size
    rotation_start = scale_start + scale_size
    offset_start = rotation_start + rotation_size

    slot_bias = projection.bias.new_zeros(param_dim)
    slot_bias[opacity_start:scale_start] = math.log(opacity / (1.0 - opacity))
    slot_bias[scale_start:rotation_start] = math.log(
        math.expm1(max(scale, 1.0e-4))
    )
    slot_bias[rotation_start] = 1.0
    nn.init.normal_(projection.weight, mean=0.0, std=0.01)
    with torch.no_grad():
        projection.bias.copy_(slot_bias.repeat(slots))
        for slot in range(slots):
            row_start = slot * param_dim + offset_start
            projection.weight[row_start:row_start + offset_size].zero_()


def pack_slots(params, anchor, token_index, slot_index, param_sizes,
               offset_bound: float):
    """Per-Gaussian raw attributes and positions for the listed slots.

    ``params`` holds one gathered ``(M, P)`` slot row per rendered Gaussian,
    ``token_index`` the token each row belongs to and ``slot_index`` its slot
    number within what that token renders.
    """
    raw = dict(zip(SLOT_PARAM_NAMES, torch.split(params, param_sizes, dim=-1)))
    offset = raw.pop("offset")
    position = anchor[token_index] + offset_bound * torch.tanh(offset)
    return {
        "token_index": token_index,
        "slot_index": slot_index,
        "raw": raw,
        "position": position,
    }


class DenseCountGaussianHead(nn.Module):
    """V15.3: every token decodes and renders the same number of Gaussians.

    ``decoder_norm -> Linear(dim, decoder_hidden) -> SiLU ->
    Linear(decoder_hidden, G * P)`` decodes ``G = gaussians_per_token`` slots
    (V15.3: five) around the token anchor, and all of them are rendered, in
    training and at evaluation alike. This is V15's G1..G5 decoder -- the same
    module names, shapes and initialization -- with no router: nothing chooses
    a count, nothing gates an opacity and there is no auxiliary render.
    ``decoder_hidden`` is read from ``dynamic_2dgs.gaussian_head``.

    ``forward`` returns ``(rendered, None, None)``, the routed heads'
    ``(routed, auxiliary, routing)`` contract with nothing routed.
    """

    PARAM_NAMES = SLOT_PARAM_NAMES

    def __init__(self, head_cfg, gs_params, dim: int, offset_bound: float,
                 gaussians_per_token: int):
        super().__init__()
        self.dim = int(dim)
        self.offset_bound = float(offset_bound)
        self.gaussians_per_token = int(gaussians_per_token)
        if self.gaussians_per_token < 1:
            raise ValueError("gaussians_per_token must be at least one")
        hidden = _cfg_get(head_cfg, "decoder_hidden", None)
        if isinstance(hidden, bool) or not isinstance(hidden, int) or hidden <= 0:
            raise ValueError(
                "the dense Gaussian head needs a positive integer "
                f"dynamic_2dgs.gaussian_head.decoder_hidden, got {hidden!r}"
            )
        self.decoder_hidden = int(hidden)
        self.param_sizes = slot_param_sizes(gs_params)
        self.param_dim = sum(self.param_sizes)

        self.decoder_norm = nn.LayerNorm(self.dim)
        self.decoder = nn.Sequential(
            nn.Linear(self.dim, self.decoder_hidden),
            nn.SiLU(),
            nn.Linear(
                self.decoder_hidden, self.gaussians_per_token * self.param_dim
            ),
        )
        initialize_slot_projection(
            self.decoder[-1], head_cfg, self.param_sizes,
            self.gaussians_per_token,
        )

    def forward(self, feature, anchor):
        """Every token's ``G`` slots, token-major, slots ascending."""
        if feature.ndim != 2 or feature.shape[1] != self.dim:
            raise ValueError(f"dense Gaussian head expects (N,{self.dim}) features")
        if tuple(anchor.shape) != (feature.shape[0], 3):
            raise ValueError("dense Gaussian anchors must be (N,3)")
        token_count = feature.shape[0]
        slots = self.gaussians_per_token
        decoded = self.decoder(self.decoder_norm(feature)).reshape(
            token_count, slots, self.param_dim
        )
        token_index = torch.arange(
            token_count, device=feature.device
        ).repeat_interleave(slots)
        slot_index = torch.arange(slots, device=feature.device).repeat(
            token_count
        )
        rendered = pack_slots(
            decoded.reshape(token_count * slots, self.param_dim),
            anchor, token_index, slot_index, self.param_sizes,
            self.offset_bound,
        )
        return rendered, None, None


__all__ = [
    "DenseCountGaussianHead",
    "SLOT_PARAM_NAMES",
    "initialize_slot_projection",
    "pack_slots",
    "slot_param_sizes",
]
