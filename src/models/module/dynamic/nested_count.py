"""V15's nested Gaussian count: one G1..G5 decoder and a reduce/keep/add router.

Every token decodes five Gaussians around its one shared anchor, the Utonia
token position. The count is a prefix of that ordered set, so the three
choices are nested:

* reduce renders ``{G1}``,
* keep renders ``{G1, G2, G3}``,
* add renders ``{G1, ..., G5}``.

A router reads the locally refined token feature through its own LayerNorm and
picks one choice per token with hard Gumbel-softmax in training and argmax at
evaluation. Unrendered Gaussians are dropped from the packed set rather than
rendered at zero opacity. The discrete choice reaches the router through
straight-through gates on activated opacity that follow the nesting: G2/G3 are
gated by ``y_keep + y_add`` (rendered unless the token reduces), G4/G5 by
``y_add``, and G1, which every choice renders, is never gated.

During training the head also returns every token's keep prefix G1..G3,
independent of the router (``auxiliary_render=dense_keep``). The wrapper renders
that set too and optimizes the mean of the two render losses, so the decoder
sees three Gaussians per token on every step whatever the router picked.

:class:`RoutedCountGaussianHead` is the part V15.2's per-option experts
(``option_experts.py``) share with this head: the count-block contract, the
router and the hard Gumbel-softmax choice.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.experiment_config.definitions import (
    NESTED_AUXILIARY_RENDERS,
    NESTED_COUNT_OPTION_COUNTS,
    SEED_MODE_TOKEN,
)

from .common import _cfg_get
from .gaussian_slots import (
    SLOT_PARAM_NAMES,
    initialize_slot_projection,
    pack_slots,
    slot_param_names,
    slot_param_sizes,
)


class RoutedCountGaussianHead(nn.Module):
    """A Gaussian head whose router picks 1, 3 or 5 Gaussians per token.

    The base owns the ``p2g.grid_query.<COUNT_BLOCK>`` contract, the decoder's
    input LayerNorm ``decoder_norm``, and the router: ``router_norm ->
    Linear(dim, h1) -> SiLU -> ... -> Linear(h_last, 3)`` scores [reduce, keep,
    add] for ``router_hidden = [h1, ...]`` (V15: [256, 128]), behind its own
    LayerNorm, with its last layer at zero so every choice starts equally
    likely. Training samples the choice with hard Gumbel-softmax; evaluation
    takes the argmax.

    A subclass builds the decoder behind ``decoder_norm`` (``_build_decoder``,
    ``_initialize_decoder``, ``_decode``), says which Gaussians each choice
    renders and how the choice gates them (``_routed_set``), and what its
    router-free training-only set is (``_auxiliary_set``, rendered when
    ``auxiliary_render`` is not ``none``, with ``auxiliary_count`` Gaussians
    per token).
    """

    COUNT_BLOCK = ""
    COUNT_LABEL = ""
    OPTION_COUNTS = NESTED_COUNT_OPTION_COUNTS
    AUXILIARY_RENDERS = ("none",)
    PARAM_NAMES = SLOT_PARAM_NAMES
    OPTION_NAMES = ("reduce", "keep", "add")

    def __init__(self, count_cfg, head_cfg, gs_params, dim: int,
                 offset_bound: float):
        super().__init__()
        block = self.COUNT_BLOCK
        settings = _cfg_get(count_cfg, block, None)
        if settings is None:
            raise ValueError(
                f"{self.COUNT_LABEL} needs p2g.grid_query.{block}"
            )
        option_counts = tuple(
            int(value) for value in _cfg_get(settings, "option_counts", ())
        )
        if option_counts != self.OPTION_COUNTS:
            raise ValueError(
                f"p2g.grid_query.{block}.option_counts must be "
                f"{list(self.OPTION_COUNTS)}, got {list(option_counts)}"
            )
        seed_mode = str(_cfg_get(settings, "seed_mode", "")).lower()
        if seed_mode != SEED_MODE_TOKEN:
            raise ValueError(
                f"{self.COUNT_LABEL} decodes around the Utonia token "
                f"anchor; {block}.seed_mode must be {SEED_MODE_TOKEN!r}"
            )
        self.gumbel_tau = float(_cfg_get(settings, "tau", 1.0))
        if not (math.isfinite(self.gumbel_tau) and self.gumbel_tau > 0.0):
            raise ValueError(f"p2g.grid_query.{block}.tau must be positive")
        self.decoder_hidden = int(_cfg_get(settings, "decoder_hidden", 0))
        if self.decoder_hidden <= 0:
            raise ValueError(f"{block}.decoder_hidden must be positive")
        router_hidden = _cfg_get(settings, "router_hidden", None)
        try:
            self.router_hidden = tuple(int(width) for width in router_hidden)
        except TypeError:
            self.router_hidden = ()
        if not self.router_hidden or min(self.router_hidden) <= 0:
            raise ValueError(
                f"{block}.router_hidden must list the router's positive "
                f"hidden widths, got {router_hidden!r}"
            )
        self.auxiliary_render = str(
            _cfg_get(settings, "auxiliary_render", "")
        ).lower()
        if self.auxiliary_render not in self.AUXILIARY_RENDERS:
            raise ValueError(
                f"p2g.grid_query.{block}.auxiliary_render must be one of "
                f"{self.AUXILIARY_RENDERS}, got {self.auxiliary_render!r}"
            )

        self.dim = int(dim)
        self.offset_bound = float(offset_bound)
        self.option_counts = option_counts
        self.keep_count = option_counts[1]
        self.max_count = option_counts[-1]
        if slot_param_names(gs_params) != self.PARAM_NAMES:
            raise ValueError(
                f"{self.COUNT_LABEL} decodes SH slots; a feature slot "
                "(p2g.gs_params.feature) is only available to V18's dense head"
            )
        self.param_sizes = slot_param_sizes(gs_params)
        self.param_dim = sum(self.param_sizes)
        # Geometry of the contract, not learned state.
        self.register_buffer(
            "option_count_values",
            torch.tensor(option_counts, dtype=torch.long),
            persistent=False,
        )

        # Module names, shapes and construction order are each subclass's
        # checkpoint contract.
        self.decoder_norm = nn.LayerNorm(self.dim)
        self._build_decoder()
        self.router_norm = nn.LayerNorm(self.dim)
        router_layers = []
        width = self.dim
        for hidden in self.router_hidden:
            router_layers += [nn.Linear(width, hidden), nn.SiLU()]
            width = hidden
        router_layers.append(self._build_router_output(width, settings))
        self.router = nn.Sequential(*router_layers)
        self._initialize_decoder(head_cfg)
        self._initialize_router_output(self.router[-1])

    def _build_router_output(self, width, settings):
        """The router's last layer: one score per option, ``Linear(width, 3)``."""
        del settings
        return nn.Linear(width, len(self.option_counts))

    def _initialize_router_output(self, output):
        """Zero, so every option starts equally likely."""
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)

    @property
    def auxiliary_count(self) -> int:
        """Gaussians per token in the auxiliary set."""
        raise NotImplementedError

    def _build_decoder(self):
        raise NotImplementedError

    def _initialize_decoder(self, head_cfg):
        raise NotImplementedError

    def _decode(self, feature):
        """``(N, S, P)`` slot parameters for ``(N, dim)`` refined features."""
        raise NotImplementedError

    def _routed_set(self, decoded, anchor, selection, option, count):
        raise NotImplementedError

    def _auxiliary_set(self, decoded, anchor):
        raise NotImplementedError

    def _select(self, logits):
        """Hard one-hot forward; soft categorical derivative in training."""
        if self.training:
            return F.gumbel_softmax(
                logits.float(), tau=self.gumbel_tau, hard=True, dim=-1,
            ).to(dtype=logits.dtype)
        return F.one_hot(
            logits.argmax(dim=-1), num_classes=logits.shape[-1]
        ).to(dtype=logits.dtype)

    def _pack(self, decoded, anchor, token_index, column, slot_index=None):
        """Gather ``decoded[token_index, column]`` into per-Gaussian fields.

        ``slot_index`` numbers each row within what its token renders and
        defaults to ``column``.
        """
        return pack_slots(
            decoded[token_index, column], anchor, token_index,
            column if slot_index is None else slot_index,
            self.param_sizes, self.offset_bound,
        )

    def forward(self, feature, anchor):
        """Return ``(routed, auxiliary, routing)`` for ``(N, dim)`` tokens.

        ``anchor`` holds each token's position in the window reference frame.
        ``routed`` and ``auxiliary`` carry ``token_index``, ``slot_index``, the
        split raw attributes and the Gaussian positions. In training only,
        ``routed`` also carries ``opacity_gate`` for post-sigmoid rendering;
        ``auxiliary`` is ``None`` outside training or with
        ``auxiliary_render=none``.
        ``routing`` carries the logits, the (straight-through) selection, the
        chosen option index and the resulting Gaussian count per token.
        """
        if feature.ndim != 2 or feature.shape[1] != self.dim:
            raise ValueError(
                f"routed Gaussian head expects (N,{self.dim}) features"
            )
        if tuple(anchor.shape) != (feature.shape[0], 3):
            raise ValueError("routed Gaussian anchors must be (N,3)")
        decoded = self._decode(feature)
        logits = self.router(self.router_norm(feature))
        selection = self._select(logits)
        option = selection.argmax(dim=-1)
        count = self.option_count_values[option]

        routed = self._routed_set(decoded, anchor, selection, option, count)
        auxiliary = None
        if self.training and self.auxiliary_render != "none":
            auxiliary = self._auxiliary_set(decoded, anchor)
        return routed, auxiliary, {
            "logits": logits,
            "selection": selection,
            "option": option,
            "count": count,
        }


class NestedCountGaussianHead(RoutedCountGaussianHead):
    """Decode G1..G5 per token and route every token to a nested prefix.

    ``decoder_norm -> Linear(dim, decoder_hidden) -> SiLU ->
    Linear(decoder_hidden, 5 * P)`` decodes all five Gaussians at once, ``P``
    being one Gaussian's raw parameters in ``PARAM_NAMES`` order. Because one
    hidden layer produces the whole set, G4/G5 are placed jointly with G1..G3
    and need no separate head. The router is :class:`RoutedCountGaussianHead`'s.

    The decoder's output layer starts from the dynamic heads' priors
    (``initial_opacity``, ``initial_scale_m``, identity rotation, zero offset)
    and the router's output layer starts at zero, so every choice starts
    equally likely and the expected count is ``(1 + 3 + 5) / 3 = 3``.
    """

    COUNT_BLOCK = "nested_count"
    COUNT_LABEL = "the nested Gaussian count"
    AUXILIARY_RENDERS = NESTED_AUXILIARY_RENDERS

    @property
    def auxiliary_count(self) -> int:
        """``dense_keep`` renders every token's keep prefix G1..G3."""
        return self.keep_count

    def _build_decoder(self):
        self.decoder = nn.Sequential(
            nn.Linear(self.dim, self.decoder_hidden),
            nn.SiLU(),
            nn.Linear(self.decoder_hidden, self.max_count * self.param_dim),
        )

    def _initialize_decoder(self, head_cfg):
        """Every slot gets the dynamic-head priors."""
        initialize_slot_projection(
            self.decoder[-1], head_cfg, self.param_sizes, self.max_count
        )

    def _decode(self, feature):
        return self.decoder(self.decoder_norm(feature)).reshape(
            feature.shape[0], self.max_count, self.param_dim
        )

    def _prefix_set(self, decoded, anchor, count):
        """Each token's first ``count`` slots, ungated.

        Token-major, slots ascending, so every token's first row is its G1.
        """
        slots = torch.arange(self.max_count, device=decoded.device)
        active = slots.unsqueeze(0) < count.unsqueeze(1)
        token_index, slot_index = active.nonzero(as_tuple=True)
        return self._pack(decoded, anchor, token_index, slot_index)

    def _dense_set(self, decoded, anchor, slots):
        """Every token's first ``slots`` slots, ungated, token-major."""
        token_count = decoded.shape[0]
        token_index = torch.arange(
            token_count, device=decoded.device
        ).repeat_interleave(slots)
        slot_index = torch.arange(
            slots, device=decoded.device
        ).repeat(token_count)
        return self._pack(decoded, anchor, token_index, slot_index)

    def _routed_set(self, decoded, anchor, selection, option, count):
        """Each token's selected prefix, with the nested opacity gates.

        Token-major, slots ascending, so every token's first row is its G1.
        """
        del option
        routed = self._prefix_set(decoded, anchor, count)
        token_index = routed["token_index"]
        slot_index = routed["slot_index"]

        # A packed G2/G3 belongs to a token that chose keep or add, and a
        # packed G4/G5 to one that chose add, so each gate's forward value is
        # one. Keep raw opacity unchanged and carry the gate separately to the
        # renderer, which multiplies it after sigmoid. This avoids a logit
        # round-trip and its clamp: even when sigmoid rounds to one, the soft
        # selection still receives o * dL/do'. G1 carries no router gradient.
        gate_keep_or_add = selection[:, 1] + selection[:, 2]
        gate_add = selection[:, 2]
        gate = torch.where(
            slot_index >= self.keep_count,
            gate_add[token_index],
            gate_keep_or_add[token_index],
        )
        if self.training:
            # Every packed gate is selected. Subtract first so the forward is
            # bit-exactly one, while keeping the nested selection's derivative.
            selected_gate = torch.ones_like(gate) + (gate - gate.detach())
            routed["opacity_gate"] = torch.where(
                slot_index > 0, selected_gate, torch.ones_like(gate),
            ).unsqueeze(-1)
        return routed

    def _auxiliary_set(self, decoded, anchor):
        """Every token's G1..G3, ungated: the router plays no part in it."""
        return self._dense_set(decoded, anchor, self.keep_count)


__all__ = ["NestedCountGaussianHead", "RoutedCountGaussianHead"]
