"""V15.2's Gaussian count: one Gaussian expert per router option.

V15's router chooses reduce / keep / add for every token, but V15.2 renders that
option's own expert instead of a prefix of one shared G1..G5 set:

* reduce renders E1, one Gaussian,
* keep renders E3, three Gaussians,
* add renders E5, five Gaussians.

The three experts are linear readouts of one shared trunk, ``LN(f) ->
Linear(dim, decoder_hidden) -> SiLU``, so the second layer predicts
``1 + 3 + 5`` Gaussians in total, all around the token anchor.

Every Gaussian of the chosen expert carries that option's straight-through gate
(forward exactly one), applied to activated opacity by the renderer. Every token
therefore sends its router a gradient, reduce tokens included; in V15 a reducing
token renders only G1, which no gate touches. The gradient is the first-order
value of scaling the chosen expert's opacity, ``sum_g o_g dL/do'_g``, pushed
through the softmax onto all three logits; the experts a token did not choose
are not rendered and are not compared.

During training the head also returns every token's E5, independent of the
router (``auxiliary_render=dense_add``). The wrapper renders that set too and
optimizes the mean of the two render losses, so the add expert and the shared
trunk are supervised on every token at every step.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.experiment_config.definitions import (
    EXPERT_AUXILIARY_RENDERS,
    EXPERT_COUNT_OPTION_COUNTS,
)

from .gaussian_slots import initialize_slot_projection
from .nested_count import RoutedCountGaussianHead


class OptionExpertGaussianHead(RoutedCountGaussianHead):
    """Route every token to one of three Gaussian experts on a shared trunk.

    ``decoder_norm -> trunk = Linear(dim, decoder_hidden) -> SiLU`` is shared,
    and ``experts[i] = Linear(decoder_hidden, option_counts[i] * P)`` reads it
    for option ``i``: E1, E3 and E5 for reduce, keep and add. The decoded slots
    of one token are laid out expert by expert, E1 | E3 | E5, nine in all. Each
    expert's output layer starts from the dynamic heads' priors, so all nine
    Gaussians start on the anchor; the router is
    :class:`RoutedCountGaussianHead`'s, starting uniform, so the expected count
    is again ``(1 + 3 + 5) / 3 = 3``.

    A token's rendered rows are its chosen expert's slots in order, so
    ``slot_index`` counts within the expert and every token's first rendered
    row has ``slot_index == 0``.
    """

    COUNT_BLOCK = "expert_count"
    COUNT_LABEL = "the per-option expert count"
    OPTION_COUNTS = EXPERT_COUNT_OPTION_COUNTS
    AUXILIARY_RENDERS = EXPERT_AUXILIARY_RENDERS

    @property
    def auxiliary_count(self) -> int:
        """``dense_add`` renders every token's add expert, E5."""
        return self.max_count

    def _build_decoder(self):
        self.trunk = nn.Sequential(
            nn.Linear(self.dim, self.decoder_hidden),
            nn.SiLU(),
        )
        self.experts = nn.ModuleList([
            nn.Linear(self.decoder_hidden, count * self.param_dim)
            for count in self.option_counts
        ])
        starts = [0]
        for count in self.option_counts[:-1]:
            starts.append(starts[-1] + count)
        # First decoded slot of each option's expert in the E1 | E3 | E5 layout.
        self.register_buffer(
            "expert_slot_start",
            torch.tensor(starts, dtype=torch.long),
            persistent=False,
        )
        self.slot_count = sum(self.option_counts)

    def _initialize_decoder(self, head_cfg):
        """Every slot of every expert gets the dynamic-head priors."""
        for expert, count in zip(self.experts, self.option_counts):
            initialize_slot_projection(
                expert, head_cfg, self.param_sizes, count
            )

    def _decode(self, feature):
        """All three experts for every token, ``(N, 1 + 3 + 5, P)``."""
        hidden = self.trunk(self.decoder_norm(feature))
        return torch.cat(
            [expert(hidden) for expert in self.experts], dim=-1
        ).reshape(feature.shape[0], self.slot_count, self.param_dim)

    def _routed_set(self, decoded, anchor, selection, option, count):
        """Each token's chosen expert, every Gaussian of it gated.

        Token-major, the expert's slots ascending, so every token's first row
        is slot 0 of its expert.
        """
        start = self.expert_slot_start[option]
        columns = torch.arange(self.slot_count, device=decoded.device)
        active = (columns.unsqueeze(0) >= start.unsqueeze(1)) & (
            columns.unsqueeze(0) < (start + count).unsqueeze(1)
        )
        token_index, column = active.nonzero(as_tuple=True)
        routed = self._pack(
            decoded, anchor, token_index, column,
            slot_index=column - start[token_index],
        )
        if self.training:
            # y_ST of the chosen option, one per token. Its forward is one,
            # and since the soft probability depends on every logit, the gate
            # reaches all three logits through the softmax. Subtract first so
            # the forward is bit-exactly one; the renderer multiplies it onto
            # activated opacity, so a saturated opacity keeps the gradient.
            chosen = selection.gather(1, option.unsqueeze(-1)).squeeze(-1)
            gate = chosen[token_index]
            routed["opacity_gate"] = (
                torch.ones_like(gate) + (gate - gate.detach())
            ).unsqueeze(-1)
        return routed

    def _auxiliary_set(self, decoded, anchor):
        """Every token's E5, ungated: the router plays no part in it."""
        token_count = decoded.shape[0]
        slots = self.max_count
        token_index = torch.arange(
            token_count, device=decoded.device
        ).repeat_interleave(slots)
        slot_index = torch.arange(slots, device=decoded.device).repeat(
            token_count
        )
        return self._pack(
            decoded, anchor, token_index,
            self.expert_slot_start[-1] + slot_index, slot_index=slot_index,
        )


__all__ = ["OptionExpertGaussianHead"]
