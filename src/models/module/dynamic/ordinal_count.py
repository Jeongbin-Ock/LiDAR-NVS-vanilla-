"""V16's ordinal nested Gaussian count: two bounded gates, masked rasterization.

V16 keeps V15's G1..G5 decoder and its nested choices -- reduce ``{G1}``, keep
``{G1, G2, G3}``, add ``{G1, ..., G5}`` -- and changes how the choice is scored,
sampled and trained.

* **Two ordinal gates instead of three option scores.** The keep gate
  ``u_keep`` decides whether G2/G3 are present; the add gate ``u_add`` decides
  whether G4/G5 are present given G2/G3. Each is a cosine logit at a fixed
  scale, ``u = s * cos(w, h)`` with no bias, so ``|u| <= s`` however the router
  weights grow (:class:`CosineGateLayer`).
* **The option probabilities follow from the gates** as a continuation-ratio
  (sequential) model. With ``q = sigmoid(u)``::

      p(reduce) = 1 - q_keep
      p(keep)   = q_keep * (1 - q_add)
      p(add)    = q_keep * q_add

  ``q_keep`` is the probability that G2/G3 are present, ``q_add`` the
  probability that G4/G5 are present given that G2/G3 are.
* **Training samples each gate** with logistic noise, the two-class
  Gumbel-max: ``u + l`` with ``l ~ Logistic(0, 1)`` is positive with
  probability ``q``. The forward is that hard 0/1 sample; the backward is the
  straight-through derivative of ``sigmoid((u + l) / tau)``. There is no
  three-way softmax. Evaluation takes the most probable option.
* **Masked rasterization instead of an opacity gate.** Training renders all
  five Gaussians of every token with a per-Gaussian blend mask -- G1 = 1,
  G2/G3 = M_keep, G4/G5 = M_keep * M_add -- and the rasterizer blends a
  Gaussian with ``mask * alpha`` while culling and alpha-cutting it on its
  unmasked alpha. A Gaussian at mask 0 is absent from every output, yet the
  backward pass still gives ``dL/dmask``: the first-order change of the loss
  from inserting it, at its current shape. A reduce token's gates therefore
  learn from inserting G2/G3 (and, when the add gate was sampled open, G2..G5),
  a keep token's from removing G2/G3 and inserting G4/G5, an add token's from
  removing them. Absent Gaussians get no gradient on their own attributes.
* **The router-free second render** (``auxiliary_render: dense_add``) is every
  token's G1..G5, unmasked, so the absent Gaussians keep training whatever the
  router chose.

Evaluation renders only the chosen prefix, without masks; for hard 0/1 masks
this is exactly the training render of the same choice.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.experiment_config.definitions import ORDINAL_AUXILIARY_RENDERS

from .common import _cfg_get
from .nested_count import NestedCountGaussianHead


class CosineGateLayer(nn.Module):
    """``u_k = scale * cos(w_k, h)`` for every gate ``k``: bounded logits.

    No bias, and ``scale`` is a fixed constant rather than a parameter, so every
    logit lies in ``[-scale, scale]`` whatever the hidden features or the
    weights grow to.
    """

    def __init__(self, in_features: int, gates: int, scale: float):
        super().__init__()
        self.in_features = int(in_features)
        self.gates = int(gates)
        self.scale = float(scale)
        self.weight = nn.Parameter(torch.empty(self.gates, self.in_features))
        self.reset_parameters()

    def reset_parameters(self):
        # Only each row's direction matters. Unit-norm rows (std 1/sqrt(in))
        # keep the angular change of one Adam step near lr * sqrt(in).
        nn.init.normal_(self.weight, mean=0.0, std=self.in_features ** -0.5)

    def forward(self, hidden):
        return self.scale * F.linear(
            F.normalize(hidden, dim=-1), F.normalize(self.weight, dim=-1)
        )

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, gates={self.gates}, "
            f"scale={self.scale}"
        )


def option_log_probabilities(gate_logits):
    """``log p`` of reduce / keep / add, ``(N, 3)``, from ``(N, 2)`` gates."""
    keep, add = gate_logits.unbind(dim=-1)
    log_keep_open = -F.softplus(-keep)
    return torch.stack([
        -F.softplus(keep),
        log_keep_open - F.softplus(add),
        log_keep_open - F.softplus(-add),
    ], dim=-1)


class OrdinalCountGaussianHead(NestedCountGaussianHead):
    """V15's G1..G5 decoder under two bounded, ordinal router gates.

    The decoder is :class:`NestedCountGaussianHead`'s, module for module. The
    router is the routed heads' ``router_norm -> Linear -> SiLU -> ...`` stack
    ending in a :class:`CosineGateLayer` with two gates, keep and add, at the
    fixed scale ``router_scale``. Its gate directions start random, so both
    gates start near 1/2 and the option probabilities near (1/2, 1/4, 1/4).

    ``forward`` keeps the routed heads' ``(routed, auxiliary, routing)``
    contract. In training ``routed`` is every token's G1..G5 with
    ``blend_mask`` and ``auxiliary`` the same Gaussians without it; in
    evaluation ``routed`` is each token's most probable prefix and there is no
    auxiliary set. ``routing["logits"]`` holds the option log-probabilities, so
    the routing statistics read them as they read V15's scores: their softmax
    is ``p`` and their argmax the evaluation choice.
    """

    COUNT_BLOCK = "ordinal_count"
    COUNT_LABEL = "the ordinal nested Gaussian count"
    AUXILIARY_RENDERS = ORDINAL_AUXILIARY_RENDERS
    GATE_NAMES = ("keep", "add")

    @property
    def auxiliary_count(self) -> int:
        """``dense_add`` renders every token's G1..G5."""
        return self.max_count

    def _build_router_output(self, width, settings):
        scale = float(_cfg_get(settings, "router_scale", 0.0))
        if not (math.isfinite(scale) and scale > 0.0):
            raise ValueError(
                f"p2g.grid_query.{self.COUNT_BLOCK}.router_scale must be a "
                f"positive number, got {scale!r}"
            )
        self.router_scale = scale
        return CosineGateLayer(width, len(self.GATE_NAMES), scale)

    def _initialize_router_output(self, output):
        """Keep the random gate directions ``CosineGateLayer`` starts from."""
        del output

    def _sample_gates(self, gate_logits):
        """Hard Bernoulli(sigmoid(u)) gates with a straight-through backward.

        Returns the straight-through gates, whose forward is exactly the hard
        0/1 sample, and the boolean sample itself.
        """
        uniform = torch.rand_like(gate_logits).clamp_min_(
            torch.finfo(gate_logits.dtype).tiny
        )
        noise = torch.log(uniform) - torch.log1p(-uniform)
        noisy = (gate_logits + noise) / self.gumbel_tau
        sampled = noisy > 0
        soft = torch.sigmoid(noisy)
        # Subtract first so the forward is bit-exactly the hard sample.
        return sampled.to(soft.dtype) + (soft - soft.detach()), sampled

    def _blend_mask(self, gates, token_index, slot_index):
        """Each packed Gaussian's blend mask under the nested prefixes, (M, 1).

        G1 renders whatever the token chose and carries no gate. G2/G3 carry
        the keep gate and G4/G5 the product of both, so a closed keep gate
        removes G4/G5 too, and the keep gate's gradient includes the G4/G5
        term whenever the add gate was sampled open.
        """
        keep = gates[:, 0]
        add = keep * gates[:, 1]
        mask = torch.where(
            slot_index >= self.keep_count, add[token_index], keep[token_index]
        )
        mask = torch.where(slot_index > 0, mask, torch.ones_like(mask))
        return mask.unsqueeze(-1)

    def forward(self, feature, anchor):
        """Return ``(routed, auxiliary, routing)`` for ``(N, dim)`` tokens.

        ``routing`` carries the option log-probabilities (``logits``), the two
        gate logits (``gate_logits``), the sampled (training) or most probable
        (evaluation) option per token, and that option's Gaussian count.
        """
        if feature.ndim != 2 or feature.shape[1] != self.dim:
            raise ValueError(
                f"routed Gaussian head expects (N,{self.dim}) features"
            )
        if tuple(anchor.shape) != (feature.shape[0], 3):
            raise ValueError("routed Gaussian anchors must be (N,3)")
        decoded = self._decode(feature)
        gate_logits = self.router(self.router_norm(feature)).float()
        log_probability = option_log_probabilities(gate_logits)

        auxiliary = None
        if self.training:
            gates, sampled = self._sample_gates(gate_logits)
            option = sampled[:, 0].long() * (1 + sampled[:, 1].long())
            routed = self._dense_set(decoded, anchor, self.max_count)
            if self.auxiliary_render != "none":
                auxiliary = {**routed, "raw": dict(routed["raw"])}
            routed["blend_mask"] = self._blend_mask(
                gates, routed["token_index"], routed["slot_index"]
            )
            count = self.option_count_values[option]
        else:
            option = log_probability.argmax(dim=-1)
            count = self.option_count_values[option]
            routed = self._prefix_set(decoded, anchor, count)
        return routed, auxiliary, {
            "logits": log_probability,
            "gate_logits": gate_logits,
            "option": option,
            "count": count,
        }


__all__ = [
    "CosineGateLayer",
    "OrdinalCountGaussianHead",
    "option_log_probabilities",
]
