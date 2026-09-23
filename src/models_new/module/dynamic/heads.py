"""Gaussian attribute and velocity prediction heads."""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from .common import _cfg_get

class GaussianAttributeHead(nn.Module):
    """Time-invariant 2DGS decoder driven by the shared token feature.

    ``gaussians_per_token`` slots share the token's feature but nothing else:
    each owns its own rows of every attribute projection, including the bounded
    position offset, so slots seeded at the same coordinate still learn
    independent geometry.

    ``seed_conditioned`` additionally concatenates every slot's seed delta to
    the normalized feature, widening the head's input by
    ``3 * gaussians_per_token``. This is what the adaptive-K router's trunk
    does with its own ``3 * K_max`` channels, so a fixed-count head can carry
    the same geometric conditioning without the router.

    ``trunk`` puts one ``Linear(in, dim) -> SiLU`` in front of the attribute
    projections, which is the other half of what the router's decoder does and
    the only nonlinearity either version has. Without it the whole head is a
    single affine map of the normalized feature.

    Both default to off, the plain affine form V11.1 checkpoints were saved
    from; V11.3/V11.5/V11.6/V13.1 turn both on.
    """

    def __init__(self, cfg, gs_params, dim: int, offset_bound: float,
                 gaussians_per_token: int = 1, seed_conditioned: bool = False,
                 trunk: bool = False):
        super().__init__()
        self.dim = int(dim)
        self.offset_bound = float(offset_bound)
        self.gaussians_per_token = int(gaussians_per_token)
        if self.gaussians_per_token < 1:
            raise ValueError("gaussians_per_token must be at least one")
        self.seed_conditioned = bool(seed_conditioned)
        self.seed_dim = (
            3 * self.gaussians_per_token if self.seed_conditioned else 0
        )
        self.output_norm = nn.LayerNorm(self.dim)
        # A trunk absorbs the conditioned input and hands every attribute the
        # same dim-wide hidden activation, exactly as the router's does.
        self.trunk = (
            nn.Sequential(nn.Linear(self.dim + self.seed_dim, self.dim), nn.SiLU())
            if bool(trunk) else None
        )
        head_in = self.dim if self.trunk is not None else self.dim + self.seed_dim

        self.sizes = {
            name: int(getattr(gs_params, name))
            for name in ("shs", "opacity", "scaling", "rotation")
        }
        self.sizes["offset"] = int(getattr(gs_params, "offset", 0) or 0)
        if self.sizes["offset"] != 3:
            raise ValueError("Dynamic 2DGS requires p2g.gs_params.offset=3")
        self.heads = nn.ModuleDict({
            name: nn.Linear(head_in, width * self.gaussians_per_token)
            for name, width in self.sizes.items()
        })
        self._initialize_outputs(cfg)

    def _initialize_outputs(self, cfg):
        for head in self.heads.values():
            nn.init.normal_(head.weight, mean=0.0, std=0.01)
            nn.init.zeros_(head.bias)
        opacity = float(_cfg_get(cfg, "initial_opacity", 0.2))
        opacity = min(max(opacity, 1.0e-4), 1.0 - 1.0e-4)
        nn.init.constant_(
            self.heads["opacity"].bias,
            math.log(opacity / (1.0 - opacity)),
        )
        scale = float(_cfg_get(cfg, "initial_scale_m", 0.3))
        raw_scale = math.log(math.expm1(max(scale, 1.0e-4)))
        nn.init.constant_(self.heads["scaling"].bias, raw_scale)
        if self.sizes["rotation"] != 4:
            raise ValueError("Dynamic 2DGS requires a 4D quaternion rotation")
        with torch.no_grad():
            # Every slot starts at the identity quaternion.
            self.heads["rotation"].bias[
                ::self.sizes["rotation"]
            ] = 1.0
        nn.init.zeros_(self.heads["offset"].weight)
        nn.init.zeros_(self.heads["offset"].bias)

    def _unpack(self, value, width):
        """(N, G*width) -> (N*G, width), token-major so slot order is stable."""
        if self.gaussians_per_token == 1:
            return value
        return value.reshape(value.shape[0] * self.gaussians_per_token, width)

    def _head_input(self, decoded, seed_delta):
        """Normalized feature, with every slot's raw metric seed delta appended.

        The deltas stay in metres rather than being normalized, which is how
        the router trunk consumes its own copy of the same quantity.
        """
        if not self.seed_conditioned:
            return decoded
        if seed_delta is None:
            raise ValueError(
                "a seed-conditioned Gaussian head requires seed deltas"
            )
        expected = (decoded.shape[0], self.gaussians_per_token, 3)
        if tuple(seed_delta.shape) != expected:
            raise ValueError(
                f"seed deltas must have shape {expected}, got "
                f"{tuple(seed_delta.shape)}"
            )
        return torch.cat([
            decoded,
            seed_delta.to(dtype=decoded.dtype).reshape(
                decoded.shape[0], self.seed_dim
            ),
        ], dim=-1)

    def _decode(self, feature, seed_delta=None):
        decoded = self.output_norm(feature)
        head_input = self._head_input(decoded, seed_delta)
        if self.trunk is not None:
            head_input = self.trunk(head_input)
        raw = {
            name: self._unpack(head(head_input), self.sizes[name])
            for name, head in self.heads.items()
        }
        offset = self.offset_bound * torch.tanh(raw.pop("offset"))
        # The third value stays the normalized feature, not the conditioned
        # input, so downstream consumers keep reading a dim-wide token.
        return raw, offset, decoded

    def forward(self, feature, seed_delta=None):
        return self._decode(feature, seed_delta)

class EmbeddedInitVelocityHead(nn.Module):
    """Velocity offset from the refined token and a detached ``v_init``.

    ``[LN(f'), MLP(v_init; 3->32->32)] -> 96 -> 3`` at the shipped widths.
    Head 0's attended feature channels never carry the metric coordinate
    expectation itself, so the initializer is embedded explicitly or the head
    could not see what it is correcting. The conditioning copy is detached; the
    additive ``v_init + v_offset`` path outside this module remains the only
    offset-loss route into matching.
    """

    def __init__(self, cfg, dim: int):
        super().__init__()
        self.dim = int(dim)
        self.velocity_embedding_dim = int(
            _cfg_get(cfg, "velocity_embedding_dim", 32)
        )
        self.residual_hidden_dim = int(
            _cfg_get(cfg, "residual_hidden_dim", 96)
        )
        self.detach_init_condition = bool(
            _cfg_get(cfg, "detach_init_condition", True)
        )
        if self.velocity_embedding_dim <= 0:
            raise ValueError("motion.velocity_embedding_dim must be positive")
        if self.residual_hidden_dim <= 0:
            raise ValueError("motion.residual_hidden_dim must be positive")

        self.feature_norm = nn.LayerNorm(self.dim)
        self.velocity_encoder = nn.Sequential(
            nn.Linear(3, self.velocity_embedding_dim),
            nn.SiLU(),
            nn.Linear(
                self.velocity_embedding_dim,
                self.velocity_embedding_dim,
            ),
        )
        self.motion_refiner = nn.Sequential(
            nn.Linear(
                self.dim + self.velocity_embedding_dim,
                self.residual_hidden_dim,
            ),
            nn.SiLU(),
        )
        self.velocity = nn.Linear(self.residual_hidden_dim, 3)
        if bool(_cfg_get(cfg, "zero_init", True)):
            nn.init.zeros_(self.velocity.weight)
            nn.init.zeros_(self.velocity.bias)

    def forward(self, refined_feature, velocity_init):
        if refined_feature.ndim != 2 or refined_feature.shape[1] != self.dim:
            raise ValueError(
                f"refined_feature must be (N,{self.dim}) for the velocity offset"
            )
        if velocity_init.shape != (refined_feature.shape[0], 3):
            raise ValueError("velocity_init must be (N,3) and align with feature")
        condition = (
            velocity_init.detach()
            if self.detach_init_condition
            else velocity_init
        )
        feature = self.feature_norm(refined_feature)
        velocity_embedding = self.velocity_encoder(
            condition.to(dtype=feature.dtype)
        )
        motion_feature = self.motion_refiner(torch.cat([
            feature,
            velocity_embedding,
        ], dim=-1))
        return self.velocity(motion_feature)

__all__ = [
    "GaussianAttributeHead",
    "EmbeddedInitVelocityHead",
]
