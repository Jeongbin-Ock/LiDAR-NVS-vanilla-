"""Straight-through helpers shared by the Gaussian heads."""
from __future__ import annotations

import torch


def gradient_scale_identity(x, scale):
    """Preserve forward values while scaling each row's upstream gradient."""
    scale = torch.as_tensor(scale, device=x.device, dtype=x.dtype)
    while scale.ndim < x.ndim:
        scale = scale.unsqueeze(-1)
    detached = x.detach()
    return detached + scale * (x - detached)


def opacity_gate_st(raw_opacity, gate):
    """Apply a hard-ST gate in activated-opacity space.

    The rasterizer applies ``sigmoid`` to raw opacity logits.  A selected hard
    gate therefore has forward value one, while its soft backward derivative
    must be applied after that sigmoid.
    """
    if raw_opacity.shape[0] == 0:
        return raw_opacity
    work_opacity = raw_opacity.float()
    work_gate = gate.to(device=raw_opacity.device, dtype=torch.float32)
    while work_gate.ndim < work_opacity.ndim:
        work_gate = work_gate.unsqueeze(-1)
    activated = torch.sigmoid(work_opacity)
    gated_activated = (activated * work_gate).clamp(
        min=1.0e-6, max=1.0 - 1.0e-6
    )
    gated_logit = torch.logit(gated_activated).to(raw_opacity.dtype)
    return raw_opacity.detach() + gated_logit - gated_logit.detach()


__all__ = [
    "gradient_scale_identity",
    "opacity_gate_st",
]
