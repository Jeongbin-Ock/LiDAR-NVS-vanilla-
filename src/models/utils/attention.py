import torch
import torch.nn as nn


class Rotary3D(nn.Module):
    """Utonia-compatible 3D RoPE with an explicit metric position scale.

    This uses the exact channel layout and frequency schedule of
    :class:`utonia.model.Point3DRoPE`: split one attention head into equal
    x/y/z chunks, split each axis chunk into two halves, and rotate matching
    half-channels with ``base ** (-2j / axis_dim)``. Therefore ``head_dim``
    must be divisible by six.

    ``position_scale`` is the only attention-specific geometry knob. For
    metric input positions, axis frequency ``j`` has wavelength
    ``2*pi / (position_scale * inv_freq[j])`` metres. Changing it preserves
    Utonia's RoPE operator while adapting its physical wavelengths to the
    metric scale the attention needs to resolve.
    """

    def __init__(self, head_dim, base=10.0, position_scale=1.0):
        super().__init__()
        head_dim = int(head_dim)
        if head_dim % 6 != 0:
            raise ValueError(
                f"head_dim must be divisible by 6 for Utonia 3D RoPE, got {head_dim}"
            )
        base = float(base)
        position_scale = float(position_scale)
        if base <= 0.0:
            raise ValueError("base must be positive")
        if position_scale <= 0.0:
            raise ValueError("position_scale must be positive")
        self.head_dim = head_dim
        self.chunk_dim = head_dim // 3
        self.base = base
        self.position_scale = position_scale
        inv_freq = 1.0 / (
            base ** (
                torch.arange(0, self.chunk_dim, 2, dtype=torch.float32)
                / self.chunk_dim
            )
        )
        # This is deterministic geometry derived from config, not learned state.
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def angles(self, pos, position_scale=None):
        """``(..., 3)`` positions -> ``(..., 3, axis_dim/2)`` fp32 angles.

        ``position_scale`` may be a scalar or match a prefix of ``pos``'s
        leading dimensions.  The latter lets heterogeneous anchor groups share
        Q/K/V weights while using coordinate-frame-specific wavelengths.
        """
        if pos.shape[-1] != 3:
            raise ValueError(f"RoPE positions must end in xyz, got {tuple(pos.shape)}")
        scale = torch.as_tensor(
            self.position_scale if position_scale is None else position_scale,
            device=pos.device, dtype=torch.float32,
        )
        leading_ndim = pos.ndim - 1
        if scale.ndim > leading_ndim:
            raise ValueError(
                "position_scale has too many dimensions for positions: "
                f"scale={tuple(scale.shape)}, pos={tuple(pos.shape)}"
            )
        scale = scale.reshape(
            *scale.shape, *([1] * (leading_ndim - scale.ndim))
        )
        return pos.float().mul(scale.unsqueeze(-1)).unsqueeze(-1) * self.inv_freq

    def rotate(self, x, angles):
        """Apply Utonia's per-axis half rotation to ``x``.

        ``angles`` must broadcast to ``x`` reshaped as
        ``(..., 3, chunk_dim/2)``; callers insert singleton head/query dims.
        """
        if x.shape[-1] != self.head_dim:
            raise ValueError(
                f"RoPE feature dim must be {self.head_dim}, got {x.shape[-1]}"
            )
        x_axis = x.reshape(*x.shape[:-1], 3, self.chunk_dim)
        half = self.chunk_dim // 2
        x1 = x_axis[..., :half]
        x2 = x_axis[..., half:]
        cos = torch.cos(angles)
        sin = torch.sin(angles)
        rotated = torch.cat(
            [x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1
        )
        return rotated.flatten(-2)
