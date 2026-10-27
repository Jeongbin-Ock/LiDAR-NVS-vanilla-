"""V18.2's LiDAR decoder: a LiDAR4D-style U-Net over the rendered maps.

V18 and V18.1 decode every pixel on its own with an MLP. V18.2 decodes the
whole range image with the U-Net that LiDAR4D (Zheng et al., CVPR 2024) trains
for raydrop refinement (``m4_raydrop_refine.UNet``), reading per pixel

    x = [f' (feature_dim), d (3), r / R (1)],

where ``f'`` is the alpha-blended Gaussian feature, ``d`` the unit ray through
the pixel in the rendering LiDAR's own panorama axes (x right, y down, z
forward, for nuScenes and KITTI-360 alike; the same vector for a pixel in
every view of a sensor), ``r`` the rendered median depth, read detached as in
V18, and ``R = range_scale_m``. The U-Net predicts the hit surface's
``(i_s, p_s) = sigmoid(U(x))``, and each pixel composites that surface over the
no-return background by its accumulated alpha ``a`` exactly as V18's decoder:
``intensity = a * i_s``, ``raydrop = 1 - a * (1 - p_s)``. Alpha is not an input.

``channels`` lists the encoder widths, ``[64, 128, 256, 256]`` for V18.2:
LiDAR4D's InConv, one 1x1 conv with no norm or activation, maps x to
``channels[0]`` at full resolution; each further width is a Down block (2x2
max-pool, then twice norm -> ReLU -> Dropout2d(0.1) -> 3x3 conv), so V18.2's
self-attention runs at 1/8 resolution (KITTI-360 66x1030: 8x128 positions,
nuScenes 32x1085: 4x135). Up blocks (bilinear x2, padded to the skip,
concatenated with it, two convs) return to ``channels[0]`` at full resolution,
and LiDAR4D's OutConv (norm -> ReLU -> 1x1 conv) gives the two logits. Each Up
block outputs the width of the skip it meets next, as in LiDAR4D.

Two departures from LiDAR4D's network, neither changing a parameter count:

- GroupNorm(``norm_groups``) replaces every BatchNorm. A training step decodes
  the target views of one or two windows, consecutive sweeps of the same
  scenes, so BatchNorm's batch statistics would describe one or two scenes and
  differ from the running averages it evaluates with. GroupNorm normalizes
  each view by itself, identically in training and evaluation.
- The self-attention merges its heads back per position. LiDAR4D's
  ``h.view(B, H, W, C)`` of the ``[B, heads, HW, C / heads]`` attention output
  does not undo the head split: each position receives one head's outputs for
  ``heads`` consecutive query positions. Here a position gets its own heads'
  outputs, concatenated in the order the channels were split.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.nn import functional as F

# Rendered channels besides the feature: the sensor-frame ray direction (3)
# and the normalized median depth (1).
RAY_CHANNELS = 3
RANGE_CHANNELS = 1
DECODER_OUTPUTS = ("intensity", "raydrop")
# LiDAR4D's U-Net: 8 attention heads, 0.1 channel dropout before every conv
# and 0.1 attention dropout.
ATTENTION_HEADS = 8
DROPOUT = 0.1


class DoubleConv(nn.Module):
    """LiDAR4D's (norm -> ReLU -> Dropout2d -> 3x3 conv) * 2, with GroupNorm."""

    def __init__(self, in_channels, out_channels, mid_channels, groups):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.GroupNorm(groups, in_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(DROPOUT),
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, mid_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(DROPOUT),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """2x2 max-pool, then a DoubleConv to ``out_channels``."""

    def __init__(self, in_channels, out_channels, groups):
        super().__init__()
        self.down = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_channels, out_channels, out_channels, groups)

    def forward(self, x):
        return self.conv(self.down(x))


class Up(nn.Module):
    """Bilinear x2, padded to the skip's size, concatenated after it, DoubleConv."""

    def __init__(self, in_channels, out_channels, groups):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels, groups)

    def forward(self, x, skip):
        x = self.up(x)
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)
        x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        return self.conv(torch.cat([skip, x], dim=1))


class SelfAttention(nn.Module):
    """LiDAR4D's AttnBlock with each position's heads merged back in place.

    Pre-norm, q/k/v from one bias-free 1x1 conv, ``ATTENTION_HEADS`` heads of
    ``C / heads`` channels attending over every position, a bias-free 1x1
    output projection and a residual add. In training each attention logit is
    pushed to -1e12 with probability ``DROPOUT``, as LiDAR4D does. There is no
    positional encoding.
    """

    def __init__(self, channels, groups):
        super().__init__()
        if channels % ATTENTION_HEADS:
            raise ValueError(
                f"self-attention needs channels divisible by {ATTENTION_HEADS}, got {channels}"
            )
        self.norm = nn.GroupNorm(groups, channels)
        self.proj_qkv = nn.Conv2d(channels, channels * 3, 1, bias=False)
        self.proj = nn.Conv2d(channels, channels, 1, bias=False)

    def forward(self, x):
        batch, channels, height, width = x.shape
        q, k, v = torch.chunk(self.proj_qkv(self.norm(x)), 3, dim=1)
        # Channels split into heads in order: head h owns channels
        # [h * C/heads, (h + 1) * C/heads).
        q = q.reshape(batch, ATTENTION_HEADS, -1, height * width).transpose(2, 3)
        k = k.reshape(batch, ATTENTION_HEADS, -1, height * width)
        v = v.reshape(batch, ATTENTION_HEADS, -1, height * width).transpose(2, 3)
        logits = torch.matmul(q, k) * (channels // ATTENTION_HEADS) ** -0.5
        if self.training:
            logits = logits + torch.bernoulli(torch.full_like(logits, DROPOUT)) * -1e12
        attended = torch.matmul(F.softmax(logits, dim=-1), v)
        # [B, heads, HW, C/heads] -> [B, C, H, W]: position p keeps its own heads.
        attended = attended.transpose(2, 3).reshape(batch, channels, height, width)
        return x + self.proj(attended)


class OutConv(nn.Module):
    """LiDAR4D's head: norm -> ReLU -> 1x1 conv."""

    def __init__(self, in_channels, out_channels, groups):
        super().__init__()
        self.conv = nn.Sequential(
            nn.GroupNorm(groups, in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
        )

    def forward(self, x):
        return self.conv(x)


def _positive_int(cfg, name):
    value = getattr(cfg, name, None)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(
            f"g2p.lidar_decoder.{name} must be a positive integer, got {value!r}"
        )
    return int(value)


class LidarUNetDecoder(nn.Module):
    """``[f', d, r / R] -> (intensity, raydrop)`` for a whole view at once.

    ``g2p.lidar_decoder`` sets ``feature_dim``, ``range_scale_m`` (R),
    ``channels`` (the encoder widths, at least two) and ``norm_groups`` (the
    GroupNorm group count, dividing every width). V18.2: 32 + 3 + 1 = 36 inputs,
    channels [64, 128, 256, 256], 32 groups, 6,464,194 parameters.
    """

    # The frame of the ray directions G2P hands this decoder: the rendering
    # LiDAR's own panorama axes (V18's MLP reads the window reference frame's).
    ray_frame = "sensor"

    def __init__(self, cfg):
        super().__init__()
        self.feature_dim = _positive_int(cfg, "feature_dim")
        groups = _positive_int(cfg, "norm_groups")
        channels = getattr(cfg, "channels", None)
        try:
            self.channels = tuple(int(width) for width in channels)
        except TypeError:
            self.channels = ()
        if len(self.channels) < 2 or min(self.channels) <= 0:
            raise ValueError(
                "g2p.lidar_decoder.channels must list at least two positive "
                f"encoder widths, got {channels!r}"
            )
        if any(width % groups for width in self.channels):
            raise ValueError(
                f"g2p.lidar_decoder.norm_groups={groups} must divide every width "
                f"in channels {list(self.channels)}"
            )
        self.range_scale_m = float(getattr(cfg, "range_scale_m", 0.0))
        if not (math.isfinite(self.range_scale_m) and self.range_scale_m > 0.0):
            raise ValueError("g2p.lidar_decoder.range_scale_m must be positive")

        self.input_dim = self.feature_dim + RAY_CHANNELS + RANGE_CHANNELS
        self.inc = nn.Conv2d(self.input_dim, self.channels[0], kernel_size=1)
        self.down = nn.ModuleList(
            Down(self.channels[level - 1], self.channels[level], groups)
            for level in range(1, len(self.channels))
        )
        self.attention = SelfAttention(self.channels[-1], groups)
        ups, width = [], self.channels[-1]
        for level in reversed(range(len(self.channels) - 1)):
            out = self.channels[max(level - 1, 0)]
            ups.append(Up(width + self.channels[level], out, groups))
            width = out
        self.up = nn.ModuleList(ups)
        self.outc = OutConv(width, len(DECODER_OUTPUTS), groups)

    def decoder_input(self, feature, ray_dir, median_depth):
        """The ``(1, input_dim, H, W)`` input ``[f', d, r / R]`` of one view.

        ``r`` is detached, as in V18 (see ``lidar_decoder``).
        """
        if feature.ndim != 3 or feature.shape[0] != self.feature_dim:
            raise ValueError(
                f"lidar decoder expects a ({self.feature_dim},H,W) feature, "
                f"got {tuple(feature.shape)}"
            )
        height, width = feature.shape[-2:]
        if tuple(ray_dir.shape) != (RAY_CHANNELS, height, width):
            raise ValueError("ray_dir must be (3,H,W) like the feature")
        if tuple(median_depth.shape) != (RANGE_CHANNELS, height, width):
            raise ValueError("median_depth must be (1,H,W) like the feature")
        return torch.cat([
            feature,
            ray_dir.to(dtype=feature.dtype),
            median_depth.detach() / self.range_scale_m,
        ], dim=0).unsqueeze(0)

    def surface(self, feature, ray_dir, median_depth):
        """The hit surface's ``(intensity, raydrop)``, each ``(1, H, W)``."""
        x = self.inc(self.decoder_input(feature, ray_dir, median_depth))
        skips = [x]
        for block in self.down:
            x = block(x)
            skips.append(x)
        x = self.attention(skips.pop())
        for block in self.up:
            x = block(x, skips.pop())
        intensity, raydrop = torch.sigmoid(self.outc(x)[0]).unsqueeze(1).unbind(0)
        return intensity, raydrop

    def forward(self, feature, ray_dir, median_depth, alpha):
        """The pixel's ``(intensity, raydrop)`` over the no-return background.

        ``alpha`` is the rendered accumulated alpha ``(1, H, W)``.
        """
        if tuple(alpha.shape) != tuple(median_depth.shape):
            raise ValueError("alpha must be (1,H,W) like the feature")
        intensity, raydrop = self.surface(feature, ray_dir, median_depth)
        return alpha * intensity, 1.0 - alpha * (1.0 - raydrop)


__all__ = ["DECODER_OUTPUTS", "LidarUNetDecoder", "SelfAttention"]
