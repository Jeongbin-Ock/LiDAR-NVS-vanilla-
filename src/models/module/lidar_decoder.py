"""V18's SplatAD-style decoder of alpha-blended Gaussian features.

SplatAD (Hess et al., CVPR 2025) renders LiDAR intensity and ray drop by
alpha-blending a learned per-Gaussian feature and decoding every pixel with a
small MLP that also reads the ray direction and a learned sensor embedding.
V18 decodes, per pixel,

    (i_s, p_s) = sigmoid(MLP([f', d, r / R, e]))
    intensity = a * i_s,    raydrop = (1 - a) + a * p_s,

where ``f'`` is the blended feature, ``d`` the unit ray from the rendering
LiDAR's origin through the pixel (written in the axes of the window's
reference frame), ``r`` the rendered median depth, ``R = range_scale_m`` the
sensor's effective range, ``e`` the sensor's learned embedding and ``a`` the
pixel's accumulated alpha. The MLP predicts the surface the ray hits: its
intensity and the probability that it still returns nothing. The pixel then
composites that surface over V15.4's background exactly as the SH render did
(``C + T * bg`` with intensity 0 and raydrop 1): a ray that hits nothing
(``a = 0``) returns nothing. Both maps keep the [0, 1] contract of the SH
renders they replace (``intensity_sh``, ``raydrop``), so every loss, metric and
evaluation reads them unchanged.

The losses reach the MLP and the embedding and the blended feature, hence
every Gaussian's feature and, through its blending weight, its opacity and
footprint; through ``a`` the raydrop loss also asks for coverage where the GT
returns and none where it drops, as in V15.4. The median depth enters
detached: it only tells the decoder how far
the pixel's surface is. Geometry is supervised by the depth losses, and a live
``r`` would let the raydrop loss move the median Gaussian's depth wherever the
decoder finds dropping easier to predict -- unopposed on GT-dropped pixels,
which carry no depth loss.

V18.1 is V18 without ``e`` (``sensor_embedding_dim: 0``): the MLP reads
``[f', d, r / R]``. With one LiDAR per training set, ``e`` is the same vector at
every pixel, so it only adds ``W_e e`` to the first layer's pre-activation, a
constant that layer's bias can hold. Both decoders represent the same
functions, and a V18 decoder folded into V18.1's (``b + W_e e`` as the bias)
decodes the same maps; trained from scratch they differ, because V18 learns
that constant through the product ``W_e e`` and V18.1 through the bias alone.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

# Rendered channels besides the feature and the sensor embedding: the ray
# direction (3) and the normalized median depth (1).
RAY_CHANNELS = 3
RANGE_CHANNELS = 1
DECODER_OUTPUTS = ("intensity", "raydrop")


def _positive_int(cfg, name):
    value = getattr(cfg, name, None)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(
            f"g2p.lidar_decoder.{name} must be a positive integer, got {value!r}"
        )
    return int(value)


def _non_negative_int(cfg, name):
    value = getattr(cfg, name, None)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(
            f"g2p.lidar_decoder.{name} must be a non-negative integer, got {value!r}"
        )
    return int(value)


class LidarFeatureDecoder(nn.Module):
    """``[f', d, r / R, e] -> (intensity, raydrop)`` for every pixel of a view.

    ``g2p.lidar_decoder`` sets ``feature_dim`` (the per-Gaussian feature width),
    ``sensor_embedding_dim``, ``num_sensors``, ``hidden_dims`` (the MLP's
    hidden widths, SiLU after each) and ``range_scale_m`` (R). V18:
    30 + 3 + 1 + 30 = 64 inputs, 64 -> 96 -> 96 -> 2. ``sensor_embedding_dim:
    0`` (V18.1) builds no embedding and reads no ``num_sensors``: 30 + 3 + 1 =
    34 inputs, and every view is sensor 0.
    """

    # The frame of the ray directions G2P hands this decoder: the window
    # reference frame's axes (V18.2's U-Net reads the LiDAR's own).
    ray_frame = "reference"

    def __init__(self, cfg):
        super().__init__()
        self.feature_dim = _positive_int(cfg, "feature_dim")
        self.sensor_embedding_dim = _non_negative_int(cfg, "sensor_embedding_dim")
        self.num_sensors = (
            _positive_int(cfg, "num_sensors") if self.sensor_embedding_dim else 1
        )
        hidden_dims = getattr(cfg, "hidden_dims", None)
        try:
            self.hidden_dims = tuple(int(width) for width in hidden_dims)
        except TypeError:
            self.hidden_dims = ()
        if not self.hidden_dims or min(self.hidden_dims) <= 0:
            raise ValueError(
                "g2p.lidar_decoder.hidden_dims must list the MLP's positive "
                f"hidden widths, got {hidden_dims!r}"
            )
        self.range_scale_m = float(getattr(cfg, "range_scale_m", 0.0))
        if not (math.isfinite(self.range_scale_m) and self.range_scale_m > 0.0):
            raise ValueError("g2p.lidar_decoder.range_scale_m must be positive")

        self.input_dim = (
            self.feature_dim + RAY_CHANNELS + RANGE_CHANNELS
            + self.sensor_embedding_dim
        )
        self.sensor_embedding = (
            nn.Embedding(self.num_sensors, self.sensor_embedding_dim)
            if self.sensor_embedding_dim else None
        )
        layers = []
        width = self.input_dim
        for hidden in self.hidden_dims:
            layers += [nn.Linear(width, hidden), nn.SiLU()]
            width = hidden
        layers.append(nn.Linear(width, len(DECODER_OUTPUTS)))
        self.mlp = nn.Sequential(*layers)

    def decoder_input(self, feature, ray_dir, median_depth, sensor_index=0):
        """The ``(H*W, input_dim)`` rows ``[f', d, r / R, e]`` of one view.

        ``r`` is detached (see the module docstring). Without a sensor
        embedding (V18.1) the rows are ``[f', d, r / R]``.
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
        columns = [
            feature,
            ray_dir.to(dtype=feature.dtype),
            median_depth.detach() / self.range_scale_m,
        ]
        if self.sensor_embedding is not None:
            index = torch.full(
                (1,), int(sensor_index), dtype=torch.long, device=feature.device
            )
            embedding = self.sensor_embedding(index).reshape(-1, 1, 1)
            columns.append(
                embedding.to(dtype=feature.dtype).expand(-1, height, width)
            )
        elif int(sensor_index) != 0:
            raise ValueError(
                "a lidar decoder without a sensor embedding decodes sensor 0 "
                f"only, got sensor_index={sensor_index}"
            )
        rows = torch.cat(columns, dim=0)
        return rows.reshape(self.input_dim, height * width).transpose(0, 1)

    def surface(self, feature, ray_dir, median_depth, sensor_index=0):
        """The hit surface's ``(intensity, raydrop)``, each ``(1, H, W)``."""
        height, width = feature.shape[-2:]
        logits = self.mlp(
            self.decoder_input(feature, ray_dir, median_depth, sensor_index)
        )
        intensity, raydrop = torch.sigmoid(logits).transpose(0, 1).reshape(
            len(DECODER_OUTPUTS), 1, height, width
        ).unbind(0)
        return intensity, raydrop

    def forward(self, feature, ray_dir, median_depth, alpha, sensor_index=0):
        """The pixel's ``(intensity, raydrop)`` over the no-return background.

        ``alpha`` is the rendered accumulated alpha ``(1, H, W)``.
        """
        if tuple(alpha.shape) != tuple(median_depth.shape):
            raise ValueError("alpha must be (1,H,W) like the feature")
        intensity, raydrop = self.surface(
            feature, ray_dir, median_depth, sensor_index
        )
        return alpha * intensity, 1.0 - alpha * (1.0 - raydrop)


__all__ = ["DECODER_OUTPUTS", "LidarFeatureDecoder"]
