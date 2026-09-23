"""Occupied Cartesian-token builder for the Dynamic 2DGS frontend.

For each frame the raw points are scattered into the Utonia bottleneck cells they
fall in (0.4/0.8 m, same grid as features['grid_coord']); each occupied cell is
one token: position = Utonia cell position, feature = that cell's Utonia
feature. Intensity already entered Utonia through its XYZI LoRA stem, so no
separate intensity stream exists.

Every token also carries its own-frame seed geometry: a fixed-count
``(N, K, 3)`` seed set, or the ``(N, K_max, K_max, 3)`` range-quantile
candidate bank a learned count router selects from. Every delta is measured
from the geometric cell center. ``raw_count`` is each token's own-frame point
count.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .grid_seed_config import resolve_grid_seed_config
from .grid_seeds import GridSeedData
from .token_aggregation import (
    aggregate_points_to_cells_with_seeds,
    split_by_offset,
)


@dataclass(frozen=True)
class OccupiedTokenBatch:
    """Per-frame occupied-token tensors consumed by ``Point2Gaus``."""

    positions: list[torch.Tensor]
    utonia_features: list[torch.Tensor]
    grid_coords: list[torch.Tensor]
    raw_counts: list[torch.Tensor]
    grid_seeds: list[GridSeedData]


class OccupiedGridTokenBuilder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.grid_seed_config = resolve_grid_seed_config(cfg)

    def forward(self, lidar_points, offset, pose, features, ptv3_input, mapper):
        device = features["feat"].device

        grid_coord_list = split_by_offset(features["grid_coord"], features["offset"])
        feat_list = split_by_offset(features["feat"], features["offset"])
        if "coord" in features:
            coord_list = split_by_offset(features["coord"], features["offset"])
        else:
            coord_list = [None] * len(feat_list)

        in_coord = ptv3_input["coord"].to(device)
        in_gc = ptv3_input["grid_coord"].to(device)
        in_off = ptv3_input["offset"].to(device)
        in_coord_list = split_by_offset(in_coord, in_off)
        in_gc_list = split_by_offset(in_gc, in_off)
        origins = [mapper.metric_origin(c, g) for c, g in zip(in_coord_list, in_gc_list)]

        pts_list = split_by_offset(lidar_points.to(device), offset)

        pos_list, ufeat_list, gc_list, raw_count_list, seed_data_list = (
            [], [], [], [], []
        )
        for i in range(len(feat_list)):
            pts = pts_list[i]
            upos, ufeat, _int5, occ, raw_count, seed_data = (
                aggregate_points_to_cells_with_seeds(
                    pts[:, :3], pts[:, 3], grid_coord_list[i], feat_list[i],
                    coord_list[i], origins[i], mapper,
                    self.grid_seed_config,
                )
            )
            pos_list.append(upos)
            ufeat_list.append(ufeat)
            gc_list.append(grid_coord_list[i].to(device)[occ])
            raw_count_list.append(raw_count)
            seed_data_list.append(seed_data)

        return OccupiedTokenBatch(
            positions=pos_list,
            utonia_features=ufeat_list,
            grid_coords=gc_list,
            raw_counts=raw_count_list,
            grid_seeds=seed_data_list,
        )
