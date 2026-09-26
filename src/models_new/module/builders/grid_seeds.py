"""Deterministic occupied-grid seed generation algorithms."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from src.experiment_config.definitions import (
    SEED_MODE_MEDOID,
    SEED_MODE_RANGE_QUANTILE,
    SEED_MODE_TOKEN,
)

from .grid_geometry import _cell_centers


@dataclass(frozen=True)
class GridSeedData:
    """Own-frame seed geometry aligned with occupied token rows.

    A fixed count stores one seed set with shape ``(N, K_max, 3)`` and its
    per-token slot count ``anchor_k`` (``K_max`` for every token). Learned
    Gumbel routing stores every K-specific candidate set with shape
    ``(N, K_max, K_max, 3)``, each row filled with ``k`` seeds (range
    quantiles, or the token anchor repeated), and ``anchor_k`` is ``None``
    because K is predicted only after temporal feature fusion.
    """

    seed_local: torch.Tensor
    delta_local: torch.Tensor
    anchor_k: torch.Tensor | None

    @property
    def seed_sensor(self):
        """Backward-compatible alias; these rows use the frame-local basis."""
        return self.seed_local

    @property
    def delta_sensor(self):
        """Backward-compatible alias; these deltas use the frame-local basis."""
        return self.delta_local


def _r_quantile_seed_data(points, cell_idx, counts, grid_coord, occupied,
                          metric_origin, mapper, k_max):
    """Select ``k_max`` deterministic range-quantile points per occupied cell.

    A token with fewer raw points than slots repeats observed points, exactly
    as the learned-count candidate bank does; the per-slot parameter blocks
    separate them afterwards.
    """
    occupied_index = occupied.nonzero(as_tuple=True)[0]
    num_cells = occupied_index.numel()
    seeds = points.new_zeros((num_cells, k_max, 3))
    delta = points.new_zeros((num_cells, k_max, 3))
    occupied_count = counts.long()[occupied_index]
    anchor_k = torch.full_like(occupied_count, int(k_max))
    if num_cells == 0:
        return GridSeedData(seeds, delta, anchor_k)

    point_range = points.norm(dim=-1)
    order = torch.arange(points.shape[0], device=points.device)
    # Stable least-to-most-significant sorting gives (cell, r, x, y, z).
    for value in (points[:, 2], points[:, 1], points[:, 0], point_range, cell_idx):
        order = order[torch.argsort(value[order], stable=True)]
    sorted_points = points[order]

    count_long = counts.long()
    starts = torch.cumsum(count_long, dim=0) - count_long
    slot = torch.arange(k_max, device=points.device).view(1, -1)
    valid = slot < anchor_k.view(-1, 1)
    rank = torch.div(
        (2 * slot + 1) * occupied_count.view(-1, 1),
        2 * anchor_k.view(-1, 1),
        rounding_mode="floor",
    )
    sorted_row = starts[occupied_index].view(-1, 1) + rank
    seeds[valid] = sorted_points[sorted_row[valid]]

    cell_center = _cell_centers(
        grid_coord, occupied, metric_origin, mapper, points.dtype
    )
    delta[valid] = seeds[valid] - cell_center[:, None, :].expand_as(seeds)[valid]
    return GridSeedData(seeds, delta, anchor_k)


def _r_quantile_seed_bank(points, cell_idx, counts, grid_coord, occupied,
                          metric_origin, mapper, k_max):
    """Precompute the range-quantile seed set for every candidate K.

    The learned count router runs after cross-frame feature fusion, while seed
    geometry is built from the raw points before it. Therefore all candidates
    are built up front. Candidate ``k - 1`` follows the same deterministic rule
    as the fixed-count range-quantile seeds:

        rank(s, K) = floor(((2s + 1) * N_raw) / (2K)),  s = 0 .. K - 1.

    If ``N_raw < K``, quantiles may intentionally select the same observed raw
    point more than once; the K-specific Gaussian head can separate those slots
    through its learned position offsets.
    """
    occupied_index = occupied.nonzero(as_tuple=True)[0]
    num_cells = occupied_index.numel()
    seeds = points.new_zeros((num_cells, k_max, k_max, 3))
    delta = points.new_zeros((num_cells, k_max, k_max, 3))
    if num_cells == 0:
        return GridSeedData(seeds, delta, None)

    point_range = points.norm(dim=-1)
    order = torch.arange(points.shape[0], device=points.device)
    # Stable least-to-most-significant sorting gives (cell, r, x, y, z).
    for value in (points[:, 2], points[:, 1], points[:, 0], point_range, cell_idx):
        order = order[torch.argsort(value[order], stable=True)]
    sorted_points = points[order]

    count_long = counts.long()
    occupied_count = count_long[occupied_index]
    starts = torch.cumsum(count_long, dim=0) - count_long
    cell_start = starts[occupied_index].view(-1, 1)
    for k in range(1, k_max + 1):
        slot = torch.arange(k, device=points.device).view(1, -1)
        rank = torch.div(
            (2 * slot + 1) * occupied_count.view(-1, 1),
            2 * k,
            rounding_mode="floor",
        )
        seeds[:, k - 1, :k] = sorted_points[cell_start + rank]

    cell_center = _cell_centers(
        grid_coord, occupied, metric_origin, mapper, points.dtype
    )
    for k in range(1, k_max + 1):
        delta[:, k - 1, :k] = (
            seeds[:, k - 1, :k] - cell_center[:, None, :]
        )
    return GridSeedData(seeds, delta, None)


def _medoid_seed_data(points, cell_idx, counts, grid_coord, occupied,
                      metric_origin, mapper, k_max, slots=1):
    """Observed centroid-medoid seeds for each occupied token.

    The seed is the raw point nearest its token's Cartesian raw-point mean.
    This deterministic, observed-point medoid avoids an O(n^2) exact
    pairwise-medoid cost while retaining a surface-supported seed even for
    sparse cells.

    ``slots`` seeds are emitted, all at that one medoid. Above one they are
    separated only by the independent parameter block each slot owns.
    """
    slots = int(slots)
    if not 1 <= slots <= k_max:
        raise ValueError("medoid seed slots must be in [1, K_max]")
    occupied_index = occupied.nonzero(as_tuple=True)[0]
    num_cells = occupied_index.numel()
    seeds = points.new_zeros((num_cells, k_max, 3))
    delta = points.new_zeros((num_cells, k_max, 3))
    anchor_k = torch.full(
        (num_cells,), slots, dtype=torch.long, device=points.device
    )
    if num_cells == 0:
        return GridSeedData(seeds, delta, anchor_k)

    count_long = counts.long()
    point_sum = points.new_zeros((counts.numel(), 3))
    point_sum.index_add_(0, cell_idx, points)
    point_mean = point_sum / count_long.clamp_min(1).to(points.dtype).unsqueeze(-1)
    distance2 = ((points - point_mean[cell_idx]) ** 2).sum(dim=-1)
    distance_key = torch.round(distance2 * 1.0e6)

    # Stable least-to-most-significant sorting gives (cell, distance, x, y, z).
    order = torch.arange(points.shape[0], device=points.device)
    for value in (points[:, 2], points[:, 1], points[:, 0], distance_key, cell_idx):
        order = order[torch.argsort(value[order], stable=True)]
    starts = torch.cumsum(count_long, dim=0) - count_long
    medoid = points[order[starts[occupied_index]]]
    seeds[:, :slots] = medoid[:, None, :]

    cell_center = _cell_centers(
        grid_coord, occupied, metric_origin, mapper, points.dtype
    )
    delta[:, :slots] = seeds[:, :slots] - cell_center[:, None, :]
    return GridSeedData(seeds, delta, anchor_k)


def _token_position_seed_data(token_position, grid_coord, occupied,
                              metric_origin, mapper, k_max, slots=1):
    """Every slot of a token anchored at the token's Utonia position.

    The position is Utonia's pooled cell coordinate: the same point the token
    carries through temporal attention. Slot identity is the output order of
    the Gaussian head: each slot owns a distinct parameter (including offset)
    block, so they start at the same anchor but learn independent
    displacements.
    """
    slots = int(slots)
    if not 1 <= slots <= k_max:
        raise ValueError("token-anchor slots must be in [1, K_max]")
    num_cells = token_position.shape[0]
    seeds = token_position.new_zeros((num_cells, k_max, 3))
    delta = token_position.new_zeros((num_cells, k_max, 3))
    anchor_k = torch.full(
        (num_cells,), slots, dtype=torch.long, device=token_position.device
    )
    seeds[:, :slots] = token_position[:, None, :]

    cell_center = _cell_centers(
        grid_coord, occupied, metric_origin, mapper, token_position.dtype
    )
    delta[:, :slots] = seeds[:, :slots] - cell_center[:, None, :]
    return GridSeedData(seeds, delta, anchor_k)


def _token_position_seed_bank(token_position, grid_coord, occupied,
                              metric_origin, mapper, k_max):
    """The learned-count candidate bank with every row on the token anchor.

    Row ``k - 1`` fills its first ``k`` slots with the token position and
    leaves the padding at zero, the same layout as the range-quantile bank.
    """
    num_cells = token_position.shape[0]
    seeds = token_position.new_zeros((num_cells, k_max, k_max, 3))
    delta = token_position.new_zeros((num_cells, k_max, k_max, 3))
    if num_cells == 0:
        return GridSeedData(seeds, delta, None)
    cell_center = _cell_centers(
        grid_coord, occupied, metric_origin, mapper, token_position.dtype
    )
    for k in range(1, k_max + 1):
        seeds[:, k - 1, :k] = token_position[:, None, :]
        delta[:, k - 1, :k] = (token_position - cell_center)[:, None, :]
    return GridSeedData(seeds, delta, None)


def _build_seed_data(config, points, cell_idx, counts, grid_coord, occupied,
                     metric_origin, mapper, token_position):
    """Dispatch one occupied frame to the seed rule its config selects."""
    k_max = config.k_max
    if config.learned:
        if config.seed_mode == SEED_MODE_TOKEN:
            return _token_position_seed_bank(
                token_position, grid_coord, occupied, metric_origin, mapper,
                k_max=k_max,
            )
        if config.seed_mode != SEED_MODE_RANGE_QUANTILE:
            raise ValueError(
                f"Unsupported learned grid seed_mode={config.seed_mode!r}"
            )
        return _r_quantile_seed_bank(
            points, cell_idx, counts, grid_coord, occupied, metric_origin,
            mapper, k_max=k_max,
        )
    # A fixed count fills every slot of every token.
    if config.seed_mode == SEED_MODE_RANGE_QUANTILE:
        return _r_quantile_seed_data(
            points, cell_idx, counts, grid_coord, occupied, metric_origin,
            mapper, k_max=k_max,
        )
    if config.seed_mode == SEED_MODE_MEDOID:
        return _medoid_seed_data(
            points, cell_idx, counts, grid_coord, occupied, metric_origin,
            mapper, k_max=k_max, slots=k_max,
        )
    if config.seed_mode == SEED_MODE_TOKEN:
        return _token_position_seed_data(
            token_position, grid_coord, occupied, metric_origin, mapper,
            k_max=k_max, slots=k_max,
        )
    raise ValueError(f"Unsupported fixed grid seed_mode={config.seed_mode!r}")

__all__ = ["GridSeedData"]
