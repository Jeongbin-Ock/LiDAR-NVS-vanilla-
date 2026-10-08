"""Validation metrics split by window group (e.g. compact-val dynamic/static/stopped).

Each validation batch adds its already-computed metrics to a fixed
groups x keys table, weighted by batch size. At epoch end every rank reduces
the same table with a single all_reduce, so the per-group means cover all
ranks. A group that never occurs on one rank cannot desynchronize the logging
collectives, which per-key ``self.log(..., sync_dist=True)`` would risk.
Accumulation stays on the device and never synchronizes with the host.
"""
from __future__ import annotations

import torch
import torch.distributed as dist


class GroupMetricAccumulator:
    def __init__(self, groups, keys):
        self.groups = tuple(groups)
        self.keys = tuple(sorted(keys))
        self._group_index = {group: index for index, group in enumerate(self.groups)}
        self._sums = None  # [group, key, (weighted sum, weight)]
        self._windows = None  # [group]
        self.mixed_batches = 0

    def _storage(self, device):
        if self._sums is None:
            self._sums = torch.zeros(len(self.groups), len(self.keys), 2, dtype=torch.float64, device=device)
            self._windows = torch.zeros(len(self.groups), dtype=torch.float64, device=device)
        return self._sums, self._windows

    def add(self, losses, labels, batch_size):
        """Add one batch whose windows share one group; mixed batches are skipped.

        Batch metrics are pooled over the whole batch, so a batch that mixes
        groups cannot be split. Validation uses one window per batch.
        """
        labels = list(labels)
        if len(set(labels)) != 1:
            self.mixed_batches += 1
            return
        group = labels[0]
        if group not in self._group_index:
            raise ValueError(f"Unknown validation window group {group!r}; expected {self.groups}")
        row = self._group_index[group]
        weight = float(batch_size)
        for column, key in enumerate(self.keys):
            value = losses.get(key)
            if not torch.is_tensor(value):
                continue
            sums, windows = self._storage(value.device)
            sums[row, column, 0] += value.detach().to(torch.float64).mean() * weight
            sums[row, column, 1] += weight
        if self._windows is not None:
            self._windows[row] += weight

    def reduce(self, device):
        """Sum over ranks; returns ``{group: {"num_windows": n, key: mean, ...}}``.

        Every rank must call this, in the same order, even without labels.
        Groups without windows and keys never computed for a group are left out.
        """
        sums, windows = self._storage(device)
        flat = torch.cat([sums.flatten(), windows, torch.tensor([float(self.mixed_batches)],
                                                                  dtype=torch.float64, device=sums.device)])
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(flat)
        size = sums.numel()
        sums = flat[:size].view_as(sums).cpu()
        windows = flat[size:size + len(self.groups)].cpu()
        self.mixed_batches = int(flat[-1].item())
        result = {}
        for row, group in enumerate(self.groups):
            if windows[row] <= 0:
                continue
            entry = {"num_windows": float(windows[row])}
            for column, key in enumerate(self.keys):
                if sums[row, column, 1] > 0:
                    entry[key] = float(sums[row, column, 0] / sums[row, column, 1])
            result[group] = entry
        return result
