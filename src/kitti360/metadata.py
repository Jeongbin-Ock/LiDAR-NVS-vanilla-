"""KITTI-360 benchmark definitions and raw sweep metadata, without model imports."""
from __future__ import annotations

from pathlib import Path

import numpy as np


KITTI360_BENCHMARK_SEQUENCE = "2013_05_28_drive_0000_sync"
# Inclusive bounds, held-out targets and scene category, from the official
# https://github.com/ispc-lab/LiDAR4D/blob/main/data/preprocess/kitti360_to_nerf.py
KITTI360_BENCHMARK_SCENES = {
    1538: (1538, 1601, (1551, 1564, 1577, 1590), "static"),
    1728: (1728, 1791, (1741, 1754, 1767, 1780), "static"),
    1908: (1908, 1971, (1921, 1934, 1947, 1960), "static"),
    3353: (3353, 3416, (3366, 3379, 3392, 3405), "static"),
    2350: (2350, 2400, (2360, 2370, 2380, 2390), "dynamic"),
    4950: (4950, 5000, (4960, 4970, 4980, 4990), "dynamic"),
    8120: (8120, 8170, (8130, 8140, 8150, 8160), "dynamic"),
    10200: (10200, 10250, (10210, 10220, 10230, 10240), "dynamic"),
    10750: (10750, 10800, (10760, 10770, 10780, 10790), "dynamic"),
    11400: (11400, 11450, (11410, 11420, 11430, 11440), "dynamic"),
}


def read_timestamps_ns(path):
    timestamps = np.asarray(Path(path).read_text().splitlines(), dtype="datetime64[ns]").astype(np.int64)
    if not len(timestamps) or np.any(np.isnat(timestamps.view("datetime64[ns]"))) or np.any(np.diff(timestamps) <= 0):
        raise ValueError(f"Expected increasing scan timestamps: {path}")
    return timestamps


def sweep_inventory(velodyne_dir, *, compact_timestamps=False):
    """Read metadata only. Test SLAM timestamps match sorted files, not IDs."""
    velodyne_dir = Path(velodyne_dir)
    frames = np.asarray(sorted(int(path.stem) for path in (velodyne_dir / "data").glob("*.bin")), dtype=np.int64)
    if not len(frames):
        raise ValueError(f"No sweep files: {velodyne_dir}")
    timestamps = read_timestamps_ns(velodyne_dir / "timestamps.txt")
    if compact_timestamps:
        if len(frames) != len(timestamps):
            raise ValueError(f"Test clip scan/timestamp count mismatch: {velodyne_dir}")
        saved = timestamps
    else:
        if frames[-1] >= len(timestamps):
            raise ValueError(f"Frame ID exceeds timestamp rows: {velodyne_dir}")
        saved = timestamps[frames]
    intervals = np.diff(saved) / 1e9
    if not len(intervals):
        raise ValueError(f"At least two sweeps needed for timing: {velodyne_dir}")
    stats = {
        "num_sweeps": len(frames), "first_frame": int(frames[0]), "last_frame": int(frames[-1]),
        "timestamp_rows": len(timestamps), "timestamp_addressing": "sorted_files" if compact_timestamps else "frame_id",
        "missing_frame_ids": int(frames[-1] - frames[0] + 1 - len(frames)),
        "interval_sec": {"mean": float(intervals.mean()), "median": float(np.median(intervals)),
                         "min": float(intervals.min()), "max": float(intervals.max())},
        "mean_saved_hz": float(1 / intervals.mean()),
        "timestamp_gaps_over_0_2_sec": [
            {"first_frame": int(frames[index]), "last_frame": int(frames[index + 1]), "duration_sec": float(intervals[index])}
            for index in np.flatnonzero(intervals > .2)
        ],
    }
    return frames, timestamps, stats
