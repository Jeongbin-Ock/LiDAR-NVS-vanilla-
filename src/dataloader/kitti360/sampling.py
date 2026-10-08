"""Physical-time KITTI-360 window selection for train and validation datasets."""
from __future__ import annotations

import numpy as np


def eligible_endpoint_frames(segment, timestamps_ns, start_frame, *, endpoint_min_sec, endpoint_max_sec):
    """Available endpoints inside one allowed segment and physical time limits."""
    first, last, start = int(segment["first_frame"]), int(segment["last_frame"]), int(start_frame)
    minimum, maximum = float(endpoint_min_sec), float(endpoint_max_sec)
    if not all(np.isfinite([minimum, maximum])) or not 0 < minimum <= maximum:
        raise ValueError("Window durations must be finite with 0 < min <= max")
    if not first <= start <= last:
        raise ValueError("Start frame must belong to the retained segment")
    frames = np.arange(start + 1, last + 1, dtype=np.int64)
    # Subtract integer nanoseconds before converting anything to seconds.
    duration_ns = np.asarray(timestamps_ns)[frames] - int(timestamps_ns[start])
    return frames[(duration_ns >= round(minimum * 1e9)) & (duration_ns <= round(maximum * 1e9))].tolist()


def sample_window_frames(segment, timestamps_ns, start_frame, *, endpoint_min_sec,
                         endpoint_max_sec, gt_middle_count=3, rng):
    """nuScenes-style draw: random available endpoint, unique middle GT, endpoints.

    Use the worker-seeded Python RNG for training and a per-anchor seeded RNG
    for repeatable validation. The result never duplicates a short-window GT.
    """
    if int(gt_middle_count) != gt_middle_count or gt_middle_count < 0:
        raise ValueError("gt_middle_count must be a nonnegative integer")
    candidates = eligible_endpoint_frames(
        segment, timestamps_ns, start_frame,
        endpoint_min_sec=endpoint_min_sec, endpoint_max_sec=endpoint_max_sec,
    )
    if not candidates:
        raise ValueError("No available endpoint in the requested duration range")
    endpoint = int(rng.choice(candidates))
    interior = list(range(int(start_frame) + 1, endpoint))
    middle = sorted(rng.sample(interior, min(int(gt_middle_count), len(interior))))
    return endpoint, middle, [int(start_frame), *middle, endpoint]


def window_start_frames(segment, timestamps_ns, *, stride_sec, endpoint_min_sec, endpoint_max_sec):
    """Snap a physical start-time grid to saved sweeps, keeping valid anchors.

    The grid starts at the first usable sweep of each retained segment. Near
    the end, a shorter random-duration window is kept if any endpoint qualifies.
    """
    stride = float(stride_sec)
    if not np.isfinite(stride) or stride <= 0 or round(stride * 1e9) < 1:
        raise ValueError("window stride must be a positive finite number of seconds")
    first, last = int(segment["first_frame"]), int(segment["last_frame"])
    stamps = np.asarray(timestamps_ns)[first:last + 1]
    if not len(stamps) or np.any(np.diff(stamps) <= 0):
        raise ValueError("Start anchors require a nonempty increasing segment")
    elapsed = int(stamps[-1]) - int(stamps[0])
    grid = int(stamps[0]) + np.arange(0, elapsed + 1, round(stride * 1e9), dtype=np.int64)
    right = np.minimum(np.searchsorted(stamps, grid), len(stamps) - 1)
    left = np.maximum(right - 1, 0)
    nearest = np.where(grid - stamps[left] <= stamps[right] - grid, left, right)
    return [int(frame) for frame in np.unique(nearest + first) if eligible_endpoint_frames(
        segment, timestamps_ns, int(frame),
        endpoint_min_sec=endpoint_min_sec, endpoint_max_sec=endpoint_max_sec,
    )]
