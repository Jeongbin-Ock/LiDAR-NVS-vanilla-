"""Compatibility API and CLI for the modular KITTI-360 data utilities.

New runtime code imports ``src.kitti360.metadata`` or ``src.kitti360.sampling``.
Split preparation uses ``python -m src.kitti360 --config ...``. Historical
imports and ``python -m src.kitti360_split`` remain supported here.
"""
from .kitti360.metadata import (
    KITTI360_BENCHMARK_SCENES,
    KITTI360_BENCHMARK_SEQUENCE,
    read_timestamps_ns,
    sweep_inventory,
)
from .kitti360.sampling import (
    eligible_endpoint_frames,
    sample_window_frames,
    window_start_frames,
)
from .kitti360.splits import (
    build_training_split,
    frame_range_mask,
    partition_frame_segments,
    read_official_3d_ranges,
    retained_frame_segments,
    training_window_allowed,
)
from .kitti360.__main__ import main

__all__ = [
    "KITTI360_BENCHMARK_SCENES", "KITTI360_BENCHMARK_SEQUENCE",
    "read_timestamps_ns", "sweep_inventory", "eligible_endpoint_frames",
    "sample_window_frames", "window_start_frames", "build_training_split",
    "frame_range_mask", "partition_frame_segments", "read_official_3d_ranges",
    "retained_frame_segments", "training_window_allowed", "main",
]


if __name__ == "__main__":
    main()
