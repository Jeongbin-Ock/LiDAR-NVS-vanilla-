"""Prepare raw KITTI-360 train/val segments and their provenance manifest.

Building the manifest is separate from runtime sampling. This module reads raw
metadata and official split lists; only the CLI writes the resulting JSON.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np

from .metadata import (
    KITTI360_BENCHMARK_SCENES, KITTI360_BENCHMARK_SEQUENCE,
    read_timestamps_ns, sweep_inventory,
)


def retained_frame_segments(frame_ids, excluded_ranges=(), *, timestamps_ns=None, max_sweep_gap_sec=None):
    """Keep contiguous runs so a window cannot jump across an excluded clip.

    The timestamp array is indexed by original frame ID (Train/Val layout).
    Timestamp gaps can break runs without deleting either adjacent sweep.
    """
    frames = np.asarray(frame_ids, dtype=np.int64)
    if frames.ndim != 1 or np.any(frames < 0) or np.any(np.diff(frames) <= 0):
        raise ValueError("frame_ids must be increasing nonnegative IDs")
    keep = np.ones(len(frames), dtype=bool)
    for first, last in excluded_ranges:
        if int(first) < 0 or int(first) > int(last):
            raise ValueError("Exclusion bounds must be nonnegative and inclusive")
        keep &= (frames < int(first)) | (frames > int(last))
    frames = frames[keep]
    if not len(frames):
        return []
    breaks = np.diff(frames) != 1
    if max_sweep_gap_sec is not None:
        threshold = float(max_sweep_gap_sec)
        if not np.isfinite(threshold) or threshold <= 0 or timestamps_ns is None:
            raise ValueError("A positive max_sweep_gap_sec requires physical timestamps")
        delta_ns = np.diff(np.asarray(timestamps_ns)[frames])
        if np.any(delta_ns <= 0):
            raise ValueError("Retained sweep timestamps must be increasing")
        breaks |= delta_ns > round(threshold * 1e9)
    return [
        {"first_frame": int(run[0]), "last_frame": int(run[-1]), "num_sweeps": len(run)}
        for run in np.split(frames, np.flatnonzero(breaks) + 1)
    ]


def training_window_allowed(segments, first_frame, last_frame):
    """Both endpoints and the whole intervening interval must be in one run."""
    first, last = int(first_frame), int(last_frame)
    return first < last and any(
        run["first_frame"] <= first < last <= run["last_frame"] for run in segments
    )


def read_official_3d_ranges(path):
    """Official semantic split entries are accumulated clouds, not single scans."""
    ranges = {}
    pattern = r"data_3d_semantics/train/(2013_05_28_drive_\d{4}_sync)/static/(\d{10})_(\d{10})\.ply"
    for line in Path(path).read_text().splitlines():
        match = re.fullmatch(pattern, line.strip())
        if not match:
            raise ValueError(f"Invalid official 3D split entry: {line}")
        sequence, first, last = match.groups()
        if int(first) > int(last):
            raise ValueError(f"Invalid inclusive frame range: {line}")
        ranges.setdefault(sequence, []).append((int(first), int(last)))
    if not ranges:
        raise ValueError(f"Empty official 3D split: {path}")
    return ranges


def frame_range_mask(frame_ids, ranges):
    """Union of inclusive ranges; overlapping annotation windows count once."""
    frames = np.asarray(frame_ids, dtype=np.int64)
    selected = np.zeros(len(frames), dtype=bool)
    for first, last in ranges:
        selected |= (frames >= int(first)) & (frames <= int(last))
    return selected


def partition_frame_segments(frame_ids, benchmark_ranges, val_ranges, *, timestamps_ns, max_sweep_gap_sec):
    """Benchmark exclusions win over val; val wins over the remaining raw pool."""
    frames = np.asarray(frame_ids, dtype=np.int64)
    kwargs = {"timestamps_ns": timestamps_ns, "max_sweep_gap_sec": max_sweep_gap_sec}
    candidates = retained_frame_segments(frames, benchmark_ranges, **kwargs)
    train = retained_frame_segments(frames, [*benchmark_ranges, *val_ranges], **kwargs)
    val_mask = frame_range_mask(frames, val_ranges)
    val = retained_frame_segments(frames[val_mask], benchmark_ranges, **kwargs)
    candidate_count = sum(run["num_sweeps"] for run in candidates)
    train_count = sum(run["num_sweeps"] for run in train)
    val_count = sum(run["num_sweeps"] for run in val)
    if candidate_count != train_count + val_count:
        raise ValueError("Training and validation failed to partition the candidate pool")
    return {
        "candidate_sweeps_before_val": candidate_count,
        "excluded_benchmark_sweeps": len(frames) - candidate_count,
        "official_val_raw_sweeps": int(val_mask.sum()),
        "val_benchmark_overlap_sweeps": int(val_mask.sum()) - val_count,
        "train": {"num_sweeps": train_count, "segments": train},
        "val": {"num_sweeps": val_count, "segments": val},
    }


def build_training_split(config):
    """Project the official 3D val intervals onto the bbox-free raw NVS pool."""
    root = Path(str(config["dataroot"]))
    sequences = list(config["sequences"])
    scene_ids = [int(scene) for scene in config["excluded_benchmark_scenes"]]
    if not sequences or len(set(sequences)) != len(sequences):
        raise ValueError("sequences must be nonempty and unique")
    if not scene_ids or len(set(scene_ids)) != len(scene_ids) or set(scene_ids) != set(KITTI360_BENCHMARK_SCENES):
        raise ValueError("Exclude all ten distinct official LiDAR4D benchmark scenes")
    if KITTI360_BENCHMARK_SEQUENCE not in sequences:
        raise ValueError("Benchmark drive must be in the candidate pool for this clip-only split")
    ranges = [KITTI360_BENCHMARK_SCENES[scene][:2] for scene in scene_ids]
    validation = config["validation"]
    if validation["protocol"] != "official_3d_window_ranges" or not validation["keep_unlisted_raw_for_training"]:
        raise ValueError("This raw NVS pool uses official_3d_window_ranges with unlisted raw scans retained")
    official_train = read_official_3d_ranges(validation["train_list_path"])
    official_val = read_official_3d_ranges(validation["val_list_path"])
    if not set(official_train) <= set(sequences) or not set(official_val) <= set(sequences):
        raise ValueError("Official split contains sequences outside the configured raw pool")
    inventory, training, val_segments, split_stats, unique_frames = {}, {}, {}, {}, set()
    for sequence in sequences:
        frames, timestamps, stats = sweep_inventory(root / "data_3d_raw" / sequence / "velodyne_points")
        exclusions = ranges if sequence == KITTI360_BENCHMARK_SEQUENCE else ()
        partition = partition_frame_segments(
            frames, exclusions, official_val.get(sequence, ()),
            timestamps_ns=timestamps, max_sweep_gap_sec=config["max_sweep_gap_sec"],
        )
        inventory[sequence] = stats
        training[sequence], val_segments[sequence] = partition.pop("train"), partition.pop("val")
        train_mask = frame_range_mask(frames, official_train.get(sequence, ()))
        val_mask = frame_range_mask(frames, official_val.get(sequence, ()))
        partition.update({
            "official_train_windows": len(official_train.get(sequence, ())),
            "official_val_windows": len(official_val.get(sequence, ())),
            "official_train_val_raw_overlap_sweeps": int((train_mask & val_mask).sum()),
            "raw_sweeps_outside_semantic_windows": int((~(train_mask | val_mask)).sum()),
        })
        split_stats[sequence] = partition
        unique_frames.update((sequence, int(frame)) for frame in frames)
    test_clips = {}
    # Separate official Test SLAM files are inventory only, never train candidates.
    for velodyne_dir in sorted((root / "data_3d_raw").glob("test_*/*/velodyne_points")):
        frames, _, stats = sweep_inventory(velodyne_dir, compact_timestamps=True)
        test_clips[str(velodyne_dir.relative_to(root / "data_3d_raw"))] = stats
        unique_frames.update((velodyne_dir.parent.name, int(frame)) for frame in frames)
    benchmark_dir = root / "data_3d_raw" / KITTI360_BENCHMARK_SEQUENCE / "velodyne_points"
    timestamps = read_timestamps_ns(benchmark_dir / "timestamps.txt")
    scan_ids = set(int(path.stem) for path in (benchmark_dir / "data").glob("*.bin"))
    offsets = list(map(float, config["eval_input_offsets_sec"]))
    if len(offsets) != 2 or not all(np.isfinite(offsets)) or not offsets[0] < 0 < offsets[1]:
        raise ValueError("eval_input_offsets_sec must be finite and straddle the target")
    benchmark, all_inputs, all_targets = [], set(), set()
    for scene in scene_ids:
        first, last, targets, category = KITTI360_BENCHMARK_SCENES[scene]
        if not set(range(first, last + 1)) <= scan_ids:
            raise ValueError(f"Missing raw scans in benchmark clip {scene}")
        candidates = np.asarray([frame for frame in range(first, last + 1) if frame not in targets])
        windows = []
        for target in targets:
            inputs = []
            for offset in offsets:
                desired = int(timestamps[target]) + round(offset * 1e9)
                side = candidates[candidates < target] if offset < 0 else candidates[candidates > target]
                if not len(side) or not int(timestamps[first]) <= desired <= int(timestamps[last]):
                    raise ValueError(f"Input offset exceeds clip {scene}")
                inputs.append(int(side[np.argmin(np.abs(timestamps[side] - desired))]))
            windows.append({"input_frame_ids": inputs, "target_frame_id": target})
            all_inputs.update(inputs)
            all_targets.add(target)
        benchmark.append({"scene_id": scene, "first_frame": first, "last_frame": last,
                          "category": category, "num_sweeps": last - first + 1, "eval_windows": windows})
    retained_segments = training[KITTI360_BENCHMARK_SEQUENCE]["segments"]
    overlap = [frame for first, last in ranges for frame in range(first, last + 1)
               if any(run["first_frame"] <= frame <= run["last_frame"] for run in retained_segments)]
    if overlap:
        raise ValueError(f"Benchmark sweeps leaked into training: {overlap}")
    raw_total = sum(row["num_sweeps"] for row in inventory.values())
    test_total = sum(row["num_sweeps"] for row in test_clips.values())
    train_total = sum(row["num_sweeps"] for row in training.values())
    val_total = sum(row["num_sweeps"] for row in val_segments.values())
    sum_split = lambda key: sum(row[key] for row in split_stats.values())
    return {
        "manifest_version": 2, "dataroot": str(root), "config": dict(config),
        "stage": "raw train/val split for KITTI360DynamicDataset; pose bounds and start stride applied at dataset initialization",
        "window_rule": "A complete window must lie inside one retained segment; no crossing exclusions or timestamp gaps.",
        "summary": {"train_val_raw_sweeps": raw_total, "test_slam_files": test_total,
                    "all_raw_files": raw_total + test_total, "unique_raw_sweeps": len(unique_frames),
                    "excluded_benchmark_clips": len(benchmark),
                    "excluded_benchmark_sweeps": sum_split("excluded_benchmark_sweeps"),
                    "candidate_sweeps_before_val": train_total + val_total,
                    "official_val_raw_sweeps": sum_split("official_val_raw_sweeps"),
                    "val_benchmark_overlap_sweeps": sum_split("val_benchmark_overlap_sweeps"),
                    "validation_sweeps": val_total, "train_candidate_sweeps": train_total,
                    "official_train_windows": sum_split("official_train_windows"),
                    "official_val_windows": sum_split("official_val_windows"),
                    "raw_sweeps_outside_semantic_windows": sum_split("raw_sweeps_outside_semantic_windows"),
                    "eval_targets": len(all_targets),
                    "eval_unique_inputs": len(all_inputs), "eval_unique_sweeps": len(all_inputs | all_targets),
                    "benchmark_train_overlap_sweeps": len(overlap)},
        "train_segments": training, "val_segments": val_segments, "split_statistics": split_stats,
        "official_split_sources": [
            {"path": validation[key], "sha256": hashlib.sha256(Path(validation[key]).read_bytes()).hexdigest()}
            for key in ("train_list_path", "val_list_path")
        ],
        "benchmark_clips": benchmark,
        "train_val_inventory": inventory, "test_slam_inventory": test_clips,
    }
