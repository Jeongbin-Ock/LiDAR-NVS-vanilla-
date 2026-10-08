"""KITTI-360 train/val windows with shared calibrated LiDAR GT and cameras."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random

import numpy as np
from omegaconf import OmegaConf
import torch
from torch.utils.data import Dataset

from ..kitti360 import compact_val
from ..kitti360.metadata import KITTI360_BENCHMARK_SCENES
from ..kitti360.sampling import sample_window_frames, window_start_frames
from .kitti360_lidar4d_test import (
    KITTI360SequenceReader, LiDAR4DKITTI360TestDataset, _transform_points,
)
from .nuscene_dynamic_lidar4d import dynamic_lidar4d_multiframe_collate_fn


class KITTI360DynamicDataset(Dataset):
    """Sample two inputs and up to three middle GT sweeps from one split run.

    ``window_subset`` is a KITTI-360 compact-val manifest path (val only). It
    keeps the manifest's anchors and their full-val random draws, and every
    item then carries its group in ``item["meta"]["window_group"]``.
    """

    collate_fn = staticmethod(dynamic_lidar4d_multiframe_collate_fn)

    def __init__(self, cfg, split="train", window_subset=None):
        self.cfg, self.split = cfg, str(split)
        self.window_labels = None
        if window_subset and self.split != "val":
            raise ValueError("A KITTI360 compact-val manifest selects val windows only")
        if str(cfg.mode) != "nvs":
            raise ValueError("KITTI360 dynamic training requires bbox-free data.mode=nvs")
        self._benchmark = None
        if self.split == "test":
            self._benchmark = LiDAR4DKITTI360TestDataset(cfg)
            return
        if self.split not in {"train", "val"}:
            raise ValueError("KITTI360 split must be train, val or test")
        if str(cfg.pose_missing_policy) != "interpolate":
            raise ValueError("KITTI360 training uses timestamp-based pose interpolation without extrapolation")
        if int(cfg.sweep_hop) != 1:
            raise ValueError("KITTI360 dynamic training uses every saved sweep (sweep_hop=1)")
        self.minimum = float(cfg.dynamic_window.endpoint_min_sec)
        self.maximum = float(cfg.dynamic_window.endpoint_max_sec)
        self.middle_count = int(cfg.gt_middle_count)
        self.eval_seed = int(cfg.dynamic_window.eval_seed)
        self.stride_sec = float(cfg.window_stride_sec)
        if not np.isfinite(self.minimum) or not np.isfinite(self.maximum) or not 0 < self.minimum <= self.maximum:
            raise ValueError("Endpoint durations must satisfy 0 < min <= max")
        if self.middle_count < 0:
            raise ValueError("gt_middle_count must be nonnegative")
        path = Path(str(cfg.split_manifest_path))
        content = path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        if not cfg.split_manifest_sha256 or digest != str(cfg.split_manifest_sha256):
            raise ValueError(f"KITTI360 split manifest hash mismatch: {path}")
        manifest = json.loads(content)
        if int(manifest.get("manifest_version", 0)) != 2:
            raise ValueError("KITTI360 training requires the version-2 train/val manifest")
        if Path(manifest["dataroot"]).resolve() != Path(str(cfg.dataroot)).resolve():
            raise ValueError("KITTI360 manifest dataroot differs from the training data root")
        if float(manifest["config"]["max_sweep_gap_sec"]) != float(cfg.max_sweep_gap_sec):
            raise ValueError("KITTI360 timestamp-gap policy differs from its split manifest")
        self.readers, self.index, self.active_segments = {}, [], {}
        self.index_counts, self.pose_unavailable_sweeps = {}, 0
        split_key = "train_segments" if self.split == "train" else "val_segments"
        for sequence, record in manifest[split_key].items():
            reader_cfg = OmegaConf.merge(cfg, {"sequence": sequence, "verbose": False})
            reader = KITTI360SequenceReader(reader_cfg)
            self.readers[sequence] = reader
            pose_first, pose_last = int(reader.poses.frames[0]), int(reader.poses.frames[-1])
            segments = []
            count = 0
            for stored in record["segments"]:
                first, last = int(stored["first_frame"]), int(stored["last_frame"])
                if sequence == "2013_05_28_drive_0000_sync" and any(
                    first <= upper and lower <= last for lower, upper, _, _ in KITTI360_BENCHMARK_SCENES.values()
                ):
                    raise ValueError("A KITTI360 train/val segment intersects a LiDAR4D test clip")
                if first < 0 or first > last or last >= len(reader.timestamps_ns):
                    raise ValueError("Invalid KITTI360 split segment bounds")
                if last - first + 1 != int(stored["num_sweeps"]):
                    raise ValueError("KITTI360 split segment count does not match inclusive bounds")
                if any(not reader.scan_path(frame).is_file() for frame in range(first, last + 1)):
                    raise ValueError("A KITTI360 split segment contains an absent raw scan")
                delta = np.diff(reader.timestamps_ns[first:last + 1])
                if np.any(delta > round(float(cfg.max_sweep_gap_sec) * 1e9)):
                    raise ValueError("A KITTI360 split segment crosses a timestamp gap")
                clipped_first, clipped_last = max(first, pose_first), min(last, pose_last)
                retained = max(0, clipped_last - clipped_first + 1)
                self.pose_unavailable_sweeps += last - first + 1 - retained
                if not retained:
                    continue
                segment = {"first_frame": clipped_first, "last_frame": clipped_last, "num_sweeps": retained}
                segment_index = len(segments)
                segments.append(segment)
                starts = window_start_frames(
                    segment, reader.timestamps_ns, stride_sec=self.stride_sec,
                    endpoint_min_sec=self.minimum, endpoint_max_sec=self.maximum,
                )
                self.index.extend((sequence, segment_index, start) for start in starts)
                count += len(starts)
            self.active_segments[sequence] = segments
            self.index_counts[sequence] = count
        if not self.index:
            raise ValueError(f"No usable KITTI360 {self.split} windows")
        if bool(getattr(cfg, "verbose", False)):
            print(f"[kitti360-{self.split}] {len(self.index)} starts, stride={self.stride_sec:g}s, "
                  f"endpoints=[{self.minimum:g},{self.maximum:g}]s, middle_gt<={self.middle_count}; "
                  f"{self.pose_unavailable_sweeps} sweeps outside direct/bracketing pose bounds")
        if window_subset:
            manifest = compact_val.load_manifest(window_subset, compact_val.dataset_contract(cfg, self.split))
            keys = [compact_val.window_key(sequence, start) for sequence, _, start in self.index]
            positions, self.window_labels = compact_val.select_windows(keys, manifest)
            self.index = [self.index[position] for position in positions]
            if bool(getattr(cfg, "verbose", False)):
                counts = {group: self.window_labels.count(group) for group in compact_val.COMPACT_VAL_GROUPS}
                print(f"[kitti360-{self.split}] compact val {window_subset}: {len(self.index)} windows {counts}")

    def __len__(self):
        return len(self._benchmark) if self._benchmark is not None else len(self.index)

    def sample_window_frames(self, index):
        sequence, segment_index, first = self.index[index]
        segment = self.active_segments[sequence][segment_index]
        if self.split == "train":
            rng = random
        else:
            material = f"{self.eval_seed}|{sequence}|{first}|{segment['last_frame']}".encode()
            rng = random.Random(int.from_bytes(hashlib.blake2b(material, digest_size=8).digest(), "big"))
        _, _, used = sample_window_frames(
            segment, self.readers[sequence].timestamps_ns, first,
            endpoint_min_sec=self.minimum, endpoint_max_sec=self.maximum,
            gt_middle_count=self.middle_count, rng=rng,
        )
        return sequence, used

    def __getitem__(self, index):
        if self._benchmark is not None:
            item = self._benchmark[index]
            return {"input": item["input"], "gt": {"cameras": item["gt"]["gt_cameras"]}}
        sequence, frames = self.sample_window_frames(index)
        reader = self.readers[sequence]
        world_sensor, world_local, world_to_ref = reader._world_transforms(frames)
        selected = (0, len(frames) - 1)
        sensor = [reader._load_points(frame) for frame in frames]
        seconds = torch.tensor([
            (int(reader.timestamps_ns[frame]) - int(reader.timestamps_ns[frames[0]])) / 1e9
            for frame in frames
        ], dtype=torch.float32)
        duration = seconds[-1].clone()
        normalized = seconds / duration
        cameras = [reader._gt_camera(
            points, np.linalg.inv(world_sensor[slot]) @ np.linalg.inv(world_to_ref),
            float(normalized[slot]), float(seconds[slot]),
        ) for slot, points in enumerate(sensor)]
        endpoint_sensor = [sensor[slot] for slot in selected]
        radius = float(self.cfg.ego_radius)
        endpoint_sensor = [points[torch.linalg.norm(points[:, :3], dim=1) > radius] for points in endpoint_sensor]
        endpoint_ego = [_transform_points(points, reader.ego_from_lidar) for points in endpoint_sensor]
        endpoint_local = endpoint_ego if reader.coordinate_frame == "ego_vehicle" else endpoint_sensor
        pose = torch.from_numpy(np.stack([world_to_ref @ world_local[slot] for slot in selected]).astype(np.float32))
        lidar_pose = torch.from_numpy(np.stack([world_to_ref @ world_sensor[slot] for slot in selected]).astype(np.float32))
        item = {
            "input": {
                "lidar_points_local": torch.cat(endpoint_local),
                "lidar_points_ego": torch.cat(endpoint_ego),
                "lidar_points_sensor": torch.cat(endpoint_sensor),
                "frame_counts": torch.tensor([len(points) for points in endpoint_sensor], dtype=torch.long),
                "pose": pose, "pose_lidar_to_ref": lidar_pose,
                "timestamps": normalized[list(selected)], "timestamps_sec": seconds[list(selected)],
                "window_duration_sec": duration,
            },
            "gt": {"cameras": cameras},
        }
        if self.window_labels is not None:
            item["meta"] = {"window_group": self.window_labels[index]}
        return item
