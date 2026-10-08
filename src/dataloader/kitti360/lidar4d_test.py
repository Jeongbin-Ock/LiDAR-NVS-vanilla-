"""Inference-only KITTI-360 clips and held-out frames from LiDAR4D/GS-LiDAR.

The ten benchmark "scenes" are clips of drive 0000, not ten driving logs.
Targets match the official split; inputs are two available non-target sweeps.
Physical timing comes from nanosecond timestamps, not a presumed 10 Hz grid.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np
import torch
from torch.utils.data import Dataset

from .metadata import KITTI360_BENCHMARK_SCENES
from .reader import KITTI360SequenceReader, _transform_points


class LiDAR4DKITTI360TestDataset(KITTI360SequenceReader, Dataset):
    dataset_name = "kitti360"
    target_cam_index = 1

    def __init__(self, cfg, split="test", input_offsets_sec=(-0.5, 0.5)):
        if split not in {"test", "val"}:
            raise ValueError("KITTI360 LiDAR4D loader is inference-only (test/val)")
        if str(getattr(cfg, "mode", "nvs")) != "nvs":
            raise ValueError("KITTI360 inference uses bbox-free data.mode=nvs")
        if str(getattr(cfg, "sequence", "2013_05_28_drive_0000_sync")) != "2013_05_28_drive_0000_sync":
            raise ValueError("Official LiDAR4D benchmark clips belong to drive 0000")
        super().__init__(cfg)
        self.split = split
        self.input_offsets_sec = tuple(map(float, input_offsets_sec))
        if (len(self.input_offsets_sec) != 2 or not all(np.isfinite(self.input_offsets_sec))
                or not self.input_offsets_sec[0] < 0 < self.input_offsets_sec[1]):
            raise ValueError("Two finite input offsets must straddle the target")
        self.index_meta, self.windows = [], []
        scenes = [int(value) for value in getattr(cfg, "benchmark_scenes", KITTI360_BENCHMARK_SCENES)]
        if not scenes or len(scenes) != len(set(scenes)):
            raise ValueError("benchmark_scenes must contain unique scene IDs")
        for scene_id in scenes:
            if scene_id not in KITTI360_BENCHMARK_SCENES:
                raise ValueError(f"Unknown official benchmark scene: {scene_id}")
            first, last, targets, category = KITTI360_BENCHMARK_SCENES[scene_id]
            candidates = np.array([frame for frame in range(first, last + 1)
                                   if frame not in targets and self.scan_path(frame).is_file()])
            for target in targets:
                inputs = []
                for offset in self.input_offsets_sec:
                    desired = int(self.timestamps_ns[target]) + round(offset * 1e9)
                    side = candidates[candidates < target] if offset < 0 else candidates[candidates > target]
                    if not len(side) or desired < self.timestamps_ns[first] or desired > self.timestamps_ns[last]:
                        raise ValueError(f"Scene {scene_id}: offsets exceed clip bounds for {target}")
                    inputs.append(int(side[np.argmin(np.abs(self.timestamps_ns[side] - desired))]))
                frame_ids = (inputs[0], target, inputs[1])
                pose_sources = []
                for frame in frame_ids:
                    if not self.scan_path(frame).is_file():
                        raise FileNotFoundError(self.scan_path(frame))
                    _, origin = self.poses.at(frame)
                    pose_sources.append(origin)
                self.windows.append(frame_ids)
                self.index_meta.append({
                    "seq_name": f"seq_{first:04d}_{last:04d}", "scene_id": scene_id,
                    "scene": self.sequence, "dataset_split": category,
                    "target_s": (int(self.timestamps_ns[target]) - int(self.timestamps_ns[first])) / 1e9,
                    "target_frame_id": target, "input_frame_ids": inputs,
                    "actual_input_offsets_sec": [(int(self.timestamps_ns[f]) - int(self.timestamps_ns[target])) / 1e9 for f in inputs],
                    "pose_sources": pose_sources,
                })
        self.protocol = {
            "name": "LiDAR4D/GS-LiDAR KITTI-360 held-out target frames",
            "scene_ids": scenes, "num_windows": len(self.windows),
            "input_mode": "two nearest non-held-out endpoint sweeps (feed-forward)",
            "timing": "actual KITTI-360 scan timestamps",
            "pose_missing_policy": self.poses.policy,
            "gt_ego_radius_m": self.gt_ego_radius,
            "vertical_projection": self._projection_protocol(),
            "height": int(cfg.image_height), "width": int(cfg.image_width),
            "nominal_sensor_hz": 10.0,
            "measured_sequence_hz": (len(self.timestamps_ns) - 1) * 1e9 / float(self.timestamps_ns[-1] - self.timestamps_ns[0]),
        }
        self._dynamic_boxes = None
        if bool(getattr(cfg, "verbose", False)):
            print(f"[kitti360-test] {len(scenes)} clips, {len(self)} targets, "
                  f"{cfg.image_height}x{cfg.image_width}; {self.protocol['measured_sequence_hz']:.6f}Hz; "
                  f"pose_missing_policy={self.poses.policy}")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        frames = self.windows[index]
        world_from_lidar, world_from_local, world_to_ref = self._world_transforms(frames)
        pose = torch.from_numpy(np.stack([world_to_ref @ matrix for matrix in world_from_local]).astype(np.float32))
        lidar_pose = torch.from_numpy(np.stack([world_to_ref @ matrix for matrix in world_from_lidar]).astype(np.float32))
        sensor = [self._load_points(frame) for frame in frames]
        ego = [_transform_points(points, self.ego_from_lidar) for points in sensor]
        local = ego if self.coordinate_frame == "ego_vehicle" else sensor
        reference = [_transform_points(points, matrix) for points, matrix in zip(sensor, lidar_pose)]
        # Subtract integer timestamps BEFORE converting to float, avoiding epoch precision loss.
        seconds = torch.tensor([(int(self.timestamps_ns[f]) - int(self.timestamps_ns[frames[0]])) / 1e9
                                for f in frames], dtype=torch.float32)
        duration = seconds[-1].clone()
        normalized = seconds / duration
        cameras = [
            self._gt_camera(points, np.linalg.inv(world_from_lidar[slot]) @ np.linalg.inv(world_to_ref),
                            float(normalized[slot]), float(seconds[slot]))
            for slot, points in enumerate(sensor)
        ]
        selected = (0, 2)
        radius = float(getattr(self.cfg, "ego_radius", 0.0))
        masks = [torch.linalg.norm(sensor[slot][:, :3], dim=1) > radius for slot in selected]

        def cat_selected(values):
            return torch.cat([values[slot][mask] for slot, mask in zip(selected, masks)])

        counts = torch.tensor([int(mask.sum()) for mask in masks], dtype=torch.long)
        gt_counts = torch.tensor([len(points) for points in sensor], dtype=torch.long)
        empty_boxes = [torch.empty((0, 7), dtype=torch.float32) for _ in frames]
        empty_ids = [torch.empty(0, dtype=torch.long) for _ in frames]
        return {
            "input": {
                "lidar_points": cat_selected(reference), "lidar_points_local": cat_selected(local),
                "lidar_points_ego": cat_selected(ego), "lidar_points_sensor": cat_selected(sensor),
                "lidar_ring": torch.full((int(counts.sum()),), -1, dtype=torch.long),
                "point_iid": torch.full((int(counts.sum()),), -1, dtype=torch.long),
                "frame_counts": counts, "bbox": [empty_boxes[i] for i in selected],
                "bbox_instance_ids": [empty_ids[i] for i in selected],
                "pose": pose[list(selected)], "pose_lidar_to_ref": lidar_pose[list(selected)],
                "timestamps": normalized[list(selected)], "timestamps_sec": seconds[list(selected)],
                "window_duration_sec": duration, "input_indices": torch.tensor(selected),
                "input_window_indices": torch.tensor([0, frames[2] - frames[0]]),
            },
            "gt": {
                "gt_lidar_points": torch.cat(reference), "gt_lidar_points_ego": torch.cat(ego),
                "gt_lidar_points_sensor": torch.cat(sensor),
                "gt_lidar_ring": torch.full((int(gt_counts.sum()),), -1, dtype=torch.long),
                "gt_point_iid": torch.full((int(gt_counts.sum()),), -1, dtype=torch.long),
                "gt_frame_counts": gt_counts, "gt_bbox": empty_boxes, "gt_bbox_instance_ids": empty_ids,
                "gt_pose": pose, "gt_pose_lidar_to_ref": lidar_pose,
                "gt_timestamps": normalized, "gt_timestamps_sec": seconds,
                "window_duration_sec": duration, "gt_cameras": cameras,
                "window_indices": torch.tensor([frame - frames[0] for frame in frames]),
            },
        }

    def target_boxes_ref(self, index):
        """Exact-time dynamic GT boxes for visualization only; never model input.

        Missing annotation is left missing, rather than inventing a GT track.
        Static bounding primitives are shared world geometry, not motion labels.
        """
        if self._dynamic_boxes is None:
            self._dynamic_boxes = {}
            path = self.dataroot / "data_3d_bboxes" / "train" / f"{self.sequence}.xml"
            if path.exists():
                for _, obj in ET.iterparse(path, events=("end",)):
                    if not obj.tag.startswith("object"):
                        continue
                    frame = int(obj.findtext("timestamp"))
                    if frame >= 0:
                        transform = np.fromstring(obj.findtext("transform/data"), sep=" ").reshape(4, 4)
                        vertices = np.fromstring(obj.findtext("vertices/data"), sep=" ").reshape(-1, 3)
                        self._dynamic_boxes.setdefault(frame, []).append((transform, vertices))
                    obj.clear()
        frames = self.windows[index]
        _, _, world_to_ref = self._world_transforms(frames)
        boxes = []
        for transform, vertices in self._dynamic_boxes.get(frames[1], []):
            local_min, local_max = vertices.min(axis=0), vertices.max(axis=0)
            box_to_ref = world_to_ref @ transform
            center = box_to_ref[:3, :3] @ ((local_min + local_max) / 2) + box_to_ref[:3, 3]
            extent = (local_max - local_min) * np.linalg.norm(box_to_ref[:3, :3], axis=0)
            yaw = np.arctan2(box_to_ref[1, 0], box_to_ref[0, 0])
            boxes.append([*center, extent[1], extent[0], extent[2], yaw])
        return np.asarray(boxes, dtype=np.float32).reshape(-1, 7)
