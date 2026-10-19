"""Inference-only H3D windows for zero-shot evaluation of a KITTI-360 checkpoint.

Every scenario of the selected official splits whose first-to-last frame span
reaches ``min_duration_sec`` contributes one target per whole second
T = 1, 2, ... : the target is the sweep nearest to T s after the scenario's
first sweep, and the two inputs are the nearest other sweeps to
target time + ``input_offsets_sec`` (default +-0.1 s, the 0.2 s gap). A target
is kept only when every chosen sweep lies within ``max_frame_time_error_sec``
of the requested time, so the last whole second needs a sweep after it.

H3D and KITTI-360 both carry a Velodyne HDL-64E with the LiDAR frame
x forward / y left / z up, so sweeps, GT panoramas and the uniform 66x1030
grid follow the KITTI-360 LiDAR4D path (``kitti360/reader.py``). H3D has no
LiDAR-to-vehicle calibration; ``data.ego_from_lidar`` supplies the 4x4 used
to build the checkpoint's ego frame (config/h3d_inference.md).
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from ...models.utils.camera import Camera
from ..kitti360.reader import CAMERA_FROM_LIDAR, _transform_points, project_kitti_panorama
from .metadata import H3D_SPLITS, H3DScenario, h3d_scan, read_labels, read_split


class LiDAR4DH3DTestDataset(Dataset):
    dataset_name = "h3d"
    target_cam_index = 1

    def __init__(self, cfg, split="test", input_offsets_sec=(-0.1, 0.1)):
        if str(getattr(cfg, "mode", "nvs")) != "nvs":
            raise ValueError("H3D inference uses bbox-free data.mode=nvs")
        if str(getattr(cfg, "coordinate_frame", "ego_vehicle")) != "ego_vehicle":
            raise ValueError("H3D inference builds the checkpoint's ego_vehicle frame")
        if str(getattr(cfg, "panorama_projection", "uniform")) != "uniform":
            raise ValueError("H3D inference renders the uniform LiDAR4D grid only")
        self.cfg = cfg
        self.root = str(cfg.dataroot)
        self.input_offsets_sec = tuple(map(float, input_offsets_sec))
        if (len(self.input_offsets_sec) != 2 or not all(np.isfinite(self.input_offsets_sec))
                or not self.input_offsets_sec[0] < 0 < self.input_offsets_sec[1]):
            raise ValueError("Two finite input offsets must straddle the target")
        self.ego_from_lidar = np.asarray(cfg.ego_from_lidar, dtype=np.float64).reshape(4, 4)
        if not np.allclose(self.ego_from_lidar[3], [0, 0, 0, 1]):
            raise ValueError("data.ego_from_lidar must be a rigid 4x4 transform")
        self.intensity_divisor = float(getattr(cfg, "intensity_divisor", 256.0))
        self.ego_radius = float(getattr(cfg, "ego_radius", 0.0) or 0.0)
        self.gt_ego_radius = float(getattr(cfg, "gt_ego_radius", 0.0) or 0.0)
        self.input_max_range = float(getattr(cfg, "input_max_range_m", 0.0) or 0.0)
        min_duration = float(getattr(cfg, "min_duration_sec", 5.0))
        tolerance = float(getattr(cfg, "max_frame_time_error_sec", 0.05))

        splits = getattr(cfg, "splits", None) or [split]
        splits = [str(value) for value in splits]
        if any(value not in H3D_SPLITS for value in splits):
            raise ValueError(f"data.splits must be drawn from {H3D_SPLITS}")
        requested = getattr(cfg, "scenarios", None)
        requested = None if requested is None else {str(value) for value in requested}

        self.scenarios, self.windows, self.index_meta, skipped = {}, [], [], []
        for split_name in splits:
            for name in read_split(self.root, split_name):
                if requested is not None and name not in requested:
                    continue
                scenario = H3DScenario(self.root, name)
                if scenario.duration_sec < min_duration:
                    skipped.append((name, split_name, scenario.duration_sec))
                    continue
                self.scenarios[name] = scenario
                for second in range(1, int(np.floor(scenario.duration_sec)) + 1):
                    window = self._window(scenario, second, tolerance)
                    if window is None:
                        continue
                    frames, actual = window
                    self.windows.append((name, frames))
                    self.index_meta.append({
                        "seq_name": name, "scene": name, "dataset_split": split_name,
                        "target_second": second,
                        "target_s": scenario.seconds(frames[1]),
                        "target_frame_id": frames[1], "input_frame_ids": [frames[0], frames[2]],
                        "actual_input_offsets_sec": actual,
                        "scenario_duration_sec": scenario.duration_sec,
                    })
        if requested is not None and requested - set(self.scenarios) - {name for name, _, _ in skipped}:
            raise ValueError(f"Unknown H3D scenarios: {sorted(requested - set(self.scenarios))}")
        self.skipped_short = [{"scenario": name, "split": split_name, "duration_sec": duration}
                              for name, split_name, duration in skipped]
        self.protocol = {
            "name": "H3D zero-shot: one target per whole second of each scenario",
            "splits": splits, "min_duration_sec": min_duration,
            "num_scenarios": len(self.scenarios), "num_windows": len(self.windows),
            "skipped_short_scenarios": self.skipped_short,
            "input_mode": "two nearest non-target sweeps at target time + input_offsets_sec",
            "timing": "midpoint of the ADMA GPS samples bundled with each sweep",
            "max_frame_time_error_sec": tolerance,
            "pose": "odom_NNN.txt LiDAR pose, Rz(yaw) Ry(pitch) Rx(roll)",
            "ego_from_lidar": self.ego_from_lidar.tolist(),
            "intensity": f"PLY radius / {self.intensity_divisor:g}",
            "input_max_range_m": self.input_max_range,
            "gt_ego_radius_m": self.gt_ego_radius,
            "vertical_projection": {"mode": "uniform", "hardware_ring_ids_available": False},
            "height": int(cfg.image_height), "width": int(cfg.image_width),
        }
        if bool(getattr(cfg, "verbose", False)):
            print(f"[h3d-test] splits={splits}: {len(self.scenarios)} scenarios >= {min_duration:g}s "
                  f"({len(skipped)} shorter skipped), {len(self.windows)} targets, "
                  f"inputs at {self.input_offsets_sec}s, {cfg.image_height}x{cfg.image_width}")

    def _window(self, scenario, second, tolerance):
        target = scenario.nearest_frame(second)
        target_time = scenario.seconds(target)
        if abs(target_time - second) > tolerance:
            return None
        inputs, actual = [], []
        for offset in self.input_offsets_sec:
            frame = scenario.nearest_frame(target_time + offset, exclude=(target,))
            delta = scenario.seconds(frame) - target_time
            if abs(delta - offset) > tolerance or np.sign(delta) != np.sign(offset):
                return None
            inputs.append(frame)
            actual.append(delta)
        return (inputs[0], target, inputs[1]), actual

    def __len__(self):
        return len(self.windows)

    def _load_points(self, scenario, frame):
        points, _ = h3d_scan(scenario.path("ply", frame), intensity_divisor=self.intensity_divisor)
        return torch.from_numpy(points)

    def _world_transforms(self, scenario, frames):
        world_from_lidar = [scenario.pose(frame) for frame in frames]
        lidar_from_ego = np.linalg.inv(self.ego_from_lidar)
        world_from_local = [matrix @ lidar_from_ego for matrix in world_from_lidar]
        world_to_ref = np.linalg.inv(world_from_local[0])
        return world_from_lidar, world_from_local, world_to_ref

    def _gt_camera(self, points, lidar_from_ref, timestamp, timestamp_sec):
        """Uniform LiDAR4D panorama seen from the sensor origin, as for KITTI-360."""
        width, height = int(self.cfg.image_width), int(self.cfg.image_height)
        max_depth = float(getattr(self.cfg, "max_depth_m", 80.0))
        if self.gt_ego_radius > 0:
            points = points[torch.linalg.norm(points[:, :3], dim=1) > self.gt_ego_radius]
        depth, intensity = project_kitti_panorama(points.numpy(), height, width, self.cfg.vfov, max_depth)
        return Camera(
            R=np.eye(3), T=np.zeros(3), vfov=tuple(self.cfg.vfov), hfov=tuple(self.cfg.hfov),
            timestamp=timestamp, timestamp_sec=timestamp_sec, resolution=(width, height),
            pts_depth=torch.from_numpy(depth), pts_intensity=torch.from_numpy(intensity),
            points=points[:, :3], viewmatrix=(CAMERA_FROM_LIDAR @ lidar_from_ref).astype(np.float32),
        )

    def __getitem__(self, index):
        name, frames = self.windows[index]
        scenario = self.scenarios[name]
        world_from_lidar, world_from_local, world_to_ref = self._world_transforms(scenario, frames)
        pose = torch.from_numpy(np.stack([world_to_ref @ matrix for matrix in world_from_local]).astype(np.float32))
        lidar_pose = torch.from_numpy(np.stack([world_to_ref @ matrix for matrix in world_from_lidar]).astype(np.float32))
        sensor = [self._load_points(scenario, frame) for frame in frames]
        ego = [_transform_points(points, self.ego_from_lidar) for points in sensor]
        reference = [_transform_points(points, matrix) for points, matrix in zip(sensor, lidar_pose)]
        # Integer microseconds are differenced before the float conversion.
        seconds = torch.tensor([(int(scenario.time_us[scenario.index(f)]) - int(scenario.time_us[scenario.index(frames[0])])) / 1e6
                                for f in frames], dtype=torch.float32)
        duration = seconds[-1].clone()
        normalized = seconds / duration
        cameras = [
            self._gt_camera(points, np.linalg.inv(world_from_lidar[slot]) @ np.linalg.inv(world_to_ref),
                            float(normalized[slot]), float(seconds[slot]))
            for slot, points in enumerate(sensor)
        ]
        selected = (0, 2)
        masks = []
        for slot in selected:
            ranges = torch.linalg.norm(sensor[slot][:, :3], dim=1)
            mask = ranges > self.ego_radius
            if self.input_max_range > 0:
                mask &= ranges < self.input_max_range
            masks.append(mask)

        def cat_selected(values):
            return torch.cat([values[slot][mask] for slot, mask in zip(selected, masks)])

        counts = torch.tensor([int(mask.sum()) for mask in masks], dtype=torch.long)
        gt_counts = torch.tensor([len(points) for points in sensor], dtype=torch.long)
        empty_boxes = [torch.empty((0, 7), dtype=torch.float32) for _ in frames]
        empty_ids = [torch.empty(0, dtype=torch.long) for _ in frames]
        return {
            "input": {
                "lidar_points": cat_selected(reference), "lidar_points_local": cat_selected(ego),
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
        """Target-frame dynamic boxes for visualization only; never model input.

        H3D rows are (x, y, z centre, w, l, h, yaw) with the length along the
        box's local y axis, so the heading of motion is yaw + pi/2.
        """
        name, frames = self.windows[index]
        scenario = self.scenarios[name]
        world_from_lidar, _, world_to_ref = self._world_transforms(scenario, frames)
        ref_from_lidar = world_to_ref @ world_from_lidar[1]
        boxes = []
        for row in read_labels(scenario.path("labels", frames[1])):
            if row[2] != "dynamic":
                continue
            x, y, z, width, length, height, yaw = row[3:10]
            center = ref_from_lidar[:3, :3] @ np.array([x, y, z]) + ref_from_lidar[:3, 3]
            heading = yaw + np.pi / 2 + np.arctan2(ref_from_lidar[1, 0], ref_from_lidar[0, 0])
            boxes.append([*center, width, length, height, heading])
        return np.asarray(boxes, dtype=np.float32).reshape(-1, 7)
