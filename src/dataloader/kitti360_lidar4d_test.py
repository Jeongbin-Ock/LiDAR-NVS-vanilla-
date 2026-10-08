"""Inference-only KITTI-360 clips and held-out frames from LiDAR4D/GS-LiDAR.

The ten benchmark "scenes" are clips of drive 0000, not ten driving logs.
Targets match the official split; inputs are two available non-target sweeps.
Physical timing comes from nanosecond timestamps, not a presumed 10 Hz grid.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import torch
from scipy.spatial.transform import Rotation, Slerp
from torch.utils.data import Dataset

from ..kitti360.metadata import KITTI360_BENCHMARK_SCENES, read_timestamps_ns
from ..models_new.utils.camera import Camera, StackedPanoramaCamera
from .kitti360_hdl64e import (
    COLUMN_ASSIGNMENTS,
    KITTI360_HDL64E_APEX_HEIGHT_M,
    KITTI360_HDL64E_BLOCK_ORIGIN_Z_M,
    KITTI360_HDL64E_BLOCKS,
    KITTI360_HDL64E_ELEVATION_DEG,
    project_hdl64e_panorama,
)

# uniform: the official LiDAR4D/GS-LiDAR grid, every row rendered from the sensor origin.
# calibrated: laser rows, the unit's own beam angles, one render origin per emitter block.
CALIBRATED_PROJECTION = "hdl64e_kitti360_calibrated"
PANORAMA_PROJECTIONS = ("uniform", CALIBRATED_PROJECTION)


# KITTI IMU is forward/right/down; checkpoint ego/Utonia coordinates are FLU.
EGO_FROM_IMU = np.diag([1.0, -1.0, -1.0, 1.0])
# KITTI panorama: column=W/2 faces sensor +x, column increases toward -y.
CAMERA_FROM_LIDAR = np.array([
    [0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1],
], dtype=np.float64)


def _homogeneous(matrix):
    result = np.eye(4, dtype=np.float64)
    result[:3, :4] = np.asarray(matrix, dtype=np.float64).reshape(3, 4)
    return result


def kitti_panorama_pixel_indices(points, height, width, vfov, max_depth=80.0):
    """Return original point IDs, pixel IDs and ranges eligible for GT projection.

    Official LiDAR4D uniform grid seen from the sensor origin: rows split the
    VFOV evenly and columns the full turn. No hardware ring IDs exist.
    """
    points = np.asarray(points, dtype=np.float32)
    height, width = int(height), int(width)
    if height < 1 or width < 1:
        raise ValueError("Panorama height and width must be positive")
    xyz = points[:, :3]
    ranges = np.linalg.norm(xyz, axis=1)
    valid = np.isfinite(points).all(axis=1) & (ranges > 0) & (ranges < max_depth)
    point_ids = np.flatnonzero(valid)
    xyz, ranges = xyz[valid], ranges[valid]
    # Upcast angles to preserve the official Python scalar rounding convention.
    x, y, z = xyz.astype(np.float64).T
    beta = np.pi - np.arctan2(y, x)
    elevation = np.arctan2(z, np.hypot(x, y))
    rows = np.rint((np.radians(float(vfov[1])) - elevation)
                   / np.radians(float(vfov[1]) - float(vfov[0])) * height).astype(np.int64)
    columns = np.rint(beta / (2 * np.pi) * width).astype(np.int64)
    valid = (rows >= 0) & (rows < height) & (columns >= 0) & (columns < width)
    pixels = rows[valid] * width + columns[valid]
    return point_ids[valid], pixels, ranges[valid]


def project_kitti_panorama(points, height, width, vfov, max_depth=80.0):
    """Official LiDAR4D uniform spherical projection + nearest-return z-buffer.

    Equal-depth ties keep the first point. KITTI intensity remains normalized;
    no ring IDs are added.
    """
    points = np.asarray(points, dtype=np.float32)
    height, width = int(height), int(width)
    point_ids, pixels, ranges = kitti_panorama_pixel_indices(
        points, height, width, vfov, max_depth,
    )
    strength = points[point_ids, 3]
    order = np.argsort(ranges, kind="stable")
    _, first = np.unique(pixels[order], return_index=True)
    chosen = order[first]
    depth = np.zeros(height * width, dtype=np.float32)
    intensity = np.zeros_like(depth)
    depth[pixels[chosen]] = ranges[chosen]
    intensity[pixels[chosen]] = strength[chosen]
    return depth.reshape(1, height, width), intensity.reshape(1, height, width)


class KITTI360Poses:
    def __init__(self, path, timestamps_ns, missing_policy="previous"):
        values = np.atleast_2d(np.loadtxt(path, dtype=np.float64))
        if values.shape[1] != 13:
            raise ValueError(f"poses.txt needs frame_id + 12 entries: {path}")
        self.frames = values[:, 0].astype(np.int64)
        if np.any(np.diff(self.frames) <= 0):
            raise ValueError("Pose frame IDs must be strictly increasing")
        self.matrices = np.repeat(np.eye(4)[None], len(values), axis=0)
        self.matrices[:, :3, :] = values[:, 1:].reshape(-1, 3, 4)
        self.timestamps_ns = timestamps_ns
        self.policy = str(missing_policy)
        if self.policy not in {"previous", "interpolate", "error"}:
            raise ValueError("pose_missing_policy must be previous, interpolate, or error")

    def at(self, frame):
        position = int(np.searchsorted(self.frames, int(frame)))
        if position < len(self.frames) and self.frames[position] == frame:
            return self.matrices[position].copy(), "direct"
        if self.policy == "error" or position == 0 or position == len(self.frames):
            raise ValueError(f"No direct/bracketing optimized pose for frame {frame}")
        if self.policy == "previous":
            # Matches the official LiDAR4D KITTI360Loader.load_lidars fallback.
            return self.matrices[position - 1].copy(), f"previous:{self.frames[position - 1]}"
        left, right = self.frames[position - 1:position + 1]
        duration = int(self.timestamps_ns[right]) - int(self.timestamps_ns[left])
        alpha = (int(self.timestamps_ns[frame]) - int(self.timestamps_ns[left])) / duration
        result = np.eye(4)
        rotations = Rotation.from_matrix(self.matrices[position - 1:position + 1, :3, :3])
        result[:3, :3] = Slerp([0.0, 1.0], rotations)([alpha]).as_matrix()[0]
        result[:3, 3] = ((1 - alpha) * self.matrices[position - 1, :3, 3]
                         + alpha * self.matrices[position, :3, 3])
        return result, f"interpolated:{left}:{right}"


def _transform_points(points, matrix):
    output = points.clone()
    transform = torch.as_tensor(matrix, dtype=points.dtype)
    output[:, :3] = points[:, :3] @ transform[:3, :3].T + transform[:3, 3]
    return output


class KITTI360SequenceReader:
    """Shared sensor calibration, optimized poses and GT cameras for one drive."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.dataroot = Path(str(cfg.dataroot))
        self.sequence = str(getattr(cfg, "sequence", "2013_05_28_drive_0000_sync"))
        self.coordinate_frame = str(getattr(cfg, "coordinate_frame", "ego_vehicle"))
        if self.coordinate_frame not in {"ego_vehicle", "lidar_sensor"}:
            raise ValueError(f"Unsupported coordinate_frame: {self.coordinate_frame}")
        self.panorama_projection = str(getattr(cfg, "panorama_projection", "uniform"))
        if self.panorama_projection not in PANORAMA_PROJECTIONS:
            raise ValueError(f"panorama_projection must be one of {PANORAMA_PROJECTIONS}")
        self.row_elevation_deg = None
        self.column_assignment = None
        # GT-side crop of returns near the sensor, as the inputs (ego_radius)
        # and GS-LiDAR's KITTI-360 GT do. 0 keeps every return (LiDAR4D).
        self.gt_ego_radius = float(getattr(cfg, "gt_ego_radius", 0.0) or 0.0)
        if self.panorama_projection == CALIBRATED_PROJECTION:
            if int(cfg.image_height) != 64:
                raise ValueError(f"{self.panorama_projection} projection requires image_height=64")
            self.row_elevation_deg = tuple(float(value) for value in KITTI360_HDL64E_ELEVATION_DEG)
            self.column_assignment = str(getattr(cfg, "column_assignment", "firing_slot"))
            if self.column_assignment not in COLUMN_ASSIGNMENTS:
                raise ValueError(f"column_assignment must be one of {COLUMN_ASSIGNMENTS}")
        self.scan_dir = self.dataroot / "data_3d_raw" / self.sequence / "velodyne_points"
        self.timestamps_ns = read_timestamps_ns(self.scan_dir / "timestamps.txt")
        self.poses = KITTI360Poses(
            self.dataroot / "data_poses" / self.sequence / "poses.txt",
            self.timestamps_ns, getattr(cfg, "pose_missing_policy", "previous"),
        )
        calibration = self.dataroot / "calibration"
        lines = (calibration / "calib_cam_to_pose.txt").read_text().splitlines()
        cam_line = next(line.split(":", 1)[1] for line in lines if line.startswith("image_00:"))
        imu_from_cam = _homogeneous(np.fromstring(cam_line, sep=" "))
        lidar_from_cam = _homogeneous(np.loadtxt(calibration / "calib_cam_to_velo.txt"))
        self.imu_from_lidar = imu_from_cam @ np.linalg.inv(lidar_from_cam)
        # Preserve the calibrated KITTI GPS/IMU origin; only convert FRD to FLU.
        self.ego_from_lidar = EGO_FROM_IMU @ self.imu_from_lidar

    def scan_path(self, frame):
        return self.scan_dir / "data" / f"{int(frame):010d}.bin"

    @lru_cache(maxsize=16)
    def _load_points(self, frame):
        raw = np.fromfile(self.scan_path(frame), dtype=np.float32)
        if raw.size % 4:
            raise ValueError(f"Not a KITTI-360 XYZI scan: {frame}")
        points = raw.reshape(-1, 4)
        points = points[np.isfinite(points).all(axis=1)]
        return torch.from_numpy(np.ascontiguousarray(points))

    def _world_transforms(self, frames):
        world_from_imu = [self.poses.at(frame)[0] for frame in frames]
        world_from_lidar = [matrix @ self.imu_from_lidar for matrix in world_from_imu]
        world_from_local = ([matrix @ EGO_FROM_IMU for matrix in world_from_imu]
                            if self.coordinate_frame == "ego_vehicle" else world_from_lidar)
        world_to_ref = np.linalg.inv(world_from_local[0])
        return world_from_lidar, world_from_local, world_to_ref

    def _projection_protocol(self):
        protocol = {
            "mode": self.panorama_projection,
            "row_elevation_deg": (list(self.row_elevation_deg)
                                  if self.row_elevation_deg is not None else None),
            "hardware_ring_ids_available": False,
        }
        if self.panorama_projection != CALIBRATED_PROJECTION:
            protocol.update({
                "device_beam_calibration_available": False,
                "angular_support": "finite returns below max_depth_m inside the uniform VFOV rows",
            })
            return protocol
        protocol.update({
            "rows": "one per laser; laser = best-fitting per-laser cone z = rho*tan(elevation) + apex",
            "device_beam_calibration": "fitted from 722 KITTI-360 scans (kitti360_hdl64e.py)",
            "laser_apex_height_m": KITTI360_HDL64E_APEX_HEIGHT_M.tolist(),
            "blocks": [list(block) for block in KITTI360_HDL64E_BLOCKS],
            "block_origin_z_m": list(KITTI360_HDL64E_BLOCK_ORIGIN_Z_M),
            "range": "distance from the block origin; GT keeps returns < max_depth_m",
            "column_assignment": self.column_assignment,
            "angular_support": "every finite return below max_depth_m (no FOV mask)",
        })
        return protocol

    def _gt_camera(self, points, lidar_from_ref, timestamp, timestamp_sec):
        """GT panorama Camera for one sweep; ``lidar_from_ref`` maps ref to sensor."""
        width, height = int(self.cfg.image_width), int(self.cfg.image_height)
        max_depth = float(getattr(self.cfg, "max_depth_m", 80.0))
        common = dict(R=np.eye(3), T=np.zeros(3), vfov=tuple(self.cfg.vfov), hfov=tuple(self.cfg.hfov),
                      timestamp=timestamp, timestamp_sec=timestamp_sec)
        view = (CAMERA_FROM_LIDAR @ lidar_from_ref).astype(np.float32)
        if self.gt_ego_radius > 0:
            points = points[torch.linalg.norm(points[:, :3], dim=1) > self.gt_ego_radius]
        if self.panorama_projection != CALIBRATED_PROJECTION:
            depth, intensity = project_kitti_panorama(
                points.numpy(), height, width, self.cfg.vfov, max_depth,
            )
            return Camera(**common, resolution=(width, height),
                          pts_depth=torch.from_numpy(depth), pts_intensity=torch.from_numpy(intensity),
                          points=points[:, :3], viewmatrix=view)
        depth, intensity, stats = project_hdl64e_panorama(
            points.numpy(), width, max_depth, self.column_assignment,
        )
        # Each block renders the full panorama from its own emitter origin and
        # keeps only its rows, so the rasterizer's handling of Gaussians beyond
        # the table edge matches a single full-panorama render.
        blocks, kept_rows, row_origin = [], [], []
        for (start, end), origin_z in zip(KITTI360_HDL64E_BLOCKS, KITTI360_HDL64E_BLOCK_ORIGIN_Z_M):
            block_from_lidar = np.eye(4)
            block_from_lidar[2, 3] = -origin_z
            blocks.append(Camera(
                **common, resolution=(width, height),
                viewmatrix=(CAMERA_FROM_LIDAR @ block_from_lidar @ lidar_from_ref).astype(np.float32),
                ring_to_elevation_deg=self.row_elevation_deg,
            ))
            kept_rows.append((start, end))
            row_origin += [CAMERA_FROM_LIDAR[:3, :3] @ np.array([0.0, 0.0, origin_z])] * (end - start)
        camera = StackedPanoramaCamera(
            blocks, np.asarray(row_origin), kept_rows, **common, resolution=(width, height),
            pts_depth=torch.from_numpy(depth), pts_intensity=torch.from_numpy(intensity),
            points=points[:, :3], viewmatrix=view, ring_to_elevation_deg=self.row_elevation_deg,
        )
        camera.projection_stats = stats
        return camera


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
