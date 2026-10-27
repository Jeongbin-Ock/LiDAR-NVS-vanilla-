"""Waymo v2 TOP sweeps for the two-input Dynamic 2DGS training contract."""
from __future__ import annotations

from functools import lru_cache
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from ..collate import dynamic_lidar4d_multiframe_collate_fn
from ..kitti360.reader import CAMERA_FROM_LIDAR, _transform_points
from ..kitti360.sampling import sample_window_frames, window_start_frames
from ...models.utils.camera import Camera
from .geometry import project_calibrated_panorama, top_range_to_sensor_points


@lru_cache(maxsize=8)
def _parquet_file(path):
    # Optional dependency: other dataset loaders must not require pyarrow.
    import pyarrow.parquet as pq
    return pq.ParquetFile(path)


def _timestamp_map(table):
    stamps = np.asarray(table["key.frame_timestamp_micros"].to_numpy(), dtype=np.int64)
    if len(set(stamps.tolist())) != len(stamps):
        raise ValueError("Waymo component contains duplicate timestamps")
    return {int(stamp): index for index, stamp in enumerate(stamps)}


@lru_cache(maxsize=1)
def _raw_tables(split_root, sequence):
    """Hold one uncached raw sequence per loader process, without writing it."""
    import pyarrow.parquet as pq
    root = Path(split_root)
    range_prefix = "[LiDARComponent].range_image_return1"
    pose_prefix = "[LiDARPoseComponent].range_image_return1"
    lidar = pq.read_table(root / "lidar" / f"{sequence}.parquet",
        columns=["key.frame_timestamp_micros", "key.laser_name",
                 f"{range_prefix}.values", f"{range_prefix}.shape"],
        filters=[("key.laser_name", "=", 1)])
    poses = pq.read_table(root / "lidar_pose" / f"{sequence}.parquet",
        columns=["key.frame_timestamp_micros", f"{pose_prefix}.values", f"{pose_prefix}.shape"])
    lidar_index, pose_index = _timestamp_map(lidar), _timestamp_map(poses)
    if set(lidar_index) != set(pose_index):
        raise ValueError(f"Waymo raw LiDAR and pixel-pose timestamps differ: {sequence}")
    return lidar, poses, lidar_index, pose_index


def _raw_metadata(split_root, sequence):
    """Read lightweight timestamps, vehicle poses and TOP calibration only."""
    import pyarrow.parquet as pq
    root = Path(split_root)
    for component in ("lidar", "lidar_pose", "vehicle_pose", "lidar_calibration"):
        if not (root / component / f"{sequence}.parquet").is_file():
            raise FileNotFoundError(root / component / f"{sequence}.parquet")
    stamp = "key.frame_timestamp_micros"
    lidar = pq.read_table(root / "lidar" / f"{sequence}.parquet",
        columns=[stamp, "key.laser_name", "[LiDARComponent].range_image_return1.shape"],
        filters=[("key.laser_name", "=", 1)])
    poses = pq.read_table(root / "lidar_pose" / f"{sequence}.parquet", columns=[stamp])
    vehicles = pq.read_table(root / "vehicle_pose" / f"{sequence}.parquet",
        columns=[stamp, "[VehiclePoseComponent].world_from_vehicle.transform"])
    calib = pq.read_table(root / "lidar_calibration" / f"{sequence}.parquet",
        filters=[("key.laser_name", "=", 1)])
    li, pi, vi = map(_timestamp_map, (lidar, poses, vehicles))
    if not li or set(li) != set(pi) or set(li) != set(vi) or len(calib) != 1:
        raise ValueError(f"Waymo raw LiDAR, pose or calibration mismatch: {sequence}")
    timestamps = np.asarray(sorted(li), dtype=np.int64)
    shape = lidar["[LiDARComponent].range_image_return1.shape"][li[int(timestamps[0])]].as_py()
    if len(shape) != 3 or shape[0] != 64 or shape[2] != 4:
        raise ValueError(f"Unexpected Waymo TOP range image shape: {sequence} {shape}")
    height, width = int(shape[0]), int(shape[1])
    extrinsic = np.asarray(calib["[LiDARCalibrationComponent].extrinsic.transform"][0].as_py(),
                           dtype=np.float64).reshape(4, 4)
    beams = calib["[LiDARCalibrationComponent].beam_inclination.values"][0].as_py()
    if not beams:
        lo = float(calib["[LiDARCalibrationComponent].beam_inclination.min"][0].as_py())
        hi = float(calib["[LiDARCalibrationComponent].beam_inclination.max"][0].as_py())
        beams = lo + (np.arange(height) + 0.5) / height * (hi - lo)
    beams = np.asarray(beams, dtype=np.float64)[::-1].copy()
    vehicle = np.stack([np.asarray(vehicles["[VehiclePoseComponent].world_from_vehicle.transform"]
        [vi[int(t)]].as_py(), dtype=np.float64).reshape(4, 4) for t in timestamps])
    return timestamps, vehicle, extrinsic, beams, height, width


class WaymoSequenceReader:
    def __init__(self, path, cfg, raw_split_root=None):
        self.path, self.cfg = Path(path), cfg
        self.raw_split_root = None if raw_split_root is None else str(raw_split_root)
        if self.raw_split_root is not None:
            (self.timestamps_us, self.world_from_vehicle, self.vehicle_from_lidar,
             self.inclination_rad, self.height, self.width) = _raw_metadata(self.raw_split_root, self.path.name)
            ready_frames = len(self.timestamps_us)
        else:
            ready = json.loads((self.path / "READY.json").read_text())
            if ready.get("version") != 1 or ready.get("sequence") != self.path.name:
                raise ValueError(f"Invalid Waymo cache manifest: {self.path}")
            with np.load(self.path / "metadata.npz", allow_pickle=False) as meta:
                self.timestamps_us = meta["timestamps_us"].copy()
                self.world_from_vehicle = meta["world_from_vehicle"].copy()
                self.vehicle_from_lidar = meta["vehicle_from_lidar"].copy()
                self.inclination_rad = meta["inclination_rad"].copy()
                self.height, self.width = int(meta["height"]), int(meta["width"])
            ready_frames = ready["frames"]
        self.timestamps_ns = self.timestamps_us * np.int64(1000)
        if (len(self.timestamps_us) != ready_frames or self.height != 64
                or self.world_from_vehicle.shape != (len(self.timestamps_us), 4, 4)
                or self.vehicle_from_lidar.shape != (4, 4)
                or self.inclination_rad.shape != (64,)
                or np.any(np.diff(self.timestamps_us) <= 0)):
            raise ValueError(f"Invalid Waymo cache metadata: {self.path}")
        if self.raw_split_root is None and not (self.path / "frames.parquet").is_file():
            raise FileNotFoundError(self.path / "frames.parquet")

    @lru_cache(maxsize=4)
    def points(self, index):
        if (self.path / "READY.json").is_file():
            parquet = _parquet_file(str(self.path / "frames.parquet"))
            if parquet.num_row_groups != len(self.timestamps_us):
                raise ValueError(f"Waymo cache row groups do not match timestamps: {self.path}")
            frame = parquet.read_row_group(int(index))
            if int(frame["timestamp_us"][0].as_py()) != int(self.timestamps_us[index]):
                raise ValueError(f"Waymo cache frame order mismatch: {self.path} {index}")
            def array(column):
                return frame[column][0].values.to_numpy(zero_copy_only=False).reshape(self.height, self.width)
            ranges, intensity = array("range"), array("intensity")
            pixel_pose = np.stack([array(f"pose_{i}") for i in range(6)], axis=-1)
        else:
            if self.raw_split_root is None:
                raise FileNotFoundError(f"Waymo cache is not ready: {self.path}")
            lidar, poses, lidar_idx, pose_idx = _raw_tables(self.raw_split_root, self.path.name)
            stamp = int(self.timestamps_us[index])
            li, pi = lidar_idx[stamp], pose_idx[stamp]
            rp, pp = "[LiDARComponent].range_image_return1", "[LiDARPoseComponent].range_image_return1"
            rshape, pshape = lidar[f"{rp}.shape"][li].as_py(), poses[f"{pp}.shape"][pi].as_py()
            if rshape != [self.height, self.width, 4] or pshape != [self.height, self.width, 6]:
                raise ValueError(f"Unexpected Waymo raw frame shape: {self.path} {stamp}")
            scan = lidar[f"{rp}.values"][li].values.to_numpy(zero_copy_only=False).reshape(rshape)
            pixel_pose = poses[f"{pp}.values"][pi].values.to_numpy(zero_copy_only=False).reshape(pshape)
            ranges, intensity = scan[..., 0], scan[..., 1]
        result = top_range_to_sensor_points(ranges, intensity, pixel_pose, self.inclination_rad,
                                            self.vehicle_from_lidar, self.world_from_vehicle[index])
        # Waymo intensity is ordinarily in [0, 1], with occasional much larger
        # returns. This model predicts normalized intensity, so clip explicitly.
        result[:, 3] = np.clip(result[:, 3], 0, float(self.cfg.intensity_max)) / float(self.cfg.intensity_max)
        return torch.from_numpy(np.ascontiguousarray(result))

    def camera(self, points, lidar_from_ref, timestamp, timestamp_sec):
        points = points[torch.linalg.norm(points[:, :3], dim=1) > float(self.cfg.gt_ego_radius)]
        depth, intensity = project_calibrated_panorama(points.numpy(), self.inclination_rad,
                                                      self.cfg.image_width, self.cfg.max_depth_m)
        return Camera(R=np.eye(3), T=np.zeros(3), vfov=tuple(self.cfg.vfov), hfov=tuple(self.cfg.hfov),
                      timestamp=float(timestamp), timestamp_sec=float(timestamp_sec),
                      resolution=(int(self.cfg.image_width), int(self.cfg.image_height)),
                      pts_depth=torch.from_numpy(depth), pts_intensity=torch.from_numpy(intensity),
                      points=points[:, :3],
                      viewmatrix=(CAMERA_FROM_LIDAR @ lidar_from_ref).astype(np.float32),
                      ring_to_elevation_deg=np.degrees(self.inclination_rad).tolist())


class WaymoDynamicDataset(Dataset):
    """Two source sweeps with middle-frame supervision and official scene splits."""

    collate_fn = staticmethod(dynamic_lidar4d_multiframe_collate_fn)

    def __init__(self, cfg, split="train", window_subset=None):
        if window_subset:
            raise ValueError("Waymo uses data.max_val_windows, not a compact-val manifest")
        if str(cfg.mode) != "nvs" or str(cfg.coordinate_frame) != "ego_vehicle":
            raise ValueError("Waymo dynamic training requires bbox-free ego_vehicle coordinates")
        if int(cfg.image_height) != 64 or float(cfg.intensity_max) <= 0:
            raise ValueError("Waymo TOP requires 64 calibrated rows and positive intensity_max")
        self.cfg, self.split = cfg, str(split)
        if self.split not in ("train", "training", "val", "validation"):
            raise ValueError(f"Unsupported Waymo split: {self.split}")
        is_train = self.split in ("train", "training")
        split_dir = "training" if is_train else "validation"
        root = Path(str(cfg.dataroot))
        allow_raw = bool(getattr(cfg, "allow_raw_fallback", False))
        raw_root = Path(str(cfg.raw_dataroot)) if allow_raw else None
        if allow_raw:
            names = sorted(p.stem for p in (raw_root / split_dir / "lidar").glob("*.parquet"))
            directories = [root / split_dir / name for name in names]
        else:
            directories = sorted(p.parent for p in (root / split_dir).glob("*/READY.json"))
        expected = int(cfg.expected_train_sequences if is_train else cfg.expected_val_sequences)
        if expected and len(directories) != expected:
            raise ValueError(f"Waymo {split_dir} cache has {len(directories)} sequences; expected {expected}")
        other = "validation" if is_train else "training"
        other_names = ({p.stem for p in (raw_root / other / "lidar").glob("*.parquet")}
                       if allow_raw else {p.parent.name for p in (root / other).glob("*/READY.json")})
        overlap = {p.name for p in directories} & other_names
        if overlap:
            raise ValueError(f"Waymo train/validation segment overlap: {sorted(overlap)[:3]}")
        self.is_train = is_train
        self.readers, self.index = {}, []
        self.minimum = float(cfg.dynamic_window.endpoint_min_sec)
        self.maximum = float(cfg.dynamic_window.endpoint_max_sec)
        self.middle_count = int(cfg.gt_middle_count)
        self.eval_seed = int(cfg.dynamic_window.eval_seed)
        max_gap = float(getattr(cfg, "max_sweep_gap_sec", 0.15))
        if not np.isfinite(max_gap) or max_gap <= 0:
            raise ValueError("Waymo max_sweep_gap_sec must be finite and positive")
        for path in directories:
            reader = WaymoSequenceReader(path, cfg, raw_root / split_dir if allow_raw else None)
            self.readers[path.name] = reader
            breaks = np.flatnonzero(np.diff(reader.timestamps_ns) > round(max_gap * 1e9)) + 1
            boundaries = [0, *breaks.tolist(), len(reader.timestamps_us)]
            for first, end in zip(boundaries[:-1], boundaries[1:]):
                segment = {"first_frame": first, "last_frame": end - 1}
                starts = window_start_frames(segment, reader.timestamps_ns, stride_sec=cfg.window_stride_sec,
                                             endpoint_min_sec=self.minimum, endpoint_max_sec=self.maximum)
                self.index.extend((path.name, first, end - 1, start) for start in starts)
        if not self.index:
            raise ValueError(f"No usable Waymo {split_dir} windows in {root}")
        limit = int(getattr(cfg, "max_val_windows", 0) or 0)
        if not is_train and limit > 0 and len(self.index) > limit:
            positions = np.linspace(0, len(self.index) - 1, limit, dtype=np.int64)
            self.index = [self.index[int(position)] for position in positions]
        if bool(getattr(cfg, "verbose", False)):
            print(f"[waymo-{split_dir}] {len(directories)} sequences, {len(self.index)} windows")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        name, first, last, start = self.index[index]
        reader = self.readers[name]
        segment = {"first_frame": first, "last_frame": last}
        if self.is_train:
            rng = random
        else:
            seed = f"{self.eval_seed}|{name}|{start}".encode()
            rng = random.Random(int.from_bytes(hashlib.blake2b(seed, digest_size=8).digest(), "big"))
        _, _, frames = sample_window_frames(segment, reader.timestamps_ns, start,
            endpoint_min_sec=self.minimum, endpoint_max_sec=self.maximum,
            gt_middle_count=self.middle_count, rng=rng)
        world_vehicle = [reader.world_from_vehicle[i] for i in frames]
        world_sensor = [pose @ reader.vehicle_from_lidar for pose in world_vehicle]
        ref_from_world = np.linalg.inv(world_vehicle[0])
        sensor = [reader.points(i) for i in frames]
        seconds = torch.tensor([(int(reader.timestamps_us[i]) - int(reader.timestamps_us[frames[0]])) / 1e6
                                for i in frames], dtype=torch.float32)
        duration = seconds[-1].clone()
        normalized = seconds / duration
        cameras = [reader.camera(points, np.linalg.inv(world_sensor[i]) @ np.linalg.inv(ref_from_world),
                                 normalized[i], seconds[i]) for i, points in enumerate(sensor)]
        selected = (0, len(frames) - 1)
        endpoint_sensor = [sensor[i][torch.linalg.norm(sensor[i][:, :3], dim=1) > float(self.cfg.ego_radius)]
                           for i in selected]
        endpoint_ego = [_transform_points(points, reader.vehicle_from_lidar) for points in endpoint_sensor]
        pose = torch.from_numpy(np.stack([ref_from_world @ world_vehicle[i] for i in selected]).astype(np.float32))
        lidar_pose = torch.from_numpy(np.stack([ref_from_world @ world_sensor[i] for i in selected]).astype(np.float32))
        return {"input": {"lidar_points_local": torch.cat(endpoint_ego), "lidar_points_ego": torch.cat(endpoint_ego),
                          "lidar_points_sensor": torch.cat(endpoint_sensor),
                          "frame_counts": torch.tensor([len(points) for points in endpoint_sensor], dtype=torch.long),
                          "pose": pose, "pose_lidar_to_ref": lidar_pose,
                          "timestamps": normalized[list(selected)], "timestamps_sec": seconds[list(selected)],
                          "window_duration_sec": duration},
                "gt": {"cameras": cameras}}
