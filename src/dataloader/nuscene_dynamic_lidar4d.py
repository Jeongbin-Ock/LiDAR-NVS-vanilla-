"""LiDAR4D-held-out nuScenes loader for the Dynamic 2DGS V11 family.

The legacy :mod:`src.dataloader.nuscene` loader serves both bbox-rigid and
Dynamic experiments, so it still builds box trajectories, point-instance
labels, reference-frame point copies, and a large compatibility payload.  V11
uses none of those.  This loader keeps only the two-endpoint physical-time
contract consumed by V11 and the target cameras consumed by the renderer.

Sampling contract
-----------------
The flat index deliberately keeps the legacy keyframe starts and their nominal
``pair_kf_stride`` endpoint.  The nominal endpoint is used only as a stable
compact-validation address.  On every training access, the actual endpoint is
drawn uniformly from the available 0.1-second-grid sweeps whose *actual*
timestamp is 0.2--1.2 seconds after the start.  Near a scene boundary the
candidate set is truncated to the available sweeps; no endpoint is clamped or
duplicated.  Validation/test temporal draws are pseudo-random but stable per
start anchor, so the supervised timestamps do not change between epochs.

Up to ``data.gt_middle_count`` (three in the shipped V11 configs) interior
sweeps are sampled without replacement.  Both endpoints are always included
among the render targets, so they continue to receive reconstruction loss.
"""
from __future__ import annotations

import hashlib
import os
import random

import numpy as np
import torch
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from torch.utils.data import Dataset

from . import compact_val
from .nuscenes_frames import (
    COORDINATE_FRAME_EGO,
    ego_to_world,
    reference_to_sensor as make_reference_to_sensor,
    resolve_coordinate_frame,
    sensor_to_ego,
    sensor_to_world,
    transform_points,
)
from ..models_new import utonia
from ..models_new.utils.camera import Camera


US_PER_SEC = 1_000_000
NUSCENES_SWEEP_US = 50_000
DEFAULT_SAMPLE_GAP_US = 100_000
DEFAULT_WINDOW_US = 1_000_000
DEFAULT_NOVEL_VIEW_COUNT = 3
LIDAR4D_TRAIN_SEQUENCE_SCENES = {
    "seq_1600_1650": "scene-0757",
    "seq_3180_3230": "scene-1094",
}
DEFAULT_TRAIN_EXCLUDED_SCENES = tuple(LIDAR4D_TRAIN_SEQUENCE_SCENES.values())


def _cfg_get(cfg, key: str, default=None):
    return getattr(cfg, key, default)


def _seconds_to_us(value: float, name: str) -> int:
    seconds = float(value)
    if not np.isfinite(seconds) or seconds <= 0.0:
        raise ValueError(f"{name} must be a positive finite number of seconds")
    return int(round(seconds * US_PER_SEC))


def _resolve_nuscenes_version(cfg, split: str) -> str:
    test_split = _cfg_get(cfg, "test_split", "test")
    if split == test_split or split == "test":
        return _cfg_get(cfg, "test_version", _cfg_get(cfg, "version", "v1.0-test"))
    return _cfg_get(cfg, "version", "v1.0-trainval")


def relative_time_coordinates(timestamps_us, start_us, end_us):
    """Return normalized time, relative seconds, and physical pair duration."""
    span_us = int(end_us) - int(start_us)
    if span_us <= 0:
        raise ValueError("dynamic endpoint timestamps must be strictly increasing")
    normalized = torch.tensor(
        [(int(timestamp) - int(start_us)) / span_us for timestamp in timestamps_us],
        dtype=torch.float32,
    )
    relative_sec = torch.tensor(
        [
            (int(timestamp) - int(start_us)) / US_PER_SEC
            for timestamp in timestamps_us
        ],
        dtype=torch.float32,
    )
    duration_sec = torch.tensor(span_us / US_PER_SEC, dtype=torch.float32)
    return normalized, relative_sec, duration_sec


def _sensor_to_world(nusc, sample_data_token: str) -> np.ndarray:
    return sensor_to_world(nusc, sample_data_token)


def _load_points(nusc, dataroot: str, sample_data_token: str, ego_radius: float):
    """Load normalized XYZI plus the raw nuScenes ring index."""
    sd = nusc.get("sample_data", sample_data_token)
    raw = np.fromfile(os.path.join(dataroot, sd["filename"]), dtype=np.float32)
    if raw.size % 5 == 0:
        scan5 = raw.reshape(-1, 5)
        scan = scan5[:, :4]
        ring = scan5[:, 4].astype(np.int64)
    elif raw.size % 4 == 0:
        scan = raw.reshape(-1, 4)
        ring = np.full((scan.shape[0],), -1, dtype=np.int64)
    else:
        raise ValueError(f"Unexpected point cloud size {raw.size}")

    points = torch.from_numpy(np.ascontiguousarray(scan))
    lidar_ring = torch.from_numpy(np.ascontiguousarray(ring))
    points[:, 3] = points[:, 3] / 255.0
    if ego_radius > 0.0:
        keep = torch.linalg.norm(points[:, :3], dim=1) > float(ego_radius)
        points = points[keep].contiguous()
        lidar_ring = lidar_ring[keep].contiguous()
    return points, lidar_ring


def eligible_endpoint_indices(
    frames,
    start_idx: int,
    *,
    sample_hop: int,
    min_duration_us: int,
    max_duration_us: int,
) -> list[int]:
    """Available aligned endpoint sweeps inside the requested timestamp range."""
    start_timestamp = int(frames[start_idx][1])
    candidates = []
    for end_idx in range(start_idx + sample_hop, len(frames), sample_hop):
        duration_us = int(frames[end_idx][1]) - start_timestamp
        if duration_us > max_duration_us:
            break
        if duration_us >= min_duration_us:
            candidates.append(end_idx)
    return candidates


def sample_target_indices(
    start_idx: int,
    end_candidates: list[int],
    *,
    sample_hop: int,
    novel_view_count: int,
    rng,
) -> tuple[int, list[int], list[int]]:
    """Draw one endpoint and up to ``novel_view_count`` interior targets.

    Returns ``(end_idx, novel_indices, used_indices)``.  ``used_indices`` is
    chronological and always contains both endpoint indices.
    """
    if not end_candidates:
        raise ValueError("dynamic window has no eligible endpoint sweep")
    end_idx = int(rng.choice(end_candidates))
    interior = list(range(start_idx + sample_hop, end_idx, sample_hop))
    count = min(int(novel_view_count), len(interior))
    novel = sorted(rng.sample(interior, count)) if count else []
    return end_idx, novel, [start_idx, *novel, end_idx]


class NuScenesDynamicLiDAR4DDataset(Dataset):
    """V11 random-duration loader with LiDAR4D train scenes held out."""

    def __init__(self, cfg, split: str, window_subset: str = None):
        self.cfg = cfg
        self.dataroot = cfg.dataroot
        self.split = split
        self.verbose = bool(_cfg_get(cfg, "verbose", False))
        self.version = _resolve_nuscenes_version(cfg, split)

        self.sample_gap_us = int(
            _cfg_get(cfg, "sample_gap_us", DEFAULT_SAMPLE_GAP_US)
        )
        if self.sample_gap_us <= 0:
            raise ValueError("data.sample_gap_us must be positive")
        if self.sample_gap_us % NUSCENES_SWEEP_US != 0:
            raise ValueError(
                "NuScenesDynamicLiDAR4DDataset requires sample_gap_us to be an exact "
                f"multiple of the {NUSCENES_SWEEP_US}-us LiDAR sweep cadence"
            )
        self.sample_hop = self.sample_gap_us // NUSCENES_SWEEP_US
        self.novel_view_count = int(
            _cfg_get(cfg, "gt_middle_count", DEFAULT_NOVEL_VIEW_COUNT)
        )
        if self.novel_view_count < 0:
            raise ValueError("data.gt_middle_count must be non-negative")

        dynamic_window = _cfg_get(cfg, "dynamic_window", None)
        if dynamic_window is None:
            raise ValueError(
                "NuScenesDynamicLiDAR4DDataset requires data.dynamic_window config"
            )
        self.endpoint_min_us = _seconds_to_us(
            _cfg_get(dynamic_window, "endpoint_min_sec", 0.2),
            "data.dynamic_window.endpoint_min_sec",
        )
        self.endpoint_max_us = _seconds_to_us(
            _cfg_get(dynamic_window, "endpoint_max_sec", 1.2),
            "data.dynamic_window.endpoint_max_sec",
        )
        if self.endpoint_min_us > self.endpoint_max_us:
            raise ValueError(
                "data.dynamic_window endpoint_min_sec must be <= endpoint_max_sec"
            )
        self.eval_seed = int(_cfg_get(dynamic_window, "eval_seed", 42))
        self.train_excluded_scenes = tuple(
            str(scene_name) for scene_name in _cfg_get(
                dynamic_window,
                "train_excluded_scenes",
                DEFAULT_TRAIN_EXCLUDED_SCENES,
            )
        )
        if len(self.train_excluded_scenes) != len(set(self.train_excluded_scenes)):
            raise ValueError("data.dynamic_window.train_excluded_scenes has duplicates")

        # Retained only for stable start-anchor selection and compact-val keys.
        self.window_us = int(_cfg_get(cfg, "window_us", DEFAULT_WINDOW_US))
        self.pair_mode = str(_cfg_get(cfg, "pair_mode", "keyframe"))
        self.pair_kf_stride = int(_cfg_get(cfg, "pair_kf_stride", 2))
        if self.pair_mode != "keyframe":
            raise ValueError(
                "NuScenesDynamicLiDAR4DDataset requires data.pair_mode=keyframe"
            )
        if self.pair_kf_stride <= 0:
            raise ValueError("data.pair_kf_stride must be positive")

        self.nusc = NuScenes(
            version=self.version, dataroot=self.dataroot, verbose=False
        )
        split_scenes = set(create_splits_scenes()[split])
        train_split = str(_cfg_get(cfg, "train_split", "train"))
        if split == train_split:
            unknown = set(self.train_excluded_scenes) - split_scenes
            if unknown:
                raise ValueError(
                    "Configured LiDAR4D train exclusions are not in the train split: "
                    + ", ".join(sorted(unknown))
                )
            split_scenes.difference_update(self.train_excluded_scenes)
        scenes = [scene for scene in self.nusc.scene if scene["name"] in split_scenes]
        self.active_scene_names = tuple(scene["name"] for scene in scenes)

        self.scene_frames = []
        self.index = []
        skipped_without_endpoint = 0
        for scene in scenes:
            frames = []
            first_sample = self.nusc.get("sample", scene["first_sample_token"])
            current = first_sample["data"]["LIDAR_TOP"]
            while current:
                sd = self.nusc.get("sample_data", current)
                frames.append(
                    (
                        current,
                        int(sd["timestamp"]),
                        bool(sd["is_key_frame"]),
                    )
                )
                current = sd["next"]
            scene_idx = len(self.scene_frames)
            self.scene_frames.append(frames)

            keyframes = [i for i, frame in enumerate(frames) if frame[2]]
            for anchor in range(len(keyframes) - self.pair_kf_stride):
                start_idx = keyframes[anchor]
                nominal_end_idx = keyframes[anchor + self.pair_kf_stride]
                candidates = self._eligible_endpoints(frames, start_idx)
                if not candidates:
                    skipped_without_endpoint += 1
                    continue
                self.index.append((scene_idx, start_idx, nominal_end_idx))

        if self.verbose:
            print(
                f"[{split}] dynamic windows: {len(self.index)} keyframe starts; "
                f"skipped_without_endpoint={skipped_without_endpoint}"
            )
            if split == train_split:
                print(
                    f"[{split}] held out LiDAR4D scenes: "
                    + ", ".join(self.train_excluded_scenes)
                )

        self.window_labels = None
        self.window_subset = window_subset
        if window_subset:
            # The existing manifest selects exactly the legacy keyframe starts.
            # ``window_key`` intentionally exposes their nominal fixed endpoint;
            # the actual endpoint remains the deterministic dynamic draw below.
            compact_val.apply_manifest(self, window_subset, verbose=True)

    def __len__(self):
        return len(self.index)

    def window_span(self, idx: int):
        return self.index[idx]

    def window_key(self, idx: int) -> str:
        scene_idx, start_idx, nominal_end_idx = self.index[idx]
        frames = self.scene_frames[scene_idx]
        return compact_val.window_key(
            frames[start_idx][0], frames[nominal_end_idx][0]
        )

    def _eligible_endpoints(self, frames, start_idx: int) -> list[int]:
        return eligible_endpoint_indices(
            frames,
            start_idx,
            sample_hop=self.sample_hop,
            min_duration_us=self.endpoint_min_us,
            max_duration_us=self.endpoint_max_us,
        )

    def _sampling_rng(self, idx: int):
        train_split = str(_cfg_get(self.cfg, "train_split", "train"))
        if self.split == train_split:
            # Lightning's workers=True seeding initializes Python's RNG once per
            # worker/rank, so repeated accesses vary but complete runs replay.
            return random

        scene_idx, start_idx, nominal_end_idx = self.index[idx]
        frames = self.scene_frames[scene_idx]
        material = (
            f"{self.eval_seed}|{self.split}|{frames[start_idx][0]}|"
            f"{frames[nominal_end_idx][0]}"
        ).encode("utf-8")
        seed = int.from_bytes(hashlib.blake2b(material, digest_size=8).digest(), "big")
        return random.Random(seed)

    def sample_window_indices(self, idx: int) -> tuple[int, int, list[int], list[int]]:
        """Expose the sampled temporal layout for tests and diagnostics."""
        scene_idx, start_idx, _nominal_end_idx = self.index[idx]
        frames = self.scene_frames[scene_idx]
        end_idx, novel, used = sample_target_indices(
            start_idx,
            self._eligible_endpoints(frames, start_idx),
            sample_hop=self.sample_hop,
            novel_view_count=self.novel_view_count,
            rng=self._sampling_rng(idx),
        )
        return scene_idx, end_idx, novel, used

    def __getitem__(self, idx):
        scene_idx, end_idx, _novel_indices, used_indices = (
            self.sample_window_indices(idx)
        )
        frames = self.scene_frames[scene_idx]
        used = [frames[i] for i in used_indices]
        lidar_tokens = [frame[0] for frame in used]
        timestamps_us = [frame[1] for frame in used]

        coordinate_frame = resolve_coordinate_frame(self.cfg)
        sensor_to_ego_matrices = [
            sensor_to_ego(self.nusc, token) for token in lidar_tokens
        ]
        ego_to_world_matrices = [
            ego_to_world(self.nusc, token) for token in lidar_tokens
        ]
        sensor_to_world_matrices = [
            (ego_pose @ sensor_pose).astype(np.float32)
            for ego_pose, sensor_pose in zip(
                ego_to_world_matrices, sensor_to_ego_matrices
            )
        ]
        local_to_world_matrices = (
            ego_to_world_matrices
            if coordinate_frame == COORDINATE_FRAME_EGO
            else sensor_to_world_matrices
        )
        reference_to_world = local_to_world_matrices[0]
        world_to_reference = np.linalg.inv(reference_to_world)
        pose = torch.stack(
            [
                torch.from_numpy(
                    (world_to_reference @ local_to_world).astype(np.float32)
                )
                for local_to_world in local_to_world_matrices
            ]
        )
        pose_lidar_to_ref = torch.stack([
            torch.from_numpy(
                (world_to_reference @ transform).astype(np.float32)
            )
            for transform in sensor_to_world_matrices
        ])

        timestamps, timestamps_sec, window_duration_sec = relative_time_coordinates(
            timestamps_us,
            timestamps_us[0],
            frames[end_idx][1],
        )
        ego_radius = float(_cfg_get(self.cfg, "ego_radius", 0.0) or 0.0)
        loaded = [
            _load_points(self.nusc, self.dataroot, token, ego_radius)
            for token in lidar_tokens
        ]
        all_points_sensor = [item[0] for item in loaded]
        all_lidar_ring = [item[1] for item in loaded]
        all_points_ego = [
            transform_points(points, sensor_pose)
            for points, sensor_pose in zip(
                all_points_sensor, sensor_to_ego_matrices
            )
        ]
        all_points_local = (
            all_points_ego
            if coordinate_frame == COORDINATE_FRAME_EGO
            else all_points_sensor
        )

        # Chronological ``used`` always starts/ends at the model endpoints.
        input_indices = (0, len(used) - 1)
        input_points_local = [all_points_local[i] for i in input_indices]
        input_points_ego = [all_points_ego[i] for i in input_indices]
        input_points_sensor = [all_points_sensor[i] for i in input_indices]
        input_pose = pose[list(input_indices)]
        input_pose_lidar_to_ref = pose_lidar_to_ref[list(input_indices)]
        input_timestamps = timestamps[list(input_indices)]
        input_timestamps_sec = timestamps_sec[list(input_indices)]
        input_frame_counts = torch.tensor(
            [points.shape[0] for points in input_points_local], dtype=torch.long
        )

        reference_to_sensor = [
            make_reference_to_sensor(self.nusc, token, reference_to_world)
            for token in lidar_tokens
        ]
        gt_cameras = [
            Camera.from_nuscenes(
                nusc=self.nusc,
                lidar_token=lidar_token,
                timestamp_normalized=float(timestamp_normalized),
                timestamp_seconds=float(timestamp_seconds),
                pts_sensor=points_sensor,
                lidar_ring=lidar_ring,
                cfg=self.cfg,
                uid=uid,
                ref_to_sensor=ref_pose,
            )
            for uid, (
                lidar_token,
                points_sensor,
                lidar_ring,
                timestamp_normalized,
                timestamp_seconds,
                ref_pose,
            ) in enumerate(
                zip(
                    lidar_tokens,
                    all_points_sensor,
                    all_lidar_ring,
                    timestamps.tolist(),
                    timestamps_sec.tolist(),
                    reference_to_sensor,
                )
            )
        ]

        return {
            "input": {
                "lidar_points_local": torch.cat(input_points_local, dim=0),
                "lidar_points_ego": torch.cat(input_points_ego, dim=0),
                "lidar_points_sensor": torch.cat(input_points_sensor, dim=0),
                "frame_counts": input_frame_counts,
                "pose": input_pose,
                "pose_lidar_to_ref": input_pose_lidar_to_ref,
                # Normalized time remains a tiny compatibility field in the
                # shared Dynamic backend payload; V11 conditions on seconds.
                "timestamps": input_timestamps,
                "timestamps_sec": input_timestamps_sec,
                "window_duration_sec": window_duration_sec,
            },
            "gt": {"cameras": gt_cameras},
        }


# V11's Utonia stem consumes exact [scaled xyz, normalized intensity].  Unlike
# the legacy transform, this does not construct zero RGB/normal, an inverse map,
# or the checkpoint-era 9D feature that Point2Gaus would immediately overwrite.
dynamic_transform = utonia.transform.Compose(
    [
        dict(type="RandomScale", scale=[0.2, 0.2]),
        dict(
            type="GridSample",
            grid_size=0.01,
            hash_type="fnv",
            mode="train",
            return_grid_coord=True,
            return_inverse=False,
        ),
        dict(type="ToTensor"),
        dict(
            type="Collect",
            keys=("coord", "grid_coord", "strength"),
            feat_keys=("coord", "strength"),
        ),
    ]
)


def dynamic_lidar4d_multiframe_collate_fn(batch):
    """Pack the minimal variable-point V11 batch consumed by ModelWrapper."""
    input_points_local = []
    input_points_ego = []
    input_points_sensor = []
    input_frame_counts = []
    input_batch_index = []
    input_pose = []
    input_pose_lidar_to_ref = []
    input_timestamps = []
    input_timestamps_sec = []
    input_duration_sec = []
    gt_cameras = []

    for batch_idx, item in enumerate(batch):
        source = item["input"]
        points_local = source["lidar_points_local"]
        points_sensor = source["lidar_points_sensor"]
        input_points_local.append(points_local)
        input_points_ego.append(source["lidar_points_ego"])
        input_points_sensor.append(points_sensor)
        input_frame_counts.append(source["frame_counts"])
        input_batch_index.append(
            torch.full((points_local.shape[0],), batch_idx, dtype=torch.long)
        )
        input_pose.append(source["pose"])
        input_pose_lidar_to_ref.append(source["pose_lidar_to_ref"])
        input_timestamps.append(source["timestamps"])
        input_timestamps_sec.append(source["timestamps_sec"])
        input_duration_sec.append(source["window_duration_sec"])
        gt_cameras.append(item["gt"]["cameras"])

    lidar_points_local = torch.cat(input_points_local, dim=0)
    lidar_points_ego = torch.cat(input_points_ego, dim=0)
    lidar_points_sensor = torch.cat(input_points_sensor, dim=0)
    flat_counts = torch.cat(input_frame_counts, dim=0)
    offset = torch.cumsum(flat_counts, dim=0)
    batch_idx = torch.cat(input_batch_index, dim=0)

    ptv3_coord = []
    ptv3_grid_coord = []
    ptv3_feat = []
    ptv3_strength = []
    starts = torch.cat([torch.zeros(1, dtype=offset.dtype), offset[:-1]])
    for start, end in zip(starts.tolist(), offset.tolist()):
        frame_points = lidar_points_local[start:end]
        transformed = dynamic_transform(
            {
                "coord": frame_points[:, :3].detach().cpu().numpy().copy(),
                "strength": frame_points[:, 3:4].detach().cpu().numpy().copy(),
            }
        )
        ptv3_coord.append(transformed["coord"])
        ptv3_grid_coord.append(transformed["grid_coord"])
        ptv3_feat.append(transformed["feat"])
        ptv3_strength.append(transformed["strength"])

    sampled_counts = torch.tensor(
        [coord.shape[0] for coord in ptv3_coord], dtype=torch.long
    )
    ptv3_input = {
        "coord": torch.cat(ptv3_coord, dim=0),
        "grid_coord": torch.cat(ptv3_grid_coord, dim=0),
        "feat": torch.cat(ptv3_feat, dim=0),
        "strength": torch.cat(ptv3_strength, dim=0),
        "offset": torch.cumsum(sampled_counts, dim=0),
    }

    packed = {
        "input": {
            "lidar_points_local": lidar_points_local,
            "lidar_points_ego": lidar_points_ego,
            "lidar_points_sensor": lidar_points_sensor,
            "batch_idx": batch_idx,
            "offset": offset,
            "pose": input_pose,
            "pose_lidar_to_ref": input_pose_lidar_to_ref,
            "timestamps": input_timestamps,
            "timestamps_sec": input_timestamps_sec,
            "window_duration_sec": torch.stack(input_duration_sec),
            "ptv3_input": ptv3_input,
        },
        "gt": {"cameras": gt_cameras},
    }
    # Per-window metadata such as a compact-val group, one list entry per item.
    if any("meta" in item for item in batch):
        keys = sorted({key for item in batch for key in item.get("meta", {})})
        packed["meta"] = {key: [item.get("meta", {}).get(key) for item in batch] for key in keys}
    return packed


# DataModule resolves this attribute from the selected dataset class.
NuScenesDynamicLiDAR4DDataset.collate_fn = staticmethod(
    dynamic_lidar4d_multiframe_collate_fn
)


__all__ = [
    "NuScenesDynamicLiDAR4DDataset",
    "dynamic_lidar4d_multiframe_collate_fn",
    "dynamic_transform",
    "eligible_endpoint_indices",
    "LIDAR4D_TRAIN_SEQUENCE_SCENES",
    "sample_target_indices",
]
