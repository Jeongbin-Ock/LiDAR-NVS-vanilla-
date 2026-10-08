"""Batch packing shared by the random-duration loaders (nuScenes dynamic, KITTI-360).

Each item holds two endpoint sweeps in their own local frame plus target GT
cameras. The collate keeps the raw points for diagnostics and builds the
Utonia input: every sweep is scaled by 0.2 and grid-sampled at 0.01 (5 cm),
with features [scaled xyz, intensity]. ``meta`` entries, such as a
compact-val window group, pass through as one list per key.
"""
from __future__ import annotations

import torch

from ..models import utonia


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
