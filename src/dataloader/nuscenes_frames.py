"""Coordinate transforms for nuScenes LiDAR and ego frames.

nuScenes point files are expressed in the calibrated LiDAR sensor frame. The
calibration matrix maps those coordinates into that sweep's ego frame; ego pose
then maps them into the log's world frame. Keep these transform directions
explicit so the Utonia input frame, temporal reference frame, and render sensor
frame cannot be confused.
"""
from __future__ import annotations

import numpy as np
import torch
from pyquaternion import Quaternion


COORDINATE_FRAME_EGO = "ego_vehicle"
COORDINATE_FRAME_LIDAR = "lidar_sensor"
SUPPORTED_COORDINATE_FRAMES = (
    COORDINATE_FRAME_EGO,
    COORDINATE_FRAME_LIDAR,
)


def resolve_coordinate_frame(cfg) -> str:
    """Return the configured local/reference frame.

    Missing values select the historical LiDAR-sensor contract. This preserves
    checkpoints whose embedded config predates ``data.coordinate_frame``.
    Fresh configs set ``ego_vehicle`` explicitly.
    """
    value = str(getattr(cfg, "coordinate_frame", COORDINATE_FRAME_LIDAR)).lower()
    if value not in SUPPORTED_COORDINATE_FRAMES:
        raise ValueError(
            "data.coordinate_frame must be one of "
            f"{SUPPORTED_COORDINATE_FRAMES}, got {value!r}"
        )
    return value


def sensor_to_ego(nusc, sample_data_token: str) -> np.ndarray:
    """Return float32 ``T_ego<-sensor`` from nuScenes calibration."""
    sample_data = nusc.get("sample_data", sample_data_token)
    calibration = nusc.get(
        "calibrated_sensor", sample_data["calibrated_sensor_token"]
    )
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = Quaternion(calibration["rotation"]).rotation_matrix
    transform[:3, 3] = calibration["translation"]
    return transform


def ego_to_world(nusc, sample_data_token: str) -> np.ndarray:
    """Return float32 ``T_world<-ego`` for one sweep timestamp."""
    sample_data = nusc.get("sample_data", sample_data_token)
    ego_pose = nusc.get("ego_pose", sample_data["ego_pose_token"])
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = Quaternion(ego_pose["rotation"]).rotation_matrix
    transform[:3, 3] = ego_pose["translation"]
    return transform


def sensor_to_world(nusc, sample_data_token: str) -> np.ndarray:
    """Return float32 ``T_world<-sensor`` for one sweep timestamp."""
    return (
        ego_to_world(nusc, sample_data_token)
        @ sensor_to_ego(nusc, sample_data_token)
    ).astype(np.float32)


def reference_to_sensor(
    nusc, sample_data_token: str, reference_to_world: np.ndarray
) -> np.ndarray:
    """Return ``T_sensor<-reference`` for a target LiDAR sweep."""
    world_from_sensor = sensor_to_world(nusc, sample_data_token)
    return (np.linalg.inv(world_from_sensor) @ reference_to_world).astype(
        np.float32
    )


def transform_points(points: torch.Tensor, transform: np.ndarray) -> torch.Tensor:
    """Apply a homogeneous pose to point rows, preserving non-XYZ columns."""
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError("points must have shape (N, >=3)")
    matrix = torch.as_tensor(transform, device=points.device, dtype=points.dtype)
    result = points.clone()
    result[:, :3] = points[:, :3] @ matrix[:3, :3].T + matrix[:3, 3]
    return result


__all__ = [
    "COORDINATE_FRAME_EGO",
    "COORDINATE_FRAME_LIDAR",
    "SUPPORTED_COORDINATE_FRAMES",
    "ego_to_world",
    "reference_to_sensor",
    "resolve_coordinate_frame",
    "sensor_to_ego",
    "sensor_to_world",
    "transform_points",
]
