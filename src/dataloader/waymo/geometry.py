"""Waymo TOP first-return geometry and calibrated panorama projection.

Waymo's pixel poses map the vehicle frame at each laser measurement into the
world frame; the frame pose specifies the vehicle frame used for the sweep.
The equations mirror the public Waymo v2 range-image utilities, while keeping
this training-side module independent of TensorFlow.
"""
from __future__ import annotations

import numpy as np


def panorama_from_lidar_yaw(vehicle_from_lidar):
    """Rotate LiDAR axes so the panorama's azimuth follows Waymo's native grid.

    Waymo subtracts the sensor extrinsic yaw when decoding each raw image
    column. Undo that yaw for the panorama while keeping the LiDAR origin and
    its calibrated vertical beam angles unchanged.
    """
    extrinsic = np.asarray(vehicle_from_lidar, dtype=np.float64)
    if extrinsic.shape != (4, 4):
        raise ValueError("vehicle_from_lidar must be 4x4")
    yaw = np.arctan2(extrinsic[1, 0], extrinsic[0, 0])
    cosine, sine = np.cos(yaw), np.sin(yaw)
    result = np.eye(4, dtype=np.float64)
    result[:2, :2] = [[cosine, -sine], [sine, cosine]]
    return result


def top_range_to_sensor_points(ranges, intensity, pixel_pose, inclination,
                               vehicle_from_lidar, world_from_vehicle):
    """Return first-return XYZI deskewed into the frame-time LiDAR frame.

    ``inclination`` is the top-to-bottom beam table in radians; pixel pose is
    ``[roll, pitch, yaw, x, y, z]`` in world coordinates. The output preserves
    the native image's row-major order and all positive, finite returns.
    """
    ranges = np.asarray(ranges, dtype=np.float32)
    intensity = np.asarray(intensity, dtype=np.float32)
    pixel_pose = np.asarray(pixel_pose, dtype=np.float32)
    beams = np.asarray(inclination, dtype=np.float32)
    extrinsic = np.asarray(vehicle_from_lidar, dtype=np.float32)
    frame = np.asarray(world_from_vehicle, dtype=np.float32)
    if ranges.ndim != 2 or intensity.shape != ranges.shape:
        raise ValueError("range and intensity must be matching 2D arrays")
    height, width = ranges.shape
    if pixel_pose.shape != (height, width, 6) or beams.shape != (height,):
        raise ValueError("pixel pose and beam calibration do not match the image")
    if extrinsic.shape != (4, 4) or frame.shape != (4, 4):
        raise ValueError("extrinsic and frame poses must be 4x4 matrices")
    valid = (ranges > 0) & np.isfinite(ranges) & np.isfinite(intensity)
    if not np.isfinite(pixel_pose[valid]).all():
        raise ValueError("a valid return has a non-finite pixel pose")
    rows, cols = np.nonzero(valid)
    r = ranges[valid]
    correction = np.arctan2(extrinsic[1, 0], extrinsic[0, 0])
    azimuth = ((width - cols.astype(np.float32) - 0.5) / width * 2 - 1) * np.pi - correction
    elevation = beams[rows]
    cos_el = np.cos(elevation)
    lidar = np.stack((r * cos_el * np.cos(azimuth),
                      r * cos_el * np.sin(azimuth), r * np.sin(elevation)), axis=1)
    vehicle = lidar @ extrinsic[:3, :3].T + extrinsic[:3, 3]

    roll, pitch, yaw = pixel_pose[valid, :3].T
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    x, y, z = vehicle.T
    yr, zr = cr * y - sr * z, sr * y + cr * z
    xp, zp = cp * x + sp * zr, -sp * x + cp * zr
    world = np.stack((cy * xp - sy * yr, sy * xp + cy * yr, zp), axis=1)
    world += pixel_pose[valid, 3:6]
    vehicle_from_world = np.linalg.inv(frame)
    frame_vehicle = world @ vehicle_from_world[:3, :3].T + vehicle_from_world[:3, 3]
    lidar_from_vehicle = np.linalg.inv(extrinsic)
    frame_lidar = frame_vehicle @ lidar_from_vehicle[:3, :3].T + lidar_from_vehicle[:3, 3]
    return np.column_stack((frame_lidar, intensity[valid])).astype(np.float32, copy=False)


def project_calibrated_panorama(points, beam_elevation_rad, width, max_depth):
    """Project sensor-frame XYZI to nearest calibrated row/column and range.

    The nearest point wins for collisions. Columns agree with the repository's
    LiDAR-to-camera convention: the middle column faces sensor +x.
    """
    points = np.asarray(points, dtype=np.float32)
    beams = np.sort(np.asarray(beam_elevation_rad, dtype=np.float64))[::-1]
    width = int(width)
    if points.ndim != 2 or points.shape[1] < 4 or not len(beams) or width < 1:
        raise ValueError("expected XYZI points and a nonempty panorama grid")
    distance = np.linalg.norm(points[:, :3], axis=1)
    valid = np.isfinite(points[:, :4]).all(axis=1) & (distance > 0) & (distance < float(max_depth))
    xyz, strength, distance = points[valid, :3], points[valid, 3], distance[valid]
    elevation = np.arctan2(xyz[:, 2], np.hypot(xyz[:, 0], xyz[:, 1]))
    position = np.searchsorted(-beams, -elevation)
    lo = np.clip(position - 1, 0, len(beams) - 1)
    hi = np.clip(position, 0, len(beams) - 1)
    row = np.where(np.abs(elevation - beams[lo]) <= np.abs(elevation - beams[hi]), lo, hi)
    azimuth = np.arctan2(xyz[:, 1], xyz[:, 0])
    column = np.rint((np.pi - azimuth) * (width / (2 * np.pi))).astype(np.int64) % width
    pixels = row * width + column
    order = np.argsort(distance, kind="stable")
    _, first = np.unique(pixels[order], return_index=True)
    chosen = order[first]
    depth = np.zeros(len(beams) * width, dtype=np.float32)
    image_intensity = np.zeros_like(depth)
    depth[pixels[chosen]] = distance[chosen]
    image_intensity[pixels[chosen]] = strength[chosen]
    return depth.reshape(1, len(beams), width), image_intensity.reshape(1, len(beams), width)
