"""Honda H3D raw sweeps, odometry, frame times and splits, without model imports.

Release layout (``icra_benchmark_20200103_with_odom``): one flat directory per
scenario with per-frame files ``pointcloud1_NNN.ply``, ``odom_NNN.txt``,
``gps_NNN.csv``, ``CAN_vel/CAN_yaw_NNN.csv`` and ``labels_3d1_NNN.txt``; the
official splits are ``train_test_split/detection_{train,val,test}.txt``.

PLY vertices reuse PCL surfel fields. The official ``convert_to_bin.py`` reads
XYZ plus ``radius`` / 255 as intensity. ``confidence`` holds the HDL-64E S2
laser ID (0-63) and ``curvature`` the encoder azimuth in degrees. Each file is
an organized 64 x C scan (C = 2082 +- 6k columns): rays without a return are
stored, with raw distance 0 pushed to 1.0-1.5 m by the per-laser distance
correction, so a return is any point beyond ``NO_RETURN_MAX_RANGE_M``.

``odom_NNN.txt`` is ``x y z roll pitch yaw`` (metres, radians) of the LiDAR in
a per-scenario world frame; the official AB3DMOT code builds
``Rz(yaw) @ Ry(pitch) @ Rx(roll)`` and maps ``p_world = T @ p_lidar``.

There is no LiDAR capture timestamp. Each frame's ``gps_NNN.csv`` bundles the
~10 ADMA samples (100 Hz) recorded with that sweep; the midpoint of the first
and last sample is the frame-time proxy (median spacing 0.100 s).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

H3D_DEFAULT_ROOT = "/data1/LiDAR/H3D/data/icra_benchmark_20200103_with_odom"
H3D_SPLITS = ("train", "val", "test")
# Placeholders for rays without a return sit at 1.0-1.5 m (raw distance 0 plus
# the distance correction); 0.03 % of the real returns lie within 1.5-2.0 m.
NO_RETURN_MAX_RANGE_M = 1.5

_PLY_TYPES = {
    "char": "i1", "uchar": "u1", "short": "<i2", "ushort": "<u2",
    "int": "<i4", "uint": "<u4", "float": "<f4", "double": "<f8",
}


def read_h3d_ply(path):
    """Return the raw structured vertex array of one binary little-endian PLY."""
    data = Path(path).read_bytes()
    end = data.find(b"end_header\n")
    if end < 0:
        raise ValueError(f"Not a PLY file: {path}")
    body = end + len(b"end_header\n")
    fields, count = [], None
    for line in data[:end].decode("latin1").splitlines():
        parts = line.split()
        if parts[:1] == ["format"] and parts[1] != "binary_little_endian":
            raise ValueError(f"Unsupported PLY format {parts[1]}: {path}")
        if parts[:2] == ["element", "vertex"]:
            count = int(parts[2])
        elif parts[:1] == ["property"]:
            fields.append((parts[2], _PLY_TYPES[parts[1]]))
    vertices = np.frombuffer(data, dtype=np.dtype(fields), count=count, offset=body)
    if body + vertices.nbytes != len(data):
        raise ValueError(f"PLY payload size mismatch: {path}")
    return vertices


def h3d_scan(path, min_range=NO_RETURN_MAX_RANGE_M, intensity_divisor=255.0):
    """Return float32 XYZI returns (intensity = radius / divisor) and their laser IDs.

    The official converter divides by 255; KITTI-360 scans store the same
    8-bit value divided by 256.
    """
    vertices = read_h3d_ply(path)
    points = np.stack([
        vertices["x"], vertices["y"], vertices["z"],
        vertices["radius"].astype(np.float32) / np.float32(intensity_divisor),
    ], axis=1).astype(np.float32)
    lasers = vertices["confidence"].astype(np.int64)
    keep = np.isfinite(points).all(axis=1) & (np.linalg.norm(points[:, :3], axis=1) > min_range)
    return np.ascontiguousarray(points[keep]), lasers[keep]


def pose_from_xyzrpy(values):
    """4x4 world-from-LiDAR pose from ``x y z roll pitch yaw`` (radians)."""
    x, y, z, roll, pitch, yaw = (float(value) for value in values)
    cr, sr, cp, sp, cy, sy = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    pose = np.eye(4)
    pose[:3, :3] = rz @ ry @ rx
    pose[:3, 3] = (x, y, z)
    return pose


def read_odom(path):
    values = np.loadtxt(path, dtype=np.float64).reshape(-1)
    if values.size != 6:
        raise ValueError(f"odom needs x y z roll pitch yaw: {path}")
    return pose_from_xyzrpy(values)


def gps_frame_time_us(path):
    """Midpoint of the first and last ADMA sample bundled with one sweep (µs)."""
    stamps = []
    with open(path) as handle:
        for line in handle:
            if line.startswith("#") or not line.strip():
                continue
            stamps.append(int(round(float(line.split(",", 1)[0]) * 1e6)))
    if not stamps:
        raise ValueError(f"No GPS samples: {path}")
    return (min(stamps) + max(stamps)) // 2


def read_split(root, split):
    if split not in H3D_SPLITS:
        raise ValueError(f"H3D split must be one of {H3D_SPLITS}")
    path = Path(root) / "train_test_split" / f"detection_{split}.txt"
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def read_labels(path):
    """Rows of (class, track_id, state, x, y, z, d0, d1, d2, yaw) in the LiDAR frame."""
    rows = []
    path = Path(path)
    if not path.is_file():
        return rows
    for line in path.read_text().splitlines():
        parts = line.strip().split(",")
        if len(parts) < 10:
            continue
        rows.append((parts[0], parts[1], parts[2], *map(float, parts[3:10])))
    return rows


class H3DScenario:
    """Frame IDs, frame-time proxy (µs) and LiDAR poses of one scenario."""

    def __init__(self, root, name):
        self.root = Path(root)
        self.name = str(name)
        self.directory = self.root / self.name
        self.frames = np.asarray(sorted(
            int(path.stem.rsplit("_", 1)[1]) for path in self.directory.glob("pointcloud1_*.ply")
        ), dtype=np.int64)
        if not len(self.frames):
            raise FileNotFoundError(f"No H3D sweeps in {self.directory}")
        self.time_us = np.asarray([gps_frame_time_us(self.path("gps", frame)) for frame in self.frames],
                                  dtype=np.int64)
        if np.any(np.diff(self.time_us) <= 0):
            raise ValueError(f"{self.name}: frame times must increase")

    def path(self, kind, frame):
        name = {"ply": "pointcloud1_{:03d}.ply", "odom": "odom_{:03d}.txt", "gps": "gps_{:03d}.csv",
                "labels": "labels_3d1_{:03d}.txt"}[kind]
        return self.directory / name.format(int(frame))

    @property
    def duration_sec(self):
        return float(self.time_us[-1] - self.time_us[0]) / 1e6

    def seconds(self, frame):
        """Seconds since the scenario's first frame."""
        return float(self.time_us[self.index(frame)] - self.time_us[0]) / 1e6

    def index(self, frame):
        position = int(np.searchsorted(self.frames, int(frame)))
        if position >= len(self.frames) or self.frames[position] != int(frame):
            raise KeyError(f"{self.name}: no frame {frame}")
        return position

    def nearest_frame(self, seconds, exclude=()):
        exclude = {int(frame) for frame in exclude}
        candidates = [i for i, frame in enumerate(self.frames) if int(frame) not in exclude]
        offsets = (self.time_us[candidates] - self.time_us[0]) / 1e6
        return int(self.frames[candidates[int(np.argmin(np.abs(offsets - float(seconds))))]])

    def pose(self, frame):
        return read_odom(self.path("odom", frame))
