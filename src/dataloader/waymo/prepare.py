"""Repack Waymo v2 TOP first returns into lossless, per-frame row groups.

The original component files remain untouched. Each cache row holds range,
intensity, and the six per-pixel pose channels as float32 arrays, allowing a
random training window to read individual sweeps without decompressing an
entire 20-second segment. Geometry is computed by ``geometry.py`` at load time.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


RANGE_PREFIX = "[LiDARComponent].range_image_return1"
POSE_PREFIX = "[LiDARPoseComponent].range_image_return1"
STAMP = "key.frame_timestamp_micros"


def _list_column(value):
    value = np.ascontiguousarray(np.asarray(value, dtype=np.float32).ravel())
    offsets = pa.array([0, len(value)], type=pa.int64())
    return pa.LargeListArray.from_arrays(offsets, pa.array(value))


def _value_array(table, name, row, shape):
    value = table[name][row].values.to_numpy(zero_copy_only=False)
    return np.asarray(value, dtype=np.float32).reshape(shape)


def _timestamp_index(table):
    stamps = np.asarray(table[STAMP].to_numpy(), dtype=np.int64)
    if len(set(stamps.tolist())) != len(stamps):
        raise ValueError("Duplicate frame timestamps in a Waymo component")
    return {int(stamp): index for index, stamp in enumerate(stamps)}


def prepare_sequence(raw_root, cache_root, split, sequence, min_free_gib=50.0):
    """Write one sequence atomically; refuse to overwrite incomplete outputs."""
    raw_root, cache_root = Path(raw_root), Path(cache_root)
    destination = cache_root / split / sequence
    data_path, meta_path, ready_path = (destination / x for x in ("frames.parquet", "metadata.npz", "READY.json"))
    if ready_path.exists():
        info = json.loads(ready_path.read_text())
        if not data_path.is_file() or not meta_path.is_file() or pq.ParquetFile(data_path).num_row_groups != info["frames"]:
            raise ValueError(f"Incomplete existing Waymo cache: {destination}")
        return sequence, info["frames"], "existing"
    if data_path.exists() or meta_path.exists():
        raise FileExistsError(f"Incomplete cache exists; inspect before retrying: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(destination).free < float(min_free_gib) * 2**30:
        raise OSError(f"Less than {min_free_gib:g} GiB remains on the cache filesystem")

    def path(component):
        result = raw_root / split / component / f"{sequence}.parquet"
        if not result.is_file():
            raise FileNotFoundError(result)
        return result

    lidar = pq.read_table(path("lidar"), columns=[STAMP, "key.laser_name",
        f"{RANGE_PREFIX}.values", f"{RANGE_PREFIX}.shape"], filters=[("key.laser_name", "=", 1)])
    poses = pq.read_table(path("lidar_pose"), columns=[STAMP,
        f"{POSE_PREFIX}.values", f"{POSE_PREFIX}.shape"])
    vehicles = pq.read_table(path("vehicle_pose"), columns=[STAMP,
        "[VehiclePoseComponent].world_from_vehicle.transform"])
    calibration = pq.read_table(path("lidar_calibration"), filters=[("key.laser_name", "=", 1)])
    if len(calibration) != 1:
        raise ValueError(f"Expected one TOP LiDAR calibration: {sequence}")
    lidar_idx, pose_idx, vehicle_idx = map(_timestamp_index, (lidar, poses, vehicles))
    if set(lidar_idx) != set(pose_idx) or set(lidar_idx) != set(vehicle_idx) or not lidar_idx:
        raise ValueError(f"LiDAR and pose timestamps do not match: {sequence}")
    timestamps = np.array(sorted(lidar_idx), dtype=np.int64)
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError(f"Non-increasing timestamps: {sequence}")
    shape = lidar[f"{RANGE_PREFIX}.shape"][lidar_idx[int(timestamps[0])]].as_py()
    if len(shape) != 3 or shape[0] != 64 or shape[2] != 4:
        raise ValueError(f"Unexpected TOP range image shape {shape}: {sequence}")
    height, width = int(shape[0]), int(shape[1])
    extrinsic = np.asarray(calibration["[LiDARCalibrationComponent].extrinsic.transform"][0].as_py(), dtype=np.float64).reshape(4, 4)
    beams = calibration["[LiDARCalibrationComponent].beam_inclination.values"][0].as_py()
    if not beams:
        lo = float(calibration["[LiDARCalibrationComponent].beam_inclination.min"][0].as_py())
        hi = float(calibration["[LiDARCalibrationComponent].beam_inclination.max"][0].as_py())
        beams = lo + (np.arange(height) + 0.5) / height * (hi - lo)
    beams = np.asarray(beams, dtype=np.float64)[::-1].copy()
    if beams.shape != (height,):
        raise ValueError(f"Unexpected TOP beam calibration: {sequence}")
    vehicle_matrices = np.stack([
        np.asarray(vehicles["[VehiclePoseComponent].world_from_vehicle.transform"][vehicle_idx[int(stamp)]].as_py(), dtype=np.float64).reshape(4, 4)
        for stamp in timestamps
    ])
    fd, tmp_data = tempfile.mkstemp(prefix=".frames-", suffix=".parquet", dir=destination)
    os.close(fd)
    fd, tmp_meta = tempfile.mkstemp(prefix=".metadata-", suffix=".npz", dir=destination)
    os.close(fd)
    try:
        writer = None
        try:
            for stamp in timestamps:
                i, j = lidar_idx[int(stamp)], pose_idx[int(stamp)]
                rshape = lidar[f"{RANGE_PREFIX}.shape"][i].as_py()
                pshape = poses[f"{POSE_PREFIX}.shape"][j].as_py()
                if rshape != [height, width, 4] or pshape != [height, width, 6]:
                    raise ValueError(f"Inconsistent range/pose image size: {sequence} {stamp}")
                scan = _value_array(lidar, f"{RANGE_PREFIX}.values", i, rshape)
                pixel = _value_array(poses, f"{POSE_PREFIX}.values", j, pshape)
                columns = {"timestamp_us": pa.array([int(stamp)], type=pa.int64()),
                           "range": _list_column(scan[..., 0]), "intensity": _list_column(scan[..., 1])}
                columns.update({f"pose_{k}": _list_column(pixel[..., k]) for k in range(6)})
                frame = pa.table(columns)
                if writer is None:
                    writer = pq.ParquetWriter(tmp_data, frame.schema, compression="zstd", use_dictionary=False)
                writer.write_table(frame, row_group_size=1)
        finally:
            if writer is not None:
                writer.close()
        with open(tmp_meta, "wb") as stream:
            np.savez(stream, timestamps_us=timestamps, world_from_vehicle=vehicle_matrices,
                     vehicle_from_lidar=extrinsic, inclination_rad=beams,
                     height=np.int64(height), width=np.int64(width))
        if pq.ParquetFile(tmp_data).num_row_groups != len(timestamps):
            raise ValueError(f"Waymo cache row-group count mismatch: {sequence}")
        os.replace(tmp_data, data_path)
        os.replace(tmp_meta, meta_path)
        ready_path.write_text(json.dumps({"version": 1, "split": split, "sequence": sequence,
            "frames": len(timestamps), "image_shape": [height, width],
            "format": "float32 TOP first return, pixel poses, zstd row groups"}, indent=2) + "\n")
    finally:
        for name in (tmp_data, tmp_meta):
            if os.path.exists(name):
                os.unlink(name)
    return sequence, len(timestamps), "created"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--splits", nargs="+", default=["training", "validation"])
    parser.add_argument("--sequence", action="append", help="Limit to these segment IDs")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--min-free-gib", type=float, default=50.0)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    tasks = []
    for split in args.splits:
        sequences = args.sequence or sorted(p.stem for p in (Path(args.raw_root) / split / "lidar").glob("*.parquet"))
        if not sequences:
            raise ValueError(f"No raw LiDAR files for split {split}")
        tasks.extend((args.raw_root, args.cache_root, split, sequence, args.min_free_gib) for sequence in sequences)
    if args.workers == 1:
        for task in tasks:
            print(*prepare_sequence(*task), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(prepare_sequence, *task) for task in tasks]
            for future in as_completed(futures):
                print(*future.result(), flush=True)


if __name__ == "__main__":
    main()
