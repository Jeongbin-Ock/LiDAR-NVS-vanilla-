from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from src.dataloader.waymo.dynamic import WaymoDynamicDataset
from src.dataloader.waymo.prepare import prepare_sequence


def _raw_sequence(root, split, name, gap_after=None):
    count, height, width = 6, 64, 4
    stamps = [1_000_000 + i * 100_000 + (300_000 if gap_after is not None and i > gap_after else 0)
              for i in range(count)]
    scan = np.zeros((height, width, 4), dtype=np.float32)
    scan[..., 0] = 10.0
    scan[..., 1] = 0.2
    scan[0, 0, 1] = 4.0
    pose = np.zeros((height, width, 6), dtype=np.float32)
    eye = np.eye(4).reshape(-1).tolist()
    tables = {
        "lidar": pa.table({"key.frame_timestamp_micros": stamps, "key.laser_name": [1] * count,
            "[LiDARComponent].range_image_return1.values": [scan.ravel().tolist()] * count,
            "[LiDARComponent].range_image_return1.shape": [[height, width, 4]] * count}),
        "lidar_pose": pa.table({"key.frame_timestamp_micros": stamps,
            "[LiDARPoseComponent].range_image_return1.values": [pose.ravel().tolist()] * count,
            "[LiDARPoseComponent].range_image_return1.shape": [[height, width, 6]] * count}),
        "vehicle_pose": pa.table({"key.frame_timestamp_micros": stamps,
            "[VehiclePoseComponent].world_from_vehicle.transform": [eye] * count}),
        "lidar_calibration": pa.table({"key.laser_name": [1],
            "[LiDARCalibrationComponent].extrinsic.transform": [eye],
            "[LiDARCalibrationComponent].beam_inclination.values": [np.linspace(-0.3, 0.04, height).tolist()],
            "[LiDARCalibrationComponent].beam_inclination.min": [-0.3],
            "[LiDARCalibrationComponent].beam_inclination.max": [0.04]}),
    }
    for component, table in tables.items():
        path = Path(root) / split / component / f"{name}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, path)
    return scan


def test_waymo_cache_and_training_sample_preserve_frames(tmp_path):
    raw, cache = tmp_path / "raw", tmp_path / "cache"
    scan = _raw_sequence(raw, "training", "training_sequence")
    assert prepare_sequence(raw, cache, "training", "training_sequence", min_free_gib=0)[1:] == (6, "created")
    assert prepare_sequence(raw, cache, "training", "training_sequence", min_free_gib=0)[1:] == (6, "existing")
    cached = pq.ParquetFile(cache / "training/training_sequence/frames.parquet")
    assert cached.num_row_groups == 6
    np.testing.assert_array_equal(cached.read_row_group(2)["range"][0].values.to_numpy().reshape(64, 4), scan[..., 0])
    cfg = OmegaConf.create({"dataroot": str(cache), "mode": "nvs", "coordinate_frame": "ego_vehicle",
        "image_height": 64, "image_width": 16, "vfov": [-17.6, 2.4], "hfov": [-180, 180],
        "expected_train_sequences": 1, "expected_val_sequences": 0, "intensity_max": 1.0,
        "window_stride_sec": 0.1, "gt_middle_count": 3, "ego_radius": 0.0, "gt_ego_radius": 0.0,
        "max_depth_m": 80.0, "dynamic_window": {"endpoint_min_sec": 0.2, "endpoint_max_sec": 0.4, "eval_seed": 42}})
    sample = WaymoDynamicDataset(cfg, "train")[0]
    assert sample["input"]["frame_counts"].tolist() == [256, 256]
    assert len(sample["gt"]["cameras"]) >= 2
    assert sample["gt"]["cameras"][0].pts_depth.shape == (1, 64, 16)
    assert sample["input"]["lidar_points_local"][:, 3].max().item() == 1.0


def test_waymo_cache_rejects_training_validation_leak(tmp_path):
    root = tmp_path / "cache"
    for split in ("training", "validation"):
        path = root / split / "same_sequence"
        path.mkdir(parents=True)
        (path / "READY.json").write_text("{}")
    cfg = OmegaConf.create({"dataroot": str(root), "mode": "nvs", "coordinate_frame": "ego_vehicle",
        "image_height": 64, "intensity_max": 1.0, "expected_train_sequences": 1,
        "expected_val_sequences": 1})
    with pytest.raises(ValueError, match="overlap"):
        WaymoDynamicDataset(cfg, "train")


def test_waymo_windows_never_cross_missing_sweeps(tmp_path):
    raw, cache = tmp_path / "raw", tmp_path / "cache"
    _raw_sequence(raw, "training", "gap_sequence", gap_after=2)
    prepare_sequence(raw, cache, "training", "gap_sequence", min_free_gib=0)
    cfg = OmegaConf.create({"dataroot": str(cache), "mode": "nvs", "coordinate_frame": "ego_vehicle",
        "image_height": 64, "image_width": 16, "vfov": [-17.6, 2.4], "hfov": [-180, 180],
        "expected_train_sequences": 1, "expected_val_sequences": 0, "intensity_max": 1.0,
        "window_stride_sec": 0.1, "max_sweep_gap_sec": 0.15, "gt_middle_count": 3,
        "ego_radius": 0.0, "gt_ego_radius": 0.0, "max_depth_m": 80.0,
        "dynamic_window": {"endpoint_min_sec": 0.2, "endpoint_max_sec": 0.4, "eval_seed": 42}})
    dataset = WaymoDynamicDataset(cfg, "train")
    assert [(first, last, start) for _, first, last, start in dataset.index] == [(0, 2, 0), (3, 5, 3)]
    for sample in dataset:
        assert sample["input"]["window_duration_sec"].item() == pytest.approx(0.2)
