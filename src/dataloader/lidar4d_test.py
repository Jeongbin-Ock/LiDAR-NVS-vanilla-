"""Dataset dispatch for the shared inference evaluator (not training)."""
from __future__ import annotations


def canonical_lidar4d_dataset_name(cfg):
    name = str(getattr(cfg, "dataset_name", "nuscenes")).lower()
    if name in {"nuscenes", "nuscene", "nuscenes_dynamic_lidar4d"}:
        return "nuscenes"
    if name in {"kitti360", "kitti360_lidar4d", "kitti360_dynamic"}:
        return "kitti360"
    raise ValueError(f"Unsupported LiDAR4D inference dataset: {name!r}")


def build_lidar4d_test_dataset(cfg, *, input_offsets_sec=(-0.5, 0.5)):
    name = canonical_lidar4d_dataset_name(cfg)
    split = str(getattr(cfg, "test_split", "test"))
    if name == "nuscenes":
        from .nuscene_lidar4d_test import LiDAR4DNuScenesTestDataset
        dataset = LiDAR4DNuScenesTestDataset(
            cfg, split=split, input_offsets_sec=input_offsets_sec,
        )
        for meta in dataset.index_meta:
            meta["dataset_split"] = meta["nuscenes_split"]
        dataset.dataset_name = "nuscenes"
        return dataset
    if name == "kitti360":
        from .kitti360_lidar4d_test import LiDAR4DKITTI360TestDataset
        return LiDAR4DKITTI360TestDataset(
            cfg, split=split, input_offsets_sec=input_offsets_sec,
        )
