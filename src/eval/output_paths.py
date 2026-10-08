"""Checkpoint-local result locations for the shared LiDAR4D evaluator."""
from pathlib import Path

from omegaconf import OmegaConf

from ..dataloader.lidar4d_test import canonical_lidar4d_dataset_name


def resolve_lidar4d_output_dir(cfg):
    """An explicit path wins; otherwise place results beside the checkpoint.

    Using the checkpoint parent avoids a stale/different logger.dir in a
    restored experiment config. Fresh runs without a checkpoint use logger.dir.
    """
    explicit = OmegaConf.select(cfg, "out_dir", default=None)
    if explicit is not None and str(explicit).strip():
        return Path(str(explicit)).expanduser().resolve()
    checkpoint = OmegaConf.select(cfg, "test.ckpt_path", default=None)
    if checkpoint:
        model_dir = Path(str(checkpoint)).expanduser().resolve().parent
    else:
        model_dir = Path(str(cfg.logger.dir)).expanduser().resolve()
    dataset = canonical_lidar4d_dataset_name(cfg.data)
    return model_dir / f"lidar4d_{dataset}_eval"
