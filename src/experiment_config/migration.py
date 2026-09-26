"""Rewrite retired spellings in configs restored from checkpoints."""
from __future__ import annotations

from omegaconf import OmegaConf

from .definitions import LEGACY_EXP_SEED_MODES


def migrate_checkpoint_config(config):
    """Return ``config`` with retired key spellings replaced by current ones.

    ``p2g.grid_query.exp`` (``null``/``1``/``2``) became
    ``p2g.grid_query.seed_mode`` (``range_quantile``/``medoid``/``token``) with
    the same placement rules, so a checkpoint saved under the old name restores
    to the same model.
    """
    grid_query = OmegaConf.select(config, "p2g.grid_query", default=None)
    if not OmegaConf.is_dict(grid_query) or "exp" not in grid_query:
        return config
    if "seed_mode" in grid_query:
        raise ValueError(
            "p2g.grid_query declares both the retired exp and its replacement "
            "seed_mode"
        )
    exp = grid_query.exp
    key = None if exp is None else int(exp)
    if key not in LEGACY_EXP_SEED_MODES:
        raise ValueError(f"Unsupported legacy p2g.grid_query.exp={exp!r}")
    container = OmegaConf.to_container(config, resolve=False)
    stored = container["p2g"]["grid_query"]
    stored.pop("exp")
    stored["seed_mode"] = LEGACY_EXP_SEED_MODES[key]
    return OmegaConf.create(container)


__all__ = ["migrate_checkpoint_config"]
