"""Compose experiment configs and restore checkpoint-local configs safely.

The project intentionally keeps OmegaConf dot-list CLI overrides (``key=value``)
instead of introducing a second configuration framework. Fresh runs compose the
dataset-agnostic config/default.yaml, one allow-listed model-variant overlay and
one data profile from config/data/. Checkpoint-backed runs prefer embedded
configs and retain the historical W&B fallback.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from .definitions import (
    ALL_HEAD_ROPE_MATCH_DYNAMIC_VARIANTS,
    BARRIER_MATCH_DYNAMIC_VARIANTS,
    COSINE_ROPE_MATCH_DYNAMIC_VARIANTS,
    DATA_CONFIG_DIR,
    DEFAULT_CONFIG_PATH,
    DENSE_COUNT_DYNAMIC_VARIANTS,
    DENSE_COUNT_GAUSSIANS_PER_TOKEN,
    DENSE_COUNT_TEMPORAL_HEADS_BY_VARIANT,
    DYNAMIC_VARIANT,
    DYNAMIC_VARIANT_V11,
    DYNAMIC_VARIANT_V11_1,
    DYNAMIC_VARIANT_V11_2,
    DYNAMIC_VARIANT_V11_3,
    DYNAMIC_VARIANT_V11_4,
    DYNAMIC_VARIANT_V11_5,
    DYNAMIC_VARIANT_V11_6,
    DYNAMIC_VARIANT_V12,
    DYNAMIC_VARIANT_V12_1,
    DYNAMIC_VARIANT_V12_2,
    DYNAMIC_VARIANT_V13,
    DYNAMIC_VARIANT_V13_1,
    DYNAMIC_VARIANT_V13_2,
    DYNAMIC_VARIANT_V13_3,
    DYNAMIC_VARIANT_V13_4,
    DYNAMIC_VARIANT_V14,
    DYNAMIC_VARIANT_V14_1,
    DYNAMIC_VARIANT_V15,
    DYNAMIC_VARIANT_V15_1,
    DYNAMIC_VARIANT_V15_2,
    DYNAMIC_VARIANT_V15_3,
    DYNAMIC_VARIANT_V15_4,
    DYNAMIC_VARIANT_V16,
    DYNAMIC_VARIANT_V16_1,
    DYNAMIC_VARIANTS,
    EXPERT_AUXILIARY_RENDER_BY_VARIANT,
    EXPERT_AUXILIARY_RENDERS,
    EXPERT_COUNT_DYNAMIC_VARIANTS,
    EXPERT_COUNT_MODE,
    EXPERT_COUNT_OPTION_COUNTS,
    FEATURE_HEAD_ROPE_MATCH_DYNAMIC_VARIANTS,
    FLASH_ROPE_MATCH_DYNAMIC_VARIANTS,
    HYBRID_UTONIA_DYNAMIC_VARIANTS,
    LEARNED_COUNT_MODES,
    LEGACY_COUNT_MODE,
    LOCAL_ATTENTION_DYNAMIC_VARIANTS,
    LOCAL_ATTENTION_IMPLEMENTATIONS,
    NESTED_AUXILIARY_RENDER_BY_VARIANT,
    NESTED_AUXILIARY_RENDERS,
    NESTED_COUNT_DYNAMIC_VARIANTS,
    NESTED_COUNT_MODE,
    NESTED_COUNT_OPTION_COUNTS,
    NO_TEMPORAL_LAYER_SCALE_DYNAMIC_VARIANTS,
    ORDINAL_AUXILIARY_RENDER_BY_VARIANT,
    ORDINAL_AUXILIARY_RENDERS,
    ORDINAL_COUNT_DYNAMIC_VARIANTS,
    ORDINAL_COUNT_MODE,
    ORDINAL_COUNT_OPTION_COUNTS,
    POST_ACTIVATION_STEM_DYNAMIC_VARIANTS,
    REPO_ROOT,
    RMSNORM_ROPE_MATCH_DYNAMIC_VARIANTS,
    ROUTED_COUNT_BLOCKS,
    ROUTER_CAPABLE_DYNAMIC_VARIANTS,
    STRUCTURAL_DEFAULTS,
    V15_FAMILY_DYNAMIC_VARIANTS,
    VARIANT_CONFIG_PATHS,
    _CHECKPOINT_PROTECTED_PATHS,
    _CHECKPOINT_STRUCTURAL_ROOTS,
    _REQUIRED_ROOTS,
    _UTONIA_ENCODER_STAGE_DIMS,
)
from .helpers import (
    _as_config,
    _as_experiment_config,
    _has_path,
    _is_experiment_config,
    _reject_unknown_keys,
    _require_choice,
    _require_in_open_range,
    _require_positive,
    _variant,
)
from .migration import migrate_checkpoint_config
from .validation import (
    _configured_count_mode,
    _configured_p2g_trunk_dim,
    _validate_3d_rope_width,
    _validate_adaptive_gaussian_count,
    _validate_dynamic_gaussian_count,
    _validate_fixed_gaussian_count,
    _validate_utonia_lora_config,
    _variant_label,
    validate_experiment_config,
)


def _load_variant_overlay(variant: str) -> tuple[DictConfig, Path]:
    try:
        path = VARIANT_CONFIG_PATHS[variant]
    except KeyError as exc:
        allowed = ", ".join(sorted(VARIANT_CONFIG_PATHS))
        raise ValueError(
            f"Unknown model.variant={variant!r}; expected one of: {allowed}"
        ) from exc
    if not path.is_file():
        raise FileNotFoundError(f"Variant config does not exist: {path}")
    overlay = OmegaConf.load(path)
    declared = _variant(overlay, default="")
    if declared != variant:
        raise ValueError(
            f"Variant overlay {path} declares model.variant={declared!r}, "
            f"expected {variant!r}"
        )
    return overlay, path


# One suffix per KITTI-360 GT mode (data.panorama_projection): the calibrated
# HDL-64E rows (data_config=kitti360) and the LiDAR4D uniform grid
# (data_config=kitti360_lidar4d).
KITTI360_EXP_NAME_SUFFIXES = {
    "hdl64e_kitti360_calibrated": "_kitti",
    "uniform": "_kitti_lidar4d",
}


def _kitti360_exp_name(config, *, named: bool) -> str:
    """A KITTI-360 run's W&B name and logs/ folder always carry its GT mode.

    An unnamed run takes the variant's short label plus the mode suffix
    (``v15.3_kitti`` calibrated, ``v15.3_kitti_lidar4d`` LiDAR4D grid). A given
    name gains the suffix unless it already mentions kitti, so a KITTI-360 run
    never writes into a nuScenes run's ``logs/<exp_name>`` checkpoints and the
    two GT modes never share one by default.
    """
    suffix = KITTI360_EXP_NAME_SUFFIXES[str(config.data.panorama_projection)]
    name = str(config.exp_name) if named else _variant_label(_variant(config)).lower()
    return name if "kitti" in name.lower() else name + suffix


def resolve_data_profile(name) -> Path:
    """``kitti360`` -> config/data/kitti360.yaml; any other value is a file path."""

    text = str(name)
    candidate = DATA_CONFIG_DIR / f"{text}.yaml"
    if "/" not in text and not text.endswith((".yaml", ".yml")) and candidate.is_file():
        return candidate
    path = Path(text).expanduser().resolve()
    if not path.is_file():
        known = ", ".join(sorted(p.stem for p in DATA_CONFIG_DIR.glob("*.yaml")))
        raise FileNotFoundError(f"Unknown data_config={text!r}; expected one of {known} or a profile path")
    return path


def compose_fresh_config(cli=None) -> tuple[DictConfig, str]:
    """Compose default.yaml -> model variant -> data profile -> CLI.

    The variant overlay is model-only and names its default data profile with
    ``data_config``; a CLI ``data_config=`` (profile name or path) overrides
    it. The profile replaces the whole data block, and ``data.*`` CLI keys
    still apply on top.
    """

    cli = OmegaConf.create(OmegaConf.to_container(_as_config(cli), resolve=False))
    cli_profile = cli.pop("data_config", None)
    named = "exp_name" in cli
    base = OmegaConf.load(DEFAULT_CONFIG_PATH)
    if "data" in base or "data_config" in base:
        raise ValueError(f"{DEFAULT_CONFIG_PATH} is dataset-agnostic; data belongs in {DATA_CONFIG_DIR}")
    selected = _variant(cli, default=_variant(base))
    overlay, overlay_path = _load_variant_overlay(selected)
    if "data" in overlay:
        raise ValueError(f"{overlay_path} is a model overlay; move its data block into {DATA_CONFIG_DIR}")
    variant_profile = overlay.pop("data_config", None)
    if not (cli_profile or variant_profile):
        raise ValueError(f"No data profile: {overlay_path} names none and the CLI passed no data_config=")
    profile_path = resolve_data_profile(cli_profile or variant_profile)
    profile = OmegaConf.load(profile_path)
    unknown = set(profile) - {"data", "exp_name"}
    if unknown or not OmegaConf.is_dict(profile.get("data")):
        raise ValueError(f"{profile_path}: a data profile holds data and can only add exp_name")
    config = OmegaConf.merge(
        OmegaConf.create(STRUCTURAL_DEFAULTS),
        base,
        overlay,
    )
    config.data = profile.data
    if "exp_name" in profile:
        config.exp_name = OmegaConf.to_container(profile, resolve=False)["exp_name"]
        named = True
    data_source = f" data_profile={profile_path.resolve()}"
    config = OmegaConf.merge(config, cli)
    if str(config.data.dataset_name) == "kitti360_dynamic":
        manifest = Path(str(config.data.split_manifest_path))
        digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
        expected = OmegaConf.select(config, "data.split_manifest_sha256")
        if expected is not None and str(expected) != digest:
            raise ValueError("KITTI360 split manifest hash differs from data.split_manifest_sha256")
        config.data.split_manifest_sha256 = digest
        config.exp_name = _kitti360_exp_name(config, named=named)
    if _variant(config) != selected:
        raise ValueError(
            "The composed model.variant changed after selecting its overlay: "
            f"selected={selected!r}, composed={_variant(config)!r}"
        )
    validate_experiment_config(config)
    source = f"base={DEFAULT_CONFIG_PATH.resolve()} overlay={overlay_path.resolve()}{data_source}"
    return config, source


def _load_embedded_checkpoint_config(checkpoint: Path):
    """Read config metadata embedded by ``ModelWrapper.on_save_checkpoint``."""

    try:
        import torch

        try:
            blob = torch.load(
                checkpoint,
                map_location="cpu",
                weights_only=False,
                mmap=True,
            )
        except TypeError:
            blob = torch.load(checkpoint, map_location="cpu")
    except Exception:
        # Historical checkpoints and lightweight test fixtures may not contain
        # readable checkpoint metadata. Their W&B artifact is the fallback.
        return None

    if not isinstance(blob, Mapping) or "experiment_config" not in blob:
        return None
    config = _as_experiment_config(blob["experiment_config"])
    if config is None:
        raise RuntimeError(
            f"Checkpoint contains invalid experiment_config metadata: {checkpoint}"
        )
    return config


def load_checkpoint_config(
    checkpoint,
    *,
    allow_legacy_wandb_fallback: bool = False,
) -> tuple[DictConfig, str]:
    """Load checkpoint-bound config, with a historical W&B fallback."""

    checkpoint = Path(str(checkpoint)).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")

    embedded = _load_embedded_checkpoint_config(checkpoint)
    if embedded is not None:
        config = migrate_checkpoint_config(
            OmegaConf.merge(OmegaConf.create(STRUCTURAL_DEFAULTS), embedded)
        )
        validate_experiment_config(config)
        return config, f"embedded checkpoint config: {checkpoint}"

    if not allow_legacy_wandb_fallback:
        raise RuntimeError(
            "Historical checkpoint has no embedded experiment config. Its only "
            "available config source is the mutable wandb/latest-run link, which "
            "may belong to a later resumed run. Inspect that artifact and pass "
            "allow_legacy_wandb_fallback=true to opt in explicitly."
        )

    config_file = checkpoint.parent / "wandb/latest-run/files/config.yaml"
    if not config_file.is_file():
        raise FileNotFoundError(
            "Checkpoint execution needs its run-local W&B config, but it was "
            f"not found: {config_file}"
        )

    raw = OmegaConf.to_container(OmegaConf.load(config_file), resolve=False)
    if not isinstance(raw, Mapping):
        raise RuntimeError(f"Unexpected W&B config format: {config_file}")
    unwrapped = {
        key: value["value"]
        if isinstance(value, Mapping) and "value" in value
        else value
        for key, value in raw.items()
        if key != "_wandb"
    }
    stored = _as_experiment_config(unwrapped)
    if stored is None:
        raise RuntimeError(
            f"W&B config does not contain a valid experiment config: {config_file}"
        )

    config = migrate_checkpoint_config(
        OmegaConf.merge(OmegaConf.create(STRUCTURAL_DEFAULTS), stored)
    )
    validate_experiment_config(config)
    return config, (
        "legacy W&B fallback (mutable latest-run): "
        f"{config_file.resolve()}"
    )


def _merge_checkpoint_cli(stored, cli) -> DictConfig:
    cli = _as_config(cli)
    config = OmegaConf.merge(stored, cli)
    changed_roots = []
    for root in _CHECKPOINT_STRUCTURAL_ROOTS:
        if root not in cli:
            continue
        before = OmegaConf.select(stored, root, default=None)
        after = OmegaConf.select(config, root, default=None)
        before_value = (
            OmegaConf.to_container(before, resolve=True)
            if OmegaConf.is_config(before)
            else before
        )
        after_value = (
            OmegaConf.to_container(after, resolve=True)
            if OmegaConf.is_config(after)
            else after
        )
        if before_value != after_value:
            changed_roots.append(root)
    changed_paths = []
    for path in _CHECKPOINT_PROTECTED_PATHS:
        if not _has_path(cli, path):
            continue
        before = OmegaConf.select(stored, path, default=None)
        after = OmegaConf.select(config, path, default=None)
        if before != after:
            changed_paths.append(path)
    if changed_roots or changed_paths:
        details = []
        if changed_roots:
            details.append("roots: " + ", ".join(changed_roots))
        if changed_paths:
            details.append("paths: " + ", ".join(changed_paths))
        raise ValueError(
            "Checkpoint structural config cannot be changed by CLI; changed "
            + "; ".join(details)
        )
    validate_experiment_config(config)
    return config


def _resolve_with_checkpoint(cli, checkpoint_key: str) -> tuple[DictConfig, str]:
    cli = _as_config(cli)
    checkpoint = OmegaConf.select(cli, checkpoint_key, default=None)
    if not checkpoint:
        return compose_fresh_config(cli)
    if OmegaConf.select(cli, "data_config") is not None:
        raise ValueError("data_config is for fresh runs; checkpoint runs restore their protected data config")
    allow_legacy = bool(
        OmegaConf.select(cli, "allow_legacy_wandb_fallback", default=False)
    )
    stored, source = load_checkpoint_config(
        checkpoint,
        allow_legacy_wandb_fallback=allow_legacy,
    )
    return _merge_checkpoint_cli(stored, cli), source


def resolve_main_config(cli=None) -> tuple[DictConfig, str]:
    """Resolve train/test entrypoint config, including exact resume configs."""

    cli = _as_config(cli)
    mode = str(OmegaConf.select(cli, "mode", default="train")).lower()
    checkpoint_key = "train.ckpt_path" if mode == "train" else "test.ckpt_path"
    return _resolve_with_checkpoint(cli, checkpoint_key)


def resolve_eval_config(cli=None) -> tuple[DictConfig, str]:
    """Restore model weights/config, optionally applying an inference data file.

    ``eval_config=...yaml`` is an explicit dataset/rendering overlay. Training
    and ordinary checkpoint resumes retain their existing structural guards.
    Model roots and the checkpoint's local coordinate convention stay locked.
    """
    cli = _as_config(cli)
    overlay_path = OmegaConf.select(cli, "eval_config", default=None)
    if not overlay_path:
        return _resolve_with_checkpoint(cli, "test.ckpt_path")
    cli = OmegaConf.create(OmegaConf.to_container(cli, resolve=False))
    del cli["eval_config"]
    path = Path(str(overlay_path)).expanduser().resolve()
    overlay = OmegaConf.merge(OmegaConf.load(path), cli)
    allowed = {
        "mode", "data", "test", "device", "seed", "out_dir", "limit",
        "depth_statistic", "allow_legacy_wandb_fallback",
    }
    unknown = set(overlay) - allowed
    if unknown:
        raise ValueError("Inference overlay cannot change model/training roots: "
                         + ", ".join(sorted(unknown)))
    if str(overlay.get("mode", "test")) != "test":
        raise ValueError("eval_config is inference-only and requires mode=test")
    checkpoint = OmegaConf.select(overlay, "test.ckpt_path", default=None)
    if not checkpoint:
        raise ValueError("eval_config requires test.ckpt_path")
    stored, source = load_checkpoint_config(
        checkpoint,
        allow_legacy_wandb_fallback=bool(overlay.get("allow_legacy_wandb_fallback", False)),
    )
    config = OmegaConf.merge(stored, overlay, {"mode": "test"})
    if config.data.coordinate_frame != stored.data.coordinate_frame:
        raise ValueError("Inference must preserve checkpoint data.coordinate_frame")
    validate_experiment_config(config)
    return config, f"{source}; inference data overlay: {path}"

__all__ = [
    "ALL_HEAD_ROPE_MATCH_DYNAMIC_VARIANTS",
    "BARRIER_MATCH_DYNAMIC_VARIANTS",
    "COSINE_ROPE_MATCH_DYNAMIC_VARIANTS",
    "DATA_CONFIG_DIR",
    "DEFAULT_CONFIG_PATH",
    "DENSE_COUNT_DYNAMIC_VARIANTS",
    "DENSE_COUNT_GAUSSIANS_PER_TOKEN",
    "DENSE_COUNT_TEMPORAL_HEADS_BY_VARIANT",
    "DYNAMIC_VARIANT",
    "DYNAMIC_VARIANT_V11",
    "DYNAMIC_VARIANT_V11_1",
    "DYNAMIC_VARIANT_V11_2",
    "DYNAMIC_VARIANT_V11_3",
    "DYNAMIC_VARIANT_V11_4",
    "DYNAMIC_VARIANT_V11_5",
    "DYNAMIC_VARIANT_V11_6",
    "DYNAMIC_VARIANT_V12",
    "DYNAMIC_VARIANT_V12_1",
    "DYNAMIC_VARIANT_V12_2",
    "DYNAMIC_VARIANT_V13",
    "DYNAMIC_VARIANT_V13_1",
    "DYNAMIC_VARIANT_V13_2",
    "DYNAMIC_VARIANT_V13_3",
    "DYNAMIC_VARIANT_V13_4",
    "DYNAMIC_VARIANT_V14",
    "DYNAMIC_VARIANT_V14_1",
    "DYNAMIC_VARIANT_V15",
    "DYNAMIC_VARIANT_V15_1",
    "DYNAMIC_VARIANT_V15_2",
    "DYNAMIC_VARIANT_V15_3",
    "DYNAMIC_VARIANT_V15_4",
    "DYNAMIC_VARIANT_V16",
    "DYNAMIC_VARIANT_V16_1",
    "DYNAMIC_VARIANTS",
    "EXPERT_AUXILIARY_RENDER_BY_VARIANT",
    "EXPERT_AUXILIARY_RENDERS",
    "EXPERT_COUNT_DYNAMIC_VARIANTS",
    "EXPERT_COUNT_MODE",
    "EXPERT_COUNT_OPTION_COUNTS",
    "FEATURE_HEAD_ROPE_MATCH_DYNAMIC_VARIANTS",
    "FLASH_ROPE_MATCH_DYNAMIC_VARIANTS",
    "HYBRID_UTONIA_DYNAMIC_VARIANTS",
    "LOCAL_ATTENTION_DYNAMIC_VARIANTS",
    "LOCAL_ATTENTION_IMPLEMENTATIONS",
    "NESTED_AUXILIARY_RENDER_BY_VARIANT",
    "NESTED_AUXILIARY_RENDERS",
    "NESTED_COUNT_DYNAMIC_VARIANTS",
    "NESTED_COUNT_MODE",
    "NESTED_COUNT_OPTION_COUNTS",
    "NO_TEMPORAL_LAYER_SCALE_DYNAMIC_VARIANTS",
    "ORDINAL_AUXILIARY_RENDER_BY_VARIANT",
    "ORDINAL_AUXILIARY_RENDERS",
    "ORDINAL_COUNT_DYNAMIC_VARIANTS",
    "ORDINAL_COUNT_MODE",
    "ORDINAL_COUNT_OPTION_COUNTS",
    "POST_ACTIVATION_STEM_DYNAMIC_VARIANTS",
    "RMSNORM_ROPE_MATCH_DYNAMIC_VARIANTS",
    "ROUTED_COUNT_BLOCKS",
    "ROUTER_CAPABLE_DYNAMIC_VARIANTS",
    "STRUCTURAL_DEFAULTS",
    "V15_FAMILY_DYNAMIC_VARIANTS",
    "VARIANT_CONFIG_PATHS",
    "compose_fresh_config",
    "resolve_data_profile",
    "load_checkpoint_config",
    "resolve_eval_config",
    "resolve_main_config",
    "validate_experiment_config",
]
