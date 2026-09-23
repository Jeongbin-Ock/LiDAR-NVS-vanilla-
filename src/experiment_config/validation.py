"""Static validation for composed and checkpoint experiment configs."""
from __future__ import annotations

import math

from omegaconf import ListConfig, OmegaConf

from .definitions import (
    BARRIER_MATCH_DYNAMIC_VARIANTS,
    COSINE_ROPE_MATCH_DYNAMIC_VARIANTS,
    DYNAMIC_VARIANT_V12_1,
    FEATURE_HEAD_ROPE_MATCH_DYNAMIC_VARIANTS,
    FLASH_ROPE_MATCH_DYNAMIC_VARIANTS,
    LEARNED_COUNT_MODES,
    LEGACY_COUNT_MODE,
    RMSNORM_ROPE_MATCH_DYNAMIC_VARIANTS,
    ROUTER_CAPABLE_DYNAMIC_VARIANTS,
    VARIANT_CONFIG_PATHS,
    _REQUIRED_ROOTS,
    _UTONIA_ENCODER_STAGE_DIMS,
)
from .helpers import (
    _has_path,
    _reject_unknown_keys,
    _require_choice,
    _require_positive,
    _variant,
)


def _validate_3d_rope_width(dim: int, num_heads: int, label: str) -> None:
    if dim % num_heads != 0:
        raise ValueError(f"{label}: dim={dim} must be divisible by heads={num_heads}")
    head_dim = dim // num_heads
    if head_dim % 6 != 0:
        raise ValueError(
            f"{label}: head_dim={head_dim} must be divisible by six for "
            "the xyz/sin-cos 3D RoPE split"
        )


def _validate_utonia_lora_config(config, label: str) -> None:
    """Every variant feeds Utonia's XYZI tokens through LoRA adapters."""
    block = OmegaConf.select(config, "p2g.utonia_lora", default=None)
    if block is None:
        raise ValueError(f"{label} requires p2g.utonia_lora.enable=true")
    _reject_unknown_keys(
        config,
        "p2g.utonia_lora",
        {
            "enable", "input_mode", "rank", "linear_rank", "conv_rank",
            "alpha", "linear_alpha", "conv_alpha", "dropout",
        },
    )
    if not bool(OmegaConf.select(config, "p2g.utonia_lora.enable", default=False)):
        raise ValueError(f"{label} requires p2g.utonia_lora.enable=true")
    if not bool(OmegaConf.select(config, "p2g.freeze_utonia", default=True)):
        raise ValueError(
            "p2g.utonia_lora.enable=true requires p2g.freeze_utonia=true"
        )
    input_mode = str(OmegaConf.select(
        config, "p2g.utonia_lora.input_mode", default="xyzi"
    )).lower()
    if input_mode != "xyzi":
        raise ValueError(
            "p2g.utonia_lora.input_mode currently supports only 'xyzi'"
        )
    shared_rank = int(OmegaConf.select(
        config, "p2g.utonia_lora.rank", default=16
    ))
    for name in ("linear_rank", "conv_rank"):
        value = int(OmegaConf.select(
            config, f"p2g.utonia_lora.{name}", default=shared_rank
        ))
        if value <= 0:
            raise ValueError(f"p2g.utonia_lora.{name} must be positive")
    shared_alpha = float(OmegaConf.select(
        config, "p2g.utonia_lora.alpha", default=shared_rank
    ))
    for name in ("linear_alpha", "conv_alpha"):
        value = float(OmegaConf.select(
            config, f"p2g.utonia_lora.{name}", default=shared_alpha
        ))
        if value <= 0.0:
            raise ValueError(f"p2g.utonia_lora.{name} must be positive")
    dropout = float(OmegaConf.select(
        config, "p2g.utonia_lora.dropout", default=0.0
    ))
    if not 0.0 <= dropout < 1.0:
        raise ValueError("p2g.utonia_lora.dropout must be in [0, 1)")


def _configured_p2g_trunk_dim(config) -> int:
    """Token width entering temporal attention: the Utonia exit-stage width."""
    stage = int(OmegaConf.select(config, "p2g.utonia_feature_stage"))
    if not 0 <= stage < len(_UTONIA_ENCODER_STAGE_DIMS):
        raise ValueError(
            "p2g.utonia_feature_stage must be in [0, "
            f"{len(_UTONIA_ENCODER_STAGE_DIMS) - 1}], got {stage}"
        )
    return _UTONIA_ENCODER_STAGE_DIMS[stage]


def _allow_disabled_retired_block(config, path: str, switch: str, what: str):
    """Accept a removed feature's block only while it stays switched off.

    Checkpoints saved before ``what`` was removed still embed its block with
    the switch off, and resuming or evaluating them must keep working. A
    config that would turn the feature on is rejected rather than silently
    running without it.
    """
    block = OmegaConf.select(config, path, default=None)
    if block is None:
        return
    if not OmegaConf.is_dict(block):
        raise ValueError(f"{path} must be a config mapping")
    if bool(OmegaConf.select(config, f"{path}.{switch}", default=False)):
        raise ValueError(
            f"{path}.{switch}=true is no longer supported: {what} was removed"
        )


def _configured_count_mode(config) -> str:
    """The Gaussian-count contract ``p2g.grid_query`` asks for.

    An absent block is the historical fixed-count default: one Gaussian per
    occupied token, seeded at that token's observed medoid.
    """
    if not _has_path(config, "p2g.grid_query"):
        return LEGACY_COUNT_MODE
    mode = OmegaConf.select(config, "p2g.grid_query.count_mode")
    return LEGACY_COUNT_MODE if mode is None else str(mode).lower()


def _validate_fixed_gaussian_count(config, label: str) -> None:
    """The per-token Gaussian count and seed placement, when both are fixed.

    ``K_max`` is how many Gaussians every occupied token emits, and ``exp``
    says where they start: ``1`` on the token's observed medoid, ``2`` on its
    Utonia cell coordinate, and ``null`` spread over the token's own raw points
    by range quantile. Slots that coincide are separated by the independent
    parameter block each one owns in the Gaussian head.
    """
    count_path = "p2g.grid_query"
    if not _has_path(config, count_path):
        return
    _reject_unknown_keys(config, count_path, {"count_mode", "K_max", "exp"})
    k_max = OmegaConf.select(config, f"{count_path}.K_max")
    if k_max is None or int(k_max) < 1:
        raise ValueError(
            f"{label} requires a positive p2g.grid_query.K_max Gaussians "
            "per token"
        )
    exp = OmegaConf.select(config, f"{count_path}.exp")
    if exp is not None and int(exp) not in (1, 2):
        raise ValueError(f"{label} grid_query.exp must be null, 1, or 2")
    if exp is not None and int(exp) > int(k_max):
        raise ValueError(
            f"{label} grid_query.exp={int(exp)} requires K_max >= {int(exp)}"
        )


def _validate_dynamic_gaussian_count(config, variant, label: str) -> None:
    """Route the config to the count contract it selected, and check it fits."""
    count_mode = _configured_count_mode(config)
    if count_mode == LEGACY_COUNT_MODE:
        _validate_fixed_gaussian_count(config, label)
        return
    if count_mode not in LEARNED_COUNT_MODES:
        raise ValueError(
            f"{label} p2g.grid_query.count_mode must be 'legacy' or "
            "'learned_gumbel'"
        )
    if variant not in ROUTER_CAPABLE_DYNAMIC_VARIANTS:
        raise ValueError(
            f"{label} emits a fixed number of Gaussians per token and has no "
            f"count router; p2g.grid_query.count_mode={count_mode!r} is only "
            "available to "
            + ", ".join(
                _variant_label(name) for name in ROUTER_CAPABLE_DYNAMIC_VARIANTS
            )
        )
    _validate_adaptive_gaussian_count(config, label)


def _validate_adaptive_gaussian_count(config, label: str) -> None:
    """Shared learned_gumbel K={1,2,3} router contract for router variants."""

    count_path = "p2g.grid_query"
    _reject_unknown_keys(
        config,
        count_path,
        {"count_mode", "grad_balance", "learned_count"},
    )
    _require_choice(config, f"{count_path}.grad_balance", ("sqrt_k", "none"))
    learned_path = f"{count_path}.learned_count"
    _reject_unknown_keys(
        config,
        learned_path,
        {"K_max", "tau", "seed_mode", "grad_balance_scope", "budget", "logging"},
    )
    if int(OmegaConf.select(config, f"{learned_path}.K_max")) != 3:
        raise ValueError(f"{label} requires grid_query.learned_count.K_max=3")
    _require_positive(config, f"{learned_path}.tau")
    if str(OmegaConf.select(
        config, f"{learned_path}.seed_mode"
    )).lower() != "range_quantile":
        raise ValueError(
            f"{label} grid_query.learned_count.seed_mode must be "
            "'range_quantile'"
        )
    _require_choice(
        config, f"{learned_path}.grad_balance_scope", ("output", "token")
    )
    _allow_disabled_retired_block(
        config, f"{learned_path}.budget", "enable", "the routing budget loss"
    )
    logging_path = f"{learned_path}.logging"
    _reject_unknown_keys(
        config, logging_path, {"enable", "interval", "range_edges_m"}
    )
    _require_positive(config, f"{logging_path}.interval", kind=int)


def _variant_label(variant: str) -> str:
    """Short human label ("V11.4", "V13") for a registered variant identifier."""

    head, separator, tail = variant.rpartition("_v")
    if head and separator and tail.replace("_", "").isdigit():
        return "V" + tail.replace("_", ".")
    return variant


def _validate_dynamic_window_loader(config, variant: str) -> None:
    """Validate the random-duration data contract for fresh V11/V12 runs."""
    dataset_name = str(OmegaConf.select(config, "data.dataset_name", default=""))
    if dataset_name != "nuscenes_dynamic_lidar4d":
        # Historical V11 checkpoints embed ``nuscenes`` and must remain
        # reconstructable under their original fixed-duration loader.
        return
    if variant not in BARRIER_MATCH_DYNAMIC_VARIANTS:
        raise ValueError(
            "data.dataset_name=nuscenes_dynamic_lidar4d is reserved for the "
            "V11/V12 family"
        )
    if str(OmegaConf.select(config, "data.pair_mode", default="")) != "keyframe":
        raise ValueError("nuscenes_dynamic_lidar4d requires data.pair_mode=keyframe")
    if str(OmegaConf.select(config, "data.mode", default="")) != "nvs":
        raise ValueError("nuscenes_dynamic_lidar4d requires data.mode=nvs")

    path = "data.dynamic_window"
    if OmegaConf.select(config, path, default=None) is None:
        raise ValueError("nuscenes_dynamic_lidar4d requires data.dynamic_window")
    _reject_unknown_keys(
        config,
        path,
        {
            "endpoint_min_sec", "endpoint_max_sec", "eval_seed",
            "train_excluded_scenes",
        },
    )
    missing = [
        name for name in (
            "endpoint_min_sec", "endpoint_max_sec", "eval_seed",
            "train_excluded_scenes",
        )
        if not _has_path(config, f"{path}.{name}")
    ]
    if missing:
        raise ValueError(
            "data.dynamic_window is missing keys: " + ", ".join(missing)
        )
    minimum = float(OmegaConf.select(config, f"{path}.endpoint_min_sec"))
    maximum = float(OmegaConf.select(config, f"{path}.endpoint_max_sec"))
    if not math.isfinite(minimum) or not math.isfinite(maximum):
        raise ValueError("dynamic endpoint durations must be finite")
    if not 0.0 < minimum <= maximum:
        raise ValueError(
            "dynamic endpoint durations must satisfy 0 < min <= max"
        )
    sample_gap_us = int(OmegaConf.select(config, "data.sample_gap_us"))
    if sample_gap_us <= 0 or sample_gap_us % 50_000 != 0:
        raise ValueError(
            "nuscenes_dynamic_lidar4d sample_gap_us must be a positive multiple of 50000"
        )
    if int(OmegaConf.select(config, "data.gt_middle_count")) < 0:
        raise ValueError(
            "nuscenes_dynamic_lidar4d gt_middle_count must be non-negative"
        )
    if int(OmegaConf.select(config, "data.pair_kf_stride")) <= 0:
        raise ValueError(
            "nuscenes_dynamic_lidar4d pair_kf_stride must be positive"
        )
    int(OmegaConf.select(config, f"{path}.eval_seed"))
    excluded_scenes = list(OmegaConf.select(
        config, f"{path}.train_excluded_scenes"
    ))
    if not excluded_scenes:
        raise ValueError(
            "nuscenes_dynamic_lidar4d requires at least one held-out train scene"
        )
    if any(not str(scene).strip() for scene in excluded_scenes):
        raise ValueError("train_excluded_scenes entries must be non-empty")
    if len(excluded_scenes) != len(set(map(str, excluded_scenes))):
        raise ValueError("train_excluded_scenes must not contain duplicates")


def _validate_legacy_train_exclusions(config) -> None:
    """Validate whole-scene holdouts consumed by the legacy nuscene.py loader."""
    path = "data.train_excluded_scenes"
    excluded = OmegaConf.select(config, path, default=None)
    if excluded is None:
        return
    if str(OmegaConf.select(config, "data.dataset_name")) != "nuscenes":
        raise ValueError(f"{path} is only supported by data.dataset_name=nuscenes")
    if not isinstance(excluded, (list, tuple, ListConfig)):
        raise ValueError(f"{path} must be a list of scene names")
    names = [str(name) for name in excluded]
    if any(not name.strip() for name in names):
        raise ValueError(f"{path} entries must be non-empty")
    if len(names) != len(set(names)):
        raise ValueError(f"{path} must not contain duplicates")


def _validate_temporal_config(config, variant: str, label: str) -> None:
    """The temporal-attention keys and contract of the variant's family."""
    temporal_path = "dynamic_2dgs.temporal"
    temporal_keys = {
        "implementation", "layers", "num_heads", "mlp_ratio",
        "use_time_embedding", "time_frequencies", "time_embedding_dim",
        "time_reference_sec", "position_encoding", "rope_base",
        "rope_position_scale", "qk_norm", "layer_scale_init",
    }
    if variant in FLASH_ROPE_MATCH_DYNAMIC_VARIANTS:
        if variant in COSINE_ROPE_MATCH_DYNAMIC_VARIANTS:
            temporal_keys.add("motion_temperature")
    else:
        temporal_keys.update({
            "barrier_speed_mps", "barrier_weight", "barrier_exponent",
            "match_chunk_size",
        })
        if variant == DYNAMIC_VARIANT_V12_1:
            temporal_keys.add("barrier_lambda_init")
    _reject_unknown_keys(config, temporal_path, temporal_keys)

    if not bool(OmegaConf.select(
        config, f"{temporal_path}.use_time_embedding"
    )):
        raise ValueError(
            f"{label} requires dynamic_2dgs.temporal.use_time_embedding=true"
        )

    if variant in FLASH_ROPE_MATCH_DYNAMIC_VARIANTS:
        expected_position_encoding = (
            "rope_except_head0"
            if variant in FEATURE_HEAD_ROPE_MATCH_DYNAMIC_VARIANTS
            else "rope_all_heads"
        )
        if str(OmegaConf.select(
            config, f"{temporal_path}.position_encoding"
        )).lower() != expected_position_encoding:
            raise ValueError(
                f"{label} temporal.position_encoding must be "
                f"'{expected_position_encoding}'"
            )
        qk_norm = OmegaConf.select(
            config, f"{temporal_path}.qk_norm", default=None
        )
        expected_qk_norm = variant in RMSNORM_ROPE_MATCH_DYNAMIC_VARIANTS
        if qk_norm is None or bool(qk_norm) != expected_qk_norm:
            raise ValueError(
                f"{label} requires temporal.qk_norm="
                f"{str(expected_qk_norm).lower()}"
            )
        if variant in COSINE_ROPE_MATCH_DYNAMIC_VARIANTS:
            temperature_value = OmegaConf.select(
                config, f"{temporal_path}.motion_temperature"
            )
            if temperature_value is None or not (
                math.isfinite(float(temperature_value))
                and float(temperature_value) > 0.0
            ):
                raise ValueError(
                    f"{label} temporal.motion_temperature must be "
                    "positive and finite"
                )
        return

    if str(OmegaConf.select(
        config, f"{temporal_path}.position_encoding"
    )).lower() != "barrier_rope_split":
        raise ValueError(
            f"{label} temporal.position_encoding must be 'barrier_rope_split'"
        )
    _require_positive(config, f"{temporal_path}.barrier_speed_mps")
    if variant == DYNAMIC_VARIANT_V12_1:
        _require_positive(config, f"{temporal_path}.barrier_lambda_init")
    if float(OmegaConf.select(
        config, f"{temporal_path}.barrier_weight"
    )) < 0.0:
        raise ValueError(
            f"{label} temporal.barrier_weight must be non-negative"
        )
    # The hinge grows quadratically (V11's shipped 2) or linearly (V11.4's 1)
    # past the radius. Nothing else folds into the score GEMM's beta
    # accumulate, so nothing else is supported.
    barrier_exponent = OmegaConf.select(
        config, f"{temporal_path}.barrier_exponent"
    )
    if barrier_exponent is not None and int(barrier_exponent) not in (1, 2):
        raise ValueError(
            f"{label} temporal.barrier_exponent must be 1 or 2"
        )
    _require_positive(config, f"{temporal_path}.match_chunk_size", kind=int)
    if not bool(OmegaConf.select(config, f"{temporal_path}.qk_norm")):
        raise ValueError(f"{label} requires temporal.qk_norm=true")
    # Head 0 is the match head; every remaining head keeps 3D RoPE, so every
    # barrier-match variant needs at least one of each.
    if int(OmegaConf.select(config, f"{temporal_path}.num_heads")) < 2:
        raise ValueError(
            f"{label} requires dynamic_2dgs.temporal.num_heads >= 2"
        )


def _validate_motion_regularization(config) -> None:
    """The velocity prior: independent squared norms on v_init and v_offset."""
    regularization_path = "dynamic_2dgs.regularization"
    _reject_unknown_keys(
        config, regularization_path, {"small_motion", "velocity_l2"},
    )
    _allow_disabled_retired_block(
        config, f"{regularization_path}.small_motion", "enabled",
        "the small-motion displacement prior",
    )
    velocity_l2_path = f"{regularization_path}.velocity_l2"
    if not _has_path(config, velocity_l2_path):
        return
    _reject_unknown_keys(
        config,
        velocity_l2_path,
        {
            "enabled", "mode", "init_weight", "offset_weight",
            "warmup_steps", "ramp_steps",
        },
    )
    mode = OmegaConf.select(config, f"{velocity_l2_path}.mode")
    if str(mode) != "split_group_l2":
        raise ValueError(
            "dynamic_2dgs.regularization.velocity_l2.mode must be "
            "'split_group_l2'"
        )
    for name in ("init_weight", "offset_weight"):
        value = OmegaConf.select(config, f"{velocity_l2_path}.{name}")
        if value is None or float(value) < 0.0:
            raise ValueError(
                "velocity_l2.mode='split_group_l2' requires a "
                f"non-negative velocity_l2.{name}"
            )


def validate_experiment_config(config) -> None:
    missing = [key for key in _REQUIRED_ROOTS if key not in config]
    if missing:
        raise ValueError("Experiment config is missing root keys: " + ", ".join(missing))

    variant = _variant(config)
    if variant not in VARIANT_CONFIG_PATHS:
        allowed = ", ".join(sorted(VARIANT_CONFIG_PATHS))
        raise ValueError(
            f"Unknown model.variant={variant!r}; expected one of: {allowed}"
        )
    label = _variant_label(variant)

    _validate_dynamic_window_loader(config, variant)
    _validate_legacy_train_exclusions(config)
    _validate_utonia_lora_config(config, label)

    # Checkpoints embed the anchor mode from when a second one existed.
    anchor_mode = OmegaConf.select(config, "p2g.anchor_mode", default=None)
    if anchor_mode is not None and str(anchor_mode) != "grid":
        raise ValueError(f"{variant} requires p2g.anchor_mode=grid")
    if OmegaConf.select(config, "dynamic_2dgs", default=None) is None:
        raise ValueError(f"{variant} requires a dynamic_2dgs config block")
    # How many Gaussians a token becomes is a config decision, not a property
    # of the variant name: p2g.grid_query selects a fixed count per token or,
    # where the backend has a router, a predicted one.
    _validate_dynamic_gaussian_count(config, variant, label)
    temporal_heads = int(OmegaConf.select(
        config, "dynamic_2dgs.temporal.num_heads"
    ))
    _validate_3d_rope_width(
        _configured_p2g_trunk_dim(config), temporal_heads,
        "dynamic temporal attention",
    )
    _reject_unknown_keys(
        config,
        "dynamic_2dgs",
        {"temporal", "gaussian_head", "motion", "regularization"},
    )
    _validate_temporal_config(config, variant, label)
    _reject_unknown_keys(
        config,
        "dynamic_2dgs.gaussian_head",
        {"initial_opacity", "initial_scale_m"},
    )
    # Every offset head embeds the v_init it corrects.
    _reject_unknown_keys(
        config,
        "dynamic_2dgs.motion",
        {
            "zero_init", "residual_hidden_dim", "velocity_embedding_dim",
            "detach_init_condition",
        },
    )
    if _has_path(config, "dynamic_2dgs.motion.residual_hidden_dim"):
        residual_hidden_dim = OmegaConf.select(
            config, "dynamic_2dgs.motion.residual_hidden_dim"
        )
        if residual_hidden_dim is not None and int(residual_hidden_dim) <= 0:
            raise ValueError(
                "dynamic_2dgs.motion.residual_hidden_dim must be positive "
                "when set"
            )
    _validate_motion_regularization(config)
