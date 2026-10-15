"""Static validation for composed and checkpoint experiment configs."""
from __future__ import annotations

import math
from typing import NamedTuple

from omegaconf import ListConfig, OmegaConf

from .definitions import (
    BARRIER_MATCH_DYNAMIC_VARIANTS,
    COSINE_ROPE_MATCH_DYNAMIC_VARIANTS,
    COORDINATE_FRAMES,
    DENSE_COUNT_DYNAMIC_VARIANTS,
    DENSE_COUNT_GAUSSIANS_PER_TOKEN,
    DENSE_COUNT_TEMPORAL_HEADS_BY_VARIANT,
    DYNAMIC_VARIANT_V12_1,
    EXPERT_AUXILIARY_RENDER_BY_VARIANT,
    EXPERT_AUXILIARY_RENDERS,
    EXPERT_COUNT_DYNAMIC_VARIANTS,
    EXPERT_COUNT_MODE,
    EXPERT_COUNT_OPTION_COUNTS,
    FEATURE_HEAD_ROPE_MATCH_DYNAMIC_VARIANTS,
    FIXED_COUNT_SEED_MODES,
    FLASH_ROPE_MATCH_DYNAMIC_VARIANTS,
    HYBRID_UTONIA_DYNAMIC_VARIANTS,
    LEARNED_COUNT_MODES,
    LEARNED_COUNT_SEED_MODES,
    LEGACY_COUNT_MODE,
    LOCAL_ATTENTION_DYNAMIC_VARIANTS,
    LOCAL_ATTENTION_IMPLEMENTATIONS,
    LOCAL_ATTENTION_MAX_QUERY_CHUNK_SIZE,
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
    RMSNORM_ROPE_MATCH_DYNAMIC_VARIANTS,
    ROUTED_COUNT_BLOCKS,
    ROUTER_CAPABLE_DYNAMIC_VARIANTS,
    SEED_MODE_TOKEN,
    UTONIA_ADAPTATION_MODE_HYBRID,
    UTONIA_ADAPTER_ACTIVATIONS,
    UTONIA_LORA_TARGETS,
    UTONIA_PARAM_GROUP_ROLES,
    UTONIA_STEM_ADAPTER_POSITIONS,
    VARIANT_CONFIG_PATHS,
    _REQUIRED_ROOTS,
    _UTONIA_ENCODER_STAGE_DIMS,
)
from .helpers import (
    _has_path,
    _reject_unknown_keys,
    _require_choice,
    _require_in_open_range,
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


def _require_flag(config, path: str) -> None:
    """The named path must hold a YAML boolean."""
    if not isinstance(OmegaConf.select(config, path), bool):
        raise ValueError(f"{path} must be true or false")


def _require_finite(config, path: str, *, minimum: float,
                    inclusive: bool) -> None:
    value = OmegaConf.select(config, path)
    if value is None or not math.isfinite(float(value)) or (
        float(value) < minimum if inclusive else float(value) <= minimum
    ):
        bound = "non-negative" if inclusive else "positive"
        raise ValueError(f"{path} must be {bound} and finite")


def _require_int(config, path: str) -> int:
    """The named path must hold a positive YAML integer (not a float or bool)."""
    value = OmegaConf.select(config, path)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{path} must be a positive integer")
    return value


def _only_available_to(variants) -> str:
    """``V15`` / ``V14, V14.1, V15``: who a variant-gated key is reserved for."""
    return ", ".join(_variant_label(name) for name in variants)


def _validate_utonia_adaptation_config(config, variant, label: str) -> None:
    """The hybrid Utonia adaptation block and its AdamW role groups.

    V14-family variants require ``p2g.utonia_adaptation`` (mode ``hybrid``)
    and ``train.utonia_param_groups``. Every other variant adapts Utonia with
    LoRA on every Linear/SubMConv3d and rejects both blocks, so neither can be
    carried along silently unused. Within the family, ``stem.adapter_position``
    is the one key that tells V15's post-activation stem adapter from
    V14/V14.1's pre-LayerNorm one.
    """
    path = "p2g.utonia_adaptation"
    groups_path = "train.utonia_param_groups"
    block = OmegaConf.select(config, path, default=None)
    groups = OmegaConf.select(config, groups_path, default=None)
    if variant not in HYBRID_UTONIA_DYNAMIC_VARIANTS:
        allowed = ", ".join(
            _variant_label(name) for name in HYBRID_UTONIA_DYNAMIC_VARIANTS
        )
        if block is not None:
            raise ValueError(
                f"{label} adapts Utonia with LoRA on every Linear/SubMConv3d; "
                f"{path} is only available to {allowed}"
            )
        if groups is not None:
            raise ValueError(
                f"{groups_path} only applies to the hybrid Utonia adaptation "
                f"of {allowed}"
            )
        return
    if block is None:
        raise ValueError(
            f"{label} requires {path} with "
            f"mode={UTONIA_ADAPTATION_MODE_HYBRID!r}"
        )
    _reject_unknown_keys(
        config, path,
        {"mode", "lora_targets", "train_layer_norm", "stem", "ffn_adapter"},
    )
    _require_choice(config, f"{path}.mode", (UTONIA_ADAPTATION_MODE_HYBRID,))

    targets = OmegaConf.select(config, f"{path}.lora_targets")
    if not isinstance(targets, (list, tuple, ListConfig)):
        raise ValueError(f"{path}.lora_targets must be a list")
    names = [str(target).lower() for target in targets]
    for name in names:
        if name in ("ffn", "mlp"):
            raise ValueError(
                f"{path}.lora_targets cannot include {name!r}: FFN LoRA and the "
                "parallel FFN adapter are mutually exclusive, and the FFN "
                "Linears stay plain frozen layers"
            )
        if name == "stem":
            raise ValueError(
                f"{path}.lora_targets cannot include 'stem': the hybrid stem "
                "adapts through its intensity column and nonlinear adapter"
            )
        if name not in UTONIA_LORA_TARGETS:
            raise ValueError(
                f"{path}.lora_targets entries must be one of {UTONIA_LORA_TARGETS}"
            )
    if len(set(names)) != len(names):
        raise ValueError(f"{path}.lora_targets must not repeat an entry")
    _require_flag(config, f"{path}.train_layer_norm")

    stem_path = f"{path}.stem"
    _reject_unknown_keys(
        config, stem_path,
        {
            "train_intensity_column", "adapter_hidden", "adapter_activation",
            "adapter_scale", "adapter_position",
        },
    )
    _require_flag(config, f"{stem_path}.train_intensity_column")
    _require_positive(config, f"{stem_path}.adapter_hidden", kind=int)
    _require_choice(
        config, f"{stem_path}.adapter_activation", UTONIA_ADAPTER_ACTIVATIONS
    )
    _require_finite(
        config, f"{stem_path}.adapter_scale", minimum=0.0, inclusive=False
    )
    # An absent adapter_position is V14/V14.1's adapter before the stem
    # LayerNorm; V15 moves it behind the activation and must say so.
    position_path = f"{stem_path}.adapter_position"
    if _has_path(config, position_path):
        _require_choice(config, position_path, UTONIA_STEM_ADAPTER_POSITIONS)
    position = str(OmegaConf.select(
        config, position_path, default="pre_norm"
    )).lower()
    if variant in POST_ACTIVATION_STEM_DYNAMIC_VARIANTS:
        if position != "post_activation":
            raise ValueError(
                f"{label} requires {position_path}='post_activation' (the stem "
                "adapter reads GELU(LN(W x)) and adds a residual to it)"
            )
    elif position != "pre_norm":
        raise ValueError(
            f"{label} keeps the stem adapter before the stem LayerNorm; "
            f"{position_path}={position!r} is only available to "
            + _only_available_to(POST_ACTIVATION_STEM_DYNAMIC_VARIANTS)
        )

    ffn_path = f"{path}.ffn_adapter"
    _reject_unknown_keys(
        config, ffn_path, {"widths", "activation", "scale", "dropout"}
    )
    widths = OmegaConf.select(config, f"{ffn_path}.widths")
    stages = int(OmegaConf.select(config, "p2g.utonia_feature_stage")) + 1
    if not isinstance(widths, (list, tuple, ListConfig)) or len(widths) != stages:
        raise ValueError(
            f"{ffn_path}.widths needs one bottleneck width per executed "
            f"Utonia stage ({stages} for utonia_feature_stage={stages - 1})"
        )
    if any(int(width) <= 0 for width in widths):
        raise ValueError(f"{ffn_path}.widths must be positive")
    _require_choice(
        config, f"{ffn_path}.activation", UTONIA_ADAPTER_ACTIVATIONS
    )
    _require_finite(config, f"{ffn_path}.scale", minimum=0.0, inclusive=False)
    dropout = OmegaConf.select(config, f"{ffn_path}.dropout")
    if dropout is None or not 0.0 <= float(dropout) < 1.0:
        raise ValueError(f"{ffn_path}.dropout must be in [0, 1)")

    if groups is None:
        raise ValueError(
            f"{label} requires {groups_path} with a learning rate and weight "
            "decay for each Utonia role"
        )
    _reject_unknown_keys(config, groups_path, set(UTONIA_PARAM_GROUP_ROLES))
    for role in UTONIA_PARAM_GROUP_ROLES:
        role_path = f"{groups_path}.{role}"
        if OmegaConf.select(config, role_path, default=None) is None:
            raise ValueError(f"{role_path} is required")
        decay_keys = ["weight_decay"]
        if role == "adapter":
            decay_keys.append("bias_weight_decay")
        _reject_unknown_keys(config, role_path, {"lr", *decay_keys})
        _require_finite(config, f"{role_path}.lr", minimum=0.0, inclusive=False)
        for key in decay_keys:
            _require_finite(
                config, f"{role_path}.{key}", minimum=0.0, inclusive=True
            )


def _validate_adam_settings(config) -> None:
    """Optional explicit AdamW betas/eps; absent keys keep torch's defaults."""
    betas = OmegaConf.select(config, "train.adam_betas", default=None)
    if betas is not None:
        if not isinstance(betas, (list, tuple, ListConfig)) or len(betas) != 2:
            raise ValueError("train.adam_betas must be a two-entry list")
        if any(not 0.0 <= float(beta) < 1.0 for beta in betas):
            raise ValueError("train.adam_betas entries must be in [0, 1)")
    if OmegaConf.select(config, "train.adam_eps", default=None) is not None:
        _require_finite(config, "train.adam_eps", minimum=0.0, inclusive=False)


def _validate_raydrop_refine(config) -> None:
    """Optional ``raydrop_refine`` block of a frozen-model refinement run.

    Absent or ``enable: false`` is an ordinary model run. Enabled, the block
    names the frozen checkpoint (recorded with its hash, epoch and step when
    ``refine_config`` composes the run) and the U-Net's input settings.
    """
    if not _has_path(config, "raydrop_refine"):
        return
    root = "raydrop_refine"
    _reject_unknown_keys(config, root, {
        "enable", "base_ckpt", "base_ckpt_sha256", "base_epoch",
        "base_global_step", "base_exp_name", "channels", "depth_scale_m",
        "box_mask",
    })
    _require_flag(config, f"{root}.enable")
    if not bool(OmegaConf.select(config, f"{root}.enable")):
        return
    for key in ("base_ckpt", "base_ckpt_sha256"):
        if not OmegaConf.select(config, f"{root}.{key}", default=None):
            raise ValueError(f"{root}.{key} is required when {root}.enable=true")
    _require_positive(config, f"{root}.channels", kind=int)
    _require_finite(config, f"{root}.depth_scale_m", minimum=0.0, inclusive=False)
    _reject_unknown_keys(config, f"{root}.box_mask",
                         {"enable", "max_boxes", "max_size_frac"})
    _require_flag(config, f"{root}.box_mask.enable")
    _require_positive(config, f"{root}.box_mask.max_boxes", kind=int)
    _require_in_open_range(config, f"{root}.box_mask.max_size_frac", 0.0, 1.0)


def _validate_spconv_precision(config) -> None:
    """Optional ``train.spconv_allow_tf32``: a YAML boolean when present.

    Absent means false -- FP32 sparse convolutions, which is what every
    checkpoint saved before the key existed trained with.
    """
    path = "train.spconv_allow_tf32"
    if _has_path(config, path):
        _require_flag(config, path)


def _validate_downstream_no_decay(config, variant, label: str) -> None:
    """Optional ``train.downstream_no_decay``: downstream parameters without decay.

    Names are relative to Point2Gaus (``dynamic_backend.temporal.v_raw``) and
    move from the ``downstream`` optimizer group into a last group at
    ``train.lr`` with weight decay 0. Only the hybrid Utonia variants build
    per-role optimizer groups; every other variant trains one flat parameter
    list, so the key would have nothing to act on. Utonia parameters are
    excluded because their roles already set their own decay.
    """
    path = "train.downstream_no_decay"
    names = OmegaConf.select(config, path, default=None)
    if names is None:
        return
    if variant not in HYBRID_UTONIA_DYNAMIC_VARIANTS:
        raise ValueError(
            f"{label} trains one flat parameter group; {path} is only "
            "available to " + _only_available_to(HYBRID_UTONIA_DYNAMIC_VARIANTS)
        )
    if not isinstance(names, (list, tuple, ListConfig)):
        raise ValueError(f"{path} must be a list of parameter names")
    entries = list(names)
    if any(not isinstance(name, str) or not name.strip() for name in entries):
        raise ValueError(f"{path} entries must be non-empty strings")
    if len(set(entries)) != len(entries):
        raise ValueError(f"{path} must not repeat an entry")
    for name in entries:
        if name.startswith("feature_extractor."):
            raise ValueError(
                f"{path} lists downstream parameters only; {name!r} is a Utonia "
                "parameter, whose weight decay comes from train.utonia_param_groups"
            )


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


def _validate_retired_loss_options(config) -> None:
    """Loss switches that were removed, accepted only at their inert value."""
    w_scale = OmegaConf.select(config, "loss.w_scale", default=None)
    if w_scale is not None and float(w_scale) != 0.0:
        raise ValueError(
            "loss.w_scale was removed: the scale term is only logged as "
            "loss_scale and never enters the total; drop the key or set it to 0"
        )
    if bool(OmegaConf.select(config, "loss.enable_lpips", default=False)):
        raise ValueError(
            "loss.enable_lpips was removed; LPIPS is reported by "
            "test_lidar4d_nvs.py"
        )


def _configured_count_mode(config) -> str:
    """The Gaussian-count contract ``p2g.grid_query`` asks for.

    An absent block is the historical fixed-count default: one Gaussian per
    occupied token, seeded at that token's observed medoid. Otherwise this is
    the ``count_mode`` string as written, lower-cased: ``legacy``,
    ``learned_gumbel``, V15's ``nested_gumbel`` or V15.2's ``expert_gumbel``.
    """
    if not _has_path(config, "p2g.grid_query"):
        return LEGACY_COUNT_MODE
    mode = OmegaConf.select(config, "p2g.grid_query.count_mode")
    return LEGACY_COUNT_MODE if mode is None else str(mode).lower()


def _validate_fixed_gaussian_count(config, label: str) -> None:
    """The per-token Gaussian count and seed placement, when both are fixed.

    ``K_max`` is how many Gaussians every occupied token emits, and
    ``seed_mode`` says where they start: ``range_quantile`` (the default when
    the key is absent) spreads them over the token's own raw points by range
    quantile, ``medoid`` puts every slot on the token's observed medoid, and
    ``token`` puts every slot on the token's Utonia position. Slots that
    coincide are separated by the independent parameter block each one owns in
    the Gaussian head.
    """
    count_path = "p2g.grid_query"
    if not _has_path(config, count_path):
        return
    if _has_path(config, f"{count_path}.exp"):
        raise ValueError(
            f"{label} p2g.grid_query.exp was renamed to "
            "p2g.grid_query.seed_mode (null -> range_quantile, 1 -> medoid, "
            "2 -> token)"
        )
    _reject_unknown_keys(
        config, count_path, {"count_mode", "K_max", "seed_mode"}
    )
    k_max = OmegaConf.select(config, f"{count_path}.K_max")
    if k_max is None or int(k_max) < 1:
        raise ValueError(
            f"{label} requires a positive p2g.grid_query.K_max Gaussians "
            "per token"
        )
    if _has_path(config, f"{count_path}.seed_mode"):
        _require_choice(
            config, f"{count_path}.seed_mode", FIXED_COUNT_SEED_MODES
        )


class _RoutedCount(NamedTuple):
    """One count mode whose router picks 1, 3 or 5 Gaussians after refinement."""

    variants: tuple
    option_counts: tuple
    renders: tuple
    render_by_variant: dict
    head: str
    purpose: str
    options: str
    # Required block keys beyond the ones every routed count shares.
    extra_keys: tuple = ()


_ROUTED_COUNTS = {
    NESTED_COUNT_MODE: _RoutedCount(
        variants=NESTED_COUNT_DYNAMIC_VARIANTS,
        option_counts=NESTED_COUNT_OPTION_COUNTS,
        renders=NESTED_AUXILIARY_RENDERS,
        render_by_variant=NESTED_AUXILIARY_RENDER_BY_VARIANT,
        head="nested count head",
        purpose="its head decodes G1..G5 and a router picks how many to render",
        options="reduce {G1}, keep {G1..G3}, add {G1..G5}",
    ),
    EXPERT_COUNT_MODE: _RoutedCount(
        variants=EXPERT_COUNT_DYNAMIC_VARIANTS,
        option_counts=EXPERT_COUNT_OPTION_COUNTS,
        renders=EXPERT_AUXILIARY_RENDERS,
        render_by_variant=EXPERT_AUXILIARY_RENDER_BY_VARIANT,
        head="per-option expert head",
        purpose="a router picks one of its E1 / E3 / E5 experts per token",
        options="reduce E1, keep E3, add E5",
    ),
    ORDINAL_COUNT_MODE: _RoutedCount(
        variants=ORDINAL_COUNT_DYNAMIC_VARIANTS,
        option_counts=ORDINAL_COUNT_OPTION_COUNTS,
        renders=ORDINAL_AUXILIARY_RENDERS,
        render_by_variant=ORDINAL_AUXILIARY_RENDER_BY_VARIANT,
        head="ordinal nested count head",
        purpose=(
            "its head decodes G1..G5 and two bounded gates pick how many to "
            "render, trained through masked rasterization"
        ),
        options="reduce {G1}, keep {G1..G3}, add {G1..G5}",
        extra_keys=("router_scale",),
    ),
}


def _validate_dynamic_gaussian_count(config, variant, label: str) -> None:
    """Route the config to the count contract it selected, and check it fits."""
    count_mode = _configured_count_mode(config)
    own_mode = next(
        (
            mode for mode, routed in _ROUTED_COUNTS.items()
            if variant in routed.variants
        ),
        None,
    )
    if count_mode in _ROUTED_COUNTS and count_mode != own_mode:
        routed = _ROUTED_COUNTS[count_mode]
        raise ValueError(
            f"{label} has no {routed.head}; p2g.grid_query.count_mode="
            f"{count_mode!r} is only available to "
            + _only_available_to(routed.variants)
        )
    if own_mode is not None:
        routed = _ROUTED_COUNTS[own_mode]
        if count_mode != own_mode:
            raise ValueError(
                f"{label} requires p2g.grid_query.count_mode={own_mode!r} "
                f"({routed.purpose}), got {count_mode!r}"
            )
        _validate_routed_gaussian_count(config, label, own_mode)
        # Whether training also renders a router-free set is what separates
        # V15 from V15.1, so each variant pins its own value.
        block = ROUTED_COUNT_BLOCKS[own_mode]
        expected_render = routed.render_by_variant[variant]
        render = str(OmegaConf.select(
            config, f"p2g.grid_query.{block}.auxiliary_render"
        )).lower()
        if render != expected_render:
            raise ValueError(
                f"{label} requires p2g.grid_query.{block}."
                f"auxiliary_render={expected_render!r}, got {render!r}"
            )
        return
    if variant in DENSE_COUNT_DYNAMIC_VARIANTS:
        _validate_dense_gaussian_count(config, label, count_mode)
        return
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


def _validate_routed_gaussian_count(config, label: str, count_mode: str) -> None:
    """A V15-family routed count: one token anchor and a {1, 3, 5} router.

    V15/V15.1 (``nested_gumbel``) render a prefix of G1..G5, V15.2
    (``expert_gumbel``) one of three experts and V16 (``nested_ordinal``) a
    prefix of G1..G5 chosen by two ordinal gates, each mode configured by its
    own block (``nested_count`` / ``expert_count`` / ``ordinal_count``) with the
    same keys; the ordinal block also needs its gates' fixed cosine scale,
    ``router_scale``. ``p2g.grid_query`` holds exactly ``count_mode`` and that
    block, and every block key but ``logging`` and ``auxiliary_views`` is
    required: the decoder and router widths, the router temperature and the
    auxiliary render are all experiment choices, so none of them falls back to
    a code default. ``auxiliary_views`` (absent or null = every target view)
    renders the auxiliary set at that many target views per window, drawn
    uniformly at random each step; a block without an auxiliary render rejects
    it.
    """
    routed = _ROUTED_COUNTS[count_mode]
    block = ROUTED_COUNT_BLOCKS[count_mode]
    count_path = "p2g.grid_query"
    _reject_unknown_keys(config, count_path, {"count_mode", block})
    block_path = f"{count_path}.{block}"
    if OmegaConf.select(config, block_path, default=None) is None:
        raise ValueError(f"{label} requires {block_path}")
    required = (
        "option_counts", "seed_mode", "tau", "decoder_hidden", "router_hidden",
        "auxiliary_render", *routed.extra_keys,
    )
    _reject_unknown_keys(
        config, block_path, {*required, "logging", "auxiliary_views"}
    )
    missing = [
        name for name in required if not _has_path(config, f"{block_path}.{name}")
    ]
    if missing:
        raise ValueError(f"{block_path} is missing keys: " + ", ".join(missing))

    option_counts = OmegaConf.select(config, f"{block_path}.option_counts")
    if not isinstance(option_counts, (list, tuple, ListConfig)) or (
        list(option_counts) != list(routed.option_counts)
    ):
        raise ValueError(
            f"{block_path}.option_counts must be "
            f"{list(routed.option_counts)}: {routed.options}"
        )
    # One shared anchor: every Gaussian starts at its token's Utonia position
    # and the head's bounded offsets place it.
    _require_choice(config, f"{block_path}.seed_mode", (SEED_MODE_TOKEN,))
    _require_finite(config, f"{block_path}.tau", minimum=0.0, inclusive=False)
    if "router_scale" in routed.extra_keys:
        _require_finite(
            config, f"{block_path}.router_scale", minimum=0.0, inclusive=False
        )
    _require_int(config, f"{block_path}.decoder_hidden")
    router_hidden = OmegaConf.select(config, f"{block_path}.router_hidden")
    if (
        not isinstance(router_hidden, (list, tuple, ListConfig))
        or not router_hidden
        or any(
            isinstance(width, bool) or not isinstance(width, int) or width <= 0
            for width in router_hidden
        )
    ):
        raise ValueError(
            f"{block_path}.router_hidden must be a non-empty list of positive "
            "integer hidden widths (V15: [256, 128])"
        )
    _require_choice(config, f"{block_path}.auxiliary_render", routed.renders)
    views_path = f"{block_path}.auxiliary_views"
    if OmegaConf.select(config, views_path, default=None) is not None:
        render = str(OmegaConf.select(config, f"{block_path}.auxiliary_render"))
        if render.lower() == "none":
            raise ValueError(
                f"{label} renders no auxiliary set ({block_path}."
                f"auxiliary_render='none'), so {views_path} cannot be set"
            )
        _require_int(config, views_path)
    if _has_path(config, f"{block_path}.logging"):
        logging_path = f"{block_path}.logging"
        _reject_unknown_keys(
            config, logging_path, {"enable", "interval", "range_edges_m"}
        )
        _require_positive(config, f"{logging_path}.interval", kind=int)


def _validate_dense_gaussian_count(config, label: str, count_mode: str) -> None:
    """V15.3's fixed count: five Gaussians per token, all on the token anchor.

    V15.3 has no router, so its count is the ordinary fixed-count contract,
    pinned: ``count_mode: legacy``, ``K_max`` = five and ``seed_mode: token``
    (its decoder places every Gaussian around the one token anchor, so per-slot
    seeds would be ignored).
    """
    count_path = "p2g.grid_query"
    expected = (
        f"count_mode='legacy', K_max={DENSE_COUNT_GAUSSIANS_PER_TOKEN}, "
        f"seed_mode={SEED_MODE_TOKEN!r}"
    )
    if count_mode != LEGACY_COUNT_MODE or not _has_path(config, count_path):
        raise ValueError(
            f"{label} has no count router and requires p2g.grid_query with "
            f"{expected}, got count_mode={count_mode!r}"
        )
    _validate_fixed_gaussian_count(config, label)
    k_max = OmegaConf.select(config, f"{count_path}.K_max")
    if (
        isinstance(k_max, bool) or not isinstance(k_max, int)
        or k_max != DENSE_COUNT_GAUSSIANS_PER_TOKEN
    ):
        raise ValueError(
            f"{label} renders {DENSE_COUNT_GAUSSIANS_PER_TOKEN} Gaussians per "
            f"token; p2g.grid_query.K_max must be "
            f"{DENSE_COUNT_GAUSSIANS_PER_TOKEN}, got {k_max!r}"
        )
    seed_mode = OmegaConf.select(config, f"{count_path}.seed_mode", default=None)
    if seed_mode is None or str(seed_mode).lower() != SEED_MODE_TOKEN:
        raise ValueError(
            f"{label} decodes every Gaussian around the token anchor; "
            f"p2g.grid_query.seed_mode must be {SEED_MODE_TOKEN!r}, got "
            f"{seed_mode!r}"
        )


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
    _require_choice(
        config, f"{learned_path}.seed_mode", LEARNED_COUNT_SEED_MODES
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


def _validate_kitti360_training(config, variant: str) -> None:
    if str(OmegaConf.select(config, "data.dataset_name")) != "kitti360_dynamic":
        return
    if variant not in BARRIER_MATCH_DYNAMIC_VARIANTS:
        raise ValueError("kitti360_dynamic requires a V11 or later physical-time variant")
    if str(OmegaConf.select(config, "data.mode")) != "nvs":
        raise ValueError("kitti360_dynamic requires data.mode=nvs")
    if bool(OmegaConf.select(config, "data.compact_val.enable", default=False)):
        # The dataset checks the manifest's sampling contract when val loads.
        if not str(OmegaConf.select(config, "data.compact_val.path", default="") or "").strip():
            raise ValueError("kitti360_dynamic data.compact_val.enable=true requires data.compact_val.path")
    _require_choice(config, "data.pose_missing_policy", ("interpolate",))
    _require_choice(config, "data.panorama_projection", ("uniform", "hdl64e_kitti360_calibrated"))
    _require_positive(config, "data.window_stride_sec", "data.max_sweep_gap_sec", "data.max_depth_m")
    for path in ("data.window_stride_sec", "data.max_sweep_gap_sec", "data.max_depth_m"):
        if not math.isfinite(float(OmegaConf.select(config, path))):
            raise ValueError(f"{path} must be finite")
    if int(OmegaConf.select(config, "data.sweep_hop", default=0)) != 1:
        raise ValueError("kitti360_dynamic requires data.sweep_hop=1")
    middle = OmegaConf.select(config, "data.gt_middle_count")
    if isinstance(middle, bool) or not isinstance(middle, int) or middle < 0:
        raise ValueError("kitti360_dynamic gt_middle_count must be a nonnegative integer")
    minimum = float(OmegaConf.select(config, "data.dynamic_window.endpoint_min_sec"))
    maximum = float(OmegaConf.select(config, "data.dynamic_window.endpoint_max_sec"))
    if not all(map(math.isfinite, (minimum, maximum))) or not 0 < minimum <= maximum:
        raise ValueError("KITTI360 endpoint durations must satisfy 0 < min <= max")
    if OmegaConf.select(config, "data.ring_to_elevation_deg") is not None:
        raise ValueError("KITTI360 raw scans have no ring field; data.ring_to_elevation_deg must be null")
    if str(OmegaConf.select(config, "data.panorama_projection")) == "hdl64e_kitti360_calibrated":
        if int(OmegaConf.select(config, "data.image_height")) != 64:
            raise ValueError("KITTI360 calibrated projection requires image_height=64")
        _require_choice(config, "data.column_assignment", ("nearest", "firing_slot"))
    if not OmegaConf.select(config, "data.split_manifest_path") or not OmegaConf.select(config, "data.split_manifest_sha256"):
        raise ValueError("kitti360_dynamic requires a checkpoint-bound split manifest path and hash")


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


def _validate_temporal_layer_scale(config, variant: str, label: str) -> None:
    """The temporal LayerScale gate: kept by every variant but V15.

    An absent ``layer_scale`` is the historical gate, initialized by
    ``layer_scale_init``. V15 drops the gate, so it must say
    ``layer_scale: false`` and has no init to configure.
    """
    temporal_path = "dynamic_2dgs.temporal"
    flag_path = f"{temporal_path}.layer_scale"
    declared = _has_path(config, flag_path)
    if declared:
        _require_flag(config, flag_path)
    gated = bool(OmegaConf.select(config, flag_path)) if declared else True
    if variant in NO_TEMPORAL_LAYER_SCALE_DYNAMIC_VARIANTS:
        if gated:
            raise ValueError(
                f"{label} requires {flag_path}=false: its temporal "
                "cross-attention adds the out_proj branch without a LayerScale"
            )
        if _has_path(config, f"{temporal_path}.layer_scale_init"):
            raise ValueError(
                f"{label} has no temporal LayerScale, so "
                f"{temporal_path}.layer_scale_init cannot be set"
            )
    elif not gated:
        raise ValueError(
            f"{label} keeps the temporal LayerScale; {flag_path}=false is only "
            "available to " + _only_available_to(
                NO_TEMPORAL_LAYER_SCALE_DYNAMIC_VARIANTS
            )
        )


def _validate_temporal_config(config, variant: str, label: str) -> None:
    """The temporal-attention keys and contract of the variant's family."""
    temporal_path = "dynamic_2dgs.temporal"
    temporal_keys = {
        "implementation", "layers", "num_heads", "mlp_ratio",
        "use_time_embedding", "time_frequencies", "time_embedding_dim",
        "time_reference_sec", "position_encoding", "rope_base",
        "rope_position_scale", "qk_norm", "layer_scale_init", "layer_scale",
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
    _validate_temporal_layer_scale(config, variant, label)

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
    num_heads = int(OmegaConf.select(config, f"{temporal_path}.num_heads"))
    if num_heads < 2:
        raise ValueError(
            f"{label} requires dynamic_2dgs.temporal.num_heads >= 2"
        )
    # V15.3 and V15.4 differ only in the temporal head count.
    expected_heads = DENSE_COUNT_TEMPORAL_HEADS_BY_VARIANT.get(variant)
    if expected_heads is not None and num_heads != expected_heads:
        raise ValueError(
            f"{label} requires dynamic_2dgs.temporal.num_heads="
            f"{expected_heads}, got {num_heads}"
        )


def _validate_local_attention_config(config, variant: str, label: str) -> None:
    """V15's radius local-attention block; every other variant has none."""
    path = "dynamic_2dgs.local_attention"
    block = OmegaConf.select(config, path, default=None)
    if variant not in LOCAL_ATTENTION_DYNAMIC_VARIANTS:
        if block is not None:
            raise ValueError(
                f"{label} has no radius local-attention block; {path} is only "
                "available to " + _only_available_to(
                    LOCAL_ATTENTION_DYNAMIC_VARIANTS
                )
            )
        return
    if block is None:
        raise ValueError(f"{label} requires {path}")
    keys = (
        "radius_m", "num_heads", "mlp_ratio", "rope_base",
        "rope_position_scale", "implementation", "query_chunk_size",
    )
    _reject_unknown_keys(config, path, keys)
    missing = [name for name in keys if not _has_path(config, f"{path}.{name}")]
    if missing:
        raise ValueError(f"{path} is missing keys: " + ", ".join(missing))

    for name in ("radius_m", "mlp_ratio", "rope_base", "rope_position_scale"):
        _require_finite(config, f"{path}.{name}", minimum=0.0, inclusive=False)
    _validate_3d_rope_width(
        _configured_p2g_trunk_dim(config),
        _require_int(config, f"{path}.num_heads"),
        "dynamic local attention",
    )
    _require_choice(
        config, f"{path}.implementation", LOCAL_ATTENTION_IMPLEMENTATIONS
    )
    # Every query is one FlashAttention2 varlen sequence, so a chunk larger than
    # the kernel's sequence-count limit fails at launch rather than at config.
    chunk_size = _require_int(config, f"{path}.query_chunk_size")
    if chunk_size > LOCAL_ATTENTION_MAX_QUERY_CHUNK_SIZE:
        raise ValueError(
            f"{path}.query_chunk_size must be at most "
            f"{LOCAL_ATTENTION_MAX_QUERY_CHUNK_SIZE}, the FlashAttention2 "
            f"varlen sequence-count limit; got {chunk_size}"
        )


def _validate_gaussian_head_config(config, variant: str, label: str) -> None:
    """``dynamic_2dgs.gaussian_head``: the priors, and V15.3's decoder width.

    Every variant may set the initial opacity and scale. ``decoder_hidden`` is
    the hidden width of V15.3's router-free decoder and is required there;
    every other variant sizes its decoder elsewhere (the routed V15-family
    heads under their ``p2g.grid_query`` count block) and rejects it.
    """
    head_path = "dynamic_2dgs.gaussian_head"
    _reject_unknown_keys(
        config, head_path, {"initial_opacity", "initial_scale_m", "decoder_hidden"}
    )
    hidden_path = f"{head_path}.decoder_hidden"
    if variant in DENSE_COUNT_DYNAMIC_VARIANTS:
        if not _has_path(config, hidden_path):
            raise ValueError(f"{label} requires {hidden_path}")
        _require_int(config, hidden_path)
    elif _has_path(config, hidden_path):
        raise ValueError(
            f"{label} sizes its Gaussian decoder elsewhere; {hidden_path} is "
            "only available to " + _only_available_to(DENSE_COUNT_DYNAMIC_VARIANTS)
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

    coordinate_frame = str(OmegaConf.select(
        config, "data.coordinate_frame", default="lidar_sensor"
    )).lower()
    if coordinate_frame not in COORDINATE_FRAMES:
        raise ValueError(
            "data.coordinate_frame must be one of "
            f"{COORDINATE_FRAMES}, got {coordinate_frame!r}"
        )

    _validate_dynamic_window_loader(config, variant)
    _validate_kitti360_training(config, variant)
    _validate_legacy_train_exclusions(config)
    _validate_retired_loss_options(config)
    _validate_utonia_lora_config(config, label)
    _validate_utonia_adaptation_config(config, variant, label)
    _validate_adam_settings(config)
    _validate_spconv_precision(config)
    _validate_downstream_no_decay(config, variant, label)
    _validate_raydrop_refine(config)

    # Checkpoints embed the anchor mode from when a second one existed.
    anchor_mode = OmegaConf.select(config, "p2g.anchor_mode", default=None)
    if anchor_mode is not None and str(anchor_mode) != "grid":
        raise ValueError(f"{variant} requires p2g.anchor_mode=grid")
    # They also embed g2p.dynamic=false from the removed PVG-style
    # temporal-opacity (marginal_t) prefilter.
    if bool(OmegaConf.select(config, "g2p.dynamic", default=False)):
        raise ValueError(
            "g2p.dynamic=true is no longer supported: the marginal_t temporal "
            "prefilter was removed; Gaussians move by velocity transport only"
        )
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
        {
            "temporal", "gaussian_head", "motion", "regularization",
            "local_attention",
        },
    )
    _validate_temporal_config(config, variant, label)
    _validate_local_attention_config(config, variant, label)
    _validate_gaussian_head_config(config, variant, label)
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
