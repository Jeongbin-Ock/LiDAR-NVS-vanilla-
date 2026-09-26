"""Experiment variant constants, groups, and schema key sets."""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "nuscene_train.yaml"

DYNAMIC_VARIANT_V11 = "dynamic_2dgs_attention_velocity_v11"
DYNAMIC_VARIANT_V11_1 = "dynamic_2dgs_attention_velocity_v11_1"
DYNAMIC_VARIANT_V11_2 = "dynamic_2dgs_attention_velocity_v11_2"
DYNAMIC_VARIANT_V11_3 = "dynamic_2dgs_attention_velocity_v11_3"
DYNAMIC_VARIANT_V11_4 = "dynamic_2dgs_attention_velocity_v11_4"
DYNAMIC_VARIANT_V11_5 = "dynamic_2dgs_attention_velocity_v11_5"
DYNAMIC_VARIANT_V11_6 = "dynamic_2dgs_attention_velocity_v11_6"
DYNAMIC_VARIANT_V12 = "dynamic_2dgs_attention_velocity_v12"
DYNAMIC_VARIANT_V12_1 = "dynamic_2dgs_attention_velocity_v12_1"
DYNAMIC_VARIANT_V12_2 = "dynamic_2dgs_attention_velocity_v12_2"
DYNAMIC_VARIANT_V13 = "dynamic_2dgs_attention_velocity_v13"
DYNAMIC_VARIANT_V13_1 = "dynamic_2dgs_attention_velocity_v13_1"
DYNAMIC_VARIANT_V13_2 = "dynamic_2dgs_attention_velocity_v13_2"
DYNAMIC_VARIANT_V13_3 = "dynamic_2dgs_attention_velocity_v13_3"
DYNAMIC_VARIANT_V13_4 = "dynamic_2dgs_attention_velocity_v13_4"
# The unsuffixed constant always names the current fresh-run baseline. Keep an
# explicit constant for each variant so checkpoint semantics stay exact.
DYNAMIC_VARIANT = DYNAMIC_VARIANT_V11_4
# Every variant reads correspondence from a fixed match head rather than a
# configured one: head 0 at every temporal layer, except V11.2, which gives every
# layer head RoPE and reads one dedicated head after refinement.
#
# Variants whose backend owns the learned_gumbel K={1,2,3} grid Gaussian head,
# and can therefore let a router decide how many Gaussians each token becomes.
# Every other variant emits a fixed count per token. Which of the two a run
# uses is still a config decision (p2g.grid_query.count_mode) for the variants
# listed here. V11.3 shares V11's backend class but is deliberately absent: its
# identity is the fixed two Gaussians per token, so asking it for the router is
# rejected rather than quietly turning it back into V11. V11.5 and V11.6 are
# absent for the same reason relative to V11.4.
ROUTER_CAPABLE_DYNAMIC_VARIANTS = (
    DYNAMIC_VARIANT_V11,
    DYNAMIC_VARIANT_V11_2,
    DYNAMIC_VARIANT_V11_4,
    DYNAMIC_VARIANT_V12,
    DYNAMIC_VARIANT_V12_1,
    DYNAMIC_VARIANT_V12_2,
    DYNAMIC_VARIANT_V13,
    DYNAMIC_VARIANT_V13_2,
    DYNAMIC_VARIANT_V13_3,
    DYNAMIC_VARIANT_V13_4,
)
# V13-family correspondence reads head 0 at every temporal layer, but unlike
# the barrier family all heads carry 3D RoPE and share one FlashAttention call.
# In V13/V13.1 only, head 0 is L2-normalized and temperature-scaled.
COSINE_ROPE_MATCH_DYNAMIC_VARIANTS = (
    DYNAMIC_VARIANT_V13,
    DYNAMIC_VARIANT_V13_1,
)
# V13.3/V13.4 restore V11.4's per-layer Q/K RMSNorm. V13.3 applies 3D RoPE to
# every head; V13.4 leaves head 0 unrotated while heads 1..H-1 keep RoPE.
RMSNORM_ROPE_MATCH_DYNAMIC_VARIANTS = (
    DYNAMIC_VARIANT_V13_3,
    DYNAMIC_VARIANT_V13_4,
)
# V13-family variants in this group use all-head 3D RoPE and one Flash call.
# V13/V13.1 specialize head 0 with cosine temperature; V13.2 leaves every
# head on the ordinary raw QK / sqrt(d) score; V13.3 RMS-normalizes every
# head before the same score.
ALL_HEAD_ROPE_MATCH_DYNAMIC_VARIANTS = (
    *COSINE_ROPE_MATCH_DYNAMIC_VARIANTS,
    DYNAMIC_VARIANT_V13_2,
    DYNAMIC_VARIANT_V13_3,
)
# V13.4 still runs one all-head FlashAttention call, but applies 3D RoPE only
# to feature heads 1..H-1. Match head 0 uses RMS-normalized content alone.
FEATURE_HEAD_ROPE_MATCH_DYNAMIC_VARIANTS = (
    DYNAMIC_VARIANT_V13_4,
)
# Shared single-Flash packed coordinate-readout family. Position encoding and
# Q/K normalization remain variant-specific contracts inside this group.
FLASH_ROPE_MATCH_DYNAMIC_VARIANTS = (
    *ALL_HEAD_ROPE_MATCH_DYNAMIC_VARIANTS,
    *FEATURE_HEAD_ROPE_MATCH_DYNAMIC_VARIANTS,
)
# Variants whose correspondence runs under the max-speed barrier. In V11 and
# V11.1 head 0 carries it at every layer while every other head keeps 3D RoPE,
# and V11.1 additionally splits the QK-Norm gain along that head boundary.
# V11.3 is V11's correspondence exactly, differing only in Gaussian count.
# V11.4 is V11 with the hinge made linear: barrier_exponent=1 at weight 1.
# V11.5 is V11.3's fixed K=2 decoder under V11.4's linear hinge.
# V11.6 is V11.5 with K=3, trained on the random-duration loader.
# V12 keeps V11.4's linear hinge and shared QK-Norm gains, but learns its one
# global max-speed radius in log space from an initial 30 m/s.
# V12.1 keeps V12's learned radius, fixes head 0's RMS gain to one, and lets
# heads 1..H-1 share a separate learned gain.
# V12.2 is V12 with one learned radius per layer, each softplus(v_raw_l) from an
# initial 30 m/s, trained on the random-duration loader.
# V11.2 gives head 0 RoPE too and moves the barrier to a single post-refinement
# readout head with its own gain, which is what lets its layer stack run
# entirely on FlashAttention.
BARRIER_MATCH_DYNAMIC_VARIANTS = (
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
)
DYNAMIC_VARIANTS = (
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
)

# The overlay allow-list: one config/variants/<variant>.yaml per variant.
VARIANT_CONFIG_PATHS = {
    variant: REPO_ROOT / "config" / "variants" / f"{variant}.yaml"
    for variant in DYNAMIC_VARIANTS
}

# W&B omits empty mappings, so a restored config may have no g2g block. The
# coordinate fallback preserves pre-key checkpoints' LiDAR-frame data contract.
STRUCTURAL_DEFAULTS = {
    "g2g": {},
    # Checkpoints saved before the coordinate-frame key used LiDAR sensor
    # coordinates for both Utonia and the frame-0 temporal reference.
    "data": {"coordinate_frame": "lidar_sensor"},
}

COORDINATE_FRAMES = ("ego_vehicle", "lidar_sensor")

_REQUIRED_ROOTS = (
    "model",
    "p2g",
    "g2g",
    "g2p",
    "data",
    "loss",
    "train",
    "test",
    "logger",
)

# These roots control model construction or shape-neutral rendering semantics.
# Runtime knobs can change during resume/evaluation, but saved weights must not
# be reinterpreted by modifying one of these subtrees from the CLI.
_CHECKPOINT_STRUCTURAL_ROOTS = (
    "model",
    "p2g",
    "g2g",
    "g2p",
    "dynamic_2dgs",
)

_CHECKPOINT_PROTECTED_PATHS = (
    "data.dataset_name",
    "data.coordinate_frame",
    "data.window_us",
    "data.sample_gap_us",
    "data.gt_middle_count",
    "data.pair_mode",
    "data.pair_kf_stride",
    "data.mode",
    "data.bbox_json_path",
    "data.bbox_json_paths",
    "data.vfov",
    "data.hfov",
    "data.image_height",
    "data.image_width",
    "data.ring_to_elevation_deg",
    "data.ego_radius",
    "data.train_excluded_scenes",
    "data.dynamic_window",
)

# Pointcept/Utonia encoder widths from config/utonia_pretrained.yaml. The token
# width that reaches temporal attention is the width of the configured encoder
# exit stage, so static config validation reads it from here.
_UTONIA_ENCODER_STAGE_DIMS = (54, 108, 216, 432, 576)


LEGACY_COUNT_MODE = "legacy"
LEARNED_COUNT_MODES = ("learned_gumbel",)
