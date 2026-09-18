"""Experiment variant constants, groups, and schema key sets."""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "nuscene_train.yaml"

LEGACY_VARIANT = "bbox_rigid_v1"
DYNAMIC_VARIANT_V1 = "dynamic_2dgs_direct_velocity_v1"
DYNAMIC_VARIANT_V3 = "dynamic_2dgs_physical_velocity_v3"
DYNAMIC_VARIANT_V3_1 = "dynamic_2dgs_physical_velocity_v3_1"
DYNAMIC_VARIANT_V4 = "dynamic_2dgs_attention_velocity_v4"
DYNAMIC_VARIANT_V5 = "dynamic_2dgs_attention_velocity_v5"
DYNAMIC_VARIANT_V6 = "dynamic_2dgs_attention_velocity_v6"
DYNAMIC_VARIANT_V7 = "dynamic_2dgs_attention_velocity_v7"
DYNAMIC_VARIANT_V7_1 = "dynamic_2dgs_attention_velocity_v7_1"
DYNAMIC_VARIANT_V7_2 = "dynamic_2dgs_attention_velocity_v7_2"
DYNAMIC_VARIANT_V8 = "dynamic_2dgs_attention_velocity_v8"
DYNAMIC_VARIANT_V9 = "dynamic_2dgs_attention_velocity_v9"
DYNAMIC_VARIANT_V10 = "dynamic_2dgs_attention_velocity_v10"
DYNAMIC_VARIANT_V11 = "dynamic_2dgs_attention_velocity_v11"
DYNAMIC_VARIANT_V11_1 = "dynamic_2dgs_attention_velocity_v11_1"
DYNAMIC_VARIANT_V11_2 = "dynamic_2dgs_attention_velocity_v11_2"
DYNAMIC_VARIANT_V11_3 = "dynamic_2dgs_attention_velocity_v11_3"
DYNAMIC_VARIANT_V11_4 = "dynamic_2dgs_attention_velocity_v11_4"
DYNAMIC_VARIANT_V12 = "dynamic_2dgs_attention_velocity_v12"
DYNAMIC_VARIANT_V12_1 = "dynamic_2dgs_attention_velocity_v12_1"
DYNAMIC_VARIANT_V13 = "dynamic_2dgs_attention_velocity_v13"
DYNAMIC_VARIANT_V13_1 = "dynamic_2dgs_attention_velocity_v13_1"
DYNAMIC_VARIANT_V13_2 = "dynamic_2dgs_attention_velocity_v13_2"
DYNAMIC_VARIANT_V13_3 = "dynamic_2dgs_attention_velocity_v13_3"
# The unsuffixed constant always names the current fresh-run baseline. Keep an
# explicit constant for each predecessor so checkpoint semantics stay exact.
DYNAMIC_VARIANT = DYNAMIC_VARIANT_V7_2
# V4/V5 use a dense coordinate expectation from selected attention heads. V8
# reuses selected final-layer heads with a consensus readout; V10 reads every
# head at every layer and therefore has no `motion_head_count` config key. V11
# reads head 0 at every layer, as do V11.3 and the V13 family; V11.2 gives every
# layer head RoPE and reads one dedicated head after refinement. In every case
# the match head is fixed rather than configured.
_SELECTED_HEAD_ATTENTION_VELOCITY_VARIANTS = (
    DYNAMIC_VARIANT_V4,
    DYNAMIC_VARIANT_V5,
    DYNAMIC_VARIANT_V8,
)
ATTENTION_VELOCITY_VARIANTS = (
    *_SELECTED_HEAD_ATTENTION_VELOCITY_VARIANTS,
    DYNAMIC_VARIANT_V10,
    DYNAMIC_VARIANT_V11,
    DYNAMIC_VARIANT_V11_1,
    DYNAMIC_VARIANT_V11_2,
    DYNAMIC_VARIANT_V11_3,
    DYNAMIC_VARIANT_V11_4,
    DYNAMIC_VARIANT_V12,
    DYNAMIC_VARIANT_V12_1,
    DYNAMIC_VARIANT_V13,
    DYNAMIC_VARIANT_V13_1,
    DYNAMIC_VARIANT_V13_2,
    DYNAMIC_VARIANT_V13_3,
)
# Variants whose backend owns the learned_gumbel K={1,2,3} grid Gaussian head,
# and can therefore let a router decide how many Gaussians each token becomes.
# Every other dynamic variant emits a fixed count per token. Which of the two a
# run uses is still a config decision (p2g.grid_query.count_mode) for the
# variants listed here. V11.3 shares V11's backend class but is deliberately
# absent: its identity is the fixed two Gaussians per token, so asking it for
# the router is rejected rather than quietly turning it back into V11.
ROUTER_CAPABLE_DYNAMIC_VARIANTS = (
    DYNAMIC_VARIANT_V10,
    DYNAMIC_VARIANT_V11,
    DYNAMIC_VARIANT_V11_2,
    DYNAMIC_VARIANT_V11_4,
    DYNAMIC_VARIANT_V12,
    DYNAMIC_VARIANT_V12_1,
    DYNAMIC_VARIANT_V13,
    DYNAMIC_VARIANT_V13_2,
    DYNAMIC_VARIANT_V13_3,
)
# Historical public name retained for scripts and configs that imported it.
# The tuple describes backend capability; the selected count mode still comes
# from p2g.grid_query.
ADAPTIVE_COUNT_DYNAMIC_VARIANTS = ROUTER_CAPABLE_DYNAMIC_VARIANTS
# V13-family correspondence reads head 0 at every temporal layer, but unlike
# the barrier family all heads carry 3D RoPE and share one FlashAttention call.
# In V13/V13.1 only, head 0 is L2-normalized and temperature-scaled.
COSINE_ROPE_MATCH_DYNAMIC_VARIANTS = (
    DYNAMIC_VARIANT_V13,
    DYNAMIC_VARIANT_V13_1,
)
# V13.3 restores V11.4's per-layer Q/K RMSNorm and applies 3D RoPE to every
# head.
RMSNORM_ROPE_MATCH_DYNAMIC_VARIANTS = (
    DYNAMIC_VARIANT_V13_3,
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
# Shared single-Flash packed coordinate-readout family. Position encoding and
# Q/K normalization remain variant-specific contracts inside this group.
FLASH_ROPE_MATCH_DYNAMIC_VARIANTS = (
    *ALL_HEAD_ROPE_MATCH_DYNAMIC_VARIANTS,
)
# Variants whose correspondence runs under the max-speed barrier. In V11 and
# V11.1 head 0 carries it at every layer while every other head keeps 3D RoPE,
# and V11.1 additionally splits the QK-Norm gain along that head boundary.
# V11.3 is V11's correspondence exactly, differing only in Gaussian count.
# V11.4 is V11 with the hinge made linear: barrier_exponent=1 at weight 1.
# V12 keeps V11.4's linear hinge and shared QK-Norm gains, but learns its one
# global max-speed radius in log space from an initial 30 m/s.
# V12.1 keeps V12's learned radius, fixes head 0's RMS gain to one, and lets
# heads 1..H-1 share a separate learned gain.
# V11.2 gives head 0 RoPE too and moves the barrier to a single post-refinement
# readout head with its own gain, which is what lets its layer stack run
# entirely on FlashAttention.
BARRIER_MATCH_DYNAMIC_VARIANTS = (
    DYNAMIC_VARIANT_V11,
    DYNAMIC_VARIANT_V11_1,
    DYNAMIC_VARIANT_V11_2,
    DYNAMIC_VARIANT_V11_3,
    DYNAMIC_VARIANT_V11_4,
    DYNAMIC_VARIANT_V12,
    DYNAMIC_VARIANT_V12_1,
)
DYNAMIC_VARIANTS = (
    DYNAMIC_VARIANT_V1,
    DYNAMIC_VARIANT_V3,
    DYNAMIC_VARIANT_V3_1,
    DYNAMIC_VARIANT_V4,
    DYNAMIC_VARIANT_V5,
    DYNAMIC_VARIANT_V6,
    DYNAMIC_VARIANT_V7,
    DYNAMIC_VARIANT_V7_1,
    DYNAMIC_VARIANT_V7_2,
    DYNAMIC_VARIANT_V8,
    DYNAMIC_VARIANT_V9,
    DYNAMIC_VARIANT_V10,
    DYNAMIC_VARIANT_V11,
    DYNAMIC_VARIANT_V11_1,
    DYNAMIC_VARIANT_V11_2,
    DYNAMIC_VARIANT_V11_3,
    DYNAMIC_VARIANT_V11_4,
    DYNAMIC_VARIANT_V12,
    DYNAMIC_VARIANT_V12_1,
    DYNAMIC_VARIANT_V13,
    DYNAMIC_VARIANT_V13_1,
    DYNAMIC_VARIANT_V13_2,
    DYNAMIC_VARIANT_V13_3,
)

# V3 and later share the physical-time contract. V4/V5 add attention-derived
# initialization; V6 uses an independent proposal branch.
PHYSICAL_VELOCITY_VARIANTS = (
    DYNAMIC_VARIANT_V3,
    DYNAMIC_VARIANT_V3_1,
    DYNAMIC_VARIANT_V4,
    DYNAMIC_VARIANT_V5,
    DYNAMIC_VARIANT_V6,
    DYNAMIC_VARIANT_V7,
    DYNAMIC_VARIANT_V7_1,
    DYNAMIC_VARIANT_V7_2,
    DYNAMIC_VARIANT_V8,
    DYNAMIC_VARIANT_V9,
    DYNAMIC_VARIANT_V10,
    DYNAMIC_VARIANT_V11,
    DYNAMIC_VARIANT_V11_1,
    DYNAMIC_VARIANT_V11_2,
    DYNAMIC_VARIANT_V11_3,
    DYNAMIC_VARIANT_V11_4,
    DYNAMIC_VARIANT_V12,
    DYNAMIC_VARIANT_V12_1,
    DYNAMIC_VARIANT_V13,
    DYNAMIC_VARIANT_V13_1,
    DYNAMIC_VARIANT_V13_2,
    DYNAMIC_VARIANT_V13_3,
)

WARPED_PROPOSAL_VELOCITY_VARIANTS = (
    DYNAMIC_VARIANT_V7,
    DYNAMIC_VARIANT_V7_1,
    DYNAMIC_VARIANT_V7_2,
)
DURATION_OFFSET_VELOCITY_VARIANTS = (
    DYNAMIC_VARIANT_V7,
    DYNAMIC_VARIANT_V7_1,
)
# These proposals run *before* temporal attention on raw LoRA Utonia tokens,
# which is what makes them need a detached query warp to reach the trunk.
PROPOSAL_VELOCITY_VARIANTS = (
    DYNAMIC_VARIANT_V6,
    *WARPED_PROPOSAL_VELOCITY_VARIANTS,
)
# V9 keeps a standalone proposal branch but reads the refined feature instead,
# so it shares the `motion_proposal` config root and nothing else with the above.
REFINED_PROPOSAL_VELOCITY_VARIANTS = (
    DYNAMIC_VARIANT_V9,
)

VARIANT_CONFIG_PATHS = {
    LEGACY_VARIANT: REPO_ROOT / "config" / "variants" / f"{LEGACY_VARIANT}.yaml",
    DYNAMIC_VARIANT_V1: (
        REPO_ROOT / "config" / "variants" / f"{DYNAMIC_VARIANT_V1}.yaml"
    ),
    DYNAMIC_VARIANT_V3: (
        REPO_ROOT / "config" / "variants" / f"{DYNAMIC_VARIANT_V3}.yaml"
    ),
    DYNAMIC_VARIANT_V3_1: (
        REPO_ROOT / "config" / "variants" / f"{DYNAMIC_VARIANT_V3_1}.yaml"
    ),
    DYNAMIC_VARIANT_V4: (
        REPO_ROOT / "config" / "variants" / f"{DYNAMIC_VARIANT_V4}.yaml"
    ),
    DYNAMIC_VARIANT_V5: (
        REPO_ROOT / "config" / "variants" / f"{DYNAMIC_VARIANT_V5}.yaml"
    ),
    DYNAMIC_VARIANT_V6: (
        REPO_ROOT / "config" / "variants" / f"{DYNAMIC_VARIANT_V6}.yaml"
    ),
    DYNAMIC_VARIANT_V7: (
        REPO_ROOT / "config" / "variants" / f"{DYNAMIC_VARIANT_V7}.yaml"
    ),
    DYNAMIC_VARIANT_V7_1: (
        REPO_ROOT / "config" / "variants" / f"{DYNAMIC_VARIANT_V7_1}.yaml"
    ),
    DYNAMIC_VARIANT_V7_2: (
        REPO_ROOT / "config" / "variants" / f"{DYNAMIC_VARIANT_V7_2}.yaml"
    ),
    DYNAMIC_VARIANT_V8: (
        REPO_ROOT / "config" / "variants" / f"{DYNAMIC_VARIANT_V8}.yaml"
    ),
    DYNAMIC_VARIANT_V9: (
        REPO_ROOT / "config" / "variants" / f"{DYNAMIC_VARIANT_V9}.yaml"
    ),
    DYNAMIC_VARIANT_V10: (
        REPO_ROOT / "config" / "variants" / f"{DYNAMIC_VARIANT_V10}.yaml"
    ),
    DYNAMIC_VARIANT_V11: (
        REPO_ROOT / "config" / "variants" / f"{DYNAMIC_VARIANT_V11}.yaml"
    ),
    DYNAMIC_VARIANT_V11_1: (
        REPO_ROOT / "config" / "variants" / f"{DYNAMIC_VARIANT_V11_1}.yaml"
    ),
    DYNAMIC_VARIANT_V11_2: (
        REPO_ROOT / "config" / "variants" / f"{DYNAMIC_VARIANT_V11_2}.yaml"
    ),
    DYNAMIC_VARIANT_V11_3: (
        REPO_ROOT / "config" / "variants" / f"{DYNAMIC_VARIANT_V11_3}.yaml"
    ),
    DYNAMIC_VARIANT_V11_4: (
        REPO_ROOT / "config" / "variants" / f"{DYNAMIC_VARIANT_V11_4}.yaml"
    ),
    DYNAMIC_VARIANT_V12: (
        REPO_ROOT / "config" / "variants" / f"{DYNAMIC_VARIANT_V12}.yaml"
    ),
    DYNAMIC_VARIANT_V12_1: (
        REPO_ROOT / "config" / "variants" / f"{DYNAMIC_VARIANT_V12_1}.yaml"
    ),
    DYNAMIC_VARIANT_V13: (
        REPO_ROOT / "config" / "variants" / f"{DYNAMIC_VARIANT_V13}.yaml"
    ),
    DYNAMIC_VARIANT_V13_1: (
        REPO_ROOT / "config" / "variants" / f"{DYNAMIC_VARIANT_V13_1}.yaml"
    ),
    DYNAMIC_VARIANT_V13_2: (
        REPO_ROOT / "config" / "variants" / f"{DYNAMIC_VARIANT_V13_2}.yaml"
    ),
    DYNAMIC_VARIANT_V13_3: (
        REPO_ROOT / "config" / "variants" / f"{DYNAMIC_VARIANT_V13_3}.yaml"
    ),
}

# W&B omits empty mappings, and configs saved before model variants existed have
# no model block.  Left-merging these defaults restores the historical contract.
STRUCTURAL_DEFAULTS = {
    "model": {"variant": LEGACY_VARIANT},
    "g2g": {},
}

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

_IMPLEMENTED_VARIANTS = {LEGACY_VARIANT, *DYNAMIC_VARIANTS}

# Pointcept/Utonia encoder widths from config/utonia_pretrained.yaml. LoRA mode
# exits at the configured encoder stage and bypasses the legacy fusion MLP, so
# static config validation must use this width for downstream attention.
_UTONIA_ENCODER_STAGE_DIMS = (54, 108, 216, 432, 576)


LEGACY_COUNT_MODE = "legacy"
LEARNED_COUNT_MODES = ("learned_gumbel", "learned_gumbel_viewpt")
