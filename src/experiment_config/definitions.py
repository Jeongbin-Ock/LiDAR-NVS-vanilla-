"""Experiment variant constants, groups, and schema key sets."""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
# Dataset-agnostic model/training defaults; data comes from one profile in
# DATA_CONFIG_DIR, named by the variant overlay or the CLI (data_config=).
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "default.yaml"
DATA_CONFIG_DIR = REPO_ROOT / "config" / "data"
# Stage overlays applied on top of a finished checkpoint's embedded config
# (refine_config=<name or path>), e.g. the frozen-model raydrop refinement.
REFINE_CONFIG_DIR = REPO_ROOT / "config" / "refine"
# Sensor-head adaptation overlays (adapt_config=<name or path>): V18's LiDAR
# decoder retrained behind a frozen model, usually on another dataset.
ADAPT_CONFIG_DIR = REPO_ROOT / "config" / "adapt"

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
DYNAMIC_VARIANT_V14 = "dynamic_2dgs_attention_velocity_v14"
DYNAMIC_VARIANT_V14_1 = "dynamic_2dgs_attention_velocity_v14_1"
DYNAMIC_VARIANT_V15 = "dynamic_2dgs_attention_velocity_v15"
DYNAMIC_VARIANT_V15_1 = "dynamic_2dgs_attention_velocity_v15_1"
DYNAMIC_VARIANT_V15_2 = "dynamic_2dgs_attention_velocity_v15_2"
DYNAMIC_VARIANT_V15_3 = "dynamic_2dgs_attention_velocity_v15_3"
DYNAMIC_VARIANT_V15_4 = "dynamic_2dgs_attention_velocity_v15_4"
DYNAMIC_VARIANT_V15_5 = "dynamic_2dgs_attention_velocity_v15_5"
DYNAMIC_VARIANT_V16 = "dynamic_2dgs_attention_velocity_v16"
DYNAMIC_VARIANT_V16_1 = "dynamic_2dgs_attention_velocity_v16_1"
DYNAMIC_VARIANT_V17_1 = "dynamic_2dgs_attention_velocity_v17_1"
DYNAMIC_VARIANT_V18 = "dynamic_2dgs_attention_velocity_v18"
DYNAMIC_VARIANT_V18_1 = "dynamic_2dgs_attention_velocity_v18_1"
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
    DYNAMIC_VARIANT_V14,
    DYNAMIC_VARIANT_V14_1,
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
# V14 runs V11.4's temporal stack unchanged under the hybrid Utonia adaptation;
# V14.1 is V14 with one global learned radius, softplus(v_raw) from 30 m/s.
# V15 keeps V14.1's temporal stack and random-duration loader, minus the
# LayerScale gate, and adds one radius local-attention block after the stack;
# V15.1 is V15 trained on its routed render alone. V15.2 and V15.3 keep V15's
# stack and change only the Gaussian head: V15.2 gives each router option its
# own expert, V15.3 drops the router and renders five Gaussians per token.
# V15.4 is V15.3 with eight temporal heads instead of twelve. V15.5 is V15.4
# with v_init read from the plain 1/L mean of the layers' head-0 readouts
# instead of their learned softmax mixture. V18 is V15.4 whose Gaussians carry
# a feature, alpha-blended and decoded into intensity and raydrop by G2P's lidar
# MLP, instead of SH coefficients. V18.1 is V18 whose MLP reads no sensor
# embedding.
# V16 keeps V15's stack and G1..G5 decoder and changes the router: two bounded
# ordinal gates trained through masked rasterization. V16.1 is V16 whose second
# training render is a uniformly random prefix of G1..G5 per token instead of
# every token's G1..G5.
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
    DYNAMIC_VARIANT_V14,
    DYNAMIC_VARIANT_V14_1,
    DYNAMIC_VARIANT_V15,
    DYNAMIC_VARIANT_V15_1,
    DYNAMIC_VARIANT_V15_2,
    DYNAMIC_VARIANT_V15_3,
    DYNAMIC_VARIANT_V15_4,
    DYNAMIC_VARIANT_V15_5,
    DYNAMIC_VARIANT_V16,
    DYNAMIC_VARIANT_V16_1,
    DYNAMIC_VARIANT_V17_1,
    DYNAMIC_VARIANT_V18,
    DYNAMIC_VARIANT_V18_1,
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
    DYNAMIC_VARIANT_V14,
    DYNAMIC_VARIANT_V14_1,
    DYNAMIC_VARIANT_V15,
    DYNAMIC_VARIANT_V15_1,
    DYNAMIC_VARIANT_V15_2,
    DYNAMIC_VARIANT_V15_3,
    DYNAMIC_VARIANT_V15_4,
    DYNAMIC_VARIANT_V15_5,
    DYNAMIC_VARIANT_V16,
    DYNAMIC_VARIANT_V16_1,
    DYNAMIC_VARIANT_V17_1,
    DYNAMIC_VARIANT_V18,
    DYNAMIC_VARIANT_V18_1,
)
# Variants whose Utonia encoder is adapted by p2g.utonia_adaptation.mode=hybrid:
# frozen XYZ stem columns plus a trained intensity column and a nonlinear stem
# adapter, AdaptFormer-style parallel adapters beside every executed FFN,
# trainable LayerNorm affine, and LoRA only on attention, CPE and pooling, each
# role with its own AdamW LR (train.utonia_param_groups). Every other variant
# keeps LoRA on every Utonia Linear/SubMConv3d and rejects both blocks. V14 and
# V14.1 put the stem adapter before the stem LayerNorm; V15 puts it after the
# activation (see POST_ACTIVATION_STEM_DYNAMIC_VARIANTS). Only these variants
# have per-role optimizer groups, so only they accept train.downstream_no_decay.
# Every V15-family variant (V15, V15.1, V15.2, V15.3, V15.4, V15.5, V16, V16.1,
# V18, V18.1) shares V15's Utonia adaptation, temporal stack and local-attention block;
# they differ only in how the Gaussian head turns a token into Gaussians, except
# V15.4, which is V15.3 with eight temporal heads instead of twelve, V15.5,
# which is V15.4 with v_init read from the uniform mean over layers, and V18
# and V18.1, which are V15.4 with a rendered feature in place of the SH
# coefficients.
V15_FAMILY_DYNAMIC_VARIANTS = (
    DYNAMIC_VARIANT_V15,
    DYNAMIC_VARIANT_V15_1,
    DYNAMIC_VARIANT_V15_2,
    DYNAMIC_VARIANT_V15_3,
    DYNAMIC_VARIANT_V15_4,
    DYNAMIC_VARIANT_V15_5,
    DYNAMIC_VARIANT_V16,
    DYNAMIC_VARIANT_V16_1,
    DYNAMIC_VARIANT_V18,
    DYNAMIC_VARIANT_V18_1,
)
HYBRID_UTONIA_DYNAMIC_VARIANTS = (
    DYNAMIC_VARIANT_V14,
    DYNAMIC_VARIANT_V14_1,
    *V15_FAMILY_DYNAMIC_VARIANTS,
)
# V17.1 keeps V15.4's downstream path and trains only the zero-init intensity
# column of the pretrained Utonia stem. No LoRA, nonlinear adapters or LN fit.
INTENSITY_ONLY_UTONIA_DYNAMIC_VARIANTS = (DYNAMIC_VARIANT_V17_1,)
UTONIA_PARAM_GROUP_DYNAMIC_VARIANTS = (
    *HYBRID_UTONIA_DYNAMIC_VARIANTS,
    *INTENSITY_ONLY_UTONIA_DYNAMIC_VARIANTS,
)
# V15's nested Gaussian count (p2g.grid_query.count_mode=nested_gumbel): one
# shared token anchor, G1..G5 decoded from the refined feature, and a router over
# the prefixes {G1}, {G1..G3}, {G1..G5}. This is not the learned_gumbel K={1,2,3}
# router, so V15 is deliberately absent from ROUTER_CAPABLE_DYNAMIC_VARIANTS and
# no other variant may ask for it.
NESTED_COUNT_DYNAMIC_VARIANTS = (
    DYNAMIC_VARIANT_V15,
    DYNAMIC_VARIANT_V15_1,
)
# V15.2's per-option experts (p2g.grid_query.count_mode=expert_gumbel): the same
# reduce/keep/add router as V15, but each option owns its own Gaussians -- E1,
# E3 or E5, linear experts on one shared trunk -- instead of a prefix of G1..G5.
EXPERT_COUNT_DYNAMIC_VARIANTS = (DYNAMIC_VARIANT_V15_2,)
# V15.3 has no router: every token decodes and renders five Gaussians around its
# token anchor (count_mode=legacy, K_max=5, seed_mode=token) through V15's
# two-layer decoder, whose hidden width is dynamic_2dgs.gaussian_head.decoder_hidden.
# V15.4 is V15.3 with eight temporal heads (54 channels each) instead of twelve
# (36 each). V15.5 is V15.4 whose v_init averages the layers' head-0 readouts
# uniformly instead of under learned layer logits. V18 and V18.1 decode V15.4's
# five slots with a feature in place of the SH coefficients.
DENSE_COUNT_DYNAMIC_VARIANTS = (
    DYNAMIC_VARIANT_V15_3,
    DYNAMIC_VARIANT_V15_4,
    DYNAMIC_VARIANT_V15_5,
    DYNAMIC_VARIANT_V17_1,
    DYNAMIC_VARIANT_V18,
    DYNAMIC_VARIANT_V18_1,
)
# The temporal head count is what separates V15.4 from V15.3, so each pins its
# own dynamic_2dgs.temporal.num_heads; V15.5, V18 and V18.1 keep V15.4's eight.
DENSE_COUNT_TEMPORAL_HEADS_BY_VARIANT = {
    DYNAMIC_VARIANT_V15_3: 12,
    DYNAMIC_VARIANT_V15_4: 8,
    DYNAMIC_VARIANT_V15_5: 8,
    DYNAMIC_VARIANT_V17_1: 8,
    DYNAMIC_VARIANT_V18: 8,
    DYNAMIC_VARIANT_V18_1: 8,
}
# V18 (SplatAD-style LiDAR appearance): every Gaussian carries a feature of
# p2g.gs_params.feature channels instead of SH coefficients (shs: 0); the
# rasterizer alpha-blends it, and G2P's lidar decoder (g2p.lidar_decoder) maps
# [blended feature, world-frame ray direction, median depth / range_scale_m,
# learned sensor embedding] to intensity and raydrop with one MLP.
FEATURE_DECODER_DYNAMIC_VARIANTS = (DYNAMIC_VARIANT_V18, DYNAMIC_VARIANT_V18_1)
# V18.1's decoder reads no sensor embedding (g2p.lidar_decoder.sensor_embedding_dim:
# 0, no num_sensors). With one LiDAR per training set, V18's embedding is the same
# vector at every pixel, so it only adds W_e e to the first layer's pre-activation,
# a constant its bias can hold: the two decoders represent the same functions,
# but V18.1 trains that bias directly instead of through the product W_e e.
NO_SENSOR_EMBEDDING_DYNAMIC_VARIANTS = (DYNAMIC_VARIANT_V18_1,)
# The rasterizer blends at most this many feature channels per Gaussian
# (MAX_FEATURE_CHANNELS in its cuda_rasterizer/config.h).
RASTERIZER_MAX_FEATURE_CHANNELS = 32
# V16's ordinal nested count (p2g.grid_query.count_mode=nested_ordinal): V15's
# G1..G5 prefixes, chosen by two bounded gates (G2/G3 present; G4/G5 present
# given G2/G3) that train through masked rasterization of all five Gaussians.
# V16.1 differs from V16 only in its router-free second render.
ORDINAL_COUNT_DYNAMIC_VARIANTS = (DYNAMIC_VARIANT_V16, DYNAMIC_VARIANT_V16_1)
# V15 runs one radius local-attention block (dynamic_2dgs.local_attention) after
# the temporal stack. The velocity offset head still reads the stack's output.
LOCAL_ATTENTION_DYNAMIC_VARIANTS = (
    *V15_FAMILY_DYNAMIC_VARIANTS, DYNAMIC_VARIANT_V17_1,
)
# V15's temporal cross-attention adds its branch to the residual stream without
# the per-layer LayerScale gate (dynamic_2dgs.temporal.layer_scale=false).
# Every other variant keeps the gate and its layer_scale_init.
NO_TEMPORAL_LAYER_SCALE_DYNAMIC_VARIANTS = LOCAL_ATTENTION_DYNAMIC_VARIANTS
# V15's Utonia stem adapter reads GELU(LN(W x)) and adds a residual to it
# (p2g.utonia_adaptation.stem.adapter_position=post_activation). V14 and V14.1
# keep the adapter on [x, y, z, I] before the stem LayerNorm.
POST_ACTIVATION_STEM_DYNAMIC_VARIANTS = V15_FAMILY_DYNAMIC_VARIANTS

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
    # Absent (= disabled) in every model checkpoint; only refine_config
    # (adapt_config) can add it, and a refinement (adaptation) checkpoint
    # keeps it fixed.
    "raydrop_refine",
    "sensor_adapt",
)

# Roots a refine_config overlay (and its CLI) may set on top of the frozen
# checkpoint's config; the model and data stay the checkpoint's.
REFINE_OVERLAY_ROOTS = (
    "raydrop_refine",
    "train",
    "test",
    "logger",
    "metrics",
    "exp_name",
    "project_name",
    "device",
    "seed",
    "mode",
)

# Roots an adapt_config overlay (and its CLI) may set: the refinement roots
# with sensor_adapt in place of raydrop_refine, plus the loss weights, since
# the decoder trains on intensity and raydrop alone.
ADAPT_OVERLAY_ROOTS = (
    "sensor_adapt",
    "loss",
    *(root for root in REFINE_OVERLAY_ROOTS if root != "raydrop_refine"),
)
# sensor_adapt.decoder_init: where the trained sensor head starts. ``base``
# loads the base checkpoint's sensor embedding (V18.1 has none) and MLP with
# every other weight; ``random`` keeps the head at its fresh initialization.
SENSOR_ADAPT_DECODER_INITS = ("base", "random")

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
    "data.split_manifest_path",
    "data.split_manifest_sha256",
    "data.window_stride_sec",
    "data.max_sweep_gap_sec",
    "data.sweep_hop",
    "data.pose_missing_policy",
    "data.panorama_projection",
    "data.column_assignment",
    "data.gt_ego_radius",
    "data.max_depth_m",
)

# Pointcept/Utonia encoder widths from config/utonia_pretrained.yaml. The token
# width that reaches temporal attention is the width of the configured encoder
# exit stage, so static config validation reads it from here.
_UTONIA_ENCODER_STAGE_DIMS = (54, 108, 216, 432, 576)


LEGACY_COUNT_MODE = "legacy"
LEARNED_COUNT_MODES = ("learned_gumbel",)
# V15's nested count: every token decodes G1..G5 around one shared token
# anchor, and a router picks the prefix it renders -- reduce {G1}, keep
# {G1..G3} or add {G1..G5}. ``dense_keep`` additionally renders every token's
# G1..G3 during training and optimizes the mean of the two render losses;
# ``none`` renders the routed set alone.
NESTED_COUNT_MODE = "nested_gumbel"
NESTED_COUNT_OPTION_COUNTS = (1, 3, 5)
NESTED_AUXILIARY_RENDERS = ("dense_keep", "none")
# The auxiliary render is each nested variant's identity: V15 trains on the
# mean of the routed render and every token's G1..G3; V15.1 is V15 without
# that second render.
NESTED_AUXILIARY_RENDER_BY_VARIANT = {
    "dynamic_2dgs_attention_velocity_v15": "dense_keep",
    "dynamic_2dgs_attention_velocity_v15_1": "none",
}
# V15.2's per-option experts: the router picks reduce / keep / add as in V15,
# and the token renders that option's own expert, E1, E3 or E5, every Gaussian
# of it gated by the chosen option. ``dense_add`` also renders every token's E5
# during training and optimizes the mean of the two render losses; ``none``
# renders the routed set alone.
EXPERT_COUNT_MODE = "expert_gumbel"
EXPERT_COUNT_OPTION_COUNTS = NESTED_COUNT_OPTION_COUNTS
EXPERT_AUXILIARY_RENDERS = ("dense_add", "none")
EXPERT_AUXILIARY_RENDER_BY_VARIANT = {
    "dynamic_2dgs_attention_velocity_v15_2": "dense_add",
}
# V16's ordinal nested count: V15's G1..G5 and nested choices, but the router
# ends in two cosine gates at the fixed scale ``router_scale`` -- keep (G2/G3
# present) and add (G4/G5 present given G2/G3) -- sampled as hard Bernoulli
# gates with a straight-through backward. Training renders all five Gaussians
# of every token with per-Gaussian blend masks (masked rasterization). The
# second training render is unmasked and router-free: ``dense_add`` renders
# every token's G1..G5; ``random_prefix`` renders, for every token, a prefix
# of G1..G5 whose length is drawn uniformly from {1, 3, 5} each step; ``none``
# renders the masked set alone.
ORDINAL_COUNT_MODE = "nested_ordinal"
ORDINAL_COUNT_OPTION_COUNTS = NESTED_COUNT_OPTION_COUNTS
ORDINAL_AUXILIARY_RENDERS = ("dense_add", "random_prefix", "none")
# The auxiliary render is what separates V16 from V16.1.
ORDINAL_AUXILIARY_RENDER_BY_VARIANT = {
    "dynamic_2dgs_attention_velocity_v16": "dense_add",
    "dynamic_2dgs_attention_velocity_v16_1": "random_prefix",
}
# The count modes whose router picks 1, 3 or 5 Gaussians after the local
# refinement, each with the p2g.grid_query block that configures it.
ROUTED_COUNT_BLOCKS = {
    NESTED_COUNT_MODE: "nested_count",
    EXPERT_COUNT_MODE: "expert_count",
    ORDINAL_COUNT_MODE: "ordinal_count",
}
# V15.3's fixed count: K_max Gaussians per token, all on the token anchor.
DENSE_COUNT_GAUSSIANS_PER_TOKEN = 5

# Where a token's Gaussians start. ``range_quantile`` spreads distinct seeds
# over the token's raw points; ``medoid`` and ``token`` give every slot one
# shared anchor (the observed medoid, or the Utonia token position) from which
# the Gaussian head's bounded offsets place each slot.
SEED_MODE_RANGE_QUANTILE = "range_quantile"
SEED_MODE_MEDOID = "medoid"
SEED_MODE_TOKEN = "token"
FIXED_COUNT_SEED_MODES = (
    SEED_MODE_RANGE_QUANTILE,
    SEED_MODE_MEDOID,
    SEED_MODE_TOKEN,
)
LEARNED_COUNT_SEED_MODES = (SEED_MODE_RANGE_QUANTILE, SEED_MODE_TOKEN)

# ``p2g.grid_query.exp`` before it was renamed to ``seed_mode``. Checkpoint
# configs are rewritten on restore; fresh configs must use the new name.
LEGACY_EXP_SEED_MODES = {
    None: SEED_MODE_RANGE_QUANTILE,
    1: SEED_MODE_MEDOID,
    2: SEED_MODE_TOKEN,
}

# p2g.utonia_adaptation vocabulary. ``hybrid`` enables the V14/V15 adapters;
# ``intensity_only`` enables only V17.1's new intensity column. Configs without
# the block keep the all-linear LoRA. LoRA targets name module roles: attention
# (qkv and output projection), cpe (SubMConv3d and the Linear after it) and
# pooling (GridPooling projection). The stem and FFN are deliberately absent:
# they adapt through the intensity column and the parallel adapters instead.
UTONIA_ADAPTATION_MODE_HYBRID = "hybrid"
UTONIA_ADAPTATION_MODE_INTENSITY_ONLY = "intensity_only"
UTONIA_LORA_TARGETS = ("attention", "cpe", "pooling")
UTONIA_ADAPTER_ACTIVATIONS = ("relu", "gelu")
# Where the stem's nonlinear adapter sits. ``pre_norm`` (the key's absent
# default, V14/V14.1) adds an adapter on [x, y, z, I] to the stem Linear output
# before the stem LayerNorm; ``post_activation`` (V15) leaves the pre-norm stem
# at frozen XYZ plus the intensity column and adds a residual adapter to
# h = GELU(LN(W x)).
UTONIA_STEM_ADAPTER_POSITIONS = ("pre_norm", "post_activation")
# train.utonia_param_groups roles; adapter also takes bias_weight_decay.
UTONIA_PARAM_GROUP_ROLES = ("intensity", "adapter", "lora", "layer_norm")

# dynamic_2dgs.local_attention.implementation: ``flash_varlen`` runs the packed
# neighbour lists through FlashAttention2, ``sdpa`` is the PyTorch reference
# (also the CPU path) and ``auto`` picks flash_varlen on CUDA when available.
LOCAL_ATTENTION_IMPLEMENTATIONS = ("flash_varlen", "sdpa", "auto")
# Every query is one FlashAttention2 varlen sequence, and one call takes at most
# this many sequences (the grid dimension limit); local_attention.query_chunk_size
# bounds how many queries go into a call.
LOCAL_ATTENTION_MAX_QUERY_CHUNK_SIZE = 65535
