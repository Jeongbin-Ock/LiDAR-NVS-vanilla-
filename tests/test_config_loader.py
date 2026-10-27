from __future__ import annotations

import tempfile
import unittest
import math
from pathlib import Path

import torch
from omegaconf import OmegaConf

from src.experiment_config import (
    ALL_HEAD_ROPE_MATCH_DYNAMIC_VARIANTS,
    BARRIER_MATCH_DYNAMIC_VARIANTS,
    COSINE_ROPE_MATCH_DYNAMIC_VARIANTS,
    FEATURE_HEAD_ROPE_MATCH_DYNAMIC_VARIANTS,
    FLASH_ROPE_MATCH_DYNAMIC_VARIANTS,
    RMSNORM_ROPE_MATCH_DYNAMIC_VARIANTS,
    ROUTER_CAPABLE_DYNAMIC_VARIANTS,
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
    DYNAMIC_VARIANT_V15_5,
    DYNAMIC_VARIANT_V16,
    DYNAMIC_VARIANT_V16_1,
    DYNAMIC_VARIANT_V17_1,
    DYNAMIC_VARIANT_V18,
    DYNAMIC_VARIANT_V18_1,
    DYNAMIC_VARIANT_V18_2,
    DYNAMIC_VARIANTS,
    DENSE_COUNT_DYNAMIC_VARIANTS,
    DENSE_COUNT_TEMPORAL_HEADS_BY_VARIANT,
    EXPERT_AUXILIARY_RENDER_BY_VARIANT,
    EXPERT_AUXILIARY_RENDERS,
    EXPERT_COUNT_DYNAMIC_VARIANTS,
    EXPERT_COUNT_MODE,
    FEATURE_DECODER_DYNAMIC_VARIANTS,
    HYBRID_UTONIA_DYNAMIC_VARIANTS,
    LOCAL_ATTENTION_DYNAMIC_VARIANTS,
    LOCAL_ATTENTION_IMPLEMENTATIONS,
    NESTED_AUXILIARY_RENDER_BY_VARIANT,
    NESTED_AUXILIARY_RENDERS,
    NESTED_COUNT_DYNAMIC_VARIANTS,
    NESTED_COUNT_MODE,
    NESTED_COUNT_OPTION_COUNTS,
    NO_SENSOR_EMBEDDING_DYNAMIC_VARIANTS,
    NO_TEMPORAL_LAYER_SCALE_DYNAMIC_VARIANTS,
    ORDINAL_AUXILIARY_RENDER_BY_VARIANT,
    ORDINAL_AUXILIARY_RENDERS,
    ORDINAL_COUNT_DYNAMIC_VARIANTS,
    ORDINAL_COUNT_MODE,
    ORDINAL_COUNT_OPTION_COUNTS,
    POST_ACTIVATION_STEM_DYNAMIC_VARIANTS,
    ROUTED_COUNT_BLOCKS,
    UNET_DECODER_DYNAMIC_VARIANTS,
    V15_FAMILY_DYNAMIC_VARIANTS,
    VARIANT_CONFIG_PATHS,
    compose_fresh_config,
    load_checkpoint_config,
    resolve_eval_config,
    resolve_main_config,
    validate_experiment_config,
)

# Opts a barrier-family overlay into the random-duration loader, the contract
# the V11.6 and V12.2 overlays ship with. Its holdouts live under
# dynamic_window, so the legacy loader's data.train_excluded_scenes has to be
# cleared.
DYNAMIC_LOADER_CLI = (
    "data.dataset_name=nuscenes_dynamic_lidar4d",
    "data.mode=nvs",
    "data.train_excluded_scenes=null",
    "data.dynamic_window.endpoint_min_sec=0.2",
    "data.dynamic_window.endpoint_max_sec=1.2",
    "data.dynamic_window.eval_seed=42",
    "data.dynamic_window.train_excluded_scenes=[scene-0757,scene-1094]",
)


class ConfigLoaderTest(unittest.TestCase):
    def test_default_compose_is_current_dynamic(self):
        config, source = compose_fresh_config()
        self.assertEqual(config.model.variant, DYNAMIC_VARIANT)
        self.assertIn(f"{DYNAMIC_VARIANT}.yaml", source)

    def test_dynamic_effective_config_excludes_legacy_blocks(self):
        config, _ = compose_fresh_config()
        for removed in (
            "squery", "anchor_mode", "intensity_encoder", "int_proj",
            "intensity_sparse", "utonia_adapter", "agg_mlp", "joint_refiner",
            "r_far",
        ):
            self.assertNotIn(removed, config.p2g)
        self.assertNotIn("small_motion", config.dynamic_2dgs.regularization)
        self.assertNotIn("budget", config.p2g.grid_query.learned_count)
        self.assertNotIn("w_scale", config.loss)
        self.assertNotIn("enable_lpips", config.loss)
        self.assertNotIn("dynamic", config.g2p)
        self.assertEqual(config.loss.scale_max_m, 2.5)

    def test_dynamic_overlay_and_cli_precedence(self):
        cli = OmegaConf.from_dotlist([
            f"model.variant={DYNAMIC_VARIANT}",
            "train.batch_size=3",
        ])
        config, source = compose_fresh_config(cli)
        self.assertEqual(config.model.variant, DYNAMIC_VARIANT)
        self.assertEqual(config.model.variant, DYNAMIC_VARIANT_V11_4)
        temporal = config.dynamic_2dgs.temporal
        self.assertEqual(temporal.time_reference_sec, 1.0)
        self.assertEqual(temporal.rope_base, 100.0)
        self.assertAlmostEqual(temporal.rope_position_scale, 2.0 * math.pi / 5.0)
        self.assertEqual(temporal.position_encoding, "barrier_rope_split")
        self.assertEqual(temporal.barrier_weight, 1.0)
        self.assertEqual(temporal.barrier_exponent, 1)
        self.assertTrue(config.p2g.freeze_utonia)
        self.assertTrue(config.p2g.utonia_lora.enable)
        self.assertEqual(config.p2g.utonia_lora.input_mode, "xyzi")
        self.assertEqual(config.p2g.utonia_lora.rank, 16)
        self.assertEqual(config.p2g.utonia_lora.alpha, 16)
        self.assertEqual(config.p2g.utonia_feature_stage, 3)
        self.assertEqual(temporal.layers, 12)
        self.assertEqual(temporal.num_heads, 12)
        self.assertEqual(config.p2g.grid_query.count_mode, "learned_gumbel")
        self.assertEqual(config.dynamic_2dgs.motion.residual_hidden_dim, 96)
        self.assertEqual(list(config.device), [2, 3])
        self.assertIsNone(config.data.bbox_json_path)
        self.assertEqual(config.train.batch_size, 3)
        self.assertEqual(
            OmegaConf.to_container(config, resolve=True)["exp_name"],
            f"{DYNAMIC_VARIANT}-bs-3",
        )
        self.assertIn(f"{DYNAMIC_VARIANT}.yaml", source)

    def test_utonia_lora_is_required(self):
        with self.assertRaisesRegex(ValueError, "utonia_lora.enable=true"):
            compose_fresh_config(OmegaConf.from_dotlist([
                "p2g.utonia_lora.enable=false",
            ]))

    def test_utonia_lora_requires_frozen_base(self):
        with self.assertRaisesRegex(ValueError, "freeze_utonia=true"):
            compose_fresh_config(OmegaConf.from_dotlist([
                "p2g.freeze_utonia=false",
            ]))

    def test_utonia_lora_rejects_invalid_adapter_settings(self):
        with self.assertRaisesRegex(ValueError, "linear_rank must be positive"):
            compose_fresh_config(OmegaConf.from_dotlist([
                "p2g.utonia_lora.linear_rank=0",
            ]))

    def test_utonia_lora_stage_width_validates_temporal_attention(self):
        # Stage 2 is 216D; 216/8=27 is invalid for xyz/sin-cos 3D RoPE.
        with self.assertRaisesRegex(ValueError, "head_dim=27"):
            compose_fresh_config(OmegaConf.from_dotlist([
                "p2g.utonia_feature_stage=2",
                "dynamic_2dgs.temporal.num_heads=8",
            ]))

    def test_current_dynamic_enables_the_split_velocity_prior(self):
        config, _ = compose_fresh_config()
        prior = config.dynamic_2dgs.regularization.velocity_l2
        self.assertTrue(prior.enabled)
        self.assertEqual(prior.mode, "split_group_l2")
        self.assertEqual(prior.init_weight, 0.01)
        self.assertEqual(prior.offset_weight, 0.01)
        self.assertEqual(prior.warmup_steps, 0)
        self.assertEqual(prior.ramp_steps, 0)

    def test_v11_1_composes_the_single_gaussian_contract(self):
        config, _source = compose_fresh_config(OmegaConf.from_dotlist([
            f"model.variant={DYNAMIC_VARIANT_V11_1}",
        ]))
        temporal = config.dynamic_2dgs.temporal
        # Everything V11 established is inherited unchanged.
        self.assertEqual(temporal.position_encoding, "barrier_rope_split")
        self.assertEqual(temporal.barrier_speed_mps, 30.0)
        self.assertEqual(temporal.barrier_weight, 4.0)
        self.assertEqual(temporal.rope_base, 100.0)
        self.assertTrue(temporal.qk_norm)
        self.assertEqual(config.dynamic_2dgs.regularization.velocity_l2.mode,
                         "split_group_l2")
        self.assertTrue(config.p2g.utonia_lora.enable)
        # One Gaussian per token: no grid_query block is configured, and
        # V11.1's backend has no router it could be pointed at.
        self.assertIsNone(OmegaConf.select(config, "p2g.grid_query"))
        self.assertNotIn(DYNAMIC_VARIANT_V11_1, ROUTER_CAPABLE_DYNAMIC_VARIANTS)
        self.assertIn(DYNAMIC_VARIANT_V11, ROUTER_CAPABLE_DYNAMIC_VARIANTS)

    def test_v11_2_is_value_identical_to_v11_and_keeps_adaptive_k(self):
        v11, _source = compose_fresh_config(OmegaConf.from_dotlist([
            f"model.variant={DYNAMIC_VARIANT_V11}",
        ]))
        v11_2, source = compose_fresh_config(OmegaConf.from_dotlist([
            f"model.variant={DYNAMIC_VARIANT_V11_2}",
        ]))
        left = OmegaConf.to_container(v11, resolve=False)
        right = OmegaConf.to_container(v11_2, resolve=False)
        self.assertNotEqual(left["model"]["variant"], right["model"]["variant"])
        self.assertNotEqual(v11.exp_name, v11_2.exp_name)
        left["model"]["variant"] = right["model"]["variant"]
        self.assertEqual(left, right)
        self.assertIn(f"{DYNAMIC_VARIANT_V11_2}.yaml", source)

        for group in (
            BARRIER_MATCH_DYNAMIC_VARIANTS,
            ROUTER_CAPABLE_DYNAMIC_VARIANTS,
        ):
            self.assertIn(DYNAMIC_VARIANT_V11_2, group)
        count = v11_2.p2g.grid_query
        self.assertEqual(count.count_mode, "learned_gumbel")
        self.assertEqual(count.learned_count.K_max, 3)
        self.assertEqual(count.learned_count.seed_mode, "range_quantile")

        with self.assertRaisesRegex(ValueError, "K_max=3"):
            compose_fresh_config(OmegaConf.from_dotlist([
                f"model.variant={DYNAMIC_VARIANT_V11_2}",
                "p2g.grid_query.learned_count.K_max=4",
            ]))

    def test_v11_3_is_v11_with_a_fixed_count_of_two_gaussians(self):
        """Only model.variant and the Gaussian count separate it from V11."""
        v11, _source = compose_fresh_config(OmegaConf.from_dotlist([
            f"model.variant={DYNAMIC_VARIANT_V11}",
        ]))
        v11_3, source = compose_fresh_config(OmegaConf.from_dotlist([
            f"model.variant={DYNAMIC_VARIANT_V11_3}",
        ]))
        self.assertIn(f"{DYNAMIC_VARIANT_V11_3}.yaml", source)
        self.assertNotEqual(v11.exp_name, v11_3.exp_name)

        left = OmegaConf.to_container(v11, resolve=False)
        right = OmegaConf.to_container(v11_3, resolve=False)
        left["model"]["variant"] = right["model"]["variant"]
        self.assertNotEqual(
            left["p2g"].pop("grid_query"), right["p2g"].pop("grid_query")
        )
        self.assertEqual(left, right)

        count = v11_3.p2g.grid_query
        self.assertEqual(count.count_mode, "legacy")
        self.assertEqual(count.K_max, 2)
        self.assertEqual(count.seed_mode, "range_quantile")

        for group in (
            BARRIER_MATCH_DYNAMIC_VARIANTS,
        ):
            self.assertIn(DYNAMIC_VARIANT_V11_3, group)
        self.assertNotIn(DYNAMIC_VARIANT_V11_3, ROUTER_CAPABLE_DYNAMIC_VARIANTS)

    def test_v11_4_is_v11_with_a_linear_hinge(self):
        """Only the barrier's shape and weight separate it from V11."""
        v11, _source = compose_fresh_config(OmegaConf.from_dotlist([
            f"model.variant={DYNAMIC_VARIANT_V11}",
        ]))
        v11_4, source = compose_fresh_config(OmegaConf.from_dotlist([
            f"model.variant={DYNAMIC_VARIANT_V11_4}",
        ]))
        self.assertIn(f"{DYNAMIC_VARIANT_V11_4}.yaml", source)

        left = OmegaConf.to_container(v11, resolve=False)
        right = OmegaConf.to_container(v11_4, resolve=False)
        left["model"]["variant"] = right["model"]["variant"]
        left_temporal = left["dynamic_2dgs"]["temporal"]
        right_temporal = right["dynamic_2dgs"]["temporal"]
        self.assertEqual(left_temporal.pop("barrier_weight"), 4.0)
        self.assertEqual(right_temporal.pop("barrier_weight"), 1.0)
        self.assertNotIn("barrier_exponent", left_temporal)
        self.assertEqual(right_temporal.pop("barrier_exponent"), 1)
        self.assertEqual(left, right)

        # V11.4 keeps V11's router and barrier-match membership.
        for group in (
            BARRIER_MATCH_DYNAMIC_VARIANTS,
            ROUTER_CAPABLE_DYNAMIC_VARIANTS,
        ):
            self.assertIn(DYNAMIC_VARIANT_V11_4, group)
        self.assertEqual(v11_4.p2g.grid_query.count_mode, "learned_gumbel")

    def test_barrier_exponent_only_accepts_one_or_two(self):
        for value in (0, 3):
            with self.subTest(value=value), self.assertRaises(ValueError):
                compose_fresh_config(OmegaConf.from_dotlist([
                    f"model.variant={DYNAMIC_VARIANT_V11_4}",
                    f"dynamic_2dgs.temporal.barrier_exponent={value}",
                ]))

    def test_v11_5_is_v11_4_with_only_a_fixed_k2_count(self):
        """V11.4 with V11.3's count; the bound and data are V11.4's own."""
        v11_3, _source = compose_fresh_config(OmegaConf.from_dotlist([
            f"model.variant={DYNAMIC_VARIANT_V11_3}",
        ]))
        v11_4, _source = compose_fresh_config(OmegaConf.from_dotlist([
            f"model.variant={DYNAMIC_VARIANT_V11_4}",
        ]))
        v12, _source = compose_fresh_config(OmegaConf.from_dotlist([
            f"model.variant={DYNAMIC_VARIANT_V12}",
        ]))
        v11_5, source = compose_fresh_config(OmegaConf.from_dotlist([
            f"model.variant={DYNAMIC_VARIANT_V11_5}",
        ]))
        self.assertIn(f"{DYNAMIC_VARIANT_V11_5}.yaml", source)

        left = OmegaConf.to_container(v11_4, resolve=False)
        right = OmegaConf.to_container(v11_5, resolve=False)
        left["model"]["variant"] = right["model"]["variant"]
        left_count = left["p2g"].pop("grid_query")
        right_count = right["p2g"].pop("grid_query")
        # V11.5 pins the 0.4 m bound and the legacy loader with scene holdouts
        # in its own overlay; both equal what V11.4 composes to.
        self.assertEqual(right["p2g"]["head_offset_bound"], 0.4)
        self.assertEqual(left, right)
        right_data = right["data"]

        self.assertEqual(left_count["count_mode"], "learned_gumbel")
        self.assertEqual(
            right_count, OmegaConf.to_container(v11_3.p2g.grid_query)
        )
        self.assertEqual(right_count["count_mode"], "legacy")
        self.assertEqual(right_count["K_max"], 2)
        self.assertEqual(right_count["seed_mode"], "range_quantile")

        temporal = v11_5.dynamic_2dgs.temporal
        self.assertEqual(temporal.barrier_weight, 1.0)
        self.assertEqual(temporal.barrier_exponent, 1)
        self.assertEqual(temporal, v11_4.dynamic_2dgs.temporal)

        self.assertEqual(right_data["dataset_name"], "nuscenes")
        self.assertEqual(right_data["mode"], "bbox")
        self.assertIsNone(right_data["bbox_json_path"])
        self.assertNotIn("dynamic_window", right_data)
        self.assertEqual(
            list(right_data["train_excluded_scenes"]),
            ["scene-0757", "scene-1094"],
        )
        self.assertEqual(
            OmegaConf.to_container(v11_5.data, resolve=True),
            OmegaConf.to_container(v12.data, resolve=True),
        )

        for group in (
            BARRIER_MATCH_DYNAMIC_VARIANTS,
        ):
            self.assertIn(DYNAMIC_VARIANT_V11_5, group)
        self.assertNotIn(DYNAMIC_VARIANT_V11_5, ROUTER_CAPABLE_DYNAMIC_VARIANTS)

        with self.assertRaises(ValueError) as raised:
            compose_fresh_config(OmegaConf.from_dotlist([
                f"model.variant={DYNAMIC_VARIANT_V11_5}",
                "p2g.grid_query.count_mode=learned_gumbel",
            ]))
        self.assertIn("V11.5", str(raised.exception))
        self.assertIn("count router", str(raised.exception))

    def test_v11_6_is_v11_5_at_k3_on_the_dynamic_loader(self):
        """V11.5's model with three Gaussians, on the random-duration loader."""
        v11_5, _source = compose_fresh_config(OmegaConf.from_dotlist([
            f"model.variant={DYNAMIC_VARIANT_V11_5}",
        ]))
        # The data contract the V11.4 dynamic-loader run was launched with.
        v11_4_dynamic, _source = compose_fresh_config(OmegaConf.from_dotlist([
            f"model.variant={DYNAMIC_VARIANT_V11_4}", *DYNAMIC_LOADER_CLI,
        ]))
        v11_6, source = compose_fresh_config(OmegaConf.from_dotlist([
            f"model.variant={DYNAMIC_VARIANT_V11_6}",
        ]))
        self.assertIn(f"{DYNAMIC_VARIANT_V11_6}.yaml", source)

        left = OmegaConf.to_container(v11_5, resolve=False)
        right = OmegaConf.to_container(v11_6, resolve=False)
        left["model"]["variant"] = right["model"]["variant"]
        left_count = left["p2g"].pop("grid_query")
        right_count = right["p2g"].pop("grid_query")
        left.pop("data")
        right.pop("data")
        self.assertEqual(left, right)
        self.assertEqual(left_count, {**right_count, "K_max": 2})
        self.assertEqual(right_count["count_mode"], "legacy")
        self.assertEqual(right_count["K_max"], 3)
        self.assertEqual(right_count["seed_mode"], "range_quantile")

        self.assertEqual(
            OmegaConf.to_container(v11_6.data, resolve=True),
            OmegaConf.to_container(v11_4_dynamic.data, resolve=True),
        )
        self.assertEqual(v11_6.data.dataset_name, "nuscenes_dynamic_lidar4d")
        self.assertEqual(v11_6.data.mode, "nvs")
        self.assertIsNone(v11_6.data.train_excluded_scenes)
        self.assertEqual(
            list(v11_6.data.dynamic_window.train_excluded_scenes),
            ["scene-0757", "scene-1094"],
        )

        for group in (
            BARRIER_MATCH_DYNAMIC_VARIANTS,
        ):
            self.assertIn(DYNAMIC_VARIANT_V11_6, group)
        self.assertNotIn(DYNAMIC_VARIANT_V11_6, ROUTER_CAPABLE_DYNAMIC_VARIANTS)

        with self.assertRaises(ValueError) as raised:
            compose_fresh_config(OmegaConf.from_dotlist([
                f"model.variant={DYNAMIC_VARIANT_V11_6}",
                "p2g.grid_query.count_mode=learned_gumbel",
            ]))
        self.assertIn("V11.6", str(raised.exception))
        self.assertIn("count router", str(raised.exception))

    def test_v12_composes_to_v11_4_and_only_learns_the_speed(self):
        """The learnable speed is a backend change; the config is V11.4's."""
        v11_4, _source = compose_fresh_config(OmegaConf.from_dotlist([
            f"model.variant={DYNAMIC_VARIANT_V11_4}",
        ]))
        v12, source = compose_fresh_config(OmegaConf.from_dotlist([
            f"model.variant={DYNAMIC_VARIANT_V12}",
        ]))
        self.assertIn(f"{DYNAMIC_VARIANT_V12}.yaml", source)

        left = OmegaConf.to_container(v11_4, resolve=False)
        right = OmegaConf.to_container(v12, resolve=False)
        left["model"]["variant"] = right["model"]["variant"]
        self.assertEqual(left, right)

        right_data = right["data"]
        self.assertEqual(right_data["dataset_name"], "nuscenes")
        self.assertEqual(right_data["mode"], "bbox")
        self.assertIsNone(right_data["bbox_json_path"])
        self.assertNotIn("dynamic_window", right_data)
        self.assertEqual(
            list(right_data["train_excluded_scenes"]),
            ["scene-0757", "scene-1094"],
        )

        temporal = v12.dynamic_2dgs.temporal
        self.assertEqual(temporal.barrier_speed_mps, 30.0)
        self.assertEqual(temporal.barrier_weight, 1.0)
        self.assertEqual(temporal.barrier_exponent, 1)
        self.assertTrue(temporal.qk_norm)
        self.assertEqual(v12.p2g.grid_query.count_mode, "learned_gumbel")
        for group in (
            BARRIER_MATCH_DYNAMIC_VARIANTS,
            ROUTER_CAPABLE_DYNAMIC_VARIANTS,
        ):
            self.assertIn(DYNAMIC_VARIANT_V12, group)

    def test_v12_1_adds_lambda_and_pins_the_offset_bound(self):
        """The backend adds split gains and lambda; config adds its initial."""
        v12, _source = compose_fresh_config(OmegaConf.from_dotlist([
            f"model.variant={DYNAMIC_VARIANT_V12}",
        ]))
        v12_1, source = compose_fresh_config(OmegaConf.from_dotlist([
            f"model.variant={DYNAMIC_VARIANT_V12_1}",
        ]))
        self.assertIn(f"{DYNAMIC_VARIANT_V12_1}.yaml", source)

        left = OmegaConf.to_container(v12, resolve=False)
        right = OmegaConf.to_container(v12_1, resolve=False)
        left["model"]["variant"] = right["model"]["variant"]
        # V12 inherits the shared 0.4 m bound; V12.1 pins the same value.
        self.assertEqual(left["p2g"]["head_offset_bound"], 0.4)
        self.assertEqual(right["p2g"]["head_offset_bound"], 0.4)
        left_temporal = left["dynamic_2dgs"]["temporal"]
        right_temporal = right["dynamic_2dgs"]["temporal"]
        self.assertNotIn("barrier_lambda_init", left_temporal)
        self.assertEqual(right_temporal.pop("barrier_lambda_init"), 1.0)
        self.assertEqual(left, right)

        self.assertEqual(v12_1.data.dataset_name, "nuscenes")
        self.assertEqual(v12_1.data.mode, "bbox")
        self.assertIsNone(v12_1.data.bbox_json_path)
        self.assertNotIn("dynamic_window", v12_1.data)
        self.assertEqual(
            list(v12_1.data.train_excluded_scenes),
            ["scene-0757", "scene-1094"],
        )
        for group in (
            BARRIER_MATCH_DYNAMIC_VARIANTS,
            ROUTER_CAPABLE_DYNAMIC_VARIANTS,
        ):
            self.assertIn(DYNAMIC_VARIANT_V12_1, group)

        for value in (0.0, -1.0):
            with self.subTest(barrier_lambda_init=value), self.assertRaises(
                ValueError
            ):
                compose_fresh_config(OmegaConf.from_dotlist([
                    f"model.variant={DYNAMIC_VARIANT_V12_1}",
                    f"dynamic_2dgs.temporal.barrier_lambda_init={value}",
                ]))

        with self.assertRaisesRegex(ValueError, "duplicates"):
            compose_fresh_config(OmegaConf.from_dotlist([
                f"model.variant={DYNAMIC_VARIANT_V12_1}",
                "data.train_excluded_scenes=[scene-0757,scene-0757]",
            ]))

    def test_v12_2_is_v12_on_the_dynamic_loader(self):
        """Per-layer softplus speeds are a backend change; config is V12's."""
        # V12 opted into the loader V11.6 ships with, exactly as the V11.4
        # dynamic-loader run was launched.
        v12_dynamic, _source = compose_fresh_config(OmegaConf.from_dotlist([
            f"model.variant={DYNAMIC_VARIANT_V12}", *DYNAMIC_LOADER_CLI,
        ]))
        v11_6, _source = compose_fresh_config(OmegaConf.from_dotlist([
            f"model.variant={DYNAMIC_VARIANT_V11_6}",
        ]))
        v12_2, source = compose_fresh_config(OmegaConf.from_dotlist([
            f"model.variant={DYNAMIC_VARIANT_V12_2}",
        ]))
        self.assertIn(f"{DYNAMIC_VARIANT_V12_2}.yaml", source)

        left = OmegaConf.to_container(v12_dynamic, resolve=False)
        right = OmegaConf.to_container(v12_2, resolve=False)
        left["model"]["variant"] = right["model"]["variant"]
        # The overlay pins the shared 0.4 m bound V12 inherits.
        self.assertEqual(right["p2g"]["head_offset_bound"], 0.4)
        self.assertEqual(left, right)
        self.assertEqual(
            OmegaConf.to_container(v12_2.data, resolve=True),
            OmegaConf.to_container(v11_6.data, resolve=True),
        )
        self.assertEqual(v12_2.data.dataset_name, "nuscenes_dynamic_lidar4d")
        self.assertEqual(v12_2.data.mode, "nvs")
        self.assertIsNone(v12_2.data.train_excluded_scenes)

        temporal = v12_2.dynamic_2dgs.temporal
        self.assertEqual(temporal.layers, 12)
        self.assertEqual(temporal.barrier_speed_mps, 30.0)
        self.assertEqual(temporal.barrier_weight, 1.0)
        self.assertEqual(temporal.barrier_exponent, 1)
        self.assertEqual(v12_2.p2g.grid_query.count_mode, "learned_gumbel")
        for group in (
            BARRIER_MATCH_DYNAMIC_VARIANTS,
            ROUTER_CAPABLE_DYNAMIC_VARIANTS,
        ):
            self.assertIn(DYNAMIC_VARIANT_V12_2, group)

        # V12.2 has no learned lambda to initialize.
        with self.assertRaisesRegex(ValueError, "barrier_lambda_init"):
            compose_fresh_config(OmegaConf.from_dotlist([
                f"model.variant={DYNAMIC_VARIANT_V12_2}",
                "dynamic_2dgs.temporal.barrier_lambda_init=1.0",
            ]))

    def test_v14_is_the_v11_4_dynamic_run_with_hybrid_utonia(self):
        """Only Utonia's adaptation and the token anchors differ from V11.4.

        The V11.4 comparison run opted into the random-duration loader from
        the CLI; V14 ships that data block in its overlay.
        """
        v11_4_dynamic, _source = compose_fresh_config(OmegaConf.from_dotlist([
            f"model.variant={DYNAMIC_VARIANT_V11_4}", *DYNAMIC_LOADER_CLI,
        ]))
        v14, source = compose_fresh_config(OmegaConf.from_dotlist([
            f"model.variant={DYNAMIC_VARIANT_V14}",
        ]))
        self.assertIn(f"{DYNAMIC_VARIANT_V14}.yaml", source)

        left = OmegaConf.to_container(v11_4_dynamic, resolve=False)
        right = OmegaConf.to_container(v14, resolve=False)
        left["model"]["variant"] = right["model"]["variant"]
        adaptation = right["p2g"].pop("utonia_adaptation")
        groups = right["train"].pop("utonia_param_groups")
        self.assertEqual(right["train"].pop("adam_betas"), [0.9, 0.999])
        self.assertEqual(right["train"].pop("adam_eps"), 1.0e-8)
        self.assertEqual(
            right["p2g"]["grid_query"]["learned_count"].pop("seed_mode"),
            "token",
        )
        self.assertEqual(
            left["p2g"]["grid_query"]["learned_count"].pop("seed_mode"),
            "range_quantile",
        )
        self.assertEqual(left, right)

        self.assertEqual(adaptation["mode"], "hybrid")
        self.assertEqual(
            adaptation["lora_targets"], ["attention", "cpe", "pooling"]
        )
        self.assertTrue(adaptation["train_layer_norm"])
        self.assertEqual(adaptation["stem"], {
            "train_intensity_column": True,
            "adapter_hidden": 16,
            "adapter_activation": "gelu",
            "adapter_scale": 1.0,
        })
        self.assertEqual(adaptation["ffn_adapter"], {
            "widths": [8, 16, 32, 64],
            "activation": "relu",
            "scale": 0.1,
            "dropout": 0.0,
        })
        # Since 2026-10-08 every role follows train.lr and keeps its decay.
        self.assertTrue(all(role["lr"] == "${train.lr}" for role in groups.values()))
        self.assertEqual(OmegaConf.to_container(v14.train.utonia_param_groups, resolve=True), {
            "intensity": {"lr": 5.0e-4, "weight_decay": 1.0e-4},
            "adapter": {
                "lr": 5.0e-4, "weight_decay": 1.0e-4, "bias_weight_decay": 0.0,
            },
            "lora": {"lr": 5.0e-4, "weight_decay": 1.0e-4},
            "layer_norm": {"lr": 5.0e-4, "weight_decay": 0.0},
        })
        # The downstream group, warmup and clip are the shared base values.
        self.assertEqual(v14.train.lr, 5.0e-4)
        self.assertEqual(v14.train.weight_decay, 1.0e-4)
        self.assertEqual(v14.train.warmup_iters, 500)
        self.assertEqual(v14.train.grad_clip, 1.0)
        self.assertEqual(v14.p2g.utonia_lora.rank, 16)
        self.assertEqual(v14.p2g.utonia_lora.alpha, 16)
        self.assertEqual(v14.p2g.utonia_lora.dropout, 0.0)
        self.assertEqual(v14.data.coordinate_frame, "ego_vehicle")
        self.assertEqual(v14.data.dataset_name, "nuscenes_dynamic_lidar4d")
        temporal = v14.dynamic_2dgs.temporal
        self.assertEqual(temporal.barrier_speed_mps, 30.0)
        self.assertEqual(temporal.barrier_weight, 1.0)
        self.assertEqual(temporal.barrier_exponent, 1)
        for group in (
            BARRIER_MATCH_DYNAMIC_VARIANTS,
            ROUTER_CAPABLE_DYNAMIC_VARIANTS,
            HYBRID_UTONIA_DYNAMIC_VARIANTS,
        ):
            self.assertIn(DYNAMIC_VARIANT_V14, group)
            self.assertIn(DYNAMIC_VARIANT_V14_1, group)

    def test_v14_1_is_v14_with_its_speed_learned(self):
        """The global softplus speed is a backend change; config is V14's."""
        v14, _source = compose_fresh_config(OmegaConf.from_dotlist([
            f"model.variant={DYNAMIC_VARIANT_V14}",
        ]))
        v14_1, source = compose_fresh_config(OmegaConf.from_dotlist([
            f"model.variant={DYNAMIC_VARIANT_V14_1}",
        ]))
        self.assertIn(f"{DYNAMIC_VARIANT_V14_1}.yaml", source)
        left = OmegaConf.to_container(v14, resolve=False)
        right = OmegaConf.to_container(v14_1, resolve=False)
        left["model"]["variant"] = right["model"]["variant"]
        self.assertEqual(left, right)
        with self.assertRaisesRegex(ValueError, "barrier_lambda_init"):
            compose_fresh_config(OmegaConf.from_dotlist([
                f"model.variant={DYNAMIC_VARIANT_V14_1}",
                "dynamic_2dgs.temporal.barrier_lambda_init=1.0",
            ]))

    def test_hybrid_utonia_blocks_are_reserved_for_v14(self):
        for override, message in (
            ("p2g.utonia_adaptation.mode=hybrid", "only available to V14"),
            ("train.utonia_param_groups.lora.lr=1e-4", "only applies"),
        ):
            with self.subTest(override=override):
                with self.assertRaisesRegex(ValueError, message):
                    compose_fresh_config(OmegaConf.from_dotlist([
                        f"model.variant={DYNAMIC_VARIANT_V11_4}", override,
                    ]))
        with self.assertRaisesRegex(ValueError, "requires p2g.utonia_adaptation"):
            compose_fresh_config(OmegaConf.from_dotlist([
                f"model.variant={DYNAMIC_VARIANT_V14}",
                "p2g.utonia_adaptation=null",
            ]))
        with self.assertRaisesRegex(ValueError, "requires train.utonia_param"):
            compose_fresh_config(OmegaConf.from_dotlist([
                f"model.variant={DYNAMIC_VARIANT_V14}",
                "train.utonia_param_groups=null",
            ]))

    def test_hybrid_utonia_rejects_invalid_settings(self):
        cases = (
            ("p2g.utonia_adaptation.mode=lora", "utonia_adaptation.mode"),
            (
                "p2g.utonia_adaptation.lora_targets=[attention,ffn]",
                "mutually exclusive",
            ),
            ("p2g.utonia_adaptation.lora_targets=[attention,stem]", "stem"),
            ("p2g.utonia_adaptation.lora_targets=[cpe,cpe]", "repeat"),
            ("p2g.utonia_adaptation.lora_targets=[qkv]", "must be one of"),
            ("p2g.utonia_adaptation.train_layer_norm=1", "true or false"),
            ("p2g.utonia_adaptation.stem.adapter_hidden=0", "adapter_hidden"),
            (
                "p2g.utonia_adaptation.stem.adapter_activation=tanh",
                "adapter_activation",
            ),
            ("p2g.utonia_adaptation.stem.adapter_scale=0", "adapter_scale"),
            ("p2g.utonia_adaptation.ffn_adapter.widths=[8,16,32]", "widths"),
            ("p2g.utonia_adaptation.ffn_adapter.widths=[8,16,0,64]", "widths"),
            ("p2g.utonia_adaptation.ffn_adapter.scale=-0.1", "scale"),
            ("p2g.utonia_adaptation.ffn_adapter.dropout=1.0", "dropout"),
            ("p2g.utonia_adaptation.ffn_adapter.rank=4", "Unsupported keys"),
            ("train.utonia_param_groups.lora.lr=0", "lora.lr"),
            ("train.utonia_param_groups.adapter.lr=inf", "adapter.lr"),
            (
                "train.utonia_param_groups.layer_norm.weight_decay=-1",
                "layer_norm.weight_decay",
            ),
            (
                "train.utonia_param_groups.adapter.bias_weight_decay=null",
                "bias_weight_decay",
            ),
            ("train.utonia_param_groups.stem.lr=1e-3", "Unsupported keys"),
            ("train.adam_betas=[0.9]", "adam_betas"),
            ("train.adam_eps=0", "adam_eps"),
        )
        for override, message in cases:
            with self.subTest(override=override):
                with self.assertRaisesRegex(ValueError, message):
                    compose_fresh_config(OmegaConf.from_dotlist([
                        f"model.variant={DYNAMIC_VARIANT_V14}", override,
                    ]))

    def test_v14_checkpoint_restores_and_protects_its_utonia_adaptation(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "epoch=0.ckpt"
            stored, _ = compose_fresh_config(OmegaConf.from_dotlist([
                f"model.variant={DYNAMIC_VARIANT_V14_1}",
            ]))
            torch.save({
                "experiment_config": OmegaConf.to_container(stored, resolve=True)
            }, checkpoint)

            config, _source = load_checkpoint_config(checkpoint)
            self.assertEqual(config.p2g.utonia_adaptation.mode, "hybrid")
            # The checkpoint embeds the resolved ${train.lr}.
            self.assertEqual(
                config.train.utonia_param_groups.lora.lr, 5.0e-4
            )
            with self.assertRaisesRegex(ValueError, "changed roots: p2g"):
                resolve_main_config(OmegaConf.from_dotlist([
                    f"train.ckpt_path={checkpoint}",
                    "p2g.utonia_adaptation.ffn_adapter.scale=0.2",
                ]))

    def test_v13_uses_all_head_rope_cosine_and_adaptive_range_quantiles(self):
        config, source = compose_fresh_config(OmegaConf.from_dotlist([
            f"model.variant={DYNAMIC_VARIANT_V13}",
        ]))
        self.assertIn(f"{DYNAMIC_VARIANT_V13}.yaml", source)
        temporal = config.dynamic_2dgs.temporal
        self.assertEqual(temporal.position_encoding, "rope_all_heads")
        self.assertFalse(temporal.qk_norm)
        self.assertAlmostEqual(temporal.motion_temperature, 0.03)
        self.assertEqual(temporal.rope_base, 100.0)
        self.assertAlmostEqual(
            temporal.rope_position_scale, 2.0 * math.pi / 5.0
        )
        for removed in (
            "barrier_speed_mps", "barrier_weight", "barrier_exponent",
            "barrier_lambda_init", "match_chunk_size",
            "distance_bias_speed_mps", "layer_weight_hidden_dim",
        ):
            self.assertNotIn(removed, temporal)

        self.assertEqual(config.p2g.head_offset_bound, 0.4)
        count = config.p2g.grid_query
        self.assertEqual(count.count_mode, "learned_gumbel")
        self.assertEqual(count.learned_count.K_max, 3)
        self.assertEqual(count.learned_count.seed_mode, "range_quantile")
        self.assertEqual(config.data.dataset_name, "nuscenes")
        self.assertEqual(
            list(config.data.train_excluded_scenes),
            ["scene-0757", "scene-1094"],
        )
        for group in (
            COSINE_ROPE_MATCH_DYNAMIC_VARIANTS,
            ROUTER_CAPABLE_DYNAMIC_VARIANTS,
        ):
            self.assertIn(DYNAMIC_VARIANT_V13, group)
        self.assertNotIn(DYNAMIC_VARIANT_V13, BARRIER_MATCH_DYNAMIC_VARIANTS)

        for override, message in (
            ("dynamic_2dgs.temporal.qk_norm=true", "qk_norm=false"),
            ("dynamic_2dgs.temporal.motion_temperature=0", "motion_temperature"),
            (
                "dynamic_2dgs.temporal.position_encoding=barrier_rope_split",
                "rope_all_heads",
            ),
        ):
            with self.subTest(override=override), self.assertRaisesRegex(
                ValueError, message
            ):
                compose_fresh_config(OmegaConf.from_dotlist([
                    f"model.variant={DYNAMIC_VARIANT_V13}", override,
                ]))

    def test_v13_1_differs_from_v13_only_by_fixed_k2_count(self):
        v13, _source = compose_fresh_config(OmegaConf.from_dotlist([
            f"model.variant={DYNAMIC_VARIANT_V13}",
        ]))
        v13_1, source = compose_fresh_config(OmegaConf.from_dotlist([
            f"model.variant={DYNAMIC_VARIANT_V13_1}",
        ]))
        self.assertIn(f"{DYNAMIC_VARIANT_V13_1}.yaml", source)

        left = OmegaConf.to_container(v13, resolve=False)
        right = OmegaConf.to_container(v13_1, resolve=False)
        left["model"]["variant"] = right["model"]["variant"]
        left_count = left["p2g"].pop("grid_query")
        right_count = right["p2g"].pop("grid_query")
        self.assertEqual(left, right)
        self.assertEqual(left_count["count_mode"], "learned_gumbel")
        self.assertEqual(
            left_count["learned_count"]["seed_mode"], "range_quantile"
        )
        self.assertEqual(right_count["count_mode"], "legacy")
        self.assertEqual(right_count["K_max"], 2)
        self.assertEqual(right_count["seed_mode"], "range_quantile")
        self.assertEqual(v13_1.p2g.head_offset_bound, 0.4)

        for group in (
            COSINE_ROPE_MATCH_DYNAMIC_VARIANTS,
        ):
            self.assertIn(DYNAMIC_VARIANT_V13_1, group)
        self.assertNotIn(
            DYNAMIC_VARIANT_V13_1, ROUTER_CAPABLE_DYNAMIC_VARIANTS
        )
        self.assertNotIn(DYNAMIC_VARIANT_V13_1, BARRIER_MATCH_DYNAMIC_VARIANTS)

        with self.assertRaisesRegex(ValueError, "no count router"):
            compose_fresh_config(OmegaConf.from_dotlist([
                f"model.variant={DYNAMIC_VARIANT_V13_1}",
                "p2g.grid_query.count_mode=learned_gumbel",
                "p2g.grid_query.grad_balance=sqrt_k",
                "p2g.grid_query.learned_count.K_max=3",
                "p2g.grid_query.learned_count.tau=1.0",
                "p2g.grid_query.learned_count.seed_mode=range_quantile",
                "p2g.grid_query.learned_count.grad_balance_scope=token",
                "p2g.grid_query.learned_count.logging.enable=true",
                "p2g.grid_query.learned_count.logging.interval=20",
                "p2g.grid_query.learned_count.logging.range_edges_m=[0,10]",
            ]))

    def test_v13_2_changes_only_head0_from_cosine_to_raw_qk(self):
        v13, _source = compose_fresh_config(OmegaConf.from_dotlist([
            f"model.variant={DYNAMIC_VARIANT_V13}",
        ]))
        v13_2, source = compose_fresh_config(OmegaConf.from_dotlist([
            f"model.variant={DYNAMIC_VARIANT_V13_2}",
        ]))
        self.assertIn(f"{DYNAMIC_VARIANT_V13_2}.yaml", source)

        left = OmegaConf.to_container(v13, resolve=False)
        right = OmegaConf.to_container(v13_2, resolve=False)
        left["model"]["variant"] = right["model"]["variant"]
        self.assertEqual(
            left["dynamic_2dgs"]["temporal"].pop("motion_temperature"),
            0.03,
        )
        self.assertEqual(left, right)

        temporal = v13_2.dynamic_2dgs.temporal
        self.assertEqual(temporal.position_encoding, "rope_all_heads")
        self.assertFalse(temporal.qk_norm)
        self.assertNotIn("motion_temperature", temporal)
        self.assertEqual(v13_2.p2g.head_offset_bound, 0.4)
        count = v13_2.p2g.grid_query
        self.assertEqual(count.count_mode, "learned_gumbel")
        self.assertEqual(count.learned_count.seed_mode, "range_quantile")

        for group in (
            ALL_HEAD_ROPE_MATCH_DYNAMIC_VARIANTS,
            ROUTER_CAPABLE_DYNAMIC_VARIANTS,
        ):
            self.assertIn(DYNAMIC_VARIANT_V13_2, group)
        self.assertNotIn(
            DYNAMIC_VARIANT_V13_2, COSINE_ROPE_MATCH_DYNAMIC_VARIANTS
        )
        self.assertNotIn(DYNAMIC_VARIANT_V13_2, BARRIER_MATCH_DYNAMIC_VARIANTS)

        with self.assertRaisesRegex(ValueError, "motion_temperature"):
            compose_fresh_config(OmegaConf.from_dotlist([
                f"model.variant={DYNAMIC_VARIANT_V13_2}",
                "dynamic_2dgs.temporal.motion_temperature=0.03",
            ]))

    def test_v13_3_is_v11_4_with_all_head_rope_and_v13_2_data(self):
        v11_4, _source = compose_fresh_config(OmegaConf.from_dotlist([
            f"model.variant={DYNAMIC_VARIANT_V11_4}",
        ]))
        v13_2, _source = compose_fresh_config(OmegaConf.from_dotlist([
            f"model.variant={DYNAMIC_VARIANT_V13_2}",
        ]))
        v13_3, source = compose_fresh_config(OmegaConf.from_dotlist([
            f"model.variant={DYNAMIC_VARIANT_V13_3}",
        ]))
        self.assertIn(f"{DYNAMIC_VARIANT_V13_3}.yaml", source)

        temporal = v13_3.dynamic_2dgs.temporal
        self.assertEqual(temporal.position_encoding, "rope_all_heads")
        self.assertTrue(temporal.qk_norm)
        self.assertEqual(temporal.implementation, "flash_varlen")
        self.assertEqual(temporal.layers, 12)
        self.assertEqual(temporal.num_heads, 12)
        self.assertEqual(temporal.rope_base, 100.0)
        self.assertAlmostEqual(
            temporal.rope_position_scale, 2.0 * math.pi / 5.0
        )
        for removed in (
            "barrier_speed_mps", "barrier_weight", "barrier_exponent",
            "barrier_lambda_init", "match_chunk_size",
            "distance_bias_speed_mps", "layer_weight_hidden_dim",
            "motion_temperature",
        ):
            self.assertNotIn(removed, temporal)

        # Everything outside the requested score/data/offset changes remains
        # on V11.4's adaptive correspondence-initialized motion path.
        self.assertEqual(v13_3.p2g.grid_query, v11_4.p2g.grid_query)
        self.assertEqual(
            v13_3.dynamic_2dgs.gaussian_head,
            v11_4.dynamic_2dgs.gaussian_head,
        )
        self.assertEqual(
            v13_3.dynamic_2dgs.motion, v11_4.dynamic_2dgs.motion
        )
        self.assertEqual(
            v13_3.dynamic_2dgs.regularization,
            v11_4.dynamic_2dgs.regularization,
        )
        self.assertEqual(v13_3.loss, v11_4.loss)

        self.assertEqual(v13_3.p2g.head_offset_bound, 0.4)
        self.assertEqual(
            OmegaConf.to_container(v13_3.data, resolve=True),
            OmegaConf.to_container(v13_2.data, resolve=True),
        )
        self.assertEqual(v13_3.data.dataset_name, "nuscenes")
        self.assertEqual(v13_3.data.mode, "bbox")
        self.assertIsNone(v13_3.data.bbox_json_path)
        self.assertEqual(
            list(v13_3.data.train_excluded_scenes),
            ["scene-0757", "scene-1094"],
        )

        for group in (
            ALL_HEAD_ROPE_MATCH_DYNAMIC_VARIANTS,
            RMSNORM_ROPE_MATCH_DYNAMIC_VARIANTS,
            ROUTER_CAPABLE_DYNAMIC_VARIANTS,
        ):
            self.assertIn(DYNAMIC_VARIANT_V13_3, group)
        self.assertNotIn(
            DYNAMIC_VARIANT_V13_3, COSINE_ROPE_MATCH_DYNAMIC_VARIANTS
        )
        self.assertNotIn(DYNAMIC_VARIANT_V13_3, BARRIER_MATCH_DYNAMIC_VARIANTS)

        for override, message in (
            ("dynamic_2dgs.temporal.qk_norm=false", "qk_norm=true"),
            (
                "dynamic_2dgs.temporal.position_encoding=barrier_rope_split",
                "rope_all_heads",
            ),
        ):
            with self.subTest(override=override), self.assertRaisesRegex(
                ValueError, message
            ):
                compose_fresh_config(OmegaConf.from_dotlist([
                    f"model.variant={DYNAMIC_VARIANT_V13_3}", override,
                ]))

    def test_v13_4_differs_from_v13_3_only_by_head0_rope(self):
        v13_3, _source = compose_fresh_config(OmegaConf.from_dotlist([
            f"model.variant={DYNAMIC_VARIANT_V13_3}",
        ]))
        v13_4, source = compose_fresh_config(OmegaConf.from_dotlist([
            f"model.variant={DYNAMIC_VARIANT_V13_4}",
        ]))
        self.assertIn(f"{DYNAMIC_VARIANT_V13_4}.yaml", source)

        left = OmegaConf.to_container(v13_3, resolve=False)
        right = OmegaConf.to_container(v13_4, resolve=False)
        left["model"]["variant"] = right["model"]["variant"]
        left["dynamic_2dgs"]["temporal"]["position_encoding"] = (
            right["dynamic_2dgs"]["temporal"]["position_encoding"]
        )
        self.assertEqual(left, right)

        temporal = v13_4.dynamic_2dgs.temporal
        self.assertEqual(temporal.position_encoding, "rope_except_head0")
        self.assertTrue(temporal.qk_norm)
        self.assertEqual(temporal.implementation, "flash_varlen")
        for removed in (
            "barrier_speed_mps", "barrier_weight", "barrier_exponent",
            "barrier_lambda_init", "match_chunk_size",
            "distance_bias_speed_mps", "layer_weight_hidden_dim",
            "motion_temperature",
        ):
            self.assertNotIn(removed, temporal)

        self.assertEqual(v13_4.p2g.head_offset_bound, 0.4)
        self.assertEqual(v13_4.data, v13_3.data)
        self.assertEqual(
            list(v13_4.data.train_excluded_scenes),
            ["scene-0757", "scene-1094"],
        )
        for group in (
            FEATURE_HEAD_ROPE_MATCH_DYNAMIC_VARIANTS,
            FLASH_ROPE_MATCH_DYNAMIC_VARIANTS,
            RMSNORM_ROPE_MATCH_DYNAMIC_VARIANTS,
            ROUTER_CAPABLE_DYNAMIC_VARIANTS,
        ):
            self.assertIn(DYNAMIC_VARIANT_V13_4, group)
        for group in (
            ALL_HEAD_ROPE_MATCH_DYNAMIC_VARIANTS,
            COSINE_ROPE_MATCH_DYNAMIC_VARIANTS,
            BARRIER_MATCH_DYNAMIC_VARIANTS,
        ):
            self.assertNotIn(DYNAMIC_VARIANT_V13_4, group)

        for override, message in (
            ("dynamic_2dgs.temporal.qk_norm=false", "qk_norm=true"),
            (
                "dynamic_2dgs.temporal.position_encoding=rope_all_heads",
                "rope_except_head0",
            ),
        ):
            with self.subTest(override=override), self.assertRaisesRegex(
                ValueError, message
            ):
                compose_fresh_config(OmegaConf.from_dotlist([
                    f"model.variant={DYNAMIC_VARIANT_V13_4}", override,
                ]))

    def test_v11_3_head_is_seed_conditioned_at_three_channels_per_slot(self):
        """The router trunk's geometry input survives as head input columns."""
        from src.models.module.dynamic.heads import GaussianAttributeHead

        gs_params = OmegaConf.create(
            {"shs": 32, "opacity": 1, "scaling": 2, "rotation": 4, "offset": 3}
        )
        head = GaussianAttributeHead(
            None, gs_params, dim=64, offset_bound=0.8,
            gaussians_per_token=2, seed_conditioned=True, trunk=True,
        )
        self.assertEqual(head.seed_dim, 6)
        # Two layers with the one nonlinearity, matching the router's decoder:
        # the seed deltas enter the trunk, the attributes read its output.
        self.assertEqual(head.trunk[0].in_features, 64 + 6)
        self.assertEqual(head.trunk[0].out_features, 64)
        self.assertIsInstance(head.trunk[1], torch.nn.SiLU)
        for name, width in head.sizes.items():
            self.assertEqual(head.heads[name].in_features, 64)
            self.assertEqual(head.heads[name].out_features, width * 2)

        feature = torch.randn(5, 64)
        seed_delta = torch.randn(5, 2, 3)
        raw, offset, decoded = head(feature, seed_delta)
        self.assertEqual(decoded.shape, (5, 64))
        self.assertEqual(offset.shape, (10, 3))
        # Zero-initialized offset rows keep a fresh run on its own seeds.
        self.assertTrue(torch.equal(offset, torch.zeros_like(offset)))
        # The deltas reach the attributes, so the two slots are not twins.
        other, _offset, _decoded = head(feature, seed_delta + 1.0)
        self.assertFalse(torch.allclose(raw["shs"], other["shs"]))

        with self.assertRaisesRegex(ValueError, "requires seed deltas"):
            head(feature, None)
        with self.assertRaisesRegex(ValueError, "seed deltas must have shape"):
            head(feature, torch.randn(5, 3, 3))

    def test_default_gaussian_head_keeps_its_feature_only_input(self):
        """Widening V11.3's head must not touch the shared V3-V11.1 shapes."""
        from src.models.module.dynamic.heads import GaussianAttributeHead

        gs_params = OmegaConf.create(
            {"shs": 32, "opacity": 1, "scaling": 2, "rotation": 4, "offset": 3}
        )
        head = GaussianAttributeHead(None, gs_params, dim=64, offset_bound=0.8)
        self.assertFalse(head.seed_conditioned)
        self.assertEqual(head.seed_dim, 0)
        self.assertIsNone(head.trunk)
        for name in head.sizes:
            self.assertEqual(head.heads[name].in_features, 64)
        raw, offset, _decoded = head(torch.randn(5, 64))
        self.assertEqual(offset.shape, (5, 3))
        self.assertEqual(raw["shs"].shape, (5, 32))

    def test_v11_3_rejects_the_router_it_was_defined_to_drop(self):
        with self.assertRaises(ValueError) as raised:
            compose_fresh_config(OmegaConf.from_dotlist([
                f"model.variant={DYNAMIC_VARIANT_V11_3}",
                "p2g.grid_query.count_mode=learned_gumbel",
            ]))
        self.assertIn("V11.3", str(raised.exception))
        self.assertIn("count router", str(raised.exception))

    def test_v11_1_rejects_a_router_its_backend_cannot_run(self):
        """V11.1 has no count router, so asking for one is an error."""
        with self.assertRaises(ValueError) as raised:
            compose_fresh_config(OmegaConf.from_dotlist([
                f"model.variant={DYNAMIC_VARIANT_V11_1}",
                "p2g.grid_query.count_mode=learned_gumbel",
            ]))
        self.assertIn("grid_query", str(raised.exception))
        self.assertIn("V11.1", str(raised.exception))

    def test_grid_query_sets_the_per_token_gaussian_count(self):
        """K_max is how many Gaussians each token becomes; seed_mode seeds them."""
        for variant in (
            DYNAMIC_VARIANT_V11_1, DYNAMIC_VARIANT_V11_3, DYNAMIC_VARIANT_V11_5,
        ):
            for k_max, seed_mode in (
                (1, "medoid"), (2, "medoid"), (1, "token"), (3, "token"),
                (4, "range_quantile"),
            ):
                with self.subTest(
                    variant=variant, K_max=k_max, seed_mode=seed_mode
                ):
                    config, _source = compose_fresh_config(
                        OmegaConf.from_dotlist([
                            f"model.variant={variant}",
                            f"p2g.grid_query.K_max={k_max}",
                            f"p2g.grid_query.seed_mode={seed_mode}",
                        ])
                    )
                    self.assertEqual(config.p2g.grid_query.K_max, k_max)
                    self.assertEqual(config.p2g.grid_query.seed_mode, seed_mode)

    def test_absent_grid_query_is_one_gaussian_per_token(self):
        config, _source = compose_fresh_config(OmegaConf.from_dotlist([
            f"model.variant={DYNAMIC_VARIANT_V11_1}",
        ]))
        self.assertIsNone(OmegaConf.select(config, "p2g.grid_query"))

    def test_fixed_gaussian_count_rejects_impossible_settings(self):
        invalid = (
            ("p2g.grid_query.K_max=0", "positive"),
            (
                "p2g.grid_query.K_max=2 p2g.grid_query.exp=1",
                "renamed to p2g.grid_query.seed_mode",
            ),
            (
                "p2g.grid_query.K_max=2 p2g.grid_query.seed_mode=quantile",
                "must be one of",
            ),
            (
                "p2g.grid_query.K_max=2 p2g.grid_query.points_per_gaussian=2",
                "Unsupported keys",
            ),
            ("p2g.grid_query.K_max=2 p2g.grid_query.grad_balance=sqrt_k",
             "Unsupported keys"),
        )
        for override, message in invalid:
            with self.subTest(override=override), self.assertRaisesRegex(
                ValueError, message
            ):
                compose_fresh_config(OmegaConf.from_dotlist([
                    f"model.variant={DYNAMIC_VARIANT_V11_1}",
                    *override.split(),
                ]))

    def test_router_seed_mode_accepts_the_token_anchor_only(self):
        """The learned-count bank has range-quantile and token rows, no medoid."""
        config, _source = compose_fresh_config(OmegaConf.from_dotlist([
            f"model.variant={DYNAMIC_VARIANT_V11_4}",
            "p2g.grid_query.learned_count.seed_mode=token",
        ]))
        self.assertEqual(config.p2g.grid_query.learned_count.seed_mode, "token")
        with self.assertRaisesRegex(ValueError, "must be one of"):
            compose_fresh_config(OmegaConf.from_dotlist([
                f"model.variant={DYNAMIC_VARIANT_V11_4}",
                "p2g.grid_query.learned_count.seed_mode=medoid",
            ]))

    def test_no_barrier_variant_accepts_the_removed_gate_keys(self):
        """The gate is gone from every barrier variant, not just V11."""
        for variant in BARRIER_MATCH_DYNAMIC_VARIANTS:
            for key in (
                "support_rho_m", "support_gate_lo", "support_gate_width",
            ):
                with self.subTest(variant=variant, key=key):
                    with self.assertRaises(ValueError):
                        compose_fresh_config(OmegaConf.from_dotlist([
                            f"model.variant={variant}",
                            f"dynamic_2dgs.temporal.{key}=1.0",
                        ]))

    def test_v11_1_validates_the_barrier_geometry(self):
        for override, message in (
            ("dynamic_2dgs.temporal.position_encoding=distance_bias",
             "barrier_rope_split"),
            ("dynamic_2dgs.temporal.num_heads=1", "num_heads"),
            ("dynamic_2dgs.temporal.barrier_speed_mps=0.0",
             "barrier_speed_mps"),
            ("p2g.utonia_lora.enable=false", "utonia_lora"),
        ):
            with self.assertRaises(ValueError) as raised:
                compose_fresh_config(OmegaConf.from_dotlist([
                    f"model.variant={DYNAMIC_VARIANT_V11_1}", override,
                ]))
            self.assertIn(message, str(raised.exception))

    def test_barrier_variant_errors_name_the_variant_they_came_from(self):
        with self.assertRaises(ValueError) as raised:
            compose_fresh_config(OmegaConf.from_dotlist([
                f"model.variant={DYNAMIC_VARIANT_V11_1}",
                "dynamic_2dgs.temporal.qk_norm=false",
            ]))
        self.assertIn("V11.1", str(raised.exception))
        with self.assertRaises(ValueError) as raised:
            compose_fresh_config(OmegaConf.from_dotlist([
                f"model.variant={DYNAMIC_VARIANT_V11}",
                "dynamic_2dgs.temporal.qk_norm=false",
            ]))
        self.assertIn("V11 ", str(raised.exception))

    def test_v11_composes_split_position_encoding_contract(self):
        config, source = compose_fresh_config(OmegaConf.from_dotlist([
            f"model.variant={DYNAMIC_VARIANT_V11}",
        ]))
        temporal = config.dynamic_2dgs.temporal
        self.assertEqual(config.model.variant, DYNAMIC_VARIANT_V11)
        self.assertEqual(temporal.layers, 12)
        self.assertEqual(temporal.num_heads, 12)
        self.assertTrue(temporal.use_time_embedding)
        self.assertEqual(temporal.position_encoding, "barrier_rope_split")
        self.assertEqual(temporal.barrier_speed_mps, 30.0)
        self.assertEqual(temporal.barrier_weight, 4.0)
        self.assertGreater(temporal.match_chunk_size, 0)
        self.assertEqual(temporal.rope_base, 100.0)
        self.assertAlmostEqual(
            2.0 * math.pi / temporal.rope_position_scale, 5.0
        )
        self.assertTrue(temporal.qk_norm)
        # V10's global-token scorer and quadratic prior are both gone.
        for removed in (
            "layer_weight_hidden_dim", "distance_bias_speed_mps",
            "motion_head_count", "tie_motion_qk_init",
        ):
            self.assertNotIn(removed, temporal)
        self.assertNotIn("motion_proposal", config.dynamic_2dgs)
        self.assertTrue(config.p2g.utonia_lora.enable)
        self.assertEqual(config.p2g.utonia_lora.input_mode, "xyzi")
        self.assertEqual(config.p2g.utonia_lora.rank, 16)
        count = config.p2g.grid_query
        self.assertEqual(count.count_mode, "learned_gumbel")
        self.assertEqual(count.learned_count.K_max, 3)
        self.assertEqual(count.learned_count.seed_mode, "range_quantile")
        motion = config.dynamic_2dgs.motion
        self.assertEqual(motion.residual_hidden_dim, 96)
        self.assertEqual(motion.velocity_embedding_dim, 32)
        self.assertTrue(motion.detach_init_condition)
        prior = config.dynamic_2dgs.regularization.velocity_l2
        self.assertEqual(prior.mode, "split_group_l2")
        self.assertEqual(prior.init_weight, 0.01)
        self.assertEqual(prior.offset_weight, 0.01)
        self.assertNotIn("weight", prior)
        self.assertEqual(config.loss.w_chamfer, 0.02)
        self.assertEqual(config.data.dataset_name, "nuscenes")
        self.assertEqual(config.data.mode, "bbox")
        self.assertNotIn("dynamic_window", config.data)
        self.assertEqual(
            list(config.data.train_excluded_scenes),
            ["scene-0757", "scene-1094"],
        )
        self.assertEqual(config.data.gt_middle_count, 3)
        self.assertIn(f"{DYNAMIC_VARIANT_V11}.yaml", source)

    def test_every_v11_family_config_uses_the_legacy_loader_with_holdouts(self):
        for variant in (
            DYNAMIC_VARIANT_V11,
            DYNAMIC_VARIANT_V11_1,
            DYNAMIC_VARIANT_V11_2,
            DYNAMIC_VARIANT_V11_3,
            DYNAMIC_VARIANT_V11_4,
        ):
            with self.subTest(variant=variant):
                config, _source = compose_fresh_config(
                    OmegaConf.from_dotlist([f"model.variant={variant}"])
                )
                self.assertEqual(config.data.dataset_name, "nuscenes")
                self.assertEqual(config.data.pair_mode, "keyframe")
                self.assertEqual(config.data.sample_gap_us, 100_000)
                self.assertEqual(config.data.gt_middle_count, 3)
                self.assertEqual(config.data.mode, "bbox")
                self.assertIsNone(config.data.bbox_json_path)
                self.assertNotIn("dynamic_window", config.data)
                self.assertEqual(
                    list(config.data.train_excluded_scenes),
                    ["scene-0757", "scene-1094"],
                )

    def test_the_dynamic_loader_is_still_selectable_from_the_cli(self):
        config, _source = compose_fresh_config(OmegaConf.from_dotlist([
            f"model.variant={DYNAMIC_VARIANT_V11_4}", *DYNAMIC_LOADER_CLI,
        ]))
        self.assertEqual(config.data.dataset_name, "nuscenes_dynamic_lidar4d")
        self.assertEqual(config.data.mode, "nvs")
        self.assertIsNone(config.data.train_excluded_scenes)
        self.assertAlmostEqual(config.data.dynamic_window.endpoint_min_sec, 0.2)
        self.assertAlmostEqual(config.data.dynamic_window.endpoint_max_sec, 1.2)
        self.assertEqual(
            list(config.data.dynamic_window.train_excluded_scenes),
            ["scene-0757", "scene-1094"],
        )

    def test_v11_dynamic_loader_rejects_invalid_sampling_contracts(self):
        invalid = (
            (
                "data.dynamic_window.endpoint_min_sec=1.3",
                "0 < min <= max",
            ),
            ("data.dynamic_window.endpoint_max_sec=0", "0 < min <= max"),
            ("data.sample_gap_us=75000", "multiple of 50000"),
            ("data.pair_mode=sweep", "pair_mode=keyframe"),
            ("data.mode=bbox", "mode=nvs"),
        )
        for override, message in invalid:
            with self.subTest(override=override), self.assertRaisesRegex(
                ValueError, message
            ):
                compose_fresh_config(OmegaConf.from_dotlist([
                    f"model.variant={DYNAMIC_VARIANT_V11}",
                    *DYNAMIC_LOADER_CLI,
                    override,
                ]))

    def test_v11_validates_barrier_and_rope_contract(self):
        invalid = (
            ("dynamic_2dgs.temporal.position_encoding=distance_bias",
             "position_encoding"),
            ("dynamic_2dgs.temporal.barrier_speed_mps=0", "barrier_speed_mps"),
            ("dynamic_2dgs.temporal.barrier_weight=-1", "barrier_weight"),
            ("dynamic_2dgs.temporal.match_chunk_size=0", "match_chunk_size"),
            ("dynamic_2dgs.temporal.qk_norm=false", "qk_norm"),
            ("dynamic_2dgs.temporal.use_time_embedding=false",
             "use_time_embedding"),
            ("p2g.utonia_lora.enable=false", "utonia_lora.enable=true"),
            ("p2g.grid_query.learned_count.K_max=4", "K_max=3"),
            ("p2g.grid_query.learned_count.budget.enable=true", "budget"),
        )
        for override, message in invalid:
            with self.subTest(override=override), self.assertRaisesRegex(
                ValueError, message
            ):
                compose_fresh_config(OmegaConf.from_dotlist([
                    f"model.variant={DYNAMIC_VARIANT_V11}", override,
                ]))

    def test_v11_still_requires_a_rope_capable_head_width(self):
        # Heads 1-11 keep 3D RoPE, so 432/16=27 is rejected where V10 allows it.
        with self.assertRaisesRegex(ValueError, "divisible by six"):
            compose_fresh_config(OmegaConf.from_dotlist([
                f"model.variant={DYNAMIC_VARIANT_V11}",
                "dynamic_2dgs.temporal.num_heads=16",
            ]))

    def test_v11_split_prior_requires_both_weights(self):
        for override in (
            "dynamic_2dgs.regularization.velocity_l2.init_weight=-1",
            "dynamic_2dgs.regularization.velocity_l2.offset_weight=-1",
        ):
            with self.subTest(override=override), self.assertRaisesRegex(
                ValueError, "split_group_l2"
            ):
                compose_fresh_config(OmegaConf.from_dotlist([
                    f"model.variant={DYNAMIC_VARIANT_V11}", override,
                ]))

    def test_velocity_prior_accepts_only_the_split_mode(self):
        for mode in ("final_l2", "final_group_l2", "offset_l2", "component_group_l1"):
            with self.subTest(mode=mode), self.assertRaisesRegex(
                ValueError, "split_group_l2"
            ):
                compose_fresh_config(OmegaConf.from_dotlist([
                    f"model.variant={DYNAMIC_VARIANT_V11}",
                    f"dynamic_2dgs.regularization.velocity_l2.mode={mode}",
                ]))
        with self.assertRaisesRegex(ValueError, "Unsupported keys"):
            compose_fresh_config(OmegaConf.from_dotlist([
                "dynamic_2dgs.regularization.velocity_l2.weight=0.01",
            ]))

    def test_v11_rejects_v10_only_temporal_keys(self):
        with self.assertRaisesRegex(ValueError, "Unsupported keys"):
            compose_fresh_config(OmegaConf.from_dotlist([
                f"model.variant={DYNAMIC_VARIANT_V11}",
                "dynamic_2dgs.temporal.layer_weight_hidden_dim=96",
            ]))

    def test_velocity_l2_rejects_unknown_keys(self):
        with self.assertRaisesRegex(ValueError, "Unsupported keys"):
            compose_fresh_config(OmegaConf.from_dotlist([
                "dynamic_2dgs.regularization.velocity_l2.epsilon_m=0.1",
            ]))

    def test_current_dynamic_rejects_unimplemented_semantic_keys(self):
        with self.assertRaisesRegex(ValueError, "Unsupported keys"):
            compose_fresh_config(OmegaConf.from_dotlist([
                f"model.variant={DYNAMIC_VARIANT}",
                "dynamic_2dgs.motion.deform_rotation=true",
            ]))

    def test_unknown_variant_fails(self):
        with self.assertRaisesRegex(ValueError, "Unknown model.variant"):
            compose_fresh_config(
                OmegaConf.from_dotlist(["model.variant=does_not_exist"])
            )

    def _legacy_checkpoint(self, root: Path) -> Path:
        """A checkpoint whose only config is its run-local W&B export."""
        checkpoint = root / "run" / "epoch=1.ckpt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.touch()

        fresh, _ = compose_fresh_config(
            OmegaConf.from_dotlist([f"model.variant={DYNAMIC_VARIANT}"])
        )
        stored = OmegaConf.to_container(fresh, resolve=False)
        # W&B omits empty mappings.
        stored.pop("g2g", None)
        wrapped = {key: {"value": value} for key, value in stored.items()}
        config_file = checkpoint.parent / "wandb/latest-run/files/config.yaml"
        config_file.parent.mkdir(parents=True)
        OmegaConf.save(OmegaConf.create(wrapped), config_file)
        return checkpoint

    def test_old_wandb_config_recovers_structural_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = self._legacy_checkpoint(Path(tmp))
            config, source = load_checkpoint_config(
                checkpoint,
                allow_legacy_wandb_fallback=True,
            )
            self.assertEqual(config.model.variant, DYNAMIC_VARIANT)
            self.assertIn("g2g", config)
            self.assertEqual(len(config.g2g), 0)
            self.assertIn("wandb/latest-run/files/config.yaml", source)

    def test_checkpoint_variant_cannot_be_changed_by_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = self._legacy_checkpoint(Path(tmp))
            cli = OmegaConf.from_dotlist([
                f"test.ckpt_path={checkpoint}",
                "allow_legacy_wandb_fallback=true",
                f"model.variant={DYNAMIC_VARIANT_V11}",
            ])
            with self.assertRaisesRegex(ValueError, "cannot be changed by CLI"):
                resolve_eval_config(cli)

    def test_checkpoint_structural_subtree_cannot_be_changed_by_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = self._legacy_checkpoint(Path(tmp))
            cli = OmegaConf.from_dotlist([
                f"test.ckpt_path={checkpoint}",
                "allow_legacy_wandb_fallback=true",
                "p2g.head_offset_bound=0.8",
            ])
            with self.assertRaisesRegex(ValueError, "changed roots: p2g"):
                resolve_eval_config(cli)

    def test_checkpoint_data_geometry_cannot_be_changed_by_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = self._legacy_checkpoint(Path(tmp))
            cli = OmegaConf.from_dotlist([
                f"test.ckpt_path={checkpoint}",
                "allow_legacy_wandb_fallback=true",
                "data.window_us=2000000",
            ])
            with self.assertRaisesRegex(ValueError, "paths: data.window_us"):
                resolve_eval_config(cli)

    def test_checkpoint_bbox_source_alias_cannot_be_added_by_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = self._legacy_checkpoint(Path(tmp))
            cli = OmegaConf.from_dotlist([
                f"test.ckpt_path={checkpoint}",
                "allow_legacy_wandb_fallback=true",
                "data.bbox_json_paths.val=/other/tracking.json",
            ])
            with self.assertRaisesRegex(ValueError, "data.bbox_json_paths"):
                resolve_eval_config(cli)

    def test_legacy_wandb_fallback_requires_explicit_opt_in(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = self._legacy_checkpoint(Path(tmp))
            cli = OmegaConf.from_dotlist([f"test.ckpt_path={checkpoint}"])
            with self.assertRaisesRegex(RuntimeError, "opt in explicitly"):
                resolve_eval_config(cli)

    def test_embedded_config_wins_over_mutable_wandb_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = self._legacy_checkpoint(Path(tmp))
            embedded, _ = compose_fresh_config(
                OmegaConf.from_dotlist(["train.batch_size=9"])
            )
            torch.save(
                {
                    "experiment_config": OmegaConf.to_container(
                        embedded,
                        resolve=True,
                    )
                },
                checkpoint,
            )
            config, source = load_checkpoint_config(checkpoint)
            self.assertEqual(config.train.batch_size, 9)
            self.assertIn("embedded checkpoint config", source)

    def test_historical_v11_checkpoint_keeps_its_unfiltered_train_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "epoch=1.ckpt"
            stored, _ = compose_fresh_config(OmegaConf.from_dotlist([
                f"model.variant={DYNAMIC_VARIANT_V11}",
            ]))
            stored = OmegaConf.to_container(stored, resolve=True)
            # Saved before the overlay gained scene holdouts.
            stored["data"].pop("train_excluded_scenes")
            torch.save({"experiment_config": stored}, checkpoint)

            config, _source = load_checkpoint_config(checkpoint)

            self.assertEqual(config.data.dataset_name, "nuscenes")
            self.assertNotIn("train_excluded_scenes", config.data)
            self.assertNotIn("dynamic_window", config.data)

    def test_checkpoint_exp_key_restores_as_seed_mode(self):
        """Checkpoints saved before the rename keep their placement rule."""
        for exp, seed_mode in ((None, "range_quantile"), (1, "medoid"), (2, "token")):
            with self.subTest(exp=exp), tempfile.TemporaryDirectory() as tmp:
                checkpoint = Path(tmp) / "epoch=1.ckpt"
                stored, _ = compose_fresh_config(OmegaConf.from_dotlist([
                    f"model.variant={DYNAMIC_VARIANT_V11_6}",
                ]))
                stored = OmegaConf.to_container(stored, resolve=True)
                stored["p2g"]["grid_query"].pop("seed_mode")
                stored["p2g"]["grid_query"]["exp"] = exp
                torch.save({"experiment_config": stored}, checkpoint)

                config, _source = load_checkpoint_config(checkpoint)

                self.assertNotIn("exp", config.p2g.grid_query)
                self.assertEqual(config.p2g.grid_query.seed_mode, seed_mode)
                self.assertEqual(config.p2g.grid_query.K_max, 3)

    def test_checkpoint_with_both_exp_and_seed_mode_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "epoch=1.ckpt"
            stored, _ = compose_fresh_config(OmegaConf.from_dotlist([
                f"model.variant={DYNAMIC_VARIANT_V11_6}",
            ]))
            stored = OmegaConf.to_container(stored, resolve=True)
            stored["p2g"]["grid_query"]["exp"] = None
            torch.save({"experiment_config": stored}, checkpoint)
            with self.assertRaisesRegex(ValueError, "both the retired exp"):
                load_checkpoint_config(checkpoint)

    def test_dynamic_loader_sampling_contract_is_immutable_on_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "epoch=1.ckpt"
            stored, _ = compose_fresh_config(OmegaConf.from_dotlist([
                f"model.variant={DYNAMIC_VARIANT_V11}", *DYNAMIC_LOADER_CLI,
            ]))
            torch.save(
                {"experiment_config": OmegaConf.to_container(stored, resolve=True)},
                checkpoint,
            )
            config, _source = load_checkpoint_config(checkpoint)
            self.assertEqual(config.data.dataset_name, "nuscenes_dynamic_lidar4d")
            cli = OmegaConf.from_dotlist([
                f"train.ckpt_path={checkpoint}",
                "data.dynamic_window.endpoint_max_sec=0.8",
            ])
            with self.assertRaisesRegex(ValueError, "data.dynamic_window"):
                resolve_main_config(cli)

    def test_legacy_train_scene_holdouts_are_immutable_on_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "epoch=1.ckpt"
            stored, _ = compose_fresh_config(OmegaConf.from_dotlist([
                f"model.variant={DYNAMIC_VARIANT_V12_1}",
            ]))
            torch.save(
                {"experiment_config": OmegaConf.to_container(stored, resolve=True)},
                checkpoint,
            )
            cli = OmegaConf.from_dotlist([
                f"train.ckpt_path={checkpoint}",
                "data.train_excluded_scenes=[scene-0757]",
            ])
            with self.assertRaisesRegex(
                ValueError, "data.train_excluded_scenes"
            ):
                resolve_main_config(cli)

    def test_pre_lora_checkpoint_config_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "epoch=1.ckpt"
            stored, _ = compose_fresh_config()
            stored = OmegaConf.to_container(stored, resolve=True)
            stored["p2g"].pop("utonia_lora")
            torch.save({"experiment_config": stored}, checkpoint)

            with self.assertRaisesRegex(ValueError, "utonia_lora.enable=true"):
                load_checkpoint_config(checkpoint)

    @staticmethod
    def _save_with_retired_blocks(root: Path, **overrides) -> Path:
        """A V11-era embedded config that still carries the removed blocks."""
        checkpoint = root / "epoch=1.ckpt"
        stored, _ = compose_fresh_config(OmegaConf.from_dotlist([
            f"model.variant={DYNAMIC_VARIANT_V11_4}",
        ]))
        stored = OmegaConf.to_container(stored, resolve=True)
        stored["p2g"].update({
            "anchor_mode": "grid",
            "r_far": 80.0,
            "agg_mlp": {"hidden_dim": 1024, "hidden_layers": 2, "out_dim": 576},
            "joint_refiner": {"enable": True, "type": "sparse", "num_heads": 12},
            "intensity_encoder": {"type": "ptv3"},
            "int_proj": {"in_dim": 5, "out_dim": 64},
            "intensity_sparse": {"enable": True},
            "utonia_adapter": {"bottleneck": None},
        })
        stored["p2g"]["grid_query"]["learned_count"]["budget"] = {
            "enable": overrides.get("budget", False),
        }
        stored["dynamic_2dgs"]["regularization"]["small_motion"] = {
            "enabled": overrides.get("small_motion", False),
            "weight": 0.005, "epsilon_m": 0.001,
            "warmup_steps": 300, "ramp_steps": 1700,
        }
        stored["loss"]["w_scale"] = overrides.get("w_scale", 0.0)
        stored["loss"]["enable_lpips"] = overrides.get("enable_lpips", False)
        stored["g2p"]["dynamic"] = overrides.get("g2p_dynamic", False)
        torch.save({"experiment_config": stored}, checkpoint)
        return checkpoint

    def test_v11_checkpoint_with_disabled_retired_blocks_still_loads(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = self._save_with_retired_blocks(Path(tmp))
            config, source = load_checkpoint_config(checkpoint)
            self.assertIn("embedded checkpoint config", source)
            self.assertEqual(config.model.variant, DYNAMIC_VARIANT_V11_4)
            self.assertFalse(
                config.dynamic_2dgs.regularization.small_motion.enabled
            )
            # Resuming it with runtime keys only must also validate.
            resolved, _ = resolve_main_config(OmegaConf.from_dotlist([
                f"train.ckpt_path={checkpoint}", "device=[2,3]",
            ]))
            self.assertEqual(list(resolved.device), [2, 3])

    def test_v11_checkpoint_that_enables_a_retired_feature_is_rejected(self):
        for overrides, message in (
            ({"budget": True}, "routing budget"),
            ({"small_motion": True}, "small-motion"),
            ({"w_scale": 0.02}, "w_scale"),
            ({"enable_lpips": True}, "enable_lpips"),
            ({"g2p_dynamic": True}, "g2p.dynamic"),
        ):
            with self.subTest(overrides=overrides):
                with tempfile.TemporaryDirectory() as tmp:
                    checkpoint = self._save_with_retired_blocks(
                        Path(tmp), **overrides
                    )
                    with self.assertRaisesRegex(ValueError, message):
                        load_checkpoint_config(checkpoint)

    def test_dynamic_variant_is_executable(self):
        config, _ = compose_fresh_config(
            OmegaConf.from_dotlist([f"model.variant={DYNAMIC_VARIANT}"])
        )
        self.assertTrue(config.dynamic_2dgs.motion.zero_init)
        self.assertNotIn("max_speed_mps", config.dynamic_2dgs.motion)

    def test_train_resume_uses_checkpoint_local_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = self._legacy_checkpoint(Path(tmp))
            cli = OmegaConf.from_dotlist([
                f"train.ckpt_path={checkpoint}",
                "allow_legacy_wandb_fallback=true",
                "train.batch_size=7",
            ])
            config, source = resolve_main_config(cli)
            self.assertEqual(config.model.variant, DYNAMIC_VARIANT)
            self.assertEqual(config.train.batch_size, 7)
            self.assertIn("wandb/latest-run/files/config.yaml", source)

    def test_fresh_main_and_eval_use_identical_model_config(self):
        cli = OmegaConf.from_dotlist(["train.batch_size=3"])
        main_config, _ = resolve_main_config(cli)
        eval_config, _ = resolve_eval_config(cli)
        for key in ("model", "p2g", "g2g", "g2p"):
            self.assertEqual(
                OmegaConf.to_container(main_config[key], resolve=True),
                OmegaConf.to_container(eval_config[key], resolve=True),
            )


class V15ConfigTest(unittest.TestCase):
    """V15's overlay, and the V15-only keys every other variant must refuse."""

    @staticmethod
    def _compose(*overrides, variant=DYNAMIC_VARIANT_V15):
        return compose_fresh_config(OmegaConf.from_dotlist([
            f"model.variant={variant}", *overrides,
        ]))

    def _container(self, variant=DYNAMIC_VARIANT_V15):
        config, _source = self._compose(variant=variant)
        return OmegaConf.to_container(config, resolve=False)

    @staticmethod
    def _validate(container):
        validate_experiment_config(OmegaConf.create(container))

    def _rejected(self, message, *overrides, variant=DYNAMIC_VARIANT_V15):
        with self.assertRaisesRegex(ValueError, message):
            self._compose(*overrides, variant=variant)

    def test_v15_composes_and_validates(self):
        config, source = self._compose()
        self.assertIn(f"{DYNAMIC_VARIANT_V15}.yaml", source)
        self.assertEqual(config.model.variant, DYNAMIC_VARIANT_V15)
        self.assertEqual(
            OmegaConf.to_container(config, resolve=True)["exp_name"],
            f"{DYNAMIC_VARIANT_V15}-bs-{config.train.batch_size}",
        )
        nested = config.p2g.grid_query
        self.assertEqual(nested.count_mode, NESTED_COUNT_MODE)
        self.assertEqual(set(nested), {"count_mode", "nested_count"})
        count = nested.nested_count
        self.assertEqual(list(count.option_counts), [1, 3, 5])
        self.assertEqual(count.seed_mode, "token")
        self.assertEqual(count.tau, 1.0)
        self.assertEqual(count.decoder_hidden, 576)
        self.assertEqual(list(count.router_hidden), [256, 128])
        self.assertEqual(count.auxiliary_render, "dense_keep")
        self.assertNotIn("gumbel_noise", count)
        self.assertTrue(count.logging.enable)
        self.assertEqual(count.logging.interval, 20)
        self.assertEqual(
            list(count.logging.range_edges_m), [0, 10, 20, 30, 40, 60, 80, 110]
        )
        local = config.dynamic_2dgs.local_attention
        self.assertEqual(local.radius_m, 0.8)
        self.assertEqual(local.num_heads, 12)
        self.assertEqual(local.mlp_ratio, 4)
        self.assertAlmostEqual(local.rope_base, 10 ** 1.2)
        self.assertAlmostEqual(local.rope_position_scale, 2.0 * math.pi / 0.4)
        self.assertEqual(local.implementation, "flash_varlen")
        self.assertEqual(local.query_chunk_size, 4096)
        temporal = config.dynamic_2dgs.temporal
        self.assertIs(temporal.layer_scale, False)
        self.assertNotIn("layer_scale_init", temporal)
        self.assertEqual(
            config.p2g.utonia_adaptation.stem.adapter_position,
            "post_activation",
        )
        self.assertEqual(
            list(config.train.downstream_no_decay),
            ["dynamic_backend.temporal.v_raw"],
        )
        self.assertEqual(config.data.dataset_name, "nuscenes_dynamic_lidar4d")
        self.assertEqual(config.data.coordinate_frame, "ego_vehicle")

    def test_v15_is_v14_1_with_five_changes(self):
        """Nothing outside the five documented changes differs from V14.1."""
        v14_1 = self._container(DYNAMIC_VARIANT_V14_1)
        v15 = self._container()
        left = OmegaConf.to_container(OmegaConf.create(v14_1), resolve=False)
        right = OmegaConf.to_container(OmegaConf.create(v15), resolve=False)
        left["model"]["variant"] = right["model"]["variant"]
        # 1. post-activation stem adapter.
        stem = left["p2g"]["utonia_adaptation"]["stem"]
        self.assertNotIn("adapter_position", stem)
        stem["adapter_position"] = "post_activation"
        # 5. nested count block replaces the learned_gumbel one wholesale.
        self.assertEqual(left["p2g"]["grid_query"]["count_mode"], "learned_gumbel")
        left["p2g"]["grid_query"] = right["p2g"]["grid_query"]
        # 2. v_raw without weight decay.
        self.assertNotIn("downstream_no_decay", left["train"])
        left["train"]["downstream_no_decay"] = ["dynamic_backend.temporal.v_raw"]
        # 3. no temporal LayerScale.
        self.assertEqual(left["dynamic_2dgs"]["temporal"]["layer_scale_init"], 0.1)
        del left["dynamic_2dgs"]["temporal"]["layer_scale_init"]
        left["dynamic_2dgs"]["temporal"]["layer_scale"] = False
        # 4. the radius local-attention block.
        self.assertNotIn("local_attention", left["dynamic_2dgs"])
        left["dynamic_2dgs"]["local_attention"] = right["dynamic_2dgs"][
            "local_attention"
        ]
        self.assertEqual(left, right)

    def test_v15_registration_and_variant_groups(self):
        self.assertIn(DYNAMIC_VARIANT_V15, DYNAMIC_VARIANTS)
        self.assertIn(DYNAMIC_VARIANT_V15, VARIANT_CONFIG_PATHS)
        self.assertTrue(VARIANT_CONFIG_PATHS[DYNAMIC_VARIANT_V15].is_file())
        for group in (BARRIER_MATCH_DYNAMIC_VARIANTS, HYBRID_UTONIA_DYNAMIC_VARIANTS):
            self.assertIn(DYNAMIC_VARIANT_V15, group)
            self.assertIn(DYNAMIC_VARIANT_V14_1, group)
        # The nested router is not the K={1,2,3} learned_gumbel one.
        self.assertNotIn(DYNAMIC_VARIANT_V15, ROUTER_CAPABLE_DYNAMIC_VARIANTS)
        self.assertNotIn(DYNAMIC_VARIANT_V15, FLASH_ROPE_MATCH_DYNAMIC_VARIANTS)
        self.assertEqual(
            NESTED_COUNT_DYNAMIC_VARIANTS,
            (DYNAMIC_VARIANT_V15, DYNAMIC_VARIANT_V15_1),
        )
        # The whole V15 family shares the stem, LayerScale removal and the
        # local-attention block; only the Gaussian head differs.
        self.assertEqual(V15_FAMILY_DYNAMIC_VARIANTS, (
            DYNAMIC_VARIANT_V15, DYNAMIC_VARIANT_V15_1,
            DYNAMIC_VARIANT_V15_2, DYNAMIC_VARIANT_V15_3, DYNAMIC_VARIANT_V15_4,
            DYNAMIC_VARIANT_V15_5, DYNAMIC_VARIANT_V16, DYNAMIC_VARIANT_V16_1,
            DYNAMIC_VARIANT_V18, DYNAMIC_VARIANT_V18_1, DYNAMIC_VARIANT_V18_2,
        ))
        for group in (
            LOCAL_ATTENTION_DYNAMIC_VARIANTS,
            NO_TEMPORAL_LAYER_SCALE_DYNAMIC_VARIANTS,
        ):
            self.assertEqual(group, (*V15_FAMILY_DYNAMIC_VARIANTS, DYNAMIC_VARIANT_V17_1))
        self.assertEqual(POST_ACTIVATION_STEM_DYNAMIC_VARIANTS, V15_FAMILY_DYNAMIC_VARIANTS)
        # V15.1 shares every V15 group; its auxiliary render is its identity.
        self.assertIn(DYNAMIC_VARIANT_V15_1, DYNAMIC_VARIANTS)
        self.assertTrue(VARIANT_CONFIG_PATHS[DYNAMIC_VARIANT_V15_1].is_file())
        for group in (BARRIER_MATCH_DYNAMIC_VARIANTS, HYBRID_UTONIA_DYNAMIC_VARIANTS):
            self.assertIn(DYNAMIC_VARIANT_V15_1, group)
        self.assertNotIn(DYNAMIC_VARIANT_V15_1, ROUTER_CAPABLE_DYNAMIC_VARIANTS)
        self.assertEqual(NESTED_AUXILIARY_RENDER_BY_VARIANT, {
            DYNAMIC_VARIANT_V15: "dense_keep",
            DYNAMIC_VARIANT_V15_1: "none",
        })
        self.assertEqual(NESTED_COUNT_MODE, "nested_gumbel")
        self.assertEqual(NESTED_COUNT_OPTION_COUNTS, (1, 3, 5))
        self.assertEqual(NESTED_AUXILIARY_RENDERS, ("dense_keep", "none"))
        self.assertEqual(
            LOCAL_ATTENTION_IMPLEMENTATIONS, ("flash_varlen", "sdpa", "auto")
        )
        # The fresh-run default is still V11.4.
        self.assertEqual(DYNAMIC_VARIANT, DYNAMIC_VARIANT_V11_4)

    def test_v15_only_keys_are_rejected_on_v14_1(self):
        self._rejected(
            "no nested count head.*only available to V15",
            "p2g.grid_query.count_mode=nested_gumbel",
            variant=DYNAMIC_VARIANT_V14_1,
        )
        self._rejected(
            "layer_scale=false is only available to V15",
            "dynamic_2dgs.temporal.layer_scale=false",
            variant=DYNAMIC_VARIANT_V14_1,
        )
        self._rejected(
            "adapter_position='post_activation' is only available to V15",
            "p2g.utonia_adaptation.stem.adapter_position=post_activation",
            variant=DYNAMIC_VARIANT_V14_1,
        )
        # A complete, otherwise valid block is still refused.
        block = self._container()["dynamic_2dgs"]["local_attention"]
        with self.assertRaisesRegex(ValueError, "only available to V15"):
            compose_fresh_config(OmegaConf.create({
                "model": {"variant": DYNAMIC_VARIANT_V14_1},
                "dynamic_2dgs": {"local_attention": block},
            }))

    def test_v15_1_is_v15_without_the_auxiliary_render(self):
        v15, _ = self._compose()
        v15_1, _ = self._compose(variant=DYNAMIC_VARIANT_V15_1)

        def leaves(node, prefix=""):
            if isinstance(node, dict):
                out = {}
                for key, value in node.items():
                    out.update(leaves(value, f"{prefix}{key}."))
                return out
            return {prefix[:-1]: node}

        a = leaves(OmegaConf.to_container(v15, resolve=True))
        b = leaves(OmegaConf.to_container(v15_1, resolve=True))
        self.assertEqual(set(a), set(b))
        changed = {key for key in a if a[key] != b[key]}
        self.assertEqual(changed, {
            "model.variant",
            "exp_name",
            "logger.dir",
            "p2g.grid_query.nested_count.auxiliary_render",
        })
        self.assertEqual(
            v15_1.p2g.grid_query.nested_count.auxiliary_render, "none"
        )

    def test_nested_count_mode_is_refused_by_every_other_variant(self):
        for variant in DYNAMIC_VARIANTS:
            if variant in NESTED_COUNT_DYNAMIC_VARIANTS:
                continue
            with self.subTest(variant=variant):
                self._rejected(
                    "no nested count head",
                    "p2g.grid_query.count_mode=nested_gumbel",
                    variant=variant,
                )

    def test_v15_only_keys_are_refused_outside_their_variant_groups(self):
        for variant in DYNAMIC_VARIANTS:
            if variant in LOCAL_ATTENTION_DYNAMIC_VARIANTS:
                self.assertIn(variant, NO_TEMPORAL_LAYER_SCALE_DYNAMIC_VARIANTS)
                continue
            with self.subTest(variant=variant, key="local_attention"):
                self._rejected(
                    "no radius local-attention block",
                    "dynamic_2dgs.local_attention.radius_m=0.8",
                    variant=variant,
                )
            with self.subTest(variant=variant, key="layer_scale"):
                self._rejected(
                    "keeps the temporal LayerScale",
                    "dynamic_2dgs.temporal.layer_scale=false",
                    variant=variant,
                )
        # An explicit layer_scale=true is the historical gate, for anyone.
        config, _ = self._compose(
            "dynamic_2dgs.temporal.layer_scale=true",
            variant=DYNAMIC_VARIANT_V14_1,
        )
        self.assertIs(config.dynamic_2dgs.temporal.layer_scale, True)
        self.assertEqual(config.dynamic_2dgs.temporal.layer_scale_init, 0.1)
        self._rejected(
            "true or false",
            "dynamic_2dgs.temporal.layer_scale=1",
            variant=DYNAMIC_VARIANT_V14_1,
        )

    def test_downstream_no_decay_is_only_for_hybrid_variants(self):
        names = "train.downstream_no_decay=[dynamic_backend.temporal.v_raw]"
        for variant in DYNAMIC_VARIANTS:
            if variant in (*HYBRID_UTONIA_DYNAMIC_VARIANTS, DYNAMIC_VARIANT_V17_1):
                continue
            with self.subTest(variant=variant):
                self._rejected(
                    "only available to V14, V14.1, V15", names, variant=variant
                )
        # Even an empty list has nothing to act on without role groups.
        self._rejected(
            "flat parameter group", "train.downstream_no_decay=[]",
            variant=DYNAMIC_VARIANT_V11_4,
        )
        for variant in (DYNAMIC_VARIANT_V14, DYNAMIC_VARIANT_V14_1):
            with self.subTest(variant=variant):
                config, _ = self._compose(names, variant=variant)
                self.assertEqual(
                    list(config.train.downstream_no_decay),
                    ["dynamic_backend.temporal.v_raw"],
                )
        # Absent, null and empty all mean "no extra group".
        for value in ("null", "[]"):
            config, _ = self._compose(f"train.downstream_no_decay={value}")
            self.assertFalse(config.train.downstream_no_decay or [])

    def test_downstream_no_decay_rejects_malformed_lists(self):
        cases = (
            ("train.downstream_no_decay=v_raw", "must be a list"),
            (
                "train.downstream_no_decay=[dynamic_backend.temporal.v_raw,"
                "dynamic_backend.temporal.v_raw]",
                "must not repeat",
            ),
            ("train.downstream_no_decay=[a,'']", "non-empty strings"),
            ("train.downstream_no_decay=['  ']", "non-empty strings"),
            ("train.downstream_no_decay=[1]", "non-empty strings"),
            (
                "train.downstream_no_decay=[feature_extractor.embedding.stem."
                "linear.intensity_weight]",
                "Utonia parameter",
            ),
        )
        for override, message in cases:
            with self.subTest(override=override):
                self._rejected(message, override)

    def test_nested_count_contract_is_enforced(self):
        base = "p2g.grid_query.nested_count."
        cases = (
            ("p2g.grid_query.count_mode=learned_gumbel", "requires p2g.grid_query"),
            ("p2g.grid_query.count_mode=legacy", "requires p2g.grid_query"),
            (base + "option_counts=[1,2,3]", "option_counts must be"),
            (base + "option_counts=[1,3]", "option_counts must be"),
            (base + "option_counts=[1,3,5,7]", "option_counts must be"),
            (base + "option_counts=[5,3,1]", "option_counts must be"),
            (base + "option_counts=3", "option_counts must be"),
            (base + "seed_mode=range_quantile", "seed_mode must be one of"),
            (base + "seed_mode=medoid", "seed_mode must be one of"),
            (base + "auxiliary_render=sparse", "auxiliary_render must be one of"),
            (base + "auxiliary_render=null", "auxiliary_render must be one of"),
            (base + "tau=0", "tau must be positive"),
            (base + "tau=-1.0", "tau must be positive"),
            (base + "tau=inf", "tau must be positive and finite"),
            (base + "tau=null", "tau must be positive"),
            (base + "decoder_hidden=0", "decoder_hidden must be a positive integer"),
            (base + "decoder_hidden=576.5", "decoder_hidden must be a positive integer"),
            (base + "decoder_hidden=null", "decoder_hidden must be a positive integer"),
            (base + "router_hidden=-432", "router_hidden must be a non-empty list"),
            (base + "router_hidden=432", "router_hidden must be a non-empty list"),
            (base + "router_hidden=[]", "router_hidden must be a non-empty list"),
            (base + "router_hidden=[256,0]", "router_hidden must be a non-empty list"),
            (base + "router_hidden=[256,128.5]", "router_hidden must be a non-empty list"),
            (base + "gumbel_noise.schedule=cosine", "Unsupported keys under"),
            (base + "budget=1", "Unsupported keys under p2g.grid_query.nested_count"),
            ("p2g.grid_query.K_max=3", "Unsupported keys under p2g.grid_query"),
            ("p2g.grid_query.grad_balance=sqrt_k", "Unsupported keys under p2g.grid_query"),
            (base + "logging.interval=0", "interval must be positive"),
            (base + "logging.bins=3", "Unsupported keys under"),
        )
        for override, message in cases:
            with self.subTest(override=override):
                self._rejected(message, override)
        # Each nested variant accepts only its own auxiliary render: V15 the
        # dense G1..G3 render, V15.1 none. Logging is optional.
        for variant, render in NESTED_AUXILIARY_RENDER_BY_VARIANT.items():
            with self.subTest(variant=variant, render=render):
                config, _ = self._compose(
                    base + f"auxiliary_render={render}", variant=variant
                )
                self.assertEqual(
                    config.p2g.grid_query.nested_count.auxiliary_render, render
                )
                other = "none" if render == "dense_keep" else "dense_keep"
                self._rejected(
                    f"auxiliary_render='{render}', got '{other}'",
                    base + f"auxiliary_render={other}",
                    variant=variant,
                )
        container = self._container()
        del container["p2g"]["grid_query"]["nested_count"]["logging"]
        self._validate(container)

    def test_nested_count_keys_are_all_required_except_logging(self):
        for key in (
            "option_counts", "seed_mode", "tau", "decoder_hidden",
            "router_hidden", "auxiliary_render",
        ):
            with self.subTest(key=key):
                container = self._container()
                del container["p2g"]["grid_query"]["nested_count"][key]
                with self.assertRaisesRegex(
                    ValueError, f"nested_count is missing keys: {key}"
                ):
                    self._validate(container)
        container = self._container()
        del container["p2g"]["grid_query"]["nested_count"]
        with self.assertRaisesRegex(ValueError, "requires p2g.grid_query.nested_count"):
            self._validate(container)
        container = self._container()
        del container["p2g"]["grid_query"]
        with self.assertRaisesRegex(ValueError, "requires p2g.grid_query.count_mode"):
            self._validate(container)

    def test_local_attention_contract_is_enforced(self):
        base = "dynamic_2dgs.local_attention."
        cases = (
            ("dynamic_2dgs.local_attention=null", "requires dynamic_2dgs.local_attention"),
            (base + "query_chunk_size=70000", "at most 65535"),
            (base + "query_chunk_size=65536", "at most 65535"),
            (base + "query_chunk_size=0", "query_chunk_size must be a positive integer"),
            (base + "query_chunk_size=4096.5", "query_chunk_size must be a positive integer"),
            (base + "num_heads=7", "dim=432 must be divisible by heads=7"),
            (base + "num_heads=16", "head_dim=27 must be divisible by six"),
            (base + "num_heads=0", "num_heads must be a positive integer"),
            (base + "num_heads=true", "num_heads must be a positive integer"),
            (base + "implementation=triton", "implementation must be one of"),
            (base + "radius_m=0", "radius_m must be positive and finite"),
            (base + "radius_m=-0.8", "radius_m must be positive and finite"),
            (base + "radius_m=inf", "radius_m must be positive and finite"),
            (base + "mlp_ratio=0", "mlp_ratio must be positive and finite"),
            (base + "rope_base=-1", "rope_base must be positive and finite"),
            (base + "rope_position_scale=nan", "rope_position_scale must be positive and finite"),
            (base + "qk_norm=true", "Unsupported keys under dynamic_2dgs.local_attention"),
        )
        for override, message in cases:
            with self.subTest(override=override):
                self._rejected(message, override)
        for implementation in LOCAL_ATTENTION_IMPLEMENTATIONS:
            with self.subTest(implementation=implementation):
                config, _ = self._compose(base + f"implementation={implementation}")
                self.assertEqual(
                    config.dynamic_2dgs.local_attention.implementation,
                    implementation,
                )
        # 65535 is FlashAttention2's limit and is allowed.
        config, _ = self._compose(base + "query_chunk_size=65535")
        self.assertEqual(config.dynamic_2dgs.local_attention.query_chunk_size, 65535)
        for key in (
            "radius_m", "num_heads", "mlp_ratio", "rope_base",
            "rope_position_scale", "implementation", "query_chunk_size",
        ):
            with self.subTest(missing=key):
                container = self._container()
                del container["dynamic_2dgs"]["local_attention"][key]
                with self.assertRaisesRegex(
                    ValueError, f"local_attention is missing keys: {key}"
                ):
                    self._validate(container)
        container = self._container()
        del container["dynamic_2dgs"]["local_attention"]
        with self.assertRaisesRegex(ValueError, "requires dynamic_2dgs.local_attention"):
            self._validate(container)

    def test_v15_requires_no_layer_scale(self):
        self._rejected(
            "requires dynamic_2dgs.temporal.layer_scale=false",
            "dynamic_2dgs.temporal.layer_scale=true",
        )
        self._rejected(
            "layer_scale_init cannot be set",
            "dynamic_2dgs.temporal.layer_scale_init=0.1",
        )
        self._rejected("true or false", "dynamic_2dgs.temporal.layer_scale=0")
        container = self._container()
        del container["dynamic_2dgs"]["temporal"]["layer_scale"]
        with self.assertRaisesRegex(ValueError, "requires dynamic_2dgs.temporal.layer_scale=false"):
            self._validate(container)

    def test_v15_requires_the_post_activation_stem(self):
        stem = "p2g.utonia_adaptation.stem.adapter_position"
        self._rejected("requires .*adapter_position='post_activation'", f"{stem}=pre_norm")
        self._rejected("adapter_position must be one of", f"{stem}=after_ln")
        self._rejected("adapter_position must be one of", f"{stem}=null")
        container = self._container()
        del container["p2g"]["utonia_adaptation"]["stem"]["adapter_position"]
        with self.assertRaisesRegex(ValueError, "requires .*adapter_position='post_activation'"):
            self._validate(container)
        # V14 and V14.1 keep the historical spelling, present or absent.
        for variant in (DYNAMIC_VARIANT_V14, DYNAMIC_VARIANT_V14_1):
            with self.subTest(variant=variant):
                config, _ = self._compose(f"{stem}=pre_norm", variant=variant)
                self.assertEqual(
                    config.p2g.utonia_adaptation.stem.adapter_position, "pre_norm"
                )
                config, _ = self._compose(variant=variant)
                self.assertNotIn("adapter_position", config.p2g.utonia_adaptation.stem)
        self._rejected("adapter_position must be one of", f"{stem}=after_ln", variant=DYNAMIC_VARIANT_V14)

    def test_v14_family_configs_keep_validating_without_the_v15_keys(self):
        for variant in (DYNAMIC_VARIANT_V14, DYNAMIC_VARIANT_V14_1):
            with self.subTest(variant=variant):
                config, _ = self._compose(variant=variant)
                self.assertEqual(config.p2g.grid_query.count_mode, "learned_gumbel")
                self.assertNotIn("local_attention", config.dynamic_2dgs)
                self.assertNotIn("layer_scale", config.dynamic_2dgs.temporal)
                self.assertNotIn("downstream_no_decay", config.train)

    def test_v15_checkpoint_restores_and_protects_its_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "epoch=0.ckpt"
            stored, _ = self._compose()
            torch.save({
                "experiment_config": OmegaConf.to_container(stored, resolve=True)
            }, checkpoint)

            config, _source = load_checkpoint_config(checkpoint)
            self.assertEqual(config.model.variant, DYNAMIC_VARIANT_V15)
            self.assertEqual(config.p2g.grid_query.count_mode, NESTED_COUNT_MODE)
            self.assertEqual(config.dynamic_2dgs.local_attention.radius_m, 0.8)
            self.assertIs(config.dynamic_2dgs.temporal.layer_scale, False)
            for override, root in (
                ("p2g.grid_query.nested_count.tau=2.0", "p2g"),
                ("p2g.grid_query.nested_count.auxiliary_render=none", "p2g"),
                ("p2g.utonia_adaptation.stem.adapter_position=pre_norm", "p2g"),
                ("dynamic_2dgs.local_attention.radius_m=1.0", "dynamic_2dgs"),
                ("dynamic_2dgs.temporal.layer_scale=true", "dynamic_2dgs"),
            ):
                with self.subTest(override=override):
                    with self.assertRaisesRegex(
                        ValueError, f"changed roots: {root}"
                    ):
                        resolve_main_config(OmegaConf.from_dotlist([
                            f"train.ckpt_path={checkpoint}", override,
                        ]))
            # Runtime knobs stay adjustable.
            resumed, _ = resolve_main_config(OmegaConf.from_dotlist([
                f"train.ckpt_path={checkpoint}", "train.batch_size=3",
            ]))
            self.assertEqual(resumed.train.batch_size, 3)
            self.assertEqual(
                list(resumed.train.downstream_no_decay),
                ["dynamic_backend.temporal.v_raw"],
            )



class V15SiblingConfigTest(unittest.TestCase):
    """V15.2 (per-option experts) and V15.3 (no router): V15 with a new head."""

    @staticmethod
    def _compose(*overrides, variant):
        return compose_fresh_config(OmegaConf.from_dotlist([
            f"model.variant={variant}", *overrides,
        ]))

    def _container(self, variant):
        config, _source = self._compose(variant=variant)
        return OmegaConf.to_container(config, resolve=False)

    def _rejected(self, message, *overrides, variant):
        with self.assertRaisesRegex(ValueError, message):
            self._compose(*overrides, variant=variant)

    @staticmethod
    def _leaves(node, prefix=""):
        if isinstance(node, dict):
            out = {}
            for key, value in node.items():
                out.update(V15SiblingConfigTest._leaves(value, f"{prefix}{key}."))
            return out
        return {prefix[:-1]: node}

    def _changed_from_v15(self, variant):
        v15, _ = self._compose(variant=DYNAMIC_VARIANT_V15)
        sibling, _ = self._compose(variant=variant)
        a = self._leaves(OmegaConf.to_container(v15, resolve=True))
        b = self._leaves(OmegaConf.to_container(sibling, resolve=True))
        return a, b, {
            key for key in set(a) | set(b) if a.get(key, "<absent>") != b.get(key, "<absent>")
        }

    def test_registration(self):
        for variant in (
            DYNAMIC_VARIANT_V15_2, DYNAMIC_VARIANT_V15_3, DYNAMIC_VARIANT_V15_4,
            DYNAMIC_VARIANT_V15_5, DYNAMIC_VARIANT_V18, DYNAMIC_VARIANT_V18_1,
            DYNAMIC_VARIANT_V18_2,
        ):
            with self.subTest(variant=variant):
                self.assertIn(variant, DYNAMIC_VARIANTS)
                self.assertTrue(VARIANT_CONFIG_PATHS[variant].is_file())
                for group in (
                    BARRIER_MATCH_DYNAMIC_VARIANTS,
                    HYBRID_UTONIA_DYNAMIC_VARIANTS,
                    V15_FAMILY_DYNAMIC_VARIANTS,
                ):
                    self.assertIn(variant, group)
                self.assertNotIn(variant, ROUTER_CAPABLE_DYNAMIC_VARIANTS)
                self.assertNotIn(variant, NESTED_COUNT_DYNAMIC_VARIANTS)
        self.assertEqual(EXPERT_COUNT_DYNAMIC_VARIANTS, (DYNAMIC_VARIANT_V15_2,))
        self.assertEqual(
            DENSE_COUNT_DYNAMIC_VARIANTS,
            (DYNAMIC_VARIANT_V15_3, DYNAMIC_VARIANT_V15_4, DYNAMIC_VARIANT_V15_5,
             DYNAMIC_VARIANT_V17_1, DYNAMIC_VARIANT_V18, DYNAMIC_VARIANT_V18_1,
             DYNAMIC_VARIANT_V18_2),
        )
        self.assertEqual(DENSE_COUNT_TEMPORAL_HEADS_BY_VARIANT, {
            DYNAMIC_VARIANT_V15_3: 12, DYNAMIC_VARIANT_V15_4: 8,
            DYNAMIC_VARIANT_V15_5: 8,
            DYNAMIC_VARIANT_V17_1: 8,
            DYNAMIC_VARIANT_V18: 8,
            DYNAMIC_VARIANT_V18_1: 8,
            DYNAMIC_VARIANT_V18_2: 8,
        })
        self.assertEqual(EXPERT_COUNT_MODE, "expert_gumbel")
        self.assertEqual(EXPERT_AUXILIARY_RENDERS, ("dense_add", "none"))
        self.assertEqual(
            EXPERT_AUXILIARY_RENDER_BY_VARIANT,
            {DYNAMIC_VARIANT_V15_2: "dense_add"},
        )
        self.assertEqual(ROUTED_COUNT_BLOCKS, {
            NESTED_COUNT_MODE: "nested_count",
            EXPERT_COUNT_MODE: "expert_count",
            ORDINAL_COUNT_MODE: "ordinal_count",
        })

    def test_v15_2_is_v15_with_per_option_experts(self):
        a, b, changed = self._changed_from_v15(DYNAMIC_VARIANT_V15_2)
        nested = {key for key in a if key.startswith("p2g.grid_query.nested_count.")}
        expert = {key for key in b if key.startswith("p2g.grid_query.expert_count.")}
        self.assertEqual(changed, {
            "model.variant", "exp_name", "logger.dir",
            "p2g.grid_query.count_mode", *nested, *expert,
        })
        self.assertEqual(b["p2g.grid_query.count_mode"], "expert_gumbel")
        # The block's settings are V15's except the auxiliary render.
        for key in nested:
            name = key.rsplit("nested_count.", 1)[1]
            expected = "dense_add" if name == "auxiliary_render" else a[key]
            self.assertEqual(b[f"p2g.grid_query.expert_count.{name}"], expected)
        # Plus the E5 render's view subsampling, which V15 does not set.
        self.assertEqual(
            {key.rsplit("nested_count.", 1)[1] for key in nested}
            | {"auxiliary_views"},
            {key.rsplit("expert_count.", 1)[1] for key in expert},
        )
        self.assertEqual(b["p2g.grid_query.expert_count.auxiliary_views"], 2)

    def test_v15_3_is_v15_without_the_router(self):
        a, b, changed = self._changed_from_v15(DYNAMIC_VARIANT_V15_3)
        nested = {key for key in a if key.startswith("p2g.grid_query.nested_count.")}
        self.assertEqual(changed, {
            "model.variant", "exp_name", "logger.dir",
            "p2g.grid_query.count_mode", "p2g.grid_query.K_max",
            "p2g.grid_query.seed_mode", "dynamic_2dgs.gaussian_head.decoder_hidden",
            *nested,
        })
        self.assertEqual(b["p2g.grid_query.count_mode"], "legacy")
        self.assertEqual(b["p2g.grid_query.K_max"], 5)
        self.assertEqual(b["p2g.grid_query.seed_mode"], "token")
        # V15's decoder width, moved to the Gaussian head.
        self.assertEqual(
            b["dynamic_2dgs.gaussian_head.decoder_hidden"],
            a["p2g.grid_query.nested_count.decoder_hidden"],
        )

    def test_v15_4_is_v15_3_with_eight_temporal_heads(self):
        v15_3, _ = self._compose(variant=DYNAMIC_VARIANT_V15_3)
        v15_4, _ = self._compose(variant=DYNAMIC_VARIANT_V15_4)
        a = self._leaves(OmegaConf.to_container(v15_3, resolve=True))
        b = self._leaves(OmegaConf.to_container(v15_4, resolve=True))
        changed = {
            key for key in set(a) | set(b)
            if a.get(key, "<absent>") != b.get(key, "<absent>")
        }
        self.assertEqual(changed, {
            "model.variant", "exp_name", "logger.dir",
            "dynamic_2dgs.temporal.num_heads", "dynamic_2dgs.temporal.rope_base",
        })
        temporal = "dynamic_2dgs.temporal."
        self.assertEqual((a[temporal + "num_heads"], b[temporal + "num_heads"]), (12, 8))
        self.assertEqual(b["dynamic_2dgs.local_attention.num_heads"], 12)
        # Nine RoPE bands per axis over V15.3's six-band span, 5 m to 232 m.
        from src.models.utils.attention import Rotary3D

        def wavelengths(config):
            heads = config.dynamic_2dgs.temporal.num_heads
            rope = Rotary3D(
                432 // heads,
                base=config.dynamic_2dgs.temporal.rope_base,
                position_scale=config.dynamic_2dgs.temporal.rope_position_scale,
            )
            return 2.0 * math.pi / (rope.position_scale * rope.inv_freq.double())

        old, new = wavelengths(v15_3), wavelengths(v15_4)
        self.assertEqual((old.numel(), new.numel()), (6, 9))
        self.assertAlmostEqual(float(new[0]), float(old[0]), places=5)
        self.assertAlmostEqual(float(new[-1]), float(old[-1]), places=3)
        self.assertAlmostEqual(float(new[0]), 5.0, places=5)

    def test_v15_5_config_is_v15_4s(self):
        # V15.5's change (the uniform layer mean) lives in its backend class,
        # so its composed config differs from V15.4's only in its name.
        v15_4, _ = self._compose(variant=DYNAMIC_VARIANT_V15_4)
        v15_5, _ = self._compose(variant=DYNAMIC_VARIANT_V15_5)
        a = self._leaves(OmegaConf.to_container(v15_4, resolve=True))
        b = self._leaves(OmegaConf.to_container(v15_5, resolve=True))
        changed = {
            key for key in set(a) | set(b)
            if a.get(key, "<absent>") != b.get(key, "<absent>")
        }
        self.assertEqual(changed, {"model.variant", "exp_name", "logger.dir"})
        self.assertEqual(b["dynamic_2dgs.temporal.num_heads"], 8)
        self.assertEqual(b["data.dataset_name"], "nuscenes_dynamic_lidar4d")

    def test_v18_config_is_v15_4s_with_a_feature_and_its_decoder(self):
        v15_4, _ = self._compose(variant=DYNAMIC_VARIANT_V15_4)
        v18, _ = self._compose(variant=DYNAMIC_VARIANT_V18)
        a = self._leaves(OmegaConf.to_container(v15_4, resolve=True))
        b = self._leaves(OmegaConf.to_container(v18, resolve=True))
        changed = {
            key for key in set(a) | set(b)
            if a.get(key, "<absent>") != b.get(key, "<absent>")
        }
        decoder = "g2p.lidar_decoder."
        self.assertEqual(changed, {
            "model.variant", "exp_name", "logger.dir",
            "p2g.gs_params.shs", "p2g.gs_params.feature",
            *(decoder + key for key in (
                "feature_dim", "sensor_embedding_dim", "num_sensors",
                "hidden_dims", "range_scale_m",
            )),
        })
        self.assertEqual((a["p2g.gs_params.shs"], b["p2g.gs_params.shs"]), (32, 0))
        self.assertEqual(b["p2g.gs_params.feature"], 30)
        # [f' (30), ray (3), r / R (1), sensor embedding (30)] = 64 inputs.
        self.assertEqual(
            b[decoder + "feature_dim"] + 3 + 1 + b[decoder + "sensor_embedding_dim"],
            64,
        )
        self.assertEqual(b[decoder + "hidden_dims"], [96, 96])
        self.assertEqual(b[decoder + "range_scale_m"], 80.0)

    def test_v18_appearance_keys_are_its_own(self):
        self._rejected(
            "p2g.gs_params.shs must be 0", "p2g.gs_params.shs=32",
            variant=DYNAMIC_VARIANT_V18,
        )
        self._rejected(
            "feature_dim must equal p2g.gs_params.feature",
            "g2p.lidar_decoder.feature_dim=16", variant=DYNAMIC_VARIANT_V18,
        )
        self._rejected(
            "at most 32", "p2g.gs_params.feature=33",
            "g2p.lidar_decoder.feature_dim=33", variant=DYNAMIC_VARIANT_V18,
        )
        self._rejected(
            "Unsupported keys under g2p.lidar_decoder",
            "g2p.lidar_decoder.depth_input=mean", variant=DYNAMIC_VARIANT_V18,
        )
        for override in (
            "p2g.gs_params.feature=30", "g2p.lidar_decoder.feature_dim=30",
        ):
            with self.subTest(override=override):
                self._rejected(
                    "only available to V18", override,
                    variant=DYNAMIC_VARIANT_V15_4,
                )

    def test_v18_1_config_is_v18s_without_the_sensor_embedding(self):
        self.assertEqual(
            FEATURE_DECODER_DYNAMIC_VARIANTS,
            (DYNAMIC_VARIANT_V18, DYNAMIC_VARIANT_V18_1, DYNAMIC_VARIANT_V18_2),
        )
        self.assertEqual(NO_SENSOR_EMBEDDING_DYNAMIC_VARIANTS, (DYNAMIC_VARIANT_V18_1,))
        v18, _ = self._compose(variant=DYNAMIC_VARIANT_V18)
        v18_1, _ = self._compose(variant=DYNAMIC_VARIANT_V18_1)
        a = self._leaves(OmegaConf.to_container(v18, resolve=True))
        b = self._leaves(OmegaConf.to_container(v18_1, resolve=True))
        changed = {
            key for key in set(a) | set(b)
            if a.get(key, "<absent>") != b.get(key, "<absent>")
        }
        decoder = "g2p.lidar_decoder."
        self.assertEqual(changed, {
            "model.variant", "exp_name", "logger.dir",
            decoder + "sensor_embedding_dim", decoder + "num_sensors",
        })
        self.assertEqual(b[decoder + "sensor_embedding_dim"], 0)
        self.assertNotIn(decoder + "num_sensors", b)
        # [f' (30), ray (3), r / R (1)] = 34 inputs.
        self.assertEqual(b[decoder + "feature_dim"] + 3 + 1, 34)

    def test_v18_1_and_v18_sensor_embedding_keys_are_their_own(self):
        self._rejected(
            "decodes without a sensor embedding; "
            "g2p.lidar_decoder.sensor_embedding_dim must be 0",
            "g2p.lidar_decoder.sensor_embedding_dim=30",
            variant=DYNAMIC_VARIANT_V18_1,
        )
        self._rejected(
            "num_sensors is only available to V18$",
            "g2p.lidar_decoder.num_sensors=1", variant=DYNAMIC_VARIANT_V18_1,
        )
        self._rejected(
            "sensor_embedding_dim must be a positive integer",
            "g2p.lidar_decoder.sensor_embedding_dim=0", variant=DYNAMIC_VARIANT_V18,
        )
        self._rejected(
            "only available to V18, V18.1, V18.2$", "p2g.gs_params.feature=30",
            variant=DYNAMIC_VARIANT_V15_4,
        )

    def test_v18_2_config_is_v18_1s_with_a_32_channel_feature_and_a_unet(self):
        self.assertEqual(UNET_DECODER_DYNAMIC_VARIANTS, (DYNAMIC_VARIANT_V18_2,))
        self.assertNotIn(DYNAMIC_VARIANT_V18_2, NO_SENSOR_EMBEDDING_DYNAMIC_VARIANTS)
        v18_1, _ = self._compose(variant=DYNAMIC_VARIANT_V18_1)
        v18_2, _ = self._compose(variant=DYNAMIC_VARIANT_V18_2)
        a = self._leaves(OmegaConf.to_container(v18_1, resolve=True))
        b = self._leaves(OmegaConf.to_container(v18_2, resolve=True))
        changed = {
            key for key in set(a) | set(b)
            if a.get(key, "<absent>") != b.get(key, "<absent>")
        }
        decoder = "g2p.lidar_decoder."
        self.assertEqual(changed, {
            "model.variant", "exp_name", "logger.dir", "p2g.gs_params.feature",
            decoder + "feature_dim", decoder + "sensor_embedding_dim",
            decoder + "hidden_dims", decoder + "channels", decoder + "norm_groups",
        })
        self.assertEqual(b["p2g.gs_params.feature"], 32)
        self.assertEqual(b[decoder + "feature_dim"], 32)
        self.assertEqual(b[decoder + "channels"], [64, 128, 256, 256])
        self.assertEqual(b[decoder + "norm_groups"], 32)
        self.assertEqual(b[decoder + "range_scale_m"], 80.0)
        for key in ("sensor_embedding_dim", "num_sensors", "hidden_dims"):
            self.assertNotIn(decoder + key, b)

    def test_v18_2_unet_decoder_keys_are_its_own(self):
        decoder = "g2p.lidar_decoder"
        self._rejected(
            "Unsupported keys under g2p.lidar_decoder: hidden_dims",
            f"{decoder}.hidden_dims=[96,96]", variant=DYNAMIC_VARIANT_V18_2,
        )
        self._rejected(
            "Unsupported keys under g2p.lidar_decoder: sensor_embedding_dim",
            f"{decoder}.sensor_embedding_dim=0", variant=DYNAMIC_VARIANT_V18_2,
        )
        self._rejected(
            "channels must list at least two positive integer encoder widths",
            f"{decoder}.channels=[64]", variant=DYNAMIC_VARIANT_V18_2,
        )
        self._rejected(
            "norm_groups must divide every width",
            f"{decoder}.norm_groups=48", variant=DYNAMIC_VARIANT_V18_2,
        )
        self._rejected(
            "divisible by the attention's 8 heads",
            f"{decoder}.channels=[64,128,252]", f"{decoder}.norm_groups=4",
            variant=DYNAMIC_VARIANT_V18_2,
        )
        self._rejected(
            "feature_dim must equal p2g.gs_params.feature",
            f"{decoder}.feature_dim=30", variant=DYNAMIC_VARIANT_V18_2,
        )
        self._rejected(
            "must be at most 32",
            "p2g.gs_params.feature=33", variant=DYNAMIC_VARIANT_V18_2,
        )
        # The MLP variants keep rejecting the U-Net's keys.
        self._rejected(
            "Unsupported keys under g2p.lidar_decoder: channels",
            f"{decoder}.channels=[64,128]", variant=DYNAMIC_VARIANT_V18_1,
        )

    def test_v15_3_and_v15_4_pin_their_temporal_head_counts(self):
        for variant, heads, other in (
            (DYNAMIC_VARIANT_V15_3, 12, 8),
            (DYNAMIC_VARIANT_V15_4, 8, 12),
            (DYNAMIC_VARIANT_V15_5, 8, 12),
            (DYNAMIC_VARIANT_V18, 8, 12),
            (DYNAMIC_VARIANT_V18_1, 8, 12),
            (DYNAMIC_VARIANT_V18_2, 8, 12),
        ):
            with self.subTest(variant=variant):
                self._rejected(
                    f"requires dynamic_2dgs.temporal.num_heads={heads}, "
                    f"got {other}",
                    f"dynamic_2dgs.temporal.num_heads={other}",
                    variant=variant,
                )
        # Other variants keep the head count a free (RoPE-checked) setting.
        config, _ = self._compose(
            "dynamic_2dgs.temporal.num_heads=8", variant=DYNAMIC_VARIANT_V15_2
        )
        self.assertEqual(config.dynamic_2dgs.temporal.num_heads, 8)

    def test_expert_count_mode_is_refused_by_every_other_variant(self):
        for variant in DYNAMIC_VARIANTS:
            if variant in EXPERT_COUNT_DYNAMIC_VARIANTS:
                continue
            with self.subTest(variant=variant):
                self._rejected(
                    "no per-option expert head.*only available to V15.2",
                    "p2g.grid_query.count_mode=expert_gumbel",
                    variant=variant,
                )
        self._rejected(
            "V15.2 has no nested count head",
            "p2g.grid_query.count_mode=nested_gumbel",
            variant=DYNAMIC_VARIANT_V15_2,
        )

    def test_expert_count_contract_is_enforced(self):
        base = "p2g.grid_query.expert_count."
        cases = (
            ("p2g.grid_query.count_mode=learned_gumbel", "requires p2g.grid_query.count_mode='expert_gumbel'"),
            ("p2g.grid_query.count_mode=legacy", "requires p2g.grid_query.count_mode='expert_gumbel'"),
            (base + "option_counts=[1,2,3]", "option_counts must be .*reduce E1, keep E3, add E5"),
            (base + "seed_mode=range_quantile", "seed_mode must be one of"),
            (base + "auxiliary_render=dense_keep", "auxiliary_render must be one of"),
            (base + "auxiliary_render=none", "renders no auxiliary set"),
            (base + "tau=0", "tau must be positive"),
            (base + "decoder_hidden=0", "decoder_hidden must be a positive integer"),
            (base + "router_hidden=[]", "router_hidden must be a non-empty list"),
            (base + "budget=1", "Unsupported keys under p2g.grid_query.expert_count"),
            ("p2g.grid_query.K_max=5", "Unsupported keys under p2g.grid_query"),
            (base + "logging.interval=0", "interval must be positive"),
        )
        for override, message in cases:
            with self.subTest(override=override):
                self._rejected(message, override, variant=DYNAMIC_VARIANT_V15_2)
        # Without the view subsampling, the render itself is still pinned.
        self._rejected(
            "auxiliary_render='dense_add', got 'none'",
            base + "auxiliary_render=none", base + "auxiliary_views=null",
            variant=DYNAMIC_VARIANT_V15_2,
        )
        for key in (
            "option_counts", "seed_mode", "tau", "decoder_hidden",
            "router_hidden", "auxiliary_render",
        ):
            with self.subTest(missing=key):
                container = self._container(DYNAMIC_VARIANT_V15_2)
                del container["p2g"]["grid_query"]["expert_count"][key]
                with self.assertRaisesRegex(
                    ValueError, f"expert_count is missing keys: {key}"
                ):
                    validate_experiment_config(OmegaConf.create(container))

    def test_v15_3_count_contract_is_enforced(self):
        cases = (
            ("p2g.grid_query.K_max=3", "K_max must be 5, got 3"),
            ("p2g.grid_query.K_max=1", "K_max must be 5, got 1"),
            ("p2g.grid_query.seed_mode=medoid", "seed_mode must be 'token', got 'medoid'"),
            ("p2g.grid_query.seed_mode=range_quantile", "seed_mode must be 'token'"),
            ("p2g.grid_query.count_mode=learned_gumbel", "has no count router"),
            ("p2g.grid_query.count_mode=nested_gumbel", "V15.3 has no nested count head"),
            ("p2g.grid_query.count_mode=expert_gumbel", "V15.3 has no per-option expert head"),
            ("dynamic_2dgs.gaussian_head.decoder_hidden=0", "decoder_hidden must be a positive integer"),
            ("dynamic_2dgs.gaussian_head.decoder_hidden=576.5", "decoder_hidden must be a positive integer"),
            ("dynamic_2dgs.gaussian_head.decoder_hidden=null", "decoder_hidden must be a positive integer"),
        )
        for override, message in cases:
            with self.subTest(override=override):
                self._rejected(message, override, variant=DYNAMIC_VARIANT_V15_3)
        container = self._container(DYNAMIC_VARIANT_V15_3)
        del container["dynamic_2dgs"]["gaussian_head"]["decoder_hidden"]
        with self.assertRaisesRegex(
            ValueError, "requires dynamic_2dgs.gaussian_head.decoder_hidden"
        ):
            validate_experiment_config(OmegaConf.create(container))
        container = self._container(DYNAMIC_VARIANT_V15_3)
        del container["p2g"]["grid_query"]
        with self.assertRaisesRegex(ValueError, "has no count router"):
            validate_experiment_config(OmegaConf.create(container))

    def test_decoder_hidden_is_only_for_v15_3(self):
        for variant in DYNAMIC_VARIANTS:
            if variant in DENSE_COUNT_DYNAMIC_VARIANTS:
                continue
            with self.subTest(variant=variant):
                self._rejected(
                    "decoder_hidden is only available to V15.3",
                    "dynamic_2dgs.gaussian_head.decoder_hidden=576",
                    variant=variant,
                )

    def test_v15_2_renders_its_auxiliary_set_at_two_random_views(self):
        config, _ = self._compose(variant=DYNAMIC_VARIANT_V15_2)
        self.assertEqual(config.p2g.grid_query.expert_count.auxiliary_views, 2)
        for variant in (DYNAMIC_VARIANT_V15, DYNAMIC_VARIANT_V15_1):
            config, _ = self._compose(variant=variant)
            self.assertNotIn("auxiliary_views", config.p2g.grid_query.nested_count)
        path = "p2g.grid_query.expert_count.auxiliary_views"
        for value in ("0", "-1", "1.5", "true", "two"):
            with self.subTest(value=value):
                self._rejected(
                    "auxiliary_views must be a positive integer",
                    f"{path}={value}", variant=DYNAMIC_VARIANT_V15_2,
                )
        config, _ = self._compose(f"{path}=null", variant=DYNAMIC_VARIANT_V15_2)
        self.assertIsNone(config.p2g.grid_query.expert_count.auxiliary_views)
        # V15 may subsample its G1..G3 render too; V15.1 has none to subsample.
        config, _ = self._compose(
            "p2g.grid_query.nested_count.auxiliary_views=3",
            variant=DYNAMIC_VARIANT_V15,
        )
        self.assertEqual(config.p2g.grid_query.nested_count.auxiliary_views, 3)
        self._rejected(
            "renders no auxiliary set",
            "p2g.grid_query.nested_count.auxiliary_views=2",
            variant=DYNAMIC_VARIANT_V15_1,
        )

    def test_fresh_runs_use_tf32_sparse_convolutions(self):
        for variant in (DYNAMIC_VARIANT_V11_4, DYNAMIC_VARIANT_V15_2, DYNAMIC_VARIANT_V15_3):
            with self.subTest(variant=variant):
                config, _ = self._compose(variant=variant)
                self.assertIs(config.train.spconv_allow_tf32, True)
        config, _ = self._compose(
            "train.spconv_allow_tf32=false", variant=DYNAMIC_VARIANT_V15_3
        )
        self.assertIs(config.train.spconv_allow_tf32, False)
        self._rejected(
            "spconv_allow_tf32 must be true or false",
            "train.spconv_allow_tf32=1", variant=DYNAMIC_VARIANT_V15_3,
        )
        # A checkpoint config saved before the key existed stays FP32 and may
        # still be switched at resume, since train.* is a runtime root.
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "epoch=0.ckpt"
            stored, _ = self._compose(variant=DYNAMIC_VARIANT_V15)
            stored = OmegaConf.to_container(stored, resolve=True)
            del stored["train"]["spconv_allow_tf32"]
            torch.save({"experiment_config": stored}, checkpoint)
            resumed, _ = resolve_main_config(OmegaConf.from_dotlist([
                f"train.ckpt_path={checkpoint}",
            ]))
            self.assertIsNone(
                OmegaConf.select(resumed, "train.spconv_allow_tf32", default=None)
            )
            resumed, _ = resolve_main_config(OmegaConf.from_dotlist([
                f"train.ckpt_path={checkpoint}", "train.spconv_allow_tf32=true",
            ]))
            self.assertIs(resumed.train.spconv_allow_tf32, True)

    def test_checkpoints_protect_the_new_heads(self):
        with tempfile.TemporaryDirectory() as tmp:
            for variant, overrides in (
                (DYNAMIC_VARIANT_V15_2, (
                    "p2g.grid_query.expert_count.decoder_hidden=432",
                    "p2g.grid_query.expert_count.auxiliary_render=none",
                )),
                (DYNAMIC_VARIANT_V15_3, (
                    "p2g.grid_query.K_max=3",
                    "dynamic_2dgs.gaussian_head.decoder_hidden=432",
                )),
                (DYNAMIC_VARIANT_V15_4, (
                    "dynamic_2dgs.temporal.num_heads=12",
                    "dynamic_2dgs.temporal.rope_base=100.0",
                )),
                (DYNAMIC_VARIANT_V15_5, (
                    "model.variant=dynamic_2dgs_attention_velocity_v15_4",
                    "dynamic_2dgs.temporal.rope_base=100.0",
                )),
            ):
                checkpoint = Path(tmp) / f"{variant}.ckpt"
                stored, _ = self._compose(variant=variant)
                torch.save({
                    "experiment_config": OmegaConf.to_container(stored, resolve=True)
                }, checkpoint)
                config, _source = load_checkpoint_config(checkpoint)
                self.assertEqual(config.model.variant, variant)
                for override in overrides:
                    with self.subTest(variant=variant, override=override):
                        with self.assertRaisesRegex(ValueError, "changed roots"):
                            resolve_main_config(OmegaConf.from_dotlist([
                                f"train.ckpt_path={checkpoint}", override,
                            ]))



class V16ConfigTest(unittest.TestCase):
    """V16: V15 with two bounded ordinal gates and masked rasterization."""

    BLOCK = "p2g.grid_query.ordinal_count."

    @staticmethod
    def _compose(*overrides, variant=DYNAMIC_VARIANT_V16):
        return compose_fresh_config(OmegaConf.from_dotlist([
            f"model.variant={variant}", *overrides,
        ]))

    def _rejected(self, message, *overrides, variant=DYNAMIC_VARIANT_V16):
        with self.assertRaisesRegex(ValueError, message):
            self._compose(*overrides, variant=variant)

    def test_registration(self):
        self.assertIn(DYNAMIC_VARIANT_V16, DYNAMIC_VARIANTS)
        self.assertTrue(VARIANT_CONFIG_PATHS[DYNAMIC_VARIANT_V16].is_file())
        for group in (
            BARRIER_MATCH_DYNAMIC_VARIANTS,
            HYBRID_UTONIA_DYNAMIC_VARIANTS,
            V15_FAMILY_DYNAMIC_VARIANTS,
            LOCAL_ATTENTION_DYNAMIC_VARIANTS,
            NO_TEMPORAL_LAYER_SCALE_DYNAMIC_VARIANTS,
            POST_ACTIVATION_STEM_DYNAMIC_VARIANTS,
        ):
            self.assertIn(DYNAMIC_VARIANT_V16, group)
        for group in (
            ROUTER_CAPABLE_DYNAMIC_VARIANTS,
            NESTED_COUNT_DYNAMIC_VARIANTS,
            EXPERT_COUNT_DYNAMIC_VARIANTS,
            DENSE_COUNT_DYNAMIC_VARIANTS,
        ):
            self.assertNotIn(DYNAMIC_VARIANT_V16, group)
        self.assertEqual(
            ORDINAL_COUNT_DYNAMIC_VARIANTS,
            (DYNAMIC_VARIANT_V16, DYNAMIC_VARIANT_V16_1),
        )
        self.assertEqual(ORDINAL_COUNT_MODE, "nested_ordinal")
        self.assertEqual(ORDINAL_COUNT_OPTION_COUNTS, (1, 3, 5))
        self.assertEqual(
            ORDINAL_AUXILIARY_RENDERS, ("dense_add", "random_prefix", "none")
        )
        self.assertEqual(
            ORDINAL_AUXILIARY_RENDER_BY_VARIANT,
            {
                DYNAMIC_VARIANT_V16: "dense_add",
                DYNAMIC_VARIANT_V16_1: "random_prefix",
            },
        )

    def test_v16_is_v15_with_the_ordinal_count(self):
        v15, _ = self._compose(variant=DYNAMIC_VARIANT_V15)
        v16, _ = self._compose()
        a = V15SiblingConfigTest._leaves(OmegaConf.to_container(v15, resolve=True))
        b = V15SiblingConfigTest._leaves(OmegaConf.to_container(v16, resolve=True))
        changed = {
            key for key in set(a) | set(b)
            if a.get(key, "<absent>") != b.get(key, "<absent>")
        }
        nested = {key for key in a if key.startswith("p2g.grid_query.nested_count.")}
        ordinal = {key for key in b if key.startswith(self.BLOCK)}
        self.assertEqual(changed, {
            "model.variant", "exp_name", "logger.dir",
            "p2g.grid_query.count_mode", *nested, *ordinal,
        })
        self.assertEqual(b["p2g.grid_query.count_mode"], "nested_ordinal")
        # V15's block, with the G1..G5 render instead of G1..G3 ...
        for key in nested:
            name = key.rsplit("nested_count.", 1)[1]
            expected = "dense_add" if name == "auxiliary_render" else a[key]
            self.assertEqual(b[self.BLOCK + name], expected)
        # ... plus the gates' fixed scale and the render's view subsampling.
        self.assertEqual(
            {key.rsplit("nested_count.", 1)[1] for key in nested}
            | {"router_scale", "auxiliary_views"},
            {key.rsplit("ordinal_count.", 1)[1] for key in ordinal},
        )
        self.assertEqual(b[self.BLOCK + "router_scale"], 3.0)
        self.assertEqual(b[self.BLOCK + "auxiliary_views"], 2)

    def test_ordinal_count_contract_is_enforced(self):
        base = self.BLOCK
        cases = (
            ("p2g.grid_query.count_mode=nested_gumbel", "V16 has no nested count head"),
            ("p2g.grid_query.count_mode=expert_gumbel", "V16 has no per-option expert head"),
            ("p2g.grid_query.count_mode=legacy", "requires p2g.grid_query.count_mode='nested_ordinal'"),
            (base + "option_counts=[1,2,3]", r"option_counts must be .*reduce \{G1\}"),
            (base + "seed_mode=medoid", "seed_mode must be one of"),
            (base + "auxiliary_render=dense_keep", "auxiliary_render must be one of"),
            (base + "auxiliary_render=none", "renders no auxiliary set"),
            (base + "tau=0", "tau must be positive"),
            (base + "router_scale=0", "router_scale must be positive and finite"),
            (base + "router_scale=-3", "router_scale must be positive and finite"),
            (base + "router_scale=inf", "router_scale must be positive and finite"),
            (base + "router_scale=null", "router_scale must be positive and finite"),
            (base + "decoder_hidden=0", "decoder_hidden must be a positive integer"),
            (base + "router_hidden=[]", "router_hidden must be a non-empty list"),
            (base + "budget=1", "Unsupported keys under p2g.grid_query.ordinal_count"),
            (base + "auxiliary_views=0", "auxiliary_views must be a positive integer"),
            (base + "logging.interval=0", "interval must be positive"),
        )
        for override, message in cases:
            with self.subTest(override=override):
                self._rejected(message, override)
        # Without the view subsampling, the render itself is still pinned.
        self._rejected(
            "auxiliary_render='dense_add', got 'none'",
            base + "auxiliary_render=none", base + "auxiliary_views=null",
        )
        for key in (
            "option_counts", "seed_mode", "tau", "decoder_hidden",
            "router_hidden", "router_scale", "auxiliary_render",
        ):
            with self.subTest(missing=key):
                config, _ = self._compose()
                container = OmegaConf.to_container(config, resolve=False)
                del container["p2g"]["grid_query"]["ordinal_count"][key]
                with self.assertRaisesRegex(
                    ValueError, f"ordinal_count is missing keys: {key}"
                ):
                    validate_experiment_config(OmegaConf.create(container))
        # The ordinal count is V16's and V16.1's alone, and its scale only its
        # own key.
        for variant in DYNAMIC_VARIANTS:
            if variant in ORDINAL_COUNT_DYNAMIC_VARIANTS:
                continue
            with self.subTest(variant=variant):
                self._rejected(
                    "only available to V16, V16.1",
                    "p2g.grid_query.count_mode=nested_ordinal",
                    variant=variant,
                )
        self._rejected(
            "Unsupported keys under p2g.grid_query.nested_count: router_scale",
            "p2g.grid_query.nested_count.router_scale=3.0",
            variant=DYNAMIC_VARIANT_V15,
        )

    def test_checkpoints_protect_the_ordinal_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "v16.ckpt"
            stored, _ = self._compose()
            torch.save({
                "experiment_config": OmegaConf.to_container(stored, resolve=True)
            }, checkpoint)
            config, _source = load_checkpoint_config(checkpoint)
            self.assertEqual(config.model.variant, DYNAMIC_VARIANT_V16)
            for override in (
                self.BLOCK + "router_scale=2.0",
                self.BLOCK + "tau=0.5",
                self.BLOCK + "auxiliary_views=null",
            ):
                with self.subTest(override=override):
                    with self.assertRaisesRegex(ValueError, "changed roots"):
                        resolve_main_config(OmegaConf.from_dotlist([
                            f"train.ckpt_path={checkpoint}", override,
                        ]))



class V16_1ConfigTest(unittest.TestCase):
    """V16.1: V16 whose second render is a uniformly random prefix per token."""

    BLOCK = "p2g.grid_query.ordinal_count."

    @staticmethod
    def _compose(*overrides, variant=DYNAMIC_VARIANT_V16_1):
        return compose_fresh_config(OmegaConf.from_dotlist([
            f"model.variant={variant}", *overrides,
        ]))

    def _rejected(self, message, *overrides, variant=DYNAMIC_VARIANT_V16_1):
        with self.assertRaisesRegex(ValueError, message):
            self._compose(*overrides, variant=variant)

    def test_registration(self):
        self.assertIn(DYNAMIC_VARIANT_V16_1, DYNAMIC_VARIANTS)
        self.assertTrue(VARIANT_CONFIG_PATHS[DYNAMIC_VARIANT_V16_1].is_file())
        for group in (
            BARRIER_MATCH_DYNAMIC_VARIANTS,
            HYBRID_UTONIA_DYNAMIC_VARIANTS,
            V15_FAMILY_DYNAMIC_VARIANTS,
            LOCAL_ATTENTION_DYNAMIC_VARIANTS,
            NO_TEMPORAL_LAYER_SCALE_DYNAMIC_VARIANTS,
            POST_ACTIVATION_STEM_DYNAMIC_VARIANTS,
            ORDINAL_COUNT_DYNAMIC_VARIANTS,
        ):
            self.assertIn(DYNAMIC_VARIANT_V16_1, group)
        for group in (
            ROUTER_CAPABLE_DYNAMIC_VARIANTS,
            NESTED_COUNT_DYNAMIC_VARIANTS,
            EXPERT_COUNT_DYNAMIC_VARIANTS,
            DENSE_COUNT_DYNAMIC_VARIANTS,
        ):
            self.assertNotIn(DYNAMIC_VARIANT_V16_1, group)

    def test_v16_1_is_v16_with_the_random_prefix_render(self):
        v16, _ = self._compose(variant=DYNAMIC_VARIANT_V16)
        v16_1, _ = self._compose()
        a = V15SiblingConfigTest._leaves(OmegaConf.to_container(v16, resolve=True))
        b = V15SiblingConfigTest._leaves(OmegaConf.to_container(v16_1, resolve=True))
        changed = {
            key for key in set(a) | set(b)
            if a.get(key, "<absent>") != b.get(key, "<absent>")
        }
        self.assertEqual(changed, {
            "model.variant", "exp_name", "logger.dir",
            self.BLOCK + "auxiliary_render",
        })
        self.assertEqual(b[self.BLOCK + "auxiliary_render"], "random_prefix")
        self.assertEqual(b[self.BLOCK + "auxiliary_views"], 2)

    def test_each_ordinal_variant_pins_its_auxiliary_render(self):
        render = self.BLOCK + "auxiliary_render"
        self._rejected(
            "auxiliary_render='random_prefix', got 'dense_add'",
            render + "=dense_add",
        )
        self._rejected(
            "auxiliary_render='dense_add', got 'random_prefix'",
            render + "=random_prefix", variant=DYNAMIC_VARIANT_V16,
        )
        self._rejected("renders no auxiliary set", render + "=none")
        self._rejected("auxiliary_render must be one of", render + "=dense_keep")
        # The random prefix is the ordinal block's render only.
        self._rejected(
            "auxiliary_render must be one of",
            "p2g.grid_query.nested_count.auxiliary_render=random_prefix",
            variant=DYNAMIC_VARIANT_V15,
        )

    def test_checkpoints_protect_the_random_prefix_render(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "v16_1.ckpt"
            stored, _ = self._compose()
            torch.save({
                "experiment_config": OmegaConf.to_container(stored, resolve=True)
            }, checkpoint)
            config, _source = load_checkpoint_config(checkpoint)
            self.assertEqual(config.model.variant, DYNAMIC_VARIANT_V16_1)
            self.assertEqual(
                config.p2g.grid_query.ordinal_count.auxiliary_render,
                "random_prefix",
            )
            with self.assertRaisesRegex(ValueError, "changed roots"):
                resolve_main_config(OmegaConf.from_dotlist([
                    f"train.ckpt_path={checkpoint}",
                    self.BLOCK + "auxiliary_render=dense_add",
                ]))


if __name__ == "__main__":
    unittest.main()
