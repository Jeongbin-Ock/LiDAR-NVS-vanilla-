from __future__ import annotations

import tempfile
import unittest
import math
from pathlib import Path

import torch
from omegaconf import OmegaConf

from src.config_loader import (
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
    DYNAMIC_VARIANT_V13,
    DYNAMIC_VARIANT_V13_1,
    DYNAMIC_VARIANT_V13_2,
    DYNAMIC_VARIANT_V13_3,
    DYNAMIC_VARIANT_V13_4,
    compose_fresh_config,
    load_checkpoint_config,
    resolve_eval_config,
    resolve_main_config,
)

# Opts a barrier-family overlay into the random-duration loader, the contract
# the V11.6 overlay ships with. Its holdouts live under dynamic_window, so the
# legacy loader's data.train_excluded_scenes has to be cleared.
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
        self.assertEqual(list(config.device), [0])
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
        self.assertIsNone(count.exp)

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
        self.assertIsNone(right_count["exp"])

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
        self.assertIsNone(right_count["exp"])

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
        self.assertIsNone(right_count["exp"])
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
        from src.models_new.module.dynamic.heads import GaussianAttributeHead

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
        from src.models_new.module.dynamic.heads import GaussianAttributeHead

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
        """K_max is how many Gaussians each token becomes; exp seeds them."""
        for variant in (
            DYNAMIC_VARIANT_V11_1, DYNAMIC_VARIANT_V11_3, DYNAMIC_VARIANT_V11_5,
        ):
            for k_max, exp in ((1, 1), (2, 1), (2, 2), (4, None)):
                with self.subTest(variant=variant, K_max=k_max, exp=exp):
                    config, _source = compose_fresh_config(
                        OmegaConf.from_dotlist([
                            f"model.variant={variant}",
                            f"p2g.grid_query.K_max={k_max}",
                            f"p2g.grid_query.exp={'null' if exp is None else exp}",
                        ])
                    )
                    self.assertEqual(config.p2g.grid_query.K_max, k_max)
                    self.assertEqual(config.p2g.grid_query.exp, exp)

    def test_absent_grid_query_is_one_gaussian_per_token(self):
        config, _source = compose_fresh_config(OmegaConf.from_dotlist([
            f"model.variant={DYNAMIC_VARIANT_V11_1}",
        ]))
        self.assertIsNone(OmegaConf.select(config, "p2g.grid_query"))

    def test_fixed_gaussian_count_rejects_impossible_settings(self):
        invalid = (
            ("p2g.grid_query.K_max=0", "positive"),
            ("p2g.grid_query.K_max=1 p2g.grid_query.exp=2", "K_max >= 2"),
            ("p2g.grid_query.K_max=2 p2g.grid_query.exp=3", "null, 1, or 2"),
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
        self.assertEqual(config.loss.w_scale, 0.0)
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


if __name__ == "__main__":
    unittest.main()
