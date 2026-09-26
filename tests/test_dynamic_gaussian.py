from __future__ import annotations

import json
import math
import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from src.dataloader.nuscene import relative_time_coordinates
from src.eval.gaussian_viz import (
    SOURCE_FRAME_COLORS,
    _source_frame_velocity_layer,
    _velocity_arrow_trace,
    save_sequence_html,
    velocity_transport_from_output,
)
from src.models_new.module.dynamic_gaussian import (
    CosineRoPELayerWeightedCrossAttention,
    CosineRoPEVelocityGaussianBackend,
    DynamicGaussianBackend,
    EmbeddedInitVelocityHead,
    FinalFeatureBarrierCrossAttention,
    FinalFeatureBarrierVelocityGaussianBackend,
    FixedMotionGainLearnableMaxSpeedBarrierCrossAttention,
    FixedMotionGainMaxSpeedBarrierVelocityGaussianBackend,
    FixedMotionHeadRMSNorm,
    GaussianAttributeHead,
    LayerwiseSoftplusMaxSpeedBarrierCrossAttention,
    LayerwiseSoftplusMaxSpeedBarrierVelocityGaussianBackend,
    LearnableMaxSpeedBarrierLayerWeightedCrossAttention,
    LearnableMaxSpeedBarrierVelocityGaussianBackend,
    MaxSpeedBarrierLayerWeightedCrossAttention,
    MaxSpeedBarrierVelocityGaussianBackend,
    GroupedHeadRMSNorm,
    GroupedGainBarrierCrossAttention,
    SingleGaussianBarrierVelocityGaussianBackend,
    SeedConditionedCosineRoPEVelocityGaussianBackend,
    RawQKRoPELayerWeightedCrossAttention,
    RawQKRoPEVelocityGaussianBackend,
    RMSNormFeatureRoPELayerWeightedCrossAttention,
    RMSNormFeatureRoPEVelocityGaussianBackend,
    RMSNormRoPELayerWeightedCrossAttention,
    RMSNormRoPEVelocityGaussianBackend,
)
from src.models_new.module.m3_g2p import DynamicGausRender
from src.models_new.utils.attention import Rotary3D
from src.models_new.utils.loss import Loss
from src.models_new.utonia.model import Point3DRoPE
from src.model_wrapper import ModelWrapper


def _gs_params():
    return OmegaConf.create({
        "shs": 32, "opacity": 1, "scaling": 2, "rotation": 4, "offset": 3,
    })


def backend_cfg(dim=24):
    return OmegaConf.create({
        "temporal": OmegaConf.to_container(v11_temporal_cfg(dim=dim), resolve=True),
        "gaussian_head": {
            "initial_opacity": 0.2,
            "initial_scale_m": 0.3,
        },
        "motion": {
            "zero_init": True,
        },
    })


def router_count_cfg():
    return OmegaConf.create({
        "count_mode": "learned_gumbel",
        "grad_balance": "sqrt_k",
        "learned_count": {
            "K_max": 3,
            "tau": 1.0,
            "seed_mode": "range_quantile",
            "grad_balance_scope": "token",
        },
    })

def adaptive_seed_bank(position):
    bank = position.new_zeros(position.shape[0], 3, 3, 3)
    delta = torch.zeros_like(bank)
    choices = (
        torch.tensor([[0.0, 0.0, 0.0]]),
        torch.tensor([[-0.1, 0.0, 0.0], [0.1, 0.0, 0.0]]),
        torch.tensor([
            [-0.2, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.2, 0.0, 0.0],
        ]),
    )
    for index, offset in enumerate(choices):
        k = index + 1
        offset = offset.to(device=position.device, dtype=position.dtype)
        bank[:, index, :k] = position[:, None, :] + offset[None]
        delta[:, index, :k] = offset[None]
    return bank, delta


def v11_temporal_cfg(dim=24, heads=4, layers=3, match_chunk_size=2):
    del dim
    return OmegaConf.create({
        "implementation": "auto",
        "layers": layers,
        "num_heads": heads,
        "mlp_ratio": 2,
        "use_time_embedding": True,
        "time_embedding_dim": 8,
        "time_frequencies": 4,
        "time_reference_sec": 1.0,
        "position_encoding": "barrier_rope_split",
        "barrier_speed_mps": 30.0,
        "barrier_weight": 4.0,
        "match_chunk_size": match_chunk_size,
        "rope_base": 100.0,
        "rope_position_scale": 2.0 * math.pi / 5.0,
        "qk_norm": True,
        "layer_scale_init": 0.1,
    })


def v11_backend_cfg(dim=24):
    cfg = backend_cfg(dim=dim)
    cfg.temporal = v11_temporal_cfg(dim=dim)
    cfg.motion = OmegaConf.create({
        "zero_init": True,
        "residual_hidden_dim": 8,
    })
    return cfg


def v11_4_backend_cfg(dim=24):
    """V11's backend config under V11.4's linear hinge."""
    cfg = v11_backend_cfg(dim=dim)
    cfg.temporal = v12_temporal_cfg(dim=dim)
    return cfg


def v12_temporal_cfg(dim=24, heads=4, layers=3, match_chunk_size=2):
    cfg = v11_temporal_cfg(
        dim=dim, heads=heads, layers=layers,
        match_chunk_size=match_chunk_size,
    )
    cfg.barrier_weight = 1.0
    cfg.barrier_exponent = 1
    return cfg


def v12_backend_cfg(dim=24):
    cfg = v11_backend_cfg(dim=dim)
    cfg.temporal = v12_temporal_cfg(dim=dim)
    return cfg


def v12_1_temporal_cfg(dim=24, heads=4, layers=3, match_chunk_size=2):
    cfg = v12_temporal_cfg(
        dim=dim, heads=heads, layers=layers,
        match_chunk_size=match_chunk_size,
    )
    cfg.barrier_lambda_init = 1.0
    return cfg


def v12_1_backend_cfg(dim=24):
    cfg = v12_backend_cfg(dim=dim)
    cfg.temporal = v12_1_temporal_cfg(dim=dim)
    return cfg


def v13_temporal_cfg(dim=24, heads=4, layers=3):
    del dim
    return OmegaConf.create({
        "implementation": "auto",
        "layers": layers,
        "num_heads": heads,
        "mlp_ratio": 2,
        "use_time_embedding": True,
        "time_embedding_dim": 8,
        "time_frequencies": 4,
        "time_reference_sec": 1.0,
        "position_encoding": "rope_all_heads",
        "rope_base": 100.0,
        "rope_position_scale": 2.0 * math.pi / 5.0,
        "qk_norm": False,
        "motion_temperature": 0.07,
        "layer_scale_init": 0.1,
    })


def v13_backend_cfg(dim=24):
    cfg = backend_cfg(dim=dim)
    cfg.temporal = v13_temporal_cfg(dim=dim)
    cfg.motion = OmegaConf.create({
        "zero_init": True,
        "residual_hidden_dim": 8,
        "velocity_embedding_dim": 8,
        "detach_init_condition": True,
    })
    return cfg


def v13_2_temporal_cfg(dim=24, heads=4, layers=3):
    cfg = v13_temporal_cfg(dim=dim, heads=heads, layers=layers)
    del cfg.motion_temperature
    return cfg


def v13_2_backend_cfg(dim=24):
    cfg = v13_backend_cfg(dim=dim)
    cfg.temporal = v13_2_temporal_cfg(dim=dim)
    return cfg


def v13_3_temporal_cfg(dim=24, heads=4, layers=3):
    cfg = v13_2_temporal_cfg(dim=dim, heads=heads, layers=layers)
    cfg.qk_norm = True
    return cfg


def v13_3_backend_cfg(dim=24):
    cfg = v13_backend_cfg(dim=dim)
    cfg.temporal = v13_3_temporal_cfg(dim=dim)
    return cfg


def v13_4_temporal_cfg(dim=24, heads=4, layers=3):
    cfg = v13_3_temporal_cfg(dim=dim, heads=heads, layers=layers)
    cfg.position_encoding = "rope_except_head0"
    return cfg


def v13_4_backend_cfg(dim=24):
    cfg = v13_backend_cfg(dim=dim)
    cfg.temporal = v13_4_temporal_cfg(dim=dim)
    return cfg


def v11_2_temporal_cfg(dim=24, heads=4, layers=3, match_chunk_size=2):
    return v11_temporal_cfg(
        dim=dim, heads=heads, layers=layers,
        match_chunk_size=match_chunk_size,
    )


def v11_2_backend_cfg(dim=24):
    return v11_backend_cfg(dim=dim)


def v11_1_temporal_cfg(dim=24, heads=4, layers=3, match_chunk_size=2):
    # V11.1's temporal config is V11's; only the QK gain's shape differs, and
    # that is a module decision rather than a configured one.
    return v11_temporal_cfg(
        dim=dim, heads=heads, layers=layers,
        match_chunk_size=match_chunk_size,
    )


def v11_1_backend_cfg(dim=24):
    cfg = v11_backend_cfg(dim=dim)
    cfg.temporal = v11_1_temporal_cfg(dim=dim)
    cfg.motion.velocity_embedding_dim = 8
    cfg.motion.detach_init_condition = True
    return cfg


class DynamicGaussianTest(unittest.TestCase):
    def test_training_loss_skips_disabled_metrics_and_chamfer(self):
        loss = Loss(OmegaConf.create({
            "w_chamfer": 0.0,
            "w_depth": 0.5,
            "w_depth_median": 0.5,
            "w_intensity": 0.2,
            "w_raydrop": 0.5,
        }))
        maps = {
            "depth": torch.ones(1, 1, 2, 2),
            "depth_median": torch.ones(1, 1, 2, 2),
            "intensity_sh": torch.full((1, 1, 2, 2), 0.5),
            "raydrop": torch.full((1, 1, 2, 2), 0.25),
            "gt_depth": torch.ones(1, 1, 2, 2),
            "gt_intensity_sh": torch.full((1, 1, 2, 2), 0.5),
            "gt_raydrop": torch.zeros(1, 1, 2, 2),
            "render_points": [],
            "gt_points": [],
        }
        output = loss(
            maps,
            metric_mode="train",
            compute_valid_metrics=False,
            compute_raydrop_metrics=False,
        )
        self.assertEqual(set(output), {
            "loss_depth", "loss_depth_median", "loss_intensity",
            "loss_raydrop", "loss_scale", "total",
        })

    @staticmethod
    def _prior_wrapper(init_weight=0.01, offset_weight=0.01, step=0):
        return SimpleNamespace(
            _velocity_l2_cfg=OmegaConf.create({
                "enabled": True,
                "mode": "split_group_l2",
                "init_weight": init_weight,
                "offset_weight": offset_weight,
                "warmup_steps": 0,
                "ramp_steps": 0,
            }),
            global_step=step,
            _active_motion_items=ModelWrapper._active_motion_items,
        )

    @classmethod
    def _apply_split_velocity_l2(
        cls, inits, offsets, *, prefix="train", init_weight=0.01,
        offset_weight=0.01, step=0,
    ):
        wrapper = cls._prior_wrapper(init_weight, offset_weight, step)
        losses = {"total": torch.zeros(())}
        items = [
            {
                "velocity": init + offset,
                "velocity_init": init,
                "velocity_offset": offset,
            }
            for init, offset in zip(inits, offsets)
        ]
        ModelWrapper._add_velocity_l2_prior(
            wrapper, losses, items, prefix=prefix
        )
        return losses

    def test_split_group_l2_charges_init_and_offset_independently(self):
        velocity_init = torch.tensor([[3.0, 4.0, 0.0]], requires_grad=True)
        velocity_offset = torch.tensor([[0.0, 0.0, 2.0]], requires_grad=True)
        losses = self._apply_split_velocity_l2(
            [velocity_init], [velocity_offset],
            init_weight=0.01, offset_weight=0.02,
        )
        self.assertAlmostEqual(
            float(losses["loss_velocity_l2_init"]), 25.0, places=5
        )
        self.assertAlmostEqual(
            float(losses["loss_velocity_l2_offset"]), 4.0, places=5
        )
        # loss_velocity_l2 stays raw and unweighted, as in every other mode;
        # only `total` sees init_weight/offset_weight.
        self.assertAlmostEqual(
            float(losses["loss_velocity_l2"]), 25.0 + 4.0, places=5
        )
        self.assertAlmostEqual(
            float(losses["total"]), 0.01 * 25.0 + 0.02 * 4.0, places=6
        )
        losses["total"].backward()
        torch.testing.assert_close(
            velocity_init.grad, torch.tensor([[0.06, 0.08, 0.0]])
        )
        torch.testing.assert_close(
            velocity_offset.grad, torch.tensor([[0.0, 0.0, 0.08]])
        )

    def test_split_group_l2_has_no_cancellation_cross_term(self):
        """An offset that merely negates v_init must not be free."""
        opposed = torch.tensor([[-3.0, -4.0, 0.0]])
        aligned = torch.tensor([[3.0, 4.0, 0.0]])
        init = torch.tensor([[3.0, 4.0, 0.0]])
        split_opposed = self._apply_split_velocity_l2(
            [init], [opposed], init_weight=0.01, offset_weight=0.01
        )
        split_aligned = self._apply_split_velocity_l2(
            [init], [aligned], init_weight=0.01, offset_weight=0.01
        )
        # Both offsets have the same magnitude, so the split prior is blind to
        # the sign; only the render loss decides direction.
        self.assertAlmostEqual(
            float(split_opposed["total"]), float(split_aligned["total"]),
            places=6,
        )
        # A squared-norm prior on the sum instead rewards the cancelling
        # offset: ||v_init + v_offset||^2 is exactly zero for it.
        total_opposed = (init + opposed).square().sum(dim=-1).mean()
        total_aligned = (init + aligned).square().sum(dim=-1).mean()
        self.assertLess(float(total_opposed), float(total_aligned))
        self.assertAlmostEqual(float(total_opposed), 0.0, places=6)

    def test_split_group_l2_reports_but_does_not_optimize_validation(self):
        losses = self._apply_split_velocity_l2(
            [torch.tensor([[3.0, 4.0, 0.0]])],
            [torch.tensor([[1.0, 0.0, 0.0]])],
            prefix="val",
        )
        self.assertIn("loss_velocity_l2_init", losses)
        self.assertIn("loss_velocity_l2_offset", losses)
        self.assertNotIn("wc_velocity_l2", losses)
        self.assertEqual(float(losses["total"]), 0.0)

    def test_split_group_l2_requires_both_velocity_fields(self):
        wrapper = self._prior_wrapper()
        losses = {"total": torch.zeros(())}
        with self.assertRaisesRegex(ValueError, "velocity_init"):
            ModelWrapper._add_velocity_l2_prior(
                wrapper,
                losses,
                [{"velocity": torch.zeros(1, 3)}],
                prefix="train",
            )

    @staticmethod
    def _routed_items(token_fields, selected_k, sample_tokens):
        """Adaptive-K items: every token copied once per selected slot."""
        items = []
        for tokens in sample_tokens:
            anchor = torch.cat([
                torch.full((selected_k[t],), t, dtype=torch.long)
                for t in tokens
            ])
            slot = torch.cat([torch.arange(selected_k[t]) for t in tokens])
            item = {
                name: value[anchor] for name, value in token_fields.items()
            }
            item["gaussian_slot_index"] = slot
            items.append(item)
        return items

    def test_split_group_l2_counts_each_routed_token_once(self):
        init = torch.tensor(
            [[3.0, 4.0, 0.0], [0.0, 0.0, 1.0]], requires_grad=True
        )
        offset = torch.tensor(
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], requires_grad=True
        )
        items = self._routed_items(
            {
                "velocity": init + offset,
                "velocity_init": init,
                "velocity_offset": offset,
            },
            [1, 3],
            [[0, 1]],
        )
        wrapper = self._prior_wrapper(init_weight=0.01, offset_weight=0.02)
        losses = {"total": torch.zeros(())}
        ModelWrapper._add_velocity_l2_prior(
            wrapper, losses, items, prefix="train"
        )
        # Token means, where Gaussian means would be 7.0 and 3.0.
        self.assertAlmostEqual(
            float(losses["loss_velocity_l2_init"]), 13.0, places=5
        )
        self.assertAlmostEqual(
            float(losses["loss_velocity_l2_offset"]), 2.0, places=5
        )
        losses["total"].backward()
        torch.testing.assert_close(init.grad, 0.01 * init.detach())
        torch.testing.assert_close(offset.grad, 0.02 * offset.detach())

    def test_velocity_statistics_count_each_routed_token_once(self):
        velocity = torch.tensor([[3.0, 4.0, 0.0], [0.0, 0.0, 1.0]])
        items = self._routed_items(
            {
                "velocity": velocity,
                "velocity_init": velocity,
                "velocity_offset": torch.zeros_like(velocity),
            },
            [3, 1],
            [[0, 1]],
        )
        logged = {}

        def log(name, value, **kwargs):
            logged[name] = (float(value), kwargs["batch_size"])

        wrapper = SimpleNamespace(
            _active_motion_items=ModelWrapper._active_motion_items,
            _motion_statistics=ModelWrapper._motion_statistics,
            log=log,
        )
        ModelWrapper._log_velocity_statistics(wrapper, items, prefix="train")
        # Speeds 5 and 1 m/s: a token mean of 3.0, where the Gaussian mean
        # would be (3 * 5 + 1) / 4 = 4.0.
        for name in ("velocity_speed_mean", "velocity_init_norm_mean"):
            value, batch_size = logged[f"train/dynamic/{name}"]
            self.assertAlmostEqual(value, 3.0, places=5)
            self.assertEqual(batch_size, 2)

    def test_irregular_pair_keeps_normalized_and_physical_time(self):
        normalized, seconds, duration = relative_time_coordinates(
            [1_000_000, 1_400_000, 1_870_000],
            1_000_000,
            1_870_000,
        )
        self.assertTrue(torch.allclose(seconds, torch.tensor([0.0, 0.4, 0.87])))
        self.assertAlmostEqual(float(duration), 0.87, places=6)
        self.assertTrue(torch.allclose(
            normalized,
            torch.tensor([0.0, 0.4 / 0.87, 1.0]),
            atol=1.0e-6,
        ))

    def test_normalized_time_is_bit_exact_with_legacy_formula(self):
        timestamps_us = [1_000_003, 1_317_219, 1_998_641]
        normalized, _seconds, _duration = relative_time_coordinates(
            timestamps_us,
            timestamps_us[0],
            timestamps_us[-1],
        )
        span_us = timestamps_us[-1] - timestamps_us[0]
        legacy = torch.tensor(
            [(timestamp - timestamps_us[0]) / span_us for timestamp in timestamps_us],
            dtype=torch.float32,
        )
        self.assertTrue(torch.equal(normalized, legacy))

    def test_parallel_cross_attention_is_frame_swap_equivariant(self):
        torch.manual_seed(7)
        module = MaxSpeedBarrierLayerWeightedCrossAttention(
            v11_temporal_cfg(), dim=24
        ).eval()
        feat0 = torch.randn(3, 24)
        feat1 = torch.randn(2, 24)
        pos0 = torch.randn(3, 3)
        pos1 = torch.randn(2, 3)
        duration = torch.full((5,), 0.87)
        out, _fields = module(
            torch.cat([feat0, feat1]),
            torch.cat([pos0, pos1]),
            torch.tensor([3, 5]),
            torch.tensor([0, 0]),
            torch.tensor([0.0, 0.0, 0.0, 0.87, 0.87]),
            duration,
        )
        swapped, _fields = module(
            torch.cat([feat1, feat0]),
            torch.cat([pos1, pos0]),
            torch.tensor([2, 5]),
            torch.tensor([0, 0]),
            torch.tensor([0.87, 0.87, 0.0, 0.0, 0.0]),
            duration,
        )
        restored = torch.cat([swapped[2:], swapped[:2]])
        self.assertTrue(torch.allclose(out, restored, atol=2.0e-5, rtol=2.0e-5))

    def test_fixed_reference_time_preserves_irregular_pair_duration(self):
        backend = SingleGaussianBarrierVelocityGaussianBackend(
            v11_1_backend_cfg(), _gs_params(), dim=24, offset_bound=0.8
        )
        geometry = {
            "time_sec": torch.tensor([0.0, 0.8, 0.0, 1.0]),
        }
        coordinate = backend._temporal_coordinate(geometry)
        self.assertTrue(torch.equal(
            coordinate,
            torch.tensor([0.0, 0.8, 0.0, 1.0]),
        ))

    def test_dynamic_rope_is_numerically_equal_to_active_utonia_rope(self):
        torch.manual_seed(13)
        metric_position = torch.randn(7, 3) * 20.0
        q = torch.randn(7, 2, 24)
        k = torch.randn(7, 2, 24)

        utonia = Point3DRoPE(head_dim=24, base=10.0)
        expected_q, expected_k = utonia(q, k, metric_position * 0.2)

        dynamic = Rotary3D(head_dim=24, base=10.0, position_scale=0.2)
        angles = dynamic.angles(metric_position)
        actual_q = dynamic.rotate(q, angles[:, None])
        actual_k = dynamic.rotate(k, angles[:, None])
        self.assertTrue(torch.allclose(actual_q, expected_q, atol=1.0e-6))
        self.assertTrue(torch.allclose(actual_k, expected_k, atol=1.0e-6))

    def _backend_output(self, gaussians_per_token=1):
        torch.manual_seed(11)
        gs_params = OmegaConf.create({
            "shs": 32,
            "opacity": 1,
            "scaling": 2,
            "rotation": 4,
            "offset": 3,
        })
        backend = SingleGaussianBarrierVelocityGaussianBackend(
            v11_1_backend_cfg(), gs_params, dim=24, offset_bound=0.8,
            gaussians_per_token=gaussians_per_token,
        )
        feature = torch.randn(5, 24)
        token = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [1.1, 0.0, 0.0],
        ])
        seed = (token + torch.tensor([0.02, 0.01, 0.0])).unsqueeze(1).expand(
            -1, gaussians_per_token, -1
        ).contiguous()
        pose = [torch.eye(4).repeat(2, 1, 1)]
        output = backend(
            feature,
            token,
            seed,
            torch.tensor([3, 5]),
            torch.tensor([0, 0]),
            pose,
            [torch.tensor([0.0, 1.0])],
            [torch.tensor([0.0, 0.87])],
            torch.tensor([0.87]),
        )
        return backend, output

    def test_backend_emits_one_gaussian_per_token_at_its_seed(self):
        _backend, output = self._backend_output()
        item = output["batch_gaussians"][0]
        self.assertEqual(item["position"].shape, (5, 3))
        self.assertTrue(torch.allclose(item["position"], torch.tensor([
            [0.02, 0.01, 0.0],
            [1.02, 0.01, 0.0],
            [2.02, 0.01, 0.0],
            [0.12, 0.01, 0.0],
            [1.12, 0.01, 0.0],
        ])))
        self.assertEqual(item["shs"].shape, (5, 32))
        self.assertEqual(item["rotation"].shape, (5, 4))
        self.assertTrue(torch.allclose(
            item["source_time_sec"], torch.tensor([0.0, 0.0, 0.0, 0.87, 0.87])
        ))
        # The zero-initialized offset starts every token at v_init.
        self.assertTrue(torch.equal(
            item["velocity_offset"], torch.zeros_like(item["velocity_offset"])
        ))
        self.assertTrue(torch.equal(item["velocity"], item["velocity_init"]))
        self.assertEqual(int(output["batch"].numel()), 5)

    def test_config_sets_how_many_gaussians_a_token_becomes(self):
        """p2g.grid_query.K_max is the per-token Gaussian count, not a cap."""
        backend, output = self._backend_output(gaussians_per_token=3)
        item = output["batch_gaussians"][0]
        self.assertEqual(backend.gaussians_per_token, 3)
        self.assertEqual(item["position"].shape, (15, 3))
        self.assertEqual(item["shs"].shape, (15, 32))
        self.assertEqual(item["rotation"].shape, (15, 4))
        self.assertEqual(int(output["batch"].numel()), 15)
        # One head per attribute, three slots wide.
        self.assertEqual(backend.gaussian_head.heads["shs"].out_features, 96)

    def test_slot_rows_stay_token_major_so_per_token_fields_line_up(self):
        _backend, output = self._backend_output(gaussians_per_token=3)
        item = output["batch_gaussians"][0]
        # Token 0-2 come from frame 0 at t=0; tokens 3-4 from frame 1 at 0.87.
        self.assertTrue(torch.allclose(
            item["source_time_sec"],
            torch.tensor([0.0] * 9 + [0.87] * 6),
        ))
        velocity = item["velocity"].reshape(5, 3, 3)
        self.assertTrue(torch.allclose(
            velocity, velocity[:, :1].expand_as(velocity)
        ))

    def test_identical_seeds_separate_through_independent_slot_offsets(self):
        """Every slot starts at the same seed but owns its own offset rows."""
        backend, output = self._backend_output(gaussians_per_token=3)
        offset_head = backend.gaussian_head.heads["offset"]
        self.assertEqual(offset_head.out_features, 9)
        position = output["batch_gaussians"][0]["position"].reshape(5, 3, 3)
        # Zero-initialized offsets leave all three slots on the shared seed.
        self.assertTrue(torch.allclose(
            position, position[:, :1].expand_as(position)
        ))
        # Once a slot's own rows move, only that slot's Gaussian moves.
        with torch.no_grad():
            offset_head.bias.copy_(torch.tensor(
                [0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 0.0]
            ))
        _raw, offset, _feature = backend.gaussian_head(torch.zeros(2, 24))
        offset = offset.reshape(2, 3, 3)
        self.assertTrue(torch.allclose(offset[:, 0], torch.zeros(2, 3)))
        self.assertFalse(torch.allclose(offset[:, 1], offset[:, 2]))

    def test_every_slot_starts_at_the_identity_quaternion(self):
        gs_params = OmegaConf.create({
            "shs": 4, "opacity": 1, "scaling": 2, "rotation": 4, "offset": 3,
        })
        head = GaussianAttributeHead(
            None, gs_params, dim=8, offset_bound=0.8, gaussians_per_token=3
        )
        self.assertTrue(torch.equal(
            head.heads["rotation"].bias,
            torch.tensor([1.0, 0.0, 0.0, 0.0] * 3),
        ))

    def test_signed_endpoint_interval_gives_both_frames_forward_velocity(self):
        geometry = {
            "local_frame": torch.tensor([0, 1]),
            "duration_sec": torch.tensor([1.0, 1.0]),
        }
        pair_delta_t = DynamicGaussianBackend._signed_pair_delta_t(geometry)
        self.assertTrue(torch.equal(pair_delta_t, torch.tensor([1.0, -1.0])))
        # Frame 0 sees its partner 2 m ahead, frame 1 sees it 2 m behind; both
        # divide by their own signed interval into one forward-time velocity.
        delta_p = torch.tensor([[2.0, 0.0, 0.0], [-2.0, 0.0, 0.0]])
        velocity = delta_p / pair_delta_t.unsqueeze(-1)
        self.assertTrue(torch.equal(
            velocity, torch.tensor([[2.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        ))
        item = {
            "position": torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
            "velocity": velocity,
            "source_time_sec": torch.tensor([0.0, 1.0]),
        }
        midpoint = DynamicGausRender(OmegaConf.create({})).get_means3D(item, 0.5)
        self.assertTrue(torch.equal(
            midpoint,
            torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        ))

    def test_physical_seconds_transport_and_velocity_gradient(self):
        backend, output = self._backend_output()
        item = output["batch_gaussians"][0]
        renderer = DynamicGausRender(OmegaConf.create({}))
        means = renderer.get_means3D(item, 0.4)
        delta_t = 0.4 - item["source_time_sec"]
        self.assertTrue(torch.allclose(
            means, item["position"] + item["velocity"] * delta_t.unsqueeze(-1)
        ))

        target = item["position"].detach() + torch.tensor([0.4, 0.0, 0.0])
        loss = (renderer.get_means3D(item, 0.4) - target).square().mean()
        loss.backward()
        grad = backend.velocity_head.velocity.weight.grad
        self.assertIsNotNone(grad)
        self.assertGreater(float(grad.abs().sum()), 0.0)

    def test_renderer_declares_transport_contract(self):
        self.assertEqual(DynamicGausRender.transport_mode, "velocity")

    def test_velocity_viz_uses_actual_signed_source_to_target_time(self):
        renderer = DynamicGausRender(OmegaConf.create({}))
        item = {
            "position": torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            "velocity": torch.tensor([[2.0, 0.0, 0.0], [0.0, -1.0, 0.5]]),
            "source_time_sec": torch.tensor([0.0, 1.0]),
        }
        transport = velocity_transport_from_output(renderer, item, 0.5)
        np.testing.assert_allclose(
            transport["target_centers"],
            np.array([[2.0, 2.0, 3.0], [4.0, 5.5, 5.75]], np.float32),
        )
        np.testing.assert_allclose(
            transport["source_centers"], item["position"].numpy()
        )
        np.testing.assert_allclose(
            transport["velocity_mps"], item["velocity"].numpy()
        )
        np.testing.assert_allclose(transport["delta_t_sec"], [0.5, -0.5])
        np.testing.assert_allclose(transport["source_times_sec"], [0.0, 1.0])
        self.assertEqual(len(transport["target_centers_by_source_frame"]), 2)

    def test_velocity_viz_arrow_encodes_actual_signed_travel_time(self):
        trace = _velocity_arrow_trace(
            np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], np.float32),
            np.array([[4.0, -2.0, 0.0], [0.0, -1.0, 0.5]], np.float32),
            delta_t_sec=np.array([0.5, -0.5], np.float32),
            name="velocity",
            color="#123456",
        )
        # Each arrow occupies nine vertices; indices 0/1 and 9/10 are shafts.
        np.testing.assert_allclose(
            np.array([trace[axis][0] for axis in "xyz"]), [1.0, 2.0, 3.0]
        )
        np.testing.assert_allclose(
            np.array([trace[axis][1] for axis in "xyz"]), [3.0, 1.0, 3.0]
        )
        np.testing.assert_allclose(
            np.array([trace[axis][9] for axis in "xyz"]), [4.0, 5.0, 6.0]
        )
        np.testing.assert_allclose(
            np.array([trace[axis][10] for axis in "xyz"]), [4.0, 5.5, 5.75]
        )

    def test_center_html_adds_velocity_layer_only_when_vectors_exist(self):
        base = {
            "label": "T=1s",
            "static": np.zeros((0, 3), np.float32),
            "dynamic": np.array([[1.0, 2.0, 3.0]], np.float32),
        }
        with tempfile.TemporaryDirectory() as tmp:
            without_path = Path(tmp) / "without.html"
            with_path = Path(tmp) / "with.html"
            save_sequence_html(without_path, "bbox", [base])
            save_sequence_html(with_path, "velocity", [{
                "label": "T=1s",
                "source_frame_centers": [
                    np.array([[1.5, 2.0, 3.0]], np.float32),
                    np.array([[4.0, 5.5, 5.75]], np.float32),
                ],
                "velocity_origins": np.array(
                    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], np.float32
                ),
                "velocity_mps": np.array(
                    [[1.0, 0.0, 0.0], [0.0, -1.0, 0.5]], np.float32
                ),
                "velocity_delta_t_sec": np.array([0.5, -0.5], np.float32),
                "velocity_source_frame_index": np.array([0, 1]),
            }])

            without = without_path.read_text()
            with_velocity = with_path.read_text()
            self.assertNotIn("gaussians from source frame", without)
            self.assertIn('"gaussians from source frame 0"', with_velocity)
            self.assertIn('"gaussians from source frame 1"', with_velocity)
            self.assertIn(
                json.dumps(_source_frame_velocity_layer(0)), with_velocity
            )
            self.assertIn(
                json.dumps(_source_frame_velocity_layer(1)), with_velocity
            )
            for color in SOURCE_FRAME_COLORS[:2]:
                self.assertIn(
                    json.dumps({"color": color, "width": 2}), with_velocity
                )
                self.assertIn(
                    json.dumps({
                        "size": 2.6, "color": color, "opacity": 0.9,
                    }),
                    with_velocity,
                )
            self.assertNotIn('"gaussians (static)"', with_velocity)
            self.assertNotIn('"gaussians (dynamic)"', with_velocity)

    def test_velocity_offset_does_not_depend_on_sample_duration(self):
        backend, _output = self._backend_output()
        head = backend.velocity_head
        with torch.no_grad():
            head.velocity.bias.fill_(0.1)
        velocity = head(torch.zeros(2, 24), torch.zeros(2, 3))
        self.assertTrue(torch.equal(velocity, torch.full((2, 3), 0.1)))

class RouterBackendTest(unittest.TestCase):
    """The learned K={1,2,3} router shared by every router-capable backend."""

    def test_seed_delta_bank_rotates_without_pose_translation(self):
        gs_params = OmegaConf.create({
            "shs": 32, "opacity": 1, "scaling": 2, "rotation": 4, "offset": 3,
        })
        backend = MaxSpeedBarrierVelocityGaussianBackend(
            v11_4_backend_cfg(), gs_params, dim=24, offset_bound=0.8,
            gaussian_count_cfg=router_count_cfg(),
        )
        position = torch.tensor([
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ])
        seed, seed_delta = adaptive_seed_bank(position)
        poses = torch.eye(4).repeat(2, 1, 1)
        poses[1, :3, :3] = torch.tensor([
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        poses[1, :3, 3] = torch.tensor([10.0, -4.0, 2.0])
        geometry = backend._reference_geometry_and_time(
            position,
            seed,
            torch.tensor([1, 2]),
            torch.tensor([0, 0]),
            [poses],
            [torch.tensor([0.0, 1.0])],
            [torch.tensor([0.0, 0.8])],
            torch.tensor([0.8]),
            seed_delta_local=seed_delta,
        )
        # Translation cancels from a delta; +x rotates to +y in frame 1.
        torch.testing.assert_close(
            geometry["seed_delta_ref"][1, 2, 2],
            torch.tensor([0.0, 0.2, 0.0]),
        )
        # Padding slots remain zero rather than becoming a pose translation.
        torch.testing.assert_close(
            geometry["seed_delta_ref"][1, 0, 1], torch.zeros(3)
        )

    def test_render_loss_trains_k_router_without_a_budget_loss(self):
        torch.manual_seed(47)
        gs_params = OmegaConf.create({
            "shs": 32, "opacity": 1, "scaling": 2, "rotation": 4, "offset": 3,
        })
        backend = MaxSpeedBarrierVelocityGaussianBackend(
            v11_4_backend_cfg(), gs_params, dim=24, offset_bound=0.8,
            gaussian_count_cfg=router_count_cfg(),
        ).train()

        def deterministic_ste(logits):
            soft = torch.softmax(logits / backend.gaussian_head.gumbel_tau, dim=-1)
            selected = torch.arange(logits.shape[0], device=logits.device) % 3
            hard = F.one_hot(selected, num_classes=3).to(logits.dtype)
            return soft + (hard - soft).detach()

        backend.gaussian_head._gumbel_selection = deterministic_ste
        n_per_frame = 3
        feature = torch.randn(2 * n_per_frame, 24, requires_grad=True)
        position = torch.randn(2 * n_per_frame, 3)
        seed, seed_delta = adaptive_seed_bank(position)
        output = backend(
            feature,
            position,
            seed,
            torch.tensor([n_per_frame, 2 * n_per_frame]),
            torch.tensor([0, 0]),
            [torch.eye(4).repeat(2, 1, 1)],
            [torch.tensor([0.0, 1.0])],
            [torch.tensor([0.0, 0.8])],
            torch.tensor([0.8]),
            seed_delta_sensor=seed_delta,
        )
        self.assertTrue(torch.equal(
            output["routing_stats"]["selected_k"],
            torch.tensor([1, 2, 3, 1, 2, 3]),
        ))
        self.assertFalse(output["routing_stats"]["k_logits"].requires_grad)
        self.assertNotIn("routing_budget_logits", output)
        item = output["batch_gaussians"][0]
        self.assertEqual(item["opacity"].shape[0], 12)
        torch.sigmoid(item["opacity"]).sum().backward()
        router_grad = backend.gaussian_head.count_predictor[-1].weight.grad
        self.assertIsNotNone(router_grad)
        self.assertGreater(float(router_grad.abs().sum()), 0.0)
        self.assertGreater(float(feature.grad.abs().sum()), 0.0)


class V11MaxSpeedBarrierAttentionTest(unittest.TestCase):
    @staticmethod
    def _forward_args(feature, position, first_frame_tokens, duration=0.8):
        total = feature.shape[0]
        time = torch.cat([
            torch.zeros(first_frame_tokens),
            torch.full((total - first_frame_tokens,), duration),
        ])
        return (
            feature,
            position,
            torch.tensor([first_frame_tokens, total]),
            torch.tensor([0, 0]),
            time,
            torch.full((total,), duration),
        )

    def test_barrier_is_free_inside_the_reachable_radius(self):
        module = MaxSpeedBarrierLayerWeightedCrossAttention(
            v11_temporal_cfg(layers=1), dim=24
        )
        query = torch.zeros(1, 3)
        # 0.5R, R, 1.5R, 2R and 3R at a 30 m radius (30 m/s over 1.0 s).
        key = torch.tensor([
            [15.0, 0.0, 0.0], [30.0, 0.0, 0.0], [45.0, 0.0, 0.0],
            [60.0, 0.0, 0.0], [90.0, 0.0, 0.0],
        ])
        bias = module._barrier_bias(query, key, 30.0)
        torch.testing.assert_close(
            bias, torch.tensor([[0.0, 0.0, 1.0, 4.0, 16.0]])
        )
        # The radius follows the physical endpoint interval, not the distance.
        halved = module._barrier_bias(query, key, 15.0)
        torch.testing.assert_close(
            halved, torch.tensor([[0.0, 4.0, 16.0, 36.0, 100.0]])
        )

    def test_linear_hinge_penalizes_proportionally_to_the_excess(self):
        """V11.4's `-relu(d/R-1)` against V11's `-4 relu(d/R-1)^2`."""
        cfg = v11_temporal_cfg(layers=1)
        cfg.barrier_weight = 1.0
        cfg.barrier_exponent = 1
        module = MaxSpeedBarrierLayerWeightedCrossAttention(cfg, dim=24)
        self.assertEqual(module.barrier_exponent, 1)
        query = torch.zeros(1, 3)
        key = torch.tensor([
            [15.0, 0.0, 0.0], [30.0, 0.0, 0.0], [45.0, 0.0, 0.0],
            [60.0, 0.0, 0.0], [90.0, 0.0, 0.0],
        ])
        # Still exactly free inside R, then one logit per extra radius.
        torch.testing.assert_close(
            module._barrier_bias(query, key, 30.0),
            torch.tensor([[0.0, 0.0, 0.5, 1.0, 2.0]]),
        )
        # The radius still tracks the physical interval rather than distance.
        torch.testing.assert_close(
            module._barrier_bias(query, key, 15.0),
            torch.tensor([[0.0, 1.0, 2.0, 3.0, 5.0]]),
        )

    def test_linear_hinge_folds_into_the_score_gemm_too(self):
        """`(a/R) relu(d-R)` must equal `a relu(d/R-1)`, as at p=2."""
        torch.manual_seed(13)
        cfg = v11_temporal_cfg(layers=1)
        cfg.barrier_weight = 1.0
        cfg.barrier_exponent = 1
        module = MaxSpeedBarrierLayerWeightedCrossAttention(cfg, dim=24)
        query = torch.randn(5, module.head_dim)
        key = torch.randn(7, module.head_dim)
        value = torch.randn(7, module.head_dim + 3)
        query_position = torch.randn(5, 3) * 40.0
        key_position = torch.randn(7, 3) * 40.0
        radius = 21.0

        feature, matched = module._barrier_chunk(
            query, key, value, query_position, key_position, radius
        )
        reference_score = (query @ key.transpose(0, 1)) * module.scale
        reference_score = reference_score - module._barrier_bias(
            query_position, key_position, radius
        )
        expected = torch.softmax(reference_score, dim=-1) @ value
        torch.testing.assert_close(
            feature, expected[..., :module.head_dim], atol=1.0e-5, rtol=1.0e-5
        )
        torch.testing.assert_close(
            matched, expected[..., module.head_dim:], atol=1.0e-5, rtol=1.0e-5
        )

    def test_hinge_exponent_changes_attention_beyond_the_radius(self):
        """The two hinges must actually disagree once a key is out of reach."""
        torch.manual_seed(5)
        quadratic = MaxSpeedBarrierLayerWeightedCrossAttention(
            v11_temporal_cfg(layers=1), dim=24
        )
        linear_cfg = v11_temporal_cfg(layers=1)
        linear_cfg.barrier_weight = 1.0
        linear_cfg.barrier_exponent = 1
        linear = MaxSpeedBarrierLayerWeightedCrossAttention(linear_cfg, dim=24)

        query = torch.randn(4, quadratic.head_dim)
        key = torch.randn(6, quadratic.head_dim)
        value = torch.randn(6, quadratic.head_dim + 3)
        query_position = torch.zeros(4, 3)
        # 0.5R and 2.5R at R = 20: half the keys are unreachable.
        key_position = torch.zeros(6, 3)
        key_position[3:, 0] = 50.0
        key_position[:3, 0] = 10.0
        radius = 20.0

        args = (query, key, value, query_position, key_position, radius)
        near_only = [m._barrier_chunk(*args)[0] for m in (quadratic, linear)]
        self.assertFalse(torch.allclose(near_only[0], near_only[1]))
        # The quadratic charges 9 logits at 2.5R against the linear's 1.5, so
        # it must put strictly less mass on the unreachable half.
        masses = []
        for module in (quadratic, linear):
            bias = module._barrier_bias(query_position, key_position, radius)
            score = (query @ key.transpose(0, 1)) * module.scale - bias
            masses.append(float(torch.softmax(score, dim=-1)[:, 3:].sum()))
        self.assertLess(masses[0], masses[1])

    def test_fused_score_matches_the_reference_barrier_form(self):
        """`(a/R^2) relu(d-R)^2` in the GEMM must equal `a relu(d/R-1)^2`."""
        torch.manual_seed(13)
        module = MaxSpeedBarrierLayerWeightedCrossAttention(
            v11_temporal_cfg(layers=1), dim=24
        )
        query = torch.randn(5, module.head_dim)
        key = torch.randn(7, module.head_dim)
        value = torch.randn(7, module.head_dim + 3)
        query_position = torch.randn(5, 3) * 40.0
        key_position = torch.randn(7, 3) * 40.0
        radius = 21.0

        feature, matched = module._barrier_chunk(
            query, key, value, query_position, key_position, radius
        )
        reference_score = (query @ key.transpose(0, 1)) * module.scale
        reference_score = reference_score - module._barrier_bias(
            query_position, key_position, radius
        )
        expected = torch.softmax(reference_score, dim=-1) @ value
        torch.testing.assert_close(
            feature, expected[..., :module.head_dim], atol=1.0e-5, rtol=1.0e-5
        )
        torch.testing.assert_close(
            matched, expected[..., module.head_dim:], atol=1.0e-5, rtol=1.0e-5
        )

    def test_mixed_durations_within_a_frame_are_rejected(self):
        module = MaxSpeedBarrierLayerWeightedCrossAttention(
            v11_temporal_cfg(layers=1), dim=24
        )
        feature = torch.randn(6, 24)
        position = torch.randn(6, 3)
        duration = torch.tensor([0.8, 0.8, 0.5, 0.8, 0.8, 0.8])
        with self.assertRaisesRegex(ValueError, "one endpoint interval"):
            module(
                feature,
                position,
                torch.tensor([3, 6]),
                torch.tensor([0, 0]),
                torch.tensor([0.0, 0.0, 0.0, 0.8, 0.8, 0.8]),
                duration,
            )

    def test_match_head_readout_equals_an_explicit_biased_softmax(self):
        torch.manual_seed(3)
        module = MaxSpeedBarrierLayerWeightedCrossAttention(
            v11_temporal_cfg(layers=1, match_chunk_size=4), dim=24
        ).eval()
        feature = torch.randn(9, 24)
        position = torch.randn(9, 3) * 30.0

        captured = {}
        original = module._barrier_attention

        def capture(query, key, value, query_position, key_position,
                    q_count_values, k_count_values, frame_radius_values):
            captured.update(
                query=query, key=key, query_position=query_position,
                key_position=key_position, q_counts=q_count_values,
                k_counts=k_count_values, radii=frame_radius_values,
            )
            return original(
                query, key, value, query_position, key_position,
                q_count_values, k_count_values, frame_radius_values,
            )

        module._barrier_attention = capture
        with torch.no_grad():
            _refined, fields = module(
                *self._forward_args(feature, position, 4, duration=0.7)
            )

        # Only head 0 ever reaches the coordinate readout.
        self.assertEqual(tuple(captured["query"].shape), (9, module.head_dim))
        start = 0
        key_start = 0
        # 30 m/s over the 0.7 s endpoint interval.
        self.assertEqual(len(captured["radii"]), 2)
        for radius in captured["radii"]:
            self.assertAlmostEqual(radius, 21.0, places=4)
        for q_count, k_count, radius in zip(
            captured["q_counts"], captured["k_counts"], captured["radii"]
        ):
            rows = slice(start, start + q_count)
            keys = slice(key_start, key_start + k_count)
            query_position = captured["query_position"][rows]
            key_position = captured["key_position"][keys]
            score = (
                captured["query"][rows] @ captured["key"][keys].transpose(0, 1)
            ) * module.scale - 4.0 * torch.relu(
                torch.cdist(query_position, key_position) / radius - 1.0
            ).square()
            expected = torch.softmax(score, dim=-1) @ key_position
            torch.testing.assert_close(
                fields["matched_position"][rows], expected, atol=1.0e-4,
                rtol=1.0e-4,
            )
            start += q_count
            key_start += k_count
        torch.testing.assert_close(
            fields["delta_p_init"], fields["matched_position"] - position
        )

    def test_query_chunking_changes_neither_output_nor_gradient(self):
        reference = None
        for chunk in (1, 3, 1000):
            torch.manual_seed(7)
            module = MaxSpeedBarrierLayerWeightedCrossAttention(
                v11_temporal_cfg(layers=2, match_chunk_size=chunk), dim=24
            ).train()
            torch.manual_seed(11)
            feature = torch.randn(11, 24, requires_grad=True)
            position = torch.randn(11, 3) * 25.0
            refined, fields = module(
                *self._forward_args(feature, position, 5, duration=0.9)
            )
            (
                refined.square().sum()
                + fields["matched_position"].square().sum()
            ).backward()
            observed = (
                refined.detach(),
                fields["matched_position"].detach(),
                feature.grad.clone(),
                module.q_proj[0].weight.grad.clone(),
            )
            if reference is None:
                reference = observed
                continue
            for expected, actual in zip(reference, observed):
                torch.testing.assert_close(
                    actual, expected, atol=1.0e-5, rtol=1.0e-4
                )

    def test_rope_covers_every_head_but_the_match_head(self):
        module = MaxSpeedBarrierLayerWeightedCrossAttention(
            v11_temporal_cfg(layers=2, heads=4), dim=24
        ).eval()
        self.assertEqual(module.rope_head_count, 3)
        rotated_shapes = []
        original_rotate = module.rope.rotate

        def capture(x, angles):
            rotated_shapes.append(tuple(x.shape))
            return original_rotate(x, angles)

        module.rope.rotate = capture
        feature = torch.randn(6, 24)
        position = torch.randn(6, 3) * 10.0
        with torch.no_grad():
            module(*self._forward_args(feature, position, 3))
        # Two layers x (Q, K), each carrying heads 1..3 only.
        self.assertEqual(len(rotated_shapes), 4)
        for shape in rotated_shapes:
            self.assertEqual(shape, (6, 3, module.head_dim))

    def test_layer_mixture_is_uniform_scalars_that_receive_gradient(self):
        torch.manual_seed(41)
        module = MaxSpeedBarrierLayerWeightedCrossAttention(
            v11_temporal_cfg(layers=3), dim=24
        )
        self.assertFalse(hasattr(module, "global_token"))
        self.assertFalse(hasattr(module, "layer_score_mlp"))
        self.assertEqual(tuple(module.layer_logits.shape), (3,))
        torch.testing.assert_close(module.layer_logits, torch.zeros(3))

        feature = torch.randn(7, 24, requires_grad=True)
        position = torch.randn(7, 3) * 12.0
        refined, fields = module(
            *self._forward_args(feature, position, 3)
        )
        self.assertEqual(tuple(refined.shape), (7, 24))
        # One shared mixture, broadcast to every token rather than predicted.
        torch.testing.assert_close(
            fields["motion_layer_weights"], torch.full((7, 3), 1.0 / 3.0)
        )
        torch.testing.assert_close(
            fields["motion_layer_logits"], torch.zeros(7, 3)
        )

        loss = fields["matched_position"].square().sum() + refined.square().mean()
        loss.backward()
        self.assertGreater(float(feature.grad.abs().sum()), 0.0)
        self.assertGreater(float(module.layer_logits.grad.abs().sum()), 0.0)
        for layer in range(module.n_layers):
            self.assertGreater(
                float(module.q_proj[layer].weight.grad.abs().sum()), 0.0
            )
            self.assertGreater(
                float(module.k_proj[layer].weight.grad.abs().sum()), 0.0
            )

    def test_layer_weights_select_a_single_layer_readout(self):
        module = MaxSpeedBarrierLayerWeightedCrossAttention(
            v11_temporal_cfg(layers=3), dim=24
        )
        with torch.no_grad():
            module.layer_logits.copy_(torch.tensor([-20.0, 20.0, -20.0]))
        layer_matches = torch.tensor([
            [[1.0, 0.0, 0.0], [3.0, 0.0, 0.0], [5.0, 0.0, 0.0]],
            [[2.0, 1.0, 0.0], [4.0, 1.0, 0.0], [6.0, 1.0, 0.0]],
        ])
        position = torch.tensor([[0.5, 0.0, 0.0], [1.0, 1.0, 0.0]])
        matched, displacement, logits, weights = module._combine_layer_matches(
            layer_matches, position
        )
        torch.testing.assert_close(matched, layer_matches[:, 1], atol=1.0e-6,
                                   rtol=1.0e-6)
        torch.testing.assert_close(displacement, matched - position)
        torch.testing.assert_close(weights[:, 1], torch.ones(2), atol=1.0e-6,
                                   rtol=1.0e-6)
        torch.testing.assert_close(
            logits, module.layer_logits.expand(2, -1)
        )

    def test_config_rejects_a_single_head_and_a_missing_barrier(self):
        with self.assertRaises(ValueError):
            MaxSpeedBarrierLayerWeightedCrossAttention(
                v11_temporal_cfg(layers=1, heads=1), dim=24
            )
        no_qk_norm = v11_temporal_cfg(layers=1)
        no_qk_norm.qk_norm = False
        with self.assertRaises(ValueError):
            MaxSpeedBarrierLayerWeightedCrossAttention(no_qk_norm, dim=24)
        wrong_encoding = v11_temporal_cfg(layers=1)
        wrong_encoding.position_encoding = "distance_bias"
        with self.assertRaises(ValueError):
            MaxSpeedBarrierLayerWeightedCrossAttention(wrong_encoding, dim=24)

    def test_backend_shares_one_token_velocity_across_adaptive_gaussians(self):
        torch.manual_seed(29)
        gs_params = OmegaConf.create({
            "shs": 32, "opacity": 1, "scaling": 2, "rotation": 4, "offset": 3,
        })
        backend = MaxSpeedBarrierVelocityGaussianBackend(
            v11_backend_cfg(), gs_params, dim=24, offset_bound=0.8,
            gaussian_count_cfg=router_count_cfg(),
        ).eval()
        self.assertIsInstance(
            backend.temporal, MaxSpeedBarrierLayerWeightedCrossAttention
        )
        self.assertEqual(backend.gaussian_head.count_mode, "learned_gumbel")
        self.assertEqual(backend.gaussian_head.k_max, 3)
        # V11 conditions the offset on the initializer it corrects.
        self.assertIsInstance(backend.velocity_head, EmbeddedInitVelocityHead)
        self.assertEqual(backend.velocity_head.velocity_embedding_dim, 32)
        self.assertTrue(backend.velocity_head.detach_init_condition)
        encoder = backend.velocity_head.velocity_encoder
        self.assertEqual(encoder[0].in_features, 3)
        self.assertEqual(encoder[-1].out_features, 32)
        self.assertEqual(
            backend.velocity_head.motion_refiner[0].in_features, 24 + 32
        )
        with torch.no_grad():
            backend.gaussian_head.count_predictor[-1].bias.copy_(
                torch.tensor([-10.0, -10.0, 10.0])
            )

        n_per_frame = 4
        feature = torch.randn(2 * n_per_frame, 24)
        position = torch.randn(2 * n_per_frame, 3) * 2.0
        seed, seed_delta = (
            adaptive_seed_bank(position)
        )
        output = backend(
            feature,
            position,
            seed,
            torch.tensor([n_per_frame, 2 * n_per_frame]),
            torch.tensor([0, 0]),
            [torch.eye(4).repeat(2, 1, 1)],
            [torch.tensor([0.0, 1.0])],
            [torch.tensor([0.0, 0.8])],
            torch.tensor([0.8]),
            seed_delta_sensor=seed_delta,
        )
        item = output["batch_gaussians"][0]
        self.assertTrue(torch.equal(item["selected_k"], torch.full((24,), 3)))
        shared_velocity = item["velocity"].reshape(8, 3, 3)
        torch.testing.assert_close(
            shared_velocity, shared_velocity[:, :1].expand_as(shared_velocity)
        )
        # zero_init keeps a fresh run at v_total = v_init.
        torch.testing.assert_close(
            item["velocity_offset"], torch.zeros_like(item["velocity_offset"])
        )
        torch.testing.assert_close(
            item["velocity"], item["velocity_init"] + item["velocity_offset"]
        )
        # v_init is the layer-mixed displacement over the signed interval.
        torch.testing.assert_close(
            item["velocity_init"] * item["pair_delta_t_sec"].unsqueeze(-1),
            item["delta_p_init"],
        )
        stats = ModelWrapper._motion_statistics([item])
        self.assertIn("motion_layer_weight_entropy_mean", stats)
        for layer in range(3):
            self.assertIn(f"motion_layer_weight_l{layer}_mean", stats)

    def test_barrier_caps_v_init_near_the_configured_top_speed(self):
        """A token with no reachable key still cannot outrun the barrier much."""
        torch.manual_seed(5)
        module = MaxSpeedBarrierLayerWeightedCrossAttention(
            v11_temporal_cfg(layers=1), dim=24
        ).eval()
        duration = 1.0
        # Frame 0 holds one query at the origin; frame 1 holds keys at 20 m
        # (inside the 30 m radius) and 200 m (far outside it).
        position = torch.tensor([
            [0.0, 0.0, 0.0], [20.0, 0.0, 0.0], [200.0, 0.0, 0.0],
        ])
        feature = torch.randn(3, 24)
        with torch.no_grad():
            _refined, fields = module(
                *self._forward_args(feature, position, 1, duration=duration)
            )
        speed = float(fields["delta_p_init"][0].norm() / duration)
        self.assertLess(speed, 30.0)


class V12LearnableMaxSpeedBarrierAttentionTest(unittest.TestCase):
    """V12 changes only V11.4's fixed global barrier speed."""

    @staticmethod
    def _forward_args(feature, position, first_frame_tokens, duration=0.8):
        return V11MaxSpeedBarrierAttentionTest._forward_args(
            feature, position, first_frame_tokens, duration
        )

    def test_v0_starts_at_log_30_and_is_the_only_new_parameter(self):
        module = LearnableMaxSpeedBarrierLayerWeightedCrossAttention(
            v12_temporal_cfg(layers=2), dim=24
        )
        self.assertIsInstance(module.v_0, torch.nn.Parameter)
        self.assertTrue(module.v_0.requires_grad)
        self.assertEqual(tuple(module.v_0.shape), ())
        self.assertAlmostEqual(float(module.v_0), math.log(30.0), places=6)
        self.assertAlmostEqual(float(module.v_max), 30.0, places=5)

        reference = MaxSpeedBarrierLayerWeightedCrossAttention(
            v12_temporal_cfg(layers=2), dim=24
        )
        new_names = set(dict(module.named_parameters())) - set(
            dict(reference.named_parameters())
        )
        self.assertEqual(new_names, {"v_0"})

    def test_qk_rms_gamma_remains_shared_across_every_head(self):
        module = LearnableMaxSpeedBarrierLayerWeightedCrossAttention(
            v12_temporal_cfg(heads=4, layers=2), dim=24
        )
        for layer in range(module.n_layers):
            for norm in (module.q_norm[layer], module.k_norm[layer]):
                self.assertIsInstance(norm, torch.nn.RMSNorm)
                # One head_dim vector broadcasts over the head axis. There is
                # no per-head or V11.1-style match/RoPE group dimension.
                self.assertEqual(tuple(norm.weight.shape), (module.head_dim,))

    def test_initial_forward_matches_v11_4(self):
        cfg = v12_temporal_cfg(layers=2, match_chunk_size=2)
        torch.manual_seed(120)
        reference = MaxSpeedBarrierLayerWeightedCrossAttention(
            cfg, dim=24
        ).eval()
        torch.manual_seed(120)
        module = LearnableMaxSpeedBarrierLayerWeightedCrossAttention(
            cfg, dim=24
        ).eval()

        feature = torch.randn(7, 24)
        position = torch.tensor([
            [0.0, 0.0, 0.0],
            [8.0, 1.0, 0.0],
            [24.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [20.0, 1.0, 0.0],
            [36.0, -1.0, 0.0],
            [70.0, 2.0, 0.0],
        ])
        args = self._forward_args(feature, position, 3, duration=0.8)
        with torch.no_grad():
            refined, fields = module(*args)
            expected_refined, expected_fields = reference(*args)
        torch.testing.assert_close(
            refined, expected_refined, atol=2.0e-6, rtol=2.0e-6
        )
        for name in (
            "delta_p_match", "delta_p_init", "matched_position",
            "motion_layer_logits", "motion_layer_weights",
        ):
            torch.testing.assert_close(
                fields[name], expected_fields[name], atol=2.0e-6, rtol=2.0e-6
            )

    def test_full_checkpointed_readout_backpropagates_to_v0(self):
        torch.manual_seed(121)
        module = LearnableMaxSpeedBarrierLayerWeightedCrossAttention(
            v12_temporal_cfg(layers=1, match_chunk_size=1), dim=24
        ).train()
        feature = torch.randn(3, 24)
        # For the origin query, one key is inside R and one is outside R. Its
        # coordinate expectation therefore changes continuously with v_max.
        position = torch.tensor([
            [0.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
            [60.0, 0.0, 0.0],
        ])
        _refined, fields = module(
            *self._forward_args(feature, position, 1, duration=1.0)
        )
        fields["matched_position"][0, 0].backward()
        self.assertIsNotNone(module.v_0.grad)
        self.assertTrue(torch.isfinite(module.v_0.grad))
        self.assertGreater(float(module.v_0.grad.abs()), 0.0)

    def test_backend_and_stats_expose_one_current_global_speed(self):
        backend = LearnableMaxSpeedBarrierVelocityGaussianBackend(
            v12_backend_cfg(), _gs_params(), dim=24, offset_bound=0.8,
            gaussian_count_cfg=router_count_cfg(),
        )
        self.assertIsInstance(
            backend.temporal,
            LearnableMaxSpeedBarrierLayerWeightedCrossAttention,
        )
        self.assertIsInstance(backend.velocity_head, EmbeddedInitVelocityHead)
        self.assertEqual(backend.gaussian_head.count_mode, "learned_gumbel")

        with torch.no_grad():
            backend.temporal.v_0.copy_(torch.tensor(math.log(42.0)))
        stats = backend.temporal.attention_temperature_stats()
        self.assertAlmostEqual(stats["barrier_speed_initial_mps"], 30.0)
        self.assertAlmostEqual(stats["barrier_v_0"], math.log(42.0), places=5)
        self.assertAlmostEqual(stats["barrier_speed_mps"], 42.0, places=4)


class V12_1FixedMotionGainAttentionTest(unittest.TestCase):
    """V12.1 fixes head 0's gain and learns one gain for heads 1..H-1."""

    @staticmethod
    def _forward_args(feature, position, first_frame_tokens, duration=0.8):
        return V11MaxSpeedBarrierAttentionTest._forward_args(
            feature, position, first_frame_tokens, duration
        )

    def test_only_the_rope_group_has_a_gain_parameter(self):
        module = FixedMotionGainLearnableMaxSpeedBarrierCrossAttention(
            v12_1_temporal_cfg(heads=4, layers=2), dim=24
        )
        for layer in range(module.n_layers):
            for norm in (module.q_norm[layer], module.k_norm[layer]):
                self.assertIsInstance(norm, FixedMotionHeadRMSNorm)
                self.assertEqual(norm.match_head, 0)
                self.assertEqual(
                    norm.is_match_head.tolist(), [True, False, False, False]
                )
                parameters = dict(norm.named_parameters())
                self.assertEqual(set(parameters), {"rope_weight"})
                self.assertEqual(
                    tuple(parameters["rope_weight"].shape), (module.head_dim,)
                )

    def test_motion_norm_is_fixed_while_rope_gain_learns(self):
        torch.manual_seed(122)
        norm = FixedMotionHeadRMSNorm(head_dim=6, num_heads=4)
        value = torch.randn(5, 4, 6, requires_grad=True)
        with torch.no_grad():
            norm.rope_weight.fill_(2.0)
        output = norm(value)
        motion_rms = output[:, 0].float().square().mean(dim=-1).sqrt()
        rope_rms = output[:, 1:].float().square().mean(dim=-1).sqrt()
        torch.testing.assert_close(
            motion_rms, torch.ones_like(motion_rms), atol=2.0e-5, rtol=2.0e-5
        )
        torch.testing.assert_close(
            rope_rms, torch.full_like(rope_rms, 2.0), atol=4.0e-5, rtol=2.0e-5
        )

        output[:, 0].square().sum().backward()
        self.assertIsNotNone(norm.rope_weight.grad)
        self.assertEqual(float(norm.rope_weight.grad.abs().sum()), 0.0)

        norm.zero_grad(set_to_none=True)
        value = torch.randn(5, 4, 6, requires_grad=True)
        norm(value)[:, 1:].square().sum().backward()
        self.assertGreater(float(norm.rope_weight.grad.abs().sum()), 0.0)

    def test_initial_forward_matches_v12(self):
        cfg = v12_1_temporal_cfg(layers=2, match_chunk_size=2)
        torch.manual_seed(123)
        reference = LearnableMaxSpeedBarrierLayerWeightedCrossAttention(
            cfg, dim=24
        ).eval()
        torch.manual_seed(123)
        module = FixedMotionGainLearnableMaxSpeedBarrierCrossAttention(
            cfg, dim=24
        ).eval()
        feature = torch.randn(8, 24)
        position = torch.randn(8, 3) * 12.0
        args = self._forward_args(feature, position, 4, duration=0.8)
        with torch.no_grad():
            refined, fields = module(*args)
            expected_refined, expected_fields = reference(*args)
        torch.testing.assert_close(
            refined, expected_refined, atol=3.0e-6, rtol=3.0e-6
        )
        for name in (
            "delta_p_match", "delta_p_init", "matched_position",
            "motion_layer_logits", "motion_layer_weights",
        ):
            torch.testing.assert_close(
                fields[name], expected_fields[name], atol=3.0e-6, rtol=3.0e-6
            )

    def test_lambda_starts_at_one_scales_penalty_and_receives_gradient(self):
        torch.manual_seed(124)
        module = FixedMotionGainLearnableMaxSpeedBarrierCrossAttention(
            v12_1_temporal_cfg(layers=1, match_chunk_size=1), dim=24
        ).train()
        self.assertIsInstance(module.log_barrier_lambda, torch.nn.Parameter)
        self.assertEqual(float(module.log_barrier_lambda), 0.0)
        self.assertEqual(float(module.barrier_lambda), 1.0)

        query = torch.zeros(1, 3)
        key = torch.tensor([[20.0, 0.0, 0.0], [60.0, 0.0, 0.0]])
        initial = module._barrier_bias(query, key, module.v_max)
        torch.testing.assert_close(initial, torch.tensor([[0.0, 1.0]]))
        with torch.no_grad():
            module.log_barrier_lambda.copy_(torch.tensor(math.log(2.0)))
        doubled = module._barrier_bias(query, key, module.v_max)
        torch.testing.assert_close(doubled, 2.0 * initial)

        with torch.no_grad():
            module.log_barrier_lambda.zero_()
        feature = torch.randn(3, 24)
        position = torch.cat([query, key], dim=0)
        _refined, fields = module(
            *self._forward_args(feature, position, 1, duration=1.0)
        )
        fields["matched_position"][0, 0].backward()
        for parameter in (module.v_0, module.log_barrier_lambda):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad))
            self.assertGreater(float(parameter.grad.abs()), 0.0)

    def test_stats_keep_match_gamma_one_and_report_rope_gamma(self):
        module = FixedMotionGainLearnableMaxSpeedBarrierCrossAttention(
            v12_1_temporal_cfg(layers=1), dim=24
        )
        with torch.no_grad():
            module.q_norm[0].rope_weight.fill_(2.0)
            module.k_norm[0].rope_weight.fill_(3.0)
        stats = module.attention_temperature_stats()
        self.assertEqual(stats["qk_gamma_q_match_layer0"], 1.0)
        self.assertEqual(stats["qk_gamma_k_match_layer0"], 1.0)
        self.assertAlmostEqual(stats["qk_gamma_q_rope_layer0"], 2.0)
        self.assertAlmostEqual(stats["qk_gamma_k_rope_layer0"], 3.0)
        self.assertAlmostEqual(
            stats["qk_max_logit_bound_match"], module.head_dim ** 0.5
        )
        self.assertGreater(
            stats["qk_max_logit_bound_rope"],
            stats["qk_max_logit_bound_match"],
        )
        self.assertAlmostEqual(stats["barrier_speed_mps"], 30.0, places=4)
        self.assertEqual(stats["barrier_lambda_initial"], 1.0)
        self.assertEqual(stats["barrier_log_lambda"], 0.0)
        self.assertEqual(stats["barrier_lambda"], 1.0)

    def test_backend_keeps_v12_router_motion_head_and_learned_speed(self):
        backend = FixedMotionGainMaxSpeedBarrierVelocityGaussianBackend(
            v12_1_backend_cfg(), _gs_params(), dim=24, offset_bound=0.4,
            gaussian_count_cfg=router_count_cfg(),
        )
        self.assertIsInstance(
            backend.temporal,
            FixedMotionGainLearnableMaxSpeedBarrierCrossAttention,
        )
        self.assertIsInstance(backend.temporal.v_0, torch.nn.Parameter)
        self.assertAlmostEqual(float(backend.temporal.v_max), 30.0, places=5)
        self.assertIsInstance(
            backend.temporal.log_barrier_lambda, torch.nn.Parameter
        )
        self.assertEqual(float(backend.temporal.barrier_lambda), 1.0)
        self.assertIsInstance(backend.velocity_head, EmbeddedInitVelocityHead)
        self.assertEqual(backend.gaussian_head.count_mode, "learned_gumbel")
        self.assertAlmostEqual(backend.offset_bound, 0.4)


class V12_2LayerwiseSoftplusMaxSpeedBarrierAttentionTest(unittest.TestCase):
    """V12.2 learns V12's barrier speed once per layer, through softplus."""

    @staticmethod
    def _forward_args(feature, position, first_frame_tokens, duration=0.8):
        return V11MaxSpeedBarrierAttentionTest._forward_args(
            feature, position, first_frame_tokens, duration
        )

    @staticmethod
    def _softplus_inverse(speed):
        return speed + math.log(-math.expm1(-speed))

    def _set_speeds(self, module, speeds):
        with torch.no_grad():
            module.v_raw.copy_(torch.tensor(
                [self._softplus_inverse(speed) for speed in speeds]
            ))

    def test_every_layer_starts_at_30_and_v_raw_is_the_only_new_parameter(self):
        module = LayerwiseSoftplusMaxSpeedBarrierCrossAttention(
            v12_temporal_cfg(layers=3), dim=24
        )
        self.assertIsInstance(module.v_raw, torch.nn.Parameter)
        self.assertTrue(module.v_raw.requires_grad)
        self.assertEqual(tuple(module.v_raw.shape), (3,))
        torch.testing.assert_close(module.v_max, torch.full((3,), 30.0))
        torch.testing.assert_close(module.v_max, F.softplus(module.v_raw))

        reference = MaxSpeedBarrierLayerWeightedCrossAttention(
            v12_temporal_cfg(layers=3), dim=24
        )
        names = [name for name, _ in module.named_parameters()]
        # V11.4's parameters in V11.4's order plus the per-layer speeds; V12's
        # global v_0 does not exist here.
        self.assertEqual(names.count("v_raw"), 1)
        self.assertEqual(
            [name for name in names if name != "v_raw"],
            [name for name, _ in reference.named_parameters()],
        )

    def test_the_configured_speed_is_inverted_exactly(self):
        for speed in (0.5, 3.0, 30.0):
            cfg = v12_temporal_cfg(layers=2)
            cfg.barrier_speed_mps = speed
            module = LayerwiseSoftplusMaxSpeedBarrierCrossAttention(cfg, dim=24)
            with self.subTest(speed=speed):
                torch.testing.assert_close(
                    module.v_max, torch.full((2,), speed),
                    atol=1.0e-6, rtol=1.0e-6,
                )

    def test_equal_layer_speeds_reproduce_v12(self):
        """At any shared speed, V12.2 is V12 with v_0 = log(speed)."""
        cfg = v12_temporal_cfg(layers=2, match_chunk_size=2)
        torch.manual_seed(125)
        reference = LearnableMaxSpeedBarrierLayerWeightedCrossAttention(
            cfg, dim=24
        ).eval()
        torch.manual_seed(125)
        module = LayerwiseSoftplusMaxSpeedBarrierCrossAttention(
            cfg, dim=24
        ).eval()

        feature = torch.randn(7, 24)
        position = torch.tensor([
            [0.0, 0.0, 0.0],
            [8.0, 1.0, 0.0],
            [24.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [20.0, 1.0, 0.0],
            [36.0, -1.0, 0.0],
            [70.0, 2.0, 0.0],
        ])
        args = self._forward_args(feature, position, 3, duration=0.8)
        for speed in (30.0, 12.0):
            with self.subTest(speed=speed), torch.no_grad():
                reference.v_0.fill_(math.log(speed))
                self._set_speeds(module, (speed, speed))
                refined, fields = module(*args)
                expected_refined, expected_fields = reference(*args)
                torch.testing.assert_close(
                    refined, expected_refined, atol=1.0e-5, rtol=1.0e-5
                )
                for name in (
                    "delta_p_match", "delta_p_init", "matched_position",
                    "motion_layer_logits", "motion_layer_weights",
                ):
                    torch.testing.assert_close(
                        fields[name], expected_fields[name],
                        atol=1.0e-5, rtol=1.0e-5,
                    )

    def test_each_layer_reads_under_its_own_radius(self):
        torch.manual_seed(126)
        module = LayerwiseSoftplusMaxSpeedBarrierCrossAttention(
            v12_temporal_cfg(layers=3, match_chunk_size=2), dim=24
        ).eval()
        speeds = (30.0, 8.0, 2.5)
        self._set_speeds(module, speeds)
        feature = torch.randn(8, 24)
        position = torch.randn(8, 3) * 12.0
        with mock.patch.object(
            module, "_barrier_attention", wraps=module._barrier_attention
        ) as spy, torch.no_grad():
            module(*self._forward_args(feature, position, 4, duration=0.8))

        self.assertEqual(spy.call_count, 3)
        for layer, (call, speed) in enumerate(zip(spy.call_args_list, speeds)):
            radii = call.args[-1]
            with self.subTest(layer=layer):
                # One radius per endpoint frame, both at the 0.8 s interval.
                self.assertEqual(len(radii), 2)
                for radius in radii:
                    self.assertTrue(torch.is_tensor(radius))
                    self.assertAlmostEqual(float(radius), speed * 0.8, places=4)

    def test_full_checkpointed_readout_backpropagates_to_every_layer_speed(self):
        torch.manual_seed(127)
        module = LayerwiseSoftplusMaxSpeedBarrierCrossAttention(
            v12_temporal_cfg(layers=2, match_chunk_size=1), dim=24
        ).train()
        feature = torch.randn(3, 24)
        # For the origin query, one key is inside R and one is outside R, so
        # every layer's coordinate expectation moves with that layer's speed.
        position = torch.tensor([
            [0.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
            [60.0, 0.0, 0.0],
        ])
        _refined, fields = module(
            *self._forward_args(feature, position, 1, duration=1.0)
        )
        fields["matched_position"][0, 0].backward()
        gradient = module.v_raw.grad
        self.assertIsNotNone(gradient)
        self.assertEqual(tuple(gradient.shape), (2,))
        self.assertTrue(bool(torch.isfinite(gradient).all()))
        self.assertTrue(bool((gradient.abs() > 0.0).all()))

    def test_stats_report_every_layer_speed_and_their_range(self):
        module = LayerwiseSoftplusMaxSpeedBarrierCrossAttention(
            v12_temporal_cfg(layers=3), dim=24
        )
        speeds = (30.0, 8.0, 2.5)
        self._set_speeds(module, speeds)
        stats = module.attention_temperature_stats()
        # No single current speed exists, so V11.4's scalar entry is gone
        # rather than left reporting the configured initialization.
        self.assertNotIn("barrier_speed_mps", stats)
        self.assertEqual(stats["barrier_speed_initial_mps"], 30.0)
        for layer, speed in enumerate(speeds):
            self.assertAlmostEqual(
                stats[f"barrier_speed_mps_layer{layer}"], speed, places=4
            )
            self.assertAlmostEqual(
                stats[f"barrier_v_raw_layer{layer}"],
                self._softplus_inverse(speed),
                places=4,
            )
        self.assertAlmostEqual(stats["barrier_speed_mps_min"], 2.5, places=4)
        self.assertAlmostEqual(stats["barrier_speed_mps_max"], 30.0, places=4)
        self.assertAlmostEqual(
            stats["barrier_speed_mps_mean"], sum(speeds) / 3.0, places=4
        )
        self.assertIn("qk_gamma_q_layer2", stats)
        self.assertIn("qk_max_logit_bound", stats)

    def test_backend_keeps_v12_router_motion_head_and_shared_qk_gains(self):
        backend = LayerwiseSoftplusMaxSpeedBarrierVelocityGaussianBackend(
            v12_backend_cfg(), _gs_params(), dim=24, offset_bound=0.4,
            gaussian_count_cfg=router_count_cfg(),
        )
        temporal = backend.temporal
        self.assertIsInstance(
            temporal, LayerwiseSoftplusMaxSpeedBarrierCrossAttention
        )
        self.assertNotIsInstance(
            temporal, LearnableMaxSpeedBarrierLayerWeightedCrossAttention
        )
        self.assertFalse(hasattr(temporal, "v_0"))
        torch.testing.assert_close(
            temporal.v_max, torch.full((temporal.n_layers,), 30.0)
        )
        for layer in range(temporal.n_layers):
            for norm in (temporal.q_norm[layer], temporal.k_norm[layer]):
                self.assertIsInstance(norm, torch.nn.RMSNorm)
        self.assertIsInstance(backend.velocity_head, EmbeddedInitVelocityHead)
        self.assertEqual(backend.gaussian_head.count_mode, "learned_gumbel")
        self.assertAlmostEqual(backend.offset_bound, 0.4)


class V13CosineRoPEAttentionTest(unittest.TestCase):
    """V13 removes QK RMSNorm/barriers and fuses all RoPE heads."""

    @staticmethod
    def _forward_args(feature, position, first_frame_tokens, duration=0.8):
        return V11MaxSpeedBarrierAttentionTest._forward_args(
            feature, position, first_frame_tokens, duration
        )

    def test_only_head0_is_l2_normalized_at_temperature_007(self):
        torch.manual_seed(130)
        module = CosineRoPELayerWeightedCrossAttention(
            v13_temporal_cfg(heads=4, layers=2), dim=24
        )
        self.assertFalse(hasattr(module, "q_norm"))
        self.assertFalse(hasattr(module, "k_norm"))
        self.assertFalse(any(
            isinstance(child, torch.nn.RMSNorm) for child in module.modules()
        ))
        for removed in (
            "barrier_speed_mps", "v_0", "log_barrier_lambda",
            "distance_bias_speed_mps",
        ):
            self.assertFalse(hasattr(module, removed))

        query = torch.randn(3, 4, 6)
        key = torch.randn(5, 4, 6)
        query_angles = module.rope.angles(torch.randn(3, 3))
        key_angles = module.rope.angles(torch.randn(5, 3))
        encoded_query, encoded_key = module._encode_qk(
            query, key, query_angles, key_angles
        )

        unit_query = module.rope.rotate(
            F.normalize(query[:, :1].float(), dim=-1),
            query_angles[:, None, :, :],
        )
        unit_key = module.rope.rotate(
            F.normalize(key[:, :1].float(), dim=-1),
            key_angles[:, None, :, :],
        )
        actual_motion_score = (
            encoded_query[:, 0] @ encoded_key[:, 0].transpose(0, 1)
        ) * module.scale
        expected_motion_score = (
            unit_query[:, 0] @ unit_key[:, 0].transpose(0, 1)
        ) / 0.07
        torch.testing.assert_close(
            actual_motion_score, expected_motion_score,
            atol=2.0e-5, rtol=2.0e-5,
        )

        raw_query_rope = module.rope.rotate(
            query.float(), query_angles[:, None, :, :]
        )
        raw_key_rope = module.rope.rotate(
            key.float(), key_angles[:, None, :, :]
        )
        torch.testing.assert_close(encoded_query[:, 1:], raw_query_rope[:, 1:])
        torch.testing.assert_close(encoded_key[:, 1:], raw_key_rope[:, 1:])
        stats = module.attention_temperature_stats()
        self.assertEqual(stats["qk_norm_enabled"], 0.0)
        self.assertAlmostEqual(stats["motion_temperature"], 0.07)
        self.assertAlmostEqual(
            stats["motion_max_abs_content_logit"], 1.0 / 0.07
        )

    def test_every_head_uses_one_attention_call_per_layer(self):
        torch.manual_seed(131)
        module = CosineRoPELayerWeightedCrossAttention(
            v13_temporal_cfg(heads=4, layers=3), dim=24
        )
        calls = []

        def capture_flash(query, key, value, q_counts, k_counts,
                          *, flash_dtype=None):
            calls.append((
                query.detach().clone(), key.detach().clone(),
                value.detach().clone(), flash_dtype,
            ))
            return module._sdpa_attention(
                query, key, value, q_counts, k_counts
            )

        module._use_flash = lambda _value: True
        module._flash_attention = capture_flash
        feature = torch.randn(9, 24, requires_grad=True)
        position = torch.randn(9, 3) * 4.0
        refined, fields = module(
            *self._forward_args(feature, position, 4)
        )
        self.assertEqual(len(calls), 3)
        for query, key, value, flash_dtype in calls:
            self.assertEqual(tuple(query.shape[1:]), (4, 16))
            self.assertEqual(tuple(key.shape[1:]), (4, 16))
            self.assertEqual(tuple(value.shape[1:]), (4, 16))
            self.assertIs(flash_dtype, torch.float16)
            torch.testing.assert_close(query[..., 6:], torch.zeros_like(query[..., 6:]))
            torch.testing.assert_close(key[..., 6:], torch.zeros_like(key[..., 6:]))
            # Only head 0 carries the three xyz readout channels.
            torch.testing.assert_close(
                value[:, 1:, 6:], torch.zeros_like(value[:, 1:, 6:])
            )
        self.assertEqual(tuple(refined.shape), (9, 24))
        self.assertEqual(tuple(fields["motion_layer_weights"].shape), (9, 3))
        torch.testing.assert_close(
            fields["motion_layer_weights"], torch.full((9, 3), 1.0 / 3.0)
        )
        self.assertTrue(bool(torch.isfinite(fields["matched_position"]).all()))

        (refined.square().mean() + fields["matched_position"].square().mean()).backward()
        self.assertGreater(float(feature.grad.abs().sum()), 0.0)
        for layer in range(module.n_layers):
            self.assertGreater(
                float(module.q_proj[layer].weight.grad.abs().sum()), 0.0
            )
            self.assertGreater(
                float(module.k_proj[layer].weight.grad.abs().sum()), 0.0
            )

    def test_v13_and_v13_1_backends_differ_only_in_count_path(self):
        adaptive = CosineRoPEVelocityGaussianBackend(
            v13_backend_cfg(), _gs_params(), dim=24, offset_bound=0.4,
            gaussian_count_cfg=router_count_cfg(),
        )
        fixed = SeedConditionedCosineRoPEVelocityGaussianBackend(
            v13_backend_cfg(), _gs_params(), dim=24, offset_bound=0.4,
            gaussians_per_token=2,
        ).eval()
        for backend in (adaptive, fixed):
            self.assertIsInstance(
                backend.temporal, CosineRoPELayerWeightedCrossAttention
            )
            self.assertIsInstance(backend.velocity_head, EmbeddedInitVelocityHead)
        self.assertAlmostEqual(adaptive.offset_bound, 0.4)
        self.assertAlmostEqual(fixed.gaussian_head.offset_bound, 0.4)
        self.assertTrue(adaptive.adaptive_count)
        self.assertEqual(adaptive.gaussian_head.count_mode, "learned_gumbel")
        self.assertFalse(fixed.adaptive_count)
        self.assertEqual(fixed.gaussians_per_token, 2)
        self.assertTrue(fixed.seed_conditioned_head)
        self.assertIsInstance(fixed.gaussian_head, GaussianAttributeHead)
        self.assertFalse(hasattr(fixed.gaussian_head, "count_predictor"))

        tokens = 6
        first = 3
        feature = torch.randn(tokens, 24)
        position = torch.randn(tokens, 3)
        delta = torch.tensor([[-0.1, 0.0, 0.0], [0.1, 0.0, 0.0]])
        seed_delta = delta.unsqueeze(0).expand(tokens, -1, -1).clone()
        seed = position.unsqueeze(1) + seed_delta
        output = fixed(
            feature,
            position,
            seed,
            torch.tensor([first, tokens]),
            torch.tensor([0, 0]),
            [torch.eye(4).repeat(2, 1, 1)],
            [torch.tensor([0.0, 1.0])],
            [torch.tensor([0.0, 0.8])],
            torch.tensor([0.8]),
            seed_delta_sensor=seed_delta,
        )
        item = output["batch_gaussians"][0]
        self.assertEqual(item["position"].shape[0], 2 * tokens)
        shared_velocity = item["velocity"].reshape(tokens, 2, 3)
        torch.testing.assert_close(
            shared_velocity, shared_velocity[:, :1].expand_as(shared_velocity)
        )


class V13_2RawQKRoPEAttentionTest(unittest.TestCase):
    """V13.2 gives head 0 the same raw QK score as every other head."""

    def test_all_heads_keep_raw_qk_and_shared_sqrt_d_scale(self):
        torch.manual_seed(132)
        module = RawQKRoPELayerWeightedCrossAttention(
            v13_2_temporal_cfg(heads=4, layers=2), dim=24
        )
        self.assertFalse(hasattr(module, "q_norm"))
        self.assertFalse(hasattr(module, "k_norm"))
        self.assertFalse(hasattr(module, "motion_temperature"))
        self.assertFalse(hasattr(module, "motion_qk_scale"))
        self.assertFalse(any(
            isinstance(child, torch.nn.RMSNorm) for child in module.modules()
        ))

        query = torch.randn(3, 4, 6)
        key = torch.randn(5, 4, 6)
        query_angles = module.rope.angles(torch.randn(3, 3))
        key_angles = module.rope.angles(torch.randn(5, 3))
        encoded_query, encoded_key = module._encode_qk(
            query, key, query_angles, key_angles
        )
        expected_query = module.rope.rotate(
            query.float(), query_angles[:, None, :, :]
        )
        expected_key = module.rope.rotate(
            key.float(), key_angles[:, None, :, :]
        )
        torch.testing.assert_close(encoded_query, expected_query)
        torch.testing.assert_close(encoded_key, expected_key)

        actual_motion_score = (
            encoded_query[:, 0] @ encoded_key[:, 0].transpose(0, 1)
        ) * module.scale
        expected_motion_score = (
            expected_query[:, 0] @ expected_key[:, 0].transpose(0, 1)
        ) / math.sqrt(module.head_dim)
        torch.testing.assert_close(actual_motion_score, expected_motion_score)

        scaled_query = query.clone()
        scaled_query[:, 0] *= 3.0
        scaled_encoded, _ = module._encode_qk(
            scaled_query, key, query_angles, key_angles
        )
        torch.testing.assert_close(
            scaled_encoded[:, 0], 3.0 * encoded_query[:, 0]
        )
        stats = module.attention_temperature_stats()
        self.assertEqual(stats["qk_norm_enabled"], 0.0)
        self.assertAlmostEqual(stats["motion_score_scale"], 1.0 / math.sqrt(6))
        self.assertNotIn("motion_temperature", stats)

    def test_v13_2_keeps_one_all_head_call_and_v13_adaptive_backend(self):
        torch.manual_seed(133)
        module = RawQKRoPELayerWeightedCrossAttention(
            v13_2_temporal_cfg(heads=4, layers=2), dim=24
        )
        calls = []

        def capture_flash(query, key, value, q_counts, k_counts,
                          *, flash_dtype=None):
            calls.append((query.shape, key.shape, value.shape, flash_dtype))
            return module._sdpa_attention(
                query, key, value, q_counts, k_counts
            )

        module._use_flash = lambda _value: True
        module._flash_attention = capture_flash
        feature = torch.randn(8, 24, requires_grad=True)
        position = torch.randn(8, 3)
        time = torch.tensor([0.0] * 4 + [0.8] * 4)
        refined, fields = module(
            feature,
            position,
            torch.tensor([4, 8]),
            torch.tensor([0, 0]),
            time,
            torch.full((8,), 0.8),
        )
        self.assertEqual(len(calls), 2)
        for query_shape, key_shape, value_shape, flash_dtype in calls:
            self.assertEqual(tuple(query_shape[1:]), (4, 16))
            self.assertEqual(tuple(key_shape[1:]), (4, 16))
            self.assertEqual(tuple(value_shape[1:]), (4, 16))
            self.assertIs(flash_dtype, torch.float16)
        (refined.square().mean() + fields["matched_position"].square().mean()).backward()
        self.assertGreater(float(feature.grad.abs().sum()), 0.0)

        backend = RawQKRoPEVelocityGaussianBackend(
            v13_2_backend_cfg(), _gs_params(), dim=24, offset_bound=0.4,
            gaussian_count_cfg=router_count_cfg(),
        )
        self.assertIsInstance(
            backend.temporal, RawQKRoPELayerWeightedCrossAttention
        )
        self.assertTrue(backend.adaptive_count)
        self.assertEqual(backend.gaussian_head.count_mode, "learned_gumbel")
        self.assertAlmostEqual(backend.offset_bound, 0.4)


class V13_3RMSNormRoPEAttentionTest(unittest.TestCase):
    """V13.3 applies V11.4 Q/K RMSNorm before all-head 3D RoPE."""

    def test_all_heads_are_rms_normalized_then_rotated(self):
        torch.manual_seed(134)
        module = RMSNormRoPELayerWeightedCrossAttention(
            v13_3_temporal_cfg(heads=4, layers=2), dim=24
        )
        self.assertTrue(module.qk_norm)
        self.assertEqual(len(module.q_norm), 2)
        self.assertEqual(len(module.k_norm), 2)
        self.assertTrue(all(
            isinstance(norm, torch.nn.RMSNorm) for norm in module.q_norm
        ))
        self.assertFalse(hasattr(module, "motion_temperature"))
        self.assertFalse(hasattr(module, "motion_qk_scale"))
        for removed in (
            "barrier_speed_mps", "v_0", "log_barrier_lambda",
            "distance_bias_speed_mps",
        ):
            self.assertFalse(hasattr(module, removed))

        query = torch.randn(3, 4, 6)
        key = torch.randn(5, 4, 6)
        query_angles = module.rope.angles(torch.randn(3, 3))
        key_angles = module.rope.angles(torch.randn(5, 3))
        normalized_query = module.q_norm[0](query)
        normalized_key = module.k_norm[0](key)
        encoded_query, encoded_key = module._encode_qk(
            normalized_query, normalized_key, query_angles, key_angles
        )
        torch.testing.assert_close(
            encoded_query,
            module.rope.rotate(
                normalized_query.float(), query_angles[:, None, :, :]
            ),
        )
        torch.testing.assert_close(
            encoded_key,
            module.rope.rotate(
                normalized_key.float(), key_angles[:, None, :, :]
            ),
        )
        torch.testing.assert_close(
            normalized_query.square().mean(dim=-1),
            torch.ones(3, 4),
            atol=2.0e-5,
            rtol=2.0e-5,
        )

        stats = module.attention_temperature_stats()
        self.assertEqual(stats["qk_norm_enabled"], 1.0)
        self.assertAlmostEqual(stats["qk_gamma_q_layer0"], 1.0)
        self.assertAlmostEqual(
            stats["qk_max_logit_bound"], math.sqrt(module.head_dim)
        )

    def test_projection_gain_is_removed_and_all_heads_share_one_call(self):
        torch.manual_seed(135)
        module = RMSNormRoPELayerWeightedCrossAttention(
            v13_3_temporal_cfg(heads=4, layers=2), dim=24
        ).eval()
        calls = []

        def capture_flash(query, key, value, q_counts, k_counts,
                          *, flash_dtype=None):
            calls.append((query.shape, key.shape, value.shape, flash_dtype))
            return module._sdpa_attention(
                query, key, value, q_counts, k_counts
            )

        module._use_flash = lambda _value: True
        module._flash_attention = capture_flash
        feature = torch.randn(8, 24)
        position = torch.randn(8, 3)
        time = torch.tensor([0.0] * 4 + [0.8] * 4)
        args = (
            feature,
            position,
            torch.tensor([4, 8]),
            torch.tensor([0, 0]),
            time,
            torch.full((8,), 0.8),
        )
        with torch.no_grad():
            refined_before, fields_before = module(*args)
        self.assertEqual(len(calls), 2)
        for query_shape, key_shape, value_shape, flash_dtype in calls:
            self.assertEqual(tuple(query_shape[1:]), (4, 16))
            self.assertEqual(tuple(key_shape[1:]), (4, 16))
            self.assertEqual(tuple(value_shape[1:]), (4, 16))
            self.assertIs(flash_dtype, torch.float16)

        with torch.no_grad():
            for layer in range(module.n_layers):
                module.q_proj[layer].weight.mul_(100.0)
                module.q_proj[layer].bias.mul_(100.0)
                module.k_proj[layer].weight.mul_(100.0)
                module.k_proj[layer].bias.mul_(100.0)
            calls.clear()
            refined_after, fields_after = module(*args)
        self.assertEqual(len(calls), 2)
        torch.testing.assert_close(
            refined_after, refined_before, atol=2.0e-5, rtol=2.0e-5
        )
        torch.testing.assert_close(
            fields_after["matched_position"],
            fields_before["matched_position"],
            atol=2.0e-5,
            rtol=2.0e-5,
        )

        backend = RMSNormRoPEVelocityGaussianBackend(
            v13_3_backend_cfg(), _gs_params(), dim=24, offset_bound=0.4,
            gaussian_count_cfg=router_count_cfg(),
        )
        self.assertIsInstance(
            backend.temporal, RMSNormRoPELayerWeightedCrossAttention
        )
        self.assertTrue(backend.adaptive_count)
        self.assertEqual(backend.gaussian_head.count_mode, "learned_gumbel")
        self.assertAlmostEqual(backend.offset_bound, 0.4)


class V13_4RMSNormFeatureRoPEAttentionTest(unittest.TestCase):
    """V13.4 removes 3D RoPE from RMS-normalized match head 0 only."""

    def test_head0_is_unrotated_and_feature_heads_keep_rope(self):
        torch.manual_seed(136)
        module = RMSNormFeatureRoPELayerWeightedCrossAttention(
            v13_4_temporal_cfg(heads=4, layers=2), dim=24
        )
        self.assertTrue(module.qk_norm)
        self.assertEqual(module.position_encoding, "rope_except_head0")
        self.assertEqual(len(module.q_norm), 2)
        self.assertEqual(len(module.k_norm), 2)
        for removed in (
            "barrier_speed_mps", "v_0", "log_barrier_lambda",
            "distance_bias_speed_mps", "motion_temperature",
        ):
            self.assertFalse(hasattr(module, removed))

        query = torch.randn(3, 4, 6)
        key = torch.randn(5, 4, 6)
        query_angles = module.rope.angles(torch.randn(3, 3) * 4.0)
        key_angles = module.rope.angles(torch.randn(5, 3) * 4.0)
        normalized_query = module.q_norm[0](query)
        normalized_key = module.k_norm[0](key)
        encoded_query, encoded_key = module._encode_qk(
            normalized_query, normalized_key, query_angles, key_angles
        )

        torch.testing.assert_close(
            encoded_query[:, 0], normalized_query[:, 0].float()
        )
        torch.testing.assert_close(
            encoded_key[:, 0], normalized_key[:, 0].float()
        )
        torch.testing.assert_close(
            encoded_query[:, 1:],
            module.rope.rotate(
                normalized_query[:, 1:].float(),
                query_angles[:, None, :, :],
            ),
        )
        torch.testing.assert_close(
            encoded_key[:, 1:],
            module.rope.rotate(
                normalized_key[:, 1:].float(),
                key_angles[:, None, :, :],
            ),
        )

        actual_head0_score = (
            encoded_query[:, 0] @ encoded_key[:, 0].transpose(0, 1)
        ) * module.scale
        expected_head0_score = (
            normalized_query[:, 0].float()
            @ normalized_key[:, 0].float().transpose(0, 1)
        ) / math.sqrt(module.head_dim)
        torch.testing.assert_close(actual_head0_score, expected_head0_score)
        self.assertEqual(
            module.attention_temperature_stats()["qk_norm_enabled"], 1.0
        )

    def test_v13_4_keeps_one_all_head_call_and_v13_3_backend_contract(self):
        torch.manual_seed(137)
        module = RMSNormFeatureRoPELayerWeightedCrossAttention(
            v13_4_temporal_cfg(heads=4, layers=2), dim=24
        ).eval()
        calls = []

        def capture_flash(query, key, value, q_counts, k_counts,
                          *, flash_dtype=None):
            calls.append((query.shape, key.shape, value.shape, flash_dtype))
            return module._sdpa_attention(
                query, key, value, q_counts, k_counts
            )

        module._use_flash = lambda _value: True
        module._flash_attention = capture_flash
        feature = torch.randn(8, 24, requires_grad=True)
        position = torch.randn(8, 3)
        time = torch.tensor([0.0] * 4 + [0.8] * 4)
        refined, fields = module(
            feature,
            position,
            torch.tensor([4, 8]),
            torch.tensor([0, 0]),
            time,
            torch.full((8,), 0.8),
        )
        self.assertEqual(len(calls), 2)
        for query_shape, key_shape, value_shape, flash_dtype in calls:
            self.assertEqual(tuple(query_shape[1:]), (4, 16))
            self.assertEqual(tuple(key_shape[1:]), (4, 16))
            self.assertEqual(tuple(value_shape[1:]), (4, 16))
            self.assertIs(flash_dtype, torch.float16)
        (refined.square().mean() + fields["matched_position"].square().mean()).backward()
        self.assertGreater(float(feature.grad.abs().sum()), 0.0)

        backend = RMSNormFeatureRoPEVelocityGaussianBackend(
            v13_4_backend_cfg(), _gs_params(), dim=24, offset_bound=0.4,
            gaussian_count_cfg=router_count_cfg(),
        )
        self.assertIsInstance(
            backend.temporal, RMSNormFeatureRoPELayerWeightedCrossAttention
        )
        self.assertTrue(backend.adaptive_count)
        self.assertEqual(backend.gaussian_head.count_mode, "learned_gumbel")
        self.assertAlmostEqual(backend.offset_bound, 0.4)


class V11_2FinalFeatureBarrierAttentionTest(unittest.TestCase):
    """V11.2 changes only the source of V11's coordinate readout."""

    @staticmethod
    def _forward_args(feature, position, first_frame_tokens, duration=0.8):
        return V11MaxSpeedBarrierAttentionTest._forward_args(
            feature, position, first_frame_tokens, duration
        )

    def test_432d_readout_head_mirrors_one_v11_readout_head(self):
        torch.manual_seed(101)
        module = FinalFeatureBarrierCrossAttention(
            v11_2_temporal_cfg(heads=12, layers=1), dim=432
        )
        # Same chain and same widths as V11's LN -> head-0 rows -> RMSNorm.
        self.assertEqual(module.head_dim, 36)
        self.assertEqual(module.match_input_norm.normalized_shape, (432,))
        self.assertEqual(module.match_q_proj.in_features, 432)
        self.assertEqual(module.match_q_proj.out_features, 36)
        self.assertEqual(module.match_k_proj.in_features, 432)
        self.assertEqual(module.match_k_proj.out_features, 36)
        self.assertIsNotNone(module.match_q_proj.bias)
        self.assertIsNotNone(module.match_k_proj.bias)
        self.assertEqual(tuple(module.match_q_norm.weight.shape), (36,))
        self.assertEqual(tuple(module.match_k_norm.weight.shape), (36,))

        # Linear draws from fan_in, which is 432 for both the readout head and
        # V11's Linear(432,432), so the head starts at V11's own scale.
        bound = 432 ** -0.5
        for projection in (module.match_q_proj, module.match_k_proj):
            self.assertLessEqual(float(projection.weight.abs().max()), bound)
            self.assertLessEqual(float(projection.bias.abs().max()), bound)
            self.assertGreater(float(projection.weight.abs().max()), 0.5 * bound)
        torch.testing.assert_close(
            module.match_q_norm.weight, torch.ones(36)
        )
        torch.testing.assert_close(
            module.match_k_norm.weight, torch.ones(36)
        )

        refined = torch.randn(5, 432)
        memory_rows = torch.tensor([3, 4, 3, 0, 1])
        query, key = module._final_match_descriptors(refined, memory_rows)
        conditioned = module.match_input_norm(refined)
        torch.testing.assert_close(
            query, module.match_q_norm(module.match_q_proj(conditioned)).float()
        )
        torch.testing.assert_close(
            key,
            module.match_k_norm(
                module.match_k_proj(conditioned)[memory_rows]
            ).float(),
        )

    def test_state_schema_adds_the_readout_head_and_drops_layer_logits(self):
        torch.manual_seed(102)
        v11 = MaxSpeedBarrierLayerWeightedCrossAttention(
            v11_temporal_cfg(layers=2), dim=24
        )
        torch.manual_seed(102)
        v11_2 = FinalFeatureBarrierCrossAttention(
            v11_2_temporal_cfg(layers=2), dim=24
        )
        v11_names = [name for name, _p in v11.named_parameters()]
        v11_2_names = [name for name, _p in v11_2.named_parameters()]
        # V11's layer mixture weighted one coordinate readout per layer. V11.2
        # has no per-layer readouts, so the scalar is removed rather than left
        # in the checkpoint collecting no gradient.
        self.assertIn("layer_logits", v11_names)
        self.assertNotIn("layer_logits", v11_2_names)
        self.assertFalse(hasattr(v11_2, "layer_logits"))

        added = [n for n in v11_2_names if n.startswith("match_")]
        self.assertEqual(added, [
            "match_input_norm.weight", "match_input_norm.bias",
            "match_q_proj.weight", "match_q_proj.bias",
            "match_k_proj.weight", "match_k_proj.bias",
            "match_q_norm.weight", "match_k_norm.weight",
        ])
        # Everything else is V11's stack, in V11's order, appended to.
        self.assertEqual(
            [n for n in v11_2_names if not n.startswith("match_")],
            [n for n in v11_names if n != "layer_logits"],
        )
        head_dim, dim = v11_2.head_dim, v11_2.dim
        self.assertEqual(
            sum(p.numel() for n, p in v11_2.named_parameters()
                if n.startswith("match_")),
            2 * dim + 2 * (head_dim * dim + head_dim) + 2 * head_dim,
        )
        # One readout head, not one per layer: depth must not grow it.
        deeper = FinalFeatureBarrierCrossAttention(
            v11_2_temporal_cfg(layers=5), dim=24
        )
        self.assertEqual(
            [n for n in deeper.state_dict() if n.startswith("match_")],
            [n for n in v11_2.state_dict() if n.startswith("match_")],
        )

    def test_readout_head_gain_is_reported_next_to_v11s_layer_gains(self):
        torch.manual_seed(107)
        module = FinalFeatureBarrierCrossAttention(
            v11_2_temporal_cfg(layers=2), dim=24
        )
        stats = module.attention_temperature_stats()
        for name in ("qk_gamma_q_layer0", "qk_gamma_q_layer1",
                     "qk_max_logit_bound", "barrier_weight"):
            self.assertIn(name, stats)
        self.assertAlmostEqual(stats["qk_match_gamma_q"], 1.0, places=5)
        self.assertAlmostEqual(stats["qk_match_gamma_k"], 1.0, places=5)
        self.assertAlmostEqual(
            stats["qk_match_logit_bound"], module.head_dim ** 0.5, places=4
        )
        with torch.no_grad():
            module.match_q_norm.weight.mul_(3.0)
        self.assertAlmostEqual(
            module.attention_temperature_stats()["qk_match_logit_bound"],
            3.0 * module.head_dim ** 0.5, places=4,
        )

    def test_every_layer_head_carries_rope_and_no_layer_reads_coordinates(self):
        """The layer stack is plain all-head RoPE cross-attention.

        V11 holds head 0 out of the kernel to give it the barrier. V11.2 reads
        correspondence after the stack, so nothing inside needs geometry beyond
        RoPE and every head goes through one attention call.
        """
        torch.manual_seed(103)
        module = FinalFeatureBarrierCrossAttention(
            v11_2_temporal_cfg(heads=4, layers=3), dim=24
        ).eval()
        self.assertEqual(module.rope_head_count, module.num_heads)
        feature = torch.randn(9, 24)
        position = torch.randn(9, 3) * 3.0
        args = self._forward_args(feature, position, 4)

        barrier_calls = []
        original_barrier = module._barrier_attention
        original_combine = module._combine_layer_matches

        def counting_barrier(*call_args, **call_kwargs):
            barrier_calls.append(call_args[0].shape)
            return original_barrier(*call_args, **call_kwargs)

        def forbidden_combine(*_args, **_kwargs):
            raise AssertionError("V11.2 must not mix per-layer readouts")

        module._barrier_attention = counting_barrier
        module._combine_layer_matches = forbidden_combine
        with torch.no_grad():
            refined, fields = module(*args)
        # Exactly one barrier softmax per forward: the final readout.
        self.assertEqual(len(barrier_calls), 1)
        self.assertEqual(barrier_calls[0], (9, module.head_dim))
        self.assertEqual(
            set(fields), {"delta_p_match", "delta_p_init", "matched_position"}
        )

        # Reference layer stack: RoPE on all heads, one SDPA call per layer.
        module._barrier_attention = original_barrier
        module._combine_layer_matches = original_combine
        q_counts, k_counts, memory_rows = module._frame_layout(args[2], args[3])
        q_angles = module.rope.angles(position.float())
        k_angles = module.rope.angles(position[memory_rows].float())
        with torch.no_grad():
            x = feature + module.time_to_feature(
                module.time_encoder(args[4])
            ).to(feature.dtype)
            for layer in range(module.n_layers):
                c = module.input_norm[layer](x)
                shape = (9, module.num_heads, module.head_dim)
                q = module.q_norm[layer](
                    module.q_proj[layer](c).reshape(shape)
                ).float()
                k = module.k_norm[layer](
                    module.k_proj[layer](c)[memory_rows].reshape(shape)
                ).float()
                v = module.v_proj[layer](c)[memory_rows].reshape(shape).float()
                attended = module._sdpa_attention(
                    module.rope.rotate(q, q_angles[:, None, :, :]),
                    module.rope.rotate(k, k_angles[:, None, :, :]),
                    v, q_counts, k_counts,
                ).reshape(9, 24).to(x.dtype)
                x = x + module.cross_layer_scale[layer] * module.out_proj[
                    layer
                ](attended)
                x = x + module.ffn[layer](module.ffn_norm[layer](x))
        torch.testing.assert_close(refined, x, atol=0.0, rtol=0.0)

    def test_layer_stack_differs_from_v11_because_head0_gained_rope(self):
        """The refinement is deliberately no longer bit-identical to V11.

        V11.2 moves two things at once, and this pins the second: with head 0
        on RoPE the barrier is gone from the layer loop, so f' must differ.
        """
        torch.manual_seed(109)
        v11 = MaxSpeedBarrierLayerWeightedCrossAttention(
            v11_temporal_cfg(layers=3), dim=24
        ).eval()
        v11_2 = FinalFeatureBarrierCrossAttention(
            v11_2_temporal_cfg(layers=3), dim=24
        ).eval()
        shared = {
            name: value for name, value in v11.state_dict().items()
            if name != "layer_logits"
        }
        incompatible = v11_2.load_state_dict(shared, strict=False)
        self.assertEqual(incompatible.unexpected_keys, [])
        self.assertTrue(all(
            key.startswith("match_") for key in incompatible.missing_keys
        ))
        feature = torch.randn(9, 24)
        position = torch.randn(9, 3) * 3.0
        args = self._forward_args(feature, position, 4)
        with torch.no_grad():
            refined_v11, fields_v11 = v11(*args)
            refined_v11_2, fields_v11_2 = v11_2(*args)
        self.assertGreater(
            float((refined_v11_2 - refined_v11).abs().max()), 1.0e-4
        )
        self.assertIn("motion_layer_weights", fields_v11)
        self.assertNotIn("motion_layer_weights", fields_v11_2)

    def test_final_readout_is_exact_barrier_softmax_over_key_coordinates(self):
        torch.manual_seed(104)
        module = FinalFeatureBarrierCrossAttention(
            v11_2_temporal_cfg(layers=2, match_chunk_size=2), dim=24
        ).eval()
        feature = torch.randn(9, 24)
        position = torch.randn(9, 3) * 20.0
        duration = 0.7
        args = self._forward_args(feature, position, 4, duration)
        with torch.no_grad():
            refined, fields = module(*args)
            q_counts, k_counts, memory_rows = module._frame_layout(
                args[2], args[3]
            )
            query, key = module._final_match_descriptors(
                refined, memory_rows
            )

        key_position = position[memory_rows]
        query_start = 0
        key_start = 0
        radius = module.barrier_speed_mps * duration
        for q_count_tensor, k_count_tensor in zip(q_counts, k_counts):
            q_count = int(q_count_tensor)
            k_count = int(k_count_tensor)
            rows = slice(query_start, query_start + q_count)
            keys = slice(key_start, key_start + k_count)
            score = query[rows] @ key[keys].transpose(0, 1) * module.scale
            score = score - module.barrier_weight * torch.relu(
                torch.cdist(position[rows], key_position[keys]) / radius - 1.0
            ).square()
            probability = torch.softmax(score, dim=-1)
            expected_position = probability @ key_position[keys]
            torch.testing.assert_close(
                fields["matched_position"][rows], expected_position,
                atol=1.0e-4, rtol=1.0e-4,
            )
            query_start += q_count
            key_start += k_count
        torch.testing.assert_close(
            fields["delta_p_init"], fields["matched_position"] - position
        )
        torch.testing.assert_close(
            fields["delta_p_match"], fields["delta_p_init"]
        )

    def test_final_feature_and_readout_head_receive_correspondence_gradient(self):
        torch.manual_seed(105)
        module = FinalFeatureBarrierCrossAttention(
            v11_2_temporal_cfg(layers=2), dim=24
        ).train()
        feature = torch.randn(8, 24, requires_grad=True)
        position = torch.randn(8, 3) * 4.0
        refined, fields = module(
            *self._forward_args(feature, position, 4)
        )
        refined.retain_grad()
        fields["matched_position"].square().sum().backward()
        self.assertGreater(float(refined.grad.abs().sum()), 0.0)
        self.assertGreater(float(feature.grad.abs().sum()), 0.0)
        for name, parameter in module.named_parameters():
            if name.startswith("match_"):
                self.assertIsNotNone(parameter.grad, name)
                self.assertGreater(float(parameter.grad.abs().sum()), 0.0, name)
        # V11's layer-mixture scalar does not exist here at all.
        self.assertFalse(hasattr(module, "layer_logits"))

    def test_readout_path_touches_no_layer_parameter_of_its_own(self):
        """Layer 11's Q/K see correspondence only through the trunk.

        The readout owns its projection, so a gradient taken on the descriptor
        path alone must leave every temporal-layer parameter untouched. That is
        what separates "f' is a worse source" from "layer 11's weights were
        split across two jobs".
        """
        torch.manual_seed(108)
        module = FinalFeatureBarrierCrossAttention(
            v11_2_temporal_cfg(layers=2), dim=24
        ).train()
        refined = torch.randn(8, 24)
        memory_rows = torch.tensor([4, 5, 6, 7, 0, 1, 2, 3])
        query, key = module._final_match_descriptors(refined, memory_rows)
        (query.square().sum() + key.square().sum()).backward()
        for name, parameter in module.named_parameters():
            if name.startswith("match_"):
                self.assertIsNotNone(parameter.grad, name)
                self.assertGreater(float(parameter.grad.abs().sum()), 0.0, name)
            else:
                self.assertIsNone(parameter.grad, name)

    def test_backend_keeps_embedded_offset_and_child_velocity_sharing(self):
        torch.manual_seed(106)
        backend = FinalFeatureBarrierVelocityGaussianBackend(
            v11_2_backend_cfg(), _gs_params(), dim=24, offset_bound=0.8,
            gaussian_count_cfg=router_count_cfg(),
        ).eval()
        self.assertIsInstance(
            backend.temporal, FinalFeatureBarrierCrossAttention
        )
        self.assertIsInstance(backend.velocity_head, EmbeddedInitVelocityHead)
        self.assertTrue(backend.velocity_head.detach_init_condition)
        self.assertEqual(backend.gaussian_head.count_mode, "learned_gumbel")
        with torch.no_grad():
            backend.gaussian_head.count_predictor[-1].bias.copy_(
                torch.tensor([-10.0, -10.0, 10.0])
            )

        tokens, first, duration = 8, 4, 0.8
        feature = torch.randn(tokens, 24)
        position = torch.randn(tokens, 3) * 2.0
        seed, seed_delta = adaptive_seed_bank(
            position
        )
        with torch.no_grad():
            output = backend(
                feature,
                position,
                seed,
                torch.tensor([first, tokens]),
                torch.tensor([0, 0]),
                [torch.eye(4).repeat(2, 1, 1)],
                [torch.tensor([0.0, 1.0])],
                [torch.tensor([0.0, duration])],
                torch.tensor([duration]),
                seed_delta_sensor=seed_delta,
            )
        item = output["batch_gaussians"][0]
        self.assertTrue(torch.equal(item["selected_k"], torch.full((24,), 3)))
        shared_velocity = item["velocity"].reshape(tokens, 3, 3)
        torch.testing.assert_close(
            shared_velocity, shared_velocity[:, :1].expand_as(shared_velocity)
        )
        torch.testing.assert_close(
            item["velocity_offset"], torch.zeros_like(item["velocity_offset"])
        )
        torch.testing.assert_close(
            item["velocity"], item["velocity_init"] + item["velocity_offset"]
        )
        torch.testing.assert_close(
            item["velocity_init"] * item["pair_delta_t_sec"].unsqueeze(-1),
            item["delta_p_init"],
        )
        self.assertNotIn("motion_layer_logits", item)
        self.assertNotIn("motion_layer_weights", item)
        stats = ModelWrapper._motion_statistics([item])
        self.assertFalse(any("motion_layer" in name for name in stats))


class V11_1GroupedGainSingleGaussianTest(unittest.TestCase):
    """V11.1 = V11's stack, a two-group QK gain, one Gaussian per token."""

    @staticmethod
    def _forward_args(feature, position, first_frame_tokens, duration=0.8):
        total = feature.shape[0]
        time = torch.cat([
            torch.zeros(first_frame_tokens),
            torch.full((total - first_frame_tokens,), duration),
        ])
        return (
            feature,
            position,
            torch.tensor([first_frame_tokens, total]),
            torch.tensor([0, 0]),
            time,
            torch.full((total,), duration),
        )

    def test_the_gain_carries_one_row_per_head_group(self):
        module = GroupedGainBarrierCrossAttention(
            v11_1_temporal_cfg(layers=2), dim=24
        )
        for layer in range(2):
            for norm in (module.q_norm[layer], module.k_norm[layer]):
                self.assertIsInstance(norm, GroupedHeadRMSNorm)
                self.assertEqual(norm.num_groups, 2)
                self.assertEqual(
                    tuple(norm.weight.shape), (2, module.head_dim)
                )
                # Group 0 is the match head; every RoPE head shares group 1.
                self.assertEqual(
                    norm.group_of_head.tolist(), [0, 1, 1, 1]
                )

    def test_a_uniform_gain_reproduces_the_v11_readout(self):
        torch.manual_seed(0)
        module = GroupedGainBarrierCrossAttention(
            v11_1_temporal_cfg(layers=2), dim=24
        ).eval()
        reference = MaxSpeedBarrierLayerWeightedCrossAttention(
            v11_temporal_cfg(layers=2), dim=24
        ).eval()
        state = module.state_dict()
        collapsed = {}
        for name, value in state.items():
            if ".weight" in name and (
                name.startswith("q_norm") or name.startswith("k_norm")
            ):
                # Both rows equal, so the split gain is the shared gain.
                value = value[0]
            collapsed[name] = value
        reference.load_state_dict(collapsed, strict=True)

        feature = torch.randn(9, 24)
        position = torch.randn(9, 3) * 4.0
        args = self._forward_args(feature, position, 4)
        with torch.no_grad():
            refined, fields = module(*args)
            ref_refined, ref_fields = reference(*args)
        # Not bitwise: nn.RMSNorm runs a fused kernel while the grouped gain
        # does the same arithmetic with an explicit rsqrt and an index gather,
        # so the two reduce in a different order. On a real 28,760-token batch
        # the readouts agree to a relative 2.7e-06, against a 1.07 m response
        # to actually scaling the match group's gain.
        torch.testing.assert_close(refined, ref_refined)
        self.assertEqual(sorted(fields), sorted(ref_fields))
        for key in ("delta_p_init", "delta_p_match", "matched_position"):
            torch.testing.assert_close(fields[key], ref_fields[key])

    def test_each_group_moves_under_its_own_gradient(self):
        torch.manual_seed(1)
        module = GroupedGainBarrierCrossAttention(
            v11_1_temporal_cfg(layers=1), dim=24
        )
        feature = torch.randn(8, 24, requires_grad=True)
        position = torch.randn(8, 3) * 3.0
        _refined, fields = module(*self._forward_args(feature, position, 4))
        fields["delta_p_init"].square().sum().backward()
        gain = module.q_norm[0].weight.grad
        self.assertIsNotNone(gain)
        self.assertEqual(tuple(gain.shape), (2, module.head_dim))
        # The readout only passes through head 0, so only its row is charged
        # by a displacement objective. That separation is the whole point.
        self.assertGreater(float(gain[0].abs().sum()), 0.0)
        self.assertEqual(float(gain[1].abs().sum()), 0.0)

    def test_stats_report_the_two_groups_separately(self):
        module = GroupedGainBarrierCrossAttention(
            v11_1_temporal_cfg(layers=2), dim=24
        )
        with torch.no_grad():
            module.q_norm[0].weight[0].mul_(2.0)
        stats = module.attention_temperature_stats()
        self.assertAlmostEqual(stats["qk_gamma_q_match_layer0"], 2.0, places=5)
        self.assertAlmostEqual(stats["qk_gamma_q_rope_layer0"], 1.0, places=5)
        self.assertGreater(
            stats["qk_max_logit_bound_match"],
            stats["qk_max_logit_bound_rope"],
        )
        self.assertEqual(
            stats["qk_max_logit_bound"], stats["qk_max_logit_bound_match"]
        )
        # The gate is gone; nothing should still advertise it.
        for name in stats:
            self.assertNotIn("support", name)

    def test_no_gate_field_survives_on_the_temporal_output(self):
        torch.manual_seed(2)
        module = GroupedGainBarrierCrossAttention(
            v11_1_temporal_cfg(layers=1), dim=24
        ).eval()
        feature = torch.randn(8, 24)
        position = torch.randn(8, 3) * 3.0
        with torch.no_grad():
            _refined, fields = module(*self._forward_args(feature, position, 4))
        self.assertEqual(
            set(fields),
            {
                "delta_p_match", "delta_p_init", "matched_position",
                "motion_layer_logits", "motion_layer_weights",
            },
        )
        # Nothing scales the readout any more, so the two are the same tensor.
        torch.testing.assert_close(
            fields["delta_p_init"], fields["delta_p_match"]
        )
        self.assertFalse(
            any(hasattr(module, name) for name in (
                "support_lo", "support_rho_m", "support_gate_width",
            ))
        )

    def test_the_backend_emits_exactly_one_gaussian_per_token(self):
        torch.manual_seed(3)
        backend = SingleGaussianBarrierVelocityGaussianBackend(
            v11_1_backend_cfg(dim=24), _gs_params(), dim=24, offset_bound=0.8,
        ).eval()
        self.assertIsInstance(
            backend.temporal, GroupedGainBarrierCrossAttention
        )
        self.assertIsInstance(backend.gaussian_head, GaussianAttributeHead)
        # No router, no seed bank, no per-K experts.
        for name in ("count_predictor", "k_heads", "gaussian_output_norm"):
            self.assertFalse(hasattr(backend, name))
            self.assertFalse(hasattr(backend.gaussian_head, name))

        tokens, first = 10, 5
        feature = torch.randn(tokens, 24)
        position = torch.randn(tokens, 3) * 4.0
        with torch.no_grad():
            out = backend(
                feature,
                position,
                position.unsqueeze(1).clone(),
                torch.tensor([first, tokens]),
                torch.tensor([0, 0]),
                [torch.eye(4).repeat(2, 1, 1)],
                [torch.tensor([0.0, 1.0])],
                [torch.tensor([0.0, 0.8])],
                torch.tensor([0.8]),
            )
        item = out["batch_gaussians"][0]
        self.assertEqual(item["position"].shape, (tokens, 3))
        self.assertEqual(item["velocity"].shape, (tokens, 3))
        # One Gaussian per token means the packing indices vanish entirely.
        for name in ("selected_k", "gaussian_slot_index", "source_token_index"):
            self.assertNotIn(name, item)
        self.assertIsNone(out.get("routing_stats"))

    def test_the_backend_publishes_the_v11_motion_fields(self):
        torch.manual_seed(4)
        backend = SingleGaussianBarrierVelocityGaussianBackend(
            v11_1_backend_cfg(dim=24), _gs_params(), dim=24, offset_bound=0.8,
        ).eval()
        tokens, first, duration = 10, 5, 0.8
        feature = torch.randn(tokens, 24)
        position = torch.randn(tokens, 3) * 4.0
        with torch.no_grad():
            out = backend(
                feature,
                position,
                position.unsqueeze(1).clone(),
                torch.tensor([first, tokens]),
                torch.tensor([0, 0]),
                [torch.eye(4).repeat(2, 1, 1)],
                [torch.tensor([0.0, 1.0])],
                [torch.tensor([0.0, duration])],
                torch.tensor([duration]),
            )
        item = out["batch_gaussians"][0]
        for name in (
            "velocity_init", "velocity_offset", "velocity_match",
            "delta_p_init", "delta_p_match", "matched_position",
            "motion_layer_weights",
        ):
            self.assertIn(name, item)
        # Nothing is gated, so match and init agree and the prior has one value
        # to charge rather than two.
        torch.testing.assert_close(
            item["velocity_match"], item["velocity_init"]
        )
        self.assertNotIn("velocity_init_ungated", item)
        torch.testing.assert_close(
            item["velocity"],
            item["velocity_init"] + item["velocity_offset"],
        )
        # Frame 0 travels forward, frame 1 backward, over one endpoint interval.
        signed = torch.where(
            torch.arange(tokens) < first,
            torch.full((tokens,), duration),
            torch.full((tokens,), -duration),
        )
        torch.testing.assert_close(
            item["velocity_init"],
            item["delta_p_init"] / signed.unsqueeze(-1),
        )

    def test_zero_initialized_offset_starts_at_the_initializer(self):
        torch.manual_seed(5)
        backend = SingleGaussianBarrierVelocityGaussianBackend(
            v11_1_backend_cfg(dim=24), _gs_params(), dim=24, offset_bound=0.8,
        ).eval()
        tokens, first = 8, 4
        feature = torch.randn(tokens, 24)
        position = torch.randn(tokens, 3) * 4.0
        with torch.no_grad():
            out = backend(
                feature,
                position,
                position.unsqueeze(1).clone(),
                torch.tensor([first, tokens]),
                torch.tensor([0, 0]),
                [torch.eye(4).repeat(2, 1, 1)],
                [torch.tensor([0.0, 1.0])],
                [torch.tensor([0.0, 0.8])],
                torch.tensor([0.8]),
            )
        item = out["batch_gaussians"][0]
        torch.testing.assert_close(
            item["velocity_offset"], torch.zeros_like(item["velocity_offset"])
        )
        torch.testing.assert_close(item["velocity"], item["velocity_init"])


if __name__ == "__main__":
    unittest.main()
