"""Checkpoint and optimizer compatibility contracts for Dynamic 2DGS.

The digests below were captured from pre-refactor commit
``a44d941a236948b1d9248936deaf2616d12de453``. They deliberately cover ordered
names, tensor shapes, and dtypes without making the test depend on a Git
checkout or copying a large historical source file.
"""
from __future__ import annotations

import hashlib
import importlib
import pathlib
import sys
import types
import unittest

import torch
from omegaconf import OmegaConf

from src.experiment_config import (
    DYNAMIC_VARIANTS,
    ROUTED_COUNT_BLOCKS,
    compose_fresh_config,
)


_BACKEND_CLASS = {
    "dynamic_2dgs_attention_velocity_v11": (
        "MaxSpeedBarrierVelocityGaussianBackend"
    ),
    "dynamic_2dgs_attention_velocity_v11_1": (
        "SingleGaussianBarrierVelocityGaussianBackend"
    ),
    "dynamic_2dgs_attention_velocity_v11_2": (
        "FinalFeatureBarrierVelocityGaussianBackend"
    ),
    "dynamic_2dgs_attention_velocity_v11_3": (
        "SeedConditionedBarrierVelocityGaussianBackend"
    ),
    "dynamic_2dgs_attention_velocity_v11_4": (
        "MaxSpeedBarrierVelocityGaussianBackend"
    ),
    "dynamic_2dgs_attention_velocity_v11_5": (
        "SeedConditionedBarrierVelocityGaussianBackend"
    ),
    "dynamic_2dgs_attention_velocity_v11_6": (
        "SeedConditionedBarrierVelocityGaussianBackend"
    ),
    "dynamic_2dgs_attention_velocity_v12": (
        "LearnableMaxSpeedBarrierVelocityGaussianBackend"
    ),
    "dynamic_2dgs_attention_velocity_v12_1": (
        "FixedMotionGainMaxSpeedBarrierVelocityGaussianBackend"
    ),
    "dynamic_2dgs_attention_velocity_v12_2": (
        "LayerwiseSoftplusMaxSpeedBarrierVelocityGaussianBackend"
    ),
    "dynamic_2dgs_attention_velocity_v13": (
        "CosineRoPEVelocityGaussianBackend"
    ),
    "dynamic_2dgs_attention_velocity_v13_1": (
        "SeedConditionedCosineRoPEVelocityGaussianBackend"
    ),
    "dynamic_2dgs_attention_velocity_v13_2": (
        "RawQKRoPEVelocityGaussianBackend"
    ),
    "dynamic_2dgs_attention_velocity_v13_3": (
        "RMSNormRoPEVelocityGaussianBackend"
    ),
    "dynamic_2dgs_attention_velocity_v13_4": (
        "RMSNormFeatureRoPEVelocityGaussianBackend"
    ),
    "dynamic_2dgs_attention_velocity_v14": (
        "MaxSpeedBarrierVelocityGaussianBackend"
    ),
    "dynamic_2dgs_attention_velocity_v14_1": (
        "GlobalSoftplusMaxSpeedBarrierVelocityGaussianBackend"
    ),
    "dynamic_2dgs_attention_velocity_v15": (
        "NestedCountBarrierVelocityGaussianBackend"
    ),
    "dynamic_2dgs_attention_velocity_v15_1": (
        "NestedCountBarrierVelocityGaussianBackend"
    ),
    "dynamic_2dgs_attention_velocity_v15_2": (
        "OptionExpertBarrierVelocityGaussianBackend"
    ),
    "dynamic_2dgs_attention_velocity_v15_3": (
        "DenseCountBarrierVelocityGaussianBackend"
    ),
    "dynamic_2dgs_attention_velocity_v15_4": (
        "DenseCountBarrierVelocityGaussianBackend"
    ),
    "dynamic_2dgs_attention_velocity_v15_5": (
        "UniformLayerDenseCountBarrierVelocityGaussianBackend"
    ),
    "dynamic_2dgs_attention_velocity_v16": (
        "OrdinalCountBarrierVelocityGaussianBackend"
    ),
    "dynamic_2dgs_attention_velocity_v16_1": (
        "OrdinalCountBarrierVelocityGaussianBackend"
    ),
    "dynamic_2dgs_attention_velocity_v17_1": (
        "DenseCountBarrierVelocityGaussianBackend"
    ),
    "dynamic_2dgs_attention_velocity_v18": (
        "DenseCountBarrierVelocityGaussianBackend"
    ),
    "dynamic_2dgs_attention_velocity_v18_1": (
        "DenseCountBarrierVelocityGaussianBackend"
    ),
}


# (ordered state_dict schema, ordered named-parameter list), SHA-256 prefixes.
_PRE_REFACTOR_MANIFEST = {
    "dynamic_2dgs_attention_velocity_v11": (
        "a7f420bb2b15d7023d1f", "b72527798ae7652e170c"
    ),
    "dynamic_2dgs_attention_velocity_v11_1": (
        "faa47cf242bad2189a94", "7fbbbd9820a3046f0aa3"
    ),
    # V11.2 keeps V11's layer parameters in V11's order, drops the layer
    # mixture scalar it no longer has readouts to weight, and appends one
    # correspondence readout head.
    "dynamic_2dgs_attention_velocity_v11_2": (
        "4c40f14d3df7daad233b", "00d8592912ab2b795fa1"
    ),
    # V11.3 keeps V11's stack but drops the router, so its Gaussian head
    # rebuilds what the router's decoder did around it: a trunk taking the
    # two slots' 6 seed-delta columns, then attribute projections twice as
    # wide. That trunk is what separates it from V11.1's parameter names.
    "dynamic_2dgs_attention_velocity_v11_3": (
        "27b61a7e5c78bec30446", "274e243c1107ab2fa1b0"
    ),
    # V11.4 changes only the hinge's shape, and the barrier is a constant
    # of the graph rather than a parameter, so its digests are V11's. The
    # two are weight-compatible and behaviourally different; what keeps
    # them apart is the config embedded in the checkpoint.
    "dynamic_2dgs_attention_velocity_v11_4": (
        "a7f420bb2b15d7023d1f", "b72527798ae7652e170c"
    ),
    # V11.5 is V11.3's fixed K=2 decoder under V11.4's linear hinge. Neither
    # the hinge nor the offset bound is a parameter, so its digests are V11.3's.
    "dynamic_2dgs_attention_velocity_v11_5": (
        "27b61a7e5c78bec30446", "274e243c1107ab2fa1b0"
    ),
    # V11.6 is V11.5 at three slots. Parameter names and order are V11.5's, but
    # the trunk reads 9 seed-delta columns and every attribute projection is
    # three slots wide, so only the state schema digest differs.
    "dynamic_2dgs_attention_velocity_v11_6": (
        "ead603d18db213e6f8fc", "274e243c1107ab2fa1b0"
    ),
    # V12 appends one scalar v_0 parameter to V11.4's temporal module. Its
    # checkpoint stores v_0 itself, so the learned speed restores exactly.
    "dynamic_2dgs_attention_velocity_v12": (
        "daea4e3817d9fe55886a", "15c9a7df0cb498389fa0"
    ),
    # V12.1 replaces each shared RMS gain with one RoPE-group gain of the same
    # width; head 0's fixed gain is absent and one learned log-lambda is added.
    "dynamic_2dgs_attention_velocity_v12_1": (
        "018ea06086c89eddba66", "7b477ca6c13d6632021a"
    ),
    # V12.2 registers one per-layer v_raw vector where V12 registers its
    # scalar v_0, in the same position; every other parameter is V12's.
    "dynamic_2dgs_attention_velocity_v12_2": (
        "c2063628e8f8f09a5d5e", "e6eb26eaef8f1c0b77d3"
    ),
    # V13 removes every temporal QK gain and both barrier scalars. V13.1 keeps
    # that temporal schema but replaces the adaptive router with the fixed K=2
    # seed-conditioned decoder.
    "dynamic_2dgs_attention_velocity_v13": (
        "080ce3c67577b3ee8594", "92153e92839df09bd2f3"
    ),
    "dynamic_2dgs_attention_velocity_v13_1": (
        "90b35c80dbc62899965c", "aa231a6fe40641f9f90a"
    ),
    # V13.2 changes only parameter-free score arithmetic, so its checkpoint and
    # optimizer schemas are exactly V13's.
    "dynamic_2dgs_attention_velocity_v13_2": (
        "080ce3c67577b3ee8594", "92153e92839df09bd2f3"
    ),
    # V13.3 restores V11.4's per-layer Q/K RMSNorm pairs and otherwise changes
    # only parameter-free score packing, so its ordered schema matches V11.4.
    "dynamic_2dgs_attention_velocity_v13_3": (
        "a7f420bb2b15d7023d1f", "b72527798ae7652e170c"
    ),
    # V13.4 changes only which heads receive parameter-free RoPE, so its state
    # and optimizer schemas remain exactly V13.3/V11.4's.
    "dynamic_2dgs_attention_velocity_v13_4": (
        "a7f420bb2b15d7023d1f", "b72527798ae7652e170c"
    ),
    # V14 is V11.4's backend with token anchors, so its router trunk reads the
    # feature alone: V11.4's parameter names in V11.4's order, but the trunk's
    # input width loses the 9 seed-delta columns. (Its Utonia changes live in
    # Point2Gaus, outside this backend-only schema.)
    "dynamic_2dgs_attention_velocity_v14": (
        "d8e521e1d84176093fff", "b72527798ae7652e170c"
    ),
    # V14.1 registers one scalar v_raw where V12.2 registers its per-layer
    # vector, under the same name and in the same position.
    "dynamic_2dgs_attention_velocity_v14_1": (
        "81ff74b836e5d501ae22", "e6eb26eaef8f1c0b77d3"
    ),
    # V15 keeps V14.1's temporal parameters, v_raw included, without the
    # per-layer LayerScale gates; swaps GridSlotHead for the nested G1..G5 head
    # (decoder and a 256/128 router, each behind its own LayerNorm); keeps the
    # velocity head; and appends one radius local-attention block.
    "dynamic_2dgs_attention_velocity_v15": (
        "b104c12785ea21ef2180", "be541a7d2f4d71a3c4fc"
    ),
    # V15.1 drops only V15's second training render, which owns no parameter,
    # so its checkpoint and optimizer schemas are exactly V15's.
    "dynamic_2dgs_attention_velocity_v15_1": (
        "b104c12785ea21ef2180", "be541a7d2f4d71a3c4fc"
    ),
    # V15.2 keeps V15's backend and router and replaces the head's decoder with
    # a shared trunk and three per-option experts (E1, E3, E5).
    "dynamic_2dgs_attention_velocity_v15_2": (
        "76303187e805cc7f997f", "dac67a1efead903e18ca"
    ),
    # V15.3 keeps V15's backend and decoder and has no router.
    "dynamic_2dgs_attention_velocity_v15_3": (
        "f2b456dc5cd89feb5c00", "ceac3d77acf6877837cd"
    ),
    # V15.4 is V15.3 with eight temporal heads instead of twelve. Its parameter
    # names and order are V15.3's; only the per-layer Q/K RMSNorm gains, one
    # head wide, change shape.
    "dynamic_2dgs_attention_velocity_v15_4": (
        "0d8954b0eef60c36271e", "ceac3d77acf6877837cd"
    ),
    # V15.5 is V15.4 without temporal.layer_logits: V15.4's other tensors in
    # V15.4's order and shapes.
    "dynamic_2dgs_attention_velocity_v15_5": (
        "2af16e29d261637ef45e", "38c6142795775fb5865f"
    ),
    # V16 is V15 with the router's last layer swapped for two bias-free cosine
    # gates: router.4 holds one (2, 128) weight instead of (3, 128) + bias.
    "dynamic_2dgs_attention_velocity_v16": (
        "db3159b8dc10ce6df033", "8404a6a2e2418a971004"
    ),
    # V16.1 changes only V16's second training render, which owns no
    # parameter, so its checkpoint and optimizer schemas are exactly V16's.
    "dynamic_2dgs_attention_velocity_v16_1": (
        "db3159b8dc10ce6df033", "8404a6a2e2418a971004"
    ),
    # V17.1 changes only Utonia; its entire downstream schema is V15.4's.
    "dynamic_2dgs_attention_velocity_v17_1": (
        "0d8954b0eef60c36271e", "ceac3d77acf6877837cd"
    ),
    # V18 is V15.4's backend in V15.4's parameter order; only the decoder's
    # last layer narrows, five 40-channel slots (a 30-channel feature) instead
    # of five 42-channel ones (32 SH coefficients). Its lidar decoder lives in
    # the renderer (see test_v18_renderer_owns_the_lidar_decoder).
    "dynamic_2dgs_attention_velocity_v18": (
        "8cadb1f651af5e46ad32", "ceac3d77acf6877837cd"
    ),
    # V18.1 changes only G2P's lidar decoder; its backend schema is V18's.
    "dynamic_2dgs_attention_velocity_v18_1": (
        "8cadb1f651af5e46ad32", "ceac3d77acf6877837cd"
    ),
}


def _digest(lines) -> str:
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()[:20]


def _load_dynamic_module_without_renderer():
    """Import the backend package without initializing the CUDA renderer."""
    package_name = "src.models._checkpoint_compat_module"
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [
            str(pathlib.Path(__file__).parents[1] / "src/models/module")
        ]
        package.__package__ = package_name
        sys.modules[package_name] = package
    return importlib.import_module(f"{package_name}.dynamic_gaussian")


_DYNAMIC = _load_dynamic_module_without_renderer()
_SEMANTIC_MODULE_BY_EXPORT = {
    "BidirectionalCrossAttentionBase": "temporal",
    "RouterCapableGaussianBackend": "backends",
    "EmbeddedInitVelocityHead": "heads",
    "CosineRoPELayerWeightedCrossAttention": "temporal",
    "CosineRoPEVelocityGaussianBackend": "backends",
    "DynamicGaussianBackend": "backends",
    "FinalFeatureBarrierCrossAttention": "temporal",
    "FinalFeatureBarrierVelocityGaussianBackend": "backends",
    "FixedMotionGainLearnableMaxSpeedBarrierCrossAttention": "temporal",
    "FixedMotionGainMaxSpeedBarrierVelocityGaussianBackend": "backends",
    "FixedMotionHeadRMSNorm": "temporal",
    "GlobalSoftplusMaxSpeedBarrierCrossAttention": "temporal",
    "GlobalSoftplusMaxSpeedBarrierVelocityGaussianBackend": "backends",
    "LayerwiseSoftplusMaxSpeedBarrierCrossAttention": "temporal",
    "LayerwiseSoftplusMaxSpeedBarrierVelocityGaussianBackend": "backends",
    "LearnableMaxSpeedBarrierLayerWeightedCrossAttention": "temporal",
    "LearnableMaxSpeedBarrierVelocityGaussianBackend": "backends",
    "LocalAttentionBarrierVelocityGaussianBackend": "backends",
    "NestedCountBarrierVelocityGaussianBackend": "backends",
    "NestedCountGaussianHead": "nested_count",
    "RoutedCountGaussianHead": "nested_count",
    "OptionExpertBarrierVelocityGaussianBackend": "backends",
    "OptionExpertGaussianHead": "option_experts",
    "OrdinalCountBarrierVelocityGaussianBackend": "backends",
    "OrdinalCountGaussianHead": "ordinal_count",
    "CosineGateLayer": "ordinal_count",
    "DenseCountBarrierVelocityGaussianBackend": "backends",
    "DenseCountGaussianHead": "gaussian_slots",
    "UniformLayerDenseCountBarrierVelocityGaussianBackend": "backends",
    "UniformLayerMeanSoftplusBarrierCrossAttention": "temporal",
    "RadiusLocalAttentionBlock": "local_attention",
    "RawQKRoPELayerWeightedCrossAttention": "temporal",
    "RawQKRoPEVelocityGaussianBackend": "backends",
    "RMSNormRoPELayerWeightedCrossAttention": "temporal",
    "RMSNormRoPEVelocityGaussianBackend": "backends",
    "RMSNormFeatureRoPELayerWeightedCrossAttention": "temporal",
    "RMSNormFeatureRoPEVelocityGaussianBackend": "backends",
    "SeedConditionedCosineRoPEVelocityGaussianBackend": "backends",
    "GaussianAttributeHead": "heads",
    "DynamicGausTemp": "trajectory",
}


def _load_lidar_decoder_module():
    """``src/models/module/lidar_decoder.py`` without the CUDA renderer."""
    _load_dynamic_module_without_renderer()
    return importlib.import_module(
        "src.models._checkpoint_compat_module.lidar_decoder"
    )


def _build_backend(variant):
    config, _source = compose_fresh_config(
        OmegaConf.from_dotlist([f"model.variant={variant}"])
    )
    kwargs = {}
    count = getattr(config.p2g, "grid_query", None)
    count_mode = str(getattr(count, "count_mode", "legacy")).lower()
    if count is not None and count_mode == "learned_gumbel":
        kwargs["gaussian_count_cfg"] = count
        seed_mode = OmegaConf.select(
            config, "p2g.grid_query.learned_count.seed_mode"
        )
    elif count is not None and count_mode in ROUTED_COUNT_BLOCKS:
        # V15/V15.1, V15.2 and V16: one shared token anchor per token; the
        # router, not the seed contract, decides how many Gaussians are rendered.
        kwargs["gaussian_count_cfg"] = count
        kwargs["gaussians_per_token"] = 1
        seed_mode = OmegaConf.select(
            config,
            f"p2g.grid_query.{ROUTED_COUNT_BLOCKS[count_mode]}.seed_mode",
        )
    else:
        kwargs["gaussians_per_token"] = int(getattr(count, "K_max", 1))
        seed_mode = OmegaConf.select(config, "p2g.grid_query.seed_mode")
    # Point2Gaus passes the seed mode, which decides whether the Gaussian head
    # reads seed deltas (range_quantile) or the token feature alone.
    kwargs["seed_mode"] = seed_mode or "range_quantile"
    backend_cls = getattr(_DYNAMIC, _BACKEND_CLASS[variant])
    return backend_cls(
        config.dynamic_2dgs,
        config.p2g.gs_params,
        dim=144,
        offset_bound=float(config.p2g.head_offset_bound),
        **kwargs,
    )


class DynamicCheckpointCompatibilityTest(unittest.TestCase):
    def test_every_variant_keeps_checkpoint_and_optimizer_parameter_order(self):
        self.assertEqual(set(DYNAMIC_VARIANTS), set(_PRE_REFACTOR_MANIFEST))
        for variant in DYNAMIC_VARIANTS:
            with self.subTest(variant=variant):
                torch.manual_seed(777)
                source = _build_backend(variant)
                state_schema = [
                    f"{name}|{tuple(value.shape)}|{value.dtype}"
                    for name, value in source.state_dict().items()
                ]
                parameter_names = [name for name, _ in source.named_parameters()]
                expected_schema, expected_order = _PRE_REFACTOR_MANIFEST[variant]
                self.assertEqual(_digest(state_schema), expected_schema)
                self.assertEqual(_digest(parameter_names), expected_order)

                # This is the exact strict restore operation used by a checkpoint.
                torch.manual_seed(991)
                restored = _build_backend(variant)
                incompatible = restored.load_state_dict(
                    source.state_dict(), strict=True
                )
                self.assertEqual(incompatible.missing_keys, [])
                self.assertEqual(incompatible.unexpected_keys, [])

                # Optimizer state_dict uses positional parameter ids. Give every
                # slot a unique scalar and ensure load maps it back to the same
                # ordered named parameter.
                old_optimizer = torch.optim.SGD(source.parameters(), lr=0.1)
                for index, parameter in enumerate(source.parameters()):
                    old_optimizer.state[parameter]["compatibility_index"] = (
                        torch.tensor(index)
                    )
                new_optimizer = torch.optim.SGD(restored.parameters(), lr=0.1)
                new_optimizer.load_state_dict(old_optimizer.state_dict())
                for index, (_name, parameter) in enumerate(
                    restored.named_parameters()
                ):
                    self.assertEqual(
                        new_optimizer.state[parameter]["compatibility_index"].item(),
                        index,
                    )

    def test_v18_is_v15_4_with_a_narrower_slot_projection(self):
        v15_4 = _build_backend("dynamic_2dgs_attention_velocity_v15_4")
        backend = _build_backend("dynamic_2dgs_attention_velocity_v18")
        self.assertIs(type(backend), type(v15_4))
        old, new = v15_4.state_dict(), backend.state_dict()
        self.assertEqual(list(new), list(old))
        changed = [
            name for name in new if tuple(new[name].shape) != tuple(old[name].shape)
        ]
        self.assertEqual(
            changed, ["gaussian_head.decoder.2.weight", "gaussian_head.decoder.2.bias"]
        )
        self.assertEqual(tuple(new["gaussian_head.decoder.2.weight"].shape), (200, 576))
        self.assertEqual(
            backend.gaussian_head.param_names,
            ("feature", "opacity", "scaling", "rotation", "offset"),
        )

    def test_v18_renderer_owns_the_lidar_decoder(self):
        """G2P's checkpoint keys: the sensor embedding, then the 64-96-96-2 MLP."""
        config, _source = compose_fresh_config(OmegaConf.from_dotlist(
            ["model.variant=dynamic_2dgs_attention_velocity_v18"]
        ))
        decoder = _load_lidar_decoder_module().LidarFeatureDecoder(
            config.g2p.lidar_decoder
        )
        self.assertEqual(
            [(name, tuple(value.shape)) for name, value in decoder.state_dict().items()],
            [
                ("sensor_embedding.weight", (1, 30)),
                ("mlp.0.weight", (96, 64)), ("mlp.0.bias", (96,)),
                ("mlp.2.weight", (96, 96)), ("mlp.2.bias", (96,)),
                ("mlp.4.weight", (2, 96)), ("mlp.4.bias", (2,)),
            ],
        )
        for variant in DYNAMIC_VARIANTS:
            if variant in (
                "dynamic_2dgs_attention_velocity_v18",
                "dynamic_2dgs_attention_velocity_v18_1",
            ):
                continue
            with self.subTest(variant=variant):
                config, _source = compose_fresh_config(
                    OmegaConf.from_dotlist([f"model.variant={variant}"])
                )
                self.assertIsNone(
                    OmegaConf.select(config, "g2p.lidar_decoder", default=None)
                )

    def test_v18_1_renderer_owns_v18s_lidar_decoder_without_the_embedding(self):
        """G2P's checkpoint keys: the 34-96-96-2 MLP, no sensor embedding."""
        config, _source = compose_fresh_config(OmegaConf.from_dotlist(
            ["model.variant=dynamic_2dgs_attention_velocity_v18_1"]
        ))
        decoder = _load_lidar_decoder_module().LidarFeatureDecoder(
            config.g2p.lidar_decoder
        )
        self.assertIsNone(decoder.sensor_embedding)
        self.assertEqual(
            [(name, tuple(value.shape)) for name, value in decoder.state_dict().items()],
            [
                ("mlp.0.weight", (96, 34)), ("mlp.0.bias", (96,)),
                ("mlp.2.weight", (96, 96)), ("mlp.2.bias", (96,)),
                ("mlp.4.weight", (2, 96)), ("mlp.4.bias", (2,)),
            ],
        )
        self.assertEqual(sum(p.numel() for p in decoder.parameters()), 12866)

    def test_v11_5_builds_the_v11_3_decoder_under_the_v11_4_hinge(self):
        backend = _build_backend("dynamic_2dgs_attention_velocity_v11_5")
        self.assertIs(
            type(backend), _DYNAMIC.SeedConditionedBarrierVelocityGaussianBackend
        )
        self.assertFalse(backend.adaptive_count)
        self.assertEqual(backend.gaussians_per_token, 2)
        self.assertTrue(backend.seed_conditioned_head)
        self.assertEqual(backend.gaussian_head.seed_dim, 6)
        self.assertAlmostEqual(backend.gaussian_head.offset_bound, 0.4)
        self.assertEqual(backend.temporal.barrier_exponent, 1)
        self.assertEqual(backend.temporal.barrier_weight, 1.0)
        self.assertEqual(backend.temporal.barrier_speed_mps, 30.0)

    def test_v11_6_builds_the_v11_5_decoder_with_three_slots(self):
        backend = _build_backend("dynamic_2dgs_attention_velocity_v11_6")
        self.assertIs(
            type(backend), _DYNAMIC.SeedConditionedBarrierVelocityGaussianBackend
        )
        self.assertFalse(backend.adaptive_count)
        self.assertEqual(backend.gaussians_per_token, 3)
        self.assertTrue(backend.seed_conditioned_head)
        self.assertEqual(backend.gaussian_head.seed_dim, 9)
        self.assertAlmostEqual(backend.gaussian_head.offset_bound, 0.4)
        self.assertEqual(backend.temporal.barrier_exponent, 1)
        self.assertEqual(backend.temporal.barrier_weight, 1.0)
        self.assertEqual(backend.temporal.barrier_speed_mps, 30.0)

    def test_v12_2_swaps_v12_v0_for_one_softplus_speed_per_layer(self):
        v12 = _build_backend("dynamic_2dgs_attention_velocity_v12")
        backend = _build_backend("dynamic_2dgs_attention_velocity_v12_2")
        self.assertIs(
            type(backend),
            _DYNAMIC.LayerwiseSoftplusMaxSpeedBarrierVelocityGaussianBackend,
        )
        self.assertTrue(backend.adaptive_count)
        self.assertEqual(
            [name for name, _ in backend.named_parameters()],
            [
                "temporal.v_raw" if name == "temporal.v_0" else name
                for name, _ in v12.named_parameters()
            ],
        )
        self.assertEqual(tuple(backend.temporal.v_raw.shape), (12,))
        torch.testing.assert_close(
            backend.temporal.v_max.detach(), torch.full((12,), 30.0)
        )
        self.assertAlmostEqual(backend.offset_bound, 0.4)
        self.assertEqual(backend.temporal.barrier_exponent, 1)
        self.assertEqual(backend.temporal.barrier_weight, 1.0)

    def test_v14_1_swaps_v12_2_speeds_for_one_global_softplus_speed(self):
        v14 = _build_backend("dynamic_2dgs_attention_velocity_v14")
        v12_2 = _build_backend("dynamic_2dgs_attention_velocity_v12_2")
        backend = _build_backend("dynamic_2dgs_attention_velocity_v14_1")
        self.assertIs(
            type(backend),
            _DYNAMIC.GlobalSoftplusMaxSpeedBarrierVelocityGaussianBackend,
        )
        self.assertIs(type(v14), _DYNAMIC.MaxSpeedBarrierVelocityGaussianBackend)
        self.assertTrue(backend.adaptive_count)
        self.assertEqual(
            [name for name, _ in backend.named_parameters()],
            [name for name, _ in v12_2.named_parameters()],
        )
        self.assertEqual(tuple(backend.temporal.v_raw.shape), ())
        torch.testing.assert_close(
            backend.temporal.v_max.detach(), torch.tensor(30.0)
        )
        # Token anchors: the router trunk reads the refined feature alone.
        for model in (v14, backend):
            self.assertEqual(model.seed_mode, "token")
            self.assertFalse(model.gaussian_head.seed_conditioned)
            self.assertEqual(model.gaussian_head.trunk[0].in_features, 144)
        self.assertEqual(v12_2.gaussian_head.trunk[0].in_features, 153)
        self.assertAlmostEqual(backend.offset_bound, 0.4)
        self.assertEqual(backend.temporal.barrier_exponent, 1)
        self.assertEqual(backend.temporal.barrier_weight, 1.0)

    def test_v15_drops_layer_scale_and_appends_local_attention(self):
        v14_1 = _build_backend("dynamic_2dgs_attention_velocity_v14_1")
        backend = _build_backend("dynamic_2dgs_attention_velocity_v15")
        self.assertIs(
            type(backend),
            _DYNAMIC.NestedCountBarrierVelocityGaussianBackend,
        )
        names = [name for name, _ in backend.named_parameters()]
        v14_1_names = [name for name, _ in v14_1.named_parameters()]
        self.assertEqual(
            [name for name in names if name.startswith("temporal.")],
            [
                name for name in v14_1_names
                if name.startswith("temporal.")
                and "cross_layer_scale" not in name
            ],
        )
        self.assertEqual(
            [name for name in names if name.startswith("velocity_head.")],
            [name for name in v14_1_names if name.startswith("velocity_head.")],
        )
        top_level = []
        for name in names:
            root = name.split(".")[0]
            if root not in top_level:
                top_level.append(root)
        self.assertEqual(
            top_level,
            ["temporal", "gaussian_head", "velocity_head", "local_attention"],
        )
        self.assertIsNone(backend.temporal.cross_layer_scale)
        torch.testing.assert_close(
            backend.temporal.v_max.detach(), torch.tensor(30.0)
        )
        self.assertEqual(backend.seed_mode, "token")
        head = backend.gaussian_head
        self.assertEqual(head.option_counts, (1, 3, 5))
        self.assertEqual(head.auxiliary_render, "dense_keep")
        self.assertIsNot(head.router_norm, head.decoder_norm)
        self.assertEqual(
            [layer.out_features for layer in head.decoder
             if isinstance(layer, torch.nn.Linear)],
            [576, 5 * 42],
        )
        self.assertEqual(
            [layer.out_features for layer in head.router
             if isinstance(layer, torch.nn.Linear)],
            [256, 128, 3],
        )
        self.assertEqual(backend.local_attention.radius_m, 0.8)

    def test_v15_1_is_v15_without_the_auxiliary_render(self):
        v15 = _build_backend("dynamic_2dgs_attention_velocity_v15")
        backend = _build_backend("dynamic_2dgs_attention_velocity_v15_1")
        self.assertIs(type(backend), type(v15))
        self.assertEqual(
            [(name, tuple(value.shape)) for name, value in backend.state_dict().items()],
            [(name, tuple(value.shape)) for name, value in v15.state_dict().items()],
        )
        self.assertEqual(v15.gaussian_head.auxiliary_render, "dense_keep")
        self.assertEqual(backend.gaussian_head.auxiliary_render, "none")

    def test_v15_2_swaps_the_nested_decoder_for_option_experts(self):
        v15 = _build_backend("dynamic_2dgs_attention_velocity_v15")
        backend = _build_backend("dynamic_2dgs_attention_velocity_v15_2")
        self.assertIs(
            type(backend), _DYNAMIC.OptionExpertBarrierVelocityGaussianBackend
        )
        names = [name for name, _ in backend.named_parameters()]
        v15_names = [name for name, _ in v15.named_parameters()]
        # Everything outside the head's decoder is V15's, in V15's order.
        self.assertEqual(
            [name for name in names if not name.startswith(
                ("gaussian_head.trunk.", "gaussian_head.experts.")
            )],
            [name for name in v15_names if not name.startswith(
                "gaussian_head.decoder."
            )],
        )
        head = backend.gaussian_head
        self.assertIsInstance(head, _DYNAMIC.OptionExpertGaussianHead)
        self.assertEqual(head.option_counts, (1, 3, 5))
        self.assertEqual(head.auxiliary_render, "dense_add")
        self.assertEqual(head.auxiliary_count, 5)
        self.assertEqual(head.trunk[0].out_features, 576)
        self.assertEqual(
            [expert.out_features for expert in head.experts], [42, 126, 210]
        )
        self.assertEqual(
            {expert.in_features for expert in head.experts}, {576}
        )
        self.assertEqual(
            [layer.out_features for layer in head.router
             if isinstance(layer, torch.nn.Linear)],
            [256, 128, 3],
        )
        self.assertIsNot(head.router_norm, head.decoder_norm)
        self.assertEqual(backend.local_attention.radius_m, 0.8)
        self.assertIsNone(backend.temporal.cross_layer_scale)

    def test_v15_3_is_v15s_decoder_without_the_router(self):
        v15 = _build_backend("dynamic_2dgs_attention_velocity_v15")
        backend = _build_backend("dynamic_2dgs_attention_velocity_v15_3")
        self.assertIs(
            type(backend), _DYNAMIC.DenseCountBarrierVelocityGaussianBackend
        )
        self.assertFalse(backend.adaptive_count)
        self.assertEqual(backend.gaussians_per_token, 5)
        self.assertEqual(backend.seed_mode, "token")
        self.assertFalse(backend.seed_conditioned_head)
        # V15's module tree minus the router: same names, shapes and order.
        v15_schema = [
            (name, tuple(value.shape))
            for name, value in v15.state_dict().items()
            if not name.startswith(("gaussian_head.router", "gaussian_head.router_norm"))
        ]
        self.assertEqual(
            [(name, tuple(value.shape)) for name, value in backend.state_dict().items()],
            v15_schema,
        )
        head = backend.gaussian_head
        self.assertIsInstance(head, _DYNAMIC.DenseCountGaussianHead)
        self.assertEqual(
            [layer.out_features for layer in head.decoder
             if isinstance(layer, torch.nn.Linear)],
            [576, 5 * 42],
        )

    def test_v15_4_is_v15_3_with_eight_temporal_heads(self):
        v15_3 = _build_backend("dynamic_2dgs_attention_velocity_v15_3")
        backend = _build_backend("dynamic_2dgs_attention_velocity_v15_4")
        self.assertIs(
            type(backend), _DYNAMIC.DenseCountBarrierVelocityGaussianBackend
        )
        self.assertEqual(backend.gaussians_per_token, 5)
        temporal = backend.temporal
        self.assertEqual(
            (v15_3.temporal.num_heads, temporal.num_heads), (12, 8)
        )
        self.assertEqual(temporal.head_dim, backend.dim // 8)
        self.assertEqual(temporal.rope_head_count, 7)
        self.assertEqual(temporal.rope.head_dim, temporal.head_dim)
        self.assertAlmostEqual(temporal.rope.base, 10.0 ** 1.875)
        # The local-attention block keeps V15.3's twelve heads.
        self.assertEqual(
            backend.local_attention.num_heads,
            v15_3.local_attention.num_heads,
        )
        # V15.3's tensors in V15.3's order; only the Q/K RMSNorm gains, one
        # head wide, change shape.
        old = v15_3.state_dict()
        new = backend.state_dict()
        self.assertEqual(list(new), list(old))
        reshaped = {
            name for name in new if new[name].shape != old[name].shape
        }
        self.assertEqual(reshaped, {
            f"temporal.{norm}.{layer}.weight"
            for norm in ("q_norm", "k_norm")
            for layer in range(temporal.n_layers)
        })
        for name in reshaped:
            self.assertEqual(tuple(new[name].shape), (temporal.head_dim,))

    def test_v15_5_is_v15_4_without_the_layer_logits(self):
        v15_4 = _build_backend("dynamic_2dgs_attention_velocity_v15_4")
        backend = _build_backend("dynamic_2dgs_attention_velocity_v15_5")
        self.assertIs(
            type(backend),
            _DYNAMIC.UniformLayerDenseCountBarrierVelocityGaussianBackend,
        )
        self.assertIsInstance(
            backend, _DYNAMIC.DenseCountBarrierVelocityGaussianBackend
        )
        self.assertIs(
            type(backend.temporal),
            _DYNAMIC.UniformLayerMeanSoftplusBarrierCrossAttention,
        )
        self.assertEqual(backend.gaussians_per_token, 5)
        self.assertEqual(backend.temporal.num_heads, 8)
        self.assertAlmostEqual(backend.temporal.rope.base, 10.0 ** 1.875)
        self.assertFalse(hasattr(backend.temporal, "layer_logits"))
        old = v15_4.state_dict()
        new = backend.state_dict()
        self.assertEqual(
            list(new), [name for name in old if name != "temporal.layer_logits"]
        )
        for name in new:
            self.assertEqual(new[name].shape, old[name].shape)
        # A V15.4 checkpoint does not load into V15.5 strictly: the logits
        # are the one tensor it has no place for.
        incompatible = backend.load_state_dict(old, strict=False)
        self.assertEqual(incompatible.missing_keys, [])
        self.assertEqual(
            incompatible.unexpected_keys, ["temporal.layer_logits"]
        )

    def test_v16_is_v15_with_two_bounded_ordinal_gates(self):
        v15 = _build_backend("dynamic_2dgs_attention_velocity_v15")
        backend = _build_backend("dynamic_2dgs_attention_velocity_v16")
        self.assertIs(
            type(backend), _DYNAMIC.OrdinalCountBarrierVelocityGaussianBackend
        )
        last = "gaussian_head.router.4."
        schema = [
            (name, tuple(value.shape))
            for name, value in backend.state_dict().items()
        ]
        v15_schema = [
            (name, tuple(value.shape))
            for name, value in v15.state_dict().items()
        ]
        # Everything but the router's last layer is V15's, in V15's order.
        self.assertEqual(
            [item for item in schema if not item[0].startswith(last)],
            [item for item in v15_schema if not item[0].startswith(last)],
        )
        self.assertEqual(
            [item for item in schema if item[0].startswith(last)],
            [(last + "weight", (2, 128))],
        )
        head = backend.gaussian_head
        self.assertIsInstance(head, _DYNAMIC.OrdinalCountGaussianHead)
        self.assertIsInstance(head.router[-1], _DYNAMIC.CosineGateLayer)
        self.assertEqual(head.router[-1].scale, 3.0)
        self.assertEqual(head.router_scale, 3.0)
        self.assertEqual(head.option_counts, (1, 3, 5))
        self.assertEqual(head.auxiliary_render, "dense_add")
        self.assertEqual(head.auxiliary_count, 5)
        self.assertIsNot(head.router_norm, head.decoder_norm)
        self.assertEqual(
            [layer.out_features for layer in head.decoder
             if isinstance(layer, torch.nn.Linear)],
            [576, 5 * 42],
        )
        self.assertEqual(backend.local_attention.radius_m, 0.8)
        self.assertIsNone(backend.temporal.cross_layer_scale)

    def test_v16_1_is_v16_with_a_random_prefix_auxiliary_render(self):
        v16 = _build_backend("dynamic_2dgs_attention_velocity_v16")
        backend = _build_backend("dynamic_2dgs_attention_velocity_v16_1")
        self.assertIs(type(backend), type(v16))
        self.assertEqual(
            [(name, tuple(value.shape)) for name, value in backend.state_dict().items()],
            [(name, tuple(value.shape)) for name, value in v16.state_dict().items()],
        )
        self.assertEqual(v16.gaussian_head.auxiliary_render, "dense_add")
        self.assertEqual(
            backend.gaussian_head.auxiliary_render, "random_prefix"
        )

    def test_single_seed_slot_preserves_the_legacy_geometry_transform(self):
        """The new explicit slot axis must preserve legacy (N, 3) inputs."""
        variant = "dynamic_2dgs_attention_velocity_v11_1"
        torch.manual_seed(31)
        backend = _build_backend(variant).eval()
        feature = torch.randn(4, 144)
        token = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.2, 0.0],
            [0.1, 0.0, 0.0],
            [1.1, 0.2, 0.0],
        ])
        legacy_seed = token + torch.tensor([0.02, 0.01, 0.0])
        pose = torch.eye(4).repeat(2, 1, 1)
        pose[0, :3, 3] = torch.tensor([4.0, -2.0, 0.5])
        pose[1, :3, 3] = torch.tensor([-1.0, 3.0, -0.25])
        arguments = (
            feature,
            token,
            torch.tensor([2, 4]),
            torch.tensor([0, 0]),
            [pose],
            [torch.tensor([0.0, 1.0])],
            [torch.tensor([0.0, 0.8])],
            torch.tensor([0.8]),
        )
        legacy = backend(arguments[0], arguments[1], legacy_seed, *arguments[2:])
        slotted = backend(
            arguments[0], arguments[1], legacy_seed.unsqueeze(1), *arguments[2:]
        )
        for name in ("position", "velocity", "velocity_init"):
            torch.testing.assert_close(
                legacy["batch_gaussians"][0][name],
                slotted["batch_gaussians"][0][name],
            )

    def test_historical_facade_reexports_the_moved_class_objects(self):
        for name, module_name in _SEMANTIC_MODULE_BY_EXPORT.items():
            with self.subTest(name=name):
                semantic_module = importlib.import_module(
                    "src.models._checkpoint_compat_module.dynamic."
                    + module_name
                )
                self.assertIs(
                    getattr(_DYNAMIC, name),
                    getattr(semantic_module, name),
                )


if __name__ == "__main__":
    unittest.main()
