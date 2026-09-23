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

from src.config_loader import (
    DYNAMIC_VARIANTS,
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
}


def _digest(lines) -> str:
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()[:20]


def _load_dynamic_module_without_renderer():
    """Import the backend package without initializing the CUDA renderer."""
    package_name = "src.models_new._checkpoint_compat_module"
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [
            str(pathlib.Path(__file__).parents[1] / "src/models_new/module")
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
    "LearnableMaxSpeedBarrierLayerWeightedCrossAttention": "temporal",
    "LearnableMaxSpeedBarrierVelocityGaussianBackend": "backends",
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


def _build_backend(variant):
    config, _source = compose_fresh_config(
        OmegaConf.from_dotlist([f"model.variant={variant}"])
    )
    kwargs = {}
    count = getattr(config.p2g, "grid_query", None)
    count_mode = str(getattr(count, "count_mode", "legacy")).lower()
    if count is not None and count_mode == "learned_gumbel":
        kwargs["gaussian_count_cfg"] = count
    else:
        kwargs["gaussians_per_token"] = int(getattr(count, "K_max", 1))
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
                    "src.models_new._checkpoint_compat_module.dynamic."
                    + module_name
                )
                self.assertIs(
                    getattr(_DYNAMIC, name),
                    getattr(semantic_module, name),
                )


if __name__ == "__main__":
    unittest.main()
