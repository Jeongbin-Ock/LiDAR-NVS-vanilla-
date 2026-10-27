"""Sensor-head adaptation: V18's LiDAR decoder retrained behind a frozen model (adapt_config)."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from omegaconf import OmegaConf

from src.experiment_config import (
    DYNAMIC_VARIANT_V15_4,
    DYNAMIC_VARIANT_V18,
    checkpoint_sha256,
    compose_adapt_config,
    compose_fresh_config,
    compose_refine_config,
    raydrop_refine_enabled,
    resolve_data_profile,
    resolve_main_config,
    sensor_adapt_base_record,
    sensor_adapt_enabled,
)

MODEL_ROOTS = ("model", "p2g", "g2g", "g2p", "dynamic_2dgs")


def _base_checkpoint(tmp, variant=DYNAMIC_VARIANT_V18, exp_name="v18_wint0.5"):
    stored, _ = compose_fresh_config(OmegaConf.from_dotlist([
        f"model.variant={variant}", f"exp_name={exp_name}", "loss.w_intensity=0.5",
        "logger.wandb_run_id=a42uhqor",
    ]))
    checkpoint = Path(tmp) / "epoch=28.ckpt"
    torch.save({
        "experiment_config": OmegaConf.to_container(stored, resolve=True),
        "epoch": 28, "global_step": 46_000, "state_dict": {},
    }, checkpoint)
    return stored, checkpoint


def _compose(checkpoint, *extra):
    return compose_adapt_config(OmegaConf.from_dotlist([
        "adapt_config=sensor_head", f"sensor_adapt.base_ckpt={checkpoint}",
        "exp_name=v18_sensorhead", *extra,
    ]))


def _container(node):
    return OmegaConf.to_container(node, resolve=True)


class AdaptConfigTest(unittest.TestCase):
    def test_model_runs_have_no_adaptation(self):
        stored, _ = compose_fresh_config(OmegaConf.from_dotlist([
            f"model.variant={DYNAMIC_VARIANT_V18}",
        ]))
        self.assertFalse(sensor_adapt_enabled(stored))
        self.assertIsNone(sensor_adapt_base_record(stored))

    def test_adapt_run_is_the_base_model_on_kitti_with_the_decoder_objective(self):
        with tempfile.TemporaryDirectory() as tmp:
            stored, checkpoint = _base_checkpoint(tmp)
            config, source = _compose(checkpoint, "data_config=kitti360_lidar4d")
            self.assertIn("adapt overlay", source)
            self.assertIn("data profile", source)
            adapt = config.sensor_adapt
            self.assertTrue(adapt.enable)
            self.assertEqual(adapt.base_ckpt, str(checkpoint.resolve()))
            self.assertEqual(adapt.base_ckpt_sha256, checkpoint_sha256(checkpoint))
            self.assertEqual((adapt.base_epoch, adapt.base_global_step), (28, 46_000))
            self.assertEqual(adapt.base_exp_name, "v18_wint0.5")
            self.assertEqual(adapt.data_profile, str(resolve_data_profile("kitti360_lidar4d")))
            self.assertEqual(sensor_adapt_base_record(config), {
                "ckpt": str(checkpoint.resolve()), "sha256": checkpoint_sha256(checkpoint),
                "epoch": 28, "global_step": 46_000, "exp_name": "v18_wint0.5",
            })
            for root in MODEL_ROOTS:
                self.assertEqual(_container(config[root]), _container(stored[root]), root)
            kitti, _ = compose_fresh_config(OmegaConf.from_dotlist([
                f"model.variant={DYNAMIC_VARIANT_V18}", "data_config=kitti360_lidar4d",
            ]))
            self.assertEqual(_container(config.data), _container(kitti.data))
            # Intensity L1 + raydrop BCE only; the rest of the loss block is the base run's.
            self.assertEqual(
                {key: float(config.loss[key]) for key in
                 ("w_depth", "w_depth_median", "w_chamfer", "w_intensity", "w_raydrop")},
                {"w_depth": 0.0, "w_depth_median": 0.0, "w_chamfer": 0.0,
                 "w_intensity": 1.0, "w_raydrop": 1.0},
            )
            self.assertEqual(config.loss.scale_max_m, stored.loss.scale_max_m)
            self.assertEqual(config.exp_name, "v18_sensorhead_kitti_lidar4d")
            self.assertEqual(config.logger.dir, "logs/v18_sensorhead_kitti_lidar4d")
            self.assertIsNone(config.logger.wandb_run_id)
            self.assertIsNone(config.train.ckpt_path)
            self.assertEqual(
                (config.train.lr, config.train.weight_decay, config.train.warmup_iters,
                 config.train.max_epochs, config.train.batch_size,
                 config.train.grad_accum_steps, config.train.grad_clip),
                (5.0e-4, 1.0e-4, 500, 20, 2, 4, stored.train.grad_clip),
            )
            self.assertEqual(config.mode, "train")
            self.assertFalse(raydrop_refine_enabled(config))

    def test_main_entry_dispatches_adapt_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            _stored, checkpoint = _base_checkpoint(tmp)
            config, _ = resolve_main_config(OmegaConf.from_dotlist([
                "adapt_config=sensor_head", f"sensor_adapt.base_ckpt={checkpoint}",
                "exp_name=adapt_a", "device=[4,5]",
            ]))
            self.assertTrue(sensor_adapt_enabled(config))
            self.assertEqual(list(config.device), [4, 5])

    def test_only_a_model_with_a_lidar_decoder_can_be_adapted(self):
        with tempfile.TemporaryDirectory() as tmp:
            _stored, checkpoint = _base_checkpoint(
                tmp, variant=DYNAMIC_VARIANT_V15_4, exp_name="v15.4",
            )
            with self.assertRaisesRegex(ValueError, "no LiDAR decoder to adapt"):
                _compose(checkpoint)

    def test_sensor_head_starts_from_the_base_unless_asked_for_a_fresh_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            _stored, checkpoint = _base_checkpoint(tmp)
            config, _ = _compose(checkpoint, "data_config=kitti360_lidar4d")
            self.assertEqual(config.sensor_adapt.decoder_init, "base")
            config, _ = _compose(
                checkpoint, "data_config=kitti360_lidar4d", "sensor_adapt.decoder_init=random",
            )
            self.assertEqual(config.sensor_adapt.decoder_init, "random")
            with self.assertRaisesRegex(ValueError, "decoder_init must be one of"):
                _compose(checkpoint, "sensor_adapt.decoder_init=zeros")
            # Runs composed before the key existed loaded the base's head.
            del config.sensor_adapt["decoder_init"]
            from src.experiment_config import validate_experiment_config
            validate_experiment_config(config)

    def test_the_objective_is_intensity_and_raydrop_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            _stored, checkpoint = _base_checkpoint(tmp)
            for extra in ("loss.w_depth=0.5", "loss.w_depth_median=0.5", "loss.w_chamfer=0.02"):
                with self.assertRaisesRegex(ValueError, "gives the decoder no gradient"):
                    _compose(checkpoint, extra)
            with self.assertRaisesRegex(ValueError, "w_intensity or loss.w_raydrop > 0"):
                _compose(checkpoint, "loss.w_intensity=0.0", "loss.w_raydrop=0.0")
            config, _ = _compose(checkpoint, "loss.w_intensity=0.5")
            self.assertEqual(float(config.loss.w_intensity), 0.5)

    def test_adapt_overlay_cannot_touch_the_frozen_model_or_its_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            _stored, checkpoint = _base_checkpoint(tmp)
            for extra, message in (
                ("g2p.lidar_decoder.hidden_dims=[64,64]", "cannot set g2p"),
                ("data.gt_ego_radius=1.0", "cannot set data"),
                ("raydrop_refine.enable=true", "cannot set raydrop_refine"),
                ("train.ckpt_path=/x/epoch=1.ckpt", "resume one with"),
            ):
                with self.assertRaisesRegex(ValueError, message):
                    _compose(checkpoint, extra)

    def test_adapt_run_needs_its_own_name_and_a_base(self):
        with tempfile.TemporaryDirectory() as tmp:
            _stored, checkpoint = _base_checkpoint(tmp)
            with self.assertRaisesRegex(ValueError, "own exp_name"):
                _compose(checkpoint, "exp_name=v18_wint0.5")
            with self.assertRaisesRegex(ValueError, "requires sensor_adapt.base_ckpt"):
                compose_adapt_config(OmegaConf.from_dotlist([
                    "adapt_config=sensor_head", "exp_name=x",
                ]))

    def test_adaptation_checkpoint_resumes_locked_and_is_no_stage_base(self):
        with tempfile.TemporaryDirectory() as tmp:
            _stored, base = _base_checkpoint(tmp)
            config, _ = _compose(base, "data_config=kitti360_lidar4d")
            adapted = Path(tmp) / "adapt_epoch=3.ckpt"
            torch.save({
                "experiment_config": OmegaConf.to_container(config, resolve=True),
                "epoch": 3, "global_step": 3_200, "state_dict": {},
            }, adapted)
            restored, _ = resolve_main_config(OmegaConf.from_dotlist([
                f"train.ckpt_path={adapted}", "device=[4,5]",
            ]))
            self.assertTrue(sensor_adapt_enabled(restored))
            self.assertEqual(_container(restored.data), _container(config.data))
            self.assertEqual(float(restored.loss.w_depth), 0.0)
            with self.assertRaisesRegex(ValueError, "changed roots: sensor_adapt"):
                resolve_main_config(OmegaConf.from_dotlist([
                    f"train.ckpt_path={adapted}", "sensor_adapt.base_epoch=5",
                ]))
            with self.assertRaisesRegex(ValueError, "already a sensor-adaptation checkpoint"):
                compose_adapt_config(OmegaConf.from_dotlist([
                    "adapt_config=sensor_head", f"sensor_adapt.base_ckpt={adapted}",
                    "exp_name=again",
                ]))
            with self.assertRaisesRegex(ValueError, "already a sensor-adaptation checkpoint"):
                compose_refine_config(OmegaConf.from_dotlist([
                    "refine_config=raydrop_unet", f"raydrop_refine.base_ckpt={adapted}",
                    "exp_name=again",
                ]))

    def test_model_checkpoint_cannot_switch_adaptation_on_from_the_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            _stored, base = _base_checkpoint(tmp)
            with self.assertRaisesRegex(ValueError, "changed roots: sensor_adapt"):
                resolve_main_config(OmegaConf.from_dotlist([
                    f"train.ckpt_path={base}", "sensor_adapt.enable=true",
                ]))


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="building the model imports the CUDA rasterizer"
)


@requires_cuda
def test_wrapper_trains_only_the_sensor_head_and_keeps_the_model_in_eval_mode():
    from src.model_wrapper import build_model_wrapper
    from src.sensor_adapt_wrapper import SensorAdaptWrapper

    with tempfile.TemporaryDirectory() as tmp:
        _stored, checkpoint = _base_checkpoint(tmp)
        config, _ = _compose(checkpoint, "data_config=kitti360_lidar4d")
        wrapper = build_model_wrapper(config)
    assert isinstance(wrapper, SensorAdaptWrapper)
    decoder = wrapper.g2p_model.lidar_decoder
    trainable = {name for name, p in wrapper.named_parameters() if p.requires_grad}
    assert trainable == {f"g2p_model.lidar_decoder.{name}" for name, _ in decoder.named_parameters()}
    assert sum(p.numel() for p in decoder.parameters()) == 15_776
    wrapper.train()
    assert wrapper.training and wrapper.loss.training
    assert not any(module.training for module in (
        *wrapper.p2g_model.modules(), *wrapper.g2g_model.modules(), *wrapper.g2p_model.modules(),
    ))
    with mock.patch.object(
        SensorAdaptWrapper, "trainer", new=SimpleNamespace(estimated_stepping_batches=16_000),
    ):
        optimizer = wrapper.configure_optimizers()["optimizer"]
    (group,) = optimizer.param_groups
    assert group["name"] == "lidar_decoder"
    assert [id(p) for p in group["params"]] == [id(p) for p in decoder.parameters()]
    assert (group["initial_lr"], group["weight_decay"]) == (5.0e-4, 1.0e-4)
    assert group["lr"] == pytest.approx(5.0e-4 / 500)  # first warmup step


@requires_cuda
@pytest.mark.parametrize("decoder_init", ["base", "random"])
def test_base_checkpoint_loads_the_sensor_head_only_when_asked(decoder_init):
    """``random`` leaves exactly the sensor head at its seeded fresh init."""
    from src.model_wrapper import build_model_wrapper
    from src.sensor_adapt_wrapper import DECODER_PREFIX

    with tempfile.TemporaryDirectory() as tmp:
        _stored, checkpoint = _base_checkpoint(tmp)
        config, _ = _compose(
            checkpoint, "data_config=kitti360_lidar4d",
            f"sensor_adapt.decoder_init={decoder_init}",
        )
        torch.manual_seed(42)
        wrapper = build_model_wrapper(config)
        fresh = {key: value.clone() for key, value in wrapper.state_dict().items()}
        # A base whose every floating-point weight differs from the fresh init.
        base = {
            key: value + 1.0 if value.is_floating_point() else value.clone()
            for key, value in fresh.items()
        }
        blob = {"state_dict": base, "epoch": 28, "global_step": 46_000}
        with mock.patch("src.sensor_adapt_wrapper.torch.load", return_value=blob):
            wrapper.load_base_checkpoint()
        loaded = wrapper.state_dict()
    decoder_keys = {key for key in loaded if key.startswith(DECODER_PREFIX)}
    assert decoder_keys == {
        f"{DECODER_PREFIX}{name}" for name in ("sensor_embedding.weight", "mlp.0.weight",
                                                "mlp.0.bias", "mlp.2.weight", "mlp.2.bias",
                                                "mlp.4.weight", "mlp.4.bias")
    }
    for key, value in loaded.items():
        expected = fresh[key] if decoder_init == "random" and key in decoder_keys else base[key]
        assert torch.equal(value, expected), key


if __name__ == "__main__":
    unittest.main()
