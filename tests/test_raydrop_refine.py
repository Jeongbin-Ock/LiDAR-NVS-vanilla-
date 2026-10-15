"""Raydrop refinement: the LiDAR4D U-Net, its input recipe and the refine config stage."""
from __future__ import annotations

import importlib
import pathlib
import sys
import tempfile
import types
import unittest
from pathlib import Path

import torch
from omegaconf import OmegaConf

from src.experiment_config import (
    DYNAMIC_VARIANT_V15_3,
    checkpoint_sha256,
    compose_fresh_config,
    compose_refine_config,
    raydrop_refine_enabled,
    resolve_main_config,
)


def _load_refine_module():
    """Import m4_raydrop_refine without initializing the CUDA renderer."""
    package_name = "src.models._raydrop_refine_test_module"
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(pathlib.Path(__file__).parents[1] / "src/models/module")]
        package.__package__ = package_name
        sys.modules[package_name] = package
    return importlib.import_module(f"{package_name}.m4_raydrop_refine")


_REFINE = _load_refine_module()


def _refiner_cfg(**box_mask):
    return OmegaConf.create({
        "channels": 32,
        "depth_scale_m": 80.0,
        "box_mask": {"enable": True, "max_boxes": 32, "max_size_frac": 0.1, **box_mask},
    })


def _renders(batch=2, views=3, height=32, width=64, view_counts=(3, 2)):
    generator = torch.Generator().manual_seed(0)
    shape = (batch, views, 1, height, width)
    view_mask = torch.arange(views).unsqueeze(0) < torch.tensor(view_counts).unsqueeze(1)
    return {
        "raydrop": torch.rand(shape, generator=generator),
        "intensity_sh": torch.rand(shape, generator=generator),
        "depth_median": 80.0 * torch.rand(shape, generator=generator),
        "gt_raydrop": (torch.rand(shape, generator=generator) > 0.7).float(),
        "view_mask": view_mask,
    }


class UNetTest(unittest.TestCase):
    def test_unet_is_the_lidar4d_network(self):
        unet = _REFINE.UNet(in_channels=3, out_channels=1)
        # Official LiDAR4D/GS-LiDAR/LiDAR-RT U-Net at three inputs.
        self.assertEqual(sum(p.numel() for p in unet.parameters()), 6_535_969)
        self.assertEqual(sum(isinstance(m, torch.nn.BatchNorm2d) for m in unet.modules()), 18)

    def test_unet_keeps_both_benchmark_grids(self):
        unet = _REFINE.UNet(in_channels=3, out_channels=1).eval()
        for height, width in ((32, 1085), (66, 1030)):
            with torch.no_grad():
                out = unet(torch.rand(2, 3, height, width))
            self.assertEqual(tuple(out.shape), (2, 1, height, width))
            self.assertTrue(bool(((out >= 0) & (out <= 1)).all()))


class RaydropRefinerTest(unittest.TestCase):
    def test_input_is_raydrop_intensity_and_scaled_median_depth_of_real_views(self):
        refiner = _REFINE.RaydropRefiner(_refiner_cfg())
        renders = _renders()
        x, view_mask = refiner.unet_input(renders)
        self.assertEqual(tuple(x.shape), (5, 3, 32, 64))
        self.assertTrue(torch.equal(x[:, 0:1], renders["raydrop"][view_mask]))
        self.assertTrue(torch.equal(x[:, 1:2], renders["intensity_sh"][view_mask]))
        self.assertTrue(torch.allclose(x[:, 2:3], renders["depth_median"][view_mask] / 80.0))

    def test_padded_views_keep_their_input_and_real_views_are_refined(self):
        refiner = _REFINE.RaydropRefiner(_refiner_cfg()).eval()
        renders = _renders()
        with torch.no_grad():
            refined = refiner(renders)
        self.assertEqual(refined.shape, renders["raydrop"].shape)
        self.assertTrue(torch.equal(refined[1, 2], renders["raydrop"][1, 2]))
        self.assertFalse(torch.equal(refined[0, 0], renders["raydrop"][0, 0]))

    def test_gradient_reaches_only_the_unet(self):
        refiner = _REFINE.RaydropRefiner(_refiner_cfg()).train()
        renders = _renders()
        refined = refiner(renders)
        scores = _REFINE.raydrop_scores(refined, renders["gt_raydrop"], renders["view_mask"])
        scores["loss_raydrop"].backward()
        self.assertTrue(all(p.grad is not None for p in refiner.parameters()))
        self.assertFalse(renders["raydrop"].requires_grad)

    def test_box_mask_only_in_training(self):
        refiner = _REFINE.RaydropRefiner(_refiner_cfg(max_boxes=32))
        x = torch.ones(4, 3, 32, 1085)
        torch.manual_seed(3)
        masked = refiner.random_box_mask(x)
        # One box draw shared by every image and channel, as in LiDAR4D.
        self.assertTrue(torch.equal(masked[0], masked[3]))
        self.assertTrue(torch.equal(masked[0, 0], masked[0, 2]))
        refiner.eval()
        renders = _renders()
        with torch.no_grad():
            self.assertTrue(torch.equal(refiner(renders), refiner(renders)))


class RaydropScoresTest(unittest.TestCase):
    def test_scores_count_real_views_only_with_drop_as_positive(self):
        gt = torch.tensor([1.0, 1.0, 0.0, 0.0]).reshape(1, 1, 1, 1, 4)
        pred = torch.tensor([0.9, 0.2, 0.7, 0.1]).reshape(1, 1, 1, 1, 4)
        padded_gt = torch.cat([gt, torch.zeros_like(gt)], dim=1)
        padded_pred = torch.cat([pred, torch.ones_like(pred)], dim=1)
        scores = _REFINE.raydrop_scores(
            padded_pred, padded_gt, torch.tensor([[True, False]]),
        )
        self.assertAlmostEqual(float(scores["raydrop_acc"]), 0.5)
        # tp=1, fp=1, fn=1 -> precision = recall = 0.5.
        self.assertAlmostEqual(float(scores["raydrop_f1"]), 0.5)
        expected_bce = torch.nn.functional.binary_cross_entropy(pred.flatten(), gt.flatten())
        self.assertAlmostEqual(float(scores["loss_raydrop"]), float(expected_bce), places=6)


class RefineConfigTest(unittest.TestCase):
    def _base_checkpoint(self, tmp, **overrides):
        stored, _ = compose_fresh_config(OmegaConf.from_dotlist([
            f"model.variant={DYNAMIC_VARIANT_V15_3}", "exp_name=v15.3",
            "train.ckpt_path=/old/epoch=16.ckpt", "logger.wandb_run_id=abc123",
            *(f"{key}={value}" for key, value in overrides.items()),
        ]))
        checkpoint = Path(tmp) / "epoch=29.ckpt"
        torch.save({
            "experiment_config": OmegaConf.to_container(stored, resolve=True),
            "epoch": 29, "global_step": 50_000, "state_dict": {},
        }, checkpoint)
        return stored, checkpoint

    def _compose(self, checkpoint, *extra):
        return compose_refine_config(OmegaConf.from_dotlist([
            "refine_config=raydrop_unet", f"raydrop_refine.base_ckpt={checkpoint}",
            "exp_name=v15.3_raydrop_refine", *extra,
        ]))

    def test_fresh_and_model_runs_have_no_refinement(self):
        stored, _ = compose_fresh_config(OmegaConf.from_dotlist([
            f"model.variant={DYNAMIC_VARIANT_V15_3}",
        ]))
        self.assertFalse(raydrop_refine_enabled(stored))

    def test_refine_run_is_the_base_config_plus_the_stage_overlay(self):
        with tempfile.TemporaryDirectory() as tmp:
            stored, checkpoint = self._base_checkpoint(tmp)
            config, source = self._compose(checkpoint)
            self.assertIn("refine overlay", source)
            refine = config.raydrop_refine
            self.assertTrue(refine.enable)
            self.assertEqual(refine.base_ckpt, str(checkpoint.resolve()))
            self.assertEqual(refine.base_ckpt_sha256, checkpoint_sha256(checkpoint))
            self.assertEqual((refine.base_epoch, refine.base_global_step), (29, 50_000))
            self.assertEqual(refine.base_exp_name, "v15.3")
            self.assertEqual((refine.channels, refine.depth_scale_m), (32, 80.0))
            for root in ("model", "p2g", "g2g", "g2p", "dynamic_2dgs", "data", "loss"):
                self.assertEqual(
                    OmegaConf.to_container(config[root], resolve=True),
                    OmegaConf.to_container(stored[root], resolve=True), root,
                )
            self.assertEqual(config.logger.dir, "logs/v15.3_raydrop_refine")
            self.assertIsNone(config.logger.wandb_run_id)
            self.assertIsNone(config.train.ckpt_path)
            self.assertEqual(
                (config.train.lr, config.train.weight_decay, config.train.max_epochs,
                 config.train.batch_size, config.train.grad_accum_steps),
                (1.0e-3, 0.0, 20, 2, 1),
            )
            self.assertEqual(config.mode, "train")

    def test_main_entry_dispatches_refine_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            _stored, checkpoint = self._base_checkpoint(tmp)
            config, _ = resolve_main_config(OmegaConf.from_dotlist([
                "refine_config=raydrop_unet", f"raydrop_refine.base_ckpt={checkpoint}",
                "exp_name=refine_a", "device=[4,5]",
            ]))
            self.assertTrue(raydrop_refine_enabled(config))
            self.assertEqual(list(config.device), [4, 5])

    def test_refine_overlay_cannot_touch_the_frozen_model_or_its_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            _stored, checkpoint = self._base_checkpoint(tmp)
            for extra, message in (
                ("p2g.head_offset_bound=0.3", "cannot set p2g"),
                ("data.gt_ego_radius=1.0", "cannot set data"),
                ("data_config=kitti360", "data_config is for fresh runs"),
                ("train.ckpt_path=/x/epoch=1.ckpt", "resume one with"),
            ):
                with self.assertRaisesRegex(ValueError, message):
                    self._compose(checkpoint, extra)

    def test_refine_run_needs_its_own_name_and_a_base(self):
        with tempfile.TemporaryDirectory() as tmp:
            _stored, checkpoint = self._base_checkpoint(tmp)
            with self.assertRaisesRegex(ValueError, "own exp_name"):
                self._compose(checkpoint, "exp_name=v15.3")
            with self.assertRaisesRegex(ValueError, "requires raydrop_refine.base_ckpt"):
                compose_refine_config(OmegaConf.from_dotlist([
                    "refine_config=raydrop_unet", "exp_name=x",
                ]))
            with self.assertRaisesRegex(ValueError, "requires exp_name"):
                compose_refine_config(OmegaConf.from_dotlist([
                    "refine_config=raydrop_unet", f"raydrop_refine.base_ckpt={checkpoint}",
                ]))

    def test_refinement_checkpoint_resumes_with_its_block_locked(self):
        with tempfile.TemporaryDirectory() as tmp:
            _stored, base = self._base_checkpoint(tmp)
            config, _ = self._compose(base)
            refined = Path(tmp) / "refine_epoch=3.ckpt"
            torch.save({"experiment_config": OmegaConf.to_container(config, resolve=True)}, refined)
            restored, _ = resolve_main_config(OmegaConf.from_dotlist([
                f"train.ckpt_path={refined}", "device=[4,5]",
            ]))
            self.assertTrue(raydrop_refine_enabled(restored))
            self.assertEqual(restored.raydrop_refine.base_epoch, 29)
            with self.assertRaisesRegex(ValueError, "changed roots: raydrop_refine"):
                resolve_main_config(OmegaConf.from_dotlist([
                    f"train.ckpt_path={refined}", "raydrop_refine.channels=64",
                ]))
            with self.assertRaisesRegex(ValueError, "already a refinement checkpoint"):
                compose_refine_config(OmegaConf.from_dotlist([
                    "refine_config=raydrop_unet", f"raydrop_refine.base_ckpt={refined}",
                    "exp_name=again",
                ]))

    def test_model_checkpoint_cannot_switch_refinement_on_from_the_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            _stored, base = self._base_checkpoint(tmp)
            with self.assertRaisesRegex(ValueError, "changed roots: raydrop_refine"):
                resolve_main_config(OmegaConf.from_dotlist([
                    f"train.ckpt_path={base}", "raydrop_refine.enable=true",
                ]))


if __name__ == "__main__":
    unittest.main()
