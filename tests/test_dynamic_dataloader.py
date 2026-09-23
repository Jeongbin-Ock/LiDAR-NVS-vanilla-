from __future__ import annotations

import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from omegaconf import OmegaConf

from src.dataloader.nuscene_dynamic_lidar4d import (
    LIDAR4D_TRAIN_SEQUENCE_SCENES,
    NuScenesDynamicLiDAR4DDataset,
    dynamic_lidar4d_multiframe_collate_fn,
    dynamic_transform,
    eligible_endpoint_indices,
    sample_target_indices,
)
from src.dataloader.nuscene import (
    filter_train_scene_names,
    transform as legacy_transform,
)
from src.models_new.module.m3_g2p import DynamicGausRender
from src.models_new.utonia.lora import build_xyzi_utonia_input
from src.models_new.utils.loss import Loss
from src.models_new.utils.model_utils import DataModule


def _frames(count: int, cadence_us: int = 50_000):
    return [
        (f"lidar-{index}", index * cadence_us, index % 10 == 0)
        for index in range(count)
    ]


class _FirstChoiceRng:
    def choice(self, values):
        return values[0]

    def sample(self, values, count):
        return list(values[:count])


class DynamicWindowSamplingTest(unittest.TestCase):
    def test_lidar4d_train_sequences_map_to_the_held_out_scenes(self):
        self.assertEqual(
            LIDAR4D_TRAIN_SEQUENCE_SCENES,
            {
                "seq_1600_1650": "scene-0757",
                "seq_3180_3230": "scene-1094",
            },
        )

    def test_endpoints_are_uniform_candidates_on_the_point_one_second_grid(self):
        candidates = eligible_endpoint_indices(
            _frames(40),
            0,
            sample_hop=2,
            min_duration_us=200_000,
            max_duration_us=1_200_000,
        )
        self.assertEqual(candidates, list(range(4, 25, 2)))

    def test_endpoint_candidates_truncate_at_scene_end_without_clamping(self):
        candidates = eligible_endpoint_indices(
            _frames(20),
            0,
            sample_hop=2,
            min_duration_us=200_000,
            max_duration_us=1_200_000,
        )
        self.assertEqual(candidates, list(range(4, 20, 2)))
        self.assertEqual(len(candidates), len(set(candidates)))

    def test_novel_views_use_three_when_available_and_all_when_short(self):
        rng = _FirstChoiceRng()
        expected = {
            4: [2],       # 0.2 s: one interior view
            6: [2, 4],    # 0.3 s: two interior views
            8: [2, 4, 6], # 0.4 s: three interior views
            24: [2, 4, 6],
        }
        for endpoint, novel_expected in expected.items():
            with self.subTest(endpoint=endpoint):
                end, novel, used = sample_target_indices(
                    0,
                    [endpoint],
                    sample_hop=2,
                    novel_view_count=3,
                    rng=rng,
                )
                self.assertEqual(end, endpoint)
                self.assertEqual(novel, novel_expected)
                self.assertEqual(used, [0, *novel_expected, endpoint])

    def test_eval_randomness_is_stable_per_nominal_start_anchor(self):
        dataset = NuScenesDynamicLiDAR4DDataset.__new__(
            NuScenesDynamicLiDAR4DDataset
        )
        dataset.cfg = OmegaConf.create({"train_split": "train"})
        dataset.split = "val"
        dataset.eval_seed = 17
        dataset.scene_frames = [_frames(30)]
        dataset.index = [(0, 0, 20)]
        first = dataset._sampling_rng(0)
        second = dataset._sampling_rng(0)
        self.assertEqual(
            [first.random() for _ in range(4)],
            [second.random() for _ in range(4)],
        )


class LegacySceneExclusionTest(unittest.TestCase):
    def test_train_removes_complete_lidar4d_scenes(self):
        active = filter_train_scene_names(
            {"scene-0001", "scene-0757", "scene-1094"},
            split="train",
            train_split="train",
            train_excluded_scenes=("scene-0757", "scene-1094"),
        )
        self.assertEqual(active, {"scene-0001"})

    def test_val_keeps_the_same_scene_names(self):
        scenes = {"scene-0757", "scene-1094", "scene-0002"}
        active = filter_train_scene_names(
            scenes,
            split="val",
            train_split="train",
            train_excluded_scenes=("scene-0757", "scene-1094"),
        )
        self.assertEqual(active, scenes)

    def test_duplicate_or_unknown_train_exclusion_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicates"):
            filter_train_scene_names(
                {"scene-0757"},
                split="train",
                train_split="train",
                train_excluded_scenes=("scene-0757", "scene-0757"),
            )
        with self.assertRaisesRegex(ValueError, "not in the train split"):
            filter_train_scene_names(
                {"scene-0757"},
                split="train",
                train_split="train",
                train_excluded_scenes=("scene-1094",),
            )


class DynamicCollateTest(unittest.TestCase):
    @staticmethod
    def _item(first_count, second_count, cameras):
        first = torch.stack(
            [
                torch.tensor([float(i), 0.0, 0.0, 0.25 + 0.01 * i])
                for i in range(first_count)
            ]
        )
        second = torch.stack(
            [
                torch.tensor([float(i), 1.0, 0.0, 0.5 + 0.01 * i])
                for i in range(second_count)
            ]
        )
        return {
            "input": {
                "lidar_points_sensor": torch.cat([first, second]),
                "frame_counts": torch.tensor([first_count, second_count]),
                "pose": torch.eye(4).repeat(2, 1, 1),
                "timestamps": torch.tensor([0.0, 1.0]),
                "timestamps_sec": torch.tensor([0.0, 0.4]),
                "window_duration_sec": torch.tensor(0.4),
            },
            "gt": {"cameras": cameras},
        }

    def test_minimal_collate_matches_v11_input_contract_and_keeps_ragged_cameras(self):
        batch = dynamic_lidar4d_multiframe_collate_fn(
            [self._item(3, 2, ["a", "b", "c"]), self._item(2, 3, ["d"] * 5)]
        )
        source = batch["input"]
        self.assertEqual(
            set(source),
            {
                "lidar_points_sensor", "batch_idx", "offset", "pose",
                "timestamps", "timestamps_sec", "window_duration_sec",
                "ptv3_input",
            },
        )
        self.assertEqual(source["offset"].tolist(), [3, 5, 7, 10])
        self.assertEqual(source["batch_idx"].tolist(), [0] * 5 + [1] * 5)
        self.assertEqual([len(value) for value in batch["gt"]["cameras"]], [3, 5])
        self.assertEqual(
            set(source["ptv3_input"]),
            {"coord", "grid_coord", "feat", "strength", "offset"},
        )
        self.assertEqual(source["ptv3_input"]["feat"].shape[-1], 4)
        torch.testing.assert_close(
            source["ptv3_input"]["feat"][:, :3],
            source["ptv3_input"]["coord"],
        )
        torch.testing.assert_close(
            source["ptv3_input"]["feat"][:, 3:4],
            source["ptv3_input"]["strength"],
        )

    def test_datamodule_selects_the_dynamic_collate_function(self):
        module = DataModule(NuScenesDynamicLiDAR4DDataset, OmegaConf.create({}))
        self.assertIs(
            module._collate_fn(), dynamic_lidar4d_multiframe_collate_fn
        )

    def test_minimal_transform_matches_legacy_v11_xyzi_input(self):
        coord = np.array(
            [[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [1.0, 2.0, 3.0]],
            dtype=np.float32,
        )
        strength = np.array([[0.1], [0.2], [0.3]], dtype=np.float32)
        np.random.seed(81)
        legacy = legacy_transform({
            "coord": coord.copy(),
            "color": np.zeros_like(coord),
            "normal": np.zeros_like(coord),
            "strength": strength.copy(),
        })
        legacy = build_xyzi_utonia_input(legacy)
        np.random.seed(81)
        dynamic = dynamic_transform({
            "coord": coord.copy(),
            "strength": strength.copy(),
        })
        for key in ("coord", "grid_coord", "strength", "feat", "offset"):
            torch.testing.assert_close(dynamic[key], legacy[key])


class RaggedViewLossTest(unittest.TestCase):
    @staticmethod
    def _loss():
        return Loss(OmegaConf.create({
            "w_chamfer": 0.0,
            "w_depth": 0.5,
            "w_depth_median": 0.5,
            "w_intensity": 0.2,
            "w_raydrop": 0.5,
            "w_scale": 0.0,
            "scale_max_m": 2.5,
            "enable_lpips": False,
        }))

    def test_view_padding_is_ignored_by_every_reconstruction_loss(self):
        shape = (2, 5, 2, 2)
        mask = torch.tensor(
            [[True, True, True, False, False], [True, True, True, True, True]]
        )
        valid_pixels = mask[:, :, None, None].expand(shape)
        gt_depth = torch.full(shape, 100.0)
        pred_depth = torch.full(shape, -50.0)
        gt_intensity = torch.full(shape, 100.0)
        pred_intensity = torch.full(shape, -50.0)
        gt_raydrop = torch.ones(shape)
        pred_raydrop = torch.full(shape, 0.99)
        gt_depth[valid_pixels] = 1.0
        pred_depth[valid_pixels] = 2.0
        gt_intensity[valid_pixels] = 0.25
        pred_intensity[valid_pixels] = 0.75
        gt_raydrop[valid_pixels] = 0.0
        pred_raydrop[valid_pixels] = 0.25
        renders = {
            "depth": pred_depth,
            "depth_median": pred_depth.clone(),
            "intensity_sh": pred_intensity,
            "raydrop": pred_raydrop,
            "gt_depth": gt_depth,
            "gt_intensity_sh": gt_intensity,
            "gt_raydrop": gt_raydrop,
            "view_mask": mask,
            "render_points": [[], []],
            "gt_points": [[], []],
        }
        losses = self._loss()(
            renders,
            metric_mode="train",
            compute_valid_metrics=False,
            compute_raydrop_metrics=False,
        )
        self.assertAlmostEqual(float(losses["loss_depth"]), 1.0, places=6)
        self.assertAlmostEqual(float(losses["loss_depth_median"]), 1.0, places=6)
        self.assertAlmostEqual(float(losses["loss_intensity"]), 0.5, places=6)
        self.assertAlmostEqual(
            float(losses["loss_raydrop"]), -math.log(0.75), places=6
        )

    def test_view_padding_is_ignored_by_image_metrics(self):
        generator = torch.Generator().manual_seed(12)
        shape = (2, 5, 8, 8)
        mask = torch.tensor(
            [[True, True, True, False, False], [True, True, True, True, True]]
        )
        valid_pixels = mask[:, :, None, None].expand(shape)
        gt_depth = 1.0 + 20.0 * torch.rand(shape, generator=generator)
        gt_intensity = torch.rand(shape, generator=generator)
        base = {
            "depth": gt_depth * 0.9,
            "depth_median": gt_depth * 0.95,
            "intensity_sh": gt_intensity * 0.8,
            "raydrop": torch.full(shape, 0.2),
            "gt_depth": gt_depth,
            "gt_intensity_sh": gt_intensity,
            "gt_raydrop": torch.zeros(shape),
            "view_mask": mask,
            "render_points": [[], []],
            "gt_points": [[], []],
        }
        corrupted_padding = {
            key: value.clone() if torch.is_tensor(value) else value
            for key, value in base.items()
        }
        for key in (
            "depth", "depth_median", "intensity_sh", "raydrop",
            "gt_depth", "gt_intensity_sh", "gt_raydrop",
        ):
            corrupted_padding[key][~valid_pixels] = 999.0
        expected = self._loss()(
            base,
            metric_mode="train",
            compute_valid_metrics=True,
            compute_raydrop_metrics=True,
        )
        actual = self._loss()(
            corrupted_padding,
            metric_mode="train",
            compute_valid_metrics=True,
            compute_raydrop_metrics=True,
        )
        for key in (
            "loss_depth", "loss_depth_median", "loss_intensity",
            "loss_raydrop", "intensity_psnr_valid", "intensity_ssim_valid",
            "depth_psnr_valid", "depth_ssim_valid",
            "intensity_psnr_raydrop", "intensity_ssim_raydrop",
            "depth_psnr_raydrop", "depth_ssim_raydrop",
        ):
            torch.testing.assert_close(actual[key], expected[key])

    def test_renderer_padding_helper_preserves_real_rows_and_zeroes_missing_rows(self):
        short = torch.ones(3, 1, 2, 2)
        long = torch.full((5, 1, 2, 2), 2.0)
        packed = DynamicGausRender._pad_view_batches([short, long], 5)
        self.assertEqual(tuple(packed.shape), (2, 5, 1, 2, 2))
        torch.testing.assert_close(packed[0, :3], short)
        self.assertEqual(int(torch.count_nonzero(packed[0, 3:])), 0)
        torch.testing.assert_close(packed[1], long)

    def test_dynamic_renderer_accepts_ragged_camera_lists_and_emits_mask(self):
        renderer = DynamicGausRender(OmegaConf.create({})).train()
        gaussian = {
            "position": torch.zeros(2, 3),
            "velocity": torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            "source_time_sec": torch.tensor([0.0, 0.4]),
            "opacity": torch.ones(2, 1),
            "scaling": torch.zeros(2, 2),
            "rotation": torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(2, 1),
            "shs": torch.zeros(2, 32),
        }

        def camera(timestamp):
            return SimpleNamespace(
                timestamp_sec=float(timestamp),
                pts_depth=torch.ones(1, 2, 2),
                pts_intensity=torch.full((1, 2, 2), 0.5),
            )

        cameras = [
            [camera(value) for value in (0.0, 0.2, 0.4)],
            [camera(value) for value in (0.0, 0.1, 0.2, 0.3, 0.4)],
        ]

        def fake_render(*, viewpoint_camera, pc, **_kwargs):
            value = pc["position"].mean()
            image = value.expand(1, 2, 2)
            return {
                "depth": image,
                "depth_median": image,
                "intensity_sh": image.sigmoid(),
                "raydrop": image.sigmoid(),
            }

        with patch("src.models_new.module.m3_g2p.render", side_effect=fake_render):
            renders = renderer(
                [gaussian, gaussian], {"cameras": cameras}, compute_points=False
            )
        self.assertEqual(tuple(renders["depth"].shape), (2, 5, 1, 2, 2))
        self.assertEqual(
            renders["view_mask"].tolist(),
            [[True, True, True, False, False], [True, True, True, True, True]],
        )
        self.assertEqual(int(torch.count_nonzero(renders["depth"][0, 3:])), 0)


if __name__ == "__main__":
    unittest.main()
