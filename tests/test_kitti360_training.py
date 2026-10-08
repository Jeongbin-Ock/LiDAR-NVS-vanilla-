import hashlib
import json
from pathlib import Path
import random
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from omegaconf import OmegaConf
import torch

from src.experiment_config import compose_fresh_config, resolve_main_config
from src.dataloader import dataset_dict
from src.dataloader.kitti360.dynamic import KITTI360DynamicDataset
from src.dataloader.kitti360.hdl64e import (
    KITTI360_HDL64E_APEX_HEIGHT_M, KITTI360_HDL64E_ELEVATION_DEG, project_hdl64e_panorama,
)
from src.dataloader.kitti360.reader import EGO_FROM_IMU, project_kitti_panorama
from src.models.utils.camera import StackedPanoramaCamera


class KITTI360TrainingTest(unittest.TestCase):
    def test_profile_replaces_nuscenes_data_and_preserves_model_and_loss(self):
        for variant in ['dynamic_2dgs_attention_velocity_v11_4', 'dynamic_2dgs_attention_velocity_v14_1',
                        'dynamic_2dgs_attention_velocity_v16']:
            base, _ = compose_fresh_config({'model': {'variant': variant}})
            cfg, source = compose_fresh_config({'model': {'variant': variant}, 'data_config': 'kitti360'})
            for key in ['model', 'p2g', 'dynamic_2dgs', 'loss', 'train']:
                self.assertEqual(OmegaConf.to_container(cfg[key], resolve=True), OmegaConf.to_container(base[key], resolve=True))
            self.assertEqual(cfg.data.dataset_name, 'kitti360_dynamic')
            self.assertNotIn('pair_mode', cfg.data)
            self.assertNotIn('train_excluded_scenes', cfg.data.dynamic_window)
            self.assertTrue(cfg.data.compact_val.enable)
            self.assertEqual(cfg.data.compact_val.path, 'config/kitti360_compact_val_windows.json')
            self.assertEqual(cfg.data.window_stride_sec, .5)
            self.assertEqual(cfg.data.dynamic_window.endpoint_max_sec, 1.0)
            self.assertIn('data_profile=', source)
        self.assertIs(dataset_dict['kitti360_dynamic'], KITTI360DynamicDataset)

    def test_kitti_runs_are_named_with_a_kitti_suffix(self):
        profile = {'data_config': 'kitti360'}
        for cli, expected in [
            ({'model': {'variant': 'dynamic_2dgs_attention_velocity_v15_3'}}, 'v15.3_kitti'),
            ({}, 'v11.4_kitti'),
            ({'model': {'variant': 'dynamic_2dgs_attention_velocity_v15_3'}, 'exp_name': 'v15.3'}, 'v15.3_kitti'),
            ({'exp_name': 'v15.3_kitti_lr2'}, 'v15.3_kitti_lr2'),
            ({'exp_name': 'KITTI-ablation'}, 'KITTI-ablation'),
        ]:
            cfg, _ = compose_fresh_config({**cli, **profile})
            resolved = OmegaConf.to_container(cfg, resolve=True)
            self.assertEqual(resolved['exp_name'], expected)
            self.assertEqual(resolved['logger']['dir'], f'logs/{expected}')
        for cli, expected in [
            ({'model': {'variant': 'dynamic_2dgs_attention_velocity_v15_3'}}, 'v15.3_kitti_lidar4d'),
            ({'model': {'variant': 'dynamic_2dgs_attention_velocity_v15_3'}, 'exp_name': 'v15.3'}, 'v15.3_kitti_lidar4d'),
            ({'exp_name': 'v15.3_kitti_lr2'}, 'v15.3_kitti_lr2'),
        ]:
            cfg, _ = compose_fresh_config({**cli, 'data_config': 'kitti360_lidar4d'})
            self.assertEqual(OmegaConf.to_container(cfg, resolve=True)['logger']['dir'], f'logs/{expected}')
        nuscenes, _ = compose_fresh_config({'exp_name': 'v15.3'})
        self.assertEqual(nuscenes.exp_name, 'v15.3')

    def test_lidar4d_profile_changes_only_the_gt_grid(self):
        variant = {'model': {'variant': 'dynamic_2dgs_attention_velocity_v15_3'}}
        calibrated, _ = compose_fresh_config({**variant, 'data_config': 'kitti360'})
        lidar4d, _ = compose_fresh_config({**variant, 'data_config': 'kitti360_lidar4d'})
        a, b = (OmegaConf.to_container(c.data, resolve=True) for c in (calibrated, lidar4d))
        changed = {key for key in a.keys() | b.keys() if a.get(key) != b.get(key)}
        self.assertEqual(changed, {'panorama_projection', 'column_assignment', 'image_height', 'image_width'})
        self.assertEqual((b['panorama_projection'], b['image_height'], b['image_width']), ('uniform', 66, 1030))
        self.assertEqual((b['ego_radius'], b['gt_ego_radius']), (2.5, 2.5))
        for root in ('model', 'p2g', 'dynamic_2dgs', 'loss', 'train'):
            self.assertEqual(OmegaConf.to_container(calibrated[root], resolve=True),
                             OmegaConf.to_container(lidar4d[root], resolve=True))

    def test_checkpoint_locks_manifest_projection_and_sampling(self):
        cfg, _ = compose_fresh_config({'data_config': 'kitti360'})
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'model.ckpt'
            torch.save({'experiment_config': OmegaConf.to_container(cfg, resolve=True)}, checkpoint)
            base_cli = {'train': {'ckpt_path': str(checkpoint)}}
            resumed, _ = resolve_main_config(base_cli)
            self.assertEqual(resumed.data.split_manifest_sha256, cfg.data.split_manifest_sha256)
            for change in [{'window_stride_sec': .4}, {'gt_ego_radius': 0},
                           {'panorama_projection': 'uniform'}, {'dynamic_window': {'endpoint_max_sec': 1.2}}]:
                with self.assertRaisesRegex(ValueError, 'structural'):
                    resolve_main_config(OmegaConf.merge(base_cli, {'data': change}))
            with self.assertRaisesRegex(ValueError, 'fresh runs'):
                resolve_main_config(OmegaConf.merge(base_cli, {'data_config': 'kitti360'}))

    @staticmethod
    def fixture(root):
        cfg = OmegaConf.load('config/data/kitti360.yaml').data
        cfg.dataroot, cfg.verbose = str(root), False
        seq = '2013_05_28_drive_0000_sync'
        scans = root / 'data_3d_raw' / seq / 'velodyne_points'
        (scans / 'data').mkdir(parents=True)
        timestamps = np.datetime64('2013-05-28T00:00:00', 'ns') + np.arange(50) * np.timedelta64(104_500_000, 'ns')
        (scans / 'timestamps.txt').write_text('\n'.join(map(str, timestamps)) + '\n')
        lasers = np.array([0, 31, 32, 63])
        rho, azimuth = np.array([8, 9, 10, 20]), np.radians([0, 90, -120, -30])
        z = rho * np.tan(np.radians(KITTI360_HDL64E_ELEVATION_DEG[lasers])) + KITTI360_HDL64E_APEX_HEIGHT_M[lasers]
        points = np.c_[rho * np.cos(azimuth), rho * np.sin(azimuth), z, [.7, .3, .5, .9]].astype(np.float32)
        for frame in range(50):
            points.tofile(scans / 'data' / f'{frame:010d}.bin')
        poses = []
        for frame in range(1, 48):
            if frame == 3:
                continue
            matrix = EGO_FROM_IMU.copy()
            matrix[0, 3] = 1_000_000 + frame * .1
            poses.append(np.r_[frame, matrix[:3].ravel()])
        directory = root / 'data_poses' / seq
        directory.mkdir(parents=True)
        np.savetxt(directory / 'poses.txt', poses, fmt='%.12f')
        calibration = root / 'calibration'
        calibration.mkdir()
        (calibration / 'calib_cam_to_pose.txt').write_text('image_00: ' + ' '.join(map(str, EGO_FROM_IMU[:3].ravel())) + '\n')
        np.savetxt(calibration / 'calib_cam_to_velo.txt', np.eye(4)[:3].reshape(1, -1))
        manifest = {
            'manifest_version': 2, 'dataroot': str(root), 'config': {'max_sweep_gap_sec': .2},
            'train_segments': {seq: {'segments': [{'first_frame': 0, 'last_frame': 17, 'num_sweeps': 18}]}},
            'val_segments': {seq: {'segments': [{'first_frame': 20, 'last_frame': 49, 'num_sweeps': 30}]}},
        }
        path = root / 'split.json'
        path.write_text(json.dumps(manifest))
        cfg.split_manifest_path = str(path)
        cfg.split_manifest_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        return cfg, points

    def test_pose_bounds_physical_stride_and_val_draws_are_stable(self):
        with TemporaryDirectory() as directory:
            cfg, _ = self.fixture(Path(directory))
            train = KITTI360DynamicDataset(cfg)
            val = KITTI360DynamicDataset(cfg, split='val')
            self.assertEqual(len(train), 4)
            self.assertEqual(len(val), 6)
            self.assertEqual(train.pose_unavailable_sweeps, 1)
            self.assertEqual(val.pose_unavailable_sweeps, 2)
            self.assertEqual(val.sample_window_frames(0), val.sample_window_frames(0))
            random.seed(42)
            draws = {tuple(train.sample_window_frames(0)[1]) for _ in range(20)}
            self.assertGreater(len(draws), 1)
            for frames in draws:
                self.assertEqual(frames[0], 1)
                duration = (frames[-1] - frames[0]) * .1045
                self.assertGreaterEqual(duration, .2)
                self.assertLessEqual(duration, 1.0)
                self.assertEqual(len(frames), len(set(frames)))

    def test_gt_cameras_and_collation_share_calibrated_inference_projection(self):
        with TemporaryDirectory() as directory:
            cfg, points = self.fixture(Path(directory))
            val = KITTI360DynamicDataset(cfg, split='val')
            long_item, short_item = val[0], val[len(val) - 1]
            expected_depth, expected_intensity, _ = project_hdl64e_panorama(points, cfg.image_width)
            for item in [long_item, short_item]:
                self.assertGreater(float(item['input']['lidar_points_sensor'][:, 3].max()), .8)
                for camera in item['gt']['cameras']:
                    self.assertIsInstance(camera, StackedPanoramaCamera)
                    np.testing.assert_array_equal(camera.pts_depth.numpy(), expected_depth)
                    np.testing.assert_array_equal(camera.pts_intensity.numpy(), expected_intensity)
                    self.assertEqual(camera.pts_depth.shape, (1, 64, 2178))
                    self.assertGreaterEqual(camera.timestamp_sec, 0)
            self.assertLessEqual(len(short_item['gt']['cameras']), 4)
            batch = val.collate_fn([long_item, short_item])
            self.assertEqual(batch['input']['window_duration_sec'].shape, (2,))
            self.assertEqual(batch['input']['ptv3_input']['feat'].shape[1], 4)
            self.assertEqual(batch['input']['ptv3_input']['offset'].shape, (4,))
            self.assertEqual([len(views) for views in batch['gt']['cameras']],
                             [len(long_item['gt']['cameras']), len(short_item['gt']['cameras'])])

    def test_lidar4d_grid_renders_one_sensor_origin_camera_per_view(self):
        with TemporaryDirectory() as directory:
            cfg, points = self.fixture(Path(directory))
            profile = OmegaConf.load('config/data/kitti360_lidar4d.yaml').data
            for key in ('panorama_projection', 'image_height', 'image_width', 'vfov'):
                cfg[key] = profile[key]
            item = KITTI360DynamicDataset(cfg, split='val')[0]
            expected, _ = project_kitti_panorama(points, 66, 1030, cfg.vfov, cfg.max_depth_m)
            for camera in item['gt']['cameras']:
                self.assertNotIsInstance(camera, StackedPanoramaCamera)
                self.assertIsNone(camera.row_origin)
                self.assertEqual(tuple(camera.pts_depth.shape), (1, 66, 1030))
                np.testing.assert_array_equal(camera.pts_depth.numpy(), expected)

    def test_manifest_tampering_and_benchmark_leakage_fail_before_loading_points(self):
        with TemporaryDirectory() as directory:
            cfg, _ = self.fixture(Path(directory))
            path = Path(cfg.split_manifest_path)
            manifest = json.loads(path.read_text())
            path.write_text(path.read_text() + '\n')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                KITTI360DynamicDataset(cfg)
            manifest['train_segments']['2013_05_28_drive_0000_sync']['segments'] = [
                {'first_frame': 1537, 'last_frame': 1540, 'num_sweeps': 4},
            ]
            path.write_text(json.dumps(manifest))
            cfg.split_manifest_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError, 'LiDAR4D test clip'):
                KITTI360DynamicDataset(cfg)

    @unittest.skipUnless(torch.cuda.is_available(), 'needs the CUDA rasterizer')
    def test_calibrated_training_gt_renders_and_backpropagates_with_ragged_views(self):
        from src.models.module.m3_g2p import DynamicGausRender
        from src.models.utils.loss import Loss
        with TemporaryDirectory() as directory:
            cfg, points = self.fixture(Path(directory))
            val = KITTI360DynamicDataset(cfg, split='val')
            items = [val[0], val[len(val) - 1]]
            gt = {'cameras': [item['gt']['cameras'] for item in items]}
            device = torch.device('cuda')
            torch.manual_seed(42)
            position = (torch.from_numpy(points[:, :3]).to(device).repeat(32, 1)
                        + .02 * torch.randn(128, 3, device=device)).requires_grad_()
            scales = torch.full((128, 3), float(np.log(.08)), device=device)
            scales[:, 2] = -10
            scales.requires_grad_()
            velocity = torch.zeros_like(position, requires_grad=True)
            opacity = torch.full((128, 1), 2., device=device, requires_grad=True)
            rotation = torch.nn.functional.normalize(torch.randn(128, 4, device=device), dim=1).requires_grad_()
            shs = (.1 * torch.randn(128, 16, 4, device=device)).requires_grad_()
            gaussians = {'position': position, 'velocity': velocity,
                         'source_time_sec': torch.zeros(128, device=device),
                         'scales': scales, 'scaling': scales, 'rotation': rotation,
                         'opacity': opacity, 'shs': shs}
            renderer = DynamicGausRender(OmegaConf.create({'neg_fov': True, 'sh_degree': 3, 'scale_factor': 1.0}))
            renders = renderer([gaussians, gaussians], gt, compute_points=False)
            self.assertEqual(renders['depth'].shape[-2:], (64, 2178))
            self.assertEqual(renders['view_mask'].sum(dim=1).tolist(), [len(views) for views in gt['cameras']])
            self.assertGreater(int((renders['depth'] > 0).sum()), 0)
            cfg_loss = OmegaConf.create({'w_chamfer': 0., 'w_depth': .5, 'w_depth_median': .5,
                                        'w_intensity': .2, 'w_raydrop': .5, 'scale_max_m': 2.5})
            losses = Loss(cfg_loss)(renders, compute_valid_metrics=False, compute_raydrop_metrics=False)
            self.assertTrue(torch.isfinite(losses['total']))
            losses['total'].backward()
            for value in [position, velocity, scales, opacity, rotation, shs]:
                self.assertIsNotNone(value.grad)
                self.assertTrue(torch.isfinite(value.grad).all())
            self.assertGreater(float(position.grad.abs().sum()), 0)


if __name__ == '__main__':
    unittest.main()
