from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch
from omegaconf import OmegaConf

from src.experiment_config import compose_fresh_config, resolve_eval_config, resolve_main_config
from src.dataloader.kitti360.lidar4d_test import LiDAR4DKITTI360TestDataset
from src.dataloader.kitti360.metadata import KITTI360_BENCHMARK_SCENES
from src.dataloader.kitti360.reader import (
    CAMERA_FROM_LIDAR, EGO_FROM_IMU, KITTI360Poses, project_kitti_panorama,
)
from src.dataloader.kitti360.hdl64e import (
    KITTI360_HDL64E_APEX_HEIGHT_M, KITTI360_HDL64E_BLOCK_ORIGIN_Z_M, KITTI360_HDL64E_ELEVATION_DEG,
    assign_lasers, firing_slot_columns, project_hdl64e_panorama,
)
from src.models.utils.camera import Camera, StackedPanoramaCamera
from src.models.utils.graphics_utils import pano_to_lidar
from src.eval.depth_diagnostics import depth_error_breakdown, summarize_depth_breakdowns
from src.dataloader.lidar4d_test import build_lidar4d_test_dataset


class KITTI360InferenceTest(unittest.TestCase):
    def test_depth_rmse_breakdown_separates_missing_and_extra_returns(self):
        result = depth_error_breakdown(np.array([12, 5, 0, 0]), np.array([10, 0, 10, 0]))
        self.assertAlmostEqual(result['all_pixel_rmse'], np.sqrt(129/4), places=5)
        self.assertEqual(result['regions']['both_return']['pixels'], 1)
        self.assertAlmostEqual(result['regions']['both_return']['rmse'], 2)
        self.assertAlmostEqual(result['regions']['missed_gt_return']['squared_error_sum'], 100, places=4)
        self.assertAlmostEqual(result['regions']['extra_pred_return']['squared_error_sum'], 25, places=4)
        self.assertAlmostEqual(sum(r['squared_error_sum'] for r in result['regions'].values()),
                               result['squared_error_sum'])
        combined = summarize_depth_breakdowns([result, result])
        self.assertAlmostEqual(combined['pooled_all_pixel_rmse'], result['all_pixel_rmse'])
        self.assertEqual(combined['regions']['missed_gt_return']['pixels'], 2)

    def test_empty_return_depth_diagnostics_have_no_invented_error(self):
        result = depth_error_breakdown(np.zeros((2, 3)), np.zeros((2, 3)))
        self.assertEqual(result['all_pixel_rmse'], 0)
        self.assertIsNone(result['regions']['both_return']['rmse'])
        self.assertEqual(result['regions']['both_no_return']['pixels'], 6)

    def test_official_held_out_counts(self):
        self.assertEqual(len(KITTI360_BENCHMARK_SCENES), 10)
        self.assertEqual(sum(len(value[2]) for value in KITTI360_BENCHMARK_SCENES.values()), 40)
        self.assertEqual(KITTI360_BENCHMARK_SCENES[1538][2], (1551, 1564, 1577, 1590))
        self.assertEqual(KITTI360_BENCHMARK_SCENES[11400][2], (11410, 11420, 11430, 11440))

    def test_panorama_nearest_return_intensity_and_far_cutoff(self):
        points = np.array([[10, 0, 0, .7], [5, 0, 0, .4], [80, 0, 0, .9],
                           [0, 10, 0, .3], [np.nan, 0, 0, .2]], dtype=np.float32)
        depth, intensity = project_kitti_panorama(points, 66, 1030, (-24.9, 2.0))
        self.assertEqual(depth.shape, (1, 66, 1030))
        self.assertEqual(depth[0, 5, 515], 5)
        self.assertAlmostEqual(float(intensity[0, 5, 515]), .4)
        self.assertEqual(depth[0, 5, 258], 10)
        self.assertEqual(np.count_nonzero(depth), 2)

    def test_camera_axes_match_official_kitti_panorama(self):
        point = np.array([10.0, 0.0, 0.0, 1.0])
        camera = CAMERA_FROM_LIDAR @ point
        self.assertEqual(float(np.arctan2(camera[0], camera[2])), 0.0)
        np.testing.assert_allclose(np.linalg.det(EGO_FROM_IMU[:3, :3]), 1)
        np.testing.assert_allclose(EGO_FROM_IMU @ np.array([1, 2, 3, 1]), [1, -2, -3, 1])

    @staticmethod
    def _laser_points(lasers, azimuth_deg, rho, intensity=0.5):
        """Returns exactly on the fitted cones: z = rho*tan(elevation) + apex."""
        lasers, azimuth, rho = np.broadcast_arrays(np.asarray(lasers), np.radians(azimuth_deg), rho)
        z = rho*np.tan(np.radians(KITTI360_HDL64E_ELEVATION_DEG[lasers])) + KITTI360_HDL64E_APEX_HEIGHT_M[lasers]
        return np.c_[rho*np.cos(azimuth), rho*np.sin(azimuth), z, np.full(len(z), intensity)].astype(np.float32)

    def test_calibrated_laser_table_blocks_and_cone_assignment(self):
        angles, apex = KITTI360_HDL64E_ELEVATION_DEG, KITTI360_HDL64E_APEX_HEIGHT_M
        self.assertEqual((len(angles), len(apex)), (64, 64))
        self.assertTrue(np.all(np.diff(angles) < 0))
        self.assertTrue(np.all((apex[:32] > .19) & (apex[:32] < .22)))
        self.assertTrue(np.all((apex[32:] > .11) & (apex[32:] < .13)))
        upper, lower = KITTI360_HDL64E_BLOCK_ORIGIN_Z_M
        self.assertLess(np.abs(apex[:32]-upper).max(), .006)
        self.assertLess(np.abs(apex[32:]-lower).max(), .005)
        lasers = np.arange(64).repeat(4)
        points = self._laser_points(lasers, np.tile([0, 90, 180, -45], 64), np.tile([1.9, 7.5, 30, 79], 64))
        found, residual = assign_lasers(points[:, :3])
        np.testing.assert_array_equal(found, lasers)
        self.assertLess(residual.max(), 1e-5)
        stray = np.array([[10, 0, 10*np.tan(np.radians(angles[5]))+apex[5]+.03, .1]], dtype=np.float32)
        with self.assertRaisesRegex(ValueError, 'laser cone'):
            project_hdl64e_panorama(stray, 2178)

    def test_firing_slots_split_encoder_quantized_returns(self):
        width = 2178
        # One firing per 48 us at 9.57 Hz, azimuths stored on the 0.09 deg encoder grid.
        firing = np.arange(2170) * 360/2176.67 + 0.037
        azimuth = np.round(firing/0.09)*0.09
        position = (180 - azimuth)/360*width % width
        rounded = np.rint(position).astype(np.int64) % width
        self.assertGreater(len(rounded) - len(np.unique(rounded)), 50)
        columns, shift = firing_slot_columns(position, width)
        self.assertEqual(len(np.unique(columns)), len(columns))
        self.assertLessEqual(np.abs(shift).max(), 1.0)
        # Column order follows azimuth order, wrapping the seam at most once.
        steps = np.diff(columns[np.argsort(position)]) % width
        self.assertTrue(np.all(steps >= 1))
        self.assertLess(int(steps.sum()), width)

    def test_calibrated_projection_rows_block_ranges_and_backprojection(self):
        width = 2178
        lasers = np.array([0, 17, 31, 32, 50, 63, 63])
        # Away from half-column boundaries, so float32 storage cannot flip a column.
        azimuth = np.array([0, 91, -91, 179.9, 45, -151, -151])
        rho = np.array([20, 8, 3, 12, 6, 4, 2.5])
        points = self._laser_points(lasers, azimuth, rho)
        depth, intensity, stats = project_hdl64e_panorama(points, width, column_assignment='nearest')
        self.assertEqual(stats['gt_pixels'], 6)
        self.assertEqual(stats['collision_points'], 1)
        block_z = np.where(lasers < 32, *KITTI360_HDL64E_BLOCK_ORIGIN_Z_M)
        ranges = np.linalg.norm(points[:, :3] - np.c_[np.zeros((7, 2)), block_z], axis=1)
        columns = np.rint((180 - azimuth)/360*width).astype(int) % width
        np.testing.assert_allclose(depth[0, lasers[:5], columns[:5]], ranges[:5], rtol=1e-6)
        # The two returns sharing laser 63's pixel keep the nearer one.
        self.assertAlmostEqual(float(depth[0, 63, columns[5]]), float(ranges[6]), places=5)
        theta = Camera._build_row_to_theta(64, (-24.9, 2), KITTI360_HDL64E_ELEVATION_DEG)
        origin = np.where(np.arange(64)[:, None] < 32, *[CAMERA_FROM_LIDAR[:3, :3] @ [0, 0, z]
                                                         for z in KITTI360_HDL64E_BLOCK_ORIGIN_Z_M])
        reconstructed = pano_to_lidar(torch.from_numpy(depth), (-24.9, 2), (-180, 180), theta,
                                      row_origin=torch.from_numpy(origin)).numpy()
        expected = (points[[0, 1, 2, 3, 4, 6], :3] @ CAMERA_FROM_LIDAR[:3, :3].T)
        order = np.lexsort((np.r_[columns[:5], columns[6]], lasers[[0, 1, 2, 3, 4, 6]]))
        # Error is the half-column azimuth rounding plus the <=5.6 mm block-origin offset.
        np.testing.assert_allclose(reconstructed, expected[order], atol=0.03)

    def test_pose_interpolation_uses_time_and_rotation_not_frame_fraction(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary)/'poses.txt'
            first = np.eye(4)
            last = np.array([[0, -1, 0, 4], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1.]])
            np.savetxt(path, np.stack([np.r_[0, first[:3].ravel()], np.r_[2, last[:3].ravel()]]))
            stamps = np.array([0, 100_000_000, 400_000_000])
            poses = KITTI360Poses(path, stamps, 'interpolate')
            matrix, source = poses.at(1)
            self.assertEqual(source, 'interpolated:0:2')
            self.assertAlmostEqual(matrix[0, 3], 1.0)
            self.assertAlmostEqual(np.arctan2(matrix[1, 0], matrix[0, 0]), np.pi/8)
            previous, source = KITTI360Poses(path, stamps, 'previous').at(1)
            np.testing.assert_allclose(previous, first)
            self.assertEqual(source, 'previous:0')
            with self.assertRaises(ValueError):
                KITTI360Poses(path, stamps, 'error').at(1)

    def _fixture(self, root):
        cfg = OmegaConf.load('config/eval/kitti360_inference.yaml').data
        # Generic fixture retains the historical uniform benchmark. Dedicated
        # tests below exercise the new default/profile independently.
        cfg.panorama_projection = 'uniform'
        cfg.pose_missing_policy = 'previous'  # fixture exercises the carry-forward path
        cfg.image_height = 66
        cfg.image_width = 1030
        cfg.dataroot = str(root)
        cfg.benchmark_scenes = [2350]
        cfg.verbose = False
        sequence = cfg.sequence
        scans = root/'data_3d_raw'/sequence/'velodyne_points'
        (scans/'data').mkdir(parents=True)
        stamps = np.datetime64('2013-05-28T00:00:00', 'ns') + np.arange(2401)*np.timedelta64(104_500_000, 'ns')
        (scans/'timestamps.txt').write_text('\n'.join(str(value) for value in stamps)+'\n')
        poses = []
        for frame in range(2350, 2401):
            np.array([[10, 0, 0, .6], [0, 10, 0, .4]], dtype=np.float32).tofile(scans/'data'/f'{frame:010d}.bin')
            if frame == 2365:
                continue
            pose = EGO_FROM_IMU.copy()
            pose[0, 3] = 1_000_000.0 + (frame-2350)*.1
            poses.append(np.r_[frame, pose[:3].ravel()])
        directory = root/'data_poses'/sequence
        directory.mkdir(parents=True)
        np.savetxt(directory/'poses.txt', poses, fmt='%.12f')
        calibration = root/'calibration'
        calibration.mkdir()
        (calibration/'calib_cam_to_pose.txt').write_text('image_00: '+' '.join(map(str,EGO_FROM_IMU[:3].ravel()))+'\n')
        np.savetxt(calibration/'calib_cam_to_velo.txt',np.eye(4)[:3].reshape(1,-1))
        return cfg

    def test_real_time_endpoints_pose_fallback_and_input_contract(self):
        with TemporaryDirectory() as temporary:
            cfg = self._fixture(Path(temporary))
            dataset = build_lidar4d_test_dataset(cfg)
            self.assertEqual(len(dataset), 4)
            self.assertEqual(dataset.windows[0], (2355, 2360, 2365))
            held_out = set(KITTI360_BENCHMARK_SCENES[2350][2])
            for before,target,after in dataset.windows:
                self.assertNotIn(before, held_out)
                self.assertNotIn(after, held_out)
                self.assertIn(target, held_out)
            item = dataset[0]
            inp,gt = item['input'],item['gt']
            np.testing.assert_allclose(inp['timestamps_sec'],[0,1.045],atol=1e-6)
            np.testing.assert_allclose(gt['gt_timestamps_sec'],[0,.5225,1.045],atol=1e-6)
            self.assertEqual(dataset.index_meta[0]['pose_sources'][-1], 'previous:2364')
            self.assertAlmostEqual(float(inp['pose'][1,0,3]), .9, places=5)
            self.assertAlmostEqual(float(inp['lidar_points_sensor'][0,3]), .6, places=6)
            np.testing.assert_allclose(inp['lidar_points_local'],inp['lidar_points_sensor'])
            camera = gt['gt_cameras'][1]
            point_ref = gt['gt_lidar_points'][2,:3].numpy()
            np.testing.assert_allclose((camera.world_view_transform.T.numpy() @ np.r_[point_ref,1])[:3], [0,0,10],atol=1e-5)
            self.assertEqual(dataset.target_boxes_ref(0).shape,(0,7))
            with self.assertRaisesRegex(ValueError,'clip bounds'):
                LiDAR4DKITTI360TestDataset(cfg,input_offsets_sec=(-2,2))
            with self.assertRaisesRegex(ValueError,'inference-only'):
                LiDAR4DKITTI360TestDataset(cfg,split='train')

    def test_native_imu_origin_with_calibrated_lidar_and_turning_vehicle(self):
        with TemporaryDirectory() as temporary:
            root=Path(temporary)
            cfg=self._fixture(root)
            # Nonzero sensor extrinsics and a turn exercise both local inputs
            # and relative poses without any dataset-dependent origin shift.
            pose_file=root/'data_poses'/cfg.sequence/'poses.txt'
            values=np.loadtxt(pose_file)
            angle=np.deg2rad(20)
            yaw=np.array([[np.cos(angle),-np.sin(angle),0],
                          [np.sin(angle),np.cos(angle),0],[0,0,1]])
            for row in values:
                if int(row[0])>=2360:
                    matrix=row[1:].reshape(3,4)
                    matrix[:3,:3]=yaw@matrix[:3,:3]
            np.savetxt(pose_file,values,fmt='%.12f')
            extrinsic=np.eye(4)
            extrinsic[:3,3]=[.7,-.3,.8]
            imu_from_cam=EGO_FROM_IMU@extrinsic
            calibration=root/'calibration'/'calib_cam_to_pose.txt'
            calibration.write_text('image_00: '+' '.join(map(str,imu_from_cam[:3].ravel()))+'\n')
            dataset=LiDAR4DKITTI360TestDataset(cfg)
            item=dataset[0]
            np.testing.assert_allclose(dataset.ego_from_lidar,extrinsic)
            sensor=item['input']['lidar_points_sensor'].numpy()
            np.testing.assert_allclose(item['input']['lidar_points_local'][:,:3],
                                       sensor[:,:3]+extrinsic[:3,3],atol=2e-6)
            frames=dataset.windows[0]
            world_lidar,world_local,world_to_ref=dataset._world_transforms(frames)
            expected_local=[dataset.poses.at(frame)[0]@EGO_FROM_IMU for frame in frames]
            np.testing.assert_allclose(world_local,expected_local)
            np.testing.assert_allclose(item['input']['pose'].numpy(),
                                       np.stack([world_to_ref@expected_local[i] for i in [0,2]]),atol=2e-6)
            for sensor,local in zip(world_lidar,world_local):
                np.testing.assert_allclose(local@extrinsic,sensor,atol=1e-8)
            for slot,camera in enumerate(item['gt']['gt_cameras']):
                points=dataset._load_points(frames[slot]).numpy()
                depth,intensity=project_kitti_panorama(points,cfg.image_height,cfg.image_width,cfg.vfov)
                np.testing.assert_array_equal(camera.pts_depth.numpy(),depth)
                np.testing.assert_array_equal(camera.pts_intensity.numpy(),intensity)
                np.testing.assert_allclose(camera.c2w.numpy(),
                                           world_to_ref@world_lidar[slot]@np.linalg.inv(CAMERA_FROM_LIDAR),atol=2e-6)

    def test_calibrated_profile_loader_builds_block_cameras(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            cfg = self._fixture(root)
            cfg.panorama_projection = 'hdl64e_kitti360_calibrated'
            with self.assertRaisesRegex(ValueError, 'image_height=64'):
                LiDAR4DKITTI360TestDataset(cfg)
            cfg.image_height = 64
            cfg.image_width = 2178
            scans = root/'data_3d_raw'/cfg.sequence/'velodyne_points'/'data'
            # The 2.0 m return is inside gt_ego_radius and must not reach the GT.
            points = self._laser_points([3, 40, 63, 63], [0, 90, -120, 60], [15, 9, 3.5, 2.0])
            cfg.gt_ego_radius = 2.5
            for path in scans.glob('*.bin'):
                points.tofile(path)
            dataset = LiDAR4DKITTI360TestDataset(cfg)
            self.assertEqual(dataset.protocol['vertical_projection']['column_assignment'], 'firing_slot')
            item = dataset[0]
            frames = dataset.windows[0]
            world_lidar, _, world_to_ref = dataset._world_transforms(frames)
            for slot, camera in enumerate(item['gt']['gt_cameras']):
                self.assertIsInstance(camera, StackedPanoramaCamera)
                # Each block renders all 64 rows from its own origin and keeps its 32.
                self.assertEqual([c.image_height for c in camera.sub_cameras], [64, 64])
                self.assertEqual(camera.kept_rows, [(0, 32), (32, 64)])
                expected = Camera._build_row_to_theta(64, cfg.vfov, KITTI360_HDL64E_ELEVATION_DEG)
                self.assertTrue(torch.equal(camera.row_to_theta, expected))
                kept_points = dataset._load_points(frames[slot]).numpy()
                kept_points = kept_points[np.linalg.norm(kept_points[:, :3], axis=1) > 2.5]
                depth, intensity, stats = project_hdl64e_panorama(kept_points, 2178)
                np.testing.assert_array_equal(camera.pts_depth.numpy(), depth)
                np.testing.assert_array_equal(camera.pts_intensity.numpy(), intensity)
                self.assertEqual(camera.projection_stats, stats)
                self.assertEqual(stats['gt_pixels'], 3)
                ref_from_lidar = world_to_ref @ world_lidar[slot]
                np.testing.assert_allclose(camera.c2w.numpy(),
                                           ref_from_lidar @ np.linalg.inv(CAMERA_FROM_LIDAR), atol=2e-6)
                for block, origin_z, rows in zip(camera.sub_cameras, KITTI360_HDL64E_BLOCK_ORIGIN_Z_M,
                                                 [slice(0, 32), slice(32, 64)]):
                    # Each block's camera centre is its emitter origin on the sensor z axis.
                    np.testing.assert_allclose(block.camera_center.numpy(),
                                               (ref_from_lidar @ [0, 0, origin_z, 1])[:3], atol=2e-5)
                    np.testing.assert_allclose(camera.row_origin[rows].numpy(),
                                               np.tile([0, -origin_z, 0], (32, 1)), atol=1e-7)

    @staticmethod
    def _guarded_blocks(width, **common):
        """Both blocks render all 64 rows; upper keeps 0-31, lower keeps 32-63."""
        blocks = [Camera(**common, resolution=(width, 64), ring_to_elevation_deg=KITTI360_HDL64E_ELEVATION_DEG)
                  for _ in range(2)]
        return blocks, [(0, 32), (32, 64)]

    def test_stacked_camera_renders_blocks_top_to_bottom(self):
        from unittest import mock
        from src.models.module import m3_g2p
        common = dict(R=np.eye(3), T=np.zeros(3), vfov=(-24.9, 2), hfov=(-180, 180))
        blocks, kept = self._guarded_blocks(8, **common)
        stacked = StackedPanoramaCamera(blocks, np.zeros((64, 3)), kept, **common, resolution=(8, 64),
                                        ring_to_elevation_deg=KITTI360_HDL64E_ELEVATION_DEG)
        calls = []

        def fake_render(viewpoint_camera, **_):
            calls.append(viewpoint_camera)
            rows = torch.arange(viewpoint_camera.image_height, dtype=torch.float32)
            value = 100*len(calls) + rows.view(1, -1, 1)
            return {key: value.expand(channels, -1, 8).clone()
                    for key, channels in [('depth', 1), ('normal', 3), ('raydrop', 1)]}

        renderer = m3_g2p.DynamicGausRender(OmegaConf.create({}))
        with mock.patch.object(m3_g2p, 'render', fake_render):
            package = renderer._render_view(stacked, {'position': torch.zeros(1, 3)}, 0.0)
        self.assertEqual(calls, blocks)
        self.assertEqual(tuple(package['normal'].shape), (3, 64, 8))
        expected = torch.cat([100 + torch.arange(32.), 200 + torch.arange(32., 64.)])
        torch.testing.assert_close(package['depth'][0, :, 0], expected)
        with self.assertRaisesRegex(ValueError, 'row angles'):
            StackedPanoramaCamera(blocks[::-1], np.zeros((64, 3)), kept[::-1], **common, resolution=(8, 64),
                                  ring_to_elevation_deg=KITTI360_HDL64E_ELEVATION_DEG)
        with self.assertRaisesRegex(ValueError, 'add up'):
            StackedPanoramaCamera(blocks, np.zeros((64, 3)), [(0, 32), (0, 48)], **common,
                                  resolution=(8, 64), ring_to_elevation_deg=KITTI360_HDL64E_ELEVATION_DEG)

    @unittest.skipUnless(torch.cuda.is_available(), 'needs the CUDA rasterizer')
    def test_block_render_matches_single_origin_render_when_origins_coincide(self):
        from src.models.module import m3_g2p
        torch.manual_seed(0)
        count = 400
        azimuth = torch.rand(count)*2*np.pi
        distance = 5 + 25*torch.rand(count)
        height = -1.7 + 4*torch.rand(count)
        position = torch.stack([distance*azimuth.cos(), distance*azimuth.sin(), height], 1)
        view = CAMERA_FROM_LIDAR.astype(np.float32)
        common = dict(R=np.eye(3), T=np.zeros(3), vfov=(-24.9, 2), hfov=(-180, 180), viewmatrix=view)
        single = Camera(**common, resolution=(256, 64), ring_to_elevation_deg=KITTI360_HDL64E_ELEVATION_DEG)
        blocks, kept = self._guarded_blocks(256, **common)
        stacked = StackedPanoramaCamera(blocks, np.zeros((64, 3)), kept, **common, resolution=(256, 64),
                                        ring_to_elevation_deg=KITTI360_HDL64E_ELEVATION_DEG)
        device = torch.device('cuda')
        pc = {
            'position': position.to(device),
            'opacity': torch.full((count, 1), 2.0, device=device),
            'scales': torch.log(0.3 + 2*torch.rand(count, 3, device=device)),
            'rotations': torch.nn.functional.normalize(torch.randn(count, 4, device=device), dim=1),
            'shs': torch.randn(count, 16, 4, device=device)*.1,
        }
        pc['scales'][:, 2] = -10
        renderer = m3_g2p.DynamicGausRender(OmegaConf.create({'neg_fov': True, 'sh_degree': 3, 'scale_factor': 1.0}))
        whole = renderer._render_view(single, pc, 0.0)
        split = renderer._render_view(stacked, pc, 0.0)
        self.assertGreater(int((whole['depth'] > 0).sum()), 1000)
        for key in ['depth', 'depth_median', 'intensity_sh', 'raydrop', 'alpha']:
            torch.testing.assert_close(split[key], whole[key], atol=1e-4, rtol=1e-4)

    def test_inference_overlay_preserves_model_and_training_resume_guard(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary)/'checkpoint.ckpt'
            stored,_ = compose_fresh_config(OmegaConf.from_dotlist(['model.variant=dynamic_2dgs_attention_velocity_v14_1']))
            torch.save({'experiment_config_schema_version':1,'experiment_config':OmegaConf.to_container(stored,resolve=True)},path)
            cli = OmegaConf.create({'eval_config':'config/eval/kitti360_inference.yaml','test':{'ckpt_path':str(path)}})
            cfg,source = resolve_eval_config(cli)
            self.assertEqual(cfg.data.dataset_name,'kitti360_lidar4d')
            self.assertEqual(cfg.data.image_height,64)
            self.assertEqual(cfg.data.image_width,2178)
            self.assertEqual(cfg.data.panorama_projection,'hdl64e_kitti360_calibrated')
            self.assertEqual(cfg.data.column_assignment,'firing_slot')
            self.assertIsNone(cfg.data.ring_to_elevation_deg)
            calibrated,_ = resolve_eval_config(OmegaConf.create({
                'eval_config':'config/eval/kitti360_inference_hdl64e_calibrated.yaml','test':{'ckpt_path':str(path)}}))
            self.assertEqual(OmegaConf.to_container(calibrated.data),OmegaConf.to_container(cfg.data))
            for key in ['model','p2g','g2g','g2p','dynamic_2dgs','train']:
                self.assertEqual(OmegaConf.to_container(cfg[key],resolve=True),OmegaConf.to_container(stored[key],resolve=True))
            self.assertIn('inference data overlay',source)
            benchmark, _ = resolve_eval_config(OmegaConf.create({
                'eval_config': 'config/eval/kitti360_inference_lidar4d_benchmark.yaml',
                'test': {'ckpt_path': str(path)},
            }))
            self.assertEqual((benchmark.data.image_height, benchmark.data.image_width), (66, 1030))
            self.assertEqual(benchmark.data.panorama_projection, 'uniform')
            for key in ['model', 'p2g', 'g2g', 'g2p', 'dynamic_2dgs', 'train']:
                self.assertEqual(OmegaConf.to_container(benchmark[key], resolve=True),
                                 OmegaConf.to_container(stored[key], resolve=True))
            with self.assertRaisesRegex(ValueError,'model/training'):
                resolve_eval_config(OmegaConf.merge(cli,{'p2g':{'ego_radius':1.0}}))
            with self.assertRaisesRegex(ValueError,'coordinate_frame'):
                resolve_eval_config(OmegaConf.merge(cli,{'data':{'coordinate_frame':'lidar_sensor'}}))
            with self.assertRaisesRegex(ValueError,'structural'):
                resolve_main_config(OmegaConf.create({'mode':'train','train':{'ckpt_path':str(path)},'data':{'image_height':66}}))


if __name__ == '__main__':
    unittest.main()
