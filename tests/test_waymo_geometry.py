import numpy as np
import torch
from scipy.spatial.transform import Rotation

from src.dataloader.waymo.geometry import (
    panorama_from_lidar_yaw, project_calibrated_panorama, top_range_to_sensor_points,
)


def test_waymo_pixel_pose_deskews_to_frame_lidar():
    ranges = np.array([[2.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    intensity = np.array([[0.25, 0.0, 0.0, 0.0]], dtype=np.float32)
    pose = np.zeros((1, 4, 6), dtype=np.float32)
    pose[0, 0, 3] = 1.0
    points = top_range_to_sensor_points(ranges, intensity, pose, np.array([0.0]),
                                        np.eye(4), np.eye(4))
    expected = np.array([[1.0 - np.sqrt(2.0), np.sqrt(2.0), 0.0, 0.25]])
    np.testing.assert_allclose(points, expected, atol=1e-6)


def test_calibrated_panorama_uses_nearest_beam_and_nearest_return():
    points = np.array([[3.0, 0.0, 0.3, 0.2], [5.0, 0.0, 0.5, 0.9],
                       [3.0, 0.0, -0.3, 0.7]], dtype=np.float32)
    depth, intensity = project_calibrated_panorama(points, [0.1, -0.1], width=4, max_depth=10)
    assert np.count_nonzero(depth) == 2
    np.testing.assert_allclose(depth[0, :, 2], np.linalg.norm(points[[0, 2], :3], axis=1))
    np.testing.assert_allclose(intensity[0, :, 2], [0.2, 0.7])


def _pose(rpy, translation):
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
    matrix[:3, 3] = translation
    return matrix


def test_waymo_deskew_matches_official_vehicle_frame_chain_with_rotations():
    """The official Waymo chain is pixel vehicle -> world -> frame vehicle."""
    rng = np.random.default_rng(20261027)
    height, width = 3, 8
    ranges = rng.uniform(4, 40, size=(height, width)).astype(np.float32)
    ranges[0, 3] = 0
    intensity = rng.uniform(0, 1, size=(height, width)).astype(np.float32)
    pixel_pose = rng.normal(0, 0.1, size=(height, width, 6)).astype(np.float32)
    pixel_pose[..., 3:] += np.array([24.0, -35.0, 1.5])
    inclinations = np.array([0.17, 0.01, -0.23], dtype=np.float32)
    vehicle_from_lidar = _pose([0.04, -0.03, 0.29], [1.4, -0.2, 1.9])
    world_from_vehicle = _pose([-0.05, 0.02, -0.4], [24.0, -35.0, 1.5])
    points_lidar = top_range_to_sensor_points(
        ranges, intensity, pixel_pose, inclinations, vehicle_from_lidar, world_from_vehicle,
    )
    actual_vehicle = (vehicle_from_lidar[:3, :3] @ points_lidar[:, :3].T).T + vehicle_from_lidar[:3, 3]
    expected_vehicle = []
    azimuth_correction = np.arctan2(vehicle_from_lidar[1, 0], vehicle_from_lidar[0, 0])
    for row, column in zip(*np.nonzero(ranges > 0)):
        azimuth = ((width - column - 0.5) / width * 2 - 1) * np.pi - azimuth_correction
        inclination, distance = inclinations[row], ranges[row, column]
        sensor = np.array([distance * np.cos(inclination) * np.cos(azimuth),
                           distance * np.cos(inclination) * np.sin(azimuth),
                           distance * np.sin(inclination), 1.0])
        pixel = pixel_pose[row, column]
        world_from_pixel_vehicle = _pose(pixel[:3], pixel[3:])
        expected_vehicle.append((np.linalg.inv(world_from_vehicle) @ world_from_pixel_vehicle
                                 @ vehicle_from_lidar @ sensor)[:3])
    np.testing.assert_allclose(actual_vehicle, expected_vehicle, atol=1e-5, rtol=1e-5)


def test_waymo_cardinal_directions_match_renderer_panorama_axes():
    from src.dataloader.kitti360.reader import CAMERA_FROM_LIDAR
    from src.models.utils.graphics_utils import pano_to_lidar

    # Sensor +x/+y/-x/-y point to the middle/left/seam/right image columns.
    sensor = np.array([[10, 0, 0, 0.1], [0, 11, 0, 0.2], [-12, 0, 0, 0.3],
                       [0, -13, 0, 0.4]], dtype=np.float32)
    depth, intensity = project_calibrated_panorama(sensor, [0.0], width=8, max_depth=80)
    for i, column in enumerate([4, 2, 0, 6]):
        np.testing.assert_allclose(depth[0, 0, column], np.linalg.norm(sensor[i, :3]))
        np.testing.assert_allclose(intensity[0, 0, column], sensor[i, 3])
    camera_points = pano_to_lidar(torch.from_numpy(depth), (-1, 1), (-180, 180),
                                 row_to_theta=[np.pi / 2]).numpy()
    lidar_points = (CAMERA_FROM_LIDAR[:3, :3].T @ camera_points.T).T
    actual = sorted(tuple(np.round(point, 4)) for point in lidar_points)
    expected = sorted(tuple(point[:3]) for point in sensor)
    np.testing.assert_allclose(actual, expected, atol=1e-5)


def test_waymo_native_forward_center_keeps_extrinsic_yaw_alignment():
    """A rotated TOP sensor must still put Waymo's native forward ray at W/2."""
    width = 2650
    extrinsic = _pose([0.0, 0.0, np.deg2rad(148.5)], [1.43, 0.0, 2.184])
    ranges = np.zeros((1, width), dtype=np.float32)
    ranges[0, width // 2] = 20.0
    intensity = (ranges > 0).astype(np.float32)
    points = top_range_to_sensor_points(ranges, intensity, np.zeros((1, width, 6), dtype=np.float32),
                                        np.array([0.0]), extrinsic, np.eye(4))
    alignment = panorama_from_lidar_yaw(extrinsic)
    aligned = points.copy()
    aligned[:, :3] = points[:, :3] @ alignment[:3, :3].T
    depth, _ = project_calibrated_panorama(aligned, [0.0], width=1024, max_depth=80)
    assert depth[0, 0, 512] > 0
    old_depth, _ = project_calibrated_panorama(points, [0.0], width=1024, max_depth=80)
    assert np.flatnonzero(old_depth)[0] != 512


def test_waymo_ego_and_sensor_poses_map_static_point_to_target_camera():
    from omegaconf import OmegaConf
    from src.dataloader.kitti360.reader import CAMERA_FROM_LIDAR
    from src.dataloader.waymo.dynamic import WaymoDynamicDataset, WaymoSequenceReader

    cfg = OmegaConf.create({"ego_radius": 0.0, "gt_ego_radius": 0.0, "image_height": 64,
                            "image_width": 16, "vfov": [-17.6, 2.4], "hfov": [-180, 180],
                            "max_depth_m": 80.0})
    reader = WaymoSequenceReader.__new__(WaymoSequenceReader)
    reader.cfg = cfg
    reader.vehicle_from_lidar = _pose([0.01, -0.02, 0.3], [1.4, -0.3, 1.9])
    reader.world_from_vehicle = np.stack([
        _pose([0.01, 0.02, 0.4], [25, 75, 1]),
        _pose([0.02, 0.03, 0.5], [26, 75.3, 1.02]),
        _pose([0.03, 0.02, 0.6], [27, 75.8, 1.04]),
    ])
    reader.timestamps_us = np.array([1_000_000, 1_100_000, 1_200_000])
    reader.inclination_rad = np.linspace(0.2, -0.2, 64)
    alignment = panorama_from_lidar_yaw(reader.vehicle_from_lidar)
    forward = alignment.T @ np.array([10.0, 0.0, 0.0, 1.0])
    point_at_target = np.array([forward[0], forward[1], 10.0 * np.tan(reader.inclination_rad[32]), 1.0])
    world_from_target_lidar = reader.world_from_vehicle[1] @ reader.vehicle_from_lidar
    point_world = world_from_target_lidar @ point_at_target

    def points_at(index):
        sensor = np.linalg.inv(reader.world_from_vehicle[index] @ reader.vehicle_from_lidar) @ point_world
        return torch.tensor([[*sensor[:3], 0.5]], dtype=torch.float32)

    reader.points = points_at
    dataset = WaymoDynamicDataset.__new__(WaymoDynamicDataset)
    dataset.cfg, dataset.is_train, dataset.fixed_interpolation_val = cfg, False, True
    dataset.index, dataset.readers = [("synthetic", 0, 2, 0)], {"synthetic": reader}
    sample = dataset[0]
    source = sample["input"]
    point_ref = np.linalg.inv(reader.world_from_vehicle[0]) @ point_world
    for i, frame in enumerate((0, 2)):
        ego = np.r_[source["lidar_points_ego"][i, :3].numpy(), 1.0]
        sensor = np.r_[source["lidar_points_sensor"][i, :3].numpy(), 1.0]
        np.testing.assert_allclose(source["pose"][i].numpy() @ ego, point_ref, atol=1e-5)
        np.testing.assert_allclose(source["pose_lidar_to_ref"][i].numpy() @ sensor, point_ref, atol=1e-5)
    target_camera = sample["gt"]["cameras"][0]
    actual_camera = target_camera.world_view_transform.T.numpy() @ point_ref
    np.testing.assert_allclose(actual_camera, CAMERA_FROM_LIDAR @ alignment @ point_at_target, atol=1e-5)
    assert target_camera.pts_depth[0, 32, 8] > 0
