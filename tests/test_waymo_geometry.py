import numpy as np

from src.dataloader.waymo.geometry import project_calibrated_panorama, top_range_to_sensor_points


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
