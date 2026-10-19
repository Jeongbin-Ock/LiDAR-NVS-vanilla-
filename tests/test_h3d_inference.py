from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation

from src.dataloader.h3d.metadata import (
    H3DScenario, gps_frame_time_us, h3d_scan, pose_from_xyzrpy, read_h3d_ply, read_labels,
)
from src.dataloader.lidar4d_test import build_lidar4d_test_dataset, canonical_lidar4d_dataset_name

FIELDS = [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4"),
          ("red", "u1"), ("green", "u1"), ("blue", "u1"), ("alpha", "u1"),
          ("radius", "<f4"), ("confidence", "<f4"), ("curvature", "<f4")]
HEADER = ("ply\nformat binary_little_endian 1.0\ncomment PCL generated\nelement vertex {n}\n"
          "property float x\nproperty float y\nproperty float z\nproperty float nx\nproperty float ny\n"
          "property float nz\nproperty uchar red\nproperty uchar green\nproperty uchar blue\n"
          "property uchar alpha\nproperty float radius\nproperty float confidence\n"
          "property float curvature\nend_header\n")


def write_ply(path, xyz, radius, laser):
    vertices = np.zeros(len(xyz), dtype=np.dtype(FIELDS))
    vertices["x"], vertices["y"], vertices["z"] = np.asarray(xyz, dtype=np.float32).T
    vertices["radius"], vertices["confidence"] = radius, laser
    Path(path).write_bytes(HEADER.format(n=len(xyz)).encode() + vertices.tobytes())


# World points seen by every synthetic sweep, plus one no-return placeholder.
WORLD = np.array([[20.0, 5.0, -1.0], [30.0, -4.0, 0.5], [15.0, 0.0, -1.7], [-12.0, 3.0, 0.0]])


def make_scenario(root, name, frames, start_sec=1_500_000_000.0, period=0.1):
    directory = Path(root) / name
    directory.mkdir(parents=True)
    for frame in range(frames):
        time = start_sec + frame * period
        rows = [f"{time + 0.01 * k:.6f},0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0" for k in range(10)]
        (directory / f"gps_{frame:03d}.csv").write_text("# timestamp,Long_Rel\n" + "\n".join(rows) + "\n")
        pose = (0.5 * frame, 0.02 * frame, 0.0, 0.0, 0.0, 0.01 * frame)
        (directory / f"odom_{frame:03d}.txt").write_text(" ".join(map(str, pose)))
        lidar_from_world = np.linalg.inv(pose_from_xyzrpy(pose))
        local = WORLD @ lidar_from_world[:3, :3].T + lidar_from_world[:3, 3]
        xyz = np.vstack([local, [[1.2, 0.0, 0.0]]])
        write_ply(directory / f"pointcloud1_{frame:03d}.ply", xyz, [64, 128, 200, 254, 1], [10, 20, 30, 40, 50])
        (directory / f"labels_3d1_{frame:03d}.txt").write_text(
            "car,abc,dynamic,10.0,0.0,-1.0,2.0,4.5,1.6,-1.5707963\n")


class H3DInferenceTest(unittest.TestCase):
    def test_pose_is_rz_ry_rx(self):
        values = (1.0, -2.0, 0.5, 0.1, -0.2, 0.7)
        pose = pose_from_xyzrpy(values)
        expected = Rotation.from_euler("ZYX", [0.7, -0.2, 0.1]).as_matrix()
        np.testing.assert_allclose(pose[:3, :3], expected, atol=1e-12)
        np.testing.assert_allclose(pose[:3, 3], values[:3])

    def test_ply_fields_returns_and_intensity(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "pointcloud1_000.ply"
            write_ply(path, [[10, 0, 0], [1.2, 0, 0], [0, 3, 0]], [255, 1, 128], [63, 0, 5])
            vertices = read_h3d_ply(path)
            self.assertEqual(len(vertices), 3)
            points, lasers = h3d_scan(path, intensity_divisor=256.0)
            np.testing.assert_allclose(points[:, 3], [255 / 256, 128 / 256])
            np.testing.assert_array_equal(lasers, [63, 5])

    def test_gps_midpoint_and_labels(self):
        with TemporaryDirectory() as tmp:
            make_scenario(tmp, "scenario_900", 2)
            scenario = Path(tmp) / "scenario_900"
            self.assertEqual(gps_frame_time_us(scenario / "gps_001.csv"), round((1_500_000_000.1 + 0.045) * 1e6))
            row = read_labels(scenario / "labels_3d1_000.txt")[0]
            self.assertEqual(row[:3], ("car", "abc", "dynamic"))
            self.assertEqual(len(row), 10)

    def _config(self, root):
        cfg = OmegaConf.load("config/eval/h3d_inference_lidar4d.yaml").data
        cfg.dataroot = root
        cfg.verbose = False
        return cfg

    def test_windows_targets_per_second_and_duration_filter(self):
        with TemporaryDirectory() as tmp:
            make_scenario(tmp, "scenario_900", 61)   # 6.0 s span
            make_scenario(tmp, "scenario_901", 44)   # 4.3 s span: skipped
            split = Path(tmp) / "train_test_split"
            split.mkdir()
            (split / "detection_val.txt").write_text("scenario_900\nscenario_901")
            (split / "detection_test.txt").write_text("")
            cfg = self._config(tmp)
            self.assertEqual(canonical_lidar4d_dataset_name(cfg), "h3d")
            dataset = build_lidar4d_test_dataset(cfg, input_offsets_sec=(-0.1, 0.1))
            # T=6 has no sweep after it, so the last target is T=5.
            self.assertEqual([m["target_second"] for m in dataset.index_meta], [1, 2, 3, 4, 5])
            self.assertEqual([m["target_frame_id"] for m in dataset.index_meta], [10, 20, 30, 40, 50])
            self.assertEqual(dataset.index_meta[0]["input_frame_ids"], [9, 11])
            self.assertEqual([row["scenario"] for row in dataset.skipped_short], ["scenario_901"])
            np.testing.assert_allclose(dataset.index_meta[2]["actual_input_offsets_sec"], [-0.1, 0.1], atol=1e-6)

            item = dataset[1]
            inp, gt = item["input"], item["gt"]
            counts = inp["frame_counts"].tolist()
            self.assertEqual(counts, [4, 4])  # placeholder < 1.5 m dropped, 2.5 m crop keeps the rest
            reference = inp["lidar_points"][:, :3].numpy()
            # One static world point maps to the same reference position from both sweeps.
            np.testing.assert_allclose(reference[:4], reference[4:], atol=1e-4)
            ego_from_lidar = np.asarray(cfg.ego_from_lidar, dtype=np.float64)
            sensor = inp["lidar_points_sensor"][:4, :3].numpy()
            np.testing.assert_allclose(inp["lidar_points_ego"][:4, :3].numpy(),
                                       sensor @ ego_from_lidar[:3, :3].T + ego_from_lidar[:3, 3], atol=1e-5)
            np.testing.assert_allclose(inp["pose"][0].numpy(), np.eye(4), atol=1e-6)
            np.testing.assert_allclose(inp["timestamps_sec"].numpy(), [0.0, 0.2], atol=1e-6)
            self.assertAlmostEqual(float(gt["window_duration_sec"]), 0.2, places=6)
            np.testing.assert_allclose(inp["lidar_points"][:, 3].numpy(), np.tile([64, 128, 200, 254], 2) / 256)
            target_camera = gt["gt_cameras"][1]
            self.assertEqual(int((target_camera.pts_depth > 0).sum()), 4)

            boxes = dataset.target_boxes_ref(0)
            self.assertEqual(boxes.shape, (1, 7))
            np.testing.assert_allclose(boxes[0, 3:6], [2.0, 4.5, 1.6])

    def test_overlay_only_sets_inference_roots(self):
        overlay = OmegaConf.load("config/eval/h3d_inference_lidar4d.yaml")
        allowed = {"mode", "data", "test", "device", "seed", "out_dir", "limit", "depth_statistic"}
        self.assertLessEqual(set(overlay), allowed)
        matrix = np.asarray(overlay.data.ego_from_lidar, dtype=np.float64)
        np.testing.assert_allclose(matrix[:3, :3] @ matrix[:3, :3].T, np.eye(3), atol=1e-8)


if __name__ == "__main__":
    unittest.main()
