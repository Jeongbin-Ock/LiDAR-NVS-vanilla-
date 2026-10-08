import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock
import unittest

from omegaconf import OmegaConf
import torch

from src.experiment_config import compose_fresh_config
from src.dataloader.kitti360.dynamic import KITTI360DynamicDataset
from src.eval.group_metrics import GroupMetricAccumulator
from src.dataloader.kitti360 import compact_val
from tests import test_kitti360_training as training_tests


def write_manifest(path, contract, windows):
    path.write_text(json.dumps({"manifest_version": compact_val.MANIFEST_VERSION,
                                "contract": contract, "windows": windows}))
    return str(path)


class CompactManifestTest(unittest.TestCase):
    contract = {"split": "val", "split_manifest_sha256": "abc", "window_stride_sec": 0.5,
                "max_sweep_gap_sec": 0.2, "sweep_hop": 1, "endpoint_min_sec": 0.2,
                "endpoint_max_sec": 1.0, "pose_missing_policy": "interpolate"}

    def test_contract_labels_and_missing_windows_are_checked(self):
        with TemporaryDirectory() as directory:
            windows = [{"key": "drive|5", "label": "dynamic"}, {"key": "drive|1", "label": "stopped"}]
            path = write_manifest(Path(directory) / "m.json", self.contract, windows)
            manifest = compact_val.load_manifest(path, self.contract)
            # Dataset order wins over manifest order.
            self.assertEqual(compact_val.select_windows(["drive|1", "drive|3", "drive|5"], manifest),
                             ([0, 2], ["stopped", "dynamic"]))
            with self.assertRaisesRegex(ValueError, "absent from the dataset index"):
                compact_val.select_windows(["drive|5"], manifest)
            with self.assertRaisesRegex(ValueError, "endpoint_max_sec"):
                compact_val.load_manifest(path, {**self.contract, "endpoint_max_sec": 1.2})
            write_manifest(Path(path), self.contract, [{"key": "drive|1", "label": "moving"}])
            with self.assertRaisesRegex(ValueError, "unknown compact-val groups"):
                compact_val.load_manifest(path, self.contract)
            write_manifest(Path(path), self.contract, windows + windows[:1])
            with self.assertRaisesRegex(ValueError, "duplicate"):
                compact_val.load_manifest(path, self.contract)

    def test_committed_manifest_matches_the_kitti_profile_contract(self):
        cfg, _ = compose_fresh_config({"data_config": "kitti360"})
        self.assertTrue(cfg.data.compact_val.enable)
        manifest = compact_val.load_manifest(cfg.data.compact_val.path,
                                             compact_val.dataset_contract(cfg.data, "val"))
        labels = [entry["label"] for entry in manifest["windows"]]
        self.assertEqual({group: labels.count(group) for group in compact_val.COMPACT_VAL_GROUPS},
                         {"dynamic": 120, "static": 60, "stopped": 20})
        # Selected spans never share a frame within a drive.
        spans = sorted((e["sequence"], e["start_frame"], e["span_end_frame"]) for e in manifest["windows"])
        for (drive_a, _, end_a), (drive_b, start_b, _) in zip(spans, spans[1:]):
            self.assertTrue(drive_a != drive_b or end_a < start_b)
        with self.assertRaisesRegex(ValueError, "requires data.compact_val.path"):
            compose_fresh_config({"data_config": "kitti360", "data": {"compact_val": {"path": ""}}})


class CompactDatasetTest(unittest.TestCase):
    def test_subset_keeps_the_full_val_random_draw_and_carries_groups(self):
        with TemporaryDirectory() as directory:
            cfg, _ = training_tests.KITTI360TrainingTest.fixture(Path(directory))
            full = KITTI360DynamicDataset(cfg, split="val")
            chosen = {full.index[1][2]: "dynamic", full.index[4][2]: "static"}
            sequence = full.index[0][0]
            path = write_manifest(Path(directory) / "compact.json", compact_val.dataset_contract(cfg, "val"),
                                  [{"key": compact_val.window_key(sequence, start), "label": label}
                                   for start, label in chosen.items()])
            compact = KITTI360DynamicDataset(cfg, split="val", window_subset=path)
            self.assertEqual([start for _, _, start in compact.index], sorted(chosen))
            self.assertEqual(compact.window_labels, [chosen[start] for start in sorted(chosen)])
            for position, full_position in enumerate([1, 4]):
                self.assertEqual(compact.sample_window_frames(position), full.sample_window_frames(full_position))
            items = [compact[0], compact[1]]
            self.assertEqual([item["meta"]["window_group"] for item in items], ["dynamic", "static"])
            batch = compact.collate_fn(items)
            self.assertEqual(batch["meta"], {"window_group": ["dynamic", "static"]})
            self.assertNotIn("meta", full.collate_fn([full[0]]))
            with self.assertRaisesRegex(ValueError, "val windows only"):
                KITTI360DynamicDataset(cfg, split="train", window_subset=path)
            cfg.dynamic_window.endpoint_max_sec = 0.8
            with self.assertRaisesRegex(ValueError, "another sampling contract"):
                KITTI360DynamicDataset(cfg, split="val", window_subset=path)


class GroupMetricTest(unittest.TestCase):
    def test_means_are_per_group_and_mixed_batches_are_skipped(self):
        accumulator = GroupMetricAccumulator(("dynamic", "static", "stopped"), {"total", "depth_psnr_raydrop"})
        accumulator.add({"total": torch.tensor(1.0), "depth_psnr_raydrop": torch.tensor(20.0)}, ["dynamic"], 1)
        accumulator.add({"total": torch.tensor(3.0)}, ["dynamic"], 1)
        accumulator.add({"total": torch.tensor(5.0), "depth_psnr_raydrop": torch.tensor(30.0)}, ["static", "static"], 2)
        accumulator.add({"total": torch.tensor(100.0)}, ["static", "dynamic"], 2)
        with self.assertRaisesRegex(ValueError, "Unknown validation window group"):
            accumulator.add({"total": torch.tensor(1.0)}, ["moving"], 1)
        result = accumulator.reduce(torch.device("cpu"))
        self.assertEqual(result, {
            "dynamic": {"num_windows": 2.0, "total": 2.0, "depth_psnr_raydrop": 20.0},
            "static": {"num_windows": 2.0, "total": 5.0, "depth_psnr_raydrop": 30.0},
        })
        self.assertEqual(accumulator.mixed_batches, 1)
        self.assertEqual(GroupMetricAccumulator(("dynamic",), {"total"}).reduce(torch.device("cpu")), {})

    def test_model_wrapper_logs_val_group_means_and_writes_them_to_the_aggregate(self):
        from src.model_wrapper import ModelWrapper

        wrapper = ModelWrapper.__new__(ModelWrapper)
        torch.nn.Module.__init__(wrapper)
        with TemporaryDirectory() as directory:
            wrapper.cfg = OmegaConf.create({"loss": {"w_chamfer": 0.0}, "logger": {"dir": directory}})
            wrapper._metric_interval = 50
            wrapper._eval_pair_summaries = {}
            wrapper._wandb_loss_keys = {"total", "loss_depth"}
            wrapper.p2g_model = mock.Mock(return_value={})
            wrapper.g2g_model = mock.Mock(return_value={})
            wrapper.g2p_model = mock.Mock(return_value={})
            for name in ("_add_dynamic_motion_regularization", "_log_losses", "_log_routing_stats",
                         "_log_utonia_adaptation"):
                setattr(wrapper, name, mock.Mock())
            wrapper.log = mock.Mock()
            wrapper._device = torch.device("cpu")
            wrapper.on_validation_epoch_start()
            for group, total in [("dynamic", 1.0), ("dynamic", 3.0), ("stopped", 7.0)]:
                wrapper.loss = mock.Mock(return_value={"total": torch.tensor(total), "loss_depth": torch.tensor(0.5)})
                batch = {"input": {"timestamps_sec": None, "window_duration_sec": None, "pose": [None]},
                         "gt": {}, "meta": {"window_group": [group]}}
                wrapper._shared_step(batch, 0, prefix="val")
            wrapper.on_validation_epoch_end()
            logged = {call.args[0]: call.args[1] for call in wrapper.log.call_args_list}
            self.assertEqual(logged, {
                "val_dynamic/num_windows": 2.0, "val_dynamic/total": 2.0, "val_dynamic/loss_depth": 0.5,
                "val_stopped/num_windows": 1.0, "val_stopped/total": 7.0, "val_stopped/loss_depth": 0.5,
            })
            self.assertTrue(all(call.kwargs["sync_dist"] is False for call in wrapper.log.call_args_list))
            summary = json.loads((Path(directory) / "eval_val" / "aggregate_summary.json").read_text())
            self.assertEqual(summary["group_aggregate"]["dynamic"]["total"], 2.0)


if __name__ == "__main__":
    unittest.main()
