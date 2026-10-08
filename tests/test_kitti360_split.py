from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import random
import subprocess
import sys

import numpy as np

from src.kitti360.metadata import sweep_inventory
from src.kitti360.sampling import eligible_endpoint_frames, sample_window_frames
from src.kitti360.splits import (
    retained_frame_segments, frame_range_mask, partition_frame_segments,
    read_official_3d_ranges, training_window_allowed,
)


class KITTI360SplitTest(unittest.TestCase):
    def test_legacy_module_forwards_the_same_public_functions_and_constants(self):
        from src import kitti360_split as legacy
        from src.kitti360 import metadata, sampling, splits
        expected = {
            metadata: ['KITTI360_BENCHMARK_SCENES', 'KITTI360_BENCHMARK_SEQUENCE',
                       'read_timestamps_ns', 'sweep_inventory'],
            sampling: ['eligible_endpoint_frames', 'sample_window_frames', 'window_start_frames'],
            splits: ['build_training_split', 'frame_range_mask', 'partition_frame_segments',
                     'read_official_3d_ranges', 'retained_frame_segments', 'training_window_allowed'],
        }
        for module, names in expected.items():
            for name in names:
                with self.subTest(name=name):
                    self.assertIs(getattr(legacy, name), getattr(module, name))

    def test_metadata_and_sampling_import_without_datasets_or_torch(self):
        code = '''
import sys
from src.kitti360 import metadata, sampling
assert "torch" not in sys.modules
assert "src.dataloader" not in sys.modules
assert "src.kitti360.splits" not in sys.modules
import src.kitti360_split
assert "torch" not in sys.modules
'''
        subprocess.run([sys.executable, '-c', code], cwd=Path(__file__).resolve().parents[1],
                       check=True, capture_output=True, text=True)

    def test_both_cli_module_paths_accept_the_existing_config_argument(self):
        for module in ['src.kitti360', 'src.kitti360_split']:
            result = subprocess.run([sys.executable, '-m', module, '--help'],
                                    cwd=Path(__file__).resolve().parents[1],
                                    check=True, capture_output=True, text=True)
            self.assertIn('--config', result.stdout)

    def test_val_interval_union_deduplicates_windows_and_benchmark_exclusion_has_priority(self):
        frames = np.arange(20)
        timestamps = frames * 104_500_000
        split = partition_frame_segments(frames, [(5, 9)], [(3, 6), (6, 12)],
                                         timestamps_ns=timestamps, max_sweep_gap_sec=.2)
        self.assertEqual(split['candidate_sweeps_before_val'], 15)
        self.assertEqual(split['official_val_raw_sweeps'], 10)
        self.assertEqual(split['val_benchmark_overlap_sweeps'], 5)
        self.assertEqual(split['val']['num_sweeps'], 5)
        self.assertEqual(split['train']['num_sweeps'], 10)
        self.assertFalse(training_window_allowed(split['val']['segments'], 4, 10))
        self.assertFalse(training_window_allowed(split['train']['segments'], 2, 13))
        self.assertEqual(int(frame_range_mask(frames, [(3, 6), (6, 12)]).sum()), 10)

    def test_official_3d_split_paths_retain_inclusive_raw_frame_ranges(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'val.txt'
            path.write_text('data_3d_semantics/train/2013_05_28_drive_0000_sync/static/0000000372_0000000610.ply\n')
            self.assertEqual(read_official_3d_ranges(path), {'2013_05_28_drive_0000_sync': [(372, 610)]})

    def test_endpoint_duration_uses_actual_timestamps_and_inclusive_limits(self):
        epoch = 1_369_700_000_000_000_000
        timestamps = epoch + np.array([0, 104_500_000, 209_000_000, 940_500_000,
                                       1_000_000_000, 1_045_000_000, 1_200_000_000])
        segment = {'first_frame': 0, 'last_frame': 6}
        candidates = eligible_endpoint_frames(segment, timestamps, 0, endpoint_min_sec=.2, endpoint_max_sec=1.0)
        self.assertEqual(candidates, [2, 3, 4])
        # The retained segment bounds also constrain endpoint selection.
        self.assertEqual(eligible_endpoint_frames({'first_frame': 0, 'last_frame': 2}, timestamps, 0,
                                                 endpoint_min_sec=.2, endpoint_max_sec=1.0), [2])

    def test_short_window_uses_existing_middle_gt_once_and_always_includes_endpoints(self):
        timestamps = np.arange(12, dtype=np.int64) * 104_500_000
        endpoint, middle, used = sample_window_frames(
            {'first_frame': 0, 'last_frame': 2}, timestamps, 0,
            endpoint_min_sec=.2, endpoint_max_sec=1.0, gt_middle_count=3, rng=random.Random(42),
        )
        self.assertEqual((endpoint, middle, used), (2, [1], [0, 1, 2]))
        for seed in range(20):
            endpoint, middle, used = sample_window_frames(
                {'first_frame': 0, 'last_frame': 11}, timestamps, 0,
                endpoint_min_sec=.2, endpoint_max_sec=1.0, gt_middle_count=3, rng=random.Random(seed),
            )
            self.assertGreaterEqual(timestamps[endpoint] / 1e9, .2)
            self.assertLessEqual(timestamps[endpoint] / 1e9, 1.0)
            self.assertEqual(len(middle), min(3, endpoint - 1))
            self.assertEqual(len(used), len(set(used)))
            self.assertEqual(used[0], 0)
            self.assertEqual(used[-1], endpoint)
        with self.assertRaisesRegex(ValueError, 'No available endpoint'):
            sample_window_frames({'first_frame': 0, 'last_frame': 1}, timestamps, 0,
                                 endpoint_min_sec=.2, endpoint_max_sec=1.0, rng=random.Random(42))

    def test_windows_cannot_cross_an_excluded_clip_even_with_retained_endpoints(self):
        segments = retained_frame_segments(range(20), [(5, 9)])
        self.assertTrue(training_window_allowed(segments, 0, 4))
        self.assertTrue(training_window_allowed(segments, 10, 19))
        self.assertFalse(training_window_allowed(segments, 4, 10))
        self.assertFalse(training_window_allowed(segments, 3, 5))
        self.assertFalse(training_window_allowed(segments, 9, 11))
        self.assertFalse(training_window_allowed(segments, 12, 12))
        self.assertEqual(sum(segment['num_sweeps'] for segment in segments), 15)

    def test_missing_scans_and_timestamp_gaps_break_windows_without_deleting_sweeps(self):
        timestamps = np.array([0, 100_000_000, 200_000_000, 1_000_000_000,
                               1_100_000_000, 1_200_000_000, 1_300_000_000])
        segments = retained_frame_segments([0, 1, 2, 3, 4, 6], timestamps_ns=timestamps, max_sweep_gap_sec=.2)
        self.assertEqual(sum(segment['num_sweeps'] for segment in segments), 6)
        self.assertTrue(training_window_allowed(segments, 0, 2))
        self.assertFalse(training_window_allowed(segments, 2, 3))
        self.assertFalse(training_window_allowed(segments, 4, 6))

    def test_original_frame_ids_address_trainval_timestamps_and_test_clips_use_file_order(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'data').mkdir()
            for frame in (4, 5):
                (root / 'data' / f'{frame:010d}.bin').touch()
            timestamps = [f'2013-05-28 12:00:00.{frame * 100_000_000:09d}' for frame in range(6)]
            (root / 'timestamps.txt').write_text('\n'.join(timestamps) + '\n')
            frames, raw_times, stats = sweep_inventory(root)
            self.assertEqual(frames.tolist(), [4, 5])
            self.assertEqual(len(raw_times), 6)
            self.assertAlmostEqual(stats['interval_sec']['mean'], .1)
            (root / 'timestamps.txt').write_text('\n'.join(timestamps[4:]) + '\n')
            _, compact_times, stats = sweep_inventory(root, compact_timestamps=True)
            np.testing.assert_array_equal(compact_times, raw_times[4:])
            self.assertEqual(stats['timestamp_addressing'], 'sorted_files')
            with self.assertRaisesRegex(ValueError, 'timestamp rows'):
                sweep_inventory(root)


if __name__ == '__main__':
    unittest.main()
