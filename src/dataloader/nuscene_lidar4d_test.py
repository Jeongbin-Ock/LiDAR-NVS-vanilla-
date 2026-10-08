"""LiDAR4D / GS-LiDAR nuScenes test set (Table S6).

These 5 sequences live in **v1.0-trainval** (they need GT range images for
evaluation). Each spans 5.0 s at 0.1 s spacing (51 LiDAR4D "frames"). nuScenes
LIDAR_TOP sweeps are 0.05 s, so one LiDAR4D frame = every 2nd sweep -- verified:
the sequence end timestamp lands exactly on sweep index 100 (= 5.0 s).

Evaluation protocol (1-second temporal NVS, per user spec):
  for each target second T in {1, 2, 3, 4}:
      inputs  = frames at (T - 0.5)s and (T + 0.5)s   (the two endpoints)
      target  = the frame at T s                       (novel temporal view)
This reuses NuScenesNVSDataset's 1.0 s / 0.1 s window machinery: a window
anchored at sweep (T-0.5)s = sweep (T*20 - 10) gives 11 frames at 0.1 s, input =
the 2 endpoints, and the window midpoint = T s.

The ``input_offsets_sec`` constructor arg makes the two input offsets
configurable (default ``(-0.5, 0.5)`` = the protocol above): the window is
resized to span exactly ``[T+before, T+after]`` so ``window_duration_sec`` /
``timestamps_sec`` (derived from the window's own first/last raw timestamps)
stay the true physical gap between whatever input frames are actually used --
e.g. ``(-0.1, 0.2)`` feeds the model 0.9 s / 1.2 s for T=1 s.
"""
from __future__ import annotations

import json
import os

import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes

from .nuscene import (
    NuScenesNVSDataset,
    _cfg_get,
    _resolve_bbox_json_path,
    _sensor_to_world,
    NUSCENES_SWEEP_US,
    MAX_WIN_US,
    MIN_GAP_US,
)
from .nuscenes_frames import (
    COORDINATE_FRAME_EGO,
    ego_to_world,
    resolve_coordinate_frame,
)

# (name, start LIDAR_TOP sweep timestamp in microseconds). The start frame is
# LiDAR4D "frame index 0" of each Table S6 sequence.
LIDAR4D_SEQUENCES = [
    ("seq_0450_0500", 1533151606948845),  # n008-2018-08-01-15-16-36-0400
    ("seq_1250_1300", 1535489300196771),  # n008-2018-08-28-16-43-51-0400 (ego-stationary)
    ("seq_1600_1650", 1535657110100754),  # n008-2018-08-30-15-16-55-0400
    ("seq_2200_2250", 1538448756347324),  # n015-2018-10-02-10-50-40+0800
    ("seq_3180_3230", 1542800849297337),  # n015-2018-11-21-19-38-26+0800
]

# Integer-second targets to evaluate (skip the 0 s / 5 s endpoints).
TARGET_SECONDS = (1, 2, 3, 4)


class LiDAR4DNuScenesTestDataset(NuScenesNVSDataset):
    def __init__(self, cfg, split: str = "test",
                 sequences=LIDAR4D_SEQUENCES, target_seconds=TARGET_SECONDS,
                 use_gt_boxes: bool = False, input_offsets_sec=(-0.5, 0.5)):
        # NOTE: we intentionally do NOT call super().__init__ -- the parent builds
        # an index over a whole nuScenes split. We replicate the small amount of
        # setup it needs and inject our own scene_frames / index.
        self.cfg = cfg
        self.dataroot = cfg.dataroot
        self.split = split
        self.mode = getattr(cfg, "mode", "bbox")
        self.verbose = bool(getattr(cfg, "verbose", False))
        # Table S6 sequences are trainval scenes.
        self.version = _cfg_get(cfg, "version", "v1.0-trainval")

        self.input_offsets_sec = (float(input_offsets_sec[0]), float(input_offsets_sec[1]))
        before_s, after_s = self.input_offsets_sec
        if not (before_s < 0.0 < after_s):
            raise ValueError(
                "input_offsets_sec must straddle the target time (before < 0 "
                f"< after), got {self.input_offsets_sec}"
            )
        # Window spans exactly the two input offsets (default (-0.5, 0.5) =
        # the original fixed 1.0 s protocol) so window_duration_sec /
        # timestamps_sec -- derived below from this window's own first/last
        # raw timestamps -- equal the *actual* input frame gap.
        self.window_us = int(round((after_s - before_s) * 1_000_000))
        self.sample_gap_us = int(_cfg_get(cfg, "sample_gap_us", MIN_GAP_US))
        self.gt_middle_count = 1  # only the target T is evaluated
        self.pair_mode = "sweep"  # this test set uses fixed-anchor sweep windows

        self.sample_hop = max(1, self.sample_gap_us // NUSCENES_SWEEP_US)
        self.window_hop = self.window_us // NUSCENES_SWEEP_US
        self.window_frame_count = self.window_us // self.sample_gap_us + 1
        self.window_sample_hops = [
            i * self.sample_hop for i in range(self.window_frame_count)
        ]

        sweeps_per_sec_grid = 1_000_000 // NUSCENES_SWEEP_US
        before_sweeps = int(round(before_s * sweeps_per_sec_grid))
        target_hop_sweeps = -before_sweeps
        if target_hop_sweeps % self.sample_hop != 0:
            raise ValueError(
                f"input_offsets_sec={self.input_offsets_sec} is not aligned "
                f"to the {self.sample_gap_us / 1e6:g}s sampling grid"
            )
        # window index of the target frame T (offset 0) inside the resized
        # window; used by _select_gt_indices instead of the old V//2 (only
        # valid for symmetric offsets).
        self._target_window_idx = target_hop_sweeps // self.sample_hop

        self.nusc = NuScenes(version=self.version, dataroot=cfg.dataroot, verbose=False)

        # BBox source. DEFAULT = predicted tracking. These 5 sequences span the
        # nuScenes *train* (2) and *val* (3) splits, and the two tracking files
        # use DIFFERENT cadences/keys:
        #   * tracking_train.json : keyframe-only, keyed by keyframe sample_token
        #   * tracking_val.json   : 0.1s (every 2nd sweep), keyed by sweep lidar_token
        # so we load + merge BOTH (sample tokens are globally unique). The earlier
        # 0-box symptom came from loading tracking_{split=test}.json, whose tokens
        # belong to the nuScenes test split and never match these trainval frames.
        self.use_gt_boxes = bool(use_gt_boxes)
        self.bbox_data = None
        if not self.use_gt_boxes:
            self.bbox_data = {}
            for sp in ("train", "val"):
                p = _resolve_bbox_json_path(cfg, sp)
                if p and os.path.exists(p):
                    with open(p) as f:
                        self.bbox_data.update(json.load(f)["results"])
                    if self.verbose:
                        print(f"[lidar4d-test] loaded predicted bbox: {p}")
            if not self.bbox_data:
                self.bbox_data = None
        elif self.verbose:
            print(f"[lidar4d-test] using GT boxes (mode={self.mode})")

        # sweeps per second (1.0 s / 0.05 s = 20) and the before-offset in sweeps.
        sweeps_per_sec = 1_000_000 // NUSCENES_SWEEP_US
        min_anchor = min(target_seconds) * sweeps_per_sec + before_sweeps
        if min_anchor < 0:
            raise ValueError(
                f"input_offsets_sec={self.input_offsets_sec} runs before the "
                f"start of the sequence for target second {min(target_seconds)}"
            )
        max_anchor = max(target_seconds) * sweeps_per_sec + before_sweeps
        need_len = max_anchor + self.window_hop + 1

        start_tokens = self._resolve_start_tokens([ts for _, ts in sequences])

        # nuScenes official train/val split membership of each sequence's scene.
        # Two sequences are in the official train split. Their exposure to the
        # model depends on the training loader's exclusions (V15.3 excludes
        # both whole scenes); official split membership alone is not leakage.
        splits = create_splits_scenes()
        train_set, val_set = set(splits["train"]), set(splits["val"])

        self.scene_frames = []          # List[List[(lidar_tok, sample_tok, ts_us, is_key)]]
        self.index = []                 # List[(seq_idx, anchor_sweep_idx)]
        self.index_meta = []            # List[{"seq_name", "target_s", "scene", "nuscenes_split"}]
        for s_idx, (name, ts_start) in enumerate(sequences):
            start_tok = start_tokens[ts_start]
            frames = self._walk_chain(start_tok, need_len + 4)
            if len(frames) < need_len:
                raise RuntimeError(
                    f"{name}: sweep chain too short ({len(frames)} < {need_len})")
            sd0 = self.nusc.get("sample_data", start_tok)
            scene = self.nusc.get(
                "scene", self.nusc.get("sample", sd0["sample_token"])["scene_token"])
            sname = scene["name"]
            nsplit = ("train" if sname in train_set
                      else "val" if sname in val_set else "other")
            self.scene_frames.append(frames)
            for T in target_seconds:
                anchor = T * sweeps_per_sec + before_sweeps  # window start = (T+before_s)s sweep
                self.index.append((s_idx, anchor))
                self.index_meta.append({"seq_name": name, "target_s": int(T),
                                        "scene": sname, "nuscenes_split": nsplit})

        dynamic_cfg = _cfg_get(cfg, "dynamic_window", None)
        excluded = list(_cfg_get(dynamic_cfg, "train_excluded_scenes",
                                 _cfg_get(cfg, "train_excluded_scenes", ())) or ())
        for (scene_idx, anchor), meta in zip(self.index, self.index_meta):
            window = self._sample_frames(scene_idx, anchor)
            target = window[self._target_window_idx]
            meta.update({
                "input_lidar_tokens": [window[0][0], window[-1][0]],
                "target_lidar_token": target[0],
                "actual_input_offsets_sec": [(int(frame[2]) - int(target[2])) / 1e6
                                              for frame in [window[0], window[-1]]],
                "model_training_scene_excluded": meta["scene"] in excluded,
            })
        spans = [m["actual_input_offsets_sec"][1] - m["actual_input_offsets_sec"][0]
                 for m in self.index_meta]
        self.protocol = {
            "name": "LiDAR4D Table S6 nuScenes fixed target seconds",
            "num_sequences": len(self.scene_frames), "num_windows": len(self.index),
            "target_seconds": list(target_seconds),
            "input_offsets_sec": list(self.input_offsets_sec),
            "timing": "actual nuScenes LIDAR_TOP sweep timestamps",
            "actual_input_span_mean_sec": float(np.mean(spans)),
            "actual_input_span_min_sec": min(spans), "actual_input_span_max_sec": max(spans),
            "height": int(cfg.image_height), "width": int(cfg.image_width),
            "vertical_mapping": "provided hardware ring IDs and configured ring elevations",
            "model_training_scene_exclusions": excluded,
        }

        # Phase-align off-grid frames to the predicted tracking (see method doc).
        if self.bbox_data is not None:
            self._alias_offgrid_frames()

        if self.verbose:
            print(f"[lidar4d-test] {len(self.scene_frames)} sequences, "
                  f"{len(self.index)} windows (version={self.version})")

    def _alias_offgrid_frames(self):
        """val/test predictions are on a 0.1s grid (every 2nd sweep) aligned to
        keyframes; some LiDAR4D sequences start 1 sweep (0.05s) off that grid
        (e.g. seq_1250), so their window frames never coincide with a tracked
        sweep. For each frame whose lidar_token/sample_token is not in the
        tracking data, alias its lidar_token to the boxes of the nearest tracked
        sweep in the same chain (<= 2 sweeps / 0.1s away). Train frames already
        resolve via sample_token (parent keyframe) and are skipped."""
        bd = self.bbox_data
        n_alias = 0
        for frames in self.scene_frames:
            n = len(frames)
            for i, (lid, samp, ts, isk) in enumerate(frames):
                if lid in bd or samp in bd:
                    continue
                hit = None
                for d in (1, 2):
                    for j in (i - d, i + d):
                        if 0 <= j < n and frames[j][0] in bd:
                            hit = frames[j][0]
                            break
                    if hit is not None:
                        break
                if hit is not None:
                    bd[lid] = bd[hit]
                    n_alias += 1
        if self.verbose:
            print(f"[lidar4d-test] phase-aliased {n_alias} off-grid frames")

    # ── target frame inside the selected GT cameras ─────────────────────────
    @property
    def target_cam_index(self) -> int:
        """gt cameras are [input-before, target, input-after]; target = T."""
        return 1

    def _select_gt_indices(self, window):
        # window endpoints (== the two input frames, since the window is
        # sized to span exactly input_offsets_sec) + the target-T frame,
        # which sits at self._target_window_idx (only == V//2 when the
        # offsets are symmetric, e.g. the default (-0.5, 0.5)).
        V = len(window)
        return sorted({0, self._target_window_idx, V - 1})

    def target_boxes_ref(self, window_index: int):
        """GT boxes at the target-second frame, in the window reference frame.

        The collated ``gt["bbox"]`` comes from ``_boxes_in_sensor_frame``, which
        reads the raw *keyframe* ``sample['anns']`` — for a non-keyframe target
        sweep (T rarely lands on a 2 Hz keyframe) a moving object's box is stale
        by up to ~0.5 s (metres of drift). ``nusc.get_boxes`` instead linearly
        interpolates each annotation to the sweep timestamp, so the boxes line
        up with the target-time LiDAR. Returned as ``(M, 7)``
        ``[x,y,z,w,l,h,yaw]`` in the configured frame-0 reference: ego frame
        for new configs, or LiDAR sensor frame for historical checkpoints.
        """
        scene_idx, anchor = self.index[window_index]
        window = self._sample_frames(scene_idx, anchor)
        used = [window[i] for i in self._select_gt_indices(window)]
        ref_token = used[0][0]
        target_token = used[self.target_cam_index][0]

        if resolve_coordinate_frame(self.cfg) == COORDINATE_FRAME_EGO:
            reference_to_world = ego_to_world(self.nusc, ref_token)
        else:
            reference_to_world = _sensor_to_world(self.nusc, ref_token)
        world_to_ref = np.linalg.inv(reference_to_world.astype(np.float64))
        rot, trans = world_to_ref[:3, :3], world_to_ref[:3, 3]
        yaw_offset = float(np.arctan2(rot[1, 0], rot[0, 0]))

        boxes = []
        for box in self.nusc.get_boxes(target_token):   # global frame, interpolated
            cx, cy, cz = (rot @ np.asarray(box.center, np.float64) + trans).tolist()
            w, l, h = (float(v) for v in box.wlh)
            yaw = float(box.orientation.yaw_pitch_roll[0]) + yaw_offset
            boxes.append([cx, cy, cz, w, l, h, yaw])
        return np.asarray(boxes, np.float32).reshape(-1, 7)

    # ── helpers ─────────────────────────────────────────────────────────────
    def _walk_chain(self, start_token: str, n: int):
        frames, cur = [], start_token
        while cur and len(frames) < n:
            sd = self.nusc.get("sample_data", cur)
            frames.append((sd["token"], sd["sample_token"], sd["timestamp"],
                           bool(sd["is_key_frame"])))
            cur = sd["next"]
        return frames

    def _resolve_start_tokens(self, timestamps):
        targets = {str(int(t)): int(t) for t in timestamps}
        found = {}
        for sd in self.nusc.sample_data:
            fn = sd["filename"]
            if "LIDAR_TOP" not in fn:
                continue
            for s, ti in targets.items():
                if s in fn:
                    found[ti] = sd["token"]
        missing = [int(t) for t in timestamps if int(t) not in found]
        if missing:
            raise RuntimeError(
                f"LIDAR_TOP start timestamps not found in {self.version}: {missing}")
        return found
