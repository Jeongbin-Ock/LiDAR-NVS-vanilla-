"""KITTI-360 compact validation: window identity, sampling contract and group labels.

A compact window is a validation *anchor*, one start sweep of one drive. Its
group is decided over the anchor's longest possible window, from the start to
the last endpoint within ``endpoint_max_sec``. The window actually validated
keeps the full-val draw: a random endpoint and random middle GT views, stable
across epochs for a given ``eval_seed``. Compact validation therefore keeps
random window durations, and the selection does not depend on the draw.

The contract records the settings that decide which anchors exist and how far
they reach. ``eval_seed`` and ``gt_middle_count`` change only the draw, so they
are not part of it. Built by ``tools/build_kitti360_compact_val.py``.
"""
from __future__ import annotations

import json
from pathlib import Path

MANIFEST_VERSION = 1
# dynamic: a GT-annotated mover (>= 1 m/s) observable in the start sweep.
# static: the ego moves and no observable annotated object moves.
# stopped: the ego itself stands still. KITTI-360 has neither optimized poses
# nor dynamic boxes there, so these windows carry no motion labels.
COMPACT_VAL_GROUPS = ("dynamic", "static", "stopped")
CONTRACT_KEYS = (
    "split", "split_manifest_sha256", "window_stride_sec", "max_sweep_gap_sec",
    "sweep_hop", "endpoint_min_sec", "endpoint_max_sec", "pose_missing_policy",
)


def window_key(sequence, start_frame):
    return f"{sequence}|{int(start_frame)}"


def dataset_contract(cfg, split):
    """Anchor-defining settings of a KITTI-360 data config for ``split``."""
    return {
        "split": str(split),
        "split_manifest_sha256": str(cfg.split_manifest_sha256),
        "window_stride_sec": float(cfg.window_stride_sec),
        "max_sweep_gap_sec": float(cfg.max_sweep_gap_sec),
        "sweep_hop": int(cfg.sweep_hop),
        "endpoint_min_sec": float(cfg.dynamic_window.endpoint_min_sec),
        "endpoint_max_sec": float(cfg.dynamic_window.endpoint_max_sec),
        "pose_missing_policy": str(cfg.pose_missing_policy),
    }


def load_manifest(path, contract):
    """Read a compact manifest and refuse one built under another contract."""
    path = Path(str(path))
    if not path.is_file():
        raise FileNotFoundError(f"KITTI360 compact-val manifest not found: {path}")
    manifest = json.loads(path.read_text())
    if int(manifest.get("manifest_version", 0)) != MANIFEST_VERSION:
        raise ValueError(f"{path}: expected KITTI360 compact-val manifest_version {MANIFEST_VERSION}")
    stored = manifest.get("contract", {})
    mismatched = [f"{key}: manifest={stored.get(key)!r} != data={contract[key]!r}"
                  for key in CONTRACT_KEYS if stored.get(key) != contract[key]]
    if mismatched:
        raise ValueError(
            f"{path} was built for another sampling contract; rebuild it with "
            "tools/build_kitti360_compact_val.py:\n  " + "\n  ".join(mismatched))
    windows = manifest.get("windows") or []
    if not windows:
        raise ValueError(f"{path}: manifest contains no windows")
    keys = [entry["key"] for entry in windows]
    if len(keys) != len(set(keys)):
        raise ValueError(f"{path}: duplicate window keys")
    unknown = sorted({entry["label"] for entry in windows} - set(COMPACT_VAL_GROUPS))
    if unknown:
        raise ValueError(f"{path}: unknown compact-val groups {unknown}")
    return manifest


def select_windows(index_keys, manifest):
    """Positions of the manifest's windows in ``index_keys`` order, and their groups."""
    wanted = {entry["key"]: entry["label"] for entry in manifest["windows"]}
    positions, labels = [], []
    for position, key in enumerate(index_keys):
        if key in wanted:
            positions.append(position)
            labels.append(wanted.pop(key))
    if wanted:
        raise ValueError(
            f"{len(wanted)} compact-val window(s) are absent from the dataset index "
            f"(e.g. {sorted(wanted)[0]}); the split or anchor grid changed")
    return positions, labels
