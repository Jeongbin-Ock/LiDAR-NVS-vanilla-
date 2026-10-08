# KITTI-360 training

Run from the repository root, in the `2dgs` environment:

```bash
python main.py data_config=kitti360 device='[6]'
```

This selects KITTI-360 data while composing the existing shared training config
and model variant. The current default variant is V11.4. Choose a different
registered V11-or-later variant explicitly, for example:

```bash
python main.py data_config=kitti360 \
  model.variant=dynamic_2dgs_attention_velocity_v16 device='[6]'
```

GPU indices are examples; check shared GPU availability before launching.
Model, loss, optimizer and adaptation settings come from the selected variant
and ordinary CLI overrides. The data profile replaces the entire nuScenes
data block, including its compact-val settings and scene exclusions.

## Two GT modes

Both profiles share every data setting except the GT panorama grid. The
shared settings are the split manifest, sampling, poses, the 2.5 m input and
GT crops, the 80 m range and compact validation.

| `data_config` | GT panorama | Renders per view | GT keeps in-range returns | V15.3 step (bs=1) | Run name |
| --- | --- | ---: | ---: | ---: | --- |
| `kitti360` | calibrated 64×2178, one row per laser, per-emitter-block origins | 2 | ~99% | 2.31 s | `v15.3_kitti` |
| `kitti360_lidar4d` | LiDAR4D/GS-LiDAR uniform 66×1030 from the sensor origin | 1 | ~42% | 1.61 s | `v15.3_kitti_lidar4d` |

- **`kitti360_lidar4d` matches the benchmark grid.** Training sees exactly the
  rows, columns and empty pixels that LiDAR4D/GS-LiDAR evaluation scores.
- **Its rows do not line up with the lasers.** The 66 uniform rows do not
  follow the 64 lasers: 8 rows hold no laser centre and 5 hold two. On the 40
  benchmark targets, row fill ranges from 0.00 to 0.92, and the bottom two
  rows are always empty. Each pixel keeps its nearest return.
- **Evaluate it with the benchmark overlay and the same GT crop:**
  `eval_config=config/eval/kitti360_inference_lidar4d_benchmark.yaml data.gt_ego_radius=2.5`.
- **Checkpoints lock the mode.** The projection and image size are
  checkpoint-protected, so one run cannot switch modes.

The step times are single-GPU means over the same six val windows,
4.3 views on average. The gain is in rendering and the Chamfer loss; the
temporal attention cost does not depend on the GT grid.

A KITTI-360 run's W&B name and `logs/` folder end in its mode suffix: `_kitti`
(calibrated) or `_kitti_lidar4d`. Without `exp_name` the run is named after the
variant label, e.g. `v16_kitti`. A given `exp_name=v15.3` becomes `v15.3_kitti`
or `v15.3_kitti_lidar4d`, and a name that already mentions kitti is kept as
is. A KITTI-360 run therefore never writes into a nuScenes run's
`logs/<exp_name>` checkpoints, and the two modes do not share a folder by
default. The resolved full experiment is saved to
`logs/<exp_name>/effective_config.yaml`.

## Data and temporal contract

- Dataset: `src/dataloader/kitti360/dynamic.py`, registered as `kitti360_dynamic`.
- Raw split: 61,809 train / 14,344 val sweeps, with all ten LiDAR4D clips held
  out from both and no crossing of split or timestamp-gap boundaries.
- Start stride: a 0.5s physical-time grid snapped to the nearest available sweep
  inside each retained segment. Pose eligibility is applied before this grid.
- Pose policy: timestamp-based translation interpolation and rotation SLERP.
  Sweeps outside the first/last optimized-pose bounds are excluded; poses are
  never extrapolated. There are 663 such train sweeps and zero val sweeps.
- Actual dataset index: **12,770 train / 2,996 val windows**.
- Training: resample an endpoint uniformly among available sweeps 0.2–1.0s
  after the start, then sample up to three distinct interior GT sweeps.
- Validation: the same draw is stable per anchor and `eval_seed`.
- Both endpoints are inputs and reconstruction targets. Short windows keep
  fewer middle views; collation and rendering support the ragged GT list.
- `test` uses the existing 40 held-out LiDAR4D target windows and endpoint
  protocol. The dedicated `test_lidar4d_nvs.py` evaluator remains available.

## Compact validation

Per-epoch validation runs on 200 of the 2,996 val anchors
(`config/kitti360_compact_val_windows.json`, `data.compact_val`). Set
`data.compact_val.enable=false` for full val. The groups are:

| Group | Windows | Pool | Meaning |
|---|---:|---:|---|
| `dynamic` | 120 | 617 | Ego moving; a GT-annotated object moving >= 1 m/s, observable (<= 80 m, >= 10 returns) in the start sweep. Four difficulty quartiles x 30. |
| `static` | 60 | 2,102 | Ego moving; no observable annotated object moves >= 0.2 m/s. Ego speed <5 / 5–10 / >=10 m/s, each 5 turning + 15 straight. |
| `stopped` | 20 | 172 | Ego below 0.5 m/s. KITTI-360 has no optimized poses or dynamic boxes there, so these windows carry no motion labels. |

The other 105 anchors are excluded. Their motion is borderline, or their span
touches an interpolated pose bridging > 2 m.

- **Labels describe the anchor, not the drawn window.** Each label is computed
  over the anchor's longest window, start + 1.0 s. Validation keeps each
  anchor's full-val draw: a random endpoint 0.2–1.0 s after the start and
  random middle GT views, stable for a given `eval_seed`.
- **Durations stay unbiased.** Selected windows average 0.565 s, against
  0.578 s for all val. The 95% range for random 200-anchor subsets is
  0.546–0.61 s.
- **Spans never overlap.** No two chosen spans share a frame. The 200
  windows cover all nine drives.
- **Difficulty** mixes percentile ranks of peak speed (0.35), log moving
  returns (0.30), mover count (0.20) and proximity (0.15). It uses speed
  rather than displacement, because displacement grows with window length.

W&B logs the usual `val/<key>` over all 200 windows. It also logs
`val_dynamic/<key>`, `val_static/<key>` and `val_stopped/<key>`, plus each
group's `num_windows`. The group values are means over every rank's windows,
reduced once per epoch, and are also written to
`logs/<exp_name>/eval_val/aggregate_summary.json`. Group metrics need one
window per val batch (`test.batch_size=1`, the default).

The manifest records the settings that decide which anchors exist: the split
manifest hash, stride, gap limit, sweep hop, endpoint range and pose policy.
The dataset refuses a manifest built under other settings. Changing
`eval_seed` or `gt_middle_count` only changes the draw. Rebuild in about 35 s
on CPU:

```bash
python tools/build_kitti360_compact_val.py --csv outputs/kitti360_compact_val/val_anchors.csv
```

## GT maps and rendering

Training and inference share `KITTI360SequenceReader` in
`src/dataloader/kitti360/reader.py`, so calibration, native ego coordinates,
physical timestamps, GT projection and camera transforms follow one code path.

The `kitti360_lidar4d` profile projects each sweep onto the uniform 66×1030 grid
from the sensor origin: one ordinary camera per view and nearest return per pixel.
The default `kitti360` profile generates calibrated **64×2178** HDL-64E panoramas when a
sweep is loaded; no separate range-image preprocessing job is needed. GT depth
is range from the matching emitter-block origin, intensity stays in its native
0–1 range, and raydrop is `1 - (depth > 0)`. Both GT and inputs use the 2.5m
ego crop; GT range is below 80m. Each GT camera contains two emitter-block
cameras, and the existing renderer stacks their retained row blocks and uses
their origins for point reconstruction and Chamfer.

The loader reuses the nuScenes dynamic loader's exact XYZI Utonia collation.
It returns endpoint points, relative ego/sensor poses, physical seconds and
GT cameras in the same contract consumed by `ModelWrapper`.

## Checkpoints and split provenance

Fresh config composition records the split JSON's SHA-256 in
`data.split_manifest_sha256`. Dataset initialization checks it, and a
checkpoint locks the manifest identity, projection and temporal settings.
Changes to the JSON after saving a checkpoint cause a clear mismatch error.

Resume directly from the embedded checkpoint config, without `data_config`:

```bash
python main.py train.ckpt_path=/absolute/path/to/epoch=9.ckpt device='[6]'
```

The official split files, raw counts and annotation-window projection are
documented in [kitti360_train_split.md](kitti360_train_split.md).

## Validation performed

The new loader's full train/val indices matched the counts above. Real CPU
samples generated nonempty 64×2178 depth/intensity maps with 3–5 GT cameras.
Fixture tests verify random training draws, fixed validation draws, pose bounds,
calibrated GT parity, batch collation, benchmark exclusions and checkpoint
protection. GPU unit tests verify stacked calibrated renders, reconstruction
loss and finite gradients with ragged GT views.

No full-model training run was started as part of this preparation. GPU memory
and throughput for the selected model/batch size still need a real training
step; the synthetic renderer unit test does not measure those.
