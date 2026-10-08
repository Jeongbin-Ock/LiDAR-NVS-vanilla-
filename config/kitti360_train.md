# KITTI-360 training

Run from the repository root, in the `2dgs` environment:

```bash
python main.py data_config=config/kitti360_train.yaml device='[6]'
```

This selects KITTI-360 data while composing the existing shared training config
and model variant. The current default variant is V11.4. Choose a different
registered V11-or-later variant explicitly, for example:

```bash
python main.py data_config=config/kitti360_train.yaml \
  model.variant=dynamic_2dgs_attention_velocity_v16 device='[6]'
```

GPU indices are examples; check shared GPU availability before launching.
Model, loss, optimizer and adaptation settings come from the selected variant
and ordinary CLI overrides. The data profile replaces the entire nuScenes
data block, including its compact-val settings and scene exclusions.

A KITTI-360 run's W&B name and `logs/` folder end in `_kitti`. Without
`exp_name` the run is named after the variant label, e.g. `v16_kitti`; a given
`exp_name=v15.3` becomes `v15.3_kitti`, and a name that already mentions kitti
is kept as is. A KITTI-360 run therefore never writes into a nuScenes run's
`logs/<exp_name>` checkpoints. The resolved full experiment is saved to
`logs/<exp_name>/effective_config.yaml`.

## Data and temporal contract

- Dataset: `src/dataloader/kitti360_dynamic.py`, registered as `kitti360_dynamic`.
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

## GT maps and rendering

Training and inference share `KITTI360SequenceReader` in
`src/dataloader/kitti360_lidar4d_test.py`, so calibration, native ego coordinates,
physical timestamps, GT projection and camera transforms follow one code path.

The default profile generates calibrated **64×2178** HDL-64E panoramas when a
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
