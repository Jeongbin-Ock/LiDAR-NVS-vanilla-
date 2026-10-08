# KITTI-360 training split preparation

Measured on 2026-10-08 from `/data1/kitti360/` scan filenames and nanosecond
timestamps. The YAML and generated JSON prepare the train/validation split and
random-duration sampling contract. The registered `kitti360_dynamic` training
loader consumes this manifest with a 0.5s starting-anchor grid. The runnable
data profile is `kitti360_train.yaml`; this split-preparation YAML is for
rebuilding the manifest. See [kitti360_train.md](kitti360_train.md).

## LiDAR4D overlap and exclusion

All ten LiDAR4D benchmark clips belong to `2013_05_28_drive_0000_sync`, which
is part of the available Train/Val raw pool. Per the requested split, exclude
the complete clips below and keep the rest of drive 0000. Bounds are inclusive.
Under the official 3D semantic split, the four static clips lie in validation
and the six dynamic clips lie in training. All ten are held out for our test.

| Scene | Category | Excluded frame IDs | Sweeps |
|---|---|---|---:|
| 1538 | static | 1538–1601 | 64 |
| 1728 | static | 1728–1791 | 64 |
| 1908 | static | 1908–1971 | 64 |
| 3353 | static | 3353–3416 | 64 |
| 2350 | dynamic | 2350–2400 | 51 |
| 4950 | dynamic | 4950–5000 | 51 |
| 8120 | dynamic | 8120–8170 | 51 |
| 10200 | dynamic | 10200–10250 | 51 |
| 10750 | dynamic | 10750–10800 | 51 |
| 11400 | dynamic | 11400–11450 | 51 |
| **Total** | | | **562** |

With the current inference offsets `[-0.5, +0.5]`, the evaluator uses 40
held-out GT targets and 62 distinct endpoint sweeps: 102 distinct sweeps,
all inside the excluded clips. Repeated input frames account for the difference
between distinct sweeps and the 120 frame appearances across 40 windows.

LiDAR4D's own scene-fitting protocol trains on the other 522 frames in these
clips and holds out 40 targets. That per-clip train/test split is distinct from
our feed-forward training candidate pool. See the
[official split implementation](https://github.com/ispc-lab/LiDAR4D/blob/main/data/preprocess/kitti360_to_nerf.py).

The manifest's `train_segments` are the allowed contiguous frame ranges. A
training window must be wholly inside one segment; checking only that its
endpoints survived exclusion would allow it to jump across a test clip.
`src/kitti360/splits.py:training_window_allowed` implements this rule for the
training loader. The measured held-out/train intersection is zero.

## Sweep counts and timing

Train/Val timestamps use original frame IDs, rather than the sorted-file index.
In particular, drive 0002 starts at frame 4391, while its timestamps have
19,240 rows starting at frame zero. Separate Test SLAM clips use compact
timestamps matched to their sorted scan files.

| Drive | Raw sweeps | Retained candidates | Mean interval, s | Median interval, s | Mean saved Hz |
|---|---:|---:|---:|---:|---:|
| 0000 | 11,518 | 10,956 | 0.104480349 | 0.104495253 | 9.571178 |
| 0002 | 14,849 | 14,849 | 0.104619641 | 0.104610109 | 9.558435 |
| 0003 | 1,031 | 1,031 | 0.104625915 | 0.104612714 | 9.557861 |
| 0004 | 11,587 | 11,587 | 0.104512956 | 0.104513480 | 9.568192 |
| 0005 | 6,743 | 6,743 | 0.104552631 | 0.104545088 | 9.564561 |
| 0006 | 9,699 | 9,699 | 0.104507126 | 0.104503549 | 9.568725 |
| 0007 | 3,396 | 3,396 | 0.104455970 | 0.104445919 | 9.573412 |
| 0009 | 14,056 | 14,056 | 0.104558353 | 0.104231666 | 9.564037 |
| 0010 | 3,836 | 3,836 | 0.104192924 | 0.104203121 | 9.597581 |
| **Total** | **76,715** | **76,153** | | | |

The retained-candidate column is before subtracting the official validation
intervals, checking pose eligibility or constructing training windows. Optimized poses are sparse;
the existing inference loader supports timestamp-based interpolation. Adjacent
scans normally arrive about 0.1045s apart, rather than exactly 0.1s.

Drive 0009 has seven adjacent-scan intervals longer than 0.2s, with a maximum
of 1.458871731s (frames 8848→8849). The preparation config cuts segments at
gaps longer than 0.2s, without deleting scans. Use physical timestamps for
window duration, conditioning and motion transport.

There are also 8,340 separate Test SLAM files. Overall there are **85,055
`.bin` files**, but **79,195 unique `(drive, frame)` sweeps**: test_2/test_3
repeat 5,860 scans from drives 0004/0002. The four Test SLAM clip directories
are inventory only and never enter the training candidate pool.

## Official 3D validation intervals mapped to raw sweeps

The official semantic archive contains 239 train and 61 validation accumulated
point-cloud windows. Those are approximately 200m annotation regions, separate
from our 0.2–1.0s training windows. Their inclusive frame bounds are projected
onto existing `.bin` frame IDs. Overlapping annotation windows count only once.
See the [official 3D benchmark documentation](https://github.com/autonomousvision/kitti360Scripts/blob/master/kitti360scripts/evaluation/semantic_3d/README.md).

| Drive | Official val windows | Raw val sweeps | Already removed for LiDAR4D test | Remaining val sweeps | Remaining train sweeps |
|---|---:|---:|---:|---:|---:|
| 0000 | 10 | 2,559 | 256 | 2,303 | 8,653 |
| 0002 | 9 | 2,227 | 0 | 2,227 | 12,622 |
| 0003 | 1 | 281 | 0 | 281 | 750 |
| 0004 | 6 | 1,639 | 0 | 1,639 | 9,948 |
| 0005 | 6 | 1,537 | 0 | 1,537 | 5,206 |
| 0006 | 8 | 2,107 | 0 | 2,107 | 7,592 |
| 0007 | 6 | 546 | 0 | 546 | 2,850 |
| 0009 | 11 | 2,655 | 0 | 2,655 | 11,401 |
| 0010 | 4 | 1,049 | 0 | 1,049 | 2,787 |
| **Total** | **61** | **14,600** | **256** | **14,344** | **61,809** |

Thus **76,153 − 14,344 = 61,809** training sweeps. The 256 already excluded
validation sweeps are the four static LiDAR4D clips. Train and val frame sets
are disjoint, and neither includes a benchmark sweep. Windows must stay within
one segment of their respective split.

This is our raw NVS split derived from the official **3D semantic validation
regions**, rather than a claim that KITTI-360 provides an NVS split for every
raw scan. The semantic train and val window ranges overlap at some boundaries;
validation takes priority there. The raw pool also contains 7,820 sweeps outside
both semantic-window lists. We retain those for bbox-free NVS training, which
does not require semantic labels. Strictly restricting training to listed
semantic train regions as well would leave 53,989 sweeps instead.

Only the ZIP directory and two small official split files were fetched
(103,736 bytes total), rather than the 11.9GB accumulated-cloud archive. Exact
files, source URLs and SHA-256 digests are in `kitti360_official_splits/`.

## Random-duration windows and GT sweeps

The selected physical duration range is **0.2–1.0s**. Every training access
draws uniformly from available endpoint sweeps inside that range, matching
the nuScenes loader's discrete endpoint sampling. It does not invent an exact
0.2s or 1.0s frame when no sweep exists at that timestamp.

Both endpoints are model inputs and reconstruction GT. Choose up to three
distinct interior sweeps without replacement for novel-view GT, sort the whole
target list by time, and keep fewer GT views when a short window has fewer than
three interior sweeps. No endpoint clamping or GT duplication is allowed.
`sweep_hop: 1` keeps the approximately 0.1045s saved sweep grid. Training uses
the worker-seeded RNG; validation uses a fixed per-anchor RNG with `eval_seed`.
`eligible_endpoint_frames` and `sample_window_frames` in `src/kitti360/sampling.py`
implement the duration and GT selection used by `KITTI360DynamicDataset`.

## Starting-anchor stride

Measured across the 76,153 pre-validation candidates, excluding windows crossing
benchmark clips or timestamp-gap boundaries:

| Stride, sweeps | Mean start interval, s | Median start interval, s | 1st–99th percentile, s |
|---:|---:|---:|---|
| 1 | 0.104460 | 0.104497 | 0.103964–0.104880 |
| 2 | 0.208921 | 0.208997 | 0.207925–0.209759 |
| 5 | 0.522301 | 0.522501 | 0.519824–0.524361 |
| 10 | 1.044603 | 1.045013 | 1.039712–1.048621 |

**Recommended initial stride: 5 sweeps**, to retain approximately the existing
nuScenes loader's 0.5s start cadence. This gives roughly one fifth as many
anchors as stride 1. This is a sampling recommendation, not a measured optimum
for training quality. It moves window starts only: interior sweep availability
and the endpoint-duration range are separate settings. A ten-frame interval
lasts about 1.045s, so a requested physical 1.0s duration must use timestamps.
The selected runtime setting is `window_stride_sec: 0.5`, snapped to the
nearest actual sweep. After pose eligibility is applied before constructing
the per-segment grids, this gives **12,770 train / 2,996 val windows**. Physical
0.5s and five saved sweeps (about 0.522s) are separate sampling policies.

## Rebuild and validation

```bash
/data1/jeongbin/miniconda3/envs/2dgs/bin/python -m src.kitti360 \
  --config config/kitti360_train_split.yaml
```

The version-2 manifest records separate allowed train and val segments, exact
benchmark windows, official split-file hashes and raw inventory statistics.
The earlier live inference metadata check matched all 40 benchmark windows.
Seven focused CPU tests now cover split boundaries, overlapping val intervals,
timestamp addressing, physical endpoint limits and short-window GT sampling.
The earlier 13 inference CPU tests also passed; two renderer tests were excluded.

The lightweight data utilities are organized in `src/kitti360/`: raw metadata,
runtime sampling, split assembly and the CLI each have their own module. The
old `src.kitti360_split` imports and CLI remain supported by a compatibility
facade. Refactoring these modules does not require regenerating the manifest.

## Test SLAM

Yes: this is the official **semantic localization and mapping** test set.
It consists of four short clips, with 933, 1,547, 2,848 and 3,012 scans. The
benchmark evaluates pose trajectories, reconstructed geometry and semantic maps.
Slow-moving frames are filtered, so IDs can be discontinuous. See the
[official Semantic SLAM README](https://github.com/autonomousvision/kitti360Scripts/blob/master/kitti360scripts/evaluation/semantic_slam/README.md).

These clips are a separate protocol from LiDAR4D's held-out NVS targets.
`test_0`/`test_1` use drive 0008; `test_2`/`test_3` are copies of some scans from
the original drive 0004/0002 raw pool. We do not add the Test SLAM directories as
extra NVS training data.
