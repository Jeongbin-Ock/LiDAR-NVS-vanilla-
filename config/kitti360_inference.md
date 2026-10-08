# KITTI-360 inference and data contract

## Driving sequences versus benchmark scenes

`2013_05_28_drive_XXXX_sync` identifies one long, continuous driving sequence.
The nine raw Train/Val sequences in the official download are 0000, 0002,
0003, 0004, 0005, 0006, 0007, 0009, and 0010. A paper's reconstruction
"scene" can instead mean a short frame interval inside one such sequence.
LiDAR4D and GS-LiDAR use ten short scenes from drive 0000.

The public sequence IDs are not consecutive: 0001 is absent from the official
release, and the public materials checked here do not state why it was omitted.
0008 is outside the standard Train/Val raw download. The downloaded Test SLAM
archive includes two partial drive-0008 clips under `data_3d_raw/test_0/` and
`test_1/` (933 and 1,547 scans respectively). Pose packages also include 0008
and 0018; this does not imply full raw scans/annotations were released for both.

Official KITTI-360 task splits and LiDAR4D's reconstruction split are separate
protocols. LiDAR4D's `test` here means held-out frames inside its selected clips.

## Sensor rate and actual saved rate

The HDL-64E has 64 laser channels, 360° horizontal FOV and nominal 26.8°
vertical FOV. Its manufacturer's S2 data sheet specifies a selectable
5–15Hz update rate. LiDAR4D's 51-frame dynamic clips are conventionally treated
as about 5 seconds, i.e. approximately 10Hz. This is not a guarantee that the
stored timestamps advance by precisely 100ms.

Measured from the downloaded timestamp files, keyed to actual `.bin` frame IDs:

| Drive | Stored scans | Mean saved Hz | Scans with direct optimized pose | Frames containing dynamic bbox records |
|---|---:|---:|---:|---:|
| 0000 | 11,518 | 9.57118 | 10,514 | 3,748 |
| 0002 | 14,849 | 9.55843 | 14,122 | 1,297 |
| 0003 | 1,031 | 9.55786 | 1,010 | 522 |
| 0004 | 11,587 | 9.56819 | 11,052 | 963 |
| 0005 | 6,743 | 9.56456 | 6,291 | 1,617 |
| 0006 | 9,699 | 9.56873 | 9,186 | 1,676 |
| 0007 | 3,396 | 9.57341 | 2,890 | 1,401 |
| 0009 | 14,056 | 9.56404 | 13,247 | 3,713 |
| 0010 | 3,836 | 9.59758 | 3,026 | 2,570 |

Drive 0000's scan intervals are 103.847–105.120ms. Its 50-frame interval
therefore lasts about 5.22s. The loader uses integer nanosecond timestamp
differences for physical seconds and actual input-to-target transport, rather
than dividing frame offsets by 10. The default requested offsets ±0.5s choose
approximately ±5 frames, whose actual offsets are approximately ±0.522s.

The data's timestamps establish the saved scan rate. Raw sensor UDP packets or
RPM telemetry were not downloaded, so an independent hardware rotation-rate
measurement is unavailable. Consecutive frame IDs and approximately 104ms
intervals do not indicate a nuScenes-style every-other-sweep subsampling scheme.
Drive 0009 additionally has a timestamp gap of about 1.459s; do not globally
replace timestamp differences with a fixed frame period.

## Pose and annotation availability

This is not nuScenes' fixed 2Hz annotated sample / 20Hz sweep organization.
KITTI files are organized by sequence and frame ID:

- Raw `.bin` contains only float32 XYZI; timestamps are provided separately.
- Optimized `poses.txt` is sparse in frame ID. The official documentation says
  low-motion frames can be omitted by a movement threshold. There are direct
  poses for 71,338 of the 76,715 downloaded Train/Val scans. The presence of a
  pose file does not mean every sweep has its own optimized pose row.
- Static 3D primitives are in world coordinates and have `timestamp=-1`; they
  can be reused across visible frames in their annotated region.
- Dynamic bbox records have frame-ID timestamps and shared semantic/instance
  IDs. They are not stored as one complete bbox file per sweep, and there is
  no global nuScenes-style keyframe cadence. The last table column counts
  frames with at least one dynamic record; a zero count at a frame can also
  mean that no dynamic object is present, not necessarily missing labels.
- Dense 3D semantic labels belong to accumulated/windowed point clouds; they
  are not additional fields in the raw XYZI scans. Those PLY archives were
  outside this download's scope.

All 40 benchmark target frames have direct optimized poses. Six distinct input
frame IDs (8125/8135/8145 and 10225/10235/10245) lack a direct pose. Since
2026-10-07 the default overlay uses `pose_missing_policy: interpolate`.
`previous` reproduces the official preprocessing of both baselines. LiDAR4D (`data/preprocess/kitti360_loader.py`,
`load_lidars`) and GS-LiDAR (`preprocess/kitti360_loader.py`, the same function)
append the last pose seen in the requested frame list. Neither paper discusses
missing poses. The vehicle is not stationary at these frames: the bracketing
direct poses are 0.21-0.30 m apart over 2-3 frames. The carried pose is
therefore off by about 0.11-0.20 m. `interpolate` reduces the target-to-input
median nearest-neighbour distance for those inputs by 30-45%. Each window
records `pose_sources` in the results.
`pose_missing_policy=interpolate` optionally uses timestamp-based translation
interpolation and rotation SLERP, with extrapolation rejected; `error` requires
direct poses. Dynamic GT boxes are used only for visualization, never model
inputs, and exact timestamp records are used without synthesizing missing GT.

## Official inference targets

All intervals below are inclusive and belong to `2013_05_28_drive_0000_sync`.

| Scene ID | Category | Full clip | Held-out target frames |
|---|---|---|---|
| 1538 | static | 1538–1601 | 1551, 1564, 1577, 1590 |
| 1728 | static | 1728–1791 | 1741, 1754, 1767, 1780 |
| 1908 | static | 1908–1971 | 1921, 1934, 1947, 1960 |
| 3353 | static | 3353–3416 | 3366, 3379, 3392, 3405 |
| 2350 | dynamic | 2350–2400 | 2360, 2370, 2380, 2390 |
| 4950 | dynamic | 4950–5000 | 4960, 4970, 4980, 4990 |
| 8120 | dynamic | 8120–8170 | 8130, 8140, 8150, 8160 |
| 10200 | dynamic | 10200–10250 | 10210, 10220, 10230, 10240 |
| 10750 | dynamic | 10750–10800 | 10760, 10770, 10780, 10790 |
| 11400 | dynamic | 11400–11450 | 11410, 11420, 11430, 11440 |

The evaluator uses the same targets, panorama projection and existing GS-LiDAR
metric formulas. Our feed-forward model receives only two endpoint scans;
LiDAR4D/GS-LiDAR fit a model to the other 47/60 frames of each scene. This
input/fitting difference must accompany comparisons to their reported numbers.

## Config and coordinates

```bash
python test_lidar4d_nvs.py eval_config=config/kitti360_inference.yaml \
    test.ckpt_path=/absolute/path/to/epoch=29.ckpt device='[1]'
```

The YAML is an inference data overlay. Model architecture, adaptation settings
and weights are recovered from the embedded checkpoint config. Training and
resume configuration guards stay active. The dataset factory also preserves
the existing nuScenes evaluation when `eval_config` is omitted.

With `out_dir: null`, the evaluator writes beside the checkpoint to
`logs/<model>/lidar4d_kitti360_eval/`; nuScenes defaults to
`logs/<model>/lidar4d_nuscenes_eval/`. An explicit `out_dir` remains supported.
For multiple retained runs, specify distinct child folders under this directory.
Historical runs retain their original effective-config and provenance files.

Hardware FOV/range, ring availability, alternative projection sizes and the
training-config checklist are in [sensor_data_contract.md](sensor_data_contract.md).

The default inference overlay uses **64×2178** with the calibrated laser
projection described below (`hdl64e_kitti360_calibrated`). The separate
`kitti360_inference_lidar4d_benchmark.yaml` retains **66×1030**, the grid in
both official preprocessing implementations. In that benchmark projection,
VFOV is **[-24.9°, +2°]** and HFOV **[-180°, +180°]**. Its 66 rows are a
uniform projection grid, not a claim of 66 physical laser channels.
The earlier nominal-angle projection (`hdl64e_s2_s3_nominal`) was removed on
2026-10-07; its runs keep their `effective_config.yaml` under `logs/`.

Returns at or beyond 80m are excluded from GT as in LiDAR4D. Intensity is
already [0,1] and is not divided by 255. Model endpoint inputs retain their
trained 2.5m ego exclusion. Since 2026-10-07 the default overlay sets
`gt_ego_radius: 2.5`, which drops the same near returns from the GT panorama.
`gt_ego_radius: 0` keeps them, as LiDAR4D's `generate_rangeview.py` does. GS-LiDAR's training/eval loader
(`scene/kitti360_loader.py`) instead drops returns within 2.5 m before
building its GT. The KITTI-360 scans themselves contain no return closer than
1.80 m.

For ego-frame checkpoints, KITTI's IMU forward/right/down frame is converted
to forward/left/up. The calibrated LiDAR-to-IMU transform is applied before
tokenization; all temporal matching uses the first endpoint's ego reference.
Sensor-to-reference camera matrices include the KITTI panorama axis convention.
Global poses are combined in float64 before local transforms are cast to
float32, avoiding loss of precision from large world coordinates.

## Calibrated HDL-64E laser projection (default)

KITTI-360 records with a Velodyne HDL-64E (KITTI: 64 beams, 0.09° angular
resolution, 26.8° VFOV). The `.bin` files have no ring column. Even so, every
return was computed with the vendor model for one of the 64 lasers: beam
elevation `δ_k`, a vertical emitter offset, and a horizontal offset (see
`rawdata.cc` in the Velodyne ROS driver). The returns of one laser therefore lie
on a cone around the sensor z axis, `z = ρ·tan δ_k + V_k`. `V_k` is where that
laser's rays cross the axis.

The horizontal offset is ±26.0 mm for every laser, with the sign alternating
between neighbouring lasers. It only turns a return about the z axis by
`asin(h/ρ)`: 0.74° at 2 m, 0.15° at 10 m and 0.04° at 40 m. Removing that turn
puts every near return of all 64 lasers back on the 0.09° encoder grid. The
stored points still lie on the cones: the residual is ≤ 2 µm from 1.5 to 40 m
for lasers 56-63, where a true 26 mm off-axis ray would leave about 44 µm at
3 m. A ray from the cone apex in the return's measured azimuth therefore passes
through the return.

The scans are stored laser by laser, from the top laser down. Each sweep starts
facing +x, so a new laser begins wherever the scan order re-crosses +x. That
yields 64 lasers in 1,099 of 1,102 checked scans. Per-laser line fits on 722
scans from all nine drives give the table in `src/dataloader/kitti360_hdl64e.py`:

- δ runs from +1.937° to -24.999°.
- `V` is 0.198-0.209 m for lasers 0-31 (upper block) and 0.116-0.126 m for
  32-63 (lower block).
- The maximum fit residual is 3 µm, the float32 limit. A fit on only the 40
  targets agrees to 5e-7.

The loader assigns each return to the cone it fits best. This needs no scan
order and fit all 40 targets with residual < 0.002 mm.

Why rows must follow lasers: measured from the sensor origin, one laser's
elevation drifts with range. The drift is +0.24° at 40-80 m and +1.54° at
2.5-5 m, i.e. 1-5 row spacings. Nearest-elevation rows therefore mix lasers.
The removed nominal-angle 64×2178 GT kept 82.2% of returns: 4.3% were lost to its +2.21°
upper mask (91% of the top laser), 7.1% to cross-laser collisions and 6.3% to
same-laser collisions. Row 63 was empty in all 40 targets.

The calibrated projection works as follows:

- Row `k` is laser `k`, and its row angle is `δ_k`.
- The upper and lower blocks each have one render origin on the sensor z axis,
  at +0.2035 m and +0.1209 m (the midrange of each block's `V_k`). GT range is
  the distance from that origin.
- Each block renders the full 64-row panorama from its origin and keeps its own
  32 rows. Cutting a block camera at the block boundary would leave the
  rasterizer's clamped Gaussian centres and low-pass filter on kept rows.
- `StackedPanoramaCamera.row_origin` carries the two origins into the
  range-to-point back-projection used by the CD/F-score metrics.
- `column_assignment: firing_slot` (default) gives every return its own column
  per laser. It keeps azimuth order and moves a return at most 1 column (median
  0.27). Stored azimuths sit on the 0.09° encoder grid apart from the near-range
  horizontal-offset turn above, so plain rounding into 0.165° columns puts two consecutive firings in one pixel 7.5% of the time.
  `nearest` keeps plain rounding.

On the 40 targets:

| GT | returns kept | pixels filled | back-projected GT to raw point, p50 / p99 |
|---|---:|---:|---:|
| calibrated, `firing_slot` | 99.37% | 82.56% | 6.7 / 55 mm |
| calibrated, `nearest` | 92.46% | 76.83% | 6.2 / 47 mm |
| nominal 64×2178 (removed 2026-10-07) | 82.21% | 68.31% | 17.0 / 105 mm |

The remaining approximation is the up-to-5.6 mm offset between each laser's
`V_k` and its block origin. That is at most 0.18° of direction at 1.8 m and
p50 0.005-0.04° overall. Rendering with two coincident origins reproduces a
single 64-row render (CUDA test in `tests/test_kitti360_inference.py`).

## Native KITTI ego origin

The loader uses the original KITTI GPS/IMU origin. Its LiDAR position after
the FRD-to-FLU axis conversion is `[0.771049336, -0.298541436,
0.836280219]`m. No nuScenes calibration or additional origin translation is
used. `T_world_from_ego = T_world_from_imu @ EGO_FROM_IMU`, and relative
poses and cameras are expressed in the first endpoint's native ego frame.

The temporary nuScenes reference-origin experiment was removed on 2026-10-06
at the user's request. Its saved results retain their original provenance.
Tests cover a nonzero sensor extrinsic and a turning vehicle, including
sensor-world composition, relative poses and camera-to-reference transforms.

## Sources and validation

- [KITTI-360 official structure, poses and annotations](https://www.cvlibs.net/datasets/kitti-360/documentation.php)
- [nuScenes official sensor rate and keyframe annotation convention](https://www.nuscenes.org/nuscenes)
- [KITTI-360 paper, including task split and annotation details](https://www.cvlibs.net/publications/Liao2022PAMI.pdf)
- [Velodyne HDL-64E S2 manufacturer data sheet](https://www.termocam.it/pdf/scheda-tecnica-HDL-64E.pdf)
- [LiDAR4D exact frame splits](https://github.com/ispc-lab/LiDAR4D/blob/main/data/preprocess/kitti360_to_nerf.py)
- [LiDAR4D panorama grid](https://github.com/ispc-lab/LiDAR4D/blob/main/data/preprocess/generate_rangeview.py)
- [GS-LiDAR exact frame splits and grid](https://github.com/fudan-zvg/GS-LiDAR/blob/main/preprocess/kitti360_to_gs.py)

Validation covers physical timestamp handling, pose fallback/SLERP, axis
conventions, no held-out targets among input frames, intensity and z-buffering,
and checkpoint/resume guards. A real target scan's projected depth and
intensity maps matched the official LiDAR4D converter exactly (zero differing
pixels). Real-data collation and all 40 target-frame bbox lookups were checked.
