# H3D zero-shot inference and data contract

Honda H3D is used only for inference: a KITTI-360 checkpoint (e.g.
`logs/v15.3_kitti_lidar4d/epoch=29.ckpt` or its raydrop-refinement checkpoint)
renders H3D sweeps without any H3D training. Overlay:
[`eval/h3d_inference_lidar4d.yaml`](eval/h3d_inference_lidar4d.yaml); loader:
`src/dataloader/h3d/` (`metadata.py` is torch-free).

```bash
python test_lidar4d_nvs.py eval_config=config/eval/h3d_inference_lidar4d.yaml \
    test.ckpt_path=/abs/path/epoch=29.ckpt device='[6]'
```

Results default to `<checkpoint folder>/lidar4d_h3d_eval/`. `data.scenarios`
restricts the run to named scenarios (used to shard one run across processes).
The overlay sets `test.save_gaussian_viz=false`: the per-sequence Gaussian HTML
holds every window in memory and costs ~140 MB per window.

## Release layout

`/data1/LiDAR/H3D/data/icra_benchmark_20200103_with_odom/` holds 160 flat
`scenario_NNN/` folders and `train_test_split/`. The official detection splits
list 50 / 30 / 80 scenarios (train / val / test) with 8,899 / 5,270 / 15,190
sweeps; the paper's Table I gives the same scenario counts and 8,873 / 5,170 /
13,678 annotated frames. Each frame has `pointcloud1_NNN.ply`, `odom_NNN.txt`,
`gps_NNN.csv`, `CAN_vel_NNN.csv`, `CAN_yaw_NNN.csv` and `labels_3d1_NNN.txt`.

146 scenarios span at least 5 s (train 44, val 26, test 76); the other 14 have
44 sweeps (4.27-4.30 s). An earlier CPU audit of timing and poses is in
`/data1/LiDAR/H3D/audit_temporal_20261016/README_KO.md`.

## Sensor: same HDL-64E model as KITTI-360, a different unit

The H3D paper names a Velodyne HDL-64E S2 (10 Hz); KITTI-360 names an HDL-64E.
Measured on 480 H3D sweeps (3 per scenario) and 360 KITTI-360 raw scans
(`tools/h3d_sensor_audit.py`):

| | H3D | KITTI-360 |
| --- | --- | --- |
| Lasers | 64 (PLY `confidence` = laser ID) | 64 (cone fit, `kitti360/hdl64e.py`) |
| Beam elevations | +2.19 to -24.61 deg | +1.94 to -25.00 deg |
| Emitter-block apex heights | 0.204-0.217 / 0.144-0.163 m | 0.198-0.209 / 0.116-0.126 m |
| Columns per turn, rate | 2,082 median at 10.0 Hz | ~2,178 slots at 9.57 Hz |
| Firing rate per laser | ~20.8 kHz | ~20.8 kHz |
| Stored points per sweep | 133,248 median, no-return rays included | 119,254 median, returns only |
| Returns per sweep | 124,516 median | 119,254 median |
| Returns beyond 80 m | 0.5 % | none |
| Intensity encoding | 8-bit `radius` (1-254) | 8-bit / 256 |
| Intensity mean / median (2.5-80 m) | 0.340 / 0.33 | 0.290 / 0.31 |
| Intensity > 0.5 | 15.2 % | 6.1 % |
| Per-laser cone residual | ~1e-7 m (raw, not deskewed) | <= 3 um (raw) |
| Road plane in LiDAR frame | z = -1.879 m, tilt 0.50 / -0.15 deg | z = -1.727 m, tilt 0.39 / -0.03 deg |
| Lowest lasers' returns per turn | 96-99.6 % (bottom 8), ground at 4-5 m all round | 46-67 % (bottom 7), ~0 within +-10 deg ahead and 160-200 deg behind |

Same model family: 64 lasers in two emitter blocks, the same 26.8 deg span and
firing rate, 8-bit intensity. A different physical unit and mount: beam angles
differ by 0.32 deg on average (max 0.50 deg), per-laser intensity gains
differ, the H3D unit returns more high-intensity points, and it sits 0.152 m
higher. KITTI-360's lowest lasers lose their front and rear returns (the
vehicle body, presumably, of a roof mount); H3D's see the ground all round,
with 1.2 % of their rays hitting something at 1.5-2.5 m. The paper does not
state the mount. Both releases store raw scans; the paper's deskewing (motion
compensation within a sweep) is not in the released points, which lie exactly
on the per-laser cones.

## PLY fields

The PLY reuses PCL surfel properties. The official `convert_to_bin.py` keeps
`x y z` and `radius / 255` as intensity. `confidence` is the laser ID (63 = top
beam, 0 = bottom) and `curvature` the encoder azimuth in degrees (0.09 deg
encoder steps). Each file is an organized 64 x C scan with C = 2,082 +- 6k
(whole 6-firing packets). Rays without a return are stored at 1.0-1.5 m
(raw distance 0 plus the per-laser distance correction); the loader drops
every point within 1.5 m.

## Preprocessing for a KITTI-360 checkpoint

- **Axes.** The LiDAR frame is x forward, y left, z up, as KITTI's Velodyne:
  in right-hand traffic all 37 labelled oncoming cars within 25 m have y > 0.
- **Pose.** `odom_NNN.txt` is `x y z roll pitch yaw` of the LiDAR in a
  per-scenario world frame; `T = [Rz(yaw) Ry(pitch) Rx(roll) | t]` and
  `p_world = T p_lidar` (official AB3DMOT `get_matrix`). Over the 35 val/test
  scenarios that turn by more than 20 deg, the odometry yaw change matches the
  ADMA GPS heading change.
- **Ego frame.** The checkpoint's `coordinate_frame` is `ego_vehicle`
  (KITTI GPS/IMU origin, forward/left/up), but H3D ships no LiDAR-to-vehicle
  calibration. `data.ego_from_lidar` is KITTI-360's `ego_from_lidar` (rotation
  and x/y kept) with z raised by 0.152 m, the height difference of the fitted
  road planes, so the road sits at the same ego height as in training. The
  plane tilts agree within 0.12 deg, so KITTI's rotation is kept.
- **Intensity.** `radius / 256`, the KITTI-360 encoding of the same 8-bit value.
- **Time.** No LiDAR capture timestamp exists. Each sweep's `gps_NNN.csv`
  bundles ~10 ADMA samples (100 Hz); the midpoint of the first and last is the
  frame time (median spacing 0.100 s, 0.091-0.110 s). Seconds are differenced
  as integer microseconds.
- **Crops.** Inputs keep 2.5 m < r < 80 m. The KITTI-360 pipeline has no
  80 m input crop, but its raw scans hold no return beyond ~80 m (4.7e-7 of
  returns in 80-100 m over 360 scans), so the checkpoint never saw one;
  `data.input_max_range_m=0` disables the crop. GT keeps returns beyond 2.5 m
  and below 80 m, as the KITTI-360 LiDAR4D benchmark overlay.
- **Grid.** 66 x 1030 uniform rows over vfov [-24.5, 2.4] deg: LiDAR4D's
  KITTI-360 grid [-24.9, 2.0] (set by that unit's +1.94 to -25.00 deg beams)
  moved by the H3D unit's beam offset. Three model-free estimates agree on
  +0.32 to +0.43 deg: the fitted beam angles (+0.32), the elevation seen from
  the sensor origin including the emitter apex heights (+0.43 / +0.375 /
  +0.35 at 10 / 20 / 40 m), and the best alignment of per-row GT fill
  profiles (+0.40, `tools/h3d_vfov_alignment.py`). The 66 rows keep their
  0.41 deg pitch, so each laser lands on the row it occupies for KITTI-360.
  `data.vfov=[-24.9,2.0]` restores the KITTI grid.

## Protocol

Every val/test scenario of at least `min_duration_sec` (5 s, first-to-last
sweep) contributes one target per whole second T = 1, 2, ...: the sweep nearest
to T s after the first sweep. The inputs are the nearest other sweeps to the
target time + `test.input_offsets_sec` (default -0.1 / +0.1 s, normally the
adjacent sweeps). A target is kept only if every sweep lies within 0.05 s of
its requested time, so an 8 s scenario yields T = 1..7. The val+test run has
102 scenarios and 1,967 targets. Metrics are those of `test_lidar4d_nvs.py`.

## Known zero-shot gaps

- Per-row GT fill profiles of the two units correlate at 0.58 on the KITTI
  grid and 0.66 on the H3D grid: beam angles are only part of it. The lowest
  rows' drop pattern (KITTI's body-occluded front/rear) does not exist in H3D.
- The mount difference shows in the near field, which carries little of the
  error: returns at 2.5-5 m cause 2.6 % of the H3D depth squared error (V15.3
  ep29, 197 windows, `tools/h3d_error_maps.py`), and excluding < 5 m from both
  prediction and GT leaves the depth-PSNR gap to KITTI-360 at 4.6 dB. As on
  KITTI-360, most error is hit/miss decisions in the upper rows and at 40-80 m;
  H3D has about twice as many such pixels (17.7 % vs 8.6 %).
- A raydrop-refinement U-Net trained on KITTI-360 learns these image-space
  patterns; on six H3D windows it lowered depth PSNR (24.43 vs 25.04 dB) and
  raydrop F1 (0.41 vs 0.50), so zero-shot runs use the base checkpoint.
- Odometry: point-to-plane ICP corrections between adjacent sweeps are
  ~1.3 cm (median) but larger at 1 s spacing than KITTI-360's optimized
  poses; check a clip's poses before per-scene optimization
  (`tools/h3d_clip_pose_check.py`).
- H3D scenes are dense urban intersections; the ego is stopped in a large
  share of windows.
