# Waymo Perception v2

The `waymo_dynamic` profile trains the existing two-source-sweep Dynamic 2DGS
backend on the official Waymo training split. It reads TOP LiDAR first returns
from `lidar`, `lidar_pose`, `lidar_calibration`, and `vehicle_pose` Parquet
components. The original component files are retained. `lidar_box` is not
required by this bbox-free loader; it can be retained for later object-level
diagnostics.

`prepare.py` copies range, intensity and per-pixel vehicle poses into one
lossless Zstandard-compressed Parquet row group per frame. Metadata includes
timestamps, frame poses, TOP extrinsics, and the calibrated beam table. A
`READY.json` is written only after the whole segment has been prepared. The
training loader converts the full native 64×2650 TOP range image to XYZI,
including Waymo's per-pixel motion compensation, and expresses points in the
frame-time LiDAR and vehicle frames. It projects target points to a calibrated
64×1024 panorama, keeping the nearest point per ray. Intensities are clipped
to `[0, 1]`; raw values and the original data remain unchanged.
Training windows never cross a gap greater than `data.max_sweep_gap_sec`
(0.15 s by default); the Waymo release contains occasional missing sweeps.

Waymo vehicle axes are +x forward, +y left, +z up; sensor orientation comes
from the recorded `vehicle_from_lidar` extrinsic. Write `W_Ei` for the recorded
world-from-vehicle pose and `E_L` for that extrinsic. Points sent to Utonia
are in each frame's vehicle axes, and `pose[i] = inverse(W_E0) @ W_Ei` maps
them into the first input's vehicle frame. `pose_lidar_to_ref[i]` additionally
includes `E_L`. The target camera maps that shared frame to its LiDAR frame
and then to yaw-aligned panorama axes followed by the renderer's
`(x, y, z) = (-panorama_y, -panorama_z, panorama_x)` axes. Waymo's native
range image compensates sensor extrinsic yaw so its centre faces vehicle
forward; the same yaw adjustment is applied to the GT points and the camera
view. The sensor origin and calibrated beam elevations remain unchanged.
Per-pixel deskew accounts for ego motion during a sweep, not object motion.

The default profile requires all 798 training and 202 validation segments and
rejects segment IDs present in both splits. Per-epoch validation uses only the
20 validation sequences in `config/waymo_lidarverse_top20.json`, each split into
nine fixed non-overlapping 20-frame chunks. Each chunk uses input pairs (0, 4),
(4, 9) and (9, 14) and renders every interior sweep only: 540 input pairs and
1,980 interpolation target views. Timestamps are the recorded physical times;
missing sweeps are not invented. `data.max_val_windows` applies only when
compact validation is disabled; it never truncates this protocol. Training
still uses all 798 train sequences and its original random time sampling.
This interpolation-only validation uses two inputs per pass and is distinct
from the LidarVerse four-input aggregate benchmark, including extrapolation.
When `data.allow_raw_fallback` is true, a segment whose cache is still being
created is read from the original v2 components. The reader switches to the
lossless cache after its `READY.json` appears. Each loader process retains at
most one uncached raw sequence in memory; this can use substantially more RAM
and I/O than reading a prepared cache. Set it to false to require all caches
before constructing the dataset.

```bash
python -m src.dataloader.waymo.prepare \
  --raw-root /data/waymo/waymo_v2 \
  --cache-root /data/waymo/waymo_v2_cache --workers 4

python main.py model.variant=dynamic_2dgs_attention_velocity_v18_2 data_config=waymo
```

For the paper's top-20 travel-distance subset, see
`config/waymo_lidarverse_top20.json`. The paper uses four source frames per
20-frame window, whereas this model accepts exactly two source frames per
forward pass. Its exact range limits, target masks, Chamfer formulation and
window selection should be verified before presenting numbers as an exact
reproduction. The profile's 80 m range limit is the repository's existing
training convention, not a claimed paper setting.
