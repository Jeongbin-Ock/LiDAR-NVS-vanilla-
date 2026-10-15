"""nuScenes LIDAR_TOP sweeps as range-image GT cameras for the shared renderer."""
from __future__ import annotations

import numpy as np
import torch
from pyquaternion import Quaternion

from ...models.utils.camera import Camera


def camera_from_nuscenes(nusc, lidar_token, timestamp_normalized,
                         pts_sensor, cfg, uid=0, lidar_ring=None, ref_to_sensor=None,
                         timestamp_seconds=None):
    """
    NuScenes lidar_token으로 Camera 객체 생성
    pts_sensor : (N, 4) sensor frame xyz+intensity
    """
    sd = nusc.get('sample_data', lidar_token)
    cs = nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
    ep = nusc.get('ego_pose', sd['ego_pose_token'])

    s2e = np.eye(4, dtype=np.float32)
    s2e[:3, :3] = Quaternion(cs['rotation']).rotation_matrix
    s2e[:3,  3] = cs['translation']
    e2w = np.eye(4, dtype=np.float32)
    e2w[:3, :3] = Quaternion(ep['rotation']).rotation_matrix
    e2w[:3,  3] = ep['translation']
    s2w = e2w @ s2e

    R = s2w[:3, :3]
    T = s2w[:3,  3]

    vfov = cfg.vfov
    hfov = cfg.hfov
    H    = cfg.image_height
    W    = cfg.image_width
    ring_to_elevation_deg = getattr(cfg, "ring_to_elevation_deg", None)
    # "ring" (default, the historical behaviour) snaps every return to its
    # sensor ring's row. "uniform" is the GS-LiDAR / LiDAR4D convention: rows
    # are uniform elevation bins over vfov and each return is rasterised by its
    # own angle (nearest wins), so no ring table may be set.
    projection = str(getattr(cfg, "panorama_projection", None) or "ring")
    if projection not in ("ring", "uniform"):
        raise ValueError(f"nuScenes panorama_projection must be 'ring' or 'uniform', got {projection!r}")
    if projection == "uniform" and ring_to_elevation_deg is not None:
        raise ValueError("panorama_projection=uniform requires ring_to_elevation_deg=null")

    # range image 생성
    #####
    #현재는 균등 상태. GT intensity, depth 만들고 싶으면 참고. 
    #####
    # nuScenes LiDAR(z-up) → 카메라(y-down) 축 swap: (x,y,z)->(x,-z,y).
    # range image/rasterizer가 y를 수직축으로 쓰므로 z-up 점을 그대로 넣으면 elevation(행)이
    # 스크램블된다(검증: corr(ring, rasterizer행)≈0). GS-LiDAR nuscenes_loader.py:73-82와 동일.
    x = pts_sensor[:, 0].numpy()
    y = -pts_sensor[:, 2].numpy()
    z = pts_sensor[:, 1].numpy()

    intensity = pts_sensor[:, 3].numpy()

    phi   = np.arctan2(x, z)
    theta = np.arctan2(np.sqrt(x**2 + z**2), -y)
    r     = np.sqrt(x**2 + y**2 + z**2)

    VFOV_max = np.pi / 2 - vfov[0] * np.pi / 180
    VFOV_min = np.pi / 2 - vfov[1] * np.pi / 180
    HFOV_max = hfov[1] * np.pi / 180
    HFOV_min = hfov[0] * np.pi / 180

    theta_px = (theta - VFOV_min) * H / (VFOV_max - VFOV_min)
    phi_px   = (phi   - HFOV_min) * W / (HFOV_max - HFOV_min)


    uvz = np.stack([theta_px, phi_px, r, intensity], axis=1)
    # ring으로 각 점을 정확한 beam 행에 snap → 기하 elevation 노이즈/한 elevation 충돌 방지(GT 정확도↑).
    # render/Camera convention은 row0=최고 elevation이므로, 실제 beam table을 높은 elevation부터
    # 정렬한 순서로 panorama row를 만든다.
    if lidar_ring is not None and projection == "ring":
        ring_np = lidar_ring.numpy() if torch.is_tensor(lidar_ring) else np.asarray(lidar_ring)
        ring_to_el = ring_to_elevation_deg
        if ring_to_el is not None:
            ring_to_el = np.asarray(list(ring_to_el), dtype=np.float32)
            if ring_to_el.shape[0] != H:
                raise ValueError(
                    f"ring_to_elevation_deg length {ring_to_el.shape[0]} "
                    f"does not match image_height {H}"
                )
            ring_order_top_down = np.argsort(-ring_to_el)
            ring_to_row = np.empty_like(ring_order_top_down)
            ring_to_row[ring_order_top_down] = np.arange(H)
            valid_ring = (ring_np >= 0) & (ring_np < H)
            uvz[valid_ring, 0] = ring_to_row[ring_np[valid_ring]].astype(np.float32)
        else:
            valid_ring = (ring_np >= 0) & (ring_np < H)
            el_ring = vfov[0] + ring_np[valid_ring] / (H - 1) * (vfov[1] - vfov[0])
            theta_ring = np.pi / 2 - np.radians(el_ring)
            uvz[valid_ring, 0] = (theta_ring - VFOV_min) * H / (VFOV_max - VFOV_min)
    uvz = uvz[(uvz[:, 0] >= -0.5) & (uvz[:, 0] < H - 0.5)]
    uvz = uvz[(uvz[:, 1] >= -0.5) & (uvz[:, 1] < W - 0.5)]
    uv  = np.around(uvz[:, :2]).astype(int)

    pts_depth_np     = np.zeros([1, H, W], dtype=np.float32)
    pts_intensity_np = np.zeros([1, H, W], dtype=np.float32)
    for i in range(uv.shape[0]):
        row, col = uv[i]
        if pts_depth_np[0, row, col] == 0 or uvz[i, 2] < pts_depth_np[0, row, col]:
            pts_depth_np[0, row, col]     = uvz[i, 2]
            pts_intensity_np[0, row, col] = uvz[i, 3]

    # viewmatrix(ref→sensor)에도 동일 swap을 곱해 gaussian 렌더를 swapped 카메라 프레임으로 맞춘다.
    # M4: sensor(z-up) → camera(y-down). render와 GT range image가 같은 프레임을 쓰게 됨.
    if ref_to_sensor is not None:
        M4 = np.array([[1, 0, 0, 0],
                       [0, 0, -1, 0],
                       [0, 1, 0, 0],
                       [0, 0, 0, 1]], dtype=np.float32)
        ref_to_sensor = (M4 @ np.asarray(ref_to_sensor, dtype=np.float32)).astype(np.float32)

    return Camera(
        R=R, T=T,
        vfov=vfov, hfov=hfov,
        data_device="cpu",
        timestamp=timestamp_normalized,
        timestamp_sec=timestamp_seconds,
        resolution=(W, H),
        points = pts_sensor[:,:3],
        pts_depth=torch.from_numpy(pts_depth_np).float(),
        pts_intensity=torch.from_numpy(pts_intensity_np).float(),
        viewmatrix=ref_to_sensor,
        ring_to_elevation_deg=ring_to_elevation_deg,
    )
