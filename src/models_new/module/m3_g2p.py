import torch
import torch.nn as nn
from ..gaussian_renderer import render
from ..utils.graphics_utils import lidar4d_range_image_to_points


class DynamicGausRender(nn.Module):
    """Render linearly transported Dynamic 2D Gaussians in physical seconds.

    Every Gaussian moves as ``position + velocity * (t_target - t_source)``
    while its rotation stays fixed.
    """

    # Evaluation tools read this explicit rendering contract instead of
    # inferring it from ``data.mode``.
    transport_mode = "velocity"

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # Channels are [unused, unused, intensity, raydrop]. No-return background
        # should carry zero intensity/depth support and raydrop probability 1.
        self.background = torch.tensor([0, 0, 0, 1], dtype=torch.float32)

    def pack_lidar_shs(self, shs):
        if shs.dim() == 3:
            return shs
        if shs.shape[-1] == 64:
            return shs.reshape(shs.shape[0], 16, 4)
        if shs.shape[-1] != 32:
            raise ValueError(f"Expected SH tensor with 32 or 64 channels, got {shs.shape[-1]}.")

        coeff = shs.reshape(shs.shape[0], 16, 2)
        unused = torch.zeros(shs.shape[0], 16, 2, device=shs.device, dtype=shs.dtype)
        return torch.cat([unused, coeff], dim=-1)

    @staticmethod
    def pack_scales(scales):
        if scales.shape[-1] == 3:
            return scales
        if scales.shape[-1] != 2:
            raise ValueError(f"Expected scale tensor with 2 or 3 channels, got {scales.shape[-1]}.")
        z_log_scale = torch.zeros(scales.shape[0], 1, device=scales.device, dtype=scales.dtype)
        return torch.cat([scales, z_log_scale], dim=-1)

    @staticmethod
    def _pad_view_batches(items, max_views):
        """Pad per-sample ``[V,...]`` tensors without inventing valid views."""
        padded = []
        for item in items:
            missing = int(max_views) - int(item.shape[0])
            if missing < 0:
                raise ValueError("max_views is smaller than a rendered view batch")
            if missing:
                item = torch.cat(
                    [item, item.new_zeros((missing, *item.shape[1:]))], dim=0
                )
            padded.append(item)
        return torch.stack(padded, dim=0)

    @staticmethod
    def render_timestamp(camera):
        if getattr(camera, "timestamp_sec", None) is None:
            raise AttributeError(
                "Dynamic rendering requires Camera.timestamp_sec from the dataloader"
            )
        return camera.timestamp_sec

    def get_means3D(self, b_gs, t):
        position = b_gs["position"]
        velocity = b_gs.get("velocity")
        source_time = b_gs.get("source_time_sec")
        if velocity is None or source_time is None:
            raise KeyError(
                "Dynamic Gaussian render input needs velocity and source_time_sec"
            )
        if velocity.shape != position.shape:
            raise ValueError("Dynamic velocity must align with Gaussian positions")
        if source_time.shape != (position.shape[0],):
            raise ValueError("source_time_sec must provide one value per Gaussian")
        target_time = torch.as_tensor(
            t, device=position.device, dtype=position.dtype
        )
        delta_t = target_time - source_time.to(
            device=position.device, dtype=position.dtype
        )
        return position + velocity * delta_t.unsqueeze(-1)

    def get_rotations(self, b_gs, t):
        # Dynamic 2DGS transports only position.
        return b_gs["rotation"]

    def forward(
        self,
        gaussians,
        gt,
        *,
        compute_points=True,
    ):
        if isinstance(gaussians, dict):
            gaussians = gaussians.get("batch_gaussians", gaussians.get("gaussians"))
        if isinstance(gt, dict):
            gt_cameras_batch = gt["cameras"]
        else:
            gt_cameras_batch = gt

        # 텐서로 묶을 데이터용 리스트
        all_depths, all_depths_median, all_intensity_shs, all_raydrops = [], [], [], []
        # 진단 전용: eval에서만 수집해 학습 메모리/그래프를 그대로 둔다.
        all_alphas = [] if not self.training else None
        all_gt_depths, all_gt_intensity_shs, all_gt_raydrops = [], [], []
        
        # 포인트 개수가 유동적일 수 있으므로 파이썬 리스트로 유지
        all_render_points = []
        all_gt_points = []
        
        for b, (b_gs, gt_cameras) in enumerate(zip(gaussians, gt_cameras_batch)):
            if b_gs is None:
                raise ValueError(f"Batch {b}의 Gaussian 데이터가 None입니다.")

            b_depths, b_depths_median, b_intensity_shs, b_raydrops = [], [], [], []
            b_alphas = [] if all_alphas is not None else None
            b_gt_depths, b_gt_intensity_shs, b_gt_raydrops = [], [], []
            b_render_points, b_gt_points = [], []
            for gt_cam in gt_cameras:
                t = self.render_timestamp(gt_cam)

                # gt timestamp 기준 position 계산
                means3D = self.get_means3D(b_gs, t)
                rotations = self.get_rotations(b_gs, t)

                pc = {
                    "position": means3D,
                    "opacity":  b_gs["opacity"],
                    "scales":   self.pack_scales(b_gs.get("scales", b_gs.get("scale", b_gs["scaling"]))),
                    "rotation": rotations,
                    "rotations": rotations,
                    "shs":      self.pack_lidar_shs(b_gs["shs"]),
                }
                if "opacity_gate" in b_gs:
                    pc["opacity_gate"] = b_gs["opacity_gate"]
                if "blend_mask" in b_gs:
                    pc["blend_mask"] = b_gs["blend_mask"]

                # 랜더링
                render_pkg = render(
                    viewpoint_camera=gt_cam,
                    pc=pc,
                    cfg=self.cfg,
                    bg_color=self.background.to(means3D.device),
                    input_timestamp=t,
                    is_training=self.training,
                )

                gt_depth = gt_cam.pts_depth.to(device=means3D.device, dtype=means3D.dtype)
                gt_intensity_sh = gt_cam.pts_intensity.to(device=means3D.device, dtype=means3D.dtype)
                gt_raydrop = 1.0 - (gt_depth > 0).float()


                if compute_points:
                    # Chamfer/evaluation support is built only when a caller
                    # consumes it. Normal training with w_chamfer=0 skips both
                    # back-projections entirely.
                    render_points = lidar4d_range_image_to_points(
                        render_pkg["depth_median"],
                        gt_cam.vfov,
                        gt_cam.hfov,
                        row_to_theta=gt_cam.row_to_theta,
                        raydrop=render_pkg["raydrop"],
                    )

                # 픽셀 기반 맵 모으기
                b_depths.append(render_pkg["depth"])
                b_depths_median.append(render_pkg["depth_median"])
                b_intensity_shs.append(render_pkg["intensity_sh"])
                b_raydrops.append(render_pkg["raydrop"])
                if b_alphas is not None:
                    b_alphas.append(render_pkg["alpha"])

                b_gt_depths.append(gt_depth)
                b_gt_intensity_shs.append(gt_intensity_sh)
                b_gt_raydrops.append(gt_raydrop)

                # 포인트 데이터 모으기 (개수가 다를 수 있으므로 리스트 유지)
                # gt도 pred와 동일한 LiDAR4D range/mapping 규칙으로 복원한다.
                # raw gt_cam.points(z-up 센서)는 pano(y-up view) pred와 프레임이 달라 chamfer 폭증 →
                # round-trip 검증상 pano는 rasterizer를 정확히 역변환하므로 양쪽을 pano로 맞춘다.
                if compute_points:
                    b_render_points.append(render_points)
                    b_gt_points.append(
                        lidar4d_range_image_to_points(
                            gt_depth,
                            gt_cam.vfov,
                            gt_cam.hfov,
                            row_to_theta=gt_cam.row_to_theta,
                        )
                    )

            # 카메라 차원 stack
            all_depths.append(torch.stack(b_depths, dim=0))
            all_depths_median.append(torch.stack(b_depths_median, dim=0))
            all_intensity_shs.append(torch.stack(b_intensity_shs, dim=0))
            all_raydrops.append(torch.stack(b_raydrops, dim=0))
            if all_alphas is not None:
                all_alphas.append(torch.stack(b_alphas, dim=0))
            all_gt_depths.append(torch.stack(b_gt_depths, dim=0))
            all_gt_intensity_shs.append(torch.stack(b_gt_intensity_shs, dim=0))
            all_gt_raydrops.append(torch.stack(b_gt_raydrops, dim=0))

            # 포인트 리스트 저장
            all_render_points.append(b_render_points)
            all_gt_points.append(b_gt_points)

        # Dynamic-duration batches can contain 3, 4, or 5 supervised views per
        # sample. Pad only the returned image tensors; ``view_mask`` tells the
        # loss which rows are real, while point lists remain naturally ragged.
        view_counts = [int(value.shape[0]) for value in all_depths]
        if not view_counts or min(view_counts) <= 0:
            raise ValueError("Every render sample must contain at least one target view")
        max_views = max(view_counts)
        view_mask = (
            torch.arange(max_views, device=all_depths[0].device).unsqueeze(0)
            < torch.tensor(
                view_counts, device=all_depths[0].device, dtype=torch.long
            ).unsqueeze(1)
        )

        # 최종 딕셔너리 반환
        return {
            # 고정 차원 텐서 (Batch, Cam, H, W)
            "depth": self._pad_view_batches(all_depths, max_views),
            "depth_median": self._pad_view_batches(all_depths_median, max_views),
            "intensity_sh": self._pad_view_batches(all_intensity_shs, max_views),
            "raydrop": self._pad_view_batches(all_raydrops, max_views),
            **({"alpha": self._pad_view_batches(all_alphas, max_views)}
               if all_alphas is not None else {}),
            "gt_depth": self._pad_view_batches(all_gt_depths, max_views),
            "gt_intensity_sh": self._pad_view_batches(
                all_gt_intensity_shs, max_views
            ),
            "gt_raydrop": self._pad_view_batches(all_gt_raydrops, max_views),
            "view_mask": view_mask,
            
            # 가변 크기 가능 리스트 (중첩 리스트 형태: [Batch][Cam])
            "render_points": all_render_points,
            "gt_points": all_gt_points
        }


__all__ = ["DynamicGausRender"]
