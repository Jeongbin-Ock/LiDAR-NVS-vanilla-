import torch
import math
from .diff_gaussian_rasterization_2d import GaussianRasterizationSettings, GaussianRasterizer

from ..utils.camera import Camera
#from utils.sh_utils import eval_sh
from ..utils.render import Gaussianutil

 

gu = Gaussianutil(None)


def panorama_ray_directions(viewpoint_camera, device, dtype):
    """Every pixel's unit ray direction in the render's world frame, ``(3, H, W)``.

    The rasterizer casts pixel (row y, column x) along azimuth
    ``phi = x * (hfov_max - hfov_min) / W + hfov_min`` and polar angle
    ``theta = row_to_theta[y]``, i.e. along
    ``(sin(theta) sin(phi), -cos(theta), sin(theta) cos(phi))`` in its camera
    frame. ``world_view_transform[:3, :3]`` is the transposed world-to-camera
    rotation, so it takes that direction to the world frame, which for Dynamic
    2DGS is the window's reference frame the Gaussians live in.
    """
    height = int(viewpoint_camera.image_height)
    width = int(viewpoint_camera.image_width)
    hfov_min, hfov_max = (math.radians(float(value)) for value in viewpoint_camera.hfov)
    phi = torch.arange(width, device=device, dtype=dtype) * (
        (hfov_max - hfov_min) / width
    ) + hfov_min
    theta = viewpoint_camera.row_to_theta.to(device=device, dtype=dtype)
    sin_theta = theta.sin().unsqueeze(1)
    camera_direction = torch.stack([
        sin_theta * phi.sin().unsqueeze(0),
        (-theta.cos()).unsqueeze(1).expand(height, width),
        sin_theta * phi.cos().unsqueeze(0),
    ], dim=0)
    camera_to_world = viewpoint_camera.world_view_transform[:3, :3].to(
        device=device, dtype=dtype
    )
    return torch.einsum("ij,jhw->ihw", camera_to_world, camera_direction)


def render(viewpoint_camera, pc, cfg, bg_color, input_timestamp, scaling_modifier=1.0,
           override_color=None, env_map=None, other=[], mask=None, is_training=False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!

    ``pc["shs"]`` renders intensity and raydrop from per-Gaussian SH
    coefficients. ``pc["feature"]`` (V18) instead alpha-blends a per-Gaussian
    feature with no background term and returns it as ``feature`` beside each
    pixel's world-frame ``ray_dir``; G2P's lidar decoder turns those into
    intensity and raydrop, so this package then has neither.
    """
    render_device = pc["position"].device
    render_dtype = pc["position"].dtype

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros((pc["position"].shape[0], 4), dtype=render_dtype, requires_grad=True, device=render_device) + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    if cfg.neg_fov:
        # we find that set fov as -1 slightly improves the results
        tanfovx = math.tan(-0.5)
        tanfovy = math.tan(-0.5)
    else:
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color.to(device=render_device, dtype=render_dtype),
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform.to(device=render_device, dtype=render_dtype),
        projmatrix=viewpoint_camera.full_proj_transform.to(device=render_device, dtype=render_dtype),
        sh_degree=cfg.sh_degree,
        campos=viewpoint_camera.camera_center.to(device=render_device, dtype=render_dtype),
        prefiltered=False,
        debug=False,
        vfov=viewpoint_camera.vfov,
        hfov=viewpoint_camera.hfov,
        row_to_theta=viewpoint_camera.row_to_theta.to(device=render_device, dtype=render_dtype),
        scale_factor=float(1.0 if cfg.scale_factor is None else cfg.scale_factor)
    )

    assert raster_settings.bg.shape[0] == 4

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means2D = screenspace_points


    #params
    means3D = pc["position"]
    opacity = gu.get_opacity(pc["opacity"], opacity_gate=pc.get("opacity_gate"))
    scales = gu.get_scaling(pc["scales"])
    rotations = gu.get_rotation(pc["rotations"])
    cov3D_precomp = None
    feature_mode = "feature" in pc
    if feature_mode:
        # The colour channels blend zeros: only the feature carries appearance.
        shs = None
        colors_precomp = torch.zeros(
            (means3D.shape[0], 4), device=render_device, dtype=render_dtype
        )
    else:
        shs = pc["shs"]
        colors_precomp = None

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    # colors_precomp = None
    # if override_color is None:
    #     if cfg.convert_SHs_python:
    #         shs_view = pc["shs"]
    #         dir_pp = (means3D.detach() - viewpoint_camera.camera_center.repeat(pc["shs"].shape[0], 1)).detach()
    #         dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
    #         sh2rgb = eval_sh(cfg.active_sh_degree, shs_view, dir_pp_normalized)
    #         colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
    #     else:
            
    # else:
    #     colors_precomp = override_color

    feature_list = ([pc["feature"]] if feature_mode else []) + list(other)

    if len(feature_list) > 0:
        features = torch.cat(feature_list, dim=1)
        S_other = features.shape[1]
    else:
        features = torch.zeros_like(means3D[:, :0])
        S_other = 0

    # Prefilter
    mask = (opacity[:, 0] > 1 / 255) if mask is None else mask & (opacity[:, 0] > 1 / 255)

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    # A ``blend_mask`` (V16's routed training render) scales each Gaussian's
    # blending weight only, after the prefilter above and the kernels' culling
    # and alpha cut, which all see the unmasked opacity.
    contrib, rendered_image, rendered_feature, rendered_depth, rendered_opacity, radii = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        features=features,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
        mask=mask,
        blend_mask=pc.get("blend_mask"))

    _, rendered_intensity_sh, rendered_raydrop = rendered_image.split([2, 1, 1], dim=0)
    rendered_other, rendered_normal = rendered_feature.split([S_other, 3], dim=0)
    rendered_normal = rendered_normal / (rendered_normal.norm(dim=0, keepdim=True) + 1e-8)

    if env_map is not None:
        lidar_raydrop_prior_from_envmap = env_map(viewpoint_camera.towards)
        rendered_raydrop = lidar_raydrop_prior_from_envmap + (1 - lidar_raydrop_prior_from_envmap) * rendered_raydrop

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    # return {
    #     "viewspace_points": screenspace_points,
    #     "visibility_filter": radii > 0,
    #     "radii": radii,
    #     "contrib": contrib,
    #     "depth": rendered_depth[[1]] if cfg.median_depth else rendered_depth[[0]],
    #     "depth_mean": rendered_depth[[0]],
    #     "depth_median": rendered_depth[[1]],
    #     "distortion": rendered_depth[[2]],
    #     "depth_square": rendered_depth[[3]],
    #     "alpha": rendered_opacity,
    #     "feature": rendered_other,
    #     "normal": rendered_normal,
    #     "intensity_sh": rendered_intensity_sh,
    #     "raydrop": rendered_raydrop.clamp(0, 1)
    # }
    if feature_mode:
        feature_dim = pc["feature"].shape[1]
        return {
            "depth": rendered_depth[[0]],
            "depth_median": rendered_depth[[1]],
            "normal": rendered_normal,
            "alpha": rendered_opacity,
            "feature": rendered_other[:feature_dim],
            "ray_dir": panorama_ray_directions(
                viewpoint_camera, render_device, render_dtype
            ),
        }
    return {
        "depth": rendered_depth[[0]],          # mean: 알파 가중 기대 depth (depth loss + metric)
        "depth_median": rendered_depth[[1]],   # median: T가 0.5를 넘는 contributor의 depth (chamfer + median loss)
        "normal": rendered_normal,
        "intensity_sh": rendered_intensity_sh,
        # 누적 alpha. raydrop = alpha*(가우시안 SH 가중평균) + (1-alpha)*bg 이고
        # bg의 raydrop 채널이 1이므로, 이 값 없이는 "가우시안이 안 덮어서 drop"과
        # "덮었지만 SH가 drop이라 함"을 구분할 수 없다. 진단 전용 출력.
        "alpha": rendered_opacity,
        "raydrop": rendered_raydrop.clamp(0, 1)
    }
