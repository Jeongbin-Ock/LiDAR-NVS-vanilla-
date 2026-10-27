"""V18: alpha-blended Gaussian features decoded into intensity and raydrop.

The rasterizer's feature channels must blend and differentiate exactly like
its colour channels, whose kernels every earlier variant trains through:
same per-pixel values, same gradients to the per-Gaussian values, and the
same gradients to opacity and geometry through the blending weights (which
the feature channels did not receive before V18). G2P's decoder then reads
[f', ray direction, median depth / R, sensor embedding] per pixel; V18.1's
reads [f', ray direction, median depth / R].
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs the CUDA rasterizer"
)

DECODER_CFG = {
    "feature_dim": 30, "sensor_embedding_dim": 30, "num_sensors": 1,
    "hidden_dims": [96, 96], "range_scale_m": 80.0,
}
_RENDER_CFG = OmegaConf.create({"neg_fov": True, "sh_degree": 3, "scale_factor": 1.0})
_RING = [
    -30.67, -29.33, -28.00, -26.66, -25.33, -24.00, -22.67, -21.33,
    -20.00, -18.67, -17.33, -16.00, -14.67, -13.33, -12.00, -10.67,
    -9.33, -8.00, -6.66, -5.33, -4.00, -2.67, -1.33, 0.00,
    1.33, 2.67, 4.00, 5.33, 6.67, 8.00, 9.33, 10.67,
]
# nuScenes' sensor(z-up) -> camera(y-down) axis swap (src/dataloader/nuscenes/camera.py).
_M4 = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
               dtype=np.float32)


def _reference_to_sensor(yaw=0.4, translation=(1.5, -0.7, -1.8)):
    """A rotated, displaced sensor, so the world frame is not the camera's."""
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, :3] = np.array(
        [[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0],
         [0, 0, 1]], dtype=np.float32,
    )
    matrix[:3, 3] = translation
    return matrix


def _camera(width=512):
    from src.models.utils.camera import Camera

    height = len(_RING)
    return Camera(
        R=np.eye(3), T=np.zeros(3), vfov=(-30.67, 10.67), hfov=(-180.0, 180.0),
        viewmatrix=_M4 @ _reference_to_sensor(), resolution=(width, height),
        timestamp_sec=0.0, ring_to_elevation_deg=_RING,
        pts_depth=torch.zeros(1, height, width),
        pts_intensity=torch.zeros(1, height, width),
    )


def _scene(count=3000, seed=0):
    """Gaussians around the sensor, in the reference (world) frame."""
    generator = torch.Generator().manual_seed(seed)
    azimuth = torch.rand(count, generator=generator) * 2 * math.pi
    distance = 3 + 40 * torch.rand(count, generator=generator)
    elevation = torch.deg2rad(-25 + 30 * torch.rand(count, generator=generator))
    sensor = torch.stack([
        distance * elevation.cos() * azimuth.cos(),
        distance * elevation.cos() * azimuth.sin(),
        distance * elevation.sin(),
    ], 1)
    to_reference = torch.from_numpy(np.linalg.inv(_reference_to_sensor()))
    position = sensor @ to_reference[:3, :3].T + to_reference[:3, 3]
    scales = torch.log(torch.expm1(0.1 + 0.5 * torch.rand(count, 3, generator=generator)))
    scales[:, 2] = -10.0
    return {
        "position": position.cuda(),
        "opacity": (-0.5 + torch.randn(count, 1, generator=generator)).cuda(),
        "scales": scales.cuda(),
        "rotations": F.normalize(torch.randn(count, 4, generator=generator), dim=1).cuda(),
    }


def _rasterize(pc, *, colors=None, features=None, blend_mask=None):
    """The rasterizer with explicit colours/features and a zero background.

    ``render`` fixes the colour source (SH or zeros), so this sets the
    rasterizer up the same way and passes ``colors_precomp`` directly.
    """
    from src.models.gaussian_renderer import (
        GaussianRasterizationSettings,
        GaussianRasterizer,
        gu,
    )

    camera = _camera()
    count = pc["position"].shape[0]
    device = pc["position"].device
    settings = GaussianRasterizationSettings(
        image_height=int(camera.image_height), image_width=int(camera.image_width),
        tanfovx=math.tan(-0.5), tanfovy=math.tan(-0.5),
        bg=torch.zeros(4, device=device), scale_modifier=1.0,
        viewmatrix=camera.world_view_transform.to(device),
        projmatrix=camera.full_proj_transform.to(device), sh_degree=3,
        campos=camera.camera_center.to(device), prefiltered=False, debug=False,
        vfov=camera.vfov, hfov=camera.hfov,
        row_to_theta=camera.row_to_theta.to(device), scale_factor=1.0,
    )
    opacity = gu.get_opacity(pc["opacity"])
    if colors is None:
        colors = torch.zeros(count, 4, device=device)
    if features is None:
        features = torch.zeros(count, 0, device=device)
    _contrib, color, feature, depth, alpha, _radii = GaussianRasterizer(settings)(
        means3D=pc["position"],
        means2D=torch.zeros(count, 4, device=device, requires_grad=True),
        opacities=opacity, colors_precomp=colors, features=features,
        scales=gu.get_scaling(pc["scales"]), rotations=gu.get_rotation(pc["rotations"]),
        mask=opacity[:, 0] > 1 / 255, blend_mask=blend_mask,
    )
    return color, feature, depth, alpha


def _leaves(pc):
    return {key: value.detach().clone().requires_grad_(True) for key, value in pc.items()}


def _relative(a, b):
    return ((a - b).abs().max() / max(a.abs().max(), b.abs().max(), 1e-12)).item()


GEOMETRY = ("position", "opacity", "scales", "rotations")


@requires_cuda
@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("feature_channels", [30, 32, 13])
def test_feature_channels_blend_and_differentiate_like_colour_channels(feature_channels, masked):
    """``feature_channels`` features rendered at once == the same values as 4-channel colours.

    30 and 32 run the kernels compiled for V18's and V18.2's feature widths, 13
    the run-time path. The loss is linear in the channels, so the gradient of
    one feature render must be the sum over eight colour renders of four
    channels each -- for the per-Gaussian values and, through the blending
    weights, for opacity, position, scale and rotation.
    """
    pc = _scene()
    count = pc["position"].shape[0]
    generator = torch.Generator().manual_seed(5)
    values = torch.randn(count, 32, generator=generator).cuda()
    values[:, feature_channels:] = 0.0
    weights = torch.randn(32, len(_RING), 512, generator=generator).cuda()
    weights[feature_channels:] = 0.0
    blend_mask = (
        torch.rand(count, generator=generator).cuda() if masked else None
    )

    leaves = _leaves(pc)
    feature_values = values[:, :feature_channels].clone().requires_grad_(True)
    _color, feature, _depth, _alpha = _rasterize(
        leaves, features=feature_values, blend_mask=blend_mask
    )
    assert feature.shape == (feature_channels + 3, len(_RING), 512)
    (feature[:feature_channels] * weights[:feature_channels]).sum().backward()
    feature_grads = {key: leaves[key].grad for key in GEOMETRY}

    colour_grads = {key: torch.zeros_like(pc[key]) for key in GEOMETRY}
    colour_value_grad = torch.zeros(count, 32, device="cuda")
    # The features padded with empty channels to eight colour groups.
    blended = torch.cat([
        feature.detach()[:feature_channels],
        feature.new_zeros((32 - feature_channels, *feature.shape[1:])),
    ])
    for group in range(8):
        channels = slice(4 * group, 4 * group + 4)
        leaves = _leaves(pc)
        colours = values[:, channels].clone().requires_grad_(True)
        color, _feature, _depth, _alpha = _rasterize(
            leaves, colors=colours, blend_mask=blend_mask
        )
        torch.testing.assert_close(color, blended[channels], rtol=0, atol=0)
        (color * weights[channels]).sum().backward()
        for key in GEOMETRY:
            colour_grads[key] += leaves[key].grad
        colour_value_grad[:, channels] = colours.grad

    # Atomic accumulation order differs between launches, so gradients agree to
    # float32 summation noise relative to their scale, not bitwise.
    assert _relative(feature_values.grad, colour_value_grad[:, :feature_channels]) < 1e-5
    for key in GEOMETRY:
        assert feature_grads[key].abs().max() > 0, key
        assert _relative(feature_grads[key], colour_grads[key]) < 1e-4, key


@requires_cuda
@pytest.mark.parametrize("masked", [False, True])
def test_rendered_alpha_differentiates_like_a_unit_colour(masked):
    """V18 composites by the rendered alpha; its gradient is the colour path's.

    A colour channel of 1 over a zero background renders exactly 1 - T, so the
    alpha output and that channel must agree in value and in every gradient.
    """
    pc = _scene()
    count = pc["position"].shape[0]
    generator = torch.Generator().manual_seed(6)
    weight = torch.randn(1, len(_RING), 512, generator=generator).cuda()
    blend_mask = torch.rand(count, generator=generator).cuda() if masked else None

    leaves = _leaves(pc)
    _color, _feature, _depth, alpha = _rasterize(leaves, blend_mask=blend_mask)
    (alpha * weight).sum().backward()
    alpha_grads = {key: leaves[key].grad for key in GEOMETRY}

    leaves = _leaves(pc)
    colours = torch.zeros(count, 4, device="cuda")
    colours[:, 0] = 1.0
    color, _feature, _depth, _alpha = _rasterize(
        leaves, colors=colours, blend_mask=blend_mask
    )
    torch.testing.assert_close(color[:1], alpha, rtol=0, atol=1e-6)
    (color[:1] * weight).sum().backward()
    for key in GEOMETRY:
        assert alpha_grads[key].abs().max() > 0, key
        assert _relative(alpha_grads[key], leaves[key].grad) < 1e-4, key


@requires_cuda
def test_feature_blending_has_no_background_and_rejects_too_many_channels():
    pc = _scene(count=200)
    count = pc["position"].shape[0]
    ones = torch.ones(count, 30, device="cuda")
    _color, feature, _depth, alpha = _rasterize(pc, features=ones)
    # A constant feature blends to the accumulated alpha: nothing behind it.
    torch.testing.assert_close(
        feature[:30], alpha.expand(30, -1, -1), rtol=1e-5, atol=1e-6
    )
    with pytest.raises(RuntimeError, match="at most 32"):
        _rasterize(pc, features=torch.ones(count, 33, device="cuda"))


@requires_cuda
def test_ray_directions_back_project_median_depth_onto_the_surface():
    """``camera_center + depth_median * ray_dir`` lands on a rendered wall.

    One large opaque Gaussian is a wall at x = 12 m in the world (reference)
    frame, seen by a rotated and displaced sensor. Every pixel it covers must
    back-project onto that plane, which holds only for world-frame rays.
    """
    from src.models.gaussian_renderer import render

    camera = _camera(width=1085)
    rotation = torch.tensor([[math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0]])
    pc = {
        # A disc in the y-z plane: rotated 90 degrees about y, normal along x.
        "position": torch.tensor([[12.0, 0.0, 0.0]]),
        "opacity": torch.tensor([[8.0]]),
        "scales": torch.tensor([[math.log(math.expm1(4.0)), math.log(math.expm1(4.0)), -10.0]]),
        "rotations": rotation,
        "feature": torch.zeros(1, 30),
    }
    pc = {key: value.cuda() for key, value in pc.items()}
    package = render(
        viewpoint_camera=camera, pc=pc, cfg=_RENDER_CFG,
        bg_color=torch.tensor([0.0, 0.0, 0.0, 1.0]).cuda(), input_timestamp=0.0,
    )
    assert "intensity_sh" not in package and "raydrop" not in package
    ray = package["ray_dir"]
    torch.testing.assert_close(ray.norm(dim=0), torch.ones_like(ray[0]))
    hit = package["alpha"][0] > 0.9
    assert int(hit.sum()) > 50
    center = camera.camera_center.cuda()
    points = center[:, None] + package["depth_median"][0][hit] * ray[:, hit]
    assert (points[0] - 12.0).abs().max() < 0.05


def test_decoder_reads_feature_ray_range_and_sensor_embedding_in_order():
    from src.models.module.lidar_decoder import LidarFeatureDecoder

    decoder = LidarFeatureDecoder(OmegaConf.create(DECODER_CFG))
    assert decoder.input_dim == 64
    assert [layer.out_features for layer in decoder.mlp
            if isinstance(layer, torch.nn.Linear)] == [96, 96, 2]
    assert all(isinstance(layer, (torch.nn.Linear, torch.nn.SiLU)) for layer in decoder.mlp)
    feature = torch.randn(30, 4, 6)
    ray = F.normalize(torch.randn(3, 4, 6), dim=0)
    depth = 100 * torch.rand(1, 4, 6)
    rows = decoder.decoder_input(feature, ray, depth)
    assert rows.shape == (24, 64)
    torch.testing.assert_close(rows[:, :30], feature.reshape(30, -1).T)
    torch.testing.assert_close(rows[:, 30:33], ray.reshape(3, -1).T)
    torch.testing.assert_close(rows[:, 33], depth.reshape(-1) / 80.0)
    torch.testing.assert_close(
        rows[:, 34:], decoder.sensor_embedding.weight[0].expand(24, -1)
    )
    intensity, raydrop = decoder.surface(feature, ray, depth)
    assert intensity.shape == raydrop.shape == (1, 4, 6)
    assert bool(((intensity > 0) & (intensity < 1)).all())
    assert bool(((raydrop > 0) & (raydrop < 1)).all())
    # The pixel composites the hit surface over the no-return background.
    alpha = torch.rand(1, 4, 6)
    alpha[0, 0] = 0.0
    alpha[0, 1] = 1.0
    pixel_intensity, pixel_raydrop = decoder(feature, ray, depth, alpha)
    torch.testing.assert_close(pixel_intensity, alpha * intensity)
    torch.testing.assert_close(pixel_raydrop, (1 - alpha) + alpha * raydrop)
    assert torch.equal(pixel_intensity[0, 0], torch.zeros(6))
    assert torch.equal(pixel_raydrop[0, 0], torch.ones(6))
    torch.testing.assert_close(pixel_raydrop[0, 1], raydrop[0, 1])


def test_v18_1_decoder_reads_no_sensor_embedding():
    from src.models.module.lidar_decoder import LidarFeatureDecoder

    config = {key: value for key, value in DECODER_CFG.items() if key != "num_sensors"}
    decoder = LidarFeatureDecoder(OmegaConf.create({**config, "sensor_embedding_dim": 0}))
    assert decoder.input_dim == 34
    assert decoder.sensor_embedding is None
    assert decoder.mlp[0].in_features == 34
    assert sum(p.numel() for p in decoder.parameters()) == 12866
    feature = torch.randn(30, 4, 6)
    ray = F.normalize(torch.randn(3, 4, 6), dim=0)
    depth = 100 * torch.rand(1, 4, 6)
    rows = decoder.decoder_input(feature, ray, depth)
    assert rows.shape == (24, 34)
    torch.testing.assert_close(rows[:, :30], feature.reshape(30, -1).T)
    torch.testing.assert_close(rows[:, 30:33], ray.reshape(3, -1).T)
    torch.testing.assert_close(rows[:, 33], depth.reshape(-1) / 80.0)
    with pytest.raises(ValueError, match="sensor 0 only"):
        decoder.decoder_input(feature, ray, depth, sensor_index=1)


def test_v18_decoder_folds_its_sensor_embedding_into_v18_1s_first_bias():
    """One sensor: V18's embedding is a constant input, i.e. first-layer bias."""
    from src.models.module.lidar_decoder import LidarFeatureDecoder

    torch.manual_seed(0)
    v18 = LidarFeatureDecoder(OmegaConf.create(DECODER_CFG)).double()
    config = {key: value for key, value in DECODER_CFG.items() if key != "num_sensors"}
    v18_1 = LidarFeatureDecoder(
        OmegaConf.create({**config, "sensor_embedding_dim": 0})
    ).double()
    state = v18.state_dict()
    first = state["mlp.0.weight"]
    folded = {
        name: value for name, value in state.items()
        if name not in ("sensor_embedding.weight", "mlp.0.weight", "mlp.0.bias")
    }
    folded["mlp.0.weight"] = first[:, :34]
    folded["mlp.0.bias"] = (
        state["mlp.0.bias"] + first[:, 34:] @ state["sensor_embedding.weight"][0]
    )
    v18_1.load_state_dict(folded, strict=True)
    feature = torch.randn(30, 5, 7, dtype=torch.float64)
    ray = F.normalize(torch.randn(3, 5, 7, dtype=torch.float64), dim=0)
    depth = 100 * torch.rand(1, 5, 7, dtype=torch.float64)
    alpha = torch.rand(1, 5, 7, dtype=torch.float64)
    for expected, actual in zip(
        v18(feature, ray, depth, alpha), v18_1(feature, ray, depth, alpha)
    ):
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("overrides, message", [
    ({"hidden_dims": []}, "hidden_dims"),
    ({"feature_dim": 0}, "feature_dim"),
    ({"range_scale_m": 0.0}, "range_scale_m"),
    ({"sensor_embedding_dim": -1}, "sensor_embedding_dim"),
    ({"num_sensors": 0}, "num_sensors"),
])
def test_decoder_rejects_an_invalid_config(overrides, message):
    from src.models.module.lidar_decoder import LidarFeatureDecoder

    with pytest.raises(ValueError, match=message):
        LidarFeatureDecoder(OmegaConf.create({**DECODER_CFG, **overrides}))


def _gt_camera(seed=3):
    """The test camera with a LiDAR GT: returns at 5-40 m, 30% dropped rays."""
    camera = _camera()
    generator = torch.Generator().manual_seed(seed)
    shape = (1, int(camera.image_height), int(camera.image_width))
    depth = 5 + 35 * torch.rand(shape, generator=generator)
    depth[torch.rand(shape, generator=generator) < 0.3] = 0.0
    camera.pts_depth = depth
    camera.pts_intensity = torch.rand(shape, generator=generator)
    return camera


def _v18_gaussians(pc, seed=4):
    generator = torch.Generator().manual_seed(seed)
    count = pc["position"].shape[0]
    leaves = {
        "position": pc["position"],
        "opacity": pc["opacity"],
        "scaling": pc["scales"][:, :2],
        "rotation": pc["rotations"],
        "feature": 0.5 * torch.randn(count, 30, generator=generator).cuda(),
    }
    return {key: value.detach().clone().requires_grad_(True)
            for key, value in leaves.items()}


def _decoder_loss(renders):
    """V15.4's intensity L1 on returns and raydrop BCE on every pixel."""
    valid = renders["gt_depth"] > 0
    intensity = F.l1_loss(
        renders["intensity_sh"][valid], renders["gt_intensity_sh"][valid]
    )
    raydrop = F.binary_cross_entropy(
        renders["raydrop"].clamp(1e-7, 1 - 1e-7), renders["gt_raydrop"]
    )
    return 0.2 * intensity + 0.5 * raydrop


@requires_cuda
def test_v18_render_losses_reach_decoder_features_opacity_and_geometry():
    """No stage between the losses and the Gaussians cuts the gradient."""
    from src.models.module.m3_g2p import DynamicGausRender

    torch.manual_seed(0)
    g2p = DynamicGausRender(
        OmegaConf.create({**_RENDER_CFG, "lidar_decoder": DECODER_CFG})
    ).cuda().train()
    leaves = _v18_gaussians(_scene())
    count = leaves["position"].shape[0]
    gaussians = [{
        **leaves,
        "velocity": torch.zeros(count, 3, device="cuda"),
        "source_time_sec": torch.zeros(count, device="cuda"),
    }]
    renders = g2p(gaussians, {"cameras": [[_gt_camera()]]}, compute_points=False)
    assert renders["intensity_sh"].shape == renders["raydrop"].shape == (1, 1, 1, 32, 512)
    # A ray that hits no Gaussian returns nothing, as in V15.4.
    g2p.eval()
    with torch.no_grad():
        alpha = g2p(gaussians, {"cameras": [[_gt_camera()]]},
                    compute_points=False)["alpha"]
    g2p.train()
    empty = alpha == 0
    assert int(empty.sum()) > 0
    assert bool((renders["raydrop"].detach()[empty] == 1).all())
    assert bool((renders["intensity_sh"].detach()[empty] == 0).all())
    _decoder_loss(renders).backward()
    for name, parameter in g2p.lidar_decoder.named_parameters():
        assert parameter.grad is not None and parameter.grad.abs().max() > 0, name
    for name in ("feature", "opacity", "position", "scaling", "rotation"):
        assert leaves[name].grad.abs().max() > 0, name


@requires_cuda
def test_v18_decoder_reads_median_depth_without_moving_geometry_through_it():
    """r / R conditions the decoder but is detached; f' still trains geometry."""
    from src.models.gaussian_renderer import render
    from src.models.module.lidar_decoder import LidarFeatureDecoder

    torch.manual_seed(0)
    decoder = LidarFeatureDecoder(OmegaConf.create(DECODER_CFG)).cuda()
    leaves = _v18_gaussians(_scene())
    package = render(
        viewpoint_camera=_camera(), cfg=_RENDER_CFG,
        pc={
            "position": leaves["position"], "opacity": leaves["opacity"],
            "scales": torch.cat([leaves["scaling"], leaves["scaling"][:, :1] * 0], 1),
            "rotations": leaves["rotation"], "feature": leaves["feature"],
        },
        bg_color=torch.tensor([0.0, 0.0, 0.0, 1.0]).cuda(), input_timestamp=0.0,
    )
    depth = package["depth_median"]
    depth.retain_grad()
    intensity, raydrop = decoder(
        package["feature"], package["ray_dir"], depth, package["alpha"]
    )
    # The decoder reads the range: a different median depth changes its output.
    moved, _ = decoder(
        package["feature"], package["ray_dir"], depth + 10.0, package["alpha"]
    )
    assert not torch.allclose(moved, intensity)
    (intensity.sum() + raydrop.sum()).backward()
    assert depth.grad is None
    # Through the blended feature's weights the decoder losses still reach
    # every Gaussian attribute.
    for name in ("feature", "opacity", "position", "scaling", "rotation"):
        assert leaves[name].grad.abs().max() > 0, name
