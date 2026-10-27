"""V18.2: a LiDAR4D-style U-Net decodes the rendered feature, ray and depth.

V18.2's G2P decoder reads, for the whole view at once,
[f' (32), the ray in the rendering LiDAR's own axes (3), median depth / R (1)]
through LiDAR4D's raydrop-refinement U-Net shape (1x1 input conv, Down blocks
to 1/8, self-attention, Up blocks, norm-ReLU-1x1 head) with GroupNorm instead
of BatchNorm and the attention heads merged per position, and composites the
decoded surface over the no-return background by alpha exactly as V18.
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from src.experiment_config import DYNAMIC_VARIANT_V18_2, compose_fresh_config

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs the CUDA rasterizer"
)

_RENDER_CFG = {"neg_fov": True, "sh_degree": 3, "scale_factor": 1.0}
_RING = [
    -30.67, -29.33, -28.00, -26.66, -25.33, -24.00, -22.67, -21.33,
    -20.00, -18.67, -17.33, -16.00, -14.67, -13.33, -12.00, -10.67,
    -9.33, -8.00, -6.66, -5.33, -4.00, -2.67, -1.33, 0.00,
    1.33, 2.67, 4.00, 5.33, 6.67, 8.00, 9.33, 10.67,
]
# The loaders' LiDAR -> render camera axis maps: nuScenes' LIDAR_TOP (x right,
# y forward, z up; src/dataloader/nuscenes/camera.py) and KITTI-360's Velodyne
# (x forward, y left, z up; src/dataloader/kitti360/reader.py).
_NUSCENES_CAMERA_FROM_LIDAR = np.array(
    [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=np.float32
)
_KITTI360_CAMERA_FROM_LIDAR = np.array(
    [[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]], dtype=np.float32
)


def _unet_module():
    """``lidar_unet_decoder.py`` by path: ``src.models.module`` loads the CUDA renderer."""
    path = Path(__file__).parents[1] / "src/models/module/lidar_unet_decoder.py"
    spec = importlib.util.spec_from_file_location("v18_2_lidar_unet_decoder", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


UNET = _unet_module()


def _v18_2_decoder_cfg():
    config, _source = compose_fresh_config(
        OmegaConf.from_dotlist([f"model.variant={DYNAMIC_VARIANT_V18_2}"])
    )
    return config.g2p.lidar_decoder


def _inputs(height, width, seed=0):
    generator = torch.Generator().manual_seed(seed)
    feature = torch.randn(32, height, width, generator=generator)
    ray = F.normalize(torch.randn(3, height, width, generator=generator), dim=0)
    depth = 80 * torch.rand(1, height, width, generator=generator)
    alpha = torch.rand(1, height, width, generator=generator)
    return feature, ray, depth, alpha


def test_v18_2_decoder_is_lidar4ds_unet_shape_with_groupnorm():
    decoder = UNET.LidarUNetDecoder(_v18_2_decoder_cfg())
    assert sum(p.numel() for p in decoder.parameters()) == 6_464_194
    # LiDAR4D's InConv: one 1x1 conv with bias, nothing after it.
    assert isinstance(decoder.inc, torch.nn.Conv2d)
    assert (decoder.inc.in_channels, decoder.inc.out_channels) == (36, 64)
    assert decoder.inc.kernel_size == (1, 1) and decoder.inc.bias is not None
    assert [block.conv.double_conv[7].out_channels for block in decoder.down] == [128, 256, 256]
    assert decoder.attention.proj.in_channels == 256
    assert [block.conv.double_conv[7].out_channels for block in decoder.up] == [128, 64, 64]
    assert decoder.outc.conv[2].out_channels == 2
    norms = [m for m in decoder.modules() if isinstance(m, torch.nn.modules.batchnorm._NormBase)]
    groups = [m for m in decoder.modules() if isinstance(m, torch.nn.GroupNorm)]
    assert norms == [] and groups and {m.num_groups for m in groups} == {32}
    # GroupNorm keeps no running statistics: training and evaluation normalize alike.
    assert list(decoder.buffers()) == []


@pytest.mark.parametrize("height,width", [(66, 1030), (32, 1085)])
def test_decoder_composites_the_decoded_surface_by_alpha(height, width):
    """KITTI-360's and nuScenes' grids (odd sizes after pooling) decode in shape."""
    torch.manual_seed(0)
    decoder = UNET.LidarUNetDecoder(_v18_2_decoder_cfg()).eval()
    feature, ray, depth, alpha = _inputs(height, width)
    alpha[:, :4] = 0.0
    alpha[:, -4:] = 1.0
    with torch.no_grad():
        intensity, raydrop = decoder(feature, ray, depth, alpha)
        surface_i, surface_p = decoder.surface(feature, ray, depth)
    assert intensity.shape == raydrop.shape == surface_i.shape == (1, height, width)
    # A ray that hits no Gaussian returns nothing; a covered one is the surface.
    assert bool((raydrop[:, :4] == 1).all()) and bool((intensity[:, :4] == 0).all())
    torch.testing.assert_close(intensity[:, -4:], surface_i[:, -4:])
    torch.testing.assert_close(raydrop[:, -4:], surface_p[:, -4:])
    torch.testing.assert_close(intensity, alpha * surface_i)
    torch.testing.assert_close(raydrop, 1 - alpha * (1 - surface_p))


def test_self_attention_merges_each_positions_heads_back_in_place():
    """Standard multi-head attention, unlike LiDAR4D's ``view(B, H, W, C)``."""
    torch.manual_seed(0)
    block = UNET.SelfAttention(256, groups=32).eval()
    x = torch.randn(2, 256, 4, 7)
    with torch.no_grad():
        out = block(x)
        batch, channels, height, width = x.shape
        q, k, v = block.proj_qkv(block.norm(x)).flatten(2).chunk(3, dim=1)
        heads = []
        for head in range(8):
            rows = slice(32 * head, 32 * (head + 1))
            weights = torch.softmax(
                torch.einsum("bcq,bck->bqk", q[:, rows], k[:, rows]) / math.sqrt(32), dim=-1
            )
            heads.append(torch.einsum("bqk,bck->bcq", weights, v[:, rows]))
        merged = torch.cat(heads, dim=1).reshape(batch, channels, height, width)
        torch.testing.assert_close(out, x + block.proj(merged))
        # LiDAR4D's merge of the same per-head outputs scrambles positions.
        per_head = torch.stack(heads, dim=1).transpose(2, 3)  # [B, heads, HW, C/heads]
        lidar4d = x + block.proj(per_head.reshape(batch, height, width, channels).permute(0, 3, 1, 2))
        assert not torch.allclose(out, lidar4d, atol=1e-4)


def test_dropout_acts_in_training_only():
    torch.manual_seed(0)
    decoder = UNET.LidarUNetDecoder(_v18_2_decoder_cfg())
    feature, ray, depth, alpha = _inputs(32, 64)
    with torch.no_grad():
        decoder.eval()
        assert torch.equal(decoder(feature, ray, depth, alpha)[0], decoder(feature, ray, depth, alpha)[0])
        decoder.train()
        assert not torch.equal(decoder(feature, ray, depth, alpha)[0], decoder(feature, ray, depth, alpha)[0])


def test_decoder_reads_depth_detached_and_trains_through_the_feature():
    torch.manual_seed(0)
    decoder = UNET.LidarUNetDecoder(_v18_2_decoder_cfg()).eval()
    feature, ray, depth, alpha = _inputs(32, 64)
    feature.requires_grad_(True)
    depth.requires_grad_(True)
    intensity, raydrop = decoder(feature, ray, depth, alpha)
    with torch.no_grad():
        moved, _ = decoder(feature, ray, depth + 10.0, alpha)
    assert not torch.allclose(moved, intensity)
    (intensity.sum() + raydrop.sum()).backward()
    assert depth.grad is None
    assert feature.grad.abs().max() > 0
    for name, parameter in decoder.named_parameters():
        assert parameter.grad is not None and parameter.grad.abs().max() > 0, name


@pytest.mark.parametrize("overrides,message", [
    ({"channels": [64]}, "at least two positive"),
    ({"norm_groups": 48}, "must divide every width"),
    ({"channels": [64, 128, 252], "norm_groups": 4}, "divisible by 8"),
    ({"range_scale_m": 0.0}, "range_scale_m must be positive"),
    ({"feature_dim": 0}, "feature_dim must be a positive integer"),
])
def test_decoder_rejects_an_invalid_config(overrides, message):
    cfg = {
        **OmegaConf.to_container(_v18_2_decoder_cfg(), resolve=True), **overrides,
    }
    with pytest.raises(ValueError, match=message):
        UNET.LidarUNetDecoder(OmegaConf.create(cfg))


def _camera(viewmatrix, width=512):
    from src.models.utils.camera import Camera

    height = len(_RING)
    return Camera(
        R=np.eye(3), T=np.zeros(3), vfov=(-30.67, 10.67), hfov=(-180.0, 180.0),
        viewmatrix=viewmatrix, resolution=(width, height), timestamp_sec=0.0,
        ring_to_elevation_deg=_RING,
        pts_depth=torch.zeros(1, height, width), pts_intensity=torch.zeros(1, height, width),
    )


def _lidar_from_reference(yaw, translation):
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, :3] = np.array(
        [[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]],
        dtype=np.float32,
    )
    matrix[:3, 3] = translation
    return matrix


@requires_cuda
def test_lidar_frame_rays_are_one_map_per_sensor_in_shared_axes():
    """The same unit vector per pixel at every pose; x right, y down, z forward."""
    from src.models.gaussian_renderer import (
        panorama_camera_ray_directions,
        panorama_ray_directions,
    )

    rays = []
    for camera_from_lidar in (_NUSCENES_CAMERA_FROM_LIDAR, _KITTI360_CAMERA_FROM_LIDAR):
        for yaw, translation in ((0.4, (1.5, -0.7, -1.8)), (-1.1, (-6.0, 2.0, 0.3))):
            camera = _camera(camera_from_lidar @ _lidar_from_reference(yaw, translation))
            ray = panorama_camera_ray_directions(camera, "cpu", torch.float32)
            torch.testing.assert_close(ray.norm(dim=0), torch.ones_like(ray[0]))
            # Rotated into the reference frame it is the world-frame ray map.
            world = panorama_ray_directions(camera, "cpu", torch.float32)
            torch.testing.assert_close(
                torch.einsum("ij,jhw->ihw", camera.world_view_transform[:3, :3], ray), world
            )
            rays.append(ray)
    for ray in rays[1:]:
        torch.testing.assert_close(ray, rays[0])
    # Rows run from the highest elevation down; columns 256 and 384 sit at
    # azimuth 0 and +90 degrees.
    level, ahead, right = sorted(_RING, reverse=True).index(0.00), 256, 384
    torch.testing.assert_close(rays[0][:, level, ahead], torch.tensor([0.0, 0.0, 1.0]))
    torch.testing.assert_close(rays[0][:, level, right], torch.tensor([1.0, 0.0, 0.0]), atol=1e-6, rtol=0)
    # In each sensor's own axes that is its forward and its right side.
    for camera_from_lidar, forward, side in (
        (_NUSCENES_CAMERA_FROM_LIDAR, (0, 1, 0), (1, 0, 0)),
        (_KITTI360_CAMERA_FROM_LIDAR, (1, 0, 0), (0, -1, 0)),
    ):
        lidar_from_camera = torch.from_numpy(camera_from_lidar[:3, :3].T.copy())
        torch.testing.assert_close(lidar_from_camera @ rays[0][:, level, ahead], torch.tensor(forward, dtype=torch.float32))
        torch.testing.assert_close(
            lidar_from_camera @ rays[0][:, level, right], torch.tensor(side, dtype=torch.float32),
            atol=1e-6, rtol=0,
        )


def _scene(count=3000, seed=0):
    """Gaussians with a 32-channel feature around the sensor, reference frame."""
    generator = torch.Generator().manual_seed(seed)
    azimuth = torch.rand(count, generator=generator) * 2 * math.pi
    distance = 3 + 40 * torch.rand(count, generator=generator)
    elevation = torch.deg2rad(-25 + 30 * torch.rand(count, generator=generator))
    sensor = torch.stack([
        distance * elevation.cos() * azimuth.cos(),
        distance * elevation.cos() * azimuth.sin(),
        distance * elevation.sin(),
    ], 1)
    to_reference = torch.from_numpy(np.linalg.inv(_lidar_from_reference(0.4, (1.5, -0.7, -1.8))))
    scaling = torch.log(torch.expm1(0.1 + 0.5 * torch.rand(count, 2, generator=generator)))
    leaves = {
        "position": sensor @ to_reference[:3, :3].T + to_reference[:3, 3],
        "opacity": -0.5 + torch.randn(count, 1, generator=generator),
        "scaling": scaling,
        "rotation": F.normalize(torch.randn(count, 4, generator=generator), dim=1),
        "feature": 0.5 * torch.randn(count, 32, generator=generator),
    }
    return {key: value.cuda().requires_grad_(True) for key, value in leaves.items()}


def _gt_camera(seed=3):
    """A nuScenes-axis camera with GT returns at 5-40 m and 30% dropped rays."""
    camera = _camera(_NUSCENES_CAMERA_FROM_LIDAR @ _lidar_from_reference(0.4, (1.5, -0.7, -1.8)))
    generator = torch.Generator().manual_seed(seed)
    shape = (1, int(camera.image_height), int(camera.image_width))
    depth = 5 + 35 * torch.rand(shape, generator=generator)
    depth[torch.rand(shape, generator=generator) < 0.3] = 0.0
    camera.pts_depth = depth
    camera.pts_intensity = torch.rand(shape, generator=generator)
    return camera


@requires_cuda
def test_v18_2_render_losses_reach_the_unet_and_every_gaussian_attribute():
    """G2P hands the U-Net LiDAR-frame rays; no stage cuts the gradient."""
    from unittest import mock

    from src.models.gaussian_renderer import panorama_camera_ray_directions
    from src.models.module.lidar_unet_decoder import LidarUNetDecoder
    from src.models.module.m3_g2p import DynamicGausRender

    torch.manual_seed(0)
    g2p = DynamicGausRender(OmegaConf.create({
        **_RENDER_CFG,
        "lidar_decoder": OmegaConf.to_container(_v18_2_decoder_cfg(), resolve=True),
    })).cuda().train()
    assert isinstance(g2p.lidar_decoder, LidarUNetDecoder)
    leaves = _scene()
    count = leaves["position"].shape[0]
    gaussians = [{
        **leaves,
        "velocity": torch.zeros(count, 3, device="cuda"),
        "source_time_sec": torch.zeros(count, device="cuda"),
    }]
    camera = _gt_camera()
    decode = g2p.lidar_decoder.forward
    with mock.patch.object(g2p.lidar_decoder, "forward", wraps=decode) as spy:
        renders = g2p(gaussians, {"cameras": [[camera]]}, compute_points=False)
    (_feature, ray, _depth, _alpha), _kwargs = spy.call_args
    torch.testing.assert_close(ray, panorama_camera_ray_directions(camera, "cuda", torch.float32))
    assert renders["intensity_sh"].shape == renders["raydrop"].shape == (1, 1, 1, 32, 512)
    g2p.eval()
    with torch.no_grad():
        alpha = g2p(gaussians, {"cameras": [[camera]]}, compute_points=False)["alpha"]
    g2p.train()
    empty = alpha == 0
    assert int(empty.sum()) > 0
    assert bool((renders["raydrop"].detach()[empty] == 1).all())
    assert bool((renders["intensity_sh"].detach()[empty] == 0).all())
    valid = renders["gt_depth"] > 0
    loss = 0.5 * F.l1_loss(renders["intensity_sh"][valid], renders["gt_intensity_sh"][valid]) + F.binary_cross_entropy(
        renders["raydrop"].clamp(1e-7, 1 - 1e-7), renders["gt_raydrop"]
    )
    loss.backward()
    for name, parameter in g2p.lidar_decoder.named_parameters():
        assert parameter.grad is not None and parameter.grad.abs().max() > 0, name
    for name in ("feature", "opacity", "position", "scaling", "rotation"):
        assert leaves[name].grad.abs().max() > 0, name
