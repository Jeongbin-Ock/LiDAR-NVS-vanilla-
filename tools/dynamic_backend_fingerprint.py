"""Deterministic construction/forward fingerprint for every Dynamic 2DGS backend.

Builds each registered dynamic variant's backend exactly the way ``Point2Gaus``
does, runs one CPU forward on fixed synthetic input, and prints a hash of the
parameter layout and of every returned tensor.  Two runs of this script across a
refactor must print identical lines for every variant whose behaviour is meant
to be preserved.

    python tools/dynamic_backend_fingerprint.py > before.txt
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiment_config import loader as cl  # noqa: E402
from src.models.module import dynamic_gaussian as dg  # noqa: E402

BACKENDS = {
    cl.DYNAMIC_VARIANT_V11: dg.MaxSpeedBarrierVelocityGaussianBackend,
    cl.DYNAMIC_VARIANT_V11_1: dg.SingleGaussianBarrierVelocityGaussianBackend,
    cl.DYNAMIC_VARIANT_V11_2: dg.FinalFeatureBarrierVelocityGaussianBackend,
    cl.DYNAMIC_VARIANT_V11_3: dg.SeedConditionedBarrierVelocityGaussianBackend,
    cl.DYNAMIC_VARIANT_V11_4: dg.MaxSpeedBarrierVelocityGaussianBackend,
    cl.DYNAMIC_VARIANT_V11_5: dg.SeedConditionedBarrierVelocityGaussianBackend,
    cl.DYNAMIC_VARIANT_V11_6: dg.SeedConditionedBarrierVelocityGaussianBackend,
    cl.DYNAMIC_VARIANT_V12: dg.LearnableMaxSpeedBarrierVelocityGaussianBackend,
    cl.DYNAMIC_VARIANT_V12_1: (
        dg.FixedMotionGainMaxSpeedBarrierVelocityGaussianBackend
    ),
    cl.DYNAMIC_VARIANT_V12_2: (
        dg.LayerwiseSoftplusMaxSpeedBarrierVelocityGaussianBackend
    ),
    cl.DYNAMIC_VARIANT_V13: dg.CosineRoPEVelocityGaussianBackend,
    cl.DYNAMIC_VARIANT_V13_1: dg.SeedConditionedCosineRoPEVelocityGaussianBackend,
    cl.DYNAMIC_VARIANT_V13_2: dg.RawQKRoPEVelocityGaussianBackend,
    cl.DYNAMIC_VARIANT_V13_3: dg.RMSNormRoPEVelocityGaussianBackend,
    cl.DYNAMIC_VARIANT_V13_4: dg.RMSNormFeatureRoPEVelocityGaussianBackend,
    cl.DYNAMIC_VARIANT_V14: dg.MaxSpeedBarrierVelocityGaussianBackend,
    cl.DYNAMIC_VARIANT_V14_1: (
        dg.GlobalSoftplusMaxSpeedBarrierVelocityGaussianBackend
    ),
}

DIM = 72
HEADS = 6
LAYERS = 2
TOKENS_PER_FRAME = 5


def _digest(tensor):
    value = tensor.detach().to(torch.float64).contiguous()
    return hashlib.sha1(value.numpy().tobytes()).hexdigest()[:16]


def _shrink(cfg):
    """Keep every semantic key; shrink only widths so a CPU forward is cheap."""
    temporal = cfg.temporal
    temporal.implementation = "auto"
    temporal.layers = LAYERS
    temporal.num_heads = HEADS
    temporal.mlp_ratio = 2
    if "match_chunk_size" in temporal:
        temporal.match_chunk_size = 4
    return cfg


def _inputs(variant, backend, generator):
    """Two endpoint frames for one sample, matching the P2G forward contract."""
    n_frames, n = 2, TOKENS_PER_FRAME
    total = n_frames * n
    feature = torch.randn(total, DIM, generator=generator)
    token_position = torch.randn(total, 3, generator=generator) * 4.0
    adaptive = hasattr(backend.gaussian_head, "k_max")
    if adaptive:
        k_max = backend.gaussian_head.k_max
        seed = torch.randn(total, k_max, k_max, 3, generator=generator)
        delta = torch.randn(total, k_max, k_max, 3, generator=generator) * 0.1
    else:
        # A fixed count fills exactly that many slots. Repeated coordinates
        # are the real contract for seed_mode=medoid/token, and the per-slot
        # parameter blocks are what separate them.
        slots = backend.gaussians_per_token
        seed = token_position.clone().unsqueeze(1).repeat(1, slots, 1)
        delta = (
            torch.randn(total, slots, 3, generator=generator) * 0.1
            if getattr(backend, "seed_conditioned_head", False) else None
        )
    token_offset = torch.tensor([n, total], dtype=torch.long)
    frame_batch_idx = torch.zeros(n_frames, dtype=torch.long)
    pose = [[torch.eye(4), torch.eye(4)]]
    kwargs = {}
    if delta is not None:
        kwargs["seed_delta_sensor"] = delta
    return (
        feature, token_position, seed, token_offset, frame_batch_idx, pose,
        [torch.tensor([0.0, 1.0])], [torch.tensor([0.0, 0.8])], [0.8],
    ), kwargs


def main():
    for variant, backend_cls in BACKENDS.items():
        overlay = OmegaConf.load(cl.VARIANT_CONFIG_PATHS[variant])
        cfg = _shrink(overlay.dynamic_2dgs)
        gs_params = OmegaConf.create(
            {"shs": 4, "opacity": 1, "scaling": 2, "rotation": 4, "offset": 3}
        )
        kwargs = {}
        count = OmegaConf.select(overlay, "p2g.grid_query")
        count_mode = str(
            OmegaConf.select(overlay, "p2g.grid_query.count_mode") or "legacy"
        ).lower()
        if count is not None and count_mode == "learned_gumbel":
            kwargs["gaussian_count_cfg"] = count
            seed_mode = OmegaConf.select(
                overlay, "p2g.grid_query.learned_count.seed_mode"
            )
        else:
            seed_mode = OmegaConf.select(overlay, "p2g.grid_query.seed_mode")
            if count is not None:
                kwargs["gaussians_per_token"] = int(
                    OmegaConf.select(overlay, "p2g.grid_query.K_max") or 1
                )
        # As in Point2Gaus: the seed mode decides whether the Gaussian head
        # reads seed deltas (range_quantile) or the token feature alone.
        kwargs["seed_mode"] = seed_mode or "range_quantile"
        torch.manual_seed(0)
        backend = backend_cls(cfg, gs_params, dim=DIM, offset_bound=0.8, **kwargs)
        backend.eval()

        state = backend.state_dict()
        layout = hashlib.sha1(
            "\n".join(
                f"{k}:{tuple(v.shape)}" for k, v in sorted(state.items())
            ).encode()
        ).hexdigest()[:16]
        params = sum(v.numel() for v in state.values())
        weights = hashlib.sha1(
            b"".join(_digest(v).encode() for _, v in sorted(state.items()))
        ).hexdigest()[:16]

        generator = torch.Generator().manual_seed(1234)
        args, forward_kwargs = _inputs(variant, backend, generator)
        with torch.no_grad():
            out = backend(*args, **forward_kwargs)
        rows = []
        for item in out["batch_gaussians"]:
            if item is None:
                continue
            for name, value in sorted(item.items()):
                if torch.is_tensor(value):
                    rows.append(f"{name}={tuple(value.shape)}:{_digest(value)}")
        forward = hashlib.sha1("\n".join(rows).encode()).hexdigest()[:16]
        print(
            f"{variant:44s} layout={layout} weights={weights} "
            f"params={params:9d} forward={forward} n_out={len(rows)}"
        )


if __name__ == "__main__":
    main()
