"""LiDAR4D / GS-LiDAR-style temporal NVS evaluation on nuScenes or KITTI-360.

KITTI-360: ``eval_config=config/eval/kitti360_inference.yaml`` selects the ten
official drive-0000 clips and their 40 held-out frames. Sensor timing, render
geometry, and target labels come from the dataset; model config comes from the
checkpoint. This feed-forward evaluation uses two sweeps per target.

Protocol (1-second temporal novel-view synthesis):
  per sequence, for each target second T in {1,2,3,4}:
      input  = LiDAR at (T-0.5)s and (T+0.5)s
      render = predict the panorama at T s and compare to the GT frame at T s.

Metrics follow the official GS-LiDAR code (utils/metrics_utils.py):
  depth / intensity : RMSE, MedAE, LPIPS(alex), SSIM, PSNR  (raydrop-masked, full map)
  raydrop           : RMSE, Acc, F1
  points            : Chamfer distance, F-score@0.05 (+ precision, recall)
Our depth is already in meters (g2p.scale_factor == 1.0), so GS-LiDAR's
`scale_factor` rescale is NOT applied (scale=1.0).

Usage:
  python test_lidar4d_nvs.py test.ckpt_path=/path/to/epoch=XX.ckpt \
      [data.mode=bbox] [out_dir=outputs/lidar4d_eval] [device='[4]'] \
      [test.velocity_source=total|init|match|zero] \
      [test.input_offsets_sec='[-0.5,0.5]'] [test.shuffle_orders=false|true]

  test.input_offsets_sec overrides the two input offsets relative to target T
  (default [-0.5, 0.5], the protocol above). E.g. '[-0.1,0.2]' feeds the model
  (T-0.1)s / (T+0.2)s instead of (T-0.5)s / (T+0.5)s -- must straddle T
  (before < 0 < after) and land on the 0.1 s sampling grid.
"""
from __future__ import annotations

import json
import os
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image

from src.model_wrapper import ModelWrapper
from src.experiment_config import resolve_eval_config
from src.dataloader.nuscenes.legacy import multiframe_collate_fn
from src.dataloader.lidar4d_test import build_lidar4d_test_dataset
from src.eval.output_paths import resolve_lidar4d_output_dir
from src.eval.gslidar_metrics import (
    MetricBackends, depth_errors, intensity_errors, raydrop_errors, point_metrics,
)
from src.eval.gaussian_viz import (
    save_sequence_html,
    surfels_from_output, save_sequence_surfel_html,
    velocity_transport_from_output,
)
from src.eval.gaussian_stats import collect_window_stats, analyze_gaussian_sizes
from src.eval.depth_diagnostics import depth_error_breakdown, summarize_depth_breakdowns
from src.models.utils.graphics_utils import lidar4d_range_image_to_points
from src.models.utils.render import visualize_depth

RAYDROP_THRESHOLD = 0.5
EVAL_MAX_DEPTH_M = 80.0


def to_device(obj, device):
    """Move all tensors (leave Camera/other objects untouched -- render moves
    their tensors internally)."""
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_device(v, device) for v in obj]
    if isinstance(obj, tuple):
        return tuple(to_device(v, device) for v in obj)
    return obj


def load_model(cfg, ckpt_path, device, *, allow_checkpoint_mismatch=False):
    model = ModelWrapper(cfg)
    if ckpt_path:
        blob = torch.load(ckpt_path, map_location="cpu")
        state = blob.get("state_dict", blob) if isinstance(blob, dict) else blob
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[ckpt] {ckpt_path}\n       loaded "
              f"(missing={len(missing)}, unexpected={len(unexpected)})")
        if (missing or unexpected) and not allow_checkpoint_mismatch:
            raise RuntimeError(
                "Checkpoint/config mismatch. Refusing to evaluate partial weights; "
                "pass allow_checkpoint_mismatch=true only for an intentional "
                f"ablation. missing={list(missing)}, unexpected={list(unexpected)}"
            )
    else:
        print("[ckpt] WARNING: no checkpoint -> randomly initialized weights")
    model.eval().to(device)
    return model


def _viz_depth_black_holes(depth_1hw, near=2, far=50):
    """Colorize depth (turbo), but paint no-return pixels (depth==0) BLACK so
    they are not confused with near-range red (the -log curve maps both 0 and
    near depth to turbo's red end)."""
    vis = visualize_depth(depth_1hw, near=near, far=far)      # [3,H,W]
    keep = (depth_1hw[0] > 0).to(vis.dtype).to(vis.device)   # [H,W]
    return vis * keep


def _binary_raydrop(raydrop, threshold=RAYDROP_THRESHOLD):
    """Return 1 for predicted/GT drop and 0 for non-drop (keep)."""
    return (raydrop > float(threshold)).to(raydrop.dtype)


def prediction_points_ref(depth, raydrop, camera):
    """Back-project raydrop-kept prediction ranges into the Gaussian ref frame.

    ``lidar4d_range_image_to_points`` returns points in the target panorama's
    swapped camera frame.  The surfel meshes and raw LiDAR overlays live in the
    window's frame-0 reference coordinates, so apply the camera-to-ref matrix
    before serializing the point trace.
    """
    points_camera = lidar4d_range_image_to_points(
        depth,
        camera.vfov,
        camera.hfov,
        row_to_theta=camera.row_to_theta,
        raydrop=raydrop,
        raydrop_threshold=RAYDROP_THRESHOLD,
        max_range=EVAL_MAX_DEPTH_M,
        row_origin=getattr(camera, "row_origin", None),
    )
    c2w = camera.c2w.to(device=points_camera.device, dtype=points_camera.dtype)
    return points_camera @ c2w[:3, :3].T + c2w[:3, 3]


def save_viz(
    out_png,
    depth_keep,
    gt_depth,
    intensity_keep,
    gt_intensity,
    raydrop,
    gt_raydrop,
):
    pred_drop = _binary_raydrop(raydrop)
    gt_drop = _binary_raydrop(gt_raydrop)
    rows = [
        _viz_depth_black_holes(depth_keep),                   # pred depth  [3,H,W]
        _viz_depth_black_holes(gt_depth),                     # gt depth (holes=black)
        intensity_keep.clamp(0, 1).repeat(3, 1, 1),           # pred intensity
        gt_intensity.clamp(0, 1).repeat(3, 1, 1),             # gt intensity
        pred_drop.repeat(3, 1, 1),                            # pred drop, >0.5 = white
        gt_drop.repeat(3, 1, 1),                              # GT drop, 1 = white
    ]
    grid = make_grid(torch.stack([r.detach().cpu() for r in rows], dim=0), nrow=1)
    save_image(grid, str(out_png))


def _agg(window_metrics, key, sub):
    vals = [w[key][sub] for w in window_metrics if not np.isnan(w[key][sub])]
    return float(np.mean(vals)) if vals else float("nan")


def summarize(window_metrics):
    out = {}
    for key, subs in (("depth", ["rmse", "medae", "lpips", "ssim", "psnr"]),
                      ("intensity", ["rmse", "medae", "lpips", "ssim", "psnr"]),
                      ("raydrop", ["rmse", "acc", "f1"]),
                      ("points", ["cd", "fscore", "precision", "recall"])):
        out[key] = {s: _agg(window_metrics, key, s) for s in subs}
    return out


def select_render_velocity(batch_gaussians, velocity_source: str):
    """Select the checkpoint velocity component used by the renderer.

    ``total`` preserves the trained model output. ``init`` uses the dustbin-aware
    proposal, while ``match`` uses its conditional matched velocity before the
    dustbin probability scale. Both are inference-only component ablations.
    ``zero`` freezes every Gaussian at its source position, measuring how much
    the velocity field contributes to reconstruction at all.
    """
    velocity_source = str(velocity_source).lower()
    if velocity_source not in {"total", "init", "match", "zero"}:
        raise ValueError(
            "test.velocity_source must be one of "
            "{'total', 'init', 'match', 'zero'}, got "
            f"{velocity_source!r}"
        )
    if velocity_source == "total":
        return batch_gaussians

    if velocity_source == "zero":
        selected = []
        for item in batch_gaussians:
            if item is None:
                selected.append(None)
                continue
            velocity = item.get("velocity")
            if not torch.is_tensor(velocity):
                raise KeyError("test.velocity_source=zero requires velocity")
            selected.append({**item, "velocity": torch.zeros_like(velocity)})
        return selected

    selected = []
    for batch_index, item in enumerate(batch_gaussians):
        if item is None:
            selected.append(None)
            continue
        velocity = item.get("velocity")
        source_key = f"velocity_{velocity_source}"
        selected_velocity = item.get(source_key)
        if not torch.is_tensor(selected_velocity):
            raise KeyError(
                f"test.velocity_source={velocity_source} requires {source_key} "
                "in every "
                f"Gaussian batch item; missing at batch index {batch_index}"
            )
        if (
            not torch.is_tensor(velocity)
            or selected_velocity.shape != velocity.shape
        ):
            raise ValueError(
                f"{source_key} must be a tensor aligned one-to-one with velocity"
            )
        selected.append({**item, "velocity": selected_velocity})
    return selected


# ---------------------------------------------------------------------------
@torch.no_grad()
def main(cfg, config_source="unspecified"):
    seed = int(cfg.get("seed", 0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = f"cuda:{int(cfg.device[0])}" if len(cfg.device) else "cuda:0"
    torch.cuda.set_device(device)
    out_dir = resolve_lidar4d_output_dir(cfg)
    cfg.out_dir = str(out_dir)
    viz_dir = out_dir / "viz"
    viz_dir.mkdir(parents=True, exist_ok=True)
    effective_config_path = out_dir / "effective_config.yaml"
    OmegaConf.save(config=cfg, f=str(effective_config_path), resolve=True)

    print(
        f"[setup] device={device} seed={seed} "
        f"mode={cfg.data.mode} out_dir={out_dir}"
    )
    print(f"[config] source={config_source}")
    print(f"[config] effective={effective_config_path}")
    print("[setup] scale_factor=1.0 (depth already in meters; GS-LiDAR rescale N/A)")
    depth_statistic = str(cfg.get("depth_statistic", "median")).lower()
    depth_key = {"mean": "depth", "median": "depth_median"}.get(depth_statistic)
    if depth_key is None:
        raise ValueError(
            "depth_statistic must be one of {'mean', 'median'}, got "
            f"{depth_statistic!r}"
        )
    print(
        f"[setup] depth_statistic={depth_statistic} "
        f"metric_clamp=[1e-6,{EVAL_MAX_DEPTH_M:g}]m"
    )
    velocity_source = str(
        OmegaConf.select(cfg, "test.velocity_source", default="total")
    ).lower()
    if velocity_source not in {"total", "init", "match", "zero"}:
        raise ValueError(
            "test.velocity_source must be one of "
            "{'total', 'init', 'match', 'zero'}, got "
            f"{velocity_source!r}"
        )

    model = load_model(
        cfg,
        cfg.test.ckpt_path,
        device,
        allow_checkpoint_mismatch=bool(cfg.get("allow_checkpoint_mismatch", False)),
    )
    print(
        "[setup] gaussian_transport=velocity velocity_arrows=on "
        "(source + velocity * actual signed delta_t -> target)"
    )
    print(
        f"[setup] render_velocity={velocity_source}"
        + (
            " (inference-only velocity ablation; weights unchanged)"
            if velocity_source != "total" else ""
        )
    )
    # Utonia shuffles its serialization-order permutation with torch.randperm
    # in every forward (train and eval alike), so a fixed seed is still needed
    # for a repeatable eval. Pinning it off makes the encoder deterministic;
    # test.shuffle_orders=true restores the checkpoint's own setting.
    shuffle_orders = bool(OmegaConf.select(cfg, "test.shuffle_orders", default=False))
    pinned = [m for m in model.modules() if hasattr(m, "shuffle_orders")]
    for module in pinned:
        module.shuffle_orders = shuffle_orders
    print(f"[setup] utonia shuffle_orders={shuffle_orders} on {len(pinned)} modules")
    backends = MetricBackends(lpips_net="alex")

    input_offsets_sec = OmegaConf.select(cfg, "test.input_offsets_sec", default=None)
    input_offsets_sec = (
        (-0.5, 0.5) if input_offsets_sec is None
        else tuple(float(x) for x in input_offsets_sec)
    )
    print(f"[setup] input_offsets_sec={input_offsets_sec} "
          f"(input = T{input_offsets_sec[0]:+g}s / T{input_offsets_sec[1]:+g}s)")
    dataset = build_lidar4d_test_dataset(cfg.data, input_offsets_sec=input_offsets_sec)
    dataset_name = dataset.dataset_name
    print(f"[setup] dataset={dataset_name} windows={len(dataset)}")
    target_cam = dataset.target_cam_index
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0,
                        collate_fn=multiframe_collate_fn)

    vfov = tuple(cfg.data.vfov)
    window_metrics = []  # list of dicts (with seq_name / target_s)
    seq_gauss = defaultdict(list)   # seq_name -> center-point HTML frames
    seq_surfel = defaultdict(list)  # seq_name -> [{label, static, dynamic, boxes}] 1σ-surfel HTML
    gauss_dir = viz_dir / "gaussians"
    gauss_dir.mkdir(parents=True, exist_ok=True)
    size_records = []                     # per-window Gaussian-size stats (all)
    size_by_split = defaultdict(list)     # per nuScenes split

    limit = int(cfg.get("limit", 0)) or len(dataset)
    for i, batch in enumerate(loader):
        if i >= limit:
            break
        meta = dataset.index_meta[i]
        seq_name, target_s = meta["seq_name"], meta["target_s"]
        scene, dataset_split = meta["scene"], meta["dataset_split"]
        target_label = (f"f{meta['target_frame_id']:010d}" if "target_frame_id" in meta
                        else f"T{target_s}s")
        batch = to_device(batch, device)
        _input, gt = batch["input"], batch["gt"]

        out = model.p2g_model(_input)
        out = model.g2g_model(
            out,
            _input["timestamps_sec"],
            _input["window_duration_sec"],
        )
        out = select_render_velocity(out, velocity_source)
        renders = model.g2p_model(out, gt)

        b = 0
        # Keep the image metrics, point metrics, PNGs, and predicted LiDAR HTML
        # on the same renderer statistic. This evaluator defaults to median
        # depth; pass depth_statistic=mean for an explicit ablation.
        depth = renders[depth_key][b, target_cam]           # [1,H,W]
        intensity = renders["intensity_sh"][b, target_cam]
        raydrop = renders["raydrop"][b, target_cam]
        gt_depth = renders["gt_depth"][b, target_cam]
        gt_intensity = renders["gt_intensity_sh"][b, target_cam]
        gt_raydrop = renders["gt_raydrop"][b, target_cam]

        # GS-LiDAR: zero out predicted-drop pixels before depth/intensity/points.
        keep = 1.0 - _binary_raydrop(raydrop).to(depth.dtype)
        depth_keep = depth * keep
        intensity_keep = intensity * keep

        gt_cam = gt["cameras"][b][target_cam]
        row_to_theta = gt_cam.row_to_theta
        pred_pts = prediction_points_ref(depth, raydrop, gt_cam).detach().cpu().numpy()

        wm = {
            **meta,
            "dataset": dataset_name,
            # GS-LiDAR-style capped range-image metric: no-return stays zero
            # until depth_errors maps it to epsilon, while positive Pred/GT
            # returns above 80 m saturate at 80 m.
            "depth": depth_errors(
                depth_keep, gt_depth, backends,
                max_depth=EVAL_MAX_DEPTH_M,
            ),
            "intensity": intensity_errors(intensity_keep, gt_intensity, backends),
            "raydrop": raydrop_errors(
                raydrop, gt_raydrop, ratio=RAYDROP_THRESHOLD
            ),
            "points": point_metrics(
                depth_keep, gt_depth, row_to_theta, backends,
                vfov=vfov, far=EVAL_MAX_DEPTH_M,
                row_origin=getattr(gt_cam, "row_origin", None),
            ),
            "depth_diagnostics": depth_error_breakdown(
                depth_keep, gt_depth, max_depth=EVAL_MAX_DEPTH_M,
            ),
        }
        if getattr(gt_cam, "projection_stats", None):
            wm["gt_projection"] = gt_cam.projection_stats
        window_metrics.append(wm)
        save_viz(viz_dir / f"{seq_name}_{target_label}.png",
                 depth_keep, gt_depth, intensity_keep, gt_intensity,
                 raydrop, gt_raydrop)

        # Raw LiDAR in the same (ref) frame as the Gaussian means, so the HTML
        # can show whether a Gaussian actually sits on a measured surface: the
        # two input endpoints (what the model saw) and the GT sweep at the
        # target time (what it must reproduce).
        in_pts = _input["lidar_points"][:, :3].detach().cpu().numpy()
        gt_off = gt["offset"].tolist()
        gt_start = ([0] + gt_off[:-1])[target_cam]
        gt_pts = gt["lidar_points"][gt_start:gt_off[target_cam], :3].detach().cpu().numpy()

        # Gaussian centers + 1σ surfels at this window's target time (per-seq HTML).
        target_gaussians = out[b]
        render_t = float(model.g2p_model.render_timestamp(gt_cam))
        transport = velocity_transport_from_output(
            model.g2p_model, target_gaussians, render_t
        )
        source_times = transport["source_times_sec"]
        source_delta_t = render_t - source_times
        delta_label = ", ".join(
            f"{float(value):+.6f}" for value in source_delta_t
        )
        center_frame = {
            "label": (
                f"{target_label} | target={render_t:.6f}s | "
                f"source→target Δt=[{delta_label}]s"
            ),
            "source_frame_centers": transport[
                "target_centers_by_source_frame"
            ],
            "velocity_origins": transport["source_centers"],
            "velocity_mps": transport["velocity_mps"],
            "velocity_delta_t_sec": transport["delta_t_sec"],
            "velocity_source_frame_index": transport[
                "source_frame_index"
            ],
            "input_points": in_pts,
            "gt_points": gt_pts,
        }
        seq_gauss[seq_name].append(center_frame)
        # GT boxes at the target-second frame, interpolated to the sweep time
        # (not the stale parent-keyframe anns) and in the Gaussian ref frame, so
        # the surfel HTML can highlight in-bbox Gaussians + draw the boxes.
        gt_boxes_ref = dataset.target_boxes_ref(i)
        surf = surfels_from_output(
            model.g2p_model, target_gaussians, render_t,
            gt_boxes_ref=gt_boxes_ref,
        )
        seq_surfel[seq_name].append({
            "label": target_label, "static": surf["static"],
            "dynamic": surf["dynamic"], "dynamic_in_box": surf["dynamic_in_box"],
            "gt_boxes": surf["gt_boxes"],
            "input_points": in_pts, "gt_points": gt_pts,
            "pred_points": pred_pts})

        # per-window Gaussian-size statistics (effective radius vs geometry)
        st = collect_window_stats(
            model.g2p_model, target_gaussians, render_t
        )
        size_records.append(st)
        size_by_split[dataset_split].append(st)

        print(f"[{i+1:02d}/{len(dataset)}] {seq_name} {target_label} t={target_s:.6f}s | "
              f"depth RMSE={wm['depth']['rmse']:.3f} PSNR={wm['depth']['psnr']:.2f} | "
              f"int RMSE={wm['intensity']['rmse']:.3f} | "
              f"CD={wm['points']['cd']:.4f} F={wm['points']['fscore']:.3f} "
              f"P={wm['points']['precision']:.3f} R={wm['points']['recall']:.3f}")

    # ── per-sequence 3D Gaussian HTML (buttons 1..4 = target seconds) ───────
    for seq_name, frames in seq_gauss.items():
        sp = next((w["dataset_split"] for w in window_metrics
                   if w["seq_name"] == seq_name), "")
        save_sequence_html(gauss_dir / f"{seq_name}.html",
                           title=(
                               f"{seq_name}  ({sp})  — Gaussian centers "
                               f"[velocity={velocity_source}]"
                           ), frames=frames)
        save_sequence_surfel_html(gauss_dir / f"{seq_name}_surfel.html",
                                  title=f"{seq_name}  ({sp})", frames=seq_surfel[seq_name])
    print(f"[gaussians] wrote {len(seq_gauss)} center + {len(seq_surfel)} surfel HTML -> {gauss_dir}")

    # ── Gaussian-size statistical analysis (overall + per nuScenes split) ────
    print("\n" + "-" * 70)
    print(analyze_gaussian_sizes(size_records, out_dir, tag="all"))
    for sp in sorted(size_by_split):
        print(analyze_gaussian_sizes(size_by_split[sp], out_dir, tag=sp))

    # ── aggregate (overall + per nuScenes split + per sequence) ─────────────
    overall = summarize(window_metrics)
    per_seq = {}
    by_seq = defaultdict(list)
    for wm in window_metrics:
        by_seq[wm["seq_name"]].append(wm)
    for name, wms in by_seq.items():
        per_seq[name] = summarize(wms)

    # val = clean held-out; train = seen during training (leakage) for a
    # feed-forward model trained on the nuScenes train split.
    per_split = {}
    by_split = defaultdict(list)
    for wm in window_metrics:
        by_split[wm["dataset_split"]].append(wm)
    for sp, wms in by_split.items():
        per_split[sp] = summarize(wms)
        per_split[sp]["sequences"] = sorted({w["seq_name"] for w in wms})
        per_split[sp]["num_windows"] = len(wms)

    payload = {
        "config": {"ckpt": cfg.test.ckpt_path, "mode": cfg.data.mode,
                   "dataset": dataset_name,
                   "version": cfg.data.version, "scale_factor": 1.0,
                   "seed": seed,
                   "source": config_source,
                   "effective_config": str(effective_config_path),
                   "depth_statistic": depth_statistic,
                   "max_depth_m": EVAL_MAX_DEPTH_M,
                   "gt_range_policy": ("clamp_positive_returns" if dataset_name == "nuscenes"
                                       else "project_returns_below_max_depth"),
                   "velocity_source": velocity_source,
                   "input_offsets_sec": list(input_offsets_sec),
                   "shuffle_orders": shuffle_orders,
                   "vfov": list(vfov),
                   "image_height": int(cfg.data.image_height),
                   "image_width": int(cfg.data.image_width),
                   "protocol": getattr(dataset, "protocol", {"target_seconds": [1, 2, 3, 4]}),
                   "target_seconds": [1, 2, 3, 4] if dataset_name == "nuscenes" else None},
        "overall": overall,
        "depth_diagnostics": summarize_depth_breakdowns(
            [row["depth_diagnostics"] for row in window_metrics]
        ),
        "per_dataset_split": per_split,
        "per_sequence": per_seq,
        "windows": window_metrics,
    }
    if dataset_name == "nuscenes":
        payload["per_nuscenes_split"] = per_split  # retain existing result consumers
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(payload, f, indent=2)

    # ── print table ─────────────────────────────────────────────────────────
    def fmt(s):
        d, it, rd, pt = s["depth"], s["intensity"], s["raydrop"], s["points"]
        return (f"  depth : RMSE={d['rmse']:.3f} MedAE={d['medae']:.3f} "
                f"LPIPS={d['lpips']:.3f} SSIM={d['ssim']:.3f} PSNR={d['psnr']:.2f}\n"
                f"  inten : RMSE={it['rmse']:.3f} MedAE={it['medae']:.3f} "
                f"LPIPS={it['lpips']:.3f} SSIM={it['ssim']:.3f} PSNR={it['psnr']:.2f}\n"
                f"  raydp : RMSE={rd['rmse']:.3f} Acc={rd['acc']:.3f} F1={rd['f1']:.3f}\n"
                f"  point : CD={pt['cd']:.4f} F-score@0.05={pt['fscore']:.3f} "
                f"P={pt['precision']:.3f} R={pt['recall']:.3f}")

    seq_split = {wm["seq_name"]: wm["dataset_split"] for wm in window_metrics}
    print("\n" + "=" * 70)
    print("PER-SEQUENCE")
    for name in sorted(per_seq):
        print(f"[{name}  ({seq_split[name]})]\n{fmt(per_seq[name])}")
    print("-" * 70)
    print("PER nuScenes OFFICIAL SPLIT  (training exposure depends on scene exclusions)"
          if dataset_name == "nuscenes" else "PER KITTI-360 SCENE CATEGORY")
    for sp in sorted(per_split):
        print(f"<{sp}>  seqs={per_split[sp]['sequences']}  "
              f"n={per_split[sp]['num_windows']}\n{fmt(per_split[sp])}")
    print("-" * 70)
    print(f"OVERALL (mean over {len(window_metrics)} windows)")
    print(fmt(overall))
    print("=" * 70)
    print(f"\nsaved: {out_dir/'metrics.json'}  |  viz: {viz_dir}")


if __name__ == "__main__":
    cli = OmegaConf.from_cli()
    cfg, config_source = resolve_eval_config(cli)
    cfg.mode = "test"
    main(cfg, config_source=config_source)
