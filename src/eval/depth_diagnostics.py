"""Explain range-image RMSE without changing the benchmark metric."""
from __future__ import annotations

import numpy as np


def _array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def depth_error_breakdown(pred, gt, min_depth=1e-6, max_depth=80.0):
    """Split squared error by return presence BEFORE metric clamping.

    Pred must already be masked by predicted raydrop, exactly like depth_errors.
    Return-only scores are diagnostics; the standard all-pixel score is retained.
    """
    pred, gt = _array(pred), _array(gt)
    if pred.shape != gt.shape or not np.isfinite(pred).all() or not np.isfinite(gt).all():
        raise ValueError("Depth diagnostic inputs must have matching finite arrays")
    p, g = pred > 0, gt > 0
    error = np.square(np.clip(pred, min_depth, max_depth) - np.clip(gt, min_depth, max_depth))
    regions = {
        "both_return": p & g,
        "missed_gt_return": ~p & g,
        "extra_pred_return": p & ~g,
        "both_no_return": ~p & ~g,
    }
    total_sse = float(error.sum())
    result = {"pixels": int(error.size), "squared_error_sum": total_sse,
              "all_pixel_rmse": float(np.sqrt(error.mean())), "regions": {}}
    for name, mask in regions.items():
        count, sse = int(mask.sum()), float(error[mask].sum())
        result["regions"][name] = {
            "pixels": count, "pixel_fraction": count / error.size,
            "squared_error_sum": sse,
            "rmse": float(np.sqrt(sse / count)) if count else None,
            "squared_error_fraction": sse / total_sse if total_sse else 0.0,
        }
    return result


def summarize_depth_breakdowns(items):
    if not items:
        return {}
    pixels = sum(item["pixels"] for item in items)
    sse = sum(item["squared_error_sum"] for item in items)
    result = {"windows": len(items), "pixels": pixels, "squared_error_sum": sse,
              "mean_window_all_pixel_rmse": float(np.mean([item["all_pixel_rmse"] for item in items])),
              "pooled_all_pixel_rmse": float(np.sqrt(sse / pixels)), "regions": {}}
    for name in items[0]["regions"]:
        count = sum(item["regions"][name]["pixels"] for item in items)
        error = sum(item["regions"][name]["squared_error_sum"] for item in items)
        result["regions"][name] = {
            "pixels": count, "pixel_fraction": count / pixels,
            "squared_error_sum": error,
            "pooled_rmse": float(np.sqrt(error / count)) if count else None,
            "squared_error_fraction": error / sse if sse else 0.0,
        }
    return result
