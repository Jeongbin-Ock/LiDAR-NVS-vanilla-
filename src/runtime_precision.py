"""Process-wide arithmetic switches that a run's config selects."""
from __future__ import annotations

from omegaconf import OmegaConf


def apply_spconv_tf32(cfg) -> bool:
    """Set spconv's TF32 switch from ``train.spconv_allow_tf32``; return it.

    Utonia's sparse convolutions (every block's CPE ``SubMConv3d`` and the LoRA
    on it) run through spconv, which ignores PyTorch's
    ``set_float32_matmul_precision`` and multiplies in full FP32 unless
    ``spconv.constants.SPCONV_ALLOW_TF32`` is set. With it, spconv may pick
    TF32 Tensor Core kernels: the multiply inputs are rounded to TF32 (FP32's
    exponent, a 10-bit mantissa) and the products accumulate in FP32 -- the
    trade PyTorch's ``"high"`` matmul precision already makes for every Linear
    during training. Tensors, parameters and checkpoints stay FP32.

    An absent key is ``false``: checkpoints saved before the key existed
    trained with FP32 sparse convolutions, and their embedded configs resume
    with them. spconv caches the kernel it picks for each problem shape at the
    first call, so this must run before the process's first sparse
    convolution; setting it later silently keeps the cached choice.
    """
    import spconv.constants

    enabled = bool(OmegaConf.select(cfg, "train.spconv_allow_tf32", default=False))
    spconv.constants.SPCONV_ALLOW_TF32 = enabled
    return enabled


__all__ = ["apply_spconv_tf32"]
