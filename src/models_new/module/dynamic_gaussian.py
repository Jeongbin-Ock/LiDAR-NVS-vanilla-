"""Stable import path for the Dynamic 2DGS implementation.

The implementation lives in :mod:`.dynamic`; this module re-exports its public
symbols so ``Point2Gaus``, the tests and the analysis tools import them from one
place.
"""
from __future__ import annotations

from .dynamic.common import SinusoidalScalarEncoder, _cfg_get
from .dynamic.temporal import (
    BidirectionalCrossAttentionBase,
    CosineRoPELayerWeightedCrossAttention,
    FinalFeatureBarrierCrossAttention,
    FixedMotionGainLearnableMaxSpeedBarrierCrossAttention,
    FixedMotionHeadRMSNorm,
    GroupedGainBarrierCrossAttention,
    GroupedHeadRMSNorm,
    LayerwiseSoftplusMaxSpeedBarrierCrossAttention,
    LearnableMaxSpeedBarrierLayerWeightedCrossAttention,
    MaxSpeedBarrierLayerWeightedCrossAttention,
    RawQKRoPELayerWeightedCrossAttention,
    RMSNormFeatureRoPELayerWeightedCrossAttention,
    RMSNormRoPELayerWeightedCrossAttention,
)
from .dynamic.heads import EmbeddedInitVelocityHead, GaussianAttributeHead
from .dynamic.backends import (
    CosineRoPEVelocityGaussianBackend,
    DynamicGaussianBackend,
    FinalFeatureBarrierVelocityGaussianBackend,
    FixedMotionGainMaxSpeedBarrierVelocityGaussianBackend,
    LayerwiseSoftplusMaxSpeedBarrierVelocityGaussianBackend,
    LearnableMaxSpeedBarrierVelocityGaussianBackend,
    MaxSpeedBarrierVelocityGaussianBackend,
    RawQKRoPEVelocityGaussianBackend,
    RMSNormFeatureRoPEVelocityGaussianBackend,
    RMSNormRoPEVelocityGaussianBackend,
    RouterCapableGaussianBackend,
    SeedConditionedBarrierVelocityGaussianBackend,
    SeedConditionedCosineRoPEVelocityGaussianBackend,
    SingleGaussianBarrierVelocityGaussianBackend,
)
from .dynamic.trajectory import DynamicGausTemp

__all__ = [
    "BidirectionalCrossAttentionBase",
    "CosineRoPELayerWeightedCrossAttention",
    "CosineRoPEVelocityGaussianBackend",
    "DynamicGaussianBackend",
    "DynamicGausTemp",
    "EmbeddedInitVelocityHead",
    "FinalFeatureBarrierCrossAttention",
    "FinalFeatureBarrierVelocityGaussianBackend",
    "FixedMotionGainLearnableMaxSpeedBarrierCrossAttention",
    "FixedMotionGainMaxSpeedBarrierVelocityGaussianBackend",
    "FixedMotionHeadRMSNorm",
    "GaussianAttributeHead",
    "GroupedGainBarrierCrossAttention",
    "GroupedHeadRMSNorm",
    "LayerwiseSoftplusMaxSpeedBarrierCrossAttention",
    "LayerwiseSoftplusMaxSpeedBarrierVelocityGaussianBackend",
    "LearnableMaxSpeedBarrierLayerWeightedCrossAttention",
    "LearnableMaxSpeedBarrierVelocityGaussianBackend",
    "MaxSpeedBarrierLayerWeightedCrossAttention",
    "MaxSpeedBarrierVelocityGaussianBackend",
    "RawQKRoPELayerWeightedCrossAttention",
    "RawQKRoPEVelocityGaussianBackend",
    "RMSNormFeatureRoPELayerWeightedCrossAttention",
    "RMSNormFeatureRoPEVelocityGaussianBackend",
    "RMSNormRoPELayerWeightedCrossAttention",
    "RMSNormRoPEVelocityGaussianBackend",
    "RouterCapableGaussianBackend",
    "SeedConditionedBarrierVelocityGaussianBackend",
    "SeedConditionedCosineRoPEVelocityGaussianBackend",
    "SingleGaussianBarrierVelocityGaussianBackend",
    "SinusoidalScalarEncoder",
]
