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
    GlobalSoftplusMaxSpeedBarrierCrossAttention,
    GroupedGainBarrierCrossAttention,
    GroupedHeadRMSNorm,
    LayerwiseSoftplusMaxSpeedBarrierCrossAttention,
    LearnableMaxSpeedBarrierLayerWeightedCrossAttention,
    MaxSpeedBarrierLayerWeightedCrossAttention,
    RawQKRoPELayerWeightedCrossAttention,
    RMSNormFeatureRoPELayerWeightedCrossAttention,
    RMSNormRoPELayerWeightedCrossAttention,
    UniformLayerMeanSoftplusBarrierCrossAttention,
)
from .dynamic.gaussian_slots import DenseCountGaussianHead
from .dynamic.heads import EmbeddedInitVelocityHead, GaussianAttributeHead
from .dynamic.local_attention import RadiusLocalAttentionBlock
from .dynamic.nested_count import NestedCountGaussianHead, RoutedCountGaussianHead
from .dynamic.option_experts import OptionExpertGaussianHead
from .dynamic.ordinal_count import CosineGateLayer, OrdinalCountGaussianHead
from .dynamic.backends import (
    CosineRoPEVelocityGaussianBackend,
    DenseCountBarrierVelocityGaussianBackend,
    DynamicGaussianBackend,
    FinalFeatureBarrierVelocityGaussianBackend,
    FixedMotionGainMaxSpeedBarrierVelocityGaussianBackend,
    GlobalSoftplusMaxSpeedBarrierVelocityGaussianBackend,
    LayerwiseSoftplusMaxSpeedBarrierVelocityGaussianBackend,
    LearnableMaxSpeedBarrierVelocityGaussianBackend,
    LocalAttentionBarrierVelocityGaussianBackend,
    MaxSpeedBarrierVelocityGaussianBackend,
    NestedCountBarrierVelocityGaussianBackend,
    OptionExpertBarrierVelocityGaussianBackend,
    OrdinalCountBarrierVelocityGaussianBackend,
    RawQKRoPEVelocityGaussianBackend,
    RMSNormFeatureRoPEVelocityGaussianBackend,
    RMSNormRoPEVelocityGaussianBackend,
    RouterCapableGaussianBackend,
    SeedConditionedBarrierVelocityGaussianBackend,
    SeedConditionedCosineRoPEVelocityGaussianBackend,
    SingleGaussianBarrierVelocityGaussianBackend,
    UniformLayerDenseCountBarrierVelocityGaussianBackend,
)
from .dynamic.trajectory import DynamicGausTemp

__all__ = [
    "BidirectionalCrossAttentionBase",
    "CosineGateLayer",
    "CosineRoPELayerWeightedCrossAttention",
    "CosineRoPEVelocityGaussianBackend",
    "DenseCountBarrierVelocityGaussianBackend",
    "DenseCountGaussianHead",
    "DynamicGaussianBackend",
    "DynamicGausTemp",
    "EmbeddedInitVelocityHead",
    "FinalFeatureBarrierCrossAttention",
    "FinalFeatureBarrierVelocityGaussianBackend",
    "FixedMotionGainLearnableMaxSpeedBarrierCrossAttention",
    "FixedMotionGainMaxSpeedBarrierVelocityGaussianBackend",
    "FixedMotionHeadRMSNorm",
    "GaussianAttributeHead",
    "GlobalSoftplusMaxSpeedBarrierCrossAttention",
    "GlobalSoftplusMaxSpeedBarrierVelocityGaussianBackend",
    "GroupedGainBarrierCrossAttention",
    "GroupedHeadRMSNorm",
    "LayerwiseSoftplusMaxSpeedBarrierCrossAttention",
    "LayerwiseSoftplusMaxSpeedBarrierVelocityGaussianBackend",
    "LearnableMaxSpeedBarrierLayerWeightedCrossAttention",
    "LearnableMaxSpeedBarrierVelocityGaussianBackend",
    "LocalAttentionBarrierVelocityGaussianBackend",
    "MaxSpeedBarrierLayerWeightedCrossAttention",
    "MaxSpeedBarrierVelocityGaussianBackend",
    "NestedCountBarrierVelocityGaussianBackend",
    "NestedCountGaussianHead",
    "OptionExpertBarrierVelocityGaussianBackend",
    "OptionExpertGaussianHead",
    "OrdinalCountBarrierVelocityGaussianBackend",
    "OrdinalCountGaussianHead",
    "RadiusLocalAttentionBlock",
    "RawQKRoPELayerWeightedCrossAttention",
    "RawQKRoPEVelocityGaussianBackend",
    "RMSNormFeatureRoPELayerWeightedCrossAttention",
    "RMSNormFeatureRoPEVelocityGaussianBackend",
    "RMSNormRoPELayerWeightedCrossAttention",
    "RMSNormRoPEVelocityGaussianBackend",
    "RoutedCountGaussianHead",
    "RouterCapableGaussianBackend",
    "SeedConditionedBarrierVelocityGaussianBackend",
    "SeedConditionedCosineRoPEVelocityGaussianBackend",
    "SingleGaussianBarrierVelocityGaussianBackend",
    "SinusoidalScalarEncoder",
    "UniformLayerDenseCountBarrierVelocityGaussianBackend",
    "UniformLayerMeanSoftplusBarrierCrossAttention",
]
