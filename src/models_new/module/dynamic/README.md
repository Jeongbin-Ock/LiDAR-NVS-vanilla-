# Dynamic 2DGS module map

`dynamic_gaussian.py` re-exports the public symbols of this package as one
stable import path. New code may import either from there or from the semantic
module that owns a symbol:

- `common.py`: config lookup and scalar encoding
- `temporal.py`: the shared bidirectional attention base and the temporal
  families built on it
- `heads.py`: the Gaussian attribute decoder and the embedded-`v_init` velocity
  offset head
- `backends.py`: backend composition; `DynamicGaussianBackend` owns the
  fixed-count path and the `v = v_init + v_offset` contract,
  `RouterCapableGaussianBackend` adds the learned K={1,2,3} router
- `trajectory.py`: output trajectory contract validation

Config variant names in the table omit their shared `dynamic_2dgs_` prefix.
Every variant's velocity head is `EmbeddedInitVelocityHead`.

| Config variant suffix | Backend | Temporal / matching | Gaussian count |
| --- | --- | --- | --- |
| `attention_velocity_v11` | `MaxSpeedBarrierVelocityGaussianBackend` | `MaxSpeedBarrierLayerWeightedCrossAttention` | learned K=1..3 by default; fixed available by config |
| `attention_velocity_v11_1` | `SingleGaussianBarrierVelocityGaussianBackend` | `GroupedGainBarrierCrossAttention` | fixed by config |
| `attention_velocity_v11_2` | `FinalFeatureBarrierVelocityGaussianBackend` | `FinalFeatureBarrierCrossAttention` | learned K=1..3 by default; fixed available by config |
| `attention_velocity_v11_3` | `SeedConditionedBarrierVelocityGaussianBackend` | `MaxSpeedBarrierLayerWeightedCrossAttention` | fixed at K=2 by default |
| `attention_velocity_v11_4` | `MaxSpeedBarrierVelocityGaussianBackend` | linear-hinge `MaxSpeedBarrierLayerWeightedCrossAttention` | learned K=1..3 by default; fixed available by config |
| `attention_velocity_v11_5` | `SeedConditionedBarrierVelocityGaussianBackend` | linear-hinge `MaxSpeedBarrierLayerWeightedCrossAttention` | fixed K=2, range-quantile seeds |
| `attention_velocity_v11_6` | `SeedConditionedBarrierVelocityGaussianBackend` | linear-hinge `MaxSpeedBarrierLayerWeightedCrossAttention` | fixed K=3, range-quantile seeds |
| `attention_velocity_v12` | `LearnableMaxSpeedBarrierVelocityGaussianBackend` | `LearnableMaxSpeedBarrierLayerWeightedCrossAttention`, one global `v_max=exp(v_0)` | learned K=1..3 by default; fixed available by config |
| `attention_velocity_v12_1` | `FixedMotionGainMaxSpeedBarrierVelocityGaussianBackend` | learned `v_max`/lambda barrier, fixed head-0 gain, learned heads-1..11 gain | learned K=1..3 by default; fixed available by config |
| `attention_velocity_v12_2` | `LayerwiseSoftplusMaxSpeedBarrierVelocityGaussianBackend` | `LayerwiseSoftplusMaxSpeedBarrierCrossAttention`, one `v_max_l=softplus(v_raw_l)` per layer | learned K=1..3 by default; fixed available by config |
| `attention_velocity_v13` | `CosineRoPEVelocityGaussianBackend` | `CosineRoPELayerWeightedCrossAttention`, all-head RoPE/Flash, L2 head 0 at temperature 0.03 | learned K=1..3, range-quantile seeds |
| `attention_velocity_v13_1` | `SeedConditionedCosineRoPEVelocityGaussianBackend` | V13 cosine/RoPE matcher | fixed K=2, range-quantile seeds |
| `attention_velocity_v13_2` | `RawQKRoPEVelocityGaussianBackend` | all-head RoPE/Flash with raw QK / sqrt(d) on every head | learned K=1..3, range-quantile seeds |
| `attention_velocity_v13_3` | `RMSNormRoPEVelocityGaussianBackend` | all-head RMSNorm + 3D RoPE in one Flash call; no distance bias | learned K=1..3, range-quantile seeds |
| `attention_velocity_v13_4` | `RMSNormFeatureRoPEVelocityGaussianBackend` | RMSNorm on all heads, 3D RoPE only on heads 1-11, one Flash call | learned K=1..3, range-quantile seeds |

Checkpoint compatibility depends on the registered module tree, parameter
assignment order, and tensor shapes, not on class names: Lightning checkpoints
store tensors by attribute path. `tests/test_checkpoint_compatibility.py` pins
the ordered state and parameter schema of every variant.
