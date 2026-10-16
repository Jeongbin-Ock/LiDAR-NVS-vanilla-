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
  `RouterCapableGaussianBackend` adds the learned K={1,2,3} router, and
  `LocalAttentionBarrierVelocityGaussianBackend` is the V15 family's backend
  (V14.1's temporal stack, local attention, a token-anchored head), with one
  subclass per head: `NestedCountBarrierVelocityGaussianBackend` (V15/V15.1),
  `OptionExpertBarrierVelocityGaussianBackend` (V15.2),
  `DenseCountBarrierVelocityGaussianBackend` (V15.3) and
  `OrdinalCountBarrierVelocityGaussianBackend` (V16/V16.1)
- `local_attention.py`: V15's pre-LN radius local-attention block over both
  sweeps' tokens in the window reference frame
- `gaussian_slots.py`: the token-anchored Gaussian slot layout, initialization
  and packing every V15-family head shares, and V15.3's router-free
  `DenseCountGaussianHead`
- `nested_count.py`: `RoutedCountGaussianHead` (the reduce/keep/add router and
  count-block contract V15 and V15.2 share) and V15's G1..G5 decoder
  (`NestedCountGaussianHead`)
- `option_experts.py`: V15.2's `OptionExpertGaussianHead`, E1/E3/E5 linear
  experts on one shared trunk
- `ordinal_count.py`: V16's and V16.1's `OrdinalCountGaussianHead`, V15's
  G1..G5 decoder under two bounded ordinal gates (`CosineGateLayer`), whose
  training routed set carries per-Gaussian `blend_mask` values for the
  rasterizer's masked blending; its auxiliary set is every G1..G5 (V16) or a
  uniformly random prefix per token (V16.1)
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
| `attention_velocity_v14` | `MaxSpeedBarrierVelocityGaussianBackend` | V11.4's linear-hinge `MaxSpeedBarrierLayerWeightedCrossAttention`; hybrid Utonia adaptation (`src/models/utonia/adapters.py`) | learned K=1..3, token-position anchors, feature-only trunk |
| `attention_velocity_v14_1` | `GlobalSoftplusMaxSpeedBarrierVelocityGaussianBackend` | `GlobalSoftplusMaxSpeedBarrierCrossAttention`, one global `v_max=softplus(v_raw)`; V14's Utonia | learned K=1..3, token-position anchors, feature-only trunk |
| `attention_velocity_v15` | `NestedCountBarrierVelocityGaussianBackend` | V14.1's cross-attention without LayerScale, then one 0.8 m radius local-attention block (`RadiusLocalAttentionBlock`); post-activation Utonia stem adapter | nested: G1..G5 per token, router picks {G1}/{G1..G3}/{G1..G5}; training also renders every token's G1..G3 and averages the two render losses |
| `attention_velocity_v15_1` | `NestedCountBarrierVelocityGaussianBackend` | V15's, unchanged | V15's nested G1..G5 and router, trained on the routed render alone (no G1..G3 render; `auxiliary_render: none`) |
| `attention_velocity_v15_2` | `OptionExpertBarrierVelocityGaussianBackend` | V15's, unchanged | V15's router over per-option experts E1/E3/E5 (linear readouts of one shared 576-wide trunk), every Gaussian of the chosen expert gated; training also renders every token's E5 and averages the two render losses |
| `attention_velocity_v15_3` | `DenseCountBarrierVelocityGaussianBackend` | V15's, unchanged | no router: V15's decoder renders all five Gaussians of every token (fixed `K_max: 5`, token anchor) |
| `attention_velocity_v15_4` | `DenseCountBarrierVelocityGaussianBackend` | V15's with eight temporal heads of 54 channels instead of twelve of 36 (RoPE base 10^1.875 keeps the 5-232 m band span); local attention unchanged | V15.3's |
| `attention_velocity_v15_5` | `UniformLayerDenseCountBarrierVelocityGaussianBackend` | V15.4's, but `v_init` is the plain 1/L mean of the layers' head-0 readouts (`UniformLayerMeanSoftplusBarrierCrossAttention`, no `layer_logits`) | V15.3's |
| `attention_velocity_v16` | `OrdinalCountBarrierVelocityGaussianBackend` | V15's, unchanged | V15's nested G1..G5, chosen by a keep gate (G2/G3) and an add gate (G4/G5 given G2/G3), cosine logits bounded by 3, hard Bernoulli samples with a straight-through sigmoid; training renders every G1..G5 under blend masks (masked rasterization) plus every token's G1..G5 unmasked at 2 views and averages the two render losses |
| `attention_velocity_v16_1` | `OrdinalCountBarrierVelocityGaussianBackend` | V15's, unchanged | V16, except that the unmasked auxiliary render is each token's G1..Gk with k drawn uniformly from {1, 3, 5} per step, independently of the router |

Checkpoint compatibility depends on the registered module tree, parameter
assignment order, and tensor shapes, not on class names: Lightning checkpoints
store tensors by attribute path. `tests/test_checkpoint_compatibility.py` pins
the ordered state and parameter schema of every variant.
