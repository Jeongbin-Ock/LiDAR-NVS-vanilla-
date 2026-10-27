# Experiment configuration

The Python implementation is the `src/experiment_config` package, split by
responsibility. `loader.py` keeps composition and checkpoint restore, and the
package re-exports its public API (`from src.experiment_config import
compose_fresh_config`). `definitions.py` owns variant constants and schema key
sets, `helpers.py` owns shared OmegaConf operations, `validation.py` owns static
config validation, and `migration.py` upgrades older checkpoint configs.
Dependencies flow from definitions through helpers to validation; none of those
modules imports `loader`.

Fresh runs are composed in this order:

1. `default.yaml`: dataset-agnostic model, loss, optimizer and logging defaults
2. one allow-listed model overlay under `variants/` (model-only; no `data:`)
3. one data profile under `data/`, which replaces the whole `data` block
4. OmegaConf CLI overrides

Each overlay names its default profile with `data_config:` (V11-V13:
`nuscenes`, the legacy fixed-window loader; V11.6, V12.2, V14 and later:
`nuscenes_dynamic`, the random-duration loader). A CLI `data_config=<name>` or
`data_config=<path>` picks another profile, e.g. `data_config=kitti360`.
`default.yaml` and the overlays are rejected if they carry a `data` block.
Checkpoint resumes ignore all of these files and restore the embedded config.

| Profile | Dataset / loader | Used by default for |
| --- | --- | --- |
| `data/nuscenes.yaml` | nuScenes, `nuscenes` (legacy keyframe windows, bbox mode) | V11-V13 |
| `data/nuscenes_dynamic.yaml` | nuScenes, `nuscenes_dynamic_lidar4d` (random-duration) | V11.6, V12.2, V14-V16.1 |
| `data/kitti360.yaml` | KITTI-360, `kitti360_dynamic`, calibrated 64×2178 GT | opt-in (`data_config=kitti360`) |
| `data/kitti360_lidar4d.yaml` | KITTI-360, `kitti360_dynamic`, LiDAR4D uniform 66×1030 GT | opt-in (`data_config=kitti360_lidar4d`) |

Inference overlays for `test_lidar4d_nvs.py eval_config=...` live in `eval/`.
Stage overlays applied on top of a finished checkpoint live in `refine/` and
`adapt/` (see "Raydrop refinement" and "Sensor-head adaptation" at the end).
The JSON manifests (`compact_val_windows.json`, `kitti360_compact_val_windows.json`,
`kitti360_train_split.json`) stay at this level because checkpoints record
their paths.

The selector uses the existing dot-list syntax:

```bash
python main.py model.variant=dynamic_2dgs_attention_velocity_v11
python main.py model.variant=dynamic_2dgs_attention_velocity_v11_1
python main.py model.variant=dynamic_2dgs_attention_velocity_v11_2
python main.py model.variant=dynamic_2dgs_attention_velocity_v11_3
python main.py model.variant=dynamic_2dgs_attention_velocity_v11_4
python main.py model.variant=dynamic_2dgs_attention_velocity_v11_5
python main.py model.variant=dynamic_2dgs_attention_velocity_v11_6
python main.py model.variant=dynamic_2dgs_attention_velocity_v12
python main.py model.variant=dynamic_2dgs_attention_velocity_v12_1
python main.py model.variant=dynamic_2dgs_attention_velocity_v12_2
python main.py model.variant=dynamic_2dgs_attention_velocity_v13
python main.py model.variant=dynamic_2dgs_attention_velocity_v13_1
python main.py model.variant=dynamic_2dgs_attention_velocity_v13_2
python main.py model.variant=dynamic_2dgs_attention_velocity_v13_3
python main.py model.variant=dynamic_2dgs_attention_velocity_v13_4
python main.py model.variant=dynamic_2dgs_attention_velocity_v14
python main.py model.variant=dynamic_2dgs_attention_velocity_v14_1
python main.py model.variant=dynamic_2dgs_attention_velocity_v15
python main.py model.variant=dynamic_2dgs_attention_velocity_v15_1
python main.py model.variant=dynamic_2dgs_attention_velocity_v15_2
python main.py model.variant=dynamic_2dgs_attention_velocity_v15_3
python main.py model.variant=dynamic_2dgs_attention_velocity_v15_4
python main.py model.variant=dynamic_2dgs_attention_velocity_v15_5
python main.py model.variant=dynamic_2dgs_attention_velocity_v16
python main.py model.variant=dynamic_2dgs_attention_velocity_v16_1
```

A fresh run defaults to `dynamic_2dgs_attention_velocity_v11_4`. Every listed
variant remains available for exact checkpoint reconstruction and controlled
A/B runs. The base config's `device: [2, 3]` is the default GPU pair; every
overlay inherits it and it stays an ordinary CLI-overridable runtime setting.

Every variant feeds the Utonia stage-3 XYZI token (432D, 0.4 m cells) with
adapters on the frozen base (`p2g.utonia_lora.enable=true` is required) straight
into temporal attention; there is no separate intensity stream, fusion MLP or
token refiner. V11-V13 put LoRA on every Utonia Linear and SubMConv3d; V14,
V14.1 and the V15 family use the hybrid adaptation described under **V14**
below. V11.1,
V11.3, V11.5, V11.6 and V13.1 emit a fixed number of Gaussians per occupied
endpoint token -- one observed-medoid seed by default, two range quantiles in
V11.3/V11.5/V13.1 and three in V11.6; V11, V11.2, V11.4, V12, V12.1, V12.2,
V13, V13.2, V13.3, V13.4, V14 and V14.1 let a router select one to three
Gaussians instead, range-quantile-seeded except in V14/V14.1, which anchor
every slot at the token position. V15 decodes five Gaussians per token around
that anchor and lets a router pick one, three or five (`nested_gumbel`); V15.2
gives each of those three options its own expert (`expert_gumbel`), V15.3
has no router and renders five Gaussians per token (`legacy`, `K_max: 5`),
V15.4 is V15.3 with eight temporal heads instead of twelve, V15.5 is V15.4
with `v_init` read from the uniform mean over layers instead of a learned layer
mixture, V18 is V15.4 whose Gaussians carry a rendered feature that G2P
decodes into intensity and raydrop instead of SH coefficients, and
V16 picks V15's prefixes with two bounded ordinal gates trained through masked
rasterization (`nested_ordinal`); V16.1 is V16 whose second training render is
a uniformly random prefix per token.
`p2g.grid_query` chooses between these, sets the fixed count and selects where
seeds start (`seed_mode`) -- see "Gaussians per token" below.

Every variant embeds `time_coordinate = relative_seconds / time_reference_sec`,
where `time_reference_sec=1.0` is fixed across samples. An irregular 0.8-second
pair therefore supplies endpoint coordinates `[0, 0.8]`, rather than `[0, 1]`.
Its scalar time becomes 17 raw Fourier values, passes through a
`17 -> 64 -> 64` MLP and one bias-free `Linear(64, 432)`, and is added once to
the token before the first layer. Motion is `v_init + v_offset`: `v_init` is the
temporal module's coordinate-readout displacement divided by the signed endpoint
interval, and a zero-initialized offset head that embeds a detached `v_init`
adds `v_offset` in m/s. The renderer applies physical seconds:

`position(t) = position_source + velocity_mps * delta_time_sec`.

V11 routes `LN(f')` through the `learned_gumbel` K={1,2,3} router and
index-expands one token velocity to every child Gaussian; its identity is how
correspondence is produced inside temporal attention.

Its position encoding is split across heads. Head 0 is the sole correspondence
head and is the only head whose probabilities touch coordinates; it carries no
RoPE. Its logit instead takes one additive max-speed soft barrier,

`bias_ij = -a * relu(||x_i - x_j|| / (v_max * |dt|) - 1)^2`, with
`v_max = 30 m/s` and `a = 4`.

The bias is exactly zero inside the radius `R = v_max * dt` a top-speed object
could cover over the physical endpoint interval, and grows quadratically outside
it: 0 at `R`, 1 logit at `1.5R`, 4 at `2R`, and 16 at `3R`. Because `v_init` is
the readout displacement divided by that same interval, the barrier is a soft
cap on `v_init` at `v_max`. QK-Norm holds the content logit spread near +-1 at
initialization, so `a=4` is decisive at `2R` -- the same penalty V10's `-(d/R)^2`
charges there -- while leaving everything inside `R` free, which V10's prior does
not. Heads 1-11 keep ordinary metric 3D RoPE at `base=100` and
`1.2566 rad/m` (`2*pi/scale = 5 m` for the highest-frequency band).

The hinge is not a bilinear form, so unlike V10's quadratic it cannot be encoded
in extra Q/K channels. Head 0 therefore runs an explicit softmax, chunked over
`match_chunk_size` query rows and recomputed in backward; heads 1-11 stay on
varlen FlashAttention. Head 0's V carries the
ordinary feature channels plus opposite-frame xyz, so one pass returns both its
feature update and `W_l @ xyz`. `match_chunk_size` trades attention recompute
against peak `N_q x N_k` score memory and never changes the result.

`v_init` mixes the twelve head-0 coordinate expectations under `softmax(a_l)`
over twelve plain learned scalars. Unlike V10 there is no query-only global token
and no per-sample scorer MLP: the mixture is shared by every token and every
sample, so it can only learn which *depth* resolves correspondence, starting from
an exact uniform 1/12.

V11's offset head is V8's rather than V10's feature-only one. Head 0's attended
*feature* channels never carry the metric coordinate expectation itself, so the
correction cannot see what it is correcting unless the initializer is embedded:
`e_v = Linear(3,32) -> SiLU -> Linear(32,32)` on a detached `v_init`, then
`Linear(96,3)(SiLU(Linear(432+32,96)([LN(f'), e_v])))`. The output Linear is still
zero-initialized, so a fresh run starts at `v_total = v_init`, and detaching keeps
`v_init + v_offset` the only route from the offset loss back into correspondence.

V11 keeps Chamfer at `0.02` and regularizes with `velocity_l2.mode=split_group_l2` at `init_weight = offset_weight = 0.01`.

**V11.2** takes V11 and moves correspondence out of the layer stack. Every
configured value is identical to V11 except `model.variant` and the `exp_name`
derived from it. What changes is which parameters exist and where geometry
enters a probability.

V11 splits its heads because head 0 doubles as the correspondence head: it
drops 3D RoPE, carries the barrier instead, and reads opposite-frame
coordinates at every layer. The barrier is a hinge on Euclidean distance rather
than a bilinear form, so it cannot be folded into Q/K channels, and head 0 has
to run an explicit chunked softmax while heads 1-11 stay on varlen
FlashAttention.

V11.2 builds `v_init` after the stack, so nothing inside it reads coordinates.
Head 0 carries ordinary 3D RoPE like every other head, the barrier leaves the
layer loop, and all twelve heads go through one FlashAttention call per layer.
Measured on one A100 at 7.5k tokens per frame, batch 2, forward and backward of
the temporal module alone, this cuts 1,034 ms to 645 ms and 13.67 GiB to 12.84
GiB.

Correspondence is read once, from the complete final feature `f'` including
layer 11's FFN, by one dedicated head: `LayerNorm(432)` shared by q and k, then
`Linear(432, 36)` for each, then `RMSNorm(36)` for each. That is the same chain
and the same width as a single V11 readout head, and PyTorch initializes
`Linear` from `fan_in`, which is 432 for both, so it starts from V11's own
distribution. For every query it scores every opposite-frame key as
`q @ k * 36^-0.5 - 4 * relu(distance / (30 * duration_sec) - 1)^2`, keeping
V11's query chunking and backward recomputation. Dense softmax probability
multiplies the opposite-frame key coordinates to obtain `matched_position`;
only then is the query coordinate subtracted to obtain displacement, and the
downstream signed endpoint duration converts that displacement to `v_init`.
This readout is the only place in the variant where geometry enters a
probability.

The head owns its 32,112 parameters, 0.12% of the temporal stack, rather than
borrowing the head-0 rows of `q_proj[-1]`/`k_proj[-1]`. Borrowing costs nothing
and keeps the parameter count at V11's, but it asks one projection to serve
both layer 11's attention over its *pre*-FFN input and this readout over the
*post*-FFN output, two input distributions no V11 readout ever spans. V11's
`layer_logits` is dropped in the other direction: there are no per-layer
readouts left for it to weight, so it would sit in the checkpoint collecting no
gradient. V11's remaining layer parameters keep V11's names, shapes and order.

The readout's gain is logged as `qk_match_gamma_q`/`qk_match_gamma_k` and
`qk_match_logit_bound`, separate from V11's comparable per-layer keys, so its
race against the fixed barrier stays visible. `qk_motion_head_count` reads zero
here because no layer head touches coordinates. V11.2 publishes no
layer-mixture diagnostics.

Against V11 this moves two things on purpose: where correspondence is read, and
whether the layer stack still needs a geometric prior of its own once it no
longer produces correspondence. Adaptive K, token-velocity sharing, the
detached embedded-init offset, split regularizer and Chamfer contracts are
inherited unchanged from V11.

`velocity_l2` is the one motion prior, and `mode: split_group_l2` its only
mode: `init_weight * mean(||v_init||_2^2) + offset_weight *
mean(||v_offset||_2^2)`, charged independently. Only the optimized total sees
those weights: `loss_velocity_l2` stays the raw unweighted magnitude
(`mean(||v_init||^2) + mean(||v_offset||^2)`), and `loss_velocity_l2_init` /
`loss_velocity_l2_offset` log the two means beside it. The weighted
contribution (`wc_velocity_l2`) is computed for the total but is not sent to
wandb. The single-term modes (`final_l2`, `final_group_l2`, `final_group_l1`,
`offset_l2`, `offset_group_l2`), the `weight` key and the `small_motion`
displacement prior were removed together with the pre-V11 variants that used
them; checkpoint configs that still carry `small_motion` switched off load
unchanged, and one that switches it on is rejected.

The prior and the logged motion diagnostics (`velocity_speed_mean`,
`velocity_init_norm_mean`, ...) average over tokens rather than Gaussians. A
token's velocity is copied onto each of its Gaussians, so under the adaptive-K
router a Gaussian mean would charge a K=3 token three times what a K=1 token
pays for the same velocity. They keep one Gaussian per token
(`gaussian_slot_index == 0`); with a fixed count that is exactly the Gaussian
mean. Router checkpoints trained before this change optimized and logged the
K-weighted means, so resuming one moves its `loss_velocity_l2*` and
motion-diagnostic curves onto the token means.

The historical single-term priors failed in opposite directions, which is why
V11 splits them.
Charging the *sum* makes the cross term `2<v_init, v_offset>` payable by
cancellation rather than by being right: V5's residual became a pure shrink
operator at median `cos(init, offset) = -0.997`, and the V8 `finall2` run ended at
`|v_init| = 13.4`, `|v_offset| = 13.0`, `|v_total| = 0.50` m/s -- an offset that
existed only to erase its own initializer. Charging *only the offset* removes that
incentive but leaves the matcher unbounded, and then `v_init` is worth exactly
what the matcher is worth: 0.9-1.8 m/s under V6/V7/V7.1's descriptor matchers,
4.5 under V7.2's, and **64.3 m/s** under V8's broken Top-K readout -- the same
architecture as the `finall2` run above, differing only in this key. Under
offset-only reg `|v_offset|` also sits at 0.33-0.55 m/s in every such run, i.e.
the prior pins the correction to near-nothing. Splitting bounds `v_init` directly
while leaving the correction free to be worth what it explains.

**V11.1** is V11 with two changes, neither of them inside correspondence.

The first is `qk_norm` with **one gain per head group** instead of one shared by
all twelve heads. Under RMSNorm `|q| = gamma_rms * sqrt(head_dim)` holds exactly
for every head, so a shared gain pins every head to one logit temperature by
construction. That is right when the heads share a job and wrong here: head 0
needs a logit gap near 6.8 for its readout to be metric -- about 37.7 tokens sit
inside a 2 m ball against about 3735 candidates the barrier admits -- while the
RoPE feature heads want to keep averaging broadly. V10's shared gain climbed
1.108 to 1.401 over five epochs with no plateau, i.e. something was demanding
sharpness and every head was paying for it. The gain splits on exactly the
head-0/heads-1..H-1 boundary the position encoding already uses, costing 864
extra parameters, and `attention_temperature_stats` reports
`qk_max_logit_bound_match` and `qk_max_logit_bound_rope` separately so the match
head can finally be watched on its own.

Three epochs of measurement say the split is doing real work, and that the two
groups want opposite things at different depths. Ten of twelve layers diverge by
more than 5%: head 0 runs 15-25% hotter than the RoPE heads in layers 1-5 (layer
4: 1.308 against 1.045, a logit ceiling of 10.27 against 6.59) and 4-7% cooler in
layers 8-11. Layer 4's RoPE gain barely moves at all across two epochs
(1.037 -> 1.045) while its match gain goes 1.172 -> 1.308. V11's single shared
gain sits *above both* at nearly every layer (layer mean 1.368 against 1.292 and
1.215), which is what a compromise pulled by twelve summed gradients looks like.
The divergence widens every epoch rather than converging back.

The second change is that the **adaptive-K Gaussian router is removed**. V11
and V11.2 route `LN(f')` through a hard Gumbel-Softmax over K={1,2,3} and
run only the selected K-specific joint head, so one token can emit up to three
Gaussians that all share its velocity. V11.1 instead uses the fixed-count
contract described under "Gaussians per token" below, and its shipped overlay
carries no
`p2g.grid_query` block, which means one Gaussian per occupied token at that
token's observed medoid. There is no count predictor, no straight-through
opacity gate, no per-K gradient balancing and no seed bank. Token count and
Gaussian count become the same number. At the time that also removed an index
expansion which weighted every per-token diagnostic and regularizer by that
token's K; the router variants now count each token once as well. Measured peak
memory is unchanged (31.1 GiB against V11's 31.4 at 12 layers, `batch_size=2`).

An **earlier V11.1 also gated the readout** by how much attention mass sat within
`support_rho_m` of it, through a saturating ramp `clip((W - lo)/width, 0, 1)` with
`lo` learned. That gate has been removed, and the reason generalizes. `lo`
received gradient only from tokens inside the ramp, and that band is a sliding
window that always sits on the least-supported tokens still open. Their render
gradient always says "close me", and a token already closed is clamped, so it can
never argue back -- the counterfactual is invisible to the optimizer. `lo`
therefore climbed at a constant 0.033/epoch (0.006 -> 0.026 -> 0.059 -> 0.092,
perfectly linear) while the statistic it thresholded grew far more slowly (W p90
0.016 -> 0.026 -> 0.032). By epoch 3 the gate was shut on 98.4% of Gaussians and
was discarding 92% of the matcher's displacement (`|v_init|` 0.049 m/s against an
ungated `|v_match|` of 0.644), and V11.1 trailed V11 on depth, raydrop and
chamfer at every epoch boundary. Abstention needs a *distribution-referenced*
threshold -- a running quantile of the support statistic, say -- not an absolute
one learned by gradient.

**V11.3** is V11 with the router replaced by a fixed count of **two** Gaussians
per occupied token, seeded at that token's own range quantiles. Every other
configured value is identical to V11: it composes to the same mapping apart from
`model.variant`, the derived `exp_name`, and the `p2g.grid_query` block. It also
reuses V11's backend class outright, and a legacy count config sends that class
down its fixed-count branch, so the router head is never constructed.

The two seeds are the token's own raw points at range ranks `floor(N/4)` and
`floor(3N/4)`. That is the same
`rank(s, K) = floor(((2s + 1) * N) / (2K))` rule V11's candidate bank uses for
its K=2 row, so V11.3's pair is exactly the pair V11 would have used on a token
it routed to two. A token holding a single raw point repeats it, and the
per-slot parameter blocks separate the pair afterwards.

Removing the router removes the count predictor, the straight-through opacity
gate, the per-K gradient balancing and the candidate bank with it. Because the
count is now the same for every token, a mean over Gaussians is exactly a mean
over tokens. V11's router (mean K 1.638) used to K-weight the velocity priors
and motion diagnostics; they now count each token once there too.

The **Gaussian head keeps the rest of what the router's decoder did**, so the
count is the only thing that differs. V11's decoder is two layers around the
router:

```
LayerNorm(432) -> cat(9 deltas) -> Linear(441,432) -> SiLU -> Linear(432, K*42)
LayerNorm(432) -> cat(6 deltas) -> Linear(438,432) -> SiLU -> Linear(432, w*2)   # V11.3
```

Nine delta channels become six because two fixed slots replace three candidates.
The quantity and units are unchanged: seed minus Utonia cell centre, transformed
to the reference frame, raw metres rather than cell-normalized.

Both halves matter and neither is decoration. The deltas are what make a token's
slots distinguishable at all -- they share one token feature, so without them the
slots would receive identical input and could only diverge through the parameter
rows each owns. The trunk is the only nonlinearity either decoder has; the
other fixed-count variant (V11.1) is a single affine map of the normalized
feature, and V11.3 would have been one too. The head is 226,884 parameters
against V11's 300,060 decoder, the gap being three K-specific output heads
collapsing to one fixed width.

The offset projection stays zero-initialized, so a fresh run still starts every
Gaussian exactly on its own seed.

`seed_conditioned` and `trunk` are properties of the V11.3 backend, not config
keys, and both default to off in `GaussianAttributeHead`. The backend turns
`seed_conditioned` on only under `seed_mode: range_quantile`; the shared-anchor
modes build `Linear(dim, dim) -> SiLU` and read the token feature alone. V11.1
checkpoints were saved from the plain affine form and restore unchanged.

V11.3 is deliberately absent from `ROUTER_CAPABLE_DYNAMIC_VARIANTS` even though
its backend class owns the router: `count_mode=learned_gumbel` is rejected
rather than quietly turning V11.3 back into V11. `p2g.grid_query.K_max` and
`seed_mode` remain overridable to select another fixed count or seed rule.

V11's trained router settled at a mean K of 1.638 (58.7% K=1, 18.8% K=2, 22.5%
K=3) and 91.6k Gaussians per step at `batch_size=2`, so a fixed two is about 22%
more Gaussians. Peak memory does not follow: on the smoke pair at 12 layers and
`batch_size=2`, V11.3 measures 31.1 GiB against V11's 31.4, because the router's
trunk, K-specific heads and seed bank cost roughly what the extra Gaussians do.
Trainable parameters go 7.48M to 7.22M on the same measurement.
`gaussian_head.initial_opacity` stays at 0.2 rather than being rescaled to the
higher density, which is what V11.1 did when it went the other way to a single
Gaussian per token.

**V11.4** is V11 with the correspondence barrier made linear. Every configured
value is identical to V11 except `model.variant`, the derived `exp_name`, and the
two temporal keys that shape the hinge:

```
V11     bias = -4 * relu(d / R - 1)^2       # barrier_weight 4, barrier_exponent 2
V11.4   bias = -1 * relu(d / R - 1)         # barrier_weight 1, barrier_exponent 1
```

`R = barrier_speed_mps * |dt|` in both, so both are still exactly zero inside the
radius a 30 m/s object could cover over the endpoint interval, and both still
share V11's backend, router, offset head and loss weights. What changes is only
how the penalty grows outside the radius:

| distance | V11 penalty | V11.4 penalty |
| --- | --- | --- |
| `R` | 0 | 0 |
| `1.5R` | 1.0 | 0.5 |
| `2R` | 4.0 | 1.0 |
| `3R` | 16.0 | 2.0 |

The quadratic starts flat at the boundary and accelerates, so a key just past the
radius is barely discouraged while a distant one is foreclosed outright. At init
QK-Norm holds the content logit spread near +-1, which means V11's hinge is below
the content signal near `R` and far above it past `2R`. The linear hinge has the
same slope everywhere outside and stays inside that range across the distances a
real mismatch occupies, so a match beyond the speed limit remains arguable on
feature evidence rather than being decided by geometry alone.

`barrier_exponent` is a temporal config key available to every barrier variant,
defaulting to `2`. `relu` is positively homogeneous, so `a * relu(d/R - 1)^p`
equals `(a/R^p) * relu(d - R)^p` at `p=1` exactly as at `p=2`; the hinge still
folds into the score GEMM's `beta * C` accumulate and nothing about the chunked
softmax or its backward recomputation changes.

Because the barrier is a constant of the graph rather than a parameter, V11.4 has
V11's parameter names and shapes exactly. The two are weight-compatible and
behaviourally different; what keeps a checkpoint's identity is the config
embedded in it.

**V11.5** is V11.4 with the router replaced by V11.3's fixed count of **two**
range-quantile Gaussians per occupied token. Equivalently, it is V11.3 under
V11.4's linear hinge (`barrier_weight: 1`, `barrier_exponent: 1`). The temporal
stack, correspondence, offset head, velocity regularizer and loss weights are
V11.4's. It reuses V11.3's `SeedConditionedBarrierVelocityGaussianBackend`,
seed-conditioned trunk head included, so its parameter names and shapes are
V11.3's exactly. Neither the hinge nor the offset bound is a parameter, so what
keeps the two checkpoints apart is the embedded config.

The V11.5 overlay also pins `p2g.head_offset_bound` to 0.4 m and the data
contract: the inherited `dataset_name: nuscenes` (legacy `nuscenes/legacy.py`),
`data.mode: bbox`, `data.bbox_json_path: null`, and train-only whole-scene
holdouts `data.train_excluded_scenes: [scene-0757, scene-1094]`. Both equal what
V11.4's overlay composes to now, so a fresh V11.4 and V11.5 differ only in
`p2g.grid_query`. V11.4 checkpoints trained earlier embed a 0.8 m bound and no
scene holdouts.

Like V11.3, V11.5 is not router-capable, so `count_mode=learned_gumbel` is
rejected. On the smoke pair at 12 layers and `batch_size=2` it emits 58,686
Gaussians and peaks at 31.2 GiB, the same as V11.3.

**V11.6** is V11.5 with a fixed count of **three** range-quantile Gaussians per
occupied token, at raw-point ranks `floor(N/6)`, `floor(N/2)` and
`floor(5N/6)` (the adaptive router's K=3 candidate row), trained on the
random-duration `nuscenes_dynamic_lidar4d` loader instead of `nuscenes/legacy.py`. The
temporal stack, linear hinge, correspondence, offset head, 0.4 m offset bound,
velocity regularizer and loss weights are V11.5's. It reuses
`SeedConditionedBarrierVelocityGaussianBackend`, so its parameter names and
their order are V11.5's, but the trunk reads 9 seed-delta columns instead of 6
and every attribute projection is three slots wide. A V11.5 checkpoint
therefore does not load into V11.6.

Its data block is exactly the contract the V11.4 dynamic-loader run was
launched with from the CLI: `dataset_name: nuscenes_dynamic_lidar4d`,
`mode: nvs`, `bbox_json_path: null`, an empty legacy
`train_excluded_scenes`, and `dynamic_window` with endpoints 0.2-1.2 s,
`eval_seed: 42` and the `scene-0757`/`scene-1094` holdouts. Like V11.5 it is not
router-capable. On the smoke pair (val indices 0 and 1 under its own loader) at
12 layers and `batch_size=2` it emits 87,807 Gaussians from 29,269 tokens, and
the `--memory-only` probe, which runs P2G and G2G without the rasterizer, peaks
at 31.2 GiB.

**V12** is V11.4 with only that fixed max-speed scalar made learnable. The
temporal module registers one scalar parameter shared by every layer, both
endpoint directions, and all samples:

```
v_0   = Parameter(log(30.0))
v_max = exp(v_0)
R_i   = v_max * |dt_i|
```

`barrier_speed_mps: 30.0` in the overlay is therefore an initialization value,
not the speed used as a fixed hyperparameter after construction. The learned
`v_0` is part of the checkpoint and the ordinary optimizer parameter list.
`attention_temperature_stats` logs `barrier_v_0`, the current learned value as
`barrier_speed_mps`, and the initialization as `barrier_speed_initial_mps`.

V12 uses the legacy `src/dataloader/nuscenes/legacy.py` contract, as every V11-V13
overlay except V11.6 and V12.2 does: its overlay names `data_config: nuscenes`,
the profile with `data.dataset_name: nuscenes`, the fixed keyframe/window
settings, `data.mode: bbox`, `data.bbox_json_path: null` and
`data.train_excluded_scenes: [scene-0757, scene-1094]`. The legacy loader
removes those complete scenes only when constructing the train split. V11.6,
V12.2 and V14 onward (`data_config: nuscenes_dynamic`) hold out the same two
scenes under `data.dynamic_window`, where their loader reads them. Validation and test scenes remain unchanged. The key is
checkpoint-protected sampling state, so a resume cannot silently put either
held-out scene back.

The QK RMSNorm path is exactly V11.4's: each layer has one Q gain vector and one
K gain vector, and each is broadcast across all twelve heads. V12 does not use
V11.1's separate match/RoPE gain rows. To let the reconstruction objective train
the radius, V12 keeps pairwise source-coordinate distances detached but retains
the scalar radius in the barrier graph; converting the radius to a Python float
or wrapping the complete barrier in `no_grad` would silently freeze `v_0`.

Log-space parameterization guarantees only `v_max > 0` and makes changes to the
parameter act multiplicatively on speed. It is **not an upper bound** and does
not by itself prevent the learned radius from growing: increasing `v_max`
removes more keys from the penalized region and can become an escape route if
the reconstruction objective prefers broader matching. V12 intentionally adds
neither a clamp nor a barrier-specific prior, so this remains the controlled
difference from V11.4; add an explicit upper-bounded transform or a prior around
`log(30)` in a separate variant if runaway control is required.

**V12.1** keeps V12's learned `v_0`, adaptive-K router,
initializer-conditioned offset head, losses, and `nuscenes/legacy.py` data contract.
Its barrier learns both the reachable radius and the penalty strength:

```
v_max = exp(v_0)                         # v_0 starts at log(30)
lambda = exp(log_barrier_lambda)         # log_barrier_lambda starts at 0
bias   = -lambda * relu(d / (v_max * |dt|) - 1)
```

`lambda` is one global scalar shared by all layers, directions, and samples.
Keeping it positive prevents the supposed penalty from becoming a reward; its
initial value one makes the fresh barrier identical to V12. The config records
this as `barrier_lambda_init: 1.0`, while checkpoints store the learned
`log_barrier_lambda`.

V12.1 also changes the affine part of QK RMSNorm on exactly the motion/RoPE
boundary:

- Head 0 is normalized with an implicit fixed gain of one. No head-0 gamma is
  registered as a parameter, so its Q/K vector scale cannot grow in training.
- Heads 1-11 share one learned gamma vector for Q and one for K at each layer.
  Those vectors are independent of head 0 and retain feature-head temperature
  learning.

The normalization statistic was already computed per head; "split" here means
the gain contract. At initialization V12.1 reproduces V12 because every gain is
one. Afterwards `qk_gamma_*_match_layer*` stays exactly one while the separately
logged `qk_gamma_*_rope_layer*` may move. V12.1 also pins
`p2g.head_offset_bound` to 0.4 m, so each Gaussian position offset is
`0.4 * tanh(raw_offset)` per axis. That is now the shared default too; V12
checkpoints trained under the earlier 0.8 m default keep it.

**V12.2** is V12 with its learned max speed split by layer and moved from an
exponential to a softplus:

```
v_raw_l = Parameter(softplus^-1(30.0))   # l = 0..11, one per temporal layer
v_max_l = softplus(v_raw_l)
R_l,i   = v_max_l * |dt_i|
```

Head 0 of layer `l` reads correspondence under its own radius `R_l,i`, shared
by both endpoint directions and every sample. Every layer starts at
`barrier_speed_mps: 30.0`, so a fresh V12.2 reproduces V12 exactly, and the
checkpoint stores the twelve-entry `v_raw` where V12 stores `v_0`. Everything
else is V12's: shared per-layer QK RMSNorm gains, no `lambda`, the adaptive-K
router, the initializer-conditioned offset head and the split velocity prior.
The overlay pins `p2g.head_offset_bound` to the 0.4 m shared default.

Softplus keeps each speed positive but changes what an optimizer step does to
it. Its slope `sigmoid(v_raw)` is one to float precision at 30 m/s and still
0.95 at 3 m/s, so a step moves `v_max_l` by about the same number of metres per
second at any speed, whereas a step on V12's `v_0` multiplies the speed by a
constant factor. Adam moves a parameter by at most about the learning rate per
step, so at `lr: 5e-4` a speed changes by at most about 0.5 m/s per 1,000
optimizer steps, less as the cosine schedule decays. There is still no clamp
and no prior toward 30 m/s. `attention_temperature_stats` logs
`barrier_v_raw_layer{l}`, `barrier_speed_mps_layer{l}`,
`barrier_speed_mps_min/mean/max` and `barrier_speed_initial_mps`; the scalar
`barrier_speed_mps` is dropped because no single current speed exists.

Unlike V12, V12.2 trains on the random-duration `nuscenes_dynamic_lidar4d`
loader with V11.6's data block (see "Random-duration windows" below).

**V13** keeps V12.1's fixed-window `nuscenes/legacy.py` data contract, complete-scene
train holdouts, embedded-init offset head, split velocity regularizer,
adaptive-K router, and 0.4 m Gaussian offset bound. It replaces the complete
cross-attention score contract:

```
head 0:     score_ij = dot(RoPE(L2(q_i)), RoPE(L2(k_j))) / 0.03
heads 1-11: score_ij = dot(RoPE(q_i),     RoPE(k_j))     / sqrt(head_dim)
```

No head has RMSNorm or a learned Q/K gain. Head 0 alone has parameter-free L2
normalization; heads 1-11 retain their raw projected magnitude. The V12/V12.1
max-speed barrier is absent, so `v_0`, `log_barrier_lambda`, all pairwise
distance work, and every barrier config key disappear. Metric 3D RoPE at
`base=100`, `position_scale=2*pi/5` is applied to all twelve heads, including
head 0.

The motion temperature is fixed at **0.03**, not learned. A unit-normalized head
under the ordinary FlashAttention scale would have logits only in
`[-1/sqrt(36), +1/sqrt(36)]`, which is essentially uniform over the several
thousand opposite-frame keys. At 0.03 its maximum absolute content logit is
`1/0.03 = 33.3`. The stage-3 LiDAR4D val windows hold 5,582-7,923 keys per
frame, so one key needs a logit gap of `ln(N) ~ 8.8` over the rest just to hold
half the mass and about 13.6 for 99%: a cosine gap of 0.27 / 0.41 at 0.03,
against 0.63 / 0.95 at the 0.07 the earlier direct matchers used over 4-16
candidates or under a distance prior. Unconcentrated mass is not neutral: the
uniform readout `centroid(keys) - x` points at the key frame's sensor (median
cosine 0.985, median length 20.4 m), so a flat head 0 drags tokens toward the
ego.

All heads run in **one varlen FlashAttention call per layer**. The native
stage-3 width is 36 channels per head. V13 appends three opposite-frame xyz
channels to head 0's V payload and zero-pads Q/K/V to 40 channels for the Flash
kernel; the Q/K padding is zero, so it changes no score. FlashAttention exposes
one scale for the whole call, therefore head 0's normalized Q and K are each
multiplied by `sqrt(1 / (0.03 * (1/sqrt(36))))`. After the shared
`1/sqrt(36)` scale this is exactly cosine/0.03, while all other heads retain the
ordinary dot-product scale. The returned xyz slice is `W_l @ key_xyz`; only
after subtracting the query coordinate does it become displacement. Plain
learned layer logits mix the twelve per-layer coordinate expectations as in
V11-V12.1.

V13's adaptive K={1,2,3} router uses the existing `GridSlotHead`, range-quantile
candidate bank and hard Gumbel-Softmax STE; no count-budget objective exists.

**V13.1** is V13 with only the count mechanism changed. The router, count
predictor, opacity STE gate and K-specific output heads are removed; every token
emits exactly two Gaussians. `grid_query.K_max=2, seed_mode=range_quantile`
chooses the token's
raw points at `floor(N/4)` and `floor(3N/4)`, exactly the adaptive bank's K=2
range-quantile row. The fixed Gaussian decoder remains seed-conditioned, so the
two slots see their six metric seed-delta channels. Temporal parameters,
temperature, RoPE, motion head, data, losses and the 0.4 m offset bound are
otherwise identical to V13.

**V13.2** is V13 with only head 0's content score changed. It removes the
parameter-free L2 normalization and the fixed `motion_temperature`; head 0 now
uses the same raw projected Q/K contract as heads 1-11:

```
all heads: score_ij = dot(RoPE(q_i), RoPE(k_j)) / sqrt(head_dim)
```

There is still no Q/K RMSNorm, distance barrier, `v_0`, or barrier lambda. The
all-head 3D RoPE path, 36-to-40 channel FlashAttention packing, head-0 xyz V
payload, per-layer coordinate readouts and learned layer mixture are unchanged.
Unlike V13, raw Q/K magnitude can now change the head-0 softmax sharpness during
training; V13.2 therefore has no finite norm-derived logit bound and no
temperature config key. Adaptive K={1,2,3}, range-quantile seeds, the 0.4 m
offset bound, data, motion head and losses remain exactly V13's.

**V13.3** starts from V11.4's adaptive-K, twelve-layer learned match mixture,
initializer-conditioned velocity offset, split velocity regularizer and loss
contract. It removes head 0's linear max-speed soft barrier and applies the
same metric 3D RoPE to all twelve heads. Q and K are RMS-normalized per head
before RoPE, using V11.4's one learned Q gain vector and one learned K gain
vector per layer, each shared across heads:

```
all heads: score_ij = dot(RoPE(RMSNorm(q_i)),
                          RoPE(RMSNorm(k_j))) / sqrt(head_dim)
```

There is no distance bias, barrier radius, barrier weight, temperature, or
pairwise-distance score computation. Head 0 still carries opposite-frame xyz
in its V payload for the coordinate expectation, while every head is packed
into the same 36-to-40-channel varlen FlashAttention call once per layer.
Consequently the Q/K RMS gains give a finite content-logit bound, but geometry
enters only through 3D RoPE rather than an additive distance term. V13.3 uses
V13.2's `nuscenes/legacy.py` data contract and train-only `scene-0757`/`scene-1094`
holdouts, and sets `p2g.head_offset_bound=0.4`. All other model and loss values
remain those of V11.4.

**V13.4** is V13.3 with 3D RoPE removed from correspondence head 0 only:

```
head 0:     score_ij = dot(RMSNorm(q_i),
                           RMSNorm(k_j)) / sqrt(head_dim)
heads 1-11: score_ij = dot(RoPE(RMSNorm(q_i)),
                           RoPE(RMSNorm(k_j))) / sqrt(head_dim)
```

All heads retain their per-layer Q/K RMSNorm modules and shared learned gain
contract. Head 0 still carries opposite-frame xyz in V and supplies every
per-layer coordinate expectation. Selectively skipping RoPE requires no split
attention kernel: Q/K are transformed per head first, then all twelve heads use
the same packed varlen FlashAttention call. V13.4 therefore differs from V13.3
only in `temporal.position_encoding=rope_except_head0`; adaptive K, time
conditioning, layer mixture, motion head, regularizers, losses,
`head_offset_bound=0.4`, and the `nuscenes/legacy.py` scene-holdout contract are
identical. It has no distance bias, barrier, or separate temperature.

**V14** is V11.4's dynamic model -- linear 30 m/s barrier on match head 0,
RoPE heads 1-11, parameter-only layer mixture, adaptive-K router, embedded-init
offset head, split velocity prior and loss weights -- trained fresh from the
pretrained Utonia checkpoint with a different Utonia adaptation
(`p2g.utonia_adaptation.mode: hybrid`, specified in
`docs/research/utonia_finetuning_2026_09_25/adapter_ln_lr_spec_2026_09_26.md`).
Only the stages the encoder executes (stem, enc0-enc3) are adapted, and enc4
receives nothing:

| Part | V11-V13 | V14 |
| --- | --- | --- |
| Stem XYZ columns and bias | frozen, LoRA on the Linear | frozen, no LoRA |
| Stem intensity | frozen zero base column, reached through LoRA | own column `w_I` (54, zero init), trained directly |
| Stem residual | linear LoRA 4->16->54 | `4 -> 16 -> 54` GELU adapter on `[x, y, z, I]`, zero output at init, added before the stem LN |
| FFN `fc1`/`fc2` | frozen base + LoRA | plain frozen Linears, no LoRA |
| FFN branch | none | `C -> b -> C` ReLU adapter on the same `norm2` output, `y = u + FFN(z) + 0.1 * adapter(z)`, `b = 8/16/32/64` for enc0-enc3 |
| LayerNorm affine (67 LNs) | frozen | pretrained gamma/beta, trained |
| Attention qkv/proj, CPE conv/Linear, pooling proj | LoRA r16/a16 | LoRA r16/a16 (fresh A/B) |

The adapters start at zero output (`down` Kaiming-uniform with zero bias, `up`
weight and bias zero), so a fresh V14 encoder reproduces the pretrained Utonia
function exactly: on a real two-frame val batch its stage-3 output matched the
unadapted checkpoint bit for bit under the same serialization seed. Utonia then
trains 4,423,130 parameters: `w_I` 54, stem adapter 998, FFN adapters 725,238,
LayerNorm 39,528 and LoRA 3,657,312. LoRA wraps an explicit allow-list
(`lora_targets: [attention, cpe, pooling]`) resolved before any adapter is
attached; `ffn` and `stem` targets are rejected, since an FFN LoRA and the
parallel FFN adapter are mutually exclusive by design.

AdamW gets one parameter group per role, each read from
`train.utonia_param_groups`, and a last `downstream` group for everything after
Utonia at `train.lr` / `train.weight_decay`:

| Group | LR | weight decay |
| --- | ---: | ---: |
| `utonia_intensity` (`w_I`) | `train.lr` (5e-4) | 1e-4 |
| `utonia_adapter_weight` (stem + FFN adapters) | `train.lr` (5e-4) | 1e-4 |
| `utonia_adapter_bias` | `train.lr` (5e-4) | 0 |
| `utonia_lora` | `train.lr` (5e-4) | 1e-4 |
| `utonia_layer_norm` | `train.lr` (5e-4) | 0 |
| `downstream` (temporal, router, Gaussian and motion heads) | `train.lr` (5e-4) | 1e-4 |

Since 2026-10-08 every hybrid overlay (V14 through V16.1) sets each Utonia
role's `lr` to `${train.lr}`, so the whole model trains at one learning rate.
Each role keeps its own weight decay. Before that date, `w_I` and the adapters
trained at 1e-3 and LoRA and LayerNorm at 5e-5. Checkpoints embed their
resolved rates, so earlier runs resume unchanged.

Roles come from module types, so the 1-D `w_I` is a weight rather than a bias
and a new adapter deep inside a pretrained block still trains at the adapter
rate. The one warmup (500 optimizer steps) + cosine schedule scales every group
alike, grad clip 1.0 covers all of them, and the overlay records
`adam_betas: [0.9, 0.999]` and `adam_eps: 1e-8`. The Utonia backbone stays in
eval mode (no DropPath); only LoRA and the adapters follow `train()`. The
V11-V13 variants keep their single flat parameter group, and they reject both
`p2g.utonia_adaptation` and `train.utonia_param_groups`.

Each step logs `utonia/stem_{xyz,intensity,adapter}_rms`, the three pre-LN
stem summands, and `utonia/ffn_adapter_rms_ratio_enc{0..3}`, the scaled adapter
output RMS over the frozen FFN output RMS averaged over each stage's blocks, so
initial drift of the new paths can be read against the frozen ones.

Two further differences from the V11.4 dynamic-window run it is compared with:
`learned_count.seed_mode: token` anchors every slot at the Utonia token position
and gives the router trunk the refined feature alone, and the run uses the
fresh-run default `data.coordinate_frame: ego_vehicle` (the V11.4 run predates
that key and ran in the LiDAR sensor frame). V14 trains on the random-duration
`nuscenes_dynamic_lidar4d` loader with V11.6's data block.

**V14.1** is V14 with its barrier speed learned as one global scalar through a
softplus, V12's single speed in V12.2's parameterization:

```
v_raw = Parameter(softplus^-1(30.0))
v_max = softplus(v_raw)
R_i   = v_max * |dt_i|
```

All twelve layers, both endpoint directions and every sample share that
radius, and a fresh V14.1 reproduces V14. `v_raw` sits in the `downstream`
group, so at `lr: 5e-4` the speed moves by at most about 0.5 m/s per 1,000
optimizer steps; there is no clamp and no prior toward 30 m/s.
`attention_temperature_stats` logs `barrier_v_raw`, the current speed as
`barrier_speed_mps`, and `barrier_speed_initial_mps`. Every other config value
is V14's.

**V15** is V14.1 with five changes. Every key it adds is opt-in: an overlay
without it keeps today's behaviour, and V11-V14.1 reject the V15-only values, so
their checkpoints resume unchanged (same module tree, state-dict keys and
optimizer groups).

| Part | V14.1 | V15 |
| --- | --- | --- |
| Utonia stem adapter | `4 -> 16 -> 54` on `[x, y, z, I]`, added before the stem LN | no adapter before the LN; a `54 -> 16 -> 54` GELU residual adapter on `h = GELU(LN(W x))` (`stem.adapter_position: post_activation`) |
| `v_raw` (learned max speed) | `downstream` group, weight decay 1e-4 | `downstream_no_decay` group, weight decay 0 (`train.downstream_no_decay`) |
| Temporal LayerScale | `cross_layer_scale` gate, init 0.1 | none (`temporal.layer_scale: false`) |
| After the temporal stack | -- | one radius local-attention block (`dynamic_2dgs.local_attention`); `v_offset` still reads the stack's output |
| Gaussian count | `learned_gumbel` router, K = 1, 2, 3 | `nested_gumbel`: five decoded Gaussians, a router over {1, 3, 5}, two training renders |

**Stem.** `p2g.utonia_adaptation.stem.adapter_position` is `pre_norm` (the
default when the key is absent; V14/V14.1) or `post_activation` (V15). V15's
pre-LN stem is the frozen XYZ columns, the bias and the zero-initialized
`w_I`, nothing else. The adapter adds `B GELU(A h)` to `h`, with `A` 54x16 plus
bias and `B` 16x54 plus bias, `B` starting at zero, so a fresh stem is exactly
the pretrained one. The stem LayerNorm stays trainable. The adapter's 1,798
parameters replace V14's 998 and land in the same `utonia_adapter_weight` /
`utonia_adapter_bias` groups, so Utonia trains 4,423,930 parameters (V14:
4,423,130). V14 and V14.1 reject `post_activation`; V15 requires it. The stem
statistics logged as `utonia/stem_*` are `stem_xyz_rms` and
`stem_intensity_rms` (pre-LN summands), `stem_activation_rms` (RMS of `h`) and
`stem_adapter_rms` (RMS of what the adapter adds to `h`); `pre_norm` keeps its
`xyz`/`intensity`/`adapter` trio.

**Optimizer groups.** `train.downstream_no_decay` (optional) lists downstream
parameters by name relative to Point2Gaus. They leave `downstream` for a last
group named `downstream_no_decay` at `train.lr` with weight decay 0; V15 lists
`dynamic_backend.temporal.v_raw`, and every other downstream parameter keeps
`train.weight_decay`. Each name must match exactly one trainable downstream
parameter, or building the optimizer raises and lists the unknown or frozen
names. The list must hold unique non-empty strings and may not name a Utonia
(`feature_extractor.*`) parameter, whose decay comes from
`train.utonia_param_groups`. Only the hybrid variants (V14, V14.1, V15) build
role groups, so only they accept the key; absent or empty gives exactly the
groups it always did.

**Temporal LayerScale.** `dynamic_2dgs.temporal.layer_scale` (bool, absent =
true) switches the per-layer gate. `false` builds no `cross_layer_scale`
parameters, so each layer computes `x = snapshot + out_proj(attended)` with
`out_proj` at its default initialization (the FFN's last layer still starts at
zero), and `layer_scale_init` must then be absent. V15 requires `false`; every
other variant accepts only an absent or `true` value, which is the historical
gate.

**Local attention.** V15 alone carries a `dynamic_2dgs.local_attention` block,
and every key is required:

```yaml
dynamic_2dgs:
  local_attention:
    radius_m: 0.8
    num_heads: 12
    mlp_ratio: 4
    rope_base: 15.848931924611133            # 10^1.2
    rope_position_scale: 15.707963267948966  # 2*pi/0.4 -> wavelengths 0.4-4.0 m
    implementation: flash_varlen             # flash_varlen | sdpa | auto
    query_chunk_size: 4096                   # at most 65535
```

`radius_m`, `mlp_ratio`, `rope_base` and `rope_position_scale` are positive and
finite. `num_heads` must divide the token width into heads whose width is
divisible by six (432 / 12 = 36), the same xyz/sin-cos 3D RoPE constraint as the
temporal stack. Each query is one FlashAttention2 varlen sequence, so
`query_chunk_size`, how many queries go into one call, may not exceed the
kernel's 65535-sequence limit; larger values are rejected when the config is
validated rather than when the kernel launches.

**Count.** See "Gaussians per token" below for the `nested_gumbel` contract.
V15 keeps V14.1's temporal stack, loader and coordinate frame; its overlay
differs from V14.1's in exactly the five changes above.

**V15.1, V15.2, V15.3, V16 and V16.1** keep every V15 key above -- stem,
optimizer groups, temporal stack without LayerScale, local attention, loader --
and change only how the Gaussian head turns a token into Gaussians (V15.4 and
V15.5, below the table, change the temporal stack instead):

| Variant | Count contract | Rendered per token | Training renders |
| --- | --- | --- | --- |
| V15 | `nested_gumbel` | router: prefix {G1} / {G1..G3} / {G1..G5} of one G1..G5 decoder | routed set + every token's G1..G3, mean of the two losses |
| V15.1 | `nested_gumbel` | as V15 | routed set only |
| V15.2 | `expert_gumbel` | router: expert E1 / E3 / E5 on one shared trunk | routed set + every token's E5, mean of the two losses |
| V15.3 | `legacy`, `K_max: 5`, `seed_mode: token` | all five decoded Gaussians, no router | one render |
| V16 | `nested_ordinal` | V15's prefixes, chosen by a keep gate and an add gate (cosine logits, \|u\| <= 3) | every token's G1..G5 with blend masks (masked rasterization) + every token's G1..G5 unmasked at 2 views, mean of the two losses |
| V16.1 | `nested_ordinal` | as V16 | V16's masked routed set + every token's G1..Gk unmasked at 2 views, k drawn uniformly from {1, 3, 5} per token and step, mean of the two losses |

V15.2's head is `LN(f) -> Linear(432, 576) -> SiLU` shared, then three linear
experts `576 -> 1x42`, `576 -> 3x42`, `576 -> 5x42` (the second layer predicts
1 + 3 + 5 Gaussians); the router is V15's. Every Gaussian of the chosen expert
carries that option's opacity gate, so every token sends the router a gradient,
reduce included. V15.3's head is V15's decoder, `LN(f) -> Linear(432, 576) ->
SiLU -> Linear(576, 5x42)`, module for module, without the router; it is also
V15.2's E5 path. See "Gaussians per token" for both contracts.

**V15.4** is V15.3 with `dynamic_2dgs.temporal.num_heads: 8` instead of 12, so
each temporal head is 54 channels wide instead of 36. Head 0 stays the barrier
match head and heads 1-7 keep 3D RoPE; 54 is divisible by six, so the xyz
sin/cos split holds with nine bands per axis instead of six.
`temporal.rope_base` moves from 100 to 10^1.875 so the nine bands cover
V15.3's span exactly: the longest band is `5 m * base^((d/6 - 1)/(d/6))` at
head width `d`, and 100^(5/6) = (10^1.875)^(8/9) = 46.4, i.e. 232 m (base 100
would stretch it to 300 m). The bands are 5, 8.1, 13.1, 21.1, 34.1, 55.0, 88.9,
143.6 and 232.1 m. FlashAttention2 pads the 54-wide heads to 56 internally, as
it pads V15.3's 36 to 40. The local-attention block keeps twelve heads. V15.3
and V15.4 each pin their head count (12 and 8); any other value is rejected.
The checkpoint differs from V15.3's only in the per-layer Q/K RMSNorm gains,
which are one head wide.

**V15.5** is V15.4 whose `v_init` reads the plain mean of the twelve layers'
head-0 coordinate expectations, `matched_position = (1/L) * sum_l mu_l`, instead
of V15.4's `softmax(layer_logits)` mixture over twelve learned scalars. Zero
logits are that same uniform mixture, so a fresh V15.5 and a fresh V15.4 agree;
V15.5 only keeps it fixed at `1/L` for the whole run. Its overlay is V15.4's key
for key except `model.variant`: the change lives in its backend class,
`UniformLayerDenseCountBarrierVelocityGaussianBackend`, whose temporal stack
(`UniformLayerMeanSoftplusBarrierCrossAttention`) is V14.1's without
`layer_logits`. The checkpoint is V15.4's minus `temporal.layer_logits`, and,
like V11.2, it logs no `motion_layer_weight_*` diagnostics because no mixture is
learned.

**V17.1** is V15.4 with Utonia adaptation restricted to its new intensity input
column (`p2g.utonia_adaptation.mode: intensity_only`, `utonia_lora.enable: false`).
The original pretrained 9-to-54 stem supplies its frozen XYZ columns and bias;
the other six input columns are discarded. A separate zero-initialized
54-element parameter adds `I * w_I`. Every pretrained Utonia parameter,
including LayerNorm, remains frozen, and no LoRA or nonlinear adapters are
installed. Autograd still traverses the encoder to train `w_I`. Only the
`utonia_intensity` optimizer role exists, using V15.4's LR and weight decay;
the downstream groups, learned layer mixture, eight temporal heads, local
attention, dense five-Gaussian decoder, losses and nuScenes data profile are
V15.4's. V17.1 checkpoints retain and protect this adaptation mode.

**V18** is V15.4 with SplatAD-style LiDAR appearance (Hess et al., CVPR 2025).
Each Gaussian slot carries a feature of `p2g.gs_params.feature: 30` channels
instead of the 32 SH coefficients (`shs: 0`), so the decoder's last layer is
`576 -> 5x40` instead of `5x42`; the feature rows start like the SH rows they
replace (N(0, 0.01) weights, zero bias). The rasterizer alpha-blends the
feature into `f'` with no background term, and `g2p.lidar_decoder`
(`src/models/module/lidar_decoder.py`, inside G2P) decodes every pixel:

```text
[f' (30), ray direction (3), median depth / range_scale_m (1), sensor embedding (30)]
  = 64 -> Linear 96 -> SiLU -> Linear 96 -> SiLU -> Linear 2 -> sigmoid -> surface (i, p)
intensity = a * i,   raydrop = (1 - a) + a * p      (a = the pixel's accumulated alpha)
```

The MLP predicts the surface the ray hits -- its intensity `i` and the
probability `p` that it still returns nothing -- and each pixel composites it
over V15.4's background (intensity 0, raydrop 1) by its alpha, exactly as the
SH render composites `C + T * bg`: a ray that hits no Gaussian returns nothing,
and through `a` the raydrop loss asks for coverage where the GT returns and
none where it drops, as in V15.4.

The ray direction is the unit ray from the rendering LiDAR's origin (a KITTI
block's own emitter origin) through the pixel -- `normalize(hit point - sensor
origin)` -- written in the axes of the window reference frame, sweep 0's ego
frame, where the Gaussians and their features live (SplatAD uses world-frame
rays too); the median depth is the renderer's `depth_median`,
`range_scale_m: 80` the sensor's effective range; `num_sensors: 1` gives every
view embedding 0. The decoded maps keep the SH render's keys (`intensity_sh`,
`raydrop`) and [0, 1] range, so losses, metrics, evaluation and the raydrop
refinement read them unchanged. The losses train the MLP and the embedding,
every Gaussian's feature, and its opacity and footprint through the blending
weights and the composite's alpha -- the rasterizer's feature channels now
differentiate their alpha like its colour channels, which they did not before
V18. The median depth enters
detached: it only tells the decoder the range, while geometry is supervised by
the depth losses; a live `r` would let the raydrop loss push the median
Gaussian's depth wherever dropping is easier to predict, unopposed on GT-dropped
pixels, which carry no depth loss. `feature_dim` must equal
`p2g.gs_params.feature` (at most 32, the rasterizer's `MAX_FEATURE_CHANNELS`),
and every other variant rejects both keys. The MLP and the embedding train in a
last `lidar_decoder` optimizer group at `train.lr` / `train.weight_decay`; the
backend checkpoint keys are V15.4's in V15.4's order, plus G2P's
`g2p_model.lidar_decoder.*`.

`barrier_weight` stays at 4 **in V11 itself**. Re-scoring V11's trained head-0
features at other
weights shows a=0 to a=4 is decisive (mass beyond the radius 11.2% -> 0.85%,
`|dp|` 8.5 m -> 2.26 m) and a=4 to a=64 changes nothing past the fourth decimal:
the residual mass sits just outside the radius, where the hinge is near zero by
construction and no weight reaches it.

The fixed time reference only conditions network features; it is not the
sample-dependent endpoint duration and is not used to reinterpret renderer
units.

`src/experiment_config/loader.py` owns composition for both training and evaluation. New
checkpoints embed the resolved config, so every checkpoint carries immutable
configuration provenance. Historical checkpoints fall back to
`<checkpoint-dir>/wandb/latest-run/files/config.yaml`; that path is a mutable
legacy fallback and is reported explicitly. Neither path merges current
checkout overlays into an older run. Checkpoint-backed commands may change
runtime/training knobs, but CLI changes under `model`, `p2g`, `g2g`, `g2p`, or
`dynamic_2dgs` are rejected if they alter the stored value. A config must name
one of the listed variants; configs saved before variants existed, and the
box-routed `bbox_rigid_v1` model, can no longer be restored.

Here, "resolved config" means the final effective mapping after base, variant,
and CLI merging, with OmegaConf interpolations such as `${model.variant}` and
`${train.batch_size}` replaced by their concrete values. It records what the
program was given, rather than only which YAML files were selected. Checkpoints
saved before the cleanup also carry the removed intensity-encoder, fusion and
refiner blocks under `p2g`; nothing reads them any more. Removed switches are
accepted only at the inert value those checkpoints stored --
`p2g.anchor_mode: grid`, `p2g.grid_query.learned_count.budget.enable: false`,
`dynamic_2dgs.regularization.small_motion.enabled: false`, `loss.w_scale: 0`,
`loss.enable_lpips: false` and `g2p.dynamic: false` (the PVG-style
`marginal_t` temporal-opacity prefilter, never implemented here) -- and
rejected otherwise, so no config silently
loses a term it asked for. `loss_scale` is still computed against
`loss.scale_max_m` and logged, but never enters the total. The selected
variant's `dynamic_2dgs` block rejects unsupported keys.

The resolved config does not currently record Git `HEAD` or uncommitted edits.
A Git SHA would identify the exact checked-out commit at launch (not necessarily
the newest commit on the remote), while a dirty flag/diff hash would be needed
to distinguish local changes made after that commit.

Because `wandb/latest-run` can point at a later resume, historical checkpoints
do not use that fallback automatically. After inspecting the artifact, opt in
with `allow_legacy_wandb_fallback=true`. Ray geometry and temporal-window fields
under `data` are protected as structural semantics; dataset paths, splits, and
worker counts remain runtime-overridable.

`logger.*` is runtime-only. A checkpoint resume opens a new W&B run under the
same name by default; pass `logger.wandb_run_id=<run id>` to continue that run
instead (`resume="must"`, so a wrong id fails at start-up). Lightning keeps
logging `trainer/global_step` from the checkpoint, so history the old run wrote
after the checkpoint would overlap the resumed steps; rewind the run to the
checkpoint's last `_step` first if that matters.

## Gaussians per token (`p2g.grid_query`)

How many 2D Gaussians one occupied Utonia token becomes is a config decision,
not a property of the variant name. `p2g.grid_query.count_mode` selects between
two contracts, and both are available to any Dynamic variant whose backend can
run them.

**Fixed count (`count_mode: legacy`, the default).** Every occupied token emits
exactly `K_max` Gaussians, which is required whenever the block is present.
`seed_mode` says where those seeds start:

| `seed_mode` | seed placement |
| --- | --- |
| `range_quantile` (default) | `K_max` deterministic range quantiles over the token's own raw points |
| `medoid` | every slot on the token's observed medoid -- the raw point nearest its own-frame Cartesian mean |
| `token` | every slot on the token's Utonia position -- the pooled cell coordinate temporal attention also uses |

`medoid` and `token` are shared anchors. Every slot of a token starts there,
and its Gaussian lands at `anchor + head_offset_bound * tanh(offset_k)` per
axis, the anchor-plus-bounded-offset form of anchor-based Gaussian decoders
(Scaffold-GS, SCube, TRELLIS). Seeds that coincide are not redundant: the
Gaussian head owns one parameter block per slot, including the bounded
position offset, so slots starting at the same coordinate learn independent
geometry. V11.3, V11.5, V11.6 and V13.1 additionally feed every slot's seed
delta into that head under `range_quantile`, so their slots also differ by
input; under a shared anchor every slot's delta would be the same vector,
which only records where the anchor sits in its voxel, so those heads read the
token feature alone. Every other fixed-count variant always reads the token
feature alone. Under `range_quantile` a token holding fewer raw points than
slots repeats observed points, the same way the learned-count candidate bank
does.

`seed_mode` replaced the numeric `exp` key (`null` -> `range_quantile`,
`1` -> `medoid`, `2` -> `token`). Checkpoint configs that still spell `exp`
are rewritten on restore and build the same model; fresh configs must use
`seed_mode`.

Omitting the whole `p2g.grid_query` block means `K_max: 1, seed_mode: medoid`:
one Gaussian per token at its medoid, which is what V11.1 ships. V11.3, V11.5
and V13.1 ship the block explicitly at `K_max: 2, seed_mode: range_quantile`,
and V11.6 at `K_max: 3, seed_mode: range_quantile`. Every fixed-count variant
supports larger counts.

Because the count is fixed and identical for every token, a mean over Gaussians
is exactly a mean over tokens. `velocity_l2` and the motion diagnostics stay
unweighted no matter what `K_max` is set to.

```bash
# V11.6's three Gaussians per token, anchored at the Utonia token position.
python main.py model.variant=dynamic_2dgs_attention_velocity_v11_6 \
  p2g.grid_query.seed_mode=token

# Two Gaussians per token, both starting at the token's observed medoid.
python main.py model.variant=dynamic_2dgs_attention_velocity_v11_1 \
  p2g.grid_query.K_max=2 p2g.grid_query.seed_mode=medoid

# Three Gaussians per token, spread over that token's raw points by range.
python main.py model.variant=dynamic_2dgs_attention_velocity_v11_1 \
  p2g.grid_query.K_max=3 p2g.grid_query.seed_mode=range_quantile
```

Peak memory and step time scale with the Gaussian count, not the token count.
Measured on one A100 at `batch_size=2` with a single temporal layer, V11.1 emits
29,343 Gaussians at `K_max=1` and 58,686 at `K_max=2` from the same tokens;
V11.3 emits that same 58,686 from its shipped `K_max: 2`.

**Learned count (`count_mode: learned_gumbel`).** The refined token routes
through the grid `GridSlotHead` and a hard Gumbel-Softmax over K={1,2,3} picks
one K-specific joint head per token. Only V11, V11.2, V11.4, V12, V12.1, V12.2,
V13, V13.2, V13.3, V13.4, V14 and V14.1 may use it; asking any other variant for it is
rejected at config time rather than silently ignored. V11.3, V11.5, V11.6 and
V13.1 are rejected too because their identities are the fixed count. The seed
geometry it consumes is the K-specific candidate bank, not the padded
fixed-count seed set, so the two contracts cannot be mixed.
`learned_count.seed_mode` fills that bank: `range_quantile` gives each row its
own raw-point quantiles, which the router trunk reads beside the token
feature, and `token` puts every slot of every row on the token anchor, so the
trunk reads the feature alone (`Linear(dim, dim)` instead of
`Linear(dim + 3 * K_max, dim)`).

**Nested count (`count_mode: nested_gumbel`).** V15 only, and it is not the
learned count above. Every token has one anchor, its Utonia position
(`seed_mode: token`, the only accepted value: the seed builder resolves it to a
single `(N, 1, 3)` token-position seed per token). A decoder places G1..G5 around
that anchor from the refined feature, and a router picks a prefix: reduce
{G1}, keep {G1..G3} or add {G1..G5}. `p2g.grid_query` then holds exactly these
two keys, and every `nested_count` key but `logging` is required:

```yaml
p2g:
  grid_query:
    count_mode: nested_gumbel
    nested_count:
      option_counts: [1, 3, 5]      # must be exactly this
      seed_mode: token              # must be exactly this
      tau: 1.0                      # Gumbel-softmax temperature, positive
      decoder_hidden: 576           # positive integer
      router_hidden: [256, 128]     # non-empty list of positive integers
      auxiliary_render: dense_keep  # dense_keep | none
      logging:                      # optional
        enable: true
        interval: 20                # positive integer
        range_edges_m: [0, 10, 20, 30, 40, 60, 80, 110]
```

The router samples its choice with hard Gumbel-softmax in training (`tau` is
fixed and shapes only the straight-through backward) and takes the argmax of
its logits at evaluation; nothing is annealed.

`auxiliary_render: dense_keep` also renders every token's G1..G3 during
training and optimizes the mean of that render's loss and the routed set's;
`none` renders the routed set alone. Evaluation renders the routed set in both.
During training the routed set carries an `opacity_gate` tensor: G1 is ungated,
G2/G3 use `y_keep + y_add`, and G4/G5 use `y_add`. Raw opacity logits remain
unchanged; the Python renderer applies the gates after sigmoid, before the
CUDA rasterizer. Selected forward opacities are bit-identical to their ungated
values, and saturated opacity does not lose router gradients to a logit clamp.
The auxiliary set and evaluation carry no gates. Existing learned-count
variants keep their historical raw-logit STE path.
`nested_gumbel` is rejected for every variant but V15 and V15.1, and both reject
`legacy` and `learned_gumbel`. `auxiliary_render` is each variant's identity:
V15 requires `dense_keep` and V15.1 (`config/variants/dynamic_2dgs_attention_velocity_v15_1.yaml`,
otherwise identical to V15) requires `none`.

**Per-option experts (`count_mode: expert_gumbel`).** V15.2 only. The token
anchor, the router (`router_hidden`, `tau`, hard Gumbel-softmax in training,
argmax at evaluation) and the block's keys are the nested count's, under
`p2g.grid_query.expert_count` instead of `nested_count`:

```yaml
p2g:
  grid_query:
    count_mode: expert_gumbel
    expert_count:
      option_counts: [1, 3, 5]      # must be exactly this
      seed_mode: token              # must be exactly this
      tau: 1.0
      decoder_hidden: 576           # the shared trunk's width
      router_hidden: [256, 128]
      auxiliary_render: dense_add   # dense_add | none; V15.2 requires dense_add
      auxiliary_views: 2            # optional; null/absent = every target view
      logging:                      # optional, as for nested_count
        enable: true
        interval: 20
        range_edges_m: [0, 10, 20, 30, 40, 60, 80, 110]
```

What changes is what each option renders. `LN(f) -> Linear(dim,
decoder_hidden) -> SiLU` is one shared trunk and `experts[i] =
Linear(decoder_hidden, option_counts[i] * 42)` reads it, so reduce, keep and
add render E1, E3 and E5 -- separate Gaussians, not prefixes of one set. All of
them start from the dynamic-head priors on the anchor. During training every
Gaussian of the chosen expert carries `opacity_gate = y_ST[chosen option]`
(forward exactly one, applied after sigmoid like V15's gates), so the router
receives `sum o * dL/do'` over the chosen expert through the softmax from
every token; in V15 a token that reduces renders only the ungated G1 and sends
nothing. Experts a token did not choose are neither rendered nor compared.
`dense_add` also renders every token's E5, ungated, during training and
optimizes the mean of the two render losses (`aux_*` / `routed_total` are
logged as in V15); evaluation renders the routed set alone.
`auxiliary_views: k` (optional in both routed blocks, a positive integer)
renders that second set at only k target views per window, drawn uniformly
from all of the window's targets -- input endpoints and novel sweeps alike --
anew every step and kept in time order; a window with at most k targets keeps
them all. Each view is equally likely, so the auxiliary loss keeps its
all-view expectation (up to the pooled-pixel normalization of its means) at a
fraction of the render cost. V15.2 uses 2 of its 3-5 views; absent or null
renders every view (V15). A block with `auxiliary_render: none` rejects it. Each token's
rendered rows count from `gaussian_slot_index` 0 within its expert, so the
velocity prior still counts every token once. `expert_gumbel` is rejected for
every variant but V15.2, and V15.2 rejects every other count mode.

**Ordinal nested count (`count_mode: nested_ordinal`).** V16 and V16.1. The
token anchor, V15's G1..G5 decoder and the nested choices reduce {G1}, keep
{G1..G3} and add {G1..G5} are the nested count's; the block lives under
`p2g.grid_query.ordinal_count`, with the nested keys plus `router_scale`:

```yaml
p2g:
  grid_query:
    count_mode: nested_ordinal
    ordinal_count:
      option_counts: [1, 3, 5]      # must be exactly this
      seed_mode: token              # must be exactly this
      tau: 1.0                      # straight-through sigmoid temperature
      decoder_hidden: 576
      router_hidden: [256, 128]
      router_scale: 3.0             # fixed cosine scale of both gates, positive
      auxiliary_render: dense_add   # dense_add | random_prefix | none;
                                    # V16: dense_add, V16.1: random_prefix
      auxiliary_views: 2            # optional; null/absent = every target view
      logging:                      # optional, as for nested_count
        enable: true
        interval: 20
        range_edges_m: [0, 10, 20, 30, 40, 60, 80, 110]
```

The router (own LayerNorm, `router_hidden` SiLU layers) ends in two gates
without bias, `u = router_scale * cos(w, h)`, so `|u| <= router_scale` however
the weights grow: the keep gate opens G2/G3 and the add gate opens G4/G5 given
G2/G3. With `q = sigmoid(u)` the options have probabilities
`p(reduce) = 1 - q_keep`, `p(keep) = q_keep (1 - q_add)` and
`p(add) = q_keep q_add`; at scale 3 each gate stays in [0.047, 0.953]. Training
samples each gate as a hard Bernoulli(q) through logistic noise (the two-class
Gumbel-max) with the straight-through derivative of `sigmoid((u + noise) /
tau)`; there is no three-way softmax. Evaluation takes the most probable
option. The routing statistics read the option log-probabilities, so V16's
`logit_margin` is the log-probability margin of the top two options.

What changes most is how the choice is rendered in training. The routed set is
every token's G1..G5, each carrying a `blend_mask` -- G1 = 1, G2/G3 = M_keep,
G4/G5 = M_keep * M_add, forward values exactly 0 or 1 -- which the rasterizer
multiplies onto the blending weight only (`mask * alpha`); culling, the Python
`opacity > 1/255` prefilter and the kernels' 1/255 alpha cut use the unmasked
alpha (MaskGaussian's masked rasterization). A Gaussian at mask 0 changes no
output, median depth included, and its own attributes get no gradient, but its
mask does: `dL/dmask` is the first-order change of the loss from inserting it
(at mask 1, from removing it). So every token's gates learn from both sides,
whichever option it sampled; a reduce token sends its keep gate the slope of
inserting G2/G3 (with G4/G5 when the add gate was sampled open) and nothing to
the add gate. The second render is unmasked and router-free, at
`auxiliary_views` target views, so the Gaussians a token did not render keep
training: `dense_add` (V16) renders every token's G1..G5; `random_prefix`
(V16.1) renders every token's G1..Gk with k drawn uniformly from {1, 3, 5} per
token and step, independently of the gates, so G1 alone, G1..G3 and G1..G5
each get a third of the auxiliary renders whatever the router prefers. The
optimized loss is the mean of the two renders' losses (`aux_*` /
`routed_total` are logged as in V15). Evaluation renders each token's chosen
prefix without masks, which for 0/1 masks is exactly the training render of the
same choice. There is no Gaussian-count cost. `nested_ordinal` is rejected for
every variant but V16 and V16.1, both reject every other count mode, and each
pins its own `auxiliary_render`.

**V15.3's fixed five.** V15.3 has no router. Its count is the ordinary
fixed-count contract pinned to `count_mode: legacy`, `K_max: 5`,
`seed_mode: token` (any other value is rejected): the seed builder gives every
token five slots on its anchor, and the head decodes them with V15's two-layer
decoder and renders all five, in training and at evaluation alike, with no
gate, no auxiliary render and no routing statistics. Without a count block,
the decoder's hidden width is `dynamic_2dgs.gaussian_head.decoder_hidden`
(576), which V15.3 requires and every other variant rejects.

**Sparse-convolution precision (`train.spconv_allow_tf32`).** main.py sets
`torch.set_float32_matmul_precision("high")`, so every torch matmul -- Utonia's
stem and attention Linears included -- trains on TF32 Tensor Cores. Utonia's
sparse convolutions (each block's CPE `SubMConv3d` and its LoRA) run through
spconv, which ignores that switch. `train.spconv_allow_tf32: true` (the base
config's value since 2026-10-02) sets `spconv.constants.SPCONV_ALLOW_TF32` so
they may use TF32 kernels as well; an absent key means false, so checkpoints
saved before the key existed resume with FP32 sparse convolutions unless the
resume passes `train.spconv_allow_tf32=true`. `src/runtime_precision.py` applies
it at the start of `main()` and in `tests/run_dynamic_smoke.py`: spconv fixes
the kernel it picks for each shape at the first sparse convolution, so setting
it later in a process has no effect. `test_lidar4d_nvs.py` does not apply it and
evaluates in full FP32 (it also leaves torch matmuls at FP32). Measured on a
V15.2 training-shaped batch, it cut the step from 2.39 s to 2.09 s; the change
in Utonia features and gradients was the size of the change from Utonia's
random per-cell input point choice (`GridSample(mode="train")`).

Every count contract is structural: `p2g` is a checkpoint-protected root (and
V15.3's `dynamic_2dgs.gaussian_head.decoder_hidden` sits under the protected
`dynamic_2dgs` root), so a resume or evaluation cannot change how many
Gaussians a checkpoint's tokens produce.

## Random-duration windows (`data.dataset_name=nuscenes_dynamic_lidar4d`)

`src/dataloader/nuscenes/dynamic.py` is the barrier family's
alternative to the legacy bbox-compatible `nuscenes/legacy.py`. V11.6, V12.2, V14, V14.1
and the V15 family are the only overlays that ship with it; the other barrier-family
variants (V11-V11.5, V12, V12.1) train on `nuscenes/legacy.py` with the scene holdouts
and can opt into this loader from the CLI, as V11-V11.4 briefly did. The loader retains the old flat index's keyframe start sweep and 0.5-second sliding
stride, but samples the actual second endpoint uniformly from the available
0.1-second-grid sweeps whose measured timestamp is between 0.2 and 1.2 seconds
after that start. It is the `data/nuscenes_dynamic.yaml` profile; its holdouts
live under `dynamic_window`, so the legacy `data.train_excluded_scenes` is
cleared. It keeps the legacy profile's `pair_mode: keyframe`,
`sample_gap_us: 100000` and `gt_middle_count: 3`. Its distinctive keys:

```yaml
data:
  dataset_name: nuscenes_dynamic_lidar4d
  mode: nvs
  train_excluded_scenes: null
  dynamic_window:
    endpoint_min_sec: 0.2
    endpoint_max_sec: 1.2
    eval_seed: 42
    train_excluded_scenes: [scene-0757, scene-1094]
```

A start near the end of a scene may not have a 1.2-second sweep. Its draw is
uniform over the eligible endpoints that really exist; the loader does not
clamp several draws onto the final sweep and does not drop a start that still
has valid candidates. Training resamples the endpoint on every access using the
worker-seeded Python RNG. Validation/test use a stable hash of the start anchor
and `eval_seed`, so the pseudo-random endpoint and target views repeat exactly.

The loader samples up to `gt_middle_count=3` distinct interior views. A normal
0.2-second pair has one interior 0.1-second view, a 0.3-second pair has two, and
any pair with at least three available interior sweeps samples exactly three.
The chronological render-target list always includes the start and end sweeps,
so both model inputs retain the same reconstruction loss as the novel views.

Short pairs make the number of cameras ragged across a batch. The renderer pads
only its returned image tensors and emits `view_mask`; `Loss` excludes padded
rows from depth, median-depth, intensity, raydrop, and image metrics. It never
duplicates a target view to make shapes match. Chamfer already consumes ragged
per-sample point lists and needs no padding.

The dynamic loader returns endpoint XYZI in ego and LiDAR coordinates, frame
layout, local-to-reference poses/physical times, the minimal XYZI Utonia input,
and GT `Camera` objects. With `data.coordinate_frame: ego_vehicle`, each sweep
is transformed from its calibrated LiDAR frame into its own ego frame before
Utonia; the backend maps those local tokens through `E_i -> E_0`, and each
target camera uses `E_0 -> LiDAR_j` when rendering. The exact per-sweep nuScenes
calibration rotation and translation are used because LIDAR_TOP axes do not
align with ego axes. `data.coordinate_frame: lidar_sensor` preserves the old
contract for checkpoints that do not embed this setting. The loader does not
load boxes, assign point instance IDs, create raw reference-frame point copies,
or collate redundant GT point tensors. Historical checkpoints keep their
`data.dataset_name` and use the historical LiDAR coordinate contract; the
loader and dynamic-window block remain checkpoint-protected sampling semantics.

Two of the five LiDAR4D Table-S6 sequences are located in the official nuScenes
train split: `seq_1600_1650` belongs to `scene-0757`, and `seq_3180_3230`
belongs to `scene-1094`. The dynamic loader removes both complete scenes from
its train scene set before constructing start anchors. Holding out only each
five-second interval would still leak adjacent random-duration windows into the
test segment. Validation, compact validation, and the three official val-split
LiDAR4D sequences are not filtered by `train_excluded_scenes`.

## Compact validation (`data.compact_val`)

Validation after every epoch runs the whole `val` split by default: 5,719
windows in the shipped `pair_mode=keyframe` contract. Consecutive flat-index
entries step one keyframe at a time, so they overlap heavily, and GT-bbox
labelling shows the split is 85% dynamic / 6% static — the aggregate metric is
therefore both slow and dominated by geometry the velocity head cannot change.

`data.compact_val` replaces the val dataloader's index with a fixed subset:

```yaml
data:
  compact_val:
    enable: true
    path: config/compact_val_windows.json
```

The shipped manifest holds 200 windows weighted toward motion: **180 dynamic + 20
static**. The static 20 are a no-motion control, not a representative sample —
they exist to catch a regression in plain geometry, while the metric that moves
when the velocity head changes comes from the dynamic 180.
`tools/build_compact_val_windows.py` builds it from **nuScenes GT annotations**
(never the tracking JSON): for each window it measures every annotated
instance's global-frame motion between the keyframes the window spans, keeping
only instances with `>= 10` LiDAR returns within 80 m of the ego. A window is
`dynamic` at a peak instance speed `>= 1.0 m/s`, `static` below `0.2 m/s`
(including windows with no qualifying box), and excluded in between. The 180
dynamic windows are stratified over four difficulty quartiles — scored by peak
displacement, moving-point count, mover count, and proximity of the nearest
mover — so the pool spans easy single-distant-car windows through dense
close-range traffic rather than only the easy tail. The 20 static windows are
stratified by box count so they are not all empty streets. Both pools cap
per-scene picks and forbid overlapping windows within a scene; the dynamic 180
lands on 119 of the 148 scenes that contain any motion at all.

Note the population this is drawn from: GT-bbox labelling puts the val split at
85% dynamic, 6% static and 9% borderline, and all 348 static windows live in
just 40 scenes. Nothing-moving-within-80m is the rare case on nuScenes, which is
why the static side is a small control rather than half the subset.

A window is addressed by its two endpoint LIDAR_TOP `sample_data` tokens, which
is a function of the sampling contract. The manifest records that contract
(`version`, `split`, `window_us`, `sample_gap_us`, `pair_mode`,
`pair_kf_stride`) and loading re-checks it, so a manifest cannot be silently
applied to windows it does not address; change any of those keys and rebuild:

```bash
python tools/build_compact_val_windows.py --n-static 20 --n-dynamic 180
```

For `nuscenes_dynamic_lidar4d`, that stored pair is a **nominal address only**: it
selects the same legacy keyframe start anchor, while the actual endpoint comes
from the stable validation draw described above. This lets the existing
180-dynamic/20-static manifest retain the same start population even though
the physical interval varies under that loader.

`enable: false` restores full-split validation. `train`/`test` dataloaders are
untouched.

Every overlay except V11.6, V12.2, V14, V14.1 and the V15 family (V15,
V15.1, V15.2, V15.3) uses the shared bbox-compatible loader; those ship the
minimal dynamic loader above
(`data.mode=nvs`), which
the other barrier-family overlays can opt into. Both loader contracts expose
relative `timestamps_sec` plus `window_duration_sec`; every variant uses those
physical values for time conditioning and transport. Normalized endpoint
timestamps remain only as a small historical backend-API field.

Only keys consumed by the implementation remain in each dynamic overlay.
Attention geometry uses the shared Utonia-compatible `Rotary3D`; no variant
concatenates a second explicit position encoding into either prediction head.

Add new variants to the loader allow-list instead of accepting arbitrary config
paths.  This keeps a misspelled or path-traversing selector from silently loading
an unintended architecture.

## KITTI-360 training

Use `python main.py data_config=kitti360 device='[6]'` from
the repository root. The registered `kitti360_dynamic` loader reuses the
inference loader's calibrated GT/camera path and the nuScenes dynamic XYZI
collation. Model/loss/optimizer settings come from the selected variant.

There are two GT modes, and they differ only in the GT panorama:

- `data_config=kitti360`: calibrated 64×2178 rows, one per laser. Runs are
  named `v15.3_kitti`.
- `data_config=kitti360_lidar4d`: the LiDAR4D/GS-LiDAR uniform 66×1030 grid
  from the sensor origin. It is 1.43× faster per V15.3 step, and its GT keeps
  ~42% of in-range returns. Runs are named `v15.3_kitti_lidar4d`.

A CLI `exp_name` without "kitti" gains the mode suffix.

The training pool and complete LiDAR4D clip exclusions are in
[`kitti360_train_split.md`](kitti360_train_split.md), with rebuild settings in
`kitti360_train_split.yaml` and allowed frame segments in
`kitti360_train_split.json`. The raw NVS split projects official 3D validation
intervals onto sweeps, leaving 61,809 train / 14,344 val sweeps after benchmark
exclusions. Random endpoint durations are 0.2–1.0s with up to three interior GT
views. The selected start stride is a 0.5s time grid, snapped to saved sweeps.
After pose eligibility, there are 12,770 train / 2,996 val windows. Per-epoch
validation uses a 200-anchor compact set
(`kitti360_compact_val_windows.json`: 120 dynamic / 60 static / 20 stopped-ego).
Each anchor keeps its random-duration draw, and the per-group means are logged
as `val_<group>/<key>`. The split manifest hash and sensor/time settings are
protected on resume. See
[`kitti360_train.md`](kitti360_train.md) for configuration, GT maps and checks.

## KITTI-360 inference

`config/eval/kitti360_inference.yaml` is a data/rendering overlay for
`test_lidar4d_nvs.py`, selected with `eval_config=...`. Model and optimizer
structure still come from the checkpoint. This explicit inference path permits
another dataset's panorama geometry while retaining all training/resume guards
and the checkpoint's `data.coordinate_frame`. It rejects model/training roots
in the overlay. Ordinary evaluation without `eval_config` retains the existing
checkpoint-bound behavior.

```bash
python test_lidar4d_nvs.py eval_config=config/eval/kitti360_inference.yaml \
    test.ckpt_path=/absolute/path/to/checkpoint.ckpt device='[1]'
```

With `out_dir: null`, results default to the checkpoint's own directory:
`logs/<model>/lidar4d_kitti360_eval/` or
`logs/<model>/lidar4d_nuscenes_eval/`. An explicit CLI `out_dir` still wins.
Use distinct child paths when retaining multiple input-offset evaluations.

The dataset dispatcher is `src/dataloader/lidar4d_test.py`; KITTI loading is
`src/dataloader/kitti360/lidar4d_test.py`. The loader uses the ten official
LiDAR4D/GS-LiDAR clips in drive 0000 and their 40 held-out target frame IDs.
It supplies two non-held-out input sweeps nearest the configured offsets,
with actual scan timestamps for conditioning and motion transport. The
default GT (`panorama_projection: hdl64e_kitti360_calibrated`, 64×2178) has
one row per HDL-64E laser at the unit's fitted angle. Its upper and lower
emitter blocks render from their own origins. Returns at or beyond 80m are
dropped. `kitti360_inference_lidar4d_benchmark.yaml`
(`panorama_projection: uniform`) retains the official uniform 66×1030 grid
seen from the sensor origin. No nuScenes ring table is used for KITTI cameras.

KITTI XYZI intensity is already normalized. IMU forward/right/down axes are
converted to forward/left/up for ego-frame checkpoints. Both overlays
interpolate missing poses (the benchmark overlay since 2026-10-12);
`pose_missing_policy=previous` reproduces the official LiDAR4D/GS-LiDAR
preprocessing. `error` is the remaining
option. Dynamic GT boxes are
exact-frame visualization metadata; no GT boxes enter the model input.

KITTI retains its native GPS/IMU origin. Only the physical FRD-to-FLU axis
conversion is applied; local inputs, relative poses and cameras all use that
native ego frame. The temporary nuScenes reference-origin translation and its
reference calibration file were removed on 2026-10-06.

Details, measured sweep rates, annotation coverage, and clip splits:
[`kitti360_inference.md`](kitti360_inference.md).
Sensor specifications, panorama construction, and configuration changes for a
different training dataset: [`sensor_data_contract.md`](sensor_data_contract.md).

## H3D zero-shot inference

`config/eval/h3d_inference_lidar4d.yaml` evaluates a KITTI-360 checkpoint on
Honda H3D (another HDL-64E) without H3D training: every val/test scenario of
at least 5 s gives one target per whole second, rendered from the sweeps at
-0.1/+0.1 s on a uniform 66×1030 grid whose vfov ([-24.5, 2.4]°) follows the
H3D unit's beams. H3D has no LiDAR-to-vehicle calibration, so the overlay
supplies `data.ego_from_lidar` (KITTI-360's, raised to H3D's mounting height). Loader: `src/dataloader/h3d/`. Release
layout, sensor comparison, pose and timing conventions:
[`h3d_inference.md`](h3d_inference.md).

## Raydrop refinement (`refine_config=raydrop_unet`)

A second training stage on top of a finished model, as LiDAR4D, GS-LiDAR,
SN-LiDAR and LiDAR-RT do after their scene optimization. All four use the same
6.54M-parameter U-Net (GS-LiDAR's `scene/unet.py` is byte-identical to
LiDAR4D's `model/unet.py`); `src/models/module/m4_raydrop_refine.py` carries it
unchanged.

```bash
python main.py refine_config=raydrop_unet \
    raydrop_refine.base_ckpt=/abs/logs/v15.3/epoch=29.ckpt \
    exp_name=v15.3_raydrop_refine device='[4,5]'
```

- The run's config is the base checkpoint's embedded config with
  `refine/raydrop_unet.yaml` and the CLI on top. Those may set only
  `raydrop_refine`, `train`, `test`, `logger`, `metrics`, `exp_name`,
  `project_name`, `device`, `seed` and `mode`; model, data and rendering stay
  the base run's. `train.ckpt_path` and `data.*` keys are rejected.
- `data_config=<profile>` trains the U-Net on another dataset under the same
  frozen model, e.g. a nuScenes base on KITTI-360's 66×1030 grid:

  ```bash
  python main.py refine_config=raydrop_unet data_config=kitti360_lidar4d \
      raydrop_refine.base_ckpt=/abs/logs/v15.4/epoch=29.ckpt \
      exp_name=v15.4_nus_base_raydrop_refine_kitti_lidar4d device='[6]'
  ```

  The profile replaces the base run's whole data block, as in a fresh run.
  Its `coordinate_frame` must be the base run's. The loader records the
  profile path as `raydrop_refine.data_profile`. A KITTI-360 profile binds
  the split manifest hash and adds the GT-mode suffix to `exp_name`. The
  frozen model then renders the new dataset zero-shot, and `base/` logs those
  zero-shot values.
- The loader records the base checkpoint's `base_ckpt_sha256`, `base_epoch`,
  `base_global_step` and `base_exp_name`. Before training, `main.py` loads the
  base weights and checks the hash again.
- A refinement run needs its own `exp_name`. Its `logger.dir` is
  `logs/<exp_name>` and it opens a new W&B run. The epoch count, optimizer and
  schedule start fresh.
- The frozen model renders every batch in eval mode under `no_grad`. Only
  `raydrop_refiner.unet` trains:
  - **Input:** every target view (input endpoints and novel sweeps) as
    `[raydrop, intensity_sh, depth_median / depth_scale_m]`.
  - **Target:** BCE against the GT no-return mask (1 = no return).
  - **Augmentation:** LiDAR4D's random zero boxes, in training only.
  - **Optimizer:** AdamW at `train.lr` 1e-3 with weight decay 0, which is
    LiDAR4D's Adam. Warmup and cosine use the usual helper.
- `train/` and `val/` log the refined `loss_raydrop`, `raydrop_acc`,
  `raydrop_f1` and `raydrop_rmse`. Validation adds the official
  `depth_/intensity_*_raydrop` metrics and `loss_chamfer`. The frozen model's
  values for the same windows go under `base/`.
- Resume a refinement run with `train.ckpt_path=<its checkpoint>`. CLI changes
  to `raydrop_refine` are rejected there, and a model checkpoint cannot turn
  refinement on from the CLI.
- `test_lidar4d_nvs.py` builds the refiner from a refinement checkpoint and
  replaces only `renders["raydrop"]`, so the depth/intensity masks, raydrop
  metrics and points all use the refined mask. `test.raydrop_refine=false`
  evaluates the same checkpoint's frozen base model.

## Sensor-head adaptation (`adapt_config=sensor_head`)

V18's LiDAR decoder (the sensor embedding and the MLP that decodes the
alpha-blended Gaussian feature, the ray and the median depth into intensity
and raydrop) retrained for a new sensor behind a frozen scene model, usually
on another dataset:

```bash
python main.py adapt_config=sensor_head data_config=kitti360_lidar4d \
    sensor_adapt.base_ckpt=/abs/logs/v18_wint0.5/epoch=28.ckpt \
    exp_name=v18_wint0.5_ep28_sensorhead_wint1_wrd1 device='[4,5]'
```

- Composition is the refinement stage's (`_compose_frozen_model_stage` in
  `src/experiment_config/loader.py`): the base checkpoint's embedded config,
  then `adapt/sensor_head.yaml`, then the CLI. Those may set `sensor_adapt`,
  `loss`, `train`, `test`, `logger`, `metrics`, `exp_name`, `project_name`,
  `device`, `seed` and `mode`. `data_config=<profile>` swaps the whole data
  block (same `coordinate_frame`) and is recorded as
  `sensor_adapt.data_profile`; a KITTI-360 profile adds the GT-mode suffix to
  `exp_name`. The base's hash, epoch, step and run name go under
  `sensor_adapt.base_*`, and `main.py` re-checks the hash before loading every
  weight from it.
- Only a variant with `g2p.lidar_decoder` (V18) can be adapted, and a run
  either adapts the sensor head or refines raydrop. A refinement or
  adaptation checkpoint is not accepted as the base of either stage.
- `SensorAdaptWrapper` (`src/sensor_adapt_wrapper.py`) runs P2G and G2G under
  `no_grad` and keeps P2G, G2G and G2P in eval mode, as at test time. Only
  `g2p_model.lidar_decoder` (15,776 parameters) trains, in one
  `lidar_decoder` AdamW group. `sensor_adapt.decoder_init` sets where it
  starts: `base` (the overlay's default) loads the base's sensor embedding and
  MLP with every other weight; `random` loads every other weight and leaves
  those two at the fresh V18 initialization, seeded (`seed`) as in a
  from-scratch run. Configs composed before the key existed loaded the base's.
- **Objective:** `loss.w_intensity` x intensity L1 on GT returns +
  `loss.w_raydrop` x raydrop BCE on every pixel of every target view (1 and 1
  in the overlay). `w_depth`, `w_depth_median` and `w_chamfer` must be 0: the
  frozen geometry gives the decoder no gradient. Depth and (in val) Chamfer
  are still logged for the frozen geometry; no velocity prior is computed.
- **Optimizer:** V18's decoder group, `train.lr` 5e-4 and weight decay 1e-4,
  500 warmup steps and cosine decay; 20 epochs at bs 2 x accumulation 4.
- The checkpoint keeps the model's state_dict keys. `test_lidar4d_nvs.py`
  renders with the adapted decoder and records the frozen base as
  `sensor_adapt_base` in `metrics.json`. Resume with
  `train.ckpt_path=<its checkpoint>`; CLI changes to `sensor_adapt` are
  rejected there.
