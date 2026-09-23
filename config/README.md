# Experiment configuration

The Python implementation is split by responsibility. `src/config_loader.py`
keeps composition, checkpoint restore, and the historical import API.
`src/experiment_config/definitions.py` owns variant constants and schema key
sets, `helpers.py` owns shared OmegaConf operations, and `validation.py` owns
static config validation. Dependencies flow from definitions through helpers to
validation; none of those modules imports `config_loader`.

Fresh runs are composed in this order:

1. `nuscene_train.yaml` shared base
2. one allow-listed file under `variants/`
3. OmegaConf CLI overrides

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
python main.py model.variant=dynamic_2dgs_attention_velocity_v13
python main.py model.variant=dynamic_2dgs_attention_velocity_v13_1
python main.py model.variant=dynamic_2dgs_attention_velocity_v13_2
python main.py model.variant=dynamic_2dgs_attention_velocity_v13_3
python main.py model.variant=dynamic_2dgs_attention_velocity_v13_4
```

A fresh run defaults to `dynamic_2dgs_attention_velocity_v11_4`. Every listed
variant remains available for exact checkpoint reconstruction and controlled
A/B runs. The base config's `device: [2, 3]` is the default GPU pair; every
overlay inherits it and it stays an ordinary CLI-overridable runtime setting.

Every variant feeds the Utonia stage-3 XYZI token (432D, 0.4 m cells) with LoRA
adapters on the frozen base (`p2g.utonia_lora.enable=true` is required) straight
into temporal attention; there is no separate intensity stream, fusion MLP or
token refiner. V11.1, V11.3, V11.5, V11.6 and V13.1 emit a fixed number of
Gaussians per occupied endpoint token -- one observed-medoid seed by default,
two range quantiles in V11.3/V11.5/V13.1 and three in V11.6; V11, V11.2, V11.4,
V12, V12.1, V13, V13.2, V13.3 and V13.4 let a router select one to three
range-quantile-seeded Gaussians instead. `p2g.grid_query` chooses between the
two and sets the fixed count -- see "Gaussians per token" below.

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
keys, and both default to off in `GaussianAttributeHead`. V11.1 checkpoints
were saved from the plain affine form and restore unchanged.

V11.3 is deliberately absent from `ROUTER_CAPABLE_DYNAMIC_VARIANTS` even though
its backend class owns the router: `count_mode=learned_gumbel` is rejected
rather than quietly turning V11.3 back into V11. `p2g.grid_query.K_max` and
`exp` remain overridable to select another fixed count or seed rule.

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
contract: the inherited `dataset_name: nuscenes` (legacy `nuscene.py`),
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
random-duration `nuscenes_dynamic_lidar4d` loader instead of `nuscene.py`. The
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

V12 uses the legacy `src/dataloader/nuscene.py` contract, as every V11-V13
overlay except V11.6 does. Its overlay inherits `data.dataset_name: nuscenes`
and the fixed keyframe/window settings from `nuscene_train.yaml`, while setting
`data.mode: bbox` and `data.bbox_json_path: null`. V11, V11.1-V11.5, V12,
V12.1, V13, V13.1, V13.2, V13.3 and V13.4 additionally set
`data.train_excluded_scenes: [scene-0757, scene-1094]`; `nuscene.py` removes
those complete scenes only when constructing the train split. V11.6 holds out
the same two scenes under `data.dynamic_window`, where its loader reads them. Validation and
test scenes remain unchanged. The key is checkpoint-protected sampling state,
so a resume cannot silently put either held-out scene back.

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
initializer-conditioned offset head, losses, and `nuscene.py` data contract.
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

**V13** keeps V12.1's fixed-window `nuscene.py` data contract, complete-scene
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
emits exactly two Gaussians. `grid_query.K_max=2, exp=null` chooses the token's
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
V13.2's `nuscene.py` data contract and train-only `scene-0757`/`scene-1094`
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
`head_offset_bound=0.4`, and the `nuscene.py` scene-holdout contract are
identical. It has no distance bias, barrier, or separate temperature.

`barrier_weight` stays at 4 **in V11 itself**. Re-scoring V11's trained head-0
features at other
weights shows a=0 to a=4 is decisive (mass beyond the radius 11.2% -> 0.85%,
`|dp|` 8.5 m -> 2.26 m) and a=4 to a=64 changes nothing past the fourth decimal:
the residual mass sits just outside the radius, where the hinge is near zero by
construction and no weight reaches it.

The fixed time reference only conditions network features; it is not the
sample-dependent endpoint duration and is not used to reinterpret renderer
units.

`src/config_loader.py` owns composition for both training and evaluation. New
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
`exp` says where those seeds start:

| `exp` | seed placement |
| --- | --- |
| `1` | the token's observed medoid -- the raw point nearest its own-frame Cartesian mean |
| `2` | the token's Utonia cell coordinate |
| `null` | `K_max` deterministic range quantiles over the token's own raw points |

Seeds that coincide are not redundant: the Gaussian head owns one parameter
block per slot, including the bounded position offset, so slots starting at the
same coordinate learn independent geometry. V11.3, V11.5, V11.6 and V13.1
additionally feed every slot's seed delta into that head, so their slots also
differ by input and not only by parameters; every other fixed-count variant
reads the token feature alone. Under `exp: null` a token holding fewer raw points than
slots repeats observed points, the same way the learned-count candidate bank
does.

Omitting the whole `p2g.grid_query` block means `K_max: 1, exp: 1`: one
Gaussian per token at its medoid, which is what V11.1 ships. V11.3, V11.5 and
V13.1 ship the block explicitly at `K_max: 2, exp: null`, and V11.6 at
`K_max: 3, exp: null`. Every fixed-count variant supports larger counts.

Because the count is fixed and identical for every token, a mean over Gaussians
is exactly a mean over tokens. `velocity_l2` and the motion diagnostics stay
unweighted no matter what `K_max` is set to.

```bash
# Two Gaussians per token, both starting at the token's observed medoid.
python main.py model.variant=dynamic_2dgs_attention_velocity_v11_1 \
  p2g.grid_query.K_max=2 p2g.grid_query.exp=1

# Three Gaussians per token, spread over that token's raw points by range.
python main.py model.variant=dynamic_2dgs_attention_velocity_v11_1 \
  p2g.grid_query.K_max=3 p2g.grid_query.exp=null
```

Peak memory and step time scale with the Gaussian count, not the token count.
Measured on one A100 at `batch_size=2` with a single temporal layer, V11.1 emits
29,343 Gaussians at `K_max=1` and 58,686 at `K_max=2` from the same tokens;
V11.3 emits that same 58,686 from its shipped `K_max: 2`.

**Learned count (`count_mode: learned_gumbel`).** The refined token routes
through the grid `GridSlotHead` and a hard Gumbel-Softmax over K={1,2,3} picks
one K-specific joint head per token. Only V11, V11.2, V11.4, V12, V12.1,
V13, V13.2, V13.3 and V13.4 may use it; asking any other variant for it is
rejected at config time rather than silently ignored. V11.3, V11.5, V11.6 and
V13.1 are rejected too because their identities are the fixed count. The seed
geometry it consumes is the K-specific candidate bank, not the padded
fixed-count seed set, so the two contracts cannot be mixed.

Both counts are structural: `p2g` is a checkpoint-protected root, so a resume or
evaluation cannot change how many Gaussians a checkpoint's tokens produce.

## Random-duration windows (`data.dataset_name=nuscenes_dynamic_lidar4d`)

`src/dataloader/nuscene_dynamic_lidar4d.py` is the barrier family's
alternative to the legacy bbox-compatible `nuscene.py`. V11.6 is the only
overlay that ships with it; the other barrier-family variants (V11-V11.5, V12,
V12.1) train on `nuscene.py` with the scene holdouts and can opt into this
loader from the CLI, as V11-V11.4 briefly did. The loader retains the old flat index's keyframe start sweep and 0.5-second sliding
stride, but samples the actual second endpoint uniformly from the available
0.1-second-grid sweeps whose measured timestamp is between 0.2 and 1.2 seconds
after that start. Selecting it replaces the data block; its holdouts live under
`dynamic_window`, so the legacy `data.train_excluded_scenes` has to be cleared.
`pair_mode: keyframe`, `sample_gap_us: 100000` and `gt_middle_count: 3` are
inherited from `nuscene_train.yaml`:

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

The dynamic loader returns only sensor-frame endpoint XYZI, frame layout,
endpoint poses/physical times, the minimal XYZI Utonia input, and GT `Camera`
objects. It does not load boxes, assign point instance IDs, create raw
reference-frame point copies, or collate redundant GT point tensors. Historical
checkpoints keep the `data.dataset_name` embedded when they were saved, so an
old fixed-duration V11 checkpoint still reconstructs with `nuscene.py`; the
loader and dynamic-window block are checkpoint-protected sampling semantics.

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

Every overlay except V11.6 uses the shared bbox-compatible loader;
V11.6 ships the minimal dynamic loader above (`data.mode=nvs`), which the other
barrier-family overlays can opt into. Both loader contracts expose
relative `timestamps_sec` plus `window_duration_sec`; every variant uses those
physical values for time conditioning and transport. Normalized endpoint
timestamps remain only as a small historical backend-API field.

Only keys consumed by the implementation remain in each dynamic overlay.
Attention geometry uses the shared Utonia-compatible `Rotary3D`; no variant
concatenates a second explicit position encoding into either prediction head.

Add new variants to the loader allow-list instead of accepting arbitrary config
paths.  This keeps a misspelled or path-traversing selector from silently loading
an unintended architecture.
