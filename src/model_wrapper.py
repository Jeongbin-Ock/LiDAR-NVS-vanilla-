import torch
import json
import math
from pathlib import Path
from omegaconf import OmegaConf

from src.models.module import (
    Point2Gaus,
    DynamicGausTemp,
    DynamicGausRender,
)
from lightning.pytorch import LightningModule
from src.models.utils.loss import (
    Loss,
    distributed_token_mean,
    ramped_weight_at_step,
)
from src.models.utils.routing_logging import (
    distributed_sum_statistics,
    range_bin_labels,
    routing_option_counts,
    routing_sufficient_statistics,
)
from src.experiment_config import DYNAMIC_VARIANTS, ROUTED_COUNT_BLOCKS
from src.eval.group_metrics import GroupMetricAccumulator
from src.dataloader.kitti360.compact_val import COMPACT_VAL_GROUPS

WANDB_BASE_LOSS_KEYS = {
    "loss_depth",
    "loss_depth_median",
    "loss_intensity",
    "loss_raydrop",
    "loss_scale",
    "total",
    # Reconstruction quality. Keys absent from a given step's loss dict are
    # skipped by _log_losses, so these follow whatever schedule produced them:
    # chamfer is computed for val/test (and for train only when w_chamfer>0),
    # while the PSNR/SSIM pairs follow _metric_schedule.
    "loss_chamfer",
    "intensity_psnr_valid",
    "intensity_ssim_valid",
    "depth_psnr_valid",
    "depth_ssim_valid",
    "intensity_psnr_raydrop",
    "intensity_ssim_raydrop",
    "depth_psnr_raydrop",
    "depth_ssim_raydrop",
}

# Already reduced over every rank's tokens inside the loss, so their forward
# values are identical on all ranks and need no logging collective.
# V15 and V15.2 train on two renders: the routed set, whose terms keep the
# ordinary keys, and a router-free set (V15: every token's G1..G3; V15.2: every
# token's E5). ``aux_*`` are the second render's terms and ``routed_total`` the
# first render's total before the two are averaged.
AUXILIARY_RENDER_LOSS_KEYS = (
    "loss_depth",
    "loss_depth_median",
    "loss_intensity",
    "loss_raydrop",
    "loss_chamfer",
    "total",
)

GLOBAL_REDUCED_LOSS_KEYS = {
    "loss_velocity_l2",
    "loss_velocity_l2_init",
    "loss_velocity_l2_offset",
}


def _per_token_rows(item, field):
    """``item[field]`` reduced to one row per source token.

    Motion is predicted per token and copied onto each of that token's
    Gaussians. A fixed count gives every token the same number of identical
    copies, so a mean over Gaussians is already the token mean, in value and in
    gradient. The adaptive-K router gives a token K={1,2,3} copies (the V15
    family's routers 1, 3 or 5), and the same mean would charge a large-K token
    more than a K=1 token pays for the same velocity. Router items carry
    ``gaussian_slot_index``, and every token owns exactly one slot-0 Gaussian
    (V15's and V16's G1, the first Gaussian of V15.2's chosen expert), so
    keeping slot 0 counts each token once.
    """
    value = item[field]
    slot = item.get("gaussian_slot_index")
    if slot is None:
        return value
    return value[slot == 0]


class ModelWrapper(LightningModule):
    def __init__(
        self,
        cfg,
    ):
        super().__init__()
        self.cfg = cfg
        model_cfg = getattr(cfg, "model", None)
        self.model_variant = str(
            getattr(model_cfg, "variant", "") if model_cfg is not None else ""
        )
        if self.model_variant not in DYNAMIC_VARIANTS:
            raise ValueError(f"Unsupported model.variant={self.model_variant!r}")
        self.p2g_cfg = cfg.p2g
        self.g2g_cfg = cfg.g2g
        self.g2p_cfg = cfg.g2p
        loss_module = Loss(self.cfg.loss)

        # Set up the model.
        self.dynamic_cfg = cfg.dynamic_2dgs
        self.p2g_model = Point2Gaus(
            self.p2g_cfg,
            dynamic_cfg=self.dynamic_cfg,
            dynamic_variant=self.model_variant,
        )
        self.g2g_model = DynamicGausTemp(self.g2g_cfg)
        self.g2p_model = DynamicGausRender(self.g2p_cfg)
        self.loss = loss_module
        self._eval_pair_summaries: dict[str, list[dict]] = {}
        # Validation metrics per compact-val window group (val_<group>/<key>).
        self._val_group_metrics: GroupMetricAccumulator | None = None
        self._metric_interval = int(self._cfg_get("metrics.interval", 50))
        if self._metric_interval <= 0:
            raise ValueError("metrics.interval must be positive")
        # Gradient accumulation repeats one global_step across several batches;
        # this makes the sparse train metric fire once per optimizer step.
        self._last_train_metric_step = -1
        # The learned_gumbel router and the V15 family's routers (V15's nested
        # prefixes, V15.2's per-option experts) keep the same logging block
        # under their own count config.
        learned_count_block = next(
            (
                f"p2g.grid_query.{block}"
                for block in ROUTED_COUNT_BLOCKS.values()
                if self._cfg_get(f"p2g.grid_query.{block}") is not None
            ),
            "p2g.grid_query.learned_count",
        )
        self._routing_logging_enable = bool(self._cfg_get(
            f"{learned_count_block}.logging.enable", False
        ))
        self._routing_logging_interval = int(self._cfg_get(
            f"{learned_count_block}.logging.interval", 20
        ))
        self._routing_range_edges = tuple(float(value) for value in self._cfg_get(
            f"{learned_count_block}.logging.range_edges_m",
            (0, 10, 20, 30, 40, 60, 80, 110),
        ))
        if self._routing_logging_interval <= 0:
            raise ValueError(
                "learned_count.logging.interval must be positive"
            )
        # Validates lower-bound ordering once, before the first train batch.
        range_bin_labels(self._routing_range_edges)
        self._last_train_routing_step = -1
        self._wandb_loss_keys = set(WANDB_BASE_LOSS_KEYS)
        # The router-free second render may use a random subset of each
        # window's target views (V15.2: two); null renders every view.
        auxiliary_views = self._cfg_get(f"{learned_count_block}.auxiliary_views")
        self._auxiliary_views = (
            None if auxiliary_views is None else int(auxiliary_views)
        )
        # Every routed V15-family block names its router-free second render;
        # ``none`` (V15.1) and the other count blocks render once.
        if str(self._cfg_get(
            f"{learned_count_block}.auxiliary_render", "none"
        )).lower() != "none":
            self._wandb_loss_keys.update(
                f"aux_{key}" for key in AUXILIARY_RENDER_LOSS_KEYS
            )
            self._wandb_loss_keys.add("routed_total")
        regularization = getattr(self.dynamic_cfg, "regularization", None)
        velocity_l2 = getattr(regularization, "velocity_l2", None)
        self._velocity_l2_cfg = (
            velocity_l2
            if velocity_l2 is not None and bool(getattr(velocity_l2, "enabled", False))
            else None
        )
        if self._velocity_l2_cfg is not None:
            mode = str(getattr(self._velocity_l2_cfg, "mode", None))
            if mode != "split_group_l2":
                raise ValueError(
                    "velocity_l2.mode must be 'split_group_l2'; got "
                    f"{mode!r}"
                )
            for name in ("init_weight", "offset_weight"):
                value = getattr(self._velocity_l2_cfg, name, None)
                if value is None or float(value) < 0.0:
                    raise ValueError(
                        "velocity_l2.mode='split_group_l2' requires a "
                        f"non-negative velocity_l2.{name}"
                    )
            self._wandb_loss_keys.update({
                "loss_velocity_l2", "loss_velocity_l2_init",
                "loss_velocity_l2_offset",
            })

    def on_save_checkpoint(self, checkpoint) -> None:
        """Bind the resolved experiment config immutably to each checkpoint."""

        checkpoint["experiment_config_schema_version"] = 1
        checkpoint["experiment_config"] = OmegaConf.to_container(
            self.cfg,
            resolve=True,
        )

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, prefix="train")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, prefix="test")


    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, prefix="val")


    def _shared_step(self, batch, batch_idx, *, prefix: str):
        _input, gt = batch["input"], batch["gt"]
        compute_valid_metrics, compute_official_metrics = self._metric_schedule(
            prefix, batch_idx
        )
        p2g_out = self.p2g_model(_input)
        # Keep detached diagnostics outside the renderer/temporal model input.
        routing_stats = p2g_out.pop("routing_stats", None)
        # Only V15, V15.2 and V16 emit this second Gaussian set, in training only.
        auxiliary_gaussians = p2g_out.pop("auxiliary_batch_gaussians", None)
        out = self.g2g_model(
            p2g_out,
            _input["timestamps_sec"],
            _input["window_duration_sec"],
        )
        compute_train_points = float(self.cfg.loss.w_chamfer) > 0.0
        compute_points = prefix != "train" or compute_train_points
        all_renders = self.g2p_model(
            out,
            gt,
            compute_points=compute_points,
        )

        loss_dict = self.loss(
            all_renders,
            gaussians=out,
            metric_mode=prefix,
            compute_valid_metrics=compute_valid_metrics,
            compute_raydrop_metrics=compute_official_metrics,
        )
        if auxiliary_gaussians is not None:
            self._average_auxiliary_render_loss(
                loss_dict,
                auxiliary_gaussians,
                _input,
                gt,
                prefix=prefix,
                compute_points=compute_points,
            )
        self._add_dynamic_motion_regularization(loss_dict, out, prefix=prefix)
        self._log_losses(loss_dict, prefix=prefix, batch_size=self._batch_size(_input))
        self._log_routing_stats(routing_stats, prefix=prefix)
        self._log_utonia_adaptation(prefix=prefix)

        if prefix != "train":
            self._record_eval_summary(loss_dict, batch_idx=batch_idx, prefix=prefix)
        groups = batch.get("meta", {}).get("window_group")
        if prefix == "val" and groups and self._val_group_metrics is not None:
            self._val_group_metrics.add(loss_dict, groups, self._batch_size(_input))

        return loss_dict["total"]

    def _average_auxiliary_render_loss(
        self, losses, auxiliary_gaussians, _input, gt, *, prefix,
        compute_points,
    ):
        """Optimize the mean of the routed and the router-free render losses.

        A V15-family router decides which of a token's Gaussians are rendered
        (V15/V16: a prefix of G1..G5; V15.2: one of the experts E1/E3/E5), and
        that routed set stays the render every metric, validation and the test
        script see; V16's training renders it as every G1..G5 under blend
        masks. During training the backend also emits a router-free set (V15:
        every token's keep prefix G1..G3; V15.2: every token's E5; V16: every
        token's G1..G5), so the decoder is supervised on every token at every
        step. Both sets share one forward pass, velocity and target views; the
        optimized render loss is ``0.5 * (routed + router-free)``, and the
        velocity prior is added once afterwards, exactly as for a single
        render. The router-free set is rendered at every target view, or at
        ``auxiliary_views`` of them per window (see
        ``_auxiliary_render_targets``).
        """
        auxiliary = self.g2g_model(
            {"batch_gaussians": auxiliary_gaussians},
            _input["timestamps_sec"],
            _input["window_duration_sec"],
        )
        renders = self.g2p_model(
            auxiliary, self._auxiliary_render_targets(gt),
            compute_points=compute_points,
        )
        auxiliary_losses = self.loss(
            renders,
            gaussians=auxiliary,
            metric_mode=prefix,
            compute_valid_metrics=False,
            compute_raydrop_metrics=False,
        )
        for key in AUXILIARY_RENDER_LOSS_KEYS:
            if key in auxiliary_losses:
                losses[f"aux_{key}"] = auxiliary_losses[key].detach()
        losses["routed_total"] = losses["total"].detach()
        losses["total"] = 0.5 * (losses["total"] + auxiliary_losses["total"])

    def _auxiliary_render_targets(self, gt):
        """The target views the router-free set is rendered at.

        With ``auxiliary_views = k`` every window keeps k distinct target
        views drawn uniformly from all of its targets -- input endpoints and
        novel sweeps alike -- in time order; a window with at most k targets
        keeps them all, and ``None`` keeps every view. Because each view is
        equally likely, the auxiliary loss keeps its all-view expectation up to
        the pooled-pixel normalization of its means, while the render costs
        k views instead of all of them.
        """
        if self._auxiliary_views is None:
            return gt
        cameras = gt["cameras"] if isinstance(gt, dict) else gt
        sampled = []
        for window in cameras:
            window = list(window)
            if len(window) > self._auxiliary_views:
                keep = torch.randperm(len(window))[:self._auxiliary_views]
                window = [window[index] for index in sorted(keep.tolist())]
            sampled.append(window)
        return {**gt, "cameras": sampled} if isinstance(gt, dict) else sampled

    def _add_dynamic_motion_regularization(self, losses, gaussians, *, prefix):
        """Apply the configured motion prior and log runaway diagnostics."""
        self._add_velocity_l2_prior(losses, gaussians, prefix=prefix)
        self._log_velocity_statistics(gaussians, prefix=prefix)
        self._log_attention_temperature(prefix=prefix)

    def _log_attention_temperature(self, *, prefix: str) -> None:
        """Log the active motion-attention scale without building probabilities.

        QK-Norm variants report their learned gain bounds. V13 reports its fixed
        head-0 cosine temperature and bound. V13.2 uses raw unnormalized Q/K on
        every head and intentionally has no finite norm-derived bound. V13.3
        and V13.4 report V11.4-style RMS gain bounds; V13.4 merely leaves
        match head 0 out of the RoPE transform.
        """
        temporal = getattr(
            getattr(self.p2g_model, "dynamic_backend", None), "temporal", None
        )
        stats = temporal.attention_temperature_stats() if temporal is not None else None
        if not stats:
            return
        for name, value in stats.items():
            self.log(
                f"{prefix}/dynamic/{name}",
                float(value),
                on_step=False,
                on_epoch=True,
                sync_dist=False,
                batch_size=1,
            )

    def _log_utonia_adaptation(self, *, prefix: str) -> None:
        """Hybrid Utonia adaptation magnitudes from this step's forward.

        ``stem_{xyz,intensity}_rms`` are the frozen XYZ projection and the
        trained intensity column before the stem LayerNorm. ``stem_adapter_rms``
        is the nonlinear stem adapter's output: pre-LN in V14/V14.1, and in V15
        added after the GELU, next to ``stem_activation_rms`` (the GELU output
        it is added to). ``ffn_adapter_rms_ratio_enc{s}`` is the scaled adapter output
        RMS over the frozen FFN output RMS, averaged over stage ``s``'s blocks.
        All-linear LoRA variants log nothing here.
        """
        stats = self.p2g_model.utonia_adaptation_statistics()
        for name, value in stats.items():
            self.log(
                f"{prefix}/utonia/{name}",
                value,
                on_step=(prefix == "train"),
                on_epoch=True,
                sync_dist=False,
                batch_size=1,
            )

    @staticmethod
    def _active_motion_items(gaussians):
        return [
            item for item in gaussians
            if item is not None and item["velocity"].numel() > 0
        ]

    def _add_velocity_l2_prior(self, losses, gaussians, *, prefix):
        """Independent squared-norm priors on ``v_init`` and ``v_offset``.

        Penalizing ``||v_init + v_offset||^2`` pays the correction to point
        backwards: the cross term ``2<v_init, v_offset>`` is minimized by
        cancellation rather than by being right (V5's residual head became a
        pure shrink operator under exactly that objective, median
        cos(init, offset) = -0.997). Charging each magnitude on its own removes
        the cross term, so the offset is only worth what it explains, while
        ``init_weight`` still bounds the matcher directly (regularizing the
        offset alone let V6's background ``|v_init|`` grow to 5.8 m/s, past the
        real vehicles it was supposed to track).

        Both terms average over tokens rather than Gaussians (see
        ``_per_token_rows``), so a token's velocity costs the same whatever K
        the router gave it. ``loss_velocity_l2`` is the raw, unweighted sum of
        the two component means; only ``total`` sees the weights, because two
        different weights cannot factor out of the sum. The raw terms are
        logged for train, val and test alike, but only train adds them to the
        optimized total, so validation totals stay a pure reconstruction score.
        """
        cfg = self._velocity_l2_cfg
        if cfg is None:
            return
        active = self._active_motion_items(gaussians)
        raw = {}
        for name, field in (
            ("init", "velocity_init"), ("offset", "velocity_offset"),
        ):
            if active:
                missing = [
                    index for index, item in enumerate(active)
                    if field not in item
                ]
                if missing:
                    raise ValueError(
                        "velocity_l2.mode='split_group_l2' requires every "
                        f"active Gaussian item to provide {field}; missing at "
                        f"active item indices {missing}"
                    )
                value = torch.cat(
                    [_per_token_rows(item, field) for item in active], dim=0
                )
                local_sum = value.square().sum()
                local_count = value.shape[0]
            else:
                # Keep a local graph edge while still entering the same
                # all-reduce as ranks that received occupied tokens.
                local_sum = losses["total"] * 0.0
                local_count = 0
            raw[name], _global_count = distributed_token_mean(
                local_sum, local_count
            )
            losses[f"loss_velocity_l2_{name}"] = raw[name]

        losses["loss_velocity_l2"] = raw["init"] + raw["offset"]

        if prefix != "train":
            return
        # Each term keeps its own weight; the schedule is a shared 0->1 ramp.
        schedule = ramped_weight_at_step(
            1.0,
            int(self.global_step),
            int(getattr(cfg, "warmup_steps", 0)),
            int(getattr(cfg, "ramp_steps", 0)),
        )
        weighted = schedule * (
            float(getattr(cfg, "init_weight", 0.0)) * raw["init"]
            + float(getattr(cfg, "offset_weight", 0.0)) * raw["offset"]
        )
        losses["wc_velocity_l2"] = weighted.detach()
        losses["total"] = losses["total"] + weighted

    @torch.no_grad()
    def _log_velocity_statistics(self, gaussians, *, prefix: str) -> None:
        """Motion-magnitude alarms for a velocity field that runs away.

        Like the velocity priors, every statistic counts each token once (see
        ``_per_token_rows``), so the router's K does not weight it.
        """
        active = self._active_motion_items(gaussians)
        if not active:
            # Unreachable during training: the renderer rejects a None batch and
            # every occupied token emits a Gaussian, so all ranks reach the
            # sync_dist logging below together. This guard only covers callers
            # that hand in an empty list.
            return
        velocity = torch.cat(
            [_per_token_rows(item, "velocity") for item in active], dim=0
        )
        speed = velocity.norm(dim=-1)
        statistics = {
            "velocity_speed_mean": speed.mean(),
            # Under DDP this reduces to the mean of per-rank maxima rather than
            # a global max. It is still a usable runaway alarm.
            "velocity_speed_max": speed.max(),
            "velocity_abs_mean": velocity.abs().mean(),
            # nuScenes traffic tops out near 30 m/s, so a rising fraction here
            # is motion the scene cannot physically contain.
            "velocity_frac_over_30mps": (speed > 30.0).to(speed.dtype).mean(),
        }
        statistics.update(self._motion_statistics(active))
        for name, value in statistics.items():
            self.log(
                f"{prefix}/dynamic/{name}",
                value,
                on_step=(prefix == "train"),
                on_epoch=True,
                sync_dist=True,
                batch_size=int(speed.numel()),
            )

    @staticmethod
    def _motion_statistics(active):
        """Compact correspondence diagnostics shared by every variant."""
        stats = {}

        def _cat(name):
            if any(name not in item for item in active):
                return None
            return torch.cat(
                [_per_token_rows(item, name) for item in active], dim=0
            ).float()

        for field, label in (
            ("velocity_match", "velocity_match_norm_mean"),
            ("velocity_init", "velocity_init_norm_mean"),
            ("velocity_offset", "velocity_offset_norm_mean"),
        ):
            value = _cat(field)
            if value is not None:
                stats[label] = value.norm(dim=-1).mean()

        layer_weight = _cat("motion_layer_weights")
        if layer_weight is not None:
            if layer_weight.ndim != 2:
                raise ValueError(
                    "motion_layer_weights must have shape (tokens, layers)"
                )
            normalized = layer_weight / layer_weight.sum(
                dim=-1, keepdim=True
            ).clamp_min(1.0e-12)
            stats["motion_layer_weight_entropy_mean"] = -(
                normalized * normalized.clamp_min(1.0e-12).log()
            ).sum(dim=-1).mean()
            stats["motion_layer_weight_max_mean"] = normalized.max(
                dim=-1
            ).values.mean()
            for layer in range(normalized.shape[1]):
                stats[f"motion_layer_weight_l{layer}_mean"] = normalized[
                    :, layer
                ].mean()
        return stats

    def _metric_schedule(self, prefix: str, batch_idx: int) -> tuple[bool, bool]:
        """Return ``(valid, official_raydrop)`` metric switches for this batch.

        Train computes both diagnostic and official metrics once per configured
        optimizer-step interval. Validation/test computes official LiDAR4D /
        GS-LiDAR metrics for every frame, while the valid-only metrics stay
        sparse. Train keys off ``global_step`` and validation off ``batch_idx``,
        both of which advance identically on every rank, so conditional
        ``sync_dist=True`` logging cannot deadlock under DDP.
        """
        if prefix == "train":
            step = int(self.global_step)
            due = (
                step % self._metric_interval == 0
                and step != self._last_train_metric_step
            )
            if due:
                self._last_train_metric_step = step
            return due, due
        if prefix in {"val", "test", "eval"}:
            return int(batch_idx) % self._metric_interval == 0, True
        return True, True

    def on_validation_epoch_start(self) -> None:
        self._eval_pair_summaries["val"] = []
        self._val_group_metrics = GroupMetricAccumulator(
            COMPACT_VAL_GROUPS, self._wandb_loss_keys,
        )

    def on_validation_epoch_end(self) -> None:
        group_aggregate = self._log_val_group_metrics()
        self._write_eval_aggregate(prefix="val", group_aggregate=group_aggregate)

    def _log_val_group_metrics(self) -> dict:
        """Log ``val_<group>/<key>`` means over all ranks; one collective per epoch."""
        if self._val_group_metrics is None:
            return {}
        accumulator, self._val_group_metrics = self._val_group_metrics, None
        aggregate = accumulator.reduce(self.device)
        if accumulator.mixed_batches and self._is_rank_zero():
            print(f"[val] {accumulator.mixed_batches} batches mixed window groups and were left "
                  "out of val_<group>/ metrics; use test.batch_size=1", flush=True)
        for group, metrics in aggregate.items():
            for key, value in metrics.items():
                self.log(f"val_{group}/{key}", value, on_step=False, on_epoch=True,
                         sync_dist=False)  # already summed over ranks
        return aggregate

    def on_test_epoch_start(self) -> None:
        self._eval_pair_summaries["test"] = []

    def on_test_epoch_end(self) -> None:
        self._write_eval_aggregate(prefix="test")

    @staticmethod
    def _batch_size(_input: dict) -> int:
        pose = _input.get("pose")
        if isinstance(pose, list):
            return len(pose)
        if torch.is_tensor(pose) and pose.dim() > 0:
            return int(pose.shape[0])
        return 1

    def _log_losses(self, losses: dict, *, prefix: str, batch_size: int) -> None:
        for key, value in losses.items():
            if key not in self._wandb_loss_keys:
                continue
            if not torch.is_tensor(value):
                continue
            self.log(
                f"{prefix}/{key}",
                value,
                on_step=True,
                on_epoch=True,
                prog_bar=(key == "total" or key.startswith("loss_")),
                # Velocity-prior terms already contain an autograd-safe global
                # token reduction and therefore have identical forward values
                # on all ranks. Avoid redundant logging collectives for them.
                sync_dist=(key not in GLOBAL_REDUCED_LOSS_KEYS),
                batch_size=batch_size,
            )

    def _log_routing_stats(self, routing: dict | None, *, prefix: str) -> None:
        """Accumulate detached learned-count diagnostics for epoch-only logs."""
        if not self._routing_logging_enable or routing is None:
            return
        if prefix == "train":
            step = int(self.global_step)
            if (
                step % self._routing_logging_interval != 0
                or step == self._last_train_routing_step
            ):
                return
            self._last_train_routing_step = step

        statistics = routing_sufficient_statistics(
            routing, self._routing_range_edges,
        )
        statistics = distributed_sum_statistics(statistics)
        unit_count = statistics["token_count"]
        if not bool((unit_count > 0).item()):
            return

        base_batch_size = max(1, int(unit_count.item()))

        def log_value(name, value, *, batch_size=base_batch_size):
            self.log(
                f"{prefix}/routing/{name}",
                value,
                on_step=False,
                on_epoch=True,
                sync_dist=False,  # statistics were explicitly summed above.
                batch_size=max(1, int(batch_size)),
            )

        log_value("gaussians", statistics["sampled_k_sum"])
        log_value(
            "sampled/mean_k",
            statistics["sampled_k_sum"] / unit_count,
        )
        log_value(
            "argmax/mean_k",
            statistics["argmax_k_sum"] / unit_count,
        )
        log_value(
            "expected/mean_k",
            statistics["expected_k_sum"] / unit_count,
        )
        log_value(
            "logit_margin_mean",
            statistics["logit_margin_sum"] / unit_count,
        )
        # Options are labelled by the Gaussian count they render: k1..k3 for
        # the learned_gumbel router, k1/k3/k5 for the V15 family's
        # reduce/keep/add.
        option_counts = routing_option_counts(
            routing, int(statistics["selected_counts"].numel())
        ).tolist()
        for index, count in enumerate(option_counts):
            label = f"k{int(count)}"
            log_value(
                f"sampled/frac_{label}",
                statistics["selected_counts"][index] / unit_count,
            )
            log_value(
                f"argmax/frac_{label}",
                statistics["argmax_counts"][index] / unit_count,
            )
            log_value(
                f"probability/mean_{label}",
                statistics["probability_sums"][index] / unit_count,
            )

        range_labels = range_bin_labels(self._routing_range_edges)
        for index, label in enumerate(range_labels):
            count = statistics["range_token_counts"][index]
            log_value(f"range/{label}/token_frac", count / unit_count)
            if not bool((count > 0).item()):
                continue
            bin_batch_size = int(count.item())
            log_value(
                f"range/{label}/sampled_mean_k",
                statistics["range_sampled_k_sums"][index] / count,
                batch_size=bin_batch_size,
            )
            log_value(
                f"range/{label}/argmax_mean_k",
                statistics["range_argmax_k_sums"][index] / count,
                batch_size=bin_batch_size,
            )

    def _record_eval_summary(self, losses: dict, *, batch_idx: int, prefix: str) -> None:
        summary = {
            "epoch": int(getattr(self, "current_epoch", 0)),
            "global_step": int(getattr(self, "global_step", 0)),
            "batch_idx": int(batch_idx),
        }
        for key, value in losses.items():
            if torch.is_tensor(value):
                summary[key] = float(value.detach().mean().cpu())
        self._eval_pair_summaries.setdefault(prefix, []).append(summary)

    def _write_eval_aggregate(self, *, prefix: str, group_aggregate: dict | None = None) -> None:
        if not self._is_rank_zero():
            return
        summaries = self._eval_pair_summaries.get(prefix, [])
        if not summaries:
            return

        metric_keys = sorted(
            key
            for key in summaries[0].keys()
            if key not in {"epoch", "global_step", "batch_idx"}
        )
        aggregate = {}
        for key in metric_keys:
            values = [item[key] for item in summaries if key in item]
            if values:
                aggregate[key] = sum(values) / len(values)

        out_dir = self._eval_output_dir(prefix)
        out_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "epoch": int(getattr(self, "current_epoch", 0)),
            "global_step": int(getattr(self, "global_step", 0)),
            "split": prefix,
            "num_batches": len(summaries),
            "aggregate": aggregate,
            "batches": summaries,
        }
        if group_aggregate:
            # Means over every rank's windows; "aggregate" above is rank 0's.
            payload["group_aggregate"] = group_aggregate
        with open(out_dir / "aggregate_summary.json", "w") as f:
            json.dump(payload, f, indent=2)

    def _eval_output_dir(self, prefix: str) -> Path:
        logger_dir = self._cfg_get("logger.dir", "")
        base = Path(str(logger_dir)) if logger_dir else Path("outputs")
        return base / f"eval_{prefix}"

    def _cfg_get(self, dotted_key: str, default=None):
        cur = self.cfg
        for part in dotted_key.split("."):
            if cur is None:
                return default
            if isinstance(cur, dict):
                cur = cur.get(part, default)
            else:
                try:
                    cur = getattr(cur, part)
                except (AttributeError, KeyError):
                    return default
        return cur

    def _is_rank_zero(self) -> bool:
        return int(getattr(self, "global_rank", 0)) == 0

    def configure_optimizers(self):
        opt_cfg = self.cfg.train
        lr = opt_cfg.lr
        weight_decay = opt_cfg.weight_decay
        # All-linear LoRA variants get their historical flat parameter list;
        # the hybrid Utonia adaptation (V14) gets one group per role, each with
        # its own LR / weight decay, plus the downstream group at train.lr.
        param_groups = self.p2g_model.optimizer_param_groups(opt_cfg)
        betas = tuple(
            float(beta)
            for beta in (getattr(opt_cfg, "adam_betas", None) or (0.9, 0.999))
        )
        eps = float(getattr(opt_cfg, "adam_eps", None) or 1.0e-8)
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps,
        )

        # Linear warmup (over warmup_iters optimizer steps) -> cosine decay to 0
        # over the full run. Stepped per-optimizer-step (interval="step"), so the
        # warmup count is in optimizer steps (matches config warmup_iters).
        warmup = int(getattr(opt_cfg, "warmup_iters", 0) or 0)
        total_steps = int(self.trainer.estimated_stepping_batches)
        total_steps = max(total_steps, warmup + 1)
        if self._is_rank_zero():
            print(f"[sched] AdamW lr={lr} warmup_iters={warmup} "
                  f"total_optimizer_steps={total_steps} (cosine decay)", flush=True)
            for group in optimizer.param_groups:
                if "name" not in group:
                    continue
                print(
                    f"[sched]   group {group['name']}: "
                    f"{sum(p.numel() for p in group['params']):,} params in "
                    f"{len(group['params'])} tensors, lr={group['lr']:g}, "
                    f"weight_decay={group['weight_decay']:g}",
                    flush=True,
                )

        def lr_lambda(step):
            if warmup > 0 and step < warmup:
                return float(step + 1) / float(warmup)
            progress = float(step - warmup) / float(max(1, total_steps - warmup))
            progress = min(max(progress, 0.0), 1.0)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }
