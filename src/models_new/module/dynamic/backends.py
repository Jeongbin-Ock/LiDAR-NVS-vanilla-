"""Dynamic Gaussian backend composition."""
from __future__ import annotations

import math
from collections import defaultdict

import torch
import torch.nn as nn

from src.experiment_config.definitions import (
    FIXED_COUNT_SEED_MODES,
    SEED_MODE_RANGE_QUANTILE,
    SEED_MODE_TOKEN,
)

from ...utils import boxes as box_utils
from .common import _cfg_get
from .gaussian_slots import DenseCountGaussianHead
from .heads import EmbeddedInitVelocityHead, GaussianAttributeHead
from .local_attention import RadiusLocalAttentionBlock
from .nested_count import NestedCountGaussianHead
from .option_experts import OptionExpertGaussianHead
from .ordinal_count import OrdinalCountGaussianHead
from .temporal import (
    CosineRoPELayerWeightedCrossAttention,
    FinalFeatureBarrierCrossAttention,
    FixedMotionGainLearnableMaxSpeedBarrierCrossAttention,
    GlobalSoftplusMaxSpeedBarrierCrossAttention,
    GroupedGainBarrierCrossAttention,
    LayerwiseSoftplusMaxSpeedBarrierCrossAttention,
    LearnableMaxSpeedBarrierLayerWeightedCrossAttention,
    MaxSpeedBarrierLayerWeightedCrossAttention,
    RawQKRoPELayerWeightedCrossAttention,
    RMSNormFeatureRoPELayerWeightedCrossAttention,
    RMSNormRoPELayerWeightedCrossAttention,
)

class DynamicGaussianBackend(nn.Module):
    """Shared Dynamic 2DGS backend with a fixed Gaussian count per token.

    A subclass supplies the temporal stack through ``_build_temporal``. That
    module returns the refined token feature beside a coordinate-readout
    displacement ``delta_p_init``. Dividing it by the signed endpoint interval
    turns both source frames into one forward-time ``v_init``; the offset head,
    which embeds the ``v_init`` it corrects, adds ``v_offset``. Every one of a
    token's ``gaussians_per_token`` Gaussians shares that token's velocity.
    """

    def __init__(self, cfg, gs_params, dim: int, offset_bound: float,
                 gaussians_per_token: int = 1, gaussian_count_cfg=None,
                 seed_mode: str = SEED_MODE_RANGE_QUANTILE):
        super().__init__()
        self.cfg = cfg
        self.dim = int(dim)
        self.gaussians_per_token = int(gaussians_per_token)
        self.seed_mode = str(seed_mode).lower()
        if self.seed_mode not in FIXED_COUNT_SEED_MODES:
            raise ValueError(
                f"dynamic seed_mode must be one of {FIXED_COUNT_SEED_MODES}, "
                f"got {seed_mode!r}"
            )
        self.temporal = self._build_temporal(cfg)
        self.time_reference_sec = float(
            _cfg_get(cfg.temporal, "time_reference_sec", 1.0)
        )
        if self.time_reference_sec <= 0.0:
            raise ValueError("time_reference_sec must be positive")
        self.adaptive_count = gaussian_count_cfg is not None
        self.gaussian_head = self._build_gaussian_head(
            cfg, gs_params, offset_bound, gaussian_count_cfg
        )
        self.velocity_head = self._build_velocity_head(cfg)

    def _build_temporal(self, cfg):
        """The cross-attention stack this variant refines its tokens with."""
        raise NotImplementedError

    @property
    def distinct_slot_seeds(self) -> bool:
        """Whether a token's slots start from different seeds.

        Only range-quantile seeds differ per slot. ``medoid`` and ``token``
        share one anchor, so their seed deltas carry nothing slot-specific and
        a decoder reads the token feature alone.
        """
        return self.seed_mode == SEED_MODE_RANGE_QUANTILE

    @property
    def seed_conditioned_head(self) -> bool:
        """Whether the fixed-count head reads per-slot seed deltas.

        ``Point2Gaus`` reads this to decide whether to send the deltas. The
        learned-count router always receives its candidate bank.
        """
        return not self.adaptive_count and bool(
            getattr(self.gaussian_head, "seed_conditioned", False)
        )

    def _check_fixed_count_seeds(self, seed_local, token_position_local):
        """Normalize the source seeds to ``(N, G, 3)`` and check the count.

        The token builder always emits a padded slot axis; a backend that emits
        ``G`` Gaussians per token needs exactly ``G`` of those slots filled, and
        a mismatch here means the config's Gaussian count and the seed contract
        have drifted apart.
        """
        if token_position_local.ndim != 2 or token_position_local.shape[1] != 3:
            raise ValueError("dynamic token positions must be (N,3)")
        slots = self.gaussians_per_token
        if seed_local.ndim == 2:
            seed_local = seed_local.unsqueeze(1)
        expected = (token_position_local.shape[0], slots, 3)
        if tuple(seed_local.shape) != expected:
            raise ValueError(
                f"Dynamic 2DGS expects {slots} seed(s) per token with shape "
                f"{expected}, got {tuple(seed_local.shape)}"
            )
        return seed_local

    def _expand_to_gaussians(self, value):
        """Repeat one per-token row for each of that token's Gaussians."""
        if self.gaussians_per_token == 1:
            return value
        return value.repeat_interleave(self.gaussians_per_token, dim=0)

    @staticmethod
    def _frame_bounds(token_offset, frame):
        start = int(token_offset[frame - 1]) if frame > 0 else 0
        return start, int(token_offset[frame])

    @staticmethod
    def _assemble_batch_gaussians(
        *,
        sample_count,
        batch_index,
        position,
        raw,
        velocity,
        time_sec,
        time_normalized,
        duration_sec,
        extra_fields=None,
    ):
        """Split the packed per-Gaussian tensors into one payload per sample.

        Every dynamic backend produces the same rendering contract regardless
        of how it obtained position, attributes, and velocity, so the split is
        shared. ``extra_fields`` carries whatever the variant additionally
        exposes -- motion diagnostics, and for the adaptive-K router the token
        each Gaussian came from. ``is_dynamic`` marks every Gaussian as moving
        for the evaluation viewers; nothing here gates rendering.
        """
        extra_fields = extra_fields or {}
        batch_gaussians = []
        for batch_id in range(sample_count):
            rows = (batch_index == batch_id).nonzero(as_tuple=True)[0]
            if rows.numel() == 0:
                batch_gaussians.append(None)
                continue
            batch_gaussians.append({
                "position": position[rows],
                "shs": raw["shs"][rows],
                "opacity": raw["opacity"][rows],
                "scaling": raw["scaling"][rows],
                "rotation": raw["rotation"][rows],
                "velocity": velocity[rows],
                **{name: value[rows] for name, value in extra_fields.items()},
                "source_time_sec": time_sec[rows],
                "source_time_normalized": time_normalized[rows],
                "window_duration_sec": duration_sec[rows],
                "is_dynamic": torch.ones(
                    int(rows.numel()), dtype=torch.bool, device=rows.device
                ),
            })
        return batch_gaussians

    @staticmethod
    def _signed_pair_delta_t(geometry):
        """Forward-time travel interval for each token's source frame.

        Dynamic inputs contain exactly two endpoint frames per sample, and
        their physical timestamps are relative to the same window start. Local
        frame 0 therefore travels forward by ``+duration`` and frame 1 backward
        by ``-duration``, which is what turns both endpoints' displacements into
        one forward-time velocity convention.
        """
        local_frame = geometry["local_frame"]
        if not bool(torch.all((local_frame == 0) | (local_frame == 1))):
            raise ValueError(
                "dynamic pair velocity requires endpoint indices 0/1"
            )
        duration = geometry["duration_sec"]
        if not bool(torch.all(torch.isfinite(duration) & (duration > 0.0))):
            raise ValueError(
                "dynamic pair velocity requires a positive finite duration"
            )
        return torch.where(local_frame == 0, duration, -duration)

    def _reference_geometry_and_time(
        self,
        token_local,
        seed_local,
        token_offset,
        frame_batch_idx,
        pose_list,
        timestamps_normalized,
        timestamps_sec,
        window_duration_sec,
        seed_delta_local=None,
    ):
        """Map frame-local token and seed coordinates into the window reference."""
        device, dtype = token_local.device, token_local.dtype
        token_ref = torch.empty_like(token_local)
        seed_ref = torch.empty_like(seed_local)
        seed_delta_ref = (
            torch.empty_like(seed_delta_local)
            if seed_delta_local is not None else None
        )
        if seed_delta_local is not None and seed_delta_local.shape != seed_local.shape:
            raise ValueError("seed_delta_local must have the same shape as seeds")
        token_time_norm = torch.empty(token_local.shape[0], device=device, dtype=dtype)
        token_time_sec = torch.empty_like(token_time_norm)
        token_duration = torch.empty_like(token_time_norm)
        token_batch = torch.empty(
            token_local.shape[0], device=device, dtype=torch.long
        )
        local_frame_index = torch.empty_like(token_batch)
        seen = defaultdict(int)
        for global_frame, batch_id_value in enumerate(frame_batch_idx.tolist()):
            batch_id = int(batch_id_value)
            local_frame = seen[batch_id]
            seen[batch_id] += 1
            start, end = self._frame_bounds(token_offset, global_frame)
            pose = pose_list[batch_id][local_frame].to(device=device, dtype=dtype)
            token_ref[start:end] = box_utils.apply_pose(
                token_local[start:end], pose
            )
            seed_frame = seed_local[start:end]
            seed_ref[start:end] = box_utils.apply_pose(
                seed_frame.reshape(-1, 3), pose
            ).reshape_as(seed_frame)
            if seed_delta_ref is not None:
                delta_frame = seed_delta_local[start:end]
                center_frame = seed_frame - delta_frame
                center_ref = box_utils.apply_pose(
                    center_frame.reshape(-1, 3), pose
                ).reshape_as(center_frame)
                seed_delta_ref[start:end] = seed_ref[start:end] - center_ref
            t_norm = timestamps_normalized[batch_id].to(device=device, dtype=dtype)
            t_sec = timestamps_sec[batch_id].to(device=device, dtype=dtype)
            if local_frame >= t_norm.numel() or local_frame >= t_sec.numel():
                raise ValueError("source timestamp count does not match input frames")
            token_time_norm[start:end] = t_norm[local_frame]
            token_time_sec[start:end] = t_sec[local_frame]
            duration = torch.as_tensor(
                window_duration_sec[batch_id], device=device, dtype=dtype
            )
            token_duration[start:end] = duration
            token_batch[start:end] = batch_id
            local_frame_index[start:end] = local_frame
        return {
            "token_ref": token_ref,
            "seed_ref": seed_ref,
            "seed_delta_ref": seed_delta_ref,
            "time_normalized": token_time_norm,
            "time_sec": token_time_sec,
            "duration_sec": token_duration,
            "batch": token_batch,
            "local_frame": local_frame_index,
        }

    def _build_gaussian_head(self, cfg, gs_params, offset_bound,
                             gaussian_count_cfg):
        """The 2DGS attribute decoder, fixed-count unless a router is given.

        A backend that supports the learned-count router overrides this; every
        other one emits ``gaussians_per_token`` Gaussians from one head and
        rejects a router config it could not honour.
        """
        if gaussian_count_cfg is not None:
            raise ValueError(
                f"{type(self).__name__} emits a fixed number of Gaussians per "
                "token and cannot use p2g.grid_query.count_mode=learned_gumbel"
            )
        return GaussianAttributeHead(
            _cfg_get(cfg, "gaussian_head", None),
            gs_params,
            self.dim,
            offset_bound,
            gaussians_per_token=self.gaussians_per_token,
        )

    def _build_velocity_head(self, cfg):
        """The additive correction applied on top of ``v_init``."""
        return EmbeddedInitVelocityHead(cfg.motion, self.dim)

    def _temporal_coordinate(self, geometry):
        # A fixed reference unit preserves cadence: endpoints 0.8 s apart are
        # encoded as [0, 0.8], not collapsed to [0, 1].
        return geometry["time_sec"] / self.time_reference_sec

    def _temporal_refine(self, fused_feature, geometry, token_offset,
                         frame_batch_idx):
        return self.temporal(
            fused_feature,
            geometry["token_ref"],
            token_offset,
            frame_batch_idx,
            self._temporal_coordinate(geometry),
            geometry["duration_sec"],
        )

    def _velocity_offset(self, refined, velocity_init):
        """The additive correction applied on top of ``v_init``."""
        return self.velocity_head(refined, velocity_init)

    def _predict_motion(self, refined, geometry, position, temporal_fields):
        del position
        pair_delta_t_sec = self._signed_pair_delta_t(geometry)
        delta_p_init = temporal_fields["delta_p_init"]
        velocity_init = delta_p_init / pair_delta_t_sec.unsqueeze(-1)
        velocity_offset = self._velocity_offset(refined, velocity_init)
        velocity = velocity_init + velocity_offset
        motion_fields = {
            "delta_p_match": temporal_fields["delta_p_match"],
            "delta_p_init": delta_p_init,
            "matched_position": temporal_fields["matched_position"],
            "pair_delta_t_sec": pair_delta_t_sec,
            "velocity_match": velocity_init,
            "velocity_init": velocity_init,
            "velocity_offset": velocity_offset,
        }
        # Layer-mixture variants publish their per-layer logits and weights.
        # V11.2 reads correspondence from final f' instead, so it deliberately
        # publishes neither a dormant logit nor a misleading layer-weight chart.
        for name in ("motion_layer_logits", "motion_layer_weights"):
            if name in temporal_fields:
                motion_fields[name] = temporal_fields[name]
        return velocity, motion_fields

    def _predict_gaussian(self, refined, geometry):
        del geometry
        return self.gaussian_head(refined)

    def forward(
        self,
        fused_feature,
        token_position_local,
        seed_local,
        token_offset,
        frame_batch_idx,
        pose_list,
        timestamps_normalized,
        timestamps_sec,
        window_duration_sec,
        seed_delta_local=None,
        seed_delta_sensor=None,
    ):
        if seed_delta_local is not None and seed_delta_sensor is not None:
            raise ValueError("pass only one of seed_delta_local and seed_delta_sensor")
        if seed_delta_local is None:
            seed_delta_local = seed_delta_sensor
        if fused_feature.ndim != 2 or fused_feature.shape[1] != self.dim:
            raise ValueError(f"dynamic fused_feature must be (N,{self.dim})")
        seed_local = self._check_fixed_count_seeds(
            seed_local, token_position_local
        )
        if seed_delta_local is not None:
            seed_delta_local = self._check_fixed_count_seeds(
                seed_delta_local, token_position_local
            )

        geometry = self._reference_geometry_and_time(
            token_position_local,
            seed_local,
            token_offset,
            frame_batch_idx,
            pose_list,
            timestamps_normalized,
            timestamps_sec,
            window_duration_sec,
            seed_delta_local=seed_delta_local,
        )
        refined, temporal_fields = self._temporal_refine(
            fused_feature,
            geometry,
            token_offset,
            frame_batch_idx,
        )
        raw, position_offset, _gaussian_feature = self._predict_gaussian(
            refined, geometry
        )
        # Attributes and position are per Gaussian; correspondence, motion and
        # source timing are per token. With a fixed count per token the second
        # group is index-expanded, so a per-Gaussian mean over any of these
        # stays exactly the per-token mean.
        position = geometry["seed_ref"].reshape(-1, 3) + position_offset
        token_velocity, token_motion_fields = self._predict_motion(
            refined, geometry, position, temporal_fields
        )
        velocity = self._expand_to_gaussians(token_velocity)
        motion_fields = {
            name: self._expand_to_gaussians(value)
            for name, value in token_motion_fields.items()
        }

        batch_gaussians = self._assemble_batch_gaussians(
            sample_count=len(pose_list),
            batch_index=self._expand_to_gaussians(geometry["batch"]),
            position=position,
            raw=raw,
            velocity=velocity,
            time_sec=self._expand_to_gaussians(geometry["time_sec"]),
            time_normalized=self._expand_to_gaussians(
                geometry["time_normalized"]
            ),
            duration_sec=self._expand_to_gaussians(geometry["duration_sec"]),
            extra_fields=motion_fields,
        )
        return {
            "batch_gaussians": batch_gaussians,
            "gaussians": batch_gaussians,
            "batch": self._expand_to_gaussians(geometry["batch"]),
        }

class RouterCapableGaussianBackend(DynamicGaussianBackend):
    """A backend whose Gaussian count a learned K={1,2,3} router may choose.

    ``p2g.grid_query.count_mode`` decides. With ``learned_gumbel`` the
    normalized refined token routes through the grid ``GridSlotHead``: only the
    selected K-specific joint head runs, and every child of a token shares its
    velocity while the additive correction stays at token resolution. A
    ``legacy`` count config instead builds the shared fixed-count head, and the
    temporal stack is then the only thing that distinguishes the backend.
    """

    def _build_gaussian_head(self, cfg, gs_params, offset_bound,
                             gaussian_count_cfg):
        """The learned-count router, or the shared fixed-count head.

        ``p2g.grid_query.count_mode`` decides. With ``learned_gumbel`` the
        refined token routes through the grid ``GridSlotHead`` and only the
        selected K-specific joint head runs; otherwise this backend behaves
        like every other fixed-count variant and the temporal stack is the only
        thing that distinguishes it.
        """
        if gaussian_count_cfg is None:
            return super()._build_gaussian_head(
                cfg, gs_params, offset_bound, None
            )
        from ..grid_query_head import GridSlotHead

        self.gaussian_output_norm = nn.LayerNorm(self.dim)
        head = GridSlotHead(
            gaussian_count_cfg, gs_params, dim=self.dim,
            seed_conditioned=self.distinct_slot_seeds,
        )
        if head.count_mode != "learned_gumbel":
            raise ValueError("adaptive-K Gaussian count must use learned_gumbel")
        if head.k_max != 3:
            raise ValueError("adaptive-K Gaussian count requires K_max=3")
        self.offset_bound = float(offset_bound)
        self.gaussian_param_names = [
            "shs", "opacity", "scaling", "rotation", "offset",
        ]
        self.gaussian_param_sizes = [
            int(getattr(gs_params, name)) for name in self.gaussian_param_names
        ]
        if self.gaussian_param_sizes[-2:] != [4, 3]:
            raise ValueError(
                "adaptive-K Gaussian head requires rotation=4 and offset=3"
            )
        if sum(self.gaussian_param_sizes) != head.param_dim:
            raise ValueError("adaptive-K Gaussian parameter widths disagree")
        self._initialize_adaptive_gaussian_heads(
            _cfg_get(cfg, "gaussian_head", None), head
        )
        return head

    def _initialize_adaptive_gaussian_heads(self, cfg, head=None):
        """Give every K expert the existing dynamic-head output priors."""
        head = self.gaussian_head if head is None else head
        opacity = float(_cfg_get(cfg, "initial_opacity", 0.2))
        opacity = min(max(opacity, 1.0e-4), 1.0 - 1.0e-4)
        opacity_bias = math.log(opacity / (1.0 - opacity))
        scale = float(_cfg_get(cfg, "initial_scale_m", 0.3))
        scale_bias = math.log(math.expm1(max(scale, 1.0e-4)))

        sh_size, opacity_size, scale_size, rotation_size, offset_size = (
            self.gaussian_param_sizes
        )
        opacity_start = sh_size
        scale_start = opacity_start + opacity_size
        rotation_start = scale_start + scale_size
        offset_start = rotation_start + rotation_size
        block_bias = head.k_heads[0].bias.new_zeros(head.param_dim)
        block_bias[opacity_start:scale_start] = opacity_bias
        block_bias[scale_start:rotation_start] = scale_bias
        block_bias[rotation_start] = 1.0

        for k, k_head in enumerate(head.k_heads, start=1):
            nn.init.normal_(k_head.weight, mean=0.0, std=0.01)
            with torch.no_grad():
                k_head.bias.copy_(block_bias.repeat(k))
                for slot in range(k):
                    row_start = slot * head.param_dim + offset_start
                    row_end = row_start + offset_size
                    k_head.weight[row_start:row_end].zero_()

    def _predict_adaptive_gaussians(self, refined, geometry, token_offset):
        seed_ref = geometry["seed_ref"]
        expected_seed_shape = (
            refined.shape[0],
            self.gaussian_head.k_max,
            self.gaussian_head.k_max,
            3,
        )
        if tuple(seed_ref.shape) != expected_seed_shape:
            raise ValueError(
                "adaptive-K seed bank must have shape "
                f"{expected_seed_shape}, got {tuple(seed_ref.shape)}"
            )
        gaussian_feature = self.gaussian_output_norm(refined)
        delta_ref = geometry.get("seed_delta_ref")
        if delta_ref is None or tuple(delta_ref.shape) != expected_seed_shape:
            raise ValueError(
                "adaptive-K dynamic backend requires a transformed K-specific seed-delta bank"
            )
        raw_params, packing = self.gaussian_head(
            gaussian_feature, delta_ref, token_offset,
        )
        selected_index = packing["anchor_k"] - 1
        token_rows = torch.arange(
            refined.shape[0], device=refined.device
        )
        selected_seed = seed_ref[token_rows, selected_index]
        anchor_index = packing["anchor_index"]
        slot_index = packing["slot_index"]
        packed_seed = selected_seed[anchor_index, slot_index]

        parts = torch.split(raw_params, self.gaussian_param_sizes, dim=-1)
        raw = dict(zip(self.gaussian_param_names, parts))
        position = packed_seed + self.offset_bound * torch.tanh(
            raw.pop("offset")
        )
        gradient_weight = self.gaussian_head.gradient_weight(
            packing["slot_k"], position.dtype
        )
        if (
            self.gaussian_head.grad_balance == "sqrt_k"
            and self.gaussian_head.grad_balance_scope == "output"
        ):
            from ..gaussian_assembly import gradient_scale_identity

            position = gradient_scale_identity(position, gradient_weight)
            raw = {
                name: gradient_scale_identity(value, gradient_weight)
                for name, value in raw.items()
            }
        return raw, position, packing, gradient_weight

    def forward(
        self,
        fused_feature,
        token_position_local,
        seed_local,
        token_offset,
        frame_batch_idx,
        pose_list,
        timestamps_normalized,
        timestamps_sec,
        window_duration_sec,
        seed_delta_local=None,
        seed_delta_sensor=None,
    ):
        if seed_delta_local is not None and seed_delta_sensor is not None:
            raise ValueError("pass only one of seed_delta_local and seed_delta_sensor")
        if seed_delta_local is None:
            seed_delta_local = seed_delta_sensor
        if not self.adaptive_count:
            # Without a router this backend is an ordinary fixed-count variant;
            # the shared forward already implements that. Deltas are then the
            # per-slot
            # ``(N, G, 3)`` set a seed-conditioned head reads, never the
            # K-specific ``(N, K, K, 3)`` bank, and the shared forward
            # validates that shape.
            if seed_delta_local is not None and not getattr(
                self, "seed_conditioned_head", False
            ):
                raise ValueError(
                    "a fixed Gaussian count needs no K-specific seed-delta bank"
                )
            return super().forward(
                fused_feature,
                token_position_local,
                seed_local,
                token_offset,
                frame_batch_idx,
                pose_list,
                timestamps_normalized,
                timestamps_sec,
                window_duration_sec,
                seed_delta_local=seed_delta_local,
            )
        if fused_feature.ndim != 2 or fused_feature.shape[1] != self.dim:
            raise ValueError(f"dynamic fused_feature must be (N,{self.dim})")
        expected_seed_shape = (
            fused_feature.shape[0],
            self.gaussian_head.k_max,
            self.gaussian_head.k_max,
            3,
        )
        if tuple(seed_local.shape) != expected_seed_shape:
            raise ValueError(
                "adaptive-K dynamic backend requires the learned-count seed bank with shape "
                f"{expected_seed_shape}, got {tuple(seed_local.shape)}"
            )
        if seed_delta_local is None or tuple(
            seed_delta_local.shape
        ) != expected_seed_shape:
            raise ValueError(
                "adaptive-K dynamic backend requires a learned-count seed-delta bank with shape "
                f"{expected_seed_shape}"
            )
        if token_position_local.shape != (fused_feature.shape[0], 3):
            raise ValueError("adaptive-K token positions must align with features")

        geometry = self._reference_geometry_and_time(
            token_position_local,
            seed_local,
            token_offset,
            frame_batch_idx,
            pose_list,
            timestamps_normalized,
            timestamps_sec,
            window_duration_sec,
            seed_delta_local=seed_delta_local,
        )
        refined, temporal_fields = self._temporal_refine(
            fused_feature,
            geometry,
            token_offset,
            frame_batch_idx,
        )
        raw, position, packing, gradient_weight = (
            self._predict_adaptive_gaussians(
                refined, geometry, token_offset
            )
        )

        token_velocity, token_motion_fields = self._predict_motion(
            refined, geometry, None, temporal_fields
        )
        anchor_index = packing["anchor_index"]
        velocity = token_velocity[anchor_index]
        motion_fields = {
            name: value[anchor_index]
            for name, value in token_motion_fields.items()
        }
        if (
            self.gaussian_head.grad_balance == "sqrt_k"
            and self.gaussian_head.grad_balance_scope == "output"
        ):
            from ..gaussian_assembly import gradient_scale_identity

            velocity = gradient_scale_identity(velocity, gradient_weight)

        packed_batch = geometry["batch"][anchor_index]
        packed_time_sec = geometry["time_sec"][anchor_index]
        packed_time_normalized = geometry["time_normalized"][anchor_index]
        packed_duration = geometry["duration_sec"][anchor_index]
        packed_selected_k = packing["anchor_k"][anchor_index]

        batch_gaussians = self._assemble_batch_gaussians(
            sample_count=len(pose_list),
            batch_index=packed_batch,
            position=position,
            raw=raw,
            velocity=velocity,
            time_sec=packed_time_sec,
            time_normalized=packed_time_normalized,
            duration_sec=packed_duration,
            extra_fields={
                **motion_fields,
                "source_token_index": anchor_index,
                "selected_k": packed_selected_k,
                "gaussian_slot_index": packing["slot_index"],
            },
        )

        routing_stats = {
            "k_logits": packing["k_logits"].detach(),
            "selected_k": packing["anchor_k"].detach(),
            "token_position_local": token_position_local.detach(),
        }
        return {
            "batch_gaussians": batch_gaussians,
            "gaussians": batch_gaussians,
            "batch": packed_batch,
            "routing_stats": routing_stats,
        }

class MaxSpeedBarrierVelocityGaussianBackend(RouterCapableGaussianBackend):
    """V11/V11.4/V14: count-selectable backend with split position encoding.

    The shipped config uses the ``learned_gumbel`` K={1,2,3} router, while a
    legacy count config selects the fixed-count head. Both keep one token velocity
    shared by its Gaussian children. Correspondence comes from head 0 of each
    layer under the max-speed soft barrier, and plain learned scalars mix the
    layer readouts.
    """

    def _build_temporal(self, cfg):
        return MaxSpeedBarrierLayerWeightedCrossAttention(
            cfg.temporal, self.dim
        )


class LearnableMaxSpeedBarrierVelocityGaussianBackend(
    MaxSpeedBarrierVelocityGaussianBackend
):
    """V12: V11.4 with only its global barrier speed made learnable."""

    def _build_temporal(self, cfg):
        return LearnableMaxSpeedBarrierLayerWeightedCrossAttention(
            cfg.temporal, self.dim
        )


class FixedMotionGainMaxSpeedBarrierVelocityGaussianBackend(
    LearnableMaxSpeedBarrierVelocityGaussianBackend
):
    """V12.1: V12 with no learnable RMS gain on correspondence head 0."""

    def _build_temporal(self, cfg):
        return FixedMotionGainLearnableMaxSpeedBarrierCrossAttention(
            cfg.temporal, self.dim
        )


class LayerwiseSoftplusMaxSpeedBarrierVelocityGaussianBackend(
    MaxSpeedBarrierVelocityGaussianBackend
):
    """V12.2: V12 with one softplus-parameterized barrier speed per layer."""

    def _build_temporal(self, cfg):
        return LayerwiseSoftplusMaxSpeedBarrierCrossAttention(
            cfg.temporal, self.dim
        )


class GlobalSoftplusMaxSpeedBarrierVelocityGaussianBackend(
    MaxSpeedBarrierVelocityGaussianBackend
):
    """V14.1: V14 with one softplus-parameterized global barrier speed."""

    def _build_temporal(self, cfg):
        return GlobalSoftplusMaxSpeedBarrierCrossAttention(
            cfg.temporal, self.dim
        )


class CosineRoPEVelocityGaussianBackend(
    MaxSpeedBarrierVelocityGaussianBackend
):
    """V13 adaptive-K backend with all-head RoPE cosine correspondence."""

    def _build_temporal(self, cfg):
        return CosineRoPELayerWeightedCrossAttention(
            cfg.temporal, self.dim
        )


class RawQKRoPEVelocityGaussianBackend(
    CosineRoPEVelocityGaussianBackend
):
    """V13.2 adaptive-K backend with raw QK / sqrt(d) on every head."""

    def _build_temporal(self, cfg):
        return RawQKRoPELayerWeightedCrossAttention(
            cfg.temporal, self.dim
        )


class RMSNormRoPEVelocityGaussianBackend(
    CosineRoPEVelocityGaussianBackend
):
    """V13.3 adaptive-K backend with RMSNorm and 3D RoPE on every head."""

    def _build_temporal(self, cfg):
        return RMSNormRoPELayerWeightedCrossAttention(
            cfg.temporal, self.dim
        )


class RMSNormFeatureRoPEVelocityGaussianBackend(
    RMSNormRoPEVelocityGaussianBackend
):
    """V13.4 adaptive-K backend with no 3D RoPE on match head 0."""

    def _build_temporal(self, cfg):
        return RMSNormFeatureRoPELayerWeightedCrossAttention(
            cfg.temporal, self.dim
        )


class FinalFeatureBarrierVelocityGaussianBackend(
    MaxSpeedBarrierVelocityGaussianBackend
):
    """V11.2: V11 with correspondence recomputed from complete final ``f'``.

    Adaptive K, Gaussian construction, the embedded velocity offset, physical
    signed-duration conversion, and child-velocity sharing are inherited from
    V11 without modification.
    """

    def _build_temporal(self, cfg):
        return FinalFeatureBarrierCrossAttention(cfg.temporal, self.dim)


class SeedConditionedBarrierVelocityGaussianBackend(
    MaxSpeedBarrierVelocityGaussianBackend
):
    """V11.3/V11.5/V11.6: V11's temporal stack and correspondence, fixed count.

    V11.5 shares this class under V11.4's linear-hinge temporal config, the
    same way V11.4 shares V11's class; V11.6 is V11.5 at three slots.

    The adaptive-K router is gone, so ``p2g.grid_query`` fixes how many
    Gaussians every token emits. What the router's decoder did beyond routing
    is rebuilt here, so only the count differs from V11:

    * Its trunk was the only place V11 fed seed geometry into the decoder,
      concatenating the selected candidate row's ``3 * K_max`` deltas to the
      normalized token feature. With range-quantile seeds the head is
      therefore built seed-conditioned, at ``3 * gaussians_per_token``
      channels, one row per slot it owns. Shared-anchor seed modes
      (``medoid``, ``token``) give every slot the same delta, so their head
      reads the token feature alone, as anchor-based Gaussian decoders do,
      and the slots diverge through their own parameter rows.
    * That trunk was also the decoder's only nonlinearity. The head therefore
      keeps it: ``Linear(dim + 3 * G, dim) -> SiLU`` before the attribute
      projections, which is V11's shape with ``G`` slots in place of ``K_max``
      candidates, or ``Linear(dim, dim) -> SiLU`` without seed conditioning.
    """

    #: Subclasses keep the same fixed-count decoder under another temporal stack.
    fixed_variant_label = "V11.3/V11.5/V11.6"

    def _build_gaussian_head(self, cfg, gs_params, offset_bound,
                             gaussian_count_cfg):
        if gaussian_count_cfg is not None:
            raise ValueError(
                f"{self.fixed_variant_label} emits a fixed number of Gaussians "
                "per token and cannot "
                "use p2g.grid_query.count_mode=learned_gumbel"
            )
        return GaussianAttributeHead(
            _cfg_get(cfg, "gaussian_head", None),
            gs_params,
            self.dim,
            offset_bound,
            gaussians_per_token=self.gaussians_per_token,
            seed_conditioned=self.distinct_slot_seeds,
            trunk=True,
        )

    def _predict_gaussian(self, refined, geometry):
        if not self.gaussian_head.seed_conditioned:
            return self.gaussian_head(refined)
        seed_delta_ref = geometry.get("seed_delta_ref")
        if seed_delta_ref is None:
            raise ValueError(
                f"{self.fixed_variant_label} requires the transformed per-slot "
                "seed deltas; "
                "Point2Gaus must pass seed_delta_local"
            )
        return self.gaussian_head(refined, seed_delta_ref)


class SeedConditionedCosineRoPEVelocityGaussianBackend(
    SeedConditionedBarrierVelocityGaussianBackend
):
    """V13.1: V13 temporal matching with fixed K=2 seed conditioning."""

    fixed_variant_label = "V13.1"

    def _build_temporal(self, cfg):
        return CosineRoPELayerWeightedCrossAttention(
            cfg.temporal, self.dim
        )


class SingleGaussianBarrierVelocityGaussianBackend(DynamicGaussianBackend):
    """V11.1: V11's temporal stack with a fixed Gaussian count.

    The shipped config emits one medoid-seeded Gaussian per occupied token, and
    ``p2g.grid_query.K_max`` can select another fixed K. Repeated token fields are
    index-expanded by the same K, preserving token-mean motion diagnostics.

    1. **The QK-Norm gain is split per head group** (see
       :class:`GroupedGainBarrierCrossAttention`).

    2. **The adaptive-K router is absent.** The plain
       :class:`GaussianAttributeHead` owns one parameter block per configured
       slot, with no count predictor, opacity gate, or per-K gradient balancing.
    """

    def _build_temporal(self, cfg):
        return GroupedGainBarrierCrossAttention(cfg.temporal, self.dim)


class LocalAttentionBarrierVelocityGaussianBackend(DynamicGaussianBackend):
    """The V15 family: V14.1's temporal stack, local attention, a token head.

    * The temporal stack is V14.1's global-softplus barrier cross-attention
      (built without LayerScale by ``temporal.layer_scale=false``), and the
      velocity is read from its output exactly as in V14.1: ``v_init`` from the
      head-0 layer mixture, ``v_offset`` from the temporal feature.
    * One pre-LN radius local-attention block then refines every token from the
      tokens of both sweeps within ``local_attention.radius_m`` of it, in the
      window reference frame. Only the Gaussian head reads this refined feature.
    * The Gaussian head decodes every token's Gaussians around its one anchor,
      the Utonia token position. A subclass picks the head, and with it which
      Gaussians a token renders.

    Every head returns ``(rendered, auxiliary, routing)``. ``rendered`` becomes
    ``batch_gaussians``, so evaluation, the test script and every tool render
    exactly it. A routed head also returns ``routing``, published as
    ``routing_stats``, and in training it may return ``auxiliary``, a
    router-free second set published as ``auxiliary_batch_gaussians``, which
    ``ModelWrapper`` renders and averages into the optimized loss. Every
    Gaussian of a token shares that token's velocity, and every token's first
    packed Gaussian has ``gaussian_slot_index == 0``, which is what keeps the
    per-token velocity prior counting each token once.

    ``ROUTED`` heads choose the count after refinement, so their tokens carry
    one seed (``gaussians_per_token=1``) and need ``p2g.grid_query``'s routed
    count block; the dense head takes ``gaussians_per_token`` from the
    fixed-count seed contract and accepts no count config.
    """

    ROUTED = True

    def __init__(self, cfg, gs_params, dim: int, offset_bound: float,
                 gaussians_per_token: int = 1, gaussian_count_cfg=None,
                 seed_mode: str = SEED_MODE_TOKEN):
        if self.ROUTED:
            if gaussian_count_cfg is None:
                raise ValueError(
                    f"{type(self).__name__} needs p2g.grid_query with its "
                    "routed count block"
                )
            if int(gaussians_per_token) != 1:
                raise ValueError(
                    "routed tokens carry one shared anchor; the router, not "
                    "the seed contract, decides how many Gaussians a token "
                    "renders"
                )
        elif gaussian_count_cfg is not None:
            raise ValueError(
                f"{type(self).__name__} has no count router; its count is "
                "the fixed-count seed contract's K_max"
            )
        if str(seed_mode).lower() != SEED_MODE_TOKEN:
            raise ValueError(
                "the V15 family anchors every Gaussian at its Utonia token "
                f"position (seed_mode={SEED_MODE_TOKEN!r}), got {seed_mode!r}"
            )
        super().__init__(
            cfg, gs_params, dim, offset_bound,
            gaussians_per_token=gaussians_per_token,
            gaussian_count_cfg=gaussian_count_cfg,
            seed_mode=seed_mode,
        )
        self.local_attention = self._build_local_attention(cfg)

    def _build_temporal(self, cfg):
        return GlobalSoftplusMaxSpeedBarrierCrossAttention(
            cfg.temporal, self.dim
        )

    def _build_local_attention(self, cfg):
        local_cfg = _cfg_get(cfg, "local_attention", None)
        if local_cfg is None:
            raise ValueError(
                "the V15 family needs a dynamic_2dgs.local_attention block"
            )
        return RadiusLocalAttentionBlock(local_cfg, self.dim)

    @staticmethod
    def _token_gaussian_fields(packed, geometry, token_velocity,
                               token_motion_fields, token_gaussian_count,
                               sample_count):
        """Expand per-token fields onto a packed Gaussian set and split it."""
        token_index = packed["token_index"]
        motion_fields = {
            name: value[token_index]
            for name, value in token_motion_fields.items()
        }
        if "opacity_gate" in packed:
            motion_fields["opacity_gate"] = packed["opacity_gate"]
        if "blend_mask" in packed:
            motion_fields["blend_mask"] = packed["blend_mask"]
        return DynamicGaussianBackend._assemble_batch_gaussians(
            sample_count=sample_count,
            batch_index=geometry["batch"][token_index],
            position=packed["position"],
            raw=packed["raw"],
            velocity=token_velocity[token_index],
            time_sec=geometry["time_sec"][token_index],
            time_normalized=geometry["time_normalized"][token_index],
            duration_sec=geometry["duration_sec"][token_index],
            extra_fields={
                **motion_fields,
                "source_token_index": token_index,
                "selected_k": token_gaussian_count[token_index],
                "gaussian_slot_index": packed["slot_index"],
            },
        )

    def forward(
        self,
        fused_feature,
        token_position_local,
        seed_local,
        token_offset,
        frame_batch_idx,
        pose_list,
        timestamps_normalized,
        timestamps_sec,
        window_duration_sec,
        seed_delta_local=None,
        seed_delta_sensor=None,
    ):
        if seed_delta_local is not None or seed_delta_sensor is not None:
            raise ValueError(
                "the V15 family decodes around the shared token anchor and "
                "reads no seed deltas"
            )
        if fused_feature.ndim != 2 or fused_feature.shape[1] != self.dim:
            raise ValueError(f"dynamic fused_feature must be (N,{self.dim})")
        seed_local = self._check_fixed_count_seeds(
            seed_local, token_position_local
        )
        geometry = self._reference_geometry_and_time(
            token_position_local,
            seed_local,
            token_offset,
            frame_batch_idx,
            pose_list,
            timestamps_normalized,
            timestamps_sec,
            window_duration_sec,
        )
        refined, temporal_fields = self._temporal_refine(
            fused_feature,
            geometry,
            token_offset,
            frame_batch_idx,
        )
        # The velocity offset reads the temporal feature, before the local
        # refinement, exactly where V14.1 reads it.
        token_velocity, token_motion_fields = self._predict_motion(
            refined, geometry, None, temporal_fields
        )
        local_feature = self.local_attention(
            refined, geometry["token_ref"], geometry["batch"]
        )
        # Every slot of a token-seeded token sits on the same anchor.
        rendered, auxiliary, routing = self.gaussian_head(
            local_feature, geometry["seed_ref"][:, 0]
        )

        sample_count = len(pose_list)
        if routing is None:
            token_count = torch.full(
                (local_feature.shape[0],),
                self.gaussian_head.gaussians_per_token,
                dtype=torch.long, device=local_feature.device,
            )
        else:
            token_count = routing["count"]
        batch_gaussians = self._token_gaussian_fields(
            rendered, geometry, token_velocity, token_motion_fields,
            token_count, sample_count,
        )
        output = {
            "batch_gaussians": batch_gaussians,
            "gaussians": batch_gaussians,
            "batch": geometry["batch"][rendered["token_index"]],
        }
        if routing is not None:
            output["routing_stats"] = {
                "k_logits": routing["logits"].detach(),
                "selected_option": routing["option"].detach(),
                "selected_k": routing["count"].detach(),
                "option_counts": self.gaussian_head.option_count_values,
                "token_position_local": token_position_local.detach(),
            }
        if auxiliary is not None:
            auxiliary_count = torch.full_like(
                token_count, self.gaussian_head.auxiliary_count
            )
            output["auxiliary_batch_gaussians"] = self._token_gaussian_fields(
                auxiliary, geometry, token_velocity, token_motion_fields,
                auxiliary_count, sample_count,
            )
        return output


class NestedCountBarrierVelocityGaussianBackend(
    LocalAttentionBarrierVelocityGaussianBackend
):
    """V15 and V15.1: a router picks a nested prefix of G1..G5 per token.

    :class:`NestedCountGaussianHead` decodes G1..G5 around the token anchor and
    routes each token to ``{G1}``, ``{G1..G3}`` or ``{G1..G5}``. In V15's
    training the auxiliary set is every token's G1..G3; V15.1 has none.
    """

    def _build_gaussian_head(self, cfg, gs_params, offset_bound,
                             gaussian_count_cfg):
        return NestedCountGaussianHead(
            gaussian_count_cfg,
            _cfg_get(cfg, "gaussian_head", None),
            gs_params,
            self.dim,
            offset_bound,
        )


class OptionExpertBarrierVelocityGaussianBackend(
    LocalAttentionBarrierVelocityGaussianBackend
):
    """V15.2: V15's router, but every option renders its own expert.

    :class:`OptionExpertGaussianHead` decodes E1, E3 and E5 from one shared
    trunk around the token anchor and renders the chosen option's expert,
    every Gaussian of it gated by that option. In training the auxiliary set is
    every token's E5.
    """

    def _build_gaussian_head(self, cfg, gs_params, offset_bound,
                             gaussian_count_cfg):
        return OptionExpertGaussianHead(
            gaussian_count_cfg,
            _cfg_get(cfg, "gaussian_head", None),
            gs_params,
            self.dim,
            offset_bound,
        )


class OrdinalCountBarrierVelocityGaussianBackend(
    LocalAttentionBarrierVelocityGaussianBackend
):
    """V16: V15's nested G1..G5, chosen by two bounded ordinal gates.

    :class:`OrdinalCountGaussianHead` decodes G1..G5 around the token anchor
    with V15's decoder; a keep gate (G2/G3) and an add gate (G4/G5 given
    G2/G3) pick ``{G1}``, ``{G1..G3}`` or ``{G1..G5}``. In training the routed
    set is every token's G1..G5 carrying per-Gaussian ``blend_mask`` values for
    masked rasterization, and the auxiliary set is the same Gaussians
    unmasked; evaluation renders each token's most probable prefix.
    """

    def _build_gaussian_head(self, cfg, gs_params, offset_bound,
                             gaussian_count_cfg):
        return OrdinalCountGaussianHead(
            gaussian_count_cfg,
            _cfg_get(cfg, "gaussian_head", None),
            gs_params,
            self.dim,
            offset_bound,
        )


class DenseCountBarrierVelocityGaussianBackend(
    LocalAttentionBarrierVelocityGaussianBackend
):
    """V15.3: V15 without the router; every token renders five Gaussians.

    :class:`DenseCountGaussianHead` is V15's two-layer decoder with nothing
    routed: every token's ``gaussians_per_token`` (V15.3: five) decoded
    Gaussians are rendered, in training and at evaluation, with no opacity
    gate, no auxiliary render and no ``routing_stats``.
    """

    ROUTED = False

    def _build_gaussian_head(self, cfg, gs_params, offset_bound,
                             gaussian_count_cfg):
        del gaussian_count_cfg
        return DenseCountGaussianHead(
            _cfg_get(cfg, "gaussian_head", None),
            gs_params,
            self.dim,
            offset_bound,
            gaussians_per_token=self.gaussians_per_token,
        )


__all__ = [
    "DynamicGaussianBackend",
    "RouterCapableGaussianBackend",
    "MaxSpeedBarrierVelocityGaussianBackend",
    "LearnableMaxSpeedBarrierVelocityGaussianBackend",
    "FixedMotionGainMaxSpeedBarrierVelocityGaussianBackend",
    "LayerwiseSoftplusMaxSpeedBarrierVelocityGaussianBackend",
    "GlobalSoftplusMaxSpeedBarrierVelocityGaussianBackend",
    "CosineRoPEVelocityGaussianBackend",
    "RawQKRoPEVelocityGaussianBackend",
    "RMSNormRoPEVelocityGaussianBackend",
    "RMSNormFeatureRoPEVelocityGaussianBackend",
    "FinalFeatureBarrierVelocityGaussianBackend",
    "SeedConditionedBarrierVelocityGaussianBackend",
    "SeedConditionedCosineRoPEVelocityGaussianBackend",
    "SingleGaussianBarrierVelocityGaussianBackend",
    "LocalAttentionBarrierVelocityGaussianBackend",
    "NestedCountBarrierVelocityGaussianBackend",
    "OptionExpertBarrierVelocityGaussianBackend",
    "OrdinalCountBarrierVelocityGaussianBackend",
    "DenseCountBarrierVelocityGaussianBackend",
]
