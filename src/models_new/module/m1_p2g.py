from __future__ import annotations

import os

import torch
import torch.nn as nn

from .. import utonia
from ..utonia.lora import (
    adapt_utonia_embedding_to_xyzi,
    build_xyzi_utonia_input,
    freeze_utonia_and_enable_active_lora,
    inject_utonia_lora,
    resolve_utonia_lora_settings,
    set_active_lora_mode,
)
from .builders import OccupiedGridTokenBuilder
from .builders.common import GridSeedConfig, UtoniaGridMapper


def _uses_learned_count(cfg) -> bool:
    """Whether ``p2g.grid_query`` asks the router to predict the count."""
    grid_query = getattr(cfg, "grid_query", None)
    if grid_query is None:
        return False
    count_mode = str(getattr(grid_query, "count_mode", "legacy")).lower()
    return count_mode in GridSeedConfig.LEARNED_COUNT_MODES


class Point2Gaus(nn.Module):
    """Point cloud to Dynamic 2D Gaussians.

    Utonia runs on XYZI input with LoRA adapters on its frozen base. Every
    occupied Utonia cell becomes one token, and the variant's backend turns the
    two endpoint frames' tokens into Gaussians with a per-token velocity. New
    configs feed each scan in its own ego frame; backend poses map those local
    coordinates into the frame-0 ego reference before temporal matching.
    """

    def __init__(self, cfg, dynamic_cfg, dynamic_variant):
        super().__init__()
        self.cfg = cfg
        self.dynamic_cfg = dynamic_cfg
        self.dynamic_variant = dynamic_variant

        self.freeze_utonia = bool(getattr(cfg, "freeze_utonia", True))
        self.utonia_lora_settings = resolve_utonia_lora_settings(cfg)
        self.utonia_lora_enabled = self.utonia_lora_settings.enabled
        if not self.utonia_lora_enabled:
            raise ValueError(
                "Point2Gaus requires p2g.utonia_lora.enable=true: intensity "
                "enters Utonia through its XYZI LoRA stem"
            )
        if not self.freeze_utonia:
            raise ValueError(
                "p2g.utonia_lora.enable=true requires p2g.freeze_utonia=true; "
                "LoRA fine-tunes adapters while the pretrained Utonia base "
                "stays frozen"
            )
        # Load the frozen Utonia encoder from a repo-local checkpoint so the weights
        # and their embedded architecture config live under the project (not ~/.cache).
        # The architecture config is also dumped to config/utonia_pretrained.yaml for
        # inspection. Falls back to a HuggingFace download if the local file is absent.
        _repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        _default_ckpt = os.path.join(_repo_root, "checkpoints", "utonia", "utonia.pth")
        utonia_ckpt = getattr(cfg, "utonia_ckpt", None) or _default_ckpt
        utonia_name = utonia_ckpt if os.path.isfile(utonia_ckpt) else "utonia"
        self.feature_extractor = utonia.load(
            utonia_name,
            repo_id="Pointcept/Utonia",
            custom_config={"freeze_encoder": True},
        )
        self.utonia_coord_scale = 0.2
        self.utonia_input_grid_size = 0.01
        self.utonia_feature_stage = int(
            getattr(cfg, "utonia_feature_stage", len(self.feature_extractor.enc) - 1)
        )
        self._validate_utonia_feature_stage()
        self.utonia_stride_factor = self._infer_utonia_stride_factor()
        self.utonia_feature_grid_size = self._infer_utonia_feature_grid_size()
        self.utonia_feature_dim = self._infer_utonia_feature_dim()

        # Load the untouched 9D checkpoint before migrating its functional
        # [xyz, 0, ..., 0] stem to [xyz, intensity]. Both the new intensity
        # base column and every LoRA up projection start at zero.
        old_input_channels = adapt_utonia_embedding_to_xyzi(self.feature_extractor)
        self.utonia_lora_report = inject_utonia_lora(
            self.feature_extractor, self.utonia_lora_settings
        )
        self.utonia_lora_trainable_parameters = (
            freeze_utonia_and_enable_active_lora(
                self.feature_extractor, self._active_utonia_lora_roots()
            )
        )
        print(
            "Utonia LoRA: "
            f"input={old_input_channels}->4 (xyzi), "
            f"linear={self.utonia_lora_report.linear_modules} "
            f"rank={self.utonia_lora_settings.linear_rank}, "
            f"subm_conv3d={self.utonia_lora_report.conv_modules} "
            f"rank={self.utonia_lora_settings.conv_rank}, "
            "active_trainable="
            f"{self.utonia_lora_trainable_parameters / 1e6:.2f}M"
        )
        self.feature_extractor.eval()
        set_active_lora_mode(self._active_utonia_lora_roots(), self.training)

        self.grid_mapper = UtoniaGridMapper(
            self.utonia_coord_scale, self.utonia_input_grid_size, self.utonia_stride_factor
        )
        # p2g.grid_query decides how many Gaussians a token becomes: absent or
        # count_mode=legacy gives every token the same K_max (default one),
        # while a learned count_mode hands the decision to the router. Nothing
        # about that choice is pinned to the variant name.
        self.dynamic_adaptive_count = _uses_learned_count(cfg)
        self.anchor_builder = OccupiedGridTokenBuilder(cfg)
        self.dynamic_gaussians_per_token = (
            1 if self.dynamic_adaptive_count
            else self.anchor_builder.grid_seed_config.gaussians_per_token
        )
        # The direct 432D (stage 3) Utonia XYZI feature is the temporal token.
        self.trunk_dim = self.utonia_feature_dim
        self.offset_bound = float(getattr(cfg, "head_offset_bound", 0.8))

        from src.config_loader import (
            DYNAMIC_VARIANT_V11,
            DYNAMIC_VARIANT_V11_1,
            DYNAMIC_VARIANT_V11_2,
            DYNAMIC_VARIANT_V11_3,
            DYNAMIC_VARIANT_V11_4,
            DYNAMIC_VARIANT_V11_5,
            DYNAMIC_VARIANT_V11_6,
            DYNAMIC_VARIANT_V12,
            DYNAMIC_VARIANT_V12_1,
            DYNAMIC_VARIANT_V12_2,
            DYNAMIC_VARIANT_V13,
            DYNAMIC_VARIANT_V13_1,
            DYNAMIC_VARIANT_V13_2,
            DYNAMIC_VARIANT_V13_3,
            DYNAMIC_VARIANT_V13_4,
        )
        from .dynamic_gaussian import (
            CosineRoPEVelocityGaussianBackend,
            FinalFeatureBarrierVelocityGaussianBackend,
            FixedMotionGainMaxSpeedBarrierVelocityGaussianBackend,
            LayerwiseSoftplusMaxSpeedBarrierVelocityGaussianBackend,
            LearnableMaxSpeedBarrierVelocityGaussianBackend,
            MaxSpeedBarrierVelocityGaussianBackend,
            RawQKRoPEVelocityGaussianBackend,
            RMSNormFeatureRoPEVelocityGaussianBackend,
            RMSNormRoPEVelocityGaussianBackend,
            SeedConditionedBarrierVelocityGaussianBackend,
            SeedConditionedCosineRoPEVelocityGaussianBackend,
            SingleGaussianBarrierVelocityGaussianBackend,
        )

        # V11 reads correspondence from head 0 at every layer under a
        # max-speed barrier and mixes layers with plain learned scalars. V11.2
        # moves correspondence out of the stack: every layer head gets RoPE so
        # the loop is one FlashAttention call, and one dedicated readout head
        # builds the initializer from the complete final feature under V11's
        # barrier. V11.3 keeps V11's stack and correspondence but replaces the
        # router with a fixed count, and rebuilds the seed conditioning the
        # router's trunk used to supply on the plain Gaussian head. V11.4 is
        # V11 with only the hinge shape changed, so it shares V11's class and
        # differs by dynamic_2dgs.temporal config alone. V11.5 likewise shares
        # V11.3's fixed-count class, run under V11.4's linear-hinge temporal
        # config. V12 keeps V11.4's complete path and swaps in one learned
        # log-space barrier speed. V12.1 fixes only head 0's RMS gain while the
        # RoPE heads retain their own shared learned gain. V12.2 instead gives
        # every layer its own learned speed, softplus(v_raw_l). V13 removes
        # every QK RMSNorm and the barrier, L2-normalizes only head 0 at a
        # fixed temperature, and runs all RoPE heads in one FlashAttention call.
        # V13.1 keeps that matcher but replaces adaptive K with fixed
        # range-quantile K=2. V13.2 instead keeps V13's adaptive count and makes
        # head 0 use the same raw QK/sqrt(d) score as every feature head. V13.3
        # restores V11.4's Q/K RMSNorm, removes the barrier, and gives every
        # head 3D RoPE so the complete stack stays on FlashAttention. V13.4
        # keeps that stack and RMSNorm but leaves match head 0 unrotated.
        backend_cls = {
            DYNAMIC_VARIANT_V11: MaxSpeedBarrierVelocityGaussianBackend,
            DYNAMIC_VARIANT_V11_1: SingleGaussianBarrierVelocityGaussianBackend,
            DYNAMIC_VARIANT_V11_2: FinalFeatureBarrierVelocityGaussianBackend,
            DYNAMIC_VARIANT_V11_3: SeedConditionedBarrierVelocityGaussianBackend,
            DYNAMIC_VARIANT_V11_4: MaxSpeedBarrierVelocityGaussianBackend,
            DYNAMIC_VARIANT_V11_5: SeedConditionedBarrierVelocityGaussianBackend,
            DYNAMIC_VARIANT_V11_6: SeedConditionedBarrierVelocityGaussianBackend,
            DYNAMIC_VARIANT_V12: LearnableMaxSpeedBarrierVelocityGaussianBackend,
            DYNAMIC_VARIANT_V12_1: (
                FixedMotionGainMaxSpeedBarrierVelocityGaussianBackend
            ),
            DYNAMIC_VARIANT_V12_2: (
                LayerwiseSoftplusMaxSpeedBarrierVelocityGaussianBackend
            ),
            DYNAMIC_VARIANT_V13: CosineRoPEVelocityGaussianBackend,
            DYNAMIC_VARIANT_V13_1: (
                SeedConditionedCosineRoPEVelocityGaussianBackend
            ),
            DYNAMIC_VARIANT_V13_2: RawQKRoPEVelocityGaussianBackend,
            DYNAMIC_VARIANT_V13_3: RMSNormRoPEVelocityGaussianBackend,
            DYNAMIC_VARIANT_V13_4: RMSNormFeatureRoPEVelocityGaussianBackend,
        }.get(self.dynamic_variant)
        if backend_cls is None:
            raise ValueError(
                f"Unsupported dynamic model variant={self.dynamic_variant!r}"
            )
        # The seed mode decides whether the Gaussian head reads per-slot seed
        # deltas (range quantiles) or the token feature alone (shared anchors).
        backend_kwargs = {
            "seed_mode": self.anchor_builder.grid_seed_config.seed_mode,
        }
        if self.dynamic_adaptive_count:
            backend_kwargs["gaussian_count_cfg"] = cfg.grid_query
        else:
            backend_kwargs["gaussians_per_token"] = (
                self.dynamic_gaussians_per_token
            )
        self.dynamic_backend = backend_cls(
            self.dynamic_cfg,
            cfg.gs_params,
            dim=self.trunk_dim,
            offset_bound=self.offset_bound,
            **backend_kwargs,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        # The frozen base keeps its eval-mode normalization; only the active
        # LoRA adapters follow the requested mode.
        if getattr(self, "freeze_utonia", False):
            self.feature_extractor.eval()
            set_active_lora_mode(self._active_utonia_lora_roots(), mode)
        return self

    def _active_utonia_lora_roots(self):
        """Modules executed by the configured early-exit encoder stage.

        Adapters are installed throughout Utonia, but stages after the selected
        feature stage stay frozen so DDP never sees trainable unused parameters.
        """
        roots = [self.feature_extractor.embedding]
        roots.extend(
            self.feature_extractor.enc[index]
            for index in range(self.utonia_feature_stage + 1)
        )
        return roots

    def _validate_utonia_feature_stage(self):
        max_stage = len(self.feature_extractor.enc) - 1
        if self.utonia_feature_stage < 0 or self.utonia_feature_stage > max_stage:
            raise ValueError(
                f"p2g.utonia_feature_stage must be in [0, {max_stage}], "
                f"got {self.utonia_feature_stage}"
            )

    def _infer_utonia_stride_factor(self, stage: int = None):
        stage = self.utonia_feature_stage if stage is None else int(stage)
        stride_factor = 1
        if getattr(self.feature_extractor, "enc_mode", False):
            for stage_idx in range(1, stage + 1):
                down = getattr(self.feature_extractor.enc[stage_idx], "down", None)
                if down is not None:
                    stride_factor *= int(down.stride)
        return stride_factor

    def _infer_utonia_feature_grid_size(self):
        return (
            self.utonia_input_grid_size
            / self.utonia_coord_scale
            * self.utonia_stride_factor
        )

    def _infer_utonia_feature_dim(self):
        stage_module = self.feature_extractor.enc[self.utonia_feature_stage]
        for module in reversed(list(stage_module.modules())):
            if hasattr(module, "channels"):
                return int(module.channels)
            if hasattr(module, "out_channels"):
                return int(module.out_channels)
        raise RuntimeError(f"Could not infer Utonia feature dim for stage {self.utonia_feature_stage}")

    def _forward_utonia_features(self, ptv3_input):
        ptv3_input = build_xyzi_utonia_input(ptv3_input)
        max_stage = len(self.feature_extractor.enc) - 1
        if self.utonia_feature_stage == max_stage:
            return self.feature_extractor(ptv3_input)

        point = utonia.structure.Point(ptv3_input)
        point = self.feature_extractor.embedding(point)
        point.serialization(
            order=self.feature_extractor.order,
            shuffle_orders=self.feature_extractor.shuffle_orders,
        )
        point.sparsify()

        for stage_idx in range(self.utonia_feature_stage + 1):
            point = self.feature_extractor.enc[stage_idx](point)
        return point

    def forward(self, _input):
        """Tokenize both endpoint frames and run the variant's backend.

        Train/eval behaviour follows ``nn.Module.training``, so this module
        needs neither a split name nor a step counter from the caller.
        """
        # ``lidar_points_local`` is ego_i in the current config and the
        # historical LiDAR sensor frame for checkpoint configs saved before
        # data.coordinate_frame existed. Tokens and seeds share this basis.
        if "lidar_points_local" in _input:
            lidar_points = _input["lidar_points_local"]
        elif "lidar_points_sensor" in _input:
            lidar_points = _input["lidar_points_sensor"]
        else:
            lidar_points = _input["lidar_points"]
        offset = _input["offset"]
        point_batch_idx = _input["batch_idx"]
        pose = _input["pose"]
        # Bboxes remain in the dataloader's contract, but the backend does not
        # even read their keys, which keeps the model genuinely bbox-free.
        features = self._forward_utonia_features(_input["ptv3_input"])

        token_batch = self.anchor_builder(
            lidar_points, offset, pose, features, _input["ptv3_input"], self.grid_mapper
        )
        pos_list = token_batch.positions
        grid_seed_list = token_batch.grid_seeds

        # Per-frame batch index.
        frame_starts = torch.cat([torch.tensor([0], device=offset.device), offset[:-1]])
        frame_batch_idx = point_batch_idx[frame_starts.long()]  # (n_frames,)

        all_offset = []
        cumsum = 0
        for pos_i in pos_list:
            cumsum += pos_i.shape[0]
            all_offset.append(torch.tensor(cumsum, device=pos_i.device))

        all_pos = torch.cat(pos_list, dim=0)                          # (N_valid, 3)
        all_ufeat = torch.cat(token_batch.utonia_features, dim=0)     # (N_valid, C_stage)
        new_offset = torch.stack(all_offset)                          # (n_frames,)
        frame_batch_idx = frame_batch_idx.to(
            device=all_ufeat.device, dtype=torch.long
        )
        all_seed_local = torch.cat(
            [item.seed_local for item in grid_seed_list], dim=0
        )
        all_delta_local = torch.cat(
            [item.delta_local for item in grid_seed_list], dim=0
        )

        if all_ufeat.shape[1] != self.utonia_feature_dim:
            raise RuntimeError(
                f"Utonia stage {self.utonia_feature_stage} feature dim mismatch: "
                f"expected {self.utonia_feature_dim}, got {all_ufeat.shape[1]}"
            )

        timestamps_sec = _input.get("timestamps_sec")
        window_duration_sec = _input.get("window_duration_sec")
        if timestamps_sec is None or window_duration_sec is None:
            raise KeyError(
                "Dynamic 2DGS requires timestamps_sec and "
                "window_duration_sec from the dataloader"
            )
        dynamic_kwargs = {}
        # The router selects a candidate row from the seed bank, and a
        # seed-conditioned fixed-count head reads its slots directly. Either
        # way the decoder needs the own-frame seed deltas.
        if self.dynamic_adaptive_count or getattr(
            self.dynamic_backend, "seed_conditioned_head", False
        ):
            dynamic_kwargs["seed_delta_local"] = all_delta_local
        dynamic_out = self.dynamic_backend(
            all_ufeat,
            all_pos,
            all_seed_local,
            new_offset,
            frame_batch_idx,
            pose,
            _input["timestamps"],
            timestamps_sec,
            window_duration_sec,
            **dynamic_kwargs,
        )
        routing_stats = dynamic_out.pop("routing_stats", None)
        return {
            **dynamic_out,
            "pose": pose,
            "timestamps": _input["timestamps"],
            "timestamps_sec": timestamps_sec,
            "window_duration_sec": window_duration_sec,
            "routing_stats": routing_stats,
        }
