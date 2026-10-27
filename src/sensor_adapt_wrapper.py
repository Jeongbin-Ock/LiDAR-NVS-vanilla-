"""Adapt V18's sensor head to a new LiDAR behind a frozen Dynamic 2DGS model.

A sensor-adaptation run's config is a finished V18 checkpoint's embedded
config plus a ``sensor_adapt`` block (``adapt_config=``, see config/adapt/),
usually with another dataset's data block (``data_config=``). The scene model
-- P2G, temporal G2G and the Gaussians with their features -- runs in eval
mode under ``no_grad`` on every batch, exactly as at test time. Only G2P's
``lidar_decoder`` trains, the head that decodes the alpha-blended feature, ray
and depth into intensity and raydrop (V18's sensor embedding and MLP, V18.1's
MLP, V18.2's U-Net), in the wrapper's train/eval mode while every other module
stays in eval mode. It starts from the base checkpoint's weights, or with
``sensor_adapt.decoder_init=random`` from the fresh initialization the model
was built with. The objective
is the loss block's intensity L1 and raydrop BCE; depth and Chamfer stay logged
as diagnostics of the frozen geometry with weight 0.

The checkpoint keeps the model's state_dict keys, so test_lidar4d_nvs.py reads
it like any model checkpoint and renders with the adapted decoder.
"""
import torch

from src.experiment_config import checkpoint_sha256
from src.model_wrapper import WANDB_BASE_LOSS_KEYS, ModelWrapper

# The trained sensor head's state_dict keys.
DECODER_PREFIX = "g2p_model.lidar_decoder."


class SensorAdaptWrapper(ModelWrapper):
    def __init__(self, cfg):
        super().__init__(cfg)
        if self.g2p_model.lidar_decoder is None:
            raise ValueError(
                "sensor_adapt needs a model with g2p.lidar_decoder (V18, V18.1, V18.2)"
            )
        self.requires_grad_(False)
        self.g2p_model.lidar_decoder.requires_grad_(True)
        self.train(self.training)
        # The frozen motion does not train, so no velocity prior is computed.
        self._wandb_loss_keys = set(WANDB_BASE_LOSS_KEYS)

    def _frozen_modules(self):
        # G2P too, so it renders as at test time; ``train`` hands the decoder
        # it holds back to the wrapper's mode.
        return (self.p2g_model, self.g2g_model, self.g2p_model)

    def train(self, mode: bool = True):
        """Lightning switches the whole module to train mode around every
        validation; the frozen model must keep rendering in eval mode. The
        trained head follows ``mode``: V18.2's U-Net has dropout, active in
        training only (V18's and V18.1's MLPs have no mode-dependent layer)."""
        super().train(mode)
        for module in self._frozen_modules():
            module.train(False)
        self.g2p_model.lidar_decoder.train(mode)
        return self

    def load_base_checkpoint(self) -> None:
        """Load the frozen model's weights from ``sensor_adapt.base_ckpt``.

        Called once for a fresh adaptation run; a resumed run restores every
        weight from its own checkpoint instead. The file must still match the
        hash recorded when the run was composed and hold exactly this model's
        weights. ``sensor_adapt.decoder_init`` decides the sensor head:
        ``base`` (or absent) loads it too, ``random`` leaves it at the fresh
        initialization ``__init__`` gave it, seeded like a from-scratch run.
        """
        adapt = self.cfg.sensor_adapt
        path = str(adapt.base_ckpt)
        digest = checkpoint_sha256(path)
        if digest != str(adapt.base_ckpt_sha256):
            raise RuntimeError(
                f"{path} changed since the run was composed: sha256 {digest} != "
                f"{adapt.base_ckpt_sha256}"
            )
        blob = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        state = blob["state_dict"]
        # Compared by key, not via load_state_dict's report: BatchNorm does
        # not list an absent num_batches_tracked as missing.
        own_keys = set(self.state_dict())
        missing, unexpected = own_keys - set(state), set(state) - own_keys
        if missing or unexpected:
            raise RuntimeError(
                "Base checkpoint does not match the model: "
                f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
            )
        decoder_init = str(adapt.get("decoder_init", "base")).lower()
        if decoder_init == "random":
            decoder_keys = {key for key in own_keys if key.startswith(DECODER_PREFIX)}
            incompatible = self.load_state_dict(
                {key: value for key, value in state.items() if key not in decoder_keys},
                strict=False,
            )
            if set(incompatible.missing_keys) != decoder_keys or incompatible.unexpected_keys:
                raise RuntimeError(
                    "decoder_init=random must leave exactly the sensor head unloaded: "
                    f"missing={sorted(incompatible.missing_keys)}, "
                    f"unexpected={sorted(incompatible.unexpected_keys)}"
                )
        else:
            self.load_state_dict(state)
        if self._is_rank_zero():
            print(f"[sensor_adapt] frozen base {path} (epoch {int(blob['epoch'])}, "
                  f"step {int(blob['global_step'])}, sha256 {digest[:12]}); "
                  f"sensor head from {'a fresh initialization' if decoder_init == 'random' else 'the base'}",
                  flush=True)

    def configure_optimizers(self):
        return self._scheduled_optimizer([{
            "name": "lidar_decoder",
            "params": list(self.g2p_model.lidar_decoder.parameters()),
        }])

    def _shared_step(self, batch, batch_idx, *, prefix: str):
        _input, gt = batch["input"], batch["gt"]
        compute_valid_metrics, compute_official_metrics = self._metric_schedule(
            prefix, batch_idx
        )
        with torch.no_grad():
            p2g_out = self.p2g_model(_input)
            p2g_out.pop("routing_stats", None)
            p2g_out.pop("auxiliary_batch_gaussians", None)
            gaussians = self.g2g_model(
                p2g_out, _input["timestamps_sec"], _input["window_duration_sec"],
            )
        # Rendered with grad enabled for the decoder alone: no frozen input
        # requires grad, so autograd records nothing before its parameters.
        renders = self.g2p_model(gaussians, gt, compute_points=prefix != "train")
        losses = self.loss(
            renders,
            gaussians=gaussians,
            metric_mode=prefix,
            compute_valid_metrics=compute_valid_metrics,
            compute_raydrop_metrics=compute_official_metrics,
        )

        self._log_losses(losses, prefix=prefix, batch_size=self._batch_size(_input))
        if prefix != "train":
            self._record_eval_summary(losses, batch_idx=batch_idx, prefix=prefix)
        groups = batch.get("meta", {}).get("window_group")
        if prefix == "val" and groups and self._val_group_metrics is not None:
            self._val_group_metrics.add(losses, groups, self._batch_size(_input))
        return losses["total"]


__all__ = ["SensorAdaptWrapper"]
