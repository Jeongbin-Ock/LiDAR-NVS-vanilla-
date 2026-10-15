"""Train LiDAR4D's raydrop-refinement U-Net behind a frozen Dynamic 2DGS model.

A refinement run's config is a finished checkpoint's embedded config plus a
``raydrop_refine`` block (``refine_config=``, see config/refine/). The frozen
model (P2G, temporal G2G, G2P renderer) runs in eval mode under ``no_grad`` on
every batch, exactly as at test time, and ``RaydropRefiner`` turns its
[raydrop, intensity, median depth] maps into a refined no-return probability.
Only the U-Net trains, on BCE against the GT no-return mask over every target
view. Validation reports the refined metrics under the usual keys and the
frozen model's under ``base/``, from the same windows and renders.
"""
import torch

from src.experiment_config import checkpoint_sha256
from src.model_wrapper import ModelWrapper
from src.models.module.m4_raydrop_refine import RaydropRefiner, raydrop_scores
from src.models.utils.graphics_utils import lidar4d_range_image_to_points

# Every step, for the refined raydrop and (``base/``) the frozen model's.
RAYDROP_SCORE_KEYS = ("loss_raydrop", "raydrop_acc", "raydrop_f1", "raydrop_rmse")
# Validation/test only: Loss's official image metrics and Chamfer distance,
# which change with the raydrop mask while depth and intensity stay frozen.
IMAGE_METRIC_KEYS = (
    "depth_psnr_raydrop",
    "depth_ssim_raydrop",
    "intensity_psnr_raydrop",
    "intensity_ssim_raydrop",
    "loss_chamfer",
)
BASE_PREFIX = "base/"


class RaydropRefineWrapper(ModelWrapper):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.raydrop_refiner = RaydropRefiner(cfg.raydrop_refine)
        for module in self._frozen_modules():
            module.requires_grad_(False)
            module.eval()
        self._wandb_loss_keys = {
            *RAYDROP_SCORE_KEYS,
            *IMAGE_METRIC_KEYS,
            *(BASE_PREFIX + key for key in (*RAYDROP_SCORE_KEYS, *IMAGE_METRIC_KEYS)),
            "total",
        }

    def _frozen_modules(self):
        return (self.p2g_model, self.g2g_model, self.g2p_model)

    def train(self, mode: bool = True):
        """Lightning switches the whole module to train mode around every
        validation; the frozen model must keep rendering in eval mode."""
        super().train(mode)
        for module in self._frozen_modules():
            module.train(False)
        return self

    def load_base_checkpoint(self) -> None:
        """Load the frozen model's weights from ``raydrop_refine.base_ckpt``.

        Called once for a fresh refinement run; a resumed run restores every
        weight from its own checkpoint instead. The file must still match the
        hash recorded when the run was composed, and it must supply every
        weight except the U-Net's.
        """
        refine = self.cfg.raydrop_refine
        path = str(refine.base_ckpt)
        digest = checkpoint_sha256(path)
        if digest != str(refine.base_ckpt_sha256):
            raise RuntimeError(
                f"{path} changed since the run was composed: sha256 {digest} != "
                f"{refine.base_ckpt_sha256}"
            )
        blob = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        state = blob["state_dict"]
        # Compared by key, not via load_state_dict's report: BatchNorm does
        # not list an absent num_batches_tracked as missing.
        base_keys = {
            key for key in self.state_dict() if not key.startswith("raydrop_refiner.")
        }
        missing, unexpected = base_keys - set(state), set(state) - base_keys
        if missing or unexpected:
            raise RuntimeError(
                "Base checkpoint does not match the frozen model: "
                f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
            )
        self.load_state_dict(state, strict=False)
        if self._is_rank_zero():
            print(f"[raydrop_refine] frozen base {path} (epoch {int(blob['epoch'])}, "
                  f"step {int(blob['global_step'])}, sha256 {digest[:12]})", flush=True)

    def configure_optimizers(self):
        return self._scheduled_optimizer([{
            "name": "raydrop_refiner",
            "params": list(self.raydrop_refiner.parameters()),
        }])

    @torch.no_grad()
    def render_frozen(self, _input, gt, *, compute_points):
        """The frozen model's Gaussians and renders for one batch."""
        p2g_out = self.p2g_model(_input)
        p2g_out.pop("routing_stats", None)
        p2g_out.pop("auxiliary_batch_gaussians", None)
        gaussians = self.g2g_model(
            p2g_out, _input["timestamps_sec"], _input["window_duration_sec"],
        )
        return gaussians, self.g2p_model(gaussians, gt, compute_points=compute_points)

    @staticmethod
    def _points_with_raydrop(renders, raydrop, gt):
        """G2P's Chamfer point sets, rebuilt with another raydrop mask."""
        cameras = gt["cameras"] if isinstance(gt, dict) else gt
        return [
            [
                lidar4d_range_image_to_points(
                    renders["depth_median"][b, v],
                    camera.vfov,
                    camera.hfov,
                    row_to_theta=camera.row_to_theta,
                    raydrop=raydrop[b, v],
                    row_origin=getattr(camera, "row_origin", None),
                )
                for v, camera in enumerate(window)
            ]
            for b, window in enumerate(cameras)
        ]

    def _shared_step(self, batch, batch_idx, *, prefix: str):
        _input, gt = batch["input"], batch["gt"]
        evaluating = prefix != "train"
        gaussians, renders = self.render_frozen(_input, gt, compute_points=evaluating)
        refined = self.raydrop_refiner(renders)

        losses = {}
        for name, raydrop in (("", refined), (BASE_PREFIX, renders["raydrop"])):
            scores = raydrop_scores(raydrop, renders["gt_raydrop"], renders["view_mask"])
            losses.update({name + key: value for key, value in scores.items()})
        if evaluating:
            refined_renders = {
                **renders,
                "raydrop": refined.detach(),
                "render_points": self._points_with_raydrop(renders, refined.detach(), gt),
            }
            for name, maps in (("", refined_renders), (BASE_PREFIX, renders)):
                image = self.loss(
                    maps,
                    gaussians=gaussians,
                    metric_mode=prefix,
                    compute_valid_metrics=False,
                    compute_raydrop_metrics=True,
                )
                losses.update({name + key: image[key] for key in IMAGE_METRIC_KEYS})
        losses["total"] = losses["loss_raydrop"]

        self._log_losses(losses, prefix=prefix, batch_size=self._batch_size(_input))
        if evaluating:
            self._record_eval_summary(losses, batch_idx=batch_idx, prefix=prefix)
        groups = batch.get("meta", {}).get("window_group")
        if prefix == "val" and groups and self._val_group_metrics is not None:
            self._val_group_metrics.add(losses, groups, self._batch_size(_input))
        return losses["total"]


__all__ = ["RaydropRefineWrapper"]
