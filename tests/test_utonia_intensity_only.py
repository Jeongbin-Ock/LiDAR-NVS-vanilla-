"""V17.1's frozen backbone, intensity gradient path and config boundaries."""
import copy

import pytest
import torch
from torch import nn
from omegaconf import OmegaConf

from src.experiment_config import resolve_main_config, resolve_eval_config
from src.models.utonia.adapters import (
    XYZIStem,
    adapt_utonia_stem_to_intensity_only,
    build_hybrid_param_groups,
    check_param_group_coverage,
    resolve_utonia_adaptation_settings,
    UtoniaIntensityOnlySettings,
)
from src.models.utonia.model import Embedding


def config(*overrides, variant="17_1"):
    return resolve_main_config(OmegaConf.from_dotlist([
        f"model.variant=dynamic_2dgs_attention_velocity_v{variant}", *overrides,
    ]))[0]


def test_config_preserves_v15_4_downstream_and_training():
    baseline, ablation = config(variant="15_4"), config()
    a, b = [OmegaConf.to_container(c, resolve=True) for c in (baseline, ablation)]
    for container in (a, b):
        for key in ("model", "exp_name", "logger"):
            container.pop(key)
        container["p2g"].pop("utonia_lora")
        container["p2g"].pop("utonia_adaptation")
        container["train"]["utonia_param_groups"] = {
            "intensity": container["train"]["utonia_param_groups"]["intensity"]
        }
    assert a == b
    assert ablation.p2g.utonia_lora.enable is False
    assert isinstance(resolve_utonia_adaptation_settings(ablation.p2g), UtoniaIntensityOnlySettings)


@pytest.mark.parametrize("override", [
    "p2g.utonia_lora.enable=true", "p2g.freeze_utonia=false",
    "p2g.utonia_adaptation.mode=hybrid", "p2g.utonia_adaptation.train_layer_norm=true",
    "train.utonia_param_groups.lora.lr=0.001", "p2g.utonia_adaptation=null",
    "train.utonia_param_groups=null", "dynamic_2dgs.temporal.num_heads=12",
])
def test_config_rejects_changes_to_ablation_scope(override):
    with pytest.raises(ValueError):
        config(override)


def test_only_intensity_updates_and_gradients_cross_frozen_layers():
    torch.manual_seed(54)
    backbone = nn.Module()
    backbone.embedding = Embedding(9, 54, norm_layer=nn.LayerNorm, act_layer=nn.GELU)
    # A frozen layer after the stem must transmit gradients into the I column.
    backbone.enc = nn.Sequential(nn.Linear(54, 54), nn.GELU(), nn.LayerNorm(54))
    original = copy.deepcopy(backbone)
    original_state = {k: v.clone() for k, v in backbone.state_dict().items()}
    assert adapt_utonia_stem_to_intensity_only(backbone) == 9
    stem = backbone.embedding.stem
    assert type(stem.linear) is XYZIStem
    assert stem.linear.intensity_weight.shape == (54,)
    torch.testing.assert_close(stem.linear.xyz.weight, original_state["embedding.stem.linear.weight"][:, :3], rtol=0, atol=0)
    assert torch.count_nonzero(stem.linear.intensity_weight) == 0
    trainable = [(n, p) for n, p in backbone.named_parameters() if p.requires_grad]
    assert [(n, p.numel()) for n, p in trainable] == [("embedding.stem.linear.intensity_weight", 54)]
    x = torch.randn(19, 4)
    x9 = torch.cat([x[:, :3], torch.zeros(19, 6)], dim=-1)
    projected = stem.act(stem.norm(stem.linear(x)))
    expected = original.embedding.stem
    torch.testing.assert_close(projected, expected.act(expected.norm(expected.linear(x9))), rtol=1e-6, atol=1e-6)

    downstream = nn.Linear(54, 2)
    cfg = config()
    groups = build_hybrid_param_groups(
        [("embedding", backbone.embedding), ("enc", backbone.enc)],
        list(downstream.named_parameters()),
        OmegaConf.create({
            "lr": cfg.train.lr, "weight_decay": cfg.train.weight_decay,
            "utonia_param_groups": OmegaConf.to_container(cfg.train.utonia_param_groups, resolve=True),
        }),
    )
    check_param_group_coverage(groups, trainable + list(downstream.named_parameters()))
    assert [g["name"] for g in groups] == ["utonia_intensity", "downstream"]
    optimizer = torch.optim.AdamW(groups)
    frozen = {n: p.clone() for n, p in backbone.named_parameters() if not p.requires_grad}
    loss = downstream(backbone.enc(projected)).square().mean()
    loss.backward()
    assert stem.linear.intensity_weight.grad is not None
    assert stem.linear.intensity_weight.grad.norm() > 0
    assert all(p.grad is None for p in backbone.parameters() if not p.requires_grad)
    optimizer.step()
    assert torch.count_nonzero(stem.linear.intensity_weight) > 0
    for name, value in backbone.named_parameters():
        if name in frozen:
            torch.testing.assert_close(value, frozen[name], rtol=0, atol=0)


def test_checkpoint_config_preserves_intensity_only_and_protects_scope(tmp_path):
    cfg = config()
    checkpoint = tmp_path / "v17_1.ckpt"
    torch.save({"experiment_config": OmegaConf.to_container(cfg, resolve=True)}, checkpoint)
    restored, _ = resolve_eval_config(OmegaConf.from_dotlist([f"test.ckpt_path={checkpoint}"]))
    assert restored.p2g.utonia_adaptation.mode == "intensity_only"
    with pytest.raises(ValueError):
        resolve_eval_config(OmegaConf.from_dotlist([
            f"test.ckpt_path={checkpoint}", "p2g.utonia_lora.enable=true",
        ]))
