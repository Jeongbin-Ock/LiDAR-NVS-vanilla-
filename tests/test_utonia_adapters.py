"""The V14 hybrid Utonia adaptation: stem, FFN adapters, LN, LoRA scope, groups.

The census tests build the real Utonia encoder architecture from
``config/utonia_pretrained.yaml`` with random weights on the CPU, so they need
neither the pretrained checkpoint nor a GPU.
"""
import copy
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from src.models_new.utonia.adapters import (
    NonlinearAdapter,
    ParallelAdapterMLP,
    XYZIAdapterStem,
    adapt_utonia_stem_to_hybrid,
    attach_utonia_ffn_adapters,
    build_hybrid_param_groups,
    check_param_group_coverage,
    freeze_utonia_and_enable_hybrid,
    hybrid_adaptation_statistics,
    inject_selected_utonia_lora,
    resolve_utonia_adaptation_settings,
    set_active_adaptation_mode,
    utonia_parameter_roles,
)
from src.models_new.utonia.lora import (
    LoRALinear,
    LoRASubMConv3d,
    resolve_utonia_lora_settings,
)
from src.models_new.utonia.model import (
    MLP,
    Block,
    Embedding,
    PointTransformerV3,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
STAGE = 3


def _hybrid_cfg(**overrides):
    block = {
        "mode": "hybrid",
        "lora_targets": ["attention", "cpe", "pooling"],
        "train_layer_norm": True,
        "stem": {
            "train_intensity_column": True,
            "adapter_hidden": 16,
            "adapter_activation": "gelu",
            "adapter_scale": 1.0,
        },
        "ffn_adapter": {
            "widths": [8, 16, 32, 64],
            "activation": "relu",
            "scale": 0.1,
            "dropout": 0.0,
        },
    }
    block.update(overrides)
    return OmegaConf.create({"utonia_adaptation": block})


def _settings(**overrides):
    return resolve_utonia_adaptation_settings(_hybrid_cfg(**overrides))


def _lora_settings():
    return resolve_utonia_lora_settings(SimpleNamespace(utonia_lora=SimpleNamespace(
        enable=True, rank=16, alpha=16.0, dropout=0.0, input_mode="xyzi",
    )))


def _train_cfg():
    return OmegaConf.create({
        "lr": 5.0e-4,
        "weight_decay": 1.0e-4,
        "utonia_param_groups": {
            "intensity": {"lr": 1.0e-3, "weight_decay": 1.0e-4},
            "adapter": {
                "lr": 1.0e-3, "weight_decay": 1.0e-4, "bias_weight_decay": 0.0,
            },
            "lora": {"lr": 5.0e-5, "weight_decay": 1.0e-4},
            "layer_norm": {"lr": 5.0e-5, "weight_decay": 0.0},
        },
    })


def _pretrained_architecture(seed):
    config = OmegaConf.to_container(
        OmegaConf.load(REPO_ROOT / "config" / "utonia_pretrained.yaml")
    )
    config["freeze_encoder"] = True
    torch.manual_seed(seed)
    return PointTransformerV3(**config)


def _hybrid_encoder(seed=0, settings=None):
    """The real Utonia tree with the V14 adaptation, as Point2Gaus builds it."""
    settings = settings or _settings()
    encoder = _pretrained_architecture(seed)
    adapt_utonia_stem_to_hybrid(encoder, settings)
    report = inject_selected_utonia_lora(
        encoder, STAGE, _lora_settings(), settings.lora_targets
    )
    ffn_count = attach_utonia_ffn_adapters(encoder, STAGE, settings)
    roots = _named_roots(encoder)
    trainable = freeze_utonia_and_enable_hybrid(
        encoder, [module for _, module in roots], settings
    )
    return encoder, report, ffn_count, trainable


def _named_roots(encoder):
    return [("embedding", encoder.embedding)] + [
        (f"enc.enc{index}", encoder.enc[index]) for index in range(STAGE + 1)
    ]


@pytest.fixture(scope="module")
def hybrid():
    encoder, report, ffn_count, trainable = _hybrid_encoder()
    return SimpleNamespace(
        encoder=encoder, report=report, ffn_count=ffn_count,
        trainable=trainable,
    )


def _stem_backbone(in_channels=9, width=7):
    backbone = nn.Module()
    backbone.embedding = Embedding(
        in_channels=in_channels,
        embed_channels=width,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
    )
    return backbone


def test_hybrid_stem_starts_as_the_pretrained_xyz_stem():
    torch.manual_seed(3)
    backbone = _stem_backbone()
    with torch.no_grad():
        backbone.embedding.stem.norm.weight.uniform_(0.5, 1.5)
        backbone.embedding.stem.norm.bias.uniform_(-0.2, 0.2)
    original = copy.deepcopy(backbone.embedding.stem)
    xyz = torch.randn(11, 3) * 4.0
    intensity = torch.rand(11, 1)
    expected = original(torch.cat([xyz, torch.zeros(11, 6)], dim=-1))

    old_width = adapt_utonia_stem_to_hybrid(backbone, _settings())
    stem = backbone.embedding.stem.linear
    actual = backbone.embedding.stem(torch.cat([xyz, intensity], dim=-1))

    assert old_width == 9
    assert isinstance(stem, XYZIAdapterStem)
    assert backbone.embedding.in_channels == 4
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(stem.xyz.weight, original.linear.weight[:, :3])
    torch.testing.assert_close(stem.xyz.bias, original.linear.bias)
    assert torch.count_nonzero(stem.intensity_weight) == 0
    # The pre-LN output is W_xyz p + b0 exactly, and the LN/GELU are the
    # original modules with their pretrained affine.
    assert backbone.embedding.stem.norm is not None
    torch.testing.assert_close(
        backbone.embedding.stem.norm.weight, original.norm.weight
    )


def test_intensity_column_and_stem_adapter_train_while_xyz_stays_frozen():
    torch.manual_seed(5)
    backbone = _stem_backbone()
    settings = _settings()
    adapt_utonia_stem_to_hybrid(backbone, settings)
    freeze_utonia_and_enable_hybrid(backbone, [backbone.embedding], settings)
    stem = backbone.embedding.stem.linear
    feature = torch.cat([torch.randn(16, 3), torch.rand(16, 1)], dim=-1)

    backbone.embedding.stem(feature).square().sum().backward()

    assert stem.xyz.weight.grad is None and stem.xyz.bias.grad is None
    assert not stem.xyz.weight.requires_grad
    # w_I is linear in I from the first step.
    assert stem.intensity_weight.grad.abs().sum() > 0
    # Zero up-projection: up receives a gradient, down's is exactly zero.
    assert stem.adapter.up.weight.grad.abs().sum() > 0
    assert torch.count_nonzero(stem.adapter.down.weight.grad) == 0

    optimizer = torch.optim.SGD(
        [p for p in backbone.parameters() if p.requires_grad], lr=0.1
    )
    optimizer.step()
    optimizer.zero_grad()
    backbone.embedding.stem(feature).square().sum().backward()
    assert stem.adapter.down.weight.grad.abs().sum() > 0
    assert stem.xyz.weight.grad is None


def test_parallel_ffn_adapter_adds_a_scaled_branch_on_the_same_input():
    torch.manual_seed(7)
    block = Block(
        channels=12, num_heads=2, enable_flash=False,
        upcast_attention=True, upcast_softmax=True,
    )
    original = copy.deepcopy(block.mlp[0])
    encoder = nn.Module()
    stage = nn.Sequential()
    stage.add_module("block0", block)
    encoder.enc = [stage]
    settings = _settings(ffn_adapter={
        "widths": [4], "activation": "relu", "scale": 0.1, "dropout": 0.0,
    })
    assert attach_utonia_ffn_adapters(encoder, 0, settings) == 1
    wrapped = block.mlp[0]
    assert isinstance(wrapped, ParallelAdapterMLP)
    assert wrapped.ffn is not None and type(wrapped.ffn) is MLP
    assert wrapped.adapter.down.out_features == 4

    z = torch.randn(9, 12)
    torch.testing.assert_close(block.mlp(z), original(z))

    with torch.no_grad():
        wrapped.adapter.up.weight.normal_()
        wrapped.adapter.up.bias.normal_()
    raw_branch = wrapped.adapter.up(torch.relu(wrapped.adapter.down(z)))
    torch.testing.assert_close(block.mlp(z), original(z) + 0.1 * raw_branch)
    stats = wrapped.statistics
    assert set(stats) == {"ffn_rms", "adapter_rms"}


def test_hybrid_scope_matches_the_spec_census(hybrid):
    encoder = hybrid.encoder
    roles = utonia_parameter_roles(_named_roots(encoder))
    counts = {
        role: sum(parameter.numel() for _, parameter in items)
        for role, items in roles.items()
    }
    assert hybrid.trainable == 4_423_130
    assert counts["intensity"] == 54
    assert counts["lora"] == 3_657_312
    assert counts["layer_norm"] == 39_528
    # 21 FFN adapters (725,238) plus the 4 -> 16 -> 54 stem adapter (998).
    assert counts["adapter_weight"] + counts["adapter_bias"] == 725_238 + 998
    assert hybrid.ffn_count == 21
    assert hybrid.report.linear_modules == 66
    assert hybrid.report.conv_modules == 21
    layer_norms = [
        module for _, root in _named_roots(encoder)
        for module in root.modules() if isinstance(module, nn.LayerNorm)
    ]
    assert len(layer_norms) == 67
    assert all(
        parameter.requires_grad
        for module in layer_norms for parameter in module.parameters()
    )


def test_lora_only_on_attention_cpe_and_pooling(hybrid):
    encoder = hybrid.encoder
    for name, module in encoder.named_modules():
        if isinstance(module, (LoRALinear, LoRASubMConv3d)):
            assert name.startswith("enc.enc"), name
            assert any(
                part in name
                for part in (".attn.qkv", ".attn.proj", ".cpe.0", ".cpe.1",
                             ".down.proj")
            ), name
            assert ".mlp." not in name and "adapter" not in name, name
    wrapped = [
        module for module in encoder.modules()
        if isinstance(module, ParallelAdapterMLP)
    ]
    assert len(wrapped) == 21
    for module in wrapped:
        assert type(module.ffn.fc1) is nn.Linear
        assert type(module.ffn.fc2) is nn.Linear
        assert not module.ffn.fc1.weight.requires_grad
        assert not module.ffn.fc2.weight.requires_grad
    stem = encoder.embedding.stem.linear
    assert isinstance(stem, XYZIAdapterStem)
    assert type(stem.xyz) is nn.Linear
    assert not stem.xyz.weight.requires_grad


def test_unexecuted_stage_and_mask_token_stay_frozen_and_unadapted(hybrid):
    encoder = hybrid.encoder
    enc4 = encoder.enc[4]
    assert not any(
        isinstance(module, (LoRALinear, LoRASubMConv3d, NonlinearAdapter,
                            ParallelAdapterMLP))
        for module in enc4.modules()
    )
    assert not any(parameter.requires_grad for parameter in enc4.parameters())
    assert not encoder.embedding.mask_token.requires_grad
    trainable_names = [
        name for name, parameter in encoder.named_parameters()
        if parameter.requires_grad
    ]
    assert not any(name.startswith("enc.enc4.") for name in trainable_names)
    assert not any("base_layer" in name for name in trainable_names)
    assert not any(".mlp.0.ffn." in name for name in trainable_names)


def test_optimizer_groups_follow_the_role_table(hybrid):
    encoder = hybrid.encoder
    downstream = nn.Sequential(nn.Linear(4, 3), nn.LayerNorm(3))
    groups = build_hybrid_param_groups(
        _named_roots(encoder), list(downstream.named_parameters()), _train_cfg()
    )
    trainable = [
        (f"encoder.{name}", parameter)
        for name, parameter in encoder.named_parameters()
        if parameter.requires_grad
    ] + list(downstream.named_parameters())
    check_param_group_coverage(groups, trainable)

    by_name = {group["name"]: group for group in groups}
    assert list(by_name) == [
        "utonia_intensity", "utonia_adapter_weight", "utonia_adapter_bias",
        "utonia_lora", "utonia_layer_norm", "downstream",
    ]
    expected = {
        "utonia_intensity": (1.0e-3, 1.0e-4),
        "utonia_adapter_weight": (1.0e-3, 1.0e-4),
        "utonia_adapter_bias": (1.0e-3, 0.0),
        "utonia_lora": (5.0e-5, 1.0e-4),
        "utonia_layer_norm": (5.0e-5, 0.0),
        "downstream": (5.0e-4, 1.0e-4),
    }
    for name, (lr, weight_decay) in expected.items():
        assert by_name[name]["lr"] == pytest.approx(lr)
        assert by_name[name]["weight_decay"] == pytest.approx(weight_decay)
    stem = encoder.embedding.stem.linear
    # The 1-D intensity column is a weight, not a bias or LN parameter.
    assert [id(p) for p in by_name["utonia_intensity"]["params"]] == [
        id(stem.intensity_weight)
    ]
    assert all(
        parameter.ndim == 1 for parameter in by_name["utonia_adapter_bias"]["params"]
    )
    # Downstream keeps its own LayerNorm/bias under the shared decay.
    assert len(by_name["downstream"]["params"]) == 4

    optimizer = torch.optim.AdamW(groups, lr=5.0e-4, weight_decay=1.0e-4)
    warmup, total = 500, 3000

    def lr_lambda(step):
        if step < warmup:
            return float(step + 1) / float(warmup)
        progress = min(max((step - warmup) / (total - warmup), 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    for step in (0, 1, 499, 500, 1700):
        while scheduler.last_epoch < step:
            optimizer.step()
            scheduler.step()
        factor = lr_lambda(step)
        for group, (name, (lr, _)) in zip(
            optimizer.param_groups, expected.items()
        ):
            assert group["name"] == name
            assert group["lr"] == pytest.approx(lr * factor)


def test_coverage_check_rejects_missing_duplicate_and_frozen_parameters():
    trained = nn.Parameter(torch.zeros(2))
    other = nn.Parameter(torch.zeros(2))
    frozen = nn.Parameter(torch.zeros(2), requires_grad=False)
    named = [("trained", trained), ("other", other)]
    check_param_group_coverage(
        [{"name": "a", "params": [trained]}, {"name": "b", "params": [other]}],
        named,
    )
    with pytest.raises(RuntimeError, match="without an optimizer group"):
        check_param_group_coverage([{"name": "a", "params": [trained]}], named)
    with pytest.raises(RuntimeError, match="two optimizer groups"):
        check_param_group_coverage(
            [
                {"name": "a", "params": [trained, other]},
                {"name": "b", "params": [other]},
            ],
            named,
        )
    with pytest.raises(RuntimeError, match="frozen parameter"):
        check_param_group_coverage(
            [{"name": "a", "params": [trained, other, frozen]}], named
        )


def test_hybrid_state_and_optimizer_groups_roundtrip(hybrid):
    # The optimizer step below updates parameters; keep the fixture intact.
    source = copy.deepcopy(hybrid.encoder)
    restored, *_ = _hybrid_encoder(seed=1)
    incompatible = restored.load_state_dict(source.state_dict(), strict=True)
    assert incompatible.missing_keys == [] and incompatible.unexpected_keys == []

    downstream_source = nn.Linear(3, 2)
    downstream_restored = nn.Linear(3, 2)
    source_groups = build_hybrid_param_groups(
        _named_roots(source), list(downstream_source.named_parameters()),
        _train_cfg(),
    )
    restored_groups = build_hybrid_param_groups(
        _named_roots(restored), list(downstream_restored.named_parameters()),
        _train_cfg(),
    )
    old = torch.optim.AdamW(source_groups, lr=5.0e-4)
    # One real step with a distinct constant gradient per slot, so every
    # exp_avg identifies the (group, position) it was saved from.
    slot = 0
    for group in old.param_groups:
        for parameter in group["params"]:
            slot += 1
            parameter.grad = torch.full_like(parameter, float(slot))
    old.step()
    new = torch.optim.AdamW(restored_groups, lr=5.0e-4)
    new.load_state_dict(old.state_dict())
    for index, group in enumerate(new.param_groups):
        source_group = old.param_groups[index]
        assert group["name"] == source_group["name"]
        assert group["lr"] == source_group["lr"]
        assert group["weight_decay"] == source_group["weight_decay"]
        for position, parameter in enumerate(group["params"]):
            saved = source_group["params"][position]
            assert parameter.shape == saved.shape
            torch.testing.assert_close(
                new.state[parameter]["exp_avg"], old.state[saved]["exp_avg"]
            )


def test_active_mode_keeps_the_backbone_in_eval(hybrid):
    encoder = hybrid.encoder
    roots = [module for _, module in _named_roots(encoder)]
    encoder.train()
    encoder.eval()
    set_active_adaptation_mode(roots, True)
    adapters = [
        module for module in encoder.modules()
        if isinstance(module, (NonlinearAdapter, LoRALinear, LoRASubMConv3d))
    ]
    assert adapters and all(module.training for module in adapters)
    assert not any(
        module.training for module in encoder.modules()
        if isinstance(module, (Block, MLP, nn.LayerNorm))
    )
    set_active_adaptation_mode(roots, False)
    assert not any(module.training for module in adapters)


def test_statistics_report_stem_terms_and_stage_ratios():
    torch.manual_seed(11)
    backbone = _stem_backbone(width=6)
    adapt_utonia_stem_to_hybrid(backbone, _settings())
    backbone.embedding.stem(torch.cat([torch.randn(5, 3), torch.rand(5, 1)], 1))
    backbone.enc = []
    stats = hybrid_adaptation_statistics(backbone, stage=-1)
    assert set(stats) == {"stem_xyz_rms", "stem_intensity_rms", "stem_adapter_rms"}
    assert stats["stem_xyz_rms"] > 0
    assert stats["stem_intensity_rms"] == 0 and stats["stem_adapter_rms"] == 0


@pytest.mark.parametrize(
    "override, message",
    [
        ({"mode": "lora"}, "mode must be"),
        ({"lora_targets": ["attention", "ffn"]}, "lora_targets"),
        ({"lora_targets": ["cpe", "cpe"]}, "lora_targets"),
        ({"train_layer_norm": "yes"}, "train_layer_norm"),
        (
            {"ffn_adapter": {
                "widths": [8, 0, 32, 64], "activation": "relu",
                "scale": 0.1, "dropout": 0.0,
            }},
            "widths",
        ),
        (
            {"ffn_adapter": {
                "widths": [8, 16, 32, 64], "activation": "tanh",
                "scale": 0.1, "dropout": 0.0,
            }},
            "activation",
        ),
        (
            {"ffn_adapter": {
                "widths": [8, 16, 32, 64], "activation": "relu",
                "scale": 0.0, "dropout": 0.0,
            }},
            "scale",
        ),
    ],
)
def test_settings_reject_invalid_hybrid_blocks(override, message):
    with pytest.raises(ValueError, match=message):
        _settings(**override)


def test_absent_block_keeps_all_linear_lora():
    assert resolve_utonia_adaptation_settings(OmegaConf.create({})) is None
