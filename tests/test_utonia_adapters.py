"""The hybrid Utonia adaptation: stem, FFN adapters, LN, LoRA scope, groups.

V14 puts the stem adapter before the stem LayerNorm; V15 puts it behind the
activation (``stem.adapter_position``) and can keep a learned scalar out of
weight decay (``train.downstream_no_decay``).

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

from src.models.utonia.adapters import (
    NonlinearAdapter,
    ParallelAdapterMLP,
    PostActivationStemAdapter,
    UtoniaHybridSettings,
    XYZIAdapterStem,
    XYZIStem,
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
from src.models.utonia.lora import (
    LoRALinear,
    LoRASubMConv3d,
    resolve_utonia_lora_settings,
)
from src.models.utonia.model import (
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


# --- V15: the stem adapter behind the activation ------------------------------

V14_STEM_STATE_KEYS = [
    "embedding.stem.linear.intensity_weight",
    "embedding.stem.linear.xyz.weight",
    "embedding.stem.linear.xyz.bias",
    "embedding.stem.linear.adapter.down.weight",
    "embedding.stem.linear.adapter.down.bias",
    "embedding.stem.linear.adapter.up.weight",
    "embedding.stem.linear.adapter.up.bias",
    "embedding.stem.norm.weight",
    "embedding.stem.norm.bias",
]
V15_STEM_STATE_KEYS = [
    "embedding.stem.linear.intensity_weight",
    "embedding.stem.linear.xyz.weight",
    "embedding.stem.linear.xyz.bias",
    "embedding.stem.norm.weight",
    "embedding.stem.norm.bias",
    "embedding.stem.act.adapter.down.weight",
    "embedding.stem.act.adapter.down.bias",
    "embedding.stem.act.adapter.up.weight",
    "embedding.stem.act.adapter.up.bias",
]


def _stem_block(position=None, **overrides):
    stem = {
        "train_intensity_column": True,
        "adapter_hidden": 16,
        "adapter_activation": "gelu",
        "adapter_scale": 1.0,
    }
    if position is not None:
        stem["adapter_position"] = position
    stem.update(overrides)
    return stem


def _post_activation_settings(**stem_overrides):
    return _settings(stem=_stem_block("post_activation", **stem_overrides))


@pytest.fixture(scope="module")
def hybrid_v15():
    encoder, report, ffn_count, trainable = _hybrid_encoder(
        settings=_post_activation_settings()
    )
    return SimpleNamespace(
        encoder=encoder, report=report, ffn_count=ffn_count,
        trainable=trainable,
    )


def test_stem_adapter_position_defaults_to_pre_norm_and_is_validated():
    assert _settings().stem_adapter_position == "pre_norm"
    assert _settings(
        stem=_stem_block("pre_norm")
    ).stem_adapter_position == "pre_norm"
    assert _post_activation_settings().stem_adapter_position == "post_activation"
    assert _settings(
        stem=_stem_block("Post_Activation")
    ).stem_adapter_position == "post_activation"
    with pytest.raises(ValueError, match="adapter_position"):
        _settings(stem=_stem_block("after_layer_norm"))
    # Constructions that predate the field keep working and mean V14's stem.
    settings = UtoniaHybridSettings(
        lora_targets=("attention",), train_layer_norm=True,
        train_intensity_column=True, stem_hidden=16, stem_activation="gelu",
        stem_scale=1.0, ffn_widths=(8, 16, 32, 64), ffn_activation="relu",
        ffn_scale=0.1, ffn_dropout=0.0,
    )
    assert settings.stem_adapter_position == "pre_norm"


def test_post_activation_stem_starts_as_the_pretrained_xyz_stem():
    torch.manual_seed(13)
    backbone = _stem_backbone()
    with torch.no_grad():
        backbone.embedding.stem.norm.weight.uniform_(0.5, 1.5)
        backbone.embedding.stem.norm.bias.uniform_(-0.2, 0.2)
    original = copy.deepcopy(backbone.embedding.stem)
    xyz = torch.randn(11, 3) * 4.0
    intensity = torch.rand(11, 1)
    expected = original(torch.cat([xyz, torch.zeros(11, 6)], dim=-1))

    old_width = adapt_utonia_stem_to_hybrid(backbone, _post_activation_settings())
    stem = backbone.embedding.stem
    actual = stem(torch.cat([xyz, intensity], dim=-1))

    assert old_width == 9
    assert backbone.embedding.in_channels == 4
    # w_I = 0 and a zero-output adapter: exactly the pretrained 9D stem.
    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)
    assert type(stem.linear) is XYZIStem
    torch.testing.assert_close(stem.linear.xyz.weight, original.linear.weight[:, :3])
    torch.testing.assert_close(stem.linear.xyz.bias, original.linear.bias)
    assert torch.count_nonzero(stem.linear.intensity_weight) == 0
    assert torch.count_nonzero(stem.act.adapter.up.weight) == 0
    assert torch.count_nonzero(stem.act.adapter.up.bias) == 0
    # The LayerNorm keeps its pretrained affine.
    torch.testing.assert_close(stem.norm.weight, original.norm.weight)
    torch.testing.assert_close(stem.norm.bias, original.norm.bias)


def test_post_activation_stem_module_tree_and_state_keys():
    backbone = _stem_backbone(width=7)
    original_norm = backbone.embedding.stem.norm
    original_act = backbone.embedding.stem.act
    adapt_utonia_stem_to_hybrid(backbone, _post_activation_settings())
    stem = backbone.embedding.stem

    assert list(stem._modules) == ["linear", "norm", "act"]
    assert type(stem.linear) is XYZIStem
    assert not isinstance(stem.linear, XYZIAdapterStem)
    assert not hasattr(stem.linear, "adapter")
    # No pre-LN adapter; the pretrained LayerNorm and GELU are the same objects.
    assert stem.norm is original_norm
    assert isinstance(stem.act, PostActivationStemAdapter)
    assert stem.act.activation is original_act
    adapter = stem.act.adapter
    assert isinstance(adapter, NonlinearAdapter)
    assert (adapter.down.in_features, adapter.down.out_features) == (7, 16)
    assert (adapter.up.in_features, adapter.up.out_features) == (16, 7)
    assert isinstance(adapter.act, nn.GELU) and adapter.scale == 1.0
    assert list(backbone.state_dict()) == V15_STEM_STATE_KEYS


def test_v14_stem_keeps_its_state_dict_layout():
    """pre_norm (V14/V14.1) is untouched: same keys, same order."""
    backbone = _stem_backbone(width=7)
    adapt_utonia_stem_to_hybrid(backbone, _settings())
    stem = backbone.embedding.stem
    assert type(stem.linear) is XYZIAdapterStem
    assert isinstance(stem.linear, XYZIStem)
    assert list(stem._modules) == ["linear", "norm", "act"]
    assert type(stem.act) is nn.GELU
    assert list(backbone.state_dict()) == V14_STEM_STATE_KEYS
    explicit = _stem_backbone(width=7)
    adapt_utonia_stem_to_hybrid(explicit, _settings(stem=_stem_block("pre_norm")))
    assert list(explicit.state_dict()) == list(backbone.state_dict())


def test_encoder_state_keys_pin_both_stem_layouts(hybrid, hybrid_v15):
    for fixture, expected in ((hybrid, V14_STEM_STATE_KEYS),
                              (hybrid_v15, V15_STEM_STATE_KEYS)):
        keys = [
            key for key in fixture.encoder.state_dict()
            if key.startswith("embedding.stem.")
        ]
        assert keys == expected
    # Outside the stem the two encoders are the same tree.
    other = [
        key for key in hybrid.encoder.state_dict()
        if not key.startswith("embedding.stem.")
    ]
    assert other == [
        key for key in hybrid_v15.encoder.state_dict()
        if not key.startswith("embedding.stem.")
    ]


def test_post_activation_adapter_adds_a_residual_to_the_activation():
    torch.manual_seed(17)
    backbone = _stem_backbone(width=7)
    adapt_utonia_stem_to_hybrid(backbone, _post_activation_settings())
    stem = backbone.embedding.stem
    with torch.no_grad():
        stem.linear.intensity_weight.normal_()
        stem.act.adapter.up.weight.normal_()
        stem.act.adapter.up.bias.normal_()
    feature = torch.cat([torch.randn(9, 3), torch.rand(9, 1)], dim=-1)

    actual = stem(feature)

    activated = torch.nn.functional.gelu(stem.norm(stem.linear(feature)))
    adapter = stem.act.adapter
    branch = adapter.up(torch.nn.functional.gelu(adapter.down(activated)))
    torch.testing.assert_close(actual, activated + branch)
    assert branch.abs().sum() > 0


def test_post_activation_stem_trains_adapter_intensity_and_norm_only():
    torch.manual_seed(19)
    backbone = _stem_backbone(width=7)
    settings = _post_activation_settings()
    adapt_utonia_stem_to_hybrid(backbone, settings)
    freeze_utonia_and_enable_hybrid(backbone, [backbone.embedding], settings)
    stem = backbone.embedding.stem
    assert stem.linear.intensity_weight.requires_grad
    assert not stem.linear.xyz.weight.requires_grad
    assert not stem.linear.xyz.bias.requires_grad
    assert stem.norm.weight.requires_grad and stem.norm.bias.requires_grad
    assert all(p.requires_grad for p in stem.act.adapter.parameters())
    feature = torch.cat([torch.randn(16, 3), torch.rand(16, 1)], dim=-1)

    stem(feature).square().sum().backward()

    assert stem.linear.xyz.weight.grad is None and stem.linear.xyz.bias.grad is None
    assert stem.linear.intensity_weight.grad.abs().sum() > 0
    # Zero up-projection: up receives a gradient, down's is exactly zero.
    assert stem.act.adapter.up.weight.grad.abs().sum() > 0
    assert torch.count_nonzero(stem.act.adapter.down.weight.grad) == 0

    optimizer = torch.optim.SGD(
        [p for p in backbone.parameters() if p.requires_grad], lr=0.1
    )
    optimizer.step()
    optimizer.zero_grad()
    stem(feature).square().sum().backward()
    assert stem.act.adapter.down.weight.grad.abs().sum() > 0


def test_post_activation_adapter_lands_in_the_adapter_roles(hybrid_v15):
    encoder = hybrid_v15.encoder
    roles = utonia_parameter_roles(_named_roots(encoder))
    weights = [name for name, _ in roles["adapter_weight"]]
    biases = [name for name, _ in roles["adapter_bias"]]
    for layer in ("down", "up"):
        assert f"embedding.stem.act.adapter.{layer}.weight" in weights
        assert f"embedding.stem.act.adapter.{layer}.bias" in biases
    assert [name for name, _ in roles["intensity"]] == [
        "embedding.stem.linear.intensity_weight"
    ]
    # No stem parameter is left without a role except the frozen XYZ Linear
    # and the LayerNorm, whose affine is the layer_norm role.
    layer_norms = [name for name, _ in roles["layer_norm"]]
    assert "embedding.stem.norm.weight" in layer_norms
    trainable_stem = {
        name for name, parameter in encoder.named_parameters()
        if name.startswith("embedding.stem.") and parameter.requires_grad
    }
    assigned = {
        name for items in roles.values() for name, _ in items
        if name.startswith("embedding.stem.")
    }
    assert trainable_stem == assigned
    assert trainable_stem.isdisjoint({
        "embedding.stem.linear.xyz.weight", "embedding.stem.linear.xyz.bias",
    })


def test_post_activation_optimizer_groups_follow_the_role_table(hybrid_v15):
    encoder = hybrid_v15.encoder
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
    act_adapter = encoder.embedding.stem.act.adapter
    weight_ids = {id(p) for p in by_name["utonia_adapter_weight"]["params"]}
    bias_ids = {id(p) for p in by_name["utonia_adapter_bias"]["params"]}
    assert {id(act_adapter.down.weight), id(act_adapter.up.weight)} <= weight_ids
    assert {id(act_adapter.down.bias), id(act_adapter.up.bias)} <= bias_ids
    assert by_name["utonia_adapter_bias"]["weight_decay"] == 0.0
    assert by_name["utonia_adapter_weight"]["lr"] == pytest.approx(1.0e-3)
    assert [id(p) for p in by_name["utonia_intensity"]["params"]] == [
        id(encoder.embedding.stem.linear.intensity_weight)
    ]


def test_post_activation_adapter_follows_the_active_adaptation_mode(hybrid_v15):
    encoder = hybrid_v15.encoder
    roots = [module for _, module in _named_roots(encoder)]
    encoder.train()
    encoder.eval()
    set_active_adaptation_mode(roots, True)
    stem = encoder.embedding.stem
    assert stem.act.adapter.training
    # The wrapper and the pretrained activation are frozen modules: eval.
    assert not stem.act.training and not stem.act.activation.training
    assert not stem.norm.training
    set_active_adaptation_mode(roots, False)
    assert not stem.act.adapter.training


def test_post_activation_statistics_report_the_activation_and_adapter():
    torch.manual_seed(23)
    backbone = _stem_backbone(width=6)
    adapt_utonia_stem_to_hybrid(backbone, _post_activation_settings())
    backbone.embedding.stem(
        torch.cat([torch.randn(5, 3), torch.rand(5, 1)], 1)
    )
    backbone.enc = []
    stats = hybrid_adaptation_statistics(backbone, stage=-1)
    assert list(stats) == [
        "stem_xyz_rms", "stem_intensity_rms", "stem_activation_rms",
        "stem_adapter_rms",
    ]
    assert all(value.ndim == 0 for value in stats.values())
    assert stats["stem_xyz_rms"] > 0 and stats["stem_activation_rms"] > 0
    assert stats["stem_intensity_rms"] == 0 and stats["stem_adapter_rms"] == 0

    stem = backbone.embedding.stem
    with torch.no_grad():
        stem.linear.intensity_weight.fill_(0.5)
        stem.act.adapter.up.bias.fill_(2.0)
    backbone.embedding.stem(torch.cat([torch.randn(5, 3), torch.rand(5, 1)], 1))
    stats = hybrid_adaptation_statistics(backbone, stage=-1)
    assert stats["stem_intensity_rms"] > 0
    # up's zero weight leaves only the bias: a constant 2.0 adapter output.
    torch.testing.assert_close(stats["stem_adapter_rms"], torch.tensor(2.0))

    # pre_norm keeps its three pre-LN keys and has no activation statistic.
    legacy = _stem_backbone(width=6)
    adapt_utonia_stem_to_hybrid(legacy, _settings())
    legacy.embedding.stem(torch.cat([torch.randn(5, 3), torch.rand(5, 1)], 1))
    legacy.enc = []
    assert set(hybrid_adaptation_statistics(legacy, stage=-1)) == {
        "stem_xyz_rms", "stem_intensity_rms", "stem_adapter_rms",
    }


def test_post_activation_needs_the_pretrained_norm_and_activation():
    bare = nn.Module()
    bare.embedding = Embedding(in_channels=9, embed_channels=7)
    assert list(bare.embedding.stem._modules) == ["linear"]
    with pytest.raises(ValueError, match="LayerNorm and activation"):
        adapt_utonia_stem_to_hybrid(bare, _post_activation_settings())
    # The stem is left untouched when the adaptation refuses.
    assert type(bare.embedding.stem.linear) is nn.Linear
    assert bare.embedding.in_channels == 9


def test_v15_scope_matches_the_spec_census(hybrid, hybrid_v15):
    encoder = hybrid_v15.encoder
    roles = utonia_parameter_roles(_named_roots(encoder))
    counts = {
        role: sum(parameter.numel() for _, parameter in items)
        for role, items in roles.items()
    }
    stem_adapter = encoder.embedding.stem.act.adapter
    # 54*16 + 16 down, 16*54 + 54 up.
    assert sum(p.numel() for p in stem_adapter.parameters()) == 1_798
    assert hybrid_v15.trainable == 4_423_930
    assert hybrid_v15.trainable - hybrid.trainable == 1_798 - 998
    assert counts["intensity"] == 54
    assert counts["lora"] == 3_657_312
    assert counts["layer_norm"] == 39_528
    # 21 FFN adapters (725,238) plus the 54 -> 16 -> 54 stem adapter (1,798).
    assert counts["adapter_weight"] + counts["adapter_bias"] == 725_238 + 1_798
    assert hybrid_v15.ffn_count == 21
    assert hybrid_v15.report.linear_modules == 66
    assert hybrid_v15.report.conv_modules == 21
    layer_norms = [
        module for _, root in _named_roots(encoder)
        for module in root.modules() if isinstance(module, nn.LayerNorm)
    ]
    assert len(layer_norms) == 67
    assert all(
        parameter.requires_grad
        for module in layer_norms for parameter in module.parameters()
    )
    # The scope's frozen parts are V14's: XYZ stem, stage 4, the mask token.
    stem = encoder.embedding.stem
    assert type(stem.linear) is XYZIStem and not stem.linear.xyz.weight.requires_grad
    assert not any(p.requires_grad for p in encoder.enc[4].parameters())
    assert not encoder.embedding.mask_token.requires_grad
    assert not any(
        isinstance(module, XYZIAdapterStem) for module in encoder.modules()
    )


# --- train.downstream_no_decay --------------------------------------------------


def _downstream_root():
    temporal = nn.Module()
    temporal.v_raw = nn.Parameter(torch.tensor(29.9))
    temporal.proj = nn.Linear(3, 2)
    backend = nn.Module()
    backend.temporal = temporal
    backend.head = nn.Linear(2, 2)
    root = nn.Module()
    root.dynamic_backend = backend
    return root


def _tiny_hybrid_roots(position="post_activation"):
    torch.manual_seed(29)
    backbone = _stem_backbone(width=7)
    settings = _settings(stem=_stem_block(position))
    adapt_utonia_stem_to_hybrid(backbone, settings)
    freeze_utonia_and_enable_hybrid(backbone, [backbone.embedding], settings)
    return backbone, [("embedding", backbone.embedding)]


def _train_cfg_with(**overrides):
    cfg = _train_cfg()
    for key, value in overrides.items():
        cfg[key] = value
    return cfg


def _group_summary(groups):
    return [
        (
            group["name"], group["lr"], group["weight_decay"],
            [id(parameter) for parameter in group["params"]],
        )
        for group in groups
    ]


def test_downstream_no_decay_moves_the_named_parameters_to_a_last_group():
    backbone, roots = _tiny_hybrid_roots()
    downstream = _downstream_root()
    named = list(downstream.named_parameters())
    cfg = _train_cfg_with(downstream_no_decay=["dynamic_backend.temporal.v_raw"])

    groups = build_hybrid_param_groups(roots, named, cfg)

    plain = build_hybrid_param_groups(roots, named, _train_cfg())
    assert [group["name"] for group in groups] == [
        *[group["name"] for group in plain], "downstream_no_decay",
    ]
    by_name = {group["name"]: group for group in groups}
    v_raw = downstream.dynamic_backend.temporal.v_raw
    assert [id(p) for p in by_name["downstream_no_decay"]["params"]] == [id(v_raw)]
    assert by_name["downstream_no_decay"]["lr"] == pytest.approx(5.0e-4)
    assert by_name["downstream_no_decay"]["weight_decay"] == 0.0
    # It left the downstream group, which keeps everything else, decayed.
    assert id(v_raw) not in {id(p) for p in by_name["downstream"]["params"]}
    assert len(by_name["downstream"]["params"]) == len(named) - 1
    assert by_name["downstream"]["weight_decay"] == pytest.approx(1.0e-4)
    assert by_name["downstream"]["lr"] == pytest.approx(5.0e-4)
    # Every Utonia group is exactly what it was.
    assert _group_summary(groups[:-2]) == _group_summary(plain[:-1])

    trainable = [
        (f"encoder.{name}", parameter)
        for name, parameter in backbone.named_parameters()
        if parameter.requires_grad
    ] + named
    check_param_group_coverage(groups, trainable)


def test_downstream_no_decay_group_takes_several_names_in_module_order():
    _backbone, roots = _tiny_hybrid_roots()
    downstream = _downstream_root()
    named = list(downstream.named_parameters())
    cfg = _train_cfg_with(downstream_no_decay=[
        "dynamic_backend.head.bias", "dynamic_backend.temporal.v_raw",
    ])
    groups = build_hybrid_param_groups(roots, named, cfg)
    assert groups[-1]["name"] == "downstream_no_decay"
    # Module order, not the order the config lists them in.
    assert [id(p) for p in groups[-1]["params"]] == [
        id(downstream.dynamic_backend.temporal.v_raw),
        id(downstream.dynamic_backend.head.bias),
    ]


@pytest.mark.parametrize("value", ["absent", None, []])
def test_absent_or_empty_downstream_no_decay_keeps_todays_groups(value):
    _backbone, roots = _tiny_hybrid_roots()
    downstream = _downstream_root()
    named = list(downstream.named_parameters())
    cfg = _train_cfg() if value == "absent" else _train_cfg_with(
        downstream_no_decay=value
    )
    groups = build_hybrid_param_groups(roots, named, cfg)
    assert [group["name"] for group in groups] == [
        "utonia_intensity", "utonia_adapter_weight", "utonia_adapter_bias",
        "utonia_layer_norm", "downstream",
    ]
    assert groups[-1]["weight_decay"] == pytest.approx(1.0e-4)
    assert [id(p) for p in groups[-1]["params"]] == [id(p) for _, p in named]
    reference = build_hybrid_param_groups(roots, named, _train_cfg())
    assert _group_summary(groups) == _group_summary(reference)


def test_downstream_no_decay_rejects_unknown_frozen_and_repeated_names():
    _backbone, roots = _tiny_hybrid_roots()
    downstream = _downstream_root()
    downstream.dynamic_backend.head.bias.requires_grad = False
    named = list(downstream.named_parameters())

    def build(names):
        return build_hybrid_param_groups(
            roots, named, _train_cfg_with(downstream_no_decay=names)
        )

    with pytest.raises(ValueError, match="unknown: dynamic_backend.temporal.v_rawx"):
        build(["dynamic_backend.temporal.v_raw", "dynamic_backend.temporal.v_rawx"])
    with pytest.raises(ValueError, match="frozen: dynamic_backend.head.bias"):
        build(["dynamic_backend.head.bias"])
    # Both problems are listed at once.
    with pytest.raises(ValueError) as error:
        build(["nope", "dynamic_backend.head.bias"])
    assert "unknown: nope" in str(error.value)
    assert "frozen: dynamic_backend.head.bias" in str(error.value)
    with pytest.raises(ValueError, match="repeat"):
        build(["dynamic_backend.temporal.v_raw", "dynamic_backend.temporal.v_raw"])
    # A Utonia parameter is not a downstream one.
    with pytest.raises(ValueError, match="unknown"):
        build(["feature_extractor.embedding.stem.linear.intensity_weight"])
    # A bare string is a mistake, not a list of one-letter names.
    with pytest.raises(ValueError, match="must be a list"):
        build("dynamic_backend.temporal.v_raw")


def test_adamw_leaves_the_no_decay_parameter_alone_and_decays_the_rest():
    _backbone, roots = _tiny_hybrid_roots()

    def step(no_decay):
        downstream = _downstream_root()
        cfg = _train_cfg_with(weight_decay=1.0e-2)
        if no_decay:
            cfg["downstream_no_decay"] = ["dynamic_backend.temporal.v_raw"]
        groups = build_hybrid_param_groups(
            roots, list(downstream.named_parameters()), cfg
        )
        optimizer = torch.optim.AdamW(groups, lr=5.0e-4, weight_decay=1.0e-2)
        for parameter in downstream.parameters():
            parameter.grad = torch.zeros_like(parameter)
        v_raw = downstream.dynamic_backend.temporal.v_raw
        before = v_raw.detach().clone()
        proj_before = downstream.dynamic_backend.temporal.proj.weight.detach().clone()
        optimizer.step()
        return (
            v_raw.detach() - before,
            downstream.dynamic_backend.temporal.proj.weight.detach() - proj_before,
            optimizer,
        )

    # Zero gradient: only AdamW's decoupled decay can move a parameter.
    decayed_v, decayed_proj, _ = step(no_decay=False)
    kept_v, kept_proj, optimizer = step(no_decay=True)
    # lr * weight_decay * p, to within float32's spacing at 29.9 (1.9e-6).
    assert float(decayed_v) == pytest.approx(-29.9 * 5.0e-4 * 1.0e-2, abs=2.0e-6)
    assert float(kept_v) == 0.0
    # Everything that stayed downstream decays exactly as before.
    torch.testing.assert_close(kept_proj, decayed_proj)
    assert kept_proj.abs().sum() > 0
    assert [group["name"] for group in optimizer.param_groups][-2:] == [
        "downstream", "downstream_no_decay",
    ]


def test_v15_state_and_optimizer_groups_roundtrip(hybrid_v15):
    """A V15 checkpoint restores strictly, and its groups map back position by position."""
    source = copy.deepcopy(hybrid_v15.encoder)
    restored, *_ = _hybrid_encoder(seed=1, settings=_post_activation_settings())
    incompatible = restored.load_state_dict(source.state_dict(), strict=True)
    assert incompatible.missing_keys == [] and incompatible.unexpected_keys == []

    def downstream():
        module = nn.ModuleDict({"dynamic_backend": nn.ModuleDict({
            "temporal": nn.ModuleDict({"proj": nn.Linear(3, 2)}),
        })})
        module["dynamic_backend"]["temporal"].v_raw = nn.Parameter(
            torch.tensor(30.0)
        )
        return module

    cfg = _train_cfg_with(downstream_no_decay=["dynamic_backend.temporal.v_raw"])
    source_downstream, restored_downstream = downstream(), downstream()
    source_groups = build_hybrid_param_groups(
        _named_roots(source), list(source_downstream.named_parameters()), cfg
    )
    restored_groups = build_hybrid_param_groups(
        _named_roots(restored), list(restored_downstream.named_parameters()), cfg
    )
    assert [group["name"] for group in source_groups] == [
        "utonia_intensity", "utonia_adapter_weight", "utonia_adapter_bias",
        "utonia_lora", "utonia_layer_norm", "downstream", "downstream_no_decay",
    ]
    old = torch.optim.AdamW(source_groups, lr=5.0e-4)
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
    assert new.param_groups[-1]["weight_decay"] == 0.0
