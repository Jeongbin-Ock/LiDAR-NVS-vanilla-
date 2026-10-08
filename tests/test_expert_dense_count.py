"""V15.2's per-option expert head and V15.3's router-free dense head."""
from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from src.models_new.module.dynamic_gaussian import (
    DenseCountBarrierVelocityGaussianBackend,
    DenseCountGaussianHead,
    NestedCountGaussianHead,
    OptionExpertBarrierVelocityGaussianBackend,
    OptionExpertGaussianHead,
)
from src.models_new.utils.render import Gaussianutil

DIM = 24
PARAM_DIM = 42
REDUCE, KEEP, ADD = 0, 1, 2
COUNTS = (1, 3, 5)
EXPERT_START = (0, 1, 4)
HEAD_CFG = {"initial_opacity": 0.2, "initial_scale_m": 0.3}


def _gs_params():
    return OmegaConf.create({
        "shs": 32, "opacity": 1, "scaling": 2, "rotation": 4, "offset": 3,
    })


def _routed_block(**overrides):
    block = {
        "option_counts": [1, 3, 5],
        "seed_mode": "token",
        "tau": 1.0,
        "decoder_hidden": 32,
        "router_hidden": [16, 8],
    }
    block.update(overrides)
    return block


def expert_count_cfg(**overrides):
    overrides.setdefault("auxiliary_render", "dense_add")
    return OmegaConf.create({
        "count_mode": "expert_gumbel",
        "expert_count": _routed_block(**overrides),
    })


def _expert_head(**overrides):
    return OptionExpertGaussianHead(
        expert_count_cfg(**overrides),
        OmegaConf.create(HEAD_CFG),
        _gs_params(),
        DIM,
        offset_bound=0.4,
    )


def _dense_head(gaussians_per_token=5, decoder_hidden=32):
    return DenseCountGaussianHead(
        OmegaConf.create({**HEAD_CFG, "decoder_hidden": decoder_hidden}),
        _gs_params(),
        DIM,
        offset_bound=0.4,
        gaussians_per_token=gaussians_per_token,
    )


def _force_choices(head, choices):
    """Pin the hard choice while keeping the soft straight-through backward."""
    chosen = torch.as_tensor(choices, dtype=torch.long)

    def select(logits):
        soft = torch.softmax(logits.float(), dim=-1)
        hard = F.one_hot(chosen, num_classes=logits.shape[-1]).to(soft.dtype)
        selection = hard - soft.detach() + soft
        selection.retain_grad()
        head.captured_selection = selection
        return selection

    head._select = select


def _randomize(module):
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.normal_(0.0, 0.2)


def _render_opacity(packed):
    return Gaussianutil(None).get_opacity(
        packed["raw"]["opacity"], opacity_gate=packed.get("opacity_gate"),
    )


def _expert_outputs(head, feature, option):
    hidden = head.trunk(head.decoder_norm(feature))
    return head.experts[option](hidden).reshape(
        feature.shape[0], COUNTS[option], PARAM_DIM
    )


# ---------------------------------------------------------------- V15.2 head


def test_expert_head_layout_and_initial_priors():
    torch.manual_seed(0)
    head = _expert_head().eval()
    names = [name for name, _ in head.named_parameters()]
    assert names == [
        "decoder_norm.weight", "decoder_norm.bias",
        "trunk.0.weight", "trunk.0.bias",
        "experts.0.weight", "experts.0.bias",
        "experts.1.weight", "experts.1.bias",
        "experts.2.weight", "experts.2.bias",
        "router_norm.weight", "router_norm.bias",
        "router.0.weight", "router.0.bias",
        "router.2.weight", "router.2.bias",
        "router.4.weight", "router.4.bias",
    ]
    assert tuple(head.trunk[0].weight.shape) == (32, DIM)
    assert [tuple(expert.weight.shape) for expert in head.experts] == [
        (count * PARAM_DIM, 32) for count in COUNTS
    ]
    assert head.router_norm is not head.decoder_norm
    assert head.auxiliary_count == 5
    for expert, count in zip(head.experts, COUNTS):
        bias = expert.bias.reshape(count, PARAM_DIM)
        torch.testing.assert_close(
            bias[:, 32], torch.full((count,), math.log(0.2 / 0.8))
        )
        torch.testing.assert_close(
            bias[:, 33:35], torch.full((count, 2), math.log(math.expm1(0.3)))
        )
        torch.testing.assert_close(bias[:, 35], torch.ones(count))
        torch.testing.assert_close(bias[:, 39:42], torch.zeros(count, 3))
        weight = expert.weight.reshape(count, PARAM_DIM, -1)
        assert torch.count_nonzero(weight[:, 39:42]) == 0

    feature = torch.randn(7, DIM)
    anchor = torch.randn(7, 3)
    routed, auxiliary, routing = head(feature, anchor)
    assert auxiliary is None
    torch.testing.assert_close(routing["logits"], torch.zeros(7, 3))
    torch.testing.assert_close(routed["position"], anchor[routed["token_index"]])


def test_routed_set_is_each_tokens_chosen_expert():
    torch.manual_seed(1)
    head = _expert_head().train()
    _randomize(head)
    choices = [ADD, REDUCE, KEEP, KEEP, ADD]
    _force_choices(head, choices)
    feature = torch.randn(5, DIM)
    anchor = torch.randn(5, 3)
    routed, _auxiliary, routing = head(feature, anchor)

    counts = [COUNTS[choice] for choice in choices]
    assert routing["count"].tolist() == counts
    assert routing["option"].tolist() == choices
    expected_token = torch.cat([
        torch.full((count,), token) for token, count in enumerate(counts)
    ])
    expected_slot = torch.cat([torch.arange(count) for count in counts])
    assert torch.equal(routed["token_index"], expected_token)
    # Slots count within the chosen expert, so every token starts at slot 0.
    assert torch.equal(routed["slot_index"], expected_slot)

    rows = torch.cat([
        _expert_outputs(head, feature[token:token + 1], choice)[0]
        for token, choice in enumerate(choices)
    ])
    torch.testing.assert_close(routed["raw"]["opacity"], rows[:, 32:33])
    torch.testing.assert_close(routed["raw"]["shs"], rows[:, :32])
    torch.testing.assert_close(
        routed["position"],
        anchor[expected_token] + 0.4 * torch.tanh(rows[:, 39:42]),
    )
    # Every packed Gaussian is gated, G1-equivalents included; forward is one.
    assert torch.equal(
        routed["opacity_gate"], torch.ones_like(routed["raw"]["opacity"])
    )
    assert routed["opacity_gate"].requires_grad
    assert torch.equal(
        _render_opacity(routed), routed["raw"]["opacity"].sigmoid()
    )


def test_every_choice_sends_the_router_a_gradient():
    """Reduce tokens reach the router too: g_k = sum over E_k of o * dL/do'."""
    torch.manual_seed(2)
    head = _expert_head().train()
    _randomize(head)
    _force_choices(head, [REDUCE, KEEP, ADD])
    feature = torch.randn(3, DIM)
    routed, _auxiliary, _routing = head(feature, torch.randn(3, 3))
    activated = _render_opacity(routed).squeeze(-1)
    activated.sum().backward()  # dL/do' = 1 on every rendered Gaussian.

    selection = head.captured_selection
    opacity = routed["raw"]["opacity"].detach().squeeze(-1).sigmoid()
    for token, choice in enumerate([REDUCE, KEEP, ADD]):
        rows = routed["token_index"] == token
        expected = torch.zeros(3)
        expected[choice] = opacity[rows].sum()
        torch.testing.assert_close(selection.grad[token], expected)
    # Through the softmax the router's parameters move, reduce token included.
    router_bias = head.router[-1].bias
    assert router_bias.grad is not None and router_bias.grad.abs().sum() > 0


def test_router_logit_gradient_is_the_softmax_jacobian_of_the_chosen_gate():
    torch.manual_seed(3)
    head = _expert_head().train()
    _randomize(head)
    _force_choices(head, [REDUCE, ADD])
    captured = {}

    def hook(_module, _inputs, output):
        output.retain_grad()
        captured["logits"] = output

    handle = head.router.register_forward_hook(hook)
    try:
        routed, _auxiliary, _routing = head(torch.randn(2, DIM), torch.randn(2, 3))
    finally:
        handle.remove()
    _render_opacity(routed).sum().backward()
    logits = captured["logits"]
    probability = torch.softmax(logits.detach(), dim=-1)
    opacity = routed["raw"]["opacity"].detach().squeeze(-1).sigmoid()
    for token, choice in enumerate([REDUCE, ADD]):
        gate_gradient = opacity[routed["token_index"] == token].sum()
        one_hot = F.one_hot(torch.tensor(choice), 3).float()
        expected = gate_gradient * probability[token, choice] * (
            one_hot - probability[token]
        )
        torch.testing.assert_close(logits.grad[token], expected)
        assert torch.count_nonzero(logits.grad[token]) == 3


def test_nested_v15_head_sends_no_gradient_from_reducing_tokens():
    """The V15 behaviour V15.2 fixes, measured on the same forced choices."""
    torch.manual_seed(4)
    nested = NestedCountGaussianHead(
        OmegaConf.create({
            "count_mode": "nested_gumbel",
            "nested_count": _routed_block(auxiliary_render="dense_keep"),
        }),
        OmegaConf.create(HEAD_CFG), _gs_params(), DIM, offset_bound=0.4,
    ).train()
    _randomize(nested)
    _force_choices(nested, [REDUCE, KEEP])
    routed, _auxiliary, _routing = nested(torch.randn(2, DIM), torch.randn(2, 3))
    _render_opacity(routed).sum().backward()
    assert torch.count_nonzero(nested.captured_selection.grad[0]) == 0

    expert = _expert_head().train()
    _randomize(expert)
    _force_choices(expert, [REDUCE, KEEP])
    routed, _auxiliary, _routing = expert(torch.randn(2, DIM), torch.randn(2, 3))
    _render_opacity(routed).sum().backward()
    assert torch.count_nonzero(expert.captured_selection.grad[0]) > 0


@pytest.mark.parametrize("raw_value", [-14.0, 17.0, 100.0])
def test_saturated_expert_opacity_keeps_the_gate_gradient(raw_value):
    head = _expert_head().train()
    _force_choices(head, [REDUCE, KEEP])
    with torch.no_grad():
        for expert, count in zip(head.experts, COUNTS):
            expert.weight.reshape(count, PARAM_DIM, -1)[:, 32].zero_()
            expert.bias.reshape(count, PARAM_DIM)[:, 32] = raw_value
    routed, _auxiliary, _routing = head(torch.randn(2, DIM), torch.zeros(2, 3))
    activated = _render_opacity(routed)
    assert torch.equal(activated, routed["raw"]["opacity"].sigmoid())
    activated.sum().backward()
    opacity = torch.tensor(raw_value).sigmoid()
    gradient = head.captured_selection.grad
    torch.testing.assert_close(gradient[0, REDUCE], opacity)
    torch.testing.assert_close(gradient[1, KEEP], 3 * opacity)
    assert torch.isfinite(gradient).all()


def test_unchosen_experts_receive_no_gradient():
    torch.manual_seed(5)
    head = _expert_head().train()
    _randomize(head)
    _force_choices(head, [REDUCE, REDUCE, REDUCE])
    head.zero_grad(set_to_none=True)
    routed, _auxiliary, _routing = head(torch.randn(3, DIM), torch.randn(3, 3))
    (routed["raw"]["shs"].sum() + routed["position"].sum()).backward()
    assert head.experts[REDUCE].weight.grad.abs().sum() > 0
    assert head.trunk[0].weight.grad.abs().sum() > 0
    for option in (KEEP, ADD):
        grad = head.experts[option].weight.grad
        assert grad is None or torch.count_nonzero(grad) == 0


def test_auxiliary_set_is_every_tokens_add_expert_and_ignores_the_router():
    torch.manual_seed(6)
    head = _expert_head().train()
    _randomize(head)
    _force_choices(head, [REDUCE, KEEP, ADD])
    feature = torch.randn(3, DIM)
    anchor = torch.randn(3, 3)
    head.zero_grad(set_to_none=True)
    _routed, auxiliary, _routing = head(feature, anchor)
    assert torch.equal(auxiliary["token_index"], torch.arange(3).repeat_interleave(5))
    assert torch.equal(auxiliary["slot_index"], torch.arange(5).repeat(3))
    rows = _expert_outputs(head, feature, ADD).reshape(15, PARAM_DIM)
    torch.testing.assert_close(auxiliary["raw"]["opacity"], rows[:, 32:33])
    assert "opacity_gate" not in auxiliary
    _render_opacity(auxiliary).sum().backward()
    for parameter in head.router.parameters():
        assert parameter.grad is None or torch.count_nonzero(parameter.grad) == 0
    assert head.experts[ADD].weight.grad.abs().sum() > 0
    for option in (REDUCE, KEEP):
        grad = head.experts[option].weight.grad
        assert grad is None or torch.count_nonzero(grad) == 0

    evaluated, auxiliary, _routing = head.eval()(feature, anchor)
    assert auxiliary is None and "opacity_gate" not in evaluated
    silent = _expert_head(auxiliary_render="none").train()
    assert silent(feature, anchor)[1] is None


def test_expert_head_evaluates_the_argmax_expert():
    torch.manual_seed(7)
    head = _expert_head().eval()
    _randomize(head)
    feature = torch.randn(11, DIM)
    routed, _auxiliary, routing = head(feature, torch.randn(11, 3))
    choice = routing["logits"].argmax(dim=-1)
    assert torch.equal(routing["option"], choice)
    assert torch.equal(routing["count"], torch.tensor(COUNTS)[choice])
    for token in range(11):
        rows = routed["token_index"] == token
        torch.testing.assert_close(
            routed["raw"]["opacity"][rows],
            _expert_outputs(head, feature[token:token + 1], int(choice[token]))[0][:, 32:33],
        )


@pytest.mark.parametrize("overrides, message", [
    ({"option_counts": [1, 2, 3]}, "option_counts"),
    ({"seed_mode": "range_quantile"}, "seed_mode"),
    ({"tau": 0.0}, "tau"),
    ({"decoder_hidden": 0}, "decoder_hidden"),
    ({"router_hidden": []}, "router_hidden"),
    ({"auxiliary_render": "dense_keep"}, "auxiliary_render"),
])
def test_expert_head_rejects_an_invalid_contract(overrides, message):
    with pytest.raises(ValueError, match=message):
        _expert_head(**overrides)


def test_expert_head_reads_only_its_own_count_block():
    with pytest.raises(ValueError, match="needs p2g.grid_query.expert_count"):
        OptionExpertGaussianHead(
            OmegaConf.create({
                "count_mode": "nested_gumbel",
                "nested_count": _routed_block(auxiliary_render="dense_keep"),
            }),
            OmegaConf.create(HEAD_CFG), _gs_params(), DIM, offset_bound=0.4,
        )


# ---------------------------------------------------------------- V15.3 head


def test_dense_head_renders_every_slot_of_every_token():
    torch.manual_seed(8)
    head = _dense_head()
    _randomize(head)
    assert [name for name, _ in head.named_parameters()] == [
        "decoder_norm.weight", "decoder_norm.bias",
        "decoder.0.weight", "decoder.0.bias",
        "decoder.2.weight", "decoder.2.bias",
    ]
    feature = torch.randn(4, DIM)
    anchor = torch.randn(4, 3)
    for mode in (head.train(), head.eval()):
        rendered, auxiliary, routing = mode(feature, anchor)
        assert auxiliary is None and routing is None
        assert "opacity_gate" not in rendered
        assert torch.equal(rendered["token_index"], torch.arange(4).repeat_interleave(5))
        assert torch.equal(rendered["slot_index"], torch.arange(5).repeat(4))
        rows = head.decoder(head.decoder_norm(feature)).reshape(20, PARAM_DIM)
        torch.testing.assert_close(rendered["raw"]["opacity"], rows[:, 32:33])
        torch.testing.assert_close(rendered["raw"]["rotation"], rows[:, 35:39])
        torch.testing.assert_close(
            rendered["position"],
            anchor.repeat_interleave(5, dim=0) + 0.4 * torch.tanh(rows[:, 39:42]),
        )


def test_dense_head_is_the_v15_decoder_without_the_router():
    """Same initialization, and V15's G1..G5 are exactly V15.3's five slots."""
    nested = NestedCountGaussianHead(
        OmegaConf.create({
            "count_mode": "nested_gumbel",
            "nested_count": _routed_block(auxiliary_render="dense_keep"),
        }),
        OmegaConf.create(HEAD_CFG), _gs_params(), DIM, offset_bound=0.4,
    ).train()
    dense = _dense_head()
    torch.testing.assert_close(
        dense.decoder[-1].bias, nested.decoder[-1].bias
    )
    _randomize(nested)
    missing, unexpected = dense.load_state_dict(
        nested.state_dict(), strict=False
    )
    assert missing == []
    assert all(name.startswith("router") for name in unexpected)
    _force_choices(nested, [ADD, ADD, ADD])
    feature = torch.randn(3, DIM)
    anchor = torch.randn(3, 3)
    routed, _auxiliary, _routing = nested(feature, anchor)
    rendered, _auxiliary, _routing = dense(feature, anchor)
    for name in ("token_index", "slot_index", "position"):
        torch.testing.assert_close(rendered[name], routed[name])
    for name, value in routed["raw"].items():
        torch.testing.assert_close(rendered["raw"][name], value)


@pytest.mark.parametrize("decoder_hidden", [None, 0, 576.0, True])
def test_dense_head_requires_an_integer_decoder_width(decoder_hidden):
    cfg = dict(HEAD_CFG)
    if decoder_hidden is not None:
        cfg["decoder_hidden"] = decoder_hidden
    with pytest.raises(ValueError, match="decoder_hidden"):
        DenseCountGaussianHead(
            OmegaConf.create(cfg), _gs_params(), DIM, offset_bound=0.4,
            gaussians_per_token=5,
        )


# ---------------------------------------------------------------- backends


def _backend_cfg(heads=4, decoder_hidden=None):
    head = dict(HEAD_CFG)
    if decoder_hidden is not None:
        head["decoder_hidden"] = decoder_hidden
    return OmegaConf.create({
        "temporal": {
            "implementation": "auto",
            "layers": 2,
            "num_heads": heads,
            "mlp_ratio": 2,
            "use_time_embedding": True,
            "time_embedding_dim": 8,
            "time_frequencies": 4,
            "time_reference_sec": 1.0,
            "position_encoding": "barrier_rope_split",
            "barrier_speed_mps": 30.0,
            "barrier_weight": 1.0,
            "barrier_exponent": 1,
            "match_chunk_size": 4,
            "rope_base": 100.0,
            "rope_position_scale": 2.0 * math.pi / 5.0,
            "qk_norm": True,
            "layer_scale": False,
        },
        "local_attention": {
            "radius_m": 0.8,
            "num_heads": heads,
            "mlp_ratio": 2,
            "rope_base": 10.0 ** 1.2,
            "rope_position_scale": 2.0 * math.pi / 0.4,
            "implementation": "auto",
            "query_chunk_size": 64,
        },
        "gaussian_head": head,
        "motion": {"zero_init": True, "residual_hidden_dim": 8},
    })


def _expert_backend(**overrides):
    return OptionExpertBarrierVelocityGaussianBackend(
        _backend_cfg(), _gs_params(), dim=DIM, offset_bound=0.4,
        gaussians_per_token=1, gaussian_count_cfg=expert_count_cfg(**overrides),
        seed_mode="token",
    )


def _dense_backend():
    return DenseCountBarrierVelocityGaussianBackend(
        _backend_cfg(decoder_hidden=32), _gs_params(), dim=DIM,
        offset_bound=0.4, gaussians_per_token=5, seed_mode="token",
    )


def _inputs(slots=1, seed=0):
    generator = torch.Generator().manual_seed(seed)
    frame_tokens = [5, 4, 6, 3]
    tokens = sum(frame_tokens)
    # Two windows of two sweeps, packed into a few metres so the 0.8 m
    # neighbourhoods are not all singletons.
    position = torch.rand(tokens, 3, generator=generator) * 2.0
    feature = torch.randn(tokens, DIM, generator=generator)
    return (
        feature,
        position,
        position.unsqueeze(1).repeat(1, slots, 1),
        torch.tensor([5, 9, 15, 18]),
        torch.tensor([0, 0, 1, 1]),
        [torch.eye(4).repeat(2, 1, 1), torch.eye(4).repeat(2, 1, 1)],
        [torch.tensor([0.0, 1.0])] * 2,
        [torch.tensor([0.0, 0.6])] * 2,
        torch.tensor([0.6, 0.6]),
    )


def _check_token_contract(items, per_token_count=None):
    for item in items:
        token = item["source_token_index"]
        # One slot-0 Gaussian per token keeps the per-token velocity prior exact.
        assert int((item["gaussian_slot_index"] == 0).sum()) == int(
            token.unique().numel()
        )
        for index in token.unique():
            rows = token == index
            velocity = item["velocity"][rows]
            assert torch.equal(velocity, velocity[:1].expand_as(velocity))
            count = int(rows.sum())
            assert torch.equal(
                item["selected_k"][rows], torch.full((count,), count)
            )
            if per_token_count is not None:
                assert count == per_token_count


def test_v15_2_backend_emits_routed_experts_and_every_e5_in_training():
    torch.manual_seed(9)
    backend = _expert_backend().train()
    with torch.no_grad():
        backend.gaussian_head.router[-1].weight.normal_(0.0, 1.0)
    output = backend(*_inputs())
    routing = output["routing_stats"]
    assert routing["option_counts"].tolist() == [1, 3, 5]
    counts = routing["selected_k"]
    routed = output["batch_gaussians"]
    auxiliary = output["auxiliary_batch_gaussians"]
    assert [item["position"].shape[0] for item in routed] == [
        int(counts[:9].sum()), int(counts[9:].sum()),
    ]
    assert [item["position"].shape[0] for item in auxiliary] == [45, 45]
    for item in routed:
        assert torch.equal(item["opacity_gate"], torch.ones_like(item["opacity"]))
        assert item["opacity_gate"].requires_grad
    for item in auxiliary:
        assert "opacity_gate" not in item
    _check_token_contract(routed)
    _check_token_contract(auxiliary, per_token_count=5)


def test_v15_2_backend_evaluates_the_argmax_experts_alone():
    torch.manual_seed(10)
    backend = _expert_backend().eval()
    with torch.no_grad():
        backend.gaussian_head.router[-1].weight.normal_(0.0, 1.0)
    output = backend(*_inputs())
    assert "auxiliary_batch_gaussians" not in output
    assert all("opacity_gate" not in item for item in output["batch_gaussians"])
    routing = output["routing_stats"]
    assert torch.equal(
        routing["selected_k"],
        torch.tensor(COUNTS)[routing["k_logits"].argmax(dim=-1)],
    )
    _check_token_contract(output["batch_gaussians"])


def test_v15_3_backend_renders_five_per_token_without_a_router():
    torch.manual_seed(11)
    backend = _dense_backend()
    for mode in (backend.train(), backend.eval()):
        output = mode(*_inputs(slots=5))
        assert "routing_stats" not in output
        assert "auxiliary_batch_gaussians" not in output
        items = output["batch_gaussians"]
        assert [item["position"].shape[0] for item in items] == [45, 45]
        assert all("opacity_gate" not in item for item in items)
        _check_token_contract(items, per_token_count=5)


def test_v15_3_velocity_reads_the_temporal_feature_not_the_local_refinement():
    torch.manual_seed(12)
    backend = _dense_backend().eval()

    def token_velocity(output):
        return torch.cat([
            item["velocity"][item["gaussian_slot_index"] == 0]
            for item in output["batch_gaussians"]
        ])

    before = backend(*_inputs(slots=5))
    with torch.no_grad():
        backend.local_attention.out_proj.weight.normal_(0.0, 1.0)
        backend.local_attention.ffn[-1].weight.normal_(0.0, 1.0)
    after = backend(*_inputs(slots=5))
    assert torch.equal(token_velocity(before), token_velocity(after))
    assert not torch.equal(
        torch.cat([item["shs"] for item in before["batch_gaussians"]]),
        torch.cat([item["shs"] for item in after["batch_gaussians"]]),
    )


def test_backends_reject_count_contracts_they_cannot_honour():
    with pytest.raises(ValueError, match="has no count router"):
        DenseCountBarrierVelocityGaussianBackend(
            _backend_cfg(decoder_hidden=32), _gs_params(), dim=DIM,
            offset_bound=0.4, gaussians_per_token=5,
            gaussian_count_cfg=expert_count_cfg(), seed_mode="token",
        )
    with pytest.raises(ValueError, match="seed_mode"):
        DenseCountBarrierVelocityGaussianBackend(
            _backend_cfg(decoder_hidden=32), _gs_params(), dim=DIM,
            offset_bound=0.4, gaussians_per_token=5, seed_mode="medoid",
        )
    with pytest.raises(ValueError, match="routed count block"):
        OptionExpertBarrierVelocityGaussianBackend(
            _backend_cfg(), _gs_params(), dim=DIM, offset_bound=0.4,
            seed_mode="token",
        )
    with pytest.raises(ValueError, match="one shared anchor"):
        OptionExpertBarrierVelocityGaussianBackend(
            _backend_cfg(), _gs_params(), dim=DIM, offset_bound=0.4,
            gaussians_per_token=3, gaussian_count_cfg=expert_count_cfg(),
            seed_mode="token",
        )
    backend = _dense_backend()
    with pytest.raises(ValueError, match="Dynamic 2DGS expects 5 seed"):
        backend(*_inputs(slots=1))


def test_reducing_tokens_reach_the_router_through_the_render_pipeline(monkeypatch):
    """Exercise head -> sample packing -> trajectory -> Python renderer.

    Only the CUDA kernel is replaced: its differentiable stand-in sums the
    activated opacities, so dL/do' = 1. Every token reduces to E1, whose one
    Gaussian has opacity exactly one, so each token's reduce gate receives
    exactly 1 -- in V15 the same tokens would send nothing.
    """
    from src.models_new import gaussian_renderer
    from src.models_new.module.dynamic_gaussian import DynamicGausTemp
    from src.models_new.module.m3_g2p import DynamicGausRender

    backend = _expert_backend().train()
    arguments = _inputs()
    head = backend.gaussian_head
    _force_choices(head, [REDUCE] * arguments[0].shape[0])
    with torch.no_grad():
        for expert, count in zip(head.experts, COUNTS):
            expert.weight.reshape(count, PARAM_DIM, -1)[:, 32].zero_()
            expert.bias.reshape(count, PARAM_DIM)[:, 32] = 17.0
    output = backend(*arguments)
    gaussians = DynamicGausTemp()(output, arguments[-2], arguments[-1])
    height, width = 2, 3
    camera = SimpleNamespace(
        timestamp_sec=0.3, image_height=height, image_width=width,
        FoVx=1.0, FoVy=1.0, world_view_transform=torch.eye(4),
        full_proj_transform=torch.eye(4), camera_center=torch.zeros(3),
        vfov=(10.0, -30.0), hfov=(-180.0, 180.0),
        row_to_theta=torch.zeros(height), pts_depth=torch.zeros(height, width),
        pts_intensity=torch.zeros(height, width),
    )
    kernel_inputs = []

    def rasterize(**kwargs):
        opacity = kwargs["opacities"]
        kernel_inputs.append(opacity)
        image = opacity.sum().expand(4, height, width)
        return (
            opacity.new_zeros(opacity.shape[0]), image,
            opacity.new_zeros(3, height, width), image,
            image[:1], opacity.new_ones(opacity.shape[0]),
        )

    monkeypatch.setattr(
        gaussian_renderer, "GaussianRasterizationSettings",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )
    monkeypatch.setattr(
        gaussian_renderer, "GaussianRasterizer", lambda **kwargs: rasterize,
    )
    renderer = DynamicGausRender(OmegaConf.create({
        "neg_fov": False, "sh_degree": 3, "scale_factor": 1.0,
    })).train()
    renders = renderer(
        gaussians, {"cameras": [[camera], [camera]]}, compute_points=False,
    )
    assert [value.shape[0] for value in kernel_inputs] == [9, 9]
    assert all(torch.equal(value, torch.ones_like(value)) for value in kernel_inputs)
    (renders["depth"].sum() / (height * width)).backward()
    gradient = head.captured_selection.grad
    assert torch.equal(gradient[:, REDUCE], torch.ones_like(gradient[:, REDUCE]))
    assert torch.count_nonzero(gradient[:, KEEP:]) == 0
    assert head.router[-1].bias.grad.abs().sum() > 0


# ---------------------------------------------------------------- aux views


def _view_wrapper(views):
    from src.model_wrapper import ModelWrapper

    wrapper = ModelWrapper.__new__(ModelWrapper)
    torch.nn.Module.__init__(wrapper)
    wrapper._auxiliary_views = views
    return wrapper


def test_auxiliary_views_keep_k_distinct_time_ordered_views_per_window():
    wrapper = _view_wrapper(2)
    windows = [
        [f"w0v{i}" for i in range(5)],
        [f"w1v{i}" for i in range(3)],
        [f"w2v{i}" for i in range(2)],
    ]
    gt = {"cameras": windows, "other": "kept"}
    torch.manual_seed(0)
    counts = [torch.zeros(len(window)) for window in windows]
    for _ in range(600):
        sampled = wrapper._auxiliary_render_targets(gt)
        assert sampled["other"] == "kept"
        assert [len(window) for window in sampled["cameras"]] == [2, 2, 2]
        for index, (picked, window) in enumerate(zip(sampled["cameras"], windows)):
            positions = [window.index(view) for view in picked]
            # Distinct views of the same window, in time order.
            assert positions == sorted(set(positions))
            counts[index][positions] += 1
    # Every view is drawn, uniformly: 600 * 2 / 5 = 240 per view.
    assert counts[0].min() > 180 and counts[0].max() < 300
    assert counts[1].min() > 340 and counts[1].max() < 460
    assert torch.equal(counts[2], torch.full((2,), 600.0))
    # The input is not modified in place.
    assert [len(window) for window in windows] == [5, 3, 2]


def test_auxiliary_views_none_renders_every_view():
    gt = {"cameras": [["a", "b", "c"]]}
    assert _view_wrapper(None)._auxiliary_render_targets(gt) is gt


def test_auxiliary_render_loss_renders_the_sampled_views():
    from unittest import mock

    wrapper = _view_wrapper(2)
    wrapper.g2g_model = mock.Mock(side_effect=lambda x, *_: x["batch_gaussians"])
    wrapper.g2p_model = mock.Mock(return_value=object())
    wrapper.loss = mock.Mock(return_value={"total": torch.tensor(2.0)})
    losses = {"total": torch.tensor(4.0)}
    gt = {"cameras": [list(range(5)), list(range(4))]}
    torch.manual_seed(1)
    wrapper._average_auxiliary_render_loss(
        losses, [{"marker": 1}],
        {"timestamps_sec": None, "window_duration_sec": None},
        gt, prefix="train", compute_points=True,
    )
    rendered_gt = wrapper.g2p_model.call_args.args[1]
    assert [len(window) for window in rendered_gt["cameras"]] == [2, 2]
    assert gt["cameras"] == [list(range(5)), list(range(4))]
    torch.testing.assert_close(losses["total"], torch.tensor(3.0))
