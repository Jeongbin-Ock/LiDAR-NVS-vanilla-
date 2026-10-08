"""V15's nested Gaussian count: the G1..G5 head, its router, and its logging."""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from src.models.module.dynamic_gaussian import NestedCountGaussianHead
from src.models.utils.render import Gaussianutil
from src.models.utils.routing_logging import (
    routing_option_counts,
    routing_sufficient_statistics,
)

DIM = 24
PARAM_DIM = 42
REDUCE, KEEP, ADD = 0, 1, 2


def _gs_params():
    return OmegaConf.create({
        "shs": 32, "opacity": 1, "scaling": 2, "rotation": 4, "offset": 3,
    })


def nested_count_cfg(**overrides):
    nested = {
        "option_counts": [1, 3, 5],
        "seed_mode": "token",
        "tau": 1.0,
        "decoder_hidden": 32,
        "router_hidden": [16, 8],
        "auxiliary_render": "dense_keep",
    }
    nested.update(overrides)
    return OmegaConf.create({
        "count_mode": "nested_gumbel",
        "nested_count": nested,
    })


def _head(**overrides):
    return NestedCountGaussianHead(
        nested_count_cfg(**overrides),
        OmegaConf.create({"initial_opacity": 0.2, "initial_scale_m": 0.3}),
        _gs_params(),
        DIM,
        offset_bound=0.4,
    )


def _force_choices(head, choices):
    """Pin the hard choice while keeping the soft straight-through backward."""
    chosen = torch.as_tensor(choices, dtype=torch.long)

    def select(logits):
        soft = torch.softmax(logits.float(), dim=-1)
        hard = F.one_hot(chosen, num_classes=logits.shape[-1]).to(soft.dtype)
        return hard - soft.detach() + soft

    head._select = select


def _randomize(module):
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.normal_(0.0, 0.2)


def _render_opacity(packed):
    return Gaussianutil(None).get_opacity(
        packed["raw"]["opacity"], opacity_gate=packed.get("opacity_gate"),
    )


def test_initial_head_is_uniform_and_starts_on_the_anchor():
    torch.manual_seed(0)
    head = _head().eval()
    feature = torch.randn(7, DIM)
    anchor = torch.randn(7, 3)
    routed, auxiliary, routing = head(feature, anchor)

    assert auxiliary is None
    torch.testing.assert_close(routing["logits"], torch.zeros(7, 3))
    # Offset rows start at zero, so every Gaussian sits on its token anchor.
    torch.testing.assert_close(
        routed["position"], anchor[routed["token_index"]]
    )
    opacity = torch.sigmoid(routed["raw"]["opacity"])
    torch.testing.assert_close(
        opacity, torch.full_like(opacity, 0.2), atol=0.05, rtol=0.0
    )
    output = head.decoder[-1]
    bias = output.bias.reshape(5, PARAM_DIM)
    torch.testing.assert_close(
        bias[:, 32], torch.full((5,), math.log(0.2 / 0.8))
    )
    torch.testing.assert_close(
        bias[:, 33:35], torch.full((5, 2), math.log(math.expm1(0.3)))
    )
    torch.testing.assert_close(bias[:, 35], torch.ones(5))
    torch.testing.assert_close(bias[:, 39:42], torch.zeros(5, 3))
    weight = output.weight.reshape(5, PARAM_DIM, -1)
    assert torch.count_nonzero(weight[:, 39:42]) == 0


def test_router_and_decoder_have_separate_layer_norms():
    head = _head()
    assert head.router_norm is not head.decoder_norm
    names = [name for name, _ in head.named_parameters()]
    assert names == [
        "decoder_norm.weight", "decoder_norm.bias",
        "decoder.0.weight", "decoder.0.bias",
        "decoder.2.weight", "decoder.2.bias",
        "router_norm.weight", "router_norm.bias",
        "router.0.weight", "router.0.bias",
        "router.2.weight", "router.2.bias",
        "router.4.weight", "router.4.bias",
    ]
    assert tuple(head.router[0].weight.shape) == (16, DIM)
    assert tuple(head.router[2].weight.shape) == (8, 16)
    assert tuple(head.router[4].weight.shape) == (3, 8)
    assert tuple(head.decoder[2].weight.shape) == (5 * PARAM_DIM, 32)


def test_routed_set_is_each_tokens_nested_prefix():
    torch.manual_seed(1)
    head = _head().train()
    _randomize(head)
    choices = [ADD, REDUCE, KEEP, KEEP, ADD]
    _force_choices(head, choices)
    feature = torch.randn(5, DIM)
    anchor = torch.randn(5, 3)
    routed, auxiliary, routing = head(feature, anchor)

    counts = [5, 1, 3, 3, 5]
    assert routing["count"].tolist() == counts
    assert routing["option"].tolist() == choices
    expected_token = torch.cat([
        torch.full((count,), token) for token, count in enumerate(counts)
    ])
    expected_slot = torch.cat([torch.arange(count) for count in counts])
    assert torch.equal(routed["token_index"], expected_token)
    assert torch.equal(routed["slot_index"], expected_slot)

    decoded = head.decoder(head.decoder_norm(feature)).reshape(5, 5, PARAM_DIM)
    rows = decoded[expected_token, expected_slot]
    # No inverse-sigmoid round-trip: every raw and rendered opacity is exact.
    assert torch.equal(routed["raw"]["opacity"], rows[:, 32:33])
    assert torch.equal(_render_opacity(routed), rows[:, 32:33].sigmoid())
    assert torch.equal(
        routed["opacity_gate"], torch.ones_like(routed["raw"]["opacity"]),
    )
    assert torch.equal(routed["raw"]["shs"], rows[:, :32])
    torch.testing.assert_close(
        routed["position"],
        anchor[expected_token] + 0.4 * torch.tanh(rows[:, 39:42]),
    )

    # The keep prefix is every token's G1..G3, whatever the router chose.
    assert torch.equal(
        auxiliary["token_index"], torch.arange(5).repeat_interleave(3)
    )
    assert torch.equal(auxiliary["slot_index"], torch.arange(3).repeat(5))
    keep_rows = decoded[:, :3].reshape(15, PARAM_DIM)
    assert torch.equal(auxiliary["raw"]["opacity"], keep_rows[:, 32:33])


def _router_gradient(head, feature, anchor, weights_by_slot):
    """Router-logit gradient of a loss on the routed activated opacities."""
    head.zero_grad(set_to_none=True)
    captured = {}
    router = head.router

    def hook(_module, _inputs, output):
        output.retain_grad()
        captured["logits"] = output

    handle = router.register_forward_hook(hook)
    try:
        routed, _auxiliary, _routing = head(feature, anchor)
    finally:
        handle.remove()
    slot_weight = torch.as_tensor(weights_by_slot, dtype=torch.float32)
    loss = (
        _render_opacity(routed).squeeze(-1)
        * slot_weight[routed["slot_index"]]
    ).sum()
    loss.backward()
    return captured["logits"].grad


def test_router_gradient_follows_the_nested_gates():
    torch.manual_seed(2)
    head = _head().train()
    _randomize(head)
    _force_choices(head, [REDUCE, KEEP, ADD])
    feature = torch.randn(3, DIM)
    anchor = torch.randn(3, 3)

    # A loss on G1 alone never reaches the router: every choice renders G1.
    grad = _router_gradient(head, feature, anchor, [1, 0, 0, 0, 0])
    assert torch.count_nonzero(grad) == 0

    # G2/G3 opacity reaches the router through y_keep + y_add, from the keep
    # and add tokens; the reducing token renders no gated Gaussian at all.
    grad = _router_gradient(head, feature, anchor, [0, 1, 1, 0, 0])
    assert torch.count_nonzero(grad[0]) == 0
    assert torch.count_nonzero(grad[1]) > 0
    assert torch.count_nonzero(grad[2]) > 0

    # G4/G5 opacity reaches the router only from the adding token.
    grad = _router_gradient(head, feature, anchor, [0, 0, 0, 1, 1])
    assert torch.count_nonzero(grad[:2]) == 0
    assert torch.count_nonzero(grad[2]) > 0


def test_straight_through_gate_matches_the_activated_opacity_derivative():
    """dL/dgate = o * dL/do' for a gated Gaussian of a keep token."""
    torch.manual_seed(3)
    head = _head().train()
    _randomize(head)
    _force_choices(head, [KEEP])
    feature = torch.randn(1, DIM)
    anchor = torch.randn(1, 3)
    routed, _auxiliary, routing = head(feature, anchor)
    selection = routing["selection"]
    selection.retain_grad()
    activated = _render_opacity(routed).squeeze(-1)
    # dL/do' = 1 on G2 only.
    loss = activated[1]
    loss.backward()
    # gate_keep_or_add = y_keep + y_add feeds G2/G3; the reduce entry gets none.
    expected = activated[1].detach()
    torch.testing.assert_close(selection.grad[0, KEEP], expected)
    torch.testing.assert_close(selection.grad[0, ADD], expected)
    assert selection.grad[0, REDUCE] == 0


@pytest.mark.parametrize("raw_value", [-100.0, -14.0, 14.0, 17.0, 20.0, 100.0])
def test_saturated_opacity_preserves_the_gate_gradient(raw_value):
    head = _head().train()
    _force_choices(head, [KEEP])
    with torch.no_grad():
        output = head.decoder[-1]
        output.weight.reshape(5, PARAM_DIM, -1)[:, 32].zero_()
        output.bias.reshape(5, PARAM_DIM)[:, 32] = raw_value
    routed, _auxiliary, routing = head(torch.randn(1, DIM), torch.zeros(1, 3))
    raw = routed["raw"]["opacity"]
    raw.retain_grad()
    selection = routing["selection"]
    selection.retain_grad()
    activated = _render_opacity(routed)
    assert torch.equal(raw, torch.full_like(raw, raw_value))
    assert torch.equal(activated, raw.sigmoid())
    activated[1].sum().backward()  # dL/do' = 1 for G2.
    expected_opacity = raw.detach()[1, 0].sigmoid()
    torch.testing.assert_close(selection.grad[0, KEEP], expected_opacity)
    torch.testing.assert_close(selection.grad[0, ADD], expected_opacity)
    assert selection.grad[0, REDUCE] == 0
    torch.testing.assert_close(
        raw.grad[1, 0], expected_opacity * (1 - expected_opacity),
        atol=0.0, rtol=0.0,
    )
    assert torch.isfinite(activated).all()
    assert torch.isfinite(raw.grad).all()
    assert torch.isfinite(selection.grad).all()


def test_opacity_gate_rejects_broadcasting_between_gaussians():
    with pytest.raises(ValueError, match="align one-to-one"):
        Gaussianutil(None).get_opacity(
            torch.zeros(3, 1), opacity_gate=torch.ones(3),
        )


def test_auxiliary_set_ignores_the_router_and_only_exists_in_training():
    torch.manual_seed(4)
    head = _head().train()
    _randomize(head)
    _force_choices(head, [KEEP, ADD])
    feature = torch.randn(2, DIM)
    anchor = torch.randn(2, 3)
    head.zero_grad(set_to_none=True)
    _routed, auxiliary, _routing = head(feature, anchor)
    assert "opacity_gate" not in auxiliary
    _render_opacity(auxiliary).sum().backward()
    for parameter in head.router.parameters():
        assert parameter.grad is None or torch.count_nonzero(parameter.grad) == 0

    evaluated, auxiliary, _routing = head.eval()(feature, anchor)
    assert auxiliary is None and "opacity_gate" not in evaluated
    silent = _head(auxiliary_render="none").train()
    assert silent(feature, anchor)[1] is None


def test_evaluation_takes_the_argmax_choice():
    torch.manual_seed(5)
    head = _head().eval()
    _randomize(head)
    feature = torch.randn(9, DIM)
    _routed, _auxiliary, routing = head(feature, torch.randn(9, 3))
    assert torch.equal(routing["option"], routing["logits"].argmax(dim=-1))
    assert torch.equal(
        routing["count"],
        torch.tensor([1, 3, 5])[routing["logits"].argmax(dim=-1)],
    )


@pytest.mark.parametrize("overrides, message", [
    ({"option_counts": [1, 2, 3]}, "option_counts"),
    ({"seed_mode": "range_quantile"}, "seed_mode"),
    ({"tau": 0.0}, "tau"),
    ({"decoder_hidden": 0}, "hidden"),
    ({"router_hidden": 16}, "router_hidden"),
    ({"router_hidden": []}, "router_hidden"),
    ({"router_hidden": [16, 0]}, "router_hidden"),
    ({"auxiliary_render": "dense_all"}, "auxiliary_render"),
])
def test_head_rejects_an_invalid_nested_contract(overrides, message):
    with pytest.raises(ValueError, match=message):
        _head(**overrides)


def _routing(**fields):
    base = {
        "k_logits": torch.tensor([
            [2.0, 0.0, -1.0],
            [0.0, 3.0, 1.0],
            [0.0, 1.0, 4.0],
            [1.0, 0.5, 0.0],
        ]),
        "token_position_local": torch.tensor([
            [5.0, 0.0, 0.0],
            [15.0, 0.0, 0.0],
            [25.0, 0.0, 0.0],
            [120.0, 0.0, 0.0],
        ]),
    }
    base.update(fields)
    return base


def test_routing_statistics_count_gaussians_per_option():
    edges = (0, 10, 20, 30)
    nested = routing_sufficient_statistics(
        _routing(
            selected_option=torch.tensor([0, 1, 2, 2]),
            option_counts=torch.tensor([1, 3, 5]),
        ),
        edges,
    )
    assert nested["sampled_k_sum"].item() == 1 + 3 + 5 + 5
    # argmax options are [0, 1, 2, 0] -> counts [1, 3, 5, 1]
    assert nested["argmax_k_sum"].item() == 10
    assert nested["selected_counts"].tolist() == [1, 1, 2]
    assert nested["argmax_counts"].tolist() == [2, 1, 1]
    assert nested["range_sampled_k_sums"].tolist() == [1, 3, 5, 5]
    logits = _routing()["k_logits"]
    probability = torch.softmax(logits, dim=-1)
    torch.testing.assert_close(
        nested["expected_k_sum"],
        (probability * torch.tensor([1.0, 3.0, 5.0])).sum(),
    )
    torch.testing.assert_close(nested["probability_sums"], probability.sum(0))
    top = logits.topk(2, dim=-1).values
    torch.testing.assert_close(
        nested["logit_margin_sum"], (top[:, 0] - top[:, 1]).sum()
    )


def test_learned_gumbel_routing_statistics_are_unchanged():
    """``selected_k = option + 1`` with no mapping keeps K={1,2,3} counts."""
    stats = routing_sufficient_statistics(
        _routing(selected_k=torch.tensor([1, 2, 3, 3])), (0, 10, 20, 30),
    )
    assert stats["sampled_k_sum"].item() == 9
    assert stats["argmax_k_sum"].item() == 1 + 2 + 3 + 1
    assert stats["selected_counts"].tolist() == [1, 1, 2]
    assert routing_option_counts({}, 3).tolist() == [1, 2, 3]


def test_routing_statistics_reject_an_option_outside_the_logits():
    with pytest.raises(ValueError, match="invalid option"):
        routing_sufficient_statistics(
            _routing(
                selected_option=torch.tensor([0, 1, 3, 2]),
                option_counts=torch.tensor([1, 3, 5]),
            ),
            (0, 10),
        )
    with pytest.raises(ValueError, match="one count per router logit"):
        routing_option_counts({"option_counts": torch.tensor([1, 3])}, 3)


def test_training_total_averages_the_routed_and_keep_prefix_renders():
    """The mean of the two render totals is optimized; aux terms are logged."""
    from unittest import mock

    from src.model_wrapper import ModelWrapper

    wrapper = ModelWrapper.__new__(ModelWrapper)
    torch.nn.Module.__init__(wrapper)
    wrapper._auxiliary_views = None  # V15 renders every target view.
    auxiliary_items = [{"marker": 1}]
    wrapper.g2g_model = mock.Mock(side_effect=lambda x, *_: x["batch_gaussians"])
    renders = object()
    wrapper.g2p_model = mock.Mock(return_value=renders)
    auxiliary_total = torch.tensor(2.0, requires_grad=True)
    wrapper.loss = mock.Mock(return_value={
        "total": auxiliary_total,
        "loss_depth": torch.tensor(0.5),
        "loss_raydrop": torch.tensor(0.25),
    })
    routed_total = torch.tensor(4.0, requires_grad=True)
    losses = {"total": routed_total, "loss_depth": torch.tensor(1.0)}
    _input = {
        "timestamps_sec": [torch.tensor([0.0, 0.5])],
        "window_duration_sec": torch.tensor([0.5]),
    }

    wrapper._average_auxiliary_render_loss(
        losses, auxiliary_items, _input, gt="gt", prefix="train",
        compute_points=True,
    )

    wrapper.g2p_model.assert_called_once_with(
        auxiliary_items, "gt", compute_points=True
    )
    _args, kwargs = wrapper.loss.call_args
    assert kwargs["compute_valid_metrics"] is False
    assert kwargs["compute_raydrop_metrics"] is False
    torch.testing.assert_close(losses["total"], torch.tensor(3.0))
    torch.testing.assert_close(losses["routed_total"], torch.tensor(4.0))
    torch.testing.assert_close(losses["aux_total"], torch.tensor(2.0))
    torch.testing.assert_close(losses["aux_loss_depth"], torch.tensor(0.5))
    # The routed render keeps the ordinary keys every metric reads.
    torch.testing.assert_close(losses["loss_depth"], torch.tensor(1.0))
    losses["total"].backward()
    assert routed_total.grad.item() == 0.5
    assert auxiliary_total.grad.item() == 0.5


def v15_backend_cfg(heads=4):
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
        "gaussian_head": {"initial_opacity": 0.2, "initial_scale_m": 0.3},
        "motion": {"zero_init": True, "residual_hidden_dim": 8},
    })


def _v15_backend(**nested_overrides):
    from src.models.module.dynamic_gaussian import (
        NestedCountBarrierVelocityGaussianBackend,
    )

    return NestedCountBarrierVelocityGaussianBackend(
        v15_backend_cfg(),
        _gs_params(),
        dim=DIM,
        offset_bound=0.4,
        gaussians_per_token=1,
        gaussian_count_cfg=nested_count_cfg(**nested_overrides),
        seed_mode="token",
    )


def _v15_inputs(seed=0):
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
        position.unsqueeze(1),
        torch.tensor([5, 9, 15, 18]),
        torch.tensor([0, 0, 1, 1]),
        [torch.eye(4).repeat(2, 1, 1), torch.eye(4).repeat(2, 1, 1)],
        [torch.tensor([0.0, 1.0])] * 2,
        [torch.tensor([0.0, 0.6])] * 2,
        torch.tensor([0.6, 0.6]),
    )


def _token_velocity(output):
    return torch.cat([
        item["velocity"][item["gaussian_slot_index"] == 0]
        for item in output["batch_gaussians"]
    ])


def test_v15_backend_emits_routed_and_keep_prefix_sets_in_training():
    torch.manual_seed(6)
    backend = _v15_backend().train()
    with torch.no_grad():
        backend.gaussian_head.router[-1].weight.normal_(0.0, 1.0)
    output = backend(*_v15_inputs())
    counts = output["routing_stats"]["selected_k"]
    assert output["routing_stats"]["option_counts"].tolist() == [1, 3, 5]
    routed = output["batch_gaussians"]
    auxiliary = output["auxiliary_batch_gaussians"]
    for item in routed:
        assert torch.equal(item["opacity_gate"], torch.ones_like(item["opacity"]))
        assert item["opacity_gate"].requires_grad
    assert [item["position"].shape[0] for item in routed] == [
        int(counts[:9].sum()), int(counts[9:].sum()),
    ]
    assert [item["position"].shape[0] for item in auxiliary] == [27, 27]
    for item in routed + auxiliary:
        token = item["source_token_index"]
        # One G1 per token keeps the per-token velocity prior exact.
        assert int((item["gaussian_slot_index"] == 0).sum()) == int(
            token.unique().numel()
        )
        for index in token.unique():
            velocity = item["velocity"][token == index]
            assert torch.equal(velocity, velocity[:1].expand_as(velocity))
    for item in auxiliary:
        assert "opacity_gate" not in item
        assert torch.equal(item["selected_k"], torch.full_like(item["selected_k"], 3))


def test_v15_backend_renders_only_the_argmax_prefix_in_evaluation():
    torch.manual_seed(7)
    backend = _v15_backend().eval()
    with torch.no_grad():
        backend.gaussian_head.router[-1].weight.normal_(0.0, 1.0)
    output = backend(*_v15_inputs())
    assert "auxiliary_batch_gaussians" not in output
    assert all("opacity_gate" not in item for item in output["batch_gaussians"])
    routing = output["routing_stats"]
    assert torch.equal(
        routing["selected_k"],
        torch.tensor([1, 3, 5])[routing["k_logits"].argmax(dim=-1)],
    )


def test_saturated_gate_reaches_rasterizer_through_the_render_pipeline(monkeypatch):
    """Exercise head -> sample packing -> trajectory -> Python renderer.

    Only the CUDA kernel is replaced: its differentiable stand-in sums the
    activated opacities, so dL/do'=1 and the expected nested gradient is known.
    """
    from types import SimpleNamespace

    from src.models import gaussian_renderer
    from src.models.module.dynamic_gaussian import DynamicGausTemp
    from src.models.module.m3_g2p import DynamicGausRender

    backend = _v15_backend().train()
    arguments = _v15_inputs()
    head = backend.gaussian_head
    _force_choices(head, [KEEP] * arguments[0].shape[0])
    original_select = head._select
    captured = {}

    def select(logits):
        selection = original_select(logits)
        selection.retain_grad()
        captured["selection"] = selection
        return selection

    head._select = select
    with torch.no_grad():
        head.decoder[-1].weight.reshape(5, PARAM_DIM, -1)[:, 32].zero_()
        head.decoder[-1].bias.reshape(5, PARAM_DIM)[:, 32] = 17.0
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
        assert kwargs["mask"].all()
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
    assert len(kernel_inputs) == 2
    assert all(torch.equal(value, torch.ones_like(value)) for value in kernel_inputs)
    (renders["depth"].sum() / (height * width)).backward()
    gradient = captured["selection"].grad
    assert torch.isfinite(gradient).all()
    assert torch.equal(gradient[:, REDUCE], torch.zeros_like(gradient[:, REDUCE]))
    # Each keep token renders two gated Gaussians, both with opacity exactly 1.
    assert torch.equal(gradient[:, KEEP], torch.full_like(gradient[:, KEEP], 2))
    assert torch.equal(gradient[:, ADD], torch.full_like(gradient[:, ADD], 2))
    assert head.router[-1].bias.grad.abs().sum() > 0


def test_v15_velocity_reads_the_temporal_feature_not_the_local_refinement():
    torch.manual_seed(8)
    backend = _v15_backend().eval()
    before = backend(*_v15_inputs())
    with torch.no_grad():
        backend.local_attention.out_proj.weight.normal_(0.0, 1.0)
        backend.local_attention.ffn[-1].weight.normal_(0.0, 1.0)
    after = backend(*_v15_inputs())
    assert torch.equal(_token_velocity(before), _token_velocity(after))
    decoded_before = torch.cat([item["shs"] for item in before["batch_gaussians"]])
    decoded_after = torch.cat([item["shs"] for item in after["batch_gaussians"]])
    assert decoded_before.shape != decoded_after.shape or not torch.equal(
        decoded_before, decoded_after
    )


def test_v15_temporal_stack_has_no_layer_scale():
    backend = _v15_backend()
    assert backend.temporal.cross_layer_scale is None
    assert not any(
        "cross_layer_scale" in name for name, _ in backend.named_parameters()
    )


def test_v15_backend_rejects_seed_contracts_it_cannot_honour():
    from src.models.module.dynamic_gaussian import (
        NestedCountBarrierVelocityGaussianBackend,
    )

    with pytest.raises(ValueError, match="seed_mode"):
        NestedCountBarrierVelocityGaussianBackend(
            v15_backend_cfg(), _gs_params(), dim=DIM, offset_bound=0.4,
            gaussian_count_cfg=nested_count_cfg(), seed_mode="medoid",
        )
    with pytest.raises(ValueError, match="one shared anchor"):
        NestedCountBarrierVelocityGaussianBackend(
            v15_backend_cfg(), _gs_params(), dim=DIM, offset_bound=0.4,
            gaussians_per_token=3, gaussian_count_cfg=nested_count_cfg(),
            seed_mode="token",
        )
    backend = _v15_backend()
    arguments = list(_v15_inputs())
    with pytest.raises(ValueError, match="seed deltas"):
        backend(*arguments, seed_delta_local=torch.zeros_like(arguments[2]))

