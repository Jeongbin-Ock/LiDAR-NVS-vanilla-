"""V16's ordinal nested count: bounded gates, blend masks, masked rasterization.

V16.1 is the same head with a random-prefix auxiliary render.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from src.models.module.dynamic_gaussian import (
    CosineGateLayer,
    NestedCountGaussianHead,
    OrdinalCountGaussianHead,
)
from src.models.module.dynamic.ordinal_count import option_log_probabilities
from src.models.utils.routing_logging import routing_sufficient_statistics

DIM = 24
PARAM_DIM = 42
SCALE = 3.0
REDUCE, KEEP, ADD = 0, 1, 2

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs the CUDA rasterizer"
)


def _gs_params():
    return OmegaConf.create({
        "shs": 32, "opacity": 1, "scaling": 2, "rotation": 4, "offset": 3,
    })


def ordinal_count_cfg(**overrides):
    block = {
        "option_counts": [1, 3, 5],
        "seed_mode": "token",
        "tau": 1.0,
        "decoder_hidden": 32,
        "router_hidden": [16, 8],
        "router_scale": SCALE,
        "auxiliary_render": "dense_add",
    }
    block.update(overrides)
    return OmegaConf.create({
        "count_mode": "nested_ordinal",
        "ordinal_count": block,
    })


def _head(**overrides):
    return OrdinalCountGaussianHead(
        ordinal_count_cfg(**overrides),
        OmegaConf.create({"initial_opacity": 0.2, "initial_scale_m": 0.3}),
        _gs_params(),
        DIM,
        offset_bound=0.4,
    )


def _force_gates(head, keep, add):
    """Pin the hard gates; keep the straight-through sigmoid backward."""
    hard = torch.stack([
        torch.as_tensor(keep, dtype=torch.bool),
        torch.as_tensor(add, dtype=torch.bool),
    ], dim=-1)

    def sample(gate_logits):
        soft = torch.sigmoid(gate_logits)
        return hard.to(soft) + (soft - soft.detach()), hard.to(gate_logits.device)

    head._sample_gates = sample


def _fixed_router(head, gate_logits):
    """Make the router return the given ``(N, 2)`` gate logits."""

    class Fixed(torch.nn.Module):
        def forward(self, _feature):
            return gate_logits

    head.router = Fixed()


def test_head_is_v15s_decoder_with_a_two_gate_cosine_router():
    head = _head()
    names = [name for name, _ in head.named_parameters()]
    assert names == [
        "decoder_norm.weight", "decoder_norm.bias",
        "decoder.0.weight", "decoder.0.bias",
        "decoder.2.weight", "decoder.2.bias",
        "router_norm.weight", "router_norm.bias",
        "router.0.weight", "router.0.bias",
        "router.2.weight", "router.2.bias",
        "router.4.weight",
    ]
    assert isinstance(head.router[-1], CosineGateLayer)
    assert tuple(head.router[-1].weight.shape) == (2, 8)
    assert head.router[-1].scale == SCALE == head.router_scale
    assert head.router_norm is not head.decoder_norm
    v15 = NestedCountGaussianHead(
        OmegaConf.create({"count_mode": "nested_gumbel", "nested_count": {
            key: value for key, value in ordinal_count_cfg()[
                "ordinal_count"
            ].items() if key != "router_scale"
        } | {"auxiliary_render": "dense_keep"}}),
        OmegaConf.create({"initial_opacity": 0.2, "initial_scale_m": 0.3}),
        _gs_params(), DIM, offset_bound=0.4,
    )
    decoder = {
        name: tuple(value.shape) for name, value in head.state_dict().items()
        if name.startswith("decoder")
    }
    assert decoder == {
        name: tuple(value.shape) for name, value in v15.state_dict().items()
        if name.startswith("decoder")
    }


def test_initial_gates_are_bounded_and_gaussians_start_on_the_anchor():
    torch.manual_seed(0)
    head = _head().eval()
    feature = torch.randn(64, DIM)
    anchor = torch.randn(64, 3)
    routed, auxiliary, routing = head(feature, anchor)
    assert auxiliary is None
    gates = routing["gate_logits"]
    assert gates.shape == (64, 2)
    assert gates.abs().max() <= SCALE * (1 + 1e-6)
    torch.testing.assert_close(
        routed["position"], anchor[routed["token_index"]]
    )
    probability = routing["logits"].exp()
    torch.testing.assert_close(probability.sum(-1), torch.ones(64))


@pytest.mark.parametrize("magnitude", [1.0, 1.0e3, 1.0e6])
def test_gate_logits_are_bounded_by_the_fixed_scale(magnitude):
    torch.manual_seed(1)
    head = _head()
    with torch.no_grad():
        for parameter in head.parameters():
            parameter.normal_(0.0, magnitude)
    _routed, _auxiliary, routing = head.eval()(
        magnitude * torch.randn(200, DIM), torch.zeros(200, 3)
    )
    gates = routing["gate_logits"]
    assert torch.isfinite(gates).all()
    assert gates.abs().max() <= SCALE * (1 + 1e-6)
    # The scale is a constant, not a parameter the optimizer could grow.
    assert all("scale" not in name for name, _ in head.named_parameters())


def test_option_probabilities_are_the_continuation_ratio_model():
    gates = torch.tensor([[0.0, 0.0], [3.0, -3.0], [-3.0, 3.0], [0.4, 0.4],
                          [1.2, -0.7], [-2.5, -2.5]])
    q_keep, q_add = torch.sigmoid(gates).unbind(-1)
    probability = option_log_probabilities(gates).exp()
    expected = torch.stack([
        1 - q_keep, q_keep * (1 - q_add), q_keep * q_add,
    ], dim=-1)
    torch.testing.assert_close(probability, expected)
    torch.testing.assert_close(probability.sum(-1), torch.ones(6))
    # q_keep: G2/G3 present at all; q_add: G4/G5 present given G2/G3.
    torch.testing.assert_close(probability[:, KEEP] + probability[:, ADD], q_keep)
    torch.testing.assert_close(
        probability[:, ADD] / (probability[:, KEEP] + probability[:, ADD]), q_add
    )
    # At the scale bound every option keeps a floor of probability.
    bounded = option_log_probabilities(
        torch.tensor([[SCALE, SCALE], [SCALE, -SCALE], [-SCALE, SCALE]])
    ).exp()
    assert bounded.min() > 0.002
    assert bounded.max() < 0.96


def test_evaluation_renders_the_most_probable_prefix():
    torch.manual_seed(2)
    head = _head().eval()
    # q = (0.6, 0.6): per-gate thresholds would say add, but the most
    # probable option is reduce (p = 0.40 / 0.24 / 0.36).
    logit = math.log(0.6 / 0.4)
    gates = torch.tensor([
        [logit, logit], [2.0, -2.0], [2.0, 2.0], [-2.0, 2.0],
    ])
    _fixed_router(head, gates)
    routed, auxiliary, routing = head(torch.randn(4, DIM), torch.randn(4, 3))
    assert auxiliary is None and "blend_mask" not in routed
    assert routing["option"].tolist() == [REDUCE, KEEP, ADD, REDUCE]
    assert routing["count"].tolist() == [1, 3, 5, 1]
    assert routed["token_index"].tolist() == [0, 1, 1, 1, 2, 2, 2, 2, 2, 3]
    assert routed["slot_index"].tolist() == [0, 0, 1, 2, 0, 1, 2, 3, 4, 0]
    torch.testing.assert_close(routing["logits"], option_log_probabilities(gates))


def test_training_renders_every_slot_with_the_nested_blend_masks():
    torch.manual_seed(3)
    head = _head().train()
    keep = [False, False, True, True]
    add = [False, True, False, True]
    _force_gates(head, keep, add)
    feature = torch.randn(4, DIM)
    anchor = torch.randn(4, 3)
    routed, auxiliary, routing = head(feature, anchor)
    assert routing["option"].tolist() == [REDUCE, REDUCE, KEEP, ADD]
    assert routing["count"].tolist() == [1, 1, 3, 5]
    assert routed["token_index"].tolist() == [i for i in range(4) for _ in range(5)]
    assert routed["slot_index"].tolist() == list(range(5)) * 4
    mask = routed["blend_mask"].squeeze(-1).reshape(4, 5)
    # Forward values are exactly the hard gates: G1 always, G2/G3 = keep,
    # G4/G5 = keep * add (reduce closes G4/G5 even when add was sampled open).
    assert torch.equal(mask, torch.tensor([
        [1.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0, 1.0],
    ]))
    assert routed["blend_mask"].requires_grad
    # The auxiliary set is the same Gaussians, unmasked.
    assert "blend_mask" not in auxiliary
    assert torch.equal(auxiliary["token_index"], routed["token_index"])
    assert torch.equal(auxiliary["position"], routed["position"])
    for name in routed["raw"]:
        assert torch.equal(auxiliary["raw"][name], routed["raw"][name])
    silent = _head(auxiliary_render="none").train()
    _force_gates(silent, keep, add)
    routed, auxiliary, _ = silent(feature, anchor)
    assert auxiliary is None and "blend_mask" in routed


def test_random_prefix_auxiliary_is_uniform_and_independent_of_the_router():
    torch.manual_seed(11)
    head = _head(auxiliary_render="random_prefix").train()
    tokens = 6000
    # Pin both gates at the bound: the router samples add ~91% of the time,
    # yet the auxiliary prefixes stay uniform over {1, 3, 5}.
    _fixed_router(head, torch.full((tokens, 2), SCALE))
    feature = torch.randn(tokens, DIM)
    anchor = torch.randn(tokens, 3)
    routed, auxiliary, routing = head(feature, anchor)
    # The routed set is V16's: every token's G1..G5 under blend masks.
    assert routed["token_index"].numel() == 5 * tokens
    assert "blend_mask" in routed
    assert (routing["option"] == ADD).float().mean() > 0.85
    counts = routing["auxiliary_count"]
    assert counts.shape == (tokens,) and counts.dtype == torch.long
    assert set(counts.unique().tolist()) == {1, 3, 5}
    frequency = torch.bincount(counts, minlength=6)[[1, 3, 5]].float() / tokens
    # The binomial standard error at p = 1/3, n = 6000 is 0.0061.
    assert torch.all((frequency - 1.0 / 3.0).abs() < 0.03)
    # Each token renders exactly its prefix G1..Gk, token-major and unmasked.
    assert "blend_mask" not in auxiliary and "opacity_gate" not in auxiliary
    assert torch.equal(
        torch.bincount(auxiliary["token_index"], minlength=tokens), counts
    )
    assert torch.equal(
        auxiliary["slot_index"],
        torch.cat([torch.arange(int(k)) for k in counts]),
    )
    # They are the routed set's decoded Gaussians, slot for slot.
    rows = auxiliary["token_index"] * 5 + auxiliary["slot_index"]
    assert torch.equal(auxiliary["position"], routed["position"][rows])
    for name in auxiliary["raw"]:
        assert torch.equal(auxiliary["raw"][name], routed["raw"][name][rows])
    # V16's dense render reports no per-token count, and evaluation has no
    # auxiliary set at all.
    _, _, dense_routing = _head().train()(feature[:8], anchor[:8])
    assert "auxiliary_count" not in dense_routing
    evaluation = _head(auxiliary_render="random_prefix").eval()
    _, auxiliary, routing = evaluation(feature[:8], anchor[:8])
    assert auxiliary is None and "auxiliary_count" not in routing


def test_random_prefix_auxiliary_trains_the_decoder_but_not_the_router():
    torch.manual_seed(12)
    head = _head(auxiliary_render="random_prefix").train()
    _routed, auxiliary, _routing = head(torch.randn(64, DIM), torch.randn(64, 3))
    loss = auxiliary["position"].square().sum() + sum(
        value.square().sum() for value in auxiliary["raw"].values()
    )
    loss.backward()
    for module in (head.router, head.router_norm):
        assert all(
            parameter.grad is None or not parameter.grad.any()
            for parameter in module.parameters()
        )
    assert head.decoder[-1].weight.grad is not None
    assert head.decoder[-1].weight.grad.any()


def test_sampled_gates_are_bernoulli_with_a_straight_through_sigmoid():
    head = _head(tau=0.5)
    gate_logits = torch.tensor([-2.0, -0.5, 0.0, 1.0, 2.5]).repeat(40000, 1)
    gate_logits = gate_logits.reshape(-1, 2)
    torch.manual_seed(4)
    gates, sampled = head._sample_gates(gate_logits)
    assert torch.equal(gates, sampled.to(gates.dtype))
    frequency = sampled.float().reshape(-1, 5).mean(0)
    expected = torch.sigmoid(torch.tensor([-2.0, -0.5, 0.0, 1.0, 2.5]))
    torch.testing.assert_close(frequency, expected, atol=0.01, rtol=0.0)

    leaf = gate_logits[:8].clone().requires_grad_(True)
    torch.manual_seed(5)
    gates, _ = head._sample_gates(leaf)
    gates.sum().backward()
    torch.manual_seed(5)
    uniform = torch.rand_like(leaf).clamp_min(torch.finfo(leaf.dtype).tiny)
    noisy = (leaf.detach() + torch.log(uniform) - torch.log1p(-uniform)) / 0.5
    soft = torch.sigmoid(noisy)
    torch.testing.assert_close(leaf.grad, soft * (1 - soft) / 0.5)


def test_gate_gradients_follow_the_masked_group_slopes():
    """dL/dM per Gaussian -> keep gate g_A + M_add g_B, add gate M_keep g_B."""
    head = _head()
    torch.manual_seed(6)
    tokens = 6
    gate_logits = torch.randn(tokens, 2).requires_grad_(True)
    keep = torch.tensor([0, 0, 1, 1, 1, 0], dtype=torch.bool)
    add = torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.bool)
    soft = torch.sigmoid(gate_logits)
    hard = torch.stack([keep, add], dim=-1)
    gates = hard.to(soft) + (soft - soft.detach())
    token_index = torch.arange(tokens).repeat_interleave(5)
    slot_index = torch.arange(5).repeat(tokens)
    mask = head._blend_mask(gates, token_index, slot_index).squeeze(-1)
    slope = torch.randn(tokens * 5)  # stands in for the rasterizer's dL/dmask
    (mask * slope).sum().backward()
    per_slot = slope.reshape(tokens, 5)
    group_a = per_slot[:, 1:3].sum(-1)
    group_b = per_slot[:, 3:5].sum(-1)
    derivative = (soft * (1 - soft)).detach()
    torch.testing.assert_close(
        gate_logits.grad[:, 0], (group_a + add.float() * group_b) * derivative[:, 0]
    )
    torch.testing.assert_close(
        gate_logits.grad[:, 1], keep.float() * group_b * derivative[:, 1]
    )
    # G1 carries no gate.
    assert torch.equal(mask.reshape(tokens, 5)[:, 0], torch.ones(tokens))


@pytest.mark.parametrize("overrides, message", [
    ({"router_scale": 0.0}, "router_scale must be a positive number"),
    ({"router_scale": -1.0}, "router_scale must be a positive number"),
    ({"router_scale": float("inf")}, "router_scale must be a positive number"),
    ({"auxiliary_render": "dense_keep"}, "auxiliary_render"),
    ({"option_counts": [1, 2, 3]}, "option_counts"),
    ({"seed_mode": "medoid"}, "seed_mode"),
])
def test_head_rejects_an_invalid_ordinal_contract(overrides, message):
    with pytest.raises(ValueError, match=message):
        _head(**overrides)


def test_head_needs_its_own_count_block():
    with pytest.raises(ValueError, match="needs p2g.grid_query.ordinal_count"):
        OrdinalCountGaussianHead(
            OmegaConf.create({"count_mode": "nested_gumbel", "nested_count": {}}),
            OmegaConf.create({}), _gs_params(), DIM, offset_bound=0.4,
        )


def test_routing_statistics_read_the_option_log_probabilities():
    gates = torch.tensor([[2.0, -2.0], [-1.0, 0.0], [3.0, 3.0]])
    log_probability = option_log_probabilities(gates)
    statistics = routing_sufficient_statistics({
        "k_logits": log_probability,
        "selected_option": torch.tensor([KEEP, REDUCE, ADD]),
        "option_counts": torch.tensor([1, 3, 5]),
        "token_position_local": torch.tensor([[1.0, 0, 0], [15.0, 0, 0], [50.0, 0, 0]]),
    }, (0, 10, 20))
    probability = log_probability.exp()
    torch.testing.assert_close(statistics["probability_sums"], probability.sum(0))
    torch.testing.assert_close(
        statistics["expected_k_sum"],
        (probability * torch.tensor([1.0, 3.0, 5.0])).sum(),
    )
    assert statistics["sampled_k_sum"].item() == 9.0
    # Most probable options: keep, reduce, add -- one token each.
    assert statistics["argmax_counts"].tolist() == [1.0, 1.0, 1.0]
    assert statistics["argmax_k_sum"].item() == 9.0


def test_scale_diagnostic_counts_only_rendered_gaussians():
    from src.models.utils.loss import Loss

    loss = Loss.__new__(Loss)
    torch.nn.Module.__init__(loss)
    loss.scale_max_m = 1.0
    # softplus(10) ~ 10 m exceeds the 1 m threshold; softplus(-5) does not.
    item = {
        "scaling": torch.tensor([[-5.0, -5.0], [10.0, 10.0]]),
        "blend_mask": torch.tensor([[1.0], [0.0]]),
    }
    assert loss._scale_diagnostic([item], torch.zeros(())).item() == 0.0
    item["blend_mask"] = torch.ones(2, 1)
    assert loss._scale_diagnostic([item], torch.zeros(())).item() > 0.0


def v16_backend_cfg(heads=4):
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


def _v16_backend(**overrides):
    from src.models.module.dynamic_gaussian import (
        OrdinalCountBarrierVelocityGaussianBackend,
    )

    return OrdinalCountBarrierVelocityGaussianBackend(
        v16_backend_cfg(),
        _gs_params(),
        dim=DIM,
        offset_bound=0.4,
        gaussians_per_token=1,
        gaussian_count_cfg=ordinal_count_cfg(**overrides),
        seed_mode="token",
    )


def _v16_inputs(seed=0):
    generator = torch.Generator().manual_seed(seed)
    frame_tokens = [5, 4, 6, 3]
    tokens = sum(frame_tokens)
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


def test_v16_backend_carries_blend_masks_in_training():
    torch.manual_seed(7)
    backend = _v16_backend().train()
    output = backend(*_v16_inputs())
    routing = output["routing_stats"]
    assert routing["option_counts"].tolist() == [1, 3, 5]
    torch.testing.assert_close(
        routing["k_logits"].exp().sum(-1), torch.ones(18)
    )
    routed = output["batch_gaussians"]
    auxiliary = output["auxiliary_batch_gaussians"]
    # Both sets hold every token's five Gaussians: 9 tokens per window.
    assert [item["position"].shape[0] for item in routed] == [45, 45]
    assert [item["position"].shape[0] for item in auxiliary] == [45, 45]
    counts = routing["selected_k"]
    for window, item in enumerate(routed):
        mask = item["blend_mask"]
        assert mask.shape == item["opacity"].shape and mask.requires_grad
        assert set(mask.detach().unique().tolist()) <= {0.0, 1.0}
        # The open Gaussians are exactly the sampled prefixes.
        assert int(mask.detach().sum()) == int(counts[9 * window:9 * (window + 1)].sum())
        open_slots = item["gaussian_slot_index"][mask.squeeze(-1) > 0]
        assert torch.equal(
            open_slots,
            torch.cat([torch.arange(int(k)) for k in counts[9 * window:9 * (window + 1)]]),
        )
    for item in routed + auxiliary:
        token = item["source_token_index"]
        assert int((item["gaussian_slot_index"] == 0).sum()) == int(
            token.unique().numel()
        )
    for item in auxiliary:
        assert "blend_mask" not in item and "opacity_gate" not in item
        assert torch.equal(item["selected_k"], torch.full_like(item["selected_k"], 5))


def test_v16_backend_renders_only_the_most_probable_prefix_in_evaluation():
    torch.manual_seed(8)
    backend = _v16_backend().eval()
    output = backend(*_v16_inputs())
    assert "auxiliary_batch_gaussians" not in output
    assert all("blend_mask" not in item for item in output["batch_gaussians"])
    routing = output["routing_stats"]
    expected = torch.tensor([1, 3, 5])[routing["k_logits"].argmax(dim=-1)]
    assert torch.equal(routing["selected_k"], expected)
    assert sum(item["position"].shape[0] for item in output["batch_gaussians"]) == int(
        expected.sum()
    )


def test_v16_1_backend_renders_random_prefixes_as_the_auxiliary_set():
    torch.manual_seed(9)
    backend = _v16_backend(auxiliary_render="random_prefix").train()
    output = backend(*_v16_inputs())
    # The routed set is V16's: every token's five Gaussians, 9 tokens a window.
    assert [item["position"].shape[0] for item in output["batch_gaussians"]] == [45, 45]
    auxiliary = output["auxiliary_batch_gaussians"]
    tokens = 0
    for item in auxiliary:
        assert "blend_mask" not in item and "opacity_gate" not in item
        token = item["source_token_index"]
        for index in token.unique().tolist():
            rows = token == index
            k = int(rows.sum())
            assert k in (1, 3, 5)
            # A token renders G1..Gk and carries its own k.
            assert torch.equal(item["gaussian_slot_index"][rows], torch.arange(k))
            assert torch.equal(item["selected_k"][rows], torch.full((k,), k))
            tokens += 1
    assert tokens == 18


# --------------------------------------------------------------------------
# The CUDA rasterizer's masked blending.
# --------------------------------------------------------------------------

_CAMERA_FROM_LIDAR = np.array(
    [[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]], dtype=np.float32
)
_RENDER_CFG = OmegaConf.create({"neg_fov": True, "sh_degree": 3, "scale_factor": 1.0})
_RENDER_KEYS = ("depth", "depth_median", "normal", "intensity_sh", "alpha", "raydrop")


def _camera(timestamp=0.0):
    from src.models.utils.camera import Camera

    height, width = 32, 512
    return Camera(
        R=np.eye(3), T=np.zeros(3), vfov=(-30.0, 10.0), hfov=(-180.0, 180.0),
        viewmatrix=_CAMERA_FROM_LIDAR, resolution=(width, height),
        timestamp_sec=timestamp, pts_depth=torch.zeros(height, width),
        pts_intensity=torch.zeros(height, width),
    )


def _scene(count=500, seed=0):
    generator = torch.Generator().manual_seed(seed)
    azimuth = torch.rand(count, generator=generator) * 2 * math.pi
    distance = 4 + 30 * torch.rand(count, generator=generator)
    height = -1.7 + 3 * torch.rand(count, generator=generator)
    position = torch.stack(
        [distance * azimuth.cos(), distance * azimuth.sin(), height], 1
    )
    scales = torch.log(torch.expm1(0.2 + torch.rand(count, 3, generator=generator)))
    scales[:, 2] = -10.0
    return {
        "position": position.cuda(),
        "opacity": (-0.8 + 0.5 * torch.randn(count, 1, generator=generator)).cuda(),
        "scales": scales.cuda(),
        "rotations": F.normalize(torch.randn(count, 4, generator=generator), dim=1).cuda(),
        "shs": (0.3 * torch.randn(count, 16, 4, generator=generator)).cuda(),
    }


def _render(pc, blend_mask=None, keep=None):
    from src.models.gaussian_renderer import render

    pc = dict(pc)
    if keep is not None:
        pc = {key: value[keep] for key, value in pc.items()}
        blend_mask = None if blend_mask is None else blend_mask[keep]
    if blend_mask is not None:
        pc["blend_mask"] = blend_mask
    return render(
        viewpoint_camera=_camera(), pc=pc, cfg=_RENDER_CFG,
        bg_color=torch.tensor([0.0, 0.0, 0.0, 1.0]).cuda(), input_timestamp=0.0,
    )


def _weights(seed=1):
    generator = torch.Generator().manual_seed(seed)
    return {
        "depth": (0.05 * torch.randn(1, 32, 512, generator=generator)).cuda(),
        "intensity_sh": torch.randn(1, 32, 512, generator=generator).cuda(),
        "alpha": torch.randn(1, 32, 512, generator=generator).cuda(),
    }


def _linear_loss(out, weights):
    """Linear in the outputs, so linear in any one Gaussian's mask."""
    return sum((out[key] * weight).sum() for key, weight in weights.items())


def _gradients(pc, weights, blend_mask=None, keep=None):
    leaves = {key: value.detach().clone().requires_grad_(True) for key, value in pc.items()}
    if blend_mask is not None:
        blend_mask = blend_mask.detach().clone().requires_grad_(True)
    out = _render(leaves, blend_mask=blend_mask, keep=keep)
    _linear_loss(out, weights).backward()
    grads = {key: value.grad for key, value in leaves.items()}
    if blend_mask is not None:
        grads["blend_mask"] = blend_mask.grad
    return out, grads


def _relative(a, b):
    return ((a - b).abs().max() / max(a.abs().max(), b.abs().max(), 1e-12)).item()


@requires_cuda
def test_unit_blend_mask_renders_exactly_the_unmasked_scene():
    pc, weights = _scene(), _weights()
    plain, plain_grads = _gradients(pc, weights)
    ones = torch.ones(pc["position"].shape[0], 1, device="cuda")
    masked, masked_grads = _gradients(pc, weights, blend_mask=ones)
    assert int((plain["depth"] > 0).sum()) > 5000
    for key in _RENDER_KEYS:
        assert torch.equal(plain[key], masked[key]), key
    for key in pc:
        assert _relative(plain_grads[key], masked_grads[key]) < 1e-4, key


@requires_cuda
def test_zero_blend_mask_is_exactly_the_gaussians_absence():
    pc, weights = _scene(), _weights()
    count = pc["position"].shape[0]
    present = torch.rand(count, generator=torch.Generator().manual_seed(2)) < 0.4
    mask = present.float().unsqueeze(-1).cuda()
    keep = present.nonzero(as_tuple=True)[0].cuda()
    masked, masked_grads = _gradients(pc, weights, blend_mask=mask)
    subset, subset_grads = _gradients(pc, weights, keep=keep)
    for key in _RENDER_KEYS:
        assert torch.equal(masked[key], subset[key]), key
    absent = (~present).cuda()
    for key in pc:
        assert _relative(masked_grads[key][keep], subset_grads[key][keep]) < 1e-4, key
        # An absent Gaussian composites nothing, so its own attributes get
        # no gradient at all ...
        assert torch.count_nonzero(masked_grads[key][absent]) == 0, key
    # ... but its mask does: the slope of inserting it.
    assert (masked_grads["blend_mask"][absent] != 0).float().mean() > 0.9


@requires_cuda
def test_blend_mask_gradient_is_the_insertion_and_removal_difference():
    """For a loss linear in the outputs, one Gaussian's mask enters linearly.

    So dL/dmask at mask 0 (insertion), at mask 1 (removal) and at 0.5 all
    equal L(mask=1) - L(mask=0) exactly, up to float rounding.
    """
    pc, weights = _scene(), _weights()
    count = pc["position"].shape[0]
    mask = (torch.rand(count, generator=torch.Generator().manual_seed(3)) < 0.5)
    mask = mask.float().unsqueeze(-1).cuda()
    _, grads = _gradients(pc, weights, blend_mask=mask)
    slope = grads["blend_mask"].squeeze(-1)
    chosen = (slope.abs() > 1e-3 * slope.abs().max()).nonzero(as_tuple=True)[0][:8]
    assert chosen.numel() == 8
    for index in chosen.tolist():
        with torch.no_grad():
            on, off = mask.clone(), mask.clone()
            on[index], off[index] = 1.0, 0.0
            difference = (
                _linear_loss(_render(pc, blend_mask=on), weights)
                - _linear_loss(_render(pc, blend_mask=off), weights)
            ).item()
        for value in (0.0, 0.5, 1.0):
            at = mask.clone()
            at[index] = value
            _, grads_at = _gradients(pc, weights, blend_mask=at)
            measured = grads_at["blend_mask"][index].item()
            assert abs(measured - difference) <= 1e-3 * abs(difference) + 1e-5, (
                index, value, measured, difference,
            )


@requires_cuda
def test_reduce_tokens_train_the_keep_gate_through_masked_rasterization():
    """V15's reduce tokens sent their router nothing; V16's send the keep gate
    the slope of inserting G2/G3 and nothing to the add gate."""
    from src.models.module.m3_g2p import DynamicGausRender

    torch.manual_seed(9)
    head = _head().cuda().train()
    with torch.no_grad():
        # Spread each token's five Gaussians around its anchor and make them
        # visible, so inserting G2..G5 changes the render.
        head.decoder[-1].weight.normal_(0.0, 0.5)
    tokens = 64
    azimuth = torch.rand(tokens) * 2 * math.pi
    distance = 6 + 20 * torch.rand(tokens)
    anchor = torch.stack(
        [distance * azimuth.cos(), distance * azimuth.sin(), torch.rand(tokens) - 1.0], 1
    ).cuda()
    _force_gates(head, [False] * tokens, [True] * tokens)
    captured = {}

    def keep_gate_logits(_module, _inputs, output):
        output.retain_grad()
        captured["gate_logits"] = output

    head.router.register_forward_hook(keep_gate_logits)
    routed, _auxiliary, routing = head(torch.randn(tokens, DIM).cuda(), anchor)
    assert routing["option"].eq(REDUCE).all()
    raw = routed["raw"]
    raw["opacity"].retain_grad()
    gaussians = [{
        "position": routed["position"],
        "opacity": raw["opacity"],
        "scaling": raw["scaling"],
        "rotation": raw["rotation"],
        "shs": raw["shs"],
        "blend_mask": routed["blend_mask"],
        "velocity": torch.zeros_like(routed["position"]),
        "source_time_sec": torch.zeros(routed["position"].shape[0], device="cuda"),
    }]
    renderer = DynamicGausRender(_RENDER_CFG).train()
    camera = _camera()
    camera.pts_depth = torch.full((32, 512), 15.0)
    renders = renderer(gaussians, {"cameras": [[camera]]}, compute_points=False)
    (renders["depth"] - renders["gt_depth"]).abs().mean().backward()
    gradient = captured["gate_logits"].grad
    assert (gradient[:, 0] != 0).float().mean() > 0.5
    assert torch.count_nonzero(gradient[:, 1]) == 0
    assert head.router[-1].weight.grad[0].abs().sum() > 0
    assert torch.count_nonzero(head.router[-1].weight.grad[1]) == 0
    # Only G1 is rendered, so this render trains only G1's attributes.
    slot = routed["slot_index"]
    opacity_grad = raw["opacity"].grad.squeeze(-1)
    assert torch.count_nonzero(opacity_grad[slot > 0]) == 0
    assert torch.count_nonzero(opacity_grad[slot == 0]) > 0
