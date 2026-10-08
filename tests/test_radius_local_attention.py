"""The V15 radius local-attention block: neighbor search, math, packing.

Everything above the CUDA section at the end runs on the CPU fallback path,
with tiny blocks (dim 72, two 36-wide heads: the production head width).
Importing ``src.models_new.module`` still loads a CUDA extension, so one GPU
must be visible even for the CPU tests.

Random clouds are drawn until no pair distance lies within 1e-4 m of the
radius, so float32 versus float64 rounding can never flip a neighbor decision;
the boundary itself is tested separately on dyadic coordinates, where every
difference and square is exact.
"""
import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from src.models_new.module.dynamic import local_attention
from src.models_new.module.dynamic.local_attention import (
    MAX_QUERY_CHUNK_SIZE,
    RadiusLocalAttentionBlock,
)

DIM = 72
HEADS = 2
RADIUS = 0.8

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is unavailable"
)
requires_flash = pytest.mark.skipif(
    not torch.cuda.is_available() or local_attention.flash_attn is None,
    reason="CUDA and flash_attn are required",
)


def _block(dim=DIM, **overrides):
    values = {"num_heads": HEADS, "implementation": "sdpa"}
    values.update(overrides)
    return RadiusLocalAttentionBlock(SimpleNamespace(**values), dim)


def _randomize(block, seed=0, std=0.1):
    """Random weights everywhere, including the zero-init FFN output layer."""
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, parameter in block.named_parameters():
            noise = torch.randn(parameter.shape, generator=generator) * std
            parameter.copy_(noise + 1.0 if name.endswith("norm.weight") else noise)
    return block


def _pairwise_distance(position):
    diff = position.double()[:, None, :] - position.double()[None, :, :]
    return torch.linalg.vector_norm(diff, dim=-1)


def _clear_of_radius(position, sample, radius=RADIUS):
    """No same-sample pair lies within 1e-4 m of the radius."""
    same = sample[:, None] == sample[None, :]
    return bool(((_pairwise_distance(position) - radius).abs()[same] > 1.0e-4).all())


def _draw(sizes, generator, extent):
    """Uniform positions in a cube; sample ids interleaved by a random shuffle."""
    total = sum(sizes)
    position = torch.rand(total, 3, generator=generator) * extent
    sample = torch.repeat_interleave(torch.arange(len(sizes)), torch.tensor(sizes))
    order = torch.randperm(total, generator=generator)
    return position[order], sample[order]


def _cloud(sizes, seed=0, extent=3.0):
    """Positions and interleaved sample ids, clear of the radius."""
    generator = torch.Generator().manual_seed(seed)
    while True:
        position, sample = _draw(sizes, generator, extent)
        if _clear_of_radius(position, sample):
            return position, sample


def _features(n_tokens, seed=1, dim=DIM):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(n_tokens, dim, generator=generator)


def _brute_force_csr(position, sample, radius=RADIUS):
    """Float64 all-pairs neighbor lists in CSR form, straight from the definition."""
    distance = _pairwise_distance(position)
    hit = (distance <= radius) & (sample[:, None] == sample[None, :])
    counts = hit.sum(dim=1)
    offsets = torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(0)])
    return offsets, hit.nonzero(as_tuple=True)[1]


def _capture_attended(block):
    """Record what the block feeds ``out_proj``: the attention output."""
    captured = {}

    def hook(_module, args):
        captured["attended"] = args[0].detach().clone()

    block.out_proj.register_forward_pre_hook(hook)
    return captured


def _explicit_rope(x, position, base, scale):
    """Float64 re-derivation of Utonia's per-axis half rotation.

    ``x`` is ``(heads, head_dim)``. Each axis owns a third of the channels and
    rotates the first half of them against the second by
    ``position[axis] * scale * base ** (-2j / axis_dim)``.
    """
    axis_dim = x.shape[-1] // 3
    half = axis_dim // 2
    rotated = torch.empty_like(x)
    frequency = base ** (-2.0 * torch.arange(half, dtype=torch.float64) / axis_dim)
    for axis in range(3):
        lo = axis * axis_dim
        first = x[:, lo:lo + half]
        second = x[:, lo + half:lo + axis_dim]
        angle = position[axis] * scale * frequency
        rotated[:, lo:lo + half] = first * angle.cos() - second * angle.sin()
        rotated[:, lo + half:lo + axis_dim] = (
            first * angle.sin() + second * angle.cos()
        )
    return rotated


def _reference_forward(block, feature, position, sample):
    """Float64 forward that loops over queries and only reads the weights."""
    w = {name: p.detach().double() for name, p in block.named_parameters()}
    x = feature.double()
    pos = position.double()
    n_tokens, dim = x.shape
    heads, head_dim = block.num_heads, block.head_dim
    base, scale = block.rope.base, block.rope.position_scale

    h = F.layer_norm(
        x, (dim,), w["input_norm.weight"], w["input_norm.bias"],
        eps=block.input_norm.eps,
    )
    q = F.linear(h, w["q_proj.weight"], w["q_proj.bias"]).view(n_tokens, heads, head_dim)
    k = F.linear(h, w["k_proj.weight"], w["k_proj.bias"]).view(n_tokens, heads, head_dim)
    v = F.linear(h, w["v_proj.weight"], w["v_proj.bias"]).view(n_tokens, heads, head_dim)
    q = torch.stack([_explicit_rope(q[i], pos[i], base, scale) for i in range(n_tokens)])
    k = torch.stack([_explicit_rope(k[i], pos[i], base, scale) for i in range(n_tokens)])

    attended = torch.empty(n_tokens, heads, head_dim, dtype=torch.float64)
    for i in range(n_tokens):
        near = (
            (torch.linalg.vector_norm(pos - pos[i], dim=-1) <= block.radius_m)
            & (sample == sample[i])
        ).nonzero().flatten()
        logits = torch.einsum("hd,khd->hk", q[i], k[near]) * head_dim ** -0.5
        attended[i] = torch.einsum("hk,khd->hd", logits.softmax(dim=-1), v[near])

    x = x + F.linear(
        attended.reshape(n_tokens, dim), w["out_proj.weight"], w["out_proj.bias"]
    )
    y = F.layer_norm(
        x, (dim,), w["ffn_norm.weight"], w["ffn_norm.bias"],
        eps=block.ffn_norm.eps,
    )
    y = F.silu(F.linear(y, w["ffn.0.weight"], w["ffn.0.bias"]))
    return x + F.linear(y, w["ffn.2.weight"], w["ffn.2.bias"])


# --- neighbor search -------------------------------------------------------


@pytest.mark.parametrize("chunk", [1, 7, 4096])
def test_radius_neighbors_match_brute_force(chunk):
    position, sample = _cloud([37, 29, 41], seed=1, extent=2.0)
    block = _block(query_chunk_size=chunk)
    offsets, index = block.radius_neighbors(position, sample)
    expected_offsets, expected_index = _brute_force_csr(position, sample)

    assert offsets.dtype == index.dtype == torch.int64
    assert torch.equal(offsets, expected_offsets)
    assert torch.equal(index, expected_index)

    n_tokens = position.shape[0]
    counts = offsets[1:] - offsets[:-1]
    assert int(offsets[0]) == 0 and int(offsets[-1]) == index.numel()
    assert bool((counts >= 1).all())
    query = torch.repeat_interleave(torch.arange(n_tokens), counts)
    # A token is always its own neighbor, exactly once, and no pair crosses
    # samples.
    assert torch.equal(
        torch.zeros(n_tokens).index_add_(0, query, (index == query).float()),
        torch.ones(n_tokens),
    )
    assert torch.equal(sample[query], sample[index])
    # The cloud is dense enough that the test sees real neighborhoods.
    assert float(counts.float().mean()) > 4.0 and int(counts.max()) > 8


def test_radius_neighbors_bound_is_inclusive():
    """Pairs exactly at the radius are in; one float32 step outside is out."""
    block = _block(radius_m=0.625)
    # Original token labels; rows below are a shuffled, sample-interleaved
    # layout of them.
    xyz = {
        "origin": [0.0, 0.0, 0.0],
        "on_axis": [0.625, 0.0, 0.0],            # exactly at the radius
        "three_four_five": [0.375, 0.5, 0.0],    # |.| = 0.625 exactly
        "just_outside": [0.0, 0.0, 0.625 + 2.0 ** -20],
        "other_sample_origin": [0.0, 0.0, 0.0],  # same spot, another sample
        "other_sample_near": [0.125, 0.0, 0.0],
    }
    owner = {
        "origin": 0, "on_axis": 0, "three_four_five": 0, "just_outside": 0,
        "other_sample_origin": 1, "other_sample_near": 1,
    }
    rows = [
        "origin", "other_sample_origin", "on_axis", "three_four_five",
        "other_sample_near", "just_outside",
    ]
    position = torch.tensor([xyz[name] for name in rows])
    sample = torch.tensor([owner[name] for name in rows])
    expected = {
        "origin": {"origin", "on_axis", "three_four_five"},
        "on_axis": {"origin", "on_axis", "three_four_five"},
        "three_four_five": {"origin", "on_axis", "three_four_five"},
        "just_outside": {"just_outside"},
        "other_sample_origin": {"other_sample_origin", "other_sample_near"},
        "other_sample_near": {"other_sample_origin", "other_sample_near"},
    }

    offsets, index = block.radius_neighbors(position, sample)
    for row, name in enumerate(rows):
        found = {rows[int(j)] for j in index[offsets[row]:offsets[row + 1]]}
        assert found == expected[name], name


def test_radius_neighbors_validate_inputs():
    block = _block()
    position, sample = _cloud([10], seed=0)
    with pytest.raises(ValueError, match="float"):
        block.radius_neighbors(position[:, :2], sample)
    with pytest.raises(ValueError, match="integer"):
        block.radius_neighbors(position, sample.float())
    with pytest.raises(ValueError, match="integer"):
        block.radius_neighbors(position, sample[:-1])
    bad = position.clone()
    bad[3, 1] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        block.radius_neighbors(bad, sample)
    offsets, index = block.radius_neighbors(position[:0], sample[:0])
    assert offsets.tolist() == [0] and index.numel() == 0


# --- forward math ----------------------------------------------------------


def test_forward_matches_float64_reference():
    position, sample = _cloud([31, 27, 22], seed=2)
    block = _randomize(_block(), seed=3)
    feature = _features(position.shape[0])

    out = block(feature, position, sample)
    expected = _reference_forward(block, feature, position, sample)

    assert out.dtype == torch.float32
    # Attention and the FFN both move the features well past the tolerance.
    assert float((expected - feature.double()).abs().max()) > 0.3
    torch.testing.assert_close(out.double(), expected, rtol=2e-5, atol=2e-5)


def test_forward_keeps_feature_dtype():
    position, sample = _cloud([20, 20], seed=2)
    block = _randomize(_block(), seed=3).double()
    out = block(_features(40).double(), position, sample)
    assert out.dtype == torch.float64 and out.shape == (40, DIM)


def test_translation_invariance():
    """RoPE is relative, so moving a whole sample changes nothing."""
    position, sample = _cloud([31, 27, 22], seed=2)
    block = _randomize(_block(), seed=3)
    feature = _features(position.shape[0])
    shift = torch.tensor([[7.5, -3.25, 1.0], [-12.0, 4.5, 0.25], [2.0, 9.0, -6.5]])

    moved = position + shift[sample]
    assert torch.equal(
        block.radius_neighbors(position, sample)[1],
        block.radius_neighbors(moved, sample)[1],
    )
    torch.testing.assert_close(
        block(feature, moved, sample), block(feature, position, sample),
        rtol=5e-5, atol=5e-5,
    )


def test_isolated_token_receives_its_own_value():
    position, sample = _cloud([20, 20], seed=4)
    # Token 0 is 50 m from everything in its sample; a token of the other
    # sample sits at exactly the same spot and must not count.
    position[0] = torch.tensor([50.0, 50.0, 50.0])
    other = int((sample != sample[0]).nonzero()[0])
    position[other] = position[0]
    block = _randomize(_block(), seed=5)
    feature = _features(40)
    captured = _capture_attended(block)

    out = block(feature, position, sample)

    v_self = block.v_proj(block.input_norm(feature)).detach()
    torch.testing.assert_close(
        captured["attended"][0], v_self[0], rtol=0.0, atol=1e-6
    )
    x = feature[0] + block.out_proj(v_self[0])
    expected = x + block.ffn(block.ffn_norm(x))
    torch.testing.assert_close(out[0], expected, rtol=1e-5, atol=1e-5)
    offsets, index = block.radius_neighbors(position, sample)
    assert int(offsets[1] - offsets[0]) == 1 and int(index[offsets[0]]) == 0


def test_ffn_contributes_exactly_zero_at_init():
    block = _block()
    assert not block.ffn[-1].weight.any() and not block.ffn[-1].bias.any()
    assert not block.ffn(_features(16) * 5.0).any()

    position, sample = _cloud([25, 25], seed=6)
    feature = _features(50)
    captured = _capture_attended(block)
    out = block(feature, position, sample)
    assert torch.equal(out, feature + block.out_proj(captured["attended"]))


def test_default_rope_wavelengths():
    for cfg in (None, OmegaConf.create({}), SimpleNamespace()):
        block = RadiusLocalAttentionBlock(cfg, 432)
        assert (block.num_heads, block.head_dim) == (12, 36)
        wavelength = 2 * math.pi / (block.rope.position_scale * block.rope.inv_freq)
        torch.testing.assert_close(
            wavelength,
            torch.tensor([0.400, 0.634, 1.005, 1.592, 2.524, 4.000]),
            rtol=1e-3, atol=0.0,
        )
        # One voxel at the short end; the long band cannot wrap in the radius.
        assert float(wavelength[0]) == pytest.approx(0.4, rel=1e-6)
        assert float(wavelength[-1]) / 2 > block.radius_m


def test_gradients_reach_every_parameter():
    position, sample = _cloud([31, 27, 22], seed=2)
    block = _randomize(_block(), seed=3)
    with torch.no_grad():
        block.ffn[-1].weight.zero_()
        block.ffn[-1].bias.zero_()
    feature = _features(position.shape[0]).requires_grad_()
    weights = _features(position.shape[0], seed=9)

    (block(feature, position, sample) * weights).sum().backward()
    # A zero-initialized last FFN layer blocks the gradient to what feeds it.
    for name in ("ffn_norm.weight", "ffn_norm.bias", "ffn.0.weight", "ffn.0.bias"):
        assert not dict(block.named_parameters())[name].grad.any(), name
    for name in ("input_norm", "q_proj", "k_proj", "v_proj", "out_proj"):
        for parameter in getattr(block, name).parameters():
            assert parameter.grad.abs().sum() > 0, name

    # Once the FFN is non-zero every parameter and the input learn.
    block.zero_grad()
    feature.grad = None
    with torch.no_grad():
        torch.nn.init.normal_(block.ffn[-1].weight, std=0.1)
        torch.nn.init.normal_(block.ffn[-1].bias, std=0.1)
    (block(feature, position, sample) * weights).sum().backward()
    for name, parameter in block.named_parameters():
        assert parameter.grad is not None, name
        assert bool(torch.isfinite(parameter.grad).all()), name
        assert parameter.grad.abs().sum() > 0, name
    assert bool(torch.isfinite(feature.grad).all()) and feature.grad.abs().sum() > 0


def test_gradients_match_finite_differences():
    """Autograd through the CSR gather and the padding mask, in float64.

    One central difference along a random direction per tensor: the input and
    every parameter.
    """
    generator = torch.Generator().manual_seed(40)
    block = _randomize(_block(dim=24, num_heads=2, query_chunk_size=5), seed=41, std=0.3)
    block = block.double()
    position = torch.rand(14, 3, generator=generator) * 1.5
    sample = torch.tensor([0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1])
    feature = torch.randn(14, 24, generator=generator, dtype=torch.float64)
    weights = torch.randn(14, 24, generator=generator, dtype=torch.float64)
    feature.requires_grad_()

    def objective():
        return (block(feature, position, sample) * weights).sum()

    objective().backward()
    eps = 1.0e-6
    for name, tensor in [("feature", feature), *block.named_parameters()]:
        direction = torch.randn(tensor.shape, generator=generator, dtype=torch.float64)
        analytic = float((tensor.grad * direction).sum())
        with torch.no_grad():
            tensor += eps * direction
            plus = float(objective())
            tensor -= 2.0 * eps * direction
            minus = float(objective())
            tensor += eps * direction
        assert (plus - minus) / (2.0 * eps) == pytest.approx(
            analytic, rel=1e-5, abs=1e-7
        ), name


# --- layout, isolation, chunking -------------------------------------------


def test_samples_never_attend_to_each_other():
    position, sample = _cloud([30, 30], seed=7)
    block = _randomize(_block(), seed=8)
    feature = _features(60)
    changed = feature.clone()
    changed[sample == 1] = _features(int((sample == 1).sum()), seed=99)

    out = block(feature, position, sample)
    out_changed = block(changed, position, sample)

    torch.testing.assert_close(
        out_changed[sample == 0], out[sample == 0], rtol=0.0, atol=1e-6
    )
    assert float((out_changed[sample == 1] - out[sample == 1]).abs().max()) > 0.1


def test_rows_of_one_sample_need_not_be_contiguous():
    position, sample = _cloud([30, 25, 20], seed=10)
    block = _randomize(_block(), seed=11)
    feature = _features(75)

    order = torch.argsort(sample, stable=True)
    inverse = torch.argsort(order)
    interleaved = block(feature, position, sample)
    grouped = block(feature[order], position[order], sample[order])[inverse]

    torch.testing.assert_close(interleaved, grouped, rtol=0.0, atol=1e-5)


def test_int32_sample_index_and_arbitrary_ids():
    position, sample = _cloud([25, 25], seed=12)
    block = _randomize(_block(), seed=13)
    feature = _features(50)
    reference = block(feature, position, sample)
    relabeled = torch.where(sample == 0, 1000, -7).to(torch.int32)
    torch.testing.assert_close(
        block(feature, position, relabeled), reference, rtol=0.0, atol=1e-6
    )


def test_query_chunk_size_does_not_change_the_result():
    position, sample = _cloud([40, 30], seed=14)
    feature = _features(70)
    reference = _randomize(_block(query_chunk_size=4096), seed=15)
    expected = reference(feature, position, sample)
    for chunk in (1, 5, 33):
        other = _block(query_chunk_size=chunk)
        other.load_state_dict(reference.state_dict())
        torch.testing.assert_close(
            other(feature, position, sample), expected, rtol=0.0, atol=1e-5
        )


def test_empty_and_single_token_windows():
    block = _randomize(_block(), seed=18)
    empty = block(torch.zeros(0, DIM), torch.zeros(0, 3), torch.zeros(0, dtype=torch.long))
    assert empty.shape == (0, DIM)
    offsets, index = block.radius_neighbors(
        torch.zeros(0, 3), torch.zeros(0, dtype=torch.long)
    )
    assert offsets.tolist() == [0] and index.numel() == 0

    feature = _features(1)
    out = block(feature, torch.zeros(1, 3), torch.zeros(1, dtype=torch.long))
    x = feature + block.out_proj(block.v_proj(block.input_norm(feature)))
    torch.testing.assert_close(
        out, x + block.ffn(block.ffn_norm(x)), rtol=1e-5, atol=1e-5
    )
    offsets, index = block.radius_neighbors(
        torch.zeros(1, 3), torch.zeros(1, dtype=torch.long)
    )
    assert offsets.tolist() == [0, 1] and index.tolist() == [0]


def test_forward_validates_inputs():
    block = _block()
    position, sample = _cloud([10], seed=0)
    feature = _features(10)
    with pytest.raises(ValueError, match="feature"):
        block(feature[:, :-1], position, sample)
    with pytest.raises(ValueError, match="position"):
        block(feature, position[:-1], sample)
    with pytest.raises(ValueError, match="sample_index"):
        block(feature, position, sample[:-1])
    with pytest.raises(ValueError, match="integer"):
        block(feature, position, sample.float())


# --- construction ----------------------------------------------------------


def test_state_dict_schema_is_the_checkpoint_contract():
    block = RadiusLocalAttentionBlock(None, 432)
    assert [name for name, _ in block.named_children()] == [
        "input_norm", "q_proj", "k_proj", "v_proj", "out_proj",
        "ffn_norm", "ffn", "rope",
    ]
    expected = [
        f"{module}.{name}"
        for module in ("input_norm", "q_proj", "k_proj", "v_proj", "out_proj", "ffn_norm")
        for name in ("weight", "bias")
    ] + ["ffn.0.weight", "ffn.0.bias", "ffn.2.weight", "ffn.2.bias"]
    state = block.state_dict()
    assert list(state) == expected  # the RoPE buffer is not persistent
    assert tuple(state["q_proj.weight"].shape) == (432, 432)
    assert tuple(state["ffn.0.weight"].shape) == (1728, 432)
    assert tuple(state["ffn.2.weight"].shape) == (432, 1728)
    assert sum(p.numel() for p in block.parameters()) == 2_245_104


def test_config_accepts_omegaconf_and_defaults():
    block = RadiusLocalAttentionBlock(
        OmegaConf.create({
            "radius_m": 0.5, "num_heads": 4, "mlp_ratio": 2.0,
            "implementation": "AUTO", "query_chunk_size": MAX_QUERY_CHUNK_SIZE,
        }),
        144,
    )
    assert (block.radius_m, block.num_heads, block.head_dim) == (0.5, 4, 36)
    assert block.ffn[0].out_features == 288 and block.implementation == "auto"
    assert block.query_chunk_size == 65535

    default = RadiusLocalAttentionBlock(None, 144)
    assert default.radius_m == 0.8 and default.num_heads == 12
    assert default.head_dim == 12 and default.implementation == "flash_varlen"
    assert default.query_chunk_size == 4096 and default.ffn[0].out_features == 576
    assert default.rope.base == pytest.approx(10 ** 1.2)
    assert default.rope.position_scale == pytest.approx(2 * math.pi / 0.4)


@pytest.mark.parametrize(
    "dim, overrides, message",
    [
        (72, {"radius_m": 0.0}, "radius_m must be positive"),
        (72, {"radius_m": -0.8}, "radius_m must be positive"),
        (72, {"radius_m": float("nan")}, "radius_m must be a finite number"),
        (72, {"radius_m": float("inf")}, "radius_m must be a finite number"),
        (72, {"radius_m": None}, "radius_m must be a finite number"),
        (72, {"radius_m": "wide"}, "radius_m must be a finite number"),
        (30, {"num_heads": 2}, "divisible by 6"),
        (432, {"num_heads": 16}, "divisible by 6"),
        (432, {"num_heads": 7}, "divisible by num_heads"),
        (72, {"num_heads": 0}, "must be positive"),
        (72, {"num_heads": 1.5}, "num_heads must be an integer"),
        (72, {"implementation": "triton"}, "implementation must be"),
        (72, {"query_chunk_size": MAX_QUERY_CHUNK_SIZE + 1}, "query_chunk_size must be in"),
        (72, {"query_chunk_size": 0}, "query_chunk_size must be in"),
        (72, {"query_chunk_size": 2.5}, "query_chunk_size must be an integer"),
        (72, {"mlp_ratio": 0.0}, "mlp_ratio must be positive"),
        (72, {"rope_base": 0.0}, "rope_base must be positive"),
        (72, {"rope_position_scale": -1.0}, "rope_position_scale must be positive"),
    ],
)
def test_constructor_rejects_bad_configs(dim, overrides, message):
    with pytest.raises(ValueError, match=message):
        RadiusLocalAttentionBlock(SimpleNamespace(**overrides), dim)


# --- CUDA: packed varlen versus the padded reference -----------------------


def _dense_cloud(seed=20):
    """Three samples; a 150-token ball gives queries more than 128 neighbors.

    No cap means such queries keep every key, across several key blocks of the
    kernel, next to the sparse ones.
    """
    generator = torch.Generator().manual_seed(seed)
    while True:
        position, sample = _draw([220, 150, 120], generator, 3.5)
        ball = (sample == 0).nonzero().flatten()[:150]
        position[ball] = 1.0 + 0.2 * torch.rand(150, 3, generator=generator)
        if _clear_of_radius(position, sample):
            return position, sample


def _relative_error(actual, expected):
    return float((actual - expected).norm() / expected.norm())


@requires_flash
def test_flash_matches_fallback_forward_and_gradients():
    position, sample = _dense_cloud()
    n_tokens = position.shape[0]
    chunk = 64  # eight chunks, so chunk boundaries are exercised
    fallback = _randomize(_block(query_chunk_size=chunk), seed=21)
    packed = _block(query_chunk_size=chunk, implementation="flash_varlen")
    packed.load_state_dict(fallback.state_dict())
    fallback, packed = fallback.cuda(), packed.cuda()

    counts = torch.diff(packed.radius_neighbors(position.cuda(), sample.cuda())[0])
    assert n_tokens > 4 * chunk and int(counts.max()) > 128

    weights = _features(n_tokens, seed=22).cuda()
    results = {}
    for name, block in (("fallback", fallback), ("packed", packed)):
        feature = _features(n_tokens, seed=23).cuda().requires_grad_()
        captured = _capture_attended(block)
        out = block(feature, position.cuda(), sample.cuda())
        (out * weights).sum().backward()
        results[name] = (
            captured["attended"], out.detach(), feature.grad,
            {n: p.grad for n, p in block.named_parameters()},
        )

    (att_s, out_s, grad_s, param_s), (att_f, out_f, grad_f, param_f) = (
        results["fallback"], results["packed"]
    )
    # bf16 inputs to the kernel: about 3 significant digits. Measured on an
    # A100: attention 2.7e-3, output 0.9e-3, input gradient 1.4e-3, parameter
    # gradients up to 4.9e-3. A 1% error in the softmax scale already exceeds
    # these bounds.
    assert _relative_error(att_f, att_s) < 5e-3
    assert _relative_error(out_f, out_s) < 2e-3
    assert _relative_error(grad_f, grad_s) < 4e-3
    for name in param_s:
        assert _relative_error(param_f[name], param_s[name]) < 1e-2, name


@requires_flash
def test_flash_chunking_does_not_change_the_result():
    position, sample = _dense_cloud(seed=24)
    reference = _randomize(_block(implementation="flash_varlen"), seed=25).cuda()
    feature = _features(position.shape[0], seed=26).cuda()
    expected = reference(feature, position.cuda(), sample.cuda())
    for chunk in (1, 37, 128):
        other = _block(implementation="flash_varlen", query_chunk_size=chunk).cuda()
        other.load_state_dict(reference.state_dict())
        # Every query is its own sequence, so this is bf16 noise at most.
        assert _relative_error(
            other(feature, position.cuda(), sample.cuda()), expected
        ) < 1e-3


@requires_flash
@pytest.mark.parametrize("n_tokens", [1, 2, 5])
def test_flash_handles_tiny_windows(n_tokens):
    """Few sequences make the kernel split its keys; the result must not change."""
    position, sample = _cloud([n_tokens], seed=30, extent=1.0)
    fallback = _randomize(_block(), seed=31)
    packed = _block(implementation="flash_varlen")
    packed.load_state_dict(fallback.state_dict())
    feature = _features(n_tokens, seed=32)
    expected = fallback(feature, position, sample)
    actual = packed.cuda()(feature.cuda(), position.cuda(), sample.cuda()).cpu()
    assert _relative_error(actual, expected) < 3e-3


@requires_cuda
def test_cuda_neighbors_equal_cpu_neighbors():
    position, sample = _dense_cloud(seed=27)
    block = _block(query_chunk_size=100)
    cpu = block.radius_neighbors(position, sample)
    gpu = block.radius_neighbors(position.cuda(), sample.cuda())
    assert torch.equal(cpu[0], gpu[0].cpu()) and torch.equal(cpu[1], gpu[1].cpu())
    assert torch.equal(cpu[1], _brute_force_csr(position, sample)[1])


@requires_cuda
def test_flash_implementation_requires_flash_attn(monkeypatch):
    monkeypatch.setattr(local_attention, "flash_attn", None)
    position, sample = _cloud([20, 20], seed=28)
    feature = _features(40).cuda()

    strict = _block(implementation="flash_varlen").cuda()
    with pytest.raises(RuntimeError, match="flash_attn is unavailable"):
        strict(feature, position.cuda(), sample.cuda())

    # "auto" falls back to the padded path on CUDA, and CPU tensors never need
    # the kernel, whatever the implementation says.
    expected = _randomize(_block(implementation="sdpa"), seed=29)
    reference = expected(feature.cpu(), position, sample)
    auto = _block(implementation="auto")
    auto.load_state_dict(expected.state_dict())
    torch.testing.assert_close(
        auto.cuda()(feature, position.cuda(), sample.cuda()).cpu(),
        reference, rtol=1e-4, atol=1e-4,
    )
    strict.load_state_dict(expected.state_dict())
    torch.testing.assert_close(
        strict.cpu()(feature.cpu(), position, sample), reference,
        rtol=1e-6, atol=1e-6,
    )
