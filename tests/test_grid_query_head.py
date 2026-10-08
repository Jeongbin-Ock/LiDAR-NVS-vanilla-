from types import SimpleNamespace
import unittest

import torch

from src.models.module.builders.common import (
    GridSeedConfig,
    _aggregate_points_to_cells,
    aggregate_points_to_cells_with_membership,
    aggregate_points_to_cells_with_seeds,
)
from src.models.module.builders.grid_tokens import OccupiedGridTokenBuilder
from src.models.module.gaussian_assembly import gradient_scale_identity
from src.models.module.grid_query_head import GridSlotHead
from src.models.utils.loss import Loss


def _cfg(**overrides):
    values = {"K_max": 3, "grad_balance": "sqrt_k"}
    values.update(overrides)
    return SimpleNamespace(**values)


def _gs_params(**overrides):
    values = {
        "shs": 2,
        "opacity": 1,
        "scaling": 1,
        "rotation": 1,
        "offset": 1,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class _UnitMapper:
    coord_scale = 1.0
    feature_grid_size = 1.0

    @staticmethod
    def to_feature_grid(points, metric_origin):
        return points - metric_origin

    @staticmethod
    def metric_origin(input_coord, input_grid_coord):
        return input_coord.new_zeros(3)


def _aggregate_seed_cell(
    points,
    k_max=3,
    voxel_coord=None,
    count_mode="legacy",
    seed_mode="range_quantile",
):
    intensity = torch.arange(1, points.shape[0] + 1, dtype=points.dtype)
    grid_coord = torch.tensor([[0, 0, 0]])
    voxel_feature = torch.tensor([[3.0, 4.0]])
    if voxel_coord is None:
        voxel_coord = torch.tensor([[0.5, 0.5, 0.5]])
    return aggregate_points_to_cells_with_seeds(
        points, intensity, grid_coord, voxel_feature, voxel_coord,
        torch.zeros(3), _UnitMapper(),
        GridSeedConfig(
            count_mode=count_mode,
            k_max=k_max,
            seed_mode=seed_mode,
        ),
    )


def _builder_cfg(**grid_overrides):
    return SimpleNamespace(grid_query=_cfg(**grid_overrides))


def _learned_cfg(k_max=4, tau=1.0, **overrides):
    return _cfg(
        count_mode="learned_gumbel",
        learned_count=SimpleNamespace(
            K_max=k_max,
            tau=tau,
            seed_mode="range_quantile",
        ),
        **overrides,
    )


class _FixedKHead(GridSlotHead):
    """Route every token to a scripted K while keeping the soft backward."""

    selected_k = None

    def _gumbel_selection(self, logits):
        selected = torch.as_tensor(
            self.selected_k, device=logits.device, dtype=torch.long
        ) - 1
        soft = torch.softmax(logits / self.gumbel_tau, dim=-1)
        hard = torch.zeros_like(soft).scatter_(1, selected[:, None], 1.0)
        return hard - soft.detach() + soft


def _fixed_k_head(selected_k, **cfg_overrides):
    k_max = cfg_overrides.pop("k_max", 4)
    head = _FixedKHead(_learned_cfg(k_max=k_max, **cfg_overrides), _gs_params(), dim=8)
    head.selected_k = list(selected_k)
    return head


def _non_opacity(head, raw_params):
    """Every output column except opacity, which also carries router gradient."""
    return torch.cat([
        raw_params[:, :head.opacity_start], raw_params[:, head.opacity_end:],
    ], dim=-1)


def test_builder_raw_count_is_aligned_with_occupied_token_order():
    class Mapper:
        coord_scale = 1.0
        feature_grid_size = 1.0

        @staticmethod
        def to_feature_grid(points, metric_origin):
            return points - metric_origin

    points = torch.tensor([
        [0.1, 0.1, 0.1], [0.2, 0.1, 0.1], [1.1, 0.1, 0.1],
    ])
    intensity = torch.tensor([1.0, 3.0, 7.0])
    grid_coord = torch.tensor([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
    voxel_feature = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    voxel_coord = torch.tensor([
        [0.5, 0.5, 0.5], [1.5, 0.5, 0.5], [2.5, 0.5, 0.5],
    ])
    pos, feature, _, occupied, raw_count = _aggregate_points_to_cells(
        points, intensity, grid_coord, voxel_feature, voxel_coord,
        torch.zeros(3), Mapper(),
    )
    assert occupied.tolist() == [True, True, False]
    assert raw_count.tolist() == [2, 1]
    torch.testing.assert_close(feature, voxel_feature[:2])
    torch.testing.assert_close(pos, voxel_coord[:2])


def test_grid_builder_returns_frame_local_seeds():
    lidar_points = torch.tensor([
        [0.10, 0.1, 0.1, 1.0], [0.20, 0.1, 0.1, 2.0],
        [0.70, 0.1, 0.1, 3.0], [0.80, 0.1, 0.1, 4.0],
    ])
    offset = torch.tensor([2, 4])
    features = {
        "feat": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        "grid_coord": torch.tensor([[0, 0, 0], [0, 0, 0]]),
        "coord": torch.tensor([[0.4, 0.4, 0.4], [0.6, 0.4, 0.4]]),
        "offset": torch.tensor([1, 2]),
    }
    ptv3_input = {
        "coord": torch.tensor([[0.1, 0.1, 0.1], [0.7, 0.1, 0.1]]),
        "grid_coord": torch.tensor([[0, 0, 0], [0, 0, 0]]),
        "offset": torch.tensor([1, 2]),
    }

    grid_out = OccupiedGridTokenBuilder(_builder_cfg(K_max=1))(
        lidar_points, offset, None, features, ptv3_input, _UnitMapper()
    )
    seed_data = grid_out.grid_seeds
    assert len(seed_data) == 2
    assert [count.tolist() for count in grid_out.raw_counts] == [[2], [2]]
    torch.testing.assert_close(seed_data[0].seed_sensor[0, 0], lidar_points[1, :3])
    torch.testing.assert_close(seed_data[1].seed_sensor[0, 0], lidar_points[3, :3])
    assert seed_data[0].anchor_k.tolist() == [1]
    assert seed_data[1].anchor_k.tolist() == [1]


def test_raw_membership_preserves_input_points_and_compact_token_rows():
    points = torch.tensor([
        [1.10, 0.1, 0.1], [0.10, 0.1, 0.1],
        [1.20, 0.1, 0.1], [0.20, 0.1, 0.1],
    ])
    grid_coord = torch.tensor([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
    voxel_feature = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    voxel_coord = torch.tensor([
        [0.5, 0.5, 0.5], [1.5, 0.5, 0.5], [2.5, 0.5, 0.5],
    ])
    result = aggregate_points_to_cells_with_membership(
        points, torch.ones(4), grid_coord, voxel_feature, voxel_coord,
        torch.zeros(3), _UnitMapper(),
    )
    _, _, _, occupied, raw_count, membership = result

    assert occupied.tolist() == [True, True, False]
    assert raw_count.tolist() == [2, 2]
    torch.testing.assert_close(membership.points_sensor, points)
    assert membership.token_index.tolist() == [1, 0, 1, 0]


def test_r_quantile_seeds_for_k1_k2_k3():
    points = torch.tensor([
        [0.05, 0.1, 0.1],
        [0.15, 0.1, 0.1],
        [0.25, 0.1, 0.1],
        [0.35, 0.1, 0.1],
        [0.45, 0.1, 0.1],
        [0.55, 0.1, 0.1],
    ])
    seed_k1 = _aggregate_seed_cell(points, k_max=1)[-1].seed_sensor
    seed_k2 = _aggregate_seed_cell(points, k_max=2)[-1].seed_sensor
    seed_k3 = _aggregate_seed_cell(points, k_max=3)[-1].seed_sensor

    torch.testing.assert_close(seed_k1[0, 0], points[3])
    torch.testing.assert_close(seed_k2[0], points[[1, 4]])
    torch.testing.assert_close(seed_k3[0], points[[1, 3, 5]])


def test_medoid_mode_seeds_the_observed_centroid_medoid():
    points = torch.tensor([
        [0.05, 0.1, 0.1], [0.15, 0.1, 0.1], [0.25, 0.1, 0.1],
        [0.35, 0.1, 0.1], [0.45, 0.1, 0.1],
    ])
    seed_data = _aggregate_seed_cell(points, k_max=1, seed_mode="medoid")[-1]

    # The mean is x=0.25, so the observed medoid is that raw point, not a
    # synthetic cell center.
    assert seed_data.anchor_k.tolist() == [1]
    torch.testing.assert_close(seed_data.seed_sensor[0, 0], points[2])


def test_token_mode_anchors_every_slot_at_the_token_position():
    points = torch.tensor([[0.1, 0.1, 0.1], [0.9, 0.1, 0.1]])
    token_coord = torch.tensor([[0.2, 0.3, 0.4]])
    seed_data = _aggregate_seed_cell(
        points, k_max=2, voxel_coord=token_coord, seed_mode="token"
    )[-1]

    assert seed_data.anchor_k.tolist() == [2]
    torch.testing.assert_close(seed_data.seed_sensor[0], token_coord.expand(2, -1))
    torch.testing.assert_close(
        seed_data.delta_sensor[0],
        (token_coord - torch.tensor([[0.5, 0.5, 0.5]])).expand(2, -1),
    )


def test_learned_count_builds_k1_to_k4_range_quantile_seed_bank():
    points = torch.tensor([
        [0.05, 0.1, 0.1], [0.15, 0.1, 0.1],
        [0.25, 0.1, 0.1], [0.35, 0.1, 0.1],
        [0.45, 0.1, 0.1], [0.55, 0.1, 0.1],
        [0.65, 0.1, 0.1], [0.75, 0.1, 0.1],
    ])
    seed_data = _aggregate_seed_cell(
        points,
        k_max=4,
        count_mode="learned_gumbel",
        seed_mode="range_quantile",
    )[-1]

    assert seed_data.anchor_k is None
    assert seed_data.seed_sensor.shape == (1, 4, 4, 3)
    # N=8: K1->[4], K2->[2,6], K3->[1,4,6], K4->[1,3,5,7].
    expected_rows = ([4], [2, 6], [1, 4, 6], [1, 3, 5, 7])
    for candidate_index, rows in enumerate(expected_rows):
        k = candidate_index + 1
        torch.testing.assert_close(
            seed_data.seed_sensor[0, candidate_index, :k],
            points[torch.tensor(rows)],
        )
        torch.testing.assert_close(
            seed_data.delta_sensor[0, candidate_index, :k],
            points[torch.tensor(rows)] - torch.tensor([0.5, 0.5, 0.5]),
        )
        torch.testing.assert_close(
            seed_data.seed_sensor[0, candidate_index, k:],
            torch.zeros(4 - k, 3),
        )


def test_fixed_count_gives_every_token_k_max_medoid_slots():
    points = torch.tensor([[0.1, 0.1, 0.1], [0.9, 0.9, 0.9], [0.5, 0.5, 0.5]])
    seed_data = _aggregate_seed_cell(points, k_max=3, seed_mode="medoid")[-1]
    medoid = seed_data.seed_sensor[0, 0]
    assert seed_data.anchor_k.tolist() == [3]
    torch.testing.assert_close(
        seed_data.seed_sensor[0], medoid.expand(3, -1)
    )
    # The delta is measured from the geometric cell center for every slot.
    torch.testing.assert_close(
        seed_data.delta_sensor[0],
        (medoid - torch.tensor([0.5, 0.5, 0.5])).expand(3, -1),
    )


def test_fixed_count_gives_every_token_k_max_token_coordinate_slots():
    points = torch.tensor([[0.1, 0.1, 0.1], [0.9, 0.9, 0.9]])
    token_coord = torch.tensor([[0.3, 0.4, 0.6]])
    seed_data = _aggregate_seed_cell(
        points, k_max=3, voxel_coord=token_coord, seed_mode="token",
    )[-1]
    assert seed_data.anchor_k.tolist() == [3]
    torch.testing.assert_close(
        seed_data.seed_sensor[0], token_coord.expand(3, -1)
    )


def test_fixed_count_range_quantiles_ignore_the_raw_point_total():
    """A fixed K spreads K quantile seeds however many points a token holds."""
    sparse = torch.tensor([[0.10, 0.1, 0.1], [0.20, 0.1, 0.1]])
    dense = torch.tensor([
        [0.10, 0.1, 0.1], [0.20, 0.1, 0.1], [0.30, 0.1, 0.1],
        [0.40, 0.1, 0.1], [0.50, 0.1, 0.1], [0.60, 0.1, 0.1],
    ])
    for points in (sparse, dense):
        seed_data = _aggregate_seed_cell(points, k_max=4)[-1]
        assert seed_data.anchor_k.tolist() == [4]
        assert seed_data.seed_sensor.shape == (1, 4, 3)
    # Two observations for four slots repeat observed points: ranks [0,0,1,1].
    sparse_seed = _aggregate_seed_cell(sparse, k_max=4)[-1]
    torch.testing.assert_close(
        sparse_seed.seed_sensor[0], sparse[[0, 0, 1, 1]]
    )


def test_absent_grid_query_is_one_medoid_gaussian_per_token():
    """An absent grid_query block keeps the historical fixed-count contract."""
    builder = OccupiedGridTokenBuilder(SimpleNamespace())
    assert builder.grid_seed_config == GridSeedConfig(
        count_mode="legacy", k_max=1, seed_mode="medoid",
    )
    assert builder.grid_seed_config.gaussians_per_token == 1


def test_fixed_count_builder_reads_k_max_from_config():
    builder = OccupiedGridTokenBuilder(_builder_cfg(K_max=3, seed_mode="medoid"))
    assert builder.grid_seed_config.gaussians_per_token == 3


def test_learned_count_has_no_single_gaussian_count():
    builder = OccupiedGridTokenBuilder(SimpleNamespace(grid_query=_learned_cfg()))
    try:
        builder.grid_seed_config.gaussians_per_token
    except ValueError as error:
        assert "learned count router" in str(error)
    else:
        raise AssertionError("a learned count must not report a fixed count")


def test_grid_builder_seed_mode_switches_only_fixed_count_seeds():
    grid_builder = OccupiedGridTokenBuilder(_builder_cfg(seed_mode="token"))
    assert grid_builder.grid_seed_config == GridSeedConfig(
        count_mode="legacy", k_max=3, seed_mode="token",
    )
    assert not grid_builder.grid_seed_config.distinct_slot_seeds
    learned_builder = OccupiedGridTokenBuilder(_builder_cfg(
        count_mode="learned_gumbel",
        # These invalid fixed-count values prove this branch does not read them.
        K_max=-1,
        exp=99,
        learned_count=SimpleNamespace(
            K_max=4, tau=1.0, seed_mode="range_quantile",
        ),
    ))
    assert learned_builder.grid_seed_config == GridSeedConfig(
        count_mode="learned_gumbel", k_max=4, seed_mode="range_quantile",
    )


def test_token_mode_anchors_a_single_slot_too():
    points = torch.tensor([[0.1, 0.1, 0.1], [0.9, 0.1, 0.1]])
    token_coord = torch.tensor([[0.2, 0.3, 0.4]])
    seed_data = _aggregate_seed_cell(
        points, k_max=1, voxel_coord=token_coord, seed_mode="token"
    )[-1]
    assert seed_data.anchor_k.tolist() == [1]
    torch.testing.assert_close(seed_data.seed_sensor[0], token_coord)


def test_learned_token_bank_repeats_the_anchor_in_every_row():
    points = torch.tensor([[0.1, 0.1, 0.1], [0.9, 0.1, 0.1]])
    token_coord = torch.tensor([[0.2, 0.3, 0.4]])
    seed_data = _aggregate_seed_cell(
        points,
        k_max=3,
        voxel_coord=token_coord,
        count_mode="learned_gumbel",
        seed_mode="token",
    )[-1]

    assert seed_data.anchor_k is None
    assert seed_data.seed_sensor.shape == (1, 3, 3, 3)
    center = torch.tensor([0.5, 0.5, 0.5])
    for k in (1, 2, 3):
        torch.testing.assert_close(
            seed_data.seed_sensor[0, k - 1, :k], token_coord.expand(k, -1)
        )
        torch.testing.assert_close(
            seed_data.delta_sensor[0, k - 1, :k],
            (token_coord - center).expand(k, -1),
        )
        torch.testing.assert_close(
            seed_data.seed_sensor[0, k - 1, k:], torch.zeros(3 - k, 3)
        )


def test_fixed_count_rejects_the_retired_exp_key():
    try:
        OccupiedGridTokenBuilder(_builder_cfg(exp=2))
    except ValueError as error:
        assert "renamed to p2g.grid_query.seed_mode" in str(error)
    else:
        raise AssertionError("the retired exp key must not be read silently")


def test_router_seed_modes_exclude_the_medoid():
    try:
        GridSeedConfig(count_mode="learned_gumbel", k_max=3, seed_mode="medoid")
    except ValueError as error:
        assert "seed_mode" in str(error)
    else:
        raise AssertionError("the router has no medoid candidate bank")


def test_router_without_seed_conditioning_reads_the_feature_alone():
    torch.manual_seed(3)
    conditioned = _FixedKHead(_learned_cfg(k_max=3), _gs_params(), dim=8)
    anchored = _FixedKHead(
        _learned_cfg(k_max=3), _gs_params(), dim=8, seed_conditioned=False,
    )
    assert conditioned.trunk[0].in_features == 8 + 9
    assert anchored.trunk[0].in_features == 8
    assert not anchored.seed_conditioned
    anchored.selected_k = [2, 3]
    anchored.eval()

    feature = torch.randn(2, 8)
    offsets = torch.tensor([2])
    with torch.no_grad():
        first, _ = anchored(feature, torch.zeros(2, 3, 3, 3), offsets)
        second, _ = anchored(feature, torch.randn(2, 3, 3, 3), offsets)
    # The candidate bank still has to match the contract, but its values are
    # never read without seed conditioning.
    torch.testing.assert_close(first, second)


def test_equal_range_ties_use_xyz_order_and_ignore_input_permutation():
    points = torch.tensor([
        [0.10, 0.20, 0.0],
        [0.20, 0.10, 0.0],
    ])
    permuted = points[[1, 0]]
    seed_a = _aggregate_seed_cell(points, k_max=1)[-1].seed_sensor
    seed_b = _aggregate_seed_cell(permuted, k_max=1)[-1].seed_sensor

    # Equal ranges sort by x then y then z; rank floor(0.5 * 2)=1 selects row 1.
    torch.testing.assert_close(seed_a[0, 0], torch.tensor([0.20, 0.10, 0.0]))
    torch.testing.assert_close(seed_b, seed_a)


def test_seed_selection_is_isolated_per_frame():
    frame0 = torch.tensor([[0.10, 0.1, 0.1], [0.20, 0.1, 0.1]])
    frame1 = torch.tensor([[0.70, 0.1, 0.1], [0.80, 0.1, 0.1]])
    seed0 = _aggregate_seed_cell(frame0, k_max=1)[-1].seed_sensor
    seed1 = _aggregate_seed_cell(frame1, k_max=1)[-1].seed_sensor

    torch.testing.assert_close(seed0[0, 0], frame0[1])
    torch.testing.assert_close(seed1[0, 0], frame1[1])
    assert not torch.equal(seed0, seed1)


def test_seed_rows_follow_occupied_token_order_across_cells():
    points = torch.tensor([
        [1.10, 0.1, 0.1], [0.10, 0.1, 0.1],
        [1.20, 0.1, 0.1], [0.20, 0.1, 0.1],
    ])
    grid_coord = torch.tensor([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
    voxel_feature = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    voxel_coord = torch.tensor([
        [0.4, 0.5, 0.5], [1.4, 0.5, 0.5], [2.4, 0.5, 0.5],
    ])
    result = aggregate_points_to_cells_with_seeds(
        points, torch.ones(4), grid_coord, voxel_feature, voxel_coord,
        torch.zeros(3), _UnitMapper(),
        GridSeedConfig(count_mode="legacy", k_max=1),
    )
    _, feature, _, occupied, raw_count, seed_data = result
    seeds = seed_data.seed_sensor

    assert occupied.tolist() == [True, True, False]
    assert raw_count.tolist() == [2, 2]
    torch.testing.assert_close(feature, voxel_feature[:2])
    torch.testing.assert_close(seeds[:, 0], torch.stack([points[3], points[2]]))


def test_delta_uses_geometric_center_not_pooled_position():
    points = torch.tensor([[0.1, 0.1, 0.1], [0.8, 0.6, 0.4]])
    pooled_position = torch.tensor([[0.2, 0.2, 0.2]])
    pos, _, _, _, _, seed_data = _aggregate_seed_cell(
        points, k_max=1, voxel_coord=pooled_position
    )
    seeds = seed_data.seed_sensor
    delta = seed_data.delta_sensor

    torch.testing.assert_close(pos, pooled_position)
    torch.testing.assert_close(seeds[0, 0], points[1])
    torch.testing.assert_close(delta[0, 0], points[1] - torch.tensor([0.5, 0.5, 0.5]))
    assert not torch.equal(delta[0, 0], points[1] - pooled_position[0])


def test_k_heads_derive_the_production_parameter_width_from_gs_params():
    production_params = _gs_params(shs=32, scaling=2, rotation=4, offset=3)
    head = GridSlotHead(_learned_cfg(k_max=3), production_params, dim=8)
    assert head.param_dim == 42
    assert [module.out_features for module in head.k_heads] == [42, 84, 126]


def test_learned_count_builds_one_head_per_k_and_a_post_fusion_router():
    learned = GridSlotHead(_learned_cfg(), _gs_params(), dim=8)

    assert learned.count_predictor[-1].out_features == 4
    assert [module.out_features for module in learned.k_heads] == [6, 12, 18, 24]
    assert learned.trunk[0].in_features == 8 + 4 * 3


def test_router_rejects_a_fixed_count_config():
    try:
        GridSlotHead(_cfg(), _gs_params(), dim=8)
    except ValueError as error:
        assert "learned_gumbel" in str(error)
    else:
        raise AssertionError("the router must not accept a fixed count")


def test_scale_diagnostic_is_logged_but_never_enters_the_total():
    loss_fn = Loss(SimpleNamespace(
        w_chamfer=0.0,
        w_depth=0.0,
        w_depth_median=0.0,
        w_intensity=0.0,
        w_raydrop=0.0,
        scale_max_m=2.5,
    ))
    assert not hasattr(loss_fn, "w_scale")
    raw_scaling = torch.tensor([[4.0, 4.0], [0.0, 0.0]])
    diagnostic = loss_fn._scale_diagnostic(
        [{"scaling": raw_scaling}], torch.zeros(())
    )
    # softplus(4) = 4.018 m > 2.5 m; the second Gaussian does not violate.
    expected = (torch.log(torch.nn.functional.softplus(torch.tensor(4.0)))
                - torch.log(torch.tensor(2.5))).square()
    torch.testing.assert_close(diagnostic, expected)
    assert float(loss_fn._scale_diagnostic(None, torch.zeros(()))) == 0.0


def test_learned_count_eval_argmax_routes_exactly_to_k2_joint_head():
    head = GridSlotHead(_learned_cfg(), _gs_params(), dim=8)
    head.eval()
    with torch.no_grad():
        head.count_predictor[-1].weight.zero_()
        head.count_predictor[-1].bias.copy_(
            torch.tensor([-3.0, 5.0, -2.0, -1.0])
        )
        for k, k_head in enumerate(head.k_heads, start=1):
            k_head.weight.zero_()
            bias = torch.arange(k, dtype=torch.float32).add_(10 * k)
            k_head.bias.copy_(
                bias[:, None].expand(k, head.param_dim).reshape(-1)
            )

    raw_params, packing = head(
        torch.randn(2, 8),
        torch.zeros(2, 4, 4, 3),
        torch.tensor([2]),
    )

    assert packing["anchor_k"].tolist() == [2, 2]
    assert packing["gaussian_offset"].tolist() == [4]
    torch.testing.assert_close(
        packing["k_selection"],
        torch.tensor([[0.0, 1.0, 0.0, 0.0]]).expand(2, -1),
    )
    expected = torch.tensor([20.0, 21.0, 20.0, 21.0])
    torch.testing.assert_close(
        raw_params, expected[:, None].expand(-1, head.param_dim)
    )


def test_learned_count_selected_opacity_gate_trains_router_and_only_k3_head():
    torch.manual_seed(17)
    head = _fixed_k_head([3], tau=0.7)
    feature = torch.randn(1, 8, requires_grad=True)
    raw_params, packing = head(
        feature,
        torch.randn(1, 4, 4, 3),
        torch.tensor([1]),
    )

    assert packing["anchor_k"].tolist() == [3]
    assert torch.all((packing["k_selection"] == 0) | (packing["k_selection"] == 1))
    assert torch.allclose(
        packing["k_selection"].sum(dim=-1),
        torch.ones(1),
    )
    torch.testing.assert_close(
        packing["packed_selected_gate"],
        torch.ones(3),
    )

    packing["k_logits"].retain_grad()
    opacity = raw_params[:, head.opacity_start:head.opacity_end]
    torch.sigmoid(opacity).sum().backward()

    predictor_grad = head.count_predictor[-1].weight.grad
    assert predictor_grad is not None and predictor_grad.abs().sum() > 0
    assert packing["k_logits"].grad is not None
    assert torch.all(packing["k_logits"].grad.abs() > 0)
    assert feature.grad is not None and feature.grad.abs().sum() > 0
    assert head.k_heads[0].weight.grad is None
    assert head.k_heads[1].weight.grad is None
    assert head.k_heads[2].weight.grad is not None
    assert head.k_heads[2].weight.grad.abs().sum() > 0
    assert head.k_heads[3].weight.grad is None


def test_learned_count_handles_empty_frames():
    head = GridSlotHead(_learned_cfg(), _gs_params(), dim=8)
    raw_params, packing = head(
        torch.zeros(0, 8),
        torch.zeros(0, 4, 4, 3),
        torch.tensor([0, 0]),
    )

    assert raw_params.shape == (0, head.param_dim)
    assert packing["k_selection"].shape == (0, 4)
    assert packing["gaussian_offset"].tolist() == [0, 0]


def test_k_specific_heads_route_mixed_tokens_to_slot_flat_order():
    head = _fixed_k_head([2, 1, 3, 2], k_max=3)
    for k, k_head in enumerate(head.k_heads, start=1):
        torch.nn.init.zeros_(k_head.weight)
        bias = torch.arange(k, dtype=torch.float32).add_(10 * k)
        with torch.no_grad():
            k_head.bias.copy_(bias[:, None].expand(k, head.param_dim).reshape(-1))

    raw_params, packing = head(
        torch.randn(4, 8), torch.zeros(4, 3, 3, 3), torch.tensor([1, 4])
    )
    expected = torch.tensor([20, 21, 10, 30, 31, 32, 20, 21], dtype=torch.float32)
    torch.testing.assert_close(
        _non_opacity(head, raw_params),
        _non_opacity(head, expected[:, None].expand(-1, head.param_dim)),
    )
    assert packing["anchor_index"].tolist() == [0, 0, 1, 2, 2, 2, 3, 3]
    assert packing["slot_index"].tolist() == [0, 1, 0, 0, 1, 2, 0, 1]
    assert packing["slot_k"].tolist() == [2, 2, 1, 3, 3, 3, 2, 2]
    # Frame 0 holds token 0 (K=2); frame 1 holds tokens 1-3 (K=1+3+2).
    assert packing["gaussian_offset"].tolist() == [2, 8]


def test_k_head_backward_reaches_trunk_features_and_only_selected_head():
    torch.manual_seed(11)
    head = _fixed_k_head([2, 2], k_max=3)
    feature = torch.randn(2, 8, requires_grad=True)
    raw_params, _ = head(
        feature, torch.randn(2, 3, 3, 3), torch.tensor([2])
    )
    _non_opacity(head, raw_params).square().sum().backward()

    assert feature.grad is not None and feature.grad.abs().sum() > 0
    assert head.trunk[0].weight.grad is not None
    assert head.trunk[0].weight.grad.abs().sum() > 0
    assert head.k_heads[0].weight.grad is None
    assert head.k_heads[1].weight.grad is not None
    assert head.k_heads[1].weight.grad.abs().sum() > 0
    assert head.k_heads[2].weight.grad is None


def test_gradient_balance_is_forward_identity_on_every_final_output():
    slot_k = torch.tensor([1.0, 2.0, 3.0])
    weight = slot_k.rsqrt()
    widths = {"position": 3, "shs": 32, "opacity": 1, "scaling": 2, "rotation": 4}
    tensors = {
        name: torch.randn(3, width, requires_grad=True)
        for name, width in widths.items()
    }
    balanced = {
        name: gradient_scale_identity(value, weight)
        for name, value in tensors.items()
    }
    for name in tensors:
        torch.testing.assert_close(balanced[name], tensors[name])
    sum(value.sum() for value in balanced.values()).backward()
    for value in tensors.values():
        expected = weight[:, None].expand_as(value)
        torch.testing.assert_close(value.grad, expected)


def test_missing_grad_balance_scope_preserves_historical_output_balancing():
    head = GridSlotHead(_learned_cfg(), _gs_params(), dim=8)
    assert head.grad_balance_scope == "output"
    torch.testing.assert_close(
        head.gradient_weight(torch.tensor([1, 2, 4]), torch.float32),
        torch.tensor([1.0, 2.0**-0.5, 0.5]),
    )


def test_token_grad_balance_scales_only_shared_token_path():
    torch.manual_seed(29)
    unbalanced = _fixed_k_head(
        [4, 4], grad_balance="none", grad_balance_scope="token",
    )
    balanced = _fixed_k_head(
        [4, 4], grad_balance="sqrt_k", grad_balance_scope="token",
    )
    balanced.load_state_dict(unbalanced.state_dict())

    base_feature = torch.randn(2, 8)
    feature_unbalanced = base_feature.clone().requires_grad_()
    feature_balanced = base_feature.clone().requires_grad_()
    delta = torch.randn(2, 4, 4, 3)
    raw_unbalanced, _ = unbalanced(
        feature_unbalanced, delta, torch.tensor([2])
    )
    raw_balanced, packing = balanced(
        feature_balanced, delta, torch.tensor([2])
    )

    # The balancing operator is a backward-only identity.
    torch.testing.assert_close(raw_balanced, raw_unbalanced, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        balanced.gradient_weight(packing["slot_k"], raw_balanced.dtype),
        torch.ones(8),
    )

    # Opacity also carries the router's gradient, which token balancing leaves
    # alone, so only the other outputs isolate the shared token path.
    _non_opacity(unbalanced, raw_unbalanced).square().sum().backward()
    _non_opacity(balanced, raw_balanced).square().sum().backward()

    # K-head parameters see the same training signal. Only the gradient
    # returning through the shared trunk/token feature is scaled by 1/sqrt(4).
    torch.testing.assert_close(
        balanced.k_heads[3].weight.grad,
        unbalanced.k_heads[3].weight.grad,
    )
    torch.testing.assert_close(
        balanced.k_heads[3].bias.grad,
        unbalanced.k_heads[3].bias.grad,
    )
    torch.testing.assert_close(
        balanced.trunk[0].weight.grad,
        0.5 * unbalanced.trunk[0].weight.grad,
    )
    torch.testing.assert_close(
        balanced.trunk[0].bias.grad,
        0.5 * unbalanced.trunk[0].bias.grad,
    )
    torch.testing.assert_close(
        feature_balanced.grad,
        0.5 * feature_unbalanced.grad,
    )


def test_token_grad_balance_does_not_scale_learned_router_gradient():
    torch.manual_seed(31)
    unbalanced = _fixed_k_head(
        [4, 4], grad_balance="none", grad_balance_scope="token",
    )
    balanced = _fixed_k_head(
        [4, 4], grad_balance="sqrt_k", grad_balance_scope="token",
    )
    balanced.load_state_dict(unbalanced.state_dict())

    base_feature = torch.randn(2, 8)
    delta = torch.randn(2, 4, 4, 3)
    outputs = []
    packings = []
    for head in (unbalanced, balanced):
        feature = base_feature.clone().requires_grad_()
        raw_params, packing = head(feature, delta, torch.tensor([2]))
        packing["k_logits"].retain_grad()
        opacity = raw_params[:, head.opacity_start:head.opacity_end]
        torch.sigmoid(opacity).sum().backward()
        outputs.append(raw_params.detach())
        packings.append(packing)

    torch.testing.assert_close(outputs[1], outputs[0], rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        packings[1]["k_logits"].grad,
        packings[0]["k_logits"].grad,
    )
    torch.testing.assert_close(
        balanced.count_predictor[-1].weight.grad,
        unbalanced.count_predictor[-1].weight.grad,
    )
    torch.testing.assert_close(
        balanced.count_predictor[-1].bias.grad,
        unbalanced.count_predictor[-1].bias.grad,
    )


class GridQueryHeadTests(unittest.TestCase):
    """Expose the function-style checks to Python's dependency-free test runner."""


for _name, _function in list(globals().items()):
    if _name.startswith("test_") and callable(_function):
        setattr(GridQueryHeadTests, _name, staticmethod(_function))
del _name, _function


if __name__ == "__main__":
    unittest.main(verbosity=2)
