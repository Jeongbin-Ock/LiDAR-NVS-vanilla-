"""Detached, fixed-size statistics for learned Gaussian-count routing."""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.distributed as dist


def _validated_range_edges(range_edges_m: Sequence[float]) -> tuple[float, ...]:
    edges = tuple(float(value) for value in range_edges_m)
    if not edges or edges[0] != 0.0:
        raise ValueError("routing range_edges_m must start at 0")
    if any(right <= left for left, right in zip(edges, edges[1:])):
        raise ValueError("routing range_edges_m must be strictly increasing")
    return edges


def routing_option_counts(
    routing: Mapping[str, torch.Tensor],
    option_count: int,
    device=None,
) -> torch.Tensor:
    """How many Gaussians each routing option renders.

    The learned_gumbel router's option ``k - 1`` renders ``k`` Gaussians and
    publishes no mapping. V15's nested router publishes ``option_counts``
    ([1, 3, 5] for reduce/keep/add).
    """
    counts = routing.get("option_counts")
    if counts is None:
        return torch.arange(1, int(option_count) + 1, device=device)
    counts = torch.as_tensor(counts, device=device).long()
    if tuple(counts.shape) != (int(option_count),):
        raise ValueError(
            "routing option_counts must list one count per router logit"
        )
    return counts


def routing_sufficient_statistics(
    routing: Mapping[str, torch.Tensor],
    range_edges_m: Sequence[float],
) -> dict[str, torch.Tensor]:
    """Summarize learned-count routing without retaining its autograd graph.

    Produces the option distribution, the Gaussian count it implies, the
    router's expected count and top-2 logit margin, and the count's breakdown
    by token range. A router reports its sampled choice either as
    ``selected_option`` (an index into the logits) or, like the
    learned_gumbel K={1,2,3} router, as ``selected_k = option + 1``.
    """
    with torch.no_grad():
        logits = routing["k_logits"].detach().float()

        if logits.ndim != 2 or logits.shape[1] < 2:
            raise ValueError("routing k_logits must have shape (N, K) with K >= 2")
        token_count, k_max = logits.shape
        expected_shape = (token_count,)
        if "selected_option" in routing:
            selected_index = routing["selected_option"].detach().long()
        else:
            selected_index = routing["selected_k"].detach().long() - 1
        if tuple(selected_index.shape) != expected_shape:
            raise ValueError("routing selection must align with k_logits")

        dtype = logits.dtype
        device = logits.device
        option_counts = routing_option_counts(
            routing, k_max, device=device
        ).to(dtype)
        argmax_index = logits.argmax(dim=-1)
        if token_count > 0 and (
            (selected_index < 0).any() or (selected_index >= k_max).any()
        ):
            raise ValueError("routing selection contains an invalid option")
        sampled_k = option_counts[selected_index]
        argmax_k = option_counts[argmax_index]

        selected_counts = torch.bincount(
            selected_index, minlength=k_max
        ).to(dtype)
        argmax_counts = torch.bincount(
            argmax_index, minlength=k_max
        ).to(dtype)
        probability = torch.softmax(logits, dim=-1)
        top_two = logits.topk(2, dim=-1).values
        statistics = {
            "token_count": logits.new_tensor(float(token_count)),
            "selected_counts": selected_counts,
            "argmax_counts": argmax_counts,
            "sampled_k_sum": sampled_k.sum(),
            "argmax_k_sum": argmax_k.sum(),
            "expected_k_sum": (probability * option_counts).sum(),
            "probability_sums": probability.sum(dim=0),
            "logit_margin_sum": (top_two[:, 0] - top_two[:, 1]).sum(),
        }
        edges = _validated_range_edges(range_edges_m)
        token_position_local = routing["token_position_local"].detach().float()
        if tuple(token_position_local.shape) != (token_count, 3):
            raise ValueError(
                "routing token_position_local must have shape (N, 3)"
            )

        ranges = token_position_local.norm(dim=-1)
        boundaries = torch.tensor(
            edges[1:], device=device, dtype=ranges.dtype
        )
        range_index = torch.bucketize(ranges, boundaries, right=True)
        num_range_bins = len(edges)

        def scatter_sum(values, index, size):
            output = torch.zeros(size, device=device, dtype=dtype)
            if values.numel() > 0:
                output.scatter_add_(0, index, values.to(dtype))
            return output

        ones = torch.ones(token_count, device=device, dtype=dtype)
        range_token_counts = scatter_sum(
            ones, range_index, num_range_bins
        )
        range_sampled_k_sums = scatter_sum(
            sampled_k, range_index, num_range_bins
        )
        range_argmax_k_sums = scatter_sum(
            argmax_k, range_index, num_range_bins
        )
        statistics.update({
            "range_token_counts": range_token_counts,
            "range_sampled_k_sums": range_sampled_k_sums,
            "range_argmax_k_sums": range_argmax_k_sums,
        })
        return statistics


def distributed_sum_statistics(
    statistics: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Sum a statistics mapping across DDP ranks with one collective."""
    keys = tuple(statistics.keys())
    shapes = {key: statistics[key].shape for key in keys}
    sizes = {key: statistics[key].numel() for key in keys}
    flat = torch.cat([statistics[key].reshape(-1) for key in keys])
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)

    reduced = {}
    start = 0
    for key in keys:
        end = start + sizes[key]
        reduced[key] = flat[start:end].reshape(shapes[key])
        start = end
    return reduced


def range_bin_labels(range_edges_m: Sequence[float]) -> tuple[str, ...]:
    edges = _validated_range_edges(range_edges_m)
    labels = []
    for index, lower in enumerate(edges):
        if index + 1 < len(edges):
            upper = edges[index + 1]
            labels.append(f"{lower:g}_{upper:g}m")
        else:
            labels.append(f"{lower:g}_infm")
    return tuple(labels)


__all__ = [
    "distributed_sum_statistics",
    "range_bin_labels",
    "routing_option_counts",
    "routing_sufficient_statistics",
]
