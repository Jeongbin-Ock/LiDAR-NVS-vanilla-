"""Configuration contract for occupied-grid seed generation."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GridSeedConfig:
    """How many seeds each occupied token gets, and where they are placed.

    ``count_mode="legacy"`` gives *every* token exactly ``k_max`` seeds in one
    padded ``(N, K_max, 3)`` set, which is what a fixed-count Gaussian head
    needs: it owns one parameter block per slot, so the slot count cannot vary
    from token to token. ``learned_gumbel`` defers the count to a router that
    runs after temporal fusion and stores a ``(N, K_max, K_max, 3)`` candidate
    bank instead.

    ``exp`` selects the fixed-count placement rule: ``None`` spreads the seeds
    over the token's raw points by range quantile, ``1`` puts them on the
    token's observed medoid, and ``2`` puts them on the Utonia cell coordinate.
    Repeated slots start at the same coordinate on purpose; independent
    predictor outputs and learned offsets separate them.
    """

    count_mode: str
    k_max: int
    exp: int | None = None
    seed_mode: str | None = None

    LEGACY_COUNT_MODE = "legacy"
    LEARNED_COUNT_MODES = ("learned_gumbel",)

    def __post_init__(self):
        if self.count_mode not in (
            self.LEGACY_COUNT_MODE, *self.LEARNED_COUNT_MODES
        ):
            raise ValueError(
                "grid count_mode must be 'legacy' or 'learned_gumbel'"
            )
        if int(self.k_max) <= 0:
            raise ValueError("grid K_max must be positive")
        if self.exp is not None and int(self.exp) not in (1, 2):
            raise ValueError("grid seed exp must be null, 1, or 2")

    @property
    def learned(self) -> bool:
        return self.count_mode in self.LEARNED_COUNT_MODES

    @property
    def gaussians_per_token(self) -> int:
        """Slots every token fills, for the modes where that is a constant."""
        if self.learned:
            raise ValueError(
                "a learned count router has no single Gaussian count per token"
            )
        return int(self.k_max)


def resolve_grid_seed_config(cfg):
    """Read ``p2g.grid_query`` into the seed contract the token builder needs.

    Omitting the ``grid_query`` block means one Gaussian per token seeded at its
    observed medoid, which is what the single-Gaussian variant ships with.
    """
    grid_query = getattr(cfg, "grid_query", None)
    if grid_query is None:
        return GridSeedConfig(count_mode="legacy", k_max=1, exp=1)

    count_mode = str(getattr(grid_query, "count_mode", "legacy")).lower()
    if count_mode not in (
        GridSeedConfig.LEGACY_COUNT_MODE, *GridSeedConfig.LEARNED_COUNT_MODES
    ):
        raise ValueError(
            "p2g.grid_query.count_mode must be 'legacy' or 'learned_gumbel'"
        )

    if count_mode != GridSeedConfig.LEGACY_COUNT_MODE:
        learned_count = getattr(grid_query, "learned_count", None)
        if learned_count is None:
            raise ValueError(
                "p2g.grid_query.learned_count is required for "
                f"count_mode={count_mode!r}"
            )
        k_max = getattr(learned_count, "K_max", None)
        if k_max is None or int(k_max) <= 0:
            raise ValueError(
                "p2g.grid_query.learned_count.K_max must be positive"
            )
        seed_mode = str(
            getattr(learned_count, "seed_mode", "range_quantile")
        ).lower()
        if seed_mode != "range_quantile":
            raise ValueError(
                "p2g.grid_query.learned_count.seed_mode currently supports "
                "only 'range_quantile'"
            )
        # exp is deliberately not read here: K is predicted after temporal
        # feature fusion and every candidate row uses range quantiles.
        return GridSeedConfig(
            count_mode=count_mode, k_max=int(k_max), seed_mode=seed_mode,
        )

    k_max = getattr(grid_query, "K_max", None)
    if k_max is None:
        raise ValueError("p2g.grid_query.K_max is required for fixed-count seeds")
    k_max = int(k_max)
    if k_max <= 0:
        raise ValueError("p2g.grid_query.K_max must be positive")
    exp = getattr(grid_query, "exp", None)
    exp = None if exp is None else int(exp)
    if exp is not None and exp not in (1, 2):
        raise ValueError("p2g.grid_query.exp must be null, 1, or 2")
    if exp is not None and exp > k_max:
        raise ValueError(f"p2g.grid_query.exp={exp} requires K_max >= {exp}")
    return GridSeedConfig(count_mode="legacy", k_max=k_max, exp=exp)


__all__ = ["GridSeedConfig", "resolve_grid_seed_config"]
