"""Configuration contract for occupied-grid seed generation."""
from __future__ import annotations

from dataclasses import dataclass

from src.experiment_config.definitions import (
    FIXED_COUNT_SEED_MODES,
    LEARNED_COUNT_SEED_MODES,
    ROUTED_COUNT_BLOCKS,
    SEED_MODE_MEDOID,
    SEED_MODE_RANGE_QUANTILE,
    SEED_MODE_TOKEN,
)


@dataclass(frozen=True)
class GridSeedConfig:
    """How many seeds each occupied token gets, and where they are placed.

    ``count_mode="legacy"`` gives *every* token exactly ``k_max`` seeds in one
    padded ``(N, K_max, 3)`` set, which is what a fixed-count Gaussian head
    needs: it owns one parameter block per slot, so the slot count cannot vary
    from token to token. ``learned_gumbel`` defers the count to a router that
    runs after temporal fusion and stores a ``(N, K_max, K_max, 3)`` candidate
    bank instead.

    ``seed_mode`` selects the placement rule. ``range_quantile`` spreads the
    seeds over the token's raw points by range quantile, ``medoid`` puts every
    slot on the token's observed medoid, and ``token`` puts every slot on the
    token's Utonia position. The last two are shared anchors: independent
    predictor outputs and learned offsets separate the slots, and only
    ``range_quantile`` gives a decoder per-slot seed geometry to read.

    V15's ``nested_gumbel`` and V15.2's ``expert_gumbel`` counts are not
    further ``count_mode`` values here: their tokens have one shared anchor,
    the Utonia token position, and the router picks 1, 3 or 5 Gaussians after
    refinement, so the builder only has to supply that one seed per token.
    ``resolve_grid_seed_config`` maps both to
    ``count_mode="legacy", k_max=1, seed_mode="token"`` -- a padded
    ``(N, 1, 3)`` token-position seed set.
    """

    count_mode: str
    k_max: int
    seed_mode: str = SEED_MODE_RANGE_QUANTILE

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
        allowed = (
            LEARNED_COUNT_SEED_MODES if self.learned else FIXED_COUNT_SEED_MODES
        )
        if self.seed_mode not in allowed:
            raise ValueError(
                f"grid seed_mode for count_mode={self.count_mode!r} must be one "
                f"of {allowed}, got {self.seed_mode!r}"
            )

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

    @property
    def distinct_slot_seeds(self) -> bool:
        """Whether the slots of one token start from different seeds."""
        return self.seed_mode == SEED_MODE_RANGE_QUANTILE


def resolve_grid_seed_config(cfg):
    """Read ``p2g.grid_query`` into the seed contract the token builder needs.

    Omitting the ``grid_query`` block means one Gaussian per token seeded at its
    observed medoid, which is what the single-Gaussian variant ships with. A
    fixed-count block without ``seed_mode`` keeps range-quantile seeds.
    ``count_mode: nested_gumbel`` (V15) and ``expert_gumbel`` (V15.2) give
    every token the single token-position anchor its head decodes around; the
    1/3/5 choice is made later by the router, so no per-count seed set exists.
    """
    grid_query = getattr(cfg, "grid_query", None)
    if grid_query is None:
        return GridSeedConfig(
            count_mode="legacy", k_max=1, seed_mode=SEED_MODE_MEDOID
        )

    count_mode = str(getattr(grid_query, "count_mode", "legacy")).lower()
    if count_mode in ROUTED_COUNT_BLOCKS:
        block = ROUTED_COUNT_BLOCKS[count_mode]
        routed_count = getattr(grid_query, block, None)
        if routed_count is None:
            raise ValueError(
                f"p2g.grid_query.{block} is required for "
                f"count_mode={count_mode!r}"
            )
        seed_mode = str(
            getattr(routed_count, "seed_mode", SEED_MODE_TOKEN)
        ).lower()
        if seed_mode != SEED_MODE_TOKEN:
            raise ValueError(
                f"p2g.grid_query.{block}.seed_mode must be "
                f"{SEED_MODE_TOKEN!r}, got {seed_mode!r}: routed tokens have "
                "one shared token-position anchor"
            )
        return GridSeedConfig(
            count_mode=GridSeedConfig.LEGACY_COUNT_MODE, k_max=1,
            seed_mode=SEED_MODE_TOKEN,
        )
    if count_mode not in (
        GridSeedConfig.LEGACY_COUNT_MODE, *GridSeedConfig.LEARNED_COUNT_MODES
    ):
        raise ValueError(
            "p2g.grid_query.count_mode must be one of 'legacy', "
            "'learned_gumbel', "
            + ", ".join(repr(mode) for mode in ROUTED_COUNT_BLOCKS)
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
            getattr(learned_count, "seed_mode", SEED_MODE_RANGE_QUANTILE)
        ).lower()
        if seed_mode not in LEARNED_COUNT_SEED_MODES:
            raise ValueError(
                "p2g.grid_query.learned_count.seed_mode must be one of "
                f"{LEARNED_COUNT_SEED_MODES}, got {seed_mode!r}"
            )
        # The fixed-count keys are deliberately not read here: K is predicted
        # after temporal feature fusion from a bank with one row per K.
        return GridSeedConfig(
            count_mode=count_mode, k_max=int(k_max), seed_mode=seed_mode,
        )

    if _declares(grid_query, "exp"):
        raise ValueError(
            "p2g.grid_query.exp was renamed to p2g.grid_query.seed_mode "
            "(null -> range_quantile, 1 -> medoid, 2 -> token); restore "
            "checkpoint configs through src.config_loader, which rewrites it"
        )
    k_max = getattr(grid_query, "K_max", None)
    if k_max is None:
        raise ValueError("p2g.grid_query.K_max is required for fixed-count seeds")
    k_max = int(k_max)
    if k_max <= 0:
        raise ValueError("p2g.grid_query.K_max must be positive")
    seed_mode = getattr(grid_query, "seed_mode", None)
    seed_mode = (
        SEED_MODE_RANGE_QUANTILE if seed_mode is None else str(seed_mode).lower()
    )
    if seed_mode not in FIXED_COUNT_SEED_MODES:
        raise ValueError(
            "p2g.grid_query.seed_mode must be one of "
            f"{FIXED_COUNT_SEED_MODES}, got {seed_mode!r}"
        )
    return GridSeedConfig(count_mode="legacy", k_max=k_max, seed_mode=seed_mode)


def _declares(block, key: str) -> bool:
    """Whether a mapping-like or namespace config block spells ``key``."""
    try:
        return key in block
    except TypeError:
        return key in vars(block)


__all__ = ["GridSeedConfig", "resolve_grid_seed_config"]
