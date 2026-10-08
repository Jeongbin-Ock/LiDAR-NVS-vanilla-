# Grid token builder module map

The occupied-grid builder follows one-way dependencies:

```text
grid_geometry ──► grid_seeds ──► token_aggregation ──► grid_intensity
                                                       ▲
grid_seed_config ──────────────────────────────────────┘
```

`grid_intensity` resolves the seed config, then passes that config object
through token aggregation to the seed generator.

- `grid_geometry.py` owns metric coordinates, Utonia grid mapping, and cell
  centers.
- `grid_seed_config.py` validates the config-selected seed/count contract:
  a fixed count per token, the learned-count candidate bank, or a V15-family
  routed count (V15's `nested_gumbel`, V15.2's `expert_gumbel`), which has one
  shared token-position anchor per token (a `legacy`, `K_max=1`,
  `seed_mode=token` config) because its router picks 1, 3 or 5 Gaussians only
  after refinement. V15.3 is an ordinary fixed count (`K_max=5`,
  `seed_mode=token`).
- `grid_seeds.py` owns deterministic seed placement and learned-count banks.
- `token_aggregation.py` maps raw points to occupied token rows and optionally
  attaches membership or seeds.
- `grid_intensity.py` composes those pieces with the selected intensity encoder.
- `common.py` explicitly preserves the historical helper import path.

The split does not change function or class bodies. `GridSeedConfig`,
`GridSeedData`, and `RawTokenMembership` remain reachable through
`builders.common`, which keeps older pickle lookups and downstream imports
resolvable.
