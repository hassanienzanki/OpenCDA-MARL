# SUMO-MARL Config Family

This folder is the SUMO-only family for fast-turnaround experiments: the
same intersection and learners without CARLA.

Use it explicitly with:

```bash
pixi run python opencda.py -t sumo-marl/intersection/sac --marl
```

Notes:

- `route_*` profiles are the SUMO subset of the path + speed example family (`momarl/`); `sac` and `random_speed` are single-learner examples.
- Keep it separate from `marl/` so SUMO-specific overrides do not leak into CARLA-first configs.
- Prefer validating reward-contract changes here before pushing them into heavier CARLA runs.
- Keep `intersection` as the scenario filename and `runtime.backend: sumo` as the runtime selector.
- Use `cooperative_intersection_v1` when a run is meant to align with the simple cooperative CARLA benchmark.
- Use scenario folders such as `intersection/` so the path itself identifies the scenario family.
