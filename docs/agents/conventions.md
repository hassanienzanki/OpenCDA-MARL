# Conventions

Guide version 1.1 — OpenCDA-MARL 2.1 (2026-10)

Open this when naming something, bumping the version, or checking whether
an oddity you hit is already a known issue. The behaviour rules are in
[Agent Workflow](workflow.md); the map is the [Code Index](code-index.md).

- [Naming](#naming)
- [Versioning](#versioning)
- [Known issues](#known-issues)

## Naming

- Scenario references: `<family>/<scenario>/<profile>` with families
  `marl/`, `pcma/`, `momarl/`, `sumo-marl/`, `cosim-marl/`. SUMO
  route-intent profiles are `opencda_marl/_bundled/configs/sumo-marl/intersection/route_*.yaml`.
- Commit scopes in use: `feat(marl)`, `fix(traffic)`, `perf(sac)`,
  `docs(marl)`, `test(route_intent)`, `chore(repo)`, and similar.
- `momarl`, `pcma`, the `aligned` / `conflicting` regimes and the artifact
  slug shapes are feature names and part of the experiment vocabulary —
  renaming one is an API decision, not cleanup. `registry.label` (and the
  rest of `register_algorithm`'s signature) is documented public API;
  renaming it is a breaking change.
- Adding an algorithm must never require editing `opencda_marl/`: register
  it and declare capabilities. If a new dispatch site seems to need a name
  list or `isinstance` check, the capability is missing — add it instead.

## Versioning

- `major.minor.revision`. A verified patch-level fix bumps `revision` in
  the same commit that lands it; features bump minor, breaking changes
  bump major.
- The version is declared in three places that must agree — `pixi.toml`
  (`version`), `opencda_marl/__init__.py` (`__version__`) and
  `CITATION.cff` (`version`). `test/basic/test_docs_boundary.py` checks
  that they do; bump all three together.

## Known issues

Open items a contributor may hit; do not "fix" one by weakening an
assertion.

- Suite baseline: measure it with `pixi run pytest test -q` rather than
  trusting a recorded count; the expected skips are CUDA-graph headroom
  and live-CARLA cases.
- Console encoding: anything printed on the entry path (banners, launcher
  status) stays ASCII. A Windows console whose code page is not UTF-8
  raises `UnicodeEncodeError` at startup on any other character — that
  error is this, not a logic bug.
- Torch import order: `opencda.py` imports torch before the MARL runtime,
  and `opencda_marl.builtin.algorithms` registers its learners lazily (a
  learner module loads only when its factory is called), because importing
  a learner module before torch in a fresh process can fault while Windows
  loads torch's DLLs. New entry points and tests keep that order: torch
  first.
- Planner hard-coding: `opencda_marl/builtin/planning/local_planner.py`
  (`get_current_lane`) and `opencda/core/plan/local_planner_behavior.py`
  (`max_acc = 3.5`, "hard-coded, need to be tuned").
- SAC/TD3 soft-update `tau` is duplicated per family
  (`opencda_marl/builtin/algorithms/sac.py`, `td3.py`).
- SUMO spawn backpressure: `SumoMARLEnv` active-agent sync under heavy
  traffic (see `max_active_agents`).
- Deferred by choice in the vendored core: `weakref.ref(x)()`
  immediate-deref sites (`opencda/core/safety/safety_manager.py`,
  `v2x_manager.py`, `platooning_manager.py`) and stray `world.tick()`
  ownership at episode/reload boundaries.
- On CARLA, vehicles that time out have their route intents finalised
  after the episode's intent diagnostics were snapshotted, so those
  counters lag one episode.
