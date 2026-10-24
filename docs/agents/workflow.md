# Agent Workflow

Guide version 1.1 — OpenCDA-MARL 2.1 (2026-10)

Open this before changing code: it is how changes are made to OpenCDA-MARL,
written for coding agents (Claude Code, Codex, Cursor, Copilot, …) and
applying equally to people. Read
it with the [Code Index](code-index.md), which maps where things live, and
[Conventions](conventions.md) for naming, versioning and known issues.
Nothing here is decorative — each rule exists because its absence cost a
debugging session.

- [The map](#the-map)
- [The seams](#the-seams)
- [Commands](#commands)
- [Verification duties](#verification-duties)
- [Commits and pull requests](#commits-and-pull-requests)
- [Reporting](#reporting)

## The map

The [Code Index](code-index.md) is the module-level map; this table is the
edit policy per area.

| path | what it is | how to treat it |
|---|---|---|
| `opencda_marl/` | the framework, in four zones: the sealed kernel (`_kernel/`), the public surface (`api/`), the plugin system (`plugins/`), the built-in plugin set (`builtin/`), plus facades (`envs/`) and services (`telemetry/`, `evaluation/`, `utils/`, `tools/`) | keep the seams below; the architecture test enforces the zone boundaries |
| `opencda_marl/api/` | everything third parties import: contracts (`policy.py`, `contracts.py`, `events.py`) and every `register_*` door | public API — a change here updates [Add an Algorithm](../marl/custom_algorithms.md) in the same commit and may bump the plugin API version |
| `opencda_marl/_kernel/` | the complete lifecycle — sealed | never imports a plugin; a new behaviour is a registration, never a new `if algorithm == …`. External code and plugins never import it |
| `opencda/` | vendored OpenCDA core with local patches | every modification is a row in the [patch ledger](../opencda/marl_patches.md) |
| `opencda_marl/_bundled/configs/<family>/` | `default.yaml` → `<scenario>/base.yaml` → profile; the package's own data, shipped in the wheel | profiles state only what they change |
| `artifacts/_runtime/` | per-cell profiles written by the matrix launcher, resolved as an extra config root | generated and gitignored — never hand-edit |
| `scripts/experiments/` | matrix launcher, evaluation launcher, analyzers | the experiment-matrix tooling in [Run Experiments](../run-experiments.md#the-experiment-matrix) |
| `opencda_marl/_bundled/recordings/*.json` | deterministic traffic replays | evaluation ground truth: changing one invalidates every comparison made on it — do not modify |
| `artifacts/`, `checkpoints/`, `output/` | run outputs (gitignored) | possibly someone's only copy — inventory before deleting |
| `test/` | pytest suites (`basic`, `marl`, `opencda`, `route_intent`) | every change ships with its test |
| `docs/` | this portal (`mkdocs.yml` nav is the table of contents; the agent guides in `docs/agents/` are built but kept out of the nav) | put a feature on the page that owns it; a new topic gets a new page listed in the nav |
| `opencda_marl/builtin/algorithms/` | the first-party learners; `contrib/` inside it is the drop-in folder for cloned-repo algorithms | loaded through the wheel's own entry point — engine code never imports a learner module |
| `.pixi/`, `dependencies/whl/` | environment and vendored wheels | never edit; everything runs through `pixi run` |

## The seams

These are the invariants that keep the framework pluggable. A change that
needs to break one is a design change, and the PR says so.

- **Algorithms do not edit the engine — and the engine imports none.**
  The engine guarantees observation delivery and action application; every
  learner, the wheel's own `builtin/algorithms/` included, arrives through
  `register_algorithm(...)` + the `opencda_marl.plugins` entry point and
  declares `PolicyCapabilities`. Dispatch sites read capabilities, never
  algorithm names or `isinstance` checks; if a feature needs a name list,
  the capability is missing — add it.
- **One episode loop for both backends.** `MARLCoordinator` drives CARLA
  (`MARLEnv`) and SUMO (`SumoMARLEnv`) through the same `run()` / `step()` /
  `reset_episode()` / `close()`; backend-specific behaviour lives in the
  environment, not in the coordinator, launcher or monitor.
- **World reloads carry state.** `TrainingStateSnapshot` (in
  `_kernel/lifecycle/world_manager.py`) captures and restores learner state across a
  CARLA world reload. Anything kept on the environment or manager that must
  survive a reload goes through it.
- **Per-episode state has a home.** SUMO: `SumoEpisodeState` (`env.episode`).
  New per-episode fields go there, never as loose attributes with hand-written
  clears.
- **Observations are str-keyed downstream of the extractor.** Raw dicts are
  keyed by CARLA actor ids (int) or SUMO ids (str); never `int(agent_id)` —
  a source-scan test enforces it.
- **Metrics have one sink.** Learners log through `TrainingLogger`
  (`log_scalar` / `log_episode_metrics`), which feeds the metrics store (and
  TensorBoard when enabled). External policies are covered by
  `MARLManager.record_update_metrics`.

## Commands

```bash
pixi run verify                                          # the repeatable battery: suite + registry + components + a real SUMO training run (--full adds CARLA)
pixi run pytest test -q                                  # the whole suite (~5 min, no CARLA needed)
pixi run pytest test/route_intent -q                     # the lifecycle / learner suite
pixi run python -m opencda_marl.plugins            # what is registered, and from where
pixi run python opencda.py -t sumo-marl/intersection/route_speed_only_pcmappo --marl   # SUMO smoke
pixi run marl-quick-test                                 # CARLA smoke (server running)
pixi run -e docs mkdocs build --strict                   # docs must build clean
```

Tests do not touch CARLA; they skip rather than fail when a GPU lacks
headroom. `test/conftest.py` puts the repo root on `sys.path`, forces the
`Agg` backend and disables the metrics store — new test files add no path
hacks and do not switch the backend; `test/opencda/` never gets an
`__init__.py` (it would shadow the real package).

## Verification duties

A change is done when it is verified, not when it is written. In order of
cost:

1. **Unit tests for the change** — a new behaviour gets a test that fails
   without it. Known failures are never "fixed" by weakening assertions.
2. **The suites green** on the platform you changed (`pixi run pytest test
   -q`). Lifecycle, launcher, config and registry changes are run on both
   SUMO and CARLA code paths (the tests cover both without a server).
3. **A live smoke run** for anything that touches the episode loop, world
   reloads, the launcher, checkpoints/resume or the monitor: one
   short SUMO run always; one CARLA run when CARLA code changed. Read the
   learner's counters (`training_step`, updates, buffer size) and the
   monitor, not just the episode summary.
4. **`git grep` after any move or rename** for stale paths — in code, configs,
   pixi tasks, docs.
5. **Docs in the same commit**: a new flag, config key or command appears on
   the page that owns it ([Run Experiments](../run-experiments.md),
   [Add an Algorithm](../marl/custom_algorithms.md), the
   [Code Index](code-index.md), this page).

Pitfalls that have fooled agents before:

- Learners whose `update()` catches exceptions return *stale* metrics; an
  `isfinite` assertion passes on a stale `0.0`. Assert that a counter
  advanced and the loss is non-zero, or read the ERROR log.
- Bit-identical summaries across two runs of a stochastic learner mean it
  never acted (a warmup that never ends). Determinism where noise is expected
  is a failure signal.
- Seeded outcome numbers are not comparable across commits that change
  numerics at the ulp level; pin semantics with equivalence tests, not with
  reward numbers.
- cProfile inflates tiny functions; device syncs (`.item()`, `.any()`,
  boolean-mask gathers) and `gc.collect()` are real wall-clock. Microbenchmark
  before optimising.
- A failed CUDA-graph capture poisons the process — the feature stays opt-in
  and fails fast.

## Commits and pull requests

- **Conventional commits, one verified change each**: `feat(sac): …`,
  `fix(launcher): …`, `perf(envs): …`, `docs(marl): …`, `test(route_intent): …`,
  `chore(repo): …`. The body says what was wrong, what changed, and how it
  was verified (the command and the observed result).
- Work on a branch of your fork; open the PR against `radar-lab/OpenCDA-MARL`
  `main` using the [PR template](https://github.com/radar-lab/OpenCDA-MARL/blob/main/.github/PR_TEMPLATE.md).
  Maintainers review and squash-merge; nobody pushes `main` directly.
- Agent-authored PRs are welcome and say so. The template's checklist is
  the review contract:

    - [ ] the change is scoped to one concern; unrelated cleanups are separate PRs
    - [ ] tests added or updated; `pixi run pytest test -q` is green (paste the summary line)
    - [ ] live smoke run for lifecycle / launcher / monitor changes (paste the command and the counters you read)
    - [ ] `opencda/` edits are recorded in the patch ledger
    - [ ] docs updated on the owning page; `mkdocs build --strict` passes
    - [ ] no new algorithm-name dispatch; capabilities declared instead
    - [ ] no generated files (`_runtime/`, `artifacts/`, checkpoints, recordings), no secrets, hostnames, absolute paths or private URLs
    - [ ] the PR description states what an agent could not verify (no GPU, no CARLA) so a maintainer can

## Reporting

Lead with the outcome. Failing tests are reported with their output, never
summarised away; a skipped verification step is named as skipped. A number
quoted anywhere (speed, reward, pass count) comes from a run you can point
to.
