# Code Index

Guide version 1.1 — OpenCDA-MARL 2.1 (2026-10)

Open this when you need to find where something lives, how a run executes,
or which seam a change belongs to. Every path below exists in the
repository (a test checks), so the index is a map you can trust rather
than a narrative. The edit policy per area is in the
[Agent Workflow](workflow.md).

The package tree states the architecture, and a test enforces it
(`test/basic/test_architecture.py`):

```
opencda_marl/
  _kernel/      the complete lifecycle — SEALED. Never imports a plugin;
                external code never imports it. Everything replaceable is
                reached through api/ and the plugin registry.
  api/          the public surface: data contracts, the Policy contract,
                and the registration door for every plugin kind.
  plugins/      the one plugin system: discovery (entry points, env,
                config, contrib), enumeration, API-version rollback,
                and the startup freeze. No hot reload — a run's
                behaviour set is fixed at initialize.
  builtin/      the framework's own plugin set: default implementations
                of every kind, loaded through the same doors an external
                package uses. Never imports the kernel.
  envs/         public facades: the PettingZoo port, the backend env
                classes, the dependency-free schema env.
  telemetry/  evaluation/  utils/  tools/    services both sides may use.
```

- [How a run executes](#how-a-run-executes)
- [Package map](#package-map)
- [Where state lives](#where-state-lives)
- [Extension points](#extension-points)
- [Configs](#configs)
- [Scripts](#scripts)
- [Tests](#tests)

## How a run executes

```
opencda.py -t <family>/<scenario>/<profile> --marl
 └─ opencda_marl/utils/config_locator.py     resolve default.yaml → base.yaml → profile (+ extra roots)
 └─ opencda_marl/utils/config_merger.py      smart_merge of the three layers (an empty key keeps the lower layer's mapping)
 └─ opencda_marl/_kernel/lifecycle/coordinator.py    MARLCoordinator(config) turns null top-level sections into {}
     │    (normalize_run_config), then initialize() → run()
     │    plugins load (entry points, $OPENCDA_MARL_PLUGINS, MARL.plugins, contrib),
     │    every part constructs through the registry, then the registry FREEZES
     ├─ CARLA: _kernel/lifecycle/pipeline_builder.py builds
     │    MARLScenarioManager (_kernel/lifecycle/scenario_manager.py)  CARLA client, world, CavWorld
     │    MARLManager          (_kernel/policy/manager.py)             observations → policy → rewards
     │    MARLEnv              (_kernel/envs/marl_env.py)              the CARLA episode
     │    WorldManager         (_kernel/lifecycle/world_manager.py)    periodic world reloads with state carry-over
     └─ SUMO:  SumoMARLEnv    (_kernel/envs/sumo_marl_env.py)         the whole backend in one class (libsumo/traci)
 └─ one step (coordinator.step):
      env.step → MARLTrafficManager spawns / retires vehicles (_kernel/traffic/traffic_manager.py)
               → MARLAgentManager drives each vehicle's agent (_kernel/agent_manager.py)
               → MARLManager.extract states (the observation_extractor plugin) → algorithm.select_action(s)
               → the reward_calculator plugin (api/rewards.py resolves profiles) → objective vectors (api/objectives.py)
               → algorithm.store_transition / update via AlgorithmStrategy (_kernel/policy/strategy.py)
               → TrainingMetrics (evaluation/training_metrics.py), MetricsStore (telemetry/store.py)
 └─ episode end (coordinator.reset_episode):
      EvaluationManager.end_episode (evaluation/manager.py) → CheckpointManager (_kernel/policy/checkpoint.py)
      → MARLManager.reset_episode → env.reset
```

## Package map

### `opencda_marl/api/` — the public surface

What external packages import; the kernel and the plugins meet here.

| module | role |
|---|---|
| `events.py` | `StepEvent`: the per-step event record every backend emits |
| `contracts.py` | `NormalizedObservation`, `RewardSpec`, `RewardOutput`, `normalize_observation` |
| `policy.py` | the `Policy` / `IntentPolicy` / `PolicyCapabilities` contract every algorithm implements |
| `algorithms.py` | `register_algorithm`, `AlgorithmSpec`, `resolve_algorithm_spec` — the algorithm kind |
| `components.py` | `register_component` / `resolve_component` / `component_key`, the `_BUILTINS` default rows, `BASELINE_AGENT_TYPES`, `KIND_SIGNATURES` |
| `rewards.py` | reward profiles: `register_reward_profile`, `resolve_reward_configuration`, the built-in profile rows |
| `objectives.py` | objective profiles and vector mechanics: `register_objective_profile`, `ObjectiveSpec`, scalarisation |
| `preferences.py` | preference profiles: `register_preference_profile`, `PreferenceRegistry` |
| `route_intent.py` | `RouteIntent` candidates, legality masks and diagnostics; the per-exit-lane commitment counts (`committed_exit_lane_occupancy`) |
| `intent.py` | the `IntentProvider` callable shape and `build_provider_from_config` |
| `intent_observation.py` | the spawn observation an intent policy sees, and its dimensions |
| `safety.py` | `TtcMetricState` — TTC metric mechanics shared by envs and calculators |
| `exceptions.py` | `PlannerFailureException` and friends |

### `opencda_marl/plugins/` — the one plugin system

| module | role |
|---|---|
| `registry.py` | the federation: every kind reports here — journal, cross-kind rollback, `describe_plugins`, `freeze()` / `thawed()` |
| `loader.py` | discovery: entry-point group `opencda_marl.plugins`, `$OPENCDA_MARL_PLUGINS`, `MARL.plugins`, the API-version pin |
| `__main__.py` | `python -m opencda_marl.plugins` — list everything registered |

### `opencda_marl/_kernel/` — the sealed lifecycle

Edit policy: changes here are framework contributions; plugins never need
them. The kernel resolves every replaceable part through the registry.

| area | role |
|---|---|
| `lifecycle/` | `MARLCoordinator` (initialize / step / run / reset_episode / close, plugin freeze, traffic view), `PipelineBuilder`, `MARLScenarioManager`, `WorldManager` + `TrainingStateSnapshot`, `ActorRegistry`, `SensorRig` |
| `envs/` | `MARLEnv` (CARLA), `SumoMARLEnv`, `sumo_runtime.py` predicates, `SumoEpisodeState` |
| `policy/` | `MARLManager` (also the mixed-autonomy gate: the learner acts on and learns from its own population only), `factory.py`, per-family `strategy.py`, `checkpoint.py`, `action_space.py`, `path_lifecycle.py`, `competition.py` (crossing order, per junction), `accumulator.py`, `decision_zone.py` (spawn vs junction-approach trigger), `intent_bridge.py` |
| `agent_manager.py` | `MARLAgentManager`: drives each vehicle's agent, world snapshots |
| `adapter/` | `MARLVehicleAdapter` (one per vehicle), `world_snapshot.py`, `_nearby/`, `_physics/`, `_spatial/` |
| `traffic/` | `MARLTrafficManager`, flows, spawn events, recordings, SUMO bindings (`_sumo_home.py`, `_sumo_lib.py`, `sumo_adapter.py` — also the typed carla stand-in for slim installs), preference sampling at spawn, per-vehicle population (`population.py`), and the SUMO↔CARLA frame bridge read from the map (`coordinate_bridge.py`) |
| `sensing/` | `CarlaMonitor`, `CarlaSpectator`, `MARLCollisionSensor` (the future perception axis gets an api contract) |
| `scenarios/` | `ScenarioBuilder` and templates; a scenario names its map with `scenario.map.town` or `scenario.map.xodr_path` — the latter builds the world from OpenDRIVE alone (road surface and signs, no buildings) |

### `opencda_marl/builtin/` — the built-in plugin set

Default implementations of every kind; replace any of them from your own
package without touching anything here.

| area | kind(s) |
|---|---|
| `algorithms/` | the eight learners + `intent/` + `contrib/` drop-in folder (auto-imported; `contrib/random_speed.py` is the smallest complete learner) — registered via the wheel's own `opencda_marl.plugins` entry point |
| `agents/` | agent types `behavior` / `vanilla` / `rule_based` / `marl` and `AgentFactory` |
| `planners/` | `marl_planner.py` (`MARLPlanner`) + waypoint helpers |
| `rewards/` | `calculator.py` — the default `reward_calculator` |
| `extractors.py` | `ObservationExtractor` — the default `observation_extractor` |
| `intent_providers/` | `tao_fixed` / `naive_random` / `rule_based_path` providers |

### `opencda_marl/envs/` — public facades

| module | role |
|---|---|
| `parallel_env.py` | `OpenCDAMARLParallelEnv`: the PettingZoo port; bring your own training loop |
| `traffic_schema_intersection.py`, `traffic_schema.py`, `spaces.py` | the dependency-free schema env used by unit tests and `tools/schema_env_demo.py` |

### `opencda_marl/evaluation/`, `telemetry/`, `utils/`, `tools/` — services

| package | role |
|---|---|
| `evaluation/` | `EvaluationManager` (summary mode; mirrors `Eval/*` into the run's own metrics store, or a separate `evaluation/` run when the run has none), `TrainingMetrics` (per-episode archive), `aggregate.py`, `hypervolume.py`, `objective_metrics.py` |
| `telemetry/` | `MetricsStore` (SQLite per run, `store.py`), the monitor — the framework's one GUI, auto-started with every run (`monitor.py` backend, `static/` frontend), the schematic traffic view (`traffic_view.py`), the trace writer (`trace.py`) |
| `utils/` | config resolution (`config_locator.py`, `config_merger.py`), `seeding.py` — pure library helpers |
| `tools/` | runnable `python -m` maintenance tools: `config_lint.py`, `xodr_fixer.py`, `list_vehicle_types.py`, `schema_env_demo.py` |

### `opencda/` — vendored OpenCDA

The upstream OpenCDA core (scenario testing, perception, localization,
planning, control, V2X, platooning). Every local modification is a row in
the [patch ledger](../opencda/marl_patches.md); `opencda/scenario_testing/`
holds the non-MARL tutorial scenarios run without `--marl`.

## Where state lives

| state | owner | notes |
|---|---|---|
| per-episode SUMO fields | `SumoEpisodeState` (`_kernel/envs/episode_state.py`) | never loose attributes on the env |
| what survives a CARLA world reload | `TrainingStateSnapshot` (`_kernel/lifecycle/world_manager.py`) | learner networks/optimizers/buffers, counters, route-intent providers |
| checkpoints, resume offsets | `CheckpointManager` (`_kernel/policy/checkpoint.py`), `training_metadata.json` | `latest` / `best` / `episode_NNNN`; a built-in learner's checkpoint `config` is `TrainingLogger.checkpoint_config()`: plain values, no `tensorboard` / `metrics_store` block, nested-block interpolations kept as written (top-level values arrive resolved), values that are absolute paths replaced. No loader reads it |
| per-update and per-episode scalars | `MetricsStore` (`telemetry/store.py`) | `artifacts/metrics/<algo>/<run>/metrics.sqlite`; `<algo>` is the registered algorithm key (the class-derived name only for a learner built without `MARLManager`) |
| TensorBoard event files | `TrainingLogger` (`builtin/algorithms/training_logger.py`) | `<tensorboard dir>/<algo>/<run>/`, with the metrics store's `<algo>`: both writers take it from `TrainingLogger._run_algorithm_name` |
| full per-episode metrics archive | `TrainingMetrics` (`evaluation/training_metrics.py`) | `metrics_history/<run>/metrics_history_epN.pkl` |
| CARLA actors | `ActorRegistry` (`_kernel/lifecycle/actor_registry.py`) | the only place actors are destroyed |
| the run's plugin set | `opencda_marl/plugins/registry.py` | frozen at the end of `initialize()` |

## Extension points

Everything goes through `opencda_marl.api`; `python -m opencda_marl.plugins`
lists what is registered and from where. The user-facing map of the same
seams — contracts, doors, config keys, built-ins — is
[Architecture](../marl/architecture.md); keep the two in step.

| you want to… | touch |
|---|---|
| add a learner | nothing inside the framework: drop a module into `opencda_marl/builtin/algorithms/contrib/` (cloned repo) or ship a package with an `opencda_marl.plugins` entry point — [Add an Algorithm](../marl/custom_algorithms.md) |
| drive the scenarios from your own RL loop | `opencda_marl/envs/parallel_env.py` |
| add a planner / agent type / reward calculator / observation extractor / intent provider | `register_component` (`opencda_marl.api`) in your package; select with `scenario.traffic.planner.type`, `agents.agent_type`, `MARL.components.<kind>` |
| add a reward / objective / preference profile | `register_reward_profile` / `register_objective_profile` / `register_preference_profile` (`opencda_marl.api`) |
| invent a new axis | `register_component("your_kind", "your_key")` — unknown kinds are accepted and listed |
| run mixed autonomy | `agents.population: {mix: {marl: 0.5, rule_based: 0.5}, seed: N}` — shares exact over each window of `resolution` vehicles, reproducible per seed; the learner commands and learns from its own population only. The mix must name `agents.agent_type`; on CARLA it is validated at construction (no learning population: rejected; zero learner share: warning; every positive-share name must be a registered agent type, case-sensitive). On SUMO only that population gets the RL runtime mode and a commanded speed; every other population keeps SUMO's safeguards and car-following model |
| swap the neighbour-context encoder | `MARL.<algo>.context_encoder: <registered key>` (default `lstm`); a `context_encoder` component implements `forward` / `forward_padded` over the ego-first neighbour sequence |
| give a vehicle its own driver parameters | put them in the spawn event's `metadata.driver`; the agent factory receives the whole block as `config["spawn_metadata"]`, and SUMO applies the known keys to background vehicles |
| fix the SUMO frame convention for a map | `runtime.coordinate_landmarks` (or a `<net>_landmarks.json` / `<net>_junction_graph.json` beside the network, keyed by junction id as `{"12": [x, y]}` or `{"junctions": {"12": {"centre": [x, y]}}}`): junction centres in CARLA's frame decide `y_flip`; a map without landmarks is read with no y flip, so a CARLA town export needs them |
| give vehicles per-vehicle objective preferences | `agents.preference: {mode, sampler, dim, alpha, seed, regime}` — the canonical home (moved out of `route_intent`, where the legacy location still resolves) |
| choose what one path action means | `MARL.action_space.path.provider: <registered intent_provider>`; the built-in semantic action (left / straight / right / keep) is the framework's example, and `selector: learned \| rule_based \| random_legal \| keep_current` still names the built-ins |
| decide at the junction instead of at spawn | `MARL.action_space.path.decision_zone: {trigger: approach, distance_m: 30}` |
| observe neighbours' committed exit lanes | `MARL.<algo>.features.committed_exit_lanes: <width>` (off by default; the state width changes with it) |
| add a scenario or profile | a YAML under `opencda_marl/_bundled/configs/<family>/<scenario>/` stating only what changes — or the same path under your own `--marl-config-root` |
| add a traffic recording | record mode writes one under `artifacts/recordings/`; copy it into `opencda_marl/_bundled/recordings/` to adopt it as a benchmark — the packaged files are evaluation ground truth |
| add a monitor chart type | `opencda_marl/telemetry/static/` (vanilla JS, no build step) against the `/api/runs/...` endpoints in `monitor.py` |
| feed the live traffic view from a new backend | implement `collect_traffic_view()` / `collect_traffic_geometry()` on the env (`opencda_marl/telemetry/traffic_view.py` documents the file shapes) |

## Configs

The families are package data, under `opencda_marl/_bundled/configs/`:
`<family>/default.yaml` → `<family>/<scenario>/base.yaml` →
`<family>/<scenario>/<profile>.yaml`, merged by
`opencda_marl/utils/config_merger.py`. Families: `marl/` (baselines,
tutorials, the RAL-era learners), `pcma/` (preference-conditioned speed
control on CARLA), `momarl/` (the path + speed example family on CARLA), `sumo-marl/`
(the SUMO equivalents), `cosim-marl/`, and `opencda/` (the classic OpenCDA
scenarios for standard mode). Resolution order: explicit roots
(`--marl-config-root`, `OPENCDA_MARL_CONFIG_ROOTS`), a checkout-level
`configs/` if one exists, the bundle last. The matrix launcher writes its
per-cell profiles under `artifacts/_runtime/` and passes that as an extra
root. `opencda_marl/tools/config_lint.py` checks every shipped profile
statically (`test/basic/test_config_lint.py`); `test/basic/test_bundled_data.py`
pins the layout. Next to the configs, `opencda_marl/_bundled/checkpoints/ral/`
carries the framework paper's baseline weights (eval-only) with a manifest
and checksums; `load_checkpoint: ral/<run>` resolves there through
`bundled_checkpoint`, and `test/basic/test_bundled_baselines.py` loads every
entry through the manager.

## Scripts

| script | role |
|---|---|
| `scripts/verify.py` | `pixi run verify`: the repeatable battery — suite, registry, plugins, a real SUMO training run (`--full` adds CARLA) |
| `scripts/convert_opendrive_maps.py` | OpenDRIVE → SUMO with netconvert, then checks each network loads and reports its junction/edge/lane/connection counts (`--emit-opendrive` for the reverse) |
| `scripts/experiments/launch_matrix.py` | plan and run training cells (parallel slots, per-slot CARLA, resume, partitions) |
| `scripts/experiments/launch_eval_matrix.py`, `run_matrix_eval.py` | evaluation of trained cells (two routes) |
| `scripts/experiments/analyze_matrix_results.py`, `analyze_matrix_eval.py` | outcome tables |
| `scripts/experiments/launch_distributed_matrix.py`, `workers.example.yaml` | several machines |
| `scripts/experiments/matrix_common.py` | slugs, map profiles (`canonical_map` / `canonical_maps` / `map_profile`: a slug always carries the profile key), condition tables, schema validation shared by the launchers |
| `scripts/docker_carla.py`, `scripts/docker_train.py` | Docker helpers |
| `scripts/inspect_traffic_manager.py`, `scripts/experiments/render_traffic_trace.py` | traffic / route-trace inspection |

## Tests

| suite | covers |
|---|---|
| `test/basic/` | config resolution and lint, rewards, replay buffer, map loading, docs boundary, the architecture seal |
| `test/marl/` | lifecycle contracts: checkpoints, resume numbering, evaluation close, metrics flush, SUMO reset, telemetry store, traffic view, on-policy rollouts |
| `test/route_intent/` | the plugin system (unified registry, cross-kind rollback, freeze), policy contract, strategies, components, route intent, intent learners, SAC/TD3 internals, world snapshot and reload, launcher planning, seeding |
| `test/opencda/` | vendored-core behaviour the patches rely on |

`test/conftest.py` puts the repository root on `sys.path`, forces the
`Agg` backend, disables the metrics store, and unfreezes the plugin
registry around each test. Source-scan tests enforce invariants
(str-keyed observations, no `int(agent_id)`).
