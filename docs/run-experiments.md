# Run Experiments

Everything runs through `pixi run` from the repository root. One scenario
reference selects a config family, a scenario and a profile:

```
<family>/<scenario>/<profile>      e.g.  momarl/intersection/learn_path_pcmappo
```

| family | backend | what it is |
|---|---|---|
| `marl/` | CARLA | baseline agents (`rule_based`, `vanilla`, `behavior`) and the single-learner tutorial profiles |
| `pcma/` | CARLA | preference-conditioned speed control (MASAC / PCMASAC / MAPPO / PCMAPPO profiles) |
| `momarl/` | CARLA | the path + speed example family: one learner (PCMAPPO) under every built-in path provider; the profiles the matrix launcher runs |
| `sumo-marl/` | SUMO | the same intersection and learners without CARLA (`route_*` profiles) |
| `cosim-marl/` | CARLA + SUMO | co-simulation |

Configuration is three layers merged in order: the family's `default.yaml`,
the scenario's `base.yaml`, then the profile. A profile only states what it
changes. A key left empty in a higher layer (`outputs:` with nothing under
it, or `null`) keeps the lower layer's section instead of erasing it. Any
other scalar, a bare `true` / `false` included, still replaces it; a
mapping still merges into it key by key, and a list over it is an error; an
empty value over a scalar or a list still sets it to null. Extra config
roots (`--marl-config-root DIR` or `OPENCDA_MARL_CONFIG_ROOTS`) may hold
just a profile and inherit the other two layers from the repository — that
is how an external package ships its own experiment
([Add an Algorithm](marl/custom_algorithms.md)).

## One run

```bash
# SUMO: nothing to start, runs headless in ~10 ms/step
pixi run python opencda.py -t sumo-marl/intersection/route_speed_only_pcmappo --marl

# CARLA: start a server first (see Installation), then
pixi run python opencda.py -t momarl/intersection/speed_only_pcmappo --marl
pixi run python opencda.py -t marl/intersection/rule_based --marl
```

What a run writes, under `outputs.root_dir` (default `artifacts/`):

| path | content |
|---|---|
| `checkpoints/<run>/<algo>/` | `latest_checkpoint.pth`, `best_checkpoint.pth`, `episode_NNNN.pth` (pruned to `keep_checkpoints` every `save_freq` episodes), `training_metadata.json` |
| `metrics/<algo>/<run>/metrics.sqlite` | the metrics store read by the monitor (losses, Q values, episode summaries); `<algo>` is the registered algorithm key (`masac`, `pcmasac`, `pcmappo`, ...) |
| `<outputs.tensorboard.dir>/<algo>/<run>/` | TensorBoard event files, in `segment_NNNN/` windows of `max_episodes_per_run` episodes; when `keep_last_runs` is above 0 only that many newest windows are kept (the family defaults keep 5; 0 keeps all). `dir` is `tensorboard` in the `marl/`, `sumo-marl/` and `cosim-marl/` families and `runs` in `momarl/` and `pcma/`; `<algo>` is the same registered key as the metrics store's, and `<run>` a timestamp unless `outputs.tensorboard.run_name` sets it |
| `metrics_history/<run>/metrics_history_epN.pkl` | full per-episode metrics archive, one pickle per `export_interval` episodes, flushed at exit |
| `evaluation/...` | per-episode summary CSV and `final_summary.json` when `outputs.evaluation.mode: summary`; the monitor shows the same numbers in the run's own entry (see [Watch it](#watch-it)) |
| `logs/<slug>/run.log` | the cell's log when launched by the matrix launcher |
| `recordings/<name>.json` | the traffic drawn by a `mode: record` run, named after `replay_file`'s basename (`outputs.recordings.dir` moves it) |

A built-in learner's checkpoint also stores a copy of its config, for
reference only (nothing loads it back): plain values, without the
`tensorboard` and `metrics_store` blocks that locate the run's outputs, and
with any value that is an absolute path replaced by
`<absolute path removed>` (a path inside a longer string is not). The
hyperparameters are kept.

Key knobs live under `scenario.simulation` (`max_steps`, `max_episodes`),
`scenario.traffic` (`mode: replay|live|record`, `replay_file` — the
recordings under `opencda_marl/_bundled/recordings/` are deterministic
evaluation ground truth, and a `record` run writes its own draw to
`artifacts/recordings/` instead: adopting one as a benchmark is a
deliberate copy into the package), and
`MARL.training` (`training_mode`, `save_checkpoints`, `save_freq`,
`keep_checkpoints`, `load_checkpoint`, `checkpoint_dir`). A baseline agent
(`rule_based`, `vanilla`, `behavior`) on CARLA runs `max_episodes` like any
other run. Every bundled family base sets 500, so a baseline profile that
wants one episode says `max_episodes: 1`, as the bundled
`marl/intersection` baselines do.

On CARLA, `world.timeout` (seconds; 20 in the bundled MARL families) is how
long the client waits for the server on each call. Building an OpenDRIVE
world or loading a town that times out is retried once with twice that
timeout; a second time-out stops the run with a message naming the key, so
raise it on a heavily loaded machine.

**Evaluation** is a run with `MARL.training.training_mode: false` and
`load_checkpoint: <path>`: no learner updates, replay traffic, the configured
number of episodes on both backends. A configured checkpoint that cannot be
loaded aborts the run rather than scoring random weights. The framework
paper's baselines ship in the wheel as eval-only weights:
`load_checkpoint: ral/<run>` (e.g. `ral/td3_43d_300vph`) resolves inside the
package, and `opencda_marl/_bundled/checkpoints/ral/manifest.json` names the
profile and the `traffic_<vph>vph` recording each one pairs with.

**SUMO → CARLA transfer** is the same mechanism with training left on: train
on a `sumo-marl/` profile, then point a CARLA profile's `load_checkpoint` at
the SUMO checkpoint. A loaded checkpoint marks the learner pretrained, so the
random-action warmup is skipped and fine-tuning starts immediately. SUMO
details live under `world` (`sumo_gui: true` opens the SUMO window,
`sumo_port` when 8873 is taken).

## Watch it

**Starting a simulation starts the monitor.** Every run ensures a web
service on `http://127.0.0.1:8765` (a detached process that outlives the
run and is reused by later runs; `outputs.monitor: {enabled, host, port,
allow_hosts}`, or a bare `true` / `false`, controls it) and prints the URL. Manual start
still works:

```bash
pixi run monitor            # http://127.0.0.1:8765 — every run under artifacts/
pixi run monitor-list       # the same, in the terminal
```

A **Traffic view** panel draws a live schematic of the selected run —
SUMO-style lane lines, the learning population's vehicles as blue triangles
and background traffic in grey — for headless runs on either backend, with
no CARLA rendering involved (`outputs.traffic_view: {enabled, interval}`, or
a bare bool, controls the writer). It shows the latest published frame, so
it trails the simulation by up to one publish interval (`interval`, 0.2 s of
wall time by default) plus the page's poll. When no newer frame arrives for
`max(3 s, 3 × interval)`, the last frame stays on screen dimmed and the
status line reads `stale`; once the run has finished (its metrics store was
closed after that frame) it reads `run ended`. A frozen frame is never
labelled live.

The monitor reads the per-run metrics stores live (a training process never
blocks it), overlays runs, smooths, and lists episode summaries:

- **One run, one entry.** The run list labels each run `<algorithm> · <mode>`:
  the registered algorithm key (`masac`, `pcmasac`, ...), which also names
  the directory `metrics/<algorithm>/<run>/`, and `train` or `eval`. A
  learner run's evaluation statistics go into its own store: the `Eval/*`
  tags, one `evaluation_episode` event per episode and `final_summary` in
  the run's metadata. Only a run without a store of its own (a baseline
  agent, or a learner with `outputs.metrics_store.enabled: false`) gets a
  separate run under `metrics/evaluation/`, labelled with the learner key or
  the agent type.
- **Outcome columns first.** Every built-in learner logs `success_rate`,
  `collision_rate`, `timeout_rate` and the vehicle counts `successes`,
  `collisions` and `timeouts` per episode, as external policies do. The
  episode table shows the outcome, throughput, TTC and near-miss columns
  first, then the `reward_component_*` columns, then whatever else the run
  logs in alphabetical order, up to 24 extra columns. Columns whose shown
  values are all whole numbers (counts, episode and step numbers) print as
  integers; other columns keep the page's usual precision on every row.

It is standard-library only, so it runs on a headless training node:
`ssh -L 8765:127.0.0.1:8765 <node>` and open the same URL. TensorBoard event
files are also written by default (`outputs.tensorboard.enabled: false`
turns them off); `pixi run tensorboard` serves every family's under
`artifacts/`, but the monitor does not need them.

One monitor serves one artifacts root: a run reuses a monitor only when it
serves the run's own root, and when the port already serves another root
the next port is tried, up to port + 7; the run prints the URL it got. The
service outlives every run. To stop it, end the process whose `pid`
`GET /api/root` reports; on Windows, stop it before deleting or renaming
the root, which it holds as its working directory and `<root>/monitor.log`.

A monitor bound to a loopback address (the default `127.0.0.1`) answers
only requests whose `Host` header is a loopback name (`127.0.0.1`,
`localhost`, `[::1]`, `*.localhost`), so the SSH tunnel above works
unchanged. Behind a proxy that forwards under its own host name (Jupyter
server-proxy, code-server `/proxy`, Open OnDemand, dev tunnels) the page
gets a 403 `host not allowed: NAME; start the monitor with --allow-host
NAME`. Start it by hand for that root:

```bash
pixi run python -m opencda_marl.telemetry.monitor --root artifacts --allow-host NAME   # repeatable
```

Later runs on the same root reuse it, because a run reuses any monitor that
serves its root.

The store is on by default, for training and evaluation runs alike. Under
`outputs.metrics_store`:

| key | default | meaning |
|---|---|---|
| `enabled` | `true` | write `<root_dir>/<dir>/<algorithm>/<run_name>/metrics.sqlite` |
| `dir` | `metrics` | under `outputs.root_dir` (`log_dir` overrides with an absolute path) |
| `run_name` | timestamp | the launchers set the cell slug |
| `log_frequency` | `10` | stride for per-update scalars (losses, Q values); episode scalars are always kept |

```python
from opencda_marl.telemetry import MetricsStore, discover_runs

run = discover_runs("artifacts")[0]                 # newest activity first
store = MetricsStore(run.path, read_only=True)
store.tags()                                        # tag, count, last step/value
store.scalars("Loss/critic", max_points=500)        # [(step, wall, value)], min/max downsampled
store.episodes(limit=50)                            # per-episode summaries
```

## Maps on both backends

A comparison of CARLA and SUMO wants the *same* roads on both, so derive
the SUMO network from the CARLA map's own OpenDRIVE rather than building a
lookalike:

```bash
pixi run python scripts/convert_opendrive_maps.py            # every map it can find
```

Converted networks live in `opencda_marl/assets/maps/` with a
[README](https://github.com/radar-lab/OpenCDA-MARL/blob/main/opencda_marl/assets/maps/README.md)
recording provenance. `.xodr → .net.xml` keeps CARLA's junction ids as SUMO
node ids, so a site chosen in CARLA is findable in SUMO.

Pointing a scenario at an `.xodr` instead of a town name builds the world
from the road network alone — no buildings or props, which loads in seconds
and matches the SUMO network by construction:

```yaml
scenario:
  map:
    xodr_path: "opencda_marl/assets/maps/Town10HD.xodr"
```

SUMO does not read `scenario.map`: it loads the network named by the
`<net-file>` of the `.sumocfg` at `runtime.sumo_cfg`, so pairing the two
backends takes a `.sumocfg` that names the derived `.net.xml`. On SUMO, a
network derived from a CARLA map is read in CARLA's frame only with
junction landmarks (`runtime.coordinate_landmarks`, or a
`<net>_landmarks.json` beside the network; see the frame-convention row in
[Architecture](marl/architecture.md#adjustable-seams)). No `.sumocfg`
names the derived town networks in `opencda_marl/assets/maps/`, and no
landmarks ship for any town; without landmarks SUMO positions come out
mirrored in y relative to CARLA's frame.

OpenDRIVE worlds skip pedestrian navigation generation by default. Vehicle
experiments do not need a walker navmesh, and generating it on every world
reload can leave expensive navigation-build jobs running alongside training.
Set `world.pedestrian_navigation: true` when a scenario needs AI walkers.
This setting does not change the road mesh or vehicle dynamics and does not
affect prebuilt CARLA towns.

## Driver options

Traffic-signal behavior depends on the driver. For Vanilla-derived drivers,
`ignore_traffic_light: true` disables the BasicAgent fallback light check.
Explicit learned speed commands use a separate route-following path that
does not call that check. A signalized experiment needs a driver or safety
layer that also enforces signals for explicit commands. Verify signal handling
on both backends when comparing them; sharing a map does not align it.

The bundled rule-based driver (`RuleBasedAgent`, CARLA) reads these switches
from its driver configuration (`agents.rule_based`, over `vehicle.behavior`);
the [changelog](marl/changelog/v2.1.0.md) lists how it differs from 1.x:

- **Forward corridor.** Another vehicle is a leader only when its projected
  body overlaps the ego's forward driving corridor. A queue in an adjacent
  lane does not block a clear lane; vehicles merging into the corridor still
  constrain speed.
- **Junction yielding** (`junction_stop_on_conflict`, on by default). A
  vehicle that must yield at a predicted conflict stops before the junction
  entry until the conflict clears, and the car-following speed cap stays
  active: granting priority to crossing traffic does not permit driving into
  a stopped leader.
- **Lane hold** (`lane_hold`, on by default). The vehicle keeps its current
  lane, and drops back, while the planner's next lane is occupied alongside,
  instead of moving laterally into the occupied lane.
- **Merge order** (`lane_hold_geometry_priority`, on by default). When nearby
  merging vehicles are standing and face the same direction, the front
  vehicle clears forward in its current lane first. They are ordered along
  their shared mean heading; positions within half a metre use actor IDs to
  break ties, and missing or incompatible headings fall back to ID order.
  `false` orders by actor ID alone, which can hold a merging vehicle and the
  standing vehicle behind it in place. The option does not bypass following,
  junction or other speed caps.
- **Abreast guard** (`abreast_guard`, on by default). The rear vehicle of a
  same-direction pair alongside that is closing laterally drops below its
  neighbour's speed until the pair is no longer alongside.

An agent may expose a safety-filter state through a `shield_state()` method
that returns a dict. Traces keep its `applied_speed`, `shield_active`,
`shield_stage`, `lane_hold` and `merge_priority_enabled`,
`merge_priority_release` and `merge_priority_override` fields, so behavior
diagnostics can tell a speed the agent's own rules set from the commanded
one, and an agent that reports `shield_stage` also says which rule set it.
Only the CARLA vehicle adapter reads the hook. The bundled
`RuleBasedAgent` implements it: `lane_hold` (the hold replaced the
planner's target this step), `merge_priority_enabled` (the configured
`lane_hold_geometry_priority`), this step's `merge_priority_release` and
`merge_priority_override`, and `applied_speed` (the target speed its rules
gave the controller; null on a step whose rules did not run). It has no
safety filter, so its `shield_active` and `shield_stage` are null: its
trace separates lane holding, but not following from junction yielding.
The other bundled agents (`marl`, `vanilla`, `behavior`) do not implement
the hook, so their vehicles, the learner's `marl` vehicles included,
record null in all seven fields. For a vehicle the learner does not
command, the record's `target_speed` is the vehicle's current speed, so
the gap to `applied_speed` is not an intervention. On SUMO every field is
null: SUMO builds no driver agent, the learner commands its vehicles'
speeds through TraCI, and background populations such as `rule_based`
stay on SUMO's own car-following and lane-change models. CARLA collision
events also include the actual other actor's ID/type, event frame and
impulse in `metadata`, distinguishing vehicle contacts from static
geometry without changing the collision threshold.

## The experiment matrix

`scripts/experiments/launch_matrix.py` plans *conditions × regimes × seeds*
as cells, writes one runtime profile per cell under `artifacts/_runtime/`
(passed to each run as an extra config root), and runs them — sequentially
or in parallel slots, each slot with its own CARLA server:

```bash
pixi run python scripts/experiments/launch_matrix.py \
    --conditions speed_only_pcmappo learn_path_pcmappo \
    --regimes aligned --seeds 0 1 \
    --max-episodes 10 --parallel 2 --resume
```

- `--resume` skips cells whose archive already holds `--max-episodes`
  episodes and restarts interrupted cells from their `latest_checkpoint.pth`
  with continuous episode numbering and the pre-interruption best preserved.
- `--cell-timeout` kills the whole cell process tree; a failed cell bounces
  only its own CARLA server; the launcher exits non-zero when any cell failed.
- `--parallel N` runs N slots on RPC ports `2000 + 10k`; `--cuda-visible-devices`
  and `--carla-gpus` spread learners and servers over the cards (slot `k`
  takes entry `k mod n`). Servers the launcher started are stopped at exit
  (`--keep-carla` to keep them).
- `--carla-restart-mode process --carla-bin /path/to/CarlaUE4.sh` uses a host
  binary; `docker` restarts pre-started `opencda-carla-<port>` containers
  ([Throughput](marl/throughput.md) has the Docker recipe and measured
  multi-GPU numbers).
- The default `--carla-restart-mode process` starts and restarts CARLA
  itself, so it needs `CARLA_ROOT` (or `--carla-root` / `--carla-bin` /
  `CARLA_BIN`); with servers managed elsewhere pass
  `--carla-restart-mode none` (or `docker`).
- `--partition k/n` takes every n-th cell starting at k (stride slicing), so
  n machines sharing one artifact tree each run one partition of the same plan.
- `--slug-prefix` (default `matrix_`) names the run: every cell slug is
  `<prefix><map>_<condition>_<regime>_seed<N>`, and `--resume` only matches
  cells under the active prefix. Continuation cells (`continue_*`) start
  from the same prefix's source cells unless `--continue-source-prefix`
  names another run.
- `--maps` names map profiles (default `custom`), and `<map>` in a slug is
  always the profile's key. `--maps`, `--continue-source-map` and
  `launch_eval_matrix.py --checkpoint-map` accept any case, so
  `--maps Custom` names `matrix_custom_*` cells. Spellings of one map
  (`--maps custom Custom`) plan its cells once, and an unknown map stops the
  launcher before any cell is planned.
- `--dry-run` prints the plan and the exact commands.

`pixi run matrix …`, `pixi run matrix-eval …` and `pixi run matrix-analyze`
are shorthands for the three scripts; extra arguments are appended.

Then evaluate and summarise:

```bash
# replay-traffic evaluation of every cell's best checkpoint, isolated under *_eval200vph slugs
pixi run python scripts/experiments/launch_eval_matrix.py --conditions ... --regimes ... --seeds ... --eval-episodes 3

# or the checkpoint-eval route, which writes eval_summary.json per cell
pixi run python scripts/experiments/run_matrix_eval.py --episodes 3

pixi run python scripts/experiments/analyze_matrix_results.py   # training outcomes per cell / action space / regime
pixi run python scripts/experiments/analyze_matrix_eval.py      # evaluation outcomes
```

Each analyzer and `run_matrix_eval.py` reads every slug prefix under its
root and prints the prefix in each row; pass `--slug-prefix <prefix>` to read
or evaluate one run (they warn when several are present).

Several machines: `launch_distributed_matrix.py --manifest workers.yaml`
partitions the plan over the workers described in
`scripts/experiments/workers.example.yaml`.

The launcher is wired to the bundled CARLA `momarl/intersection` family.
SUMO runs need no CARLA server: run `sumo-marl/` profiles directly
(`pixi run python opencda.py -t sumo-marl/intersection/<profile> --marl`,
about 10 ms per step), as many processes as you have cores.

## Verify before you trust a number

- A learner that never acted produces plausible summaries: read the
  learner's own counters (`training_step`, `updates_performed`, buffer size)
  in the monitor or the log, not the episode summary alone.
- Compare speeds back-to-back on the same machine state; absolute ms/step
  drifts 2–3× with background load.
- Seeded outcome numbers are not comparable across commits that change
  numerics at the ulp level; the runs are chaotic. Semantics are pinned by
  tests, not by outcome numbers.
