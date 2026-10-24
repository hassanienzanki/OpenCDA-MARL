# OpenCDA-MARL

Cooperative multi-agent reinforcement learning for autonomous driving,
built on [OpenCDA](https://github.com/ucla-mobility/OpenCDA): the same
signal-free intersection scenarios run on CARLA (full vehicle stack) or on
SUMO (traffic only, orders of magnitude faster), and any learner that
implements one small contract trains inside the framework's episode loop
with checkpoints, metrics and the experiment matrix included.

![the monitor's live traffic view during a SUMO training run](images/monitor-traffic-view.gif)

*The web monitor's traffic view: a signal-free intersection during a live
SUMO run, learning agents in blue — headless, no CARLA rendering.*

## Four pages

<div class="grid cards" markdown>

-   **[Quick Start](quick-start.md)**

    Run a bundled learner untouched — SUMO, headless, monitor included —
    then change one thing.

-   **[Architecture](marl/architecture.md)**

    One sealed lifecycle, many replaceable joints: the map of every plugin
    kind, port and config seam an experiment can swap or vary.

-   **[Run Experiments](run-experiments.md)**

    One run, the monitor, the experiment matrix: the launcher with parallel
    runs on a multi-GPU box, resume, evaluation and analysis.

-   **[Customize](marl/custom_algorithms.md)**

    Plug in a learner ([Add an Algorithm](marl/custom_algorithms.md)),
    swap a lifecycle joint ([Replace a
    Component](marl/replace_component.md)), or drive the scenarios from
    your own RL stack through the PettingZoo port.

</div>

Install first: [Installation](installation.md) and
[CARLA Version Support](carla-version-support.md). Developing the
framework itself — as a person or a coding agent — starts at
[`AGENTS.md`](https://github.com/radar-lab/OpenCDA-MARL/blob/main/AGENTS.md)
([Contributing](contributing.md) is the short form).

## What ships

- **Learners**: MAPPO, MASAC, MATD3, DQN and tabular Q-learning; PCMAPPO and
  PCMASAC build on [PCMA](marl/pcma.md) (without its preference planner) for
  multi-objective training, and
  PCMATD3 is preference-conditioned only — all registered through the same
  interface an external package uses.
- **Baselines**: the framework paper's DQN, MAPPO and SAC checkpoints and
  most of its TD3 ones (all but the two 1024-wide TD3-44d runs, which
  retrain from the `td3_44d` profile) ship in the wheel with their
  profiles and the deterministic recordings they were trained against — a
  fixed reference point for comparison.
- **Backends**: CARLA 0.9.15 / 0.9.16 (0.10.0 experimental), SUMO via
  libsumo, CARLA–SUMO co-simulation.
- **Infrastructure**: one verified lifecycle across backends with CARLA world
  reloads that carry the full learner state; resumable runs; a SQLite
  metrics store with a live web monitor and schematic traffic view; the
  experiment matrix launcher with per-slot CARLA servers and GPU spreading.

## Citation

If you use OpenCDA-MARL, please cite the framework paper (and OpenCDA); the
BibTeX is in the repository
[README](https://github.com/radar-lab/OpenCDA-MARL#citation). The
preference-conditioned learners build on
[arXiv:2606.14693](https://arxiv.org/abs/2606.14693) (without its preference
planner) — cite it with them
([PCMA Learners](marl/pcma.md#citation)).

## Acknowledgments

- Developed at the [Radar Lab](https://github.com/radar-lab), University of
  Arizona ([About](about.md)).
- Built on [OpenCDA](https://github.com/ucla-mobility/OpenCDA) by the UCLA
  Mobility Lab, [CARLA](https://carla.org/), [SUMO](https://eclipse.dev/sumo/)
  and [Pixi](https://github.com/prefix-dev/pixi).
- The OpenCDA-MARL code is licensed under the
  [Apache License 2.0](https://github.com/radar-lab/OpenCDA-MARL/blob/main/LICENSE); the vendored OpenCDA core
  keeps UCLA's academic, non-commercial licence, and other third-party material keeps its own terms — see
  [THIRD_PARTY_NOTICES.md](https://github.com/radar-lab/OpenCDA-MARL/blob/main/THIRD_PARTY_NOTICES.md).
