# Training Throughput

Where the wall-clock goes in a MARL training step, and how to fill a
multi-GPU machine. Numbers below were measured on a single-GPU workstation
(RTX 5090, ~15 concurrent CAVs, `-RenderOffScreen -quality-level=Low`) with
cProfile attached; treat them as proportions, not absolutes.

## What one run costs

| Backend / learner | ms per env step | Largest pieces |
|---|---|---|
| CARLA + MASAC (SAC), 15 CAVs | ~60 (profiled) / ~45 unprofiled | CARLA physics tick + client reads (~35), learner update (~24: two backward passes, ~500 small kernel launches) |
| SUMO + PCMASAC (SAC), 25 CAVs | ~15–20 | learner update (~10), observations/rewards/extractor (~5), SUMO step (~0.4) |
| SUMO + PCMAPPO / MAPPO, 25 CAVs | ~5–7 | acting forward (~2.5), env side; one PPO update per `n_steps` |

Two facts follow from the profiles:

1. **A single run is CPU/launch-bound, not GPU-bound.** The networks are
   small (256-wide MLPs, batch 256, an LSTM over ≤ 30 rows) so the learner
   update is dominated by kernel-launch latency and Python; the GPU is idle
   most of the time. One run cannot saturate one large card, let alone two.
   Faster GPUs barely move the number; more CPU cores and more concurrent
   runs do.
2. **CARLA is the other half.** Even with everything else removed, the
   physics tick plus the per-step client reads are ~35 ms at 15 CAVs. SUMO
   steps in well under a millisecond.

## Filling a multi-GPU box: run cells in parallel

The matrix launcher runs independent cells concurrently, one CARLA server
per slot, and spreads learners and servers over the GPUs:

```bash
pixi run python scripts/experiments/launch_matrix.py \
    --parallel 8 \
    --cuda-visible-devices 0,1 \
    --carla-gpus 0,1 \
    --carla-rpc-port 2000 \
    ...
```

- `--cuda-visible-devices 0,1` puts the learner of slot `k` on GPU
  `k mod 2`; `--carla-gpus 0,1` starts its CARLA server with
  `-graphicsadapter=k mod 2`; with `--carla-rpc-port 2000`, slot `k` uses
  port `2000 + 10*k`.
- Every slot restarts **only its own** CARLA server (by process handle or
  by the `-carla-rpc-port=<port>` marker in the command line), so one
  slot's failure never takes the others down.
- Choose `--parallel` from CPU cores, not VRAM: each slot needs roughly
  1.5–2 cores for CARLA plus one for the trainer; a headless CARLA server
  uses ~1.5–2 GB of VRAM and a learner well under 1 GB, so a large card is
  never the constraint. Start with cores/3, watch aggregate steps/s, and
  raise until it stops growing.
- With many small learner processes on one GPU, NVIDIA MPS
  (`nvidia-cuda-mps-control -d`) removes context-switch overhead between
  them; optional, measure before keeping it.
- The docker path is equivalent and preferable on shared servers; see
  **Running CARLA in Docker** below.
- SUMO runs need no CARLA at all: they are CPU-bound and can be run in
  dozens on one machine, each learner sharing a GPU. The launcher is wired
  to the CARLA `momarl/intersection` family, so start SUMO profiles
  directly (`pixi run python opencda.py -t sumo-marl/intersection/<profile>
  --marl`), one process per run.
- At exit the launcher stops the CARLA servers it started (`--keep-carla`
  leaves them up); servers that were already listening are never touched.
- Linux notes: pass `--carla-bin /path/to/CarlaUE4.sh` when CARLA is not at
  `/opt/CARLA_0.9.16` (the pixi activation default). The launcher strips
  `SDL_VIDEODRIVER` from the server's environment — the pixi Linux
  activation sets it to `offscreen` for pygame, and UE4's bundled SDL2 exits
  immediately under it. `-graphicsadapter=k` follows the Vulkan device
  order, which matched the CUDA index on the nodes tested.

Measured on a 4-GPU Linux node (CARLA 0.9.16
`-RenderOffScreen -quality-level=Low`, intersection map, 600-step episodes,
`--cuda-graph-update`):

| Concurrent cells | ms per env step, MASAC cell | Notes |
|---|---|---|
| 1 | 10–13 | one CARLA server (~2 GB VRAM) + one learner |
| 4 (one per GPU) | 12–19 | ≈ 2.7–3× the aggregate throughput of one cell |

A CARLA server on the intersection map holds ~2.0–2.4 GB of VRAM, so VRAM
is never the limit on these cards; CPU cores and the per-slot CARLA tick
are. Start-up (CARLA launch + OpenDRIVE world + first spawn) is ~45 s per
slot and overlaps across slots.

## Running CARLA in Docker

`--carla-restart-mode docker` makes each slot bounce **an existing container**
(`docker restart`); it never creates one. Start the containers first, one per
slot, named by their RPC port — that is the name the launcher restarts:

```bash
# One container per slot: port 2000+10k on GPU k.
for k in 0 1 2 3; do
  pixi run python scripts/docker_carla.py start \
      --rpc-port $((2000 + 10 * k)) --streaming-port 0 --gpu $k
done
pixi run python scripts/docker_carla.py status --rpc-port 2000   # or docker ps
```

Each container runs `carlasim/carla:0.9.16` with `--network host` (so the port
is the host port), `--gpus device=k`, and `--restart unless-stopped` — CARLA
crashes are restarted by docker instead of vanishing the container.

Then run the matrix against them:

```bash
pixi run python scripts/experiments/launch_matrix.py \
    --parallel 4 \
    --carla-restart-mode docker \
    --carla-rpc-port 2000 \
    --cuda-visible-devices 0,1,2,3 \
    ...
```

Notes:

- Slot `k` uses port `2000 + 10*k` and restarts the container
  `opencda-carla-<port>`.
- Do **not** pass `--carla-gpus` in docker mode: `--gpus device=k` already
  pins the container to one card, and CARLA sees it as adapter 0.
- `--carla-bin` / `CARLA_BIN` are irrelevant here — no host CARLA install is
  needed at all.
- The launcher's exit-time cleanup stops only servers it started itself, so
  containers survive the run (stop them with `scripts/docker_carla.py stop
  --rpc-port <port>`).
- The account running docker must be in the host's `docker` group; a
  `permission denied ... /var/run/docker.sock` from `docker ps` means it is
  not, and no flag in this repo can work around it — use the host-binary path
  (`--carla-restart-mode process --carla-bin /path/to/CarlaUE4.sh`) until an
  administrator adds the account.

## Per-run speed-ups

- **Act on a CPU replica, train on the GPU (`<algorithm>.acting_device`,
  default `auto`).** At a dozen agents per step the GPU is launch-bound:
  a full acting step (context encode, actor, critic, transfers) measures
  0.74 ms on the CPU against 1.34 ms on CUDA at these shapes.
  So with `auto` a learner that trains on CUDA acts on CPU copies of its
  actor, critic and context encoder, refreshed after every optimiser step
  or checkpoint load through one pinned flat copy (0.3 ms), with the
  value normalisers mirrored. Back-to-back on the same machine state, one
  SUMO PCMAPPO episode under cProfile: 47–49 s acting on the GPU, 41 s on
  the replica (the acting path itself 11.5 s → 6.6 s); the update is
  unaffected. Two caveats. The replica caps torch's intra-op pool
  (`acting_threads`, default `auto` = at most 8): with the default pool
  of 24 the same step costs 1.45 ms and the run is *slower* than the GPU.
  And CPU and GPU arithmetic differ in the last bits, so a run is no
  longer bit-identical to one that acted on the GPU. `acting_device:
  train` keeps acting on the training device.
- **CUDA-graph update for SAC (`<algorithm>.cuda_graph_update: true`).**
  The tensor-only core of the update (actor/critic/alpha forward + backward
  + optimizer steps, Polyak, metrics) is captured once into a CUDA graph and
  replayed every update; the data-dependent packed-LSTM encode stays eager
  and feeds static inputs. Needs CUDA and the frozen encoder
  (`train_context_encoder: false`, the default); optimizers become
  `capturable`; the graph is re-captured after checkpoint loads and world
  reloads and the mode falls back to eager on any capture problem. Measured
  back-to-back: SUMO PCMASAC episode 23.8 → 9.9 ms/step (2.4×), CARLA
  1200-step MASAC smoke 73 → 60 s (CARLA physics dominates the rest). No
  Triton/compiler needed, so it works on Windows and Linux alike; the
  update it replays is numerically the eager update (equivalence test with
  a deterministic actor). Off by default — enable it in training configs.
  One rule: nothing may hold an autograd graph through the SAC actor/critic
  built outside `update()` when the graph is captured (every acting path
  here runs under `no_grad`, so normal training is fine); if capture fails
  the learner raises `CudaGraphCaptureError` rather than continuing, because
  torch cannot restore the CUDA RNG state in-process afterwards. The same
  recipe applies to TD3.
- **`torch.compile` on the acting forward** would need Triton (Linux only)
  and is not wired up.
- **Update-to-data ratio.** SAC/TD3 take one gradient step per environment
  step; each step yields one transition per agent, so the effective ratio is
  already ~1 gradient step per 15–25 transitions. Fewer gradient steps per
  env step cut learner time proportionally at the cost of sample efficiency
  (a hyperparameter trade-off, not a code change); batching several
  gradient steps into one call does not help, since the cost is per
  gradient step.
- What has already been removed: per-transition
  LSTM forwards at storage time (all learners now store raw neighbour
  sequences and encode once per update), per-update device syncs, the
  V2X O(N²) search (no consumer in MARL scenarios), per-spawn map-table
  construction, per-step metrics assembly with evaluation disabled, and
  the extra world snapshot per step.
- **Engine-side costs removed before 2.1.0** (one SUMO episode
  of the bundled example, 2,400 steps, ~14 vehicles, libsumo, under
  cProfile: 62 → 13 ms/step): the feature extractor walked an OmegaConf mapping per
  agent per step (31% of the step) and ran three times per step on
  identical inputs; the neighbour block was a per-ego scalar search with
  five TTC solves each; a vehicle's population was re-derived per step;
  the rollout was re-stacked and re-uploaded per PPO epoch. Details in the
  changelog's Performance section. **libsumo must be installed** — the
  engine falls back to TCP TraCI with one warning line and a SUMO step
  then runs about 13× slower.

## Measuring

Always compare back-to-back on the same machine state — absolute ms/step
drifts by 2–3× with background load — and read the learner's own counters
(`training_step`, `updates_performed`, buffer size) rather than the episode
summary alone: a learner that never acts still produces plausible-looking
summaries.

Read the learner's init line too: it names the training device
(`... device=cuda ...`). A `torch` resolved from PyPI is the CPU build, and
a manifest that does not pin PyTorch's own index per platform (as this
one does under `[target.<platform>.pypi-dependencies]`) trains on the CPU
without any other sign -- the acting replica then never engages either.

A one-episode profile is the quickest honest measurement. Copy the bundled
config root, set `max_episodes: 1` in the profile you want to time, and run
the CLI under cProfile:

```bash
pixi run python -m cProfile -o prof.out -m opencda_marl.cli run -t sumo-marl/intersection/route_speed_only_pcmappo --config-root ./prof_configs
```

Then read it with `pstats` sorted by cumulative time. Two checks before
trusting a number: the episode summary must be identical to the unprofiled
run's (a behaviour-preserving change reproduces the reward bit for bit on
the same seed), and unrelated functions must not have moved together — a
uniform 1.5–2× inflation across everything is another process on the
machine, not the change under test.
