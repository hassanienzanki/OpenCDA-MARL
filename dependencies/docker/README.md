# CARLA Docker Workflows

This directory separates the CARLA server from the training code.

## Mode A: CARLA in Docker, Training on the Host

This is the default Linux-server workflow. Run one CARLA container per matrix
worker/GPU/port, then run Pixi training on the host against that RPC port.

```bash
pixi run carla-docker-pull
pixi run carla-docker-start -- --gpu 0 --rpc-port 2000 --streaming-port 0
pixi run python scripts/experiments/launch_matrix.py \
  --conditions speed_only_pcmappo \
  --regimes aligned \
  --seeds 0 \
  --carla-rpc-port 2000 \
  --carla-restart-mode none
```

`--carla-restart-mode none` is important when CARLA is externally managed by
Docker. It makes the matrix fail/skip if the RPC server is down instead of
killing local CARLA processes or trying to launch a host binary.

For multiple workers, start one server per port:

```bash
pixi run carla-docker-start -- --name opencda-carla-2000 --gpu 0 --rpc-port 2000
pixi run carla-docker-start -- --name opencda-carla-2010 --gpu 1 --rpc-port 2010
```

Use `scripts/experiments/workers.example.yaml` as the assignment template. The
parallelism is cell-level: one CARLA server plus one matrix cell per GPU/port.

## Mode B: Optional Mounted Training Runner

Use this when you want the training Python environment in Docker as well. The
repository is bind-mounted, so code/config changes on the host are visible
inside the container immediately and do not require rebuilding the image.

```bash
pixi run train-docker-build
pixi run train-docker-run -- pixi run python scripts/experiments/launch_matrix.py --dry-run
pixi run train-docker-run -- pixi run python scripts/experiments/launch_matrix.py \
  --conditions speed_only_pcmappo \
  --regimes aligned \
  --seeds 0 \
  --carla-rpc-port 2000 \
  --carla-restart-mode none
```

The runner image does not start CARLA. Keep the CARLA server container separate
so each training runner can target an explicit RPC port. For first-time parallel
setup, prewarm the shared Pixi volume once, or give each worker a distinct
`--pixi-volume` to avoid simultaneous environment writes.

## Images

- `dependencies/docker/Dockerfile` is a thin wrapper around
  `carlasim/carla:0.9.16` with a stable headless entry point.
- `dependencies/docker/Dockerfile.train` is an optional CUDA/Pixi/SUMO runner
  image. It intentionally does not copy the repo into the image.

The official CARLA image can also be used directly by `scripts/docker_carla.py`;
building the wrapper is optional.
