# Thin CARLA server wrapper for OpenCDA-MARL experiments.
#
# You can run the official image directly, but keeping this wrapper gives the
# repo a stable command name and default headless flags:
#
#   docker build -f dependencies/docker/Dockerfile -t opencda-carla:0.9.16 .
#   docker run --rm --gpus all --network host opencda-carla:0.9.16
#
# For normal use, prefer:
#
#   pixi run carla-docker-start -- --gpu 0 --rpc-port 2000

FROM carlasim/carla:0.9.16

ENV CARLA_VERSION=0.9.16 \
    CARLA_ROOT=/home/carla \
    CARLA_HOME=/home/carla \
    CARLA_BIN=/home/carla/CarlaUE4.sh \
    SDL_VIDEODRIVER=offscreen \
    NVIDIA_DRIVER_CAPABILITIES=all

USER root
RUN printf '%s\n' \
    '#!/usr/bin/env bash' \
    'set -euo pipefail' \
    'cd "${CARLA_ROOT:-/home/carla}"' \
    'exec bash "${CARLA_BIN:-./CarlaUE4.sh}" -RenderOffScreen -nosound "$@"' \
    > /usr/local/bin/run-carla-headless \
    && chmod +x /usr/local/bin/run-carla-headless

WORKDIR /home/carla
CMD ["run-carla-headless", "-carla-rpc-port=2000", "-carla-streaming-port=0"]
