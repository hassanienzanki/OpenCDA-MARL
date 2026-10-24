# CARLA wheel inventory

This directory stores the repo-local CARLA Python wheels referenced by the Pixi manifests.

## Bundled Windows wheels

- `carla-0.9.15-cp310-cp310-win_amd64.whl`
- `carla-0.9.15-cp312-cp312-win_amd64.whl`
- `carla-0.9.16-cp312-cp312-win_amd64.whl`
- `carla-0.10.0-cp312-cp312-win_amd64.whl`

## Bundled Linux wheels

- `carla-0.9.15-cp310-cp310-manylinux_2_27_x86_64.whl`
- `carla-0.9.16-cp310-cp310-manylinux_2_31_x86_64.whl`

The `carla-0.9.15-cp312-cp312-win_amd64.whl` artifact was built locally from the
CARLA `0.9.15` source tag on Windows with Visual Studio 2022 and Python 3.12.

The `carla-0.9.16-cp310-cp310-manylinux_2_31_x86_64.whl` artifact is the
official PyPI wheel for CARLA 0.9.16 on CPython 3.10 / Linux x86-64. SHA256:
`52b1f2fafb0655e25954f9f6d1e97c211a4da404217fd1d0094b4b7350737c95`.
