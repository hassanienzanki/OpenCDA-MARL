# CARLA Version Support

This page tracks the CARLA variants currently wired into this repository.

## Support Matrix

| Target | Python | Wheel | Pixi manifest | Status |
| --- | --- | --- | --- | --- |
| Default Windows experiment baseline | 3.12 | `dependencies/whl/carla-0.9.16-cp312-cp312-win_amd64.whl` | `pixi.toml`, `dependencies/pixi/pixi_win_py312_carla0916.toml` | Default Windows path, import-validated |
| Windows 0.9.15 compatibility path | 3.12 | `dependencies/whl/carla-0.9.15-cp312-cp312-win_amd64.whl` | `dependencies/pixi/pixi_win_py312_carla0915.toml` | Compatibility manifest |
| Windows 0.9.15 backup baseline | 3.10 | `dependencies/whl/carla-0.9.15-cp310-cp310-win_amd64.whl` | `dependencies/pixi/pixi_win_py310_carla0915_backup.toml`, `dependencies/pixi/pixi_winCUDA.toml`, `dependencies/pixi/pixi_winRTX5090.toml` | Legacy backup |
| Windows 0.10.0 experiment | 3.12 | `dependencies/whl/carla-0.10.0-cp312-cp312-win_amd64.whl` | `dependencies/pixi/pixi_win_py312_carla0100_experimental.toml` | Experimental only |
| **Linux baseline** | 3.10 | `dependencies/whl/carla-0.9.16-cp310-cp310-manylinux_2_31_x86_64.whl` | `pixi.toml` (`[target.linux-64]`) | Default Linux path, **training-validated** (see below) |
| Linux 0.9.15 fallback | 3.10 | `dependencies/whl/carla-0.9.15-cp310-cp310-manylinux_2_27_x86_64.whl` | not wired into a manifest | Available, unused |

## Validation Summary

- Local import probe passed for `carla-0.9.16-cp312-cp312-win_amd64.whl`.
- Local import probe passed for `carla-0.9.15-cp312-cp312-win_amd64.whl`.
- Local import probe passed for `carla-0.10.0-cp312-cp312-win_amd64.whl`.
- The `0.9.15 + Python 3.12` wheel was built locally from the CARLA `0.9.15` source tag.
- Full scenario regression is still pending, so the root Windows manifest should still be treated as an experiment default rather than a fully locked support baseline.
- Linux (`0.9.16` server + `cp310` client wheel, Ubuntu 24.04) is validated beyond an import
  probe: MARL training runs (MASAC, PCMASAC, MAPPO, PCMAPPO and the path-action profiles) run to
  completion, single and four-way parallel with one server per GPU, through periodic world
  reloads, checkpoint save/resume, and both evaluation paths. No crash or core dump was observed
  across repeated server start/stop cycles.
- The server binary is not packaged here: extract the official Linux release and point the
  launchers at it with `--carla-bin /path/to/CarlaUE4.sh` (or `CARLA_BIN`), or run it in Docker
  (`scripts/docker_carla.py`, see the MARL throughput page).
- CARLA prints `Disabling core dumps.` during normal start-up on Linux — it is a UE4 status
  line, not a failure. A server that exits within a few seconds of start-up, leaving only that
  line, is almost always an environment problem rather than a CARLA bug: the known one here is
  `SDL_VIDEODRIVER=offscreen` leaking into the server's environment, which UE4's bundled SDL2
  does not support (the matrix launcher strips it).

The import probe checked the presence of symbols heavily used by the current project:
`command`, `TrafficManager`, `VehicleLightState`, `WeatherParameters`, and `LaneType`.

## Assessment

### CARLA 0.9.16

`0.9.16` is the main follow-up candidate for this repository.

- The current codebase only contains explicit CARLA compatibility branches for older `0.9.11` and `0.9.12` blueprint naming differences.
- No repo-level evidence was found that `0.9.15` depends on a deep API behavior that is absent in `0.9.16`.
- Python `3.10` is covered: `0.9.16` ships here as both a `cp312` Windows wheel and a `cp310`
  manylinux wheel (the Linux target pins Python 3.10 because of it).

Recommended validation targets for `0.9.16`:

- `single_2lanefree_carla`
- `single_town06_carla`
- `single_town06_cosim`
- `v2xp_online_carla`
- `marl-quick-test`

### CARLA 0.10.0

`0.10.0` is intentionally tracked as an experiment, not a supported runtime.

- Local import succeeded, which confirms that the Python module itself can be installed on Windows Python `3.12`.
- Current project support still looks mismatched at the feature level.
- This repository depends heavily on Town05/Town06 flows, customized `2lanefree` maps, Traffic Manager, SUMO co-simulation, V2X, and custom map import workflows.
- CARLA's official `0.10.0` release notes describe multiple gaps in exactly those areas, so the project should not advertise `0.10.0` as supported yet.

Reference:

- [CARLA 0.10.0 release notes](https://carla.org/2025/01/30/release-0.10.0/)

## Windows Manifest Selection

Use the Windows manifest that matches the CARLA wheel you want to validate:

- `pixi.toml`: default Windows experiment path `0.9.16 + Python 3.12`
- `dependencies/pixi/pixi_win_py310_carla0915_backup.toml`: backup `0.9.15 + Python 3.10`
- `dependencies/pixi/pixi_win_py312_carla0915.toml`: win-only `0.9.15 + Python 3.12`
- `dependencies/pixi/pixi_win_py312_carla0916.toml`: explicit `0.9.16 + Python 3.12`
- `dependencies/pixi/pixi_win_py312_carla0100_experimental.toml`: experimental `0.10.0 + Python 3.12`
