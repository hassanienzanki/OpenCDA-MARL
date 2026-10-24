# Installation Guide

## Prerequisites

- **Windows 10/11** (64-bit)
- **Python 3.12** for the root Windows manifest, or **Python 3.10** if you intentionally select the backup manifest
- **CUDA 12.8** (for GPU acceleration)
- [**CARLA 0.9.15 / 0.9.16 / 0.10.0**](https://github.com/carla-simulator/carla/releases)
- **Git**

Before installing, check the [CARLA Version Support](carla-version-support.md) matrix and choose the Windows manifest that matches your CARLA wheel and Python ABI.

## Step 1: Clone Repository

```bash
git clone https://github.com/radar-lab/opencda-marl.git
cd opencda-marl
```

## Step 2: Install Pixi

If you don't have pixi installed:

```bash
# Install pixi (Windows)
powershell -c "irm https://pixi.sh/install.ps1 | iex"

# Linux/macOS
curl -fsSL https://pixi.sh/install.sh | sh
```

## Step 3: Setup Environment

```bash
# Default root manifest on Windows: CARLA 0.9.16 + Python 3.12
# Alternative Windows manifests live in dependencies/pixi/
#   pixi_win_py310_carla0915_backup.toml
#   pixi_win_py312_carla0915.toml
#   pixi_win_py312_carla0916.toml
#   pixi_win_py312_carla0100_experimental.toml
#
# Linux remains on Python 3.10 because the bundled Linux CARLA wheel is cp310.
# Linux ROCm users should still use dependencies/pixi/pixi_Linux_ROCm.toml

# Install all dependencies
pixi install

# Test the installation: the test suite, the plugin registry, the SUMO
# bindings and a short SUMO run -- no CARLA server needed
pixi run verify
```

Windows manifest summary:

- `pixi.toml`: default Windows experiment path `CARLA 0.9.16 + Python 3.12`
- `dependencies/pixi/pixi_win_py310_carla0915_backup.toml`: backup `CARLA 0.9.15 + Python 3.10`
- `dependencies/pixi/pixi_win_py312_carla0915.toml`: win-only `CARLA 0.9.15 + Python 3.12`, backed by a locally built Windows wheel
- `dependencies/pixi/pixi_win_py312_carla0916.toml`: explicit `CARLA 0.9.16 + Python 3.12`
- `dependencies/pixi/pixi_win_py312_carla0100_experimental.toml`: experimental `CARLA 0.10.0 + Python 3.12`

## Step 4: Install CARLA

1. Download the CARLA version that matches your chosen Windows manifest:
   - `0.9.16` for the default Windows path
   - `0.9.15` for the compatibility manifests
   - `0.10.0` only for import-level experiments
2. Extract to a folder (for example `D:\Applications\CARLA_0.9.16`, `D:\Applications\CARLA_0.9.15`, or `D:\Applications\Carla-0.10.0-Win64-Shipping`)
3. Point the environment at it. The root manifest's Windows activation
   (`[target.win-64.activation.env]` in `pixi.toml`) sets `CARLA_ROOT`,
   `CARLA_HOME` and `CARLA_BIN` to the example location
   `D:\Applications\CARLA_0.9.16`, and activation overrides values set in
   your shell, so when CARLA lives elsewhere edit those three lines in
   `pixi.toml` (the matrix launcher also takes `--carla-root` /
   `--carla-bin`). Outside pixi, set the variable yourself:

   ```bash
   $env:CARLA_HOME = "D:\Applications\CARLA_0.9.16"
   ```
4. [Optional] Download our customized pre-built maps package from [Here](https://drive.google.com/drive/folders/1igtXiaSJL73HWWxwP0dh_zvvGX9oBSBa?usp=sharing) and extract it into the matching `CARLA_0.9.15` or `CARLA_0.9.16` folder.

Linux servers: download the Linux release (`CARLA_0.9.16.tar.gz` from the
official CARLA release bucket, ~8.3 GB) and extract it on a local disk
(not NFS). The pixi activation expects `/opt/CARLA_0.9.16`; if it lives
elsewhere, edit the `[target.linux-64.activation.env]` lines in `pixi.toml`
(activation overrides shell values) or pass `--carla-bin
/path/to/CarlaUE4.sh` to the launchers. The bundled Linux client wheel needs glibc ≥ 2.31
(declared in `pixi.toml`), and `pixi install --locked` needs pixi ≥ 0.73
(lock-file format 7). Keep the environment itself on local disk too:
`pixi config set --local detached-environments /local/path` before
`pixi install` when the checkout is on a network home directory.

Notes:

- `0.9.16` is the main follow-up candidate for current OpenCDA-MARL workflows.
- `0.10.0` is not a supported project runtime yet. Keep it limited to import and capability smoke tests until Traffic Manager, SUMO co-simulation, V2X, Town05/Town06, and custom map workflows are revalidated.
- The root Windows manifest now points to `0.9.16 + Python 3.12` for ongoing experiments.
- Keep `dependencies/pixi/pixi_win_py310_carla0915_backup.toml` as the immediate fallback if scenario regression reveals a `3.12` incompatibility.


## Step 5: Verify Installation

```bash
# Run CARLA server (in one terminal)
cd $env:CARLA_HOME
.\CarlaUE4.exe
# Or manually start the server by clicking the CarlaUE4.exe file

# Run OpenCDA test (in another terminal)
pixi run marl-quick-test
```

Recommended Windows verification order:

1. `CARLA 0.9.16 + Python 3.12` root experiment path
2. `CARLA 0.9.15 + Python 3.12` compatibility manifest
3. `CARLA 0.9.15 + Python 3.10` backup manifest
4. `CARLA 0.10.0 + Python 3.12` import-only experiment

## Development Environment

For documentation development:

```bash
# Activate docs environment
pixi shell -e docs

# Serve documentation locally
pixi run -e docs docs-serve
```

## Known issues

- **CARLA connection errors**: the server must be running before the client
  starts (`CarlaUE4 -RenderOffScreen` on a headless machine); the matrix
  launcher starts and restarts servers itself.
- **Perception with `--apply_ml`** loads YOLOv8 from the `ultralytics` package (the
  pixi environment installs it; the `yolov8m.pt` weights download on first use).
  Only when that package is missing does it fall back to YOLOv5 through
  `torch.hub`; on recent PyTorch a cached `models/common.py` from before
  Ultralytics' fix then raises
  `Expected 'device_type' of type 'str', got: '<class 'bool'>'`: delete the
  cached hub folder (`~/.cache/torch/hub/ultralytics_yolov5_master`) so
  `torch.hub` fetches the current YOLOv5 source, which passes the device type
  to `autocast`. That code is Ultralytics' (AGPL-3.0) and is not part of this
  repository.
- **A stale pixi environment** after a lock change: `pixi clean cache` then
  `pixi install`.
- **`libsumo import failed ...; falling back to traci`**: SUMO then runs
  over TraCI, about 13× slower per step. A pixi environment declares
  `libsumo`, so `pixi install` restores it. After a pip install of the
  wheel, install the binding that matches the bundled SUMO, as the warning
  says: `pip install libsumo==<SUMO version>` (the warning fills in the
  version of the SUMO binary it found). `OPENCDA_SUMO_LIB=traci` selects
  TraCI and silences the warning.
- **A system SUMO on the same machine** (named by `SUMO_HOME`, or by a
  `PYTHONPATH` entry for its `tools` directory) is ignored while the
  environment's `eclipse-sumo` package is installed: the engine uses that
  package's binary and bindings and drops every other SUMO `tools`
  directory from `sys.path`, so the client and the server always match.
  `SUMO_HOME` is used only when the package is absent.

## Getting help

1. Search existing [issues](https://github.com/radar-lab/OpenCDA-MARL/issues)
2. Open a new issue with the scenario reference, the command and the log
   lines around the failure
