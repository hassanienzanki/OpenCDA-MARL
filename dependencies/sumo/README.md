# SUMO dependency setup

This folder documents the **user-mode (no sudo)** SUMO install used by the OpenCDA-MARL experiments. SUMO is installed through `pixi` as a regular PyPI wheel (`eclipse-sumo`), so the simulator binary lives inside the pixi env alongside `sumolib` and `traci`. No `apt-get install sumo`, no PPA, no system-level changes.

## The one-shot install

If you have pixi installed, the entire SUMO setup is:

```bash
pixi install
```

`pixi.toml` declares `eclipse-sumo = ">=1.24.0, <2"` under `[pypi-dependencies]`, which pulls a ~150 MB pre-built wheel from PyPI (Linux x86_64, Linux aarch64, macOS arm64, Windows x86_64). The wheel bundles the entire SUMO traffic simulator binary plus all its data files — no compilation, no system libraries needed.

After `pixi install` finishes you can use SUMO directly:

```bash
pixi run sumo --version
pixi run python -c "import sumolib, traci; print(sumolib.__version__, traci.__version__)"
pixi run python opencda.py -t sumo-marl/intersection/route_speed_only_pcmappo --marl
```

## Sanity check script

`install_sumo_ubuntu.sh` is no longer a sudo installer (despite its name kept for backward compatibility with old docs). It now verifies that the pixi-managed SUMO is healthy and optionally appends `SUMO_HOME` exports to your `~/.bashrc` for use *outside* the pixi env:

```bash
bash dependencies/sumo/install_sumo_ubuntu.sh             # full sanity + offer to update .bashrc
bash dependencies/sumo/install_sumo_ubuntu.sh --skip-shellrc
bash dependencies/sumo/install_sumo_ubuntu.sh --print-only  # print SUMO_HOME + binary path and exit
```

The script:
1. Confirms `pixi` is on PATH.
2. Locates the bundled SUMO under the pixi env via `python -c "import sumo, os; print(os.path.dirname(sumo.__file__))"`.
3. Resolves the actual `sumo` binary via `sumolib.checkBinary("sumo")`.
4. Runs `pixi run sumo --version` to verify it executes.
5. Imports `sumo`, `sumolib`, `traci` to confirm Python bindings work.
6. (Optional) Appends `export SUMO_HOME=<pixi-env-sumo-path>` and `PATH=$SUMO_HOME/bin:$PATH` to `~/.bashrc`.

## Why pixi-managed instead of system apt?

| Concern | apt+PPA install (old plan) | pixi wheel install (current plan) |
|---|---|---|
| Requires `sudo` on the target machine | yes | no |
| Version-pinned across machines | no (PPA serves whatever's latest) | yes (`pixi.lock` records exact wheel hash) |
| Reproducible after server rebuild | depends on PPA + apt cache | yes (just `pixi install`) |
| Works inside Docker / restricted containers | needs root | yes |
| First-time install time | 30 s + sudo prompt | ~30 s wheel download |
| Disk footprint | ~200 MB system-wide | ~200 MB inside pixi env |
| Coexistence with system SUMO | conflicts via PATH | isolated, pixi wins |

The pixi-wheel path also matches how this repo handles CARLA's Python bindings (`dependencies/whl/carla-0.9.16-*.whl` via `pixi.toml [target.linux-64.pypi-dependencies]`) — same pattern.

## Windows local dev

Same one-shot: `pixi install` brings down `eclipse_sumo-1.27.0-py3-none-win_amd64.whl` (~161 MB) automatically. No separate Windows installer / no manual `SUMO_HOME` env var work needed inside the pixi env.

If you want SUMO on PATH outside the pixi env on Windows, run the sanity script inside a Git Bash or WSL2 shell — the export it appends to `~/.bashrc` works there too.

## Platform support

Per the `eclipse-sumo` 1.27.0 wheels on PyPI:

| Platform | Wheel filename | Size |
|---|---|---|
| Linux x86_64 (manylinux 2.28) | `eclipse_sumo-1.27.0-py3-none-manylinux_2_28_x86_64.whl` | 145 MB |
| Linux x86_64 (manylinux 2014) | `eclipse_sumo-1.27.0-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl` | 87 MB |
| Linux aarch64 | `eclipse_sumo-1.27.0-py3-none-manylinux2014_aarch64.manylinux_2_17_aarch64.whl` | 77 MB |
| macOS arm64 (14+) | `eclipse_sumo-1.27.0-py3-none-macosx_14_0_arm64.whl` | 54 MB |
| Windows x86_64 | `eclipse_sumo-1.27.0-py3-none-win_amd64.whl` | 161 MB |

Pixi auto-selects the right wheel for your platform.

## Files in this folder

| File | Purpose |
|---|---|
| `README.md` | This file. |
| `install_sumo_ubuntu.sh` | Sanity-check script (no longer a sudo installer; legacy name). |

## Pixi-side configuration (where `eclipse-sumo` is wired in)

```toml
# pixi.toml
[pypi-dependencies]
eclipse-sumo = ">=1.24.0, <2"   # SUMO simulator binary (wheel-bundled, user-mode)
sumolib       = ">=1.24.0, <2"  # Python route/network parsing
traci         = ">=1.24.0, <2"  # Python TraCI runtime (TCP socket)
libsumo       = "==1.27.1"      # Shared-memory C++ binding (every platform)
```

The engine itself needs only `eclipse-sumo`: the package bundles `traci` and
`sumolib` under its own `sumo/tools`, which the engine's SUMO resolver puts
first on `sys.path`, so a plain `pip install` of the wheel imports them
without the separate distributions. pixi declares them anyway, and
`libsumo` is the optional accelerator.

### `traci` vs `libsumo` — two ways to call SUMO from Python

| Binding | How it works | Speed | Use when |
|---|---|---|---|
| `traci` | Python <-> SUMO over TCP socket; lets you talk to `sumo-gui` from a separate process | baseline | interactive debugging in `sumo-gui` (`OPENCDA_SUMO_LIB=traci`) |
| `libsumo` | CPython extension linked against SUMO's C++ library; same-process shared memory, no socket | **2-5x faster** per simulator call | training, headless eval (default; `OPENCDA_SUMO_LIB=libsumo`) |

Switching is a single env var:

```bash
OPENCDA_SUMO_LIB=libsumo pixi run python opencda.py -t sumo-marl/intersection/route_speed_only_pcmappo --marl
OPENCDA_SUMO_LIB=traci   pixi run python opencda.py -t sumo-marl/intersection/route_speed_only_pcmappo --marl
```

All in-repo SUMO call sites import via `opencda_marl/_kernel/traffic/_sumo_lib.py`, which resolves `traci` (the name) to either backend at import time. If `libsumo` is unavailable (an install without it, an old SUMO) the helper falls back to `traci` with a one-line warning so headless training still runs, markedly slower per SUMO step.

`libsumo` 1.27.1 ships wheels for Windows as well as Linux, so it is declared for every platform in `[pypi-dependencies]`; the `linux-64` target keeps a wider range only so older Linux solves stay valid.

The engine resolves SUMO from the `eclipse-sumo` package itself (`opencda_marl/_kernel/traffic/_sumo_home.py`) and falls back to `SUMO_HOME` only when the package is absent. On win-64 the pixi activation points `SUMO_HOME` at the environment's own package (`%CONDA_PREFIX%\Lib\site-packages\sumo`) for tools that read the variable; run the sanity script to print the path on Linux.

## Related files

- `pixi.toml` — wheel pin + SUMO_HOME env vars.
- `dependencies/docker/` — CARLA dependency setup (parallel pattern, container-based for GPU).
- `opencda_marl/_bundled/configs/sumo-marl/` — the SUMO scenario family that consumes this install.
