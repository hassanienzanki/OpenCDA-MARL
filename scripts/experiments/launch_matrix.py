"""Drive the experiment matrix of the bundled ``momarl/intersection`` family.

Generates per-cell config copies under artifacts/_runtime/, patches the
regime / seed / output directories, and launches opencda.py for each
cell. A cell's slug is ``<prefix><map>_<condition>_<regime>_seed<N>`` with
``--slug-prefix`` defaulting to ``matrix_``. Resume and status checks match
only cells under the active prefix (and never the unprefixed
``--legacy-slugs`` names), so pass the prefix an earlier run used to resume
it. Continuation cells (``continue_*``) load their source cell from the
same prefix unless ``--continue-source-prefix`` names another run.

The launcher is wired to that CARLA family (``CONFIG_BASE``) and manages
the CARLA servers its cells need. SUMO profiles need no CARLA server: run
them directly (``python opencda.py -t sumo-marl/intersection/<profile>
--marl``), as many processes as there are cores.

Typical usage on a single machine:

    pixi run python scripts/experiments/launch_matrix.py \\
        --conditions speed_only_pcmappo rule_path_pcmappo learn_path_pcmappo \\
        --regimes aligned conflicting \\
        --seeds 0 1 2 \\
        --resume

The default --carla-restart-mode process starts and restarts CARLA itself,
so it needs CARLA_ROOT (or --carla-root / --carla-bin / CARLA_BIN); with
servers managed elsewhere pass --carla-restart-mode none (or docker).

The --resume flag skips cells whose final metrics archive already
exists (so a crashed machine can re-attach without redoing finished
cells).
"""
from __future__ import annotations

import argparse
import os
import re
import traceback
import shutil
import sys
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from omegaconf import OmegaConf

from matrix_common import (
    DEFAULT_MAP,
    DEFAULT_SLUG_PREFIX,
    canonical_map,
    canonical_maps,
    preference_block_path,
    patch_map_profile,
    prepend_marl_config_root,
    resolve_carla_binary,
    slug_for,
    validate_matrix_schema,
)

REPO = Path(__file__).resolve().parents[2]
CONFIG_ROOT = REPO / "opencda_marl" / "_bundled" / "configs"
CONFIG_BASE = CONFIG_ROOT / "momarl" / "intersection"
# Per-cell runtime profiles are run data: they live under artifacts/ as an
# extra config root (profile only; default and base come from the bundle).
RUNTIME_ROOT = REPO / "artifacts" / "_runtime"
RUNTIME_CONFIG_ROOT = RUNTIME_ROOT / "momarl" / "intersection"
ARTIFACTS_ROOT = REPO / "artifacts"
LOGS_ROOT = ARTIFACTS_ROOT / "logs"
CHECKPOINTS_MATRIX_ROOT = ARTIFACTS_ROOT / "checkpoints" / "matrix"
METRICS_MATRIX_ROOT = ARTIFACTS_ROOT / "metrics_history" / "matrix"
# The monitor's metrics store: <root>/metrics/<algorithm>/<run name>, and the
# run name of a cell is its slug (outputs.metrics_store.run_name).
METRICS_STORE_ROOT = ARTIFACTS_ROOT / "metrics"
TRACES_ROOT = ARTIFACTS_ROOT / "traces"

CONDITION_CONFIGS = {
    # The example family: one learner (PCMAPPO) under every built-in path
    # provider. This launcher is wired to the bundled family.
    "speed_only_pcmappo":          "speed_only_pcmappo.yaml",
    "keep_current_path_pcmappo":   "keep_current_path_pcmappo.yaml",
    "random_legal_path_pcmappo":   "random_legal_path_pcmappo.yaml",
    "rule_path_pcmappo":           "rule_path_pcmappo.yaml",
    "learn_path_pcmappo":          "learn_path_pcmappo.yaml",
    # A continuation cell: fine-tunes from the checkpoint of an existing
    # cell (its source, looked up under --continue-source-root) and writes
    # fresh artifacts under its own continue_* slug.
    "continue_learn_path_pcmappo": "learn_path_pcmappo.yaml",
}

# Continuation condition -> the condition of the cell it starts from.
CONTINUE_SOURCES = {
    "continue_learn_path_pcmappo": "learn_path_pcmappo",
}


@dataclass(frozen=True)
class Cell:
    map_slug: str
    condition: str
    regime: str
    seed: int

    def slug(self, prefix: str = DEFAULT_SLUG_PREFIX,
             legacy: bool = False) -> str:
        return slug_for(
            map_slug=self.map_slug,
            condition=self.condition,
            regime=self.regime,
            seed=self.seed,
            prefix=prefix,
            legacy=legacy,
        )


def _cells(maps: Iterable[str], conditions: Iterable[str], regimes: Iterable[str],
           seeds: Iterable[int]) -> List[Cell]:
    return [
        Cell(str(m), c, r, int(s))
        for m in maps for c in conditions for r in regimes for s in seeds
    ]


def _patch_config(src: Path, dst: Path, regime: str, seed: int, slug: str,
                   condition: str,
                   map_slug: str = DEFAULT_MAP,
                   max_episodes: int | None = None,
                   max_steps: int | None = None,
                   active_junctions: Iterable[int] | None = None,
                   carla_rpc_port: int | None = None,
                   enable_trace: bool = False,
                   trace_sample_every: int = 1,
                   continue_source_root: Path | None = None,
                   continue_source_map: str = DEFAULT_MAP,
                   cuda_graph_update: bool | None = None,
                   continue_source_prefix: str = DEFAULT_SLUG_PREFIX) -> None:
    cfg = OmegaConf.load(src)
    if cuda_graph_update is not None:
        # Learner-side CUDA-graph replay of the update (SAC family; ignored
        # by learners without the option). Set on the active algorithm's
        # section so the cell needs no hand-edited config.
        algo = cfg.get("MARL", {}).get("algorithm") if cfg.get("MARL") is not None else None
        if algo:
            OmegaConf.update(cfg, f"MARL.{algo}.cuda_graph_update", bool(cuda_graph_update),
                             merge=True, force_add=True)
    validate_matrix_schema(cfg, source=str(src))
    patch_map_profile(
        cfg,
        map_slug,
        active_junctions=active_junctions,
        carla_rpc_port=carla_rpc_port,
    )
    OmegaConf.update(cfg, "world.seed", int(seed), merge=True, force_add=True)
    pref_path = preference_block_path(cfg)
    if pref_path:
        OmegaConf.update(cfg, f"{pref_path}.regime", regime,
                          merge=True, force_add=True)
        OmegaConf.update(cfg, f"{pref_path}.seed", int(seed) + 100,
                          merge=True, force_add=True)
        OmegaConf.update(cfg, f"{pref_path}.dim", 2,
                          merge=True, force_add=True)
    if cfg.get("MARL") is not None and cfg["MARL"].get("preference") is not None:
        OmegaConf.update(cfg, "MARL.preference.profile", "balanced",
                          merge=True, force_add=True)
        OmegaConf.update(cfg, "MARL.preference.dim", 2,
                          merge=True, force_add=True)
    # The metrics store is keyed by run name: the slug, so the monitor shows
    # cells by name instead of by timestamp.
    OmegaConf.update(cfg, "outputs.metrics_store.run_name", slug, force_add=True)
    OmegaConf.update(cfg, "outputs.metrics_archive.dir",
                      f"metrics_history/matrix/{slug}",
                      merge=True, force_add=True)
    if enable_trace:
        OmegaConf.update(cfg, "outputs.trace.enabled", True,
                          merge=True, force_add=True)
        OmegaConf.update(cfg, "outputs.trace.dir", f"traces/{slug}",
                          merge=True, force_add=True)
        OmegaConf.update(cfg, "outputs.trace.sample_every", int(trace_sample_every),
                          merge=True, force_add=True)
        OmegaConf.update(cfg, "outputs.trace.include_routes", True,
                          merge=True, force_add=True)
    if cfg.get("MARL") is not None and cfg["MARL"].get("training") is not None:
        OmegaConf.update(cfg, "MARL.training.checkpoint_dir",
                          f"artifacts/checkpoints/matrix/{slug}",
                          merge=True, force_add=True)
    if max_episodes is not None:
        OmegaConf.update(cfg, "scenario.simulation.max_episodes", int(max_episodes),
                          merge=True, force_add=True)
    if max_steps is not None:
        OmegaConf.update(cfg, "scenario.simulation.max_steps", int(max_steps),
                          merge=True, force_add=True)
    _maybe_inject_continue_checkpoint(
        cfg,
        condition=condition,
        regime=regime,
        seed=seed,
        continue_source_root=continue_source_root,
        continue_source_map=continue_source_map,
        continue_source_prefix=continue_source_prefix,
    )
    resume_ckpt = _maybe_inject_resume_checkpoint(cfg, slug)
    if resume_ckpt is not None:
        print(f"[matrix]   partial-resume: {slug} loading {resume_ckpt}",
              flush=True)
    dst.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, dst)


def _maybe_inject_continue_checkpoint(
    cfg,
    *,
    condition: str,
    regime: str,
    seed: int,
    continue_source_root: Path | None,
    continue_source_map: str,
    continue_source_prefix: str = DEFAULT_SLUG_PREFIX,
) -> None:
    """Load the source cell's checkpoint for a continuation cell.

    The continuation conditions are new artifact slugs, but the loaded source
    checkpoint is the corresponding non-continued cell, e.g.
    continue_learn_path_pcmappo -> learn_path_pcmappo. The continuation
    writes under its own slug, so it never overwrites the source cell. A
    load_checkpoint already set in the profile wins.
    """
    source_condition = CONTINUE_SOURCES.get(str(condition))
    if source_condition is None:
        return
    if cfg.get("MARL") is None or cfg["MARL"].get("training") is None:
        return
    training = cfg["MARL"]["training"]
    existing_ckpt = (training.get("load_checkpoint") or "").strip()
    if existing_ckpt:
        return
    root = continue_source_root or CHECKPOINTS_MATRIX_ROOT
    source_slug = slug_for(
        map_slug=continue_source_map,
        condition=source_condition,
        regime=regime,
        seed=seed,
        prefix=continue_source_prefix,
    )
    source_dir = root / source_slug
    candidates = list(source_dir.rglob("best_checkpoint.pth"))
    if not candidates:
        candidates = list(source_dir.rglob("latest_checkpoint.pth"))
    if not candidates:
        raise FileNotFoundError(
            f"continuation source checkpoint not found for {condition}: "
            f"{source_dir}"
        )
    ckpt = candidates[0]
    ckpt_abs = ckpt if ckpt.is_absolute() else (REPO / ckpt).resolve()
    try:
        ckpt_value = ckpt_abs.relative_to(REPO).as_posix()
    except ValueError:
        ckpt_value = ckpt_abs.as_posix()
    OmegaConf.update(
        cfg, "MARL.training.load_checkpoint", ckpt_value,
        merge=True, force_add=True,
    )
    print(
        f"[matrix]   continuation: {condition} loads {ckpt_value}",
        flush=True,
    )


def _maybe_inject_resume_checkpoint(cfg, slug: str) -> str | None:
    """Inject load_checkpoint and start_episode into the runtime YAML when
    a partial cell has saved weights from a prior interrupted run.

    Pairs with the launcher's --resume flag, which only skips fully-done
    cells (pkl_count >= max_episodes via _cell_done). Partial cells
    (1 <= pkl_count < max_episodes) fall through and would otherwise
    re-train from scratch with random init. This helper detects them and:
      1. Injects load_checkpoint = <abs path to latest_checkpoint.pth>
         so the inner coordinator loads the saved policy/optimizer/
         training_step before iterating its episode loop.
      2. Injects start_episode = <pkl_count> so the coordinator's loop
         runs `range(start_episode, max_episodes)` (true mid-cell
         continuation) and the metrics exporter's _episode_counter is
         offset to write ep{N+1}, ep{N+2}, ... pkls instead of
         overwriting the existing ep1..epN files.

    Resume contract (as of this version):
      * Cell broken at ep 35 with pkls ep1..ep35 on disk:
        - load_checkpoint  = latest_checkpoint.pth (warm weights)
        - start_episode    = 35
        - coordinator runs ep36..ep50, writes pkls ep36..ep50
        - earlier pkls (ep1..ep35) preserved untouched
      * Effective gradient steps: 35 (interrupted) + 15 (resumed) = 50

    Conservative gating:
      * MARL.training section must already exist in the config
      * MARL.training.load_checkpoint must NOT already be set
        (so explicit user overrides are honored)
      * latest_checkpoint.pth must exist under the cell's checkpoint dir
    """
    if cfg.get("MARL") is None or cfg["MARL"].get("training") is None:
        return None
    existing = (cfg["MARL"]["training"].get("load_checkpoint") or "").strip()
    if existing:
        return None
    cp_root = CHECKPOINTS_MATRIX_ROOT / slug
    if not cp_root.exists():
        return None
    candidates = sorted(cp_root.rglob("latest_checkpoint.pth"))
    if not candidates:
        return None
    latest = candidates[0]
    abs_path = str(latest.resolve())
    OmegaConf.update(
        cfg, "MARL.training.load_checkpoint", abs_path,
        merge=True, force_add=True,
    )
    # True mid-cell resume: count existing pkls on disk and inject
    # start_episode so the coordinator's loop skips already-finished episodes
    # AND the metrics exporter offsets its filename counter to write
    # ep{N+1}, ep{N+2}, ... instead of overwriting ep1..epN. If no pkls
    # exist yet, start_episode=0 and the run is identical to a fresh start
    # warm-started from the checkpoint.
    existing_pkls = _count_completed_episodes(slug)
    if existing_pkls > 0:
        OmegaConf.update(
            cfg, "MARL.training.start_episode", int(existing_pkls),
            merge=True, force_add=True,
        )
    return abs_path


def _count_completed_episodes(slug: str) -> int:
    """Count metrics_history_ep*.pkl files already on disk for a cell.

    Returns the highest valid episode index found, so a corrupt or sparse
    metrics dir doesn't trigger an incorrect resume offset. Returns 0 if
    no pkls exist or the dir is missing.
    """
    metrics_dir = METRICS_MATRIX_ROOT / slug
    if not metrics_dir.exists():
        return 0
    max_ep = 0
    for pkl_path in metrics_dir.glob("metrics_history_ep*.pkl"):
        stem = pkl_path.stem  # "metrics_history_ep35"
        try:
            ep_num = int(stem.split("ep")[-1])
            if ep_num > max_ep:
                max_ep = ep_num
        except (ValueError, IndexError):
            continue
    return max_ep


def _scenario_ref(cell_yaml: Path) -> str:
    """``family/scenario/profile`` for a runtime cell or a shipped profile."""
    for root in (RUNTIME_ROOT, CONFIG_ROOT):
        try:
            rel = cell_yaml.relative_to(root)
        except ValueError:
            continue
        return "/".join(rel.with_suffix("").parts)
    raise ValueError(f"{cell_yaml} is under neither {RUNTIME_ROOT} nor {CONFIG_ROOT}")


def _carla_responsive(timeout_s: float = 10.0, rpc_port: int = 2000) -> bool:
    """Ping the CARLA server at 127.0.0.1:<rpc_port>."""
    try:
        import carla  # type: ignore[import-not-found]
    except Exception:
        return True  # if carla isn't importable here, assume cell will fail-fast itself
    try:
        client = carla.Client("127.0.0.1", int(rpc_port))
        client.set_timeout(timeout_s)
        _ = client.get_server_version()
        return True
    except Exception as e:
        print(f"[matrix]   CARLA ping failed: {e}", flush=True)
        return False


# CARLA server processes this launcher started, by RPC port. A restart
# kills the process tree it owns (Windows: taskkill /T; POSIX: the process
# group) and never touches servers on other ports -- with --parallel every
# slot runs its own server, and the old "kill every CarlaUE4 on the host"
# took the other slots down whenever one slot bounced.
_OWNED_CARLA: dict[int, "subprocess.Popen[bytes]"] = {}


def _kill_carla(rpc_port: int) -> None:
    """Stop the CARLA server bound to ``rpc_port`` (and only that one).

    Owned processes are killed by handle (tree / process group). Servers we
    did not start (or orphans of a previous run on the same port) are found
    by their command line, which carries ``-carla-rpc-port=<port>`` on both
    the launcher and the shipping binary.
    """
    proc = _OWNED_CARLA.pop(int(rpc_port), None)
    if proc is not None and proc.poll() is None:
        try:
            if os.name == "nt":
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15)
            else:
                import signal
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    proc.kill()
        except (subprocess.SubprocessError, OSError) as e:
            print(f"[matrix]   kill owned CARLA pid={proc.pid} warning: {e}", flush=True)
    marker = f"-carla-rpc-port={int(rpc_port)}"
    try:
        if os.name == "nt":
            # marker has no regex metacharacters, so it is used verbatim.
            ps = ("Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match "
                  f"'{marker}' " + "-and $_.Name -match 'CarlaUE4' } | "
                  "ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }")
            subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)
        else:
            # Anchored on the CARLA binary name and terminated after the port
            # digits: a bare "-carla-rpc-port=2000" also matched the launcher
            # process itself (its own argv carries the flag when the = form is
            # used) and any server on port 20000+.
            pattern = rf"CarlaUE4.*{re.escape(marker)}(\s|$)"
            subprocess.run(["pkill", "-9", "-f", "--", pattern],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15)
    except (subprocess.SubprocessError, OSError) as e:
        print(f"[matrix]   kill CARLA on port {rpc_port} warning: {e}", flush=True)


def _stop_owned_carla() -> None:
    """Stop every CARLA server this launcher started (never external ones).

    Called at exit unless ``--keep-carla`` is given: a finished matrix must
    not leave headless servers holding GPU memory on a shared machine.
    Servers found responsive at start-up are not owned and are left alone.
    """
    for port in sorted(_OWNED_CARLA):
        print(f"[matrix] stopping owned CARLA on port {port}", flush=True)
        _kill_carla(port)


def _carla_server_env() -> dict[str, str]:
    """Environment for a spawned CARLA server.

    The Linux pixi activation exports ``SDL_VIDEODRIVER=offscreen`` for the
    pygame client windows. UE4's bundled SDL2 has no such video driver, so
    a server started with it set exits immediately (rc=1, nothing in its
    log past "Disabling core dumps."). The server is headless through
    ``-RenderOffScreen`` and never needs the variable, so it is dropped.
    """
    return {k: v for k, v in os.environ.items() if k != "SDL_VIDEODRIVER"}


def _restart_carla(
    *,
    carla_bin: Path | None,
    rpc_port: int = 2000,
    streaming_port: int = 0,
    render: bool = False,
    mode: str = "process",
    docker_name: str = "opencda-carla-2000",
    graphics_adapter: int | None = None,
) -> bool:
    """Stop the CARLA server on ``rpc_port`` and start a fresh one there.

    Returns True if CARLA responds to a ping after the restart, else False.
    The matrix will skip the next cell if this fails.

    When ``render=False`` (default) CARLA is started with ``-RenderOffScreen``
    and ``-quality-level=Low`` for fast headless training. When ``render=True``
    CARLA is started in normal windowed mode at ``-quality-level=High`` so the
    simulation is watchable; window size and position are left to CARLA's
    defaults (the same behavior you get when launching ``CarlaUE4.exe`` by
    hand). This mode is meant for debugging and smoke runs, not full-matrix
    training.

    Important: CarlaUE4.exe is only the launcher; the actual engine runs as
    CarlaUE4-Win64-Shipping.exe. Killing only CarlaUE4.exe leaves the engine
    orphaned, which leads to multiple shipping processes accumulating and
    competing for the GPU.
    """
    print("[matrix]   restarting CARLA...", flush=True)
    if mode == "docker":
        # CARLA lives in a docker container. Skip the host-binary path
        # entirely; just bounce the container and wait for the RPC port to
        # come back up.
        print(f"[matrix]   docker restart {docker_name}", flush=True)
        try:
            subprocess.run(
                ["docker", "restart", docker_name],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                timeout=60, check=True,
            )
        except (subprocess.SubprocessError, OSError) as e:
            print(f"[matrix]   docker restart failed: {e}", flush=True)
            return False
        # CARLA inside the container needs ~10-20s to bind its RPC port.
        for attempt in range(24):
            time.sleep(5)
            if _carla_responsive(timeout_s=5.0, rpc_port=rpc_port):
                print(f"[matrix]   CARLA (docker) ready after "
                      f"{(attempt + 1) * 5}s", flush=True)
                return True
        print("[matrix]   CARLA (docker) failed to come up within 120s",
              flush=True)
        return False
    _kill_carla(rpc_port)
    time.sleep(5)
    if carla_bin is None or not carla_bin.exists():
        print(f"[matrix]   CARLA binary not at {carla_bin}; skipping restart", flush=True)
        return _carla_responsive(rpc_port=rpc_port)
    if render:
        # Normal windowed CARLA; let the engine pick its default size/position
        # like a hand-launched CarlaUE4.exe. -quality-level=High for a sharp
        # debug view without going all the way to Epic.
        carla_args = [str(carla_bin), "-quality-level=High",
                      f"-carla-rpc-port={int(rpc_port)}",
                      f"-carla-streaming-port={int(streaming_port)}"]
        print("[matrix]   render mode ON (windowed default, quality=High)",
              flush=True)
    else:
        # Headless training path. -RenderOffScreen disables the UE4 framebuffer
        # entirely, so window-size flags are no-ops; we omit them. Quality
        # stays Low to minimize per-tick simulation cost.
        carla_args = [str(carla_bin), "-RenderOffScreen", "-quality-level=Low",
                      f"-carla-rpc-port={int(rpc_port)}",
                      f"-carla-streaming-port={int(streaming_port)}"]
    if graphics_adapter is not None:
        # UE4 flag: which GPU renders this server. With several servers on
        # a multi-GPU box, spread them (see --carla-gpus).
        carla_args.append(f"-graphicsadapter={int(graphics_adapter)}")
    try:
        popen_kwargs = {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL,
                        "env": _carla_server_env()}
        if os.name != "nt":
            popen_kwargs["start_new_session"] = True   # own process group -> killpg
        _OWNED_CARLA[int(rpc_port)] = subprocess.Popen(carla_args, **popen_kwargs)
    except (OSError, subprocess.SubprocessError) as e:
        print(f"[matrix]   CARLA launch failed: {e}", flush=True)
        return False
    # Wait up to 60s for the RPC port to come up.
    for attempt in range(12):
        time.sleep(5)
        if _carla_responsive(timeout_s=5.0, rpc_port=rpc_port):
            print(f"[matrix]   CARLA ready after {(attempt + 1) * 5}s", flush=True)
            return True
    print("[matrix]   CARLA failed to come up within 60s", flush=True)
    return False


def _cell_done(slug: str, min_episodes: int = 1) -> bool:
    """A cell is 'done' if either the metrics_archive directory has
    enough exported records, OR the checkpoints dir has a long-enough
    history in training_metadata.json. The latter is what the
    OpenCDA-MARL platform actually populates. min_episodes lets a
    resumed matrix reject partial data from an earlier, shorter run
    that fell short of the current budget."""
    import json
    import pickle
    metrics_dir = METRICS_MATRIX_ROOT / slug
    if metrics_dir.exists():
        records = list(metrics_dir.glob("*.json"))
        if len(records) >= min_episodes:
            return True
        pkl_count = 0
        for pkl_path in sorted(metrics_dir.glob("metrics_history_ep*.pkl")):
            try:
                with pkl_path.open("rb") as f:
                    history = pickle.load(f)
                pkl_count += (
                    len(history.get("rewards", []))
                    if isinstance(history, dict)
                    else 0
                )
            except (OSError, pickle.PickleError, EOFError):
                continue
        if pkl_count >= min_episodes:
            return True
    cp_root = CHECKPOINTS_MATRIX_ROOT / slug
    if cp_root.exists():
        for meta in cp_root.rglob("training_metadata.json"):
            try:
                hist = json.loads(meta.read_text()).get("history", [])
                if len(hist) >= min_episodes:
                    return True
            except (json.JSONDecodeError, OSError):
                continue
    return False


def _safe_rmtree(path: Path, root: Path) -> None:
    resolved = path.resolve()
    resolved_root = root.resolve()
    if resolved == resolved_root or resolved_root not in resolved.parents:
        raise ValueError(f"refusing to delete outside artifact root: {resolved}")
    if path.exists():
        shutil.rmtree(path)


def _clean_cell_artifacts(slug: str) -> None:
    """Remove every artifact of one planned slug, and only that slug.

    Intended for cleanup after a smoke run, before a fresh launch. It never
    touches old unprefixed or previous-prefix diagnostic directories unless
    the caller explicitly planned those exact slugs.

    The monitor's metrics store goes too: it is keyed by the slug, so a
    relaunch would otherwise reopen the old ``metrics.sqlite`` and append the
    new run's steps and episodes to the old run's.
    """
    for root in (
        LOGS_ROOT,
        CHECKPOINTS_MATRIX_ROOT,
        METRICS_MATRIX_ROOT,
        TRACES_ROOT,
    ):
        _safe_rmtree(root / slug, root)
    if METRICS_STORE_ROOT.is_dir():
        # One directory per algorithm, the slug below it. Joined, not
        # globbed: a slug prefix may carry glob metacharacters.
        for algo_dir in METRICS_STORE_ROOT.iterdir():
            if algo_dir.is_dir() and (algo_dir / slug).is_dir():
                _safe_rmtree(algo_dir / slug, METRICS_STORE_ROOT)
    runtime_yaml = RUNTIME_CONFIG_ROOT / f"{slug}.yaml"
    if runtime_yaml.exists():
        runtime_yaml.unlink()


def _kill_process_tree(proc: "subprocess.Popen") -> None:
    """Kill a launched cell and everything it started.

    POSIX: the cell has its own session (``start_new_session``), so one
    ``killpg`` reaches the ``pixi`` wrapper and the python process under it.
    Windows: ``taskkill /T`` walks the child tree, which also ends a
    detached monitor the cell started (the cell is its parent); the next
    cell's run starts a fresh one for the same root.
    """
    if proc.poll() is not None:
        return
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)
        else:
            import signal
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                proc.kill()
    except (subprocess.SubprocessError, OSError) as e:
        print(f"[matrix]   kill cell pid={proc.pid} warning: {e}", flush=True)
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        pass


def _needs_ml(cell_yaml: Path) -> bool:
    """True when the cell's *effective* scenario activates camera perception.

    ``--apply_ml`` makes CavWorld build an MLManager, which loads YOLOv8 into
    memory (and downloads ~52 MB on a fresh checkout, per cell, racing between
    parallel slots). The perception manager only consults it when
    ``perception.activate`` is true; every matrix condition resolves to
    ``activate: false``, so the launcher used to pay for a model nothing reads
    -- and turned a firewalled host into a wall of failed cells.

    The runtime cell YAML is only the condition overlay, so the three layers
    the runtime merges (family default, scenario base, cell with its
    ``include:`` fragments) are merged here the same way, with
    ``smart_merge`` as ``opencda.py --marl`` and the ``opencda-marl`` CLI
    do: a plain merge would let an empty section in the cell hide a lower
    layer's ``perception.activate: true``. Anything unreadable falls back to
    True, i.e. the old behaviour.
    """
    try:
        # Imported here, not at module scope: this file is also run directly
        # as a script (python launch_matrix.py), where the repo root is not
        # yet on sys.path.
        if str(REPO) not in sys.path:
            sys.path.insert(0, str(REPO))
        from opencda_marl.utils import smart_merge
        from opencda_marl.utils.config_locator import (
            load_profile_with_includes, resolve_marl_config_paths)

        # The runtime cell is a profile-only extra root; default and base
        # resolve from the bundle behind it, exactly as the child run sees it.
        resolved = resolve_marl_config_paths(
            REPO, _scenario_ref(cell_yaml), custom_config_roots=[RUNTIME_ROOT])
        merged = smart_merge(OmegaConf.load(resolved["default_yaml"]),
                             OmegaConf.load(resolved["scenario_base_yaml"]))
        merged = smart_merge(
            merged, load_profile_with_includes(resolved["scenario_yaml"]))
        cfg = OmegaConf.to_container(merged, resolve=False)
    except Exception:                       # noqa: BLE001 - unreadable config
        return True
    stack = [cfg]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            perception = node.get("perception")
            if isinstance(perception, dict) and bool(perception.get("activate", False)):
                return True
            stack.extend(v for v in node.values() if isinstance(v, (dict, list)))
        elif isinstance(node, list):
            stack.extend(v for v in node if isinstance(v, (dict, list)))
    return False


def _launch_cell(cell: Cell, dry_run: bool = False,
                  max_episodes: int | None = None,
                  max_steps: int | None = None,
                  cell_timeout_s: float | None = None,
                  slug_prefix: str = DEFAULT_SLUG_PREFIX,
                  legacy_slug: bool = False,
                  active_junctions: Iterable[int] | None = None,
                  carla_rpc_port: int = 2000,
                  cuda_visible_devices: str | None = None,
                  enable_trace: bool = False,
                  trace_sample_every: int = 1,
                  continue_source_root: Path | None = None,
                  continue_source_map: str = DEFAULT_MAP,
                  cuda_graph_update: bool | None = None,
                  continue_source_prefix: str = DEFAULT_SLUG_PREFIX) -> int:
    src_name = CONDITION_CONFIGS.get(cell.condition)
    if src_name is None:
        raise ValueError(
            f"unknown condition {cell.condition!r}; "
            f"valid: {sorted(CONDITION_CONFIGS.keys())}"
        )
    src = CONFIG_BASE / src_name
    slug = cell.slug(slug_prefix, legacy=legacy_slug)
    cell_yaml = RUNTIME_CONFIG_ROOT / f"{slug}.yaml"
    if not dry_run:
        _patch_config(
            src,
            cell_yaml,
            cell.regime,
            cell.seed,
            slug,
            condition=cell.condition,
            map_slug=cell.map_slug,
            max_episodes=max_episodes,
            max_steps=max_steps,
            active_junctions=active_junctions,
            carla_rpc_port=carla_rpc_port,
            enable_trace=enable_trace,
            trace_sample_every=trace_sample_every,
            continue_source_root=continue_source_root,
            continue_source_map=continue_source_map,
            cuda_graph_update=cuda_graph_update,
            continue_source_prefix=continue_source_prefix,
        )

    log_dir = LOGS_ROOT / slug
    log_file = log_dir / "run.log"
    scenario_ref = _scenario_ref(cell_yaml)
    cmd = [
        "pixi", "run", "python", "opencda.py",
        "-t", scenario_ref,
        "--marl",
    ]
    # Decided from the source condition config: the runtime overlay does not
    # exist yet in --dry-run, and _patch_config never touches perception.
    if _needs_ml(src):
        cmd.append("--apply_ml")
    print(f"[matrix] launching {slug} ({cell.map_slug}) -> {scenario_ref}", flush=True)
    print(f"[matrix]   cmd: {' '.join(cmd)}", flush=True)
    print(f"[matrix]   log: {log_file}", flush=True)
    if cell_timeout_s is not None:
        print(f"[matrix]   timeout: {cell_timeout_s:.0f}s", flush=True)
    if dry_run:
        return 0
    log_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    prepend_marl_config_root(env, RUNTIME_ROOT)
    if cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)
    # The cell runs in its own process group / job so a timeout kills the whole
    # tree. subprocess.run(timeout=...) only kills the direct child -- here
    # `pixi`, whose `python opencda.py` grandchild survived on POSIX, kept
    # ticking the shared CARLA server and raced the next cell on the same RPC
    # port (and, on a shared node, kept its GPU memory).
    popen_kwargs = {"cwd": REPO, "stdout": None, "stderr": subprocess.STDOUT, "env": env}
    if os.name != "nt":
        popen_kwargs["start_new_session"] = True
    with log_file.open("w") as f:
        popen_kwargs["stdout"] = f
        proc = subprocess.Popen(cmd, **popen_kwargs)
        try:
            rc = proc.wait(timeout=cell_timeout_s)
        except subprocess.TimeoutExpired:
            _kill_process_tree(proc)
            print(f"[matrix]   {slug} TIMEOUT after {cell_timeout_s:.0f}s -- killed",
                  flush=True)
            return 124
        except KeyboardInterrupt:
            _kill_process_tree(proc)
            raise
    print(f"[matrix]   {slug} exited rc={rc}", flush=True)
    return rc


def _round_robin(spec, k: int):
    """Entry ``k mod n`` of a comma-separated list, or the value itself when it
    holds a single entry; ``None`` stays ``None``."""
    if spec is None:
        return None
    entries = [e.strip() for e in str(spec).split(",") if e.strip()]
    if not entries:
        return None
    return entries[k % len(entries)]


def _run_parallel(cells, args, *, slug_prefix: str, carla_bin, resume_min: int) -> int:
    """Run cells concurrently on one machine, one CARLA server per slot.

    Slot ``k`` owns rpc port ``base + 10*k`` and streaming port
    ``base_stream + 10*k`` (CARLA also binds rpc+1 and rpc+2, so a stride
    of 10 keeps slots apart). All slots share the GPU(s) in
    ``--cuda-visible-devices``; a single large card serves several
    learners comfortably because the per-step CARLA physics tick, not the
    network update, dominates wall-clock. Cells are pulled from one queue
    so a slot that finishes early immediately takes the next cell.
    """
    import queue
    import threading

    work: "queue.Queue[Cell]" = queue.Queue()
    for cell in cells:
        work.put(cell)
    n_slots = max(1, int(args.parallel))
    print(f"[matrix] parallel mode: {n_slots} slot(s), {work.qsize()} cell(s)",
          flush=True)
    results: dict[str, int] = {}
    lock = threading.Lock()

    def slot_worker(k: int) -> None:
        rpc = int(args.carla_rpc_port) + 10 * k
        stream = (int(args.carla_streaming_port) + 10 * k) if args.carla_streaming_port else 0
        # docker_carla.py names containers opencda-carla-<rpc-port>; a slot must
        # ask for the container that actually exists for its port (the old
        # "<base>-<slot>" suffix matched nothing, so every docker-mode restart
        # failed and the cell was abandoned).
        docker_name = (f"opencda-carla-{rpc}" if args.carla_restart_mode == "docker"
                       else args.carla_docker_name)
        # Learner GPU: with a list in --cuda-visible-devices, slot k takes
        # entry k mod n so the learners spread over the cards; a single
        # entry (or none) is passed through unchanged.
        learner_gpu = _round_robin(args.cuda_visible_devices, k)
        carla_gpu = _round_robin(getattr(args, "carla_gpus", None), k)
        carla_gpu = int(carla_gpu) if carla_gpu is not None else None
        tag = f"[matrix][slot {k} rpc={rpc}" + (f" gpu={learner_gpu}" if learner_gpu is not None else "") + "]"
        while True:
            try:
                cell = work.get_nowait()
            except queue.Empty:
                return
            slug = cell.slug(slug_prefix, legacy=args.legacy_slugs)
            try:
                if args.resume and _cell_done(slug, min_episodes=resume_min):
                    print(f"{tag} skip (already done): {slug}", flush=True)
                    continue
                if not args.dry_run and not _carla_responsive(rpc_port=rpc):
                    print(f"{tag} CARLA unresponsive; restarting", flush=True)
                    if args.carla_restart_mode == "none" or not _restart_carla(
                        carla_bin=carla_bin, rpc_port=rpc, streaming_port=stream,
                        render=args.render, mode=args.carla_restart_mode,
                        docker_name=docker_name, graphics_adapter=carla_gpu,
                    ):
                        print(f"{tag} abandon {slug}: CARLA unavailable", flush=True)
                        with lock:
                            results[slug] = -1
                        continue
                rc = _launch_cell(
                    cell, dry_run=args.dry_run,
                    max_episodes=args.max_episodes, max_steps=args.max_steps,
                    cell_timeout_s=args.cell_timeout, slug_prefix=slug_prefix,
                    legacy_slug=args.legacy_slugs,
                    active_junctions=args.active_junctions,
                    carla_rpc_port=rpc,
                    cuda_visible_devices=learner_gpu,
                    enable_trace=args.enable_trace,
                    trace_sample_every=args.trace_sample_every,
                    continue_source_root=args.continue_source_root,
                    continue_source_map=args.continue_source_map,
                    cuda_graph_update=getattr(args, "cuda_graph_update", None),
                    continue_source_prefix=getattr(
                        args, "continue_source_prefix", DEFAULT_SLUG_PREFIX),
                )
                with lock:
                    results[slug] = rc
                if rc != 0 and not args.dry_run and args.carla_restart_mode != "none":
                    print(f"{tag} cell rc={rc}; bouncing this slot's CARLA", flush=True)
                    _restart_carla(
                        carla_bin=carla_bin, rpc_port=rpc, streaming_port=stream,
                        render=args.render, mode=args.carla_restart_mode,
                        docker_name=docker_name, graphics_adapter=carla_gpu,
                    )
            except Exception as exc:                      # noqa: BLE001
                # A slot that raised (bad config, unreadable checkpoint, a
                # full disk) used to kill its thread quietly: the cell never
                # reached `results`, so the summary said '0 cell(s), 0
                # failed' and the launcher exited 0 with nothing run.
                print(f"{tag} ERROR {slug}: {type(exc).__name__}: {exc}", flush=True)
                traceback.print_exc()
                with lock:
                    results[slug] = -2
            finally:
                work.task_done()

    threads = [threading.Thread(target=slot_worker, args=(k,), daemon=True, name=f"slot-{k}")
               for k in range(n_slots)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    failed = {s: rc for s, rc in results.items() if rc != 0}
    print(f"[matrix] parallel run complete: {len(results)} cell(s), "
          f"{len(failed)} failed", flush=True)
    for slug, rc in sorted(failed.items()):
        print(f"[matrix]   FAILED rc={rc}: {slug}", flush=True)
    return len(failed)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--maps", nargs="+", default=[DEFAULT_MAP],
                    help="Map profiles to run (default: custom). Valid: custom.")
    p.add_argument("--conditions", nargs="+",
                    default=["speed_only_pcmappo", "rule_path_pcmappo", "learn_path_pcmappo"])
    p.add_argument("--regimes", nargs="+",
                    default=["aligned", "conflicting"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--resume", action="store_true")
    p.add_argument("--clean-planned-slugs", action="store_true",
                    help="Delete logs/checkpoints/metrics (archive and monitor "
                         "store)/traces for planned slugs before launch. "
                         "Use after smoke tests, before a fresh run.")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--max-episodes", type=int, default=None,
                    help="Override per-seed max_episodes (default: config value).")
    p.add_argument("--max-steps", type=int, default=None,
                    help="Override per-episode max_steps (default: config value).")
    p.add_argument("--cell-timeout", type=float, default=None,
                    help="Per-cell wall-clock timeout (seconds). A timed-out "
                         "cell is killed and reported rc=124; the matrix continues.")
    p.add_argument("--active-junctions", nargs="+", type=int, default=None,
                    help="Override scenario.traffic.active_junctions for every map.")
    p.add_argument("--carla-root", default=None,
                    help="CARLA root directory. Defaults to CARLA_ROOT. Required "
                         "(or --carla-bin / CARLA_BIN) for --carla-restart-mode process.")
    p.add_argument("--carla-bin", default=None,
                    help="Explicit CARLA executable. Overrides CARLA_BIN/CARLA_ROOT.")
    p.add_argument("--carla-rpc-port", type=int, default=2000)
    p.add_argument("--carla-streaming-port", type=int, default=0)
    p.add_argument("--carla-restart-mode", choices=["process", "docker", "none"],
                    default="process",
                    help="process manages a local CARLA binary; docker runs "
                         "`docker restart <--carla-docker-name>` between cells "
                         "and on hangs; none assumes CARLA is externally "
                         "managed and never touches it.")
    p.add_argument("--carla-docker-name", default="opencda-carla-2000",
                    help="Container name used when --carla-restart-mode=docker. "
                         "Defaults to opencda-carla-2000.")
    p.add_argument("--render", action="store_true",
                    help="Run CARLA with rendering ON (UE4 window visible) at "
                         "default window size and -quality-level=High. Use "
                         "for debug / smoke runs; significantly slower than "
                         "headless. Default: headless.")
    p.add_argument("--cuda-visible-devices", default=None,
                   help="CUDA_VISIBLE_DEVICES for the launched cell process. With "
                        "--parallel > 1 a comma-separated list is spread over the "
                        "slots round-robin (slot k -> entry k mod n), so learners "
                        "share the cards evenly; a single entry applies to all.")
    p.add_argument("--cuda-graph-update", dest="cuda_graph_update", action="store_const",
                   const=True, default=None,
                   help="Set MARL.<algorithm>.cuda_graph_update=true in every cell "
                        "(SAC family: replay the learner update as one CUDA graph; "
                        "see docs/marl/throughput.md). Off unless given.")
    p.add_argument("--carla-gpus", default=None,
                   help="Comma-separated GPU indices for the CARLA servers "
                        "(passed as -graphicsadapter=<idx>), spread over the "
                        "slots round-robin like --cuda-visible-devices. Only "
                        "for --carla-restart-mode process.")
    p.add_argument("--parallel", type=int, default=1,
                    help="Run this many cells concurrently on this machine, "
                         "one CARLA server per slot (slot k uses rpc port "
                         "--carla-rpc-port + 10*k, and streaming port "
                         "--carla-streaming-port + 10*k when set; docker "
                         "restart mode uses container opencda-carla-<rpc port>). "
                         "Slots share the GPU(s) in --cuda-visible-devices. "
                         "Default 1 = the sequential loop.")
    p.add_argument("--keep-carla", action="store_true",
                    help="Leave the CARLA servers this launcher started running "
                         "after the matrix (default: stop them at exit; servers "
                         "that were already up are never touched).")
    p.add_argument("--enable-trace", action="store_true",
                    help="Write per-step route-action traces under artifacts/traces/<slug>.")
    p.add_argument("--trace-sample-every", type=int, default=1,
                    help="Record one trace frame every N simulator steps when --enable-trace is set.")
    p.add_argument("--continue-source-root", type=Path, default=CHECKPOINTS_MATRIX_ROOT,
                    help="Checkpoint root holding the source cells that "
                         "continue_* conditions start from (default: the "
                         "matrix checkpoint root, artifacts/checkpoints/matrix).")
    p.add_argument("--continue-source-map", default=DEFAULT_MAP,
                    help="Map profile used to look up continue_* source checkpoints "
                         "(default: custom).")
    p.add_argument("--continue-source-prefix", default=None,
                    help="Slug prefix of the source cells that continue_* "
                         "conditions start from (default: the --slug-prefix in "
                         f"effect; {DEFAULT_SLUG_PREFIX} with --legacy-slugs).")
    p.add_argument("--slug-prefix", default=DEFAULT_SLUG_PREFIX,
                    help=f"Prefix for artifact slugs (default: {DEFAULT_SLUG_PREFIX}); "
                         "pass the prefix an earlier run used to resume it.")
    p.add_argument("--legacy-slugs", action="store_true",
                    help="Use historical unprefixed slugs; only for diagnostics.")
    p.add_argument("--partition", default=None,
                    help="Run only the K-th partition of the cell list out of "
                         "N total (e.g. --partition 1/3 / 2/3 / 3/3 for three "
                         "machines). Uses stride slicing (cells[K-1::N]). "
                         "Because seeds are the innermost loop of the cell "
                         "Cartesian product, --seeds 0 1 2 + --partition K/3 "
                         "perfectly assigns one seed to each of three machines. "
                         "Each machine produces a complete per-seed slice of "
                         "the matrix; if any machine fails, the surviving "
                         "machines still yield a valid (fewer-seeds) "
                         "result.")
    args = p.parse_args()

    # Validate map names before creating runtime configs, and keep the
    # canonical profile key from here on: it names the cells and their
    # continuation sources, and the eval runner and analyzers match it exactly.
    args.maps = canonical_maps(args.maps)
    args.continue_source_map = canonical_map(args.continue_source_map)
    cells = _cells(args.maps, args.conditions, args.regimes, args.seeds)
    if args.partition is not None:
        try:
            parts = args.partition.split("/")
            if len(parts) != 2:
                raise ValueError("expected exactly one '/'")
            k, n = int(parts[0]), int(parts[1])
            if n < 1 or not (1 <= k <= n):
                raise ValueError(f"K must satisfy 1 <= K <= N (got K={k}, N={n})")
        except (ValueError, TypeError) as exc:
            raise SystemExit(
                f"--partition must be K/N with integer K, N (e.g. 1/3); "
                f"got {args.partition!r}: {exc}"
            )
        full_len = len(cells)
        cells = cells[k - 1::n]
        print(
            f"[matrix] partition {k}/{n}: selected {len(cells)}/{full_len} cells "
            f"via stride slicing (cells[{k - 1}::{n}])",
            flush=True,
        )
    slug_prefix = "" if args.legacy_slugs else str(args.slug_prefix)
    # Continuation cells start from the same run's source cells unless told
    # otherwise; unprefixed legacy runs keep looking under the default.
    if args.continue_source_prefix is None:
        args.continue_source_prefix = (
            DEFAULT_SLUG_PREFIX if args.legacy_slugs else slug_prefix)
    # Only process mode starts CARLA from a binary; docker and none never
    # touch one, and a dry run starts nothing. Process mode without a known
    # binary stops here, before any cell, instead of after the first crash.
    carla_bin = None
    if args.carla_restart_mode == "process" and not args.dry_run:
        try:
            carla_bin = resolve_carla_binary(
                carla_root=args.carla_root,
                carla_bin=args.carla_bin,
            )
        except ValueError as exc:
            raise SystemExit(f"[matrix] {exc}")
    print(f"[matrix] planning {len(cells)} cells", flush=True)
    print(f"[matrix] maps: {', '.join(args.maps)}", flush=True)
    print(f"[matrix] slug prefix: {slug_prefix!r}", flush=True)
    if any(cell.condition in CONTINUE_SOURCES for cell in cells):
        print(f"[matrix] continuation source prefix: "
              f"{args.continue_source_prefix!r}", flush=True)
    print(f"[matrix] CARLA binary: {carla_bin if carla_bin is not None else 'not used'}",
          flush=True)
    print(f"[matrix] CARLA rpc/streaming ports: "
          f"{args.carla_rpc_port}/{args.carla_streaming_port}", flush=True)
    print(f"[matrix] CARLA restart mode: {args.carla_restart_mode}", flush=True)
    if args.render:
        print("[matrix] CARLA render mode: ON (windowed default, quality=High)",
              flush=True)
    if args.cuda_visible_devices is not None:
        print(f"[matrix] CUDA_VISIBLE_DEVICES={args.cuda_visible_devices}", flush=True)
    if args.max_episodes is not None:
        print(f"[matrix] per-seed max_episodes override: {args.max_episodes}", flush=True)
    if args.max_steps is not None:
        print(f"[matrix] per-episode max_steps override: {args.max_steps}", flush=True)
    if args.enable_trace:
        print(
            f"[matrix] route-action trace enabled "
            f"(sample_every={args.trace_sample_every})",
            flush=True,
        )
    if args.cell_timeout is not None:
        print(f"[matrix] per-cell timeout: {args.cell_timeout:.0f}s", flush=True)
    resume_min = args.max_episodes or 1
    if args.clean_planned_slugs:
        for cell in cells:
            slug = cell.slug(slug_prefix, legacy=args.legacy_slugs)
            print(f"[matrix] clean planned slug artifacts: {slug}", flush=True)
            if not args.dry_run:
                _clean_cell_artifacts(slug)
    try:
        if int(args.parallel) > 1:
            return _run_parallel(cells, args, slug_prefix=slug_prefix,
                                 carla_bin=carla_bin, resume_min=resume_min)
        failures = 0
        for cell in cells:
            slug = cell.slug(slug_prefix, legacy=args.legacy_slugs)
            if args.resume and _cell_done(slug, min_episodes=resume_min):
                print(f"[matrix] skip (already done, >= {resume_min} eps): {slug}",
                      flush=True)
                continue
            # Health-check CARLA before each cell; restart if dead.
            if not args.dry_run and not _carla_responsive(rpc_port=args.carla_rpc_port):
                print("[matrix]   CARLA unresponsive before cell launch", flush=True)
                if args.carla_restart_mode == "none":
                    print(
                        f"[matrix]   abandon {slug}: external CARLA is down and "
                        "restart-mode=none",
                        flush=True,
                    )
                    failures += 1
                    continue
                if not _restart_carla(
                    carla_bin=carla_bin,
                    rpc_port=args.carla_rpc_port,
                    streaming_port=args.carla_streaming_port,
                    render=args.render,
                    mode=args.carla_restart_mode,
                    docker_name=args.carla_docker_name,
                    graphics_adapter=_round_robin(getattr(args, "carla_gpus", None), 0),
                ):
                    print(f"[matrix]   abandon {slug}: CARLA failed to recover",
                          flush=True)
                    failures += 1
                    continue
            rc = _launch_cell(cell, dry_run=args.dry_run,
                               max_episodes=args.max_episodes,
                               max_steps=args.max_steps,
                               cell_timeout_s=args.cell_timeout,
                               slug_prefix=slug_prefix,
                               legacy_slug=args.legacy_slugs,
                               active_junctions=args.active_junctions,
                               carla_rpc_port=args.carla_rpc_port,
                               cuda_visible_devices=args.cuda_visible_devices,
                               enable_trace=args.enable_trace,
                               trace_sample_every=args.trace_sample_every,
                               continue_source_root=args.continue_source_root,
                               continue_source_map=args.continue_source_map,
                               cuda_graph_update=getattr(args, "cuda_graph_update", None),
                               continue_source_prefix=getattr(
                                   args, "continue_source_prefix", DEFAULT_SLUG_PREFIX))
            # If the cell failed or timed out, give CARLA a moment to recover and
            # explicitly bounce it before the next cell so a leaked RPC session does
            # not poison the next launch.
            if rc != 0:
                failures += 1
            if rc != 0 and not args.dry_run:
                if args.carla_restart_mode == "none":
                    print(
                        f"[matrix]   cell rc={rc}; external CARLA left untouched "
                        "because restart-mode=none",
                        flush=True,
                    )
                    continue
                print(f"[matrix]   cell rc={rc}; bouncing CARLA before next cell",
                      flush=True)
                _restart_carla(
                    carla_bin=carla_bin,
                    rpc_port=args.carla_rpc_port,
                    streaming_port=args.carla_streaming_port,
                    render=args.render,
                    mode=args.carla_restart_mode,
                    docker_name=args.carla_docker_name,
                    graphics_adapter=_round_robin(getattr(args, "carla_gpus", None), 0),
                )

        if failures:
            print(f"[matrix] {failures} cell(s) failed or were abandoned", flush=True)
        return failures
    finally:
        if not args.dry_run and not args.keep_carla:
            _stop_owned_carla()


if __name__ == "__main__":
    # Non-zero exit when any cell failed: a wrapper that only checks the
    # return code used to see success even when nothing ran.
    sys.exit(1 if main() else 0)
