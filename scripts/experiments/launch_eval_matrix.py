"""Evaluate trained matrix cells against a deterministic traffic recording.

For each completed training cell (one with a best_checkpoint.pth), this
script derives an "eval" runtime config that:

  * loads the saved best_checkpoint.pth,
  * switches traffic to replay mode against the selected recording,
  * sets training_mode=false (no gradient updates),
  * caps episodes at ``--eval-episodes`` (default 5).

The eval pass runs the same opencda.py entry point with the patched
config; output goes to ``artifacts/checkpoints/matrix/<slug><eval-suffix>/``
so it does not collide with training outputs. Training cells are looked up
under ``--slug-prefix`` (default ``matrix_``), the prefix their training run
was launched with; unprefixed ``--legacy-slugs`` names never satisfy the
eval checks by accident.

Usage:

    pixi run python scripts/experiments/launch_eval_matrix.py \\
        --conditions speed_only_pcmappo rule_path_pcmappo learn_path_pcmappo \\
        --regimes aligned conflicting \\
        --seeds 0 1 2 \\
        --eval-episodes 5

The script is idempotent: a cell whose eval directory already has the
target episode count is skipped (resume-safe).
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

from omegaconf import OmegaConf

from launch_matrix import _needs_ml
from matrix_common import (
    DEFAULT_MAP,
    DEFAULT_SLUG_PREFIX,
    canonical_map,
    canonical_maps,
    map_profile,
    preference_block_path,
    patch_map_profile,
    prepend_marl_config_root,
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
CHECKPOINTS_ROOT = ARTIFACTS_ROOT / "checkpoints" / "matrix"
METRICS_ROOT = ARTIFACTS_ROOT / "metrics_history" / "matrix"
LOGS_ROOT = ARTIFACTS_ROOT / "logs"

CONDITION_CONFIGS = {
    "speed_only_pcmappo":          "speed_only_pcmappo.yaml",
    "keep_current_path_pcmappo":   "keep_current_path_pcmappo.yaml",
    "random_legal_path_pcmappo":   "random_legal_path_pcmappo.yaml",
    "rule_path_pcmappo":           "rule_path_pcmappo.yaml",
    "learn_path_pcmappo":          "learn_path_pcmappo.yaml",
    "continue_learn_path_pcmappo": "learn_path_pcmappo.yaml",
}

CHECKPOINT_SOURCE_CONDITIONS = {
    # RulePath is not a trained route policy. It applies the rule-based route
    # selector on top of the matching speed-only checkpoint: the speed
    # policy, the legal route set and the mask are that condition's own, and
    # only the route selector differs.
    "rule_path_pcmappo": "speed_only_pcmappo",
}


@dataclass(frozen=True)
class Cell:
    map_slug: str
    condition: str
    regime: str
    seed: int

    def train_slug(self, prefix: str = DEFAULT_SLUG_PREFIX,
                   legacy: bool = False) -> str:
        return slug_for(
            map_slug=self.map_slug,
            condition=self.condition,
            regime=self.regime,
            seed=self.seed,
            prefix=prefix,
            legacy=legacy,
        )

    def eval_slug(self, prefix: str = DEFAULT_SLUG_PREFIX,
                  legacy: bool = False,
                  suffix: str = "_eval200vph") -> str:
        return f"{self.train_slug(prefix, legacy=legacy)}{suffix}"


def _runtime_root(map_slug: str) -> Path:
    return RUNTIME_CONFIG_ROOT


def _best_checkpoint(
    cell: Cell,
    slug_prefix: str,
    legacy_slug: bool,
    *,
    checkpoint_root: Path,
    checkpoint_map: str | None = None,
) -> Optional[Path]:
    lookup_cell = (
        Cell(
            str(checkpoint_map),
            CHECKPOINT_SOURCE_CONDITIONS.get(cell.condition, cell.condition),
            cell.regime,
            cell.seed,
        )
        if checkpoint_map is not None
        else Cell(
            cell.map_slug,
            CHECKPOINT_SOURCE_CONDITIONS.get(cell.condition, cell.condition),
            cell.regime,
            cell.seed,
        )
    )
    cp_root = checkpoint_root / lookup_cell.train_slug(
        slug_prefix,
        legacy=legacy_slug,
    )
    if not cp_root.exists():
        return None
    for cand in cp_root.rglob("best_checkpoint.pth"):
        return cand
    for cand in cp_root.rglob("latest_checkpoint.pth"):
        return cand
    return None


def _scenario_ref(cell_yaml: Path) -> str:
    """``family/scenario/profile`` for a runtime cell or a shipped profile."""
    for root in (RUNTIME_ROOT, CONFIG_ROOT):
        try:
            rel = cell_yaml.relative_to(root)
        except ValueError:
            continue
        return "/".join(rel.with_suffix("").parts)
    raise ValueError(f"{cell_yaml} is under neither {RUNTIME_ROOT} nor {CONFIG_ROOT}")


def _patch_eval_config(src: Path, dst: Path, cell: Cell,
                        checkpoint: Path, eval_episodes: int,
                        max_steps: Optional[int],
                        slug_prefix: str,
                        legacy_slug: bool = False,
                        active_junctions: Iterable[int] | None = None,
                        replay_file: str | None = None,
                        carla_rpc_port: int = 2000,
                        eval_suffix: str = "_eval200vph",
                        force_intent: str | None = None) -> None:
    cfg = OmegaConf.load(src)
    validate_matrix_schema(cfg, source=str(src))

    profile = map_profile(cell.map_slug)
    patch_map_profile(
        cfg,
        cell.map_slug,
        active_junctions=active_junctions,
        replay_file=replay_file or profile.replay_file,
        carla_rpc_port=carla_rpc_port,
    )
    OmegaConf.update(cfg, "world.seed", int(cell.seed), merge=True, force_add=True)

    pref_path = preference_block_path(cfg)
    if pref_path:
        OmegaConf.update(cfg, f"{pref_path}.regime", cell.regime,
                          merge=True, force_add=True)
        OmegaConf.update(cfg, f"{pref_path}.seed",
                          int(cell.seed) + 100, merge=True, force_add=True)
        OmegaConf.update(cfg, f"{pref_path}.dim", 2,
                          merge=True, force_add=True)
    if cfg.get("MARL") is not None and cfg["MARL"].get("preference") is not None:
        OmegaConf.update(cfg, "MARL.preference.profile", "balanced",
                          merge=True, force_add=True)
        OmegaConf.update(cfg, "MARL.preference.dim", 2,
                          merge=True, force_add=True)
    # Switch traffic to replay against the selected deterministic recording,
    # so every condition is evaluated on the same traffic.
    OmegaConf.update(cfg, "scenario.traffic.mode", "replay",
                      merge=True, force_add=True)
    OmegaConf.update(cfg, "scenario.traffic.replay_file",
                      replay_file or profile.replay_file,
                      merge=True, force_add=True)

    OmegaConf.update(cfg, "scenario.simulation.max_episodes", int(eval_episodes),
                      merge=True, force_add=True)
    if max_steps is not None:
        OmegaConf.update(cfg, "scenario.simulation.max_steps", int(max_steps),
                          merge=True, force_add=True)

    # Disable learning, load the trained checkpoint.
    OmegaConf.update(cfg, "MARL.training.training_mode", False,
                      merge=True, force_add=True)
    OmegaConf.update(cfg, "MARL.training.save_checkpoints", False,
                      merge=True, force_add=True)
    OmegaConf.update(cfg, "MARL.training.load_checkpoint", str(checkpoint),
                      merge=True, force_add=True)
    if force_intent is not None:
        OmegaConf.update(
            cfg,
            "MARL.action_space.path.force_intent",
            str(force_intent),
            merge=True,
            force_add=True,
        )

    eval_slug = cell.eval_slug(
        slug_prefix,
        legacy=legacy_slug,
        suffix=eval_suffix,
    )

    # Output directories isolated under the eval suffix slug.
    OmegaConf.update(cfg, "outputs.metrics_store.run_name", eval_slug, force_add=True)
    OmegaConf.update(cfg, "outputs.metrics_archive.dir",
                      f"metrics_history/matrix/{eval_slug}",
                      merge=True, force_add=True)
    OmegaConf.update(cfg, "MARL.training.checkpoint_dir",
                      f"artifacts/checkpoints/matrix/{eval_slug}",
                      merge=True, force_add=True)

    dst.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, dst)


def _eval_done(cell: Cell, min_episodes: int, slug_prefix: str,
               legacy_slug: bool, eval_suffix: str) -> bool:
    eval_slug = cell.eval_slug(
        slug_prefix,
        legacy=legacy_slug,
        suffix=eval_suffix,
    )
    cp_root = CHECKPOINTS_ROOT / eval_slug
    if cp_root.exists():
        for meta in cp_root.rglob("training_metadata.json"):
            try:
                hist = json.loads(meta.read_text()).get("history", [])
                if len(hist) >= min_episodes:
                    return True
            except (json.JSONDecodeError, OSError):
                continue
    if _metrics_episode_count(METRICS_ROOT / eval_slug) >= min_episodes:
        return True
    return False


def _metrics_episode_count(archive: Path) -> int:
    if not archive.exists():
        return 0
    json_records = [
        path for path in archive.glob("*.json")
        if path.name != "training_metadata.json"
    ]
    if json_records:
        return len(json_records)
    count = 0
    for pkl_path in sorted(archive.glob("metrics_history_ep*.pkl")):
        try:
            with pkl_path.open("rb") as f:
                history = pickle.load(f)
            count += len(history.get("rewards", [])) if isinstance(history, dict) else 0
        except (OSError, pickle.PickleError, EOFError):
            continue
    return count


def _launch_eval(cell: Cell, eval_episodes: int, max_steps: Optional[int],
                  cell_timeout_s: Optional[float], dry_run: bool,
                  slug_prefix: str,
                  legacy_slug: bool = False,
                  active_junctions: Iterable[int] | None = None,
                  replay_file: str | None = None,
                  carla_rpc_port: int = 2000,
                  cuda_visible_devices: str | None = None,
                  checkpoint_root: Path = CHECKPOINTS_ROOT,
                  checkpoint_map: str | None = None,
                  eval_suffix: str = "_eval200vph",
                  force_intent: str | None = None) -> int:
    src_name = CONDITION_CONFIGS.get(cell.condition)
    if src_name is None:
        raise ValueError(f"unknown condition {cell.condition!r}")
    src = CONFIG_BASE / src_name
    train_slug = cell.train_slug(slug_prefix, legacy=legacy_slug)
    eval_slug = cell.eval_slug(
        slug_prefix,
        legacy=legacy_slug,
        suffix=eval_suffix,
    )
    checkpoint = _best_checkpoint(
        cell,
        slug_prefix,
        legacy_slug,
        checkpoint_root=checkpoint_root,
        checkpoint_map=checkpoint_map,
    )
    if checkpoint is None:
        checkpoint_condition = CHECKPOINT_SOURCE_CONDITIONS.get(
            cell.condition,
            cell.condition,
        )
        checkpoint_cell = (
            Cell(str(checkpoint_map), checkpoint_condition, cell.regime, cell.seed)
            if checkpoint_map is not None
            else Cell(cell.map_slug, checkpoint_condition, cell.regime, cell.seed)
        )
        checkpoint_slug = checkpoint_cell.train_slug(
            slug_prefix,
            legacy=legacy_slug,
        )
        print(
            f"[eval] skip (no checkpoint yet): {checkpoint_root / checkpoint_slug}",
            flush=True,
        )
        return -1

    cell_yaml = _runtime_root(cell.map_slug) / f"{eval_slug}.yaml"
    if not dry_run:
        _patch_eval_config(src, cell_yaml, cell, checkpoint, eval_episodes,
                           max_steps, slug_prefix, legacy_slug=legacy_slug,
                           active_junctions=active_junctions,
                           replay_file=replay_file,
                           carla_rpc_port=carla_rpc_port,
                           eval_suffix=eval_suffix,
                           force_intent=force_intent)

    log_dir = LOGS_ROOT / eval_slug
    log_file = log_dir / "run.log"
    scenario_ref = _scenario_ref(cell_yaml)
    # --apply_ml only when the scenario activates perception (see
    # launch_matrix._needs_ml): otherwise it loads YOLOv8 for nothing.
    cmd = [sys.executable, "opencda.py", "-t", scenario_ref, "--marl"]
    if _needs_ml(src):
        cmd.append("--apply_ml")
    print(f"[eval] launching {eval_slug} ({cell.map_slug}) -> {scenario_ref}", flush=True)
    print(f"[eval]   checkpoint: {checkpoint}", flush=True)
    print(f"[eval]   log: {log_file}", flush=True)
    if cell_timeout_s is not None:
        print(f"[eval]   timeout: {cell_timeout_s:.0f}s", flush=True)
    if dry_run:
        return 0
    log_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    prepend_marl_config_root(env, RUNTIME_ROOT)
    if cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)
    with log_file.open("w") as f:
        try:
            rc = subprocess.run(cmd, cwd=REPO, stdout=f, stderr=subprocess.STDOUT,
                                 timeout=cell_timeout_s, env=env).returncode
        except subprocess.TimeoutExpired:
            print(f"[eval]   {eval_slug} TIMEOUT after {cell_timeout_s:.0f}s",
                  flush=True)
            return 124
    print(f"[eval]   {eval_slug} exited rc={rc}", flush=True)
    return rc


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--maps", nargs="+", default=[DEFAULT_MAP],
                    help="Map profiles to evaluate (default: custom). Valid: custom.")
    p.add_argument("--conditions", nargs="+",
                    default=["speed_only_pcmappo", "rule_path_pcmappo", "learn_path_pcmappo"])
    p.add_argument("--regimes", nargs="+", default=["aligned", "conflicting"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--eval-episodes", type=int, default=5)
    p.add_argument("--max-steps", type=int, default=2400,
                    help="Per-episode max steps. The 200vph recording has spawns up "
                         "to step ~2156, so eval should run >= 2400 to include the "
                         "full traffic pattern.")
    p.add_argument("--cell-timeout", type=float, default=2000.0,
                    help="Per-eval-cell timeout, seconds. 5 episodes x 2400 steps x "
                         "~100 ms/step ~= 1200s; add safety margin.")
    p.add_argument("--resume", action="store_true",
                    help="Skip cells whose eval dir already has >= eval-episodes records.")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--active-junctions", nargs="+", type=int, default=None,
                    help="Override scenario.traffic.active_junctions for every map.")
    p.add_argument("--replay-file", default=None,
                    help="Override the map-profile replay file.")
    p.add_argument("--carla-rpc-port", type=int, default=2000)
    p.add_argument("--cuda-visible-devices", default=None,
                    help="Set CUDA_VISIBLE_DEVICES for the launched eval process.")
    p.add_argument("--slug-prefix", default=DEFAULT_SLUG_PREFIX,
                    help=f"Prefix for artifact slugs (default: {DEFAULT_SLUG_PREFIX}); "
                         "pass the prefix the training run was launched with.")
    p.add_argument("--legacy-slugs", action="store_true",
                    help="Use historical unprefixed slugs; only for diagnostics.")
    p.add_argument("--checkpoint-root", type=Path, default=CHECKPOINTS_ROOT,
                    help="Root containing trained checkpoint cell directories.")
    p.add_argument("--checkpoint-map", default=None,
                    help="Use checkpoints from this map profile while evaluating --maps.")
    p.add_argument("--eval-suffix", default="_eval200vph",
                    help="Suffix appended to each eval output slug.")
    p.add_argument(
        "--force-intent",
        choices=["left", "straight", "right", "keep-current"],
        default=None,
        help=(
            "Optional what-if evaluation of the same checkpoint with its route "
            "head pinned: the legal mask is collapsed to the requested intent; "
            "use a unique --eval-suffix."
        ),
    )
    args = p.parse_args()

    # The canonical profile key names the eval cells and the training cells
    # they load, as the launcher named them (``--maps Custom`` reads and
    # writes ``matrix_custom_*``).
    args.maps = canonical_maps(args.maps)
    if args.checkpoint_map is not None:
        args.checkpoint_map = canonical_map(args.checkpoint_map)
    cells = [Cell(m, c, r, int(s)) for m in args.maps for c in args.conditions
             for r in args.regimes for s in args.seeds]
    slug_prefix = "" if args.legacy_slugs else str(args.slug_prefix)
    print(f"[eval] planning {len(cells)} eval cells", flush=True)
    print(f"[eval] maps: {', '.join(args.maps)}", flush=True)
    print(f"[eval] eval_episodes={args.eval_episodes} max_steps={args.max_steps}", flush=True)
    print(f"[eval] slug prefix: {slug_prefix!r}", flush=True)
    for cell in cells:
        if args.resume and _eval_done(
            cell,
            args.eval_episodes,
            slug_prefix,
            args.legacy_slugs,
            args.eval_suffix,
        ):
            print(
                f"[eval] skip (already done): "
                f"{cell.eval_slug(slug_prefix, legacy=args.legacy_slugs, suffix=args.eval_suffix)}",
                flush=True,
            )
            continue
        _launch_eval(cell, args.eval_episodes, args.max_steps,
                      args.cell_timeout, args.dry_run, slug_prefix,
                      legacy_slug=args.legacy_slugs,
                      active_junctions=args.active_junctions,
                      replay_file=args.replay_file,
                      carla_rpc_port=args.carla_rpc_port,
                      cuda_visible_devices=args.cuda_visible_devices,
                      checkpoint_root=args.checkpoint_root,
                      checkpoint_map=args.checkpoint_map,
                      eval_suffix=args.eval_suffix,
                      force_intent=args.force_intent)


if __name__ == "__main__":
    main()
