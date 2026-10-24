"""Run a 200vph-replay eval pass against every matrix cell's best_checkpoint.

Workflow per cell:
  1. Parse the slug to recover (condition, regime, seed, map).
  2. Regenerate the runtime YAML via launch_matrix._patch_config so the
     eval has the same scenario config the training cell used.
  3. Invoke scripts/run_carla_checkpoint_eval.py with --checkpoint pointing
     at the cell's best_checkpoint.pth and --eval-replay-file pointing at
     recordings/traffic_200vph.json.
  4. Collect the per-cell eval JSON.

After all cells eval, writes a combined summary JSON to
artifacts/eval/matrix_eval_summary.json.

Usage:
    pixi run python scripts/experiments/run_matrix_eval.py
    pixi run python scripts/experiments/run_matrix_eval.py --episodes 5
    pixi run python scripts/experiments/run_matrix_eval.py --slug-prefix matrix_
"""
from __future__ import annotations
import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "experiments"))

from launch_matrix import (  # noqa: E402
    CONFIG_BASE,
    CONFIG_ROOT,
    CONDITION_CONFIGS,
    RUNTIME_CONFIG_ROOT,
    _patch_config,
)
from matrix_common import (  # noqa: E402
    DEFAULT_SLUG_PREFIX,
    MAP_PROFILES,
    SLUG_PREFIX_PATTERN,
    mixed_prefix_warning,
    prepend_marl_config_root,
)

CHECKPOINTS_ROOT = REPO / "artifacts" / "checkpoints" / "matrix"
EVAL_OUT_ROOT = REPO / "artifacts" / "eval" / "matrix"
SUMMARY_PATH = REPO / "artifacts" / "eval" / "matrix_eval_summary.json"

# The map and condition alternations come from the launcher's own tables, so
# they cannot drift apart: a condition the launcher can run is one this can
# parse, and a slug this parses always has a map profile to regenerate from.
# Longest names first, so a prefix never shadows the name it is a prefix of.
_MAPS = "|".join(re.escape(m) for m in sorted(MAP_PROFILES, key=len, reverse=True))
_CONDITIONS = "|".join(re.escape(c) for c in sorted(CONDITION_CONFIGS, key=len, reverse=True))
SLUG_RE = re.compile(
    SLUG_PREFIX_PATTERN
    + rf"(?P<map>{_MAPS})_"
    rf"(?P<condition>{_CONDITIONS})_"
    r"(?P<regime>aligned|conflicting)_"
    r"seed(?P<seed>\d+)"
)


def parse_slug(slug: str):
    # fullmatch, not match: an evaluation slug is a training slug plus a
    # suffix (`..._seed0_eval200vph`), and a prefix match parsed it as the
    # training cell it was derived from -- so eval outcomes silently replaced
    # the training numbers in the aggregates keyed by (action, algo, regime).
    m = SLUG_RE.fullmatch(slug)
    if not m:
        return None
    return m.groupdict()


def regenerate_runtime_yaml(slug: str, meta: dict, *, max_steps: int,
                            active_junctions, carla_rpc_port: int) -> Path:
    """Re-run the launcher's _patch_config to write a fresh runtime YAML.

    ``condition`` is a required argument of ``_patch_config`` (it selects the
    continuation source); calling without it raised TypeError, which made
    this entry point unusable. The run-time knobs (active junctions, CARLA
    RPC port) are parameters rather than constants; ``active_junctions=None``
    keeps the map profile's own.

    A continuation cell (``continue_*``) looks its source cell up under
    the cell's own slug prefix, the run it belongs to; an empty prefix is a
    valid one.
    """
    condition = meta["condition"]
    regime = meta["regime"]
    seed = int(meta["seed"])
    map_slug = meta["map"]
    src_name = CONDITION_CONFIGS[condition]
    src = CONFIG_BASE / src_name
    dst = RUNTIME_CONFIG_ROOT / f"{slug}.yaml"
    _patch_config(
        src, dst,
        regime=regime, seed=seed, slug=slug,
        condition=condition, map_slug=map_slug,
        max_episodes=50, max_steps=max_steps,
        active_junctions=active_junctions, carla_rpc_port=carla_rpc_port,
        continue_source_prefix=meta.get("prefix", DEFAULT_SLUG_PREFIX),
    )
    return dst


def select_slugs(slugs, *, cells=None, slug_prefix: str | None = None) -> list:
    """The checkpoint directories to evaluate, in order.

    ``cells`` keeps only the named slugs; ``slug_prefix`` keeps only the
    cells of that run (slugs that parse under exactly that prefix). ``None``
    keeps every prefix.
    """
    selected = list(slugs)
    if cells:
        selected = [s for s in selected if s in cells]
    if slug_prefix is not None:
        selected = [s for s in selected
                    if (parse_slug(s) or {}).get("prefix") == slug_prefix]
    return selected


def find_best_checkpoint(cell_dir: Path) -> Path | None:
    candidates = sorted(cell_dir.rglob("best_checkpoint.pth"))
    return candidates[0] if candidates else None


def scenario_ref_for(yaml_path: Path) -> str:
    rel = yaml_path.relative_to(CONFIG_ROOT)
    return str(rel.with_suffix("")).replace("\\", "/")


def run_eval_for_cell(slug: str, ckpt: Path, scenario_ref: str,
                      episodes: int, *, max_steps: int = 2400,
                      replay_file: str = "recordings/traffic_200vph.json") -> dict:
    out_dir = EVAL_OUT_ROOT / slug
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "eval_summary.json"
    cmd = [
        "pixi", "run", "python", "scripts/run_carla_checkpoint_eval.py",
        "--scenario", scenario_ref,
        "--checkpoint", str(ckpt),
        "--episodes", str(episodes),
        "--seed", "3",
        "--max-steps", str(max_steps),
        "--eval-traffic-mode", "replay",
        "--eval-replay-file", replay_file,
        "--output", str(out_file),
    ]
    print(f"\n[eval] {slug}")
    print(f"  ckpt: {ckpt}")
    print(f"  out:  {out_file}")
    print(f"  cmd:  {' '.join(cmd)}")
    # The eval subprocess needs OPENCDA_MARL_CONFIG_ROOTS to point at
    # configs so config_locator can find the runtime YAMLs we just
    # wrote there. Mirrors launch_matrix.py's pattern.
    env = os.environ.copy()
    prepend_marl_config_root(env, CONFIG_ROOT)
    t0 = time.time()
    rc = subprocess.run(cmd, cwd=REPO, env=env).returncode
    dt = time.time() - t0
    print(f"  rc={rc} in {dt:.0f}s")
    if rc != 0 or not out_file.exists():
        return {"slug": slug, "rc": rc, "error": "eval_failed_or_no_output"}
    return json.loads(out_file.read_text())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=3,
                   help="Eval episodes per cell (default 3)")
    p.add_argument("--cells", nargs="+", default=None,
                   help="Optional cell-slug filter; default all.")
    p.add_argument("--max-steps", type=int, default=2400,
                   help="Steps per eval episode (default 2400).")
    p.add_argument("--active-junctions", nargs="+", type=int, default=None,
                   help="Override scenario.traffic.active_junctions for every "
                        "map (default: each map profile's own).")
    p.add_argument("--carla-rpc-port", type=int, default=2000)
    p.add_argument("--replay-file", default="recordings/traffic_200vph.json",
                   help="Replay traffic for the eval pass.")
    p.add_argument("--slug-prefix", default=None,
                   help="Evaluate only the cells of the run launched with this "
                        "slug prefix (default: every prefix found).")
    args = p.parse_args()

    EVAL_OUT_ROOT.mkdir(parents=True, exist_ok=True)

    if not CHECKPOINTS_ROOT.exists():
        print(f"No checkpoints at {CHECKPOINTS_ROOT}; nothing to evaluate.")
        return
    all_cells = select_slugs(
        sorted(d.name for d in CHECKPOINTS_ROOT.iterdir() if d.is_dir()),
        cells=args.cells, slug_prefix=args.slug_prefix)

    print(f"=== EVAL PASS ({len(all_cells)} cells, {args.episodes} eps each) ===")
    print(f"Replay file: {args.replay_file}")
    warning = mixed_prefix_warning(
        meta["prefix"] for meta in map(parse_slug, all_cells) if meta)
    if warning:
        print(warning)

    summaries = {}
    for i, slug in enumerate(all_cells, start=1):
        print(f"\n--- [{i}/{len(all_cells)}] {slug} ---")
        meta = parse_slug(slug)
        if meta is None:
            print(f"  SKIP: unparseable slug")
            continue
        cell_dir = CHECKPOINTS_ROOT / slug
        ckpt = find_best_checkpoint(cell_dir)
        if ckpt is None:
            print(f"  SKIP: no best_checkpoint.pth")
            continue
        try:
            yaml_path = regenerate_runtime_yaml(
                slug, meta, max_steps=args.max_steps,
                active_junctions=args.active_junctions,
                carla_rpc_port=args.carla_rpc_port)
        except FileNotFoundError as exc:
            # A continuation cell whose source checkpoint is gone: record it
            # and go on, so one cell never costs the summary of the others.
            print(f"  SKIP: {exc}")
            summaries[slug] = {"slug": slug, "meta": meta,
                               "error": f"runtime_config_failed: {exc}"}
            continue
        scenario_ref = scenario_ref_for(yaml_path)
        result = run_eval_for_cell(slug, ckpt, scenario_ref, args.episodes,
                                 max_steps=args.max_steps,
                                 replay_file=args.replay_file)
        result["slug"] = slug
        result["meta"] = meta
        summaries[slug] = result

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"\n=== DONE. Summary written to {SUMMARY_PATH} ===")
    print(f"Cells evaluated: {len(summaries)}/{len(all_cells)}")


if __name__ == "__main__":
    main()
