"""How route-concentrated are the route choices of trained matrix cells?

The route selector's candidate_pressure_score adds
  same_route_risk_weight * same_route_count
to risk_cost (weight default 4.0): two CAVs that pick the EXACT same internal
route share a physical lane sequence through the intersection, and the
diversity term penalises that.

This tool reads the metrics archive of every route-head cell (rule_path,
learn_path, random_legal_path) under a matrix metrics root and counts how
often several agents picked the same route_id within one episode. High
overlap means the picks were route-concentrated and the diversity weight
would shift them onto less-shared variants; low overlap means it would
barely matter. Pure offline analysis of the archived pickles -- no
retraining.

Usage:
    pixi run python scripts/experiments/inspect_route_diversity.py \
        [--metrics-dir artifacts/metrics_history/matrix] [--slug-prefix matrix_]
"""
from __future__ import annotations
import argparse
import json
import pathlib
import pickle
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from matrix_common import SLUG_PREFIX_PATTERN, mixed_prefix_warning  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_METRICS_DIR = REPO / "artifacts" / "metrics_history" / "matrix"

SLUG_RE = re.compile(
    SLUG_PREFIX_PATTERN
    + r"(?P<map>custom)_"
    r"(?P<action>speed_only|rule_path|learn_path|keep_current_path|random_legal_path)_"
    r"(?P<algo>masac|pcmasac|mappo|pcmappo)_"
    r"(?P<regime>aligned|conflicting)_"
    r"seed(?P<seed>\d+)"
)


def per_episode_route_concentration(pkl_path: pathlib.Path) -> dict:
    """For one episode, return (n_agents, unique_route_ids, max_share, hist).

    max_share = max count of any single route_id across agents (i.e. how
    many CAVs followed the most popular route this episode). hist =
    {route_id: count}.
    """
    with pkl_path.open("rb") as f:
        d = pickle.load(f)
    cs = d.get("competition_summaries", [{}])[0]
    if not isinstance(cs, dict):
        return None
    tr = cs.get("terminal_records", [])
    if not tr:
        return None
    route_ids = []
    for rec in tr:
        ri = rec.get("route_intent", {})
        if isinstance(ri, dict):
            rid = ri.get("selected_route_id")
            if rid is not None:
                route_ids.append(int(rid))
    if not route_ids:
        return None
    hist = Counter(route_ids)
    n_agents = len(route_ids)
    return {
        "n_agents": n_agents,
        "n_unique_routes": len(hist),
        "max_share": max(hist.values()),
        "max_share_pct": 100.0 * max(hist.values()) / n_agents,
        "n_routes_with_overlap": sum(1 for c in hist.values() if c >= 2),
        "agents_in_overlap": sum(c for c in hist.values() if c >= 2),
        "hist": dict(hist),
    }


def analyze_cell(metrics_dir: pathlib.Path, slug: str) -> dict | None:
    cdir = metrics_dir / slug
    if not cdir.exists():
        return None
    pkls = sorted(cdir.glob("metrics_history_ep*.pkl"),
                  key=lambda p: int(p.stem.split("ep")[-1]))
    if not pkls:
        return None
    last_window = pkls[-25:]  # last 25 episodes
    metrics_per_ep = []
    for p in last_window:
        m = per_episode_route_concentration(p)
        if m is not None:
            metrics_per_ep.append(m)
    if not metrics_per_ep:
        return None
    # Aggregate
    avg_n_agents = sum(m["n_agents"] for m in metrics_per_ep) / len(metrics_per_ep)
    avg_unique = sum(m["n_unique_routes"] for m in metrics_per_ep) / len(metrics_per_ep)
    avg_max_share = sum(m["max_share"] for m in metrics_per_ep) / len(metrics_per_ep)
    avg_max_share_pct = sum(m["max_share_pct"] for m in metrics_per_ep) / len(metrics_per_ep)
    avg_agents_in_overlap = sum(m["agents_in_overlap"] for m in metrics_per_ep) / len(metrics_per_ep)
    return {
        "slug": slug,
        "n_episodes_analyzed": len(metrics_per_ep),
        "avg_n_agents": avg_n_agents,
        "avg_n_unique_routes": avg_unique,
        "avg_max_share": avg_max_share,
        "avg_max_share_pct": avg_max_share_pct,
        "avg_agents_in_overlap": avg_agents_in_overlap,
        "avg_pct_in_overlap": 100.0 * avg_agents_in_overlap / avg_n_agents if avg_n_agents else 0,
    }


def short_name(name: str) -> str:
    """The cell name without its map, for the table.

    The prefix stays: it names the run, and two runs of one cell must not
    print as two identical rows.
    """
    m = SLUG_RE.fullmatch(name)
    return m["prefix"] + name[m.start("action"):] if m else name


def route_head_cells(metrics_dir: pathlib.Path, slug_prefix: str | None = None) -> list:
    """(name, match) of the training cells that exercise the route head,
    sorted; with ``slug_prefix``, only the cells of that run."""
    cells = []
    for d in sorted(metrics_dir.iterdir()):
        if not d.is_dir():
            continue
        # fullmatch: eval slugs (`..._seed0_eval200vph`) are not training cells.
        m = SLUG_RE.fullmatch(d.name)
        if not m:
            continue
        if slug_prefix is not None and m["prefix"] != slug_prefix:
            continue
        # Only cells that actually exercise the route head
        if m["action"] not in ("rule_path", "learn_path", "random_legal_path"):
            continue
        cells.append((d.name, m))
    return cells


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metrics-dir", type=pathlib.Path, default=DEFAULT_METRICS_DIR,
                        help="Matrix metrics_history root (default: artifacts/metrics_history/matrix).")
    parser.add_argument("--slug-prefix", default=None,
                        help="Inspect only the cells of the run launched with this "
                             "slug prefix (default: every prefix found).")
    args = parser.parse_args(argv)
    metrics_dir = args.metrics_dir
    if not metrics_dir.is_dir():
        raise SystemExit(f"No metrics root at {metrics_dir}")
    cells = route_head_cells(metrics_dir, args.slug_prefix)
    warning = mixed_prefix_warning(m["prefix"] for _, m in cells)
    if warning:
        print(warning + "\n")
    print(f"{'cell':70} {'eps':>3} {'n_ag':>6} {'uniq':>5} {'max_share':>10} {'max%':>5} {'overlap%':>9}")
    print("-" * 115)
    rows = []
    for name, m in cells:
        info = analyze_cell(metrics_dir, name)
        if not info:
            continue
        rows.append(info)
        short = short_name(name)
        print(f"{short:70} {info['n_episodes_analyzed']:>3} "
              f"{info['avg_n_agents']:>6.1f} {info['avg_n_unique_routes']:>5.1f} "
              f"{info['avg_max_share']:>10.1f} {info['avg_max_share_pct']:>5.1f} "
              f"{info['avg_pct_in_overlap']:>9.1f}")

    print()
    print("=== interpretation ===")
    print("  n_ag             : avg agents that completed (success+collision+timeout) per ep")
    print("  uniq             : avg unique route_ids picked per ep (out of ~9 candidates total)")
    print("  max_share        : avg max count of agents on the SAME route_id per ep")
    print("  max%             : max_share / n_ag * 100  (concentration on the most-popular route)")
    print("  overlap%         : fraction of agents on a route shared with >=1 other agent")
    print()
    print("same_route_risk_weight (default 4.0) adds same_route_count to risk_cost;")
    print("the higher the overlap%, the more it redistributes picks away from")
    print("over-shared routes. If overlap% is already low, its effect is small.")
    print()
    print("Note: rule_path cells are not shown when their summaries record")
    print("route_intent=None (the deterministic rule selector does not emit the")
    print("per-decision record the learned head does).")


if __name__ == "__main__":
    main()
