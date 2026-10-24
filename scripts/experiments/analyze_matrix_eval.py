"""Aggregate per-cell eval JSONs and print summary tables.

Reads artifacts/eval/matrix/<slug>/eval_summary.json for each cell, parses
the slug into (condition, regime, seed), and produces five tiers of output:

  Tier 1 - Sanity: every cell has a completed eval, no broken JSONs
  Tier 2 - Per-cell summary: succ%, coll%, tmo%, throughput, utility
  Tier 3 - Action-space comparison: speed_only vs rule_path vs learn_path per (algo, regime)
  Tier 4 - Pareto spread: conflicting-regime cells, group_a vs group_b
  Tier 5 - Behavioral: route_intent_direction_preserved_violations
           (direction-preserving invariant must be 0)

Every row and Tier 3 group carries the cell's slug prefix, so the cells of
two runs under one root are never merged; ``--slug-prefix`` reads one run.

Usage:
    pixi run python scripts/experiments/analyze_matrix_eval.py
    pixi run python scripts/experiments/analyze_matrix_eval.py --eval-dir artifacts/eval/matrix
    pixi run python scripts/experiments/analyze_matrix_eval.py --slug-prefix matrix_
"""
from __future__ import annotations
import argparse
import json
import pathlib
import re
import statistics
import sys
from collections import defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from matrix_common import SLUG_PREFIX_PATTERN, mixed_prefix_warning  # noqa: E402

_CELL = (
    r"(?P<action>speed_only|rule_path|learn_path|keep_current_path|"
    r"random_legal_path)_"
    r"(?P<algo>masac|pcmasac|mappo|pcmappo)_"
    r"(?P<regime>aligned|conflicting)_"
    r"seed(?P<seed>\d+)"
)
SLUG_RE = re.compile(SLUG_PREFIX_PATTERN + r"(?P<map>custom)_" + _CELL)
# Every cell a run launched, continuation cells (``continue_<action>_<algo>``)
# included: --slug-prefix keeps all of them, as a load without the flag does,
# while the tier tables compare only the fresh cells SLUG_RE parses.
RUN_CELL_RE = re.compile(
    SLUG_PREFIX_PATTERN + r"(?P<map>custom)_(?:continue_)?" + _CELL)


def parse_slug(slug: str):
    # Prefix match on purpose: this analyzer reads the eval tree, where every
    # slug carries the eval suffix (`..._seed0_eval200vph`) after the seed.
    m = SLUG_RE.match(slug)
    return m.groupdict() if m else None


def slug_prefix_of(slug: str):
    """The run prefix of any cell of a run; None for a non-cell."""
    m = RUN_CELL_RE.match(slug)
    return m["prefix"] if m else None


def short_slug(slug: str) -> str:
    """The slug without its map, for the per-cell tables.

    The prefix stays: it names the run, and two runs of one cell must not
    print as two identical rows.
    """
    m = SLUG_RE.match(slug)
    return m["prefix"] + slug[m.start("action"):] if m else slug


def load_cells(eval_dir: pathlib.Path, slug_prefix: str | None = None):
    """Eval summaries by directory name. With ``slug_prefix``, only the
    cells of that run (directories that parse under exactly that prefix);
    ``None`` loads every directory."""
    cells = {}
    for sub in sorted(eval_dir.iterdir()):
        if not sub.is_dir():
            continue
        if slug_prefix is not None and slug_prefix_of(sub.name) != slug_prefix:
            continue
        summary = sub / "eval_summary.json"
        if not summary.exists():
            cells[sub.name] = None
            continue
        try:
            cells[sub.name] = json.loads(summary.read_text())
        except json.JSONDecodeError as exc:
            print(f"  WARN parsing {summary}: {exc}")
            cells[sub.name] = None
    return cells


def tier1_sanity(cells):
    print("=" * 100)
    print("TIER 1 - Sanity (each cell should have a complete eval)")
    print("=" * 100)
    bad = []
    for slug, data in cells.items():
        if data is None:
            print(f"  FAIL {slug}: missing or unreadable eval_summary.json")
            bad.append(slug)
        elif data.get("status") != "completed":
            print(f"  WARN {slug}: status={data.get('status')!r}")
            bad.append(slug)
        else:
            ep = data.get("completed_episodes", "?")
            ck = pathlib.Path(data.get("checkpoint", "")).name
            print(f"  OK   {slug}: {ep} eval eps, ckpt={ck}")
    if bad:
        print(f"\n  {len(bad)} cells need attention.")


def _overall(data):
    return data.get("overall", {}) if data else {}


def tier2_per_cell(cells):
    print("\n" + "=" * 100)
    print("TIER 2 - Per-cell eval summary (deterministic replay, mean over eval episodes)")
    print("=" * 100)
    print(f"{'cell':75} {'succ%':>6} {'coll%':>6} {'tput':>7} {'util':>10} {'eff':>10} {'risk':>8}")
    print("-" * 130)
    for slug, data in cells.items():
        if data is None:
            continue
        ov = _overall(data)
        short = short_slug(slug)
        print(
            f"{short:75} "
            f"{ov.get('mean_success_rate', 0):>6.1f} "
            f"{ov.get('mean_collision_rate', 0):>6.1f} "
            f"{ov.get('mean_throughput', 0):>7.0f} "
            f"{ov.get('mean_utility', 0):>10.0f} "
            f"{ov.get('mean_efficiency_total', 0):>10.0f} "
            f"{ov.get('mean_interaction_risk_total', 0):>8.0f}"
        )


def tier3_groups(cells):
    """(prefix, algo, regime) -> {action: overall} for the Tier 3 table.

    Keyed by the slug prefix too, so a cell of another run never becomes
    the speed_only baseline a path condition is compared against.
    """
    by_key = defaultdict(dict)
    for slug, data in cells.items():
        if data is None:
            continue
        m = parse_slug(slug)
        if not m:
            continue
        by_key[(m["prefix"], m["algo"], m["regime"])][m["action"]] = _overall(data)
    return by_key


def tier3_action_space_comparison(cells):
    """Action-space comparison: speed_only vs rule_path vs learn_path."""
    print("\n" + "=" * 100)
    print("TIER 3 - Action-space comparison per (run prefix, algo, regime)")
    print("=" * 100)
    by_key = tier3_groups(cells)

    print(f"\n{'run prefix':20} {'algo':10} {'regime':12} {'action':18} "
          f"{'succ%':>6} {'coll%':>6} {'tput':>7}")
    print("-" * 91)
    for (prefix, algo, regime), action_stats in sorted(by_key.items()):
        for action in ["speed_only", "rule_path", "learn_path"]:
            if action in action_stats:
                ov = action_stats[action]
                print(f"{prefix!r:20} {algo:10} {regime:12} {action:18} "
                      f"{ov.get('mean_success_rate', 0):>6.1f} "
                      f"{ov.get('mean_collision_rate', 0):>6.1f} "
                      f"{ov.get('mean_throughput', 0):>7.0f}")
        base = action_stats.get("speed_only")
        if base:
            for action in ["rule_path", "learn_path"]:
                if action in action_stats:
                    ov = action_stats[action]
                    d_succ = ov.get("mean_success_rate", 0) - base.get("mean_success_rate", 0)
                    d_coll = ov.get("mean_collision_rate", 0) - base.get("mean_collision_rate", 0)
                    print(f"  -> delta {action} vs speed_only: succ {d_succ:+.1f}pp, coll {d_coll:+.1f}pp")
        print()


def tier4_pareto_spread(cells):
    """Per conflicting cell: group_a (efficiency-leaning) vs group_b (risk-averse).

    Use per-agent utilities + preferences from episode_metrics[0] (deterministic
    replay - all eval eps identical so episode 0 is representative).
    """
    print("=" * 100)
    print("TIER 4 - Pareto spread (conflicting-regime cells; group_a vs group_b)")
    print("=" * 100)
    for slug, data in cells.items():
        if data is None:
            continue
        m = parse_slug(slug)
        if not m or m["regime"] != "conflicting":
            continue
        em_list = data.get("episode_metrics", [])
        if not em_list:
            continue
        em = em_list[0]
        au = em.get("agent_utilities", {})
        ap = em.get("agent_preferences", {})
        group_utils = defaultdict(list)
        for aid, u in au.items():
            pref = ap.get(aid) or ap.get(int(aid)) if isinstance(aid, str) else None
            if pref is None:
                continue
            group = "group_a" if pref[0] > pref[1] else "group_b"
            group_utils[group].append(float(u))
        short = short_slug(slug)
        print(f"\n  {short}:")
        if "group_a" in group_utils and "group_b" in group_utils:
            a_vals = group_utils["group_a"]
            b_vals = group_utils["group_b"]
            print(f"    group_a (n={len(a_vals)}, pref [0.75, 0.25]): "
                  f"mean util = {statistics.mean(a_vals):.0f}")
            print(f"    group_b (n={len(b_vals)}, pref [0.25, 0.75]): "
                  f"mean util = {statistics.mean(b_vals):.0f}")
            delta = statistics.mean(a_vals) - statistics.mean(b_vals)
            verdict = "OK spread" if abs(delta) > 50 else "WARN small/no spread"
            print(f"    delta a-b = {delta:+.0f}  [{verdict}]")
        else:
            print(f"    only one preference group found; "
                  f"agent_preferences may be sparse for replay traffic")
        # Also report the launcher-side competition summary
        comp = _overall(data).get("competition_summary", {})
        rank_by = comp.get("avg_completion_rank_by_preference_group", {})
        if rank_by:
            print(f"    avg_completion_rank by group: {rank_by}")


def tier5_invariant(cells):
    print("\n" + "=" * 100)
    print("TIER 5 - Behavioral invariant: direction_preserved_violations must be 0")
    print("=" * 100)
    bad = []
    for slug, data in cells.items():
        if data is None:
            continue
        em_list = data.get("episode_metrics", [])
        max_viol = 0
        for em in em_list:
            v = em.get("route_intent_direction_preserved_violations", 0) or 0
            max_viol = max(max_viol, v)
        if max_viol > 0:
            bad.append((slug, max_viol))
    if bad:
        print("  VIOLATIONS DETECTED (direction-preserving invariant broken):")
        for slug, v in bad:
            print(f"    {slug}: max={v}")
    else:
        print(f"  All {len(cells)} cells: direction_preserved_violations = 0 [OK] invariant holds")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--eval-dir", default="artifacts/eval/matrix")
    p.add_argument("--slug-prefix", default=None,
                   help="Analyze only the cells of the run launched with this "
                        "slug prefix (default: every prefix found).")
    args = p.parse_args(argv)
    eval_dir = pathlib.Path(args.eval_dir)
    if not eval_dir.exists():
        print(f"No eval dir at {eval_dir}")
        return
    cells = load_cells(eval_dir, args.slug_prefix)
    n_ok = sum(1 for v in cells.values() if v is not None)
    print(f"=== LOADED {n_ok}/{len(cells)} cells ===\n")
    if not cells:
        return
    warning = mixed_prefix_warning(
        prefix for prefix in map(slug_prefix_of, cells) if prefix is not None)
    if warning:
        print(warning + "\n")
    tier1_sanity(cells)
    tier2_per_cell(cells)
    tier3_action_space_comparison(cells)
    tier4_pareto_spread(cells)
    tier5_invariant(cells)


if __name__ == "__main__":
    main()
