"""Post-matrix analysis: aggregate per-cell metrics across the matrix cells
(condition x regime x seed) and print summary tables.

Assumes every worker's matrix metrics and checkpoints have been copied under
the local `artifacts/` tree (`metrics_history/matrix/*`,
`checkpoints/matrix/*`), so this single script sees every partition's data.

Outputs to stdout:
  Tier 1 - Mechanism sanity (training_step > 0 across all cells)
  Tier 2 - Per-cell outcome rates (last 25 episodes)
  Tier 3 - Action-space comparison: speed_only vs rule_path vs learn_path
  Tier 4 - Pareto spread per conflicting-regime cell (group_a vs group_b)
  Tier 5 - Behavioral diagnostic: direction_preserved_violations summary

Every row and Tier 3 group carries the cell's slug prefix, so the cells of
two runs under one root are never merged; ``--slug-prefix`` reads one run.

Usage:
    pixi run python scripts/experiments/analyze_matrix_results.py
    pixi run python scripts/experiments/analyze_matrix_results.py --slug-prefix matrix_
"""
from __future__ import annotations
import argparse
import pickle
import pathlib
import statistics
import re
import sys
from collections import defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from matrix_common import SLUG_PREFIX_PATTERN, mixed_prefix_warning  # noqa: E402

MATRIX_METRICS = pathlib.Path("artifacts/metrics_history/matrix")
MATRIX_CKPTS = pathlib.Path("artifacts/checkpoints/matrix")

CELL_RE = re.compile(
    SLUG_PREFIX_PATTERN
    + r"(?P<map>custom)_"
    r"(?P<action>speed_only|rule_path|learn_path|keep_current_path|random_legal_path)_"
    r"(?P<algo>masac|pcmasac|mappo|pcmappo)_"
    r"(?P<regime>aligned|conflicting)_"
    r"seed(?P<seed>\d+)"
)


def parse_cell_name(name: str):
    # fullmatch, not match: an evaluation slug is a training slug plus a
    # suffix (`..._seed0_eval200vph`), and a prefix match parsed it as the
    # training cell it was derived from -- so eval outcomes silently replaced
    # the training numbers in the aggregates keyed by (action, algo, regime).
    m = CELL_RE.fullmatch(name)
    if not m:
        return None
    return m.groupdict()


def short_cell_name(name: str) -> str:
    """The cell name without its map, for the per-cell tables.

    The prefix stays: it names the run, and two runs of one cell must not
    print as two identical rows.
    """
    m = CELL_RE.fullmatch(name)
    return m["prefix"] + name[m.start("action"):] if m else name


def last_n_stats(files, n=25):
    """Return mean (succ%, coll%, tmo%, reward) over last n episodes."""
    tail = files[-n:] if len(files) >= n else files
    s_rates, c_rates, t_rates, rewards, utils = [], [], [], [], []
    for fp in tail:
        with fp.open("rb") as fh:
            d = pickle.load(fh)
        s = d["success_counts"][0]
        c = d["collision_counts"][0]
        t = d["timeout_counts"][0]
        tot = s + c + t
        if tot:
            s_rates.append(100 * s / tot)
            c_rates.append(100 * c / tot)
            t_rates.append(100 * t / tot)
        rewards.append(d["rewards"][0])
        utils.append(d["utilities"][0])
    if not s_rates:
        return None
    return dict(
        n_eps=len(files),
        n_window=len(tail),
        succ_mean=statistics.mean(s_rates),
        coll_mean=statistics.mean(c_rates),
        tmo_mean=statistics.mean(t_rates),
        reward_mean=statistics.mean(rewards),
        utility_mean=statistics.mean(utils),
    )


def load_cells(slug_prefix: str | None = None):
    """Training cells under MATRIX_METRICS; with ``slug_prefix``, only the
    cells of that run (``None`` loads every prefix)."""
    cells = {}
    if not MATRIX_METRICS.is_dir():
        return cells
    for d in sorted(MATRIX_METRICS.iterdir()):
        if not d.is_dir():
            continue
        meta = parse_cell_name(d.name)
        if not meta:
            continue
        if slug_prefix is not None and meta["prefix"] != slug_prefix:
            continue
        files = sorted(d.glob("metrics_history_ep*.pkl"),
                       key=lambda p: int(p.stem.split("ep")[-1]))
        if not files:
            continue
        cells[d.name] = dict(meta=meta, files=files, stats=last_n_stats(files))
    return cells


def tier1_mechanism(cells):
    """Quick check: any cells with training_step=0 or no episodes?"""
    print("=" * 100)
    print("TIER 1 - Mechanism sanity (each cell should have training happen)")
    print("=" * 100)
    bad = []
    for name, info in cells.items():
        n_eps = len(info["files"])
        # Try to load the latest checkpoint and check training_step
        algo = info["meta"]["algo"]
        ck_dir = MATRIX_CKPTS / name / algo
        ck_path = ck_dir / "latest_checkpoint.pth"
        if not ck_path.exists():
            print(f"  {name}: {n_eps} eps, no checkpoint at {ck_path}")
            continue
        try:
            import torch
            ck = torch.load(ck_path, map_location="cpu", weights_only=False)
            ts = ck.get("training_step", ck.get("updates_performed", "?"))
            alpha = ck.get("alpha", "n/a")
            if hasattr(alpha, "item"):
                alpha = alpha.item()
            status = "OK" if (isinstance(ts, int) and ts > 0) else "BROKEN"
            if status == "BROKEN":
                bad.append(name)
            print(f"  [{status}] {name}: {n_eps} eps, training_step={ts}, alpha={alpha}")
        except Exception as e:
            print(f"  ERROR loading {ck_path}: {e}")
    if bad:
        print(f"\nWARNING: {len(bad)} cell(s) appear broken (training_step=0):")
        for b in bad:
            print(f"  - {b}")


def tier2_per_cell(cells):
    print("\n" + "=" * 100)
    print("TIER 2 - Per-cell outcome rates (last 25 episodes)")
    print("=" * 100)
    print(f"{'cell':75} {'eps':>4} {'succ%':>6} {'coll%':>6} {'tmo%':>6} {'reward':>9} {'util':>9}")
    print("-" * 120)
    for name, info in cells.items():
        s = info["stats"]
        if not s:
            continue
        short = short_cell_name(name)
        print(f"{short:75} {s['n_eps']:>4} {s['succ_mean']:>6.1f} {s['coll_mean']:>6.1f} "
              f"{s['tmo_mean']:>6.1f} {s['reward_mean']:>9.0f} {s['utility_mean']:>9.0f}")


def tier3_groups(cells):
    """(prefix, algo, regime) -> {action: stats} for the Tier 3 table.

    Keyed by the slug prefix too, so a cell of another run never becomes
    the speed_only baseline a path condition is compared against.
    """
    by_key = defaultdict(dict)
    for info in cells.values():
        m = info["meta"]
        s = info["stats"]
        if not s:
            continue
        by_key[(m["prefix"], m["algo"], m["regime"])][m["action"]] = s
    return by_key


def tier3_action_space_comparison(cells):
    """Action-space comparison across algorithms.

    speed_only vs rule_path vs learn_path -- per (run prefix, algo, regime),
    with each path condition's delta against speed_only of the same run.
    """
    print("\n" + "=" * 100)
    print("TIER 3 - Action-space comparison: speed_only vs rule_path vs learn_path")
    print("=" * 100)
    by_key = tier3_groups(cells)

    print(f"\n{'run prefix':20} {'algo':10} {'regime':12} {'action':18} "
          f"{'succ%':>6} {'coll%':>6} {'tmo%':>6}")
    print("-" * 86)
    for (prefix, algo, regime), action_stats in sorted(by_key.items()):
        for action in ["speed_only", "rule_path", "learn_path"]:
            if action in action_stats:
                s = action_stats[action]
                print(f"{prefix!r:20} {algo:10} {regime:12} {action:18} {s['succ_mean']:>6.1f} "
                      f"{s['coll_mean']:>6.1f} {s['tmo_mean']:>6.1f}")
        # Compute delta vs speed_only for this (prefix, algo, regime)
        if "speed_only" in action_stats:
            base = action_stats["speed_only"]
            for action in ["rule_path", "learn_path"]:
                if action in action_stats:
                    s = action_stats[action]
                    d_succ = s["succ_mean"] - base["succ_mean"]
                    d_coll = s["coll_mean"] - base["coll_mean"]
                    print(f"  -> delta {action} vs speed_only: succ {d_succ:+.1f}pp, coll {d_coll:+.1f}pp")
        print()


def tier4_pareto_spread(cells):
    """Per conflicting-regime cell, compare group_a vs group_b outcomes."""
    print("=" * 100)
    print("TIER 4 - Pareto spread (conflicting-regime cells only)")
    print("=" * 100)
    for name, info in cells.items():
        if info["meta"]["regime"] != "conflicting":
            continue
        # Aggregate per-group outcomes across all episodes
        group_outcomes = defaultdict(lambda: defaultdict(int))
        group_succ_steps = defaultdict(list)
        for fp in info["files"]:
            with fp.open("rb") as fh:
                d = pickle.load(fh)
            cs = d.get("competition_summaries", [{}])[0]
            if not isinstance(cs, dict):
                continue
            for rec in cs.get("terminal_records", []):
                g = rec.get("preference_group")
                o = rec.get("outcome")
                if g and o:
                    group_outcomes[g][o] += 1
                    if o == "success":
                        group_succ_steps[g].append(rec.get("completion_step", 0))
        short = short_cell_name(name)
        print(f"\n  {short}:")
        for g in sorted(group_outcomes.keys()):
            counts = group_outcomes[g]
            total = sum(counts.values())
            if total == 0:
                continue
            succ_pct = 100 * counts.get("success", 0) / total
            coll_pct = 100 * counts.get("collision", 0) / total
            tmo_pct = 100 * counts.get("timeout", 0) / total
            avg_step = (statistics.mean(group_succ_steps[g])
                        if group_succ_steps[g] else float("nan"))
            print(f"    {g}: n={total}, succ={succ_pct:.1f}%, coll={coll_pct:.1f}%, "
                  f"tmo={tmo_pct:.1f}%, avg_succ_step={avg_step:.0f}")
        # Verdict
        if "group_a" in group_outcomes and "group_b" in group_outcomes:
            a_succ = 100 * group_outcomes["group_a"].get("success", 0) / max(1, sum(group_outcomes["group_a"].values()))
            b_succ = 100 * group_outcomes["group_b"].get("success", 0) / max(1, sum(group_outcomes["group_b"].values()))
            d = a_succ - b_succ
            verdict = "OK SPREAD" if abs(d) > 2 else "FAIL no spread"
            print(f"    -> delta succ (a-b) = {d:+.1f}pp [{verdict}]")


def tier5_behavioral(cells):
    print("\n" + "=" * 100)
    print("TIER 5 - Behavioral diagnostic (direction_preservation invariant)")
    print("=" * 100)
    bad_invariant = []
    for name, info in cells.items():
        # Sample last episode
        with info["files"][-1].open("rb") as fh:
            d = pickle.load(fh)
        cs = d.get("competition_summaries", [{}])[0]
        ris = cs.get("route_intent_summary", {}) if isinstance(cs, dict) else {}
        viol = ris.get("direction_preserved_violations", 0)
        if viol > 0:
            bad_invariant.append((name, viol))
    if bad_invariant:
        print("  VIOLATIONS DETECTED (direction-preserving invariant broken):")
        for name, v in bad_invariant:
            print(f"    {name}: {v} violations")
    else:
        print(f"  All {len(cells)} cells: direction_preserved_violations = 0 (invariant holds)")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--slug-prefix", default=None,
                   help="Analyze only the cells of the run launched with this "
                        "slug prefix (default: every prefix found).")
    args = p.parse_args(argv)
    cells = load_cells(args.slug_prefix)
    print(f"=== LOADED {len(cells)} CELLS ===\n")
    if not cells:
        print(f"No matrix cells found under {MATRIX_METRICS}.")
        return
    warning = mixed_prefix_warning(info["meta"]["prefix"] for info in cells.values())
    if warning:
        print(warning + "\n")
    tier1_mechanism(cells)
    tier2_per_cell(cells)
    tier3_action_space_comparison(cells)
    tier4_pareto_spread(cells)
    tier5_behavioral(cells)


if __name__ == "__main__":
    main()
