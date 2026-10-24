"""Aggregation helpers: median/IQR, paired Wilcoxon, bootstrap hypervolume CI.

These helpers aggregate per-seed metric archives into median/IQR summary
tables, plus the statistical companions (paired Wilcoxon p-values,
bootstrap HV CIs).
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from scipy.stats import wilcoxon

from opencda_marl.evaluation.hypervolume import hypervolume_2d


def median_iqr(values: Iterable[float]) -> Dict[str, float]:
    """Summarize a sequence as median, Q1, Q3, min, max."""
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return {
            "median": float("nan"),
            "q1": float("nan"),
            "q3": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    return {
        "median": float(np.median(arr)),
        "q1": float(np.percentile(arr, 25)),
        "q3": float(np.percentile(arr, 75)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def paired_wilcoxon(baseline: Iterable[float], treatment: Iterable[float]) -> float:
    """Return the two-sided Wilcoxon signed-rank p-value comparing paired arrays.

    With N < 3, the test is not meaningful; returns NaN.
    If all paired differences are zero, the test cannot run; returns 1.0.
    """
    a = np.asarray(list(baseline), dtype=float)
    b = np.asarray(list(treatment), dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    if a.size < 3:
        return float("nan")
    if np.allclose(a, b):
        return 1.0
    result = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
    return float(result.pvalue)


def bootstrap_hv_ci(
    baseline_pts: np.ndarray,
    treatment_pts: np.ndarray,
    reference: Tuple[float, float],
    n_bootstrap: int = 1000,
    rng: np.random.Generator | None = None,
) -> Tuple[float, float]:
    """Bootstrap 95% CI for HV(treatment) - HV(baseline)."""
    rng = rng if rng is not None else np.random.default_rng(0)
    baseline_pts = np.asarray(baseline_pts, dtype=float)
    treatment_pts = np.asarray(treatment_pts, dtype=float)
    deltas = np.empty(n_bootstrap, dtype=float)
    for k in range(n_bootstrap):
        b_idx = rng.integers(0, len(baseline_pts), size=len(baseline_pts))
        t_idx = rng.integers(0, len(treatment_pts), size=len(treatment_pts))
        try:
            hv_b = hypervolume_2d(baseline_pts[b_idx], reference)
        except ValueError:
            hv_b = 0.0
        try:
            hv_t = hypervolume_2d(treatment_pts[t_idx], reference)
        except ValueError:
            hv_t = 0.0
        deltas[k] = hv_t - hv_b
    return float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))


def _records_for_archive(archive: Path) -> List[Dict]:
    """Resolve per-episode metric records from one archive directory.

    Three layouts are supported:
      (a) Flat directory of ``<file>.json`` per episode.
      (b) Checkpoint directory containing ``training_metadata.json`` with a
          ``history`` array of ``{episode, metrics}`` entries (the
          OpenCDA-MARL platform's actual output).
      (c) Metrics archive directory containing ``metrics_history_ep*.pkl``
          exports from ``TrainingMetrics`` (evaluation mode output).
    """
    # (a) flat directory
    flat = []
    for record_file in sorted(archive.glob("*.json")):
        try:
            with record_file.open() as f:
                flat.append(json.load(f))
        except (json.JSONDecodeError, OSError):
            continue
    if flat and "metrics" not in flat[0]:
        return flat

    # (b) training_metadata.json with history array
    metadata_paths = list(archive.rglob("training_metadata.json"))
    if metadata_paths:
        try:
            with metadata_paths[0].open() as f:
                meta = json.load(f)
        except (json.JSONDecodeError, OSError):
            return []
        return [
            entry.get("metrics", {})
            for entry in meta.get("history", [])
            if isinstance(entry, dict) and isinstance(entry.get("metrics"), dict)
        ]
    pkl_records = []
    for pkl_path in sorted(archive.glob("metrics_history_ep*.pkl")):
        try:
            with pkl_path.open("rb") as f:
                history = pickle.load(f)
        except (OSError, pickle.PickleError, EOFError):
            continue
        pkl_records.extend(_records_from_metrics_history(history))
    if pkl_records:
        return pkl_records
    return flat


def _records_from_metrics_history(history: Dict) -> List[Dict]:
    """Reconstruct per-episode records from TrainingMetrics pickle exports."""
    if not isinstance(history, dict):
        return []
    rewards = list(history.get("rewards") or [])
    count = len(rewards)
    records: List[Dict] = []
    for idx in range(count):
        success = _list_get(history, "success_counts", idx, 0)
        collision = _list_get(history, "collision_counts", idx, 0)
        physics = _list_get(history, "physics_anomaly_event_counts", idx, 0)
        timeout = _list_get(history, "timeout_counts", idx, 0)
        total = float(success + collision + physics + timeout)
        episode_length = int(_list_get(history, "episode_lengths", idx, 0) or 0)
        throughput = (
            float(success) / float(episode_length) / 0.05 * 3600.0
            if episode_length > 0
            else 0.0
        )
        objective_totals = _list_get(history, "objective_component_totals", idx, {}) or {}
        objective_spec = _list_get(history, "objective_specs", idx, None)
        competition_summary = _list_get(history, "competition_summaries", idx, {}) or {}
        record = {
            "total_reward": float(rewards[idx]),
            "total_utility": float(_list_get(history, "utilities", idx, rewards[idx])),
            "objective_component_totals": objective_totals,
            "objective_vector_dim": len(objective_totals),
            "objective_spec": objective_spec,
            "success_rate": float(success) / total * 100.0 if total > 0 else 0.0,
            "collision_rate": float(collision) / total * 100.0 if total > 0 else 0.0,
            "physics_anomaly_rate": float(physics) / total * 100.0 if total > 0 else 0.0,
            "timeout_rate": float(timeout) / total * 100.0 if total > 0 else 0.0,
            "successes": int(success),
            "collisions": int(collision),
            "physics_anomalies": int(physics),
            "timeouts": int(timeout),
            "near_miss_count": int(_list_get(history, "near_miss_counts", idx, 0) or 0),
            "episode_length": episode_length,
            "throughput": throughput,
            "competition_summary": competition_summary,
            "avg_speed": float(_list_get(history, "actual_speed_means", idx, 0.0) or 0.0),
            "speed_variance": float(_list_get(history, "actual_speed_variances", idx, 0.0) or 0.0),
            "target_speed_mean": float(_list_get(history, "target_speed_means", idx, 0.0) or 0.0),
        }
        route_intent = (competition_summary.get("route_intent_summary") or {})
        if route_intent:
            distribution = route_intent.get("intent_distribution") or {}
            direction_violations = float(
                route_intent.get(
                    "direction_preserved_violations",
                    route_intent.get(
                        "same_exit_direction_violations",
                        route_intent.get("same_destination_violations", 0.0),
                    ),
                )
            )
            same_exit_candidates = float(
                route_intent.get(
                    "mean_candidate_count_same_exit_direction",
                    route_intent.get("mean_candidate_count_same_destination", 0.0),
                )
            )
            record.update({
                "route_intent_decision_count": float(sum(int(v) for v in distribution.values())),
                "route_intent_keep_current_rate": float(route_intent.get("keep_current_rate", 0.0)),
                "route_intent_mask_legal_fraction": float(route_intent.get("mask_legal_fraction", 0.0)),
                "route_intent_mean_mask_sparsity": float(route_intent.get("mean_mask_sparsity", 0.0)),
                "route_intent_direction_preserved_violations": direction_violations,
                "route_intent_same_exit_direction_violations": direction_violations,
                "route_intent_same_destination_violations": direction_violations,
                "route_intent_fallback_count": float(route_intent.get("fallback_count", 0.0)),
                "route_intent_route_change_count": float(route_intent.get("route_change_count", 0.0)),
                "route_intent_route_change_rate": float(route_intent.get("route_change_rate", 0.0)),
                "route_intent_queued_lane_avoidance_count": float(route_intent.get("queued_lane_avoidance_count", 0.0)),
                "route_intent_queued_lane_avoidance_opportunities": float(route_intent.get("queued_lane_avoidance_opportunities", 0.0)),
                "route_intent_queued_lane_avoidance_rate": float(route_intent.get("queued_lane_avoidance_rate", 0.0)),
                "route_intent_mean_selected_queued_same_exit_lane_count": float(route_intent.get("mean_selected_queued_same_exit_lane_count", 0.0)),
                "route_intent_mean_original_queued_same_exit_lane_count": float(route_intent.get("mean_original_queued_same_exit_lane_count", 0.0)),
                "route_intent_mean_candidate_count_total": float(route_intent.get("mean_candidate_count_total", 0.0)),
                "route_intent_mean_candidate_count_same_exit_direction": same_exit_candidates,
                "route_intent_mean_candidate_count_same_destination": same_exit_candidates,
            })
        records.append(record)
    return records


def _list_get(history: Dict, key: str, idx: int, default):
    values = history.get(key)
    if not isinstance(values, list) or idx >= len(values):
        return default
    return values[idx]


def _safety_efficiency_points(records: Iterable[Dict]) -> List[Tuple[float, float]]:
    """Extract per-episode (safety, efficiency) pairs from a record stream.

    Safety = 1 - collision_rate / 100  (rate is reported as 0..100).
    Efficiency = success_rate / 100, with a small throughput tie-breaker so
    episodes with equal success rates rank by throughput.
    Both dimensions are in [0, 1]; the Pareto reference is (0, 0).
    """
    pts: List[Tuple[float, float]] = []
    for rec in records:
        cr = rec.get("collision_rate")
        sr = rec.get("success_rate")
        if cr is None or sr is None:
            continue
        safety = max(0.0, 1.0 - float(cr) / 100.0)
        efficiency = max(0.0, float(sr) / 100.0)
        pts.append((safety, efficiency))
    return pts


def summarize_cell(metric_archives: Iterable[str | Path]) -> Dict:
    """Aggregate N seed archives of one experiment configuration.

    Returns a dict keyed by metric name with median_iqr() output, plus two
    Pareto-related fields:
      - ``pareto_points``: list of (safety, efficiency) tuples across all
        per-episode records (for a safety/efficiency scatter plot).
      - ``pareto_hypervolume``: median/IQR of the per-seed hypervolume
        (or a single value if only one seed is present).
    """
    per_metric: Dict[str, List[float]] = {}
    all_points: List[Tuple[float, float]] = []
    per_seed_hv: List[float] = []
    for archive_path in metric_archives:
        archive = Path(archive_path)
        if not archive.exists():
            continue
        records = list(_records_for_archive(archive))
        for rec in records:
            for k, v in rec.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    per_metric.setdefault(k, []).append(float(v))
        seed_pts = _safety_efficiency_points(records)
        all_points.extend(seed_pts)
        if seed_pts:
            try:
                hv = hypervolume_2d(np.asarray(seed_pts), (0.0, 0.0))
            except ValueError:
                hv = 0.0
            per_seed_hv.append(hv)

    out: Dict = {k: median_iqr(v) for k, v in per_metric.items()}
    # Only emit Pareto fields when we have at least one record; an empty
    # archive list should yield an empty summary (callers expect this).
    if all_points or per_seed_hv:
        out["pareto_points"] = [list(p) for p in all_points]
        out["pareto_hypervolume"] = (median_iqr(per_seed_hv) if per_seed_hv else {
            "median": float("nan"), "q1": float("nan"), "q3": float("nan"),
            "min": float("nan"), "max": float("nan"),
        })
    return out


def _cli():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archives",
        nargs="+",
        required=True,
        help="One per seed; each path is a directory of per-episode JSON records.",
    )
    parser.add_argument(
        "--out",
        default="-",
        help="Where to write summary JSON; '-' for stdout.",
    )
    args = parser.parse_args()
    summary = summarize_cell(args.archives)
    out = json.dumps(summary, indent=2)
    if args.out == "-":
        print(out)
    else:
        with open(args.out, "w") as f:
            f.write(out)


if __name__ == "__main__":
    _cli()
