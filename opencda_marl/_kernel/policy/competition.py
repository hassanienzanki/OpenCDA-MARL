from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from typing import Any, Dict, Iterable, Mapping, Sequence

# Lazy import: evaluation.momarl_metrics triggers evaluation.__init__ → training_metrics
# → competition (circular). Import inside the function instead.


def _summarize_intent_decisions(route_intent_decisions):
    from opencda_marl.evaluation.objective_metrics import summarize_intent_decisions
    return summarize_intent_decisions(route_intent_decisions)


def preference_group_from_values(preference: Sequence[float] | None) -> str:
    if not preference:
        return "unknown"
    values = [float(value) for value in preference]
    if len(values) < 2:
        return "unknown"
    return "group_a" if values[0] >= values[1] else "group_b"


def normalize_competition_record(record: Mapping[str, Any]) -> Dict[str, Any]:
    route_intent = record.get("route_intent")
    normalized = {
        "agent_id": str(record.get("agent_id", "")),
        "outcome": str(record.get("outcome", "unknown")),
        "completion_step": int(record.get("completion_step", -1)),
        "entry_direction": str(record.get("entry_direction", "unknown") or "unknown").lower(),
        "route_type": str(record.get("route_type", "unknown") or "unknown"),
        "lane_id": record.get("lane_id"),
        "preference": [float(value) for value in (record.get("preference") or [])],
        "preference_group": str(record.get("preference_group", "unknown") or "unknown"),
        "junction_id": (str(record["junction_id"])
                        if record.get("junction_id") is not None else None),
        "route_intent": deepcopy(dict(route_intent)) if isinstance(route_intent, Mapping) else None,
    }
    if normalized["preference_group"] == "unknown" and normalized["preference"]:
        normalized["preference_group"] = preference_group_from_values(normalized["preference"])
    return normalized


def build_competition_summary(
    records: Iterable[Mapping[str, Any]] | None,
    *,
    yielding_total: float = 0.0,
) -> Dict[str, Any]:
    normalized_records = [normalize_competition_record(record) for record in (records or [])]
    successful = sorted(
        [record for record in normalized_records if record["outcome"] == "success" and record["completion_step"] >= 0],
        key=lambda item: (item["completion_step"], item["agent_id"]),
    )

    crossing_order = [record["agent_id"] for record in successful]
    first_mover = successful[0]["entry_direction"] if successful else "none"
    first_mover_counts = Counter()
    if successful:
        first_mover_counts[first_mover] += 1

    rank_by_direction = defaultdict(list)
    rank_by_group = defaultdict(list)
    for rank, record in enumerate(successful, start=1):
        rank_by_direction[record["entry_direction"]].append(float(rank))
        rank_by_group[record["preference_group"]].append(float(rank))

    group_stats: Dict[str, Dict[str, Any]] = {}
    grouped_records = defaultdict(list)
    route_intent_decisions = []
    for record in normalized_records:
        grouped_records[record["preference_group"]].append(record)
        route_intent = record.get("route_intent")
        if isinstance(route_intent, Mapping):
            decision = deepcopy(dict(route_intent))
            decision["outcome"] = record["outcome"]
            route_intent_decisions.append(decision)

    for group_name, group_records in grouped_records.items():
        successes = sum(1 for record in group_records if record["outcome"] == "success")
        collisions = sum(1 for record in group_records if record["outcome"] == "collision")
        timeouts = sum(1 for record in group_records if record["outcome"] == "timeout")
        group_stats[group_name] = {
            "agent_count": len(group_records),
            "success_count": successes,
            "collision_count": collisions,
            "timeout_count": timeouts,
            "success_rate": (successes / len(group_records) * 100.0) if group_records else 0.0,
            "mean_completion_rank": (
                sum(rank_by_group[group_name]) / len(rank_by_group[group_name])
                if rank_by_group[group_name]
                else 0.0
            ),
        }

    return {
        "terminal_records": normalized_records,
        "crossing_order": crossing_order,
        "crossing_order_stats": {
            "successful_agents": len(successful),
            "first_mover": first_mover,
            "first_mover_by_entry_direction": dict(first_mover_counts),
            "avg_completion_rank_by_entry_direction": {
                direction: sum(ranks) / len(ranks)
                for direction, ranks in rank_by_direction.items()
                if ranks
            },
            "avg_completion_rank_by_preference_group": {
                group: sum(ranks) / len(ranks)
                for group, ranks in rank_by_group.items()
                if ranks
            },
        },
        "yielding_summary": {
            "yielding_total": float(yielding_total),
            "yielding_per_success": float(yielding_total) / len(successful) if successful else 0.0,
        },
        "competition_group_stats": deepcopy(group_stats),
        "crossing_order_by_junction": _crossing_order_by_junction(successful),
        "route_intent_summary": _summarize_intent_decisions(route_intent_decisions),
    }


def _crossing_order_by_junction(successful) -> Dict[str, Any]:
    """Crossing order per junction.

    A whole-map setting activates several junctions at once, and vehicles
    crossing *different* intersections never competed with each other — one
    global ordering would interleave them and read as contention that never
    happened. Runs with a single junction get one entry, so the existing
    global ordering keeps its meaning.
    """
    per_junction = defaultdict(list)
    for record in successful:
        per_junction[record.get("junction_id") or "unknown"].append(record)
    summary: Dict[str, Any] = {}
    for junction_id, records in per_junction.items():
        ranks = defaultdict(list)
        for rank, record in enumerate(records, start=1):
            ranks[record["entry_direction"]].append(float(rank))
        summary[str(junction_id)] = {
            "crossing_order": [record["agent_id"] for record in records],
            "successful_agents": len(records),
            "first_mover": records[0]["entry_direction"] if records else "none",
            "avg_completion_rank_by_entry_direction": {
                direction: sum(values) / len(values)
                for direction, values in ranks.items() if values
            },
        }
    return summary
