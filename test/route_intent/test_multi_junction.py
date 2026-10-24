"""Whole-map settings: several junctions active at once.

Vehicles crossing different intersections never competed with each other,
so their crossing order must not be interleaved into one ranking.
"""
from opencda_marl._kernel.policy.competition import (
    build_competition_summary,
    normalize_competition_record,
)


def _record(agent, step, junction, direction="north", outcome="success"):
    return {"agent_id": agent, "outcome": outcome, "completion_step": step,
            "entry_direction": direction, "junction_id": junction,
            "preference": [0.5, 0.5]}


def test_records_keep_their_junction():
    normalized = normalize_competition_record(_record("a", 10, 4))
    assert normalized["junction_id"] == "4"
    # A record without one stays None rather than inventing a junction.
    assert normalize_competition_record({"agent_id": "b"})["junction_id"] is None


def test_crossing_order_is_reported_per_junction():
    records = [
        _record("j4_first", 10, 4, "north"),
        _record("j9_first", 20, 9, "south"),
        _record("j4_second", 30, 4, "east"),
        _record("j9_second", 40, 9, "west"),
    ]
    summary = build_competition_summary(records)
    per_junction = summary["crossing_order_by_junction"]

    assert set(per_junction) == {"4", "9"}
    assert per_junction["4"]["crossing_order"] == ["j4_first", "j4_second"]
    assert per_junction["9"]["crossing_order"] == ["j9_first", "j9_second"]
    # Each junction has its own first mover; the global order would have
    # claimed a single one for the whole map.
    assert per_junction["4"]["first_mover"] == "north"
    assert per_junction["9"]["first_mover"] == "south"
    # Ranks restart per junction: j9_first is second globally but first at 9.
    assert per_junction["9"]["avg_completion_rank_by_entry_direction"]["south"] == 1.0


def test_a_single_junction_run_is_unchanged():
    records = [_record("a", 10, 4, "north"), _record("b", 20, 4, "east")]
    summary = build_competition_summary(records)
    assert summary["crossing_order"] == ["a", "b"]
    assert list(summary["crossing_order_by_junction"]) == ["4"]
    assert summary["crossing_order_by_junction"]["4"]["crossing_order"] == ["a", "b"]


def test_records_without_a_junction_group_under_unknown():
    summary = build_competition_summary([
        {"agent_id": "a", "outcome": "success", "completion_step": 5,
         "entry_direction": "north", "preference": [0.5, 0.5]},
    ])
    assert list(summary["crossing_order_by_junction"]) == ["unknown"]
