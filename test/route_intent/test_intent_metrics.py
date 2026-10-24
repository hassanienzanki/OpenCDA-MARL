"""Route-intent metrics surface for evaluation aggregation."""
from opencda_marl.evaluation.objective_metrics import summarize_intent_decisions


_OUTCOMES = ("success", "collision", "planner_failure", "other")
_RTYPES = ("left", "straight", "right", "other")


def _empty_outcome_cell() -> dict:
    cell = {o: 0 for o in _OUTCOMES}
    cell["total"] = 0
    return cell


def _zero_cross_tab() -> dict:
    intents = ("left", "straight", "right", "keep-current")
    return {intent: {rt: _empty_outcome_cell() for rt in _RTYPES} for intent in intents}


def test_summarize_distribution_and_keep_rate():
    decisions = [
        {"selected_intent": "left",
         "mask": {"left": True, "straight": True, "right": False, "keep-current": True},
         "used_keep_current": False,
         "candidate_count_total": 4,
         "candidate_count_same_exit_direction": 3,
         "direction_preserved": True,
         "mask_sparsity": 0.25},
        {"selected_intent": "keep-current",
         "mask": {"left": False, "straight": False, "right": False, "keep-current": True},
         "used_keep_current": True,
         "candidate_count_total": 3,
         "candidate_count_same_exit_direction": 1,
         "direction_preserved": True,
         "fallback_reason": "no_same_exit_direction_alternatives",
         "mask_sparsity": 0.75},
        {"selected_intent": "straight",
         "mask": {"left": True, "straight": True, "right": True, "keep-current": True},
         "used_keep_current": False,
         "candidate_count_total": 5,
         "candidate_count_same_exit_direction": 4,
         "direction_preserved": False,
         "mask_sparsity": 0.0},
    ]
    summary = summarize_intent_decisions(decisions)
    assert summary["intent_distribution"]["left"] == 1
    assert summary["intent_distribution"]["straight"] == 1
    assert summary["intent_distribution"]["keep-current"] == 1
    assert summary["keep_current_rate"] == 1 / 3
    assert summary["invalid_action_attempts"] == 0
    # mask_sparsity = mean(legal_count / 4)
    assert summary["mask_sparsity"] == ((3 + 1 + 4) / 3) / 4
    assert summary["mean_mask_sparsity"] == (0.25 + 0.75 + 0.0) / 3
    assert summary["keep_only_mask_count"] == 1
    assert summary["direction_preserved_violations"] == 1
    assert summary["same_destination_violations"] == 1
    assert summary["fallback_count"] == 1
    assert summary["mean_candidate_count_total"] == 4.0
    assert summary["mean_candidate_count_same_exit_direction"] == 8 / 3
    assert summary["mean_candidate_count_same_destination"] == 8 / 3


def test_route_type_outcome_cross_tab_counts_intent_x_route_type_x_outcome():
    decisions = [
        # Agent picked LEFT, route resolved to left, ended successfully.
        {"selected_intent": "left", "selected_route_type": "left",
         "outcome": "success",
         "mask": {"left": True, "straight": True, "right": True, "keep-current": True},
         "used_keep_current": False},
        # Agent picked LEFT again, this time crashed.
        {"selected_intent": "left", "selected_route_type": "left",
         "outcome": "collision",
         "mask": {"left": True, "straight": True, "right": True, "keep-current": True},
         "used_keep_current": False},
        # Agent picked KEEP, original route was straight, succeeded.
        {"selected_intent": "keep-current", "selected_route_type": "straight",
         "outcome": "success",
         "mask": {"left": False, "straight": False, "right": False, "keep-current": True},
         "used_keep_current": True},
        # Agent picked KEEP, original route was a U-turn (normalized to "other"),
        # planner gave up mid-route.
        {"selected_intent": "keep-current", "selected_route_type": "other",
         "outcome": "planner_failure",
         "mask": {"left": False, "straight": False, "right": False, "keep-current": True},
         "used_keep_current": True},
        # Agent picked STRAIGHT, no outcome attached (e.g., still in flight) -> "other".
        {"selected_intent": "straight", "selected_route_type": "straight",
         "mask": {"left": True, "straight": True, "right": True, "keep-current": True},
         "used_keep_current": False},
    ]
    summary = summarize_intent_decisions(decisions)
    cross = summary["route_type_outcome"]

    # Schema: every (intent, route_type) cell exists with the full outcome dict.
    for intent in ("left", "straight", "right", "keep-current"):
        for rt in _RTYPES:
            cell = cross[intent][rt]
            for o in _OUTCOMES:
                assert o in cell, f"missing outcome {o} for {intent}/{rt}"
            assert "total" in cell

    # LEFT -> left: 1 success, 1 collision, total 2.
    assert cross["left"]["left"]["success"] == 1
    assert cross["left"]["left"]["collision"] == 1
    assert cross["left"]["left"]["total"] == 2

    # KEEP -> straight: 1 success.
    assert cross["keep-current"]["straight"]["success"] == 1
    assert cross["keep-current"]["straight"]["total"] == 1

    # KEEP -> other (U-turn folded): 1 planner_failure.
    assert cross["keep-current"]["other"]["planner_failure"] == 1
    assert cross["keep-current"]["other"]["total"] == 1

    # STRAIGHT -> straight with no outcome attached -> "other" bucket.
    assert cross["straight"]["straight"]["other"] == 1
    assert cross["straight"]["straight"]["total"] == 1

    # Cells with no observations remain zeroed.
    assert cross["right"]["left"] == _empty_outcome_cell()
    assert cross["left"]["straight"] == _empty_outcome_cell()


def test_route_type_outcome_handles_missing_route_type_as_other():
    # Decision entries from older callers might omit selected_route_type entirely.
    decisions = [
        {"selected_intent": "right", "outcome": "success",
         "mask": {"left": True, "straight": True, "right": True, "keep-current": True},
         "used_keep_current": False},
    ]
    summary = summarize_intent_decisions(decisions)
    assert summary["route_type_outcome"]["right"]["other"]["success"] == 1
    assert summary["route_type_outcome"]["right"]["other"]["total"] == 1


def test_route_type_outcome_buckets_unknown_outcome_string_into_other():
    decisions = [
        {"selected_intent": "left", "selected_route_type": "left",
         "outcome": "definitely-not-a-canonical-bucket",
         "mask": {"left": True, "straight": True, "right": True, "keep-current": True},
         "used_keep_current": False},
    ]
    summary = summarize_intent_decisions(decisions)
    assert summary["route_type_outcome"]["left"]["left"]["other"] == 1
    # Sanity: the original 4 metrics still produced.
    assert summary["intent_distribution"]["left"] == 1
    assert summary["keep_current_rate"] == 0.0
