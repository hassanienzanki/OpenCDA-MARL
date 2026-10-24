"""The episode summary describes decisions it does not define.

A path provider owns its own vocabulary, so its decision metadata may use a
mask indexed by action rather than keyed by intent name. The summary must
read both and must never end an episode over a shape it does not own: it
runs at the episode reset, so a shape it cannot read would stop a run only
after its first full episode had already been simulated.
"""
import pytest

from opencda_marl.evaluation.objective_metrics import summarize_intent_decisions


def _builtin(intent="straight", legal=("straight", "keep-current")):
    return {
        "selected_intent": intent,
        "mask": {"left": "left" in legal, "straight": "straight" in legal,
                 "right": "right" in legal, "keep-current": "keep-current" in legal},
        "selected_route_type": intent if intent != "keep-current" else "straight",
        "selected_route_id": 12, "original_route_id": 11, "outcome": "success",
    }


def _plugin(slot=2, mask=(False, True, True, False)):
    """What an external provider writes: a sequence indexed by action, and a
    vocabulary the framework has never heard of."""
    return {
        "interface": "custom", "selected_slot": slot,
        "mask": list(mask), "selected_route_id": 13, "original_route_id": 12,
        "route_changed": True, "outcome": "success",
    }


def test_a_sequence_mask_is_read_by_action_index():
    summary = summarize_intent_decisions([_plugin(slot=2)])
    assert summary["invalid_action_attempts"] == 0
    # Two of four slots legal.
    assert summary["mean_mask_sparsity"] == pytest.approx(0.5)
    assert summary["mask_legal_fraction"] == pytest.approx(0.5)


def test_a_selection_outside_its_own_mask_is_still_counted_invalid():
    assert summarize_intent_decisions([_plugin(slot=0)])["invalid_action_attempts"] == 1
    # An index the provider did not record is not evidence of an invalid action.
    entry = _plugin(); entry.pop("selected_slot")
    assert summarize_intent_decisions([entry])["invalid_action_attempts"] == 0


def test_the_builtin_mapping_mask_is_unchanged():
    summary = summarize_intent_decisions([_builtin()])
    assert summary["invalid_action_attempts"] == 0
    assert summary["mean_mask_sparsity"] == pytest.approx(0.5)
    # An intent outside its own mask is invalid, as before.
    assert summarize_intent_decisions(
        [_builtin(intent="left")])["invalid_action_attempts"] == 1
    # keep-current alone is still recognised as a collapsed mask.
    assert summarize_intent_decisions(
        [_builtin(intent="keep-current", legal=("keep-current",))])["keep_only_mask_count"] == 1


def test_a_missing_or_odd_mask_never_raises():
    for entry in ({"selected_intent": "straight"},
                  {"selected_intent": "straight", "mask": None},
                  {"selected_intent": "straight", "mask": "nonsense"},
                  {"selected_slot": 1, "mask": []}):
        assert summarize_intent_decisions([entry])["invalid_action_attempts"] == 0


def test_both_vocabularies_summarise_together():
    """One episode may mix a built-in provider and a plugin one; the summary
    aggregates across both without either shape distorting the other."""
    summary = summarize_intent_decisions([_builtin(), _plugin(), _plugin(slot=1)])
    # All three fixtures name a route other than their original, whichever
    # vocabulary they used, so all three count as a route change.
    assert summary["route_change_count"] == 3
    assert summary["invalid_action_attempts"] == 0
    # Three decisions, each with two of its own slots legal.
    assert summary["mean_mask_sparsity"] == pytest.approx(0.5)
