"""Tests for the intent provider's force_intent override, which pins the
feasibility mask to a single intent (diagnostics / what-if evaluation)."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

_TEST_DIR = str(Path(__file__).resolve().parent)
if _TEST_DIR not in sys.path:
    sys.path.insert(0, _TEST_DIR)

from opencda_marl._kernel.policy.intent_bridge import make_marl_intent_provider
from opencda_marl.api.route_intent import RouteIntent


class _StubAlgorithm:
    """Minimal stand-in for IntentMAPPOAlgorithm/IntentPCMAPPOAlgorithm.

    Records the mask it was given so tests can inspect it.
    """

    def __init__(self):
        self.last_mask: np.ndarray | None = None

    def select_intent(self, obs, mask, preference, training):
        self.last_mask = np.asarray(mask, dtype=bool).copy()
        legal = np.where(self.last_mask)[0]
        action_index = int(legal[0]) if legal.size else 3  # 3 == KEEP
        return action_index, 0.0, 0.0

    def store_pending(self, **kwargs):
        pass


def _stub_context_for_three_way():
    """Build a context dict the provider can consume against a 3-way junction."""
    from test_route_intent import _three_way_planner  # type: ignore[import-not-found]

    planner = _three_way_planner()
    event = SimpleNamespace(
        route_id=11,
        junction_id=100,
        metadata={
            "entry_direction": "north",
            "entry_lane_idx": 0,
            "exit_lane_idx": 0,
        },
    )
    adapter = SimpleNamespace(actor_id="vehicle_0")
    return {"event": event, "planner": planner, "adapter": adapter}


def test_force_intent_keep_collapses_mask_to_keep_only():
    algo = _StubAlgorithm()
    provider = make_marl_intent_provider(
        algorithm=algo,
        get_active_adapters=lambda: [],
        training=False,
        force_intent="keep-current",
    )
    decision = provider(_stub_context_for_three_way())
    assert decision is not None
    assert decision.selected_intent is RouteIntent.KEEP
    assert decision.used_keep_current is True
    # mask should have only KEEP legal (index 3 in LEFT/STRAIGHT/RIGHT/KEEP order)
    assert algo.last_mask is not None
    assert algo.last_mask.tolist() == [False, False, False, True]


def test_force_intent_none_preserves_natural_mask():
    algo = _StubAlgorithm()
    provider = make_marl_intent_provider(
        algorithm=algo,
        get_active_adapters=lambda: [],
        training=False,
        force_intent=None,
    )
    decision = provider(_stub_context_for_three_way())
    assert decision is not None
    # Direction-preserving mask: LEFT would change the exit, STRAIGHT has a
    # same-exit-direction alternative, RIGHT is unavailable, KEEP is legal.
    assert algo.last_mask.tolist() == [False, True, False, True]


def test_force_intent_unknown_value_raises():
    with pytest.raises(ValueError):
        make_marl_intent_provider(
            algorithm=_StubAlgorithm(),
            get_active_adapters=lambda: [],
            training=False,
            force_intent="not-a-real-intent",
        )


def test_force_intent_left_falls_back_to_keep_when_exit_direction_changes():
    algo = _StubAlgorithm()
    provider = make_marl_intent_provider(
        algorithm=algo,
        get_active_adapters=lambda: [],
        training=False,
        force_intent="left",
    )
    decision = provider(_stub_context_for_three_way())
    assert decision is not None
    assert algo.last_mask.tolist() == [False, False, False, True]
    assert decision.selected_intent is RouteIntent.KEEP
    assert decision.fallback_reason.startswith("forced_intent_illegal")
