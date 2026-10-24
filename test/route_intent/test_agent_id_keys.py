"""Agent-id key normalization at the extractor boundary."""
from __future__ import annotations
import pytest


def test_extractor_emits_only_str_keys():
    """ObservationExtractor.extract returns a dict whose keys are all str."""
    from opencda_marl.builtin.extractors import ObservationExtractor
    extractor = ObservationExtractor(
        config={"q_learning": {}}, algorithm="q_learning"
    )
    raw = {
        42: {"speed": 30.0, "position_x": 0.0, "position_y": 0.0,
             "heading_angle": 0.0, "dist_to_intersection": 0.0,
             "dist_to_front_vehicle": 100.0},
        "13": {"speed": 30.0, "position_x": 0.0, "position_y": 0.0,
               "heading_angle": 0.0, "dist_to_intersection": 0.0,
               "dist_to_front_vehicle": 100.0},
    }
    extracted = extractor.extract(raw)
    for key in extracted.keys():
        assert isinstance(key, str), f"extractor leaked non-str key {key!r}"


def test_manager_has_no_int_coercion_blocks():
    """The policy manager never coerces agent ids with int(agent_id).

    Keys are normalized to str at the extractor boundary, so the manager
    has no int/str fallback. A source-level scan: any non-comment line
    containing `int(agent_id)` is forbidden.
    """
    import inspect
    from opencda_marl._kernel.policy import manager as mgr_mod
    src = inspect.getsource(mgr_mod)
    suspicious = []
    for line in src.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if "int(agent_id)" in stripped:
            suspicious.append(stripped)
    assert not suspicious, f"manager still coerces agent_id: {suspicious}"


def test_strategy_has_no_int_alias_helpers():
    """algorithm_strategy.py no longer needs _has_int_alias or _get_reward."""
    import inspect
    from opencda_marl._kernel.policy import strategy as strat_mod
    src = inspect.getsource(strat_mod)
    for name in ("_has_int_alias", "_get_reward"):
        assert f"def {name}" not in src, (
            f"{name} must stay deleted: keys are normalized at the extractor"
        )
