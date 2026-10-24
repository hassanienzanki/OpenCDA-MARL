"""The competition metrics follow the objective, not a built-in profile name.

Both environments enabled the per-agent competition metadata and the
terminal records -- the source of the route-intent and preference
summaries -- only when the reward or objective profile carried one of the
two built-in names. An objective profile registered through the plugin
door silently switched them off: the learner kept deciding, the episode
summaries reported zero decisions, zero route changes and a mask sparsity
of 0.0. Now any run with an objective spec records them, and so does a
reward profile whose name says it is competitive.
"""
import re
from pathlib import Path
from types import SimpleNamespace

from opencda_marl.api.objectives import (
    competition_metrics_enabled,
    register_objective_profile,
    resolve_objective_configuration,
)
from opencda_marl.api.rewards import resolve_reward_configuration

REPO = Path(__file__).resolve().parents[2]


def _profiles(reward=None, objective=None):
    reward_config = resolve_reward_configuration({"reward_profile": reward} if reward else {})
    objective_config = resolve_objective_configuration({"objective_mode": objective} if objective else {})
    return reward_config, objective_config


def test_the_built_in_competitive_profiles_still_enable_the_metrics():
    assert competition_metrics_enabled(*_profiles("competitive_intersection_v1", None))
    assert competition_metrics_enabled(*_profiles(None, "competitive_ordering_v1"))
    assert competition_metrics_enabled(*_profiles("competitive_intersection_v2", "competitive_ordering_v1"))


def test_a_registered_objective_profile_enables_them_too():
    register_objective_profile("plugin_ordering_for_the_gate_test", {
        "objective_spec": {
            "objectives": ["efficiency", "interaction_risk"],
            "groups": {"efficiency": {"progress": 1.0}, "interaction_risk": {"collision": 1.0, "ttc": 1.0}},
        },
    })
    assert competition_metrics_enabled(*_profiles("competitive_intersection_v2", "plugin_ordering_for_the_gate_test"))
    assert competition_metrics_enabled(*_profiles(None, "plugin_ordering_for_the_gate_test")), "the objective alone is enough"


def test_a_plain_scalar_run_records_nothing():
    assert not competition_metrics_enabled(*_profiles("dense_intersection_v1", None)) or \
        str(_profiles("dense_intersection_v1", None)[0].profile_name).startswith("competitive")
    assert not competition_metrics_enabled(SimpleNamespace(profile_name="custom"), SimpleNamespace(profile_name="disabled", objective_spec=None))
    assert not competition_metrics_enabled(None, None)


def test_no_environment_compares_profile_names_any_more():
    for rel in ("opencda_marl/_kernel/envs/sumo_marl_env.py", "opencda_marl/_kernel/envs/marl_env.py"):
        source = (REPO / rel).read_text(encoding="utf-8")
        assert not re.search(r'profile_name.*==\s*"competitive_', source), rel
        assert "competition_metrics_enabled(" in source, rel
