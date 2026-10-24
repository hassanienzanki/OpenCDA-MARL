"""Population mixes: which guarantees the assignment gives, and which mixes are rejected.

The shares are exact over every window of ``resolution`` consecutive draws
and reproducible per seed; a shorter prefix only approximates them. A mix
must name the learning population, and on CARLA, where every population's
agent is built through the agent registry, every name must be registered.
Both used to surface only mid-run, or never: a mix without the learner
silently trained on nothing, and an unregistered name failed at its first
spawn.
"""
from types import SimpleNamespace

import pytest
from loguru import logger

from opencda_marl._kernel.traffic.population import (
    PopulationSampler,
    validate_population_mix,
)


def _draws(sampler, n):
    return [sampler.assign() for _ in range(n)]


@pytest.mark.parametrize("mix,resolution,seed", [
    ({"marl": 0.5, "background": 0.5}, 100, 7),
    ({"marl": 0.25, "background": 0.75}, 100, 1),
    ({"marl": 0.3, "rule_based": 0.2, "behavior": 0.5}, 20, 3),
])
def test_every_window_of_resolution_draws_holds_the_quotas_exactly(mix, resolution, seed):
    sampler = PopulationSampler(mix, seed=seed, resolution=resolution)
    quotas = {key: round(share * resolution) for key, share in mix.items()}
    draws = _draws(sampler, 3 * resolution)
    for start in range(2 * resolution + 1):          # aligned and unaligned windows alike
        window = draws[start:start + resolution]
        assert {key: window.count(key) for key in mix} == quotas, start


def test_a_short_prefix_only_approximates_the_ratio():
    # The pattern is shuffled once, so a prefix is a sample of it, not a
    # scaled-down copy: the docstring promises windows, not prefixes.
    draws = _draws(PopulationSampler({"marl": 0.5, "background": 0.5}, seed=7), 100)
    deviations = [abs(draws[:n].count("marl") - n / 2) for n in range(1, 101)]
    assert max(deviations) > 1
    assert deviations[-1] == 0


def test_a_homogeneous_or_disabled_block_needs_no_validation():
    assert validate_population_mix(None, "marl") is None
    assert validate_population_mix({}, "marl") is None
    assert validate_population_mix({"mix": {"background": 1.0}, "enabled": False}, "marl") is None


def test_a_mix_without_the_learning_population_is_rejected():
    with pytest.raises(ValueError, match="names no learning population"):
        validate_population_mix({"mix": {"MARL": 0.5, "rule_based": 0.5}}, "marl")
    # The SUMO case: every other name is background there, so a mix of
    # background alone would give the learner no vehicle at all.
    with pytest.raises(ValueError, match="names no learning population"):
        validate_population_mix({"mix": {"background": 1.0}}, "marl")


def test_a_zero_learner_share_is_allowed_but_not_silent():
    messages = []
    sink = logger.add(lambda message: messages.append(str(message)), level="WARNING")
    try:
        sampler = validate_population_mix({"mix": {"marl": 0.0, "rule_based": 1.0}}, "marl")
    finally:
        logger.remove(sink)
    assert set(sampler.shares) == {"rule_based"}
    assert any("zero share" in message for message in messages), messages


def test_on_sumo_any_other_name_is_background_traffic():
    sampler = validate_population_mix({"mix": {"marl": 0.5, "background": 0.5}, "seed": 2}, "marl")
    assert sampler.shares == {"marl": 0.5, "background": 0.5}


def test_with_a_registry_every_name_must_be_registered():
    known = ("behavior", "vanilla", "rule_based", "marl")
    assert validate_population_mix({"mix": {"marl": 0.5, "rule_based": 0.5}}, "marl",
                                   known_agent_types=known) is not None
    with pytest.raises(ValueError, match=r"unregistered agent type\(s\) \['background'\]"):
        validate_population_mix({"mix": {"marl": 0.5, "background": 0.5}}, "marl",
                                 known_agent_types=known)
    # Names match exactly, as the spawn-time registry lookup does: a label
    # that differs only in case would pass a folded check and still fail at
    # its first spawn.
    with pytest.raises(ValueError, match="Rule_Based"):
        validate_population_mix({"mix": {"marl": 0.5, "Rule_Based": 0.5}}, "marl",
                                 known_agent_types=known)


def test_the_carla_agent_manager_rejects_an_unregistered_population_at_construction():
    pytest.importorskip("carla", reason="the CARLA agent manager needs the carla package")
    from opencda_marl._kernel.agent_manager import MARLAgentManager

    world = SimpleNamespace(get_map=lambda: None)
    config = {"agent_type": "marl", "population": {"mix": {"marl": 0.5, "rule_based": 0.5}}}
    assert MARLAgentManager(config, {}, world, cav_world=None).agent_type == "marl"
    config["population"] = {"mix": {"marl": 0.5, "not_a_registered_driver": 0.5}}
    with pytest.raises(ValueError, match="not_a_registered_driver"):
        MARLAgentManager(config, {}, world, cav_world=None)
