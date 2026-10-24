"""SUMO selects its reward calculator from the MARL block, as CARLA does.

``MARL.components.reward_calculator`` is where the docs put the selection and
where CARLA's MARLEnv reads it. SumoMARLEnv used to read the full run config,
which looks for a top-level ``components`` block, so a custom calculator ran
on CARLA while SUMO silently used the default one.
"""
import pytest


@pytest.fixture
def sumo_env_cls():
    try:
        from opencda_marl._kernel.envs.sumo_marl_env import SumoMARLEnv
    except (ImportError, SystemExit) as exc:
        pytest.skip(f"SUMO runtime unavailable: {exc}")
    return SumoMARLEnv


@pytest.fixture
def calculator_key():
    from opencda_marl.api import components
    from opencda_marl.plugins import registry

    key = "test_sumo_marl_block_calculator"

    class _Calculator:
        def __init__(self, reward_params, reward_spec):
            self.reward_params, self.reward_spec = reward_params, reward_spec

    with registry.thawed():
        components.register_component("reward_calculator", key)(_Calculator)
    yield key, _Calculator
    components._kind_remove("reward_calculator", key)


def _env(cls, marl_config, full_config):
    env = object.__new__(cls)
    env.config = full_config
    env.marl_config = marl_config
    env.reward_params = {"p": 1}
    env.reward_spec = "spec"
    return env


def test_the_calculator_named_in_the_marl_block_is_the_one_built(sumo_env_cls, calculator_key):
    key, calculator_cls = calculator_key
    marl = {"algorithm": "pcmappo", "components": {"reward_calculator": key}}
    env = _env(sumo_env_cls, marl, {"MARL": marl})

    calculator = env._build_reward_calculator()

    assert isinstance(calculator, calculator_cls)
    assert (calculator.reward_params, calculator.reward_spec) == ({"p": 1}, "spec")


def test_a_top_level_components_block_does_not_select_it(sumo_env_cls, calculator_key):
    key, calculator_cls = calculator_key
    from opencda_marl.builtin.rewards.calculator import RewardCalculator

    marl = {"algorithm": "pcmappo"}
    env = _env(sumo_env_cls, marl, {"MARL": marl, "components": {"reward_calculator": key}})

    calculator = env._build_reward_calculator()

    assert type(calculator) is RewardCalculator
    assert not isinstance(calculator, calculator_cls)
