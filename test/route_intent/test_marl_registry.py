"""Strict canonical algorithm registry tests."""

import pytest
from omegaconf import OmegaConf

from opencda_marl.api.algorithms import (
    algorithm_label,
    registered_algorithm_keys,
    resolve_algorithm_spec,
)
from scripts.experiments.matrix_common import validate_matrix_schema


def test_canonical_registry_labels_are_display_names():
    expected = {
        # The canonical grid.
        "matd3": "MATD3",
        "masac": "MASAC",
        "mappo": "MAPPO",
        "pcmatd3": "PCMATD3",
        "pcmasac": "PCMASAC",
        "pcmappo": "PCMAPPO",
        # Legacy single-agent demo families, kept registered so the tutorial
        # scenarios run (dispatched via multi_agent=False capabilities).
        "dqn": "DQN",
        "q_learning": "Q-Learning",
    }
    assert set(registered_algorithm_keys()) >= set(expected)
    for key, label in expected.items():
        spec = resolve_algorithm_spec(key)
        assert spec.key == key
        assert spec.label == label
        assert algorithm_label(key) == label


def test_single_agent_demo_families_declare_single_agent_dispatch():
    for key in ("dqn", "q_learning"):
        assert resolve_algorithm_spec(key).capabilities.multi_agent is False


@pytest.mark.parametrize("old_key", ["sac", "td3"])
def test_the_1x_algorithm_names_fail_validation(old_key):
    # 2.x registers masac / matd3 under the new names; the old keys name nothing.
    with pytest.raises(ValueError, match="Unsupported MARL.algorithm"):
        resolve_algorithm_spec(old_key)


def test_unknown_algorithm_names_fail_validation():
    with pytest.raises(ValueError, match="Unsupported MARL.algorithm"):
        resolve_algorithm_spec("pcma-sac")


def test_algorithm_keys_are_case_sensitive():
    with pytest.raises(ValueError, match="Unsupported MARL.algorithm"):
        resolve_algorithm_spec("PCMAPPO")


def test_launcher_schema_rejects_legacy_intent_policy():
    cfg = OmegaConf.create({
        "MARL": {
            "algorithm": "pcmappo",
            "action_space": {"mode": "speed_only"},
            "intent_policy": {"algorithm": "mappo"},
        }
    })
    with pytest.raises(ValueError, match="MARL.intent_policy"):
        validate_matrix_schema(cfg, source="unit-test")
