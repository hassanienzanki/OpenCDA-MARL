from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import os
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from opencda_marl.builtin.extractors import ObservationExtractor
from opencda_marl.api.objectives import (
    ObjectiveOutput,
    ObjectiveSpec,
    build_objective_outputs,
    scalarize_objective_outputs,
)
from opencda_marl.api.preferences import PreferenceRegistry
from opencda_marl._kernel.traffic.sumo_spawner import (
    SUMO_RL_LANE_CHANGE_MODE,
    SUMO_RL_SPEED_MODE,
    SumoVehicleSpawner,
)


_SAC_IMPORT_ERROR = None
_PCMAPPO_IMPORT_ERROR = None


def _load_sac_algorithm():
    # Match the repo's current experimental Windows workaround so the test can
    # exercise SAC without depending on external shell initialization order.
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    global _SAC_IMPORT_ERROR
    try:
        import torch  # noqa: F401
        from opencda_marl.builtin.algorithms.sac import SACAlgorithm
    except OSError as exc:
        _SAC_IMPORT_ERROR = exc
        return None

    return SACAlgorithm


def _load_pcmappo_algorithm():
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    global _PCMAPPO_IMPORT_ERROR
    try:
        import torch  # noqa: F401
        from opencda_marl.builtin.algorithms.pcmappo import PCMAPPOAlgorithm
    except OSError as exc:
        _PCMAPPO_IMPORT_ERROR = exc
        return None

    return PCMAPPOAlgorithm


def test_objective_grouping_maps_canonical_components_to_efficiency_and_safety():
    spec = ObjectiveSpec.from_config(
        {
            "objectives": ["efficiency", "safety"],
            "groups": {
                "efficiency": {
                    "progress": 1.0,
                    "success": 1.0,
                    "step_penalty": 1.0,
                },
                "safety": {
                    "collision": 1.0,
                },
            },
        }
    )

    reward_outputs = {
        "veh_1": {
            "progress": 0.4,
            "success": 0.0,
            "step_penalty": -0.8,
            "collision": 0.0,
        },
        "veh_2": {
            "progress": 0.0,
            "success": 300.0,
            "step_penalty": 0.0,
            "collision": -400.0,
        },
    }

    objective_outputs = build_objective_outputs(reward_outputs, spec)

    assert objective_outputs["veh_1"].components == {
        "efficiency": -0.4,
        "safety": 0.0,
    }
    assert objective_outputs["veh_2"].vector == [300.0, -400.0]


def test_preference_registry_assigns_directional_profiles_deterministically():
    registry = PreferenceRegistry({"profile": "mild_asymmetric_split"})

    north_pref = registry.ensure("veh_north", {"entry_direction": "north"})
    south_pref = registry.ensure("veh_south", {"entry_direction": "south"})
    unknown_pref_first = registry.ensure("veh_42", {})
    unknown_pref_second = registry.ensure("veh_42", {})

    assert np.allclose(north_pref, np.array([0.6, 0.4], dtype=np.float32))
    assert np.allclose(south_pref, np.array([0.4, 0.6], dtype=np.float32))
    assert np.allclose(unknown_pref_first, unknown_pref_second)


def test_preference_registry_uses_spawn_metadata_preference_before_profile():
    registry = PreferenceRegistry({"profile": "mild_asymmetric_split"})

    pref = registry.ensure(
        "veh_north",
        {"entry_direction": "north", "preference": [0.25, 0.75]},
    )

    assert np.allclose(pref, np.array([0.25, 0.75], dtype=np.float32))


def test_observation_extractor_appends_preferences_to_ego_and_all_states():
    extractor = ObservationExtractor(
        {
            "masac": {
                "features": {
                    "rel_x": 1,
                    "rel_y": 1,
                    "speed": 1,
                }
            }
        },
        "masac",
    )

    base_states = extractor.extract(
        {
            "veh_a": {
                "relative_position_to_intersection": {"x": 1.0, "y": 2.0},
                "speed": 30.0,
            },
            "veh_b": {
                "relative_position_to_intersection": {"x": -1.0, "y": -2.0},
                "speed": 20.0,
            },
        }
    )
    augmented = extractor.augment_with_preferences(
        base_states,
        {
            "veh_a": np.array([0.8, 0.2], dtype=np.float32),
            "veh_b": np.array([0.2, 0.8], dtype=np.float32),
        },
    )

    assert augmented["veh_a"]["ego_state"].shape[0] == 5
    assert np.allclose(augmented["veh_a"]["ego_state"][-2:], np.array([0.8, 0.2], dtype=np.float32))
    assert np.allclose(
        augmented["veh_a"]["all_states"]["veh_b"][-2:],
        np.array([0.2, 0.8], dtype=np.float32),
    )


def _three_agent_states():
    rng = np.random.default_rng(7)
    ids = ["veh_a", "veh_b", "veh_c"]
    base = {i: rng.normal(size=3).astype(np.float32) for i in ids}
    return {i: {"ego_state": base[i], "all_states": dict(base)} for i in ids}


def test_augmentation_is_identical_for_every_ego_and_matches_reference():
    """The O(N) augmentation must equal the per-(ego, agent) reference exactly.

    Every ego's ``all_states`` table holds the same augmented vectors, so the
    extractor computes each once and shares it. This pins (a) exact values
    against a naive re-implementation and (b) the ego_state == all_states[ego]
    invariant the batched action path in ``MARLManager`` relies on.
    """
    extractor = ObservationExtractor({"masac": {"features": {"rel_x": 1}}}, "masac")
    states = _three_agent_states()
    prefs = {
        "veh_a": np.array([0.9, 0.1], dtype=np.float32),
        "veh_b": np.array([0.5, 0.5], dtype=np.float32),
        "veh_c": np.array([0.1, 0.9], dtype=np.float32),
    }
    augmented = extractor.augment_with_preferences(states, prefs)

    for ego_id, data in states.items():
        expected_ego = np.concatenate([data["ego_state"], prefs[ego_id]]).astype(np.float32)
        np.testing.assert_array_equal(augmented[ego_id]["ego_state"], expected_ego)
        assert augmented[ego_id]["ego_state"] is augmented[ego_id]["all_states"][ego_id]
        for other_id, other_state in data["all_states"].items():
            expected = np.concatenate([other_state, prefs[other_id]]).astype(np.float32)
            np.testing.assert_array_equal(augmented[ego_id]["all_states"][other_id], expected)
    # One shared table (identity), not N copies of it.
    tables = {id(d["all_states"]) for d in augmented.values()}
    assert len(tables) == 1


def test_augmentation_accepts_int_keyed_preferences_and_missing_table_entry():
    """Preference lookup falls back across str/int keys, and an ego absent
    from the shared table (defensive; extractor never produces this) is
    still augmented from its own ego_state."""
    extractor = ObservationExtractor({"masac": {"features": {"rel_x": 1}}}, "masac")
    states = _three_agent_states()
    # Drop veh_c from every table but keep it as an ego.
    for data in states.values():
        data["all_states"].pop("veh_c", None)
    prefs = {
        "veh_a": np.array([1.0, 0.0], dtype=np.float32),
        "veh_b": np.array([0.0, 1.0], dtype=np.float32),
        "veh_c": np.array([0.5, 0.5], dtype=np.float32),
    }
    augmented = extractor.augment_with_preferences(states, prefs)
    np.testing.assert_array_equal(
        augmented["veh_c"]["ego_state"][-2:], np.array([0.5, 0.5], dtype=np.float32))
    assert augmented["veh_c"]["ego_state"].shape[0] == 5
    # The table is augmented as given — never widened behind the caller's back.
    assert set(augmented["veh_a"]["all_states"]) == {"veh_a", "veh_b"}

    # Int-keyed states + int-keyed preferences resolve too.
    int_states = {1: {"ego_state": np.ones(3, np.float32),
                      "all_states": {"1": np.ones(3, np.float32)}}}
    out = extractor.augment_with_preferences(
        int_states, {1: np.array([0.3, 0.7], dtype=np.float32)})
    np.testing.assert_array_equal(out["1"]["ego_state"][-2:], np.array([0.3, 0.7], np.float32))


def test_objective_scalarization_changes_with_preferences():
    spec = ObjectiveSpec.from_config(
        {
            "objectives": ["efficiency", "safety"],
            "groups": {
                "efficiency": {"progress": 1.0},
                "safety": {"collision": 1.0},
            },
        }
    )
    objective_output = ObjectiveOutput.from_reward_components(
        {"progress": 2.0, "collision": -5.0},
        spec,
    )

    scalarized = scalarize_objective_outputs(
        {"veh_1": objective_output},
        {
            "veh_1": [0.8, 0.2],
        },
    )
    scalarized_alt = scalarize_objective_outputs(
        {"veh_1": objective_output},
        {
            "veh_1": [0.2, 0.8],
        },
    )

    assert abs(scalarized["veh_1"] - 0.6) < 1e-6
    assert abs(scalarized_alt["veh_1"] + 3.6) < 1e-6


def test_sac_preference_modes_support_forward_update_and_checkpoint_roundtrip():
    SACAlgorithm = _load_sac_algorithm()
    if SACAlgorithm is None:
        return
    multi_agent_obs = {
        "veh_a": np.array([1.0, 0.0, 0.5, 0.8, 0.2], dtype=np.float32),
        "veh_b": np.array([-1.0, 0.5, 0.3, 0.2, 0.8], dtype=np.float32),
    }
    next_multi_agent_obs = {
        "veh_a": np.array([0.9, 0.0, 0.6, 0.8, 0.2], dtype=np.float32),
        "veh_b": np.array([-0.8, 0.4, 0.35, 0.2, 0.8], dtype=np.float32),
    }

    for pref_mode in ("direct", "internalpref"):
        config = {
            "warmup_steps": 0,
            "memory_size": 32,
            "batch_size": 4,
            "auto_entropy_tuning": False,
            "init_alpha": 0.2,
            "tau": 0.01,
            "gamma": 0.95,
            "actor_hidden_dims": [16, 16],
            "critic_hidden_dims": [16, 16],
            "lstm_hidden_size": 8,
            "pref_mode": pref_mode,
            "pref_dim": 2,
            "pref_hidden_dim": 8,
            "pref_shift_l2_coef": 0.1,
            "tensorboard": {"enabled": False},
        }

        algorithm = SACAlgorithm(config, state_dim=5, action_dim=1)
        ego_state, context = algorithm._prepare_inputs(multi_agent_obs, "veh_a")
        combined_state = np.concatenate(
            [ego_state.detach().cpu().numpy(), context.detach().cpu().numpy()],
            axis=1,
        )
        _, _, _, aux = algorithm.actor.sample_with_aux(
            algorithm.actor.mean_head.weight.new_tensor(combined_state)
        )

        if pref_mode == "internalpref":
            assert "pref_shift" in aux
            assert aux["shifted_pref"].shape[-1] == 2
        else:
            assert aux == {}

        for step_idx in range(6):
            action = algorithm.select_action(multi_agent_obs, "veh_a", training=True)
            assert action is not None
            assert algorithm.min_action <= action <= algorithm.max_action
            algorithm.store_transition(
                multi_agent_obs,
                "veh_a",
                float(action),
                reward=1.0 - 0.1 * step_idx,
                next_multi_agent_obs=next_multi_agent_obs,
                done=(step_idx == 5),
            )

        metrics = algorithm.update()
        assert len(algorithm.memory) >= algorithm.batch_size
        assert metrics["memory_size"] >= algorithm.batch_size

        if pref_mode == "internalpref":
            assert metrics["pref_shift_l2"] >= 0.0
            assert metrics["pref_shift_norm"] >= 0.0
        else:
            assert metrics["pref_shift_l2"] == 0.0
            assert metrics["pref_shift_norm"] == 0.0

        with TemporaryDirectory() as tmp_dir:
            checkpoint_path = Path(tmp_dir) / f"sac_{pref_mode}.pt"
            algorithm.save(str(checkpoint_path))
            assert checkpoint_path.exists()

            loaded = SACAlgorithm(config, state_dim=5, action_dim=1)
            loaded.load(str(checkpoint_path))
            training_info = loaded.get_training_info()
            assert training_info["pref_mode"] == pref_mode


def test_sac_internalpref_sep_supports_scalar_update_and_checkpoint_roundtrip():
    SACAlgorithm = _load_sac_algorithm()
    if SACAlgorithm is None:
        return
    multi_agent_obs = {
        "veh_a": np.array([1.0, 0.0, 0.5, 0.8, 0.2], dtype=np.float32),
        "veh_b": np.array([-1.0, 0.5, 0.3, 0.2, 0.8], dtype=np.float32),
        "veh_c": np.array([0.4, -0.1, 0.7, 0.8, 0.2], dtype=np.float32),
        "veh_d": np.array([-0.6, -0.4, 0.2, 0.2, 0.8], dtype=np.float32),
    }
    next_multi_agent_obs = {
        "veh_a": np.array([0.9, 0.0, 0.6, 0.8, 0.2], dtype=np.float32),
        "veh_b": np.array([-0.8, 0.4, 0.35, 0.2, 0.8], dtype=np.float32),
        "veh_c": np.array([0.3, -0.2, 0.75, 0.8, 0.2], dtype=np.float32),
        "veh_d": np.array([-0.5, -0.3, 0.25, 0.2, 0.8], dtype=np.float32),
    }

    config = {
        "warmup_steps": 0,
        "memory_size": 32,
        "batch_size": 4,
        "auto_entropy_tuning": False,
        "init_alpha": 0.2,
        "tau": 0.01,
        "gamma": 0.95,
        "actor_hidden_dims": [16, 16],
        "critic_hidden_dims": [16, 16],
        "lstm_hidden_size": 8,
        "pref_mode": "internalpref_sep",
        "critic_mode": "scalar",
        "pref_dim": 2,
        "pref_hidden_dim": 8,
        "pref_shift_l2_coef": 0.1,
        "pref_sep_coef": 0.1,
        "pref_sep_margin": 0.25,
        "pref_sep_use_effective_pref": True,
        "tensorboard": {"enabled": False},
    }

    algorithm = SACAlgorithm(config, state_dim=5, action_dim=1)

    for step_idx, agent_id in enumerate(("veh_a", "veh_b", "veh_c", "veh_d", "veh_a", "veh_b")):
        action = algorithm.select_action(multi_agent_obs, agent_id, training=True)
        algorithm.store_transition(
            multi_agent_obs,
            agent_id,
            float(action),
            reward=1.0 - 0.1 * step_idx,
            next_multi_agent_obs=next_multi_agent_obs,
            done=(step_idx == 5),
        )

    metrics = algorithm.update()
    assert len(algorithm.memory) >= algorithm.batch_size
    assert metrics["memory_size"] >= algorithm.batch_size
    assert metrics["pref_shift_l2"] >= 0.0
    assert metrics["pref_shift_norm"] >= 0.0
    assert metrics["pref_sep_loss"] >= 0.0
    assert metrics["pref_sep_active_pairs"] >= 0.0

    with TemporaryDirectory() as tmp_dir:
        checkpoint_path = Path(tmp_dir) / "sac_internalpref_sep.pt"
        algorithm.save(str(checkpoint_path))
        loaded = SACAlgorithm(config, state_dim=5, action_dim=1)
        loaded.load(str(checkpoint_path))
        training_info = loaded.get_training_info()
        assert training_info["pref_mode"] == "internalpref_sep"
        assert training_info["critic_mode"] == "scalar"
        assert training_info["pref_sep_coef"] == 0.1
        assert training_info["pref_sep_margin"] == 0.25
        assert training_info["pref_sep_use_effective_pref"] is True


def test_pcmappo_preference_modes_support_update_and_checkpoint_roundtrip():
    PCMAPPOAlgorithm = _load_pcmappo_algorithm()
    if PCMAPPOAlgorithm is None:
        return

    multi_agent_obs = {
        "veh_a": np.array([1.0, 0.0, 0.5, 0.8, 0.2], dtype=np.float32),
        "veh_b": np.array([-1.0, 0.5, 0.3, 0.2, 0.8], dtype=np.float32),
    }
    next_multi_agent_obs = {
        "veh_a": np.array([0.9, 0.0, 0.6, 0.8, 0.2], dtype=np.float32),
        "veh_b": np.array([-0.8, 0.4, 0.35, 0.2, 0.8], dtype=np.float32),
    }

    for pref_mode in ("direct", "internalpref", "internalpref_sep"):
        config = {
            "warmup_steps": 0,
            "batch_size": 4,
            "n_steps": 8,
            "clip_param": 0.1,
            "ppo_epochs": 2,
            "num_mini_batch": 2,
            "entropy_coef": 0.01,
            "value_loss_coef": 0.5,
            "max_grad_norm": 0.5,
            "gamma": 0.95,
            "gae_lambda": 0.95,
            "learning_rate": 3e-4,
            "actor_hidden_dims": [16, 16],
            "critic_hidden_dims": [16, 16],
            "lstm_hidden_size": 8,
            "pref_mode": pref_mode,
            "pref_dim": 2,
            "pref_hidden_dim": 8,
            "actor_pref_conditioning": "pref_encoder_fusion",
            "critic_pref_conditioning": "concat",
            "actor_use_pref_shift_head": pref_mode != "direct",
            "pref_shift_l2_coef": 0.1,
            "pref_sep_coef": 0.1,
            "pref_sep_margin": 0.25,
            "reward_aggregation": "team_mean",
            "max_action": 45.0,
            "min_action": 0.0,
            "tensorboard": {"enabled": False},
        }

        algorithm = PCMAPPOAlgorithm(config, state_dim=5, action_dim=1)
        for step_idx, agent_id in enumerate(("veh_a", "veh_b", "veh_a", "veh_b", "veh_a", "veh_b")):
            action = algorithm.select_action(multi_agent_obs, agent_id, training=True)
            assert action is not None
            algorithm.store_transition(
                multi_agent_obs,
                agent_id,
                float(action),
                reward=1.0 - 0.1 * step_idx,
                next_multi_agent_obs=next_multi_agent_obs,
                done=(step_idx == 5),
            )

        metrics = algorithm.update()
        assert metrics["buffer_size"] >= algorithm.batch_size
        assert metrics["scalar_value_loss"] >= 0.0
        if pref_mode == "direct":
            assert metrics["pref_shift_l2"] == 0.0
            assert metrics["pref_sep_loss"] == 0.0
        elif pref_mode == "internalpref":
            assert metrics["pref_shift_l2"] >= 0.0
            assert metrics["pref_sep_loss"] == 0.0
        else:
            assert metrics["pref_shift_l2"] >= 0.0
            assert metrics["pref_sep_loss"] >= 0.0

        with TemporaryDirectory() as tmp_dir:
            checkpoint_path = Path(tmp_dir) / f"pcmappo_{pref_mode}.pt"
            algorithm.save(str(checkpoint_path))
            loaded = PCMAPPOAlgorithm(config, state_dim=5, action_dim=1)
            loaded.load(str(checkpoint_path))
            training_info = loaded.get_training_info()
            assert training_info["pref_mode"] == pref_mode


def test_pcmappo_vector_critic_supports_objective_rewards_and_checkpoint_roundtrip():
    PCMAPPOAlgorithm = _load_pcmappo_algorithm()
    if PCMAPPOAlgorithm is None:
        return

    multi_agent_obs = {
        "veh_a": np.array([1.0, 0.0, 0.5, 0.8, 0.2], dtype=np.float32),
        "veh_b": np.array([-1.0, 0.5, 0.3, 0.2, 0.8], dtype=np.float32),
    }
    next_multi_agent_obs = {
        "veh_a": np.array([0.9, 0.0, 0.6, 0.8, 0.2], dtype=np.float32),
        "veh_b": np.array([-0.8, 0.4, 0.35, 0.2, 0.8], dtype=np.float32),
    }

    config = {
        "warmup_steps": 0,
        "batch_size": 4,
        "n_steps": 8,
        "clip_param": 0.1,
        "ppo_epochs": 2,
        "num_mini_batch": 2,
        "entropy_coef": 0.01,
        "value_loss_coef": 0.5,
        "max_grad_norm": 0.5,
        "gamma": 0.95,
        "gae_lambda": 0.95,
        "learning_rate": 3e-4,
        "actor_hidden_dims": [16, 16],
        "critic_hidden_dims": [16, 16],
        "lstm_hidden_size": 8,
        "pref_mode": "internalpref",
        "pref_dim": 2,
        "pref_hidden_dim": 8,
        "actor_pref_conditioning": "pref_encoder_fusion",
        "critic_pref_conditioning": "concat",
        "actor_use_pref_shift_head": True,
        "pref_shift_l2_coef": 0.1,
        "critic_mode": "vector",
        "reward_dim": 2,
        "critic_vector_l2_coef": 1.0,
        "reward_aggregation": "team_mean",
        "max_action": 45.0,
        "min_action": 0.0,
        "tensorboard": {"enabled": False},
    }

    algorithm = PCMAPPOAlgorithm(config, state_dim=5, action_dim=1)
    for step_idx, agent_id in enumerate(("veh_a", "veh_b", "veh_a", "veh_b", "veh_a", "veh_b")):
        action = algorithm.select_action(multi_agent_obs, agent_id, training=True)
        algorithm.store_transition(
            multi_agent_obs,
            agent_id,
            float(action),
            reward=np.array([1.0 - 0.1 * step_idx, -0.2 * step_idx], dtype=np.float32),
            next_multi_agent_obs=next_multi_agent_obs,
            done=(step_idx == 5),
        )

    metrics = algorithm.update()
    assert metrics["buffer_size"] >= algorithm.batch_size
    assert metrics["vector_value_loss"] >= 0.0
    assert metrics["pref_shift_l2"] >= 0.0

    with TemporaryDirectory() as tmp_dir:
        checkpoint_path = Path(tmp_dir) / "pcmappo_vector.pt"
        algorithm.save(str(checkpoint_path))
        loaded = PCMAPPOAlgorithm(config, state_dim=5, action_dim=1)
        loaded.load(str(checkpoint_path))
        training_info = loaded.get_training_info()
        assert training_info["critic_mode"] == "vector"
        assert training_info["reward_dim"] == 2


def test_manager_routes_objective_vectors_only_for_vector_critic():
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    try:
        from opencda_marl._kernel.policy.manager import MARLManager
    except OSError:
        return

    config = {
        "algorithm": "masac",
        "action_space": {"mode": "speed_only"},
        "reward_profile": "cooperative_intersection_v1",
        "objective_mode": "cooperative_pref_v1",
        "reward_mode": "objective_vector",
        "state_dim": 5,
        "action_dim": 1,
        "preference": {
            "profile": "mild_asymmetric_split",
            "assignment": "entry_direction",
            "injection": "state_augmentation",
            "dim": 2,
        },
        "masac": {
            "pref_mode": "direct",
            "critic_mode": "vector",
            "pref_dim": 2,
            "reward_dim": 2,
            "warmup_steps": 0,
            "memory_size": 32,
            "batch_size": 4,
            "auto_entropy_tuning": False,
            "init_alpha": 0.2,
            "tau": 0.01,
            "gamma": 0.95,
            "actor_hidden_dims": [16, 16],
            "critic_hidden_dims": [16, 16],
            "lstm_hidden_size": 8,
            "tensorboard": {"enabled": False},
        },
    }
    manager = MARLManager(config, "masac")
    spec = ObjectiveSpec.from_config(
        {
            "objectives": ["efficiency", "safety"],
            "groups": {
                "efficiency": {"progress": 1.0},
                "safety": {"collision": 1.0},
            },
        }
    )
    objective_outputs = {
        "veh_1": ObjectiveOutput.from_reward_components(
            {"progress": 1.5, "collision": -2.0},
            spec,
        )
    }
    manager.ensure_agent_preferences({"veh_1": {"entry_direction": "north"}})
    learning_rewards = manager._prepare_learning_rewards({}, objective_outputs=objective_outputs)

    assert isinstance(learning_rewards["veh_1"], np.ndarray)
    assert learning_rewards["veh_1"].shape == (2,)
    assert np.allclose(learning_rewards["veh_1"], np.array([1.5, -2.0], dtype=np.float32))


def test_manager_routes_objective_vectors_only_for_vector_critic_pcmappo():
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    try:
        from opencda_marl._kernel.policy.manager import MARLManager
    except OSError:
        return

    config = {
        "algorithm": "pcmappo",
        "action_space": {"mode": "speed_only"},
        "reward_profile": "cooperative_intersection_v1",
        "objective_mode": "cooperative_pref_v1",
        "reward_mode": "objective_vector",
        "state_dim": 5,
        "action_dim": 1,
        "preference": {
            "profile": "mild_asymmetric_split",
            "assignment": "entry_direction",
            "injection": "state_augmentation",
            "dim": 2,
        },
        "pcmappo": {
            "pref_mode": "direct",
            "critic_mode": "vector",
            "reward_dim": 2,
            "pref_dim": 2,
            "actor_hidden_dims": [16, 16],
            "critic_hidden_dims": [16, 16],
            "lstm_hidden_size": 8,
            "tensorboard": {"enabled": False},
        },
    }
    manager = MARLManager(config, "pcmappo")
    spec = ObjectiveSpec.from_config(
        {
            "objectives": ["efficiency", "safety"],
            "groups": {
                "efficiency": {"progress": 1.0},
                "safety": {"collision": 1.0},
            },
        }
    )
    objective_outputs = {
        "veh_1": ObjectiveOutput.from_reward_components(
            {"progress": 1.5, "collision": -2.0},
            spec,
        )
    }
    manager.ensure_agent_preferences({"veh_1": {"entry_direction": "north"}})
    learning_rewards = manager._prepare_learning_rewards({}, objective_outputs=objective_outputs)

    assert isinstance(learning_rewards["veh_1"], np.ndarray)
    assert learning_rewards["veh_1"].shape == (2,)
    assert np.allclose(learning_rewards["veh_1"], np.array([1.5, -2.0], dtype=np.float32))


def test_manager_routes_intent_rewards_from_objective_vectors_even_with_scalar_speed_rewards():
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    try:
        from opencda_marl._kernel.policy.manager import MARLManager
    except OSError:
        return

    manager = MARLManager(
        {
            "algorithm": None,
            "action_space": {"mode": "speed_only"},
            "reward_profile": "cooperative_intersection_v1",
            "objective_mode": "cooperative_pref_v1",
            "state_dim": 5,
            "action_dim": 1,
            "preference": {"profile": "balanced", "dim": 2},
        },
        None,
    )
    manager.intent_algorithm = object()
    manager.intent_reward_accumulator = SimpleNamespace(reward_dim=2)
    spec = ObjectiveSpec.from_config(
        {
            "objectives": ["efficiency", "safety"],
            "groups": {
                "efficiency": {"progress": 1.0},
                "safety": {"collision": 1.0},
            },
        }
    )
    objective_outputs = {
        "veh_1": ObjectiveOutput.from_reward_components(
            {"progress": 1.5, "collision": -2.0},
            spec,
        )
    }
    scalar_speed_rewards = {"veh_1": -99.0}

    intent_rewards = manager._prepare_intent_rewards(
        scalar_speed_rewards,
        objective_outputs=objective_outputs,
        learning_rewards=scalar_speed_rewards,
    )

    assert isinstance(intent_rewards["veh_1"], np.ndarray)
    assert intent_rewards["veh_1"].shape == (2,)
    assert np.allclose(intent_rewards["veh_1"], np.array([1.5, -2.0], dtype=np.float32))


def test_sac_vector_critic_supports_objective_rewards_and_checkpoint_roundtrip():
    SACAlgorithm = _load_sac_algorithm()
    if SACAlgorithm is None:
        return
    import torch

    config = {
        "warmup_steps": 0,
        "memory_size": 32,
        "batch_size": 4,
        "auto_entropy_tuning": False,
        "init_alpha": 0.2,
        "tau": 0.01,
        "gamma": 0.95,
        "actor_hidden_dims": [16, 16],
        "critic_hidden_dims": [16, 16],
        "lstm_hidden_size": 8,
        "pref_mode": "internalpref",
        "critic_mode": "vector",
        "reward_dim": 2,
        "pref_dim": 2,
        "pref_hidden_dim": 8,
        "pref_shift_l2_coef": 0.1,
        "pref_update_mode": "shifted_pref_actor",
        "pref_shift_clip": 0.5,
        "pref_shift_mix": 0.5,
        "device": "cpu",
        "tensorboard": {"enabled": False},
    }
    algorithm = SACAlgorithm(config, state_dim=5, action_dim=1)
    multi_agent_obs = {
        "veh_a": np.array([1.0, 0.0, 0.5, 0.8, 0.2], dtype=np.float32),
        "veh_b": np.array([-1.0, 0.5, 0.3, 0.2, 0.8], dtype=np.float32),
    }
    next_multi_agent_obs = {
        "veh_a": np.array([0.9, 0.0, 0.6, 0.8, 0.2], dtype=np.float32),
        "veh_b": np.array([-0.8, 0.4, 0.35, 0.2, 0.8], dtype=np.float32),
    }

    ego_state, context = algorithm._prepare_inputs(multi_agent_obs, "veh_a")
    combined_state = torch.from_numpy(
        np.concatenate(
            [ego_state.detach().cpu().numpy(), context.detach().cpu().numpy()],
            axis=1,
        ).astype(np.float32)
    ).to(algorithm.device)
    q1, q2 = algorithm.critic(
        combined_state,
        torch.zeros((1, 1), dtype=torch.float32, device=algorithm.device),
    )
    assert q1.shape == (1, 2)
    assert q2.shape == (1, 2)

    for step_idx in range(6):
        action = algorithm.select_action(multi_agent_obs, "veh_a", training=True)
        algorithm.store_transition(
            multi_agent_obs,
            "veh_a",
            float(action),
            reward=np.array([1.0 - 0.1 * step_idx, -0.2 * step_idx], dtype=np.float32),
            next_multi_agent_obs=next_multi_agent_obs,
            done=(step_idx == 5),
        )

    metrics = algorithm.update()
    assert len(algorithm.memory) >= algorithm.batch_size
    assert metrics["memory_size"] >= algorithm.batch_size
    assert "q1_obj_0_mean" in metrics
    assert "q2_obj_1_mean" in metrics
    assert "actor_q_scalarized_mean" in metrics
    assert "actor_q_rawpref_mean" in metrics
    assert "actor_q_effectivepref_mean" in metrics
    assert "actor_q_shiftedpref_mean" in metrics
    assert "effective_pref_obj_0_mean" in metrics
    assert "effective_pref_obj_1_mean" in metrics
    assert "shifted_pref_obj_0_mean" in metrics
    assert "shifted_pref_obj_1_mean" in metrics
    assert "pref_shift_abs_max" in metrics

    with TemporaryDirectory() as tmp_dir:
        checkpoint_path = Path(tmp_dir) / "sac_vector.pt"
        algorithm.save(str(checkpoint_path))
        loaded = SACAlgorithm(config, state_dim=5, action_dim=1)
        loaded.load(str(checkpoint_path))
        training_info = loaded.get_training_info()
        assert training_info["critic_mode"] == "vector"
        assert training_info["reward_dim"] == 2
        assert training_info["pref_update_mode"] == "shifted_pref_actor"
        assert training_info["pref_shift_clip"] == 0.5
        assert training_info["pref_shift_mix"] == 0.5


def test_sac_internalpref_effective_preference_and_resolution_modes():
    SACAlgorithm = _load_sac_algorithm()
    if SACAlgorithm is None:
        return
    import torch

    config = {
        "warmup_steps": 0,
        "memory_size": 8,
        "batch_size": 2,
        "auto_entropy_tuning": False,
        "init_alpha": 0.2,
        "tau": 0.01,
        "gamma": 0.95,
        "actor_hidden_dims": [16, 16],
        "critic_hidden_dims": [16, 16],
        "lstm_hidden_size": 8,
        "pref_mode": "internalpref",
        "critic_mode": "vector",
        "reward_dim": 2,
        "pref_dim": 2,
        "pref_hidden_dim": 8,
        "pref_shift_l2_coef": 0.1,
        "pref_shift_clip": 0.25,
        "pref_shift_mix": 0.5,
        "tensorboard": {"enabled": False},
    }
    algorithm = SACAlgorithm(config, state_dim=5, action_dim=1)
    raw_pref = torch.tensor([[0.8, 0.2]], dtype=torch.float32)
    actor_aux = {
        "shifted_pref": torch.tensor([[0.2, 0.8]], dtype=torch.float32, requires_grad=True),
        "effective_pref": torch.tensor([[0.5, 0.5]], dtype=torch.float32, requires_grad=True),
    }

    algorithm.pref_update_mode = "direct_pref_actor"
    resolved = algorithm._resolve_actor_preferences(raw_pref, actor_aux)
    assert torch.allclose(resolved, raw_pref)

    algorithm.pref_update_mode = "shifted_pref_actor"
    resolved = algorithm._resolve_actor_preferences(raw_pref, actor_aux)
    assert torch.allclose(resolved, actor_aux["effective_pref"])
    assert resolved.requires_grad is True

    algorithm.pref_update_mode = "shifted_pref_actor_detached"
    resolved = algorithm._resolve_actor_preferences(raw_pref, actor_aux)
    assert torch.allclose(resolved, actor_aux["effective_pref"])
    assert resolved.requires_grad is False


def test_sumo_spawner_configures_rl_runtime_modes():
    spawner = SumoVehicleSpawner.__new__(SumoVehicleSpawner)

    with patch("opencda_marl._kernel.traffic.sumo_spawner.traci.vehicle.setSpeedMode") as set_speed_mode, \
            patch("opencda_marl._kernel.traffic.sumo_spawner.traci.vehicle.setLaneChangeMode") as set_lane_change_mode:
        spawner._configure_rl_vehicle_runtime("veh_1")

    set_speed_mode.assert_called_once_with("veh_1", SUMO_RL_SPEED_MODE)
    set_lane_change_mode.assert_called_once_with("veh_1", SUMO_RL_LANE_CHANGE_MODE)


def test_sumo_env_records_and_removes_collided_vehicles():
    try:
        from opencda_marl._kernel.envs.sumo_marl_env import SumoMARLEnv
    except (OSError, ModuleNotFoundError):
        return
    from opencda_marl._kernel.envs.episode_state import SumoEpisodeState

    env = SumoMARLEnv.__new__(SumoMARLEnv)
    env.current_step = 17
    env.episode = SumoEpisodeState(active_vehicles={
        "veh_a": {"spawn_step": 1, "collisions": 0},
        "veh_b": {"spawn_step": 2, "collisions": 0},
    })

    class _Spawner:
        def __init__(self):
            self.removed = []

        def remove_vehicle(self, vehicle_id):
            self.removed.append(vehicle_id)

    env.vehicle_spawner = _Spawner()

    collision = SimpleNamespace(collider="veh_a", victim="veh_b")

    with patch("opencda_marl._kernel.envs.sumo_marl_env.traci.simulation.getArrivedIDList", return_value=[]), \
            patch("opencda_marl._kernel.envs.sumo_marl_env.traci.simulation.getCollisions", return_value=[collision]), \
            patch("opencda_marl._kernel.envs.sumo_marl_env.traci.vehicle.getIDList", return_value=["veh_a", "veh_b"]), \
            patch("opencda_marl._kernel.envs.sumo_marl_env.traci.vehicle.remove") as remove_vehicle:
        events = env._check_events()

    assert [event.event_type for event in events] == ["collision", "collision"]
    assert env.episode.collided_vehicles == {"veh_a", "veh_b"}
    assert env.episode.terminal_agents == {"veh_a", "veh_b"}
    assert env.episode.active_vehicles["veh_a"]["collisions"] == 1
    assert env.episode.active_vehicles["veh_b"]["collisions"] == 1
    assert set(env.vehicle_spawner.removed) == {"veh_a", "veh_b"}
    remove_vehicle.assert_any_call("veh_a")
    remove_vehicle.assert_any_call("veh_b")
