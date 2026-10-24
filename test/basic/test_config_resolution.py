"""Config resolution and merge tests for MARL, MOMARL, SUMO, and cosim scenarios."""

from pathlib import Path
import sys

_TEST_DIR = str(Path(__file__).resolve().parent)
if _TEST_DIR not in sys.path:
    sys.path.insert(0, _TEST_DIR)

from omegaconf import OmegaConf

from test_helpers import (
    REPO_ROOT,
    load_module,
    merged_scenario_config,
)

from opencda_marl.api.rewards import resolve_reward_configuration

# ── module loading ──────────────────────────────────────────────────────────

config_merger = load_module("config_merger", "opencda_marl/utils/config_merger.py")
config_locator = load_module("config_locator", "opencda_marl/utils/config_locator.py")
vehicle_defaults = load_module("vehicle_defaults", "opencda_marl/_kernel/adapter/vehicle_defaults.py")


# ── helper for raw OmegaConf merges (needed by some tests) ─────────────────

def _raw_merge(scenario_ref):
    resolved = config_locator.resolve_marl_config_paths(
        REPO_ROOT,
        scenario_ref,
        custom_config_roots=[REPO_ROOT / "opencda_marl" / "_bundled" / "configs"],
    )
    merged = config_merger.smart_merge(
        OmegaConf.load(resolved["default_yaml"]),
        OmegaConf.load(resolved["scenario_base_yaml"]),
    )
    merged = config_merger.smart_merge(merged, OmegaConf.load(resolved["scenario_yaml"]))
    return resolved, merged


# ═══════════════════════════════════════════════════════════════════════════
# Tests: default world config contract (from test_default_world_config.py)
# ═══════════════════════════════════════════════════════════════════════════

def test_marl_vanilla_world_contract_is_stable():
    resolved = merged_scenario_config("marl/intersection/vanilla")

    world = resolved["world"]
    localization = resolved["agents"]["vehicle"]["sensing"]["localization"]
    controller = resolved["agents"]["vehicle"]["controller"]["args"]
    map_cfg = resolved["scenario"]["map"]
    spectator = resolved["scenario"]["spectator"]
    runtime = resolved["runtime"]
    behavior = resolved["agents"]["vanilla"]
    local_planner = resolved["agents"]["vanilla"]["local_planner"]
    vehicle_local_planner = resolved["agents"]["vehicle"]["behavior"]["local_planner"]
    route_visualization = resolved["agents"]["route_visualization"]

    assert world["sync_mode"] is True
    assert world["host"] == "127.0.0.1"
    assert world["client_port"] == 2000
    assert world["timeout"] == 20.0
    assert world["fixed_delta_seconds"] == 0.05
    assert world["seed"] == 11
    assert runtime["backend"] == "carla"
    assert runtime["reset"]["enabled"] is False
    assert route_visualization["arrow_step"] == 3
    assert route_visualization["arrow_thickness"] == 0.18
    assert route_visualization["arrow_size"] == 0.45

    assert localization["dt"] == world["fixed_delta_seconds"]
    assert controller["dt"] == world["fixed_delta_seconds"]

    assert controller["lat"]["k_p"] == 1.95
    assert controller["lat"]["k_d"] == 0.2
    assert controller["lat"]["k_i"] == 0.05
    assert controller["lon"]["k_p"] == 0.37
    assert controller["max_steering"] == 0.8
    assert controller["max_brake"] == 1.0
    assert controller["max_throttle"] == 1.0
    assert behavior["destination_reach_threshold"] == 3.0
    assert behavior["debug"]["global_planner"]["localize"] is False
    assert behavior["debug"]["global_planner"]["lane_change_links"] is False
    assert behavior["debug"]["global_planner"]["graph"]["enabled"] is False
    assert behavior["debug"]["global_planner"]["graph"]["junction_only"] is False
    assert behavior["debug"]["global_planner"]["graph"]["route_only"] is False
    assert behavior["debug"]["global_planner"]["graph"]["node_labels"] is False
    assert behavior["debug"]["global_planner"]["graph"]["edge_labels"] is False
    assert behavior["debug"]["global_planner"]["path"]["enabled"] is False
    assert behavior["debug"]["global_planner"]["path"]["hop_labels"] is False
    assert behavior["debug"]["global_planner"]["path"]["start_end_labels"] is False
    planner_visualize = resolved["scenario"]["traffic"]["planner"]["visualize"]
    assert planner_visualize["junction_center"] is False
    assert planner_visualize["extent_box"] is False
    assert planner_visualize["text"] is False

    assert resolved["scenario"]["type"] == "intersection"
    assert resolved["agents"]["agent_type"] == "vanilla"
    assert map_cfg["town"] == "intersection_local_xodr"
    assert map_cfg["xodr_path"] == "opencda_marl/assets/intersection/intersection.xodr"
    assert spectator["preset"] == "intersection_bird_eye"
    assert resolved["scenario"]["simulation"]["max_steps"] == 2400
    assert resolved["outputs"]["evaluation"]["mode"] == "disabled"
    assert resolved["outputs"]["tensorboard"]["enabled"] is False
    assert local_planner["debug"] is False
    assert local_planner["debug_trajectory"] is False
    assert vehicle_local_planner["trajectory_point_size"] == 0.05
    assert vehicle_local_planner["trajectory_arrow_size"] == 0.2
    assert vehicle_local_planner["trajectory_arrow_thickness"] == 0.08
    assert vehicle_local_planner["route_debug_life_time"] == 8.0
    assert vehicle_local_planner["route_debug_point_size"] == 0.05


def test_vehicle_defaults_match_yaml():
    resolved = merged_scenario_config("marl/intersection/vanilla")
    yaml_ctrl = resolved["agents"]["vehicle"]["controller"]["args"]
    defaults = vehicle_defaults.get_controller_defaults(world_dt=yaml_ctrl["dt"])["args"]

    assert defaults["lat"]["k_p"] == yaml_ctrl["lat"]["k_p"]
    assert defaults["lat"]["k_d"] == yaml_ctrl["lat"]["k_d"]
    assert defaults["lat"]["k_i"] == yaml_ctrl["lat"]["k_i"]
    assert defaults["max_steering"] == yaml_ctrl["max_steering"]
    assert defaults["max_brake"] == yaml_ctrl["max_brake"]
    assert defaults["max_throttle"] == yaml_ctrl["max_throttle"]


def test_rule_based_debug_config_uses_single_master_switch():
    resolved = merged_scenario_config("marl/intersection/rule_based")
    rule_based = resolved["agents"]["rule_based"]
    assert rule_based["debug"] is False
    assert "debug_rules" not in rule_based


# ═══════════════════════════════════════════════════════════════════════════
# Tests: PCMA config resolution (from test_pcma_config_resolution.py)
# ═══════════════════════════════════════════════════════════════════════════

def test_pcma_smoke_config_resolves_and_merges():
    resolved_paths, merged = _raw_merge("pcma/intersection/smoke")

    assert resolved_paths["family"] == "pcma"
    assert resolved_paths["scenario_name"] == "intersection"
    assert resolved_paths["profile_name"] == "smoke"
    assert merged["scenario"]["type"] == "intersection"
    assert merged["scenario"]["map"]["xodr_path"] == "opencda_marl/assets/intersection/intersection.xodr"
    assert merged["agents"]["agent_type"] == "marl"
    assert merged["MARL"]["algorithm"] == "masac"
    assert merged["scenario"]["simulation"]["max_steps"] == 1200
    assert merged["outputs"]["evaluation"]["mode"] == "disabled"
    assert merged["outputs"]["tensorboard"]["enabled"] is False
    assert merged["MARL"]["training"]["save_checkpoints"] is False
    assert merged["MARL"]["reward_profile"] == "cooperative_intersection_v1"


def test_marl_smoke_config_resolves_and_merges():
    resolved_paths, merged = _raw_merge("marl/intersection/smoke")

    assert resolved_paths["family"] == "marl"
    assert resolved_paths["scenario_name"] == "intersection"
    assert resolved_paths["profile_name"] == "smoke"
    assert merged["scenario"]["type"] == "intersection"
    assert merged["scenario"]["map"]["xodr_path"] == "opencda_marl/assets/intersection/intersection.xodr"
    assert merged["agents"]["agent_type"] == "marl"
    assert merged["MARL"]["algorithm"] == "masac"
    assert merged["scenario"]["simulation"]["max_steps"] == 1200
    assert merged["agents"]["marl"]["max_speed"] == 45
    assert merged["MARL"]["masac"]["max_action"] == 45.0
    assert merged["outputs"]["evaluation"]["mode"] == "disabled"
    assert merged["outputs"]["tensorboard"]["enabled"] is False
    assert merged["MARL"]["training"]["save_checkpoints"] is False
    assert merged["MARL"]["reward_profile"] == "simple_progress_v1"


def test_marl_behavior_config_resolves_to_local_xodr_and_quiet_baseline():
    resolved_paths, merged = _raw_merge("marl/intersection/behavior")

    assert resolved_paths["family"] == "marl"
    assert resolved_paths["scenario_name"] == "intersection"
    assert resolved_paths["profile_name"] == "behavior"
    assert merged["scenario"]["map"]["town"] == "intersection_local_xodr"
    assert merged["scenario"]["map"]["xodr_path"] == "opencda_marl/assets/intersection/intersection.xodr"
    assert merged["agents"]["agent_type"] == "behavior"
    assert merged["agents"]["behavior"]["overtake_allowed"] is False
    assert merged["agents"]["behavior"]["route_end_policy"] == "terminate"
    assert merged["outputs"]["evaluation"]["mode"] == "disabled"
    assert merged["outputs"]["tensorboard"]["enabled"] is False
    assert merged["runtime"]["reset"]["enabled"] is False


def test_pcma_intersection_config_resolves_and_merges():
    resolved_paths, merged = _raw_merge("pcma/intersection/vanilla")

    assert resolved_paths["family"] == "pcma"
    assert resolved_paths["scenario_name"] == "intersection"
    assert resolved_paths["profile_name"] == "vanilla"
    assert merged["scenario"]["type"] == "intersection"
    assert merged["scenario"]["map"]["xodr_path"] == "opencda_marl/assets/intersection/intersection.xodr"
    assert merged["agents"]["agent_type"] == "vanilla"
    assert merged["scenario"]["simulation"]["max_steps"] == 2400
    assert merged["outputs"]["evaluation"]["mode"] == "disabled"
    assert merged["outputs"]["tensorboard"]["enabled"] is False


def test_pcma_intersection_sac_config_resolves_and_merges():
    resolved_paths, merged = _raw_merge("pcma/intersection/masac")

    assert resolved_paths["family"] == "pcma"
    assert resolved_paths["scenario_name"] == "intersection"
    assert resolved_paths["profile_name"] == "masac"
    assert merged["scenario"]["type"] == "intersection"
    assert merged["scenario"]["map"]["xodr_path"] == "opencda_marl/assets/intersection/intersection.xodr"
    assert merged["agents"]["agent_type"] == "marl"
    assert merged["MARL"]["algorithm"] == "masac"
    assert merged["scenario"]["simulation"]["max_steps"] == 2400
    assert merged["outputs"]["evaluation"]["mode"] == "disabled"
    assert merged["outputs"]["tensorboard"]["enabled"] is True
    assert merged["outputs"]["tensorboard"]["dir"] == "runs"
    assert merged["outputs"]["tensorboard"]["mode"] == "episode_first"
    assert merged["outputs"]["tensorboard"]["keep_last_runs"] == 0
    assert merged["outputs"]["tensorboard"]["max_episodes_per_run"] == 100
    assert merged["outputs"]["metrics_archive"]["export_interval"] == 1
    assert merged["outputs"]["metrics_archive"]["dir"] == "metrics_history/pcma_intersection"
    assert merged["MARL"]["training"]["save_checkpoints"] is False
    assert merged["MARL"]["training"]["checkpoint_dir"] == "artifacts/checkpoints/pcma_intersection"
    assert merged["MARL"]["reward_profile"] == "cooperative_intersection_v1"

    resolved_container = OmegaConf.to_container(merged, resolve=True)
    resolved_reward = resolve_reward_configuration(resolved_container["MARL"])
    assert resolved_reward.reward_spec.mode_for("collision") == "learn"
    assert resolved_reward.reward_spec.mode_for("progress") == "learn"
    assert resolved_reward.reward_spec.mode_for("ttc") == "disabled"
    assert resolved_reward.reward_spec.components == (
        "collision",
        "success",
        "step_penalty",
        "speed",
        "stop",
        "progress",
        "ttc",
        "yielding",
        "clearance",
    )
    assert resolved_reward.reward_spec.mode_for("speed") == "disabled"


def test_pcma_exemplars_resolve_and_merge():
    # The direct-preference, vector-critic profiles are the PCMA exemplars,
    # one per learner; both run the same scenario, outputs and preferences.
    refs = [
        ("pcma/intersection/pcmasac_directpref_vector", "pcmasac"),
        ("pcma/intersection/pcmappo_directpref_vector", "pcmappo"),
    ]

    for scenario_ref, algorithm_name in refs:
        resolved_paths, merged = _raw_merge(scenario_ref)

        assert resolved_paths["family"] == "pcma"
        assert resolved_paths["profile_name"] == scenario_ref.rsplit("/", 1)[1]
        assert merged["scenario"]["type"] == "intersection"
        assert merged["runtime"]["backend"] == "carla"
        assert merged["scenario"]["simulation"]["max_episodes"] == 50
        assert merged["scenario"]["traffic"]["mode"] == "live"
        assert merged["scenario"]["traffic"]["max_active_agents"] == 10
        assert merged["runtime"]["reset"]["enabled"] is True
        assert merged["runtime"]["reset"]["reset_frequency"] == 1
        assert merged["agents"]["marl"]["debug"] is False
        assert merged["agents"]["marl"]["local_planner"]["debug"] is False
        assert merged["agents"]["marl"]["local_planner"]["debug_trajectory"] is False
        assert merged["outputs"]["evaluation"]["mode"] == "disabled"
        assert merged["outputs"]["tensorboard"]["enabled"] is True
        assert merged["outputs"]["tensorboard"]["dir"] == "runs"
        assert merged["outputs"]["metrics_archive"]["enabled"] is True
        assert merged["MARL"]["algorithm"] == algorithm_name
        assert merged["MARL"]["reward_profile"] == "cooperative_intersection_v1"
        assert merged["MARL"]["state_dim"] == 11
        assert merged["MARL"]["training"]["save_checkpoints"] is True
        assert merged["MARL"]["training"]["checkpoint_dir"].startswith("artifacts/checkpoints/")
        assert merged["MARL"]["training"]["save_freq"] == 1
        assert merged["MARL"]["training"]["keep_checkpoints"] == 1

        assert merged["MARL"]["objective_mode"] == "cooperative_pref_v1"
        assert merged["MARL"]["preference"]["profile"] == "mild_asymmetric_split"
        assert merged["MARL"]["preference"]["assignment"] == "entry_direction"
        assert merged["MARL"]["preference"]["injection"] == "state_augmentation"
        assert merged["MARL"]["preference"]["dim"] == 2
        assert merged["MARL"][algorithm_name]["pref_mode"] == "direct"

        assert merged["MARL"]["reward_mode"] == "objective_vector"
        assert merged["MARL"][algorithm_name]["critic_mode"] == "vector"


def test_momarl_route_intent_config_uses_artifact_paths():
    # learn_path_pcmappo is the learned-route profile.
    resolved_paths, merged = _raw_merge("momarl/intersection/learn_path_pcmappo")

    assert resolved_paths["family"] == "momarl"
    assert resolved_paths["scenario_name"] == "intersection"
    assert resolved_paths["profile_name"] == "learn_path_pcmappo"
    assert merged["MARL"]["algorithm"] == "pcmappo"
    # agents.preference is the canonical sampler home (moved out of route_intent)
    assert merged["agents"]["preference"]["mode"] == "per_vehicle"
    assert merged["outputs"]["tensorboard"]["enabled"] is True
    assert merged["outputs"]["tensorboard"]["dir"] == "runs"
    assert merged["MARL"]["training"]["save_checkpoints"] is True
    assert merged["MARL"]["training"]["checkpoint_dir"].startswith("artifacts/checkpoints/")


def test_sumo_gui_is_a_profile_knob_not_a_family_default_for_runs():
    # The retired *_gui profile only flipped world.sumo_gui, a documented
    # knob. The kept `sac` profile is the windowed demo (it sets it on), while
    # the route profiles used for headless training switch it off.
    resolved_paths, merged = _raw_merge("sumo-marl/intersection/sac")
    assert resolved_paths["profile_name"] == "sac"
    assert merged["runtime"]["backend"] == "sumo"
    assert merged["world"]["sumo_gui"] is True

    _, headless = _raw_merge("sumo-marl/intersection/route_speed_only_pcmappo")
    assert headless["world"]["sumo_gui"] is False


def test_marl_td3_config_resolves_and_merges():
    resolved_paths, merged = _raw_merge("marl/intersection/td3")

    assert resolved_paths["family"] == "marl"
    assert resolved_paths["scenario_name"] == "intersection"
    assert resolved_paths["profile_name"] == "td3"
    assert merged["MARL"]["algorithm"] == "matd3"
    assert merged["MARL"]["reward_profile"] == "simple_progress_v1"
    assert merged["MARL"]["state_dim"] == 8
    assert merged["MARL"]["matd3"]["features"]["lane_position"] == 1


# ═══════════════════════════════════════════════════════════════════════════
# Tests: runtime backend config (from test_runtime_backend_config.py)
# ═══════════════════════════════════════════════════════════════════════════

def test_runtime_backend_config_is_explicit():
    marl_resolved = merged_scenario_config("marl/intersection/vanilla")
    sumo_resolved = merged_scenario_config("sumo-marl/intersection/sac")
    cosim_resolved = merged_scenario_config("cosim-marl/intersection/sac")

    assert marl_resolved["runtime"]["backend"] == "carla"
    assert marl_resolved["runtime"]["reset"]["enabled"] is False
    assert marl_resolved["scenario"]["type"] == "intersection"
    assert marl_resolved["outputs"]["root_dir"] == "artifacts"
    assert marl_resolved["outputs"]["tensorboard"]["mode"] == "episode_first"
    assert sumo_resolved["runtime"]["backend"] == "sumo"
    assert sumo_resolved["runtime"]["sumo_cfg"] == "opencda_marl/assets/intersection/intersection.sumocfg"
    assert sumo_resolved["scenario"]["type"] == "intersection"
    assert sumo_resolved["scenario"]["map"]["town"] == "intersection_local_xodr"
    assert sumo_resolved["scenario"]["map"]["xodr_path"] == "opencda_marl/assets/intersection/intersection.xodr"
    assert sumo_resolved["MARL"]["algorithm"] == "masac"
    assert sumo_resolved["MARL"]["reward_profile"] == "cooperative_intersection_v1"
    assert sumo_resolved["outputs"]["root_dir"] == "artifacts"
    assert "reset" not in sumo_resolved["runtime"]
    assert cosim_resolved["runtime"]["backend"] == "cosim"
    assert cosim_resolved["runtime"]["sumo_cfg"] == "opencda_marl/assets/intersection/intersection.sumocfg"
    assert cosim_resolved["scenario"]["type"] == "intersection"
    assert cosim_resolved["scenario"]["map"]["town"] == "intersection_local_xodr"
    assert cosim_resolved["scenario"]["map"]["xodr_path"] == "opencda_marl/assets/intersection/intersection.xodr"
    assert cosim_resolved["MARL"]["algorithm"] == "masac"
    assert cosim_resolved["outputs"]["root_dir"] == "artifacts"
    assert "reset" not in cosim_resolved["runtime"]


# ── runner ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name}: ok")
