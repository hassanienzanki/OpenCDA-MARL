from unittest.mock import patch
from types import SimpleNamespace


def test_sumo_spawner_resolves_shielded_runtime_modes():
    from opencda_marl._kernel.traffic.sumo_spawner import (
        SUMO_RL_RUNTIME_MODES,
        SUMO_RL_SPEED_CHECKS_MODE,
        SUMO_RL_UNSHIELDED_LANE_CHANGE_MODE,
        SumoVehicleSpawner,
    )

    speed_mode, lane_change_mode = SumoVehicleSpawner._resolve_runtime_modes(
        "speed_checks",
        speed_mode=None,
        lane_change_mode=None,
    )

    assert speed_mode == SUMO_RL_SPEED_CHECKS_MODE
    assert lane_change_mode == SUMO_RL_UNSHIELDED_LANE_CHANGE_MODE

    custom_speed_mode, custom_lane_change_mode = SumoVehicleSpawner._resolve_runtime_modes(
        "sumo_default",
        speed_mode=7,
        lane_change_mode=8,
    )

    assert custom_speed_mode == 7
    assert custom_lane_change_mode == 8
    assert set(SUMO_RL_RUNTIME_MODES) == {"unshielded", "speed_checks", "sumo_default"}


def test_sumo_spawner_applies_configured_runtime_modes():
    from opencda_marl._kernel.traffic.sumo_spawner import SumoVehicleSpawner

    spawner = SumoVehicleSpawner.__new__(SumoVehicleSpawner)
    spawner.speed_mode = 31
    spawner.lane_change_mode = 512

    with patch("opencda_marl._kernel.traffic.sumo_spawner.traci.vehicle.setSpeedMode") as set_speed_mode, \
            patch("opencda_marl._kernel.traffic.sumo_spawner.traci.vehicle.setLaneChangeMode") as set_lane_change_mode:
        spawner._configure_rl_vehicle_runtime("veh_1")

    set_speed_mode.assert_called_once_with("veh_1", 31)
    set_lane_change_mode.assert_called_once_with("veh_1", 512)


def test_sumo_env_syncs_active_agent_state_for_live_backpressure():
    from opencda_marl._kernel.envs.sumo_marl_env import SumoMARLEnv
    from opencda_marl._kernel.envs.episode_state import SumoEpisodeState

    env = SumoMARLEnv.__new__(SumoMARLEnv)
    env.traffic_manager = SimpleNamespace(state={})
    env.current_step = 42
    env.current_episode = 3
    env.episode = SumoEpisodeState(terminal_agents={"done_vehicle"})

    with patch(
        "opencda_marl._kernel.envs.sumo_marl_env.traci.vehicle.getIDList",
        return_value=["live_a", "done_vehicle", "live_b"],
    ):
        env._sync_traffic_manager_state()

    assert env.traffic_manager.state == {
        "step": 42,
        "episode": 3,
        "active_agents": 2,
    }
