"""Standalone traffic-schema MOMARL intersection environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from opencda_marl.envs.traffic_schema import load_traffic_schema_spec
from opencda_marl.envs.traffic_schema import DIRECTIONS, MOVEMENTS
from opencda_marl.envs.spaces import Box, Discrete

DEFAULT_ACTION_PROFILES = np.asarray(
    [
        [0.30, 0.40, 0.30],  # historical
        [0.15, 0.70, 0.15],  # straight-heavy
        [0.60, 0.25, 0.15],  # left-heavy
        [0.15, 0.25, 0.60],  # right-heavy
        [0.45, 0.10, 0.45],  # turn-heavy
    ],
    dtype=np.float32,
)


@dataclass(frozen=True)
class RewardModel:
    cap_left: float = 8.0
    cap_straight: float = 12.0
    cap_right: float = 10.0
    safety_scale: float = 0.05


def parallel_env(**kwargs):
    """Factory for the standalone traffic-schema MOMARL env."""
    return TrafficSchemaIntersectionParallelEnv(**kwargs)


class TrafficSchemaIntersectionParallelEnv:
    """One-shot MOMARL route-allocation game for an intersection schema."""

    metadata = {
        "name": "traffic_schema_intersection",
        "render_modes": [],
        "is_parallelizable": True,
    }

    def __init__(
        self,
        traffic_config_path: str,
        recording_path: str | None = None,
        episode_horizon_s: int = 120,
        queue_scale: int = 20,
        action_profiles: list[list[float]] | None = None,
    ):
        self.schema_spec = load_traffic_schema_spec(
            traffic_config_path=traffic_config_path,
            recording_path=recording_path,
            episode_horizon_s=episode_horizon_s,
        )
        self.possible_agents = list(DIRECTIONS)
        self.agents = self.possible_agents[:]
        self.queue_scale = float(queue_scale)
        self.queue_low = 1
        self.queue_high = 20
        self.reward_model = RewardModel()
        self.rng = np.random.default_rng()

        profiles = np.asarray(
            DEFAULT_ACTION_PROFILES if action_profiles is None else action_profiles,
            dtype=np.float32,
        )
        if profiles.ndim != 2 or profiles.shape[1] != len(MOVEMENTS):
            raise ValueError(
                f"action_profiles must have shape (n, {len(MOVEMENTS)}), got {profiles.shape}"
            )
        if np.any(profiles < 0.0):
            raise ValueError("action_profiles must be non-negative.")
        row_sums = profiles.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-6):
            raise ValueError("Each action profile must sum to 1.")
        self.action_profiles = profiles

        self.action_spaces = {
            agent: Discrete(self.action_profiles.shape[0])
            for agent in self.possible_agents
        }
        self.observation_spaces = {
            agent: Box(low=0.0, high=4.0, shape=(8,), dtype=np.float32)
            for agent in self.possible_agents
        }
        self.reward_spaces = {
            agent: Box(low=-1.0e6, high=0.0, shape=(2,), dtype=np.float32)
            for agent in self.possible_agents
        }

        self.terminations = {agent: False for agent in self.possible_agents}
        self.truncations = {agent: False for agent in self.possible_agents}
        self._queues: dict[str, int] = {}
        self._last_infos: dict[str, dict[str, Any]] = {}

    def action_space(self, agent: str) -> Discrete:
        return self.action_spaces[agent]

    def observation_space(self, agent: str) -> Box:
        return self.observation_spaces[agent]

    def reward_space(self, agent: str) -> Box:
        return self.reward_spaces[agent]

    def reset(self, seed: int | None = None, options: dict | None = None):
        del options
        if seed is not None:
            self.rng = np.random.default_rng(int(seed))
        self.agents = self.possible_agents[:]
        self.terminations = {agent: False for agent in self.possible_agents}
        self.truncations = {agent: False for agent in self.possible_agents}
        self._queues = self._sample_queues()
        observations = self._build_observations()
        infos = {
            agent: {
                "queue": self._queues[agent],
                "movement_prior": self.schema_spec.movement_priors[agent].copy(),
            }
            for agent in self.agents
        }
        self._last_infos = infos
        return observations, infos

    def close(self):
        """No resources to release for the standalone env."""

    def render(self):
        raise NotImplementedError("The standalone traffic-schema env does not implement rendering.")

    def step(self, actions: dict[str, int]):
        if not self.agents or not actions:
            self.agents = []
            return {}, {}, {}, {}, {}

        missing = [agent for agent in self.agents if agent not in actions]
        if missing:
            raise ValueError(f"Missing actions for agents: {missing}")
        for agent, action in actions.items():
            if not self.action_spaces[agent].contains(action):
                raise ValueError(f"Invalid action {action!r} for agent {agent!r}")

        rewards, infos = self._evaluate_joint_action(actions)
        self.terminations = {agent: True for agent in self.possible_agents}
        self.truncations = {agent: False for agent in self.possible_agents}
        self._last_infos = infos
        terminations = self.terminations.copy()
        truncations = self.truncations.copy()
        self.agents = []
        return {}, rewards, terminations, truncations, infos

    def _sample_queues(self) -> dict[str, int]:
        queues = {}
        for direction in self.possible_agents:
            lam = float(self.schema_spec.approach_lambdas[direction])
            sample = int(self.rng.poisson(lam=lam))
            queues[direction] = int(np.clip(sample, self.queue_low, self.queue_high))
        return queues

    def _build_observations(self) -> dict[str, np.ndarray]:
        global_queue = sum(self._queues.values()) / self.queue_scale
        obs = {}
        for agent in self.possible_agents:
            prior = self.schema_spec.movement_priors[agent]
            obs[agent] = np.asarray(
                [
                    self._queues[agent] / self.queue_scale,
                    self._queues[self._opposite(agent)] / self.queue_scale,
                    self._queues[self._cw(agent)] / self.queue_scale,
                    self._queues[self._ccw(agent)] / self.queue_scale,
                    global_queue,
                    float(prior[0]),
                    float(prior[1]),
                    float(prior[2]),
                ],
                dtype=np.float32,
            )
        return obs

    def _evaluate_joint_action(self, actions: dict[str, int]):
        loads = self._compute_loads(actions)
        conflicts = self._compute_conflicts(loads)
        travel_times = self._compute_travel_times(loads, conflicts)

        rewards = {}
        infos = {}
        for agent in self.possible_agents:
            queue = max(float(self._queues[agent]), 1.0)
            weighted_tt = sum(
                loads[agent][movement] * travel_times[agent][movement]
                for movement in MOVEMENTS
            )
            weighted_conflict = sum(
                loads[agent][movement] * conflicts[agent][movement]
                for movement in MOVEMENTS
            )
            efficiency = -weighted_tt / queue
            safety = -self.reward_model.safety_scale * weighted_conflict / queue
            rewards[agent] = np.asarray([efficiency, safety], dtype=np.float32)
            infos[agent] = {
                "queue": int(self._queues[agent]),
                "action_index": int(actions[agent]),
                "action_profile": self.action_profiles[int(actions[agent])].copy(),
                "loads": loads[agent].copy(),
                "conflict_pressure": conflicts[agent].copy(),
                "travel_times": travel_times[agent].copy(),
                "movement_prior": self.schema_spec.movement_priors[agent].copy(),
            }
        return rewards, infos

    def _compute_loads(self, actions: dict[str, int]) -> dict[str, dict[str, float]]:
        loads = {}
        for agent in self.possible_agents:
            profile = self.action_profiles[int(actions[agent])]
            queue = float(self._queues[agent])
            loads[agent] = {
                movement: queue * float(profile[idx])
                for idx, movement in enumerate(MOVEMENTS)
            }
        return loads

    def _compute_conflicts(self, loads: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
        conflicts = {}
        for agent in self.possible_agents:
            opposite = self._opposite(agent)
            cw = self._cw(agent)
            ccw = self._ccw(agent)
            conflicts[agent] = {
                "left": (
                    1.0 * loads[opposite]["straight"]
                    + 0.7 * loads[opposite]["right"]
                    + 0.8 * (
                        loads[cw]["straight"]
                        + loads[cw]["left"]
                        + loads[ccw]["straight"]
                        + loads[ccw]["left"]
                    )
                ),
                "straight": (
                    1.0 * (loads[cw]["straight"] + loads[ccw]["straight"])
                    + 0.7 * (loads[cw]["left"] + loads[ccw]["left"])
                    + 0.6 * loads[opposite]["left"]
                ),
                "right": (
                    0.4 * loads[opposite]["left"]
                    + 0.3 * (loads[cw]["left"] + loads[ccw]["left"])
                ),
            }
        return conflicts

    def _compute_travel_times(
        self,
        loads: dict[str, dict[str, float]],
        conflicts: dict[str, dict[str, float]],
    ) -> dict[str, dict[str, float]]:
        times = {}
        caps = {
            "left": self.reward_model.cap_left,
            "straight": self.reward_model.cap_straight,
            "right": self.reward_model.cap_right,
        }
        base = {"left": 1.3, "straight": 1.0, "right": 1.0}
        congestion = {"left": 1.8, "straight": 1.2, "right": 1.2}
        conflict_scale = {"left": 0.035, "straight": 0.03, "right": 0.025}

        for agent in self.possible_agents:
            times[agent] = {}
            for movement in MOVEMENTS:
                flow = loads[agent][movement]
                times[agent][movement] = (
                    base[movement]
                    + congestion[movement] * (flow / caps[movement]) ** 2
                    + conflict_scale[movement] * conflicts[agent][movement]
                )
        return times

    def _opposite(self, direction: str) -> str:
        mapping = {"north": "south", "south": "north", "east": "west", "west": "east"}
        return mapping[direction]

    def _cw(self, direction: str) -> str:
        mapping = {"north": "east", "east": "south", "south": "west", "west": "north"}
        return mapping[direction]

    def _ccw(self, direction: str) -> str:
        mapping = {"north": "west", "west": "south", "south": "east", "east": "north"}
        return mapping[direction]
