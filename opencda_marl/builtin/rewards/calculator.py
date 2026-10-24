from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Mapping, MutableMapping

import numpy as np

from opencda_marl.api.events import StepEvent
from opencda_marl.api.contracts import (  # direct module; avoids core.marl.__init__ cycle
    REWARD_MODE_DISABLED,
    RewardSpec,
    build_reward_outputs,
    normalize_observations,
)
from opencda_marl.api.safety import (
    TtcMetricState,
    calculate_ttc_reward,
    track_ttc_metrics,
)


class RewardCalculator:
    def __init__(self, reward_params: Mapping[str, Any], reward_spec: RewardSpec):
        self.reward_params = dict(reward_params)
        self.reward_spec = reward_spec

    def calculate_carla(
        self,
        agent_ids: Iterable[Any],
        observations: Mapping[Any, Mapping[str, Any]] | None,
        events: Iterable[StepEvent],
        *,
        previous_observations: Mapping[Any, Mapping[str, Any]] | None,
        terminal_agents: set[Any],
        ttc_metrics: TtcMetricState,
    ) -> Dict[Any, Any]:
        normalized_observations = normalize_observations(observations or {})
        previous_observations = dict(previous_observations or {})
        reward_components: Dict[Any, Dict[str, float]] = {}

        for agent_id in agent_ids:
            if agent_id in terminal_agents:
                continue

            obs = normalized_observations.get(agent_id)
            prev_obs = previous_observations.get(agent_id)
            components = self._build_base_components()
            if obs is not None:
                self._apply_speed_and_stop_components(components, obs)
                self._apply_ttc_component(components, obs, agent_id, ttc_metrics)
                self._apply_clearance_component(components, obs)
                self._apply_yielding_component(components, obs, prev_obs)
                self._apply_progress_component(components, obs, prev_obs)
            reward_components[agent_id] = components

        self._apply_terminal_events(
            reward_components,
            events,
            normalized_observations,
            terminal_agents,
        )
        return build_reward_outputs(reward_components, self.reward_spec)

    def calculate_sumo(
        self,
        observations: Mapping[Any, Mapping[str, Any]] | None,
        events: Iterable[StepEvent],
        *,
        previous_observations: Mapping[Any, Mapping[str, Any]] | None,
        terminal_agents: set[Any],
        ttc_metrics: TtcMetricState,
    ) -> Dict[Any, Any]:
        normalized_observations = normalize_observations(observations or {})
        previous_observations = dict(previous_observations or {})
        reward_components: Dict[Any, Dict[str, float]] = {}

        for agent_id, obs in normalized_observations.items():
            if agent_id in terminal_agents:
                continue

            components = self._build_base_components()
            self._apply_speed_and_stop_components(components, obs)
            self._apply_ttc_component(components, obs, agent_id, ttc_metrics)
            self._apply_clearance_component(components, obs)
            self._apply_yielding_component(components, obs, previous_observations.get(agent_id))
            self._apply_progress_component(components, obs, previous_observations.get(agent_id))
            reward_components[agent_id] = components

        self._apply_terminal_events(
            reward_components,
            events,
            normalized_observations,
            terminal_agents,
        )
        return build_reward_outputs(reward_components, self.reward_spec)

    def _build_base_components(self) -> Dict[str, float]:
        components = self.reward_spec.empty_components()
        components["step_penalty"] = float(self.reward_params.get("step_penalty", 0.0))
        return components

    def _apply_speed_and_stop_components(
        self,
        components: MutableMapping[str, float],
        observation: Mapping[str, Any],
    ) -> None:
        speed_kmh = self.get_observed_speed_kmh(observation)
        speed_bonus = float(self.reward_params.get("speed_bonus", 0.0))
        speed_threshold = float(self.reward_params.get("speed_threshold", 40.0))
        stop_threshold = float(self.reward_params.get("stop_threshold", 5.0))
        stop_penalty = float(self.reward_params.get("stop_penalty", -3.0))

        if (
            self.reward_spec.mode_for("speed") != REWARD_MODE_DISABLED
            and speed_bonus > 0.0
            and speed_kmh > speed_threshold
        ):
            components["speed"] += speed_bonus

        if self.reward_spec.mode_for("stop") != REWARD_MODE_DISABLED and speed_kmh < stop_threshold:
            components["stop"] += stop_penalty

    def _apply_ttc_component(
        self,
        components: MutableMapping[str, float],
        observation: Mapping[str, Any],
        agent_id: Any,
        ttc_metrics: TtcMetricState,
    ) -> None:
        min_ttc = float(observation.get("min_ttc", float("inf")))
        if self.reward_spec.mode_for("ttc") == REWARD_MODE_DISABLED:
            track_ttc_metrics(
                self.reward_params,
                min_ttc,
                agent_id,
                ttc_metrics,
            )
            return
        max_penalty = float(self.reward_params.get("ttc_max_penalty", 0.0))
        if max_penalty <= 0.0:
            track_ttc_metrics(
                self.reward_params,
                min_ttc,
                agent_id,
                ttc_metrics,
            )
            return
        components["ttc"] += calculate_ttc_reward(
            self.reward_params,
            min_ttc,
            agent_id,
            ttc_metrics,
        )

    def _apply_clearance_component(
        self,
        components: MutableMapping[str, float],
        observation: Mapping[str, Any],
    ) -> None:
        if self.reward_spec.mode_for("clearance") == REWARD_MODE_DISABLED:
            return

        speed_kmh = self.get_observed_speed_kmh(observation)
        clearance_speed_bonus = float(self.reward_params.get("clearance_speed_bonus", 0.0))
        clearance_threshold = float(self.reward_params.get("clearance_threshold", 30.0))
        dist_to_front = float(observation.get("distance_to_front_vehicle", float("inf")))

        if clearance_speed_bonus <= 0.0:
            return
        if dist_to_front > clearance_threshold or math.isinf(dist_to_front):
            if speed_kmh > 20.0:
                speed_ratio = min(speed_kmh / 65.0, 1.0)
                components["clearance"] += clearance_speed_bonus * speed_ratio

    def _apply_yielding_component(
        self,
        components: MutableMapping[str, float],
        observation: Mapping[str, Any],
        previous_observation: Mapping[str, Any] | None,
    ) -> None:
        if self.reward_spec.mode_for("yielding") == REWARD_MODE_DISABLED:
            return
        if previous_observation is None:
            return

        yielding_bonus = float(self.reward_params.get("yielding_bonus", 0.0))
        if yielding_bonus <= 0.0:
            return

        min_ttc = float(observation.get("min_ttc", float("inf")))
        yielding_ttc_threshold = float(self.reward_params.get("yielding_ttc_threshold", 3.0))
        yielding_speed_drop = float(self.reward_params.get("yielding_speed_drop", 5.0))
        prev_speed = self.get_observed_speed_kmh(previous_observation)
        current_speed = self.get_observed_speed_kmh(observation)

        if min_ttc < yielding_ttc_threshold and not math.isinf(min_ttc):
            if prev_speed - current_speed >= yielding_speed_drop:
                components["yielding"] += yielding_bonus

    def _apply_progress_component(
        self,
        components: MutableMapping[str, float],
        observation: Mapping[str, Any],
        previous_observation: Mapping[str, Any] | None,
    ) -> None:
        if self.reward_spec.mode_for("progress") == REWARD_MODE_DISABLED:
            return
        if previous_observation is None:
            return
        components["progress"] += self.calculate_progress_reward(
            observation,
            previous_observation,
        )

    def _apply_terminal_events(
        self,
        reward_components: MutableMapping[Any, Dict[str, float]],
        events: Iterable[StepEvent],
        normalized_observations: Mapping[Any, Mapping[str, Any]],
        terminal_agents: set[Any],
    ) -> None:
        collision_reward = float(self.reward_params.get("collision", 0.0))
        success_reward = float(self.reward_params.get("success", 0.0))

        for event in events:
            agent_id = event.vehicle_id

            if event.event_type in {"physics_anomaly", "planner_failure"}:
                reward_components[agent_id] = self.reward_spec.empty_components()
                terminal_agents.add(agent_id)
                continue

            if agent_id not in reward_components:
                reward_components[agent_id] = self._build_base_components()
                obs = normalized_observations.get(agent_id)
                if obs is not None:
                    self._apply_speed_and_stop_components(reward_components[agent_id], obs)

            if event.event_type == "collision":
                reward_components[agent_id] = self.reward_spec.empty_components()
                reward_components[agent_id]["collision"] = collision_reward
                terminal_agents.add(agent_id)
            elif event.event_type == "success":
                reward_components[agent_id] = self.reward_spec.empty_components()
                reward_components[agent_id]["success"] = success_reward
                terminal_agents.add(agent_id)

    def calculate_progress_reward(
        self,
        current_observation: Mapping[str, Any],
        previous_observation: Mapping[str, Any],
    ) -> float:
        current_dist_to_int = float(current_observation.get("distance_to_intersection", 100.0))
        prev_dist_to_int = float(previous_observation.get("distance_to_intersection", 100.0))
        current_dist_to_dest = float(current_observation.get("distance_to_destination", float("inf")))
        prev_dist_to_dest = float(previous_observation.get("distance_to_destination", float("inf")))

        junction_threshold = float(self.reward_params.get("junction_threshold", 5.0))
        use_destination_distance = all(
            math.isfinite(value) and value < 900.0
            for value in (current_dist_to_dest, prev_dist_to_dest)
        )

        if use_destination_distance and current_dist_to_int < junction_threshold:
            progress = prev_dist_to_dest - current_dist_to_dest
        else:
            progress = prev_dist_to_int - current_dist_to_int

        progress_reward = float(np.clip(progress / 5.0, -1.0, 1.0))
        scale = float(self.reward_params.get("progress_scale", 0.5))
        return progress_reward * scale

    @staticmethod
    def get_observed_speed_kmh(observation: Mapping[str, Any]) -> float:
        return float(
            observation.get(
                "physics_speed_planar",
                observation.get("physics_speed", observation.get("speed", 0.0)),
            )
        )
