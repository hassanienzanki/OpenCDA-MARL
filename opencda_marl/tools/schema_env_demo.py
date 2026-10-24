"""Minimal demo runner for the standalone MOMARL intersection env."""

from __future__ import annotations

import argparse
import json

from opencda_marl.envs.traffic_schema_intersection import parallel_env


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a minimal standalone MOMARL traffic-schema demo.",
    )
    parser.add_argument(
        "--traffic-config-path",
        required=True,
        help="Path to an OpenCDA traffic config YAML file.",
    )
    parser.add_argument(
        "--recording-path",
        default=None,
        help="Optional path to an OpenCDA replay JSON file.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for env reset and sampled demo actions.",
    )
    return parser


def main():
    args = build_parser().parse_args()
    env = parallel_env(
        traffic_config_path=args.traffic_config_path,
        recording_path=args.recording_path,
    )
    observations, infos = env.reset(seed=args.seed)
    actions = {
        agent: env.action_space(agent).sample(env.rng)
        for agent in env.agents
    }
    next_obs, rewards, terminations, truncations, step_infos = env.step(
        actions)

    print("agents:", env.possible_agents)
    print("observation_shapes:", {agent: list(obs.shape)
          for agent, obs in observations.items()})
    print("reward_shapes:", {agent: list(reward.shape)
          for agent, reward in rewards.items()})
    print("actions:", actions)
    print("terminations:", terminations)
    print("truncations:", truncations)
    print("next_obs_keys:", list(next_obs.keys()))
    print("reset_infos:", json.dumps(
        {k: {"queue": v["queue"]} for k, v in infos.items()}, indent=2))
    print(
        "step_infos:",
        json.dumps(
            {
                k: {
                    "queue": v["queue"],
                    "action_index": v["action_index"],
                    "efficiency": float(rewards[k][0]),
                    "safety": float(rewards[k][1]),
                }
                for k, v in step_infos.items()
            },
            indent=2,
        ),
    )


if __name__ == "__main__":
    main()
