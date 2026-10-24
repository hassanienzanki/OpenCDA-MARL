import argparse
import gc
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from statistics import mean

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("LOGURU_LEVEL", "INFO")

from omegaconf import OmegaConf
from loguru import logger

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def configure_logging():
    logger.remove()
    logger.add(sys.stdout, level=os.environ.get("LOGURU_LEVEL", "INFO"))


def load_configuration(scenario_ref: str):
    from opencda_marl.utils import smart_merge, resolve_marl_config_paths

    resolved = resolve_marl_config_paths(REPO_ROOT, scenario_ref)
    merged = smart_merge(
        OmegaConf.load(resolved["default_yaml"]),
        OmegaConf.load(resolved["scenario_base_yaml"]),
    )
    return smart_merge(merged, OmegaConf.load(resolved["scenario_yaml"]))


def _slugify_part(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip()).strip("_").lower()
    return slug or "item"


def average_metrics(episodes, key):
    values = [float(episode.get(key, 0.0)) for episode in episodes]
    return mean(values) if values else 0.0


def average_objective_totals(episodes, component_name: str):
    values = [
        float((episode.get("objective_component_totals") or {}).get(component_name, 0.0))
        for episode in episodes
    ]
    return mean(values) if values else 0.0


def build_mean_objective_totals(episodes):
    objective_names = sorted({
        key
        for episode in episodes
        for key in (episode.get("objective_component_totals") or {}).keys()
    })
    return {
        objective_name: average_objective_totals(episodes, objective_name)
        for objective_name in objective_names
    }


def build_competition_window_summary(episodes):
    first_movers = Counter()
    yielding_totals = []
    yielding_per_success = []
    rank_by_direction = {}
    rank_by_group = {}

    for episode in episodes:
        competition = episode.get("competition_summary") or {}
        crossing_stats = competition.get("crossing_order_stats") or {}
        for direction, count in (crossing_stats.get("first_mover_by_entry_direction") or {}).items():
            first_movers[direction] += int(count)
        for direction, value in (crossing_stats.get("avg_completion_rank_by_entry_direction") or {}).items():
            rank_by_direction.setdefault(direction, []).append(float(value))
        for group, value in (crossing_stats.get("avg_completion_rank_by_preference_group") or {}).items():
            rank_by_group.setdefault(group, []).append(float(value))
        yielding = competition.get("yielding_summary") or {}
        if "yielding_total" in yielding:
            yielding_totals.append(float(yielding.get("yielding_total", 0.0)))
        if "yielding_per_success" in yielding:
            yielding_per_success.append(float(yielding.get("yielding_per_success", 0.0)))

    return {
        "first_mover_by_entry_direction": dict(first_movers),
        "mean_yielding_total": mean(yielding_totals) if yielding_totals else 0.0,
        "mean_yielding_per_success": mean(yielding_per_success) if yielding_per_success else 0.0,
        "avg_completion_rank_by_entry_direction": {
            direction: mean(values) for direction, values in rank_by_direction.items()
        },
        "avg_completion_rank_by_preference_group": {
            group: mean(values) for group, values in rank_by_group.items()
        },
    }


def effective_utility(episode_metrics):
    objective_spec = episode_metrics.get("objective_spec")
    if objective_spec:
        return float(episode_metrics.get("total_utility", 0.0)), "scalarized_utility"
    return float(episode_metrics.get("total_reward", 0.0)), "scalar_reward"


def build_window_summary(episodes):
    mean_objective_totals = build_mean_objective_totals(episodes)
    summary = {
        "mean_total_reward": average_metrics(episodes, "total_reward"),
        "mean_success_rate": average_metrics(episodes, "success_rate"),
        "mean_collision_rate": average_metrics(episodes, "collision_rate"),
        "mean_throughput": average_metrics(episodes, "throughput"),
        "mean_efficiency_total": mean_objective_totals.get("efficiency", 0.0),
        "mean_safety_total": mean_objective_totals.get("safety", 0.0),
        "mean_interaction_risk_total": mean_objective_totals.get("interaction_risk", 0.0),
        "mean_objective_totals": mean_objective_totals,
        "competition_summary": build_competition_window_summary(episodes),
    }
    utility_values = [effective_utility(episode)[0] for episode in episodes]
    summary["mean_utility"] = mean(utility_values) if utility_values else 0.0
    return summary


def build_summary(
    *,
    scenario,
    requested_episodes,
    seed,
    max_steps,
    checkpoint,
    episodes,
    status="completed",
    error=None,
):
    summary = {
        "scenario": scenario,
        "episodes": requested_episodes,
        "completed_episodes": len(episodes),
        "seed": seed,
        "max_steps": max_steps,
        "checkpoint": str(checkpoint),
        "status": status,
        "episode_metrics": episodes,
        "overall": build_window_summary(episodes),
        "first_window": build_window_summary(episodes[:min(10, len(episodes))]),
        "last_window": build_window_summary(episodes[-min(10, len(episodes)):]),
    }
    if error:
        summary["error"] = error
    return summary


def resolve_output_path(args, checkpoint_path: Path):
    return (
        Path(args.output)
        if args.output
        else (
            REPO_ROOT
            / "artifacts"
            / "eval_runs"
            / f"{_slugify_part(args.scenario)}_seed{args.seed}_{_slugify_part(checkpoint_path.stem)}_eval_ep{args.episodes}"
            / "eval_summary.json"
        )
    )


def write_summary(path: Path, summary: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def finalize_episode_metrics_only(coordinator):
    episode_metrics = coordinator.marl_env.reset_episode()
    gc.collect()
    return dict(episode_metrics)


def reset_coordinator_episode(coordinator):
    episode_metrics = finalize_episode_metrics_only(coordinator)

    if coordinator.scenario_manager is not None:
        coordinator.scenario_manager.reset_episode()

    for callback in coordinator.episode_callbacks:
        callback()

    if coordinator.world_manager:
        coordinator.world_manager.on_episode_end()

    return dict(episode_metrics)


def is_recoverable_carla_timeout(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(
        token in message
        for token in (
            "time-out of",
            "timed out",
            "simulator",
            "actively refused",
            "127.0.0.1:2000",
            "spawn_actor",
        )
    )


def recover_aborted_episode(coordinator, *, reason: str):
    logger.warning(f"Recovering aborted CARLA evaluation episode after {reason}.")
    episode_metrics = finalize_episode_metrics_only(coordinator)

    reloaded = False
    if coordinator.world_manager:
        try:
            reloaded = coordinator.world_manager.reload_world()
        except Exception as exc:
            logger.warning(f"World reload after aborted evaluation episode failed: {exc}")

    if coordinator.scenario_manager is not None:
        coordinator.scenario_manager.reset_episode()

    for callback in coordinator.episode_callbacks:
        callback()

    if coordinator.world_manager and not reloaded:
        coordinator.world_manager.on_episode_end()

    return dict(episode_metrics)


def detect_stalled_episode(
    coordinator,
    *,
    previous_observations: dict,
    speed_threshold_kmh: float,
    progress_threshold_m: float,
):
    metrics = coordinator.get_metrics() or {}
    active_agents = int(metrics.get("active_agents", 0) or 0)
    pending_spawns = int(metrics.get("pending_spawns", 0) or 0)
    current_step = int(metrics.get("step", 0) or 0)

    if active_agents <= 0 or pending_spawns > 0 or current_step < 200:
        return False, previous_observations, {}

    try:
        observations = coordinator.get_observations() or {}
    except Exception:
        return False, previous_observations, {}

    speeds = []
    progress_deltas = []
    for obs in observations.values():
        if not isinstance(obs, dict):
            continue
        speed = obs.get("physics_speed_planar")
        if speed is None:
            speed = obs.get("speed")
        if speed is None:
            continue
        try:
            speeds.append(abs(float(speed)))
        except Exception:
            continue

    for agent_id, obs in observations.items():
        if not isinstance(obs, dict):
            continue
        prev_obs = previous_observations.get(agent_id) or previous_observations.get(str(agent_id))
        if not isinstance(prev_obs, dict):
            continue

        current_dist = obs.get("distance_to_destination")
        prev_dist = prev_obs.get("distance_to_destination")
        if current_dist is None or prev_dist is None:
            current_dist = obs.get("distance_to_intersection")
            prev_dist = prev_obs.get("distance_to_intersection")
        if current_dist is None or prev_dist is None:
            continue
        try:
            progress_deltas.append(float(prev_dist) - float(current_dist))
        except Exception:
            continue

    diagnostics = {
        "max_speed_kmh": max(speeds) if speeds else None,
        "mean_progress_m": mean(progress_deltas) if progress_deltas else None,
        "active_agents": active_agents,
        "pending_spawns": pending_spawns,
    }
    if not speeds or not progress_deltas:
        return False, observations, diagnostics

    stalled = (
        max(speeds) <= speed_threshold_kmh
        and mean(progress_deltas) <= progress_threshold_m
    )
    return stalled, observations, diagnostics


def main():
    configure_logging()

    parser = argparse.ArgumentParser(description="Evaluate a saved CARLA checkpoint without further training.")
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=2400)
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--eval-traffic-mode", type=str, default="replay")
    parser.add_argument("--eval-replay-file", type=str, default="recordings/traffic_200vph.json")
    parser.add_argument("--stall-steps", type=int, default=400)
    parser.add_argument("--stall-speed-threshold", type=float, default=0.5)
    parser.add_argument("--stall-progress-threshold", type=float, default=0.05)
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    output_path = resolve_output_path(args, checkpoint_path)

    import torch  # Preload torch to stabilize the local Windows DLL load order.
    from opencda_marl import MARLCoordinator

    config = load_configuration(args.scenario)
    config["world"]["seed"] = args.seed
    config["scenario"]["simulation"]["max_steps"] = args.max_steps
    config["scenario"]["simulation"]["max_episodes"] = args.episodes
    traffic_cfg = config["scenario"].setdefault("traffic", {})
    traffic_cfg["mode"] = args.eval_traffic_mode
    if args.eval_replay_file:
        traffic_cfg["replay_file"] = args.eval_replay_file
    config["outputs"]["evaluation"]["mode"] = "disabled"
    config["outputs"]["tensorboard"]["enabled"] = False
    config["outputs"]["metrics_archive"]["enabled"] = False

    if "marl" in config.get("agents", {}):
        config["agents"]["marl"]["debug"] = False
        local_planner = config["agents"]["marl"].get("local_planner", {})
        local_planner["debug"] = False
        local_planner["debug_trajectory"] = False
        config["agents"]["marl"]["local_planner"] = local_planner

    training_cfg = config.get("MARL", {}).get("training", {})
    training_cfg["training_mode"] = False
    training_cfg["save_checkpoints"] = False
    training_cfg["load_checkpoint"] = str(checkpoint_path)
    config["MARL"]["training"] = training_cfg

    config["opt"] = {
        "test_scenario": args.scenario,
        "marl": True,
        "gui": False,
        "debug": False,
        "record": False,
        "apply_ml": False,
    }

    coordinator = MARLCoordinator(config)
    episodes = []
    last_error = None

    try:
        coordinator.initialize()
        for episode_idx in range(args.episodes):
            stalled_steps = 0
            previous_observations = {}
            aborted_episode_reason = None
            for _ in range(args.max_steps):
                try:
                    coordinator.step()
                except Exception as exc:
                    if is_recoverable_carla_timeout(exc):
                        aborted_episode_reason = f"{type(exc).__name__}: {exc}"
                        logger.warning(
                            "Treating CARLA eval step failure as timeout episode and attempting recovery: "
                            f"{aborted_episode_reason}"
                        )
                        break
                    raise
                is_stalled, previous_observations, diagnostics = detect_stalled_episode(
                    coordinator,
                    previous_observations=previous_observations,
                    speed_threshold_kmh=args.stall_speed_threshold,
                    progress_threshold_m=args.stall_progress_threshold,
                )
                if is_stalled:
                    stalled_steps += 1
                    if stalled_steps >= args.stall_steps:
                        logger.warning(
                            "Ending stalled CARLA evaluation episode early after "
                            f"{args.stall_steps} consecutive low-speed/no-progress steps "
                            f"(max_speed_kmh={diagnostics.get('max_speed_kmh')}, "
                            f"mean_progress_m={diagnostics.get('mean_progress_m')})."
                        )
                        break
                else:
                    stalled_steps = 0

            is_last_episode = episode_idx == args.episodes - 1
            if aborted_episode_reason:
                metrics = recover_aborted_episode(
                    coordinator,
                    reason=aborted_episode_reason,
                )
            elif is_last_episode:
                metrics = finalize_episode_metrics_only(coordinator)
            else:
                metrics = reset_coordinator_episode(coordinator)
            utility_value, utility_source = effective_utility(metrics)
            objective_totals = metrics.get("objective_component_totals") or {}
            episodes.append(metrics)
            print(json.dumps({
                "episode": episode_idx + 1,
                "utility": utility_value,
                "utility_source": utility_source,
                "total_reward": metrics.get("total_reward"),
                "success_rate": metrics.get("success_rate"),
                "collision_rate": metrics.get("collision_rate"),
                "throughput": metrics.get("throughput"),
                "efficiency_total": objective_totals.get("efficiency"),
                "safety_total": objective_totals.get("safety"),
                "interaction_risk_total": objective_totals.get("interaction_risk"),
                "objective_component_totals": objective_totals,
            }))
            write_summary(
                output_path,
                build_summary(
                    scenario=args.scenario,
                    requested_episodes=args.episodes,
                    seed=args.seed,
                    max_steps=args.max_steps,
                    checkpoint=checkpoint_path,
                    episodes=episodes,
                    status="in_progress" if len(episodes) < args.episodes else "completed",
                ),
            )
    except Exception as exc:
        last_error = f"{type(exc).__name__}: {exc}"
        write_summary(
            output_path,
            build_summary(
                scenario=args.scenario,
                requested_episodes=args.episodes,
                seed=args.seed,
                max_steps=args.max_steps,
                checkpoint=checkpoint_path,
                episodes=episodes,
                status="failed",
                error=last_error,
            ),
        )
        print(f"EVAL_SUMMARY_FILE={output_path}")
        raise
    finally:
        try:
            coordinator.close()
        except Exception as exc:
            logger.warning(f"Coordinator close failed after checkpoint eval: {exc}")

    summary = build_summary(
        scenario=args.scenario,
        requested_episodes=args.episodes,
        seed=args.seed,
        max_steps=args.max_steps,
        checkpoint=checkpoint_path,
        episodes=episodes,
        status="completed",
    )
    write_summary(output_path, summary)

    print(f"EVAL_SUMMARY_FILE={output_path}")


if __name__ == "__main__":
    main()
