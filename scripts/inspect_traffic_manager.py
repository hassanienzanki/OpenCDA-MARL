import argparse
import importlib.util
import math
from pathlib import Path
import sys
import time

from loguru import logger
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from opencda.core.common.cav_world import CavWorld
from opencda.core.plan.global_route_planner import GlobalRoutePlanner
from opencda.core.plan.global_route_planner_dao import GlobalRoutePlannerDAO
from opencda_marl._kernel.scenarios import ScenarioBuilder
from opencda_marl.utils import resolve_marl_config_paths, smart_merge


def _load_carla_spectator_class():
    module_path = REPO_ROOT / "opencda_marl" / "_kernel" / "sensing" / "carla_spectator.py"
    spec = importlib.util.spec_from_file_location("carla_spectator", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CarlaSpectator


CarlaSpectator = _load_carla_spectator_class()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inspect MARL traffic-manager spawn and destination contracts without running the scenario loop."
    )
    parser.add_argument(
        "--scenario",
        default="marl/intersection/vanilla",
        help="Scenario ref in <family>/<scenario>/<profile> form.",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=12,
        help="Maximum number of queued events to print.",
    )
    parser.add_argument(
        "--push-steps",
        type=int,
        default=0,
        help="Advance only the traffic-manager event queue for N logical steps without ticking the CARLA world.",
    )
    parser.add_argument(
        "--flush-ticks",
        type=int,
        default=0,
        help="Optional CARLA world ticks after drawing debug overlays. Leave at 0 for load-only inspection.",
    )
    parser.add_argument(
        "--playback",
        action="store_true",
        help="Review scheduled traffic events in chronological order without spawning vehicles.",
    )
    parser.add_argument(
        "--event-hold-seconds",
        type=float,
        default=3.0,
        help="How long to keep each scheduled event batch visible during playback.",
    )
    parser.add_argument(
        "--max-playback-batches",
        type=int,
        default=0,
        help="Maximum number of scheduled event batches to play back. 0 means all batches.",
    )
    parser.add_argument(
        "--route-playback",
        action="store_true",
        help="Review scheduled events using the actual global-planner route path instead of only a straight spawn-destination connection.",
    )
    parser.add_argument(
        "--route-arrow-step",
        type=int,
        default=3,
        help="Draw one route arrow for every N traced waypoints during route playback.",
    )
    parser.add_argument(
        "--route-arrow-thickness",
        type=float,
        default=0.35,
        help="Arrow thickness for planned-route playback.",
    )
    parser.add_argument(
        "--route-arrow-size",
        type=float,
        default=0.70,
        help="Arrow head size for planned-route playback.",
    )
    parser.add_argument(
        "--hold-open-seconds",
        type=float,
        default=0.0,
        help="Keep the inspector process alive for N wall-clock seconds after drawing, useful for load-only review.",
    )
    parser.add_argument(
        "--render-ticks",
        type=int,
        default=1,
        help="CARLA world ticks used only to apply spectator/debug rendering without running scenario steps.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args()


def load_marl_config(repo_root: Path, scenario_ref: str):
    resolved = resolve_marl_config_paths(repo_root, scenario_ref)
    merged = smart_merge(
        OmegaConf.load(resolved["default_yaml"]),
        OmegaConf.load(resolved["scenario_base_yaml"]),
    )
    merged = smart_merge(merged, OmegaConf.load(resolved["scenario_yaml"]))
    return merged


def _format_location(location) -> str:
    return f"({location.x:.2f}, {location.y:.2f}, {location.z:.2f})"


def _event_row(event) -> str:
    route_type = event.metadata.get("route_type", "unknown")
    entry_direction = event.metadata.get("entry_direction", "?")
    exit_direction = event.metadata.get("exit_direction", "?")
    return (
        f"step={event.spawn_step:4d} "
        f"id={event.vehicle_id:<18} "
        f"flow={event.flow_name:<12} "
        f"lane={event.lane_id} "
        f"route={event.route_id:<3d} "
        f"type={route_type:<8} "
        f"{entry_direction}->{exit_direction} "
        f"spawn={_format_location(event.transform.location)} "
        f"dest={_format_location(event.destination.location)} "
        f"speed={event.target_speed:.1f}"
    )


def print_event_summary(traffic_manager, max_events: int):
    events = traffic_manager.events
    logger.info(f"Queued traffic events: {len(events)}")
    for event in events[:max_events]:
        print(_event_row(event))


def push_traffic_steps(traffic_manager, steps: int):
    if steps <= 0:
        return

    logger.info(f"Pushing traffic-manager logical steps without CARLA world ticks: {steps}")
    for step in range(steps):
        spawn_events = traffic_manager.update(step)
        if not spawn_events:
            continue
        print(f"\n[traffic step {step}] {len(spawn_events)} event(s)")
        for event in spawn_events:
            print(_event_row(event))


def draw_event_batch(world, events, hold_seconds: float):
    import carla

    for event in events:
        spawn_loc = event.transform.location + carla.Location(z=0.6)
        dest_loc = event.destination.location + carla.Location(z=0.6)
        route_type = event.metadata.get("route_type", "unknown")
        label = f"{event.vehicle_id} r{event.route_id} {route_type}"

        world.debug.draw_point(
            spawn_loc,
            size=0.18,
            color=carla.Color(255, 0, 255),
            life_time=hold_seconds,
        )
        world.debug.draw_point(
            dest_loc,
            size=0.18,
            color=carla.Color(255, 255, 0),
            life_time=hold_seconds,
        )
        world.debug.draw_line(
            spawn_loc,
            dest_loc,
            thickness=0.12,
            color=carla.Color(0, 255, 0),
            life_time=hold_seconds,
        )
        world.debug.draw_string(
            spawn_loc + carla.Location(z=0.8),
            label,
            draw_shadow=False,
            color=carla.Color(255, 255, 255),
            life_time=hold_seconds,
        )


def build_global_route_planner(world, sampling_resolution: float = 2.0):
    dao = GlobalRoutePlannerDAO(world.get_map(), sampling_resolution=sampling_resolution)
    planner = GlobalRoutePlanner(dao, world, debug=False)
    planner.setup()
    return planner


def draw_planned_route_batch(
    world,
    planner,
    events,
    hold_seconds: float,
    arrow_step: int,
    arrow_thickness: float,
    arrow_size: float,
):
    import carla

    arrow_step = max(1, arrow_step)
    for event in events:
        route = planner.trace_route(event.transform.location, event.destination.location)
        if not route:
            continue

        route_type = event.metadata.get("route_type", "unknown")
        label = f"{event.vehicle_id} r{event.route_id} {route_type} wp={len(route)}"
        spawn_loc = event.transform.location + carla.Location(z=0.9)
        dest_loc = event.destination.location + carla.Location(z=0.9)

        world.debug.draw_point(
            spawn_loc,
            size=0.18,
            color=carla.Color(255, 128, 0),
            life_time=hold_seconds,
        )
        world.debug.draw_point(
            dest_loc,
            size=0.18,
            color=carla.Color(0, 255, 255),
            life_time=hold_seconds,
        )
        world.debug.draw_string(
            spawn_loc + carla.Location(z=0.8),
            label,
            draw_shadow=False,
            color=carla.Color(255, 255, 255),
            life_time=hold_seconds,
        )

        for index in range(0, max(0, len(route) - 1), arrow_step):
            start_wp = route[index][0]
            end_index = min(index + arrow_step, len(route) - 1)
            end_wp = route[end_index][0]
            start_loc = start_wp.transform.location + carla.Location(z=1.1)
            end_loc = end_wp.transform.location + carla.Location(z=1.1)

            world.debug.draw_arrow(
                start_loc,
                end_loc,
                thickness=arrow_thickness,
                arrow_size=arrow_size,
                color=carla.Color(255, 128, 0),
                life_time=hold_seconds,
            )


def playback_scheduled_events(world, traffic_manager, fixed_dt: float, hold_seconds: float, max_batches: int):
    grouped_events = {}
    for event in list(traffic_manager.events):
        grouped_events.setdefault(event.spawn_step, []).append(event)

    scheduled_steps = sorted(grouped_events.keys())
    if max_batches > 0:
        scheduled_steps = scheduled_steps[:max_batches]

    if not scheduled_steps:
        logger.warning("No scheduled events available for playback.")
        return

    hold_ticks = max(1, int(math.ceil(hold_seconds / fixed_dt)))
    logger.info(
        f"Playing back {len(scheduled_steps)} scheduled event batches "
        f"with hold={hold_seconds:.2f}s ({hold_ticks} ticks at dt={fixed_dt:.2f})."
    )

    for batch_index, spawn_step in enumerate(scheduled_steps, start=1):
        batch = grouped_events[spawn_step]
        print(f"\n[playback batch {batch_index}/{len(scheduled_steps)} | traffic step {spawn_step}] {len(batch)} event(s)")
        for event in batch:
            print(_event_row(event))

        draw_event_batch(world, batch, hold_seconds)
        for _ in range(hold_ticks):
            world.tick()
            time.sleep(fixed_dt)


def playback_planned_routes(
    world,
    traffic_manager,
    fixed_dt: float,
    hold_seconds: float,
    max_batches: int,
    arrow_step: int,
    arrow_thickness: float,
    arrow_size: float,
):
    grouped_events = {}
    for event in list(traffic_manager.events):
        grouped_events.setdefault(event.spawn_step, []).append(event)

    scheduled_steps = sorted(grouped_events.keys())
    if max_batches > 0:
        scheduled_steps = scheduled_steps[:max_batches]

    if not scheduled_steps:
        logger.warning("No scheduled events available for route playback.")
        return

    planner = build_global_route_planner(world)
    hold_ticks = max(1, int(math.ceil(hold_seconds / fixed_dt)))
    logger.info(
        f"Playing back {len(scheduled_steps)} planned-route batches "
        f"with hold={hold_seconds:.2f}s ({hold_ticks} ticks at dt={fixed_dt:.2f})."
    )

    for batch_index, spawn_step in enumerate(scheduled_steps, start=1):
        batch = grouped_events[spawn_step]
        print(f"\n[route batch {batch_index}/{len(scheduled_steps)} | traffic step {spawn_step}] {len(batch)} event(s)")
        for event in batch:
            print(_event_row(event))

        draw_planned_route_batch(
            world,
            planner,
            batch,
            hold_seconds,
            arrow_step,
            arrow_thickness,
            arrow_size,
        )
        for _ in range(hold_ticks):
            world.tick()
            time.sleep(fixed_dt)


def main():
    args = parse_args()
    logger.remove()
    logger.add(lambda msg: print(msg, end=""), level="DEBUG" if args.debug else "INFO")

    config = load_marl_config(REPO_ROOT, args.scenario)
    config["opt"] = {
        "apply_ml": False,
        "marl": True,
        "gui": False,
        "debug": args.debug,
        "test_scenario": args.scenario,
    }

    scenario_manager = None
    cav_world = None

    try:
        cav_world = CavWorld(apply_ml=False)
        scenario_manager = ScenarioBuilder.build_from_config(
            config=config,
            cav_world=cav_world,
        )

        world = scenario_manager.world
        traffic_manager = scenario_manager.traffic_manager
        traffic_cfg = config.get("scenario", {}).get("traffic", {})
        spectator_cfg = config.get("scenario", {}).get("spectator", {})

        if spectator_cfg:
            spectator = CarlaSpectator(world, {})
            preset_name = spectator_cfg.get("preset")
            if preset_name:
                spectator.use_preset(preset_name)

        logger.info(f"Scenario: {args.scenario}")
        logger.info(f"Map: {world.get_map().name}")
        logger.info(f"Traffic mode: {traffic_cfg.get('mode', 'live')}")
        logger.info(f"Fixed delta seconds: {scenario_manager.fixed_dt}")
        logger.info(f"Active junctions: {traffic_cfg.get('active_junctions', [])}")

        print_event_summary(traffic_manager, args.max_events)
        push_traffic_steps(traffic_manager, args.push_steps)
        if args.route_playback:
            playback_planned_routes(
                world,
                traffic_manager,
                scenario_manager.fixed_dt,
                args.event_hold_seconds,
                args.max_playback_batches,
                args.route_arrow_step,
                args.route_arrow_thickness,
                args.route_arrow_size,
            )
        elif args.playback:
            playback_scheduled_events(
                world,
                traffic_manager,
                scenario_manager.fixed_dt,
                args.event_hold_seconds,
                args.max_playback_batches,
            )
        elif args.render_ticks > 0:
            logger.info(f"Applying spectator/debug render with {args.render_ticks} world tick(s)")
            for _ in range(args.render_ticks):
                world.tick()

        if args.flush_ticks > 0:
            logger.info(f"Flushing CARLA world ticks for debug overlay visibility: {args.flush_ticks}")
            for _ in range(args.flush_ticks):
                world.tick()

        if args.hold_open_seconds > 0:
            logger.info(f"Holding inspector open for {args.hold_open_seconds:.1f}s")
            time.sleep(args.hold_open_seconds)

    finally:
        if scenario_manager is not None:
            try:
                scenario_manager.close()
            except Exception as exc:
                logger.warning(f"Scenario cleanup failed: {exc}")


if __name__ == "__main__":
    main()
