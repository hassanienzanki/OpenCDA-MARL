'''
Author       : AXIBA leolihao@arizona.edu
Date         : 2025-08-29 09:39:24
FilePath     : /OpenCDA-MARL/opencda_marl/traffic/traffic_manager.py
Description  : 
Copyright (c) 2025-2026 Lihao Guo and the Radar Lab, University of Arizona
SPDX-License-Identifier: Apache-2.0
'''
from __future__ import annotations
try:
    import carla
except ImportError:  # SUMO-only installs: the sumo_adapter shims stand in
    from opencda_marl._kernel.traffic import sumo_adapter as carla
from loguru import logger
from collections.abc import Mapping
from typing import List, Dict, Any
from pathlib import Path

from .events import SpawnEvent
from .flows import TrafficFlow
from .serializer import EventSerializer


def _resolve_planner(planner_cfg):
    """The route planner is a registered component (default: the built-in
    ``marl`` planner); ``planner.type`` in the traffic config selects it."""
    from opencda_marl.api.components import resolve_component

    return resolve_component("planner", str(planner_cfg.get("type", "marl")),
                             config=planner_cfg)


def resolve_record_root(outputs_cfg: Any = None) -> Path:
    """Where ``record`` mode writes: ``<outputs.root_dir>/<recordings.dir>``.

    A recording is produced by a run, so it lands with the run's other
    outputs (``artifacts/recordings/`` by default) instead of next to the
    packaged ground truth. Promoting one to a benchmark — copying it into
    ``opencda_marl/_bundled/recordings/`` — stays a deliberate act.
    """
    outputs = outputs_cfg if isinstance(outputs_cfg, Mapping) else {}
    root = str(outputs.get("root_dir") or "artifacts")
    section = outputs.get("recordings")
    sub = section.get("dir") if isinstance(section, Mapping) else None
    return Path(root) / str(sub or "recordings")


class MARLTrafficManager:
    flows: List[TrafficFlow]
    #: Class-level defaults so partially constructed instances (tests build
    #: one with ``object.__new__``) still answer the sampler lookups.
    preference_sampler = None
    population_sampler = None
    #: Set from the outputs config; None means the default record root.
    record_root = None

    def __init__(self, world: carla.World, config: dict, state: Dict[str, Any], fix_dlt: float = 0.05,
                 intent_cfg: dict = None, population_cfg: dict = None,
                 outputs_cfg: dict = None):
        self.world = world
        self.config = config
        self.state = state

        planner_cfg = self.config.get("planner", {}) or {}
        planner_cls = _resolve_planner(planner_cfg)
        self.planner = planner_cls(self.world, planner_cfg)
        self.active_junctions = self.config.get("active_junctions", [])
        self.vbp = self._get_vehicle_blueprints()
        # ---- Flow configuration ----
        self.base_speed = self.config.get('base_speed', 30.0)
        self.fix_dlt = fix_dlt
        self.max_active_agents = int(self.config.get("max_active_agents", 0) or 0)
        self.flows = self._parse_flows()

        # ---- Preference sampler wiring (per-vehicle / per-episode) ----
        # Wired per flow INSTANCE: the previous class-level injection meant a
        # second manager's construction (every world reload) reset or
        # replaced the sampler of every other live manager's flows.
        self.preference_sampler = None
        pref_cfg = (intent_cfg.get("preference") or {}) if isinstance(intent_cfg, Mapping) else {}
        if pref_cfg and pref_cfg.get("mode") in {"per_vehicle", "per_episode"}:
            from opencda_marl._kernel.traffic.preference_sampler import PreferenceSampler
            self.preference_sampler = PreferenceSampler(
                mode=str(pref_cfg.get("mode")),
                sampler=str(pref_cfg.get("sampler", "dirichlet")),
                dim=int(pref_cfg.get("dim", 0)),
                seed=int(pref_cfg.get("seed", 0)),
                alpha=float(pref_cfg.get("alpha", 1.0)),
                regime=str(pref_cfg.get("regime", "aligned")),
            )
        # ---- Population sampler (mixed autonomy: who is a CAV) ----
        # Same per-instance wiring as the preference sampler above.
        from opencda_marl._kernel.traffic.population import build_population_sampler

        self.population_sampler = build_population_sampler(population_cfg)
        if self.population_sampler is not None:
            logger.info(f"Mixed population enabled: "
                        f"{self.population_sampler.describe()}")

        for flow in self.flows:
            flow.preference_sampler = self.preference_sampler
            flow.population_sampler = self.population_sampler

        # ---- Events ----
        self.mode = config.get('mode', 'live')  # 'record', 'replay', 'live'
        # record mode writes the replay file once per run; later episodes
        # replay what was written instead of drawing (and overwriting) new
        # traffic each time. See _initialize_events.
        self._recording_saved = False
        self.replay_file = config.get('replay_file', 'traffic_events.h5')
        self.record_root = resolve_record_root(outputs_cfg)
        self._events: List[SpawnEvent] = []
        self._event_traces: List[List[SpawnEvent]] = []
        self._initialize_events()
        self._visualize_events()

        #print(f"DEBUG: events: {len(self._events)}")
        #for i, event in enumerate(self._events):
        #    print(
        #        f"DEBUG: event {i}: {event.spawn_step}, {event.target_speed}, {event.flow_name}, {event.lane_id}, {event.route_id}")
    # --------------------------------------------------------------------- #
    # Public control API
    # --------------------------------------------------------------------- #
    def update(self, current_step: int) -> List[SpawnEvent]:
        due_events = []
        # retrive the corresponding events that should spawn at the current step
        count = 0
        for event in self._events:
            if event.spawn_step <= current_step:
                due_events.append(event)
                count += 1
            # since we sort the events by spawn_step,
            # we can break the loop early
            if event.spawn_step > current_step:
                break

        #delete the events that have been processed
        self._events = self._events[count:]

        spawn_events = due_events
        if self.mode == "live" and self.max_active_agents > 0 and due_events:
            active_agents = int(self.state.get("active_agents", 0) or 0)
            available_slots = max(0, self.max_active_agents - active_agents)
            if available_slots < len(due_events):
                spawn_events = due_events[:available_slots]
                deferred_events = due_events[available_slots:]
                self._events = deferred_events + self._events
                logger.debug(
                    "Deferring {} traffic events due to active-agent backpressure "
                    "(active={}, limit={}, released={})",
                    len(deferred_events),
                    active_agents,
                    self.max_active_agents,
                    len(spawn_events),
                )

        self._event_traces.extend(spawn_events)
        return spawn_events
    
    @property
    def events(self) -> List[SpawnEvent]:
        return self._events
    
    @property
    def total_events(self) -> int:
        return len(self._events)
    # --------------------------------------------------------------------- #
    # Private API
    # --------------------------------------------------------------------- #

    def _initialize_events(self):
        """Initialize events for the episode about to start.

        ``record`` draws traffic and writes it to ``replay_file`` -- but only
        for the first episode of the run. Every later episode replays that
        file. Re-recording per episode meant the file on disk was the last
        draw made, and since the coordinator also resets after the final
        episode, that draw had never been simulated: a one-episode record
        run saved traffic nobody ran, and an N-episode run saved the
        (N+1)-th draw.
        """
        if self.mode == 'replay':
            self._load_events()
        elif self.mode == 'record':
            if self._recording_saved:
                self._load_events()
            else:
                self._generate_and_save_events()
        else:  # 'live'
            self._generate_events()

    def _visualize_events(self):
        """Draw generated traffic events for debugging spawn/destination selection."""
        vis_cfg = self.config.get('event_visualization', {})
        if not vis_cfg.get('enabled', False) or not self._events:
            return

        max_events = int(vis_cfg.get('max_events', 12))
        life_time = float(vis_cfg.get('life_time', 120.0))
        draw_spawn = vis_cfg.get('spawn_point', True)
        draw_destination = vis_cfg.get('destination_point', True)
        draw_route_line = vis_cfg.get('route_line', True)
        draw_text = vis_cfg.get('text', True)

        for event in self._events[:max_events]:
            spawn_loc = event.transform.location + carla.Location(z=0.6)
            dest_loc = event.destination.location + carla.Location(z=0.6)

            if draw_spawn:
                self.world.debug.draw_point(
                    spawn_loc,
                    size=0.18,
                    color=carla.Color(255, 0, 255),  # MAGENTA
                    life_time=life_time,
                )

            if draw_destination:
                self.world.debug.draw_point(
                    dest_loc,
                    size=0.18,
                    color=carla.Color(255, 255, 0),  # YELLOW
                    life_time=life_time,
                )

            if draw_route_line:
                self.world.debug.draw_line(
                    spawn_loc,
                    dest_loc,
                    thickness=0.12,
                    color=carla.Color(0, 255, 0),  # GREEN
                    life_time=life_time,
                )

            if draw_text:
                route_type = event.metadata.get('route_type', 'unknown')
                text = f"{event.vehicle_id} r{event.route_id} {route_type}"
                self.world.debug.draw_string(
                    spawn_loc + carla.Location(z=0.8),
                    text,
                    draw_shadow=False,
                    color=carla.Color(255, 255, 255),
                    life_time=life_time,
                )

    def _load_events(self):
        """Load events from replay file (supports both JSON and HDF5)."""

        replay_path = Path(self.replay_file)
        if not replay_path.is_absolute() and not replay_path.exists():
            # pip installs run outside a checkout: fall back to the replay
            # bundled into the engine wheel (basename match).
            try:
                from opencda_marl._bundled import bundled_recording

                packaged = bundled_recording(replay_path.name)
                if packaged is not None:
                    replay_path = packaged
            except Exception:  # pragma: no cover - the bundle is optional
                pass
        if not replay_path.is_absolute():
            replay_path = Path.cwd() / replay_path

        # Validate file first
        validation = EventSerializer.validate_event_file(str(replay_path))
        if not validation['valid']:
            raise FileNotFoundError(
                f"Cannot load replay file {replay_path}: {validation['error']}")

        logger.info(
            f"Loading {validation['format'].upper()} replay file: {replay_path}")
        logger.info(f"File info: {validation['total_events']} events, "
                    f"{validation['file_size_mb']:.1f} MB, version {validation['version']}")

        # Load events using auto-detection
        self._events = EventSerializer.load_events(str(replay_path), self.world)

        if self._events is None:
            raise RuntimeError(f"Failed to load events from {replay_path}")

        self._enrich_loaded_events()
        logger.success(f"Successfully loaded {len(self._events)} events")

    def _enrich_loaded_events(self) -> None:
        """Patch replay events with runtime-only metadata used by MARL.

        Older replay archives were recorded before per-vehicle preferences and
        ``entry_lane_idx`` were serialized. Training-generated live events
        already contain these fields; replay events must be enriched here so
        eval runs exercise the same policy inputs as training.
        """
        if not self._events:
            return
        per_flow_counts: Dict[str, int] = {}
        for global_idx, event in enumerate(self._events):
            metadata = dict(event.metadata or {})
            routes = self.planner.get_routes(int(event.junction_id)) or {}
            route = self._resolve_replay_route(event, metadata, routes)
            if route:
                metadata.setdefault("route_type", route.get("route_type", "unknown"))
                metadata.setdefault("entry_direction", route.get("entry_direction", ""))
                metadata.setdefault("exit_direction", route.get("exit_direction", ""))
                metadata["entry_lane_idx"] = route.get("entry_lane_idx", event.lane_id)
                metadata["exit_lane_idx"] = route.get("exit_lane_idx", -1)
                metadata["route_type"] = route.get("route_type", metadata.get("route_type", "unknown"))
                metadata["exit_direction"] = route.get("exit_direction", metadata.get("exit_direction", ""))
                event.transform = route["spawn_tf"]
                event.destination = route["dest_tf"]
                spawn_wp = route.get("spawn_wp")
                depart_lane_idx = getattr(spawn_wp, "lane_index", None)
                if depart_lane_idx is not None:
                    event.lane_id = int(depart_lane_idx)
                    metadata["lane_id"] = int(depart_lane_idx)
            metadata.setdefault("entry_lane_idx", metadata.get("lane_id", event.lane_id))
            metadata.setdefault("lane_id", event.lane_id)

            if self.population_sampler is not None and metadata.get("population") is None:
                # Replay archives recorded before mixed autonomy carry no
                # population; assign one so an old recording can still be
                # replayed at a configured penetration.
                metadata["population"] = self.population_sampler.assign()

            if self.preference_sampler is not None and metadata.get("preference") is None:
                flow_name = str(getattr(event, "flow_name", "") or "")
                vehicle_idx = per_flow_counts.get(flow_name, 0) if flow_name else global_idx
                pref = self.preference_sampler.sample(vehicle_idx=vehicle_idx)
                if getattr(pref, "size", 0) > 0:
                    metadata["preference"] = pref
                if flow_name:
                    per_flow_counts[flow_name] = vehicle_idx + 1

            event.metadata = metadata

    def _resolve_replay_route(
        self,
        event: SpawnEvent,
        metadata: Dict[str, Any],
        routes: Dict[int, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Resolve a replay event onto the current planner route table."""
        current_route = routes.get(int(event.route_id), {})
        entry_direction = str(
            metadata.get("entry_direction")
            or current_route.get("entry_direction", "")
            or ""
        )
        exit_direction = str(
            metadata.get("exit_direction")
            or current_route.get("exit_direction", "")
            or ""
        )
        lane_idx = int(metadata.get("entry_lane_idx", metadata.get("lane_id", event.lane_id)))
        route_type = str(metadata.get("route_type", "") or "")
        exit_lane_idx = metadata.get("exit_lane_idx")
        try:
            exit_lane_idx = int(exit_lane_idx)
        except (TypeError, ValueError):
            exit_lane_idx = None

        matches = []
        for route_id, route in routes.items():
            if str(route.get("entry_direction", "")) != entry_direction:
                continue
            if int(route.get("entry_lane_idx", -1)) != lane_idx:
                continue
            if exit_direction and str(route.get("exit_direction", "")) != exit_direction:
                continue
            route_exit_lane = int(route.get("exit_lane_idx", -1))
            lane_penalty = (
                abs(route_exit_lane - exit_lane_idx)
                if exit_lane_idx is not None and route_exit_lane >= 0
                else 0
            )
            route_type_penalty = 0 if str(route.get("route_type", "")) == route_type else 1
            matches.append((route_type_penalty, lane_penalty, int(route_id), route))

        if matches:
            matches.sort(key=lambda item: (item[0], item[1], item[2]))
            resolved_route_id, resolved_route = matches[0][2], matches[0][3]
            if int(event.route_id) != resolved_route_id:
                logger.debug(
                    "Replay route remapped for {}: {} -> {} "
                    "(entry={}, lane={}, exit={})",
                    event.vehicle_id,
                    event.route_id,
                    resolved_route_id,
                    entry_direction,
                    lane_idx,
                    exit_direction,
                )
                event.route_id = resolved_route_id
            return resolved_route

        return current_route

    def _record_destination(self) -> Path:
        """The file ``record`` mode writes for this run.

        A configured ``replay_file`` names *which* traffic to replay, so in
        record mode only its basename is meaningful: the file itself is run
        output and lands under the record root. An absolute path is honoured
        as given (an explicit destination is a deliberate one).
        """
        configured = Path(self.replay_file)
        if configured.is_absolute():
            return configured
        root = getattr(self, "record_root", None) or resolve_record_root()
        return (Path(root) / configured.name).resolve()

    def _generate_and_save_events(self):
        self._generate_events()

        if self._events:
            save_path = self._record_destination()
            save_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Saving {len(self._events)} events to {save_path}")

            # Prepare metadata
            metadata = {
                'flows': [flow.name for flow in self.flows],
                'vbps': [vbp.id for vbp in self.vbp],
                'total_steps': max((flow.end_step for flow in self.flows), default=0),
                'total_events': len(self._events)
            }

            # if save path ends with .h5, save as hdf5, otherwise save as json
            if save_path.suffix == '.h5':
                success = EventSerializer.save_events_to_hdf5(
                    self._events, str(save_path), self.config, metadata)
            else:
                success = EventSerializer.export_events_to_json(
                    self._events, str(save_path))

            if not success:
                logger.error(f"Failed to save events to {save_path}")
            else:
                self._recording_saved = True
                # Later episodes replay exactly what was just written, not a
                # same-named file that happens to sit somewhere else (the
                # packaged ground truth included).
                self.replay_file = str(save_path)
                logger.success(f"Events saved successfully to {save_path}")

    def _generate_events(self):
        logger.debug(f"Generating events for {len(self.active_junctions)} junctions and {len(self.flows)} flows")
        logger.debug(f"Active junctions: {self.active_junctions}")
        logger.debug(f"Flows: {[f.name for f in self.flows]}")

        for j_id in self.active_junctions:
            logger.debug(f"Processing junction {j_id}")
            for flow in self.flows:
                logger.debug(f"Generating events for flow '{flow.name}' at junction {j_id}")
                events = flow._generate_events(self.vbp, self.planner,
                                               j_id, fix_dlt=self.fix_dlt,
                                               base_speed=self.base_speed)
                logger.debug(f"Generated {len(events)} events for flow '{flow.name}'")
                self._events.extend(events)
        # sort events by spawn_step
        self._events.sort(key=lambda x: x.spawn_step)
        logger.debug(f"Total events generated: {len(self._events)}")

    def _parse_flows(self) -> List[Dict[str, Any]]:
        flow_dicts = self.config.get('flows', [])

        if not flow_dicts:
            logger.warning("No traffic flows configured")
            return []

        # Parse into TrafficFlowConfig objects
        flows = []
        for flow_dict in flow_dicts:
            try:
                flow = TrafficFlow(
                    name=flow_dict['name'],
                    rate_vph=float(flow_dict['rate_vph']),
                    lanes=flow_dict['lanes'],
                    start_step=int(flow_dict['start_step']),
                    end_step=int(flow_dict['end_step']),
                    direction=flow_dict['direction'],
                    speed_variation=float(
                        flow_dict.get('speed_variation', 0.0)),
                    middle_peak=flow_dict.get('middle_peak')
                )
                flows.append(flow)
            except Exception as e:
                raise ValueError(
                    f"Error parsing flow '{flow_dict.get('name', 'unknown')}': {e}")

        return flows

    def _get_vehicle_blueprints(self) -> List[carla.ActorBlueprint]:
        included_types = self.config.get('included_vehicle_types', [])
        excluded_types = self.config.get('excluded_vehicle_types', [])

        all_bps = self.world.get_blueprint_library().filter('vehicle.*')
        vehicle_bps = []

        if included_types:
            # Use ONLY included types (priority)
            for bp in all_bps:
                if any(included in bp.id.lower() for included in included_types):
                    vehicle_bps.append(bp)
        elif excluded_types:
            # Use all EXCEPT excluded types
            for bp in all_bps:
                if not any(excluded in bp.id.lower() for excluded in excluded_types):
                    vehicle_bps.append(bp)
        else:
            # Use all vehicles
            vehicle_bps = list(all_bps)

        logger.success(
            f"Selected {len(vehicle_bps)} vehicle blueprints from {len(all_bps)} total available")
        return vehicle_bps

    # --------------------------------------------------------------------- #
    # Reset and Cleanup
    # --------------------------------------------------------------------- #
    def reset(self):
        """Reset traffic manager for new episode."""
        logger.info("Resetting MARLTrafficManager for new episode")

        # Clear existing events and traces
        self._events.clear()
        self._event_traces.clear()

        # per_episode preference mode caches one vector per episode; without
        # this call every episode after the first reused episode 0's vector.
        if self.preference_sampler is not None:
            self.preference_sampler.reset_episode()
        if self.population_sampler is not None:
            self.population_sampler.reset_episode()

        # Re-initialize events based on mode
        self._initialize_events()

        logger.success(f"MARLTrafficManager reset completed with {len(self._events)} new events")
    
    def cleanup(self):
        if hasattr(self.planner, 'cleanup'):
            self.planner.cleanup()
        self.world = None
