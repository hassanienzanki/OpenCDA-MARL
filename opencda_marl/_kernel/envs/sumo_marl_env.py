'''
Author       : AXIBA leolihao@arizona.edu
Date         : 2025-11-16
FilePath     : /OpenCDA-MARL/opencda_marl/envs/sumo_marl_env.py
Description  : SUMO-only MARL environment for fast training
Copyright (c) 2025-2026 Lihao Guo and the Radar Lab, University of Arizona
SPDX-License-Identifier: Apache-2.0
'''
import math
import os
import socket
import sys
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple
from loguru import logger

# One consistent SUMO install (binary + tools) is resolved by
# opencda_marl._kernel.traffic._sumo_home before the bindings load — preferring
# the environment's eclipse-sumo package over a system SUMO_HOME so client
# and server versions cannot diverge. (Previously this module appended
# $SUMO_HOME/tools to sys.path and sys.exit()'d at import when unset.)
from opencda_marl._kernel.traffic._sumo_lib import traci  # libsumo by default; OPENCDA_SUMO_LIB=traci to fall back
from opencda_marl._kernel.traffic._sumo_home import sumo_binary
import sumolib  # pylint: disable=import-error

from opencda_marl.api.events import StepEvent
from opencda_marl.api.path_action import decision_is_admissible, decision_route_update
from opencda_marl.api.rewards import resolve_reward_configuration
from opencda_marl.api.safety import TtcMetricState
from opencda_marl.api.contracts import (
    normalize_observation,
    normalize_observations,
    reward_outputs_to_components,
    reward_outputs_to_scalars,
    reward_outputs_to_vectors,
)
from opencda_marl._kernel.policy.manager import MARLManager
from opencda_marl._kernel.policy.action_space import derive_route_intent_config
from opencda_marl._kernel.lifecycle.pipeline_builder import scenario_name_from_config
from opencda_marl._kernel.policy.competition import preference_group_from_values
from opencda_marl._kernel.envs.episode_state import SumoEpisodeState
from opencda_marl._kernel.envs.sumo_runtime import (
    build_sumo_episode_states,
    has_pending_sumo_traffic,
    should_end_sumo_episode,
)
from opencda_marl.evaluation.training_metrics import TrainingMetrics as Metrics
from opencda_marl._kernel.policy.checkpoint import CheckpointManager
from opencda_marl.api.objectives import (
    build_objective_outputs,
    objective_outputs_to_components,
    objective_outputs_to_vectors,
)
from opencda_marl._kernel.traffic.sumo_adapter import SumoWorld, SumoMARLPlanner
from opencda_marl._kernel.traffic.traffic_manager import MARLTrafficManager
from opencda_marl._kernel.traffic.population import is_learning_population, validate_population_mix
from opencda_marl.api.intent import build_provider_from_config
from opencda_marl._kernel.traffic.sumo_spawner import SumoVehicleSpawner
from opencda_marl._kernel.traffic.sumo_route_intent import (
    apply_route_decision_to_event,
    make_active_vehicle_proxy,
    make_spawn_event_proxy,
    spawn_lane_heading_rad,
)


def _find_free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _resolve_sumo_port(port_value: Any) -> int:
    if port_value in (None, "", 0, "0", "auto"):
        return _find_free_tcp_port()
    return int(port_value)


class SumoMARLEnv:
    """
    SUMO-only MARL environment for fast training.

    Uses TraCI to control SUMO vehicles directly without CARLA rendering.
    Provides same observation/action space as CARLA-based MARLEnv for transfer learning.
    """

    def __init__(self, config: Dict):
        """
        Initialize SUMO MARL environment.

        Args:
            config: Full configuration dictionary including:
                - runtime.sumo_cfg: Path to SUMO configuration file
                - world.fixed_delta_seconds: Simulation step length
                - scenario.simulation: Episode parameters
                - MARL: Algorithm configuration
                - rewards: Reward parameters
        """
        self.config = config

        # Extract configuration sections
        self.runtime_config = config.get('runtime', {})
        self.world_config = config.get('world', {})
        self.scenario_config = config.get('scenario', {})
        self.simulation_config = self.scenario_config.get('simulation', {})
        self.marl_config = config.get('MARL', {})
        self.training_config = self.marl_config.get('training', {})

        # SUMO connection parameters
        self.sumo_cfg = self.runtime_config.get('sumo_cfg')
        if self.sumo_cfg and not os.path.isabs(self.sumo_cfg) and not os.path.isfile(self.sumo_cfg):
            # Repo-relative paths (e.g. "opencda_marl/assets/...") resolve
            # against the installed package for pip installs run outside a
            # checkout; the assets ship inside the wheel.
            from pathlib import Path as _Path

            import opencda_marl as _pkg

            # Resolve against the installed package's parent (site-packages
            # or the repo root) — never by counting path segments, which a
            # package reorganisation silently breaks.
            packaged = _Path(_pkg.__file__).resolve().parent.parent / self.sumo_cfg
            if packaged.is_file():
                self.sumo_cfg = str(packaged)
        if not self.sumo_cfg or not os.path.isfile(self.sumo_cfg):
            raise ValueError(f"SUMO configuration file not found: {self.sumo_cfg}")

        self.step_length = self.world_config.get('fixed_delta_seconds', 0.05)
        self.sumo_port = _resolve_sumo_port(self.world_config.get('sumo_port', 8873))
        self.use_gui = self.world_config.get('sumo_gui', False)
        self.sumo_warnings = bool(self.world_config.get('sumo_warnings', True))
        self.sumo_step_log = bool(self.world_config.get('sumo_step_log', False))
        self.sumo_rl_runtime_mode = self.world_config.get('sumo_rl_runtime_mode', 'unshielded')
        self.sumo_rl_speed_mode = self.world_config.get('sumo_rl_speed_mode')
        self.sumo_rl_lane_change_mode = self.world_config.get('sumo_rl_lane_change_mode')
        # A trip ends within this distance of its destination point, as on CARLA,
        # not at the end of whatever SUMO edge contains the point.
        self.arrival_radius = float(self.world_config.get('arrival_radius_m', 5.0))

        # Episode parameters
        self.max_steps = self.simulation_config.get('max_steps', 2400)
        self.max_episodes = self.simulation_config.get('max_episodes', 1000)

        # State tracking
        self.current_step = 0
        self.current_episode = 0
        self.total_steps = 0

        # Per-episode vehicle/intent registries and caches, grouped so
        # reset() swaps one object instead of clearing fields one by one.
        self.episode = SumoEpisodeState()
        # Vehicles holding a TraCI speed command from the last applied step
        # (see _apply_actions).
        self._commanded_vehicles = set()
        self._sumo_lane_length_cache = {}
        self._sumo_edge_length_cache = {}

        # Intersection center (will be calculated from network)
        self.intersection_center = None

        # Traffic Manager (will be initialized after SUMO starts)
        self.traffic_manager = None
        self.vehicle_spawner = None
        self.use_traffic_manager = self.scenario_config.get('traffic', {}).get('mode') in ['live', 'record', 'replay']

        # MARL Manager. The manager reads the shared ``outputs`` block
        # (TensorBoard mode/frequency/dir, segment rotation, retention) from
        # its own config, so hand it over exactly as the CARLA pipeline
        # builder does. Without it the learner falls back to 'full' mode —
        # every scalar on every update — written into ./runs regardless of
        # what the scenario configured.
        algorithm = self.marl_config.get('algorithm')
        if algorithm is None:
            raise ValueError("MARL.algorithm is required for SUMO MARL agents")
        manager_config = dict(self.marl_config)
        if 'outputs' not in manager_config and 'outputs' in config:
            manager_config['outputs'] = config.get('outputs', {})
        # The scenario reference labels the run in the metrics store, as on
        # CARLA (PipelineBuilder.build_marl_manager).
        scenario_ref = scenario_name_from_config(config)
        if scenario_ref and 'scenario_name' not in manager_config:
            manager_config['scenario_name'] = scenario_ref
        self.marl_manager = MARLManager(manager_config, algorithm)
        self.intent_cfg = derive_route_intent_config(
            config.get('agents', {}),
            self.marl_config,
        )
        self.intent_provider = None
        self._sumo_intent_agent_manager = SimpleNamespace(_vehicle_adapters=[])
        self.marl_manager.agent_manager = self._sumo_intent_agent_manager

        # Reward parameters
        self.reward_config = resolve_reward_configuration(self.marl_config)
        self.reward_params = self.reward_config.reward_params
        self.reward_spec = self.reward_config.reward_spec
        self.objective_config = self.marl_manager.objective_config
        self.objective_spec = self.marl_manager.objective_spec
        self.reward_calculator = self._build_reward_calculator()
        self.ttc_metrics = TtcMetricState()

        # Episode tracking
        self.episode_events = []
        self.previous_observations = {}
        self.current_step_rewards = {}
        self.current_step_reward_components = {}
        self.current_step_reward_vectors = {}
        self.current_step_objective_components = {}
        self.current_step_objective_vectors = {}
        self.current_step_agent_preferences = {}
        from opencda_marl.api.objectives import competition_metrics_enabled
        self.competition_enabled = competition_metrics_enabled(self.reward_config, self.objective_config)
        self.competition_agent_metadata = {}
        self.competition_terminal_records = []
        self.last_episode_metrics = {}

        # Training metrics (export history every N episodes)
        outputs_cfg = config.get('outputs', {})
        archive_cfg = outputs_cfg.get('metrics_archive', {})
        archive_enabled = archive_cfg.get('enabled', True)
        metrics_export_interval = archive_cfg.get('export_interval', 10) if archive_enabled else 0
        archive_root_dir = outputs_cfg.get('root_dir')
        archive_dir = archive_cfg.get('dir', 'metrics_history')
        metrics_export_dir = os.path.join(archive_root_dir, archive_dir) if archive_root_dir else archive_dir
        # Mid-cell resume: offset pkl exporter when launcher injects
        # MARL.training.start_episode (see opencda_marl/_kernel/envs/marl_env.py).
        start_episode_counter = int(self.training_config.get('start_episode', 0))
        if start_episode_counter > 0:
            self.current_episode = start_episode_counter
        self.metrics = Metrics(
            export_interval=metrics_export_interval,
            export_dir=metrics_export_dir,
            start_episode_counter=start_episode_counter,
        )

        # Training mode
        self.is_training_mode = self.training_config.get('training_mode', True)

        # Checkpoint manager
        if self.is_training_mode:
            checkpoint_dir = self.training_config.get(
                'checkpoint_dir', f'checkpoints/sumo_{algorithm}')
            self.checkpoint_manager = CheckpointManager(
                checkpoint_dir, algorithm)
        else:
            self.checkpoint_manager = None

        # Load checkpoint if specified. Evaluation runs also need this path so
        # SUMO stress tests can replay fixed policies without online updates.
        load_checkpoint = self.training_config.get('load_checkpoint')
        if load_checkpoint:
            self._load_checkpoint_from_config(load_checkpoint)

        # Initialize SUMO
        self._start_sumo()

    def _build_reward_calculator(self):
        """The configured reward calculator, ``MARL.components.reward_calculator``.

        Selected from the MARL block, exactly as CARLA's MARLEnv selects it,
        so a calculator configured under ``MARL.components`` takes effect on
        both backends (the full run config has no top-level ``components``
        block).
        """
        from opencda_marl.api.components import component_key, resolve_component

        marl_config = self.marl_config or {}
        return resolve_component(
            "reward_calculator", component_key(marl_config, "reward_calculator"),
            config=marl_config,
        )(self.reward_params, self.reward_spec)

    def _start_sumo(self):
        """Start SUMO simulation via TraCI."""
        logger.info(f"Starting SUMO with config: {self.sumo_cfg}")

        # Choose SUMO binary from the resolved install (matches the client).
        binary = sumo_binary(gui=self.use_gui)

        # Start TraCI
        sumo_cmd = [
            binary,
            '--step-length', str(self.step_length),
            '--collision.action', 'warn',  # Keep collision events visible instead of teleporting RL vehicles away.
            '--collision.check-junctions', 'true',
            '--collision.mingap-factor', '1.0',  # Minimum gap for collision detection
            '--no-warnings', 'false' if self.sumo_warnings else 'true',
            '--quit-on-end', 'false',  # Don't auto-quit
        ]
        if not self.sumo_step_log:
            sumo_cmd.extend(['--no-step-log', 'true', '--duration-log.disable', 'true'])

        # If using traffic manager, only load network file (no routes)
        if self.use_traffic_manager:
            # Extract network file from sumocfg
            from opencda_marl._kernel.traffic.coordinate_bridge import CoordinateBridge

            net_file = self._get_network_file_from_config()
            self.coordinates = CoordinateBridge.from_net_file(net_file)
            self._calibrate_coordinates(net_file)
            sumo_cmd.extend(['--net-file', net_file])
            logger.info("Using MARLTrafficManager - loading network only (no route file)")
        else:
            # Use full config (includes routes)
            sumo_cmd.extend(['--configuration-file', self.sumo_cfg])
            logger.info("Using static route file from sumocfg")

        # Add auto-start flag for GUI to prevent manual start button
        if self.use_gui:
            sumo_cmd.append('--start')

        traci.start(sumo_cmd, port=self.sumo_port)

        # Calculate intersection center from network
        self._calculate_intersection_center()

        # Initialize traffic manager if using dynamic traffic
        if self.use_traffic_manager:
            self._initialize_traffic_manager()

        logger.info("SUMO started successfully")

    def _get_network_file_from_config(self) -> str:
        """Extract network file path from SUMO configuration file."""
        import xml.etree.ElementTree as ET
        from pathlib import Path

        # Parse sumocfg file
        tree = ET.parse(self.sumo_cfg)
        root = tree.getroot()

        # Find net-file element
        net_elem = root.find('.//net-file')
        if net_elem is None:
            raise ValueError(f"No net-file found in {self.sumo_cfg}")

        net_file = net_elem.get('value')

        # Resolve relative path
        cfg_dir = Path(self.sumo_cfg).parent
        net_path = cfg_dir / net_file

        return str(net_path)

    def _calibrate_coordinates(self, net_file) -> None:
        """Pick the frame convention from landmarks, then publish the bridge
        to the SUMO map shims.

        A CARLA town export mirrors CARLA's y axis; the bundled intersection
        does not, and the difference is invisible on a symmetric map until
        every position is wrong. Landmarks settle it: netconvert keeps
        CARLA's junction ids as node ids, so the convention that lands the
        network's nodes on the known CARLA-frame centres wins.

        The landmark file is ``runtime.coordinate_landmarks``, or else the
        first of ``<net>_landmarks.json`` and ``<net>_junction_graph.json``
        found next to the network; the second name is an alternative for
        the same landmark data. Either file may use either of two shapes,
        keyed by junction id: a flat ``{"12": [x, y]}`` or a nested
        ``{"junctions": {"12": {"centre": [x, y]}}}``.
        With no landmarks the bridge keeps its default (no flip), which is
        the bundled intersection's convention: a map that ships none is
        read without a y flip, so a CARLA town export needs landmarks to
        be read in the right frame.
        """
        from opencda_marl._kernel.traffic.sumo_adapter import set_coordinate_bridge

        try:
            samples, source = self._coordinate_landmarks(net_file)
            if samples:
                flip = self.coordinates.calibrate(samples)
                logger.info(f"coordinate bridge calibrated on {len(samples)} "
                            f"landmarks from {source}: y_flip={flip}")
        except Exception as exc:  # noqa: BLE001 - calibration is best effort
            logger.warning(f"coordinate bridge calibration skipped: {exc}")
        set_coordinate_bridge(self.coordinates)

    def _coordinate_landmarks(self, net_file):
        """``([(sumo_xy, carla_xy), ...], source_name)`` for the calibration."""
        import json
        import xml.etree.ElementTree as ET
        from pathlib import Path

        net_path = Path(net_file)
        stem = net_path.name.split(".")[0]
        configured = (self.runtime_config or {}).get("coordinate_landmarks")
        candidates = []
        if configured:
            path = Path(str(configured))
            candidates.append(path if path.is_absolute() else Path.cwd() / path)
        candidates.append(net_path.with_name(f"{stem}_landmarks.json"))
        candidates.append(net_path.with_name(f"{stem}_junction_graph.json"))
        source = next((c for c in candidates if c.is_file()), None)
        if source is None:
            if configured:
                logger.warning(
                    f"runtime.coordinate_landmarks={configured!r} not found; "
                    "the frame bridge keeps its default convention")
            return [], ""

        data = json.loads(source.read_text(encoding="utf-8"))
        rows = data.get("junctions") if isinstance(data.get("junctions"), dict) else data
        centres = {}
        for key, value in (rows or {}).items():
            centre = value.get("centre") if isinstance(value, dict) else value
            if isinstance(centre, (list, tuple)) and len(centre) >= 2:
                centres[str(key)] = (float(centre[0]), float(centre[1]))

        samples = []
        for node in ET.parse(net_path).getroot().iter("junction"):
            centre = centres.get(str(node.get("id")))
            if centre is None or node.get("x") is None or node.get("y") is None:
                continue
            samples.append(((float(node.get("x")), float(node.get("y"))), centre))
        return samples, source.name

    def _calculate_intersection_center(self):
        """Calculate the center of the intersection from SUMO network."""
        # Get all junctions
        junctions = traci.junction.getIDList()

        if not junctions:
            logger.warning("No junctions found in SUMO network")
            self.intersection_center = (0.0, 0.0)
            self.junction_centers = {}
            return

        # Every configured junction gets a centre, in CARLA coordinates via
        # the map's own netOffset. A whole-map setting activates several
        # junctions at once, and a vehicle must be measured against the
        # junction it is actually approaching -- not against one global point.
        active = [str(j) for j in (
            self.scenario_config.get("traffic", {}).get("active_junctions") or [])]
        wanted = [j for j in active if j in junctions] or [
            '4' if '4' in junctions else junctions[0]]
        self.junction_centers = {}
        for junction_id in wanted:
            pos_sumo = traci.junction.getPosition(junction_id)
            self.junction_centers[str(junction_id)] = self.coordinates.sumo_to_carla(
                pos_sumo[0], pos_sumo[1])
        # The historical single-centre field stays: it is the default frame
        # for scenarios that activate exactly one junction.
        self.intersection_center = next(iter(self.junction_centers.values()))

        logger.info(f"Junction centres (CARLA coords): {self.junction_centers}")
        logger.debug(f"Available junctions: {junctions}")

    def _initialize_traffic_manager(self):
        """Initialize MARLTrafficManager with SUMO adapter."""
        logger.info("Initializing MARL Traffic Manager for SUMO")

        try:
            # Create SUMO world adapter
            sumo_world = SumoWorld()
            logger.debug("SUMO world adapter created")

            # Get traffic configuration
            traffic_config = self.scenario_config.get('traffic', {})
            logger.debug(f"Traffic config: {traffic_config}")

            # Create state dict for traffic manager
            state = {
                'step': 0,
                'episode': 0
            }

            # A population mix must name the learning population, or the
            # learner gets no vehicle at all; every other name is background
            # traffic SUMO drives itself, so no registry check applies here.
            agents_cfg = self.config.get('agents', {}) or {}
            validate_population_mix(agents_cfg.get('population'),
                                    str(agents_cfg.get('agent_type', 'marl')))

            # Initialize traffic manager with SUMO world
            logger.debug("Creating MARLTrafficManager...")
            self.traffic_manager = MARLTrafficManager(
                world=sumo_world,
                config=traffic_config,
                state=state,
                fix_dlt=self.step_length,
                intent_cfg=self.intent_cfg,
                population_cfg=self.config.get('agents', {}).get('population'),
                outputs_cfg=self.config.get('outputs'),
            )
            logger.debug("MARLTrafficManager created")

            # Initialize vehicle spawner
            self.vehicle_spawner = SumoVehicleSpawner(
                runtime_mode=self.sumo_rl_runtime_mode,
                speed_mode=self.sumo_rl_speed_mode,
                lane_change_mode=self.sumo_rl_lane_change_mode,
                learning_agent_type=str(
                    self.config.get('agents', {}).get('agent_type', 'marl')),
                bridge=self.coordinates,
            )
            logger.debug("Vehicle spawner created")
            self._bind_route_intent_provider()

            logger.success(f"Traffic Manager initialized with {self.traffic_manager.total_events} spawn events")

        except Exception as e:
            logger.error(f"Failed to initialize traffic manager: {e}")
            import traceback
            traceback.print_exc()
            raise

    def _bind_route_intent_provider(self):
        """Bind the shared route-intent provider to the SUMO traffic loop."""
        planner = getattr(self.traffic_manager, "planner", None)
        if planner is None:
            self.intent_provider = None
            return
        self.intent_provider = build_provider_from_config(
            self.intent_cfg,
            planner,
            marl_manager=self.marl_manager,
            training=self.is_training_mode,
        )
        mode = (self.intent_cfg or {}).get("mode", "disabled")
        logger.info(f"SUMO route-intent mode: {mode}")

    def _build_active_route_adapters(self):
        """Build lightweight active-vehicle proxies for route selection."""
        if self.vehicle_spawner is None:
            return []
        adapters = []
        for veh_id in traci.vehicle.getIDList():
            if veh_id in self.episode.terminal_agents:
                continue
            try:
                adapters.append(self._active_route_proxy(
                    veh_id, self.vehicle_spawner.get_spawn_event(veh_id)))
            except traci.exceptions.TraCIException:
                continue
        return adapters

    def _active_route_proxy(self, veh_id, spawn_event):
        """One live vehicle as the route provider sees it, in the CARLA frame.

        A spawn-time ego is built from its spawn event, whose transform is in
        the CARLA frame, so every neighbour has to be in that frame too: the
        nearest-distance and time-to-collision features between a SUMO-frame
        neighbour and a CARLA-frame ego carry the whole netOffset, and the
        mirror on a y-flipped map. Raises ``TraCIException`` for a vehicle
        that has left the network.
        """
        pos = traci.vehicle.getPosition(veh_id)
        speed_kmh = float(traci.vehicle.getSpeed(veh_id)) * 3.6
        angle_deg = float(traci.vehicle.getAngle(veh_id))
        x, y = self.coordinates.sumo_to_carla(float(pos[0]), float(pos[1]))
        proxy = make_active_vehicle_proxy(
            vehicle_id=veh_id,
            spawn_event=spawn_event,
            position_xy=(x, y),
            speed_kmh=speed_kmh,
            angle_deg=angle_deg,
        )
        # The proxy builder turns the compass angle into a heading in SUMO's
        # own frame; the bridged frame mirrors it on a y-flipped map, so the
        # velocity is rebuilt from the bridged heading.
        heading = self.coordinates.heading_from_sumo_angle(angle_deg)
        speed_ms = speed_kmh / 3.6
        proxy.velocity = SimpleNamespace(
            x=speed_ms * math.cos(heading), y=speed_ms * math.sin(heading), z=0.0)
        return proxy

    # ------------------------------------------------------------------ #
    # Schematic traffic view (telemetry/traffic_view.py)
    # ------------------------------------------------------------------ #

    def collect_traffic_geometry(self):
        """Lane polylines for the schematic view, from the loaded network."""
        lanes = []
        try:
            for lane_id in traci.lane.getIDList():
                shape = traci.lane.getShape(lane_id)
                if len(shape) >= 2:
                    lanes.append([[float(x), float(y)] for x, y in shape])
        except Exception as exc:  # noqa: BLE001 - the view must never kill a run
            logger.debug(f"traffic view geometry unavailable: {exc}")
            return None
        junction = None
        try:
            jx, jy = traci.junction.getPosition(
                '4' if '4' in traci.junction.getIDList() else traci.junction.getIDList()[0])
            junction = [float(jx), float(jy)]
        except Exception:  # noqa: BLE001
            pass
        return {"lanes": lanes, "junction": junction, "y_up": True}

    def collect_traffic_view(self):
        """Vehicle poses for the schematic view: (vehicles, episode, step).

        ``role`` is ``agent`` only for a live vehicle of the learning
        population; background vehicles of a population mix, and vehicles
        that already finished, are ``traffic``.
        """
        import math as _math
        vehicles = []
        try:
            for veh_id in traci.vehicle.getIDList():
                pos = traci.vehicle.getPosition(veh_id)
                # SUMO angle: degrees, clockwise from north -> math radians
                yaw = _math.radians(90.0 - float(traci.vehicle.getAngle(veh_id)))
                learning = (veh_id not in self.episode.terminal_agents
                            and self._population_of(veh_id)[1])
                vehicles.append({
                    "id": str(veh_id),
                    "x": float(pos[0]), "y": float(pos[1]),
                    "yaw": yaw,
                    "speed": float(traci.vehicle.getSpeed(veh_id)),
                    "role": "agent" if learning else "traffic",
                })
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"traffic view state unavailable: {exc}")
        return vehicles, self.current_episode, self.current_step

    def _spawn_lane_heading(self, event):
        """Heading (CARLA frame) of the lane a SUMO-generated spawn event starts on, else None.

        ``traffic.mode`` live/record builds spawn transforms from lane
        waypoints whose yaw is always 0; a replayed recording carries the
        vehicle's own yaw, which is kept.
        """
        if getattr(self.traffic_manager, 'mode', 'replay') == 'replay':
            return None
        loc = event.transform.location
        try:
            sx, sy = self.coordinates.carla_to_sumo(float(loc.x), float(loc.y))
            edge_id, _pos, lane_index = traci.simulation.convertRoad(sx, sy)
            shape = traci.lane.getShape(f"{edge_id}_{lane_index}")
        except traci.exceptions.TraCIException:
            return None
        return spawn_lane_heading_rad(shape, (loc.x, loc.y), bridge=self.coordinates)

    def _apply_route_intent_to_spawn_events(self, events):
        """Apply route-intent decisions to pending SUMO spawn events."""
        if not events or self.intent_provider is None or self.traffic_manager is None:
            return events

        active_adapters = self._build_active_route_adapters()
        self._sumo_intent_agent_manager._vehicle_adapters = list(active_adapters)
        planner = getattr(self.traffic_manager, "planner", None)
        for event in events:
            vehicle_id = (
                self.vehicle_spawner.resolve_vehicle_id(event)
                if self.vehicle_spawner is not None
                else str(event.vehicle_id)
            )
            # The spawn point does not move when the route does, so one
            # lane heading serves both proxies below.
            heading = self._spawn_lane_heading(event)
            ego_adapter = make_spawn_event_proxy(event, actor_id=vehicle_id,
                                                 heading_rad=heading)
            agent_type = (getattr(self, "config", {}) or {}).get("agents", {}).get("agent_type", "marl")
            if not is_learning_population(getattr(event, "metadata", None), agent_type):
                # Keep background traffic in the policy's context, but never
                # let the policy choose or learn a route action for it.
                active_adapters.append(ego_adapter)
                continue
            if not self.decision_zone.decides_at_spawn:
                # Approach mode: the vehicle departs on its demand route and
                # decides once, later, inside the decision zone.
                self.decision_zone.register(vehicle_id)
                active_adapters.append(ego_adapter)
                continue
            self._sumo_intent_agent_manager._vehicle_adapters = list(active_adapters)
            decision = self.intent_provider({
                "event": event,
                "adapter": ego_adapter,
                "planner": planner,
                "active_adapters": list(active_adapters),
            })
            if decision is None:
                active_adapters.append(ego_adapter)
                continue
            changed = apply_route_decision_to_event(event, decision)
            ego_adapter = make_spawn_event_proxy(event, actor_id=vehicle_id,
                                                 heading_rad=heading)
            active_adapters.append(ego_adapter)
            # Log through the seam too: loguru evaluates its arguments eagerly,
            # so a reach into selected_candidate here raised for a provider
            # with its own decision type the moment a route changed.
            update = decision_route_update(decision) or {}
            selected_route = update.get("route_id")
            original_route = getattr(decision, "original_route_id", None)
            if not decision_is_admissible(decision):
                logger.warning(
                    "SUMO path decision failed its own invariant for {}: "
                    "original_route_id={}, selected_route_id={}; keeping the demand route.",
                    vehicle_id, original_route, selected_route,
                )
            elif changed:
                logger.debug("SUMO route-intent changed {}: {} -> {}",
                             vehicle_id, original_route, selected_route)
        self._sumo_intent_agent_manager._vehicle_adapters = list(active_adapters)
        return events

    def _committed_exit_lanes(self, veh_id, metadata):
        """Exit lanes other vehicles at this junction have already claimed."""
        from opencda_marl.api.route_intent import committed_exit_lane_occupancy

        spawner = self.vehicle_spawner
        if spawner is None:
            return []
        rows = self._exit_lane_rows(spawner)
        others = [row for other_id, row in rows.items() if other_id != veh_id]
        junction_id = metadata.get("junction_id")
        if junction_id is None:
            event = spawner.get_spawn_event(veh_id)
            junction_id = getattr(event, "junction_id", None)
        return committed_exit_lane_occupancy(others, junction_id=junction_id)

    def _exit_lane_rows(self, spawner) -> Dict[str, Dict]:
        """Every active vehicle's spawn metadata row, built once per step and
        vehicle set: a per-vehicle rebuild would be quadratic in the number
        of vehicles."""
        vehicle_ids = tuple(traci.vehicle.getIDList())
        key = (self.current_step, vehicle_ids)
        cached = getattr(self, "_exit_lane_rows_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        rows: Dict[str, Dict] = {}
        for other_id in vehicle_ids:
            event = spawner.get_spawn_event(other_id)
            if event is None:
                continue
            row = dict(getattr(event, "metadata", None) or {})
            row.setdefault("junction_id", getattr(event, "junction_id", None))
            rows[other_id] = row
        self._exit_lane_rows_cache = (key, rows)
        return rows

    def _junction_centre_for(self, veh_id, metadata=None):
        """The centre of the junction this vehicle belongs to (CARLA frame).

        Vehicles carry their junction in the spawn event, so a multi-junction
        scenario measures each vehicle against its own intersection.
        """
        centers = getattr(self, "junction_centers", None) or {}
        if not centers:
            return self.intersection_center or (0.0, 0.0)
        if metadata is None:
            spawner = self.vehicle_spawner
            event = spawner.get_spawn_event(veh_id) if spawner is not None else None
            metadata = dict(getattr(event, "metadata", None) or {})
            junction_id = str(getattr(event, "junction_id", "") or "")
        else:
            junction_id = str(metadata.get("junction_id", "") or "")
        if not junction_id:
            spawner = self.vehicle_spawner
            event = spawner.get_spawn_event(veh_id) if spawner is not None else None
            junction_id = str(getattr(event, "junction_id", "") or "")
        return centers.get(junction_id, self.intersection_center or (0.0, 0.0))

    @property
    def coordinates(self):
        """The SUMO<->CARLA frame bridge (built from the network file)."""
        bridge = self.__dict__.get("_coordinates")
        if bridge is None:
            from opencda_marl._kernel.traffic.coordinate_bridge import CoordinateBridge

            bridge = CoordinateBridge()
            self.__dict__["_coordinates"] = bridge
        return bridge

    @coordinates.setter
    def coordinates(self, bridge):
        self.__dict__["_coordinates"] = bridge

    @property
    def decision_zone(self):
        """The route-decision trigger, built on first use.

        Resolved lazily so a partially constructed env (tests build one with
        ``object.__new__``) still answers, and so the zone reads whatever
        ``intent_cfg`` the env ended up with.
        """
        zone = self.__dict__.get("_decision_zone")
        if zone is None:
            from opencda_marl._kernel.policy.decision_zone import build_decision_zone

            zone = build_decision_zone(getattr(self, "intent_cfg", None))
            self.__dict__["_decision_zone"] = zone
        return zone

    def _process_decision_zone(self):
        """Approach mode: decide for vehicles that reached the zone.

        The vehicle is already driving its demand route, so the outcome is
        applied by re-routing it in place (traci.vehicle.setRoute) rather
        than by editing a spawn event.
        """
        if self.decision_zone.decides_at_spawn or self.intent_provider is None:
            return 0
        if self.traffic_manager is None or self.vehicle_spawner is None:
            return 0
        planner = getattr(self.traffic_manager, "planner", None)
        if planner is None:
            return 0
        decided = 0
        active_adapters = self._build_active_route_adapters()
        self._sumo_intent_agent_manager._vehicle_adapters = list(active_adapters)
        for veh_id in list(self.decision_zone.pending):
            if veh_id in self.episode.terminal_agents:
                self.decision_zone.forget(veh_id)
                continue
            try:
                distance = self._distance_to_junction(veh_id)
            except Exception:  # noqa: BLE001 - a vanished vehicle is not an error
                self.decision_zone.forget(veh_id)
                continue
            if not self.decision_zone.due(veh_id, distance):
                continue
            event = self.vehicle_spawner.get_spawn_event(veh_id)
            if event is None:
                self.decision_zone.forget(veh_id)
                continue
            agent_type = (getattr(self, "config", {}) or {}).get("agents", {}).get("agent_type", "marl")
            if not is_learning_population(getattr(event, "metadata", None), agent_type):
                self.decision_zone.forget(veh_id)
                continue
            try:
                # Same frame as the active list it is scored against.
                ego_adapter = self._active_route_proxy(veh_id, event)
            except traci.exceptions.TraCIException:
                self.decision_zone.forget(veh_id)
                continue
            decision = self.intent_provider({
                "event": event,
                "adapter": ego_adapter,
                "planner": planner,
                "active_adapters": list(active_adapters),
            })
            self.decision_zone.mark_decided(veh_id)
            if decision is None:
                continue
            changed = apply_route_decision_to_event(event, decision)
            decided += 1
            if changed and decision_is_admissible(decision):
                self.vehicle_spawner.retarget_vehicle(veh_id, event)
        return decided

    def _distance_to_junction(self, veh_id) -> float:
        """Planar distance from a live SUMO vehicle to its own junction."""
        pos = traci.vehicle.getPosition(veh_id)
        carla_x, carla_y = self.coordinates.sumo_to_carla(pos[0], pos[1])
        centre = self._junction_centre_for(veh_id)
        return math.hypot(carla_x - centre[0], carla_y - centre[1])

    def _notify_route_intent_terminal(self, vehicle_id):
        """Finalize any learned one-shot route transition once per SUMO vehicle."""
        if vehicle_id in self.episode.terminal_intent_notified:
            return
        self.episode.terminal_intent_notified.add(vehicle_id)
        if self.marl_manager is None:
            return
        try:
            self.marl_manager.notify_vehicle_terminal(vehicle_id)
        except Exception as exc:
            logger.warning(f"notify_vehicle_terminal failed for SUMO vehicle {vehicle_id}: {exc}")

    def _queue_route_intent_terminal(self, vehicle_id):
        """Defer terminal finalization until after this step's reward update."""
        if vehicle_id in self.episode.terminal_intent_notified:
            return
        if vehicle_id not in self.episode.route_intent_terminal_queue:
            self.episode.route_intent_terminal_queue.append(vehicle_id)

    def _flush_route_intent_terminal_queue(self):
        """Flush deferred terminal route decisions in a stable order."""
        terminal_ids = list(self.episode.route_intent_terminal_queue)
        self.episode.route_intent_terminal_queue.clear()
        for vehicle_id in terminal_ids:
            self._notify_route_intent_terminal(vehicle_id)

    def _sync_traffic_manager_state(self):
        """Keep live-flow backpressure state aligned with the SUMO world."""
        if self.traffic_manager is None:
            return
        state = getattr(self.traffic_manager, "state", None)
        if not isinstance(state, dict):
            return
        try:
            active_agents = sum(
                1
                for vehicle_id in traci.vehicle.getIDList()
                if vehicle_id not in self.episode.terminal_agents
            )
        except Exception:
            active_agents = 0
        state["step"] = self.current_step
        state["episode"] = self.current_episode
        state["active_agents"] = int(active_agents)

    def step(self) -> Dict[str, Any]:
        """
        Execute one simulation step.

        Returns:
            Dict containing rewards and metrics
        """
        # Spawn vehicles from traffic manager
        if self.use_traffic_manager and self.traffic_manager:
            self._sync_traffic_manager_state()
            spawn_events = self.traffic_manager.update(self.current_step)
            if spawn_events:
                spawn_events = self._apply_route_intent_to_spawn_events(spawn_events)
                spawned_ids = self.vehicle_spawner.spawn_vehicles(spawn_events)
                self.episode.observation_cache = {}
                spawned = set(spawned_ids)
                for event in spawn_events:
                    vehicle_id = self.vehicle_spawner.resolve_vehicle_id(event)
                    if vehicle_id not in spawned:
                        self._notify_route_intent_terminal(vehicle_id)

        # Approach-trigger decisions: vehicles that reached the decision zone
        # decide now, on fresh observations, before this step's actions.
        self._process_decision_zone()

        # Get current observations: the step's own view of the last read
        observations = self._pending_observations()

        if not observations:
            # No active vehicles yet - step simulation. The step still counts in
            # the metrics tracker: step_length, total_simulation_steps and the
            # throughput denominator come from Metrics.current_length, and CARLA
            # calls update_step every step, so skipping it here made the two
            # backends' episode lengths mean different things.
            traci.simulationStep()
            self.episode.observation_cache = {}
            self.metrics.update_step({}, {})
            self.current_step += 1
            self.total_steps += 1
            done = self._is_episode_done({})
            return {
                'rewards': {},
                'reward_components': {},
                'reward_vectors': {},
                'reward_spec': self.reward_spec.to_dict(),
                'objective_components': {},
                'objective_vectors': {},
                'objective_spec': self.objective_spec.to_dict() if self.objective_spec else None,
                'agent_preferences': {},
                'done': done,
            }

        # Compute actions from MARL manager
        target_speeds = self.marl_manager.compute_actions(
            observations, training=self.is_training_mode)
        pre_step_preferences = self.marl_manager.get_agent_preferences(observations.keys())
        self._update_competition_metadata(observations, pre_step_preferences)

        # Apply actions to SUMO vehicles
        self._apply_actions(target_speeds)

        # Step SUMO simulation
        traci.simulationStep()

        # Check for events (collisions, arrivals)
        events = self._check_events()

        # Observe the post-action state so SUMO matches the CARLA transition
        # contract: (S_t, a_t, r_t, S_t+1).
        next_observations = self._get_observations()
        # The post-step read is the next step's pre-step observation, the
        # very same object: the manager memoises state extraction on dict
        # identity, so handing it a copy would make it extract the same
        # features again (three extractions per step). External readers still
        # get a copy (_get_current_observations).
        self.episode.observation_cache = next_observations

        # Calculate rewards for the current transition (state -> action -> next_state)
        reward_outputs = self._calculate_rewards(events, next_observations)
        rewards = reward_outputs_to_scalars(reward_outputs)
        objective_outputs = self._build_objective_outputs(reward_outputs)
        self.marl_manager.ensure_agent_preferences(next_observations)
        agent_preferences = self.marl_manager.get_agent_preferences(next_observations.keys())
        self._update_competition_metadata(next_observations, agent_preferences)
        self._record_competition_events(events)
        learning_rewards = self.marl_manager.scalarize_objective_outputs(objective_outputs)

        # Update MARL algorithm with CURRENT step's transition
        # Transition: (observations, action, reward, next_observations)
        # - observations: state before action (S_t)
        # - action: stored in last_actions during compute_actions
        # - reward: calculated for taking action from observations
        # - next_observations: state after action (S_t+1)
        if self.is_training_mode:
            self.marl_manager.update(
                rewards=reward_outputs,
                observations=observations,
                next_observations=next_observations,
                objective_outputs=objective_outputs,
            )
        self._flush_route_intent_terminal_queue()

        # Store post-step observations for the next progress/yielding/TTC delta.
        self.previous_observations = normalize_observations(next_observations)
        # This step's events and raw observations (population, learning and
        # the route decision survive only before normalisation), for the
        # trace writer — parity with MARLEnv.events and the CARLA pass.
        self.events = events
        self.trace_observations = next_observations
        self.current_step_rewards = rewards
        self.current_step_reward_components = reward_outputs_to_components(reward_outputs)
        self.current_step_reward_vectors = reward_outputs_to_vectors(reward_outputs)
        self.current_step_objective_components = objective_outputs_to_components(objective_outputs)
        self.current_step_objective_vectors = objective_outputs_to_vectors(objective_outputs)
        self.current_step_agent_preferences = agent_preferences

        # Update step counters
        self.current_step += 1
        self.total_steps += 1

        # Update metrics with post-step observations for parity with CARLA.
        self._update_metrics(
            rewards,
            events,
            next_observations,
            reward_outputs=reward_outputs,
            objective_outputs=objective_outputs,
            scalarized_utilities=learning_rewards,
            agent_preferences=agent_preferences,
        )

        # Check if episode is done
        done = self._is_episode_done(next_observations)

        return {
            'rewards': rewards,
            'reward_components': self.current_step_reward_components.copy(),
            'reward_vectors': self.current_step_reward_vectors.copy(),
            'reward_spec': self.reward_spec.to_dict(),
            'objective_components': self.current_step_objective_components.copy(),
            'objective_vectors': self.current_step_objective_vectors.copy(),
            'objective_spec': self.objective_spec.to_dict() if self.objective_spec else None,
            'agent_preferences': dict(self.current_step_agent_preferences),
            'events': events,
            'done': done,
            'observations': next_observations
        }

    def _get_current_observations(self) -> Dict[int, Dict]:
        """Return the current SUMO state, reusing the previous post-step read when valid.

        Callers outside the step get a copy: mutating it never reaches the
        cached read."""
        if self.episode.observation_cache:
            return self._copy_observations(self.episode.observation_cache)
        return self._get_observations()

    def _pending_observations(self) -> Dict[int, Dict]:
        """The step's own view: the previous post-step read itself when
        still valid (the same dict object the learner already extracted
        features from), else a fresh read."""
        if self.episode.observation_cache:
            return self.episode.observation_cache
        return self._get_observations()

    @staticmethod
    def _copy_observations(observations: Dict[int, Dict]) -> Dict[int, Dict]:
        return {agent_id: dict(observation) for agent_id, observation in observations.items()}

    def _population_of(self, veh_id) -> tuple:
        """(population, learning) for one SUMO vehicle.

        Mixed autonomy on SUMO means the learner simply does not command
        background vehicles: they stay on SUMO's own car-following model
        because they never appear in the target-speed dict.
        """
        # Resolved once per vehicle per episode: the per-step re-derivation
        # -- a config read through the resolver plus a spawn event lookup
        # for every vehicle -- would repeat work for an answer that never
        # changes.
        cache = self.episode.populations
        hit = cache.get(veh_id)
        if hit is not None:
            return hit
        agent_type = str(self.config.get('agents', {}).get('agent_type', 'marl'))
        metadata = {}
        event = None
        spawner = self.vehicle_spawner
        if spawner is not None:
            event = spawner.get_spawn_event(veh_id)
            metadata = dict(getattr(event, 'metadata', None) or {})
        population = str(metadata.get('population') or agent_type)
        result = (population, population == agent_type)
        if spawner is None or event is not None:
            # A vehicle seen before its spawn event is registered keeps the
            # default for now and is asked again next step.
            cache[veh_id] = result
        return result

    def _get_observations(self) -> Dict[int, Dict]:
        """
        Extract observations from SUMO.

        Returns same format as CARLA MARLEnv for compatibility.
        """
        observations = {}

        # Get all vehicle IDs in simulation
        vehicle_ids = traci.vehicle.getIDList()
        # Every vehicle's bridged position, heading and velocity, once per
        # step: the neighbour block and the time-to-collision come from the
        # same code CARLA uses, on the same kind of snapshot.
        snapshot = self._kinematics_snapshot(vehicle_ids)

        for veh_id in vehicle_ids:
            # Skip if vehicle has terminated
            if veh_id in self.episode.terminal_agents:
                continue

            try:
                # Get vehicle state from SUMO
                pos = traci.vehicle.getPosition(veh_id)
                speed_ms = traci.vehicle.getSpeed(veh_id)
                speed_kmh = speed_ms * 3.6  # Convert m/s to km/h
                angle_deg = traci.vehicle.getAngle(veh_id)
                # The heading in the bridged (CARLA) frame, converted from
                # SUMO's compass angle.
                angle_rad = self.coordinates.heading_from_sumo_angle(angle_deg)

                # Get leader vehicle
                leader_info = traci.vehicle.getLeader(veh_id, 100.0)  # Look ahead 100m
                if leader_info is not None:
                    distance_to_front_vehicle = leader_info[1]
                else:
                    distance_to_front_vehicle = 999.0  # No vehicle ahead
                nearby_features, min_ttc = self._nearby_from_snapshot(snapshot, veh_id)

                # Relative position to *this vehicle's* junction. Both
                # sides must be in the same frame: the vehicle position is
                # SUMO's, the centres are CARLA's, so convert the position
                # to the CARLA frame first.
                carla_x, carla_y = self.coordinates.sumo_to_carla(pos[0], pos[1])
                centre = self._junction_centre_for(veh_id)
                rel_x = carla_x - centre[0]
                rel_y = carla_y - centre[1]

                # Calculate distance to intersection
                distance_to_intersection = math.sqrt(rel_x**2 + rel_y**2)

                # Get lane position (simplified)
                # 0 = at intersection, 1 = left lane, 2 = middle, 3 = right lane
                lane_id = traci.vehicle.getLaneID(veh_id)
                lane_index = traci.vehicle.getLaneIndex(veh_id)
                lane_position = lane_index if distance_to_intersection < 10 else lane_index + 1
                distance_to_destination = self._estimate_distance_to_destination(veh_id, lane_id)
                velocity_x = speed_ms * math.cos(angle_rad)
                velocity_y = speed_ms * math.sin(angle_rad)
                spawn_event = self.vehicle_spawner.get_spawn_event(veh_id) if self.vehicle_spawner else None
                metadata = dict(spawn_event.metadata or {}) if spawn_event is not None else {}

                # Build observation dictionary (matching CARLA format)
                observations[veh_id] = normalize_observation({
                    'vehicle_id': veh_id,
                    # One frame for everything the learner and the trace see:
                    # the bridged one relative_position_to_intersection uses.
                    'location': {'x': carla_x, 'y': carla_y, 'z': 0.0},
                    'speed': speed_kmh,
                    'physics_speed': speed_kmh,
                    'physics_speed_planar': speed_kmh,
                    'physics_speed_3d': speed_kmh,
                    'velocity': {'x': velocity_x, 'y': velocity_y, 'z': 0.0},
                    'heading_angle': angle_rad,
                    'relative_position_to_intersection': {'x': rel_x, 'y': rel_y},
                    'distance_to_intersection': distance_to_intersection,
                    'distance_to_front_vehicle': distance_to_front_vehicle,
                    'distance_to_destination': distance_to_destination,
                    'lane_position': lane_position,
                    'min_ttc': min_ttc,
                    'nearby_vehicles': nearby_features,
                })
                observations[veh_id]['entry_direction'] = metadata.get('entry_direction')
                observations[veh_id]['exit_direction'] = metadata.get('exit_direction')
                observations[veh_id]['route_type'] = metadata.get('route_type')
                observations[veh_id]['route_id'] = metadata.get('route_id')
                observations[veh_id]['lane_id'] = metadata.get('lane_id', lane_index)
                observations[veh_id]['exit_lane_idx'] = metadata.get('exit_lane_idx')
                observations[veh_id]['preference'] = metadata.get('preference')
                observations[veh_id]['route_intent'] = metadata.get('route_intent')
                population, learning = self._population_of(veh_id)
                observations[veh_id]['population'] = population
                observations[veh_id]['learning'] = learning
                observations[veh_id]['committed_exit_lanes'] = (
                    self._committed_exit_lanes(veh_id, metadata))

                # Track active vehicle
                if veh_id not in self.episode.active_vehicles:
                    self.episode.active_vehicles[veh_id] = {
                        'spawn_step': self.current_step,
                        'collisions': 0
                    }

            except traci.exceptions.TraCIException as e:
                logger.warning(f"Failed to get observation for vehicle {veh_id}: {e}")
                continue

        return observations

    def _get_lane_length(self, lane_id: str) -> float:
        cached = self._sumo_lane_length_cache.get(lane_id)
        if cached is not None:
            return cached
        length = float(traci.lane.getLength(lane_id))
        self._sumo_lane_length_cache[lane_id] = length
        return length

    def _nearby_settings(self) -> Dict[str, float]:
        """The neighbour block's radius and clamps, read where the CARLA
        adapter reads them, with the same defaults -- resolved once per
        config object rather than per vehicle per step (each read goes
        through the config resolver)."""
        marl = self.marl_config or {}
        cached = getattr(self, "_nearby_settings_cache", None)
        if cached is not None and cached[0] is marl:
            return cached[1]
        settings = self._read_nearby_settings(marl)
        self._nearby_settings_cache = (marl, settings)
        return settings

    @staticmethod
    def _read_nearby_settings(marl) -> Dict[str, float]:
        from opencda_marl._kernel.adapter._nearby.nearby import (
            DEFAULT_MAX_RELATIVE_VELOCITY, DEFAULT_MAX_TTC, DEFAULT_NEARBY_DETECTION_RADIUS)
        from opencda_marl.api.contracts import DEFAULT_MAX_NEARBY_VEHICLES

        block: Dict[str, Any] = {}
        for key in (marl.get("algorithm"), "td3"):
            candidate = (marl.get(key) or {}).get("nearby_vehicle_config") if key else None
            if candidate:
                block = dict(candidate)
                break
        return {
            "detection_radius": float(block.get("detection_radius", DEFAULT_NEARBY_DETECTION_RADIUS)),
            "max_vehicles": int(block.get("max_vehicles", DEFAULT_MAX_NEARBY_VEHICLES)),
            "max_relative_velocity": float(block.get("max_relative_velocity", DEFAULT_MAX_RELATIVE_VELOCITY)),
            "max_ttc": float(block.get("max_ttc", DEFAULT_MAX_TTC)),
        }

    def _kinematics_snapshot(self, vehicle_ids):
        """Every vehicle's bridged position, heading and velocity, as the
        world snapshot CARLA's adapter builds -- so neighbour features and
        the time-to-collision are the same computation on both backends.
        Rows carry surrogate integer ids; ``row_of`` maps SUMO ids to them."""
        from opencda_marl._kernel.adapter.world_snapshot import VehicleKinematics, WorldSnapshot

        records = []
        row_of: Dict[str, int] = {}
        for vid in vehicle_ids:
            try:
                pos = traci.vehicle.getPosition(vid)
                speed = float(traci.vehicle.getSpeed(vid))
                angle = float(traci.vehicle.getAngle(vid))
            except traci.exceptions.TraCIException:
                continue
            x, y = self.coordinates.sumo_to_carla(float(pos[0]), float(pos[1]))
            yaw = self.coordinates.heading_from_sumo_angle(angle)
            row_of[vid] = len(records)
            records.append(VehicleKinematics(
                actor_id=len(records), x=x, y=y, z=0.0,
                vx=speed * math.cos(yaw), vy=speed * math.sin(yaw), yaw_rad=yaw))
        snapshot = WorldSnapshot(records)
        snapshot.row_of = row_of
        return snapshot

    def _nearby_from_snapshot(self, snapshot, veh_id) -> Tuple[List[float], float]:
        """The k-slot neighbour block and the geometric min TTC for one vehicle."""
        settings = self._nearby_settings()
        row = getattr(snapshot, "row_of", {}).get(veh_id)
        if row is None:
            return [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0] * settings["max_vehicles"], float("inf")
        return snapshot.neighbour_features(
            row, radius=settings["detection_radius"], k=settings["max_vehicles"],
            max_relative_velocity=settings["max_relative_velocity"],
            max_ttc=settings["max_ttc"], features_per_vehicle=7)

    def _estimate_distance_to_destination(self, veh_id: str, lane_id: str) -> float:
        """
        Approximate remaining route distance for progress parity with CARLA.

        SUMO exposes route edges rather than a direct destination metric, so we
        accumulate remaining lane lengths from the current route index onward.
        """
        try:
            route_edges = list(traci.vehicle.getRoute(veh_id))
            route_index = int(traci.vehicle.getRouteIndex(veh_id))
            lane_position = float(traci.vehicle.getLanePosition(veh_id))
        except traci.exceptions.TraCIException:
            return 999.0

        if not route_edges or route_index < 0:
            return 999.0

        remaining_distance = 0.0
        try:
            current_lane_length = self._get_lane_length(lane_id)
            remaining_distance += max(current_lane_length - lane_position, 0.0)
        except traci.exceptions.TraCIException:
            return 999.0

        for edge_id in route_edges[route_index + 1:]:
            remaining_distance += self._estimate_edge_length(edge_id)

        return remaining_distance

    def _estimate_edge_length(self, edge_id: str) -> float:
        cached = self._sumo_edge_length_cache.get(edge_id)
        if cached is not None:
            return cached

        try:
            lane_count = int(traci.edge.getLaneNumber(edge_id))
        except traci.exceptions.TraCIException:
            return 0.0

        if lane_count <= 0:
            self._sumo_edge_length_cache[edge_id] = 0.0
            return 0.0

        lane_lengths = []
        for lane_index in range(lane_count):
            lane_id = f"{edge_id}_{lane_index}"
            try:
                lane_lengths.append(self._get_lane_length(lane_id))
            except traci.exceptions.TraCIException:
                continue

        if not lane_lengths:
            self._sumo_edge_length_cache[edge_id] = 0.0
            return 0.0
        length = max(lane_lengths)
        self._sumo_edge_length_cache[edge_id] = length
        return length

    def _apply_actions(self, target_speeds: Dict[str, float]):
        """
        Apply MARL actions to SUMO vehicles.

        A vehicle the policy commanded on the previous step but not on this
        one (a warmup or no-action step, an external loop that sent nothing
        for it) is handed back to SUMO's own car-following model for as long
        as it stays un-commanded: TraCI's ``setSpeed`` persists until released
        with ``-1``, so it used to keep its last commanded speed instead. A
        learning vehicle that has never been commanded still holds the
        departure speed the spawner set.

        Args:
            target_speeds: Dict[vehicle_id, speed_kmh]
        """
        commanded = set()
        for veh_id, speed_kmh in target_speeds.items():
            try:
                # Convert km/h to m/s
                speed_ms = speed_kmh / 3.6

                # Clip to reasonable range
                speed_ms = max(0.0, min(speed_ms, 30.0))  # Max ~108 km/h

                # Apply speed to SUMO vehicle
                traci.vehicle.setSpeed(veh_id, speed_ms)
                commanded.add(veh_id)

            except traci.exceptions.TraCIException as e:
                logger.warning(f"Failed to set speed for vehicle {veh_id}: {e}")

        # Release only on the commanded -> un-commanded transition: one TraCI
        # call per hand-back, none for a vehicle already released or never
        # commanded (the latter keeps the spawner's departure speed).
        terminal = self.episode.terminal_agents
        for veh_id in getattr(self, "_commanded_vehicles", set()) - commanded:
            if veh_id in terminal:
                continue
            try:
                traci.vehicle.setSpeed(veh_id, -1)
            except traci.exceptions.TraCIException as exc:
                logger.debug(f"Could not release {veh_id} to SUMO: {exc}")
        self._commanded_vehicles = commanded

    def _check_events(self) -> List[StepEvent]:
        """
        Check for events (collisions, arrivals).

        Returns:
            List of StepEvent objects
        """
        events = []
        collided_vehicle_ids = set()

        # A vehicle at its destination point has finished its trip: record the
        # success and remove it. SUMO's own arrival, at the end of the target
        # edge, is the fallback below; left to it alone, a trip ended wherever
        # the target edge happens to end, so its length depended on how the
        # network splits its edges rather than on the destination point, and
        # differed between exits with edges of different lengths.
        reached = self._vehicles_at_destination()
        for veh_id in reached:
            try:
                traci.vehicle.remove(veh_id)
            except traci.exceptions.TraCIException as exc:
                logger.debug(f"Failed to remove {veh_id} at its destination: {exc}")
        arrived_vehicles = list(reached) + [
            veh_id for veh_id in traci.simulation.getArrivedIDList() if veh_id not in reached]
        for veh_id in arrived_vehicles:
            if veh_id not in self.episode.departed_vehicles:
                self.episode.departed_vehicles.add(veh_id)
                self.episode.terminal_agents.add(veh_id)
                self._queue_route_intent_terminal(veh_id)
                if self.vehicle_spawner is not None:
                    self.vehicle_spawner.remove_vehicle(veh_id)
                events.append(StepEvent(
                    step=self.current_step,
                    event_id=f'success_{veh_id}_{self.current_step}',
                    vehicle_id=veh_id,
                    event_type='success'
                ))

        # Check for collisions
        collisions = traci.simulation.getCollisions()
        for collision in collisions:
            # Get colliding vehicle IDs
            collider = collision.collider
            victim = collision.victim

            for veh_id in [collider, victim]:
                if veh_id and veh_id not in self.episode.collided_vehicles:
                    self.episode.collided_vehicles.add(veh_id)
                    self.episode.terminal_agents.add(veh_id)
                    self._queue_route_intent_terminal(veh_id)
                    collided_vehicle_ids.add(veh_id)
                    if veh_id in self.episode.active_vehicles:
                        self.episode.active_vehicles[veh_id]['collisions'] += 1
                    events.append(StepEvent(
                        step=self.current_step,
                        event_id=f'collision_{veh_id}_{self.current_step}',
                        vehicle_id=veh_id,
                        event_type='collision'
                    ))

        self._remove_collided_vehicles(collided_vehicle_ids)
        return events

    def _pending_insertions(self) -> List[str]:
        """Vehicles added this episode that SUMO has not inserted into the network.

        SUMO queues a vehicle whose departure lane has no room and inserts it
        when there is; ``traci.vehicle.getIDList`` lists only vehicles in the
        network. The queue is read from SUMO where the API exists and
        completed from the spawner's registry (added, neither arrived nor
        collided, not in the network), so the set does not depend on the SUMO
        version. A short approach edge admits one vehicle at a time, so under
        demand a sizeable share of an episode's spawns can still be queued
        when the episode ends.
        """
        pending: List[str] = []
        getter = getattr(traci.simulation, "getPendingVehicles", None)
        if getter is not None:
            try:
                pending = list(getter())
            except Exception as exc:  # noqa: BLE001 - libsumo raises its own FatalTraCIError without a network
                logger.debug(f"getPendingVehicles failed: {exc}")
        if self.vehicle_spawner is not None:
            in_network = set(traci.vehicle.getIDList())
            for veh_id in list(self.vehicle_spawner.spawned_vehicles):
                if (veh_id not in in_network and veh_id not in pending
                        and veh_id not in self.episode.departed_vehicles
                        and veh_id not in self.episode.collided_vehicles):
                    pending.append(veh_id)
        return pending

    def pending_spawn_count(self) -> int:
        """Spawn events of this episode not yet due -- CARLA's ``pending_spawns``.

        While it is non-zero the episode's traffic is not complete, however
        many of the vehicles already in the network have finished. A vehicle
        SUMO holds in its insertion queue is not counted here: it was spawned,
        so it is an unfinished route (see :meth:`queued_insertions`).
        """
        if self.use_traffic_manager and self.traffic_manager is not None:
            return int(self.traffic_manager.total_events or 0)
        return 0

    def queued_insertions(self) -> List[str]:
        """Vehicles added this episode that SUMO has not inserted yet.

        Unfinished routes, as CARLA's never-entered spawns are: the
        evaluation counts them with the active vehicles, so an episode that
        ends with a queue reports them as timeouts -- exactly as the training
        summary written by :meth:`reset` does -- instead of leaving them out
        of its totals.
        """
        return self._pending_insertions()

    def _vehicles_at_destination(self):
        """Active vehicles within ``arrival_radius`` of their own destination point."""
        lookup = getattr(getattr(self, "vehicle_spawner", None), "get_spawn_event", None)
        coordinates = getattr(self, "coordinates", None)
        if lookup is None or coordinates is None:
            return []
        radius = float(getattr(self, "arrival_radius", 5.0))
        reached = []
        for veh_id in traci.vehicle.getIDList():
            if veh_id in self.episode.departed_vehicles or veh_id in self.episode.collided_vehicles:
                continue
            event = lookup(veh_id)
            location = getattr(getattr(event, "destination", None), "location", None)
            if location is None:
                continue
            try:
                sumo_x, sumo_y = traci.vehicle.getPosition(veh_id)
            except traci.exceptions.TraCIException:
                continue
            x, y = coordinates.sumo_to_carla(float(sumo_x), float(sumo_y))
            if math.hypot(x - float(location.x), y - float(location.y)) <= radius:
                reached.append(veh_id)
        return reached

    def _remove_collided_vehicles(self, vehicle_ids):
        """Remove collided vehicles after recording the event so they do not block the junction."""
        for veh_id in vehicle_ids:
            try:
                if veh_id in traci.vehicle.getIDList():
                    traci.vehicle.remove(veh_id)
            except traci.exceptions.TraCIException as exc:
                logger.debug(f"Failed to remove collided vehicle {veh_id}: {exc}")
            if self.vehicle_spawner is not None:
                self.vehicle_spawner.remove_vehicle(veh_id)

    def _calculate_rewards(self, events: List[StepEvent], observations: Dict):
        """Calculate rewards via the shared reward calculator."""
        return self.reward_calculator.calculate_sumo(
            observations,
            events,
            previous_observations=self.previous_observations,
            terminal_agents=self.episode.terminal_agents,
            ttc_metrics=self.ttc_metrics,
        )

    def _update_metrics(
        self,
        rewards: Dict,
        events: List[StepEvent],
        observations: Dict = None,
        reward_outputs=None,
        objective_outputs=None,
        scalarized_utilities=None,
        agent_preferences=None,
    ):
        """Update training metrics."""
        # Count successes this step for accurate episode_length tracking
        step_successes = sum(1 for e in events if e.event_type == 'success')

        # Update step metrics with all agent rewards and observations for traffic tracking
        self.metrics.update_step(
            rewards,
            observations,
            step_successes=step_successes,
            reward_outputs=reward_outputs,
            objective_outputs=objective_outputs,
            scalarized_utilities=scalarized_utilities,
            agent_preferences=agent_preferences,
        )

        # Track events
        for event in events:
            if event.event_type == 'collision':
                self.metrics.collisions += 1
            elif event.event_type == 'success':
                self.metrics.successes += 1

    def reset(self, prepare_next_episode: bool = True):
        """Reset the environment for a new episode."""
        logger.info(f"Resetting episode {self.current_episode}")
        # Every vehicle of the next episode owes a fresh decision.
        self.decision_zone.reset_episode()

        # Finish the episode that just ran. reset() is only called at episode
        # end (the coordinator never resets before the first episode), so the
        # condition is 'did an episode advance', not the episode number:
        # gating on current_episode > 0 would drop the first episode's
        # finish, merging its steps and rewards into the second episode's
        # metrics and skipping the learner's episode hook once per run.
        # Vehicles SUMO still holds in its insertion queue never entered the
        # network: they are unfinished routes of this episode, and they are
        # removed below with everything else SUMO knows.
        pending_vehicle_ids = self._pending_insertions()
        if self.current_step > 0:
            timeout_vehicle_ids = [
                veh_id for veh_id in self.episode.active_vehicles
                if veh_id not in self.episode.departed_vehicles and veh_id not in self.episode.collided_vehicles
            ]
            for veh_id in timeout_vehicle_ids + pending_vehicle_ids:
                self._notify_route_intent_terminal(veh_id)
            timeout_count = len(timeout_vehicle_ids) + len(pending_vehicle_ids)
            if pending_vehicle_ids:
                logger.info(
                    f"{len(pending_vehicle_ids)} vehicles were still waiting for insertion at the end of "
                    f"episode {self.current_episode}: counted as unfinished and removed"
                )
            episode_states = build_sumo_episode_states(
                current_step=self.current_step,
                collision_count=len(self.episode.collided_vehicles),
                success_count=len(self.episode.departed_vehicles),
                timeout_count=timeout_count,
                step_length=self.step_length,
            )
            episode_states['near_miss_count'] = self.ttc_metrics.near_miss_count
            if self.ttc_metrics.check_count > 0:
                episode_states['ttc_violation_rate'] = (
                    self.ttc_metrics.violation_count / self.ttc_metrics.check_count
                ) * 100.0
            else:
                episode_states['ttc_violation_rate'] = 0.0
            episode_states['competition_records'] = self._build_timeout_competition_records(
                list(self.previous_observations.keys()),
                self.current_step,
            )
            episode_metrics = self.metrics.finish_episode(episode_states)
            if self.marl_manager is not None:
                episode_metrics.update(self.marl_manager.get_intent_diagnostics())
            self.last_episode_metrics = episode_metrics
            logger.info(f"Episode {self.current_episode} metrics: {episode_metrics}")

            self.marl_manager.reset_episode(episode_metrics=episode_metrics)

            save_checkpoints = bool(self.training_config.get('save_checkpoints', True))
            if self.is_training_mode and save_checkpoints and self.checkpoint_manager:
                # Every episode, as on CARLA: save_freq gates how many
                # per-episode files are *kept* (inside _save_checkpoint), not
                # whether latest/best are written. Gating the save itself
                # would leave the last save_freq-1 episodes of every run
                # unpersisted and let 'best' be picked from a sampled subset
                # of episodes.
                self._save_checkpoint(episode_metrics)

        # Remove every vehicle SUMO still knows: those in the network and those
        # queued for insertion. getIDList lists only the former, so a queued
        # vehicle would survive the reset, enter the next episode with this
        # episode's route, and block its own id there ("to add already exists").
        for veh_id in list(traci.vehicle.getIDList()) + pending_vehicle_ids:
            try:
                traci.vehicle.remove(veh_id)
            except traci.exceptions.TraCIException as exc:
                logger.warning(f"Failed to remove {veh_id} at episode end: {exc}")

        # Reset state: a fresh SumoEpisodeState replaces per-field clears for
        # everything episode-scoped, so new fields can never miss a clear.
        self.current_step = 0
        self.episode = SumoEpisodeState()
        self._commanded_vehicles = set()
        self.previous_observations.clear()
        self.episode_events.clear()
        self.current_step_rewards.clear()
        self.current_step_reward_components.clear()
        self.current_step_reward_vectors.clear()
        self.current_step_objective_components.clear()
        self.current_step_objective_vectors.clear()
        self.current_step_agent_preferences.clear()
        self.competition_agent_metadata.clear()
        self.competition_terminal_records.clear()
        self._sumo_intent_agent_manager._vehicle_adapters = []
        self.ttc_metrics.reset()

        # The spawner's per-episode registries would otherwise grow across
        # the whole run (every timed-out SpawnEvent retained forever). The
        # route-id cache deliberately survives: SUMO keeps route definitions
        # until the connection closes, so cached ids stay valid.
        if self.vehicle_spawner is not None:
            self.vehicle_spawner.reset_episode_registries()

        if self.use_traffic_manager and self.traffic_manager:
            self._sync_traffic_manager_state()

        if not prepare_next_episode:
            return

        # Reset traffic manager for new episode
        if self.use_traffic_manager and self.traffic_manager:
            self.traffic_manager.reset()
            self._sync_traffic_manager_state()

        self.current_episode += 1

        # Step SUMO a few times to spawn initial vehicles
        for _ in range(10):
            traci.simulationStep()

    def close(self):
        """Close the SUMO connection and flush training outputs."""
        logger.info("Closing SUMO environment")
        if getattr(self, 'marl_manager', None) is not None:
            try:
                self.marl_manager.close()
            except Exception as exc:
                logger.warning(f"MARL manager close failed: {exc}")
        try:
            traci.close()
        except Exception:
            pass

    def get_episode_metrics(self) -> Dict:
        """Get current episode metrics (includes traffic metrics like max_speed)."""
        metrics = self.metrics.get_current_metrics()
        # Add fixed_dt for evaluation manager
        metrics['fixed_dt'] = self.step_length
        return metrics

    def get_current_step_rewards(self) -> Dict[int, float]:
        """Get rewards from the current step for evaluation."""
        return self.current_step_rewards.copy()

    def get_current_step_reward_components(self):
        """Get reward components from the current step for evaluation."""
        return self.current_step_reward_components.copy()

    def get_current_step_reward_vectors(self):
        """Get reward vectors from the current step for evaluation."""
        return self.current_step_reward_vectors.copy()

    def get_current_step_objective_components(self):
        """Get grouped objective components from the current step for evaluation."""
        return self.current_step_objective_components.copy()

    def get_current_step_objective_vectors(self):
        """Get grouped objective vectors from the current step for evaluation."""
        return self.current_step_objective_vectors.copy()

    def get_current_agent_preferences(self):
        """Get per-agent preferences from the current step."""
        return dict(self.current_step_agent_preferences)

    def get_reward_spec(self):
        """Get reward specification metadata."""
        return self.reward_spec.to_dict()

    def get_objective_spec(self):
        """Get grouped objective specification metadata."""
        return self.objective_spec.to_dict() if self.objective_spec else None

    def reset_episode(self) -> Dict:
        """
        Reset episode and return metrics (alias for coordinator compatibility).
        Returns episode metrics before resetting.

        Note: This uses get_current_metrics() which returns metrics without
        resetting. The actual reset happens in self.reset().
        """
        metrics = self.get_episode_metrics()
        self.reset()
        return self.last_episode_metrics or metrics

    def _has_pending_traffic(self) -> bool:
        min_expected_number = 0
        try:
            min_expected_number = traci.simulation.getMinExpectedNumber()
        except Exception:
            min_expected_number = 0

        pending_events = 0
        if self.use_traffic_manager and self.traffic_manager:
            pending_events = self.traffic_manager.total_events

        return has_pending_sumo_traffic(min_expected_number, pending_events)

    def _is_episode_done(self, next_observations: Dict) -> bool:
        min_expected_number = 0
        try:
            min_expected_number = traci.simulation.getMinExpectedNumber()
        except Exception:
            min_expected_number = 0

        pending_events = self.traffic_manager.total_events if self.use_traffic_manager and self.traffic_manager else 0
        return should_end_sumo_episode(
            current_step=self.current_step,
            max_steps=self.max_steps,
            has_observations=bool(next_observations),
            min_expected_number=min_expected_number,
            pending_events=pending_events,
        )

    def _save_checkpoint(self, episode_metrics=None):
        """Save training checkpoint."""
        if self.checkpoint_manager is None or self.marl_manager is None:
            return
        self.checkpoint_manager.save_checkpoint(
            self.marl_manager.algorithm,
            self.current_episode,
            episode_metrics or {},
            intent_algorithm=getattr(self.marl_manager, "intent_algorithm", None),
        )
        # Cleanup is the periodic part (CARLA does the same): every episode is
        # saved, and every save_freq episodes the per-episode files are pruned
        # to keep_checkpoints.
        save_freq = int(self.training_config.get('save_freq', 10) or 1)
        if self.current_episode % save_freq == 0:
            keep_checkpoints = self.training_config.get('keep_checkpoints', 5)
            self.checkpoint_manager.cleanup_old_checkpoints(keep_checkpoints)

    def _load_checkpoint_from_config(self, checkpoint_path: str):
        """Load the checkpoint the config asked for, or fail loudly.

        See MARLEnv._load_checkpoint_from_config: the failure must not be
        swallowed, or an evaluation would score random weights and a resumed
        run would restart from scratch.
        """
        if not checkpoint_path:
            return
        logger.info(f"Loading checkpoint from: {checkpoint_path}")
        if not os.path.exists(checkpoint_path):
            # A kit-relative name (``<set>/<run>``, e.g. the bundled SAC
            # baseline) resolves inside the wheel's bundled baselines before
            # it is treated as missing.
            from opencda_marl._bundled import bundled_checkpoint
            packaged = bundled_checkpoint(checkpoint_path)
            if packaged is None:
                raise FileNotFoundError(
                    f"Configured checkpoint not found: {checkpoint_path} "
                    "(MARL.training.load_checkpoint)")
            logger.info(f"Bundled checkpoint {checkpoint_path} -> {packaged}")
            checkpoint_path = str(packaged)
        try:
            self.marl_manager.load_checkpoint(checkpoint_path)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load configured checkpoint {checkpoint_path}: {exc}") from exc

    def _build_objective_outputs(self, reward_outputs):
        if self.objective_spec is None:
            return {}
        return build_objective_outputs(reward_outputs, self.objective_spec)

    def _update_competition_metadata(self, observations: Dict, agent_preferences: Dict[str, list[float]]):
        if not self.competition_enabled:
            return
        for agent_id, observation in dict(observations or {}).items():
            preference = agent_preferences.get(agent_id)
            if preference is None:
                preference = agent_preferences.get(str(agent_id))
            self.competition_agent_metadata[str(agent_id)] = {
                "entry_direction": observation.get("entry_direction", "unknown"),
                "route_type": observation.get("route_type", "unknown"),
                "lane_id": observation.get("lane_id"),
                "junction_id": observation.get("junction_id"),
                "preference": list(preference or []),
                "preference_group": preference_group_from_values(preference),
                "route_intent": observation.get("route_intent"),
            }

    def _record_competition_events(self, events: List[StepEvent]):
        if not self.competition_enabled:
            return
        for event in events:
            if event.event_type not in {"success", "collision"}:
                continue
            self.competition_terminal_records.append(
                self._build_competition_record(
                    agent_id=event.vehicle_id,
                    outcome=event.event_type,
                    completion_step=event.step,
                )
            )

    def _build_timeout_competition_records(self, active_agent_ids: List[Any], completion_step: int):
        if not self.competition_enabled:
            return []
        records = list(self.competition_terminal_records)
        recorded_ids = {record.get("agent_id") for record in records}
        for agent_id in active_agent_ids:
            agent_key = str(agent_id)
            if agent_key in recorded_ids:
                continue
            records.append(
                self._build_competition_record(
                    agent_id=agent_key,
                    outcome="timeout",
                    completion_step=completion_step,
                )
            )
        return records

    def _build_competition_record(self, agent_id: Any, outcome: str, completion_step: int):
        metadata = dict(self.competition_agent_metadata.get(str(agent_id), {}) or {})
        preference = list(metadata.get("preference") or [])
        return {
            "agent_id": str(agent_id),
            "outcome": outcome,
            "completion_step": int(completion_step),
            "entry_direction": metadata.get("entry_direction", "unknown"),
            "route_type": metadata.get("route_type", "unknown"),
            "lane_id": metadata.get("lane_id"),
            "junction_id": metadata.get("junction_id"),
            "preference": preference,
            "preference_group": metadata.get("preference_group", preference_group_from_values(preference)),
            "route_intent": metadata.get("route_intent"),
        }
