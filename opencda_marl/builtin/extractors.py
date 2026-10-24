import numpy as np
from typing import Dict, Mapping
from loguru import logger

from opencda_marl.api.contracts import normalize_observations


SUPPORTED_CUSTOM_FEATURES = {
    'rel_x',
    'rel_y',
    'position_x',
    'position_y',
    'lane_position',
    'lane_position_onehot',
    'speed',
    'heading_angle',
    'dist_to_intersection',
    'dist_to_front_vehicle',
    'waypoint_buffer',
    'nearby_vehicles',
    # How many other vehicles at this junction have already committed to
    # each exit lane, one slot per lane. Off by default; enabling it adds
    # its width to the state dimension.
    'committed_exit_lanes',
}

SUPPORTED_DISCRETE_FEATURES = {
    'speed',
    'heading_angle',
    'distance_to_intersection',
    'distance_to_front_vehicle',
    'lane_position',
    'waypoint_buffer_size',
    'min_ttc',
    'distance_to_destination',
    'target_speed',
    'position_x',
    'position_y',
}


def _plain_features(mapping) -> Dict[str, int]:
    """A feature mapping as a plain ``dict`` of integer widths, whatever
    container it arrived in (OmegaConf's ``DictConfig`` included), so the
    per-step code never touches a config resolver."""
    if not mapping:
        return {}
    try:
        from omegaconf import DictConfig, OmegaConf
    except ImportError:  # pragma: no cover - omegaconf is a dependency
        DictConfig = None
    if DictConfig is not None and isinstance(mapping, DictConfig):
        mapping = OmegaConf.to_container(mapping, resolve=True)
    return {str(name): int(width) for name, width in dict(mapping).items()}


_EMPTY_NEARBY_SLOT = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0]


def _feature_plan(features: Dict[str, int]) -> list:
    """``[(name, width, reader)]`` for the configured features, in order.

    A reader appends one feature's values to the list it is given, with
    the same defaults for an absent key as the observation contract's.
    """
    plan = []
    for name, width in features.items():
        if name == 'rel_x':
            def read(obs, out):
                rel = obs.get('relative_position_to_intersection')
                out.append(rel['x'] if rel is not None else 0.0)
        elif name == 'rel_y':
            def read(obs, out):
                rel = obs.get('relative_position_to_intersection')
                out.append(rel['y'] if rel is not None else 0.0)
        elif name == 'position_x':
            def read(obs, out):
                # CARLA's legacy location can be junction-relative. The
                # explicit position fields carry the shared map coordinates.
                loc = obs.get('location') or {}
                out.append(obs.get('position_x', loc.get('x', 0.0)))
        elif name == 'position_y':
            def read(obs, out):
                loc = obs.get('location') or {}
                out.append(obs.get('position_y', loc.get('y', 0.0)))
        elif name == 'lane_position':
            def read(obs, out):
                out.append(float(obs['lane_position']) if 'lane_position' in obs else 2.0)   # middle lane
        elif name == 'lane_position_onehot':
            def read(obs, out):
                if 'lane_position' in obs:
                    lane = int(obs['lane_position'])
                    onehot = [0.0, 0.0, 0.0, 0.0]
                    if 0 <= lane <= 3:
                        onehot[lane] = 1.0
                    out.extend(onehot)
                else:
                    out.extend([0.0, 0.0, 1.0, 0.0])
        elif name == 'speed':
            def read(obs, out):
                out.append(obs.get('speed', 0.0))
        elif name == 'heading_angle':
            def read(obs, out):
                out.append(obs.get('heading_angle', 0.0))
        elif name == 'dist_to_intersection':
            def read(obs, out):
                out.append(obs.get('distance_to_intersection', 100.0))
        elif name == 'dist_to_front_vehicle':
            def read(obs, out):
                out.append(obs.get('distance_to_front_vehicle', 999.0))
        elif name == 'waypoint_buffer':
            def read(obs, out):
                if 'waypoint_buffer_size' in obs:
                    out.append(float(obs['waypoint_buffer_size']))
                else:
                    out.append(min(50.0, max(0.0, obs.get('distance_to_intersection', 100.0) / 2.0)))
        elif name == 'committed_exit_lanes':
            lanes_width = int(width or 4)

            def read(obs, out, _w=lanes_width):
                occupancy = list(obs.get('committed_exit_lanes') or [])
                out.extend(float(v) for v in (occupancy + [0.0] * _w)[:_w])
        elif name == 'nearby_vehicles':
            def read(obs, out):
                nearby = obs.get('nearby_vehicles')
                if nearby is not None:
                    out.extend(nearby)
                else:
                    for _ in range(5):
                        out.extend(_EMPTY_NEARBY_SLOT)
        else:
            logger.warning(f"Unknown custom feature: {name}")

            def read(obs, out, _w=int(width)):
                out.extend([0.0] * _w)
        plan.append((name, int(width), read))
    return plan


class ObservationExtractor:
    """Extract MARL-relevant features from simulation observations"""

    #: Built-in families that consume ``{agent_id: state}`` multi-agent
    #: observations. Used only as the fallback when the caller does not pass
    #: an explicit ``multi_agent`` capability (registered policies do).
    _MULTI_AGENT_ALGORITHMS = (
        'matd3', 'pcmatd3', 'mappo', 'pcmappo', 'masac', 'pcmasac',
    )

    def __init__(self, config: Dict, algorithm: str, *, multi_agent: bool | None = None):
        self.config = config
        self.algorithm = algorithm
        # Observation shape follows a declared capability rather than a name
        # list, so an externally registered multi-agent policy gets the same
        # observation contract as the built-ins.
        self.multi_agent = (
            bool(multi_agent) if multi_agent is not None
            else algorithm in self._MULTI_AGENT_ALGORITHMS
        )

        if algorithm == 'q_learning':
            self.q_config = config.get('q_learning', {})
            self.state_features = self.q_config.get('state_features', {})
            self._validate_feature_names(self.state_features, SUPPORTED_DISCRETE_FEATURES, "q_learning")
            self.state_dim = self._calculate_state_dim()
        elif algorithm in ('matd3', 'pcmatd3'):
            self.matd3_config = config.get(algorithm, {})
            self.features_config = self.matd3_config.get(
                'features',
                config.get('matd3', {}).get('features', {}),
            )
            self.custom_features = _plain_features(self.features_config)
            self._validate_feature_names(self.custom_features, SUPPORTED_CUSTOM_FEATURES, algorithm)
            self.state_dim = self._calculate_custom_feature_dim()
        elif algorithm == 'mappo':
            # MAPPO uses the same feature family as MATD3 for fair comparison
            self.mappo_config = config.get('mappo', {})
            # Try to get features from mappo config, fallback to matd3 config
            self.features_config = self.mappo_config.get('features',
                                    config.get('matd3', {}).get('features', {}))
            self.custom_features = _plain_features(self.features_config)
            self._validate_feature_names(self.custom_features, SUPPORTED_CUSTOM_FEATURES, "mappo")
            self.state_dim = self._calculate_custom_feature_dim()
        elif algorithm == 'pcmappo':
            self.pcmappo_config = config.get('pcmappo', {})
            self.features_config = self.pcmappo_config.get(
                'features',
                config.get('mappo', {}).get(
                    'features',
                    config.get('matd3', {}).get('features', {}),
                ),
            )
            self.custom_features = _plain_features(self.features_config)
            self._validate_feature_names(self.custom_features, SUPPORTED_CUSTOM_FEATURES, "pcmappo")
            self.state_dim = self._calculate_custom_feature_dim()
        elif algorithm in ('masac', 'pcmasac'):
            # SAC-family controllers use the same feature extraction as MATD3/MAPPO.
            self.sac_config = config.get(algorithm, {})
            self.features_config = self.sac_config.get('features', {})
            self.custom_features = _plain_features(self.features_config)
            self._validate_feature_names(self.custom_features, SUPPORTED_CUSTOM_FEATURES, algorithm)
            self.state_dim = self._calculate_custom_feature_dim()
        elif algorithm == 'dqn':
            # DQN can use configurable features (8D) or fallback to legacy 7D
            self.dqn_config = config.get('dqn', {})
            self.features_config = self.dqn_config.get('features', {})
            if self.features_config:
                self.custom_features = _plain_features(self.features_config)
                self._validate_feature_names(self.custom_features, SUPPORTED_CUSTOM_FEATURES, "dqn")
                self.state_dim = self._calculate_custom_feature_dim()
            else:
                # Fallback to legacy 7D continuous features
                self.state_dim = 7
        else:
            # Any other registered algorithm: its features live in its own
            # config section (``MARL.<algorithm>.features``). This is what
            # lets an externally registered policy use the shared observation
            # pipeline without an entry in this file. An unknown name used to
            # fall through here and produce a half-initialized extractor (no
            # state_dim, no validation) that only failed much later, so a
            # missing features block is now a loud error instead.
            self.features_config = config.get(algorithm, {}).get('features', {})
            if not self.features_config:
                raise ValueError(
                    f"No observation features configured for algorithm "
                    f"'{algorithm}'. Set MARL.{algorithm}.features (a mapping "
                    f"of feature name -> width). Supported feature names: "
                    f"{sorted(SUPPORTED_CUSTOM_FEATURES)}")
            self.custom_features = _plain_features(self.features_config)
            self._validate_feature_names(
                self.custom_features, SUPPORTED_CUSTOM_FEATURES, algorithm)
            self.state_dim = self._calculate_custom_feature_dim()
        self._feature_plan = _feature_plan(getattr(self, "custom_features", None) or {})
        self._expected_dim = sum(width for _name, width, _read in self._feature_plan)

    def extract(self, observations: Dict) -> Dict[str, np.ndarray]:
        """
        Extract state features for MARL from simulation observations.

        All output keys are str(actor_id). Input keys are coerced to str at
        this boundary so downstream code (manager, strategy) can rely on str
        keys exclusively.

        Args:
            observations: Dict from scenario_manager.get_observations()

        Returns:
            Dict[str(agent_id), state_vector] for MARL decision making
        """
        if observations:
            observations = {str(k): v for k, v in observations.items()}
        observations = normalize_observations(observations)

        if self.algorithm == 'q_learning':
            return self._extract_discrete(observations)
        elif self.multi_agent:
            # MATD3, MAPPO, SAC, preference-conditioned variants, and any
            # registered multi-agent policy share this observation contract.
            return self._extract_multi_agent(observations)
        elif self.algorithm == 'dqn':
            # DQN uses custom features but returns simple state arrays (not multi-agent format)
            if hasattr(self, 'custom_features') and self.custom_features:
                return self._extract_custom_features(observations)
            else:
                return self._extract_continuous(observations)
        else:
            return self._extract_continuous(observations)

    def augment_with_preferences(
        self,
        states: Dict,
        preferences: Mapping[str | int, np.ndarray] | None,
    ) -> Dict:
        """Append per-agent preferences after state assembly."""
        if not preferences:
            return states

        if self.multi_agent:
            return self._augment_multi_agent_states(states, preferences)

        if self.algorithm == 'dqn':
            augmented = {}
            for agent_id, state in states.items():
                preference = self._resolve_preference(preferences, agent_id)
                augmented[agent_id] = np.concatenate([state, preference]).astype(np.float32)
            return augmented

        return states

    def _validate_feature_names(self, features: Dict, supported_features: set, algorithm: str):
        if not features:
            return

        unknown = sorted(set(features.keys()) - supported_features)
        if unknown:
            raise ValueError(
                f"Unsupported feature names for {algorithm}: {unknown}. "
                f"Supported features: {sorted(supported_features)}"
            )

    def _extract_continuous(self, observations: Dict) -> Dict[int, np.ndarray]:
        """Extract continuous state features for DQN and other deep RL algorithms (7D enhanced feature set)"""
        states = {}

        for agent_id, obs in observations.items():
            # New 7D feature set for enhanced intersection navigation
            state_features = []

            # 1-2. Relative position to intersection (x, y)
            if 'relative_position_to_intersection' in obs:
                state_features.extend([
                    obs['relative_position_to_intersection']['x'], 
                    obs['relative_position_to_intersection']['y']
                ])
            else:
                # Fallback to legacy location if available
                if 'location' in obs:
                    state_features.extend([obs['location']['x'], obs['location']['y']])
                else:
                    state_features.extend([0.0, 0.0])

            # 3. Speed (magnitude of velocity)
            if 'speed' in obs:
                state_features.append(obs['speed'])
            else:
                state_features.append(0.0)

            # 4. Heading angle (vehicle orientation)
            if 'heading_angle' in obs:
                state_features.append(obs['heading_angle'])
            else:
                state_features.append(0.0)

            # 5. Distance to intersection
            if 'distance_to_intersection' in obs:
                state_features.append(obs['distance_to_intersection'])
            else:
                state_features.append(100.0)  # Default large distance

            # 6. Distance to front vehicle
            if 'distance_to_front_vehicle' in obs:
                state_features.append(obs['distance_to_front_vehicle'])
            else:
                state_features.append(999.0)  # Default large distance

            # 7. Lane position (discrete but normalized to continuous)
            if 'lane_position' in obs:
                # Normalize lane position from [-3, 3] to [-1, 1]
                lane_pos = float(obs['lane_position']) / 3.0
                state_features.append(lane_pos)
            else:
                state_features.append(0.0)  # Unknown lane

            # Convert to numpy array (should be 7 features)
            states[agent_id] = np.array(state_features, dtype=np.float32)

            # Debug: Log feature dimensions occasionally
            if len(state_features) != 7:
                logger.warning(f"Agent {agent_id}: Expected 7 features, got {len(state_features)}: {state_features}")

        return states

    def _extract_multi_agent(self, observations: Dict) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Extract multi-agent observations for actor-critic MARL algorithms.
        
        Returns a structured format where each agent has access to all agents' states.
        
        Args:
            observations: Dict from scenario_manager.get_observations()
            
        Returns:
            Dict mapping agent_id to dict containing:
                - 'ego_state': Own custom feature vector
                - 'all_states': Dict of all agents' custom feature vectors
        """
        # Extract individual states using custom feature extractor
        individual_states = self._extract_custom_features(observations)

        # Every ego sees the same all-agents table, so build it once and
        # share it (consumers only read it). Building a fresh dict per ego
        # made this O(N^2) in dict construction for N agents.
        all_states = {str(agent_id): state for agent_id, state in individual_states.items()}
        return {
            str(ego_id): {'ego_state': ego_state, 'all_states': all_states}
            for ego_id, ego_state in individual_states.items()
        }

    def _augment_multi_agent_states(
        self,
        states: Dict[str, Dict[str, np.ndarray]],
        preferences: Mapping[str | int, np.ndarray],
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Append each agent's preference vector to its state — once per
        agent, not once per (ego, agent) pair.

        The previous version rebuilt and re-concatenated the full all-agents
        table for every ego (N^2 concatenations and N^2 preference lookups),
        which was the single largest Python cost per SUMO step once the
        learner's own hot spots were removed. Every ego's ``all_states``
        holds identical vectors, so they are computed once and shared.
        """
        if not states:
            return {}
        # The all_states tables are shared (or identical) across egos, so the
        # first one is representative: augment its rows once.
        first_table = next(iter(states.values()))['all_states']
        augmented_states: Dict[str, np.ndarray] = {
            str(agent_id): np.concatenate(
                [agent_state, self._resolve_preference(preferences, agent_id)]
            ).astype(np.float32)
            for agent_id, agent_state in first_table.items()
        }

        augmented = {}
        for ego_id, multi_agent_data in states.items():
            key = str(ego_id)
            ego_vec = augmented_states.get(key)
            if ego_vec is None:  # ego absent from the table: augment its own row
                ego_vec = np.concatenate(
                    [multi_agent_data['ego_state'],
                     self._resolve_preference(preferences, ego_id)]
                ).astype(np.float32)
            augmented[key] = {'ego_state': ego_vec, 'all_states': augmented_states}
        return augmented

    def _resolve_preference(
        self,
        preferences: Mapping[str | int, np.ndarray],
        agent_id: str | int,
    ) -> np.ndarray:
        preference = preferences.get(agent_id)
        if preference is None:
            preference = preferences.get(str(agent_id))
        if preference is None:
            try:
                preference = preferences.get(int(agent_id))
            except (TypeError, ValueError):
                preference = None
        if preference is None:
            raise KeyError(f"Missing preference vector for agent {agent_id}")
        return np.asarray(preference, dtype=np.float32)

    def _extract_custom_features(self, observations: Dict) -> Dict[int, np.ndarray]:
        """Extract custom configurable features for actor-critic MARL algorithms.

        The feature list is resolved once at construction into a plan of
        readers over plain Python values: a per-agent walk over the
        configured mapping would go through OmegaConf's resolver for every
        feature of every agent, which costs far more than the observations
        themselves.
        """
        plan = self._feature_plan
        expected_dim = self._expected_dim
        states = {}
        for agent_id, obs in observations.items():
            state_features: list = []
            for _name, _width, read in plan:
                read(obs, state_features)
            if len(state_features) != expected_dim:
                logger.warning(f"Agent {agent_id}: Expected {expected_dim} features, got {len(state_features)}: {state_features}")
            states[agent_id] = np.array(state_features, dtype=np.float32)
        return states

    def _extract_discrete(self, observations: Dict) -> Dict[int, int]:
        """Extract and discretize states for Q-learning based on config"""
        states = {}

        for agent_id, obs in observations.items():
            state_indices = []

            # Process each configured feature
            for feature_name, settings in self.state_features.items():
                if feature_name in obs:
                    value = obs[feature_name]
                    bins = settings.get('bins', [])
                    
                    if bins:
                        # Use numpy.digitize to bin the value
                        bin_idx = np.digitize(value, bins)
                        state_indices.append(bin_idx)
                    else:
                        # No bins specified, treat as binary (0 or 1)
                        state_indices.append(1 if value > 0 else 0)
                else:
                    logger.warning(f"Feature '{feature_name}' not found in observations for agent {agent_id}")
                    logger.debug(f"Available features: {list(obs.keys())}")
                    # Default to 0 if feature not available
                    state_indices.append(0)

            # Convert multi-dimensional state indices to single state index
            single_state_idx = self._to_single_index(state_indices)
            states[agent_id] = single_state_idx

        return states

    def _to_single_index(self, state_indices: list) -> int:
        """Convert multi-dimensional state indices to single state index"""
        if not state_indices:
            return 0
            
        # Calculate single index using positional encoding
        single_idx = 0
        multiplier = 1
        
        for feature_name, settings in self.state_features.items():
            if state_indices:
                idx = state_indices.pop(0)
                bins = settings.get('bins', [])
                num_bins = len(bins) + 1 if bins else 2
                
                single_idx += idx * multiplier
                multiplier *= num_bins
        
        return single_idx

    def _calculate_state_dim(self) -> int:
        """Calculate total discrete state space size from feature bins"""
        if not self.state_features:
            return 1  # Fallback to single state
            
        state_dim = 1
        for feature, settings in self.state_features.items():
            bins = settings.get('bins', [])
            # Number of bins is len(bins) + 1 (for values beyond last bin)
            num_bins = len(bins) + 1 if bins else 2  # Default to 2 bins if none specified
            state_dim *= num_bins
            
        return state_dim

    def _calculate_custom_feature_dim(self) -> int:
        """Calculate total custom feature dimension from configuration"""
        if not self.custom_features:
            logger.warning("No custom features defined, falling back to default 7D features")
            return 7  # Fallback to continuous features
            
        total_dim = sum(self.custom_features.values())
        logger.info(f"Custom feature dimension: {total_dim}D from features: {list(self.custom_features.keys())}")
        return total_dim
