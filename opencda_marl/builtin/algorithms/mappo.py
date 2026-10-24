'''
Author       : AXIBA leolihao@arizona.edu
Date         : 2025-12-03
FilePath     : /OpenCDA-MARL/opencda_marl/algorithms/mappo.py
Description  : Multi-Agent Proximal Policy Optimization (MAPPO) algorithm
               for cooperative autonomous intersection management.

Reference    : "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games"
               Yu et al., NeurIPS 2021
               GitHub: https://github.com/marlbenchmark/on-policy

               @article{yu2022surprising,
                 title={The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games},
                 author={Yu, Chao and Velu, Akash and Vinitsky, Eugene and others},
                 journal={Advances in Neural Information Processing Systems},
                 volume={35},
                 pages={24611--24624},
                 year={2022}
               }

Copyright (c) 2025-2026 Lihao Guo and the Radar Lab, University of Arizona
SPDX-License-Identifier: Apache-2.0
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import gc
from typing import Dict, Any, List, Tuple, Optional
from torch.distributions import Normal
from loguru import logger

from .base_algorithm import BaseAlgorithm
from .encoders import LSTMEncoder
from .rollout_buffer import RolloutBuffer
from .route_head import RouteHeadMixin
from .value_norm import ValueNormalizer


class GaussianActor(nn.Module):
    """
    Gaussian policy network for continuous action space.

    Outputs mean and std for a Gaussian distribution over actions.
    Uses learnable log_std parameter for stable training.

    Architecture follows MAPPO paper recommendations:
    - Layer normalization for stability
    - Orthogonal initialization for ReLU layers
    - Small std initialization for output layer

    Parameters
    ----------
    state_dim : int
        Input observation dimension
    action_dim : int
        Output action dimension
    hidden_dims : List[int]
        Hidden layer dimensions
    max_action : float
        Maximum action value (for clamping)
    min_action : float
        Minimum action value
    init_std : float
        Initial standard deviation for exploration
    """

    def __init__(self, state_dim: int, action_dim: int,
                 hidden_dims: List[int] = [256, 256],
                 max_action: float = 65.0, min_action: float = 0.0,
                 init_std: float = 0.5):
        super(GaussianActor, self).__init__()

        self.max_action = max_action
        self.min_action = min_action
        self.action_dim = action_dim

        # Build feature extraction network
        layers = []
        in_dim = state_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim

        self.feature_net = nn.Sequential(*layers)

        # Action mean head
        self.action_mean = nn.Linear(hidden_dims[-1], action_dim)

        # Learnable log standard deviation (shared across all states)
        # Initialize to produce init_std as initial standard deviation
        self.action_log_std = nn.Parameter(torch.ones(action_dim) * np.log(init_std))

        # Min/max log std to prevent numerical issues
        self.log_std_min = np.log(0.01)  # Min std = 0.01
        self.log_std_max = np.log(2.0)   # Max std = 2.0

        # Initialize weights properly
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights for stable training."""
        # Initialize feature network with orthogonal initialization
        for layer in self.feature_net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.constant_(layer.bias, 0.0)

        # Initialize action mean with small weights for controlled initial actions
        nn.init.orthogonal_(self.action_mean.weight, gain=0.01)
        # Bias initialized to produce a middle-range speed. (The action IS the
        # target speed in km/h; a zero bias — the previous code, despite its
        # comment — made the initial policy command ~0 km/h, so nothing moved
        # until the mean had crawled up in steps of std <= 2 km/h.)
        nn.init.constant_(self.action_mean.bias, 0.5 * (self.min_action + self.max_action))

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass to get action distribution parameters.

        Parameters
        ----------
        state : torch.Tensor
            Input observation [batch_size, state_dim]

        Returns
        -------
        action_mean : torch.Tensor
            Mean of Gaussian distribution [batch_size, action_dim]
        action_std : torch.Tensor
            Standard deviation [batch_size, action_dim]
        """
        features = self.feature_net(state)
        action_mean = self.action_mean(features)

        # Clamp log_std for numerical stability
        log_std = torch.clamp(self.action_log_std, self.log_std_min, self.log_std_max)
        action_std = log_std.exp().expand_as(action_mean)

        return action_mean, action_std

    def get_action(self, state: torch.Tensor, deterministic: bool = False
                   ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action from policy.

        Parameters
        ----------
        state : torch.Tensor
            Current observation
        deterministic : bool
            If True, return mean action (no sampling)

        Returns
        -------
        action : torch.Tensor
            Sampled or deterministic action
        log_prob : torch.Tensor
            Log probability of action under current policy
        """
        mean, std = self.forward(state)
        dist = Normal(mean, std, validate_args=False)  # std = exp(clamped log_std) > 0; validation costs device syncs

        # The RAW (unclipped) action is returned together with its log-prob:
        # it is what the rollout stores and what evaluate_actions re-scores,
        # so the PPO ratio is exact. Callers clip it to [min, max] for the
        # environment. (Clipping before storing made the ratio of every
        # clipped sample wrong.)
        if deterministic:
            action = mean
        else:
            action = dist.rsample()  # Reparameterized sampling for gradients
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob

    def evaluate_actions(self, state: torch.Tensor, action: torch.Tensor
                         ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate given actions under current policy.

        Used during PPO update to compute ratio between old and new policies.

        Parameters
        ----------
        state : torch.Tensor
            Batch of observations
        action : torch.Tensor
            Batch of actions to evaluate

        Returns
        -------
        log_prob : torch.Tensor
            Log probability of actions
        entropy : torch.Tensor
            Entropy of the action distribution
        """
        mean, std = self.forward(state)
        dist = Normal(mean, std, validate_args=False)  # std = exp(clamped log_std) > 0; validation costs device syncs

        # Log probability of given actions
        log_prob = dist.log_prob(action).sum(dim=-1)

        # Entropy (encourages exploration)
        entropy = dist.entropy().sum(dim=-1)

        return log_prob, entropy


class CentralizedCritic(nn.Module):
    """
    Centralized value function for MAPPO (CTDE paradigm).

    Takes global state (potentially all agents' observations) as input
    and outputs a single value estimate.

    In our intersection scenario, the global state includes:
    - Ego agent observation
    - Multi-agent context from LSTM encoder

    Parameters
    ----------
    global_state_dim : int
        Dimension of global state input
    hidden_dims : List[int]
        Hidden layer dimensions
    """

    def __init__(self, global_state_dim: int, hidden_dims: List[int] = [256, 256]):
        super(CentralizedCritic, self).__init__()

        # Build value network
        layers = []
        in_dim = global_state_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim

        # Final value output
        layers.append(nn.Linear(hidden_dims[-1], 1))

        self.value_net = nn.Sequential(*layers)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights."""
        for layer in self.value_net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.constant_(layer.bias, 0.0)

        # Last layer with smaller gain for stable initial values
        final_layer = self.value_net[-1]
        nn.init.orthogonal_(final_layer.weight, gain=1.0)

    def forward(self, global_state: torch.Tensor) -> torch.Tensor:
        """
        Compute state value.

        Parameters
        ----------
        global_state : torch.Tensor
            Global state observation [batch_size, global_state_dim]

        Returns
        -------
        value : torch.Tensor
            State value estimate [batch_size, 1]
        """
        return self.value_net(global_state)



class MAPPOAlgorithm(RouteHeadMixin, BaseAlgorithm):
    """
    Multi-Agent Proximal Policy Optimization (MAPPO) algorithm.

    Implements the CTDE (Centralized Training, Decentralized Execution) paradigm:
    - Training: Centralized critic sees global state (all agents)
    - Execution: Decentralized actors use only local observations

    Key differences from TD3:
    - On-policy: Experiences used once then discarded
    - Stochastic policy: Gaussian distribution over actions
    - PPO clipping: Prevents large policy updates
    - Entropy bonus: Encourages exploration

    Reference
    ---------
    "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games"
    Yu et al., NeurIPS 2021
    GitHub: https://github.com/marlbenchmark/on-policy

    Parameters
    ----------
    config : Dict[str, Any]
        MAPPO configuration
    state_dim : int
        Individual agent observation dimension (44D in our case)
    action_dim : int
        Action dimension (1 for continuous speed control)
    """

    #: IntentPolicy capability (RouteHeadMixin): the PPO route head trains
    #: through this controller's optimizer.
    route_head_cls = "HybridRoutePPOHead"
    route_head_shares_optimizer = True

    def __init__(self, config: Dict[str, Any], state_dim: int, action_dim: int):
        super().__init__(config, state_dim, action_dim)

        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # =========== MAPPO Hyperparameters ===========
        # PPO clipping parameters
        self.clip_param = config.get('clip_param', 0.2)
        self.ppo_epochs = config.get('ppo_epochs', 10)
        self.num_mini_batch = config.get('num_mini_batch', 4)

        # Loss coefficients
        self.entropy_coef = config.get('entropy_coef', 0.01)
        self.value_loss_coef = config.get('value_loss_coef', 0.5)
        self.max_grad_norm = config.get('max_grad_norm', 0.5)

        # GAE parameters
        self.gamma = config.get('gamma', 0.99)
        self.gae_lambda = config.get('gae_lambda', 0.95)
        self.use_gae = config.get('use_gae', True)

        # Learning rate
        self.learning_rate = config.get('learning_rate', 3e-4)

        # Rollout settings
        self.n_steps = config.get('n_steps', 128)  # Steps before update
        self.batch_size = config.get('batch_size', 32)

        # Action bounds (same as TD3)
        self.max_action = config.get('max_action', 65.0)
        self.min_action = config.get('min_action', 0.0)

        # Warmup configuration (NEW: random exploration before learning)
        self.warmup_steps = config.get('warmup_steps', 0)

        # CTDE settings
        self.use_centralized_V = config.get('use_centralized_V', True)
        self.share_policy = config.get('share_policy', True)

        # Exposed even though the speed actor ignores preferences: the route
        # head's spawn observation appends a preference vector of this width
        # (the policy intent bridge reads ``pref_dim`` to build it), so the
        # observation it receives must match the width the head was built for.
        self.pref_dim = int(config.get('pref_dim', 0))

        # =========== Network Architecture ===========
        # LSTM encoder for multi-agent context
        self.lstm_hidden_size = config.get('lstm_hidden_size', 256)
        self.lstm_num_layers = config.get('lstm_num_layers', 1)

        self.lstm_encoder = self._build_context_encoder(state_dim)

        # Actor (Gaussian policy)
        actor_hidden_dims = config.get('actor_hidden_dims', [256, 256])
        init_std = config.get('init_std', 0.5)

        self.actor = GaussianActor(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=actor_hidden_dims,
            max_action=self.max_action,
            min_action=self.min_action,
            init_std=init_std
        ).to(self.device)

        # Critic (centralized value function)
        # Input: ego state + LSTM context
        critic_input_dim = state_dim + self.lstm_hidden_size
        critic_hidden_dims = config.get('critic_hidden_dims', [256, 256])

        self.critic = CentralizedCritic(
            global_state_dim=critic_input_dim,
            hidden_dims=critic_hidden_dims
        ).to(self.device)

        # =========== Optimizers ===========
        # Single optimizer for all networks (MAPPO paper recommendation)
        self.optimizer = optim.Adam([
            {'params': self.actor.parameters()},
            {'params': self.critic.parameters()},
            {'params': self.lstm_encoder.parameters()}
        ], lr=self.learning_rate)

        # Learning rate scheduler (optional)
        self.use_lr_decay = config.get('use_lr_decay', False)
        if self.use_lr_decay:
            self.lr_scheduler = optim.lr_scheduler.LinearLR(
                self.optimizer, start_factor=1.0, end_factor=0.1, total_iters=1000
            )

        # =========== Rollout Buffer ===========
        self.rollout_buffer = RolloutBuffer(
            buffer_size=self.n_steps * 4,  # Allow for multiple agents
            state_dim=state_dim,
            action_dim=action_dim,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            device=self.device
        )

        # =========== Rollout / context policy ===========
        # See PCMAPPOAlgorithm for the rationale: one PPO update per
        # ``n_steps`` environment steps, per-trajectory GAE with a critic
        # bootstrap for trajectories cut by the rollout boundary, and the
        # context re-encoded from stored raw sequences at update time
        # (frozen unless ``train_context_encoder`` is on).
        raw_cap = config.get('max_context_agents')
        self.max_context_agents: Optional[int] = int(raw_cap) if raw_cap else None
        self.train_context_encoder = bool(config.get('train_context_encoder', False))
        self._steps_since_update = 0
        self._open_next_obs: Dict[Any, Tuple[Dict[str, np.ndarray], str]] = {}
        # Value normalisation (opt-in): the critic regresses returns scaled
        # by their running mean/std; every value that enters GAE or a
        # bootstrap is de-normalised back to reward units (see value_norm.py).
        self.value_normalization = bool(config.get('value_normalization', False))
        self.value_normalizer = ValueNormalizer(1, self.device) if self.value_normalization else None

        # =========== Training State ===========
        self.training = True
        self.steps_collected = 0
        self.updates_performed = 0
        self._pretrained = False

        # Track last action per agent for store_transition
        self.last_log_probs: Dict[str, float] = {}
        self.last_values: Dict[str, float] = {}
        # Raw (unclipped) Gaussian sample per agent — the action the rollout
        # stores, so evaluate_actions re-scores exactly what was sampled.
        self.last_raw_actions: Dict[str, float] = {}

        # Training metrics
        self.training_metrics = {
            'policy_loss': 0.0,
            'value_loss': 0.0,
            'entropy': 0.0,
            'approx_kl': 0.0,
            'clip_fraction': 0.0,
            'buffer_size': 0,
        }
        logger.info(f"MAPPO initialized: state_dim={state_dim}, action_dim={action_dim}, "
                   f"device={self.device}")
        logger.info(f"MAPPO hyperparameters: clip={self.clip_param}, epochs={self.ppo_epochs}, "
                   f"entropy_coef={self.entropy_coef}, warmup_steps={self.warmup_steps}")

    def _ppo_in_warmup(self) -> bool:
        """True while the uniform-random warmup is still running.

        ``_pretrained`` (set by :meth:`load`) short-circuits it: a resumed or
        warm-started run must not throw away its first ``warmup_steps``
        agent-steps on random speeds, which is what the bare
        ``steps_collected < warmup_steps`` comparison did after every
        checkpoint load (``steps_collected`` is per-process and starts at 0).
        """
        if self.warmup_steps <= 0 or getattr(self, "_pretrained", False):
            return False
        return self.steps_collected < self.warmup_steps

    def select_action(self, multi_agent_obs: Dict[str, np.ndarray],
                      ego_agent_id: str, training: bool = True) -> Optional[float]:
        """
        Select action for ego agent using MAPPO policy.

        Parameters
        ----------
        multi_agent_obs : Dict[str, np.ndarray]
            All agents' observations
        ego_agent_id : str
            ID of the ego agent
        training : bool
            Whether in training mode

        Returns
        -------
        action : float or None
            Target speed in km/h, or None during warmup
        """
        try:
            # Warmup phase: random exploration before policy learning
            if training and self._ppo_in_warmup():
                # Random action during warmup for exploration
                random_speed = np.random.uniform(self.min_action, self.max_action)

                # Still need to store log_prob and value for transition
                self.last_log_probs[ego_agent_id] = 0.0  # Placeholder
                self.last_values[ego_agent_id] = 0.0  # Placeholder

                if self.steps_collected % 200 == 0:
                    logger.debug(f"MAPPO Warmup: step {self.steps_collected}/{self.warmup_steps}, "
                               f"random_speed={random_speed:.1f}")

                return random_speed

            # Prepare inputs
            ego_state, multi_agent_context = self._prepare_inputs(
                multi_agent_obs, ego_agent_id
            )

            with torch.no_grad():
                # Get action from actor
                if training:
                    action, log_prob = self.actor.get_action(ego_state, deterministic=False)
                else:
                    action, log_prob = self.actor.get_action(ego_state, deterministic=True)

                # Get value estimate from critic (reward units)
                critic_input = torch.cat([ego_state, multi_agent_context], dim=1)
                value = self._critic_value(critic_input)

                # Store for later transition storage
                self.last_log_probs[ego_agent_id] = log_prob.item()
                self.last_values[ego_agent_id] = value.item()
                self.last_raw_actions[ego_agent_id] = float(action.squeeze().item())

                # Extract scalar action
                final_speed = action.squeeze().item()
                final_speed = np.clip(final_speed, self.min_action, self.max_action)

                # Debug logging (periodic)
                if self.steps_collected % 1000 == 0:
                    mean, std = self.actor.forward(ego_state)
                    logger.debug(f"MAPPO Action: speed={final_speed:.1f}, "
                               f"mean={mean.item():.1f}, std={std.item():.2f}, "
                               f"value={value.item():.2f}")

                return final_speed

        except Exception as e:
            logger.error(f"Error in MAPPO action selection: {e}")
            import traceback
            traceback.print_exc()
            return None

    def store_transition(self, multi_agent_obs: Dict[str, np.ndarray],
                        ego_agent_id: str, action: float, reward: float,
                        next_multi_agent_obs: Dict[str, np.ndarray], done: bool):
        """
        Store transition in rollout buffer.

        Parameters
        ----------
        multi_agent_obs : Dict[str, np.ndarray]
            Current observations
        ego_agent_id : str
            Ego agent ID
        action : float
            Action taken (speed)
        reward : float
            Reward received
        next_multi_agent_obs : Dict[str, np.ndarray]
            Next observations
        done : bool
            Episode termination flag
        """
        try:
            # Warmup transitions come from the uniform-random policy and
            # carry placeholder log-probs; they are off-policy data and must
            # not enter the PPO rollout (they used to poison the first update
            # after warmup). Count them towards warmup and move on.
            if self._ppo_in_warmup():
                self.steps_collected += 1
                return

            # Get stored log_prob and value from action selection
            log_prob = self.last_log_probs.get(ego_agent_id, 0.0)
            value = self.last_values.get(ego_agent_id, 0.0)
            # The rollout stores the raw sample the log-prob refers to; the
            # clipped speed the environment received is a fallback only.
            stored_action = self.last_raw_actions.get(ego_agent_id, action)

            # Raw neighbour sequence (ego first); the context is encoded
            # from it at update time, so the critic trains on the same
            # inputs it is evaluated on when acting.
            seq, _ = self._context_sequence(multi_agent_obs, ego_agent_id)
            trajectory_id = (self.episode_count, str(ego_agent_id))

            # Add to buffer
            self.rollout_buffer.add(
                obs=seq[0],
                action=stored_action,
                reward=reward,
                value=value,
                log_prob=log_prob,
                done=done,
                sequence=seq,
                trajectory_id=trajectory_id,
            )
            if done:
                self._open_next_obs.pop(trajectory_id, None)
            else:
                self._open_next_obs[trajectory_id] = (next_multi_agent_obs, str(ego_agent_id))

            self.steps_collected += 1

        except Exception as e:
            logger.error(f"Error storing MAPPO transition: {e}")

    def update(self) -> Dict[str, float]:
        """
        Perform PPO update if enough data collected.

        Returns
        -------
        metrics : Dict[str, float]
            Training metrics from update
        """
        # Skip updates during warmup phase
        if self._ppo_in_warmup():
            return self.training_metrics

        # One PPO update per rollout of n_steps environment steps (the
        # strategy calls update() once per data-bearing step), never on
        # fewer than batch_size transitions.
        self._steps_since_update += 1
        if self._steps_since_update < self.n_steps or len(self.rollout_buffer) < self.batch_size:
            return self.training_metrics

        try:
            # Per-trajectory GAE; open trajectories bootstrap from the
            # critic's value of their last next-state.
            self.rollout_buffer.compute_returns_and_advantages(
                last_values=self._bootstrap_values(),
                normalize_advantages=True
            )

            # PPO update epochs
            total_policy_loss = 0.0
            total_value_loss = 0.0
            total_entropy = 0.0
            total_approx_kl = 0.0
            total_clip_fraction = 0.0
            num_updates = 0

            for epoch in range(self.ppo_epochs):
                # Get mini-batches
                batch_size = max(len(self.rollout_buffer) // self.num_mini_batch, 1)

                for batch in self.rollout_buffer.get_batches(batch_size=batch_size):
                    # Unpack batch
                    obs = batch['observations']
                    actions = batch['actions'].unsqueeze(-1)  # [batch, 1]
                    old_log_probs = batch['old_log_probs']
                    advantages = batch['advantages']
                    returns = batch['returns']

                    # Evaluate current policy
                    new_log_probs, entropy = self.actor.evaluate_actions(obs, actions)

                    # Critic on the same input it sees when acting: ego state
                    # + LSTM context re-encoded from the stored sequences
                    # (through the encoder graph only when it trains).
                    if self.train_context_encoder:
                        contexts = self._encode_context_batch(batch['sequences'], batch['lengths'])
                    else:
                        with torch.no_grad():
                            contexts = self._encode_context_batch(batch['sequences'], batch['lengths'])
                    critic_input = torch.cat([obs, contexts], dim=1)
                    values = self.critic(critic_input).squeeze(-1)
                    if self.value_normalizer is not None:
                        # Critic learns in normalised space; the running
                        # statistics track the returns of this rollout.
                        self.value_normalizer.update(returns)
                        returns = self.value_normalizer.normalize(returns.unsqueeze(-1)).squeeze(-1)

                    # PPO policy loss with clipping
                    ratio = (new_log_probs - old_log_probs).exp()
                    surr1 = ratio * advantages
                    surr2 = torch.clamp(
                        ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
                    ) * advantages
                    policy_loss = -torch.min(surr1, surr2).mean()

                    # Value loss
                    value_loss = F.mse_loss(values, returns)

                    # Entropy bonus (negative because we want to maximize entropy)
                    entropy_loss = -entropy.mean()

                    # Combined loss
                    loss = (policy_loss +
                           self.value_loss_coef * value_loss +
                           self.entropy_coef * entropy_loss)

                    # Optimization step. Actor and critic(+encoder) are
                    # clipped separately, as in the MAPPO reference
                    # implementation: with unnormalized returns the critic's
                    # gradient norm can be orders of magnitude above the
                    # actor's, and one joint clip would scale the policy
                    # gradient away.
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                    nn.utils.clip_grad_norm_(
                        list(self.critic.parameters()) + list(self.lstm_encoder.parameters()),
                        self.max_grad_norm
                    )
                    self.optimizer.step()

                    # Track metrics
                    total_policy_loss += policy_loss.item()
                    total_value_loss += value_loss.item()
                    total_entropy += entropy.mean().item()

                    # Approximate KL divergence
                    with torch.no_grad():
                        approx_kl = ((ratio - 1) - (ratio.log())).mean().item()
                        clip_fraction = ((ratio - 1.0).abs() > self.clip_param).float().mean().item()

                    total_approx_kl += approx_kl
                    total_clip_fraction += clip_fraction
                    num_updates += 1

            # Average metrics
            if num_updates > 0:
                self.training_metrics = {
                    'policy_loss': total_policy_loss / num_updates,
                    'value_loss': total_value_loss / num_updates,
                    'entropy': total_entropy / num_updates,
                    'approx_kl': total_approx_kl / num_updates,
                    'clip_fraction': total_clip_fraction / num_updates,
                    'buffer_size': len(self.rollout_buffer),
                }

            # Clear buffer (on-policy requirement)
            self.rollout_buffer.clear()
            self._open_next_obs.clear()
            self._steps_since_update = 0
            self.updates_performed += 1
            self._mark_acting_dirty()
            # Loss scalars are logged with training_step as the global
            # step; without this every PPO curve collapsed onto step 0.
            self.training_step += 1

            # LR decay if enabled
            if self.use_lr_decay:
                self.lr_scheduler.step()

            # TensorBoard logging
            self.log_scalar('Loss/policy', self.training_metrics['policy_loss'], category='losses')
            self.log_scalar('Loss/value', self.training_metrics['value_loss'], category='losses')
            self.log_scalar('Policy/entropy', self.training_metrics['entropy'], category='losses')
            self.log_scalar('Policy/approx_kl', self.training_metrics['approx_kl'], category='losses')
            self.log_scalar('Policy/clip_fraction', self.training_metrics['clip_fraction'], category='losses')

            # Log progress periodically
            if self.updates_performed % 10 == 0:
                logger.info(f"MAPPO Update {self.updates_performed}: "
                          f"policy_loss={self.training_metrics['policy_loss']:.4f}, "
                          f"value_loss={self.training_metrics['value_loss']:.4f}, "
                          f"entropy={self.training_metrics['entropy']:.4f}")

            # GPU cleanup
            if torch.cuda.is_available() and self.updates_performed % 10 == 0:
                torch.cuda.empty_cache()
                gc.collect()

            return self.training_metrics

        except Exception as e:
            logger.error(f"Error in MAPPO update: {e}")
            import traceback
            traceback.print_exc()
            return self.training_metrics

    # Batched action support (see BaseAlgorithm.select_actions).
    # The orchestration loop lives in the base class; MAPPO only needs to
    # provide the warmup-stash helper and the algorithm-specific forward.
    supports_batched_actions = True

    def _warmup_action_batch(self, ego_agent_ids) -> Dict[str, float]:
        """Stash last_log_probs / last_values placeholders so
        :meth:`store_transition` can read them next step without KeyError."""
        results: Dict[str, float] = {}
        for agent_id in ego_agent_ids:
            key = str(agent_id)
            results[key] = float(np.random.uniform(self.min_action, self.max_action))
            self.last_log_probs[key] = 0.0
            self.last_values[key] = 0.0
        return results

    def _batched_forward(
        self,
        ego_states: torch.Tensor,
        context: torch.Tensor,
        resolved_ids,
        training: bool,
    ) -> Dict[str, float]:
        """One Gaussian-actor + centralized-critic pass over all ego agents."""
        with torch.no_grad():
            action, log_prob = self._acting_module("actor").get_action(
                ego_states, deterministic=not training,
            )
            critic_input = torch.cat([ego_states, context], dim=1)
            value = self._critic_value(critic_input)

        # One device-to-host transfer for the step's outputs (see pcmappo).
        from opencda_marl.builtin.algorithms.pcmappo import _to_host
        actions_np, log_probs_np, values_np, _ = _to_host(action, log_prob, value)

        results: Dict[str, float] = {}
        for i, agent_id in enumerate(resolved_ids):
            self.last_log_probs[agent_id] = float(
                log_probs_np[i].item() if log_probs_np.ndim > 0 else log_probs_np
            )
            self.last_values[agent_id] = float(
                values_np[i].item() if values_np.ndim > 0 else values_np
            )
            a = actions_np[i]
            raw = float(a.flatten()[0]) if isinstance(a, np.ndarray) and a.shape else float(a)
            self.last_raw_actions[agent_id] = raw
            results[agent_id] = float(np.clip(raw, self.min_action, self.max_action))
        return results

    def _prepare_inputs(self, multi_agent_obs: Dict[str, np.ndarray],
                        ego_agent_id: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare inputs for MAPPO networks.

        Returns
        -------
        ego_state : torch.Tensor
            Ego agent features [1, state_dim]
        multi_agent_context : torch.Tensor
            LSTM encoding [1, lstm_hidden_size]
        """
        # Same ordering as the batched acting path and the stored sequences
        # (BaseAlgorithm._context_sequence): ego first, others nearest-first.
        seq, seq_len = self._context_sequence(multi_agent_obs, ego_agent_id)
        ego_state = torch.as_tensor(seq[:1], dtype=torch.float32, device=self.device)
        if seq_len > 1:
            multi_agent_context = self.lstm_encoder(
                torch.as_tensor(seq[None], dtype=torch.float32, device=self.device))
        else:
            multi_agent_context = torch.zeros(1, self.lstm_hidden_size, device=self.device)
        return ego_state, multi_agent_context

    def _critic_value(self, critic_input: torch.Tensor) -> torch.Tensor:
        """Critic output in reward units (de-normalised when value
        normalisation is on) — what GAE and bootstraps consume."""
        replica = self.__dict__.get("_acting_replica")
        on_replica = bool(replica) and critic_input.device == replica.device
        critic = replica.modules["critic"] if on_replica else self.critic
        value = critic(critic_input)
        normalizer = self._acting_normalizer("value", value)
        if normalizer is not None:
            value = normalizer.denormalize(value)
        return value

    def _acting_modules(self):
        return {"actor": self.actor, "critic": self.critic, "lstm_encoder": self.lstm_encoder}

    def _acting_normalizers(self):
        return {"value": self.value_normalizer}

    def _bootstrap_values(self) -> Dict[Any, float]:
        """Critic values V(s_next) for every open trajectory in the rollout
        (one batched pass per distinct next-state table)."""
        last_done: Dict[Any, bool] = {}
        for trajectory_id, done in zip(self.rollout_buffer.trajectory_ids, self.rollout_buffer.dones):
            last_done[trajectory_id] = done
        by_table: Dict[int, Tuple[Dict[str, np.ndarray], List[Tuple[Any, str]]]] = {}
        for trajectory_id, done in last_done.items():
            if done:
                continue
            entry = self._open_next_obs.get(trajectory_id)
            if entry is None:
                continue
            table, agent_id = entry
            if not table or agent_id not in table:
                continue
            by_table.setdefault(id(table), (table, []))[1].append((trajectory_id, agent_id))

        values: Dict[Any, float] = {}
        with torch.no_grad():
            for table, members in by_table.values():
                ego_states, context, resolved = self._prepare_batch_inputs(
                    table, [agent_id for _, agent_id in members])
                critic_values = self._critic_value(torch.cat([ego_states, context], dim=1))
                values_np = critic_values.reshape(len(resolved), -1)[:, 0].cpu().numpy()
                for i, (trajectory_id, _) in enumerate(members):
                    values[trajectory_id] = float(values_np[i])
        return values

    def reset_episode(self):
        """Reset for new episode."""
        self.episode_count += 1
        self.last_log_probs.clear()
        self.last_values.clear()
        self.last_raw_actions.clear()

        # Log to TensorBoard
        self.log_scalar('MAPPO/steps_collected', self.steps_collected, step=self.episode_count, category='episode')
        self.log_scalar('MAPPO/updates_performed', self.updates_performed, step=self.episode_count, category='episode')

        # GPU cleanup every few episodes
        if self.episode_count % 5 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        self.finalize_episode_logging()

    def get_training_info(self) -> Dict[str, Any]:
        """Get training information."""
        info = {
            'algorithm': 'MAPPO',
            'algorithm_type': 'multi_agent_ppo',
            'episode_count': self.episode_count,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'buffer_size': len(self.rollout_buffer),
            'steps_collected': self.steps_collected,
            'updates_performed': self.updates_performed,
                'steps_collected': self.steps_collected,
            'clip_param': self.clip_param,
            'entropy_coef': self.entropy_coef,
            'warmup_steps': self.warmup_steps,
            'in_warmup': self._ppo_in_warmup(),
            'policy_loss': self.training_metrics.get('policy_loss', 0.0),
            'value_loss': self.training_metrics.get('value_loss', 0.0),
            'entropy': self.training_metrics.get('entropy', 0.0),
            'epsilon': self.entropy_coef,  # For GUI compatibility
            'device': str(self.device),
            'max_action': self.max_action,
        }
        if self.route_intent_head is not None:
            info["hybrid_route_intent_enabled"] = True
            info.update(self.route_intent_head.get_training_info())
            info["route_intent_buffer_size"] = len(self.route_intent_head.buffer)
        return info

    def log_episode_metrics(self, episode_reward: float, episode_length: int,
                           success_rate: float = 0.0, collision_rate: float = 0.0,
                           near_miss_count: int = 0, ttc_violation_rate: float = 0.0,
                           additional_metrics: Dict[str, float] = None,
                           traffic_metrics: Dict[str, float] = None):
        """Log episode-level metrics to TensorBoard."""
        # Add MAPPO-specific metrics
        mappo_metrics = {
            'buffer_size': len(self.rollout_buffer),
            'updates_performed': self.updates_performed,
                'steps_collected': self.steps_collected,
        }
        if additional_metrics:
            mappo_metrics.update(additional_metrics)

        # Call base class method
        super().log_episode_metrics(
            episode_reward, episode_length, success_rate, collision_rate,
            near_miss_count=near_miss_count,
            ttc_violation_rate=ttc_violation_rate,
            additional_metrics=mappo_metrics,
            traffic_metrics=traffic_metrics
        )

    def save(self, path: str):
        """Save MAPPO model."""
        try:
            save_data = {
                'actor_state_dict': self.actor.state_dict(),
                'critic_state_dict': self.critic.state_dict(),
                'lstm_encoder_state_dict': self.lstm_encoder.state_dict(),
                'value_normalizer_state_dict': (self.value_normalizer.state_dict()
                                                if self.value_normalizer is not None else None),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'episode_count': self.episode_count,
                'updates_performed': self.updates_performed,
                'steps_collected': self.steps_collected,
                'training_metrics': self.training_metrics,
                'config': self.checkpoint_config(),
            }
            if self.route_intent_head is not None:
                save_data['route_intent_head_state_dict'] = self.route_intent_head.state_dict()
            torch.save(save_data, path)
            logger.info(f"MAPPO model saved to {path}")
        except Exception as e:
            logger.error(f"Error saving MAPPO model: {e}")

    def load(self, path: str):
        """Load MAPPO model."""
        try:
            save_data = torch.load(path, map_location=self.device, weights_only=False)

            self.actor.load_state_dict(save_data['actor_state_dict'])
            self.critic.load_state_dict(save_data['critic_state_dict'])
            self.lstm_encoder.load_state_dict(save_data['lstm_encoder_state_dict'])
            if self.value_normalizer is not None and save_data.get('value_normalizer_state_dict'):
                self.value_normalizer.load_state_dict(save_data['value_normalizer_state_dict'])
            # Eval-only checkpoints carry no optimizer state, and a stale one
            # must not turn a successful weight load into a failed one.
            optimizer_state = save_data.get('optimizer_state_dict')
            if optimizer_state:
                try:
                    self.optimizer.load_state_dict(optimizer_state)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(f"MAPPO optimizer state not restored ({exc}); "
                                   "continuing with a fresh optimizer")
            self.episode_count = save_data.get('episode_count', 0)
            self.updates_performed = save_data.get('updates_performed', 0)
            self.steps_collected = save_data.get('steps_collected', self.steps_collected)

            if 'training_metrics' in save_data:
                self.training_metrics = save_data['training_metrics']
            if self.route_intent_head is not None and 'route_intent_head_state_dict' in save_data:
                self.route_intent_head.load_state_dict(save_data['route_intent_head_state_dict'])

            self._pretrained = True
            self._mark_acting_dirty()
            logger.info(f"MAPPO model loaded from {path}")
        except Exception as e:
            logger.error(f"Error loading MAPPO model: {e}")
