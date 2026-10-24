'''
Author       : AXIBA leolihao@arizona.edu
Date         : 2025-12-05
FilePath     : /OpenCDA-MARL/opencda_marl/builtin/algorithms/sac.py
Description  : Soft Actor-Critic (SAC) algorithm with auto-tuning entropy
               for multi-agent intersection control.

Reference    : "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning
                with a Stochastic Actor" - Haarnoja et al., ICML 2018 (arXiv:1801.01290)
               "Soft Actor-Critic Algorithms and Applications" - Haarnoja et al., 2018
               (arXiv:1812.05905)
               PCMASAC (pref_mode internalpref / internalpref_sep, vector critic,
               preference shift and separation): "Learning Coordinated Preference
               for Multi-Objective Multi-Agent Reinforcement Learning" - Wang,
               Guo, Xie, Liu, Cao, Chen, 2026 (arXiv:2606.14693); see
               docs/marl/pcma.md.

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
from .route_head import RouteHeadMixin
from .smart_replay_buffer import SmartReplayBuffer


def _uses_internal_preference_mode(pref_mode: str) -> bool:
    return pref_mode in {"internalpref", "internalpref_sep"}


class CudaGraphCaptureError(RuntimeError):
    """Raised when the opt-in CUDA-graph update could not be captured; the
    process's CUDA state is unusable for training afterwards."""



class SquashedGaussianActor(nn.Module):
    """
    Stochastic actor network for SAC with squashed Gaussian policy.

    Outputs mean and log_std for a Gaussian distribution, then applies
    tanh squashing to bound actions. Uses reparameterization trick for
    gradient backpropagation.

    Reference: SAC paper Section 4.2
    """

    def __init__(self, state_dim: int, action_dim: int,
                 hidden_dims: List[int] = [256, 256],
                 max_action: float = 65.0, min_action: float = 0.0,
                 log_std_min: float = -20.0, log_std_max: float = 2.0,
                 pref_mode: str = "direct", pref_dim: int = 0,
                 ego_state_dim: int | None = None, context_dim: int = 0,
                 pref_hidden_dim: int = 16,
                 pref_shift_clip: float = 1.0,
                 pref_shift_mix: float = 1.0):
        super(SquashedGaussianActor, self).__init__()

        self.max_action = max_action
        self.min_action = min_action
        self.action_scale = (max_action - min_action) / 2.0
        self.action_bias = (max_action + min_action) / 2.0
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.pref_mode = pref_mode
        self.pref_dim = int(pref_dim)
        self.ego_state_dim = int(ego_state_dim or state_dim)
        self.context_dim = int(context_dim)
        self.pref_hidden_dim = int(pref_hidden_dim)
        self.pref_shift_clip = max(0.0, float(pref_shift_clip))
        self.pref_shift_mix = float(np.clip(pref_shift_mix, 0.0, 1.0))

        feature_input_dim = state_dim
        actor_head_input_dim = hidden_dims[-1]

        if _uses_internal_preference_mode(self.pref_mode):
            if self.pref_dim <= 0:
                raise ValueError(f"{self.pref_mode} actor requires pref_dim > 0")
            if self.ego_state_dim <= self.pref_dim:
                raise ValueError(f"ego_state_dim must be larger than pref_dim for {self.pref_mode} actor")
            feature_input_dim = state_dim - self.pref_dim
            self.pref_shift_head = nn.Linear(hidden_dims[-1], self.pref_dim)
            self.pref_encoder = nn.Sequential(
                nn.Linear(self.pref_dim, self.pref_hidden_dim),
                nn.ReLU(),
            )
            actor_head_input_dim += self.pref_hidden_dim
        else:
            self.pref_shift_head = None
            self.pref_encoder = None

        # Build feature network
        layers = []
        in_dim = feature_input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim

        self.feature_net = nn.Sequential(*layers)

        # Separate heads for mean and log_std
        self.mean_head = nn.Linear(actor_head_input_dim, action_dim)
        self.log_std_head = nn.Linear(actor_head_input_dim, action_dim)

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights for stable training."""
        for layer in self.feature_net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.constant_(layer.bias, 0.0)

        if self.pref_shift_head is not None:
            nn.init.orthogonal_(self.pref_shift_head.weight, gain=0.01)
            nn.init.constant_(self.pref_shift_head.bias, 0.0)
        if self.pref_encoder is not None:
            for layer in self.pref_encoder:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                    nn.init.constant_(layer.bias, 0.0)

        # Small initialization for output heads
        nn.init.orthogonal_(self.mean_head.weight, gain=0.01)
        nn.init.constant_(self.mean_head.bias, 0.0)
        nn.init.orthogonal_(self.log_std_head.weight, gain=0.01)
        nn.init.constant_(self.log_std_head.bias, 0.0)

    def forward(
        self,
        state: torch.Tensor,
        return_aux: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass to get mean and log_std.

        Parameters
        ----------
        state : torch.Tensor
            Input state [batch_size, state_dim]

        Returns
        -------
        mean : torch.Tensor
            Mean of Gaussian [batch_size, action_dim]
        log_std : torch.Tensor
            Log standard deviation [batch_size, action_dim]
        """
        if _uses_internal_preference_mode(self.pref_mode):
            features, aux = self._build_internalpref_features(state)
        else:
            features = self.feature_net(state)
            aux = {}

        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        # Clamp log_std for numerical stability
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        if return_aux:
            return mean, log_std, aux
        return mean, log_std

    def _build_internalpref_features(
        self,
        state: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        ego_obs_dim = self.ego_state_dim - self.pref_dim
        raw_obs = state[:, :ego_obs_dim]
        raw_pref = state[:, ego_obs_dim:self.ego_state_dim]
        context = state[:, self.ego_state_dim:]

        obs_input = torch.cat([raw_obs, context], dim=-1)
        obs_features = self.feature_net(obs_input)
        pref_shift = torch.tanh(self.pref_shift_head(obs_features))
        if self.pref_shift_clip < 1.0:
            pref_shift = torch.clamp(pref_shift, min=-self.pref_shift_clip, max=self.pref_shift_clip)
        shifted_pref = torch.clamp(raw_pref + pref_shift, min=1e-6)
        shifted_pref = shifted_pref / shifted_pref.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        effective_pref = ((1.0 - self.pref_shift_mix) * raw_pref) + (self.pref_shift_mix * shifted_pref)
        effective_pref = effective_pref / effective_pref.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        pref_features = self.pref_encoder(effective_pref)

        fused_features = torch.cat([obs_features, pref_features], dim=-1)
        aux = {
            "raw_pref": raw_pref,
            "pref_shift": pref_shift,
            "shifted_pref": shifted_pref,
            "effective_pref": effective_pref,
        }
        return fused_features, aux

    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        action, log_prob, mean_action, _ = self.sample_with_aux(state)
        return action, log_prob, mean_action

    def sample_with_aux(
        self,
        state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Sample action using reparameterization trick with tanh squashing.

        Parameters
        ----------
        state : torch.Tensor
            Input state

        Returns
        -------
        action : torch.Tensor
            Squashed action in [min_action, max_action]
        log_prob : torch.Tensor
            Log probability of action (corrected for tanh)
        mean : torch.Tensor
            Mean action (deterministic)
        """
        mean, log_std, aux = self.forward(state, return_aux=True)
        std = log_std.exp()

        # Sample from Gaussian using reparameterization trick. Argument
        # validation is off: it costs several device syncs per call
        # (constraint checks) and std = exp(clamped log_std) is positive by
        # construction.
        normal = Normal(mean, std, validate_args=False)
        x_t = normal.rsample()  # Reparameterized sample

        # Apply tanh squashing
        y_t = torch.tanh(x_t)

        # Scale to action bounds: [-1, 1] -> [min_action, max_action]
        action = y_t * self.action_scale + self.action_bias

        # Compute log probability with tanh correction
        # log_prob = log_pi(a|s) = log_pi(u|s) - sum(log(1 - tanh(u)^2))
        # where u is pre-tanh action
        log_prob = normal.log_prob(x_t)
        # Tanh correction (numerically stable version)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        # Mean action (deterministic)
        mean_action = torch.tanh(mean) * self.action_scale + self.action_bias

        return action, log_prob, mean_action, aux

    def get_action(self, state: torch.Tensor, deterministic: bool = False
                   ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get action for inference.

        Parameters
        ----------
        state : torch.Tensor
            Input state
        deterministic : bool
            If True, return mean action

        Returns
        -------
        action : torch.Tensor
            Action value
        log_prob : torch.Tensor
            Log probability
        """
        action, log_prob, mean_action = self.sample(state)
        if deterministic:
            return mean_action, log_prob
        return action, log_prob


class TwinQNetwork(nn.Module):
    """
    Twin Q-networks for SAC (reduces overestimation bias).

    Takes state and action as input, outputs Q-values from both networks.
    """

    def __init__(self, state_dim: int, action_dim: int,
                 hidden_dims: List[int] = [256, 256], output_dim: int = 1):
        super(TwinQNetwork, self).__init__()

        input_dim = state_dim + action_dim
        self.output_dim = int(output_dim)

        # Q1 network
        q1_layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            q1_layers.append(nn.Linear(in_dim, h_dim))
            q1_layers.append(nn.ReLU())
            in_dim = h_dim
        q1_layers.append(nn.Linear(hidden_dims[-1], self.output_dim))
        self.q1 = nn.Sequential(*q1_layers)

        # Q2 network
        q2_layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            q2_layers.append(nn.Linear(in_dim, h_dim))
            q2_layers.append(nn.ReLU())
            in_dim = h_dim
        q2_layers.append(nn.Linear(hidden_dims[-1], self.output_dim))
        self.q2 = nn.Sequential(*q2_layers)

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize Q-network weights."""
        for network in [self.q1, self.q2]:
            for layer in network:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.constant_(layer.bias, 0)

    def forward(self, state: torch.Tensor, action: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through both Q-networks.

        Parameters
        ----------
        state : torch.Tensor
            State input [batch_size, state_dim]
        action : torch.Tensor
            Action input [batch_size, action_dim]

        Returns
        -------
        q1 : torch.Tensor
            Q-value from first network
        q2 : torch.Tensor
            Q-value from second network
        """
        x = torch.cat([state, action], dim=-1)
        return self.q1(x), self.q2(x)

    def Q1(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Get Q-value from first network only."""
        x = torch.cat([state, action], dim=-1)
        return self.q1(x)


class SACAlgorithm(RouteHeadMixin, BaseAlgorithm):
    """
    Soft Actor-Critic (SAC) algorithm with auto-tuning entropy.

    SAC maximizes both expected return and entropy, encouraging exploration
    while learning optimal policies. Key features:
    - Maximum entropy framework
    - Twin Q-networks (reduces overestimation)
    - Auto-tuning temperature (alpha)
    - Off-policy learning with replay buffer

    The objective is: J(pi) = E[sum(r + alpha * H(pi(.|s)))]
    where H is the entropy and alpha is automatically tuned.

    Reference
    ---------
    Haarnoja et al., "Soft Actor-Critic: Off-Policy Maximum Entropy Deep
    Reinforcement Learning with a Stochastic Actor", ICML 2018

    Parameters
    ----------
    config : Dict[str, Any]
        SAC configuration
    state_dim : int
        Individual agent observation dimension
    action_dim : int
        Action dimension (1 for continuous speed control)
    """

    #: IntentPolicy capability (RouteHeadMixin): the discrete SAC route head
    #: owns its own optimizers, unlike the PPO families.
    route_head_cls = "HybridRouteSACHead"

    def __init__(self, config: Dict[str, Any], state_dim: int, action_dim: int):
        super().__init__(config, state_dim, action_dim)

        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # =========== SAC Hyperparameters ===========
        self.tau = config.get('tau', 0.005)  # Soft update coefficient
        self.gamma = config.get('gamma', 0.99)  # Discount factor

        # Action bounds
        self.max_action = config.get('max_action', 65.0)
        self.min_action = config.get('min_action', 0.0)

        # Learning rates
        self.actor_lr = config.get('learning_rate_actor', 3e-4)
        self.critic_lr = config.get('learning_rate_critic', 3e-4)
        self.alpha_lr = config.get('learning_rate_alpha', 3e-4)

        # Entropy tuning
        self.auto_entropy_tuning = config.get('auto_entropy_tuning', True)
        # Note: textbook SAC default is -|A| but for action_dim=1 that drives
        # alpha into a death spiral on long training. We use -0.5*|A| as the
        # default; configs may override.
        self.target_entropy = config.get(
            'target_entropy', -0.5 * float(action_dim)
        )
        init_alpha = config.get('init_alpha', 0.2)
        # log_alpha clamp floor/ceiling — applied after every alpha update so
        # auto-entropy can't push alpha below ~0.05 (log -3) or above ~7.4
        # (log 2). Configurable; with these defaults alpha cannot collapse
        # toward zero over long training.
        self.log_alpha_min = float(config.get('log_alpha_min', -3.0))
        self.log_alpha_max = float(config.get('log_alpha_max', 2.0))
        # freeze_speed_head warm start: when True, update() is a no-op for
        # the speed head (actor/critic/alpha all frozen at loaded weights),
        # so the route head trains against a fixed speed policy.
        # Pair with MARL.training.load_checkpoint pointing at a trained
        # speed_only checkpoint so the frozen weights are useful, not random.
        self.freeze_speed_head = bool(config.get('freeze_speed_head', False))

        # Warmup and buffer
        self.warmup_steps = config.get('warmup_steps', 1000)
        self.memory_size = config.get('memory_size', 25000)
        self.batch_size = config.get('batch_size', 256)
        self.pref_mode = str(config.get('pref_mode', 'direct'))
        self.pref_dim = int(config.get('pref_dim', 2))
        self.pref_hidden_dim = int(config.get('pref_hidden_dim', 16))
        self.pref_shift_l2_coef = float(config.get('pref_shift_l2_coef', 0.0))
        self.pref_sep_coef = float(config.get('pref_sep_coef', 0.0))
        self.pref_sep_margin = float(config.get('pref_sep_margin', 0.25))
        self.pref_sep_use_effective_pref = bool(config.get('pref_sep_use_effective_pref', True))
        self.pref_update_mode = str(config.get('pref_update_mode', 'shifted_pref_actor'))
        self.pref_shift_clip = float(config.get('pref_shift_clip', 1.0))
        self.pref_shift_mix = float(config.get('pref_shift_mix', 1.0))
        self.critic_mode = str(config.get('critic_mode', 'scalar'))
        configured_reward_dim = int(config.get('reward_dim', 1))
        self.reward_dim = configured_reward_dim if self.critic_mode == "vector" else 1
        if self.pref_mode not in {"direct", "internalpref", "internalpref_sep"}:
            raise ValueError(f"Unsupported pref_mode: {self.pref_mode}")
        if self.critic_mode not in {"scalar", "vector"}:
            raise ValueError(f"Unsupported critic_mode: {self.critic_mode}")
        if self.critic_mode == "vector":
            if self.reward_dim <= 1:
                raise ValueError("Vector critic requires reward_dim > 1")
            if self.pref_dim <= 0:
                raise ValueError("Vector critic actor optimization requires pref_dim > 0")
            if self.pref_mode == "internalpref_sep":
                raise ValueError("internalpref_sep requires critic_mode: scalar")
        if self.pref_update_mode not in {
            "direct_pref_actor",
            "shifted_pref_actor",
            "shifted_pref_actor_detached",
        }:
            raise ValueError(f"Unsupported pref_update_mode: {self.pref_update_mode}")

        # Network architecture
        self.lstm_hidden_size = config.get('lstm_hidden_size', 128)
        self.lstm_num_layers = config.get('lstm_num_layers', 1)
        actor_hidden_dims = config.get('actor_hidden_dims', [256, 256])
        critic_hidden_dims = config.get('critic_hidden_dims', [256, 256])

        # Log std bounds for actor
        log_std_min = config.get('log_std_min', -20)
        log_std_max = config.get('log_std_max', 2)

        # =========== Networks ===========
        # LSTM encoder for multi-agent context
        self.lstm_encoder = self._build_context_encoder(state_dim)

        # Combined state dim for actor/critic: ego_state + lstm_context
        combined_state_dim = state_dim + self.lstm_hidden_size

        # Actor (squashed Gaussian policy)
        self.actor = SquashedGaussianActor(
            state_dim=combined_state_dim,
            action_dim=action_dim,
            hidden_dims=actor_hidden_dims,
            max_action=self.max_action,
            min_action=self.min_action,
            log_std_min=log_std_min,
            log_std_max=log_std_max,
            pref_mode=self.pref_mode,
            pref_dim=self.pref_dim if _uses_internal_preference_mode(self.pref_mode) else 0,
            ego_state_dim=state_dim,
            context_dim=self.lstm_hidden_size,
            pref_hidden_dim=self.pref_hidden_dim,
            pref_shift_clip=self.pref_shift_clip,
            pref_shift_mix=self.pref_shift_mix,
        ).to(self.device)

        # Twin Q-networks
        self.critic = TwinQNetwork(
            state_dim=combined_state_dim,
            action_dim=action_dim,
            hidden_dims=critic_hidden_dims,
            output_dim=self.reward_dim,
        ).to(self.device)

        # Target Q-networks
        self.critic_target = TwinQNetwork(
            state_dim=combined_state_dim,
            action_dim=action_dim,
            hidden_dims=critic_hidden_dims,
            output_dim=self.reward_dim,
        ).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # =========== Context encoding policy ===========
        # The replay buffer stores each transition's raw neighbour sequence
        # (ego first, the others nearest-first) and update() encodes the
        # whole batch in one packed LSTM call — instead of two per-transition
        # forwards at storage time (a large share of an episode).
        # The stored width grows with the widest sequence seen (see
        # ``_seq_width``), so by default nothing is truncated and the learner
        # sees exactly the context the actor saw. ``max_context_agents``
        # (optional, default unset) caps both to the ego plus the nearest
        # ``max_context_agents - 1`` others — the cap applies to acting and
        # learning alike, so the two never diverge.
        # ``train_context_encoder`` additionally optimises the encoder with
        # the critic loss; default False keeps the encoder frozen exactly as
        # before, so existing checkpoints load unchanged.
        raw_cap = config.get('max_context_agents')
        self.max_context_agents: Optional[int] = int(raw_cap) if raw_cap else None
        if self.max_context_agents is not None and self.max_context_agents < 1:
            raise ValueError("max_context_agents must be >= 1 when set")
        self.train_context_encoder = bool(config.get('train_context_encoder', False))
        self._seq_width = 1  # rows per stored sequence; widened on demand

        # =========== CUDA-graph update (opt-in) ===========
        # The update is launch-bound (~500 small kernels at batch 256). With
        # ``cuda_graph_update: true`` the tensor-only core of update() —
        # actor/critic/alpha forward+backward+step, Polyak, metrics gather —
        # is captured once into a CUDA graph and replayed each update; the
        # data-dependent packed-LSTM encode stays eager and feeds static
        # input tensors. Requires CUDA and the frozen encoder
        # (train_context_encoder off: the encoder's backward would have to
        # be inside the graph). Optimizers become ``capturable`` (state on
        # the device). Opt-in: a capture that fails leaves the CUDA RNG in
        # capture mode for the whole process (torch cannot unwind it), so a
        # failure is raised loudly rather than hidden — enable it in training
        # configs, where nothing else builds autograd graphs through the SAC
        # networks between updates (the one condition capture needs; every
        # acting path here runs under no_grad). Verified equal to the eager
        # update; it removes most of the per-update launch overhead.
        self.cuda_graph_update = bool(config.get('cuda_graph_update', False))
        if self.cuda_graph_update and (self.device.type != 'cuda' or self.train_context_encoder):
            logger.warning("SAC cuda_graph_update needs CUDA and train_context_encoder=false; running eagerly")
            self.cuda_graph_update = False
        self._update_graph = None   # captured state, see _capture_update_graph

        # =========== Optimizers ===========
        self.actor_optimizer = self._make_adam(self.actor.parameters(), self.actor_lr)
        critic_params = list(self.critic.parameters())
        if self.train_context_encoder:
            critic_params += list(self.lstm_encoder.parameters())
        self.critic_optimizer = self._make_adam(critic_params, self.critic_lr)

        # Auto-tuning alpha (temperature)
        if self.auto_entropy_tuning:
            self.log_alpha = torch.tensor(np.log(init_alpha), requires_grad=True,
                                          device=self.device, dtype=torch.float32)
            self.alpha_optimizer = self._make_adam([self.log_alpha], self.alpha_lr)
            self.alpha = self.log_alpha.exp().item()
        else:
            self.alpha = init_alpha
        # Device-side copy of alpha for the fixed-temperature case (the
        # update core is tensor-only so it can be graph-captured).
        self._alpha_const = torch.tensor(float(self.alpha), device=self.device)

        # =========== Replay Buffer ===========
        self.memory = SmartReplayBuffer(capacity=self.memory_size, recency_ratio=0.5)

        # =========== Training State ===========
        self.training = True
        self.training_step = 0
        self._pretrained = False

        # Training metrics
        self.training_metrics = {
            'actor_loss': 0.0,
            'critic_loss': 0.0,
            'alpha_loss': 0.0,
            'alpha': self.alpha,
            'q1_mean': 0.0,
            'q2_mean': 0.0,
            'entropy': 0.0,
            'memory_size': 0,
            'pref_shift_l2': 0.0,
            'pref_shift_norm': 0.0,
            'pref_shift_abs_max': 0.0,
            'pref_sep_loss': 0.0,
            'pref_sep_active_pairs': 0.0,
        }

        logger.info(f"SAC initialized: state_dim={state_dim}, action_dim={action_dim}, "
                   f"device={self.device}, pref_mode={self.pref_mode}, "
                   f"critic_mode={self.critic_mode}, reward_dim={self.reward_dim}, "
                   f"pref_update_mode={self.pref_update_mode}")
        logger.info(f"SAC hyperparameters: tau={self.tau}, gamma={self.gamma}, "
                   f"auto_entropy={self.auto_entropy_tuning}, target_entropy={self.target_entropy}")

    def select_action(self, multi_agent_obs: Dict[str, np.ndarray],
                      ego_agent_id: str, training: bool = True) -> Optional[float]:
        """
        Select action for ego agent using SAC policy.

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
            # Warmup phase: random exploration
            if training and len(self.memory) < self.warmup_steps and not self._pretrained:
                if len(self.memory) % 500 == 0:
                    logger.debug(f"SAC Warmup: {len(self.memory)}/{self.warmup_steps}")
                return np.random.uniform(self.min_action, self.max_action)

            # Prepare inputs
            ego_state, multi_agent_context = self._prepare_inputs(
                multi_agent_obs, ego_agent_id
            )

            # Combine ego state and multi-agent context
            combined_state = torch.cat([ego_state, multi_agent_context], dim=1)

            with torch.no_grad():
                if training:
                    action, log_prob, _ = self.actor.sample(combined_state)
                else:
                    _, _, action = self.actor.sample(combined_state)  # Deterministic (mean)

                final_speed = action.squeeze().item()
                final_speed = np.clip(final_speed, self.min_action, self.max_action)

                # Debug logging (periodic)
                if len(self.memory) % 2000 == 0:
                    logger.debug(f"SAC Action: speed={final_speed:.1f}, alpha={self.alpha:.4f}")

                return final_speed

        except Exception as e:
            logger.error(f"Error in SAC action selection: {e}")
            import traceback
            traceback.print_exc()
            return None

    # Batched action support (see BaseAlgorithm.select_actions).
    # SAC is off-policy and stochastic-policy, so we don't need to stash
    # per-agent log_prob/value during warmup -- the base class default
    # (uniform random in [min_action, max_action]) is sufficient.
    supports_batched_actions = True

    def _acting_modules(self):
        return {"actor": self.actor, "lstm_encoder": self.lstm_encoder}

    def _batched_forward(
        self,
        ego_states: torch.Tensor,
        context: torch.Tensor,
        resolved_ids: List[str],
        training: bool,
    ) -> Dict[str, float]:
        """One actor.sample pass over all live ego agents.

        Twin Q critics are not invoked at action-selection time (only at
        update), so this collapses to a single actor forward.
        """
        combined_state = torch.cat([ego_states, context], dim=1)
        actor = self._acting_module("actor")
        with torch.no_grad():
            if training:
                action, _, _ = actor.sample(combined_state)
            else:
                _, _, action = actor.sample(combined_state)
        speeds = action.squeeze(-1).detach().cpu().numpy()
        speeds = np.clip(speeds, self.min_action, self.max_action)
        return {
            agent_id: float(speed)
            for agent_id, speed in zip(resolved_ids, speeds)
        }

    def store_transition(self, multi_agent_obs: Dict[str, np.ndarray],
                        ego_agent_id: str, action: float, reward: float | np.ndarray | List[float],
                        next_multi_agent_obs: Dict[str, np.ndarray], done: bool):
        """
        Store transition in replay buffer.

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
            # Store the RAW neighbour sequences; update() encodes the whole
            # batch in one packed LSTM call. No network runs here at all.
            seq, seq_len = self._context_sequence(multi_agent_obs, ego_agent_id)
            next_seq, next_seq_len = self._context_sequence(next_multi_agent_obs, ego_agent_id)

            # Store transition
            if self.critic_mode == "vector":
                reward_value = np.asarray(reward, dtype=np.float32).reshape(self.reward_dim)
            else:
                reward_value = np.float32(reward)

            # Field layout (SmartReplayBuffer preallocates fixed shapes; the
            # sequence width W = self._seq_width grows on demand):
            #   0 ego_state [state_dim]      1 seq [W, state_dim]   2 seq_len
            #   3 action                     4 reward
            #   5 next_ego_state [state_dim] 6 next_seq [W, state_dim] 7 next_seq_len
            #   8 done
            self._sync_seq_width()
            widest = max(seq_len, next_seq_len)
            if widest > self._seq_width:
                self._widen_sequences(widest)
            self.memory.push(
                seq[0].copy(),
                self._pad_sequence(seq),
                int(seq_len),
                action,
                reward_value,
                next_seq[0].copy(),
                self._pad_sequence(next_seq),
                int(next_seq_len),
                done
            )

            # Log buffer stats periodically
            if len(self.memory) % 5000 == 0:
                logger.debug(f"SAC Buffer: {len(self.memory)}/{self.memory_size}")

        except Exception as e:
            logger.error(f"Error storing SAC transition: {e}")

    def update(self) -> Dict[str, float]:
        """
        Perform SAC update.

        If ``self.freeze_speed_head`` is set (frozen-speed warm start),
        the speed actor, critic, and alpha optimizer steps are skipped --
        the speed head is held at its loaded weights and only the route
        head (if enabled) trains. Returns early metrics in that case.

        Returns
        -------
        metrics : Dict[str, float]
            Training metrics from update
        """
        # freeze_speed_head warm start: speed head frozen, only the route
        # head trains. Caller still ticks the route head separately via
        # update_route_intent.
        if getattr(self, "freeze_speed_head", False):
            return self.training_metrics

        # Only update if we have enough samples
        if len(self.memory) < self.batch_size:
            return self.training_metrics

        try:
            states, next_states, actions, rewards, dones = self._sample_update_batch()
            if self.cuda_graph_update:
                names, values_t = self._graphed_update_core(states, next_states, actions, rewards, dones)
            else:
                names, values_t = self._update_core(states, next_states, actions, rewards, dones)
            self.training_step += 1
            self._mark_acting_dirty()
            values = values_t.tolist()
            self.training_metrics.update(zip(names, values))
            self.training_metrics['memory_size'] = len(self.memory)
            self.alpha = self.training_metrics['alpha']
            self._log_update_metrics()
            return self.training_metrics

        except CudaGraphCaptureError:
            raise   # the process cannot continue training after this; do not hide it
        except Exception as e:
            logger.error(f"Error in SAC update: {e}")
            import traceback
            traceback.print_exc()
            return self.training_metrics

    def _make_adam(self, params, lr):
        """Adam; ``capturable`` (state on the device) when the update is
        graph-captured, since a captured optimizer.step() requires it."""
        params = list(params)
        if self.cuda_graph_update:
            return optim.Adam(params, lr=lr, capturable=True)
        return optim.Adam(params, lr=lr)

    def _invalidate_update_graph(self) -> None:
        """Drop the captured update graph. Call whenever tensors the graph
        holds by address are replaced: optimizer state loaded from a
        checkpoint / world-reload snapshot, a rebuilt alpha optimizer."""
        self._update_graph = None

    def _sample_update_batch(self):
        """Sample a batch and encode its contexts (eagerly): returns the
        static-shaped tensors the update core consumes —
        ``states [B, S+H]``, ``next_states``, ``actions [B,1]``,
        ``rewards [B,R]``, ``dones [B,1]``."""
        # Sample batch: one stacked array per field (layout documented in
        # store_transition), then one host->device copy per field.
        dev = self.device
        (ego_np, seq_np, len_np, act_np, rew_np,
         next_ego_np, next_seq_np, next_len_np, done_np) = self.memory.sample_fields(self.batch_size)

        ego_states = torch.from_numpy(np.ascontiguousarray(ego_np, dtype=np.float32)).to(dev)
        seqs = torch.from_numpy(np.ascontiguousarray(seq_np, dtype=np.float32)).to(dev)
        # Sequence lengths stay on the CPU: packing needs them there and
        # the ego-only mask is a single tiny copy.
        seq_lens = torch.from_numpy(np.ascontiguousarray(len_np, dtype=np.int64))
        actions = torch.from_numpy(
            np.ascontiguousarray(act_np, dtype=np.float32).reshape(-1, 1)).to(dev)
        if self.critic_mode == "vector":
            rewards_np = np.ascontiguousarray(rew_np, dtype=np.float32).reshape(-1, self.reward_dim)
        else:
            rewards_np = np.ascontiguousarray(rew_np, dtype=np.float32).reshape(-1, 1)
        rewards = torch.from_numpy(rewards_np).to(dev)
        next_ego_states = torch.from_numpy(np.ascontiguousarray(next_ego_np, dtype=np.float32)).to(dev)
        next_seqs = torch.from_numpy(np.ascontiguousarray(next_seq_np, dtype=np.float32)).to(dev)
        next_seq_lens = torch.from_numpy(np.ascontiguousarray(next_len_np, dtype=np.int64))
        dones = torch.from_numpy(
            np.ascontiguousarray(done_np, dtype=np.float32).reshape(-1, 1)).to(dev)

        # =========== Encode contexts (batched) ===========
        # ONE packed LSTM call for next-state and current-state
        # sequences ([next; current], 2B rows). The next-state context
        # feeds only the bootstrap target, so it is detached; the
        # current-state context carries gradient to the encoder only when
        # train_context_encoder is on.
        B = ego_states.size(0)
        all_seqs = torch.cat([next_seqs, seqs], dim=0)
        all_lens = torch.cat([next_seq_lens, seq_lens], dim=0)
        if self.train_context_encoder:
            all_contexts = self._encode_context_batch(all_seqs, all_lens)
        else:
            with torch.no_grad():
                all_contexts = self._encode_context_batch(all_seqs, all_lens)
        next_contexts = all_contexts[:B].detach()
        contexts = all_contexts[B:]

        # Combine states for networks
        states = torch.cat([ego_states, contexts], dim=1)
        next_states = torch.cat([next_ego_states, next_contexts], dim=1)
        return states, next_states, actions, rewards, dones

    def _update_core(self, states, next_states, actions, rewards, dones):
        """One SAC update from encoded batch tensors. Tensor-only on the
        device — no Python floats derived from tensors, no host syncs — so it
        can run eagerly or be captured into a CUDA graph. Returns the metric
        names and one stacked device tensor of their values."""
        B = states.size(0)
        # Temperature as a tensor: exp(log_alpha) is exactly the float
        # self.alpha holds at this point (both are set at the end of the
        # previous update); a Python float would be baked into a graph.
        if self.auto_entropy_tuning:
            alpha_t = self.log_alpha.detach().exp()
        else:
            alpha_t = self._alpha_const


        # =========== Actor forward, once ===========
        # The actor's parameters are the same for the bootstrap target
        # (next_states) and the actor loss (states) — the critic step in
        # between does not touch them — so both are computed in one
        # forward over [next; current]. The target half is detached; the
        # current half keeps its graph for the actor loss below. (When
        # the encoder trains, the actor sees the context detached: the
        # encoder learns from the critic objective only.)
        actor_states = states.detach() if self.train_context_encoder else states
        all_actions, all_log_probs, _, all_aux = self.actor.sample_with_aux(
            torch.cat([next_states, actor_states], dim=0))
        next_actions = all_actions[:B].detach()
        next_log_probs = all_log_probs[:B].detach()
        new_actions = all_actions[B:]
        log_probs = all_log_probs[B:]
        actor_aux = {k: (v[B:] if torch.is_tensor(v) and v.dim() > 0 and v.size(0) == 2 * B else v)
                     for k, v in all_aux.items()}

        # =========== Update Critics ===========
        with torch.no_grad():
            # Target Q-values (minimum of two targets for conservatism)
            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)

            # Soft Q-target with entropy
            target_q = rewards + (1 - dones) * self.gamma * (target_q - alpha_t * next_log_probs)

        # Current Q-values
        current_q1, current_q2 = self.critic(states, actions)

        # Critic loss (MSE)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        # Update critics (and the context encoder, when it trains)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        if self.train_context_encoder:
            nn.utils.clip_grad_norm_(self.lstm_encoder.parameters(), 1.0)
        self.critic_optimizer.step()

        # =========== Update Actor ===========
        # new_actions / log_probs / actor_aux come from the single actor
        # forward above; the critic used here is the just-updated one.
        states = actor_states
        q1_new, q2_new = self.critic(states, new_actions)
        q_new = torch.min(q1_new, q2_new)

        # Actor loss: maximize Q - alpha * log_prob
        if self.critic_mode == "vector":
            raw_preferences = self._extract_preferences_from_states(states)
            actor_preferences = self._resolve_actor_preferences(raw_preferences, actor_aux)
            scalarized_q = (actor_preferences * q_new).sum(dim=-1, keepdim=True)
            actor_loss = (alpha_t * log_probs - scalarized_q).mean()
        else:
            actor_loss = (alpha_t * log_probs - q_new).mean()
        zero = self._zero_scalar()
        pref_shift_l2 = pref_shift_norm = pref_shift_abs_max = zero
        pref_sep_loss = pref_sep_active_pairs = zero
        if _uses_internal_preference_mode(self.pref_mode):
            pref_shift = actor_aux.get("pref_shift")
            if pref_shift is not None:
                pref_shift_l2 = pref_shift.pow(2).mean()
                pref_shift_norm = pref_shift.norm(dim=-1).mean()
                pref_shift_abs_max = pref_shift.abs().max()
                actor_loss = actor_loss + self.pref_shift_l2_coef * pref_shift_l2
        if self.pref_mode == "internalpref_sep":
            raw_preferences = self._extract_preferences_from_states(states)
            pref_target = actor_aux.get("effective_pref") if self.pref_sep_use_effective_pref else actor_aux.get("shifted_pref")
            if pref_target is None:
                pref_target = actor_aux.get("shifted_pref")
            pref_sep_loss, pref_sep_active_pairs = self._compute_pref_sep_loss(raw_preferences, pref_target)
            actor_loss = actor_loss + self.pref_sep_coef * pref_sep_loss

        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()

        # =========== Update Alpha (Temperature) ===========
        if self.auto_entropy_tuning:
            # Alpha loss: minimize -alpha * (log_prob + target_entropy)
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            # Clamp log_alpha to a floor so auto-entropy can't death-spiral
            # alpha to ~0 over long training. With log(min)=-3,
            # alpha >= 0.05; with log(max)=2, alpha <= ~7.4.
            # Configurable via log_alpha_min/log_alpha_max; the defaults keep
            # the temperature in a stable range.
            log_alpha_min = self.log_alpha_min
            log_alpha_max = self.log_alpha_max
            with torch.no_grad():
                self.log_alpha.clamp_(min=log_alpha_min, max=log_alpha_max)
            alpha_after = self.log_alpha.detach().exp()
        else:
            alpha_loss = self._zero_scalar()
            alpha_after = alpha_t

        # =========== Soft Update Target Networks ===========
        self._soft_update_target()

        # =========== Metrics ===========
        # Every scalar the metrics dict / TensorBoard need is gathered on
        # the device and read back in ONE transfer by the caller.
        gathered: Dict[str, torch.Tensor] = {
            'actor_loss': actor_loss,
            'critic_loss': critic_loss,
            'alpha_loss': alpha_loss,
            'alpha': alpha_after,
            'q1_mean': current_q1.mean(),
            'q2_mean': current_q2.mean(),
            'entropy': -log_probs.mean(),
            'pref_shift_l2': pref_shift_l2,
            'pref_shift_norm': pref_shift_norm,
            'pref_shift_abs_max': pref_shift_abs_max,
            'pref_sep_loss': pref_sep_loss,
            'pref_sep_active_pairs': pref_sep_active_pairs,
        }
        if self.critic_mode == "vector":
            gathered['actor_q_scalarized_mean'] = scalarized_q.mean()
            gathered['q_efficiency_mean'] = current_q1[:, 0].mean()
            if self.reward_dim > 1:
                gathered['q_safety_mean'] = current_q1[:, 1].mean()
            for idx in range(self.reward_dim):
                gathered[f'q1_obj_{idx}_mean'] = current_q1[:, idx].mean()
                gathered[f'q2_obj_{idx}_mean'] = current_q2[:, idx].mean()
            if _uses_internal_preference_mode(self.pref_mode):
                for name, pref in (("raw", actor_aux.get("raw_pref")),
                                   ("effective", actor_aux.get("effective_pref")),
                                   ("shifted", actor_aux.get("shifted_pref"))):
                    if pref is None:
                        continue
                    gathered[f'{name}_pref_obj_0_mean'] = pref[:, 0].mean()
                    gathered[f'{name}_pref_obj_1_mean'] = pref[:, 1].mean()
                    gathered[f'actor_q_{name}pref_mean'] = (pref * q_new).sum(dim=-1, keepdim=True).mean()
        names = list(gathered.keys())
        return names, torch.stack([gathered[n].detach().reshape(()).float() for n in names])

    def _active_pref_loss_terms(self) -> tuple:
        """The ``pref_*`` actor-loss terms this configuration computes: the
        shift terms under an internal preference mode, the separation terms
        under ``internalpref_sep`` only. ``training_metrics`` still carries
        all five (zero when inactive)."""
        terms = ()
        if _uses_internal_preference_mode(self.pref_mode):
            terms += ('pref_shift_l2', 'pref_shift_norm', 'pref_shift_abs_max')
        if self.pref_mode == "internalpref_sep":
            terms += ('pref_sep_loss', 'pref_sep_active_pairs')
        return terms

    def _log_update_metrics(self) -> None:
        """TensorBoard logging from the metrics dict (no device work): gate
        per category BEFORE touching the writer."""
        m = self.training_metrics
        if self._should_log_category('losses', None):
            self.log_scalar('Loss/actor', m['actor_loss'], category='losses')
            self.log_scalar('Loss/critic', m['critic_loss'], category='losses')
            self.log_scalar('SAC/alpha', m['alpha'], category='losses')
            self.log_scalar('SAC/entropy', m['entropy'], category='losses')
            # The preference-shift and separation terms exist only under an
            # internal preference mode; elsewhere (plain MASAC, direct
            # preference input) they are constant zeros, so they are not logged.
            for key in self._active_pref_loss_terms():
                self.log_scalar(f'SAC/{key}', m[key], category='losses')
            if self.critic_mode == "vector":
                self.log_scalar('SAC/actor_q_scalarized_mean', m['actor_q_scalarized_mean'], category='losses')
                if _uses_internal_preference_mode(self.pref_mode):
                    for key in ('actor_q_rawpref_mean', 'actor_q_effectivepref_mean',
                                'actor_q_shiftedpref_mean',
                                'raw_pref_obj_0_mean', 'raw_pref_obj_1_mean',
                                'effective_pref_obj_0_mean', 'effective_pref_obj_1_mean',
                                'shifted_pref_obj_0_mean', 'shifted_pref_obj_1_mean'):
                        if key in m:
                            self.log_scalar(f'SAC/{key}', m[key], category='losses')
        if self._should_log_category('q_values', None):
            self.log_scalar('Q_values/Q1_mean', m['q1_mean'], category='q_values')
            self.log_scalar('Q_values/Q2_mean', m['q2_mean'], category='q_values')
            if self.critic_mode == "vector":
                for idx in range(self.reward_dim):
                    self.log_scalar(f'Q_values/Q1_obj_{idx}_mean', m[f'q1_obj_{idx}_mean'], category='q_values')
                    self.log_scalar(f'Q_values/Q2_obj_{idx}_mean', m[f'q2_obj_{idx}_mean'], category='q_values')
        self.log_scalar('Buffer/size', len(self.memory), category='buffer')

        # Log progress periodically
        if self.training_step % 200 == 0:
            logger.info(f"SAC Step {self.training_step}: "
                        f"critic_loss={m['critic_loss']:.4f}, "
                        f"actor_loss={m['actor_loss']:.4f}, "
                        f"alpha={self.alpha:.4f}, "
                        f"entropy={m['entropy']:.4f}")

    # ------------------------------------------------------------------ #
    # CUDA-graph update
    # ------------------------------------------------------------------ #
    def _graphed_update_core(self, states, next_states, actions, rewards, dones):
        """Run the update core through a captured CUDA graph, capturing it
        first (after warm-up updates on fresh batches) if necessary. Any
        capture problem disables graph mode and runs the core eagerly."""
        g = self._update_graph
        if g is None:
            try:
                g = self._capture_update_graph(states, next_states, actions, rewards, dones)
            except Exception as exc:
                # A failed capture leaves torch's CUDA generator in capture
                # mode and the capture stream current for the rest of the
                # process (graph.__exit__ never reaches its epilogue), so
                # eager updates fail from here on too. Stop loudly instead of
                # limping on with a learner that no longer learns.
                self.cuda_graph_update = False
                self._update_graph = None
                raise CudaGraphCaptureError(
                    "SAC cuda_graph_update: CUDA graph capture failed and the CUDA state "
                    "cannot be recovered in-process. Typical cause: an autograd graph "
                    "through the SAC actor/critic built outside update() (e.g. a forward "
                    "with grad enabled) was still alive at capture. Run with "
                    f"cuda_graph_update: false or remove that reference. ({exc})"
                ) from exc
            self._update_graph = g
        else:
            g['states'].copy_(states)
            g['next_states'].copy_(next_states)
            g['actions'].copy_(actions)
            g['rewards'].copy_(rewards)
            g['dones'].copy_(dones)
        g['graph'].replay()
        return g['names'], g['out']

    def _capture_update_graph(self, states, next_states, actions, rewards, dones):
        """Warm up on a side stream, then capture one update into a graph.

        Warm-up updates run on FRESH batches so they are ordinary gradient
        steps; capture records (does not execute) the core on static copies
        of the given batch, and one replay then performs this update. The
        static tensors are the graph's inputs from now on.
        """
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                self._update_core(*self._sample_update_batch())
        torch.cuda.current_stream().wait_stream(side)

        static = {
            'states': states.detach().clone(),
            'next_states': next_states.detach().clone(),
            'actions': actions.detach().clone(),
            'rewards': rewards.detach().clone(),
            'dones': dones.detach().clone(),
        }
        for opt in (self.critic_optimizer, self.actor_optimizer,
                    getattr(self, 'alpha_optimizer', None)):
            if opt is not None:
                opt.zero_grad(set_to_none=True)   # grads get allocated from the graph pool
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            names, out = self._update_core(static['states'], static['next_states'],
                                           static['actions'], static['rewards'], static['dones'])
        static.update(graph=graph, names=names, out=out)
        logger.info(f"SAC update captured into a CUDA graph ({len(names)} metrics, batch {states.size(0)})")
        return static

    def _soft_update_target(self):
        """Polyak update ``target <- (1 - tau) * target + tau * online``.

        One fused multi-tensor op pair for the whole parameter list instead
        of four kernel launches per parameter.
        """
        with torch.no_grad():
            targets = [p.data for p in self.critic_target.parameters()]
            sources = [p.data for p in self.critic.parameters()]
            torch._foreach_mul_(targets, 1.0 - self.tau)
            torch._foreach_add_(targets, sources, alpha=self.tau)

    def _zero_scalar(self) -> torch.Tensor:
        """A shared constant 0-d zero on the device (never mutated in place)."""
        zero = self.__dict__.get("_zero_scalar_cache")
        if zero is None or zero.device != self.device:
            zero = torch.zeros((), device=self.device)
            self._zero_scalar_cache = zero
        return zero

    def _extract_preferences_from_states(self, states: torch.Tensor) -> torch.Tensor:
        """Extract raw per-agent preferences from the ego-state slice."""
        if self.pref_dim <= 0:
            raise ValueError("Preference extraction requires pref_dim > 0")
        pref_start = self.state_dim - self.pref_dim
        preferences = states[:, pref_start:self.state_dim]
        if preferences.shape[-1] != self.pref_dim:
            raise ValueError(
                f"Expected preference dim {self.pref_dim}, got {preferences.shape[-1]}"
            )
        return preferences

    def _resolve_actor_preferences(
        self,
        raw_preferences: torch.Tensor,
        actor_aux: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Resolve which preference weights the actor should use to scalarize vector Q."""
        if _uses_internal_preference_mode(self.pref_mode):
            if self.pref_update_mode == "direct_pref_actor":
                return raw_preferences
            effective_pref = actor_aux.get("effective_pref")
            shifted_pref = actor_aux.get("shifted_pref")
            selected_pref = effective_pref if effective_pref is not None else shifted_pref
            if selected_pref is not None:
                if self.pref_update_mode == "shifted_pref_actor_detached":
                    return selected_pref.detach()
                return selected_pref
        return raw_preferences

    def _compute_pref_sep_loss(
        self,
        raw_preferences: torch.Tensor,
        effective_preferences: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if effective_preferences is None or raw_preferences.size(0) < 2:
            zero = self._zero_scalar()
            return zero, zero

        # Hinge on the distance between effective preferences of agents in
        # different preference groups (upper triangle, no self-pairs).
        # Computed as a masked sum / count so the whole thing stays on the
        # device: no boolean gather and no numel() sync per update. With no
        # active pair the loss is 0 and the count is 0, as before.
        group_labels = torch.argmax(raw_preferences, dim=-1)
        pairwise_distances = torch.cdist(effective_preferences, effective_preferences, p=2)
        pair_mask = group_labels.unsqueeze(0) != group_labels.unsqueeze(1)
        pair_mask = torch.triu(pair_mask, diagonal=1).to(pairwise_distances.dtype)

        hinge = F.relu(self.pref_sep_margin - pairwise_distances) * pair_mask
        active_pairs = pair_mask.sum()
        sep_loss = hinge.sum() / active_pairs.clamp_min(1.0)
        return sep_loss, active_pairs

    # _context_sequence / _encode_context_batch / _pad_sequence /
    # _widen_sequences live on BaseAlgorithm: the whole family stores raw
    # neighbour sequences and encodes them in one packed LSTM call at
    # update time.

    def _prepare_inputs(self, multi_agent_obs: Dict[str, np.ndarray],
                        ego_agent_id: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare inputs for SAC networks.

        Returns
        -------
        ego_state : torch.Tensor
            Ego agent features [1, state_dim]
        multi_agent_context : torch.Tensor
            LSTM encoding [1, lstm_hidden_size]
        """
        # One ordering for the whole family: ego first, then the others by
        # distance to the intersection, optionally capped — exactly the
        # sequence store_transition keeps for this transition.
        seq, seq_len = self._context_sequence(multi_agent_obs, ego_agent_id)
        ego_state = torch.as_tensor(seq[:1], dtype=torch.float32, device=self.device)

        if seq_len > 1:
            sequence_tensor = torch.as_tensor(seq[None], dtype=torch.float32, device=self.device)
            multi_agent_context = self.lstm_encoder(sequence_tensor)
        else:
            multi_agent_context = torch.zeros(1, self.lstm_hidden_size, device=self.device)

        return ego_state, multi_agent_context

    # SAC-specific _prepare_batch_inputs was removed in the unified-
    # batched-action refactor; the base-class implementation
    # (BaseAlgorithm._prepare_batch_inputs) is used instead. It is
    # behaviour-equivalent for SAC's typical inputs and additionally
    # supports variable-length neighbour sequences via zero padding,
    # which is strictly safer.

    def reset_episode(self):
        """Reset for new episode."""
        self.episode_count += 1

        # Log to TensorBoard
        self.log_scalar('SAC/memory_size', len(self.memory), step=self.episode_count, category='episode')
        self.log_scalar('SAC/alpha', self.alpha, step=self.episode_count, category='episode')

        # GPU cleanup every few episodes
        if self.episode_count % 5 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        self.finalize_episode_logging()

    def get_training_info(self) -> Dict[str, Any]:
        """Get training information."""
        return {
            'algorithm': 'SAC',
            'algorithm_type': 'soft_actor_critic',
            'episode_count': self.episode_count,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'buffer_size': len(self.memory),
            'training_step': self.training_step,
            'tau': self.tau,
            'gamma': self.gamma,
            'alpha': self.alpha,
            'target_entropy': self.target_entropy,
            'actor_loss': self.training_metrics.get('actor_loss', 0.0),
            'critic_loss': self.training_metrics.get('critic_loss', 0.0),
            'entropy': self.training_metrics.get('entropy', 0.0),
            'epsilon': self.alpha,  # the exploration key other learners report
            'device': str(self.device),
            'max_action': self.max_action,
            'warmup_steps': self.warmup_steps,
            'in_warmup': len(self.memory) < self.warmup_steps and not self._pretrained,
            'pref_mode': self.pref_mode,
            'critic_mode': self.critic_mode,
            'pref_update_mode': self.pref_update_mode,
            'reward_dim': self.reward_dim,
            'pref_shift_l2': self.training_metrics.get('pref_shift_l2', 0.0),
            'pref_shift_norm': self.training_metrics.get('pref_shift_norm', 0.0),
            'pref_shift_abs_max': self.training_metrics.get('pref_shift_abs_max', 0.0),
            'pref_sep_loss': self.training_metrics.get('pref_sep_loss', 0.0),
            'pref_sep_active_pairs': self.training_metrics.get('pref_sep_active_pairs', 0.0),
            'pref_shift_clip': self.pref_shift_clip,
            'pref_shift_mix': self.pref_shift_mix,
            'pref_sep_coef': self.pref_sep_coef,
            'pref_sep_margin': self.pref_sep_margin,
            'pref_sep_use_effective_pref': self.pref_sep_use_effective_pref,
            **({
                'hybrid_route_intent_enabled': True,
                'route_intent_buffer_size': len(self.route_intent_head.buffer),
                **self.route_intent_head.get_training_info(),
            } if self.route_intent_head is not None else {}),
        }

    def log_episode_metrics(self, episode_reward: float, episode_length: int,
                           success_rate: float = 0.0, collision_rate: float = 0.0,
                           near_miss_count: int = 0, ttc_violation_rate: float = 0.0,
                           additional_metrics: Dict[str, float] = None,
                           traffic_metrics: Dict[str, float] = None):
        """Log episode-level metrics to TensorBoard."""
        # Add SAC-specific metrics
        sac_metrics = {
            'alpha': self.alpha,
            'training_step': self.training_step,
        }
        if additional_metrics:
            sac_metrics.update(additional_metrics)

        # Call base class method
        super().log_episode_metrics(
            episode_reward, episode_length, success_rate, collision_rate,
            near_miss_count=near_miss_count,
            ttc_violation_rate=ttc_violation_rate,
            additional_metrics=sac_metrics,
            traffic_metrics=traffic_metrics
        )

    def save(self, path: str):
        """Save SAC model."""
        try:
            save_data = {
                'actor_state_dict': self.actor.state_dict(),
                'critic_state_dict': self.critic.state_dict(),
                'critic_target_state_dict': self.critic_target.state_dict(),
                'lstm_encoder_state_dict': self.lstm_encoder.state_dict(),
                'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
                'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
                'episode_count': self.episode_count,
                'training_step': self.training_step,
                'training_metrics': self.training_metrics,
                'alpha': self.alpha,
                'config': self.checkpoint_config(),
            }

            if self.auto_entropy_tuning:
                save_data['log_alpha'] = self.log_alpha.detach().cpu()
                save_data['alpha_optimizer_state_dict'] = self.alpha_optimizer.state_dict()
            if self.route_intent_head is not None:
                save_data['route_intent_head_state_dict'] = self.route_intent_head.state_dict()

            torch.save(save_data, path)
            logger.info(f"SAC model saved to {path}")
        except Exception as e:
            logger.error(f"Error saving SAC model: {e}")

    def load(self, path: str):
        """Load SAC model with graceful per-piece degradation.

        Each component is loaded independently; mismatches log a warning
        and the affected piece stays at init (rather than crashing the
        whole load). Optimizer state is loaded only if param-group count
        matches. Designed for the freeze_speed_head warm start: a
        speed-only checkpoint into a learned-path SAC model (extra
        route-head params on this side).
        """
        try:
            save_data = torch.load(path, map_location=self.device, weights_only=False)

            def _safe_load(module, key, label):
                if key not in save_data:
                    logger.warning(f"SAC load: {label} key '{key}' absent in checkpoint; skipping")
                    return
                try:
                    try:
                        result = module.load_state_dict(save_data[key], strict=False)
                    except TypeError as exc:
                        if "strict" not in str(exc):
                            raise
                        result = module.load_state_dict(save_data[key])
                    missing, unexpected = result if result is not None else ([], [])
                    if missing or unexpected:
                        logger.info(f"SAC load: {label} loaded with {len(missing)} missing, {len(unexpected)} unexpected keys")
                except Exception as exc:
                    logger.warning(f"SAC load: {label} failed ({exc}); leaving at init")

            def _safe_load_opt(optimizer, key, label):
                if key not in save_data:
                    return
                try:
                    saved = save_data[key]
                    saved_n = len(saved.get("param_groups", []))
                    current_n = len(optimizer.param_groups)
                    if saved_n == current_n:
                        optimizer.load_state_dict(saved)
                    else:
                        logger.warning(
                            f"SAC load: {label} param-group count mismatch "
                            f"(saved={saved_n}, current={current_n}); skipping"
                        )
                except Exception as exc:
                    logger.warning(f"SAC load: {label} load failed ({exc}); leaving at init")

            _safe_load(self.actor, 'actor_state_dict', 'actor')
            _safe_load(self.critic, 'critic_state_dict', 'critic')
            _safe_load(self.critic_target, 'critic_target_state_dict', 'critic_target')
            _safe_load(self.lstm_encoder, 'lstm_encoder_state_dict', 'lstm_encoder')
            _safe_load_opt(self.actor_optimizer, 'actor_optimizer_state_dict', 'actor_optimizer')
            _safe_load_opt(self.critic_optimizer, 'critic_optimizer_state_dict', 'critic_optimizer')
            self.episode_count = save_data.get('episode_count', 0)
            self.training_step = save_data.get('training_step', 0)
            self.alpha = save_data.get('alpha', 0.2)

            if 'training_metrics' in save_data:
                self.training_metrics = save_data['training_metrics']

            if self.auto_entropy_tuning and 'log_alpha' in save_data:
                self.log_alpha = save_data['log_alpha'].to(self.device).requires_grad_(True)
                # If the checkpoint pre-dates the log_alpha clamp (e.g.
                # one whose alpha had collapsed toward zero), clamp now so
                # the first forward pass already operates on a sane
                # temperature. Subsequent updates also clamp; this guards
                # the load->step gap.
                with torch.no_grad():
                    self.log_alpha.clamp_(min=self.log_alpha_min, max=self.log_alpha_max)
                self.alpha = self.log_alpha.exp().item()
                self.alpha_optimizer = self._make_adam([self.log_alpha], self.alpha_lr)
                _safe_load_opt(self.alpha_optimizer, 'alpha_optimizer_state_dict', 'alpha_optimizer')
            else:
                with torch.no_grad():
                    self._alpha_const.fill_(float(self.alpha))
            if self.route_intent_head is not None and 'route_intent_head_state_dict' in save_data:
                _safe_load(self.route_intent_head, 'route_intent_head_state_dict', 'route_intent_head')

            # Optimizer state (and possibly the alpha optimizer object) were
            # replaced: a captured update graph would still point at the
            # old tensors.
            self._invalidate_update_graph()
            self._pretrained = True
            self._mark_acting_dirty()
            logger.info(f"SAC model loaded from {path}")
        except Exception as e:
            logger.error(f"Error loading SAC model: {e}")
    supports_vector_rewards = True
