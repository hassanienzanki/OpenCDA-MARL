"""
Preference-conditioned MAPPO (PCMAPPO).

A separate learner from plain MAPPO, which it leaves untouched. Adds:
- direct preference conditioning
- internal preference shifts
- preference separation regularization
- optional vector critic auxiliary supervision

Reference: "Learning Coordinated Preference for Multi-Objective Multi-Agent
Reinforcement Learning" - Wang, Guo, Xie, Liu, Cao, Chen, 2026
(arXiv:2606.14693). The mechanism, its configuration keys and the metrics to
watch are documented in docs/marl/pcma.md.
"""

from __future__ import annotations

import gc
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from loguru import logger
from torch.distributions import Normal

from .base_algorithm import BaseAlgorithm
from .gae import compute_gae
from .route_head import RouteHeadMixin
from .encoders import LSTMEncoder
from .value_norm import ValueNormalizer


def _orthogonal_init(module: nn.Module, gain: float = np.sqrt(2.0)) -> None:
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        nn.init.constant_(module.bias, 0.0)


def _build_mlp(input_dim: int, hidden_dims: List[int]) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.ReLU())
        in_dim = hidden_dim
    return nn.Sequential(*layers)


def _normalize_preference_tensor(pref: torch.Tensor, min_pref: float) -> torch.Tensor:
    pref = torch.clamp(pref, min=min_pref)
    denom = pref.sum(dim=-1, keepdim=True).clamp_min(min_pref * pref.shape[-1])
    return pref / denom


def _to_host(actions, log_probs, scalar_values, vector_values=None):
    """The step's outputs as host arrays through ONE device-to-host
    transfer. Each ``.cpu()`` is a device sync, so three or four per step
    add up over an episode."""
    n = actions.shape[0]
    actions = actions.reshape(n, -1)
    parts = [actions, log_probs.reshape(n, 1), scalar_values.reshape(n, 1)]
    if vector_values is not None:
        parts.append(vector_values.reshape(n, -1))
    host = torch.cat([p.to(dtype=torch.float32) for p in parts], dim=1).detach().cpu().numpy()
    width = actions.shape[1]
    return (host[:, :width], host[:, width], host[:, width + 1],
            host[:, width + 2:] if vector_values is not None else None)


class PCMAPPORolloutBuffer:
    """On-policy buffer with scalar and optional vector critic targets."""

    def __init__(
        self,
        buffer_size: int,
        state_dim: int,
        context_dim: int,
        action_dim: int,
        pref_dim: int,
        *,
        gamma: float,
        gae_lambda: float,
        device: torch.device,
        reward_dim: int = 0,
    ):
        self.buffer_size = buffer_size
        self.state_dim = state_dim
        self.context_dim = context_dim
        self.action_dim = action_dim
        self.pref_dim = pref_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device
        self.reward_dim = reward_dim

        self.observations: List[np.ndarray] = []
        # Raw neighbour sequences ``[n_i, state_dim]`` (ego first); the
        # context is encoded from these at update time.
        self.sequences: List[np.ndarray] = []
        self.actions: List[np.ndarray] = []
        self.scalar_rewards: List[float] = []
        self.vector_rewards: List[np.ndarray] = []
        self.scalar_values: List[float] = []
        self.vector_values: List[np.ndarray] = []
        self.raw_preferences: List[np.ndarray] = []
        self.log_probs: List[float] = []
        self.dones: List[bool] = []
        # ``(episode, agent)`` per transition: GAE runs per trajectory.
        self.trajectory_ids: List[Any] = []

        self.advantages: Optional[torch.Tensor] = None
        self.scalar_returns: Optional[torch.Tensor] = None
        self.vector_returns: Optional[torch.Tensor] = None

    def add(
        self,
        *,
        observation: np.ndarray,
        sequence: np.ndarray,
        action: np.ndarray,
        scalar_reward: float,
        scalar_value: float,
        raw_preference: np.ndarray,
        log_prob: float,
        done: bool,
        trajectory_id: Any,
        vector_reward: np.ndarray | None = None,
        vector_value: np.ndarray | None = None,
    ) -> None:
        self._device_cache = None
        self.observations.append(observation.astype(np.float32, copy=True))
        self.sequences.append(np.asarray(sequence, dtype=np.float32))
        self.actions.append(action.astype(np.float32, copy=True))
        self.scalar_rewards.append(float(scalar_reward))
        self.scalar_values.append(float(scalar_value))
        self.raw_preferences.append(raw_preference.astype(np.float32, copy=True))
        self.log_probs.append(float(log_prob))
        self.dones.append(bool(done))
        self.trajectory_ids.append(trajectory_id)

        if vector_reward is not None:
            self.vector_rewards.append(np.asarray(vector_reward, dtype=np.float32))
        if vector_value is not None:
            self.vector_values.append(np.asarray(vector_value, dtype=np.float32))

    def compute_returns_and_advantages(
        self,
        *,
        last_scalar_values: Dict[Any, float] | None = None,
        last_vector_values: Dict[Any, np.ndarray] | None = None,
        normalize_advantages: bool = True,
    ) -> None:
        """Per-trajectory GAE (see :mod:`opencda_marl.builtin.algorithms.gae`).

        ``last_*_values`` map an open trajectory id to the critic's value of
        that agent's last next-state (bootstrap at rollout cut / time-limit);
        trajectories that ended with ``done`` need no entry.
        """
        if not self.scalar_rewards:
            return

        dones = np.asarray(self.dones, dtype=np.float32)
        advantages, scalar_returns = compute_gae(
            np.asarray(self.scalar_rewards, dtype=np.float32),
            np.asarray(self.scalar_values, dtype=np.float32),
            dones, gamma=self.gamma, gae_lambda=self.gae_lambda,
            trajectory_ids=self.trajectory_ids, last_values=last_scalar_values,
        )
        self.advantages = torch.from_numpy(advantages).float().to(self.device)
        self.scalar_returns = torch.from_numpy(scalar_returns).float().to(self.device)

        if normalize_advantages and len(advantages) > 1:
            self.advantages = (
                self.advantages - self.advantages.mean()
            ) / (self.advantages.std() + 1e-8)

        if self.reward_dim > 0 and self.vector_rewards and self.vector_values \
                and len(self.vector_rewards) == len(self.scalar_rewards) \
                and len(self.vector_values) == len(self.scalar_rewards):
            _, vector_returns = compute_gae(
                np.asarray(self.vector_rewards, dtype=np.float32),
                np.asarray(self.vector_values, dtype=np.float32),
                dones, gamma=self.gamma, gae_lambda=self.gae_lambda,
                trajectory_ids=self.trajectory_ids, last_values=last_vector_values,
            )
            self.vector_returns = torch.from_numpy(vector_returns).float().to(self.device)
        else:
            self.vector_returns = None

    def open_trajectories(self) -> List[Any]:
        """Trajectory ids whose last stored transition did not terminate."""
        last_done: Dict[Any, bool] = {}
        for trajectory_id, done in zip(self.trajectory_ids, self.dones):
            last_done[trajectory_id] = done
        return [trajectory_id for trajectory_id, done in last_done.items() if not done]

    def get_batches(self, batch_size: int, shuffle: bool = True):
        if self.advantages is None or self.scalar_returns is None:
            raise RuntimeError("compute_returns_and_advantages must run before get_batches")

        n_samples = len(self.observations)
        batch_size = min(max(1, batch_size), n_samples)

        tensors = self._device_tensors()
        observations, actions = tensors["observations"], tensors["actions"]
        raw_preferences, old_log_probs = tensors["raw_preferences"], tensors["old_log_probs"]
        sequences, lengths_t = tensors["sequences"], tensors["lengths"]

        indices = np.arange(n_samples)
        if shuffle:
            np.random.shuffle(indices)

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch_indices = indices[start:end]
            batch = {
                "observations": observations[batch_indices],
                "sequences": sequences[batch_indices],
                "lengths": lengths_t[batch_indices],
                "actions": actions[batch_indices],
                "raw_preferences": raw_preferences[batch_indices],
                "old_log_probs": old_log_probs[batch_indices],
                "advantages": self.advantages[batch_indices],
                "scalar_returns": self.scalar_returns[batch_indices],
            }
            if self.vector_returns is not None:
                batch["vector_returns"] = self.vector_returns[batch_indices]
            yield batch

    def _device_tensors(self) -> Dict[str, torch.Tensor]:
        """The rollout as device tensors, stacked and uploaded once per
        fill. ``get_batches`` runs once per PPO epoch, and re-stacking and
        re-uploading the whole rollout in every epoch would repeat the same
        copy; ``add`` and ``clear`` drop the cache."""
        n_samples = len(self.observations)
        cached = getattr(self, "_device_cache", None)
        if cached is not None and cached[0] == n_samples:
            return cached[1]
        # Pad the rollout's sequences once; index per mini-batch.
        width = max(s.shape[0] for s in self.sequences)
        padded = np.zeros((n_samples, width, self.sequences[0].shape[1]), dtype=np.float32)
        lengths = np.empty(n_samples, dtype=np.int64)
        for i, seq in enumerate(self.sequences):
            padded[i, :seq.shape[0]] = seq
            lengths[i] = seq.shape[0]
        tensors = {
            "observations": torch.from_numpy(np.stack(self.observations)).float().to(self.device),
            "actions": torch.from_numpy(np.stack(self.actions)).float().to(self.device),
            "raw_preferences": torch.from_numpy(np.stack(self.raw_preferences)).float().to(self.device),
            "old_log_probs": torch.tensor(self.log_probs, dtype=torch.float32, device=self.device),
            "sequences": torch.from_numpy(padded).to(self.device),
            "lengths": torch.from_numpy(lengths),   # CPU: packing needs it there
        }
        self._device_cache = (n_samples, tensors)
        return tensors

    def clear(self) -> None:
        self._device_cache = None
        self.observations.clear()
        self.sequences.clear()
        self.actions.clear()
        self.scalar_rewards.clear()
        self.vector_rewards.clear()
        self.scalar_values.clear()
        self.vector_values.clear()
        self.raw_preferences.clear()
        self.log_probs.clear()
        self.dones.clear()
        self.trajectory_ids.clear()
        self.advantages = None
        self.scalar_returns = None
        self.vector_returns = None

    def __len__(self) -> int:
        return len(self.observations)


class PreferenceConditionedGaussianActor(nn.Module):
    def __init__(
        self,
        *,
        state_dim: int,
        action_dim: int,
        pref_dim: int,
        hidden_dims: List[int],
        max_action: float,
        min_action: float,
        init_std: float,
        pref_conditioning: str,
        pref_hidden_dim: int,
        use_pref_shift_head: bool,
        pref_shift_min_pref: float,
        pref_shift_max_abs: float,
        pref_shift_mix: float,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.pref_dim = pref_dim
        self.base_state_dim = state_dim - pref_dim if pref_dim > 0 else state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.min_action = min_action
        self.pref_conditioning = str(pref_conditioning or "concat")
        self.use_pref_shift_head = bool(use_pref_shift_head and pref_dim > 0)
        self.pref_shift_min_pref = float(pref_shift_min_pref)
        self.pref_shift_max_abs = float(pref_shift_max_abs)
        self.pref_shift_mix = float(pref_shift_mix)
        hidden_dim = hidden_dims[-1]

        self.state_encoder = _build_mlp(self.base_state_dim, hidden_dims)
        self.pref_encoder = (
            _build_mlp(self.pref_dim, [pref_hidden_dim, hidden_dim]) if self.pref_dim > 0 else None
        )

        if self.pref_dim > 0 and self.pref_conditioning in {"concat", "pref_encoder_fusion"}:
            self.fusion = _build_mlp(hidden_dim * 2, [hidden_dim])
        elif self.pref_dim > 0 and self.pref_conditioning == "film":
            self.film_gamma = nn.Linear(hidden_dim, hidden_dim)
            self.film_beta = nn.Linear(hidden_dim, hidden_dim)
            self.film_out = _build_mlp(hidden_dim, [hidden_dim])
        else:
            self.fusion = None

        if self.use_pref_shift_head:
            self.pref_shift_head = nn.Linear(hidden_dim * 2, self.pref_dim)
        else:
            self.pref_shift_head = None

        self.action_mean = nn.Linear(hidden_dim, action_dim)
        self.action_log_std = nn.Parameter(torch.ones(action_dim) * np.log(init_std))
        self.log_std_min = np.log(0.01)
        self.log_std_max = np.log(2.0)

        self.apply(_orthogonal_init)
        nn.init.orthogonal_(self.action_mean.weight, gain=0.01)
        # The action is the target speed in km/h: start the mean mid-range
        # instead of at 0 km/h (a zero bias made the initial policy stop
        # every vehicle; see GaussianActor in mappo.py).
        nn.init.constant_(self.action_mean.bias, 0.5 * (self.min_action + self.max_action))
        if self.pref_shift_head is not None:
            nn.init.zeros_(self.pref_shift_head.weight)
            nn.init.zeros_(self.pref_shift_head.bias)

    def _split_state(self, observation: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor | None]:
        if self.pref_dim <= 0:
            return observation, None
        return observation[..., : self.base_state_dim], observation[..., self.base_state_dim :]

    def _effective_preference(
        self,
        raw_pref: torch.Tensor | None,
        state_features: torch.Tensor,
        pref_features: torch.Tensor | None,
    ) -> Tuple[torch.Tensor | None, Dict[str, torch.Tensor]]:
        aux: Dict[str, torch.Tensor] = {}
        if raw_pref is None or pref_features is None:
            return raw_pref, aux

        aux["raw_pref"] = raw_pref
        if not self.use_pref_shift_head or self.pref_shift_head is None:
            aux["effective_pref"] = raw_pref
            return raw_pref, aux

        shift_input = torch.cat([state_features, pref_features], dim=-1)
        pref_shift = torch.tanh(self.pref_shift_head(shift_input))
        if self.pref_shift_max_abs > 0.0:
            pref_shift = pref_shift * self.pref_shift_max_abs
        shifted_pref = _normalize_preference_tensor(raw_pref + pref_shift, self.pref_shift_min_pref)
        if self.pref_shift_mix < 1.0:
            effective_pref = _normalize_preference_tensor(
                (1.0 - self.pref_shift_mix) * raw_pref + self.pref_shift_mix * shifted_pref,
                self.pref_shift_min_pref,
            )
        else:
            effective_pref = shifted_pref
        aux["pref_shift"] = pref_shift
        aux["shifted_pref"] = shifted_pref
        aux["effective_pref"] = effective_pref
        return effective_pref, aux

    def _encode(self, observation: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        base_state, raw_pref = self._split_state(observation)
        state_features = self.state_encoder(base_state)

        if raw_pref is None or self.pref_encoder is None:
            return state_features, {}

        pref_features = self.pref_encoder(raw_pref)
        effective_pref, aux = self._effective_preference(raw_pref, state_features, pref_features)
        effective_pref = raw_pref if effective_pref is None else effective_pref
        effective_pref_features = self.pref_encoder(effective_pref)

        if self.pref_conditioning in {"concat", "pref_encoder_fusion"}:
            features = self.fusion(torch.cat([state_features, effective_pref_features], dim=-1))
        elif self.pref_conditioning == "film":
            gamma = self.film_gamma(effective_pref_features)
            beta = self.film_beta(effective_pref_features)
            features = self.film_out((1.0 + gamma) * state_features + beta)
        else:
            features = state_features

        if "effective_pref" not in aux:
            aux["effective_pref"] = effective_pref
        return features, aux

    def forward(self, observation: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        features, aux = self._encode(observation)
        action_mean = self.action_mean(features)
        log_std = torch.clamp(self.action_log_std, self.log_std_min, self.log_std_max)
        action_std = log_std.exp().expand_as(action_mean)
        return action_mean, action_std, aux

    def get_action(
        self,
        observation: torch.Tensor,
        *,
        deterministic: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        mean, std, aux = self.forward(observation)
        dist = Normal(mean, std, validate_args=False)  # std = exp(clamped log_std) > 0; validation costs device syncs
        # Returns the RAW (unclipped) action with its log-prob — the rollout
        # stores it and evaluate_actions re-scores it, so the PPO ratio is
        # exact; callers clip it for the environment.
        action = mean if deterministic else dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        aux["action_mean"] = mean
        aux["action_std"] = std
        return action, log_prob, aux

    def evaluate_actions(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        mean, std, aux = self.forward(observation)
        dist = Normal(mean, std, validate_args=False)  # std = exp(clamped log_std) > 0; validation costs device syncs
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        aux["action_mean"] = mean
        aux["action_std"] = std
        return log_prob, entropy, aux


class PreferenceConditionedCritic(nn.Module):
    def __init__(
        self,
        *,
        state_dim: int,
        context_dim: int,
        pref_dim: int,
        hidden_dims: List[int],
        pref_conditioning: str,
        pref_hidden_dim: int,
        critic_mode: str,
        reward_dim: int,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.context_dim = context_dim
        self.pref_dim = pref_dim
        self.base_state_dim = state_dim - pref_dim if pref_dim > 0 else state_dim
        self.pref_conditioning = str(pref_conditioning or "concat")
        self.critic_mode = str(critic_mode)
        self.reward_dim = int(reward_dim)

        hidden_dim = hidden_dims[-1]
        self.state_encoder = _build_mlp(self.base_state_dim, hidden_dims)
        self.pref_encoder = (
            _build_mlp(self.pref_dim, [pref_hidden_dim, hidden_dim]) if self.pref_dim > 0 else None
        )

        critic_input_dim = hidden_dim + context_dim
        if self.pref_dim > 0 and self.pref_conditioning == "concat":
            critic_input_dim += hidden_dim

        self.fusion = _build_mlp(critic_input_dim, [hidden_dim])
        self.scalar_head = nn.Linear(hidden_dim, 1)
        self.vector_head = nn.Linear(hidden_dim, self.reward_dim) if self.critic_mode == "vector" else None

        self.apply(_orthogonal_init)
        nn.init.orthogonal_(self.scalar_head.weight, gain=1.0)
        nn.init.constant_(self.scalar_head.bias, 0.0)
        if self.vector_head is not None:
            nn.init.orthogonal_(self.vector_head.weight, gain=1.0)
            nn.init.constant_(self.vector_head.bias, 0.0)

    def _split_state(self, observation: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor | None]:
        if self.pref_dim <= 0:
            return observation, None
        return observation[..., : self.base_state_dim], observation[..., self.base_state_dim :]

    def forward(
        self,
        observation: torch.Tensor,
        context: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor | None]:
        base_state, raw_pref = self._split_state(observation)
        state_features = self.state_encoder(base_state)
        inputs = [state_features, context]
        if raw_pref is not None and self.pref_encoder is not None and self.pref_conditioning == "concat":
            pref_features = self.pref_encoder(raw_pref)
            inputs.append(pref_features)
        features = self.fusion(torch.cat(inputs, dim=-1))
        scalar_value = self.scalar_head(features)
        vector_value = self.vector_head(features) if self.vector_head is not None else None
        return scalar_value, vector_value


class PCMAPPOAlgorithm(RouteHeadMixin, BaseAlgorithm):
    supports_vector_rewards = True

    #: IntentPolicy capability (RouteHeadMixin): the PPO route head trains
    #: through this controller's optimizer.
    route_head_cls = "HybridRoutePPOHead"
    route_head_shares_optimizer = True

    def __init__(self, config: Dict[str, Any], state_dim: int, action_dim: int):
        super().__init__(config, state_dim, action_dim)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.clip_param = config.get("clip_param", 0.1)
        self.ppo_epochs = config.get("ppo_epochs", 5)
        self.num_mini_batch = config.get("num_mini_batch", 4)
        self.entropy_coef = config.get("entropy_coef", 0.05)
        self.value_loss_coef = config.get("value_loss_coef", 0.5)
        self.max_grad_norm = config.get("max_grad_norm", 0.5)
        # freeze_speed_head warm start: when True, update() is a no-op for
        # the speed head (PPO actor/critic frozen at loaded weights). Route
        # head trains independently via update_route_intent. See sac.py for
        # mirror docs.
        self.freeze_speed_head = bool(config.get("freeze_speed_head", False))
        self.gamma = config.get("gamma", 0.99)
        self.gae_lambda = config.get("gae_lambda", 0.95)
        self.use_gae = config.get("use_gae", True)
        self.learning_rate = config.get("learning_rate", 3e-4)
        self.n_steps = config.get("n_steps", 128)
        self.batch_size = config.get("batch_size", 32)
        self.max_action = config.get("max_action", 45.0)
        self.min_action = config.get("min_action", 0.0)
        self.warmup_steps = config.get("warmup_steps", 0)
        self.share_policy = bool(config.get("share_policy", True))
        self.reward_aggregation = str(config.get("reward_aggregation", "team_mean"))

        self.pref_mode = str(config.get("pref_mode", "direct"))
        self.pref_dim = int(config.get("pref_dim", 2))
        self.pref_hidden_dim = int(config.get("pref_hidden_dim", 64))
        self.actor_pref_conditioning = str(config.get("actor_pref_conditioning", "pref_encoder_fusion"))
        self.critic_pref_conditioning = str(config.get("critic_pref_conditioning", "concat"))
        self.actor_use_pref_shift_head = bool(
            config.get("actor_use_pref_shift_head", self.pref_mode in {"internalpref", "internalpref_sep"})
        )
        self.pref_shift_l2_coef = float(config.get("pref_shift_l2_coef", 0.0))
        self.pref_sep_coef = float(config.get("pref_sep_coef", 0.0))
        self.pref_sep_margin = float(config.get("pref_sep_margin", 0.25))
        self.pref_shift_min_pref = float(config.get("pref_shift_min_pref", 1e-3))
        self.pref_shift_max_abs = float(config.get("pref_shift_max_abs", config.get("pref_shift_clip", 0.5)))
        self.pref_shift_mix = float(config.get("pref_shift_mix", 1.0))

        self.critic_mode = str(config.get("critic_mode", "scalar"))
        self.reward_dim = int(config.get("reward_dim", 1 if self.critic_mode != "vector" else 2))
        self.critic_vector_l2_coef = float(config.get("critic_vector_l2_coef", 1.0))

        self.lstm_hidden_size = config.get("lstm_hidden_size", 128)
        self.lstm_num_layers = config.get("lstm_num_layers", 1)
        self.lstm_encoder = self._build_context_encoder(state_dim)

        actor_hidden_dims = config.get("actor_hidden_dims", [256, 256])
        critic_hidden_dims = config.get("critic_hidden_dims", [256, 256])
        init_std = config.get("init_std", 0.3)

        self.actor = PreferenceConditionedGaussianActor(
            state_dim=state_dim,
            action_dim=action_dim,
            pref_dim=self.pref_dim,
            hidden_dims=actor_hidden_dims,
            max_action=self.max_action,
            min_action=self.min_action,
            init_std=init_std,
            pref_conditioning=self.actor_pref_conditioning,
            pref_hidden_dim=self.pref_hidden_dim,
            use_pref_shift_head=self.actor_use_pref_shift_head,
            pref_shift_min_pref=self.pref_shift_min_pref,
            pref_shift_max_abs=self.pref_shift_max_abs,
            pref_shift_mix=self.pref_shift_mix,
        ).to(self.device)

        self.critic = PreferenceConditionedCritic(
            state_dim=state_dim,
            context_dim=self.lstm_hidden_size,
            pref_dim=self.pref_dim,
            hidden_dims=critic_hidden_dims,
            pref_conditioning=self.critic_pref_conditioning,
            pref_hidden_dim=self.pref_hidden_dim,
            critic_mode=self.critic_mode,
            reward_dim=self.reward_dim,
        ).to(self.device)

        self.optimizer = optim.Adam(
            [
                {"params": self.actor.parameters()},
                {"params": self.critic.parameters()},
                {"params": self.lstm_encoder.parameters()},
            ],
            lr=self.learning_rate,
        )

        self.rollout_buffer = PCMAPPORolloutBuffer(
            buffer_size=self.n_steps * 4,
            state_dim=state_dim,
            context_dim=self.lstm_hidden_size,
            action_dim=action_dim,
            pref_dim=self.pref_dim,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            device=self.device,
            reward_dim=self.reward_dim if self.critic_mode == "vector" else 0,
        )

        # =========== Rollout / context policy ===========
        # A PPO update runs once every ``n_steps`` environment steps (the
        # strategy calls update() once per data-bearing step) provided at
        # least ``batch_size`` transitions were collected; the buffer is
        # cleared afterwards. Trajectories cut by the rollout boundary or an
        # episode time-limit bootstrap from the critic's value of the agent's
        # last next-state (``_open_next_obs``), not from zero.
        # The context is (re-)encoded from the stored raw sequences at
        # update time: under no_grad when ``train_context_encoder`` is off
        # (default — the encoder stays a fixed projection, as it effectively
        # always was), through the graph when on, in which case the encoder
        # learns from the critic/actor losses via the shared optimizer.
        raw_cap = config.get("max_context_agents")
        self.max_context_agents: Optional[int] = int(raw_cap) if raw_cap else None
        self.train_context_encoder = bool(config.get("train_context_encoder", False))
        self._steps_since_update = 0
        self._open_next_obs: Dict[Any, Tuple[Dict[str, np.ndarray], str]] = {}
        # Value normalisation (opt-in): scalar and vector critics regress
        # returns scaled by running mean/std; every value that enters GAE or
        # a bootstrap is de-normalised back to reward units (value_norm.py).
        self.value_normalization = bool(config.get("value_normalization", False))
        self.value_normalizer = ValueNormalizer(1, self.device) if self.value_normalization else None
        self.vector_value_normalizer = (
            ValueNormalizer(self.reward_dim, self.device)
            if self.value_normalization and self.critic_mode == "vector" else None)

        self.steps_collected = 0
        self._pretrained = False   # set by load(): skips the random warmup
        self.updates_performed = 0
        self.last_log_probs: Dict[str, float] = {}
        self.last_scalar_values: Dict[str, float] = {}
        self.last_vector_values: Dict[str, np.ndarray] = {}
        # Raw (unclipped) Gaussian sample per agent — what the rollout stores.
        self.last_raw_actions: Dict[str, float] = {}
        self.training_metrics = {
            "policy_loss": 0.0,
            "scalar_value_loss": 0.0,
            "vector_value_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "clip_fraction": 0.0,
            "pref_shift_l2": 0.0,
            "pref_shift_norm": 0.0,
            "pref_sep_loss": 0.0,
            "pref_sep_active_pairs": 0.0,
            "pref_sep_dist_mean": 0.0,
            "buffer_size": 0,
        }

        logger.info(
            "PCMAPPO initialized: "
            f"state_dim={state_dim}, action_dim={action_dim}, device={self.device}, "
            f"pref_mode={self.pref_mode}, critic_mode={self.critic_mode}, reward_dim={self.reward_dim}"
        )

    def _extract_ego_obs(self, multi_agent_obs: Dict[str, np.ndarray], ego_agent_id: str) -> np.ndarray:
        if ego_agent_id in multi_agent_obs:
            return np.asarray(multi_agent_obs[ego_agent_id], dtype=np.float32)
        return np.asarray(next(iter(multi_agent_obs.values())), dtype=np.float32)

    def _extract_raw_preference(self, ego_obs: np.ndarray) -> np.ndarray:
        if self.pref_dim <= 0:
            return np.zeros(0, dtype=np.float32)
        return np.asarray(ego_obs[-self.pref_dim :], dtype=np.float32)

    def _prepare_inputs(
        self,
        multi_agent_obs: Dict[str, np.ndarray],
        ego_agent_id: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Same ordering as the batched acting path and the stored sequences
        # (BaseAlgorithm._context_sequence): ego first, others nearest-first.
        seq, seq_len = self._context_sequence(multi_agent_obs, ego_agent_id)
        ego_state = torch.as_tensor(seq[:1], dtype=torch.float32, device=self.device)
        if seq_len > 1:
            context = self.lstm_encoder(torch.as_tensor(seq[None], dtype=torch.float32, device=self.device))
        else:
            context = torch.zeros(1, self.lstm_hidden_size, device=self.device)
        return ego_state, context

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

    def select_action(
        self,
        multi_agent_obs: Dict[str, np.ndarray],
        ego_agent_id: str,
        training: bool = True,
    ) -> Optional[float]:
        try:
            if training and self._ppo_in_warmup():
                random_speed = np.random.uniform(self.min_action, self.max_action)
                self.last_log_probs[ego_agent_id] = 0.0
                self.last_scalar_values[ego_agent_id] = 0.0
                if self.critic_mode == "vector":
                    self.last_vector_values[ego_agent_id] = np.zeros(self.reward_dim, dtype=np.float32)
                return float(random_speed)

            ego_obs, context = self._prepare_inputs(multi_agent_obs, ego_agent_id)
            with torch.no_grad():
                action, log_prob, _ = self.actor.get_action(ego_obs, deterministic=not training)
                scalar_value, vector_value = self._critic_values(ego_obs, context)

            self.last_log_probs[ego_agent_id] = float(log_prob.item())
            self.last_scalar_values[ego_agent_id] = float(scalar_value.item())
            if vector_value is not None:
                self.last_vector_values[ego_agent_id] = vector_value.squeeze(0).detach().cpu().numpy().astype(np.float32)
            self.last_raw_actions[ego_agent_id] = float(action.squeeze().item())

            final_speed = float(torch.clamp(action.squeeze(), self.min_action, self.max_action).item())
            return final_speed
        except Exception as exc:
            logger.error(f"Error in PCMAPPO action selection: {exc}")
            return None

    def _scalarize_reward_vector(self, reward_vector: np.ndarray, raw_preference: np.ndarray) -> float:
        if raw_preference.size == 0:
            return float(np.mean(reward_vector))
        return float(np.dot(raw_preference, reward_vector))

    # Batched action support (see BaseAlgorithm.select_actions).
    # Only the algorithm-specific hooks live here; the orchestration loop
    # (empty/warmup/try/forward/unpack/fallback) is shared in the base class.
    supports_batched_actions = True

    def _warmup_action_batch(
        self, ego_agent_ids: List[Any],
    ) -> Dict[str, float]:
        """Stash PPO per-agent placeholders so store_transition stays happy
        when warmup-phase random actions hit the rollout buffer next step."""
        results: Dict[str, float] = {}
        for agent_id in ego_agent_ids:
            key = str(agent_id)
            results[key] = float(np.random.uniform(self.min_action, self.max_action))
            self.last_log_probs[key] = 0.0
            self.last_scalar_values[key] = 0.0
            if self.critic_mode == "vector":
                self.last_vector_values[key] = np.zeros(self.reward_dim, dtype=np.float32)
        return results

    def _batched_forward(
        self,
        ego_states: torch.Tensor,
        context: torch.Tensor,
        resolved_ids: List[str],
        training: bool,
    ) -> Dict[str, float]:
        """One actor + critic pass over the full batch, plus per-agent stash.

        Mirrors :meth:`select_action` semantics: log_prob, scalar_value,
        and (if vector critic) vector_value are stashed by agent id so
        :meth:`store_transition` can pick them up next step.
        """
        actor = self._acting_module("actor")
        with torch.no_grad():
            actions, log_probs, _ = actor.get_action(
                ego_states, deterministic=not training,
            )
            scalar_values, vector_values = self._critic_values(ego_states, context)

        actions_np, log_probs_np, scalar_values_np, vector_values_np = _to_host(
            actions, log_probs, scalar_values, vector_values)

        results: Dict[str, float] = {}
        for i, agent_id in enumerate(resolved_ids):
            self.last_log_probs[agent_id] = float(
                log_probs_np[i].item() if log_probs_np.ndim > 0 else log_probs_np
            )
            self.last_scalar_values[agent_id] = float(
                scalar_values_np[i].item() if scalar_values_np.ndim > 0 else scalar_values_np
            )
            if vector_values_np is not None:
                self.last_vector_values[agent_id] = vector_values_np[i].astype(np.float32)
            a = actions_np[i]
            raw = float(a.flatten()[0]) if isinstance(a, np.ndarray) and a.shape else float(a)
            self.last_raw_actions[agent_id] = raw
            results[agent_id] = float(np.clip(raw, self.min_action, self.max_action))
        return results

    def store_transition(
        self,
        multi_agent_obs: Dict[str, np.ndarray],
        ego_agent_id: str,
        action: float,
        reward: Any,
        next_multi_agent_obs: Dict[str, np.ndarray],
        done: bool,
    ):
        try:
            # Warmup transitions come from the uniform-random policy with
            # placeholder log-probs: off-policy data that must not enter the
            # PPO rollout. Count them towards warmup and move on.
            if self._ppo_in_warmup():
                self.steps_collected += 1
                return

            # Raw neighbour sequence (ego first); no network runs here — the
            # context is encoded from it at update time.
            seq, _ = self._context_sequence(multi_agent_obs, ego_agent_id)
            ego_obs_np = seq[0]
            raw_preference = self._extract_raw_preference(ego_obs_np)

            log_prob = self.last_log_probs.get(ego_agent_id, 0.0)
            scalar_value = self.last_scalar_values.get(ego_agent_id, 0.0)
            vector_value = self.last_vector_values.get(ego_agent_id)
            stored_action = self.last_raw_actions.get(ego_agent_id, float(action))

            vector_reward = None
            if isinstance(reward, np.ndarray):
                vector_reward = np.asarray(reward, dtype=np.float32)
                scalar_reward = self._scalarize_reward_vector(vector_reward, raw_preference)
            elif isinstance(reward, (list, tuple)) and reward and isinstance(reward[0], (int, float, np.floating)):
                vector_reward = np.asarray(reward, dtype=np.float32)
                scalar_reward = self._scalarize_reward_vector(vector_reward, raw_preference)
            else:
                scalar_reward = float(reward)

            trajectory_id = (self.episode_count, str(ego_agent_id))
            self.rollout_buffer.add(
                observation=ego_obs_np,
                sequence=seq,
                action=np.asarray([float(stored_action)], dtype=np.float32),
                scalar_reward=scalar_reward,
                scalar_value=scalar_value,
                raw_preference=raw_preference if raw_preference.size > 0 else np.zeros(self.pref_dim, dtype=np.float32),
                log_prob=log_prob,
                done=done,
                trajectory_id=trajectory_id,
                vector_reward=vector_reward,
                vector_value=vector_value,
            )
            # Bootstrap bookkeeping: an open trajectory's last next-state is
            # what the critic values at the rollout cut; a terminal one needs
            # nothing (the done mask removes the bootstrap).
            if done:
                self._open_next_obs.pop(trajectory_id, None)
            else:
                self._open_next_obs[trajectory_id] = (next_multi_agent_obs, str(ego_agent_id))
            self.steps_collected += 1
        except Exception as exc:
            logger.error(f"Error storing PCMAPPO transition: {exc}")

    def _critic_values(self, ego_states: torch.Tensor, context: torch.Tensor):
        """Critic outputs in reward units (de-normalised when value
        normalisation is on) — what GAE and bootstraps consume. Inputs on
        the acting device go through the acting replica's critic and
        normaliser mirrors; inputs on the training device through the
        live ones."""
        replica = self.__dict__.get("_acting_replica")
        on_replica = bool(replica) and ego_states.device == replica.device
        critic = replica.modules["critic"] if on_replica else self.critic
        scalar, vector = critic(ego_states, context)
        value_normalizer = self._acting_normalizer("value", scalar)
        if value_normalizer is not None:
            scalar = value_normalizer.denormalize(scalar)
            vector_normalizer = self._acting_normalizer("vector_value", scalar)
            if vector is not None and vector_normalizer is not None:
                vector = vector_normalizer.denormalize(vector)
        return scalar, vector

    def _acting_modules(self):
        return {"actor": self.actor, "critic": self.critic, "lstm_encoder": self.lstm_encoder}

    def _acting_normalizers(self):
        return {"value": self.value_normalizer, "vector_value": self.vector_value_normalizer}

    def _bootstrap_values(self) -> Tuple[Dict[Any, float], Dict[Any, np.ndarray]]:
        """Critic values V(s_next) for every open trajectory in the rollout.

        Open trajectories of one step share the same next-state table, so
        the values come from one batched critic pass per distinct table
        (usually one or two).
        """
        open_ids = set(self.rollout_buffer.open_trajectories())
        by_table: Dict[int, Tuple[Dict[str, np.ndarray], List[Tuple[Any, str]]]] = {}
        for trajectory_id in open_ids:
            entry = self._open_next_obs.get(trajectory_id)
            if entry is None:
                continue
            table, agent_id = entry
            if not table or agent_id not in table:
                continue
            by_table.setdefault(id(table), (table, []))[1].append((trajectory_id, agent_id))

        scalar_values: Dict[Any, float] = {}
        vector_values: Dict[Any, np.ndarray] = {}
        with torch.no_grad():
            for table, members in by_table.values():
                ego_states, context, resolved = self._prepare_batch_inputs(
                    table, [agent_id for _, agent_id in members])
                scalar, vector = self._critic_values(ego_states, context)
                scalar_np = scalar.reshape(len(resolved), -1)[:, 0].cpu().numpy()
                vector_np = vector.cpu().numpy() if vector is not None else None
                for i, (trajectory_id, _) in enumerate(members):
                    scalar_values[trajectory_id] = float(scalar_np[i])
                    if vector_np is not None:
                        vector_values[trajectory_id] = vector_np[i].astype(np.float32)
        return scalar_values, vector_values

    def _compute_pref_sep_loss(
        self,
        raw_preferences: torch.Tensor,
        effective_preferences: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        zero = torch.zeros((), dtype=torch.float32, device=self.device)
        if raw_preferences.numel() == 0 or effective_preferences.numel() == 0:
            return zero, zero, zero

        pref_l1 = torch.cdist(raw_preferences, raw_preferences, p=1)
        pref_dist = torch.cdist(effective_preferences, effective_preferences, p=2)
        mask = torch.triu((pref_l1 > 1e-3).float(), diagonal=1)
        active_pairs = mask.sum()
        if active_pairs.item() <= 0:
            return zero, zero, zero

        pref_sep_loss = (torch.relu(self.pref_sep_margin - pref_dist) * mask).sum() / active_pairs
        pref_sep_dist_mean = (pref_dist * mask).sum() / active_pairs
        return pref_sep_loss, active_pairs, pref_sep_dist_mean

    def update(self) -> Dict[str, float]:
        # freeze_speed_head warm start: speed head frozen, only the route
        # head trains. The caller still ticks route_intent_head.update()
        # separately.
        if getattr(self, "freeze_speed_head", False):
            return self.training_metrics
        if self._ppo_in_warmup():
            return self.training_metrics
        # One PPO update per rollout of n_steps environment steps (the
        # strategy calls update() once per data-bearing step), never on
        # fewer than batch_size transitions.
        self._steps_since_update += 1
        if self._steps_since_update < self.n_steps or len(self.rollout_buffer) < self.batch_size:
            self.training_metrics["buffer_size"] = len(self.rollout_buffer)
            return self.training_metrics

        try:
            last_scalar_values, last_vector_values = self._bootstrap_values()
            self.rollout_buffer.compute_returns_and_advantages(
                last_scalar_values=last_scalar_values,
                last_vector_values=last_vector_values if self.critic_mode == "vector" else None,
                normalize_advantages=True,
            )

            total_policy_loss = 0.0
            total_scalar_value_loss = 0.0
            total_vector_value_loss = 0.0
            total_entropy = 0.0
            total_approx_kl = 0.0
            total_clip_fraction = 0.0
            total_pref_shift_l2 = 0.0
            total_pref_shift_norm = 0.0
            total_pref_sep_loss = 0.0
            total_pref_sep_active_pairs = 0.0
            total_pref_sep_dist_mean = 0.0
            num_updates = 0

            batch_size = max(len(self.rollout_buffer) // self.num_mini_batch, 1)
            for _ in range(self.ppo_epochs):
                for batch in self.rollout_buffer.get_batches(batch_size=batch_size):
                    observations = batch["observations"]
                    actions = batch["actions"]
                    old_log_probs = batch["old_log_probs"]
                    advantages = batch["advantages"]
                    scalar_returns = batch["scalar_returns"]
                    # Context from the stored raw sequences: through the
                    # encoder graph only when it is meant to train.
                    if self.train_context_encoder:
                        contexts = self._encode_context_batch(batch["sequences"], batch["lengths"])
                    else:
                        with torch.no_grad():
                            contexts = self._encode_context_batch(batch["sequences"], batch["lengths"])

                    new_log_probs, entropy, actor_aux = self.actor.evaluate_actions(observations, actions)
                    scalar_values, vector_values = self.critic(observations, contexts)
                    scalar_values = scalar_values.squeeze(-1)
                    if self.value_normalizer is not None:
                        # Critics learn in normalised space; statistics track
                        # this rollout's returns (scalar and per-objective).
                        self.value_normalizer.update(scalar_returns)
                        scalar_returns = self.value_normalizer.normalize(scalar_returns.unsqueeze(-1)).squeeze(-1)
                        if self.vector_value_normalizer is not None and "vector_returns" in batch:
                            self.vector_value_normalizer.update(batch["vector_returns"])
                            batch = dict(batch)
                            batch["vector_returns"] = self.vector_value_normalizer.normalize(batch["vector_returns"])

                    ratio = (new_log_probs - old_log_probs).exp()
                    surr1 = ratio * advantages
                    surr2 = torch.clamp(
                        ratio,
                        1.0 - self.clip_param,
                        1.0 + self.clip_param,
                    ) * advantages
                    policy_loss = -torch.min(surr1, surr2).mean()
                    scalar_value_loss = F.mse_loss(scalar_values, scalar_returns)
                    entropy_bonus = entropy.mean()

                    loss = policy_loss + self.value_loss_coef * scalar_value_loss - self.entropy_coef * entropy_bonus

                    vector_value_loss = torch.zeros((), dtype=torch.float32, device=self.device)
                    if self.critic_mode == "vector" and vector_values is not None and "vector_returns" in batch:
                        vector_value_loss = F.mse_loss(vector_values, batch["vector_returns"])
                        loss = loss + self.critic_vector_l2_coef * vector_value_loss

                    pref_shift_l2 = torch.zeros((), dtype=torch.float32, device=self.device)
                    pref_shift_norm = torch.zeros((), dtype=torch.float32, device=self.device)
                    if self.actor_use_pref_shift_head and "pref_shift" in actor_aux:
                        pref_shift = actor_aux["pref_shift"]
                        pref_shift_l2 = pref_shift.pow(2).mean()
                        pref_shift_norm = pref_shift.norm(dim=-1).mean()
                        loss = loss + self.pref_shift_l2_coef * pref_shift_l2

                    pref_sep_loss = torch.zeros((), dtype=torch.float32, device=self.device)
                    pref_sep_active_pairs = torch.zeros((), dtype=torch.float32, device=self.device)
                    pref_sep_dist_mean = torch.zeros((), dtype=torch.float32, device=self.device)
                    if self.pref_mode == "internalpref_sep" and self.pref_sep_coef > 0.0:
                        effective_preferences = actor_aux.get("effective_pref")
                        if effective_preferences is None:
                            effective_preferences = batch["raw_preferences"]
                        (
                            pref_sep_loss,
                            pref_sep_active_pairs,
                            pref_sep_dist_mean,
                        ) = self._compute_pref_sep_loss(batch["raw_preferences"], effective_preferences)
                        loss = loss + self.pref_sep_coef * pref_sep_loss

                    # Actor and critic(+encoder) clipped separately (MAPPO
                    # reference practice): with unnormalized returns the
                    # critic's gradient norm can be orders of magnitude above
                    # the actor's and one joint clip would starve the policy
                    # gradient (see mappo.py).
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                    nn.utils.clip_grad_norm_(
                        list(self.critic.parameters()) + list(self.lstm_encoder.parameters()),
                        self.max_grad_norm,
                    )
                    self.optimizer.step()

                    with torch.no_grad():
                        approx_kl = ((ratio - 1.0) - ratio.log()).mean().item()
                        clip_fraction = ((ratio - 1.0).abs() > self.clip_param).float().mean().item()

                    total_policy_loss += float(policy_loss.item())
                    total_scalar_value_loss += float(scalar_value_loss.item())
                    total_vector_value_loss += float(vector_value_loss.item())
                    total_entropy += float(entropy_bonus.item())
                    total_approx_kl += approx_kl
                    total_clip_fraction += clip_fraction
                    total_pref_shift_l2 += float(pref_shift_l2.item())
                    total_pref_shift_norm += float(pref_shift_norm.item())
                    total_pref_sep_loss += float(pref_sep_loss.item())
                    total_pref_sep_active_pairs += float(pref_sep_active_pairs.item())
                    total_pref_sep_dist_mean += float(pref_sep_dist_mean.item())
                    num_updates += 1

            if num_updates > 0:
                self.training_metrics = {
                    "policy_loss": total_policy_loss / num_updates,
                    "scalar_value_loss": total_scalar_value_loss / num_updates,
                    "vector_value_loss": total_vector_value_loss / num_updates,
                    "entropy": total_entropy / num_updates,
                    "approx_kl": total_approx_kl / num_updates,
                    "clip_fraction": total_clip_fraction / num_updates,
                    "pref_shift_l2": total_pref_shift_l2 / num_updates,
                    "pref_shift_norm": total_pref_shift_norm / num_updates,
                    "pref_sep_loss": total_pref_sep_loss / num_updates,
                    "pref_sep_active_pairs": total_pref_sep_active_pairs / num_updates,
                    "pref_sep_dist_mean": total_pref_sep_dist_mean / num_updates,
                    "buffer_size": len(self.rollout_buffer),
                }

            self.rollout_buffer.clear()
            self._open_next_obs.clear()
            self._steps_since_update = 0
            self.updates_performed += 1
            self._mark_acting_dirty()
            # Loss scalars are logged with training_step as the global
            # step; without this every PPO curve collapsed onto step 0.
            self.training_step += 1

            self.log_scalar("Loss/policy", self.training_metrics["policy_loss"], category="losses")
            self.log_scalar("Loss/scalar_value", self.training_metrics["scalar_value_loss"], category="losses")
            self.log_scalar("Loss/vector_value", self.training_metrics["vector_value_loss"], category="losses")
            self.log_scalar("Policy/entropy", self.training_metrics["entropy"], category="losses")
            self.log_scalar("Policy/approx_kl", self.training_metrics["approx_kl"], category="losses")
            self.log_scalar("Policy/clip_fraction", self.training_metrics["clip_fraction"], category="losses")
            self.log_scalar("Pref/shift_l2", self.training_metrics["pref_shift_l2"], category="losses")
            self.log_scalar("Pref/sep_loss", self.training_metrics["pref_sep_loss"], category="losses")

            if self.updates_performed % 10 == 0:
                logger.info(
                    f"PCMAPPO Update {self.updates_performed}: "
                    f"policy_loss={self.training_metrics['policy_loss']:.4f}, "
                    f"scalar_value_loss={self.training_metrics['scalar_value_loss']:.4f}, "
                    f"vector_value_loss={self.training_metrics['vector_value_loss']:.4f}, "
                    f"pref_sep_loss={self.training_metrics['pref_sep_loss']:.4f}"
                )

            if torch.cuda.is_available() and self.updates_performed % 10 == 0:
                torch.cuda.empty_cache()
                gc.collect()

            return self.training_metrics
        except Exception as exc:
            logger.error(f"Error in PCMAPPO update: {exc}")
            raise

    def reset_episode(self):
        self.episode_count += 1
        self.last_log_probs.clear()
        self.last_scalar_values.clear()
        self.last_vector_values.clear()
        self.last_raw_actions.clear()
        self.log_scalar("PCMAPPO/steps_collected", self.steps_collected, step=self.episode_count, category="episode")
        self.log_scalar("PCMAPPO/updates_performed", self.updates_performed, step=self.episode_count, category="episode")
        if self.episode_count % 5 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        self.finalize_episode_logging()

    def get_training_info(self) -> Dict[str, Any]:
        info = {
            "algorithm": "PCMAPPO",
            "algorithm_type": "preference_conditioned_multi_agent_ppo",
            "episode_count": self.episode_count,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "buffer_size": len(self.rollout_buffer),
            "steps_collected": self.steps_collected,
            "updates_performed": self.updates_performed,
            "steps_collected": self.steps_collected,
            "clip_param": self.clip_param,
            "entropy_coef": self.entropy_coef,
            "warmup_steps": self.warmup_steps,
            "pref_mode": self.pref_mode,
            "pref_dim": self.pref_dim,
            "actor_pref_conditioning": self.actor_pref_conditioning,
            "critic_pref_conditioning": self.critic_pref_conditioning,
            "actor_use_pref_shift_head": self.actor_use_pref_shift_head,
            "pref_shift_l2_coef": self.pref_shift_l2_coef,
            "pref_sep_coef": self.pref_sep_coef,
            "pref_sep_margin": self.pref_sep_margin,
            "critic_mode": self.critic_mode,
            "reward_dim": self.reward_dim,
            "critic_vector_l2_coef": self.critic_vector_l2_coef,
            "reward_aggregation": self.reward_aggregation,
            "policy_loss": self.training_metrics.get("policy_loss", 0.0),
            "scalar_value_loss": self.training_metrics.get("scalar_value_loss", 0.0),
            "vector_value_loss": self.training_metrics.get("vector_value_loss", 0.0),
            "entropy": self.training_metrics.get("entropy", 0.0),
            "device": str(self.device),
            "max_action": self.max_action,
        }
        if self.route_intent_head is not None:
            info["hybrid_route_intent_enabled"] = True
            info.update(self.route_intent_head.get_training_info())
            info["route_intent_buffer_size"] = len(self.route_intent_head.buffer)
        return info

    def log_episode_metrics(
        self,
        episode_reward: float,
        episode_length: int,
        success_rate: float = 0.0,
        collision_rate: float = 0.0,
        near_miss_count: int = 0,
        ttc_violation_rate: float = 0.0,
        additional_metrics: Dict[str, float] = None,
        traffic_metrics: Dict[str, float] = None,
    ):
        pcmappo_metrics = {
            "buffer_size": len(self.rollout_buffer),
            "updates_performed": self.updates_performed,
            "steps_collected": self.steps_collected,
        }
        if additional_metrics:
            pcmappo_metrics.update(additional_metrics)
        super().log_episode_metrics(
            episode_reward,
            episode_length,
            success_rate,
            collision_rate,
            near_miss_count=near_miss_count,
            ttc_violation_rate=ttc_violation_rate,
            additional_metrics=pcmappo_metrics,
            traffic_metrics=traffic_metrics,
        )

    def save(self, path: str):
        save_data = {
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "lstm_encoder_state_dict": self.lstm_encoder.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "value_normalizer_state_dict": (self.value_normalizer.state_dict()
                                            if self.value_normalizer is not None else None),
            "vector_value_normalizer_state_dict": (self.vector_value_normalizer.state_dict()
                                                   if self.vector_value_normalizer is not None else None),
            "episode_count": self.episode_count,
            "updates_performed": self.updates_performed,
            "steps_collected": self.steps_collected,
            "training_metrics": self.training_metrics,
            "config": self.checkpoint_config(),
        }
        if self.route_intent_head is not None:
            save_data["route_intent_head_state_dict"] = self.route_intent_head.state_dict()
        torch.save(save_data, path)
        logger.info(f"PCMAPPO model saved to {path}")

    def load(self, path: str):
        """Load PCMAPPO model with graceful per-piece degradation.

        Each component (actor/critic/encoder/optimizer/route_head) is
        loaded independently. If the checkpoint was saved from a
        different model variant (e.g., a freeze_speed_head warm start from
        a speed-only checkpoint into a learned-path model), pieces that
        don't match log a warning and are skipped rather than crashing the
        run.

        For model state_dicts we use strict=False so extra keys in
        either direction are tolerated. The optimizer state is loaded
        only if the saved param-group count matches the current
        optimizer's; otherwise it's skipped (no harm: PPO's
        first-update behavior is well-defined without optimizer
        history, and freeze_speed_head mode doesn't step the speed-head
        optimizer anyway).
        """
        save_data = torch.load(path, map_location=self.device, weights_only=False)

        def _safe_load(module, key, label):
            if key not in save_data:
                logger.warning(f"PCMAPPO load: {label} key '{key}' absent in checkpoint; skipping")
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
                    logger.info(f"PCMAPPO load: {label} loaded with {len(missing)} missing, {len(unexpected)} unexpected keys")
            except Exception as exc:
                logger.warning(f"PCMAPPO load: {label} failed ({exc}); leaving at init")

        _safe_load(self.actor, "actor_state_dict", "actor")
        _safe_load(self.critic, "critic_state_dict", "critic")
        _safe_load(self.lstm_encoder, "lstm_encoder_state_dict", "lstm_encoder")
        for normalizer, key in ((self.value_normalizer, "value_normalizer_state_dict"),
                                (self.vector_value_normalizer, "vector_value_normalizer_state_dict")):
            if normalizer is not None and save_data.get(key):
                normalizer.load_state_dict(save_data[key])

        # Optimizer: only load if param-group count matches.
        try:
            if bool(self.config.get("reset_optimizer_on_load", False)):
                logger.info("PCMAPPO load: reset_optimizer_on_load=true; skipping optimizer state")
            else:
                saved_opt = save_data["optimizer_state_dict"]
                saved_n = len(saved_opt.get("param_groups", []))
                current_n = len(self.optimizer.param_groups)
                if saved_n == current_n:
                    self.optimizer.load_state_dict(saved_opt)
                    configured_lr = float(self.config.get("learning_rate", 0.0) or 0.0)
                    if configured_lr > 0.0:
                        for group in self.optimizer.param_groups:
                            group["lr"] = configured_lr
                    logger.info(f"PCMAPPO load: optimizer lr set to {configured_lr:g}")
                else:
                    logger.warning(
                        f"PCMAPPO load: optimizer param-group count mismatch "
                        f"(saved={saved_n}, current={current_n}); skipping "
                        f"optimizer state (model weights still loaded)"
                    )
        except Exception as exc:
            logger.warning(f"PCMAPPO load: optimizer load failed ({exc}); leaving at init")

        self.episode_count = save_data.get("episode_count", 0)
        self.updates_performed = save_data.get("updates_performed", 0)
        self.steps_collected = save_data.get("steps_collected", self.steps_collected)
        if "training_metrics" in save_data:
            self.training_metrics = save_data["training_metrics"]
        if self.route_intent_head is not None and "route_intent_head_state_dict" in save_data:
            _safe_load(self.route_intent_head, "route_intent_head_state_dict", "route_intent_head")
        self._pretrained = True
        self._mark_acting_dirty()
        logger.info(f"PCMAPPO model loaded from {path}")
