"""Hybrid route-action components owned by MOMARL speed algorithms.

These components are not standalone RL algorithms. They attach a masked
one-shot route head to an existing MARL algorithm object so the same policy
family controls the hybrid action tuple ``(a_path, a_speed)`` while preserving
the different action cadences: route at spawn, speed at every simulation step.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from loguru import logger

from .actor import MaskedCategoricalActor
from .buffer import OneShotRolloutBuffer
from .critic import IntentCritic


class HybridRouteQCritic(torch.nn.Module):
    """Action-value critic over masked discrete route intents."""

    def __init__(self, *, state_dim: int, pref_dim: int, num_actions: int,
                 hidden_dims: List[int]):
        super().__init__()
        from .actor import _build_mlp

        self.state_dim = int(state_dim)
        self.pref_dim = int(pref_dim)
        self.num_actions = int(num_actions)
        self.body = _build_mlp(self.state_dim + self.pref_dim, hidden_dims)
        self.head = torch.nn.Linear(hidden_dims[-1], self.num_actions)
        for module in self.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.orthogonal_(module.weight, gain=float(np.sqrt(2.0)))
                torch.nn.init.zeros_(module.bias)
        torch.nn.init.orthogonal_(self.head.weight, gain=1.0)
        torch.nn.init.zeros_(self.head.bias)

    def forward(self, state: torch.Tensor,
                preference: torch.Tensor | None = None) -> torch.Tensor:
        if self.pref_dim > 0 and preference is not None:
            features = torch.cat([state, preference], dim=-1)
        else:
            features = state
        return self.head(self.body(features))


class HybridRoutePPOHead:
    """Masked categorical route head trained with the owning PPO optimizer."""

    NUM_ACTIONS = 4
    supports_vector_rewards = True

    def __init__(
        self,
        *,
        config: Dict[str, Any],
        state_dim: int,
        num_actions: int = 4,
        pref_dim: int,
        reward_dim: int,
        device: torch.device,
        owner_optimizer: optim.Optimizer,
    ):
        if int(pref_dim) <= 0 or int(reward_dim) <= 0:
            raise ValueError("HybridRoutePPOHead requires pref_dim>0 and reward_dim>0")
        self.config = dict(config)
        self.state_dim = int(state_dim)
        self.num_actions = int(num_actions)
        self.pref_dim = int(pref_dim)
        self.reward_dim = int(reward_dim)
        self.device = device
        self.owner_optimizer = owner_optimizer
        hidden_dims: List[int] = list(config.get("hidden_dims", [128, 128]))
        self.clip_param = float(config.get("clip_param", 0.2))
        self.ppo_epochs = int(config.get("ppo_epochs", 4))
        self.batch_size = int(config.get("batch_size", 64))
        self.entropy_coef = float(config.get("entropy_coef", 0.01))
        self.value_coef = float(config.get("value_coef", 0.5))
        self.max_grad_norm = float(config.get("max_grad_norm", 0.5))

        self.actor = MaskedCategoricalActor(
            state_dim=state_dim,
            num_actions=self.num_actions,
            pref_dim=self.pref_dim,
            hidden_dims=hidden_dims,
        ).to(self.device)
        self.critic = IntentCritic(
            state_dim=state_dim,
            pref_dim=self.pref_dim,
            reward_dim=self.reward_dim,
            hidden_dims=hidden_dims,
        ).to(self.device)
        owner_optimizer.add_param_group(
            {"params": list(self.actor.parameters()) + list(self.critic.parameters())}
        )

        self.buffer = OneShotRolloutBuffer(
            state_dim=state_dim,
            pref_dim=self.pref_dim,
            reward_dim=self.reward_dim,
            device=self.device,
            buffer_size=int(config.get("buffer_size", 4096)),
        )
        self._pending: Dict[Any, Dict[str, Any]] = {}
        self.diagnostics: Dict[str, float] = {
            "decisions": 0.0,
            "mask_invalid": 0.0,
            "pending_overwrites": 0.0,
            "terminal_samples": 0.0,
            "updates": 0.0,
        }
        self.training_metrics: Dict[str, float] = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "clip_fraction": 0.0,
            "buffer_size": 0.0,
        }

    def state_dict(self) -> Dict[str, Any]:
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "diagnostics": dict(self.diagnostics),
            "training_metrics": dict(self.training_metrics),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        # Runtime counters/losses are episode-local diagnostics. Keep them
        # fresh on checkpoint load so eval summaries do not inherit stale
        # decisions or updates from the training run that produced the file.

    def get_training_info(self) -> Dict[str, float]:
        info = {f"route_{key}": value for key, value in self.training_metrics.items()}
        info.update({f"route_{key}": value for key, value in self.diagnostics.items()})
        info["route_buffer_size"] = float(len(self.buffer))
        return info

    def _to_tensor(self, x, *, dtype=torch.float32) -> torch.Tensor:
        return torch.as_tensor(x, dtype=dtype, device=self.device)

    def _expand_preference(self, preference) -> torch.Tensor:
        pref = np.asarray(preference, dtype=np.float32)
        if pref.size == 0:
            pref = np.zeros(self.pref_dim, dtype=np.float32)
        if pref.shape[-1] != self.pref_dim:
            raise ValueError(
                f"preference size {pref.shape[-1]} != route pref_dim {self.pref_dim}"
            )
        return self._to_tensor(pref).unsqueeze(0)

    def select_intent(
        self,
        observation: np.ndarray,
        action_mask: np.ndarray,
        preference: np.ndarray,
        *,
        training: bool = True,
    ) -> Tuple[int, float, Any]:
        raw_mask = np.asarray(action_mask, dtype=bool)
        if raw_mask.ndim != 1 or raw_mask.shape[0] != self.num_actions or not bool(raw_mask.any()):
            self.diagnostics["mask_invalid"] += 1.0
            raise ValueError(
                f"Invalid route-intent mask shape={raw_mask.shape}, "
                f"num_actions={self.num_actions}, any={bool(raw_mask.any())}"
            )
        self.diagnostics["decisions"] += 1.0
        state = self._to_tensor(observation).unsqueeze(0)
        mask = self._to_tensor(raw_mask, dtype=torch.bool).unsqueeze(0)
        pref_tensor = self._expand_preference(preference)

        with torch.no_grad():
            if training:
                action, log_prob = self.actor.sample(state, mask, pref_tensor)
            else:
                logits = self.actor.forward(state, mask, pref_tensor)
                action = torch.argmax(logits, dim=-1)
                log_prob = self.actor.evaluate(state, mask, action, pref_tensor)["log_probs"]
            value = self.critic(state, pref_tensor)

        return (
            int(action.item()),
            float(log_prob.item()),
            value.squeeze(0).cpu().numpy().astype(np.float32),
        )

    def store_pending(self, *, actor_id: Any, observation: np.ndarray,
                      action_index: int, action_mask: np.ndarray,
                      log_prob: float, value, preference: np.ndarray) -> None:
        if actor_id in self._pending:
            logger.warning(f"Hybrid route: overwriting pending entry for actor {actor_id}")
            self.diagnostics["pending_overwrites"] += 1.0
        mask = np.asarray(action_mask, dtype=bool)
        if mask.ndim != 1 or mask.shape[0] != self.num_actions or not bool(mask.any()):
            self.diagnostics["mask_invalid"] += 1.0
            raise ValueError(
                f"Invalid pending route-intent mask shape={mask.shape}, "
                f"num_actions={self.num_actions}, any={bool(mask.any())}"
            )
        self._pending[actor_id] = {
            "observation": np.asarray(observation, dtype=np.float32),
            "action_index": int(action_index),
            "action_mask": mask,
            "log_prob": float(log_prob),
            "value": value,
            "preference": np.asarray(preference, dtype=np.float32),
        }

    def has_pending(self, actor_id: Any) -> bool:
        return actor_id in self._pending

    def finalize_terminal(self, *, actor_id: Any, terminal_reward) -> None:
        if actor_id not in self._pending:
            raise KeyError(f"finalize_terminal: no pending route entry for actor {actor_id}")
        entry = self._pending.pop(actor_id)
        self.buffer.add(
            observation=entry["observation"],
            action_index=entry["action_index"],
            action_mask=entry["action_mask"],
            log_prob=entry["log_prob"],
            value=entry["value"],
            preference=entry["preference"],
            terminal_reward=terminal_reward,
        )
        self.diagnostics["terminal_samples"] += 1.0

    def update(self) -> Dict[str, float]:
        if len(self.buffer) < max(2, self.batch_size // 2):
            self.training_metrics["buffer_size"] = float(len(self.buffer))
            return self.training_metrics

        # Actor and critic are clipped separately: with unnormalised terminal
        # returns the critic's gradient norm can be orders of magnitude above
        # the actor's, and one joint clip scales the policy gradient away
        # (measured for the speed controllers, see mappo.py).
        actor_params = list(self.actor.parameters())
        critic_params = list(self.critic.parameters())
        total_policy = total_value = total_entropy = 0.0
        total_kl = total_clip = 0.0
        num_batches = 0
        for _ in range(self.ppo_epochs):
            for batch in self.buffer.get_batches(batch_size=self.batch_size):
                obs = batch["observations"]
                actions = batch["actions"]
                masks = batch["action_masks"]
                old_log_probs = batch["old_log_probs"]
                advantages = batch["advantages"]
                prefs = batch["preferences"]
                vector_returns = batch["vector_returns"]

                eval_out = self.actor.evaluate(obs, masks, actions, prefs)
                new_log_probs = eval_out["log_probs"]
                entropy = eval_out["entropy"].mean()

                ratio = torch.exp(new_log_probs - old_log_probs)
                unclipped = ratio * advantages
                clipped = torch.clamp(
                    ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
                ) * advantages
                policy_loss = -torch.min(unclipped, clipped).mean()

                values = self.critic(obs, prefs)
                value_loss = F.mse_loss(values, vector_returns)
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                self.owner_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(actor_params, self.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(critic_params, self.max_grad_norm)
                self.owner_optimizer.step()

                with torch.no_grad():
                    log_ratio = new_log_probs - old_log_probs
                    approx_kl = (torch.exp(log_ratio) - 1.0 - log_ratio).mean()
                    clip_fraction = ((ratio - 1.0).abs() > self.clip_param).float().mean()
                total_policy += float(policy_loss.item())
                total_value += float(value_loss.item())
                total_entropy += float(entropy.item())
                total_kl += float(approx_kl.item())
                total_clip += float(clip_fraction.item())
                num_batches += 1

        denom = max(1, num_batches)
        self.training_metrics = {
            "policy_loss": total_policy / denom,
            "value_loss": total_value / denom,
            "entropy": total_entropy / denom,
            "approx_kl": total_kl / denom,
            "clip_fraction": total_clip / denom,
            "buffer_size": float(len(self.buffer)),
        }
        self.diagnostics["updates"] += 1.0
        self.buffer.clear()
        return self.training_metrics


class HybridRouteSACHead:
    """Masked discrete SAC route head owned by a SAC controller."""

    NUM_ACTIONS = 4
    supports_vector_rewards = True

    def __init__(
        self,
        *,
        config: Dict[str, Any],
        state_dim: int,
        num_actions: int = 4,
        pref_dim: int,
        reward_dim: int,
        device: torch.device,
    ):
        if int(pref_dim) <= 0 or int(reward_dim) <= 0:
            raise ValueError("HybridRouteSACHead requires pref_dim>0 and reward_dim>0")
        self.config = dict(config)
        self.state_dim = int(state_dim)
        self.num_actions = int(num_actions)
        self.pref_dim = int(pref_dim)
        self.reward_dim = int(reward_dim)
        self.device = device
        hidden_dims: List[int] = list(config.get("hidden_dims", [128, 128]))
        self.batch_size = int(config.get("batch_size", 64))
        self.gradient_steps = int(config.get("gradient_steps", 1))
        self.max_grad_norm = float(config.get("max_grad_norm", 0.5))

        self.actor = MaskedCategoricalActor(
            state_dim=state_dim,
            num_actions=self.num_actions,
            pref_dim=self.pref_dim,
            hidden_dims=hidden_dims,
        ).to(self.device)
        self.critic1 = HybridRouteQCritic(
            state_dim=state_dim,
            pref_dim=self.pref_dim,
            num_actions=self.num_actions,
            hidden_dims=hidden_dims,
        ).to(self.device)
        self.critic2 = HybridRouteQCritic(
            state_dim=state_dim,
            pref_dim=self.pref_dim,
            num_actions=self.num_actions,
            hidden_dims=hidden_dims,
        ).to(self.device)

        actor_lr = float(config.get("learning_rate_actor",
                                    config.get("learning_rate", 3e-4)))
        critic_lr = float(config.get("learning_rate_critic",
                                     config.get("learning_rate", 3e-4)))
        alpha_lr = float(config.get("learning_rate_alpha",
                                    config.get("learning_rate", 3e-4)))
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=critic_lr,
        )
        self.auto_entropy_tuning = bool(config.get("auto_entropy_tuning", True))
        init_alpha = float(config.get("init_alpha", config.get("alpha", 0.2)))
        self.log_alpha = torch.tensor(
            np.log(max(init_alpha, 1e-8)),
            dtype=torch.float32,
            device=self.device,
            requires_grad=self.auto_entropy_tuning,
        )
        self.alpha_optimizer = (
            optim.Adam([self.log_alpha], lr=alpha_lr)
            if self.auto_entropy_tuning else None
        )
        default_target_entropy = 0.75 * float(np.log(self.num_actions))
        self.target_entropy = float(config.get("target_entropy", default_target_entropy))
        # log_alpha clamp bounds — same defensive pattern as the speed-head
        # SAC (opencda_marl/builtin/algorithms/sac.py). For a 4-way
        # categorical route head, log(0.05)=-3 still leaves meaningful
        # exploration; log(7.4)=2 is plenty of head-room. Keeps the route
        # head from collapsing onto keep-current (keep_current_rate -> 1.0)
        # once alpha decays toward zero.
        self.log_alpha_min = float(config.get("log_alpha_min", -3.0))
        self.log_alpha_max = float(config.get("log_alpha_max", 2.0))

        self.buffer = OneShotRolloutBuffer(
            state_dim=state_dim,
            pref_dim=self.pref_dim,
            reward_dim=self.reward_dim,
            device=self.device,
            buffer_size=int(config.get("buffer_size", 8192)),
        )
        self._pending: Dict[Any, Dict[str, Any]] = {}
        self.diagnostics: Dict[str, float] = {
            "decisions": 0.0,
            "mask_invalid": 0.0,
            "pending_overwrites": 0.0,
            "terminal_samples": 0.0,
            "updates": 0.0,
        }
        self.training_metrics: Dict[str, float] = {
            "actor_loss": 0.0,
            "critic1_loss": 0.0,
            "critic2_loss": 0.0,
            "alpha_loss": 0.0,
            "alpha": float(self.log_alpha.exp().detach().item()),
            "entropy": 0.0,
            "q_mean": 0.0,
            "target_mean": 0.0,
            "buffer_size": 0.0,
        }

    def state_dict(self) -> Dict[str, Any]:
        state = {
            "actor": self.actor.state_dict(),
            "critic1": self.critic1.state_dict(),
            "critic2": self.critic2.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "diagnostics": dict(self.diagnostics),
            "training_metrics": dict(self.training_metrics),
        }
        if self.alpha_optimizer is not None:
            state["alpha_optimizer"] = self.alpha_optimizer.state_dict()
        return state

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.actor.load_state_dict(state["actor"])
        self.critic1.load_state_dict(state["critic1"])
        self.critic2.load_state_dict(state["critic2"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        if "log_alpha" in state:
            with torch.no_grad():
                self.log_alpha.copy_(state["log_alpha"].to(self.device))
                # Apply current-config clamp so an old checkpoint with a
                # collapsed log_alpha doesn't briefly silence the route
                # head on the first forward pass after resume.
                self.log_alpha.clamp_(min=self.log_alpha_min, max=self.log_alpha_max)
        if self.alpha_optimizer is not None and "alpha_optimizer" in state:
            self.alpha_optimizer.load_state_dict(state["alpha_optimizer"])
        # Runtime counters/losses are episode-local diagnostics. Keep them
        # fresh on checkpoint load so eval summaries do not inherit stale
        # decisions or updates from the training run that produced the file.
        self.training_metrics["alpha"] = float(self.log_alpha.exp().detach().item())

    def get_training_info(self) -> Dict[str, float]:
        info = {f"route_{key}": value for key, value in self.training_metrics.items()}
        info.update({f"route_{key}": value for key, value in self.diagnostics.items()})
        info["route_buffer_size"] = float(len(self.buffer))
        return info

    def _to_tensor(self, x, *, dtype=torch.float32) -> torch.Tensor:
        return torch.as_tensor(x, dtype=dtype, device=self.device)

    def _expand_preference(self, preference) -> torch.Tensor:
        pref = np.asarray(preference, dtype=np.float32)
        if pref.size == 0:
            pref = np.zeros(self.pref_dim, dtype=np.float32)
        if pref.shape[-1] != self.pref_dim:
            raise ValueError(
                f"preference size {pref.shape[-1]} != route pref_dim {self.pref_dim}"
            )
        return self._to_tensor(pref).unsqueeze(0)

    def select_intent(
        self,
        observation: np.ndarray,
        action_mask: np.ndarray,
        preference: np.ndarray,
        *,
        training: bool = True,
    ) -> Tuple[int, float, Any]:
        raw_mask = np.asarray(action_mask, dtype=bool)
        if raw_mask.ndim != 1 or raw_mask.shape[0] != self.num_actions or not bool(raw_mask.any()):
            self.diagnostics["mask_invalid"] += 1.0
            raise ValueError(
                f"Invalid route-intent mask shape={raw_mask.shape}, "
                f"num_actions={self.num_actions}, any={bool(raw_mask.any())}"
            )
        self.diagnostics["decisions"] += 1.0
        state = self._to_tensor(observation).unsqueeze(0)
        mask = self._to_tensor(raw_mask, dtype=torch.bool).unsqueeze(0)
        pref_tensor = self._expand_preference(preference)

        with torch.no_grad():
            if training:
                action, log_prob = self.actor.sample(state, mask, pref_tensor)
            else:
                logits = self.actor.forward(state, mask, pref_tensor)
                action = torch.argmax(logits, dim=-1)
                log_prob = self.actor.evaluate(state, mask, action, pref_tensor)["log_probs"]
        return int(action.item()), float(log_prob.item()), np.zeros(self.reward_dim, dtype=np.float32)

    def store_pending(self, *, actor_id: Any, observation: np.ndarray,
                      action_index: int, action_mask: np.ndarray,
                      log_prob: float, value, preference: np.ndarray) -> None:
        if actor_id in self._pending:
            logger.warning(f"Hybrid route: overwriting pending entry for actor {actor_id}")
            self.diagnostics["pending_overwrites"] += 1.0
        mask = np.asarray(action_mask, dtype=bool)
        if mask.ndim != 1 or mask.shape[0] != self.num_actions or not bool(mask.any()):
            self.diagnostics["mask_invalid"] += 1.0
            raise ValueError(
                f"Invalid pending route-intent mask shape={mask.shape}, "
                f"num_actions={self.num_actions}, any={bool(mask.any())}"
            )
        self._pending[actor_id] = {
            "observation": np.asarray(observation, dtype=np.float32),
            "action_index": int(action_index),
            "action_mask": mask,
            "log_prob": float(log_prob),
            "value": value,
            "preference": np.asarray(preference, dtype=np.float32),
        }

    def has_pending(self, actor_id: Any) -> bool:
        return actor_id in self._pending

    def finalize_terminal(self, *, actor_id: Any, terminal_reward) -> None:
        if actor_id not in self._pending:
            raise KeyError(f"finalize_terminal: no pending route entry for actor {actor_id}")
        entry = self._pending.pop(actor_id)
        self.buffer.add(
            observation=entry["observation"],
            action_index=entry["action_index"],
            action_mask=entry["action_mask"],
            log_prob=entry["log_prob"],
            value=entry["value"],
            preference=entry["preference"],
            terminal_reward=terminal_reward,
        )
        self.diagnostics["terminal_samples"] += 1.0

    def _next_batch(self) -> Dict[str, torch.Tensor]:
        batches = self.buffer.get_batches(
            batch_size=self.batch_size,
            normalize_advantages=False,
        )
        return next(batches)

    def _policy_terms(
        self,
        obs: torch.Tensor,
        masks: torch.Tensor,
        prefs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.actor.forward(obs, masks, prefs)
        log_probs = torch.log_softmax(logits, dim=-1)
        probs = torch.softmax(logits, dim=-1)
        safe_log_probs = torch.where(masks, log_probs, torch.zeros_like(log_probs))
        entropy = -(probs * safe_log_probs).sum(dim=-1)
        return probs, safe_log_probs, entropy

    def update(self) -> Dict[str, float]:
        if len(self.buffer) < max(2, self.batch_size // 2):
            self.training_metrics["buffer_size"] = float(len(self.buffer))
            return self.training_metrics

        total_actor = total_q1 = total_q2 = total_alpha = 0.0
        total_entropy = total_q = total_target = 0.0
        num_updates = 0
        for _ in range(max(1, self.gradient_steps)):
            batch = self._next_batch()
            obs = batch["observations"]
            actions = batch["actions"]
            masks = batch["action_masks"]
            prefs = batch["preferences"]
            targets = batch["returns"].detach()

            q1_all = self.critic1(obs, prefs)
            q2_all = self.critic2(obs, prefs)
            q1 = q1_all.gather(1, actions.unsqueeze(-1)).squeeze(-1)
            q2 = q2_all.gather(1, actions.unsqueeze(-1)).squeeze(-1)
            critic1_loss = F.mse_loss(q1, targets)
            critic2_loss = F.mse_loss(q2, targets)
            self.critic_optimizer.zero_grad(set_to_none=True)
            (critic1_loss + critic2_loss).backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.critic1.parameters()) + list(self.critic2.parameters()),
                self.max_grad_norm,
            )
            self.critic_optimizer.step()

            probs, safe_log_probs, entropy_per_sample = self._policy_terms(obs, masks, prefs)
            with torch.no_grad():
                min_q = torch.min(self.critic1(obs, prefs), self.critic2(obs, prefs))
            alpha = self.log_alpha.exp().detach()
            actor_loss = (probs * (alpha * safe_log_probs - min_q)).sum(dim=-1).mean()
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()

            entropy = entropy_per_sample.mean()
            alpha_loss = torch.tensor(0.0, device=self.device)
            if self.alpha_optimizer is not None:
                alpha_loss = self.log_alpha * (entropy.detach() - self.target_entropy)
                self.alpha_optimizer.zero_grad(set_to_none=True)
                alpha_loss.backward()
                self.alpha_optimizer.step()
                # Defensive clamp — see opencda_marl/builtin/algorithms/sac.py
                # for the matching pattern. Prevents log_alpha from drifting
                # below log(0.05)=-3 (which would silence route-head
                # exploration and collapse the head onto keep-current).
                with torch.no_grad():
                    self.log_alpha.clamp_(min=self.log_alpha_min, max=self.log_alpha_max)

            total_actor += float(actor_loss.item())
            total_q1 += float(critic1_loss.item())
            total_q2 += float(critic2_loss.item())
            total_alpha += float(alpha_loss.item())
            total_entropy += float(entropy.item())
            total_q += float(torch.min(q1, q2).mean().item())
            total_target += float(targets.mean().item())
            num_updates += 1

        denom = max(1, num_updates)
        self.training_metrics = {
            "actor_loss": total_actor / denom,
            "critic1_loss": total_q1 / denom,
            "critic2_loss": total_q2 / denom,
            "alpha_loss": total_alpha / denom,
            "alpha": float(self.log_alpha.exp().detach().item()),
            "entropy": total_entropy / denom,
            "q_mean": total_q / denom,
            "target_mean": total_target / denom,
            "buffer_size": float(len(self.buffer)),
        }
        self.diagnostics["updates"] += 1.0
        return self.training_metrics
