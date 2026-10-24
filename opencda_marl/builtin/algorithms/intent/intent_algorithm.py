"""One-shot PPO algorithms for route-intent learning."""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from loguru import logger

from ..base_algorithm import BaseAlgorithm

from .actor import MaskedCategoricalActor
from .buffer import OneShotRolloutBuffer
from .critic import IntentCritic


class _BaseIntentAlgorithm(BaseAlgorithm):
    """Shared PPO-clip update for masked discrete one-shot decisions."""

    NUM_ACTIONS = 4
    supports_vector_rewards = True

    def __init__(
        self,
        config: Dict[str, Any],
        state_dim: int,
        num_actions: int = 4,
        *,
        pref_dim: int = 0,
        reward_dim: int = 0,
    ):
        super().__init__(config=config, state_dim=state_dim, action_dim=num_actions)
        self.num_actions = int(num_actions)
        self.pref_dim = int(pref_dim)
        self.reward_dim = int(reward_dim)
        self.device = torch.device(config.get("device", "cpu"))
        hidden_dims: List[int] = list(config.get("hidden_dims", [128, 128]))
        self.clip_param = float(config.get("clip_param", 0.2))
        self.ppo_epochs = int(config.get("ppo_epochs", 4))
        self.batch_size = int(config.get("batch_size", 64))
        self.entropy_coef = float(config.get("entropy_coef", 0.01))
        self.value_coef = float(config.get("value_coef", 0.5))
        self.max_grad_norm = float(config.get("max_grad_norm", 0.5))

        self.actor = MaskedCategoricalActor(
            state_dim=state_dim, num_actions=self.num_actions,
            pref_dim=self.pref_dim, hidden_dims=hidden_dims,
        ).to(self.device)
        self.critic = IntentCritic(
            state_dim=state_dim, pref_dim=self.pref_dim,
            reward_dim=self.reward_dim, hidden_dims=hidden_dims,
        ).to(self.device)
        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=self.learning_rate,
        )
        self.buffer = OneShotRolloutBuffer(
            state_dim=state_dim, pref_dim=self.pref_dim,
            reward_dim=self.reward_dim, device=self.device,
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
            "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
            "approx_kl": 0.0, "clip_fraction": 0.0, "buffer_size": 0.0,
        }

    def select_action(self, state, training: bool = True):
        raise NotImplementedError("Intent algorithm uses select_intent(state, mask, preference).")

    def store_transition(self, *args, **kwargs):
        raise NotImplementedError("Intent algorithm uses store_pending + finalize_terminal.")

    def reset_episode(self) -> None:
        return None

    def get_training_info(self) -> Dict[str, Any]:
        info = dict(self.training_metrics)
        info.update(self.diagnostics)
        return info

    def save(self, path: str) -> None:
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            path,
        )

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.optimizer.load_state_dict(ckpt["optimizer"])

    def _to_tensor(self, x, *, dtype=torch.float32) -> torch.Tensor:
        return torch.as_tensor(x, dtype=dtype, device=self.device)

    def _expand_preference(self, preference) -> torch.Tensor | None:
        if self.pref_dim == 0:
            return None
        pref = np.asarray(preference, dtype=np.float32)
        if pref.size == 0:
            pref = np.zeros(self.pref_dim, dtype=np.float32)
        if pref.shape[-1] != self.pref_dim:
            raise ValueError(
                f"preference size {pref.shape[-1]} != algorithm pref_dim {self.pref_dim}"
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

        if self.reward_dim == 0:
            return int(action.item()), float(log_prob.item()), float(value.squeeze().item())
        return (
            int(action.item()),
            float(log_prob.item()),
            value.squeeze(0).cpu().numpy().astype(np.float32),
        )

    def store_pending(self, *, actor_id: Any, observation: np.ndarray,
                      action_index: int, action_mask: np.ndarray,
                      log_prob: float, value, preference: np.ndarray) -> None:
        if actor_id in self._pending:
            logger.warning(f"Intent: overwriting pending entry for actor {actor_id}")
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
            raise KeyError(f"finalize_terminal: no pending entry for actor {actor_id}")
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
                returns = batch["returns"]
                prefs = batch["preferences"] if self.pref_dim > 0 else None

                eval_out = self.actor.evaluate(obs, masks, actions, prefs)
                new_log_probs = eval_out["log_probs"]
                entropy = eval_out["entropy"].mean()

                ratio = torch.exp(new_log_probs - old_log_probs)
                unclipped = ratio * advantages
                clipped = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages
                policy_loss = -torch.min(unclipped, clipped).mean()

                # PPO contract: advantages above are the buffer-computed scalarized GAE values,
                # held fixed across all PPO epochs. Only the policy ratio and value_loss (below)
                # are recomputed per minibatch; advantages are intentionally NOT refreshed
                # against the updated critic, since doing so would break the surrogate objective.
                values = self.critic(obs, prefs)
                if self.reward_dim == 0:
                    value_loss = F.mse_loss(values, returns)
                else:
                    vector_returns = batch["vector_returns"]
                    value_loss = F.mse_loss(values, vector_returns)

                loss = policy_loss + self.value_coef * value_loss \
                    - self.entropy_coef * entropy
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                # Separate clips: a large value loss must not scale the
                # policy gradient away (see hybrid_policy.py / mappo.py).
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    log_ratio = new_log_probs - old_log_probs
                    approx_kl = (torch.exp(log_ratio) - 1.0 - log_ratio).mean()
                    clip_fraction = (
                        (ratio - 1.0).abs() > self.clip_param
                    ).float().mean()
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
        self.training_step += 1
        self.diagnostics["updates"] += 1.0
        self.buffer.clear()
        return self.training_metrics


class IntentMAPPOAlgorithm(_BaseIntentAlgorithm):
    """Scalar reward variant."""

    def __init__(self, config: Dict[str, Any], state_dim: int,
                 num_actions: int = 4):
        super().__init__(config=config, state_dim=state_dim,
                         num_actions=num_actions, pref_dim=0, reward_dim=0)


class IntentPCMAPPOAlgorithm(_BaseIntentAlgorithm):
    """Preference-conditioned vector-reward variant."""

    def __init__(self, config: Dict[str, Any], state_dim: int,
                 num_actions: int = 4, *, pref_dim: int, reward_dim: int):
        if int(pref_dim) <= 0 or int(reward_dim) <= 0:
            raise ValueError("IntentPCMAPPO requires pref_dim>0 and reward_dim>0")
        super().__init__(config=config, state_dim=state_dim,
                         num_actions=num_actions,
                         pref_dim=pref_dim, reward_dim=reward_dim)
