"""Buffer for one-shot post-spawn intent transitions.

Every transition is terminal: ``advantage = R - V(s)`` and ``return = R``.
The buffer holds at most ``buffer_size`` entries as a FIFO ring: once full,
``add`` overwrites the oldest entry so off-policy heads keep training on
recent data instead of a frozen early-training snapshot. ``get_batches``
yields shuffled mini-batches as PyTorch tensors. Vector critics and
preferences are supported via the ``reward_dim`` and ``pref_dim``
constructor args.
"""
from __future__ import annotations

import logging
from typing import Iterator, List

import numpy as np
import torch

logger = logging.getLogger(__name__)


class OneShotRolloutBuffer:
    def __init__(
        self,
        *,
        state_dim: int,
        pref_dim: int,
        reward_dim: int,
        device: torch.device,
        buffer_size: int = 4096,
    ):
        self.state_dim = int(state_dim)
        self.pref_dim = int(pref_dim)
        self.reward_dim = int(reward_dim)
        self.device = device
        self.buffer_size = int(buffer_size)
        self.observations: List[np.ndarray] = []
        self.actions: List[int] = []
        self.action_masks: List[np.ndarray] = []
        self.log_probs: List[float] = []
        self.scalar_values: List[float] = []
        self.vector_values: List[np.ndarray] = []
        self.preferences: List[np.ndarray] = []
        self.scalar_returns: List[float] = []
        self.vector_returns: List[np.ndarray] = []
        self._next_idx = 0
        self.total_overwrites = 0

    def add(
        self,
        *,
        observation: np.ndarray,
        action_index: int,
        action_mask: np.ndarray,
        log_prob: float,
        value,
        preference: np.ndarray,
        terminal_reward,
    ) -> None:
        if len(self.observations) < self.buffer_size:
            self.observations.append(np.asarray(observation, dtype=np.float32))
            self.actions.append(int(action_index))
            self.action_masks.append(np.asarray(action_mask, dtype=bool))
            self.log_probs.append(float(log_prob))
            self.preferences.append(np.asarray(preference, dtype=np.float32))
            if self.reward_dim == 0:
                self.scalar_values.append(float(value))
                self.scalar_returns.append(float(terminal_reward))
            else:
                self.vector_values.append(np.asarray(value, dtype=np.float32))
                self.vector_returns.append(
                    np.asarray(terminal_reward, dtype=np.float32))
            return
        # Full: overwrite the oldest entry (FIFO ring). Silent capping here
        # previously froze off-policy heads on their first buffer_size samples.
        if self.total_overwrites == 0:
            logger.info(
                "OneShotRolloutBuffer reached capacity (%d); "
                "oldest samples are now overwritten (FIFO ring)",
                self.buffer_size,
            )
        i = self._next_idx
        self.observations[i] = np.asarray(observation, dtype=np.float32)
        self.actions[i] = int(action_index)
        self.action_masks[i] = np.asarray(action_mask, dtype=bool)
        self.log_probs[i] = float(log_prob)
        self.preferences[i] = np.asarray(preference, dtype=np.float32)
        if self.reward_dim == 0:
            self.scalar_values[i] = float(value)
            self.scalar_returns[i] = float(terminal_reward)
        else:
            self.vector_values[i] = np.asarray(value, dtype=np.float32)
            self.vector_returns[i] = np.asarray(terminal_reward, dtype=np.float32)
        self._next_idx = (i + 1) % self.buffer_size
        self.total_overwrites += 1

    def __len__(self) -> int:
        return len(self.observations)

    def clear(self) -> None:
        self.observations.clear()
        self.actions.clear()
        self.action_masks.clear()
        self.log_probs.clear()
        self.scalar_values.clear()
        self.vector_values.clear()
        self.preferences.clear()
        self.scalar_returns.clear()
        self.vector_returns.clear()
        self._next_idx = 0
        self.total_overwrites = 0

    def get_batches(self, *, batch_size: int,
                    normalize_advantages: bool = True) -> Iterator[dict]:
        n = len(self.observations)
        if n == 0:
            return
        obs = torch.from_numpy(np.stack(self.observations)).to(self.device)
        actions = torch.tensor(self.actions, dtype=torch.long, device=self.device)
        masks = torch.from_numpy(np.stack(self.action_masks)).to(self.device)
        old_log_probs = torch.tensor(self.log_probs, dtype=torch.float32,
                                     device=self.device)
        if self.reward_dim == 0:
            values = torch.tensor(self.scalar_values, dtype=torch.float32,
                                  device=self.device)
            returns = torch.tensor(self.scalar_returns, dtype=torch.float32,
                                   device=self.device)
            vector_values = torch.zeros((n, 0), device=self.device)
            vector_returns = torch.zeros((n, 0), device=self.device)
        else:
            vector_values = torch.from_numpy(np.stack(self.vector_values))\
                .to(self.device)
            vector_returns = torch.from_numpy(np.stack(self.vector_returns))\
                .to(self.device)
            if (self.preferences and self.pref_dim > 0
                    and self.pref_dim == self.reward_dim):
                prefs = torch.from_numpy(np.stack(self.preferences)).to(self.device)
                values = (vector_values * prefs).sum(dim=-1)
                returns = (vector_returns * prefs).sum(dim=-1)
            else:
                values = vector_values.mean(dim=-1)
                returns = vector_returns.mean(dim=-1)

        prefs_tensor = (torch.from_numpy(np.stack(self.preferences))
                          .to(self.device)
                        if self.pref_dim > 0 and self.preferences
                        else torch.zeros((n, self.pref_dim), device=self.device))
        advantages = returns - values
        if normalize_advantages and advantages.numel() > 1 and advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        order = torch.randperm(n, device=self.device)
        for start in range(0, n, max(1, batch_size)):
            idx = order[start:start + batch_size]
            yield {
                "observations": obs[idx],
                "actions": actions[idx],
                "action_masks": masks[idx],
                "old_log_probs": old_log_probs[idx],
                "values": values[idx],
                "vector_values": vector_values[idx],
                "preferences": prefs_tensor[idx],
                "advantages": advantages[idx],
                "returns": returns[idx],
                "vector_returns": vector_returns[idx],
            }
