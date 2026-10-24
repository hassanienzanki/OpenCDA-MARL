"""Masked categorical actor for one-shot route-intent decisions."""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


def _build_mlp(input_dim: int, hidden_dims: List[int]) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_dim = input_dim
    for h in hidden_dims:
        layers.append(nn.Linear(in_dim, h))
        layers.append(nn.LayerNorm(h))
        layers.append(nn.ReLU())
        in_dim = h
    return nn.Sequential(*layers)


class MaskedCategoricalActor(nn.Module):
    """Discrete masked categorical policy with optional preference concat."""

    def __init__(self, *, state_dim: int, num_actions: int,
                 pref_dim: int = 0, hidden_dims: List[int] | None = None):
        super().__init__()
        self.state_dim = int(state_dim)
        self.num_actions = int(num_actions)
        self.pref_dim = int(pref_dim)
        hidden_dims = list(hidden_dims or [128, 128])
        self.body = _build_mlp(self.state_dim + self.pref_dim, hidden_dims)
        self.head = nn.Linear(hidden_dims[-1], self.num_actions)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=float(np.sqrt(2.0)))
                nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.head.weight, gain=0.01)
        nn.init.zeros_(self.head.bias)

    def _features(self, state: torch.Tensor,
                  preference: Optional[torch.Tensor]) -> torch.Tensor:
        if self.pref_dim == 0 or preference is None:
            return self.body(state)
        if preference.shape[-1] != self.pref_dim:
            raise ValueError(
                f"preference dim {preference.shape[-1]} != actor pref_dim {self.pref_dim}"
            )
        return self.body(torch.cat([state, preference], dim=-1))

    def forward(self, state: torch.Tensor, mask: torch.Tensor,
                preference: Optional[torch.Tensor] = None) -> torch.Tensor:
        logits = self.head(self._features(state, preference))
        return logits.masked_fill(~mask, float("-inf"))

    def sample(self, state: torch.Tensor, mask: torch.Tensor,
               preference: Optional[torch.Tensor] = None
               ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.forward(state, mask, preference)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action)

    def evaluate(self, state: torch.Tensor, mask: torch.Tensor,
                 actions: torch.Tensor,
                 preference: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        logits = self.forward(state, mask, preference)
        dist = Categorical(logits=logits)
        return {
            "log_probs": dist.log_prob(actions),
            "entropy": dist.entropy(),
        }
