"""Scalar or vector value critic for one-shot intent training."""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

from .actor import _build_mlp


class IntentCritic(nn.Module):
    def __init__(self, *, state_dim: int, pref_dim: int, reward_dim: int,
                 hidden_dims: List[int] | None = None):
        super().__init__()
        self.state_dim = int(state_dim)
        self.pref_dim = int(pref_dim)
        self.reward_dim = int(reward_dim)
        hidden_dims = list(hidden_dims or [128, 128])
        self.body = _build_mlp(self.state_dim + self.pref_dim, hidden_dims)
        out_dim = 1 if self.reward_dim == 0 else self.reward_dim
        self.head = nn.Linear(hidden_dims[-1], out_dim)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=float(np.sqrt(2.0)))
                nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.head.weight, gain=1.0)
        nn.init.zeros_(self.head.bias)

    def forward(self, state: torch.Tensor,
                preference: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.pref_dim > 0 and preference is not None:
            features = torch.cat([state, preference], dim=-1)
        else:
            features = state
        out = self.head(self.body(features))
        if self.reward_dim == 0:
            return out.squeeze(-1)
        return out
