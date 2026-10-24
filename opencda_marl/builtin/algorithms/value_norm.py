"""Running return normalisation for value critics (MAPPO-style ``ValueNorm``).

With unnormalised returns in the thousands the value loss dominates the
optimisation (and, per network, its own gradient scale). The critic can
instead regress the *normalised* return: keep a running mean/variance of
the return targets, train the critic on ``(return - mean) / std`` and
de-normalise its outputs wherever a value in reward units is needed (the
value stashed for GAE while acting, the bootstrap at a rollout cut).
This is the running-statistics variant used by the MAPPO reference code
(no output-preserving weight rescale); it works per component so a vector
critic gets one scale per objective.
"""
from __future__ import annotations

from typing import Dict

import torch


class ValueNormalizer:
    """Per-component running mean/variance with de-biased estimates.

    ``update(x)`` folds a batch ``[N, dim]`` (or ``[N]`` for ``dim == 1``)
    into exponentially decayed sums (decay ``beta``, close to 1) that are
    de-biased on read, so the first batch already yields its own mean and
    variance. ``normalize``/``denormalize`` map between reward units and the
    critic's output space. State round-trips through :meth:`state_dict` /
    :meth:`load_state_dict`.
    """

    def __init__(self, dim: int, device: torch.device, beta: float = 0.99999, epsilon: float = 1e-5):
        self.dim = int(dim)
        self.device = device
        self.beta = float(beta)
        self.epsilon = float(epsilon)
        self.running_mean = torch.zeros(self.dim, device=device)
        self.running_mean_sq = torch.zeros(self.dim, device=device)
        self.debiasing_term = torch.zeros((), device=device)

    def _as_2d(self, x: torch.Tensor) -> torch.Tensor:
        x = x.detach().to(self.device, dtype=torch.float32)
        return x.reshape(-1, self.dim)

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        x = self._as_2d(x)
        batch_mean = x.mean(dim=0)
        batch_sq_mean = (x * x).mean(dim=0)
        weight = self.beta
        self.running_mean.mul_(weight).add_(batch_mean * (1.0 - weight))
        self.running_mean_sq.mul_(weight).add_(batch_sq_mean * (1.0 - weight))
        self.debiasing_term.mul_(weight).add_(1.0 - weight)

    def mean_var(self):
        debias = self.debiasing_term.clamp(min=self.epsilon)
        mean = self.running_mean / debias
        var = (self.running_mean_sq / debias - mean * mean).clamp(min=1e-2)
        return mean, var

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        mean, var = self.mean_var()
        return (x - mean) / torch.sqrt(var)

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        mean, var = self.mean_var()
        return x * torch.sqrt(var) + mean

    def mirror(self, device) -> "ValueNormalizer":
        """A copy of this normaliser on *device*, for a module replica that
        acts there; keep it current with :meth:`copy_from`."""
        other = ValueNormalizer(self.dim, torch.device(device), beta=self.beta, epsilon=self.epsilon)
        other.copy_from(self)
        return other

    @torch.no_grad()
    def copy_from(self, other: "ValueNormalizer") -> None:
        self.running_mean.copy_(other.running_mean)
        self.running_mean_sq.copy_(other.running_mean_sq)
        self.debiasing_term.copy_(other.debiasing_term)

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {"running_mean": self.running_mean.clone(),
                "running_mean_sq": self.running_mean_sq.clone(),
                "debiasing_term": self.debiasing_term.clone()}

    def load_state_dict(self, state: Dict[str, torch.Tensor]) -> None:
        with torch.no_grad():
            self.running_mean.copy_(state["running_mean"].to(self.device))
            self.running_mean_sq.copy_(state["running_mean_sq"].to(self.device))
            self.debiasing_term.copy_(state["debiasing_term"].to(self.device))


__all__ = ["ValueNormalizer"]
