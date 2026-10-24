"""MaskedCategoricalActor tests."""
# Import torch before numpy to avoid a Windows DLL load conflict in this pixi env
# (numpy's loaded MKL DLLs shadow ones torch needs from torch/lib/shm.dll).
import torch  # noqa: I001  (import order matters)
import math

import numpy as np
import pytest
from torch.distributions import Categorical

from opencda_marl.builtin.algorithms.intent.actor import MaskedCategoricalActor


def _actor(state_dim=6, pref_dim=0, seed=0):
    torch.manual_seed(seed)
    return MaskedCategoricalActor(state_dim=state_dim,
                                  num_actions=4, pref_dim=pref_dim,
                                  hidden_dims=[16, 16])


def test_sampling_never_returns_masked_action():
    actor = _actor()
    state = torch.zeros(64, 6)
    mask = torch.tensor([True, False, True, True]).expand(64, 4)
    actions, log_probs = actor.sample(state, mask)
    assert actions.shape == (64,)
    assert torch.all(mask.gather(1, actions.unsqueeze(-1)).squeeze(-1))


def test_keep_only_mask_yields_probability_one_on_keep():
    actor = _actor()
    state = torch.zeros(1, 6)
    mask = torch.tensor([[False, False, False, True]])
    logits = actor.forward(state, mask)
    probs = torch.softmax(logits, dim=-1)
    assert pytest.approx(probs[0, 3].item(), abs=1e-6) == 1.0
    assert probs[0, :3].abs().sum().item() < 1e-6


def test_log_prob_matches_torch_categorical():
    actor = _actor()
    state = torch.randn(8, 6)
    mask = torch.tensor([[True, True, False, True]]).expand(8, 4).clone()
    logits = actor.forward(state, mask)
    masked = logits.masked_fill(~mask, float("-inf"))
    dist = Categorical(logits=masked)
    actions = dist.sample()
    expected = dist.log_prob(actions)
    actual = actor.evaluate(state, mask, actions)["log_probs"]
    assert torch.allclose(actual, expected, atol=1e-6)


def test_preference_conditioning_changes_logits():
    actor = _actor(pref_dim=3)
    state = torch.zeros(1, 6)
    mask = torch.tensor([[True, True, True, True]])
    pref_a = torch.tensor([[1.0, 0.0, 0.0]])
    pref_b = torch.tensor([[0.0, 0.0, 1.0]])
    logits_a = actor.forward(state, mask, preference=pref_a)
    logits_b = actor.forward(state, mask, preference=pref_b)
    assert not torch.allclose(logits_a, logits_b)
