"""Scalar and vector critic tests."""
# Import torch before numpy to avoid a Windows DLL load conflict in this pixi env
# (numpy's loaded MKL DLLs shadow ones torch needs from torch/lib/shm.dll).
import torch  # noqa: I001  (import order matters)

from opencda_marl.builtin.algorithms.intent.critic import IntentCritic


def test_scalar_critic_outputs_scalar():
    critic = IntentCritic(state_dim=6, pref_dim=0, reward_dim=0,
                          hidden_dims=[16, 16])
    state = torch.randn(8, 6)
    out = critic(state)
    assert out.shape == (8,)


def test_vector_critic_outputs_vector():
    critic = IntentCritic(state_dim=6, pref_dim=2, reward_dim=3,
                          hidden_dims=[16, 16])
    state = torch.randn(8, 6)
    pref = torch.softmax(torch.randn(8, 2), dim=-1)
    out = critic(state, preference=pref)
    assert out.shape == (8, 3)
