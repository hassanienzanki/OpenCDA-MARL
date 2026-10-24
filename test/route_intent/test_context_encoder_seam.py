"""The neighbour-context encoder is a registrable kind.

Every learner encodes the same ego-first, nearest-first neighbour sequence
into one context vector; which module does it is an experiment axis, selected by
``<algo>.context_encoder`` and defaulting to the historical LSTM.
"""
import torch  # noqa: F401, I001  (import order: torch before numpy on Windows)
import torch.nn as nn

from opencda_marl.api import register_component
from opencda_marl.api.components import KIND_SIGNATURES, registered_components
from opencda_marl.builtin.algorithms.encoders import LSTMEncoder
from opencda_marl.builtin.algorithms.mappo import MAPPOAlgorithm


def _cfg(**extra):
    cfg = {
        "actor_hidden_dims": [16, 16],
        "critic_hidden_dims": [16, 16],
        "lstm_hidden_size": 8,
        "lstm_num_layers": 1,
        "max_action": 45.0,
        "min_action": 0.0,
        "warmup_steps": 0,
        "batch_size": 4,
        "share_policy": True,
    }
    cfg.update(extra)
    return cfg


class _MeanEncoder(nn.Module):
    """A permutation-invariant stand-in with the encoder contract."""

    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()
        self.proj = nn.Linear(input_size, hidden_size)
        self.hidden_size = hidden_size

    def forward(self, sequences):
        return self.proj(sequences).mean(dim=1)

    def forward_padded(self, sequences, lengths):
        steps = torch.arange(sequences.shape[1], device=sequences.device)[None, :]
        mask = (steps < lengths.to(sequences.device)[:, None]).to(sequences.dtype).unsqueeze(-1)
        return (self.proj(sequences) * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)


def test_the_kind_is_declared_with_the_lstm_as_default():
    assert "context_encoder" in KIND_SIGNATURES
    assert "lstm" in registered_components("context_encoder")
    algo = MAPPOAlgorithm(_cfg(), state_dim=9, action_dim=1)
    assert isinstance(algo.lstm_encoder, LSTMEncoder)
    assert algo.lstm_encoder.hidden_size == 8


def test_a_registered_encoder_replaces_the_lstm_by_config():
    register_component("context_encoder", "mean_test", override=True)(_MeanEncoder)
    algo = MAPPOAlgorithm(_cfg(context_encoder="mean_test"), state_dim=9, action_dim=1)
    assert isinstance(algo.lstm_encoder, _MeanEncoder)
    # The update path encodes through the same attribute: ego-only rows keep
    # a zero context, the rest go through the replacement.
    seqs = torch.ones(2, 3, 9, device=algo.device)   # the learner's device, as the update path does
    lengths = torch.tensor([1, 3])                     # lengths stay on the CPU (packing convention)
    out = algo._encode_context_batch(seqs, lengths)
    assert out.shape == (2, 8)
    assert torch.all(out[0] == 0)
    assert not torch.all(out[1] == 0)
    # The encoder's parameters train with the learner.
    trained = {id(p) for group in algo.optimizer.param_groups for p in group["params"]}
    assert all(id(p) in trained for p in algo.lstm_encoder.parameters())
