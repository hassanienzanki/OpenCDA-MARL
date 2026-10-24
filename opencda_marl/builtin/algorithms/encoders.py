"""Shared neural building blocks for the algorithm family.

Modules here are algorithm-agnostic and were previously duplicated per
algorithm file. Keeping one implementation guarantees every learner encodes
the multi-agent context identically — a precondition for matched-protocol
comparisons between them.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class LSTMEncoder(nn.Module):
    """LSTM encoder for sequences of neighbouring-agent observations.

    Returns the final hidden state of the last layer, giving a fixed-width
    context vector for a variable-length neighbour set.

    Weight init: Xavier for input-to-hidden, orthogonal for hidden-to-hidden,
    zero biases with the forget-gate bias set to 1 for gradient flow.
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
                n = param.size(0)
                with torch.no_grad():
                    param.data[n // 4:n // 2] = 1.0  # forget gate

    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        """Encode ``[batch, seq_len, input_size]`` -> ``[batch, hidden_size]``."""
        batch_size = sequences.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size,
                         device=sequences.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size,
                         device=sequences.device)
        _, (hn, _) = self.lstm(sequences, (h0, c0))
        return hn[-1]

    def forward_padded(self, sequences: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Encode a zero-padded batch exactly, honouring true lengths.

        ``sequences`` is ``[batch, max_len, input_size]``, ``lengths`` the
        true length of each row (>= 1). Rows are packed so the returned final
        hidden state of row i is what :meth:`forward` would return for that
        row's un-padded sequence — trailing zero rows do not leak into the
        state. This is what makes batched update-time encoding numerically
        equal to the per-transition encoding it replaces.
        """
        # Out-of-place clamp: when ``lengths`` already lives on the CPU,
        # ``.to`` returns the same storage and an in-place clamp would
        # mutate the caller's tensor.
        lengths_cpu = lengths.detach().to("cpu", dtype=torch.int64).clamp(min=1)
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            sequences, lengths_cpu, batch_first=True, enforce_sorted=False,
        )
        batch_size = sequences.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size,
                         device=sequences.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size,
                         device=sequences.device)
        _, (hn, _) = self.lstm(packed, (h0, c0))
        return hn[-1]


__all__ = ["LSTMEncoder"]
