"""A small LSTM sequence model for trajectory prediction.

The model encodes the observed positions with an LSTM, then rolls out the
future positions one step at a time. To make learning easier and to keep the
model translation invariant, we predict step to step displacements relative to
the last observed point rather than absolute coordinates.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TrajectoryLSTM(nn.Module):
    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 64,
        num_layers: int = 1,
        pred_len: int = 12,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.pred_len = pred_len

        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.decoder_cell = nn.LSTMCell(input_dim, hidden_dim)
        self.readout = nn.Linear(hidden_dim, input_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Predict future positions from observed positions.

        ``obs`` has shape ``(batch, obs_len, 2)``. Returns predicted future
        positions of shape ``(batch, pred_len, 2)`` in the same absolute
        coordinate frame as ``obs``.
        """
        if obs.dim() != 3 or obs.size(-1) != self.input_dim:
            raise ValueError(
                f"expected obs of shape (batch, obs_len, {self.input_dim}), "
                f"got {tuple(obs.shape)}"
            )

        batch = obs.size(0)
        # The encoder works on step to step velocities so the model does not
        # have to memorize absolute position. The first velocity is zero.
        vel = torch.zeros_like(obs)
        vel[:, 1:, :] = obs[:, 1:, :] - obs[:, :-1, :]

        _, (h, c) = self.encoder(vel)
        # take the top layer hidden and cell state to seed the decoder
        h_dec = h[-1]
        c_dec = c[-1]

        last_pos = obs[:, -1, :]  # (batch, 2)
        last_vel = vel[:, -1, :]  # (batch, 2)

        outputs = []
        cur_pos = last_pos
        step_in = last_vel
        for _ in range(self.pred_len):
            h_dec, c_dec = self.decoder_cell(step_in, (h_dec, c_dec))
            delta = self.readout(h_dec)  # predicted velocity for this step
            cur_pos = cur_pos + delta
            outputs.append(cur_pos)
            step_in = delta

        pred = torch.stack(outputs, dim=1)  # (batch, pred_len, 2)
        return pred
