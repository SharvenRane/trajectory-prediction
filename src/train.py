"""Training loop for the trajectory LSTM.

This is a small, fully runnable loop on CPU. It minimizes mean squared error
between predicted and true future positions using Adam.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .model import TrajectoryLSTM


def train_model(
    obs: torch.Tensor,
    target: torch.Tensor,
    pred_len: int | None = None,
    hidden_dim: int = 64,
    epochs: int = 200,
    lr: float = 1e-2,
    batch_size: int = 64,
    seed: int = 0,
    verbose: bool = False,
) -> TrajectoryLSTM:
    """Fit a :class:`TrajectoryLSTM` on the given observed and target tensors."""
    torch.manual_seed(seed)
    if pred_len is None:
        pred_len = target.size(1)

    model = TrajectoryLSTM(
        input_dim=obs.size(-1), hidden_dim=hidden_dim, pred_len=pred_len
    )
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    n = obs.size(0)
    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(n)
        total = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            ob = obs[idx]
            tg = target[idx]
            opt.zero_grad()
            pred = model(ob)
            loss = loss_fn(pred, tg)
            loss.backward()
            opt.step()
            total += loss.item() * ob.size(0)
        if verbose and (epoch % 20 == 0 or epoch == epochs - 1):
            print(f"epoch {epoch:4d}  mse {total / n:.4f}")

    model.eval()
    return model
