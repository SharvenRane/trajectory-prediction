"""Simple analytic baselines used as references for the learned model.

These need no training and give a floor that any useful model should beat or
match on the relevant trajectory family.
"""

from __future__ import annotations

import torch


def stationary_prediction(obs: torch.Tensor, pred_len: int) -> torch.Tensor:
    """Predict that the agent stays at its last observed position.

    Returns shape ``(batch, pred_len, 2)``.
    """
    last = obs[:, -1, :]  # (batch, 2)
    return last.unsqueeze(1).expand(-1, pred_len, -1).contiguous()


def constant_velocity_prediction(
    obs: torch.Tensor, pred_len: int
) -> torch.Tensor:
    """Extrapolate using the last observed velocity.

    The velocity is estimated from the final two observed points and applied
    repeatedly. Returns shape ``(batch, pred_len, 2)``.
    """
    if obs.size(1) < 2:
        raise ValueError("need at least two observed steps for velocity")
    last = obs[:, -1, :]  # (batch, 2)
    vel = obs[:, -1, :] - obs[:, -2, :]  # (batch, 2)
    steps = torch.arange(
        1, pred_len + 1, device=obs.device, dtype=obs.dtype
    )  # (pred_len,)
    pred = last[:, None, :] + steps[None, :, None] * vel[:, None, :]
    return pred
