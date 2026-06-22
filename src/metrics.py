"""Displacement error metrics.

Average displacement error (ADE) is the mean Euclidean distance between the
predicted and true positions over all future steps. Final displacement error
(FDE) is the Euclidean distance at the last predicted step. Lower is better for
both.
"""

from __future__ import annotations

import torch


def _check(pred: torch.Tensor, target: torch.Tensor) -> None:
    if pred.shape != target.shape:
        raise ValueError(
            f"pred shape {tuple(pred.shape)} does not match target shape "
            f"{tuple(target.shape)}"
        )
    if pred.dim() != 3 or pred.size(-1) != 2:
        raise ValueError(
            f"expected (batch, pred_len, 2), got {tuple(pred.shape)}"
        )


def average_displacement_error(
    pred: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Mean Euclidean distance across all steps and all agents."""
    _check(pred, target)
    dist = torch.linalg.norm(pred - target, dim=-1)  # (batch, pred_len)
    return dist.mean()


def final_displacement_error(
    pred: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Mean Euclidean distance at the final predicted step."""
    _check(pred, target)
    dist = torch.linalg.norm(pred[:, -1, :] - target[:, -1, :], dim=-1)
    return dist.mean()
