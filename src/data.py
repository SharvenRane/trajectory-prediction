"""Synthetic trajectory generation.

We build two families of agent paths in 2D:

* constant velocity: straight lines, the agent keeps a fixed speed and heading.
* curved: the agent turns at a constant angular rate, tracing an arc.

Each trajectory is split into an observed window (the past) and a target
window (the future). A model sees the observed window and predicts the future.
"""

from __future__ import annotations

import numpy as np
import torch


def make_constant_velocity(
    n: int,
    obs_len: int,
    pred_len: int,
    rng: np.random.Generator,
    speed_range: tuple[float, float] = (0.2, 1.0),
):
    """Generate ``n`` straight line trajectories.

    Returns an array of shape ``(n, obs_len + pred_len, 2)``.
    """
    total = obs_len + pred_len
    starts = rng.uniform(-5.0, 5.0, size=(n, 2))
    angles = rng.uniform(0.0, 2.0 * np.pi, size=n)
    speeds = rng.uniform(speed_range[0], speed_range[1], size=n)

    velocities = np.stack(
        [np.cos(angles) * speeds, np.sin(angles) * speeds], axis=1
    )  # (n, 2)

    steps = np.arange(total, dtype=np.float64)  # (total,)
    # (n, total, 2) = start + step * velocity
    traj = (
        starts[:, None, :]
        + steps[None, :, None] * velocities[:, None, :]
    )
    return traj


def make_curved(
    n: int,
    obs_len: int,
    pred_len: int,
    rng: np.random.Generator,
    speed_range: tuple[float, float] = (0.2, 1.0),
    turn_range: tuple[float, float] = (0.05, 0.25),
):
    """Generate ``n`` constant turn rate (arc) trajectories.

    The agent moves at a fixed speed while its heading rotates by a fixed
    angle each step. Returns shape ``(n, obs_len + pred_len, 2)``.
    """
    total = obs_len + pred_len
    starts = rng.uniform(-5.0, 5.0, size=(n, 2))
    headings0 = rng.uniform(0.0, 2.0 * np.pi, size=n)
    speeds = rng.uniform(speed_range[0], speed_range[1], size=n)
    turn = rng.uniform(turn_range[0], turn_range[1], size=n)
    # random turn direction
    turn = turn * rng.choice([-1.0, 1.0], size=n)

    steps = np.arange(total, dtype=np.float64)  # (total,)
    # heading at each step: (n, total)
    headings = headings0[:, None] + turn[:, None] * steps[None, :]
    # per step displacement
    dx = np.cos(headings) * speeds[:, None]  # (n, total)
    dy = np.sin(headings) * speeds[:, None]
    disp = np.stack([dx, dy], axis=2)  # (n, total, 2)

    # position is start plus cumulative displacement, with the first sample at
    # the start itself (so we shift the cumulative sum by one step).
    cum = np.cumsum(disp, axis=1)
    cum = cum - disp  # make position at step 0 equal to start
    traj = starts[:, None, :] + cum
    return traj


def make_dataset(
    n: int,
    obs_len: int = 8,
    pred_len: int = 12,
    seed: int = 0,
    curved_fraction: float = 0.5,
):
    """Build a mixed dataset of straight and curved trajectories.

    Returns two float32 tensors:

    * ``obs`` of shape ``(n, obs_len, 2)``
    * ``target`` of shape ``(n, pred_len, 2)``
    """
    rng = np.random.default_rng(seed)
    n_curved = int(round(n * curved_fraction))
    n_straight = n - n_curved

    parts = []
    if n_straight > 0:
        parts.append(make_constant_velocity(n_straight, obs_len, pred_len, rng))
    if n_curved > 0:
        parts.append(make_curved(n_curved, obs_len, pred_len, rng))

    traj = np.concatenate(parts, axis=0)
    # shuffle so straight and curved are interleaved
    perm = rng.permutation(traj.shape[0])
    traj = traj[perm]

    obs = traj[:, :obs_len, :]
    target = traj[:, obs_len:, :]

    obs_t = torch.from_numpy(obs).float()
    target_t = torch.from_numpy(target).float()
    return obs_t, target_t
