import numpy as np
import torch

from src.data import make_constant_velocity, make_curved, make_dataset


def test_constant_velocity_shapes_and_straightness():
    rng = np.random.default_rng(1)
    obs_len, pred_len = 8, 12
    traj = make_constant_velocity(16, obs_len, pred_len, rng)
    assert traj.shape == (16, obs_len + pred_len, 2)

    # On a straight constant velocity path every step displacement is equal,
    # so the second difference along time is essentially zero.
    second_diff = np.diff(traj, n=2, axis=1)
    assert np.allclose(second_diff, 0.0, atol=1e-9)


def test_curved_is_not_straight():
    rng = np.random.default_rng(2)
    traj = make_curved(16, 8, 12, rng)
    assert traj.shape == (16, 20, 2)
    # A turning path has a nonzero second difference for at least some agents.
    second_diff = np.diff(traj, n=2, axis=1)
    assert np.abs(second_diff).max() > 1e-3


def test_curved_speed_is_constant():
    rng = np.random.default_rng(3)
    traj = make_curved(8, 8, 12, rng)
    step = np.diff(traj, axis=1)
    speed = np.linalg.norm(step, axis=-1)  # (n, total-1)
    # each agent keeps a constant speed across the whole path
    spread = speed.max(axis=1) - speed.min(axis=1)
    assert np.all(spread < 1e-6)


def test_make_dataset_split():
    obs, target = make_dataset(20, obs_len=8, pred_len=12, seed=0)
    assert obs.shape == (20, 8, 2)
    assert target.shape == (20, 12, 2)
    assert obs.dtype == torch.float32
    assert target.dtype == torch.float32


def test_make_dataset_deterministic():
    a_obs, a_tg = make_dataset(10, seed=7)
    b_obs, b_tg = make_dataset(10, seed=7)
    assert torch.equal(a_obs, b_obs)
    assert torch.equal(a_tg, b_tg)
