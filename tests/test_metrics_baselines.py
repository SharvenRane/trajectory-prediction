import torch

from src.metrics import average_displacement_error, final_displacement_error
from src.baselines import stationary_prediction, constant_velocity_prediction


def test_metrics_zero_when_perfect():
    target = torch.randn(4, 12, 2)
    ade = average_displacement_error(target, target)
    fde = final_displacement_error(target, target)
    assert torch.isclose(ade, torch.tensor(0.0))
    assert torch.isclose(fde, torch.tensor(0.0))


def test_ade_known_value():
    # constant offset of (3, 4) gives a Euclidean distance of 5 everywhere
    target = torch.zeros(2, 5, 2)
    pred = torch.zeros(2, 5, 2)
    pred[..., 0] = 3.0
    pred[..., 1] = 4.0
    assert torch.isclose(average_displacement_error(pred, target), torch.tensor(5.0))
    assert torch.isclose(final_displacement_error(pred, target), torch.tensor(5.0))


def test_metric_shape_mismatch_raises():
    a = torch.zeros(2, 5, 2)
    b = torch.zeros(2, 4, 2)
    try:
        average_displacement_error(a, b)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError on shape mismatch")


def test_stationary_baseline_shape_and_value():
    obs = torch.randn(3, 8, 2)
    pred = stationary_prediction(obs, pred_len=12)
    assert pred.shape == (3, 12, 2)
    # every predicted step equals the last observed position
    last = obs[:, -1, :]
    assert torch.allclose(pred, last.unsqueeze(1).expand(-1, 12, -1))


def test_constant_velocity_baseline_extrapolates_a_line():
    # build a single straight path with velocity (1, 0)
    obs = torch.zeros(1, 8, 2)
    obs[0, :, 0] = torch.arange(8, dtype=torch.float32)
    pred = constant_velocity_prediction(obs, pred_len=4)
    expected_x = torch.tensor([8.0, 9.0, 10.0, 11.0])
    assert torch.allclose(pred[0, :, 0], expected_x)
    assert torch.allclose(pred[0, :, 1], torch.zeros(4))
