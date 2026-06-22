import torch

from src.model import TrajectoryLSTM


def test_forward_shape_single_and_batched():
    model = TrajectoryLSTM(hidden_dim=16, pred_len=12)
    for batch in (1, 5, 32):
        obs = torch.randn(batch, 8, 2)
        pred = model(obs)
        assert pred.shape == (batch, 12, 2)


def test_handles_varying_obs_length():
    model = TrajectoryLSTM(hidden_dim=16, pred_len=6)
    for obs_len in (2, 8, 20):
        obs = torch.randn(4, obs_len, 2)
        pred = model(obs)
        assert pred.shape == (4, 6, 2)


def test_batched_agents_are_independent():
    # Two agents processed together must give the same output as processed
    # alone. This checks there is no leakage across the batch dimension.
    torch.manual_seed(0)
    model = TrajectoryLSTM(hidden_dim=16, pred_len=5)
    model.eval()
    a = torch.randn(1, 8, 2)
    b = torch.randn(1, 8, 2)
    together = model(torch.cat([a, b], dim=0))
    alone_a = model(a)
    alone_b = model(b)
    assert torch.allclose(together[0:1], alone_a, atol=1e-5)
    assert torch.allclose(together[1:2], alone_b, atol=1e-5)


def test_bad_input_shape_raises():
    model = TrajectoryLSTM()
    try:
        model(torch.randn(4, 8))  # missing feature dim
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError on bad input shape")


def test_gradients_flow():
    model = TrajectoryLSTM(hidden_dim=16, pred_len=4)
    obs = torch.randn(3, 8, 2)
    pred = model(obs)
    pred.sum().backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None and torch.any(g != 0) for g in grads)
