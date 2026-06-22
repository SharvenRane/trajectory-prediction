import torch

from src.data import make_dataset
from src.train import train_model
from src.metrics import average_displacement_error, final_displacement_error
from src.baselines import stationary_prediction


def test_trained_model_beats_stationary_baseline():
    # train on one synthetic set, evaluate on a held out set drawn from the
    # same generator with a different seed
    obs_tr, tg_tr = make_dataset(256, obs_len=8, pred_len=12, seed=0)
    obs_te, tg_te = make_dataset(128, obs_len=8, pred_len=12, seed=99)

    model = train_model(
        obs_tr, tg_tr, pred_len=12, hidden_dim=64, epochs=120, lr=1e-2, seed=0
    )

    with torch.no_grad():
        pred = model(obs_te)

    ade_model = average_displacement_error(pred, tg_te).item()
    fde_model = final_displacement_error(pred, tg_te).item()

    base = stationary_prediction(obs_te, pred_len=12)
    ade_base = average_displacement_error(base, tg_te).item()
    fde_base = final_displacement_error(base, tg_te).item()

    # the learned model should clearly beat sitting still
    assert ade_model < ade_base
    assert fde_model < fde_base
    # and by a comfortable margin, not a coin flip
    assert ade_model < 0.5 * ade_base


def test_trained_model_handles_batched_eval():
    obs_tr, tg_tr = make_dataset(128, obs_len=8, pred_len=10, seed=1)
    model = train_model(
        obs_tr, tg_tr, pred_len=10, hidden_dim=32, epochs=60, lr=1e-2, seed=1
    )
    # feed a batch of many agents at once
    obs_batch, _ = make_dataset(64, obs_len=8, pred_len=10, seed=2)
    with torch.no_grad():
        pred = model(obs_batch)
    assert pred.shape == (64, 10, 2)
    assert torch.isfinite(pred).all()
