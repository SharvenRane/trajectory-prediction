# trajectory-prediction

Predict where a moving agent will go next from a short history of its past
positions. The model is a small LSTM that reads the observed window of a 2D
path and rolls out the future, one step at a time. Quality is measured with
average displacement error and final displacement error, the two metrics that
the trajectory forecasting community usually reports.

## Why this exists

Forecasting motion is a building block for self driving stacks, robotics, and
crowd analysis. This repo is a compact, self contained version of that problem.
Everything runs offline on CPU with synthetic data, so you can read the code,
run the tests, and see a learned model beat a naive baseline in a few seconds.

## The data

There is no download. Two families of paths are generated on the fly:

* Constant velocity. The agent picks a random start, heading, and speed, then
  walks in a straight line. The second difference of position over time is zero.
* Curved. The agent keeps a constant speed while its heading rotates at a fixed
  rate, so it traces an arc. Speed stays constant but the path bends.

Each trajectory is cut into an observed window (the past, 8 steps by default)
and a target window (the future, 12 steps by default). A dataset mixes straight
and curved agents and shuffles them together. See `src/data.py`.

## The model

`TrajectoryLSTM` in `src/model.py` encodes the observed motion as step to step
velocities, which makes it translation invariant: the same turning behavior
looks the same wherever it happens on the plane. An LSTM encoder summarizes the
past into a hidden state, and an LSTM cell decoder predicts a velocity at each
future step. Positions come from accumulating those predicted velocities on top
of the last observed point.

Working in velocity space keeps the targets small and centered, which is why a
tiny model trained for a hundred or so epochs already forecasts well.

## Metrics and baselines

`src/metrics.py` gives the two standard numbers:

* Average displacement error is the mean distance between predicted and true
  positions over every future step.
* Final displacement error is the distance at the last future step.

`src/baselines.py` provides two reference predictors. The stationary baseline
assumes the agent freezes at its last seen position. The constant velocity
baseline extrapolates the last observed velocity in a straight line. The
learned model is expected to beat the stationary baseline comfortably.

## Install and run

```
python -m venv .venv
.venv/Scripts/activate     # on Windows
pip install -r requirements.txt
```

Train and evaluate from a short script:

```python
import torch
from src.data import make_dataset
from src.train import train_model
from src.metrics import average_displacement_error, final_displacement_error
from src.baselines import stationary_prediction

obs_tr, tg_tr = make_dataset(256, obs_len=8, pred_len=12, seed=0)
obs_te, tg_te = make_dataset(128, obs_len=8, pred_len=12, seed=99)

model = train_model(obs_tr, tg_tr, pred_len=12, epochs=120)

with torch.no_grad():
    pred = model(obs_te)

print("model ADE", average_displacement_error(pred, tg_te).item())
print("model FDE", final_displacement_error(pred, tg_te).item())

base = stationary_prediction(obs_te, pred_len=12)
print("stationary ADE", average_displacement_error(base, tg_te).item())
```

## Tests

```
python -m pytest tests/ -q
```

The suite covers the data generators (shapes, straightness of constant velocity
paths, constant speed of curved paths, determinism), the metrics against known
hand computed values, the baselines, and the model. The model tests confirm
that batched agents are processed independently, that varying observation
lengths work, and that gradients flow. The headline behavior test trains a
model and checks that it beats the stationary baseline on both metrics, by a
clear margin, on a held out set.

## Layout

```
src/
  data.py        synthetic constant velocity and curved trajectories
  model.py       the LSTM encoder and rollout decoder
  metrics.py     average and final displacement error
  baselines.py   stationary and constant velocity references
  train.py       the training loop
tests/           pytest behavior and property checks
```
