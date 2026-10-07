# Installation and quick start

SurvCraft requires Python 3.9 or newer. Install the published package with:

```bash
pip install survcraft
```

To use the code described by this default-branch documentation, install from
GitHub instead:

```bash
pip install git+https://github.com/compbiomed-unito/survcraft.git
```

## Fit and predict

This small example fits a Weibull model to synthetic outcomes. Features are a
dense numeric matrix. Targets contain an event indicator followed by an observed
time; `False` indicates right censoring.

```python
import numpy as np

from survcraft import adapters as ad
from survcraft import loss_modules as lm

rng = np.random.default_rng(0)
X = rng.normal(size=(128, 4)).astype(np.float32)
y = np.zeros(128, dtype=[("event", "?"), ("time", "f4")])
y["event"] = rng.random(128) < 0.7
y["time"] = rng.uniform(0.1, 5.0, size=128)

model = ad.SurvivalPredictor(
    input=ad.FeedForwardNetAdapter(hidden_sizes=[16, 16]),
    survival=ad.WeibullSurvivalAdapter(),
    loss=lm.FullLikelihoodLoss(),
    epochs=10,
    batch_size=32,
    learning_rate=1e-3,
)
model.fit(X, y)

times = np.linspace(0.1, 5.0, 50, dtype=np.float32)
survival = model.predict("survival", X[:5], times)
assert survival.shape == (5, 50)

median = model.predict("median_time", X[:5])
assert median.shape == (5,)
```

The synthetic outcomes demonstrate the interface; they do not represent a
scientific dataset or provide a model-quality benchmark.

## Inspect training and predictions

Install optional plotting dependencies:

```bash
pip install "survcraft[plotting]"
```

Enable history before fitting to plot recorded training losses:

```python
import matplotlib.pyplot as plt
from survcraft import plotting

model.set_params(history=True)
model.fit(X, y)
plotting.plot_training_history(model)

survival = model.predict("survival", X[:5], times)
plt.figure()
plt.plot(times, survival.T)
plt.xlabel("Time")
plt.ylabel("Survival probability")
```

This uses a positive time grid because some Weibull predictions are singular at
time zero. The general `plot_outputs` helper includes zero in its time grid.

See {py:class}`survcraft.adapters.SurvivalPredictor` for training options and
{py:meth}`survcraft.adapters.SurvivalEstimator.predict` for prediction modes.
