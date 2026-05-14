# Survcraft

Survcraft is a Python library for building survival prediction models with
PyTorch. It provides composable neural network input modules, survival
distribution modules, loss functions, and scikit-learn-style adapters for
training models on right-censored time-to-event data.

The package is intended for research and experimentation where the model
architecture, survival distribution, and training loss need to be mixed,
matched, or extended.

## Features

- PyTorch modules for common survival distributions, including exponential,
  Weibull, log-normal, Levy, inverse Gaussian, proportional hazards, accelerated 
  failure time, and mixtures.
- Scikit-learn-compatible adapters for assembling and fitting survival
  predictors.
- Loss modules for full likelihood, partial likelihood, Brier score,
  squared loss, and batch-time classification losses.
- A simple feed-forward input adapter plus lower-level module APIs for custom
  input networks and custom survival modules.
- Optional plotting helpers for model outputs and training histories.

## Installation

Survcraft requires Python 3.9 or newer.

The package is published on PyPI:

```bash
pip install survcraft
```

You can also install it directly from GitHub:

```bash
pip install git+https://github.com/compbiomed-unito/survcraft.git
```

Optional dependency groups are available for common workflows:

```bash
pip install "survcraft[plotting]"
pip install "survcraft[tutorials]"
pip install "survcraft[sksurv]"
pip install "survcraft[test]"
```

The core package depends on PyTorch, NumPy, and scikit-learn.

## Quick Start

The high-level API combines three parts:

1. an input adapter that maps features to raw survival parameters,
2. a survival adapter that turns parameters into a survival distribution, and
3. a loss module used during training.

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
    epochs=50,
    batch_size=32,
    learning_rate=1e-3,
)

model.fit(X, y)

times = np.linspace(0.1, 5.0, 50, dtype=np.float32)
survival = model.predict("survival", X[:5], times)
failure = model.predict("failure", X[:5], times)
```

`y` is expected to be a NumPy structured array with boolean `event` and numeric
`time` fields. This is compatible with the common `scikit-survival` target
format.

## Core Concepts

### Input adapters

Input adapters construct PyTorch modules that map feature vectors to the raw
parameters required by a survival module.

- `FeedForwardNetAdapter` builds a configurable multilayer perceptron.
- `LinearFunctionInputAdapter` is useful for deterministic simulations and
  simple baseline models.

Custom input models can be integrated by following the adapter pattern in
`survcraft.adapters.BaseInputAdapter`.

### Survival adapters

Survival adapters construct the survival distribution module used by the model.
Available adapters include:

- `ExponentialSurvivalAdapter`
- `WeibullSurvivalAdapter`
- `LogNormalSurvivalAdapter`
- `LevySurvivalAdapter`
- `InverseGaussianSurvivalAdapter`
- `StepExpSurvivalAdapter`
- `ProportionalHazardSurvivalAdapter`
- `AcceleratedFailureTimeSurvivalAdapter`
- `MixtureSurvivalAdapter`
- `FractalNoiseSurvivalAdapter`

The resulting model can predict distribution quantities such as survival,
failure, density, hazard, expected time, and median time when implemented by the
selected survival module.

### Loss modules

Training losses live in `survcraft.loss_modules`. The main public losses are:

- `FullLikelihoodLoss`
- `PartialLikelihoodLoss`
- `BrierLoss`
- `SquaredLoss`
- `BrierBatchTimesLoss`
- `BCEBatchTimesLoss`

Loss modules can be combined arithmetically:

```python
loss = lm.FullLikelihoodLoss() + 0.1 * lm.BrierLoss()
```

## Plotting

Plotting helpers require the optional plotting dependencies:

```bash
pip install "survcraft[plotting]"
```

```python
from survcraft import plotting

plotting.plot_outputs(model, X[:20])
plotting.plot_training_history(model)
```

## Tutorials

Notebook tutorials are included in the repository under `tutorials/`:

- `01 Basic usage.ipynb`
- `02 Hyperparameter optimization with Optuna.ipynb`
- `03 Implementing custom layers.ipynb`

Install the tutorial dependencies with:

```bash
pip install "survcraft[tutorials]"
```

Some tutorials may also require optional survival-analysis dependencies:

```bash
pip install "survcraft[sksurv]"
```

## Development

Clone the repository and install the package in editable mode:

```bash
git clone https://github.com/compbiomed-unito/survcraft.git
cd survcraft
pip install -e ".[test,plotting,sksurv]"
```

Run the test suite:

```bash
python -m pytest -q
```

Build the distribution artifacts:

```bash
python -m build
```

Before publishing to PyPI, validate the artifacts with:

```bash
python -m twine check dist/*
```

## Project Status

Survcraft is currently an alpha-stage research library. Public APIs may change
before a stable release.

## License

Survcraft is distributed under the Apache License 2.0. See `LICENSE` for the
full license text.
