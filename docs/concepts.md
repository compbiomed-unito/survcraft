# Models, data, and losses

## Three model components

A {py:class}`survcraft.adapters.SurvivalPredictor` combines:

1. An input adapter that constructs a feature network and maps features to raw
   distribution parameters.
2. A survival adapter that constructs a distribution module and transforms raw
   parameters into the distribution's parameterization.
3. A loss module that evaluates a scalar training objective using predictions
   and observed outcomes.

Adapters configure and construct fresh PyTorch modules. The resulting assembled
model becomes available as `model_` during initialization. Predictors require
training before prediction; {py:class}`survcraft.adapters.SurvivalSimulator`
constructs a fixed model lazily and samples outcomes on a time grid.

## Features and outcomes

Features have shape `(n_samples, n_features)`. Encode categorical variables and
handle missing data before calling the estimator; it does not scale or impute
features.

Targets are one-dimensional NumPy structured arrays with exactly two scalar
fields in order: a boolean event indicator, then a real numeric time. Field
names are arbitrary. `True` means the event was observed; `False` means the
sample was right censored at that time. Times must be finite and nonnegative.

The estimator converts features and times to float32 tensors and events to
boolean tensors. Its `device` option defaults to CPU. Direct PyTorch callers
must place model, losses, and data on compatible devices; the estimator does
not automatically move loss modules to its device.

The full contract is in {py:class}`survcraft.adapters.SurvivalEstimator`.

## Choose a distribution

Exponential and Weibull distributions are useful starting points. Log-normal,
Levy, and inverse Gaussian modules offer other distribution shapes. StepExp
(piecewise constant density with an exponential tail), proportional hazards,
accelerated failure time, and mixture
modules support more structured models.

Each distribution has its own parameterization and supported predictions.
Consult the [survival module reference](api/survival.rst) before choosing a
summary such as median or expected time: not every module implements every
summary, and a distribution may have an infinite expectation.

## Choose a training objective

{py:class}`survcraft.loss_modules.FullLikelihoodLoss` uses density for observed
events and survival probability for censored observations.
{py:class}`survcraft.loss_modules.PartialLikelihoodLoss` uses risk rankings and
batch-local risk sets. Classification losses evaluate failure probabilities
at selected times and omit uninformative censored pairs; they do not implement
inverse-probability-of-censoring weighting.

Losses support addition and scalar multiplication:

```python
from survcraft import loss_modules as lm

loss = lm.FullLikelihoodLoss() + 0.1 * lm.BrierLoss()
```

The [loss reference](api/losses.rst) defines each objective, reduction, and
censoring treatment. Batch size can affect risk sets and classification time
grids, so it can change the objective as well as runtime. Training skips batches
without observed events or with non-finite scalar losses.

## Prediction shapes

For `survival`, `failure`, `density`, and `hazard`, a shared vector of evaluation
times produces `(n_samples, n_times)` outputs. A scalar time produces
`(n_samples,)`. `expected_time`, `median_time`, and `risk` take no time argument
and return one value per sample. Higher risk indicates earlier events.

The `params` mode returns a dictionary of transformed parameter arrays with
shape `(n_samples, group_width)`. Estimator predictions return CPU NumPy arrays.
See {py:meth}`survcraft.adapters.SurvivalEstimator.predict` and
{py:class}`survcraft.survival_modules.BaseSurvivalModule` for details.
