# Extend a model

Use the existing adapters when configuring standard networks and distributions.
For a custom feature network, implement
{py:class}`survcraft.adapters.BaseInputAdapter`: expose `module_class` and
construct a fresh module through `get_module(input_size, output_size)`. The
network must map `(n_samples, n_features)` inputs to the raw parameter width
required by the survival module.

For a custom distribution, follow
{py:class}`survcraft.survival_modules.BaseSurvivalModule`. Declare its parameter
groups and implement its distribution-specific methods. The base contract
describes parameter transformations, time broadcasting, prediction modes, and
fallback identities. Expose the module through a
{py:class}`survcraft.adapters.BaseSurvivalAdapter` for estimator use.

A custom loss subclasses {py:class}`survcraft.loss_modules.BaseSurvivalLoss` and
implements `forward(model, x, event, time)`. It must return a scalar tensor and
preserve gradients needed for training. Follow the base contract for shapes,
devices, event semantics, and arithmetic composition.

See the [custom layers notebook](tutorials.md) for a worked example.
