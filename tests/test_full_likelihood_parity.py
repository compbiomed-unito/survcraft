"""Compare the three FullLikelihoodLoss paths and their parity boundaries."""
import math

import pytest
import torch

from survcraft import adapters as ad, loss_modules as lm, survival_modules as sm


PATHS = {"cpu": False, "gpu": True, "reference": None}


def evaluate_paths(model, x, events, times, censoring_alpha=1.0):
    return {
        name: lm.FullLikelihoodLoss(censoring_alpha, gpu_mode=mode)(
            model, x, events, times, check_divergence="no")
        for name, mode in PATHS.items()
    }


@pytest.mark.parametrize("module_cls", [
    sm.ExponentialSurvivalModule, sm.WeibullSurvivalModule, sm.LogNormalSurvivalModule,
])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("event_values", [
    [True, False, True, False, True, False, False],
    [True] * 7,
    [False] * 7,
], ids=["mixed", "all-events", "all-censored"])
@pytest.mark.parametrize("censoring_alpha", [0.0, 0.7, 1.0])
def test_full_likelihood_paths_agree_for_regular_predictions(module_cls, dtype, event_values, censoring_alpha):
    module = module_cls().to(dtype)
    x = torch.linspace(-0.4, 0.4, 7 * module.get_param_number(), dtype=dtype).reshape(7, -1)
    events = torch.tensor(event_values)
    times = torch.linspace(0.5, 2.0, 7, dtype=dtype)
    model = ad.TorchModel(torch.nn.Identity(), module)
    # Stay above the reference's log floor so all paths use the same objective.
    for mode, mask in [("density", events), ("survival", ~events)]:
        probs = torch.diagonal(model(mode, x[mask], times[mask]))
        assert (probs >= 1e-12).all() and probs.isfinite().all()
    values = evaluate_paths(model, x, events, times, censoring_alpha)
    for value in values.values():
        assert value.isfinite()
        # Reference group weights use float32 even for float64 predictions.
        torch.testing.assert_close(value, values["reference"], rtol=1e-6, atol=1e-7)


def fixed_probabilities(mode, x, times, *, check_divergence):
    """Return specified probabilities, independent of prediction mode/time."""
    return x.expand(-1, times.numel())


@pytest.mark.parametrize("probabilities,optimized,reference", [
    ([0.0] * 4, 0.0, -math.log(1e-12)),
    ([1e-20] * 4, -math.log(1e-20), -math.log(1e-12)),
    # The reference replaces the entire NaN event-group mean with zero;
    # the other paths retain the finite event contribution.
    ([float("nan"), 0.5, 0.25, 0.5], math.log(2), math.log(2) / 2),
], ids=["zero", "below-reference-floor", "nan-in-event-group"])
def test_full_likelihood_paths_have_different_log_edge_cases(probabilities, optimized, reference):
    x = torch.tensor(probabilities, dtype=torch.float64).reshape(-1, 1)
    values = evaluate_paths(fixed_probabilities, x, torch.tensor([True, False, True, False]),
                            torch.arange(1., 5., dtype=x.dtype))
    for name in ["cpu", "gpu"]:
        torch.testing.assert_close(values[name], x.new_tensor(optimized))
    torch.testing.assert_close(values["reference"], x.new_tensor(reference))
    assert not torch.isclose(values["cpu"], values["reference"])


def test_full_likelihood_whole_batch_path_changes_training_batch_norm_values():
    # Separate event/censor groups have different statistics from the full batch.
    model = ad.TorchModel(torch.nn.BatchNorm1d(1, affine=False, track_running_stats=False),
                          sm.ExponentialSurvivalModule()).train()
    x = torch.arange(-3., 4.).reshape(-1, 1)
    events = torch.tensor([True, True, True, False, False, False, False])
    times = torch.linspace(0.5, 2., 7)
    values = evaluate_paths(model, x, events, times)
    torch.testing.assert_close(values["cpu"], values["reference"])
    assert not torch.isclose(values["gpu"], values["reference"])
