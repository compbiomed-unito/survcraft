"""Finite-input regressions for operations whose masked backward was unsafe."""
import pytest
import torch

from survcraft import adapters as ad, loss_modules as lm, survival_modules as sm


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("mode", ["expected_time", "median_time"])
def test_exponential_underflowed_rate_has_safe_filtered_gradients(dtype, mode):
    linear = torch.nn.Linear(1, 1, dtype=dtype)
    with torch.no_grad():
        linear.weight.fill_(1)
        linear.bias.zero_()
    model = ad.TorchModel(linear, sm.ExponentialSurvivalModule().to(dtype))
    x = torch.tensor([[-1000.], [0.]], dtype=dtype, requires_grad=True)
    preds = model(mode, x, check_divergence="skip")
    assert preds[0].isposinf()
    preds[preds.isfinite()].sum().backward()
    assert x.grad.isfinite().all()
    assert x.grad[0] == 0
    actual_grads = [p.grad.clone() for p in linear.parameters()]
    linear.zero_grad()
    model(mode, x.detach()[1:]).sum().backward()
    for actual, parameter in zip(actual_grads, linear.parameters()):
        assert actual.isfinite().all()
        torch.testing.assert_close(actual, parameter.grad)


def test_squared_loss_skips_infinite_exponential_mean_safely():
    x = torch.tensor([[-1000.], [0.]], requires_grad=True)
    model = ad.TorchModel(torch.nn.Identity(), sm.ExponentialSurvivalModule())
    loss = lm.SquaredLoss()(model, x, torch.tensor([True, True]),
                            torch.tensor([1., 2.]), check_divergence="skip")
    loss.backward()
    assert loss.isfinite()
    assert x.grad.isfinite().all()
    assert x.grad[0] == 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("mode", ["density", "hazard"])
def test_weibull_zero_time_gradients_match_positive_time_contributions(dtype, mode):
    # Include shapes below, equal to, and above one.
    raw_shape_one = torch.log(torch.expm1(torch.tensor(1., dtype=dtype)))
    raw = torch.tensor([[0., 0.], [0., raw_shape_one], [0., 2.]],
                       dtype=dtype, requires_grad=True)
    module = sm.WeibullSurvivalModule().to(dtype)
    output = module(mode, raw, torch.tensor([0., 1.], dtype=dtype))
    if mode == "density":
        torch.testing.assert_close(output[:, 0], torch.zeros(3, dtype=dtype))
    else:
        assert output[0, 0].isposinf()
        torch.testing.assert_close(output[1, 0], 1 / torch.nn.functional.softplus(raw[1, 0]))
        assert output[2, 0] == 0
    output[:, 1].sum().backward()
    actual_grad = raw.grad.clone()
    raw.grad = None
    module(mode, raw, torch.tensor([1.], dtype=dtype)).sum().backward()
    assert actual_grad.isfinite().all()
    torch.testing.assert_close(actual_grad, raw.grad)


@pytest.mark.parametrize("mode", ["density", "hazard"])
def test_weibull_zero_time_alone_has_safe_gradients(mode):
    raw = torch.tensor([[0., 0.], [0., 2.]], requires_grad=True)
    output = sm.WeibullSurvivalModule()(mode, raw, torch.tensor([0.]))
    output[output.isfinite()].sum().backward()
    torch.testing.assert_close(raw.grad, torch.zeros_like(raw))


@pytest.mark.parametrize("gpu_mode", [False, True])
@pytest.mark.parametrize("skip", [False, True])
@pytest.mark.parametrize("events", [[True, False], [True, True], [False, False]])
def test_full_likelihood_zero_probabilities_and_empty_groups_have_safe_gradients(gpu_mode, skip, events):
    model = ad.TorchModel(torch.nn.Identity(), sm.ExponentialSurvivalModule())
    x = torch.zeros((2, 1), requires_grad=True)
    loss = lm.FullLikelihoodLoss(gpu_mode=gpu_mode)(
        model, x, torch.tensor(events), torch.tensor([1., 10000.]),
        check_divergence="skip" if skip else "no")
    assert loss.isfinite()
    loss.backward()
    assert x.grad.isfinite().all()


@pytest.mark.parametrize("gpu_mode", [False, True])
def test_full_likelihood_matches_previous_finite_value_and_gradient(gpu_mode):
    x = torch.tensor([[0.2], [0.4], [0.6], [0.8]], requires_grad=True)
    model = ad.TorchModel(torch.nn.Identity(), sm.ExponentialSurvivalModule())
    events = torch.tensor([True, False, True, False])
    times = torch.tensor([1., 2., 3., 4.])
    loss = lm.FullLikelihoodLoss(censoring_alpha=0.7, gpu_mode=gpu_mode)(model, x, events, times)
    density = torch.diagonal(model("density", x[events], times[events]))
    survival = torch.diagonal(model("survival", x[~events], times[~events]))
    expected = (-density.log().sum() - 0.7 * survival.log().sum()) / len(x)
    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(torch.autograd.grad(loss, x)[0],
                               torch.autograd.grad(expected, x)[0])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_likelihood_log_preserves_values_without_invalid_log_gradients(dtype):
    probs = torch.tensor([0., -1., float("nan"), float("inf"),
                          -float("inf"), 0.5, 2.], dtype=dtype, requires_grad=True)
    actual = lm._likelihood_log(probs)
    expected = torch.nan_to_num(probs.detach().log(), nan=0.0, neginf=0.0)
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    torch.testing.assert_close(probs.grad, torch.tensor([0., 0., 0., 0., 0., 2., 0.5], dtype=dtype))


def test_partial_likelihood_skips_singular_weibull_hazard_with_shared_weights():
    linear = torch.nn.Linear(2, 2)
    with torch.no_grad():
        linear.weight.copy_(torch.eye(2))
        linear.bias.zero_()
    model = ad.TorchModel(linear, sm.WeibullSurvivalModule())
    x = torch.tensor([[0., 0.], [0., 2.], [0., 2.], [0., 2.]])
    events = torch.tensor([True, False, True, False])
    times = torch.tensor([0., 1., 2., 3.])
    loss = lm.PartialLikelihoodLoss()(model, x, events, times, check_divergence="skip")
    loss.backward()
    actual_grads = [p.grad.clone() for p in linear.parameters()]
    linear.zero_grad()
    expected = lm.PartialLikelihoodLoss()(model, x[1:], events[1:], times[1:])
    expected.backward()
    torch.testing.assert_close(loss, expected)
    for actual, parameter in zip(actual_grads, linear.parameters()):
        assert actual.isfinite().all()
        torch.testing.assert_close(actual, parameter.grad)


@pytest.mark.parametrize("gpu_mode", [False, True])
def test_full_likelihood_weibull_zero_time_has_finite_shared_gradients(gpu_mode):
    linear = torch.nn.Linear(1, 2)
    with torch.no_grad():
        linear.weight.fill_(0.1)
        linear.bias.zero_()
    model = ad.TorchModel(linear, sm.WeibullSurvivalModule())
    x = torch.tensor([[0.], [1.], [2.], [3.]])
    loss = lm.FullLikelihoodLoss(gpu_mode=gpu_mode)(
        model, x, torch.tensor([True, False, True, False]),
        torch.tensor([0., 1., 2., 3.]), check_divergence="skip")
    assert loss.isfinite()
    loss.backward()
    assert all(p.grad.isfinite().all() for p in linear.parameters())
