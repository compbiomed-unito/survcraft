import numpy as np
import pytest
import torch

from survcraft import adapters as ad
from survcraft import survival_modules as sm
from survcraft.loss_modules import BCEBatchTimesLoss


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("mode", ["survival", "failure"])
@pytest.mark.parametrize(
    "values,times",
    [
        ([-21.05615234375, 2.9237217903137207], [11179.0]),
        ([0.0, 0.0], [0.0, 0.5, 1.0]),
        ([-1000.0, 0.0], [0.0, 1.0]),
        ([0.0, 3.0], None),
    ],
    ids=["framingham-overflow", "zero-time", "scale-underflow", "large-time"],
)
def test_weibull_probabilities_have_finite_parameter_gradients(dtype, mode, values, times):
    module = sm.WeibullSurvivalModule().to(dtype=dtype)
    raw = torch.tensor([values], dtype=dtype, requires_grad=True)
    if times is None:
        times = [torch.finfo(dtype).max / 4]
    out = module(mode, raw, torch.tensor(times, dtype=dtype))
    out.sum().backward()

    assert torch.isfinite(out).all()
    assert torch.isfinite(raw.grad).all()
    assert ((out >= 0) & (out <= 1)).all()


@pytest.mark.parametrize("mode", ["survival", "failure"])
def test_weibull_zero_and_capped_tail_have_zero_parameter_gradients(mode):
    module = sm.WeibullSurvivalModule()
    raw = torch.zeros((1, 2), requires_grad=True)
    zero = module(mode, raw, torch.tensor(0.0))
    zero.sum().backward()
    torch.testing.assert_close(zero, torch.tensor([1.0 if mode == "survival" else 0.0]))
    torch.testing.assert_close(raw.grad, torch.zeros_like(raw))

    raw.grad = None
    tail = module(mode, raw, torch.tensor(1e10))
    tail.sum().backward()
    expected = torch.exp(torch.tensor(-20.0))
    if mode == "failure":
        expected = 1 - expected
    torch.testing.assert_close(tail, expected.reshape(1))
    torch.testing.assert_close(raw.grad, torch.zeros_like(raw))


def test_weibull_survival_preserves_values_and_gradients_without_overflow():
    module = sm.WeibullSurvivalModule().double()
    raw = torch.tensor([[0.3, -0.2], [-0.5, 0.8]], dtype=torch.float64, requires_grad=True)
    times = torch.tensor([0.1, 0.5, 1.0, 4.0, 100.0], dtype=torch.float64)
    scale, shape = torch.nn.functional.softplus(raw).split(1, dim=-1)
    reference = torch.exp(-((times / scale) ** shape).clamp(max=20.0))
    actual = module("survival", raw, times)

    torch.testing.assert_close(actual, reference)
    actual_grad = torch.autograd.grad(actual.sum(), raw)[0]
    reference_grad = torch.autograd.grad(reference.sum(), raw)[0]
    torch.testing.assert_close(actual_grad, reference_grad)


@pytest.mark.parametrize("mode", ["survival", "failure"])
def test_weibull_survival_passes_gradcheck(mode):
    module = sm.WeibullSurvivalModule().double()
    raw = torch.tensor([[0.3, -0.2]], dtype=torch.float64, requires_grad=True)
    times = torch.tensor([0.0, 0.1, 0.5, 1.0, 100.0], dtype=torch.float64)
    assert torch.autograd.gradcheck(lambda p: module(mode, p, times), (raw,))


def test_bce_training_with_an_overflowing_weibull_mixture_component():
    predictor = ad.SurvivalPredictor(
        input=ad.FeedForwardNetAdapter(hidden_sizes=[]),
        survival=ad.MixtureSurvivalAdapter([ad.WeibullSurvivalAdapter() for _ in range(3)]),
        loss=BCEBatchTimesLoss(max_times=8),
        epochs=2,
        batch_size=4,
        device="cpu",
        verbose=0,
        history=True,
    )
    X = np.zeros((4, 1), dtype=np.float32)
    events = np.array([True, False, True, False])
    times = np.array([11179.0, 8127.0, 3858.0, 10786.0], dtype=np.float32)
    predictor._init_model(X, events, times)
    predictor.train_history_ = []
    # One component reproduces the failing scale/shape; the others keep BCE
    # informative so this also exercises learning and a second optimizer step.
    linear = predictor.model_.input_module.layers[0]
    with torch.no_grad():
        linear.weight.zero_()
        linear.bias.copy_(torch.tensor([
            0.0, 0.0, 0.0, 8000.0, 0.0,
            -21.05615234375, 2.9237217903137207, 8000.0, 0.0,
        ]))
    before = linear.bias.detach().clone()

    predictor.train(X, events, times, warm_start=True)

    assert len(predictor.train_history_) == 2
    assert not torch.equal(linear.bias, before)
    for parameter in predictor.model_.parameters():
        assert torch.isfinite(parameter).all()
        assert torch.isfinite(parameter.grad).all()
    assert np.isfinite(predictor.predict("failure", X, times)).all()
