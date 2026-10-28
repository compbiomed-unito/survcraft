"""Loss-level skipping of invalid predictions from otherwise finite computations.

Upstream NaN arithmetic is outside this contract: discarding its outputs
cannot generally repair its backward graph. Numerical source fixes have
separate coverage in test_skip_gradient_numerics.py.
"""
import pytest
import torch

from survcraft import adapters as ad, loss_modules as losses, survival_modules as sm


class InvalidOutput(sm.ExponentialSurvivalModule):
    """Inject NaN outputs for sentinel rows without unsafe upstream math."""

    def forward(self, mode, raw_params, times=None):
        result = super().forward(mode, raw_params, times)
        invalid = raw_params[:, 0].abs() == 2
        if result.ndim > 1:
            invalid = invalid[:, None]
        return torch.where(invalid, torch.full_like(result, float("nan")), result)


@pytest.mark.parametrize("mode,times", [
    ("expected_time", None), ("survival", torch.tensor(1.0)),
    ("density", torch.tensor([1.0, 2.0])),
])
def test_skip_prediction_preserves_rows(mode, times):
    model = ad.TorchModel(torch.nn.Identity(), InvalidOutput())
    x = torch.tensor([[-1.0], [float("nan")], [1.0], [2.0]])
    preds = model(mode, x, times, check_divergence="skip")
    assert len(preds) == len(x)
    assert not preds[1].isfinite().all()
    assert not hasattr(model, "check_divergence")


def test_skip_parameter_prediction_preserves_rows():
    model = ad.TorchModel(torch.nn.Identity(), sm.WeibullSurvivalModule())
    x = torch.tensor([[0.0, 1.0], [float("nan"), 0.0], [1.0, float("inf")]])
    params = model("params", x, check_divergence="skip")
    assert all(len(value) == len(x) for value in params.values())


@pytest.mark.parametrize("loss", [
    losses.SquaredLoss(), losses.FullLikelihoodLoss(), losses.PartialLikelihoodLoss(),
    losses.BrierLoss(), losses.BrierBatchTimesLoss(), losses.BCEBatchTimesLoss(),
])
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_skip_losses_align_outcomes_and_gradients(loss, device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    model = ad.TorchModel(torch.nn.Identity(), InvalidOutput()).to(device)
    clean_model = ad.TorchModel(torch.nn.Identity(), sm.ExponentialSurvivalModule()).to(device)
    x = torch.tensor([[-1.0], [-2.0], [1.0], [2.0]], requires_grad=True, device=device)
    events = torch.tensor([True, True, False, False], device=device)
    times = torch.tensor([1.0, 3.0, 2.0, 4.0], device=device)
    retained = torch.tensor([True, False, True, False], device=device)
    clean_x = x.detach()[retained].requires_grad_()
    actual = loss(model, x, events, times, check_divergence="skip")
    if isinstance(loss, losses.BrierLoss):
        grid = times[events].unique()
        expected = losses.brier(clean_model("failure", clean_x, grid), grid,
                                events[retained], times[retained])
    elif isinstance(loss, losses.ClassificationLoss):
        grid = loss.get_times(events, times)
        preds = clean_model("failure", clean_x, grid)
        mask = (times[retained, None] > grid) | events[retained, None]
        labels = (times[retained, None] <= grid) & events[retained, None]
        expected = loss.get_classification_loss()(preds[mask], labels[mask].float())
    else:
        expected = loss(clean_model, clean_x, events[retained], times[retained])
    torch.testing.assert_close(actual, expected)
    actual.backward()
    expected.backward()
    assert x.grad.isfinite().all()
    torch.testing.assert_close(x.grad[retained], clean_x.grad)
    assert (x.grad[~retained] == 0).all()


@pytest.mark.parametrize("loss", [
    losses.SquaredLoss(), losses.FullLikelihoodLoss(), losses.PartialLikelihoodLoss(),
    losses.BrierLoss(), losses.BrierBatchTimesLoss(), losses.BCEBatchTimesLoss(),
])
def test_skip_all_individuals_returns_nonfinite_loss(loss):
    model = ad.TorchModel(torch.nn.Identity(), InvalidOutput())
    x = torch.full((2, 1), 2.0)
    value = loss(model, x, torch.tensor([True, False]), torch.tensor([1.0, 2.0]), check_divergence="skip")
    assert not value.isfinite()
