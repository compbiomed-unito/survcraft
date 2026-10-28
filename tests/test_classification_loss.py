"""Classification masking, individual skipping, and CUDA synchronization."""
import pytest
import torch

from survcraft import adapters as ad, loss_modules as lm, survival_modules as sm


class FixedClassificationLoss(lm.ClassificationFixedTimesLoss):
    def __init__(self, times, criterion, gpu_mode=False):
        super().__init__(times, gpu_mode=gpu_mode)
        self.criterion = criterion

    def get_classification_loss(self):
        return self.criterion


class PredictionTable(torch.nn.Module):
    """Create bad outputs from finite logits without invalid upstream math."""
    def __init__(self, dtype, device, invalid_value=None):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.linspace(-2., 2., 12, dtype=dtype, device=device).reshape(4, 3))
        self.invalid_value = invalid_value

    def forward(self, mode, x, times, *, check_divergence):
        assert mode == "failure"
        preds = self.logits.sigmoid()
        if self.invalid_value is not None:
            invalid = torch.zeros_like(preds, dtype=torch.bool)
            # The first NaN is in an uninformative pair: skip still removes
            # the entire individual, including its other, informative pair.
            invalid[1, 2] = True
            invalid[2, 1] = True
            preds = preds.masked_fill(invalid, self.invalid_value)
        return preds


@pytest.fixture(params=["cpu", "cuda:0"])
def device(request):
    if request.param.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    return request.param


@pytest.mark.parametrize("criterion", [torch.nn.functional.mse_loss, torch.nn.functional.binary_cross_entropy], ids=["brier", "bce"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("invalid_value", [None, float("nan"), float("inf"), -float("inf")], ids=["finite", "nan", "inf", "negative-inf"])
@pytest.mark.parametrize("gpu_mode", [False, True], ids=["indexed", "masked"])
def test_classification_values_and_gradients_match_indexed_reference(device, criterion, dtype, invalid_value, gpu_mode):
    model = PredictionTable(dtype, device, invalid_value)
    grid = torch.tensor([1., 2., 3.], dtype=dtype, device=device)
    times = torch.tensor([0.5, 1.5, 2.5, 4.], dtype=dtype, device=device)
    events = torch.tensor([True, False, True, False], device=device)
    x = torch.zeros((4, 1), dtype=dtype, device=device)
    loss = FixedClassificationLoss(grid, criterion, gpu_mode=gpu_mode)
    actual = loss(model, x, events, times, check_divergence="skip")
    actual_grad = torch.autograd.grad(actual, model.logits)[0]

    preds = model("failure", x, grid, check_divergence="no")
    retained = preds.isfinite().all(dim=1)
    informative = ((times[:, None] > grid) | events[:, None]) & retained[:, None]
    labels = (times[:, None] <= grid) & events[:, None]
    expected = criterion(preds[informative], labels[informative].to(dtype))
    expected_grad = torch.autograd.grad(expected, model.logits)[0]
    torch.testing.assert_close(actual, expected)
    assert actual_grad.isfinite().all()
    torch.testing.assert_close(actual_grad, expected_grad)
    assert (actual_grad[~retained] == 0).all()


@pytest.mark.parametrize("criterion", [torch.nn.functional.mse_loss, torch.nn.functional.binary_cross_entropy], ids=["brier", "bce"])
@pytest.mark.parametrize("policy", ["no", "warn", "raise", "skip"])
@pytest.mark.parametrize("gpu_mode", [False, True], ids=["indexed", "masked"])
def test_uninformative_invalid_probabilities_never_reach_criterion(device, criterion, policy, gpu_mode):
    # These outcomes have no informative pairs at the final grid point.
    grid = torch.tensor([1., 2., 3.], device=device)
    times = torch.full((4,), 2.5, device=device)
    events = torch.zeros(4, dtype=torch.bool, device=device)
    model = PredictionTable(torch.float32, device)
    # Finite, out-of-domain values would make BCE raise/assert if evaluated.
    original_forward = model.forward

    def predict(mode, x, times, *, check_divergence):
        result = original_forward(mode, x, times, check_divergence=check_divergence)
        return torch.cat([result[:, :2], torch.full_like(result[:, 2:], -1.)], dim=1)

    model.forward = predict
    value = FixedClassificationLoss(grid, criterion, gpu_mode=gpu_mode)(
        model, torch.zeros((4, 1), device=device), events, times, check_divergence=policy)
    assert value.isfinite()
    value.backward()
    assert model.logits.grad.isfinite().all()
    assert (model.logits.grad[:, 2] == 0).all()


@pytest.mark.parametrize("criterion", [torch.nn.functional.mse_loss, torch.nn.functional.binary_cross_entropy], ids=["brier", "bce"])
@pytest.mark.parametrize("case", ["all-skipped", "no-informative-pairs", "empty-grid"])
@pytest.mark.parametrize("gpu_mode", [False, True], ids=["indexed", "masked"])
def test_empty_classification_objective_returns_nonfinite_loss(device, criterion, case, gpu_mode):
    grid = torch.tensor([] if case == "empty-grid" else [1., 2., 3.], device=device)
    events = torch.zeros(4, dtype=torch.bool, device=device)
    times = torch.zeros(4, device=device)

    def model(mode, x, times, *, check_divergence):
        value = float("nan") if case == "all-skipped" else 0.5
        return torch.full((4, len(times)), value, device=device, requires_grad=True)

    loss = FixedClassificationLoss(grid, criterion, gpu_mode=gpu_mode)(
        model, torch.zeros((4, 1), device=device), events, times, check_divergence="skip")
    assert loss.ndim == 0
    assert not loss.isfinite()


@pytest.mark.parametrize("events", [[False] * 6, [True, False, True, True, False, False]])
@pytest.mark.parametrize("unique", [False, True])
@pytest.mark.parametrize("random_sampling", [False, True])
def test_batch_grid_preserves_event_selection_fallback_order_and_cap(device, events, unique, random_sampling):
    times = torch.tensor([3., 1., 3., 2., 4., 1.], device=device)
    events = torch.tensor(events, device=device)
    loss = lm.BrierBatchTimesLoss(max_times=2, unique_times=unique, random_sampling=random_sampling)
    expected = times[events] if events.any() else times
    if unique:
        expected = expected.unique()
    torch.manual_seed(23)
    if len(expected) > loss.max_times:
        if random_sampling:
            expected = expected[torch.randperm(len(expected), device=device)]
        expected = expected[:loss.max_times]
    torch.manual_seed(23)
    actual = loss.get_times(events, times)
    torch.testing.assert_close(actual, expected)
    assert actual.device == times.device


@pytest.mark.parametrize("grid_kind", ["fixed", "all-batch-times"])
@pytest.mark.parametrize("criterion", [torch.nn.functional.mse_loss, torch.nn.functional.binary_cross_entropy], ids=["brier", "bce"])
def test_skip_reduction_has_no_dynamic_indexing_or_scalar_readback(device, criterion, grid_kind):
    model = PredictionTable(torch.float32, device, float("nan"))
    grid = torch.tensor([1., 2., 3.], device=device)
    events = torch.tensor([True, False, True, False], device=device)
    times = torch.tensor([0.5, 1.5, 2.5, 4.], device=device)
    x = torch.zeros((4, 1), device=device)
    if grid_kind == "fixed":
        loss = FixedClassificationLoss(grid, criterion, gpu_mode=True)
    else:
        loss_cls = lm.BrierBatchTimesLoss if criterion is torch.nn.functional.mse_loss else lm.BCEBatchTimesLoss
        loss = loss_cls(max_times=3, only_event_times=False, gpu_mode=True)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
        value = loss(model, x, events, times, check_divergence="skip")
        value.backward()
    operations = {e.key for e in prof.key_averages()}
    assert "aten::nonzero" not in operations
    # CPU BCE performs input checks internally; they are not GPU readbacks.
    if device.startswith("cuda"):
        assert "aten::_local_scalar_dense" not in operations


@pytest.mark.parametrize("criterion", [torch.nn.functional.mse_loss, torch.nn.functional.binary_cross_entropy], ids=["brier", "bce"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("invalid_value", [None, float("nan"), float("inf")], ids=["finite", "nan", "inf"])
def test_classification_cpu_and_gpu_paths_match_values_and_gradients(device, criterion, dtype, invalid_value):
    model = PredictionTable(dtype, device, invalid_value)
    grid = torch.tensor([1., 2., 3.], dtype=dtype, device=device)
    events = torch.tensor([True, False, True, False], device=device)
    times = torch.tensor([0.5, 1.5, 2.5, 4.], dtype=dtype, device=device)
    x = torch.zeros((4, 1), dtype=dtype, device=device)
    values, gradients = [], []
    for gpu_mode in [False, True]:
        loss = FixedClassificationLoss(grid, criterion, gpu_mode=gpu_mode)
        assert loss.gpu is gpu_mode
        value = loss(model, x, events, times, check_divergence="skip")
        values.append(value)
        gradients.append(torch.autograd.grad(value, model.logits)[0])
    torch.testing.assert_close(values[0], values[1])
    torch.testing.assert_close(gradients[0], gradients[1])


@pytest.mark.parametrize("loss_cls", [lm.BrierBatchTimesLoss, lm.BCEBatchTimesLoss])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("options", [
    {}, {"only_event_times": False}, {"unique_times": True, "random_sampling": True},
], ids=["event-times", "all-times", "random-unique-times"])
def test_batch_classification_paths_match_with_shared_model_parameters(device, loss_cls, dtype, options):
    torch.manual_seed(31)
    model = ad.TorchModel(torch.nn.Linear(2, 1), sm.ExponentialSurvivalModule()).to(device=device, dtype=dtype)
    x = torch.linspace(-0.5, 0.5, 12, device=device, dtype=dtype).reshape(6, 2)
    events = torch.tensor([True, False, True, True, False, True], device=device)
    times = torch.tensor([1., 2., 3., 1., 4., 2.], device=device, dtype=dtype)
    values, gradients = [], []
    for gpu_mode in [False, True]:
        loss = loss_cls(max_times=2, gpu_mode=gpu_mode, **options)
        assert loss.gpu is gpu_mode
        # Both paths must select the same randomly sampled evaluation grid.
        torch.manual_seed(19)
        value = loss(model, x, events, times, check_divergence="no")
        values.append(value)
        gradients.append(torch.autograd.grad(value, tuple(model.parameters())))
    torch.testing.assert_close(values[0], values[1])
    for indexed, masked in zip(gradients[0], gradients[1]):
        torch.testing.assert_close(indexed, masked)
