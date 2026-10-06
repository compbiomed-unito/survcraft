import numpy as np
import pytest
import torch
from sklearn.base import clone

from survcraft import adapters as ad
from survcraft import survival_modules as sm


COMPOSITE_ADAPTERS = [
    ad.ProportionalHazardSurvivalAdapter,
    ad.AcceleratedFailureTimeSurvivalAdapter,
    ad.MixtureSurvivalAdapter,
]


@pytest.mark.parametrize("base_class", [ad.BaseInputAdapter, ad.BaseSurvivalAdapter])
def test_base_adapters_are_abstract(base_class):
    with pytest.raises(TypeError, match="abstract.*module_class"):
        base_class()

    class MissingModuleAdapter(base_class):
        pass

    with pytest.raises(TypeError, match="abstract.*module_class"):
        MissingModuleAdapter()


@pytest.mark.parametrize("adapter_class", [
    ad.LinearFunctionInputAdapter,
    ad.FeedForwardNetAdapter,
    ad.ExponentialSurvivalAdapter,
    ad.StepExpSurvivalAdapter,
])
def test_module_class_attribute_satisfies_abstract_contract(adapter_class):
    adapter = clone(adapter_class())
    assert isinstance(adapter.module_class, type)
    assert issubclass(adapter.module_class, torch.nn.Module)
    assert "module_class" not in adapter.get_params()


@pytest.mark.parametrize("adapter_class", COMPOSITE_ADAPTERS)
@pytest.mark.parametrize("use_data", [False, True])
def test_default_baselines_construct_and_predict(adapter_class, use_data):
    adapter = adapter_class()
    event = np.ones(20, dtype=bool) if use_data else None
    time = np.linspace(0.1, 2, 20) if use_data else None
    module = clone(adapter).get_module(event, time).eval()
    raw = torch.zeros(3, module.get_param_number())
    survival = module("survival", raw, torch.tensor([0.0, 0.5, 2.0]))
    assert survival.shape == (3, 3)
    assert torch.isfinite(survival).all()
    assert torch.all(survival[:, 1:] <= survival[:, :-1])
    torch.testing.assert_close(survival[:, 0], torch.ones(3))


def test_default_baselines_are_independent():
    for adapter_class in COMPOSITE_ADAPTERS:
        first, second = adapter_class(), adapter_class()
        if adapter_class is ad.MixtureSurvivalAdapter:
            assert len(first.baselines) == 3
            assert len({id(b) for b in first.baselines + second.baselines}) == 6
        else:
            assert first.baseline is not second.baseline


@pytest.mark.parametrize("adapter_class", COMPOSITE_ADAPTERS[:2])
def test_default_baseline_parameters_are_trainable(adapter_class):
    module = adapter_class(baseline=ad.ExponentialSurvivalAdapter()).get_module(None, None)
    assert isinstance(module.baseline_params, torch.nn.Parameter)
    raw = torch.zeros(2, module.get_param_number(), requires_grad=True)
    module("survival", raw, torch.tensor([0.5, 2.0])).sum().backward()
    assert torch.isfinite(module.baseline_params.grad).all()
    assert torch.isfinite(raw.grad).all()


@pytest.mark.parametrize("adapter_class", COMPOSITE_ADAPTERS[:2])
@pytest.mark.parametrize("baseline_params", [[0.5], np.array([0.5], dtype=np.float64)])
def test_fixed_baseline_parameters_control_predictions(adapter_class, baseline_params):
    module = adapter_class(
        baseline=ad.ExponentialSurvivalAdapter(), baseline_params=baseline_params
    ).get_module(None, None).eval()
    torch.testing.assert_close(module.baseline_params, torch.tensor([0.5]))
    assert not module.baseline_params.requires_grad
    assert all(p is not module.baseline_params for p in module.parameters())
    raw = torch.tensor([[-1.0], [0.5]], requires_grad=True)
    times = torch.tensor([0.0, 0.5, 2.0])
    rate = torch.nn.functional.softplus(torch.tensor(0.5))
    relative_risk = torch.nn.functional.softplus(raw)
    survival = module("survival", raw, times)
    torch.testing.assert_close(survival, torch.exp(-rate * relative_risk * times))
    survival.sum().backward()
    assert torch.isfinite(raw.grad).all()


@pytest.mark.parametrize("adapter_class", COMPOSITE_ADAPTERS[:2])
@pytest.mark.parametrize("baseline_params", [0.5, [], [0.5, 1.0], [[0.5]]])
def test_invalid_baseline_parameter_shape_rejected(adapter_class, baseline_params):
    with pytest.raises(ValueError, match="bad baseline_params shape"):
        adapter_class(
            baseline=ad.ExponentialSurvivalAdapter(), baseline_params=baseline_params
        ).get_module(None, None)


@pytest.mark.parametrize("baselines", [[], (), [ad.ExponentialSurvivalAdapter()], (ad.ExponentialSurvivalAdapter(),)])
def test_mixture_requires_at_least_two_components(baselines):
    with pytest.raises(ValueError, match="baselines.*at least two"):
        ad.MixtureSurvivalAdapter(baselines=baselines).get_module(None, None)


@pytest.mark.parametrize("sequence_class", [list, tuple])
def test_two_component_mixture_matches_weighted_baselines(sequence_class):
    baseline = ad.ExponentialSurvivalAdapter().get_module(None, None)
    mixture = ad.MixtureSurvivalAdapter(baselines=sequence_class([
        ad.ExponentialSurvivalAdapter(), ad.ExponentialSurvivalAdapter()
    ])).get_module(None, None)
    times = torch.tensor([0.0, 0.5, 2.0])
    logits = torch.tensor([[-1.0, 0.5], [0.5, -1.0]])
    first_raw = torch.tensor([[-1.0], [0.5]])
    second_raw = torch.tensor([[0.5], [-1.0]])
    mixture_raw = torch.cat([logits, first_raw, second_raw], dim=1).requires_grad_()
    weights = logits.softmax(dim=-1)
    for mode in ["survival", "failure", "density", "expected_time"]:
        t = None if mode == "expected_time" else times
        first = baseline(mode, first_raw, t)
        second = baseline(mode, second_raw, t)
        w = weights if t is None else weights.unsqueeze(-1)
        expected = w[:, 0] * first + w[:, 1] * second
        torch.testing.assert_close(mixture(mode, mixture_raw, t), expected)
    mixture("survival", mixture_raw, times).sum().backward()
    assert torch.isfinite(mixture_raw.grad).all()


@pytest.mark.parametrize("breaks, message", [
    (0, "at least two"), (1, "at least two"), (-2, "at least two"),
    (2.5, "one-dimensional"), (None, "one-dimensional"),
    ([], "at least two"), ([0], "at least two"), ([[0, 1]], "one-dimensional"),
    ([[0], [1, 2]], "one-dimensional"),
    ([False, True], "numeric"),
    ([0, np.nan], "finite"), ([0, np.inf], "finite"),
    ([0, "a"], "numeric"), ([0, 1j], "numeric"),
    ([1, 2], "start at zero"), ([-1, 0], "finite nonnegative"),
    ([0, 0], "strictly increasing"), ([0, 2, 1], "strictly increasing"),
    ([0, 1, 1 + 1e-10], "float32"), ([0, 1e40], "float32"),
    ([0, 1e-50], "float32"),
])
def test_invalid_step_breaks_rejected(breaks, message):
    with pytest.raises(ValueError, match=message):
        ad.StepExpSurvivalAdapter(breaks=breaks).get_module(None, None)


@pytest.mark.parametrize("breaks", [2, np.int64(3), [0, 0.5, 1], np.array([0, 1]), [0, 1e-8]])
def test_valid_step_breaks_construct(breaks):
    module = ad.StepExpSurvivalAdapter(breaks=breaks).get_module(None, None)
    result = module("survival", torch.zeros(2, module.get_param_number()), torch.tensor([0.0, 0.5, 2.0]))
    assert torch.isfinite(result).all()


@pytest.mark.parametrize("event, time, message", [
    (None, [1, 2], "both"), ([True], None, "both"),
    ([True, False], [1], "matching"), ([[True]], [[1]], "one-dimensional"),
    ([1, 0], [1, 2], "boolean"),
    ([True, True], [np.nan, 2], "finite nonnegative"),
    ([True, True], [np.inf, 2], "finite nonnegative"),
    ([True, True], [-1, 2], "finite nonnegative"),
    ([True, True], ["a", "b"], "finite nonnegative"),
    ([True, False], [1, np.nan], "finite nonnegative"),
    ([True, False], [1, np.inf], "finite nonnegative"),
    ([True, False], [1, -1], "finite nonnegative"),
    ([True, True], [1j, 2j], "finite nonnegative"),
    ([True, True], [True, False], "finite nonnegative"),
])
@pytest.mark.parametrize("adapter", [
    ad.ExponentialSurvivalAdapter(), ad.WeibullSurvivalAdapter(),
    ad.LogNormalSurvivalAdapter(), ad.LevySurvivalAdapter(),
    ad.InverseGaussianSurvivalAdapter(), ad.FractalNoiseSurvivalAdapter(),
    ad.StepExpSurvivalAdapter(), ad.StepExpSurvivalAdapter(breaks=[0, 1]),
    *(cls() for cls in COMPOSITE_ADAPTERS),
])
def test_shared_event_time_validation(adapter, event, time, message):
    with pytest.raises(ValueError, match=message):
        adapter.get_module(event, time)


@pytest.mark.parametrize("event, time, message", [
    (np.array([], dtype=bool), [], "without observed"),
    ([False, False], [1, 2], "without observed"),
    ([True, True], [0, 0], "two unique"),
    ([True, True], [2, 2], "two unique"),
])
def test_invalid_event_data_for_breaks_rejected(event, time, message):
    with pytest.raises(ValueError, match=message):
        ad.StepExpSurvivalAdapter().get_module(event, time)


@pytest.mark.parametrize("adapter", [
    ad.ExponentialSurvivalAdapter(),
    ad.StepExpSurvivalAdapter(breaks=[0, 1]),
    ad.ProportionalHazardSurvivalAdapter(baseline=ad.ExponentialSurvivalAdapter()),
    ad.AcceleratedFailureTimeSurvivalAdapter(baseline=ad.ExponentialSurvivalAdapter()),
    ad.MixtureSurvivalAdapter(baselines=[
        ad.ExponentialSurvivalAdapter(), ad.ExponentialSurvivalAdapter()
    ]),
])
@pytest.mark.parametrize("event, time", [
    (None, None), (np.array([], dtype=bool), []), ([False, False], [0, 2]),
])
def test_adapters_without_derived_breaks_accept_event_free_data(adapter, event, time):
    assert isinstance(adapter.get_module(event, time), sm.BaseSurvivalModule)


def test_duplicate_quantiles_warn_when_enough_breaks_remain():
    adapter = ad.StepExpSurvivalAdapter(breaks=4)
    with pytest.warns(UserWarning, match="only 2 unique breaks"):
        module = adapter.get_module(np.ones(4, dtype=bool), np.array([1, 1, 1, 2]))
    actual, _ = module._get_time_breaks()
    torch.testing.assert_close(actual, torch.tensor([0.0, 1.25]))


def test_break_quantiles_use_only_observed_events():
    breaks = ad.StepExpSurvivalAdapter.preprocess_breaks(
        3, np.array([True, False, True, True]), np.array([1, 100, 2, 3])
    )
    torch.testing.assert_close(breaks, torch.tensor([0, 5 / 3, 7 / 3]))


@pytest.mark.parametrize("breaks", [torch.tensor([]), torch.tensor([0.0]), torch.tensor([0.0, float("nan")])])
def test_direct_step_module_rejects_invalid_breaks(breaks):
    with pytest.raises(ValueError, match="breaks"):
        sm.StepExpSurvivalModule(breaks)
