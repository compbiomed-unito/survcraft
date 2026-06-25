import numpy as np
import pytest
import torch

from survcraft import adapters as ad
from survcraft import loss_modules as lm
from survcraft.util import get_loss_modules, get_subclasses_in_module, get_survival_adapters


def build_predictor(survival, loss):
    return ad.SurvivalPredictor(
        input=ad.FeedForwardNetAdapter(hidden_sizes=[16]),
        survival=survival,
        loss=loss,
        epochs=4,
        batch_size=32,
        verbose=0,
        check_divergence="raise",
    )


def test_predictor_fit_and_predict_core_modes_on_reference_data(whas500_data):
    predictor = build_predictor(
        survival=ad.StepExpSurvivalAdapter(breaks=8),
        loss=lm.BrierLoss(),
    )
    predictor.fit(whas500_data["X_train"], whas500_data["y_train"])

    times = np.array([1.0, 3.0, 5.0], dtype=np.float32)
    for mode in ["failure", "survival", "density", "hazard"]:
        pred = predictor.predict(mode, whas500_data["X_test"], times)
        assert pred.shape == (len(whas500_data["X_test"]), len(times))
        assert np.isfinite(pred).all()
        if mode in {"failure", "survival"}:
            assert ((pred >= 0.0) & (pred <= 1.0)).all()
        else:
            assert (pred >= 0.0).all()


def test_simulated_targets_can_roundtrip_through_predictor(simulated_dataset):
    X, y, times = simulated_dataset
    assert y.dtype.names == ("event", "time")
    assert y["event"].any()
    assert (~y["event"]).any()

    predictor = build_predictor(
        survival=ad.WeibullSurvivalAdapter(),
        loss=lm.BrierLoss(),
    )
    predictor.fit(X, y)

    pred = predictor.predict("failure", X[:12], times[:5])
    assert pred.shape == (12, 5)
    assert np.isfinite(pred).all()


def module_id(module_cls):
    return module_cls.__name__.removesuffix("Adapter").removesuffix("Loss")


def build_discovered_input(input_cls):
    if input_cls is ad.FeedForwardNetAdapter:
        return input_cls(hidden_sizes=[4])
    return input_cls()


def build_discovered_survival(survival_cls):
    if survival_cls is ad.ProportionalHazardSurvivalAdapter:
        return survival_cls(baseline=ad.ExponentialSurvivalAdapter())
    if survival_cls is ad.AcceleratedFailureTimeSurvivalAdapter:
        return survival_cls(baseline=ad.ExponentialSurvivalAdapter())
    if survival_cls is ad.MixtureSurvivalAdapter:
        return survival_cls(
            baselines=[
                ad.ExponentialSurvivalAdapter(),
                ad.WeibullSurvivalAdapter(),
            ]
        )
    if survival_cls is ad.StepExpSurvivalAdapter:
        return survival_cls(breaks=3)
    return survival_cls()


def build_discovered_loss(loss_cls):
    if loss_cls is lm.LinearCombinationLoss:
        return loss_cls(
            [lm.BrierLoss(), lm.PartialLikelihoodLoss()],
            torch.tensor([1.0, 0.1]),
        )
    if issubclass(loss_cls, lm.ClassificationBatchTimesLoss):
        return loss_cls(max_times=3, unique_times=True)
    return loss_cls()


DISCOVERED_INPUT_ADAPTERS = [
    input_cls
    for input_cls in get_subclasses_in_module(ad, ad.BaseInputAdapter)
    if input_cls is not ad.LinearFunctionInputAdapter
]
DISCOVERED_SURVIVAL_ADAPTERS = [
    survival_cls
    for survival_cls in get_survival_adapters()
    if survival_cls is not ad.FractalNoiseSurvivalAdapter
]
DISCOVERED_LOSSES = get_loss_modules()

EXPECTED_INCOMPATIBLE_COMBINATIONS = {
    (ad.LevySurvivalAdapter, lm.SquaredLoss),
    (ad.ProportionalHazardSurvivalAdapter, lm.SquaredLoss),
}


@pytest.mark.parametrize("input_cls", DISCOVERED_INPUT_ADAPTERS, ids=module_id)
@pytest.mark.parametrize("survival_cls", DISCOVERED_SURVIVAL_ADAPTERS, ids=module_id)
@pytest.mark.parametrize("loss_cls", DISCOVERED_LOSSES, ids=module_id)
def test_predictor_fit_and_predict_with_all_discovered_trainable_modules(
    input_cls,
    survival_cls,
    loss_cls,
    simulated_dataset,
):
    if (survival_cls, loss_cls) in EXPECTED_INCOMPATIBLE_COMBINATIONS:
        pytest.xfail(f"{survival_cls.__name__} does not implement expected_time")

    X, y, times = simulated_dataset
    predictor = ad.SurvivalPredictor(
        input=build_discovered_input(input_cls),
        survival=build_discovered_survival(survival_cls),
        loss=build_discovered_loss(loss_cls),
        epochs=1,
        batch_size=len(X),
        learning_rate=0.001,
        verbose=0,
        check_divergence="raise",
    )

    predictor.fit(X, y)

    pred = predictor.predict("failure", X[:8], times[:4])
    assert pred.shape == (8, 4)
    assert np.isfinite(pred).all()
    assert ((pred >= 0.0) & (pred <= 1.0)).all()
