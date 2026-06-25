import numpy as np
import pytest
import torch

from survcraft import adapters as ad
from survcraft import loss_modules as lm
from survcraft.util import get_loss_modules, get_subclasses_in_module, get_survival_adapters


@pytest.mark.parametrize("warm_start", [True, False])
@pytest.mark.parametrize("early_stopping", [True, False])
@pytest.mark.parametrize("gradient_clipping", [True, False])
@pytest.mark.parametrize("preload_data", [True, False])
@pytest.mark.parametrize("history", [True, False])
@pytest.mark.parametrize("check_divergence", ad.check_divergence_values)
def test_default_simulator_and_predictor_training_parameters_work_together(history, check_divergence, preload_data, gradient_clipping, early_stopping, warm_start):
    rng = np.random.default_rng(1)
    X = rng.normal(size=(72, 3)).astype(np.float32)
    times = np.linspace(0.1, 2.5, 32, dtype=np.float32)

    simulator = ad.SurvivalSimulator(device="cpu", check_divergence="warn")
    y = simulator.simulate(X=X, times=times, seed=7)
    y_again = simulator.simulate(X=X, times=times, seed=7)

    assert y.dtype.names == ("event", "time")
    np.testing.assert_array_equal(y, y_again)
    assert np.isfinite(y["time"]).all()
    assert (y["time"] >= 0.0).all()
    assert y["event"].any()

    predictor = ad.SurvivalPredictor(
        device="cpu",
        verbose=0,
        batch_size=64,
        learning_rate=0.001,
        weight_decay=0.01,
        epochs=3,
        warm_start=warm_start,
        early_stopping=early_stopping,
        validation_ratio=0.25,
        early_stopping_patience=2,
        data_loader_num_workers=0,
        preload_data=preload_data,
        gradient_clipping=gradient_clipping,
        check_divergence=check_divergence,
        history=history,
    )

    predictor.fit(X, y, X_test=X[:16], y_test=y[:16])

    assert hasattr(predictor, "model_")
    assert predictor.model_.check_divergence == check_divergence
    if history:
        assert 1 <= len(predictor.train_history_) <= predictor.epochs
        for train_losses, test_losses in predictor.train_history_:
            assert train_losses.size > 0
            assert np.isfinite(train_losses).all()
            assert set(test_losses) == {"BrierLoss"}
            assert np.isfinite(list(test_losses.values())).all()

    pred = predictor.predict("failure", X[:10], times[:6])
    assert pred.shape == (10, 6)
    assert np.isfinite(pred).all()
    assert ((pred >= 0.0) & (pred <= 1.0)).all()


def test_invalid_check_divergence_is_rejected_when_model_is_initialized():
    simulator = ad.SurvivalSimulator(check_divergence="invalid")

    with pytest.raises(ValueError, match="check_divergence"):
        simulator.predict("failure", np.zeros((2, 1), dtype=np.float32), np.array([1.0]))

