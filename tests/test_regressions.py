import numpy as np
import torch

from survcraft import adapters as ad
from survcraft import loss_modules as lm


def test_linear_combination_loss_flattens_nested_losses():
    loss = lm.BrierLoss() + (lm.PartialLikelihoodLoss() + 2 * lm.BrierLoss())

    assert isinstance(loss, lm.LinearCombinationLoss)
    assert [type(component) for component in loss.losses] == [
        lm.BrierLoss,
        lm.PartialLikelihoodLoss,
        lm.BrierLoss,
    ]
    torch.testing.assert_close(loss.coeffs, torch.tensor([1.0, 1.0, 2.0]))


def test_partial_likelihood_implementations(whas500_data):
    predictor = ad.SurvivalPredictor(
        epochs=5,
        batch_size=64,
        verbose=0,
    )
    predictor.fit(whas500_data["X_train"], whas500_data["y_train"])

    x = torch.tensor(whas500_data["X_test"], dtype=torch.float32)
    event = torch.tensor(whas500_data["y_test"]["event"].copy(), dtype=torch.bool)
    time = torch.tensor(whas500_data["y_test"]["time"].copy(), dtype=torch.float32)

    slow = lm.partial_likelihood_iter(predictor.model_, x, event, time)
    vec = lm.partial_likelihood_vec(
        predictor.model_,
        x,
        event,
        time,
        epsilon=torch.tensor(1e-12),
    )
    vec_safe = lm.partial_likelihood_vec_safe(
        predictor.model_,
        x,
        event,
        time,
        epsilon=torch.tensor(1e-12),
    )

    torch.testing.assert_close(vec, slow, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(vec_safe, slow, atol=1e-6, rtol=1e-6)


def test_squared_loss_supports_aft_predictor(whas500_data):
    predictor = ad.SurvivalPredictor(
        input=ad.FeedForwardNetAdapter(hidden_sizes=[16]),
        survival=ad.AcceleratedFailureTimeSurvivalAdapter(
            baseline=ad.ExponentialSurvivalAdapter()
        ),
        loss=lm.SquaredLoss(),
        epochs=2,
        batch_size=64,
        verbose=0,
    )
    predictor.fit(whas500_data["X_train"], whas500_data["y_train"])

    expected_time = predictor.predict("expected_time", whas500_data["X_test"])
    assert expected_time.shape == (len(whas500_data["X_test"]),)
    assert np.isfinite(expected_time).all()
