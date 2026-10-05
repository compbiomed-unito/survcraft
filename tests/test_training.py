import copy
import warnings

import numpy as np
import pytest
import torch

from survcraft import adapters as ad


class ScheduledValidationLoss:
    def __init__(self):
        self.validation_states = []
        self.training_states = []
        self.training_calls = 0

    def __call__(self, model, X, event, time):
        if model.training:
            self.training_calls += 1
            self.training_states.append(copy.deepcopy(model.state_dict()))
            return model.get_raw_params(X).square().mean()
        self.validation_states.append(copy.deepcopy(model.state_dict()))
        return torch.tensor([3.0, 1.0, 2.0][len(self.validation_states) - 1])


@pytest.mark.parametrize("epochs,patience", [(3, 10), (5, 1)])
@pytest.mark.parametrize("warm_start", [False, True])
def test_early_stopping_restores_best_parameters_and_buffers(epochs, patience, warm_start):
    X = np.arange(64, dtype=np.float32).reshape(32, 2) / 64
    event = np.ones(32, dtype=bool)
    time = np.ones(32, dtype=np.float32)
    loss = ScheduledValidationLoss()
    predictor = ad.SurvivalPredictor(
        input=ad.FeedForwardNetAdapter(hidden_sizes=[4], batch_norm=True),
        loss=loss,
        early_stopping=True,
        early_stopping_patience=patience,
        validation_ratio=0.25,
        epochs=epochs,
        batch_size=32,
        device="cpu",
        verbose=0,
    )
    if warm_start:
        predictor.train(X, event, time)
        # A second call must select its own best epoch.
        loss.validation_states.clear()
        loss.training_calls = 0

    predictor.train(X, event, time, warm_start=warm_start)

    assert loss.training_calls == len(loss.validation_states) == 3
    best_state = loss.validation_states[1]
    assert any(not torch.equal(best_state[key], loss.validation_states[2][key])
               for key in best_state)
    for key, value in predictor.model_.state_dict().items():
        torch.testing.assert_close(value, best_state[key], rtol=0, atol=0)


def test_no_early_stopping_keeps_final_weights():
    loss = ScheduledValidationLoss()
    predictor = ad.SurvivalPredictor(loss=loss, epochs=3, batch_size=4,
                                     device="cpu", verbose=0)
    X = np.ones((4, 2), dtype=np.float32)
    predictor.train(X, np.ones(4, dtype=bool), np.ones(4))
    assert loss.training_calls == 3
    assert loss.validation_states == []
    assert any(not torch.equal(value, loss.training_states[-1][key])
               for key, value in predictor.model_.state_dict().items())


class BatchLoss:
    def __init__(self, nonfinite_samples=()):
        self.nonfinite_samples = nonfinite_samples
        self.calls = 0

    def __call__(self, model, X, event, time):
        self.calls += 1
        if X[0, 0].item() in self.nonfinite_samples:
            return torch.tensor(float("nan"))
        return model.get_raw_params(X).square().mean()


@pytest.mark.parametrize(
    "events,nonfinite_samples,expected,failed",
    [
        ([False, False], (), ["batches without events: 2"], True),
        ([True, True], (0, 1), ["batches with non-finite losses: 2"], True),
        ([False, True], (1,), ["batches without events: 1",
                              "batches with non-finite losses: 1"], True),
        ([False, True, True], (1,), ["batches without events: 1",
                                    "batches with non-finite losses: 1"], False),
        ([False, True], (), ["batches without events: 1"], False),
    ],
)
def test_skipped_batches_report_the_actual_reasons(events, nonfinite_samples, expected, failed):
    X = np.arange(len(events), dtype=np.float32).reshape(-1, 1)
    loss = BatchLoss(nonfinite_samples)
    predictor = ad.SurvivalPredictor(loss=loss, epochs=1, batch_size=1,
                                     device="cpu", verbose=0, history=True)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if failed:
            with pytest.raises(ad.FailedConvergence) as error:
                predictor.train(X, np.array(events), np.ones(len(events)))
            message = str(error.value)
            assert "no usable training batches remain" in message
        else:
            predictor.train(X, np.array(events), np.ones(len(events)))
            message = str(caught[0].message)
            assert len(predictor.train_history_[0][0]) == 1
            assert "no usable training batches remain" not in message

    assert len(caught) == 1
    for reason in expected:
        assert reason in message
        assert reason in str(caught[0].message)
    if not nonfinite_samples:
        assert "non-finite" not in message
    if all(events):
        assert "without events" not in message
    assert loss.calls == sum(events)


def test_zero_epochs_with_early_stopping_does_not_restore_missing_state():
    predictor = ad.SurvivalPredictor(epochs=0, early_stopping=True, device="cpu", verbose=0)
    predictor.train(np.ones((8, 2)), np.ones(8, dtype=bool), np.ones(8))
    assert hasattr(predictor, "model_")
