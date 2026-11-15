import copy
import warnings

import numpy as np
import pytest
import torch

from survcraft import adapters as ad
from survcraft.loss_modules import BaseSurvivalLoss


class BufferedLoss(BaseSurvivalLoss):
    def __init__(self, scale):
        super().__init__()
        self.register_buffer("scale", torch.tensor(scale))

    def forward(self, model, X, event, time, *, check_divergence="raise"):
        assert self.scale.device == X.device
        if model.training:
            return self.scale * model.get_raw_params(X).square().mean()
        return self.scale * len(X)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_loss_buffers_and_validation_batch_mean(device, capsys):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    loss = BufferedLoss(0.5) + BufferedLoss(0.5)
    test_loss = BufferedLoss(2.0)
    predictor = ad.SurvivalPredictor(
        loss=loss, device=device, epochs=1, batch_size=2,
        early_stopping=True, validation_ratio=0.375, history=True, verbose=3,
    )
    X = np.ones((8, 2), dtype=np.float32)
    event = np.ones(8, dtype=bool)
    time = np.ones(8, dtype=np.float32)
    predictor.train(X, event, time, test_data=(X, event, time), test_losses=[test_loss])

    assert all(buffer.device == torch.device(device) for buffer in loss.buffers())
    assert test_loss.scale.device == torch.device(device)
    # Validation has batches of sizes 2 and 1: preserve their unweighted mean.
    assert "validation loss = 1.5" in capsys.readouterr().out
    assert predictor.train_history_[0][1] == {"BufferedLoss": 16.0}


class ScheduledValidationLoss:
    def __init__(self):
        self.validation_states = []
        self.training_states = []
        self.training_calls = 0

    def __call__(self, model, X, event, time, *, check_divergence="raise"):
        if model.training:
            self.training_calls += 1
            self.training_states.append(copy.deepcopy(model.state_dict()))
            return model.get_raw_params(X).square().mean()
        self.validation_states.append(copy.deepcopy(model.state_dict()))
        return torch.tensor([3.0, 1.0, 2.0][len(self.validation_states) - 1])


@pytest.mark.parametrize("epochs,patience", [(3, 10), (5, 1)])
@pytest.mark.parametrize("warm_start", [False, True])
@pytest.mark.parametrize("bulk_batching", [False, True])
def test_early_stopping_restores_best_parameters_and_buffers(epochs, patience, warm_start, bulk_batching):
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
        bulk_batching=bulk_batching,
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

    def __call__(self, model, X, event, time, *, check_divergence="raise"):
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
@pytest.mark.parametrize("bulk_batching", [False, True])
@pytest.mark.parametrize("policy", ["warn", "skip"])
def test_skipped_batches_report_the_actual_reasons(events, nonfinite_samples, expected, failed, bulk_batching, policy):
    X = np.arange(len(events), dtype=np.float32).reshape(-1, 1)
    loss = BatchLoss(nonfinite_samples)
    predictor = ad.SurvivalPredictor(loss=loss, epochs=1, batch_size=1,
                                     device="cpu", verbose=0, history=True,
                                     bulk_batching=bulk_batching, check_divergence=policy,
                                     skip_event_free_batches=True)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if failed:
            with pytest.raises(ad.FailedConvergence) as error:
                predictor.train(X, np.array(events), np.ones(len(events)))
            message = str(error.value)
            assert "no usable training batches remain" in message
        else:
            predictor.train(X, np.array(events), np.ones(len(events)))
            message = str(caught[-1].message)
            assert len(predictor.train_history_[0][0]) == 1
            assert "no usable training batches remain" not in message

    summaries = [str(w.message) for w in caught if "Skipped training batches" in str(w.message)]
    assert len(summaries) == 1
    expected_individual_warnings = len(nonfinite_samples) if policy == "warn" else 0
    assert len(caught) == 1 + expected_individual_warnings
    for reason in expected:
        assert reason in message
        assert reason in summaries[0]
    if not nonfinite_samples:
        assert "non-finite" not in message
    if all(events):
        assert "without events" not in message
    assert loss.calls == sum(events)


def test_zero_epochs_with_early_stopping_does_not_restore_missing_state():
    predictor = ad.SurvivalPredictor(epochs=0, early_stopping=True, device="cpu", verbose=0)
    predictor.train(np.ones((8, 2)), np.ones(8, dtype=bool), np.ones(8))
    assert hasattr(predictor, "model_")


def test_no_divergence_checks_optimizes_nonfinite_loss():
    def nonfinite_loss(model, X, event, time, *, check_divergence="raise"):
        return model.get_raw_params(X).square().mean() + float("nan")

    predictor = ad.SurvivalPredictor(
        loss=nonfinite_loss, check_divergence="no", epochs=1,
        batch_size=4, history=True, device="cpu", verbose=0,
    )
    X = np.ones((4, 2), dtype=np.float32)
    predictor._init_model(X)
    before = copy.deepcopy(predictor.model_.state_dict())
    predictor.train_history_ = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        predictor.train(X, np.ones(4, dtype=bool), np.ones(4), warm_start=True)

    assert not caught
    assert np.isnan(predictor.train_history_[0][0]).all()
    assert any(not torch.equal(value, before[key])
               for key, value in predictor.model_.state_dict().items())


def test_raise_policy_rejects_nonfinite_loss_before_update():
    predictor = ad.SurvivalPredictor(
        loss=BatchLoss((0,)), check_divergence="raise", epochs=1, device="cpu",
    )
    X = np.zeros((4, 2), dtype=np.float32)
    predictor._init_model(X)
    predictor.train_history_ = []
    before = copy.deepcopy(predictor.model_.state_dict())
    with pytest.raises(ad.FailedConvergence, match="non-finite loss"):
        predictor.train(X, np.ones(4, dtype=bool), np.ones(4), warm_start=True)
    for key, value in predictor.model_.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)


@pytest.mark.parametrize("early_stopping", [False, True])
@pytest.mark.parametrize("device,preload_data,num_workers", [
    ("cpu", False, 0),
    ("cpu", False, 2),
    ("cpu", True, 0),
    ("cuda:0", False, 0),
    ("cuda:0", False, 2),
    ("cuda:0", True, 0),
])
def test_bulk_batching_preserves_batches_weights_and_history(early_stopping, device, preload_data, num_workers):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    X = np.arange(111, dtype=np.float32).reshape(37, 3) / 111
    event = np.ones(37, dtype=bool)
    time = np.linspace(0.1, 3.0, 37, dtype=np.float32)
    runs = []
    for bulk_batching in (False, True):
        np.random.seed(123)
        torch.manual_seed(123)
        batches = []

        def recording_loss(model, Xb, eb, tb, *, check_divergence="raise"):
            batches.append((model.training, tuple(t.detach().cpu().clone()
                                                  for t in (Xb, eb, tb))))
            return model.get_raw_params(Xb).square().mean()

        predictor = ad.SurvivalPredictor(
            loss=recording_loss, device=device, batch_size=8, epochs=2,
            early_stopping=early_stopping, validation_ratio=0.25,
            preload_data=preload_data, data_loader_num_workers=num_workers,
            history=True, bulk_batching=bulk_batching,
        )
        predictor.train(X, event, time)
        runs.append((predictor, batches))

    (legacy, legacy_batches), (bulk, bulk_batches) = runs
    assert len(legacy_batches) == len(bulk_batches)
    for (legacy_training, legacy_tensors), (bulk_training, bulk_tensors) in zip(legacy_batches, bulk_batches):
        assert legacy_training == bulk_training
        for old, new in zip(legacy_tensors, bulk_tensors):
            torch.testing.assert_close(old, new, rtol=0, atol=0)
    # Neither loader drops partial training or validation batches.
    for training in ([True, False] if early_stopping else [True]):
        sizes = [len(tensors[0]) for is_training, tensors in bulk_batches
                 if is_training == training]
        assert any(size < bulk.batch_size for size in sizes)
    for key, value in legacy.model_.state_dict().items():
        torch.testing.assert_close(value, bulk.model_.state_dict()[key], rtol=0, atol=0)
    assert len(legacy.train_history_) == len(bulk.train_history_) == 2
    for old, new in zip(legacy.train_history_, bulk.train_history_):
        np.testing.assert_array_equal(old[0], new[0])


@pytest.mark.parametrize("events", [[False, False], [False, True]])
@pytest.mark.parametrize("bulk_batching", [False, True])
def test_default_trains_event_free_batches_without_checking(events, bulk_batching, monkeypatch):
    def unexpected_any(*args, **kwargs):
        raise AssertionError("Training must not check for events by default")

    loss = BatchLoss()
    predictor = ad.SurvivalPredictor(
        loss=loss, epochs=1, batch_size=1, history=True,
        bulk_batching=bulk_batching, check_divergence="no")
    assert predictor.skip_event_free_batches is False
    monkeypatch.setattr(torch.Tensor, "any", unexpected_any)
    predictor.train(np.ones((2, 1), dtype=np.float32),
                    np.array(events), np.ones(2))
    assert loss.calls == 2
    assert len(predictor.train_history_[0][0]) == 2
