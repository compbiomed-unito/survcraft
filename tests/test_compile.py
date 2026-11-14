"""Integration checks for the optional training-loss compiler."""
import numpy as np
import pytest
import torch
from sklearn.base import clone

from survcraft import adapters as ad, loss_modules as lm


@pytest.mark.parametrize("loss_type", [lm.FullLikelihoodLoss, lm.BCEBatchTimesLoss, lm.BrierBatchTimesLoss])
def test_compiled_training_matches_eager(loss_type):
    rng = np.random.default_rng(12)
    X = rng.normal(size=(35, 3)).astype("float32")
    event = np.ones(35, dtype=bool)
    event[::3] = False
    times = rng.uniform(.1, 3, 35).astype("float32")
    predictors = []
    for enabled in [False, True]:
        torch.manual_seed(42)
        np.random.seed(42)
        predictor = ad.SurvivalPredictor(
            loss=loss_type(gpu_mode=True), epochs=3, batch_size=8,
            early_stopping=True, validation_ratio=.2,
            check_divergence="no", compile=enabled,
            compile_kwargs={"backend": "aot_eager"})
        predictor.train(X, event, times)
        predictors.append(predictor)
    eager, compiled = predictors
    assert isinstance(compiled.model_, ad.TorchModel)
    assert clone(compiled).compile is True
    assert clone(compiled).compile_kwargs == {"backend": "aot_eager"}
    assert eager.model_.state_dict().keys() == compiled.model_.state_dict().keys()
    for name, expected in eager.model_.state_dict().items():
        torch.testing.assert_close(compiled.model_.state_dict()[name], expected)
    np.testing.assert_allclose(compiled.predict_survival(X, [.5, 1.]),
                               eager.predict_survival(X, [.5, 1.]), rtol=1e-5)


def test_compile_options_and_training_boundary(monkeypatch):
    calls = []
    compiled_calls = []

    def compile_loss(fn, **kwargs):
        calls.append(kwargs)
        def wrapped(model, *args, **options):
            assert model.training
            compiled_calls.append(options["check_divergence"])
            return fn(model, *args, **options)
        return wrapped

    monkeypatch.setattr(torch, "compile", compile_loss)
    predictor = ad.SurvivalPredictor(epochs=1, batch_size=4,
                                    check_divergence="no", compile=True,
                                    early_stopping=True, validation_ratio=.2)
    X = np.ones((12, 2), dtype="float32")
    event = np.ones(12, dtype=bool)
    time = np.ones(12, dtype="float32")
    predictor.train(X, event, time)
    predictor.predict_survival(X, [1.])
    assert calls == [{"mode": "reduce-overhead"}]
    assert compiled_calls == ["no"] * 3
    predictor.compile_kwargs = {"mode": "default", "dynamic": True}
    predictor.train(X, event, time, warm_start=True)
    assert calls[-1] == predictor.compile_kwargs
    predictor.compile_kwargs = {}
    predictor.train(X, event, time)
    assert calls[-1] == {}
    predictor.compile = False
    predictor.train(X, event, time)
    assert len(calls) == 3


def test_compile_defaults_are_independent():
    first, second = ad.SurvivalPredictor(), ad.SurvivalPredictor()
    assert first.compile is False
    first.compile_kwargs["dynamic"] = True
    assert second.compile_kwargs == {"mode": "reduce-overhead"}
