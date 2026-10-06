import numpy as np
import pytest

from survcraft import adapters as ad


def test_fit_accepts_independent_target_field_names():
    X = np.ones((4, 2), dtype=np.float32)
    y = np.array([(True, 1), (False, 2), (True, 3), (True, 4)],
                 dtype=[("status", "?"), ("duration", "i4")])
    y_test = np.array([(True, 1.5), (False, 2.5)],
                      dtype=[("observed", "?"), ("followup", "f8")])
    predictor = ad.SurvivalPredictor(device="cpu", epochs=1, verbose=0, history=True)

    assert predictor.fit(X, y, X_test=X[:2], y_test=y_test) is predictor
    assert len(predictor.train_history_) == 1
    assert np.isfinite(predictor.predict_survival(X, [1.0, 2.0])).all()


@pytest.mark.parametrize("target, message", [
    ([True, 1], "one-dimensional NumPy structured array"),
    (np.zeros((2, 2), dtype=[("event", "?"), ("time", "f4")]), "one-dimensional"),
    (np.zeros(2), "exactly two fields"),
    (np.zeros(2, dtype=[("event", "?")]), "exactly two fields"),
    (np.zeros(2, dtype=[("event", "?"), ("time", "f4"), ("extra", "f4")]), "exactly two fields"),
    (np.zeros(2, dtype=[("event", "i4"), ("time", "f4")]), "boolean events"),
    (np.zeros(2, dtype=[("time", "f4"), ("event", "?")]), "boolean events"),
    (np.zeros(2, dtype=[("event", "?", (2,)), ("time", "f4")]), "boolean events"),
    *[(np.zeros(2, dtype=[("event", "?"), ("time", dtype)]), "real numeric times")
      for dtype in ["?", "U4", "O", "c8", "M8[D]"]],
    (np.zeros(2, dtype=[("event", "?"), ("time", "f4", (2,))]), "real numeric times"),
    *[(np.array([(True, time)], dtype=[("event", "?"), ("time", "f8")]),
       "finite nonnegative") for time in [np.nan, np.inf, -1]],
])
@pytest.mark.parametrize("target_name", ["y", "y_test"])
@pytest.mark.parametrize("warm_start", [False, True])
def test_fit_rejects_malformed_targets_before_training(monkeypatch, target, message,
                                                       target_name, warm_start):
    predictor = ad.SurvivalPredictor(warm_start=warm_start)

    def unexpected_train(**kwargs):
        pytest.fail("invalid targets must be rejected before training")

    monkeypatch.setattr(predictor, "train", unexpected_train)
    X = np.ones((2, 1), dtype=np.float32)
    valid = np.array([(True, 1), (False, 2)], dtype=[("event", "?"), ("time", "f4")])
    kwargs = {"y": valid, "y_test": valid, target_name: target}
    with pytest.raises(ValueError, match=message) as error:
        predictor.fit(X, X_test=X, **kwargs)
    assert str(error.value).startswith(target_name)
