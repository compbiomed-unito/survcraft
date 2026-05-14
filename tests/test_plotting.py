import pytest

matplotlib = pytest.importorskip("matplotlib")
pytest.importorskip("pandas")
pytest.importorskip("seaborn")

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from survcraft import plotting


class DummyRawParamModel:
    def get_raw_params(self, X):
        rows = X.shape[0]
        base = torch.arange(rows, dtype=torch.float32).unsqueeze(1)
        return torch.cat([base, base + 0.5], dim=1)


class DummyPlotModel:
    def __init__(self):
        self.model_ = DummyRawParamModel()
        self.predict_calls = []
        self.train_history_ = [
            ([np.float32(4.0), np.float32(2.0)], {"brier": np.float32(3.0), "nll": np.float32(6.0)}),
            ([np.float32(2.0), np.float32(1.0)], {"brier": np.float32(1.5), "nll": np.float32(3.0)}),
            ([np.float32(1.0), np.float32(0.5)], {"brier": np.float32(0.75), "nll": np.float32(1.5)}),
        ]

    def _tensor(self, X):
        return torch.as_tensor(X, dtype=torch.float32)

    def predict(self, mode, X, times=None):
        X = np.asarray(X, dtype=np.float32)
        times_array = None if times is None else np.asarray(times, dtype=np.float32)
        self.predict_calls.append((mode, X.shape, times_array))

        if mode == "failure":
            scaled = times_array / max(times_array.max(), 1.0)
            return np.clip(0.2 + 0.6 * scaled + 0.01 * np.arange(len(X))[:, None], 0.0, 1.0)
        if mode == "survival":
            scaled = times_array / max(times_array.max(), 1.0)
            return np.clip(0.8 - 0.6 * scaled - 0.01 * np.arange(len(X))[:, None], 0.0, 1.0)
        if mode == "density":
            return np.full((len(X), len(times_array)), 0.15, dtype=np.float32)
        if mode == "hazard":
            return np.full((len(X), len(times_array)), 0.2, dtype=np.float32)
        raise AssertionError(f"Unexpected prediction mode: {mode}")


@pytest.fixture(autouse=True)
def close_figures():
    plt.close("all")
    yield
    plt.close("all")


def test_plot_model_draws_failure_curves_and_event_markers(monkeypatch, capsys):
    model = DummyPlotModel()
    X = np.arange(32, dtype=np.float32).reshape(8, 4)
    event = np.array([True, False, True, False, True, False, True, False])
    time = np.array([5.0, 12.0, 8.0, 15.0, 10.0, 20.0, 7.0, 18.0], dtype=np.float32)

    class FixedRng:
        def choice(self, population, size):
            assert population == len(X)
            assert size == 5
            return np.array([0, 1, 2, 3, 4])

    monkeypatch.setattr(plotting.np.random, "default_rng", lambda: FixedRng())

    plotting.plot_model(model, X, event, time)
    capsys.readouterr()

    fig = plt.gcf()
    assert len(fig.axes) == 1
    assert len(fig.axes[0].lines) == 10

    mode, shape, times_used = model.predict_calls[0]
    assert mode == "failure"
    assert shape == (5, 4)
    assert len(times_used) == 200


def test_plot_outputs_uses_inferred_max_time_and_builds_expected_axes(monkeypatch, capsys):
    model = DummyPlotModel()
    X = np.arange(12, dtype=np.float32).reshape(3, 4)

    monkeypatch.setattr(plotting, "detect_max_survival_time", lambda model, X: 7.5)

    plotting.plot_outputs(model, X)
    captured = capsys.readouterr()

    assert "estimated max time:" in captured.out
    assert [call[0] for call in model.predict_calls] == [
        "failure",
        "survival",
        "density",
        "hazard",
    ]

    fig = plt.gcf()
    assert len(fig.axes) == 4
    assert [ax.get_title() for ax in fig.axes] == [
        "failure",
        "survival",
        "density",
        "hazard",
    ]
    assert fig.axes[3].get_yscale() == "log"

    first_times = model.predict_calls[0][2]
    assert len(first_times) == 101
    assert first_times[0] == pytest.approx(0.0)
    assert first_times[-1] == pytest.approx(7.5)


def test_get_training_history_table_returns_train_and_test_metrics():
    model = DummyPlotModel()

    table = plotting.get_training_history_table(model)

    assert table.shape == (3, 4)
    assert table.columns.tolist() == [
        ("train", 0),
        ("train", 1),
        ("test", "brier"),
        ("test", "nll"),
    ]
    assert table.iloc[0].tolist() == [4.0, 2.0, 3.0, 6.0]


def test_plot_training_history_normalizes_metrics_and_uses_log_scale():
    model = DummyPlotModel()

    plotting.plot_training_history(model)

    fig = plt.gcf()
    assert len(fig.axes) == 1
    assert fig.axes[0].get_yscale() == "log"
    assert len(fig.axes[0].lines) >= 3
