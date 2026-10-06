import numpy as np
import pytest

from survcraft import adapters as ad


@pytest.mark.parametrize(
    "times, message",
    [
        (0, "one-dimensional"),
        ([], "at least two"),
        ([0], "at least two"),
        ([[0, 1]], "one-dimensional"),
        ([[0], [1, 2]], "one-dimensional"),
        (["0", "1"], "finite nonnegative"),
        ([False, True], "finite nonnegative"),
        ([0j, 1j], "finite nonnegative"),
        ([0, np.nan], "finite nonnegative"),
        ([0, np.inf], "finite nonnegative"),
        ([0, -np.inf], "finite nonnegative"),
        ([-1, 0, 1], "finite nonnegative"),
        ([0, -1], "finite nonnegative"),
        ([0.1, 1], "start at zero"),
        ([0, 0], "strictly increasing"),
        ([0, 1, 1], "strictly increasing"),
        ([0, 2, 1], "strictly increasing"),
        ([0, 1, 1 + 1e-9], "strictly increasing after float32"),
        ([0, 1e-50], "strictly increasing after float32"),
        ([0, 1e40], "finite after float32"),
    ],
)
def test_invalid_grid_is_rejected_before_model_initialization(times, message):
    simulator = ad.SurvivalSimulator()
    with pytest.raises(ValueError, match=message):
        simulator.simulate(np.zeros((2, 1), dtype=np.float32), times)
    assert not hasattr(simulator, "model_")


@pytest.mark.parametrize("grid", [[0, 1, 3], (0.0, 1.0, 3.0), np.array([0, 1, 3])])
def test_sampling_uses_interval_probabilities_midpoints_and_final_censoring(grid, monkeypatch):
    simulator = ad.SurvivalSimulator()
    X = np.zeros((128, 1), dtype=np.float32)

    def predict(mode, features, times):
        assert mode == "failure"
        assert times.dtype == np.float32
        return np.tile(np.array([0, 0.25, 0.75], dtype=np.float32), (len(features), 1))

    monkeypatch.setattr(simulator, "predict", predict)
    result = simulator.simulate(X, grid, seed=7)
    repeated = simulator.simulate(X, grid, seed=7)
    rng = np.random.default_rng(7)
    intervals = np.array([rng.choice(3, p=[0.25, 0.5, 0.25]) for _ in X])

    assert set(intervals) == {0, 1, 2}
    assert result.dtype == np.dtype([("event", "?"), ("time", "f4")])
    np.testing.assert_array_equal(result["time"], np.array([0.5, 2, 3])[intervals])
    np.testing.assert_array_equal(result["event"], intervals < 2)
    np.testing.assert_array_equal(result, repeated)


@pytest.mark.parametrize("grid", [[0, 2], [0, 2e38, 3e38]])
def test_certain_events_and_censoring_with_minimal_and_large_grids(grid, monkeypatch):
    simulator = ad.SurvivalSimulator()
    X = np.zeros((2, 1), dtype=np.float32)
    failure = np.zeros((2, len(grid)), dtype=np.float32)
    failure[0, -1] = 1
    monkeypatch.setattr(simulator, "predict", lambda *args: failure)

    result = simulator.simulate(X, grid, seed=0)
    normalized_grid = np.asarray(grid, dtype=np.float32).astype(np.float64)
    midpoint = np.float32((normalized_grid[-2] + normalized_grid[-1]) / 2)
    np.testing.assert_array_equal(result["event"], [True, False])
    np.testing.assert_array_equal(result["time"], [midpoint, np.float32(grid[-1])])
    assert np.isfinite(result["time"]).all()


def test_default_simulator_includes_early_events_and_tail_censoring():
    simulator = ad.SurvivalSimulator()
    X = np.zeros((10000, 1), dtype=np.float32)
    grid = np.array([0, 1, 3], dtype=np.float32)
    failure = simulator.predict("failure", X[:1], grid)[0]
    result = simulator.simulate(X, grid, seed=10)

    assert failure[0] == 0
    frequencies = np.array([(result["time"] == t).mean() for t in [0.5, 2, 3]])
    expected = [failure[1], failure[2] - failure[1], 1 - failure[2]]
    np.testing.assert_allclose(frequencies, expected, atol=0.015, rtol=0)
    np.testing.assert_array_equal(result["event"], result["time"] < grid[-1])
