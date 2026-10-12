"""Public dataset loading, split reproducibility, and optional sources."""

import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from survcraft import datasets


def test_missing_survset_keeps_sksurv_datasets_loadable(monkeypatch):
    monkeypatch.setitem(sys.modules, "SurvSet", None)
    monkeypatch.setitem(sys.modules, "SurvSet.data", None)
    with pytest.warns(UserWarning, match="SurvSet is not installed"):
        names = datasets.list_datasets()
    assert "sksurv:gbsg2" in names
    assert all(name.startswith("sksurv:") for name in names)
    with pytest.warns(UserWarning, match="SurvSet is not installed"):
        data = datasets.load_dataset("gbsg2", seed=7)
    assert data["metadata"]["dataset"] == "sksurv:gbsg2"
    assert data["X_train"].shape[0] == len(data["y_train"])
    with pytest.warns(UserWarning, match="SurvSet is not installed"):
        with pytest.raises(ValueError, match="unavailable"):
            datasets.load_dataset("survset:mgus")


def test_survset_discovery_excludes_duplicate_and_time_varying_cohorts(monkeypatch):
    catalog = pd.DataFrame({"ds": ["mgus", "actg", "dynamic"],
                            "is_td": [False, False, True]})
    monkeypatch.setitem(sys.modules, "SurvSet.data", SimpleNamespace(
        SurvLoader=lambda: SimpleNamespace(df_ds=catalog)))
    names = datasets.list_datasets()
    assert "survset:mgus" in names
    assert "survset:actg" not in names
    assert "survset:dynamic" not in names
    with pytest.raises(ValueError, match="unavailable"):
        datasets.load_dataset("survset:actg")


def test_public_split_is_reproducible_and_preserves_original_test_targets():
    first = datasets.load_dataset("sksurv:whas500", seed=7, test_size=0.3)
    second = datasets.load_dataset("whas500", seed=7, test_size=0.3)
    for key in first.keys() - {"metadata"}:
        np.testing.assert_array_equal(first[key], second[key])
    assert first["X_train"].dtype == first["X_test"].dtype == np.float32
    assert np.isfinite(first["X_train"]).all()
    assert np.isfinite(first["X_test"]).all()
    assert first["y_train"].dtype == np.dtype([("event", "?"), ("time", "f8")])
    assert len(first["y_test"]) == 150
    assert set(first["train_indices"]).isdisjoint(first["test_indices"])
    assert len(first["train_indices"]) + len(first["test_indices"]) == 500
    _, raw_y, _, _ = datasets._load_raw_dataset(
        {"source": "sksurv", "loader": "load_whas500"})
    np.testing.assert_array_equal(first["y_test"], raw_y[first["test_indices"]])
    beyond = first["y_test"]["time"] > first["metadata"]["evaluation_horizon"]
    assert not first["y_eval"]["event"][beyond].any()
    assert (first["y_eval"]["time"][beyond] ==
            first["metadata"]["evaluation_horizon"]).all()


def test_unknown_and_ambiguous_names_are_rejected(monkeypatch):
    monkeypatch.setattr(datasets, "_discover_datasets", lambda: [
        {"dataset": "sksurv:same", "status": "included"},
        {"dataset": "survset:same", "status": "included"},
    ])
    for name in ("unknown", "same"):
        with pytest.raises(ValueError, match="unavailable or ambiguous"):
            datasets.load_dataset(name)


@pytest.mark.parametrize("test_size", [0, 1, -0.1, 1.1])
def test_invalid_test_fraction_is_rejected(test_size):
    with pytest.raises(ValueError, match="test_size"):
        datasets.load_dataset("gbsg2", test_size=test_size)


def test_zero_time_shift_reports_original_zero_count(monkeypatch):
    from sksurv import datasets as sksurv_datasets

    X = pd.DataFrame({"feature": [1., 2., 3.]})
    y = np.array([(True, 0.), (False, 2.), (True, 4.)],
                 dtype=[("event", "?"), ("time", "f8")])
    monkeypatch.setattr(sksurv_datasets, "load_whas500", lambda: (X.copy(), y))
    entry = {"source": "sksurv", "loader": "load_whas500"}
    _, shifted, _, metadata = datasets._load_raw_dataset(entry)
    np.testing.assert_array_equal(shifted["time"], [2., 4., 6.])
    np.testing.assert_array_equal(y["time"], [0., 2., 4.])
    assert metadata["zero_times"] == 1
    y["time"] = 0
    with pytest.raises(ValueError, match="strictly positive"):
        datasets._load_raw_dataset(entry)
