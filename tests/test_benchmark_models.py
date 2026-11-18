"""Regression checks for benchmark isolation, metrics, and result persistence."""

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from survcraft import adapters as ad, loss_modules as lm


spec = importlib.util.spec_from_file_location(
    "benchmark_models", Path(__file__).resolve().parents[1] / "benchmark/benchmark_models.py",
)
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


@pytest.fixture
def data():
    def targets(times):
        result = np.empty(len(times), dtype=[("event", "?"), ("time", "f8")])
        result["event"] = np.arange(len(times)) % 3 != 0
        result["time"] = times
        return result

    rng = np.random.default_rng(0)
    prepared = {
        "X_train": rng.normal(size=(20, 2)).astype(np.float32),
        "X_test": rng.normal(size=(10, 2)).astype(np.float32),
        "y_train": targets(np.linspace(1, 10, 20)),
        "y_test": targets(np.linspace(2, 8, 10)),
        "y_eval": targets(np.linspace(2, 8, 10)),
        "train_indices": np.arange(20), "test_indices": np.arange(20, 30),
        "metadata": {"n_train": 20, "n_test": 10, "n_features": 2,
                     "train_events": 13, "eval_events": 6},
    }
    prepared["times"] = benchmark._brier_grid(prepared)
    return prepared


def predictor():
    # Factory-side randomness must also be isolated from warm-up fits.
    torch.rand(3)
    return ad.SurvivalPredictor(
        input=ad.FeedForwardNetAdapter(hidden_sizes=[4]),
        survival=ad.ExponentialSurvivalAdapter(), loss=lm.FullLikelihoodLoss(),
        epochs=1, device="cpu",
    )


def test_failures_do_not_prevent_later_cases_or_persisted_results(tmp_path, monkeypatch, data):
    pytest.importorskip("survmetr")

    def loader(name, **kwargs):
        if name == "missing":
            raise ValueError("missing dataset")
        return data

    def unsupported():
        raise NotImplementedError("unsupported example")

    def failing_fit():
        model = predictor()
        model.learning_rate = -1
        return model

    monkeypatch.setattr(benchmark, "load_dataset", loader)
    output = tmp_path / "results.csv"
    rows = benchmark.run_benchmark(
        {"unsupported": unsupported, "failing-fit": failing_fit, "good": predictor},
        ["missing", "valid"],
        repeats=2, warmup_epochs=0, output=output,
    )
    assert [r["status"] for r in rows] == [
        "failed", "success", "unsupported", "unsupported", "failed", "failed",
        "success", "success",
    ]
    assert rows[0]["errors"][0]["stage"] == "preparation"
    assert rows[4]["errors"][0]["stage"] == "fit"
    assert rows[-1]["training_seed"] == 124
    assert rows[-1]["fit_seconds"] > 0
    saved = list(csv.DictReader(output.open()))
    assert len(saved) == len(rows)
    assert saved[0]["fit_seconds"] == ""
    assert "options" not in saved[-1]
    assert saved[-1]["option_epochs"] == "1"
    assert json.loads(saved[-1]["option_input"])["options"]["hidden_sizes"] == [4]
    assert saved[1]["option_alpha"] == "1.0"
    assert saved[1]["option_epochs"] == ""
    assert saved[-1]["option_alpha"] == ""
    assert saved[0]["option_epochs"] == ""
    metadata = json.loads(output.with_suffix(".metadata.json").read_text())
    assert metadata["counts"] == {"failed": 3, "success": 3, "unsupported": 2}
    assert metadata["datasets"]["valid"]["train_indices"] == list(range(20))


@pytest.mark.parametrize("failed_metric", ["antolini_concordance", "integrated_brier_score"])
def test_scoring_failures_are_independent(data, failed_metric):
    class ConstantSurvival:
        def predict_survival(self, X, times):
            return np.full((len(X), len(times)), 0.5)

    def scorer(model, X, y):
        if failed_metric == "antolini_concordance":
            raise ValueError("no comparable pairs")
        return 0.6

    if failed_metric == "integrated_brier_score":
        data["times"] = np.array([3.0])
    row = {"errors": []}
    benchmark._score(ConstantSurvival(), data, scorer, row)
    other = ("integrated_brier_score" if failed_metric == "antolini_concordance"
             else "antolini_concordance")
    assert failed_metric not in row
    assert np.isfinite(row[other])
    assert row["errors"][0]["stage"] == failed_metric


def test_warmup_does_not_change_seeded_measured_model(data):
    scorer = pytest.importorskip("survmetr").AntoliniCIndexVecScorer()
    cold = benchmark._fit_case(predictor, data, {}, 123, scorer)
    warm = benchmark._fit_case(predictor, data, {}, 123, scorer, warmup_epochs=1)
    assert cold["status"] == warm["status"] == "success"
    assert cold["antolini_concordance"] == warm["antolini_concordance"]
    assert cold["integrated_brier_score"] == warm["integrated_brier_score"]


def test_dataset_filter_prevents_fitting(tmp_path, monkeypatch, data):
    pytest.importorskip("survmetr")
    monkeypatch.setattr(benchmark, "load_dataset", lambda *args, **kwargs: data)

    def must_not_run():
        raise AssertionError("excluded dataset reached model factory")

    rows = benchmark.run_benchmark(
        {"unused": must_not_run}, ["small"], min_train_samples=21,
        output=tmp_path / "filtered.csv",
    )
    assert len(rows) == 1
    assert rows[0]["status"] == "excluded"


def test_brier_grid_excludes_endpoint_and_duplicate_quartiles(data):
    data["y_eval"]["time"] = 8
    data["y_eval"]["event"] = True
    assert benchmark._brier_grid(data).size == 0


def test_required_loss_arguments_are_excluded_only_for_all():
    with pytest.warns(UserWarning, match="requires constructor arguments"):
        selected = benchmark._select(["all"], [lm.FullLikelihoodLoss, lm.LinearCombinationLoss], "losses")
    assert selected == [lm.FullLikelihoodLoss]
    with pytest.raises(ValueError, match="requires constructor arguments"):
        benchmark._select(["LinearCombinationLoss"], [lm.LinearCombinationLoss], "losses")


def test_training_cap_is_reproducible_and_preserves_test_split(data):
    data["y_test"]["time"][-1] = 100
    original_indices = data["train_indices"].copy()
    capped = benchmark._cap_training_data(data, 8, seed=3)
    repeated = benchmark._cap_training_data(data, 8, seed=3)
    assert len(capped["X_train"]) == len(capped["y_train"]) == 8
    np.testing.assert_array_equal(capped["train_indices"], repeated["train_indices"])
    selected = capped["train_indices"]
    np.testing.assert_array_equal(capped["X_train"], data["X_train"][selected])
    np.testing.assert_array_equal(capped["y_train"], data["y_train"][selected])
    np.testing.assert_array_equal(data["train_indices"], original_indices)
    assert capped["X_test"] is data["X_test"]
    assert capped["y_test"] is data["y_test"]
    horizon = capped["y_train"]["time"].max()
    assert capped["y_eval"]["time"][-1] == horizon
    assert not capped["y_eval"]["event"][-1]
    assert capped["metadata"]["n_train_before_cap"] == 20
    assert capped["metadata"]["train_events"] == capped["y_train"]["event"].sum()
    assert data["metadata"]["n_train"] == 20


@pytest.mark.parametrize("limit", [None, 20, 30])
def test_training_cap_leaves_small_datasets_unchanged(data, limit):
    assert benchmark._cap_training_data(data, limit, seed=3) is data


@pytest.mark.parametrize("early_stopping", [False, True])
def test_cli_early_stopping_and_training_cap(monkeypatch, early_stopping):
    captured = {}

    def run(models, datasets, **kwargs):
        captured.update(kwargs)
        captured["models"] = [factory() for factory in models.values()]
        return [{"status": "success"}]

    monkeypatch.setattr(benchmark, "run_benchmark", run)
    arguments = ["benchmark_models.py", "--survivals", "ExponentialSurvivalAdapter",
                 "--losses", "FullLikelihoodLoss", "--epochs", "1", "2",
                 "--devices", "cpu", "--max-train-samples", "8"]
    if early_stopping:
        arguments.append("--early-stopping")
    monkeypatch.setattr("sys.argv", arguments)
    assert benchmark.main() == 0
    assert captured["output"] == Path("benchmark/results/models.csv")
    assert captured["max_train_samples"] == 8
    assert len(captured["models"]) == 2
    assert all(model.early_stopping is early_stopping for model in captured["models"])
    assert all(model.learning_rate == 0.005 for model in captured["models"])


def test_cli_rejects_removed_learning_rate_option(monkeypatch):
    monkeypatch.setattr("sys.argv", ["benchmark_models.py", "--learning-rates", "0.01"])
    with pytest.raises(SystemExit) as exc:
        benchmark.main()
    assert exc.value.code == 2


def test_parallel_workers_preserve_results_and_isolate_failures(tmp_path, monkeypatch, data):
    import os

    pytest.importorskip("survmetr")
    monkeypatch.setattr(benchmark, "load_dataset", lambda *args, **kwargs: data)
    parent_pid = os.getpid()
    runs = []
    for workers in (1, 2):
        def make_model():
            if workers > 1:
                assert os.getpid() != parent_pid, "Model ran in the parent process"
            return predictor()

        def broken_model():
            raise ValueError("deliberate factory failure")

        output = tmp_path / f"workers-{workers}.csv"
        rows = benchmark.run_benchmark(
            {"first": make_model, "broken": broken_model, "second": make_model},
            ["valid"], workers=workers, repeats=2, warmup_epochs=1, output=output,
        )
        assert len(rows) == len(list(csv.DictReader(output.open()))) == 7
        metadata = json.loads(output.with_suffix(".metadata.json").read_text())
        assert metadata["settings"]["workers"] == workers
        assert metadata["counts"] == {"success": 5, "failed": 2}
        runs.append({(row["configuration"], row.get("repeat")): row for row in rows})
    assert runs[0].keys() == runs[1].keys()
    for key, sequential in runs[0].items():
        parallel = runs[1][key]
        assert sequential["status"] == parallel["status"]
        assert sequential["errors"] == parallel["errors"]
        if sequential["status"] == "success":
            for metric in ("antolini_concordance", "integrated_brier_score"):
                assert sequential[metric] == pytest.approx(parallel[metric])


def test_worker_count_must_be_positive(tmp_path):
    pytest.importorskip("survmetr")
    with pytest.raises(ValueError, match="workers must be positive"):
        benchmark.run_benchmark({"model": predictor}, ["valid"], workers=0,
                                output=tmp_path / "unused.csv")
