"""Check PDF grouping and metric filtering without running any model fits."""

import csv
import importlib.util
from pathlib import Path

import pytest

plt = pytest.importorskip("matplotlib.pyplot")
sns = pytest.importorskip("seaborn")

spec = importlib.util.spec_from_file_location(
    "benchmark_models",
    Path(__file__).resolve().parents[1] / "benchmark/benchmark_models.py",
)
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "results.csv"
    fields = ["dataset", "status", "configuration", "option_input", *report.METRICS]
    cases = [
        ("a", "success", "m1", '{"x": 1, "y": 2}', 0.6, 0.3, 2),
        ("a", "partial", "m2", '{"y": 2, "x": 1}', 0.8, 0.1, 4),
        ("a", "failed", "bad", "bad", 0.9, 0.9, 99),
        ("a", "success", "invalid", "bad", "nan", "inf", ""),
        ("b", "success", "cox", "", 0.7, 0.2, 1),
    ]
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        writer.writerows(cases)
    return path


@pytest.mark.parametrize("metric,expected", [
    ("antolini_concordance", [0.6, 0.8]),
    ("integrated_brier_score", [0.3, 0.1]),
    ("fit_seconds", [2, 4]),
])
def test_pdf_groups_by_dataset_and_option(source, monkeypatch, metric, expected):
    original = sns.boxplot
    calls = []

    def boxplot(**kwargs):
        calls.append(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(sns, "boxplot", boxplot)
    output = report.create_report(source, metric=metric)
    assert output.read_bytes().startswith(b"%PDF-")
    # The structured option is constant despite different JSON key ordering.
    # Dataset b has only one configuration and is also skipped.
    assert len(calls) == 1
    assert calls[0]["data"]["value"].tolist() == expected
    assert calls[0]["order"] == ["m1", "m2"]
    assert calls[0]["hue"] is None
    assert not plt.get_fignums()


def test_column_selection_and_invalid_input(source):
    output = report.create_report(source, columns=["configuration"])
    assert output.is_file()
    with pytest.raises(ValueError, match="Missing CSV columns"):
        report.create_report(source, columns=["option_missing"])
    with pytest.raises(ValueError, match="Unknown metric"):
        report.create_report(source, metric="missing")
    with pytest.raises(ValueError, match="Output must differ"):
        report.create_report(source, output=source)
    with pytest.raises(ValueError, match="Missing CSV columns"):
        report.create_report(source, hue="missing")


def test_constant_columns_do_not_create_pdf(source):
    with pytest.raises(ValueError, match="No nonconstant columns"):
        report.create_report(source, columns=["option_input"])
    assert not source.with_name("results_antolini_concordance.pdf").exists()


def test_hue_splits_groups_and_excludes_blank_values(source, monkeypatch):
    source.write_text(
        "dataset,status,configuration,option_device,fit_seconds\n"
        "a,success,m1,cpu,1\na,success,m1,cuda,2\n"
        "a,success,m2,cpu,3\na,partial,m2,cuda,4\n"
        "a,success,m3,,5\na,failed,m4,cpu,6\n"
    )
    original = sns.boxplot
    calls = []

    def boxplot(**kwargs):
        calls.append(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(sns, "boxplot", boxplot)
    report.create_report(source, metric="fit_seconds", columns=["configuration"],
                         hue="option_device")
    assert len(calls) == 1
    assert calls[0]["hue"] == "hue"
    assert calls[0]["hue_order"] == ["cpu", "cuda"]
    data = calls[0]["data"]
    assert data.groupby(["category", "hue"])["value"].apply(list).to_dict() == {
        ("m1", "cpu"): [1], ("m1", "cuda"): [2],
        ("m2", "cpu"): [3], ("m2", "cuda"): [4],
    }


def test_no_usable_values_does_not_create_pdf(source):
    source.write_text("dataset,status,configuration,fit_seconds\na,failed,m,1\n")
    with pytest.raises(ValueError, match="No successful or partial runs"):
        report.create_report(source, metric="fit_seconds")
    assert not source.with_name("results_fit_seconds.pdf").exists()


def test_cli_plots_csv_without_running_benchmarks(source, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("Plotting must not initialize devices or run benchmarks")

    monkeypatch.setattr(report, "_device", unexpected)
    monkeypatch.setattr(report, "run_benchmark", unexpected)
    output = source.with_name("report.pdf")
    monkeypatch.setattr("sys.argv", [
        "benchmark_models.py", str(source), "--metric", "fit_seconds",
        "--columns", "configuration", "--hue", "status", "--output", str(output),
    ])
    assert report.main() == 0
    assert output.read_bytes().startswith(b"%PDF-")


def test_cli_reports_invalid_csv(source, monkeypatch):
    monkeypatch.setattr("sys.argv", [
        "benchmark_models.py", str(source), "--columns", "missing",
    ])
    with pytest.raises(SystemExit) as exc:
        report.main()
    assert exc.value.code == 2
