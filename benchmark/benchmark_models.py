"""Benchmark fit time and survival prediction quality.

Run ``python benchmark/benchmark_models.py --help`` for CLI options. The CLI
sweeps zero-argument survival/loss constructors; ``run_benchmark`` accepts
named factories for arbitrary predictor configurations. Pass an existing CSV as a
positional argument to generate a PDF report instead of running fits, for example
``python benchmark/benchmark_models.py results.csv --metric fit_seconds``.
Reporting requires SurvCraft's plotting extra. Output files are replaced.
"""

import argparse
from collections import Counter, defaultdict
from functools import partial
import csv
import inspect
from importlib import metadata as package_metadata
import itertools
import json
import math
from pathlib import Path
import platform
import random
import statistics
import sys
import textwrap
import time
import warnings

import numpy as np
import torch
from joblib import Parallel, delayed
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import integrated_brier_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from survcraft import adapters as ad
from survcraft.datasets import list_datasets, load_dataset
from survcraft.util import get_loss_modules, get_survival_adapters


FIELDS = [
    "dataset", "configuration", "device", "repeat", "split_seed", "training_seed",
    "n_train", "n_test", "n_features", "train_events", "eval_events",
    "batch_size", "learning_rate", "epochs", "early_stopping",
    "fit_seconds", "antolini_concordance", "integrated_brier_score",
    "status", "errors", "warnings",
]


METRICS = {
    "antolini_concordance": "Antolini concordance (higher is better)",
    "integrated_brier_score": "Integrated Brier score (lower is better)",
    "fit_seconds": "Fit time in seconds (lower is better)",
}
ARGUMENTS = {"configuration", "device", "batch_size", "learning_rate", "epochs",
             "early_stopping"}


def _category(value):
    # Canonicalize structured options so JSON key order does not split groups.
    try:
        parsed = json.loads(value)
    except (ValueError, TypeError):
        return value
    if isinstance(parsed, (dict, list)):
        return json.dumps(parsed, sort_keys=True, ensure_ascii=False)
    return value


def create_report(source, output=None, metric="antolini_concordance", columns=None, hue=None):
    """Write per-dataset boxplots and return the output PDF path.

    Parameters
    ----------
    source : str or pathlib.Path
        CSV produced by benchmark_models.py with expanded option columns.
    output : str or pathlib.Path or None, default=None
        Destination; defaults to ``<source_stem>_<metric>.pdf`` beside the CSV.
    metric : str, default="antolini_concordance"
        One of ``antolini_concordance``, ``integrated_brier_score``, ``fit_seconds``.
    columns : sequence of str or None, default=None
        Columns to group by. Defaults to configuration, device, training arguments,
        and every ``option_`` column. Blank values are omitted from each grouping.
        Columns with fewer than two distinct nonblank values per dataset are skipped.
    hue : str or None, default=None
        CSV column used to split boxes by color. Rows with a blank hue are omitted.

    Returns
    -------
    pathlib.Path
        Written PDF. Each column has separate pages per dataset, with at most
        eight categories per page. Boxes pool runs across all other settings.

    Raises
    ------
    ValueError
        If columns or metric are invalid, or no nonconstant plots remain.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    import pandas as pd
    import seaborn as sns

    if metric not in METRICS:
        raise ValueError(f"Unknown metric: {metric}")
    source = Path(source)
    output = Path(output) if output is not None else source.with_name(
        f"{source.stem}_{metric}.pdf")
    if source.resolve() == output.resolve():
        raise ValueError("Output must differ from the input CSV")
    with source.open(newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        selected = (list(columns) if columns is not None else
                    [key for key in fields if key in ARGUMENTS or key.startswith("option_")])
        missing = {"dataset", "status", metric, *selected} - set(fields)
        if hue is not None and hue not in fields:
            missing.add(hue)
        if missing:
            raise ValueError(f"Missing CSV columns: {', '.join(sorted(missing))}")
        if not selected:
            raise ValueError("No model argument or option columns to plot")
        datasets = defaultdict(list)
        total = 0
        for row in reader:
            total += 1
            try:
                value = float(row[metric])
            except (ValueError, TypeError):
                continue
            if row["status"] in ("success", "partial") and math.isfinite(value):
                datasets[row["dataset"]].append((row, value))
    included = sum(map(len, datasets.values()))
    if not included:
        raise ValueError(f"No successful or partial runs with finite {metric}")
    plots = []
    for dataset, cases in sorted(datasets.items()):
        for column in selected:
            records = []
            for row, value in cases:
                category = row.get(column, "").strip()
                color = row.get(hue, "").strip() if hue is not None else ""
                if category and (hue is None or color):
                    records.append({"category": _category(category), "value": value,
                                    "hue": _category(color)})
            if len({record["category"] for record in records}) > 1:
                plots.append((dataset, len(cases), column, pd.DataFrame(records)))
    if not plots:
        raise ValueError("No nonconstant columns remain after filtering blank options and hue")

    output.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output) as pdf:
        for dataset, case_count, column, data in plots:
            counts = data["category"].value_counts()
            categories = sorted(counts.index)
            hue_order = sorted(data["hue"].unique()) if hue is not None else None
            palette = (dict(zip(hue_order, sns.color_palette(n_colors=len(hue_order))))
                       if hue_order is not None else None)
            for start in range(0, len(categories), 8):
                page_categories = categories[start:start + 8]
                labels = [textwrap.fill(f"{key}  (n={counts[key]})", 65,
                                        break_long_words=True)
                          for key in page_categories]
                # Reserve space per row for its labels and split boxes, with
                # fixed margins rather than a large minimum page height.
                height = max(2.2, 1.35 + sum(
                    max((label.count("\n") + 1) * 0.22,
                        len(hue_order or [None]) * 0.18)
                    for label in labels))
                fig, ax = plt.subplots(figsize=(14, height))
                try:
                    sns.boxplot(data=data[data["category"].isin(page_categories)],
                                x="value", y="category", order=page_categories,
                                hue="hue" if hue is not None else None,
                                hue_order=hue_order, palette=palette, dodge=True,
                                orient="h", showmeans=True, ax=ax)
                    ax.set_yticks(range(len(page_categories)), labels=labels)
                    ax.set_ylabel(column)
                    if hue is not None:
                        legend = ax.legend(title=hue, loc="upper left",
                                           bbox_to_anchor=(1.02, 1))
                        for label in legend.get_texts():
                            label.set_text(textwrap.fill(label.get_text(), 40))
                    ax.set_xlabel(METRICS[metric])
                    ax.set_title(f"{dataset} — {column}\n"
                                 f"Categories {start + 1}–{start + len(page_categories)}"
                                 f" of {len(categories)}")
                    ax.grid(axis="x", alpha=0.3)
                    fig.text(0.02, 0.06 / height,
                             f"{source.name} | Usable rows: {included}/{total}; "
                             f"dataset: {case_count}; grouped: {len(data)}\n"
                             "Blank options/hue omitted. Boxes pool other settings and repeats; "
                             "group differences do not isolate an option's effect.", fontsize=8)
                    fig.tight_layout(rect=(0, 0.38 / height, 1, 1), pad=0.6)
                    pdf.savefig(fig)
                finally:
                    plt.close(fig)
    return output


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, torch.nn.Module):
        return {"class": type(value).__name__, "repr": repr(value),
                "options": {key: item for key, item in vars(value).items()
                            if not key.startswith("_") and key != "training"},
                "state": value.state_dict()}
    if hasattr(value, "get_params"):
        return {"class": type(value).__name__, "options": value.get_params(deep=False)}
    return repr(value)


def _error(stage, exc):
    return {"stage": stage, "type": type(exc).__name__, "message": str(exc)}


def _seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _device(device):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    resolved = torch.device(device)
    if resolved.type not in ("cpu", "cuda"):
        raise ValueError("Only CPU and CUDA devices are supported")
    if resolved.type == "cuda":
        # Allocate before any measurements, including on explicitly indexed GPUs.
        torch.ones(1, device=resolved).add_(1)
        torch.cuda.synchronize(resolved)
    return str(resolved)


def _synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


class _CheckedSurvival:
    def __init__(self, model):
        self.model = model

    def predict_survival(self, X, times):
        if isinstance(self.model, CoxPHSurvivalAnalysis):
            values = np.asarray([fn(times) for fn in self.model.predict_survival_function(X)])
        else:
            values = np.asarray(self.model.predict_survival(X, times))
        if values.shape != (len(X), len(times)):
            raise ValueError("Invalid survival prediction shape")
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite survival predictions")
        if not ((values >= -1e-6) & (values <= 1 + 1e-6)).all():
            raise ValueError("Survival predictions outside [0, 1]")
        if not (np.diff(values[:, np.argsort(times)], axis=1) <= 1e-5).all():
            raise ValueError("Increasing survival curves")
        return values


def _brier_grid(data):
    targets = data["y_eval"]
    event_times = targets["time"][targets["event"]]
    if not len(event_times):
        return np.array([])
    times = np.unique(np.quantile(event_times, np.arange(4)[1:] / 4))
    lower = max(targets["time"].min(), data["y_train"]["time"].min())
    upper = min(targets["time"].max(), data["y_train"]["time"].max())
    return times[(times >= lower) & (times < upper)]


def _cap_training_data(data, limit, seed):
    if limit is None or len(data["y_train"]) <= limit:
        return data
    indices = np.sort(np.random.default_rng(seed).choice(len(data["y_train"]), limit, replace=False))
    data = {**data, "metadata": dict(data["metadata"])}
    original_size = len(data["y_train"])
    for key in ("X_train", "y_train", "train_indices"):
        data[key] = data[key][indices]
    # Keep evaluation within the follow-up of the actual training subset.
    data["y_eval"] = data["y_test"].copy()
    horizon = float(data["y_train"]["time"].max())
    beyond = data["y_eval"]["time"] > horizon
    data["y_eval"]["time"][beyond] = horizon
    data["y_eval"]["event"][beyond] = False
    data["metadata"].update(
        n_train_before_cap=original_size, n_train=limit,
        train_events=int(data["y_train"]["event"].sum()),
        eval_events=int(data["y_eval"]["event"].sum()),
        evaluation_horizon=horizon, administratively_censored=int(beyond.sum()),
    )
    return data


def _score(model, data, scorer, row):
    checked = _CheckedSurvival(model)
    for name in ("antolini_concordance", "integrated_brier_score"):
        try:
            if name == "antolini_concordance":
                value = scorer(checked, data["X_test"], data["y_eval"])
            else:
                times = data["times"]
                if len(times) < 2:
                    raise ValueError("Brier score requires at least two valid distinct grid times")
                probabilities = checked.predict_survival(data["X_test"], times)
                value = integrated_brier_score(
                    data["y_train"], data["y_eval"], probabilities, times,
                )
            if not np.isfinite(value):
                raise ValueError(f"Nonfinite {name}")
            row[name] = float(value)
        except Exception as exc:
            row["errors"].append(_error(name, exc))


def _fit_case(factory, data, base, seed, scorer, warmup_epochs=0):
    row = {**base, "errors": [], "warnings": []}
    stage = "construction"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            _seed(seed)
            model = factory()
            if not isinstance(model, (ad.SurvivalPredictor, CoxPHSurvivalAnalysis)):
                raise TypeError("A model factory must return a SurvivalPredictor")
            if isinstance(model, ad.SurvivalPredictor):
                model.device = _device(model.device)
                model.loss = model.loss.to(model.device)
                options = model.get_params(deep=False)
                row.update(device=model.device, options=options,
                           **{key: options[key] for key in (
                               "batch_size", "learning_rate", "epochs", "early_stopping")})
                if warmup_epochs:
                    # Warm up a separate model; failures do not suppress measured fits.
                    stage = "warmup"
                    warmup = None
                    try:
                        warmup = factory()
                        warmup.device = model.device
                        warmup.loss = warmup.loss.to(model.device)
                        warmup.epochs = warmup_epochs
                        warmup.fit(data["X_train"], data["y_train"])
                        _synchronize(model.device)
                    except Exception as exc:
                        row["warnings"].append(f"Warm-up failed: {type(exc).__name__}: {exc}")
                    # Recreate after resetting RNGs, including factory-side randomness.
                    del warmup
                    _seed(seed)
                    stage = "construction"
                    model = factory()
                    model.device = row["device"]
                    model.loss = model.loss.to(model.device)
                device = model.device
            else:
                device = "cpu"
                row.update(device=device, options=model.get_params(deep=False))
            stage = "fit"
            _synchronize(device)
            started = time.perf_counter()
            model.fit(data["X_train"], data["y_train"])
            _synchronize(device)
            row["fit_seconds"] = time.perf_counter() - started
            stage = "parameters"
            if isinstance(model, ad.SurvivalPredictor):
                if not all(torch.isfinite(p).all().item() for p in model.model_.parameters()):
                    raise ValueError("Nonfinite fitted parameters")
            elif not np.isfinite(model.coef_).all():
                raise ValueError("Nonfinite fitted Cox coefficients")
            _score(model, data, scorer, row)
            row["status"] = "partial" if row["errors"] else "success"
        except Exception as exc:
            row["errors"].append(_error(stage, exc))
            row["status"] = "unsupported" if isinstance(exc, NotImplementedError) else "failed"
        row["warnings"].extend(f"{w.category.__name__}: {w.message}" for w in caught)
    return row


def _fit_repeats(factory, data, base, training_seed, repeats, warmup_epochs):
    from survmetr import AntoliniCIndexVecScorer

    scorer = AntoliniCIndexVecScorer()
    for repeat in range(repeats):
        seed = training_seed + repeat
        row = _fit_case(factory, data, {**base, "repeat": repeat, "training_seed": seed},
                        seed, scorer, warmup_epochs if repeat == 0 else 0)
        # Return plain configuration data rather than live modules/GPU tensors.
        if "options" in row:
            row["options"] = json.loads(json.dumps(row["options"], default=_json_default))
        yield row


def _fit_worker(*args):
    return list(_fit_repeats(*args))


def run_benchmark(models, datasets=None, *, repeats=3, warmup_epochs=1,
                  split_seed=0, training_seed=123, test_size=0.2,
                  min_train_samples=0, max_train_samples=None, filter_small_datasets=False,
                  workers=1, output="benchmark/results/models.csv", run_metadata=None):
    """Fit named predictor factories and save timing and predictive metrics.

    Parameters
    ----------
    models : mapping of str to callable
        Names mapped to zero-argument factories returning fresh SurvivalPredictor
        instances. Factories control all model options, including device, epochs,
        input maps, nested survival adapters, and composed losses. Devices must
        be CPU, CUDA, or 'auto'. Models are never reused between fits.
    datasets : sequence of str or None, optional
        Public dataset names; None or ['all'] selects the full available catalog.
    repeats : int, default=3
        Fresh measured fits per model on a single fixed dataset split.
    warmup_epochs : int, default=1
        Epochs in a separate unmeasured fit per model/dataset; zero disables it.
    split_seed : int, default=0
        Seed passed to the public dataset loader.
    training_seed : int, default=123
        Python, NumPy, and PyTorch seed; repeat index is added for measured fits.
    test_size : float, default=0.2
        Held-out fraction passed to the dataset loader.
    min_train_samples : int, default=0
        Exclude prepared datasets with fewer training rows; zero disables this.
    max_train_samples : int or None, default=None
        Cap training rows by sampling without replacement using split_seed.
        The subset is shared by all models and repeats, including Cox. Test rows
        are unchanged; evaluation censoring uses the subset's follow-up horizon.
        Sampling follows loader preprocessing, which uses the full training split.
    filter_small_datasets : bool, default=False
        Also exclude datasets with fewer than two training rows per feature.
    workers : int, default=1
        Concurrent model configurations in separate joblib worker processes.
        One runs sequentially in the main process. Repeats of a configuration
        stay together; datasets and the Cox reference are prepared in the parent.
    output : str or pathlib.Path, optional
        CSV path. This file and its .metadata.json companion are replaced.
        Model options use one ``option_<key>`` column per top-level key;
        nested dictionaries and lists are JSON-encoded. Missing options are blank.
    run_metadata : dict or None, optional
        Additional caller metadata, such as CLI arguments.

    Returns
    -------
    list of dict
        Case results, including one CPU ridge Cox reference per prepared dataset,
        failures, and dataset exclusions. Numeric metrics are absent if unavailable.

    Raises
    ------
    ImportError
        If survmetr is unavailable.
    ValueError
        If models is empty or shared run settings are invalid.

    Notes
    -----
    Both metrics use administratively censored y_eval targets, adjusted to the
    sampled training horizon when capped. Case errors are recorded; interrupts
    propagate. With workers > 1, the parent writes results as configurations
    finish; row order may vary. Factories must be serializable by joblib and
    multiprocessing callers should use an ``if __name__ == "__main__"`` guard.
    Keep predictor data-loader workers at zero to avoid nested multiprocessing.
    Concurrent fits share CPU/GPU resources: use workers=1 for isolated timing.
    Repeats measure training variability, not split uncertainty.

    Examples
    --------
    >>> models = {"custom": lambda: ad.SurvivalPredictor(
    ...     survival=ad.WeibullSurvivalAdapter(), epochs=2, device="cpu")}
    >>> rows = run_benchmark(models, ["sksurv:whas500"], repeats=1,
    ...                      warmup_epochs=0, output="/tmp/custom-models.csv")
    """
    from survmetr import AntoliniCIndexVecScorer

    if not models or repeats < 1 or warmup_epochs < 0 or min_train_samples < 0:
        raise ValueError("Provide models, positive repeats, and nonnegative warm-up/minimum sizes")
    if max_train_samples is not None and max_train_samples < 1:
        raise ValueError("max_train_samples must be positive or None")
    if workers < 1:
        raise ValueError("workers must be positive")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between zero and one")
    names = list_datasets() if datasets is None or list(datasets) == ["all"] else list(datasets)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = output.with_suffix(".metadata.json")
    metadata = {
        "settings": dict(repeats=repeats, warmup_epochs=warmup_epochs, split_seed=split_seed,
                         training_seed=training_seed, test_size=test_size, workers=workers,
                         min_train_samples=min_train_samples, max_train_samples=max_train_samples,
                         filter_small_datasets=filter_small_datasets),
        "caller": run_metadata, "python": platform.python_version(),
        "torch": torch.__version__, "numpy": np.__version__, "package": ad.__file__,
        "platform": platform.platform(), "threads": torch.get_num_threads(),
        "cuda_build": torch.version.cuda,
        "gpus": ([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
                 if torch.cuda.is_available() else []),
        "configurations": {}, "datasets": {},
    }
    rows = []
    scorer = AntoliniCIndexVecScorer()
    metadata["versions"] = {}
    for name in ("torch", "numpy", "scikit-learn", "scikit-survival", "survmetr"):
        try:
            metadata["versions"][name] = package_metadata.version(name)
        except package_metadata.PackageNotFoundError:
            metadata["versions"][name] = "not installed as a distribution"

    def save_metadata():
        metadata_path.write_text(json.dumps(metadata, default=_json_default, indent=2) + "\n")

    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS.copy())
        writer.writeheader()
        stream.flush()

        def csv_row(row):
            values = {key: value for key, value in row.items() if key != "options"}
            values.update({f"option_{key}": value
                           for key, value in row.get("options", {}).items()})
            return {key: json.dumps(value, default=_json_default)
                    if isinstance(value, (dict, list)) else value
                    for key, value in values.items()}

        def record(row):
            values = csv_row(row)
            new_fields = [key for key in values if key not in writer.fieldnames]
            if new_fields:
                writer.fieldnames.extend(new_fields)
                # Models can expose different options. Expand earlier rows too.
                stream.seek(0)
                stream.truncate()
                writer.writeheader()
                writer.writerows(csv_row(previous) for previous in rows)
            writer.writerow(values)
            stream.flush()
            rows.append(row)
            if "options" in row:
                metadata["configurations"][row["configuration"]] = row["options"]
            print(f"{row['dataset']} | {row.get('configuration', 'preparation')} | "
                  f"repeat={row.get('repeat', '-')} {row['status']} "
                  f"fit={row.get('fit_seconds', float('nan')):.4f}s "
                  f"C={row.get('antolini_concordance', float('nan')):.4f} "
                  f"IBS={row.get('integrated_brier_score', float('nan')):.4f}", flush=True)

        save_metadata()
        try:
            for name in names:
                print(f"Loading {name}", flush=True)
                base = {"dataset": name, "split_seed": split_seed}
                started = time.perf_counter()
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    try:
                        data = load_dataset(name, seed=split_seed, test_size=test_size)
                        data = _cap_training_data(data, max_train_samples, split_seed)
                    except Exception as exc:
                        record({**base, "status": "failed", "errors": [_error("preparation", exc)],
                                "warnings": [str(w.message) for w in caught]})
                        continue
                for w in caught:
                    print(f"{name}: {w.category.__name__}: {w.message}", flush=True)
                data["times"] = _brier_grid(data)
                metadata["datasets"][name] = {
                    **data["metadata"], "load_seconds": time.perf_counter() - started,
                    "train_indices": data["train_indices"], "test_indices": data["test_indices"],
                    "brier_times": data["times"], "warnings": [str(w.message) for w in caught],
                }
                base.update({key: data["metadata"][key] for key in (
                    "n_train", "n_test", "n_features", "train_events", "eval_events")})
                save_metadata()
                if base["n_train"] < min_train_samples or (
                    filter_small_datasets and base["n_train"] < 2 * base["n_features"]
                ):
                    record({**base, "status": "excluded", "errors": [{"stage": "preparation",
                            "type": "DatasetFilter", "message": "Training sample/feature limits"}],
                            "warnings": []})
                    continue
                record(_fit_case(partial(CoxPHSurvivalAnalysis, alpha=1.0), data,
                                 {**base, "configuration": "cox-ridge"}, training_seed, scorer))
                jobs = [(factory, data, {**base, "configuration": label},
                         training_seed, repeats, warmup_epochs)
                        for label, factory in models.items()]
                if workers == 1:
                    results = (_fit_repeats(*job) for job in jobs)
                else:
                    results = Parallel(n_jobs=workers, backend="loky", return_as="generator_unordered")(
                        delayed(_fit_worker)(*job) for job in jobs)
                for configuration_rows in results:
                    times = []
                    for row in configuration_rows:
                        record(row)
                        if row["status"] in ("success", "partial"):
                            times.append(row["fit_seconds"])
                    if times:
                        label = row["configuration"]
                        print(f"{name} | {label}: median {statistics.median(times):.4f}s "
                              f"range {min(times):.4f}–{max(times):.4f}s", flush=True)
                save_metadata()
        finally:
            metadata["counts"] = dict(Counter(row["status"] for row in rows))
            save_metadata()
    print(f"Final counts: {metadata['counts']}", flush=True)
    return rows


def _select(names, classes, kind):
    available = {cls.__name__: cls for cls in classes}
    selected = list(available) if names == ["all"] else names
    unknown = set(selected) - available.keys()
    if unknown:
        raise ValueError(f"Unknown {kind}: {', '.join(sorted(unknown))}; choices: {', '.join(available)}")
    result = []
    for name in dict.fromkeys(selected):
        cls = available[name]
        try:
            inspect.signature(cls).bind()
        except TypeError:
            message = f"{name} requires constructor arguments; use run_benchmark with a factory"
            if names != ["all"]:
                raise ValueError(message)
            warnings.warn(message, stacklevel=2)
        else:
            result.append(cls)
    return result


def _make_predictor(survival, loss, batch_size, epochs, device, early_stopping=False):
    return ad.SurvivalPredictor(
        input=ad.FeedForwardNetAdapter(hidden_sizes=[32, 32]),
        survival=survival(), loss=loss(), batch_size=batch_size,
        epochs=epochs, device=device, early_stopping=early_stopping,
    )


def main():
    """Run the benchmark grid or plot an existing CSV; return an exit status."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="?", type=Path,
                        help="Plot this existing benchmark CSV instead of running fits")
    parser.add_argument("--metric", choices=METRICS, default="antolini_concordance",
                        help="Report score or fit time (default: antolini_concordance)")
    parser.add_argument("--columns", nargs="+",
                        help="Report only these CSV columns (default: nonconstant arguments/options)")
    parser.add_argument("--hue", help="Report CSV column used to split boxes by color")
    parser.add_argument("--datasets", nargs="+", default=["all"],
                        help="Dataset names such as sksurv:whas500, or all (default: all)")
    parser.add_argument("--survivals", nargs="+", default=["all"],
                        help="Survival adapter class names to sweep, or all (default: all)")
    parser.add_argument("--losses", nargs="+", default=["all"],
                        help="Loss class names to sweep, or all constructible without arguments (default: all)")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[256],
                        help="Positive training batch sizes to sweep (default: 256)")
    parser.add_argument("--epochs", nargs="+", type=int, default=[100],
                        help="Positive maximum epoch counts to sweep (default: 100)")
    parser.add_argument("--early-stopping", action="store_true",
                        help="Enable early stopping using a training validation subset (default: off)")
    parser.add_argument("--devices", nargs="+", default=["auto"],
                        help="Devices to sweep: cpu, cuda, cuda:N, or auto for CUDA with CPU fallback (default: auto)")
    parser.add_argument("--repeats", type=int, default=3,
                        help="Positive number of fresh measured fits per model and dataset (default: 3)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel model worker processes; 1 runs sequentially (default: 1)")
    parser.add_argument("--warmup-epochs", type=int, default=1,
                        help="Epochs in a separate unmeasured warm-up fit; 0 disables it (default: 1)")
    parser.add_argument("--split-seed", type=int, default=0,
                        help="Random seed for dataset splitting, preprocessing, and training downsampling (default: 0)")
    parser.add_argument("--training-seed", type=int, default=123,
                        help="Base Python, NumPy, and PyTorch training seed; repeat index is added (default: 123)")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Held-out fraction, strictly between 0 and 1 (default: 0.2)")
    parser.add_argument("--min-train-samples", type=int, default=0,
                        help="Exclude datasets below this training-row count after downsampling; 0 disables (default: 0)")
    parser.add_argument("--max-train-samples", type=int,
                        help="Positive cap on training rows, sampled after preprocessing (default: no cap)")
    parser.add_argument("--filter-small-datasets", action="store_true",
                        help="Exclude datasets with fewer than two training samples per feature after downsampling (default: off)")
    parser.add_argument("--output", type=Path,
                        help="Output path (default: benchmark/results/models.csv for fits, "
                             "<csv_stem>_<metric>.pdf for reports); existing files are replaced")
    args = parser.parse_args()
    if args.csv is not None:
        try:
            output = create_report(args.csv, args.output, args.metric, args.columns, args.hue)
        except (ValueError, OSError, ImportError) as exc:
            parser.error(str(exc))
        print(f"Saved {output}")
        return 0
    if args.output is None:
        args.output = Path("benchmark/results/models.csv")
    try:
        if min(*args.batch_sizes, *args.epochs, args.repeats) < 1:
            raise ValueError("Batch sizes, epochs, and repeats must be positive")
        devices = list(dict.fromkeys(_device(device) for device in args.devices))
        survivals = _select(args.survivals, get_survival_adapters(), "survivals")
        losses = _select(args.losses, get_loss_modules(), "losses")
        models = {}
        for survival, loss, batch, epochs, device in itertools.product(
            survivals, losses, args.batch_sizes, args.epochs, devices
        ):
            label = f"{survival.__name__}-{loss.__name__}-bs{batch}-ep{epochs}-{device}"
            if args.early_stopping:
                label += "-early-stopping"
            models[label] = partial(_make_predictor, survival, loss, batch, epochs, device,
                                    args.early_stopping)
        rows = run_benchmark(
            models, args.datasets, repeats=args.repeats, warmup_epochs=args.warmup_epochs,
            split_seed=args.split_seed, training_seed=args.training_seed, test_size=args.test_size,
            min_train_samples=args.min_train_samples, max_train_samples=args.max_train_samples,
            filter_small_datasets=args.filter_small_datasets, workers=args.workers,
            output=args.output, run_metadata=vars(args),
        )
    except (ValueError, ImportError, RuntimeError) as exc:
        parser.error(str(exc))
    return 0 if any(row["status"] in ("success", "partial") for row in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
