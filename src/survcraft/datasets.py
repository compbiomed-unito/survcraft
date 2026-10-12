"""Discover survival datasets and prepare normalized train/test splits."""

import inspect
import warnings

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer

__all__ = ["list_datasets", "load_dataset"]


# These are the same cohorts, even when SurvSet drops incomplete observations.
SURVSET_OVERLAPS = {
    "actg": "aids",
    "GBSG2": "gbsg2",
    "flchain": "flchain",
    "veteran": "veterans_lung_cancer",
    "whas500": "whas500",
}

# Future outcomes, not measurements available when follow-up starts.
SURVSET_POST_OUTCOME = {
    "mgus": ("num_pctime", "fac_pcdx"),
    "support2": ("fac_sfdm2",),
}


def list_datasets():
    """List available baseline, right-censored survival datasets.

    Returns
    -------
    list of str
        Qualified names such as ``"sksurv:gbsg2"`` and ``"survset:mgus"``.
        Names are ordered by source, then alphabetically within each source.
        Only names accepted by :func:`load_dataset` are included.

    Warns
    -----
    UserWarning
        If optional SurvSet is missing, its datasets are skipped. Install
        ``survcraft[datasets]`` to include them.

    Notes
    -----
    Includes scikit-survival's zero-argument loaders and SurvSet's static
    datasets. Duplicate cohorts in SurvSet are excluded in favor of
    scikit-survival. Time-varying/start-stop datasets are excluded. Listing
    inspects the catalog without loading or preprocessing individual datasets;
    validation or splitting can still fail for a listed dataset.

    Examples
    --------
    >>> "sksurv:gbsg2" in list_datasets()
    True
    """
    return [entry["dataset"] for entry in _discover_datasets()
            if entry["status"] == "included"]


def load_dataset(name, seed=0, test_size=0.2):
    """Load a named survival dataset with normalized train/test features.

    Parameters
    ----------
    name : str
        Qualified name returned by :func:`list_datasets`, or an unambiguous
        short name such as ``"gbsg2"``. Names are case-sensitive.
    seed : int, default=0
        Random seed for the event-stratified split and quantile transform.
    test_size : float, default=0.2
        Held-out fraction, strictly between zero and one.

    Returns
    -------
    dict
        ``X_train`` and ``X_test`` are dense float32 arrays of shape
        ``(n_train, n_features)`` and ``(n_test, n_features)``.
        ``y_train`` and ``y_test`` are structured arrays with boolean
        ``event`` (True for observed events) and float64 ``time`` fields.
        ``y_eval`` copies test targets, censoring observations beyond the
        maximum training time at that horizon. ``train_indices`` and
        ``test_indices`` identify rows in the source dataset. ``metadata``
        contains the qualified dataset name, reference, dropped columns,
        original event codes and zero-time count, split sizes, event counts,
        evaluation horizon, censoring count, and preprocessing duration.

    Raises
    ------
    ValueError
        If the name is unknown, ambiguous, or excluded; test_size is outside
        (0, 1); outcomes are missing, nonfinite, negative, or all zero; the
        dataset has start-stop outcomes or repeated subject identifiers; or
        stratified splitting/preprocessing cannot be performed.
    Warns
    -----
    UserWarning
        If optional SurvSet is missing, its datasets are skipped. Encoding
        can also warn about previously unseen categories in the test set.

    Notes
    -----
    Preprocessing is fitted on training rows only: numeric median imputation
    and a quantile transform to a normal distribution; categorical mode
    imputation and dense one-hot encoding with the first level dropped and
    unseen levels ignored; then removal of constant transformed features.
    Numeric infinities are treated as missing. Identifiers and known
    post-outcome features are excluded (FLCHAIN chapter, MGUS num_pctime and
    fac_pcdx, SUPPORT2 fac_sfdm2).

    Positive event codes are combined into all-cause events. Targets are
    never imputed. If any observed time is zero, all times are shifted by the
    smallest positive time, preserving the benchmark's convention. Otherwise
    original time units and values are retained. Only y_eval receives the
    administrative censoring adjustment; y_test retains the loaded targets.

    Examples
    --------
    >>> data = load_dataset("sksurv:gbsg2", seed=0)
    >>> data["X_train"].dtype.name
    'float32'
    >>> data["y_train"].dtype.names
    ('event', 'time')
    """
    if not 0 < test_size < 1:
        raise ValueError("test_size must be strictly between zero and one")
    entries = [entry for entry in _discover_datasets() if entry["status"] == "included"]
    matches = [entry for entry in entries
               if name in (entry["dataset"], entry["dataset"].split(":", 1)[1])]
    if len(matches) != 1:
        raise ValueError(f"dataset {name!r} is unavailable or ambiguous; inspect list_datasets()")
    entry = matches[0]
    X, y, categorical, metadata = _load_raw_dataset(entry)
    arrays, split_metadata = _prepare_split(X, y, categorical, seed, test_size)
    arrays["metadata"] = {"dataset": entry["dataset"], **metadata, **split_metadata}
    return arrays


def _discover_datasets():
    """Return a catalog including duplicate and unsupported dataset entries.

    Returns
    -------
    list of dict
        Qualified names, loader names, sources, and inclusion/skip reasons.
        Time-varying SurvSet datasets are listed but cannot be fitted by the
        current right-censored estimators.
    """
    from sksurv import datasets

    catalog = []
    for name, loader in inspect.getmembers(datasets, inspect.isfunction):
        if not name.startswith("load_"):
            continue
        required = [p for p in inspect.signature(loader).parameters.values()
                    if p.default is p.empty and p.kind in (p.POSITIONAL_ONLY,
                                                          p.POSITIONAL_OR_KEYWORD)]
        if not required:
            # this currently skips only load_arff_files_standardized()
            catalog.append({"dataset": "sksurv:" + name[5:], "source": "sksurv",
                            "loader": name, "status": "included", "reason": ""})
    try:
        from SurvSet.data import SurvLoader
    except ModuleNotFoundError as exc:
        if exc.name not in ("SurvSet", "SurvSet.data"):
            raise
        warnings.warn(
            'SurvSet is not installed; skipping its datasets. '
            'Install them with pip install "survcraft[datasets]".',
            UserWarning, stacklevel=2,
        )
        return catalog
    for row in SurvLoader().df_ds.sort_values("ds").to_dict("records"):
        name = row["ds"]
        reason = ""
        if row["is_td"]:
            reason = "time-varying features / start-stop outcomes are unsupported"
        elif name in SURVSET_OVERLAPS:
            reason = "same cohort as sksurv:" + SURVSET_OVERLAPS[name]
        catalog.append({"dataset": "survset:" + name, "source": "survset",
                        "loader": name, "status": "skipped" if reason else "included",
                        "reason": reason, **row})
    return catalog


def _load_raw_dataset(entry):
    """Load baseline features and canonical all-cause survival outcomes.

    Parameters
    ----------
    entry : dict
        Included entry returned by :func:`_discover_datasets`.

    Returns
    -------
    X : pandas.DataFrame
        Features with identifiers and known post-outcome fields removed.
    y : numpy.ndarray
        Two-field structured target: boolean event, float64 observed time.
        Event codes greater than zero denote any observed event.
    categorical : list of str
        Categorical feature names, using SurvSet's explicit prefixes.
    metadata : dict
        Source reference, dropped columns, event codes, and zero-time count.

    Raises
    ------
    ValueError
        If targets are missing, nonfinite, or negative, or a SurvSet record
        has start-stop outcomes or repeated subject identifiers.
    """
    dropped = []
    reference = ""
    if entry["source"] == "sksurv":
        from sksurv import datasets
        X, original_y = getattr(datasets, entry["loader"])()
        event, time = (original_y[field] for field in original_y.dtype.names)
        if entry["loader"] == "load_flchain":
            # chapter describes the cause of death and is absent for survivors.
            dropped = ["chapter"]
            X = X.drop(columns=dropped)
        # Integer, unsigned, float, and complex columns are numeric;
        # booleans and categorical/string/object columns are categorical.
        categorical = [c for c in X if X[c].dtype.kind not in "iufc"]
    else:
        from SurvSet.data import SurvLoader
        data = SurvLoader().load_dataset(entry["loader"])
        frame, reference = data["df"], data["ref"]
        if "time2" in frame or frame["pid"].duplicated().any():
            raise ValueError("start-stop outcomes or repeated subjects are unsupported")
        event, time = frame["event"].to_numpy(), frame["time"].to_numpy()
        excluded = SURVSET_POST_OUTCOME.get(entry["loader"], ())
        features = [c for c in frame if c.startswith(("num_", "fac_")) and c not in excluded]
        X = frame[features].copy()
        dropped = [c for c in frame if c not in features and c not in ("event", "time")]
        categorical = [c for c in X if c.startswith("fac_")]
    event, time = np.asarray(event, dtype=float), np.asarray(time, dtype=float)
    if not (np.isfinite(event).all() and np.isfinite(time).all()) or (event < 0).any() or (time < 0).any():
        raise ValueError("outcomes must be finite and nonnegative; targets are not imputed")
    y = np.empty(len(time), dtype=[("event", "?"), ("time", "f8")])
    zero_times = int((time == 0).sum())
    min_time = time.min()
    if min_time == 0:
        # this can be a problem for many models
        positive_times = time[time > 0]
        if not len(positive_times):
            raise ValueError("outcomes must contain at least one strictly positive time")
        time = time + positive_times.min()
    y["event"], y["time"] = event > 0, time
    # Uniform string categories avoid mixed numeric/string imputation errors.
    for column in categorical:
        values = X[column]
        X[column] = values.astype(object).map(str).where(values.notna(), np.nan)
    numeric = [c for c in X if c not in categorical]
    X[numeric] = X[numeric].replace([np.inf, -np.inf], np.nan)
    return X, y, categorical, {"reference": reference, "dropped_columns": dropped,
                               "event_codes": np.unique(event).tolist(),
                               "zero_times": zero_times}


def _prepare_split(X, y, categorical, seed=0, test_size=0.2):
    """Split once, then impute, normalize, and encode using training data.

    Parameters
    ----------
    X : pandas.DataFrame
        Baseline features returned by :func:`_load_raw_dataset`.
    y : numpy.ndarray
        Canonical structured survival target.
    categorical : list of str
        Categorical columns.
    seed : int, default=0
        Seed for event-stratified splitting and quantile transformation.
    test_size : float, default=0.2
        Held-out fraction between zero and one.

    Returns
    -------
    arrays : dict of numpy.ndarray
        Dense float32 train/test features, original targets, evaluation
        targets, and original row indices. Test outcomes beyond the largest
        training time are administratively censored at that time for every
        model, keeping Cox predictions inside their supported domain.
    metadata : dict
        Sizes, event counts, preprocessing time, and follow-up restriction.
    """
    from time import perf_counter

    started = perf_counter()
    train, test = train_test_split(np.arange(len(y)), test_size=test_size,
                                   random_state=seed, stratify=y["event"])
    numeric = [c for c in X if c not in categorical]
    transformer = make_pipeline(
        ColumnTransformer([
            ("numeric", make_pipeline(
                SimpleImputer(strategy="median", keep_empty_features=True),
                QuantileTransformer(n_quantiles=min(1000, len(train)),
                                    output_distribution="normal", subsample=None,
                                    random_state=seed)), numeric),
            ("categorical", make_pipeline(
                SimpleImputer(strategy="most_frequent", keep_empty_features=True),
                OneHotEncoder(handle_unknown="ignore", drop="first",
                              sparse_output=False, dtype=np.float32)), categorical),
        ], sparse_threshold=0),
        VarianceThreshold(),
    )
    X_train = transformer.fit_transform(X.iloc[train]).astype(np.float32)
    X_test = transformer.transform(X.iloc[test]).astype(np.float32)
    if not (np.isfinite(X_train).all() and np.isfinite(X_test).all()):
        raise ValueError("preprocessing produced nonfinite features")
    y_eval = y[test].copy()
    horizon = float(y["time"][train].max())
    beyond_horizon = y_eval["time"] > horizon
    y_eval["event"][beyond_horizon] = False
    y_eval["time"][beyond_horizon] = horizon
    arrays = {"X_train": X_train, "X_test": X_test, "y_train": y[train],
              "y_test": y[test], "y_eval": y_eval, "train_indices": train,
              "test_indices": test}
    metadata = {"n_train": len(train), "n_test": len(test),
                "n_features": X_train.shape[1], "train_events": int(y["event"][train].sum()),
                "test_events": int(y["event"][test].sum()),
                "eval_events": int(y_eval["event"].sum()), "evaluation_horizon": horizon,
                "administratively_censored": int(beyond_horizon.sum()),
                "preprocess_seconds": perf_counter() - started}
    return arrays, metadata
