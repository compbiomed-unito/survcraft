r"""Discovery, data preparation, and plotting support utilities.
"""

import numpy as np
import torch
import inspect
from . import adapters, loss_modules, survival_modules#, input_modules

def get_subclasses_in_module(module, base_class):
    r"""Find locally defined subclasses of a supplied base class.

    Parameters
    ----------
    module : module
        Python module to inspect.
    base_class : type
        Parent class to match; the parent itself is excluded.

    Returns
    -------
    iterator of classes
        Locally defined subclasses in inspect.getmembers name order.

    Notes
    -----
    Abstract classes and classes imported from another module are excluded. 
    Inherited stubs do not make a class abstract unless the Python ABC 
    machinery marks it so.
    """

    for _, obj in inspect.getmembers(module, inspect.isclass):
        if obj.__module__ == module.__name__ and issubclass(obj, base_class) and obj is not base_class:
            if not inspect.isabstract(obj):
                yield obj

def get_input_adapters(include_simulation=False):
    return [
        ia for ia in get_subclasses_in_module(adapters, adapters.BaseInputAdapter)
        if include_simulation or not ia.simulation_only
    ]

def get_survival_adapters(include_simulation=False):
    r"""List concrete survival adapter classes defined in adapters.

    Returns
    -------
    list of type
        Locally defined subclasses of BaseSurvivalAdapter.

    See Also
    --------
    get_subclasses_in_module : Discovery rules.
    """
    return [
        sa for sa in get_subclasses_in_module(adapters, adapters.BaseSurvivalAdapter)
        if include_simulation or not sa.simulation_only
    ]

def get_loss_modules():
    r"""List concrete survival loss classes defined in loss_modules.

    Returns
    -------
    list of type
        Locally defined subclasses of BaseSurvivalLoss.

    See Also
    --------
    get_subclasses_in_module : Discovery rules.
    """
    return get_subclasses_in_module(loss_modules, loss_modules.BaseSurvivalLoss)

def get_quantiles(x, n, drop_extremes=True):
    r"""Compute evenly spaced quantiles using the input backend.

    Parameters
    ----------
    x : numpy.ndarray or torch.Tensor
        Numeric values; backend quantile defaults flatten the input.
    n : int
        Number of quantile levels from zero through one, including extremes.
    drop_extremes : bool, default=True
        Remove the first and last quantiles after computation.

    Returns
    -------
    numpy.ndarray or torch.Tensor
        Quantiles using the same backend as x; normally length n-2 if
        extremes are dropped, otherwise n.

    Notes
    -----
    The PyTorch quantile grid is created with default device and dtype;
    nondefault tensor devices or float64 inputs may need caller handling.
    """
    mod = torch if isinstance(x, torch.Tensor) else np

    qt = mod.quantile(x, mod.linspace(0.0, 1.0, n))
    if drop_extremes:
        qt = qt[1:-1]
    return qt


def detect_max_survival_time(model, X, tol=1e-2, q=0.5):
    r"""Estimate a plotting horizon where a survival quantile reaches tolerance.

    Parameters
    ----------
    model : SurvivalEstimator
        Initialized estimator supporting survival prediction.
    X : numpy.ndarray, shape (n_samples, n_features)
        Samples whose survival curves define the horizon.
    tol : float, default=0.01
        Target survival probability.
    q : float, default=0.5
        Across-sample survival quantile in [0, 1].

    Returns
    -------
    float
        Approximate upper horizon from three logarithmic grid refinements.

    Notes
    -----
    Searches between 1e-6 and 1e6 initially, using 50 points per refinement.
    This is a plotting heuristic, not an exact quantile inversion or root
    solver; grid boundary cases and unreachable tolerances need review.
    """
    min_time_log = -6.0
    max_time_log = 6.0
    for _ in range(3):
        time = np.logspace(min_time_log, max_time_log, 50)
        survs = model.predict("survival", X, time)
        qsurvs = np.quantile(survs, q, axis=0)
        max_time_idx = np.argmin(abs(qsurvs - tol))
        min_time_log = np.log10(time[max_time_idx - 1])
        max_time_log = np.log10(time[max_time_idx])
    return np.pow(10.0, max_time_log)


def replace_zero_times(times):
    r"""Replace zero times with the smallest strictly positive time.

    Parameters
    ----------
    times : numpy.ndarray
        Observed times containing at least one positive value.

    Returns
    -------
    numpy.ndarray
        New array with zeros replaced; other values are retained.

    Raises
    ------
    ValueError
        If no strictly positive time is present.

    Notes
    -----
    Negative values are left unchanged; this is not target validation.
    """
    min_non_zero = np.min(times[times > 0])
    return np.where(times == 0, min_non_zero, times)
