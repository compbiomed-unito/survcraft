r"""Scikit-learn estimators and factories for composable survival models.

Notes
-----
Adapters build fresh PyTorch modules from estimator parameters. See
:class:`SurvivalEstimator` for data, prediction, device, and fitted-state
contracts; :class:`SurvivalPredictor` trains models and
:class:`SurvivalSimulator` samples from fixed models.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from . import input_modules
from . import survival_modules
from sklearn.base import BaseEstimator
from . import loss_modules
from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar, Literal, Mapping, TypeVar, cast
from torch import Tensor
from torch.nn import Module
from typing import Union, Optional, Sequence
from numpy.typing import ArrayLike
from sklearn.model_selection import train_test_split
import copy
import numpy
import torch
import random
import warnings
import math
import functools


CheckDivergence = Literal["warn", "raise", "no"]
TensorPrediction = Union[Tensor, dict[str, Tensor]]
ArrayPrediction = Union[numpy.ndarray, dict[str, numpy.ndarray]]
Method = TypeVar("Method", bound=Callable[..., Any])


__all__ = [
    "BaseInputAdapter",
    "LinearFunctionInputAdapter",
    "FeedForwardNetAdapter",
    "BaseSurvivalAdapter",
    "ExponentialSurvivalAdapter",
    "WeibullSurvivalAdapter",
    "LogNormalSurvivalAdapter",
    "LevySurvivalAdapter",
    "InverseGaussianSurvivalAdapter",
    "StepExpSurvivalAdapter",
    "ProportionalHazardSurvivalAdapter",
    "AcceleratedFailureTimeSurvivalAdapter",
    "MixtureSurvivalAdapter",
    "FractalNoiseSurvivalAdapter",
    "SurvivalEstimator",
    "SurvivalPredictor",
    "SurvivalSimulator",
    "FailedConvergence",
]


def _validate_time_grid(values: ArrayLike, *, name: str) -> numpy.ndarray:
    r"""Validate a zero-based increasing grid in model float32 precision.

    Parameters
    ----------
    values : array-like
        One-dimensional numeric time grid with at least two points.
    name : str
        Argument name used in diagnostics.

    Returns
    -------
    numpy.ndarray of float32
        Finite nonnegative grid starting at zero and strictly increasing.

    Raises
    ------
    ValueError
        If the grid violates these conditions, including after conversion.
    """

    try:
        grid = numpy.asarray(values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a one-dimensional numeric grid") from exc
    if grid.ndim != 1 or grid.size < 2:
        raise ValueError(f"{name} must be one-dimensional with at least two points")
    if (
        grid.dtype.kind not in "iuf"
        or not numpy.isfinite(grid).all()
        or (grid < 0).any()
    ):
        raise ValueError(f"{name} must contain finite nonnegative numeric values")
    if grid[0] != 0:
        raise ValueError(f"{name} must start at zero")
    if not (grid[1:] > grid[:-1]).all():
        raise ValueError(f"{name} must be strictly increasing")
    with numpy.errstate(over="ignore", under="ignore", invalid="ignore"):
        grid = grid.astype(numpy.float32)
    if not numpy.isfinite(grid).all():
        raise ValueError(f"{name} must remain finite after float32 conversion")
    if not (grid[1:] > grid[:-1]).all():
        raise ValueError(f"{name} must remain strictly increasing after float32 conversion")
    return grid


# Input modules
class BaseInputAdapter(BaseEstimator, ABC):
    r"""Construct input modules from scikit-learn estimator parameters.

    Attributes
    ----------
    module_class : type of torch.nn.Module
        Constructor supplied by each concrete adapter.

    Notes
    -----
    ``get_module(d, p)`` constructs a fresh module using ``input_size=d``,
    ``output_size=p``, and the adapter parameters from ``get_params()``. Its
    forward method maps float32 features of shape ``(n_samples, d)`` to
    unconstrained distribution parameters of shape ``(n_samples, p)``.
    Construction does not fit the module or move it to an estimator device;
    :class:`SurvivalEstimator` moves the assembled model. Adapter instances
    expose scikit-learn ``get_params`` and ``set_params``.
    """
    @property
    @abstractmethod
    def module_class(self) -> type[Module]:
        r"""Return the module constructor.

        Returns
        -------
        type of torch.nn.Module
            Constructor accepting feature and parameter widths and adapter options.
        """
        raise NotImplementedError

    def get_module(self, input_size: int, output_size: int) -> Module:
        r"""Build a fresh feature-to-parameter module.

        Parameters
        ----------
        input_size : int
            Number of input features.
        output_size : int
            Number of raw survival parameters.

        Returns
        -------
        torch.nn.Module
            Newly constructed module following the :class:`BaseInputAdapter` contract.
        """
        return self.module_class(
            input_size=input_size, output_size=output_size, **self.get_params()
        )


@dataclass
class LinearFunctionInputAdapter(BaseInputAdapter):
    r"""Build a fixed affine feature map.

    Parameters
    ----------
    mode : {'identity', 'random'}, default='identity'
        Use a rectangular identity matrix or uniform random weights.
    multiplier : float, default=1.0
        Multiply the weight matrix by this value.
    shift : float, default=0.0
        Add this scalar to every output.
    use_first_n_feats : int, optional
        Set weight rows after this many features to zero.
    seed : int, optional
        Seed the local PyTorch generator in random mode.

    Notes
    -----
    Uses the construction contract in :class:`BaseInputAdapter`. The output
    is ``X @ W + shift``; weights are buffers, not trainable parameters.

    See Also
    --------
    survcraft.input_modules.LinearFunctionInputModule : Affine implementation.
    """
    module_class = input_modules.LinearFunctionInputModule

    mode: Literal["identity", "random"] = "identity"
    multiplier: float = 1.0
    shift: float = 0.0
    use_first_n_feats: Optional[int] = None
    seed: Optional[int] = None


@dataclass
class FeedForwardNetAdapter(BaseInputAdapter):
    r"""Build a trainable feed-forward feature map.

    Parameters
    ----------
    hidden_sizes : sequence of int, default=[]
        Hidden layer widths; the default has no hidden layers.
    hidden_activation : callable, default=torch.nn.ReLU
        Zero-argument constructor called for each hidden activation.
    output_activation : callable, optional
        Zero-argument constructor for an optional output activation.
    batch_norm : bool, default=False
        Add batch normalization before each hidden activation.
    dropout : float, default=0.0
        Dropout probability after each hidden activation.

    Notes
    -----
    Uses the construction contract in :class:`BaseInputAdapter`. Output
    values are raw parameters; distribution constraints are applied later
    by the survival module.

    See Also
    --------
    survcraft.input_modules.FeedForwardNet : Network implementation.
    """
    module_class = input_modules.FeedForwardNet

    hidden_sizes: Sequence[int] = field(default_factory=list)
    #shape: Optional[(Literal['barrel'], Literal['input'] | int, int)] = None
    hidden_activation: Callable[[], Module] = torch.nn.ReLU
    output_activation: Optional[Callable[[], Module]] = None
    batch_norm: bool = False
    dropout: float = 0.0

    # this cannot work because here we do not know the input size...
    #def __post_init__(self):
    #    if self.shape is not None:
    #        assert self.hidden_sizes == []
    #        shape, width, layers = self.shape
    #        if width == 'input':


#####################
# Survival adapters #
#####################


class BaseSurvivalAdapter(BaseEstimator, ABC):
    r"""Construct survival modules, optionally using observed outcomes.

    Attributes
    ----------
    module_class : type of survcraft.survival_modules.BaseSurvivalModule
        Constructor supplied by concrete adapters.
    param_funcs : dict of str to callable
        Parameter transformations called as ``func(value, event, time)``.

    Notes
    -----
    ``get_module`` validates paired outcome vectors, transforms constructor
    parameters from ``get_params(deep=False)``, and creates a fresh module.
    The input adapter uses its ``get_param_number()`` as the output width.
    See :class:`survcraft.survival_modules.BaseSurvivalModule` for raw and
    processed parameter shapes and prediction modes. Construction neither
    fits the module nor moves it to a device. Outcomes are boolean event
    indicators and finite nonnegative times; both may be omitted for
    simulation. Outcome-dependent adapters then use their documented
    fallbacks. Nested adapters are recursively constructed by ``param_funcs``.
    """
    @property
    @abstractmethod
    def module_class(self) -> type[survival_modules.BaseSurvivalModule]:
        r"""Return the survival module constructor.

        Returns
        -------
        type of survcraft.survival_modules.BaseSurvivalModule
            Constructor accepting transformed adapter parameters.
        """
        raise NotImplementedError

    # functions to be applied to parameters for preprocessing before passing it to the torch module (for instance for arrays that needs to be converted to tensors or for derived modules that need to initialized their submodules)
    param_funcs: ClassVar[
        dict[str, Callable[[Any, Optional[numpy.ndarray], Optional[numpy.ndarray]], Any]]
    ] = {}

    def get_module(
        self, event: Optional[ArrayLike], time: Optional[ArrayLike]
    ) -> survival_modules.BaseSurvivalModule:
        r"""Build a fresh survival module using optional outcomes.

        Parameters
        ----------
        event : array-like of bool, shape (n_samples,), or None
            True for observed events, False for right censoring.
        time : array-like, shape (n_samples,), or None
            Finite nonnegative observed event or censoring times.

        Returns
        -------
        survcraft.survival_modules.BaseSurvivalModule
            Module following the :class:`BaseSurvivalAdapter` construction contract.

        Raises
        ------
        ValueError
            If only one outcome vector is supplied, their shapes or dtypes are
            invalid, or parameter preprocessing rejects an option.
        """
        if (event is None) != (time is None):
            raise ValueError("event and time must both be supplied or both be None")
        if event is not None:
            event, time = numpy.asarray(event), numpy.asarray(time)
            if event.ndim != 1 or time.ndim != 1 or event.shape != time.shape:
                raise ValueError("event and time must be matching one-dimensional arrays")
            if event.dtype != numpy.bool_:
                raise ValueError("event must be a boolean array")
            if (
                time.dtype.kind not in "iuf"
                or not numpy.isfinite(time).all()
                or (time < 0).any()
            ):
                raise ValueError("time must contain finite nonnegative numbers")
        params = {
            k: self.param_funcs.get(k, lambda x, event, time: x)(v, event, time)
            for k, v in self.get_params(deep=False).items()
        }
        return self.module_class(**params)


class ExponentialSurvivalAdapter(BaseSurvivalAdapter):
    r"""Build a rate-parameterized exponential survival module.

    Notes
    -----
    Uses the construction contract in :class:`BaseSurvivalAdapter` and the
    parameterization and formulas in
    :class:`survcraft.survival_modules.ExponentialSurvivalModule`.
    """
    module_class = survival_modules.ExponentialSurvivalModule


class WeibullSurvivalAdapter(BaseSurvivalAdapter):
    r"""Build a Weibull survival module.

    Notes
    -----
    Uses the construction contract in :class:`BaseSurvivalAdapter` and the
    parameterization and formulas in
    :class:`survcraft.survival_modules.WeibullSurvivalModule`.
    """
    module_class = survival_modules.WeibullSurvivalModule


class LogNormalSurvivalAdapter(BaseSurvivalAdapter):
    r"""Build a log-normal survival module.

    Notes
    -----
    Uses the construction contract in :class:`BaseSurvivalAdapter` and the
    parameterization and formulas in
    :class:`survcraft.survival_modules.LogNormalSurvivalModule`.
    """
    module_class = survival_modules.LogNormalSurvivalModule

class LevySurvivalAdapter(BaseSurvivalAdapter):
    r"""Build a Lévy first-passage survival module.

    Notes
    -----
    Uses the construction contract in :class:`BaseSurvivalAdapter` and the
    parameterization and formulas in
    :class:`survcraft.survival_modules.LevySurvivalModule`.
    """
    module_class = survival_modules.LevySurvivalModule
    
class InverseGaussianSurvivalAdapter(BaseSurvivalAdapter):
    r"""Build a inverse-Gaussian first-passage survival module.

    Notes
    -----
    Uses the construction contract in :class:`BaseSurvivalAdapter` and the
    parameterization and formulas in
    :class:`survcraft.survival_modules.InverseGaussianSurvivalModule`.
    """
    module_class = survival_modules.InverseGaussianSurvivalModule


@dataclass
class FractalNoiseSurvivalAdapter(BaseSurvivalAdapter):
    r"""Build a random interpolated survival profile for simulation.

    Parameters
    ----------
    max_time : float, default=1.0
        End of the interpolation grid; should be positive.
    backbone_length : int, default=1
        Number of random backbone points before the terminal zero.
    seed : int, optional
        Seed for Python's global random generator during construction.

    Notes
    -----
    Uses :class:`BaseSurvivalAdapter`. This module has no trainable raw
    parameters and uses NumPy interpolation; see the numerical and device
    limitations in :class:`survcraft.survival_modules.FractalNoiseSurvivalModule`.
    """
    max_time: float = 1.0
    backbone_length: int = 1
    seed: Optional[int] = None

    module_class = survival_modules.FractalNoiseSurvivalModule


@dataclass
class StepExpSurvivalAdapter(BaseSurvivalAdapter):
    r"""Build a piecewise constant density with an exponential tail.

    Parameters
    ----------
    breaks : int or array-like, default=10
        Number of quantile breakpoints or an explicit zero-based, finite,
        strictly increasing grid with at least two float32-distinct points.
    trainable_breaks : bool, default=False
        Optimize the positive interval lengths during training.

    Raises
    ------
    ValueError
        If the grid is invalid or fewer than two breaks can be derived.

    Warnings
    --------
    UserWarning
        If fewer unique quantile breaks are available than requested.

    Notes
    -----
    Uses :class:`BaseSurvivalAdapter`. An integer ``b`` requests quantiles
    ``arange(b) / b`` of observed event times; the first break is set to zero
    and duplicates are removed. Without outcomes the grid is ``arange(b)/b``.
    This describes a piecewise constant *density*, not a piecewise constant
    hazard. See :class:`survcraft.survival_modules.StepExpSurvivalModule`
    for formulas.
    """
    breaks: Union[int, ArrayLike] = 10
    trainable_breaks: bool = False

    @staticmethod
    def preprocess_breaks(
        breaks: Union[int, ArrayLike],
        event: Optional[numpy.ndarray],
        time: Optional[numpy.ndarray],
    ) -> Tensor:
        r"""Convert an explicit grid or event quantiles to float32 breaks.

        Parameters
        ----------
        breaks : int or array-like
            Break specification described in :class:`StepExpSurvivalAdapter`.
        event : numpy.ndarray of bool or None
            Validated event indicators.
        time : numpy.ndarray or None
            Matching observed times.

        Returns
        -------
        torch.Tensor, shape (n_breaks,)
            Zero-based, finite, strictly increasing grid on the CPU.

        Raises
        ------
        ValueError
            If breaks are invalid or observed events cannot provide two breaks.

        Notes
        -----
        When outcomes are supplied, ``event`` and ``time`` must both be present.
        Duplicate quantiles are removed with a warning if the count decreases.
        """
        if isinstance(breaks, (int, numpy.integer)):
            if breaks < 2:
                raise ValueError("breaks count must be at least two")
            time_breaks = numpy.arange(breaks, dtype=float) / breaks
            if event is not None:
                event_times = time[event]
                if event_times.size == 0:
                    raise ValueError("cannot derive breaks without observed event times")
                time_breaks = numpy.unique(numpy.quantile(event_times, time_breaks))
                time_breaks[0] = 0.0
                if len(time_breaks) < 2:
                    raise ValueError("cannot derive at least two unique breaks from observed event times; provide explicit breaks")
                if len(time_breaks) < breaks:
                    warnings.warn(f'only {len(time_breaks)} unique breaks were obtained instead of the {breaks} breaks requested in the argument for the StepExpSurvivalAdapter', stacklevel=2)
        else:
            time_breaks = breaks
        return torch.tensor(_validate_time_grid(time_breaks, name="breaks"))
 

    module_class = survival_modules.StepExpSurvivalModule
    param_funcs = {
        "breaks": preprocess_breaks,
    }


@dataclass
class DerivedSurvivalAdapter(BaseSurvivalAdapter):
    # abstract class for common behaviour between PH and AFT adapters
    r"""Share baseline construction for transformed survival models.

    Parameters
    ----------
    baseline : BaseSurvivalAdapter, optional
        Baseline distribution factory; defaults to a new StepExpSurvivalAdapter.
    baseline_params : array-like, shape (n_baseline_params,), optional
        Fixed raw baseline parameters, converted to a float32 tensor. If None,
        the module creates trainable baseline parameters.

    Notes
    -----
    Abstract factory using :class:`BaseSurvivalAdapter`. Its concrete
    subclasses supply ``module_class``; nested baselines receive the same
    outcome vectors.
    """
    baseline: BaseSurvivalAdapter = field(default_factory=StepExpSurvivalAdapter)
    baseline_params: Optional[ArrayLike] = None

    param_funcs = {
        "baseline": lambda baseline, event, time: baseline.get_module(event, time),
        "baseline_params": lambda x, event, time: (
            None if x is None else torch.tensor(x, dtype=torch.float32)
        ),
    }

@dataclass
class ProportionalHazardSurvivalAdapter(DerivedSurvivalAdapter):
    r"""Build a proportional hazards survival module.

    Parameters
    ----------
    baseline : BaseSurvivalAdapter, optional
        Baseline distribution factory; defaults to a new StepExpSurvivalAdapter.
    baseline_params : array-like, shape (n_baseline_params,), optional
        Fixed raw baseline parameters, converted to a float32 tensor. If None,
        the module creates trainable baseline parameters.

    Notes
    -----
    Includes the inherited :class:`DerivedSurvivalAdapter` options and
    uses :class:`BaseSurvivalAdapter` construction. See
    :class:`survcraft.survival_modules.ProportionalHazardSurvivalModule` for
    formulas and the handling of fixed baseline tensors.
    """
    module_class = survival_modules.ProportionalHazardSurvivalModule

@dataclass
class AcceleratedFailureTimeSurvivalAdapter(DerivedSurvivalAdapter):
    r"""Build a time-scaled survival module.

    Parameters
    ----------
    baseline : BaseSurvivalAdapter, optional
        Baseline distribution factory; defaults to a new StepExpSurvivalAdapter.
    baseline_params : array-like, shape (n_baseline_params,), optional
        Fixed raw baseline parameters, converted to a float32 tensor. If None,
        the module creates trainable baseline parameters.

    Notes
    -----
    Includes the inherited :class:`DerivedSurvivalAdapter` options and
    uses :class:`BaseSurvivalAdapter` construction. See
    :class:`survcraft.survival_modules.AcceleratedFailureTimeSurvivalModule` for
    formulas and the handling of fixed baseline tensors.
    """
    module_class = survival_modules.AcceleratedFailureTimeSurvivalModule


@dataclass
class MixtureSurvivalAdapter(BaseSurvivalAdapter):
    r"""Build a mixture with feature-dependent weights and component parameters.

    Parameters
    ----------
    baselines : sequence of BaseSurvivalAdapter, optional
        At least two component factories; defaults to three independent
        StepExpSurvivalAdapter instances.

    Raises
    ------
    ValueError
        If fewer than two baselines are supplied.

    Notes
    -----
    Uses :class:`BaseSurvivalAdapter`. Every component receives the same
    outcomes. See :class:`survcraft.survival_modules.MixtureSurvivalModule`
    for formulas and raw parameter ordering.
    """
    baselines: Sequence[BaseSurvivalAdapter] = field(
        default_factory=lambda: [StepExpSurvivalAdapter() for _ in range(3)]
    )
    # TODO add baseline_parameters like ProportionalHazards and AFT

    @staticmethod
    def preprocess_baselines(
        baselines: Sequence[BaseSurvivalAdapter],
        event: Optional[numpy.ndarray],
        time: Optional[numpy.ndarray],
    ) -> list[survival_modules.BaseSurvivalModule]:
        r"""Construct each mixture component from the shared outcomes.

        Parameters
        ----------
        baselines : sequence of BaseSurvivalAdapter
            At least two component factories.
        event : numpy.ndarray of bool or None
            Event indicators supplied to every factory.
        time : numpy.ndarray or None
            Matching observed times.

        Returns
        -------
        list of survcraft.survival_modules.BaseSurvivalModule
            Fresh component modules in the supplied order.

        Raises
        ------
        ValueError
            If fewer than two baselines are supplied or construction fails.
        """
        if len(baselines) < 2:
            raise ValueError("baselines must be a sequence of at least two survival adapter instances")
        return [
            b.get_module(event, time) for b in baselines
        ]

    module_class = survival_modules.MixtureSurvivalModule
    param_funcs = {
        "baselines": preprocess_baselines,
    }

def shape2str(x: Union[Tensor, numpy.ndarray]) -> str:
    r"""Format tensor or array dimensions for diagnostics.

    Parameters
    ----------
    x : torch.Tensor or numpy.ndarray
        Object whose shape is formatted.

    Returns
    -------
    str
        Dimensions separated by ``x``; empty for a scalar.
    """
    return 'x'.join(map(str, x.shape))


check_divergence_values: tuple[CheckDivergence, ...] = "warn", "raise", "no"
class TorchModel(torch.nn.Module):
    r"""Compose a feature map and a survival distribution module.

    Parameters
    ----------
    input_module : torch.nn.Module
        Feature-to-raw-parameter map following :class:`BaseInputAdapter`.
    survival_module : survcraft.survival_modules.BaseSurvivalModule
        Distribution module following :class:`BaseSurvivalAdapter`.
    check_divergence : {"raise", "warn", "no"}, default="raise"
        Policy for non-finite raw parameters and predictions.

    Attributes
    ----------
    input_module : torch.nn.Module
        Registered feature map.
    survival_module : survcraft.survival_modules.BaseSurvivalModule
        Registered distribution module.

    Raises
    ------
    ValueError
        If the divergence policy is unknown.

    Notes
    -----
    Calling the model uses the prediction modes and shapes documented in
    :class:`SurvivalEstimator`, returning tensors on the model device.
    """

    def __init__(
        self,
        input_module: Module,
        survival_module: survival_modules.BaseSurvivalModule,
        check_divergence: CheckDivergence = "raise",
    ) -> None:
        r"""Initialize the two registered modules and divergence policy.

        See Also
        --------
        TorchModel : Constructor parameters and module contracts.
        """
        super().__init__()
        self.input_module = input_module
        self.survival_module = survival_module
        if check_divergence not in check_divergence_values:
            raise ValueError(f"unknown value {check_divergence} for `check_divergence`, must be one of {check_divergence_values}")
        self.check_divergence = check_divergence

    def get_raw_params(self, x: Tensor) -> Tensor:
        r"""Compute unconstrained distribution parameters.

        Parameters
        ----------
        x : torch.Tensor, shape (n_samples, n_features)
            Features on the model device.

        Returns
        -------
        torch.Tensor, shape (n_samples, n_raw_params)
            Feature-map output; gradients remain enabled by the caller.
        """
        return self.input_module(x)

    def get_processed_params(self, x: Tensor) -> dict[str, Tensor]:
        r"""Compute named, transformed distribution parameters.

        Parameters
        ----------
        x : torch.Tensor, shape (n_samples, n_features)
            Features on the model device.

        Returns
        -------
        dict of str to torch.Tensor
            Parameter groups with shape ``(n_samples, group_width)``.

        Notes
        -----
        This inspection method bypasses the divergence checks in ``forward``.
        """
        return self.survival_module.preprocess_params(self.input_module(x))

    def _check_tensor(
        self,
        x: Tensor,
        name: str,
        times: Optional[Tensor] = None,
        context: Mapping[str, Any] = {},
    ) -> None:
        r"""Apply the configured policy to non-finite tensor values.

        Parameters
        ----------
        x : torch.Tensor
            Tensor to check.
        name : str
            Tensor description for diagnostics.
        times : torch.Tensor, optional
            Scalar or vector of associated prediction times.
        context : mapping, optional
            Additional diagnostic values attached to a raised exception.

        Returns
        -------
        None
            Returns after checking or when checks are disabled.

        Raises
        ------
        ValueError
            If non-finite values occur under the raise policy.

        Notes
        -----
        The warn policy emits a warning. Raised exceptions include a dict
        of tensor values and context as their second argument.
        """
        if self.check_divergence != "no":
            x_fail = ~x.isfinite()
            if x_fail.any():
                desc = f"non-finite value(s) in {name}: {x.isinf().sum()}inf+{x.isnan().sum()}nan/{shape2str(x)}tot"
                if times is not None:
                    if times.ndim == 0:
                        bad_sample_num = x_fail.sum()
                        bad_times = times.reshape(1)
                    else:
                        bad_sample_num = (x_fail.any(dim=1)).sum()
                        bad_times = times[x_fail.any(dim=0)]
                    bad_time_head = ', '.join(map(str, bad_times[:5].tolist()))
                    if len(bad_times) > 5: bad_time_head += '...'
                    desc += f", for {bad_sample_num} samples at {len(bad_times)} times ({bad_time_head})"
                if self.check_divergence == "raise":
                    ctx = {"tensor_name": name, "tensor_values": x.detach().cpu(), "times": times}
                    ctx.update(context)
                    raise ValueError(desc, ctx)
                elif self.check_divergence == "warn":
                    warnings.warn(desc)

    def forward(
        self, mode: str, x: Tensor, times: Optional[Tensor] = None
    ) -> TensorPrediction:
        r"""Predict a distribution quantity and check for non-finite values.

        Parameters
        ----------
        mode : str
            Prediction mode from :class:`SurvivalEstimator`.
        x : torch.Tensor, shape (n_samples, n_features)
            Features on the model device.
        times : torch.Tensor, scalar or shape (n_times,), optional
            Shared evaluation times for time-dependent modes.

        Returns
        -------
        torch.Tensor or dict of str to torch.Tensor
            Output shapes follow :meth:`SurvivalEstimator.predict`.

        Raises
        ------
        ValueError
            If a value is non-finite and ``check_divergence="raise"``, or the
            survival module rejects the mode or parameter width.

        See Also
        --------
        survcraft.survival_modules.BaseSurvivalModule.forward : Mode validation.
        """
        params = self.input_module(x)
        self._check_tensor(params, name="raw params") # times not needed since params are not indexed by times

        preds = self.survival_module(mode, params, times)
        if mode == "params":
            for name, values in preds.items():
                self._check_tensor(values, name=f"params[{name}]", context={"raw_params": params})
        else:
            self._check_tensor(preds, name=mode, times=times, context={"raw_params": params})

        return preds

# decorator to add context to exceptions in SurvivalEstimator methods
def add_exception_context(method: Method) -> Method:
    r"""Annotate estimator exceptions with their method and estimator.

    Parameters
    ----------
    method : callable
        Estimator method to wrap.

    Returns
    -------
    callable
        Wrapped method preserving the original signature and docstring.

    Notes
    -----
    Exception notes are added on Python 3.11 and later.
    """
    @functools.wraps(method)
    def wrapper(self: SurvivalEstimator, *args: Any, **kwargs: Any) -> Any:
        try:
            return method(self, *args, **kwargs)
        except Exception as exc:
            note = f"Raised in method {method.__name__} of {self}"
            try:
                exc.add_note(note)
            except AttributeError:
                pass # add_note was added in python 3.11
            raise
    return cast(Method, wrapper)
# TEST CODE
#import numpy as np
#import survcraft.adapters as ad
#y = np.array([(0.0, False),], dtype=[('time', 'f8'), ('event', '?')])
#ad.SurvivalPredictor().fit(np.array([0.0]), y)


@dataclass
class SurvivalEstimator(BaseEstimator):
    r"""Define shared data, prediction, device, and model-state contracts.

    Parameters
    ----------
    input : BaseInputAdapter, optional
        Feature-map factory; the base estimator defaults to None.
    survival : BaseSurvivalAdapter, optional
        Distribution factory; the base estimator defaults to None.
    loss : survcraft.loss_modules.BaseSurvivalLoss, optional
        Training objective; the base estimator defaults to None.
    device : str or sequence of str, default='cpu'
        PyTorch device or nonempty list of candidate devices. One candidate
        is randomly chosen and cached on first use, not distributed across.
    verbose : int, default=0
        Training output level: 0 is quiet, 1 periodic, 2 per epoch, 3 per batch.
    check_divergence : {'raise', 'warn', 'no'}, default='raise'
        Policy for non-finite raw parameters and predictions.

    Attributes
    ----------
    model_ : TorchModel
        Assembled model, present after predictor initialization or the first
        simulator prediction. Its presence alone does not prove successful training.
    device_ : str
        Cached device, created on first device use.

    Notes
    -----
    Features are dense numeric arrays of shape ``(n_samples, n_features)``.
    The caller encodes categories and handles missing values; no feature
    scaling or imputation is performed. Features and times are copied to
    float32 tensors; events use boolean tensors. Direct tensor callers must
    put model, losses, and data on compatible devices.

    Predictor targets are one-dimensional NumPy structured arrays with
    exactly two scalar fields: boolean event first, finite nonnegative time
    second. Field names are arbitrary. True means an observed event; False
    means right censoring. ``train`` accepts separate outcome vectors.

    Time-dependent modes are ``failure``, ``survival``, ``density``, and
    ``hazard``. A shared time vector gives ``(n_samples, n_times)`` outputs;
    a scalar gives ``(n_samples,)``. ``expected_time``, ``median_time``, and
    ``risk`` take no times and give ``(n_samples,)``. Higher risk means
    earlier events. Not every distribution implements every summary mode.
    ``params`` takes no times and returns named transformed parameter arrays
    of shape ``(n_samples, group_width)``. All array predictions return to CPU.

    The base ``fit`` is a no-op and does not create ``model_``. Predictors
    require ``fit`` or ``train`` before prediction; simulators initialize
    lazily. Module construction creates fresh input and survival modules
    and moves the assembled model to ``device_``. Changing estimator options
    does not rebuild an existing model automatically.

    See Also
    --------
    SurvivalPredictor : Train a model from censored outcomes.
    SurvivalSimulator : Sample outcomes from a lazily constructed model.
    BaseInputAdapter : Feature-map construction.
    BaseSurvivalAdapter : Distribution construction.
    """
    input: Optional[BaseInputAdapter] = None
    survival: Optional[BaseSurvivalAdapter] = None
    loss: Optional[loss_modules.BaseSurvivalLoss] = None

    device: Union[str, Sequence[str]] = "cpu"
    verbose: int = 0
    check_divergence: CheckDivergence = "raise"
    # precision: torch.dtype = torch.float32

    def _get_device(self) -> str:
        r"""Select and cache one configured device.

        Returns
        -------
        str
            Existing ``device_`` or a newly selected device.

        Notes
        -----
        A sequence is sampled using Python random.choice once. Changing the
        device option later does not clear the cached choice.
        """
        try:
            return self.device_
        except AttributeError:
            # device_ not present, set it only once
            self.device_ = (
                self.device
                if isinstance(self.device, str) else
                # multiple devices, randomly pick one for multiprocessing purposes
                random.choice(self.device)
            )
        return self.device_

    def _tensor(
        self,
        a: ArrayLike,
        dtype: Optional[torch.dtype] = torch.float32,
        device: Optional[Union[str, torch.device]] = None,
    ) -> Tensor:
        r"""Copy array-like values into a tensor on the requested device.

        Parameters
        ----------
        a : array-like
            Numeric values to copy.
        dtype : torch.dtype or None, default=torch.float32
            Requested dtype; None lets PyTorch infer it.
        device : str or torch.device, optional
            Target device; defaults to the cached estimator device.

        Returns
        -------
        torch.Tensor
            Tensor containing the copied values.

        Notes
        -----
        Retries a ValueError using a.copy(), accommodating structured-target
        field views whose strides PyTorch cannot convert directly.
        """
        # if dtype is None:
        #    dtype = self.precision
        if device is None:
            device = self._get_device()

        try:
            return torch.tensor(a, device=device, dtype=dtype)
        # while converting survival y in sksurv format:
        # ValueError: given numpy array strides not a multiple of the element byte size. Copy the numpy array to reallocate the memory.
        except ValueError:
            return torch.tensor(a.copy(), device=device, dtype=dtype)

    def _validate_dataset(
        self,
        X: numpy.ndarray,
        event: numpy.ndarray,
        time: Optional[numpy.ndarray] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        # sksurv seems to always put event before time, we follow the convention
        r"""Check basic dataset invariants and copy features and outcomes.

        Parameters
        ----------
        X : numpy.ndarray, shape (n_samples, n_features)
            Numeric feature matrix.
        event : numpy.ndarray of bool, shape (n_samples,)
            Observed-event indicators.
        time : numpy.ndarray, shape (n_samples,), optional
            Required observed times; the None path is unimplemented.
        device : str or torch.device, optional
            Tensor device; defaults to the estimator device.

        Returns
        -------
        tuple of torch.Tensor
            Float32 features, boolean events, and float32 times.

        Raises
        ------
        ValueError
            If the minimum time is negative.
        AssertionError
            If counts differ, events are not boolean, time is omitted, or the
            nonnegative-time assertion fails.

        Notes
        -----
        Warns for zero times. This helper does not fully validate feature
        shape or time finiteness; fit performs separate structured-target checks.
        """
        assert X.shape[0] == event.shape[0]
        if time is None:
            assert False, "implement scikit-survival style structured array"
        else:
            min_time = time.min()
            if min_time < 0:
                raise ValueError('times cannot be negative')
            if min_time == 0:
                ztime = time == 0
                warnings.warn(f'Found {ztime.sum()} ({ztime.mean():.2%}) times equal to zero, these can cause problem in training for some survival modules (e.g. Weibull)')

            assert X.shape[0] == time.shape[0]
            assert event.dtype == bool, (
                "event is not boolean, perhaps we should accept 0/1 arrays? numpy.unique(event)="
                + str(numpy.unique(event))
            )
            assert all(time >= 0.0)
            event_t = self._tensor(event, dtype=torch.bool, device=device)
            time_t = self._tensor(time, device=device)
        X_t = self._tensor(X, device=device)
        return X_t, event_t, time_t

    def _init_model(
        self,
        X: numpy.ndarray,
        event: Optional[ArrayLike] = None,
        time: Optional[ArrayLike] = None,
    ) -> None:
        r"""Construct and move a fresh model from adapter factories.

        Parameters
        ----------
        X : numpy.ndarray, shape (n_samples, n_features)
            Features used to infer input width.
        event : array-like of bool, optional
            Observed events for outcome-dependent adapter construction.
        time : array-like, optional
            Matching observed times.

        Returns
        -------
        None
            Assigns ``model_`` and selects ``device_`` if needed.

        Notes
        -----
        The input output width comes from survival_module.get_param_number().
        Requires input and survival adapters to be configured.
        """
        survival_module = self.survival.get_module(
            event=event,
            time=time,
        )
        self.model_ = TorchModel(
            input_module=self.input.get_module(
                input_size=X.shape[1],
                output_size=survival_module.get_param_number(),
            ),
            survival_module=survival_module,
            check_divergence=self.check_divergence,
        ).to(self._get_device())

    def fit(self, X: numpy.ndarray, y: numpy.ndarray) -> SurvivalEstimator:
        # warnings.warn(f"Fit of {self.__class__.__name__} does nothing")
        # print(f"Fit of {self.__class__.__name__} does nothing") # maybe use a warning
        r"""Return this base estimator without constructing or training a model.

        Parameters
        ----------
        X : numpy.ndarray
            Unused feature data.
        y : numpy.ndarray
            Unused target data.

        Returns
        -------
        SurvivalEstimator
            This estimator unchanged.

        See Also
        --------
        SurvivalPredictor.fit : Training implementation.
        """
        return self

    def predict_survival(
        self, X: numpy.ndarray, times: Optional[ArrayLike] = None
    ) -> numpy.ndarray:
        r"""Evaluate survival probabilities at shared times.

        Parameters
        ----------
        X : numpy.ndarray, shape (n_samples, n_features)
            Numeric feature matrix.
        times : array-like, scalar or shape (n_times,), optional
            Required evaluation times despite the default None.

        Returns
        -------
        numpy.ndarray
            Shape ``(n_samples, n_times)`` for a vector or ``(n_samples,)`` for a
            scalar.

        See Also
        --------
        SurvivalEstimator.predict : Prediction contract and exceptions.
        """
        return cast(numpy.ndarray, self.predict(mode="survival", X=X, times=times))

    @add_exception_context
    def predict(
        self,
        mode: str,
        X: numpy.ndarray,
        times: Optional[ArrayLike] = None,
    ) -> ArrayPrediction:
        r"""Evaluate a distribution quantity without tracking gradients.

        Parameters
        ----------
        mode : {'failure', 'survival', 'density', 'hazard', 'expected_time', 'median_time', 'risk', 'params'}
            Distribution quantity to evaluate; see :class:`SurvivalEstimator`.
        X : numpy.ndarray, shape (n_samples, n_features)
            Numeric features following the shared estimator contract.
        times : array-like, scalar or shape (n_times,), optional
            Shared nonnegative evaluation times; required for time-dependent
            modes and omitted for summaries and parameters.

        Returns
        -------
        numpy.ndarray or dict of str to numpy.ndarray
            A matrix ``(n_samples, n_times)`` for a time vector, a vector
            ``(n_samples,)`` for a scalar time or summary, or named parameter
            arrays ``(n_samples, group_width)`` for ``params``.

        Raises
        ------
        AttributeError
            If ``model_`` has not been initialized.
        TypeError
            If mode is not a string.
        ValueError
            If mode, parameter width, or output shape is invalid, or a non-finite
            output violates the divergence policy.
        AssertionError
            If times are missing, have more than one dimension, or are supplied
            for a summary mode.
        NotImplementedError
            If the selected distribution does not implement the mode.

        Notes
        -----
        Puts ``model_`` in evaluation mode and returns CPU NumPy arrays.
        See :class:`SurvivalEstimator` for precision and device handling.
        """
        self.model_.eval()
        with torch.no_grad():
            result = self.model_(
                    mode,
                    self._tensor(X),
                    None if times is None else self._tensor(times),
                )
        if mode == 'params':
            return {k: p.detach().cpu().numpy() for k, p in result.items()}
        else:
            return result.detach().cpu().numpy()

    def plot(self, X: numpy.ndarray, max_time: Optional[float] = None) -> None:
        r"""Create plots of failure, survival, density, and hazard.

        Parameters
        ----------
        X : numpy.ndarray, shape (n_samples, n_features)
            Samples whose distributions are plotted.
        max_time : float, optional
            Final evaluation time, inferred from survival curves when omitted.

        Returns
        -------
        None
            The figure is available through matplotlib.pyplot.

        Notes
        -----
        Requires the optional plotting dependencies. The model must support
        all four time-dependent modes.

        See Also
        --------
        survcraft.plotting.plot_outputs : Plot implementation.
        """
        from .plotting import plot_outputs

        plot_outputs(self, X, max_time=max_time)



@dataclass
class SurvivalPredictor(SurvivalEstimator):
    r"""Fit a neural survival model to right-censored outcomes with Adam.

    Parameters
    ----------
    input : BaseInputAdapter, optional
        Feature-map factory; defaults to a new FeedForwardNetAdapter.
    survival : BaseSurvivalAdapter, optional
        Distribution factory; defaults to a new ExponentialSurvivalAdapter.
    loss : survcraft.loss_modules.BaseSurvivalLoss, optional
        Training objective; defaults to a new BrierLoss.
    device : str or sequence of str, default='cpu'
        PyTorch device or nonempty list of candidate devices. One candidate
        is randomly chosen and cached on first use, not distributed across.
    verbose : int, default=0
        Training output level: 0 is quiet, 1 periodic, 2 per epoch, 3 per batch.
    check_divergence : {'raise', 'warn', 'no'}, default='raise'
        Policy for non-finite raw parameters and predictions.
    batch_size : int, default=256
        Training and validation mini-batch size.
    bulk_batching : bool, default=True
        Select whole batches directly from the dataset for training and
        validation. Set False to fetch individual samples and stack them
        using PyTorch's standard automatic batching.
    learning_rate : float, default=0.005
        Adam learning rate.
    weight_decay : float, default=0.0
        Adam weight decay.
    epochs : int, default=10
        Maximum epochs per training call.
    warm_start : bool, default=False
        Reuse model weights in fit; optimizer state is recreated each call.
    early_stopping : bool, default=False
        Hold out validation data and restore the best model state.
    validation_ratio : float, default=0.1
        Fraction held out when early stopping is enabled.
    early_stopping_patience : int, default=10
        Epochs without improvement before stopping.
    data_loader_num_workers : int, default=0
        Number of PyTorch data loader worker processes.
    preload_data : bool, default=False
        Store training tensors on the selected device before batching.
    gradient_clipping : bool, default=False
        Clip parameter gradient norm to 1.0 before each optimizer step.
    history : bool, default=False
        Record batch losses and optional test losses for each completed epoch.

    Attributes
    ----------
    model_ : TorchModel
        Assembled model, present after predictor initialization or the first
        simulator prediction. Its presence alone does not prove successful training.
    device_ : str
        Cached device, created on first device use.
    train_history_ : list of tuple
        Each recorded epoch contains a NumPy vector of usable batch losses
        and a dict of test loss names to Python floats. Reset when the model
        is rebuilt; populated only when history=True.

    Notes
    -----
    Uses the shared :class:`SurvivalEstimator` contracts and all its
    inherited constructor options. For each usable batch :math:`B`, Adam
    minimizes :math:`L_B = L(M_\theta, X_B, e_B, t_B)`. The selected
    :class:`survcraft.loss_modules.BaseSurvivalLoss` defines the objective
    and reduction. Risk sets and classification time grids use the current
    batch, so batch size can change the objective. Batches without observed
    events or with non-finite scalar losses are skipped; an epoch with no
    usable batches raises :class:`FailedConvergence`.

    Early stopping uses a random, unstratified split and the unweighted mean
    of validation batch losses. Test data are evaluated for history only;
    they do not supply the early-stopping validation set. Random network
    initialization, shuffling, and validation splitting are not seeded by
    this estimator. Loss modules are not automatically moved to ``device_``.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[0.0], [1.0], [2.0]], dtype=np.float32)
    >>> y = np.array([(True, 1.0), (False, 2.0), (True, 3.0)],
    ...              dtype=[("event", "?"), ("time", "f4")])
    >>> predictor = SurvivalPredictor(epochs=1).fit(X, y)
    >>> predictor.predict_survival(X, [0.5, 1.0]).shape
    (3, 2)

    See Also
    --------
    survcraft.loss_modules.BaseSurvivalLoss : Scalar objective contract and composition.
    """
    input: BaseInputAdapter = field(default_factory=FeedForwardNetAdapter)
    survival: BaseSurvivalAdapter = field(default_factory=ExponentialSurvivalAdapter)
    loss: loss_modules.BaseSurvivalLoss = field(default_factory=loss_modules.BrierLoss)

    batch_size: int = 256
    learning_rate: float = 0.005
    weight_decay: float = 0.
    epochs: int = 10
    warm_start: bool = False

    early_stopping: bool = False
    validation_ratio: float = 0.1
    early_stopping_patience: int = 10
    data_loader_num_workers: int = 0
    preload_data: bool = False # preload data on gpu (if device is gpu), maybe find better name
    gradient_clipping: bool = False

    # these options give convenience but may have speed impact, should evaluate with some tests
    history: bool = False # collect training history data
    bulk_batching: bool = True

    def _make_data_loader(
        self, dataset: torch.utils.data.Dataset, *, shuffle: bool = False
    ) -> torch.utils.data.DataLoader:
        r"""Build a loader for tensor data or a validation/training subset.

        Parameters
        ----------
        dataset : torch.utils.data.TensorDataset or torch.utils.data.Subset
            Dataset supporting whole-batch indexing by a list of indices.
        shuffle : bool, default=False
            Shuffle sample indices each epoch.

        Returns
        -------
        torch.utils.data.DataLoader
            Feature, event, and time batches, including the final partial batch.
        """
        if self.bulk_batching:
            sampler = (
                torch.utils.data.RandomSampler(dataset) if shuffle
                else torch.utils.data.SequentialSampler(dataset)
            )
            return torch.utils.data.DataLoader(
                dataset,
                batch_size=None,
                sampler=torch.utils.data.BatchSampler(
                    sampler, batch_size=self.batch_size, drop_last=False
                ),
                num_workers=self.data_loader_num_workers,
            )
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.data_loader_num_workers,
        )

    @staticmethod
    def _extract_target(
        y: numpy.ndarray, name: str
    ) -> tuple[numpy.ndarray, numpy.ndarray]:
        r"""Validate and copy a positional structured survival target.

        Parameters
        ----------
        y : numpy.ndarray, shape (n_samples,)
            Exactly two scalar fields: boolean event, then real numeric time.
        name : str
            Argument name used in diagnostics.

        Returns
        -------
        tuple of numpy.ndarray
            Copies of the event and time fields.

        Raises
        ------
        ValueError
            If the target format is invalid or times are non-finite or negative.
        """
        if not isinstance(y, numpy.ndarray) or y.ndim != 1:
            raise ValueError(f"{name} must be a one-dimensional NumPy structured array")
        names = y.dtype.names
        if names is None or len(names) != 2:
            raise ValueError(f"{name} must have exactly two fields: event first, time second")
        event, time = y[names[0]], y[names[1]]
        if event.ndim != 1 or event.dtype != numpy.bool_:
            raise ValueError(f"{name}'s first field must contain scalar boolean events")
        if time.ndim != 1 or time.dtype.kind not in "iuf":
            raise ValueError(f"{name}'s second field must contain scalar real numeric times")
        if not numpy.isfinite(time).all() or (time < 0).any():
            raise ValueError(f"{name}'s times must be finite nonnegative numbers")
        return event.copy(), time.copy()

    def fit(
        self,
        X: numpy.ndarray,
        y: numpy.ndarray,
        X_test: Optional[numpy.ndarray] = None,
        y_test: Optional[numpy.ndarray] = None,
    ) -> SurvivalPredictor:
        r"""Train using a structured array of event and censoring times.

        Parameters
        ----------
        X : numpy.ndarray, shape (n_samples, n_features)
            Numeric features following :class:`SurvivalEstimator`.
        y : numpy.ndarray, shape (n_samples,)
            Structured target with boolean event first and numeric time second;
            field names are arbitrary. Times must be finite and nonnegative.
        X_test : numpy.ndarray, optional
            Features for optional test-loss history.
        y_test : numpy.ndarray, optional
            Matching structured test outcomes. Both test arguments are needed;
            supplying only one currently leaves test evaluation disabled.

        Returns
        -------
        SurvivalPredictor
            This estimator after training.

        Raises
        ------
        ValueError
            If a supplied structured target violates the target contract.
        FailedConvergence
            If an epoch has no usable training batches.

        Notes
        -----
        Uses the constructor warm_start option. Test data do not control early
        stopping. See :meth:`train` for the lower-level training interface.
        """
        event, time = self._extract_target(y, "y")
        train_kwargs = {
            'X': X,
            'event': event,
            'time': time,
            'warm_start': self.warm_start,
        }
        
        if X_test is not None and y_test is not None:
            event_test, time_test = self._extract_target(y_test, "y_test")
            train_kwargs['test_data'] = (
                X_test,
                event_test,
                time_test,
            )
        
        self.train(**train_kwargs)
        return self

    @add_exception_context
    def train(
        self,
        X: numpy.ndarray,
        event: numpy.ndarray,
        time: numpy.ndarray,
        warm_start: bool = False,
        test_data: Optional[tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]] = None,
        test_losses: Optional[Sequence[loss_modules.BaseSurvivalLoss]] = None,
    ) -> None:
        r"""Train with separate event and time vectors.

        Parameters
        ----------
        X : numpy.ndarray, shape (n_samples, n_features)
            Numeric feature matrix.
        event : numpy.ndarray of bool, shape (n_samples,)
            True for observed events, False for right censoring.
        time : numpy.ndarray, shape (n_samples,)
            Finite nonnegative event or censoring times. Zero times trigger a
            warning because some distributions are singular there.
        warm_start : bool, default=False
            Reuse existing model weights if present. Overrides the constructor
            option for this call; optimizer state is always recreated.
        test_data : tuple of numpy.ndarray, optional
            ``(X_test, event_test, time_test)`` for test-loss history.
        test_losses : sequence of survcraft.loss_modules.BaseSurvivalLoss, optional
            Test objectives; defaults to the training loss. Recorded by class
            name, so duplicate classes overwrite earlier entries.

        Returns
        -------
        None
            Updates ``model_`` and, if requested, ``train_history_``.

        Raises
        ------
        ValueError
            If times are negative, module construction fails, or prediction
            divergence violates the policy.
        AssertionError
            If sample counts do not match or event dtype is not boolean.
        FailedConvergence
            If every training batch in an epoch is skipped.

        Notes
        -----
        Uses the algorithm and batching behavior in :class:`SurvivalPredictor`.
        This lower-level entry point does not perform the full structured-target
        validation of :meth:`fit`. Move losses with buffers to the model device
        before training. Non-finite validation losses are not filtered.
        """
        if not hasattr(self, "model_") or not warm_start:
            self._init_model(X, event, time)
            self.train_history_ = []

        data_device = self._get_device() if self.preload_data else 'cpu'
        dataset = torch.utils.data.TensorDataset(
            *self._validate_dataset(X, event, time, device=data_device)
        )
        if self.early_stopping:
            train_idx, val_idx = train_test_split(
                list(range(len(dataset))), test_size=self.validation_ratio
            )
            train_dataset = torch.utils.data.Subset(dataset, train_idx)
            val_dataset = torch.utils.data.Subset(dataset, val_idx)
        else:
            train_dataset = dataset
            val_dataset = None

        # Reshuffle event-free batches so the same samples are not skipped each epoch.
        train_dl = self._make_data_loader(train_dataset, shuffle=True)
        min_val_model_state = None
        if self.early_stopping:
            val_dl = self._make_data_loader(val_dataset, shuffle=False)
            min_val_loss = None

        # if test_data > 0 and test_data < 1: # could implement splitting
        if test_data is not None:
            assert len(test_data) == 3
            X_test, event_test, time_test = self._validate_dataset(*test_data)
            if test_losses is None:
                test_losses = [self.loss]
            median_time = torch.median(time_test[event_test])

        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        epoch = -1
        transfer_batches_to = self._get_device() if self._get_device() != data_device else None
        
        for epoch in range(self.epochs):
            # TRAINING
            self.model_.train()
            
            train_losses = []
            event_free_batches = 0
            nonfinite_loss_batches = 0
            for batch, (Xb, eb, tb) in enumerate(train_dl):
                if not eb.any():  # # no event
                    event_free_batches += 1
                    continue # ignore batch
                if transfer_batches_to is not None:
                    Xb = Xb.to(transfer_batches_to)
                    eb = eb.to(transfer_batches_to)
                    tb = tb.to(transfer_batches_to)

                loss = self.loss(self.model_, Xb, eb, tb)
                loss_num = loss.item()

                if not math.isfinite(loss_num):
                    nonfinite_loss_batches += 1
                    if self.verbose >= 1:
                        warnings.warn(
                            f"Epoch {epoch}, batch {batch} produced non-finite loss: {loss_num}"
                        )
                    continue  # ignore batch

                train_losses.append(loss_num)
                optimizer.zero_grad()
                loss.backward()
                if self.gradient_clipping:
                    torch.nn.utils.clip_grad_norm_(self.model_.parameters(), max_norm=1.0)
                optimizer.step()

                if self.verbose >= 3:
                    print(f"Epoch {epoch}, training batch {batch}, loss = {loss.item()}")
            if event_free_batches or nonfinite_loss_batches:
                reasons = []
                if event_free_batches:
                    reasons.append(f"batches without events: {event_free_batches}")
                if nonfinite_loss_batches:
                    reasons.append(f"batches with non-finite losses: {nonfinite_loss_batches}")
                msg = f"Skipped training batches in epoch {epoch} ({len(train_dl)} total): " + "; ".join(reasons)
                if not train_losses:
                    msg += "; no usable training batches remain"
                warnings.warn(msg)
                if not train_losses:
                    raise FailedConvergence(msg)

            train_losses = numpy.array(train_losses)

            # VALIDATION
            if val_dataset is not None:
                self.model_.eval()
                epoch_val_losses = torch.zeros(len(val_dl), dtype=torch.float32)
                with torch.no_grad():
                    for batch, (Xb, eb, tb) in enumerate(val_dl):
                        if transfer_batches_to is not None:
                            Xb = Xb.to(transfer_batches_to)
                            eb = eb.to(transfer_batches_to)
                            tb = tb.to(transfer_batches_to)
                        val_loss = self.loss(self.model_, Xb, eb, tb)
                        epoch_val_losses[batch] = val_loss
                    mean_val_loss = torch.mean(epoch_val_losses)

                    # early stopping code
                    if (
                        min_val_loss is None or mean_val_loss < min_val_loss
                    ):  # first epoch or new min loss
                        min_val_model_state = copy.deepcopy(self.model_.state_dict())
                        min_val_loss = mean_val_loss
                        min_val_epoch = epoch
                    else:  # no improvement
                        if epoch - min_val_epoch >= self.early_stopping_patience:
                            if self.verbose > 0:
                                print(f"Early stop at epoch {epoch}")
                            break

            if self.verbose >= 1:
                if self.verbose == 2 or epoch % int(max(1, self.epochs / 20)) == 0:
                    print(
                        f"Epoch {epoch}, training loss = {train_losses.mean().item()}",
                        end="",
                    )
                    if val_dataset is not None:
                        print(f", validation loss = {mean_val_loss.item()}", end="")
                    print()

            # test dataset
            test_losses_vals = {}
            if self.history and test_data:
                self.model_.eval()
                with torch.no_grad():
                    test_losses_vals = {
                        lf.__class__.__name__: lf(
                            self.model_, X_test, event_test, time_test
                        ).item()
                        for lf in test_losses
                    }

            if self.history:
                self.train_history_.append((train_losses, test_losses_vals))
        if self.early_stopping and min_val_model_state is not None:
            self.model_.load_state_dict(min_val_model_state)
        if epoch >= 0 and self.verbose >= 1:
            print(
                f"Final epoch {epoch}, training loss = {train_losses.mean()}",
                end="",
            )
            if val_dataset is not None:
                print(f", validation loss = {mean_val_loss}", end="")
            print()

    
    def get_distribution_params(self, X: numpy.ndarray) -> dict[str, Tensor]:
        r"""Inspect named distribution parameters as device-resident tensors.

        Parameters
        ----------
        X : numpy.ndarray, shape (n_samples, n_features)
            Numeric feature matrix.

        Returns
        -------
        dict of str to torch.Tensor
            Transformed parameter groups of shape ``(n_samples, group_width)``
            on ``device_``, computed without gradients.

        Raises
        ------
        RuntimeError
            If ``model_`` has not been initialized.

        Notes
        -----
        Sets evaluation mode. Unlike ``predict("params", X)``, this returns
        PyTorch tensors and bypasses the divergence checks in TorchModel.forward.

        See Also
        --------
        SurvivalEstimator.predict : NumPy parameter inspection.
        """
        if not hasattr(self, "model_"):
            raise RuntimeError("Model is not trained. Call 'train' before using this method.")
        
        self.model_.eval()
        X_tensor = self._tensor(X)
        with torch.no_grad():
            processed_params = self.model_.get_processed_params(X_tensor)
        return processed_params  # Return PyTorch tensors directly


@dataclass
class SurvivalSimulator(SurvivalEstimator):
    r"""Sample discretized right-censored outcomes from a fixed survival model.

    Parameters
    ----------
    input : BaseInputAdapter, optional
        Feature-map factory; defaults to a new LinearFunctionInputAdapter.
    survival : BaseSurvivalAdapter, optional
        Distribution factory; defaults to a new ExponentialSurvivalAdapter.
    loss : survcraft.loss_modules.BaseSurvivalLoss, optional
        Inherited option, unused by simulation; defaults to None.
    device : str or sequence of str, default='cpu'
        PyTorch device or nonempty list of candidate devices. One candidate
        is randomly chosen and cached on first use, not distributed across.
    verbose : int, default=0
        Training output level: 0 is quiet, 1 periodic, 2 per epoch, 3 per batch.
    check_divergence : {'raise', 'warn', 'no'}, default='raise'
        Policy for non-finite raw parameters and predictions.

    Attributes
    ----------
    model_ : TorchModel
        Assembled model, present after predictor initialization or the first
        simulator prediction. Its presence alone does not prove successful training.
    device_ : str
        Cached device, created on first device use.

    Notes
    -----
    Uses all inherited options and contracts in :class:`SurvivalEstimator`.
    The first prediction or simulation builds the model without outcomes.
    The inherited ``fit`` is a no-op. Later calls reuse the model, requiring
    the same feature width; changing adapter options does not rebuild it.

    For a zero-based grid :math:`t_0,\ldots,t_{m-1}`, sampling uses
    :math:`p_j = F(t_{j+1})-F(t_j)` for :math:`j<m-1`, and
    :math:`p_c = 1-\sum_j p_j`. Probabilities are clipped to [0, 1] and
    renormalized. An event is placed at its interval midpoint; the last
    category is censoring at :math:`t_{m-1}`. This is a grid approximation,
    not continuous inverse-CDF sampling. When :math:`F(0)>0`, that initial
    mass is included in censoring rather than the first interval.

    Examples
    --------
    >>> import numpy as np
    >>> simulator = SurvivalSimulator()
    >>> y = simulator.simulate(np.ones((3, 1)), [0.0, 1.0, 2.0], seed=0)
    >>> y.dtype.names
    ('event', 'time')
    """
    input: BaseInputAdapter = field(default_factory=LinearFunctionInputAdapter)
    survival: BaseSurvivalAdapter = field(default_factory=ExponentialSurvivalAdapter)
    # precision: torch.dtype = torch.float64

    def predict(
        self,
        mode: str,
        X: numpy.ndarray,
        times: Optional[ArrayLike] = None,
    ) -> ArrayPrediction:
        # in simulator
        # assert isinstance(mode, str)
        r"""Initialize the simulation model if needed, then predict.

        Parameters
        ----------
        mode : {'failure', 'survival', 'density', 'hazard', 'expected_time', 'median_time', 'risk', 'params'}
            Distribution quantity to evaluate; see :class:`SurvivalEstimator`.
        X : numpy.ndarray, shape (n_samples, n_features)
            Numeric features following the shared estimator contract.
        times : array-like, scalar or shape (n_times,), optional
            Shared nonnegative evaluation times; required for time-dependent
            modes and omitted for summaries and parameters.

        Returns
        -------
        numpy.ndarray or dict of str to numpy.ndarray
            A matrix ``(n_samples, n_times)`` for a time vector, a vector
            ``(n_samples,)`` for a scalar time or summary, or named parameter
            arrays ``(n_samples, group_width)`` for ``params``.

        Notes
        -----
        Uses :meth:`SurvivalEstimator.predict` and its mode validation.
        Initialization uses no observed outcomes and caches ``model_`` on first use.
        """
        if not hasattr(self, "model_"):
            self._init_model(X, None, None)

        return super().predict(mode, X, times=times)

    def simulate(
        self,
        X: numpy.ndarray,
        times: ArrayLike,
        seed: Optional[int] = None,
    ) -> numpy.ndarray:
        r"""Sample event times on a grid with censoring at its final point.

        Parameters
        ----------
        X : numpy.ndarray, shape (n_samples, n_features)
            Numeric features following :class:`SurvivalEstimator`.
        times : array-like, shape (n_times,)
            At least two finite nonnegative points starting at zero and strictly
            increasing, including after float32 conversion.
        seed : int, optional
            Seed for the local NumPy sampling generator; does not seed model
            initialization or random input weights.

        Returns
        -------
        numpy.ndarray, shape (n_samples,)
            Structured array with boolean event and float32 time fields. Events
            occur at interval midpoints; censoring occurs at the final grid point.

        Raises
        ------
        ValueError
            If the grid is invalid or model prediction fails.

        Notes
        -----
        Uses the probability formulas in :class:`SurvivalSimulator`. Negative
        CDF increments print diagnostics and are clipped before normalization;
        this can hide a nonmonotone failure curve. Initial mass F(0) is not
        sampled as an event.
        """
        times = _validate_time_grid(times, name="times")
        if not hasattr(self, "model_"):
            self._init_model(X, None, None)

        # compute probability of event for each time (interval)
        f = cast(numpy.ndarray, self.predict("failure", X, times))
        p = numpy.zeros_like(f)
        p[..., :-1] = f[..., 1:] - f[..., :-1]
        monot = f[..., 1:] >= f[..., :-1]
        if not monot.all():
            print(f"{self.survival} monotonicity failure(s): {1 - (monot).mean()}")
        # assert monot.all(), f'monotonicity failure {~(monot).sum()}'
        # FIXME we have precision problems here, slightly negative probs and sums that are not 1 due to rounding
        # assert p.min() > -3e-8, f'p.min() == {p.min()} at {numpy.argmin(p)} of {p.shape}'
        if p.min() < -3e-8:
            print(
                f"{self.survival} positivity failure: p.min() == {p.min()} at {numpy.argmin(p)} of {p.shape}"
            )
        p[:, -1] = 1.0 - p[:, :-1].sum(axis=1)
        # fix rounding errors
        p = p.clip(min=0, max=1.0)
        p = p / p.sum(axis=1).reshape(-1, 1)

        # FIXME in systematic simulations there is a monotonicity failure with the LogNormal survival...
        # for i in range(f.shape[1] - 1):
        #    fail = f[:, i] > f[:, i + 1]
        #    if torch.any(fail):
        #        print('monoton fail', numpy.min(p), self.model_, fail.sum().item(), i, f[fail, i - 1:i+3])

        # print('p', p.min(axis=1))

        # print('ciao', p.shape)
        # sample events
        rng = numpy.random.default_rng(seed=seed)
        event_time = numpy.zeros(p.shape[0])
        # get the center of each time interval, keep last value unchanged
        # Compute in float64 to avoid overflow when adding large finite points.
        midpoint_grid = times.astype(numpy.float64)
        mid_times = numpy.concatenate(((midpoint_grid[1:] + midpoint_grid[:-1]) / 2, midpoint_grid[-1:]))
        event_indicator = numpy.zeros(p.shape[0], dtype=bool)
        for i, pi in enumerate(p):
            # if any(pi < 0):
            #    print(i, pi.min(), pi.argmin(), pi)
            #    event_time[i] = -1
            # else:
            interval = rng.choice(len(pi), p=pi)
            event_time[i] = mid_times[interval]
            event_indicator[i] = interval < len(pi) - 1
        # event_time = numpy.array([
        #   times[rng.choice(p.shape[1], p=p[i])]
        #    for i in range(p.shape[0])
        # ])

        # build a structure array like in scikit-survival
        y = numpy.zeros(len(X), dtype=[("event", "?"), ("time", "f4")])
        y["event"], y["time"] = event_indicator, event_time

        return y

class FailedConvergence(Exception):
    r"""Signal that a training epoch has no usable batches.

    See Also
    --------
    SurvivalPredictor.train : Skipped-batch handling.
    """
    pass
