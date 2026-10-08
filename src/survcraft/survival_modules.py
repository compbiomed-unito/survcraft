r"""Differentiable survival distributions and parameter transformations.

Notes
-----
A survival module maps unconstrained raw parameters to named distribution
parameters and evaluates a selected quantity. See :class:`BaseSurvivalModule`
for mode, shape, and device contracts. For nonnegative event time T,
:math:`F(t)=P(T\leq t)`, :math:`S(t)=1-F(t)`, :math:`f(t)=F'(t)`, and
:math:`h(t)=f(t)/S(t)`. Concrete docstrings describe numerical departures
from these identities. Risk is a ranking score, with larger values
indicating earlier events; it is implemented only by selected modules.
"""

import math
import torch
from warnings import warn

# for compatibility with older torch version (e.g. 1.4)
if not hasattr(torch, "pi"):
    import math

    torch.pi = math.pi
if not hasattr(torch, "square"):
    torch.square = lambda x: x * x


class SurvivalParameter(torch.nn.Module):
    r"""Transform a named group of unconstrained distribution parameters.

    Parameters
    ----------
    name : str
        Dictionary key for the transformed parameter group.
    n : int, default=1
        Number of raw values in the group; zero is used for parameter-free models.
    func : callable, optional
        Tensor transformation applied after optional soft bounding.
    bound : float or array-like, optional
        Soft bound b applied as ``b * tanh(x / b)`` before transformation;
        should be positive and broadcastable to x.

    Attributes
    ----------
    name : str
        Parameter key used by BaseSurvivalModule.preprocess_params.
    n : int
        Raw group width.
    func : callable or None
        Optional transformation.
    bound : torch.Tensor or None
        Registered soft-bound buffer when supplied.

    Notes
    -----
    For raw input x, the output is :math:`g(b\tanh(x/b))` when a bound
    and transformation g are supplied. Either operation may be omitted.
    Input and output keep the same shape.
    """

    # FIXME bound is currently never used, remove if not needed
    def __init__(self, name, n=1, func=None, bound=None):
        r"""Initialize parameter groups and registered distribution state.

        See Also
        --------
        SurvivalParameter : Constructor parameters, inherited options, and formulas.
        """
        super().__init__()
        self.name = name
        self.n = n
        self.func = func
        if bound is None:
            self.bound = None
        else:
            self.register_buffer("bound", torch.tensor(bound))

    def forward(self, x):
        r"""Apply optional soft bounding followed by the parameter transform.

        Parameters
        ----------
        x : torch.Tensor
            Unconstrained parameter values, typically shape (n_samples, n).

        Returns
        -------
        torch.Tensor
            Transformed values with the same shape as x.
        """
        y = x
        # soft clamp if any bound
        if self.bound is not None:
            y = torch.tanh(y / self.bound) * self.bound
        # apply trasformation if any
        if self.func is not None:
            y = self.func(y)
        return y


class FreeParameter(SurvivalParameter):
    r"""Keep raw parameter values unconstrained by default.

    Parameters
    ----------
    name : str
        Dictionary key for the transformed parameter group.
    n : int, default=1
        Number of raw values in the group; zero is used for parameter-free models.
    func : callable, optional
        Tensor transformation applied after optional soft bounding.
    bound : float or array-like, optional
        Soft bound b applied as ``b * tanh(x / b)`` before transformation;
        should be positive and broadcastable to x.

    Notes
    -----
    Inherits :class:`SurvivalParameter`, including optional func and bound.
    With defaults, the transformation is the identity.
    """
    pass


class PositiveParameter(SurvivalParameter):
    r"""Map raw values to positive parameters with softplus.

    Parameters
    ----------
    name : str
        Dictionary key for the transformed parameter group.
    n : int, default=1
        Number of raw values in the group; zero is used for parameter-free models.
    func : callable, optional
        Transformation after soft bounding; defaults to torch.nn.Softplus().
    bound : float or array-like, optional
        Soft bound b applied as ``b * tanh(x / b)`` before transformation;
        should be positive and broadcastable to x.

    Notes
    -----
    Default transformation: :math:`g(x)=\log(1+\exp(x))`, using
    PyTorch's stable softplus implementation. Custom functions are not
    checked for positivity. Inherits the :class:`SurvivalParameter` contract.
    """

    def __init__(self, name, n=1, func=torch.nn.Softplus(), bound=None):
        r"""Initialize parameter groups and registered distribution state.

        See Also
        --------
        PositiveParameter : Constructor parameters, inherited options, and formulas.
        """
        super().__init__(name, n, func=func, bound=bound)


class SoftmaxParameter(SurvivalParameter):
    r"""Map a raw vector to normalized probability weights.

    Parameters
    ----------
    name : str
        Dictionary key for the transformed parameter group.
    n : int
        Number of weights; must exceed one.
    func : callable, optional
        Transformation after soft bounding; defaults to softmax over the last axis.
    bound : float or array-like, optional
        Soft bound b applied as ``b * tanh(x / b)`` before transformation;
        should be positive and broadcastable to x.

    Raises
    ------
    AssertionError
        If n is not greater than one.

    Notes
    -----
    Default transformation: :math:`p_j=\exp(x_j)/\sum_k\exp(x_k)`.
    Inherits the :class:`SurvivalParameter` shape and bounding contract.
    """
    def __init__(self, name, n, func=torch.nn.Softmax(dim=-1), bound=None):
        r"""Initialize parameter groups and registered distribution state.

        See Also
        --------
        SoftmaxParameter : Constructor parameters, inherited options, and formulas.
        """
        assert (
            n > 1
        ), f"softmax with {n} parameter(s), must be greater than 1!"  # FIXME make this a warning
        super().__init__(name, n, func=func, bound=bound)



# TODO explore if we can in someway automate batch handling, maybe with functorch or torch.vmap?
class BaseSurvivalModule(torch.nn.Module):
    r"""Define parameter preprocessing and survival prediction contracts.

    Parameters
    ----------
    params : sequence of SurvivalParameter
        Ordered named groups consuming the final raw parameter axis.
    epsilon : float, default=1e-8
        Numerical floor used by selected formulas and the default hazard.
    enable_checks : bool, default=False
        Check output ranges before clamping; warn on small violations and
        raise RuntimeError on violations of magnitude at least 1e6.

    Attributes
    ----------
    params : torch.nn.ModuleList
        Registered ordered parameter transforms.
    epsilon : torch.Tensor
        Scalar numerical floor registered as a buffer.
    enable_checks : bool
        Whether to check value ranges.
    modes : set of str
        Accepted output modes; acceptance does not imply implementation.

    Notes
    -----
    Raw parameters normally have shape ``(n_samples, n_raw_params)``.
    ``preprocess_params`` splits the last axis in declared order and returns
    groups of shape ``(n_samples, group_width)``. A single raw vector is used
    internally by composite baselines; arbitrary batch axes are not uniformly
    supported by concrete modules.

    ``failure``, ``survival``, ``density``, and ``hazard`` require a shared
    scalar or one-dimensional time tensor, returning ``(n_samples,)`` or
    ``(n_samples, n_times)`` respectively. ``expected_time``, ``median_time``,
    and ``risk`` require times=None and return ``(n_samples,)``. ``params``
    returns the transformed parameter dictionary. All tensors should have
    compatible floating dtypes and devices; this module performs no automatic
    input conversion or device transfer. Use ``module.to(device)`` to move
    registered parameters and buffers.

    Subclasses must override at least one of failure or survival to avoid
    mutual recursion, and implement density or override hazard if needed.
    Unimplemented summary modes raise NotImplementedError. The default
    identities are :math:`F=1-S`, :math:`S=1-F`, and
    :math:`h=f/\operatorname{clamp}(S,\epsilon,1)`.
    ``forward`` clamps F and S to [0, 1], other quantities except risk to
    nonnegative values, and leaves NaNs unchanged. Range checks do not check
    finiteness or monotonicity; the estimator's TorchModel checks finiteness.

    See Also
    --------
    survcraft.adapters.BaseSurvivalAdapter : Construction from estimator options.
    survcraft.adapters.SurvivalEstimator : Array prediction interface.
    """

    modes = {
        "failure",
        "survival",
        "density",
        "hazard",
        "expected_time",
        "median_time",
        "risk",
        "params",
    }

    def __init__(self, params, epsilon: float = 1e-8, enable_checks: bool = False):
        r"""Initialize parameter groups and registered distribution state.

        See Also
        --------
        BaseSurvivalModule : Constructor parameters, inherited options, and formulas.
        """
        super().__init__()
        self.params = torch.nn.ModuleList(params)
        self.register_buffer("epsilon", torch.tensor(epsilon))
        self.enable_checks = enable_checks

    def get_param_number(self):
        r"""Return the total number of unconstrained parameter values.

        Returns
        -------
        int
            Sum of the declared parameter group widths.
        """
        return sum(p.n for p in self.params)

    def preprocess_params(self, raw_params):
        r"""Split raw parameters into named transformed groups.

        Parameters
        ----------
        raw_params : torch.Tensor, shape (..., n_raw_params)
            Unconstrained output values with the required final-axis width.

        Returns
        -------
        dict of str to torch.Tensor
            Named groups with shape ``(..., group_width)``.

        Raises
        ------
        ValueError
            If the final axis does not match get_param_number().
        """
        expected_params = self.get_param_number()
        if expected_params != raw_params.shape[-1]:
            raise ValueError(
                f"expected {expected_params} params in {type(self)}, found {raw_params.shape[-1]}"
            )
        return {
            m.name: m(p)
            for m, p in zip(
                self.params, torch.split(raw_params, [p.n for p in self.params], dim=-1)
            )
        }

    def forward(self, mode: str, raw_params, times=None):
        r"""Transform raw parameters and evaluate one distribution quantity.

        Parameters
        ----------
        mode : str
            Mode from the class contract, including params for parameter inspection.
        raw_params : torch.Tensor, shape (n_samples, n_raw_params)
            Unconstrained distribution parameters.
        times : torch.Tensor, scalar or shape (n_times,), optional
            Shared evaluation times, required for time-dependent modes.

        Returns
        -------
        torch.Tensor or dict of str to torch.Tensor
            Shapes and clamping follow the :class:`BaseSurvivalModule` contract.

        Raises
        ------
        TypeError
            If mode is not a string.
        ValueError
            If mode, parameter width, or returned shape is invalid.
        AssertionError
            If required times are absent, times have more than one dimension,
            or summary modes receive times.
        NotImplementedError
            If the requested method is unavailable.
        RuntimeError
            If range checks are enabled and a severe range violation occurs.

        Notes
        -----
        The params mode returns before value and output-shape checks. Concrete
        methods receive transformed parameters and a one-dimensional time vector.
        """
        # check mode
        if not isinstance(mode, str):
            raise TypeError(
                f"survival output mode must be a string, got a {type(mode)} instead"
            )
        if mode not in self.modes:
            modes_str = '"' + '", "'.join(self.modes) + '"'
            raise ValueError(
                f'unknown survival output mode "{mode}", must be one of {modes_str}'
            )

        if not hasattr(self, mode):
            raise NotImplementedError(
                f"Survival model {type(self).__name__} does not implement mode {mode}"
            )

        params = self.preprocess_params(raw_params)
        if mode == "params":
            return params

        if mode in {"expected_time", "median_time", "risk"}:
            assert times is None
            ret = getattr(self, mode)(params)
            expected_shape = raw_params.shape[:-1]
        else:
            assert times is not None
            assert (
                len(times.shape) <= 1
            ), f"times must be a scalar or a vector, has shape {times.shape} instead"

            # if times is scalar, transform to vector
            if times.shape == ():
                times_ = times.reshape(1)
            else:
                times_ = times
            assert len(times_.shape) == 1

            ret = getattr(self, mode)(params, times_)

            # if times is scalar remove its dimension
            if times.shape == ():  # scalar time
                assert ret.shape[1] == 1
                ret = ret[:, 0]

            expected_shape = raw_params.shape[:-1] + times.shape

        if ret.shape != expected_shape:
            raise ValueError(
                f'Expected {expected_shape} for {mode} with {"no" if times is None else times.shape} times in {self.name}, got: {ret.shape}'
            )

        if ret.numel() == 0:
            # skip value checks when ret is an empty tensor
            return ret

        if self.enable_checks:
            # check for negative values
            ret_min = ret.min().item()
            if mode != "risk" and ret_min < 0:
                msg = f"Found {(ret < 0).sum().item()} negative values in {mode} survival output of {self.__class__.__name__} with minimum value {ret_min}"
                if ret_min > -1e6:
                    warn(msg)
                else:
                    raise RuntimeError(msg)

            # check for values above 1
            ret_max = ret.max().item()
            if mode in {"failure", "survival"} and ret_max > 1:
                msg = f"Found {(ret > 1).sum().item()} values greater than 1.0 in {mode} survival output  of {self.__class__.__name__} with maximum value {ret_max} (exceeding 1.0 by {ret_max - 1.0})"
                if ret_max - 1.0 < 1e6:
                    warn(msg)
                else:
                    raise RuntimeError(msg)
        
        # clamping results to correct mistakes due to numerical instability
        if mode in {"failure", "survival"}:
            ret = ret.clamp(min=0.0, max=1.0)
        elif mode != "risk":
            ret = ret.clamp(min=0.0)

        return ret

    # basic implementations from the definition (one of failure or survival must be reimplemented to avoid circularity)
    def failure(self, params, times):
        r"""Return cumulative failure probabilities F(t).

        Parameters
        ----------
        params : dict of str to torch.Tensor
            Transformed groups from preprocess_params, normally with shape
            (n_samples, group_width); see BaseSurvivalModule.
        times : torch.Tensor, shape (n_times,)
            Shared nonnegative evaluation times on the module device.

        Returns
        -------
        torch.Tensor, shape (n_samples, n_times)
            Values before forward applies output clamping.

        Notes
        -----
        Default implementation is 1 - survival; override at least one of the pair.
        """
        return 1.0 - self.survival(params, times)

    #  raise NotImplementedError(f'failure() not implemented in {type(self)}')

    def survival(self, params, times):
        r"""Return survival probabilities S(t).

        Parameters
        ----------
        params : dict of str to torch.Tensor
            Transformed groups from preprocess_params, normally with shape
            (n_samples, group_width); see BaseSurvivalModule.
        times : torch.Tensor, shape (n_times,)
            Shared nonnegative evaluation times on the module device.

        Returns
        -------
        torch.Tensor, shape (n_samples, n_times)
            Values before forward applies output clamping.

        Notes
        -----
        Default implementation is 1 - failure; override at least one of the pair.
        """
        return 1.0 - self.failure(params, times)

    def hazard(self, params, times):
        r"""Return the instantaneous hazard h(t).

        Parameters
        ----------
        params : dict of str to torch.Tensor
            Transformed groups from preprocess_params, normally with shape
            (n_samples, group_width); see BaseSurvivalModule.
        times : torch.Tensor, shape (n_times,)
            Shared nonnegative evaluation times on the module device.

        Returns
        -------
        torch.Tensor, shape (n_samples, n_times)
            Values before forward applies output clamping.

        Notes
        -----
        Default implementation is density / clamp(survival, epsilon, 1).
        """
        survival = self.survival(params, times)
        density = self.density(params, times)
        return density / torch.clamp(survival, self.epsilon, 1.0)

    def density(self, params, times):
        r"""Return the event-time probability density f(t).

        Parameters
        ----------
        params : dict of str to torch.Tensor
            Transformed groups from preprocess_params, normally with shape
            (n_samples, group_width); see BaseSurvivalModule.
        times : torch.Tensor, shape (n_times,)
            Shared nonnegative evaluation times on the module device.

        Returns
        -------
        torch.Tensor, shape (n_samples, n_times)
            Values before forward applies output clamping.

        Raises
        ------
        NotImplementedError
            If the subclass does not implement this quantity.

        Notes
        -----
        Subclasses must supply a density implementation when this mode is used.
        """
        raise NotImplementedError(
            f"density method not implemented in {type(self).__name__}"
        )

    def expected_time(self, params):
        r"""Return the expected event time.

        Parameters
        ----------
        params : dict of str to torch.Tensor
            Transformed groups from preprocess_params, normally with shape
            (n_samples, group_width); see BaseSurvivalModule.

        Returns
        -------
        torch.Tensor, shape (n_samples,)
            One summary value per sample.

        Raises
        ------
        NotImplementedError
            If the subclass does not implement this quantity.

        Notes
        -----
        The base implementation raises NotImplementedError.
        """
        raise NotImplementedError(
            f"expected_time method not implemented in {type(self).__name__}"
        )

    def median_time(self, params):
        r"""Return the median event time.

        Parameters
        ----------
        params : dict of str to torch.Tensor
            Transformed groups from preprocess_params, normally with shape
            (n_samples, group_width); see BaseSurvivalModule.

        Returns
        -------
        torch.Tensor, shape (n_samples,)
            One summary value per sample.

        Raises
        ------
        NotImplementedError
            If the subclass does not implement this quantity.

        Notes
        -----
        The base implementation raises NotImplementedError.
        """
        raise NotImplementedError(
            f"median_time method not implemented in {type(self).__name__}"
        )

    def risk(self, params):
        r"""Return a score whose larger values indicate earlier events.

        Parameters
        ----------
        params : dict of str to torch.Tensor
            Transformed groups from preprocess_params, normally with shape
            (n_samples, group_width); see BaseSurvivalModule.

        Returns
        -------
        torch.Tensor, shape (n_samples,)
            One summary value per sample.

        Raises
        ------
        NotImplementedError
            If the subclass does not implement this quantity.

        Notes
        -----
        The base implementation raises NotImplementedError; risk is not a probability.
        """
        raise NotImplementedError(
            f"risk method not implemented in {type(self).__name__}"
        )


"""### New survival models"""


class ExponentialSurvivalModule(BaseSurvivalModule):
    r"""Model an exponential distribution with a positive rate.

    Parameters
    ----------
    pp_func : callable, default=PositiveParameter
        Constructor called with each positive parameter name. The default
        maps unconstrained values using softplus.
    *args : tuple
        Positional BaseSurvivalModule options after params: epsilon, enable_checks.
    **kwargs : dict
        Keyword BaseSurvivalModule options: epsilon, enable_checks.

    Notes
    -----
    The single parameter named ``scale`` is a rate :math:`\lambda`,
    not a time scale. The implemented formulas are

    .. math::

        S(t)=e^{-\lambda t},\quad f(t)=\lambda e^{-\lambda t},
        \quad h(t)=\lambda,

        E[T]=1/\lambda,\quad \operatorname{median}(T)=\log(2)/\lambda.

    Risk is the rate itself. Failure uses 1 - S.

    Uses the shapes, modes, and device contract in :class:`BaseSurvivalModule`.
    Inherited epsilon and enable_checks options are forwarded to that class.
    """

    name = "Exponential"

    def __init__(self, pp_func=PositiveParameter, *args, **kwargs):
        r"""Initialize parameter groups and registered distribution state.

        See Also
        --------
        ExponentialSurvivalModule : Constructor parameters, inherited options, and formulas.
        """
        super().__init__([pp_func("scale")], *args, **kwargs)
        self.register_buffer("ln2", torch.log(torch.tensor(2.0)))

    def density(self, params, times):
        r"""Return the event-time probability density f(t).

        Parameters
        ----------
        params : dict of str to torch.Tensor
            Transformed groups from preprocess_params, normally with shape
            (n_samples, group_width); see BaseSurvivalModule.
        times : torch.Tensor, shape (n_times,)
            Shared nonnegative evaluation times on the module device.

        Returns
        -------
        torch.Tensor, shape (n_samples, n_times)
            Values from the class formula, before forward output clamping.

        See Also
        --------
        ExponentialSurvivalModule : Parameterization and implemented formulas.
        BaseSurvivalModule : Shared tensor and device contract.
        """
        l = params["scale"]
        return l * torch.exp(-times * l)

    def survival(self, params, times):
        r"""Return survival probabilities S(t).

        Parameters
        ----------
        params : dict of str to torch.Tensor
            Transformed groups from preprocess_params, normally with shape
            (n_samples, group_width); see BaseSurvivalModule.
        times : torch.Tensor, shape (n_times,)
            Shared nonnegative evaluation times on the module device.

        Returns
        -------
        torch.Tensor, shape (n_samples, n_times)
            Values from the class formula, before forward output clamping.

        See Also
        --------
        ExponentialSurvivalModule : Parameterization and implemented formulas.
        BaseSurvivalModule : Shared tensor and device contract.
        """
        l = params["scale"]
        return torch.exp(-times * l)

    def hazard(self, params, times):
        r"""Return the instantaneous hazard h(t).

        Parameters
        ----------
        params : dict of str to torch.Tensor
            Transformed groups from preprocess_params, normally with shape
            (n_samples, group_width); see BaseSurvivalModule.
        times : torch.Tensor, shape (n_times,)
            Shared nonnegative evaluation times on the module device.

        Returns
        -------
        torch.Tensor, shape (n_samples, n_times)
            Values from the class formula, before forward output clamping.

        See Also
        --------
        ExponentialSurvivalModule : Parameterization and implemented formulas.
        BaseSurvivalModule : Shared tensor and device contract.
        """
        l = params["scale"]
        return l.expand(l.shape[:-1] + (times.shape[0],)) if len(times.shape) > 0 else l

    def expected_time(self, params):
        r"""Return the expected event time.

        Parameters
        ----------
        params : dict of str to torch.Tensor
            Transformed groups from preprocess_params, normally with shape
            (n_samples, group_width); see BaseSurvivalModule.

        Returns
        -------
        torch.Tensor, shape (n_samples,)
            One value per sample.

        See Also
        --------
        ExponentialSurvivalModule : Parameterization and implemented formulas.
        BaseSurvivalModule : Shared tensor and device contract.
        """
        l = params["scale"].squeeze(-1)
        return 1.0 / l

    def median_time(self, params):
        r"""Return the median event time.

        Parameters
        ----------
        params : dict of str to torch.Tensor
            Transformed groups from preprocess_params, normally with shape
            (n_samples, group_width); see BaseSurvivalModule.

        Returns
        -------
        torch.Tensor, shape (n_samples,)
            One value per sample.

        See Also
        --------
        ExponentialSurvivalModule : Parameterization and implemented formulas.
        BaseSurvivalModule : Shared tensor and device contract.
        """
        l = params["scale"].squeeze(-1)
        return self.ln2 / l

    def risk(self, params):
        r"""Return a score whose larger values indicate earlier events.

        Parameters
        ----------
        params : dict of str to torch.Tensor
            Transformed groups from preprocess_params, normally with shape
            (n_samples, group_width); see BaseSurvivalModule.

        Returns
        -------
        torch.Tensor, shape (n_samples,)
            One value per sample.

        See Also
        --------
        ExponentialSurvivalModule : Parameterization and implemented formulas.
        BaseSurvivalModule : Shared tensor and device contract.
        """
        l = params["scale"]
        return l.squeeze(dim=1)


class WeibullSurvivalModule(BaseSurvivalModule):
    r"""Model a Weibull distribution with positive time scale and shape.

    Parameters
    ----------
    pp_func : callable, default=PositiveParameter
        Constructor called with each positive parameter name. The default
        maps unconstrained values using softplus.
    *args : tuple
        Positional BaseSurvivalModule options after params: epsilon, enable_checks.
    **kwargs : dict
        Keyword BaseSurvivalModule options: epsilon, enable_checks.

    Notes
    -----
    Parameter order is ``scale`` :math:`\lambda`, then ``shape`` :math:`k`.
    For t > 0, the implementation uses

    .. math::

        H(t)=\exp\left(\min\left[k(\log t-\log\lambda),\log 20\right]\right),
        \quad S(t)=e^{-H(t)},

        f(t)=(k/\lambda)(t/\lambda)^{k-1}S(t),\quad
        h(t)=(k/\lambda)(t/\lambda)^{k-1},

        E[T]=\lambda\Gamma(1+1/k),\quad
        \operatorname{median}(T)=\lambda(\log 2)^{1/k}.

    This is equivalent to capping :math:`(t/\lambda)^k` at 20, but applies
    the cap in log space to avoid overflowing powers and NaN gradients.
    Survival is exactly one at t=0, with zero parameter gradients. Scales
    below the dtype's smallest positive normal value are floored to that
    value before taking logs, including softplus outputs that underflow to zero.

    Density is forced to zero at t=0, even when the mathematical limit is
    nonzero or infinite. Survival clipping imposes a positive tail floor;
    the density above is then not the derivative of the clipped CDF.
    Failure uses 1 - S; risk is unimplemented.

    Uses the shapes, modes, and device contract in :class:`BaseSurvivalModule`.
    Inherited epsilon and enable_checks options are forwarded to that class.
    """

    name = "Weibull"

    def __init__(self, pp_func=PositiveParameter, *args, **kwargs):
        r"""Initialize parameter groups and registered distribution state.

        See Also
        --------
        WeibullSurvivalModule : Constructor parameters, inherited options, and formulas.
        """
        super().__init__(
            [
                pp_func("scale"),
                pp_func("shape"),
            ],
            *args,
            **kwargs,
        )
        self.register_buffer("ln2", torch.log(torch.tensor(2.0)))
        self.register_buffer("zero", torch.tensor(0.0))

    def survival(self, params, times):
        r"""Return survival probabilities S(t).

        Parameters
        ----------
        params : dict of str to torch.Tensor
            Transformed groups from preprocess_params, normally with shape
            (n_samples, group_width); see BaseSurvivalModule.
        times : torch.Tensor, shape (n_times,)
            Shared nonnegative evaluation times on the module device.

        Returns
        -------
        torch.Tensor, shape (n_samples, n_times)
            Values from the class formula, before forward output clamping.

        See Also
        --------
        WeibullSurvivalModule : Parameterization and implemented formulas.
        BaseSurvivalModule : Shared tensor and device contract.
        """
        l, k = params["scale"], params["shape"]
        l = l.clamp_min(torch.finfo(l.dtype).tiny)
        zero_times = times == 0
        safe_times = torch.where(zero_times, torch.ones_like(times), times)
        # Clamp before exponentiation: clamping an overflowing power afterwards
        # leaves inf * 0 in its backward pass, even with finite predictions.
        log_power = k * (torch.log(safe_times) - torch.log(l))
        
        # Keep the inner exponential finite, with margin for rounding.
        max_log_power = math.log(torch.finfo(log_power.dtype).max) - 1.0
        power = torch.exp(log_power.clamp(max=max_log_power))
        #power = torch.exp(log_power.clamp(max=math.log(20.0)))
        return torch.where(zero_times, torch.ones_like(power), torch.exp(-power))

    def density(self, params, times):
        r"""Return the event-time probability density f(t).

        Parameters
        ----------
        params : dict of str to torch.Tensor
            Transformed groups from preprocess_params, normally with shape
            (n_samples, group_width); see BaseSurvivalModule.
        times : torch.Tensor, shape (n_times,)
            Shared nonnegative evaluation times on the module device.

        Returns
        -------
        torch.Tensor, shape (n_samples, n_times)
            Values from the class formula, before forward output clamping.

        See Also
        --------
        WeibullSurvivalModule : Parameterization and implemented formulas.
        BaseSurvivalModule : Shared tensor and device contract.
        """
        l, k = params["scale"], params["shape"]
        return torch.where(
            times > 0.0,
            self.hazard(params, times) * self.survival(params, times),
            self.zero,
        )
        scaled_times = times / l
        return (k / l) * torch.exp(
            torch.xlogy(k - 1, scaled_times) - torch.pow(scaled_times, k)
        )

    def hazard(self, params, times):
        r"""Return the instantaneous hazard h(t).

        Parameters
        ----------
        params : dict of str to torch.Tensor
            Transformed groups from preprocess_params, normally with shape
            (n_samples, group_width); see BaseSurvivalModule.
        times : torch.Tensor, shape (n_times,)
            Shared nonnegative evaluation times on the module device.

        Returns
        -------
        torch.Tensor, shape (n_samples, n_times)
            Values from the class formula, before forward output clamping.

        See Also
        --------
        WeibullSurvivalModule : Parameterization and implemented formulas.
        BaseSurvivalModule : Shared tensor and device contract.
        """
        l, k = params["scale"], params["shape"]
        return (k / l) * torch.pow(times / l, k - 1)

    def expected_time(self, params):
        r"""Return the expected event time.

        Parameters
        ----------
        params : dict of str to torch.Tensor
            Transformed groups from preprocess_params, normally with shape
            (n_samples, group_width); see BaseSurvivalModule.

        Returns
        -------
        torch.Tensor, shape (n_samples,)
            One value per sample.

        See Also
        --------
        WeibullSurvivalModule : Parameterization and implemented formulas.
        BaseSurvivalModule : Shared tensor and device contract.
        """
        l, k = params["scale"].squeeze(-1), params["shape"].squeeze(-1)
        return l * torch.exp(torch.lgamma(1 + 1 / k))

    def median_time(self, params):
        r"""Return the median event time.

        Parameters
        ----------
        params : dict of str to torch.Tensor
            Transformed groups from preprocess_params, normally with shape
            (n_samples, group_width); see BaseSurvivalModule.

        Returns
        -------
        torch.Tensor, shape (n_samples,)
            One value per sample.

        See Also
        --------
        WeibullSurvivalModule : Parameterization and implemented formulas.
        BaseSurvivalModule : Shared tensor and device contract.
        """
        l, k = params["scale"].squeeze(-1), params["shape"].squeeze(-1)
        return l * torch.pow(self.ln2, 1 / k)


class LogNormalSurvivalModule(BaseSurvivalModule):
    r"""Model a log-normal distribution with a free log mean.

    Parameters
    ----------
    pp_func : callable, default=PositiveParameter
        Constructor called with each positive parameter name. The default
        maps unconstrained values using softplus.
    *args : tuple
        Positional BaseSurvivalModule options after params: epsilon, enable_checks.
    **kwargs : dict
        Keyword BaseSurvivalModule options: epsilon, enable_checks.

    Notes
    -----
    Parameter order is ``log_mean`` :math:`\mu`, then positive
    ``log_stddev`` :math:`\sigma`. With :math:`u=t+\epsilon` and
    :math:`z=(\log u-\mu)/(\sigma\sqrt{2})`,

    .. math::

        S(t)=\tfrac12\operatorname{erfc}(z),\quad
        F(t)=\tfrac12(1+\operatorname{erf}(z)),

        f(t)=\frac{\exp[-(\log u-\mu)^2/(2\sigma^2)]}
                     {u\sigma\sqrt{2\pi}}\quad(t>0),

        E[T]=e^{\mu+\sigma^2/2},\quad \operatorname{median}(T)=e^\mu.

    Density is zero at t=0. The epsilon shift also applies to the CDF,
    so F(0) can be positive. Hazard uses the base numerical floor;
    risk is unimplemented.

    Uses the shapes, modes, and device contract in :class:`BaseSurvivalModule`.
    Inherited epsilon and enable_checks options are forwarded to that class.
    """
    name = "LogNormal"

    def __init__(self, pp_func=PositiveParameter, *args, **kwargs):
        r"""Initialize parameter groups and registered distribution state.

        See Also
        --------
        LogNormalSurvivalModule : Constructor parameters, inherited options, and formulas.
        """
        super().__init__(
            [
                FreeParameter("log_mean"),
                pp_func("log_stddev"),
            ],
            *args,
            **kwargs,
        )

        self.register_buffer("sqrt2", torch.sqrt(torch.tensor(2.0)))
        self.register_buffer("sqrt2pi", torch.sqrt(torch.tensor(2.0 * torch.pi)))
        self.register_buffer("zero", torch.tensor(0.0))

    def _normal(self, params, times):
        r"""Compute the epsilon-shifted standardized log time.

        Parameters
        ----------
        params : dict of str to torch.Tensor
            Transformed groups from preprocess_params, normally with shape
            (n_samples, group_width); see BaseSurvivalModule.
        times : torch.Tensor, shape (n_times,)
            Shared nonnegative evaluation times on the module device.

        Returns
        -------
        torch.Tensor, shape (n_samples, n_times)
            ``(log(times + epsilon) - log_mean) / (log_stddev * sqrt(2))``.
        """
        mu, sigma = params["log_mean"], params["log_stddev"]
        return (torch.log(times + self.epsilon) - mu) / (sigma * self.sqrt2)

    def survival(self, params, times):
        r"""Return survival probabilities S(t).

        Parameters
        ----------
        params : dict of str to torch.Tensor
            Transformed groups from preprocess_params, normally with shape
            (n_samples, group_width); see BaseSurvivalModule.
        times : torch.Tensor, shape (n_times,)
            Shared nonnegative evaluation times on the module device.

        Returns
        -------
        torch.Tensor, shape (n_samples, n_times)
            Values from the class formula, before forward output clamping.

        See Also
        --------
        LogNormalSurvivalModule : Parameterization and implemented formulas.
        BaseSurvivalModule : Shared tensor and device contract.
        """
        return 0.5 * torch.erfc(self._normal(params, times))

    def failure(self, params, times):
        # Log-normal CDF
        r"""Return cumulative failure probabilities F(t).

        Parameters
        ----------
        params : dict of str to torch.Tensor
            Transformed groups from preprocess_params, normally with shape
            (n_samples, group_width); see BaseSurvivalModule.
        times : torch.Tensor, shape (n_times,)
            Shared nonnegative evaluation times on the module device.

        Returns
        -------
        torch.Tensor, shape (n_samples, n_times)
            Values from the class formula, before forward output clamping.

        See Also
        --------
        LogNormalSurvivalModule : Parameterization and implemented formulas.
        BaseSurvivalModule : Shared tensor and device contract.
        """
        return 0.5 * (1.0 + torch.erf(self._normal(params, times)))

    def density(self, params, times):
        # Log-normal PDF
        r"""Return the event-time probability density f(t).

        Parameters
        ----------
        params : dict of str to torch.Tensor
            Transformed groups from preprocess_params, normally with shape
            (n_samples, group_width); see BaseSurvivalModule.
        times : torch.Tensor, shape (n_times,)
            Shared nonnegative evaluation times on the module device.

        Returns
        -------
        torch.Tensor, shape (n_samples, n_times)
            Values from the class formula, before forward output clamping.

        See Also
        --------
        LogNormalSurvivalModule : Parameterization and implemented formulas.
        BaseSurvivalModule : Shared tensor and device contract.
        """
        mu, sigma = params["log_mean"], params["log_stddev"]
        etimes = times + self.epsilon

        d = torch.exp(
            -0.5 * torch.square(torch.log(etimes) - mu) / torch.square(sigma)
        ) / (etimes * sigma * self.sqrt2pi)
        # for time zero the density is nan, by solving the limit it should be zero
        return torch.where(times > 0.0, d, self.zero)

    def expected_time(self, params):
        r"""Return the expected event time.

        Parameters
        ----------
        params : dict of str to torch.Tensor
            Transformed groups from preprocess_params, normally with shape
            (n_samples, group_width); see BaseSurvivalModule.

        Returns
        -------
        torch.Tensor, shape (n_samples,)
            One value per sample.

        See Also
        --------
        LogNormalSurvivalModule : Parameterization and implemented formulas.
        BaseSurvivalModule : Shared tensor and device contract.
        """
        mu, sigma = params["log_mean"].squeeze(-1), params["log_stddev"].squeeze(-1)
        return torch.exp(mu + torch.square(sigma) / 2.0)

    def median_time(self, params):
        r"""Return the median event time.

        Parameters
        ----------
        params : dict of str to torch.Tensor
            Transformed groups from preprocess_params, normally with shape
            (n_samples, group_width); see BaseSurvivalModule.

        Returns
        -------
        torch.Tensor, shape (n_samples,)
            One value per sample.

        See Also
        --------
        LogNormalSurvivalModule : Parameterization and implemented formulas.
        BaseSurvivalModule : Shared tensor and device contract.
        """
        mu = params["log_mean"].squeeze(-1)
        return torch.exp(mu)



class LevySurvivalModule(BaseSurvivalModule):
    r"""Model a zero-location Lévy first-passage distribution.

    Parameters
    ----------
    pp_func : callable, default=PositiveParameter
        Constructor called with each positive parameter name. The default
        maps unconstrained values using softplus.
    *args : tuple
        Positional BaseSurvivalModule options after params: epsilon, enable_checks.
    **kwargs : dict
        Keyword BaseSurvivalModule options: epsilon, enable_checks.

    Notes
    -----
    Parameter order is positive ``D`` (diffusion coefficient), then
    positive ``x0`` (distance). For t > 0,

    .. math::

        S(t)=\operatorname{erf}\left(\frac{x_0}{\sqrt{4Dt}}\right),\quad
        F(t)=\operatorname{erfc}\left(\frac{x_0}{\sqrt{4Dt}}\right),

        f(t)=\frac{Dx_0 e^{-x_0^2/(4Dt)}}{2\sqrt{\pi}(Dt)^{3/2}},\quad
        \operatorname{median}(T)=\frac{x_0^2}{4D[\operatorname{erfc}^{-1}(1/2)]^2}.

    The mean is infinite, so expected_time raises NotImplementedError rather
    than returning infinity. Density at t=0 is numerically undefined in
    this implementation. Hazard uses the base floor; risk is unimplemented.

    Uses the shapes, modes, and device contract in :class:`BaseSurvivalModule`.
    Inherited epsilon and enable_checks options are forwarded to that class.
    """
    name = "Levy"

    def __init__(self, pp_func=PositiveParameter, *args, **kwargs):
        r"""Initialize parameter groups and registered distribution state.

        See Also
        --------
        LevySurvivalModule : Constructor parameters, inherited options, and formulas.
        """
        super().__init__(
            [
                pp_func("D"),  # Diffusion coefficient
                pp_func("x0"),  # Distance parameter
            ],
            *args,
            **kwargs
        )
        self.register_buffer("sqrt_pi", torch.sqrt(torch.tensor(torch.pi)))
        self.register_buffer("gamma_squared", torch.tensor(0.4769362762044699**2))  # (erfcinv(1/2))^2

    
    def survival(self, params, times):
        r"""Return survival probabilities S(t).

        Parameters
        ----------
        params : dict of str to torch.Tensor
            Transformed groups from preprocess_params, normally with shape
            (n_samples, group_width); see BaseSurvivalModule.
        times : torch.Tensor, shape (n_times,)
            Shared nonnegative evaluation times on the module device.

        Returns
        -------
        torch.Tensor, shape (n_samples, n_times)
            Values from the class formula, before forward output clamping.

        See Also
        --------
        LevySurvivalModule : Parameterization and implemented formulas.
        BaseSurvivalModule : Shared tensor and device contract.
        """
        D, x0 = params["D"], params["x0"]
        return torch.erf(x0 / torch.sqrt(4 * D * times))

    def failure(self, params, times):
        # CDF
        r"""Return cumulative failure probabilities F(t).

        Parameters
        ----------
        params : dict of str to torch.Tensor
            Transformed groups from preprocess_params, normally with shape
            (n_samples, group_width); see BaseSurvivalModule.
        times : torch.Tensor, shape (n_times,)
            Shared nonnegative evaluation times on the module device.

        Returns
        -------
        torch.Tensor, shape (n_samples, n_times)
            Values from the class formula, before forward output clamping.

        See Also
        --------
        LevySurvivalModule : Parameterization and implemented formulas.
        BaseSurvivalModule : Shared tensor and device contract.
        """
        D, x0 = params["D"], params["x0"]
        return torch.erfc(x0 / torch.sqrt(4 * D * times))

    def density(self, params, times):
        # PDF (It is a Levy distribution with mu=0 and c = x0^2 * 1/(2D))
        r"""Return the event-time probability density f(t).

        Parameters
        ----------
        params : dict of str to torch.Tensor
            Transformed groups from preprocess_params, normally with shape
            (n_samples, group_width); see BaseSurvivalModule.
        times : torch.Tensor, shape (n_times,)
            Shared nonnegative evaluation times on the module device.

        Returns
        -------
        torch.Tensor, shape (n_samples, n_times)
            Values from the class formula, before forward output clamping.

        See Also
        --------
        LevySurvivalModule : Parameterization and implemented formulas.
        BaseSurvivalModule : Shared tensor and device contract.
        """
        D, x0 = params["D"], params["x0"]
        numerator = D * x0 * torch.exp(-x0**2 / (4 * D * times))
        denominator = 2 * self.sqrt_pi * (D * times)**(3/2)
        return numerator / denominator

    def expected_time(self, params):
        # The mean is infinite
        r"""Return the expected event time.

        Parameters
        ----------
        params : dict of str to torch.Tensor
            Transformed groups from preprocess_params, normally with shape
            (n_samples, group_width); see BaseSurvivalModule.

        Raises
        ------
        NotImplementedError
            This distribution does not provide this summary.

        See Also
        --------
        LevySurvivalModule : Parameterization and implemented formulas.
        BaseSurvivalModule : Shared tensor and device contract.
        """
        raise NotImplementedError("Expected time not implemented for this module")
    
    def median_time(self, params):
        r"""Return the median event time.

        Parameters
        ----------
        params : dict of str to torch.Tensor
            Transformed groups from preprocess_params, normally with shape
            (n_samples, group_width); see BaseSurvivalModule.

        Returns
        -------
        torch.Tensor, shape (n_samples,)
            One value per sample.

        See Also
        --------
        LevySurvivalModule : Parameterization and implemented formulas.
        BaseSurvivalModule : Shared tensor and device contract.
        """
        D, x0 = params["D"].squeeze(-1), params["x0"].squeeze(-1)
        return (x0.pow(2) / (4 * D * self.gamma_squared)).to(x0.device)


class InverseGaussianSurvivalModule(BaseSurvivalModule):
    r"""Model a unit-diffusion first-passage time with negative drift.

    Parameters
    ----------
    pp_func : callable, default=PositiveParameter
        Constructor called with each positive parameter name. The default
        maps unconstrained values using softplus.
    *args : tuple
        Positional BaseSurvivalModule options after params: epsilon, enable_checks.
    **kwargs : dict
        Keyword BaseSurvivalModule options: epsilon, enable_checks.

    Notes
    -----
    Raw parameter order is positive ``mu`` (drift magnitude), then positive
    ``x0`` (distance). The effective drift is
    :math:`v=-\operatorname{clamp}(\mu,0.01,5)`. For t > 0, define
    :math:`a_\pm=(vt\pm x_0)/\sqrt{t}` and standard normal CDF :math:`\Phi`.
    The implementation evaluates

    .. math::

        S(t)=\operatorname{clamp}\left[
          \Phi(a_+) - e^{\min(-2x_0v,50)}\Phi(a_-),0,1\right],

        F(t)=1-S(t),\quad
        f(t)=\frac{x_0}{\sqrt{2\pi t^3}}
             e^{-(x_0+vt)^2/(2t)},\quad E[T]=-x_0/v.

    Clamping changes the ideal inverse-Gaussian tail, and the product in S
    can lose precision. Parameter inspection reports the positive ``mu``
    before drift clipping and sign conversion. Density is numerically
    undefined at t=0. Median and risk are unimplemented; hazard uses the
    base numerical floor.

    Uses the shapes, modes, and device contract in :class:`BaseSurvivalModule`.
    Inherited epsilon and enable_checks options are forwarded to that class.
    """
    name = "InverseGaussian"
    """
    In this implementation drift mu must be negative and x0>0.
    """

    def __init__(self, pp_func=PositiveParameter, *args, **kwargs):
        r"""Initialize parameter groups and registered distribution state.

        See Also
        --------
        InverseGaussianSurvivalModule : Constructor parameters, inherited options, and formulas.
        """
        super().__init__(
            [
                pp_func("mu"),  # Drift coefficient
                pp_func("x0"),  # Distance parameter
            ],
            *args,
            **kwargs
        )
        self.register_buffer("sqrt_pi", torch.sqrt(torch.tensor(torch.pi)))
        self.register_buffer("sqrt2pi", torch.sqrt(torch.tensor(2.0 * torch.pi)))

    def _get_params(self, params):
        r"""Convert positive drift magnitude to the clipped negative drift.

        Parameters
        ----------
        params : dict of str to torch.Tensor
            Transformed groups from preprocess_params, normally with shape
            (n_samples, group_width); see BaseSurvivalModule.

        Returns
        -------
        tuple of torch.Tensor
            Distance x0 and effective drift -clamp(mu, 0.01, 5).
        """
        x0 = params['x0']
        mu = -torch.clamp(params['mu'], min=1e-2, max=5)
        return x0, mu
    
    def survival(self, params, times):
        r"""Return survival probabilities S(t).

        Parameters
        ----------
        params : dict of str to torch.Tensor
            Transformed groups from preprocess_params, normally with shape
            (n_samples, group_width); see BaseSurvivalModule.
        times : torch.Tensor, shape (n_times,)
            Shared nonnegative evaluation times on the module device.

        Returns
        -------
        torch.Tensor, shape (n_samples, n_times)
            Values from the class formula, before forward output clamping.

        See Also
        --------
        InverseGaussianSurvivalModule : Parameterization and implemented formulas.
        BaseSurvivalModule : Shared tensor and device contract.
        """
        x0, mu = self._get_params(params)
        
        # Add diagnostic prints
        #print(f"x0 range: [{x0.min().item():.4f}, {x0.max().item():.4f}]")
        #print(f"mu range: [{mu.min().item():.4f}, {mu.max().item():.4f}]")
        #print(f"times range: [{times.min().item():.4f}, {times.max().item():.4f}]")
        
        # Compute the arguments for the normal CDF
        arg1 = (mu * times + x0) / torch.sqrt(times)
        arg2 = (mu * times - x0) / torch.sqrt(times)
        
        # Add more diagnostic prints
        #print(f"arg1 range: [{arg1.min().item():.4f}, {arg1.max().item():.4f}]")
        #print(f"arg2 range: [{arg2.min().item():.4f}, {arg2.max().item():.4f}]")
        
        # Use torch.special.ndtr for numerical stability
        term1 = torch.special.ndtr(arg1)
        term2 = torch.exp(torch.clamp(-2 * x0 * mu, max=50)) * torch.special.ndtr(arg2)
        
        result = torch.clamp(term1 - term2, min=0, max=1)
        
        # Final diagnostic print
        #print(f"survival result range: [{result.min().item():.4f}, {result.max().item():.4f}]")
        
        return result
    
    #def survival(self, params, times):
    #    x0, mu = params['x0'], -params['mu']
    #    # Standard Normal CDF
    #    Phi = torch.distributions.Normal(0, 1).cdf
    #    #print('mu: ', mu[:3])
    #    #print('x0: ', x0[:3])
    #    term1 = Phi((mu * times + x0) / torch.sqrt(times))
    #    term2 = torch.exp(-2 * x0 * mu) * Phi((mu * times - x0) / torch.sqrt(times))
    #    return term1 - term2

    def failure(self, params, times):
        # CDF
        r"""Return cumulative failure probabilities F(t).

        Parameters
        ----------
        params : dict of str to torch.Tensor
            Transformed groups from preprocess_params, normally with shape
            (n_samples, group_width); see BaseSurvivalModule.
        times : torch.Tensor, shape (n_times,)
            Shared nonnegative evaluation times on the module device.

        Returns
        -------
        torch.Tensor, shape (n_samples, n_times)
            Values from the class formula, before forward output clamping.

        See Also
        --------
        InverseGaussianSurvivalModule : Parameterization and implemented formulas.
        BaseSurvivalModule : Shared tensor and device contract.
        """
        return 1 - self.survival(params, times)

    def density(self, params, times):
        r"""Return the event-time probability density f(t).

        Parameters
        ----------
        params : dict of str to torch.Tensor
            Transformed groups from preprocess_params, normally with shape
            (n_samples, group_width); see BaseSurvivalModule.
        times : torch.Tensor, shape (n_times,)
            Shared nonnegative evaluation times on the module device.

        Returns
        -------
        torch.Tensor, shape (n_samples, n_times)
            Values from the class formula, before forward output clamping.

        See Also
        --------
        InverseGaussianSurvivalModule : Parameterization and implemented formulas.
        BaseSurvivalModule : Shared tensor and device contract.
        """
        x0, mu = self._get_params(params)
        denominator = self.sqrt2pi * torch.sqrt(times**3)
        exponent = -((x0 + mu * times)**2) / (2 * times)
        #print(f"Density: denominator range: [{denominator.min().item():.4e}, {denominator.max().item():.4e}]")
        #print(f"Density: exponent range: [{exponent.min().item():.4f}, {exponent.max().item():.4f}]")

        result = (x0 / denominator) * torch.exp(exponent)
        #print(f"Density: result range: [{result.min().item():.4e}, {result.max().item():.4e}]")
        
        return result


    def expected_time(self, params):
        r"""Return the expected event time.

        Parameters
        ----------
        params : dict of str to torch.Tensor
            Transformed groups from preprocess_params, normally with shape
            (n_samples, group_width); see BaseSurvivalModule.

        Returns
        -------
        torch.Tensor, shape (n_samples,)
            One value per sample.

        See Also
        --------
        InverseGaussianSurvivalModule : Parameterization and implemented formulas.
        BaseSurvivalModule : Shared tensor and device contract.
        """
        x0, mu = self._get_params(params)
        return (-x0 / mu).squeeze(dim=-1)

    def median_time(self, params):
        r"""Return the median event time.

        Parameters
        ----------
        params : dict of str to torch.Tensor
            Transformed groups from preprocess_params, normally with shape
            (n_samples, group_width); see BaseSurvivalModule.

        Raises
        ------
        NotImplementedError
            This distribution does not provide this summary.

        See Also
        --------
        InverseGaussianSurvivalModule : Parameterization and implemented formulas.
        BaseSurvivalModule : Shared tensor and device contract.
        """
        raise NotImplementedError("Median time not implemented for this module")
        


class StepExpSurvivalModule(BaseSurvivalModule):
    r"""Combine uniform interval densities with a shifted exponential tail.

    Parameters
    ----------
    breaks : torch.Tensor, shape (n_breaks,)
        Finite strictly increasing floating grid starting at zero, with
        at least two points.
    trainable_breaks : bool, default=False
        Optimize interval lengths through a positive softplus transform.
    *args : tuple
        Positional BaseSurvivalModule options after params: epsilon, enable_checks.
    **kwargs : dict
        Keyword BaseSurvivalModule options: epsilon, enable_checks.

    Attributes
    ----------
    interval_size : torch.nn.Parameter
        Unconstrained interval lengths; gradients follow trainable_breaks.

    Raises
    ------
    ValueError
        If breaks are not a finite, increasing, zero-based vector of length >= 2.
    AssertionError
        If inverse-softplus reconstruction exceeds the relative tolerance.

    Notes
    -----
    There is one softmax group ``p`` with one mass per breakpoint.
    For breaks :math:`0=b_0<\cdots<b_{K-1}=B`, let
    :math:`\Delta_j=b_{j+1}-b_j`. Then

    .. math::

        F(t)=\sum_{j=0}^{K-2}p_j
          \operatorname{clamp}\left(\frac{t-b_j}{\Delta_j},0,1\right)
          +p_{K-1}\left[1-e^{-\max(t/B-1,0)}\right],

        f(t)=p_j/\Delta_j\quad(b_j\leq t<b_{j+1}),\qquad
        f(t)=\frac{p_{K-1}}{B}e^{-(t/B-1)}\quad(t\geq B),

        E[T]=\sum_{j=0}^{K-2}p_j(b_j+b_{j+1})/2+2Bp_{K-1}.

    Survival is 1 - F; hazard uses the base numerical floor. Median and risk
    are unimplemented. The finite intervals have constant density, not
    constant hazard. Negative evaluation times are outside the contract.

    Uses the shapes, modes, and device contract in :class:`BaseSurvivalModule`.
    Inherited epsilon and enable_checks options are forwarded to that class.
    """
    name = "step_density"

    def __init__(
        self, breaks: torch.Tensor, trainable_breaks: bool = False, *args, **kwargs
    ):
        # check that is a sorted vector
        r"""Initialize parameter groups and registered distribution state.

        See Also
        --------
        StepExpSurvivalModule : Constructor parameters, inherited options, and formulas.
        """
        if len(breaks.shape) != 1:
            raise ValueError(f"breaks must be a vector, instead found {breaks.shape=}")
        if breaks.numel() < 2:
            raise ValueError("breaks must contain at least two entries")
        if not torch.isfinite(breaks).all():
            raise ValueError("breaks must contain finite values")
        if breaks[0] != 0:
            raise ValueError(f"breaks must start with zero, instead found {breaks}")
        time_lengths = breaks[1:] - breaks[:-1]
        if any(time_lengths <= 0):
            raise ValueError(f"breaks must be strictly increasing, instead found {breaks}")

        super().__init__([SoftmaxParameter("p", n=len(breaks))], *args, **kwargs)

        self.register_buffer("zero", torch.zeros(1))
        interval_lengths_inv = self._softplus_inverse(time_lengths)

        self.interval_size = torch.nn.Parameter(
            interval_lengths_inv, requires_grad=trainable_breaks
        )

        # check relative error in break reconstruction, there can be some little deviation for large numbers
        maxrelerr = torch.max(torch.abs((self._get_time_breaks()[0][1:] - breaks[1:]) / breaks[1:]))
        assert (
            maxrelerr < 1e-4
        ), f"{maxrelerr=} -> {self._get_time_breaks()[0]} != {breaks}"

    @staticmethod
    def _softplus_inverse(x, threshold=20):
        r"""Invert softplus for positive interval lengths.

        Parameters
        ----------
        x : torch.Tensor
            Positive interval lengths.
        threshold : float, default=20
            Use the identity approximation at or above this value.

        Returns
        -------
        torch.Tensor
            ``log(expm1(x))`` below threshold and x otherwise.
        """
        return torch.where(x < threshold, torch.log(torch.expm1(x)), x)

    def _get_time_breaks(self):
        r"""Recover increasing breaks from positive interval lengths.

        Returns
        -------
        tuple of torch.Tensor
            The zero-based breakpoint vector and its softplus interval lengths.
        """
        interval_length = torch.nn.functional.softplus(self.interval_size)
        return torch.cat([self.zero, torch.cumsum(interval_length, 0)]), interval_length

    def failure(self, params, times):
        r"""Return cumulative failure probabilities F(t).

        Parameters
        ----------
        params : dict of str to torch.Tensor
            Transformed groups from preprocess_params, normally with shape
            (n_samples, group_width); see BaseSurvivalModule.
        times : torch.Tensor, shape (n_times,)
            Shared nonnegative evaluation times on the module device.

        Returns
        -------
        torch.Tensor, shape (n_samples, n_times)
            Values from the class formula, before forward output clamping.

        See Also
        --------
        StepExpSurvivalModule : Parameterization and implemented formulas.
        BaseSurvivalModule : Shared tensor and device contract.
        """
        params = params["p"]
        time_breaks0, interval_lengths = self._get_time_breaks()

        # compute the ratio of each finite interval that is smaller than times
        fin_weights = torch.clamp(
            (times.unsqueeze(-1) - time_breaks0[:-1]) / interval_lengths, 0.0, 1.0
        )
        # compute the ``ratio'' of the last interval (from the last break to +inf)
        inf_weigths = 1 - torch.exp(-torch.clamp(times / time_breaks0[-1] - 1, 0))
        # concatenate the weights for all intervals
        weights = torch.cat([fin_weights, inf_weigths.reshape(-1, 1)], dim=-1)
        
        return torch.matmul(params, weights.T)

    def density(self, params, times):
        # find the interval
        r"""Return the event-time probability density f(t).

        Parameters
        ----------
        params : dict of str to torch.Tensor
            Transformed groups from preprocess_params, normally with shape
            (n_samples, group_width); see BaseSurvivalModule.
        times : torch.Tensor, shape (n_times,)
            Shared nonnegative evaluation times on the module device.

        Returns
        -------
        torch.Tensor, shape (n_samples, n_times)
            Values from the class formula, before forward output clamping.

        See Also
        --------
        StepExpSurvivalModule : Parameterization and implemented formulas.
        BaseSurvivalModule : Shared tensor and device contract.
        """
        params = params["p"]
        time_breaks0, interval_length = self._get_time_breaks()
        # times x finite intervals: boolean if in that interval
        fin_intervals = (times.unsqueeze(-1) >= time_breaks0[:-1]) & (
            times.unsqueeze(-1) < time_breaks0[1:]
        )
        assert torch.all(torch.sum(fin_intervals, dim=-1) <= 1)
        interval_density = (
            params[..., :-1] / interval_length
        )  # (time_breaks0[1:] - time_breaks0[:-1])
        density = torch.matmul(
            interval_density, fin_intervals.T.to(interval_density.dtype)
        )
        # if no interval is true then we are in the infinite interval where density is not constant
        inf_intervals = torch.logical_not(torch.any(fin_intervals, dim=-1))
        density[..., inf_intervals] = (
            params[..., -1:]
            * torch.exp(-(times[inf_intervals] / time_breaks0[-1] - 1.0))
            / time_breaks0[-1]
        )
        return density

    def expected_time(self, params):
        r"""Return the expected event time.

        Parameters
        ----------
        params : dict of str to torch.Tensor
            Transformed groups from preprocess_params, normally with shape
            (n_samples, group_width); see BaseSurvivalModule.

        Returns
        -------
        torch.Tensor, shape (n_samples,)
            One value per sample.

        See Also
        --------
        StepExpSurvivalModule : Parameterization and implemented formulas.
        BaseSurvivalModule : Shared tensor and device contract.
        """
        params = params["p"]
        time_breaks0, _ = self._get_time_breaks()
        fin_interval_part = (
            torch.sum(params[..., :-1] * (time_breaks0[:-1] + time_breaks0[1:]), dim=-1)
            / 2.0
        )
        inf_interval_part = 2.0 * params[..., -1] * time_breaks0[-1]
        return fin_interval_part + inf_interval_part


if False:

    def median_time(self, raw_params):
        params = self.preprocess_parameters(raw_params)

        time_breaks0 = torch.cat([self.zero, self.time_breaks])
        cs = torch.cumsum(params, dim=-1)
        first_over = torch.argmax((cs > 0.5) + 0.0, dim=-1)
        print(
            params,
            time_breaks0,
            first_over,
        )
        # time_breaks0[]
        cs[first_over - 1]

        return cs
        # return cs, torch.argany(torch.clamp(cs - 0.5))


class ProportionalHazardSurvivalModule(BaseSurvivalModule):
    r"""Scale a shared baseline hazard by a positive relative risk.

    Parameters
    ----------
    baseline : BaseSurvivalModule
        Registered baseline distribution module.
    baseline_params : torch.Tensor, shape (n_baseline_params,), optional
        Raw shared baseline parameters. None creates a trainable Parameter
        initialized from a normal distribution with standard deviation 0.1.
        A supplied Parameter remains trainable; an ordinary tensor is kept
        as an unregistered fixed attribute and does not follow module.to().
    *args : tuple
        Positional BaseSurvivalModule options after params: epsilon, enable_checks.
    **kwargs : dict
        Keyword BaseSurvivalModule options: epsilon, enable_checks.

    Attributes
    ----------
    baseline : BaseSurvivalModule
        Registered shared baseline.
    baseline_params : torch.Tensor or torch.nn.Parameter
        Shared raw baseline values.

    Raises
    ------
    ValueError
        If baseline_params has an incorrect shape.

    Notes
    -----
    The only individual parameter is positive ``relative_risk`` r.
    With baseline functions S0 and h0,

    .. math::

        h(t)=r h_0(t),\quad S(t)=S_0(t)^r,\quad f(t)=h(t)S(t).

    In training mode, survival uses :math:`(S_0(t)+\epsilon)^r` to avoid
    zero-base power gradients. Evaluation uses the formula above. Failure
    is 1 - S and risk is r; expected time and median are unimplemented.

    Uses the shapes, modes, and device contract in :class:`BaseSurvivalModule`.
    Inherited epsilon and enable_checks options are forwarded to that class.
    """
    name = "proportional_hazard"

    def __init__(
        self,
        baseline: BaseSurvivalModule,
        baseline_params: torch.Tensor | None = None,
        *args,
        **kwargs,
    ):
        r"""Initialize parameter groups and registered distribution state.

        See Also
        --------
        ProportionalHazardSurvivalModule : Constructor parameters, inherited options, and formulas.
        """
        super().__init__([PositiveParameter("relative_risk")], *args, **kwargs)
        self.baseline = baseline

        # if not given, create a vector of trainable parameters for the baseline
        param_num = self.baseline.get_param_number()
        if baseline_params is None:

            baseline_params = torch.nn.Parameter(torch.empty(param_num))
            torch.nn.init.normal_(baseline_params, std=0.1)
        elif baseline_params.shape != (param_num,):
            raise ValueError(
                f"bad baseline_params shape, should be a vector of length {param_num}, got {baseline_params.shape}"
            )
        self.baseline_params = baseline_params

    def survival(self, params, times):
        r"""Return survival probabilities S(t).

        Parameters
        ----------
        params : dict of str to torch.Tensor
            Transformed groups from preprocess_params, normally with shape
            (n_samples, group_width); see BaseSurvivalModule.
        times : torch.Tensor, shape (n_times,)
            Shared nonnegative evaluation times on the module device.

        Returns
        -------
        torch.Tensor, shape (n_samples, n_times)
            Values from the class formula, before forward output clamping.

        See Also
        --------
        ProportionalHazardSurvivalModule : Parameterization and implemented formulas.
        BaseSurvivalModule : Shared tensor and device contract.
        """
        baseline_survival = self.baseline("survival", self.baseline_params, times)

        if self.training:
            # avoid null gradient with zero bases in the power operation
            baseline_survival = baseline_survival + self.epsilon
        return torch.pow(baseline_survival, params["relative_risk"])

    def hazard(self, params, times):
        r"""Return the instantaneous hazard h(t).

        Parameters
        ----------
        params : dict of str to torch.Tensor
            Transformed groups from preprocess_params, normally with shape
            (n_samples, group_width); see BaseSurvivalModule.
        times : torch.Tensor, shape (n_times,)
            Shared nonnegative evaluation times on the module device.

        Returns
        -------
        torch.Tensor, shape (n_samples, n_times)
            Values from the class formula, before forward output clamping.

        See Also
        --------
        ProportionalHazardSurvivalModule : Parameterization and implemented formulas.
        BaseSurvivalModule : Shared tensor and device contract.
        """
        baseline_hazard = self.baseline("hazard", self.baseline_params, times)
        return params["relative_risk"] * baseline_hazard

    def density(self, params, times):
        r"""Return the event-time probability density f(t).

        Parameters
        ----------
        params : dict of str to torch.Tensor
            Transformed groups from preprocess_params, normally with shape
            (n_samples, group_width); see BaseSurvivalModule.
        times : torch.Tensor, shape (n_times,)
            Shared nonnegative evaluation times on the module device.

        Returns
        -------
        torch.Tensor, shape (n_samples, n_times)
            Values from the class formula, before forward output clamping.

        See Also
        --------
        ProportionalHazardSurvivalModule : Parameterization and implemented formulas.
        BaseSurvivalModule : Shared tensor and device contract.
        """
        return self.hazard(params, times) * self.survival(params, times)

    def risk(self, params):
        r"""Return a score whose larger values indicate earlier events.

        Parameters
        ----------
        params : dict of str to torch.Tensor
            Transformed groups from preprocess_params, normally with shape
            (n_samples, group_width); see BaseSurvivalModule.

        Returns
        -------
        torch.Tensor, shape (n_samples,)
            One value per sample.

        See Also
        --------
        ProportionalHazardSurvivalModule : Parameterization and implemented formulas.
        BaseSurvivalModule : Shared tensor and device contract.
        """
        return params["relative_risk"].squeeze(dim=1)


class AcceleratedFailureTimeSurvivalModule(BaseSurvivalModule):
    r"""Rescale a shared baseline time by a positive relative risk.

    Parameters
    ----------
    baseline : BaseSurvivalModule
        Registered baseline distribution module.
    baseline_params : torch.Tensor, shape (n_baseline_params,), optional
        Raw shared baseline parameters. None creates a trainable Parameter
        initialized from a normal distribution with standard deviation 0.1.
        A supplied Parameter remains trainable; an ordinary tensor is kept
        as an unregistered fixed attribute and does not follow module.to().
    *args : tuple
        Positional BaseSurvivalModule options after params: epsilon, enable_checks.
    **kwargs : dict
        Keyword BaseSurvivalModule options: epsilon, enable_checks.

    Attributes
    ----------
    baseline : BaseSurvivalModule
        Registered shared baseline.
    baseline_params : torch.Tensor or torch.nn.Parameter
        Shared raw baseline values.

    Raises
    ------
    ValueError
        If baseline_params has an incorrect shape.

    Notes
    -----
    The only individual parameter is positive ``relative_risk`` r.
    This implementation uses inverse time scale: larger r means earlier
    events. With baseline functions F0, S0, f0, h0,

    .. math::

        F(t)=F_0(rt),\quad S(t)=S_0(rt),\quad
        f(t)=r f_0(rt),\quad h(t)=r h_0(rt),

        E[T]=E[T_0]/r,\quad \operatorname{median}(T)=\operatorname{median}(T_0)/r.

    Risk is r. Summary modes depend on baseline support.

    Uses the shapes, modes, and device contract in :class:`BaseSurvivalModule`.
    Inherited epsilon and enable_checks options are forwarded to that class.
    """
    name = "accelerated_failure_time"

    def __init__(
        self,
        baseline: BaseSurvivalModule,
        baseline_params: torch.Tensor | None = None,
        *args,
        **kwargs,
    ):
        r"""Initialize parameter groups and registered distribution state.

        See Also
        --------
        AcceleratedFailureTimeSurvivalModule : Constructor parameters, inherited options, and formulas.
        """
        super().__init__([PositiveParameter("relative_risk")], *args, **kwargs)
        self.baseline = baseline

        # if not given, create a vector of trainable parameters for the baseline
        param_num = self.baseline.get_param_number()
        if baseline_params is None:
            baseline_params = torch.nn.Parameter(torch.empty(param_num))
            torch.nn.init.normal_(baseline_params, std=0.1)
        elif baseline_params.shape != (param_num,):
            raise ValueError(
                f"bad baseline_params shape, should be a vector of length {param_num}, got {baseline_params.shape}"
            )
        self.baseline_params = baseline_params

    def _baseline(self, mode, times=None):
        r"""Evaluate shared baseline quantities at flattened individual times.

        Parameters
        ----------
        mode : str
            Baseline prediction mode.
        times : torch.Tensor, optional
            Individual scaled time matrix or None for a summary.

        Returns
        -------
        torch.Tensor
            Baseline prediction reshaped to times.shape when times is supplied.
        """
        if times is None:
            return self.baseline(mode, self.baseline_params)
        else:
            # flatten individual multiplied times, compute baseline, unflatten results
            return self.baseline(mode, self.baseline_params, times.view(-1)).reshape_as(times)

    def failure(self, params, times):
        r"""Return cumulative failure probabilities F(t).

        Parameters
        ----------
        params : dict of str to torch.Tensor
            Transformed groups from preprocess_params, normally with shape
            (n_samples, group_width); see BaseSurvivalModule.
        times : torch.Tensor, shape (n_times,)
            Shared nonnegative evaluation times on the module device.

        Returns
        -------
        torch.Tensor, shape (n_samples, n_times)
            Values from the class formula, before forward output clamping.

        See Also
        --------
        AcceleratedFailureTimeSurvivalModule : Parameterization and implemented formulas.
        BaseSurvivalModule : Shared tensor and device contract.
        """
        rr = params["relative_risk"]
        return self._baseline("failure", rr * times)

    def survival(self, params, times):
        r"""Return survival probabilities S(t).

        Parameters
        ----------
        params : dict of str to torch.Tensor
            Transformed groups from preprocess_params, normally with shape
            (n_samples, group_width); see BaseSurvivalModule.
        times : torch.Tensor, shape (n_times,)
            Shared nonnegative evaluation times on the module device.

        Returns
        -------
        torch.Tensor, shape (n_samples, n_times)
            Values from the class formula, before forward output clamping.

        See Also
        --------
        AcceleratedFailureTimeSurvivalModule : Parameterization and implemented formulas.
        BaseSurvivalModule : Shared tensor and device contract.
        """
        rr = params["relative_risk"]
        return self._baseline("survival", rr * times)

    def hazard(self, params, times):
        r"""Return the instantaneous hazard h(t).

        Parameters
        ----------
        params : dict of str to torch.Tensor
            Transformed groups from preprocess_params, normally with shape
            (n_samples, group_width); see BaseSurvivalModule.
        times : torch.Tensor, shape (n_times,)
            Shared nonnegative evaluation times on the module device.

        Returns
        -------
        torch.Tensor, shape (n_samples, n_times)
            Values from the class formula, before forward output clamping.

        See Also
        --------
        AcceleratedFailureTimeSurvivalModule : Parameterization and implemented formulas.
        BaseSurvivalModule : Shared tensor and device contract.
        """
        rr = params["relative_risk"]
        return rr * self._baseline("hazard", rr * times)

    def density(self, params, times):
        r"""Return the event-time probability density f(t).

        Parameters
        ----------
        params : dict of str to torch.Tensor
            Transformed groups from preprocess_params, normally with shape
            (n_samples, group_width); see BaseSurvivalModule.
        times : torch.Tensor, shape (n_times,)
            Shared nonnegative evaluation times on the module device.

        Returns
        -------
        torch.Tensor, shape (n_samples, n_times)
            Values from the class formula, before forward output clamping.

        See Also
        --------
        AcceleratedFailureTimeSurvivalModule : Parameterization and implemented formulas.
        BaseSurvivalModule : Shared tensor and device contract.
        """
        rr = params["relative_risk"]
        return rr * self._baseline("density", rr * times)

    def risk(self, params):
        r"""Return a score whose larger values indicate earlier events.

        Parameters
        ----------
        params : dict of str to torch.Tensor
            Transformed groups from preprocess_params, normally with shape
            (n_samples, group_width); see BaseSurvivalModule.

        Returns
        -------
        torch.Tensor, shape (n_samples,)
            One value per sample.

        See Also
        --------
        AcceleratedFailureTimeSurvivalModule : Parameterization and implemented formulas.
        BaseSurvivalModule : Shared tensor and device contract.
        """
        rr = params["relative_risk"]
        return rr.squeeze(dim=1)

    def expected_time(self, params):
        r"""Return the expected event time.

        Parameters
        ----------
        params : dict of str to torch.Tensor
            Transformed groups from preprocess_params, normally with shape
            (n_samples, group_width); see BaseSurvivalModule.

        Returns
        -------
        torch.Tensor, shape (n_samples,)
            One value per sample.

        See Also
        --------
        AcceleratedFailureTimeSurvivalModule : Parameterization and implemented formulas.
        BaseSurvivalModule : Shared tensor and device contract.
        """
        rr = params["relative_risk"]
        return (self._baseline("expected_time") / rr).squeeze(dim=-1)

    def median_time(self, params):
        r"""Return the median event time.

        Parameters
        ----------
        params : dict of str to torch.Tensor
            Transformed groups from preprocess_params, normally with shape
            (n_samples, group_width); see BaseSurvivalModule.

        Returns
        -------
        torch.Tensor, shape (n_samples,)
            One value per sample.

        See Also
        --------
        AcceleratedFailureTimeSurvivalModule : Parameterization and implemented formulas.
        BaseSurvivalModule : Shared tensor and device contract.
        """
        rr = params["relative_risk"]
        return (self._baseline("median_time") / rr).squeeze(dim=-1)


class MixtureSurvivalModule(BaseSurvivalModule):
    r"""Mix component distributions with feature-dependent softmax weights.

    Parameters
    ----------
    baselines : sequence of BaseSurvivalModule
        At least two component distributions, registered in supplied order.
    *args : tuple
        Positional BaseSurvivalModule options after params: epsilon, enable_checks.
    **kwargs : dict
        Keyword BaseSurvivalModule options: epsilon, enable_checks.

    Attributes
    ----------
    baselines : torch.nn.ModuleDict
        Components keyed by class name followed by an index.

    Raises
    ------
    AssertionError
        If fewer than two components are supplied.

    Notes
    -----
    Raw parameters consist of the softmax ``weights`` logits followed by
    the raw parameters of each component in order. For weights wj,

    .. math::

        F(t)=\sum_j w_jF_j(t),\quad S(t)=\sum_j w_jS_j(t),\quad
        f(t)=\sum_j w_jf_j(t),\quad E[T]=\sum_j w_jE[T_j].

    The parameter dictionary uses keys ``weights`` and ``ClassName_#j``;
    component groups remain raw until each component preprocesses them.
    Hazard is the ratio of mixture density and floored mixture survival,
    not a weighted average of hazards. Expected time requires all component
    means; median and risk are unimplemented.

    Uses the shapes, modes, and device contract in :class:`BaseSurvivalModule`.
    Inherited epsilon and enable_checks options are forwarded to that class.
    """
    name = "mixture"

    def __init__(self, baselines, *args, **kwargs):
        # assign names to baselines
        r"""Initialize parameter groups and registered distribution state.

        See Also
        --------
        MixtureSurvivalModule : Constructor parameters, inherited options, and formulas.
        """
        baseline_names = [
            baseline.__class__.__name__ + f"_#{i}"
            for i, baseline in enumerate(baselines)
        ]

        super().__init__(
            [
                SoftmaxParameter("weights", n=len(baselines)),
            ]
            + [
                FreeParameter(name, n=baseline.get_param_number())
                for name, baseline in zip(baseline_names, baselines)
            ],
            *args,
            **kwargs,
        )
        self.baselines = torch.nn.ModuleDict(zip(baseline_names, baselines))

    def _average_baselines(self, mode, params, times=None):
        r"""Compute a probability-weighted sum of component predictions.

        Parameters
        ----------
        mode : str
            Component prediction mode.
        params : dict of str to torch.Tensor
            Mixture weights and raw component parameter groups.
        times : torch.Tensor, optional
            Shared evaluation time vector, omitted for summaries.

        Returns
        -------
        torch.Tensor
            Weighted prediction with the shared BaseSurvivalModule output shape.
        """
        baselines = torch.stack(
            [
                baseline(mode, params[name], times)
                for name, baseline in self.baselines.items()
            ],
            dim=0,
        )
        weights = torch.transpose(params["weights"], 0, -1)
        if times is not None:
            weights = torch.transpose(params["weights"], 0, -1).unsqueeze(-1)

        return torch.sum(weights * baselines, dim=0)

    def failure(self, params, times):
        r"""Return cumulative failure probabilities F(t).

        Parameters
        ----------
        params : dict of str to torch.Tensor
            Transformed groups from preprocess_params, normally with shape
            (n_samples, group_width); see BaseSurvivalModule.
        times : torch.Tensor, shape (n_times,)
            Shared nonnegative evaluation times on the module device.

        Returns
        -------
        torch.Tensor, shape (n_samples, n_times)
            Values from the class formula, before forward output clamping.

        See Also
        --------
        MixtureSurvivalModule : Parameterization and implemented formulas.
        BaseSurvivalModule : Shared tensor and device contract.
        """
        return self._average_baselines("failure", params, times)

    def survival(self, params, times):
        r"""Return survival probabilities S(t).

        Parameters
        ----------
        params : dict of str to torch.Tensor
            Transformed groups from preprocess_params, normally with shape
            (n_samples, group_width); see BaseSurvivalModule.
        times : torch.Tensor, shape (n_times,)
            Shared nonnegative evaluation times on the module device.

        Returns
        -------
        torch.Tensor, shape (n_samples, n_times)
            Values from the class formula, before forward output clamping.

        See Also
        --------
        MixtureSurvivalModule : Parameterization and implemented formulas.
        BaseSurvivalModule : Shared tensor and device contract.
        """
        return self._average_baselines("survival", params, times)

    def density(self, params, times):
        r"""Return the event-time probability density f(t).

        Parameters
        ----------
        params : dict of str to torch.Tensor
            Transformed groups from preprocess_params, normally with shape
            (n_samples, group_width); see BaseSurvivalModule.
        times : torch.Tensor, shape (n_times,)
            Shared nonnegative evaluation times on the module device.

        Returns
        -------
        torch.Tensor, shape (n_samples, n_times)
            Values from the class formula, before forward output clamping.

        See Also
        --------
        MixtureSurvivalModule : Parameterization and implemented formulas.
        BaseSurvivalModule : Shared tensor and device contract.
        """
        return self._average_baselines("density", params, times)

    def expected_time(self, params):
        r"""Return the expected event time.

        Parameters
        ----------
        params : dict of str to torch.Tensor
            Transformed groups from preprocess_params, normally with shape
            (n_samples, group_width); see BaseSurvivalModule.

        Returns
        -------
        torch.Tensor, shape (n_samples,)
            One value per sample.

        See Also
        --------
        MixtureSurvivalModule : Parameterization and implemented formulas.
        BaseSurvivalModule : Shared tensor and device contract.
        """
        return self._average_baselines("expected_time", params)

    # def median_time(self, params):
    # FIXME check if this is correct, I do not think that the mixture median is the average of the medians!
    #    return self._average_baselines("median_time", params)


import random  # FIXME use maybe torch and a generator for random


def fractal_noise_generator(steps, backbone=2):
    r"""Refine random backbone values by noisy midpoint interpolation.

    Parameters
    ----------
    steps : int
        Number of doubling refinements.
    backbone : int or sequence of float, default=2
        Number of random initial points or explicit backbone values.

    Returns
    -------
    list of float
        Refined values, with length ``(len(backbone) - 1) * 2**steps + 1``
        for a sequence backbone.

    Notes
    -----
    At refinement s, insert :math:`(b_i+b_{i+1})/2+2^{-s}(U-1/2)`
    with U uniform on [0, 1). Uses Python's global random generator.
    """
    try:
        b = [random.random() for _ in range(backbone)]
    except TypeError:
        b = backbone

    for s in range(steps):
        c = 1 / 2 ** (s)
        bb = []
        for i in range(2 * len(b) - 1):
            i2 = int(i / 2)
            bb.append(
                b[i2]
                if i % 2 == 0
                else (b[i2] + b[i2 + 1]) / 2 + c * (random.random() - 0.5)
            )
        b = bb
    return b


import numpy  # FIXME can we just use torch?


class FractalNoiseSurvivalModule(BaseSurvivalModule):
    r"""Interpolate a random failure curve and a separate density profile.

    Parameters
    ----------
    max_time : float, default=1.0
        End of the interpolation grid; should be positive.
    backbone_length : int, default=1
        Number of random backbone points before the terminal zero.
    seed : int, optional
        Seed for Python's global random generator during construction.
    *args : tuple
        Positional BaseSurvivalModule options after params: epsilon, enable_checks.
    **kwargs : dict
        Keyword BaseSurvivalModule options: epsilon, enable_checks.

    Attributes
    ----------
    times_ : numpy.ndarray
        Uniform grid from zero to max_time.
    density_distr_ : numpy.ndarray
        Nonnegative profile normalized by its sum, not its time integral.
    failure_distr_ : numpy.ndarray
        Normalized cumulative shifted density profile.

    Notes
    -----
    Six midpoint refinements produce noise values nj. For N grid points,
    :math:`q_j=|n_j|(1-j/(N-1))`, :math:`d_j=q_j/\sum_kq_k`, and
    :math:`F_j=\sum_{k<j}d_k`, normalized to F at the final point equal to 1.
    Failure and density are independently linearly interpolated from Fj
    and dj. Failure is zero left of the grid and one to its right; density
    is zero outside the grid. These interpolants do not satisfy f=F'.

    The raw group ``none`` has width zero. No individual parameters or
    summary modes are implemented. Interpolation uses NumPy on CPU and
    does not preserve autograd; accelerator evaluation is not supported
    reliably. Setting seed resets Python's global random state. This module
    is intended for simulation; its density is not a normalized continuous
    PDF.

    Uses the shapes, modes, and device contract in :class:`BaseSurvivalModule`.
    Inherited epsilon and enable_checks options are forwarded to that class.
    """
    name = "FractalNoise"

    def __init__(
        self, max_time=1.0, backbone_length=1, seed=None, *args, **kwargs
    ):
        r"""Initialize parameter groups and registered distribution state.

        See Also
        --------
        FractalNoiseSurvivalModule : Constructor parameters, inherited options, and formulas.
        """
        super().__init__([SurvivalParameter("none", n=0)], *args, **kwargs)

        if seed is not None:
            random.seed(seed)

        noise = numpy.array(
            fractal_noise_generator(
                6, [random.random() for _ in range(backbone_length)] + [0.0]
            )
        )
        # times =
        # pos_noise = numpy.abs(noise) * numpy.exp(-times * 1e3 / max_time)
        pos_noise = numpy.abs(noise) * numpy.linspace(1, 0, len(noise))

        density = pos_noise / pos_noise.sum()
        assert density[-1] == 0

        failure = numpy.roll(density, 1).cumsum()
        failure = failure / failure.max()
        assert (
            failure[0] == 0 and failure[-1] == 1
        ), f"bad failure {failure[0]} and {failure[-1]}"

        # survival = numpy.clip(1.0 - failure, a_min=1e-8, a_max=1)
        # survival = 1.0 - failure
        # assert survival[0] == 1 and survival[-1] == 0, f'bad survival {survival[0]} and {survival[-1]}'
        self.times_ = numpy.linspace(0, max_time, len(noise))
        self.density_distr_ = density
        self.failure_distr_ = failure

    @staticmethod
    def _expand(params, x):
        r"""Expand a shared profile to the requested sample count.

        Parameters
        ----------
        params : dict of str to torch.Tensor
            Empty none parameter group carrying sample dimensions.
        x : torch.Tensor
            Shared interpolated time profile.

        Returns
        -------
        torch.Tensor
            Sample-expanded view for batched parameters, otherwise x.
        """
        s = params["none"].shape
        if len(s) == 2:
            return x.expand(s[0], -1)
        else:
            return x

    def _interp(self, params, x, fp, left=None, right=None):
        r"""Interpolate a stored profile through NumPy on CPU.

        Parameters
        ----------
        params : dict of str to torch.Tensor
            Empty none group carrying sample dimensions.
        x : torch.Tensor
            Evaluation time vector, copied to CPU.
        fp : numpy.ndarray
            Stored profile values.
        left : float, optional
            Extrapolation value left of the grid.
        right : float, optional
            Extrapolation value right of the grid.

        Returns
        -------
        torch.Tensor
            Float32 CPU interpolation, expanded over samples when needed.

        Notes
        -----
        This operation does not preserve input gradients or device placement.
        """
        try:
            x = x.cpu()
        except AttributeError:
            pass

        return self._expand(
            params,
            torch.tensor(
                numpy.interp(
                    x, self.times_, fp, left=left, right=right,
                ),
                dtype=torch.float, device=x.device,
            ),
        )

    def failure(self, params, times):
        r"""Return cumulative failure probabilities F(t).

        Parameters
        ----------
        params : dict of str to torch.Tensor
            Transformed groups from preprocess_params, normally with shape
            (n_samples, group_width); see BaseSurvivalModule.
        times : torch.Tensor, shape (n_times,)
            Shared nonnegative evaluation times on the module device.

        Returns
        -------
        torch.Tensor, shape (n_samples, n_times)
            Values from the class formula, before forward output clamping.

        See Also
        --------
        FractalNoiseSurvivalModule : Parameterization and implemented formulas.
        BaseSurvivalModule : Shared tensor and device contract.
        """
        return self._interp(params, times, self.failure_distr_, left=0.0, right=1.0)

    def density(self, params, times):
        r"""Return the event-time probability density f(t).

        Parameters
        ----------
        params : dict of str to torch.Tensor
            Transformed groups from preprocess_params, normally with shape
            (n_samples, group_width); see BaseSurvivalModule.
        times : torch.Tensor, shape (n_times,)
            Shared nonnegative evaluation times on the module device.

        Returns
        -------
        torch.Tensor, shape (n_samples, n_times)
            Values from the class formula, before forward output clamping.

        See Also
        --------
        FractalNoiseSurvivalModule : Parameterization and implemented formulas.
        BaseSurvivalModule : Shared tensor and device contract.
        """
        return self._interp(params, times, self.density_distr_, left=0.0, right=0.0)


_BASE_SURVIVAL_MODULES = {
    m.__name__: m
    for m in [
        ExponentialSurvivalModule,
        WeibullSurvivalModule,
        LogNormalSurvivalModule,
        LevySurvivalModule,
        InverseGaussianSurvivalModule,
        StepExpSurvivalModule,
    ]
}
_COMPOSITE_SURVIVAL_MODULES = {
    m.__name__: m
    for m in [
        ProportionalHazardSurvivalModule,
        AcceleratedFailureTimeSurvivalModule,
        MixtureSurvivalModule,
    ]
}
