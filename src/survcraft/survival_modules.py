import torch
from warnings import warn

# for compatibility with older torch version (e.g. 1.4)
if not hasattr(torch, "pi"):
    import math

    torch.pi = math.pi
if not hasattr(torch, "square"):
    torch.square = lambda x: x * x


class SurvivalParameter(torch.nn.Module):
    """Describes a parameter in a survival distribution.

    Small Torch Module that takes the unbounded output of an input module and maps it to an appropriate value to be a parameter of a distribution.
    """

    # FIXME bound is currently never used, remove if not needed
    def __init__(self, name, n=1, func=None, bound=None):
        super().__init__()
        self.name = name
        self.n = n
        self.func = func
        if bound is None:
            self.bound = None
        else:
            self.register_buffer("bound", torch.tensor(bound))

    def forward(self, x):
        y = x
        # soft clamp if any bound
        if self.bound is not None:
            y = torch.tanh(y / self.bound) * self.bound
        # apply trasformation if any
        if self.func is not None:
            y = self.func(y)
        return y


class FreeParameter(SurvivalParameter):
    """Unbounded parameter."""
    pass


class PositiveParameter(SurvivalParameter):
    """Positive parameter.

    Converts an unbounded parameter to a positive value, for instance with the softplus function.
    """

    def __init__(self, name, n=1, func=torch.nn.Softplus(), bound=None):
        super().__init__(name, n, func=func, bound=bound)


class SoftmaxParameter(SurvivalParameter):
    def __init__(self, name, n, func=torch.nn.Softmax(dim=-1), bound=None):
        assert (
            n > 1
        ), f"softmax with {n} parameter(s), must be greater than 1!"  # FIXME make this a warning
        super().__init__(name, n, func=func, bound=bound)



# TODO explore if we can in someway automate batch handling, maybe with functorch or torch.vmap?
class BaseSurvivalModule(torch.nn.Module):
    """Base class for survival distribution modules.

    Derived classes implement the failure, survival, density, hazard, expected_time, median_time and risk functions.

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
        super().__init__()
        self.params = torch.nn.ModuleList(params)
        self.register_buffer("epsilon", torch.tensor(epsilon))
        self.enable_checks = enable_checks

    def get_param_number(self):
        return sum(p.n for p in self.params)

    def preprocess_params(self, raw_params):
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
        """Compute the survival distribution.

        Args:
            mode: string that selects the function to be computed
            raw_params: tensor
            times: vector of times
        Return:
            depends on mode: matrix for time functions or vector

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
        """Failure function.

        Implements the cumulative failure function, that is, the probability that an individual

        Args:
            params: dictionary of tensors of shape n
                parameters that describe the
            times:
                vector of times

        Return: tensor of shape (n, m)
            Failure probability matrix


        """
        return 1.0 - self.survival(params, times)

    #  raise NotImplementedError(f'failure() not implemented in {type(self)}')

    def survival(self, params, times):
        return 1.0 - self.failure(params, times)

    def hazard(self, params, times):
        survival = self.survival(params, times)
        density = self.density(params, times)
        return density / torch.clamp(survival, self.epsilon, 1.0)

    def density(self, params, times):
        raise NotImplementedError(
            f"density method not implemented in {type(self).__name__}"
        )

    def expected_time(self, params):
        raise NotImplementedError(
            f"expected_time method not implemented in {type(self).__name__}"
        )

    def median_time(self, params):
        raise NotImplementedError(
            f"median_time method not implemented in {type(self).__name__}"
        )

    def risk(self, params):
        raise NotImplementedError(
            f"risk method not implemented in {type(self).__name__}"
        )


"""### New survival models"""


class ExponentialSurvivalModule(BaseSurvivalModule):
    """Exponential distribution.

    The exponetial distribution has a single positive scale parameter $l$.

    Density:
    $ f(t) = l exp(-t l) $
    Hazard:
    $ h(t) = l $
    """

    name = "Exponential"

    def __init__(self, pp_func=PositiveParameter, *args, **kwargs):
        super().__init__([pp_func("scale")], *args, **kwargs)
        self.register_buffer("ln2", torch.log(torch.tensor(2.0)))

    def density(self, params, times):
        l = params["scale"]
        return l * torch.exp(-times * l)

    def survival(self, params, times):
        l = params["scale"]
        return torch.exp(-times * l)

    def hazard(self, params, times):
        l = params["scale"]
        return l.expand(l.shape[:-1] + (times.shape[0],)) if len(times.shape) > 0 else l

    def expected_time(self, params):
        l = params["scale"].squeeze(-1)
        return 1.0 / l

    def median_time(self, params):
        l = params["scale"].squeeze(-1)
        return self.ln2 / l

    def risk(self, params):
        l = params["scale"]
        return l.squeeze(dim=1)


class WeibullSurvivalModule(BaseSurvivalModule):
    """Weibull distribution."""

    name = "Weibull"

    def __init__(self, pp_func=PositiveParameter, *args, **kwargs):
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
        l, k = params["scale"], params["shape"]
        exponent = torch.clamp(-torch.pow(times / l, k), min=-30.0)
        return torch.exp(exponent)

    def density(self, params, times):
        l, k = params["scale"], params["shape"]
        return torch.where(
            times > 0.0,
            (k / l) * torch.pow(times / l, k - 1) * self.survival(params, times),
            self.zero,
        )
        scaled_times = times / l
        return (k / l) * torch.exp(
            torch.xlogy(k - 1, scaled_times) - torch.pow(scaled_times, k)
        )

    def hazard(self, params, times):
        l, k = params["scale"], params["shape"]
        return (k / l) * torch.pow(times / l, k - 1)

    def expected_time(self, params):
        l, k = params["scale"].squeeze(-1), params["shape"].squeeze(-1)
        return l * torch.exp(torch.lgamma(1 + 1 / k))

    def median_time(self, params):
        l, k = params["scale"].squeeze(-1), params["shape"].squeeze(-1)
        return l * torch.pow(self.ln2, 1 / k)


class LogNormalSurvivalModule(BaseSurvivalModule):
    name = "LogNormal"

    def __init__(self, pp_func=PositiveParameter, *args, **kwargs):
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
        mu, sigma = params["log_mean"], params["log_stddev"]
        return (torch.log(times + self.epsilon) - mu) / (sigma * self.sqrt2)

    def survival(self, params, times):
        return 0.5 * torch.erfc(self._normal(params, times))

    def failure(self, params, times):
        # Log-normal CDF
        return 0.5 * (1.0 + torch.erf(self._normal(params, times)))

    def density(self, params, times):
        # Log-normal PDF
        mu, sigma = params["log_mean"], params["log_stddev"]
        etimes = times + self.epsilon

        d = torch.exp(
            -0.5 * torch.square(torch.log(etimes) - mu) / torch.square(sigma)
        ) / (etimes * sigma * self.sqrt2pi)
        # for time zero the density is nan, by solving the limit it should be zero
        return torch.where(times > 0.0, d, self.zero)

    def expected_time(self, params):
        mu, sigma = params["log_mean"].squeeze(-1), params["log_stddev"].squeeze(-1)
        return torch.exp(mu + torch.square(sigma) / 2.0)

    def median_time(self, params):
        mu = params["log_mean"].squeeze(-1)
        return torch.exp(mu)



class LevySurvivalModule(BaseSurvivalModule):
    name = "Levy"

    def __init__(self, pp_func=PositiveParameter, *args, **kwargs):
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
        D, x0 = params["D"], params["x0"]
        return torch.erf(x0 / torch.sqrt(4 * D * times))

    def failure(self, params, times):
        # CDF
        D, x0 = params["D"], params["x0"]
        return torch.erfc(x0 / torch.sqrt(4 * D * times))

    def density(self, params, times):
        # PDF (It is a Levy distribution with mu=0 and c = x0^2 * 1/(2D))
        D, x0 = params["D"], params["x0"]
        numerator = D * x0 * torch.exp(-x0**2 / (4 * D * times))
        denominator = 2 * self.sqrt_pi * (D * times)**(3/2)
        return numerator / denominator

    def expected_time(self, params):
        # The mean is infinite
        raise NotImplementedError("Expected time not implemented for this module")
    
    def median_time(self, params):
        D, x0 = params["D"].squeeze(-1), params["x0"].squeeze(-1)
        return (x0.pow(2) / (4 * D * self.gamma_squared)).to(x0.device)


class InverseGaussianSurvivalModule(BaseSurvivalModule):
    name = "InverseGaussian"
    """
    In this implementation drift mu must be negative and x0>0.
    """

    def __init__(self, pp_func=PositiveParameter, *args, **kwargs):
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
        x0 = params['x0']
        mu = -torch.clamp(params['mu'], min=1e-2, max=5)
        return x0, mu
    
    def survival(self, params, times):
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
        return 1 - self.survival(params, times)

    def density(self, params, times):
        x0, mu = self._get_params(params)
        denominator = self.sqrt2pi * torch.sqrt(times**3)
        exponent = -((x0 + mu * times)**2) / (2 * times)
        #print(f"Density: denominator range: [{denominator.min().item():.4e}, {denominator.max().item():.4e}]")
        #print(f"Density: exponent range: [{exponent.min().item():.4f}, {exponent.max().item():.4f}]")

        result = (x0 / denominator) * torch.exp(exponent)
        #print(f"Density: result range: [{result.min().item():.4e}, {result.max().item():.4e}]")
        
        return result


    def expected_time(self, params):
        x0, mu = self._get_params(params)
        return (-x0 / mu).squeeze(dim=-1)

    def median_time(self, params):
        raise NotImplementedError("Median time not implemented for this module")
        


class StepExpSurvivalModule(BaseSurvivalModule):
    name = "step_density"

    def __init__(
        self, breaks: torch.Tensor, trainable_breaks: bool = False, *args, **kwargs
    ):
        # check that is a sorted vector
        if len(breaks.shape) != 1:
            raise ValueError(f"breaks must be a vector, instead found {breaks.shape=}")
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

        maxerr = torch.max(torch.abs(self._get_time_breaks()[0] - breaks))
        assert (
            maxerr < 1e-5
        ), f"{maxerr} -> {self._get_time_breaks()[1:]} != {breaks}"

    @staticmethod
    def _softplus_inverse(x, threshold=20):
        return torch.where(x < threshold, torch.log(torch.exp(x) - 1), x)

    def _get_time_breaks(self):
        interval_length = torch.nn.functional.softplus(self.interval_size)
        return torch.cat([self.zero, torch.cumsum(interval_length, 0)]), interval_length

    def failure(self, params, times):
        """Return survival probability at set times

        params: params
        times: array
          times at which calculate survival probabilities
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
    name = "proportional_hazard"

    def __init__(
        self,
        baseline: BaseSurvivalModule,
        baseline_params: torch.Tensor | None = None,
        *args,
        **kwargs,
    ):
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
        baseline_survival = self.baseline("survival", self.baseline_params, times)

        if self.training:
            # avoid null gradient with zero bases in the power operation
            baseline_survival = baseline_survival + self.epsilon
        return torch.pow(baseline_survival, params["relative_risk"])

    def hazard(self, params, times):
        baseline_hazard = self.baseline("hazard", self.baseline_params, times)
        return params["relative_risk"] * baseline_hazard

    def density(self, params, times):
        return self.hazard(params, times) * self.survival(params, times)

    def risk(self, params):
        return params["relative_risk"].squeeze(dim=1)


class AcceleratedFailureTimeSurvivalModule(BaseSurvivalModule):
    name = "accelerated_failure_time"

    def __init__(
        self,
        baseline: BaseSurvivalModule,
        baseline_params: torch.Tensor | None = None,
        *args,
        **kwargs,
    ):
        super().__init__([PositiveParameter("relative_risk")], *args, **kwargs)
        self.baseline = baseline

        self.baseline_params = torch.nn.Parameter(
            torch.empty(self.baseline.get_param_number())
        )

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
        if times is None:
            return self.baseline(mode, self.baseline_params)
        else:
            # flatten individual multiplied times, compute baseline, unflatten results
            return self.baseline(mode, self.baseline_params, times.view(-1)).reshape_as(times)

    def failure(self, params, times):
        rr = params["relative_risk"]
        return self._baseline("failure", rr * times)

    def survival(self, params, times):
        rr = params["relative_risk"]
        return self._baseline("survival", rr * times)

    def hazard(self, params, times):
        rr = params["relative_risk"]
        return rr * self._baseline("hazard", rr * times)

    def density(self, params, times):
        rr = params["relative_risk"]
        return rr * self._baseline("density", rr * times)

    def risk(self, params):
        rr = params["relative_risk"]
        return 1.0 / rr.squeeze(dim=1)

    def expected_time(self, params):
        rr = params["relative_risk"]
        return (self._baseline("expected_time") / rr).squeeze(dim=-1)

    def median_time(self, params):
        rr = params["relative_risk"]
        return (self._baseline("median_time") / rr).squeeze(dim=-1)


class MixtureSurvivalModule(BaseSurvivalModule):
    name = "mixture"

    def __init__(self, baselines, *args, **kwargs):
        # assign names to baselines
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
        return self._average_baselines("failure", params, times)

    def survival(self, params, times):
        return self._average_baselines("survival", params, times)

    def density(self, params, times):
        return self._average_baselines("density", params, times)

    def expected_time(self, params):
        return self._average_baselines("expected_time", params)

    # def median_time(self, params):
    # FIXME check if this is correct, I do not think that the mixture median is the average of the medians!
    #    return self._average_baselines("median_time", params)


import random  # FIXME use maybe torch and a generator for random


def fractal_noise_generator(steps, backbone=2):
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
    name = "FractalNoise"

    def __init__(
        self, max_time=1.0, backbone_length=1, seed=None, *args, **kwargs
    ):
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
        s = params["none"].shape
        if len(s) == 2:
            return x.expand(s[0], -1)
        else:
            return x

    def _interp(self, params, x, fp, left=None, right=None):
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
        return self._interp(params, times, self.failure_distr_, left=0.0, right=1.0)

    def density(self, params, times):
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
