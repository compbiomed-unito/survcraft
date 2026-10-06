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


# Input modules
class BaseInputAdapter(BaseEstimator, ABC):
    @property
    @abstractmethod
    def module_class(self) -> type[Module]:
        """Module constructor, supplied by subclasses as a class attribute."""
        raise NotImplementedError

    def get_module(self, input_size: int, output_size: int) -> Module:
        return self.module_class(
            input_size=input_size, output_size=output_size, **self.get_params()
        )


@dataclass
class LinearFunctionInputAdapter(BaseInputAdapter):
    module_class = input_modules.LinearFunctionInputModule

    mode: Literal["identity", "random"] = "identity"
    multiplier: float = 1.0
    shift: float = 0.0
    use_first_n_feats: Optional[int] = None
    seed: Optional[int] = None


@dataclass
class FeedForwardNetAdapter(BaseInputAdapter):
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
    @property
    @abstractmethod
    def module_class(self) -> type[survival_modules.BaseSurvivalModule]:
        """Module constructor, supplied by subclasses as a class attribute."""
        raise NotImplementedError

    # functions to be applied to parameters for preprocessing before passing it to the torch module (for instance for arrays that needs to be converted to tensors or for derived modules that need to initialized their submodules)
    param_funcs: ClassVar[
        dict[str, Callable[[Any, Optional[numpy.ndarray], Optional[numpy.ndarray]], Any]]
    ] = {}

    def get_module(
        self, event: Optional[ArrayLike], time: Optional[ArrayLike]
    ) -> survival_modules.BaseSurvivalModule:
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
    module_class = survival_modules.ExponentialSurvivalModule


class WeibullSurvivalAdapter(BaseSurvivalAdapter):
    module_class = survival_modules.WeibullSurvivalModule


class LogNormalSurvivalAdapter(BaseSurvivalAdapter):
    module_class = survival_modules.LogNormalSurvivalModule

class LevySurvivalAdapter(BaseSurvivalAdapter):
    module_class = survival_modules.LevySurvivalModule
    
class InverseGaussianSurvivalAdapter(BaseSurvivalAdapter):
    module_class = survival_modules.InverseGaussianSurvivalModule


@dataclass
class FractalNoiseSurvivalAdapter(BaseSurvivalAdapter):
    max_time: float = 1.0
    backbone_length: int = 1
    seed: Optional[int] = None

    module_class = survival_modules.FractalNoiseSurvivalModule


@dataclass
class StepExpSurvivalAdapter(BaseSurvivalAdapter):
    breaks: Union[int, ArrayLike] = 10
    trainable_breaks: bool = False

    @staticmethod
    def preprocess_breaks(
        breaks: Union[int, ArrayLike],
        event: Optional[numpy.ndarray],
        time: Optional[numpy.ndarray],
    ) -> Tensor:
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
            time_breaks = numpy.asarray(breaks)
        if time_breaks.ndim != 1 or time_breaks.size < 2:
            raise ValueError("breaks must be a one-dimensional vector with at least two entries")
        if time_breaks.dtype.kind not in "iuf" or not numpy.isfinite(time_breaks).all():
            raise ValueError("breaks must contain finite numeric values")
        if time_breaks[0] != 0:
            raise ValueError("breaks must start with zero")
        if not (time_breaks[1:] > time_breaks[:-1]).all():
            raise ValueError("breaks must be strictly increasing")
        with numpy.errstate(over="ignore", under="ignore"):
            time_breaks = time_breaks.astype(numpy.float32)
        if not numpy.isfinite(time_breaks).all() or not (time_breaks[1:] > time_breaks[:-1]).all():
            raise ValueError("breaks must remain finite and strictly increasing in float32")
        return torch.tensor(time_breaks)
 

    module_class = survival_modules.StepExpSurvivalModule
    param_funcs = {
        "breaks": preprocess_breaks,
    }


@dataclass
class DerivedSurvivalAdapter(BaseSurvivalAdapter):
    # abstract class for common behaviour between PH and AFT adapters
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
    module_class = survival_modules.ProportionalHazardSurvivalModule

@dataclass
class AcceleratedFailureTimeSurvivalAdapter(DerivedSurvivalAdapter):
    module_class = survival_modules.AcceleratedFailureTimeSurvivalModule


@dataclass
class MixtureSurvivalAdapter(BaseSurvivalAdapter):
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
    return 'x'.join(map(str, x.shape))


check_divergence_values: tuple[CheckDivergence, ...] = "warn", "raise", "no"
class TorchModel(torch.nn.Module):
    """Simple torch module that applies the input and survival module together."""

    def __init__(
        self,
        input_module: Module,
        survival_module: survival_modules.BaseSurvivalModule,
        check_divergence: CheckDivergence = "raise",
    ) -> None:
        super().__init__()
        self.input_module = input_module
        self.survival_module = survival_module
        if check_divergence not in check_divergence_values:
            raise ValueError(f"unknown value {check_divergence} for `check_divergence`, must be one of {check_divergence_values}")
        self.check_divergence = check_divergence

    def get_raw_params(self, x: Tensor) -> Tensor:
        return self.input_module(x)

    def get_processed_params(self, x: Tensor) -> dict[str, Tensor]:
        return self.survival_module.preprocess_params(self.input_module(x))

    def _check_tensor(
        self,
        x: Tensor,
        name: str,
        times: Optional[Tensor] = None,
        context: Mapping[str, Any] = {},
    ) -> None:
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
    input: Optional[BaseInputAdapter] = None
    survival: Optional[BaseSurvivalAdapter] = None
    loss: Optional[loss_modules.BaseSurvivalLoss] = None

    device: Union[str, Sequence[str]] = "cpu"
    verbose: int = 0
    check_divergence: CheckDivergence = "raise"
    # precision: torch.dtype = torch.float32

    def _get_device(self) -> str:
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
        """Convert array to tensor"""
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
        return self

    def predict_survival(
        self, X: numpy.ndarray, times: Optional[ArrayLike] = None
    ) -> numpy.ndarray:
        return cast(numpy.ndarray, self.predict(mode="survival", X=X, times=times))

    @add_exception_context
    def predict(
        self,
        mode: str,
        X: numpy.ndarray,
        times: Optional[ArrayLike] = None,
    ) -> ArrayPrediction:
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
        from .plotting import plot_outputs

        plot_outputs(self, X, max_time=max_time)



@dataclass
class SurvivalPredictor(SurvivalEstimator):
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

    def fit(
        self,
        X: numpy.ndarray,
        y: numpy.ndarray,
        X_test: Optional[numpy.ndarray] = None,
        y_test: Optional[numpy.ndarray] = None,
    ) -> SurvivalPredictor:
        train_kwargs = {
            'X': X,
            'event': y[y.dtype.names[0]].copy(),
            'time': y[y.dtype.names[1]].copy(),
            'warm_start': self.warm_start,
        }
        
        if X_test is not None and y_test is not None:
            train_kwargs['test_data'] = (
                X_test,
                y_test[y.dtype.names[0]].copy(),
                y_test[y.dtype.names[1]].copy(),
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

        train_dl = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            num_workers=self.data_loader_num_workers,
            shuffle=True,  # some survival losses cannot be computed if there are no events in the batch, so we reshuffle to avoid losing the same batches everytime
        )
        min_val_model_state = None
        if self.early_stopping:
            val_dl = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                num_workers=self.data_loader_num_workers,
            )
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
        """Get the distribution parameters for given input data."""
        if not hasattr(self, "model_"):
            raise RuntimeError("Model is not trained. Call 'train' before using this method.")
        
        self.model_.eval()
        X_tensor = self._tensor(X)
        with torch.no_grad():
            processed_params = self.model_.get_processed_params(X_tensor)
        return processed_params  # Return PyTorch tensors directly


@dataclass
class SurvivalSimulator(SurvivalEstimator):
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
        if not hasattr(self, "model_"):
            self._init_model(X, None, None)

        return super().predict(mode, X, times=times)

    def simulate(
        self,
        X: numpy.ndarray,
        times: Optional[numpy.ndarray] = None,
        seed: Optional[int] = None,
    ) -> numpy.ndarray:
        assert times is not None, "implement auto times?"
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
        mid_times = numpy.concatenate(((times[1:] + times[:-1]) / 2, times[-1:]))
        for i, pi in enumerate(p):
            # if any(pi < 0):
            #    print(i, pi.min(), pi.argmin(), pi)
            #    event_time[i] = -1
            # else:
            event_time[i] = mid_times[rng.choice(len(pi), p=pi)]
        # event_time = numpy.array([
        #   times[rng.choice(p.shape[1], p=p[i])]
        #    for i in range(p.shape[0])
        # ])
        event_indicator = event_time < times[-1]

        # build a structure array like in scikit-survival
        y = numpy.zeros(len(X), dtype=[("event", "?"), ("time", "f4")])
        y["event"], y["time"] = event_indicator, event_time

        return y

class FailedConvergence(Exception):
    pass
