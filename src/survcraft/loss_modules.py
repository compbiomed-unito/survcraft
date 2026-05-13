import torch
from torch import Tensor
from typing import Callable
import abc

__all__ = [
    "BrierLoss",
    "SquaredLoss",
    "PartialLikelihoodLoss",
    "FullLikelihoodLoss",
    "BrierBatchTimesLoss",
    "BCEBatchTimesLoss",
]


def check_survival_outcomes(events, times):
    if events.dtype is not torch.bool:
        raise TypeError(
            f"Expected events to be a boolean vector, got {events.dtype}"
        )
    if len(events.shape) != 1:
        raise ValueError(
            f"Expected event to be a boolean vector, got {events.shape}"
        )

    if events.shape != times.shape:
        raise ValueError(
            f"Expected event and time to be equal length vectors, got {events.shape=} != {times.shape=}"
        )


class BaseSurvivalLoss(torch.nn.Module, abc.ABC):
    @abc.abstractmethod
    def forward(self, model: torch.nn.Module, x: Tensor, event: torch.BoolTensor, time: Tensor):
        """
        Compute a survival loss given a model, input features and the censored event times.

        This function takes the model as argument since most survival losses need to compute one or more survival functions at specific times. So, intead of relying on the user to compute the model predictions correctly for each loss, by passing the model each loss can decide what output it needs from the model.
        """

    def __add__(self, other):
        return LinearCombinationLoss([self, other], torch.tensor([1.0, 1.0]))

    def __mul__(self, num: float):
        return LinearCombinationLoss([self], torch.tensor([num]))

    def __rmul__(self, num: float):
        return self * num


class LinearCombinationLoss(BaseSurvivalLoss):
    def __init__(self, losses, coeffs):
        super().__init__()
        assert (len(losses),) == coeffs.shape

        self.losses = torch.nn.ModuleList()
        coeff_acc = []
        for l, c in zip(losses, coeffs):
            ll, cc = (
                (l.losses, l.coeffs)
                if isinstance(l, LinearCombinationLoss)
                else ([l], torch.tensor([1.0]))
            )
            self.losses.extend(ll)
            coeff_acc.append(c * cc)
        self.register_buffer("coeffs", torch.cat(coeff_acc))

    def forward(self, model, x, event, time):
        acc = torch.zeros_like(self.coeffs)
        for i, l in enumerate(self.losses):
            acc[i] = l(model, x, event, time)
        return torch.dot(acc, self.coeffs)

    def __str__(self):
        return " + ".join(f"{c} * {l}" for l, c in zip(self.losses, self.coeffs))

class FullLikelihoodLoss(BaseSurvivalLoss):
    def forward(self, model, x, events, times):
        check_survival_outcomes(events, times)

        p_events = model("density", x[events], times[events])
        loss_events = -torch.nan_to_num(torch.mean(safe_log(torch.diagonal(p_events))), nan=0.0)

        p_censor = model("survival", x[~events], times[~events])
        loss_censor = -torch.nan_to_num(torch.mean(safe_log(torch.diagonal(p_censor))), nan=0.0)

        weight = events.float().mean()

        return weight * loss_events + (1 - weight) * loss_censor

def brier(preds, pred_times, events, times):
    """
    preds: probability
    pred_times: time of pred
    events: event or censoring indicator
    times: event or censoring time
    TO IMPLEMENT: IPCW
    """
    informative = (times.unsqueeze(-1) > pred_times) | events.unsqueeze(-1)
    positive = (times.unsqueeze(-1) <= pred_times) & events.unsqueeze(-1)
    return torch.nn.functional.mse_loss(
        preds[informative], positive[informative].to(torch.float32)
    )


class BrierLoss(BaseSurvivalLoss):
    def forward(self, model, x, events, times):
        """use all unique event times
        could sample them
        """
        check_survival_outcomes(events, times)

        utimes = torch.unique(times[events] if events.any() else times)
        preds = model("failure", x, utimes)
        return brier(preds, utimes, events, times)


def squared_loss(expected_times, events, times):
    """A squared loss where if predicted time is greater than censoring time there is no penalty"""
    # events.unsqueeze(-1)
    deltas = expected_times - times
    weights = torch.logical_or(
        events, deltas < 0
    )  # false/zero only if there is censoring and predicted time is greater than
    return torch.mean(torch.square(deltas * weights))


# FIXME the following classes should replace BrierLoss, with more options and customizations

class ClassificationLoss(BaseSurvivalLoss):
    @abc.abstractmethod
    def get_times(self, event: torch.BoolTensor, time: Tensor) -> Tensor:
        ...
    @staticmethod
    @abc.abstractmethod
    def get_classification_loss() -> Callable[[Tensor, Tensor], Tensor]:
        ...

    def forward(self, model, x, events, times):
        check_survival_outcomes(events, times)

        # get predictions at appropriate times
        pred_times = self.get_times(events, times)
        preds = model("failure", x, pred_times)

        # mask non informative outcomes
        mask = (times.unsqueeze(-1) > pred_times) | events.unsqueeze(-1)
        # binary outcomes
        y = (times.unsqueeze(-1) <= pred_times) & events.unsqueeze(-1)

        return self.get_classification_loss()(preds[mask], y[mask].to(torch.float32))

class ClassificationBatchTimesLoss(ClassificationLoss):
    def __init__(self, max_times=100, only_event_times=True, unique_times=False, random_sampling=False):
        super().__init__()
        self.max_times = max_times
        self.only_event_times = only_event_times
        self.unique_times = unique_times
        self.random_sampling = random_sampling

    def get_times(self, events: torch.BoolTensor, times: Tensor) -> Tensor:
        t = times
        if self.only_event_times and events.any():
            t = t[events]
        if self.unique_times:
            t = torch.unique(t)
        if len(t) > self.max_times:
            if self.random_sampling:
                t = t[torch.randperm(len(t))]
            t = t[:self.max_times]

        return t


class ClassificationFixedTimesLoss(ClassificationLoss):
    def __init__(self, times=None):
        super().__init__()
        self.times = times

    def get_times(self, events, times):
        return self.times


class BrierBatchTimesLoss(ClassificationBatchTimesLoss):
    @staticmethod
    def get_classification_loss():
        return torch.nn.functional.mse_loss
    #classification_loss = test_func
    #classification_loss = torch.nn.functional.mse_loss
class BCEBatchTimesLoss(ClassificationBatchTimesLoss):
    @staticmethod
    def get_classification_loss():
        return torch.nn.functional.binary_cross_entropy

class SquaredLoss(BaseSurvivalLoss):
    def forward(self, model, x, events, times):
        check_survival_outcomes(events, times)
        preds = model("expected_time", x)
        return squared_loss(preds, events, times)


def partial_likelihood_iter(model, x, events, times):
    # dovrebbe essere la versione generale della cox, dove la hazard function dipende dal tempo
    # here we are passing the model, this is not uniform with other losses
    hazards = model(
        "hazard", x, times
    )  # inefficient, computing hazard also for censoring times and multiple times for repeated event times

    acc = 0.0
    for i, t in enumerate(times):
        if events[i]:
            risk_set = times >= t
            if torch.any(risk_set):
                acc += torch.log(hazards[i, i] / torch.sum(hazards[risk_set, i]))
    return -acc

def partial_likelihood_vec(model, x, event, time, epsilon):
    event_time = time[event]
    risk_sets = time.unsqueeze(-1) >= event_time
    # remove eventual risk set with 1 element, since if the hazard is zero then the likelihood is nan
    keep = risk_sets.sum(dim=0) > 1
    hazard = model("hazard", x, event_time)
    top = torch.diag(hazard[event])[keep]
    bot = torch.sum(hazard * risk_sets, dim=-2)[keep]
    lh = top / (bot + epsilon)

    return -torch.sum(torch.log(lh))

def safe_log(x, epsilon=1e-12):
    epsilon = epsilon.to(x.device) if isinstance(epsilon, Tensor) else torch.tensor(epsilon, device=x.device)
    return torch.log(torch.clamp(x, min=epsilon))


def partial_likelihood_vec_safe(model, x, event, time, epsilon):
    if not event.any():
        return torch.zeros((), dtype=time.dtype, device=time.device)

    event_times = time[event]
    risk_sets = time.unsqueeze(-1) >= event_times
    log_hazard = safe_log(model("hazard", x, event_times), epsilon)

    top = torch.diag(log_hazard[event])
    masked_log_hazard = log_hazard.masked_fill(~risk_sets, float("-inf"))
    bot = torch.logsumexp(masked_log_hazard, dim=0)
    log_lh = top - bot

    if torch.isnan(log_lh).any() or torch.isinf(log_lh).any():
        finite_zero = torch.zeros((), dtype=log_lh.dtype, device=log_lh.device)
        log_lh = torch.where(torch.isfinite(log_lh), log_lh, finite_zero)

    return -torch.sum(log_lh)


def partial_likelihood_sorted(model, x, event, time, epsilon):
    """
    Pycox implementation (approximate)
    """
    idx = torch.argsort(-time)
    log_hazard = safe_log(model("hazard", x, time), epsilon)

    event = event[idx]
    log_hazard = log_hazard[idx]
    gamma = log_hazard.max()

    log_cumsum_h = log_hazard.sub(gamma).exp().cumsum(0).add(epsilon).log().add(gamma)
    return - log_hazard.sub(log_cumsum_h).mul(event).sum().div(event.sum())


class PartialLikelihoodLoss(BaseSurvivalLoss):
    def __init__(self, epsilon=1e-12):
        super().__init__()
        self.register_buffer("epsilon", torch.tensor(epsilon, dtype=torch.float32))

    def forward(self, model, x, events, times):
        check_survival_outcomes(events, times)
        return partial_likelihood_vec_safe(model, x, events, times, epsilon=self.epsilon)
