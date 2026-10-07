r"""Composable scalar objectives for right-censored survival models.

Notes
-----
Losses receive the model itself and request the prediction modes they
need. See :class:`BaseSurvivalLoss` for tensor, device, and composition
contracts. Classification objectives omit uninformative censored pairs;
they do not implement inverse-probability-of-censoring weighting (IPCW).
"""

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
    r"""Check boolean event dtype and matching one-dimensional outcome shapes.

    Parameters
    ----------
    events : torch.Tensor of bool, shape (n_samples,)
        Observed-event indicators.
    times : torch.Tensor, shape (n_samples,)
        Matching event or censoring times.

    Returns
    -------
    None
        Returns normally if the checks pass.

    Raises
    ------
    TypeError
        If events is not boolean.
    ValueError
        If events is not one-dimensional or times has a different shape.

    Notes
    -----
    This helper does not check nonnegativity, finiteness, nonempty input,
    or device compatibility. Callers must satisfy the full loss contract.
    """
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
    r"""Define scalar model-aware losses and arithmetic composition.

    Notes
    -----
    Each loss is called as ``loss(model, x, event, time)``. Features have
    shape ``(n_samples, n_features)``, boolean event indicators and floating
    times have shape ``(n_samples,)``. Times must be finite and nonnegative;
    True denotes an observed event, False right censoring. The model
    implements :class:`survcraft.adapters.TorchModel` prediction modes,
    returning matrices for shared time vectors and vectors for summaries.

    Losses return a zero-dimensional tensor to minimize. Reductions differ
    by objective: full likelihood and classification losses take means,
    while partial likelihood takes a sum over events. Not all objectives
    yield finite or differentiable results for empty or event-free inputs.
    The caller controls training/evaluation mode and autograd.

    Move model, data, and loss buffers to compatible devices. Estimator
    training does not move the loss automatically. ``loss.to(device)``
    moves registered buffers and nested losses; ordinary tensor attributes
    such as ClassificationFixedTimesLoss.times do not follow this move.

    Arithmetic constructs :class:`LinearCombinationLoss`: ``a + b``,
    ``a * c``, and ``c * a`` are supported for losses a, b and scalar c.
    Nested combinations are flattened, preserving their modules and
    coefficients, and compute :math:`L=\sum_j c_jL_j`. Loss objects are
    shared rather than cloned. Subtraction, division, and ``sum(losses)``
    starting from integer zero are not implemented.

    Examples
    --------
    >>> objective = FullLikelihoodLoss() + 0.1 * PartialLikelihoodLoss()
    >>> len(objective.losses)
    2

    See Also
    --------
    survcraft.adapters.SurvivalPredictor : Mini-batch optimization.
    """
    @abc.abstractmethod
    def forward(self, model: torch.nn.Module, x: Tensor, event: torch.BoolTensor, time: Tensor):
        r"""Compute a scalar objective by requesting model predictions.

        Parameters
        ----------
        model : torch.nn.Module
            Callable ``model(mode, x, times=None)`` following the prediction
            contract of survcraft.adapters.TorchModel.
        x : torch.Tensor, shape (n_samples, n_features)
            Floating feature matrix on the model device.
        event : torch.Tensor of bool, shape (n_samples,)
            True for an observed event, False for right censoring.
        time : torch.Tensor, shape (n_samples,)
            Finite nonnegative event or censoring times on the model device.

        Returns
        -------
        torch.Tensor, shape ()
            Scalar loss to minimize, differentiable where the selected objective
            and predictions allow.

        Notes
        -----
        Subclasses choose prediction modes and reduction; see the shared
        :class:`BaseSurvivalLoss` contract.
        """

    def __add__(self, other):
        r"""Construct a sum of two losses.

        Parameters
        ----------
        other : BaseSurvivalLoss
            Objective to add.

        Returns
        -------
        LinearCombinationLoss
            Flattened weighted combination sharing the original loss modules.
        """
        return LinearCombinationLoss([self, other], torch.tensor([1.0, 1.0]))

    def __mul__(self, num: float):
        r"""Construct a scalar-weighted loss.

        Parameters
        ----------
        num : float
            Scalar loss coefficient.

        Returns
        -------
        LinearCombinationLoss
            Flattened weighted combination sharing the original loss modules.
        """
        return LinearCombinationLoss([self], torch.tensor([num]))

    def __rmul__(self, num: float):
        r"""Construct a scalar-weighted loss.

        Parameters
        ----------
        num : float
            Scalar loss coefficient.

        Returns
        -------
        LinearCombinationLoss
            Flattened weighted combination sharing the original loss modules.
        """
        return self * num


class LinearCombinationLoss(BaseSurvivalLoss):
    r"""Compute a weighted sum of component scalar objectives.

    Parameters
    ----------
    losses : sequence of BaseSurvivalLoss
        Component objectives, including optional nested combinations.
    coeffs : torch.Tensor, shape (n_losses,)
        Floating coefficients, one per supplied component.

    Attributes
    ----------
    losses : torch.nn.ModuleList
        Flattened registered component losses.
    coeffs : torch.Tensor
        Registered flattened coefficient buffer.

    Raises
    ------
    AssertionError
        If coefficients do not match the number of supplied losses.

    Notes
    -----
    Computes :math:`L=\sum_j c_jL_j` without additional normalization.
    Nested weights are multiplied and flattened. Construction currently
    creates CPU unit-coefficient tensors for simple components; build on
    CPU then move the whole combination when using an accelerator.
    Uses the scalar and device contracts of :class:`BaseSurvivalLoss`.
    """
    def __init__(self, losses, coeffs):
        r"""Initialize objective options and registered state.

        See Also
        --------
        LinearCombinationLoss : Constructor parameters and objective definition.
        """
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
        r"""Compute the scalar objective defined by this loss.

        Parameters
        ----------
        model : torch.nn.Module
            Callable ``model(mode, x, times=None)`` following the prediction
            contract of survcraft.adapters.TorchModel.
        x : torch.Tensor, shape (n_samples, n_features)
            Floating feature matrix on the model device.
        event : torch.Tensor of bool, shape (n_samples,)
            True for an observed event, False for right censoring.
        time : torch.Tensor, shape (n_samples,)
            Finite nonnegative event or censoring times on the model device.

        Returns
        -------
        torch.Tensor, shape ()
            Scalar loss to minimize, differentiable where the selected objective
            and predictions allow.

        See Also
        --------
        LinearCombinationLoss : Implemented formula and reduction.
        BaseSurvivalLoss : Shared model, tensor, and device contracts.
        """
        acc = torch.zeros_like(self.coeffs)
        for i, l in enumerate(self.losses):
            acc[i] = l(model, x, event, time)
        return torch.dot(acc, self.coeffs)

    def __str__(self):
        r"""Format the flattened weighted objectives.

        Returns
        -------
        str
            Coefficients and component descriptions joined by plus signs.
        """
        return " + ".join(f"{c} * {l}" for l, c in zip(self.losses, self.coeffs))

class FullLikelihoodLoss(BaseSurvivalLoss):
    r"""Minimize negative event-density and censor-survival log likelihood.

    Parameters
    ----------
    censoring_alpha : float, default=1.0
        Nonnegative weight multiplying the censored likelihood contribution.

    Attributes
    ----------
    censoring_alpha : float
        Weight of the censored term.

    Raises
    ------
    ValueError
        If censoring_alpha is negative.

    Notes
    -----
    For batch size N, event indicators ei, and observed times ti,

    .. math::

        L=-\frac1N\sum_i\left[
            e_i\log\max(f_i(t_i),10^{-12})
            +\alpha(1-e_i)\log\max(S_i(t_i),10^{-12})\right].

    The implementation averages event and censor groups separately and
    weights them by their batch proportions. Empty-group means become zero
    through nan_to_num; nonempty NaN means are also replaced by zero.
    Requests density and survival matrices and uses their diagonals.
    Uses the :class:`BaseSurvivalLoss` contract.
    """
    def __init__(self, censoring_alpha: float = 1.0):
        r"""Initialize objective options and registered state.

        See Also
        --------
        FullLikelihoodLoss : Constructor parameters and objective definition.
        """
        super().__init__()
        
        # parameter to lower weight of censored term of loss
        # see deep survival machines paper for rationale (long tail bias)
        if censoring_alpha < 0:
            raise ValueError(f'censoring alpha must be non negative, got {censoring_alpha}')
        self.censoring_alpha = censoring_alpha

    def forward(self, model, x, events, times):
        r"""Compute the scalar objective defined by this loss.

        Parameters
        ----------
        model : torch.nn.Module
            Callable ``model(mode, x, times=None)`` following the prediction
            contract of survcraft.adapters.TorchModel.
        x : torch.Tensor, shape (n_samples, n_features)
            Floating feature matrix on the model device.
        events : torch.Tensor of bool, shape (n_samples,)
            True for an observed event, False for right censoring.
        times : torch.Tensor, shape (n_samples,)
            Finite nonnegative event or censoring times on the model device.

        Returns
        -------
        torch.Tensor, shape ()
            Scalar loss to minimize, differentiable where the selected objective
            and predictions allow.

        Raises
        ------
        TypeError
            If event indicators are not boolean.
        ValueError
            If outcome shapes are invalid.

        See Also
        --------
        FullLikelihoodLoss : Implemented formula and reduction.
        BaseSurvivalLoss : Shared model, tensor, and device contracts.
        """
        check_survival_outcomes(events, times)

        p_events = model("density", x[events], times[events])
        loss_events = -torch.nan_to_num(torch.mean(safe_log(torch.diagonal(p_events))), nan=0.0)

        p_censor = model("survival", x[~events], times[~events])
        loss_censor = -torch.nan_to_num(torch.mean(safe_log(torch.diagonal(p_censor))), nan=0.0)

        weight = events.float().mean()

        return weight * loss_events + self.censoring_alpha * (1 - weight) * loss_censor

def brier(preds, pred_times, events, times):
    r"""Compute mean squared failure-probability error on informative pairs.

    Parameters
    ----------
    preds : torch.Tensor, shape (n_samples, n_pred_times)
        Predicted failure probabilities.
    pred_times : torch.Tensor, shape (n_pred_times,)
        Shared evaluation times.
    events : torch.Tensor of bool, shape (n_samples,)
        Observed-event indicators.
    times : torch.Tensor, shape (n_samples,)
        Observed times.

    Returns
    -------
    torch.Tensor, shape ()
        Mean squared error over informative pairs, or NaN if there are none.

    Notes
    -----
    For evaluation times uj, the informative mask and binary labels are

    .. math::

        M_{ij}=\mathbf{1}\{t_i>u_j\ \mathrm{or}\ e_i=1\},\quad
        y_{ij}=\mathbf{1}\{t_i\leq u_j\ \mathrm{and}\ e_i=1\}.

    Predictions are failure probabilities :math:`p_{ij}=F_i(u_j)`.
    The reduction is a mean over all informative pairs, not an average of
    per-time means. Censoring at or before an evaluation time is omitted.
    No IPCW correction is applied. An empty mask can produce NaN.

    The score is :math:`\sum_{ij}M_{ij}(p_{ij}-y_{ij})^2/\sum_{ij}M_{ij}`.
    """
    informative = (times.unsqueeze(-1) > pred_times) | events.unsqueeze(-1)
    positive = (times.unsqueeze(-1) <= pred_times) & events.unsqueeze(-1)
    return torch.nn.functional.mse_loss(
        preds[informative], positive[informative].to(torch.float32)
    )


class BrierLoss(BaseSurvivalLoss):
    r"""Evaluate unweighted Brier error at unique observed event times.

    Notes
    -----
    For evaluation times uj, the informative mask and binary labels are

    .. math::

        M_{ij}=\mathbf{1}\{t_i>u_j\ \mathrm{or}\ e_i=1\},\quad
        y_{ij}=\mathbf{1}\{t_i\leq u_j\ \mathrm{and}\ e_i=1\}.

    Predictions are failure probabilities :math:`p_{ij}=F_i(u_j)`.
    The reduction is a mean over all informative pairs, not an average of
    per-time means. Censoring at or before an evaluation time is omitted.
    No IPCW correction is applied. An empty mask can produce NaN.

    Evaluation times are unique event times, or all unique observed times
    when no events occur. Computes
    :math:`L=\sum_{ij}M_{ij}(F_i(u_j)-y_{ij})^2/\sum_{ij}M_{ij}`.
    Uses the :class:`BaseSurvivalLoss` contract.

    See Also
    --------
    BrierBatchTimesLoss : Configurable batch time selection.
    """
    def forward(self, model, x, events, times):
        r"""Compute the scalar objective defined by this loss.

        Parameters
        ----------
        model : torch.nn.Module
            Callable ``model(mode, x, times=None)`` following the prediction
            contract of survcraft.adapters.TorchModel.
        x : torch.Tensor, shape (n_samples, n_features)
            Floating feature matrix on the model device.
        events : torch.Tensor of bool, shape (n_samples,)
            True for an observed event, False for right censoring.
        times : torch.Tensor, shape (n_samples,)
            Finite nonnegative event or censoring times on the model device.

        Returns
        -------
        torch.Tensor, shape ()
            Scalar loss to minimize, differentiable where the selected objective
            and predictions allow.

        Raises
        ------
        TypeError
            If event indicators are not boolean.
        ValueError
            If outcome shapes are invalid.

        See Also
        --------
        BrierLoss : Implemented formula and reduction.
        BaseSurvivalLoss : Shared model, tensor, and device contracts.
        """
        check_survival_outcomes(events, times)

        utimes = torch.unique(times[events] if events.any() else times)
        preds = model("failure", x, utimes)
        return brier(preds, utimes, events, times)


def squared_loss(expected_times, events, times):
    r"""Penalize observed time errors and early predictions under censoring.

    Parameters
    ----------
    expected_times : torch.Tensor, shape (n_samples,)
        Predicted expected event times.
    events : torch.Tensor of bool, shape (n_samples,)
        Observed-event indicators.
    times : torch.Tensor, shape (n_samples,)
        Observed times.

    Returns
    -------
    torch.Tensor, shape ()
        Mean masked squared error.

    Notes
    -----
    With predicted expected times mi, residuals di = mi - ti, and event
    indicators ei, the implemented objective is

    .. math::

        L=\frac1N\sum_i\left[d_i\,\mathbf{1}\{e_i=1\ \mathrm{or}\ d_i<0\}\right]^2.

    Censored observations are penalized only if the predicted mean is
    earlier than censoring. This is a one-sided squared regression loss;
    it does not account for the censoring distribution.
    """
    # events.unsqueeze(-1)
    deltas = expected_times - times
    weights = torch.logical_or(
        events, deltas < 0
    )  # false/zero only if there is censoring and predicted time is greater than
    return torch.mean(torch.square(deltas * weights))


# FIXME the following classes should replace BrierLoss, with more options and customizations

class ClassificationLoss(BaseSurvivalLoss):
    r"""Define classification objectives on informative time/outcome pairs.

    Notes
    -----
    For evaluation times uj, the informative mask and binary labels are

    .. math::

        M_{ij}=\mathbf{1}\{t_i>u_j\ \mathrm{or}\ e_i=1\},\quad
        y_{ij}=\mathbf{1}\{t_i\leq u_j\ \mathrm{and}\ e_i=1\}.

    Predictions are failure probabilities :math:`p_{ij}=F_i(u_j)`.
    The reduction is a mean over all informative pairs, not an average of
    per-time means. Censoring at or before an evaluation time is omitted.
    No IPCW correction is applied. An empty mask can produce NaN.

    Subclasses select times with get_times and a scalar classification
    criterion with get_classification_loss. Uses :class:`BaseSurvivalLoss`.
    """
    @abc.abstractmethod
    def get_times(self, event: torch.BoolTensor, time: Tensor) -> Tensor:
        r"""Choose the shared classification evaluation grid.

        Parameters
        ----------
        event : torch.Tensor of bool, shape (n_samples,)
            Observed-event indicators.
        time : torch.Tensor, shape (n_samples,)
            Observed event or censoring times.

        Returns
        -------
        torch.Tensor, shape (n_pred_times,)
            Shared evaluation times on the data device.
        """
        ...
    @staticmethod
    @abc.abstractmethod
    def get_classification_loss() -> Callable[[Tensor, Tensor], Tensor]:
        r"""Return the scalar classification criterion.

        Returns
        -------
        callable
            Function accepting prediction and label vectors and returning a
            zero-dimensional mean loss tensor.
        """
        ...

    def forward(self, model, x, events, times):
        r"""Compute the scalar objective defined by this loss.

        Parameters
        ----------
        model : torch.nn.Module
            Callable ``model(mode, x, times=None)`` following the prediction
            contract of survcraft.adapters.TorchModel.
        x : torch.Tensor, shape (n_samples, n_features)
            Floating feature matrix on the model device.
        events : torch.Tensor of bool, shape (n_samples,)
            True for an observed event, False for right censoring.
        times : torch.Tensor, shape (n_samples,)
            Finite nonnegative event or censoring times on the model device.

        Returns
        -------
        torch.Tensor, shape ()
            Scalar loss to minimize, differentiable where the selected objective
            and predictions allow.

        Raises
        ------
        TypeError
            If event indicators are not boolean.
        ValueError
            If outcome shapes are invalid.

        See Also
        --------
        ClassificationLoss : Implemented formula and reduction.
        BaseSurvivalLoss : Shared model, tensor, and device contracts.
        """
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
    r"""Select a capped evaluation grid from batch outcomes.

    Parameters
    ----------
    max_times : int, default=100
        Maximum evaluation grid length; should be positive.
    only_event_times : bool, default=True
        Use event times if there are events; otherwise use all observed times.
    unique_times : bool, default=False
        Remove duplicates before truncation using torch.unique.
    random_sampling : bool, default=False
        Randomly permute candidates before truncation when the count exceeds
        max_times. Uses the global PyTorch RNG.

    Notes
    -----
    Abstract classification family using :class:`ClassificationLoss`.
    Without random sampling, takes the first max_times candidates in their
    current order (sorted if unique_times=True). Repeated times otherwise
    contribute repeatedly to the loss. Candidate options are not validated.
    """
    def __init__(self, max_times=100, only_event_times=True, unique_times=False, random_sampling=False):
        r"""Initialize objective options and registered state.

        See Also
        --------
        ClassificationBatchTimesLoss : Constructor parameters and objective definition.
        """
        super().__init__()
        self.max_times = max_times
        self.only_event_times = only_event_times
        self.unique_times = unique_times
        self.random_sampling = random_sampling

    def get_times(self, events: torch.BoolTensor, times: Tensor) -> Tensor:
        r"""Filter, optionally deduplicate, and cap batch times.

        Parameters
        ----------
        events : torch.Tensor of bool, shape (n_samples,)
            Observed-event indicators.
        times : torch.Tensor, shape (n_samples,)
            Observed times.

        Returns
        -------
        torch.Tensor, shape (n_pred_times,)
            Selected times; see the constructor selection rules.
        """
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
    r"""Use a fixed grid for subclass-defined classification objectives.

    Parameters
    ----------
    times : torch.Tensor, shape (n_pred_times,), optional
        Shared evaluation times on the model device. A real tensor is
        required for evaluation despite the constructor default None.

    Attributes
    ----------
    times : torch.Tensor or None
        Ordinary attribute, not a registered buffer.

    Notes
    -----
    Abstract family requiring get_classification_loss in a subclass.
    Uses :class:`ClassificationLoss`; times are not converted or moved
    by module.to().
    """
    def __init__(self, times=None):
        r"""Initialize objective options and registered state.

        See Also
        --------
        ClassificationFixedTimesLoss : Constructor parameters and objective definition.
        """
        super().__init__()
        self.times = times

    def get_times(self, events, times):
        r"""Return the stored evaluation grid unchanged.

        Parameters
        ----------
        events : torch.Tensor
            Unused batch event indicators.
        times : torch.Tensor
            Unused batch observed times.

        Returns
        -------
        torch.Tensor or None
            The constructor-supplied times attribute.
        """
        return self.times


class BrierBatchTimesLoss(ClassificationBatchTimesLoss):
    r"""Compute unweighted Brier error on a configurable batch grid.

    Parameters
    ----------
    max_times : int, default=100
        Maximum evaluation grid length; should be positive.
    only_event_times : bool, default=True
        Use event times if there are events; otherwise use all observed times.
    unique_times : bool, default=False
        Remove duplicates before truncation using torch.unique.
    random_sampling : bool, default=False
        Randomly permute candidates before truncation when the count exceeds
        max_times. Uses the global PyTorch RNG.

    Notes
    -----
    For evaluation times uj, the informative mask and binary labels are

    .. math::

        M_{ij}=\mathbf{1}\{t_i>u_j\ \mathrm{or}\ e_i=1\},\quad
        y_{ij}=\mathbf{1}\{t_i\leq u_j\ \mathrm{and}\ e_i=1\}.

    Predictions are failure probabilities :math:`p_{ij}=F_i(u_j)`.
    The reduction is a mean over all informative pairs, not an average of
    per-time means. Censoring at or before an evaluation time is omitted.
    No IPCW correction is applied. An empty mask can produce NaN.

    Uses inherited :class:`ClassificationBatchTimesLoss` selection options.
    Computes :math:`L=\sum_{ij}M_{ij}(p_{ij}-y_{ij})^2/\sum_{ij}M_{ij}`.
    Uses the :class:`BaseSurvivalLoss` scalar contract.
    """
    @staticmethod
    def get_classification_loss():
        r"""Return the mean classification criterion.

        Returns
        -------
        callable
            PyTorch mean squared error.
        """
        return torch.nn.functional.mse_loss
    #classification_loss = test_func
    #classification_loss = torch.nn.functional.mse_loss
class BCEBatchTimesLoss(ClassificationBatchTimesLoss):
    r"""Compute binary cross entropy on a configurable batch grid.

    Parameters
    ----------
    max_times : int, default=100
        Maximum evaluation grid length; should be positive.
    only_event_times : bool, default=True
        Use event times if there are events; otherwise use all observed times.
    unique_times : bool, default=False
        Remove duplicates before truncation using torch.unique.
    random_sampling : bool, default=False
        Randomly permute candidates before truncation when the count exceeds
        max_times. Uses the global PyTorch RNG.

    Notes
    -----
    For evaluation times uj, the informative mask and binary labels are

    .. math::

        M_{ij}=\mathbf{1}\{t_i>u_j\ \mathrm{or}\ e_i=1\},\quad
        y_{ij}=\mathbf{1}\{t_i\leq u_j\ \mathrm{and}\ e_i=1\}.

    Predictions are failure probabilities :math:`p_{ij}=F_i(u_j)`.
    The reduction is a mean over all informative pairs, not an average of
    per-time means. Censoring at or before an evaluation time is omitted.
    No IPCW correction is applied. An empty mask can produce NaN.

    Uses inherited :class:`ClassificationBatchTimesLoss` selection options.
    Computes

    .. math::

        L=-\frac{\sum_{ij}M_{ij}[y_{ij}\log p_{ij}
           +(1-y_{ij})\log(1-p_{ij})]}{\sum_{ij}M_{ij}}.

    PyTorch binary_cross_entropy clamps log terms to at least -100.
    Uses the :class:`BaseSurvivalLoss` scalar contract.
    """
    @staticmethod
    def get_classification_loss():
        r"""Return the mean classification criterion.

        Returns
        -------
        callable
            PyTorch binary cross entropy.
        """
        return torch.nn.functional.binary_cross_entropy

class SquaredLoss(BaseSurvivalLoss):
    r"""Regress expected event times with a one-sided censoring penalty.

    Notes
    -----
    With predicted expected times mi, residuals di = mi - ti, and event
    indicators ei, the implemented objective is

    .. math::

        L=\frac1N\sum_i\left[d_i\,\mathbf{1}\{e_i=1\ \mathrm{or}\ d_i<0\}\right]^2.

    Censored observations are penalized only if the predicted mean is
    earlier than censoring. This is a one-sided squared regression loss;
    it does not account for the censoring distribution.

    Requests expected_time; the distribution must implement a finite mean.
    Uses the :class:`BaseSurvivalLoss` contract.
    """
    def forward(self, model, x, events, times):
        r"""Compute the scalar objective defined by this loss.

        Parameters
        ----------
        model : torch.nn.Module
            Callable ``model(mode, x, times=None)`` following the prediction
            contract of survcraft.adapters.TorchModel.
        x : torch.Tensor, shape (n_samples, n_features)
            Floating feature matrix on the model device.
        events : torch.Tensor of bool, shape (n_samples,)
            True for an observed event, False for right censoring.
        times : torch.Tensor, shape (n_samples,)
            Finite nonnegative event or censoring times on the model device.

        Returns
        -------
        torch.Tensor, shape ()
            Scalar loss to minimize, differentiable where the selected objective
            and predictions allow.

        Raises
        ------
        TypeError
            If event indicators are not boolean.
        ValueError
            If outcome shapes are invalid.

        See Also
        --------
        SquaredLoss : Implemented formula and reduction.
        BaseSurvivalLoss : Shared model, tensor, and device contracts.
        """
        check_survival_outcomes(events, times)
        preds = model("expected_time", x)
        return squared_loss(preds, events, times)


def partial_likelihood_iter(model, x, events, times):
    # dovrebbe essere la versione generale della cox, dove la hazard function dipende dal tempo
    # here we are passing the model, this is not uniform with other losses
    r"""Compute an experimental partial-likelihood variant.

    Parameters
    ----------
    model : torch.nn.Module
        Callable ``model(mode, x, times=None)`` following the prediction
        contract of survcraft.adapters.TorchModel.
    x : torch.Tensor, shape (n_samples, n_features)
        Floating feature matrix on the model device.
    events : torch.Tensor of bool, shape (n_samples,)
        True for an observed event, False for right censoring.
    times : torch.Tensor, shape (n_samples,)
        Finite nonnegative event or censoring times on the model device.

    Returns
    -------
    torch.Tensor or float
        Negative sum of event log likelihoods.

    Notes
    -----
    Experimental loop implementation using unprotected log hazard ratios.
    Includes singleton risk sets. May return the Python value -0.0 when
    there are no events; does not enforce the scalar tensor contract.
    """
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
    r"""Compute an experimental partial-likelihood variant.

    Parameters
    ----------
    model : torch.nn.Module
        Callable ``model(mode, x, times=None)`` following the prediction
        contract of survcraft.adapters.TorchModel.
    x : torch.Tensor, shape (n_samples, n_features)
        Floating feature matrix on the model device.
    event : torch.Tensor of bool, shape (n_samples,)
        True for an observed event, False for right censoring.
    time : torch.Tensor, shape (n_samples,)
        Finite nonnegative event or censoring times on the model device.
    epsilon : float or torch.Tensor
        Numerical floor or denominator offset used by this implementation.

    Returns
    -------
    torch.Tensor, shape ()
        Partial-likelihood objective.

    Notes
    -----
    Experimental vectorized implementation. Drops singleton risk sets,
    adds epsilon to denominators, and takes unprotected log ratios. Uses
    a sum reduction; may produce infinities for zero numerator hazards.
    """
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
    r"""Take a logarithm after clamping to a numerical floor.

    Parameters
    ----------
    x : torch.Tensor
        Values to transform.
    epsilon : float or torch.Tensor, default=1e-12
        Lower bound, moved to x.device if needed; should be positive.

    Returns
    -------
    torch.Tensor
        ``log(clamp(x, min=epsilon))`` with the same shape as x.

    Notes
    -----
    NaNs remain NaN. Values below the floor have zero clamp gradient.
    """
    epsilon = epsilon.to(x.device) if isinstance(epsilon, Tensor) else torch.tensor(epsilon, device=x.device)
    return torch.log(torch.clamp(x, min=epsilon))


def partial_likelihood_vec_safe(model, x, event, time, epsilon):
    r"""Compute a stable summed hazard partial likelihood.

    Parameters
    ----------
    model : torch.nn.Module
        Callable ``model(mode, x, times=None)`` following the prediction
        contract of survcraft.adapters.TorchModel.
    x : torch.Tensor, shape (n_samples, n_features)
        Floating feature matrix on the model device.
    event : torch.Tensor of bool, shape (n_samples,)
        True for an observed event, False for right censoring.
    time : torch.Tensor, shape (n_samples,)
        Finite nonnegative event or censoring times on the model device.
    epsilon : float or torch.Tensor
        Numerical floor or denominator offset used by this implementation.

    Returns
    -------
    torch.Tensor, shape ()
        Partial-likelihood objective.

    Notes
    -----
    For each event i, the risk set is R(ti) = {j : tj >= ti}.
    The implemented time-dependent hazard partial likelihood is

    .. math::

        L=-\sum_{i:e_i=1}\left[\log\max(h_i(t_i),\epsilon)
           -\log\sum_{j\in R(t_i)}\max(h_j(t_i),\epsilon)\right].

    Uses logsumexp for denominators. Tied events share the same full risk
    set (a Breslow-style product); there is no Efron correction. This is a
    sum over events, not a mean, and uses batch risk sets during training.
    Event-free batches return a device-local scalar zero without gradients;
    non-finite log terms are replaced by zero.
    """
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
    r"""Compute an experimental partial-likelihood variant.

    Parameters
    ----------
    model : torch.nn.Module
        Callable ``model(mode, x, times=None)`` following the prediction
        contract of survcraft.adapters.TorchModel.
    x : torch.Tensor, shape (n_samples, n_features)
        Floating feature matrix on the model device.
    event : torch.Tensor of bool, shape (n_samples,)
        True for an observed event, False for right censoring.
    time : torch.Tensor, shape (n_samples,)
        Finite nonnegative event or censoring times on the model device.
    epsilon : float or torch.Tensor
        Numerical floor or denominator offset used by this implementation.

    Returns
    -------
    torch.Tensor, shape ()
        Partial-likelihood objective.

    Notes
    -----
    Experimental approximation using reverse-time sorting and cumulative
    hazards. The current model returns a full hazard matrix, while this
    algorithm assumes one risk value per sample; ordering and event
    broadcasting therefore need review before use. Divides by event count
    and has no event-free guard.
    """
    idx = torch.argsort(-time)
    log_hazard = safe_log(model("hazard", x, time), epsilon)

    event = event[idx]
    log_hazard = log_hazard[idx]
    gamma = log_hazard.max()

    log_cumsum_h = log_hazard.sub(gamma).exp().cumsum(0).add(epsilon).log().add(gamma)
    return - log_hazard.sub(log_cumsum_h).mul(event).sum().div(event.sum())


class PartialLikelihoodLoss(BaseSurvivalLoss):
    r"""Minimize the summed hazard partial likelihood over observed events.

    Parameters
    ----------
    epsilon : float, default=1e-12
        Lower bound for hazard values before logarithms; should be positive.

    Attributes
    ----------
    epsilon : torch.Tensor
        Registered float32 numerical-floor buffer.

    Notes
    -----
    For each event i, the risk set is R(ti) = {j : tj >= ti}.
    The implemented time-dependent hazard partial likelihood is

    .. math::

        L=-\sum_{i:e_i=1}\left[\log\max(h_i(t_i),\epsilon)
           -\log\sum_{j\in R(t_i)}\max(h_j(t_i),\epsilon)\right].

    Uses logsumexp for denominators. Tied events share the same full risk
    set (a Breslow-style product); there is no Efron correction. This is a
    sum over events, not a mean, and uses batch risk sets during training.
    Event-free batches return a device-local scalar zero without gradients;
    non-finite log terms are replaced by zero.

    Requests hazard and uses the :class:`BaseSurvivalLoss` contract.
    The positive epsilon contract is not validated by the constructor.
    """
    def __init__(self, epsilon=1e-12):
        r"""Initialize objective options and registered state.

        See Also
        --------
        PartialLikelihoodLoss : Constructor parameters and objective definition.
        """
        super().__init__()
        self.register_buffer("epsilon", torch.tensor(epsilon, dtype=torch.float32))

    def forward(self, model, x, events, times):
        r"""Compute the scalar objective defined by this loss.

        Parameters
        ----------
        model : torch.nn.Module
            Callable ``model(mode, x, times=None)`` following the prediction
            contract of survcraft.adapters.TorchModel.
        x : torch.Tensor, shape (n_samples, n_features)
            Floating feature matrix on the model device.
        events : torch.Tensor of bool, shape (n_samples,)
            True for an observed event, False for right censoring.
        times : torch.Tensor, shape (n_samples,)
            Finite nonnegative event or censoring times on the model device.

        Returns
        -------
        torch.Tensor, shape ()
            Scalar loss to minimize, differentiable where the selected objective
            and predictions allow.

        Raises
        ------
        TypeError
            If event indicators are not boolean.
        ValueError
            If outcome shapes are invalid.

        See Also
        --------
        PartialLikelihoodLoss : Implemented formula and reduction.
        BaseSurvivalLoss : Shared model, tensor, and device contracts.
        """
        check_survival_outcomes(events, times)
        return partial_likelihood_vec_safe(model, x, events, times, epsilon=self.epsilon)
