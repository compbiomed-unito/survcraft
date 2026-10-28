import numpy as np
import pytest
import torch

from survcraft import adapters as ad, loss_modules as lm, survival_modules as sm


@pytest.mark.parametrize("policy", ad.check_divergence_values)
@pytest.mark.parametrize("loss", [
    lm.FullLikelihoodLoss(), lm.FullLikelihoodLoss(gpu_mode=True),
    lm.FullLikelihoodLoss(gpu_mode=None), lm.BrierLoss(), lm.SquaredLoss(),
    lm.PartialLikelihoodLoss(), lm.BrierBatchTimesLoss(), lm.BCEBatchTimesLoss(),
    lm.BrierLoss() + lm.SquaredLoss(),
])
def test_loss_forwards_per_call_policy(loss, policy):
    if policy == "skip" and isinstance(loss, lm.FullLikelihoodLoss) and loss.gpu is None:
        pytest.skip("Reference likelihood does not implement skipping")
    calls = []
    model = ad.TorchModel(torch.nn.Identity(), sm.ExponentialSurvivalModule())

    def record_policy(module, args, kwargs):
        calls.append(kwargs["check_divergence"])

    model.register_forward_pre_hook(record_policy, with_kwargs=True)
    x = torch.tensor([[0.2], [0.4], [0.6], [0.8]], requires_grad=True)
    value = loss(model, x, torch.tensor([True, False, True, False]),
                 torch.tensor([1., 2., 3., 4.]), check_divergence=policy)
    assert value.isfinite()
    value.backward()
    assert x.grad.isfinite().all()
    assert calls and set(calls) == {"no" if policy == "skip" else policy}
    assert not hasattr(model, "check_divergence")


@pytest.mark.parametrize("policy", ad.check_divergence_values)
def test_estimator_passes_policy_to_training_validation_and_test_losses(policy):
    calls = []

    def loss(model, x, events, times, *, check_divergence):
        calls.append((model.training, torch.is_grad_enabled(), check_divergence))
        return model.get_raw_params(x).square().mean()

    predictor = ad.SurvivalPredictor(
        loss=loss, check_divergence=policy, epochs=1, batch_size=4,
        early_stopping=True, validation_ratio=0.25, history=True,
    )
    x = np.ones((16, 2), dtype=np.float32)
    predictor.train(x, np.ones(16, dtype=bool), np.ones(16),
                    test_data=(x, np.ones(16, dtype=bool), np.ones(16)))
    assert any(training and grad for training, grad, _ in calls)
    assert sum(not training and not grad for training, grad, _ in calls) >= 2
    assert {p for _, _, p in calls} == {policy}


def test_model_policy_does_not_persist_between_calls():
    model = ad.TorchModel(torch.nn.Identity(), sm.ExponentialSurvivalModule())
    x = torch.tensor([[float("nan")]])
    assert model("expected_time", x, check_divergence="no").isnan().all()
    with pytest.raises(ValueError, match="non-finite"):
        model("expected_time", x)
