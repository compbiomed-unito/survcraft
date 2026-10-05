import numpy as np
import pytest
import torch

from survcraft import adapters as ad
from survcraft import survival_modules as sm


@pytest.mark.parametrize("policy", ["raise", "warn", "no"])
def test_predict_params_returns_numpy_dictionary(policy):
    predictor = ad.SurvivalPredictor(check_divergence=policy)
    X = np.zeros((3, 2), dtype=np.float32)
    predictor._init_model(X)

    result = predictor.predict("params", X)
    expected = predictor.get_distribution_params(X)

    assert result.keys() == expected.keys() == {"scale"}
    assert isinstance(result["scale"], np.ndarray)
    assert result["scale"].shape == (3, 1)
    np.testing.assert_allclose(result["scale"], expected["scale"].numpy())


@pytest.mark.parametrize("policy", ["raise", "warn", "no"])
def test_params_checks_transformed_values_and_names(policy):
    survival = sm.BaseSurvivalModule([
        sm.FreeParameter("finite"),
        sm.PositiveParameter("overflow", func=torch.exp),
    ])
    model = ad.TorchModel(torch.nn.Identity(), survival, check_divergence=policy)
    raw = torch.tensor([[1.0, 1000.0]])

    if policy == "raise":
        with pytest.raises(ValueError, match=r"non-finite.*params\[overflow\]") as caught:
            model("params", raw)
        context = caught.value.args[1]
        assert context["tensor_name"] == "params[overflow]"
        torch.testing.assert_close(context["raw_params"], raw)
        assert torch.isinf(context["tensor_values"]).all()
    elif policy == "warn":
        with pytest.warns(UserWarning, match=r"non-finite.*params\[overflow\]") as caught:
            result = model("params", raw)
        assert len(caught) == 1
        assert result.keys() == {"finite", "overflow"}
        assert torch.isinf(result["overflow"]).all()
    else:
        assert torch.isinf(model("params", raw)["overflow"]).all()


@pytest.mark.parametrize("policy", ["raise", "warn", "no"])
@pytest.mark.parametrize("scalar", [True, False])
def test_nonfinite_time_predictions_have_correct_diagnostics(policy, scalar):
    model = ad.TorchModel(torch.nn.Identity(), sm.WeibullSurvivalModule(),
                          check_divergence=policy)
    # softplus(0) < 1 gives infinite hazard at zero; shape > 1 remains finite.
    raw = torch.tensor([[0.0, 0.0], [0.0, 2.0]])
    times = torch.tensor(0.0 if scalar else [0.0, 1.0])
    message = r"non-finite value\(s\) in hazard: 1inf.*for 1 samples at 1 times \(0.0\)"

    if policy == "raise":
        with pytest.raises(ValueError, match=message) as caught:
            model("hazard", raw, times)
        context = caught.value.args[1]
        assert context["tensor_name"] == "hazard"
        torch.testing.assert_close(context["times"], times)
        assert context["tensor_values"].shape == ((2,) if scalar else (2, 2))
    else:
        if policy == "warn":
            with pytest.warns(UserWarning, match=message):
                result = model("hazard", raw, times)
        else:
            result = model("hazard", raw, times)
        assert result.shape == ((2,) if scalar else (2, 2))
        assert torch.isinf(result).sum() == 1


def test_distribution_inspection_disables_dropout_and_preserves_batch_norm_state():
    predictor = ad.SurvivalPredictor(input=ad.FeedForwardNetAdapter(
        hidden_sizes=[8], dropout=0.5, batch_norm=True))
    X = np.arange(32, dtype=np.float32).reshape(8, 4) / 10
    predictor._init_model(X)
    predictor.model_.train()
    # Populate running statistics as training would, then inspect one sample.
    predictor.model_.get_raw_params(predictor._tensor(X))
    bn = next(m for m in predictor.model_.modules() if isinstance(m, torch.nn.BatchNorm1d))
    before = {k: v.clone() for k, v in bn.state_dict().items()}

    first = predictor.get_distribution_params(X[:1])
    second = predictor.get_distribution_params(X[:1])

    assert all(not m.training for m in predictor.model_.modules())
    for key, value in before.items():
        torch.testing.assert_close(bn.state_dict()[key], value, rtol=0, atol=0)
    assert first.keys() == second.keys() == {"scale"}
    assert isinstance(first["scale"], torch.Tensor)
    assert not first["scale"].requires_grad
    torch.testing.assert_close(first["scale"], second["scale"], rtol=0, atol=0)
    numpy_params = predictor.predict("params", X[:1])
    np.testing.assert_allclose(first["scale"].numpy(), numpy_params["scale"])
