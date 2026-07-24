import pytest
import torch

from survcraft import survival_modules as sm
from survcraft.util import get_subclasses_in_module


STEP_EXP_BREAKS = torch.tensor([0.0, 0.5, 1.0, 2.0], dtype=torch.float32)
TIMES = torch.linspace(0.05, 3.0, 2001, dtype=torch.float32)

MODULE_DEFAULTS = {
    sm.StepExpSurvivalModule: {
        "breaks": STEP_EXP_BREAKS,
    },
    sm.ProportionalHazardSurvivalModule: {
        "baseline": sm.ExponentialSurvivalModule(),
        "baseline_params": torch.nn.Parameter(torch.zeros(1, dtype=torch.float32)),
    },
    sm.AcceleratedFailureTimeSurvivalModule: {
        "baseline": sm.ExponentialSurvivalModule(),
        "baseline_params": torch.nn.Parameter(torch.zeros(1, dtype=torch.float32)),
    },
    sm.MixtureSurvivalModule: {
        "baselines": [sm.ExponentialSurvivalModule(), sm.WeibullSurvivalModule()],
    },
    sm.FractalNoiseSurvivalModule: {
        "seed": 0,
    },
}

MODULE_CLASSES = sorted(
    get_subclasses_in_module(sm, sm.BaseSurvivalModule),
    key=lambda module_cls: module_cls.name,
)


def module_id(module_cls):
    return module_cls.name


def build_module(module_cls):
    return module_cls(**MODULE_DEFAULTS.get(module_cls, {})).eval()


def raw_params_for(module):
    param_number = module.get_param_number()
    if param_number == 0:
        return torch.empty((3, 0), dtype=torch.float32)

    values = torch.linspace(-0.4, 0.4, param_number, dtype=torch.float32)
    return torch.stack(
        [
            torch.zeros(param_number, dtype=torch.float32),
            values,
            -values,
        ]
    )


def time_outputs(module):
    raw_params = raw_params_for(module)
    return {
        mode: module(mode, raw_params, TIMES)
        for mode in ("failure", "survival", "density", "hazard")
    }


def centered_failure_derivative(failure):
    return (failure[..., 2:] - failure[..., :-2]) / (TIMES[2:] - TIMES[:-2])


def derivative_mask(module):
    interior_times = TIMES[1:-1]
    mask = torch.ones_like(interior_times, dtype=torch.bool)

    if isinstance(module, sm.StepExpSurvivalModule):
        breaks, _ = module._get_time_breaks()
        for breakpoint in breaks[1:]:
            mask &= torch.abs(interior_times - breakpoint) > 0.01

    return mask


@pytest.mark.parametrize("module_cls", MODULE_CLASSES, ids=module_id)
def test_survival_and_failure_are_complements(module_cls):
    outputs = time_outputs(build_module(module_cls))

    assert torch.isfinite(outputs["survival"]).all()
    assert torch.isfinite(outputs["failure"]).all()
    torch.testing.assert_close(
        outputs["survival"] + outputs["failure"],
        torch.ones_like(outputs["survival"]),
        atol=1e-6,
        rtol=1e-6,
    )


@pytest.mark.parametrize("module_cls", MODULE_CLASSES, ids=module_id)
def test_density_matches_failure_derivative(module_cls):
    if module_cls is sm.FractalNoiseSurvivalModule:
        pytest.xfail(
            "FractalNoise density is an interpolated profile, not the derivative "
            "of the interpolated failure curve."
        )

    module = build_module(module_cls)
    outputs = time_outputs(module)
    numeric_density = centered_failure_derivative(outputs["failure"])
    direct_density = outputs["density"][..., 1:-1]
    mask = derivative_mask(module).unsqueeze(0)

    assert torch.isfinite(direct_density).all()
    assert torch.isfinite(numeric_density).all()
    torch.testing.assert_close(
        direct_density[mask.expand_as(direct_density)],
        numeric_density[mask.expand_as(numeric_density)],
        atol=1e-3,
        rtol=1e-2,
    )


@pytest.mark.parametrize("module_cls", MODULE_CLASSES, ids=module_id)
def test_hazard_matches_density_over_survival(module_cls):
    outputs = time_outputs(build_module(module_cls))
    survival = outputs["survival"]
    density = outputs["density"]
    hazard = outputs["hazard"]
    mask = survival > 1e-5

    assert torch.isfinite(density).all()
    assert torch.isfinite(hazard).all()
    torch.testing.assert_close(
        hazard[mask],
        (density / survival)[mask],
        atol=1e-6,
        rtol=1e-5,
    )
