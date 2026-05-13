import pytest
import torch

from survcraft import survival_modules as sm
from survcraft.util import get_subclasses_in_module


TIME_MODES = ("failure", "survival", "density", "hazard")
FIXED_MODES = ("expected_time", "median_time", "risk")
SCALAR_TIME = torch.tensor(0.5, dtype=torch.float32)
VECTOR_TIMES = torch.tensor([0.25, 0.5, 1.0], dtype=torch.float32)
STEP_EXP_BREAKS = torch.tensor([0.0, 0.5, 1.0, 2.0], dtype=torch.float32)


MODULE_DEFAULTS = {
    sm.StepExpSurvivalModule: {
        "breaks": STEP_EXP_BREAKS,
    },
    sm.ProportionalHazardSurvivalModule: {
        "baseline": sm.ExponentialSurvivalModule(),
    },
    sm.AcceleratedFailureTimeSurvivalModule: {
        "baseline": sm.ExponentialSurvivalModule(),
    },
    sm.MixtureSurvivalModule: {
        "baselines": [sm.ExponentialSurvivalModule(), sm.WeibullSurvivalModule()],
    },
}


MANUAL_MODULE_CLASSES = sorted(
    [
        sm.ExponentialSurvivalModule,
        sm.WeibullSurvivalModule,
        sm.LogNormalSurvivalModule,
        sm.LevySurvivalModule,
        sm.InverseGaussianSurvivalModule,
        sm.StepExpSurvivalModule,
        sm.ProportionalHazardSurvivalModule,
        sm.AcceleratedFailureTimeSurvivalModule,
        sm.MixtureSurvivalModule,
        sm.FractalNoiseSurvivalModule,
    ],
    key=lambda module_cls: module_cls.name,
)


DISCOVERED_MODULE_CLASSES = sorted(
    get_subclasses_in_module(sm, sm.BaseSurvivalModule),
    key=lambda module_cls: module_cls.name,
)


def module_id(module_cls):
    return module_cls.name


def build_module(module_cls):
    return module_cls(**MODULE_DEFAULTS.get(module_cls, {}))


def assert_time_mode_contract(module, mode):
    raw_params = torch.zeros((3, module.get_param_number()), dtype=torch.float32)

    scalar = module(mode, raw_params, SCALAR_TIME)
    vector = module(mode, raw_params, VECTOR_TIMES)

    assert scalar.shape == (3,)
    assert vector.shape == (3, len(VECTOR_TIMES))
    assert torch.isfinite(scalar).all()
    assert torch.isfinite(vector).all()


def assert_fixed_mode_contract(module, mode):
    raw_params = torch.zeros((4, module.get_param_number()), dtype=torch.float32)

    try:
        out = module(mode, raw_params)
    except NotImplementedError:
        return

    assert out.shape == (4,)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("module_cls", MANUAL_MODULE_CLASSES, ids=module_id)
@pytest.mark.parametrize("mode", TIME_MODES)
def test_time_modes_support_scalar_and_vector_times(module_cls, mode):
    assert_time_mode_contract(build_module(module_cls), mode)


@pytest.mark.parametrize("module_cls", MANUAL_MODULE_CLASSES, ids=module_id)
@pytest.mark.parametrize("mode", FIXED_MODES)
def test_fixed_modes_return_batch_vectors_or_raise_not_implemented(module_cls, mode):
    assert_fixed_mode_contract(build_module(module_cls), mode)


def test_automatic_discovery_matches_manual_module_list():
    assert [module_cls.name for module_cls in DISCOVERED_MODULE_CLASSES] == [
        module_cls.name for module_cls in MANUAL_MODULE_CLASSES
    ]


@pytest.mark.parametrize("module_cls", DISCOVERED_MODULE_CLASSES, ids=module_id)
@pytest.mark.parametrize("mode", TIME_MODES)
def test_discovered_modules_support_time_mode_contracts(module_cls, mode):
    assert_time_mode_contract(build_module(module_cls), mode)


@pytest.mark.parametrize("module_cls", DISCOVERED_MODULE_CLASSES, ids=module_id)
@pytest.mark.parametrize("mode", FIXED_MODES)
def test_discovered_modules_handle_fixed_modes_gracefully(module_cls, mode):
    assert_fixed_mode_contract(build_module(module_cls), mode)
