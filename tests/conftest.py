import random

import numpy as np
import pytest
import torch

import survcraft.adapters as ad
import survcraft.util


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@pytest.fixture(autouse=True)
def deterministic_seed():
    set_seed(0)


@pytest.fixture(scope="session")
def whas500_data():
    data = survcraft.util.get_test_data()
    return {
        "X_train": data["X_train"][:128].astype(np.float32),
        "X_test": data["X_test"][:64].astype(np.float32),
        "y_train": data["y_train"][:128].copy(),
        "y_test": data["y_test"][:64].copy(),
    }


@pytest.fixture
def simulated_dataset():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(96, 4)).astype(np.float32)
    times = np.linspace(0.1, 3.0, 48, dtype=np.float32)
    simulator = ad.SurvivalSimulator(
        input=ad.LinearFunctionInputAdapter(),
        survival=ad.StepExpSurvivalAdapter(breaks=6),
    )
    y = simulator.simulate(X=X, times=times, seed=0)
    return X, y, times
