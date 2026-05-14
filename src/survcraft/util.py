import numpy as np
import torch
import inspect
from . import adapters, loss_modules, survival_modules#, input_modules

def get_subclasses_in_module(module, base_class, include_abstract=False):
    subclasses = []
    for _, obj in inspect.getmembers(module, inspect.isclass):
        if obj.__module__ == module.__name__ and issubclass(obj, base_class) and obj is not base_class:
            if include_abstract or not inspect.isabstract(obj):
                subclasses.append(obj)

    return subclasses

#def get_input_adapters():
#    return get_subclasses_in_module(adapters, adapters.BaseInputAdapter)

def get_survival_adapters():
    return get_subclasses_in_module(adapters, adapters.BaseSurvivalAdapter)

def get_loss_modules():
    return get_subclasses_in_module(loss_modules, loss_modules.BaseSurvivalLoss)

def get_quantiles(x, n, drop_extremes=True):
    mod = torch if isinstance(x, torch.Tensor) else np

    qt = mod.quantile(x, mod.linspace(0.0, 1.0, n))
    if drop_extremes:
        qt = qt[1:-1]
    return qt


def detect_max_survival_time(model, X, tol=1e-2, q=0.5):
    min_time_log = -6.0
    max_time_log = 6.0
    for _ in range(3):
        time = np.logspace(min_time_log, max_time_log, 50)
        survs = model.predict("survival", X, time)
        if hasattr(np, "quantile"):
            qsurvs = np.quantile(survs, q, dim=0)
        else:
            qsurvs = np.quantile(survs, q, axis=0)
        max_time_idx = np.argmin((qsurvs - tol).abs())
        min_time_log = np.log10(time[max_time_idx - 1])
        max_time_log = np.log10(time[max_time_idx])
    return np.pow(10.0, max_time_log)


def get_test_data():
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import RobustScaler, OneHotEncoder
    import sksurv.datasets

    X, y = sksurv.datasets.load_whas500()
    #X, y = sksurv.datasets.load_flchain()
    #X = OneHotEncoder().fit_transform(X)
    #X, y = sksurv.datasets.load_gbsg2()  # this y seems to have events as false...
    
    y.dtype = np.dtype([("event", "?"), ("time", "<f8")])
    (
        X_train,
        X_test,
        y_train,
        y_test,
        time_train,
        time_test,
        event_train,
        event_test,
    ) = train_test_split(X.astype(float), y, y["time"], y["event"], stratify=y["event"], random_state=0)
    scaler = RobustScaler()
    X_train_norm = scaler.fit_transform(X_train)
    X_test_norm = scaler.transform(X_test)
    return dict(
        X_train=X_train_norm,
        X_test=X_test_norm,
        y_train=y_train,
        y_test=y_test,
        time_train=time_train,
        time_test=time_test,
        event_train=event_train,
        event_test=event_test,
    )

def replace_zero_times(times):
    """Replace zero times with the lowest non-zero value."""
    min_non_zero = np.min(times[times > 0])
    return np.where(times == 0, min_non_zero, times)
