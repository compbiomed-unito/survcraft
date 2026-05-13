import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn

from .util import detect_max_survival_time


def plot_model(model, X, event, time):
    rng = np.random.default_rng()
    sub_idx = rng.choice(X.shape[0], size=5)
    sub_times, sub_events = time[sub_idx], event[sub_idx]
    # max_time = np.max(sub_times[sub_events])
    # max_time = np.max(sub_times)
    max_time = 100
    times = np.linspace(0.0, max_time * 1.1, 200)
    fail = model.predict("failure", X[sub_idx], times)
    params = model.model_.get_raw_params(model._tensor(X[sub_idx]))
    fig, ax = plt.subplots(figsize=(8, 6))
    for t, e, f, p in zip(sub_times, sub_events, fail, params):
        color = ax._get_lines.get_next_color()
        label = f"time={float(t):.3g}"
        plt.plot(times, f, label=label, color=color)
        t = min(np.max(times), t)  # clip times
        plt.axvline(
            t, linestyle="solid" if e else "dotted", label=label, color=color
        )
        print(p.detach())
        # plt.xscale('log')


def plot_outputs(model, X, max_time=None):
    # time = np.linspace(0.0, times_train[events_train].max(), 101)
    if max_time is None:
        max_time = detect_max_survival_time(model, X)
        print("estimated max time:", max_time)  # time[-1].item())
    time = np.linspace(0.0, max_time, 101)

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 7))
    axes[0][0].plot(time, model.predict("failure", X, time).T)
    axes[0][0].set_title("failure")

    axes[0][1].plot(time, model.predict("survival", X, time).T)
    axes[0][1].set_title("survival")

    # decide log or no log
    axes[1][0].plot(time, model.predict("density", X, time).T)
    # axes[1][0].set_yscale("log")
    axes[1][0].set_title("density")

    axes[1][1].plot(time, model.predict("hazard", X, time).T)
    axes[1][1].set_yscale("log")
    axes[1][1].set_title("hazard")

    # axes[2][0].plot(time, model.predict("density", X, time).T)
    # axes[2][0].set_title("density")

    # axes[2][1].plot(time, model.predict("hazard", X, time).T)
    # axes[2][1].set_title("hazard")
    return  # FIXME questi histogrammi non vanno bene
    try:
        axes[2][0].hist(model.predict("expected_time", X), bins=10)
        # axes[2][0].set_yscale('log')
    except NotImplementedError as e:
        pass
    axes[2][0].set_title("expected time")

    try:
        axes[2][1].hist(model.predict("median_time", X), bins=10, density=True)
        # axes[2][1].set_yscale('log')
    except NotImplementedError as e:
        pass
    axes[2][1].set_title("median time")
    # axes[2][0].hist(model.predict('median_time', X_test))
    # axes[2][0].hist(model.predict('median_time', X_test))


def get_training_history_table(model):
    train_losses = pd.DataFrame(
        [[l.item() for l in epoch[0]] for epoch in model.train_history_]
    )
    test_losses = pd.DataFrame(
        [{n: v.item() for n, v in epoch[1].items()} for epoch in model.train_history_]
    )
    return pd.concat({"train": train_losses, "test": test_losses}, axis=1)


def plot_training_history(model):
    train_loss = pd.DataFrame(
        [epoch[0] for epoch in model.train_history_]
    )
    test_losses = pd.DataFrame(
        [{n: v for n, v in epoch[1].items()} for epoch in model.train_history_]
    )
    df = test_losses.copy()
    df["train_loss"] = train_loss.mean(axis=1)
    df = df / df.max()
    plt.figure(figsize=(10, 6))
    seaborn.lineplot(data=df)
    plt.yscale("log")
