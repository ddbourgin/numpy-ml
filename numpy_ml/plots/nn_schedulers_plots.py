# flake8: noqa

import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# https://seaborn.pydata.org/generated/seaborn.set_context.html
# https://seaborn.pydata.org/generated/seaborn.set_style.html
sns.set_style("white")
sns.set_context("notebook", font_scale=0.7)

from numpy_ml.neural_nets.schedulers import (
    ConstantScheduler,
    ExponentialScheduler,
    NoamScheduler,
    KingScheduler,
)


def king_loss_fn(x):
    if x <= 250:
        return -0.25 * x + 82.50372665317208
    elif 250 < x <= 600:
        return 20.00372665317208
    elif 600 < x <= 700:
        return -0.2 * x + 140.00372665317207
    else:
        return 0.003726653172066108


def plot_schedulers():
    fig, axes = plt.subplots(2, 2)
    schedulers = [
        (
            [ConstantScheduler(lr=0.01), "lr=1e-2"],
            [ConstantScheduler(lr=0.008), "lr=8e-3"],
            [ConstantScheduler(lr=0.006), "lr=6e-3"],
            [ConstantScheduler(lr=0.004), "lr=4e-3"],
            [ConstantScheduler(lr=0.002), "lr=2e-3"],
        ),
        (
            [
                ExponentialScheduler(
                    lr=0.01, stage_length=250, staircase=False, decay=0.4
                ),
                "lr=0.01, stage=250, stair=False, decay=0.4",
            ],
            [
                ExponentialScheduler(
                    lr=0.01, stage_length=250, staircase=True, decay=0.4
                ),
                "lr=0.01, stage=250, stair=True, decay=0.4",
            ],
            [
                ExponentialScheduler(
                    lr=0.01, stage_length=125, staircase=True, decay=0.1
                ),
                "lr=0.01, stage=125, stair=True, decay=0.1",
            ],
            [
                ExponentialScheduler(
                    lr=0.001, stage_length=250, staircase=False, decay=0.1
                ),
                "lr=0.001, stage=250, stair=False, decay=0.1",
            ],
            [
                ExponentialScheduler(
                    lr=0.001, stage_length=125, staircase=False, decay=0.8
                ),
                "lr=0.001, stage=125, stair=False, decay=0.8",
            ],
            [
                ExponentialScheduler(
                    lr=0.01, stage_length=250, staircase=False, decay=0.01
                ),
                "lr=0.01, stage=250, stair=False, decay=0.01",
            ],
        ),
        (
            [
                NoamScheduler(model_dim=512, scale_factor=1, warmup_steps=250),
                "dim=512, scale=1, warmup=250",
            ],
            [
                NoamScheduler(model_dim=256, scale_factor=1, warmup_steps=250),
                "dim=256, scale=1, warmup=250",
            ],
            [
                NoamScheduler(model_dim=512, scale_factor=1, warmup_steps=500),
                "dim=512, scale=1, warmup=500",
            ],
            [
                NoamScheduler(model_dim=256, scale_factor=1, warmup_steps=500),
                "dim=512, scale=1, warmup=500",
            ],
            [
                NoamScheduler(model_dim=512, scale_factor=2, warmup_steps=500),
                "dim=512, scale=2, warmup=500",
            ],
            [
                NoamScheduler(model_dim=512, scale_factor=0.5, warmup_steps=500),
                "dim=512, scale=0.5, warmup=500",
            ],
        ),
        (
            #  [
            #      KingScheduler(initial_lr=0.01, patience=100, decay=0.1),
            #      "lr=0.01, patience=100, decay=0.8",
            #  ],
            #  [
            #      KingScheduler(initial_lr=0.01, patience=300, decay=0.999),
            #      "lr=0.01, patience=300, decay=0.999",
            #  ],
            [
                KingScheduler(initial_lr=0.009, patience=150, decay=0.995),
                "lr=0.009, patience=150, decay=0.9999",
            ],
            [
                KingScheduler(initial_lr=0.008, patience=100, decay=0.995),
                "lr=0.008, patience=100, decay=0.995",
            ],
            [
                KingScheduler(initial_lr=0.007, patience=50, decay=0.995),
                "lr=0.007, patience=50, decay=0.995",
            ],
            [
                KingScheduler(initial_lr=0.005, patience=25, decay=0.9),
                "lr=0.005, patience=25, decay=0.99",
            ],
        ),
    ]

    for ax, schs, title in zip(
        axes.flatten(), schedulers, ["Constant", "Exponential", "Noam", "King"]
    ):
        t0 = time.time()
        print("Running {} scheduler".format(title))
        X = np.arange(1, 1000)
        loss = np.array([king_loss_fn(x) for x in X])

        # scale loss to fit on same axis as lr
        scale = 0.01 / loss[0]
        loss *= scale

        if title == "King":
            ax.plot(X, loss, ls=":", label="Loss")

        for sc, lg in schs:
            Y = np.array([sc(x, ll) for x, ll in zip(X, loss)])
            ax.plot(X, Y, label=lg, alpha=0.6)

        ax.legend(fontsize=5)
        ax.set_xlabel("Steps")
        ax.set_ylabel("Learning rate")
        ax.set_title("{} scheduler".format(title))
        print(
            "Finished plotting {} runs of {} in {:.2f}s".format(
                len(schs), title, time.time() - t0
            )
        )

    plt.tight_layout()
    plt.savefig("plot.png", dpi=300)
    plt.close("all")


if __name__ == "__main__":
    plot_schedulers()
