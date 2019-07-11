import numpy as np
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import seaborn as sns

# https://seaborn.pydata.org/generated/seaborn.set_context.html
# https://seaborn.pydata.org/generated/seaborn.set_style.html
sns.set_style("white")
sns.set_context("notebook", font_scale=0.7)

from activations import Affine, ReLU, LeakyReLU, Tanh, Sigmoid, ELU


def plot_activations():
    fig, axes = plt.subplots(2, 3, sharex=True, sharey=True)
    fns = [Affine(), Tanh(), Sigmoid(), ReLU(), LeakyReLU(), ELU()]
    for ax, fn in zip(axes.flatten(), fns):
        X = np.linspace(-3, 3, 100).astype(float).reshape(100, 1)
        ax.plot(X, fn(X), label=r"$y$", alpha=0.7)
        ax.plot(X, fn.grad(X), label=r"$\frac{dy}{dx}$", alpha=0.7)
        ax.plot(X, fn.grad2(X), label=r"$\frac{d^2 y}{dx^2}$", alpha=0.7)
        ax.hlines(0, -3, 3, lw=1, linestyles="dashed", color="k")
        ax.vlines(0, -1.2, 1.2, lw=1, linestyles="dashed", color="k")
        ax.set_ylim(-1.1, 1.1)
        ax.set_xlim(-3, 3)
        ax.set_xticks([])
        ax.set_yticks([-1, 0, 1])
        ax.xaxis.set_visible(False)
        #  ax.yaxis.set_visible(False)
        ax.set_title("{}".format(fn))
        ax.legend(frameon=False)
        sns.despine(left=True, bottom=True)

    fig.set_size_inches(8, 5)
    plt.tight_layout()
    plt.savefig("plot.png", dpi=300)
    plt.close("all")


if __name__ == "__main__":
    plot_activations()
