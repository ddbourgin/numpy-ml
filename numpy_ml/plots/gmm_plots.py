# flake8: noqa
import numpy as np
from sklearn.datasets.samples_generator import make_blobs

from scipy.stats import multivariate_normal

import matplotlib.pyplot as plt
import seaborn as sns

# https://seaborn.pydata.org/generated/seaborn.set_context.html
# https://seaborn.pydata.org/generated/seaborn.set_style.html
sns.set_style("white")
sns.set_context("paper", font_scale=1)

from numpy_ml.gmm import GMM

from matplotlib.colors import ListedColormap


def plot_countour(X, x, y, z, ax, xlim, ylim):
    def fixed_aspect_ratio(ratio, ax):
        """
        Set a fixed aspect ratio on matplotlib plots
        regardless of axis units
        """
        xvals, yvals = ax.get_xlim(), ax.get_ylim()

        xrange = xvals[1] - xvals[0]
        yrange = yvals[1] - yvals[0]
        ax.set_aspect(ratio * (xrange / yrange), adjustable="box")

    # contour the gridded data, plotting dots at the randomly spaced data points.
    ax.contour(x, y, z, 6, linewidths=0.5, colors="k")

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    fixed_aspect_ratio(1, ax)
    return ax


def plot_clusters(model, X, ax):
    C = model.C

    xmin = min(X[:, 0]) - 0.1 * (max(X[:, 0]) - min(X[:, 0]))
    xmax = max(X[:, 0]) + 0.1 * (max(X[:, 0]) - min(X[:, 0]))
    ymin = min(X[:, 1]) - 0.1 * (max(X[:, 1]) - min(X[:, 1]))
    ymax = max(X[:, 1]) + 0.1 * (max(X[:, 1]) - min(X[:, 1]))

    for c in range(C):
        rv = multivariate_normal(model.mu[c], model.sigma[c], allow_singular=True)

        x = np.linspace(xmin, xmax, 500)
        y = np.linspace(ymin, ymax, 500)

        X1, Y1 = np.meshgrid(x, y)
        xy = np.column_stack([X1.flat, Y1.flat])

        # density values at the grid points
        Z = rv.pdf(xy).reshape(X1.shape)
        ax = plot_countour(X, X1, Y1, Z, ax=ax, xlim=(xmin, xmax), ylim=(ymin, ymax))
        ax.plot(model.mu[c, 0], model.mu[c, 1], "ro")

    # plot data points
    cm = ListedColormap(sns.color_palette().as_hex())
    labels = model.Q.argmax(1)
    uniq = set(labels)
    for i in uniq:
        ax.scatter(X[labels == i, 0], X[labels == i, 1], c=cm.colors[i - 1], s=30)
    return ax


def plot():
    fig, axes = plt.subplots(4, 4)
    fig.set_size_inches(10, 10)
    for i, ax in enumerate(axes.flatten()):
        n_ex = 150
        n_in = 2
        n_classes = np.random.randint(2, 4)
        X, y = make_blobs(
            n_samples=n_ex, centers=n_classes, n_features=n_in, random_state=i
        )
        X -= X.mean(axis=0)

        # take best fit over 10 runs
        best_elbo = -np.inf
        for k in range(10):
            _G = GMM(C=n_classes, seed=i * 3)
            ret = _G.fit(X, max_iter=100, verbose=False)
            while ret != 0:
                print("Components collapsed; Refitting")
                ret = _G.fit(X, max_iter=100, verbose=False)

            if _G.best_elbo > best_elbo:
                best_elbo = _G.best_elbo
                G = _G

        ax = plot_clusters(G, X, ax)
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        ax.set_title("# Classes: {}; Final VLB: {:.2f}".format(n_classes, G.best_elbo))

    plt.tight_layout()
    plt.savefig("img/plot.png", dpi=300)
    plt.close("all")
