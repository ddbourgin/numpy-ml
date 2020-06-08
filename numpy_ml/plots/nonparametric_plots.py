# flake8: noqa
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

# https://seaborn.pydata.org/generated/seaborn.set_context.html
# https://seaborn.pydata.org/generated/seaborn.set_style.html
sns.set_style("white")
sns.set_context("paper", font_scale=0.5)

from numpy_ml.nonparametric import GPRegression, KNN, KernelRegression
from numpy_ml.linear_models.lm import LinearRegression

from sklearn.model_selection import train_test_split


def random_regression_problem(n_ex, n_in, n_out, d=3, intercept=0, std=1, seed=0):
    coef = np.random.uniform(0, 50, size=d)
    coef[-1] = intercept

    y = []
    X = np.random.uniform(-100, 100, size=(n_ex, n_in))
    for x in X:
        val = np.polyval(coef, x) + np.random.normal(0, std)
        y.append(val)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=seed
    )
    return X_train, y_train, X_test, y_test, coef


def plot_regression():
    np.random.seed(12345)
    fig, axes = plt.subplots(4, 4)
    for i, ax in enumerate(axes.flatten()):
        n_in = 1
        n_out = 1
        d = np.random.randint(1, 5)
        n_ex = np.random.randint(5, 500)
        std = np.random.randint(0, 1000)
        intercept = np.random.rand() * np.random.randint(-300, 300)
        X_train, y_train, X_test, y_test, coefs = random_regression_problem(
            n_ex, n_in, n_out, d=d, intercept=intercept, std=std, seed=i
        )

        LR = LinearRegression(fit_intercept=True)
        LR.fit(X_train, y_train)
        y_pred = LR.predict(X_test)
        loss = np.mean((y_test.flatten() - y_pred.flatten()) ** 2)

        d = 3
        best_loss = np.inf
        for gamma in np.linspace(1e-10, 1, 100):
            for c0 in np.linspace(-1, 1000, 100):
                kernel = "PolynomialKernel(d={}, gamma={}, c0={})".format(d, gamma, c0)
                KR_poly = KernelRegression(kernel=kernel)
                KR_poly.fit(X_train, y_train)
                y_pred_poly = KR_poly.predict(X_test)
                loss_poly = np.mean((y_test.flatten() - y_pred_poly.flatten()) ** 2)
                if loss_poly <= best_loss:
                    KR_poly_best = kernel
                    best_loss = loss_poly

        print("Best kernel: {} || loss: {:.4f}".format(KR_poly_best, best_loss))
        KR_poly = KernelRegression(kernel=KR_poly_best)
        KR_poly.fit(X_train, y_train)

        KR_rbf = KernelRegression(kernel="RBFKernel(sigma=1)")
        KR_rbf.fit(X_train, y_train)
        y_pred_rbf = KR_rbf.predict(X_test)
        loss_rbf = np.mean((y_test.flatten() - y_pred_rbf.flatten()) ** 2)

        xmin = min(X_test) - 0.1 * (max(X_test) - min(X_test))
        xmax = max(X_test) + 0.1 * (max(X_test) - min(X_test))
        X_plot = np.linspace(xmin, xmax, 100)
        y_plot = LR.predict(X_plot)
        y_plot_poly = KR_poly.predict(X_plot)
        y_plot_rbf = KR_rbf.predict(X_plot)

        ax.scatter(X_test, y_test, alpha=0.5)
        ax.plot(X_plot, y_plot, label="OLS", alpha=0.5)
        ax.plot(
            X_plot, y_plot_poly, label="KR (poly kernel, d={})".format(d), alpha=0.5
        )
        ax.plot(X_plot, y_plot_rbf, label="KR (rbf kernel)", alpha=0.5)
        ax.legend()
        #  ax.set_title(
        #      "MSE\nLR: {:.2f} KR (poly): {:.2f}\nKR (rbf): {:.2f}".format(
        #          loss, loss_poly, loss_rbf
        #      )
        #  )

        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])

    plt.tight_layout()
    plt.savefig("img/kr_plots.png", dpi=300)
    plt.close("all")


def plot_knn():
    np.random.seed(12345)
    fig, axes = plt.subplots(4, 4)
    for i, ax in enumerate(axes.flatten()):
        n_in = 1
        n_out = 1
        d = np.random.randint(1, 5)
        n_ex = np.random.randint(5, 500)
        std = np.random.randint(0, 1000)
        intercept = np.random.rand() * np.random.randint(-300, 300)
        X_train, y_train, X_test, y_test, coefs = random_regression_problem(
            n_ex, n_in, n_out, d=d, intercept=intercept, std=std, seed=i
        )

        LR = LinearRegression(fit_intercept=True)
        LR.fit(X_train, y_train)
        y_pred = LR.predict(X_test)
        loss = np.mean((y_test.flatten() - y_pred.flatten()) ** 2)

        knn_1 = KNN(k=1, classifier=False, leaf_size=10, weights="uniform")
        knn_1.fit(X_train, y_train)
        y_pred_1 = knn_1.predict(X_test)
        loss_1 = np.mean((y_test.flatten() - y_pred_1.flatten()) ** 2)

        knn_5 = KNN(k=5, classifier=False, leaf_size=10, weights="uniform")
        knn_5.fit(X_train, y_train)
        y_pred_5 = knn_5.predict(X_test)
        loss_5 = np.mean((y_test.flatten() - y_pred_5.flatten()) ** 2)

        knn_10 = KNN(k=10, classifier=False, leaf_size=10, weights="uniform")
        knn_10.fit(X_train, y_train)
        y_pred_10 = knn_10.predict(X_test)
        loss_10 = np.mean((y_test.flatten() - y_pred_10.flatten()) ** 2)

        xmin = min(X_test) - 0.1 * (max(X_test) - min(X_test))
        xmax = max(X_test) + 0.1 * (max(X_test) - min(X_test))
        X_plot = np.linspace(xmin, xmax, 100)
        y_plot = LR.predict(X_plot)
        y_plot_1 = knn_1.predict(X_plot)
        y_plot_5 = knn_5.predict(X_plot)
        y_plot_10 = knn_10.predict(X_plot)

        ax.scatter(X_test, y_test, alpha=0.5)
        ax.plot(X_plot, y_plot, label="OLS", alpha=0.5)
        ax.plot(X_plot, y_plot_1, label="KNN (k=1)", alpha=0.5)
        ax.plot(X_plot, y_plot_5, label="KNN (k=5)", alpha=0.5)
        ax.plot(X_plot, y_plot_10, label="KNN (k=10)", alpha=0.5)
        ax.legend()
        #  ax.set_title(
        #      "MSE\nLR: {:.2f} KR (poly): {:.2f}\nKR (rbf): {:.2f}".format(
        #          loss, loss_poly, loss_rbf
        #      )
        #  )

        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])

    plt.tight_layout()
    plt.savefig("img/knn_plots.png", dpi=300)
    plt.close("all")


def plot_gp():
    np.random.seed(12345)
    sns.set_context("paper", font_scale=0.65)

    X_test = np.linspace(-10, 10, 100)
    X_train = np.array([-3, 0, 7, 1, -9])
    y_train = np.sin(X_train)

    fig, axes = plt.subplots(2, 2)
    alphas = [0, 1e-10, 1e-5, 1]
    for ix, (ax, alpha) in enumerate(zip(axes.flatten(), alphas)):
        G = GPRegression(kernel="RBFKernel", alpha=alpha)
        G.fit(X_train, y_train)
        y_pred, conf = G.predict(X_test)

        ax.plot(X_train, y_train, "rx", label="observed")
        ax.plot(X_test, np.sin(X_test), label="true fn")
        ax.plot(X_test, y_pred, "--", label="MAP (alpha={})".format(alpha))
        ax.fill_between(X_test, y_pred + conf, y_pred - conf, alpha=0.1)
        ax.set_xticks([])
        ax.set_yticks([])
        sns.despine()

        ax.legend()

    plt.tight_layout()
    plt.savefig("img/gp_alpha.png", dpi=300)
    plt.close("all")


def plot_gp_dist():
    np.random.seed(12345)
    sns.set_context("paper", font_scale=0.95)

    X_test = np.linspace(-10, 10, 100)
    X_train = np.array([-3, 0, 7, 1, -9])
    y_train = np.sin(X_train)

    fig, axes = plt.subplots(1, 3)
    G = GPRegression(kernel="RBFKernel", alpha=0)
    G.fit(X_train, y_train)

    y_pred_prior = G.sample(X_test, 3, "prior")
    y_pred_posterior = G.sample(X_test, 3, "posterior_predictive")

    for prior_sample in y_pred_prior:
        axes[0].plot(X_test, prior_sample.ravel(), lw=1)
    axes[0].set_title("Prior samples")
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    for post_sample in y_pred_posterior:
        axes[1].plot(X_test, post_sample.ravel(), lw=1)
    axes[1].plot(X_train, y_train, "ko", ms=1.2)
    axes[1].set_title("Posterior samples")
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    y_pred, conf = G.predict(X_test)

    axes[2].plot(X_test, np.sin(X_test), lw=1, label="true function")
    axes[2].plot(X_test, y_pred, lw=1, label="MAP estimate")
    axes[2].fill_between(X_test, y_pred + conf, y_pred - conf, alpha=0.1)
    axes[2].plot(X_train, y_train, "ko", ms=1.2, label="observed")
    axes[2].legend(fontsize="x-small")
    axes[2].set_title("Posterior mean")
    axes[2].set_xticks([])
    axes[2].set_yticks([])

    fig.set_size_inches(6, 2)
    plt.tight_layout()
    plt.savefig("img/gp_dist.png", dpi=300)
    plt.close("all")
