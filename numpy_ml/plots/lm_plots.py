# flake8: noqa
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.datasets.samples_generator import make_blobs
from sklearn.linear_model import LogisticRegression as LogisticRegression_sk
from sklearn.datasets import make_regression
from sklearn.metrics import zero_one_loss

import matplotlib.pyplot as plt

import seaborn as sns

# https://seaborn.pydata.org/generated/seaborn.set_context.html
# https://seaborn.pydata.org/generated/seaborn.set_style.html
sns.set_style("white")
sns.set_context("paper", font_scale=0.5)


from numpy_ml.linear_models import (
    RidgeRegression,
    LinearRegression,
    BayesianLinearRegressionKnownVariance,
    BayesianLinearRegressionUnknownVariance,
    LogisticRegression,
)

#######################################################################
#                           Data Generators                           #
#######################################################################


def random_binary_tensor(shape, sparsity=0.5):
    X = (np.random.rand(*shape) >= (1 - sparsity)).astype(float)
    return X


def random_regression_problem(n_ex, n_in, n_out, intercept=0, std=1, seed=0):
    X, y, coef = make_regression(
        n_samples=n_ex,
        n_features=n_in,
        n_targets=n_out,
        bias=intercept,
        noise=std,
        coef=True,
        random_state=seed,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=seed
    )
    return X_train, y_train, X_test, y_test, coef


def random_classification_problem(n_ex, n_classes, n_in, seed=0):
    X, y = make_blobs(
        n_samples=n_ex, centers=n_classes, n_features=n_in, random_state=seed
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=seed
    )
    return X_train, y_train, X_test, y_test


#######################################################################
#                                Plots                                #
#######################################################################


def plot_logistic():
    np.random.seed(12345)

    fig, axes = plt.subplots(4, 4)
    for i, ax in enumerate(axes.flatten()):
        n_in = 1
        n_ex = 150
        X_train, y_train, X_test, y_test = random_classification_problem(
            n_ex, n_classes=2, n_in=n_in, seed=i
        )
        LR = LogisticRegression(penalty="l2", gamma=0.2, fit_intercept=True)
        LR.fit(X_train, y_train, lr=0.1, tol=1e-7, max_iter=1e7)
        y_pred = (LR.predict(X_test) >= 0.5) * 1.0
        loss = zero_one_loss(y_test, y_pred) * 100.0

        LR_sk = LogisticRegression_sk(
            penalty="l2", tol=0.0001, C=0.8, fit_intercept=True, random_state=i
        )
        LR_sk.fit(X_train, y_train)
        y_pred_sk = (LR_sk.predict(X_test) >= 0.5) * 1.0
        loss_sk = zero_one_loss(y_test, y_pred_sk) * 100.0

        xmin = min(X_test) - 0.1 * (max(X_test) - min(X_test))
        xmax = max(X_test) + 0.1 * (max(X_test) - min(X_test))
        X_plot = np.linspace(xmin, xmax, 100)
        y_plot = LR.predict(X_plot)
        y_plot_sk = LR_sk.predict_proba(X_plot.reshape(-1, 1))[:, 1]

        ax.scatter(X_test[y_pred == 0], y_test[y_pred == 0], alpha=0.5)
        ax.scatter(X_test[y_pred == 1], y_test[y_pred == 1], alpha=0.5)
        ax.plot(X_plot, y_plot, label="mine", alpha=0.75)
        ax.plot(X_plot, y_plot_sk, label="sklearn", alpha=0.75)
        ax.legend()
        ax.set_title("Loss mine: {:.2f} Loss sklearn: {:.2f}".format(loss, loss_sk))

        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])

    plt.tight_layout()
    plt.savefig("plot_logistic.png", dpi=300)
    plt.close("all")


def plot_bayes():
    np.random.seed(12345)
    n_in = 1
    n_out = 1
    n_ex = 20
    std = 15
    intercept = 10
    X_train, y_train, X_test, y_test, coefs = random_regression_problem(
        n_ex, n_in, n_out, intercept=intercept, std=std, seed=0
    )

    # add some outliers
    x1, x2 = X_train[0] + 0.5, X_train[6] - 0.3
    y1 = np.dot(x1, coefs) + intercept + 25
    y2 = np.dot(x2, coefs) + intercept - 31
    X_train = np.vstack([X_train, np.array([x1, x2])])
    y_train = np.hstack([y_train, [y1[0], y2[0]]])

    LR = LinearRegression(fit_intercept=True)
    LR.fit(X_train, y_train)
    y_pred = LR.predict(X_test)
    loss = np.mean((y_test - y_pred) ** 2)

    ridge = RidgeRegression(alpha=1, fit_intercept=True)
    ridge.fit(X_train, y_train)
    y_pred = ridge.predict(X_test)
    loss_ridge = np.mean((y_test - y_pred) ** 2)

    LR_var = BayesianLinearRegressionKnownVariance(
        b_mean=np.c_[intercept, coefs][0],
        b_sigma=np.sqrt(std),
        b_V=None,
        fit_intercept=True,
    )
    LR_var.fit(X_train, y_train)
    y_pred_var = LR_var.predict(X_test)
    loss_var = np.mean((y_test - y_pred_var) ** 2)

    LR_novar = BayesianLinearRegressionUnknownVariance(
        alpha=1, beta=2, b_mean=np.c_[intercept, coefs][0], b_V=None, fit_intercept=True
    )
    LR_novar.fit(X_train, y_train)
    y_pred_novar = LR_novar.predict(X_test)
    loss_novar = np.mean((y_test - y_pred_novar) ** 2)

    xmin = min(X_test) - 0.1 * (max(X_test) - min(X_test))
    xmax = max(X_test) + 0.1 * (max(X_test) - min(X_test))
    X_plot = np.linspace(xmin, xmax, 100)
    y_plot = LR.predict(X_plot)
    y_plot_ridge = ridge.predict(X_plot)
    y_plot_var = LR_var.predict(X_plot)
    y_plot_novar = LR_novar.predict(X_plot)

    y_true = [np.dot(x, coefs) + intercept for x in X_plot]
    fig, axes = plt.subplots(1, 4)

    axes = axes.flatten()
    axes[0].scatter(X_test, y_test)
    axes[0].plot(X_plot, y_plot, label="MLE")
    axes[0].plot(X_plot, y_true, label="True fn")
    axes[0].set_title("Linear Regression\nMLE Test MSE: {:.2f}".format(loss))
    axes[0].legend()
    #  axes[0].fill_between(X_plot, y_plot - error, y_plot + error)

    axes[1].scatter(X_test, y_test)
    axes[1].plot(X_plot, y_plot_ridge, label="MLE")
    axes[1].plot(X_plot, y_true, label="True fn")
    axes[1].set_title(
        "Ridge Regression (alpha=1)\nMLE Test MSE: {:.2f}".format(loss_ridge)
    )
    axes[1].legend()
    print("plotted ridge.. {:.2f} MSE".format(loss_ridge))

    axes[2].plot(X_plot, y_plot_var, label="MAP")
    mu, cov = LR_var.posterior["b"]["mu"], LR_var.posterior["b"]["cov"]
    for k in range(200):
        b_samp = np.random.multivariate_normal(mu, cov)
        y_samp = [np.dot(x, b_samp[1]) + b_samp[0] for x in X_plot]
        axes[2].plot(X_plot, y_samp, c="green", alpha=0.05)
    axes[2].scatter(X_test, y_test)
    axes[2].plot(X_plot, y_true, label="True fn")
    axes[2].legend()
    axes[2].set_title(
        "Bayesian Regression (known variance)\nMAP Test MSE: {:.2f}".format(loss_var)
    )

    axes[3].plot(X_plot, y_plot_novar, label="MAP")
    mu = LR_novar.posterior["b | sigma**2"]["mu"]
    cov = LR_novar.posterior["b | sigma**2"]["cov"]
    for k in range(200):
        b_samp = np.random.multivariate_normal(mu, cov)
        y_samp = [np.dot(x, b_samp[1]) + b_samp[0] for x in X_plot]
        axes[3].plot(X_plot, y_samp, c="green", alpha=0.05)
    axes[3].scatter(X_test, y_test)
    axes[3].plot(X_plot, y_true, label="True fn")
    axes[3].legend()
    axes[3].set_title(
        "Bayesian Regression (unknown variance)\nMAP Test MSE: {:.2f}".format(
            loss_novar
        )
    )

    for ax in axes:
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])

    #  plt.tight_layout()
    fig.set_size_inches(10, 2.5)
    plt.savefig("plot_bayes.png", dpi=300)
    plt.close("all")


def plot_regression():
    np.random.seed(12345)

    fig, axes = plt.subplots(4, 4)
    for i, ax in enumerate(axes.flatten()):
        n_in = 1
        n_out = 1
        n_ex = 50
        std = np.random.randint(0, 100)
        intercept = np.random.rand() * np.random.randint(-300, 300)
        X_train, y_train, X_test, y_test, coefs = random_regression_problem(
            n_ex, n_in, n_out, intercept=intercept, std=std, seed=i
        )

        LR = LinearRegression(fit_intercept=True)
        LR.fit(X_train, y_train)
        y_pred = LR.predict(X_test)
        loss = np.mean((y_test - y_pred) ** 2)

        LR_var = BayesianLinearRegressionKnownVariance(
            b_mean=np.c_[intercept, coefs][0],
            b_sigma=np.sqrt(std),
            b_V=None,
            fit_intercept=True,
        )
        LR_var.fit(X_train, y_train)
        y_pred_var = LR_var.predict(X_test)
        loss_var = np.mean((y_test - y_pred_var) ** 2)

        LR_novar = BayesianLinearRegressionUnknownVariance(
            alpha=1,
            beta=2,
            b_mean=np.c_[intercept, coefs][0],
            b_V=None,
            fit_intercept=True,
        )
        LR_novar.fit(X_train, y_train)
        y_pred_novar = LR_novar.predict(X_test)
        loss_novar = np.mean((y_test - y_pred_novar) ** 2)

        xmin = min(X_test) - 0.1 * (max(X_test) - min(X_test))
        xmax = max(X_test) + 0.1 * (max(X_test) - min(X_test))
        X_plot = np.linspace(xmin, xmax, 100)
        y_plot = LR.predict(X_plot)
        y_plot_var = LR_var.predict(X_plot)
        y_plot_novar = LR_novar.predict(X_plot)

        ax.scatter(X_test, y_test, alpha=0.5)
        ax.plot(X_plot, y_plot, label="linear regression", alpha=0.5)
        ax.plot(X_plot, y_plot_var, label="Bayes (w var)", alpha=0.5)
        ax.plot(X_plot, y_plot_novar, label="Bayes (no var)", alpha=0.5)
        ax.legend()
        ax.set_title(
            "MSE\nLR: {:.2f} Bayes (w var): {:.2f}\nBayes (no var): {:.2f}".format(
                loss, loss_var, loss_novar
            )
        )

        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])

    plt.tight_layout()
    plt.savefig("plot_regression.png", dpi=300)
    plt.close("all")
