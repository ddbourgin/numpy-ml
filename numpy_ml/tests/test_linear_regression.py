# flake8: noqa
import numpy as np

from sklearn.linear_model import LinearRegression as LinearRegressionGold

from numpy_ml.linear_models.lm import LinearRegression
from numpy_ml.utils.testing import random_tensor


def test_linear_regression(N=10):
    np.random.seed(12345)
    N = np.inf if N is None else N

    i = 1
    while i < N + 1:
        train_samples = np.random.randint(2, 30)
        update_samples = np.random.randint(1, 30)
        n_samples = train_samples + update_samples

        # ensure n_feats < train_samples, otherwise multiple solutions are
        # possible
        n_feats = np.random.randint(1, train_samples)
        target_dim = np.random.randint(1, 10)

        fit_intercept = np.random.choice([True, False])

        X = random_tensor((n_samples, n_feats), standardize=True)
        y = random_tensor((n_samples, target_dim), standardize=True)

        X_train, X_update = X[:train_samples], X[train_samples:]
        y_train, y_update = y[:train_samples], y[train_samples:]

        # Fit gold standard model on the entire dataset
        lr_gold = LinearRegressionGold(fit_intercept=fit_intercept, normalize=False)
        lr_gold.fit(X, y)

        # Fit our model on just (X_train, y_train)...
        lr = LinearRegression(fit_intercept=fit_intercept)
        lr.fit(X_train, y_train)

        do_single_sample_update = np.random.choice([True, False])

        # ...then update our model on the examples (X_update, y_update)
        if do_single_sample_update:
            for x_new, y_new in zip(X_update, y_update):
                lr.update(x_new, y_new)
        else:
            lr.update(X_update, y_update)

        # check that model predictions match
        np.testing.assert_almost_equal(lr.predict(X), lr_gold.predict(X), decimal=5)

        # check that model coefficients match
        beta = lr.beta.T[:, 1:] if fit_intercept else lr.beta.T
        np.testing.assert_almost_equal(beta, lr_gold.coef_, decimal=6)

        print("\tPASSED")
        i += 1
