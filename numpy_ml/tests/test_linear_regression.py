# flake8: noqa
import numpy as np

from sklearn.linear_model import LinearRegression as LinearRegressionGold

from numpy_ml.linear_models import LinearRegression
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

        weighted = np.random.choice([True, False])
        weights = np.random.rand(n_samples) if weighted else np.ones(n_samples)

        X_train, X_update = X[:train_samples], X[train_samples:]
        y_train, y_update = y[:train_samples], y[train_samples:]
        w_train, w_update = weights[:train_samples], weights[train_samples:]

        print(f"Weights: {weighted}")
        print(f"Fit intercept: {fit_intercept}")

        # Fit gold standard model on the entire dataset
        lr_gold = LinearRegressionGold(fit_intercept=fit_intercept, normalize=False)
        lr_gold.fit(X, y, sample_weight=weights)

        lr_mine = LinearRegression(fit_intercept=fit_intercept)
        lr_mine.fit(X, y, weights=weights)

        # check that model predictions match
        np.testing.assert_almost_equal(
            lr_mine.predict(X), lr_gold.predict(X), decimal=5
        )
        print("\t1. Overall model predictions match")

        # check that model coefficients match
        beta = lr_mine.beta.T[:, 1:] if fit_intercept else lr_mine.beta.T
        np.testing.assert_almost_equal(beta, lr_gold.coef_, decimal=6)
        print("\t2. Overall model coefficients match")

        # Fit our model on just (X_train, y_train)...
        lr = LinearRegression(fit_intercept=fit_intercept)
        lr.fit(X_train, y_train, weights=w_train)

        do_single_sample_update = np.random.choice([True, False])

        # ...then update our model on the examples (X_update, y_update)
        if do_single_sample_update:
            for x_new, y_new, w_new in zip(X_update, y_update, w_update):
                lr.update(x_new, y_new, w_new)
        else:
            lr.update(X_update, y_update, w_update)

        # check that model predictions match
        np.testing.assert_almost_equal(lr.predict(X), lr_gold.predict(X), decimal=5)
        print("\t3. Iterative model predictions match")

        # check that model coefficients match
        beta = lr.beta.T[:, 1:] if fit_intercept else lr.beta.T
        np.testing.assert_almost_equal(beta, lr_gold.coef_, decimal=6)
        print("\t4. Iterative model coefficients match")

        print("\tPASSED\n")
        i += 1
