# flake8: noqa
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

from sklearn import naive_bayes

from numpy_ml.linear_models import GaussianNBClassifier
from numpy_ml.utils.testing import random_tensor


def test_GaussianNB(N=10):
    np.random.seed(12345)
    N = np.inf if N is None else N

    i = 1
    eps = np.finfo(float).eps
    while i < N + 1:
        n_ex = np.random.randint(1, 300)
        n_feats = np.random.randint(1, 100)
        n_classes = np.random.randint(2, 10)

        X = random_tensor((n_ex, n_feats), standardize=True)
        y = np.random.randint(0, n_classes, size=n_ex)

        X_test = random_tensor((n_ex, n_feats), standardize=True)

        NB = GaussianNBClassifier(eps=1e-09)
        NB.fit(X, y)

        preds = NB.predict(X_test)

        sklearn_NB = naive_bayes.GaussianNB()
        sklearn_NB.fit(X, y)

        sk_preds = sklearn_NB.predict(X_test)

        for j in range(len(NB.labels)):
            P = NB.parameters
            jointi = np.log(sklearn_NB.class_prior_[j])
            jointi_mine = np.log(P["prior"][j])

            np.testing.assert_almost_equal(jointi, jointi_mine)

            n_jk = -0.5 * np.sum(np.log(2.0 * np.pi * sklearn_NB.sigma_[j, :] + eps))
            n_jk_mine = -0.5 * np.sum(np.log(2.0 * np.pi * P["sigma"][j] + eps))

            np.testing.assert_almost_equal(n_jk_mine, n_jk)

            n_jk2 = n_jk - 0.5 * np.sum(
                ((X_test - sklearn_NB.theta_[j, :]) ** 2) / (sklearn_NB.sigma_[j, :]), 1
            )

            n_jk2_mine = n_jk_mine - 0.5 * np.sum(
                ((X_test - P["mean"][j]) ** 2) / (P["sigma"][j]), 1
            )
            np.testing.assert_almost_equal(n_jk2_mine, n_jk2, decimal=4)

            llh = jointi + n_jk2
            llh_mine = jointi_mine + n_jk2_mine

            np.testing.assert_almost_equal(llh_mine, llh, decimal=4)

        np.testing.assert_almost_equal(P["prior"], sklearn_NB.class_prior_)
        np.testing.assert_almost_equal(P["mean"], sklearn_NB.theta_)
        np.testing.assert_almost_equal(P["sigma"], sklearn_NB.sigma_)
        np.testing.assert_almost_equal(
            sklearn_NB._joint_log_likelihood(X_test),
            NB._log_posterior(X_test),
            decimal=4,
        )
        np.testing.assert_almost_equal(preds, sk_preds)
        print("PASSED")
        i += 1
