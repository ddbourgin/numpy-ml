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

        for i in range(len(NB.labels)):
            P = NB.parameters
            jointi = np.log(sklearn_NB.class_prior_[i])
            jointi_mine = np.log(P["prior"][i])

            np.testing.assert_almost_equal(jointi, jointi_mine)

            n_ij = -0.5 * np.sum(np.log(2.0 * np.pi * sklearn_NB.sigma_[i, :]))
            n_ij_mine = -0.5 * np.sum(np.log(2.0 * np.pi * P["sigma"][i]))

            np.testing.assert_almost_equal(n_ij_mine, n_ij)

            n_ij2 = n_ij - 0.5 * np.sum(
                ((X_test - sklearn_NB.theta_[i, :]) ** 2) / (sklearn_NB.sigma_[i, :]), 1
            )

            n_ij2_mine = n_ij_mine - 0.5 * np.sum(
                ((X_test - P["mean"][i]) ** 2) / (P["sigma"][i]), 1
            )
            np.testing.assert_almost_equal(n_ij2_mine, n_ij2, decimal=4)

            llh = jointi + n_ij2
            llh_mine = jointi_mine + n_ij2_mine

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
