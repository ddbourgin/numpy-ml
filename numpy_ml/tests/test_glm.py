# flake8: noqa
import numpy as np

import statsmodels.api as sm
from numpy_ml.linear_models import GeneralizedLinearModel
from numpy_ml.linear_models.glm import _GLM_LINKS
from numpy_ml.utils.testing import random_tensor


def test_glm(N=20):
    np.random.seed(12345)
    N = np.inf if N is None else N

    i = 1
    while i < N + 1:
        n_samples = np.random.randint(10, 100)

        # n_feats << n_samples to avoid perfect separation / multiple solutions
        n_feats = np.random.randint(1, 1 + n_samples // 2)
        target_dim = 1

        fit_intercept = np.random.choice([True, False])
        _link = np.random.choice(list(_GLM_LINKS.keys()))

        families = {
            "identity": sm.families.Gaussian(),
            "logit": sm.families.Binomial(),
            "log": sm.families.Poisson(),
        }

        print(f"Link: {_link}")
        print(f"Fit intercept: {fit_intercept}")

        X = random_tensor((n_samples, n_feats), standardize=True)
        if _link == "logit":
            y = np.random.choice([0.0, 1.0], size=(n_samples, target_dim))
        elif _link == "log":
            y = np.random.choice(np.arange(0, 100), size=(n_samples, target_dim))
        elif _link == "identity":
            y = random_tensor((n_samples, target_dim), standardize=True)
        else:
            raise ValueError(f"Unknown link function {_link}")

        # Fit gold standard model on the entire dataset
        fam = families[_link]
        Xdesign = np.c_[np.ones(X.shape[0]), X] if fit_intercept else X

        glm_gold = sm.GLM(y, Xdesign, family=fam)
        glm_gold = glm_gold.fit()

        glm_mine = GeneralizedLinearModel(link=_link, fit_intercept=fit_intercept)
        glm_mine.fit(X, y)

        # check that model coefficients match
        beta = glm_mine.beta.T.ravel()
        np.testing.assert_almost_equal(beta, glm_gold.params, decimal=6)
        print("\t1. Overall model coefficients match")

        # check that model predictions match
        np.testing.assert_almost_equal(
            glm_mine.predict(X), glm_gold.predict(Xdesign), decimal=5
        )
        print("\t2. Overall model predictions match")

        print("\tPASSED\n")
        i += 1
