import numpy as np

from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessRegressor

from .knn import KNN
from .gp import GPRegression
from ..utils.distance_metrics import euclidean


def test_knn_regression():
    while True:
        N = np.random.randint(2, 100)
        M = np.random.randint(2, 100)
        k = np.random.randint(1, N)
        ls = np.min([np.random.randint(1, 10), N - 1])
        weights = np.random.choice(["uniform", "distance"])

        X = np.random.rand(N, M)
        X_test = np.random.rand(N, M)
        y = np.random.rand(N)

        knn = KNN(
            k=k, leaf_size=ls, metric=euclidean, classifier=False, weights=weights
        )
        knn.fit(X, y)
        preds = knn.predict(X_test)

        gold = KNeighborsRegressor(
            p=2,
            leaf_size=ls,
            n_neighbors=k,
            weights=weights,
            metric="minkowski",
            algorithm="ball_tree",
        )
        gold.fit(X, y)
        gold_preds = gold.predict(X_test)

        for mine, theirs in zip(preds, gold_preds):
            np.testing.assert_almost_equal(mine, theirs)
        print("PASSED")


def test_knn_clf():
    while True:
        N = np.random.randint(2, 100)
        M = np.random.randint(2, 100)
        k = np.random.randint(1, N)
        n_classes = np.random.randint(10)
        ls = np.min([np.random.randint(1, 10), N - 1])
        weights = "uniform"

        X = np.random.rand(N, M)
        X_test = np.random.rand(N, M)
        y = np.random.randint(0, n_classes, size=N)

        knn = KNN(k=k, leaf_size=ls, metric=euclidean, classifier=True, weights=weights)
        knn.fit(X, y)
        preds = knn.predict(X_test)

        gold = KNeighborsClassifier(
            p=2,
            leaf_size=ls,
            n_neighbors=k,
            weights=weights,
            metric="minkowski",
            algorithm="ball_tree",
        )
        gold.fit(X, y)
        gold_preds = gold.predict(X_test)

        for mine, theirs in zip(preds, gold_preds):
            np.testing.assert_almost_equal(mine, theirs)
        print("PASSED")


def test_gp_regression():
    while True:
        alpha = np.random.rand()
        N = np.random.randint(2, 100)
        M = np.random.randint(2, 100)
        K = np.random.randint(1, N)
        J = np.random.randint(1, 3)

        X = np.random.rand(N, M)
        y = np.random.rand(N, J)
        X_test = np.random.rand(K, M)

        gp = GPRegression(kernel="RBFKernel(sigma=1)", alpha=alpha)
        gold = GaussianProcessRegressor(
            kernel=None, alpha=alpha, optimizer=None, normalize_y=False
        )

        gp.fit(X, y)
        gold.fit(X, y)

        preds, _ = gp.predict(X_test)
        gold_preds = gold.predict(X_test)
        np.testing.assert_almost_equal(preds, gold_preds)

        mll = gp.marginal_log_likelihood()
        gold_mll = gold.log_marginal_likelihood()
        np.testing.assert_almost_equal(mll, gold_mll)

        print("PASSED")
