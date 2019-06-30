import sys

sys.path.append("..")

import numpy as np

from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier

from knn import KNN
from utils.distance_metrics import euclidean


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
        #  weights = np.random.choice(["uniform", "distance"])
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
