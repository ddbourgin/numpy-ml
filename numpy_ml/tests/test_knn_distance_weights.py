"""Regression coverage for inverse-distance classification votes."""

import unittest

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from numpy_ml.nonparametric.knn import KNN


class TestDistanceWeightedKNN(unittest.TestCase):
    def test_strongest_vote_survives_later_classes(self):
        X = np.array([[0.1], [1.0], [1.1]])
        for labels in ([0, 1, 1], [1, 0, 0], [2, 1, 0], [1, 2, 0]):
            with self.subTest(labels=labels):
                model = KNN(k=3, leaf_size=2, weights="distance")
                model.fit(X, np.array(labels))
                self.assertEqual(model.predict(np.array([[0.0]])).item(), labels[0])

    def test_predictions_match_independent_classifier(self):
        for seed in range(8):
            with self.subTest(seed=seed):
                rng = np.random.default_rng(seed)
                X = rng.normal(size=(31, 3))
                y = rng.integers(0, 4, size=31)
                queries = rng.normal(size=(9, 3))
                model = KNN(k=7, leaf_size=3, weights="distance")
                model.fit(X, y)
                reference = KNeighborsClassifier(n_neighbors=7, weights="distance")
                reference.fit(X, y)
                np.testing.assert_array_equal(
                    model.predict(queries), reference.predict(queries)
                )

    def test_uniform_classification_is_unchanged(self):
        model = KNN(k=3, weights="uniform")
        model.fit(np.array([[0.1], [1.0], [1.1]]), np.array([0, 1, 1]))
        self.assertEqual(model.predict(np.array([[0.0]])).item(), 1)

    def test_distance_weighted_regression_is_unchanged(self):
        X = np.array([[0.1], [1.0], [1.1]])
        y = np.array([3.0, 8.0, 12.0])
        model = KNN(k=3, classifier=False, weights="distance")
        model.fit(X, y)
        np.testing.assert_allclose(
            model.predict(np.array([[0.0]])), [np.average(y, weights=1 / X[:, 0])]
        )


if __name__ == "__main__":
    unittest.main()
