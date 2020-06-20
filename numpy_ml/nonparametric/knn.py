"""A k-Nearest Neighbors (KNN) model for both classiciation and regression."""
from collections import Counter

import numpy as np

from ..utils.data_structures import BallTree


class KNN:
    def __init__(
        self, k=5, leaf_size=40, classifier=True, metric=None, weights="uniform",
    ):
        """
        A `k`-nearest neighbors (kNN) model relying on a ball tree for efficient
        computation.

        Parameters
        ----------
        k : int
            The number of neighbors to use during prediction. Default is 5.
        leaf_size : int
            The maximum number of datapoints at each leaf in the ball tree.
            Default is 40.
        classifier : bool
            Whether to treat the values in Y as class labels (classifier =
            True) or real-valued targets (classifier = False). Default is True.
        metric : :doc:`Distance metric <numpy_ml.utils.distance_metrics>` or None
            The distance metric to use for computing nearest neighbors. If
            None, use the :func:`~numpy_ml.utils.distance_metrics.euclidean`
            metric by default. Default is None.
        weights : {'uniform', 'distance'}
            How to weight the predictions from each neighbors. 'uniform'
            assigns uniform weights to each neighbor, while 'distance' assigns
            weights proportional to the inverse of the distance from the query
            point. Default is 'uniform'.
        """
        self._ball_tree = BallTree(leaf_size=leaf_size, metric=metric)
        self.hyperparameters = {
            "id": "KNN",
            "k": k,
            "leaf_size": leaf_size,
            "classifier": classifier,
            "metric": str(metric),
            "weights": weights,
        }

    def fit(self, X, y):
        r"""
        Fit the model to the data and targets in `X` and `y`

        Parameters
        ----------
        X : numpy array of shape `(N, M)`
            An array of `N` examples to generate predictions on.
        y : numpy array of shape `(N, *)`
            Targets for the `N` rows in `X`.
        """
        if X.ndim != 2:
            raise Exception("X must be two-dimensional")
        self._ball_tree.fit(X, y)

    def predict(self, X):
        r"""
        Generate predictions for the targets associated with the rows in `X`.

        Parameters
        ----------
        X : numpy array of shape `(N', M')`
            An array of `N'` examples to generate predictions on.

        Returns
        -------
        y : numpy array of shape `(N', *)`
            Predicted targets for the `N'` rows in `X`.
        """
        predictions = []
        H = self.hyperparameters
        for x in X:
            pred = None
            nearest = self._ball_tree.nearest_neighbors(H["k"], x)
            targets = [n.val for n in nearest]

            if H["classifier"]:
                if H["weights"] == "uniform":
                    # for consistency with sklearn / scipy.stats.mode, return
                    # the smallest class ID in the event of a tie
                    counts = Counter(targets).most_common()
                    pred, _ = sorted(counts, key=lambda x: (-x[1], x[0]))[0]
                elif H["weights"] == "distance":
                    best_score = -np.inf
                    for label in set(targets):
                        scores = [1 / n.distance for n in nearest if n.val == label]
                        pred = label if np.sum(scores) > best_score else pred
            else:
                if H["weights"] == "uniform":
                    pred = np.mean(targets)
                elif H["weights"] == "distance":
                    weights = [1 / n.distance for n in nearest]
                    pred = np.average(targets, weights=weights)
            predictions.append(pred)
        return np.array(predictions)
