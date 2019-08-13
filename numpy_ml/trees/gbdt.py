import numpy as np

from .dt import DecisionTree
from .losses import MSELoss, CrossEntropyLoss


def to_one_hot(labels, n_classes=None):
    if labels.ndim > 1:
        raise ValueError("labels must have dimension 1, but got {}".format(labels.ndim))

    N = labels.size
    n_cols = np.max(labels) + 1 if n_classes is None else n_classes
    one_hot = np.zeros((N, n_cols))
    one_hot[np.arange(N), labels] = 1.0
    return one_hot


class GradientBoostedDecisionTree:
    def __init__(
        self,
        n_iter,
        max_depth=None,
        classifier=True,
        learning_rate=1,
        loss="crossentropy",
        step_size="constant",
    ):
        """
        A gradient boosted ensemble of decision trees.

        Notes
        -----
        Gradient boosted machines (GBMs) fit an ensemble of `m` weak learners such that:

        .. math::

            f_m(X) = b(X) + \eta w_1 g_1 + \ldots + \eta w_m g_m

        where `b` is a fixed initial estimate for the targets, :math:`\eta` is
        a learning rate parameter, and :math:`w_{\cdot}` and :math:`g_{\cdot}`
        denote the weights and learner predictions for subsequent fits.

        We fit each `w` and `g` iteratively using a greedy strategy so that at each
        iteration `i`,

        .. math::

            w_i, g_i = \\arg \min_{w_i, g_i} L(Y, f_{i-1}(X) + w_i g_i)

        On each iteration we fit a new weak learner to predict the negative
        gradient of the loss with respect to the previous prediction, :math:`f_{i-1}(X)`.
        We then use the element-wise product of the predictions of this weak
        learner, :math:`g_i`, with a weight, :math:`w_i`, to compute the amount to
        adjust the predictions of our model at the previous iteration, :math:`f_{i-1}(X)`:

        .. math::

            f_i(X) := f_{i-1}(X) + w_i g_i

        Parameters
        ----------
        n_iter : int
            The number of iterations / weak estimators to use when fitting each
            dimension / class of `Y`.
        max_depth : int
            The maximum depth of each decision tree weak estimator. Default is
            None.
        classifier : bool
            Whether `Y` contains class labels or real-valued targets. Default
            is True.
        learning_rate : float
            Value in [0, 1] controlling the amount each weak estimator
            contributes to the overall model prediction. Sometimes known as the
            `shrinkage parameter` in the GBM literature. Default is 1.
        loss : {'crossentropy', 'mse'}
            The loss to optimize for the GBM. Default is 'crossentropy'.
        step_size : {"constant", "adaptive"}
            How to choose the weight for each weak learner. If "constant", use
            a fixed weight of 1 for each learner. If "adaptive", use a step
            size computed via line-search on the current iteration's loss.
            Default is 'constant'.
        """
        self.loss = loss
        self.weights = None
        self.learners = None
        self.out_dims = None
        self.n_iter = n_iter
        self.base_estimator = None
        self.max_depth = max_depth
        self.step_size = step_size
        self.classifier = classifier
        self.learning_rate = learning_rate

    def fit(self, X, Y):
        """
        Fit the gradient boosted decision trees on a dataset.

        Parameters
        ----------
        X : :py:class:`ndarray <numpy.ndarray>` of shape (N, M)
            The training data of `N` examples, each with `M` features
        Y : :py:class:`ndarray <numpy.ndarray>` of shape (N,)
            An array of integer class labels for each example in `X` if
            ``self.classifier = True``, otherwise the set of target values for
            each example in `X`.
        """
        if self.loss == "mse":
            loss = MSELoss()
        elif self.loss == "crossentropy":
            loss = CrossEntropyLoss()

        # convert Y to one_hot if not already
        if self.classifier:
            Y = to_one_hot(Y.flatten())
        else:
            Y = Y.reshape(-1, 1) if len(Y.shape) == 1 else Y

        N, M = X.shape
        self.out_dims = Y.shape[1]
        self.learners = np.empty((self.n_iter, self.out_dims), dtype=object)
        self.weights = np.ones((self.n_iter, self.out_dims))
        self.weights[1:, :] *= self.learning_rate

        # fit the base estimator
        Y_pred = np.zeros((N, self.out_dims))
        for k in range(self.out_dims):
            t = loss.base_estimator()
            t.fit(X, Y[:, k])
            Y_pred[:, k] += t.predict(X)
            self.learners[0, k] = t

        # incrementally fit each learner on the negative gradient of the loss
        # wrt the previous fit (pseudo-residuals)
        for i in range(1, self.n_iter):
            for k in range(self.out_dims):
                y, y_pred = Y[:, k], Y_pred[:, k]
                neg_grad = -1 * loss.grad(y, y_pred)

                # use MSE as the surrogate loss when fitting to negative gradients
                t = DecisionTree(
                    classifier=False, max_depth=self.max_depth, criterion="mse"
                )

                # fit current learner to negative gradients
                t.fit(X, neg_grad)
                self.learners[i, k] = t

                # compute step size and weight for the current learner
                step = 1.0
                h_pred = t.predict(X)
                if self.step_size == "adaptive":
                    step = loss.line_search(y, y_pred, h_pred)

                # update weights and our overall prediction for Y
                self.weights[i, k] *= step
                Y_pred[:, k] += self.weights[i, k] * h_pred

    def predict(self, X):
        """
        Use the trained model to classify or predict the examples in `X`.

        Parameters
        ----------
        X : :py:class:`ndarray <numpy.ndarray>` of shape `(N, M)`
            The training data of `N` examples, each with `M` features

        Returns
        -------
        preds : :py:class:`ndarray <numpy.ndarray>` of shape `(N,)`
            The integer class labels predicted for each example in `X` if
            ``self.classifier = True``, otherwise the predicted target values.
        """
        Y_pred = np.zeros((X.shape[0], self.out_dims))
        for i in range(self.n_iter):
            for k in range(self.out_dims):
                Y_pred[:, k] += self.weights[i, k] * self.learners[i, k].predict(X)

        if self.classifier:
            Y_pred = Y_pred.argmax(axis=1)

        return Y_pred
