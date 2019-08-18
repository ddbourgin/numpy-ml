from ..utils.kernels import KernelInitializer


class KernelRegression:
    def __init__(self, kernel=None):
        """
        A Nadaraya-Watson kernel regression model.

        Notes
        -----
        The Nadaraya-Watson regression model is

        .. math::

            f(x) = \sum_i w_i(x) y_i

        where the sample weighting functions, :math:`w_i`, are simply

        .. math::

            w_i(x) = \\frac{k(x, x_i)}{\sum_j k(x, x_j)}

        with `k` being the kernel function.

        Observe that `k`-nearest neighbors
        (:class:`~numpy_ml.nonparametric.KNN`) regression is a special case of
        kernel regression where the `k` closest observations have a weight
        `1/k`, and all others have weight 0.

        Parameters
        ----------
        kernel : str, :doc:`Kernel <numpy_ml.utils.kernels>` object, or dict
            The kernel to use. If None, default to
            :class:`~numpy_ml.utils.kernels.LinearKernel`. Default is None.
        """
        self.parameters = {"X": None, "y": None}
        self.hyperparameters = {"kernel": str(kernel)}
        self.kernel = KernelInitializer(kernel)()

    def fit(self, X, y):
        """
        Fit the regression model to the data and targets in `X` and `y`.

        Parameters
        ----------
        X : :py:class:`ndarray <numpy.ndarray>` of shape `(N, M)`
            An array of N examples to generate predictions on
        y : :py:class:`ndarray <numpy.ndarray>` of shape `(N, ...)`
            Predicted targets for the `N` rows in `X`
        """
        self.parameters = {"X": X, "y": y}

    def predict(self, X):
        """
        Generate predictions for the targets associated with the rows in `X`.

        Parameters
        ----------
        X : :py:class:`ndarray <numpy.ndarray>` of shape `(N', M')`
            An array of `N'` examples to generate predictions on

        Returns
        -------
        y : :py:class:`ndarray <numpy.ndarray>` of shape `(N', ...)`
            Predicted targets for the `N'` rows in `X`
        """
        K = self.kernel
        P = self.parameters
        sim = K(P["X"], X)
        return (sim * P["y"][:, None]).sum(axis=0) / sim.sum(axis=0)
