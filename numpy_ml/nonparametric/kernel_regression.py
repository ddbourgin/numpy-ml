import sys

from ..utils.kernels import KernelInitializer


class KernelRegression:
    def __init__(self, kernel=None):
        """
        A Nadaraya-Watson kernel regression model.

            f(x) = sum_i w_i(x) * y_i

        where the sample weighting functions, w_i, are simply

            w_i(x) = k(x, x_i) / sum_j k(x, x_j)

        with k being the kernel function.

        Observe that k-nearest neighbors (KNN) regression is a special case of
        kernel regression where the k closest observations have a weight 1/k,
        and all others have weight 0.

        Parameters
        ----------
        kernel : str, `KernelBase` instance, or dict (default: None)
            The kernel to use. If `None`, default to `LinearKernel`
        """
        self.parameters = {"X": None, "y": None}
        self.hyperparameters = {"kernel": str(kernel)}
        self.kernel = KernelInitializer(kernel)()

    def fit(self, X, y):
        """
        Fit the regression model to the data and targets in `X` and `y`

        Parameters
        ----------
        X : numpy array of shape (N, M)
            An array of N examples to generate predictions on
        y : numpy array of shape (N, ...)
            Predicted targets for the N' rows in `X`
        """
        self.parameters = {"X": X, "y": y}

    def predict(self, X):
        """
        Generate predictions for the targets associated with the rows in `X`.

        Parameters
        ----------
        X : numpy array of shape (N', M')
            An array of N' examples to generate predictions on

        Returns
        -------
        y : numpy array of shape (N', ...)
            Predicted targets for the N' rows in `X`
        """
        K = self.kernel
        P = self.parameters
        sim = K(P["X"], X)
        return (sim * P["y"][:, None]).sum(axis=0) / sim.sum(axis=0)
