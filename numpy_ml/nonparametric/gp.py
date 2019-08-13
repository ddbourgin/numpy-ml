import warnings
import numpy as np
from numpy.linalg import slogdet, inv

try:
    _SCIPY = True
    from scipy.stats import norm
except:
    _SCIPY = False
    warnings.warn(
        "Could not import scipy.stats. Confidence scores "
        "for GPRegression are restricted to 95% bounds"
    )

from ..utils.kernels import KernelInitializer


class GPRegression:
    def __init__(self, kernel="RBFKernel", alpha=1e-10):
        """
        A Gaussian Process (GP) regression model.

        .. math::

            y \mid X, f  &\sim  \mathcal{N}( [f(x_1), \ldots, f(x_n)], \\alpha I ) \\\\
            f \mid X     &\sim  \\text{GP}(0, K)

        for data :math:`D = \{(x_1, y_1), \ldots, (x_n, y_n) \}` and a covariance matrix :math:`K_{ij}
        = \\text{kernel}(x_i, x_j)` for all :math:`i, j \in \{1, \ldots, n \}`.

        Parameters
        ----------
        kernel : str
            The kernel to use in fitting the GP prior. Default is 'RBFKernel'.
        alpha : float
            An isotropic noise term for the diagonal in the GP covariance, `K`.
            Larger values correspond to the expectation of greater noise in the
            observed data points. Default is 1e-10.
        """
        self.kernel = KernelInitializer(kernel)()
        self.parameters = {"GP_mean": None, "GP_cov": None, "X": None}
        self.hyperparameters = {"kernel": str(self.kernel), "alpha": alpha}

    def fit(self, X, y):
        """
        Fit the GP prior to the training data.

        Parameters
        ----------
        X : :py:class:`ndarray <numpy.ndarray>` of shape `(N, M)`
            A training dataset of `N` examples, each with dimensionality `M`.
        y : :py:class:`ndarray <numpy.ndarray>` of shape `(N, O)`
            A collection of real-valued training targets for the
            examples in `X`, each with dimension `O`.
        """
        mu = np.zeros(X.shape[0])
        K = self.kernel(X, X)

        self.parameters["X"] = X
        self.parameters["y"] = y
        self.parameters["GP_cov"] = K
        self.parameters["GP_mean"] = mu

    def predict(self, X, conf_interval=0.95, return_cov=False):
        """
        Return the MAP estimate for :math:`y^*`, corresponding the mean/mode of
        the posterior predictive distribution, :math:`p(y^* \mid x^*, X, y)`.

        Notes
        -----
        Under the GP regression model, the posterior predictive distribution is

        .. math::

            y^* \mid x^*, X, y \sim \mathcal{N}(\mu^*, \\text{cov}^*)

        where

        .. math::

            \mu^*  &=  K^* (K + \\alpha I)^{-1} y \\\\
            \\text{cov}^*  &=  K^{**} - K^{*'} (K + \\alpha I)^{-1} K^*

        and

        .. math::

            K  &=  \\text{kernel}(X, X) \\\\
            K^*  &=  \\text{kernel}(X, X^*) \\\\
            K^{**}  &=  \\text{kernel}(X^*, X^*)

        NB. This implementation uses the inefficient but general purpose
        `np.linalg.inv` routine to invert :math:`(K + \\alpha I)`. A more
        efficient way is to rely on the fact that `K` (and hence also :math:`K
        + \\alpha I`) is symmetric positive (semi-)definite and take the inner
        product of the inverse of its (lower) Cholesky decompositions:

        .. math::

            Q^{-1} = \\text{cholesky}(Q)^{-1 \\top} \\text{cholesky}(Q)^{-1}

        For more details on a production-grade implementation, see Algorithm
        2.1 in Rasmussen & Williams (2006).

        Parameters
        ----------
        X : :py:class:`ndarray <numpy.ndarray>` of shape (N, M)
            The collection of datapoints to generate predictions on
        conf_interval : float in (0, 1)
            The percentage confidence bound to return for each prediction. If
            the scipy package is not available, this value is always set to
            0.95. Default is 0.95.
        return_cov : bool
            If True, also return the covariance (`cov*`) of the posterior
            predictive distribution for the points in `X`. Default is False.

        Returns
        -------
        y_pred : :py:class:`ndarray <numpy.ndarray>` of shape `(N, O)`
            The predicted values for each point in `X`, each with
            dimensionality `O`.
        conf : :py:class:`ndarray <numpy.ndarray>` of shape `(N, O)`
            The % conf_interval confidence bound for each `y_pred`. The conf %
            confidence interval for the `i`'th prediction is ``[y[i] - conf[i],
            y[i] + conf[i]]``.
        cov : :py:class:`ndarray <numpy.ndarray>` of shape `(N, N)`
            The covariance (`cov*`) of the posterior predictive distribution for
            `X`. Only returned if `return_cov` is True.
        """
        if conf_interval != 0.95 and not _SCIPY:
            fstr = "Cannot compute {}% confidence score without scipy.stats"
            warnings.warn(fstr.format(conf_interval))

        X_star = X
        X = self.parameters["X"]
        y = self.parameters["y"]
        K = self.parameters["GP_cov"]
        alpha = self.hyperparameters["alpha"]

        K_star = self.kernel(X_star, X)
        K_star_star = self.kernel(X_star, X_star)

        sig = np.eye(K.shape[0]) * alpha
        K_y_inv = inv(K + sig)

        pp_mean = K_star @ K_y_inv @ y
        pp_cov = K_star_star - K_star @ K_y_inv @ K_star.T

        # if we can't use scipy, ignore the passed value for `conf_interval`
        # and return the 95% confidence bound.
        # (norm.ppf == inverse CDF for standard normal)
        percentile = 1.96 if not _SCIPY else norm.ppf(conf_interval)
        conf = percentile * np.sqrt(np.diag(pp_cov))
        return (pp_mean, conf) if not return_cov else (pp_mean, conf, pp_cov)

    def marginal_log_likelihood(self, kernel_params=None):
        """
        Compute the log of the marginal likelihood (i.e., the log model
        evidence), :math:`p(y \mid X, \\text{kernel_params})`.

        Notes
        -----
        Under the GP regression model, the marginal likelihood is normally
        distributed:

        .. math::

            y | X, \\theta  \sim  \mathcal{N}(0, K + \\alpha I)

        Hence,

        .. math::

            \log p(y \mid X, \\theta) =
                -0.5 \log \det(K + \\alpha I) -
                    0.5 y^\\top (K + \\alpha I)^{-1} y + \\frac{n}{2} \log 2 \pi

        where :math:`K = \\text{kernel}(X, X)`, :math:`\\theta` is the set of
        kernel parameters, and `n` is the number of dimensions in `K`.

        Parameters
        ----------
        kernel_params : dict
            Parameters for the kernel function. If None, calculate the
            marginal likelihood under the kernel parameters defined at model
            initialization. Default is None.

        Returns
        -------
        marginal_log_likelihood : float
            The log likelihood of the training targets given the kernel
            parameterized by `kernel_params` and the training inputs,
            marginalized over all functions `f`.
        """
        X = self.parameters["X"]
        y = self.parameters["y"]
        alpha = self.hyperparameters["alpha"]

        K = self.parameters["GP_cov"]
        if kernel_params is not None:
            # create a new kernel with parameters `kernel_params` and recalc
            # the GP covariance matrix
            summary_dict = self.kernel.summary_dict()
            summary_dict["parameters"].update(kernel_params)
            kernel = KernelInitializer(summary_dict)()
            K = kernel(X, X)

        # add isotropic noise to kernel diagonal
        K += np.eye(K.shape[0]) * alpha

        Kinv = inv(K)
        Klogdet = -0.5 * slogdet(K)[1]
        const = K.shape[0] / 2 * np.log(2 * np.pi)

        # handle both uni- and multidimensional target values
        if y.ndim == 1:
            y = y[:, np.newaxis]

        # sum over each dimension of y
        marginal_ll = np.sum([Klogdet - 0.5 * _y.T @ Kinv @ _y - const for _y in y.T])
        return marginal_ll

    def sample(self, X, n_samples=1, dist="posterior_predictive"):
        """
        Sample functions from the GP prior or posterior predictive
        distribution.

        Parameters
        ----------
        X : :py:class:`ndarray <numpy.ndarray>` of shape `(N, M)`
            The collection of datapoints to generate predictions on. Only used if
            `dist` = 'posterior_predictive'.
        n_samples: int
            The number of samples to generate. Default is 1.
        dist : {"posterior_predictive", "prior"}
            The distribution to draw samples from. Default is
            "posterior_predictive".

        Returns
        -------
        samples : :py:class:`ndarray <numpy.ndarray>` of shape `(n_samples, O, N)`
            The generated samples for the points in `X`.
        """
        mvnorm = np.random.multivariate_normal

        if dist == "prior":
            mu = np.zeros((X.shape[0], 1))
            cov = self.kernel(X, X)
        elif dist == "posterior_predictive":
            mu, _, cov = self.predict(X, return_cov=True)
        else:
            raise ValueError("Unrecognized dist: '{}'".format(dist))

        if mu.ndim == 1:
            mu = mu[:, np.newaxis]

        samples = np.array([mvnorm(_mu, cov, size=n_samples) for _mu in mu.T])
        return samples.swapaxes(0, 1)
