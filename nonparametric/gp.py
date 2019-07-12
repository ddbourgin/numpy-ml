import sys
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

sys.path.append("..")
from utils.kernels import KernelInitializer


class GPRegression:
    """
    A Gaussian process defines a prior distribution over functions mapping
    X -> ℝ, where X can be any finite (or infinite!)-dimensional set.

    More concretely, let f(xk) be the random variable corresponding to the
    value of a function f at a point xk ∈ X. We can define a random variable z
    = [f(x1), ..., f(xN)] for any finite set of points {x1,..., xN} ⊂ X. If f
    is distributed according to a Gaussian Process, we have that

        z ~ N(mu, K)

    for

        mu = [mean(x1), ..., mean(xN)]
        K[i, j] = kernel(xi, xj)

    where `mean` is the mean function (it is common to define mean(x) = 0), and
    `kernel` is a kernel / covariance function that determines the general
    shape of the GP prior over functions, p(f).
    """

    def __init__(self, kernel="RBFKernel", alpha=1e-10):
        """
        A Gaussian Process (GP) regression model.

            y | X, f  ~ N( [f(x1), ..., f(xn)], alpha * I )
            f | X     ~ GP(0, K)

        for data D = {(x1, y1), ..., (xn, yn)} and a covariance matrix K[i, j]
        = kernel(xi, xj) for all i, j in {1, ..., n}.

        Parameters
        ----------
        kernel : str (default: 'RBFKernel')
            The kernel to use in fitting the GP prior
        alpha : float (default: 1e-10)
            An isotropic noise term for the diagonal in the GP covariance, K.
            Larger values correspond to the expectation of greater noise in the
            observed data points.
        """
        self.kernel = KernelInitializer(kernel)()
        self.parameters = {"GP_mean": None, "GP_cov": None, "X": None}
        self.hyperparameters = {"kernel": str(self.kernel), "alpha": alpha}

    def fit(self, X, y):
        """
        Fit the GP prior to the training data.

        Parameters
        ----------
        X : numpy array of shape (N, M)
            A training dataset of N examples, each with dimensionality M
        y : numpy array of shape (N, O)
            A collection of real-valued training targets for the
            examples in X, each with dimension O
        """
        mu = np.zeros(X.shape[0])
        K = self.kernel(X, X)

        self.parameters["X"] = X
        self.parameters["y"] = y
        self.parameters["GP_cov"] = K
        self.parameters["GP_mean"] = mu

    def predict(self, X, conf_interval=0.95, return_cov=False):
        """
        Return the MAP estimate for y*, corresponding the mean/mode of the
        posterior predictive distribution, p(y* | x*, X, y). Under the GP
        regression model, the posterior predictive distribution is

            y* | x*, X, y ~ N(mu*, cov*)

        where

            mu*  = K* @ (K + alpha * I)^{-1} @ y
            cov* = K** - K*' @ (K + alpha * I)^{-1} @ K*

            K = kernel(X, X)
            K* = kernel(X, X*)
            K** = kernel(X*, X*)

        NB. This implementation uses the inefficient but general purpose
        `np.linalg.inv` routine to invert (K + alpha * I). A more efficient way
        is to rely on the fact that K (and hence also K + alpha * I) is
        symmetric positive (semi-)definite and take the inner product of the
        inverse of its (lower) Cholesky decompositions:

            Q^{-1} = cholesky(Q)^{-1}.T @ cholesky(Q)^{-1}

        For more details on a production-grade implementation, see Algorithm
        2.1 in Rasmussen & Williams (2006).

        Parameters
        ----------
        X : numpy array of shape (N, M)
            The collection of datapoints to generate predictions on
        conf_interval : float in (0, 1) (default: 0.95)
            The percentage confidence bound to return for each prediction. If
            the scipy package is not available, this value is always set to
            0.95.
        return_cov : bool (default: False)
            If True, also return the covariance (cov*) of the posterior
            predictive distribution for the points in `X`

        Returns
        -------
        y_pred : numpy array of shape (N, O)
            The predicted values for each point in X, each with dimensionality O
        conf : numpy array of shape (N, O)
            The %conf_interval confidence bound for each y_pred. The %conf
            confidence interval for the i'th prediction is [y[i] - conf[i],
            y[i] + conf[i]].
        cov : numpy array of shape (N, N)
            The covariance (cov*) of the posterior predictive distribution for
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
        evidence), p(y | X, kernel_params).

        Under the GP regression model, the marginal likelihood is normally
        distributed:

            y | X, kernel_params ~ N(0, K + alpha * I)

        Hence,

        log p(y | X, kernel_params) =
            -0.5 * log det(K + alpha * I) -
                0.5 * y.T @ (K + alpha * I)^{-1} @ y + n/2 * log 2*pi

        where K = kernel(X, X) and n is the number of dimensions in K.

        Parameters
        ----------
        kernel_params : dict (default: None)
            Parameters for the kernel function. If `None`, calculate the
            marginal likelihood under the kernel parameters defined at model
            initialization.

        Returns
        -------
        marginal_log_likelihood : float
            The log likelihood of the training targets given the kernel
            parameterized by `kernel_params` and the training inputs,
            marginalized over all functions f
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
        X : numpy array of shape (N, M)
            The collection of datapoints to generate predictions on. Only used if
            `dist` = 'posterior_predictive'.
        n_samples: int (default: 1)
            The number of samples to generate
        dist : str in {"posterior_predictive", "prior"} (default: "posterior_predictive")
            The distribution to draw samples from

        Returns
        -------
        samples : numpy array of shape (n_samples, O, N)
            The generated samples for the points in X
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
