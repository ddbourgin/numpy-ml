"""A module of Bayesian linear regression models."""
import numpy as np
import scipy.stats as stats

from numpy_ml.utils.testing import is_number, is_symmetric_positive_definite


class BayesianLinearRegressionUnknownVariance:
    def __init__(self, alpha=1, beta=2, mu=0, V=None, fit_intercept=True):
        r"""
        Bayesian linear regression model with unknown variance. Assumes a
        conjugate normal-inverse-gamma joint prior on the model parameters and
        error variance.

        Notes
        -----
        The current model uses a conjugate normal-inverse-gamma joint prior on
        model parameters **b** and error variance :math:`\sigma^2`. The joint
        and marginal posteriors over each are:

        .. math::

            \mathbf{b}, \sigma^2 &\sim
                \text{N-\Gamma^{-1}}(\mu, \mathbf{V}^{-1}, \alpha, \beta) \\
            \sigma^2 &\sim \text{InverseGamma}(\alpha, \beta) \\
            \mathbf{b} \mid \sigma^2 &\sim \mathcal{N}(\mu, \sigma^2 \mathbf{V})

        Parameters
        ----------
        alpha : float
            The shape parameter for the Inverse-Gamma prior on
            :math:`\sigma^2`. Must be strictly greater than 0. Default is 1.
        beta : float
            The scale parameter for the Inverse-Gamma prior on
            :math:`\sigma^2`. Must be strictly greater than 0. Default is 1.
        mu : :py:class:`ndarray <numpy.ndarray>` of shape `(M,)` or float
            The mean of the Gaussian prior on `b`. If a float, assume `mu`
            is ``np.ones(M) * mu``. Default is 0.
        V : :py:class:`ndarray <numpy.ndarray>` of shape `(N, N)` or `(N,)` or None
            A symmetric positive definite matrix that when multiplied
            element-wise by :math:`\sigma^2` gives the covariance matrix for
            the Gaussian prior on `b`. If a list, assume ``V = diag(V)``. If
            None, assume `V` is the identity matrix.  Default is None.
        fit_intercept : bool
            Whether to fit an intercept term in addition to the coefficients in
            b. If True, the estimates for b will have `M + 1` dimensions, where
            the first dimension corresponds to the intercept. Default is True.

        Attributes
        ----------
        posterior : dict or None
            Frozen random variables for the posterior distributions
            :math:`P(\sigma^2 \mid X)` and :math:`P(b \mid X, \sigma^2)`.
        posterior_predictive : dict or None
            Frozen random variable for the posterior predictive distribution,
            :math:`P(y \mid X)`. This value is only set following a call to
            :meth:`predict <numpy_ml.linear_models.BayesianLinearRegressionUnknownVariance.predict>`.
        """  # noqa: E501
        # this is a placeholder until we know the dimensions of X
        V = 1.0 if V is None else V

        if isinstance(V, list):
            V = np.array(V)

        if isinstance(V, np.ndarray):
            if V.ndim == 1:
                V = np.diag(V)
            elif V.ndim == 2:
                fstr = "V must be symmetric positive definite"
                assert is_symmetric_positive_definite(V), fstr

        self.V = V
        self.mu = mu
        self.beta = beta
        self.alpha = alpha
        self.fit_intercept = fit_intercept

        self.posterior = None
        self.posterior_predictive = None

    def fit(self, X, y):
        """
        Compute the posterior over model parameters using the data in `X` and
        `y`.

        Parameters
        ----------
        X : :py:class:`ndarray <numpy.ndarray>` of shape `(N, M)`
            A dataset consisting of `N` examples, each of dimension `M`.
        y : :py:class:`ndarray <numpy.ndarray>` of shape `(N, K)`
            The targets for each of the `N` examples in `X`, where each target
            has dimension `K`.

        Returns
        -------
        self : :class:`BayesianLinearRegressionUnknownVariance<numpy_ml.linear_models.BayesianLinearRegressionUnknownVariance>` instance
        """  # noqa: E501
        # convert X to a design matrix if we're fitting an intercept
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]

        N, M = X.shape
        alpha, beta, V, mu = self.alpha, self.beta, self.V, self.mu

        if is_number(V):
            V *= np.eye(M)

        if is_number(mu):
            mu *= np.ones(M)

        # sigma
        I = np.eye(N)  # noqa: E741
        a = y - (X @ mu)
        b = np.linalg.inv(X @ V @ X.T + I)
        c = y - (X @ mu)

        shape = N + alpha
        sigma = (1 / shape) * (alpha * beta ** 2 + a @ b @ c)
        scale = sigma ** 2

        # sigma is the mode of the inverse gamma prior on sigma^2
        sigma = scale / (shape - 1)

        # mean
        V_inv = np.linalg.inv(V)
        L = np.linalg.inv(V_inv + X.T @ X)
        R = V_inv @ mu + X.T @ y

        mu = L @ R
        cov = L * sigma

        # posterior distribution for sigma^2 and b
        self.posterior = {
            "sigma**2": stats.distributions.invgamma(a=shape, scale=scale),
            "b | sigma**2": stats.multivariate_normal(mean=mu, cov=cov),
        }
        return self

    def predict(self, X):
        """
        Return the MAP prediction for the targets associated with `X`.

        Parameters
        ----------
        X : :py:class:`ndarray <numpy.ndarray>` of shape `(Z, M)`
            A dataset consisting of `Z` new examples, each of dimension `M`.

        Returns
        -------
        y_pred : :py:class:`ndarray <numpy.ndarray>` of shape `(Z, K)`
            The model predictions for the items in `X`.
        """
        # convert X to a design matrix if we're fitting an intercept
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]

        I = np.eye(X.shape[0])  # noqa: E741
        mu = X @ self.posterior["b | sigma**2"].mean
        cov = X @ self.posterior["b | sigma**2"].cov @ X.T + I

        # MAP estimate for y corresponds to the mean of the posterior
        # predictive
        self.posterior_predictive = stats.multivariate_normal(mu, cov)
        return mu


class BayesianLinearRegressionKnownVariance:
    def __init__(self, mu=0, sigma=1, V=None, fit_intercept=True):
        r"""
        Bayesian linear regression model with known error variance and
        conjugate Gaussian prior on model parameters.

        Notes
        -----
        Uses a conjugate Gaussian prior on the model coefficients **b**. The
        posterior over model coefficients is then

        .. math::

            \mathbf{b} \mid \mu, \sigma^2, \mathbf{V}
                \sim \mathcal{N}(\mu, \sigma^2 \mathbf{V})

        Ridge regression is a special case of this model where :math:`\mu =
        \mathbf{0}`, :math:`\sigma = 1` and :math:`\mathbf{V} = \mathbf{I}`
        (ie., the prior on the model coefficients **b** is a zero-mean, unit
        covariance Gaussian).

        Parameters
        ----------
        mu : :py:class:`ndarray <numpy.ndarray>` of shape `(M,)` or float
            The mean of the Gaussian prior on `b`. If a float, assume `mu` is
            ``np.ones(M) * mu``. Default is 0.
        sigma : float
            The square root of the scaling term for covariance of the Gaussian
            prior on `b`. Default is 1.
        V : :py:class:`ndarray <numpy.ndarray>` of shape `(N,N)` or `(N,)` or None
            A symmetric positive definite matrix that when multiplied
            element-wise by ``sigma ** 2`` gives the covariance matrix for the
            Gaussian prior on `b`. If a list, assume ``V = diag(V)``. If None,
            assume `V` is the identity matrix. Default is None.
        fit_intercept : bool
            Whether to fit an intercept term in addition to the coefficients in
            `b`. If True, the estimates for `b` will have `M + 1` dimensions, where
            the first dimension corresponds to the intercept. Default is True.

        Attributes
        ----------
        posterior : dict or None
            Frozen random variable for the posterior distribution :math:`P(b
            \mid X, \sigma^2)`.
        posterior_predictive : dict or None
            Frozen random variable for the posterior predictive distribution,
            :math:`P(y \mid X)`. This value is only set following a call to
            :meth:`predict <numpy_ml.linear_models.BayesianLinearRegressionKnownVariance.predict>`.
        """  # noqa: E501
        # this is a placeholder until we know the dimensions of X
        V = 1.0 if V is None else V

        if isinstance(V, list):
            V = np.array(V)

        if isinstance(V, np.ndarray):
            if V.ndim == 1:
                V = np.diag(V)
            elif V.ndim == 2:
                fstr = "V must be symmetric positive definite"
                assert is_symmetric_positive_definite(V), fstr

        self.posterior = {}
        self.posterior_predictive = {}

        self.V = V
        self.mu = mu
        self.sigma = sigma
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        """
        Compute the posterior over model parameters using the data in `X` and
        `y`.

        Parameters
        ----------
        X : :py:class:`ndarray <numpy.ndarray>` of shape `(N, M)`
            A dataset consisting of `N` examples, each of dimension `M`.
        y : :py:class:`ndarray <numpy.ndarray>` of shape `(N, K)`
            The targets for each of the `N` examples in `X`, where each target
            has dimension `K`.
        """
        # convert X to a design matrix if we're fitting an intercept
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]

        N, M = X.shape

        if is_number(self.V):
            self.V *= np.eye(M)

        if is_number(self.mu):
            self.mu *= np.ones(M)

        V = self.V
        mu = self.mu
        sigma = self.sigma

        V_inv = np.linalg.inv(V)
        L = np.linalg.inv(V_inv + X.T @ X)
        R = V_inv @ mu + X.T @ y

        mu = L @ R
        cov = L * sigma ** 2

        # posterior distribution over b conditioned on sigma
        self.posterior["b"] = stats.multivariate_normal(mu, cov)

    def predict(self, X):
        """
        Return the MAP prediction for the targets associated with `X`.

        Parameters
        ----------
        X : :py:class:`ndarray <numpy.ndarray>` of shape `(Z, M)`
            A dataset consisting of `Z` new examples, each of dimension `M`.

        Returns
        -------
        y_pred : :py:class:`ndarray <numpy.ndarray>` of shape `(Z, K)`
            The MAP predictions for the targets associated with the items in
            `X`.
        """
        # convert X to a design matrix if we're fitting an intercept
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]

        I = np.eye(X.shape[0])  # noqa: E741
        mu = X @ self.posterior["b"].mean
        cov = X @ self.posterior["b"].cov @ X.T + I

        # MAP estimate for y corresponds to the mean/mode of the gaussian
        # posterior predictive distribution
        self.posterior_predictive = stats.multivariate_normal(mu, cov)
        return mu
