"""A collection of common linear models."""

import numpy as np
from ..utils.testing import is_symmetric_positive_definite, is_number


class LinearRegression:
    def __init__(self, fit_intercept=True):
        r"""
        An ordinary least squares regression model fit via the normal equation.

        Notes
        -----
        Given data matrix *X* and target vector *y*, the maximum-likelihood estimate
        for the regression coefficients, :math:`\\beta`, is:

        .. math::

            \hat{\beta} =
                \left(\mathbf{X}^\top \mathbf{X}\right)^{-1} \mathbf{X}^\top \mathbf{y}

        Parameters
        ----------
        fit_intercept : bool
            Whether to fit an additional intercept term in addition to the
            model coefficients. Default is True.
        """
        self.beta = None
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        """
        Fit the regression coefficients via maximum likelihood.

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

        pseudo_inverse = np.linalg.inv(X.T @ X) @ X.T
        self.beta = np.dot(pseudo_inverse, y)

    def predict(self, X):
        """
        Use the trained model to generate predictions on a new collection of
        data points.

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
        return np.dot(X, self.beta)


class RidgeRegression:
    def __init__(self, alpha=1, fit_intercept=True):
        r"""
        A ridge regression model fit via the normal equation.

        Notes
        -----
        Given data matrix **X** and target vector **y**, the maximum-likelihood
        estimate for the ridge coefficients, :math:`\\beta`, is:

        .. math::

            \hat{\beta} =
                \left(\mathbf{X}^\top \mathbf{X} + \alpha \mathbf{I} \right)^{-1}
                    \mathbf{X}^\top \mathbf{y}

        It turns out that this estimate for :math:`\beta` also corresponds to
        the MAP estimate if we assume a multivariate Gaussian prior on the model
        coefficients:

        .. math::

            \beta \sim \mathcal{N}(\mathbf{0}, \frac{1}{2M} \mathbf{I})

        Note that this assumes that the data matrix **X** has been standardized
        and the target values **y** centered at 0.

        Parameters
        ----------
        alpha : float
            L2 regularization coefficient. Higher values correspond to larger
            penalty on the L2 norm of the model coefficients. Default is 1.
        fit_intercept : bool
            Whether to fit an additional intercept term in addition to the
            model coefficients. Default is True.
        """
        self.beta = None
        self.alpha = alpha
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        """
        Fit the regression coefficients via maximum likelihood.

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

        A = self.alpha * np.eye(X.shape[1])
        pseudo_inverse = np.linalg.inv(X.T @ X + A) @ X.T
        self.beta = pseudo_inverse @ y

    def predict(self, X):
        """
        Use the trained model to generate predictions on a new collection of
        data points.

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
        return np.dot(X, self.beta)


class LogisticRegression:
    def __init__(self, penalty="l2", gamma=0, fit_intercept=True):
        r"""
        A simple logistic regression model fit via gradient descent on the
        penalized negative log likelihood.

        Notes
        -----
        For logistic regression, the penalized negative log likelihood of the
        targets **y** under the current model is

        .. math::

            - \log \mathcal{L}(\mathbf{b}, \mathbf{y}) = -\frac{1}{N} \left[
                \left(
                    \sum_{i=0}^N y_i \log(\hat{y}_i) +
                      (1-y_i) \log(1-\hat{y}_i)
                \right) - R(\mathbf{b}, \gamma) 
            \right]
        
        where
        
        .. math::
        
            R(\mathbf{b}, \gamma) = \left\{
                \begin{array}{lr}
                    \frac{\gamma}{2} ||\mathbf{beta}||_2^2 & :\texttt{ penalty = 'l2'}\\
                    \gamma ||\beta||_1 & :\texttt{ penalty = 'l1'}
                \end{array}
                \right.
                
        is a regularization penalty, :math:`\gamma` is a regularization weight, 
        `N` is the number of examples in **y**, and **b** is the vector of model 
        coefficients.

        Parameters
        ----------
        penalty : {'l1', 'l2'}
            The type of regularization penalty to apply on the coefficients
            `beta`. Default is 'l2'.
        gamma : float
            The regularization weight. Larger values correspond to larger
            regularization penalties, and a value of 0 indicates no penalty.
            Default is 0.
        fit_intercept : bool
            Whether to fit an intercept term in addition to the coefficients in
            b. If True, the estimates for `beta` will have `M + 1` dimensions,
            where the first dimension corresponds to the intercept. Default is
            True.
        """
        err_msg = "penalty must be 'l1' or 'l2', but got: {}".format(penalty)
        assert penalty in ["l2", "l1"], err_msg
        self.beta = None
        self.gamma = gamma
        self.penalty = penalty
        self.fit_intercept = fit_intercept

    def fit(self, X, y, lr=0.01, tol=1e-7, max_iter=1e7):
        """
        Fit the regression coefficients via gradient descent on the negative
        log likelihood.

        Parameters
        ----------
        X : :py:class:`ndarray <numpy.ndarray>` of shape `(N, M)`
            A dataset consisting of `N` examples, each of dimension `M`.
        y : :py:class:`ndarray <numpy.ndarray>` of shape `(N,)`
            The binary targets for each of the `N` examples in `X`.
        lr : float
            The gradient descent learning rate. Default is 1e-7.
        max_iter : float
            The maximum number of iterations to run the gradient descent
            solver. Default is 1e7.
        """
        # convert X to a design matrix if we're fitting an intercept
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]

        l_prev = np.inf
        self.beta = np.random.rand(X.shape[1])
        for _ in range(int(max_iter)):
            y_pred = sigmoid(np.dot(X, self.beta))
            loss = self._NLL(X, y, y_pred)
            if l_prev - loss < tol:
                return
            l_prev = loss
            self.beta -= lr * self._NLL_grad(X, y, y_pred)

    def _NLL(self, X, y, y_pred):
        r"""
        Penalized negative log likelihood of the targets under the current
        model.

        .. math::

            \text{NLL} = -\frac{1}{N} \left[
                \left(
                    \sum_{i=0}^N y_i \log(\hat{y}_i) + (1-y_i) \log(1-\hat{y}_i)
                \right) - R(\mathbf{b}, \gamma)
            \right]
        """
        N, M = X.shape
        beta, gamma = self.beta, self.gamma 
        order = 2 if self.penalty == "l2" else 1
        norm_beta = np.linalg.norm(beta, ord=order)
        
        nll = -np.log(y_pred[y == 1]).sum() - np.log(1 - y_pred[y == 0]).sum()
        penalty = (gamma / 2) * norm_beta ** 2 if order == 2 else gamma * norm_beta
        return (penalty + nll) / N

    def _NLL_grad(self, X, y, y_pred):
        """Gradient of the penalized negative log likelihood wrt beta"""
        N, M = X.shape
        l1norm = lambda x: np.linalg.norm(x, 1)  # noqa: E731
        p, beta, gamma = self.penalty, self.beta, self.gamma
        d_penalty = gamma * beta if p == "l2" else gamma * np.sign(beta)
        return -(np.dot(y - y_pred, X) + d_penalty) / N

    def predict(self, X):
        """
        Use the trained model to generate prediction probabilities on a new
        collection of data points.

        Parameters
        ----------
        X : :py:class:`ndarray <numpy.ndarray>` of shape `(Z, M)`
            A dataset consisting of `Z` new examples, each of dimension `M`.

        Returns
        -------
        y_pred : :py:class:`ndarray <numpy.ndarray>` of shape `(Z,)`
            The model prediction probabilities for the items in `X`.
        """
        # convert X to a design matrix if we're fitting an intercept
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]
        return sigmoid(np.dot(X, self.beta))


class BayesianLinearRegressionUnknownVariance:
    def __init__(self, alpha=1, beta=2, b_mean=0, b_V=None, fit_intercept=True):
        r"""
        Bayesian linear regression model with unknown variance and conjugate
        Normal-Gamma prior on `b` and :math:`\sigma^2`.

        Notes
        -----
        Uses a conjugate Normal-Gamma prior on `b` and :math:`\sigma^2`. The
        joint and marginal posteriors over error variance and model parameters
        are:

        .. math::

            b, \sigma^2 &\sim \text{NG}(b_{mean}, b_{V}, \alpha, \beta) \\
            \sigma^2 &\sim \text{InverseGamma}(\alpha, \beta) \\
            b &\sim \mathcal{N}(b_{mean}, \sigma^2 \cdot b_V)

        Parameters
        ----------
        alpha : float
            The shape parameter for the Inverse-Gamma prior on
            :math:`\sigma^2`. Must be strictly greater than 0. Default is 1.
        beta : float
            The scale parameter for the Inverse-Gamma prior on
            :math:`\sigma^2`. Must be strictly greater than 0. Default is 1.
        b_mean : :py:class:`ndarray <numpy.ndarray>` of shape `(M,)` or float
            The mean of the Gaussian prior on `b`. If a float, assume `b_mean`
            is ``np.ones(M) * b_mean``. Default is 0.
        b_V : :py:class:`ndarray <numpy.ndarray>` of shape `(N, N)` or `(N,)` or None
            A symmetric positive definite matrix that when multiplied
            element-wise by :math:`b_sigma^2` gives the covariance matrix for
            the Gaussian prior on `b`. If a list, assume ``b_V =
            diag(b_V)``. If None, assume `b_V` is the identity matrix.
            Default is None.
        fit_intercept : bool
            Whether to fit an intercept term in addition to the coefficients in
            b. If True, the estimates for b will have `M + 1` dimensions, where
            the first dimension corresponds to the intercept. Default is True.
        """
        # this is a placeholder until we know the dimensions of X
        b_V = 1.0 if b_V is None else b_V

        if isinstance(b_V, list):
            b_V = np.array(b_V)

        if isinstance(b_V, np.ndarray):
            if b_V.ndim == 1:
                b_V = np.diag(b_V)
            elif b_V.ndim == 2:
                fstr = "b_V must be symmetric positive definite"
                assert is_symmetric_positive_definite(b_V), fstr

        self.b_V = b_V
        self.beta = beta
        self.alpha = alpha
        self.b_mean = b_mean
        self.fit_intercept = fit_intercept
        self.posterior = {"mu": None, "cov": None}
        self.posterior_predictive = {"mu": None, "cov": None}

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
        beta = self.beta
        self.X, self.y = X, y

        if is_number(self.b_V):
            self.b_V *= np.eye(M)

        if is_number(self.b_mean):
            self.b_mean *= np.ones(M)

        # sigma
        I = np.eye(N)  # noqa: E741
        a = y - np.dot(X, self.b_mean)
        b = np.linalg.inv(np.dot(X, self.b_V).dot(X.T) + I)
        c = y - np.dot(X, self.b_mean)

        shape = N + self.alpha
        sigma = (1 / shape) * (self.alpha * beta ** 2 + np.dot(a, b).dot(c))
        scale = sigma ** 2

        # b_sigma is the mode of the inverse gamma prior on sigma^2
        b_sigma = scale / (shape - 1)

        # mean
        b_V_inv = np.linalg.inv(self.b_V)
        L = np.linalg.inv(b_V_inv + X.T @ X)
        R = b_V_inv @ self.b_mean + X.T @ y

        mu = L @ R
        cov = L * b_sigma

        # posterior distribution for sigma^2 and c
        self.posterior = {
            "sigma**2": {"dist": "InvGamma", "shape": shape, "scale": scale},
            "b | sigma**2": {"dist": "Gaussian", "mu": mu, "cov": cov},
        }

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
        mu = np.dot(X, self.posterior["b | sigma**2"]["mu"])
        cov = np.dot(X, self.posterior["b | sigma**2"]["cov"]).dot(X.T) + I

        # MAP estimate for y corresponds to the mean of the posterior
        # predictive
        self.posterior_predictive["mu"] = mu
        self.posterior_predictive["cov"] = cov
        return mu


class BayesianLinearRegressionKnownVariance:
    def __init__(self, b_mean=0, b_sigma=1, b_V=None, fit_intercept=True):
        r"""
        Bayesian linear regression model with known error variance and
        conjugate Gaussian prior on model parameters.

        Notes
        -----
        Uses a conjugate Gaussian prior on the model coefficients. The
        posterior over model parameters is

        .. math::

            b \mid b_{mean}, \sigma^2, b_V \sim \mathcal{N}(b_{mean}, \sigma^2 b_V)

        Ridge regression is a special case of this model where :math:`b_{mean}`
        = 0, :math:`\sigma` = 1 and `b_V` = I (ie., the prior on `b` is a
        zero-mean, unit covariance Gaussian).

        Parameters
        ----------
        b_mean : :py:class:`ndarray <numpy.ndarray>` of shape `(M,)` or float
            The mean of the Gaussian prior on `b`. If a float, assume `b_mean` is
            ``np.ones(M) * b_mean``. Default is 0.
        b_sigma : float
            A scaling term for covariance of the Gaussian prior on `b`. Default
            is 1.
        b_V : :py:class:`ndarray <numpy.ndarray>` of shape `(N,N)` or `(N,)` or None
            A symmetric positive definite matrix that when multiplied
            element-wise by `b_sigma^2` gives the covariance matrix for the
            Gaussian prior on `b`. If a list, assume ``b_V = diag(b_V)``. If None,
            assume `b_V` is the identity matrix. Default is None.
        fit_intercept : bool
            Whether to fit an intercept term in addition to the coefficients in
            b. If True, the estimates for b will have `M + 1` dimensions, where
            the first dimension corresponds to the intercept. Default is True.
        """
        # this is a placeholder until we know the dimensions of X
        b_V = 1.0 if b_V is None else b_V

        if isinstance(b_V, list):
            b_V = np.array(b_V)

        if isinstance(b_V, np.ndarray):
            if b_V.ndim == 1:
                b_V = np.diag(b_V)
            elif b_V.ndim == 2:
                fstr = "b_V must be symmetric positive definite"
                assert is_symmetric_positive_definite(b_V), fstr

        self.posterior = {}
        self.posterior_predictive = {}

        self.b_V = b_V
        self.b_mean = b_mean
        self.b_sigma = b_sigma
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
        self.X, self.y = X, y

        if is_number(self.b_V):
            self.b_V *= np.eye(M)

        if is_number(self.b_mean):
            self.b_mean *= np.ones(M)

        b_V = self.b_V
        b_mean = self.b_mean
        b_sigma = self.b_sigma

        b_V_inv = np.linalg.inv(b_V)
        L = np.linalg.inv(b_V_inv + X.T @ X)
        R = b_V_inv @ b_mean + X.T @ y

        mu = L @ R
        cov = L * b_sigma ** 2

        # posterior distribution over b conditioned on b_sigma
        self.posterior["b"] = {"dist": "Gaussian", "mu": mu, "cov": cov}

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
        mu = np.dot(X, self.posterior["b"]["mu"])
        cov = np.dot(X, self.posterior["b"]["cov"]).dot(X.T) + I

        # MAP estimate for y corresponds to the mean of the posterior
        # predictive distribution
        self.posterior_predictive = {"dist": "Gaussian", "mu": mu, "cov": cov}
        return mu


#######################################################################
#                                Utils                                #
#######################################################################


def sigmoid(x):
    """The logistic sigmoid function"""
    return 1 / (1 + np.exp(-x))
