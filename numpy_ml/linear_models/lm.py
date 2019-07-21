import numpy as np
from ..utils.testing import is_symmetric_positive_definite, is_number


class LinearRegression:
    """
    The simple linear regression model is

        y = bX + e  where e ~ N(0, sigma^2 * I)

    In probabilistic terms this corresponds to

        y - bX ~ N(0, sigma^2 * I)
        y | X, b ~ N(bX, sigma^2 * I)

    The loss for the model is simply the squared error between the model
    predictions and the true values:

        Loss = ||y - bX||^2

    The MLE for the model parameters b can be computed in closed form via the
    normal equation:

        b = (X^T X)^{-1} X^T y

    where (X^T X)^{-1} X^T is known as the pseudoinverse / Moore-Penrose
    inverse.
    """

    def __init__(self, fit_intercept=True):
        self.beta = None
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        # convert X to a design matrix if we're fitting an intercept
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]

        pseudo_inverse = np.dot(np.linalg.inv(np.dot(X.T, X)), X.T)
        self.beta = np.dot(pseudo_inverse, y)

    def predict(self, X):
        # convert X to a design matrix if we're fitting an intercept
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]
        return np.dot(X, self.beta)


class RidgeRegression:
    """
    Ridge regression uses the same simple linear regression model but adds an
    additional penalty on the L2-norm of the coefficients to the loss function.
    This is sometimes known as Tikhonov regularization.

    In particular, the ridge model is still simply

        y = bX + e  where e ~ N(0, sigma^2 * I)

    except now the error for the model is calcualted as

        RidgeLoss = ||y - bX||^2 + alpha * ||b||^2

    The MLE for the model parameters b can be computed in closed form via the
    adjusted normal equation:

        b = (X^T X + alpha I)^{-1} X^T y

    where (X^T X + alpha I)^{-1} X^T is the pseudoinverse / Moore-Penrose
    inverse adjusted for the L2 penalty on the model coefficients.
    """

    def __init__(self, alpha=1, fit_intercept=True):
        """
        A ridge regression model fit via the normal equation.

        Parameters
        ----------
        alpha : float (default: 1)
            L2 regularization coefficient. Higher values correspond to larger
            penalty on the l2 norm of the model coefficients
        fit_intercept : bool (default: True)
            Whether to fit an additional intercept term in addition to the
            model coefficients
        """
        self.beta = None
        self.alpha = alpha
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        # convert X to a design matrix if we're fitting an intercept
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]

        A = self.alpha * np.eye(X.shape[1])
        pseudo_inverse = np.dot(np.linalg.inv(X.T @ X + A), X.T)
        self.beta = pseudo_inverse @ y

    def predict(self, X):
        # convert X to a design matrix if we're fitting an intercept
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]
        return np.dot(X, self.beta)


class LogisticRegression:
    def __init__(self, penalty="l2", gamma=0, fit_intercept=True):
        """
        A simple logistic regression model fit via gradient descent on the
        penalized negative log likelihood.

        Parameters
        ----------
        penalty : str (default: 'l2')
            The type of regularization penalty to apply on the coefficients
            `beta`. Valid entries are {'l2', 'l1'}.
        gamma : float in [0, 1] (default: 0)
            The regularization weight. Larger values correspond to larger
            regularization penalties, and a value of 0 indicates no penalty.
        fit_intercept : bool (default: True)
            Whether to fit an intercept term in addition to the coefficients in
            b. If True, the estimates for `beta` will have M+1 dimensions,
            where the first dimension corresponds to the intercept
        """
        err_msg = "penalty must be 'l1' or 'l2', but got: {}".format(penalty)
        assert penalty in ["l2", "l1"], err_msg
        self.beta = None
        self.gamma = gamma
        self.penalty = penalty
        self.fit_intercept = fit_intercept

    def fit(self, X, y, lr=0.01, tol=1e-7, max_iter=1e7):
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
        """
        Penalized negative log likelihood of the targets under the current
        model.

            NLL = -1/N * (
                [sum_{i=0}^N y_i log(y_pred_i) + (1-y_i) log(1-y_pred_i)] -
                (gamma ||b||) / 2
            )
        """
        N, M = X.shape
        order = 2 if self.penalty == "l2" else 1
        nll = -np.log(y_pred[y == 1]).sum() - np.log(1 - y_pred[y == 0]).sum()
        penalty = 0.5 * self.gamma * np.linalg.norm(self.beta, ord=order) ** 2
        return (penalty + nll) / N

    def _NLL_grad(self, X, y, y_pred):
        """ Gradient of the penalized negative log likelihood wrt beta """
        N, M = X.shape
        p = self.penalty
        beta = self.beta
        gamma = self.gamma
        l1norm = lambda x: np.linalg.norm(x, 1)
        d_penalty = gamma * beta if p == "l2" else gamma * l1norm(beta) * np.sign(beta)
        return -(np.dot(y - y_pred, X) + d_penalty) / N

    def predict(self, X):
        # convert X to a design matrix if we're fitting an intercept
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]
        return sigmoid(np.dot(X, self.beta))


class BayesianLinearRegressionUnknownVariance:
    """
    Bayesian Linear Regression
    --------------------------
    In its general form, Bayesian linear regression extends the simple linear
    regression model by introducing priors on model parameters b and/or the
    error variance sigma^2.

    The introduction of a prior allows us to quantify the uncertainty in our
    parameter estimates for b by replacing the MLE point estimate in simple
    linear regression with an entire posterior *distribution*, p(b | X, y,
    sigma), simply by applying Bayes rule:

        p(b | X, y) = [ p(y | X, b) * p(b | sigma) ] / p(y | X)

    We can also quantify the uncertainty in our predictions y* for some new
    data X* with the posterior predictive distribution:

        p(y* | X*, X, Y) = \int_{b} p(y* | X*, b) p(b | X, y) db

    Depending on the choice of prior it may be impossible to compute an
    analytic form for the posterior / posterior predictive distribution. In
    these cases, it is common to use approximations, either via MCMC or
    variational inference.

    Bayesian Regression w/ unknown variance
    ---------------------------------------
    If *both* b and the error variance sigma^2 are unknown, the conjugate prior
    for the Gaussian likelihood is the Normal-Gamma distribution (univariate
    likelihood) or the Normal-Inverse-Wishart distribution (multivariate
    likelihood).

        Univariate:
            b, sigma^2 ~ NG(b_mean, b_V, alpha, beta)

            sigma^2 ~ InverseGamma(alpha, beta)
            b | sigma^2 ~ N(b_mean, sigma^2 * b_V)

            where alpha, beta, b_V, and b_mean are parameters of the prior.

        Multivariate:
            b, Sigma ~ NIW(b_mean, lambda, Psi, rho)

            Sigma ~ N(b_mean, 1/lambda * Sigma)
            b | Sigma ~ W^{-1}(Psi, rho)

            where b_mean, lambda, Psi, and rho are parameters of the prior.

    Due to the conjugacy of the above priors with the Gaussian likelihood of
    the linear regression model we can compute the posterior distributions for
    the model parameters in closed form:

        B = (y - X b_mean)
        shape = N + alpha
        scale = (1 / shape) * {alpha * beta + B^T ([X b_V X^T + I])^{-1} B}

        sigma^2 | X, y ~ InverseGamma(shape, scale)

        A     = (b_V^{-1} + X^T X)^{-1}
        mu_b  = A b_V^{-1} b_mean + A X^T y
        cov_b = sigma^2 A

        b | X, y, sigma^2 ~ N(mu_b, cov_b)

    This allows us a closed form for the posterior predictive distribution as
    well:

        y* | X*, X, Y ~ N(X* mu_b, X* cov_b X*^T + I)
    """

    def __init__(self, alpha=1, beta=2, b_mean=0, b_V=None, fit_intercept=True):
        """
        Bayesian linear regression model with conjugate Normal-Gamma prior on b
        and sigma^2

            b, sigma^2 ~ NG(b_mean, b_V, alpha, beta)
            sigma^2 ~ InverseGamma(alpha, beta)
            b ~ N(b_mean, sigma^2 * b_V)

        Parameters
        ----------
        alpha : float (default: 1)
            The shape parameter for the Inverse-Gamma prior on sigma^2. Must be
            strictly greater than 0.
        beta : float (default: 1)
            The scale parameter for the Inverse-Gamma prior on sigma^2. Must be
            strictly greater than 0.
        b_mean : np.array of shape (M,) or float (default: 0)
            The mean of the Gaussian prior on b. If a float, assume b_mean is
            np.ones(M) * b_mean.
        b_V : np.array of shape (N, N) or np.array of shape (N,) or None
            A symmetric positive definite matrix that when multiplied
            element-wise by b_sigma^2 gives the covariance matrix for the
            Gaussian prior on b. If a list, assume b_V=diag(b_V). If None,
            assume b_V is the identity matrix.
        fit_intercept : bool (default: True)
            Whether to fit an intercept term in addition to the coefficients in
            b. If True, the estimates for b will have M+1 dimensions, where
            the first dimension corresponds to the intercept
        """
        # this is a placeholder until we know the dimensions of X
        b_V = 1.0 if b_V is None else b_V

        if isinstance(b_V, list):
            b_V = np.array(b_V)

        if isinstance(b_V, np.ndarray):
            if b_V.ndim == 1:
                b_V = np.diag(b_V)
            elif b_V.ndim == 2:
                assert is_symmetric_positive_definite(
                    b_V
                ), "b_V must be symmetric positive definite"

        self.b_V = b_V
        self.beta = beta
        self.alpha = alpha
        self.b_mean = b_mean
        self.fit_intercept = fit_intercept
        self.posterior = {"mu": None, "cov": None}
        self.posterior_predictive = {"mu": None, "cov": None}

    def fit(self, X, y):
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
        I = np.eye(N)
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
        l = np.linalg.inv(b_V_inv + np.dot(X.T, X))
        r = np.dot(b_V_inv, self.b_mean) + np.dot(X.T, y)
        mu = np.dot(l, r)
        cov = l * b_sigma

        # posterior distribution for sigma^2 and c
        self.posterior = {
            "sigma**2": {"dist": "InvGamma", "shape": shape, "scale": scale},
            "b | sigma**2": {"dist": "Gaussian", "mu": mu, "cov": cov},
        }

    def predict(self, X):
        # convert X to a design matrix if we're fitting an intercept
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]

        I = np.eye(X.shape[0])
        mu = np.dot(X, self.posterior["b | sigma**2"]["mu"])
        cov = np.dot(X, self.posterior["b | sigma**2"]["cov"]).dot(X.T) + I

        # MAP estimate for y corresponds to the mean of the posterior
        # predictive
        self.posterior_predictive["mu"] = mu
        self.posterior_predictive["cov"] = cov
        return mu


class BayesianLinearRegressionKnownVariance:
    """
    Bayesian Linear Regression
    --------------------------
    In its general form, Bayesian linear regression extends the simple linear
    regression model by introducing priors on model parameters b and/or the
    error variance sigma^2.

    The introduction of a prior allows us to quantify the uncertainty in our
    parameter estimates for b by replacing the MLE point estimate in simple
    linear regression with an entire posterior *distribution*, p(b | X, y,
    sigma), simply by applying Bayes rule:

        p(b | X, y) = [ p(y | X, b) * p(b | sigma) ] / p(y | X)

    We can also quantify the uncertainty in our predictions y* for some new
    data X* with the posterior predictive distribution:

        p(y* | X*, X, Y) = \int_{b} p(y* | X*, b) p(b | X, y) db

    Depending on the choice of prior it may be impossible to compute an
    analytic form for the posterior / posterior predictive distribution. In
    these cases, it is common to use approximations, either via MCMC or
    variational inference.

    Bayesian linear regression with known variance
    ----------------------------------------------
    If we happen to already know the error variance sigma^2, the conjugate
    prior on b is Gaussian. A common parameterization is:

        b | sigma, b_V ~ N(b_mean, sigma^2 * b_V)

    where b_mean, sigma and b_V are hyperparameters. Ridge regression is a
    special case of this model where b_mean = 0, sigma = 1 and b_V = I (ie.,
    the prior on b is a zero-mean, unit covariance Gaussian).

    Due to the conjugacy of the above prior with the Gaussian likelihood in the
    linear regression model, we can compute the posterior distribution over the
    model parameters in closed form:

        A     = (b_V^{-1} + X^T X)^{-1}
        mu_b  = A b_V^{-1} b_mean + A X^T y
        cov_b = sigma^2 A

        b | X, y ~ N(mu_b, cov_b)

    which allows us a closed form for the posterior predictive distribution as
    well:

        y* | X*, X, Y ~ N(X* mu_b, X* cov_b X*^T + I)
    """

    def __init__(self, b_mean=0, b_sigma=1, b_V=None, fit_intercept=True):
        """
        Bayesian linear regression model with known error variance and
        conjugate Gaussian prior on b

            b | b_mean, sigma^2, b_V ~ N(b_mean, sigma^2 * b_V)

        Ridge regression is a special case of this model where b_mean = 0,
        sigma = 1 and b_V = I (ie., the prior on b is a zero-mean, unit
        covariance Gaussian).

        Parameters
        ----------
        b_mean : np.array of shape (M,) or float (default: 0)
            The mean of the Gaussian prior on b. If a float, assume b_mean is
            np.ones(M) * b_mean.
        b_sigma : float (default: 1)
            A scaling term for covariance of the Gaussian prior on b
        b_V : np.array of shape (N,N) or np.array of shape (N,) or None
            A symmetric positive definite matrix that when multiplied
            element-wise by b_sigma^2 gives the covariance matrix for the
            Gaussian prior on b. If a list, assume b_V=diag(b_V). If None,
            assume b_V is the identity matrix.
        fit_intercept : bool (default: True)
            Whether to fit an intercept term in addition to the coefficients in
            b. If True, the estimates for b will have M+1 dimensions, where
            the first dimension corresponds to the intercept
        """
        # this is a placeholder until we know the dimensions of X
        b_V = 1.0 if b_V is None else b_V

        if isinstance(b_V, list):
            b_V = np.array(b_V)

        if isinstance(b_V, np.ndarray):
            if b_V.ndim == 1:
                b_V = np.diag(b_V)
            elif b_V.ndim == 2:
                assert is_symmetric_positive_definite(
                    b_V
                ), "b_V must be symmetric positive definite"

        self.posterior = {}
        self.posterior_predictive = {}

        self.b_V = b_V
        self.b_mean = b_mean
        self.b_sigma = b_sigma
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
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
        l = np.linalg.inv(b_V_inv + np.dot(X.T, X))
        r = np.dot(b_V_inv, b_mean) + np.dot(X.T, y)
        mu = np.dot(l, r)
        cov = l * b_sigma ** 2

        # posterior distribution over b conditioned on b_sigma
        self.posterior["b"] = {"dist": "Gaussian", "mu": mu, "cov": cov}

    def predict(self, X):
        # convert X to a design matrix if we're fitting an intercept
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]

        I = np.eye(X.shape[0])
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
    return 1 / (1 + np.exp(-x))
