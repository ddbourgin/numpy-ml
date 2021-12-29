"""A module for the generalized linear model."""
import numpy as np

from numpy_ml.linear_models.linear_regression import LinearRegression

eps = np.finfo(float).eps

_GLM_LINKS = {
    "logit": {
        "link": lambda mu: np.log((mu + eps) / (1 - mu + eps)),
        "inv_link": lambda eta: 1.0 / (1.0 + np.exp(-eta)),
        "link_prime": lambda x: (1 / (x + eps)) + (1 / (1 - x + eps)),
        "theta": lambda mu: np.log((mu + eps) / (1 - mu + eps)),
        "phi": lambda x: np.ones(x.shape[0]),
        "a": lambda phi: phi,
        "b": lambda theta: np.log(1 + np.exp(theta)),
        "p": 1,
        "b_prime": lambda theta: np.exp(theta) / (1 + np.exp(theta)),
        "b_prime2": lambda theta: np.exp(theta) / ((1 + np.exp(theta)) ** 2),
    },
    "identity": {
        "link": lambda mu: mu,
        "inv_link": lambda eta: eta,
        "link_prime": lambda x: np.ones_like(x),
        "theta": lambda mu: mu,
        "phi": lambda x: np.var(x, axis=0),
        "a": lambda phi: phi,
        "b": lambda theta: 0.5 * theta ** 2,
        "p": 1,
        "b_prime": lambda theta: theta,
        "b_prime2": lambda theta: np.ones_like(theta),
    },
    "log": {
        "link": lambda mu: np.log(mu + eps),
        "inv_link": lambda eta: np.exp(eta),
        "link_prime": lambda x: 1 / (x + eps),
        "theta": lambda mu: np.log(mu + eps),
        "phi": lambda x: np.ones(x.shape[0]),
        "a": lambda phi: phi,
        "p": 1,
        "b": lambda theta: np.exp(theta),
        "b_prime": lambda theta: np.exp(theta),
        "b_prime2": lambda theta: np.exp(theta),
    },
}


class GeneralizedLinearModel:
    def __init__(self, link, fit_intercept=True, tol=1e-5, max_iter=100):
        r"""
        A generalized linear model with maximum likelihood fit via
        iteratively reweighted least squares (IRLS).

        Notes
        -----
        The generalized linear model (GLM) [7]_ [8]_ assumes that each target/dependent
        variable :math:`y_i` in target vector :math:`\mathbf{y} = (y_1, \ldots,
        y_n)`, has been drawn independently from a pre-specified distribution
        in the exponential family [11]_ with unknown mean :math:`\mu_i`. The GLM
        models a (one-to-one, continuous, differentiable) function, *g*, of
        this mean value as a linear combination of the model parameters
        :math:`\mathbf{b}` and observed covariates, :math:`\mathbf{x}_i`:

        .. math::

            g(\mathbb{E}[y_i \mid \mathbf{x}_i]) =
                g(\mu_i) = \mathbf{b}^\top \mathbf{x}_i

        where *g* is known as the "link function" associated with the GLM.  The
        choice of link function is informed by the instance of the exponential
        family the target is drawn from. Common examples:

        .. csv-table::
           :header: "Distribution", "Link", "Formula"
           :widths: 25, 20, 30

           "Normal", "Identity", ":math:`g(x) = x`"
           "Bernoulli", "Logit", ":math:`g(x) = \log(x) - \log(1 - x)`"
           "Binomial", "Logit", ":math:`g(x) = \log(x) - \log(n - x)`"
           "Poisson", "Log", ":math:`g(x) = \log(x)`"

        An iteratively re-weighted least squares (IRLS) algorithm [9]_ can be
        employed to find the maximum likelihood estimate for the model
        parameters :math:`\beta` in any instance of the generalized linear
        model. IRLS is equivalent to Fisher scoring [10]_, which itself is
        a slight modification of classic Newton-Raphson for finding the zeros
        of the first derivative of the model log-likelihood.

        References
        ----------
        .. [7] Nelder, J., & Wedderburn, R. (1972). Generalized linear
               models. *Journal of the Royal Statistical Society, Series A
               (General), 135(3)*: 370–384.
        .. [8] https://en.wikipedia.org/wiki/Generalized_linear_model
        .. [9] https://en.wikipedia.org/wiki/Iteratively_reweighted_least_squares
        .. [10] https://en.wikipedia.org/wiki/Scoring_algorithm
        .. [11] https://en.wikipedia.org/wiki/Exponential_family

        Parameters
        ----------
        link: {'identity', 'logit', 'log'}
            The link function to use during modeling.
        fit_intercept: bool
            Whether to fit an intercept term in addition to the model
            coefficients. Default is True.
        tol : float
            The minimum difference between successive iterations of IRLS
            Default is 1e-5.
        max_iter: int
            The maximum number of iteratively reweighted least squares
            iterations to run during fitting. Default is 100.

        Attributes
        ----------
        beta : :py:class:`ndarray <numpy.ndarray>` of shape `(M, 1)` or None
            Fitted model coefficients.
        """
        err_str = f"Valid link functions are {list(_GLM_LINKS.keys())} but got {link}"
        assert link in _GLM_LINKS, err_str

        self._is_fit = False

        self.tol = tol
        self.link = link
        self.beta = None
        self.max_iter = max_iter
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        """
        Find the maximum likelihood GLM coefficients via IRLS.

        Parameters
        ----------
        X : :py:class:`ndarray <numpy.ndarray>` of shape `(N, M)`
            A dataset consisting of `N` examples, each of dimension `M`.
        y : :py:class:`ndarray <numpy.ndarray>` of shape `(N,)`
            The targets for each of the `N` examples in `X`.

        Returns
        -------
        self : :class:`GeneralizedLinearModel <numpy_ml.linear_models.GeneralizedLinearModel>` instance
        """  # noqa: E501
        y = np.squeeze(y)
        assert y.ndim == 1

        N, M = X.shape
        L = _GLM_LINKS[self.link]

        # starting values for parameters
        mu = np.ones_like(y) * np.mean(y)
        eta = L["link"](mu)
        theta = L["theta"](mu)

        # convert X to a design matrix if we're fitting an intercept
        if self.fit_intercept:
            X = np.c_[np.ones(N), X]

        # IRLS for GLM
        i = 0
        diff, beta = np.inf, np.inf
        while diff > (self.tol * M):
            if i > self.max_iter:
                print("Warning: Model did not converge")
                break

            # compute first-order Taylor approx.
            z = eta + (y - mu) * L["link_prime"](mu)
            w = L["p"] / (L["b_prime2"](theta) * L["link_prime"](mu) ** 2)

            # perform weighted least-squares on z
            wlr = LinearRegression(fit_intercept=False)
            beta_new = wlr.fit(X, z, weights=w).beta.ravel()

            eta = X @ beta_new
            mu = L["inv_link"](eta)
            theta = L["theta"](mu)

            diff = np.linalg.norm(beta - beta_new, ord=1)
            beta = beta_new
            i += 1

        self.beta = beta
        self._is_fit = True
        return self

    def predict(self, X):
        r"""
        Use the trained model to generate predictions for the distribution
        means, :math:`\mu`, associated with the collection of data points in
        **X**.

        Parameters
        ----------
        X : :py:class:`ndarray <numpy.ndarray>` of shape `(Z, M)`
            A dataset consisting of `Z` new examples, each of dimension `M`.

        Returns
        -------
        mu_pred : :py:class:`ndarray <numpy.ndarray>` of shape `(Z,)`
            The model predictions for the expected value of the target
            associated with each item in `X`.
        """
        assert self._is_fit, "Must call `fit` before generating predictions"
        L = _GLM_LINKS[self.link]

        # convert X to a design matrix if we're using an intercept
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]

        mu_pred = L["inv_link"](X @ self.beta)
        return mu_pred.ravel()
