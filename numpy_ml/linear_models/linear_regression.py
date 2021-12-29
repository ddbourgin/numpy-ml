"""Linear regression module."""

import numpy as np


class LinearRegression:
    def __init__(self, fit_intercept=True):
        r"""
        A weighted linear least-squares regression model.

        Notes
        -----
        In weighted linear least-squares regression [1]_, a real-valued target
        vector, **y**, is modeled as a linear combination of covariates, **X**,
        and model coefficients, :math:`\beta`:

        .. math::

            y_i = \beta^\top \mathbf{x}_i + \epsilon_i

        In this equation :math:`\epsilon_i \sim \mathcal{N}(0, \sigma^2_i)` is
        the error term associated with example :math:`i`, and
        :math:`\sigma^2_i` is the variance of the corresponding example.

        Under this model, the maximum-likelihood estimate for the regression
        coefficients, :math:`\beta`, is:

        .. math::

            \hat{\beta} = \Sigma^{-1} \mathbf{X}^\top \mathbf{Wy}

        where :math:`\Sigma^{-1} = (\mathbf{X}^\top \mathbf{WX})^{-1}` and
        **W** is a diagonal matrix of weights, with each entry inversely
        proportional to the variance of the corresponding measurement. When
        **W** is the identity matrix the examples are weighted equally and the
        model reduces to standard linear least squares [2]_.

        References
        ----------
        .. [1] https://en.wikipedia.org/wiki/Weighted_least_squares
        .. [2] https://en.wikipedia.org/wiki/General_linear_model

        Parameters
        ----------
        fit_intercept : bool
            Whether to fit an intercept term in addition to the model
            coefficients. Default is True.

        Attributes
        ----------
        beta : :py:class:`ndarray <numpy.ndarray>` of shape `(M, K)` or None
            Fitted model coefficients.
        sigma_inv : :py:class:`ndarray <numpy.ndarray>` of shape `(N, N)` or None
            Inverse of the data covariance matrix.
        """
        self.beta = None
        self.sigma_inv = None
        self.fit_intercept = fit_intercept

        self._is_fit = False

    def update(self, X, y, weights=None):
        r"""
        Incrementally update the linear least-squares coefficients for a set of
        new examples.

        Notes
        -----
        The recursive least-squares algorithm [3]_ [4]_ is used to efficiently
        update the regression parameters as new examples become available. For
        a single new example :math:`(\mathbf{x}_{t+1}, \mathbf{y}_{t+1})`, the
        parameter updates are

        .. math::

            \beta_{t+1} = \left(
                \mathbf{X}_{1:t}^\top \mathbf{X}_{1:t} +
                    \mathbf{x}_{t+1}\mathbf{x}_{t+1}^\top \right)^{-1}
                        \mathbf{X}_{1:t}^\top \mathbf{Y}_{1:t} +
                            \mathbf{x}_{t+1}^\top \mathbf{y}_{t+1}

        where :math:`\beta_{t+1}` are the updated regression coefficients,
        :math:`\mathbf{X}_{1:t}` and :math:`\mathbf{Y}_{1:t}` are the set of
        examples observed from timestep 1 to *t*.

        In the single-example case, the RLS algorithm uses the Sherman-Morrison
        formula [5]_ to avoid re-inverting the covariance matrix on each new
        update. In the multi-example case (i.e., where :math:`\mathbf{X}_{t+1}`
        and :math:`\mathbf{y}_{t+1}` are matrices of `N` examples each), we use
        the generalized Woodbury matrix identity [6]_ to update the inverse
        covariance. This comes at a performance cost, but is still more
        performant than doing multiple single-example updates if *N* is large.

        References
        ----------
        .. [3] Gauss, C. F. (1821) *Theoria combinationis observationum
           erroribus minimis obnoxiae*, Werke, 4. Gottinge
        .. [4] https://en.wikipedia.org/wiki/Recursive_least_squares_filter
        .. [5] https://en.wikipedia.org/wiki/Sherman%E2%80%93Morrison_formula
        .. [6] https://en.wikipedia.org/wiki/Woodbury_matrix_identity

        Parameters
        ----------
        X : :py:class:`ndarray <numpy.ndarray>` of shape `(N, M)`
            A dataset consisting of `N` examples, each of dimension `M`
        y : :py:class:`ndarray <numpy.ndarray>` of shape `(N, K)`
            The targets for each of the `N` examples in `X`, where each target
            has dimension `K`
        weights : :py:class:`ndarray <numpy.ndarray>` of shape `(N,)` or None
            Weights associated with the examples in `X`. Examples
            with larger weights exert greater influence on model fit.  When
            `y` is a vector (i.e., `K = 1`), weights should be set to the
            reciporical of the variance for each measurement (i.e., :math:`w_i
            = 1/\sigma^2_i`). When `K > 1`, it is assumed that all columns of
            `y` share the same weight :math:`w_i`. If None, examples are
            weighted equally, resulting in the standard linear least squares
            update.  Default is None.

        Returns
        -------
        self : :class:`LinearRegression <numpy_ml.linear_models.LinearRegression>` instance
        """  # noqa: E501
        if not self._is_fit:
            raise RuntimeError("You must call the `fit` method before calling `update`")

        X, y = np.atleast_2d(X), np.atleast_2d(y)

        X1, Y1 = X.shape[0], y.shape[0]
        weights = np.ones(X1) if weights is None else np.atleast_1d(weights)
        weights = np.squeeze(weights) if weights.size > 1 else weights

        err_str = f"weights must have shape ({X1},) but got {weights.shape}"
        assert weights.shape == (X1,), err_str

        # scale X and y by the weight associated with each example
        W = np.diag(np.sqrt(weights))
        X, y = W @ X, W @ y

        self._update1D(X, y, W) if X1 == Y1 == 1 else self._update2D(X, y, W)
        return self

    def _update1D(self, x, y, w):
        """Sherman-Morrison update for a single example"""
        beta, S_inv = self.beta, self.sigma_inv

        # convert x to a design vector if we're fitting an intercept
        if self.fit_intercept:
            x = np.c_[np.diag(w), x]

        # update the inverse of the covariance matrix via Sherman-Morrison
        S_inv -= (S_inv @ x.T @ x @ S_inv) / (1 + x @ S_inv @ x.T)

        # update the model coefficients
        beta += S_inv @ x.T @ (y - x @ beta)

    def _update2D(self, X, y, W):
        """Woodbury update for multiple examples"""
        beta, S_inv = self.beta, self.sigma_inv

        # convert X to a design matrix if we're fitting an intercept
        if self.fit_intercept:
            X = np.c_[np.diag(W), X]

        I = np.eye(X.shape[0])  # noqa: E741

        # update the inverse of the covariance matrix via Woodbury identity
        S_inv -= S_inv @ X.T @ np.linalg.pinv(I + X @ S_inv @ X.T) @ X @ S_inv

        # update the model coefficients
        beta += S_inv @ X.T @ (y - X @ beta)

    def fit(self, X, y, weights=None):
        r"""
        Fit regression coefficients via maximum likelihood.

        Parameters
        ----------
        X : :py:class:`ndarray <numpy.ndarray>` of shape `(N, M)`
            A dataset consisting of `N` examples, each of dimension `M`.
        y : :py:class:`ndarray <numpy.ndarray>` of shape `(N, K)`
            The targets for each of the `N` examples in `X`, where each target
            has dimension `K`.
        weights : :py:class:`ndarray <numpy.ndarray>` of shape `(N,)` or None
            Weights associated with the examples in `X`. Examples
            with larger weights exert greater influence on model fit.  When
            `y` is a vector (i.e., `K = 1`), weights should be set to the
            reciporical of the variance for each measurement (i.e., :math:`w_i
            = 1/\sigma^2_i`). When `K > 1`, it is assumed that all columns of
            `y` share the same weight :math:`w_i`. If None, examples are
            weighted equally, resulting in the standard linear least squares
            update.  Default is None.

        Returns
        -------
        self : :class:`LinearRegression <numpy_ml.linear_models.LinearRegression>` instance
        """  # noqa: E501
        N = X.shape[0]

        weights = np.ones(N) if weights is None else np.atleast_1d(weights)
        weights = np.squeeze(weights) if weights.size > 1 else weights
        err_str = f"weights must have shape ({N},) but got {weights.shape}"
        assert weights.shape == (N,), err_str

        # scale X and y by the weight associated with each example
        W = np.diag(np.sqrt(weights))
        X, y = W @ X, W @ y

        # convert X to a design matrix if we're fitting an intercept
        if self.fit_intercept:
            X = np.c_[np.sqrt(weights), X]

        self.sigma_inv = np.linalg.pinv(X.T @ X)
        self.beta = np.atleast_2d(self.sigma_inv @ X.T @ y)

        self._is_fit = True
        return self

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
        return X @ self.beta
