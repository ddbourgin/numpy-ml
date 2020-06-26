"""Algorithms for approximate matrix factorization"""

from copy import deepcopy

import numpy as np


class VanillaALS:
    def __init__(self, K, alpha=1, max_iter=200, tol=1e-4):
        r"""
        Approximately factor a real-valued matrix using regularized alternating
        least-squares (ALS).

        Notes
        -----
        The regularized ALS minimization problem is

        .. math::

            \min_{\mathbf{W}, \mathbf{H}} ||\mathbf{X} - \mathbf{WH}||^2 -
                \alpha \left(
                    ||\mathbf{W}||^2 + ||\mathbf{H}||^2
                \right)

        where :math:`||\cdot||` denotes the Frobenius norm, **X** is the
        :math:`N \times M` data matrix, :math:`\mathbf{W}` and
        :math:`\mathbf{H}` are learned factor matrices with dimensions :math:`N
        \times K` and :math:`K \times M`, respectively, and :math:`\alpha` is a
        user-defined regularization weight.

        ALS proceeds by alternating between fixing **W** and optimizing for
        **H** and fixing **H** and optimizing for **W**. Vanilla ALS has no
        convergance guarantees and the objective function is prone to
        oscillation across updates, particularly for dense input matrices [1]_.

        References
        ----------
        .. [1] Gillis, N. (2014). The why and how of nonnegative matrix
           factorization.  *Regularization, optimization, kernels, and support
           vector machines, 12(257)*, 257-291.

        Parameters
        ----------
        K : int
            The number of latent factors to include in the factor matrices W
            and H.
        alpha : float
            The L2 regularization weight on the factor matrices. Larger
            values result in more aggressive regularization. Default is 1.
        max_iter : int
            The maximum number of iterations to run before stopping. Default is
            200.
        tol : float
            The tolerance for the stopping condition. Default is 1e-4.
        """
        self.K = K
        self.W = None
        self.H = None
        self.tol = tol
        self.alpha = alpha
        self.max_iter = max_iter

    @property
    def parameters(self):
        """Return a dictionary of the current model parameters"""
        return {"W": self.W, "H": self.H}

    @property
    def hyperparameters(self):
        """Return a dictionary of the model hyperparameters"""
        return {
            "id": "ALSFactor",
            "K": self.K,
            "tol": self.tol,
            "alpha": self.alpha,
            "max_iter": self.max_iter,
        }

    def _init_factor_matrices(self, X, W=None, H=None):
        """Randomly initialize the factor matrices"""
        N, M = X.shape
        scale = np.sqrt(X.mean() / self.K)
        self.W = np.random.rand(N, self.K) * scale if W is None else W
        self.H = np.random.rand(self.K, M) * scale if H is None else H

        assert self.W.shape == (N, self.K)
        assert self.H.shape == (self.K, M)

    def _loss(self, X, Xhat):
        """Regularized Frobenius loss"""
        alpha, W, H = self.alpha, self.W, self.H
        sq_fnorm = lambda x: np.sum(x ** 2)  # noqa: E731
        return sq_fnorm(X - Xhat) + alpha * (sq_fnorm(W) + sq_fnorm(H))

    def _update_factor(self, X, A):
        """Perform the ALS update"""
        T1 = np.linalg.inv(A.T @ A + self.alpha * np.eye(self.K))
        return X @ A @ T1

    def fit(self, X, W=None, H=None, n_initializations=10, verbose=False):
        """
        Factor a data matrix into two low rank factors via ALS.

        Parameters
        ----------
        X : numpy array of shape `(N, M)`
            The data matrix to factor.
        W : numpy array of shape `(N, K)` or None
            An initial value for the `W` factor matrix. If None, initialize `W`
            randomly. Default is None.
        H : numpy array of shape `(K, M)` or None
            An initial value for the `H` factor matrix. If None, initialize `H`
            randomly. Default is None.
        n_initializations : int
            Number of re-initializations of the algorithm to perform before
            taking the answer with the lowest reconstruction error. This value
            is ignored and set to 1 if both `W` and `H` are not None. Default
            is 10.
        verbose : bool
            Whether to print the loss at each iteration. Default is False.
        """
        if W is not None and H is not None:
            n_initializations = 1

        best_loss = np.inf
        for f in range(n_initializations):
            if verbose:
                print("\nINITIALIZATION {}".format(f + 1))

            new_W, new_H, loss = self._fit(X, W, H, verbose)

            if loss <= best_loss:
                best_loss = loss
                best_W, best_H = deepcopy(new_W), deepcopy(new_H)

        self.W, self.H = best_W, best_H

        if verbose:
            print("\nFINAL LOSS: {}".format(best_loss))

    def _fit(self, X, W, H, verbose):
        self._init_factor_matrices(X, W, H)
        W, H = self.W, self.H

        for i in range(self.max_iter):
            W = self._update_factor(X, H.T)
            H = self._update_factor(X.T, W).T

            loss = self._loss(X, W @ H)

            if verbose:
                print("[Iter {}] Loss: {:.8f}".format(i + 1, loss))

            if loss <= self.tol:
                break

        return W, H, loss


class NMF:
    def __init__(self, K, max_iter=200, tol=1e-4):
        r"""
        Nonnegative matrix factorization (NMF) performed using fast
        hierarchical alternating least squares (HALS) [*]_.

        Notes
        -----
        The NMF minimization problem is

        .. math::

            \min_{\mathbf{W}, \mathbf{H}} ||\mathbf{X} - \mathbf{WH}||^2
                \ \ \ \ \text{subject to } \mathbf{W}, \mathbf{H} \geq 0

        where :math:`||\cdot||` denotes the Frobenius norm, and the notation
        :math:`\mathbf{A} \geq 0` indicates that each element of **A** is
        greater than or equal to 0. In the above equation, **X** is the
        :math:`N \times M` data matrix, :math:`\mathbf{W}` and
        :math:`\mathbf{H}` are learned factor matrices with dimensions :math:`N
        \times K` and :math:`K \times M`, respectively.

        As with other ALS-based approaches, there is no guarantee that NMF will
        converge to a stationary point, let alone a global minimum. As a result
        it is generally good practice to run the algorithm multiple times with
        different initializations, taking the outcome that achieves the lowest
        reconstruction error.

        References
        ----------
        .. [*] Cichocki, A., & Phan, A. (2009). Fast local algorithms for
           large scale nonnegative matrix and tensor factorizations. *IEICE
           Transactions on Fundamentals of Electronics, Communications and
           Computer Sciences, 92(3)*, 708-721.

        Parameters
        ----------
        K : int
            The number of latent factors to include in the factor matrices **W**
            and **H**.
        max_iter : int
            The maximum number of iterations to run before stopping. Default is
            200.
        tol : float
            The tolerance for the stopping condition. Default is 1e-4.
        """
        self.K = K
        self.W = None
        self.H = None
        self.tol = tol
        self.max_iter = max_iter

    @property
    def parameters(self):
        """Return a dictionary of the current model parameters"""
        return {"W": self.W, "H": self.H}

    @property
    def hyperparameters(self):
        """Return a dictionary of the model hyperparameters"""
        return {
            "id": "NMF",
            "K": self.K,
            "tol": self.tol,
            "max_iter": self.max_iter,
        }

    def _init_factor_matrices(self, X, W, H):
        """Initialize the factor matrices using vanilla ALS"""
        ALS = None
        N, M = X.shape

        # initialize factors using ALS if not already defined
        if W is None:
            ALS = VanillaALS(self.K, alpha=0, max_iter=200)
            ALS.fit(X, verbose=False)
            W = ALS.W / np.linalg.norm(ALS.W, axis=0)

        if H is None:
            H = np.abs(np.random.rand(self.K, M)) if ALS is None else ALS.H

        assert W.shape == (N, self.K)
        assert H.shape == (self.K, M)

        self.H = H
        self.W = W

    def _loss(self, X, Xhat):
        """Return the least-squares reconstruction loss between X and Xhat"""
        return np.sum((X - Xhat) ** 2)

    def _update_H(self, X, W, H):
        """Perform the fast HALS update for H"""
        eps = np.finfo(float).eps
        XtW = X.T @ W  # dim: (M, K)
        WtW = W.T @ W  # dim: (K, K)

        for k in range(self.K):
            H[k, :] += XtW[:, k] - H.T @ WtW[:, k]
            H[k, :] = np.clip(H[k, :], eps, np.inf)  # enforce nonnegativity
        return H

    def _update_W(self, X, W, H):
        """Perform the fast HALS update for W"""
        eps = np.finfo(float).eps
        XHt = X @ H.T  # dim: (N, K)
        HHt = H @ H.T  # dim: (K, K)

        for k in range(self.K):
            W[:, k] = W[:, k] * HHt[k, k] + XHt[:, k] - W @ HHt[:, k]
            W[:, k] = np.clip(W[:, k], eps, np.inf)  # enforce nonnegativity

            # renormalize the new column
            n = np.linalg.norm(W[:, k])
            W[:, k] /= n if n > 0 else 1.0
        return W

    def fit(self, X, W=None, H=None, n_initializations=10, verbose=False):
        r"""
        Factor a data matrix into two nonnegative low rank factor matrices via
        fast HALS.

        Notes
        -----
        This method implements Algorithm 2 from [*]_. In contrast to vanilla
        ALS, HALS proceeds by minimizing a *set* of local cost functions with
        the same global minima. Each cost function is defined on a "residue" of
        the factor matrices **W** and **H**:

        .. math::

           \mathbf{X}^{(j)} :=
                \mathbf{X} - \mathbf{WH}^\top + \mathbf{w}_j \mathbf{h}_j^\top

        where :math:`\mathbf{X}^{(j)}` is the :math:`j^{th}` residue, **X** is
        the input data matrix, and :math:`\mathbf{w}_j` and
        :math:`\mathbf{h}_j` are the :math:`j^{th}` columns of the current
        factor matrices **W** and **H**. HALS proceeds by minimizing the cost
        for each residue, first with respect to :math:`\mathbf{w}_j`, and then
        with respect to :math:`\mathbf{h}_j`. In either case, the cost for
        residue `j`, :math:`\mathcal{L}^{(j)}` is simply:

        .. math::

            \mathcal{L}^{(j)} :=
                || \mathbf{X}^{(j)} - \mathbf{w}_j \mathbf{h}_j^\top ||

        where :math:`||\cdot||` denotes the Frobenius norm. For NMF,
        minimization is performed under the constraint that all elements of
        both **W** and **H** are nonnegative.

        References
        ----------
        .. [*] Cichocki, A., & Phan, A. (2009). Fast local algorithms for
           large scale nonnegative matrix and tensor factorizations. *IEICE
           Transactions on Fundamentals of Electronics, Communications and
           Computer Sciences, 92(3)*, 708-721.

        Parameters
        ----------
        X : numpy array of shape `(N, M)`
            The data matrix to factor.
        W : numpy array of shape `(N, K)` or None
            An initial value for the `W` factor matrix. If None, initialize
            **W** using vanilla ALS. Default is None.
        H : numpy array of shape `(K, M)` or None
            An initial value for the `H` factor matrix. If None, initialize
            **H** using vanilla ALS. Default is None.
        n_initializations : int
            Number of re-initializations of the algorithm to perform before
            taking the answer with the lowest reconstruction error. This value
            is ignored and set to 1 if both `W` and `H` are not None. Default
            is 10.
        verbose : bool
            Whether to print the loss at each iteration. Default is False.
        """
        if W is not None and H is not None:
            n_initializations = 1

        best_loss = np.inf
        for f in range(n_initializations):
            if verbose:
                print("\nINITIALIZATION {}".format(f + 1))

            new_W, new_H, loss = self._fit(X, W, H, verbose)

            if loss <= best_loss:
                best_loss = loss
                best_W, best_H = deepcopy(new_W), deepcopy(new_H)

        self.W, self.H = best_W, best_H
        if verbose:
            print("\nFINAL LOSS: {}".format(best_loss))

    def _fit(self, X, W, H, verbose):
        self._init_factor_matrices(X, W, H)

        W, H = self.W, self.H
        for i in range(self.max_iter):
            H = self._update_H(X, W, H)
            W = self._update_W(X, W, H)
            loss = self._loss(X, W @ H)

            if verbose:
                print("[Iter {}] Loss: {:.8f}".format(i + 1, loss))

            if loss <= self.tol:
                break
        return W, H, loss
