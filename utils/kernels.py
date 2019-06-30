import re
from abc import ABC, abstractmethod

import numpy as np


class KernelBase(ABC):
    def __init__(self):
        super().__init__()
        self.parameters = {}
        self.hyperparameters = {}

    def __call__(self, X, Y=None):
        """Refer to documentation for the `_kernel` method"""
        return self._kernel(X, Y)

    @abstractmethod
    def _kernel(self, X, Y):
        raise NotImplementedError

    @abstractmethod
    def __str__(self):
        raise NotImplementedError

    def summary(self):
        return {
            "id": self.hyperparameters["id"],
            "parameters": self.parameters,
            "hyperparameters": self.hyperparameters,
        }

    def set_params(self, summary_dict):
        kr, sd = self, summary_dict

        # collapse `parameters` and `hyperparameters` nested dicts into a single
        # merged dictionary
        flatten_keys = ["parameters", "hyperparameters"]
        for k in flatten_keys:
            if k in sd:
                entry = sd[k]
                sd.update(entry)
                del sd[k]

        for k, v in sd.items():
            if k in self.parameters:
                kr.parameters[k] = v
            if k in self.hyperparameters:
                kr.hyperparameters[k] = v
        return kr


class LinearKernel(KernelBase):
    def __init__(self):
        """
        The linear kernel (i.e., dot-product).

            k(x, y) = x @ y.T
        """
        super().__init__()
        self.hyperparameters = {"id": "LinearKernel"}

    def __str__(self):
        H = self.hyperparameters
        return "{}()".format(H["id"])

    def _kernel(self, X, Y=None):
        """
        Compute the linear kernel (i.e., dot-product) between all pairs of rows in
        X and Y.

        Parameters
        ----------
        X : numpy array of shape (N, C)
            Collection of N input vectors
        Y : numpy array of shape (M, C) (default: None)
            Collection of M input vectors. If `None`, assume Y = X.

        Returns
        -------
        out : numpy array of shape (N, M)
            Similarity between X and Y where index (i,j) gives k(x_i, y_j)
        """
        X, Y = kernel_checks(X, Y)
        return X @ Y.T


class PolynomialKernel(KernelBase):
    def __init__(self, d=3, gamma=None, c0=1):
        """
        The degree-d polynomial kernel.

            k(x, y) = (gamma * x @ y + c0) ** d

        In contrast to the linear kernel, the polynomial kernel also computes
        similarities *across* dimensions of the x and y vectors, allowing it to
        account for interactions between features.

        Parameters
        ----------
        d : int (default: 3)
            Degree of the polynomial kernel
        gamma : float (default: None)
            A scaling parameter for the dot product between x and y. If None,
            defaults to 1 / C. Sometimes referred to as the kernel bandwidth.
        c0 : float (default: 1)
            Parameter trading off the influence of higher-order versus lower-order
            terms in the polynomial. If c0 = 0, the kernel is said to be
            homogenous.
        """
        super().__init__()
        self.hyperparameters = {
            "id": "PolynomialKernel",
            "d": d,
            "c0": c0,
            "gamma": gamma,
        }

    def __str__(self):
        H = self.hyperparameters
        return "{}(d={}, gamma={}, c0={})".format(H["id"], H["d"], H["gamma"], H["c0"])

    def _kernel(self, X, Y=None):
        """
        Compute the degree-d polynomial kernel between all pairs of rows in X
        and Y.

        Parameters
        ----------
        X : numpy array of shape (N, C)
            Collection of N input vectors
        Y : numpy array of shape (M, C) (default: None)
            Collection of M input vectors. If None, assume Y = X.

        Returns
        -------
        out : numpy array of shape (N, M)
            Similarity between X and Y where index (i,j) gives k(x_i, y_j)
        """
        H = self.hyperparameters
        X, Y = kernel_checks(X, Y)
        gamma = 1 / X.shape[1] if H["gamma"] is None else H["gamma"]
        return (gamma * (X @ Y.T) + H["c0"]) ** H["d"]


class RBFKernel(KernelBase):
    def __init__(self, gamma=None):
        """
        Radial basis function (RBF) kernel.

            k(x, y) = exp(-gamma * ||x - y||^2)

        The RBF kernel decreases with distance and ranges between zero (in the
        limit) and one (when x = y).

        Parameters
        ----------
        gamma : float (default: None)
            A scaling parameter for the dot product between x and y. If None,
            defaults to 1 / C. Sometimes referred to as the kernel bandwidth.
        """
        super().__init__()
        self.hyperparameters = {"id": "RBFKernel", "gamma": gamma}

    def __str__(self):
        H = self.hyperparameters
        return "{}(gamma={})".format(H["id"], H["gamma"])

    def _kernel(self, X, Y=None):
        """
        Computes the radial basis function (RBF) kernel between all pairs of rows
        in X and Y.

        Parameters
        ----------
        X : numpy array of shape (N, C)
            Collection of N input vectors
        Y : numpy array of shape (M, C) (default: None)
            Collection of M input vectors. If None, assume Y = X.

        Returns
        -------
        out : numpy array of shape (N, M)
            Similarity between X and Y where index (i, j) gives k(x_i, y_j)
        """
        H = self.hyperparameters
        X, Y = kernel_checks(X, Y)
        gamma = 1 / X.shape[1] if H["gamma"] is None else H["gamma"]
        return np.exp(-gamma * pairwise_l2_distances(X, Y) ** 2)


class KernelInitializer(object):
    def __init__(self, param=None, lr=None):
        """
        A class for initializing learning rate schedulers. Valid inputs are:
            (a) __str__ representations of `KernelBase` instances
            (b) `KernelBase` instances
            (c) Parameter dicts (e.g., as produced via the `summary` method in
                `KernelBase` instances)

        If `param` is `None`, return `LinearKernel`.
        """
        self.param = param

    def __call__(self):
        param = self.param
        if param is None:
            kernel = LinearKernel()
        elif isinstance(param, KernelBase):
            kernel = param
        elif isinstance(param, str):
            kernel = self.init_from_str()
        elif isinstance(param, dict):
            kernel = self.init_from_dict()
        return kernel

    def init_from_str(self):
        r = r"([a-zA-Z0-9]*)=([^,)]*)"
        kr_str = self.param.lower()
        kwargs = dict([(i, eval(j)) for (i, j) in re.findall(r, self.param)])

        if "linear" in kr_str:
            kernel = LinearKernel(**kwargs)
        elif "polynomial" in kr_str:
            kernel = PolynomialKernel(**kwargs)
        elif "rbf" in kr_str:
            kernel = RBFKernel(**kwargs)
        else:
            raise NotImplementedError("{}".format(kr_str))
        return kernel

    def init_from_dict(self):
        S = self.param
        sc = S["hyperparameters"] if "hyperparameters" in S else None

        if sc is None:
            raise ValueError("Must have `hyperparameters` key: {}".format(S))

        if sc and sc["id"] == "LinearKernel":
            scheduler = LinearKernel().set_params(sc)
        elif sc and sc["id"] == "PolynomialKernel":
            scheduler = PolynomialKernel().set_params(sc)
        elif sc and sc["id"] == "RBFKernel":
            scheduler = RBFKernel().set_params(sc)
        elif sc:
            raise NotImplementedError("{}".format(sc["id"]))
        return scheduler


def kernel_checks(X, Y):
    X = X.reshape(-1, 1) if X.ndim == 1 else X
    Y = X if Y is None else Y
    Y = Y.reshape(-1, 1) if Y.ndim == 1 else Y

    assert X.ndim == 2, "X must have 2 dimensions, but got {}".format(X.ndim)
    assert Y.ndim == 2, "Y must have 2 dimensions, but got {}".format(Y.ndim)
    assert X.shape[1] == Y.shape[1], "X and Y must have the same number of columns"
    return X, Y


def pairwise_l2_distances(X, Y):
    """
    A fast, vectorized way to compute pairwise l2 distances between rows in X
    and Y.

        d[i, j] = np.sqrt((x_i - y_i) @ (x_i - y_i))
                = np.sqrt(sum (x_i - y_j)^2)
                = np.sqrt(sum (x_i)^2 - 2 x_i y_j + (y_j)^2)

    The code below computes the the third line using numpy broadcasting
    fanciness to avoid any `for` loops.

    Parameters
    ----------
    X : numpy array of shape (N, C)
        Collection of N input vectors
    Y : numpy array of shape (M, C) (default: None)
        Collection of M input vectors. If `None`, assume Y = X.

    Returns
    -------
    dists : numpy array of shape (N, M)
        Pairwise distance matrix. Entry (i, j) contains the l2 distance between
        x_i and y_j
    """
    return np.sqrt(
        -2 * X @ Y.T + np.sum(Y ** 2, axis=1) + np.sum(X ** 2, axis=1)[:, np.newaxis]
    )
