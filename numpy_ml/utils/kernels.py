import re
from abc import ABC, abstractmethod

import numpy as np


class KernelBase(ABC):
    def __init__(self):
        super().__init__()
        self.parameters = {}
        self.hyperparameters = {}

    @abstractmethod
    def _kernel(self, X, Y):
        raise NotImplementedError

    def __call__(self, X, Y=None):
        """Refer to documentation for the `_kernel` method"""
        return self._kernel(X, Y)

    def __str__(self):
        P, H = self.parameters, self.hyperparameters
        p_str = ", ".join(["{}={}".format(k, v) for k, v in P.items()])
        return "{}({})".format(H["id"], p_str)

    def summary(self):
        """Return the dictionary of model parameters, hyperparameters, and ID"""
        return {
            "id": self.hyperparameters["id"],
            "parameters": self.parameters,
            "hyperparameters": self.hyperparameters,
        }

    def set_params(self, summary_dict):
        """
        Set the model parameters and hyperparameters using the settings in
        `summary_dict`.

        Parameters
        ----------
        summary_dict : dict
            A dictionary with keys 'parameters' and 'hyperparameters',
            structured as would be returned by the :meth:`summary` method. If
            a particular (hyper)parameter is not included in this dict, the
            current value will be used.

        Returns
        -------
        new_kernel : :doc:`Kernel <numpy_ml.utils.kernels>` instance
            A kernel with parameters and hyperparameters adjusted to those
            specified in `summary_dict`.
        """
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
    def __init__(self, c0=0):
        """
        The linear (i.e., dot-product) kernel.

        Notes
        -----
        For input vectors :math:`\mathbf{x}` and :math:`\mathbf{y}`, the linear
        kernel is:

        .. math::

            k(\mathbf{x}, \mathbf{y}) = \mathbf{x}^\\top \mathbf{y} + c_0

        Parameters
        ----------
        c0 : float
            An "inhomogeneity" parameter. When `c0` = 0, the kernel is said to be
            homogenous. Default is 1.
        """
        super().__init__()
        self.hyperparameters = {"id": "LinearKernel"}
        self.parameters = {"c0": c0}

    def _kernel(self, X, Y=None):
        """
        Compute the linear kernel (i.e., dot-product) between all pairs of rows in
        `X` and `Y`.

        Parameters
        ----------
        X : :py:class:`ndarray <numpy.ndarray>` of shape `(N, C)`
            Collection of `N` input vectors
        Y : :py:class:`ndarray <numpy.ndarray>` of shape `(M, C)` or None
            Collection of `M` input vectors. If None, assume `Y` = `X`.
            Default is None.

        Returns
        -------
        out : :py:class:`ndarray <numpy.ndarray>` of shape `(N, M)`
            Similarity between `X` and `Y`, where index (`i`, `j`) gives
            :math:`k(x_i, y_j)`.
        """
        X, Y = kernel_checks(X, Y)
        return X @ Y.T + self.parameters["c0"]


class PolynomialKernel(KernelBase):
    def __init__(self, d=3, gamma=None, c0=1):
        """
        The degree-`d` polynomial kernel.

        Notes
        -----
        For input vectors :math:`\mathbf{x}` and :math:`\mathbf{y}`, the polynomial
        kernel is:

        .. math::

            k(\mathbf{x}, \mathbf{y}) = (\gamma \mathbf{x}^\\top \mathbf{y} + c_0)^d

        In contrast to the linear kernel, the polynomial kernel also computes
        similarities *across* dimensions of the **x** and **y** vectors,
        allowing it to account for interactions between features.  As an
        instance of the dot product family of kernels, the polynomial kernel is
        invariant to a rotation of the coordinates about the origin, but *not*
        to translations.

        Parameters
        ----------
        d : int
            Degree of the polynomial kernel. Default is 3.
        gamma : float or None
            A scaling parameter for the dot product between `x` and `y`,
            determining the amount of smoothing/resonlution of the kernel.
            Larger values result in greater smoothing. If None, defaults to 1 /
            `C`.  Sometimes referred to as the kernel bandwidth.  Default is
            None.
        c0 : float
            Parameter trading off the influence of higher-order versus lower-order
            terms in the polynomial. If `c0` = 0, the kernel is said to be
            homogenous. Default is 1.
        """
        super().__init__()
        self.hyperparameters = {"id": "PolynomialKernel"}
        self.parameters = {"d": d, "c0": c0, "gamma": gamma}

    def _kernel(self, X, Y=None):
        """
        Compute the degree-`d` polynomial kernel between all pairs of rows in `X`
        and `Y`.

        Parameters
        ----------
        X : :py:class:`ndarray <numpy.ndarray>` of shape `(N, C)`
            Collection of `N` input vectors
        Y : :py:class:`ndarray <numpy.ndarray>` of shape `(M, C)` or None
            Collection of `M` input vectors. If None, assume `Y = X`. Default
            is None.

        Returns
        -------
        out : :py:class:`ndarray <numpy.ndarray>` of shape `(N, M)`
            Similarity between `X` and `Y` where index (`i`, `j`) gives
            :math:`k(x_i, y_j)` (i.e., the kernel's Gram-matrix).
        """
        P = self.parameters
        X, Y = kernel_checks(X, Y)
        gamma = 1 / X.shape[1] if P["gamma"] is None else P["gamma"]
        return (gamma * (X @ Y.T) + P["c0"]) ** P["d"]


class RBFKernel(KernelBase):
    def __init__(self, sigma=None):
        """
        Radial basis function (RBF) / squared exponential kernel.

        Notes
        -----
        For input vectors :math:`\mathbf{x}` and :math:`\mathbf{y}`, the radial
        basis function kernel is:

        .. math::

            k(\mathbf{x}, \mathbf{y}) = \exp \left\{ -0.5
                \left\lVert \\frac{\mathbf{x} -
                    \mathbf{y}}{\sigma} \\right\\rVert_2^2 \\right\}

        The RBF kernel decreases with distance and ranges between zero (in the
        limit) to one (when **x** = **y**). Notably, the implied feature space
        of the kernel has an infinite number of dimensions.

        Parameters
        ----------
        sigma : float or array of shape `(C,)` or None
            A scaling parameter for the vectors **x** and **y**, producing an
            isotropic kernel if a float, or an anistropic kernel if an array of
            length `C`.  Larger values result in higher resolution / greater
            smoothing. If None, defaults to :math:`\sqrt(C / 2)`. Sometimes
            referred to as the kernel 'bandwidth'. Default is None.
        """
        super().__init__()
        self.hyperparameters = {"id": "RBFKernel"}
        self.parameters = {"sigma": sigma}

    def _kernel(self, X, Y=None):
        """
        Computes the radial basis function (RBF) kernel between all pairs of
        rows in `X` and `Y`.

        Parameters
        ----------
        X : :py:class:`ndarray <numpy.ndarray>` of shape `(N, C)`
            Collection of `N` input vectors, each with dimension `C`.
        Y : :py:class:`ndarray <numpy.ndarray>` of shape `(M, C)`
            Collection of `M` input vectors. If None, assume `Y` = `X`. Default
            is None.

        Returns
        -------
        out : :py:class:`ndarray <numpy.ndarray>` of shape `(N, M)`
            Similarity between `X` and `Y` where index (i, j) gives :math:`k(x_i, y_j)`.
        """
        P = self.parameters
        X, Y = kernel_checks(X, Y)
        sigma = np.sqrt(X.shape[1] / 2) if P["sigma"] is None else P["sigma"]
        return np.exp(-0.5 * pairwise_l2_distances(X / sigma, Y / sigma) ** 2)


class KernelInitializer(object):
    def __init__(self, param=None):
        """
        A class for initializing learning rate schedulers. Valid inputs are:
            (a) __str__ representations of `KernelBase` instances
            (b) `KernelBase` instances
            (c) Parameter dicts (e.g., as produced via the :meth:`summary` method in
                `KernelBase` instances)

        If `param` is None, return `LinearKernel`.
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
            scheduler = LinearKernel().set_params(S)
        elif sc and sc["id"] == "PolynomialKernel":
            scheduler = PolynomialKernel().set_params(S)
        elif sc and sc["id"] == "RBFKernel":
            scheduler = RBFKernel().set_params(S)
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
    A fast, vectorized way to compute pairwise l2 distances between rows in `X`
    and `Y`.

    Notes
    -----
    An entry of the pairwise Euclidean distance matrix for two vectors is

    .. math::

        d[i, j]  &=  \sqrt{(x_i - y_i) @ (x_i - y_i)} \\\\
                 &=  \sqrt{sum (x_i - y_j)^2} \\\\
                 &=  \sqrt{sum (x_i)^2 - 2 x_i y_j + (y_j)^2}

    The code below computes the the third line using numpy broadcasting
    fanciness to avoid any for loops.

    Parameters
    ----------
    X : :py:class:`ndarray <numpy.ndarray>` of shape `(N, C)`
        Collection of `N` input vectors
    Y : :py:class:`ndarray <numpy.ndarray>` of shape `(M, C)`
        Collection of `M` input vectors. If None, assume `Y` = `X`. Default is
        None.

    Returns
    -------
    dists : :py:class:`ndarray <numpy.ndarray>` of shape `(N, M)`
        Pairwise distance matrix. Entry (i, j) contains the `L2` distance between
        :math:`x_i` and :math:`y_j`.
    """
    D = -2 * X @ Y.T + np.sum(Y ** 2, axis=1) + np.sum(X ** 2, axis=1)[:, np.newaxis]
    D[D < 0] = 0  # clip any value less than 0 (a result of numerical imprecision)
    return np.sqrt(D)
