import numpy as np


def euclidean(x, y):
    """
    Compute the Euclidean (`L2`) distance between two real vectors

    Notes
    -----
    The Euclidean distance between two vectors **x** and **y** is

    .. math::

        d(\mathbf{x}, \mathbf{y}) = \sqrt{ \sum_i (x_i - y_i)^2  }

    Parameters
    ----------
    x,y : :py:class:`ndarray <numpy.ndarray>` s of shape `(N,)`
        The two vectors to compute the distance between

    Returns
    -------
    d : float
        The L2 distance between **x** and **y**.
    """
    return np.sqrt(np.sum((x - y) ** 2))


def manhattan(x, y):
    """
    Compute the Manhattan (`L1`) distance between two real vectors

    Notes
    -----
    The Manhattan distance between two vectors **x** and **y** is

    .. math::

        d(\mathbf{x}, \mathbf{y}) = \sum_i |x_i - y_i|

    Parameters
    ----------
    x,y : :py:class:`ndarray <numpy.ndarray>` s of shape `(N,)`
        The two vectors to compute the distance between

    Returns
    -------
    d : float
        The L1 distance between **x** and **y**.
    """
    return np.sum(np.abs(x - y))


def chebyshev(x, y):
    """
    Compute the Chebyshev (:math:`L_\infty`) distance between two real vectors

    Notes
    -----
    The Chebyshev distance between two vectors **x** and **y** is

    .. math::

        d(\mathbf{x}, \mathbf{y}) = \max_i |x_i - y_i|

    Parameters
    ----------
    x,y : :py:class:`ndarray <numpy.ndarray>` s of shape `(N,)`
        The two vectors to compute the distance between

    Returns
    -------
    d : float
        The Chebyshev distance between **x** and **y**.
    """
    return np.max(np.abs(x - y))


def minkowski(x, y, p):
    """
    Compute the Minkowski-`p` distance between two real vectors.

    Notes
    -----
    The Minkowski-`p` distance between two vectors **x** and **y** is

    .. math::

        d(\mathbf{x}, \mathbf{y}) = \left( \sum_i |x_i - y_i|^p \\right)^{1/p}

    Parameters
    ----------
    x,y : :py:class:`ndarray <numpy.ndarray>` s of shape `(N,)`
        The two vectors to compute the distance between
    p : float > 1
        The parameter of the distance function. When `p = 1`, this is the `L1`
        distance, and when `p=2`, this is the `L2` distance. For `p < 1`,
        Minkowski-`p` does not satisfy the triangle inequality and hence is not
        a valid distance metric.

    Returns
    -------
    d : float
        The Minkowski-`p` distance between **x** and **y**.
    """
    return np.sum(np.abs(x - y) ** p) ** (1 / p)


def hamming(x, y):
    """
    Compute the Hamming distance between two integer-valued vectors.

    Notes
    -----
    The Hamming distance between two vectors **x** and **y** is

    .. math::

        d(\mathbf{x}, \mathbf{y}) = \\frac{1}{N} \sum_i \mathbb{1}_{x_i \\neq y_i}

    Parameters
    ----------
    x,y : :py:class:`ndarray <numpy.ndarray>` s of shape `(N,)`
        The two vectors to compute the distance between. Both vectors should be
        integer-valued.

    Returns
    -------
    d : float
        The Hamming distance between **x** and **y**.
    """
    return np.sum(x != y) / len(x)
