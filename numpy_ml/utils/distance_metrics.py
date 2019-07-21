import numpy as np


def euclidean(x, y):
    """Compute the Euclidean (L2) distance between two real vectors"""
    return np.sqrt(np.sum((x - y) ** 2))


def manhattan(x, y):
    """Compute the Manhattan (L1) distance between two real vectors"""
    return np.sum(np.abs(x - y))


def chebyshev(x, y):
    """Compute the Chebyshev (L_\infty) distance between two real vectors"""
    return np.max(np.abs(x - y))


def minkowski(x, y, p):
    """Compute the Minkowski-p distance between two real vectors"""
    return np.sum(np.abs(x - y) ** p) ** (1 / p)


def hamming(x, y):
    """Compute the Hamming distance between two integer-valued vectors"""
    return np.sum(x != y) / len(x)
