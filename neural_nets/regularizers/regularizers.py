"""
Built-in regularizers.
"""

import numpy as np


class Regularizer(object):
    """
    Regularizer base class.
    """

    def __call__(self, x):
        return 0.

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class L1L2(Regularizer):
    """
    Regularizer for L1 and L2 regularization.

    Arguments
    --------
        l1: Float; L1 regularization factor.
        l2: Float; L2 regularization factor.
    """

    def __init__(self, l1=0., l2=0.):
        self.l1 = l1
        self.l2 = l2

    def __call__(self, x):
        regularization = 0.
        if self.l1:
            regularization += np.sum(self.l1 * np.abs(x))
        if self.l2:
            regularization += np.sum(self.l2 * np.square(x))
        return regularization

    def get_config(self):
        return {'l1': float(self.l1),
                'l2': float(self.l2)}


def l1(l=0.01):
    return L1L2(l1=l)


def l2(l=0.01):
    return L1L2(l2=l)


def l1_l2(l1=0.01, l2=0.01):
    return L1L2(l1=l1, l2=l2)
