from abc import ABC, abstractmethod

import numpy as np


class ActivationBase(ABC):
    def __init__(self, **kwargs):
        super().__init__()

    def __call__(self, z):
        if z.ndim == 1:
            z = z.reshape(1, -1)
        return self.fn(z)

    @abstractmethod
    def fn(self, z):
        raise NotImplementedError

    @abstractmethod
    def grad(self, x, **kwargs):
        raise NotImplementedError


class Sigmoid(ActivationBase):
    """
    A logistic sigmoid activation function.
    """

    def __init__(self):
        super().__init__()

    def __str__(self):
        return "Sigmoid"

    def fn(self, z):
        return 1 / (1 + np.exp(-z))

    def grad(self, x):
        fn_x = self.fn(x)
        return fn_x * (1 - fn_x)

    def grad2(self, x):
        fn_x = self.fn_x
        return fn_x * (1 - fn_x) * (1 - 2 * fn_x)


class ReLU(ActivationBase):
    """
    A rectified linear activation function.

    ReLU(x) =
        x   if x > 0
        0   otherwise

    ReLU units can be fragile during training and can "die". For example, a
    large gradient flowing through a ReLU neuron could cause the weights to
    update in such a way that the neuron will never activate on any datapoint
    again. If this happens, then the gradient flowing through the unit will
    forever be zero from that point on. That is, the ReLU units can
    irreversibly die during training since they can get knocked off the data
    manifold.

    For example, you may find that as much as 40% of your network can be "dead"
    (i.e. neurons that never activate across the entire training dataset) if
    the learning rate is set too high. With a proper setting of the learning
    rate this is less frequently an issue.

    - Andrej Karpathy
    """

    def __init__(self):
        super().__init__()

    def __str__(self):
        return "ReLU"

    def fn(self, z):
        return np.clip(z, 0, np.inf)

    def grad(self, x):
        return (x > 0).astype(int)

    def grad2(self, x):
        return np.zeros_like(x)


class LeakyReLU(ActivationBase):
    """
    'Leaky' version of a rectified linear unit (ReLU).

    f(x) =
        alpha * x   if x < 0
        x           otherwise

    Leaky ReLUs are designed to address the vanishing gradient problem in ReLUs
    by allowing a small non-zero gradient when x is negative.

    Parameters
    ----------
    alpha: float (default: 0.3)
        Activation slope when x < 0

    References
    ----------
    - [Rectifier Nonlinearities Improve Neural Network Acoustic Models](
    https://ai.stanford.edu/~amaas/papers/relu_hybrid_icml2013_final.pdf)
    """

    def __init__(self, alpha=0.3):
        self.alpha = alpha
        super().__init__()

    def __str__(self):
        return "Leaky ReLU(alpha={})".format(self.alpha)

    def fn(self, z):
        _z = z.copy()
        _z[z < 0] = _z[z < 0] * self.alpha
        return _z

    def grad(self, x):
        out = np.ones_like(x)
        out[x < 0] *= self.alpha
        return out

    def grad2(self, x):
        return np.zeros_like(x)


class Tanh(ActivationBase):
    """
    A hyperbolic tangent activation function.
    """

    def __init__(self):
        super().__init__()

    def __str__(self):
        return "Tanh"

    def fn(self, z):
        return np.tanh(z)

    def grad(self, x):
        return 1 - np.tanh(x) ** 2

    def grad2(self, x):
        tanh_x = np.tanh(x)
        return -2 * tanh_x * (1 - tanh_x ** 2)


class Affine(ActivationBase):
    """
    An affine activation function.

    Parameters
    ----------
    slope: float (default: 1)
        Activation slope
    intercept: float (default: 0)
        Intercept/offset term
    """

    def __init__(self, slope=1, intercept=0):
        self.slope = slope
        self.intercept = intercept
        super().__init__()

    def __str__(self):
        return "Affine(slope={}, intercept={})".format(self.slope, self.intercept)

    def fn(self, z):
        return self.slope * z + self.intercept

    def grad(self, x):
        return self.slope * np.ones_like(x)

    def grad2(self, x):
        return np.zeros_like(x)


class Identity(Affine):
    """
    Identity activation function
    """

    def __init__(self):
        super().__init__(slope=1, intercept=0)

    def __str__(self):
        return "Identity"


class ELU(ActivationBase):
    """
    Exponential linear unit.

        ELU(x) =
            x                   if x >= 0
            alpha * (e^x - 1)   otherwise

    ELUs are intended to address the fact that ReLUs are strictly nonnegative
    and thus have an average activation > 0, increasing the chances of internal
    covariate shift and slowing down learning. ELU units address this by (1)
    allowing negative values when x < 0, which (2) are bounded by a value -1 *
    `alpha`. Similar to leaky / parametric ReLUs, the negative activation
    values help to push the average unit activation towards 0. Unlike leaky /
    parametric ReLUs, however, the boundedness of the negative activation
    allows for greater robustness in the face of large negative values,
    allowing the function to avoid conveying the *degree* of "absence"
    (negative activation) in the input.

    Parameters
    ----------
    alpha : float (default: 1)
        Slope of negative segment

    References
    ----------
    - [Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)](https://arxiv.org/abs/1511.07289)
    """

    def __init__(self, alpha=1.0):
        self.alpha = alpha
        super().__init__()

    def __str__(self):
        return "ELU(alpha={})".format(self.alpha)

    def fn(self, z):
        # z if z > 0  else alpha * (e^z - 1) """
        return np.where(z > 0, z, self.alpha * (np.exp(z) - 1))

    def grad(self, x):
        # 1 if x >= 0 else alpha * e^(z)
        return np.where(x >= 0, np.ones_like(x), self.alpha * np.exp(x))

    def grad2(self, x):
        # 0 if x >= 0 else alpha * e^(z)
        return np.where(x >= 0, np.zeros_like(x), self.alpha * np.exp(x))


class Exponential(ActivationBase):
    """
    Exponential (base e) activation function.
    """

    def __init__(self):
        super().__init__()

    def __str__(self):
        return "Exponential"

    def fn(self, z):
        return np.exp(z)

    def grad(self, x):
        return np.exp(x)

    def grad2(self, x):
        return np.exp(x)


class SELU(ActivationBase):
    """
    Scaled exponential linear unit (SELU).

        SELU(x) = scale * ELU(x, alpha)
                = scale * x                     if x >= 0
                = scale * [alpha * (e^x - 1)]   otherwise

    SELU units, when used in conjunction with proper weight initialization and
    regularization techniques, encourage neuron activations to converge to
    zero-mean and unit variance without explicit use of e.g., batchnorm.

    For SELU units, the `alpha` and `scale` values are constants chosen so that
    the mean and variance of the inputs are preserved between consecutive
    layers. As such the authors propose weights be initialized using
    Lecun-Normal initialization: w ~ N(0, 1 / fan_in), and to use the dropout
    variant `alpha-dropout` during regularization. See the reference for more
    information (especially the appendix ;-) )

    References
    ----------
    - [Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515)
    """

    def __init__(self):
        self.alpha = 1.6732632423543772848170429916717
        self.scale = 1.0507009873554804934193349852946
        self.elu = ELU(alpha=self.alpha)
        super().__init__()

    def __str__(self):
        return "SELU"

    def fn(self, z):
        return self.scale * self.elu.fn(z)

    def grad(self, x):
        return np.where(
            x >= 0, np.ones_like(x) * self.scale, np.exp(x) * self.alpha * self.scale
        )

    def grad2(self, x):
        return np.where(x >= 0, np.zeros_like(x), np.exp(x) * self.alpha * self.scale)


class HardSigmoid(ActivationBase):
    """
    A "hard" sigmoid activation function.

        HardSigmoid(x) =
            0               if x < -2.5
            0.2 * x + 0.5   if -2.5 <= x <= 2.5.
            1               if x > 2.5

    The hard sigmoid is a piecewise linear approximation of the logistic
    sigmoid that is computationally more efficient to compute.
    """

    def __init__(self):
        super().__init__()

    def __str__(self):
        return "Hard Sigmoid"

    def fn(self, z):
        return np.clip((0.2 * z) + 0.5, 0.0, 1.0)

    def grad(self, x):
        return np.where((x >= -2.5) & (x <= 2.5), 0.2, 0)

    def grad2(self, x):
        return np.zeros_like(x)


class SoftPlus(ActivationBase):
    """
    A softplus activation function.

        SoftPlus(x) = log(1 + e^x)

    In contrast to the ReLU function, softplus is differentiable everywhere
    (including 0). It is, however, less computationally efficient to compute.
    The derivative of the softplus activation is the logistic sigmoid.
    """

    def __init__(self):
        super().__init__()

    def __str__(self):
        return "SoftPlus"

    def fn(self, z):
        return np.log(np.exp(z) + 1)

    def grad(self, x):
        return np.exp(x) / (np.exp(x) + 1)

    def grad2(self, x):
        return np.exp(x) / ((np.exp(x) + 1) ** 2)
