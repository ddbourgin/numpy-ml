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
    def __init__(self):
        """
        A logistic sigmoid activation function.
        """
        super().__init__()

    def __str__(self):
        return "Sigmoid"

    def fn(self, z):
        """
        Compute the logistic sigmoid, :math:`\sigma`, on the elements of input `z`.

        Notes
        ----
        The logistic sigmoid, :math:`\sigma`, is

        .. math::

            \sigma(x_i) = \\frac{1}{1 + e^{-x_i}}

        where :math:`x_i` is the `i` th dimension of input example **x**.
        """
        return 1 / (1 + np.exp(-z))

    def grad(self, x):
        """
        Compute the gradient of the logistic sigmoid on the elements of input
        `x`.

        Notes
        -----
        The first derivative of the logistic sigmoid :math:`\sigma` is

        .. math::

            \\frac{\partial \sigma}{\partial x_i} = \sigma(x_i) (1 - \sigma(x_i))
        """
        fn_x = self.fn(x)
        return fn_x * (1 - fn_x)

    def grad2(self, x):
        """
        Compute the second derivative of the logistic sigmoid on the elements of `x`.

        Notes
        -----
        The second derivative of the logistic sigmoid :math:`\sigma` is

        .. math::

            \\frac{\partial^2 \sigma}{\partial x_i^2} =
                \\frac{\partial \sigma}{\partial x_i} (1 - 2 \sigma(x_i))
        """
        fn_x = self.fn_x
        return fn_x * (1 - fn_x) * (1 - 2 * fn_x)


class ReLU(ActivationBase):
    """
    A rectified linear activation function.

    Notes
    -----
    "ReLU units can be fragile during training and can "die". For example, a
    large gradient flowing through a ReLU neuron could cause the weights to
    update in such a way that the neuron will never activate on any datapoint
    again. If this happens, then the gradient flowing through the unit will
    forever be zero from that point on. That is, the ReLU units can
    irreversibly die during training since they can get knocked off the data
    manifold.

    For example, you may find that as much as 40% of your network can be "dead"
    (i.e. neurons that never activate across the entire training dataset) if
    the learning rate is set too high. With a proper setting of the learning
    rate this is less frequently an issue." [1]_

    References
    ----------
    .. [1] Karpathy, A. "CS231n: Convolutional neural networks for visual recognition".
    """

    def __init__(self):
        super().__init__()

    def __str__(self):
        return "ReLU"

    def fn(self, z):
        """
        Compute the ReLU function on the elements of input `z`.

        Notes
        -----
        The ReLU function is

        .. math::

            \\text{ReLU}(z_i)
                &=  z_i \\ \\ \\ \\ &&\\text{if }z_i > 0 \\\\
                &=  0 \\ \\ \\ \\ &&\\text{otherwise}

        where :math:`z_i` is the `i` th dimension of an example in **z**.
        """
        return np.clip(z, 0, np.inf)

    def grad(self, x):
        """
        Compute the first derivative of the ReLU function on the elements of input `x`.

        Notes
        -----
        The first derivative of the ReLU function is

        .. math::

            \\frac{\partial \\text{ReLU}}{\partial x_i}
                &=  1 \\ \\ \\ \\ &&\\text{if }x_i > 0 \\\\
                &=  0   \\ \\ \\ \\ &&\\text{otherwise}

        where :math:`x_i` is the `i` th dimension of an example in **x**.
        """
        return (x > 0).astype(int)

    def grad2(self, x):
        """
        Compute the second derivative of the ReLU function on the elements of input `x`.

        Notes
        -----
        The second derivative of the ReLU function is

        .. math::

            \\frac{\partial^2 \\text{ReLU}}{\partial x_i^2}  =  0

        where :math:`x_i` is the `i` th dimension of an example in **x**.
        """
        return np.zeros_like(x)


class LeakyReLU(ActivationBase):
    """
    'Leaky' version of a rectified linear unit (ReLU).

    Notes
    -----
    Leaky ReLUs are designed to address the vanishing gradient problem in ReLUs
    by allowing a small non-zero gradient when `x` is negative.

    Parameters
    ----------
    alpha: float
        Activation slope when x < 0. Default is 0.3.

    References
    ----------
    .. [1] Mass, L. M., Hannun, A. Y, & Ng, A. Y. (2013). "Rectifier nonlinearities improve neural network acoustic models". *ICML, 30*.
    """

    def __init__(self, alpha=0.3):
        self.alpha = alpha
        super().__init__()

    def __str__(self):
        return "Leaky ReLU(alpha={})".format(self.alpha)

    def fn(self, z):
        """
        Compute the leaky ReLU function on the elements of input `z`.

        Notes
        -----
        The leaky ReLU function is

        .. math::

            \\text{LeakyReLU}(z_i)
                &=  z_i \\ \\ \\ \\ &&\\text{if } z_i > 0 \\\\
                &=  \\alpha z_i \\ \\ \\ \\ &&\\text{otherwise}

        where :math:`z_i` is the `i` th dimension of an example in input **z**.
        """
        _z = z.copy()
        _z[z < 0] = _z[z < 0] * self.alpha
        return _z

    def grad(self, x):
        """
        Compute the first derivative of the leaky ReLU function on the elements
        of input `x`.

        Notes
        -----
        The first derivative of the leaky ReLU function is

        .. math::

            \\frac{\partial \\text{LeakyReLU}}{\partial x_i}
                &=  1 \\ \\ \\ \\ &&\\text{if }x_i > 0 \\\\
                &=  \\alpha \\ \\ \\ \\ &&\\text{otherwise}

        where :math:`x_i` is the `i` th dimension of an example in **x**.
        """
        out = np.ones_like(x)
        out[x < 0] *= self.alpha
        return out

    def grad2(self, x):
        """
        Compute the second derivative of the leaky ReLU function on the elements of input `x`.

        Notes
        -----
        The second derivative of the leaky ReLU function is

        .. math::

            \\frac{\partial^2 \\text{LeakyReLU}}{\partial x_i^2}  =  0

        where :math:`x_i` is the `i` th dimension of an example in **x**.
        """
        return np.zeros_like(x)


class Tanh(ActivationBase):
    def __init__(self):
        """
        A hyperbolic tangent activation function.
        """
        super().__init__()

    def __str__(self):
        return "Tanh"

    def fn(self, z):
        """
        Compute the tanh function on the elements of input `z`.
        """
        return np.tanh(z)

    def grad(self, x):
        """
        Compute the first derivative of the tanh function on the elements
        of input `x`.

        Notes
        -----
        The first derivative of the tanh function is

        .. math::

            \\frac{\partial \\tanh}{\partial x_i}  =  1 - \\tanh(x)^2

        where :math:`x_i` is the `i` th dimension of an example in **x**.
        """
        return 1 - np.tanh(x) ** 2

    def grad2(self, x):
        """
        Compute the second derivative of the tanh function on the elements
        of input `x`.

        Notes
        -----
        The second derivative of the tanh function is

        .. math::

            \\frac{\partial^2 \\tanh}{\partial x_i^2} = -2 \\tanh(x) \left(\\frac{\partial \\tanh}{\partial x_i}\\right)

        where :math:`x_i` is the `i` th dimension of an example in **x**.
        """
        tanh_x = np.tanh(x)
        return -2 * tanh_x * (1 - tanh_x ** 2)


class Affine(ActivationBase):
    def __init__(self, slope=1, intercept=0):
        """
        An affine activation function.

        Parameters
        ----------
        slope: float
            Activation slope. Default is 1.
        intercept: float
            Intercept/offset term. Default is 0.
        """
        self.slope = slope
        self.intercept = intercept
        super().__init__()

    def __str__(self):
        return "Affine(slope={}, intercept={})".format(self.slope, self.intercept)

    def fn(self, z):
        """
        Compute the Affine activation on the elements of input `z`.

        Notes
        -----
        The Affine activation is

        .. math::

            \\text{Affine}(z_i)  =  \\text{slope} \\times z_i + \\text{intercept}

        where :math:`z_i` is the `i` th dimension of an example in input **z**.
        """
        return self.slope * z + self.intercept

    def grad(self, x):
        """
        Compute the first derivative of the Affine activation on the elements
        of input `x`.

        Notes
        -----
        The first derivative of the Affine activation is

        .. math::

            \\frac{\partial \\text{Affine}}{\partial x_i}  =  \\text{slope}

        where :math:`x_i` is the `i` th dimension of an example in **x**.
        """
        return self.slope * np.ones_like(x)

    def grad2(self, x):
        """
        Compute the second derivative of the Affine activation on the elements
        of input `x`.

        Notes
        -----
        The second derivative of the Affine activation is

        .. math::

            \\frac{\partial^2 \\text{Affine}}{\partial x_i^2}  =  0

        where :math:`x_i` is the `i` th dimension of an example in **x**.
        """
        return np.zeros_like(x)


class Identity(Affine):
    def __init__(self):
        """
        Identity activation function.

        Notes
        -----
        :class:`Identity` is just syntactic sugar for :class:`Affine` with
        slope = 1 and intercept = 0.
        """
        super().__init__(slope=1, intercept=0)

    def __str__(self):
        return "Identity"


class ELU(ActivationBase):
    def __init__(self, alpha=1.0):
        """
        An exponential linear unit (ELU).

        Notes
        -----
        ELUs are intended to address the fact that ReLUs are strictly nonnegative
        and thus have an average activation > 0, increasing the chances of internal
        covariate shift and slowing down learning. ELU units address this by (1)
        allowing negative values when :math:`x < 0`, which (2) are bounded by a value
        :math:`-\\alpha`. Similar to :class:`LeakyReLU`, the negative activation
        values help to push the average unit activation towards 0. Unlike
        :class:`LeakyReLU`, however, the boundedness of the negative activation
        allows for greater robustness in the face of large negative values,
        allowing the function to avoid conveying the *degree* of "absence"
        (negative activation) in the input. [1]

        Parameters
        ----------
        alpha : float
            Slope of negative segment. Default is 1.

        References
        ----------
        .. [1] Clevert, D. A., Unterthiner, T., Hochreiter, S. (2016). "Fast and accurate deep network learning by exponential linear units (ELUs)". *ICLR*
        """

        self.alpha = alpha
        super().__init__()

    def __str__(self):
        return "ELU(alpha={})".format(self.alpha)

    def fn(self, z):
        """
        Compute the ELU activation on the elements of input `z`.

        Notes
        -----
        The exponential linear unit activation function is

        .. math::

            \\text{ELU}(z_i)
                &=  z_i \\ \\ \\ \\ &&\\text{if }z_i > 0 \\\\
                &=  \\alpha (e^{z_i} - 1) \\ \\ \\ \\ &&\\text{otherwise}

        where :math:`z_i` is the `i` th dimension of an example in input **z**.
        """
        # z if z > 0  else alpha * (e^z - 1)
        return np.where(z > 0, z, self.alpha * (np.exp(z) - 1))

    def grad(self, x):
        """
        Compute the first derivative of the ELU activation on the elements
        of input `x`.

        Notes
        -----
        The first derivative of the ELU activation is

        .. math::

            \\frac{\partial \\text{ELU}}{\partial x_i}
                &=  1 \\ \\ \\ \\ &&\\text{if } x_i > 0 \\\\
                &=  \\alpha e^{x_i} \\ \\ \\ \\ &&\\text{otherwise}

        where :math:`x_i` is the `i` th dimension of an example in **x**.
        """
        # 1 if x > 0 else alpha * e^(z)
        return np.where(x > 0, np.ones_like(x), self.alpha * np.exp(x))

    def grad2(self, x):
        """
        Compute the second derivative of the ELU activation on the elements
        of input `x`.

        Notes
        -----
        The second derivative of the ELU activation is

        .. math::

            \\frac{\partial^2 \\text{ELU}}{\partial x_i^2}
                &=  0 \\ \\ \\ \\ &&\\text{if } x_i > 0 \\\\
                &=  \\alpha e^{x_i} \\ \\ \\ \\ &&\\text{otherwise}

        where :math:`x_i` is the `i` th dimension of an example in **x**.
        """
        # 0 if x > 0 else alpha * e^(z)
        return np.where(x >= 0, np.zeros_like(x), self.alpha * np.exp(x))


class Exponential(ActivationBase):
    def __init__(self):
        """
        An exponential (base e) activation function.
        """
        super().__init__()

    def __str__(self):
        return "Exponential"

    def fn(self, z):
        """Compute the activation function :math:`\\text{Exponential}(z_i) = e^{z_i}`."""
        return np.exp(z)

    def grad(self, x):
        """
        Compute the first derivative of the exponential activation on the elements
        of input `x`.

        Notes
        -----
        The first derivative of the exponential activation is

        .. math::

            \\frac{\partial \\text{Exponential}}{\partial x_i}  =  e^{x_i}

        where :math:`x_i` is the `i` th dimension of an example in **x**.
        """
        return np.exp(x)

    def grad2(self, x):
        """
        Compute the second derivative of the exponential activation on the elements
        of input `x`.

        Notes
        -----
        The second derivative of the exponential activation is

        .. math::

            \\frac{\partial^2 \\text{Exponential}}{\partial x_i^2}  =  e^{x_i}

        where :math:`x_i` is the `i` th dimension of an example in **x**.
        """
        return np.exp(x)


class SELU(ActivationBase):
    """
    A scaled exponential linear unit (SELU).

    Notes
    -----
    SELU units, when used in conjunction with proper weight initialization and
    regularization techniques, encourage neuron activations to converge to
    zero-mean and unit variance without explicit use of e.g., batchnorm.

    For SELU units, the :math:`\\alpha` and :math:`\\text{scale}` values are
    constants chosen so that the mean and variance of the inputs are preserved
    between consecutive layers. As such the authors propose weights be
    initialized using Lecun-Normal initialization: :math:`w_{ij} \sim
    \mathcal{N}(0, 1 / \\text{fan_in})`, and to use the dropout variant
    :math:`\\alpha`-dropout during regularization. [1]

    See the reference for more information (especially the appendix ;-) ).

    References
    ----------
    .. [1] Klambauer, G., Unterthiner, T., & Hochreiter, S. (2017). "Self-normalizing neural networks." *Advances in Neural Information Processing Systems, 30.*
    """

    def __init__(self):
        self.alpha = 1.6732632423543772848170429916717
        self.scale = 1.0507009873554804934193349852946
        self.elu = ELU(alpha=self.alpha)
        super().__init__()

    def __str__(self):
        return "SELU"

    def fn(self, z):
        """
        Compute the SELU activation on the elements of input `z`.

        Notes
        -----
        The SELU function is

        .. math::

            \\text{SELU}(z_i)  =  \\text{scale} \\times \\text{ELU}(z_i, \\alpha)

        which is simply

        .. math::

            \\text{SELU}(z_i)
                &= \\text{scale} \\times z_i \\ \\ \\ \\ &&\\text{if }z_i > 0 \\\\
                &= \\text{scale} \\times \\alpha (e^{z_i} - 1) \\ \\ \\ \\ &&\\text{otherwise}

        where :math:`z_i` is the `i` th dimension of an example in input **z**.
        """
        return self.scale * self.elu.fn(z)

    def grad(self, x):
        """
        Compute the first derivative of the SELU activation on the elements
        of input `x`.

        Notes
        -----
        The first derivative of the SELU activation is

        .. math::

            \\frac{\partial \\text{SELU}}{\partial x_i}
                &=  \\text{scale} \\ \\ \\ \\ &&\\text{if } x_i > 0 \\\\
                &=  \\text{scale} \\times \\alpha e^{x_i} \\ \\ \\ \\ &&\\text{otherwise}

        where :math:`x_i` is the `i` th dimension of an example in **x**.
        """
        return np.where(
            x >= 0, np.ones_like(x) * self.scale, np.exp(x) * self.alpha * self.scale
        )

    def grad2(self, x):
        """
        Compute the second derivative of the SELU activation on the elements
        of input `x`.

        Notes
        -----
        The second derivative of the SELU activation is

        .. math::

            \\frac{\partial^2 \\text{SELU}}{\partial x_i^2}
                &=  0 \\ \\ \\ \\ &&\\text{if } x_i > 0 \\\\
                &=  \\text{scale} \\times \\alpha e^{x_i} \\ \\ \\ \\ &&\\text{otherwise}

        where :math:`x_i` is the `i` th dimension of an example in **x**.
        """
        return np.where(x > 0, np.zeros_like(x), np.exp(x) * self.alpha * self.scale)


class HardSigmoid(ActivationBase):
    def __init__(self):
        """
        A "hard" sigmoid activation function.

        Notes
        -----
        The hard sigmoid is a piecewise linear approximation of the logistic
        sigmoid that is computationally more efficient to compute.
        """
        super().__init__()

    def __str__(self):
        return "Hard Sigmoid"

    def fn(self, z):
        """
        Compute the hard sigmoid activation on the elements of input `z`.

        Notes
        -----
        The hard sigmoid activation function is

        .. math::

            \\text{HardSigmoid}(z_i)
                &= 0 \\ \\ \\ \\ &&\\text{if }z_i < -2.5 \\\\
                &= 0.2 z_i + 0.5 \\ \\ \\ \\ &&\\text{if }-2.5 \leq z_i \leq 2.5 \\\\
                &= 1 \\ \\ \\ \\ &&\\text{if }z_i > 2.5

        where :math:`z_i` is the `i` th dimension of an example in input **z**.
        """
        return np.clip((0.2 * z) + 0.5, 0.0, 1.0)

    def grad(self, x):
        """
        Compute the first derivative of the hard sigmoid activation on the elements
        of input `x`.

        Notes
        -----
        The first derivative of the hard sigmoid is

        .. math::

            \\frac{\partial \\text{HardSigmoid}}{\partial x_i}
                &=  0.2 \\ \\ \\ \\ &&\\text{if } -2.5 \leq x_i \leq 2.5\\\\
                &=  0 \\ \\ \\ \\ &&\\text{otherwise}

        where :math:`x_i` is the `i` th dimension of an example in **x**.
        """
        return np.where((x >= -2.5) & (x <= 2.5), 0.2, 0)

    def grad2(self, x):
        """
        Compute the second derivative of the hard sigmoid activation on the elements
        of input `x`.

        Notes
        -----
        The second derivative of the hard sigmoid is

        .. math::

            \\frac{\partial^2 \\text{HardSigmoid}}{\partial x_i^2} =  0

        where :math:`x_i` is the `i` th dimension of an example in **x**.
        """
        return np.zeros_like(x)


class SoftPlus(ActivationBase):
    def __init__(self):
        """
        A softplus activation function.

        Notes
        -----
        In contrast to :class:`ReLU` , the softplus activation is differentiable
        everywhere (including 0). It is, however, less computationally efficient to
        compute.

        The derivative of the softplus activation is the logistic sigmoid.
        """
        super().__init__()

    def __str__(self):
        return "SoftPlus"

    def fn(self, z):
        """
        Compute the softplus activation on the elements of input `z`.

        Notes
        -----
        The softplus activation function is

        .. math::

            \\text{SoftPlus}(z_i) = \log(1 + e^{z_i})

        where :math:`z_i` is the `i` th dimension of an example in input **z**.
        """
        return np.log(np.exp(z) + 1)

    def grad(self, x):
        """
        Compute the first derivative of the softplus activation on the elements
        of input `x`.

        Notes
        -----
        The first derivative of the softplus is

        .. math::

            \\frac{\partial \\text{SoftPlus}}{\partial x_i} = \\frac{e^{x_i}}{1 + e^{x_i}}

        where :math:`x_i` is the `i` th dimension of an example in **x**.
        """
        exp_x = np.exp(x)
        return exp_x / (exp_x + 1)

    def grad2(self, x):
        """
        Compute the second derivative of the softplus activation on the elements
        of input `x`.

        Notes
        -----
        The second derivative of the softplus is

        .. math::

            \\frac{\partial^2 \\text{SoftPlus}}{\partial x_i^2} = \\frac{e^{x_i}}{(1 + e^{x_i})^2}

        where :math:`x_i` is the `i` th dimension of an example in **x**.
        """
        exp_x = np.exp(x)
        return exp_x / ((exp_x + 1) ** 2)
