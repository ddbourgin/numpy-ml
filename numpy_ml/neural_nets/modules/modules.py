from abc import ABC, abstractmethod

import re
import numpy as np

from ..wrappers import Dropout
from ..utils import calc_pad_dims_2D
from ..activations import Tanh, Sigmoid, ReLU, LeakyReLU, Affine
from ..layers import (
    DotProductAttention,
    FullyConnected,
    BatchNorm2D,
    Conv1D,
    Conv2D,
    Multiply,
    LSTMCell,
    Add,
)


class ModuleBase(ABC):
    def __init__(self):
        self.X = None
        self.trainable = True

        super().__init__()

    @abstractmethod
    def _init_params(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def forward(self, z, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def backward(self, out, **kwargs):
        raise NotImplementedError

    @property
    def components(self):
        comps = []
        for c in self.hyperparameters["component_ids"]:
            if hasattr(self, c):
                comps.append(getattr(self, c))
        return comps

    def freeze(self):
        self.trainable = False
        for c in self.components:
            c.freeze()

    def unfreeze(self):
        self.trainable = True
        for c in self.components:
            c.unfreeze()

    def update(self, cur_loss=None):
        assert self.trainable, "Layer is frozen"
        for c in self.components:
            c.update(cur_loss)
        self.flush_gradients()

    def flush_gradients(self):
        assert self.trainable, "Layer is frozen"

        self.X = []
        self._dv = {}
        for c in self.components:
            for k, v in c.derived_variables.items():
                c.derived_variables[k] = None

            for k, v in c.gradients.items():
                c.gradients[k] = np.zeros_like(v)

    def set_params(self, summary_dict):
        cids = self.hyperparameters["component_ids"]
        for k, v in summary_dict["parameters"].items():
            if k == "components":
                for c, cd in summary_dict["parameters"][k].items():
                    if c in cids:
                        getattr(self, c).set_params(cd)

            elif k in self.parameters:
                self.parameters[k] = v

        for k, v in summary_dict["hyperparameters"].items():
            if k == "components":
                for c, cd in summary_dict["hyperparameters"][k].items():
                    if c in cids:
                        getattr(self, c).set_params(cd)

            if k in self.hyperparameters:
                if k == "act_fn" and v == "ReLU":
                    self.hyperparameters[k] = ReLU()
                elif v == "act_fn" and v == "Sigmoid":
                    self.hyperparameters[k] = Sigmoid()
                elif v == "act_fn" and v == "Tanh":
                    self.hyperparameters[k] = Tanh()
                elif v == "act_fn" and "Affine" in v:
                    r = r"Affine\(slope=(.*), intercept=(.*)\)"
                    slope, intercept = re.match(r, v).groups()
                    self.hyperparameters[k] = Affine(float(slope), float(intercept))
                elif v == "act_fn" and "Leaky ReLU" in v:
                    r = r"Leaky ReLU\(alpha=(.*)\)"
                    alpha = re.match(r, v).groups()[0]
                    self.hyperparameters[k] = LeakyReLU(float(alpha))
                else:
                    self.hyperparameters[k] = v

    def summary(self):
        return {
            "parameters": self.parameters,
            "layer": self.hyperparameters["layer"],
            "hyperparameters": self.hyperparameters,
        }


class WavenetResidualModule(ModuleBase):
    def __init__(
        self,
        ch_residual,
        ch_dilation,
        dilation,
        kernel_width,
        optimizer=None,
        init="glorot_uniform",
    ):
        """
        A WaveNet-like residual block with causal dilated convolutions.

        .. code-block:: text

            *Skip path in* >-------------------------------------------> + ---> *Skip path out*
                              Causal      |--> Tanh --|                  |
            *Main    |--> Dilated Conv1D -|           * --> 1x1 Conv1D --|
             path >--|                    |--> Sigm --|                  |
             in*     |-------------------------------------------------> + ---> *Main path out*
                                         *Residual path*

        On the final block, the output of the skip path is further processed to
        produce the network predictions.

        References
        ----------
        .. [1] van den Oord et al. (2016). "Wavenet: a generative model for raw
           audio". https://arxiv.org/pdf/1609.03499.pdf

        Parameters
        ----------
        ch_residual : int
            The number of output channels for the 1x1
            :class:`~numpy_ml.neural_nets.layers.Conv1D` layer in the main path.
        ch_dilation : int
            The number of output channels for the causal dilated
            :class:`~numpy_ml.neural_nets.layers.Conv1D` layer in the main path.
        dilation : int
            The dilation rate for the causal dilated
            :class:`~numpy_ml.neural_nets.layers.Conv1D` layer in the main path.
        kernel_width : int
            The width of the causal dilated
            :class:`~numpy_ml.neural_nets.layers.Conv1D` kernel in the main
            path.
        init : {'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform'}
            The weight initialization strategy. Default is 'glorot_uniform'.
        optimizer : str or :doc:`Optimizer <numpy_ml.neural_nets.optimizers>` object or None
            The optimization strategy to use when performing gradient updates
            within the :meth:`update` method.  If None, use the
            :class:`~numpy_ml.neural_nets.optimizers.SGD` optimizer with default
            parameters. Default is None.
        """
        super().__init__()

        self.init = init
        self.dilation = dilation
        self.optimizer = optimizer
        self.ch_residual = ch_residual
        self.ch_dilation = ch_dilation
        self.kernel_width = kernel_width

        self._init_params()

    def _init_params(self):
        self._dv = {}

        self.conv_dilation = Conv1D(
            stride=1,
            pad="causal",
            init=self.init,
            kernel_width=2,
            dilation=self.dilation,
            out_ch=self.ch_dilation,
            optimizer=self.optimizer,
            act_fn=Affine(slope=1, intercept=0),
        )

        self.tanh = Tanh()
        self.sigm = Sigmoid()
        self.multiply_gate = Multiply(act_fn=Affine(slope=1, intercept=0))

        self.conv_1x1 = Conv1D(
            stride=1,
            pad="same",
            dilation=0,
            init=self.init,
            kernel_width=1,
            out_ch=self.ch_residual,
            optimizer=self.optimizer,
            act_fn=Affine(slope=1, intercept=0),
        )

        self.add_residual = Add(act_fn=Affine(slope=1, intercept=0))
        self.add_skip = Add(act_fn=Affine(slope=1, intercept=0))

    @property
    def parameters(self):
        """A dictionary of the module parameters."""
        return {
            "components": {
                "conv_1x1": self.conv_1x1.parameters,
                "add_skip": self.add_skip.parameters,
                "add_residual": self.add_residual.parameters,
                "conv_dilation": self.conv_dilation.parameters,
                "multiply_gate": self.multiply_gate.parameters,
            }
        }

    @property
    def hyperparameters(self):
        """A dictionary of the module hyperparameters"""
        return {
            "layer": "WavenetResidualModule",
            "init": self.init,
            "dilation": self.dilation,
            "optimizer": self.optimizer,
            "ch_residual": self.ch_residual,
            "ch_dilation": self.ch_dilation,
            "kernel_width": self.kernel_width,
            "component_ids": [
                "conv_1x1",
                "add_skip",
                "add_residual",
                "conv_dilation",
                "multiply_gate",
            ],
            "components": {
                "conv_1x1": self.conv_1x1.hyperparameters,
                "add_skip": self.add_skip.hyperparameters,
                "add_residual": self.add_residual.hyperparameters,
                "conv_dilation": self.conv_dilation.hyperparameters,
                "multiply_gate": self.multiply_gate.hyperparameters,
            },
        }

    @property
    def derived_variables(self):
        """A dictionary of intermediate values computed during the
        forward/backward passes."""
        dv = {
            "conv_1x1_out": None,
            "conv_dilation_out": None,
            "multiply_gate_out": None,
            "components": {
                "conv_1x1": self.conv_1x1.derived_variables,
                "add_skip": self.add_skip.derived_variables,
                "add_residual": self.add_residual.derived_variables,
                "conv_dilation": self.conv_dilation.derived_variables,
                "multiply_gate": self.multiply_gate.derived_variables,
            },
        }
        dv.update(self._dv)
        return dv

    @property
    def gradients(self):
        """A dictionary of the module parameter gradients."""
        return {
            "components": {
                "conv_1x1": self.conv_1x1.gradients,
                "add_skip": self.add_skip.gradients,
                "add_residual": self.add_residual.gradients,
                "conv_dilation": self.conv_dilation.gradients,
                "multiply_gate": self.multiply_gate.gradients,
            }
        }

    def forward(self, X_main, X_skip=None):
        """
        Compute the module output on a single minibatch.

        Parameters
        ----------
        X_main : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, in_rows, in_cols, in_ch)`
            The input volume consisting of `n_ex` examples, each with dimension
            (`in_rows`, `in_cols`, `in_ch`).
        X_skip : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, in_rows, in_cols, in_ch)`, or None
            The output of the preceding skip-connection if this is not the
            first module in the network.

        Returns
        -------
        Y_main : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, out_rows, out_cols, out_ch)`
            The output of the main pathway.
        Y_skip : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, out_rows, out_cols, out_ch)`
            The output of the skip-connection pathway.
        """
        self.X_main, self.X_skip = X_main, X_skip
        conv_dilation_out = self.conv_dilation.forward(X_main)

        tanh_gate = self.tanh.fn(conv_dilation_out)
        sigm_gate = self.sigm.fn(conv_dilation_out)

        multiply_gate_out = self.multiply_gate.forward([tanh_gate, sigm_gate])
        conv_1x1_out = self.conv_1x1.forward(multiply_gate_out)

        # if this is the first wavenet block, initialize the "previous" skip
        # connection sum to 0
        self.X_skip = np.zeros_like(conv_1x1_out) if X_skip is None else X_skip

        Y_skip = self.add_skip.forward([X_skip, conv_1x1_out])
        Y_main = self.add_residual.forward([X_main, conv_1x1_out])

        self._dv["tanh_out"] = tanh_gate
        self._dv["sigm_out"] = sigm_gate
        self._dv["conv_dilation_out"] = conv_dilation_out
        self._dv["multiply_gate_out"] = multiply_gate_out
        self._dv["conv_1x1_out"] = conv_1x1_out
        return Y_main, Y_skip

    def backward(self, dY_skip, dY_main=None):
        dX_skip, dConv_1x1_out = self.add_skip.backward(dY_skip)

        # if this is the last wavenet block, dY_main will be None. if not,
        # calculate the error contribution from dY_main and add it to the
        # contribution from the skip path
        dX_main = np.zeros_like(self.X_main)
        if dY_main is not None:
            dX_main, dConv_1x1_main = self.add_residual.backward(dY_main)
            dConv_1x1_out += dConv_1x1_main

        dMultiply_out = self.conv_1x1.backward(dConv_1x1_out)
        dTanh_out, dSigm_out = self.multiply_gate.backward(dMultiply_out)

        conv_dilation_out = self.derived_variables["conv_dilation_out"]
        dTanh_in = dTanh_out * self.tanh.grad(conv_dilation_out)
        dSigm_in = dSigm_out * self.sigm.grad(conv_dilation_out)
        dDilation_out = dTanh_in + dSigm_in

        conv_back = self.conv_dilation.backward(dDilation_out)
        dX_main += conv_back

        self._dv["dLdTanh"] = dTanh_out
        self._dv["dLdSigmoid"] = dSigm_out
        self._dv["dLdConv_1x1"] = dConv_1x1_out
        self._dv["dLdMultiply"] = dMultiply_out
        self._dv["dLdConv_dilation"] = dDilation_out
        return dX_main, dX_skip


class SkipConnectionIdentityModule(ModuleBase):
    def __init__(
        self,
        out_ch,
        kernel_shape1,
        kernel_shape2,
        stride1=1,
        stride2=1,
        act_fn=None,
        epsilon=1e-5,
        momentum=0.9,
        optimizer=None,
        init="glorot_uniform",
    ):
        """
        A ResNet-like "identity" shortcut module.

        Notes
        -----
        The identity module enforces `same` padding during each convolution to
        ensure module output has same dims as its input.

        .. code-block:: text

            X -> Conv2D -> Act_fn -> BatchNorm2D -> Conv2D -> BatchNorm2D -> + -> Act_fn
             \______________________________________________________________/

        References
        ----------
        .. [1] He et al. (2015). "Deep residual learning for image
           recognition." https://arxiv.org/pdf/1512.03385.pdf

        Parameters
        ----------
        out_ch : int
            The number of filters/kernels to compute in the first convolutional
            layer.
        kernel_shape1 : 2-tuple
            The dimension of a single 2D filter/kernel in the first
            convolutional layer.
        kernel_shape2 : 2-tuple
            The dimension of a single 2D filter/kernel in the second
            convolutional layer.
        stride1 : int
            The stride/hop of the convolution kernels in the first
            convolutional layer. Default is 1.
        stride2 : int
            The stride/hop of the convolution kernels in the second
            convolutional layer. Default is 1.
        act_fn : :doc:`Activation <numpy_ml.neural_nets.activations>` object or None
            The activation function for computing Y[t]. If None, use the
            identity :math:`f(x) = x` by default. Default is None.
        epsilon : float
            A small smoothing constant to use during
            :class:`~numpy_ml.neural_nets.layers.BatchNorm2D` computation to
            avoid divide-by-zero errors. Default is 1e-5.
        momentum : float
            The momentum term for the running mean/running std calculations in
            the :class:`~numpy_ml.neural_nets.layers.BatchNorm2D` layers.  The
            closer this is to 1, the less weight will be given to the mean/std
            of the current batch (i.e., higher smoothing). Default is 0.9.
        optimizer : str or :doc:`Optimizer <numpy_ml.neural_nets.optimizers>` object or None
            The optimization strategy to use when performing gradient updates
            within the :meth:`update` method.  If None, use the
            :class:`~numpy_ml.neural_nets.optimizers.SGD` optimizer with
            default parameters. Default is None.
        init : {'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform'}
            The weight initialization strategy. Default is 'glorot_uniform'.
        """
        super().__init__()

        self.init = init
        self.in_ch = None
        self.out_ch = out_ch
        self.epsilon = epsilon
        self.stride1 = stride1
        self.stride2 = stride2
        self.optimizer = optimizer
        self.momentum = momentum
        self.kernel_shape1 = kernel_shape1
        self.kernel_shape2 = kernel_shape2
        self.act_fn = Affine(slope=1, intercept=0) if act_fn is None else act_fn

        self._init_params()

    def _init_params(self):
        self._dv = {}

        self.conv1 = Conv2D(
            pad="same",
            init=self.init,
            out_ch=self.out_ch,
            act_fn=self.act_fn,
            stride=self.stride1,
            optimizer=self.optimizer,
            kernel_shape=self.kernel_shape1,
        )
        # we can't initialize `conv2` without X's dimensions; see `forward`
        # for further details
        self.batchnorm1 = BatchNorm2D(epsilon=self.epsilon, momentum=self.momentum)
        self.batchnorm2 = BatchNorm2D(epsilon=self.epsilon, momentum=self.momentum)
        self.add3 = Add(self.act_fn)

    def _init_conv2(self):
        self.conv2 = Conv2D(
            pad="same",
            init=self.init,
            out_ch=self.in_ch,
            stride=self.stride2,
            optimizer=self.optimizer,
            kernel_shape=self.kernel_shape2,
            act_fn=Affine(slope=1, intercept=0),
        )

    @property
    def parameters(self):
        """A dictionary of the module parameters."""
        return {
            "components": {
                "add3": self.add3.parameters,
                "conv1": self.conv1.parameters,
                "conv2": self.conv2.parameters,
                "batchnorm1": self.batchnorm1.parameters,
                "batchnorm2": self.batchnorm2.parameters,
            }
        }

    @property
    def hyperparameters(self):
        """A dictionary of the module hyperparameters."""
        return {
            "layer": "SkipConnectionIdentityModule",
            "init": self.init,
            "in_ch": self.in_ch,
            "out_ch": self.out_ch,
            "epsilon": self.epsilon,
            "stride1": self.stride1,
            "stride2": self.stride2,
            "momentum": self.momentum,
            "optimizer": self.optimizer,
            "act_fn": str(self.act_fn),
            "kernel_shape1": self.kernel_shape1,
            "kernel_shape2": self.kernel_shape2,
            "component_ids": ["conv1", "batchnorm1", "conv2", "batchnorm2", "add3"],
            "components": {
                "add3": self.add3.hyperparameters,
                "conv1": self.conv1.hyperparameters,
                "conv2": self.conv2.hyperparameters,
                "batchnorm1": self.batchnorm1.hyperparameters,
                "batchnorm2": self.batchnorm2.hyperparameters,
            },
        }

    @property
    def derived_variables(self):
        """A dictionary of intermediate values computed during the
        forward/backward passes."""
        dv = {
            "conv1_out": None,
            "conv2_out": None,
            "batchnorm1_out": None,
            "batchnorm2_out": None,
            "components": {
                "add3": self.add3.derived_variables,
                "conv1": self.conv1.derived_variables,
                "conv2": self.conv2.derived_variables,
                "batchnorm1": self.batchnorm1.derived_variables,
                "batchnorm2": self.batchnorm2.derived_variables,
            },
        }
        dv.update(self._dv)
        return dv

    @property
    def gradients(self):
        """A dictionary of the accumulated module parameter gradients."""
        return {
            "components": {
                "add3": self.add3.gradients,
                "conv1": self.conv1.gradients,
                "conv2": self.conv2.gradients,
                "batchnorm1": self.batchnorm1.gradients,
                "batchnorm2": self.batchnorm2.gradients,
            }
        }

    def forward(self, X, retain_derived=True):
        """
        Compute the module output given input volume `X`.

        Parameters
        ----------
        X : :py:class:`ndarray <numpy.ndarray>` of shape (n_ex, in_rows, in_cols, in_ch)
            The input volume consisting of `n_ex` examples, each with dimension
            (`in_rows`, `in_cols`, `in_ch`).
        retain_derived : bool
            Whether to retain the variables calculated during the forward pass
            for use later during backprop. If False, this suggests the layer
            will not be expected to backprop through wrt. this input. Default
            is True.

        Returns
        -------
        Y : :py:class:`ndarray <numpy.ndarray>` of shape (n_ex, out_rows, out_cols, out_ch)
            The module output volume.
        """
        if not hasattr(self, "conv2"):
            self.in_ch = X.shape[3]
            self._init_conv2()

        conv1_out = self.conv1.forward(X, retain_derived)
        bn1_out = self.batchnorm1.forward(conv1_out, retain_derived)
        conv2_out = self.conv2.forward(bn1_out, retain_derived)
        bn2_out = self.batchnorm2.forward(conv2_out, retain_derived)
        Y = self.add3.forward([X, bn2_out], retain_derived)

        if retain_derived:
            self._dv["conv1_out"] = conv1_out
            self._dv["conv2_out"] = conv2_out
            self._dv["batchnorm1_out"] = bn1_out
            self._dv["batchnorm2_out"] = bn2_out
        return Y

    def backward(self, dLdY, retain_grads=True):
        """
        Compute the gradient of the loss with respect to the layer parameters.

        Parameters
        ----------
        dLdy : :py:class:`ndarray <numpy.ndarray>` of shape (`n_ex, out_rows, out_cols, out_ch`) or list of arrays
            The gradient(s) of the loss with respect to the module output(s).
        retain_grads : bool
            Whether to include the intermediate parameter gradients computed
            during the backward pass in the final parameter update. Default is
            True.

        Returns
        -------
        dX : :py:class:`ndarray <numpy.ndarray>` of shape (n_ex, in_rows, in_cols, in_ch)
            The gradient of the loss with respect to the module input volume.
        """
        dX, dBn2_out = self.add3.backward(dLdY, retain_grads)
        dConv2_out = self.batchnorm2.backward(dBn2_out, retain_grads)
        dBn1_out = self.conv2.backward(dConv2_out, retain_grads)
        dConv1_out = self.batchnorm1.backward(dBn1_out, retain_grads)
        dX += self.conv1.backward(dConv1_out, retain_grads)

        self._dv["dLdAdd3_X"] = dX
        self._dv["dLdBn2"] = dBn2_out
        self._dv["dLdBn1"] = dBn1_out
        self._dv["dLdConv2"] = dConv2_out
        self._dv["dLdConv1"] = dConv1_out
        return dX


class SkipConnectionConvModule(ModuleBase):
    def __init__(
        self,
        out_ch1,
        out_ch2,
        kernel_shape1,
        kernel_shape2,
        kernel_shape_skip,
        pad1=0,
        pad2=0,
        stride1=1,
        stride2=1,
        act_fn=None,
        epsilon=1e-5,
        momentum=0.9,
        stride_skip=1,
        optimizer=None,
        init="glorot_uniform",
    ):
        """
        A ResNet-like "convolution" shortcut module.

        Notes
        -----
        In contrast to :class:`SkipConnectionIdentityModule`, the additional
        `conv2d_skip` and `batchnorm_skip` layers in the shortcut path allow
        adjusting the dimensions of `X` to match the output of the main set of
        convolutions.

        .. code-block:: text

            X -> Conv2D -> Act_fn -> BatchNorm2D -> Conv2D -> BatchNorm2D -> + -> Act_fn
             \_____________________ Conv2D -> Batchnorm2D __________________/

        References
        ----------
        .. [1] He et al. (2015). "Deep residual learning for image
           recognition." https://arxiv.org/pdf/1512.03385.pdf

        Parameters
        ----------
        out_ch1 : int
            The number of filters/kernels to compute in the first convolutional
            layer.
        out_ch2 : int
            The number of filters/kernels to compute in the second
            convolutional layer.
        kernel_shape1 : 2-tuple
            The dimension of a single 2D filter/kernel in the first
            convolutional layer.
        kernel_shape2 : 2-tuple
            The dimension of a single 2D filter/kernel in the second
            convolutional layer.
        kernel_shape_skip : 2-tuple
            The dimension of a single 2D filter/kernel in the "skip"
            convolutional layer.
        stride1 : int
            The stride/hop of the convolution kernels in the first
            convolutional layer. Default is 1.
        stride2 : int
            The stride/hop of the convolution kernels in the second
            convolutional layer. Default is 1.
        stride_skip : int
            The stride/hop of the convolution kernels in the "skip"
            convolutional layer. Default is 1.
        pad1 : int, tuple, or 'same'
            The number of rows/columns of 0's to pad the input to the first
            convolutional layer with. Default is 0.
        pad2 : int, tuple, or 'same'
            The number of rows/columns of 0's to pad the input to the second
            convolutional layer with. Default is 0.
        act_fn : :doc:`Activation <numpy_ml.neural_nets.activations>` object or None
            The activation function for computing ``Y[t]``. If None, use the
            identity :math:`f(x) = x` by default. Default is None.
        epsilon : float
            A small smoothing constant to use during
            :class:`~numpy_ml.neural_nets.layers.BatchNorm2D` computation to
            avoid divide-by-zero errors. Default is 1e-5.
        momentum : float
            The momentum term for the running mean/running std calculations in
            the :class:`~numpy_ml.neural_nets.layers.BatchNorm2D` layers.  The
            closer this is to 1, the less weight will be given to the mean/std
            of the current batch (i.e., higher smoothing). Default is 0.9.
        init : str
            The weight initialization strategy. Valid entries are
            {'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform'}.
        optimizer : str or :doc:`Optimizer <numpy_ml.neural_nets.optimizers>` object
            The optimization strategy to use when performing gradient updates
            within the :class:`update` method.  If None, use the
            :class:`~numpy_ml.neural_nets.optimizers.SGD` optimizer with
            default parameters. Default is None.
        """
        super().__init__()

        self.init = init
        self.pad1 = pad1
        self.pad2 = pad2
        self.in_ch = None
        self.out_ch1 = out_ch1
        self.out_ch2 = out_ch2
        self.epsilon = epsilon
        self.stride1 = stride1
        self.stride2 = stride2
        self.momentum = momentum
        self.optimizer = optimizer
        self.stride_skip = stride_skip
        self.kernel_shape1 = kernel_shape1
        self.kernel_shape2 = kernel_shape2
        self.kernel_shape_skip = kernel_shape_skip
        self.act_fn = Affine(slope=1, intercept=0) if act_fn is None else act_fn

        self._init_params()

    def _init_params(self, X=None):
        self._dv = {}
        self.conv1 = Conv2D(
            pad=self.pad1,
            init=self.init,
            act_fn=self.act_fn,
            out_ch=self.out_ch1,
            stride=self.stride1,
            optimizer=self.optimizer,
            kernel_shape=self.kernel_shape1,
        )
        self.conv2 = Conv2D(
            pad=self.pad2,
            init=self.init,
            out_ch=self.out_ch2,
            stride=self.stride2,
            optimizer=self.optimizer,
            kernel_shape=self.kernel_shape2,
            act_fn=Affine(slope=1, intercept=0),
        )
        # we can't initialize `conv_skip` without X's dimensions; see `forward`
        # for further details
        self.batchnorm1 = BatchNorm2D(epsilon=self.epsilon, momentum=self.momentum)
        self.batchnorm2 = BatchNorm2D(epsilon=self.epsilon, momentum=self.momentum)
        self.batchnorm_skip = BatchNorm2D(epsilon=self.epsilon, momentum=self.momentum)
        self.add3 = Add(self.act_fn)

    def _calc_skip_padding(self, X):
        pads = []
        for p in [self.pad1, self.pad2]:
            if isinstance(p, int):
                pads.append((p, p, p, p))
            elif isinstance(p, tuple) and len(p) == 2:
                pads.append((p[0], p[0], p[1], p[1]))
        self.pad1, self.pad2 = pads

        # compute the dimensions of the convolution1 output
        s1 = self.stride1
        fr1, fc1 = self.kernel_shape1
        _, in_rows, in_cols, _ = X.shape
        pr11, pr12, pc11, pc12 = self.pad1

        out_rows1 = np.floor(1 + (in_rows + pr11 + pr12 - fr1) / s1).astype(int)
        out_cols1 = np.floor(1 + (in_cols + pc11 + pc12 - fc1) / s1).astype(int)

        # compute the dimensions of the convolution2 output
        s2 = self.stride2
        fr2, fc2 = self.kernel_shape2
        pr21, pr22, pc21, pc22 = self.pad2

        out_rows2 = np.floor(1 + (out_rows1 + pr21 + pr22 - fr2) / s2).astype(int)
        out_cols2 = np.floor(1 + (out_cols1 + pc21 + pc22 - fc2) / s2).astype(int)

        # finally, compute the appropriate padding dims for the skip convolution
        desired_dims = (out_rows2, out_cols2)
        self.pad_skip = calc_pad_dims_2D(
            X.shape,
            desired_dims,
            stride=self.stride_skip,
            kernel_shape=self.kernel_shape_skip,
        )

    def _init_conv_skip(self, X):
        self._calc_skip_padding(X)
        self.conv_skip = Conv2D(
            init=self.init,
            pad=self.pad_skip,
            out_ch=self.out_ch2,
            stride=self.stride_skip,
            kernel_shape=self.kernel_shape_skip,
            act_fn=Affine(slope=1, intercept=0),
            optimizer=self.optimizer,
        )

    @property
    def parameters(self):
        """A dictionary of the module parameters."""
        return {
            "components": {
                "add3": self.add3.parameters,
                "conv1": self.conv1.parameters,
                "conv2": self.conv2.parameters,
                "conv_skip": self.conv_skip.parameters
                if hasattr(self, "conv_skip")
                else None,
                "batchnorm1": self.batchnorm1.parameters,
                "batchnorm2": self.batchnorm2.parameters,
                "batchnorm_skip": self.batchnorm_skip.parameters,
            }
        }

    @property
    def hyperparameters(self):
        """A dictionary of the module hyperparameters."""
        return {
            "layer": "SkipConnectionConvModule",
            "init": self.init,
            "pad1": self.pad1,
            "pad2": self.pad2,
            "in_ch": self.in_ch,
            "out_ch1": self.out_ch1,
            "out_ch2": self.out_ch2,
            "epsilon": self.epsilon,
            "stride1": self.stride1,
            "stride2": self.stride2,
            "momentum": self.momentum,
            "act_fn": str(self.act_fn),
            "stride_skip": self.stride_skip,
            "kernel_shape1": self.kernel_shape1,
            "kernel_shape2": self.kernel_shape2,
            "kernel_shape_skip": self.kernel_shape_skip,
            "pad_skip": self.pad_skip if hasattr(self, "pad_skip") else None,
            "component_ids": [
                "add3",
                "conv1",
                "conv2",
                "conv_skip",
                "batchnorm1",
                "batchnorm2",
                "batchnorm_skip",
            ],
            "components": {
                "add3": self.add3.hyperparameters,
                "conv1": self.conv1.hyperparameters,
                "conv2": self.conv2.hyperparameters,
                "conv_skip": self.conv_skip.hyperparameters
                if hasattr(self, "conv_skip")
                else None,
                "batchnorm1": self.batchnorm1.hyperparameters,
                "batchnorm2": self.batchnorm2.hyperparameters,
                "batchnorm_skip": self.batchnorm_skip.hyperparameters,
            },
        }

    @property
    def derived_variables(self):
        """A dictionary of intermediate values computed during the
        forward/backward passes."""
        dv = {
            "conv1_out": None,
            "conv2_out": None,
            "conv_skip_out": None,
            "batchnorm1_out": None,
            "batchnorm2_out": None,
            "batchnorm_skip_out": None,
            "components": {
                "add3": self.add3.derived_variables,
                "conv1": self.conv1.derived_variables,
                "conv2": self.conv2.derived_variables,
                "conv_skip": self.conv_skip.derived_variables
                if hasattr(self, "conv_skip")
                else None,
                "batchnorm1": self.batchnorm1.derived_variables,
                "batchnorm2": self.batchnorm2.derived_variables,
                "batchnorm_skip": self.batchnorm_skip.derived_variables,
            },
        }
        dv.update(self._dv)
        return dv

    @property
    def gradients(self):
        """A dictionary of the accumulated module parameter gradients."""
        return {
            "components": {
                "add3": self.add3.gradients,
                "conv1": self.conv1.gradients,
                "conv2": self.conv2.gradients,
                "conv_skip": self.conv_skip.gradients
                if hasattr(self, "conv_skip")
                else None,
                "batchnorm1": self.batchnorm1.gradients,
                "batchnorm2": self.batchnorm2.gradients,
                "batchnorm_skip": self.batchnorm_skip.gradients,
            }
        }

    def forward(self, X, retain_derived=True):
        """
        Compute the layer output given input volume `X`.

        Parameters
        ----------
        X : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, in_rows, in_cols, in_ch)`
            The input volume consisting of `n_ex` examples, each with dimension
            (`in_rows`, `in_cols`, `in_ch`).
        retain_derived : bool
            Whether to retain the variables calculated during the forward pass
            for use later during backprop. If False, this suggests the layer
            will not be expected to backprop through wrt. this input. Default
            is True.

        Returns
        -------
        Y : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, out_rows, out_cols, out_ch)`
            The module output volume.
        """
        # now that we have the input dims for X we can initialize the proper
        # padding in the `conv_skip` layer
        if not hasattr(self, "conv_skip"):
            self._init_conv_skip(X)
            self.in_ch = X.shape[3]

        conv1_out = self.conv1.forward(X, retain_derived)
        bn1_out = self.batchnorm1.forward(conv1_out, retain_derived)
        conv2_out = self.conv2.forward(bn1_out, retain_derived)
        bn2_out = self.batchnorm2.forward(conv2_out, retain_derived)
        conv_skip_out = self.conv_skip.forward(X, retain_derived)
        bn_skip_out = self.batchnorm_skip.forward(conv_skip_out, retain_derived)
        Y = self.add3.forward([bn_skip_out, bn2_out], retain_derived)

        if retain_derived:
            self._dv["conv1_out"] = conv1_out
            self._dv["conv2_out"] = conv2_out
            self._dv["batchnorm1_out"] = bn1_out
            self._dv["batchnorm2_out"] = bn2_out
            self._dv["conv_skip_out"] = conv_skip_out
            self._dv["batchnorm_skip_out"] = bn_skip_out
        return Y

    def backward(self, dLdY, retain_grads=True):
        """
        Compute the gradient of the loss with respect to the module parameters.

        Parameters
        ----------
        dLdy : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, out_rows, out_cols, out_ch)`
        or list of arrays
            The gradient(s) of the loss with respect to the module output(s).
        retain_grads : bool
            Whether to include the intermediate parameter gradients computed
            during the backward pass in the final parameter update. Default is
            True.

        Returns
        -------
        dX : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, in_rows, in_cols, in_ch)`
            The gradient of the loss with respect to the module input volume.
        """
        dBnskip_out, dBn2_out = self.add3.backward(dLdY)
        dConvskip_out = self.batchnorm_skip.backward(dBnskip_out)
        dX = self.conv_skip.backward(dConvskip_out)

        dConv2_out = self.batchnorm2.backward(dBn2_out)
        dBn1_out = self.conv2.backward(dConv2_out)
        dConv1_out = self.batchnorm1.backward(dBn1_out)
        dX += self.conv1.backward(dConv1_out)

        if retain_grads:
            self._dv["dLdAdd3_X"] = dX
            self._dv["dLdBn1"] = dBn1_out
            self._dv["dLdBn2"] = dBn2_out
            self._dv["dLdConv1"] = dConv1_out
            self._dv["dLdConv2"] = dConv2_out
            self._dv["dLdBnSkip"] = dBnskip_out
            self._dv["dLdConvSkip"] = dConvskip_out
        return dX


class BidirectionalLSTM(ModuleBase):
    def __init__(
        self,
        n_out,
        act_fn=None,
        gate_fn=None,
        merge_mode="concat",
        init="glorot_uniform",
        optimizer=None,
    ):
        """
        A single bidirectional long short-term memory (LSTM) layer.

        Parameters
        ----------
        n_out : int
            The dimension of a single hidden state / output on a given timestep
        act_fn : :doc:`Activation <numpy_ml.neural_nets.activations>` object or None
            The activation function for computing ``A[t]``. If not specified,
            use :class:`~numpy_ml.neural_nets.activations.Tanh` by default.
        gate_fn : :doc:`Activation <numpy_ml.neural_nets.activations>` object or None
            The gate function for computing the update, forget, and output
            gates. If not specified, use
            :class:`~numpy_ml.neural_nets.activations.Sigmoid` by default.
        merge_mode : {"sum", "multiply", "concat", "average"}
            Mode by which outputs of the forward and backward LSTMs will be
            combined. Default is 'concat'.
        optimizer : str or :doc:`Optimizer <numpy_ml.neural_nets.optimizers>` object or None
            The optimization strategy to use when performing gradient updates
            within the `update` method.  If None, use the
            :class:`~numpy_ml.neural_nets.optimizers.SGD` optimizer with
            default parameters. Default is None.
        init : {'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform'}
            The weight initialization strategy. Default is 'glorot_uniform'.
        """
        super().__init__()

        self.init = init
        self.n_in = None
        self.n_out = n_out
        self.optimizer = optimizer
        self.merge_mode = merge_mode
        self.act_fn = Tanh() if act_fn is None else act_fn
        self.gate_fn = Sigmoid() if gate_fn is None else gate_fn
        self._init_params()

    def _init_params(self):
        self.cell_fwd = LSTMCell(
            init=self.init,
            n_out=self.n_out,
            act_fn=self.act_fn,
            gate_fn=self.gate_fn,
            optimizer=self.optimizer,
        )
        self.cell_bwd = LSTMCell(
            init=self.init,
            n_out=self.n_out,
            act_fn=self.act_fn,
            gate_fn=self.gate_fn,
            optimizer=self.optimizer,
        )

    def forward(self, X):
        """
        Run a forward pass across all timesteps in the input.

        Parameters
        ----------
        X : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, n_in, n_t)`
            Input consisting of `n_ex` examples each of dimensionality `n_in`
            and extending for `n_t` timesteps.

        Returns
        -------
        Y : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, n_out, n_t)`
            The value of the hidden state for each of the `n_ex` examples
            across each of the `n_t` timesteps.
        """
        Y_fwd, Y_bwd, Y = [], [], []
        n_ex, self.n_in, n_t = X.shape

        # forward LSTM
        for t in range(n_t):
            yt, ct = self.cell_fwd.forward(X[:, :, t])
            Y_fwd.append(yt)

        # backward LSTM
        for t in reversed(range(n_t)):
            yt, ct = self.cell_bwd.forward(X[:, :, t])
            Y_bwd.insert(0, yt)

        # merge forward and backward states
        for t in range(n_t):
            if self.merge_mode == "concat":
                Y.append(np.concatenate([Y_fwd[t], Y_bwd[t]], axis=1))
            elif self.merge_mode == "sum":
                Y.append(Y_fwd[t] + Y_bwd[t])
            elif self.merge_mode == "average":
                Y.append((Y_fwd[t] + Y_bwd[t]) / 2)
            elif self.merge_mode == "multiply":
                Y.append(Y_fwd[t] * Y_bwd[t])

        self.Y_fwd, self.Y_bwd = Y_fwd, Y_bwd
        return np.dstack(Y)

    def backward(self, dLdA):
        """
        Run a backward pass across all timesteps in the input.

        Parameters
        ----------
        dLdA : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, n_out, n_t)`
            The gradient of the loss with respect to the layer output for each
            of the `n_ex` examples across all `n_t` timesteps.

        Returns
        -------
        dLdX : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, n_in, n_t)`
            The value of the hidden state for each of the `n_ex` examples
            across each of the `n_t` timesteps.
        """
        assert self.trainable, "Layer is frozen"

        n_ex, n_out, n_t = dLdA.shape
        dLdX_f, dLdX_b, dLdX = [], [], []

        # forward LSTM
        for t in reversed(range(n_t)):
            if self.merge_mode == "concat":
                dLdXt_f = self.cell_fwd.backward(dLdA[:, : self.n_out, t])
            elif self.merge_mode == "sum":
                dLdXt_f = self.cell_fwd.backward(dLdA[:, :, t])
            elif self.merge_mode == "multiplty":
                dLdXt_f = self.cell_fwd.backward(dLdA[:, :, t] * self.Y_bwd[t])
            elif self.merge_mode == "average":
                dLdXt_f = self.cell_fwd.backward(dLdA[:, :, t] * 0.5)
            dLdX_f.insert(0, dLdXt_f)

        # backward LSTM
        for t in range(n_t):
            if self.merge_mode == "concat":
                dLdXt_b = self.cell_bwd.backward(dLdA[:, self.n_out :, t])
            elif self.merge_mode == "sum":
                dLdXt_b = self.cell_bwd.backward(dLdA[:, :, t])
            elif self.merge_mode == "multiplty":
                dLdXt_b = self.cell_bwd.backward(dLdA[:, :, t] * self.Y_fwd[t])
            elif self.merge_mode == "average":
                dLdXt_b = self.cell_bwd.backward(dLdA[:, :, t] * 0.5)
            dLdX_b.append(dLdXt_b)

        for t in range(n_t):
            dLdX.append(dLdX_f[t] + dLdX_b[t])

        return np.dstack(dLdX)

    @property
    def derived_variables(self):
        """A dictionary of intermediate values computed during the
        forward/backward passes."""
        return {
            "components": {
                "cell_fwd": self.cell_fwd.derived_variables,
                "cell_bwd": self.cell_bwd.derived_variables,
            }
        }

    @property
    def gradients(self):
        """A dictionary of the accumulated module parameter gradients."""
        return {
            "components": {
                "cell_fwd": self.cell_fwd.gradients,
                "cell_bwd": self.cell_bwd.gradients,
            }
        }

    @property
    def parameters(self):
        """A dictionary of the module parameters."""
        return {
            "components": {
                "cell_fwd": self.cell_fwd.parameters,
                "cell_bwd": self.cell_bwd.parameters,
            }
        }

    @property
    def hyperparameters(self):
        """A dictionary of the module hyperparameters."""
        return {
            "layer": "BidirectionalLSTM",
            "init": self.init,
            "n_in": self.n_in,
            "n_out": self.n_out,
            "act_fn": str(self.act_fn),
            "optimizer": self.optimizer,
            "merge_mode": self.merge_mode,
            "component_ids": ["cell_fwd", "cell_bwd"],
            "components": {
                "cell_fwd": self.cell_fwd.hyperparameters,
                "cell_bwd": self.cell_bwd.hyperparameters,
            },
        }


class MultiHeadedAttentionModule(ModuleBase):
    def __init__(self, n_heads=8, dropout_p=0, init="glorot_uniform", optimizer=None):
        """
        A mutli-headed attention module.

        Notes
        -----
        Multi-head attention allows a model to jointly attend to information from
        different representation subspaces at different positions. With a
        single head, this information would get averaged away when the
        attention weights are combined with the value

        .. math::

            \\text{MultiHead}(\mathbf{Q}, \mathbf{K}, \mathbf{V})
                = [\\text{head}_1; ...; \\text{head}_h] \\mathbf{W}^{(O)}

        where

        .. math::

            \\text{head}_i = \\text{SDP_attention}(
                \mathbf{Q W}_i^{(Q)}, \mathbf{K W}_i^{(K)}, \mathbf{V W}_i^{(V)})

        and the projection weights are parameter matrices:

        .. math::

            \mathbf{W}_i^{(Q)}  &\in
                \mathbb{R}^{(\\text{kqv_dim} \ \\times \ \\text{latent_dim})} \\\\
            \mathbf{W}_i^{(K)}  &\in
                \mathbb{R}^{(\\text{kqv_dim} \ \\times \ \\text{latent_dim})} \\\\
            \mathbf{W}_i^{(V)}  &\in
                \mathbb{R}^{(\\text{kqv_dim} \ \\times \ \\text{latent_dim})} \\\\
            \mathbf{W}^{(O)}  &\in
                \mathbb{R}^{(\\text{n_heads} \cdot \\text{latent_dim} \ \\times \ \\text{kqv_dim})}

        Importantly, the current module explicitly assumes that

        .. math::

            \\text{kqv_dim} = \\text{dim(query)} = \\text{dim(keys)} = \\text{dim(values)}

        and that

        .. math::

            \\text{latent_dim} = \\text{kqv_dim / n_heads}

        **[MH Attention Head h]**:

        .. code-block:: text

            K --> W_h^(K) ------\\
            V --> W_h^(V) ------- > DP_Attention --> head_h
            Q --> W_h^(Q) ------/

        The full **[MultiHeadedAttentionModule]** then becomes

        .. code-block:: text

                  -----------------
            K --> | [Attn Head 1] | --> head_1 --\\
            V --> | [Attn Head 2] | --> head_2 --\\
            Q --> |      ...      |      ...       --> Concat --> W^(O) --> MH_out
                  | [Attn Head Z] | --> head_Z --/
                  -----------------

        Due to the reduced dimension of each head, the total computational cost
        is similar to that of a single attention head with full (i.e., kqv_dim)
        dimensionality.

        Parameters
        ----------
        n_heads : int
            The number of simultaneous attention heads to use. Note that the
            larger `n_heads`, the smaller the dimensionality of any single
            head, since ``latent_dim = kqv_dim / n_heads``. Default is 8.
        dropout_p : float in [0, 1)
            The dropout propbability during training, applied to the output of
            the softmax in each dot-product attention head. If 0, no dropout is
            applied. Default is 0.
        init : {'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform'}
            The weight initialization strategy. Default is 'glorot_uniform'.
        optimizer : str, :doc:`Optimizer <numpy_ml.neural_nets.optimizers>` object, or None
            The optimization strategy to use when performing gradient updates
            within the :meth:`update` method.  If None, use the
            :class:`~numpy_ml.neural_nets.optimizers.SGD` optimizer with default
            parameters. Default is None.
        """
        self.init = init
        self.kqv_dim = None
        self.projections = {}
        self.n_heads = n_heads
        self.optimizer = optimizer
        self.dropout_p = dropout_p
        self.is_initialized = False

    def _init_params(self):
        self._dv = {}

        # assume dim(keys) = dim(query) = dim(values)
        assert self.kqv_dim % self.n_heads == 0
        self.latent_dim = self.kqv_dim // self.n_heads

        self.attention = DotProductAttention(scale=True, dropout_p=self.dropout_p)
        self.projections = {
            k: Dropout(
                FullyConnected(
                    init=self.init,
                    n_out=self.kqv_dim,
                    optimizer=self.optimizer,
                    act_fn="Affine(slope=1, intercept=0)",
                ),
                self.dropout_p,
            )
            for k in ["Q", "K", "V", "O"]
        }

        self.is_initialized = True

    def forward(self, Q, K, V):
        if not self.is_initialized:
            self.kqv_dim = Q.shape[-1]
            self._init_params()

        # project queries, keys, and values into the `latent_dim`-dimensional subspace
        n_ex = Q.shape[0]
        for k, x in zip(["Q", "K", "V"], [Q, K, V]):
            proj = self.projections[k].forward(x)
            proj = proj.reshape(n_ex, -1, self.n_heads, self.latent_dim).swapaxes(1, 2)
            self._dv["{}_proj".format(k)] = proj

        dv = self.derived_variables
        Q_proj, K_proj, V_proj = dv["Q_proj"], dv["K_proj"], dv["V_proj"]

        # apply scaled dot-product attention to the projected vectors
        attn = self.attention
        attn_out = attn.forward(Q_proj, K_proj, V_proj)
        self._dv["attention_weights"] = attn.derived_variables["attention_weights"]

        # concatenate the different heads using `reshape` to create an
        # `kqv_dim`-dim vector
        attn_out = attn_out.swapaxes(1, 2).reshape(n_ex, self.kqv_dim)
        self._dv["attention_out"] = attn_out.reshape(n_ex, -1, self.kqv_dim)

        # apply the final output projection
        Y = self.projections["O"].forward(attn_out)
        Y = Y.reshape(n_ex, -1, self.kqv_dim)
        return Y

    def backward(self, dLdy):
        n_ex = dLdy.shape[0]
        dLdy = dLdy.reshape(n_ex, self.kqv_dim)
        dLdX = self.projections["O"].backward(dLdy)
        dLdX = dLdX.reshape(n_ex, self.n_heads, -1, self.latent_dim)

        dLdQ_proj, dLdK_proj, dLdV_proj = self.attention.backward(dLdX)

        self._dv["dQ_proj"] = dLdQ_proj
        self._dv["dK_proj"] = dLdK_proj
        self._dv["dV_proj"] = dLdV_proj

        dLdQ_proj = dLdQ_proj.reshape(n_ex, self.kqv_dim)
        dLdK_proj = dLdK_proj.reshape(n_ex, self.kqv_dim)
        dLdV_proj = dLdV_proj.reshape(n_ex, self.kqv_dim)

        dLdQ = self.projections["Q"].backward(dLdQ_proj)
        dLdK = self.projections["K"].backward(dLdK_proj)
        dLdV = self.projections["V"].backward(dLdV_proj)
        return dLdQ, dLdK, dLdV

    @property
    def derived_variables(self):
        """A dictionary of intermediate values computed during the
        forward/backward passes."""
        dv = {
            "Q_proj": None,
            "K_proj": None,
            "V_proj": None,
            "components": {
                "Q": self.projections["Q"].derived_variables,
                "K": self.projections["K"].derived_variables,
                "V": self.projections["V"].derived_variables,
                "O": self.projections["O"].derived_variables,
                "attention": self.attention.derived_variables,
            },
        }
        dv.update(self._dv)
        return dv

    @property
    def gradients(self):
        """A dictionary of the accumulated module parameter gradients."""
        return {
            "components": {
                "Q": self.projections["Q"].gradients,
                "K": self.projections["K"].gradients,
                "V": self.projections["V"].gradients,
                "O": self.projections["O"].gradients,
                "attention": self.attention.gradients,
            }
        }

    @property
    def parameters(self):
        """A dictionary of the module parameters."""
        return {
            "components": {
                "Q": self.projections["Q"].parameters,
                "K": self.projections["K"].parameters,
                "V": self.projections["V"].parameters,
                "O": self.projections["O"].parameters,
                "attention": self.attention.parameters,
            }
        }

    @property
    def hyperparameters(self):
        """A dictionary of the module hyperparameters."""
        return {
            "layer": "MultiHeadedAttentionModule",
            "init": self.init,
            "kqv_dim": self.kqv_dim,
            "latent_dim": self.latent_dim,
            "n_heads": self.n_heads,
            "dropout_p": self.dropout_p,
            "component_ids": ["attention", "Q", "K", "V", "O"],
            "components": {
                "Q": self.projections["Q"].hyperparameters,
                "K": self.projections["K"].hyperparameters,
                "V": self.projections["V"].hyperparameters,
                "O": self.projections["O"].hyperparameters,
                "attention": self.attention.hyperparameters,
            },
        }
