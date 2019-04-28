from abc import ABC, abstractmethod

import re
import numpy as np

from utils import calc_pad_dims_2D
from activations import Tanh, Sigmoid, ReLU, LeakyReLU, Affine
from layers import Conv1D, Conv2D, BatchNorm2D, Add, Multiply, LSTMCell


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
        self, ch_residual, ch_dilation, dilation, kernel_width, init="glorot_uniform"
    ):
        """
        A WaveNet-like residual block with causal dilated convolutions.

        *Skip path in* >-------------------------------------------> + --------> *Skip path out*
                          Causal      |--> Tanh --|                  |
        *Main    |--> Dilated Conv1D -|           * --> 1x1 Conv1D --|
         path >--|                    |--> Sigm --|                  |
         in*     |-------------------------------------------------> + --------> *Main path out*
                                     *Residual path*

        On the final block, the output of the skip path is further processed to
        produce the network predictions.

        See van den Oord et al. (2016) at https://arxiv.org/pdf/1609.03499.pdf
        for further details.

        Parameters
        ----------
        ch_residual : int
            The number of output channels for the 1x1 Conv1D layer in the main
            path
        ch_dilation : int
            The number of output channels for the causal dilated Conv1D layer
            in the main path
        dilation : int
            The dilation rate for the causal dilated Conv1D layer in the main
            path
        kernel_width : int
            The width of the causal dilated Conv1D kernel in the main path
        init : str (default: 'glorot_uniform')
            The weight initialization strategy. Valid entries are
            {'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform'}
        """
        super().__init__()

        self.init = init
        self.dilation = dilation
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
            act_fn=Affine(slope=1, intercept=0),
        )

        self.add_residual = Add(act_fn=Affine(slope=1, intercept=0))
        self.add_skip = Add(act_fn=Affine(slope=1, intercept=0))

    @property
    def parameters(self):
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
        return {
            "layer": "WavenetResidualModule",
            "init": self.init,
            "dilation": self.dilation,
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
        init="glorot_uniform",
    ):
        """
        A ResNet-like "identity" shortcut module. Enforces `same`
        padding during each convolution to ensure module output has same dims
        as its input.

        X -> Conv2D -> Act_fn -> BatchNorm2D -> Conv2D -> BatchNorm2D -> + -> Act_fn
         \______________________________________________________________/

        See He et al. (2015) at https://arxiv.org/pdf/1512.03385.pdf for
        further details.

        Parameters
        ----------
        out_ch : int
            The number of filters/kernels to compute in the first convolutional
            layer
        kernel_shape1 : 2-tuple
            The dimension of a single 2D filter/kernel in the first
            convolutional layer
        kernel_shape2 : 2-tuple
            The dimension of a single 2D filter/kernel in the second
            convolutional layer
        stride1 : int (default: 1)
            The stride/hop of the convolution kernels in the first convolutional layer
        stride2 : int (default: 1)
            The stride/hop of the convolution kernels in the second convolutional layer
        act_fn : `activations.Activation` instance (default: None)
            The activation function for computing Y[t]. If `None`, use the
            identity f(x) = x by default
        epsilon : float (default : 1e-5)
            A small smoothing constant to use during BatchNorm2D computation to
            avoid divide-by-zero errors.
        momentum : float (default: 0.9)
            The momentum term for the running mean/running std calculations in
            the BatchNorm2D layers.  The closer this is to 1, the less weight
            will be given to the mean/std of the current batch (i.e., higher
            smoothing)
        init : str (default: 'glorot_uniform')
            The weight initialization strategy. Valid entries are
            {'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform'}
        """
        super().__init__()

        self.init = init
        self.in_ch = None
        self.out_ch = out_ch
        self.epsilon = epsilon
        self.stride1 = stride1
        self.stride2 = stride2
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
            kernel_shape=self.kernel_shape2,
            act_fn=Affine(slope=1, intercept=0),
        )

    @property
    def parameters(self):
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
        return {
            "layer": "SkipConnectionIdentityModule",
            "init": self.init,
            "in_ch": self.in_ch,
            "out_ch": self.out_ch,
            "epsilon": self.epsilon,
            "stride1": self.stride1,
            "stride2": self.stride2,
            "momentum": self.momentum,
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
        return {
            "components": {
                "add3": self.add3.gradients,
                "conv1": self.conv1.gradients,
                "conv2": self.conv2.gradients,
                "batchnorm1": self.batchnorm1.gradients,
                "batchnorm2": self.batchnorm2.gradients,
            }
        }

    def forward(self, X):
        if not hasattr(self, "conv2"):
            self.in_ch = X.shape[3]
            self._init_conv2()

        conv1_out = self.conv1.forward(X)
        bn1_out = self.batchnorm1.forward(conv1_out)
        conv2_out = self.conv2.forward(bn1_out)
        bn2_out = self.batchnorm2.forward(conv2_out)
        Y = self.add3.forward([X, bn2_out])

        self._dv["conv1_out"] = conv1_out
        self._dv["conv2_out"] = conv2_out
        self._dv["batchnorm1_out"] = bn1_out
        self._dv["batchnorm2_out"] = bn2_out
        return Y

    def backward(self, dLdY):
        dX, dBn2_out = self.add3.backward(dLdY)
        dConv2_out = self.batchnorm2.backward(dBn2_out)
        dBn1_out = self.conv2.backward(dConv2_out)
        dConv1_out = self.batchnorm1.backward(dBn1_out)
        dX += self.conv1.backward(dConv1_out)

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
        init="glorot_uniform",
    ):
        """
        A ResNet-like "convolution" shortcut module. The additional
        `conv2d_skip` and `batchnorm_skip` layers in the shortcut path allow
        adjusting the dimensions of X to match the output of the main set of
        convolutions.

        X -> Conv2D -> Act_fn -> BatchNorm2D -> Conv2D -> BatchNorm2D -> + -> Act_fn
         \_____________________ Conv2D -> Batchnorm2D __________________/

        See He et al. (2015) at https://arxiv.org/pdf/1512.03385.pdf for
        further details.

        Parameters
        ----------
        out_ch1 : int
            The number of filters/kernels to compute in the first convolutional
            layer
        out_ch2 : int
            The number of filters/kernels to compute in the second
            convolutional layer
        kernel_shape1 : 2-tuple
            The dimension of a single 2D filter/kernel in the first
            convolutional layer
        kernel_shape2 : 2-tuple
            The dimension of a single 2D filter/kernel in the second
            convolutional layer
        kernel_shape_skip : 2-tuple
            The dimension of a single 2D filter/kernel in the "skip"
            convolutional layer
        stride1 : int (default: 1)
            The stride/hop of the convolution kernels in the first convolutional layer
        stride2 : int (default: 1)
            The stride/hop of the convolution kernels in the second convolutional layer
        stride_skip : int (default: 1)
            The stride/hop of the convolution kernels in the "skip" convolutional layer
        pad1 : int, tuple, or 'same' (default: 0)
            The number of rows/columns of 0's to pad the input to the first
            convolutional layer with
        pad2 : int, tuple, or 'same' (default: 0)
            The number of rows/columns of 0's to pad the input to the second
            convolutional layer with
        act_fn : `activations.Activation` instance (default: None)
            The activation function for computing Y[t]. If `None`, use the
            identity f(x) = x by default
        epsilon : float (default : 1e-5)
            A small smoothing constant to use during BatchNorm2D computation to
            avoid divide-by-zero errors.
        momentum : float (default: 0.9)
            The momentum term for the running mean/running std calculations in
            the BatchNorm2D layers.  The closer this is to 1, the less weight
            will be given to the mean/std of the current batch (i.e., higher
            smoothing)
        init : str (default: 'glorot_uniform')
            The weight initialization strategy. Valid entries are
            {'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform'}
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
            kernel_shape=self.kernel_shape1,
        )
        self.conv2 = Conv2D(
            pad=self.pad2,
            init=self.init,
            out_ch=self.out_ch2,
            stride=self.stride2,
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
        )

    @property
    def parameters(self):
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

    def forward(self, X):
        # now that we have the input dims for X we can initialize the proper
        # padding in the `conv_skip` layer
        if not hasattr(self, "conv_skip"):
            self._init_conv_skip(X)
            self.in_ch = X.shape[3]

        conv1_out = self.conv1.forward(X)
        bn1_out = self.batchnorm1.forward(conv1_out)
        conv2_out = self.conv2.forward(bn1_out)
        bn2_out = self.batchnorm2.forward(conv2_out)
        conv_skip_out = self.conv_skip.forward(X)
        bn_skip_out = self.batchnorm_skip.forward(conv_skip_out)
        Y = self.add3.forward([bn_skip_out, bn2_out])

        self._dv["conv1_out"] = conv1_out
        self._dv["conv2_out"] = conv2_out
        self._dv["batchnorm1_out"] = bn1_out
        self._dv["batchnorm2_out"] = bn2_out
        self._dv["conv_skip_out"] = conv_skip_out
        self._dv["batchnorm_skip_out"] = bn_skip_out
        return Y

    def backward(self, dLdY):
        dBnskip_out, dBn2_out = self.add3.backward(dLdY)
        dConvskip_out = self.batchnorm_skip.backward(dBnskip_out)
        dX = self.conv_skip.backward(dConvskip_out)

        dConv2_out = self.batchnorm2.backward(dBn2_out)
        dBn1_out = self.conv2.backward(dConv2_out)
        dConv1_out = self.batchnorm1.backward(dBn1_out)
        dX += self.conv1.backward(dConv1_out)

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
    ):
        """
        A single bidirectional long short-term memory (LSTM) layer.

        Parameters
        ----------
        n_out : int
            The dimension of a single hidden state / output on a given timestep
        act_fn : `activations.Activation` instance (default: None)
            The activation function for computing A[t]. If not specified, use
            Tanh by default.
        gate_fn : `activations.Activation` instance (default: None)
            The gate function for computing the update, forget, and output
            gates. If not specified, use Sigmoid by default.
        merge_mode : str (default: "concat")
            Mode by which outputs of the forward and backward LSTMs will be
            combined. Valid values are {"sum", "multiply", "concat", "average"}.
        init : str (default: 'glorot_uniform')
            The weight initialization strategy. Valid entries are
            {'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform'}
        """
        super().__init__()

        self.init = init
        self.n_in = None
        self.n_out = n_out
        self.merge_mode = merge_mode
        self.act_fn = Tanh() if act_fn is None else act_fn
        self.gate_fn = Sigmoid() if gate_fn is None else gate_fn
        self._init_params()

    def _init_params(self):
        self.cell_fwd = LSTMCell(
            init=self.init, n_out=self.n_out, act_fn=self.act_fn, gate_fn=self.gate_fn
        )
        self.cell_bwd = LSTMCell(
            init=self.init, n_out=self.n_out, act_fn=self.act_fn, gate_fn=self.gate_fn
        )

    def forward(self, X):
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
        return {
            "components": {
                "cell_fwd": self.cell_fwd.derived_variables,
                "cell_bwd": self.cell_bwd.derived_variables,
            }
        }

    @property
    def gradients(self):
        return {
            "components": {
                "cell_fwd": self.cell_fwd.gradients,
                "cell_bwd": self.cell_bwd.gradients,
            }
        }

    @property
    def parameters(self):
        return {
            "components": {
                "cell_fwd": self.cell_fwd.parameters,
                "cell_bwd": self.cell_bwd.parameters,
            }
        }

    @property
    def hyperparameters(self):
        return {
            "layer": "BidirectionalLSTM",
            "init": self.init,
            "n_in": self.n_in,
            "n_out": self.n_out,
            "act_fn": str(self.act_fn),
            "merge_mode": self.merge_mode,
            "component_ids": ["cell_fwd", "cell_bwd"],
            "components": {
                "cell_fwd": self.cell_fwd.hyperparameters,
                "cell_bwd": self.cell_bwd.hyperparameters,
            },
        }
