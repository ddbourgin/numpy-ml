import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

#######################################################################
#       Gold-standard implementations for testing custom layers       #
#                       (Requires Pytorch)                            #
#######################################################################


def torchify(var, requires_grad=True):
    return torch.autograd.Variable(torch.FloatTensor(var), requires_grad=requires_grad)


def torch_gradient_generator(fn, **kwargs):
    def get_grad(z):
        z1 = torch.autograd.Variable(torch.FloatTensor(z), requires_grad=True)
        z2 = fn(z1, **kwargs).sum()
        z2.backward()
        grad = z1.grad.numpy()
        return grad

    return get_grad


def torch_xe_grad(y, z):
    z = torch.autograd.Variable(torch.FloatTensor(z), requires_grad=True)
    y = torch.LongTensor(y.argmax(axis=1))
    loss = F.cross_entropy(z, y, size_average=False).sum()
    loss.backward()
    grad = z.grad.numpy()
    return grad


def torch_mse_grad(y, z, act_fn):
    y = torch.FloatTensor(y)
    z = torch.autograd.Variable(torch.FloatTensor(z), requires_grad=True)
    y_pred = act_fn(z)
    loss = F.mse_loss(y_pred, y, size_average=False).sum()
    loss.backward()
    grad = z.grad.numpy()
    return grad


class TorchVAELoss(nn.Module):
    def __init__(self):
        pass

    def extract_grads(self, X, X_recon, t_mean, t_log_var):
        eps = np.finfo(float).eps
        X = torchify(X, requires_grad=False)
        X_recon = torchify(np.clip(X_recon, eps, 1 - eps))
        t_mean = torchify(t_mean)
        t_log_var = torchify(t_log_var)

        BCE = torch.sum(F.binary_cross_entropy(X_recon, X, reduce=False), dim=1)

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + t_log_var - t_mean.pow(2) - t_log_var.exp(), dim=1)

        loss = torch.mean(BCE + KLD)
        loss.backward()

        grads = {
            "loss": loss.detach().numpy(),
            "dX_recon": X_recon.grad.numpy(),
            "dt_mean": t_mean.grad.numpy(),
            "dt_log_var": t_log_var.grad.numpy(),
        }
        return grads


class TorchLinearActivation(nn.Module):
    def __init__(self):
        super(TorchLinearActivation, self).__init__()
        pass

    @staticmethod
    def forward(input):
        return input

    @staticmethod
    def backward(grad_output):
        return torch.ones_like(grad_output)


class TorchBatchNormLayer(nn.Module):
    def __init__(self, n_in, params, mode, momentum=0.9, epsilon=1e-5):
        super(TorchBatchNormLayer, self).__init__()

        scaler = params["scaler"]
        intercept = params["intercept"]

        if mode == "1D":
            self.layer1 = nn.BatchNorm1d(
                num_features=n_in, momentum=1 - momentum, eps=epsilon, affine=True
            )
        elif mode == "2D":
            self.layer1 = nn.BatchNorm2d(
                num_features=n_in, momentum=1 - momentum, eps=epsilon, affine=True
            )

        self.layer1.weight = nn.Parameter(torch.FloatTensor(scaler))
        self.layer1.bias = nn.Parameter(torch.FloatTensor(intercept))

    def forward(self, X):
        # (N, H, W, C) -> (N, C, H, W)
        if X.ndim == 4:
            X = np.moveaxis(X, [0, 1, 2, 3], [0, -2, -1, -3])

        if not isinstance(X, torch.Tensor):
            X = torchify(X)

        self.X = X
        self.Y = self.layer1(self.X)
        self.Y.retain_grad()

    def extract_grads(self, X, Y_true=None):
        self.forward(X)

        if isinstance(Y_true, np.ndarray):
            Y_true = np.moveaxis(Y_true, [0, 1, 2, 3], [0, -2, -1, -3])
            self.loss1 = (
                0.5 * F.mse_loss(self.Y, torchify(Y_true), size_average=False).sum()
            )
        else:
            self.loss1 = self.Y.sum()

        self.loss1.backward()

        X_np = self.X.detach().numpy()
        Y_np = self.Y.detach().numpy()
        dX_np = self.X.grad.numpy()
        dY_np = self.Y.grad.numpy()

        if self.X.dim() == 4:
            orig, X_swap = [0, 1, 2, 3], [0, -1, -3, -2]
            if isinstance(Y_true, np.ndarray):
                Y_true = np.moveaxis(Y_true, orig, X_swap)
            X_np = np.moveaxis(X_np, orig, X_swap)
            Y_np = np.moveaxis(Y_np, orig, X_swap)
            dX_np = np.moveaxis(dX_np, orig, X_swap)
            dY_np = np.moveaxis(dY_np, orig, X_swap)

        grads = {
            "loss": self.loss1.detach().numpy(),
            "X": X_np,
            "momentum": 1 - self.layer1.momentum,
            "epsilon": self.layer1.eps,
            "intercept": self.layer1.bias.detach().numpy(),
            "scaler": self.layer1.weight.detach().numpy(),
            "running_mean": self.layer1.running_mean.detach().numpy(),
            "running_var": self.layer1.running_var.detach().numpy(),
            "y": Y_np,
            "dLdy": dY_np,
            "dLdIntercept": self.layer1.bias.grad.numpy(),
            "dLdScaler": self.layer1.weight.grad.numpy(),
            "dLdX": dX_np,
        }
        if isinstance(Y_true, np.ndarray):
            grads["Y_true"] = Y_true
        return grads


class TorchAddLayer(nn.Module):
    def __init__(self, act_fn, **kwargs):
        super(TorchAddLayer, self).__init__()
        self.act_fn = act_fn

    def forward(self, Xs):
        self.Xs = []
        x = Xs[0].copy()
        if not isinstance(x, torch.Tensor):
            x = torchify(x)

        self.sum = x.clone()
        x.retain_grad()
        self.Xs.append(x)

        for i in range(1, len(Xs)):
            x = Xs[i]
            if not isinstance(x, torch.Tensor):
                x = torchify(x)

            x.retain_grad()
            self.Xs.append(x)
            self.sum += x

        self.sum.retain_grad()
        self.Y = self.act_fn(self.sum)
        self.Y.retain_grad()
        return self.Y

    def extract_grads(self, X):
        self.forward(X)
        self.loss = self.Y.sum()
        self.loss.backward()
        grads = {
            "Xs": X,
            "Sum": self.sum.detach().numpy(),
            "Y": self.Y.detach().numpy(),
            "dLdY": self.Y.grad.numpy(),
            "dLdSum": self.sum.grad.numpy(),
        }
        grads.update(
            {"dLdX{}".format(i + 1): xi.grad.numpy() for i, xi in enumerate(self.Xs)}
        )
        return grads


class TorchMultiplyLayer(nn.Module):
    def __init__(self, act_fn, **kwargs):
        super(TorchMultiplyLayer, self).__init__()
        self.act_fn = act_fn

    def forward(self, Xs):
        self.Xs = []
        x = Xs[0].copy()
        if not isinstance(x, torch.Tensor):
            x = torchify(x)

        self.prod = x.clone()
        x.retain_grad()
        self.Xs.append(x)

        for i in range(1, len(Xs)):
            x = Xs[i]
            if not isinstance(x, torch.Tensor):
                x = torchify(x)

            x.retain_grad()
            self.Xs.append(x)
            self.prod *= x

        self.prod.retain_grad()
        self.Y = self.act_fn(self.prod)
        self.Y.retain_grad()
        return self.Y

    def extract_grads(self, X):
        self.forward(X)
        self.loss = self.Y.sum()
        self.loss.backward()
        grads = {
            "Xs": X,
            "Prod": self.prod.detach().numpy(),
            "Y": self.Y.detach().numpy(),
            "dLdY": self.Y.grad.numpy(),
            "dLdProd": self.prod.grad.numpy(),
        }
        grads.update(
            {"dLdX{}".format(i + 1): xi.grad.numpy() for i, xi in enumerate(self.Xs)}
        )
        return grads


class TorchSkipConnectionIdentity(nn.Module):
    def __init__(self, act_fn, pad1, pad2, params, hparams, momentum=0.9, epsilon=1e-5):
        super(TorchSkipConnectionIdentity, self).__init__()

        self.conv1 = nn.Conv2d(
            hparams["in_ch"],
            hparams["out_ch"],
            hparams["kernel_shape1"],
            padding=pad1,
            stride=hparams["stride1"],
            bias=True,
        )

        self.act_fn = act_fn

        self.batchnorm1 = nn.BatchNorm2d(
            num_features=hparams["out_ch"],
            momentum=1 - momentum,
            eps=epsilon,
            affine=True,
        )

        self.conv2 = nn.Conv2d(
            hparams["out_ch"],
            hparams["out_ch"],
            hparams["kernel_shape2"],
            padding=pad2,
            stride=hparams["stride2"],
            bias=True,
        )

        self.batchnorm2 = nn.BatchNorm2d(
            num_features=hparams["out_ch"],
            momentum=1 - momentum,
            eps=epsilon,
            affine=True,
        )

        orig, W_swap = [0, 1, 2, 3], [-2, -1, -3, -4]
        # (f[0], f[1], n_in, n_out) -> (n_out, n_in, f[0], f[1])
        W = params["components"]["conv1"]["W"]
        b = params["components"]["conv1"]["b"]
        W = np.moveaxis(W, orig, W_swap)
        assert self.conv1.weight.shape == W.shape
        assert self.conv1.bias.shape == b.flatten().shape
        self.conv1.weight = nn.Parameter(torch.FloatTensor(W))
        self.conv1.bias = nn.Parameter(torch.FloatTensor(b.flatten()))

        scaler = params["components"]["batchnorm1"]["scaler"]
        intercept = params["components"]["batchnorm1"]["intercept"]
        self.batchnorm1.weight = nn.Parameter(torch.FloatTensor(scaler))
        self.batchnorm1.bias = nn.Parameter(torch.FloatTensor(intercept))

        # (f[0], f[1], n_in, n_out) -> (n_out, n_in, f[0], f[1])
        W = params["components"]["conv2"]["W"]
        b = params["components"]["conv2"]["b"]
        W = np.moveaxis(W, orig, W_swap)
        assert self.conv2.weight.shape == W.shape
        assert self.conv2.bias.shape == b.flatten().shape
        self.conv2.weight = nn.Parameter(torch.FloatTensor(W))
        self.conv2.bias = nn.Parameter(torch.FloatTensor(b.flatten()))

        scaler = params["components"]["batchnorm2"]["scaler"]
        intercept = params["components"]["batchnorm2"]["intercept"]
        self.batchnorm2.weight = nn.Parameter(torch.FloatTensor(scaler))
        self.batchnorm2.bias = nn.Parameter(torch.FloatTensor(intercept))

    def forward(self, X):
        if not isinstance(X, torch.Tensor):
            # (N, H, W, C) -> (N, C, H, W)
            X = np.moveaxis(X, [0, 1, 2, 3], [0, -2, -1, -3])
            X = torchify(X)

        self.X = X
        self.X.retain_grad()

        self.conv1_out = self.conv1(self.X)
        self.conv1_out.retain_grad()

        self.act_fn1_out = self.act_fn(self.conv1_out)
        self.act_fn1_out.retain_grad()

        self.batchnorm1_out = self.batchnorm1(self.act_fn1_out)
        self.batchnorm1_out.retain_grad()

        self.conv2_out = self.conv2(self.batchnorm1_out)
        self.conv2_out.retain_grad()

        self.batchnorm2_out = self.batchnorm2(self.conv2_out)
        self.batchnorm2_out.retain_grad()

        self.layer3_in = self.batchnorm2_out + self.X
        self.layer3_in.retain_grad()

        self.Y = self.act_fn(self.layer3_in)
        self.Y.retain_grad()

    def extract_grads(self, X):
        self.forward(X)
        self.loss = self.Y.sum()
        self.loss.backward()

        orig, X_swap, W_swap = [0, 1, 2, 3], [0, -1, -3, -2], [-1, -2, -4, -3]
        grads = {
            # layer parameters
            "conv1_W": np.moveaxis(self.conv1.weight.detach().numpy(), orig, W_swap),
            "conv1_b": self.conv1.bias.detach().numpy().reshape(1, 1, 1, -1),
            "bn1_intercept": self.batchnorm1.bias.detach().numpy(),
            "bn1_scaler": self.batchnorm1.weight.detach().numpy(),
            "bn1_running_mean": self.batchnorm1.running_mean.detach().numpy(),
            "bn1_running_var": self.batchnorm1.running_var.detach().numpy(),
            "conv2_W": np.moveaxis(self.conv2.weight.detach().numpy(), orig, W_swap),
            "conv2_b": self.conv2.bias.detach().numpy().reshape(1, 1, 1, -1),
            "bn2_intercept": self.batchnorm2.bias.detach().numpy(),
            "bn2_scaler": self.batchnorm2.weight.detach().numpy(),
            "bn2_running_mean": self.batchnorm2.running_mean.detach().numpy(),
            "bn2_running_var": self.batchnorm2.running_var.detach().numpy(),
            # layer inputs/outputs (forward step)
            "X": np.moveaxis(self.X.detach().numpy(), orig, X_swap),
            "conv1_out": np.moveaxis(self.conv1_out.detach().numpy(), orig, X_swap),
            "act1_out": np.moveaxis(self.act_fn1_out.detach().numpy(), orig, X_swap),
            "bn1_out": np.moveaxis(self.batchnorm1_out.detach().numpy(), orig, X_swap),
            "conv2_out": np.moveaxis(self.conv2_out.detach().numpy(), orig, X_swap),
            "bn2_out": np.moveaxis(self.batchnorm2_out.detach().numpy(), orig, X_swap),
            "add_out": np.moveaxis(self.layer3_in.detach().numpy(), orig, X_swap),
            "Y": np.moveaxis(self.Y.detach().numpy(), orig, X_swap),
            # layer gradients (backward step)
            "dLdY": np.moveaxis(self.Y.grad.numpy(), orig, X_swap),
            "dLdAdd": np.moveaxis(self.layer3_in.grad.numpy(), orig, X_swap),
            "dLdBn2_out": np.moveaxis(self.batchnorm2_out.grad.numpy(), orig, X_swap),
            "dLdConv2_out": np.moveaxis(self.conv2_out.grad.numpy(), orig, X_swap),
            "dLdBn1_out": np.moveaxis(self.batchnorm1_out.grad.numpy(), orig, X_swap),
            "dLdActFn1_out": np.moveaxis(self.act_fn1_out.grad.numpy(), orig, X_swap),
            "dLdConv1_out": np.moveaxis(self.act_fn1_out.grad.numpy(), orig, X_swap),
            "dLdX": np.moveaxis(self.X.grad.numpy(), orig, X_swap),
            # layer parameter gradients (backward step)
            "dLdBn2_intercept": self.batchnorm2.bias.grad.numpy(),
            "dLdBn2_scaler": self.batchnorm2.weight.grad.numpy(),
            "dLdConv2_W": np.moveaxis(self.conv2.weight.grad.numpy(), orig, W_swap),
            "dLdConv2_b": self.conv2.bias.grad.numpy().reshape(1, 1, 1, -1),
            "dLdBn1_intercept": self.batchnorm1.bias.grad.numpy(),
            "dLdBn1_scaler": self.batchnorm1.weight.grad.numpy(),
            "dLdConv1_W": np.moveaxis(self.conv1.weight.grad.numpy(), orig, W_swap),
            "dLdConv1_b": self.conv1.bias.grad.numpy().reshape(1, 1, 1, -1),
        }
        return grads


class TorchCausalConv1d(torch.nn.Conv1d):
    """https://github.com/pytorch/pytorch/issues/1333

    NB: this is only ensures that the convolution out length is the same as
    the input length IFF stride = 1. Otherwise, in/out lengths will differ.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        groups=1,
        bias=True,
    ):
        self.__padding = (kernel_size - 1) * dilation

        super(TorchCausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.__padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, input):
        result = super(TorchCausalConv1d, self).forward(input)
        if self.__padding != 0:
            return result[:, :, : -self.__padding]
        return result


class TorchWavenetModule(nn.Module):
    def __init__(self, params, hparams, conv_1x1_pad):
        super(TorchWavenetModule, self).__init__()
        self.conv_dilation = TorchCausalConv1d(
            in_channels=hparams["components"]["conv_dilation"]["in_ch"],
            out_channels=hparams["components"]["conv_dilation"]["out_ch"],
            kernel_size=hparams["components"]["conv_dilation"]["kernel_width"],
            stride=hparams["components"]["conv_dilation"]["stride"],
            dilation=hparams["components"]["conv_dilation"]["dilation"] + 1,
            bias=True,
        )

        self.conv_1x1 = nn.Conv1d(
            in_channels=hparams["components"]["conv_1x1"]["in_ch"],
            out_channels=hparams["components"]["conv_1x1"]["out_ch"],
            kernel_size=hparams["components"]["conv_1x1"]["kernel_width"],
            stride=hparams["components"]["conv_1x1"]["stride"],
            padding=conv_1x1_pad,
            dilation=hparams["components"]["conv_1x1"]["dilation"] + 1,
            bias=True,
        )

        W = params["components"]["conv_dilation"]["W"]
        b = params["components"]["conv_dilation"]["b"]
        # (f[0], n_in, n_out) -> (n_out, n_in, f[0])
        W = np.moveaxis(W, [0, 1, 2], [-1, -2, -3])
        self.conv_dilation.weight = nn.Parameter(torch.FloatTensor(W))
        self.conv_dilation.bias = nn.Parameter(torch.FloatTensor(b.flatten()))
        assert self.conv_dilation.weight.shape == W.shape
        assert self.conv_dilation.bias.shape == b.flatten().shape

        W = params["components"]["conv_1x1"]["W"]
        b = params["components"]["conv_1x1"]["b"]
        # (f[0], n_in, n_out) -> (n_out, n_in, f[0])
        W = np.moveaxis(W, [0, 1, 2], [-1, -2, -3])
        self.conv_1x1.weight = nn.Parameter(torch.FloatTensor(W))
        self.conv_1x1.bias = nn.Parameter(torch.FloatTensor(b.flatten()))
        assert self.conv_1x1.weight.shape == W.shape
        assert self.conv_1x1.bias.shape == b.flatten().shape

    def forward(self, X_main, X_skip):
        # (N, W, C) -> (N, C, W)
        self.X_main = np.moveaxis(X_main, [0, 1, 2], [0, -1, -2])
        self.X_main = torchify(self.X_main)
        self.X_main.retain_grad()

        self.conv_dilation_out = self.conv_dilation(self.X_main)
        self.conv_dilation_out.retain_grad()

        self.tanh_out = F.tanh(self.conv_dilation_out)
        self.sigm_out = F.sigmoid(self.conv_dilation_out)

        self.tanh_out.retain_grad()
        self.sigm_out.retain_grad()

        self.multiply_gate_out = self.tanh_out * self.sigm_out
        self.multiply_gate_out.retain_grad()

        self.conv_1x1_out = self.conv_1x1(self.multiply_gate_out)
        self.conv_1x1_out.retain_grad()

        self.X_skip = torch.zeros_like(self.conv_1x1_out)
        if X_skip is not None:
            self.X_skip = torchify(np.moveaxis(X_skip, [0, 1, 2], [0, -1, -2]))
        self.X_skip.retain_grad()

        self.Y_skip = self.X_skip + self.conv_1x1_out
        self.Y_main = self.X_main + self.conv_1x1_out

        self.Y_skip.retain_grad()
        self.Y_main.retain_grad()

    def extract_grads(self, X_main, X_skip):
        self.forward(X_main, X_skip)
        self.loss = (self.Y_skip + self.Y_main).sum()
        self.loss.backward()

        # W (theirs): (n_out, n_in, f[0]) -> W (mine): (f[0], n_in, n_out)
        # X (theirs): (N, C, W)              -> X (mine): (N, W, C)
        # Y (theirs): (N, C, W)              -> Y (mine): (N, W, C)
        orig, X_swap, W_swap = [0, 1, 2], [0, -1, -2], [-1, -2, -3]
        grads = {
            "X_main": np.moveaxis(self.X_main.detach().numpy(), orig, X_swap),
            "X_skip": np.moveaxis(self.X_skip.detach().numpy(), orig, X_swap),
            "conv_dilation_W": np.moveaxis(
                self.conv_dilation.weight.detach().numpy(), orig, W_swap
            ),
            "conv_dilation_b": self.conv_dilation.bias.detach()
            .numpy()
            .reshape(1, 1, -1),
            "conv_1x1_W": np.moveaxis(
                self.conv_1x1.weight.detach().numpy(), orig, W_swap
            ),
            "conv_1x1_b": self.conv_1x1.bias.detach().numpy().reshape(1, 1, -1),
            "conv_dilation_out": np.moveaxis(
                self.conv_dilation_out.detach().numpy(), orig, X_swap
            ),
            "tanh_out": np.moveaxis(self.tanh_out.detach().numpy(), orig, X_swap),
            "sigm_out": np.moveaxis(self.sigm_out.detach().numpy(), orig, X_swap),
            "multiply_gate_out": np.moveaxis(
                self.multiply_gate_out.detach().numpy(), orig, X_swap
            ),
            "conv_1x1_out": np.moveaxis(
                self.conv_1x1_out.detach().numpy(), orig, X_swap
            ),
            "Y_main": np.moveaxis(self.Y_main.detach().numpy(), orig, X_swap),
            "Y_skip": np.moveaxis(self.Y_skip.detach().numpy(), orig, X_swap),
            "dLdY_skip": np.moveaxis(self.Y_skip.grad.numpy(), orig, X_swap),
            "dLdY_main": np.moveaxis(self.Y_main.grad.numpy(), orig, X_swap),
            "dLdConv_1x1_out": np.moveaxis(
                self.conv_1x1_out.grad.numpy(), orig, X_swap
            ),
            "dLdConv_1x1_W": np.moveaxis(
                self.conv_1x1.weight.grad.numpy(), orig, W_swap
            ),
            "dLdConv_1x1_b": self.conv_1x1.bias.grad.numpy().reshape(1, 1, -1),
            "dLdMultiply_out": np.moveaxis(
                self.multiply_gate_out.grad.numpy(), orig, X_swap
            ),
            "dLdTanh_out": np.moveaxis(self.tanh_out.grad.numpy(), orig, X_swap),
            "dLdSigm_out": np.moveaxis(self.sigm_out.grad.numpy(), orig, X_swap),
            "dLdConv_dilation_out": np.moveaxis(
                self.conv_dilation_out.grad.numpy(), orig, X_swap
            ),
            "dLdConv_dilation_W": np.moveaxis(
                self.conv_dilation.weight.grad.numpy(), orig, W_swap
            ),
            "dLdConv_dilation_b": self.conv_dilation.bias.grad.numpy().reshape(
                1, 1, -1
            ),
            "dLdX_main": np.moveaxis(self.X_main.grad.numpy(), orig, X_swap),
            "dLdX_skip": np.moveaxis(self.X_skip.grad.numpy(), orig, X_swap),
        }

        return grads


class TorchSkipConnectionConv(nn.Module):
    def __init__(
        self, act_fn, pad1, pad2, pad_skip, params, hparams, momentum=0.9, epsilon=1e-5
    ):
        super(TorchSkipConnectionConv, self).__init__()

        self.conv1 = nn.Conv2d(
            hparams["in_ch"],
            hparams["out_ch1"],
            hparams["kernel_shape1"],
            padding=pad1,
            stride=hparams["stride1"],
            bias=True,
        )

        self.act_fn = act_fn

        self.batchnorm1 = nn.BatchNorm2d(
            num_features=hparams["out_ch1"],
            momentum=1 - momentum,
            eps=epsilon,
            affine=True,
        )

        self.conv2 = nn.Conv2d(
            hparams["out_ch1"],
            hparams["out_ch2"],
            hparams["kernel_shape2"],
            padding=pad2,
            stride=hparams["stride2"],
            bias=True,
        )

        self.batchnorm2 = nn.BatchNorm2d(
            num_features=hparams["out_ch2"],
            momentum=1 - momentum,
            eps=epsilon,
            affine=True,
        )

        self.conv_skip = nn.Conv2d(
            hparams["in_ch"],
            hparams["out_ch2"],
            hparams["kernel_shape_skip"],
            padding=pad_skip,
            stride=hparams["stride_skip"],
            bias=True,
        )

        self.batchnorm_skip = nn.BatchNorm2d(
            num_features=hparams["out_ch2"],
            momentum=1 - momentum,
            eps=epsilon,
            affine=True,
        )

        orig, W_swap = [0, 1, 2, 3], [-2, -1, -3, -4]
        # (f[0], f[1], n_in, n_out) -> (n_out, n_in, f[0], f[1])
        W = params["components"]["conv1"]["W"]
        b = params["components"]["conv1"]["b"]
        W = np.moveaxis(W, orig, W_swap)
        assert self.conv1.weight.shape == W.shape
        assert self.conv1.bias.shape == b.flatten().shape
        self.conv1.weight = nn.Parameter(torch.FloatTensor(W))
        self.conv1.bias = nn.Parameter(torch.FloatTensor(b.flatten()))

        scaler = params["components"]["batchnorm1"]["scaler"]
        intercept = params["components"]["batchnorm1"]["intercept"]
        self.batchnorm1.weight = nn.Parameter(torch.FloatTensor(scaler))
        self.batchnorm1.bias = nn.Parameter(torch.FloatTensor(intercept))

        # (f[0], f[1], n_in, n_out) -> (n_out, n_in, f[0], f[1])
        W = params["components"]["conv2"]["W"]
        b = params["components"]["conv2"]["b"]
        W = np.moveaxis(W, orig, W_swap)
        assert self.conv2.weight.shape == W.shape
        assert self.conv2.bias.shape == b.flatten().shape
        self.conv2.weight = nn.Parameter(torch.FloatTensor(W))
        self.conv2.bias = nn.Parameter(torch.FloatTensor(b.flatten()))

        scaler = params["components"]["batchnorm2"]["scaler"]
        intercept = params["components"]["batchnorm2"]["intercept"]
        self.batchnorm2.weight = nn.Parameter(torch.FloatTensor(scaler))
        self.batchnorm2.bias = nn.Parameter(torch.FloatTensor(intercept))

        W = params["components"]["conv_skip"]["W"]
        b = params["components"]["conv_skip"]["b"]
        W = np.moveaxis(W, orig, W_swap)
        assert self.conv_skip.weight.shape == W.shape
        assert self.conv_skip.bias.shape == b.flatten().shape
        self.conv_skip.weight = nn.Parameter(torch.FloatTensor(W))
        self.conv_skip.bias = nn.Parameter(torch.FloatTensor(b.flatten()))

        scaler = params["components"]["batchnorm_skip"]["scaler"]
        intercept = params["components"]["batchnorm_skip"]["intercept"]
        self.batchnorm_skip.weight = nn.Parameter(torch.FloatTensor(scaler))
        self.batchnorm_skip.bias = nn.Parameter(torch.FloatTensor(intercept))

    def forward(self, X):
        if not isinstance(X, torch.Tensor):
            # (N, H, W, C) -> (N, C, H, W)
            X = np.moveaxis(X, [0, 1, 2, 3], [0, -2, -1, -3])
            X = torchify(X)

        self.X = X
        self.X.retain_grad()

        self.conv1_out = self.conv1(self.X)
        self.conv1_out.retain_grad()

        self.act_fn1_out = self.act_fn(self.conv1_out)
        self.act_fn1_out.retain_grad()

        self.batchnorm1_out = self.batchnorm1(self.act_fn1_out)
        self.batchnorm1_out.retain_grad()

        self.conv2_out = self.conv2(self.batchnorm1_out)
        self.conv2_out.retain_grad()

        self.batchnorm2_out = self.batchnorm2(self.conv2_out)
        self.batchnorm2_out.retain_grad()

        self.c_skip_out = self.conv_skip(self.X)
        self.c_skip_out.retain_grad()

        self.bn_skip_out = self.batchnorm_skip(self.c_skip_out)
        self.bn_skip_out.retain_grad()

        self.layer3_in = self.batchnorm2_out + self.bn_skip_out
        self.layer3_in.retain_grad()

        self.Y = self.act_fn(self.layer3_in)
        self.Y.retain_grad()

    def extract_grads(self, X):
        self.forward(X)
        self.loss = self.Y.sum()
        self.loss.backward()

        orig, X_swap, W_swap = [0, 1, 2, 3], [0, -1, -3, -2], [-1, -2, -4, -3]
        grads = {
            # layer parameters
            "conv1_W": np.moveaxis(self.conv1.weight.detach().numpy(), orig, W_swap),
            "conv1_b": self.conv1.bias.detach().numpy().reshape(1, 1, 1, -1),
            "bn1_intercept": self.batchnorm1.bias.detach().numpy(),
            "bn1_scaler": self.batchnorm1.weight.detach().numpy(),
            "bn1_running_mean": self.batchnorm1.running_mean.detach().numpy(),
            "bn1_running_var": self.batchnorm1.running_var.detach().numpy(),
            "conv2_W": np.moveaxis(self.conv2.weight.detach().numpy(), orig, W_swap),
            "conv2_b": self.conv2.bias.detach().numpy().reshape(1, 1, 1, -1),
            "bn2_intercept": self.batchnorm2.bias.detach().numpy(),
            "bn2_scaler": self.batchnorm2.weight.detach().numpy(),
            "bn2_running_mean": self.batchnorm2.running_mean.detach().numpy(),
            "bn2_running_var": self.batchnorm2.running_var.detach().numpy(),
            "conv_skip_W": np.moveaxis(
                self.conv_skip.weight.detach().numpy(), orig, W_swap
            ),
            "conv_skip_b": self.conv_skip.bias.detach().numpy().reshape(1, 1, 1, -1),
            "bn_skip_intercept": self.batchnorm_skip.bias.detach().numpy(),
            "bn_skip_scaler": self.batchnorm_skip.weight.detach().numpy(),
            "bn_skip_running_mean": self.batchnorm_skip.running_mean.detach().numpy(),
            "bn_skip_running_var": self.batchnorm_skip.running_var.detach().numpy(),
            # layer inputs/outputs (forward step)
            "X": np.moveaxis(self.X.detach().numpy(), orig, X_swap),
            "conv1_out": np.moveaxis(self.conv1_out.detach().numpy(), orig, X_swap),
            "act1_out": np.moveaxis(self.act_fn1_out.detach().numpy(), orig, X_swap),
            "bn1_out": np.moveaxis(self.batchnorm1_out.detach().numpy(), orig, X_swap),
            "conv2_out": np.moveaxis(self.conv2_out.detach().numpy(), orig, X_swap),
            "bn2_out": np.moveaxis(self.batchnorm2_out.detach().numpy(), orig, X_swap),
            "conv_skip_out": np.moveaxis(
                self.c_skip_out.detach().numpy(), orig, X_swap
            ),
            "bn_skip_out": np.moveaxis(self.bn_skip_out.detach().numpy(), orig, X_swap),
            "add_out": np.moveaxis(self.layer3_in.detach().numpy(), orig, X_swap),
            "Y": np.moveaxis(self.Y.detach().numpy(), orig, X_swap),
            # layer gradients (backward step)
            "dLdY": np.moveaxis(self.Y.grad.numpy(), orig, X_swap),
            "dLdAdd": np.moveaxis(self.layer3_in.grad.numpy(), orig, X_swap),
            "dLdBnSkip_out": np.moveaxis(self.bn_skip_out.grad.numpy(), orig, X_swap),
            "dLdConvSkip_out": np.moveaxis(self.c_skip_out.grad.numpy(), orig, X_swap),
            "dLdBn2_out": np.moveaxis(self.batchnorm2_out.grad.numpy(), orig, X_swap),
            "dLdConv2_out": np.moveaxis(self.conv2_out.grad.numpy(), orig, X_swap),
            "dLdBn1_out": np.moveaxis(self.batchnorm1_out.grad.numpy(), orig, X_swap),
            "dLdActFn1_out": np.moveaxis(self.act_fn1_out.grad.numpy(), orig, X_swap),
            "dLdConv1_out": np.moveaxis(self.act_fn1_out.grad.numpy(), orig, X_swap),
            "dLdX": np.moveaxis(self.X.grad.numpy(), orig, X_swap),
            # layer parameter gradients (backward step)
            "dLdBnSkip_intercept": self.batchnorm_skip.bias.grad.numpy(),
            "dLdBnSkip_scaler": self.batchnorm_skip.weight.grad.numpy(),
            "dLdConvSkip_W": np.moveaxis(
                self.conv_skip.weight.grad.numpy(), orig, W_swap
            ),
            "dLdConvSkip_b": self.conv_skip.bias.grad.numpy().reshape(1, 1, 1, -1),
            "dLdBn2_intercept": self.batchnorm2.bias.grad.numpy(),
            "dLdBn2_scaler": self.batchnorm2.weight.grad.numpy(),
            "dLdConv2_W": np.moveaxis(self.conv2.weight.grad.numpy(), orig, W_swap),
            "dLdConv2_b": self.conv2.bias.grad.numpy().reshape(1, 1, 1, -1),
            "dLdBn1_intercept": self.batchnorm1.bias.grad.numpy(),
            "dLdBn1_scaler": self.batchnorm1.weight.grad.numpy(),
            "dLdConv1_W": np.moveaxis(self.conv1.weight.grad.numpy(), orig, W_swap),
            "dLdConv1_b": self.conv1.bias.grad.numpy().reshape(1, 1, 1, -1),
        }
        return grads


class TorchBidirectionalLSTM(nn.Module):
    def __init__(self, n_in, n_out, params, **kwargs):
        super(TorchBidirectionalLSTM, self).__init__()

        self.layer1 = nn.LSTM(
            input_size=n_in,
            hidden_size=n_out,
            num_layers=1,
            bidirectional=True,
            bias=True,
        )

        Wiu = params["components"]["cell_fwd"]["Wu"][n_out:, :].T
        Wif = params["components"]["cell_fwd"]["Wf"][n_out:, :].T
        Wic = params["components"]["cell_fwd"]["Wc"][n_out:, :].T
        Wio = params["components"]["cell_fwd"]["Wo"][n_out:, :].T
        W_ih_f = np.vstack([Wiu, Wif, Wic, Wio])

        Whu = params["components"]["cell_fwd"]["Wu"][:n_out, :].T
        Whf = params["components"]["cell_fwd"]["Wf"][:n_out, :].T
        Whc = params["components"]["cell_fwd"]["Wc"][:n_out, :].T
        Who = params["components"]["cell_fwd"]["Wo"][:n_out, :].T
        W_hh_f = np.vstack([Whu, Whf, Whc, Who])

        assert self.layer1.weight_ih_l0.shape == W_ih_f.shape
        assert self.layer1.weight_hh_l0.shape == W_hh_f.shape

        self.layer1.weight_ih_l0 = nn.Parameter(torch.FloatTensor(W_ih_f))
        self.layer1.weight_hh_l0 = nn.Parameter(torch.FloatTensor(W_hh_f))

        Wiu = params["components"]["cell_bwd"]["Wu"][n_out:, :].T
        Wif = params["components"]["cell_bwd"]["Wf"][n_out:, :].T
        Wic = params["components"]["cell_bwd"]["Wc"][n_out:, :].T
        Wio = params["components"]["cell_bwd"]["Wo"][n_out:, :].T
        W_ih_b = np.vstack([Wiu, Wif, Wic, Wio])

        Whu = params["components"]["cell_bwd"]["Wu"][:n_out, :].T
        Whf = params["components"]["cell_bwd"]["Wf"][:n_out, :].T
        Whc = params["components"]["cell_bwd"]["Wc"][:n_out, :].T
        Who = params["components"]["cell_bwd"]["Wo"][:n_out, :].T
        W_hh_b = np.vstack([Whu, Whf, Whc, Who])

        assert self.layer1.weight_ih_l0_reverse.shape == W_ih_b.shape
        assert self.layer1.weight_hh_l0_reverse.shape == W_hh_b.shape

        self.layer1.weight_ih_l0_reverse = nn.Parameter(torch.FloatTensor(W_ih_b))
        self.layer1.weight_hh_l0_reverse = nn.Parameter(torch.FloatTensor(W_hh_b))

        b_f = np.concatenate(
            [
                params["components"]["cell_fwd"]["bu"],
                params["components"]["cell_fwd"]["bf"],
                params["components"]["cell_fwd"]["bc"],
                params["components"]["cell_fwd"]["bo"],
            ],
            axis=-1,
        ).flatten()

        assert self.layer1.bias_ih_l0.shape == b_f.shape
        assert self.layer1.bias_hh_l0.shape == b_f.shape

        self.layer1.bias_ih_l0 = nn.Parameter(torch.FloatTensor(b_f))
        self.layer1.bias_hh_l0 = nn.Parameter(torch.FloatTensor(b_f))

        b_b = np.concatenate(
            [
                params["components"]["cell_bwd"]["bu"],
                params["components"]["cell_bwd"]["bf"],
                params["components"]["cell_bwd"]["bc"],
                params["components"]["cell_bwd"]["bo"],
            ],
            axis=-1,
        ).flatten()

        assert self.layer1.bias_ih_l0_reverse.shape == b_b.shape
        assert self.layer1.bias_hh_l0_reverse.shape == b_b.shape

        self.layer1.bias_ih_l0_reverse = nn.Parameter(torch.FloatTensor(b_b))
        self.layer1.bias_hh_l0_reverse = nn.Parameter(torch.FloatTensor(b_b))

    def forward(self, X):
        # (batch, input_size, seq_len) -> (seq_len, batch, input_size)
        self.X = np.moveaxis(X, [0, 1, 2], [-2, -1, -3])

        if not isinstance(self.X, torch.Tensor):
            self.X = torchify(self.X)

        self.X.retain_grad()

        # initial hidden state is 0
        n_ex, n_in, n_timesteps = self.X.shape
        n_out, n_out = self.layer1.weight_hh_l0.shape

        # forward pass
        self.A, (At, Ct) = self.layer1(self.X)
        self.A.retain_grad()
        return self.A

    def extract_grads(self, X):
        self.forward(X)
        self.loss = self.A.sum()
        self.loss.backward()

        # forward
        w_ii, w_if, w_ic, w_io = self.layer1.weight_ih_l0.chunk(4, 0)
        w_hi, w_hf, w_hc, w_ho = self.layer1.weight_hh_l0.chunk(4, 0)
        bu_f, bf_f, bc_f, bo_f = self.layer1.bias_ih_l0.chunk(4, 0)

        Wu_f = torch.cat([torch.t(w_hi), torch.t(w_ii)], dim=0)
        Wf_f = torch.cat([torch.t(w_hf), torch.t(w_if)], dim=0)
        Wc_f = torch.cat([torch.t(w_hc), torch.t(w_ic)], dim=0)
        Wo_f = torch.cat([torch.t(w_ho), torch.t(w_io)], dim=0)

        dw_ii, dw_if, dw_ic, dw_io = self.layer1.weight_ih_l0.grad.chunk(4, 0)
        dw_hi, dw_hf, dw_hc, dw_ho = self.layer1.weight_hh_l0.grad.chunk(4, 0)
        dbu_f, dbf_f, dbc_f, dbo_f = self.layer1.bias_ih_l0.grad.chunk(4, 0)

        dWu_f = torch.cat([torch.t(dw_hi), torch.t(dw_ii)], dim=0)
        dWf_f = torch.cat([torch.t(dw_hf), torch.t(dw_if)], dim=0)
        dWc_f = torch.cat([torch.t(dw_hc), torch.t(dw_ic)], dim=0)
        dWo_f = torch.cat([torch.t(dw_ho), torch.t(dw_io)], dim=0)

        # backward
        w_ii, w_if, w_ic, w_io = self.layer1.weight_ih_l0_reverse.chunk(4, 0)
        w_hi, w_hf, w_hc, w_ho = self.layer1.weight_hh_l0_reverse.chunk(4, 0)
        bu_b, bf_b, bc_b, bo_b = self.layer1.bias_ih_l0_reverse.chunk(4, 0)

        Wu_b = torch.cat([torch.t(w_hi), torch.t(w_ii)], dim=0)
        Wf_b = torch.cat([torch.t(w_hf), torch.t(w_if)], dim=0)
        Wc_b = torch.cat([torch.t(w_hc), torch.t(w_ic)], dim=0)
        Wo_b = torch.cat([torch.t(w_ho), torch.t(w_io)], dim=0)

        dw_ii, dw_if, dw_ic, dw_io = self.layer1.weight_ih_l0_reverse.grad.chunk(4, 0)
        dw_hi, dw_hf, dw_hc, dw_ho = self.layer1.weight_hh_l0_reverse.grad.chunk(4, 0)
        dbu_b, dbf_b, dbc_b, dbo_b = self.layer1.bias_ih_l0_reverse.grad.chunk(4, 0)

        dWu_b = torch.cat([torch.t(dw_hi), torch.t(dw_ii)], dim=0)
        dWf_b = torch.cat([torch.t(dw_hf), torch.t(dw_if)], dim=0)
        dWc_b = torch.cat([torch.t(dw_hc), torch.t(dw_ic)], dim=0)
        dWo_b = torch.cat([torch.t(dw_ho), torch.t(dw_io)], dim=0)

        orig, X_swap = [0, 1, 2], [-1, -3, -2]
        grads = {
            "X": np.moveaxis(self.X.detach().numpy(), orig, X_swap),
            "Wu_f": Wu_f.detach().numpy(),
            "Wf_f": Wf_f.detach().numpy(),
            "Wc_f": Wc_f.detach().numpy(),
            "Wo_f": Wo_f.detach().numpy(),
            "bu_f": bu_f.detach().numpy().reshape(-1, 1),
            "bf_f": bf_f.detach().numpy().reshape(-1, 1),
            "bc_f": bc_f.detach().numpy().reshape(-1, 1),
            "bo_f": bo_f.detach().numpy().reshape(-1, 1),
            "Wu_b": Wu_b.detach().numpy(),
            "Wf_b": Wf_b.detach().numpy(),
            "Wc_b": Wc_b.detach().numpy(),
            "Wo_b": Wo_b.detach().numpy(),
            "bu_b": bu_b.detach().numpy().reshape(-1, 1),
            "bf_b": bf_b.detach().numpy().reshape(-1, 1),
            "bc_b": bc_b.detach().numpy().reshape(-1, 1),
            "bo_b": bo_b.detach().numpy().reshape(-1, 1),
            "y": np.moveaxis(self.A.detach().numpy(), orig, X_swap),
            "dLdA": self.A.grad.numpy(),
            "dLdWu_f": dWu_f.numpy(),
            "dLdWf_f": dWf_f.numpy(),
            "dLdWc_f": dWc_f.numpy(),
            "dLdWo_f": dWo_f.numpy(),
            "dLdBu_f": dbu_f.numpy().reshape(-1, 1),
            "dLdBf_f": dbf_f.numpy().reshape(-1, 1),
            "dLdBc_f": dbc_f.numpy().reshape(-1, 1),
            "dLdBo_f": dbo_f.numpy().reshape(-1, 1),
            "dLdWu_b": dWu_b.numpy(),
            "dLdWf_b": dWf_b.numpy(),
            "dLdWc_b": dWc_b.numpy(),
            "dLdWo_b": dWo_b.numpy(),
            "dLdBu_b": dbu_b.numpy().reshape(-1, 1),
            "dLdBf_b": dbf_b.numpy().reshape(-1, 1),
            "dLdBc_b": dbc_b.numpy().reshape(-1, 1),
            "dLdBo_b": dbo_b.numpy().reshape(-1, 1),
            "dLdX": np.moveaxis(self.X.grad.numpy(), orig, X_swap),
        }
        return grads


class TorchPool2DLayer(nn.Module):
    def __init__(self, in_channels, hparams, **kwargs):
        super(TorchPool2DLayer, self).__init__()

        if hparams["mode"] == "max":
            self.layer1 = nn.MaxPool2d(
                kernel_size=hparams["kernel_shape"],
                padding=hparams["pad"],
                stride=hparams["stride"],
            )
        elif hparams["mode"] == "average":
            self.layer1 = nn.AvgPool2d(
                kernel_size=hparams["kernel_shape"],
                padding=hparams["pad"],
                stride=hparams["stride"],
            )

    def forward(self, X):
        # (N, H, W, C) -> (N, C, H, W)
        self.X = np.moveaxis(X, [0, 1, 2, 3], [0, -2, -1, -3])
        if not isinstance(self.X, torch.Tensor):
            self.X = torchify(self.X)

        self.X.retain_grad()
        self.Y = self.layer1(self.X)
        self.Y.retain_grad()
        return self.Y

    def extract_grads(self, X):
        self.forward(X)
        self.loss = self.Y.sum()
        self.loss.backward()

        # W (theirs): (n_out, n_in, f[0], f[1]) -> W (mine): (f[0], f[1], n_in, n_out)
        # X (theirs): (N, C, H, W)              -> X (mine): (N, H, W, C)
        # Y (theirs): (N, C, H, W)              -> Y (mine): (N, H, W, C)
        orig, X_swap = [0, 1, 2, 3], [0, -1, -3, -2]
        grads = {
            "X": np.moveaxis(self.X.detach().numpy(), orig, X_swap),
            "y": np.moveaxis(self.Y.detach().numpy(), orig, X_swap),
            "dLdY": np.moveaxis(self.Y.grad.numpy(), orig, X_swap),
            "dLdX": np.moveaxis(self.X.grad.numpy(), orig, X_swap),
        }
        return grads


class TorchConv2DLayer(nn.Module):
    def __init__(self, in_channels, out_channels, act_fn, params, hparams, **kwargs):
        super(TorchConv2DLayer, self).__init__()

        W = params["W"]
        b = params["b"]
        self.act_fn = act_fn

        self.layer1 = nn.Conv2d(
            in_channels,
            out_channels,
            hparams["kernel_shape"],
            padding=hparams["pad"],
            stride=hparams["stride"],
            dilation=hparams["dilation"] + 1,
            bias=True,
        )

        # (f[0], f[1], n_in, n_out) -> (n_out, n_in, f[0], f[1])
        W = np.moveaxis(W, [0, 1, 2, 3], [-2, -1, -3, -4])
        assert self.layer1.weight.shape == W.shape
        assert self.layer1.bias.shape == b.flatten().shape

        self.layer1.weight = nn.Parameter(torch.FloatTensor(W))
        self.layer1.bias = nn.Parameter(torch.FloatTensor(b.flatten()))

    def forward(self, X):
        # (N, H, W, C) -> (N, C, H, W)
        self.X = np.moveaxis(X, [0, 1, 2, 3], [0, -2, -1, -3])
        if not isinstance(self.X, torch.Tensor):
            self.X = torchify(self.X)

        self.X.retain_grad()

        self.Z = self.layer1(self.X)
        self.Z.retain_grad()

        self.Y = self.act_fn(self.Z)
        self.Y.retain_grad()
        return self.Y

    def extract_grads(self, X):
        self.forward(X)
        self.loss = self.Y.sum()
        self.loss.backward()

        # W (theirs): (n_out, n_in, f[0], f[1]) -> W (mine): (f[0], f[1], n_in, n_out)
        # X (theirs): (N, C, H, W)              -> X (mine): (N, H, W, C)
        # Y (theirs): (N, C, H, W)              -> Y (mine): (N, H, W, C)
        orig, X_swap, W_swap = [0, 1, 2, 3], [0, -1, -3, -2], [-1, -2, -4, -3]
        grads = {
            "X": np.moveaxis(self.X.detach().numpy(), orig, X_swap),
            "W": np.moveaxis(self.layer1.weight.detach().numpy(), orig, W_swap),
            "b": self.layer1.bias.detach().numpy().reshape(1, 1, 1, -1),
            "y": np.moveaxis(self.Y.detach().numpy(), orig, X_swap),
            "dLdY": np.moveaxis(self.Y.grad.numpy(), orig, X_swap),
            "dLdZ": np.moveaxis(self.Z.grad.numpy(), orig, X_swap),
            "dLdW": np.moveaxis(self.layer1.weight.grad.numpy(), orig, W_swap),
            "dLdB": self.layer1.bias.grad.numpy().reshape(1, 1, 1, -1),
            "dLdX": np.moveaxis(self.X.grad.numpy(), orig, X_swap),
        }
        return grads


class TorchConv1DLayer(nn.Module):
    def __init__(self, in_channels, out_channels, act_fn, params, hparams, **kwargs):
        super(TorchConv1DLayer, self).__init__()

        W = params["W"]
        b = params["b"]
        self.act_fn = act_fn

        self.layer1 = nn.Conv1d(
            in_channels,
            out_channels,
            hparams["kernel_width"],
            padding=hparams["pad"],
            stride=hparams["stride"],
            dilation=hparams["dilation"] + 1,
            bias=True,
        )

        # (f[0], n_in, n_out) -> (n_out, n_in, f[0])
        W = np.moveaxis(W, [0, 1, 2], [-1, -2, -3])
        assert self.layer1.weight.shape == W.shape
        assert self.layer1.bias.shape == b.flatten().shape

        self.layer1.weight = nn.Parameter(torch.FloatTensor(W))
        self.layer1.bias = nn.Parameter(torch.FloatTensor(b.flatten()))

    def forward(self, X):
        # (N, W, C) -> (N, C, W)
        self.X = np.moveaxis(X, [0, 1, 2], [0, -1, -2])
        if not isinstance(self.X, torch.Tensor):
            self.X = torchify(self.X)

        self.X.retain_grad()

        self.Z = self.layer1(self.X)
        self.Z.retain_grad()

        self.Y = self.act_fn(self.Z)
        self.Y.retain_grad()
        return self.Y

    def extract_grads(self, X):
        self.forward(X)
        self.loss = self.Y.sum()
        self.loss.backward()

        # W (theirs): (n_out, n_in, f[0]) -> W (mine): (f[0], n_in, n_out)
        # X (theirs): (N, C, W)              -> X (mine): (N, W, C)
        # Y (theirs): (N, C, W)              -> Y (mine): (N, W, C)
        orig, X_swap, W_swap = [0, 1, 2], [0, -1, -2], [-1, -2, -3]
        grads = {
            "X": np.moveaxis(self.X.detach().numpy(), orig, X_swap),
            "W": np.moveaxis(self.layer1.weight.detach().numpy(), orig, W_swap),
            "b": self.layer1.bias.detach().numpy().reshape(1, 1, -1),
            "y": np.moveaxis(self.Y.detach().numpy(), orig, X_swap),
            "dLdY": np.moveaxis(self.Y.grad.numpy(), orig, X_swap),
            "dLdZ": np.moveaxis(self.Z.grad.numpy(), orig, X_swap),
            "dLdW": np.moveaxis(self.layer1.weight.grad.numpy(), orig, W_swap),
            "dLdB": self.layer1.bias.grad.numpy().reshape(1, 1, -1),
            "dLdX": np.moveaxis(self.X.grad.numpy(), orig, X_swap),
        }
        return grads


class TorchDeconv2DLayer(nn.Module):
    def __init__(self, in_channels, out_channels, act_fn, params, hparams, **kwargs):
        super(TorchDeconv2DLayer, self).__init__()

        W = params["W"]
        b = params["b"]
        self.act_fn = act_fn

        self.layer1 = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            hparams["kernel_shape"],
            padding=hparams["pad"],
            stride=hparams["stride"],
            dilation=1,
            bias=True,
        )

        # (f[0], f[1], n_in, n_out) -> (n_in, n_out, f[0], f[1])
        W = np.moveaxis(W, [0, 1, 2, 3], [-2, -1, -4, -3])
        assert self.layer1.weight.shape == W.shape
        assert self.layer1.bias.shape == b.flatten().shape

        self.layer1.weight = nn.Parameter(torch.FloatTensor(W))
        self.layer1.bias = nn.Parameter(torch.FloatTensor(b.flatten()))

    def forward(self, X):
        # (N, H, W, C) -> (N, C, H, W)
        self.X = np.moveaxis(X, [0, 1, 2, 3], [0, -2, -1, -3])
        if not isinstance(self.X, torch.Tensor):
            self.X = torchify(self.X)

        self.X.retain_grad()

        self.Z = self.layer1(self.X)
        self.Z.retain_grad()

        self.Y = self.act_fn(self.Z)
        self.Y.retain_grad()
        return self.Y

    def extract_grads(self, X):
        self.forward(X)
        self.loss = self.Y.sum()
        self.loss.backward()

        # W (theirs): (n_in, n_out, f[0], f[1]) -> W (mine): (f[0], f[1], n_in, n_out)
        # X (theirs): (N, C, H, W)              -> X (mine): (N, H, W, C)
        # Y (theirs): (N, C, H, W)              -> Y (mine): (N, H, W, C)
        orig, X_swap, W_swap = [0, 1, 2, 3], [0, -1, -3, -2], [-2, -1, -4, -3]
        grads = {
            "X": np.moveaxis(self.X.detach().numpy(), orig, X_swap),
            "W": np.moveaxis(self.layer1.weight.detach().numpy(), orig, W_swap),
            "b": self.layer1.bias.detach().numpy().reshape(1, 1, 1, -1),
            "y": np.moveaxis(self.Y.detach().numpy(), orig, X_swap),
            "dLdY": np.moveaxis(self.Y.grad.numpy(), orig, X_swap),
            "dLdZ": np.moveaxis(self.Z.grad.numpy(), orig, X_swap),
            "dLdW": np.moveaxis(self.layer1.weight.grad.numpy(), orig, W_swap),
            "dLdB": self.layer1.bias.grad.numpy().reshape(1, 1, 1, -1),
            "dLdX": np.moveaxis(self.X.grad.numpy(), orig, X_swap),
        }
        return grads


class TorchLSTMCell(nn.Module):
    def __init__(self, n_in, n_out, params, **kwargs):
        super(TorchLSTMCell, self).__init__()

        Wiu = params["Wu"][n_out:, :].T
        Wif = params["Wf"][n_out:, :].T
        Wic = params["Wc"][n_out:, :].T
        Wio = params["Wo"][n_out:, :].T
        W_ih = np.vstack([Wiu, Wif, Wic, Wio])

        Whu = params["Wu"][:n_out, :].T
        Whf = params["Wf"][:n_out, :].T
        Whc = params["Wc"][:n_out, :].T
        Who = params["Wo"][:n_out, :].T
        W_hh = np.vstack([Whu, Whf, Whc, Who])

        self.layer1 = nn.LSTMCell(input_size=n_in, hidden_size=n_out, bias=True)
        assert self.layer1.weight_ih.shape == W_ih.shape
        assert self.layer1.weight_hh.shape == W_hh.shape
        self.layer1.weight_ih = nn.Parameter(torch.FloatTensor(W_ih))
        self.layer1.weight_hh = nn.Parameter(torch.FloatTensor(W_hh))

        b = np.concatenate(
            [params["bu"], params["bf"], params["bc"], params["bo"]], axis=-1
        ).flatten()
        assert self.layer1.bias_ih.shape == b.shape
        assert self.layer1.bias_hh.shape == b.shape
        self.layer1.bias_ih = nn.Parameter(torch.FloatTensor(b))
        self.layer1.bias_hh = nn.Parameter(torch.FloatTensor(b))

    def forward(self, X):
        self.X = X
        if not isinstance(self.X, torch.Tensor):
            self.X = torchify(self.X)

        self.X.retain_grad()

        # initial hidden state is 0
        n_ex, n_in, n_timesteps = self.X.shape
        n_out, n_out = self.layer1.weight_hh.shape

        # initialize hidden states
        a0 = torchify(np.zeros((n_ex, n_out)))
        c0 = torchify(np.zeros((n_ex, n_out)))
        a0.retain_grad()
        c0.retain_grad()

        # forward pass
        A, C = [], []
        at = a0
        ct = c0
        for t in range(n_timesteps):
            A.append(at)
            C.append(ct)
            at1, ct1 = self.layer1(self.X[:, :, t], (at, ct))
            at.retain_grad()
            ct.retain_grad()
            at = at1
            ct = ct1

        at.retain_grad()
        ct.retain_grad()
        A.append(at)
        C.append(ct)

        # don't inclue a0 in our outputs
        self.A = A[1:]
        self.C = C[1:]
        return self.A, self.C

    def extract_grads(self, X):
        self.forward(X)
        self.loss = torch.stack(self.A).sum()
        self.loss.backward()

        w_ii, w_if, w_ic, w_io = self.layer1.weight_ih.chunk(4, 0)
        w_hi, w_hf, w_hc, w_ho = self.layer1.weight_hh.chunk(4, 0)
        bu, bf, bc, bo = self.layer1.bias_ih.chunk(4, 0)

        Wu = torch.cat([torch.t(w_hi), torch.t(w_ii)], dim=0)
        Wf = torch.cat([torch.t(w_hf), torch.t(w_if)], dim=0)
        Wc = torch.cat([torch.t(w_hc), torch.t(w_ic)], dim=0)
        Wo = torch.cat([torch.t(w_ho), torch.t(w_io)], dim=0)

        dw_ii, dw_if, dw_ic, dw_io = self.layer1.weight_ih.grad.chunk(4, 0)
        dw_hi, dw_hf, dw_hc, dw_ho = self.layer1.weight_hh.grad.chunk(4, 0)
        dbu, dbf, dbc, dbo = self.layer1.bias_ih.grad.chunk(4, 0)

        dWu = torch.cat([torch.t(dw_hi), torch.t(dw_ii)], dim=0)
        dWf = torch.cat([torch.t(dw_hf), torch.t(dw_if)], dim=0)
        dWc = torch.cat([torch.t(dw_hc), torch.t(dw_ic)], dim=0)
        dWo = torch.cat([torch.t(dw_ho), torch.t(dw_io)], dim=0)

        grads = {
            "X": self.X.detach().numpy(),
            "Wu": Wu.detach().numpy(),
            "Wf": Wf.detach().numpy(),
            "Wc": Wc.detach().numpy(),
            "Wo": Wo.detach().numpy(),
            "bu": bu.detach().numpy().reshape(-1, 1),
            "bf": bf.detach().numpy().reshape(-1, 1),
            "bc": bc.detach().numpy().reshape(-1, 1),
            "bo": bo.detach().numpy().reshape(-1, 1),
            "C": torch.stack(self.C).detach().numpy(),
            "y": np.swapaxes(
                np.swapaxes(torch.stack(self.A).detach().numpy(), 1, 0), 1, 2
            ),
            "dLdA": np.array([a.grad.numpy() for a in self.A]),
            "dLdWu": dWu.numpy(),
            "dLdWf": dWf.numpy(),
            "dLdWc": dWc.numpy(),
            "dLdWo": dWo.numpy(),
            "dLdBu": dbu.numpy().reshape(-1, 1),
            "dLdBf": dbf.numpy().reshape(-1, 1),
            "dLdBc": dbc.numpy().reshape(-1, 1),
            "dLdBo": dbo.numpy().reshape(-1, 1),
            "dLdX": self.X.grad.numpy(),
        }
        return grads


class TorchRNNCell(nn.Module):
    def __init__(self, n_in, n_hid, params, **kwargs):
        super(TorchRNNCell, self).__init__()

        self.layer1 = nn.RNNCell(n_in, n_hid, bias=True, nonlinearity="tanh")

        # set weights and bias to match those of RNNCell
        # NB: we pass the *transpose* of the RNNCell weights and biases to
        # pytorch, meaning we need to check against the *transpose* of our
        # outputs for any function of the weights
        self.layer1.weight_ih = nn.Parameter(torch.FloatTensor(params["Wax"].T))
        self.layer1.weight_hh = nn.Parameter(torch.FloatTensor(params["Waa"].T))
        self.layer1.bias_ih = nn.Parameter(torch.FloatTensor(params["bx"].T))
        self.layer1.bias_hh = nn.Parameter(torch.FloatTensor(params["ba"].T))

    def forward(self, X):
        self.X = X
        if not isinstance(self.X, torch.Tensor):
            self.X = torchify(self.X)

        self.X.retain_grad()

        # initial hidden state is 0
        n_ex, n_in, n_timesteps = self.X.shape
        n_out, n_out = self.layer1.weight_hh.shape

        # initialize hidden states
        a0 = torchify(np.zeros((n_ex, n_out)))
        a0.retain_grad()

        # forward pass
        A = []
        at = a0
        for t in range(n_timesteps):
            A += [at]
            at1 = self.layer1(self.X[:, :, t], at)
            at.retain_grad()
            at = at1

        at.retain_grad()
        A += [at]

        # don't inclue a0 in our outputs
        self.A = A[1:]
        return self.A

    def extract_grads(self, X):
        self.forward(X)
        self.loss = torch.stack(self.A).sum()
        self.loss.backward()
        grads = {
            "X": self.X.detach().numpy(),
            "ba": self.layer1.bias_hh.detach().numpy(),
            "bx": self.layer1.bias_ih.detach().numpy(),
            "Wax": self.layer1.weight_ih.detach().numpy(),
            "Waa": self.layer1.weight_hh.detach().numpy(),
            "y": torch.stack(self.A).detach().numpy(),
            "dLdA": np.array([a.grad.numpy() for a in self.A]),
            "dLdWaa": self.layer1.weight_hh.grad.numpy(),
            "dLdWax": self.layer1.weight_ih.grad.numpy(),
            "dLdBa": self.layer1.bias_hh.grad.numpy(),
            "dLdBx": self.layer1.bias_ih.grad.numpy(),
            "dLdX": self.X.grad.numpy(),
        }
        return grads


class TorchFCLayer(nn.Module):
    def __init__(self, n_in, n_hid, act_fn, params, **kwargs):
        super(TorchFCLayer, self).__init__()
        self.layer1 = nn.Linear(n_in, n_hid)

        # explicitly set weights and bias
        # NB: we pass the *transpose* of the weights to pytorch, meaning
        # we'll need to check against the *transpose* of our outputs for
        # any function of the weights
        self.layer1.weight = nn.Parameter(torch.FloatTensor(params["W"].T))
        self.layer1.bias = nn.Parameter(torch.FloatTensor(params["b"]))

        self.act_fn = act_fn
        self.model = nn.Sequential(self.layer1, self.act_fn)

    def forward(self, X):
        self.X = X
        if not isinstance(X, torch.Tensor):
            self.X = torchify(X)

        self.z1 = self.layer1(self.X)
        self.z1.retain_grad()

        self.out1 = self.act_fn(self.z1)
        self.out1.retain_grad()

    def extract_grads(self, X):
        self.forward(X)
        self.loss1 = self.out1.sum()
        self.loss1.backward()
        grads = {
            "X": self.X.detach().numpy(),
            "b": self.layer1.bias.detach().numpy(),
            "W": self.layer1.weight.detach().numpy(),
            "y": self.out1.detach().numpy(),
            "dLdy": self.out1.grad.numpy(),
            "dLdZ": self.z1.grad.numpy(),
            "dLdB": self.layer1.bias.grad.numpy(),
            "dLdW": self.layer1.weight.grad.numpy(),
            "dLdX": self.X.grad.numpy(),
        }
        return grads
