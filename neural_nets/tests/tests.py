from copy import deepcopy

import numpy as np
from numpy.testing import assert_almost_equal

from sklearn.metrics import log_loss, mean_squared_error

# for testing sigmoid
from scipy.special import expit

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import calc_pad_dims_2D, conv2D_naive, conv2D, pad2D, pad1D
from .torch_models import (
    torch_xe_grad,
    torch_mse_grad,
    TorchVAELoss,
    TorchFCLayer,
    TorchRNNCell,
    TorchLSTMCell,
    TorchAddLayer,
    TorchConv1DLayer,
    TorchConv2DLayer,
    TorchPool2DLayer,
    TorchWavenetModule,
    TorchMultiplyLayer,
    TorchDeconv2DLayer,
    TorchBatchNormLayer,
    TorchLinearActivation,
    TorchBidirectionalLSTM,
    torch_gradient_generator,
    TorchSkipConnectionConv,
    TorchSkipConnectionIdentity,
)


#######################################################################
#                               Asserts                               #
#######################################################################


def assert_is_binary(a):
    msg = "Matrix should be one-hot binary"
    assert np.array_equal(a, a.astype(bool)), msg
    assert np.allclose(np.sum(a, axis=1), np.ones(a.shape[0])), msg
    return True


def assert_is_stochastic(a):
    msg = "Array should be stochastic along the columns"
    assert len(a[a < 0]) == len(a[a > 1]) == 0, msg
    assert np.allclose(np.sum(a, axis=1), np.ones(a.shape[0])), msg
    return True


#######################################################################
#                           Data Generators                           #
#######################################################################


def random_one_hot_matrix(n_examples, n_classes):
    """Create a random one-hot matrix of shape n_examples x n_classes"""
    X = np.eye(n_classes)
    X = X[np.random.choice(n_classes, n_examples)]
    return X


def random_stochastic_matrix(n_examples, n_classes):
    """Create a random stochastic matrix of shape n_examples x n_classes"""
    X = np.random.rand(n_examples, n_classes)
    X /= X.sum(axis=1, keepdims=True)
    return X


def random_tensor(shape, standardize=False):
    eps = np.finfo(float).eps
    offset = np.random.randint(-300, 300, shape)
    X = np.random.rand(*shape) + offset

    if standardize:
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + eps)
    return X


def random_binary_tensor(shape, sparsity=0.5):
    X = (np.random.rand(*shape) >= (1 - sparsity)).astype(float)
    return X


#######################################################################
#                           Debug Formatter                           #
#######################################################################


def err_fmt(params, golds, ix):
    mine, label = params[ix]
    err_msg = "-" * 25 + " DEBUG " + "-" * 25 + "\n"
    prev_mine, prev_label = params[max(ix - 1, 0)]
    err_msg += "Mine (prev) [{}]:\n{}\n\nTheirs (prev) [{}]:\n{}".format(
        prev_label, prev_mine, prev_label, golds[prev_label]
    )
    err_msg += "\n\nMine [{}]:\n{}\n\nTheirs [{}]:\n{}".format(
        label, mine, label, golds[label]
    )
    err_msg += "\n" + "-" * 23 + " END DEBUG " + "-" * 23
    return err_msg


#######################################################################
#                         Loss Functions                              #
#######################################################################


def test_squared_error():
    from losses import SquaredError

    mine = SquaredError()
    gold = (
        lambda y, y_pred: mean_squared_error(y, y_pred)
        * y_pred.shape[0]
        * y_pred.shape[1]
        * 0.5
    )

    # ensure we get 0 when the two arrays are equal
    n_dims = np.random.randint(2, 100)
    n_examples = np.random.randint(1, 1000)
    y = y_pred = random_tensor((n_examples, n_dims))
    assert_almost_equal(mine.loss(y, y_pred), gold(y, y_pred))
    print("PASSED")

    while True:
        n_dims = np.random.randint(2, 100)
        n_examples = np.random.randint(1, 1000)
        y = random_tensor((n_examples, n_dims))
        y_pred = random_tensor((n_examples, n_dims))
        assert_almost_equal(mine.loss(y, y_pred), gold(y, y_pred), decimal=5)
        print("PASSED")


def test_cross_entropy():
    from losses import CrossEntropy

    mine = CrossEntropy()
    gold = log_loss

    # ensure we get 0 when the two arrays are equal
    n_classes = np.random.randint(2, 100)
    n_examples = np.random.randint(1, 1000)
    y = y_pred = random_one_hot_matrix(n_examples, n_classes)
    assert_almost_equal(mine.loss(y, y_pred), gold(y, y_pred))
    print("PASSED")

    # test on random inputs
    while True:
        n_classes = np.random.randint(2, 100)
        n_examples = np.random.randint(1, 1000)
        y = random_one_hot_matrix(n_examples, n_classes)
        y_pred = random_stochastic_matrix(n_examples, n_classes)

        assert_almost_equal(mine.loss(y, y_pred), gold(y, y_pred, normalize=False))
        print("PASSED")


def test_VAE_loss():
    from losses import VAELoss

    i = 1
    while True:
        n_ex = np.random.randint(1, 10)
        t_dim = np.random.randint(2, 10)
        t_mean = random_tensor([n_ex, t_dim], standardize=True)
        t_log_var = np.log(np.abs(random_tensor([n_ex, t_dim], standardize=True)))
        im_cols, im_rows = np.random.randint(2, 40), np.random.randint(2, 40)
        X = np.random.rand(n_ex, im_rows * im_cols)
        X_recon = np.random.rand(n_ex, im_rows * im_cols)

        mine = VAELoss()
        mine_loss = mine(X, X_recon, t_mean, t_log_var)
        dX_recon, dLogVar, dMean = mine.grad(X, X_recon, t_mean, t_log_var)
        golds = TorchVAELoss().extract_grads(X, X_recon, t_mean, t_log_var)

        params = [
            (mine_loss, "loss"),
            (dX_recon, "dX_recon"),
            (dLogVar, "dt_log_var"),
            (dMean, "dt_mean"),
        ]
        print("\nTrial {}".format(i))
        for ix, (mine, label) in enumerate(params):
            np.testing.assert_allclose(
                mine,
                golds[label],
                err_msg=err_fmt(params, golds, ix),
                rtol=0.1,
                atol=1e-2,
            )
            print("\tPASSED {}".format(label))
        i += 1


#######################################################################
#                       Loss Function Gradients                       #
#######################################################################


def test_squared_error_grad():
    from losses import SquaredError
    from activations import Tanh

    mine = SquaredError()
    gold = torch_mse_grad
    act = Tanh()

    while True:
        n_dims = np.random.randint(2, 100)
        n_examples = np.random.randint(1, 1000)
        y = random_tensor((n_examples, n_dims))

        # raw inputs
        z = random_tensor((n_examples, n_dims))
        y_pred = act.fn(z)

        assert_almost_equal(
            mine.grad(y, y_pred, z, act), 0.5 * gold(y, z, F.tanh), decimal=4
        )
        print("PASSED")


def test_cross_entropy_grad():
    from losses import CrossEntropy
    from activations import Softmax

    mine = CrossEntropy()
    gold = torch_xe_grad
    sm = Softmax()

    while True:
        n_classes = np.random.randint(2, 100)
        n_examples = np.random.randint(1, 1000)

        y = random_one_hot_matrix(n_examples, n_classes)

        # the cross_entropy_gradient returns the gradient wrt. z (NOT softmax(z))
        z = random_tensor((n_examples, n_classes))
        y_pred = sm.fn(z)

        assert_almost_equal(mine.grad(y, y_pred), gold(y, z), decimal=5)
        print("PASSED")


#######################################################################
#                          Activations                                #
#######################################################################


def test_sigmoid_activation():
    from activations import Sigmoid

    mine = Sigmoid()
    gold = expit

    while True:
        n_dims = np.random.randint(1, 100)
        z = random_tensor((1, n_dims))
        assert_almost_equal(mine.fn(z), gold(z))
        print("PASSED")


def test_softmax_activation():
    from activations import Softmax

    mine = Softmax()
    gold = lambda z: F.softmax(torch.FloatTensor(z), dim=1).numpy()

    while True:
        n_dims = np.random.randint(1, 100)
        z = random_stochastic_matrix(1, n_dims)
        assert_almost_equal(mine.fn(z), gold(z))
        print("PASSED")


def test_relu_activation():
    from activations import ReLU

    mine = ReLU()
    gold = lambda z: F.relu(torch.FloatTensor(z)).numpy()

    while True:
        n_dims = np.random.randint(1, 100)
        z = random_stochastic_matrix(1, n_dims)
        assert_almost_equal(mine.fn(z), gold(z))
        print("PASSED")


#######################################################################
#                      Activation Gradients                           #
#######################################################################


def test_sigmoid_grad():
    from activations import Sigmoid

    mine = Sigmoid()
    gold = torch_gradient_generator(F.sigmoid)

    while True:
        n_ex = np.random.randint(1, 100)
        n_dims = np.random.randint(1, 100)
        z = random_tensor((n_ex, n_dims))
        assert_almost_equal(mine.grad(z), gold(z))
        print("PASSED")


def test_tanh_grad():
    from activations import Tanh

    mine = Tanh()
    gold = torch_gradient_generator(F.tanh)

    while True:
        n_ex = np.random.randint(1, 100)
        n_dims = np.random.randint(1, 100)
        z = random_tensor((n_ex, n_dims))
        assert_almost_equal(mine.grad(z), gold(z))
        print("PASSED")


def test_relu_grad():
    from activations import ReLU

    mine = ReLU()
    gold = torch_gradient_generator(F.relu)

    while True:
        n_ex = np.random.randint(1, 100)
        n_dims = np.random.randint(1, 100)
        z = random_tensor((n_ex, n_dims))
        assert_almost_equal(mine.grad(z), gold(z))
        print("PASSED")


#######################################################################
#                          Layers                                     #
#######################################################################


def test_FullyConnected():
    from layers import FullyConnected
    from activations import Tanh, ReLU, Sigmoid, Affine

    acts = [
        (Tanh(), nn.Tanh(), "Tanh"),
        (Sigmoid(), nn.Sigmoid(), "Sigmoid"),
        (ReLU(), nn.ReLU(), "ReLU"),
        (Affine(), TorchLinearActivation(), "Affine"),
    ]

    i = 1
    while True:
        n_ex = np.random.randint(1, 100)
        n_in = np.random.randint(1, 100)
        n_out = np.random.randint(1, 100)
        X = random_tensor((n_ex, n_in), standardize=True)

        # randomly select an activation function
        act_fn, torch_fn, act_fn_name = acts[np.random.randint(0, len(acts))]

        # initialize FC layer
        L1 = FullyConnected(n_out=n_out, act_fn=act_fn)

        # forward prop
        y_pred = L1.forward(X)

        # backprop
        dLdy = np.ones_like(y_pred)
        dLdX = L1.backward(dLdy)

        # get gold standard gradients
        gold_mod = TorchFCLayer(n_in, n_out, torch_fn, L1.parameters)
        golds = gold_mod.extract_grads(X)

        params = [
            (L1.X, "X"),
            (y_pred, "y"),
            (L1.parameters["W"].T, "W"),
            (L1.parameters["b"], "b"),
            (L1.gradients["Y"], "dLdy"),
            (L1.gradients["Z"], "dLdZ"),
            (L1.gradients["W"].T, "dLdW"),
            (L1.gradients["b"], "dLdB"),
            (dLdX, "dLdX"),
        ]

        print("\nTrial {}\nact_fn={}".format(i, act_fn_name))
        for ix, (mine, label) in enumerate(params):
            assert_almost_equal(
                mine, golds[label], err_msg=err_fmt(params, golds, ix), decimal=3
            )
            print("\tPASSED {}".format(label))
        i += 1


def test_BatchNorm1D():
    from layers import BatchNorm1D

    np.random.seed(12345)

    i = 1
    while True:
        n_ex = np.random.randint(2, 1000)
        n_in = np.random.randint(1, 1000)
        X = random_tensor((n_ex, n_in), standardize=True)

        # initialize BatchNorm1D layer
        L1 = BatchNorm1D()

        # forward prop
        y_pred = L1.forward(X)

        # backprop
        dLdy = np.ones_like(y_pred)
        dLdX = L1.backward(dLdy)

        # get gold standard gradients
        gold_mod = TorchBatchNormLayer(
            n_in, L1.parameters, "1D", epsilon=L1.epsilon, momentum=L1.momentum
        )
        golds = gold_mod.extract_grads(X)

        params = [
            (L1.X, "X"),
            (y_pred, "y"),
            (L1.parameters["scaler"].T, "scaler"),
            (L1.parameters["intercept"], "intercept"),
            (L1.parameters["running_mean"], "running_mean"),
            #  (L1.parameters["running_var"], "running_var"),
            (L1.gradients["scaler"], "dLdScaler"),
            (L1.gradients["intercept"], "dLdIntercept"),
            (dLdX, "dLdX"),
        ]

        print("Trial {}".format(i))
        for ix, (mine, label) in enumerate(params):
            assert_almost_equal(
                mine, golds[label], err_msg=err_fmt(params, golds, ix), decimal=1
            )
            print("\tPASSED {}".format(label))
        i += 1


def test_MultiplyLayer():
    from layers import Multiply
    from activations import Tanh, ReLU, Sigmoid, Affine

    np.random.seed(12345)

    acts = [
        (Tanh(), nn.Tanh(), "Tanh"),
        (Sigmoid(), nn.Sigmoid(), "Sigmoid"),
        (ReLU(), nn.ReLU(), "ReLU"),
        (Affine(), TorchLinearActivation(), "Affine"),
    ]

    i = 1
    while True:
        Xs = []
        n_ex = np.random.randint(1, 100)
        n_in = np.random.randint(1, 100)
        n_entries = np.random.randint(2, 5)
        for _ in range(n_entries):
            Xs.append(random_tensor((n_ex, n_in), standardize=True))

        act_fn, torch_fn, act_fn_name = acts[np.random.randint(0, len(acts))]

        # initialize Add layer
        L1 = Multiply(act_fn)

        # forward prop
        y_pred = L1.forward(Xs)

        # backprop
        dLdy = np.ones_like(y_pred)
        dLdXs = L1.backward(dLdy)

        # get gold standard gradients
        gold_mod = TorchMultiplyLayer(torch_fn)
        golds = gold_mod.extract_grads(Xs)

        params = [(Xs, "Xs"), (y_pred, "Y")]
        params.extend(
            [(dldxi, "dLdX{}".format(i + 1)) for i, dldxi in enumerate(dLdXs)]
        )

        print("\nTrial {}".format(i))
        print("n_ex={}, n_in={}".format(n_ex, n_in))
        print("n_entries={}, act_fn={}".format(n_entries, str(act_fn)))
        for ix, (mine, label) in enumerate(params):
            assert_almost_equal(
                mine, golds[label], err_msg=err_fmt(params, golds, ix), decimal=1
            )
            print("\tPASSED {}".format(label))
        i += 1


def test_AddLayer():
    from layers import Add
    from activations import Tanh, ReLU, Sigmoid, Affine

    np.random.seed(12345)

    acts = [
        (Tanh(), nn.Tanh(), "Tanh"),
        (Sigmoid(), nn.Sigmoid(), "Sigmoid"),
        (ReLU(), nn.ReLU(), "ReLU"),
        (Affine(), TorchLinearActivation(), "Affine"),
    ]

    i = 1
    while True:
        Xs = []
        n_ex = np.random.randint(1, 100)
        n_in = np.random.randint(1, 100)
        n_entries = np.random.randint(2, 5)
        for _ in range(n_entries):
            Xs.append(random_tensor((n_ex, n_in), standardize=True))

        act_fn, torch_fn, act_fn_name = acts[np.random.randint(0, len(acts))]

        # initialize Add layer
        L1 = Add(act_fn)

        # forward prop
        y_pred = L1.forward(Xs)

        # backprop
        dLdy = np.ones_like(y_pred)
        dLdXs = L1.backward(dLdy)

        # get gold standard gradients
        gold_mod = TorchAddLayer(torch_fn)
        golds = gold_mod.extract_grads(Xs)

        params = [(Xs, "Xs"), (y_pred, "Y")]
        params.extend(
            [(dldxi, "dLdX{}".format(i + 1)) for i, dldxi in enumerate(dLdXs)]
        )

        print("\nTrial {}".format(i))
        print("n_ex={}, n_in={}".format(n_ex, n_in))
        print("n_entries={}, act_fn={}".format(n_entries, str(act_fn)))
        for ix, (mine, label) in enumerate(params):
            assert_almost_equal(
                mine, golds[label], err_msg=err_fmt(params, golds, ix), decimal=1
            )
            print("\tPASSED {}".format(label))
        i += 1


def test_SkipConnectionIdentityModule():
    from modules import SkipConnectionIdentityModule
    from activations import Tanh, ReLU, Sigmoid, Affine

    np.random.seed(12345)

    acts = [
        (Tanh(), nn.Tanh(), "Tanh"),
        (Sigmoid(), nn.Sigmoid(), "Sigmoid"),
        (ReLU(), nn.ReLU(), "ReLU"),
        (Affine(), TorchLinearActivation(), "Affine"),
    ]

    i = 1
    while True:
        n_ex = np.random.randint(2, 10)
        in_rows = np.random.randint(2, 25)
        in_cols = np.random.randint(2, 25)
        n_in = np.random.randint(2, 5)
        n_out = n_in
        f_shape1 = (
            min(in_rows, np.random.randint(1, 5)),
            min(in_cols, np.random.randint(1, 5)),
        )
        f_shape2 = (
            min(in_rows, np.random.randint(1, 5)),
            min(in_cols, np.random.randint(1, 5)),
        )
        s1 = np.random.randint(1, 5)
        s2 = np.random.randint(1, 5)

        # randomly select an activation function
        act_fn, torch_fn, act_fn_name = acts[np.random.randint(0, len(acts))]

        X = random_tensor((n_ex, in_rows, in_cols, n_in), standardize=True)

        p1 = calc_pad_dims_2D(X.shape, X.shape[1:3], f_shape1, s1)
        if p1[0] != p1[1] or p1[2] != p1[3]:
            continue

        p2 = calc_pad_dims_2D(X.shape, X.shape[1:3], f_shape2, s2)
        if p2[0] != p2[1] or p2[2] != p2[3]:
            continue

        p1 = (p1[0], p1[2])
        p2 = (p2[0], p2[2])

        # initialize SkipConnectionIdentity module
        L1 = SkipConnectionIdentityModule(
            out_ch=n_out,
            kernel_shape1=f_shape1,
            kernel_shape2=f_shape2,
            stride1=s1,
            stride2=s2,
            act_fn=act_fn,
            epsilon=1e-5,
            momentum=0.9,
        )

        # forward prop
        y_pred = L1.forward(X)

        # backprop
        dLdy = np.ones_like(y_pred)
        dLdX = L1.backward(dLdy)

        # get gold standard gradients
        gold_mod = TorchSkipConnectionIdentity(
            torch_fn,
            p1,
            p2,
            L1.parameters,
            L1.hyperparameters,
            momentum=L1.momentum,
            epsilon=L1.epsilon,
        )
        golds = gold_mod.extract_grads(X)

        params = L1.parameters["components"]
        grads = L1.gradients["components"]
        params = [
            (X, "X"),
            (params["conv1"]["W"], "conv1_W"),
            (params["conv1"]["b"], "conv1_b"),
            (params["batchnorm1"]["scaler"].T, "bn1_scaler"),
            (params["batchnorm1"]["intercept"], "bn1_intercept"),
            (params["batchnorm1"]["running_mean"], "bn1_running_mean"),
            #  (params["batchnorm1"]["running_var"], "bn1_running_var"),
            (params["conv2"]["W"], "conv2_W"),
            (params["conv2"]["b"], "conv2_b"),
            (params["batchnorm2"]["scaler"].T, "bn2_scaler"),
            (params["batchnorm2"]["intercept"], "bn2_intercept"),
            (params["batchnorm2"]["running_mean"], "bn2_running_mean"),
            #  (params["batchnorm2"]["running_var"], "bn2_running_var"),
            (L1._dv["conv1_out"], "act1_out"),
            (L1._dv["batchnorm1_out"], "bn1_out"),
            (L1._dv["conv2_out"], "conv2_out"),
            (L1._dv["batchnorm2_out"], "bn2_out"),
            (y_pred, "Y"),
            (dLdy, "dLdY"),
            (L1.derived_variables["dLdBn2"], "dLdBn2_out"),
            (L1.derived_variables["dLdConv2"], "dLdConv2_out"),
            (L1.derived_variables["dLdBn1"], "dLdBn1_out"),
            (L1.derived_variables["dLdConv1"], "dLdActFn1_out"),
            (dLdX, "dLdX"),
            (grads["batchnorm2"]["scaler"].T, "dLdBn2_scaler"),
            (grads["batchnorm2"]["intercept"], "dLdBn2_intercept"),
            (grads["conv2"]["W"], "dLdConv2_W"),
            (grads["conv2"]["b"], "dLdConv2_b"),
            (grads["batchnorm1"]["scaler"].T, "dLdBn1_scaler"),
            (grads["batchnorm1"]["intercept"], "dLdBn1_intercept"),
            (grads["conv1"]["W"], "dLdConv1_W"),
            (grads["conv1"]["b"], "dLdConv1_b"),
        ]

        print("\nTrial {}".format(i))
        print("act_fn={}, n_ex={}".format(act_fn, n_ex))
        print("in_rows={}, in_cols={}, n_in={}".format(in_rows, in_cols, n_in))
        print("pad1={}, stride1={}, f_shape1={}".format(p1, s1, f_shape1))
        print("pad2={}, stride2={}, f_shape2={}".format(p2, s2, f_shape2))
        for ix, (mine, label) in enumerate(params):
            assert_almost_equal(
                mine, golds[label], err_msg=err_fmt(params, golds, ix), decimal=2
            )
            print("\tPASSED {}".format(label))
        i += 1


def test_SkipConnectionConvModule():
    from modules import SkipConnectionConvModule
    from activations import Tanh, ReLU, Sigmoid, Affine

    np.random.seed(12345)

    acts = [
        (Tanh(), nn.Tanh(), "Tanh"),
        (Sigmoid(), nn.Sigmoid(), "Sigmoid"),
        (ReLU(), nn.ReLU(), "ReLU"),
        (Affine(), TorchLinearActivation(), "Affine"),
    ]

    i = 1
    while True:
        n_ex = np.random.randint(2, 10)
        in_rows = np.random.randint(2, 10)
        in_cols = np.random.randint(2, 10)
        n_in = np.random.randint(2, 5)
        n_out1 = np.random.randint(2, 5)
        n_out2 = np.random.randint(2, 5)
        f_shape1 = (
            min(in_rows, np.random.randint(1, 5)),
            min(in_cols, np.random.randint(1, 5)),
        )
        f_shape2 = (
            min(in_rows, np.random.randint(1, 5)),
            min(in_cols, np.random.randint(1, 5)),
        )
        f_shape_skip = (
            min(in_rows, np.random.randint(1, 5)),
            min(in_cols, np.random.randint(1, 5)),
        )

        s1 = np.random.randint(1, 5)
        s2 = np.random.randint(1, 5)
        s_skip = np.random.randint(1, 5)

        # randomly select an activation function
        act_fn, torch_fn, act_fn_name = acts[np.random.randint(0, len(acts))]

        X = random_tensor((n_ex, in_rows, in_cols, n_in), standardize=True)

        p1 = (np.random.randint(1, 5), np.random.randint(1, 5))
        p2 = (np.random.randint(1, 5), np.random.randint(1, 5))

        # initialize SkipConnectionConv module
        L1 = SkipConnectionConvModule(
            out_ch1=n_out1,
            out_ch2=n_out2,
            kernel_shape1=f_shape1,
            kernel_shape2=f_shape2,
            kernel_shape_skip=f_shape_skip,
            stride1=s1,
            stride2=s2,
            stride_skip=s_skip,
            pad1=p1,
            pad2=p2,
            act_fn=act_fn,
            epsilon=1e-5,
            momentum=0.9,
        )

        # forward prop
        try:
            y_pred = L1.forward(X)
        except (ValueError, AssertionError):
            print("Invalid padding; Retrying")
            continue

        ps = L1.hyperparameters["pad_skip"]
        if ps[0] != ps[1] or ps[2] != ps[3]:
            continue
        pad_skip = (ps[0], ps[2])

        # backprop
        dLdy = np.ones_like(y_pred)
        dLdX = L1.backward(dLdy)

        # get gold standard gradients
        gold_mod = TorchSkipConnectionConv(
            torch_fn,
            p1,
            p2,
            pad_skip,
            L1.parameters,
            L1.hyperparameters,
            momentum=L1.momentum,
            epsilon=L1.epsilon,
        )
        golds = gold_mod.extract_grads(X)

        params = L1.parameters["components"]
        grads = L1.gradients["components"]
        params = [
            (X, "X"),
            (params["conv1"]["W"], "conv1_W"),
            (params["conv1"]["b"], "conv1_b"),
            (params["batchnorm1"]["scaler"].T, "bn1_scaler"),
            (params["batchnorm1"]["intercept"], "bn1_intercept"),
            (params["batchnorm1"]["running_mean"], "bn1_running_mean"),
            #  (params["batchnorm1"]["running_var"], "bn1_running_var"),
            (params["conv2"]["W"], "conv2_W"),
            (params["conv2"]["b"], "conv2_b"),
            (params["batchnorm2"]["scaler"].T, "bn2_scaler"),
            (params["batchnorm2"]["intercept"], "bn2_intercept"),
            (params["batchnorm2"]["running_mean"], "bn2_running_mean"),
            #  (params["batchnorm2"]["running_var"], "bn2_running_var"),
            (params["conv_skip"]["W"], "conv_skip_W"),
            (params["conv_skip"]["b"], "conv_skip_b"),
            (params["batchnorm_skip"]["scaler"].T, "bn_skip_scaler"),
            (params["batchnorm_skip"]["intercept"], "bn_skip_intercept"),
            (params["batchnorm_skip"]["running_mean"], "bn_skip_running_mean"),
            #  (params["batchnorm_skip"]["running_var"], "bn_skip_running_var"),
            (L1._dv["conv1_out"], "act1_out"),
            (L1._dv["batchnorm1_out"], "bn1_out"),
            (L1._dv["conv2_out"], "conv2_out"),
            (L1._dv["batchnorm2_out"], "bn2_out"),
            (L1._dv["conv_skip_out"], "conv_skip_out"),
            (L1._dv["batchnorm_skip_out"], "bn_skip_out"),
            (y_pred, "Y"),
            (dLdy, "dLdY"),
            (L1.derived_variables["dLdBn2"], "dLdBn2_out"),
            (L1.derived_variables["dLdConv2"], "dLdConv2_out"),
            (L1.derived_variables["dLdBnSkip"], "dLdBnSkip_out"),
            (L1.derived_variables["dLdConvSkip"], "dLdConvSkip_out"),
            (L1.derived_variables["dLdBn1"], "dLdBn1_out"),
            (L1.derived_variables["dLdConv1"], "dLdActFn1_out"),
            (dLdX, "dLdX"),
            (grads["batchnorm_skip"]["scaler"].T, "dLdBnSkip_scaler"),
            (grads["batchnorm_skip"]["intercept"], "dLdBnSkip_intercept"),
            (grads["conv_skip"]["W"], "dLdConvSkip_W"),
            (grads["conv_skip"]["b"], "dLdConvSkip_b"),
            (grads["batchnorm2"]["scaler"].T, "dLdBn2_scaler"),
            (grads["batchnorm2"]["intercept"], "dLdBn2_intercept"),
            (grads["conv2"]["W"], "dLdConv2_W"),
            (grads["conv2"]["b"], "dLdConv2_b"),
            (grads["batchnorm1"]["scaler"].T, "dLdBn1_scaler"),
            (grads["batchnorm1"]["intercept"], "dLdBn1_intercept"),
            (grads["conv1"]["W"], "dLdConv1_W"),
            (grads["conv1"]["b"], "dLdConv1_b"),
        ]

        print("\nTrial {}".format(i))
        print("act_fn={}, n_ex={}".format(act_fn, n_ex))
        print("in_rows={}, in_cols={}, n_in={}".format(in_rows, in_cols, n_in))
        print("pad1={}, stride1={}, f_shape1={}".format(p1, s1, f_shape1))
        print("pad2={}, stride2={}, f_shape2={}".format(p2, s2, f_shape2))
        print("stride_skip={}, f_shape_skip={}".format(s_skip, f_shape_skip))
        for ix, (mine, label) in enumerate(params):
            assert_almost_equal(
                mine, golds[label], err_msg=err_fmt(params, golds, ix), decimal=2
            )
            print("\tPASSED {}".format(label))
        i += 1


def test_BatchNorm2D():
    from layers import BatchNorm2D

    np.random.seed(12345)

    i = 1
    while True:
        n_ex = np.random.randint(2, 10)
        in_rows = np.random.randint(1, 10)
        in_cols = np.random.randint(1, 10)
        n_in = np.random.randint(1, 3)

        # initialize BatchNorm2D layer
        X = random_tensor((n_ex, in_rows, in_cols, n_in), standardize=True)
        L1 = BatchNorm2D()

        # forward prop
        y_pred = L1.forward(X)

        # standard sum loss
        dLdy = np.ones_like(X)
        dLdX = L1.backward(dLdy)

        # get gold standard gradients
        gold_mod = TorchBatchNormLayer(
            n_in, L1.parameters, mode="2D", epsilon=L1.epsilon, momentum=L1.momentum
        )
        golds = gold_mod.extract_grads(X, Y_true=None)

        params = [
            (L1.X, "X"),
            (L1.hyperparameters["momentum"], "momentum"),
            (L1.hyperparameters["epsilon"], "epsilon"),
            (L1.parameters["scaler"].T, "scaler"),
            (L1.parameters["intercept"], "intercept"),
            (L1.parameters["running_mean"], "running_mean"),
            #  (L1.parameters["running_var"], "running_var"),
            (y_pred, "y"),
            (L1.gradients["scaler"], "dLdScaler"),
            (L1.gradients["intercept"], "dLdIntercept"),
            (dLdX, "dLdX"),
        ]

        print("Trial {}".format(i))
        for ix, (mine, label) in enumerate(params):
            assert_almost_equal(
                mine, golds[label], err_msg=err_fmt(params, golds, ix), decimal=3
            )

            print("\tPASSED {}".format(label))

        i += 1


def test_RNNCell():
    from layers import RNNCell

    np.random.seed(12345)

    i = 1
    while True:
        n_ex = np.random.randint(1, 10)
        n_in = np.random.randint(1, 10)
        n_out = np.random.randint(1, 10)
        n_t = np.random.randint(1, 10)
        X = random_tensor((n_ex, n_in, n_t), standardize=True)

        # initialize RNN layer
        L1 = RNNCell(n_out=n_out)

        # forward prop
        y_preds = []
        for t in range(n_t):
            y_pred = L1.forward(X[:, :, t])
            y_preds += [y_pred]

        # backprop
        dLdX = []
        dLdAt = np.ones_like(y_preds[t])
        for t in reversed(range(n_t)):
            dLdXt = L1.backward(dLdAt)
            dLdX.insert(0, dLdXt)
        dLdX = np.dstack(dLdX)

        # get gold standard gradients
        gold_mod = TorchRNNCell(n_in, n_out, L1.parameters)
        golds = gold_mod.extract_grads(X)

        params = [
            (X, "X"),
            (np.array(y_preds), "y"),
            (L1.parameters["ba"].T, "ba"),
            (L1.parameters["bx"].T, "bx"),
            (L1.parameters["Wax"].T, "Wax"),
            (L1.parameters["Waa"].T, "Waa"),
            (L1.gradients["ba"].T, "dLdBa"),
            (L1.gradients["bx"].T, "dLdBx"),
            (L1.gradients["Wax"].T, "dLdWax"),
            (L1.gradients["Waa"].T, "dLdWaa"),
            (dLdX, "dLdX"),
        ]

        print("Trial {}".format(i))
        for ix, (mine, label) in enumerate(params):
            np.testing.assert_allclose(
                mine,
                golds[label],
                err_msg=err_fmt(params, golds, ix),
                atol=1e-3,
                rtol=1e-3,
            )
            print("\tPASSED {}".format(label))
        i += 1


def test_conv():
    while True:
        n_ex = np.random.randint(2, 15)
        in_rows = np.random.randint(2, 15)
        in_cols = np.random.randint(2, 15)
        in_ch = np.random.randint(2, 15)
        out_ch = np.random.randint(2, 15)
        f_shape = (
            min(in_rows, np.random.randint(2, 10)),
            min(in_cols, np.random.randint(2, 10)),
        )
        s = np.random.randint(1, 3)
        p = np.random.randint(0, 5)

        X = np.random.rand(n_ex, in_rows, in_cols, in_ch)
        X_pad, p = pad2D(X, p)
        W = np.random.randn(f_shape[0], f_shape[1], in_ch, out_ch)

        gold = conv2D_naive(X, W, s, p)
        mine = conv2D(X, W, s, p)

        np.testing.assert_almost_equal(mine, gold)
        print("PASSED")


def test_Conv2D():
    from layers import Conv2D
    from activations import Tanh, ReLU, Sigmoid, Affine

    np.random.seed(12345)

    acts = [
        (Tanh(), nn.Tanh(), "Tanh"),
        (Sigmoid(), nn.Sigmoid(), "Sigmoid"),
        (ReLU(), nn.ReLU(), "ReLU"),
        (Affine(), TorchLinearActivation(), "Affine"),
    ]

    i = 1
    while True:
        n_ex = np.random.randint(1, 10)
        in_rows = np.random.randint(1, 10)
        in_cols = np.random.randint(1, 10)
        n_in, n_out = np.random.randint(1, 3), np.random.randint(1, 3)
        f_shape = (
            min(in_rows, np.random.randint(1, 5)),
            min(in_cols, np.random.randint(1, 5)),
        )
        p, s = np.random.randint(0, 5), np.random.randint(1, 3)
        d = np.random.randint(0, 5)

        fr, fc = f_shape[0] * (d + 1) - d, f_shape[1] * (d + 1) - d
        out_rows = int(1 + (in_rows + 2 * p - fr) / s)
        out_cols = int(1 + (in_cols + 2 * p - fc) / s)

        if out_rows <= 0 or out_cols <= 0:
            continue

        X = random_tensor((n_ex, in_rows, in_cols, n_in), standardize=True)

        # randomly select an activation function
        act_fn, torch_fn, act_fn_name = acts[np.random.randint(0, len(acts))]

        # initialize Conv2D layer
        L1 = Conv2D(
            out_ch=n_out,
            kernel_shape=f_shape,
            act_fn=act_fn,
            pad=p,
            stride=s,
            dilation=d,
        )

        # forward prop
        y_pred = L1.forward(X)

        # backprop
        dLdy = np.ones_like(y_pred)
        dLdX = L1.backward(dLdy)

        # get gold standard gradients
        gold_mod = TorchConv2DLayer(
            n_in, n_out, torch_fn, L1.parameters, L1.hyperparameters
        )
        golds = gold_mod.extract_grads(X)

        params = [
            (L1.X, "X"),
            (y_pred, "y"),
            (L1.parameters["W"], "W"),
            (L1.parameters["b"], "b"),
            (L1.gradients["W"], "dLdW"),
            (L1.gradients["b"], "dLdB"),
            (dLdX, "dLdX"),
        ]

        print("\nTrial {}".format(i))
        print("pad={}, stride={}, f_shape={}, n_ex={}".format(p, s, f_shape, n_ex))
        print("in_rows={}, in_cols={}, n_in={}".format(in_rows, in_cols, n_in))
        print("out_rows={}, out_cols={}, n_out={}".format(out_rows, out_cols, n_out))
        print("dilation={}".format(d))
        for ix, (mine, label) in enumerate(params):
            assert_almost_equal(
                mine, golds[label], err_msg=err_fmt(params, golds, ix), decimal=4
            )
            print("\tPASSED {}".format(label))
        i += 1


def test_Conv1D():
    from layers import Conv1D
    from activations import Tanh, ReLU, Sigmoid, Affine

    np.random.seed(12345)

    acts = [
        (Tanh(), nn.Tanh(), "Tanh"),
        (Sigmoid(), nn.Sigmoid(), "Sigmoid"),
        (ReLU(), nn.ReLU(), "ReLU"),
        (Affine(), TorchLinearActivation(), "Affine"),
    ]

    i = 1
    while True:
        n_ex = np.random.randint(1, 10)
        l_in = np.random.randint(1, 10)
        n_in, n_out = np.random.randint(1, 3), np.random.randint(1, 3)
        f_width = min(l_in, np.random.randint(1, 5))
        p, s = np.random.randint(0, 5), np.random.randint(1, 3)
        d = np.random.randint(0, 5)

        fc = f_width * (d + 1) - d
        l_out = int(1 + (l_in + 2 * p - fc) / s)

        if l_out <= 0:
            continue

        X = random_tensor((n_ex, l_in, n_in), standardize=True)

        # randomly select an activation function
        act_fn, torch_fn, act_fn_name = acts[np.random.randint(0, len(acts))]

        # initialize Conv2D layer
        L1 = Conv1D(
            out_ch=n_out,
            kernel_width=f_width,
            act_fn=act_fn,
            pad=p,
            stride=s,
            dilation=d,
        )

        # forward prop
        y_pred = L1.forward(X)

        # backprop
        dLdy = np.ones_like(y_pred)
        dLdX = L1.backward(dLdy)

        # get gold standard gradients
        gold_mod = TorchConv1DLayer(
            n_in, n_out, torch_fn, L1.parameters, L1.hyperparameters
        )
        golds = gold_mod.extract_grads(X)

        params = [
            (L1.X, "X"),
            (y_pred, "y"),
            (L1.parameters["W"], "W"),
            (L1.parameters["b"], "b"),
            (L1.gradients["W"], "dLdW"),
            (L1.gradients["b"], "dLdB"),
            (dLdX, "dLdX"),
        ]

        print("\nTrial {}".format(i))
        print("pad={}, stride={}, f_width={}, n_ex={}".format(p, s, f_width, n_ex))
        print("l_in={}, n_in={}".format(l_in, n_in))
        print("l_out={}, n_out={}".format(l_out, n_out))
        print("dilation={}".format(d))
        for ix, (mine, label) in enumerate(params):
            assert_almost_equal(
                mine, golds[label], err_msg=err_fmt(params, golds, ix), decimal=4
            )
            print("\tPASSED {}".format(label))
        i += 1


def test_pad1D():
    from layers import Conv1D
    from torch_models import TorchCausalConv1d, torchify

    i = 1
    while True:
        p = np.random.choice(["same", "causal"])
        n_ex = np.random.randint(1, 10)
        l_in = np.random.randint(1, 10)
        n_in, n_out = np.random.randint(1, 3), np.random.randint(1, 3)
        f_width = min(l_in, np.random.randint(1, 5))
        s = np.random.randint(1, 3)
        d = np.random.randint(0, 5)

        X = random_tensor((n_ex, l_in, n_in), standardize=True)
        X_pad, _ = pad1D(X, p, kernel_width=f_width, stride=s, dilation=d)

        # initialize Conv2D layer
        L1 = Conv1D(out_ch=n_out, kernel_width=f_width, pad=0, stride=s, dilation=d)

        # forward prop
        try:
            y_pred = L1.forward(X_pad)
        except ValueError:
            continue

        # ignore n. output channels
        print("Trial {}".format(i))
        print("p={} d={} s={} l_in={} f_width={}".format(p, d, s, l_in, f_width))
        print("n_ex={} n_in={} n_out={}".format(n_ex, n_in, n_out))
        assert y_pred.shape[:2] == X.shape[:2], print(
            "y_pred.shape={} X.shape={}".format(y_pred.shape, X.shape)
        )

        if p == "causal":
            gold = TorchCausalConv1d(
                in_channels=n_in,
                out_channels=n_out,
                kernel_size=f_width,
                stride=s,
                dilation=d + 1,
                bias=True,
            )
            if s != 1:
                print(
                    "TorchCausalConv1D does not do same padding for stride > 1. Skipping"
                )
                continue

            XT = torchify(np.moveaxis(X, [0, 1, 2], [0, -1, -2]))
        else:
            gold = nn.Conv1d(
                in_channels=n_in,
                out_channels=n_out,
                kernel_size=f_width,
                padding=0,
                stride=s,
                dilation=d + 1,
                bias=True,
            )
            XT = torchify(np.moveaxis(X_pad, [0, 1, 2], [0, -1, -2]))

        # import weights and biases
        # (f[0], n_in, n_out) -> (n_out, n_in, f[0])
        b = L1.parameters["b"]
        W = np.moveaxis(L1.parameters["W"], [0, 1, 2], [-1, -2, -3])
        assert gold.weight.shape == W.shape
        assert gold.bias.shape == b.flatten().shape

        gold.weight = nn.Parameter(torch.FloatTensor(W))
        gold.bias = nn.Parameter(torch.FloatTensor(b.flatten()))

        outT = gold(XT)
        if outT.ndimension() == 2:
            import ipdb

            ipdb.set_trace()

        gold_out = np.moveaxis(outT.detach().numpy(), [0, 1, 2], [0, -1, -2])
        assert gold_out.shape[:2] == X.shape[:2]

        np.testing.assert_almost_equal(
            y_pred,
            gold_out,
            err_msg=err_fmt(
                [(y_pred.shape, "out.shape"), (y_pred, "out")],
                {"out.shape": gold_out.shape, "out": gold_out},
                1,
            ),
            decimal=4,
        )
        print("PASSED\n")
        i += 1


def test_WaveNetModule():
    from modules import WavenetResidualModule

    np.random.seed(12345)

    i = 1
    while True:
        n_ex = np.random.randint(1, 10)
        l_in = np.random.randint(1, 10)
        ch_residual, ch_dilation = np.random.randint(1, 5), np.random.randint(1, 5)
        f_width = min(l_in, np.random.randint(1, 5))
        d = np.random.randint(0, 5)

        X_main = np.zeros_like(
            random_tensor((n_ex, l_in, ch_residual), standardize=True)
        )
        X_main[0][0][0] = 1.0
        X_skip = np.zeros_like(
            random_tensor((n_ex, l_in, ch_residual), standardize=True)
        )

        # initialize Conv2D layer
        L1 = WavenetResidualModule(
            ch_residual=ch_residual,
            ch_dilation=ch_dilation,
            kernel_width=f_width,
            dilation=d,
        )

        # forward prop
        Y_main, Y_skip = L1.forward(X_main, X_skip)

        # backprop
        dLdY_skip = np.ones_like(Y_skip)
        dLdY_main = np.ones_like(Y_main)
        dLdX_main, dLdX_skip = L1.backward(dLdY_skip, dLdY_main)

        _, conv_1x1_pad = pad1D(
            L1._dv["multiply_gate_out"], "same", kernel_width=1, stride=1, dilation=0
        )
        if conv_1x1_pad[0] != conv_1x1_pad[1]:
            print("Skipping")
            continue

        conv_1x1_pad = conv_1x1_pad[0]

        # get gold standard gradients
        gold_mod = TorchWavenetModule(L1.parameters, L1.hyperparameters, conv_1x1_pad)
        golds = gold_mod.extract_grads(X_main, X_skip)

        dv = L1.derived_variables
        pc = L1.parameters["components"]
        gr = L1.gradients["components"]

        params = [
            (L1.X_main, "X_main"),
            (L1.X_skip, "X_skip"),
            (pc["conv_dilation"]["W"], "conv_dilation_W"),
            (pc["conv_dilation"]["b"], "conv_dilation_b"),
            (pc["conv_1x1"]["W"], "conv_1x1_W"),
            (pc["conv_1x1"]["b"], "conv_1x1_b"),
            (dv["conv_dilation_out"], "conv_dilation_out"),
            (dv["tanh_out"], "tanh_out"),
            (dv["sigm_out"], "sigm_out"),
            (dv["multiply_gate_out"], "multiply_gate_out"),
            (dv["conv_1x1_out"], "conv_1x1_out"),
            (Y_main, "Y_main"),
            (Y_skip, "Y_skip"),
            (dLdY_skip, "dLdY_skip"),
            (dLdY_main, "dLdY_main"),
            (dv["dLdConv_1x1"], "dLdConv_1x1_out"),
            (gr["conv_1x1"]["W"], "dLdConv_1x1_W"),
            (gr["conv_1x1"]["b"], "dLdConv_1x1_b"),
            (dv["dLdMultiply"], "dLdMultiply_out"),
            (dv["dLdTanh"], "dLdTanh_out"),
            (dv["dLdSigmoid"], "dLdSigm_out"),
            (dv["dLdConv_dilation"], "dLdConv_dilation_out"),
            (gr["conv_dilation"]["W"], "dLdConv_dilation_W"),
            (gr["conv_dilation"]["b"], "dLdConv_dilation_b"),
            (dLdX_main, "dLdX_main"),
            (dLdX_skip, "dLdX_skip"),
        ]

        print("\nTrial {}".format(i))
        print("f_width={}, n_ex={}".format(f_width, n_ex))
        print("l_in={}, ch_residual={}".format(l_in, ch_residual))
        print("ch_dilation={} dilation={}".format(ch_dilation, d))
        for ix, (mine, label) in enumerate(params):
            assert_almost_equal(
                mine, golds[label], err_msg=err_fmt(params, golds, ix), decimal=4
            )
            print("\tPASSED {}".format(label))
        i += 1


def test_Deconv2D():
    from layers import Deconv2D
    from activations import Tanh, ReLU, Sigmoid, Affine

    np.random.seed(12345)

    acts = [
        (Tanh(), nn.Tanh(), "Tanh"),
        (Sigmoid(), nn.Sigmoid(), "Sigmoid"),
        (ReLU(), nn.ReLU(), "ReLU"),
        (Affine(), TorchLinearActivation(), "Affine"),
    ]

    i = 1
    while True:
        n_ex = np.random.randint(1, 10)
        in_rows = np.random.randint(1, 10)
        in_cols = np.random.randint(1, 10)
        n_in, n_out = np.random.randint(1, 3), np.random.randint(1, 3)
        f_shape = (
            min(in_rows, np.random.randint(1, 5)),
            min(in_cols, np.random.randint(1, 5)),
        )
        p, s = np.random.randint(0, 5), np.random.randint(1, 3)

        out_rows = s * (in_rows - 1) - 2 * p + f_shape[0]
        out_cols = s * (in_cols - 1) - 2 * p + f_shape[1]

        if out_rows <= 0 or out_cols <= 0:
            continue

        X = random_tensor((n_ex, in_rows, in_cols, n_in), standardize=True)

        # randomly select an activation function
        act_fn, torch_fn, act_fn_name = acts[np.random.randint(0, len(acts))]

        # initialize Deconv2D layer
        L1 = Deconv2D(
            out_ch=n_out, kernel_shape=f_shape, act_fn=act_fn, pad=p, stride=s
        )

        # forward prop
        try:
            y_pred = L1.forward(X)
        except ValueError:
            print("Improper dimensions; retrying")
            continue

        # backprop
        dLdy = np.ones_like(y_pred)
        dLdX = L1.backward(dLdy)

        # get gold standard gradients
        gold_mod = TorchDeconv2DLayer(
            n_in, n_out, torch_fn, L1.parameters, L1.hyperparameters
        )
        golds = gold_mod.extract_grads(X)

        params = [
            (L1.X, "X"),
            (L1.parameters["W"], "W"),
            (L1.parameters["b"], "b"),
            (y_pred, "y"),
            (L1.gradients["W"], "dLdW"),
            (L1.gradients["b"], "dLdB"),
            (dLdX, "dLdX"),
        ]

        print("\nTrial {}".format(i))
        print("pad={}, stride={}, f_shape={}, n_ex={}".format(p, s, f_shape, n_ex))
        print("in_rows={}, in_cols={}, n_in={}".format(in_rows, in_cols, n_in))
        print("out_rows={}, out_cols={}, n_out={}".format(out_rows, out_cols, n_out))
        for ix, (mine, label) in enumerate(params):
            assert_almost_equal(
                mine, golds[label], err_msg=err_fmt(params, golds, ix), decimal=4
            )
            print("\tPASSED {}".format(label))
        i += 1


def test_Pool2D():
    from layers import Pool2D

    np.random.seed(12345)

    i = 1
    while True:
        n_ex = np.random.randint(1, 10)
        in_rows = np.random.randint(1, 10)
        in_cols = np.random.randint(1, 10)
        n_in = np.random.randint(1, 3)
        f_shape = (
            min(in_rows, np.random.randint(1, 5)),
            min(in_cols, np.random.randint(1, 5)),
        )
        p, s = np.random.randint(0, max(1, min(f_shape) // 2)), np.random.randint(1, 3)
        #  mode = ["max", "average"][np.random.randint(0, 2)]
        mode = "average"
        out_rows = int(1 + (in_rows + 2 * p - f_shape[0]) / s)
        out_cols = int(1 + (in_cols + 2 * p - f_shape[1]) / s)

        X = random_tensor((n_ex, in_rows, in_cols, n_in), standardize=True)
        print("\nmode: {}".format(mode))
        print("pad={}, stride={}, f_shape={}, n_ex={}".format(p, s, f_shape, n_ex))
        print("in_rows={}, in_cols={}, n_in={}".format(in_rows, in_cols, n_in))
        print("out_rows={}, out_cols={}, n_out={}".format(out_rows, out_cols, n_in))

        # initialize Pool2D layer
        L1 = Pool2D(kernel_shape=f_shape, pad=p, stride=s, mode=mode)

        # forward prop
        y_pred = L1.forward(X)

        # backprop
        dLdy = np.ones_like(y_pred)
        dLdX = L1.backward(dLdy)

        # get gold standard gradients
        gold_mod = TorchPool2DLayer(n_in, L1.hyperparameters)
        golds = gold_mod.extract_grads(X)

        params = [(L1.X, "X"), (y_pred, "y"), (dLdX, "dLdX")]
        for ix, (mine, label) in enumerate(params):
            assert_almost_equal(
                mine, golds[label], err_msg=err_fmt(params, golds, ix), decimal=4
            )
            print("\tPASSED {}".format(label))
        i += 1


def test_LSTMCell():
    from layers import LSTMCell

    np.random.seed(12345)

    i = 1
    while True:
        n_ex = np.random.randint(1, 10)
        n_in = np.random.randint(1, 10)
        n_out = np.random.randint(1, 10)
        n_t = np.random.randint(1, 10)
        X = random_tensor((n_ex, n_in, n_t), standardize=True)

        # initialize LSTM layer
        L1 = LSTMCell(n_out=n_out)

        # forward prop
        Cs = []
        y_preds = []
        for t in range(n_t):
            y_pred, Ct = L1.forward(X[:, :, t])
            y_preds.append(y_pred)
            Cs.append(Ct)

        # backprop
        dLdX = []
        dLdAt = np.ones_like(y_preds[t])
        for t in reversed(range(n_t)):
            dLdXt = L1.backward(dLdAt)
            dLdX.insert(0, dLdXt)
        dLdX = np.dstack(dLdX)
        y_preds = np.dstack(y_preds)
        Cs = np.array(Cs)

        # get gold standard gradients
        gold_mod = TorchLSTMCell(n_in, n_out, L1.parameters)
        golds = gold_mod.extract_grads(X)

        params = [
            (X, "X"),
            (np.array(Cs), "C"),
            (y_preds, "y"),
            (L1.parameters["bo"].T, "bo"),
            (L1.parameters["bu"].T, "bu"),
            (L1.parameters["bf"].T, "bf"),
            (L1.parameters["bc"].T, "bc"),
            (L1.parameters["Wo"], "Wo"),
            (L1.parameters["Wu"], "Wu"),
            (L1.parameters["Wf"], "Wf"),
            (L1.parameters["Wc"], "Wc"),
            (L1.gradients["bo"].T, "dLdBo"),
            (L1.gradients["bu"].T, "dLdBu"),
            (L1.gradients["bf"].T, "dLdBf"),
            (L1.gradients["bc"].T, "dLdBc"),
            (L1.gradients["Wo"], "dLdWo"),
            (L1.gradients["Wu"], "dLdWu"),
            (L1.gradients["Wf"], "dLdWf"),
            (L1.gradients["Wc"], "dLdWc"),
            (dLdX, "dLdX"),
        ]

        print("Case {}".format(i))
        for ix, (mine, label) in enumerate(params):
            np.testing.assert_allclose(
                mine,
                golds[label],
                err_msg=err_fmt(params, golds, ix),
                atol=1e-4,
                rtol=1e-4,
            )

            print("\tPASSED {}".format(label))
        i += 1


def test_BidirectionalLSTM():
    from modules import BidirectionalLSTM

    np.random.seed(12345)

    i = 1
    while True:
        n_ex = np.random.randint(1, 10)
        n_in = np.random.randint(1, 10)
        n_out = np.random.randint(1, 10)
        n_t = np.random.randint(1, 10)
        X = random_tensor((n_ex, n_in, n_t), standardize=True)

        # initialize LSTM layer
        L1 = BidirectionalLSTM(n_out=n_out)

        # forward prop
        y_pred = L1.forward(X)

        # backprop
        dLdA = np.ones_like(y_pred)
        dLdX = L1.backward(dLdA)

        # get gold standard gradients
        gold_mod = TorchBidirectionalLSTM(n_in, n_out, L1.parameters)
        golds = gold_mod.extract_grads(X)

        pms, grads = L1.parameters["components"], L1.gradients["components"]
        params = [
            (X, "X"),
            (y_pred, "y"),
            (pms["cell_fwd"]["bo"].T, "bo_f"),
            (pms["cell_fwd"]["bu"].T, "bu_f"),
            (pms["cell_fwd"]["bf"].T, "bf_f"),
            (pms["cell_fwd"]["bc"].T, "bc_f"),
            (pms["cell_fwd"]["Wo"], "Wo_f"),
            (pms["cell_fwd"]["Wu"], "Wu_f"),
            (pms["cell_fwd"]["Wf"], "Wf_f"),
            (pms["cell_fwd"]["Wc"], "Wc_f"),
            (pms["cell_bwd"]["bo"].T, "bo_b"),
            (pms["cell_bwd"]["bu"].T, "bu_b"),
            (pms["cell_bwd"]["bf"].T, "bf_b"),
            (pms["cell_bwd"]["bc"].T, "bc_b"),
            (pms["cell_bwd"]["Wo"], "Wo_b"),
            (pms["cell_bwd"]["Wu"], "Wu_b"),
            (pms["cell_bwd"]["Wf"], "Wf_b"),
            (pms["cell_bwd"]["Wc"], "Wc_b"),
            (grads["cell_fwd"]["bo"].T, "dLdBo_f"),
            (grads["cell_fwd"]["bu"].T, "dLdBu_f"),
            (grads["cell_fwd"]["bf"].T, "dLdBf_f"),
            (grads["cell_fwd"]["bc"].T, "dLdBc_f"),
            (grads["cell_fwd"]["Wo"], "dLdWo_f"),
            (grads["cell_fwd"]["Wu"], "dLdWu_f"),
            (grads["cell_fwd"]["Wf"], "dLdWf_f"),
            (grads["cell_fwd"]["Wc"], "dLdWc_f"),
            (grads["cell_bwd"]["bo"].T, "dLdBo_b"),
            (grads["cell_bwd"]["bu"].T, "dLdBu_b"),
            (grads["cell_bwd"]["bf"].T, "dLdBf_b"),
            (grads["cell_bwd"]["bc"].T, "dLdBc_b"),
            (grads["cell_bwd"]["Wo"], "dLdWo_b"),
            (grads["cell_bwd"]["Wu"], "dLdWu_b"),
            (grads["cell_bwd"]["Wf"], "dLdWf_b"),
            (grads["cell_bwd"]["Wc"], "dLdWc_b"),
            (dLdX, "dLdX"),
        ]

        print("Case {}".format(i))
        for ix, (mine, label) in enumerate(params):
            np.testing.assert_allclose(
                mine,
                golds[label],
                err_msg=err_fmt(params, golds, ix),
                atol=1e-4,
                rtol=1e-4,
            )

            print("\tPASSED {}".format(label))
        i += 1


def test_VAE():
    # for testing
    from keras.datasets import mnist
    from models.vae import BernoulliVAE

    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # scale pixel intensities to [0, 1]
    X_train = np.expand_dims(X_train.astype("float32") / 255.0, 3)
    X_test = np.expand_dims(X_test.astype("float32") / 255.0, 3)

    X_train = X_train[: 128 * 10]

    BV = BernoulliVAE()
    BV.fit(X_train)


def grad_check_RNN(model, loss_func, param_name, n_t, X, epsilon=1e-7):
    """
    Manual gradient calc for vanilla RNN parameters
    """
    if param_name in ["Ba", "Bx"]:
        param_name = param_name.lower()
    elif param_name in ["X", "y"]:
        return None

    param_orig = model.parameters[param_name]
    model.flush_gradients()
    grads = np.zeros_like(param_orig)

    for flat_ix, val in enumerate(param_orig.flat):
        param = deepcopy(param_orig)
        md_ix = np.unravel_index(flat_ix, param.shape)

        # plus
        y_preds_plus = []
        param[md_ix] = val + epsilon
        model.parameters[param_name] = param
        for t in range(n_t):
            y_pred_plus = model.forward(X[:, :, t])
            y_preds_plus += [y_pred_plus]
        loss_plus = loss_func(y_preds_plus)
        model.flush_gradients()

        # minus
        y_preds_minus = []
        param[md_ix] = val - epsilon
        model.parameters[param_name] = param
        for t in range(n_t):
            y_pred_minus = model.forward(X[:, :, t])
            y_preds_minus += [y_pred_minus]
        loss_minus = loss_func(y_preds_minus)
        model.flush_gradients()

        grad = (loss_plus - loss_minus) / (2 * epsilon)
        grads[md_ix] = grad
    return grads.T
