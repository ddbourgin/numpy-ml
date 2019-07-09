import time
import numpy as np

from numpy.testing import assert_almost_equal
from scipy.special import expit

import torch
import torch.nn.functional as F

from .torch_models import (
    torch_gradient_generator,
)

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


#######################################################################
#                           Debug Formatter                           #
#######################################################################


def err_fmt(params, golds, ix, warn_str=""):
    mine, label = params[ix]
    err_msg = "-" * 25 + " DEBUG " + "-" * 25 + "\n"
    prev_mine, prev_label = params[max(ix - 1, 0)]
    err_msg += "Mine (prev) [{}]:\n{}\n\nTheirs (prev) [{}]:\n{}".format(
        prev_label, prev_mine, prev_label, golds[prev_label]
    )
    err_msg += "\n\nMine [{}]:\n{}\n\nTheirs [{}]:\n{}".format(
        label, mine, label, golds[label]
    )
    err_msg += warn_str
    err_msg += "\n" + "-" * 23 + " END DEBUG " + "-" * 23
    return err_msg


#######################################################################
#                            Test Suite                               #
#######################################################################


def test_everything(N=50):
    test_activations(N=N)


def test_activations(N=50):
    print("Testing Sigmoid activation")
    time.sleep(1)
    test_sigmoid_activation(N)
    test_sigmoid_grad(N)

    print("Testing Softmax activation")
    # time.sleep(1)
    # test_softmax_activation(N)
    # test_softmax_grad(N)

    print("Testing Tanh activation")
    time.sleep(1)
    test_tanh_grad(N)

    print("Testing ReLU activation")
    time.sleep(1)
    test_relu_activation(N)
    test_relu_grad(N)

    print("Testing ELU activation")
    time.sleep(1)
    test_elu_activation(N)
    test_elu_grad(N)

    print("Testing SELU activation")
    time.sleep(1)
    test_selu_activation(N)
    test_selu_grad(N)

    print("Testing LeakyRelu activation")
    time.sleep(1)
    test_leakyrelu_activation(N)
    test_leakyrelu_grad(N)


#######################################################################
#                          Activations                                #
#######################################################################


def test_sigmoid_activation(N=None):
    from activations import Sigmoid

    N = np.inf if N is None else N

    mine = Sigmoid()
    gold = expit

    i = 0
    while i < N:
        n_dims = np.random.randint(1, 100)
        z = random_tensor((1, n_dims))
        assert_almost_equal(mine.fn(z), gold(z))
        print("PASSED")
        i += 1


def test_elu_activation(N=None):
    from activations import ELU

    N = np.inf if N is None else N

    i = 0
    while i < N:
        n_dims = np.random.randint(1, 10)
        z = random_tensor((1, n_dims))

        alpha = np.random.uniform(0, 10)

        mine = ELU(alpha)
        gold = lambda z, a: F.elu(torch.from_numpy(z), alpha).numpy()

        assert_almost_equal(mine.fn(z), gold(z, alpha))
        print("PASSED")
        i += 1


def test_softmax_activation(N=None):
    from layers import Softmax

    N = np.inf if N is None else N

    mine = Softmax()
    gold = lambda z: F.softmax(torch.FloatTensor(z), dim=1).numpy()

    i = 0
    while i < N:
        n_dims = np.random.randint(1, 100)
        z = random_stochastic_matrix(1, n_dims)
        assert_almost_equal(mine.forward(z), gold(z))
        print("PASSED")
        i += 1


def test_relu_activation(N=None):
    from activations import ReLU

    N = np.inf if N is None else N

    mine = ReLU()
    gold = lambda z: F.relu(torch.FloatTensor(z)).numpy()

    i = 0
    while i < N:
        n_dims = np.random.randint(1, 100)
        z = random_stochastic_matrix(1, n_dims)
        assert_almost_equal(mine.fn(z), gold(z))
        print("PASSED")
        i += 1


def test_selu_activation(N=None):
    from activations import SELU

    N = np.inf if N is None else N

    mine = SELU()
    gold = lambda z: F.selu(torch.FloatTensor(z)).numpy()

    i = 0
    while i < N:
        n_dims = np.random.randint(1, 100)
        z = random_stochastic_matrix(1, n_dims)
        assert_almost_equal(mine.fn(z), gold(z))
        print("PASSED")
        i += 1


def test_leakyrelu_activation(N=None):
    from activations import LeakyReLU

    N = np.inf if N is None else N

    i = 0
    while i < N:
        n_dims = np.random.randint(1, 100)
        z = random_stochastic_matrix(1, n_dims)
        alpha = np.random.uniform(0, 10)

        mine = LeakyReLU(alpha=alpha)
        gold = lambda z: F.leaky_relu(torch.FloatTensor(z), alpha).numpy()
        assert_almost_equal(mine.fn(z), gold(z))

        print("PASSED")
        i += 1


#######################################################################
#                      Activation Gradients                           #
#######################################################################


def test_sigmoid_grad(N=None):
    from activations import Sigmoid

    N = np.inf if N is None else N

    mine = Sigmoid()
    gold = torch_gradient_generator(torch.sigmoid)

    i = 0
    while i < N:
        n_ex = np.random.randint(1, 100)
        n_dims = np.random.randint(1, 100)
        z = random_tensor((n_ex, n_dims))
        assert_almost_equal(mine.grad(z), gold(z))
        print("PASSED")
        i += 1


def test_elu_grad(N=None):
    from activations import ELU

    N = np.inf if N is None else N

    i = 0
    while i < N:
        n_ex = np.random.randint(1, 10)
        n_dims = np.random.randint(1, 10)
        alpha = np.random.uniform(0, 10)
        z = random_tensor((n_ex, n_dims))

        mine = ELU(alpha)
        gold = torch_gradient_generator(F.elu, alpha=alpha)
        assert_almost_equal(mine.grad(z), gold(z), decimal=6)
        print("PASSED")
        i += 1


def test_tanh_grad(N=None):
    from activations import Tanh

    N = np.inf if N is None else N

    mine = Tanh()
    gold = torch_gradient_generator(torch.tanh)

    i = 0
    while i < N:
        n_ex = np.random.randint(1, 100)
        n_dims = np.random.randint(1, 100)
        z = random_tensor((n_ex, n_dims))
        assert_almost_equal(mine.grad(z), gold(z))
        print("PASSED")
        i += 1


def test_relu_grad(N=None):
    from activations import ReLU

    N = np.inf if N is None else N

    mine = ReLU()
    gold = torch_gradient_generator(F.relu)

    i = 0
    while i < N:
        n_ex = np.random.randint(1, 100)
        n_dims = np.random.randint(1, 100)
        z = random_tensor((n_ex, n_dims))
        assert_almost_equal(mine.grad(z), gold(z))
        print("PASSED")
        i += 1


def test_softmax_grad(N=None):
    from layers import Softmax
    from functools import partial

    np.random.seed(12345)

    N = np.inf if N is None else N
    p_soft = partial(F.softmax, dim=1)
    gold = torch_gradient_generator(p_soft)

    i = 0
    while i < N:
        mine = Softmax()
        n_ex = np.random.randint(1, 3)
        n_dims = np.random.randint(1, 50)
        z = random_tensor((n_ex, n_dims), standardize=True)
        out = mine.forward(z)

        assert_almost_equal(
            gold(z),
            mine.backward(np.ones_like(out)),
            err_msg="Theirs:\n{}\n\nMine:\n{}\n".format(
                gold(z), mine.backward(np.ones_like(out))
            ),
            decimal=3,
        )
        print("PASSED")
        i += 1


def test_selu_grad(N=None):
    from activations import SELU

    N = np.inf if N is None else N

    mine = SELU()
    gold = torch_gradient_generator(F.selu)

    i = 0
    while i < N:
        n_ex = np.random.randint(1, 100)
        n_dims = np.random.randint(1, 100)
        z = random_tensor((n_ex, n_dims))
        assert_almost_equal(mine.grad(z), gold(z))
        print("PASSED")
        i += 1


def test_leakyrelu_grad(N=None):
    from activations import LeakyReLU

    N = np.inf if N is None else N

    i = 0
    while i < N:
        n_ex = np.random.randint(1, 10)
        n_dims = np.random.randint(1, 10)
        alpha = np.random.uniform(0, 10)
        z = random_tensor((n_ex, n_dims))

        mine = LeakyReLU(alpha)
        gold = torch_gradient_generator(F.leaky_relu, negative_slope=alpha)
        assert_almost_equal(mine.grad(z), gold(z), decimal=6)
        print("PASSED")
        i += 1
