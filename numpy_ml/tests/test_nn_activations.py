# flake8: noqa
import time
import numpy as np

from numpy.testing import assert_almost_equal
from scipy.special import expit

import torch
import torch.nn.functional as F

from numpy_ml.utils.testing import random_stochastic_matrix, random_tensor


def torch_gradient_generator(fn, **kwargs):
    def get_grad(z):
        z1 = torch.autograd.Variable(torch.from_numpy(z), requires_grad=True)
        z2 = fn(z1, **kwargs).sum()
        z2.backward()
        grad = z1.grad.numpy()
        return grad

    return get_grad


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
#
#
#  def test_activations(N=50):
#      print("Testing Sigmoid activation")
#      time.sleep(1)
#      test_sigmoid_activation(N)
#      test_sigmoid_grad(N)
#
#      #  print("Testing Softmax activation")
#      #  time.sleep(1)
#      #  test_softmax_activation(N)
#      #  test_softmax_grad(N)
#
#      print("Testing Tanh activation")
#      time.sleep(1)
#      test_tanh_grad(N)
#
#      print("Testing ReLU activation")
#      time.sleep(1)
#      test_relu_activation(N)
#      test_relu_grad(N)
#
#      print("Testing ELU activation")
#      time.sleep(1)
#      test_elu_activation(N)
#      test_elu_grad(N)
#
#      print("Testing SELU activation")
#      time.sleep(1)
#      test_selu_activation(N)
#      test_selu_grad(N)
#
#      print("Testing LeakyRelu activation")
#      time.sleep(1)
#      test_leakyrelu_activation(N)
#      test_leakyrelu_grad(N)
#
#      print("Testing SoftPlus activation")
#      time.sleep(1)
#      test_softplus_activation(N)
#      test_softplus_grad(N)
#

#######################################################################
#                          Activations                                #
#######################################################################


def test_sigmoid_activation(N=50):
    from numpy_ml.neural_nets.activations import Sigmoid

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


def test_softplus_activation(N=50):
    from numpy_ml.neural_nets.activations import SoftPlus

    N = np.inf if N is None else N

    mine = SoftPlus()
    gold = lambda z: F.softplus(torch.FloatTensor(z)).numpy()

    i = 0
    while i < N:
        n_dims = np.random.randint(1, 100)
        z = random_stochastic_matrix(1, n_dims)
        assert_almost_equal(mine.fn(z), gold(z))
        print("PASSED")
        i += 1


def test_elu_activation(N=50):
    from numpy_ml.neural_nets.activations import ELU

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


def test_relu_activation(N=50):
    from numpy_ml.neural_nets.activations import ReLU

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


def test_selu_activation(N=50):
    from numpy_ml.neural_nets.activations import SELU

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


def test_leakyrelu_activation(N=50):
    from numpy_ml.neural_nets.activations import LeakyReLU

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


def test_sigmoid_grad(N=50):
    from numpy_ml.neural_nets.activations import Sigmoid

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


def test_elu_grad(N=50):
    from numpy_ml.neural_nets.activations import ELU

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


def test_tanh_grad(N=50):
    from numpy_ml.neural_nets.activations import Tanh

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


def test_relu_grad(N=50):
    from numpy_ml.neural_nets.activations import ReLU

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


def test_selu_grad(N=50):
    from numpy_ml.neural_nets.activations import SELU

    N = np.inf if N is None else N

    mine = SELU()
    gold = torch_gradient_generator(F.selu)

    i = 0
    while i < N:
        n_ex = np.random.randint(1, 100)
        n_dims = np.random.randint(1, 100)
        z = random_tensor((n_ex, n_dims))
        assert_almost_equal(mine.grad(z), gold(z), decimal=6)
        print("PASSED")
        i += 1


def test_leakyrelu_grad(N=50):
    from numpy_ml.neural_nets.activations import LeakyReLU

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


def test_softplus_grad(N=50):
    from numpy_ml.neural_nets.activations import SoftPlus

    N = np.inf if N is None else N

    mine = SoftPlus()
    gold = torch_gradient_generator(F.softplus)

    i = 0
    while i < N:
        n_ex = np.random.randint(1, 100)
        n_dims = np.random.randint(1, 100)
        z = random_tensor((n_ex, n_dims), standardize=True)
        assert_almost_equal(mine.grad(z), gold(z))
        print("PASSED")
        i += 1


if __name__ == "__main__":
    test_activations(N=50)
