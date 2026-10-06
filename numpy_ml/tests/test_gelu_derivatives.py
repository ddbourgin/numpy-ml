"""GELU derivatives must differentiate the selected forward computation."""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from numpy_ml.neural_nets.activations import GELU
from numpy_ml.neural_nets.layers import FullyConnected
from numpy_ml.neural_nets.optimizers import SGD


def _activation(approximate):
    activation = GELU()
    # Set the public mode directly to isolate the separate constructor bug.
    activation.approximate = approximate
    return activation


def _torch_values_and_derivatives(x, approximate):
    value = torch.tensor(x, dtype=torch.float64, requires_grad=True)
    output = F.gelu(value, approximate="tanh" if approximate else "none")
    first = torch.autograd.grad(output.sum(), value, create_graph=True)[0]
    second = torch.autograd.grad(first.sum(), value)[0]
    return tuple(v.detach().numpy() for v in (output, first, second))


@pytest.mark.parametrize("approximate", [True, False])
@pytest.mark.parametrize("shape", [(), (6,), (2, 3), (2, 1, 3)])
def test_gelu_value_and_derivatives_match_autograd(approximate, shape):
    x = (
        np.asarray(0.7)
        if not shape
        else np.asarray([-3.0, -1.5, -0.3, 0.0, 0.7, 2.0]).reshape(shape)
    )
    original = x.copy()
    activation = _activation(approximate)
    expected = _torch_values_and_derivatives(x, approximate)
    for actual, reference in zip(
        (activation.fn(x), activation.grad(x), activation.grad2(x)), expected
    ):
        assert np.shape(actual) == shape
        np.testing.assert_allclose(actual, reference, rtol=3e-13, atol=3e-15)
    np.testing.assert_array_equal(x, original)


@pytest.mark.parametrize("approximate", [True, False])
def test_gelu_curvature_decays_at_both_saturated_tails(approximate):
    x = np.asarray([-1000.0, -40.0, -20.0, 20.0, 40.0, 1000.0])
    activation = _activation(approximate)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        actual = activation.grad2(x)
    expected = _torch_values_and_derivatives(x, approximate)[2]
    np.testing.assert_allclose(actual, expected, rtol=3e-13, atol=3e-15)
    assert np.max(np.abs(actual)) < 1e-70


@pytest.mark.parametrize("approximate", [True, False])
def test_fully_connected_gelu_gradients_and_sgd_step_match_autograd(approximate):
    x = np.asarray([[-2.0, 0.3], [-0.7, 1.5], [0.4, -1.0], [2.0, 0.8]])
    weights = np.asarray([[0.6, -0.4, 0.9], [0.2, 0.7, -0.3]])
    bias = np.asarray([[0.1, -0.2, 0.3]])
    upstream = np.asarray(
        [[0.3, -0.5, 0.2], [0.1, 0.4, -0.7], [-0.8, 0.2, 0.5], [0.6, -0.3, 0.1]]
    )
    learning_rate = 0.03
    layer = FullyConnected(
        n_out=3,
        act_fn=_activation(approximate),
        optimizer=SGD(lr=learning_rate, momentum=0),
    )
    layer.forward(x, retain_derived=False)
    layer.parameters["W"] = weights.copy()
    layer.parameters["b"] = bias.copy()

    tx = torch.tensor(x, dtype=torch.float64, requires_grad=True)
    tw = torch.tensor(weights, dtype=torch.float64, requires_grad=True)
    tb = torch.tensor(bias, dtype=torch.float64, requires_grad=True)
    expected_output = F.gelu(
        tx @ tw + tb, approximate="tanh" if approximate else "none"
    )
    loss = (expected_output * torch.tensor(upstream, dtype=torch.float64)).sum()
    expected_dx, expected_dw, expected_db = torch.autograd.grad(loss, (tx, tw, tb))

    output = layer.forward(x)
    dx = layer.backward(upstream)
    for actual, expected in (
        (output, expected_output),
        (dx, expected_dx),
        (layer.gradients["W"], expected_dw),
        (layer.gradients["b"], expected_db),
    ):
        np.testing.assert_allclose(
            actual, expected.detach().numpy(), rtol=3e-13, atol=3e-15
        )

    layer.update()
    np.testing.assert_allclose(
        layer.parameters["W"],
        weights - learning_rate * expected_dw.detach().numpy(),
        rtol=3e-13,
        atol=3e-15,
    )
    np.testing.assert_allclose(
        layer.parameters["b"],
        bias - learning_rate * expected_db.detach().numpy(),
        rtol=3e-13,
        atol=3e-15,
    )
    np.testing.assert_array_equal(layer.gradients["W"], np.zeros_like(weights))
    np.testing.assert_array_equal(layer.gradients["b"], np.zeros_like(bias))
