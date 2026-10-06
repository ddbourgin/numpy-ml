"""Independent high precision and dense-layer tests for SoftPlus tails."""

from decimal import Decimal, localcontext

import numpy as np
import pytest

from numpy_ml.neural_nets.activations import SoftPlus
from numpy_ml.neural_nets.layers import FullyConnected
from numpy_ml.neural_nets.optimizers import SGD


def _decimal_reference(values):
    result = [[], [], []]
    # 500 significant digits retain 1 + exp(-1000) before taking its logarithm.
    with localcontext() as context:
        context.prec = 500
        for value in np.asarray(values).flat:
            exponential = Decimal(str(float(value))).exp()
            denominator = 1 + exponential
            result[0].append(float(denominator.ln()))
            result[1].append(float(exponential / denominator))
            result[2].append(float(exponential / denominator**2))
    return tuple(np.asarray(values_).reshape(np.shape(values)) for values_ in result)


@pytest.mark.parametrize("shape", [(11,), (1, 11), (11, 1, 1)])
def test_softplus_extreme_values_and_derivatives_match_decimal(shape):
    x = np.asarray(
        [-1000.0, -400.0, -100.0, -40.0, -2.0, 0.0, 2.0, 40.0, 100.0, 400.0, 1000.0]
    ).reshape(shape)
    original = x.copy()
    activation = SoftPlus()
    expected = _decimal_reference(x)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        actual = (activation.fn(x), activation.grad(x), activation.grad2(x))
    for actual_values, reference in zip(actual, expected):
        assert np.isfinite(actual_values).all()
        assert actual_values.shape == shape
        # No absolute tolerance: representable negative tails must stay nonzero.
        np.testing.assert_allclose(actual_values, reference, rtol=4e-14, atol=0)
    np.testing.assert_array_equal(x, original)


def test_softplus_float32_and_scalar_inputs_keep_tail_information():
    activation = SoftPlus()
    for x in (
        np.asarray([-40.0, -2.0, 0.0, 2.0, 40.0], dtype=np.float32),
        np.asarray(-40.0),
        np.asarray(40.0),
    ):
        expected = _decimal_reference(x)
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            actual = (activation.fn(x), activation.grad(x), activation.grad2(x))
        for values, reference in zip(actual, expected):
            assert np.shape(values) == np.shape(x)
            np.testing.assert_allclose(values, reference, rtol=2e-6, atol=0)
        assert np.all(np.asarray(actual[2]) > 0)


def test_softplus_curvature_agrees_with_gradient_derivative():
    activation = SoftPlus()
    x = np.asarray([-3.0, -0.4, 0.0, 0.4, 3.0])
    step = 1e-4
    central_difference = (activation.grad(x + step) - activation.grad(x - step)) / (
        2 * step
    )
    np.testing.assert_allclose(
        activation.grad2(x), central_difference, rtol=4e-9, atol=0
    )
    # At x=40, sigma(x) rounds to one, yet its derivative remains representable.
    assert activation.grad2(np.asarray(40.0)) > 0


def test_fully_connected_softplus_extreme_training_step_stays_finite():
    x = np.eye(4)
    weights = np.asarray(
        [
            [-1000.0, -100.0, -40.0],
            [-1.0, 0.0, 1.0],
            [40.0, 100.0, 1000.0],
            [-0.3, 0.5, 4.0],
        ]
    )
    bias = np.zeros((1, 3))
    upstream = np.asarray(
        [[1.0, -0.5, 0.7], [-0.3, 0.4, 0.2], [0.6, -0.8, 0.9], [0.1, 0.5, -0.7]]
    )
    learning_rate = 0.02
    layer = FullyConnected(
        n_out=3, act_fn=SoftPlus(), optimizer=SGD(lr=learning_rate, momentum=0)
    )
    layer.forward(np.zeros_like(x), retain_derived=False)
    layer.parameters["W"] = weights.copy()
    layer.parameters["b"] = bias.copy()
    expected_output, derivative, _ = _decimal_reference(x @ weights + bias)
    dz = upstream * derivative
    expected_dx = dz @ weights.T
    expected_dw = x.T @ dz
    expected_db = dz.sum(axis=0, keepdims=True)

    with np.errstate(over="raise", invalid="raise", divide="raise"):
        output = layer.forward(x)
        dx = layer.backward(upstream)
        actual_dw = layer.gradients["W"].copy()
        actual_db = layer.gradients["b"].copy()
        layer.update()
    for actual, expected in (
        (output, expected_output),
        (dx, expected_dx),
        (actual_dw, expected_dw),
        (actual_db, expected_db),
        (layer.parameters["W"], weights - learning_rate * expected_dw),
        (layer.parameters["b"], bias - learning_rate * expected_db),
    ):
        assert np.isfinite(actual).all()
        np.testing.assert_allclose(actual, expected, rtol=4e-14, atol=0)
    np.testing.assert_array_equal(layer.gradients["W"], np.zeros_like(weights))
    np.testing.assert_array_equal(layer.gradients["b"], np.zeros_like(bias))
