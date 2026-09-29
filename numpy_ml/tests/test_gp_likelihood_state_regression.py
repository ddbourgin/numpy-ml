"""Likelihood queries must not inject additional observation noise into a fitted GP."""

import numpy as np
import pytest
from numpy_ml.nonparametric.gp import GPRegression


def model(alpha=0.3, outputs=1):
    X = np.array([[-1.0], [-0.3], [0.4], [1.0]])
    y = np.sin(2 * X[:, 0])
    if outputs > 1:
        y = np.column_stack([y, 2 * y + 0.1])
    result = GPRegression(alpha=alpha)
    result.fit(X, y)
    return result


@pytest.mark.parametrize("alpha", [0.05, 0.3, 1.0])
def test_likelihood_leaves_covariance_unchanged(alpha):
    regression = model(alpha)
    before = regression.parameters["GP_cov"].copy()
    regression.marginal_log_likelihood()
    np.testing.assert_array_equal(regression.parameters["GP_cov"], before)


@pytest.mark.parametrize("outputs", [1, 2])
def test_repeated_likelihood_is_idempotent(outputs):
    regression = model(outputs=outputs)
    values = [regression.marginal_log_likelihood() for _ in range(4)]
    np.testing.assert_allclose(values, np.full(4, values[0]), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("alpha", [0.1, 0.5])
def test_prediction_is_independent_of_prior_likelihood_queries(alpha):
    regression = model(alpha)
    test_x = np.array([[-0.8], [0.0], [0.8]])
    before = regression.predict(test_x, return_cov=True)
    for _ in range(3):
        regression.marginal_log_likelihood()
    after = regression.predict(test_x, return_cov=True)
    for actual, expected in zip(after, before):
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("outputs", [1, 2])
def test_likelihood_matches_cholesky_reference(outputs):
    regression = model(outputs=outputs)
    P = regression.parameters
    covariance = P["GP_cov"] + regression.hyperparameters["alpha"] * np.eye(len(P["X"]))
    cholesky = np.linalg.cholesky(covariance)
    y = P["y"]
    y = y[:, None] if y.ndim == 1 else y
    solved = np.linalg.solve(cholesky, y)
    expected = (
        -0.5 * np.sum(solved**2)
        - y.shape[1] * np.log(np.diag(cholesky)).sum()
        - 0.5 * y.size * np.log(2 * np.pi)
    )
    actual = regression.marginal_log_likelihood()
    assert actual == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_read_only_fitted_covariance_is_supported():
    regression = model()
    regression.parameters["GP_cov"].setflags(write=False)
    assert np.isfinite(regression.marginal_log_likelihood())


def test_zero_noise_control():
    regression = model(alpha=0.0)
    before = regression.parameters["GP_cov"].copy()
    first = regression.marginal_log_likelihood()
    assert np.isfinite(first)
    np.testing.assert_array_equal(regression.parameters["GP_cov"], before)
