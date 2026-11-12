import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from numpy_ml.preprocessing.general import Standardizer


@pytest.mark.parametrize("with_mean", [False, True])
@pytest.mark.parametrize("with_std", [False, True])
@pytest.mark.parametrize("constant_column", [False, True])
def test_fitted_statistics_support_forward_and_inverse_transforms(
    with_mean, with_std, constant_column
):
    train = np.array([[1.0, 2.0, 8.0], [3.0, 4.0, 4.0], [5.0, 6.0, 12.0]])
    if constant_column:
        train[:, 1] = 4
    unseen = np.array([[2.0, 4.0, 10.0], [6.0, 7.0, 6.0]])
    scaler = Standardizer(with_mean=with_mean, with_std=with_std)
    scaler.fit(train)
    reference = StandardScaler(with_mean=with_mean, with_std=with_std).fit(train)

    params = scaler.parameters
    expected_mean = train.mean(axis=0) if with_mean else np.zeros(3)
    expected_std = train.std(axis=0) if with_std else np.ones(3)
    expected_std[expected_std == 0] = 1
    np.testing.assert_allclose(params["mean"], expected_mean)
    np.testing.assert_allclose(params["std"], expected_std)

    for values in [train, unseen]:
        transformed = scaler.transform(values)
        assert np.isfinite(transformed).all()
        np.testing.assert_allclose(transformed, reference.transform(values))
        np.testing.assert_allclose(scaler.inverse_transform(transformed), values)
    np.testing.assert_array_equal(scaler.parameters["mean"], expected_mean)
    np.testing.assert_array_equal(scaler.parameters["std"], expected_std)


def test_unfitted_standardizer_has_no_statistics():
    assert Standardizer().parameters == {"mean": None, "std": None}
