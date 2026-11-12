import numpy as np
import pytest
from scipy.sparse import isspmatrix_csr

from numpy_ml.preprocessing.general import FeatureHasher


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize(
    "examples, expected",
    [
        ({"a": 2, "b": -3}, [[0, 2, 0, 0, 0, 0, 0, -3]]),
        (
            [{"a": 2}, {}, {"c": 4}],
            [[0, 2, 0, 0, 0, 0, 0, 0], [0] * 8, [0, 0, 0, 4, 0, 0, 0, 0]],
        ),
        ({b"a": 1.5, ("tuple", 2): 0.25}, [[0.25, 1.5, 0, 0, 0, 0, 0, 0]]),
        ({"a": 2, "alpha": -0.5}, [[0, 1.5, 0, 0, 0, 0, 0, 0]]),
    ],
)
def test_feature_hasher_constructs_dense_and_sparse_matrices(
    sparse, examples, expected
):
    # MD5 digests modulo 8: a=1, b=7, c=3, alpha=1, JSON ["tuple", 2]=0.
    # These fixed values also check that collisions add their contributions.
    result = FeatureHasher(n_dim=8, sparse=sparse).encode(examples)

    assert isspmatrix_csr(result) == sparse
    actual = result.toarray() if sparse else result
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("examples", [[], [{}], [{}, {}]])
def test_feature_hasher_preserves_empty_rows_and_matrix_shape(sparse, examples):
    result = FeatureHasher(n_dim=8, sparse=sparse).encode(examples)

    assert result.shape == (len(examples), 8)
    actual = result.toarray() if sparse else result
    np.testing.assert_array_equal(actual, np.zeros((len(examples), 8)))
