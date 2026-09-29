import pytest
import numpy as np
from numpy_ml.utils import distance_metrics as target


@pytest.mark.parametrize("name", ["euclidean", "manhattan", "chebyshev", "minkowski"])
@pytest.mark.parametrize(
    "dtype,lo,hi", [(np.uint8, 0, 255), (np.int8, -128, 127), (np.int16, -32768, 32767)]
)
def test_integer_distance(name, dtype, lo, hi):
    a, b = np.array([lo], dtype=dtype), np.array([hi], dtype=dtype)
    fn = getattr(target, name)
    args = (2,) if name == "minkowski" else ()
    assert float(fn(a, b, *args)) == pytest.approx(float(hi - lo))
    assert float(fn(b, a, *args)) == pytest.approx(float(hi - lo))


@pytest.mark.parametrize(
    "name,want",
    [("euclidean", 5), ("manhattan", 7), ("chebyshev", 4), ("minkowski", 5)],
)
def test_float_controls(name, want):
    fn = getattr(target, name)
    args = (2,) if name == "minkowski" else ()
    assert fn(np.array([0.0, 0.0]), np.array([3.0, 4.0]), *args) == pytest.approx(want)


@pytest.mark.parametrize("name", ["euclidean", "manhattan", "chebyshev", "minkowski"])
def test_large_neighboring_ints(name):
    fn = getattr(target, name)
    args = (2,) if name == "minkowski" else ()
    a = np.array([2**60], dtype=np.int64)
    b = np.array([2**60 + 1], dtype=np.int64)
    assert fn(a, b, *args) == pytest.approx(1)


def test_hamming_unchanged():
    assert target.hamming(np.array([0, 1, 2]), np.array([0, 1, 3])) == pytest.approx(
        1 / 3
    )
