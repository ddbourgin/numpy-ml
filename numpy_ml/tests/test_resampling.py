import numpy as np
import pytest
from scipy.ndimage import map_coordinates

from numpy_ml.preprocessing.dsp import batch_resample, bilinear_interpolate


@pytest.mark.parametrize("mode", ["bilinear", "neighbor"])
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("spatial_shape", [(3, 4), (1, 4), (3, 1), (1, 1)])
def test_batch_resample_identity(mode, batch_size, spatial_shape):
    rows, cols = spatial_shape
    images = np.arange(1, batch_size * rows * cols * 2 + 1, dtype=float).reshape(
        batch_size, rows, cols, 2
    )
    original = images.copy()

    result = batch_resample(images, spatial_shape, mode=mode)

    assert result.shape == images.shape
    np.testing.assert_allclose(result, images)
    np.testing.assert_array_equal(images, original)


@pytest.mark.parametrize(
    "mode, expected_plane",
    [
        ("bilinear", [[1.0, 1.5, 2.0], [2.0, 2.5, 3.0], [3.0, 3.5, 4.0]]),
        ("neighbor", [[1.0, 1.0, 2.0], [1.0, 1.0, 2.0], [3.0, 3.0, 4.0]]),
    ],
)
def test_batch_resample_keeps_images_and_channels_separate(mode, expected_plane):
    # The four input corners are 1, 2, 3, 4. Bilinear values are their
    # linear averages; nearest-neighbor ties use NumPy's round-to-even rule.
    plane = np.array([[1.0, 2.0], [3.0, 4.0]])
    image = np.stack([plane, plane + 10], axis=-1)
    images = np.stack([image, image + 100], axis=0)
    expected_plane = np.array(expected_plane)
    expected_image = np.stack([expected_plane, expected_plane + 10], axis=-1)
    expected = np.stack([expected_image, expected_image + 100], axis=0)

    result = batch_resample(images, (3, 3), mode=mode)

    assert result.shape == (2, 3, 3, 2)
    np.testing.assert_allclose(result, expected)


def test_bilinear_interpolate_preserves_every_integer_grid_sample():
    image = np.arange(1, 25, dtype=float).reshape(3, 4, 2)
    y, x = np.indices(image.shape[:2])

    result = bilinear_interpolate(image, x.ravel(), y.ravel())

    np.testing.assert_array_equal(result, image.reshape(-1, 2))


def test_bilinear_interpolate_matches_scipy_at_and_outside_boundaries():
    image = np.arange(1, 25, dtype=float).reshape(3, 4, 2)
    x = np.array([-2.0, 0.0, 0.5, 1.25, 2.5, 3.0, 8.0])
    y = np.array([0.25, -1.0, 0.5, 1.25, 2.0, 1.5, 4.0])
    expected = np.stack(
        [
            map_coordinates(image[..., channel], [y, x], order=1, mode="nearest")
            for channel in range(image.shape[-1])
        ],
        axis=-1,
    )

    result = bilinear_interpolate(image, x, y)

    np.testing.assert_allclose(result, expected)
