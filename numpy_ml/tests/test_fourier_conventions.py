import numpy as np
import pytest

from numpy_ml.preprocessing.dsp import DFT, dft_bins, magnitude_spectrum, power_spectrum


@pytest.mark.parametrize("length", [1, 5, 6, 9])
@pytest.mark.parametrize("complex_input", [False, True])
@pytest.mark.parametrize("positive_only", [False, True])
def test_dft_matches_complex_numpy_coefficients(length, complex_input, positive_only):
    samples = np.arange(length, dtype=float)
    signal = samples**2 - 0.5 * samples + 1
    if complex_input:
        signal = signal + 1j * (samples[::-1] + 0.25)
    original = signal.copy()
    expected = np.fft.fft(signal)
    if positive_only:
        expected = expected[: length // 2 + 1]

    actual = DFT(signal, positive_only=positive_only)

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-10)
    np.testing.assert_array_equal(signal, original)


def test_positive_frequency_sine_has_negative_imaginary_coefficient():
    # Under exp(-2*pi*i*k*n/N), a four-sample positive sine has X[1]=-2j.
    actual = DFT(np.array([0.0, 1.0, 0.0, -1.0]), positive_only=False)
    np.testing.assert_allclose(actual, [0, -2j, 0, 2j], atol=1e-14)


@pytest.mark.parametrize("length", [1, 5, 6, 9])
@pytest.mark.parametrize("sample_rate", [8, 44100])
@pytest.mark.parametrize("positive_only", [False, True])
def test_frequency_bins_match_the_fft_grid(length, sample_rate, positive_only):
    reference = np.fft.rfftfreq if positive_only else np.fft.fftfreq
    expected = reference(length, d=1 / sample_rate)

    actual = dft_bins(length, fs=sample_rate, positive_only=positive_only)

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-10)


@pytest.mark.parametrize("length", [5, 6])
def test_real_signal_magnitude_and_power_remain_consistent(length):
    frames = np.arange(2 * length, dtype=float).reshape(2, length) - 3
    magnitudes = np.abs(np.fft.rfft(frames, axis=1))

    np.testing.assert_allclose(magnitude_spectrum(frames), magnitudes, atol=1e-12)
    np.testing.assert_allclose(power_spectrum(frames), magnitudes**2, atol=1e-11)
