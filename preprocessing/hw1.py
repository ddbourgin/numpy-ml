"""
Using a programming language of your choice write code to directly com- pute
the Discrete Fourier Transform (DFT) of an input array. You should express
everything directly at low level using array access, for loops, and arithmetic
operations i.e do not use a complex number type if the language supports it,
and do not use any matrix multiplication facilities. Provide a listing of your
code and a plot showing that your algorithm produces the same magnitude
response as a Fast Fourier Transform routine in your lan- guage of choice that
is either built in or freely available. For testing use a linear combination of
3 sinusoids that are spaced harmonically and have different amplitudes. Plot
the result of multiplying the input signal with the ”closest” bin to the
fundamental frequency of your input as well as the result of multiplying the
input signal with the ”closest” bin to the first har- monic. Now plot the
multiplication of your input signal with a random basis function. Write 2-3
sentences about your observations.

[CSC575] Measure how long it takes to compute 100 DFT transforms of 256, 512,
1024, and 2048 using your direct DFT implementation and compare your measured
times with the ones using some implementation of the Fast Fourier Transform.
"""
#  import scipy.fftpack
import numpy as np

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import seaborn as sns

# https://seaborn.pydata.org/generated/seaborn.set_context.html
# https://seaborn.pydata.org/generated/seaborn.set_style.html
sns.set_style("white")
sns.set_context("notebook", font_scale=1)


# complex# = (real, imaginary)
# 440Hz freq;


def discreteFT(signal, sample_rate=440000):
    """
    Need to implement without using numpy's complex number functionality
    """
    N = len(signal)  # num samples/cycle
    #
    #  omega = 2.0 * np.pi * signal_freq  # 2pi*f = radial freq
    #  theta = omega * time_steps  # omega*t = theta
    #  comp_coord = np.exp(theta * 1j)  # c.f. Euler

    # compute basis vectors
    F = np.zeros((N, N))
    for k in range(N):
        for t in range(N):
            # basis vectors x frequency bins
            F[k, t] = np.exp(-1j * t * k * (2 * np.pi / N))

    # compute spectrum
    spectrum = np.zeros(N)
    for i in range(F.shape[-1]):
        spectrum[i] = np.abs(np.vdot(F[i, :], signal))  # complex dot product

    nyquist_freq = sample_rate / 2  # aliasing threshold
    spectrum_freqs = np.linspace(0, int(nyquist_freq), int(N / 2))
    return spectrum, spectrum_freqs


def compare():
    """
    Compares numpy.fft.fft to discreteFT implementation above
    """
    sr = 440000
    n_samples = 256

    samples = np.arange(n_samples)
    s1 = 3 * np.sin(2 * np.pi * 420 * samples)
    s2 = np.sin(2 * np.pi * 750 * samples)
    s3 = 7 * np.sin(2 * np.pi * 305 * samples)
    signal = s1 + s2 + s3

    hand_spectrum, hand_freqs = discreteFT(signal, sr)

    sp = np.fft.fft(signal)
    freq = np.fft.fftfreq(samples.shape[-1])

    np.testing.assert_almost_equal(hand_spectrum, sp.real)

    fig, ax = plt.subplots(3, 1)
    ax[0].plot(samples, signal)
    ax[0].set_xlabel("Samples")
    ax[0].set_ylabel("Amplitude (v)")
    ax[0].set_title("Signal")
    ax[1].plot(freq, np.abs(sp.real))
    ax[1].set_xlabel("Frequency (Hz)")
    ax[1].set_ylabel("Amplitude (v)")
    ax[1].set_title("DFT Amplitude Spectrum - np.fft.fft")
    ax[2].plot(freq, hand_spectrum)
    ax[2].set_xlabel("Frequency (Hz)")
    ax[2].set_ylabel("Amplitude (v)")
    ax[2].set_title("DFT Amplitude Spectrum - discreteFT")
    plt.show()


if __name__ == "__main__":
    compare()
