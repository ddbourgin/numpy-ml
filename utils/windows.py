import numpy as np

"""
In DSP, windowing functions are useful to counteract the assumption made by the
FFT that the data is infinite and to reduce spectral leakage.
"""


def blackman_harris(window_len, symmetric=False):
    """
    The Blackman-Harris window, an instance of the more general class of
    cosine-sum windows where K=3. Additional coefficients extend the Hamming
    window to further minimize the magnitude of the nearest side-lobe in the
    frequency response.

    bh(n) = a0 - a1 * cos(2 * pi * n / N) + a2 * cos(4 * pi * n / N) - a3 * cos(6 * pi * n / N)

    where
        N = window_len - 1,
        a0 = 0.35875,
        a1 = 0.48829,
        a2 = 0.14128,
        a3 = 0.01168.

    Parameters
    ----------
    window_len : int
        The length of the window in samples. Should be equal to the
        `frame_width` if applying to a windowed signal.
    symmetric : bool (default: False)
        If False, create a 'periodic' window that can be used in with an FFT /
        in spectral analysis.  If True, generate a symmetric window that can be
        used in, e.g., filter design.

    Returns
    -------
    window : numpy array of shape (window_len,)
        The window
    """
    return generalized_cosine(
        window_len, [0.35875, 0.48829, 0.14128, 0.01168], symmetric
    )


def hamming(window_len, symmetric=False):
    """
    The Hamming window, an instance of the more general class of cosine-sum
    windows where K=1 and a0 = 0.54. Coefficients selected to minimize the
    magnitude of the nearest side-lobe in the frequency response.

        hamming(n) = 0.54 - 0.46 * cos((2 * pi * n) / (window_len - 1))

    Parameters
    ----------
    window_len : int
        The length of the window in samples. Should be equal to the
        `frame_width` if applying to a windowed signal.
    symmetric : bool (default: False)
        If False, create a 'periodic' window that can be used in with an FFT /
        in spectral analysis.  If True, generate a symmetric window that can be
        used in, e.g., filter design.

    Returns
    -------
    window : numpy array of shape (window_len,)
        The window
    """
    return generalized_cosine(window_len, [0.54, 1 - 0.54], symmetric)


def hann(window_len, symmetric=False):
    """
    The Hann window, an instance of the more general class of cosine-sum
    windows where K=1 and a0 = 0.5. Unlike the Hamming window, the end points
    of the Hann window touch zero.

        hann(n) = 0.5 - 0.5 * cos((2 * pi * n) / (window_len - 1))

    Parameters
    ----------
    window_len : int
        The length of the window in samples. Should be equal to the
        `frame_width` if applying to a windowed signal.
    symmetric : bool (default: False)
        If False, create a 'periodic' window that can be used in with an FFT /
        in spectral analysis.  If True, generate a symmetric window that can be
        used in, e.g., filter design.

    Returns
    -------
    window : numpy array of shape (window_len,)
        The window
    """
    return generalized_cosine(window_len, [0.5, 0.5], symmetric)


def generalized_cosine(window_len, coefs, symmetric=False):
    """
    The generalized cosine family of window functions. Reflects a simple
    weighted sum of cosine terms.

    For n in [0, window_len]:

        GCW(n) = sum_{k=0}^K (-1)^k * a_k * cos((2 * pi * k * n) / window_len)

    Parameters
    ----------
    window_len : int
        The length of the window in samples. Should be equal to the
        `frame_width` if applying to a windowed signal.
    coefs: list of floats
        The a_k coefficient values
    symmetric : bool (default: False)
        If False, create a 'periodic' window that can be used in with an FFT /
        in spectral analysis.  If True, generate a symmetric window that can be
        used in, e.g., filter design.

    Returns
    -------
    window : numpy array of shape (window_len,)
        The window
    """
    window_len += 1 if not symmetric else 0
    entries = np.linspace(-np.pi, np.pi, window_len)  # (-1)^k * 2pi*n / window_len
    window = np.sum([ak * np.cos(k * entries) for k, ak in enumerate(coefs)], axis=0)
    return window[:-1] if not symmetric else window


class WindowInitializer:
    def __call__(self, window):
        if window == "hamming":
            return hamming
        elif window == "blackman_harris":
            return blackman_harris
        elif window == "hann":
            return hann
        elif window == "generalized_cosine":
            return generalized_cosine
        else:
            raise NotImplementedError("{}".format(window))
