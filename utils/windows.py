import numpy as np


def blackman_harris(window_len):
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
    """
    return generalized_cosine(window_len, [0.35875, 0.48829, 0.14128, 0.01168])


def hamming(window_len):
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
    """
    return generalized_cosine(window_len, [0.54, 1 - 0.54])


def hann(window_len):
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
    """
    return generalized_cosine(window_len, [0.5, 0.5])


def generalized_cosine(window_len, coefs):
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

    Returns
    -------
    window : numpy array of shape (window_len,)
        The window
    """
    entries = np.linspace(-np.pi, np.pi, window_len)  # (-1)^k * 2pi*n / window_len
    return np.sum([ak * np.cos(k * entries) for k, ak in enumerate(coefs)], axis=0)


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
