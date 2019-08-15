import numpy as np


def blackman_harris(window_len, symmetric=False):
    """
    The Blackman-Harris window.

    Notes
    -----
    The Blackman-Harris window is an instance of the more general class of
    cosine-sum windows where `K=3`. Additional coefficients extend the Hamming
    window to further minimize the magnitude of the nearest side-lobe in the
    frequency response.

    .. math::
        \\text{bh}(n) = a_0 - a_1 \cos\left(\\frac{2 \pi n}{N}\\right) +
            a_2 \cos\left(\\frac{4 \pi n }{N}\\right) -
                a_3 \cos\left(\\frac{6 \pi n}{N}\\right)

    where `N` = `window_len` - 1, :math:`a_0` = 0.35875, :math:`a_1` = 0.48829,
    :math:`a_2` = 0.14128, and :math:`a_3` = 0.01168.

    Parameters
    ----------
    window_len : int
        The length of the window in samples. Should be equal to the
        `frame_width` if applying to a windowed signal.
    symmetric : bool
        If False, create a 'periodic' window that can be used in with an FFT /
        in spectral analysis.  If True, generate a symmetric window that can be
        used in, e.g., filter design. Default is False.

    Returns
    -------
    window : :py:class:`ndarray <numpy.ndarray>` of shape `(window_len,)`
        The window
    """
    return generalized_cosine(
        window_len, [0.35875, 0.48829, 0.14128, 0.01168], symmetric
    )


def hamming(window_len, symmetric=False):
    """
    The Hamming window.

    Notes
    -----
    The Hamming window is an instance of the more general class of cosine-sum
    windows where `K=1` and :math:`a_0 = 0.54`. Coefficients selected to
    minimize the magnitude of the nearest side-lobe in the frequency response.

    .. math::

        \\text{hamming}(n) = 0.54 -
            0.46 \cos\left(\\frac{2 \pi n}{\\text{window_len} - 1}\\right)

    Parameters
    ----------
    window_len : int
        The length of the window in samples. Should be equal to the
        `frame_width` if applying to a windowed signal.
    symmetric : bool
        If False, create a 'periodic' window that can be used in with an FFT /
        in spectral analysis.  If True, generate a symmetric window that can be
        used in, e.g., filter design. Default is False.

    Returns
    -------
    window : :py:class:`ndarray <numpy.ndarray>` of shape `(window_len,)`
        The window
    """
    return generalized_cosine(window_len, [0.54, 1 - 0.54], symmetric)


def hann(window_len, symmetric=False):
    """
    The Hann window.

    Notes
    -----
    The Hann window is an instance of the more general class of cosine-sum
    windows where `K=1` and :math:`a_0` = 0.5. Unlike the Hamming window, the
    end points of the Hann window touch zero.

    .. math::

        \\text{hann}(n) = 0.5 - 0.5 \cos\left(\\frac{2 \pi n}{\\text{window_len} - 1}\\right)

    Parameters
    ----------
    window_len : int
        The length of the window in samples. Should be equal to the
        `frame_width` if applying to a windowed signal.
    symmetric : bool
        If False, create a 'periodic' window that can be used in with an FFT /
        in spectral analysis.  If True, generate a symmetric window that can be
        used in, e.g., filter design. Default is False.

    Returns
    -------
    window : :py:class:`ndarray <numpy.ndarray>` of shape `(window_len,)`
        The window
    """
    return generalized_cosine(window_len, [0.5, 0.5], symmetric)


def generalized_cosine(window_len, coefs, symmetric=False):
    """
    The generalized cosine family of window functions.

    Notes
    -----
    The generalized cosine window is a simple weighted sum of cosine terms.

    For :math:`n \in \{0, \ldots, \\text{window_len} \}`:

    .. math::

        \\text{GCW}(n) = \sum_{k=0}^K (-1)^k a_k \cos\left(\\frac{2 \pi k n}{\\text{window_len}}\\right)

    Parameters
    ----------
    window_len : int
        The length of the window in samples. Should be equal to the
        `frame_width` if applying to a windowed signal.
    coefs: list of floats
        The :math:`a_k` coefficient values
    symmetric : bool
        If False, create a 'periodic' window that can be used in with an FFT /
        in spectral analysis.  If True, generate a symmetric window that can be
        used in, e.g., filter design. Default is False.

    Returns
    -------
    window : :py:class:`ndarray <numpy.ndarray>` of shape `(window_len,)`
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
