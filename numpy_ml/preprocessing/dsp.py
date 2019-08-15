import numpy as np
from numpy.lib.stride_tricks import as_strided

from ..utils.windows import WindowInitializer

#######################################################################
#                          Signal Resampling                          #
#######################################################################


def batch_resample(X, new_dim, mode="bilinear"):
    """
    Resample each image (or similar grid-based 2D signal) in a batch to
    `new_dim` using the specified resampling strategy.

    Parameters
    ----------
    X : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, in_rows, in_cols, in_channels)`
        An input image volume
    new_dim : 2-tuple of `(out_rows, out_cols)`
        The dimension to resample each image to
    mode : {'bilinear', 'neighbor'}
        The resampling strategy to employ. Default is 'bilinear'.

    Returns
    -------
    resampled : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, out_rows, out_cols, in_channels)`
        The resampled image volume.
    """
    if mode == "bilinear":
        interpolate = bilinear_interpolate
    elif mode == "neighbor":
        interpolate = nn_interpolate_2D
    else:
        raise NotImplementedError("Unrecognized resampling mode: {}".format(mode))

    out_rows, out_cols = new_dim
    n_ex, in_rows, in_cols, n_in = X.shape

    # compute coordinates to resample
    x = np.tile(np.linspace(0, in_cols - 2, out_cols), out_rows)
    y = np.repeat(np.linspace(0, in_rows - 2, out_rows), out_cols)

    # resample each image
    resampled = []
    for i in range(n_ex):
        r = interpolate(X[i, ...], x, y)
        r = r.reshape(out_rows, out_cols, n_in)
        resampled.append(r)
    return np.dstack(resampled)


def nn_interpolate_2D(X, x, y):
    """
    Estimates of the pixel values at the coordinates (x, y) in `X` using a
    nearest neighbor interpolation strategy.

    Notes
    -----
    Assumes the current entries in `X` reflect equally-spaced samples from a 2D
    integer grid.

    Parameters
    ----------
    X : :py:class:`ndarray <numpy.ndarray>` of shape `(in_rows, in_cols, in_channels)`
        An input image sampled along a grid of `in_rows` by `in_cols`.
    x : list of length `k`
        A list of x-coordinates for the samples we wish to generate
    y : list of length `k`
        A list of y-coordinates for the samples we wish to generate

    Returns
    -------
    samples : :py:class:`ndarray <numpy.ndarray>` of shape `(k, in_channels)`
        The samples for each (x,y) coordinate computed via nearest neighbor
        interpolation
    """
    nx, ny = np.around(x), np.around(y)
    nx = np.clip(nx, 0, X.shape[1] - 1).astype(int)
    ny = np.clip(ny, 0, X.shape[0] - 1).astype(int)
    return X[ny, nx, :]


def nn_interpolate_1D(X, t):
    """
    Estimates of the signal values at `X[t]` using a nearest neighbor
    interpolation strategy.

    Parameters
    ----------
    X : :py:class:`ndarray <numpy.ndarray>` of shape `(in_length, in_channels)`
        An input image sampled along an integer `in_length`
    t : list of length `k`
        A list of coordinates for the samples we wish to generate

    Returns
    -------
    samples : :py:class:`ndarray <numpy.ndarray>` of shape `(k, in_channels)`
        The samples for each (x,y) coordinate computed via nearest neighbor
        interpolation
    """
    nt = np.clip(np.around(t), 0, X.shape[0] - 1).astype(int)
    return X[nt, :]


def bilinear_interpolate(X, x, y):
    """
    Estimates of the pixel values at the coordinates (x, y) in `X` via bilinear
    interpolation.

    Notes
    -----
    Assumes the current entries in X reflect equally-spaced
    samples from a 2D integer grid.

    Modified from https://bit.ly/2NMb1Dr

    Parameters
    ----------
    X : :py:class:`ndarray <numpy.ndarray>` of shape `(in_rows, in_cols, in_channels)`
        An input image sampled along a grid of `in_rows` by `in_cols`.
    x : list of length `k`
        A list of x-coordinates for the samples we wish to generate
    y : list of length `k`
        A list of y-coordinates for the samples we wish to generate

    Returns
    -------
    samples : list of length `(k, in_channels)`
        The samples for each (x,y) coordinate computed via bilinear
        interpolation
    """
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    x1 = x0 + 1
    y1 = y0 + 1

    x0 = np.clip(x0, 0, X.shape[1] - 1)
    y0 = np.clip(y0, 0, X.shape[0] - 1)
    x1 = np.clip(x1, 0, X.shape[1] - 1)
    y1 = np.clip(y1, 0, X.shape[0] - 1)

    Ia = X[y0, x0, :].T
    Ib = X[y1, x0, :].T
    Ic = X[y0, x1, :].T
    Id = X[y1, x1, :].T

    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    return (Ia * wa).T + (Ib * wb).T + (Ic * wc).T + (Id * wd).T


#######################################################################
#                        Fourier Decomposition                        #
#######################################################################


def DCT(frame, orthonormal=True):
    """
    A naive :math:`O(N^2)` implementation of the 1D discrete cosine transform-II
    (DCT-II).

    Notes
    -----
    For a signal :math:`\mathbf{x} = [x_1, \ldots, x_N]` consisting of `N`
    samples, the `k` th DCT coefficient, :math:`c_k`, is

    .. math::

        c_k = 2 \sum_{n=0}^{N-1} x_n \cos(\pi k (2 n + 1) / (2 N))

    where `k` ranges from :math:`0, \ldots, N-1`.

    The DCT is highly similar to the DFT -- whereas in a DFT the basis
    functions are sinusoids, in a DCT they are restricted solely to cosines. A
    signal's DCT representation tends to have more of its energy concentrated
    in a smaller number of coefficients when compared to the DFT, and is thus
    commonly used for signal compression. [1]

    .. [1] Smoother signals can be accurately approximated using fewer DFT / DCT
       coefficients, resulting in a higher compression ratio. The DCT naturally
       yields a continuous extension at the signal boundaries due its use of
       even basis functions (cosine). This in turn produces a smoother
       extension in comparison to DFT or DCT approximations, resulting in a
       higher compression.

    Parameters
    ----------
    frame : :py:class:`ndarray <numpy.ndarray>` of shape `(N,)`
        A signal frame consisting of N samples
    orthonormal : bool
        Scale to ensure the coefficient vector is orthonormal. Default is True.

    Returns
    -------
    dct : :py:class:`ndarray <numpy.ndarray>` of shape `(N,)`
        The discrete cosine transform of the samples in `frame`.
    """
    N = len(frame)
    out = np.zeros_like(frame)
    for k in range(N):
        for (n, xn) in enumerate(frame):
            out[k] += xn * np.cos(np.pi * k * (2 * n + 1) / (2 * N))
        scale = np.sqrt(1 / (4 * N)) if k == 0 else np.sqrt(1 / (2 * N))
        out[k] *= 2 * scale if orthonormal else 2
    return out


def __DCT2(frame):
    """Currently broken"""
    N = len(frame)  # window length

    k = np.arange(N, dtype=float)
    F = k.reshape(1, -1) * k.reshape(-1, 1)
    K = np.divide(F, k, out=np.zeros_like(F), where=F != 0)

    FC = np.cos(F * np.pi / N + K * np.pi / 2 * N)
    return 2 * (FC @ frame)


def DFT(frame, positive_only=True):
    """
    A naive :math:`O(N^2)` implementation of the 1D discrete Fourier transform (DFT).

    Notes
    -----
    The Fourier transform decomposes a signal into a linear combination of
    sinusoids (ie., basis elements in the space of continuous periodic
    functions).  For a sequence :math:`\mathbf{x} = [x_1, \ldots, x_N]` of N
    evenly spaced samples, the `k` th DFT coefficient is given by:

    .. math::

        c_k = \sum_{n=0}^{N-1} x_n \exp(-2 \pi i k n / N)

    where `i` is the imaginary unit, `k` is an index ranging from `0, ..., N-1`,
    and :math:`X_k` is the complex coefficient representing the phase
    (imaginary part) and amplitude (real part) of the `k` th sinusoid in the
    DFT spectrum. The frequency of the `k` th sinusoid is :math:`(k 2 \pi / N)`
    radians per sample.

    When applied to a real-valued input, the negative frequency terms are the
    complex conjugates of the positive-frequency terms and the overall spectrum
    is symmetric (excluding the first index, which contains the zero-frequency
    / intercept term).

    Parameters
    ----------
    frame : :py:class:`ndarray <numpy.ndarray>` of shape `(N,)`
        A signal frame consisting of N samples
    positive_only : bool
        Whether to only return the coefficients for the positive frequency
        terms. Default is True.

    Returns
    -------
    spectrum : :py:class:`ndarray <numpy.ndarray>` of shape `(N,)` or `(N // 2 + 1,)` if `real_only`
        The coefficients of the frequency spectrum for `frame`, including
        imaginary components.
    """
    N = len(frame)  # window length

    # F[i,j] = coefficient for basis vector i, timestep j (i.e., k * n)
    F = np.arange(N).reshape(1, -1) * np.arange(N).reshape(-1, 1)
    F = np.exp(F * (-1j * 2 * np.pi / N))

    # vdot only operates on vectors (rather than ndarrays), so we have to
    # loop over each basis vector in F explicitly
    spectrum = np.array([np.vdot(f, frame) for f in F])
    return spectrum[: (N // 2) + 1] if positive_only else spectrum


def dft_bins(N, fs=44000, positive_only=True):
    """
    Calc the frequency bin centers for a DFT with `N` coefficients.

    Parameters
    ----------
    N : int
        The number of frequency bins in the DFT
    fs : int
        The sample rate/frequency of the signal (in Hz). Default is 44000.
    positive_only : bool
        Whether to only return the bins for the positive frequency
        terms. Default is True.

    Returns
    -------
    bins : :py:class:`ndarray <numpy.ndarray>` of shape `(N,)` or `(N // 2 + 1,)` if `positive_only`
        The frequency bin centers associated with each coefficient in the
        DFT spectrum
    """
    if positive_only:
        freq_bins = np.linspace(0, fs / 2, 1 + N // 2, endpoint=True)
    else:
        l, r = (1 + (N - 1) / 2, (1 - N) / 2) if N % 2 else (N / 2, -N / 2)
        freq_bins = np.r_[np.arange(l), np.arange(r, 0)] * fs / N
    return freq_bins


def magnitude_spectrum(frames):
    """
    Compute the magnitude spectrum (i.e., absolute value of the DFT spectrum)
    for each frame in `frames`. Assumes each frame is real-valued only.

    Parameters
    ----------
    frames : :py:class:`ndarray <numpy.ndarray>` of shape `(M, N)`
        A sequence of `M` frames each consisting of `N` samples

    Returns
    -------
    magnitude_spec : :py:class:`ndarray <numpy.ndarray>` of shape `(M, N // 2 + 1)`
        The magnitude spectrum for each frame in `frames`. Only includes the
        coefficients for the positive spectrum frequencies.
    """
    return np.vstack([np.abs(DFT(frame, positive_only=True)) for frame in frames])


def power_spectrum(frames, scale=False):
    """
    Compute the power spectrum for a signal represented as a collection of
    frames. Assumes each frame is real-valued only.

    The power spectrum is simply the square of the magnitude spectrum, possibly
    scaled by the number of FFT bins. It measures how the energy of the signal
    is distributed over the frequency domain.

    Parameters
    ----------
    frames : :py:class:`ndarray <numpy.ndarray>` of shape `(M, N)`
        A sequence of `M` frames each consisting of `N` samples
    scale : bool
        Whether the scale by the number of DFT bins. Default is False.

    Returns
    -------
    power_spec : :py:class:`ndarray <numpy.ndarray>` of shape `(M, N // 2 + 1)`
        The power spectrum for each frame in `frames`. Only includes the
        coefficients for the positive spectrum frequencies.
    """
    scaler = frames.shape[1] // 2 + 1 if scale else 1
    return (1 / scaler) * magnitude_spectrum(frames) ** 2


#######################################################################
#                       Preprocessing Utils                           #
#######################################################################


def to_frames(x, frame_width, stride, writeable=False):
    """
    Convert a 1D signal x into overlapping windows of width `frame_width` using
    a hop length of `stride`.

    Notes
    -----
    If ``(len(x) - frame_width) % stride != 0`` then some number of the samples
    in x will be dropped. Specifically::

        n_dropped_frames = len(x) - frame_width - stride * (n_frames - 1)

    where::

        n_frames = (len(x) - frame_width) // stride + 1

    This method uses low-level stride manipulation to avoid creating an
    additional copy of `x`. The downside is that if ``writeable`=True``,
    modifying the `frame` output can result in unexpected behavior:

        >>> out = to_frames(np.arange(6), 5, 1)
        >>> out
        array([[0, 1, 2, 3, 4],
               [1, 2, 3, 4, 5]])
        >>> out[0, 1] = 99
        >>> out
        array([[ 0, 99,  2,  3,  4],
               [99,  2,  3,  4,  5]])

    Parameters
    ----------
    x : :py:class:`ndarray <numpy.ndarray>` of shape `(N,)`
        A 1D signal consisting of N samples
    frame_width : int
        The width of a single frame window in samples
    stride : int
        The hop size / number of samples advanced between consecutive frames
    writeable : bool
        If set to False, the returned array will be readonly. Otherwise it will
        be writable if `x` was. It is advisable to set this to False whenever
        possible to avoid unexpected behavior (see NB 2 above). Default is False.

    Returns
    -------
    frame: :py:class:`ndarray <numpy.ndarray>` of shape `(n_frames, frame_width)`
        The collection of overlapping frames stacked into a matrix
    """
    assert x.ndim == 1
    assert stride >= 1
    assert len(x) >= frame_width

    # get the size for an element in x in bits
    byte = x.itemsize
    n_frames = (len(x) - frame_width) // stride + 1
    return as_strided(
        x,
        shape=(n_frames, frame_width),
        strides=(byte * stride, byte),
        writeable=writeable,
    )


def autocorrelate1D(x):
    """
    Autocorrelate a 1D signal `x` with itself.

    Notes
    -----
    The `k` th term in the 1 dimensional autocorrelation is

    .. math::

        a_k = \sum_n x_{n + k} x_n

    NB. This is a naive :math:`O(N^2)` implementation.  For a faster :math:`O(N
    \log N)` approach using the FFT, see [1].

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Autocorrelation#Efficient%computation

    Parameters
    ----------
    x : :py:class:`ndarray <numpy.ndarray>` of shape `(N,)`
        A 1D signal consisting of N samples

    Returns
    -------
    auto : :py:class:`ndarray <numpy.ndarray>` of shape `(N,)`
        The autocorrelation of `x` with itself
    """
    N = len(x)
    auto = np.zeros(N)
    for k in range(N):
        for n in range(N - k):
            auto[k] += x[n + k] * x[n]
    return auto


#######################################################################
#                               Filters                               #
#######################################################################


def preemphasis(x, alpha):
    """
    Increase the amplitude of high frequency bands + decrease the amplitude of
    lower bands.

    Notes
    -----
    Preemphasis filtering is (was?) a common transform in speech processing,
    where higher frequencies tend to be more useful during signal
    disambiguation.

    .. math::

        \\text{preemphasis}( x_t ) = x_t - \\alpha x_{t-1}

    Parameters
    ----------
    x : :py:class:`ndarray <numpy.ndarray>` of shape `(N,)`
        A 1D signal consisting of `N` samples
    alpha : float in [0, 1)
        The preemphasis coefficient. A value of 0 corresponds to no
        filtering

    Returns
    -------
    out : :py:class:`ndarray <numpy.ndarray>` of shape `(N,)`
        The filtered signal
    """
    return np.concatenate([x[:1], x[1:] - alpha * x[:-1]])


def cepstral_lifter(mfccs, D):
    """
    A simple sinusoidal filter applied in the Mel-frequency domain.

    Notes
    -----
    Cepstral lifting helps to smooth the spectral envelope and dampen the
    magnitude of the higher MFCC coefficients while keeping the other
    coefficients unchanged. The filter function is:

    .. math::

        \\text{lifter}( x_n ) = x_n \left(1 + \\frac{D \sin(\pi n / D)}{2}\\right)

    Parameters
    ----------
    mfccs : :py:class:`ndarray <numpy.ndarray>` of shape `(G, C)`
        Matrix of Mel cepstral coefficients. Rows correspond to frames, columns
        to cepstral coefficients
    D : int in :math:`[0, +\infty]`
        The filter coefficient. 0 corresponds to no filtering, larger values
        correspond to greater amounts of smoothing

    Returns
    -------
    out : :py:class:`ndarray <numpy.ndarray>` of shape `(G, C)`
        The lifter'd MFCC coefficients
    """
    if D == 0:
        return mfccs
    n = np.arange(mfccs.shape[1])
    return mfccs * (1 + (D / 2) * np.sin(np.pi * n / D))


def mel_spectrogram(
    x,
    window_duration=0.025,
    stride_duration=0.01,
    mean_normalize=True,
    window="hamming",
    n_filters=20,
    center=True,
    alpha=0.95,
    fs=44000,
):
    """
    Apply the Mel-filterbank to the power spectrum for a signal `x`.

    Notes
    -----
    The Mel spectrogram is the projection of the power spectrum of the framed
    and windowed signal onto the basis set provided by the Mel filterbank.

    Parameters
    ----------
    x : :py:class:`ndarray <numpy.ndarray>` of shape `(N,)`
        A 1D signal consisting of N samples
    window_duration : float
        The duration of each frame / window (in seconds). Default is 0.025.
    stride_duration : float
        The duration of the hop between consecutive windows (in seconds).
        Default is 0.01.
    mean_normalize : bool
        Whether to subtract the coefficient means from the final filter values
        to improve the signal-to-noise ratio. Default is True.
    window : {'hamming', 'hann', 'blackman_harris'}
        The windowing function to apply to the signal before FFT. Default is
        'hamming'.
    n_filters : int
        The number of mel filters to include in the filterbank. Default is 20.
    center : bool
        Whether to the `k` th frame of the signal should *begin* at index ``x[k *
        stride_len]`` (center = False) or be *centered* at ``x[k * stride_len]``
        (center = True). Default is False.
    alpha : float in [0, 1)
        The coefficient for the preemphasis filter. A value of 0 corresponds to
        no filtering. Default is 0.95.
    fs : int
        The sample rate/frequency for the signal. Default is 44000.

    Returns
    -------
    filter_energies : :py:class:`ndarray <numpy.ndarray>` of shape `(G, n_filters)`
        The (possibly mean_normalized) power for each filter in the Mel
        filterbank (i.e., the Mel spectrogram). Rows correspond to frames,
        columns to filters
    energy_per_frame : :py:class:`ndarray <numpy.ndarray>` of shape `(G,)`
        The total energy in each frame of the signal
    """
    eps = np.finfo(float).eps
    window_fn = WindowInitializer()(window)

    stride = round(stride_duration * fs)
    frame_width = round(window_duration * fs)
    N = frame_width

    # add a preemphasis filter to the raw signal
    x = preemphasis(x, alpha)

    # convert signal to overlapping frames and apply a window function
    x = np.pad(x, N // 2, "reflect") if center else x
    frames = to_frames(x, frame_width, stride, fs)

    window = np.tile(window_fn(frame_width), (frames.shape[0], 1))
    frames = frames * window

    # compute the power spectrum
    power_spec = power_spectrum(frames)
    energy_per_frame = np.sum(power_spec, axis=1)
    energy_per_frame[energy_per_frame == 0] = eps

    # compute the power at each filter in the Mel filterbank
    fbank = mel_filterbank(N, n_filters=n_filters, fs=fs)
    filter_energies = power_spec @ fbank.T
    filter_energies -= np.mean(filter_energies, axis=0) if mean_normalize else 0
    filter_energies[filter_energies == 0] = eps
    return filter_energies, energy_per_frame


#######################################################################
#                       Mel-Frequency Features                        #
#######################################################################


def mfcc(
    x,
    fs=44000,
    n_mfccs=13,
    alpha=0.95,
    center=True,
    n_filters=20,
    window="hann",
    normalize=True,
    lifter_coef=22,
    stride_duration=0.01,
    window_duration=0.025,
    replace_intercept=True,
):
    """
    Compute the Mel-frequency cepstral coefficients (MFCC) for a signal.

    Notes
    -----
    Computing MFCC features proceeds in the following stages:

        1. Convert the signal into overlapping frames and apply a window fn
        2. Compute the power spectrum at each frame
        3. Apply the mel filterbank to the power spectra to get mel filterbank powers
        4. Take the logarithm of the mel filterbank powers at each frame
        5. Take the discrete cosine transform (DCT) of the log filterbank
           energies and retain only the first k coefficients to further reduce
           the dimensionality

    MFCCs were developed in the context of HMM-GMM automatic speech recognition
    (ASR) systems and can be used to provide a somewhat speaker/pitch
    invariant representation of phonemes.

    Parameters
    ----------
    x : :py:class:`ndarray <numpy.ndarray>` of shape `(N,)`
        A 1D signal consisting of N samples
    fs : int
        The sample rate/frequency for the signal. Default is 44000.
    n_mfccs : int
        The number of cepstral coefficients to return (including the intercept
        coefficient). Default is 13.
    alpha : float in [0, 1)
        The preemphasis coefficient. A value of 0 corresponds to no
        filtering. Default is 0.95.
    center : bool
        Whether to the kth frame of the signal should *begin* at index ``x[k *
        stride_len]`` (center = False) or be *centered* at ``x[k * stride_len]``
        (center = True). Default is True.
    n_filters : int
        The number of filters to include in the Mel filterbank. Default is 20.
    normalize : bool
        Whether to mean-normalize the MFCC values. Default is True.
    lifter_coef : int in :math:[0, + \infty]`
        The cepstral filter coefficient. 0 corresponds to no filtering, larger
        values correspond to greater amounts of smoothing. Default is 22.
    window : {'hamming', 'hann', 'blackman_harris'}
        The windowing function to apply to the signal before taking the DFT.
        Default is 'hann'.
    stride_duration : float
        The duration of the hop between consecutive windows (in seconds).
        Default is 0.01.
    window_duration : float
        The duration of each frame / window (in seconds). Default is 0.025.
    replace_intercept : bool
        Replace the first MFCC coefficient (the intercept term) with the
        log of the total frame energy instead. Default is True.

    Returns
    -------
    mfccs : :py:class:`ndarray <numpy.ndarray>` of shape `(G, C)`
        Matrix of Mel-frequency cepstral coefficients. Rows correspond to
        frames, columns to cepstral coefficients
    """
    # map the power spectrum for the (framed + windowed representation of) `x`
    # onto the mel scale
    filter_energies, frame_energies = mel_spectrogram(
        x=x,
        fs=fs,
        alpha=alpha,
        center=center,
        window=window,
        n_filters=n_filters,
        mean_normalize=False,
        window_duration=window_duration,
        stride_duration=stride_duration,
    )

    log_energies = 10 * np.log10(filter_energies)

    # perform a DCT on the log-mel coefficients to further reduce the data
    # dimensionality -- the early DCT coefficients will capture the majority of
    # the data, allowing us to discard coefficients > n_mfccs
    mfccs = np.array([DCT(frame) for frame in log_energies])[:, :n_mfccs]

    mfccs = cepstral_lifter(mfccs, D=lifter_coef)
    mfccs -= np.mean(mfccs, axis=0) if normalize else 0

    if replace_intercept:
        # the 0th MFCC coefficient doesn't tell us anything about the spectrum;
        # replace it with the log of the frame energy for something more
        # informative
        mfccs[:, 0] = np.log(frame_energies)
    return mfccs


def mel2hz(mel, formula="htk"):
    """
    Convert the mel-scale representation of a signal into Hz

    Parameters
    ----------
    mel : :py:class:`ndarray <numpy.ndarray>` of shape `(N, \*)`
        An array of mel frequencies to convert
    formula : {"htk", "slaney"}
        The Mel formula to use. "htk" uses the formula used by the Hidden
        Markov Model Toolkit, and described in O'Shaughnessy (1987). "slaney"
        uses the formula used in the MATLAB auditory toolbox (Slaney, 1998).
        Default is 'htk'

    Returns
    -------
    hz : :py:class:`ndarray <numpy.ndarray>` of shape `(N, \*)`
        The frequencies of the items in `mel`, in Hz
    """
    fstr = "formula must be either 'htk' or 'slaney' but got '{}'"
    assert formula in ["htk", "slaney"], fstr.format(formula)
    if formula == "htk":
        return 700 * (10 ** (mel / 2595) - 1)
    raise NotImplementedError("slaney")


def hz2mel(hz, formula="htk"):
    """
    Convert the frequency representaiton of a signal in Hz into the mel scale.

    Parameters
    ----------
    hz : :py:class:`ndarray <numpy.ndarray>` of shape `(N, \*)`
        The frequencies of the items in `mel`, in Hz
    formula : {"htk", "slaney"}
        The Mel formula to use. "htk" uses the formula used by the Hidden
        Markov Model Toolkit, and described in O'Shaughnessy (1987). "slaney"
        uses the formula used in the MATLAB auditory toolbox (Slaney, 1998).
        Default is 'htk'.

    Returns
    -------
    mel : :py:class:`ndarray <numpy.ndarray>` of shape `(N, \*)`
        An array of mel frequencies to convert.
    """
    fstr = "formula must be either 'htk' or 'slaney' but got '{}'"
    assert formula in ["htk", "slaney"], fstr.format(formula)

    if formula == "htk":
        return 2595 * np.log10(1 + hz / 700)
    raise NotImplementedError("slaney")


def mel_filterbank(
    N, n_filters=20, fs=44000, min_freq=0, max_freq=None, normalize=True
):
    """
    Compute the filters in a Mel filterbank and return the corresponding
    transformation matrix

    Notes
    -----
    The Mel scale is a perceptual scale designed to simulate the way the human
    ear works. Pitches judged by listeners to be equal in perceptual /
    psychological distance have equal distance on the Mel scale.  Practically,
    this corresponds to a scale with higher resolution at low frequencies and
    lower resolution at higher (> 500 Hz) frequencies.

    Each filter in the Mel filterbank is triangular with a response of 1 at its
    center and a linear decay on both sides until it reaches the center
    frequency of the next adjacent filter.

    This implementation is based on code in the (superb) LibROSA package [1].

    References
    ----------
    .. [1] McFee et al. (2015). "librosa: Audio and music signal analysis in
       Python", *Proceedings of the 14th Python in Science Conference*
       https://librosa.github.io

    Parameters
    ----------
    N : int
        The number of DFT bins
    n_filters : int
        The number of mel filters to include in the filterbank. Default is 20.
    min_freq : int
        Minimum filter frequency (in Hz). Default is 0.
    max_freq : int
        Maximum filter frequency (in Hz). Default is 0.
    fs : int
        The sample rate/frequency for the signal. Default is 44000.
    normalize : bool
        If True, scale the Mel filter weights by their area in Mel space.
        Default is True.

    Returns
    -------
    fbank : :py:class:`ndarray <numpy.ndarray>` of shape `(n_filters, N // 2 + 1)`
        The mel-filterbank transformation matrix. Rows correspond to filters,
        columns to DFT bins.
    """
    max_freq = fs / 2 if max_freq is None else max_freq
    min_mel, max_mel = hz2mel(min_freq), hz2mel(max_freq)

    fbank = np.zeros((n_filters, N // 2 + 1))

    # uniformly spaced values on the mel scale, translated back into Hz
    mel_bins = mel2hz(np.linspace(min_mel, max_mel, n_filters + 2))

    # the centers of the frequency bins for the DFT
    hz_bins = dft_bins(N, fs)

    mel_spacing = np.diff(mel_bins)

    # ramps[i] = mel_bins[i] - hz_bins
    ramps = mel_bins.reshape(-1, 1) - hz_bins.reshape(1, -1)
    for i in range(n_filters):
        # calc the filter values on the left and right across the bins ...
        left = -ramps[i] / mel_spacing[i]
        right = ramps[i + 2] / mel_spacing[i + 1]

        # .. and set them zero when they cross the x-axis
        fbank[i] = np.maximum(0, np.minimum(left, right))

    if normalize:
        energy_norm = 2.0 / (mel_bins[2 : n_filters + 2] - mel_bins[:n_filters])
        fbank *= energy_norm[:, np.newaxis]

    return fbank
