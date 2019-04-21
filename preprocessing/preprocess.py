import numpy as np
from numpy.lib.stride_tricks import as_strided

#######################################################################
#                          Signal Resampling                          #
#######################################################################


def batch_resample(X, new_dim, mode="bilinear"):
    """
    Resample each image (or similar grid-based 2D signal) in a batch to
    `new_dim` using the specified resampling strategy.

    Parameters
    ----------
    X : numpy array of shape (n_ex, in_rows, in_cols, in_channels)
        An input image volume
    new_dim : 2-tuple of (out_rows, out_cols)
        The dimension to resample each image to
    mode : str
        The resampling strategy to employ. Valid entries are {'bilinear',
        'neighbor'}

    Returns
    -------
    resampled : numpy array of shape (n_ex, out_rows, out_cols, in_channels)
        The resampled image volume
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
    Estimates of the pixel values at the coordinates (x, y) in X using a
    nearest neighbor interpolation strategy. Assumes the current entries
    in X reflect equally-spaced samples from a 2D integer grid.

    Parameters
    ----------
    X : numpy array of shape (in_rows, in_cols, in_channels)
        An input image sampled along a grid of `in_rows` by `in_cols`.
    x : list of length k
        A list of x-coordinates for the samples we wish to generate
    y : list of length k
        A list of y-coordinates for the samples we wish to generate

    Returns
    -------
    samples : numpy array of shape (k, in_channels)
        The samples for each (x,y) coordinate computed via nearest neighbor
        interpolation
    """
    nx, ny = np.around(x), np.around(y)
    nx = np.clip(nx, 0, X.shape[1] - 1).astype(int)
    ny = np.clip(ny, 0, X.shape[0] - 1).astype(int)
    return X[ny, nx, :]


def nn_interpolate_1D(X, t):
    """
    Estimates of the signal values at X[t] using a nearest neighbor
    interpolation strategy.

    Parameters
    ----------
    X : numpy array of shape (in_length, in_channels)
        An input image sampled along an integer `in_length`
    t : list of length k
        A list of coordinates for the samples we wish to generate

    Returns
    -------
    samples : numpy array of shape (k, in_channels)
        The samples for each (x,y) coordinate computed via nearest neighbor
        interpolation
    """
    nt = np.clip(np.around(t), 0, X.shape[0] - 1).astype(int)
    return X[nt, :]


def bilinear_interpolate(X, x, y):
    """
    Estimates of the pixel values at the coordinates (x, y) in X via bilinear
    interpolation. Assumes the current entries in X reflect equally-spaced
    samples from a 2D integer grid.

    Modified from: https://bit.ly/2NMb1Dr

    Parameters
    ----------
    X : numpy array of shape (in_rows, in_cols, in_channels)
        An input image sampled along a grid of `in_rows` by `in_cols`.
    x : list of length k
        A list of x-coordinates for the samples we wish to generate
    y : list of length k
        A list of y-coordinates for the samples we wish to generate

    Returns
    -------
    samples : list of length (k, in_channels)
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


def DFT(frame, fs=44000):
    """
    A naive implementation of the 1D discrete Fourier transform (DFT) for
    a real-valued input.

    Parameters
    ----------
    frame : numpy array of shape (N,)
        A signal frame consisting of N samples
    fs : int (defalt: 44000)
        The sample rate/frequency for the signal

    Returns
    -------
    spectrum : numpy array of shape (N,)
        The coefficients of the frequency spectrum for `frame`, including
        imaginary components.
    bins : numpy array of shape (N,)
        The frequency bin centers associated with each coefficient in the
        dft spectrum.
    """
    N = len(frame)  # window length

    # F[i,j] = coefficient for basis vector i, timestep j
    F = np.arange(N).reshape(1, -1) * np.arange(N).reshape(-1, 1)
    F = np.exp(F * (-1j * 2 * np.pi / N))

    # vdot only operates on vectors (rather than ndarrays), so we have to
    # loop over each basis vector in F explicitly
    spectrum = np.array([np.vdot(f, frame) for f in F])

    # calc frequency bin centers
    l, r = (1 + (N - 1) / 2, (1 - N) / 2) if N % 2 else (N / 2, -N / 2)
    freq_bins = np.r_[np.arange(l), np.arange(r, 0)] * fs / N
    return spectrum, freq_bins


#######################################################################
#                              DSP Utils                              #
#######################################################################


def to_frames(x, frame_width, stride, writeable=False):
    """
    Convert a 1D signal x into overlapping windows of width `frame_width` using
    a hop length of `stride`.

    NB 1: if (len(x) - frame_width) % stride != 0 then some number of the samples
    in x will be dropped. Specifically,
        n_dropped_frames = len(x) - frame_width - stride * (n_frames - 1)
    where
        n_frames = (len(x) - frame_width) // stride + 1

    NB 2: This method uses low-level stride manipulation to avoid creating an
    additional copy of `x`. The downside is that if `writeable`=True, modifying
    the `frame` output can result in unexpected behavior:

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
    x : numpy array of shape (N,)
        A 1D signal consisting of N samples
    frame_width : int
        The width of a single frame window in samples
    stride : int
        The hop size / number of samples advanced between consecutive frames
    writeable : bool (default: False)
        If set to False, the returned array will be readonly. Otherwise it will
        be writable if `x` was. It is advisable to set this to False if
        possible.

    Returns
    -------
    frame: numpy array of shape (n_frames, frame_width)
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


def autocorrelate(x):
    """
    Autocorrelate a 1D signal `x` with itself.

        auto[k] = sum_n x[n + k] * x[n]

    Parameters
    ----------
    x : numpy array of shape (N,)
        A 1D signal consisting of N samples

    Returns
    -------
    auto : numpy array of shape (N,)
        The autocorrelation of `x` with itself
    """
    N = len(x)
    auto = np.zeros(N)
    for k in range(N):
        for n in range(N - k):
            auto[k] += x[n + k] * x[n]
    return auto
