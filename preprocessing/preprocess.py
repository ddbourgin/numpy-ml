import numpy as np

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
