import numpy as np


def batch_resample(X, new_dim, mode="bilinear"):
    """
    Resample each image in a batch to `new_dim` using the specified resampling
    strategy.

    Parameters
    ----------
    X : numpy array of shape (n_ex, in_rows, in_cols, in_channels)
        An input image volume
    new_dim : 2-tuple of (out_rows, out_cols)
        The dimension to resample each image to
    mode : str
        The resampling strategy to employ. Valid entries are {'bilinear'}

    Returns
    -------
    resampled : numpy array of shape (n_ex, out_rows, out_cols, in_channels)
        The resampled image volume
    """
    if mode != "bilinear":
        raise NotImplementedError

    out_rows, out_cols = new_dim
    n_ex, in_rows, in_cols, n_in = X.shape

    # compute coordinates to resample
    x = np.tile(np.linspace(0, in_cols - 2, out_cols), out_rows)
    y = np.repeat(np.linspace(0, in_rows - 2, out_rows), out_cols)

    # resample each image
    resampled = []
    for i in range(n_ex):
        r = bilinear_interpolate(X[i, ...], x, y)
        r = r.reshape(out_rows, out_cols, n_in)
        resampled.append(r)
    return np.dstack(resampled)


def bilinear_interpolate(X, x, y):
    """
    Estimates of the pixel values at the coordinates (x, y) in X via bilinear
    interpolation. Modified from: https://bit.ly/2NMb1Dr

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
    samples : list of length k
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


def DFT(signal, sample_rate=440000):
    """
    1D discrete Fourier transform (DFT) for real-valued input.
    """
    N = len(signal)  # window length
    d = 1 / sample_rate  # sample spacing

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

    pos = list(range(0, (N - 1) / 2 if N % 2 else (N / 2) - 1))
    neg = list(range(-(N - 1) / 2 if N % 2 else -N / 2, -1))
    freq_bins = np.array(pos + neg) * d * N
    return spectrum, freq_bins
