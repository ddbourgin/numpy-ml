import numpy as np


def calc_pad_dims(X_shape, out_dim, kernel_shape, stride):
    """
    Compute the padding necessary to ensure that convolving X with a 2D kernel
    of shape `kernel_shape` and stride `stride` produces outputs with dimension
    `out_dim`.

    Parameters
    ----------
    X_shape : tuple of (n_ex, in_rows, in_cols, in_channels)
        Dimensions of the input volume. Padding is applied to `in_rows` and
        `in_cols`.
    out_dim : tuple of (out_rows, out_cols)
        The desired dimension of an output example after applying the
        convolution.
    kernel_shape : 2-tuple
        The dimension of the 2D convolution kernel.
    stride : int
        The stride for the convolution kernel.

    Returns
    -------
    padding_dims : 4-tuple
        Padding dims for X. Organized as (left, right, up, down)
    """
    if not isinstance(X_shape, tuple):
        raise ValueError("`X_shape` must be of type tuple")

    if not isinstance(out_dim, tuple):
        raise ValueError("`out_dim` must be of type tuple")

    if not isinstance(kernel_shape, tuple):
        raise ValueError("`kernel_shape` must be of type tuple")

    if not isinstance(stride, int):
        raise ValueError("`stride` must be of type int")

    fr, fc = kernel_shape
    out_rows, out_cols = out_dim
    n_ex, in_rows, in_cols, n_in = X_shape

    pr = int((stride * (out_rows - 1) + fr - in_rows) / 2)
    pc = int((stride * (out_cols - 1) + fc - in_cols) / 2)

    out_rows1 = int(1 + (in_rows + 2 * pr - fr) / stride)
    out_cols1 = int(1 + (in_cols + 2 * pc - fc) / stride)

    # add asymmetric padding pixels to right / bottom
    pr1, pr2 = pr, pr
    if out_rows1 == out_rows - 1:
        pr1, pr2 = pr, pr + 1
    elif out_rows1 != out_rows:
        raise AssertionError

    pc1, pc2 = pc, pc
    if out_cols1 == out_cols - 1:
        pc1, pc2 = pc, pc + 1
    elif out_cols1 != out_cols:
        raise AssertionError

    if any(np.array([pr1, pr2, pc1, pc2]) < 0):
        raise ValueError(
            "Padding cannot be less than 0. Yours: {}".format((pr1, pr2, pc1, pc2))
        )

    return (pr1, pr2, pc1, pc2)


def pad2D(X, p, kernel_shape=None, stride=None):
    """
    Two-dimensional zero-padding utility.

    Parameters
    ----------
    X : numpy array of shape (n_ex, in_rows, in_cols, in_channels)
        Input volume. Padding is applied to `in_rows` and `in_cols`.
    p : tuple, int, or 'same'
        The padding amount. If 'same', add padding to ensure that the output of
        a 2D convolution with a kernel of `kernel_shape` and stride `stride`
        produces an output volume of the same dimensions as the input.  If
        2-tuple, specifies the number of padding rows and colums to add *on both
        sides* of the rows/columns in X. If 4-tuple, specifies the number of
        rows/columns to add to the left, right, top, and bottom of the input
        volume.
    kernel_shape : tuple (default: None)
        The dimension of the 2D convolution kernel. Only relevant if p='same'.
    stride : int (default: None)
        The stride for the convolution kernel. Only relevant if p='same'.

    Returns
    -------
    X_pad : numpy array of shape (n_ex, padded_in_rows, padded_in_cols,
    in_channels)
        The padded output volume
    p : 4-tuple
        The number of 0-padded rows added to the (left, right, top, bottom) of X
    """
    if isinstance(p, int):
        p = (p, p, p, p)

    if isinstance(p, tuple):
        if len(p) == 2:
            p = (p[0], p[0], p[1], p[1])

        X_pad = np.pad(
            X,
            pad_width=((0, 0), (p[0], p[1]), (p[2], p[3]), (0, 0)),
            mode="constant",
            constant_values=0,
        )

    if p == "same" and kernel_shape and stride is not None:
        #  p = same_pad_dims(X, kernel_shape, stride)
        p = calc_pad_dims(X.shape, X.shape[1:3], kernel_shape, stride)
        X_pad, p = pad2D(X, p)
    return X_pad, p


def conv2D(X, K, s, p, b):
    """
    Perform a 2D "convolution" (technically, cross-correlation) of input X
    with a kernel K with and optional offset/bias term b.

    Parameters
    ----------
    X : numpy array of shape (in_rows, in_cols, n_in)
        A single input example
    K: numpy array of shape (kernel_size, kernel_size, n_in)
        A single convolution kernel
    s : int
        The stride of the convolution kernel
    p : tuple of length 4
        The number of 0-padded rows added to X on the (left, right, top, bottom)
    b : float
        A bias term for K

    Returns
    -------
    conv : numpy array of shape (out_rows, out_cols)
        The covolution of X with K plus a bias, b. conv = xi * wc + bc
    """
    fr, fc, n_in = K.shape
    pr1, pr2, pc1, pc2 = p
    in_rows, in_cols, n_in = X.shape

    # compute the dimensions of the convolution output (note that `in_rows` and
    # `in_cols` already include padding so we don't add it in again)
    out_rows = np.floor(1 + (in_rows - fr) / s).astype(int)
    out_cols = np.floor(1 + (in_cols - fc) / s).astype(int)

    out = np.zeros((out_rows, out_cols))
    for i in range(out_rows):
        for j in range(out_cols):
            i0, i1 = i * s, (i * s) + fr
            j0, j1 = j * s, (j * s) + fc

            window = X[i0:i1, j0:j1, :]
            out[i, j] = np.sum(window * K) + b
    return out


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


def batch_resample(X, new_dim, mode="bilinear"):
    """
    Resample each image in a batch to `new_dim`.

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
