import numpy as np

#######################################################################
#                           Training Utils                            #
#######################################################################


def minibatch(X, batchsize=256, shuffle=True):
    """
    Compute the minibatch indices for a training dataset.

    Parameters
    ----------
    X : :py:class:`ndarray <numpy.ndarray>` of shape `(N, \*)`
        The dataset to divide into minibatches. Assumes the first dimension
        represents the number of training examples.
    batchsize : int
        The desired size of each minibatch. Note, however, that if ``X.shape[0] %
        batchsize > 0`` then the final batch will contain fewer than batchsize
        entries. Default is 256.
    shuffle : bool
        Whether to shuffle the entries in the dataset before dividing into
        minibatches. Default is True.

    Returns
    -------
    mb_generator : generator
        A generator which yields the indices into X for each batch
    n_batches: int
        The number of batches
    """
    N = X.shape[0]
    ix = np.arange(N)
    n_batches = int(np.ceil(N / batchsize))

    if shuffle:
        np.random.shuffle(ix)

    def mb_generator():
        for i in range(n_batches):
            yield ix[i * batchsize : (i + 1) * batchsize]

    return mb_generator(), n_batches


#######################################################################
#                            Padding Utils                            #
#######################################################################


def calc_pad_dims_2D(X_shape, out_dim, kernel_shape, stride, dilation=0):
    """
    Compute the padding necessary to ensure that convolving `X` with a 2D kernel
    of shape `kernel_shape` and stride `stride` produces outputs with dimension
    `out_dim`.

    Parameters
    ----------
    X_shape : tuple of `(n_ex, in_rows, in_cols, in_ch)`
        Dimensions of the input volume. Padding is applied to `in_rows` and
        `in_cols`.
    out_dim : tuple of `(out_rows, out_cols)`
        The desired dimension of an output example after applying the
        convolution.
    kernel_shape : 2-tuple
        The dimension of the 2D convolution kernel.
    stride : int
        The stride for the convolution kernel.
    dilation : int
        Number of pixels inserted between kernel elements. Default is 0.

    Returns
    -------
    padding_dims : 4-tuple
        Padding dims for `X`. Organized as (left, right, up, down)
    """
    if not isinstance(X_shape, tuple):
        raise ValueError("`X_shape` must be of type tuple")

    if not isinstance(out_dim, tuple):
        raise ValueError("`out_dim` must be of type tuple")

    if not isinstance(kernel_shape, tuple):
        raise ValueError("`kernel_shape` must be of type tuple")

    if not isinstance(stride, int):
        raise ValueError("`stride` must be of type int")

    d = dilation
    fr, fc = kernel_shape
    out_rows, out_cols = out_dim
    n_ex, in_rows, in_cols, in_ch = X_shape

    # update effective filter shape based on dilation factor
    _fr, _fc = fr * (d + 1) - d, fc * (d + 1) - d

    pr = int((stride * (out_rows - 1) + _fr - in_rows) / 2)
    pc = int((stride * (out_cols - 1) + _fc - in_cols) / 2)

    out_rows1 = int(1 + (in_rows + 2 * pr - _fr) / stride)
    out_cols1 = int(1 + (in_cols + 2 * pc - _fc) / stride)

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
            "Padding cannot be less than 0. Got: {}".format((pr1, pr2, pc1, pc2))
        )
    return (pr1, pr2, pc1, pc2)


def calc_pad_dims_1D(X_shape, l_out, kernel_width, stride, dilation=0, causal=False):
    """
    Compute the padding necessary to ensure that convolving `X` with a 1D kernel
    of shape `kernel_shape` and stride `stride` produces outputs with length
    `l_out`.

    Parameters
    ----------
    X_shape : tuple of `(n_ex, l_in, in_ch)`
        Dimensions of the input volume. Padding is applied on either side of
        `l_in`.
    l_out : int
        The desired length an output example after applying the convolution.
    kernel_width : int
        The width of the 1D convolution kernel.
    stride : int
        The stride for the convolution kernel.
    dilation : int
        Number of pixels inserted between kernel elements. Default is 0.
    causal : bool
        Whether to compute the padding dims for a regular or causal
        convolution. If causal, padding is added only to the left side of the
        sequence. Default is False.

    Returns
    -------
    padding_dims : 2-tuple
        Padding dims for X. Organized as (left, right)
    """
    if not isinstance(X_shape, tuple):
        raise ValueError("`X_shape` must be of type tuple")

    if not isinstance(l_out, int):
        raise ValueError("`l_out` must be of type int")

    if not isinstance(kernel_width, int):
        raise ValueError("`kernel_width` must be of type int")

    if not isinstance(stride, int):
        raise ValueError("`stride` must be of type int")

    d = dilation
    fw = kernel_width
    n_ex, l_in, in_ch = X_shape

    # update effective filter shape based on dilation factor
    _fw = fw * (d + 1) - d
    total_pad = int((stride * (l_out - 1) + _fw - l_in))

    if not causal:
        pw = total_pad // 2
        l_out1 = int(1 + (l_in + 2 * pw - _fw) / stride)

        # add asymmetric padding pixels to right / bottom
        pw1, pw2 = pw, pw
        if l_out1 == l_out - 1:
            pw1, pw2 = pw, pw + 1
        elif l_out1 != l_out:
            raise AssertionError

    if causal:
        # if this is a causal convolution, only pad the left side of the
        # sequence
        pw1, pw2 = total_pad, 0
        l_out1 = int(1 + (l_in + total_pad - _fw) / stride)
        assert l_out1 == l_out

    if any(np.array([pw1, pw2]) < 0):
        raise ValueError("Padding cannot be less than 0. Got: {}".format((pw1, pw2)))
    return (pw1, pw2)


def pad1D(X, pad, kernel_width=None, stride=None, dilation=0):
    """
    Zero-pad a 3D input volume `X` along the second dimension.

    Parameters
    ----------
    X : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, l_in, in_ch)`
        Input volume. Padding is applied to `l_in`.
    pad : tuple, int, or {'same', 'causal'}
        The padding amount. If 'same', add padding to ensure that the output
        length of a 1D convolution with a kernel of `kernel_shape` and stride
        `stride` is the same as the input length.  If 'causal' compute padding
        such that the output both has the same length as the input AND
        ``output[t]`` does not depend on ``input[t + 1:]``. If 2-tuple,
        specifies the number of padding columns to add on each side of the
        sequence.
    kernel_width : int
        The dimension of the 2D convolution kernel. Only relevant if p='same'
        or 'causal'. Default is None.
    stride : int
        The stride for the convolution kernel. Only relevant if p='same' or
        'causal'. Default is None.
    dilation : int
        The dilation of the convolution kernel. Only relevant if p='same' or
        'causal'. Default is None.

    Returns
    -------
    X_pad : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, padded_seq, in_channels)`
        The padded output volume
    p : 2-tuple
        The number of 0-padded columns added to the (left, right) of the sequences
        in `X`.
    """
    p = pad
    if isinstance(p, int):
        p = (p, p)

    if isinstance(p, tuple):
        X_pad = np.pad(
            X,
            pad_width=((0, 0), (p[0], p[1]), (0, 0)),
            mode="constant",
            constant_values=0,
        )

    # compute the correct padding dims for a 'same' or 'causal' convolution
    if p in ["same", "causal"] and kernel_width and stride:
        causal = p == "causal"
        p = calc_pad_dims_1D(
            X.shape, X.shape[1], kernel_width, stride, causal=causal, dilation=dilation
        )
        X_pad, p = pad1D(X, p)

    return X_pad, p


def pad2D(X, pad, kernel_shape=None, stride=None, dilation=0):
    """
    Zero-pad a 4D input volume `X` along the second and third dimensions.

    Parameters
    ----------
    X : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, in_rows, in_cols, in_ch)`
        Input volume. Padding is applied to `in_rows` and `in_cols`.
    pad : tuple, int, or 'same'
        The padding amount. If 'same', add padding to ensure that the output of
        a 2D convolution with a kernel of `kernel_shape` and stride `stride`
        has the same dimensions as the input.  If 2-tuple, specifies the number
        of padding rows and colums to add *on both sides* of the rows/columns
        in `X`. If 4-tuple, specifies the number of rows/columns to add to the
        top, bottom, left, and right of the input volume.
    kernel_shape : 2-tuple
        The dimension of the 2D convolution kernel. Only relevant if p='same'.
        Default is None.
    stride : int
        The stride for the convolution kernel. Only relevant if p='same'.
        Default is None.
    dilation : int
        The dilation of the convolution kernel. Only relevant if p='same'.
        Default is 0.

    Returns
    -------
    X_pad : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, padded_in_rows, padded_in_cols, in_channels)`
        The padded output volume.
    p : 4-tuple
        The number of 0-padded rows added to the (top, bottom, left, right) of
        `X`.
    """
    p = pad
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

    # compute the correct padding dims for a 'same' convolution
    if p == "same" and kernel_shape and stride is not None:
        p = calc_pad_dims_2D(
            X.shape, X.shape[1:3], kernel_shape, stride, dilation=dilation
        )
        X_pad, p = pad2D(X, p)
    return X_pad, p


def dilate(X, d):
    """
    Dilate the 4D volume `X` by `d`.

    Notes
    -----
    For a visual depiction of a dilated convolution, see [1].

    References
    ----------
    .. [1] Dumoulin & Visin (2016). "A guide to convolution arithmetic for deep
       learning." https://arxiv.org/pdf/1603.07285v1.pdf

    Parameters
    ----------
    X : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, in_rows, in_cols, in_ch)`
        Input volume.
    d : int
        The number of 0-rows to insert between each adjacent row + column in `X`.

    Returns
    -------
    Xd : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, out_rows, out_cols, out_ch)`
        The dilated array where

        .. math::

            \\text{out_rows}  &=  \\text{in_rows} + d(\\text{in_rows} - 1) \\\\
            \\text{out_cols}  &=  \\text{in_cols} + d (\\text{in_cols} - 1)
    """
    n_ex, in_rows, in_cols, n_in = X.shape
    r_ix = np.repeat(np.arange(1, in_rows), d)
    c_ix = np.repeat(np.arange(1, in_cols), d)
    Xd = np.insert(X, r_ix, 0, axis=1)
    Xd = np.insert(Xd, c_ix, 0, axis=2)
    return Xd


#######################################################################
#                     Convolution Arithmetic                          #
#######################################################################


def calc_fan(weight_shape):
    """
    Compute the fan-in and fan-out for a weight matrix/volume.

    Parameters
    ----------
    weight_shape : tuple
        The dimensions of the weight matrix/volume. The final 2 entries must be
        `in_ch`, `out_ch`.

    Returns
    -------
    fan_in : int
        The number of input units in the weight tensor
    fan_out : int
        The number of output units in the weight tensor
    """
    if len(weight_shape) == 2:
        fan_in, fan_out = weight_shape
    elif len(weight_shape) in [3, 4]:
        in_ch, out_ch = weight_shape[-2:]
        kernel_size = np.prod(weight_shape[:-2])
        fan_in, fan_out = in_ch * kernel_size, out_ch * kernel_size
    else:
        raise ValueError("Unrecognized weight dimension: {}".format(weight_shape))
    return fan_in, fan_out


def calc_conv_out_dims(X_shape, W_shape, stride=1, pad=0, dilation=0):
    """
    Compute the dimension of the output volume for the specified convolution.

    Parameters
    ----------
    X_shape : 3-tuple or 4-tuple
        The dimensions of the input volume to the convolution. If 3-tuple,
        entries are expected to be (`n_ex`, `in_length`, `in_ch`). If 4-tuple,
        entries are expected to be (`n_ex`, `in_rows`, `in_cols`, `in_ch`).
    weight_shape : 3-tuple or 4-tuple
        The dimensions of the weight volume for the convolution. If 3-tuple,
        entries are expected to be (`f_len`, `in_ch`, `out_ch`). If 4-tuple,
        entries are expected to be (`fr`, `fc`, `in_ch`, `out_ch`).
    pad : tuple, int, or {'same', 'causal'}
        The padding amount. If 'same', add padding to ensure that the output
        length of a 1D convolution with a kernel of `kernel_shape` and stride
        `stride` is the same as the input length.  If 'causal' compute padding
        such that the output both has the same length as the input AND
        ``output[t]`` does not depend on ``input[t + 1:]``. If 2-tuple, specifies the
        number of padding columns to add on each side of the sequence. Default
        is 0.
    stride : int
        The stride for the convolution kernel. Default is 1.
    dilation : int
        The dilation of the convolution kernel. Default is 0.

    Returns
    -------
    out_dims : 3-tuple or 4-tuple
        The dimensions of the output volume. If 3-tuple, entries are (`n_ex`,
        `out_length`, `out_ch`). If 4-tuple, entries are (`n_ex`, `out_rows`,
        `out_cols`, `out_ch`).
    """
    dummy = np.zeros(X_shape)
    s, p, d = stride, pad, dilation
    if len(X_shape) == 3:
        _, p = pad1D(dummy, p)
        pw1, pw2 = p
        fw, in_ch, out_ch = W_shape
        n_ex, in_length, in_ch = X_shape

        _fw = fw * (d + 1) - d
        out_length = (in_length + pw1 + pw2 - _fw) // s + 1
        out_dims = (n_ex, out_length, out_ch)

    elif len(X_shape) == 4:
        _, p = pad2D(dummy, p)
        pr1, pr2, pc1, pc2 = p
        fr, fc, in_ch, out_ch = W_shape
        n_ex, in_rows, in_cols, in_ch = X_shape

        # adjust effective filter size to account for dilation
        _fr, _fc = fr * (d + 1) - d, fc * (d + 1) - d
        out_rows = (in_rows + pr1 + pr2 - _fr) // s + 1
        out_cols = (in_cols + pc1 + pc2 - _fc) // s + 1
        out_dims = (n_ex, out_rows, out_cols, out_ch)
    else:
        raise ValueError("Unrecognized number of input dims: {}".format(len(X_shape)))
    return out_dims


#######################################################################
#                   Convolution Vectorization Utils                   #
#######################################################################


def _im2col_indices(X_shape, fr, fc, p, s, d=0):
    """
    Helper function that computes indices into X in prep for columnization in
    :func:`im2col`.

    Code extended from Andrej Karpathy's `im2col.py`
    """
    pr1, pr2, pc1, pc2 = p
    n_ex, n_in, in_rows, in_cols = X_shape

    # adjust effective filter size to account for dilation
    _fr, _fc = fr * (d + 1) - d, fc * (d + 1) - d

    out_rows = (in_rows + pr1 + pr2 - _fr) // s + 1
    out_cols = (in_cols + pc1 + pc2 - _fc) // s + 1

    if any([out_rows <= 0, out_cols <= 0]):
        raise ValueError(
            "Dimension mismatch during convolution: "
            "out_rows = {}, out_cols = {}".format(out_rows, out_cols)
        )

    # i1/j1 : row/col templates
    # i0/j0 : n. copies (len) and offsets (values) for row/col templates
    i0 = np.repeat(np.arange(fr), fc)
    i0 = np.tile(i0, n_in) * (d + 1)
    i1 = s * np.repeat(np.arange(out_rows), out_cols)
    j0 = np.tile(np.arange(fc), fr * n_in) * (d + 1)
    j1 = s * np.tile(np.arange(out_cols), out_rows)

    # i.shape = (fr * fc * n_in, out_height * out_width)
    # j.shape = (fr * fc * n_in, out_height * out_width)
    # k.shape = (fr * fc * n_in, 1)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    k = np.repeat(np.arange(n_in), fr * fc).reshape(-1, 1)
    return k, i, j


def im2col(X, W_shape, pad, stride, dilation=0):
    """
    Pads and rearrange overlapping windows of the input volume into column
    vectors, returning the concatenated padded vectors in a matrix `X_col`.

    Notes
    -----
    A NumPy reimagining of MATLAB's ``im2col`` 'sliding' function.

    Code extended from Andrej Karpathy's ``im2col.py``.

    Parameters
    ----------
    X : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, in_rows, in_cols, in_ch)`
        Input volume (not padded).
    W_shape: 4-tuple containing `(kernel_rows, kernel_cols, in_ch, out_ch)`
        The dimensions of the weights/kernels in the present convolutional
        layer.
    pad : tuple, int, or 'same'
        The padding amount. If 'same', add padding to ensure that the output of
        a 2D convolution with a kernel of `kernel_shape` and stride `stride`
        produces an output volume of the same dimensions as the input.  If
        2-tuple, specifies the number of padding rows and colums to add *on both
        sides* of the rows/columns in X. If 4-tuple, specifies the number of
        rows/columns to add to the top, bottom, left, and right of the input
        volume.
    stride : int
        The stride of each convolution kernel
    dilation : int
        Number of pixels inserted between kernel elements. Default is 0.

    Returns
    -------
    X_col : :py:class:`ndarray <numpy.ndarray>` of shape (Q, Z)
        The reshaped input volume where where:

        .. math::

            Q  &=  \\text{kernel_rows} \\times \\text{kernel_cols} \\times \\text{n_in} \\\\
            Z  &=  \\text{n_ex} \\times \\text{out_rows} \\times \\text{out_cols}
    """
    fr, fc, n_in, n_out = W_shape
    s, p, d = stride, pad, dilation
    n_ex, in_rows, in_cols, n_in = X.shape

    # zero-pad the input
    X_pad, p = pad2D(X, p, W_shape[:2], stride=s, dilation=d)
    pr1, pr2, pc1, pc2 = p

    # shuffle to have channels as the first dim
    X_pad = X_pad.transpose(0, 3, 1, 2)

    # get the indices for im2col
    k, i, j = _im2col_indices((n_ex, n_in, in_rows, in_cols), fr, fc, p, s, d)

    X_col = X_pad[:, k, i, j]
    X_col = X_col.transpose(1, 2, 0).reshape(fr * fc * n_in, -1)
    return X_col, p


def col2im(X_col, X_shape, W_shape, pad, stride, dilation=0):
    """
    Take columns of a 2D matrix and rearrange them into the blocks/windows of
    a 4D image volume.

    Notes
    -----
    A NumPy reimagining of MATLAB's ``col2im`` 'sliding' function.

    Code extended from Andrej Karpathy's ``im2col.py``.

    Parameters
    ----------
    X_col : :py:class:`ndarray <numpy.ndarray>` of shape `(Q, Z)`
        The columnized version of `X` (assumed to include padding)
    X_shape : 4-tuple containing `(n_ex, in_rows, in_cols, in_ch)`
        The original dimensions of `X` (not including padding)
    W_shape: 4-tuple containing `(kernel_rows, kernel_cols, in_ch, out_ch)`
        The dimensions of the weights in the present convolutional layer
    pad : 4-tuple of `(left, right, up, down)`
        Number of zero-padding rows/cols to add to `X`
    stride : int
        The stride of each convolution kernel
    dilation : int
        Number of pixels inserted between kernel elements. Default is 0.

    Returns
    -------
    img : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, in_rows, in_cols, in_ch)`
        The reshaped `X_col` input matrix
    """
    if not (isinstance(pad, tuple) and len(pad) == 4):
        raise TypeError("pad must be a 4-tuple, but got: {}".format(pad))

    s, d = stride, dilation
    pr1, pr2, pc1, pc2 = pad
    fr, fc, n_in, n_out = W_shape
    n_ex, in_rows, in_cols, n_in = X_shape

    X_pad = np.zeros((n_ex, n_in, in_rows + pr1 + pr2, in_cols + pc1 + pc2))
    k, i, j = _im2col_indices((n_ex, n_in, in_rows, in_cols), fr, fc, pad, s, d)

    X_col_reshaped = X_col.reshape(n_in * fr * fc, -1, n_ex)
    X_col_reshaped = X_col_reshaped.transpose(2, 0, 1)

    np.add.at(X_pad, (slice(None), k, i, j), X_col_reshaped)

    pr2 = None if pr2 == 0 else -pr2
    pc2 = None if pc2 == 0 else -pc2
    return X_pad[:, :, pr1:pr2, pc1:pc2]


#######################################################################
#                             Convolution                             #
#######################################################################


def conv2D(X, W, stride, pad, dilation=0):
    """
    A faster (but more memory intensive) implementation of the 2D "convolution"
    (technically, cross-correlation) of input `X` with a collection of kernels in
    `W`.

    Notes
    -----
    Relies on the :func:`im2col` function to perform the convolution as a single
    matrix multiplication.

    For a helpful diagram, see Pete Warden's 2015 blogpost [1].

    References
    ----------
    .. [1] Warden (2015). "Why GEMM is at the heart of deep learning,"
       https://petewarden.com/2015/04/20/why-gemm-is-at-the-heart-of-deep-learning/

    Parameters
    ----------
    X : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, in_rows, in_cols, in_ch)`
        Input volume (unpadded).
    W: :py:class:`ndarray <numpy.ndarray>` of shape `(kernel_rows, kernel_cols, in_ch, out_ch)`
        A volume of convolution weights/kernels for a given layer.
    stride : int
        The stride of each convolution kernel.
    pad : tuple, int, or 'same'
        The padding amount. If 'same', add padding to ensure that the output of
        a 2D convolution with a kernel of `kernel_shape` and stride `stride`
        produces an output volume of the same dimensions as the input.  If
        2-tuple, specifies the number of padding rows and colums to add *on both
        sides* of the rows/columns in `X`. If 4-tuple, specifies the number of
        rows/columns to add to the top, bottom, left, and right of the input
        volume.
    dilation : int
        Number of pixels inserted between kernel elements. Default is 0.

    Returns
    -------
    Z : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, out_rows, out_cols, out_ch)`
        The covolution of `X` with `W`.
    """
    s, d = stride, dilation
    _, p = pad2D(X, pad, W.shape[:2], s, dilation=dilation)

    pr1, pr2, pc1, pc2 = p
    fr, fc, in_ch, out_ch = W.shape
    n_ex, in_rows, in_cols, in_ch = X.shape

    # update effective filter shape based on dilation factor
    _fr, _fc = fr * (d + 1) - d, fc * (d + 1) - d

    # compute the dimensions of the convolution output
    out_rows = int((in_rows + pr1 + pr2 - _fr) / s + 1)
    out_cols = int((in_cols + pc1 + pc2 - _fc) / s + 1)

    # convert X and W into the appropriate 2D matrices and take their product
    X_col, _ = im2col(X, W.shape, p, s, d)
    W_col = W.transpose(3, 2, 0, 1).reshape(out_ch, -1)

    Z = (W_col @ X_col).reshape(out_ch, out_rows, out_cols, n_ex).transpose(3, 1, 2, 0)

    return Z


def conv1D(X, W, stride, pad, dilation=0):
    """
    A faster (but more memory intensive) implementation of a 1D "convolution"
    (technically, cross-correlation) of input `X` with a collection of kernels in
    `W`.

    Notes
    -----
    Relies on the :func:`im2col` function to perform the convolution as a single
    matrix multiplication.

    For a helpful diagram, see Pete Warden's 2015 blogpost [1].

    References
    ----------
    .. [1] Warden (2015). "Why GEMM is at the heart of deep learning,"
       https://petewarden.com/2015/04/20/why-gemm-is-at-the-heart-of-deep-learning/

    Parameters
    ----------
    X : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, l_in, in_ch)`
        Input volume (unpadded)
    W: :py:class:`ndarray <numpy.ndarray>` of shape `(kernel_width, in_ch, out_ch)`
        A volume of convolution weights/kernels for a given layer
    stride : int
        The stride of each convolution kernel
    pad : tuple, int, or 'same'
        The padding amount. If 'same', add padding to ensure that the output of
        a 1D convolution with a kernel of `kernel_shape` and stride `stride`
        produces an output volume of the same dimensions as the input.  If
        2-tuple, specifies the number of padding colums to add *on both sides*
        of the columns in X.
    dilation : int
        Number of pixels inserted between kernel elements. Default is 0.

    Returns
    -------
    Z : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, l_out, out_ch)`
        The convolution of X with W.
    """
    _, p = pad1D(X, pad, W.shape[0], stride, dilation=dilation)

    # add a row dimension to X to permit us to use im2col/col2im
    X2D = np.expand_dims(X, axis=1)
    W2D = np.expand_dims(W, axis=0)
    p2D = (0, 0, p[0], p[1])
    Z2D = conv2D(X2D, W2D, stride, p2D, dilation)

    # drop the row dimension
    return np.squeeze(Z2D, axis=1)


def deconv2D_naive(X, W, stride, pad, dilation=0):
    """
    Perform a "deconvolution" (more accurately, a transposed convolution) of an
    input volume `X` with a weight kernel `W`, incorporating stride, pad, and
    dilation.

    Notes
    -----
    Rather than using the transpose of the convolution matrix, this approach
    uses a direct convolution with zero padding, which, while conceptually
    straightforward, is computationally inefficient.

    For further explanation, see [1].

    References
    ----------
    .. [1] Dumoulin & Visin (2016). "A guide to convolution arithmetic for deep
       learning." https://arxiv.org/pdf/1603.07285v1.pdf

    Parameters
    ----------
    X : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, in_rows, in_cols, in_ch)`
        Input volume (not padded)
    W: :py:class:`ndarray <numpy.ndarray>` of shape `(kernel_rows, kernel_cols, in_ch, out_ch)`
        A volume of convolution weights/kernels for a given layer
    stride : int
        The stride of each convolution kernel
    pad : tuple, int, or 'same'
        The padding amount. If 'same', add padding to ensure that the output of
        a 2D convolution with a kernel of `kernel_shape` and stride `stride`
        produces an output volume of the same dimensions as the input.  If
        2-tuple, specifies the number of padding rows and colums to add *on both
        sides* of the rows/columns in `X`. If 4-tuple, specifies the number of
        rows/columns to add to the top, bottom, left, and right of the input
        volume.
    dilation : int
        Number of pixels inserted between kernel elements. Default is 0.

    Returns
    -------
    Y : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, out_rows, out_cols, n_out)`
        The decovolution of (padded) input volume `X` with `W` using stride `s` and
        dilation `d`.
    """
    if stride > 1:
        X = dilate(X, stride - 1)
        stride = 1

    # pad the input
    X_pad, p = pad2D(X, pad, W.shape[:2], stride=stride, dilation=dilation)

    n_ex, in_rows, in_cols, n_in = X_pad.shape
    fr, fc, n_in, n_out = W.shape
    s, d = stride, dilation
    pr1, pr2, pc1, pc2 = p

    # update effective filter shape based on dilation factor
    _fr, _fc = fr * (d + 1) - d, fc * (d + 1) - d

    # compute deconvolution output dims
    out_rows = s * (in_rows - 1) - pr1 - pr2 + _fr
    out_cols = s * (in_cols - 1) - pc1 - pc2 + _fc
    out_dim = (out_rows, out_cols)

    # add additional padding to achieve the target output dim
    _p = calc_pad_dims_2D(X_pad.shape, out_dim, W.shape[:2], s, d)
    X_pad, pad = pad2D(X_pad, _p, W.shape[:2], stride=s, dilation=dilation)

    # perform the forward convolution using the flipped weight matrix (note
    # we set pad to 0, since we've already added padding)
    Z = conv2D(X_pad, np.rot90(W, 2), s, 0, d)

    pr2 = None if pr2 == 0 else -pr2
    pc2 = None if pc2 == 0 else -pc2
    return Z[:, pr1:pr2, pc1:pc2, :]


def conv2D_naive(X, W, stride, pad, dilation=0):
    """
    A slow but more straightforward implementation of a 2D "convolution"
    (technically, cross-correlation) of input `X` with a collection of kernels `W`.

    Notes
    -----
    This implementation uses ``for`` loops and direct indexing to perform the
    convolution. As a result, it is slower than the vectorized :func:`conv2D`
    function that relies on the :func:`col2im` and :func:`im2col`
    transformations.

    Parameters
    ----------
    X : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, in_rows, in_cols, in_ch)`
        Input volume.
    W: :py:class:`ndarray <numpy.ndarray>` of shape `(kernel_rows, kernel_cols, in_ch, out_ch)`
        The volume of convolution weights/kernels.
    stride : int
        The stride of each convolution kernel.
    pad : tuple, int, or 'same'
        The padding amount. If 'same', add padding to ensure that the output of
        a 2D convolution with a kernel of `kernel_shape` and stride `stride`
        produces an output volume of the same dimensions as the input.  If
        2-tuple, specifies the number of padding rows and colums to add *on both
        sides* of the rows/columns in `X`. If 4-tuple, specifies the number of
        rows/columns to add to the top, bottom, left, and right of the input
        volume.
    dilation : int
        Number of pixels inserted between kernel elements. Default is 0.

    Returns
    -------
    Z : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, out_rows, out_cols, out_ch)`
        The covolution of `X` with `W`.
    """
    s, d = stride, dilation
    X_pad, p = pad2D(X, pad, W.shape[:2], stride=s, dilation=d)

    pr1, pr2, pc1, pc2 = p
    fr, fc, in_ch, out_ch = W.shape
    n_ex, in_rows, in_cols, in_ch = X.shape

    # update effective filter shape based on dilation factor
    fr, fc = fr * (d + 1) - d, fc * (d + 1) - d

    out_rows = int((in_rows + pr1 + pr2 - fr) / s + 1)
    out_cols = int((in_cols + pc1 + pc2 - fc) / s + 1)

    Z = np.zeros((n_ex, out_rows, out_cols, out_ch))
    for m in range(n_ex):
        for c in range(out_ch):
            for i in range(out_rows):
                for j in range(out_cols):
                    i0, i1 = i * s, (i * s) + fr
                    j0, j1 = j * s, (j * s) + fc

                    window = X_pad[m, i0 : i1 : (d + 1), j0 : j1 : (d + 1), :]
                    Z[m, i, j, c] = np.sum(window * W[:, :, :, c])
    return Z


#######################################################################
#                        Weight Initialization                        #
#######################################################################


def he_uniform(weight_shape):
    """
    Initializes network weights `W` with using the He uniform initialization
    strategy.

    Notes
    -----
    The He uniform initializations trategy initializes thew eights in `W` using
    draws from Uniform(-b, b) where

    .. math::

        b = \sqrt{\\frac{6}{\\text{fan_in}}}

    Developed for deep networks with ReLU nonlinearities.

    Parameters
    ----------
    weight_shape : tuple
        The dimensions of the weight matrix/volume.

    Returns
    -------
    W : :py:class:`ndarray <numpy.ndarray>` of shape `weight_shape`
        The initialized weights.
    """
    fan_in, fan_out = calc_fan(weight_shape)
    b = np.sqrt(6 / fan_in)
    return np.random.uniform(-b, b, size=weight_shape)


def he_normal(weight_shape):
    """
    Initialize network weights `W` using the He normal initialization strategy.

    Notes
    -----
    The He normal initialization strategy initializes the weights in `W` using
    draws from TruncatedNormal(0, b) where the variance `b` is

    .. math::

        b = \\frac{2}{\\text{fan_in}}

    He normal initialization was originally developed for deep networks with
    :class:`~numpy_ml.neural_nets.activations.ReLU` nonlinearities.

    Parameters
    ----------
    weight_shape : tuple
        The dimensions of the weight matrix/volume.

    Returns
    -------
    W : :py:class:`ndarray <numpy.ndarray>` of shape `weight_shape`
        The initialized weights.
    """
    fan_in, fan_out = calc_fan(weight_shape)
    std = np.sqrt(2 / fan_in)
    return truncated_normal(0, std, weight_shape)


def glorot_uniform(weight_shape, gain=1.0):
    """
    Initialize network weights `W` using the Glorot uniform initialization
    strategy.

    Notes
    -----
    The Glorot uniform initialization strategy initializes weights using draws
    from ``Uniform(-b, b)`` where:

    .. math::

        b = \\text{gain} \sqrt{\\frac{6}{\\text{fan_in} + \\text{fan_out}}}

    The motivation for Glorot uniform initialization is to choose weights to
    ensure that the variance of the layer outputs are approximately equal to
    the variance of its inputs.

    This initialization strategy was primarily developed for deep networks with
    tanh and logistic sigmoid nonlinearities.

    Parameters
    ----------
    weight_shape : tuple
        The dimensions of the weight matrix/volume.

    Returns
    -------
    W : :py:class:`ndarray <numpy.ndarray>` of shape `weight_shape`
        The initialized weights.
    """
    fan_in, fan_out = calc_fan(weight_shape)
    b = gain * np.sqrt(6 / (fan_in + fan_out))
    return np.random.uniform(-b, b, size=weight_shape)


def glorot_normal(weight_shape, gain=1.0):
    """
    Initialize network weights `W` using the Glorot normal initialization strategy.

    Notes
    -----
    The Glorot normal initializaiton initializes weights with draws from
    TruncatedNormal(0, b) where the variance `b` is

    .. math::

        b = \\frac{2 \\text{gain}^2}{\\text{fan_in} + \\text{fan_out}}

    The motivation for Glorot normal initialization is to choose weights to
    ensure that the variance of the layer outputs are approximately equal to
    the variance of its inputs.

    This initialization strategy was primarily developed for deep networks with
    :class:`~numpy_ml.neural_nets.activations.Tanh` and
    :class:`~numpy_ml.neural_nets.activations.Sigmoid` nonlinearities.

    Parameters
    ----------
    weight_shape : tuple
        The dimensions of the weight matrix/volume.

    Returns
    -------
    W : :py:class:`ndarray <numpy.ndarray>` of shape `weight_shape`
        The initialized weights.
    """
    fan_in, fan_out = calc_fan(weight_shape)
    std = gain * np.sqrt(2 / (fan_in + fan_out))
    return truncated_normal(0, std, weight_shape)


def truncated_normal(mean, std, out_shape):
    """
    Generate draws from a truncated normal distribution via rejection sampling.

    Notes
    -----
    The rejection sampling regimen draws samples from a normal distribution
    with mean `mean` and standard deviation `std`, and resamples any values
    more than two standard deviations from `mean`.

    Parameters
    ----------
    mean : float or array_like of floats
        The mean/center of the distribution
    std : float or array_like of floats
        Standard deviation (spread or "width") of the distribution.
    out_shape : int or tuple of ints
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.

    Returns
    -------
    samples : :py:class:`ndarray <numpy.ndarray>` of shape `out_shape`
        Samples from the truncated normal distribution parameterized by `mean`
        and `std`.
    """
    samples = np.random.normal(loc=mean, scale=std, size=out_shape)
    reject = np.logical_or(samples >= mean + 2 * std, samples <= mean - 2 * std)
    while any(reject.flatten()):
        resamples = np.random.normal(loc=mean, scale=std, size=reject.sum())
        samples[reject] = resamples
        reject = np.logical_or(samples >= mean + 2 * std, samples <= mean - 2 * std)
    return samples
