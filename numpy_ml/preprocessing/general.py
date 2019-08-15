import json
import hashlib
import warnings

import numpy as np

try:
    from scipy.sparse import csr_matrix

    _SCIPY = True
except ImportError:
    warnings.warn("Scipy not installed. FeatureHasher can only create dense matrices")
    _SCIPY = False


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
        A generator which yields the indices into `X` for each batch.
    n_batches: int
        The number of batches.
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


class OneHotEncoder:
    def __init__(self):
        """
        Convert between category labels and their one-hot vector
        representations.

        Parameters
        ----------
        categories : list of length `C`
            List of the unique category labels for the items to encode.
        """
        self._is_fit = False
        self.hyperparameters = {}
        self.parameters = {"categories": None}

    def __call__(self, labels):
        return self.transform(labels)

    def fit(self, categories):
        """
        Create mappings between columns and category labels.

        Parameters
        ----------
        categories : list of length `C`
            List of the unique category labels for the items to encode.
        """
        self.parameters["categories"] = categories
        self.cat2idx = {c: i for i, c in enumerate(categories)}
        self.idx2cat = {i: c for i, c in enumerate(categories)}
        self._is_fit = True

    def transform(self, labels, categories=None):
        """
        Convert a list of labels into a one-hot encoding.

        Parameters
        ----------
        labels : list of length `N`
            A list of category labels.
        categories : list of length `C`
            List of the unique category labels for the items to encode. Default
            is None.

        Returns
        -------
        Y : :py:class:`ndarray <numpy.ndarray>` of shape `(N, C)`
            The one-hot encoded labels. Each row corresponds to an example,
            with a single 1 in the column corresponding to the respective
            label.
        """
        if not self._is_fit:
            categories = set(labels) if categories is None else categories
            self.fit(categories)

        unknown = list(set(labels) - set(self.cat2idx.keys()))
        assert len(unknown) == 0, "Unrecognized label(s): {}".format(unknown)

        N, C = len(labels), len(self.cat2idx)
        cols = np.array([self.cat2idx[c] for c in labels])

        Y = np.zeros((N, C))
        Y[np.arange(N), cols] = 1
        return Y

    def inverse_transform(self, Y):
        """
        Convert a one-hot encoding back into the corresponding labels

        Parameters
        ----------
        Y : :py:class:`ndarray <numpy.ndarray>` of shape `(N, C)`
            One-hot encoded labels. Each row corresponds to an example, with a
            single 1 in the column associated with the label for that example

        Returns
        -------
        labels : list of length `N`
            The list of category labels corresponding to the nonzero columns in
            `Y`
        """
        C = len(self.cat2idx)
        assert Y.ndim == 2, "Y must be 2D, but has shape {}".format(Y.shape)
        assert Y.shape[1] == C, "Y must have {} columns, got {}".format(C, Y.shape[1])
        return [self.idx2cat[ix] for ix in Y.nonzero()[1]]


class Standardizer:
    def __init__(self, with_mean=True, with_std=True):
        """
        Feature-wise standardization for vector inputs.

        Notes
        -----
        Due to the sensitivity of empirical mean and standard deviation
        calculations to extreme values, `Standardizer` cannot guarantee
        balanced feature scales in the presence of outliers. In particular,
        note that because outliers for each feature can have different
        magnitudes, the spread of the transformed data on each feature can be
        very different.

        Similar to sklearn, `Standardizer` uses a biased estimator for the
        standard deviation: ``numpy.std(x, ddof=0)``.

        Parameters
        ----------
        with_mean : bool
            Whether to scale samples to have 0 mean during transformation.
            Default is True.
        with_std : bool
            Whether to scale samples to have unit variance during
            transformation. Default is True.
        """
        self.with_mean = with_mean
        self.with_std = with_std
        self._is_fit = False

    @property
    def hyperparameters(self):
        H = {"with_mean": self.with_mean, "with_std": self.with_std}
        return H

    @property
    def parameters(self):
        params = {
            "mean": self._mean if hasattr(self, "mean") else None,
            "std": self._std if hasattr(self, "std") else None,
        }
        return params

    def __call__(self, X):
        return self.transform(X)

    def fit(self, X):
        """
        Store the feature-wise mean and standard deviation across the samples
        in `X` for future scaling.

        Parameters
        ----------
        X : :py:class:`ndarray <numpy.ndarray>` of shape `(N, C)`
            An array of N samples, each with dimensionality `C`
        """
        if not isinstance(X, np.ndarray):
            X = np.array(X)

        if X.shape[0] < 2:
            raise ValueError("`X` must contain at least 2 samples")

        std = np.ones(X.shape[1])
        mean = np.zeros(X.shape[1])

        if self.with_mean:
            mean = np.mean(X, axis=0)

        if self.with_std:
            std = np.std(X, axis=0, ddof=0)

        self._mean = mean
        self._std = std
        self._is_fit = True

    def transform(self, X):
        """
        Standardize features by removing the mean and scaling to unit variance.

        For a sample `x`, the standardized score is calculated as:

        .. math::

            z = (x - u) / s

        where `u` is the mean of the training samples or zero if `with_mean` is
        False, and `s` is the standard deviation of the training samples or 1
        if `with_std` is False.

        Parameters
        ----------
        X : :py:class:`ndarray <numpy.ndarray>` of shape `(N, C)`
            An array of N samples, each with dimensionality `C`.

        Returns
        -------
        Z : :py:class:`ndarray <numpy.ndarray>` of shape `(N, C)`
            The feature-wise standardized version of `X`.
        """
        if not self._is_fit:
            raise Exception("Must call `fit` before using the `transform` method")
        return (X - self._mean) / self._std

    def inverse_transform(self, Z):
        """
        Convert a collection of standardized features back into the original
        feature space.

        For a standardized sample `z`, the unstandardized score is calculated as:

        .. math::

            x = z s + u

        where `u` is the mean of the training samples or zero if `with_mean` is
        False, and `s` is the standard deviation of the training samples or 1
        if `with_std` is False.

        Parameters
        ----------
        Z : :py:class:`ndarray <numpy.ndarray>` of shape `(N, C)`
            An array of `N` standardized samples, each with dimensionality `C`.

        Returns
        -------
        X : :py:class:`ndarray <numpy.ndarray>` of shape `(N, C)`
            The unstandardixed samples from `Z`.
        """
        assert self._is_fit, "Must fit `Standardizer` before calling inverse_transform"
        P = self.parameters
        mean, std = P["mean"], P["std"]
        return Z * std + mean


class FeatureHasher:
    def __init__(self, n_dim=256, sparse=True):
        """
        Convert a collection of features to a fixed-dimensional matrix using
        the hashing trick.

        Notes
        -----
        Uses the md5 hash.

        Parameters
        ----------
        n_dim : int
            The dimensionality of each example in the output feature matrix.
            Small numbers of features are likely to cause hash collisions, but
            large numbers will cause larger overall parameter dimensions for
            any (linear) learning agent. Default is 256.
        sparse : bool
            Whether the resulting feature matrix should be a sparse
            :py:class:`csr_matrix <scipy.sparse.csr_matrix>` or dense
            :py:class:`ndarray <numpy.ndarray>`. Default is True.
        """
        self.n_dim = n_dim
        self.hash = hashlib.md5
        self.sparse = sparse and _SCIPY

    def encode(self, examples):
        """
        Encode a collection of multi-featured examples into a
        `n_dim`-dimensional feature matrix via feature hashing.

        Notes
        -----
        Feature hashing works by applying a hash function to the features of an
        example and using the hash values as column indices in the resulting
        feature matrix. The entries at each hashed feature column correspond to
        the values for that example and feature. For example, given the
        following two input examples:

            >>> examples = [
                {"furry": 1, "quadruped": 1, "domesticated": 1},
                {"nocturnal": 1, "quadruped": 1},
            ]

        and a hypothetical hash function `H` mapping strings to [0, 127], we have:

            >>> feature_mat = zeros(2, 128)
            >>> ex1_cols = [H("furry"), H("quadruped"), H("domesticated")]
            >>> ex2_cols = [H("nocturnal"), H("quadruped")]
            >>> feat_mat[0, ex1_cols] = 1
            >>> feat_mat[1, ex2_cols] = 1

        To better handle hash collisions, it is common to multiply the feature
        value by the sign of the digest for the corresponding feature name.

        Parameters
        ----------
        examples : dict or list of dicts
            A collection of `N` examples, each represented as a dict where keys
            correspond to the feature name and values correspond to the feature
            value.

        Returns
        -------
        table : :py:class:`ndarray <numpy.ndarray>` or :py:class:`csr_matrix <scipy.sparse.csr_matrix>` of shape `(N, n_dim)`
            The encoded feature matrix
        """
        if isinstance(examples, dict):
            examples = [examples]

        sparse = self.sparse
        return self._encode_sparse(examples) if sparse else self._encode_dense(examples)

    def _encode_dense(self, examples):
        N = len(examples)
        table = np.zeros(N, self.n_dim)  # dense

        for row, feat_dict in enumerate(examples):
            for f_id, val in feat_dict.items():
                if isinstance(f_id, str):
                    f_id = f_id.encode("utf-8")

                # use json module to convert the feature id into a unique
                # string compatible with the buffer API (required by hashlib)
                if isinstance(f_id, (tuple, dict, list)):
                    f_id = json.dumps(f_id, sort_keys=True).encode("utf-8")

                h = int(self.hash(f_id).hexdigest(), base=16)
                col = h % self.n_dim
                table[row, col] += np.sign(h) * val

        return table

    def _encode_sparse(self, examples):
        N = len(examples)
        idxs, data = [], []

        for row, feat_dict in enumerate(examples):
            for f_id, val in feat_dict.items():
                if isinstance(f_id, str):
                    f_id = f_id.encode("utf-8")

                # use json module to convert the feature id into a unique
                # string compatible with the buffer API (required by hashlib)
                if isinstance(f_id, (tuple, dict, list)):
                    f_id = json.dumps(f_id, sort_keys=True).encode("utf-8")

                h = int(self.hash(f_id).hexdigest(), base=16)
                col = h % self.n_dim
                idxs.append((row, col))
                data.append(np.sign(h) * val)

        table = csr_matrix((data, zip(*idxs)), shape=(N, self.n_dim))
        return table
