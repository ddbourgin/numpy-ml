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
    X : numpy array of shape (N, ...)
        The dataset to divide into minibatches. Assumes the first dimension
        represents the number of training examples.
    batchsize : int (default: 256)
        The desired size of each minibatch. Note, however, that if X.shape[0] %
        batchsize > 0 then the final batch will contain fewer than batchsize
        entries.
    shuffle : bool (default: True)
        Whether to shuffle the entries in the dataset before dividing into
        minibatches

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


class FeatureHasher:
    def __init__(self, n_dim=256, sparse=True):
        """
        Convert a collection of features to a fixed-dimensional matrix using
        the hashing trick. Uses the md5 hash by default.

        Parameters
        ----------
        n_dim : int (default: 256)
            The dimensionality of each example in the output feature matrix.
            Small numbers of features are likely to cause hash collisions, but
            large numbers will cause larger overall parameter dimensions for
            any (linear) learning agent.
        sparse : bool (default: True)
            Whether the resulting feature matrix should be a sparse
            `scipy.csr_matrix` or dense `np.ndarray`.
        """
        self.n_dim = n_dim
        self.hash = hashlib.md5
        self.sparse = sparse and _SCIPY

    def encode(self, examples):
        """
        Encode a collection of multi-featured examples into a
        `n_dim`-dimensional feature matrix via feature hashing.

        Feature hashing works by applying a hash function to the features of an
        example and using the hash values as column indices in the resulting
        feature matrix. The entries at each hashed feature column correspond to
        the values for that example and feature. For example, given the
        following two input examples:

            >>> examples = [
                {"furry": 1, "quadruped": 1, "domesticated": 1},
                {"nocturnal": 1, "quadruped": 1},
            ]

        and a hypothetical hash function H mapping strings to [0, 127], we have:

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
            A collection of N examples, each represented as a dict where keys
            correspond to the feature name and values correspond to the feature
            value.

        Returns
        -------
        table : `np.ndarray` or `scipy.sparse.csr_matrix` of shape (N, n_dim)
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
