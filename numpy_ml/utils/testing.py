import numbers
import numpy as np


#######################################################################
#                             Assertions                              #
#######################################################################


def is_symmetric(X):
    """Check that an array `X` is symmetric along its main diagonal"""
    return np.allclose(X, X.T)


def is_symmetric_positive_definite(X):
    """
    Check that a matrix `X` is a symmetric and positive-definite.
    """
    if is_symmetric(X):
        try:
            # if matrix is symmetric, check whether the Cholesky decomposition
            # (defined only for symmetric/Hermitian positive definite matrices)
            # exists
            np.linalg.cholesky(X)
            return True
        except np.linalg.LinAlgError:
            return False
    return False


def is_stochastic(X):
    """True if `X` contains probabilities that sum to 1 along the columns"""
    msg = "Array should be stochastic along the columns"
    assert len(X[X < 0]) == len(X[X > 1]) == 0, msg
    assert np.allclose(np.sum(X, axis=1), np.ones(X.shape[0])), msg
    return True


def is_number(a):
    """Check that a value `a` is numeric"""
    return isinstance(a, numbers.Number)


def is_one_hot(x):
    """Return True if array `x` is a binary array with a single 1"""
    msg = "Matrix should be one-hot binary"
    assert np.array_equal(x, x.astype(bool)), msg
    assert np.allclose(np.sum(x, axis=1), np.ones(x.shape[0])), msg
    return True


def is_binary(x):
    """Return True if array `x` consists only of binary values"""
    msg = "Matrix must be binary"
    assert np.array_equal(x, x.astype(bool)), msg
    return True


#######################################################################
#                           Data Generators                           #
#######################################################################


def random_one_hot_matrix(n_examples, n_classes):
    """Create a random one-hot matrix of shape (`n_examples`, `n_classes`)"""
    X = np.eye(n_classes)
    X = X[np.random.choice(n_classes, n_examples)]
    return X


def random_stochastic_matrix(n_examples, n_classes):
    """Create a random stochastic matrix of shape (`n_examples`, `n_classes`)"""
    X = np.random.rand(n_examples, n_classes)
    X /= X.sum(axis=1, keepdims=True)
    return X


def random_tensor(shape, standardize=False):
    """
    Create a random real-valued tensor of shape `shape`. If `standardize` is
    True, ensure each column has mean 0 and std 1.
    """
    offset = np.random.randint(-300, 300, shape)
    X = np.random.rand(*shape) + offset

    if standardize:
        eps = np.finfo(float).eps
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + eps)
    return X


def random_binary_tensor(shape, sparsity=0.5):
    """
    Create a random binary tensor of shape `shape`. `sparsity` is a value
    between 0 and 1 controlling the ratio of 0s to 1s in the output tensor.
    """
    return (np.random.rand(*shape) >= (1 - sparsity)).astype(float)


def random_paragraph(n_words, vocab=None):
    """
    Generate a random paragraph consisting of `n_words` words. If `vocab` is
    not None, words will be drawn at random from this list. Otherwise, words
    will be sampled uniformly from a collection of 26 Latin words.
    """
    if vocab is None:
        vocab = [
            "at",
            "stet",
            "accusam",
            "aliquyam",
            "clita",
            "lorem",
            "ipsum",
            "dolor",
            "dolore",
            "dolores",
            "sit",
            "amet",
            "consetetur",
            "sadipscing",
            "elitr",
            "sed",
            "diam",
            "nonumy",
            "eirmod",
            "duo",
            "ea",
            "eos",
            "erat",
            "est",
            "et",
            "gubergren",
        ]
    return [np.random.choice(vocab) for _ in range(n_words)]
