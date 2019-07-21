import numbers
import numpy as np


#######################################################################
#                             Assertions                              #
#######################################################################


def is_symmetric(X):
    return np.allclose(X, X.T)


def is_number(x):
    return isinstance(x, numbers.Number)


def is_symmetric_positive_definite(X):
    """
    Check that X is a symmetric, positive-definite matrix
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


def is_one_hot(a):
    msg = "Matrix should be one-hot binary"
    assert np.array_equal(a, a.astype(bool)), msg
    assert np.allclose(np.sum(a, axis=1), np.ones(a.shape[0])), msg
    return True


def is_binary(a):
    msg = "Matrix must be binary"
    assert np.array_equal(a, a.astype(bool)), msg
    return True


def is_stochastic(a):
    msg = "Array should be stochastic along the columns"
    assert len(a[a < 0]) == len(a[a > 1]) == 0, msg
    assert np.allclose(np.sum(a, axis=1), np.ones(a.shape[0])), msg
    return True


#######################################################################
#                           Data Generators                           #
#######################################################################


def random_one_hot_matrix(n_examples, n_classes):
    """Create a random one-hot matrix of shape n_examples x n_classes"""
    X = np.eye(n_classes)
    X = X[np.random.choice(n_classes, n_examples)]
    return X


def random_stochastic_matrix(n_examples, n_classes):
    """Create a random stochastic matrix of shape n_examples x n_classes"""
    X = np.random.rand(n_examples, n_classes)
    X /= X.sum(axis=1, keepdims=True)
    return X


def random_tensor(shape, standardize=False):
    """Create a random tensor with shape = `shape`"""
    offset = np.random.randint(-300, 300, shape)
    X = np.random.rand(*shape) + offset

    if standardize:
        eps = np.finfo(float).eps
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + eps)
    return X


def random_binary_tensor(shape, sparsity=0.5):
    return (np.random.rand(*shape) >= (1 - sparsity)).astype(float)


def random_paragraph(n_words, vocab=None):
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
