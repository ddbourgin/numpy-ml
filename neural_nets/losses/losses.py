from abc import ABC, abstractmethod

import numpy as np
from tests import assert_is_binary, assert_is_stochastic


class ObjectiveBase(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def loss(self, y_true, y_pred):
        pass

    @abstractmethod
    def grad(self, y_true, y_pred, **kwargs):
        pass


class SquaredError(ObjectiveBase):
    def __init__(self):
        super().__init__()

    def __call__(self, y, y_pred):
        return self.loss(y, y_pred)

    def __str__(self):
        return "SquaredError"

    @staticmethod
    def loss(y, y_pred):
        """
        Squared error (L2) loss.

        L(y, y') = 0.5 * ||y' - y||^2

        Parameters
        ----------
        y : numpy array of shape (n, m)
            Ground truth values for each of n examples
        y_pred : numpy array of shape (n, m)
            Predictions for the n examples in the batch

        Returns
        -------
        loss : float
            The sum of the squared error across dimensions and examples
        """
        return 0.5 * np.linalg.norm(y_pred - y) ** 2

    @staticmethod
    def grad(y, y_pred, z, act_fn):
        """
        Gradient of the squared error loss with respect to the pre-nonlinearity
        input, z.

        That is, return df/dz where
            f(z) = squared_error(g(z))
            g(z) = <act_fn>(z) = y_pred

        This is simply (g(z) - y) * g'(z)

        Parameters
        ----------
        y : numpy array of shape (n, m)
            Ground truth values for each of n examples
        y_pred : numpy array of shape (n, m)
            Predictions for the n examples in the batch
        act_fn : `Activation` object
            The activation function for the output layer of the network

        Returns
        -------
        grad : numpy array of shape (n, m)
            The gradient of the squared error loss with respect to z
        """
        return (y_pred - y) * act_fn.grad(z)


class CrossEntropy(ObjectiveBase):
    def __init__(self):
        super().__init__()

    def __call__(self, y, y_pred):
        return self.loss(y, y_pred)

    def __str__(self):
        return "CrossEntropy"

    @staticmethod
    def loss(y, y_pred):
        """
        Cross-entropy (log) loss. Returns the sum (not average!) of the
        losses per-sample.

        Parameters
        ----------
        y : numpy array of shape (n, m)
            Class labels (one-hot with m possible classes) for each of n examples
        y_pred : numpy array of shape (n, m)
            Probabilities of each of m classes for the n examples in the batch

        Returns
        -------
        loss : float
            The sum of the cross-entropy across classes and examples
        """
        assert_is_binary(y)
        assert_is_stochastic(y_pred)

        # prevent taking the log of 0
        eps = np.finfo(float).eps

        # each example is associated with a single class; sum the negative log
        # probability of the correct label over all samples in the batch.
        # observe that we are taking advantage of the fact that y is one-hot
        # encoded!
        cross_entropy = -np.sum(y * np.log(y_pred + eps))
        return cross_entropy

    @staticmethod
    def grad(y, y_pred):
        """
        Let:  f(z) = cross_entropy(softmax(z)).
        Then: df / dz = softmax(z) - y_true
                      = y_pred - y_true

        Note that this gradient goes through both the cross-entropy loss AND the
        softmax non-linearity to return df / dz (rather than df / d softmax(z) ).

        Input
        -----
        y : numpy array of shape (n, m)
            A one-hot encoding of the true class labels. Each row constitues a
            training example, and each column is a different class
        y_pred: numpy array of shape (n, m)
            The network predictions for the probability of each of m class labels on
            each of n examples in a batch.

        Returns
        -------
        grad : numpy array of shape (n, m)
            The gradient of the cross-entropy loss with respect to the *input*
            to the softmax function.
        """
        assert_is_binary(y)
        assert_is_stochastic(y_pred)

        # derivative of xe wrt z is y_pred - y_true, hence we can just
        # subtract 1 from the probability of the correct class labels
        grad = y_pred - y

        # [optional] scale the gradients by the number of examples in the batch
        # n, m = y.shape
        # grad /= n
        return grad


class VAELoss(ObjectiveBase):
    def __init__(self):
        super().__init__()

    def __call__(self, y, y_pred, t_mean, t_log_var):
        return self.loss(y, y_pred, t_mean, t_log_var)

    def __str__(self):
        return "VAELoss"

    @staticmethod
    def loss(y, y_pred, t_mean, t_log_var):
        """
        Variational lower bound for a Bernoulli VAE

        Parameters
        ----------
        y : numpy array of shape (n_ex, N)
            The original images
        y_pred : numpy array of shape (n_ex, N)
            The VAE reconstruction of the images
        t_mean: numpy array of shape (n_ex, T)
            Mean vector of the distribution q(t | x)
        t_log_var: numpy array of shape (n_ex, T)
            Log of the variance vector of the distribution q(t | x)

        Returns
        -------
        loss : float
            The VLB, averaged across the batch
        """
        # prevent nan on log(0)
        eps = np.finfo(float).eps
        y_pred = np.clip(y_pred, eps, 1 - eps)

        # reconstruction loss: binary cross-entropy
        rec_loss = -np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred), axis=1)

        # KL divergence between the variational distribution q and the prior p,
        # a unit gaussian
        kl_loss = -0.5 * np.sum(1 + t_log_var - t_mean ** 2 - np.exp(t_log_var), axis=1)
        loss = np.mean(kl_loss + rec_loss)
        return loss

    @staticmethod
    def grad(y, y_pred, t_mean, t_log_var):
        N = y.shape[0]
        eps = np.finfo(float).eps
        y_pred = np.clip(y_pred, eps, 1 - eps)

        dY_pred = -y / (N * y_pred) - (y - 1) / (N - N * y_pred)
        dLogVar = (np.exp(t_log_var) - 1) / (2 * N)
        dMean = t_mean / N
        return dY_pred, dLogVar, dMean
