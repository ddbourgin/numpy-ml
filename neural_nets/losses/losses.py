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
        Variational lower bound for a Bernoulli VAE. Equal to the sum of
        the binary cross entropy between the true input and the predicted
        output (the "reconstruction loss") and the KL divergence between the
        learned variational distribution q and the prior, p, assumed to be a
        unit Gaussian.

        Equations:

            VAELoss = BXE(y, y_pred) + KL[q || p]

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


class WGAN_GPLoss(ObjectiveBase):
    def __init__(self, lambda_=10):
        """
        The WGAN-GP value function. Assuming an optimal critic, minimizing this
        quantity wrt. the generator parameters corresponds to minimizing the
        Wasserstein-1 (earth-mover) distance between the fake and real data
        distributions.

        Parameters
        ----------
        lambda_ : float (default: 10)
            The gradient penalty coefficient
        """
        self.lambda_ = lambda_
        super().__init__()

    def __call__(self, Y_fake, module, Y_real=None, gradInterp=None):
        """
        Computes the generator and critic loss using the WGAN-GP value
        function.

        Equations
        ---------

            WGANLoss = sum([p(x) * D(x) for x in X_real]) -
                sum([p(x_) * D(x_) for x_ in X_fake])

            WGANLossGP = WGANLoss + lambda * (||∇_Xi D(Xi)|| - 1)^2

            where:

                X_fake ~ G(z) for z ~ N(0, 1)
                Xi ~ alpha * X_real + (1 - alpha) * X_fake
                alpha ~ Uniform(0, 1, dim=X_real.shape[0])

        Parameters
        ----------
        Y_fake : numpy array of shape (n_ex,)
            The output of the critic for X_fake
        module : {'C' or 'G'}
            Whether to calculate the loss for the critic ('C') or the generator
            ('G'). If calculating loss for the critic, `Y_real` and
            `gradInterp` must not be `None`
        Y_real : numpy array of shape (n_ex,) (default: None)
            The output of the critic for X_real
        gradInterp : numpy array of shape (n_ex, n_feats) (default: None)
            The gradient of the critic output for X_interp wrt. X_interp

        Returns
        -------
        loss : float
            Depending on the setting for `module`, either the critic or
            generator loss, averaged over examples in the minibatch
        """
        return self.loss(Y_fake, module, Y_real=Y_real, gradInterp=gradInterp)

    def __str__(self):
        return "WGANLossGP(lambda_={})".format(self.lambda_)

    def loss(self, Y_fake, module, Y_real=None, gradInterp=None):
        """
        Computes the generator and critic loss using the WGAN-GP value
        function.

        Equations:

            WGANLoss = sum([p(x) * D(x) for x in X_real]) -
                sum([p(x_) * D(x_) for x_ in X_fake])

            WGANLossGP = WGANLoss + lambda * (||∇_Xi D(Xi)|| - 1)^2

            where:

                X_fake ~ G(z) for z ~ N(0, 1)
                Xi ~ alpha * X_real + (1 - alpha) * X_fake
                alpha ~ Uniform(0, 1, dim=X_real.shape[0])

        Parameters
        ----------
        Y_fake : numpy array of shape (n_ex,)
            The output of the critic for `X_fake`
        module : {'C' or 'G'}
            Whether to calculate the loss for the critic ('C') or the generator
            ('G'). If calculating loss for the critic, `Y_real` and
            `gradInterp` must not be `None`
        Y_real : numpy array of shape (n_ex,) (default: None)
            The output of the critic for `X_real`
        gradInterp : numpy array of shape (n_ex, n_feats) (default: None)
            The gradient of the critic output for `X_interp` wrt. `X_interp`

        Returns
        -------
        loss : float
            Depending on the setting for `module`, either the critic or
            generator loss, averaged over examples in the minibatch
        """
        # calc critic loss including gradient penalty
        if module == "C":
            X_interp_norm = np.linalg.norm(gradInterp, axis=1, keepdims=True)
            gradient_penalty = (X_interp_norm - 1) ** 2
            loss = (
                Y_fake.mean() - Y_real.mean() + self.lambda_ * gradient_penalty.mean()
            )

        # calc generator loss
        elif module == "G":
            loss = -Y_fake.mean()

        else:
            raise ValueError("Unrecognized module: {}".format(module))

        return loss

    def grad(self, Y_fake, module, Y_real=None, gradInterp=None):
        """
        Computes the gradient of the generator or critic loss wrt to its inputs.

        Parameters
        ----------
        Y_fake : numpy array of shape (n_ex,)
            The output of the critic for X_fake
        module : {'C' or 'G'}
            Whether to calculate the gradient for the critic loss ('C') or the
            generator loss ('G'). If calculating grads for the critic, `Y_real`
            and `gradInterp` must not be `None`.
        Y_real : numpy array of shape (n_ex,) (default: None)
            The output of the critic for X_real
        gradInterp : numpy array of shape (n_ex, n_feats) (default: None)
            The gradient of the critic output for X_interp wrt. X_interp

        Returns
        -------
        grads : tuple
            If `module` == 'C', returns a 3-tuple containing the gradient of
            the critic loss wrt. (`Y_fake`, `Y_real`, `gradInterp`). If
            `module` == 'G', returns the gradient of the generator wrt. `Y_fake`.
        """
        eps = np.finfo(float).eps
        n_ex_fake = Y_fake.shape[0]

        # calc gradient of the critic loss
        if module == "C":
            n_ex_real = Y_real.shape[0]

            dY_fake = -1 / n_ex_fake * np.ones_like(Y_fake)
            dY_real = 1 / n_ex_real * np.ones_like(Y_real)

            # differentiate through gradient penalty
            X_interp_norm = np.linalg.norm(gradInterp, axis=1, keepdims=True) + eps

            dGradInterp = (
                (2 / n_ex_fake)
                * self.lambda_
                * (X_interp_norm - 1)
                * (gradInterp / X_interp_norm)
            )
            grad = (dY_fake, dY_real, dGradInterp)

        # calc gradient of the generator loss
        elif module == "G":
            grad = -1 / n_ex_fake * np.ones_like(Y_fake)

        else:
            raise ValueError("Unrecognized module: {}".format(module))
        return grad
