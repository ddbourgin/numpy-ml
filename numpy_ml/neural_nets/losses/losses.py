from abc import ABC, abstractmethod

import numpy as np

from ...utils.testing import is_binary, is_stochastic
from ..initializers import (
    WeightInitializer,
    ActivationInitializer,
    OptimizerInitializer,
)


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
        """
        A squared-error / `L2` loss.

        Notes
        -----
        For real-valued target **y** and predictions :math:`\hat{\mathbf{y}}`, the
        squared error is

        .. math::
                \mathcal{L}(\mathbf{y}, \hat{\mathbf{y}})
                    = 0.5 ||\hat{\mathbf{y}} - \mathbf{y}||_2^2
        """
        super().__init__()

    def __call__(self, y, y_pred):
        return self.loss(y, y_pred)

    def __str__(self):
        return "SquaredError"

    @staticmethod
    def loss(y, y_pred):
        """
        Compute the squared error between `y` and `y_pred`.

        Parameters
        ----------
        y : :py:class:`ndarray <numpy.ndarray>` of shape (n, m)
            Ground truth values for each of `n` examples
        y_pred : :py:class:`ndarray <numpy.ndarray>` of shape (n, m)
            Predictions for the `n` examples in the batch.

        Returns
        -------
        loss : float
            The sum of the squared error across dimensions and examples.
        """
        return 0.5 * np.linalg.norm(y_pred - y) ** 2

    @staticmethod
    def grad(y, y_pred, z, act_fn):
        """
        Gradient of the squared error loss with respect to the pre-nonlinearity
        input, `z`.

        Notes
        -----
        The current method computes the gradient :math:`\\frac{\partial
        \mathcal{L}}{\partial \mathbf{z}}`, where

        .. math::

            \mathcal{L}(\mathbf{z})
                &=  \\text{squared_error}(\mathbf{y}, g(\mathbf{z})) \\\\
            g(\mathbf{z})
                &=  \\text{act_fn}(\mathbf{z})

        The gradient with respect to :math:`\mathbf{z}` is then

        .. math::

            \\frac{\partial \mathcal{L}}{\partial \mathbf{z}}
                = (g(\mathbf{z}) - \mathbf{y}) \left(
                    \\frac{\partial g}{\partial \mathbf{z}} \\right)

        Parameters
        ----------
        y : :py:class:`ndarray <numpy.ndarray>` of shape (n, m)
            Ground truth values for each of `n` examples.
        y_pred : :py:class:`ndarray <numpy.ndarray>` of shape (n, m)
            Predictions for the `n` examples in the batch.
        act_fn : :doc:`Activation <numpy_ml.neural_nets.activations>` object
            The activation function for the output layer of the network.

        Returns
        -------
        grad : :py:class:`ndarray <numpy.ndarray>` of shape (n, m)
            The gradient of the squared error loss with respect to `z`.
        """
        return (y_pred - y) * act_fn.grad(z)


class CrossEntropy(ObjectiveBase):
    def __init__(self):
        """
        A cross-entropy loss.

        Notes
        -----
        For a one-hot target **y** and predicted class probabilities
        :math:`\hat{\mathbf{y}}`, the cross entropy is

        .. math::
                \mathcal{L}(\mathbf{y}, \hat{\mathbf{y}})
                    = \sum_i y_i \log \hat{y}_i
        """
        super().__init__()

    def __call__(self, y, y_pred):
        return self.loss(y, y_pred)

    def __str__(self):
        return "CrossEntropy"

    @staticmethod
    def loss(y, y_pred):
        """
        Compute the cross-entropy (log) loss.

        Notes
        -----
        This method returns the sum (not the average!) of the losses for each
        sample.

        Parameters
        ----------
        y : :py:class:`ndarray <numpy.ndarray>` of shape (n, m)
            Class labels (one-hot with `m` possible classes) for each of `n`
            examples.
        y_pred : :py:class:`ndarray <numpy.ndarray>` of shape (n, m)
            Probabilities of each of `m` classes for the `n` examples in the
            batch.

        Returns
        -------
        loss : float
            The sum of the cross-entropy across classes and examples.
        """
        is_binary(y)
        is_stochastic(y_pred)

        # prevent taking the log of 0
        eps = np.finfo(float).eps

        # each example is associated with a single class; sum the negative log
        # probability of the correct label over all samples in the batch.
        # observe that we are taking advantage of the fact that y is one-hot
        # encoded
        cross_entropy = -np.sum(y * np.log(y_pred + eps))
        return cross_entropy

    @staticmethod
    def grad(y, y_pred):
        """
        Compute the gradient of the cross entropy loss with regard to the
        softmax input, `z`.

        Notes
        -----
        The gradient for this method goes through both the cross-entropy loss
        AND the softmax non-linearity to return :math:`\\frac{\partial
        \mathcal{L}}{\partial \mathbf{z}}` (rather than :math:`\\frac{\partial
        \mathcal{L}}{\partial \\text{softmax}(\mathbf{z})}`).

        In particular, let:

        .. math::

            \mathcal{L}(\mathbf{z})
                = \\text{cross_entropy}(\\text{softmax}(\mathbf{z})).

        The current method computes:

        .. math::

            \\frac{\partial \mathcal{L}}{\partial \mathbf{z}}
                &= \\text{softmax}(\mathbf{z}) - \mathbf{y} \\\\
                &=  \hat{\mathbf{y}} - \mathbf{y}

        Parameters
        ----------
        y : :py:class:`ndarray <numpy.ndarray>` of shape `(n, m)`
            A one-hot encoding of the true class labels. Each row constitues a
            training example, and each column is a different class.
        y_pred: :py:class:`ndarray <numpy.ndarray>` of shape `(n, m)`
            The network predictions for the probability of each of `m` class
            labels on each of `n` examples in a batch.

        Returns
        -------
        grad : :py:class:`ndarray <numpy.ndarray>` of shape (n, m)
            The gradient of the cross-entropy loss with respect to the *input*
            to the softmax function.
        """
        is_binary(y)
        is_stochastic(y_pred)

        # derivative of xe wrt z is y_pred - y_true, hence we can just
        # subtract 1 from the probability of the correct class labels
        grad = y_pred - y

        # [optional] scale the gradients by the number of examples in the batch
        # n, m = y.shape
        # grad /= n
        return grad


class VAELoss(ObjectiveBase):
    def __init__(self):
        """
        The variational lower bound for a variational autoencoder with Bernoulli
        units.

        Notes
        -----
        The VLB to the sum of the binary cross entropy between the true input and
        the predicted output (the "reconstruction loss") and the KL divergence
        between the learned variational distribution :math:`q` and the prior,
        :math:`p`, assumed to be a unit Gaussian.

        .. math::

            \\text{VAELoss} =
                \\text{cross_entropy}(\mathbf{y}, \hat{\mathbf{y}})
                    + \\mathbb{KL}[q \ || \ p]

        where :math:`\mathbb{KL}[q \ || \ p]` is the Kullback-Leibler
        divergence between the distributions :math:`q` and :math:`p`.

        References
        ----------
        .. [1] Kingma, D. P. & Welling, M. (2014). "Auto-encoding variational Bayes".
           *arXiv preprint arXiv:1312.6114.* https://arxiv.org/pdf/1312.6114.pdf
        """
        super().__init__()

    def __call__(self, y, y_pred, t_mean, t_log_var):
        return self.loss(y, y_pred, t_mean, t_log_var)

    def __str__(self):
        return "VAELoss"

    @staticmethod
    def loss(y, y_pred, t_mean, t_log_var):
        """
        Variational lower bound for a Bernoulli VAE.

        Parameters
        ----------
        y : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, N)`
            The original images.
        y_pred : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, N)`
            The VAE reconstruction of the images.
        t_mean: :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, T)`
            Mean of the variational distribution :math:`q(t \mid x)`.
        t_log_var: :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, T)`
            Log of the variance vector of the variational distribution
            :math:`q(t \mid x)`.

        Returns
        -------
        loss : float
            The VLB, averaged across the batch.
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
        """
        Compute the gradient of the VLB with regard to the network parameters.

        Parameters
        ----------
        y : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, N)`
            The original images.
        y_pred : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, N)`
            The VAE reconstruction of the images.
        t_mean: :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, T)`
            Mean of the variational distribution :math:`q(t | x)`.
        t_log_var: :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, T)`
            Log of the variance vector of the variational distribution
            :math:`q(t | x)`.

        Returns
        -------
        dY_pred : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, N)`
            The gradient of the VLB with regard to `y_pred`.
        dLogVar : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, T)`
            The gradient of the VLB with regard to `t_log_var`.
        dMean : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, T)`
            The gradient of the VLB with regard to `t_mean`.
        """
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
        The loss function for a Wasserstein GAN [*]_ [*]_ with gradient penalty.

        Notes
        -----
        Assuming an optimal critic, minimizing this quantity wrt. the generator
        parameters corresponds to minimizing the Wasserstein-1 (earth-mover)
        distance between the fake and real data distributions.

        The formula for the WGAN-GP critic loss is

        .. math::

            \\text{WGANLoss}
                &=  \sum_{x \in X_{real}} p(x) D(x)
                    - \sum_{x' \in X_{fake}} p(x') D(x') \\\\
            \\text{WGANLossGP}
                &=  \\text{WGANLoss} + \lambda
                    (||\\nabla_{X_{interp}} D(X_{interp})||_2 - 1)^2

        where

        .. math::

            X_{fake}  &=   \\text{Generator}(\mathbf{z}) \\\\
            X_{interp}   &=   \\alpha X_{real} + (1 - \\alpha) X_{fake} \\\\

        and

        .. math::

            \mathbf{z}  &\sim  \mathcal{N}(0, \mathbb{1}) \\\\
            \\alpha  &\sim  \\text{Uniform}(0, 1)

        References
        ----------
        .. [*] Gulrajani, I., Ahmed, F., Arjovsky, M., Dumoulin, V., &
           Courville, A. (2017) "Improved training of Wasserstein GANs"
           *Advances in Neural Information Processing Systems, 31*: 5769-5779.
        .. [*] Goodfellow, I. J, Abadie, P. A., Mirza, M., Xu, B., Farley, D.
           W., Ozair, S., Courville, A., & Bengio, Y. (2014) "Generative
           adversarial nets" *Advances in Neural Information Processing
           Systems, 27*: 2672-2680.

        Parameters
        ----------
        lambda_ : float
            The gradient penalty coefficient. Default is 10.
        """
        self.lambda_ = lambda_
        super().__init__()

    def __call__(self, Y_fake, module, Y_real=None, gradInterp=None):
        """
        Computes the generator and critic loss using the WGAN-GP value
        function.

        Parameters
        ----------
        Y_fake : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex,)`
            The output of the critic for `X_fake`.
        module : {'C', 'G'}
            Whether to calculate the loss for the critic ('C') or the generator
            ('G'). If calculating loss for the critic, `Y_real` and
            `gradInterp` must not be None.
        Y_real : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex,)`, or None
            The output of the critic for `X_real`. Default is None.
        gradInterp : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, n_feats)`, or None
            The gradient of the critic output for `X_interp` wrt. `X_interp`.
            Default is None.

        Returns
        -------
        loss : float
            Depending on the setting for `module`, either the critic or
            generator loss, averaged over examples in the minibatch.
        """
        return self.loss(Y_fake, module, Y_real=Y_real, gradInterp=gradInterp)

    def __str__(self):
        return "WGANLossGP(lambda_={})".format(self.lambda_)

    def loss(self, Y_fake, module, Y_real=None, gradInterp=None):
        """
        Computes the generator and critic loss using the WGAN-GP value
        function.

        Parameters
        ----------
        Y_fake : :py:class:`ndarray <numpy.ndarray>` of shape (n_ex,)
            The output of the critic for `X_fake`.
        module : {'C', 'G'}
            Whether to calculate the loss for the critic ('C') or the generator
            ('G'). If calculating loss for the critic, `Y_real` and
            `gradInterp` must not be None.
        Y_real : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex,)` or None
            The output of the critic for `X_real`. Default is None.
        gradInterp : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, n_feats)` or None
            The gradient of the critic output for `X_interp` wrt. `X_interp`.
            Default is None.

        Returns
        -------
        loss : float
            Depending on the setting for `module`, either the critic or
            generator loss, averaged over examples in the minibatch.
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
        Computes the gradient of the generator or critic loss with regard to
        its inputs.

        Parameters
        ----------
        Y_fake : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex,)`
            The output of the critic for `X_fake`.
        module : {'C', 'G'}
            Whether to calculate the gradient for the critic loss ('C') or the
            generator loss ('G'). If calculating grads for the critic, `Y_real`
            and `gradInterp` must not be None.
        Y_real : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex,)` or None
            The output of the critic for `X_real`. Default is None.
        gradInterp : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, n_feats)` or None
            The gradient of the critic output on `X_interp` wrt. `X_interp`.
            Default is None.

        Returns
        -------
        grads : tuple
            If `module` == 'C', returns a 3-tuple containing the gradient of
            the critic loss with regard to (`Y_fake`, `Y_real`, `gradInterp`).
            If `module` == 'G', returns the gradient of the generator with
            regard to `Y_fake`.
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


class NCELoss(ObjectiveBase):
    """
    """

    def __init__(
        self,
        n_classes,
        noise_sampler,
        num_negative_samples,
        optimizer=None,
        init="glorot_uniform",
        subtract_log_label_prob=True,
    ):
        """
        A noise contrastive estimation (NCE) loss function.

        Notes
        -----
        Noise contrastive estimation is a candidate sampling method often
        used to reduce the computational challenge of training a softmax
        layer on problems with a large number of output classes. It proceeds by
        training a logistic regression model to discriminate between samples
        from the true data distribution and samples from an artificial noise
        distribution.

        It can be shown that as the ratio of negative samples to data samples
        goes to infinity, the gradient of the NCE loss converges to the
        original softmax gradient.

        For input data **X**, target labels `targets`, loss parameters **W** and
        **b**, and noise samples `noise` sampled from the noise distribution `Q`,
        the NCE loss is

        .. math::

            \\text{NCE}(X, targets) =
                \\text{cross_entropy}(\mathbf{y}_{targets}, \hat{\mathbf{y}}_{targets}) +
                \\text{cross_entropy}(\mathbf{y}_{noise}, \hat{\mathbf{y}}_{noise})

        where

        .. math::

            \hat{\mathbf{y}}_{targets}
                &=  \sigma(\mathbf{W}[targets] \mathbf{X} + \mathbf{b}[targets] - \log Q(targets)) \\\\
            \hat{\mathbf{y}}_{noise}
                &=  \sigma(\mathbf{W}[noise] \mathbf{X} + \mathbf{b}[noise] - \log Q(noise))

        In the above equations, :math:`\sigma` is the logistic sigmoid
        function, and :math:`Q(x)` corresponds to the probability of the values
        in `x` under `Q`.

        References
        ----------
        .. [1] Gutmann, M. & Hyvarinen, A. (2010). Noise-contrastive
           estimation: A new estimation principle for unnormalized statistical
           models. *AISTATS, 13*: 297-304.
        .. [2] Minh, A. & Teh, Y. W. (2012). A fast and simple algorithm for
           training neural probabilistic language models. *ICML, 29*: 1751-1758.

        Parameters
        ----------
        n_classes : int
            The total number of output classes in the model.
        noise_sampler : :class:`~numpy_ml.utils.data_structures.DiscreteSampler` instance
            The negative sampler. Defines a distribution over all classes in
            the dataset.
        num_negative_samples : int
            The number of negative samples to draw for each target / batch of
            targets.
        init : {'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform'}
            The weight initialization strategy. Default is 'glorot_uniform'.
        optimizer : str, :doc:`Optimizer <numpy_ml.neural_nets.optimizers>` object, or None
            The optimization strategy to use when performing gradient updates
            within the :meth:`update` method.  If None, use the :class:`SGD
            <numpy_ml.neural_nets.optimizers.SGD>` optimizer with
            default parameters. Default is None.
        subtract_log_label_prob : bool
            Whether to subtract the log of the probability of each label under
            the noise distribution from its respective logit. Set to False for
            negative sampling, True for NCE. Default is True.

        Attributes
        ----------
        gradients : dict
            The accumulated parameter gradients.
        parameters: dict
            The loss parameter values.
        hyperparameters: dict
            The loss hyperparameter values.
        derived_variables: dict
            Useful intermediate values computed during the loss computation.
        """
        super().__init__()

        self.init = init
        self.n_in = None
        self.trainable = True
        self.n_classes = n_classes
        self.noise_sampler = noise_sampler
        self.num_negative_samples = num_negative_samples
        self.act_fn = ActivationInitializer("Sigmoid")()
        self.optimizer = OptimizerInitializer(optimizer)()
        self.subtract_log_label_prob = subtract_log_label_prob

        self.is_initialized = False

    def _init_params(self):
        init_weights = WeightInitializer(str(self.act_fn), mode=self.init)

        self.X = []
        b = np.zeros((1, self.n_classes))
        W = init_weights((self.n_classes, self.n_in))

        self.parameters = {"W": W, "b": b}

        self.gradients = {"W": np.zeros_like(W), "b": np.zeros_like(b)}

        self.derived_variables = {
            "y_pred": [],
            "target": [],
            "true_w": [],
            "true_b": [],
            "sampled_b": [],
            "sampled_w": [],
            "out_labels": [],
            "target_logits": [],
            "noise_samples": [],
            "noise_logits": [],
        }

        self.is_initialized = True

    @property
    def hyperparameters(self):
        return {
            "id": "NCELoss",
            "n_in": self.n_in,
            "init": self.init,
            "n_classes": self.n_classes,
            "noise_sampler": self.noise_sampler,
            "num_negative_samples": self.num_negative_samples,
            "subtract_log_label_prob": self.subtract_log_label_prob,
            "optimizer": {
                "cache": self.optimizer.cache,
                "hyperparameters": self.optimizer.hyperparameters,
            },
        }

    def __call__(self, X, target, neg_samples=None, retain_derived=True):
        return self.loss(X, target, neg_samples, retain_derived)

    def __str__(self):
        keys = [
            "{}={}".format(k, v)
            for k, v in self.hyperparameters.items()
            if k not in ["id", "optimizer"]
        ] + ["optimizer={}".format(self.optimizer)]
        return "NCELoss({})".format(", ".join(keys))

    def freeze(self):
        """
        Freeze the loss parameters at their current values so they can no
        longer be updated.
        """
        self.trainable = False

    def unfreeze(self):
        """Unfreeze the layer parameters so they can be updated."""
        self.trainable = True

    def flush_gradients(self):
        """Erase all the layer's derived variables and gradients."""
        assert self.trainable, "NCELoss is frozen"
        self.X = []
        for k, v in self.derived_variables.items():
            self.derived_variables[k] = []

        for k, v in self.gradients.items():
            self.gradients[k] = np.zeros_like(v)

    def update(self, cur_loss=None):
        """
        Update the loss parameters using the accrued gradients and optimizer.
        Flush all gradients once the update is complete.
        """
        assert self.trainable, "NCELoss is frozen"
        self.optimizer.step()
        for k, v in self.gradients.items():
            if k in self.parameters:
                self.parameters[k] = self.optimizer(self.parameters[k], v, k, cur_loss)
        self.flush_gradients()

    def loss(self, X, target, neg_samples=None, retain_derived=True):
        """
        Compute the NCE loss for a collection of inputs and associated targets.

        Parameters
        ----------
        X : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, n_c, n_in)`
            Layer input. A minibatch of `n_ex` examples, where each example is
            an `n_c` by `n_in` matrix (e.g., the matrix of `n_c` context
            embeddings, each of dimensionality `n_in`, for a CBOW model).
        target : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex,)`
            Integer indices of the target class(es) for each example in the
            minibatch (e.g., the target word id for an example in a CBOW model).
        neg_samples : :py:class:`ndarray <numpy.ndarray>` of shape (`num_negative_samples`,) or None
            An optional array of negative samples to use during the loss
            calculation. These will be used instead of samples draw from
            ``self.noise_sampler``. Default is None.
        retain_derived : bool
            Whether to retain the variables calculated during the forward pass
            for use later during backprop. If False, this suggests the layer
            will not be expected to backprop through with regard to this input.
            Default is True.

        Returns
        -------
        loss : float
            The NCE loss summed over the minibatch and samples.
        y_pred : :py:class:`ndarray <numpy.ndarray>` of shape (`n_ex`, `n_c`)
            The network predictions for the conditional probability of each
            target given each context: entry (`i`, `j`) gives the predicted
            probability of target `i` under context vector `j`.
        """
        if not self.is_initialized:
            self.n_in = X.shape[-1]
            self._init_params()

        loss, Z_target, Z_neg, y_pred, y_true, noise_samples = self._loss(
            X, target, neg_samples
        )

        # cache derived variables for gradient calculation
        if retain_derived:
            self.X.append(X)

            self.derived_variables["y_pred"].append(y_pred)
            self.derived_variables["target"].append(target)
            self.derived_variables["out_labels"].append(y_true)
            self.derived_variables["target_logits"].append(Z_target)
            self.derived_variables["noise_samples"].append(noise_samples)
            self.derived_variables["noise_logits"].append(Z_neg)

        return loss, np.squeeze(y_pred[..., :1], -1)

    def _loss(self, X, target, neg_samples):
        """Actual computation of NCE loss"""
        fstr = "X must have shape (n_ex, n_c, n_in), but got {} dims instead"
        assert X.ndim == 3, fstr.format(X.ndim)

        W = self.parameters["W"]
        b = self.parameters["b"]

        # sample negative samples from the noise distribution
        if neg_samples is None:
            neg_samples = self.noise_sampler(self.num_negative_samples)
        assert len(neg_samples) == self.num_negative_samples

        # get the probability of the negative sample class and the target
        # class under the noise distribution
        p_neg_samples = self.noise_sampler.probs[neg_samples]
        p_target = np.atleast_2d(self.noise_sampler.probs[target])

        # save the noise samples for debugging
        noise_samples = (neg_samples, p_target, p_neg_samples)

        # compute the logit for the negative samples and target
        Z_target = X @ W[target].T + b[0, target]
        Z_neg = X @ W[neg_samples].T + b[0, neg_samples]

        # subtract the log probability of each label under the noise dist
        if self.subtract_log_label_prob:
            n, m = Z_target.shape[0], Z_neg.shape[0]
            Z_target[range(n), ...] -= np.log(p_target)
            Z_neg[range(m), ...] -= np.log(p_neg_samples)

        # only retain the probability of the target under its associated
        # minibatch example
        aa, _, cc = Z_target.shape
        Z_target = Z_target[range(aa), :, range(cc)][..., None]

        # p_target = (n_ex, n_c, 1)
        # p_neg = (n_ex, n_c, n_samples)
        pred_p_target = self.act_fn(Z_target)
        pred_p_neg = self.act_fn(Z_neg)

        # if we're in evaluation mode, ignore the negative samples - just
        # return the binary cross entropy on the targets
        y_pred = pred_p_target
        if self.trainable:
            # (n_ex, n_c, 1 + n_samples) (target is first column)
            y_pred = np.concatenate((y_pred, pred_p_neg), axis=-1)

        n_targets = 1
        y_true = np.zeros_like(y_pred)
        y_true[..., :n_targets] = 1

        # binary cross entropy
        eps = np.finfo(float).eps
        np.clip(y_pred, eps, 1 - eps, y_pred)
        loss = -np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss, Z_target, Z_neg, y_pred, y_true, noise_samples

    def grad(self, retain_grads=True, update_params=True):
        """
        Compute the gradient of the NCE loss with regard to the inputs,
        weights, and biases.

        Parameters
        ----------
        retain_grads : bool
            Whether to include the intermediate parameter gradients computed
            during the backward pass in the final parameter update. Default is
            True.
        update_params : bool
            Whether to perform a single step of gradient descent on the layer
            weights and bias using the calculated gradients. If `retain_grads`
            is False, this option is ignored and the parameter gradients are
            not updated. Default is True.

        Returns
        -------
        dLdX : :py:class:`ndarray <numpy.ndarray>` of shape (`n_ex`, `n_in`) or list of arrays
            The gradient of the loss with regard to the layer input(s) `X`.
        """
        assert self.trainable, "NCE loss is frozen"

        dX = []
        for input_idx, x in enumerate(self.X):
            dx, dw, db = self._grad(x, input_idx)
            dX.append(dx)

            if retain_grads:
                self.gradients["W"] += dw
                self.gradients["b"] += db

        dX = dX[0] if len(self.X) == 1 else dX

        if retain_grads and update_params:
            self.update()

        return dX

    def _grad(self, X, input_idx):
        """Actual computation of gradient wrt. loss weights + input"""
        W, b = self.parameters["W"], self.parameters["b"]

        y_pred = self.derived_variables["y_pred"][input_idx]
        target = self.derived_variables["target"][input_idx]
        y_true = self.derived_variables["out_labels"][input_idx]
        Z_neg = self.derived_variables["noise_logits"][input_idx]
        Z_target = self.derived_variables["target_logits"][input_idx]
        neg_samples = self.derived_variables["noise_samples"][input_idx][0]

        # the number of target classes per minibatch example
        n_targets = 1

        # calculate the grad of the binary cross entropy wrt. the network
        # predictions
        preds, classes = y_pred.flatten(), y_true.flatten()

        dLdp_real = ((1 - classes) / (1 - preds)) - (classes / preds)
        dLdp_real = dLdp_real.reshape(*y_pred.shape)

        # partition the gradients into target and negative sample portions
        dLdy_pred_target = dLdp_real[..., :n_targets]
        dLdy_pred_neg = dLdp_real[..., n_targets:]

        # compute gradients of the loss wrt the data and noise logits
        dLdZ_target = dLdy_pred_target * self.act_fn.grad(Z_target)
        dLdZ_neg = dLdy_pred_neg * self.act_fn.grad(Z_neg)

        # compute param gradients on target + negative samples
        dB_neg = dLdZ_neg.sum(axis=(0, 1))
        dB_target = dLdZ_target.sum(axis=(1, 2))

        dW_neg = (dLdZ_neg.transpose(0, 2, 1) @ X).sum(axis=0)
        dW_target = (dLdZ_target.transpose(0, 2, 1) @ X).sum(axis=1)

        # TODO: can this be done with np.einsum instead?
        dX_target = np.vstack(
            [dLdZ_target[[ix]] @ W[[t]] for ix, t in enumerate(target)]
        )
        dX_neg = dLdZ_neg @ W[neg_samples]

        hits = list(set(target).intersection(set(neg_samples)))
        hit_ixs = [np.where(target == h)[0] for h in hits]

        # adjust param gradients if there's an accidental hit
        if len(hits) != 0:
            hit_ixs = np.concatenate(hit_ixs)
            target = np.delete(target, hit_ixs)
            dB_target = np.delete(dB_target, hit_ixs)
            dW_target = np.delete(dW_target, hit_ixs, 0)

        dX = dX_target + dX_neg

        # use np.add.at to ensure that repeated indices in the target (or
        # possibly in neg_samples if sampling is done with replacement) are
        # properly accounted for
        dB = np.zeros_like(b).flatten()
        np.add.at(dB, target, dB_target)
        np.add.at(dB, neg_samples, dB_neg)
        dB = dB.reshape(*b.shape)

        dW = np.zeros_like(W)
        np.add.at(dW, target, dW_target)
        np.add.at(dW, neg_samples, dW_neg)

        return dX, dW, dB
