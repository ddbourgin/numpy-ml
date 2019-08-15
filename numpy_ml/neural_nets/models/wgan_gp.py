from time import time
from collections import OrderedDict

import numpy as np

from ..utils import minibatch
from ..layers import FullyConnected
from ..losses import WGAN_GPLoss


class WGAN_GP(object):
    """
    A Wasserstein generative adversarial network (WGAN) architecture with
    gradient penalty (GP).

    Notes
    -----
    In contrast to a regular WGAN, WGAN-GP uses gradient penalty on the
    generator rather than weight clipping to encourage the 1-Lipschitz
    constraint:

    .. math::

        | \\text{Generator}(\mathbf{x}_1) - \\text{Generator}(\mathbf{x}_2) |
            \leq |\mathbf{x}_1 - \mathbf{x}_2 | \ \ \ \ \\forall \mathbf{x}_1, \mathbf{x}_2

    In other words, the generator must have input gradients with a norm of at
    most 1 under the :math:`\mathbf{X}_{real}` and :math:`\mathbf{X}_{fake}`
    data distributions.

    To enforce this constraint, WGAN-GP penalizes the model if the generator
    gradient norm moves away from a target norm of 1. See
    :class:`~numpy_ml.neural_nets.losses.WGAN_GPLoss` for more details.

    In contrast to a standard WGAN, WGAN-GP avoids using BatchNorm in the
    critic, as correlation between samples in a batch can impact the stability
    of the gradient penalty.

    WGAP-GP architecture:

    .. code-block:: text

        X_real ------------------------|
                                        >---> [Critic] --> Y_out
        Z --> [Generator] --> X_fake --|

    where ``[Generator]`` is

    .. code-block:: text

        FC1 -> ReLU -> FC2 -> ReLU -> FC3 -> ReLU -> FC4

    and ``[Critic]`` is

    .. code-block:: text

        FC1 -> ReLU -> FC2 -> ReLU -> FC3 -> ReLU -> FC4

    and

    .. math::

        Z \sim \mathcal{N}(0, 1)
    """

    def __init__(
        self,
        g_hidden=512,
        init="he_uniform",
        optimizer="RMSProp(lr=0.0001)",
        debug=False,
    ):
        """
        Wasserstein generative adversarial network with gradient penalty.

        Parameters
        ----------
        g_hidden : int
            The number of units in the critic and generator hidden layers.
            Default is 512.
        init : str
            The weight initialization strategy. Valid entries are
            {'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform',
            'std_normal', 'trunc_normal'}. Default is "he_uniform".
        optimizer : str or :doc:`Optimizer <numpy_ml.neural_nets.optimizers>` object or None
            The optimization strategy to use when performing gradient updates.
            If None, use the :class:`~numpy_ml.neural_nets.optimizers.SGD`
            optimizer with default parameters. Default is "RMSProp(lr=0.0001)".
        debug : bool
            Whether to store additional intermediate output within
            ``self.derived_variables``. Default is False.
        """
        self.init = init
        self.debug = debug
        self.g_hidden = g_hidden
        self.optimizer = optimizer

        self.lambda_ = None
        self.n_steps = None
        self.batchsize = None

        self.is_initialized = False

    def _init_params(self):
        self._dv = {}
        self._gr = {}
        self._build_critic()
        self._build_generator()
        self.is_initialized = True

    def _build_generator(self):
        """
        FC1 -> ReLU -> FC2 -> ReLU -> FC3 -> ReLU -> FC4
        """
        self.generator = OrderedDict()
        self.generator["FC1"] = FullyConnected(
            self.g_hidden, act_fn="ReLU", optimizer=self.optimizer, init=self.init
        )
        self.generator["FC2"] = FullyConnected(
            self.g_hidden, act_fn="ReLU", optimizer=self.optimizer, init=self.init
        )
        self.generator["FC3"] = FullyConnected(
            self.g_hidden, act_fn="ReLU", optimizer=self.optimizer, init=self.init
        )
        self.generator["FC4"] = FullyConnected(
            self.n_feats,
            act_fn="Affine(slope=1, intercept=0)",
            optimizer=self.optimizer,
            init=self.init,
        )

    def _build_critic(self):
        """
        FC1 -> ReLU -> FC2 -> ReLU -> FC3 -> ReLU -> FC4
        """
        self.critic = OrderedDict()
        self.critic["FC1"] = FullyConnected(
            self.g_hidden, act_fn="ReLU", optimizer=self.optimizer, init=self.init
        )
        self.critic["FC2"] = FullyConnected(
            self.g_hidden, act_fn="ReLU", optimizer=self.optimizer, init=self.init
        )
        self.critic["FC3"] = FullyConnected(
            self.g_hidden, act_fn="ReLU", optimizer=self.optimizer, init=self.init
        )
        self.critic["FC4"] = FullyConnected(
            1,
            act_fn="Affine(slope=1, intercept=0)",
            optimizer=self.optimizer,
            init=self.init,
        )

    @property
    def hyperparameters(self):
        return {
            "init": self.init,
            "lambda_": self.lambda_,
            "g_hidden": self.g_hidden,
            "n_steps": self.n_steps,
            "optimizer": self.optimizer,
            "batchsize": self.batchsize,
            "c_updates_per_epoch": self.c_updates_per_epoch,
            "components": {
                "critic": {k: v.hyperparameters for k, v in self.critic.items()},
                "generator": {k: v.hyperparameters for k, v in self.generator.items()},
            },
        }

    @property
    def parameters(self):
        return {
            "components": {
                "critic": {k: v.parameters for k, v in self.critic.items()},
                "generator": {k: v.parameters for k, v in self.generator.items()},
            }
        }

    @property
    def derived_variables(self):
        C = self.critic.items()
        G = self.generator.items()
        dv = {
            "components": {
                "critic": {k: v.derived_variables for k, v in C},
                "generator": {k: v.derived_variables for k, v in G},
            }
        }
        dv.update(self._dv)
        return dv

    @property
    def gradients(self):
        grads = {
            "dC_Y_fake": None,
            "dC_Y_real": None,
            "dG_Y_fake": None,
            "dC_gradInterp": None,
            "components": {
                "critic": {k: v.gradients for k, v in self.critic.items()},
                "generator": {k: v.gradients for k, v in self.generator.items()},
            },
        }
        grads.update(self._gr)
        return grads

    def forward(self, X, module, retain_derived=True):
        """
        Perform the forward pass for either the generator or the critic.

        Parameters
        ----------
        X : :py:class:`ndarray <numpy.ndarray>` of shape `(batchsize, \*)`
            Input data
        module : {'C' or 'G'}
            Whether to perform the forward pass for the critic ('C') or for the
            generator ('G').
        retain_derived : bool
            Whether to retain the variables calculated during the forward pass
            for use later during backprop. If False, this suggests the layer
            will not be expected to backprop through wrt. this input. Default
            is True.

        Returns
        -------
        out : :py:class:`ndarray <numpy.ndarray>` of shape `(batchsize, \*)`
            The output of the final layer of the module.
        Xs : dict
            A dictionary with layer ids as keys and values corresponding to the
            input to each intermediate layer during the forward pass. Useful
            during debugging.
        """
        if module == "G":
            mod = self.generator
        elif module == "C":
            mod = self.critic
        else:
            raise ValueError("Unrecognized module name: {}".format(module))

        Xs = {}
        out, rd = X, retain_derived
        for k, v in mod.items():
            Xs[k] = out
            out = v.forward(out, retain_derived=rd)
        return out, Xs

    def backward(self, grad, module, retain_grads=True):
        """
        Perform the backward pass for either the generator or the critic.

        Parameters
        ----------
        grad : :py:class:`ndarray <numpy.ndarray>` of shape `(batchsize, \*)` or list of arrays
            Gradient of the loss with respect to module output(s).
        module : {'C' or 'G'}
            Whether to perform the backward pass for the critic ('C') or for the
            generator ('G').
        retain_grads : bool
            Whether to include the intermediate parameter gradients computed
            during the backward pass in the final parameter update. Default is True.

        Returns
        -------
        out : :py:class:`ndarray <numpy.ndarray>` of shape `(batchsize, \*)`
            The gradient of the loss with respect to the module input.
        dXs : dict
            A dictionary with layer ids as keys and values corresponding to the
            input to each intermediate layer during the backward pass. Useful
            during debugging.
        """
        if module == "G":
            mod = self.generator
        elif module == "C":
            mod = self.critic
        else:
            raise ValueError("Unrecognized module name: {}".format(module))

        dXs = {}
        out, rg = grad, retain_grads
        for k, v in reversed(list(mod.items())):
            dXs[k] = out
            out = v.backward(out, retain_grads=rg)
        return out, dXs

    def _dGradInterp(self, dLdGradInterp, dYi_outs):
        """
        Compute the gradient penalty's contribution to the critic loss and
        update the parameter gradients accordingly.

        Parameters
        ----------
        dLdGradInterp : :py:class:`ndarray <numpy.ndarray>` of shape `(batchsize, critic_in_dim)`
            Gradient of `Y_interp` with respect to `X_interp`.
        dYi_outs : dict
            The intermediate outputs generated during the backward pass when
            computing `dLdGradInterp`.
        """
        dy = dLdGradInterp
        for k, v in self.critic.items():
            X = v.X[-1]  # layer input during forward pass
            dy, dW, dB = v._bwd2(dy, X, dYi_outs[k][2])
            self.critic[k].gradients["W"] += dW
            self.critic[k].gradients["b"] += dB

    def update_critic(self, X_real):
        """
        Compute parameter gradients for the critic on a single minibatch.

        Parameters
        ----------
        X_real : :py:class:`ndarray <numpy.ndarray>` of shape `(batchsize, n_feats)`
            Input data.

        Returns
        -------
        C_loss : float
            The critic loss on the current data.
        """
        self.flush_gradients("C")

        n_ex = X_real.shape[0]
        noise = np.random.randn(*X_real.shape)

        # generate and score the real and fake data
        X_fake, Xf_outs = self.forward(noise, "G")
        Y_real, Yr_outs = self.forward(X_real, "C")
        Y_fake, Yf_outs = self.forward(X_fake, "C")

        # sample a random point on the linear interpolation between real and
        # fake data and compute its score
        alpha = np.random.rand(n_ex, 1)
        X_interp = alpha * X_real + (1 - alpha) * X_fake
        Y_interp, Yi_outs = self.forward(X_interp, "C")

        # compute the gradient of Y_interp wrt. X_interp
        # Note that we don't save intermediate gradients here since this is not
        # the real backward pass
        dLdy = [0, 0, np.ones_like(Y_interp)]
        (_, _, gradInterp), dYi_outs = self.backward(dLdy, "C", retain_grads=False)

        # calculate critic loss and differentiate with respect to each term
        C_loss = self.loss(Y_fake, "C", Y_real, gradInterp)
        dY_real, dY_fake, dGrad_interp = self.loss.grad(Y_fake, "C", Y_real, gradInterp)

        # compute `dY_real` and `dY_fake` contributions to critic loss, update
        # param gradients accordingly
        self.backward([dY_real, dY_fake, 0], "C")

        # compute `gradInterp`'s contribution to the critic loss, updating
        # param gradients accordingly
        self._dGradInterp(dGrad_interp, dYi_outs)

        # cache intermediate vars for the generator update
        self._dv["alpha"] = alpha
        self._dv["Y_fake"] = Y_fake

        # log additional intermediate values for debugging
        if self.debug:
            self._dv["G_fwd_X_fake"] = {}
            self._dv["C_fwd_Y_real"] = {}
            self._dv["C_fwd_Y_fake"] = {}
            self._dv["C_fwd_Y_interp"] = {}

            N = len(self.critic.keys())
            N2 = len(self.generator.keys())

            for i in range(N2):
                self._dv["G_fwd_X_fake"]["FC" + str(i)] = Xf_outs["FC" + str(i + 1)]

            for i in range(N):
                self._dv["C_fwd_Y_real"]["FC" + str(i)] = Yr_outs["FC" + str(i + 1)]
                self._dv["C_fwd_Y_fake"]["FC" + str(i)] = Yf_outs["FC" + str(i + 1)]
                self._dv["C_fwd_Y_interp"]["FC" + str(i)] = Yi_outs["FC" + str(i + 1)]

            self._dv["C_fwd_Y_real"]["FC" + str(N)] = Y_real
            self._dv["C_fwd_Y_fake"]["FC" + str(N)] = Y_fake
            self._dv["G_fwd_X_fake"]["FC" + str(N2)] = X_fake
            self._dv["C_fwd_Y_interp"]["FC" + str(N)] = Y_interp
            self._dv["C_dY_interp_wrt"] = {k: v[2] for k, v in dYi_outs.items()}

            self._dv["noise"] = noise
            self._dv["X_fake"] = X_fake
            self._dv["X_real"] = X_real
            self._dv["Y_real"] = Y_real
            self._dv["Y_fake"] = Y_fake
            self._dv["C_loss"] = C_loss
            self._dv["dY_real"] = dY_real
            self._dv["dC_Y_fake"] = dY_fake
            self._dv["X_interp"] = X_interp
            self._dv["Y_interp"] = Y_interp
            self._dv["gradInterp"] = gradInterp
            self._dv["dGrad_interp"] = dGrad_interp

        return C_loss

    def update_generator(self, X_shape):
        """
        Compute parameter gradients for the generator on a single minibatch.

        Parameters
        ----------
        X_shape : tuple of `(batchsize, n_feats)`
            Shape for the input batch.

        Returns
        -------
        G_loss : float
            The generator loss on the fake data (generated during the critic
            update)
        """
        self.flush_gradients("G")
        Y_fake = self.derived_variables["Y_fake"]

        n_ex, _ = Y_fake.shape
        G_loss = -Y_fake.mean()
        dG_loss = -np.ones_like(Y_fake) / n_ex
        self.backward(dG_loss, "G")

        if self.debug:
            self._dv["G_loss"] = G_loss
            self._dv["dG_Y_fake"] = dG_loss

        return G_loss

    def flush_gradients(self, module):
        """Reset parameter gradients to 0 after an update."""
        if module == "G":
            mod = self.generator
        elif module == "C":
            mod = self.critic
        else:
            raise ValueError("Unrecognized module name: {}".format(module))

        for k, v in mod.items():
            v.flush_gradients()

    def update(self, module, module_loss=None):
        """Perform gradient updates and flush gradients upon completion"""
        if module == "G":
            mod = self.generator
        elif module == "C":
            mod = self.critic
        else:
            raise ValueError("Unrecognized module name: {}".format(module))

        for k, v in reversed(list(mod.items())):
            v.update(module_loss)
        self.flush_gradients(module)

    def fit(
        self,
        X_real,
        lambda_,
        n_steps=1000,
        batchsize=128,
        c_updates_per_epoch=5,
        verbose=True,
    ):
        """
        Fit WGAN_GP on a training dataset.

        Parameters
        ----------
        X_real : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, n_feats)`
            Training dataset
        lambda_ : float
            Gradient penalty coefficient for the critic loss
        n_steps : int
            The maximum number of generator updates to perform. Default is
            1000.
        batchsize : int
            Number of examples to use in each training minibatch. Default is
            128.
        c_updates_per_epoch : int
            The number of critic updates to perform at each generator update.
        verbose : bool
            Print loss values after each update. If False, only print loss
            every 100 steps. Default is True.
        """
        self.lambda_ = lambda_
        self.verbose = verbose
        self.n_steps = n_steps
        self.batchsize = batchsize
        self.c_updates_per_epoch = c_updates_per_epoch

        # adjust output of the generator to match the dimensionality of X
        if not self.is_initialized:
            self.n_feats = X_real.shape[1]
            self._init_params()

        # (re-)initialize loss
        prev_C, prev_G = np.inf, np.inf
        self.loss = WGAN_GPLoss(lambda_=self.lambda_)

        # training loop
        NC, NG = self.c_updates_per_epoch, self.n_steps
        for i in range(NG):
            estart = time()
            batch_generator, _ = minibatch(X_real, batchsize, shuffle=False)

            for j, b_ix in zip(range(NC), batch_generator):
                bstart = time()
                X_batch = X_real[b_ix]
                C_loss = self.update_critic(X_batch)

                # for testing, don't perform gradient update so we can inspect each grad
                if not self.debug:
                    self.update("C", C_loss)

                if self.verbose:
                    fstr = "\t[Critic batch {}] Critic loss: {:.3f} {:.3f}∆ ({:.1f}s/batch)"
                    print(fstr.format(j + 1, C_loss, prev_C - C_loss, time() - bstart))
                    prev_C = C_loss

            # generator update
            G_loss = self.update_generator(X_batch.shape)

            # for testing, don't perform gradient update so we can inspect each grad
            if not self.debug:
                self.update("G", G_loss)

            if i % 99 == 0:
                fstr = "[Epoch {}] Gen. loss: {:.3f}  Critic loss: {:.3f}"
                print(fstr.format(i + 1, G_loss, C_loss))

            elif self.verbose:
                fstr = "[Epoch {}] Gen. loss: {:.3f}  {:.3f}∆ ({:.1f}s/epoch)"
                print(fstr.format(i + 1, G_loss, prev_G - G_loss, time() - estart))
                prev_G = G_loss
