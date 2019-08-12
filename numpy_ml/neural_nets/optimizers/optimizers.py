from copy import deepcopy
from abc import ABC, abstractmethod

import numpy as np
from numpy.linalg import norm


class OptimizerBase(ABC):
    def __init__(self, lr, scheduler=None):
        """
        An abstract base class for all Optimizer objects.

        This should never be used directly.
        """
        from ..initializers import SchedulerInitializer

        self.cache = {}
        self.cur_step = 0
        self.hyperparameters = {}
        self.lr_scheduler = SchedulerInitializer(scheduler, lr=lr)()

    def __call__(self, param, param_grad, param_name, cur_loss=None):
        return self.update(param, param_grad, param_name, cur_loss)

    def step(self):
        """Increment the optimizer step counter by 1"""
        self.cur_step += 1

    def reset_step(self):
        """Reset the step counter to zero"""
        self.cur_step = 0

    def copy(self):
        """Return a copy of the optimizer object"""
        return deepcopy(self)

    def set_params(self, hparam_dict=None, cache_dict=None):
        """Set the parameters of the optimizer object from a dictionary"""
        from ..initializers import SchedulerInitializer

        if hparam_dict is not None:
            for k, v in hparam_dict.items():
                if k in self.hyperparameters:
                    self.hyperparameters[k] = v
                    if k == "lr_scheduler":
                        self.lr_scheduler = SchedulerInitializer(v, lr=None)()

        if cache_dict is not None:
            for k, v in cache_dict.items():
                if k in self.cache:
                    self.cache[k] = v

    @abstractmethod
    def update(self, param, param_grad, param_name, cur_loss=None):
        raise NotImplementedError


class SGD(OptimizerBase):
    def __init__(
        self, lr=0.01, momentum=0.0, clip_norm=None, lr_scheduler=None, **kwargs
    ):
        """
        A stochastic gradient descent optimizer.

        Notes
        -----
        For model parameters :math:`\\theta`, averaged parameter gradients
        :math:`\\nabla_{\\theta} \mathcal{L}`, and learning rate :math:`\eta`,
        the SGD update at timestep `t` is

        .. math::

            \\text{update}^{(t)}
                &=  \\text{momentum} \cdot \\text{update}^{(t-1)} + \eta^{(t)} \\nabla_{\\theta} \mathcal{L}\\\\
            \\theta^{(t+1)}
                &\leftarrow  \\theta^{(t)} - \\text{update}^{(t)}

        Parameters
        ----------
        lr : float
            Learning rate for SGD. If scheduler is not None, this is used as
            the starting learning rate. Default is 0.01.
        momentum : float in range [0, 1]
            The fraction of the previous update to add to the current update.
            If 0, no momentum is applied. Default is 0.
        clip_norm : float
            If not None, all param gradients are scaled to have maximum l2 norm of
            `clip_norm` before computing update. Default is None.
        lr_scheduler : str, :doc:`Scheduler <numpy_ml.neural_nets.schedulers>` object, or None
            The learning rate scheduler. If None, use a constant learning
            rate equal to `lr`. Default is None.
        """
        super().__init__(lr, lr_scheduler)

        self.hyperparameters = {
            "id": "SGD",
            "lr": lr,
            "momentum": momentum,
            "clip_norm": clip_norm,
            "lr_scheduler": str(self.lr_scheduler),
        }

    def __str__(self):
        H = self.hyperparameters
        lr, mm, cn, sc = H["lr"], H["momentum"], H["clip_norm"], H["lr_scheduler"]
        return "SGD(lr={}, momentum={}, clip_norm={}, lr_scheduler={})".format(
            lr, mm, cn, sc
        )

    def update(self, param, param_grad, param_name, cur_loss=None):
        """
        Compute the SGD update for a given parameter

        Parameters
        ----------
        param : :py:class:`ndarray <numpy.ndarray>` of shape (n, m)
            The value of the parameter to be updated.
        param_grad : :py:class:`ndarray <numpy.ndarray>` of shape (n, m)
            The gradient of the loss function with respect to `param_name`.
        param_name : str
            The name of the parameter.
        cur_loss : float
            The training or validation loss for the current minibatch. Used for
            learning rate scheduling e.g., by
            :class:`~numpy_ml.neural_nets.schedulers.KingScheduler`.
            Default is None.

        Returns
        -------
        updated_params : :py:class:`ndarray <numpy.ndarray>` of shape (n, m)
            The value of `param` after applying the momentum update.
        """
        C = self.cache
        H = self.hyperparameters
        momentum, clip_norm = H["momentum"], H["clip_norm"]
        lr = self.lr_scheduler(self.cur_step, cur_loss)

        if param_name not in C:
            C[param_name] = np.zeros_like(param_grad)

        # scale gradient to avoid explosion
        t = np.inf if clip_norm is None else clip_norm
        if norm(param_grad) > t:
            param_grad = param_grad * t / norm(param_grad)

        update = momentum * C[param_name] + lr * param_grad
        self.cache[param_name] = update
        return param - update


#######################################################################
#                      Adaptive Gradient Methods                      #
#######################################################################


class AdaGrad(OptimizerBase):
    def __init__(self, lr=0.01, eps=1e-7, clip_norm=None, lr_scheduler=None, **kwargs):
        """
        An AdaGrad optimizer.

        Notes
        -----
        Weights that receive large gradients will have their effective learning
        rate reduced, while weights that receive small or infrequent updates
        will have their effective learning rate increased.

        Equations::

            cache[t] = cache[t-1] + grad[t] ** 2
            update[t] = lr * grad[t] / (np.sqrt(cache[t]) + eps)
            param[t+1] = param[t] - update[t]

        Note that the ``**`` and `/` operations are elementwise

        "A downside of Adagrad ... is that the monotonic learning rate usually
        proves too aggressive and stops learning too early." [1]

        References
        ----------
        .. [1] Karpathy, A. "CS231n: Convolutional neural networks for visual
           recognition" https://cs231n.github.io/neural-networks-3/

        Parameters
        ----------
        lr : float
            Global learning rate
        eps : float
            Smoothing term to avoid divide-by-zero errors in the update calc.
            Default is 1e-7.
        clip_norm : float or None
            If not None, all param gradients are scaled to have maximum `L2` norm of
            `clip_norm` before computing update. Default is None.
        lr_scheduler : str or :doc:`Scheduler <numpy_ml.neural_nets.schedulers>` object or None
            The learning rate scheduler. If None, use a constant learning
            rate equal to `lr`. Default is None.
        """
        super().__init__(lr, lr_scheduler)

        self.cache = {}
        self.hyperparameters = {
            "id": "AdaGrad",
            "lr": lr,
            "eps": eps,
            "clip_norm": clip_norm,
            "lr_scheduler": str(self.lr_scheduler),
        }

    def __str__(self):
        H = self.hyperparameters
        lr, eps, cn, sc = H["lr"], H["eps"], H["clip_norm"], H["lr_scheduler"]
        return "AdaGrad(lr={}, eps={}, clip_norm={}, lr_scheduler={})".format(
            lr, eps, cn, sc
        )

    def update(self, param, param_grad, param_name, cur_loss=None):
        """
        Compute the AdaGrad update for a given parameter.

        Notes
        -----
        Adjusts the learning rate of each weight based on the magnitudes of its
        gradients (big gradient -> small lr, small gradient -> big lr).

        Parameters
        ----------
        param : :py:class:`ndarray <numpy.ndarray>` of shape (n, m)
            The value of the parameter to be updated
        param_grad : :py:class:`ndarray <numpy.ndarray>` of shape (n, m)
            The gradient of the loss function with respect to `param_name`
        param_name : str
            The name of the parameter
        cur_loss : float or None
            The training or validation loss for the current minibatch. Used for
            learning rate scheduling e.g., by
            :class:`~numpy_ml.neural_nets.schedulers.KingScheduler`.
            Default is None.

        Returns
        -------
        updated_params : :py:class:`ndarray <numpy.ndarray>` of shape (n, m)
            The value of `param` after applying the AdaGrad update
        """
        C = self.cache
        H = self.hyperparameters
        eps, clip_norm = H["eps"], H["clip_norm"]
        lr = self.lr_scheduler(self.cur_step, cur_loss)

        if param_name not in C:
            C[param_name] = np.zeros_like(param_grad)

        # scale gradient to avoid explosion
        t = np.inf if clip_norm is None else clip_norm
        if norm(param_grad) > t:
            param_grad = param_grad * t / norm(param_grad)

        C[param_name] += param_grad ** 2
        update = lr * param_grad / (np.sqrt(C[param_name]) + eps)
        self.cache = C
        return param - update


class RMSProp(OptimizerBase):
    def __init__(
        self, lr=0.001, decay=0.9, eps=1e-7, clip_norm=None, lr_scheduler=None, **kwargs
    ):
        """
        RMSProp optimizer.

        Notes
        -----
        RMSProp was proposed as a refinement of :class:`AdaGrad` to reduce its
        aggressive, monotonically decreasing learning rate.

        RMSProp uses a *decaying average* of the previous squared gradients
        (second moment) rather than just the immediately preceding squared
        gradient for its `previous_update` value.

        Equations::

            cache[t] = decay * cache[t-1] + (1 - decay) * grad[t] ** 2
            update[t] = lr * grad[t] / (np.sqrt(cache[t]) + eps)
            param[t+1] = param[t] - update[t]

        Note that the ``**`` and ``/`` operations are elementwise.

        Parameters
        ----------
        lr : float
            Learning rate for update. Default is 0.001.
        decay : float in [0, 1]
            Rate of decay for the moving average. Typical values are [0.9,
            0.99, 0.999]. Default is 0.9.
        eps : float
            Constant term to avoid divide-by-zero errors during the update calc. Default is 1e-7.
        clip_norm : float or None
            If not None, all param gradients are scaled to have maximum l2 norm of
            `clip_norm` before computing update. Default is None.
        lr_scheduler : str or :doc:`Scheduler <numpy_ml.neural_nets.schedulers>` object or None
            The learning rate scheduler. If None, use a constant learning
            rate equal to `lr`. Default is None.
        """
        super().__init__(lr, lr_scheduler)

        self.cache = {}
        self.hyperparameters = {
            "id": "RMSProp",
            "lr": lr,
            "eps": eps,
            "decay": decay,
            "clip_norm": clip_norm,
            "lr_scheduler": str(self.lr_scheduler),
        }

    def __str__(self):
        H = self.hyperparameters
        sc = H["lr_scheduler"]
        lr, eps, dc, cn = H["lr"], H["eps"], H["decay"], H["clip_norm"]
        return "RMSProp(lr={}, eps={}, decay={}, clip_norm={}, lr_scheduler={})".format(
            lr, eps, dc, cn, sc
        )

    def update(self, param, param_grad, param_name, cur_loss=None):
        """
        Compute the RMSProp update for a given parameter.

        Parameters
        ----------
        param : :py:class:`ndarray <numpy.ndarray>` of shape (n, m)
            The value of the parameter to be updated
        param_grad : :py:class:`ndarray <numpy.ndarray>` of shape (n, m)
            The gradient of the loss function with respect to `param_name`
        param_name : str
            The name of the parameter
        cur_loss : float or None
            The training or validation loss for the current minibatch. Used for
            learning rate scheduling e.g., by
            :class:`~numpy_ml.neural_nets.schedulers.KingScheduler`.
            Default is None.

        Returns
        -------
        updated_params : :py:class:`ndarray <numpy.ndarray>` of shape (n, m)
            The value of `param` after applying the RMSProp update.
        """
        C = self.cache
        H = self.hyperparameters
        eps, decay, clip_norm = H["eps"], H["decay"], H["clip_norm"]
        lr = self.lr_scheduler(self.cur_step, cur_loss)

        if param_name not in C:
            C[param_name] = np.zeros_like(param_grad)

        # scale gradient to avoid explosion
        t = np.inf if clip_norm is None else clip_norm
        if norm(param_grad) > t:
            param_grad = param_grad * t / norm(param_grad)

        C[param_name] = decay * C[param_name] + (1 - decay) * param_grad ** 2
        update = lr * param_grad / (np.sqrt(C[param_name]) + eps)
        self.cache = C
        return param - update


class Adam(OptimizerBase):
    def __init__(
        self,
        lr=0.001,
        decay1=0.9,
        decay2=0.999,
        eps=1e-7,
        clip_norm=None,
        lr_scheduler=None,
        **kwargs
    ):
        """
        Adam (adaptive moment estimation) optimization algorithm.

        Notes
        -----
        Designed to combine the advantages of :class:`AdaGrad`, which works
        well with sparse gradients, and :class:`RMSProp`, which works well in
        online and non-stationary settings.

        Parameters
        ----------
        lr : float
            Learning rate for update. This parameter is ignored if using
            :class:`~numpy_ml.neural_nets.schedulers.NoamScheduler`.
            Default is 0.001.
        decay1 : float
            The rate of decay to use for in running estimate of the first
            moment (mean) of the gradient. Default is 0.9.
        decay2 : float
            The rate of decay to use for in running estimate of the second
            moment (variance) of the gradient. Default is 0.999.
        eps : float
            Constant term to avoid divide-by-zero errors during the update
            calc. Default is 1e-7.
        clip_norm : float
            If not None, all param gradients are scaled to have maximum l2 norm of
            `clip_norm` before computing update. Default is None.
        lr_scheduler : str, or :doc:`Scheduler <numpy_ml.neural_nets.schedulers>` object, or None
            The learning rate scheduler. If None, use a constant learning rate
            equal to `lr`. Default is None.
        """
        super().__init__(lr, lr_scheduler)

        self.cache = {}
        self.hyperparameters = {
            "id": "Adam",
            "lr": lr,
            "eps": eps,
            "decay1": decay1,
            "decay2": decay2,
            "clip_norm": clip_norm,
            "lr_scheduler": str(self.lr_scheduler),
        }

    def __str__(self):
        H = self.hyperparameters
        lr, d1, d2 = H["lr"], H["decay1"], H["decay2"]
        eps, cn, sc = H["eps"], H["clip_norm"], H["lr_scheduler"]
        return "Adam(lr={}, decay1={}, decay2={}, eps={}, clip_norm={}, lr_scheduler={})".format(
            lr, d1, d2, eps, cn, sc
        )

    def update(self, param, param_grad, param_name, cur_loss=None):
        """
        Compute the Adam update for a given parameter.

        Parameters
        ----------
        param : :py:class:`ndarray <numpy.ndarray>` of shape (n, m)
            The value of the parameter to be updated.
        param_grad : :py:class:`ndarray <numpy.ndarray>` of shape (n, m)
            The gradient of the loss function with respect to `param_name`.
        param_name : str
            The name of the parameter.
        cur_loss : float
            The training or validation loss for the current minibatch. Used for
            learning rate scheduling e.g., by
            :class:`~numpy_ml.neural_nets.schedulers.KingScheduler`. Default is
            None.

        Returns
        -------
        updated_params : :py:class:`ndarray <numpy.ndarray>` of shape (n, m)
            The value of `param` after applying the Adam update.
        """
        C = self.cache
        H = self.hyperparameters
        d1, d2 = H["decay1"], H["decay2"]
        eps, clip_norm = H["eps"], H["clip_norm"]
        lr = self.lr_scheduler(self.cur_step, cur_loss)

        if param_name not in C:
            C[param_name] = {
                "t": 0,
                "mean": np.zeros_like(param_grad),
                "var": np.zeros_like(param_grad),
            }

        # scale gradient to avoid explosion
        t = np.inf if clip_norm is None else clip_norm
        if norm(param_grad) > t:
            param_grad = param_grad * t / norm(param_grad)

        t = C[param_name]["t"] + 1
        var = C[param_name]["var"]
        mean = C[param_name]["mean"]

        # update cache
        C[param_name]["t"] = t
        C[param_name]["var"] = d2 * var + (1 - d2) * param_grad ** 2
        C[param_name]["mean"] = d1 * mean + (1 - d1) * param_grad
        self.cache = C

        # calc unbiased moment estimates and Adam update
        v_hat = C[param_name]["var"] / (1 - d2 ** t)
        m_hat = C[param_name]["mean"] / (1 - d1 ** t)
        update = lr * m_hat / (np.sqrt(v_hat) + eps)
        return param - update
