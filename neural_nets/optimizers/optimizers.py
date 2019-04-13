from copy import deepcopy
from abc import ABC, abstractmethod

import numpy as np
from numpy.linalg import norm


class OptimizerBase(ABC):
    def __init__(self):
        self.cache = {}
        self.hyperparameters = {}
        pass

    def __call__(self, param, param_grad, param_name):
        return self.update(param, param_grad, param_name)

    def copy(self):
        return deepcopy(self)

    def set_params(self, hparam_dict=None, cache_dict=None):
        if hparam_dict is not None:
            for k, v in hparam_dict.items():
                if k in self.hyperparameters:
                    self.hyperparameters[k] = v
        if cache_dict is not None:
            for k, v in cache_dict.items():
                if k in self.cache:
                    self.cache[k] = v

    @abstractmethod
    def update(self, param, param_grad, param_name):
        raise NotImplementedError


class SGD(OptimizerBase):
    def __init__(self, lr=0.01, momentum=0.0, clip_norm=None, **kwargs):
        """
        Stochastic gradient descent optimizer.

        Equations:
            update[t] = cache[t] = momentum * cache[t-1] + lr * grad[t]
            param[t+1] = param[t] - update[t]

        Parameters
        ----------
        lr : float (default: 0.01)
            Learning rate for SGD
        momentum : float in range [0, 1] (default: 0)
            The fraction of the previous update to add to the current update.
            If 0, no momentum is applied.
        clip_norm : float (default: None)
            If not None, all param gradients are scaled to have maximum l2 norm of
            `clip_norm` before computing update.
        """
        super().__init__()

        self.cache = {}
        self.hyperparameters = {
            "id": "SGD",
            "lr": lr,
            "momentum": momentum,
            "clip_norm": clip_norm,
        }

    def __str__(self):
        H = self.hyperparameters
        lr, mm, cn = H["lr"], H["momentum"], H["clip_norm"]
        return "SGD(lr={}, momentum={}, clip_norm={})".format(lr, mm, cn)

    def update(self, param, param_grad, param_name):
        """
        Compute the momentum update for a given parameter

        Parameters
        ----------
        param : numpy array of shape (n, m)
            The value of the parameter to be updated
        param_grad : numpy array of shape (n, m)
            The gradient of the loss function with respect to `param_name`
        param_name : str
            The name of the parameter

        Returns
        -------
        updated_params : numpy array of shape (n, m)
            The value of `param` after applying the momentum update
        """
        C = self.cache
        lr = self.hyperparameters["lr"]
        momentum = self.hyperparameters["momentum"]
        clip_norm = self.hyperparameters["clip_norm"]

        if param_name not in C:
            C[param_name] = np.zeros_like(param_grad)

        # scale gradient to avoid explosion
        t = np.inf if clip_norm is None else clip_norm
        if norm(param_grad) > t:
            param_grad = param_grad * t / norm(param_grad)

        update = momentum * C[param_name] + lr * param_grad
        self.cache[param_name] = update
        return param - update


class AdaGrad(OptimizerBase):
    """
    A downside of Adagrad ... is that the monotonic learning rate usually
    proves too aggressive and stops learning too early.

    -- Andrej Karpathy
    """

    def __init__(self, lr=0.01, eps=1e-7, clip_norm=None, **kwargs):
        """
        AdaGrad optimizer. Weights that receive large gradients will have their
        effective learning rate reduced, while weights that receive small or
        infrequent updates will have their effective learning rate increased.

        Equations:
            cache[t] = cache[t-1] + grad[t] ** 2
            update[t] = lr * grad[t] / (np.sqrt(cache[t]) + eps)
            param[t+1] = param[t] - update[t]

            Note that ** and / operations are elementwise

        Parameters
        ----------
        lr : float
            Global learning rate
        eps : float (default: 1e-7)
            Smoothing term to avoid divide-by-zero errors in the update calc
        clip_norm : float (default: None)
            If not None, all param gradients are scaled to have maximum l2 norm of
            `clip_norm` before computing update.
        """
        super().__init__()

        self.cache = {}
        self.hyperparameters = {
            "id": "AdaGrad",
            "lr": lr,
            "eps": eps,
            "clip_norm": clip_norm,
        }

    def __str__(self):
        H = self.hyperparameters
        lr, eps, cn = H["lr"], H["eps"], H["clip_norm"]
        return "AdaGrad(lr={}, eps={}, clip_norm={})".format(lr, eps, cn)

    def update(self, param, param_grad, param_name):
        """
        Compute the AdaGrad update for a given parameter. Adjusts the
        learning rate of each weight based on the magnitudes of its gradients
        (big gradient -> small lr, small gradient -> big lr).

        Parameters
        ----------
        param : numpy array of shape (n, m)
            The value of the parameter to be updated
        param_grad : numpy array of shape (n, m)
            The gradient of the loss function with respect to `param_name`
        param_name : str
            The name of the parameter

        Returns
        -------
        updated_params : numpy array of shape (n, m)
            The value of `param` after applying the AdaGrad update
        """
        C = self.cache
        lr = self.hyperparameters["lr"]
        eps = self.hyperparameters["eps"]
        clip_norm = self.hyperparameters["clip_norm"]

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
    def __init__(self, lr=0.001, decay=0.9, eps=1e-7, clip_norm=None, **kwargs):
        """
        RMSProp optimizer. A refinement of Adagrad to reduce its aggressive,
        monotonically decreasing learning rate. RMSProp uses a *decaying
        average* of the previous squared gradients (second moment) rather than
        just the immediately preceding squared gradient for its
        `previous_update` value.

        Equations:
            cache[t] = decay * cache[t-1] + (1 - decay) * grad[t] ** 2
            update[t] = lr * grad[t] / (np.sqrt(cache[t]) + eps)
            param[t+1] = param[t] - update[t]

            Note that ** and / operations are elementwise

        Parameters
        ----------
        lr : float (default: 0.001)
            Learning rate for update
        decay : float in [0, 1] (default: 0.9)
            Rate of decay for the moving average. Typical values are [0.9, 0.99, 0.999]
        eps : float (default: 1e-7)
            Constant term to avoid divide-by-zero errors during the update calc
        clip_norm : float (default : None)
            If not None, all param gradients are scaled to have maximum l2 norm of
            `clip_norm` before computing update.
        """
        super().__init__()

        self.cache = {}
        self.hyperparameters = {
            "id": "RMSProp",
            "lr": lr,
            "eps": eps,
            "decay": decay,
            "clip_norm": clip_norm,
        }

    def __str__(self):
        H = self.hyperparameters
        lr, eps, dc, cn = H["lr"], H["eps"], H["decay"], H["clip_norm"]
        return "RMSProp(lr={}, eps={}, decay={}, clip_norm={})".format(lr, eps, dc, cn)

    def update(self, param, param_grad, param_name):
        """
        Compute the RMSProp update for a given parameter.

        Parameters
        ----------
        param : numpy array of shape (n, m)
            The value of the parameter to be updated
        param_grad : numpy array of shape (n, m)
            The gradient of the loss function with respect to `param_name`
        param_name : str
            The name of the parameter

        Returns
        -------
        updated_params : numpy array of shape (n, m)
            The value of `param` after applying the RMSProp update
        """
        C = self.cache
        lr = self.hyperparameters["lr"]
        eps = self.hyperparameters["eps"]
        decay = self.hyperparameters["decay"]
        clip_norm = self.hyperparameters["clip_norm"]

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
        self, lr=0.001, decay1=0.9, decay2=0.999, eps=1e-7, clip_norm=None, **kwargs
    ):
        """
        Adam (adaptive moment estimation) optimization algorithm. Designed to
        combine the advantages of AdaGrad, which works well with sparse
        gradients, and RMSProp, which works well in on-line and non-stationary
        settings.

        Parameters
        ----------
        lr : float (default: 0.001)
            pass
        decay1: float (default: 0.9)
            The rate of decay to use for in running estimate of the first
            moment (mean) of the gradient
        decay2: float (default: 0.999)
            The rate of decay to use for in running estimate of the second
            moment (var) of the gradient
        eps : float (default: 1e-7)
            Constant term to avoid divide-by-zero errors during the update calc
        clip_norm : float (default : None)
            If not None, all param gradients are scaled to have maximum l2 norm of
            `clip_norm` before computing update.
        """
        super().__init__()

        self.cache = {}
        self.hyperparameters = {
            "id": "Adam",
            "lr": lr,
            "eps": eps,
            "decay1": decay1,
            "decay2": decay2,
            "clip_norm": clip_norm,
        }

    def __str__(self):
        H = self.hyperparameters
        eps, cn = H["eps"], H["clip_norm"]
        lr, d1, d2, = H["lr"], H["decay1"], H["decay2"]
        return "Adam(lr={}, decay1={}, decay2={}, eps={}, clip_norm={})".format(
            lr, d1, d2, eps, cn
        )

    def update(self, param, param_grad, param_name):
        """
        Compute the Adam update for a given parameter.

        Parameters
        ----------
        param : numpy array of shape (n, m)
            The value of the parameter to be updated
        param_grad : numpy array of shape (n, m)
            The gradient of the loss function with respect to `param_name`
        param_name : str
            The name of the parameter

        Returns
        -------
        updated_params : numpy array of shape (n, m)
            The value of `param` after applying the Adam update
        """
        C = self.cache
        H = self.hyperparameters
        eps, clip_norm = H["eps"], H["clip_norm"]
        lr, d1, d2, = H["lr"], H["decay1"], H["decay2"]

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
