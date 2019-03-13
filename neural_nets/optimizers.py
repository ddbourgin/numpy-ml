import numpy as np


class Momentum(object):
    def __init__(self, momentum, lr, param_names):
        """
        Momentum SGD optimizer.

        Equations:
            update[t] = momentum * update[t-1] + lr * grad[t]
            param[t+1] = param[t] - update[t]

        Parameters
        ----------
        momentum : float in range [0, 1]
            The fraction of the previous update to add to the current update
        lr : float
            Learning rate for SGD
        param_names : list of strings
            The names of the parameters to compute a Momentum update for
        """
        self.parameters = {"momentum": momentum, "lr": lr, "param_names": param_names}
        self.prev_updates = {n: None for n in param_names}

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
        assert param_name in self.prev_updates, "Invalid parameter"

        lr = self.parameters["lr"]
        momentum = self.parameters["momentum"]
        prev_update = self.prev_updates[param_name]

        if prev_update == None:
            prev_update = np.zeros_like(param_grad)

        update = momentum * prev_update + lr * param_grad
        self.prev_updates[param_name] = update
        return param - update


class AdaGrad(object):
    """
    A downside of Adagrad is that in case of deep learning, the monotonic
    learning rate usually proves too aggressive and stops learning too early.

    -- Karpathy
    """

    def __init__(self, lr, param_names, eps=1e-7):
        """
        AdaGrad optimizer. Weights that receive high gradients will have their
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
        param_names : list of strings
            The names of the parameters to compute an update for
        eps : float
            Smoothing term to avoid divide-by-zero errors in the update calc
        """
        self.parameters = {
            "lr": lr,
            "eps": eps,
            "param_names": param_names,
            "cache": {n: None for n in param_names},
        }

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
        assert param_name in self.prev_updates, "Invalid parameter"

        lr = self.parameters["lr"]
        eps = self.parameters["eps"]
        cache = self.parameters["cache"]

        if cache[param_name] is None:
            cache[param_name] = np.zeros_like(param_grad)

        cache[param_name] += param_grad ** 2
        update = lr * param_grad / (np.sqrt(cache) + eps)
        self.parameters["cache"] = cache
        return param - update


class RMSProp(object):
    def __init__(self, lr, decay, param_names, eps=1e-7):
        """
        RMSProp optimizer. A refinement of Adagrad to reduce its aggressive,
        monotonically decreasing learning rate. RMSProp uses a *decaying
        average* of the previous squared gradients rather than just the
        immediately preceding squared gradient (as in AdaGrad) for its cache
        value.

        Equations:
            cache[t] = decay * cache[t-1] + (1 - decay) * grad[t] ** 2
            update[t] = lr * grad[t] / (np.sqrt(cache[t]) + eps)
            param[t+1] = param[t] - update[t]

            Note that ** and / operations are elementwise

        Parameters
        ----------
        lr : float
            Learning rate for update
        decay : float in [0, 1]
            Rate of decay in the moving average. Typical values are [0.9, 0.99, 0.999]
        param_names : list of strings
            The names of the parameters to compute an update for
        eps : float
            Smoothing term to avoid divide-by-zero errors in the update calc
        """
        self.parameters = {
            "lr": lr,
            "eps": eps,
            "decay": decay,
            "param_names": param_names,
            "cache": {n: None for n in param_names},
        }

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
        assert param_name in self.prev_updates, "Invalid parameter"

        lr = self.parameters["lr"]
        eps = self.parameters["eps"]
        decay = self.parameters["decay"]
        cache = self.parameters["cache"]

        if cache[param_name] is None:
            cache[param_name] = np.zeros_like(param_grad)

        cache[param_name] = decay * cache + (1 - decay) * param_grad ** 2
        update = lr * param_grad / (np.sqrt(cache) + eps)
        self.parameters["cache"] = cache
        return param - update
