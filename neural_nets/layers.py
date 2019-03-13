from abc import ABC, abstractmethod

import numpy as np

from utils import pad2D, conv2D
from activations import Tanh, Sigmoid, ReLU, Linear
from wrappers import Dropout


class LayerBase(ABC):
    def __init__(self, n_in, n_out, act_fn):
        self.X = None
        self.trainable = True

        self.n_in = n_in
        self.n_out = n_out
        self.act_fn = act_fn

        super().__init__()

    @abstractmethod
    def _init_params(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def forward(self, z, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def backward(self, out, **kwargs):
        raise NotImplementedError

    def freeze(self):
        self.trainable = False

    def unfreeze(self):
        self.trainable = True

    def flush_gradients(self):
        assert self.trainable, "Layer is frozen"
        self.X = []
        for k, v in self.derived_variables.items():
            self.derived_variables[k] = None

        for k, v in self.gradients.items():
            self.gradients[k] = np.zeros_like(v)

    def update(self, lr):
        assert self.trainable, "Layer is frozen"
        for k, v in self.gradients.items():
            if k in self.parameters:
                self.parameters[k] -= lr * v
        self.flush_gradients()

    def set_params(self, summary_dict):
        layer = self
        if "parameters" in summary_dict:
            for k, v in summary_dict["parameters"].items():
                if k in self.parameters:
                    self.parameters[k] = v

        if "hyperparameters" in summary_dict:
            for k, v in summary_dict["hyperparameters"].items():
                if k in self.hyperparameters:
                    if k == "act_fn" and v == "ReLU":
                        self.act_fn = ReLU()
                    elif k == "act_fn" and v == "Sigmoid":
                        self.act_fn = Sigmoid()
                    elif k == "act_fn" and v == "Tanh":
                        self.act_fn = Tanh()
                    elif k == "act_fn" and v == "Linear":
                        self.act_fn = Linear()
                    if k != "wrappers":
                        self.hyperparameters[k] = v

            if "wrappers" in summary_dict["hyperparameters"]:
                for wr in summary_dict["hyperparameters"]["wrappers"]:
                    if wr["wrapper"] == "Dropout":
                        layer = Dropout(self, 1)._set_wrapper_params(wr)
                    else:
                        raise NotImplementedError

        if "hyperparameters" not in summary_dict and "parameters" not in summary_dict:
            for k, v in summary_dict.items():
                if k in self.hyperparameters:
                    if k == "act_fn" and v == "ReLU":
                        self.act_fn = ReLU()
                    elif k == "act_fn" and v == "Sigmoid":
                        self.act_fn = Sigmoid()
                    elif k == "act_fn" and v == "Tanh":
                        self.act_fn = Tanh()
                    elif k == "act_fn" and v == "Linear":
                        self.act_fn = Linear()
                    if k != "wrappers":
                        self.hyperparameters[k] = v
                if k in self.parameters:
                    self.parameters[k] = v
        return layer

    def summary(self):
        return {
            "layer": self.hyperparameters["layer"],
            "parameters": self.parameters,
            "hyperparameters": self.hyperparameters,
        }


class Add(LayerBase):
    def __init__(self, act_fn=None):
        if act_fn is None:
            act_fn = Linear()

        super().__init__(None, None, act_fn)
        self._init_params()

    def _init_params(self):
        self.parameters = {}
        self.gradients = {}
        self.derived_variables = {"sum": None}
        self.hyperparameters = {"layer": "Sum", "act_fn": str(self.act_fn)}

    def forward(self, X):
        """
        Compute the layer output on a single minibatch.

        Parameters
        ----------
        X : list of length `n_inputs`
            A list of tensors, all of the same shape.

        Returns
        -------
        Y : numpy array of shape (n_ex, n_in)
            Layer output for each of the `n_ex` examples
        """
        self.X = X
        out = X[0].copy()
        for i in range(1, len(self.X)):
            out += X[i]
        self.derived_variables["sum"] = out
        return self.act_fn.fn(out)

    def backward(self, dLdY):
        _sum = self.derived_variables["sum"]
        grads = [dLdY * self.act_fn.grad(_sum) for z in self.X]
        import ipdb

        ipdb.set_trace()

        return grads


class BatchNorm2D(LayerBase):
    def __init__(self, n_in, momentum=0.9, epsilon=1e-5):
        """
        A batch normalization layer for two-dimensional inputs with an
        additional channel dimension. This is sometimes known as "Spatial Batch
        Normalization" in the literature.

        Equations:
            Y = scaler * norm(X) + intercept
            norm(X) = (X - mean(X)) / (std(X) + epsilon)

        Parameters
        ----------
        n_in : int
            The number of channels in the input volume. The layer output will
            automatically have the same dimensionality as the input.
        momentum : float (default: 0.9)
            The momentum term for the running mean/running std calculations.
            The closer this is to 1, the less weight will be given to the
            mean/std of the current batch (i.e., higher smoothing)
        epsilon : float (default : 1e-5)
            A small smoothing constant to use during computation of norm(X) to
            avoid divide-by-zero errors.
        """
        super().__init__(n_in, n_in, None)
        self.momentum = momentum
        self.epsilon = epsilon
        self._init_params()

    def _init_params(self):
        scaler = np.random.rand(self.n_in)
        intercept = np.zeros(self.n_in)

        # init running mean and std at 0 and 1, respectively
        running_mean = np.zeros(self.n_in)
        running_var = np.ones(self.n_in)

        self.derived_variables = {}
        self.parameters = {
            "scaler": scaler,
            "intercept": intercept,
            "running_mean": running_mean,
            "running_var": running_var,
        }

        self.gradients = {
            "scaler": np.zeros_like(scaler),
            "intercept": np.zeros_like(intercept),
        }

        self.hyperparameters = {
            "layer": "BatchNorm2D",
            "act_fn": None,
            "n_in": self.n_in,
            "n_out": self.n_out,
            "epsilon": self.epsilon,
            "momentum": self.momentum,
        }

    def reset_running_stats(self):
        assert self.trainable, "Layer is frozen"
        self.parameters["running_mean"] = np.zeros(self.n_in)
        self.parameters["running_var"] = np.ones(self.n_in)

    def forward(self, X):
        """
        Compute the layer output on a single minibatch.

        Equations:
            Y = scaler * norm(X) + intercept
            norm(X) = (X - mean(X)) / std(X + epsilon)

        Parameters
        ----------
        X : numpy array of shape (n_ex, in_rows, in_cols, n_in)
            Input volume containing the `in_rows` x `in_cols`-dimensional features for a
            minibatch of `n_ex` examples.

        Returns
        -------
        Y : numpy array of shape (n_ex, in_rows, in_cols, n_in)
            Layer output for each of the `n_ex` examples
        """
        self.X = X
        ep = self.hyperparameters["epsilon"]
        mm = self.hyperparameters["momentum"]
        rm = self.parameters["running_mean"]
        rv = self.parameters["running_var"]

        scaler = self.parameters["scaler"]
        intercept = self.parameters["intercept"]

        # if the layer is frozen, use our running mean/std values rather
        # than the mean/std values for the new batch
        X_mean = self.parameters["running_mean"]
        X_var = self.parameters["running_var"]

        if self.trainable:
            X_mean, X_var = X.mean(axis=(0, 1, 2)), X.var(axis=(0, 1, 2))  # , ddof=1)
            self.parameters["running_mean"] = mm * rm + (1.0 - mm) * X_mean
            self.parameters["running_var"] = mm * rv + (1.0 - mm) * X_var

        N = (X - X_mean) / np.sqrt(X_var + ep)
        y = scaler * N + intercept
        return y

    def backward(self, dLdy):
        """
        Backprop from layer outputs to inputs

        Parameters
        ----------
        dLdY : numpy array of shape (n_ex, in_rows, in_cols, n_in)
            The gradient of the loss wrt. the layer output Y

        Returns
        -------
        dX : numpy array of shape (n_ex, in_rows, in_cols, n_in)
            The gradient of the loss wrt. the layer input X
        """
        assert self.trainable, "Layer is frozen"

        scaler = self.parameters["scaler"]
        ep = self.hyperparameters["epsilon"]

        # reshape to 2D, retaining channel dim
        X = np.reshape(self.X, (-1, self.X.shape[3]))
        dLdy = np.reshape(dLdy, (-1, dLdy.shape[3]))

        # appy 1D batchnorm to reshaped array
        n_ex, n_in = X.shape
        X_mean, X_var = X.mean(axis=0), X.var(axis=0)  # , ddof=1)

        N = (X - X_mean) / np.sqrt(X_var + ep)
        dIntercept = dLdy.sum(axis=0)
        dScaler = np.sum(dLdy * N, axis=0)

        dN = dLdy * scaler
        dX = (n_ex * dN - dN.sum(axis=0) - N * (dN * N).sum(axis=0)) / (
            n_ex * np.sqrt(X_var + ep)
        )

        # reshape gradients back to proper dimensions
        dX = np.reshape(dX, self.X.shape)
        self.gradients = {"scaler": dScaler, "intercept": dIntercept}
        return dX


class BatchNorm1D(LayerBase):
    def __init__(self, n_in, momentum=0.9, epsilon=1e-5):
        """
        A batch normalization layer for vector inputs.

        Equations:
            Y = scaler * norm(X) + intercept
            norm(X) = (X - mean(X)) / sqrt(var(X) + epsilon)

        Parameters
        ----------
        n_in : int
            The dimensionality of the layer input. The layer output will
            automatically have the same dimensionality.
        momentum : float (default: 0.9)
            The momentum term for the running mean/running std calculations.
            The closer this is to 1, the less weight will be given to the
            mean/std of the current batch (i.e., higher smoothing)
        epsilon : float (default : 1e-5)
            A small smoothing constant to use during computation of norm(X) to
            avoid divide-by-zero errors.
        """
        super().__init__(n_in, n_in, None)
        self.momentum = momentum
        self.epsilon = epsilon
        self._init_params()

    def _init_params(self):
        scaler = np.random.rand(self.n_in)
        intercept = np.zeros(self.n_in)

        # init running mean and std at 0 and 1, respectively
        running_mean = np.zeros(self.n_in)
        running_var = np.ones(self.n_in)

        self.derived_variables = {}
        self.parameters = {
            "scaler": scaler,
            "intercept": intercept,
            "running_mean": running_mean,
            "running_var": running_var,
        }

        self.gradients = {
            "scaler": np.zeros_like(scaler),
            "intercept": np.zeros_like(intercept),
        }

        self.hyperparameters = {
            "layer": "BatchNorm1D",
            "act_fn": None,
            "n_in": self.n_in,
            "n_out": self.n_out,
            "epsilon": self.epsilon,
            "momentum": self.momentum,
        }

    def reset_running_stats(self):
        assert self.trainable, "Layer is frozen"
        self.parameters["running_mean"] = np.zeros(self.n_in)
        self.parameters["running_var"] = np.ones(self.n_in)

    def forward(self, X):
        """
        Compute the layer output on a single minibatch.

        Equations:
            Y = scaler * norm(X) + intercept
            norm(X) = (X - mean(X)) / std(X + epsilon)

        Parameters
        ----------
        X : numpy array of shape (n_ex, n_in)
            Layer input, representing the `n_in`-dimensional features for a
            minibatch of `n_ex` examples

        Returns
        -------
        Y : numpy array of shape (n_ex, n_in)
            Layer output for each of the `n_ex` examples
        """
        self.X = X
        ep = self.hyperparameters["epsilon"]
        mm = self.hyperparameters["momentum"]
        rm = self.parameters["running_mean"]
        rv = self.parameters["running_var"]

        scaler = self.parameters["scaler"]
        intercept = self.parameters["intercept"]

        # if the layer is frozen, use our running mean/std values rather
        # than the mean/std values for the new batch
        X_mean = self.parameters["running_mean"]
        X_var = self.parameters["running_var"]

        if self.trainable:
            X_mean, X_var = X.mean(axis=0), X.var(axis=0)  # , ddof=1)
            self.parameters["running_mean"] = mm * rm + (1.0 - mm) * X_mean
            self.parameters["running_var"] = mm * rv + (1.0 - mm) * X_var

        N = (X - X_mean) / np.sqrt(X_var + ep)
        y = scaler * N + intercept
        return y

    def backward(self, dLdy):
        """
        Backprop from layer outputs to inputs

        Parameters
        ----------
        dLdY : numpy array of shape (n_ex, n_in)
            The gradient of the loss wrt. the layer output Y

        Returns
        -------
        dX : numpy array of shape (n_ex, n_in)
            The gradient of the loss wrt. the layer input X
        """
        assert self.trainable, "Layer is frozen"

        scaler = self.parameters["scaler"]
        ep = self.hyperparameters["epsilon"]

        X = self.X
        n_ex, n_in = X.shape
        X_mean, X_var = X.mean(axis=0), X.var(axis=0)  # , ddof=1)

        N = (X - X_mean) / np.sqrt(X_var + ep)
        dIntercept = dLdy.sum(axis=0)
        dScaler = np.sum(dLdy * N, axis=0)

        dN = dLdy * scaler
        dX = (n_ex * dN - dN.sum(axis=0) - N * (dN * N).sum(axis=0)) / (
            n_ex * np.sqrt(X_var + ep)
        )

        self.gradients = {"scaler": dScaler, "intercept": dIntercept}
        return dX


class FullyConnected(LayerBase):
    def __init__(self, n_in, n_out, act_fn):
        """
        A fully-connected (dense) layer.

        Equations:
            Y = act_fn( W . X + b )

        Parameters
        ----------
        n_in : int
            The dimensionality of the layer input
        n_out : int
            The dimensionality of the layer output
        act_fn : `activations.Activation` instance
            The element-wise output nonlinearity used in computing Y
        """
        super().__init__(n_in, n_out, act_fn)
        self._init_params()

    def _init_params(self):
        # TODO: use a flexible / sensible initialization strategy here
        b = np.zeros((1, self.n_out))
        W = np.random.randn(self.n_in, self.n_out)

        self.parameters = {"W": W, "b": b}
        self.derived_variables = {"Z": None, "Y": None}
        self.gradients = {"W": np.zeros_like(W), "b": np.zeros_like(b)}

        self.hyperparameters = {
            "layer": "FullyConnected",
            "n_in": self.n_in,
            "n_out": self.n_out,
            "act_fn": str(self.act_fn),
        }

    def forward(self, X):
        """
        Compute the layer output on a single minibatch.

        Equations:
            Y = act_fn( W . X + b )

        Parameters
        ----------
        X : numpy array of shape (n_ex, n_in)
            Layer input, representing the `n_in`-dimensional features for a
            minibatch of `n_ex` examples

        Returns
        -------
        Y : numpy array of shape (n_ex, n_out)
            Layer output for each of the `n_ex` examples
        """
        # save input for gradient calc during backward pass
        self.X = X

        # Retrieve parameters
        W = self.parameters["W"]
        b = self.parameters["b"]

        # compute next activation state
        Z = np.dot(X, W) + b
        Y = self.act_fn.fn(Z)
        self.derived_variables = {"Z": Z, "Y": Y}
        return Y

    def backward(self, dLdY):
        """
        Backprop from layer outputs to inputs

        Parameters
        ----------
        dLdY : numpy array of shape (n_ex, n_out)
            The gradient of the loss wrt. the layer output Y

        Returns
        -------
        dLdX : numpy array of shape (n_ex, n_in)
            The gradient of the loss wrt. the layer input X
        """
        assert self.trainable, "Layer is frozen"
        assert self.X is not None

        X = self.X
        W = self.parameters["W"]
        Z = self.derived_variables["Z"]

        # compute gradients
        dZ = dLdY * self.act_fn.grad(Z)
        dW = np.dot(X.T, dZ)
        dB = dZ.sum(axis=0, keepdims=True)
        dX = np.dot(dZ, W.T)

        self.gradients = {"W": dW, "b": dB, "Z": dZ, "Y": dLdY}
        return dX


class RNNCell(LayerBase):
    def __init__(self, n_in, n_out, act_fn=None):
        """
        A single step of a vanilla (Elman) RNN.

        Equations:
            Z[t] = Wax . X[t] + bax + Waa . A[t-1] + baa
            A[t] = act_fn(Z[t])

        We refer to A[t] as the hidden state at timestep t

        Parameters
        ----------
        n_in : int
            The dimension of a single input example on a given timestep
        n_out : int
            The dimension of a single hidden state / output on a given timestep
        act_fn : `activations.Activation` instance (default: None)
            The activation function for computing A[t]. If not specified, use
            Tanh by default.
        """
        # use tanh as activation function by default
        if act_fn is None:
            act_fn = Tanh()

        super().__init__(n_in, n_out, act_fn)

        self.n_timesteps = None
        self._init_params()

    def _init_params(self):
        self.X = []

        # TODO: use a flexible / sensible initialization strategy here
        Wax = np.random.randn(self.n_in, self.n_out)
        Waa = np.random.randn(self.n_out, self.n_out)
        ba = np.zeros((self.n_out, 1))
        bx = np.zeros((self.n_out, 1))

        self.parameters = {"Waa": Waa, "Wax": Wax, "ba": ba, "bx": bx}

        self.hyperparameters = {
            "layer": "RNNCell",
            "n_in": self.n_in,
            "n_out": self.n_out,
            "act_fn": str(self.act_fn),
        }

        self.gradients = {
            "Waa": np.zeros_like(Waa),
            "Wax": np.zeros_like(Wax),
            "ba": np.zeros_like(ba),
            "bx": np.zeros_like(bx),
        }

        self.derived_variables = {
            "A": [],
            "Z": [],
            "n_timesteps": 0,
            "current_step": 0,
            "dLdA_accumulator": None,
        }

    def forward(self, Xt):
        """
        Compute the network output for a single timestep.

        Equations:
            Z[t] = Wax . X[t] + bax + Waa . A[t-1] + baa
            A[t] = tanh(Z[t])

        We refer to A[t] as the hidden state at timestep t.

        Parameters
        ----------
        Xt : numpy array of shape (n_ex, n_in)
            Input at timestep t consisting of `n_ex` examples each of
            dimensionality `n_in`

        Returns
        -------
        At: numpy array of shape (n_ex, n_out)
            The value of the hidden state at timestep t for each of the `n_ex`
            examples
        """
        # increment timestep
        self.derived_variables["n_timesteps"] += 1
        self.derived_variables["current_step"] += 1

        # Retrieve parameters
        ba = self.parameters["ba"]
        bx = self.parameters["bx"]
        Wax = self.parameters["Wax"]
        Waa = self.parameters["Waa"]

        # initialize the hidden state to zero
        As = self.derived_variables["A"]
        if len(As) == 0:
            n_ex, n_in = Xt.shape
            A0 = np.zeros((n_ex, self.n_out))
            As.append(A0)

        # compute next hidden state
        Zt = np.dot(As[-1], Waa) + ba.T + np.dot(Xt, Wax) + bx.T
        At = self.act_fn.fn(Zt)

        # store intermediate variables
        self.X.append(Xt)
        self.derived_variables["Z"].append(Zt)
        self.derived_variables["A"].append(At)
        return At

    def backward(self, dLdAt):
        """
        Backprop for a single timestep.

        Equations:
            Z[t] = Wax . X[t] + bax + Waa . A[t-1] + baa
            A[t] = tanh(Z[t])

        We refer to A[t] as the hidden state at timestep t.

        Parameters
        ----------
        dLdAt : numpy array of shape (n_ex, n_out)
            The gradient of the loss wrt. the layer outputs (ie., hidden
            states) at timestep t

        Returns
        -------
        dLdXt : numpy array of shape (n_ex, n_in)
            The gradient of the loss wrt. the layer inputs at timestep t
        """
        assert self.trainable, "Layer is frozen"

        #  decrement current step
        self.derived_variables["current_step"] -= 1

        # extract context variables
        Zs = self.derived_variables["Z"]
        As = self.derived_variables["A"]
        t = self.derived_variables["current_step"]
        dA_acc = self.derived_variables["dLdA_accumulator"]

        # initialize accumulator
        if dA_acc is None:
            dA_acc = np.zeros_like(As[0])

        # get network weights for gradient calcs
        Wax = self.parameters["Wax"]
        Waa = self.parameters["Waa"]

        # compute gradient components at timestep t
        dA = dLdAt + dA_acc
        dZ = self.act_fn.grad(Zs[t]) * dA
        dXt = np.dot(dZ, Wax.T)

        # update parameter gradients with signal from current step
        self.gradients["Waa"] += np.dot(As[t].T, dZ)
        self.gradients["Wax"] += np.dot(self.X[t].T, dZ)
        self.gradients["ba"] += dZ.sum(axis=0, keepdims=True).T
        self.gradients["bx"] += dZ.sum(axis=0, keepdims=True).T

        # update accumulator variable for hidden state
        self.derived_variables["dLdA_accumulator"] = np.dot(dZ, Waa.T)
        return dXt

    def flush_gradients(self):
        assert self.trainable, "Layer is frozen"

        self.X = []
        for k, v in self.derived_variables.items():
            self.derived_variables[k] = []

        self.derived_variables["n_timesteps"] = 0
        self.derived_variables["current_step"] = 0

        # reset parameter gradients to 0
        for k, v in self.parameters.items():
            self.gradients[k] = np.zeros_like(v)


class LSTMCell(LayerBase):
    def __init__(self, n_in, n_out, act_fn=None, gate_fn=None):
        """
        A single step of a long short-term memory (LSTM) RNN.

        Notation:
            Z[t]  is the input to each of the gates at timestep t
            A[t]  is the value of the hidden state at timestep t
            Cc[t] is the value of the *candidate* cell/memory state at timestep t
            C[t]  is the value of the *final* cell/memory state at timestep t
            Gf[t] is the output of the forget gate at timestep t
            Gu[t] is the output of the update gate at timestep t
            Go[t] is the output of the output gate at timestep t

        Equations:
            Z[t]  = stack([A[t-1], X[t]])
            Gf[t] = gate_fn(Wf . Z[t] + bf)
            Gu[t] = gate_fn(Wu . Z[t] + bu)
            Go[t] = gate_fn(Wo . Z[t] + bo)
            Cc[t] = act_fn(Wc . Z[t] + bc)
            C[t]  = Gf[t] * C[t-1] + Gu[t] * Cc[t]
            A[t]  = Go[t] * act_fn(C[t])

            where '.' indicates dot/matrix product, and '*' indicates
            elementwise multiplication

        We refer to A[t] as the hidden state at timestep t and C[t] as the
        memory / cell state

        Parameters
        ----------
        n_in : int
            The dimension of a single input example on a given timestep
        n_out : int
            The dimension of a single hidden state / output on a given timestep
        act_fn : `activations.Activation` instance (default: None)
            The activation function for computing A[t]. If not specified, use
            Tanh by default.
        gate_fn : `activations.Activation` instance (default: None)
            The gate function for computing the update, forget, and output
            gates. If not specified, use Sigmoid by default.
        """
        # use tanh as activation function by default
        if act_fn is None:
            act_fn = Tanh()

        # use sigmoid as gating function by default
        if gate_fn is None:
            gate_fn = Sigmoid()

        super().__init__(n_in, n_out, act_fn)

        self.n_timesteps = None
        self.gate_fn = gate_fn
        self._init_params()

    def _init_params(self):
        self.X = []

        # TODO: use a flexible / sensible initialization strategy here
        Wf = np.random.randn(self.n_in + self.n_out, self.n_out)
        Wu = np.random.randn(self.n_in + self.n_out, self.n_out)
        Wc = np.random.randn(self.n_in + self.n_out, self.n_out)
        Wo = np.random.randn(self.n_in + self.n_out, self.n_out)

        bf = np.zeros((1, self.n_out))
        bu = np.zeros((1, self.n_out))
        bc = np.zeros((1, self.n_out))
        bo = np.zeros((1, self.n_out))

        self.parameters = {
            "Wf": Wf,
            "Wu": Wu,
            "Wc": Wc,
            "Wo": Wo,
            "bf": bf,
            "bu": bu,
            "bc": bc,
            "bo": bo,
        }

        self.hyperparameters = {
            "layer": "LSTMCell",
            "n_in": self.n_in,
            "n_out": self.n_out,
            "act_fn": str(self.act_fn),
            "gate_fn": str(self.gate_fn),
        }

        self.gradients = {
            "Wf": np.zeros_like(Wf),
            "Wu": np.zeros_like(Wu),
            "Wc": np.zeros_like(Wc),
            "Wo": np.zeros_like(Wo),
            "bf": np.zeros_like(bf),
            "bu": np.zeros_like(bu),
            "bc": np.zeros_like(bc),
            "bo": np.zeros_like(bo),
        }

        self.derived_variables = {
            "C": [],
            "A": [],
            "Gf": [],
            "Gu": [],
            "Go": [],
            "Gc": [],
            "Cc": [],
            "n_timesteps": 0,
            "current_step": 0,
            "dLdA_accumulator": None,
            "dLdC_accumulator": None,
        }

    def _get_params(self):
        Wf = self.parameters["Wf"]
        Wu = self.parameters["Wu"]
        Wc = self.parameters["Wc"]
        Wo = self.parameters["Wo"]
        bf = self.parameters["bf"]
        bu = self.parameters["bu"]
        bc = self.parameters["bc"]
        bo = self.parameters["bo"]
        return Wf, Wu, Wc, Wo, bf, bu, bc, bo

    def forward(self, Xt):
        """
        Compute the layer output for a single timestep.

        Notation:
            Z[t]  is the input to each of the gates at timestep t
            A[t]  is the value of the hidden state at timestep t
            Cc[t] is the value of the *candidate* cell/memory state at timestep t
            C[t]  is the value of the *final* cell/memory state at timestep t
            Gf[t] is the output of the forget gate at timestep t
            Gu[t] is the output of the update gate at timestep t
            Go[t] is the output of the output gate at timestep t

        Equations:
            Z[t]  = stack([A[t-1], X[t]])
            Gf[t] = gate_fn(Wf . Z[t] + bf)
            Gu[t] = gate_fn(Wu . Z[t] + bu)
            Go[t] = gate_fn(Wo . Z[t] + bo)
            Cc[t] = act_fn(Wc . Z[t] + bc)
            C[t]  = Gf[t] * C[t-1] + Gu[t] * Cc[t]
            A[t]  = Go[t] * act_fn(C[t])

            where '.' indicates dot/matrix product, and '*' indicates
            elementwise multiplication

        Parameters
        ----------
        Xt : numpy array of shape (n_ex, n_in)
            Input at timestep t consisting of `n_ex` examples each of
            dimensionality `n_in`

        Returns
        -------
        At: numpy array of shape (n_ex, n_out)
            The value of the hidden state at timestep t for each of the `n_ex`
            examples
        Ct: numpy array of shape (n_ex, n_out)
            The value of the cell/memory state at timestep t for each of the
            `n_ex` examples
        """
        Wf, Wu, Wc, Wo, bf, bu, bc, bo = self._get_params()

        self.derived_variables["n_timesteps"] += 1
        self.derived_variables["current_step"] += 1

        if len(self.derived_variables["A"]) == 0:
            n_ex, n_in = Xt.shape
            init = np.zeros((n_ex, self.n_out))
            self.derived_variables["A"].append(init)
            self.derived_variables["C"].append(init)

        A_prev = self.derived_variables["A"][-1]
        C_prev = self.derived_variables["C"][-1]

        # concatenate A_prev and Xt to create Zt
        Zt = np.hstack([A_prev, Xt])

        Gft = self.gate_fn.fn(np.dot(Zt, Wf) + bf)
        Gut = self.gate_fn.fn(np.dot(Zt, Wu) + bu)
        Got = self.gate_fn.fn(np.dot(Zt, Wo) + bo)
        Cct = self.act_fn.fn(np.dot(Zt, Wc) + bc)
        Ct = Gft * C_prev + Gut * Cct
        At = Got * self.act_fn.fn(Ct)

        # bookkeeping
        self.X.append(Xt)
        self.derived_variables["A"].append(At)
        self.derived_variables["C"].append(Ct)
        self.derived_variables["Gf"].append(Gft)
        self.derived_variables["Gu"].append(Gut)
        self.derived_variables["Go"].append(Got)
        self.derived_variables["Cc"].append(Cct)
        return At, Ct

    def backward(self, dLdAt):
        """
        Backprop for a single timestep.

        Parameters
        ----------
        dLdAt : numpy array of shape (n_ex, n_out)
            The gradient of the loss wrt. the layer outputs (ie., hidden
            states) at timestep t

        Returns
        -------
        dLdXt : numpy array of shape (n_ex, n_in)
            The gradient of the loss wrt. the layer inputs at timestep t
        """
        assert self.trainable, "Layer is frozen"

        Wf, Wu, Wc, Wo, bf, bu, bc, bo = self._get_params()

        self.derived_variables["current_step"] -= 1
        t = self.derived_variables["current_step"]

        Got = self.derived_variables["Go"][t]
        Gft = self.derived_variables["Gf"][t]
        Gut = self.derived_variables["Gu"][t]
        Cct = self.derived_variables["Cc"][t]
        At = self.derived_variables["A"][t + 1]
        Ct = self.derived_variables["C"][t + 1]
        C_prev = self.derived_variables["C"][t]
        A_prev = self.derived_variables["A"][t]

        Xt = self.X[t]
        Zt = np.hstack([A_prev, Xt])

        dA_acc = self.derived_variables["dLdA_accumulator"]
        dC_acc = self.derived_variables["dLdC_accumulator"]

        # initialize accumulators
        if dA_acc is None:
            dA_acc = np.zeros_like(At)

        if dC_acc is None:
            dC_acc = np.zeros_like(Ct)

        # Gradient calculations
        # ---------------------

        dA = dLdAt + dA_acc
        dC = dC_acc + dA * Got * self.act_fn.grad(Ct)

        # compute the input to the gate functions at timestep t
        _Go = np.dot(Zt, Wo) + bo
        _Gf = np.dot(Zt, Wf) + bo
        _Gu = np.dot(Zt, Wu) + bo
        _Gc = np.dot(Zt, Wc) + bc

        # compute gradients wrt the *input* to each gate
        dGot = dA * self.act_fn.fn(Ct) * self.gate_fn.grad(_Go)
        dCct = dC * Gut * self.act_fn.grad(_Gc)
        dGut = dC * Cct * self.gate_fn.grad(_Gu)
        dGft = dC * C_prev * self.gate_fn.grad(_Gf)

        dZ = (
            np.dot(dGft, Wf.T)
            + np.dot(dGut, Wu.T)
            + np.dot(dCct, Wc.T)
            + np.dot(dGot, Wo.T)
        )

        dXt = dZ[:, self.n_out :]

        self.gradients["Wc"] += np.dot(Zt.T, dCct)
        self.gradients["Wu"] += np.dot(Zt.T, dGut)
        self.gradients["Wf"] += np.dot(Zt.T, dGft)
        self.gradients["Wo"] += np.dot(Zt.T, dGot)
        self.gradients["bo"] += dGot.sum(axis=0, keepdims=True)
        self.gradients["bu"] += dGut.sum(axis=0, keepdims=True)
        self.gradients["bf"] += dGft.sum(axis=0, keepdims=True)
        self.gradients["bc"] += dCct.sum(axis=0, keepdims=True)

        self.derived_variables["dLdA_accumulator"] = dZ[:, : self.n_out]
        self.derived_variables["dLdC_accumulator"] = Gft * dC
        return dXt

    def flush_gradients(self):
        assert self.trainable, "Layer is frozen"

        self.X = []
        for k, v in self.derived_variables.items():
            self.derived_variables[k] = []

        self.derived_variables["n_timesteps"] = 0
        self.derived_variables["current_step"] = 0

        # reset parameter gradients to 0
        for k, v in self.parameters.items():
            self.gradients[k] = np.zeros_like(v)


class Pool2D(LayerBase):
    def __init__(self, in_channels, kernel_shape, stride=1, pad=0, mode="max"):
        """
        A single two-dimensional pooling layer.

        Parameters
        ----------
        in_channels : int
            The number of channels (depth) in the input volume
        kernel_shape : 2-tuple
            The dimension of a single 2D filter/kernel in the current layer
        stride : int (default: 1)
            The stride/hop of the convolution kernels as they move over the
            input volume
        pad : int, tuple, or 'same' (default: 0)
            The number of rows/columns of 0's to pad the input.
        mode : str (default: 'max')
            The pooling function to apply. Valid entries are {"max",
            "average"}.
        """
        self.pad = pad
        self.mode = mode
        self.stride = stride
        self.kernel_shape = kernel_shape

        super().__init__(in_channels, in_channels, None)
        self._init_params()

    def _init_params(self):
        self.X = None
        self.gradients = {}
        self.parameters = {}
        self.hyperparameters = {
            "layer": "Pool",
            "act_fn": None,
            "pad": self.pad,
            "mode": self.mode,
            "stride": self.stride,
            "kernel_shape": self.kernel_shape,
        }

        self.derived_variables = {"out_rows": None, "out_cols": None, "masks": []}

    def forward(self, X):
        self.X = X
        n_ex, in_rows, in_cols, nc_in = X.shape
        (fr, fc), s, p = self.kernel_shape, self.stride, self.pad
        X_pad, (pr1, pr2, pc1, pc2) = pad2D(X, p, self.kernel_shape, s)

        out_rows = np.floor(1 + (in_rows + pr1 + pr2 - fr) / s).astype(int)
        out_cols = np.floor(1 + (in_cols + pc1 + pc2 - fc) / s).astype(int)

        self.derived_variables["out_rows"] = out_rows
        self.derived_variables["out_cols"] = out_cols

        if self.mode == "max":
            pool_fn = np.max
        elif self.mode == "average":
            pool_fn = np.mean

        Y = np.zeros((n_ex, out_rows, out_cols, self.n_out))
        for m in range(n_ex):
            for i in range(out_rows):
                for j in range(out_cols):
                    for c in range(self.n_out):
                        # calculate window boundaries, incorporating stride
                        i0, i1 = i * s, (i * s) + fr
                        j0, j1 = j * s, (j * s) + fc

                        xi = X_pad[m, i0:i1, j0:j1, c]
                        Y[m, i, j, c] = pool_fn(xi)
        return Y

    def backward(self, dLdY):
        assert self.trainable, "Layer is frozen"

        X = self.X
        n_ex, in_rows, in_cols, nc_in = X.shape
        (fr, fc), s, p = self.kernel_shape, self.stride, self.pad
        X_pad, (pr1, pr2, pc1, pc2) = pad2D(X, p, self.kernel_shape, s)

        out_rows = self.derived_variables["out_rows"]
        out_cols = self.derived_variables["out_cols"]

        dX = np.zeros_like(X_pad)
        for m in range(n_ex):
            for i in range(out_rows):
                for j in range(out_cols):
                    for c in range(self.n_out):
                        # calculate window boundaries, incorporating stride
                        i0, i1 = i * s, (i * s) + fr
                        j0, j1 = j * s, (j * s) + fc

                        dy = dLdY[m, i, j, c]
                        if self.mode == "max":
                            xi = self.X[m, i0:i1, j0:j1, c]

                            # enforce that the mask can only consist of a
                            # single `True` entry, even if multiple entries in
                            # xi are equal to max(xi)
                            mask = np.zeros_like(xi).astype(bool)
                            x, y = np.argwhere(xi == np.max(xi))[0]
                            mask[x, y] = True

                            dX[m, i0:i1, j0:j1, c] += mask * dy
                        elif self.mode == "average":
                            dX[m, i0:i1, j0:j1, c] += (
                                np.ones((fr, fc)) * dy / np.prod((fr, fc))
                            )

        pr2 = None if pr2 == 0 else -pr2
        pc2 = None if pc2 == 0 else -pc2
        dX = dX[:, pr1:pr2, pc1:pc2, :]
        return dX


class Flatten(LayerBase):
    def __init__(self, keep_dim="first"):
        """
        Flatten a multidimensional input into a 2D matrix.

        Parameters
        ----------
        keep_dim : str, int (default : 'first')
            The dimension of the original input to retain. Typically used the
            minibatch dimension. Valid entries are {'first', 'last', -1} If -1,
            flatten all dimensions.
        """
        self.keep_dim = keep_dim
        super().__init__(None, None, None)
        self._init_params()

    def _init_params(self):
        self.X = None
        self.hyperparameters = {"layer": "Flatten", "keep_dim": self.keep_dim}

        self.gradients = {}
        self.parameters = {}
        self.derived_variables = {}

    def forward(self, X):
        self.in_dims = X.shape
        if self.keep_dim == -1:
            flat = X.flatten().reshape(1, -1)
        else:
            rs = (X.shape[0], -1) if self.keep_dim == "first" else (-1, X.shape[-1])
            flat = X.reshape(*rs)
        return flat

    def backward(self, dLdy):
        return dLdy.reshape(*self.in_dims)


class Conv2D(LayerBase):
    def __init__(
        self, in_channels, out_channels, kernel_shape, act_fn=None, pad=0, stride=1
    ):
        """
        Apply a two-dimensional convolution kernel over an input volume.

        Equations:
            out = act_fn(pad(X) * W + b)
            n_rows_out = floor(1 + (n_rows_in + pad_left + pad_right - filter_rows) / stride)
            n_cols_out = floor(1 + (n_cols_in + pad_top + pad_bottom - filter_cols) / stride)

            where '*' denotes the cross-correlation operation with stride `s`

        Parameters
        ----------
        in_channels : int
            The number of channels (depth) in the input volume
        out_channels : int
            The number of filters/kernels to compute in the current layer
        kernel_shape : 2-tuple
            The dimension of a single 2D filter/kernel in the current layer
        act_fn : `activations.Activation` instance (default: None)
            The activation function for computing Y[t]. If `None`, use Linear
            activations by default
        pad : int, tuple, or 'same' (default: 0)
            The number of rows/columns to zero-pad the input with
        stride : int (default: 1)
            The stride/hop of the convolution kernels as they move over the
            input volume
        """
        if act_fn is None:
            act_fn = Linear()

        self.pad = pad
        self.act_fn = act_fn
        self.stride = stride
        self.kernel_shape = kernel_shape

        super().__init__(in_channels, out_channels, act_fn)
        self._init_params()

    def _init_params(self):
        self.X = []

        # TODO: use a flexible / sensible initialization strategy here
        fr, fc = self.kernel_shape
        W = np.random.randn(fr, fc, self.n_in, self.n_out)
        b = np.zeros((1, 1, 1, self.n_out))

        self.parameters = {"W": W, "b": b}

        self.hyperparameters = {
            "layer": "Conv2D",
            "pad": self.pad,
            "n_in": self.n_in,
            "n_out": self.n_out,
            "stride": self.stride,
            "act_fn": str(self.act_fn),
            "kernel_shape": self.kernel_shape,
        }

        self.gradients = {"W": np.zeros_like(W), "b": np.zeros_like(b)}
        self.derived_variables = {"Z": None, "out_rows": None, "out_cols": None}

    def forward(self, X):
        """
        Compute the layer output given input volume `X`.

        Parameters
        ----------
        X : numpy array of shape (n_ex, in_rows, in_cols, n_in)
            The input volume consisting of `n_ex` examples, each with dimension
            (in_rows x in_cols x n_in)

        Returns
        -------
        Y : numpy array of shape (n_ex, out_rows, out_cols, n_out)
            The layer output
        """
        self.X = X

        W = self.parameters["W"]
        b = self.parameters["b"]

        n_ex, in_rows, in_cols, n_in = X.shape
        f, s, p = self.kernel_shape, self.stride, self.pad

        X_pad, p = pad2D(X, p, f, s)
        (fr, fc), (pr1, pr2, pc1, pc2) = f, p

        # compute the dimensions of the convolution output
        out_rows = np.floor(1 + (in_rows + pr1 + pr2 - fr) / s).astype(int)
        out_cols = np.floor(1 + (in_cols + pc1 + pc2 - fc) / s).astype(int)

        self.derived_variables["out_rows"] = out_rows
        self.derived_variables["out_cols"] = out_cols

        # proceed with the forward convolution
        Z = np.zeros((n_ex, out_rows, out_cols, self.n_out))
        for m in range(n_ex):
            for c in range(self.n_out):
                xi, k, bias = X_pad[m, :, :, :], W[:, :, :, c], b[:, :, :, c]
                Z[m, :, :, c] += conv2D(xi, k, s, p, bias)

        self.derived_variables["Z"] = Z
        Y = self.act_fn.fn(Z)
        return Y

    def backward(self, dLdY):
        """
        Compute the gradient of the loss with respect to the layer parameters.

        Parameters
        ----------
        dLdY : numpy array of shape (n_ex, out_rows, out_cols, n_out)
            The gradient of the loss with respect to the layer output.

        Returns
        -------
        dX : numpy array of shape (n_ex, in_rows, in_cols, n_in)
            The gradient of the loss with respect to the layer input volume
        """
        W = self.parameters["W"]
        b = self.parameters["b"]
        Z = self.derived_variables["Z"]

        X = self.X
        n_ex, out_rows, out_cols, n_out = dLdY.shape
        (fr, fc), s, p = self.kernel_shape, self.stride, self.pad
        X_pad, (pr1, pr2, pc1, pc2) = pad2D(X, p, self.kernel_shape, s)

        dZ = dLdY * self.act_fn.grad(Z)

        dX = np.zeros_like(X_pad)
        dW, dB = np.zeros_like(W), np.zeros_like(b)
        for m in range(n_ex):
            for i in range(0, out_rows):
                for j in range(0, out_cols):
                    for c in range(n_out):
                        # compute window boundaries, incorporating stride
                        i0, i1 = i * s, (i * s) + fr
                        j0, j1 = j * s, (j * s) + fc

                        wc = W[:, :, :, c]
                        kernel = dZ[m, i, j, c]
                        window = X_pad[m, i0:i1, j0:j1, :]

                        dB[:, :, :, c] += kernel
                        dW[:, :, :, c] += window * kernel
                        dX[m, i0:i1, j0:j1, :] += wc * kernel

        self.gradients["W"] = dW
        self.gradients["b"] = dB

        pr2 = None if pr2 == 0 else -pr2
        pc2 = None if pc2 == 0 else -pc2
        dX = dX[:, pr1:pr2, pc1:pc2, :]
        return dX


class RNN(LayerBase):
    def __init__(self, n_in, n_out, act_fn=None):
        """
        A single vanilla (Elman)-RNN layer.

        Parameters
        ----------
        n_in : int
            The dimension of a single input example on a given timestep
        n_out : int
            The dimension of a single hidden state / output on a given timestep
        act_fn : `activations.Activation` instance (default: None)
            The activation function for computing A[t]. If not specified, use
            Tanh by default.
        """
        if act_fn is None:
            act_fn = Tanh()

        self.n_timesteps = None
        super().__init__(n_in, n_out, act_fn)
        self._init_params()

    def _init_params(self):
        self.cell = RNNCell(n_in=self.n_in, n_out=self.n_out, act_fn=self.act_fn)

        self.hyperparameters = {
            "layer": "RNN",
            "n_in": self.n_in,
            "n_out": self.n_out,
            "act_fn": str(self.act_fn),
        }

    def forward(self, X):
        Y = []
        n_ex, n_in, n_t = X.shape
        for t in range(n_t):
            yt = self.cell.forward(X[:, :, t])
            Y.append(yt)
        return np.dstack(Y)

    def backward(self, dLdA):
        assert self.cell.trainable, "Layer is frozen"
        dLdX = []
        n_ex, n_out, n_t = dLdA.shape
        for t in reversed(range(n_t)):
            dLdXt = self.cell.backward(dLdA[:, :, t])
            dLdX.insert(0, dLdXt)
        dLdX = np.dstack(dLdX)
        return dLdX

    @property
    def derived_variables(self):
        return self.cell.derived_variables

    @property
    def gradients(self):
        return self.cell.gradients

    @property
    def parameters(self):
        return self.cell.parameters

    def freeze(self):
        self.cell.freeze()

    def unfreeze(self):
        self.cell.unfreeze()

    def flush_gradients(self):
        self.cell.flush_gradients()

    def update(self, lr):
        self.cell.update(lr)
        self.flush_gradients()


class LSTM(LayerBase):
    def __init__(self, n_in, n_out, act_fn=None, gate_fn=None):
        """
        A single long short-term memory (LSTM) RNN layer.

        Parameters
        ----------
        n_in : int
            The dimension of a single input example on a given timestep
        n_out : int
            The dimension of a single hidden state / output on a given timestep
        act_fn : `activations.Activation` instance (default: None)
            The activation function for computing A[t]. If not specified, use
            Tanh by default.
        gate_fn : `activations.Activation` instance (default: None)
            The gate function for computing the update, forget, and output
            gates. If not specified, use Sigmoid by default.
        """
        # use tanh as activation function by default
        if act_fn is None:
            act_fn = Tanh()

        # use sigmoid as gate function by default
        if gate_fn is None:
            gate_fn = Sigmoid()

        self.gate_fn = gate_fn
        self.n_timesteps = None
        super().__init__(n_in, n_out, act_fn)
        self._init_params()

    def _init_params(self):
        self.cell = LSTMCell(
            n_in=self.n_in, n_out=self.n_out, act_fn=self.act_fn, gate_fn=self.gate_fn
        )

        self.hyperparameters = {
            "layer": "LSTM",
            "n_in": self.n_in,
            "n_out": self.n_out,
            "act_fn": str(self.act_fn),
        }

    def forward(self, X):
        """
        Run a forward pass across all timesteps in the input.

        Parameters
        ----------
        X : numpy array of shape (n_ex, n_in, n_t)
            Input consisting of `n_ex` examples each of dimensionality `n_in`
            and extending for `n_t` timesteps

        Returns
        -------
        Y : numpy array of shape (n_ex, n_out, n_t)
            The value of the hidden state for each of the `n_ex` examples
            across each of the `n_t` timesteps
        """
        Y = []
        n_ex, n_in, n_t = X.shape
        for t in range(n_t):
            yt, _ = self.cell.forward(X[:, :, t])
            Y.append(yt)
        return np.dstack(Y)

    def backward(self, dLdA):
        """
        Run a backward pass across all timesteps in the input.

        Parameters
        ----------
        dLdA : numpy array of shape (n_ex, n_out, n_t)
            The gradient of the loss with respect to the layer output for each
            of the `n_ex` examples across all `n_t` timesteps

        Returns
        -------
        dLdX : numpy array of shape (n_ex, n_in, n_t)
            The value of the hidden state for each of the `n_ex` examples
            across each of the `n_t` timesteps
        """
        assert self.cell.trainable, "Layer is frozen"
        dLdX = []
        n_ex, n_out, n_t = dLdA.shape
        for t in reversed(range(n_t)):
            dLdXt, _ = self.cell.backward(dLdA[:, :, t])
            dLdX.insert(0, dLdXt)
        dLdX = np.dstack(dLdX)
        return dLdX

    @property
    def derived_variables(self):
        return self.cell.derived_variables

    @property
    def gradients(self):
        return self.cell.gradients

    @property
    def parameters(self):
        return self.cell.parameters

    def freeze(self):
        self.cell.freeze()

    def unfreeze(self):
        self.cell.unfreeze()

    def flush_gradients(self):
        self.cell.flush_gradients()

    def update(self, lr):
        self.cell.update(lr)
        self.flush_gradients()
