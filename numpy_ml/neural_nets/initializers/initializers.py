import re
from functools import partial
from ast import literal_eval as eval

import numpy as np

from ..optimizers import OptimizerBase, SGD, AdaGrad, RMSProp, Adam
from ..activations import ActivationBase, Affine, ReLU, Tanh, Sigmoid, LeakyReLU
from ..schedulers import (
    SchedulerBase,
    ConstantScheduler,
    ExponentialScheduler,
    NoamScheduler,
    KingScheduler,
)

from ..utils import (
    he_normal,
    he_uniform,
    glorot_normal,
    glorot_uniform,
    truncated_normal,
)


class ActivationInitializer(object):
    def __init__(self, param=None):
        """
        A class for initializing activation functions. Valid inputs are:
            (a) __str__ representations of `ActivationBase` instances
            (b) `ActivationBase` instances

        If `param` is `None`, return the identity function: f(X) = X
        """
        self.param = param

    def __call__(self):
        param = self.param
        if param is None:
            act = Affine(slope=1, intercept=0)
        elif isinstance(param, ActivationBase):
            act = param
        elif isinstance(param, str):
            act = self.init_from_str(param)
        else:
            raise ValueError("Unknown activation: {}".format(param))
        return act

    def init_from_str(self, act_str):
        act_str = act_str.lower()
        if act_str == "relu":
            act_fn = ReLU()
        elif act_str == "tanh":
            act_fn = Tanh()
        elif act_str == "sigmoid":
            act_fn = Sigmoid()
        elif "affine" in act_str:
            r = r"affine\(slope=(.*), intercept=(.*)\)"
            slope, intercept = re.match(r, act_str).groups()
            act_fn = Affine(float(slope), float(intercept))
        elif "leaky relu" in act_str:
            r = r"leaky relu\(alpha=(.*)\)"
            alpha = re.match(r, act_str).groups()[0]
            act_fn = LeakyReLU(float(alpha))
        else:
            raise ValueError("Unknown activation: {}".format(act_str))
        return act_fn


class SchedulerInitializer(object):
    def __init__(self, param=None, lr=None):
        """
        A class for initializing learning rate schedulers. Valid inputs are:
            (a) __str__ representations of `SchedulerBase` instances
            (b) `SchedulerBase` instances
            (c) Parameter dicts (e.g., as produced via the `summary` method in
                `LayerBase` instances)

        If `param` is `None`, return the ConstantScheduler with learning rate
        equal to `lr`.
        """
        if all([lr is None, param is None]):
            raise ValueError("lr and param cannot both be `None`")

        self.lr = lr
        self.param = param

    def __call__(self):
        param = self.param
        if param is None:
            scheduler = ConstantScheduler(self.lr)
        elif isinstance(param, SchedulerBase):
            scheduler = param
        elif isinstance(param, str):
            scheduler = self.init_from_str()
        elif isinstance(param, dict):
            scheduler = self.init_from_dict()
        return scheduler

    def init_from_str(self):
        r = r"([a-zA-Z]*)=([^,)]*)"
        sch_str = self.param.lower()
        kwargs = dict([(i, eval(j)) for (i, j) in re.findall(r, sch_str)])

        if "constant" in sch_str:
            scheduler = ConstantScheduler(**kwargs)
        elif "exponential" in sch_str:
            scheduler = ExponentialScheduler(**kwargs)
        elif "noam" in sch_str:
            scheduler = NoamScheduler(**kwargs)
        elif "king" in sch_str:
            scheduler = KingScheduler(**kwargs)
        else:
            raise NotImplementedError("{}".format(sch_str))
        return scheduler

    def init_from_dict(self):
        S = self.param
        sc = S["hyperparameters"] if "hyperparameters" in S else None

        if sc is None:
            raise ValueError("Must have `hyperparameters` key: {}".format(S))

        if sc and sc["id"] == "ConstantScheduler":
            scheduler = ConstantScheduler().set_params(sc)
        elif sc and sc["id"] == "ExponentialScheduler":
            scheduler = ExponentialScheduler().set_params(sc)
        elif sc and sc["id"] == "NoamScheduler":
            scheduler = NoamScheduler().set_params(sc)
        elif sc:
            raise NotImplementedError("{}".format(sc["id"]))
        return scheduler


class OptimizerInitializer(object):
    def __init__(self, param=None):
        """
        A class for initializing optimizers. Valid inputs are:
            (a) __str__ representations of `OptimizerBase` instances
            (b) `OptimizerBase` instances
            (c) Parameter dicts (e.g., as produced via the `summary` method in
                `LayerBase` instances)

        If `param` is `None`, return the SGD optimizer with default parameters.
        """
        self.param = param

    def __call__(self):
        param = self.param
        if param is None:
            opt = SGD()
        elif isinstance(param, OptimizerBase):
            opt = param
        elif isinstance(param, str):
            opt = self.init_from_str()
        elif isinstance(param, dict):
            opt = self.init_from_dict()
        return opt

    def init_from_str(self):
        r = r"([a-zA-Z]*)=([^,)]*)"
        opt_str = self.param.lower()
        kwargs = dict([(i, eval(j)) for (i, j) in re.findall(r, opt_str)])
        if "sgd" in opt_str:
            optimizer = SGD(**kwargs)
        elif "adagrad" in opt_str:
            optimizer = AdaGrad(**kwargs)
        elif "rmsprop" in opt_str:
            optimizer = RMSProp(**kwargs)
        elif "adam" in opt_str:
            optimizer = Adam(**kwargs)
        else:
            raise NotImplementedError("{}".format(opt_str))
        return optimizer

    def init_from_dict(self):
        O = self.param
        cc = O["cache"] if "cache" in O else None
        op = O["hyperparameters"] if "hyperparameters" in O else None

        if op is None:
            raise ValueError("Must have `hyperparemeters` key: {}".format(O))

        if op and op["id"] == "SGD":
            optimizer = SGD().set_params(op, cc)
        elif op and op["id"] == "RMSProp":
            optimizer = RMSProp().set_params(op, cc)
        elif op and op["id"] == "AdaGrad":
            optimizer = AdaGrad().set_params(op, cc)
        elif op and op["id"] == "Adam":
            optimizer = Adam().set_params(op, cc)
        elif op:
            raise NotImplementedError("{}".format(op["id"]))
        return optimizer


class WeightInitializer(object):
    def __init__(self, act_fn_str, mode="glorot_uniform"):
        """
        A factory for weight initializers.

        Parameters
        ----------
        act_fn_str : str
            The string representation for the layer activation function
        mode : str (default: 'glorot_uniform')
            The weight initialization strategy. Valid entries are {"he_normal",
            "he_uniform", "glorot_normal", glorot_uniform", "std_normal",
            "trunc_normal"}
        """
        if mode not in [
            "he_normal",
            "he_uniform",
            "glorot_normal",
            "glorot_uniform",
            "std_normal",
            "trunc_normal",
        ]:
            raise ValueError("Unrecognize initialization mode: {}".format(mode))

        self.mode = mode
        self.act_fn = act_fn_str

        if mode == "glorot_uniform":
            self._fn = glorot_uniform
        elif mode == "glorot_normal":
            self._fn = glorot_normal
        elif mode == "he_uniform":
            self._fn = he_uniform
        elif mode == "he_normal":
            self._fn = he_normal
        elif mode == "std_normal":
            self._fn = np.random.randn
        elif mode == "trunc_normal":
            self._fn = partial(truncated_normal, mean=0, std=1)

    def __call__(self, weight_shape):
        if "glorot" in self.mode:
            gain = self._calc_glorot_gain()
            W = self._fn(weight_shape, gain)
        elif self.mode == "std_normal":
            W = self._fn(*weight_shape)
        else:
            W = self._fn(weight_shape)
        return W

    def _calc_glorot_gain(self):
        """
        Values from:
        https://pytorch.org/docs/stable/nn.html?#torch.nn.init.calculate_gain
        """
        gain = 1.0
        act_str = self.act_fn.lower()
        if act_str == "tanh":
            gain = 5.0 / 3.0
        elif act_str == "relu":
            gain = np.sqrt(2)
        elif "leaky relu" in act_str:
            r = r"leaky relu\(alpha=(.*)\)"
            alpha = re.match(r, act_str).groups()[0]
            gain = np.sqrt(2 / 1 + float(alpha) ** 2)
        return gain
