from abc import ABC, abstractmethod

import numpy as np


class WrapperBase(ABC):
    def __init__(self, wrapped_layer):
        self._base_layer = wrapped_layer
        if hasattr(wrapped_layer, "_base_layer"):
            self._base_layer = wrapped_layer._base_layer
        super().__init__()

    @abstractmethod
    def _init_wrapper_params(self):
        raise NotImplementedError

    @abstractmethod
    def forward(self, z, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def backward(self, out, **kwargs):
        raise NotImplementedError

    @property
    def trainable(self):
        return self._base_layer.trainable

    @property
    def parameters(self):
        return self._base_layer.parameters

    @property
    def hyperparameters(self):
        hp = self._base_layer.hyperparameters
        hpw = self._wrapper_hyperparameters
        if "wrappers" in hp:
            hp["wrappers"].append(hpw)
        else:
            hp["wrappers"] = [hpw]
        return hp

    @property
    def derived_variables(self):
        dv = self._base_layer.derived_variables.copy()
        if "wrappers" in dv:
            dv["wrappers"].append(self._wrapper_derived_variables)
        else:
            dv["wrappers"] = [self._wrapper_derived_variables]
        return dv

    @property
    def gradients(self):
        return self._base_layer.gradients

    @property
    def act_fn(self):
        return self._base_layer.act_fn

    @property
    def X(self):
        return self._base_layer.X

    def _init_params(self):
        hp = self._wrapper_hyperparameters
        if "wrappers" in self._base_layer.hyperparameters:
            self._base_layer.hyperparameters["wrappers"].append(hp)
        else:
            self._base_layer.hyperparameters["wrappers"] = [hp]

    def freeze(self):
        self._base_layer.freeze()

    def unfreeze(self):
        self._base_layer.freeze()

    def flush_gradients(self):
        assert self.trainable, "Layer is frozen"
        self._base_layer.flush_gradients()

    def update(self, lr):
        assert self.trainable, "Layer is frozen"
        self._base_layer.update(lr)
        self._base_layer.flush_gradients()

    def _set_wrapper_params(self, pdict):
        for k, v in pdict.items():
            if k in self._wrapper_hyperparameters:
                self._wrapper_hyperparameters[k] = v
        return self

    def set_params(self, summary_dict):
        return self._base_layer.set_params(summary_dict)

    def summary(self):
        return {
            "layer": self.hyperparameters["layer"],
            "layer_wrappers": [i["wrapper"] for i in self.hyperparameters["wrappers"]],
            "parameters": self.parameters,
            "hyperparameters": self.hyperparameters,
        }


class Dropout(WrapperBase):
    def __init__(self, wrapped_layer, p):
        """
        A dropout regularization wrapper.

        During training, independently zeroes each element of the layer input
        with probability p and scales the activation by 1 / (1 - p) (to reflect
        the fact that on average only (1 - p) * N units are active on any
        training pass). At test time, does not adjust elements of the input at
        all (ie., simply computes the identity function).

        Parameters
        ----------
        wrapped_layer : `layers.LayerBase` instance
            The layer to apply dropout to.
        p : float in [0, 1)
            The dropout propbability during training
        """
        super().__init__(wrapped_layer)
        self.p = p
        self._init_wrapper_params()
        self._init_params()

    def _init_wrapper_params(self):
        self._wrapper_derived_variables = {"dropout_mask": None}
        self._wrapper_hyperparameters = {"wrapper": "Dropout", "p": self.p}

    def forward(self, X):
        scaler, mask = 1.0, np.ones(X.shape).astype(bool)
        if self.trainable:
            scaler = 1.0 / (1.0 - self.p)
            mask = np.random.rand(*X.shape) >= self.p
            X = mask * X
        self._wrapper_derived_variables["dropout_mask"] = mask
        return scaler * self._base_layer.forward(X)

    def backward(self, dLdy):
        assert self.trainable, "Layer is frozen"
        dLdy *= 1.0 / (1.0 - self.p)
        return self._base_layer.backward(dLdy)


def init_wrappers(layer, wrappers_list):
    for wr in wrappers_list:
        if wr["wrapper"] == "Dropout":
            layer = Dropout(layer, 1)._set_wrapper_params(wr)
        else:
            raise NotImplementedError("{}".format(wr["wrapper"]))
    return layer
