from copy import deepcopy
from abc import ABC, abstractmethod

import numpy as np

from math import erf


def gaussian_cdf(x, mean, var):
    """
    Compute the probability that a random draw from a 1D Gaussian with mean
    `mean` and variance `var` is less than or equal to x.
    """
    x_scaled = (x - mean) / np.sqrt(var)
    return (1 + erf(x_scaled / np.sqrt(2))) / 2


class SchedulerBase(ABC):
    def __init__(self):
        self.hyperparameters = {}

    def __call__(self, step=None, cur_loss=None):
        return self.learning_rate(step=step, cur_loss=cur_loss)

    def copy(self):
        return deepcopy(self)

    def set_params(self, hparam_dict):
        if hparam_dict is not None:
            for k, v in hparam_dict.items():
                if k in self.hyperparameters:
                    self.hyperparameters[k] = v

    @abstractmethod
    def learning_rate(self, step=None):
        raise NotImplementedError


class ConstantScheduler(SchedulerBase):
    def __init__(self, lr=0.01, **kwargs):
        """
        Returns a fixed learning rate, regardless of current step

        Parameters
        ----------
        initial_lr : float (default: 0.01)
            The learning rate
        """
        super().__init__()
        self.lr = lr
        self.hyperparameters = {"id": "ConstantScheduler", "lr": self.lr}

    def __str__(self):
        return "ConstantScheduler(lr={})".format(self.lr)

    def learning_rate(self, **kwargs):
        """
        Returns the learning rate, regardless of current step etc.

        Returns
        -------
        lr : float
            The learning rate
        """
        return self.lr


class ExponentialScheduler(SchedulerBase):
    def __init__(
        self, initial_lr=0.01, stage_length=500, staircase=False, decay=0.1, **kwargs
    ):
        """
        The exponential scheduler decays the learning rate by `decay` every
        `stage_length` steps, starting from `initial_lr`.

        Parameters
        ----------
        initial_lr : float (default: 0.01)
            The learning rate at the first step
        stage_length : int (default: 500)
            The length of each stage, in steps
        staircase : bool (default: False)
            If True, only adjusts the learning rate at the stage transitions,
            producing a step-like decay schedule. If False, adjusts the
            learning rate after each step, creating a smooth decay schedule.
        decay : float (default: 0.1)
            The amount to decay the learning rate at each new stage
        """
        super().__init__()
        self.decay = decay
        self.staircase = staircase
        self.initial_lr = initial_lr
        self.stage_length = stage_length
        self.hyperparameters = {
            "id": "StepScheduler",
            "decay": self.decay,
            "staircase": self.staircase,
            "initial_lr": self.initial_lr,
            "stage_length": self.stage_length,
        }

    def __str__(self):
        return "ExponentialScheduler(initial_lr={}, stage_length={}, staircase={}, decay={})".format(
            self.initial_lr, self.stage_length, self.staircase, self.decay
        )

    def learning_rate(self, step, **kwargs):
        """
        The exponential scheduler decays the learning rate by `decay` every
        `stage_length` steps, starting from `initial_lr`.

            learning_rate = initial_lr * decay^curr_stage

        where

            curr_stage = step / stage_length          if staircase=False
            curr_stage = floor(step / stage_length)   if staircase=True

        Parameters
        ----------
        step : int
            The current step number

        Returns
        -------
        lr : float
            The learning rate for the current step
        """
        cur_stage = step / self.stage_length
        if self.staircase:
            cur_stage = np.floor(cur_stage)
        return self.initial_lr * self.decay ** cur_stage


class NoamScheduler(SchedulerBase):
    def __init__(self, model_dim=512, scale_factor=1, warmup_steps=4000, **kwargs):
        """
        The Noam scheduler increases the learning rate linearly for the first
        warmup_steps steps, and decreases it thereafter proportionally to the
        inverse square root of the step number.

        lr = scale_factor * [
            model_dim^{-0.5} * min(
                step_num^{-0.5}, step_num * warmup_steps^{-1.5}
            )
        ]

        Originally used in conjunction with the Adam optimizer in Vaswani et
        al. 2017

        Parameters
        ----------
        model_dim : int (default: 512)
            The number of units in the layer output
        scale_factor : float (default: 1)
            A fixed coefficient for rescaling the final learning rate
        warmup_steps : int (default: 4000)
            The number of steps in the warmup stage of training.
        """
        super().__init__()
        self.model_dim = model_dim
        self.scale_factor = scale_factor
        self.warmup_steps = warmup_steps
        self.hyperparameters = {
            "id": "NoamScheduler",
            "model_dim": self.model_dim,
            "scale_factor": self.scale_factor,
            "warmup_steps": self.warmup_steps,
        }

    def __str__(self):
        return "NoamScheduler(model_dim={}, scale_factor={}, warmup_steps={})".format(
            self.model_dim, self.scale_factor, self.warmup_steps
        )

    def learning_rate(self, step, **kwargs):
        warmup, d_model = self.warmup_steps, self.model_dim
        new_lr = d_model ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
        return self.scale_factor * new_lr


class KingScheduler(SchedulerBase):
    def __init__(self, initial_lr=0.01, patience=1000, decay=0.1, **kwargs):
        """
        The KingScheduler exponentially decreases the learning rate by decay if
        the loss has not decreased for `patience` timesteps.

        See http://blog.dlib.net/2018/02/automatic-learning-rate-scheduling-that.html
        for details.

        Parameters
        ----------
        initial_lr : float (default: 0.01)
            The learning rate to begin at
        patience : int (default: 1000)
            Amount of time to maintain the current learning rate without a
            decrease in loss before adjustment
        decay : float (default: 0.1)
            The amount to decay the learning rate at each new stage
        """
        super().__init__()
        self.decay = decay
        self.patience = patience
        self.initial_lr = initial_lr
        self.current_lr = initial_lr
        self.max_history = np.ceil(1.1 * (patience + 1)).astype(int)

        self.loss_history = []
        self.hyperparameters = {
            "id": "KingScheduler",
            "decay": self.decay,
            "patience": self.patience,
            "initial_lr": self.initial_lr,
        }

    def __str__(self):
        return "KingScheduler(initial_lr={}, patience={}, decay={})".format(
            self.initial_lr, self.patience, self.decay
        )

    def _steps_without_decrease(self, robust=False):
        """
        Returns the maximum number of timesteps for which P(loss is decreasing)
        < 0.51.

        Parameters
        ----------
        robust : bool (default: False)
            If `robust=True`, first filter out the largest 10% of the loss
            values to remove transient spikes in the loss due to, e.g., a few
            bad minibatches.

        Returns
        -------
        steps_without_decrease: int
            The maximum number of steps back in loss_history for which P(loss
            is decreasing) < 0.51.
        """
        lh = np.array(self.loss_history)

        # drop top 10% of loss values to filter out large loss spikes
        if robust:
            thresh = np.quantile(lh, 0.9)
            lh = np.array([i for i in lh if i <= thresh])

        N = len(lh)
        steps_without_decrease = 0
        for i in reversed(range(N)):
            if self._p_decreasing(lh, i) < 0.51:
                steps_without_decrease = N - i
        return steps_without_decrease

    def _p_decreasing(self, loss_history, i):
        """
        Compute the probability that the slope of the OLS fit to the loss
        history is negative. This corresponds to calculating:

            P(Slope <= 0) where
            Slope ~ N(mean(loss), (12 * L_Var) / (N ** 3 - N))
            L_Var =  1/(N-2) * sum([loss - loss.mean()]) ^ 2)

        Parameters
        ----------
        loss_history : numpy array of shape (N,)
            The sequence of loss values for the previous N minibatches
        i : int
            Compute P(Slope < 0) beginning at index i in `history`

        Returns
        ------
        p_decreasing : float
            The probability that the slope of the OLS fit to loss_history is
            less than or equal to 0.
        """
        N = len(loss_history[i:])
        l_mean = np.mean(loss_history)
        l_var = 1 / (N - 2) * np.sum((loss_history - l_mean) ** 2)
        slope_var = (12 * l_var) / (N ** 3 - N)
        p_decreasing = gaussian_cdf(0, l_mean, slope_var)
        return p_decreasing

    def learning_rate(self, step, cur_loss):
        """
        Compute the probability that the slope of the OLS fit to the loss
        history is negative. If the probability that it is negative is less
        than 51% over the last `patience` steps, exponentially decrease the
        current learning rate by `decay`.

        Parameters
        ----------
        step : int
            The current step number. Unused.
        cur_loss : float
            The loss at the current step.

        Returns
        -------
        lr : float
            The learning rate for the current step
        """
        if cur_loss is None:
            raise ValueError("cur_loss must be a float, but got {}".format(cur_loss))

        # this happens if we initialize the scheduler from a string / dict
        if not hasattr(self, "max_history"):
            self.max_history = np.ceil(1.1 * (self.patience + 1)).astype(int)
        patience, max_history = self.patience, self.max_history

        self.loss_history.append(cur_loss)
        if len(self.loss_history) < patience:
            return self.current_lr
        self.loss_history = self.loss_history[-max_history:]

        # if the loss has not decreased for `patience` timesteps, drop the
        # learning rate
        if (
            self._steps_without_decrease() > patience
            and self._steps_without_decrease(robust=True) > patience
        ):
            self.current_lr *= self.decay

        return self.current_lr
