"""A trainer/runner object for executing and comparing MAB policies."""

import warnings
import os.path as op
from collections import defaultdict

import numpy as np

from numpy_ml.utils.testing import DependencyWarning

try:
    import matplotlib.pyplot as plt

    _PLOTTING = True
except ImportError:
    fstr = "Cannot import matplotlib. Plotting functionality disabled."
    warnings.warn(fstr, DependencyWarning)
    _PLOTTING = False


def get_scriptdir():
    """Return the directory containing the `trainer.py` script"""
    return op.dirname(op.realpath(__file__))


def mse(bandit, policy):
    """
    Computes the mean squared error between a policy's estimates of the
    expected arm payouts and the true expected payouts.
    """
    if not hasattr(policy, "ev_estimates") or len(policy.ev_estimates) == 0:
        return np.nan

    se = []
    evs = bandit.arm_evs
    ests = sorted(policy.ev_estimates.items(), key=lambda x: x[0])
    for ix, (est, ev) in enumerate(zip(ests, evs)):
        se.append((est[1] - ev) ** 2)
    return np.mean(se)


def smooth(prev, cur, weight):
    r"""
    Compute a simple weighted average of the previous and current value.

    Notes
    -----
    The smoothed value at timestep `t`, :math:`\tilde{X}_t` is calculated as

    .. math::

        \tilde{X}_t = \epsilon \tilde{X}_{t-1} + (1 - \epsilon) X_t

    where :math:`X_t` is the value at timestep `t`, :math:`\tilde{X}_{t-1}` is
    the value of the smoothed signal at timestep `t-1`, and :math:`\epsilon` is
    the smoothing weight.

    Parameters
    ----------
    prev : float or :py:class:`ndarray <numpy.ndarray>` of shape `(N,)`
        The value of the smoothed signal at the immediately preceding
        timestep.
    cur : float or :py:class:`ndarray <numpy.ndarray>` of shape `(N,)`
        The value of the signal at the current timestep
    weight : float or :py:class:`ndarray <numpy.ndarray>` of shape `(N,)`
        The smoothing weight. Values closer to 0 result in less smoothing,
        values closer to 1 produce more aggressive smoothing. If weight is an
        array, each dimension will be interpreted as a separate smoothing
        weight the corresponding dimension in `cur`.

    Returns
    -------
    smoothed : float or :py:class:`ndarray <numpy.ndarray>` of shape `(N,)`
        The smoothed signal
    """
    return weight * prev + (1 - weight) * cur


class BanditTrainer:
    def __init__(self):
        """
        An object to facilitate multi-armed bandit training, comparison, and
        evaluation.
        """
        self.logs = {}

    def compare(
        self,
        policies,
        bandit,
        n_trials,
        n_duplicates,
        plot=True,
        seed=None,
        smooth_weight=0.999,
        out_dir=None,
    ):
        """
        Compare the performance of multiple policies on the same bandit
        environment, generating a plot for each.

        Parameters
        ----------
        policies : list of :class:`BanditPolicyBase <numpy_ml.bandits.policies.BanditPolicyBase>` instances
            The multi-armed bandit policies to compare.
        bandit : :class:`Bandit <numpy_ml.bandits.bandits.Bandit>` instance
            The environment to train the policies on.
        n_trials : int
            The number of trials per run.
        n_duplicates: int
            The number of times to evaluate each policy on the bandit
            environment. Larger values permit a better estimate of the
            variance in payoff / cumulative regret for each policy.
        plot : bool
            Whether to generate a plot of the policy's average reward and
            regret across the episodes. Default is True.
        seed : int
            The seed for the random number generator. Default is None.
        smooth_weight : float in [0, 1]
            The smoothing weight. Values closer to 0 result in less smoothing,
            values closer to 1 produce more aggressive smoothing. Default is
            0.999.
        out_dir : str or None
            Plots will be saved to this directory if `plot` is True. If
            `out_dir` is None, plots will not be saved. Default is None.
        """  # noqa: E501
        self.init_logs(policies)

        all_axes = [None] * len(policies)
        if plot and _PLOTTING:
            fig, all_axes = plt.subplots(len(policies), 2, sharex=True)
            fig.set_size_inches(10.5, len(policies) * 5.25)

        for policy, axes in zip(policies, all_axes):
            if seed:
                np.random.seed(seed)

            bandit.reset()
            policy.reset()

            self.train(
                policy,
                bandit,
                n_trials,
                n_duplicates,
                axes=axes,
                plot=plot,
                verbose=False,
                out_dir=out_dir,
                smooth_weight=smooth_weight,
            )

        # enforce the same y-ranges across plots for straightforward comparison
        a1_r, a2_r = zip(*[(a1.get_ylim(), a2.get_ylim()) for (a1, a2) in all_axes])

        a1_min = min(a1_r, key=lambda x: x[0])[0]
        a1_max = max(a1_r, key=lambda x: x[1])[1]
        a2_min = min(a2_r, key=lambda x: x[0])[0]
        a2_max = max(a2_r, key=lambda x: x[1])[1]

        for (a1, a2) in all_axes:
            a1.set_ylim(a1_min, a1_max)
            a2.set_ylim(a2_min, a2_max)

        if plot and _PLOTTING:
            if out_dir is not None:
                plt.savefig(op.join(out_dir, "bandit_comparison.png"), dpi=300)
            plt.show()

    def train(
        self,
        policy,
        bandit,
        n_trials,
        n_duplicates,
        plot=True,
        axes=None,
        verbose=True,
        print_every=100,
        smooth_weight=0.999,
        out_dir=None,
    ):
        """
        Train a MAB policies on a multi-armed bandit problem, logging training
        statistics along the way.

        Parameters
        ----------
        policy : :class:`BanditPolicyBase <numpy_ml.bandits.policies.BanditPolicyBase>` instance
            The multi-armed bandit policy to train.
        bandit : :class:`Bandit <numpy_ml.bandits.bandits.Bandit>` instance
            The environment to run the policy on.
        n_trials : int
            The number of trials per run.
        n_duplicates: int
            The number of runs to evaluate
        plot : bool
            Whether to generate a plot of the policy's average reward and
            regret across the episodes. Default is True.
        axes : list of :py:class:`Axis <matplotlib.axes.Axis>` instances or None
            If not None and ``plot = True``, these are the axes that will be
            used to plot the cumulative reward and regret, respectively.
            Default is None.
        verbose : boolean
            Whether to print run statistics during training. Default is True.
        print_every : int
            The number of episodes to run before printing loss values to
            stdout. This is ignored if ``verbose`` is false. Default is 100.
        smooth_weight : float in [0, 1]
            The smoothing weight. Values closer to 0 result in less smoothing,
            values closer to 1 produce more aggressive smoothing. Default is
            0.999.
        out_dir : str or None
            Plots will be saved to this directory if `plot` is True. If
            `out_dir` is None, plots will not be saved. Default is None.

        Returns
        -------
        policy : :class:`BanditPolicyBase <numpy_ml.bandits.policies.BanditPolicyBase>` instance
            The policy trained during the last (i.e. most recent) duplicate
            run.
        """  # noqa: E501
        if not str(policy) in self.logs:
            self.init_logs(policy)

        p = str(policy)
        D, L = n_duplicates, self.logs

        for d in range(D):
            if verbose:
                print("\nDUPLICATE {}/{}\n".format(d + 1, D))

            bandit.reset()
            policy.reset()

            avg_oracle_reward, cregret = 0, 0
            for trial_id in range(n_trials):
                rwd, arm, orwd, oarm = self._train_step(bandit, policy)

                loss = mse(bandit, policy)
                regret = orwd - rwd

                avg_oracle_reward += orwd
                cregret += regret

                L[p]["mse"][trial_id + 1].append(loss)
                L[p]["reward"][trial_id + 1].append(rwd)
                L[p]["regret"][trial_id + 1].append(regret)
                L[p]["cregret"][trial_id + 1].append(cregret)
                L[p]["optimal_arm"][trial_id + 1].append(oarm)
                L[p]["selected_arm"][trial_id + 1].append(arm)
                L[p]["optimal_reward"][trial_id + 1].append(orwd)

                if (trial_id + 1) % print_every == 0 and verbose:
                    fstr = "Trial {}/{}, {}/{}, Regret: {:.4f}"
                    print(fstr.format(trial_id + 1, n_trials, d + 1, D, regret))

            avg_oracle_reward /= n_trials

            if verbose:
                self._print_run_summary(bandit, policy, regret)

        if plot and _PLOTTING:
            self._plot_reward(avg_oracle_reward, policy, smooth_weight, axes, out_dir)

        return policy

    def _train_step(self, bandit, policy):
        P, B = policy, bandit
        C = B.get_context() if hasattr(B, "get_context") else None
        rwd, arm = P.act(B, C)
        oracle_rwd, oracle_arm = B.oracle_payoff(C)
        return rwd, arm, oracle_rwd, oracle_arm

    def init_logs(self, policies):
        """
        Initialize the episode logs.

        Notes
        -----
        Training logs are represented as a nested set of dictionaries with the
        following structure:

            log[model_id][metric][trial_number][duplicate_number]

        For example, ``logs['model1']['regret'][3][1]`` holds the regret value
        accrued on the 3rd trial of the 2nd duplicate run for model1.

        Available fields are 'regret', 'cregret' (cumulative regret), 'reward',
        'mse' (mean-squared error between estimated arm EVs and the true EVs),
        'optimal_arm', 'selected_arm', and 'optimal_reward'.
        """
        if not isinstance(policies, list):
            policies = [policies]

        self.logs = {
            str(p): {
                "mse": defaultdict(lambda: []),
                "regret": defaultdict(lambda: []),
                "reward": defaultdict(lambda: []),
                "cregret": defaultdict(lambda: []),
                "optimal_arm": defaultdict(lambda: []),
                "selected_arm": defaultdict(lambda: []),
                "optimal_reward": defaultdict(lambda: []),
            }
            for p in policies
        }

    def _print_run_summary(self, bandit, policy, regret):
        if not hasattr(policy, "ev_estimates") or len(policy.ev_estimates) == 0:
            return None

        evs, se = bandit.arm_evs, []
        fstr = "Arm {}: {:.4f} v. {:.4f}"
        ests = sorted(policy.ev_estimates.items(), key=lambda x: x[0])
        print("\n\nEstimated vs. Real EV\n" + "-" * 21)
        for ix, (est, ev) in enumerate(zip(ests, evs)):
            print(fstr.format(ix + 1, est[1], ev))
            se.append((est[1] - ev) ** 2)
        fstr = "\nFinal MSE: {:.4f}\nFinal Regret: {:.4f}\n\n"
        print(fstr.format(np.mean(se), regret))

    def _plot_reward(self, optimal_rwd, policy, smooth_weight, axes=None, out_dir=None):
        L = self.logs[str(policy)]
        smds = self._smoothed_metrics(policy, optimal_rwd, smooth_weight)

        if axes is None:
            fig, [ax1, ax2] = plt.subplots(1, 2)
        else:
            assert len(axes) == 2
            ax1, ax2 = axes

        e_ids = range(1, len(L["reward"]) + 1)
        plot_params = [[ax1, ax2], ["reward", "cregret"], ["b", "r"], [optimal_rwd, 0]]

        for (ax, m, c, opt) in zip(*plot_params):
            avg, std = "sm_{}_avg sm_{}_std".format(m, m).split()
            ax.plot(e_ids, smds[avg], color=c)
            ax.axhline(opt, 0, 1, color=c, ls="--")
            ax.fill_between(
                e_ids,
                smds[avg] + smds[std],
                smds[avg] - smds[std],
                color=c,
                alpha=0.25,
            )
            ax.set_xlabel("Trial")
            m = "Cumulative Regret" if m == "cregret" else m
            ax.set_ylabel("Smoothed Avg. {}".format(m.title()))

            if axes is None:
                ax.set_aspect(np.diff(ax.get_xlim()) / np.diff(ax.get_ylim()))

            if axes is not None:
                ax.set_title(str(policy))

        if axes is None:
            fig.suptitle(str(policy))
            fig.tight_layout()

            if out_dir is not None:
                bid = policy.hyperparameters["id"]
                plt.savefig(op.join(out_dir, f"{bid}.png"), dpi=300)
            plt.show()
        return ax1, ax2

    def _smoothed_metrics(self, policy, optimal_rwd, smooth_weight):
        L = self.logs[str(policy)]

        # pre-allocate smoothed data structure
        smds = {}
        for m in L.keys():
            if m == "selections":
                continue

            smds["sm_{}_avg".format(m)] = np.zeros(len(L["reward"]))
            smds["sm_{}_avg".format(m)][0] = np.mean(L[m][1])

            smds["sm_{}_std".format(m)] = np.zeros(len(L["reward"]))
            smds["sm_{}_std".format(m)][0] = np.std(L[m][1])

        smoothed = {m: L[m][1] for m in L.keys()}
        for e_id in range(2, len(L["reward"]) + 1):
            for m in L.keys():
                if m == "selections":
                    continue
                prev, cur = smoothed[m], L[m][e_id]
                smoothed[m] = [smooth(p, c, smooth_weight) for p, c in zip(prev, cur)]
                smds["sm_{}_avg".format(m)][e_id - 1] = np.mean(smoothed[m])
                smds["sm_{}_std".format(m)][e_id - 1] = np.std(smoothed[m])
        return smds
