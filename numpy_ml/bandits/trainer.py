from collections import defaultdict

import numpy as np


def mse(bandit, policy):
    """
    Computes the mean squared error between a policy's estimates of the
    expected arm payouts and the true expected payouts.
    """
    se = []
    evs = bandit.arm_evs
    ests = sorted(policy.ev_estimates.items(), key=lambda x: x[0])
    for ix, (est, ev) in enumerate(zip(ests, evs)):
        se.append((est[1] - ev) ** 2)
    return np.mean(se)


def smooth(prev, cur, weight):
    """
    Compute a simple weighted average of the previous and current value.

    Notes
    -----
    The smoothed value at timestep `t`, :math:`\\tilde{X}_t` is calculated as

    .. math::

        \\tilde{X}_t = \epsilon \\tilde{X}_{t-1} + (1 - \epsilon) X_t

    where :math:`X_t` is the value at timestep `t`, :math:`\\tilde{X}_{t-1}` is
    the value of the smoothed signal at timestep `t-1`, and :math:`\epsilon` is
    the smoothing weight.

    Parameters
    ----------
    prev : float or np.array of shape (N,)
        The value of the smoothed signal at the immediately preceding
        timestep.
    cur : float or np.array of shape (N,)
        The value of the signal at the current timestep
    weight : float or np.array of shape (N,)
        The smoothing weight. Values closer to 0 result in less smoothing,
        values closer to 1 produce more aggressive smoothing. If weight is an
        array, each dimension will be interpreted as a separate smoothing
        weight the corresponding dimension in `cur`.

    Returns
    -------
    smoothed : float or np.array of shape (N,)
        The smoothed signal
    """
    return weight * prev + (1 - weight) * cur


class MABTrainer:
    """
    An object to facilitate multi-armed bandit training and evaluation.
    """

    def train(
        self,
        policy,
        bandit,
        ep_length,
        n_episodes,
        n_duplicates,
        plot=True,
        print_every=100,
    ):
        """
        Train an MAB policy on a multi-arm bandit problem, logging training
        statistics along the way.

        Parameters
        ----------
        policy : :class:`BanditPolicyBase <numpy_ml.bandits.policies.BanditPolicyBase>` instance
            The multi-armed bandit policy to train.
        bandit : :class:`Bandit <numpy_ml.bandits.bandits.Bandit>` instance
            The environment to run the agent on.
        ep_length : int
            The number of pulls allowed in each episode
        n_episodes : int
            The number of episodes per run
        n_duplicates: int
            The number of runs to evaluate
        plot : bool
            Whether to generate a plot of the policy's average reward and
            regret across the episodes. Default is True.
        print_every : int
            The number of episodes to run before printing loss values to
            stdout. Default is 100.

        Returns
        -------
        policy : :class:`BanditPolicyBase <numpy_ml.bandits.policies.BanditPolicyBase>` instance
            The policy trained during the last (i.e. most recent) duplicate
            run.
        """
        self.init_logs()
        D, L = n_duplicates, self.logs

        for d in range(D):
            print("\nDUPLICATE {}/{}\n".format(d + 1, D))
            bandit.reset()
            policy.reset()

            cregret = 0
            for e_id in range(n_episodes):
                ep_reward = 0

                for s in range(ep_length):
                    rwd, arm = policy.act(bandit)
                    ep_reward += rwd

                loss = mse(bandit, policy)
                regret = (bandit.best_ev * ep_length) - ep_reward
                cregret += regret

                L["mse"][e_id + 1].append(loss)
                L["regret"][e_id + 1].append(regret)
                L["cregret"][e_id + 1].append(cregret)
                L["reward"][e_id + 1].append(ep_reward)

                if (e_id + 1) % print_every == 0:
                    fstr = "Ep. {}/{}, {}/{}, Regret: {:.4f}"
                    print(fstr.format(e_id + 1, n_episodes, d + 1, D, regret))

            self._print_run_summary(bandit, policy, regret)

        if plot:
            self._plot_reward(bandit.best_ev * ep_length, policy)

        return policy

    def init_logs(self):
        """Initialize the episode logs"""
        # In the logs, keys are episode numbers, and values are lists of length
        # `n_duplicates` holding the metric values for each duplicate of that
        # episode. For example, logs['regret'][3][2] holds the regret value
        # accrued on the 2nd duplicate of the 3rd episode
        self.logs = {
            "regret": defaultdict(lambda: []),
            "cregret": defaultdict(lambda: []),
            "reward": defaultdict(lambda: []),
            "mse": defaultdict(lambda: []),
        }

    def _print_run_summary(self, bandit, policy, regret):
        se = []
        evs = bandit.arm_evs
        ests = sorted(policy.ev_estimates.items(), key=lambda x: x[0])
        print("\n\nEstimated vs. Real EV\n" + "-" * 21)
        for ix, (est, ev) in enumerate(zip(ests, evs)):
            print("Arm {}: {:.4f} v. {:.4f}".format(ix + 1, est[1], ev))
            se.append((est[1] - ev) ** 2)
        print("\nFinal MSE: {:.4f}".format(np.mean(se)))
        print("Final Regret: {:.4f}\n\n".format(regret))

    def _plot_reward(self, optimal_rwd, policy, smooth_weight=0.999):
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Cannot import matplotlib. Plotting functionality disabled.")
            return

        L = self.logs
        smds = self._smoothed_metrics(optimal_rwd, smooth_weight)

        fig, [ax1, ax2] = plt.subplots(1, 2)

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
            ax.set_ylabel("Smoothed Avg. {}".format(m.title()), color=c)
            ax.set_aspect(np.diff(ax.get_xlim()) / np.diff(ax.get_ylim()))

        fig.suptitle(str(policy))
        fig.tight_layout()

        bid = policy.hyperparameters["id"]
        plt.savefig("img/{}.png".format(bid), dpi=300)

        plt.show()
        plt.close("all")

    def _smoothed_metrics(self, optimal_rwd, smooth_weight):
        L = self.logs

        # pre-allocate smoothed data structure
        smds = {}
        for m in L.keys():
            smds["sm_{}_avg".format(m)] = np.zeros(len(L["reward"]))
            smds["sm_{}_avg".format(m)][0] = np.mean(L[m][1])

            smds["sm_{}_std".format(m)] = np.zeros(len(L["reward"]))
            smds["sm_{}_std".format(m)][0] = np.std(L[m][1])

        smoothed = {m: L[m][1] for m in L.keys()}
        for e_id in range(2, len(L["reward"]) + 1):
            for m in L.keys():
                prev, cur = smoothed[m], L[m][e_id]
                smoothed[m] = [smooth(p, c, smooth_weight) for p, c in zip(prev, cur)]
                smds["sm_{}_avg".format(m)][e_id - 1] = np.mean(smoothed[m])
                smds["sm_{}_std".format(m)][e_id - 1] = np.std(smoothed[m])
        return smds
