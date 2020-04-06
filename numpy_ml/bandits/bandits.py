from abc import ABC, abstractmethod

import numpy as np


class Bandit(ABC):
    def __init__(self, rewards, reward_probs, context=None):
        assert len(rewards) == len(reward_probs)

        self.step = 0
        self.n_arms = len(rewards)
        self.pull_counts = {i: 0 for i in range(self.n_arms)}

        super().__init__()

    def __repr__(self):
        HP = self.hyperparameters
        params = ", ".join(["{}={}".format(k, v) for (k, v) in HP.items() if k != "id"])
        return "{}({})".format(HP["id"], params)

    def pull(self, arm_id):
        """
        "Pull" (i.e., sample from) a given arm's payoff distribution.

        Parameters
        ----------
        arm_id : int
            The integer ID of the arm to sample from

        Returns
        -------
        reward : float
            The reward sampled from the given arm's payoff distribution
        """
        assert arm_id < self.n_arms

        self.step += 1
        self.pull_counts[arm_id] += 1

        return self._pull(arm_id)

    def reset(self):
        self.step = 0
        self.pull_counts = {i: 0 for i in range(self.n_arms)}

    @abstractmethod
    def _pull(self, arm_id):
        pass

    @abstractmethod
    def hyperparameters(self):
        return {}


class MultiArmedBanditMultinomialPayoff(Bandit):
    def __init__(self, payoffs, payoff_probs):
        """
        A multi-armed bandit where each arm is associated with a different
        multinomial payoff distribution.

        Parameters
        ----------
        payoffs : ragged list of length `n`
            The payoff values for each of the `n` bandits. `payoffs[k][i]`
            holds the `i`th payoff value for arm `k`.
        payoff_probs : ragged list of length `n`
            A list of the probabilities associated with each of the payoff
            values in `payoffs`. `payoff_probs[k][i]` holds the probability of
            payoff index `i` for arm `k`.
        """
        super().__init__(payoffs, payoff_probs)

        for r, rp in zip(payoffs, payoff_probs):
            assert len(r) == len(rp)
            np.testing.assert_almost_equal(sum(rp), 1.0)

        payoffs = np.array([np.array(x) for x in payoffs])
        payoff_probs = np.array([np.array(x) for x in payoff_probs])

        self.payoffs = payoffs
        self.payoff_probs = payoff_probs
        self.arm_evs = np.array([sum(p * v) for p, v in zip(payoff_probs, payoffs)])
        self.best_ev = np.max(self.arm_evs)

    def __repr__(self):
        fstr = "MultiArmedBanditMultinomialPayoff(payoffs={}, payoff_probs={})"
        return fstr.format(self.payoffs, self.payoff_probs)

    @property
    def hyperparameters(self):
        return {
            "id": "MultiArmedBanditMultinomialPayoff",
            "payoffs": self.payoffs,
            "payoff_probs": self.payoff_probs,
        }

    def _pull(self, arm_id):
        payoffs = self.payoffs[arm_id]
        probs = self.payoff_probs[arm_id]
        return np.random.choice(payoffs, p=probs)


class MultiArmedBanditBernoulliPayoff(Bandit):
    def __init__(self, payoff_probs):
        payoffs = [1] * len(payoff_probs)
        super().__init__(payoffs, payoff_probs)

        for p in payoff_probs:
            assert p >= 0 and p <= 1

        self.payoffs = np.array(payoffs)
        self.payoff_probs = np.array(payoff_probs)

        self.arm_evs = self.payoff_probs
        self.best_ev = np.max(self.arm_evs)

    @property
    def hyperparameters(self):
        return {
            "id": "MultiArmedBanditBernoulliPayoff",
            "payoff_probs": self.payoff_probs,
        }

    def _pull(self, arm_id):
        return int(np.random.rand() <= self.payoff_probs[arm_id])


class MultiArmedBanditGaussianPayoff(Bandit):
    def __init__(self, payoff_dists, payoff_probs):
        """
        A multi-armed bandit that is similar to
        :class:`MultiArmedBanditBernoulliPayoff`, but instead of each arm having
        a fixed payout of 1, the payoff values are sampled from independent
        Gaussian RVs.

        Parameters
        ----------
        payoff_dists : list 2-tuples of length `n`
            The parameters the distributions over payoff values for each of the
            `n` arms. Specifically, `payoffs[k]` is a tuple of (mean, variance)
            for the Gaussian distribution over payoffs associated with arm `k`.
        payoff_probs : list of length `n`
            A list of the probabilities associated with each of the payoff
            values in `payoffs`. `payoff_probs[k]` holds the probability of
            payoff for arm `k`.
        """
        super().__init__(payoff_dists, payoff_probs)

        for (mean, var), rp in zip(payoff_dists, payoff_probs):
            assert var > 0
            assert np.testing.assert_almost_equal(sum(rp), 1.0)

        self.payoff_dists = payoff_dists
        self.payoff_probs = payoff_probs
        self.arm_evs = np.array([mu for (mu, var) in payoff_dists])
        self.best_ev = np.max(self.arm_evs)

    @property
    def hyperparameters(self):
        return {
            "id": "MultiArmedBanditGaussianPayoff",
            "payoff_dists": self.payoff_dists,
            "payoff_probs": self.payoff_probs,
        }

    def _pull(self, arm_id):
        mean, var = self.payoff_dists[arm_id]

        reward = 0
        if np.random.rand() < self.payoff_probs[arm_id]:
            reward = np.random.normal(mean, var)

        return reward


class ShortestPathBandit(Bandit):
    def __init__(self, G, start_vertex, end_vertex):
        """
        A weighted graph shortest path problem formulated as a multi-armed
        bandit.

        Notes
        -----
        Each arm corresponds to a valid path through the graph from start to
        end vertex. The agent's goal is to find the path that minimizes the
        expected sum of the weights on the edges it traverses.

        Parameters
        ----------
        G : :doc:`Graph <numpy_ml.utils.graphs.Graph>` instance
            A weighted graph object. Weights can be fixed or probabilistic.
        start_vertex : int
            The index of the path's start vertex in the graph
        end_vertex : int
            The index of the path's end vertex in the graph
        """
        self.G = G
        self.end_vertex = end_vertex
        self.adj_dict = G.to_adj_dict()
        self.start_vertex = start_vertex
        self.paths = G.all_paths(start_vertex, end_vertex)

        self.arm_evs = self._calc_arm_evs()
        self.best_ev = np.max(self.arm_evs)

        placeholder = [None] * len(self.paths)
        super().__init__(placeholder, placeholder)

    def _calc_arm_evs(self):
        I2V = self.G.get_vertex
        evs = np.zeros(len(self.paths))
        for p_ix, path in enumerate(self.paths):
            for ix, v_i in enumerate(path[:-1]):
                e = [e for e in self.adj_dict[v_i] if e.to == I2V(path[ix + 1])][0]
                evs[p_ix] -= e.weight
        return evs

    @property
    def hyperparameters(self):
        return {
            "id": "ShortestPathBandit",
            "G": self.G,
            "end_vertex": self.end_vertex,
            "start_vertex": self.start_vertex,
        }

    def _pull(self, arm_id):
        reward = 0
        I2V = self.G.get_vertex
        path = self.paths[arm_id]
        for ix, v_i in enumerate(path[:-1]):
            e = [e for e in self.adj_dict[v_i] if e.to == I2V(path[ix + 1])][0]
            reward -= e.weight
        return reward
