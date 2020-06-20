"""A module containing different variations on multi-armed bandit environments."""

from abc import ABC, abstractmethod

import numpy as np

from numpy_ml.utils.testing import random_one_hot_matrix, is_number


class Bandit(ABC):
    def __init__(self, rewards, reward_probs, context=None):
        assert len(rewards) == len(reward_probs)
        self.step = 0
        self.n_arms = len(rewards)

        super().__init__()

    def __repr__(self):
        """A string representation for the bandit"""
        HP = self.hyperparameters
        params = ", ".join(["{}={}".format(k, v) for (k, v) in HP.items() if k != "id"])
        return "{}({})".format(HP["id"], params)

    @property
    def hyperparameters(self):
        """A dictionary of the bandit hyperparameters"""
        return {}

    @abstractmethod
    def oracle_payoff(self, context=None):
        """
        Return the expected reward for an optimal agent.

        Parameters
        ----------
        context : :py:class:`ndarray <numpy.ndarray>` of shape `(D, K)` or None
            The current context matrix for each of the bandit arms, if
            applicable. Default is None.

        Returns
        -------
        optimal_rwd : float
            The expected reward under an optimal policy.
        """
        pass

    def pull(self, arm_id, context=None):
        """
        "Pull" (i.e., sample from) a given arm's payoff distribution.

        Parameters
        ----------
        arm_id : int
            The integer ID of the arm to sample from
        context : :py:class:`ndarray <numpy.ndarray>` of shape `(D,)` or None
            The context vector for the current timestep if this is a contextual
            bandit. Otherwise, this argument is unused and defaults to None.

        Returns
        -------
        reward : float
            The reward sampled from the given arm's payoff distribution
        """
        assert arm_id < self.n_arms

        self.step += 1
        return self._pull(arm_id, context)

    def reset(self):
        """Reset the bandit step and action counters to zero."""
        self.step = 0

    @abstractmethod
    def _pull(self, arm_id):
        pass


class MultinomialBandit(Bandit):
    def __init__(self, payoffs, payoff_probs):
        """
        A multi-armed bandit where each arm is associated with a different
        multinomial payoff distribution.

        Parameters
        ----------
        payoffs : ragged list of length `K`
            The payoff values for each of the `n` bandits. ``payoffs[k][i]``
            holds the `i` th payoff value for arm `k`.
        payoff_probs : ragged list of length `K`
            A list of the probabilities associated with each of the payoff
            values in ``payoffs``. ``payoff_probs[k][i]`` holds the probability
            of payoff index `i` for arm `k`.
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
        self.best_arm = np.argmax(self.arm_evs)

    @property
    def hyperparameters(self):
        """A dictionary of the bandit hyperparameters"""
        return {
            "id": "MultinomialBandit",
            "payoffs": self.payoffs,
            "payoff_probs": self.payoff_probs,
        }

    def oracle_payoff(self, context=None):
        """
        Return the expected reward for an optimal agent.

        Parameters
        ----------
        context : :py:class:`ndarray <numpy.ndarray>` of shape `(D, K)` or None
            Unused. Default is None.

        Returns
        -------
        optimal_rwd : float
            The expected reward under an optimal policy.
        optimal_arm : float
            The arm ID with the largest expected reward.
        """
        return self.best_ev, self.best_arm

    def _pull(self, arm_id, context):
        payoffs = self.payoffs[arm_id]
        probs = self.payoff_probs[arm_id]
        return np.random.choice(payoffs, p=probs)


class BernoulliBandit(Bandit):
    def __init__(self, payoff_probs):
        """
        A multi-armed bandit where each arm is associated with an independent
        Bernoulli payoff distribution.

        Parameters
        ----------
        payoff_probs : list of length `K`
            A list of the payoff probability for each arm. ``payoff_probs[k]``
            holds the probability of payoff for arm `k`.
        """
        payoffs = [1] * len(payoff_probs)
        super().__init__(payoffs, payoff_probs)

        for p in payoff_probs:
            assert p >= 0 and p <= 1

        self.payoffs = np.array(payoffs)
        self.payoff_probs = np.array(payoff_probs)

        self.arm_evs = self.payoff_probs
        self.best_ev = np.max(self.arm_evs)
        self.best_arm = np.argmax(self.arm_evs)

    @property
    def hyperparameters(self):
        """A dictionary of the bandit hyperparameters"""
        return {
            "id": "BernoulliBandit",
            "payoff_probs": self.payoff_probs,
        }

    def oracle_payoff(self, context=None):
        """
        Return the expected reward for an optimal agent.

        Parameters
        ----------
        context : :py:class:`ndarray <numpy.ndarray>` of shape `(D, K)` or None
            Unused. Default is None.

        Returns
        -------
        optimal_rwd : float
            The expected reward under an optimal policy.
        optimal_arm : float
            The arm ID with the largest expected reward.
        """
        return self.best_ev, self.best_arm

    def _pull(self, arm_id, context):
        return int(np.random.rand() <= self.payoff_probs[arm_id])


class GaussianBandit(Bandit):
    def __init__(self, payoff_dists, payoff_probs):
        """
        A multi-armed bandit that is similar to
        :class:`BernoulliBandit`, but instead of each arm having
        a fixed payout of 1, the payoff values are sampled from independent
        Gaussian RVs.

        Parameters
        ----------
        payoff_dists : list of 2-tuples of length `K`
            The parameters the distributions over payoff values for each of the
            `n` arms. Specifically, ``payoffs[k]`` is a tuple of (mean, variance)
            for the Gaussian distribution over payoffs associated with arm `k`.
        payoff_probs : list of length `n`
            A list of the probabilities associated with each of the payoff
            values in ``payoffs``. ``payoff_probs[k]`` holds the probability of
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
        self.best_arm = np.argmax(self.arm_evs)

    @property
    def hyperparameters(self):
        """A dictionary of the bandit hyperparameters"""
        return {
            "id": "GaussianBandit",
            "payoff_dists": self.payoff_dists,
            "payoff_probs": self.payoff_probs,
        }

    def _pull(self, arm_id, context):
        mean, var = self.payoff_dists[arm_id]

        reward = 0
        if np.random.rand() < self.payoff_probs[arm_id]:
            reward = np.random.normal(mean, var)

        return reward

    def oracle_payoff(self, context=None):
        """
        Return the expected reward for an optimal agent.

        Parameters
        ----------
        context : :py:class:`ndarray <numpy.ndarray>` of shape `(D, K)` or None
            Unused. Default is None.

        Returns
        -------
        optimal_rwd : float
            The expected reward under an optimal policy.
        optimal_arm : float
            The arm ID with the largest expected reward.
        """
        return self.best_ev, self.best_arm


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
        G : :class:`Graph <numpy_ml.utils.graphs.Graph>` instance
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
        self.best_arm = np.argmax(self.arm_evs)

        placeholder = [None] * len(self.paths)
        super().__init__(placeholder, placeholder)

    @property
    def hyperparameters(self):
        """A dictionary of the bandit hyperparameters"""
        return {
            "id": "ShortestPathBandit",
            "G": self.G,
            "end_vertex": self.end_vertex,
            "start_vertex": self.start_vertex,
        }

    def oracle_payoff(self, context=None):
        """
        Return the expected reward for an optimal agent.

        Parameters
        ----------
        context : :py:class:`ndarray <numpy.ndarray>` of shape `(D, K)` or None
            Unused. Default is None.

        Returns
        -------
        optimal_rwd : float
            The expected reward under an optimal policy.
        optimal_arm : float
            The arm ID with the largest expected reward.
        """
        return self.best_ev, self.best_arm

    def _calc_arm_evs(self):
        I2V = self.G.get_vertex
        evs = np.zeros(len(self.paths))
        for p_ix, path in enumerate(self.paths):
            for ix, v_i in enumerate(path[:-1]):
                e = [e for e in self.adj_dict[v_i] if e.to == I2V(path[ix + 1])][0]
                evs[p_ix] -= e.weight
        return evs

    def _pull(self, arm_id, context):
        reward = 0
        I2V = self.G.get_vertex
        path = self.paths[arm_id]
        for ix, v_i in enumerate(path[:-1]):
            e = [e for e in self.adj_dict[v_i] if e.to == I2V(path[ix + 1])][0]
            reward -= e.weight
        return reward


class ContextualBernoulliBandit(Bandit):
    def __init__(self, context_probs):
        """
        A contextual version of :class:`BernoulliBandit` where each binary
        context feature is associated with an independent Bernoulli payoff
        distribution.

        Parameters
        ----------
        context_probs : :py:class:`ndarray <numpy.ndarray>` of shape `(D, K)`
            A matrix of the payoff probabilities associated with each of the
            `D` context features, for each of the `K` arms. Index `(i, j)`
            contains the probability of payoff for arm `j` under context `i`.
        """
        D, K = context_probs.shape

        # use a dummy placeholder variable to initialize the Bandit superclass
        placeholder = [None] * K
        super().__init__(placeholder, placeholder)

        self.context_probs = context_probs
        self.arm_evs = self.context_probs
        self.best_evs = self.arm_evs.max(axis=1)
        self.best_arms = self.arm_evs.argmax(axis=1)

    @property
    def hyperparameters(self):
        """A dictionary of the bandit hyperparameters"""
        return {
            "id": "ContextualBernoulliBandit",
            "context_probs": self.context_probs,
        }

    def get_context(self):
        """
        Sample a random one-hot context vector. This vector will be the same
        for all arms.

        Returns
        -------
        context : :py:class:`ndarray <numpy.ndarray>` of shape `(D, K)`
            A random `D`-dimensional one-hot context vector repeated for each
            of the `K` bandit arms.
        """
        D, K = self.context_probs.shape
        context = np.zeros((D, K))
        context[np.random.choice(D), :] = 1
        return random_one_hot_matrix(1, D).ravel()

    def oracle_payoff(self, context):
        """
        Return the expected reward for an optimal agent.

        Parameters
        ----------
        context : :py:class:`ndarray <numpy.ndarray>` of shape `(D, K)` or None
            The current context matrix for each of the bandit arms.

        Returns
        -------
        optimal_rwd : float
            The expected reward under an optimal policy.
        optimal_arm : float
            The arm ID with the largest expected reward.
        """
        context_id = context[:, 0].argmax()
        return self.best_evs[context_id], self.best_arms[context_id]

    def _pull(self, arm_id, context):
        D, K = self.context_probs.shape
        arm_probs = context[:, arm_id] @ self.context_probs
        arm_rwds = (np.random.rand(K) <= arm_probs).astype(int)
        return arm_rwds[arm_id]


class ContextualLinearBandit(Bandit):
    def __init__(self, K, D, payoff_variance=1):
        r"""
        A contextual linear multi-armed bandit.

        Notes
        -----
        In a contextual linear bandit the expected payoff of an arm :math:`a
        \in \mathcal{A}` at time `t` is a linear combination of its context
        vector :math:`\mathbf{x}_{t,a}` with a coefficient vector
        :math:`\theta_a`:

        .. math::

            \mathbb{E}[r_{t, a} \mid \mathbf{x}_{t, a}] = \mathbf{x}_{t,a}^\top \theta_a

        In this implementation, the arm coefficient vectors :math:`\theta` are
        initialized independently from a uniform distribution on the interval
        [-1, 1], and the specific reward at timestep `t` is normally
        distributed:

        .. math::

            r_{t, a} \mid \mathbf{x}_{t, a} \sim
                \mathcal{N}(\mathbf{x}_{t,a}^\top \theta_a, \sigma_a^2)

        Parameters
        ----------
        K : int
            The number of bandit arms
        D : int
            The dimensionality of the context vectors
        payoff_variance : float or :py:class:`ndarray <numpy.ndarray>` of shape `(K,)`
            The variance of the random noise in the arm payoffs. If a float,
            the variance is assumed to be equal for each arm. Default is 1.
        """
        if is_number(payoff_variance):
            payoff_variance = [payoff_variance] * K

        assert len(payoff_variance) == K
        assert all(v > 0 for v in payoff_variance)

        self.K = K
        self.D = D
        self.payoff_variance = payoff_variance

        # use a dummy placeholder variable to initialize the Bandit superclass
        placeholder = [None] * K
        super().__init__(placeholder, placeholder)

        # initialize the theta matrix
        self.thetas = np.random.uniform(-1, 1, size=(D, K))
        self.thetas /= np.linalg.norm(self.thetas, 2)

    @property
    def hyperparameters(self):
        """A dictionary of the bandit hyperparameters"""
        return {
            "id": "ContextualLinearBandit",
            "K": self.K,
            "D": self.D,
            "payoff_variance": self.payoff_variance,
        }

    @property
    def parameters(self):
        """A dictionary of the current bandit parameters"""
        return {"thetas": self.thetas}

    def get_context(self):
        """
        Sample the context vectors for each arm from a multivariate standard
        normal distribution.

        Returns
        -------
        context : :py:class:`ndarray <numpy.ndarray>` of shape `(D, K)`
            A `D`-dimensional context vector sampled from a standard normal
            distribution for each of the `K` bandit arms.
        """
        return np.random.normal(size=(self.D, self.K))

    def oracle_payoff(self, context):
        """
        Return the expected reward for an optimal agent.

        Parameters
        ----------
        context : :py:class:`ndarray <numpy.ndarray>` of shape `(D, K)` or None
            The current context matrix for each of the bandit arms, if
            applicable. Default is None.

        Returns
        -------
        optimal_rwd : float
            The expected reward under an optimal policy.
        optimal_arm : float
            The arm ID with the largest expected reward.
        """
        best_arm = np.argmax(self.arm_evs)
        return self.arm_evs[best_arm], best_arm

    def _pull(self, arm_id, context):
        K, thetas = self.K, self.thetas
        self._noise = np.random.normal(scale=self.payoff_variance, size=self.K)
        self.arm_evs = np.array([context[:, k] @ thetas[:, k] for k in range(K)])
        return (self.arm_evs + self._noise)[arm_id]
