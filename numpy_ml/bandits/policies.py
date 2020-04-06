import numbers
from abc import ABC, abstractmethod

import numpy as np


class BanditPolicyBase(ABC):
    def __init__(self):
        """A simple base class for multi-arm bandit policies"""
        self.step = 0
        self.pull_counts = {}
        self.ev_estimates = {}
        self.is_initialized = False
        super().__init__()

    def __repr__(self):
        HP = self.hyperparameters
        params = ", ".join(["{}={}".format(k, v) for (k, v) in HP.items() if k != "id"])
        return "{}({})".format(HP["id"], params)

    @property
    def hyperparameters(self):
        return {}

    @property
    def parameters(self):
        return {}

    def act(self, bandit):
        """
        Select an arm and sample from its payoff distribution.

        Parameters
        ----------
        bandit : :class:`bandit.Bandit` instance
            The multi-arm bandit to act upon

        Returns
        -------
        rwd : float
            The reward received after pulling `arm_id`
        arm_id : int
            The arm that was pulled to generate `rwd`
        """
        if not self.is_initialized:
            self.pull_counts = {i: 0 for i in range(bandit.n_arms)}
            self._initialize_params(bandit)

        arm_id = self._select_arm(bandit)
        rwd = self._pull_arm(bandit, arm_id)
        self._update_params(arm_id, rwd)
        return rwd, arm_id

    def _pull_arm(self, bandit, arm_id):
        self.step += 1
        self.pull_counts[arm_id] += 1
        return bandit.pull(arm_id)

    def _initialize_params(self, bandit):
        self.ev_estimates = {i: self.ev_prior for i in range(bandit.n_arms)}
        self.is_initialized = True

    def reset(self):
        """Reset policy parameters and counters to their initial state"""
        self.step = 0
        self._reset_params()
        self.pull_counts = {}
        self.is_initialized = False

    @abstractmethod
    def _select_arm(self, bandit):
        pass

    @abstractmethod
    def _update_params(self, bandit):
        pass

    @abstractmethod
    def _reset_params(self):
        self.ev_estimates = {}


class EpsilonGreedy(BanditPolicyBase):
    def __init__(self, epsilon=0.05, ev_prior=99):
        """
        An epsilon-greedy policy for multi-arm bandit problems.

        Notes
        -----
        Epsilon-greedy policies greedily select the arm with the highest
        expected payoff with probability :math:`1-\epsilon`, and selects an arm
        uniformly at random with probability :math:`\epsilon`.

        Parameters
        ----------
        epsilon : float in [0, 1]
            The probability of taking a random action. Default is 0.05.
        ev_prior : float
            The starting expected value for each arm before any data has been
            observed. Default is 99.
        """
        super().__init__()
        self.epsilon = epsilon
        self.ev_prior = ev_prior

    @property
    def parameters(self):
        return {"ev_estimates": self.ev_estimates}

    @property
    def hyperparameters(self):
        return {
            "id": "EpsilonGreedy",
            "epsilon": self.epsilon,
            "ev_prior": self.ev_prior,
        }

    def _select_arm(self, bandit):
        if np.random.rand() < self.epsilon:
            arm_id = np.random.choice(bandit.n_arms)
        else:
            ests = self.ev_estimates
            (arm_id, _) = max(ests.items(), key=lambda x: x[1])
        return arm_id

    def _update_params(self, arm_id, reward):
        E, C = self.ev_estimates, self.pull_counts
        E[arm_id] += (reward - E[arm_id]) / (C[arm_id])

    def _reset_params(self):
        self.ev_estimates = {}


class UCB1(BanditPolicyBase):
    def __init__(self, C=1, ev_prior=0.5):
        """
        A UCB1 policy [*]_ for multi-arm bandit problems.

        Notes
        -----
        The UCB1 algorithm guarantees the cumulative regret is bounded by log
        `t`, where `t` is the current timestep. To make this guarantee UCB1
        assumes all arm payoffs are between 0 and 1.

        Under UCB1, the upper confidence bound on the expected value for
        pulling arm `a` at timestep `t` is:

        .. math::

            UCB(a, t) = EV_t(a) + C \sqrt{\\frac{2 \log t}{N_t(a)}}

        where `UCB(a, t)` is the upper confidence bound on the expected value
        of arm `a` at time `t`, :math:`EV_t(a)` is the average of the rewards
        recieved so far from pulling arm `a`, `C` is a parameter controlling
        the confidence upper bound of the estimate for `UCB(a, t)` (for
        logarithmic regret bounds, `C` must equal 1), and `N_t(a)` is the
        number of times arm `a` has been pulled during the previous `t - 1`
        timesteps.

        References
        ----------
        .. [*] Auer, P., Cesa-Bianchi, N., & Fischer, P. (2002). Finite-time
           analysis of the multiarmed bandit problem. *Machine Learning,
           47(2)*.

        Parameters
        ----------
        C : float in (0, +infinity)
            A confidence/optimisim parameter affecting the degree of
            exploration. The UCB1 algorithm assumes `C=1`. Default is 1.
        ev_prior : float
            The starting expected value for each arm before any data has been
            observed. Default is 0.5.
        """
        self.C = C
        self.ev_prior = ev_prior
        super().__init__()

    @property
    def parameters(self):
        return {
            "ev_estimates": self.ev_estimates,
        }

    @property
    def hyperparameters(self):
        return {
            "C": self.C,
            "id": "UCB1",
            "ev_prior": self.ev_prior,
        }

    def _select_arm(self, bandit):
        # add eps to avoid divide-by-zero errors on the first pull of each arm
        eps = np.finfo(float).eps
        N, T = bandit.n_arms, self.step + 1
        E, C = self.ev_estimates, self.pull_counts
        scores = [E[a] + self.C * np.sqrt(np.log(T) / (C[a] + eps)) for a in range(N)]
        return np.argmax(scores)

    def _update_params(self, arm_id, reward):
        E, C = self.ev_estimates, self.pull_counts
        E[arm_id] += (reward - E[arm_id]) / (C[arm_id])

    def _reset_params(self):
        self.ev_estimates = {}


class ThompsonSamplingBetaBinomial(BanditPolicyBase):
    def __init__(self, alpha=1, beta=1):
        """
        A conjugate Thompson sampling policy for multi-arm bandits with
        Bernoulli likelihoods.

        Notes
        -----
        The policy assumes independent Beta priors on the arm payoff
        probabilities, :math:`theta`:

        ..math::

            \\theta_k \sim \\text{Beta}(\\alpha_k, \\beta_k)

        where :math:`k \in 1,\ldots,K` indexes arms in the MAB and
        :math:`\\theta_k` is the parameter of the Bernoulli likelihood
        for arm `k`. The sampler proceeds by selecting actions in proportion to
        the posterior probability that they are optimal. Thanks to the
        conjugacy between the Beta prior and Bernoulli likelihood the posterior
        for each arm is also Beta-distributed and can be sampled from
        efficiently.

        Parameters
        ----------
        alpha : float or list of length `K`
            Parameter for the Beta prior on arm payouts. If a float, this value
            will be used in the prior for all of the `K` arms.
        beta : float or list of length `K`
            Parameter for the Beta prior on arm payouts. If a float, this value
            will be used in the prior for all of the `K` arms.
        """
        super().__init__()
        self.alphas, self.betas = [], []
        self.alpha, self.beta = alpha, beta
        self.is_initialized = False

    @property
    def parameters(self):
        return {
            "ev_estimates": self.ev_estimates,
            "alphas": self.alphas,
            "betas": self.betas,
        }

    @property
    def hyperparameters(self):
        return {
            "id": "ThompsonSamplingBetaBinomial",
            "alpha": self.alpha,
            "beta": self.beta,
        }

    def _initialize_params(self, bandit):
        bhp = bandit.hyperparameters
        assert bhp["id"] == "MultiArmBanditBernoulliPayoff"

        # initialize the model prior
        if isinstance(self.alpha, numbers.Number):
            self.alphas = [self.alpha] * bandit.n_arms
        if isinstance(self.beta, numbers.Number):
            self.betas = [self.beta] * bandit.n_arms
        assert len(self.alphas) == len(self.betas) == bandit.n_arms

        self.ev_estimates = {i: self._map_estimate(i, 1) for i in range(bandit.n_arms)}
        self.is_initialized = True

    def _select_arm(self, bandit):
        if not self.is_initialized:
            self._initialize_prior(bandit)

        # draw a sample from the current model posterior
        posterior_sample = np.random.beta(self.alphas, self.betas)

        # greedily select an action based on this sample
        return np.argmax(posterior_sample)

    def _update_params(self, arm_id, rwd):
        """
        Compute the parameters of the Beta posterior, P(payoff prob | rwd),
        for arm `arm_id`.
        """
        self.alphas[arm_id] += rwd
        self.betas[arm_id] += 1 - rwd
        self.ev_estimates[arm_id] = self._map_estimate(arm_id, rwd)

    def _map_estimate(self, arm_id, rwd):
        """compute the current MAP estimate for an arm's payoff probability"""
        A, B = self.alphas, self.betas
        if A[arm_id] > 1 and B[arm_id] > 1:
            map_payoff_prob = (A[arm_id] - 1) / (A[arm_id] + B[arm_id] - 2)
        elif A[arm_id] < 1 and B[arm_id] < 1:
            map_payoff_prob = rwd  # 0 or 1 equally likely, make a guess
        elif A[arm_id] <= 1 and B[arm_id] > 1:
            map_payoff_prob = 0
        elif A[arm_id] > 1 and B[arm_id] <= 1:
            map_payoff_prob = 1
        else:
            map_payoff_prob = 0.5
        return map_payoff_prob

    def _reset_params(self):
        self.alphas, self.betas = [], []
        self.ev_estimates = {}
