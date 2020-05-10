"""A module containing exploration policies for various multi-armed bandit problems."""

from abc import ABC, abstractmethod
from collections import defaultdict

import numpy as np

from ..utils.testing import is_number


class BanditPolicyBase(ABC):
    def __init__(self):
        """A simple base class for multi-armed bandit policies"""
        self.step = 0
        self.ev_estimates = {}
        self.is_initialized = False
        super().__init__()

    def __repr__(self):
        """Return a string representation of the policy"""
        HP = self.hyperparameters
        params = ", ".join(["{}={}".format(k, v) for (k, v) in HP.items() if k != "id"])
        return "{}({})".format(HP["id"], params)

    @property
    def hyperparameters(self):
        """A dictionary containing the policy hyperparameters"""
        pass

    @property
    def parameters(self):
        """A dictionary containing the current policy parameters"""
        pass

    def act(self, bandit, context=None):
        """
        Select an arm and sample from its payoff distribution.

        Parameters
        ----------
        bandit : :class:`Bandit <numpy_ml.bandits.bandits.Bandit>` instance
            The multi-armed bandit to act upon
        context : :py:class:`ndarray <numpy.ndarray>` of shape `(D,)` or None
            The context vector for the current timestep if interacting with a
            contextual bandit. Otherwise, this argument is unused. Default is
            None.

        Returns
        -------
        rwd : float
            The reward received after pulling ``arm_id``.
        arm_id : int
            The arm that was pulled to generate ``rwd``.
        """
        if not self.is_initialized:
            self._initialize_params(bandit)

        arm_id = self._select_arm(bandit, context)
        rwd = self._pull_arm(bandit, arm_id, context)
        self._update_params(arm_id, rwd, context)
        return rwd, arm_id

    def reset(self):
        """Reset the policy parameters and counters to their initial states."""
        self.step = 0
        self._reset_params()
        self.is_initialized = False

    def _pull_arm(self, bandit, arm_id, context):
        """Execute a bandit action and return the received reward."""
        self.step += 1
        return bandit.pull(arm_id, context)

    @abstractmethod
    def _select_arm(self, bandit, context):
        """Select an arm based on the current context"""
        pass

    @abstractmethod
    def _update_params(self, bandit, context):
        """Update the policy parameters after an interaction"""
        pass

    @abstractmethod
    def _initialize_params(self, bandit):
        """
        Initialize any policy-specific parameters that depend on information
        from the bandit environment.
        """
        pass

    @abstractmethod
    def _reset_params(self):
        """
        Reset any model-specific parameters. This gets called within the
        public `self.reset()` method.
        """
        pass


class EpsilonGreedy(BanditPolicyBase):
    def __init__(self, epsilon=0.05, ev_prior=0.5):
        r"""
        An epsilon-greedy policy for multi-armed bandit problems.

        Notes
        -----
        Epsilon-greedy policies greedily select the arm with the highest
        expected payoff with probability :math:`1-\epsilon`, and selects an arm
        uniformly at random with probability :math:`\epsilon`:

        .. math::

            P(a) = \left\{
                 \begin{array}{lr}
                   \epsilon / N + (1 - \epsilon) &\text{if }
                        a = \arg \max_{a' \in \mathcal{A}}
                            \mathbb{E}_{q_{\hat{\theta}}}[r \mid a']\\
                   \epsilon / N &\text{otherwise}
                 \end{array}
               \right.

        where :math:`N = |\mathcal{A}|` is the number of arms,
        :math:`q_{\hat{\theta}}` is the estimate of the arm payoff
        distribution under current model parameters :math:`\hat{\theta}`, and
        :math:`\mathbb{E}_{q_{\hat{\theta}}}[r \mid a']` is the expected
        reward under :math:`q_{\hat{\theta}}` of receiving reward `r` after
        taking action :math:`a'`.

        Parameters
        ----------
        epsilon : float in [0, 1]
            The probability of taking a random action. Default is 0.05.
        ev_prior : float
            The starting expected payoff for each arm before any data has been
            observed. Default is 0.5.
        """
        super().__init__()
        self.epsilon = epsilon
        self.ev_prior = ev_prior
        self.pull_counts = defaultdict(lambda: 0)

    @property
    def parameters(self):
        """A dictionary containing the current policy parameters"""
        return {"ev_estimates": self.ev_estimates}

    @property
    def hyperparameters(self):
        """A dictionary containing the policy hyperparameters"""
        return {
            "id": "EpsilonGreedy",
            "epsilon": self.epsilon,
            "ev_prior": self.ev_prior,
        }

    def _initialize_params(self, bandit):
        """
        Initialize any policy-specific parameters that depend on information
        from the bandit environment.
        """
        self.ev_estimates = {i: self.ev_prior for i in range(bandit.n_arms)}
        self.is_initialized = True

    def _select_arm(self, bandit, context=None):
        if np.random.rand() < self.epsilon:
            arm_id = np.random.choice(bandit.n_arms)
        else:
            ests = self.ev_estimates
            (arm_id, _) = max(ests.items(), key=lambda x: x[1])
        return arm_id

    def _update_params(self, arm_id, reward, context=None):
        E, C = self.ev_estimates, self.pull_counts
        C[arm_id] += 1
        E[arm_id] += (reward - E[arm_id]) / (C[arm_id])

    def _reset_params(self):
        """
        Reset any model-specific parameters. This gets called within the
        public `self.reset()` method.
        """
        self.ev_estimates = {}
        self.pull_counts = defaultdict(lambda: 0)


class UCB1(BanditPolicyBase):
    def __init__(self, C=1, ev_prior=0.5):
        r"""
        A UCB1 policy for multi-armed bandit problems.

        Notes
        -----
        The UCB1 algorithm [*]_ guarantees the cumulative regret is bounded by log
        `t`, where `t` is the current timestep. To make this guarantee UCB1
        assumes all arm payoffs are between 0 and 1.

        Under UCB1, the upper confidence bound on the expected value for
        pulling arm `a` at timestep `t` is:

        .. math::

            \text{UCB}(a, t) = \text{EV}_t(a) + C \sqrt{\frac{2 \log t}{N_t(a)}}

        where :math:`\text{EV}_t(a)` is the average of the rewards recieved so
        far from pulling arm `a`, `C` is a free parameter controlling the
        "optimism" of the confidence upper bound for :math:`\text{UCB}(a, t)`
        (for logarithmic regret bounds, `C` must equal 1), and :math:`N_t(a)`
        is the number of times arm `a` has been pulled during the previous `t -
        1` timesteps.

        References
        ----------
        .. [*] Auer, P., Cesa-Bianchi, N., & Fischer, P. (2002). Finite-time
           analysis of the multiarmed bandit problem. *Machine Learning,
           47(2)*.

        Parameters
        ----------
        C : float in (0, +infinity)
            A confidence/optimisim parameter affecting the degree of
            exploration, where larger values encourage greater exploration. The
            UCB1 algorithm assumes `C=1`. Default is 1.
        ev_prior : float
            The starting expected value for each arm before any data has been
            observed. Default is 0.5.
        """
        self.C = C
        self.ev_prior = ev_prior
        super().__init__()

    @property
    def parameters(self):
        """A dictionary containing the current policy parameters"""
        return {"ev_estimates": self.ev_estimates}

    @property
    def hyperparameters(self):
        """A dictionary containing the policy hyperparameters"""
        return {
            "C": self.C,
            "id": "UCB1",
            "ev_prior": self.ev_prior,
        }

    def _initialize_params(self, bandit):
        """
        Initialize any policy-specific parameters that depend on information
        from the bandit environment.
        """
        self.ev_estimates = {i: self.ev_prior for i in range(bandit.n_arms)}
        self.is_initialized = True

    def _select_arm(self, bandit, context=None):
        # add eps to avoid divide-by-zero errors on the first pull of each arm
        eps = np.finfo(float).eps
        N, T = bandit.n_arms, self.step + 1
        E, C = self.ev_estimates, self.pull_counts
        scores = [E[a] + self.C * np.sqrt(np.log(T) / (C[a] + eps)) for a in range(N)]
        return np.argmax(scores)

    def _update_params(self, arm_id, reward, context=None):
        E, C = self.ev_estimates, self.pull_counts
        C[arm_id] += 1
        E[arm_id] += (reward - E[arm_id]) / (C[arm_id])

    def _reset_params(self):
        """
        Reset any model-specific parameters. This gets called within the
        public :method:`reset` method.
        """
        self.ev_estimates = {}
        self.pull_counts = defaultdict(lambda: 0)


class ThompsonSamplingBetaBinomial(BanditPolicyBase):
    def __init__(self, alpha=1, beta=1):
        r"""
        A conjugate Thompson sampling [1]_ [2]_ policy for multi-armed bandits with
        Bernoulli likelihoods.

        Notes
        -----
        The policy assumes independent Beta priors on the Bernoulli arm payoff
        probabilities, :math:`\theta`:

        .. math::

            \theta_k \sim \text{Beta}(\alpha_k, \beta_k) \\
            r \mid \theta_k \sim \text{Bernoulli}(\theta_k)

        where :math:`k \in \{1,\ldots,K \}` indexes arms in the MAB and
        :math:`\theta_k` is the parameter of the Bernoulli likelihood for arm
        `k`. The sampler begins by selecting an arm with probability
        proportional to its payoff probability under the initial Beta prior.
        After pulling the sampled arm and receiving a reward, `r`, the sampler
        computes the posterior over the model parameters (arm payoffs) via
        Bayes' rule, and then samples a new action in proportion to its payoff
        probability under this posterior. This process (i.e., sample action
        from posterior, take action and receive reward, compute updated
        posterior) is repeated until the number of trials is exhausted.

        Note that due to the conjugacy between the Beta prior and Bernoulli
        likelihood the posterior for each arm will also be Beta-distributed and
        can computed and sampled from efficiently:

        .. math::

            \theta_k \mid r \sim \text{Beta}(\alpha_k + r, \beta_k + 1 - r)

        References
        ----------
        .. [1] Thompson, W. (1933). On the likelihood that one unknown
           probability exceeds another in view of the evidence of two samples.
           *Biometrika, 25(3/4)*, 285-294.
        .. [2] Chapelle, O., & Li, L. (2011). An empirical evaluation of
           Thompson sampling. *Advances in Neural Information Processing
           Systems, 24*, 2249-2257.

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
        """A dictionary containing the current policy parameters"""
        return {
            "ev_estimates": self.ev_estimates,
            "alphas": self.alphas,
            "betas": self.betas,
        }

    @property
    def hyperparameters(self):
        """A dictionary containing the policy hyperparameters"""
        return {
            "id": "ThompsonSamplingBetaBinomial",
            "alpha": self.alpha,
            "beta": self.beta,
        }

    def _initialize_params(self, bandit):
        bhp = bandit.hyperparameters
        fstr = "ThompsonSamplingBetaBinomial only defined for BernoulliBandit, got: {}"
        assert bhp["id"] == "BernoulliBandit", fstr.format(bhp["id"])

        # initialize the model prior
        if is_number(self.alpha):
            self.alphas = [self.alpha] * bandit.n_arms
        if is_number(self.beta):
            self.betas = [self.beta] * bandit.n_arms
        assert len(self.alphas) == len(self.betas) == bandit.n_arms

        self.ev_estimates = {i: self._map_estimate(i, 1) for i in range(bandit.n_arms)}
        self.is_initialized = True

    def _select_arm(self, bandit, context):
        if not self.is_initialized:
            self._initialize_prior(bandit)

        # draw a sample from the current model posterior
        posterior_sample = np.random.beta(self.alphas, self.betas)

        # greedily select an action based on this sample
        return np.argmax(posterior_sample)

    def _update_params(self, arm_id, rwd, context):
        """
        Compute the parameters of the Beta posterior, P(payoff prob | rwd),
        for arm `arm_id`.
        """
        self.alphas[arm_id] += rwd
        self.betas[arm_id] += 1 - rwd
        self.ev_estimates[arm_id] = self._map_estimate(arm_id, rwd)

    def _map_estimate(self, arm_id, rwd):
        """Compute the current MAP estimate for an arm's payoff probability"""
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
        """
        Reset any model-specific parameters. This gets called within the
        public `self.reset()` method.
        """
        self.alphas, self.betas = [], []
        self.ev_estimates = {}


class LinUCB(BanditPolicyBase):
    def __init__(self, alpha=1):
        """
        A disjoint linear UCB policy [*]_ for contextual linear bandits.

        Notes
        -----
        LinUCB is only defined for :class:`ContextualLinearBandit <numpy_ml.bandits.ContextualLinearBandit>` environments.

        References
        ----------
        .. [*] Li, L., Chu, W., Langford, J., & Schapire, R. (2010). A
           contextual-bandit approach to personalized news article
           recommendation. In *Proceedings of the 19th International Conference
           on World Wide Web*, 661-670.

        Parameters
        ----------
        alpha : float
            A confidence/optimisim parameter affecting the amount of
            exploration. Default is 1.
        """  # noqa
        super().__init__()

        self.alpha = alpha
        self.A, self.b = [], []
        self.is_initialized = False

    @property
    def parameters(self):
        """A dictionary containing the current policy parameters"""
        return {"ev_estimates": self.ev_estimates, "A": self.A, "b": self.b}

    @property
    def hyperparameters(self):
        """A dictionary containing the policy hyperparameters"""
        return {
            "id": "LinUCB",
            "alpha": self.alpha,
        }

    def _initialize_params(self, bandit):
        """
        Initialize any policy-specific parameters that depend on information
        from the bandit environment.
        """
        bhp = bandit.hyperparameters
        fstr = "LinUCB only defined for contextual linear bandits, got: {}"
        assert bhp["id"] == "ContextualLinearBandit", fstr.format(bhp["id"])

        self.A, self.b = [], []
        for _ in range(bandit.n_arms):
            self.A.append(np.eye(bandit.D))
            self.b.append(np.zeros(bandit.D))

        self.is_initialized = True

    def _select_arm(self, bandit, context):
        probs = []
        for a in range(bandit.n_arms):
            C, A, b = context[:, a], self.A[a], self.b[a]
            A_inv = np.linalg.inv(A)
            theta_hat = A_inv @ b
            p = theta_hat @ C + self.alpha * np.sqrt(C.T @ A_inv @ C)

            probs.append(p)
        return np.argmax(probs)

    def _update_params(self, arm_id, rwd, context):
        """Compute the parameters for A and b."""
        self.A[arm_id] += context[:, arm_id] @ context[:, arm_id].T
        self.b[arm_id] += rwd * context[:, arm_id]

    def _reset_params(self):
        """
        Reset any model-specific parameters. This gets called within the
        public `self.reset()` method.
        """
        self.A, self.b = [], []
        self.ev_estimates = {}
