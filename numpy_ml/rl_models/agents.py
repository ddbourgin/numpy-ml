"""Reinforcement learning agents that can be run on OpenAI gym environs"""

from abc import ABC, abstractmethod
from collections import defaultdict

import numpy as np

from .rl_utils import EnvModel, env_stats, tile_state_space
from ..utils.data_structures import Dict


class AgentBase(ABC):
    def __init__(self, env):
        super().__init__()
        self.env = env
        self.parameters = {}
        self.hyperparameters = {}
        self.derived_variables = {}
        self.env_info = env_stats(env)

    def _create_2num_dicts(self, obs_encoder=None, act_encoder=None):
        E = self.env_info
        n_states = np.prod(E["n_obs_per_dim"])
        n_actions = np.prod(E["n_actions_per_dim"])

        # create action -> scalar dictionaries
        self._num2action = Dict()
        self._action2num = Dict(act_encoder)
        if n_actions != np.inf:
            self._action2num = {act: i for i, act in enumerate(E["action_ids"])}
            self._num2action = {i: act for act, i in self._action2num.items()}

        # create obs -> scalar dictionaries
        self._num2obs = Dict()
        self._obs2num = Dict(obs_encoder)
        if n_states != np.inf:
            self._obs2num = {act: i for i, act in enumerate(E["obs_ids"])}
            self._num2obs = {i: act for act, i in self._obs2num.items()}

    def flush_history(self):
        """Clear the episode history"""
        for k, v in self.episode_history.items():
            self.episode_history[k] = []

    @abstractmethod
    def act(self, obs):
        """Generate an action given the current observation"""
        raise NotImplementedError

    @abstractmethod
    def greedy_policy(self, **kwargs):
        """
        Take a greedy action.

        Returns
        -------
        total_reward : float
            The total reward on the episode.
        n_steps : float
            The total number of steps taken on the episode.
        """
        raise NotImplementedError

    @abstractmethod
    def run_episode(self, max_steps, render=False):
        """
        Run the agent on a single episode.

        Parameters
        ----------
        max_steps : int
            The maximum number of steps to run an episode
        render : bool
            Whether to render the episode during training

        Returns
        -------
        reward : float
            The total reward on the episode, averaged over the theta samples.
        steps : float
            The total number of steps taken on the episode, averaged over the
            theta samples.
        """
        raise NotImplementedError

    @abstractmethod
    def update(self):
        r"""
        Update the agent parameters according to the rewards accrued on the
        current episode.

        Returns
        -------
        avg_reward : float
            The average reward earned by the best `retain_prcnt` theta samples
            on the current episode.
        """
        raise NotImplementedError


class CrossEntropyAgent(AgentBase):
    def __init__(self, env, n_samples_per_episode=500, retain_prcnt=0.2):
        r"""
        A cross-entropy method agent.

        Notes
        -----
        The cross-entropy method [1]_ [2]_ agent only operates on ``envs`` with
        discrete action spaces.

        On each episode the agent generates `n_theta_samples` of the parameters
        (:math:`\theta`) for its behavior policy. The `i`'th sample at
        timestep `t` is:

        .. math::

            \theta_i  &=  \{\mathbf{W}_i^{(t)}, \mathbf{b}_i^{(t)} \} \\
            \theta_i  &\sim  \mathcal{N}(\mu^{(t)}, \Sigma^{(t)})

        Weights (:math:`\mathbf{W}_i`) and bias (:math:`\mathbf{b}_i`) are the
        parameters of the softmax policy:

        .. math::

            \mathbf{z}_i  &=  \text{obs} \cdot \mathbf{W}_i + \mathbf{b}_i \\
            p(a_i^{(t + 1)})  &=  \frac{e^{\mathbf{z}_i}}{\sum_j e^{z_{ij}}} \\
            a^{(t + 1)}  &=  \arg \max_j p(a_j^{(t+1)})

        At the end of each episode, the agent takes the top `retain_prcnt`
        highest scoring :math:`\theta` samples and combines them to generate
        the mean and variance of the distribution of :math:`\theta` for the
        next episode:

        .. math::

            \mu^{(t+1)}  &=  \text{avg}(\texttt{best_thetas}^{(t)}) \\
            \Sigma^{(t+1)}  &=  \text{var}(\texttt{best_thetas}^{(t)})

        References
        ----------
        .. [1] Mannor, S., Rubinstein, R., & Gat, Y. (2003). The cross entropy
           method for fast policy search. In *Proceedings of the 20th Annual
           ICML, 20*.
        .. [2] Rubinstein, R. (1997). optimization of computer simulation
           models with rare events, *European Journal of Operational Research,
           99*, 89–112.

        Parameters
        ----------
        env : :meth:`gym.wrappers` or :meth:`gym.envs` instance
            The environment to run the agent on.
        n_samples_per_episode : int
            The number of theta samples to evaluate on each episode. Default is 500.
        retain_prcnt: float
            The percentage of `n_samples_per_episode` to use when calculating
            the parameter update at the end of the episode. Default is 0.2.
        """
        super().__init__(env)

        self.retain_prcnt = retain_prcnt
        self.n_samples_per_episode = n_samples_per_episode
        self._init_params()

    def _init_params(self):
        E = self.env_info
        assert not E["continuous_actions"], "Action space must be discrete"

        self._create_2num_dicts()
        b_len = np.prod(E["n_actions_per_dim"])
        W_len = b_len * np.prod(E["obs_dim"])
        theta_dim = b_len + W_len

        # init mean and variance for mv gaussian with dimensions theta_dim
        theta_mean = np.random.rand(theta_dim)
        theta_var = np.ones(theta_dim)

        self.parameters = {"theta_mean": theta_mean, "theta_var": theta_var}
        self.derived_variables = {
            "b_len": b_len,
            "W_len": W_len,
            "W_samples": [],
            "b_samples": [],
            "episode_num": 0,
            "cumulative_rewards": [],
        }

        self.hyperparameters = {
            "agent": "CrossEntropyAgent",
            "retain_prcnt": self.retain_prcnt,
            "n_samples_per_episode": self.n_samples_per_episode,
        }

        self.episode_history = {"rewards": [], "state_actions": []}

    def act(self, obs):
        r"""
        Generate actions according to a softmax policy.

        Notes
        -----
        The softmax policy assumes that the pmf over actions in state :math:`x_t` is
        given by:

        .. math::

            \pi(a | x^{(t)}) = \text{softmax}(
                \text{obs}^{(t)} \cdot \mathbf{W}_i^{(t)} + \mathbf{b}_i^{(t)} )

        where :math:`\mathbf{W}` is a learned weight matrix, `obs` is the observation
        at timestep `t`, and **b** is a learned bias vector.

        Parameters
        ----------
        obs : int or :py:class:`ndarray <numpy.ndarray>`
            An observation from the environment.

        Returns
        -------
        action : int, float, or :py:class:`ndarray <numpy.ndarray>`
            An action sampled from the distribution over actions defined by the
            softmax policy.
        """
        E, P = self.env_info, self.parameters
        W, b = P["W"], P["b"]

        s = self._obs2num[obs]
        s = np.array([s]) if E["obs_dim"] == 1 else s

        # compute softmax
        Z = s.T @ W + b
        e_Z = np.exp(Z - np.max(Z, axis=-1, keepdims=True))
        action_probs = e_Z / e_Z.sum(axis=-1, keepdims=True)

        # sample action
        a = np.random.multinomial(1, action_probs).argmax()
        return self._num2action[a]

    def run_episode(self, max_steps, render=False):
        """
        Run the agent on a single episode.

        Parameters
        ----------
        max_steps : int
            The maximum number of steps to run an episode
        render : bool
            Whether to render the episode during training

        Returns
        -------
        reward : float
            The total reward on the episode, averaged over the theta samples.
        steps : float
            The total number of steps taken on the episode, averaged over the
            theta samples.
        """
        self._sample_thetas()

        E, D = self.env_info, self.derived_variables
        n_actions = np.prod(E["n_actions_per_dim"])
        W_len, obs_dim = D["W_len"], E["obs_dim"]
        steps, rewards = [], []

        for theta in D["theta_samples"]:
            W = theta[:W_len].reshape(obs_dim, n_actions)
            b = theta[W_len:]

            total_rwd, n_steps = self._episode(W, b, max_steps, render)
            rewards.append(total_rwd)
            steps.append(n_steps)

        # return the average reward and average number of steps across all
        # samples on the current episode
        D["episode_num"] += 1
        D["cumulative_rewards"] = rewards
        return np.mean(D["cumulative_rewards"]), np.mean(steps)

    def _episode(self, W, b, max_steps, render):
        """
        Run the agent for an episode.

        Parameters
        ----------
        W : :py:class:`ndarray <numpy.ndarray>` of shape `(obs_dim, n_actions)`
            The weights for the softmax policy.
        b : :py:class:`ndarray <numpy.ndarray>` of shape `(bias_len, )`
            The bias for the softmax policy.
        max_steps : int
            The maximum number of steps to run the episode.
        render : bool
            Whether to render the episode during training.

        Returns
        -------
        reward : float
            The total reward on the episode.
        steps : float
            The total number of steps taken on the episode.
        """
        rwds, sa = [], []
        H = self.episode_history
        total_reward, n_steps = 0.0, 1
        obs = self.env.reset()

        self.parameters["W"] = W
        self.parameters["b"] = b

        for i in range(max_steps):
            if render:
                self.env.render()

            n_steps += 1
            action = self.act(obs)
            s, a = self._obs2num[obs], self._action2num[action]
            sa.append((s, a))

            obs, reward, done, _ = self.env.step(action)
            rwds.append(reward)
            total_reward += reward

            if done:
                break

        H["rewards"].append(rwds)
        H["state_actions"].append(sa)
        return total_reward, n_steps

    def update(self):
        r"""
        Update :math:`\mu` and :math:`\Sigma` according to the rewards accrued on
        the current episode.

        Returns
        -------
        avg_reward : float
            The average reward earned by the best `retain_prcnt` theta samples
            on the current episode.
        """
        D, P = self.derived_variables, self.parameters
        n_retain = int(self.retain_prcnt * self.n_samples_per_episode)

        # sort the cumulative rewards for each theta sample from greatest to least
        sorted_y_val_idxs = np.argsort(D["cumulative_rewards"])[::-1]
        top_idxs = sorted_y_val_idxs[:n_retain]

        # update theta_mean and theta_var with the best theta value
        P["theta_mean"] = np.mean(D["theta_samples"][top_idxs], axis=0)
        P["theta_var"] = np.var(D["theta_samples"][top_idxs], axis=0)

    def _sample_thetas(self):
        """
        Sample `n_samples_per_episode` thetas from a multivariate Gaussian with
        mean `theta_mean` and covariance `diag(theta_var)`
        """
        P, N = self.parameters, self.n_samples_per_episode
        Mu, Sigma = P["theta_mean"], np.diag(P["theta_var"])
        samples = np.random.multivariate_normal(Mu, Sigma, N)
        self.derived_variables["theta_samples"] = samples

    def greedy_policy(self, max_steps, render=True):
        """
        Execute a greedy policy using the current agent parameters.

        Parameters
        ----------
        max_steps : int
            The maximum number of steps to run the episode.
        render : bool
            Whether to render the episode during execution.

        Returns
        -------
        total_reward : float
            The total reward on the episode.
        n_steps : float
            The total number of steps taken on the episode.
        """
        E, D, P = self.env_info, self.derived_variables, self.parameters
        Mu, Sigma = P["theta_mean"], np.diag(P["theta_var"])
        sample = np.random.multivariate_normal(Mu, Sigma, 1)

        W_len, obs_dim = D["W_len"], E["obs_dim"]
        n_actions = np.prod(E["n_actions_per_dim"])

        W = sample[0, :W_len].reshape(obs_dim, n_actions)
        b = sample[0, W_len:]
        total_reward, n_steps = self._episode(W, b, max_steps, render)
        return total_reward, n_steps


class MonteCarloAgent(AgentBase):
    def __init__(self, env, off_policy=False, temporal_discount=0.9, epsilon=0.1):
        """
        A Monte-Carlo learning agent trained using either first-visit Monte
        Carlo updates (on-policy) or incremental weighted importance sampling
        (off-policy).

        Parameters
        ----------
        env : :class:`gym.wrappers` or :class:`gym.envs` instance
            The environment to run the agent on.
        off_policy : bool
            Whether to use a behavior policy separate from the target policy
            during training. If False, use the same epsilon-soft policy for
            both behavior and target policies. Default is False.
        temporal_discount : float between [0, 1]
            The discount factor used for downweighting future rewards. Smaller
            values result in greater discounting of future rewards. Default is
            0.9.
        epsilon : float between [0, 1]
            The epsilon value in the epsilon-soft policy. Larger values
            encourage greater exploration during training. Default is 0.1.
        """
        super().__init__(env)

        self.epsilon = epsilon
        self.off_policy = off_policy
        self.temporal_discount = temporal_discount

        self._init_params()

    def _init_params(self):
        E = self.env_info
        assert not E["continuous_actions"], "Action space must be discrete"
        assert not E["continuous_observations"], "Observation space must be discrete"

        n_states = np.prod(E["n_obs_per_dim"])
        n_actions = np.prod(E["n_actions_per_dim"])

        self._create_2num_dicts()

        # behavior policy is stochastic, epsilon-soft policy
        self.behavior_policy = self.target_policy = self._epsilon_soft_policy
        if self.off_policy:
            self.parameters["C"] = np.zeros((n_states, n_actions))

            # target policy is deterministic, greedy policy
            self.target_policy = self._greedy

        # initialize Q function
        self.parameters["Q"] = np.random.rand(n_states, n_actions)

        # initialize returns object for each state-action pair
        self.derived_variables = {
            "returns": {(s, a): [] for s in range(n_states) for a in range(n_actions)},
            "episode_num": 0,
        }

        self.hyperparameters = {
            "agent": "MonteCarloAgent",
            "epsilon": self.epsilon,
            "off_policy": self.off_policy,
            "temporal_discount": self.temporal_discount,
        }

        self.episode_history = {"state_actions": [], "rewards": []}

    def _epsilon_soft_policy(self, s, a=None):
        r"""
        Epsilon-soft exploration policy.

        Notes
        -----
        Soft policies are necessary for first-visit Monte Carlo methods, as
        they require continual exploration (i.e., each state-action pair must
        have nonzero probability of occurring).

        In epsilon-soft policies, :math:`\pi(a \mid s) > 0` for all :math:`s
        \in S` and all :math:`a \in A(s)` at the start of training. As learning
        progresses, :math:`pi` gradually shifts closer and closer to a
        deterministic optimal policy.

        In particular, we have:

        .. math::

            \pi(a \mid s)  &=
                1 - \epsilon + \frac{\epsilon}{|A(s)|}  &&\text{if} a = a^*
            \pi(a \mid s)  &=
                \frac{\epsilon}{|A(s)|}                 &&\text{if} a \neq a^*

        where :math:`|A(s)|` is the number of actions available in state `s`
        and :math:`a^* \in A(s)` is the greedy action in state `s` (i.e.,
        :math:`a^* = \arg \max_a Q(s, a)`).

        Note that epsilon-greedy policies are instances of epsilon-soft
        policies, defined as policies for which :math:`\pi(a|s) \geq \epsilon / |A(s)|`
        for all states and actions.

        Parameters
        ----------
        s : int, float, or tuple
            The state number for the current observation, as returned by
            ``_obs2num[obs]``.
        a : int, float, tuple, or None
            The action number in the current state, as returned by
            ``self._action2num[obs]``. If None, sample an action from the
            action probabilities in state `s`, otherwise, return the
            probability of action `a` under the epsilon-soft policy. Default is
            None.

        Returns
        -------
        action : int, float, or :py:class:`ndarray <numpy.ndarray>`
            If `a` is None, this is an action sampled from the distribution
            over actions defined by the epsilon-soft policy. If `a` is not
            None, this is the probability of `a` under the epsilon-soft policy.
        """
        E, P = self.env_info, self.parameters

        # TODO: this assumes all actions are available in every state
        n_actions = np.prod(E["n_actions_per_dim"])

        a_star = P["Q"][s, :].argmax()
        p_a_star = 1.0 - self.epsilon + (self.epsilon / n_actions)
        p_a = self.epsilon / n_actions

        action_probs = np.ones(n_actions) * p_a
        action_probs[a_star] = p_a_star
        np.testing.assert_allclose(np.sum(action_probs), 1)

        if a is not None:
            return action_probs[a]

        # sample action
        a = np.random.multinomial(1, action_probs).argmax()
        return self._num2action[a]

    def _greedy(self, s, a=None):
        """
        A greedy behavior policy.

        Notes
        -----
        Only used when off-policy is True.

        Parameters
        ----------
        s : int, float, or tuple
            The state number for the current observation, as returned by
            ``self._obs2num[obs]``.
        a : int, float, or tuple
            The action number in the current state, as returned by
            ``self._action2num[obs]``. If None, sample an action from the action
            probabilities in state `s`, otherwise, return the probability of
            action `a` under the greedy policy. Default is None.

        Returns
        -------
        action : int, float, or :py:class:`ndarray <numpy.ndarray>`
            If `a` is None, this is an action sampled from the distribution
            over actions defined by the greedy policy. If `a` is not
            None, this is the probability of `a` under the greedy policy.
        """
        a_star = self.parameters["Q"][s, :].argmax()
        if a is None:
            out = self._num2action[a_star]
        else:
            out = 1 if a == a_star else 0
        return out

    def _on_policy_update(self):
        r"""
        Update the `Q` function using an on-policy first-visit Monte Carlo
        update.

        Notes
        -----
        The on-policy first-visit Monte Carlo update is

        .. math::

            Q'(s, a) \leftarrow
                \text{avg}(\text{reward following first visit to } (s, a)
                \text{ across all episodes})

        RL agents seek to learn action values conditional on subsequent optimal
        behavior, but they need to behave non-optimally in order to explore all
        actions (to find the optimal actions).

        The on-policy approach is a compromise -- it learns action values not
        for the optimal policy, but for a *near*-optimal policy that still
        explores (the epsilon-soft policy).
        """
        D, P, HS = self.derived_variables, self.parameters, self.episode_history

        ep_rewards = HS["rewards"]
        sa_tuples = set(HS["state_actions"])

        locs = [HS["state_actions"].index(sa) for sa in sa_tuples]
        cumulative_returns = [np.sum(ep_rewards[i:]) for i in locs]

        # update Q value with the average of the first-visit return across
        # episodes
        for (s, a), cr in zip(sa_tuples, cumulative_returns):
            D["returns"][(s, a)].append(cr)
            P["Q"][s, a] = np.mean(D["returns"][(s, a)])

    def _off_policy_update(self):
        """
        Update `Q` using weighted importance sampling.

        Notes
        -----
        In importance sampling updates, we account for the fact that we are
        updating a different policy from the one we used to generate behavior
        by weighting the accumulated rewards by the ratio of the probability of
        the trajectory under the target policy versus its probability under
        the behavior policies. This is known as the importance sampling weight.

        In weighted importance sampling, we scale the accumulated rewards for a
        trajectory by their importance sampling weight, then take the
        *weighted* average using the importance sampling weight. This weighted
        average then becomes the value for the trajectory.

            W   = importance sampling weight
            G_t = total discounted reward from time t until episode end
            C_n = sum of importance weights for the first n rewards

        This algorithm converges to Q* in the limit.
        """
        P = self.parameters
        HS = self.episode_history
        ep_rewards = HS["rewards"]
        T = len(ep_rewards)

        G, W = 0.0, 1.0
        for t in reversed(range(T)):
            s, a = HS["state_actions"][t]
            G = self.temporal_discount * G + ep_rewards[t]
            P["C"][s, a] += W

            # update Q(s, a) using weighted importance sampling
            P["Q"][s, a] += (W / P["C"][s, a]) * (G - P["Q"][s, a])

            # multiply the importance sampling ratio by the current weight
            W *= self.target_policy(s, a) / self.behavior_policy(s, a)

            if W == 0.0:
                break

    def act(self, obs):
        r"""
        Execute the behavior policy--an :math:`\epsilon`-soft policy used to
        generate actions during training.

        Parameters
        ----------
        obs : int, float, or :py:class:`ndarray <numpy.ndarray>` as returned by ``env.step(action)``
            An observation from the environment.

        Returns
        -------
        action : int, float, or :py:class:`ndarray <numpy.ndarray>`
            An action sampled from the distribution over actions defined by the
            epsilon-soft policy.
        """  # noqa: E501
        s = self._obs2num[obs]
        return self.behavior_policy(s)

    def run_episode(self, max_steps, render=False):
        """
        Run the agent on a single episode.

        Parameters
        ----------
        max_steps : int
            The maximum number of steps to run an episode.
        render : bool
            Whether to render the episode during training.

        Returns
        -------
        reward : float
            The total reward on the episode.
        steps : float
            The number of steps taken on the episode.
        """
        D = self.derived_variables
        total_rwd, n_steps = self._episode(max_steps, render)

        D["episode_num"] += 1
        return total_rwd, n_steps

    def _episode(self, max_steps, render):
        """
        Execute agent on an episode.

        Parameters
        ----------
        max_steps : int
            The maximum number of steps to run the episode.
        render : bool
            Whether to render the episode during training.

        Returns
        -------
        reward : float
            The total reward on the episode.
        steps : float
            The number of steps taken on the episode.
        """
        obs = self.env.reset()
        HS = self.episode_history
        total_reward, n_steps = 0.0, 0

        for i in range(max_steps):
            if render:
                self.env.render()

            n_steps += 1
            action = self.act(obs)

            s = self._obs2num[obs]
            a = self._action2num[action]

            # store (state, action) tuple
            HS["state_actions"].append((s, a))

            # take action
            obs, reward, done, info = self.env.step(action)

            # record rewards
            HS["rewards"].append(reward)
            total_reward += reward

            if done:
                break

        return total_reward, n_steps

    def update(self):
        """
        Update the parameters of the model following the completion of an
        episode. Flush the episode history after the update is complete.
        """
        H = self.hyperparameters
        if H["off_policy"]:
            self._off_policy_update()
        else:
            self._on_policy_update()

        self.flush_history()

    def greedy_policy(self, max_steps, render=True):
        """
        Execute a greedy policy using the current agent parameters.

        Parameters
        ----------
        max_steps : int
            The maximum number of steps to run the episode.
        render : bool
            Whether to render the episode during execution.

        Returns
        -------
        total_reward : float
            The total reward on the episode.
        n_steps : float
            The total number of steps taken on the episode.
        """
        H = self.episode_history
        obs = self.env.reset()

        total_reward, n_steps = 0.0, 0
        for i in range(max_steps):
            if render:
                self.env.render()

            n_steps += 1
            action = self._greedy(obs)

            s = self._obs2num[obs]
            a = self._action2num[action]

            # store (state, action) tuple
            H["state_actions"].append((s, a))

            # take action
            obs, reward, done, info = self.env.step(action)

            # record rewards
            H["rewards"].append(reward)
            total_reward += reward

            if done:
                break

        return total_reward, n_steps


class TemporalDifferenceAgent(AgentBase):
    def __init__(
        self,
        env,
        lr=0.4,
        epsilon=0.1,
        n_tilings=8,
        obs_max=None,
        obs_min=None,
        grid_dims=[8, 8],
        off_policy=False,
        temporal_discount=0.99,
    ):
        r"""
        A temporal difference learning agent with expected SARSA (on-policy) [3]_ or
        TD(0) `Q`-learning (off-policy) [4]_ updates.

        Notes
        -----
        The expected SARSA on-policy TD(0) update is:

        .. math::

            Q(s, a) \leftarrow Q(s, a) + \eta \left(
                r + \gamma \mathbb{E}_\pi[Q(s', a') \mid s'] - Q(s, a)
            \right)

        and the TD(0) off-policy Q-learning upate is:

        .. math::

            Q(s, a) \leftarrow Q(s, a) + \eta (
                r + \gamma \max_a \left\{ Q(s', a) \right\} - Q(s, a)
            )

        where in each case we have taken action `a` in state `s`, received
        reward `r`, and transitioned into state :math:`s'`. In the above
        equations, :math:`\eta` is a learning rate parameter, :math:`\gamma` is
        a temporal discount factor, and :math:`\mathbb{E}_\pi[ Q[s', a'] \mid
        s']` is the expected value under the current policy :math:`\pi` of the
        Q function conditioned that we are in state :math:`s'`.

        Observe that the expected SARSA update can be used for both on- and
        off-policy methods. In an off-policy context, if the target policy is
        greedy and the expectation is taken wrt. the target policy then the
        expected SARSA update is exactly Q-learning.

        NB. For this implementation the agent requires a discrete action
        space, but will try to discretize the observation space via tiling if
        it is continuous.

        References
        ----------
        .. [3] Rummery, G. & Niranjan, M. (1994). *On-Line Q-learning Using
           Connectionist Systems*. Tech Report 166. Cambridge University
           Department of Engineering.
        .. [4] Watkins, C. (1989). Learning from delayed rewards. *PhD thesis,
           Cambridge University*.

        Parameters
        ----------
        env : gym.wrappers or gym.envs instance
            The environment to run the agent on.
        lr : float
            Learning rate for the Q function updates. Default is 0.05.
        epsilon : float between [0, 1]
            The epsilon value in the epsilon-soft policy. Larger values
            encourage greater exploration during training. Default is 0.1.
        n_tilings : int
            The number of overlapping tilings to use if the ``env`` observation
            space is continuous. Unused if observation space is discrete.
            Default is 8.
        obs_max : float or :py:class:`ndarray <numpy.ndarray>`
            The value to treat as the max value of the observation space when
            calculating the grid widths if the observation space is continuous.
            If None, use ``env.observation_space.high``. Unused if observation
            space is discrete. Default is None.
        obs_min : float or :py:class:`ndarray <numpy.ndarray>`
            The value to treat as the min value of the observation space when
            calculating grid widths if the observation space is continuous. If
            None, use ``env.observation_space.low``. Unused if observation
            space is discrete. Default is None.
        grid_dims : list
           The number of rows and columns in each tiling grid if the env
           observation space is continuous. Unused if observation space is
           discrete. Default is [8, 8].
        off_policy : bool
            Whether to use a behavior policy separate from the target policy
            during training. If False, use the same epsilon-soft policy for
            both behavior and target policies. Default is False.
        temporal_discount : float between [0, 1]
            The discount factor used for downweighting future rewards. Smaller
            values result in greater discounting of future rewards. Default is
            0.9.
        """
        super().__init__(env)

        self.lr = lr
        self.obs_max = obs_max
        self.obs_min = obs_min
        self.epsilon = epsilon
        self.n_tilings = n_tilings
        self.grid_dims = grid_dims
        self.off_policy = off_policy
        self.temporal_discount = temporal_discount

        self._init_params()

    def _init_params(self):
        E = self.env_info
        assert not E["continuous_actions"], "Action space must be discrete"

        obs_encoder = None
        if E["continuous_observations"]:
            obs_encoder, _ = tile_state_space(
                self.env,
                self.env_info,
                self.n_tilings,
                state_action=False,
                obs_max=self.obs_max,
                obs_min=self.obs_min,
                grid_size=self.grid_dims,
            )

        self._create_2num_dicts(obs_encoder=obs_encoder)

        # behavior policy is stochastic, epsilon-soft policy
        self.behavior_policy = self.target_policy = self._epsilon_soft_policy
        if self.off_policy:
            # target policy is deterministic, greedy policy
            self.target_policy = self._greedy

        # initialize Q function
        self.parameters["Q"] = defaultdict(np.random.rand)

        # initialize returns object for each state-action pair
        self.derived_variables = {"episode_num": 0}

        self.hyperparameters = {
            "agent": "TemporalDifferenceAgent",
            "lr": self.lr,
            "obs_max": self.obs_max,
            "obs_min": self.obs_min,
            "epsilon": self.epsilon,
            "n_tilings": self.n_tilings,
            "grid_dims": self.grid_dims,
            "off_policy": self.off_policy,
            "temporal_discount": self.temporal_discount,
        }

        self.episode_history = {"state_actions": [], "rewards": []}

    def run_episode(self, max_steps, render=False):
        """
        Run the agent on a single episode without updating the priority queue
        or performing backups.

        Parameters
        ----------
        max_steps : int
            The maximum number of steps to run an episode
        render : bool
            Whether to render the episode during training

        Returns
        -------
        reward : float
            The total reward on the episode, averaged over the theta samples.
        steps : float
            The total number of steps taken on the episode, averaged over the
            theta samples.
        """
        return self._episode(max_steps, render, update=False)

    def train_episode(self, max_steps, render=False):
        """
        Train the agent on a single episode.

        Parameters
        ----------
        max_steps : int
            The maximum number of steps to run an episode.
        render : bool
            Whether to render the episode during training.

        Returns
        -------
        reward : float
            The total reward on the episode.
        steps : float
            The number of steps taken on the episode.
        """
        D = self.derived_variables
        total_rwd, n_steps = self._episode(max_steps, render, update=True)

        D["episode_num"] += 1

        return total_rwd, n_steps

    def _episode(self, max_steps, render, update=True):
        """
        Run or train the agent on an episode.

        Parameters
        ----------
        max_steps : int
            The maximum number of steps to run the episode.
        render : bool
            Whether to render the episode during training.
        update : bool
            Whether to perform the Q function backups after each step. Default
            is True.

        Returns
        -------
        reward : float
            The total reward on the episode.
        steps : float
            The number of steps taken on the episode.
        """
        self.flush_history()

        obs = self.env.reset()
        HS = self.episode_history

        action = self.act(obs)
        s = self._obs2num[obs]
        a = self._action2num[action]

        # store initial (state, action) tuple
        HS["state_actions"].append((s, a))

        total_reward, n_steps = 0.0, 0
        for i in range(max_steps):
            if render:
                self.env.render()

            # take action
            obs, reward, done, info = self.env.step(action)
            n_steps += 1

            # record rewards
            HS["rewards"].append(reward)
            total_reward += reward

            # generate next state and action
            action = self.act(obs)
            s_ = self._obs2num[obs] if not done else None
            a_ = self._action2num[action]

            # store next (state, action) tuple
            HS["state_actions"].append((s_, a_))

            if update:
                self.update()

            if done:
                break

        return total_reward, n_steps

    def _epsilon_soft_policy(self, s, a=None):
        r"""
        Epsilon-soft exploration policy.

        In epsilon-soft policies, :math:`\pi(a|s) > 0` for all s ∈ S and all a
        ∈ A(s) at the start of training. As learning progresses, :math:`\pi`
        gradually shifts closer and closer to a deterministic optimal policy.

        In particular, we have:

            pi(a|s) = 1 - epsilon + (epsilon / |A(s)|) IFF a == a*
            pi(a|s) = epsilon / |A(s)|                 IFF a != a*

        where

            |A(s)| is the number of actions available in state s
            a* ∈ A(s) is the greedy action in state s (i.e., a* = argmax_a Q(s, a))

        Note that epsilon-greedy policies are instances of epsilon-soft
        policies, defined as policies for which pi(a|s) >= epsilon / |A(s)| for
        all states and actions.

        Parameters
        ----------
        s : int, float, or tuple
            The state number for the current observation, as returned by
            ``self._obs2num[obs]``
        a : int, float, or tuple
            The action number in the current state, as returned by
            self._action2num[obs]. If None, sample an action from the action
            probabilities in state s, otherwise, return the probability of
            action `a` under the epsilon-soft policy. Default is None.

        Returns
        -------
        If `a` is None:
        action : int, float, or :py:class:`ndarray <numpy.ndarray>` as returned by `self._num2action`
            If `a` is None, returns an action sampled from the distribution
            over actions defined by the epsilon-soft policy.

        If `a` is not None:
        action_prob : float in range [0, 1]
            If `a` is not None, returns the probability of `a` under the
            epsilon-soft policy.
        """  # noqa: E501
        E, P = self.env_info, self.parameters

        # TODO: this assumes all actions are available in every state
        n_actions = np.prod(E["n_actions_per_dim"])

        a_star = np.argmax([P["Q"][(s, aa)] for aa in range(n_actions)])
        p_a_star = 1.0 - self.epsilon + (self.epsilon / n_actions)
        p_a = self.epsilon / n_actions

        action_probs = np.ones(n_actions) * p_a
        action_probs[a_star] = p_a_star
        np.testing.assert_allclose(np.sum(action_probs), 1)

        if a is not None:
            return action_probs[a]

        # sample action
        a = np.random.multinomial(1, action_probs).argmax()
        return self._num2action[a]

    def _greedy(self, s, a=None):
        """
        A greedy behavior policy. Only used when off-policy is true.

        Parameters
        ----------
        s : int, float, or tuple
            The state number for the current observation, as returned by
            ``self._obs2num[obs]``
        a : int, float, or tuple
            The action number in the current state, as returned by
            ``self._action2num[obs]``. If None, sample an action from the
            action probabilities in state `s`, otherwise, return the
            probability of action `a` under the greedy policy. Default is None.

        Returns
        -------
        If `a` is None:
        action : int, float, or :py:class:`ndarray <numpy.ndarray>` as returned by ``self._num2action``
            If `a` is None, returns an action sampled from the distribution
            over actions defined by the greedy policy.

        If `a` is not None:
        action_prob : float in range [0, 1]
            If `a` is not None, returns the probability of `a` under the
            greedy policy.
        """  # noqa: E501
        P, E = self.parameters, self.env_info
        n_actions = np.prod(E["n_actions_per_dim"])
        a_star = np.argmax([P["Q"][(s, aa)] for aa in range(n_actions)])
        if a is None:
            out = self._num2action[a_star]
        else:
            out = 1 if a == a_star else 0
        return out

    def _on_policy_update(self, s, a, r, s_, a_):
        """
        Update the Q function using the expected SARSA on-policy TD(0) update:

            Q[s, a] <- Q[s, a] + lr * [
                r + temporal_discount * E[Q[s', a'] | s'] - Q[s, a]
            ]

        where

            E[ Q[s', a'] | s'] is the expected value of the Q function over all
            a_ given that we're in state s' under the current policy

        NB. the expected SARSA update can be used for both on- and off-policy
        methods. In an off-policy context, if the target policy is greedy and
        the expectation is taken wrt. the target policy then the expected SARSA
        update is exactly Q-learning.

        Parameters
        ----------
        s : int as returned by `self._obs2num`
            The id for the state/observation at timestep t-1
        a : int as returned by `self._action2num`
            The id for the action taken at timestep t-1
        r : float
            The reward after taking action `a` in state `s` at timestep t-1
        s_ : int as returned by `self._obs2num`
            The id for the state/observation at timestep t
        a_ : int as returned by `self._action2num`
            The id for the action taken at timestep t
        """
        Q, E, pi = self.parameters["Q"], self.env_info, self.behavior_policy

        # TODO: this assumes that all actions are available in each state
        n_actions = np.prod(E["n_actions_per_dim"])

        # compute the expected value of Q(s', a') given that we are in state s'
        E_Q = np.sum([pi(s_, aa) * Q[(s_, aa)] for aa in range(n_actions)]) if s_ else 0

        # perform the expected SARSA TD(0) update
        qsa = Q[(s, a)]
        Q[(s, a)] = qsa + self.lr * (r + self.temporal_discount * E_Q - qsa)

    def _off_policy_update(self, s, a, r, s_):
        """
        Update the `Q` function using the TD(0) Q-learning update:

            Q[s, a] <- Q[s, a] + lr * (
                r + temporal_discount * max_a { Q[s', a] } - Q[s, a]
            )

        Parameters
        ----------
        s : int as returned by `self._obs2num`
            The id for the state/observation at timestep `t-1`
        a : int as returned by `self._action2num`
            The id for the action taken at timestep `t-1`
        r : float
            The reward after taking action `a` in state `s` at timestep `t-1`
        s_ : int as returned by `self._obs2num`
            The id for the state/observation at timestep `t`
        """
        Q, E = self.parameters["Q"], self.env_info
        n_actions = np.prod(E["n_actions_per_dim"])

        qsa = Q[(s, a)]
        Qs_ = [Q[(s_, aa)] for aa in range(n_actions)] if s_ else [0]
        Q[(s, a)] = qsa + self.lr * (r + self.temporal_discount * np.max(Qs_) - qsa)

    def update(self):
        """Update the parameters of the model online after each new state-action."""
        H, HS = self.hyperparameters, self.episode_history
        (s, a), r = HS["state_actions"][-2], HS["rewards"][-1]
        s_, a_ = HS["state_actions"][-1]

        if H["off_policy"]:
            self._off_policy_update(s, a, r, s_)
        else:
            self._on_policy_update(s, a, r, s_, a_)

    def act(self, obs):
        r"""
        Execute the behavior policy--an :math:`\epsilon`-soft policy used to
        generate actions during training.

        Parameters
        ----------
        obs : int, float, or :py:class:`ndarray <numpy.ndarray>` as returned by ``env.step(action)``
            An observation from the environment.

        Returns
        -------
        action : int, float, or :py:class:`ndarray <numpy.ndarray>`
            An action sampled from the distribution over actions defined by the
            epsilon-soft policy.
        """  # noqa: E501
        s = self._obs2num[obs]
        return self.behavior_policy(s)

    def greedy_policy(self, max_steps, render=True):
        """
        Execute a deterministic greedy policy using the current agent
        parameters.

        Parameters
        ----------
        max_steps : int
            The maximum number of steps to run the episode.
        render : bool
            Whether to render the episode during execution.

        Returns
        -------
        total_reward : float
            The total reward on the episode.
        n_steps : float
            The total number of steps taken on the episode.
        """
        self.flush_history()

        H = self.episode_history
        obs = self.env.reset()

        total_reward, n_steps = 0.0, 0
        for i in range(max_steps):
            if render:
                self.env.render()

            s = self._obs2num[obs]
            action = self._greedy(s)
            a = self._action2num[action]

            # store (state, action) tuple
            H["state_actions"].append((s, a))

            # take action
            obs, reward, done, info = self.env.step(action)
            n_steps += 1

            # record rewards
            H["rewards"].append(reward)
            total_reward += reward

            if done:
                break

        return total_reward, n_steps


class DynaAgent(AgentBase):
    def __init__(
        self,
        env,
        lr=0.4,
        epsilon=0.1,
        n_tilings=8,
        obs_max=None,
        obs_min=None,
        q_plus=False,
        grid_dims=[8, 8],
        explore_weight=0.05,
        temporal_discount=0.9,
        n_simulated_actions=50,
    ):
        r"""
        A Dyna-`Q` / Dyna-`Q+` agent [5]_ with full TD(0) `Q`-learning updates via
        prioritized-sweeping [6]_ .

        Notes
        -----
        This approach consists of three components: a planning method involving
        simulated actions, a direct RL method where the agent directly interacts
        with the environment, and a model-learning method where the agent
        learns to better represent the environment during planning.

        During planning, the agent performs random-sample one-step tabular
        Q-planning with prioritized sweeping. This entails using a priority
        queue to retrieve the state-action pairs from the agent's history which
        would stand to have the largest change to their Q-values if backed up.
        Specifically, for state action pair `(s, a)` the priority value is:

        .. math::

            P = \sum_{s'} p(s') | r + \gamma \max_a \{Q(s', a) \} - Q(s, a) |

        which corresponds to the absolute magnitude of the TD(0) Q-learning
        backup for the pair.

        When the first pair in the queue is backed up, the effect on each of
        its predecessor pairs is computed. If the predecessor's priority is
        greater than a small threshold the pair is added to the queue and the
        process is repeated until either the queue is empty or we have exceeded
        `n_simulated_actions` updates. These backups occur without the agent
        taking any action in the environment and thus constitute simulations
        based on the agent's current model of the environment (i.e., its
        tabular state-action history).

        During the direct RL phase, the agent takes an action based on its
        current behavior policy and Q function and receives a reward from the
        environment. The agent logs this state-action-reward-new state tuple in
        its interaction table (i.e., environment model) and updates its Q
        function using a full-backup version of the Q-learning update:

        .. math::

            Q(s, a) \leftarrow Q(s, a) + \eta \sum_{r, s'} p(r, s' \mid s, a)
                \left(r + \gamma \max_a \{ Q(s', a) \} - Q(s, a) \right)

        References
        ----------
        .. [5] Sutton, R. (1990). Integrated architectures for learning,
           planning, and reacting based on approximating dynamic programming.
           In *Proceedings of the 7th Annual ICML*, 216-224.
        .. [6] Moore, A. & Atkeson, C. (1993). Prioritized sweeping:
           Reinforcement learning with less data and less time. *Machine
           Learning, 13(1)*, 103-130.

        Parameters
        ----------
        env : :class:`gym.wrappers` or :class:`gym.envs` instance
            The environment to run the agent on
        lr : float
            Learning rate for the `Q` function updates. Default is 0.05.
        epsilon : float between [0, 1]
            The epsilon value in the epsilon-soft policy. Larger values
            encourage greater exploration during training. Default is 0.1.
        n_tilings : int
            The number of overlapping tilings to use if the env observation
            space is continuous. Unused if observation space is discrete.
            Default is 8.
        obs_max : float or :py:class:`ndarray <numpy.ndarray>` or None
            The value to treat as the max value of the observation space when
            calculating the grid widths if the observation space is continuous.
            If None, use :meth:`env.observation_space.high`. Unused if observation
            space is discrete. Default is None.
        obs_min : float or :py:class:`ndarray <numpy.ndarray>` or None
            The value to treat as the min value of the observation space when
            calculating grid widths if the observation space is continuous. If
            None, use :meth:`env.observation_space.low`. Unused if observation
            space is discrete. Default is None.
        grid_dims : list
            The number of rows and columns in each tiling grid if the env
            observation space is continuous. Unused if observation space is
            discrete. Default is `[8, 8]`.
        q_plus : bool
            Whether to add incentives for visiting states that the agent hasn't
            encountered recently. Default is False.
        explore_weight : float
            Amount to incentivize exploring states that the agent hasn't
            recently visited. Only used if `q_plus` is True. Default is 0.05.
        temporal_discount : float between [0, 1]
            The discount factor used for downweighting future rewards. Smaller
            values result in greater discounting of future rewards. Default is
            0.9.
        n_simulated_actions : int
            THe number of simulated actions to perform for each "real" action.
            Default is 50.
        """
        super().__init__(env)

        self.lr = lr
        self.q_plus = q_plus
        self.obs_max = obs_max
        self.obs_min = obs_min
        self.epsilon = epsilon
        self.n_tilings = n_tilings
        self.grid_dims = grid_dims
        self.explore_weight = explore_weight
        self.temporal_discount = temporal_discount
        self.n_simulated_actions = n_simulated_actions

        self._init_params()

    def _init_params(self):
        E = self.env_info
        assert not E["continuous_actions"], "Action space must be discrete"

        obs_encoder = None
        if E["continuous_observations"]:
            obs_encoder, _ = tile_state_space(
                self.env,
                self.env_info,
                self.n_tilings,
                state_action=False,
                obs_max=self.obs_max,
                obs_min=self.obs_min,
                grid_size=self.grid_dims,
            )

        self._create_2num_dicts(obs_encoder=obs_encoder)
        self.behavior_policy = self.target_policy = self._epsilon_soft_policy

        # initialize Q function and model
        self.parameters["Q"] = defaultdict(np.random.rand)
        self.parameters["model"] = EnvModel()

        # initialize returns object for each state-action pair
        self.derived_variables = {
            "episode_num": 0,
            "sweep_queue": {},
            "visited": set(),
            "steps_since_last_visit": defaultdict(lambda: 0),
        }

        if self.q_plus:
            self.derived_variables["steps_since_last_visit"] = defaultdict(
                np.random.rand,
            )

        self.hyperparameters = {
            "agent": "DynaAgent",
            "lr": self.lr,
            "q_plus": self.q_plus,
            "obs_max": self.obs_max,
            "obs_min": self.obs_min,
            "epsilon": self.epsilon,
            "n_tilings": self.n_tilings,
            "grid_dims": self.grid_dims,
            "explore_weight": self.explore_weight,
            "temporal_discount": self.temporal_discount,
            "n_simulated_actions": self.n_simulated_actions,
        }

        self.episode_history = {"state_actions": [], "rewards": []}

    def act(self, obs):
        r"""
        Execute the behavior policy--an :math:`\epsilon`-soft policy used to
        generate actions during training.

        Parameters
        ----------
        obs : int, float, or :py:class:`ndarray <numpy.ndarray>` as returned by ``env.step(action)``
            An observation from the environment.

        Returns
        -------
        action : int, float, or :py:class:`ndarray <numpy.ndarray>`
            An action sampled from the distribution over actions defined by the
            epsilon-soft policy.
        """  # noqa: E501
        s = self._obs2num[obs]
        return self.behavior_policy(s)

    def _epsilon_soft_policy(self, s, a=None):
        """
        Epsilon-soft exploration policy.

        In epsilon-soft policies, pi(a|s) > 0 for all s ∈ S and all a ∈ A(s) at
        the start of training. As learning progresses, pi gradually shifts
        closer and closer to a deterministic optimal policy.

        In particular, we have:

            pi(a|s) = 1 - epsilon + (epsilon / |A(s)|) IFF a == a*
            pi(a|s) = epsilon / |A(s)|                 IFF a != a*

        where

            |A(s)| is the number of actions available in state s
            a* ∈ A(s) is the greedy action in state s (i.e., a* = argmax_a Q(s, a))

        Note that epsilon-greedy policies are instances of epsilon-soft
        policies, defined as policies for which pi(a|s) >= epsilon / |A(s)| for
        all states and actions.

        Parameters
        ----------
        s : int, float, or tuple
            The state number for the current observation, as returned by
            self._obs2num[obs]
        a : int, float, or tuple
            The action number in the current state, as returned by
            self._action2num[obs]. If None, sample an action from the action
            probabilities in state s, otherwise, return the probability of
            action `a` under the epsilon-soft policy. Default is None.

        Returns
        -------
        If `a` is None:
        action : int, float, or :py:class:`ndarray <numpy.ndarray>` as returned by :meth:`_num2action`
            If `a` is None, returns an action sampled from the distribution
            over actions defined by the epsilon-soft policy.

        If `a` is not None:
        action_prob : float in range [0, 1]
            If `a` is not None, returns the probability of `a` under the
            epsilon-soft policy.
        """  # noqa: E501
        E, P = self.env_info, self.parameters

        # TODO: this assumes all actions are available in every state
        n_actions = np.prod(E["n_actions_per_dim"])

        a_star = np.argmax([P["Q"][(s, aa)] for aa in range(n_actions)])
        p_a_star = 1.0 - self.epsilon + (self.epsilon / n_actions)
        p_a = self.epsilon / n_actions

        action_probs = np.ones(n_actions) * p_a
        action_probs[a_star] = p_a_star
        np.testing.assert_allclose(np.sum(action_probs), 1)

        if a is not None:
            return action_probs[a]

        # sample action
        a = np.random.multinomial(1, action_probs).argmax()
        return self._num2action[a]

    def _greedy(self, s, a=None):
        """
        A greedy behavior policy.

        Parameters
        ----------
        s : int, float, or tuple
            The state number for the current observation, as returned by
            self._obs2num[obs]
        a : int, float, or tuple
            The action number in the current state, as returned by
            self._action2num[obs]. If None, sample an action from the action
            probabilities in state s, otherwise, return the probability of
            action `a` under the greedy policy. Default is None.

        Returns
        -------
        If `a` is None:
        action : int, float, or :py:class:`ndarray <numpy.ndarray>` as returned by :meth:`_num2action`
            If `a` is None, returns an action sampled from the distribution
            over actions defined by the greedy policy.

        If `a` is not None:
        action_prob : float in range [0, 1]
            If `a` is not None, returns the probability of `a` under the
            greedy policy.
        """  # noqa: E501
        E, Q = self.env_info, self.parameters["Q"]
        n_actions = np.prod(E["n_actions_per_dim"])
        a_star = np.argmax([Q[(s, aa)] for aa in range(n_actions)])
        if a is None:
            out = self._num2action[a_star]
        else:
            out = 1 if a == a_star else 0
        return out

    def update(self):
        """
        Update the priority queue with the most recent (state, action) pair and
        perform random-sample one-step tabular Q-planning.

        Notes
        -----
        The planning algorithm uses a priority queue to retrieve the
        state-action pairs from the agent's history which will result in the
        largest change to its `Q`-value if backed up. When the first pair in
        the queue is backed up, the effect on each of its predecessor pairs is
        computed. If the predecessor's priority is greater than a small
        threshold the pair is added to the queue and the process is repeated
        until either the queue is empty or we exceed `n_simulated_actions`
        updates.
        """
        s, a = self.episode_history["state_actions"][-1]
        self._update_queue(s, a)
        self._simulate_behavior()

    def _update_queue(self, s, a):
        """
        Update the priority queue by calculating the priority for (s, a) and
        inserting it into the queue if it exceeds a fixed (small) threshold.

        Parameters
        ----------
        s : int as returned by `self._obs2num`
            The id for the state/observation
        a : int as returned by `self._action2num`
            The id for the action taken from state `s`
        """
        sweep_queue = self.derived_variables["sweep_queue"]

        # TODO: what's a good threshold here?
        priority = self._calc_priority(s, a)
        if priority >= 0.001:
            if (s, a) in sweep_queue:
                sweep_queue[(s, a)] = max(priority, sweep_queue[(s, a)])
            else:
                sweep_queue[(s, a)] = priority

    def _calc_priority(self, s, a):
        """
        Compute the "priority" for state-action pair (s, a). The priority P is
        defined as:

            P = sum_{s_} p(s_) * abs(r + temporal_discount * max_a {Q[s_, a]} - Q[s, a])

        which corresponds to the absolute magnitude of the TD(0) Q-learning
        backup for (s, a).

        Parameters
        ----------
        s : int as returned by `self._obs2num`
            The id for the state/observation
        a : int as returned by `self._action2num`
            The id for the action taken from state `s`

        Returns
        -------
        priority : float
            The absolute magnitude of the full-backup TD(0) Q-learning update
            for (s, a)
        """
        priority = 0.0
        E = self.env_info
        Q = self.parameters["Q"]
        env_model = self.parameters["model"]
        n_actions = np.prod(E["n_actions_per_dim"])

        outcome_probs = env_model.outcome_probs(s, a)
        for (r, s_), p_rs_ in outcome_probs:
            max_q = np.max([Q[(s_, aa)] for aa in range(n_actions)])
            P = p_rs_ * (r + self.temporal_discount * max_q - Q[(s, a)])
            priority += np.abs(P)
        return priority

    def _simulate_behavior(self):
        """
        Perform random-sample one-step tabular Q-planning with prioritized
        sweeping.

        Notes
        -----
        This approach uses a priority queue to retrieve the state-action pairs
        from the agent's history with largest change to their Q-values if
        backed up. When the first pair in the queue is backed up, the effect on
        each of its predecessor pairs is computed. If the predecessor's
        priority is greater than a small threshold the pair is added to the
        queue and the process is repeated until either the queue is empty or we
        have exceeded a `n_simulated_actions` updates.
        """
        env_model = self.parameters["model"]
        sweep_queue = self.derived_variables["sweep_queue"]
        for _ in range(self.n_simulated_actions):
            if len(sweep_queue) == 0:
                break

            # select (s, a) pair with the largest update (priority)
            sq_items = list(sweep_queue.items())
            (s_sim, a_sim), _ = sorted(sq_items, key=lambda x: x[1], reverse=True)[0]

            # remove entry from queue
            del sweep_queue[(s_sim, a_sim)]

            # update Q function for (s_sim, a_sim) using the full-backup
            # version of the TD(0) Q-learning update
            self._update(s_sim, a_sim)

            # get all (_s, _a) pairs that lead to s_sim (ie., s_sim's predecessors)
            pairs = env_model.state_action_pairs_leading_to_outcome(s_sim)

            # add predecessors to queue if their priority exceeds thresh
            for (_s, _a) in pairs:
                self._update_queue(_s, _a)

    def _update(self, s, a):
        """
        Update Q using a full-backup version of the TD(0) Q-learning update:

            Q(s, a) = Q(s, a) + lr *
                sum_{r, s'} [
                    p(r, s' | s, a) * (r + gamma * max_a { Q(s', a) } - Q(s, a))
                ]

        Parameters
        ----------
        s : int as returned by ``self._obs2num``
            The id for the state/observation
        a : int as returned by ``self._action2num``
            The id for the action taken from state `s`
        """
        update = 0.0
        env_model = self.parameters["model"]
        E, D, Q = self.env_info, self.derived_variables, self.parameters["Q"]
        n_actions = np.prod(E["n_actions_per_dim"])

        # sample rewards from the model
        outcome_probs = env_model.outcome_probs(s, a)
        for (r, s_), p_rs_ in outcome_probs:
            # encourage visiting long-untried actions by adding a "bonus"
            # reward proportional to the sqrt of the time since last visit
            if self.q_plus:
                r += self.explore_weight * np.sqrt(D["steps_since_last_visit"][(s, a)])

            max_q = np.max([Q[(s_, a_)] for a_ in range(n_actions)])
            update += p_rs_ * (r + self.temporal_discount * max_q - Q[(s, a)])

        # update Q value for (s, a) pair
        Q[(s, a)] += self.lr * update

    def run_episode(self, max_steps, render=False):
        """
        Run the agent on a single episode without performing `Q`-function
        backups.

        Parameters
        ----------
        max_steps : int
            The maximum number of steps to run an episode.
        render : bool
            Whether to render the episode during training.

        Returns
        -------
        reward : float
            The total reward on the episode.
        steps : float
            The number of steps taken on the episode.
        """
        return self._episode(max_steps, render, update=False)

    def train_episode(self, max_steps, render=False):
        """
        Train the agent on a single episode.

        Parameters
        ----------
        max_steps : int
            The maximum number of steps to run an episode.
        render : bool
            Whether to render the episode during training.

        Returns
        -------
        reward : float
            The total reward on the episode.
        steps : float
            The number of steps taken on the episode.
        """
        D = self.derived_variables
        total_rwd, n_steps = self._episode(max_steps, render, update=True)
        D["episode_num"] += 1
        return total_rwd, n_steps

    def _episode(self, max_steps, render, update=True):
        """
        Run or train the agent on an episode.

        Parameters
        ----------
        max_steps : int
            The maximum number of steps to run the episode.
        render : bool
            Whether to render the episode during training.
        update : bool
            Whether to perform the `Q` function backups after each step.
            Default is True.

        Returns
        -------
        reward : float
            The total reward on the episode.
        steps : float
            The number of steps taken on the episode.
        """
        self.flush_history()

        obs = self.env.reset()
        env_model = self.parameters["model"]
        HS, D = self.episode_history, self.derived_variables

        action = self.act(obs)
        s = self._obs2num[obs]
        a = self._action2num[action]

        # store initial (state, action) tuple
        HS["state_actions"].append((s, a))

        total_reward, n_steps = 0.0, 0
        for i in range(max_steps):
            if render:
                self.env.render()

            # take action
            obs, reward, done, info = self.env.step(action)
            n_steps += 1

            # record rewards
            HS["rewards"].append(reward)
            total_reward += reward

            # generate next state and action
            action = self.act(obs)
            s_ = self._obs2num[obs] if not done else None
            a_ = self._action2num[action]

            # update model
            env_model[(s, a, reward, s_)] += 1

            # update history counter
            for k in D["steps_since_last_visit"].keys():
                D["steps_since_last_visit"][k] += 1
            D["steps_since_last_visit"][(s, a)] = 0

            if update:
                self.update()

            # store next (state, action) tuple
            HS["state_actions"].append((s_, a_))
            s, a = s_, a_

            if done:
                break

        return total_reward, n_steps

    def greedy_policy(self, max_steps, render=True):
        """
        Execute a deterministic greedy policy using the current agent
        parameters.

        Parameters
        ----------
        max_steps : int
            The maximum number of steps to run the episode.
        render : bool
            Whether to render the episode during execution.

        Returns
        -------
        total_reward : float
            The total reward on the episode.
        n_steps : float
            The total number of steps taken on the episode.
        """
        self.flush_history()

        H = self.episode_history
        obs = self.env.reset()

        total_reward, n_steps = 0.0, 0
        for i in range(max_steps):
            if render:
                self.env.render()

            s = self._obs2num[obs]
            action = self._greedy(s)
            a = self._action2num[action]

            # store (state, action) tuple
            H["state_actions"].append((s, a))

            # take action
            obs, reward, done, info = self.env.step(action)
            n_steps += 1

            # record rewards
            H["rewards"].append(reward)
            total_reward += reward

            if done:
                break

        return total_reward, n_steps
