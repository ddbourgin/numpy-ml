from abc import ABC, abstractmethod

import numpy as np

from _utils import env_stats, Dict


class AgentBase(ABC):
    def __init__(self, env):
        super().__init__()
        self.env = env
        self.parameters = {}
        self.hyperparameters = {}
        self.derived_variables = {}
        self.env_info = env_stats(env)
        self.create_2num_dicts()

    def create_2num_dicts(self):
        E = self.env_info
        n_states = np.prod(E["n_obs_per_dim"])
        n_actions = np.prod(E["n_actions_per_dim"])

        # create action -> scalar dictionaries
        self._action2num = self._num2action = Dict()
        if n_actions != np.inf:
            self._action2num = {act: i for i, act in enumerate(E["action_ids"])}
            self._num2action = {i: act for act, i in self._action2num.items()}

        # create obs -> scalar dictionaries
        self._obs2num = self._num2obs = Dict()
        if n_states != np.inf:
            self._obs2num = {act: i for i, act in enumerate(E["obs_ids"])}
            self._num2obs = {i: act for act, i in self._obs2num.items()}

    def flush_history(self):
        for k, v in self.episode_history.items():
            self.episode_history[k] = []

    @abstractmethod
    def act(self, obs):
        raise NotImplementedError

    @abstractmethod
    def greedy_policy(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def run_episode(self, max_steps, render=False):
        raise NotImplementedError

    @abstractmethod
    def update(self):
        raise NotImplementedError


class CrossEntropyAgent(AgentBase):
    def __init__(self, env, n_samples_per_episode=500, retain_prcnt=0.2):
        """
        A cross-entropy method agent. Operates on envs with discrete action
        spaces.

        On each episode the agent generates `n_theta_samples` of the parameters
        (theta) for its behavior policy. The i'th sample is:

            theta_i ~ N(theta_mean_t, diag(theta_var_t))
            theta_i = [W_i, b_i]

        Weights (W_i) and bias (b_i) are the parameters of the softmax policy:

            z_i = obs @ W_i + b_i
            p(action_i) = e^z_i / sum_j e^z_j
            action = arg max_j p(action_j)

        At the end of each episode, the agent takes the top `retain_prcnt`
        highest scoring thetas and combines them to generate the mean and
        variance for the next theta samples:

            theta_mean_{t+1} = mean(best_thetas_t)
            theta_var_{t+1} = diag( varian

        Parameters
        ----------
        env : gym.wrappers or gym.envs instance
            The environment to run the agent on
        n_samples_per_episode : int (default: 500)
            The number of theta samples to evaluate on each episode
        retain_prcnt: float (default: 0.2)
            The percentage of `n_samples_per_episode` to use when calculating
            the parameter update at the end of the episode.
        """
        super().__init__(env)

        self.retain_prcnt = retain_prcnt
        self.n_samples_per_episode = n_samples_per_episode
        self._init_params()

    def _init_params(self):
        E = self.env_info
        assert not E["continuous_actions"], "Action space must be discrete"

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
        """
        Generate actions according to a softmax policy.

        The softmax policy assumes that the pmf over actions in state `xt` is
        given by

            action_probs = softmax( obs @ W + b )

        where W is a learned weight matrix, obs is the observation at timestep
        t, and b is a learned bias vector.

        Parameters
        ----------
        obs : int or array
            An observation from the environment.
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
        Run an agent on a single episode.

        At each timestep CrossEntropyAagent generates `n_theta_samples` from the
        distribution:

            theta_i ~ N(theta_mean_t, diag(theta_var_t))
            theta_i = [W_i, b_i]

        Weights (W_i) and bias (b_i) are parameters of the softmax policy:

            z_i = obs @ W_i + b_i
            p(action_i) = e^z_i / sum_j e^z_j
            action = arg max_j p(action_j)

        At the end of each episode, take the top `retain_prcnt` highest scoring
        thetas and combine them to produce the mean and variance for the next
        theta samples:

            theta_mean_{t+1} = mean(best_thetas_t)
            theta_var_{t+1} = diag( variance(best_thetas_t) )

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
        W : numpy array of shape (obs_dim, n_actions)
            The weights for the softmax policy
        b : numpy array of shape (bias_len, )
            The bias for the softmax policy
        max_steps : int
            The maximum number of steps to run the episode for
        render : bool
            Whether to redner the episode during training

        Returns
        -------
        reward : float
            The total reward on the episode
        steps : float
            The total number of steps taken on the episode
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
        """
        Update `theta_mean` and `theta_var` according to the rewards accrued on
        the current episode.

        Returns
        -------
        avg_reward : float
            The average reward earned by the best `retain_prcnt` theta samples
            on the current episode
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
        Sample `n_samples_per_episode` thetas from a MV Gaussian with mean
        `theta_mean` and covariance `diag(theta_var)`
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
            The maximum number of steps to run the episode for
        render : bool
            Whether to redner the episode during execution

        Returns
        -------
        total_reward : float
            The total reward on the episode
        n_steps : float
            The total number of steps taken on the episode
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
    """
    Monte Carlo methods are ways of solving RL problems based on averaging
    sample returns for each state-action pair. Parameters are updated only at
    the completion of an episode.

    In on-policy learning, the agent maintains a single policy that it updates
    over the course of training. In order to ensure the policy converges to a
    (near-) optimal policy, the agent must maintain that the policy assigns
    non-zero probability to ALL state-action pairs during training to ensure
    continual exploration.
        - Thus on-policy learning is a compromise -- it learns action values
          not for the optimal policy, but for a *near*-optimal policy that
          still explores.

    In off-policy learning, the agent maintains two separate policies:
        1. Target Policy - The policy that is learned during training and that
        will eventually become the optimal policy.

        2. Behavior Policy - A policy that is more exploratory and is used to
        generate behavior during training.

    Off-policy methods are often of greater variance and are slower to
    converge. On the other hand, off-policy methods are more powerful and
    general than on-policy methods.
    """

    def __init__(self, env, off_policy=False, temporal_discount=0.9, epsilon=0.1):
        """
        A Monte-Carlo learning agent trained using either first-visit Monte
        Carlo updates (on-policy) or incremental weighted importance sampling
        (off-policy).

        Parameters
        ----------
        env : gym.wrappers or gym.envs instance
            The environment to run the agent on
        off_policy : bool (default: False)
            Whether to use a behavior policy separate from the target policy
            during training. If False, use the same epsilon-soft policy for
            both behavior and target policies.
        temporal_discount : float between [0, 1] (default: 0.9)
            The discount factor used for downweighting future rewards. Smaller
            values result in greater discounting of future rewards.
        epsilon : float between [0, 1] (default: 0.1)
            The epsilon value in the epsilon-soft policy. Larger values
            encourage greater exploration during training.
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

        self.create_2num_dicts()

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
        """
        Epsilon-soft exploration policy. This is necessary for first-visit
        Monte Carlo methods, as they require continual exploration (i.e., each
        state-action pair must have nonzero probability of occurring).

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
        a : int, float, or tuple (default : None)
            The action number in the current state, as returned by
            self._action2num[obs]. If None, sample an action from the action
            probabilities in state s, otherwise, return the probability of
            action `a` under the epsilon-soft policy.

        Returns
        -------
        If `a` is None:
        action : int, float, or np.ndarray as returned by `self._num2action`
            If `a` is None, returns an action sampled from the distribution
            over actions defined by the epsilon-soft policy.

        If `a` is not None:
        action_prob : float in range [0, 1]
            If `a` is not None, returns the probability of `a` under the
            epsilon-soft policy.
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
        A greedy behavior policy. Only used when off-policy is true.

        Parameters
        ----------
        s : int, float, or tuple
            The state number for the current observation, as returned by
            self._obs2num[obs]
        a : int, float, or tuple (default : None)
            The action number in the current state, as returned by
            self._action2num[obs]. If None, sample an action from the action
            probabilities in state s, otherwise, return the probability of
            action `a` under the greedy policy.

        Returns
        -------
        If `a` is None:
        action : int, float, or np.ndarray as returned by `self._num2action`
            If `a` is None, returns an action sampled from the distribution
            over actions defined by the greedy policy.

        If `a` is not None:
        action_prob : float in range [0, 1]
            If `a` is not None, returns the probability of `a` under the
            greedy policy.
        """
        a_star = self.parameters["Q"][s, :].argmax()
        if a is None:
            out = self._num2action[a_star]
        else:
            out = 1 if a == a_star else 0
        return out

    def _on_policy_update(self):
        """
        Update the Q function using an on-policy first-visit Monte Carlo
        update:

        Q'(s, a) <- avg(reward following first visit to (s, a), across all
                        episodes)

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
        Update Q using weighted importance sampling.

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
        """
        Behavior policy

        Parameters
        ----------
        obs : int, float, or np.ndarray as returned by `env.step(action)`
            An observation from the environment.
        """
        s = self._obs2num[obs]
        return self.behavior_policy(s)

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
            The maximum number of steps to run the episode for
        render : bool
            Whether to redner the episode during training

        Returns
        -------
        reward : float
            The total reward on the episode
        steps : float
            The number of steps taken on the episode
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
        episode. Flush the episode history after update is complete.
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
            The maximum number of steps to run the episode for
        render : bool
            Whether to redner the episode during execution

        Returns
        -------
        total_reward : float
            The total reward on the episode
        n_steps : float
            The total number of steps taken on the episode
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
