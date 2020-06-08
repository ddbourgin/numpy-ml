"""Miscellaneous plots for multi-arm bandit validation"""

from collections import namedtuple

import numpy as np

from numpy_ml.bandits import (
    MultinomialBandit,
    BernoulliBandit,
    ShortestPathBandit,
    ContextualLinearBandit,
)
from numpy_ml.bandits.trainer import BanditTrainer
from numpy_ml.bandits.policies import (
    EpsilonGreedy,
    UCB1,
    ThompsonSamplingBetaBinomial,
    LinUCB,
)
from numpy_ml.utils.graphs import random_DAG, DiGraph, Edge


def random_multinomial_mab(n_arms=10, n_choices_per_arm=5, reward_range=[0, 1]):
    """Generate a random multinomial multi-armed bandit environemt"""
    payoffs = []
    payoff_probs = []
    lo, hi = reward_range
    for a in range(n_arms):
        p = np.random.uniform(size=n_choices_per_arm)
        p = p / p.sum()
        r = np.random.uniform(low=lo, high=hi, size=n_choices_per_arm)

        payoffs.append(list(r))
        payoff_probs.append(list(p))

    return MultinomialBandit(payoffs, payoff_probs)


def random_bernoulli_mab(n_arms=10):
    """Generate a random Bernoulli multi-armed bandit environemt"""
    p = np.random.uniform(size=n_arms)
    payoff_probs = p / p.sum()
    return BernoulliBandit(payoff_probs)


def plot_epsilon_greedy_multinomial_payoff():
    """
    Evaluate an epsilon-greedy policy on a random multinomial bandit
    problem
    """
    np.random.seed(12345)
    N = np.random.randint(2, 30)  # n arms
    K = np.random.randint(2, 10)  # n payoffs / arm
    ep_length = 1

    rrange = [0, 1]
    n_duplicates = 5
    n_episodes = 5000

    mab = random_multinomial_mab(N, K, rrange)
    policy = EpsilonGreedy(epsilon=0.05, ev_prior=rrange[1] / 2)
    policy = BanditTrainer().train(policy, mab, ep_length, n_episodes, n_duplicates)


def plot_ucb1_multinomial_payoff():
    """Evaluate the UCB1 policy on a multinomial bandit environment"""
    np.random.seed(12345)
    N = np.random.randint(2, 30)  # n arms
    K = np.random.randint(2, 10)  # n payoffs / arm
    ep_length = 1

    C = 1
    rrange = [0, 1]
    n_duplicates = 5
    n_episodes = 5000

    mab = random_multinomial_mab(N, K, rrange)
    policy = UCB1(C=C, ev_prior=rrange[1] / 2)
    policy = BanditTrainer().train(policy, mab, ep_length, n_episodes, n_duplicates)


def plot_thompson_sampling_beta_binomial_payoff():
    """
    Evaluate the ThompsonSamplingBetaBinomial policy on a random Bernoulli
    multi-armed bandit.
    """
    np.random.seed(12345)
    N = np.random.randint(2, 30)  # n arms
    ep_length = 1

    n_duplicates = 5
    n_episodes = 5000

    mab = random_bernoulli_mab(N)
    policy = ThompsonSamplingBetaBinomial(alpha=1, beta=1)
    policy = BanditTrainer().train(policy, mab, ep_length, n_episodes, n_duplicates)


def plot_lin_ucb():
    """Plot the linUCB policy on a contextual linear bandit problem"""
    np.random.seed(12345)
    ep_length = 1
    K = np.random.randint(2, 25)
    D = np.random.randint(2, 10)

    n_duplicates = 5
    n_episodes = 5000

    cmab = ContextualLinearBandit(K, D, 1)
    policy = LinUCB(alpha=1)
    policy = BanditTrainer().train(policy, cmab, ep_length, n_episodes, n_duplicates)


def plot_ucb1_gaussian_shortest_path():
    """
    Plot the UCB1 policy on a graph shortest path problem each edge weight
    drawn from an independent univariate Gaussian
    """
    np.random.seed(12345)

    ep_length = 1
    n_duplicates = 5
    n_episodes = 5000
    p = np.random.rand()
    n_vertices = np.random.randint(5, 15)

    Gaussian = namedtuple("Gaussian", ["mean", "variance", "EV", "sample"])

    # create randomly-weighted edges
    print("Building graph")
    E = []
    G = random_DAG(n_vertices, p)
    V = G.vertices
    for e in G.edges:
        mean, var = np.random.uniform(0, 1), np.random.uniform(0, 1)
        w = lambda: np.random.normal(mean, var)  # noqa: E731
        rv = Gaussian(mean, var, mean, w)
        E.append(Edge(e.fr, e.to, rv))

    G = DiGraph(V, E)
    while not G.path_exists(V[0], V[-1]):
        print("Skipping")
        idx = np.random.randint(0, len(V))
        V[idx], V[-1] = V[-1], V[idx]

    mab = ShortestPathBandit(G, V[0], V[-1])
    policy = UCB1(C=1, ev_prior=0.5)
    policy = BanditTrainer().train(policy, mab, ep_length, n_episodes, n_duplicates)


def plot_comparison():
    """
    Use the BanditTrainer to compare several policies on the same bandit
    problem
    """
    np.random.seed(1234)
    ep_length = 1
    K = 10

    n_duplicates = 5
    n_episodes = 5000

    cmab = random_bernoulli_mab(n_arms=K)
    policy1 = EpsilonGreedy(epsilon=0.05, ev_prior=0.5)
    policy2 = UCB1(C=1, ev_prior=0.5)
    policy3 = ThompsonSamplingBetaBinomial(alpha=1, beta=1)
    policies = [policy1, policy2, policy3]

    BanditTrainer().compare(
        policies, cmab, ep_length, n_episodes, n_duplicates,
    )
