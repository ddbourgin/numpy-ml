from collections import namedtuple

import numpy as np

from .bandits import (
    MABMultinomialPayoff,
    MABBernoulliPayoff,
    MABShortestPath,
)
from .trainer import MABTrainer
from .policies import EpsilonGreedy, UCB1, ThompsonSamplingBetaBinomial
from ..utils.graphs import random_DAG, DiGraph, Edge


def random_multinomial_mab(n_arms=10, n_choices_per_arm=5, reward_range=[0, 1]):
    payoffs = []
    payoff_probs = []
    lo, hi = reward_range
    for a in range(n_arms):
        p = np.random.uniform(size=n_choices_per_arm)
        p = p / p.sum()
        r = np.random.uniform(low=lo, high=hi, size=n_choices_per_arm)

        payoffs.append(list(r))
        payoff_probs.append(list(p))

    return MABMultinomialPayoff(payoffs, payoff_probs)


def random_bernoulli_mab(n_arms=10):
    p = np.random.uniform(size=n_arms)
    payoff_probs = p / p.sum()
    return MABBernoulliPayoff(payoff_probs)


def plot_epsilon_greedy_multinomial_payoff():
    np.random.seed(12345)
    N = np.random.randint(2, 30)  # n arms
    K = np.random.randint(2, 10)  # n payoffs / arm
    ep_length = 1  # np.random.randint(1, 25)

    rrange = [0, 1]
    n_duplicates = 5
    n_episodes = 5000

    mab = random_multinomial_mab(N, K, rrange)
    policy = EpsilonGreedy(epsilon=0.05, ev_prior=rrange[1] / 2)
    policy = MABTrainer().train(policy, mab, ep_length, n_episodes, n_duplicates)


def plot_ucb1_multinomial_payoff():
    np.random.seed(12345)
    N = np.random.randint(2, 30)  # n arms
    K = np.random.randint(2, 10)  # n payoffs / arm
    ep_length = 1  # np.random.randint(1, 25)

    C = 1
    rrange = [0, 1]
    n_duplicates = 5
    n_episodes = 5000

    mab = random_multinomial_mab(N, K, rrange)
    policy = UCB1(C=C, ev_prior=rrange[1] / 2)
    policy = MABTrainer().train(policy, mab, ep_length, n_episodes, n_duplicates)


def plot_thompson_sampling_beta_binomial_payoff():
    np.random.seed(12345)
    N = np.random.randint(2, 30)  # n arms
    ep_length = 1  # np.random.randint(1, 25)

    n_duplicates = 5
    n_episodes = 5000

    mab = random_bernoulli_mab(N)
    policy = ThompsonSamplingBetaBinomial(alpha=1, beta=1)
    policy = MABTrainer().train(policy, mab, ep_length, n_episodes, n_duplicates)


def plot_ucb1_gaussian_shortest_path():
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
        w = lambda: np.random.normal(mean, var)
        rv = Gaussian(mean, var, mean, w)
        E.append(Edge(e.fr, e.to, rv))

    G = DiGraph(V, E)
    print("Graph built")
    while not G.path_exists(V[0], V[-1]):
        print("Skipping")
        idx = np.random.randint(0, len(V))
        V[idx], V[-1] = V[-1], V[idx]

    print("Starting bandit")
    mab = MABShortestPath(G, V[0], V[-1])
    policy = UCB1(C=1, ev_prior=0.5)
    policy = MABTrainer().train(policy, mab, ep_length, n_episodes, n_duplicates)
