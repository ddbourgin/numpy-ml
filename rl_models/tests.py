import gym

from trainer import Trainer
from agents import CrossEntropyAgent, MonteCarloAgent


def test_cross_entropy_agent():
    seed = 12345
    max_steps = 300
    n_episodes = 50
    retain_prcnt = 0.2
    n_samples_per_episode = 500
    env = gym.make("CartPole-v1")

    agent = CrossEntropyAgent(env, n_samples_per_episode, retain_prcnt)
    trainer = Trainer(agent, env)
    trainer.train(
        n_episodes, max_steps, seed=seed, plot=True, verbose=True, render_every=None
    )


def test_monte_carlo_agent():
    seed = 12345
    max_steps = 300
    n_episodes = 10000

    epsilon = 0.05
    off_policy = True
    smooth_factor = 0.001
    temporal_discount = 0.95
    env = gym.make("Copy-v0")

    agent = MonteCarloAgent(env, off_policy, temporal_discount, epsilon)
    trainer = Trainer(agent, env)
    trainer.train(
        n_episodes,
        max_steps,
        seed=seed,
        plot=True,
        verbose=True,
        render_every=None,
        smooth_factor=smooth_factor,
    )
