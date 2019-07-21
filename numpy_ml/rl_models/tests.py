import gym

from .trainer import Trainer
from .agents import (
    CrossEntropyAgent,
    MonteCarloAgent,
    TemporalDifferenceAgent,
    DynaAgent,
)


def test_cross_entropy_agent():
    seed = 12345
    max_steps = 300
    n_episodes = 50
    retain_prcnt = 0.2
    n_samples_per_episode = 500
    env = gym.make("LunarLander-v2")

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


def test_temporal_difference_agent():
    seed = 12345
    max_steps = 200
    n_episodes = 5000

    lr = 0.4
    n_tilings = 10
    epsilon = 0.10
    off_policy = True
    grid_dims = [100, 100]
    smooth_factor = 0.005
    temporal_discount = 0.999
    env = gym.make("LunarLander-v2")
    obs_max = 1
    obs_min = -1

    agent = TemporalDifferenceAgent(
        env,
        lr=lr,
        obs_max=obs_max,
        obs_min=obs_min,
        epsilon=epsilon,
        n_tilings=n_tilings,
        grid_dims=grid_dims,
        off_policy=off_policy,
        temporal_discount=temporal_discount,
    )

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


def test_dyna_agent():
    seed = 12345
    max_steps = 200
    n_episodes = 150

    lr = 0.4
    q_plus = False
    n_tilings = 10
    epsilon = 0.10
    grid_dims = [10, 10]
    smooth_factor = 0.01
    temporal_discount = 0.99
    explore_weight = 0.05
    n_simulated_actions = 25

    obs_max, obs_min = 1, -1
    env = gym.make("Taxi-v2")

    agent = DynaAgent(
        env,
        lr=lr,
        q_plus=q_plus,
        obs_max=obs_max,
        obs_min=obs_min,
        epsilon=epsilon,
        n_tilings=n_tilings,
        grid_dims=grid_dims,
        explore_weight=explore_weight,
        temporal_discount=temporal_discount,
        n_simulated_actions=n_simulated_actions,
    )

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
