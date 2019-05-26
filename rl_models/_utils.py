import copy
from itertools import product
from collections import Hashable

import numpy as np
import gym


class Dict(dict):
    """
    A dictionary subclass which returns the key value if not already in the dict
    """

    def __setitem__(self, key, value):
        if not isinstance(key, Hashable):
            key = tuple(key)
        super(Dict, self).__setitem__(key, value)

    def __getitem__(self, key):
        if not isinstance(key, Hashable):
            self._key = copy.deepcopy(key)
            key = tuple(key)
        return super(Dict, self).__getitem__(key)

    def __missing__(self, key):
        return self._key


def gym_environs():
    """
    List all valid gym environment ids
    """
    return [e.id for e in gym.envs.registry.all()]


def is_multidimensional(env):
    tuple_space = gym.spaces.Tuple
    md_action = isinstance(env.action_space, tuple_space)
    md_obs = isinstance(env.observation_space, tuple_space)
    return md_action, md_obs


def is_continuous(env, md_action, md_obs):
    """
    Check if the observation and action spaces are continuous
    """
    Continuous = gym.spaces.box.Box
    if md_obs:
        spaces = env.observation_space.spaces
        cont_obs = all([isinstance(s, Continuous) for s in spaces])
    else:
        cont_obs = isinstance(env.observation_space, Continuous)

    if md_action:
        spaces = env.action_space.spaces
        cont_action = all([isinstance(s, Continuous) for s in spaces])
    else:
        cont_action = isinstance(env.action_space, Continuous)
    return cont_action, cont_obs


def action_stats(env, md_action, cont_action):
    """
    Get information on env's action space
    """
    if cont_action:
        action_dim = 1
        action_ids = None
        n_actions_per_dim = [np.inf]

        if md_action:
            action_dim = env.action_space.shape[0]
            n_actions_per_dim = [np.inf for _ in range(action_dim)]
    else:
        if md_action:
            n_actions_per_dim = [
                space.n if hasattr(space, "n") else np.inf
                for space in env.action_space.spaces
            ]
            action_ids = (
                None
                if np.inf in n_actions_per_dim
                else list(product(*[range(i) for i in n_actions_per_dim]))
            )
            action_dim = len(n_actions_per_dim)
        else:
            action_dim = 1
            n_actions_per_dim = [env.action_space.n]
            action_ids = list(range(n_actions_per_dim[0]))
    return n_actions_per_dim, action_ids, action_dim


def obs_stats(env, md_obs, cont_obs):
    """
    Get information on env's observation space
    """
    if cont_obs:
        obs_ids = None
        obs_dim = env.observation_space.shape[0]
        n_obs_per_dim = [np.inf for _ in range(obs_dim)]
    else:
        if md_obs:
            n_obs_per_dim = [
                space.n if hasattr(space, "n") else np.inf
                for space in env.observation_space.spaces
            ]
            obs_ids = (
                None
                if np.inf in n_obs_per_dim
                else list(product(*[range(i) for i in n_obs_per_dim]))
            )
            obs_dim = len(n_obs_per_dim)
        else:
            obs_dim = 1
            n_obs_per_dim = [env.observation_space.n]
            obs_ids = list(range(n_obs_per_dim[0]))

    return n_obs_per_dim, obs_ids, obs_dim


def env_stats(env):
    """
    Compute statistics for the current environment.

    Parameters
    ----------
    env : gym.wrappers or gym.envs instance
        The environment to run the agent on

    Returns
    -------
    env_info : dict
        A dictionary containing information about the action and observation
        spaces of `env`
    """
    md_action, md_obs = is_multidimensional(env)
    cont_action, cont_obs = is_continuous(env, md_action, md_obs)

    n_actions_per_dim, action_ids, action_dim = action_stats(
        env, md_action, cont_action
    )
    n_obs_per_dim, obs_ids, obs_dim = obs_stats(env, md_obs, cont_obs)

    env_info = {
        "id": env.spec.id,
        "seed": env.spec.seed if env.spec.nondeterministic else None,
        "deterministic": bool(~env.spec.nondeterministic),
        "multidim_actions": md_action,
        "multidim_observations": md_obs,
        "continuous_actions": cont_action,
        "continuous_observations": cont_obs,
        "n_actions_per_dim": n_actions_per_dim,
        "action_dim": action_dim,
        "n_obs_per_dim": n_obs_per_dim,
        "obs_dim": obs_dim,
        "action_ids": action_ids,
        "obs_ids": obs_ids,
    }

    return env_info
