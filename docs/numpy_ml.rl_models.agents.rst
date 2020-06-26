Agents
======

``CrossEntropyAgent``
---------------------
.. autoclass:: numpy_ml.rl_models.agents.CrossEntropyAgent
    :members:
    :undoc-members:
    :inherited-members:

``DynaAgent``
-------------
.. autoclass:: numpy_ml.rl_models.agents.DynaAgent
    :members:
    :undoc-members:
    :inherited-members:

``MonteCarloAgent``
-------------------
Monte Carlo methods are ways of solving RL problems based on averaging
sample returns for each state-action pair. Parameters are updated only at
the completion of an episode.

In on-policy learning, the agent maintains a single policy that it updates
over the course of training. In order to ensure the policy converges to a
(near-) optimal policy, the agent must maintain that the policy assigns
non-zero probability to ALL state-action pairs during training to ensure
continual exploration.

- Thus on-policy learning is a compromise--it learns action values not for the optimal policy, but for a *near*-optimal policy that still explores.

In off-policy learning, the agent maintains two separate policies:

1. **Target policy**: The policy that is learned during training and that will eventually become the optimal policy.
2. **Behavior policy**: A policy that is more exploratory and is used to generate behavior during training.

Off-policy methods are often of greater variance and are slower to
converge. On the other hand, off-policy methods are more powerful and
general than on-policy methods.

.. autoclass:: numpy_ml.rl_models.agents.MonteCarloAgent
    :members:
    :undoc-members:
    :inherited-members:

``TemporalDifferenceAgent``
---------------------------

Temporal difference methods are examples of bootstrapping in that they update
their estimate for the value of state `s` on the basis of a previous estimate.

Advantages of TD algorithms:

1. They do not require a model of the environment, its reward, or its next-state probability distributions.
2. They are implemented in an online, fully incremental fashion. This allows them to be used with infinite-horizons / when episodes take prohibitively long to finish.
3. TD algorithms learn from each transition regardless of what subsequent actions are taken.
4. In practice, TD methods have usually been found to converge faster than constant-:math:`\alpha` Monte Carlo methods on stochastic tasks.

.. autoclass:: numpy_ml.rl_models.agents.TemporalDifferenceAgent
    :members:
    :undoc-members:
    :inherited-members:
