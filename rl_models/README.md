# RL Models
The `agents.py` module implements a number of standard reinforcement learning (RL) agents that
can be run on [OpenAI gym](https://gym.openai.com/) environments.

1. **Monte Carlo Methods**
- `CrossEntropyAgent` - Iteratively learns to minimize the cross-entropy between
  the optimal policy and the current behavior policy
([Mannor, Rubinstein, and Gat, 2003](https://www.aaai.org/Papers/ICML/2003/ICML03-068.pdf))
- `MonteCarloAgent` - Uses either first-visit Monte Carlo updates (on-policy) or
  incremental weighted importance sampling (off-policy) to learn an optimal
behavior policy.

