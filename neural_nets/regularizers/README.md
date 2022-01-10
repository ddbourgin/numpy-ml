## Usage of regularizers

Regularizers allow to apply penalties on layer parameters or layer activity during optimization. These penalties are incorporated in the loss function that the network optimizes.

The penalties are applied on a per-layer basis.

## Example

```python
from neural_nets import regularizers

```

## Available penalties

```python
regularizers.l1(0.)
regularizers.l2(0.)
regularizers.l1_l2(l1=0.01, l2=0.01)
```

## Developing new regularizers

Any function that takes in a weight matrix and returns a loss contribution tensor can be used as a regularizer, e.g.:

```python
import numpy as np

def l1_reg(weight_matrix):
    return 0.01 * np.sum(np.abs(weight_matrix))
```

Alternatively, you can write your regularizers in an object-oriented way;
see the [neural_nets/regularizers.py](regularizers.py) module for examples.