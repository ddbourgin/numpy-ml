# numpy-ml
Ever wish you had an inefficient but somewhat legible collection of machine
learning algorithms implemented exclusively in NumPy? No?

## Installation

### For rapid experimentation
To use this code as a starting point for ML prototyping / experimentation, just clone the repository, create a new [virtualenv](https://pypi.org/project/virtualenv/), and start hacking:

```sh
$ git clone https://github.com/ddbourgin/numpy-ml.git
$ cd numpy-ml && virtualenv npml && source npml/bin/activate
$ pip install -r requirements-dev.txt
```

### For use as a package
If you don't plan to modify the source, you can also install numpy-ml as a
Python package: `pip install -u numpy_ml`.

The reinforcement learning agents train on environments defined in the [OpenAI
gym](https://github.com/openai/gym). To install these alongside numpy-ml, you
can use `pip install -u numpy_ml[rl]`.

## Documentation
To see the available models, take a look at the [project documentation](https://numpy-ml.readthedocs.io/) or see [here](https://github.com/ddbourgin/numpy-ml/blob/master/numpy_ml/README.md).

## Contributing

Am I missing your favorite model? Is there something that could be cleaner /
less confusing? Did I mess something up? Submit a PR! The only requirement is
that your models are written with just the [Python standard
library](https://docs.python.org/3/library/) and [NumPy](https://www.numpy.org/). The
[SciPy library](https://scipy.github.io/devdocs/) is also permitted under special
circumstances ;)

See full contributing guidelines [here](./CONTRIBUTING.md).
