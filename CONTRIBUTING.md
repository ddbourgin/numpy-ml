## Contributing

Thank you for contributing to numpy-ml!

| <p align="center">⚠️ ⚠️ All PRs should reflect earnest attempts at implementing a model yourself. ⚠️⚠️ </p> It is fine to reference others' code. It is not fine to blindly copy without attribution. When in doubt, please ask. |
| --- |

### General guidelines
1. Please include a clear list of what you've done
2. For pull requests, please make sure all commits are [*atomic*](https://en.wikipedia.org/wiki/Atomic_commit) (i.e., one feature per commit)
3. If you're submitting a new model / feature / module, **please include proper documentation and unit tests.**
    - See the `test.py` file in one of the existing modules for examples of unit tests.
    - Documentation is loosely based on the [NumPy docstring style](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html). When in doubt, refer to existing examples
4. Please format your code using the [black](https://github.com/python/black) defaults. You can use this [online formatter](https://black.now.sh/).

### Specific guidelines
#### I have a new model / model component to contribute
 Awesome - create a [pull request](https://github.com/ddbourgin/numpy-ml/pulls)! When preparing your PR, please include a brief description of the model, the canonical reference(s) in the literature, and, most importantly unit tests against an existing implementation!
  - Refer to the `test.py` file in one of the existing modules for examples.

#### I have a major new enhancement / adjustment that will affect multiple models
 Please post an [issue](https://github.com/ddbourgin/numpy-ml/issues) with your proposal before you begin working on it. When outlining your proposal, please include as much detail about your intended changes as possible.

#### I found a bug
 If there isn't already an [open issue](https://github.com/ddbourgin/numpy-ml/issues), please start one! When creating your issue, include:
  1. A title and clear description
  2. As much relevant information as possible
  3. A code sample demonstrating the expected behavior that is not occurring

#### I fixed a bug
 Thank you! Please open a new [pull request](https://github.com/ddbourgin/numpy-ml/pulls) with the patch. When doing so, ensure the PR description clearly describes the problem and solution. Include the relevant issue number if applicable.
