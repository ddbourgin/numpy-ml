# Utilities

The utilities module implements a number of useful functions and objects that
power other ML algorithms across the repo.

- `data_structures.py` implements several "advanced" data structures
    - A max- and min-heap ordered priority queue
    - A [ball tree](https://en.wikipedia.org/wiki/Ball_tree) with the KNS1 algorithm ([Omohundro, 1989](http://ftp.icsi.berkeley.edu/ftp/pub/techreports/1989/tr-89-063.pdf); [Moore & Gray, 2006](http://people.ee.duke.edu/~lcarin/liu06a.pdf))

- `kernels.py` implements several general-purpose similarity kernels
    - Linear kernel
    - Polynomial kernel
    - Radial basis function kernel

- `distance_metrics.py` implements common distance metrics
    - Euclidean (L2) distance
    - Manhattan (L1) distance
    - Chebyshev (L-infinity) distance
    - Minkowski-p distance
    - Hamming distance
