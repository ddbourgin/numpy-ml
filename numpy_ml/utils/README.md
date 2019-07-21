# Utilities

The utilities module implements a number of useful functions and objects that
power other ML algorithms across the repo.

- `data_structures.py` implements a few useful data structures
    - A max- and min-heap ordered priority queue
    - A [ball tree](https://en.wikipedia.org/wiki/Ball_tree) with the KNS1 algorithm ([Omohundro, 1989](http://ftp.icsi.berkeley.edu/ftp/pub/techreports/1989/tr-89-063.pdf); [Moore & Gray, 2006](http://people.ee.duke.edu/~lcarin/liu06a.pdf))
    - A discrete sampler implementing Vose's algorithm for the [alias method](https://en.wikipedia.org/wiki/Alias_method) ([Walker, 1977](https://dl.acm.org/citation.cfm?id=355749); [Vose, 1991](https://pdfs.semanticscholar.org/f65b/cde1fcf82e05388b31de80cba10bf65acc07.pdf))

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

- `windows.py` implements several common windowing functions
    - Hann
    - Hamming
    - Blackman-Harris
    - Generalized cosine

- `testing.py` implements helper functions that prove useful when writing unit
  tests, including data generators and various assert statements
