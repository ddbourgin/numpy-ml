Nonparametric models 
####################

.. raw:: html

   <h2>Gaussian Process Regression</h2>

A Gaussian process defines a prior distribution over functions mapping
:math:`X \rightarrow \mathbb{R}`, where `X` can be any finite (or
infinite!)-dimensional set.

More concretely, let :math:`f(x_k)` be the random variable corresponding to
the value of a function `f` at a point :math:`x_k \in X`. We can define a
random variable :math:`z = [f(x_1), \ldots, f(x_N)]` for any finite set of
points :math:`\{x_1, \ldots, x_N\} \subset X`. If `f` is distributed
according to a Gaussian Process, we have that

.. math::

    z \sim \mathcal{N}(\mu, K)

for

.. math::

    \mu  &=  [\text{mean}(x_1), \ldots, \text{mean}(x_N)] \\
    K_{ij}  &=  \text{kernel}(x_i, x_j)

where mean is the mean function (it is common to define mean(`x`) = 0`),
and `kernel` is a kernel / covariance function that determines the general
shape of the GP prior over functions, `p(f)`.

.. raw:: html

   <h2>Kernel Regression</h2>

TODO

.. raw:: html

   <h2>Nearest Neighbors</h2>

TODO


.. toctree::
   :maxdepth: 2

   numpy_ml.nonparametric.gp
   numpy_ml.nonparametric.knn
   numpy_ml.nonparametric.kernel_regression
