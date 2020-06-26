Nonparametric models
####################

.. raw:: html

   <h2>K-Nearest Neighbors</h2>

The `k-nearest neighbors`_ (KNN) model is a nonparametric supervised learning
approach that can be applied to classification or regression problems. In a
classification context, the KNN model assigns a class label for a new datapoint
by taking a majority vote amongst the labels for the `k` closest points
("neighbors") in the training data. Similarly, in a regression context, the KNN
model predicts the target value associated with a new datapoint by taking the
average of the targets associated with the `k` closes points in the training
data.

.. _`k-nearest neighbors`: https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm

**Models**

- :class:`~numpy_ml.nonparametric.KNN`

.. raw:: html

   <h2>Gaussian Process Regression</h2>

A `Gaussian process`_ defines a prior distribution over functions mapping
:math:`X \rightarrow \mathbb{R}`, where `X` can be any finite (or
infinite!)-dimensional set.

Let :math:`f(x_k)` be the random variable corresponding to
the value of a function `f` at a point :math:`x_k \in X`. Define a random
variable :math:`z = [f(x_1), \ldots, f(x_N)]` for any finite set of points
:math:`\{x_1, \ldots, x_N\} \subset X`. If `f` is distributed according to a
Gaussian Process, it is the case that

.. math::

    z \sim \mathcal{N}(\mu, K)

for

.. math::

    \mu  &=  [\text{mean}(x_1), \ldots, \text{mean}(x_N)] \\
    K_{ij}  &=  \text{kernel}(x_i, x_j)

where mean is the mean function (in Gaussian process regression it is common
to define mean(`x`) = 0), and `kernel` is a :doc:`kernel
<numpy_ml.utils.kernels>` / covariance function that determines the general
shape of the GP prior over functions, `p(f)`.

In `Gaussian process regression`_ (AKA simple Kriging [2]_ [3]_), a Gaussian
process is used as a prior on functions and is combined with the Gaussian
likelihood from the linear model via Bayes' rule to compute a posterior over
functions `f`:

.. math::

    y \mid X, f  &\sim  \mathcal{N}( [f(x_1), \ldots, f(x_n)], \alpha I ) \\
    f \mid X     &\sim  \text{GP}(0, K)

Due to the conjugacy of the Gaussian Process prior with the regression model's
Gaussian likelihood, the posterior will also be Gaussian and can be computed in
closed form.

.. _`Gaussian process`: https://en.wikipedia.org/wiki/Gaussian_process
.. _`Gaussian process regression`: https://en.wikipedia.org/wiki/Kriging

**Models**

- :class:`~numpy_ml.nonparametric.GPRegression`

**References**

.. [1] Rasmussen, C. E., & Williams, C. K. I. (2006). Gaussian Processes for
   Machine Learning. MIT Press, Cambridge, MA.
.. [2] Krige, D. G., (1951). "A statistical approach to some mine valuations and
   allied problems at the Witwatersrand", *Master's thesis of the University of
   Witwatersrand*.
.. [3] Matheron, G., (1963). "Principles of geostatistics", *Economic Geology, 58*, 1246-1266.

.. raw:: html

   <h2>Kernel Regression</h2>

Kernel regression is another nonparametric approach to nonlinear regression.
Like the Gaussian Process regression approach (or, more generally, all
regression models), kernel regression attempts to learn a function `f` which
captures the conditional expectation of some targets **y** given the data
**X**, under the assumption that

.. math::
    y_i = f(x_i) + \epsilon_i \ \ \ \ \text{where } \mathbb{E}[\epsilon | \mathbf{x}] = \mathbb{E}[\epsilon] = 0

Unlike the Gaussian Process regression approach, however, kernel regression
does not place a prior over `f`. Instead, it models :math:`f = \mathbb{E}[y |
X] = \int_y \frac{p(X, y)}{p(X)} y \ \text{d}y` using a :doc:`kernel function
<numpy_ml.utils.kernels>`, `k`, to estimate the smoothed data probabilities.
For example, the :class:`Nadaraya-Watson <numpy_ml.nonparametric.KernelRegression>`
estimator [4]_ [5]_ uses the following probability estimates:

.. math::
    \hat{p}(X)  &=  \prod_{i=1}^N \hat{p}(x_i) = \prod_{i=1}^N \sum_{j=1}^N \frac{k(x_i - x_j)}{N} \\
    \hat{p}(X, y)  &  \prod_{i=1}^N \hat{p}(x_i, y_i) = \prod_{i=1}^N \sum_{j=1}^N \frac{k(x_i - x_j) k(y_i - y_j)}{N}


**Models**

- :class:`~numpy_ml.nonparametric.KernelRegression`

**References**

.. [4] Nadaraya, E. A. (1964). "On estimating regression". *Theory of
   Probability and Its Applications, 9 (1)*, 141-2.
.. [5] Watson, G. S. (1964). "Smooth regression analysis". *Sankhyā: The Indian
   Journal of Statistics, Series A. 26 (4)*, 359–372.

.. raw:: html

   <h2>See Also</h2>

The :doc:`trees <numpy_ml.trees>` module contains other classic nonparametric
approaches, including :doc:`decision trees <numpy_ml.trees.dt>`,
:doc:`random forests <numpy_ml.trees.rf>`, and :doc:`gradient
boosted decision trees <numpy_ml.trees.gbdt>`.

.. toctree::
   :maxdepth: 2
   :hidden:

   numpy_ml.nonparametric.knn
   numpy_ml.nonparametric.gp
   numpy_ml.nonparametric.kernel_regression
