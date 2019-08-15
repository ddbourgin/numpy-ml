Tree-based models 
#################
.. raw:: html

   <h2>Decision Trees</h2>

Decision trees are popular nonparametric models that iteratively split a
training dataset into smaller, more homogenous subsets.  At test time, the tree
determines which of the training subsets a new test example falls within and uses
the items in this subset to compute the model's prediction.

Decision trees greedily look for splits that minimize an inhomogeneity metric,
:math:`\mathcal{L}`.  One popular split metric is the **information entropy**:

.. math::
    
    -\sum_j P_n(\omega_j) \log P_n(\omega_j)

where :math:`P_n(\omega_j)` is the fraction of data points at split `n` that are
associated with category :math:`\omega_j`. Another useful metric is the **Gini
impurity**:

.. math::
    
    \sum_{i \neq j} P_n(\omega_i) P_n(\omega_j) = 1 - \sum_{j} P_n(\omega_j)^2

Each node in the decision tree corresponds to a partitioning of the dataset
inherited from its parent. The partition rule is chosen locally in order to
reduce the overall impurity as much as possible. In a binary tree, the
reduction in impurity after a particular split is

.. math::
    
    \Delta \mathcal{L} = \mathcal{L}(\text{Parent}) - P_{left} \mathcal{L}(\text{Left child}) - (1 - P_{left})\mathcal{L}(\text{Right child})

where :math:`\mathcal{L}(x)` is the impurity of the dataset at node `x`,
and :math:`P_{left}`/:math:`P_{right}` are the proportion of examples at the
current node that are partitioned into the left / right children, respectively,
by the proposed split.

.. raw:: html

   <h2>Bootstrap Aggregating</h2>

.. raw:: html

   <h2>Gradient Boosting</h2>

Gradient boosting is a popular ensembling technique in machine learning. It
proceeds by iteratively fitting a sequence of `m` weak learners such that:

.. math::

    f_m(X) = b(X) + \eta w_1 g_1 + \ldots + \eta w_m g_m

where `b` is a fixed initial estimate for the targets, :math:`\eta` is
a learning rate parameter, and :math:`w_{i}` and :math:`g_{i}`
denote the weights and predictions for `i` th learner.

At each iteration we fit a new weak learner to predict the negative gradient of
the loss with respect to the previous prediction, :math:`f_{i-1}(X)`.  We then
use the element-wise product of the predictions of this weak learner,
:math:`g_i`, with a weight, :math:`w_i`, to adjust the predictions of our model
from the previous iteration, :math:`f_{i-1}(X)`:

.. math::

    f_i(X) := f_{i-1}(X) + w_i g_i

.. toctree::
   :maxdepth: 3

   numpy_ml.trees.dt

   numpy_ml.trees.gbdt

   numpy_ml.trees.rf
