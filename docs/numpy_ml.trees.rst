Tree-based models
#################
.. raw:: html

   <h2>Decision Trees</h2>

`Decision trees`_ [1]_ are popular nonparametric models that iteratively split a
training dataset into smaller, more homogenous subsets. Each node in the tree
is associated with a decision rule, which dictates how to divide the data the
node inherits from its parent among each of its children. Each leaf node is
associated with at least one data point from the original training set.

.. figure:: img/decision_tree.png
    :width: 95%
    :align: center

    A binary decision tree trained on the dataset :math:`X = \{ \mathbf{x}_1,
    \ldots, \mathbf{x}_{10} \}`. Each example in the dataset is a 4-dimensional
    vector of real-valued features labeled :math:`x_1, \ldots, x_4`. Unshaded
    circles correspond to internal decision nodes, while shaded circles
    correspond to leaf nodes. Each leaf node is associated with a subset of the
    examples in `X`, selected based on the decision rules along the path from
    root to leaf.

At test time, new examples travel from the tree root to one of the leaves,
their path through the tree determined by the decision rules at each of the
nodes it visits. When a test example arrives at a leaf node, the targets for
the training examples at that leaf node are used to compute the model's
prediction.

Training decision trees corresponds to learning the set of decision rules to
partition the training data. This learning process proceeds greedily by
selecting the decision rule at each node that results in the greatest reduction
in an inhomogeneity or "impurity" metric, :math:`\mathcal{L}`. One popular
metric is the **information entropy**:

.. math::

    -\sum_j P_n(\omega_j) \log P_n(\omega_j)

where :math:`P_n(\omega_j)` is the fraction of data points at split `n` that are
associated with category :math:`\omega_j`. Another useful metric is the **Gini
impurity**:

.. math::

    \sum_{i \neq j} P_n(\omega_i) P_n(\omega_j) = 1 - \sum_{j} P_n(\omega_j)^2

For a binary tree (where each node has only two children), the reduction in
impurity after a particular split is

.. math::

    \Delta \mathcal{L} = \mathcal{L}(\text{Parent}) -
        P_{\text{left}} \mathcal{L}(\text{Left child}) -
            (1 - P_{\text{left}})\mathcal{L}(\text{Right child})

where :math:`\mathcal{L}(x)` is the impurity of the dataset at node `x`,
and :math:`P_{\text{left}}`/:math:`P_{\text{right}}` are the proportion of
examples at the current node that are partitioned into the left / right
children, respectively, by the proposed split.

.. _`Decision trees`: https://en.wikipedia.org/wiki/Decision_tree_learning

**Models**

- :class:`~numpy_ml.trees.DecisionTree`

**References**

.. [1] Breiman, L., Friedman, J. H., Olshen, R. A., and Stone, C. J. (1984).
   Classification and regression trees. Monterey, CA: Wadsworth & Brooks/Cole
   Advanced Books & Software.

.. raw:: html

   <h2>Bootstrap Aggregating</h2>

`Bootstrap aggregating`_ (bagging) methods [2]_ are an `ensembling approach`_ that
proceeds by creating `n` bootstrapped samples of a training dataset by sampling
from it with replacement. A separate learner is fit on each of the `n`
bootstrapped datasets, with the final bootstrap aggregated model prediction
corresponding to the average (or majority vote, for classifiers) across each
of the `n` learners' predictions for a given datapoint.

The `random forest`_ model [3]_ [4]_ is a canonical example of bootstrap
aggregating. For this approach, each of the `n` learners is a different
decision tree. In addition to training each decision tree on a different
bootstrapped dataset, random forests employ a `random subspace`_ approach [5]_:
each decision tree is trained on a subsample (without replacement) of the full
collection of dataset features.

.. _`Bootstrap aggregating`: https://en.wikipedia.org/wiki/Bootstrap_aggregating
.. _`random forest`: https://en.wikipedia.org/wiki/Random_forest
.. _`ensembling approach`: https://en.wikipedia.org/wiki/Ensemble_learning
.. _`random subspace`: https://en.wikipedia.org/wiki/Random_subspace_method

**Models**

- :class:`~numpy_ml.trees.RandomForest`

**References**

.. [2] Breiman, L. (1994). "Bagging predictors". *Technical Report 421.
   Statistics Department, UC Berkeley*.
.. [3] Ho, T. K. (1995). "Random decision forests". *Proceedings of the Third
   International Conference on Document Analysis and Recognition, 1*: 278-282.
.. [4] Breiman, L. (2001). "Random forests". *Machine Learning. 45(1)*: 5-32.
.. [5] Ho, T. K. (1998). "The random subspace method for constructing decision
   forests". *IEEE Transactions on Pattern Analysis and Machine Intelligence.
   20(8)*: 832-844.

.. raw:: html

   <h2>Gradient Boosting</h2>

`Gradient boosting`_ [6]_ [7]_ [8]_ is another popular `ensembling technique`_
that proceeds by iteratively fitting a sequence of `m` weak learners such that:

.. math::

    f_m(X) = b(X) + \eta w_1 g_1 + \ldots + \eta w_m g_m

where `b` is a fixed initial estimate for the targets, :math:`\eta` is
a learning rate parameter, and :math:`w_{i}` and :math:`g_{i}`
denote the weights and predictions of the :math:`i^{th}` learner.

At each training iteration a new weak learner is fit to predict the negative
gradient of the loss with respect to the previous prediction,
:math:`\nabla_{f_{i-1}} \mathcal{L}(y, \ f_{i-1}(X))`.  We then use the
element-wise product of the predictions of this weak learner, :math:`g_i`, with
a weight, :math:`w_i`, computed via, e.g., `line-search`_ on the objective
:math:`w_i = \arg \min_{w} \sum_{j=1}^n \mathcal{L}(y_j, f_{i-1}(x_j) + w g_i)`
, to adjust the predictions of the model from the previous iteration,
:math:`f_{i-1}(X)`:

.. math::

    f_i(X) := f_{i-1}(X) + w_i g_i

The current module implements gradient boosting using decision trees as the
weak learners.

.. _`Gradient boosting`: https://en.wikipedia.org/wiki/Gradient_boosting
.. _`ensembling technique`: https://en.wikipedia.org/wiki/Ensemble_learning
.. _`line-search`: https://en.wikipedia.org/wiki/Line_search

**Models**

- :class:`~numpy_ml.trees.GradientBoostedDecisionTree`

**References**

.. [6]  Breiman, L. (1997). "Arcing the edge". *Technical Report 486.
   Statistics Department, UC Berkeley*.
.. [7] Friedman, J. H. (1999). "Greedy function approximation: A gradient
   boosting machine". *IMS 1999 Reitz Lecture*.
.. [8]  Mason, L., Baxter, J., Bartlett, P. L., Frean, M. (1999). "Boosting
   algorithms as gradient descent" *Advances in Neural Information Processing
   Systems, 12*: 512–518.

.. toctree::
   :maxdepth: 3
   :hidden:

   numpy_ml.trees.dt

   numpy_ml.trees.rf

   numpy_ml.trees.gbdt
