#######################
N-gram smoothing models
#######################

When dealing with `n-gram`_ models, smoothing refers to the practice of
adjusting empirical probability estimates to account for insufficient data.

In the descriptions below, we use the notation :math:`w^{j}_{i}`, :math:`i < j`, to
denote the `(j - i)`-gram :math:`(w_{i}, w_{i+1}, \ldots, w_{j})`.

.. raw:: html

   <h3>Laplace Smoothing</h3>

`Laplace smoothing`_ is the assumption that each `n`-gram in a corpus occurs
exactly one more time than it actually does.

.. math::

    p(w_i \mid w^{i-1}_{i-n+1}) = \frac{1 + c(w^{i}_{i-n+1})}{|V| \sum_{w_i} c(w^{i}_{i-n+1})}

where :math:`c(a)` denotes the empirical count of the `n`-gram :math:`a` in the
corpus, and :math:`|V|` corresponds to the number of unique `n`-grams in the
corpus.

.. _`Laplace smoothing`: https://en.wikipedia.org/wiki/Additive_smoothing

**Models**

- :class:`~numpy_ml.ngram.AdditiveNGram`

.. raw:: html

   <h3>Additive/Lidstone Smoothing</h3>

`Additive/Lidstone smoothing`_ is a generalization of Laplace smoothing, where we
assume that each `n`-gram in a corpus occurs `k` more times than it actually
does (where `k` can be any non-negative value, but typically ranges between `[0, 1]`):

.. math::

    p(w_i \mid w^{i-1}_{i-n+1}) = \frac{k + c(w^{i}_{i-n+1})}{k |V| \sum_{w_i} c(w^{i}_{i-n+1})}

where :math:`c(a)` denotes the empirical count of the `n`-gram :math:`a` in the
corpus, and :math:`|V|` corresponds to the number of unique `n`-grams in the
corpus.

.. _`Additive/Lidstone smoothing`: https://en.wikipedia.org/wiki/Additive_smoothing

**Models**

- :class:`~numpy_ml.ngram.AdditiveNGram`


.. raw:: html

   <h3>Good-Turing Smoothing</h3>

`Good-Turing smoothing`_ is a more sophisticated technique which takes into
account the identity of the particular `n`-gram when deciding the amount of
smoothing to apply. It proceeds by allocating a portion of the probability
space occupied by `n`-grams which occur with count `r+1` and dividing it among
the `n`-grams which occur with rate `r`.

.. math::
    r^*  =  (r + 1) \frac{g(r + 1)}{g(r)} \\
    p(w^{i}_{i-n+1} \mid c(w^{i}_{i-n+1}) = r)  =  \frac{r^*}{N}

where :math:`r^*` is the adjusted count for an `n`-gram which occurs `r` times,
`g(x)` is the number of `n`-grams in the corpus which occur `x` times, and `N`
is the total number of `n`-grams in the corpus.

.. _n-gram: https://en.wikipedia.org/wiki/N-gram
.. _`Good-Turing smoothing`: https://en.wikipedia.org/wiki/Good%E2%80%93Turing_frequency_estimation

**Models**

- :class:`~numpy_ml.ngram.GoodTuringNGram`

**References**

.. [1]  Chen & Goodman (1998). "An empirical study of smoothing techniques
   for language modeling".  *Harvard Computer Science Group Technical Report
   TR-10-98*.
.. [2] Gale & Sampson (1995). "Good-Turing frequency estimation without
   tears". *Journal of Quantitative Linguistics*, 2(3), 217-237.

.. toctree::
   :maxdepth: 3
   :hidden:

   numpy_ml.ngram.mle

   numpy_ml.ngram.additive

   numpy_ml.ngram.goodturing
