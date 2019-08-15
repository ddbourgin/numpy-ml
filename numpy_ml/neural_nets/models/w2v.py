from time import time

import numpy as np

from ..layers import Embedding
from ..losses import NCELoss

from ...preprocessing.nlp import Vocabulary, tokenize_words
from ...utils.data_structures import DiscreteSampler


class Word2Vec(object):
    def __init__(
        self,
        context_len=5,
        min_count=None,
        skip_gram=False,
        max_tokens=None,
        embedding_dim=300,
        filter_stopwords=True,
        noise_dist_power=0.75,
        init="glorot_uniform",
        num_negative_samples=64,
        optimizer="SGD(lr=0.1)",
    ):
        """
        A word2vec model supporting both continuous bag of words (CBOW) and
        skip-gram architectures, with training via noise contrastive
        estimation.

        Parameters
        ----------
        context_len : int
            The number of words to the left and right of the current word to
            use as context during training. Larger values result in more
            training examples and thus can lead to higher accuracy at the
            expense of additional training time. Default is 5.
        min_count : int or None
            Minimum number of times a token must occur in order to be included
            in vocab. If None, include all tokens from `corpus_fp` in vocab.
            Default is None.
        skip_gram : bool
            Whether to train the skip-gram or CBOW model. The skip-gram model
            is trained to predict the target word i given its surrounding
            context, ``words[i - context:i]`` and ``words[i + 1:i + 1 +
            context]`` as input. Default is False.
        max_tokens : int or None
            Only add the first `max_tokens` most frequent tokens that occur
            more than `min_count` to the vocabulary.  If None, add all tokens
            that occur more than than `min_count`. Default is None.
        embedding_dim : int
            The number of dimensions in the final word embeddings. Default is
            300.
        filter_stopwords : bool
            Whether to remove stopwords before encoding the words in the
            corpus. Default is True.
        noise_dist_power : float
            The power the unigram count is raised to when computing the noise
            distribution for negative sampling. A value of 0 corresponds to a
            uniform distribution over tokens, and a value of 1 corresponds to a
            distribution proportional to the token unigram counts. Default is
            0.75.
        init : {'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform'}
            The weight initialization strategy. Default is 'glorot_uniform'.
        num_negative_samples: int
            The number of negative samples to draw from the noise distribution
            for each positive training sample. If 0, use the hierarchical
            softmax formulation of the model instead. Default is 5.
        optimizer : str, :doc:`Optimizer <numpy_ml.neural_nets.optimizers>` object, or None
            The optimization strategy to use when performing gradient updates
            within the `update` method.  If None, use the
            :class:`~numpy_ml.neural_nets.optimizers.SGD` optimizer with
            default parameters. Default is None.

        Attributes
        ----------
        parameters : dict
        hyperparameters : dict
        derived_variables : dict
        gradients : dict

        Notes
        -----
        The word2vec model is outlined in in [1].

        CBOW architecture::

            w_{t-R}   ----|
            w_{t-R+1} ----|
            ...            --> Average --> Embedding layer --> [NCE Layer / HSoftmax] --> P(w_{t} | w_{...})
            w_{t+R-1} ----|
            w_{t+R}   ----|

        Skip-gram architecture::

                                                                   |-->  P(w_{t-R} | w_{t})
                                                                   |-->  P(w_{t-R+1} | w_{t})
            w_{t} --> Embedding layer --> [NCE Layer / HSoftmax] --|     ...
                                                                   |-->  P(w_{t+R-1} | w_{t})
                                                                   |-->  P(w_{t+R} | w_{t})

        where :math:`w_{i}` is the one-hot representation of the word at position
        `i` within a sentence in the corpus and `R` is the length of the context
        window on either side of the target word.

        References
        ----------
        .. [1] Mikolov et al. (2013). "Distributed representations of words
           and phrases and their compositionality," Proceedings of the 26th
           International Conference on Neural Information Processing Systems.
           https://arxiv.org/pdf/1310.4546.pdf
        """
        self.init = init
        self.optimizer = optimizer
        self.skip_gram = skip_gram
        self.min_count = min_count
        self.max_tokens = max_tokens
        self.context_len = context_len
        self.embedding_dim = embedding_dim
        self.filter_stopwords = filter_stopwords
        self.noise_dist_power = noise_dist_power
        self.num_negative_samples = num_negative_samples
        self.special_chars = set(["<unk>", "<eol>", "<bol>"])

    def _init_params(self):
        self._dv = {}
        self._build_noise_distribution()

        self.embeddings = Embedding(
            init=self.init,
            vocab_size=self.vocab_size,
            n_out=self.embedding_dim,
            optimizer=self.optimizer,
            pool=None if self.skip_gram else "mean",
        )

        self.loss = NCELoss(
            init=self.init,
            optimizer=self.optimizer,
            n_classes=self.vocab_size,
            subtract_log_label_prob=False,
            noise_sampler=self._noise_sampler,
            num_negative_samples=self.num_negative_samples,
        )

    @property
    def parameters(self):
        """Model parameters"""
        param = {"components": {"embeddings": {}, "loss": {}}}
        if hasattr(self, "embeddings"):
            param["components"] = {
                "embeddings": self.embeddings.parameters,
                "loss": self.loss.parameters,
            }
        return param

    @property
    def hyperparameters(self):
        """Model hyperparameters"""
        hp = {
            "layer": "Word2Vec",
            "init": self.init,
            "skip_gram": self.skip_gram,
            "optimizer": self.optimizer,
            "max_tokens": self.max_tokens,
            "context_len": self.context_len,
            "embedding_dim": self.embedding_dim,
            "noise_dist_power": self.noise_dist_power,
            "filter_stopwords": self.filter_stopwords,
            "num_negative_samples": self.num_negative_samples,
            "vocab_size": self.vocab_size if hasattr(self, "vocab_size") else None,
            "components": {"embeddings": {}, "loss": {}},
        }

        if hasattr(self, "embeddings"):
            hp["components"] = {
                "embeddings": self.embeddings.hyperparameters,
                "loss": self.loss.hyperparameters,
            }
        return hp

    @property
    def derived_variables(self):
        """Variables computed during model operation"""
        dv = {"components": {"embeddings": {}, "loss": {}}}
        dv.update(self._dv)

        if hasattr(self, "embeddings"):
            dv["components"] = {
                "embeddings": self.embeddings.derived_variables,
                "loss": self.loss.derived_variables,
            }
        return dv

    @property
    def gradients(self):
        """Model parameter gradients"""
        grad = {"components": {"embeddings": {}, "loss": {}}}
        if hasattr(self, "embeddings"):
            grad["components"] = {
                "embeddings": self.embeddings.gradients,
                "loss": self.loss.gradients,
            }
        return grad

    def forward(self, X, targets, retain_derived=True):
        """
        Evaluate the network on a single minibatch.

        Parameters
        ----------
        X : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, n_in)`
            Layer input, representing a minibatch of `n_ex` examples, each
            consisting of `n_in` integer word indices
        targets : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex,)`
            Target word index for each example in the minibatch.
        retain_derived : bool
            Whether to retain the variables calculated during the forward pass
            for use later during backprop. If `False`, this suggests the layer
            will not be expected to backprop through wrt. this input. Default
            True.

        Returns
        -------
        loss : float
            The loss associated with the current minibatch
        y_pred : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex,)`
            The conditional probabilities of the words in `targets` given the
            corresponding example / context in `X`.
        """
        X_emb = self.embeddings.forward(X, retain_derived=True)
        loss, y_pred = self.loss.loss(X_emb, targets.flatten(), retain_derived=True)
        return loss, y_pred

    def backward(self):
        """
        Compute the gradient of the loss wrt the current network parameters.
        """
        dX_emb = self.loss.grad(retain_grads=True, update_params=False)
        self.embeddings.backward(dX_emb)

    def update(self, cur_loss=None):
        """Perform gradient updates"""
        self.loss.update(cur_loss)
        self.embeddings.update(cur_loss)
        self.flush_gradients()

    def flush_gradients(self):
        """Reset parameter gradients after update"""
        self.loss.flush_gradients()
        self.embeddings.flush_gradients()

    def get_embedding(self, word_ids):
        """
        Retrieve the embeddings for a collection of word IDs.

        Parameters
        ----------
        word_ids : :py:class:`ndarray <numpy.ndarray>` of shape `(M,)`
            An array of word IDs to retrieve embeddings for.

        Returns
        -------
        embeddings : :py:class:`ndarray <numpy.ndarray>` of shape `(M, n_out)`
            The embedding vectors for each of the `M` word IDs.
        """
        if isinstance(word_ids, list):
            word_ids = np.array(word_ids)
        return self.embeddings.lookup(word_ids)

    def _build_noise_distribution(self):
        """
        Construct the noise distribution for use during negative sampling.

        For a word ``w`` in the corpus, the noise distribution is::

            P_n(w) = Count(w) ** noise_dist_power / Z

        where ``Z`` is a normalizing constant, and `noise_dist_power` is a
        hyperparameter of the model. Mikolov et al. report best performance
        using a `noise_dist_power` of 0.75.
        """
        if not hasattr(self, "vocab"):
            raise ValueError("Must call `fit` before constructing noise distribution")

        probs = np.zeros(len(self.vocab))
        power = self.hyperparameters["noise_dist_power"]

        for ix, token in enumerate(self.vocab):
            count = token.count
            probs[ix] = count ** power

        probs /= np.sum(probs)
        self._noise_sampler = DiscreteSampler(probs, log=False, with_replacement=False)

    def _train_epoch(self, corpus_fps, encoding):
        total_loss = 0
        batch_generator = self.minibatcher(corpus_fps, encoding)
        for ix, (X, target) in enumerate(batch_generator):
            loss = self._train_batch(X, target)
            total_loss += loss
            if self.verbose:
                smooth_loss = 0.99 * smooth_loss + 0.01 * loss if ix > 0 else loss
                fstr = "[Batch {}] Loss: {:.5f} | Smoothed Loss: {:.5f}"
                print(fstr.format(ix + 1, loss, smooth_loss))
        return total_loss / (ix + 1)

    def _train_batch(self, X, target):
        loss, _ = self.forward(X, target)
        self.backward()
        self.update(loss)
        return loss

    def minibatcher(self, corpus_fps, encoding):
        """
        A minibatch generator for skip-gram and CBOW models.

        Parameters
        ----------
        corpus_fps : str or list of strs
            The filepath / list of filepaths to the document(s) to be encoded.
            Each document is expected to be encoded as newline-separated
            string of text, with adjacent tokens separated by a whitespace
            character.
        encoding : str
            Specifies the text encoding for corpus. This value is passed
            directly to Python's `open` builtin. Common entries are either
            'utf-8' (no header byte), or 'utf-8-sig' (header byte).

        Yields
        ------
        X : list of length `batchsize` or :py:class:`ndarray <numpy.ndarray>` of shape (`batchsize`, `n_in`)
            The context IDs for a minibatch of `batchsize` examples. If
            ``self.skip_gram`` is False, `X` will be a ragged list consisting
            of `batchsize` variable-length lists. If ``self.skip_gram`` is
            `True`, all sublists will be of the same length (`n_in`) and `X`
            will be returned as a :py:class:`ndarray <numpy.ndarray>` of shape (`batchsize`, `n_in`).
        target : :py:class:`ndarray <numpy.ndarray>` of shape (`batchsize`, 1)
            The target IDs associated with each example in `X`
        """
        batchsize = self.batchsize
        X_mb, target_mb, mb_ready = [], [], False

        for d_ix, doc_fp in enumerate(corpus_fps):
            with open(doc_fp, "r", encoding=encoding) as doc:
                for line in doc:
                    words = tokenize_words(
                        line, lowercase=True, filter_stopwords=self.filter_stopwords
                    )
                    word_ixs = self.vocab.words_to_indices(
                        self.vocab.filter(words, unk=False)
                    )
                    for word_loc, word in enumerate(word_ixs):
                        # since more distant words are usually less related to
                        # the target word, we downweight them by sampling from
                        # them less frequently during training.
                        R = np.random.randint(1, self.context_len)
                        left = word_ixs[max(word_loc - R, 0) : word_loc]
                        right = word_ixs[word_loc + 1 : word_loc + 1 + R]
                        context = left + right

                        if len(context) == 0:
                            continue

                        # in the skip-gram architecture we use each of the
                        # surrounding context to predict `word` / avoid
                        # predicting negative samples
                        if self.skip_gram:
                            X_mb.extend([word] * len(context))
                            target_mb.extend(context)
                            mb_ready = len(target_mb) >= batchsize

                        # in the CBOW architecture we use the average of the
                        # context embeddings to predict the target `word` / avoid
                        # predicting the negative samples
                        else:
                            context = np.array(context)
                            X_mb.append(context)  # X_mb will be a ragged array
                            target_mb.append(word)
                            mb_ready = len(X_mb) == batchsize

                        if mb_ready:
                            mb_ready = False
                            X_batch, target_batch = X_mb.copy(), target_mb.copy()
                            X_mb, target_mb = [], []
                            if self.skip_gram:
                                X_batch = np.array(X_batch)[:, None]
                            target_batch = np.array(target_batch)[:, None]
                            yield X_batch, target_batch

        # if we've reached the end of our final document and there are
        # remaining examples, yield the stragglers as a partial minibatch
        if len(X_mb) > 0:
            if self.skip_gram:
                X_mb = np.array(X_mb)[:, None]
            target_mb = np.array(target_mb)[:, None]
            yield X_mb, target_mb

    def fit(
        self, corpus_fps, encoding="utf-8-sig", n_epochs=20, batchsize=128, verbose=True
    ):
        """
        Learn word2vec embeddings for the examples in `X_train`.

        Parameters
        ----------
        corpus_fps : str or list of strs
            The filepath / list of filepaths to the document(s) to be encoded.
            Each document is expected to be encoded as newline-separated
            string of text, with adjacent tokens separated by a whitespace
            character.
        encoding : str
            Specifies the text encoding for corpus. Common entries are either
            'utf-8' (no header byte), or 'utf-8-sig' (header byte).  Default
            value is 'utf-8-sig'.
        n_epochs : int
            The maximum number of training epochs to run. Default is 20.
        batchsize : int
            The desired number of examples in each training batch. Default is
            128.
        verbose : bool
            Print batch information during training. Default is True.
        """
        self.verbose = verbose
        self.n_epochs = n_epochs
        self.batchsize = batchsize

        self.vocab = Vocabulary(
            lowercase=True,
            min_count=self.min_count,
            max_tokens=self.max_tokens,
            filter_stopwords=self.filter_stopwords,
        )
        self.vocab.fit(corpus_fps, encoding=encoding)
        self.vocab_size = len(self.vocab)

        # ignore special characters when training the model
        for sp in self.special_chars:
            self.vocab.counts[sp] = 0

        # now that we know our vocabulary size, we can initialize the embeddings
        self._init_params()

        prev_loss = np.inf
        for i in range(n_epochs):
            loss, estart = 0.0, time()
            loss = self._train_epoch(corpus_fps, encoding)

            fstr = "[Epoch {}] Avg. loss: {:.3f}  Delta: {:.3f} ({:.2f}m/epoch)"
            print(fstr.format(i + 1, loss, prev_loss - loss, (time() - estart) / 60.0))
            prev_loss = loss
