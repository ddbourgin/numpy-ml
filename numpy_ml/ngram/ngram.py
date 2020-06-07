"""A module for different N-gram smoothing models"""
import textwrap
from abc import ABC, abstractmethod
from collections import Counter

import numpy as np

from ..linear_models.lm import LinearRegression
from ..preprocessing.nlp import tokenize_words, ngrams, strip_punctuation


class NGramBase(ABC):
    def __init__(self, N, unk=True, filter_stopwords=True, filter_punctuation=True):
        """
        A simple word-level N-gram language model.

        Notes
        -----
        This is not optimized code and will be slow for large corpora. To see
        how industry-scale NGram models are handled, see the SRLIM-format:

            http://www.speech.sri.com/projects/srilm/
        """
        self.N = N
        self.unk = unk
        self.filter_stopwords = filter_stopwords
        self.filter_punctuation = filter_punctuation

        self.hyperparameters = {
            "N": N,
            "unk": unk,
            "filter_stopwords": filter_stopwords,
            "filter_punctuation": filter_punctuation,
        }

        super().__init__()

    def train(self, corpus_fp, vocab=None, encoding=None):
        """
        Compile the n-gram counts for the text(s) in `corpus_fp`.

        Notes
        -----
        After running `train`, the ``self.counts`` attribute will store
        dictionaries of the `N`, `N-1`, ..., 1-gram counts.

        Parameters
        ----------
        corpus_fp : str
            The path to a newline-separated text corpus file.
        vocab : :class:`~numpy_ml.preprocessing.nlp.Vocabulary` instance or None
            If not None, only the words in `vocab` will be used to construct
            the language model; all out-of-vocabulary words will either be
            mappend to ``<unk>`` (if ``self.unk = True``) or removed (if
            ``self.unk = False``). Default is None.
        encoding : str or None
            Specifies the text encoding for corpus. Common entries are 'utf-8',
            'utf-8-sig', 'utf-16'. Default is None.
        """
        return self._train(corpus_fp, vocab=vocab, encoding=encoding)

    def _train(self, corpus_fp, vocab=None, encoding=None):
        """Actual N-gram training logic"""
        H = self.hyperparameters
        grams = {N: [] for N in range(1, self.N + 1)}
        counts = {N: Counter() for N in range(1, self.N + 1)}
        filter_stop, filter_punc = H["filter_stopwords"], H["filter_punctuation"]

        _n_words = 0
        tokens = {"<unk>"}
        bol, eol = ["<bol>"], ["<eol>"]

        with open(corpus_fp, "r", encoding=encoding) as text:
            for line in text:
                line = strip_punctuation(line) if filter_punc else line
                words = tokenize_words(line, filter_stopwords=filter_stop)

                if vocab is not None:
                    words = vocab.filter(words, H["unk"])

                if len(words) == 0:
                    continue

                _n_words += len(words)
                tokens.update(words)

                # calculate n, n-1, ... 1-grams
                for N in range(1, self.N + 1):
                    words_padded = bol * max(1, N - 1) + words + eol * max(1, N - 1)
                    grams[N].extend(ngrams(words_padded, N))

        for N in counts.keys():
            counts[N].update(grams[N])

        n_words = {N: np.sum(list(counts[N].values())) for N in range(1, self.N + 1)}
        n_words[1] = _n_words

        n_tokens = {N: len(counts[N]) for N in range(2, self.N + 1)}
        n_tokens[1] = len(vocab) if vocab is not None else len(tokens)

        self.counts = counts
        self.n_words = n_words
        self.n_tokens = n_tokens

    def completions(self, words, N):
        """
        Return the distribution over proposed next words under the `N`-gram
        language model.

        Parameters
        ----------
        words : list or tuple of strings
            The initial sequence of words
        N : int
            The gram-size of the language model to use to generate completions

        Returns
        -------
        probs : list of (word, log_prob) tuples
            The list of possible next words and their log probabilities under
            the `N`-gram language model (unsorted)
        """
        N = min(N, len(words) + 1)
        assert N in self.counts, "You do not have counts for {}-grams".format(N)
        assert len(words) >= N - 1, "`words` must have at least {} words".format(N - 1)

        probs = []
        base = tuple(w.lower() for w in words[-N + 1 :])
        for k in self.counts[N].keys():
            if k[:-1] == base:
                c_prob = self._log_ngram_prob(base + k[-1:])
                probs.append((k[-1], c_prob))
        return probs

    def generate(self, N, seed_words=["<bol>"], n_sentences=5):
        """
        Use the `N`-gram language model to generate sentences.

        Parameters
        ----------
        N : int
            The gram-size of the model to generate from
        seed_words : list of strs
            A list of seed words to use to condition the initial sentence
            generation. Default is ``["<bol>"]``.
        sentences : int
            The number of sentences to generate from the `N`-gram model.
            Default is 50.

        Returns
        -------
        sentences : str
            Samples from the `N`-gram model, joined by white spaces, with
            individual sentences separated by newlines.
        """
        counter = 0
        sentences = []
        words = seed_words.copy()
        while counter < n_sentences:
            nextw, probs = zip(*self.completions(words, N))
            probs = np.exp(probs) / np.exp(probs).sum()  # renormalize probs if smoothed
            next_word = np.random.choice(nextw, p=probs)

            # if we reach the end of a sentence, save it and start a new one
            if next_word == "<eol>":
                S = " ".join([w for w in words if w != "<bol>"])
                S = textwrap.fill(S, 90, initial_indent="", subsequent_indent="   ")
                print(S)
                words.append(next_word)
                sentences.append(words)
                words = seed_words.copy()
                counter += 1
                continue

            words.append(next_word)
        return sentences

    def perplexity(self, words, N):
        r"""
        Calculate the model perplexity on a sequence of words.

        Notes
        -----
        Perplexity, `PP`, is defined as

        .. math::

            PP(W)  =  \left( \frac{1}{p(W)} \right)^{1 / n}

        or simply

        .. math::

            PP(W)  &=  \exp(-\log p(W) / n) \\
                   &=  \exp(H(W))

        where :math:`W = [w_1, \ldots, w_k]` is a sequence of words, `H(w)` is
        the cross-entropy of `W` under the current model, and `n` is the number
        of `N`-grams in `W`.

        Minimizing perplexity is equivalent to maximizing the probability of
        `words` under the `N`-gram model. It may also be interpreted as the
        average branching factor when predicting the next word under the
        language model.

        Parameters
        ----------
        N : int
            The gram-size of the model to calculate perplexity with.
        words : list or tuple of strings
            The sequence of words to compute perplexity on.

        Returns
        -------
        perplexity : float
            The model perlexity for the words in `words`.
        """
        return np.exp(self.cross_entropy(words, N))

    def cross_entropy(self, words, N):
        r"""
        Calculate the model cross-entropy on a sequence of words against the
        empirical distribution of words in a sample.

        Notes
        -----
        Model cross-entropy, `H`, is defined as

        .. math::

            H(W) = -\frac{\log p(W)}{n}

        where :math:`W = [w_1, \ldots, w_k]` is a sequence of words, and `n` is
        the number of `N`-grams in `W`.

        The model cross-entropy is proportional (not equal, since we use base
        `e`) to the average number of bits necessary to encode `W` under the
        model distribution.

        Parameters
        ----------
        N : int
            The gram-size of the model to calculate cross-entropy on.
        words : list or tuple of strings
            The sequence of words to compute cross-entropy on.

        Returns
        -------
        H : float
            The model cross-entropy for the words in `words`.
        """
        n_ngrams = len(ngrams(words, N))
        return -(1 / n_ngrams) * self.log_prob(words, N)

    def _log_prob(self, words, N):
        """
        Calculate the log probability of a sequence of words under the
        `N`-gram model
        """
        assert N in self.counts, "You do not have counts for {}-grams".format(N)

        if N > len(words):
            err = "Not enough words for a gram-size of {}: {}".format(N, len(words))
            raise ValueError(err)

        total_prob = 0
        for ngram in ngrams(words, N):
            total_prob += self._log_ngram_prob(ngram)
        return total_prob

    def _n_completions(self, words, N):
        """
        Return the number of unique word tokens that could follow the sequence
        `words` under the *unsmoothed* `N`-gram language model.
        """
        assert N in self.counts, "You do not have counts for {}-grams".format(N)
        assert len(words) <= N - 1, "Need > {} words to use {}-grams".format(N - 2, N)

        if isinstance(words, list):
            words = tuple(words)

        base = words[-N + 1 :]
        return len([k[-1] for k in self.counts[N].keys() if k[:-1] == base])

    def _num_grams_with_count(self, C, N):
        """
        Return the number of unique `N`-gram tokens that occur exactly `C`
        times
        """
        assert C > 0
        assert N in self.counts, "You do not have counts for {}-grams".format(N)
        # cache count values for future calls
        if not hasattr(self, "_NC"):
            self._NC = {N: {} for N in range(1, self.N + 1)}
        if C not in self._NC[N]:
            self._NC[N][C] = len([k for k, v in self.counts[N].items() if v == C])
        return self._NC[N][C]

    @abstractmethod
    def log_prob(self, words, N):
        """
        Compute the log probability of a sequence of words under the
        unsmoothed, maximum-likelihood `N`-gram language model.
        """
        raise NotImplementedError

    @abstractmethod
    def _log_ngram_prob(self, ngram):
        """Return the unsmoothed log probability of the ngram"""
        raise NotImplementedError


class MLENGram(NGramBase):
    def __init__(self, N, unk=True, filter_stopwords=True, filter_punctuation=True):
        """
        A simple, unsmoothed N-gram model.

        Parameters
        ----------
        N : int
            The maximum length (in words) of the context-window to use in the
            langauge model. Model will compute all n-grams from 1, ..., N.
        unk : bool
            Whether to include the ``<unk>`` (unknown) token in the LM. Default
            is True.
        filter_stopwords : bool
            Whether to remove stopwords before training. Default is True.
        filter_punctuation : bool
            Whether to remove punctuation before training. Default is True.
        """
        super().__init__(N, unk, filter_stopwords, filter_punctuation)

        self.hyperparameters["id"] = "MLENGram"

    def log_prob(self, words, N):
        """
        Compute the log probability of a sequence of words under the
        unsmoothed, maximum-likelihood `N`-gram language model.

        Parameters
        ----------
        words : list of strings
            A sequence of words
        N : int
            The gram-size of the language model to use when calculating the log
            probabilities of the sequence

        Returns
        -------
        total_prob : float
            The total log-probability of the sequence `words` under the
            `N`-gram language model
        """
        return self._log_prob(words, N)

    def _log_ngram_prob(self, ngram):
        """Return the unsmoothed log probability of the ngram"""
        N = len(ngram)
        num = self.counts[N][ngram]
        den = self.counts[N - 1][ngram[:-1]] if N > 1 else self.n_words[1]
        return np.log(num) - np.log(den) if (den > 0 and num > 0) else -np.inf


class AdditiveNGram(NGramBase):
    def __init__(
        self, N, K=1, unk=True, filter_stopwords=True, filter_punctuation=True,
    ):
        """
        An N-Gram model with smoothed probabilities calculated via additive /
        Lidstone smoothing.

        Notes
        -----
        The resulting estimates correspond to the expected value of the
        posterior, `p(ngram_prob | counts)`, when using a symmetric Dirichlet
        prior on counts with parameter `K`.

        Parameters
        ----------
        N : int
            The maximum length (in words) of the context-window to use in the
            langauge model. Model will compute all n-grams from 1, ..., N
        K : float
            The pseudocount to add to each observation. Larger values allocate
            more probability toward unseen events. When `K` = 1, the model is
            known as Laplace smoothing.  When `K` = 0.5, the model is known as
            expected likelihood estimation (ELE) or the Jeffreys-Perks law.
            Default is 1.
        unk : bool
            Whether to include the ``<unk>`` (unknown) token in the LM. Default
            is True.
        filter_stopwords : bool
            Whether to remove stopwords before training. Default is True.
        filter_punctuation : bool
            Whether to remove punctuation before training. Default is True.
        """
        super().__init__(N, unk, filter_stopwords, filter_punctuation)

        self.hyperparameters["id"] = "AdditiveNGram"
        self.hyperparameters["K"] = K

    def log_prob(self, words, N):
        r"""
        Compute the smoothed log probability of a sequence of words under the
        `N`-gram language model with additive smoothing.

        Notes
        -----
        For a bigram, additive smoothing amounts to:

        .. math::

            P(w_i \mid w_{i-1}) = \frac{A + K}{B + KV}

        where

        .. math::

            A  &=  \text{Count}(w_{i-1}, w_i) \\
            B  &=  \sum_j \text{Count}(w_{i-1}, w_j) \\
            V  &= |\{ w_j \ : \ \text{Count}(w_{i-1}, w_j) > 0 \}|

        This is equivalent to pretending we've seen every possible `N`-gram
        sequence at least `K` times.

        Additive smoothing can be problematic, as it:
            - Treats each predicted word in the same way
            - Can assign too much probability mass to unseen `N`-grams

        Parameters
        ----------
        words : list of strings
            A sequence of words.
        N : int
            The gram-size of the language model to use when calculating the log
            probabilities of the sequence.

        Returns
        -------
        total_prob : float
            The total log-probability of the sequence `words` under the
            `N`-gram language model.
        """
        return self._log_prob(words, N)

    def _log_ngram_prob(self, ngram):
        """Return the smoothed log probability of the ngram"""
        N = len(ngram)
        K = self.hyperparameters["K"]
        counts, n_words, n_tokens = self.counts, self.n_words[1], self.n_tokens[1]

        ctx = ngram[:-1]
        num = counts[N][ngram] + K
        ctx_count = counts[N - 1][ctx] if N > 1 else n_words
        den = ctx_count + K * n_tokens
        return np.log(num / den) if den != 0 else -np.inf


class GoodTuringNGram(NGramBase):
    def __init__(
        self, N, conf=1.96, unk=True, filter_stopwords=True, filter_punctuation=True,
    ):
        """
        An N-Gram model with smoothed probabilities calculated with the simple
        Good-Turing estimator from Gale (2001).

        Parameters
        ----------
        N : int
            The maximum length (in words) of the context-window to use in the
            langauge model. Model will compute all n-grams from 1, ..., N.
        conf: float
            The multiplier of the standard deviation of the empirical smoothed
            count (the default, 1.96, corresponds to a 95% confidence
            interval). Controls how many datapoints are smoothed using the
            log-linear model.
        unk : bool
            Whether to include the ``<unk>`` (unknown) token in the LM. Default
            is True.
        filter_stopwords : bool
            Whether to remove stopwords before training. Default is True.
        filter_punctuation : bool
            Whether to remove punctuation before training. Default is True.
        """
        super().__init__(N, unk, filter_stopwords, filter_punctuation)

        self.hyperparameters["id"] = "GoodTuringNGram"
        self.hyperparameters["conf"] = conf

    def train(self, corpus_fp, vocab=None, encoding=None):
        """
        Compile the n-gram counts for the text(s) in `corpus_fp`. Upon
        completion the `self.counts` attribute will store dictionaries of the
        `N`, `N-1`, ..., 1-gram counts.

        Parameters
        ----------
        corpus_fp : str
            The path to a newline-separated text corpus file
        vocab : :class:`~numpy_ml.preprocessing.nlp.Vocabulary` instance or None.
            If not None, only the words in `vocab` will be used to construct
            the language model; all out-of-vocabulary words will either be
            mappend to ``<unk>`` (if ``self.unk = True``) or removed (if
            ``self.unk = False``). Default is None.
        encoding : str  or None
            Specifies the text encoding for corpus. Common entries are 'utf-8',
            'utf-8-sig', 'utf-16'. Default is None.
        """
        self._train(corpus_fp, vocab=None, encoding=None)
        self._calc_smoothed_counts()

    def log_prob(self, words, N):
        r"""
        Compute the smoothed log probability of a sequence of words under the
        `N`-gram language model with Good-Turing smoothing.

        Notes
        -----
        For a bigram, Good-Turing smoothing amounts to:

        .. math::

            P(w_i \mid w_{i-1}) = \frac{C^*}{\text{Count}(w_{i-1})}

        where :math:`C^*` is the Good-Turing smoothed estimate of the bigram
        count:

        .. math::

            C^* = \frac{(c + 1) \text{NumCounts}(c + 1, 2)}{\text{NumCounts}(c, 2)}

        where

        .. math::

            c  &=  \text{Count}(w_{i-1}, w_i) \\
            \text{NumCounts}(r, k)  &=
                |\{ k\text{-gram} : \text{Count}(k\text{-gram}) = r \}|

        In words, the probability of an `N`-gram that occurs `r` times in the
        corpus is estimated by dividing up the probability mass occupied by
        N-grams that occur `r+1` times.

        For large values of `r`, NumCounts becomes unreliable. In this case, we
        compute a smoothed version of NumCounts using a power law function:

        .. math::

            \log \text{NumCounts}(r) = b + a \log r

        Under the Good-Turing estimator, the total probability assigned to
        unseen `N`-grams is equal to the relative occurrence of `N`-grams that
        appear only once.

        Parameters
        ----------
        words : list of strings
            A sequence of words.
        N : int
            The gram-size of the language model to use when calculating the log
            probabilities of the sequence.

        Returns
        -------
        total_prob : float
            The total log-probability of the sequence `words` under the
            `N`-gram language model.
        """
        return self._log_prob(words, N)

    def _calc_smoothed_counts(self):
        use_interp = False
        counts = self.counts
        NC = self._num_grams_with_count
        conf = self.hyperparameters["conf"]

        totals = {N: 0 for N in range(1, self.N + 1)}
        smooth_counts = {N: {} for N in range(1, self.N + 1)}

        # calculate the probability of all <unk> (i.e., unseen) n-grams
        self._p0 = {n: NC(1, n) / sum(counts[n].values()) for n in range(1, self.N + 1)}

        # fit log-linear models for predicting smoothed counts in absence of
        # real data
        self._fit_count_models()

        LM = self._count_models
        for N in range(1, self.N + 1):
            for C in sorted(set(counts[N].values())):

                # estimate the interpolated count using the log-linear model
                c1_lm = np.exp(LM[N].predict(np.c_[np.log(C + 1)])).item()
                c0_lm = np.exp(LM[N].predict(np.c_[np.log(C)])).item()
                count_interp = ((C + 1) * c1_lm) / c0_lm

                # if we have previously been using the interpolated count, or
                # if the number of ocurrences of C+1 is 0, use the interpolated
                # count as the smoothed count value C*
                c1, c0 = NC(C + 1, N), NC(C, N)
                if use_interp or c1 == 0:
                    use_interp = True
                    smooth_counts[N][C] = count_interp
                    totals[N] += c0 * smooth_counts[N][C]
                    continue

                # estimate the smoothed count C* empirically if the number of
                # terms with count C + 1 > 0
                count_emp = ((C + 1) * c1) / c0

                # compute the approximate variance of the empirical smoothed
                # count C* given C
                t = conf * np.sqrt((C + 1) ** 2 * (c1 / c0 ** 2) * (1 + c1 / c0))

                # if the difference between the empirical and interpolated
                # smoothed counts is greater than t, the empirical estimate
                # tends to be more accurate. otherwise, use interpolated
                if np.abs(count_interp - count_emp) > t:
                    smooth_counts[N][C] = count_emp
                    totals[N] += c0 * smooth_counts[N][C]
                    continue

                use_interp = True
                smooth_counts[N][C] = count_interp
                totals[N] += c0 * smooth_counts[N][C]

        self._smooth_totals = totals
        self._smooth_counts = smooth_counts

    def _log_ngram_prob(self, ngram):
        """Return the smoothed log probability of the ngram"""
        N = len(ngram)
        sc, T = self._smooth_counts[N], self._smooth_totals[N]
        n_tokens, n_seen = self.n_tokens[N], len(self.counts[N])

        # approx. prob of an out-of-vocab ngram (i.e., a fraction of p0)
        n_unseen = max((n_tokens ** N) - n_seen, 1)
        prob = np.log(self._p0[N] / n_unseen)

        if ngram in self.counts[N]:
            C = self.counts[N][ngram]
            prob = np.log(1 - self._p0[N]) + np.log(sc[C]) - np.log(T)
        return prob

    def _fit_count_models(self):
        """
        Perform the averaging transform proposed by Church and Gale (1991):
        estimate the expected count-of-counts by the *density* of
        count-of-count values.
        """
        self._count_models = {}
        NC = self._num_grams_with_count
        for N in range(1, self.N + 1):
            X, Y = [], []
            sorted_counts = sorted(set(self.counts[N].values()))  # r

            for ix, j in enumerate(sorted_counts):
                i = 0 if ix == 0 else sorted_counts[ix - 1]
                k = 2 * j - i if ix == len(sorted_counts) - 1 else sorted_counts[ix + 1]
                y = 2 * NC(j, N) / (k - i)
                X.append(j)
                Y.append(y)

            # fit log-linear model: log(counts) ~ log(average_transform(counts))
            self._count_models[N] = LinearRegression(fit_intercept=True)
            self._count_models[N].fit(np.log(X), np.log(Y))
            b, a = self._count_models[N].beta

            if a > -1:
                fstr = "[Warning] Log-log averaging transform has slope > -1 for N={}"
                print(fstr.format(N))
