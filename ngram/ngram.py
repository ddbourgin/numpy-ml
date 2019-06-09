import sys
import textwrap
from abc import ABC, abstractmethod
from collections import Counter

sys.path.append("..")

import numpy as np

from preprocessing.nlp import tokenize_words, ngrams


class NGramBase(ABC):
    def __init__(self, N, unk=True, filter_stopwords=True, filter_punctuation=True):
        """
        A simple N-gram language model.

        NB. This is not optimized code and will be slow for large corpora. To
        see how industry-scale NGram models are handled, see the SRLIM-format:

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
        Compile the n-gram counts for the text(s) in `corpus_fp`. Upon
        completion the `self.counts` attribute will store dictionaries of the
        N, N-1, ..., 1-gram counts.

        Parameters
        ----------
        corpus_fp : str
            The path to a newline-separated text corpus file
        vocab : `preprocessing.nlp.Vocabulary` instance (default: None)
            If not `None`, only the words in `vocab` will be used to construct
            the language model
        encoding : str (default: None)
            Specifies the text encoding for corpus. Common entries are 'utf-8',
            'utf-8-sig', 'utf-16'.
        """
        H = self.hyperparameters
        grams = {N: [] for N in range(1, self.N + 1)}
        counts = {N: Counter() for N in range(1, self.N + 1)}
        filter_punc, filter_stop = H["filter_punctuation"], H["filter_stopwords"]

        _n_words = 0
        tokens = set(["<unk>"])
        bol, eol = ["<bol>"], ["<eol>"]

        with open(corpus_fp, "r", encoding=encoding) as text:
            for line in text:
                words = tokenize_words(line, filter_punc, filter_stop)

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
        seed_words : list of strs (default: ["<bol>"])
            A list of seed words to use to condition the initial sentence
            generation
        sentences : int (default : 50)
            The number of sentences to generate from the `N`-gram model

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
            next_word = np.random.choice(nextw, p=np.exp(probs))

            # if we reach the end of a sentence, save it and start a new one
            if next_word == "<eol>":
                S = " ".join([w for w in words if w != "<bol>"])
                S = textwrap.fill(S, 90, initial_indent="", subsequent_indent="   ")
                print(S)
                sentences.append(words)
                words = seed_words.copy()
                counter += 1
                continue

            words.append(next_word)
        return sentences

    def _log_prob(self, words, N):
        """Calculate the log probability of a sequence of words under the `N`-gram model"""
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
        raise NotImplementedError

    @abstractmethod
    def _log_ngram_prob(self, ngram):
        raise NotImplementedError


class MLENGram(NGramBase):
    def __init__(self, N, unk=True, filter_stopwords=True, filter_punctuation=True):
        """
        A simple, unsmoothed N-gram model.

        Parameters
        ----------
        N : int
            The maximum length (in words) of the context-window to use in the
            langauge model. Model will compute all n-grams from 1, ..., N
        unk : bool (default: True)
            Whether to include the <unk> (unknown) token in the LM
        filter_stopwords : bool (default: True)
            Whether to remove stopwords before training
        filter_punctuation : bool (default: True)
            Whether to remove punctuation before training
        """
        super().__init__(N, unk, filter_stopwords, filter_punctuation)
        self.hyperparameters["id"] = "MLENGram"

    def log_prob(self, words, N):
        """
        Compute the log probability of a sequence of words under the
        unsmoothed, maximum-likelihood `N`-gram language model. For a bigram,
        this amounts to:

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
        self, N, K=1, unk=True, filter_stopwords=True, filter_punctuation=True
    ):
        """
        An N-Gram model with smoothed probabilities calculated via additive /
        Lidstone smoothing. The resulting estimates correspond to the expected
        value of the posterior, p(ngram_prob | counts), when using a symmetric
        Dirichlet prior on counts with parameter K.

        Parameters
        ----------
        N : int
            The maximum length (in words) of the context-window to use in the
            langauge model. Model will compute all n-grams from 1, ..., N
        K : float (default: 1)
            The pseudocount to add to each observation. Larger values allocate
            more probability toward unseen events. When K = 1, the model is
            known as Laplace smoothing.  When K = 0.5, the model is known as
            expected likelihood estimation (ELE) or the Jeffreys-Perks law
        unk : bool (default: True)
            Whether to include the <unk> (unknown) token in the LM
        filter_stopwords : bool (default: True)
            Whether to remove stopwords before training
        filter_punctuation : bool (default: True)
            Whether to remove punctuation before training
        """
        super().__init__(N, unk, filter_stopwords, filter_punctuation)
        self.hyperparameters["id"] = "AdditiveNGram"
        self.hyperparameters["K"] = K

    def log_prob(self, words, N):
        """
        Compute the smoothed log probability of a sequence of words under the
        `N`-gram language model with additive smoothing. For a bigram, this
        amounts to:

            P(w_i | w_{i-1}) = (A + K) / (B + K * V)

        where

            A = Count(w_{i-1}, w_i)
            B = sum_j Count(w_{i-1}, w_j)
            V = |{ w_j : Count(w_{i-1}, w_j) > 0 }|

        This is equivalent to pretending we've seen every possible N-gram
        sequence at least `K` times. This can be problematic, as it:
            - Treats each predicted word in the same way (uniform prior counts)
            - Can assign too much probability mass to unseen N-grams (too aggressive)

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
        """Return the smoothed log probability of the ngram"""
        N = len(ngram)
        K = self.hyperparameters["K"]
        counts, n_words, n_tokens = self.counts, self.n_words[1], self.n_tokens[1]

        ctx = ngram[:-1]
        ctx_count = counts[N - 1][ctx] if N > 1 else n_words
        num = counts[N][ngram] + K
        den = ctx_count + K * n_tokens
        return np.log(num / den) if den != 0 else -np.inf
