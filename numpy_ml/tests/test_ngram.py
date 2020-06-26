# flake8: noqa
import tempfile

import nltk
import numpy as np

from ..preprocessing.nlp import tokenize_words
from ..ngram import AdditiveNGram, MLENGram
from ..utils.testing import random_paragraph


class MLEGold:
    def __init__(
        self, N, K=1, unk=True, filter_stopwords=True, filter_punctuation=True
    ):
        self.N = N
        self.K = K
        self.unk = unk
        self.filter_stopwords = filter_stopwords
        self.filter_punctuation = filter_punctuation

        self.hyperparameters = {
            "N": N,
            "K": K,
            "unk": unk,
            "filter_stopwords": filter_stopwords,
            "filter_punctuation": filter_punctuation,
        }

    def train(self, corpus_fp, vocab=None, encoding=None):
        N = self.N
        H = self.hyperparameters
        models, counts = {}, {}
        grams = {n: [] for n in range(1, N + 1)}
        gg = {n: [] for n in range(1, N + 1)}
        filter_punc, filter_stop = H["filter_punctuation"], H["filter_stopwords"]

        n_words = 0
        tokens = set([])

        with open(corpus_fp, "r", encoding=encoding) as text:
            for line in text:
                words = tokenize_words(line, filter_punc, filter_stop)

                if vocab is not None:
                    words = vocab.filter(words, H["unk"])

                if len(words) == 0:
                    continue

                n_words += len(words)
                tokens.update(words)

                # calculate n, n-1, ... 1-grams
                for n in range(1, N + 1):
                    grams[n].append(
                        nltk.ngrams(
                            words,
                            n,
                            pad_left=True,
                            pad_right=True,
                            left_pad_symbol="<bol>",
                            right_pad_symbol="<eol>",
                        )
                    )

                    gg[n].extend(
                        list(
                            nltk.ngrams(
                                words,
                                n,
                                pad_left=True,
                                pad_right=True,
                                left_pad_symbol="<bol>",
                                right_pad_symbol="<eol>",
                            )
                        )
                    )

        for n in range(1, N + 1):
            counts[n] = nltk.FreqDist(gg[n])
            models[n] = nltk.lm.MLE(order=n)
            models[n].fit(grams[n], tokens)

        self.counts = counts
        self.n_words = n_words
        self._models = models
        self.n_tokens = len(vocab) if vocab is not None else len(tokens)

    def log_prob(self, words, N):
        assert N in self.counts, "You do not have counts for {}-grams".format(N)

        if N > len(words):
            err = "Not enough words for a gram-size of {}: {}".format(N, len(words))
            raise ValueError(err)

        total_prob = 0
        for ngram in nltk.ngrams(words, N):
            total_prob += self._log_ngram_prob(ngram)
        return total_prob

    def _log_ngram_prob(self, ngram):
        N = len(ngram)
        return self._models[N].logscore(ngram[-1], ngram[:-1])


class AdditiveGold:
    def __init__(
        self, N, K=1, unk=True, filter_stopwords=True, filter_punctuation=True
    ):
        self.N = N
        self.K = K
        self.unk = unk
        self.filter_stopwords = filter_stopwords
        self.filter_punctuation = filter_punctuation

        self.hyperparameters = {
            "N": N,
            "K": K,
            "unk": unk,
            "filter_stopwords": filter_stopwords,
            "filter_punctuation": filter_punctuation,
        }

    def train(self, corpus_fp, vocab=None, encoding=None):
        N = self.N
        H = self.hyperparameters
        models, counts = {}, {}
        grams = {n: [] for n in range(1, N + 1)}
        gg = {n: [] for n in range(1, N + 1)}
        filter_punc, filter_stop = H["filter_punctuation"], H["filter_stopwords"]

        n_words = 0
        tokens = set()

        with open(corpus_fp, "r", encoding=encoding) as text:
            for line in text:
                words = tokenize_words(line, filter_punc, filter_stop)

                if vocab is not None:
                    words = vocab.filter(words, H["unk"])

                if len(words) == 0:
                    continue

                n_words += len(words)
                tokens.update(words)

                # calculate n, n-1, ... 1-grams
                for n in range(1, N + 1):
                    grams[n].append(
                        nltk.ngrams(
                            words,
                            n,
                            pad_left=True,
                            pad_right=True,
                            left_pad_symbol="<bol>",
                            right_pad_symbol="<eol>",
                        )
                    )

                    gg[n].extend(
                        list(
                            nltk.ngrams(
                                words,
                                n,
                                pad_left=True,
                                pad_right=True,
                                left_pad_symbol="<bol>",
                                right_pad_symbol="<eol>",
                            )
                        )
                    )

        for n in range(1, N + 1):
            counts[n] = nltk.FreqDist(gg[n])
            models[n] = nltk.lm.Lidstone(order=n, gamma=self.K)
            models[n].fit(grams[n], tokens)

        self.counts = counts
        self._models = models
        self.n_words = n_words
        self.n_tokens = len(vocab) if vocab is not None else len(tokens)

    def log_prob(self, words, N):
        assert N in self.counts, "You do not have counts for {}-grams".format(N)

        if N > len(words):
            err = "Not enough words for a gram-size of {}: {}".format(N, len(words))
            raise ValueError(err)

        total_prob = 0
        for ngram in nltk.ngrams(words, N):
            total_prob += self._log_ngram_prob(ngram)
        return total_prob

    def _log_ngram_prob(self, ngram):
        N = len(ngram)
        return self._models[N].logscore(ngram[-1], ngram[:-1])


def test_mle():
    N = np.random.randint(2, 5)
    gold = MLEGold(N, unk=True, filter_stopwords=False, filter_punctuation=False)
    mine = MLENGram(N, unk=True, filter_stopwords=False, filter_punctuation=False)

    with tempfile.NamedTemporaryFile() as temp:
        temp.write(bytes(" ".join(random_paragraph(1000)), encoding="utf-8-sig"))
        gold.train(temp.name, encoding="utf-8-sig")
        mine.train(temp.name, encoding="utf-8-sig")

    for k in mine.counts[N].keys():
        if k[0] == k[1] and k[0] in ("<bol>", "<eol>"):
            continue

        err_str = "{}, mine: {}, gold: {}"
        assert mine.counts[N][k] == gold.counts[N][k], err_str.format(
            k, mine.counts[N][k], gold.counts[N][k]
        )

        M = mine.log_prob(k, N)
        G = gold.log_prob(k, N) / np.log2(np.e)  # convert to log base e
        np.testing.assert_allclose(M, G)
        print("PASSED")


def test_additive():
    K = np.random.rand()
    N = np.random.randint(2, 5)
    gold = AdditiveGold(
        N, K, unk=True, filter_stopwords=False, filter_punctuation=False
    )
    mine = AdditiveNGram(
        N, K, unk=True, filter_stopwords=False, filter_punctuation=False
    )

    with tempfile.NamedTemporaryFile() as temp:
        temp.write(bytes(" ".join(random_paragraph(1000)), encoding="utf-8-sig"))
        gold.train(temp.name, encoding="utf-8-sig")
        mine.train(temp.name, encoding="utf-8-sig")

    for k in mine.counts[N].keys():
        if k[0] == k[1] and k[0] in ("<bol>", "<eol>"):
            continue

        err_str = "{}, mine: {}, gold: {}"
        assert mine.counts[N][k] == gold.counts[N][k], err_str.format(
            k, mine.counts[N][k], gold.counts[N][k]
        )

        M = mine.log_prob(k, N)
        G = gold.log_prob(k, N) / np.log2(np.e)  # convert to log base e
        np.testing.assert_allclose(M, G)
        print("PASSED")
