import sys
from collections import Counter

import huffman
import numpy as np

from sklearn.datasets import fetch_20newsgroups
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

from nlp import HuffmanEncoder, TFIDFEncoder
from general import Standardizer


sys.path.append("..")
from utils.testing import random_paragraph


def load_newsgroups():
    categories = ["alt.atheism", "talk.religion.misc", "comp.graphics", "sci.space"]
    dataset = fetch_20newsgroups(
        subset="all", categories=categories, shuffle=True, random_state=42
    )
    return dataset.data, dataset.target


def test_huffman():
    while True:
        n_words = np.random.randint(1, 100)
        para = random_paragraph(n_words)
        HT = HuffmanEncoder()
        HT.fit(para)
        my_dict = HT._item2code
        their_dict = huffman.codebook(Counter(para).items())

        for k, v in their_dict.items():
            fstr = "their_dict['{}'] = {}, but my_dict['{}'] = {}"
            assert k in my_dict, "key `{}` not in my_dict".format(k)
            assert my_dict[k] == v, fstr.format(k, v, k, my_dict[k])
        print("PASSED")


def test_standardizer():
    while True:
        mean = bool(np.random.randint(2))
        std = bool(np.random.randint(2))
        N = np.random.randint(2, 100)
        M = np.random.randint(2, 100)
        X = np.random.rand(N, M)

        S = Standardizer(with_mean=mean, with_std=std)
        S.fit(X)
        mine = S.transform(X)

        theirs = StandardScaler(with_mean=mean, with_std=std)
        gold = theirs.fit_transform(X)

        np.testing.assert_almost_equal(mine, gold)
        print("PASSED")


def test_tfidf():
    while True:
        docs = []
        n_docs = np.random.randint(1, 10)
        for d in range(n_docs):
            n_lines = np.random.randint(1, 1000)
            lines = [random_paragraph(np.random.randint(1, 10)) for _ in range(n_lines)]
            docs.append("\n".join([" ".join(l) for l in lines]))

        smooth = bool(np.random.randint(2))

        tfidf = TFIDFEncoder(
            lowercase=True,
            min_count=0,
            smooth_idf=smooth,
            max_tokens=None,
            input_type="strings",
            filter_stopwords=False,
        )
        gold = TfidfVectorizer(
            input="content",
            norm=None,
            use_idf=True,
            lowercase=True,
            smooth_idf=smooth,
            sublinear_tf=False,
        )

        tfidf.fit(docs)
        mine = tfidf.transform(ignore_special_chars=True)
        theirs = gold.fit_transform(docs).toarray()

        np.testing.assert_almost_equal(mine, theirs)
        print("PASSED")
