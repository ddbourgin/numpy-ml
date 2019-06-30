from collections import Counter
import numpy as np

from general import Standardizer
from nlp import HuffmanEncoder

import huffman
from sklearn.preprocessing import StandardScaler


def random_paragraph(n_words):
    words = [
        "lorem",
        "ipsum",
        "dolor",
        "sit",
        "amet",
        "consetetur",
        "sadipscing",
        "elitr",
        "sed",
        "diam",
        "nonumy",
        "eirmod",
    ]
    return [np.random.choice(words) for _ in range(n_words)]


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
