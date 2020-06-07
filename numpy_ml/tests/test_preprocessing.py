# flake8: noqa
from collections import Counter

# gold-standard imports
import huffman
import numpy as np

from scipy.fftpack import dct

from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

from librosa.core.time_frequency import fft_frequencies
from librosa.feature import mfcc as lr_mfcc
from librosa.util import frame
from librosa.filters import mel

# numpy-ml implementations
from numpy_ml.preprocessing.general import Standardizer
from numpy_ml.preprocessing.nlp import HuffmanEncoder, TFIDFEncoder
from numpy_ml.preprocessing.dsp import (
    DCT,
    DFT,
    mfcc,
    to_frames,
    mel_filterbank,
    dft_bins,
)
from numpy_ml.utils.testing import random_paragraph


def test_huffman(N=15):
    np.random.seed(12345)

    i = 0
    while i < N:
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
        i += 1


def test_standardizer(N=15):
    np.random.seed(12345)

    i = 0
    while i < N:
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
        i += 1


def test_tfidf(N=15):
    np.random.seed(12345)

    i = 0
    while i < N:
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
        i += 1


def test_dct(N=15):
    np.random.seed(12345)

    i = 0
    while i < N:
        N = np.random.randint(2, 100)
        signal = np.random.rand(N)
        ortho = bool(np.random.randint(2))
        mine = DCT(signal, orthonormal=ortho)
        theirs = dct(signal, norm="ortho" if ortho else None)

        np.testing.assert_almost_equal(mine, theirs)
        print("PASSED")
        i += 1


def test_dft(N=15):
    np.random.seed(12345)

    i = 0
    while i < N:
        N = np.random.randint(2, 100)
        signal = np.random.rand(N)
        mine = DFT(signal)
        theirs = np.fft.rfft(signal)

        np.testing.assert_almost_equal(mine.real, theirs.real)
        print("PASSED")
        i += 1


def test_mfcc(N=1):
    """Broken"""
    np.random.seed(12345)

    i = 0
    while i < N:
        N = np.random.randint(500, 1000)
        fs = np.random.randint(50, 100)
        n_mfcc = 12
        window_len = 100
        stride_len = 50
        n_filters = 20
        window_dur = window_len / fs
        stride_dur = stride_len / fs
        signal = np.random.rand(N)

        mine = mfcc(
            signal,
            fs=fs,
            window="hann",
            window_duration=window_dur,
            stride_duration=stride_dur,
            lifter_coef=0,
            alpha=0,
            n_mfccs=n_mfcc,
            normalize=False,
            center=True,
            n_filters=n_filters,
            replace_intercept=False,
        )

        theirs = lr_mfcc(
            signal,
            sr=fs,
            n_mels=n_filters,
            n_mfcc=n_mfcc,
            n_fft=window_len,
            hop_length=stride_len,
            htk=True,
        ).T

        np.testing.assert_almost_equal(mine, theirs, decimal=4)
        print("PASSED")
        i += 1


def test_framing(N=15):
    np.random.seed(12345)

    i = 0
    while i < N:
        N = np.random.randint(500, 100000)
        window_len = np.random.randint(10, 100)
        stride_len = np.random.randint(1, 50)
        signal = np.random.rand(N)

        mine = to_frames(signal, window_len, stride_len, writeable=False)
        theirs = frame(signal, frame_length=window_len, hop_length=stride_len).T

        assert len(mine) == len(theirs), "len(mine) = {}, len(theirs) = {}".format(
            len(mine), len(theirs)
        )
        np.testing.assert_almost_equal(mine, theirs)
        print("PASSED")
        i += 1


def test_dft_bins(N=15):
    np.random.seed(12345)

    i = 0
    while i < N:
        N = np.random.randint(500, 100000)
        fs = np.random.randint(50, 1000)

        mine = dft_bins(N, fs=fs, positive_only=True)
        theirs = fft_frequencies(fs, N)
        np.testing.assert_almost_equal(mine, theirs)
        print("PASSED")
        i += 1


def test_mel_filterbank(N=15):
    np.random.seed(12345)

    i = 0
    while i < N:
        fs = np.random.randint(50, 10000)
        n_filters = np.random.randint(2, 20)
        window_len = np.random.randint(10, 100)
        norm = np.random.randint(2)

        mine = mel_filterbank(
            window_len, n_filters, fs, min_freq=0, max_freq=None, normalize=bool(norm)
        )

        theirs = mel(
            fs,
            n_fft=window_len,
            n_mels=n_filters,
            htk=True,
            norm=norm if norm == 1 else None,
        )

        np.testing.assert_almost_equal(mine, theirs)
        print("PASSED")
        i += 1
