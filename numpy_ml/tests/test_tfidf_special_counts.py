import numpy as np
import pytest

from numpy_ml.preprocessing.nlp import TFIDFEncoder


SPECIALS = ["<eol>", "<bol>", "<unk>"]


@pytest.mark.parametrize("smooth", [False, True])
@pytest.mark.parametrize(
    "documents, options, counts",
    [
        (
            ["blue sky", "blue ocean"],
            {},
            [{"blue": 1, "sky": 1}, {"blue": 1, "ocean": 1}],
        ),
        (
            ["blue\nsky", "blue"],
            {},
            [{"blue": 1, "sky": 1, "<eol>": 2, "<bol>": 2}, {"blue": 1}],
        ),
        (
            ["red red", "green blue"],
            {"vocab": ["blue"]},
            [{"<unk>": 2}, {"<unk>": 1, "blue": 1}],
        ),
        (
            ["blue rare", "blue"],
            {"min_count": 2},
            [{"blue": 1, "<unk>": 1}, {"blue": 1}],
        ),
        (["", "blue"], {}, [{}, {"blue": 1}]),
    ],
)
def test_tfidf_reindexes_special_counts_with_ordinary_tokens(
    smooth, documents, options, counts
):
    counts = [{"<bol>": 1, "<eol>": 1, **row} for row in counts]
    ordinary = sorted(set().union(*counts) - set(SPECIALS))
    tokens = ordinary + SPECIALS
    frequencies = np.array([[row.get(token, 0) for token in tokens] for row in counts])
    document_counts = np.count_nonzero(frequencies, axis=0) + int(smooth)
    idf = np.ones(len(tokens))
    present = document_counts > 0
    idf[present] = np.log((len(documents) + int(smooth)) / document_counts[present]) + 1
    expected = frequencies * idf
    encoder = TFIDFEncoder(
        input_type="strings", filter_stopwords=False, smooth_idf=smooth, **options
    ).fit(documents)

    assert encoder.token2idx == {token: index for index, token in enumerate(tokens)}
    np.testing.assert_allclose(encoder.transform(ignore_special_chars=False), expected)
    np.testing.assert_allclose(encoder.transform(), expected[:, : len(ordinary)])
    for index, row in enumerate(counts):
        assert encoder.term_freq[index] == {
            encoder.token2idx[token]: count for token, count in row.items()
        }
