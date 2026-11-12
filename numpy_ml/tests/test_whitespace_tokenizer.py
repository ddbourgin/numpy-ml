import numpy as np
import pytest

from numpy_ml.preprocessing.nlp import TFIDFEncoder, Vocabulary, tokenize_whitespace


@pytest.mark.parametrize("lowercase", [False, True])
@pytest.mark.parametrize("filter_stopwords", [False, True])
@pytest.mark.parametrize("filter_punctuation", [False, True])
def test_whitespace_tokenizer_applies_requested_filters(
    lowercase, filter_stopwords, filter_punctuation
):
    if filter_punctuation:
        expected = ["The", "Cats", "DOGS"]
    else:
        expected = ["The,", "Cats!", "...", "DOGS?"]
    if lowercase:
        expected = [word.lower() for word in expected]
    if filter_stopwords and filter_punctuation:
        expected = expected[1:]

    actual = tokenize_whitespace(
        "The, Cats! ... DOGS?",
        lowercase=lowercase,
        filter_stopwords=filter_stopwords,
        filter_punctuation=filter_punctuation,
    )

    assert actual == expected


def test_punctuation_only_tokens_are_removed():
    assert tokenize_whitespace("!!! ...\t", filter_stopwords=False) == []


def test_vocabulary_counts_normalized_whitespace_tokens(tmp_path):
    corpus_path = tmp_path / "corpus.txt"
    corpus_path.write_text("The, cats! ... dogs? cats!", encoding="utf-8")

    vocabulary = Vocabulary(tokenizer="whitespace").fit(str(corpus_path))

    assert vocabulary.counts["cats"] == 2
    assert vocabulary.counts["dogs"] == 1
    assert set(vocabulary.counts) == {"cats", "dogs", "<bol>", "<eol>", "<unk>"}


def test_tfidf_uses_normalized_whitespace_terms():
    encoder = TFIDFEncoder(input_type="strings", tokenizer="whitespace").fit(
        ["the, cats!", "dogs? cats!"]
    )

    expected = np.array([[1.0, 0.0], [1.0, np.log(3.0 / 2.0) + 1]])
    np.testing.assert_allclose(encoder.transform(), expected)
