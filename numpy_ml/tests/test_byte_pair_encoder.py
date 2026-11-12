import pytest

from numpy_ml.preprocessing.nlp import BytePairEncoder


@pytest.mark.parametrize("max_merges", [None, 100])
@pytest.mark.parametrize(
    "corpus_text", ["", "a", "abababab", "hello hello\nworld", "汉字🙂\n"]
)
def test_fit_finishes_when_available_pairs_are_exhausted(
    tmp_path, corpus_text, max_merges
):
    corpus_path = tmp_path / "corpus.txt"
    corpus_path.write_text(corpus_text, encoding="utf-8")
    encoder = (
        BytePairEncoder()
        if max_merges is None
        else BytePairEncoder(max_merges=max_merges)
    )

    assert encoder.fit(str(corpus_path)) is encoder

    # Learned tokens still compress the training text; the base alphabet
    # keeps unobserved UTF-8 bytes encodable after early termination.
    texts = ([corpus_text] if corpus_text else []) + ["unseen text: Ω🙂"]
    encoded = encoder.transform(texts)
    assert encoder.inverse_transform(encoded) == texts
    assert set(range(256)).issubset(encoder.tokens)
    if len(corpus_text.encode("utf-8")) > 1:
        assert len(encoded[0]) < len(corpus_text.encode("utf-8"))
    else:
        assert encoder.tokens == list(range(256))


@pytest.mark.parametrize(
    "max_merges, expected",
    [(0, [97, 98, 97, 98, 97, 98, 97, 98]), (1, [256, 256, 256, 256])],
)
def test_fit_still_respects_the_merge_budget(tmp_path, max_merges, expected):
    corpus_path = tmp_path / "corpus.txt"
    corpus_path.write_text("abababab", encoding="utf-8")

    encoder = BytePairEncoder(max_merges=max_merges).fit(str(corpus_path))

    assert encoder.transform("abababab") == [expected]
    assert encoder.inverse_transform([expected]) == ["abababab"]
