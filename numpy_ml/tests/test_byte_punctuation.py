import pytest

from numpy_ml.preprocessing.nlp import tokenize_bytes_raw


@pytest.mark.parametrize("splitter", [None, "punctuation"])
@pytest.mark.parametrize("text", ["café!", "🙂?", "Ω[ü]", "all ascii!", "123,456", ""])
def test_byte_tokenization_preserves_complete_utf8_bytes(text, splitter):
    pieces = tokenize_bytes_raw(text, splitter=splitter)
    reconstructed = bytes(int(value) for piece in pieces for value in piece.split())

    assert reconstructed == text.encode("utf-8")
    assert reconstructed.decode("utf-8") == text


def test_byte_punctuation_splits_only_complete_punctuation_bytes():
    actual = tokenize_bytes_raw("café,🙂!", splitter="punctuation")
    groups = [
        [int(value) for value in piece.split()] for piece in actual if piece.strip()
    ]

    assert groups == [[99, 97, 102, 195, 169], [44], [240, 159, 153, 130], [33]]
