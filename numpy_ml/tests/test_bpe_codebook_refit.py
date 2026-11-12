import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from numpy_ml.preprocessing.nlp import BytePairEncoder


@pytest.mark.parametrize("second_budget", [0, 1])
def test_refit_replaces_old_byte_mappings(tmp_path, second_budget):
    first_path = tmp_path / "first.txt"
    second_path = tmp_path / "second.txt"
    first_path.write_text("abababab", encoding="utf-8")
    second_path.write_text("xyxyxyxy", encoding="utf-8")
    encoder = BytePairEncoder(max_merges=1).fit(str(first_path))
    encoder.parameters["max_merges"] = second_budget

    encoder.fit(str(second_path))
    reference = BytePairEncoder(max_merges=second_budget).fit(str(second_path))
    texts = ["abab", "xyxy", "unseen: Ω🙂"]

    assert encoder.byte2token == reference.byte2token
    assert encoder.token2byte == reference.token2byte
    assert encoder.transform(texts) == reference.transform(texts)
    assert encoder.inverse_transform(encoder.transform(texts)) == texts


def test_learned_token_ids_do_not_depend_on_python_hash_seed(tmp_path):
    corpus_path = tmp_path / "corpus.txt"
    corpus_path.write_text("abababab\nxyxyxyxy\nmnmnmnmn\n", encoding="utf-8")
    root = Path(__file__).resolve().parents[2]
    code = """
import json
import sys
from numpy_ml.preprocessing.nlp import BytePairEncoder
encoder = BytePairEncoder(max_merges=3).fit(sys.argv[1])
print(json.dumps(list(encoder.token2byte.items())))
"""
    codebooks = []
    for seed in [1, 7, 11]:
        env = dict(os.environ, PYTHONHASHSEED=str(seed), PYTHONPATH=str(root))
        output = subprocess.check_output(
            [sys.executable, "-c", code, str(corpus_path)], cwd=root, env=env, text=True
        )
        codebooks.append(json.loads(output))

    assert len(codebooks[0]) > 257
    assert codebooks[1:] == [codebooks[0], codebooks[0]]
