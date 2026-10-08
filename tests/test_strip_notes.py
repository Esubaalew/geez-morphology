"""The trainer and evaluate_gold.py read gold geez_form cells the same way (gold_filters.clean_variants):
"(...)" notes dropped, split on "/", edge punctuation stripped."""
import json
import os
import sys
import tempfile

import pytest

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, HERE)

import gold_filters as G  # noqa: E402

DATASET = G.default_dataset_dir()


def test_clean_variants():
    assert G.clean_variants("ቀተለ") == (["ቀተለ"], [])
    assert G.clean_variants("ቀተለ / ቀቲል ፡") == (["ቀተለ", "ቀቲል"], [])
    assert G.clean_variants("ዐየሰ (ዓሰ") == ([], ["ዐየሰ (ዓሰ"])       # unclosed note: not one word
    assert G.clean_variants("ዐየሰ (ዓሰ)") == (["ዐየሰ"], [])
    assert G.clean_variants("ወኮኑ (ወኮኑ ሰብአ ፋርስ መላእክት)") == (["ወኮኑ"], [])
    assert G.clean_variants("ይፀመድ [ይፀመደኒ]") == ([], ["ይፀመድ [ይፀመደኒ"])
    assert G.clean_variants("qatala / ቀተለ") == (["ቀተለ"], ["qatala"])
    assert G.clean_variants("") == ([], []) and G.clean_variants(None) == ([], [])


def test_trainer_and_evaluate_gold_read_the_same_forms():
    import train_byt5_morphology as T
    import evaluate_gold as E
    rec = {"record_id": "aaaaaaaaaaaa", "page_number": 1, "title_or_lemma": "ቀተለ", "root": "ቀ-ተ-ለ",
           "category": "verb_conjugation", "stem": "basic",
           "table_data": [{"geez_form": f, "tense": "past", "person": "3", "gender": "m", "number": "s"}
                          for f in ("ቀተለ (note)", "ቀተልኩ / ቀተልኩ ፡", "ቀተልክሙ")]}
    with tempfile.TemporaryDirectory() as tmp:
        para = os.path.join(tmp, "data", "paradigms"); os.makedirs(para)
        open(os.path.join(para, "synthetic_morphology_train.jsonl"), "w").close()
        gold = os.path.join(para, "merho_ges_gold_train.jsonl")
        with open(gold, "w", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        train = sorted(x["input_text"][len("analyze: "):] for x in T.load_data(tmp))
        evals = sorted(p["surface"] for p in E.load_gold_test_pairs(gold)[0])
    assert train == evals == sorted(["ቀተለ", "ቀተልኩ", "ቀተልኩ", "ቀተልክሙ"])


def test_real_gold_inputs_are_single_words():
    if not os.path.exists(os.path.join(DATASET, "data", "paradigms", "merho_ges_gold_train.jsonl")):
        pytest.skip("no geez-dataset checkout")
    for split in ("train", "val", "test"):
        forms = [p["form"] for p in G.load_gold_split(DATASET, split)]
        assert forms and all(G.is_single_ethiopic_word(v) for v in forms)
