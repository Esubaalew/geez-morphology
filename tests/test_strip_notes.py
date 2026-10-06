"""The trainer reads gold geez_form cells the way evaluate_gold.py does (gold_filters.clean_variants):
"(...)" notes dropped, split on "/", only single Ethiopic words kept."""
import collections
import json
import os
import sys
import tempfile

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gold_filters as G
from test_gold_filters import write_tables

DATASET = os.path.normpath(os.path.join(HERE, "..", "geez-dataset-v1"))


def test_clean_variants():
    assert G.clean_variants("ቀተለ") == (["ቀተለ"], [])
    assert G.clean_variants("ቀተለ / ቀቲል ፡") == (["ቀተለ", "ቀቲል"], [])
    assert G.clean_variants("ዐየሰ (ዓሰ") == ([], ["ዐየሰ (ዓሰ"])       # unclosed note: dropped, as in evaluate_gold
    assert G.clean_variants("ዐየሰ (ዓሰ)") == (["ዐየሰ"], [])
    assert G.clean_variants("ወኮኑ (ወኮኑ ሰብአ ፋርስ መላእክት)") == (["ወኮኑ"], [])
    assert G.clean_variants("ይፀመድ [ይፀመደኒ]") == ([], ["ይፀመድ [ይፀመደኒ"])
    assert G.clean_variants("qatala / ቀተለ") == (["ቀተለ"], ["qatala"])
    assert G.clean_variants("") == ([], []) and G.clean_variants(None) == ([], [])


def test_trainer_and_evaluate_gold_read_the_same_forms():
    import train_byt5_morphology as T
    import evaluate_gold as E
    rec = {"page_number": 1, "title_or_lemma": "ቀተለ", "raw_transcription": "x", "root": "ቀ-ተ-ለ",
           "category": "verb_conjugation", "stem": "basic",
           "table_data": [{"geez_form": f, "tense": "past", "person": "3", "gender": "m", "number": "s"}
                          for f in ("ቀተለ (note)", "ቀተልኩ / ቀተልኩ ፡", "ቀተልከ (ቀተልከ ብእሴ", "two words", "ቀተልክሙ")]}
    with tempfile.TemporaryDirectory() as tmp:
        para = os.path.join(tmp, "data", "paradigms"); os.makedirs(para)
        open(os.path.join(para, "synthetic_morphology_train.jsonl"), "w").close()
        gold = os.path.join(para, "merho_ges_gold_train.jsonl")
        with open(gold, "w", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        write_tables(os.path.join(tmp, "data", "corrections"), [])
        stats = collections.Counter()
        train = sorted(x["input_text"][len("analyze: "):] for x in T.load_data(tmp, stats=stats))
        C = G.Corrections.from_dataset(tmp, required=True)
        evals = sorted(p["surface"] for p in E.load_gold_test_pairs(gold, corrections=C)[0])
    assert train == evals == sorted(["ቀተለ", "ቀተልኩ", "ቀተልኩ", "ቀተልክሙ"])
    assert stats["gold variants dropped: notes or extra words (not one Ethiopic word)"] == 2


def test_real_gold_train_inputs_are_single_words():
    path = os.path.join(DATASET, "data", "corrections", "gold_root_corrections.tsv")
    if not os.path.exists(path):
        return                                   # dataset checkout not next to the model repo
    C = G.Corrections.from_dataset(DATASET, required=True)
    with open(os.path.join(DATASET, "data", "paradigms", "merho_ges_gold_train.jsonl"), encoding="utf-8") as f:
        out = C.gold_records([json.loads(l) for l in f], collections.Counter())
    kept = [v for r in out for row in r["table_data"] for v in G.clean_variants(row.get("geez_form"))[0]]
    assert kept and all(G.is_single_ethiopic_word(v) for v in kept)
