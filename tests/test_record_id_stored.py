"""dataset_loader.record_id: the id stored in a corrected gold record wins, and is never recomputed
(record_id stays frozen at its value in the files as extracted, dataset commit c7bc587)."""
import dataset_loader as G


def _rec(**kw):
    rec = {"page_number": 53, "title_or_lemma": "ረገሰ", "raw_transcription": "ረገሰ ...", "root": "ረ-ገ-ለ",
           "stem": "basic", "table_data": [{"geez_form": "ረገለ", "tense": "past", "person": "3", "gender": "m",
                                            "number": "s"}]}
    rec.update(kw)
    return rec


def test_computed_from_extracted_values_when_not_stored():
    rec = _rec()
    rid = G.record_id(rec)
    assert len(rid) == 12 and rid == G.record_id(dict(rec))


def test_stored_id_survives_a_title_fix():
    extracted = _rec()
    rid = G.record_id(extracted)
    fixed = _rec(title_or_lemma="ረገለ", record_id=rid)
    assert G.record_id(_rec(title_or_lemma="ረገለ")) != rid       # recomputing would change it
    assert G.record_id(fixed) == rid                              # the stored one is used
    pairs = G.file_pairs([fixed], "train", {})
    assert pairs and all(p["record_id"] == rid for p in pairs)


def test_empty_record_root_with_row_roots_loads_rows_only():
    # p357 (train 1070): record root cleared to "" (section heading); each row carries its own root
    rec = _rec(root="", record_id="ea2f7a24e625", table_data=[
        {"geez_form": "ደየፐ", "root": "ደ-የ-ፐ", "tense": "past", "person": "3", "gender": "m", "number": "s"},
        {"geez_form": "ዳጋ", "tense": "past", "person": "3", "gender": "m", "number": "s"}])
    stats = __import__("collections").Counter()
    pairs = G.file_pairs([rec], "train", {}, stats)
    assert [(p["form"], p["root"]) for p in pairs] == [("ደየፐ", "ደ-የ-ፐ")]
    assert G.root_problem("") is not None                         # a row with no root of its own does not load
