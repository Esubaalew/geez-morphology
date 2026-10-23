"""Spellings with two readings (homographs): the loader keeps every reading, the scorer asks once per spelling,
counts a reply right if it matches any reading and reports the homograph group on its own.
Fixtures are reviewed two-reading cases: ማዕበል (ዐ-በ-ለ, and መ-ዐ-በ-ለ 'wave'), ጽልመት (ጸ-ለ-መ, ጸ-ለ-መ-ተ),
መልአክ (ለ-አ-ከ, መ-ለ-አ-ከ), ትካት (ተ-ከ-ተ, and a reading with no root), ሄላ (two glosses, one analysis, no root),
and ጵርስቅለ (perfume) vs ጵርስቅላ (a name): two spellings, not one."""
import os
import sys

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, HERE)

import dataset_loader as G  # noqa: E402
import evaluate_gold as E  # noqa: E402


def noun(surface, root, split="test", **kw):
    return {"split": split, "surface": surface, "root": root, "verb_class": None,
            **E.normalize_row({"type": kw.pop("type", "noun"), "number": "s", **kw})}


def reply(root, typ="noun"):
    return {"root": root, **E.normalize_row({"type": typ, "number": "s"}), "tier": "byt5", "raw": ""}


class Replies:
    """Answers from a {surface: reply} table and counts the calls."""
    def __init__(self, table):
        self.table, self.calls = table, []

    def predict(self, surface):
        self.calls.append(surface)
        return dict(self.table[surface])


TWO_ROOTS = [("ማዕበል", "ዐ-በ-ለ", "መ-ዐ-በ-ለ"), ("ጽልመት", "ጸ-ለ-መ", "ጸ-ለ-መ-ተ"), ("መልአክ", "ለ-አ-ከ", "መ-ለ-አ-ከ")]


def test_either_reading_is_right_and_the_spelling_counts_once():
    for form, a, b in TWO_ROOTS:
        pairs = [noun(form, a), noun(form, b)]
        assert E.is_homograph(pairs)
        for said in (a, b):
            sys_ = Replies({form: reply(said)})
            rows = []
            totals, _ = E.score(pairs, sys_, rows)
            assert sys_.calls == [form]                                  # asked once
            assert totals["ALL"]["rows"][1] == 1 and totals["ALL"]["root"] == [1, 1]
            assert totals["ALL"]["exact"] == [1, 1]
            assert totals["homograph"]["rows"][1] == 1 and totals["homograph"]["root"] == [1, 1]
            assert rows[0]["gold"]["root"] == said and len(rows[0]["readings"]) == 2


def test_a_reply_matching_no_reading_is_one_miss():
    pairs = [noun("ማዕበል", "ዐ-በ-ለ"), noun("ማዕበል", "መ-ዐ-በ-ለ")]
    totals, _ = E.score(pairs, Replies({"ማዕበል": reply("ቀ-ተ-ለ")}))
    assert totals["ALL"]["root"] == [0, 1] and totals["homograph"]["root"] == [0, 1]


def test_a_rooted_and_a_rootless_reading():
    pairs = [noun("ትካት", "ተ-ከ-ተ"), noun("ትካት", None)]
    for said in ("ተ-ከ-ተ", None):
        totals, _ = E.score(pairs, Replies({"ትካት": reply(said)}))
        assert totals["ALL"]["root"] == [1, 1] and totals["ALL"]["exact"] == [1, 1]
        assert totals["homograph"]["rows"][1] == 1


def test_two_glosses_with_one_analysis_are_not_a_homograph():
    pairs = [noun("ሄላ", None), noun("ሄላ", None)]
    assert not E.is_homograph(pairs)
    totals, _ = E.score(pairs, Replies({"ሄላ": reply(None)}))
    assert totals["ALL"]["rows"][1] == 1 and "homograph" not in totals


def test_two_spellings_are_two_items():
    pairs = [noun("ጵርስቅለ", None), noun("ጵርስቅላ", None, type="proper_noun")]
    sys_ = Replies({"ጵርስቅለ": reply(None), "ጵርስቅላ": reply(None, "proper_noun")})
    totals, _ = E.score(pairs, sys_)
    assert sys_.calls == ["ጵርስቅለ", "ጵርስቅላ"] and totals["ALL"]["rows"][1] == 2 and "homograph" not in totals


def test_one_spelling_in_two_splits_is_two_items():
    pairs = [noun("ማዕበል", "ዐ-በ-ለ", split="val"), noun("ማዕበል", "መ-ዐ-በ-ለ", split="test")]
    totals, _ = E.score(pairs, Replies({"ማዕበል": reply("ዐ-በ-ለ")}))
    assert totals["ALL"]["rows"] == [0, 2] and totals["ALL"]["root"] == [1, 2] and "homograph" not in totals


def test_loader_keeps_every_rootless_reading():
    rec = {"record_id": "aaaaaaaaaaaa", "page_number": 1, "root": "", "category": "noun",
           "table_data": [{"geez_form": "ትካት", "type": "noun", "number": "s"}]}
    one = {"form": "ትካት", "word_class": "noun", "number": "s", "source_ref": "aaaaaaaaaaaa:0"}
    two = {"form": "ትካት", "word_class": "proper_noun", "number": "s", "source_ref": "aaaaaaaaaaaa:0"}
    pairs = G.file_pairs([rec], "test", rootless={"ትካት": [one, two]})
    assert [(p["form"], p["root"], p["type"]) for p in pairs] == [("ትካት", None, "noun"), ("ትካት", None, "proper_noun")]
    (p,) = G.file_pairs([rec], "test", rootless={"ትካት": one})             # a single row still loads
    assert p["type"] == "noun"
    # the form shows up again in a later row: its readings are not loaded twice
    rec["table_data"].append({"geez_form": "ትካት", "type": "noun", "number": "s"})
    assert len(G.file_pairs([rec], "test", rootless={"ትካት": [one, two]})) == 2


# --- any-reading vs the old strict per-row score (ካዕበት: singular under ከ-ዐ-በ, plural under ከ-ዐ-በ-ተ) ---

def two_reading_pairs():
    return [noun("ካዕበት", "ከ-ዐ-በ"), noun("ካዕበት", "ከ-ዐ-በ-ተ", number="p")]


def test_two_reading_form_is_right_for_either_reading():
    for pred in (reply("ከ-ዐ-በ"), {**reply("ከ-ዐ-በ-ተ"), "number": "p"}):
        totals, _ = E.score(two_reading_pairs(), Replies({"ካዕበት": pred}))
        assert totals["ALL"]["rows"] == [0, 1]
        assert totals["ALL"]["root"] == [1, 1] and totals["ALL"]["exact"] == [1, 1]
        # strict: one row per reading, so the same reply is right on one and wrong on the other
        strict = E.score_strict(two_reading_pairs(), Replies({"ካዕበት": pred}))
        assert strict["ALL"]["rows"] == [0, 2]
        assert strict["ALL"]["root"] == [1, 2] and strict["ALL"]["exact"] == [1, 2]
        assert strict["homograph"]["rows"] == [0, 2]


def test_wrong_prediction_is_still_wrong():
    for strict in (False, True):
        totals, _ = E.score(two_reading_pairs(), Replies({"ካዕበት": reply("ቀ-ተ-ለ")}), strict=strict)
        assert totals["ALL"]["root"][0] == 0 and totals["ALL"]["exact"][0] == 0
    # right root but the singular/plural mix-up of the wrong reading: root right, exact still matches singular
    totals, _ = E.score(two_reading_pairs(), Replies({"ካዕበት": {**reply("ከ-ዐ-በ"), "number": "p"}}))
    assert totals["ALL"]["root"] == [1, 1] and totals["ALL"]["exact"] == [0, 1]


def test_single_reading_forms_score_the_same_both_ways():
    pairs = [noun("ንጉሥ", "ነ-ገ-ሠ"), noun("ቤት", "ቤ-ተ")]
    table = {"ንጉሥ": reply("ነ-ገ-ሠ"), "ቤት": reply("ቀ-ተ-ለ")}
    any_reading, _ = E.score(pairs, Replies(table))
    strict = E.score_strict(pairs, Replies(table))
    for f in ("rows", "root", "exact", "number"):
        assert any_reading["ALL"][f] == strict["ALL"][f]
    assert any_reading["ALL"]["root"] == [1, 2]
    assert "homograph" not in strict


def test_cached_system_asks_each_spelling_once():
    inner = Replies({"ካዕበት": reply("ከ-ዐ-በ")})
    cached = E.CachedSystem(inner)
    E.score(two_reading_pairs(), cached)
    E.score_strict(two_reading_pairs(), cached)
    assert inner.calls == ["ካዕበት"]
