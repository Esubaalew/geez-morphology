"""Dictionary roots and the training file, checked against the dataset checkout.

- RULED: the inferred_root of 232 Kidane Wold entries, in both dictionary files (None = no root). The
  generator reads the same root (verb_classes.entry_root with the root fixes).
- Entry 2194 is an adverb: its root w-s-T is not on the allow-list and has no synthetic lines.
- No line of the training file has a gold val/test root (computed from the gold files), and none has one of
  the LEAK_GATE roots.
- m-w-t is on the allow-list; its synthetic 3ms past form equals the gold train form.
- a-p-a, m-h1-r, n-h1-r and w-h2-n (roots no entry has any more) and S2-a-S2-a (not drawn by the quadriliteral sample) are not
  on the allow-list and have no training lines; m-h2-r and n-kh-r are gold val roots and have no training lines.
- w-S2-kh is on the allow-list; the only dictionary entries that give it are 1369, 1370 and 1390.
Roots are written with the Latin consonant symbols; the test types no Ge'ez letters.
"""
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import generate_synthetic_data as gen  # noqa: E402
import geez_script  # noqa: E402
import split_heldout_synthetic as H  # noqa: E402
import verb_classes  # noqa: E402
from conftest import DATASET_DIR  # noqa: E402

DICTIONARIES = ["data/lexicon/classical_dictionary_kidane_wold.jsonl", "data/geez_amharic_dictionary.jsonl"]
TRAIN_FILE = os.path.join(DATASET_DIR, "data", "paradigms", "synthetic_morphology_train.jsonl")

RULED = {
    361: "q-h2-m", 364: "q-h2-m", 494: "q-T-f", 495: "q-T-f", 618: "q-y-a", 729: "a-kh2-t", 930: "q-y-m", 931: "q-y-m",
    933: "q-y-m", 935: "q-y-m", 977: "q-S-b", 1139: "q-l-l", 1235: "n-q-l-q-l", 1261: "q-r-b", 1329: "q-m-h2", 1355: "w-S2-a",
    1360: "w-S2-a", 1362: "w-S2-a", 1363: "w-S2-a", 1365: "S2-a-S2-a", 1366: "S2-a-S2-a", 1367: "S2-a-S2-a", 1368: "S2-a-S2-a",
    1369: "w-S2-kh", 1370: "w-S2-kh", 1390: "w-S2-kh", 1462: "q-r-n", 1465: "q-r-n", 1531: "q-r-n", 1540: None, 1541: None,
    1609: "S-b-h2", 1715: "w-r-z-w", 1716: "w-r-z-w", 1743: "qh-r-f", 1762: "w-r-z-w", 2045: "w-kh-T", 2069: "w-T-s", 2170: "w-r-s",
    2194: "w-s-T", 2218: "w-s-n", 2262: "z-b-h2", 2311: "z-g-h2", 2353: "z-h2-b", 2355: "z-h2-z-h2", 2362: "z-h2-l", 2365: None,
    2595: "z-r-a2", 2612: None, 2751: "b-r-k", 2872: "g-a-z", 2874: "g-a-z", 2876: "g-a-z", 2931: None, 3028: "gw-d-a2", 3094: None,
    3099: "g-w-r", 3107: None, 3110: "g-z-a", 3112: "g-z-a", 3162: "g-z-r", 3171: "gw-h2-l", 3183: "g-h2-s2", 3184: "g-h2-s2",
    3189: "g-h2-f", 3192: "g-h2-f", 3194: "g-h2-f", 3196: "g-h2-f", 3201: None, 3242: "g-y-s2", 3305: "gw-l-h2", 3452: "g-n-z",
    3495: "g-n-f-l", 3512: "g-s2-S", 3529: "g-a2-z", 3551: "g-f-a2", 3597: "g-r-h1", 3603: None, 3604: "g-n-z", 3643: "g-r-r",
    3644: None, 3738: None, 3758: "d-gw-S", 3784: "d-h1-m-m", 3886: None, 3891: "d-y-n", 4076: "d-n-n", 4109: "d-f-n",
    4110: "d-f-n", 4166: "d-r-g", 4178: "d-r-g", 4180: "d-r-g", 4181: None, 4221: "d-r-s", 4232: "d-s-y", 4488: "h2-w-r",
    4492: "h2-w-r", 4568: "h2-z-r", 4773: "h2-m-S", 4820: "h2-S2-n", 4825: "h2-S2-n", 4941: "q-n-t", 4942: "q-n-t", 5006: "q-s2-f",
    5024: "S2-r-r", 5430: "s2-h1-b-b", 5436: "s2-h1-l", 5439: "s2-h1-l", 5449: "s2-w-T", 5495: "s2-h2-q", 5498: "s2-h2-q",
    5504: "s2-kh-t", 5516: "s2-T-q", 5526: "s2-y-T", 5561: "s2-l-T", 5636: "s2-n-q", 5779: "s2-r-q", 5856: "a2-g-l", 5857: "a2-g-l",
    5973: "a2-w-r", 5995: "m-a2-z-r", 5996: "m-a2-z-r", 6062: "a2-r-g", 6155: "a2-m-S2", 6156: "a2-m-S2", 6779: "f-s2-h2",
    6815: "f-S2-n", 6926: "l-a-k", 6997: "l-b-s", 6999: "l-b-s", 7000: "l-b-s", 7024: "l-h1-b", 7025: "l-h1-b", 7026: "l-h1-b",
    7076: "l-w-s", 7086: "l-h2-h2", 7127: "l-kh-l-kh", 7129: "l-kh-l-kh", 7190: "l-m-d", 7259: "l-q-w", 7429: "m-h2-r",
    7431: "m-h2-r", 7432: "m-h2-r", 7467: "m-w-q", 7468: "m-w-q", 7470: "m-w-q", 7471: "m-w-t", 7473: "m-w-t", 7474: "m-w-t",
    7475: "m-w-t", 7476: "m-w-t", 7477: "m-w-t", 7479: "m-w-t", 7851: "a2-s-b", 8213: "n-d-d", 8241: "n-h1-g", 8251: "n-kh-r",
    8308: "n-z-h1", 8309: "n-z-h1", 8357: None, 8358: "n-h2-n-h2", 8359: "n-h2-n-h2", 8360: "n-h2-n-h2", 8361: "n-h2-n-h2",
    8564: "h2-t-t", 8619: "kh-b-r", 8637: "kh-d-g", 8932: "kh-S2-b", 9079: "T-b-l-l", 9114: "T-g-a2", 9118: "T-g-a2",
    9119: "T-g-a2", 9124: "T-h1-r", 9127: "T-h1-r", 9147: "T-l-a2", 9148: "T-l-a2", 9151: "T-l-a2", 9156: "T-w-m", 9157: "T-w-m",
    9159: None, 9176: "T-h2-l", 9178: "T-h2-l", 9183: "T-h2-n", 9189: "T-h2-r", 9193: None, 9234: "T-y-t", 9237: "T-y-t",
    9238: "T-y-t", 9256: "T-l-T-l", 10339: "S-h2-l", 10386: None, 10516: "S-f-r", 10650: "q-t-l", 10964: "r-y-S2", 11018: "r-m-m",
    11341: "s-d-d", 11344: "s-d-d", 11351: None, 11444: "a-n-s-s", 11445: "a-n-s-s", 11490: "n-s-h2-l-l", 11601: None,
    11661: "s-l-b", 11674: "s-l-T", 11675: "s-l-T", 11688: "s-l-m", 11701: "s-l-f", 11849: "s-n-S", 11869: None, 11878: "s-a2-l",
    12041: "s-r-h2", 12088: "s-r-r", 12093: "s-r-r", 12141: "s-t-r", 12181: "t-b-a2", 12194: "t-g-h1", 12227: "t-s-h2",
    12241: "t-h2-l", 12243: "t-h2-t", 12251: "t-h2-t", 12252: "t-h2-t", 12253: "t-h2-t", 12262: "t-y-h1", 12284: "t-k-l",
    12286: "t-k-l", 12345: None, 12425: None, 12444: None, 12746: "b-l-h2", 13358: None, 14377: "b-z-z", 14455: "z-n-g-a2",
}

LEAK_GATE = ["w-s-n", "g-n-f-l", "T-b-l-l", "z-n-g-a2", "q-y-m", "q-r-b", "q-r-n", "z-r-a2", "s2-T-q", "l-b-s", "t-y-h1", "t-k-l",
             "w-kh-T", "S-h2-l", "s-n-S", "m-h2-r", "h2-t-t", "n-s-h2-l-l", "q-y-a", "g-z-r", "g-h2-f", "h2-w-r", "d-gw-S", "t-g-h1",
             "a-kh2-t", "g-h2-s2", "s-t-r"]


def _need_dataset():
    if not os.path.exists(TRAIN_FILE):
        pytest.skip(f"no dataset checkout at {DATASET_DIR}")


def _entries(rel):
    out = {}
    with open(os.path.join(DATASET_DIR, rel), encoding="utf-8") as f:
        for line in f:
            e = json.loads(line)
            if "id" in e:
                out[e["id"]] = e
    return out


def _train_lines():
    with open(TRAIN_FILE, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


@pytest.mark.parametrize("rel", DICTIONARIES)
def test_ruled_roots_are_in_the_dictionary(rel):
    _need_dataset()
    entries = _entries(rel)
    wrong = {i: entries[i].get("inferred_root") for i, r in RULED.items() if entries[i].get("inferred_root") != r}
    assert wrong == {}


def test_generator_reads_the_ruled_roots():
    _need_dataset()
    fixes = verb_classes.root_fixes(DATASET_DIR)
    entries = _entries("data/geez_amharic_dictionary.jsonl")
    wrong = {i: verb_classes.entry_root(entries[i], fixes) for i, r in RULED.items() if verb_classes.entry_root(entries[i], fixes) != r}
    assert wrong == {}


def test_adverb_2194_gives_no_verb_lines():
    _need_dataset()
    for rel in DICTIONARIES:
        assert _entries(rel)[2194]["pos"] == "adverb"
    assert ("w", "s", "T") not in gen.read_allowed_roots_file()
    assert not any(d["features"]["root"] == "w-s-T" for d in _train_lines())


def test_no_gold_val_or_test_root_in_the_training_file():
    _need_dataset()
    eval_roots = H.gold_eval_roots(DATASET_DIR)
    train_roots = {d["features"]["root"] for d in _train_lines()}
    leaked = sorted(r for r in train_roots
                    if geez_script.first_order_root(geez_script.to_ethiopic_root(r)) in eval_roots)
    assert leaked == []
    assert sorted(set(LEAK_GATE) & train_roots) == []


def test_m_w_t_past_matches_gold():
    _need_dataset()
    assert ("m", "w", "t") in gen.read_allowed_roots_file()
    past = [d["surface_form"] for d in _train_lines()
            if d["features"]["root"] == "m-w-t" and d["features"]["stem"] == "basic" and d["features"]["tense"] == "past"
            and (d["features"]["person"], d["features"]["gender"], d["features"]["number"]) == ("3", "m", "s")
            and "object" not in d["features"]]
    gold_root = geez_script.to_ethiopic_root("m-w-t")
    gold = []
    with open(os.path.join(DATASET_DIR, "data", "paradigms", "merho_ges_gold_train.jsonl"), encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("root") == gold_root:
                gold += [r["geez_form"] for r in rec["table_data"]
                         if (r.get("person"), r.get("gender"), r.get("number"), r.get("tense")) == ("3", "m", "s", "past")]
    assert gold and past == gold[:1]


def test_roots_without_training_lines():
    _need_dataset()
    allowed = gen.read_allowed_roots_file()
    train_roots = {d["features"]["root"] for d in _train_lines()}
    for r in ("a-p-a", "m-h1-r", "n-h1-r", "w-h2-n", "S2-a-S2-a"):
        assert tuple(r.split("-")) not in allowed, r
        assert r not in train_roots, r
    for r in ("m-h2-r", "n-kh-r"):
        assert r not in train_roots, r


def test_w_s2_kh_comes_from_its_three_entries():
    _need_dataset()
    assert ("w", "S2", "kh") in gen.read_allowed_roots_file()
    # gold test record b0e9efdf890f (ተዋፅኀ) has this root, so its synthetic lines are held out
    assert not any(d["features"]["root"] == "w-S2-kh" for d in _train_lines())
    fixes = verb_classes.root_fixes(DATASET_DIR)
    for rel in DICTIONARIES:
        ids = sorted(i for i, e in _entries(rel).items() if verb_classes.entry_root(e, fixes) == "w-S2-kh"
                     or e.get("inferred_root") == "w-S2-kh")
        assert ids == [1369, 1370, 1390], (rel, ids)
