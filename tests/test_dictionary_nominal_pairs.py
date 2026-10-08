"""Kidane Wold noun and adjective pairs (dataset_loader.load_dictionary_pairs, split_dictionary_pairs).

The first tests build a small dataset folder: every left-out kind of entry is there once, and the roots come from
the allow-list and held-out files of this repo at run time. The later tests read the dataset checkout; their
Ge'ez strings were copied from the dataset files by a script, not typed.
"""
import collections
import json
import os
import sys
import tempfile

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import dataset_loader as D  # noqa: E402
import generate_synthetic_data as gen  # noqa: E402
import geez_script  # noqa: E402
import split_heldout_synthetic as H  # noqa: E402
import train_byt5_morphology as T  # noqa: E402
from conftest import DATASET_DIR  # noqa: E402

# (dictionary id, headword_geez of that entry, pos, gold-train record_id, root of that gold record)
GOLD_BACKED = [
    (13878, 'መእገር', 'noun', '3d7f23db6a75', 'አ-ገ-ረ'),
    (13947, 'አዲም', 'adjective', 'c5baa5a5b367', 'አ-ደ-መ'),
    (14011, 'አውራስ', 'noun', '4ebfda54127e', 'ወ-ረ-ሰ'),
    (717, 'አክሌል', 'noun', 'a1d80b37a185', 'ከ-ለ-ለ'),
]
IN_GOLD_TRAIN = (14308, 'በድር', '5079fde4e60a', 4)
CONFLICTING = (7296, 7299, 'ልሳነ')   # two entries, one headword, two roots
GOLD_VAL_ROOT = (13734, 'እባብ', '53b646b8537c', 'አ-በ-በ')


# ---------------------------------------------------------------- a small dataset folder


def _word(symbols, vowels):
    return "".join(geez_script.render_unit(c, v) for c, v in zip(symbols, vowels))


def _root(symbols):
    return "-".join(symbols)


def _roots():
    """Allowed 3-radical roots, a 3-radical root of the held-out roots file that is not on the allow-list, and a
    3-radical root of known letters that is in neither file."""
    allowed = sorted(r for r in gen.read_allowed_roots_file() if len(r) == 3)
    held_out = sorted(gen.read_held_out_roots_file())
    off_list = next(r for r in held_out if len(r) == 3 and r not in gen.read_allowed_roots_file())
    plain = [c for c in geez_script.CONSONANTS if c not in ("qh", "kh2", "gw", "khw")]
    not_allowed = next((a, b, c) for a in plain for b in plain for c in plain
                       if (a, b, c) not in gen.read_allowed_roots_file() and (a, b, c) not in held_out)
    return allowed, off_list, not_allowed


def _dataset(tmp):
    allowed, held_out_root, not_allowed = _roots()
    A, B, C, VAL, TEST, F, G, X, Y, Z, FIX = allowed[:11]
    P, Q = next((p, q) for p in allowed[11:] for q in allowed[11:] if p < q and p[:2] == q[:2])   # one form holds both
    lex, para = os.path.join(tmp, "data", "lexicon"), os.path.join(tmp, "data", "paradigms")
    os.makedirs(lex)
    os.makedirs(para)
    val_form, train_form = _word(F, "eoe"), _word(G, "eoe")

    def entry(i, symbols, pos="noun", vowels="eoe", **kw):
        e = {"id": i, "headword_geez": _word(symbols, vowels), "pos": pos, "headword_type": "word",
             "inferred_root": _root(symbols), "root": None, "root_notes": "", "forms": []}
        e.update(kw)
        return e

    entries = [
        entry(1, A, forms=[_word(A, "oeo"), "ab", _word(B, "oeo")]),             # kept; forms tried only on request
        entry(2, B, pos="adjective", gender="f"),                                 # kept, gender f
        entry(3, C, pos="adjective", number="p"),                                 # kept, no gender, number p
        entry(4, A, pos="name", vowels="oeo"),
        entry(5, A, pos="particle", vowels="oee"),
        entry(6, A, pos=None, vowels="eeo"),
        entry(7, A, pos="?", vowels="eoo"),
        entry(8, A, vowels="eoe", headword_type="phrase", headword_geez=_word(A, "eoe") + " ፡ " + _word(B, "eoe")),
        entry(9, A, headword_geez=_word(A, "eoe") + " " + _word(B, "eoe")),
        entry(10, A, headword_geez="abc"),
        entry(11, X, inferred_root=None),
        entry(12, X, vowels="oeo", inferred_root=""),
        entry(13, X, vowels="oee", inferred_root="CHECK"),
        entry(14, X, vowels="eeo", root_notes="CHECK: gloss not clear"),
        entry(15, X, vowels="eoo", root_unsettled=True),
        entry(16, Y[:2], vowels="eo"),                                            # 2 radicals
        entry(17, Y + Z[:2], vowels="eoeoe"),                                     # 5 radicals
        entry(18, not_allowed),                                                   # not on the allow-list
        entry(19, VAL),                                                           # gold val root
        entry(20, TEST),                                                          # gold test root
        entry(21, held_out_root),                                                 # held-out roots file
        entry(22, F),                                                             # its form is a gold val form
        entry(23, G),                                                             # its form is a gold train form
        entry(24, P, headword_geez=_word(P + Q[2:], "eoeo")),
        entry(25, Q, headword_geez=_word(P + Q[2:], "eoeo")),                     # one form, two roots
        entry(26, Z), entry(27, Z),                                               # same pair twice
        entry(28, FIX[:2], vowels="oo", headword_geez=_word(FIX, "ooo")),         # root fixes file gives FIX
        entry(29, X, vowels="ooo", inferred_root=_root(B)),                       # root fixes file gives null
        entry(30, C, vowels="ooo", inferred_root=_root(X)),                       # root letters not in the headword
    ]
    with open(os.path.join(lex, "classical_dictionary_kidane_wold.jsonl"), "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    with open(os.path.join(lex, "kidane_wold_root_fixes.jsonl"), "w", encoding="utf-8") as f:
        for i, new in ((28, _root(FIX)), (29, None)):
            f.write(json.dumps({"id": i, "new_root": new}) + "\n")

    def gold(name, symbols, form):
        rec = {"record_id": name, "root": "-".join(geez_script.render_unit(c, "a") for c in symbols),
               "table_data": [{"geez_form": form, "type": "noun", "number": "s"}]}
        with open(os.path.join(para, D.gold_file_name(name)), "w", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    gold("val", VAL, val_form)
    gold("test", TEST, _word(TEST, "ooo"))
    gold("train", A, train_form)
    blocked = gen.read_blocked_forms_file()
    assert not {e["headword_geez"] for e in entries} & blocked
    assert not D.carries_root(entries[29]["headword_geez"], "-".join(geez_script.render_unit(c, "a") for c in X))
    return {"A": A, "B": B, "C": C, "Z": Z, "FIX": FIX, "entries": {e["id"]: e for e in entries}}


def _load(tmp, **kw):
    stats = collections.Counter()
    return D.load_dictionary_pairs(tmp, stats, **kw), stats


def test_kept_pairs_and_labels():
    with tempfile.TemporaryDirectory() as tmp:
        d = _dataset(tmp)
        pairs, stats = _load(tmp)
        e = d["entries"]
        assert [(p["record_id"], p["form"]) for p in pairs] == [
            ("1", e[1]["headword_geez"]), ("2", e[2]["headword_geez"]), ("3", e[3]["headword_geez"]),
            ("26", e[26]["headword_geez"]), ("28", e[28]["headword_geez"])]
        eth = {k: "-".join(geez_script.render_unit(c, "a") for c in d[k]) for k in ("A", "B", "C", "FIX")}
        assert all(p["task"] == D.ROOT_ONLY_TASK for p in pairs)
        targets = [D.target_text(p) for p in pairs]
        assert targets[:3] + targets[4:] == [f"root: {eth[k]}" for k in ("A", "B", "C", "FIX")]
        labels = [(p["type"], p["gender"], p["number"]) for p in pairs]
        assert labels == [("noun", "", "s"), ("adjective", "f", "s"), ("adjective", "", "p"), ("noun", "", "s"),
                          ("noun", "", "s")]
        full = dict(pairs[1], task=None)
        assert D.target_text(full) == f"root: {eth['B']} | type: adjective | gender: f | number: s"
        assert pairs[3]["record_ids"] == ["26", "27"]
        assert stats["loaded"] == 5 and stats["headwords with conflicting roots"] == 1


def test_left_out_kinds():
    with tempfile.TemporaryDirectory() as tmp:
        _dataset(tmp)
        _, stats = _load(tmp)
        expected = {
            "not loaded: pos name": 1, "not loaded: pos particle": 1, "not loaded: pos missing": 1, "not loaded: pos ?": 1,
            "not loaded: headword_type phrase": 1, "not loaded: headword is not one Ge'ez word": 2,
            "not loaded: no root": 4, "not loaded: root marked unclear": 2,
            "not loaded: headword does not hold the root's radicals": 1,
            "not loaded: root has 2 radicals": 1, "not loaded: root has 5 radicals": 1,
            "not loaded: root not on the synthetic allow-list": 2,
            "not loaded: root is a gold val/test root": 2, "not loaded: form is a gold val/test form": 1,
            "not loaded: form is in the gold train file": 1, "not loaded: headword has conflicting roots": 2,
            "not loaded: duplicate (form, target)": 1}
        assert {k: v for k, v in stats.items() if k.startswith("not loaded")} == expected


def test_held_out_roots_file_without_the_allow_list():
    with tempfile.TemporaryDirectory() as tmp:
        d = _dataset(tmp)
        pairs, stats = _load(tmp, allowed_roots_only=False)
        assert stats["not loaded: root is in the held-out roots file"] == 1
        ids = {p["record_id"] for p in pairs}
        assert {"16", "17", "18"} <= ids and not {"19", "20", "21", "22", "23", "24", "25"} & ids
        assert d


def test_forms_only_on_request():
    with tempfile.TemporaryDirectory() as tmp:
        d = _dataset(tmp)
        assert not any(p["from_forms"] for p in _load(tmp)[0])
        pairs, stats = _load(tmp, include_forms=True)
        forms = [p["form"] for p in pairs if p["from_forms"]]
        assert forms == [_word(d["A"], "oeo")]
        assert stats["forms: not one Ge'ez word"] == 1 and stats["forms: does not hold the root's radicals"] == 1


def test_carries_root():
    q, t, l, w, m = (geez_script.render_unit(c, "a") for c in ("q", "t", "l", "w", "m"))
    assert D.carries_root(q + t + l, f"{q}-{t}-{l}")
    assert not D.carries_root(q + l + t, f"{q}-{t}-{l}")                   # order counts
    assert not D.carries_root(q + t, f"{q}-{t}-{l}")
    assert D.carries_root(geez_script.render_unit("q", "o") + m, f"{q}-{w}-{m}")   # w may be missing
    assert D.carries_root(geez_script.render_unit("n", "a") + geez_script.render_unit("d", "a"),
                          "-".join(geez_script.render_unit(c, "a") for c in ("n", "d", "d")))   # doubled radical


def test_split_by_root_and_deterministic():
    pairs = [{"form": f"w{i}", "root": f"r{i % 37}"} for i in range(400)]
    train, evals = D.split_dictionary_pairs(pairs)
    assert not {p["root"] for p in train} & {p["root"] for p in evals}
    assert 40 <= len(evals) < 40 + 11 and len(train) + len(evals) == 400
    assert (train, evals) == D.split_dictionary_pairs(list(pairs))


def test_trainer_flag_and_eval_slice():
    with tempfile.TemporaryDirectory() as tmp:
        _dataset(tmp)
        train, evals = T.load_dictionary_samples(tmp)
        assert len(train) + len(evals) == 5 and all(s["source"] == "dictionary" for s in train + evals)
        assert all(s["task"] == D.ROOT_ONLY_TASK and s["target_text"].count("|") == 0 for s in train + evals)
        assert T.parse_args(["--no_dictionary_pairs"]).no_dictionary_pairs and not T.parse_args([]).no_dictionary_pairs
    with tempfile.TemporaryDirectory() as tmp:
        assert T.load_dictionary_samples(tmp) == ([], [])


# ---------------------------------------------------------------- the dataset checkout

def _need_dataset():
    if not os.path.exists(os.path.join(DATASET_DIR, D.KIDANE_WOLD_FILE)):
        pytest.skip(f"no dataset checkout at {DATASET_DIR}")


def _checkout_pairs():
    pairs = D.load_dictionary_pairs(DATASET_DIR)
    return pairs, D.split_dictionary_pairs(pairs)


def test_no_dictionary_train_pair_has_a_gold_val_or_test_root():
    _need_dataset()
    pairs, (train, evals) = _checkout_pairs()
    eval_roots = H.gold_eval_roots(DATASET_DIR)
    held_out = gen.read_held_out_roots_file()
    eval_forms = H.gold_eval_forms(DATASET_DIR)
    assert train and evals
    assert [p for p in train if geez_script.first_order_root(p["root"]) in eval_roots] == []
    assert [p for p in pairs if geez_script.first_order_root(p["root"]) in eval_roots] == []
    assert [p for p in pairs if geez_script.root_symbols(p["root"]) in held_out] == []
    assert [p for p in pairs if p["form"] in eval_forms] == []


def test_checkout_roots_are_allowed_and_split_by_root():
    _need_dataset()
    pairs, (train, evals) = _checkout_pairs()
    allowed = gen.read_allowed_roots_file()
    assert all(len(geez_script.root_symbols(p["root"])) in (3, 4) for p in pairs)
    assert all(geez_script.root_symbols(p["root"]) in allowed for p in pairs)
    assert {p["type"] for p in pairs} == {"noun", "adjective"}
    assert not {p["root"] for p in train} & {p["root"] for p in evals}
    assert len({(p["form"], D.target_text(p)) for p in pairs}) == len(pairs)
    forms = collections.defaultdict(set)
    for p in pairs:
        forms[p["form"]].add(p["root"])
    assert all(len(r) == 1 for r in forms.values())


def test_checkout_output_is_deterministic():
    _need_dataset()
    a, b = _checkout_pairs(), _checkout_pairs()
    assert a == b


def test_gold_backed_examples():
    _need_dataset()
    pairs = {p["record_id"]: p for p in D.load_dictionary_pairs(DATASET_DIR)}
    entries = {e["id"]: e for e in D.read_dictionary_entries(DATASET_DIR)}
    gold = {r["record_id"]: r for r in D.read_records(D.gold_file_path(DATASET_DIR, "train"))}
    for i, headword, pos, gold_id, root in GOLD_BACKED:
        assert entries[i]["headword_geez"] == headword and gold[gold_id]["root"] == root
        p = pairs[str(i)]
        assert (p["form"], p["root"], p["type"]) == (headword, root, pos)


def test_examples_left_out_in_the_checkout():
    _need_dataset()
    stats = collections.Counter()
    ids = {i for p in D.load_dictionary_pairs(DATASET_DIR, stats) for i in p["record_ids"]}
    entries = {e["id"]: e for e in D.read_dictionary_entries(DATASET_DIR)}
    gold = {r["record_id"]: r for r in D.read_records(D.gold_file_path(DATASET_DIR, "train"))}
    i, form, gold_id, row = IN_GOLD_TRAIN
    assert entries[i]["headword_geez"] == form and form in D.clean_variants(gold[gold_id]["table_data"][row]["geez_form"])[0]
    a, b, form = CONFLICTING
    assert entries[a]["headword_geez"] == entries[b]["headword_geez"] == form
    i, form, val_id, root = GOLD_VAL_ROOT
    val = {r["record_id"]: r for r in D.read_records(D.gold_file_path(DATASET_DIR, "val"))}
    assert entries[i]["headword_geez"] == form and val[val_id]["root"] == root
    assert not {str(IN_GOLD_TRAIN[0]), str(a), str(b), str(GOLD_VAL_ROOT[0])} & ids
    assert stats["headwords with conflicting roots"] > 0
