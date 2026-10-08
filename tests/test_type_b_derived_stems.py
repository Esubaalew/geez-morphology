"""Passive, causative and aste rows of Type B roots (verb_class_paradigms.type_b_derived, verb_classes.type_b_derived_rows).

- Gold-backed: the form of a Merho Ges gold train cell (read from the gold train file by record_id) is the form the
  generator writes for that root and cell.
- A root whose last radical is a guttural gets no 1st/2nd-person causative past and no causative imperative.
- The causative of the other Type B roots is the Type A causative of the same radicals (patterns_triliteral.json).
- Aste is past only, as for Type A. paradigm_rows (read by the analyzer and cli.py) stays basic only.
- A type_b root with a w or y radical, or with a guttural middle radical (S-a2-l), gets no derived rows; in a generator
  run a type_b root gets them and the other roots' lines are unchanged.
Roots are written with the Latin consonant symbols; the test types no Ge'ez letters.
"""
import contextlib
import io
import json
import os
import sys
import tempfile

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import generate_synthetic_data as gen  # noqa: E402
import geez_script  # noqa: E402
import verb_classes  # noqa: E402
from conftest import DATASET_DIR  # noqa: E402
from engine import InterdigitationEngine, slots_from_spec  # noqa: E402

GOLD_TRAIN = os.path.join(DATASET_DIR, verb_classes.GOLD_TRAIN_FILE)

# (record_id, root, stem, tense, person, gender, number) of gold train cells of Type B roots
GOLD_CELLS = [
    ("624fff1d0291", "q-S-l", "passive", "past", "3", "m", "s"),
    ("a2f0a250c95a", "f-s2-h2", "passive", "past", "3", "m", "s"),
    ("df0e2be52bfa", "l-q-h2", "passive", "present", "3", "m", "s"),
    ("d85c40efb920", "n-z-r", "causative", "past", "3", "m", "s"),
    ("d85c40efb920", "n-z-r", "causative", "past", "1", "c", "p"),
    ("a545a8d1f93e", "n-S-r", "causative", "past", "3", "m", "s"),
    ("a545a8d1f93e", "n-S-r", "causative", "past", "3", "m", "p"),
    ("8eb2091ebbc9", "S-b-r", "causative", "past", "3", "m", "s"),
    ("28190a53ea2b", "S-n-a2", "causative", "past", "3", "m", "s"),
    ("0411911f324e", "m-l-a2", "causative", "past", "3", "m", "s"),
]


def _rows(root):
    return verb_classes.type_b_derived_rows(tuple(root.split("-")))


def _form(root, stem, tense, person, gender, number):
    found = [f for feat, f in _rows(root) if (feat["stem"], feat["tense"], feat["person"], feat["gender"], feat["number"])
             == (stem, tense, person, gender, number)]
    assert len(found) <= 1
    return found[0] if found else None


@pytest.fixture(scope="module")
def gold_records():
    if not os.path.exists(GOLD_TRAIN):
        pytest.skip(f"no dataset checkout at {DATASET_DIR}")
    with open(GOLD_TRAIN, encoding="utf-8") as f:
        return {r["record_id"]: r for r in map(json.loads, f)}


@pytest.mark.parametrize("record_id,root,stem,tense,person,gender,number", GOLD_CELLS)
def test_gold_train_form(gold_records, record_id, root, stem, tense, person, gender, number):
    rec = gold_records[record_id]
    assert geez_script.root_symbols(rec["root"]) == tuple(root.split("-"))
    gold = set()
    for row in rec["table_data"]:
        if ((row.get("stem") or rec.get("stem")), row.get("tense"), row.get("person"), row.get("gender") or "c",
                row.get("number")) == (stem, tense, person, gender, number) and not row.get("object"):
            gold |= {v.strip() for v in row["geez_form"].split("/") if v.strip()}
    assert gold, "cell not in the gold record"
    assert _form(root, stem, tense, person, gender, number) in gold


def test_final_guttural_causative_cells():
    for root in ("S-n-a2", "m-l-a2", "s-b-h2", "l-q-h2"):
        cells = {(feat["stem"], feat["tense"], feat["person"]) for feat, _ in _rows(root)}
        assert ("causative", "past", "3") in cells
        assert not {("causative", "past", "1"), ("causative", "past", "2")} & cells
        assert not any(s == "causative" and t == "imperative" for s, t, _ in cells)
        assert ("causative", "present", "3") in cells and ("causative", "subjunctive", "3") in cells
        assert ("passive", "imperative", "2") in cells


def test_causative_is_the_type_a_causative():
    engine = InterdigitationEngine(renderer=geez_script.render_unit)
    with open(os.path.join(ROOT, "patterns_triliteral.json"), encoding="utf-8") as f:
        templates = [p for p in json.load(f) if p["features"].get("verb_type") == "strong_A"
                     and p["features"].get("stem") == "causative"]
    for root in ("q-d-s", "n-z-r", "S-b-r", "z-k-r"):
        cons = tuple(root.split("-"))
        type_a = {}
        for p in templates:
            feat = gen.normalize_features(p["features"])
            key = (feat["tense"], feat["person"], feat["gender"], feat["number"])
            type_a[key] = engine.generate(cons, slots_from_spec(p["pattern_spec"]), stem="causative")
        type_b = {(feat["tense"], feat["person"], feat["gender"], feat["number"]): form
                  for feat, form in _rows(root) if feat["stem"] == "causative"}
        assert type_b == type_a


def test_aste_past_only_and_features():
    rows = _rows("q-d-s")
    assert {feat["tense"] for feat, _ in rows if feat["stem"] == "aste"} == {"past"}
    assert {feat["stem"] for feat, _ in rows} == {"passive", "causative", "aste"}
    assert {feat["verb_type"] for feat, _ in rows} == {"strong_B"}
    assert len(rows) == 10 + 10 + 10 + 4 + 10 + 10 + 10 + 4 + 10


def test_basic_paradigm_rows_unchanged():
    assert {feat["stem"] for feat, _ in verb_classes.paradigm_rows(("q", "d", "s"), "type_b")} == {"basic"}
    assert {p["features"]["stem"] for p in verb_classes.class_patterns()} == {"basic"}


def test_weak_radical_type_b_root_has_no_derived_rows():
    for root in ("T-y-r", "r-y-S2", "w-S2-f"):
        assert _rows(root) == []


def test_guttural_middle_type_b_root_has_no_derived_rows():
    assert _rows("S-a2-l") == []
    assert {feat["stem"] for feat, _ in verb_classes.paradigm_rows(("S", "a2", "l"), "type_b")} == {"basic"}


def test_generator_writes_derived_rows_for_type_b_only():
    roots = ["q-t-l", "g-b-r", "n-g-r", "s-b-h2", "T-y-r"]
    tags = {("n", "g", "r"): "type_b", ("s", "b", "h2"): "type_b", ("T", "y", "r"): "type_b"}
    original = verb_classes.class_tags

    def run(folder, with_tags):
        def tagged(dataset_dir, exclude=()):
            found, counts = original(dataset_dir, exclude)
            return {**found, **(tags if with_tags else {})}, counts
        verb_classes.class_tags = tagged
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                gen.generate_extended_dataset(dataset_dir=tmp, output_dir=folder, include_quads=False,
                                              allowed_roots_file=os.path.join(tmp, "allowed_roots.txt"),
                                              blocked_forms_file=os.path.join(tmp, "blocked_forms.tsv"))
        finally:
            verb_classes.class_tags = original
        with open(os.path.join(folder, "synthetic_morphology_train.jsonl"), encoding="utf-8") as f:
            return [json.loads(line) for line in f]

    with tempfile.TemporaryDirectory() as tmp:
        with open(os.path.join(tmp, "dictionary.jsonl"), "w", encoding="utf-8") as f:
            for i, r in enumerate(roots, 1):
                f.write(json.dumps({"id": i, "inferred_root": r}) + "\n")
        with open(os.path.join(tmp, "allowed_roots.txt"), "w", encoding="utf-8") as f:
            f.write("# fixture\n" + "\n".join(roots) + "\n")
        with open(os.path.join(tmp, "blocked_forms.tsv"), "w", encoding="utf-8") as f:
            f.write("# fixture\ngeez_form\tsplit\trecord_id\trow\tvariant\n")
        lines = run(os.path.join(tmp, "a"), True)
        untagged = run(os.path.join(tmp, "b"), False)

    def derived(root):
        return [x for x in lines if x["features"]["root"] == root and x["features"]["stem"] != "basic"]

    for root in ("n-g-r", "s-b-h2"):
        got = [(x["features"], x["surface_form"]) for x in derived(root)]
        assert got == verb_classes.type_b_derived_rows(tuple(root.split("-")))
    assert derived("T-y-r") == []
    for root in ("q-t-l", "g-b-r"):
        assert [x for x in lines if x["features"]["root"] == root] == \
            [x for x in untagged if x["features"]["root"] == root]
