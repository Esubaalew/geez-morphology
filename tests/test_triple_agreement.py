"""Generated forms against each root's own Kidane Wold triple, and the listed exceptions.

The exception files in data/ are checked by sha256:
- triple_check_exceptions.tsv: triple_followed rows fix the generator, tagged_type_b rows tag type_b,
  S-a2-l gets the type_b subjunctive override, generator_kept rows keep the generator (the book form is a
  misprint or misread), w-kh-T is off the allow-list (CHECK), covered_by_rule rows fall under the collapsed-subjunctive rule.
- final_y_past_stem_roots.tsv: the -y past stem (l-f-y, l-q-y, m-l-y, s-t-y, a-b-y; m-s-y and s-b-y not confirmed yet).
- triple_check_exceptions_extra.tsv: S-b-r, kh-r-s and m-l-a: the generator stands; the final-guttural 4th order only
  where the guttural ends the word; a-b-y joins the -y rule; n-s2-a: the generator stands (a known mismatch with a gold
  train cell).
- class_changes.tsv: one row per class change.
- phantom_root_qh_h2_m.tsv: the restored row for qh-h2-m is withdrawn (its Type D triple was an OCR misread of
  entry 360), so qh-h2-m stays Type A.
Expected forms are built from root letters (geez_script.render_unit), so this file types no Ge'ez letters.
"""
import csv
import hashlib
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import geez_script  # noqa: E402
from conftest import DATASET_DIR  # noqa: E402
import generate_synthetic_data as gen  # noqa: E402
import verb_classes  # noqa: E402

DATASET = DATASET_DIR
EXCEPTIONS_FILE = os.path.join(HERE, "data", "triple_check_exceptions.tsv")
EXTRA_FILE = os.path.join(HERE, "data", "triple_check_exceptions_extra.tsv")
CLASS_CHANGES_FILE = os.path.join(HERE, "data", "class_changes.tsv")
FINAL_Y_FILE = os.path.join(HERE, "data", "final_y_past_stem_roots.tsv")
PHANTOM_FILE = os.path.join(HERE, "data", "phantom_root_qh_h2_m.tsv")
ALLOWED = {"-".join(r) for r in gen.read_allowed_roots_file()}
BLOCKED = gen.read_blocked_forms_file()


def _w(*units):
    return "".join(geez_script.render_unit(c, v) for c, v in units)


def need_allowed(root):
    if root not in ALLOWED:
        pytest.skip(f"{root} is not on the allow-list: no synthetic lines")


def exceptions():
    with open(EXCEPTIONS_FILE, encoding="utf-8") as f:
        return {r["root"]: r for r in csv.DictReader(f, delimiter="\t")}


@pytest.fixture(scope="module")
def synth_all():
    paths = [os.path.join(DATASET, "data", "paradigms", "synthetic_morphology_train.jsonl"),
             os.path.join(DATASET, "data", "synthetic", "synthetic_forms_for_val_test_roots.jsonl")]
    if not all(os.path.exists(p) for p in paths):
        pytest.skip("synthetic data not generated")
    out = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            out += [json.loads(line) for line in f]
    return out


@pytest.fixture(scope="module")
def cells(synth_all):
    """(root, tense, person+gender+number) -> set of basic-stem forms without object or negative."""
    out = {}
    for r in synth_all:
        f = r["features"]
        if f.get("stem") == "basic" and "object" not in f and not f.get("negative"):
            out.setdefault((f["root"], f["tense"], f["person"] + (f.get("gender") or "") + f["number"]), set()).add(r["surface_form"])
    return out


@pytest.fixture(scope="module")
def triples():
    path = os.path.join(DATASET, "data", "lexicon", "kidane_wold_verb_triples.jsonl")
    if not os.path.exists(path):
        pytest.skip("no geez-dataset checkout")
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            x = json.loads(line)
            if x.get("root"):
                out.setdefault(x["root"], []).append(x)
    return out


def one(cells, root, tense, per):
    v = cells.get((root, tense, per), set())
    assert len(v) <= 1, (root, tense, per, v)
    return next(iter(v), None)


def test_exception_files_unchanged():
    with open(EXCEPTIONS_FILE, "rb") as f:
        assert hashlib.sha256(f.read()).hexdigest().startswith("19c24527")
    kinds = [r["decision"] for r in exceptions().values()]
    assert kinds.count("triple_followed") == 27 and kinds.count("generator_kept") == 24
    assert kinds.count("tagged_type_b") == 3 and kinds.count("covered_by_rule") == 3
    with open(EXTRA_FILE, "rb") as f:
        assert hashlib.sha256(f.read()).hexdigest().startswith("2347efd6")
    with open(CLASS_CHANGES_FILE, "rb") as f:
        assert hashlib.sha256(f.read()).hexdigest().startswith("2a98caed")
    with open(PHANTOM_FILE, "rb") as f:
        assert hashlib.sha256(f.read()).hexdigest().startswith("3cf2e4df")


def extra_exceptions():
    with open(EXTRA_FILE, encoding="utf-8") as f:
        return {r["item"]: r for r in csv.DictReader(f, delimiter="\t")}


# ------------------------------------------------------------------ final guttural (ይስማዕ pattern)
FINAL_GUTT = sorted(r for r, x in exceptions().items() if x["reason"].startswith("Type A with a final guttural"))


def test_final_guttural_roots_reached():
    assert len(FINAL_GUTT) == 14
    if not os.path.exists(os.path.join(DATASET, verb_classes.TRIPLES_FILE)):
        pytest.skip("no geez-dataset checkout")
    found = {"-".join(r) for r in verb_classes.kw_final_guttural_subjunctive(DATASET)}
    assert set(FINAL_GUTT) <= found


@pytest.mark.parametrize("root", FINAL_GUTT)
def test_final_guttural_follows_own_triple(cells, triples, root):
    need_allowed(root)
    want = {x["triple"][2] for x in triples[root]}
    got = cells.get((root, "subjunctive", "3ms"), set())
    assert got & want or want & BLOCKED, (root, got, want)   # g-n-h2: the form is made, then left out as a gold val/test form


@pytest.mark.parametrize("root", ["f-t-h2", "T-g-a2"])
def test_final_guttural_orders(cells, root):
    """4th order on radical 2 only where the guttural ends the word; before any vowel suffix, -ā included, 6th order."""
    need_allowed(root)
    c1, c2, c3 = root.split("-")
    assert one(cells, root, "subjunctive", "3ms") == _w(("y", "I"), (c1, "I"), (c2, "A"), (c3, "I"))
    assert one(cells, root, "subjunctive", "3fp") == _w(("y", "I"), (c1, "I"), (c2, "I"), (c3, "A"))
    assert one(cells, root, "subjunctive", "2fs") == _w(("t", "I"), (c1, "I"), (c2, "I"), (c3, "i"))
    assert one(cells, root, "imperative", "2fp") == _w((c1, "I"), (c2, "I"), (c3, "A"))
    assert one(cells, root, "subjunctive", "3mp") == _w(("y", "I"), (c1, "I"), (c2, "I"), (c3, "u"))
    assert one(cells, root, "imperative", "2ms") == _w((c1, "I"), (c2, "A"), (c3, "I"))
    assert one(cells, root, "imperative", "2fs") == _w((c1, "I"), (c2, "I"), (c3, "i"))
    assert one(cells, root, "imperative", "2mp") == _w((c1, "I"), (c2, "I"), (c3, "u"))


def test_final_guttural_object_forms_keep_6th(synth_all):
    """An object suffix gives radical 3 a vowel, so radical 2 stays 6th (gold train ምርሐኒ)."""
    roots = {"-".join(r) for r in verb_classes.kw_final_guttural_subjunctive(DATASET)} & ALLOWED
    seen = 0
    for r in synth_all:
        f = r["features"]
        if f["root"] in roots and f.get("stem") == "basic" and "object" in f and f["tense"] in ("subjunctive", "imperative"):
            dec = [geez_script.decompose_char(ch) for ch in r["surface_form"]]
            c2 = f["root"].split("-")[1]
            assert any(d == (c2, "I") for d in dec), r
            seen += 1
    assert seen


# ------------------------------------------------------------------ initial guttural with final w/y: present የ-
YA = sorted(r for r, x in exceptions().items() if x["reason"].startswith("initial guttural with final w/y"))


@pytest.mark.parametrize("root", YA)
def test_guttural_present_prefix(cells, triples, root):
    need_allowed(root)
    pres = one(cells, root, "present", "3ms")
    assert pres in {x["triple"][1] for x in triples[root]}
    assert geez_script.decompose_char(pres[0]) == ("y", "a")


# ------------------------------------------------------------------ initial w, hollow, quadriliteral
@pytest.mark.parametrize("root", ["w-S2-a", "w-k-a2", "w-q-a2"])
def test_initial_w_dropped_final_guttural(cells, root):
    need_allowed(root)
    _, c2, c3 = root.split("-")
    assert one(cells, root, "subjunctive", "3ms") == _w(("y", "I"), (c2, "A"), (c3, "I"))
    assert one(cells, root, "subjunctive", "3mp") == _w(("y", "I"), (c2, "A"), (c3, "u"))
    assert one(cells, root, "imperative", "2ms") == _w((c2, "A"), (c3, "I"))


def test_initial_w_kept_w_k_k(cells):
    need_allowed("w-k-k")
    assert one(cells, "w-k-k", "subjunctive", "3ms") == _w(("y", "I"), ("w", "I"), ("k", "I"), ("k", "I"))
    assert one(cells, "w-k-k", "present", "3ms") == _w(("y", "I"), ("w", "a"), ("k", "I"), ("k", "I"))


def test_initial_w_other_roots_unchanged(cells):
    assert one(cells, "w-h2-n", "present", "3ms") == _w(("y", "I"), ("w", "a"), ("h2", "I"), ("n", "I"))


def test_quadriliteral_d_w_r_h1_not_contracted(cells):
    need_allowed("d-w-r-h1")
    assert one(cells, "d-w-r-h1", "present", "3ms") == _w(("y", "I"), ("d", "a"), ("w", "a"), ("r", "I"), ("h1", "I"))


def test_hollow_final_guttural_q_w_a2(cells):
    need_allowed("q-w-a2")
    assert one(cells, "q-w-a2", "subjunctive", "3ms") == _w(("y", "I"), ("q", "A"), ("a2", "I"))
    assert one(cells, "q-w-a2", "present", "3ms") == _w(("y", "I"), ("q", "a"), ("w", "I"), ("a2", "I"))


def test_q_n_S_present_made_and_left_out_as_gold_form(cells):
    """q-n-S gets its present. The generator makes it; it is a gold val/test form, so the line is left out."""
    need_allowed("q-n-S")
    pres = _w(("y", "I"), ("q", "a"), ("n", "I"), ("S", "I"))
    assert pres in BLOCKED
    assert one(cells, "q-n-S", "subjunctive", "3ms") == _w(("y", "I"), ("q", "I"), ("n", "I"), ("S", "I"))


# ------------------------------------------------------------------ type_b tags and the S-a2-l override
def test_type_b_tags(cells, triples):
    assert {r for r, c in verb_classes.CLASS_TAG_EXCEPTIONS.items() if c == "type_b"} == {("T", "y", "r"), ("r", "y", "S2"), ("w", "S2", "f")}
    assert {r for r, c in verb_classes.CLASS_TAG_EXCEPTIONS.items() if c == "type_c"} == {("d", "r", "g"), ("n", "z", "z"), ("q", "r", "h2")}
    for root in ("T-y-r", "r-y-S2", "w-S2-f"):
        need_allowed(root)
        c1, c2, c3 = root.split("-")
        assert one(cells, root, "present", "3ms") in {x["triple"][1] for x in triples[root]}
        assert one(cells, root, "subjunctive", "3ms") == _w(("y", "I"), (c1, "a"), (c2, "I"), (c3, "I"))   # regular type_b


NOT_OVERRIDDEN = ["n-z-r", "s-b-h2", "s-d-s", "f-S2-m", "g-r-m", "n-S2-r"]


def test_S_a2_l_override_scope(cells):
    orders = verb_classes.type_b_subjunctive_orders(DATASET)
    assert ("S", "a2", "l") in orders
    assert all(len(r) == 3 and r[1] in verb_classes.GUTTURALS for r in orders)
    assert not {tuple(r.split("-")) for r in NOT_OVERRIDDEN} & set(orders)
    need_allowed("S-a2-l")
    assert one(cells, "S-a2-l", "subjunctive", "3ms") == _w(("y", "I"), ("S", "I"), ("a2", "a"), ("l", "I"))


@pytest.mark.parametrize("root", NOT_OVERRIDDEN)
def test_type_b_generator_stands(cells, root):
    need_allowed(root)
    c1, c2, c3 = root.split("-")
    assert one(cells, root, "subjunctive", "3ms") == _w(("y", "I"), (c1, "a"), (c2, "I"), (c3, "I"))


# ------------------------------------------------------------------ -y past stem
def test_y_past_stem_lists():
    with open(FINAL_Y_FILE, encoding="utf-8") as f:
        rows = list(csv.DictReader((line for line in f if not line.startswith("#")), delimiter="\t"))
    confirmed = {tuple(r["root"].split("-")) for r in rows if r["status"] == "CONFIRMED"}
    assert confirmed == verb_classes.Y_PAST_C2_FIRST
    assert verb_classes.Y_PAST_C2_FIRST_PENDING == {("m", "s", "y"), ("s", "b", "y")}
    assert not verb_classes.Y_PAST_C2_FIRST & verb_classes.Y_PAST_C2_FIRST_PENDING


@pytest.mark.parametrize("root", ["l-f-y", "l-q-y", "m-l-y", "s-t-y", "a-b-y"])
def test_y_past_stem_confirmed(cells, root):
    need_allowed(root)
    c1, c2, c3 = root.split("-")
    assert one(cells, root, "past", "3ms") == _w((c1, "a"), (c2, "a"), (c3, "a"))       # ሰተየ, not ሰትየ
    assert one(cells, root, "past", "3fs") == _w((c1, "a"), (c2, "a"), (c3, "a"), ("t", "I"))
    assert one(cells, root, "past", "3mp") == _w((c1, "a"), (c2, "a"), (c3, "u"))
    assert one(cells, root, "past", "1cs") == _w((c1, "a"), (c2, "a"), (c3, "I"), ("k", "u"))   # ሰተይኩ stays


@pytest.mark.parametrize("root", ["m-s-y", "s-b-y", "g-l-y"])
def test_y_past_stem_waiting_unchanged(cells, root):
    """m-s-y and s-b-y are not confirmed yet; g-l-y stays as the book writes it."""
    need_allowed(root)
    c1, c2, c3 = root.split("-")
    assert one(cells, root, "past", "3ms") == _w((c1, "a"), (c2, "I"), (c3, "a"))
    assert one(cells, root, "past", "1cs") == _w((c1, "a"), (c2, "a"), (c3, "I"), ("k", "u"))


# ------------------------------------------------------------------ allow-list
@pytest.mark.parametrize("root", ["w-kh-T", "w-s-n", "g-n-f-l", "T-b-l-l", "z-n-g-a2", "q-y-a"])
def test_not_on_allow_list(root):
    """w-kh-T: CHECK in the triple check. w-s-n, g-n-f-l, T-b-l-l, z-n-g-a2, q-y-a: left out by hand."""
    assert root not in ALLOWED


# ------------------------------------------------------------------ triple agreement
# Roots whose 3ms basic present or subjunctive differs from their own triple and that are not in the exception
# files; the generator's forms are kept for them.
KNOWN_DISAGREEMENTS = {
    "S-f-q", "S-w-a2", "S-w-n", "T-y-q", "a-b-s", "a-m-r", "a-n-r", "a-z-l", "a2-S-r", "a2-b-T", "a2-f-r", "a2-m-d",
    "a2-m-r", "a2-s-b", "d-d-q", "d-kh-n", "d-kh-r", "d-y-b", "f-g-a2", "f-kh-r", "f-w-l", "f-w-q", "f-w-s", "f-y-t",
    "g-y-r", "h2-d-m", "h2-d-s", "kh-T-a", "kh-m-s", "kh-y-T", "kh-y-s", "m-l-h2", "q-t-r", "qh-T-f", "s-b-k",
    "s-w-r", "s-y-f", "s2-l-l", "w-T-n", "w-k-l", "w-q-s2", "w-r-d"}


def _disagreeing(cells, triples):
    out = {}
    with_lines = {k[0] for k in cells}
    for root, xs in triples.items():
        if root not in ALLOWED or root not in with_lines:
            continue
        bad = []
        for slot, i in (("present", 1), ("subjunctive", 2)):
            want = {x["triple"][i] for x in xs}
            if not (cells.get((root, slot, "3ms"), set()) & want) and not (want & BLOCKED):
                bad.append(slot)
        if bad:
            out[root] = bad
    return out


def test_triple_agreement_listed_or_known(cells, triples):
    rul = exceptions()
    listed = {r for r, x in rul.items() if x["decision"] in ("generator_kept", "covered_by_rule")}
    listed |= {r for r, x in extra_exceptions().items() if x["decision"] == "generator_kept"}   # S-b-r, kh-r-s, m-l-a
    listed |= {r for r, x in class_changes().items() if x["decision"].startswith("RESTORE_")}   # gold class outranks the new triple
    dis = _disagreeing(cells, triples)
    # w-S2-f: present from the book, subjunctive by the regular type_b rule (the book's is the minority shape)
    if dis.get("w-S2-f") == ["subjunctive"]:
        dis.pop("w-S2-f")
    left = set(dis) - listed - KNOWN_DISAGREEMENTS
    assert not left, sorted(left)
    assert not {r for r, x in rul.items() if x["decision"].startswith("a_")} & set(dis) - {"w-S2-f"}


def test_m_a2_z_misprint(cells):
    """generator_kept: the book's ይምዐዘ is a misprint for ይምዐዝ, which the generator gives."""
    need_allowed("m-a2-z")
    assert one(cells, "m-a2-z", "subjunctive", "3ms") == _w(("y", "I"), ("m", "I"), ("a2", "a"), ("z", "I"))


def test_extra_generator_stands(cells):
    """S-b-r is Type B (C1 1st order in the subjunctive); kh-r-s is the initial-guttural Type A (present with the
    1st-order prefix, subjunctive C1 6th); m-l-a's present takes the 6th-order prefix (m is not a guttural)."""
    assert {r for r, x in extra_exceptions().items() if x["decision"] == "generator_kept"} == {"S-b-r", "kh-r-s", "m-l-a"}
    need_allowed("S-b-r")
    assert one(cells, "S-b-r", "subjunctive", "3ms") == _w(("y", "I"), ("S", "a"), ("b", "I"), ("r", "I"))
    need_allowed("kh-r-s")
    assert one(cells, "kh-r-s", "present", "3ms") == _w(("y", "a"), ("kh", "a"), ("r", "I"), ("s", "I"))
    assert one(cells, "kh-r-s", "subjunctive", "3ms") == _w(("y", "I"), ("kh", "I"), ("r", "I"), ("s", "I"))
    need_allowed("m-l-a")
    assert one(cells, "m-l-a", "present", "3ms") == _w(("y", "I"), ("m", "a"), ("l", "I"), ("a", "I"))
    assert one(cells, "m-l-a", "subjunctive", "3ms") == _w(("y", "I"), ("m", "I"), ("l", "A"), ("a", "I"))


def test_n_s2_a_imperative(cells):
    """The imperative 2ms takes the final-guttural rule; gold train's 6th-order spelling is a known mismatch."""
    assert extra_exceptions()["n-s2-a"]["decision"].startswith("generator_kept")
    need_allowed("n-s2-a")
    assert one(cells, "n-s2-a", "imperative", "2ms") == _w(("n", "I"), ("s2", "A"), ("a", "I"))


# ------------------------------------------------------------------ class changes
def class_changes():
    with open(CLASS_CHANGES_FILE, encoding="utf-8") as f:
        return {r["root"]: r for r in csv.DictReader(f, delimiter="\t")}


@pytest.fixture(scope="module")
def verb_type(synth_all):
    out = {}
    for r in synth_all:
        f = r["features"]
        if f.get("stem") == "basic":
            out.setdefault(f["root"], set()).add(f.get("verb_type"))
    return out


CLASS_OF_CHANGE = {"A": "strong_A", "B": "strong_B", "C": "strong_C", "D": "strong_D", "weak_middle": "weak_middle"}


def restores_not_applied():
    """Roots whose restored Type D row does not apply (phantom_root_qh_h2_m.tsv): they keep the class the triples give them."""
    with open(PHANTOM_FILE, encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    return {r["item"].split(" / ")[1]: "strong_A" for r in rows if r["decision"] == "restore_not_applied"}


def test_class_change_outcome(verb_type):
    """Every class change has a row; RESTORE rows bring the old class back, the others give the new one."""
    changes = class_changes()
    assert len(changes) == 39
    gone = restores_not_applied()
    assert gone == {"qh-h2-m": "strong_A"}
    for root, x in changes.items():
        need = x["decision"]
        old, new = x["change"].split("->")
        want = gone.get(root) or {"restored": "strong_C", "restored": "strong_D"}.get(need, CLASS_OF_CHANGE.get(new))
        if root not in ALLOWED:
            continue
        assert verb_type.get(root) == {want}, (root, x["change"], need, verb_type.get(root))
