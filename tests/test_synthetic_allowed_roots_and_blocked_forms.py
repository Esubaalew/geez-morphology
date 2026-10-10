"""The two files that limit the synthetic data.

- data/synthetic_roots_allowed_v5.txt: the only roots the generator conjugates (roots with a dictionary
  verb entry, a gold root or a Kidane Wold verb triple, plus 8 roots kept by hand: m-h2-l, m-s-l, S-w-t, b-y-S,
  f-r-S, h2-r-g, m-s-k, w-k-a2; m-kh-l is left out). The generator stops if the file is missing.
- data/heldout_gold_forms.tsv: every gold val/test form of dataset branch crosscheck-1009. A synthetic line
  with one of these forms is not written; the root's other lines do not change. Missing file: the generator stops.
- split_heldout_synthetic.py drops a training line with a gold val/test form, and --check fails when a dataset
  synthetic line has one.
Any change to the two files shows in the sha256 tests.
"""
import contextlib
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import engine as engine_module  # noqa: E402
import generate_synthetic_data as gen  # noqa: E402
import geez_script  # noqa: E402
import split_heldout_synthetic as H  # noqa: E402
from conftest import DATASET_DIR  # noqa: E402

ALLOWED_SHA256 = "84fc1851d7d731eadaa0aaca179ee003be717114af410a50168659dd6df58fc1"
BLOCKED_SHA256 = "5966f03ede248970a007e1fbea5b6e093dabf078b20ab59dfc3398aef37c397e"
TRILITERAL = ["q-t-l", "g-b-r", "n-g-r", "s-b-r", "k-t-b", "d-m-r", "f-t-l", "m-s-l"]
QUADRILITERAL = ["d-n-g-s", "s-n-b-t", "t-r-g-m", "m-s-k-r", "q-n-d-l", "b-r-b-r"]
BLOCKED_HEADER = "geez_form\tsplit\trecord_id\trow\tvariant\n"


def _sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _word(*units):
    """Ethiopic word from (consonant symbol, vowel) units, so the test types no Ge'ez letters."""
    return "".join(geez_script.render_unit(c, v) for c, v in units)


def test_files_are_unchanged():
    assert _sha(gen.ALLOWED_ROOTS_FILE) == ALLOWED_SHA256
    assert _sha(gen.BLOCKED_FORMS_FILE) == BLOCKED_SHA256


def test_allow_list_has_1234_roots():
    roots = gen.read_allowed_roots_file()
    assert len(roots) == 1234
    # roots kept by hand without dictionary, gold or triple evidence
    restored = ["m-h2-l", "m-s-l", "S-w-t", "b-y-S", "f-r-S", "h2-r-g", "m-s-k", "w-k-a2"]
    assert {tuple(r.split("-")) for r in restored} <= roots
    assert ("m", "kh", "l") not in roots        # left out by hand
    # left out: no verb evidence after the Kidane Wold rebuild, two gold-only roots, spelling twins
    for r in ["s-T-n", "s-h2-g", "m-S2-a", "w-l-a2", "s2-w-a2", "w-a2-f", "S-w-a", "a2-s-r", "h2-d-r", "kh-q-f", "s2-m-k",
              "a-s2-r", "b-w-kh", "d-kh-l", "n-s2-q", "n-z-a"]:
        assert tuple(r.split("-")) not in roots, r
    # 11 roots added from corrected dictionary roots, and the spellings the twins move to
    for r in ["a2-g-l", "a2-g-t", "a2-m-S2", "a2-z-r", "b-S2-a2", "g-f-a2", "gw-d-a2", "m-a2-a2", "r-s-a2", "s2-a2-l", "s2-r-a2",
              "S-w-a2", "a2-s2-r", "kh-d-r", "h2-q-f", "s-m-k", "a-s-r", "m-h2-S"]:
        assert tuple(r.split("-")) in roots, r
    assert os.path.basename(gen.ALLOWED_ROOTS_FILE) == "synthetic_roots_allowed_v5.txt"
    assert [f for f in os.listdir(os.path.join(ROOT, "data")) if f.startswith("synthetic_roots_allowed")] == \
        ["synthetic_roots_allowed_v5.txt"]
    assert all(len(r) in (3, 4) for r in roots)


def test_blocked_forms_file_has_every_val_and_test_row():
    forms = gen.read_blocked_forms_file()
    assert len(forms) == 2854
    with open(gen.BLOCKED_FORMS_FILE, encoding="utf-8") as f:
        rows = [line.rstrip("\n").split("\t") for line in f if not line.startswith("#")]
    header, rows = rows[0], rows[1:]
    assert header == ["geez_form", "split", "record_id", "row", "variant", "source_row", "source_variant", "root", "cell"]
    assert len(rows) == 2934
    assert len({tuple(r[1:5]) for r in rows}) == len(rows)          # key: split + record_id + row + variant
    assert {r[1] for r in rows} == {"val", "test"}
    assert sum(1 for r in rows if not r[7]) == 203                     # rows with no root are in


def test_missing_files_stop_the_generator():
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(FileNotFoundError):
            gen.read_allowed_roots_file(os.path.join(tmp, "none.txt"))
        with pytest.raises(FileNotFoundError):
            gen.read_blocked_forms_file(os.path.join(tmp, "none.tsv"))
        out = os.path.join(tmp, "out")
        with pytest.raises(FileNotFoundError):
            gen.generate_extended_dataset(dataset_dir=tmp, output_dir=out,
                                          allowed_roots_file=os.path.join(tmp, "none.txt"))
        assert not os.path.exists(os.path.join(out, "synthetic_morphology_train.jsonl"))


def _dataset(folder, allowed, blocked_forms):
    with open(os.path.join(folder, "dictionary.jsonl"), "w", encoding="utf-8") as f:
        for i, r in enumerate(TRILITERAL + QUADRILITERAL, 1):
            f.write(json.dumps({"id": i, "inferred_root": r}) + "\n")
    with open(os.path.join(folder, "allowed.txt"), "w", encoding="utf-8") as f:
        f.write("# fixture\n" + "\n".join(allowed) + "\n")
    with open(os.path.join(folder, "blocked.tsv"), "w", encoding="utf-8") as f:
        f.write("# fixture\n" + BLOCKED_HEADER + "".join(f"{x}\tval\tr\t0\t0\n" for x in blocked_forms))


def _generate(allowed, blocked_forms=()):
    non_hollow = set(engine_module.NON_HOLLOW)
    with tempfile.TemporaryDirectory() as tmp:
        _dataset(tmp, allowed, blocked_forms)
        out = os.path.join(tmp, "out")
        log = io.StringIO()
        try:
            with contextlib.redirect_stdout(log):
                gen.generate_extended_dataset(dataset_dir=tmp, output_dir=out,
                                              allowed_roots_file=os.path.join(tmp, "allowed.txt"),
                                              blocked_forms_file=os.path.join(tmp, "blocked.tsv"))
        finally:
            engine_module.NON_HOLLOW.clear()
            engine_module.NON_HOLLOW.update(non_hollow)
        with open(os.path.join(out, "synthetic_morphology_train.jsonl"), encoding="utf-8") as f:
            train = f.read().splitlines()
        with open(os.path.join(out, "synthetic_morphology_analyze.jsonl"), encoding="utf-8") as f:
            analyze = f.read().splitlines()
    return train, analyze, log.getvalue()


def _root(line):
    return json.loads(line)["features"]["root"]


def test_root_off_the_allow_list_loses_only_its_own_lines():
    full, _, _ = _generate(TRILITERAL + QUADRILITERAL)
    assert any(_root(l) == "n-g-r" for l in full)
    fewer, _, log = _generate([r for r in TRILITERAL + QUADRILITERAL if r != "n-g-r"])
    assert fewer == [l for l in full if _root(l) != "n-g-r"]
    assert "Allow-list allowed.txt" in log


def test_blocked_form_drops_only_its_lines():
    full, full_an, _ = _generate(TRILITERAL + QUADRILITERAL)
    # a base form of q-t-l taken from the output, and its object/negative lines must stay
    form = next(json.loads(l)["surface_form"] for l in full if _root(l) == "q-t-l")
    after, after_an, log = _generate(TRILITERAL + QUADRILITERAL, [form])
    assert after == [l for l in full if json.loads(l)["surface_form"] != form]
    assert after_an == [l for l in full_an if json.loads(l)["surface_form"] != form]
    assert len(after) < len(full)
    blocked = json.loads(next(x for x in log.splitlines() if x.startswith("BLOCKED_GOLD_FORMS_JSON "))
                         .split(" ", 1)[1])
    assert list(blocked) == ["q-t-l"] and blocked["q-t-l"]["lines"] == len(full) - len(after)


def _heldout_setup(tmp, train_forms):
    os.makedirs(os.path.join(tmp, "ds", "data", "paradigms"))
    os.makedirs(os.path.join(tmp, "full"))
    rootless = _word(("k", "i"), ("d", "A"))                                  # a val row with no root
    val = {"root": "", "table_data": [{"geez_form": rootless}]}
    test = {"root": geez_script.to_ethiopic_root("s-m-a2"),
            "table_data": [{"geez_form": _word(("s", "a"), ("m", "I"), ("a2", "a"))}]}
    for name, rec in (("merho_ges_gold_val.jsonl", val), ("merho_ges_gold_test.jsonl", test)):
        with open(os.path.join(tmp, "ds", "data", "paradigms", name), "w", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    feats = {"person": "3", "gender": "m", "number": "s", "tense": "past", "stem": "basic", "root": "k-y-d"}
    with open(os.path.join(tmp, "full", H.TRAIN), "w", encoding="utf-8") as ft, \
            open(os.path.join(tmp, "full", H.ANALYZE), "w", encoding="utf-8") as fa:
        for form in train_forms:
            ft.write(json.dumps({"features": feats, "surface_form": form}, ensure_ascii=False) + "\n")
            fa.write(json.dumps({"surface_form": form, "analysis": feats}, ensure_ascii=False) + "\n")
    return rootless


def _check(tmp):
    return subprocess.run([sys.executable, os.path.join(ROOT, "split_heldout_synthetic.py"), "--full_dir", os.path.join(tmp, "full"),
                           "--dataset_dir", os.path.join(tmp, "ds"), "--check"], capture_output=True, text=True, cwd=ROOT)


def test_check_fails_on_a_gold_val_or_test_form():
    with tempfile.TemporaryDirectory() as tmp:
        rootless = _heldout_setup(tmp, [_word(("k", "a"), ("y", "a"), ("d", "a"))])
        H.split_into_dataset(os.path.join(tmp, "full"), os.path.join(tmp, "ds"))
        ok = _check(tmp)
        assert ok.returncode == 0, ok.stdout + ok.stderr
        assert "gold val/test form: 0" in ok.stdout
    with tempfile.TemporaryDirectory() as tmp:
        rootless = _heldout_setup(tmp, [_word(("k", "a"), ("y", "a"), ("d", "a")), _word(("k", "i"), ("d", "A"))])
        assert rootless in H.gold_eval_forms(os.path.join(tmp, "ds"))
        _, _, dropped = H.split_into_dataset(os.path.join(tmp, "full"), os.path.join(tmp, "ds"))
        assert [json.loads(x)["surface_form"] for x in dropped] == [rootless]   # the split drops it from the training pair
        assert _check(tmp).returncode == 0
        for p in H.targets(os.path.join(tmp, "ds"))[:2]:                        # a line written past the split
            with open(p, encoding="utf-8") as f:
                kept = f.readlines()
            with open(p, "a", encoding="utf-8") as f:
                f.write(kept[0].replace(json.loads(kept[0])["surface_form"], rootless))
        bad = _check(tmp)
        assert bad.returncode == 1
        assert "gold val/test form: 2" in bad.stdout                            # train + analyze line
        assert "folded gold val/test form: 2" in bad.stdout


def test_form_variants_match_check_dataset():
    cell = " / ".join([_word(("q", "a"), ("t", "a"), ("l", "a")), _word(("q", "a"), ("T", "a"), ("l", "a"))])
    assert H.form_variants(cell + " (" + _word(("n", "a")) + ")") == cell.split(" / ")


@pytest.mark.skipif(not os.path.exists(os.path.join(DATASET_DIR, "data", "paradigms", "merho_ges_gold_val.jsonl")),
                    reason="no geez-dataset checkout")
def test_dataset_synthetic_files_have_no_gold_val_or_test_form():
    assert H.gold_form_lines(DATASET_DIR) == {}
