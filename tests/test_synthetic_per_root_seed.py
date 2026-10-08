"""Per-root random seeding in generate_synthetic_fixed.py.

A change to one root's class tag must change only that root's lines: every other root's lines
(base, quadriliteral, object and negative) stay identical and in the same order. Runs the
generator on a small dictionary in a temporary folder; the tag change is made by wrapping
verb_classes.class_tags for the second run.
"""
import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import engine as engine_module  # noqa: E402
import generate_synthetic_fixed as gen  # noqa: E402
import verb_classes  # noqa: E402

TRILITERAL = ["q-t-l", "g-b-r", "n-g-r", "s-b-r", "k-t-b", "d-m-r", "f-t-l", "m-s-l", "n-b-r", "l-b-s",
              "s-r-q", "t-k-l", "z-b-t", "f-s-m", "g-d-f", "k-f-l"]
QUADRILITERAL = ["d-n-g-s", "s-n-b-t", "t-r-g-m", "m-s-k-r", "q-n-d-l", "b-r-b-r", "k-l-b-s", "g-l-b-b",
                 "z-n-g-r", "s-l-q-t", "d-b-l-q", "m-z-m-r", "q-r-s-m", "f-l-s-t", "g-m-g-m", "n-z-n-z"]


def _dataset(folder):
    with open(os.path.join(folder, "dictionary.jsonl"), "w", encoding="utf-8") as f:
        for i, r in enumerate(TRILITERAL + QUADRILITERAL, 1):
            f.write(json.dumps({"id": i, "inferred_root": r}) + "\n")
    # the fixture's own allow-list (all its roots) and an empty blocked-forms file
    with open(os.path.join(folder, "allowed_roots.txt"), "w", encoding="utf-8") as f:
        f.write("# fixture\n" + "\n".join(TRILITERAL + QUADRILITERAL) + "\n")
    with open(os.path.join(folder, "blocked_forms.tsv"), "w", encoding="utf-8") as f:
        f.write("# fixture\ngeez_form\tsplit\trecord_id\trow\tvariant\n")


def _generate(ds, out, tag=None):
    """Train lines of one generator run; `tag` = (root, class) set on top of the usual tags."""
    original = verb_classes.class_tags
    non_hollow = set(engine_module.NON_HOLLOW)

    def with_tag(dataset_dir, exclude=()):
        tags, counts = original(dataset_dir, exclude)
        tags = dict(tags)
        tags[tuple(tag[0].split("-"))] = tag[1]
        return tags, counts

    if tag:
        verb_classes.class_tags = with_tag
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            gen.generate_extended_dataset(dataset_dir=ds, output_dir=out,
                                          allowed_roots_file=os.path.join(ds, "allowed_roots.txt"),
                                          blocked_forms_file=os.path.join(ds, "blocked_forms.tsv"))
    finally:
        verb_classes.class_tags = original
        engine_module.NON_HOLLOW.clear()
        engine_module.NON_HOLLOW.update(non_hollow)
    with open(os.path.join(out, "synthetic_morphology_train.jsonl"), encoding="utf-8") as f:
        return f.read().splitlines()


def _root(line):
    return json.loads(line)["features"]["root"]


def _check_only_root_changes(tag):
    with tempfile.TemporaryDirectory() as tmp:
        _dataset(tmp)
        before = _generate(tmp, os.path.join(tmp, "a"))
        after = _generate(tmp, os.path.join(tmp, "b"), tag)
    target = tag[0]
    feats = [json.loads(l)["features"] for l in before if _root(l) != target]
    # the fixture exercises every phase for the other roots
    assert any("object" in f for f in feats) and any(f.get("negative") for f in feats)
    assert any(len(f["root"].split("-")) == 4 for f in feats)
    assert [l for l in before if _root(l) != target] == [l for l in after if _root(l) != target]
    assert [l for l in before if _root(l) == target] != [l for l in after if _root(l) == target]


def test_type_b_tag_on_one_triliteral_root_changes_only_that_root():
    _check_only_root_changes(("n-g-r", "type_b"))


def test_quad_c_tag_on_one_quadriliteral_root_changes_only_that_root():
    # a quad root only enters with a tag if it was not drawn; either way only its own lines may change
    _check_only_root_changes(("q-n-d-l", "quad_c"))


def test_root_rng_is_sha256_based_not_salted_hash():
    value = gen.root_rng("q-t-l", "objects").random()
    assert value == 0.3855267602400665
    assert gen.root_rng("q-t-l", "negative").random() != value
    assert gen.root_rng("q-t-r", "objects").random() != value
    code = ("import sys; sys.path.insert(0, %r); import generate_synthetic_fixed as g; "
            "print(repr(g.root_rng('q-t-l', 'objects').random()))" % ROOT)
    env = dict(os.environ, PYTHONHASHSEED="12345")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd=ROOT)
    assert out.stdout.strip().splitlines()[-1] == repr(value), out.stderr[-500:]
