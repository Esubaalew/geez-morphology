"""The TPU trainer's byte ids, buckets, teacher-forced scores, span corruption, mixture and the check tables."""
import collections
import json
import os
import random
import subprocess
import sys
import types

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from conftest import DATASET_DIR  # noqa: E402
import analyzer  # noqa: E402
import dataset_loader as D  # noqa: E402
import extra_synthetic as X  # noqa: E402
import train_tpu as TT  # noqa: E402
import training_mixture as TM  # noqa: E402

HAS_DATA = os.path.exists(os.path.join(DATASET_DIR or "", TM.SYNTHETIC_FILE))


def test_byte_ids_round_trip_and_eos():
    ids = TT.encode("analyze: ቀተለ", 64)
    assert ids[-1] == TT.EOS and TT.decode(ids) == "analyze: ቀተለ"
    assert len(TT.encode("ቀ" * 100, 32)) == 32


def test_buckets_and_fixed_batch_shapes():
    samples = [{"input_text": "x" * random.Random(i).randint(1, 120), "target_text": "y" * (i % 100 + 1)} for i in range(300)]
    batches = TT.make_batches(samples, 8, 42, 0)
    assert batches and all(len(c) == 8 and a in TT.INPUT_BUCKETS and b in TT.TARGET_BUCKETS for c, a, b in batches)
    assert batches == TT.make_batches(samples, 8, 42, 0) and batches != TT.make_batches(samples, 8, 42, 1)
    dev = TT.make_batches(samples, 8, 0, 0, drop_last=False)
    assert sorted(i for c, _, _ in dev for i in c if i >= 0) == list(range(300))


def test_teacher_forced_scores_match_greedy_rule():
    torch = pytest.importorskip("torch")
    lab = torch.tensor([TT.encode("root: ቀ-ተ-ለ | type: noun", 48)[:40]])
    logits = torch.nn.functional.one_hot(lab, 300).float()
    exact, root = TT.score_batch(logits, lab, torch)
    assert exact.item() and root.item()
    bad = logits.clone()
    bad[0, -3] = 0
    bad[0, -3, 5] = 1                      # an error after the root field
    exact, root = TT.score_batch(bad, lab, torch)
    assert not exact.item() and root.item()
    bad[0, 2] = 0
    bad[0, 2, 5] = 1                       # an error inside the root field
    assert not TT.score_batch(bad, lab, torch)[1].item()


def test_span_corruption_keeps_every_byte_once():
    ids = list(range(10, 240))
    inp, tgt = TT.corrupt(ids, random.Random(0))
    kept = [i for i in inp if i < 259] + [i for i in tgt if i < 259]
    assert sorted(kept) == ids and inp.count(TT.SENTINEL_FIRST) == 1


def test_samples_of_every_task():
    pair = {"form": "ቀተልኩ", "root": "ቀ-ተ-ለ", "type": "verb", "tense": "past", "person": "1", "gender": "",
            "number": "s", "stem": "basic", "object": ""}
    a, g, p = (TM.sample(pair, "x", t) for t in (D.ANALYZE_TASK, TM.GENERATE_TASK, TM.POS_TASK))
    assert a["input_text"] == "analyze: ቀተልኩ" and a["target_text"].startswith("root: ቀ-ተ-ለ | tense: past")
    assert g["input_text"] == "generate: " + a["target_text"] and g["target_text"] == "ቀተልኩ"
    assert p["target_text"] == "type: verb"
    w = TM.sample({"form": "ዝንቱ", "root": None, "type": "demonstrative"}, "x")
    assert w["target_text"] == "root: none | type: demonstrative"
    n = TM.sample({"form": "ዮሐንስ", "root": None, "type": "name"}, "x")
    assert n["target_text"] == "root: none | type: name" and analyzer.parse_neural_output(n["target_text"])["root"] is None


def test_rare_class_upsampling():
    s = [{"cls": "gerund"}] * 10 + [{"cls": "verb"}] * 100
    out = TM.rare_class_upsample(list(s), ("gerund",), 50, 20, collections.Counter())
    assert collections.Counter(x["cls"] for x in out) == {"gerund": 50, "verb": 100}


class _Fake(analyzer.HybridAnalyzer):
    def __init__(self, reply):
        self.reply, self.model_dir, self.engine, self.lemma_spec = reply, "m", None, None

    def _ensure_neural_model(self):
        return True

    def _neural_fallback(self, word):
        return self._analysis_from_reply(word, self.reply)


def test_neural_reply_keeps_rootless_nominal_possessor_and_negation():
    r = _Fake("root: none | type: pronoun | person: 3 | gender: m | number: s")._analysis_from_reply("ውእቱ", _Fake("").reply or
        "root: none | type: pronoun | person: 3 | gender: m | number: s")
    assert len(r) == 1 and r[0].pos == "pronoun" and r[0].root_consonants == ()
    g = _Fake("")._analysis_from_reply("ቀቲሎ", "root: ቀ-ተ-ለ | type: gerund | person: 3 | gender: m | number: s | possessor: 3ms | negated: yes")
    assert g[0].pos == "gerund" and g[0].possessor == "3ms" and g[0].negative
    assert _Fake("")._analysis_from_reply("x", "root: none") == []


def test_neural_nominal_goes_before_verb_only_rule_readings():
    fake = _Fake("root: ቀ-ተ-ለ | type: infinitive")
    verb = analyzer.MorphologicalAnalysis(surface_form="ቀቲል", root_consonants=("ቀ", "ተ", "ለ"), root_string="ቀ-ተ-ለ",
                                          canonical_lemma="ቀተለ", tense="past", person="3", number="s", gender="m",
                                          stem="basic", verb_type="t", pattern_spec="p", lexicon_entry=None)
    fake._rules_analyze = lambda w, l=True: [verb]
    fake._pattern_analyze = lambda *a, **k: []
    fake._loose_gutturals = False
    out = fake.analyze("ቀቲል")
    assert out[0].pos == "infinitive" and out[1] is verb


def test_extra_synthetic_off_by_default_and_check_table_rows(tmp_path):
    a = TT.parse_args(["--output_dir", str(tmp_path)])
    assert a.extra_synthetic == "" and not a.biradical_dictionary
    rows = X.rows_for("hollow", "ቀወመ")
    forms = {(r[0], r[3]): r[7] for r in rows if r[4] == "" or r[4] == "m"}
    assert X.rows_for("nominal", "ቀተለ")[0][7] == "ቀቲል"
    assert any(r[7] == "ቆመ" for r in rows) and any(r[7] == "ይቁም" for r in rows)
    out = tmp_path / "table.tsv"
    X.export_check_table(str(out))
    assert out.read_text(encoding="utf-8").count("\n") > 100


@pytest.mark.skipif(not HAS_DATA, reason="needs the dataset checkout")
def test_full_mixture_is_leak_free(tmp_path):
    args = TT.parse_args(["--dataset_dir", DATASET_DIR, "--output_dir", str(tmp_path), "--load_only",
                          "--export_dir", str(tmp_path / "pairs")])
    args.dataset_dir = os.path.abspath(args.dataset_dir)
    train, dev, report = TT.prepare(args, write=True)
    assert "CLEAN" in (tmp_path / "leak_audit.txt").read_text()
    assert report["pairs_by_source"]["extra_synthetic"] == 0
    held = TM.HeldOut(DATASET_DIR)
    assert not any(held.leaks(s["form"], s["root"]) for s in train)
    assert {s["task"] for s in train} == {D.ANALYZE_TASK, TM.GENERATE_TASK, TM.POS_TASK}
    assert {s["dev"] for s in dev} == {"gold_val", "unified_val"}
    # 2-radical dictionary roots stay out unless asked for
    assert all(len(s["root"].split("-")) != 2 for s in train if s["source"] == "dictionary" and s["root"])
