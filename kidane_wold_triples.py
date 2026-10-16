"""
Parse the verb triples that Kidane Wold gives in parentheses after a verb
headword, e.g. "(ቀተለ ፡ ይቀትል ፡ ይቅትል ፡፡ ...)" = perfect, imperfect, subjunctive,
and read the root and the verb class off the imperfect/subjunctive.

The class comes only from the written imperfect/subjunctive of the entry:
  type_a      ይቀትል + ይቅትል   (C1 1st order, then C1 6th order)
  type_b      ይቄድስ           (C1 5th order in the imperfect)
  type_c      ይባርክ           (C1 4th order, long ā kept)
  final_w     ይተሉ / ይትሉ     (ተለወ paradigm; C2 2nd order, root ends in w)
  type_b_final_w ይሄሉ         (ሀለወ: C1 5th order + final w; NOT the ተለወ paradigm)
  final_y     ይሰቲ / ይስተይ    (ሰተየ type)
  final_y_other ይበኪ / ይብኪ   (በከየ-type; final_y() must not be used)
  hollow      ይቀውም + ይቁም / ይሤይም + ይሢም
  initial_w_drop ይወልድ + ይለድ;  initial_w_keep ይወርስ + ይውርስ
  quadriliteral ይደነግፅ / ይደንግፅ
  type_d      ጽሕቀ ፡ ይጽሕቅ ፡ ይጽሐቅ (II-guttural, perfect C1 6th order; the
              subjunctive varies, so the generator copies each triple's vowels)
OCR gate: a triple is used only if all three forms show the same radicals in
the same order (w/y may be missing) and the perfect ends in a 1st-order letter.
Failing triples are written to kidane_wold_build/.
Anything else is "unclear" and gets no class. Nothing is guessed from the
perfect's spelling; the perfect is only used to confirm the radicals.

Root fixes carry the decisions recorded in the previous fixes file over:
a row there with new_root null means "root unknown, don't guess" (e.g. 7546).
The rebuild writes that entry's row with new_root null and that row's
root_notes; if the rebuild finds no row for the entry, the previous row is
kept as it is, at the entry's place in dictionary order. A previous row's
root_notes are kept on the rebuilt row for the same entry.

Running the script rewrites three files in <dataset>/data/lexicon/:
  kidane_wold_verb_triples.jsonl           accepted triples
  kidane_wold_verb_triples_rejected.jsonl  triples the OCR gate rejected, with the reason
  kidane_wold_root_fixes.jsonl             corrected roots (read first, for its decisions)
It reads classical_dictionary_kidane_wold.jsonl from the same folder.

  python kidane_wold_triples.py --dataset_dir ../geez-dataset
"""
import argparse
import json
import os
import re

import geez_script as gc

PAR = re.compile(r"\(([^()]*)\)")
SEP = re.compile(r"[\s፡።፣፤፥፦፧፨,.\-÷]+")
GUTTURALS = {"h1", "h2", "kh", "a", "a2", "khw"}
WEAK = {"w", "y"}


def syls(word):
    out = [gc.decompose_char(ch) for ch in word]
    return None if any(c is None for c, _ in out) else out


def strip_prefix(word):
    """ይ/የ + stem -> (prefix vowel, stem syllables) or None."""
    s = syls(word)
    if not s or len(s) < 3 or s[0][0] != "y" or s[0][1] not in ("I", "a"):
        return None
    return s[0][1], s[1:]


def candidate_groups(rec):
    """Parenthesized groups near the start of the entry (headword field or the
    first 40 characters of the definition), as token lists."""
    out = []
    hw = rec.get("headword_geez") or ""
    for m in PAR.finditer(hw):
        out.append(("headword", m.group(1)))
    d = rec.get("amharic_definition") or ""
    for m in PAR.finditer(d):
        if m.start() <= 40:
            out.append(("definition", m.group(1)))
    return [(src, [t for t in SEP.split(g) if t]) for src, g in out]


def find_triple(tokens, headword=None):
    """First token that is a plain Ethiopic word, followed by two ይ/የ-words.
    If the group starts with the two ይ/የ-words, the headword is the perfect."""
    if headword and len(tokens) >= 2 and strip_prefix(tokens[0]) and strip_prefix(tokens[1]):
        return headword, tokens[0], tokens[1]
    for i in range(len(tokens) - 2):
        p, a, b = tokens[i:i + 3]
        if syls(p) and 2 <= len(p) <= 5 and not p.startswith(("ይ", "የ")) \
                and strip_prefix(a) and strip_prefix(b):
            return p, a, b
    return None


def classify(p, imp, subj):
    """Return (class, root tuple, reason). Root None if the triple is unusable."""
    P = syls(p)
    pi, I = strip_prefix(imp)
    ps, S = strip_prefix(subj)
    pc = [c for c, _ in P]
    Ic = [c for c, _ in I]
    Iv = [v for _, v in I]
    Sv = [v for _, v in S]
    Sc = [c for c, _ in S]
    n = len(I)
    # I-guttural verbs: ይእንም (C1 6th order) next to የአንም (prefix a, C1 1st order)
    if n == 3 and len(S) == 3 and Ic == Sc and Ic[0] in GUTTURALS and {
            (pi, tuple(Iv)), (ps, tuple(Sv))} == {("I", ("I", "I", "I")), ("a", ("a", "I", "I"))}:
        return "type_a", tuple(Ic), "I-guttural: ይእንም + የአንም"
    if n == 3 and Iv[1:] == ["I", "I"]:
        r = tuple(Ic)
        if r[0] == "w" and len(S) == 2 and Sc == [r[1], r[2]]:
            if Sv == ["a", "I"]:
                return "initial_w_drop", r, "ይወልድ + ይለድ"
            return "initial_w_drop_other", r, "ይወ.. + subjunctive without w, vowel %s" % Sv[0]
        if Iv[0] in ("e", "A") and Sc != Ic:
            # Type B/C need the subjunctive to confirm the same radicals; OCR
            # noise such as ዜብብ ፡ ይዜብብ ፡ ይንብብል fails this
            return "unclear", r, "imperfect C1 order %s, subjunctive radicals differ" % Iv[0]
        if Iv[0] == "e":
            ok = len(S) == 3 and Sc == Ic and Sv == ["a", "I", "I"]
            return "type_b", r, "imperfect C1 5th order" + ("" if ok else " (subjunctive not ይቀድስ-shaped)")
        if Iv[0] == "A":
            return "type_c", r, "imperfect C1 4th order (long ā)" + ("" if (Sc == Ic and Sv == Iv) else " (subjunctive differs)")
        if Iv[0] == "a":
            if r[1] in WEAK and len(S) == 2 and Sc == [r[0], r[2]]:
                return "hollow", r, "ይቀውም + ይቁም"
            if len(S) == 3 and Sc == Ic and Sv == ["I", "I", "I"]:
                if r[0] == "w":
                    return "initial_w_keep", r, "ይወርስ + ይውርስ"
                return "type_a", r, "ይቀትል + ይቅትል"
            if len(S) == 3 and Sc == Ic and Sv == ["a", "I", "I"] and r[0] in GUTTURALS:
                return "type_a", r, "I-guttural: ይሐርድ + የሐርድ/ይሐርድ"
            if len(S) == 3 and Sc == Ic and Sv == ["I", "a", "I"]:
                return "type_a", r, "ይገብር + ይግበር (a-subjunctive)"
            return "unclear", r, "imperfect ይቀትል-shaped, subjunctive not matched"
        P3 = syls(p) or []
        if (Iv[0] == "I" and len(P3) == 3 and [c for c, _ in P3] == Ic
                and [v for _, v in P3] == ["I", "I", "a"] and r[1] in GUTTURALS
                and not set(r) & WEAK and len(S) == 3 and Sc == Ic and Sv[2] == "I"):
            return "type_d", r, "II-guttural: ጽሕቀ ፡ ይጽሕቅ, subjunctive vowels %s%s" % tuple(Sv[:2])
        return "unclear", r, "imperfect C1 order %s" % Iv[0]
    if n == 2:
        c1, c2 = Ic
        if Iv == ["a", "u"] and pc[-1:] == ["w"]:
            ok = Sc == Ic and Sv == ["I", "u"]
            return "final_w", (c1, c2, "w"), "ይተሉ" + (" + ይትሉ" if ok else " (subjunctive not ይትሉ-shaped)")
        if Iv == ["e", "u"] and pc[-1:] == ["w"]:
            return "type_b_final_w", (c1, c2, "w"), "ይሄሉ (ሀለወ-type)"
        if Iv == ["a", "i"] and pc[-1:] == ["y"]:
            if len(S) == 3 and Sc == [c1, c2, "y"] and Sv[:2] == ["I", "a"]:
                return "final_y", (c1, c2, "y"), "ይሰቲ + ይስተይ"
            if Sc == Ic and Sv == ["I", "i"]:
                return "final_y_other", (c1, c2, "y"), "ይበኪ + ይብኪ (በከየ-type)"
            return "unclear", (c1, c2, "y"), "ይሰቲ-shaped, subjunctive not matched"
        return "unclear", None, "two-syllable imperfect not matched"
    if n == 4 and Iv[1] == "e" and Iv[2:] == ["I", "I"]:
        return "quadriliteral", tuple(Ic), "ይደነግፅ"
    if n == 4 and Iv[1:] == ["a", "I", "I"] and Iv[0] == "a":
        return "quadriliteral", tuple(Ic), "ይደንገፅ-shaped"
    return "unclear", None, "imperfect length %d" % n


def perfect_agrees(p, root):
    """The perfect's consonants must be the root (weak letters may be missing)."""
    pc = [c for c, _ in syls(p)]
    if pc == list(root):
        return True
    strong = [c for c in root if c not in WEAK]
    return [c for c in pc if c not in WEAK] == strong


def ocr_gate(p, imp, subj):
    """None if the three forms agree, else the reason they don't."""
    P = syls(p)
    if not P or P[-1][1] != "a":
        return "perfect does not end in a 1st-order letter"
    forms = [[c for c, _ in P], [c for c, _ in strip_prefix(imp)[1]], [c for c, _ in strip_prefix(subj)[1]]]
    strong = [[c for c in f if c not in WEAK] for f in forms]
    if not (strong[0] == strong[1] == strong[2]):
        return "radicals differ: " + " / ".join("-".join(f) for f in forms)
    return None


def parse_entry(rec):
    hw = SEP.split((rec.get("headword_geez") or "").split("(")[0].strip())[0]
    hw = hw if syls(hw) and 2 <= len(hw) <= 5 and not hw.startswith(("ይ", "የ")) else None
    for src, toks in candidate_groups(rec):
        t = find_triple(toks, hw)
        if not t:
            continue
        swapped = False
        if (gc.decompose_char(t[0][0])[0] in GUTTURALS and t[1].startswith("ይ")
                and t[2].startswith("የ")):
            # Guttural-initial verbs: imperfect የኀፍር, subjunctive ይኀፍር.
            # Kidane Wold sometimes lists the ይ-form first.
            t, swapped = (t[0], t[2], t[1]), True
        bad = ocr_gate(*t)
        if bad:
            return {"id": rec["id"], "headword": rec.get("headword_geez"), "triple": list(t),
                    "source": src, "verb_class": "rejected_ocr", "root": None, "reason": bad,
                    "old_inferred_root": rec.get("inferred_root")}
        cls, root, reason = classify(*t)
        if swapped:
            reason += "; imperfect/subjunctive listed in reverse order, swapped"
        agrees = bool(root) and perfect_agrees(t[0], root)
        if root and not agrees:
            cls, reason = "unclear", reason + "; perfect does not match root"
        return {"id": rec["id"], "headword": rec.get("headword_geez"),
                "triple": list(t), "source": src, "verb_class": cls,
                "root": "-".join(root) if root and agrees else None,
                "reason": reason, "old_inferred_root": rec.get("inferred_root")}
    return None


DROPPED = {"a2", "S2", "gw", "khw"}   # ዐ ፀ ጐ ኈ: unknown to the old root inference


def restore_dropped(rec):
    """The old inference skipped ዐ/ፀ/ጐ/ኈ (unknown letters), so a root like ድምፀ
    came out d-m. If the old root is the headword's consonants with those letters
    removed, put them back where they stand. Only used when that turns a
    two-letter root into a three-letter one; anything else stays as it was."""
    old = rec.get("inferred_root")
    if not old:
        return None
    R = old.split(" (")[0].split("-")
    hw = SEP.split((rec.get("headword_geez") or "").split("(")[0].strip())[0]
    H = [gc.decompose_char(ch)[0] for ch in hw]
    if len(R) != 2 or not H or None in H or not any(c in DROPPED for c in H):
        return None
    keep = [i for i, c in enumerate(H) if c not in DROPPED]
    Hk = [H[i] for i in keep]
    for k in range(len(Hk) - len(R) + 1):
        if Hk[k:k + len(R)] == R:
            a, b = keep[k], keep[k + len(R) - 1]
            while a > 0 and H[a - 1] in DROPPED:
                a -= 1
            while b + 1 < len(H) and H[b + 1] in DROPPED:
                b += 1
            new = H[a:b + 1]
            return "-".join(new) if len(new) == 3 else None
    return None


def read_fix_decisions(path):
    """{id: previous fix row} for the rows of a fixes file that carry a decision: new_root null
    ("root unknown, don't guess") or root_notes."""
    out = {}
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                x = json.loads(line)
                if x.get("new_root") is None or x.get("root_notes"):
                    out[x["id"]] = x
    return out


def apply_fix_decision(row, decisions):
    """The rebuilt fix row with a previous decision for the same entry applied."""
    prev = decisions.get(row["id"])
    if prev is None:
        return row
    if prev.get("new_root") is None:
        row["new_root"] = None
    if prev.get("root_notes"):
        row["root_notes"] = prev["root_notes"]
    return row


def build(dictionary_path, triples_path, fixes_path, previous_fixes_path=None):
    """Write every accepted triple, log the rejected ones, and write the root
    fixes the generator/analyzer use. previous_fixes_path: the fixes file before
    this rebuild, read for its decisions (see the module docstring)."""
    n_verbs = n_trip = 0
    fixes = []
    decisions = read_fix_decisions(previous_fixes_path)
    seen_ids = set()
    rejected_path = triples_path.replace(".jsonl", "_rejected.jsonl")
    with open(dictionary_path, encoding="utf-8") as f, \
            open(triples_path, "w", encoding="utf-8") as out, \
            open(rejected_path, "w", encoding="utf-8") as rej:
        for line in f:
            rec = json.loads(line)
            t = parse_entry(rec) if rec.get("pos") == "verb" else None
            n_verbs += rec.get("pos") == "verb"
            if t and t["verb_class"] == "rejected_ocr":
                rej.write(json.dumps(t, ensure_ascii=False) + "\n")
                t = None
            if t:
                n_trip += 1
                out.write(json.dumps(t, ensure_ascii=False) + "\n")
            old = rec.get("inferred_root")
            row = None
            if t and t["root"]:
                if t["root"] != old:
                    row = {"id": rec["id"], "headword": rec.get("headword_geez"), "pos": rec.get("pos"),
                           "old_root": old, "new_root": t["root"], "method": "verb_triple",
                           "evidence": " ፡ ".join(t["triple"])}
            else:
                new = restore_dropped(rec)
                if new:
                    row = {"id": rec["id"], "headword": rec.get("headword_geez"), "pos": rec.get("pos"),
                           "old_root": old, "new_root": new, "method": "restored_dropped_letter",
                           "evidence": "headword"}
            if row is None and rec.get("id") in decisions and decisions[rec["id"]].get("new_root") is None:
                row = dict(decisions[rec["id"]])      # "root unknown" decision, no rebuilt row: keep the previous row
            if row is not None and row["id"] not in seen_ids:
                seen_ids.add(row["id"])
                fixes.append(apply_fix_decision(row, decisions))
    # decisions for ids that are not in the dictionary: kept at the end, in their previous order
    for i, prev in decisions.items():
        if i not in seen_ids and prev.get("new_root") is None:
            seen_ids.add(i)
            fixes.append(dict(prev))
    with open(fixes_path, "w", encoding="utf-8") as out:
        for x in fixes:
            out.write(json.dumps(x, ensure_ascii=False) + "\n")
    return n_verbs, n_trip, fixes


def parse_args(argv=None):
    import dataset_loader
    p = argparse.ArgumentParser(
        description="Rebuild the Kidane Wold verb triples and root fixes. Rewrites three files in "
                    "<dataset>/data/lexicon/: kidane_wold_verb_triples.jsonl, "
                    "kidane_wold_verb_triples_rejected.jsonl and kidane_wold_root_fixes.jsonl.")
    p.add_argument("dataset", nargs="?", default=None,
                   help="Checkout of the geez-dataset repo (same as --dataset_dir)")
    p.add_argument("--dataset_dir", default=None,
                   help="Checkout of the geez-dataset repo (default: $GEEZ_DATASET_DIR, else ../geez-dataset)")
    args = p.parse_args(argv)
    if args.dataset and args.dataset_dir and args.dataset != args.dataset_dir:
        p.error("give the dataset folder once, as a positional argument or with --dataset_dir")
    args.dataset_dir = args.dataset_dir or args.dataset or dataset_loader.default_dataset_dir()
    return args


if __name__ == "__main__":
    ds = parse_args().dataset_dir
    lex = os.path.join(ds, "data", "lexicon")
    # The dataset's dictionary already carries the corrected roots (inferred_root), so a rebuild writes into a
    # separate folder: the triples to compare with lexicon/kidane_wold_verb_triples.jsonl, the rejected triples
    # and any further root fixes. Nothing in the dataset folder is overwritten.
    out_dir = "kidane_wold_build"
    os.makedirs(out_dir, exist_ok=True)
    fixes_file = os.path.join(out_dir, "kidane_wold_root_fixes.jsonl")
    n_verbs, n_trip, fixes = build(os.path.join(lex, "classical_dictionary_kidane_wold.jsonl"),
                                   os.path.join(out_dir, "kidane_wold_verb_triples.jsonl"),
                                   fixes_file, previous_fixes_path=fixes_file)
    from collections import Counter
    print(f"verb entries: {n_verbs}; accepted triples: {n_trip}")
    with open(os.path.join(out_dir, "kidane_wold_verb_triples_rejected.jsonl"), encoding="utf-8") as f:
        rej = [json.loads(l) for l in f]
    print(f"triples rejected by the OCR gate: {len(rej)}",
          dict(Counter(r["reason"].split(":")[0] for r in rej)))
    with open(os.path.join(out_dir, "kidane_wold_verb_triples.jsonl"), encoding="utf-8") as f:
        print("accepted triples by class:", dict(Counter(json.loads(l)["verb_class"] for l in f)))
    print("root fixes:", dict(Counter(x["method"] for x in fixes)))
