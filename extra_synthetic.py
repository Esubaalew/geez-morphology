"""Extra synthetic families: derived stems of Type C and Type D, hollow (II-w / II-y), defective (derived stems of
final-y / final-w), initial-w causatives, and the nominal / non-finite templates (ቀቲል, ቀቲሎ, ቀታሊ, ቅቱል, ቅትለት).

These templates are NOT used in training unless named in --extra_synthetic (default: none). They wait for a check
of every template by a Ge'ez reader; export_check_table() writes one fully generated root per family for that check
(python3 extra_synthetic.py --table OUT.tsv).

Each family builder takes a root as a list of Ethiopic letters (any order) and returns rows
(template, stem, tense, person, gender, number, type, form, note). A note marks a cell whose spelling rule is
uncertain (guttural vowel change, hollow contraction, final-y / final-w endings).
"""
import argparse
import csv
import sys

from verb_class_paradigms import (CAUSATIVE_PREFIX, END_ORDER, GUTTURAL_LETTERS, IMP_END, IMP_PERSONS, PASSIVE_PREFIX,
                                  PERSONS, PREFIX, final_w, final_y, strong_perfect, syl, type_c, type_d)

TENSE = {"perfect": "past", "imperfect": "present", "subjunctive": "subjunctive", "imperative": "imperative"}


def split_person(p):
    return p[0], (p[1] if len(p) == 3 else ""), p[-1]


def gutt(ch):
    return syl(ch, 1) in GUTTURAL_LETTERS


def table_rows(family, stem, tables, note=""):
    rows = []
    for table, cells in tables.items():
        for p in (IMP_PERSONS if table == "imperative" else PERSONS):
            if p in cells:
                person, gender, number = split_person(p)
                rows.append((f"{family}:{stem}:{table}", stem, TENSE[table], person, gender, number, "verb", cells[p], note))
    return rows


def last(c, end):
    return syl(c, END_ORDER[end]) if end else syl(c, 6)


# ----------------------------------------------------------------------------- verb families

def derived_c(r):
    """Type C (ባረከ): passive ተባረከ / ይትባረክ / ይትባረክ / ተባረክ; causative አባረከ / ያባርክ / ያባርክ / አባርክ."""
    base = type_c(r)
    passive = {"perfect": {p: "ተ" + f for p, f in base["perfect"].items()},
               "imperfect": {p: PASSIVE_PREFIX[pre] + syl(r[0], 4) + syl(r[1], 1) + last(r[2], e) for p, (pre, e) in PREFIX.items()},
               "imperative": {p: "ተ" + syl(r[0], 4) + syl(r[1], 1) + last(r[2], e) for p, e in IMP_END.items()}}
    passive["subjunctive"] = dict(passive["imperfect"])
    causative = {"perfect": {p: "አ" + f for p, f in base["perfect"].items()},
                 "imperfect": {p: CAUSATIVE_PREFIX[pre] + syl(r[0], 4) + syl(r[1], 6) + last(r[2], e) for p, (pre, e) in PREFIX.items()}}
    causative["subjunctive"] = dict(causative["imperfect"])
    note = "final guttural: check" if gutt(r[2]) else ""
    return table_rows("derived_c", "passive", passive, note) + table_rows("derived_c", "causative", causative, note)


def derived_d(r, orders=None):
    """Type D (ጽሕቀ, C1 6th order in the perfect): passive ተ + the perfect, ይት + the subjunctive stem."""
    orders = orders or {"perfect": (6, 6), "imperfect": (6, 6), "subjunctive": (6, 1)}
    base = type_d(r, orders)
    so = orders["subjunctive"]
    passive = {"perfect": {p: "ተ" + f for p, f in base["perfect"].items()},
               "imperfect": {p: PASSIVE_PREFIX[pre] + syl(r[0], so[0]) + syl(r[1], so[1]) + last(r[2], e)
                             for p, (pre, e) in PREFIX.items()}}
    return table_rows("derived_d", "passive", passive, "guttural radical 2: vowel of ተ-stem to check")


def hollow(r):
    """II-w (ቆመ q-w-m) and II-y (ሤመ s-y-m): the middle radical contracts.
    II-w: ቆመ ቆመት ቆምከ ...; ይቀውም; ይቁም; ቁም.  II-y: ሤመ; ይሠይም; ይሢም; ሢም."""
    c1, w, c3 = r
    mid_w = syl(w, 1) == "ወ"
    v_perf, v_subj = (7, 2) if mid_w else (5, 3)
    perf = {"3ms": syl(c1, v_perf) + syl(c3, 1), "3fs": syl(c1, v_perf) + syl(c3, 1) + "ት",
            "3mp": syl(c1, v_perf) + syl(c3, 2), "3fp": syl(c1, v_perf) + syl(c3, 4)}
    for p, s in {"2ms": "ከ", "2fs": "ኪ", "1s": "ኩ", "2mp": "ክሙ", "2fp": "ክን", "1p": "ነ"}.items():
        perf[p] = syl(c1, v_perf) + syl(c3, 6) + s
    glide = "ው" if mid_w else "ይ"
    imperf = {p: pre + syl(c1, 1) + glide + last(c3, e) for p, (pre, e) in PREFIX.items()}
    subj = {p: pre + syl(c1, v_subj) + last(c3, e) for p, (pre, e) in PREFIX.items()}
    impv = {p: syl(c1, v_subj) + last(c3, e) for p, e in IMP_END.items()}
    note = "hollow contraction" + ("; final guttural: check" if gutt(c3) else "")
    return table_rows("hollow", "basic", {"perfect": perf, "imperfect": imperf, "subjunctive": subj, "imperative": impv}, note)


def defective(r):
    """Passive (ተ-) and causative (አ-) of final-y (ሰትየ) and final-w (ተለወ) roots, from their basic tables."""
    base = final_y(r) if syl(r[2], 1) == "የ" else final_w(r)
    passive = {"perfect": {p: "ተ" + f for p, f in base["perfect"].items()},
               "imperfect": {p: PASSIVE_PREFIX[f[0]] + f[1:] for p, f in base["subjunctive"].items()}}
    causative = {"perfect": {p: "አ" + syl(f[0], 6) + f[1:] for p, f in base["perfect"].items()},
                 "imperfect": {p: CAUSATIVE_PREFIX[f[0]] + f[1:] for p, f in base["subjunctive"].items()}}
    note = "final-y/w ending in a derived stem: check"
    return table_rows("defective", "passive", passive, note) + table_rows("defective", "causative", causative, note)


def initial_w(r):
    """Causative of I-w roots (ወለደ): አውለደ, ያወልድ, ያውልድ."""
    perf = {p: "አ" + syl(r[0], 6) + f[1:] for p, f in strong_perfect(r, 1).items()}
    imperf = {p: CAUSATIVE_PREFIX[pre] + syl(r[0], 1) + syl(r[1], 6) + last(r[2], e) for p, (pre, e) in PREFIX.items()}
    subj = {p: CAUSATIVE_PREFIX[pre] + syl(r[0], 6) + syl(r[1], 6) + last(r[2], e) for p, (pre, e) in PREFIX.items()}
    return table_rows("initial_w", "causative", {"perfect": perf, "imperfect": imperf, "subjunctive": subj},
                      "initial-w kept in the causative: check")


# ----------------------------------------------------------------------------- nominal / non-finite templates

GERUND_SUFFIX = {"3ms": "ኦ", "3fs": "ኣ", "2ms": "ከ", "2fs": "ኪ", "1s": "የ", "3mp": "ኦሙ", "3fp": "ኦን",
                 "2mp": "ክሙ", "2fp": "ክን", "1p": "ነ"}


def attach(stem_last_cons, suffix):
    """The last radical with a vowel suffix: ኦ -> 7th order, ኣ -> 4th; a consonant suffix after the 1st order."""
    if suffix[0] == "ኦ":
        return syl(stem_last_cons, 7) + suffix[1:]
    if suffix[0] == "ኣ":
        return syl(stem_last_cons, 4) + suffix[1:]
    if suffix == "የ":
        return syl(stem_last_cons, 6) + "የ"
    return syl(stem_last_cons, 1) + suffix


def nominal(r):
    c1, c2, c3 = r
    note = []
    if gutt(c3):
        note.append("final guttural: vowel before it to check")
    if syl(c3, 1) in ("የ", "ወ"):
        note.append("final-y/w ending to check")
    note = "; ".join(note)
    rows = [("nominal:infinitive:ቀቲል", "", "", "", "", "", "infinitive", syl(c1, 1) + syl(c2, 3) + syl(c3, 6), note),
            ("nominal:agent:ቀታሊ", "", "", "", "", "s", "noun", syl(c1, 1) + syl(c2, 4) + syl(c3, 3), note),
            ("nominal:passive_participle:ቅቱል", "", "", "", "m", "s", "adjective", syl(c1, 6) + syl(c2, 2) + syl(c3, 6), note),
            ("nominal:verbal_noun:ቅትለት", "", "", "", "", "s", "verbal_noun", syl(c1, 6) + syl(c2, 6) + syl(c3, 1) + "ት", note)]
    for p in PERSONS:
        person, gender, number = split_person(p)
        rows.append((f"nominal:gerund:ቀቲሎ:{p}", "", "", person, gender, number, "gerund",
                     syl(c1, 1) + syl(c2, 3) + attach(c3, GERUND_SUFFIX[p]), note))
    return rows


FAMILIES = {"derived_c": derived_c, "derived_d": derived_d, "hollow": hollow, "defective": defective,
            "initial_w": initial_w, "nominal": nominal}
# One worked root per family (several for the families whose spelling turns on the root's letters).
WORKED = [("derived_c", "ባረከ"), ("derived_c", "ቀረሐ"), ("derived_d", "ጸሐቀ"), ("hollow", "ቀወመ"), ("hollow", "ሠየመ"),
          ("defective", "ሰተየ"), ("defective", "ተለወ"), ("initial_w", "ወለደ"),
          ("nominal", "ቀተለ"), ("nominal", "ሰመዐ"), ("nominal", "ሰተየ"), ("nominal", "ተለወ")]
TAG_FAMILY = {"type_c": "derived_c", "type_d": "derived_d", "hollow": "hollow", "final_y": "defective",
              "final_w": "defective", "type_a": "nominal"}


def rows_for(family, root_letters):
    return FAMILIES[family](list(root_letters))


def generate_pairs(families, dataset_dir):
    """Pairs (dataset_loader pair shape) for the roots the class tags give each named family; the trainer's leak
    gate runs on them afterwards."""
    import geez_script
    import verb_classes
    unknown = set(families) - set(FAMILIES)
    if unknown:
        raise ValueError(f"unknown --extra_synthetic families: {sorted(unknown)}")
    tags, _ = verb_classes.class_tags(dataset_dir)
    out = []
    for cons, tag in sorted(tags.items()):
        fam = TAG_FAMILY.get(tag)
        if len(cons) != 3:
            continue
        letters = geez_script.to_ethiopic_root("-".join(cons)).split("-")
        targets = [f for f in families if f == fam or (f == "nominal" and fam)]
        if "initial_w" in families and cons[0] == "w":
            targets.append("initial_w")
        for f in targets:
            root = "-".join(syl(c, 1) for c in letters)
            for t, stem, tense, person, gender, number, typ, form, _ in rows_for(f, letters):
                out.append({"form": form, "root": root, "type": typ, "tense": tense or None, "person": person,
                            "gender": gender, "number": number, "stem": stem or None, "object": ""})
    return out


def export_check_table(path):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["family", "worked_root", "template", "stem", "tense", "person", "gender", "number", "type",
                    "generated_form", "note", "verdict"])
        for fam, lemma in WORKED:
            for row in rows_for(fam, lemma):
                w.writerow([fam, "-".join(syl(c, 1) for c in lemma), *row, ""])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", required=True, help="TSV to write")
    export_check_table(ap.parse_args().table)
    sys.exit(0)


CHECKED_FILE = "data/extra_synthetic_checked.tsv"


def checked_pairs(path=None):
    """Pairs for the generated cells a Ge'ez reader checked and passed (CHECKED_FILE), and only those: each row
    names the template cell (labels: template, person-gender-number, root) and the form to train, which is the
    corrected spelling where the template's own output was wrong. Each row is matched to exactly one cell of
    rows_for(family, root); a row that matches none or several raises ValueError."""
    import os
    path = path or os.path.join(os.path.dirname(os.path.abspath(__file__)), CHECKED_FILE)
    out = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            template, pgn, _, root = r["labels"].split(" ")
            family = template.split(":")[0]
            cells = [c for c in rows_for(family, root.split("-"))
                     if template.startswith(c[0]) and c[7] == r["template_form"] and pgn.endswith(c[3] + c[4] + c[5])]
            if len(cells) != 1:
                raise ValueError(f"{r['labels']} {r['template_form']}: {len(cells)} matching cells")
            _, stem, tense, person, gender, number, typ, _, _ = cells[0]
            out.append({"form": r["form"], "root": root, "type": typ, "tense": tense or None, "person": person,
                        "gender": gender, "number": number, "stem": stem or None, "object": ""})
    return out
