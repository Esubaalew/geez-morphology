"""
verb_classes.py
Connects the checked class paradigms in verb_class_paradigms.py (Ge'ez
the grammar check) to the generator and the analyzer.

  class_tags()      which roots are Type B, Type C, or ሰትየ-type final-y
  class_fits_root() whether a class's template can take a root at all
  paradigm_rows()   (features, surface) pairs for one root of a checked class
  class_patterns()  the same paradigms as pattern specs, for the analyzer

Verb class cannot be read off the spelling: a Type B perfect (ቀደሰ) is written
exactly like Type A (ቀተለ). So class tags come from lexical sources only:
  1. the class lemmas themselves (ቀደሰ, ባረከ, ሰትየ, ተለወ)
  2. the paradigm_class field of the Merho Ges TRAIN split, which files each
     verb under its class lemma. Records that name more than one class are skipped.
  3. the verb triples Kidane Wold prints after a verb headword
     (ቀተለ ፡ ይቀትል ፡ ይቅትል), parsed by kw_verb_triples.py into
     data/lexicon/kidane_wold_verb_triples.jsonl. The class is read off the
     written imperfect/subjunctive. Where a triple gives a class it wins over
     1 and 2; where it is missing or unclear, 1 and 2 are used.
Final-w roots get the ተለወ paradigm only with a tag (a ይተሉ/ይትሉ imperfect in the
dictionary, a ተለወ class in Merho Ges, or the lemma): ሀለወ (ይሄሉ) is Type B
with a final w and does not follow ተለወ.
"""

import json
import os
from typing import Dict, List, Tuple

import geez_config
import verb_class_paradigms as vcp

TENSE_OF = {"perfect": "past", "imperfect": "present",
            "subjunctive": "subjunctive", "imperative": "imperative"}
PERSON_OF = {
    "3ms": ("3", "m", "s"), "3fs": ("3", "f", "s"), "2ms": ("2", "m", "s"),
    "2fs": ("2", "f", "s"), "1s": ("1", "c", "s"), "3mp": ("3", "m", "p"),
    "3fp": ("3", "f", "p"), "2mp": ("2", "m", "p"), "2fp": ("2", "f", "p"),
    "1p": ("1", "c", "p"),
}
VERB_TYPE_OF = {"type_b": "strong_B", "type_c": "strong_C", "final_y": "weak_final",
                "final_w": "weak_final", "initial_w_drop": "weak_initial",
                "type_c_o": "strong_C_o", "type_c_e": "strong_C_e", "quad_c": "quadriliteral_C",
                "type_d": "strong_D"}
VOWEL_ORDER = {"a": 1, "u": 2, "i": 3, "A": 4, "e": 5, "I": 6, "o": 7}
GUTTURALS = ("a", "a2", "h1", "h2", "kh")
# Classes read off the written triple by shape_classes() (the grammar check of the
# the written shape types): ጦመረ ፡ ይጦምር ፡ ይጦምር, ē kept on radical 1, ማህረከ ፡ ይማህርክ ፡ ይማህርክ.
SHAPE_CLASSES = ("type_c_o", "type_c_e", "quad_c")

CLASS_LEMMAS = {"ቀደሰ": "type_b", "ባረከ": "type_c", "ሰትየ": "final_y", "ተለወ": "final_w"}
# Merho Ges class names -> checked class. ሰተየ and ሰትየ are the same class
# (confirmed by the grammar check, 2026-10-05).
MERHO_CLASS_NAMES = {"ቀደሰ": "type_b", "ባረከ": "type_c", "ሰትየ": "final_y", "ሰተየ": "final_y",
                     "ተለወ": "final_w"}
# Classes a Kidane Wold triple can show. The first group selects a checked
# paradigm; the second says the root does NOT take one (type_a and hollow use
# the ordinary templates; the others have no checked paradigm yet and are skipped).
TRIPLE_PARADIGM_CLASSES = ("type_b", "type_c", "final_w", "final_y", "initial_w_drop", "type_d")
TRIPLE_OTHER_CLASSES = ("type_a", "hollow", "type_b_final_w", "final_y_other",
                        "initial_w_drop_other", "quadriliteral")
TRIPLES_FILE = os.path.join("data", "lexicon", "kidane_wold_verb_triples.jsonl")
ROOT_FIXES_FILE = os.path.join("data", "lexicon", "kidane_wold_root_fixes.jsonl")
GOLD_TRAIN_FILE = os.path.join("data", "paradigms", "merho_ges_gold_train.jsonl")
MAIN_CLASS_NAMES = ("ቀተለ", "ቀደሰ", "ባረከ")
WEAK = ("w", "y")
# Amharic-only letters: ሸ ቸ ኘ ዠ have consonant symbols
# in geez_config; ኸ ጀ ጨ have none, so roots with them never decompose.
AMHARIC_ONLY = ("sh", "ch", "ny", "zh")
AMHARIC_ONLY_LETTERS = "ሸቸኘኸዠጀጨ"


def has_amharic_letter(cons) -> bool:
    return any(c in AMHARIC_ONLY for c in cons)


def weak_count(cons) -> int:
    """Number of w/y radicals; two or more = doubly weak (ሐይወ, የ-ወ-ሰ)."""
    return sum(1 for c in cons if c in WEAK)


def root_of_lemma(lemma: str) -> Tuple[str, ...]:
    return tuple(geez_config.decompose_char(ch)[0] for ch in lemma)


def class_fits_root(cls: str, cons: Tuple[str, ...]) -> bool:
    """The checked templates assume strong radicals except where the class says otherwise."""
    if cls == "type_d":
        return (len(cons) == 3 and cons[1] in GUTTURALS and not set(cons) & set(WEAK)
                and all(c in geez_config.CONSONANTS for c in cons))
    if cls == "quad_c":
        return len(cons) == 4 and all(c in geez_config.CONSONANTS and c not in WEAK for c in cons)
    if len(cons) != 3 or not all(c in geez_config.CONSONANTS for c in cons):
        return False
    weak = [c in WEAK for c in cons]
    if cls in ("type_b", "type_c", "type_c_o", "type_c_e"):
        return not any(weak)
    if cls == "final_y":
        return cons[2] == "y" and not weak[0] and not weak[1]
    if cls == "final_w":
        return cons[2] == "w" and not weak[0] and not weak[1]
    if cls == "initial_w_drop":
        return "-".join(cons) in vcp.W_DROP
    return False


def w_drop_shape(cons: Tuple[str, ...]) -> bool:
    """Shape the ወለደ paradigm can take: w + two strong radicals."""
    return len(cons) == 3 and cons[0] == "w" and cons[1] not in WEAK and cons[2] not in WEAK


def takes_w_drop(cons: Tuple[str, ...], tags: Dict[Tuple[str, ...], str]) -> bool:
    """W_DROP list, or a Kidane Wold triple with a ይለድ-shaped subjunctive."""
    return class_fits_root("initial_w_drop", cons) or (
        tags.get(tuple(cons)) == "initial_w_drop" and w_drop_shape(cons))


def root_fixes(dataset_dir: str) -> Dict[int, str]:
    """{dictionary entry id: corrected root} from kw_verb_triples.py. A row with new_root null stays in the
    dict with the value None: the entry's root is unknown (see entry_root)."""
    path = os.path.join(dataset_dir, ROOT_FIXES_FILE)
    fixes = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                x = json.loads(line)
                fixes[x["id"]] = x["new_root"]
    return fixes


# Values that are not roots: an unsettled root is null, but older files wrote placeholders.
ROOT_PLACEHOLDERS = {"check", "null", "none"}


def usable_root(value):
    """value if it is a root string; None for null, an empty string or a placeholder (CHECK, "null", "None")."""
    if not isinstance(value, str) or not value.strip() or value.strip().lower() in ROOT_PLACEHOLDERS:
        return None
    return value


def entry_root(entry: dict, fixes: Dict[int, str]):
    """Root of a dictionary entry, with the triple/restoration fix applied.

    An entry with a fix row takes the fix's new_root and nothing else: a fix row whose new_root is null means
    "root unknown, don't guess", so the result is None even when inferred_root or root has a
    value. Without a fix row: inferred_root, else root. A null, empty or placeholder value (CHECK, "null",
    "None") is never returned."""
    eid = entry.get("id")
    if eid in fixes:
        return usable_root(fixes[eid])
    for value in (entry.get("inferred_root"), entry.get("root")):
        if usable_root(value):
            return value
    return None


def triple_classes(dataset_dir: str) -> Tuple[Dict[Tuple[str, ...], str], int]:
    """{root: class} from the Kidane Wold triples. A root whose triples disagree
    is left out (returned count)."""
    path = os.path.join(dataset_dir, TRIPLES_FILE)
    seen: Dict[Tuple[str, ...], set] = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                x = json.loads(line)
                cls = x.get("verb_class")
                if not x.get("root") or cls not in TRIPLE_PARADIGM_CLASSES + TRIPLE_OTHER_CLASSES:
                    continue
                seen.setdefault(tuple(x["root"].split("-")), set()).add(cls)
    out = {r: next(iter(c)) for r, c in seen.items() if len(c) == 1}
    return out, sum(1 for c in seen.values() if len(c) > 1)


def _units(word):
    """[(consonant, vowel)] for an Ethiopic word, or None if a letter is unknown."""
    units = [geez_config.decompose_char(ch) for ch in word or ""]
    return None if not units or None in units else units


def _stem(word):
    """Imperfect/subjunctive without its 3ms prefix ይ (OCR often gives የ)."""
    return _units(word[1:]) if word[:1] in ("ይ", "የ") else None


def shape_classes(dataset_dir: str) -> Dict[Tuple[str, ...], str]:
    """{root: class} for triples written in a Type C shape that kw_verb_triples.py
    does not name. Imperfect and subjunctive must be written alike (Type C), with the
    same radicals as the perfect:
      type_c_o  ጦመረ ፡ ይጦምር ፡ ይጦምር   ō on radical 1 in all three
      type_c_e  ē on radical 1 in all three (perfect not 1st order, so not Type B)
      quad_c    ማህረከ ፡ ይማህርክ ፡ ይማህርክ  4 radicals, ā on radical 1
    A root whose triples disagree is left out."""
    path = os.path.join(dataset_dir, TRIPLES_FILE)
    seen: Dict[Tuple[str, ...], set] = {}
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            p, i, s = json.loads(line)["triple"]
            P, I, S = _units(p), _stem(i), _stem(s)
            if not (P and I and S) or I != S or [c for c, _ in P] != [c for c, _ in I]:
                continue
            cons = tuple(c for c, _ in P)
            cls = None
            if (len(P) == 3 and P[0][1] == I[0][1] and P[0][1] in ("o", "e")
                    and P[1][1] == "a" and I[1][1] == "I"):
                cls = "type_c_" + P[0][1]
            elif (len(P) == 4 and P[0][1] == "A" and I[0][1] == "A" and P[1][1] == "I"
                    and I[1][1] == "I" and I[2][1] == "I"):
                cls = "quad_c"
            if cls and class_fits_root(cls, cons):
                seen.setdefault(cons, set()).add(cls)
    return {r: next(iter(c)) for r, c in seen.items() if len(c) == 1}


# Order of the 2nd radical in the subjunctive/jussive and imperative when it is a
# guttural (ፍሬ ግእዝ pp. 18, 63): ይስአል, ይምሐል, ይድሐን (1st order); ይስዓር is printed with ዓ.
BOOK_SUBJ_ORDER_2ND = {"a": 1, "h1": 1, "h2": 1, "kh": 1, "a2": 4}


def book_subjunctive(root: Tuple[str, ...], orders: dict) -> dict:
    """Apply the ፍሬ ግእዝ subjunctive rule to one type_d orders dict (perfect and
    imperfect unchanged)."""
    if len(root) == 3 and root[1] in BOOK_SUBJ_ORDER_2ND:
        orders = dict(orders)
        orders["subjunctive"] = (orders["subjunctive"][0], BOOK_SUBJ_ORDER_2ND[root[1]])
    return orders


def triple_roots(dataset_dir: str) -> set:
    """Roots with at least one accepted Kidane Wold triple (any class)."""
    path = os.path.join(dataset_dir, TRIPLES_FILE)
    out = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                x = json.loads(line)
                if x.get("root"):
                    out.add(tuple(x["root"].split("-")))
    return out


def kw_collapsed_subjunctive(dataset_dir: str) -> Dict[Tuple[str, ...], dict]:
    """{root: info} for 3-radical roots whose Kidane Wold triple writes the subjunctive
    exactly like the imperfect with the guttural middle radical in the 6th order
    (ጸሕፈ ፡ ይጽሕፍ ፡ ይጽሕፍ): an OCR collapse. info has the
    triple ids, classes and the imperfect orders of radicals 1 and 2. A root whose
    triples disagree on the imperfect is left out."""
    path = os.path.join(dataset_dir, TRIPLES_FILE)
    seen: Dict[Tuple[str, ...], dict] = {}
    bad = set()
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            x = json.loads(line)
            if not x.get("root"):
                continue
            root = tuple(x["root"].split("-"))
            if len(root) != 3 or root[1] not in GUTTURALS:
                continue
            I, S = _stem(x["triple"][1]), _stem(x["triple"][2])
            if not I or not S or I != S:
                continue
            mid = [v for c, v in S if c == root[1]]
            if not mid or mid[0] != "I":
                continue
            orders = (I[0][1], mid[0])
            info = seen.setdefault(root, {"ids": [], "classes": [], "triples": [], "imperfect": orders})
            if info["imperfect"] != orders:
                bad.add(root)
            info["ids"].append(x["id"])
            info["classes"].append(x.get("verb_class"))
            info["triples"].append(" ፡ ".join(x["triple"]))
    return {r: v for r, v in seen.items() if r not in bad}


def type_d_orders(dataset_dir: str) -> Dict[Tuple[str, ...], dict]:
    """{root: {"perfect"|"imperfect"|"subjunctive": (order of radical 1, order of
    radical 2)}} from the type_d triples. Roots whose triples disagree are left out."""
    path = os.path.join(dataset_dir, TRIPLES_FILE)
    seen: Dict[Tuple[str, ...], list] = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                x = json.loads(line)
                if x.get("verb_class") != "type_d" or not x.get("root"):
                    continue
                p, i, s = x["triple"]
                P, I, S = _units(p), _stem(i), _stem(s)
                o = {name: (VOWEL_ORDER[u[0][1]], VOWEL_ORDER[u[1][1]])
                     for name, u in (("perfect", P), ("imperfect", I), ("subjunctive", S))}
                seen.setdefault(tuple(x["root"].split("-")), []).append(o)
    return {r: book_subjunctive(r, v[0]) for r, v in seen.items() if all(o == v[0] for o in v)}


def type_a_subjunctive_shapes(dataset_dir: str) -> Tuple[Dict[Tuple[str, ...], tuple], int]:
    """{root: (vowel of radical 1, vowel of radical 2)} in the subjunctive of a
    type_a triple: ("I", "I") ይቅትል, ("I", "a") ይግበር / ይስማዕ, ("a", "I") ይአድም
    (guttural first radical). Roots whose triples disagree are left out (count)."""
    path = os.path.join(dataset_dir, TRIPLES_FILE)
    seen: Dict[Tuple[str, ...], set] = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                x = json.loads(line)
                if x.get("verb_class") != "type_a" or not x.get("root"):
                    continue
                S = _stem(x["triple"][2])
                if S and len(S) == 3 and S[0][1] in ("I", "a") and S[1][1] in ("I", "a"):
                    seen.setdefault(tuple(x["root"].split("-")), set()).add((S[0][1], S[1][1]))
    out = {r: next(iter(v)) for r, v in seen.items() if len(v) == 1}
    return out, sum(1 for v in seen.values() if len(v) > 1)


def class_tags(dataset_dir: str, exclude=()) -> Tuple[Dict[Tuple[str, ...], str], Dict[str, int]]:
    """Return ({root: class}, counts). Roots in `exclude` (held-out test roots) are dropped."""
    tags, _, counts = _tags_and_sources(dataset_dir, exclude)
    return tags, counts


def class_sources(dataset_dir: str, exclude=()) -> Dict[Tuple[str, ...], str]:
    """{root: where its class tag came from}, for the roots class_tags() returns:
    "class lemma", "merho_ges train" or "kidane_wold triple"."""
    tags, source, _ = _tags_and_sources(dataset_dir, exclude)
    return {cons: source.get(cons, "?") for cons in tags}


def _tags_and_sources(dataset_dir: str, exclude=()):
    tags: Dict[Tuple[str, ...], str] = {}
    source: Dict[Tuple[str, ...], str] = {}
    for lemma, cls in CLASS_LEMMAS.items():
        tags[root_of_lemma(lemma)] = cls
        source[root_of_lemma(lemma)] = "class lemma"
    conflicts = set()
    # Merho Ges class names from the gold train file as written (train records only; the maintainer 2026-10-06
    # 15:14, ).
    path = os.path.join(dataset_dir, GOLD_TRAIN_FILE)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("category") != "verb_conjugation":
                    continue
                pc = rec.get("paradigm_class") or ""
                named = {cls for name, cls in MERHO_CLASS_NAMES.items() if name in pc}
                mains = [n for n in MAIN_CLASS_NAMES if n in pc]
                if len(named) != 1 or len(mains) > 1:
                    continue
                cls = named.pop()
                cons = geez_config.root_symbols((rec.get("root") or "").split("/")[0])
                if not class_fits_root(cls, cons):
                    continue
                if tags.get(cons, cls) != cls:
                    conflicts.add(cons)
                tags[cons] = cls
                source.setdefault(cons, "merho_ges train")
    for c in conflicts:
        tags.pop(c, None)
    # Kidane Wold triples: the written imperfect decides where it is clear.
    triples, triple_conflicts = triple_classes(dataset_dir)
    overrides = 0
    for cons, cls in triples.items():
        fits = w_drop_shape(cons) if cls == "initial_w_drop" else class_fits_root(cls, cons)
        if cls in TRIPLE_PARADIGM_CLASSES and not fits:
            continue
        if cons in tags and tags[cons] != cls:
            overrides += 1
        tags[cons] = cls
        source[cons] = "kidane_wold triple"
    for cons, cls in shape_classes(dataset_dir).items():
        if cons in tags and tags[cons] != cls:
            overrides += 1
        tags[cons] = cls
        source[cons] = "kidane_wold triple"
    held = 0
    for c in exclude:
        if tags.pop(tuple(c), None):
            held += 1
    counts = {"conflicting_roots_dropped": len(conflicts), "held_out_roots_dropped": held,
              "triple_conflicting_roots_dropped": triple_conflicts,
              "triple_overrides_of_other_tags": overrides}
    for cons, cls in tags.items():
        key = f"{cls} ({source.get(cons)})"
        counts[key] = counts.get(key, 0) + 1
    return tags, source, counts


def paradigm_rows(cons: Tuple[str, ...], cls: str, orders: dict = None) -> List[Tuple[dict, str]]:
    """Every form verb_class_paradigms builds for this root, with features.

    Type C imperfect and subjunctive are written the same (ይባርክ); both rows
    are kept, so that surface carries both labels.
    """
    letters = [geez_config.render_unit(c, "a") for c in cons]
    tables = vcp.type_d(letters, orders) if cls == "type_d" else vcp.CLASSES[cls](letters)
    rows = []
    for tname, table in tables.items():
        for pkey, form in table.items():
            person, gender, number = PERSON_OF[pkey]
            rows.append(({"person": person, "gender": gender, "number": number,
                          "tense": TENSE_OF[tname], "stem": "basic",
                          "verb_type": VERB_TYPE_OF[cls], "root": "-".join(cons)}, form))
    return rows


# Two dummy roots per class: letters that differ between the two builds are
# root slots, letters that stay the same are fixed material. Dummy radicals
# (b g d / z f r) never occur in the affixes.
_DUMMIES = {
    "type_b": (("b", "g", "d"), ("z", "f", "r")),
    "type_c": (("b", "g", "d"), ("z", "f", "r")),
    "final_y": (("b", "g", "y"), ("z", "f", "y")),
    "final_w": (("b", "g", "w"), ("z", "f", "w")),
    "initial_w_drop": (("w", "g", "d"), ("w", "f", "r")),
}
# Radicals a class writes as fixed letters or drops (ይተሉ has no w; ይለድ has no w).
IMPLIED_RADICALS = {"final_y": {2: "y"}, "final_w": {2: "w"}, "initial_w_drop": {0: "w"}}


def _spec_from_pair(fa: str, fb: str, ra) -> str:
    """Two builds of one form from two dummy roots -> pattern spec, or None."""
    if len(fa) != len(fb):
        return None
    parts = []
    for ca, cb in zip(fa, fb):
        if ca == cb:
            parts.append(f"[{ca}]")
            continue
        cons, vowel = geez_config.decompose_char(ca)
        if cons not in ra:
            return None
        parts.append(f"C{ra.index(cons)}{vowel}")
    return "-".join(parts)


def type_d_patterns(dataset_dir: str) -> List[dict]:
    """type_d paradigms as pattern specs, one set per distinct vowel-order triple,
    each restricted (only_roots) to the roots whose own triple has those orders.
    Dummies have a non-guttural first radical; no type_d root has one either."""
    groups: Dict[tuple, set] = {}
    for root, orders in type_d_orders(dataset_dir).items():
        key = tuple(sorted(orders.items()))
        groups.setdefault(key, set()).add("-".join(root))
    ra, rb = ("b", "g", "d"), ("z", "f", "r")
    out = []
    for key, roots in sorted(groups.items()):
        orders = dict(key)
        for (feat, fa), (_, fb) in zip(paradigm_rows(ra, "type_d", orders),
                                       paradigm_rows(rb, "type_d", orders)):
            spec = _spec_from_pair(fa, fb, ra)
            if spec is None:
                continue
            feat = {k: v for k, v in feat.items() if k != "root"}
            out.append({"features": feat, "pattern_spec": spec, "only_roots": set(roots),
                        "source": "verb_class_paradigms.type_d"})
    return out


def class_patterns() -> List[dict]:
    """Checked class paradigms as pattern specs (same format as patterns.json)."""
    out = []
    for cls, (ra, rb) in _DUMMIES.items():
        for (feat, fa), (_, fb) in zip(paradigm_rows(ra, cls), paradigm_rows(rb, cls)):
            if len(fa) != len(fb):
                continue
            parts, ok = [], True
            for ca, cb in zip(fa, fb):
                if ca == cb:
                    parts.append(f"[{ca}]")
                    continue
                cons, vowel = geez_config.decompose_char(ca)
                if cons not in ra:
                    ok = False
                    break
                parts.append(f"C{ra.index(cons)}{vowel}")
            if not ok:
                continue
            feat = {k: v for k, v in feat.items() if k != "root"}
            out.append({"features": feat, "pattern_spec": "-".join(parts),
                        "implied_radicals": IMPLIED_RADICALS.get(cls, {}),
                        "source": f"verb_class_paradigms.{cls}"})
    return out

