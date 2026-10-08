#!/usr/bin/env python3
"""
analyzer.py
Morphological analyzer, text lemmatizer, and lexicon lookup for Classical Ge'ez.
Integrates deterministic rule-based fast-path lookup, two-level Semitic
interdigitation pattern matching, dictionary grounding, and neural fallback.
"""

import os
import sys
import json
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any, Set

import geez_config
import gold_filters  # default_dataset_dir(): --dataset_dir, else $GEEZ_DATASET_DIR, else ../geez-dataset
try:
    from .engine import InterdigitationEngine, slots_from_spec, Pattern
except ImportError:
    from engine import InterdigitationEngine, slots_from_spec, Pattern
try:
    import verb_classes  # checked Type B/C, final-weak, initial-w paradigms
except ImportError:
    verb_classes = None

# Ge'ez has one subjunctive; patterns.json still says "jussive" for some templates.
TENSE_ALIASES = {"jussive": "subjunctive"}


def _project_dir() -> str:
    """Folder that holds patterns.json and the trained model (normally this file's folder)."""
    here = os.path.dirname(os.path.abspath(__file__))
    if os.path.exists(os.path.join(here, "patterns.json")):
        return here
    return os.path.dirname(here)


# The dictionary the analyzer reads, inside the geez-dataset checkout.
DICTIONARY_FILE = os.path.join("data", "geez_amharic_dictionary.jsonl")


# Guttural first radical (አ ዐ ሀ ሐ ኀ): imperfect prefix in the 1st order.
GUTTURAL_FIRST = ("a", "a2", "h1", "h2", "kh")
GUTTURAL_PREFIX_SPEC = {"[ይ]": "[የ]", "[ት]": "[ተ]", "[እ]": "[አ]", "[ን]": "[ነ]"}
GUTTURAL_PRESENT_PERSON = {"የ": ("3", "m", "s"), "ተ": ("2", "m", "s"),
                           "አ": ("1", None, "s"), "ነ": ("1", None, "p")}


def normalize_inferred_root(root_str: str) -> str:
    """Normalize inferred root to config consonant symbols (hyphenated).

    Accepts Ethiopic syllables, Latin letters, or mixed forms like
    ``ቀ-በ-ለ (q-b-l)``. Latin ``h`` maps to ``h1``; ``ʾ``/``'`` → ``a``;
    ``ʿ`` → ``a2``.
    """
    if not root_str:
        return ""
    clean = root_str.split("(")[0].strip()
    # Prefer parenthetical Latin if Ethiopic outside has vowels glued
    if "(" in root_str and ")" in root_str:
        inner = root_str[root_str.find("(") + 1 : root_str.find(")")].strip()
        if inner:
            clean = inner

    LATIN = {
        "h": "h1", "H": "h2", "ḥ": "h2", "ḫ": "kh",  # ḥ = ሐ (h2), ḫ = ኀ (kh)
        "'": "a", "ʾ": "a", "’": "a",
        "ʿ": "a2", "`": "a2",
        "s": "s", "š": "sh", "ś": "s2", "S": "S", "ṣ": "S",
        "t": "t", "ṭ": "T", "T": "T",
        "q": "q", "k": "k", "b": "b", "l": "l", "m": "m",
        "n": "n", "r": "r", "w": "w", "y": "y", "d": "d",
        "g": "g", "f": "f", "z": "z", "p": "p",
        "x": "h2", "c": "ch",
    }

    parts = [p.strip() for p in clean.replace("–", "-").split("-") if p.strip()]
    out = []
    for p in parts:
        if len(p) == 1 and p in LATIN:
            out.append(LATIN[p])
            continue
        # Ethiopic syllable or multi-char Latin token already a symbol
        if p in geez_config.CONSONANTS:
            out.append(p)
            continue
        if all(ord(ch) >= 0x1200 for ch in p):
            units = geez_config.decompose_string(p)
            for c, _v in units:
                if c:
                    out.append(c)
            continue
        # glued Latin like "qbl"
        if p.isascii() and p.isalpha():
            for ch in p:
                out.append(LATIN.get(ch, ch))
            continue
        out.append(p)
    return "-".join(out)


@dataclass
class MorphologicalAnalysis:
    surface_form: str
    root_consonants: Tuple[str, ...]
    root_string: str
    canonical_lemma: str
    tense: Optional[str]
    person: Optional[str]
    number: Optional[str]
    gender: Optional[str]
    stem: Optional[str]
    verb_type: Optional[str]
    pattern_spec: str
    lexicon_entry: Optional[Dict[str, Any]] = None
    negative: bool = False
    object: Optional[str] = None
    pos: Optional[str] = None        # None = verb; "noun" / "adjective" from a nominal ByT5 target


@dataclass
class LemmatizedToken:
    token: str
    cleaned_token: str
    proclitics: List[str]
    analyses: List[MorphologicalAnalysis]


class GeezMorphologicalAnalyzer:
    """
    Morphological analyzer and lemmatizer for Classical Ge'ez.
    Supports deterministic rule-based fast path, Semitic interdigitation patterns,
    and dictionary grounding against classical lexica.
    """

    _loose_gutturals = False   # set only during the spelling-variant pass in HybridAnalyzer

    def __init__(self, patterns_path: Optional[str] = None, dictionary_path: Optional[str] = None,
                 dataset_dir: Optional[str] = None):
        """dataset_dir: the geez-dataset checkout (the scripts' --dataset_dir). When it is None the
        usual default is used: $GEEZ_DATASET_DIR, else ../geez-dataset. The dictionary
        (<dataset_dir>/data/geez_amharic_dictionary.jsonl, or dictionary_path) must exist:
        without it the analyzer would run with no lexicon and give much worse answers, so a
        missing dictionary is an error, not a silent fallback."""
        base_dir = _project_dir()
        self.dataset_dir = os.path.abspath(dataset_dir or gold_filters.default_dataset_dir())
        if dictionary_path is None:
            dictionary_path = os.path.join(self.dataset_dir, DICTIONARY_FILE)
        if not os.path.exists(dictionary_path):
            raise FileNotFoundError(
                f"Dictionary not found: {dictionary_path}. The analyzer needs a geez-dataset checkout: "
                "pass --dataset_dir (dataset_dir=...), or set $GEEZ_DATASET_DIR, or clone the dataset "
                "to ../geez-dataset.")

        # Load patterns
        if patterns_path is None:
            patterns_path = os.path.join(base_dir, "patterns.json")
        with open(patterns_path, "r", encoding="utf-8") as f:
            self.patterns_data = json.load(f)

        # Quadriliteral templates (shared with the generator)
        quad_path = os.path.join(base_dir, "patterns_quadriliteral.json")
        self.quad_patterns = []
        if os.path.exists(quad_path):
            with open(quad_path, "r", encoding="utf-8") as f:
                self.quad_patterns = [dict(q, source="patterns_quadriliteral.json") for q in json.load(f)]

        # Checked class paradigms (Type B/C, final-weak, ወለደ-type) as extra patterns
        self.class_patterns = verb_classes.class_patterns() if verb_classes else []

        # Guttural first radical: basic imperfect የኀፍር, not ይኀፍር;
        # ይ + C0a with a guttural first radical is the subjunctive (ይኀፍር).
        gutt_copies = []
        for p in self.patterns_data + self.quad_patterns:
            f = p["features"]
            spec = p["pattern_spec"]
            if (f.get("stem") != "basic" or f.get("tense") != "present"
                    or f.get("verb_type") not in ("strong_A", "weak_middle", "quadriliteral")):
                continue
            for old, new in GUTTURAL_PREFIX_SPEC.items():
                if spec.startswith(old):
                    p["not_first"] = GUTTURAL_FIRST
                    gutt_copies.append(dict(p, pattern_spec=new + spec[len(old):],
                                            first_in=GUTTURAL_FIRST, not_first=None,
                                            source=p.get("source", "patterns.json") + "+guttural_prefix"))
                    if f.get("verb_type") == "strong_A" and spec[len(old):].startswith("-C0a-C1I-C2"):
                        gutt_copies.append(dict(p, features=dict(f, tense="subjunctive"),
                                                first_in=GUTTURAL_FIRST, not_first=None,
                                                source=p.get("source", "patterns.json") + "+guttural_subjunctive"))
                    break
        # Middle guttural (ፍሬ ግእዝ pp. 18, 63), non-guttural first radical: present
        # ይምሕል (all 6th, no doubling), subjunctive/jussive and imperative ይምሐል / ምሐል.
        for p in self.patterns_data:
            f = p["features"]
            spec = p["pattern_spec"]
            if f.get("stem") != "basic" or f.get("verb_type", "strong_A") != "strong_A":
                continue
            t = f.get("tense")
            if t == "present" and "-C0a-C1I-" in spec and spec[:3] in ("[ይ]", "[ት]", "[እ]", "[ን]"):
                new = spec.replace("-C0a-", "-C0I-", 1)
            elif t in ("subjunctive", "jussive", "imperative") and "C0I-C1I-" in spec:
                new = spec.replace("C1I", "C1a", 1)
            else:
                continue
            p["not_mid_gutt"] = True
            gutt_copies.append(dict(p, pattern_spec=new, not_mid_gutt=None, mid_gutt_only=True,
                                    not_first=None, first_in=None,
                                    source=p.get("source", "patterns.json") + "+middle_guttural"))
        dataset_dir = self.dataset_dir
        type_d_pats = verb_classes.type_d_patterns(dataset_dir) if verb_classes else []

        # Precompile pattern slots
        self.compiled_patterns = []
        for p in self.patterns_data + self.quad_patterns + self.class_patterns + type_d_pats + gutt_copies:
            slots = slots_from_spec(p["pattern_spec"])
            expanded_slots = []
            for s in slots:
                if s[0] == "F":
                    for ch in s[1]:
                        expanded_slots.append(("F", ch))
                elif s[0] == "R":
                    expanded_slots.append(("R", s[1], s[2]))
            self.compiled_patterns.append({
                "features": p["features"],
                "pattern_spec": p["pattern_spec"],
                "slots": slots,
                "expanded_slots": expanded_slots,
                "implied": p.get("implied_radicals", {}),
                "source": p.get("source", "patterns.json"),
            })
            for key in ("only_roots", "first_in", "not_first", "not_mid_gutt", "mid_gutt_only"):
                if p.get(key) is not None:
                    self.compiled_patterns[-1][key] = p[key]

        # Verb class per root (Type B / Type C / ሰትየ-type) from lexical sources only;
        # a Type B perfect is spelled like Type A, so spelling cannot decide it.
        self.class_tags = verb_classes.class_tags(dataset_dir)[0] if verb_classes else {}
        if verb_classes:
            # A class paradigm only matches roots of that class: Type B, Type C,
            # ተለወ-type and ሰትየ-type need a lexical tag (Kidane Wold triple, Merho
            # Ges, class lemma); ወለደ-type needs W_DROP or a ይለድ-shaped triple.
            only = {cls: {"-".join(r) for r, c in self.class_tags.items() if c == cls}
                    for cls in ("type_b", "type_c", "final_y", "final_w")}
            only["initial_w_drop"] = set(verb_classes.vcp.W_DROP) | {
                "-".join(r) for r, c in self.class_tags.items()
                if c == "initial_w_drop" and verb_classes.w_drop_shape(r)}
            for cp in self.compiled_patterns:
                cls = cp["source"].replace("verb_class_paradigms.", "")
                if cls in only:
                    cp["only_roots"] = only[cls]

        # Initialize generation engine for lemma synthesis
        self.engine = InterdigitationEngine(renderer=geez_config.render_unit)
        self.lemma_spec = slots_from_spec("C0a-C1a-C2a")
        self.lemma_spec_biliteral = slots_from_spec("C0a-C1a")

        # Load dictionary index
        self.lexicon: Dict[str, List[Dict[str, Any]]] = {}
        self.root_index: Dict[str, List[Dict[str, Any]]] = {}
        self.inferred_root_index: Dict[str, List[Dict[str, Any]]] = {}
        self.high_conf_roots: Set[str] = set()
        self.root_canonical_lemma: Dict[str, str] = {}
        self.root_verb_class: Dict[str, str] = {}

        self.root_fixes = verb_classes.root_fixes(dataset_dir) if verb_classes else {}
        with open(dictionary_path, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                hw = entry.get("headword_geez")
                if hw:
                    self.lexicon.setdefault(hw, []).append(entry)
                    
                    # Also index 1st-order lemma if headword is 6th-order (e.g. ቀተል -> ቀተለ)
                    if len(hw) >= 3 and (hw.endswith("ል") or any(hw.endswith(c) for c in ["ር", "ም", "ስ", "ቅ", "ብ", "ት", "ክ"])):
                        units = geez_config.decompose_string(hw)
                        if units and units[-1][1] == "I":
                            alt_chars = list(hw)
                            alt_chars[-1] = geez_config.render_unit(units[-1][0], "a")
                            alt_hw = "".join(alt_chars)
                            self.lexicon.setdefault(alt_hw, []).append(entry)

                # Index by consonantal root (with the Kidane Wold root fixes)
                root = (verb_classes.entry_root(entry, self.root_fixes) if verb_classes
                        else entry.get("inferred_root") or entry.get("root"))
                cleaned_root = root.split("(")[0].strip().replace("-", "") if root else ""
                if cleaned_root:  # never index an empty or blank root
                    self.root_index.setdefault(cleaned_root, []).append(entry)
                    self.inferred_root_index.setdefault(cleaned_root, []).append(entry)

                    conf = entry.get("root_confidence")
                    if conf == "high":
                        self.high_conf_roots.add(cleaned_root)
                    elif conf == "medium" and cleaned_root not in self.high_conf_roots:
                        # If no high exists yet, keep medium
                        pass

                    # Cache canonical lemma
                    if cleaned_root not in self.root_canonical_lemma:
                        if entry.get("pos") == "verb" or entry.get("verb_class"):
                            if hw:
                                units = geez_config.decompose_string(hw)
                                if units and units[-1][1] == "I":
                                    alt = list(hw)
                                    alt[-1] = geez_config.render_unit(units[-1][0], "a")
                                    self.root_canonical_lemma[cleaned_root] = "".join(alt)
                                else:
                                    self.root_canonical_lemma[cleaned_root] = hw
                            if entry.get("verb_class"):
                                self.root_verb_class[cleaned_root] = entry["verb_class"]

    def _match_compiled_pattern(self, surface: str, compiled_p: Dict[str, Any]) -> Optional[Tuple[str, ...]]:
        expanded = compiled_p["expanded_slots"]
        if len(surface) != len(expanded):
            return None

        units = geez_config.decompose_string(surface)
        root_map = {}
        for i in range(len(surface)):
            ch = surface[i]
            c, v = units[i]
            slot = expanded[i]
            if slot[0] == "F":
                if ch != slot[1] and not (self._loose_gutturals and geez_config.gutturals_equal(ch, slot[1])):
                    return None
            elif slot[0] == "R":
                slot_idx = slot[1]
                slot_v = slot[2]
                if v != slot_v and not (self._loose_gutturals and c in geez_config.GUTTURAL_SYMBOLS
                                        and {v, slot_v} == {"a", "A"}):
                    return None
                if slot_idx in root_map and root_map[slot_idx] != c:
                    return None
                root_map[slot_idx] = c

        slots = compiled_p["slots"]
        implied = compiled_p.get("implied", {})
        max_idx = max([s[1] for s in slots if s[0] == "R"] + list(implied))
        # Radicals the class writes as a fixed letter or drops (ይተሉ: w; ይለድ: w).
        for idx, cons in implied.items():
            root_map.setdefault(idx, cons)

        # Recover elided medial glide for weak_middle patterns (e.g. ቆመ -> q-w-m, ሤመ -> s-y-m)
        if max_idx == 2 and 1 not in root_map and compiled_p["features"].get("verb_type") == "weak_middle":
            spec = compiled_p.get("pattern_spec", "")
            if any(x in spec for x in ("C0o", "C0u", "[ው]")):
                root_map[1] = "w"
            else:
                root_map[1] = "y"

        if len(root_map) != max_idx + 1:
            return None

        root = tuple(root_map[i] for i in range(max_idx + 1))
        only = compiled_p.get("only_roots")
        if only is not None and "-".join(root) not in only:
            return None
        if "first_in" in compiled_p and root[0] not in compiled_p["first_in"]:
            return None
        if "not_first" in compiled_p and root[0] in compiled_p["not_first"]:
            return None
        mid_gutt = len(root) == 3 and root[1] in GUTTURAL_FIRST and root[0] not in GUTTURAL_FIRST
        if compiled_p.get("not_mid_gutt") and mid_gutt:
            return None
        if compiled_p.get("mid_gutt_only") and not mid_gutt:
            return None
        return root

    def fast_path_analyze(self, word: str) -> List[MorphologicalAnalysis]:
        """Dictionary-backed hybrid fast path.

        Tries affix splits as hypotheses; keeps only stems whose consonants
        hit the inferred-root index. Fused subject/object vowels on the last
        radical are read via decompose_char. Returns every high-confidence
        candidate, ranked: full headword match first, then fewer affixes.
        """
        cleaned = word.strip().strip("፡።፣፤፥፦():[]\"'")
        if not cleaned or len(cleaned) < 2:
            return []

        # Longest first. ይት before ይ/ት; ያ = causative imperfect. No Amharic አስ-:
        # Ge'ez has አስተ- (aste). የ/ነ only as the guttural-initial imperfect (የኀፍር).
        stem_prefixes = ["አስተ", "ይት", "ያ", "ይ", "የ", "ት", "ተ", "አ", "እ", "ን", "ነ", ""]
        # Separate-letter object suffixes. Third-person objects change shape
        # with the vowel before them: after -u they are ዎ/ዋ/ዎሙ/ዎን
        # (ቀተልኩዎ), after -i they are ዮ/ያ/ዮሙ/ዮን (ቀተልክዮ). After -o
        # (fused 3ms object, ቀተሎ) a trailing ሙ/ን makes it 3mp/3fp (ቀተሎሙ).
        obj_suffixes = [
            "ዎሙ", "ዎን", "ዮሙ", "ዮን", "ሆሙ", "ሆን", "ክሙ", "ክን",
            "ዎ", "ዋ", "ዮ", "ያ", "ኒ", "ሁ", "ሃ", "ከ", "ኪ", "ነ", "ሙ", "ን",
            "ኮሙ", "ኮን", "ኮ", "",
        ]
        # 2ms subject -ከ fused with a "him/them" object: ቀተልኮ, ቀተልኮሙ, ቀተልኮን.
        fused_2ms = {"ኮ": "3ms", "ኮሙ": "3mp", "ኮን": "3fp"}
        after_u = {"ዎ", "ዋ", "ዎሙ", "ዎን"}
        after_i = {"ዮ", "ያ", "ዮሙ", "ዮን"}
        after_o = {"ሙ", "ን"}
        # Subject suffixes that are separate letters (not fused -u/-ā).
        subj_suffixes = ["ክሙ", "ክን", "ኩ", "ከ", "ኪ", "ነ", "ት", "ተ", ""]

        candidates = []

        proclitic_options = ["ወ", "ዘ", ""] if cleaned[0] in ("ወ", "ዘ") and len(cleaned) > 3 else [""]
        for proc in proclitic_options:
            rem_proc = cleaned[len(proc):] if proc else cleaned

            negative_options = ["ኢ", ""] if rem_proc.startswith("ኢ") and len(rem_proc) > 3 else [""]
            for neg in negative_options:
                rem_neg = rem_proc[len(neg):] if neg else rem_proc

                for pref in stem_prefixes:
                    if pref and not rem_neg.startswith(pref):
                        continue
                    rem_pref = rem_neg[len(pref):] if pref else rem_neg
                    if len(rem_pref) < 2:
                        continue

                    for s_obj in obj_suffixes:
                        if s_obj and not rem_pref.endswith(s_obj):
                            continue
                        rem_obj = rem_pref[:-len(s_obj)] if s_obj else rem_pref
                        if s_obj in fused_2ms:
                            rem_obj = rem_obj + "ከ"
                        if len(rem_obj) < 2:
                            continue
                        prev_cons, prev_vowel = geez_config.decompose_char(rem_obj[-1])
                        # Before a 3rd-person object, a subject -u/-i may drop to
                        # sixth order: ቀተልክዎ = ቀተልኩዎ, ቀተልክዮ = ቀተልኪዮ, ቀተልዎ = ቀተሉዎ.
                        # Restore the vowel so the subject is read as usual.
                        if s_obj in after_u and prev_vowel == "I" and prev_cons:
                            rem_obj = rem_obj[:-1] + geez_config.render_unit(prev_cons, "u")
                            prev_vowel = "u"
                        if s_obj in after_i and prev_vowel == "I" and prev_cons:
                            rem_obj = rem_obj[:-1] + geez_config.render_unit(prev_cons, "i")
                            prev_vowel = "i"
                        if s_obj in after_u and prev_vowel != "u":
                            continue
                        if s_obj in after_i and prev_vowel != "i":
                            continue
                        if s_obj in after_o and prev_vowel != "o":
                            continue

                        for s_subj in subj_suffixes:
                            if s_obj in fused_2ms and s_subj != "ከ":
                                continue
                            if s_subj and not rem_obj.endswith(s_subj):
                                continue
                            stem = rem_obj[:-len(s_subj)] if s_subj else rem_obj
                            if len(stem) < 2:
                                continue

                            # Fused last-vowel readings on the stem (no separate letter).
                            fused_readings = self._fused_last_vowel_readings(stem)
                            if s_obj in after_o:
                                # ሙ/ን only attach to the fused -o object reading.
                                fused_readings = [r for r in fused_readings if r[2] == "3ms"]
                            for stem_core, fused_subj, fused_obj in fused_readings:
                                dec = geez_config.decompose_string(stem_core)
                                cons = tuple(c for c, v in dec if c is not None)
                                final_w = bool(fused_subj and fused_subj.get("final_w"))
                                affix_count = (
                                    (1 if proc else 0)
                                    + (1 if neg else 0)
                                    + (1 if pref else 0)
                                    + (1 if s_subj else 0)
                                    + (1 if s_obj else 0)
                                    + (1 if fused_subj and not final_w else 0)
                                    + (1 if fused_obj else 0)
                                )
                                root_keys = []
                                if final_w:
                                    # ይተሉ = ይ + t-l-w imperfect 3ms, not a 3mp ending on t-?-l
                                    if len(cons) == 2:
                                        root_keys.append((cons[0], cons[1], "w"))
                                elif len(cons) == 3:
                                    root_keys.append(cons)
                                elif len(cons) == 2:
                                    # Hollow: try both glides
                                    root_keys.append((cons[0], "w", cons[1]))
                                    root_keys.append((cons[0], "y", cons[1]))
                                elif len(cons) == 4:
                                    root_keys.append(cons)

                                for rk in root_keys:
                                    key = "".join(rk)
                                    if key not in self.high_conf_roots and key not in self.inferred_root_index:
                                        continue
                                    headword_hit = cleaned in self.lexicon or stem_core in self.lexicon
                                    candidates.append({
                                        "cons": rk,
                                        "key": key,
                                        "prefix": pref,
                                        "subj_suffix": s_subj,
                                        "obj_suffix": s_obj,
                                        "fused_subj": fused_subj,
                                        "fused_obj": fused_obj,
                                        "stem": stem_core,
                                        "proc": proc,
                                        "neg": neg,
                                        "vowels": [v for c, v in dec],
                                        "affix_count": affix_count,
                                        "headword_hit": headword_hit,
                                        "final_w": final_w,
                                    })

        if not candidates:
            return []

        # Prefer high-confidence index hits
        high_cands = [c for c in candidates if c["key"] in self.high_conf_roots]
        active = high_cands if high_cands else candidates

        # Rank: full-word headword first, then fewer affixes
        # Final-w readings go before 3mp readings of the same word (ይተሉ).
        active.sort(key=lambda c: (0 if c["headword_hit"] else 1, 0 if c.get("final_w") else 1,
                                   c["affix_count"], c["key"]))

        analyses: List[MorphologicalAnalysis] = []
        seen = set()
        for best in active:
            for analysis in self._analysis_from_fast_candidate(cleaned, best):
                sig = (
                    analysis.root_string,
                    analysis.tense,
                    analysis.person,
                    analysis.number,
                    analysis.gender,
                    analysis.stem,
                    analysis.object,
                    analysis.negative,
                )
                if sig in seen:
                    continue
                seen.add(sig)
                analyses.append(analysis)
        return analyses

    def _fused_last_vowel_readings(self, stem: str):
        """Yield (stem_for_root, fused_subj, fused_obj) for fused final vowels.

        Final -u → 3mp subject. Final -ā → ambiguous 3fp subject OR 3fs object.
        Final -o → 3ms object (ቀተሎ). Also yield the unfused stem.
        """
        readings = [(stem, None, None)]
        if not stem:
            return readings
        last = stem[-1]
        c, v = geez_config.decompose_char(last)
        if c is None:
            return readings
        # Rebuild stem with last syllable in 1st order (ä/a) for root extraction
        base_last = geez_config.render_unit(c, "a")
        core = stem[:-1] + base_last
        if v in ("u", "A", "o") and len(stem) >= 2:
            readings = []
        if v == "u":
            # Final-w reading first: the -u is the third radical w (ይተሉ, t-l-w).
            readings.append((stem, {"final_w": True}, None))
            readings.append((core, {"person": "3", "number": "p", "gender": "m"}, None))
        elif v == "A":  # long ā — 3fp subject OR 3fs object
            readings.append((core, {"person": "3", "number": "p", "gender": "f"}, None))
            readings.append((core, None, "3fs"))
        elif v == "o":
            readings.append((core, None, "3ms"))
        return readings

    def _analysis_from_fast_candidate(self, cleaned: str, best: dict) -> List[MorphologicalAnalysis]:
        cons = best["cons"]
        key = best["key"]
        pref = best["prefix"]
        s_subj = best["subj_suffix"]
        s_obj = best["obj_suffix"]
        fused_subj = best.get("fused_subj")
        fused_obj = best.get("fused_obj")
        neg = best.get("neg") or ""

        if pref == "ተ" or pref == "ይት":
            stem_feat = "passive"
        elif pref in ("አ", "ያ"):
            stem_feat = "causative"
        elif pref == "አስተ":
            stem_feat = "aste"
        else:
            stem_feat = "basic"

        tense = None
        person = None
        gender = None
        number = None
        obj = None

        obj_map = {
            "ኒ": "1s", "ከ": "2ms", "ኪ": "2fs", "ሁ": "3ms", "ሃ": "3fs",
            "ነ": "1p", "ክሙ": "2mp", "ክን": "2fp", "ሆሙ": "3mp", "ሆን": "3fp",
            "ዎ": "3ms", "ዋ": "3fs", "ዎሙ": "3mp", "ዎን": "3fp",
            "ዮ": "3ms", "ያ": "3fs", "ዮሙ": "3mp", "ዮን": "3fp",
            "ኮ": "3ms", "ኮሙ": "3mp", "ኮን": "3fp",
        }
        if s_obj:
            obj = obj_map.get(s_obj)
        if fused_obj:
            obj = fused_obj
        if s_obj == "ሙ":
            obj = "3mp"
        elif s_obj == "ን":
            obj = "3fp"

        # Imperfect vs. subjunctive after a person prefix: the imperfect stem
        # opens with -a- (ይቀትል, ይቀውም); the subjunctive opens with -ə-/-u-/-i-
        # (ይቅትል, ይቁም).
        vowels = best["vowels"]
        v0 = vowels[0] if vowels else None
        vclass = self.class_tags.get(tuple(cons))
        # Type A imperfect ይቀትል (a) / subjunctive ይቅትል (ə); Type B imperfect
        # ይቄድስ (e) / subjunctive ይቀድስ (a, so a tagged Type B root reads a as
        # subjunctive); Type C ይባርክ (ā) is both imperfect and subjunctive.
        if v0 == "e" or v0 == "A" or (v0 == "a" and vclass != "type_b"):
            prefixed_tense = "present"
        else:
            prefixed_tense = "subjunctive"
        # Middle guttural (ፍሬ ግእዝ): ይምሕል (6th, 6th) present, ይምሐል (6th, 1st) subjunctive.
        if (len(cons) == 3 and cons[1] in GUTTURAL_FIRST and cons[0] not in GUTTURAL_FIRST
                and vclass in (None, "type_a") and v0 == "I" and len(vowels) > 1):
            if vowels[1] == "I":
                prefixed_tense = "present"
            elif vowels[1] in ("a", "A"):
                prefixed_tense = "subjunctive"

        if pref == "ይ":
            person, gender, tense = "3", "m", prefixed_tense
            number = "s"
        elif pref == "ያ":
            person, gender, tense, number = "3", "m", prefixed_tense, "s"
            stem_feat = "causative"
        elif pref == "ት":
            person, tense = "2", prefixed_tense
            gender, number = "m", "s"
        elif pref == "እ":
            person, number, tense = "1", "s", prefixed_tense
        elif pref == "ን":
            person, number, tense = "1", "p", prefixed_tense
        elif pref == "ይት":
            person, gender, tense, number = "3", "m", prefixed_tense, "s"
            stem_feat = "passive"
        elif s_subj == "ኩ":
            tense, person, number = "past", "1", "s"
        elif s_subj == "ከ":
            tense, person, gender, number = "past", "2", "m", "s"
        elif s_subj == "ኪ":
            tense, person, gender, number = "past", "2", "f", "s"
        elif s_subj in ("ት", "ተ"):
            tense, person, gender, number = "past", "3", "f", "s"
        elif s_subj == "ነ":
            tense, person, number = "past", "1", "p"
        elif s_subj == "ክሙ":
            tense, person, gender, number = "past", "2", "m", "p"
        elif s_subj == "ክን":
            tense, person, gender, number = "past", "2", "f", "p"
        else:
            # No person prefix and no subject letter: a stem ending in -a is a
            # 3ms perfect in every stem (ቀተለ, ተቀትለ, አቅተለ, hollow ቆመ).
            if vowels and vowels[-1] == "a":
                tense, person, gender, number = "past", "3", "m", "s"

        # Guttural first radical: the basic imperfect prefix is የ/ተ/አ/ነ (የኀፍር,
        # ተኀፍር, አኀፍር, ነኀፍር); ይ/ት/እ/ን + C0a is the subjunctive (ይኀፍር).
        fused_ok = bool(fused_subj) and not fused_subj.get("final_w")
        gutt_ok = (bool(cons) and cons[0] in GUTTURAL_FIRST and v0 == "a" and not s_subj
                   and vclass not in ("type_b", "type_c", "type_c_o", "type_c_e", "type_d")
                   and ((vowels[-1] == "I" and not fused_subj) or fused_ok))
        if pref in ("የ", "ነ") and not gutt_ok:
            return []
        if gutt_ok and pref in GUTTURAL_PRESENT_PERSON:
            person, gender, number = GUTTURAL_PRESENT_PERSON[pref]
            tense, stem_feat = "present", "basic"
        elif gutt_ok and pref in ("ይ", "ት", "እ", "ን"):
            tense = "subjunctive"

        if fused_subj and not fused_subj.get("final_w"):
            person = fused_subj.get("person", person)
            number = fused_subj.get("number", number)
            gender = fused_subj.get("gender", gender)
            if tense is None:
                tense = "past" if not pref else tense or "present"
            if number == "p" and (pref == "ት" or (gutt_ok and pref == "ተ")):
                person = "2"   # ትቀትሉ / ተኀፍሩ 2mp, ትቀትላ 2fp

        lemma = self.root_canonical_lemma.get(key)
        if not lemma:
            try:
                if len(cons) == 3:
                    lemma = self.engine.generate(cons, self.lemma_spec)
                elif len(cons) == 2:
                    lemma = self.engine.generate(cons, self.lemma_spec_biliteral)
                else:
                    lemma = cleaned
            except Exception:
                lemma = cleaned

        lex_entries = self.inferred_root_index.get(key, [])
        lex_entry = lex_entries[0] if lex_entries else None
        root_str = "-".join(cons)

        verb_type = "strong_A"
        if cons[0] in ("w", "y"):
            verb_type = "weak_initial"
        elif len(cons) > 1 and cons[1] in ("w", "y"):
            verb_type = "weak_middle"
        elif len(cons) > 2 and cons[2] in ("w", "y"):
            verb_type = "weak_final"

        pattern_spec = (
            f"fast_path:[{pref}]-stem-[{s_subj}]-[{s_obj}]"
            f"-fused_subj={fused_subj}-fused_obj={fused_obj}"
        )

        if vclass == "type_b":
            verb_type = "strong_B"
        elif vclass == "type_c":
            verb_type = "strong_C"

        def make(t):
            return MorphologicalAnalysis(
                surface_form=cleaned,
                root_consonants=cons,
                root_string=root_str,
                canonical_lemma=lemma,
                tense=t,
                person=person,
                number=number,
                gender=gender,
                stem=stem_feat,
                verb_type=verb_type,
                pattern_spec=pattern_spec,
                lexicon_entry=lex_entry,
                negative=bool(neg),
                object=obj,
            )

        out = [make(tense)]
        if v0 == "A" and tense == "present":
            out.append(make("subjunctive"))  # ይባርክ: same spelling for both
        return out

    def analyze(self, word: str, lookup_dictionary: bool = True, use_hybrid: bool = True) -> List[MorphologicalAnalysis]:
        """
        Analyze a single Ge'ez surface word into candidate roots and morphological features.
        When use_hybrid=True, fast path deterministic stripping & dictionary lookup is attempted first.
        Falls back to two-level Semitic interdigitation pattern matching when unmatched or ambiguous.
        """
        word = word.strip().strip("፡።፣፤፥፦")
        if not word:
            return []

        # 0. Class paradigms licensed by the lexicon (tagged roots / W_DROP)
        if use_hybrid:
            licensed = self.licensed_class_analyze(word, lookup_dictionary)
            if licensed:
                return licensed

        # 1. Fast path (Deterministic Rule + Dictionary Grounding)
        if use_hybrid:
            fast_results = self.fast_path_analyze(word)
            if fast_results:
                return self._prefer_featured(word, fast_results, lookup_dictionary)

        # 2. Pattern-based fallback
        return self._pattern_analyze(word, self.compiled_patterns, lookup_dictionary)

    def _prefer_featured(self, word: str, fast_results, lookup_dictionary: bool = True):
        """A headword hit with no tense only says the word is in the dictionary.
        If a template reads the word with the same root, put that reading first."""
        if fast_results[0].tense is not None:
            return fast_results
        fast_roots = {r.root_string for r in fast_results}
        featured = [r for r in self._pattern_analyze(word, self.compiled_patterns, lookup_dictionary)
                    if r.root_string in fast_roots and r.tense is not None]
        return featured + fast_results if featured else fast_results

    def licensed_class_analyze(self, word: str, lookup_dictionary: bool = True) -> List[MorphologicalAnalysis]:
        """Only the class paradigms restricted to listed roots (see only_roots)."""
        word = word.strip().strip("፡።፣፤፥፦")
        licensed = [p for p in self.compiled_patterns if p.get("only_roots") is not None]
        return self._pattern_analyze(word, licensed, lookup_dictionary) if word else []

    def _pattern_analyze(self, word: str, patterns, lookup_dictionary: bool = True) -> List[MorphologicalAnalysis]:
        results = []
        seen = set()

        for p in patterns:
            root_cons = self._match_compiled_pattern(word, p)
            if root_cons:
                feat = p["features"]

                # Hollow verb glide recovery: reconstruct true medial radical w/y
                if len(root_cons) == 2 and feat.get("verb_type") == "weak_middle":
                    spec = p.get("pattern_spec", "")
                    if any(x in spec for x in ("C0o", "C0u", "[ው]")):
                        root_cons = (root_cons[0], "w", root_cons[1])
                    else:
                        root_cons = (root_cons[0], "y", root_cons[1])

                full_root = root_cons
                root_str = "-".join(full_root)

                # Synthesize citation lemma (3ms perfect)
                try:
                    if len(full_root) == 3:
                        canonical_lemma = self.engine.generate(full_root, self.lemma_spec)
                    elif len(full_root) == 2:
                        canonical_lemma = self.engine.generate(full_root, self.lemma_spec_biliteral)
                    else:
                        canonical_lemma = word
                except Exception:
                    lex = self.root_index.get("".join(full_root))
                    if lex and lex[0].get("headword_geez"):
                        canonical_lemma = lex[0]["headword_geez"]
                    else:
                        canonical_lemma = word

                sig = (root_str, TENSE_ALIASES.get(feat.get("tense"), feat.get("tense")),
                       feat.get("person"), feat.get("number"), feat.get("gender"), feat.get("stem"))
                if sig in seen:
                    continue
                seen.add(sig)

                # Dictionary lookup
                lex_entries = None
                if lookup_dictionary and self.lexicon:
                    lex_entries = self.lookup_lexicon(canonical_lemma) or self.lookup_lexicon(word)
                    if not lex_entries:
                        lex_entries = self.root_index.get("".join(full_root))

                analysis = MorphologicalAnalysis(
                    surface_form=word,
                    root_consonants=root_cons,
                    root_string=root_str,
                    canonical_lemma=canonical_lemma,
                    tense=TENSE_ALIASES.get(feat.get("tense"), feat.get("tense")),
                    person=feat.get("person"),
                    number=feat.get("number"),
                    gender=feat.get("gender"),
                    stem=feat.get("stem"),
                    verb_type=feat.get("verb_type"),
                    pattern_spec=p["pattern_spec"],
                    lexicon_entry=lex_entries[0] if lex_entries else None
                )
                results.append(analysis)

        return results

    def lemmatize_text(self, text: str) -> List[LemmatizedToken]:
        """
        Tokenize Ge'ez text by Ethiopic word dividers, strip common proclitics, and analyze each token.
        """
        raw_tokens = [t.strip() for t in text.replace("::", " ").replace("፡", " ").replace("።", " ").split() if t.strip()]
        token_results = []
        proclitic_prefixes = ["ወ", "ዘ", "በ", "ለ", "እም"]

        for tok in raw_tokens:
            cleaned = tok.strip("፣፤፥፦():[]")
            proclitics = []
            
            stem_candidate = cleaned
            if len(cleaned) > 3 and cleaned[0] in proclitic_prefixes:
                proclitics.append(cleaned[0])
                stem_candidate = cleaned[1:]

            analyses = self.analyze(stem_candidate)
            if not analyses and proclitics:
                analyses = self.analyze(cleaned)
                if analyses:
                    proclitics = []

            token_results.append(LemmatizedToken(
                token=tok,
                cleaned_token=stem_candidate if proclitics else cleaned,
                proclitics=proclitics,
                analyses=analyses
            ))

        return token_results

    def lookup_lexicon(self, word_or_lemma: str) -> List[Dict[str, Any]]:
        """Direct lookup in Kidane Wold Kifle lexicon."""
        results = self.lexicon.get(word_or_lemma, [])
        if not results:
            units = geez_config.decompose_string(word_or_lemma)
            cons = "".join(c for c, v in units if c is not None)
            if cons and cons in self.root_index:
                results = self.root_index[cons]
        return results


NEURAL_FIELDS = ("root", "tense", "person", "gender", "number", "stem")   # verb target
NEURAL_NOMINAL_FIELDS = ("root", "type", "number")                       # + optional gender
NOMINAL_TYPES = ("noun", "adjective")
NO_VALUE = ("", "none", "null")


def parse_neural_output(text: str) -> Optional[Dict[str, Any]]:
    """Strict parse of a ByT5 prediction in one of the two training target shapes
    (gold_filters.target_text):

      verb     "root: ቀ-ተ-ለ | tense: present | person: 3 | gender: m | number: s | stem: basic"
               with an optional " | object: 3ms" at the end;
      nominal  "root: ቀ-ተ-ለ | type: noun | gender: m | number: s" (gender optional).

    Each key appears at most once; the keys must be exactly one shape's (no others). The root
    must be non-empty and made only of Ethiopic letters (hyphens and spaces between them are
    allowed). Empty values and "None" mean "no value" for the other fields. Returns a dict with
    root, type ("verb", "noun" or "adjective"), tense, person, gender, number, stem, object and
    "root_consonants" (fields a shape does not have are None), or None for any other text."""
    if not isinstance(text, str) or not text.strip():
        return None
    out: Dict[str, Any] = {}
    for part in text.split("|"):
        if ":" not in part:
            return None
        k, v = part.split(":", 1)
        k, v = k.strip(), v.strip()
        if k in out:
            return None
        out[k] = v
    keys = set(out)
    if keys - {"object"} == set(NEURAL_FIELDS):
        out["type"] = "verb"
    elif keys - {"gender"} == set(NEURAL_NOMINAL_FIELDS) and out["type"] in NOMINAL_TYPES:
        pass
    else:
        return None
    letters = out["root"].replace("-", "").replace(" ", "")
    if not letters:
        return None
    cons = []
    for ch in letters:
        dec = geez_config.decompose_char(ch)
        if not dec or not dec[0]:
            return None
        cons.append(dec[0])
    for k in ("tense", "person", "gender", "number", "stem", "object"):
        v = out.get(k)
        out[k] = None if v is None or v.lower() in NO_VALUE else v
    out["root_consonants"] = cons
    return out


class HybridAnalyzer(GeezMorphologicalAnalyzer):
    """
    Hybrid Morphological Analyzer for Classical Ge'ez.
    Combines:
    1. Deterministic rule-based fast path (affix stripping + dictionary root index)
    2. Semitic root-and-pattern interdigitation matcher (105 morphological templates)
    3. Neural ByT5 sequence-to-sequence fallback for unseen roots and complex forms.
    """

    def __init__(
        self,
        patterns_path: Optional[str] = None,
        dictionary_path: Optional[str] = None,
        checkpoint_dir: Optional[str] = None,
        device: Optional[str] = None,
        dataset_dir: Optional[str] = None,
    ):
        super().__init__(patterns_path=patterns_path, dictionary_path=dictionary_path, dataset_dir=dataset_dir)
        base_dir = _project_dir()
        
        if checkpoint_dir is None:
            # Check default checkpoint locations
            candidates = [
                os.path.join(base_dir, "byt5_morphology_model"),     # train_byt5_morphology.py's final model
                os.path.join(base_dir, "checkpoints", "checkpoint-3936"),
                os.path.join(base_dir, "checkpoints", "checkpoint-2624"),
                os.path.join(base_dir, "checkpoints_backup_20k", "checkpoint-3936"),
                os.path.join(base_dir, "checkpoints"),
            ]
            for c in candidates:
                if os.path.exists(os.path.join(c, "model.safetensors")):
                    checkpoint_dir = c
                    break

        self.checkpoint_dir = checkpoint_dir
        self.device = device
        self._tokenizer = None
        self._model = None

    def _ensure_neural_model(self):
        """Lazily load neural ByT5 model when fallback is triggered."""
        if self._model is not None:
            return True
        if not self.checkpoint_dir or not os.path.exists(self.checkpoint_dir):
            return False

        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

            if self.device is None:
                self.device = "mps" if torch.backends.mps.is_available() else "cpu"

            self._tokenizer = AutoTokenizer.from_pretrained(self.checkpoint_dir)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(self.checkpoint_dir).to(self.device)
            self._model.eval()
            return True
        except Exception:
            return False

    def _neural_fallback(self, word: str) -> List[MorphologicalAnalysis]:
        """Perform neural prediction using ByT5 model."""
        if not self._ensure_neural_model():
            return []

        try:
            import torch
            inp = self._tokenizer(f"analyze: {word}", return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = self._model.generate(**inp, max_length=128, num_beams=2)
            pred_text = self._tokenizer.decode(out[0], skip_special_tokens=True)
            
            parsed = parse_neural_output(pred_text)
            if parsed is None:
                return []   # malformed output: no answer rather than a partial one
            root_cons = parsed["root_consonants"]
            root_tuple = tuple(root_cons)
            root_str = "-".join(root_cons)

            # Lemma synthesis
            lemma = word
            if len(root_tuple) == 3:
                try:
                    lemma = self.engine.generate(root_tuple, self.lemma_spec)
                except Exception:
                    pass

            analysis = MorphologicalAnalysis(
                surface_form=word,
                root_consonants=root_tuple,
                root_string=root_str,
                canonical_lemma=lemma,
                tense=TENSE_ALIASES.get(parsed["tense"], parsed["tense"]),
                person=parsed["person"],
                number=parsed["number"],
                gender=parsed["gender"],
                stem=parsed["stem"],
                verb_type="neural_prediction",
                pattern_spec="neural:byt5_fallback",
                lexicon_entry=None,
                object=parsed["object"],
                pos=None if parsed["type"] == "verb" else parsed["type"],
            )
            return [analysis]
        except Exception:
            return []

    def analyze(self, word: str, lookup_dictionary: bool = True, use_hybrid: bool = True) -> List[MorphologicalAnalysis]:
        """
        Analyze a Ge'ez surface word using hybrid multi-tiered resolution:
        1. Fast-path deterministic rule stripping + dictionary lookup (100% exact).
        2. Interdigitation pattern matching (patterns.json, patterns_quadriliteral.json,
           checked class paradigms).
        3. Neural ByT5 sequence-to-sequence model fallback for unseen roots.
        """
        if not use_hybrid:
            # Pattern-only behavior
            return super().analyze(word, lookup_dictionary=lookup_dictionary, use_hybrid=False)

        # Guttural 1st/4th order are spelling variants (ዐ/ዓ, ሐ/ሓ, ኀ/ኃ, ሀ/ሃ, አ/ኣ):
        # the word as written first, then its 1st-order spelling, then templates that
        # accept either order on a guttural radical.
        word = word.strip().strip("፡።፣፤፥፦")
        results = self._rules_analyze(word, lookup_dictionary)
        norm = geez_config.normalize_guttural_4th(word)
        if norm != word:
            results = self._merge(results, self._rules_analyze(norm, lookup_dictionary))
        if any(c in geez_config.GUTTURAL_SYMBOLS for c, _ in geez_config.decompose_string(norm)):
            self._loose_gutturals = True
            try:
                loose = self._pattern_analyze(norm, self.compiled_patterns, lookup_dictionary)
            finally:
                self._loose_gutturals = False
            results = self._merge(results, loose)
        if results:
            return results

        # 3. Neural fallback
        neural_res = self._neural_fallback(word)
        if neural_res:
            return neural_res

        return []

    @staticmethod
    def _merge(first, more):
        def sig(a):
            return (a.root_string, a.tense, a.person, a.number, a.gender, a.stem,
                    getattr(a, "object", None), getattr(a, "negative", None))
        seen = {sig(a) for a in first}
        out = list(first)
        for a in more:
            if sig(a) not in seen:
                seen.add(sig(a))
                out.append(a)
        return out

    def _rules_analyze(self, word: str, lookup_dictionary: bool = True) -> List[MorphologicalAnalysis]:
        """Licensed class paradigms, then the fast path, then templates (one spelling)."""
        if not word:
            return []
        # 0. Class paradigms licensed by the lexicon
        licensed = self.licensed_class_analyze(word, lookup_dictionary)
        if licensed:
            return licensed

        # 1. Fast path
        fast_res = self.fast_path_analyze(word)
        if fast_res:
            return self._prefer_featured(word, fast_res, lookup_dictionary)

        # 2. Pattern matching
        return super().analyze(word, lookup_dictionary=lookup_dictionary, use_hybrid=False)
