"""
analyzer.py
Classical Ge'ez Morphological Analyzer and Lemmatizer.
Integrates two-level Semitic interdigitation with dictionary grounding.
"""

import os
import sys
import json
import re
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import geez_config
from .engine import InterdigitationEngine, slots_from_spec, Pattern


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


@dataclass
class LemmatizedToken:
    token: str
    cleaned_token: str
    proclitics: List[str]
    analyses: List[MorphologicalAnalysis]


class GeezMorphologicalAnalyzer:
    """
    Morphological analyzer and lemmatizer for Classical Ge'ez.
    """

    def __init__(self, patterns_path: Optional[str] = None, dictionary_path: Optional[str] = None):
        base_dir = parent_dir

        if patterns_path is None:
            patterns_path = os.path.join(base_dir, "patterns.json")
        with open(patterns_path, "r", encoding="utf-8") as f:
            self.patterns_data = json.load(f)

        self.compiled_patterns = []
        for p in self.patterns_data:
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
                "expanded_slots": expanded_slots
            })

        self.engine = InterdigitationEngine(renderer=geez_config.render_unit)
        self.lemma_spec = slots_from_spec("C0a-C1a-C2a")
        self.lemma_spec_biliteral = slots_from_spec("C0a-C1a")

        self.lexicon = {}
        self.root_index = {}

        if dictionary_path is None:
            # Check canonical dataset location first, then legacy
            candidates = [
                os.path.abspath(os.path.join(base_dir, "..", "geez-dataset-v1", "data", "geez_amharic_dictionary.jsonl")),
                os.path.abspath(os.path.join(base_dir, "..", "geez-dataset-v1", "data", "lexicon", "classical_dictionary_kidane_wold.jsonl")),
                os.path.abspath(os.path.join(base_dir, "..", "geez-dataset-v1", "dictionary.jsonl")),
            ]
            for c in candidates:
                if os.path.exists(c):
                    dictionary_path = c
                    break

        if dictionary_path and os.path.exists(dictionary_path):
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

                    root = entry.get("inferred_root") or entry.get("root")
                    if root:
                        cleaned_root = root.split("(")[0].strip().replace("-", "")
                        self.root_index.setdefault(cleaned_root, []).append(entry)

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
                if ch != slot[1]:
                    return None
            elif slot[0] == "R":
                slot_idx = slot[1]
                slot_v = slot[2]
                if v != slot_v:
                    return None
                if slot_idx in root_map and root_map[slot_idx] != c:
                    return None
                root_map[slot_idx] = c

        slots = compiled_p["slots"]
        max_idx = max(s[1] for s in slots if s[0] == "R")

        # Recover elided medial glide for weak_middle patterns (e.g. ቆመ -> q-w-m, ሤመ -> s-y-m)
        if max_idx == 2 and 1 not in root_map and compiled_p["features"].get("verb_type") == "weak_middle":
            spec = compiled_p.get("pattern_spec", "")
            if any(x in spec for x in ("C0o", "C0u", "[ው]")):
                root_map[1] = "w"
            else:
                root_map[1] = "y"

        if len(root_map) != max_idx + 1:
            return None

        return tuple(root_map[i] for i in range(max_idx + 1))

    def analyze(self, word: str, lookup_dictionary: bool = True) -> List[MorphologicalAnalysis]:
        word = word.strip().strip("፡።፣፤፥፦")
        results = []
        seen = set()

        for p in self.compiled_patterns:
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
                    # Multi-tiered fallback: check dictionary first, then surface form
                    lex = self.root_index.get("".join(full_root))
                    if lex and lex[0].get("headword_geez"):
                        canonical_lemma = lex[0]["headword_geez"]
                    else:
                        canonical_lemma = word

                sig = (root_str, feat.get("tense"), feat.get("person"), feat.get("number"), feat.get("gender"), feat.get("stem"))
                if sig in seen:
                    continue
                seen.add(sig)

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
                    tense=feat.get("tense"),
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
        raw_tokens = [t.strip() for t in text.replace("::", " ").replace("፡", " ").replace("።", " ").split() if t.strip()]
        token_results = []
        proclitic_prefixes = ["ወ", "በ", "ለ", "ከ", "እም"]

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
        results = self.lexicon.get(word_or_lemma, [])
        if not results:
            units = geez_config.decompose_string(word_or_lemma)
            cons = "".join(c for c, v in units if c is not None)
            if cons in self.root_index:
                results = self.root_index[cons]
        return results
