#!/usr/bin/env python3
"""
analyzer.py
Morphological analyzer, text lemmatizer, and lexicon lookup for Classical Ge'ez.
Integrates the root-and-pattern engine with the curated lexicographical corpus.
"""

import os
import json
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any

import geez_config
from engine import InterdigitationEngine, slots_from_spec, Pattern


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
    Inverse morphological analyzer and text lemmatizer for Classical Ge'ez.
    """

    def __init__(self, patterns_path: Optional[str] = None, dictionary_path: Optional[str] = None):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Load patterns
        if patterns_path is None:
            patterns_path = os.path.join(base_dir, "patterns.json")
        with open(patterns_path, "r", encoding="utf-8") as f:
            self.patterns_data = json.load(f)

        # Precompile pattern slots
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

        # Initialize generation engine for lemma synthesis
        self.engine = InterdigitationEngine(renderer=geez_config.render_unit)
        self.lemma_spec = slots_from_spec("C0a-C1a-C2a")
        self.lemma_spec_biliteral = slots_from_spec("C0a-C1a")

        # Load dictionary index
        self.lexicon = {}
        self.root_index = {}
        
        if dictionary_path is None:
            candidate = os.path.abspath(os.path.join(base_dir, "..", "geez-dataset-v1", "dictionary.jsonl"))
            if os.path.exists(candidate):
                dictionary_path = candidate

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

                    # Index by consonantal root
                    root = entry.get("inferred_root") or entry.get("root")
                    if root:
                        cleaned_root = root.split()[0].replace("-", "")
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
        if len(root_map) != max_idx + 1:
            return None

        return tuple(root_map[i] for i in range(max_idx + 1))

    def analyze(self, word: str, lookup_dictionary: bool = True) -> List[MorphologicalAnalysis]:
        """
        Analyze a single Ge'ez surface word into candidate roots and morphological features.
        """
        word = word.strip().strip("፡።፣፤፥፦")
        results = []
        seen = set()

        for p in self.compiled_patterns:
            root_cons = self._match_compiled_pattern(word, p)
            if root_cons:
                feat = p["features"]
                root_str = "-".join(root_cons)
                
                # For weak_middle, reconstruct full triliteral root with medial glide
                if len(root_cons) == 2 and feat.get("verb_type") == "weak_middle":
                    spec = p.get("pattern_spec", "")
                    if any(x in spec for x in ("C0o", "C0u", "[ው]")):
                        root_cons = (root_cons[0], "w", root_cons[1])
                    else:
                        root_cons = (root_cons[0], "y", root_cons[1])
                full_root = root_cons
                root_str = "-".join(full_root)
                try:
                    if len(full_root) == 3:
                        canonical_lemma = self.engine.generate(full_root, self.lemma_spec)
                    elif len(full_root) == 2:
                        canonical_lemma = self.engine.generate(full_root, self.lemma_spec_biliteral)
                    else:
                        canonical_lemma = word
                except (ValueError, IndexError):
                    lex = self.root_index.get("".join(full_root))
                    if lex and lex[0].get("headword_geez"):
                        canonical_lemma = lex[0]["headword_geez"]
                    else:
                        canonical_lemma = word

                sig = (root_str, feat.get("tense"), feat.get("person"), feat.get("number"), feat.get("gender"), feat.get("stem"))
                if sig in seen:
                    continue
                seen.add(sig)

                # Dictionary lookup
                lex_entries = None
                if lookup_dictionary and self.lexicon:
                    lex_entries = self.lookup_lexicon(canonical_lemma) or self.lookup_lexicon(word)
                    if not lex_entries:
                        lex_entries = self.root_index.get("".join(root_cons))

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
        """
        Tokenize Ge'ez text by Ethiopic word dividers, strip common proclitics, and analyze each token.
        """
        raw_tokens = [t.strip() for t in text.replace("::", " ").replace("፡", " ").replace("።", " ").split() if t.strip()]
        token_results = []
        proclitic_prefixes = ["ወ", "በ", "ለ", "ከ", "እም"]

        for tok in raw_tokens:
            cleaned = tok.strip("፣፤፥፦():[]")
            proclitics = []
            
            # Check for leading conjunction/preposition clitics (e.g. ወይቤሎ -> ወ- + ይቤሎ)
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
            # Check by root
            units = geez_config.decompose_string(word_or_lemma)
            cons = "".join(c for c, v in units if c is not None)
            if cons in self.root_index:
                results = self.root_index[cons]
        return results
