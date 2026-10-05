#!/usr/bin/env python3
"""
analyzer.py
Morphological analyzer, text lemmatizer, and lexicon lookup for Classical Ge'ez.
Integrates the root-and-pattern engine with the curated lexicographical corpus.

HybridAnalyzer path: affix-hypothesis + dictionary confirmation, with pattern fallback.
"""

import os
import re
import json
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any, Set

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


# ---------------------------------------------------------------------------
# Affix inventories for hybrid hypothesis generation (not blind stripping)
# ---------------------------------------------------------------------------

# Longest-first. Derivational / agreement / proclitic prefixes.
HYBRID_PREFIXES: Tuple[str, ...] = (
    "ይት",   # imperfect passive
    "አስ",   # causative-factitive
    "ያ",    # imperfect 3ms variant
    "ይ",    # imperfect 3ms/3mp
    "ት",    # imperfect 3fs / 2ms
    "ተ",    # passive / reflexive
    "አ",    # causative
    "ኢ",    # negative
    "እ",    # imperfect 1cs
    "ን",    # imperfect 1cp
    "ወ",    # conjunction proclitic
)

# Character suffixes (subject agreement and object pronouns).
# Object "him" often fuses as a vowel on the last radical (see _detect_fused_object);
# literal -ዎ is kept for forms where it appears as its own character.
HYBRID_SUFFIXES: Tuple[str, ...] = (
    "ክሙ",   # 2mp subject
    "ክን",   # 2fp subject
    "ሆሙ",   # 3mp object
    "ት",    # 3fs subject
    "ኩ",    # 1cs subject
    "ከ",    # 2ms subject
    "ኪ",    # 2fs subject
    "ነ",    # 1cp subject
    "ዎ",    # 3mp subject / object (when separate)
    "ያ",    # 3fp subject
    "ኒ",    # 1cs object
    "ሁ",    # 3ms object
    "ሃ",    # 3fs object
)

# Feature hints from prefixes / suffixes (best-effort; dictionary is primary)
_PREFIX_FEATURES: Dict[str, Dict[str, str]] = {
    "ይት": {"tense": "present", "stem": "passive", "person": "3", "gender": "m", "number": "s"},
    "አስ": {"stem": "causative_as"},
    "ያ": {"tense": "present", "person": "3", "gender": "m", "number": "s"},
    "ይ": {"tense": "present", "person": "3", "gender": "m"},
    "ት": {"tense": "present"},
    "ተ": {"stem": "passive"},
    "አ": {"stem": "causative"},
    "ኢ": {"stem": "negative"},
    "እ": {"tense": "present", "person": "1", "gender": "c", "number": "s"},
    "ን": {"tense": "present", "person": "1", "gender": "c", "number": "p"},
    "ወ": {},  # proclitic conjunction
}

_SUFFIX_FEATURES: Dict[str, Dict[str, str]] = {
    "ክሙ": {"person": "2", "gender": "m", "number": "p", "tense": "past"},
    "ክን": {"person": "2", "gender": "f", "number": "p", "tense": "past"},
    "ሆሙ": {"object": "3mp"},
    "ት": {"person": "3", "gender": "f", "number": "s", "tense": "past"},
    "ኩ": {"person": "1", "gender": "c", "number": "s", "tense": "past"},
    "ከ": {"person": "2", "gender": "m", "number": "s", "tense": "past"},
    "ኪ": {"person": "2", "gender": "f", "number": "s", "tense": "past"},
    "ነ": {"person": "1", "gender": "c", "number": "p", "tense": "past"},
    "ዎ": {"person": "3", "gender": "m", "number": "p"},
    "ያ": {"person": "3", "gender": "f", "number": "p"},
    "ኒ": {"object": "1cs"},
    "ሁ": {"object": "3ms"},
    "ሃ": {"object": "3fs"},
}


def normalize_inferred_root(root_str: str) -> Optional[Tuple[str, ...]]:
    """
    Normalize dictionary inferred_root (Latin, Ethiopic, or mixed) to a
    tuple of abstract consonant symbols, e.g. ('q', 't', 'l').
    """
    if not root_str:
        return None
    main = root_str.split("(")[0].strip()
    main = main.replace("–", "-").replace("—", "-")
    # Drop trailing punctuation / dots often seen in OCR headwords carried into roots
    main = main.strip(" .,;:،")
    if not main:
        return None

    parts = [p.strip() for p in main.split("-") if p.strip()]
    cons: List[str] = []

    def _eth_cons(chunk: str) -> List[str]:
        return [c for c, v in geez_config.decompose_string(chunk) if c]

    if len(parts) == 1:
        chunk = parts[0]
        if any(0x1200 <= ord(ch) <= 0x137F for ch in chunk):
            cons = _eth_cons(chunk)
        else:
            # Single Latin token: either one radical ("a") or unknown concatenation
            token = chunk.lower().strip()
            if token in geez_config.CONSONANTS:
                cons = [token]
            elif re.fullmatch(r"[a-z0-9]+", token) and len(token) <= 4:
                # Cautious: treat each char as radical only for short pure-alpha roots
                # Prefer known multi-char symbols first (h1, s2, ...) — skip naive split
                cons = [token] if token in geez_config.CONSONANTS else list(token)
            else:
                cons = [token] if token else []
    else:
        for p in parts:
            if any(0x1200 <= ord(ch) <= 0x137F for ch in p):
                cons.extend(_eth_cons(p))
            else:
                token = p.lower().strip()
                # Map common scholarly glyphs to config symbols when needed
                glyph_map = {"ḥ": "h2", "ḫ": "kh", "ś": "s2", "ṣ": "S", "ḍ": "S2", "ṭ": "T"}
                if token in glyph_map:
                    cons.append(glyph_map[token])
                else:
                    cons.append(token)

    # Keep only symbols that look like consonants (filter empties)
    cons = [c for c in cons if c]
    return tuple(cons) if cons else None


class GeezMorphologicalAnalyzer:
    """
    Inverse morphological analyzer and text lemmatizer for Classical Ge'ez.

    Hybrid path (use_hybrid=True, default): hypothesize prefix/suffix splits,
    extract stem consonants, confirm against the inferred_root dictionary index,
    and return every high-confidence dictionary-backed candidate. Falls back to
    compiled pattern matching when no high-confidence hybrid hit exists.
    """

    def __init__(self, patterns_path: Optional[str] = None, dictionary_path: Optional[str] = None):
        base_dir = os.path.dirname(os.path.abspath(__file__))

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
                "expanded_slots": expanded_slots,
            })

        self.engine = InterdigitationEngine(renderer=geez_config.render_unit)
        self.lemma_spec = slots_from_spec("C0a-C1a-C2a")
        self.lemma_spec_biliteral = slots_from_spec("C0a-C1a")

        self.lexicon: Dict[str, List[Dict[str, Any]]] = {}
        # Latin-concat key (legacy): "qtl" -> entries
        self.root_index: Dict[str, List[Dict[str, Any]]] = {}
        # Normalized tuple key for hybrid: ("q","t","l") -> entries
        self.inferred_root_index: Dict[Tuple[str, ...], List[Dict[str, Any]]] = {}
        # High-confidence subset of inferred_root_index
        self.inferred_root_index_high: Dict[Tuple[str, ...], List[Dict[str, Any]]] = {}

        if dictionary_path is None:
            candidates = [
                os.path.abspath(os.path.join(base_dir, "..", "geez-dataset-v1", "dictionary.jsonl")),
                os.path.abspath(os.path.join(base_dir, "..", "geez-dataset-v1", "data", "geez_amharic_dictionary.jsonl")),
                os.path.abspath(os.path.join(base_dir, "..", "geez-dataset-v1", "data", "lexicon", "classical_dictionary_kidane_wold.jsonl")),
            ]
            for c in candidates:
                if os.path.exists(c):
                    dictionary_path = c
                    break

        self.dictionary_path = dictionary_path
        if dictionary_path and os.path.exists(dictionary_path):
            self._load_dictionary(dictionary_path)

        # Precompiled affix lists (longest first already)
        self._prefixes = HYBRID_PREFIXES
        self._suffixes = HYBRID_SUFFIXES

    def _load_dictionary(self, dictionary_path: str) -> None:
        with open(dictionary_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                hw = entry.get("headword_geez")
                if hw:
                    self.lexicon.setdefault(hw, []).append(entry)
                    # Also index 1st-order lemma if headword is 6th-order (e.g. ቀተል -> ቀተለ)
                    if len(hw) >= 3 and any(
                        hw.endswith(c) for c in ["ል", "ር", "ም", "ስ", "ቅ", "ብ", "ት", "ክ", "ድ", "ን", "ይ", "ው"]
                    ):
                        units = geez_config.decompose_string(hw)
                        if units and units[-1][1] == "I":
                            alt_chars = list(hw)
                            alt_chars[-1] = geez_config.render_unit(units[-1][0], "a")
                            alt_hw = "".join(alt_chars)
                            self.lexicon.setdefault(alt_hw, []).append(entry)

                root = entry.get("inferred_root") or entry.get("root")
                if root:
                    cleaned_root = root.split("(")[0].strip().split()[0].replace("-", "")
                    if cleaned_root:
                        self.root_index.setdefault(cleaned_root, []).append(entry)

                    norm = normalize_inferred_root(str(root))
                    if norm:
                        self.inferred_root_index.setdefault(norm, []).append(entry)
                        if entry.get("root_confidence") == "high":
                            self.inferred_root_index_high.setdefault(norm, []).append(entry)

    # ------------------------------------------------------------------
    # Pattern matching (existing path)
    # ------------------------------------------------------------------

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

        if max_idx == 2 and 1 not in root_map and compiled_p["features"].get("verb_type") == "weak_middle":
            spec = compiled_p.get("pattern_spec", "")
            if any(x in spec for x in ("C0o", "C0u", "[ው]")):
                root_map[1] = "w"
            else:
                root_map[1] = "y"

        if len(root_map) != max_idx + 1:
            return None

        return tuple(root_map[i] for i in range(max_idx + 1))

    def _pattern_analyze(self, word: str, lookup_dictionary: bool = True) -> List[MorphologicalAnalysis]:
        results: List[MorphologicalAnalysis] = []
        seen: Set[tuple] = set()

        for p in self.compiled_patterns:
            root_cons = self._match_compiled_pattern(word, p)
            if not root_cons:
                continue
            feat = p["features"]

            if len(root_cons) == 2 and feat.get("verb_type") == "weak_middle":
                spec = p.get("pattern_spec", "")
                if any(x in spec for x in ("C0o", "C0u", "[ው]")):
                    root_cons = (root_cons[0], "w", root_cons[1])
                else:
                    root_cons = (root_cons[0], "y", root_cons[1])

            full_root = root_cons
            root_str = "-".join(full_root)
            canonical_lemma = self._synthesize_lemma(full_root, word)

            sig = (
                root_str,
                feat.get("tense"),
                feat.get("person"),
                feat.get("number"),
                feat.get("gender"),
                feat.get("stem"),
            )
            if sig in seen:
                continue
            seen.add(sig)

            lex_entries = None
            if lookup_dictionary and self.lexicon:
                lex_entries = self.lookup_lexicon(canonical_lemma) or self.lookup_lexicon(word)
                if not lex_entries:
                    lex_entries = self.root_index.get("".join(full_root))
                    if not lex_entries:
                        lex_entries = self.inferred_root_index.get(full_root)

            results.append(MorphologicalAnalysis(
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
                lexicon_entry=lex_entries[0] if lex_entries else None,
            ))

        return results

    def _synthesize_lemma(self, full_root: Tuple[str, ...], fallback: str) -> str:
        try:
            if len(full_root) == 3:
                return self.engine.generate(full_root, self.lemma_spec)
            if len(full_root) == 2:
                return self.engine.generate(full_root, self.lemma_spec_biliteral)
            return fallback
        except (ValueError, IndexError, KeyError, Exception):
            lex = self.inferred_root_index.get(full_root) or self.root_index.get("".join(full_root))
            if lex and lex[0].get("headword_geez"):
                return lex[0]["headword_geez"]
            return fallback

    # ------------------------------------------------------------------
    # Hybrid path
    # ------------------------------------------------------------------

    @staticmethod
    def _stem_consonants(stem: str) -> List[str]:
        units = geez_config.decompose_string(stem)
        return [c for c, v in units if c]

    @staticmethod
    def _detect_fused_object(stem: str) -> Optional[str]:
        """
        Object 'him/her' often fuses into the last radical's vowel
        (ቀተለ → ቀተሎ, ይቀትል → ይቀትሎ) rather than a separate -ዎ character.
        """
        units = geez_config.decompose_string(stem)
        if not units or units[-1][0] is None:
            return None
        _c, v = units[-1]
        if v == "o":
            return "3ms"
        if v == "A":
            return "3fs"
        return None

    def _root_hypotheses_from_stem(self, stem: str) -> List[Tuple[str, ...]]:
        """
        From a candidate stem, propose root tuples.
        Strong: 3 consonants. Hollow: 2 consonants → try medial w and y.
        Quadriliteral: 4 consonants as-is.
        """
        cons = self._stem_consonants(stem)
        hyps: List[Tuple[str, ...]] = []
        n = len(cons)
        if n == 3:
            hyps.append(tuple(cons))
        elif n == 2:
            # Hollow / weak-middle: ቆመ → q-?-m ; try both glides
            hyps.append((cons[0], "w", cons[1]))
            hyps.append((cons[0], "y", cons[1]))
            # Also keep biliteral hypothesis
            hyps.append(tuple(cons))
        elif n == 4:
            hyps.append(tuple(cons))
        elif n == 1:
            hyps.append(tuple(cons))
        return hyps

    def _iter_affix_splits(self, word: str):
        """
        Yield (prefixes, stem, suffixes) hypotheses.
        Never commit to a blind strip: every combination is a candidate;
        dictionary confirmation decides which survive.
        """
        # Prefix combinations: empty, or one/two stacked prefixes from the inventory.
        # Stacking covers ወ+ይ, ወ+ተ, ይት, etc. without eating radicals blindly.
        prefix_opts: List[Tuple[str, ...]] = [()]
        for p in self._prefixes:
            if word.startswith(p) and len(word) > len(p) + 1:
                prefix_opts.append((p,))
        # Two-prefix stacks (outer then inner), e.g. ወ+ይ, ወ+ተ, ወ+አ
        for outer in ("ወ",):
            if not word.startswith(outer):
                continue
            rest = word[len(outer):]
            for inner in self._prefixes:
                if inner == outer:
                    continue
                if rest.startswith(inner) and len(rest) > len(inner) + 1:
                    prefix_opts.append((outer, inner))

        seen_splits: Set[Tuple[Tuple[str, ...], str, Tuple[str, ...]]] = set()

        for prefixes in prefix_opts:
            pref_len = sum(len(p) for p in prefixes)
            after_pref = word[pref_len:]
            if len(after_pref) < 2:
                continue

            suffix_opts: List[Tuple[str, ...]] = [()]
            for s in self._suffixes:
                if after_pref.endswith(s) and len(after_pref) > len(s) + 1:
                    suffix_opts.append((s,))
            # Object after subject: e.g. -ኩ+-ሁ, -ከ+-ሃ (subject then object)
            for subj in ("ኩ", "ከ", "ኪ", "ነ", "ት", "ክሙ", "ክን", "ዎ", "ያ"):
                for obj in ("ሁ", "ሃ", "ኒ", "ሆሙ"):
                    combo = subj + obj
                    if after_pref.endswith(combo) and len(after_pref) > len(combo) + 1:
                        suffix_opts.append((subj, obj))

            for suffixes in suffix_opts:
                suf_len = sum(len(s) for s in suffixes)
                stem = after_pref[: len(after_pref) - suf_len] if suf_len else after_pref
                if len(stem) < 2:
                    continue
                key = (prefixes, stem, suffixes)
                if key in seen_splits:
                    continue
                seen_splits.add(key)
                yield prefixes, stem, suffixes

    def _features_from_affixes(
        self,
        prefixes: Tuple[str, ...],
        suffixes: Tuple[str, ...],
        fused_obj: Optional[str],
        stem: str,
    ) -> Dict[str, Optional[str]]:
        feat: Dict[str, Optional[str]] = {
            "tense": None,
            "person": None,
            "number": None,
            "gender": None,
            "stem": "basic",
            "verb_type": "strong",
            "object": None,
        }
        for p in prefixes:
            for k, v in _PREFIX_FEATURES.get(p, {}).items():
                feat[k] = v
        for s in suffixes:
            for k, v in _SUFFIX_FEATURES.get(s, {}).items():
                feat[k] = v
        if fused_obj and not feat.get("object"):
            feat["object"] = fused_obj
        # Hollow cue from stem shape
        cons = self._stem_consonants(stem)
        if len(cons) == 2:
            feat["verb_type"] = "weak_middle"
        return feat

    def _lemma_from_entries(self, root: Tuple[str, ...], entries: List[Dict[str, Any]], fallback: str) -> str:
        # Prefer a verb headword, then any headword, then synthesized lemma
        verbs = [e for e in entries if (e.get("pos") or "").lower() == "verb" and e.get("headword_geez")]
        if verbs:
            return verbs[0]["headword_geez"]
        for e in entries:
            if e.get("headword_geez"):
                return e["headword_geez"]
        return self._synthesize_lemma(root, fallback)

    def hybrid_analyze(self, word: str, lookup_dictionary: bool = True) -> List[MorphologicalAnalysis]:
        """
        Fast hybrid path: hypothesize affix splits, extract consonants, confirm
        against high-confidence inferred_root index. Returns ALL matching
        MorphologicalAnalysis candidates (dictionary hit ≠ unique parse).
        """
        word = word.strip().strip("፡።፣፤፥፦")
        if not word or not self.inferred_root_index_high:
            return []

        results: List[MorphologicalAnalysis] = []
        seen_sigs: Set[tuple] = set()

        for prefixes, stem, suffixes in self._iter_affix_splits(word):
            fused_obj = self._detect_fused_object(stem)
            feat = self._features_from_affixes(prefixes, suffixes, fused_obj, stem)

            for root in self._root_hypotheses_from_stem(stem):
                entries = self.inferred_root_index_high.get(root)
                if not entries:
                    continue

                root_str = "-".join(root)
                lemma = self._lemma_from_entries(root, entries, word)
                affix_tag = f"pref[{'+'.join(prefixes) or '∅'}]+suf[{'+'.join(suffixes) or '∅'}]"
                if fused_obj:
                    affix_tag += f"+fused_obj[{fused_obj}]"
                pattern_spec = f"hybrid:{affix_tag}|root:{root_str}"

                # One analysis per distinct lexicon entry (and feature signature)
                for entry in entries:
                    if not lookup_dictionary:
                        entry_out = None
                    else:
                        entry_out = entry
                    sig = (
                        root_str,
                        feat.get("tense"),
                        feat.get("person"),
                        feat.get("number"),
                        feat.get("gender"),
                        feat.get("stem"),
                        entry.get("id") if entry else None,
                        entry.get("headword_geez") if entry else None,
                    )
                    if sig in seen_sigs:
                        continue
                    seen_sigs.add(sig)

                    results.append(MorphologicalAnalysis(
                        surface_form=word,
                        root_consonants=root,
                        root_string=root_str,
                        canonical_lemma=entry.get("headword_geez") or lemma,
                        tense=feat.get("tense"),
                        person=feat.get("person"),
                        number=feat.get("number"),
                        gender=feat.get("gender"),
                        stem=feat.get("stem"),
                        verb_type=feat.get("verb_type"),
                        pattern_spec=pattern_spec,
                        lexicon_entry=entry_out,
                    ))

        return results

    def analyze(
        self,
        word: str,
        lookup_dictionary: bool = True,
        use_hybrid: bool = True,
    ) -> List[MorphologicalAnalysis]:
        """
        Analyze a single Ge'ez surface word into candidate roots and features.

        use_hybrid=True (default): try dictionary-confirmed affix hypotheses first;
        if any high-confidence hybrid candidates exist, return them all.
        Otherwise (or if use_hybrid=False) fall back to pattern-based analysis.
        """
        word = word.strip().strip("፡።፣፤፥፦")
        if use_hybrid:
            hybrid_hits = self.hybrid_analyze(word, lookup_dictionary=lookup_dictionary)
            if hybrid_hits:
                return hybrid_hits
        return self._pattern_analyze(word, lookup_dictionary=lookup_dictionary)

    def lemmatize_text(self, text: str) -> List[LemmatizedToken]:
        """
        Tokenize Ge'ez text by Ethiopic word dividers, strip common proclitics, and analyze each token.
        """
        raw_tokens = [
            t.strip()
            for t in text.replace("::", " ").replace("፡", " ").replace("።", " ").split()
            if t.strip()
        ]
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
                analyses=analyses,
            ))

        return token_results

    def lookup_lexicon(self, word_or_lemma: str) -> List[Dict[str, Any]]:
        """Direct lookup in Kidane Wold Kifle lexicon."""
        results = self.lexicon.get(word_or_lemma, [])
        if not results:
            units = geez_config.decompose_string(word_or_lemma)
            cons = tuple(c for c, v in units if c is not None)
            if cons in self.inferred_root_index:
                results = self.inferred_root_index[cons]
            else:
                joined = "".join(cons)
                if joined in self.root_index:
                    results = self.root_index[joined]
        return results


# Public alias: HybridAnalyzer is the hybrid-capable morphological analyzer.
HybridAnalyzer = GeezMorphologicalAnalyzer
