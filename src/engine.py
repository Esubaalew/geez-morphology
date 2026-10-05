"""
engine.py
Semitic Root-and-Pattern Interdigitation Engine with Classical Ge'ez Phonology.

Implements:
1. Abstract interdigitation of root consonants into morphological pattern slots.
2. Two-level phonological glide contractions for weak roots (e.g. ቆመ, ሤመ, ይቁም, ይሢም).
3. Script rendering via Unicode vowel-order arithmetic.
"""

import os
import sys
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Callable, Optional

# Ensure parent directory is accessible for geez_config
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import geez_config

Slot = Tuple[str, ...]
Pattern = List[Slot]
AbstractUnit = Tuple[str, ...]


def make_root_slot(index: int, vowel: str) -> Slot:
    return ("R", index, vowel)


def make_fixed(text: str) -> Slot:
    return ("F", text)


@dataclass
class InterdigitationEngine:
    """
    Semitic root-and-pattern interdigitation engine with phonological contraction.
    """
    renderer: Callable[[str, str], str] = geez_config.render_unit
    fixed_renderer: Optional[Callable[[str], str]] = None

    def interdigitate(self, root: Tuple[str, ...], pattern: Pattern) -> List[AbstractUnit]:
        out: List[AbstractUnit] = []
        n = len(root)
        for slot in pattern:
            if slot[0] == "R":
                _, index, vowel = slot
                if not (0 <= index < n):
                    raise ValueError(f"Root slot index {index} out of range for root of length {n}: {root}")
                out.append(("C", root[index], vowel))
            elif slot[0] == "F":
                _, text = slot
                out.append(("F", text))
            else:
                raise ValueError(f"Unknown slot type: {slot[0]}")
        return out

    def contract_abstract(self, abstract: List[AbstractUnit]) -> List[AbstractUnit]:
        """
        Applies Classical Ge'ez phonological monophthongization rules for hollow verbs:
        - C0a + w(a) -> C0o  (e.g., q-w-m -> ቆመ, m-w-t -> ሞተ, k-w-n -> ኮነ)
        - C0a + y(a) -> C0e  (e.g., s-y-m -> ሤመ, k-y-d -> ኬደ)
        - C0I + w(I) -> C0u  (e.g., jussive/subjunctive ይ-ቁ-ም, ይ-ኩ-ን, ይ-ሙ-ት)
        - C0I + y(I) -> C0i  (e.g., jussive/subjunctive ይ-ሢ-ም, ይ-ኪ-ድ)
        """
        out = []
        i = 0
        n = len(abstract)
        while i < n:
            u = abstract[i]
            # Medial w contraction in perfect: C0a + w(a) -> C0o
            if (i + 1 < n and u[0] == "C" and u[2] == "a" and 
                abstract[i+1][0] == "C" and abstract[i+1][1] == "w" and abstract[i+1][2] == "a"):
                out.append(("C", u[1], "o"))
                i += 2
                continue
            # Medial y contraction in perfect: C0a + y(a) -> C0e
            if (i + 1 < n and u[0] == "C" and u[2] == "a" and 
                abstract[i+1][0] == "C" and abstract[i+1][1] == "y" and abstract[i+1][2] == "a"):
                out.append(("C", u[1], "e"))
                i += 2
                continue
            # Medial w contraction in jussive/subjunctive: C0I + w(I) -> C0u
            if (i + 1 < n and u[0] == "C" and u[2] in ("I", "a") and 
                abstract[i+1][0] == "C" and abstract[i+1][1] == "w"):
                out.append(("C", u[1], "u"))
                i += 2
                continue
            # Medial y contraction in jussive/subjunctive: C0I + y(I) -> C0i
            if (i + 1 < n and u[0] == "C" and u[2] in ("I", "a") and 
                abstract[i+1][0] == "C" and abstract[i+1][1] == "y"):
                out.append(("C", u[1], "i"))
                i += 2
                continue
            out.append(u)
            i += 1
        return out

    def render(self, abstract: List[AbstractUnit]) -> str:
        if self.renderer is None:
            raise ValueError("No renderer configured")
        parts = []
        for unit in abstract:
            if unit[0] == "C":
                _, consonant, vowel = unit
                parts.append(self.renderer(consonant, vowel))
            elif unit[0] == "F":
                _, text = unit
                if self.fixed_renderer is not None:
                    parts.append(self.fixed_renderer(text))
                else:
                    parts.append(text)
        return "".join(parts)

    def generate(self, root: Tuple[str, ...], pattern: Pattern) -> str:
        abstract = self.interdigitate(root, pattern)
        contracted = self.contract_abstract(abstract)
        return self.render(contracted)


def slots_from_spec(spec: str) -> Pattern:
    import re
    pattern: Pattern = []
    for seg in spec.split("-"):
        seg = seg.strip()
        m = re.fullmatch(r"C(\d+)([A-Za-z_]+)", seg)
        if m:
            pattern.append(("R", int(m.group(1)), m.group(2)))
            continue
        m = re.fullmatch(r"\[(.*)\]", seg, re.DOTALL)
        if m:
            pattern.append(("F", m.group(1)))
            continue
        raise ValueError(f"Cannot parse pattern segment: {seg!r}")
    return pattern


def spec_from_slots(pattern: Pattern) -> str:
    parts = []
    for slot in pattern:
        if slot[0] == "R":
            parts.append(f"C{slot[1]}{slot[2]}")
        else:
            parts.append(f"[{slot[1]}]")
    return "-".join(parts)
