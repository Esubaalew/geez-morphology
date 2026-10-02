"""
General Semitic Root-and-Pattern Interdigitation Engine.

This module implements the UNIVERSAL mechanism of Semitic morphology:
interdigitation of root consonants into pattern templates. It is
language-agnostic. All language-specific knowledge (script rendering,
vowel inventory, affix inventories) lives in a configuration object.

The core operation:
    interdigitate(root, pattern) -> abstract_form
    render(abstract_form, config) -> surface string

A root is an ordered tuple of abstract consonant symbols, e.g. ("q", "t", "l").
A pattern is a list of slots. Each slot is one of:
    ("R", i, v)  -- root consonant i (0-indexed) realized with vowel v
    ("F", text)  -- fixed material (affix or pattern-internal segment)

Example (abstract):
    root    = ("q", "t", "l")
    pattern = [("R", 0, "a"), ("R", 1, "a"), ("R", 2, "a")]
    result  = [("q","a"), ("t","a"), ("l","a")]

Rendering to a specific script is delegated to the language configuration,
which maps (consonant, vowel) pairs to characters. For Ge'ez this uses
Unicode vowel-order arithmetic on the Ethiopic block; for Arabic it would
map to consonant + diacritic sequences.

This separation is the thesis contribution: the interdigitation mechanism
is universal across Semitic languages; only the renderer and the pattern
inventory are language-specific.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Union, Dict, Any, Callable


# ---------------------------------------------------------------------------
# Abstract pattern representation
# ---------------------------------------------------------------------------

# A slot is either a root slot or fixed material.
#   ("R", index, vowel)  -- index into the root, vowel is an abstract label
#   ("F", text)          -- literal fixed string (affix, particle, etc.)
Slot = Tuple[str, ...]
Pattern = List[Slot]

# An abstract form is a list of (consonant, vowel) pairs and fixed strings.
#   ("C", consonant, vowel) -- a consonant+vowel unit
#   ("F", text)             -- fixed material passed through
AbstractUnit = Tuple[str, ...]


def make_root_slot(index: int, vowel: str) -> Slot:
    """A slot filled by root consonant `index` with vowel `vowel`."""
    return ("R", index, vowel)


def make_fixed(text: str) -> Slot:
    """Fixed material (prefix, suffix, infix, pattern-internal consonant)."""
    return ("F", text)


@dataclass
class InterdigitationEngine:
    """
    Language-agnostic root-and-pattern engine.

    The engine knows nothing about any specific language. It manipulates
    abstract consonant symbols and vowel labels. A `renderer` callable
    converts abstract (consonant, vowel) units into script characters.
    """

    renderer: Callable[[str, str], str] = None
    fixed_renderer: Callable[[str], str] = None

    def interdigitate(self, root: Tuple[str, ...], pattern: Pattern) -> List[AbstractUnit]:
        """
        Fill a pattern template with root consonants.

        Args:
            root: ordered consonant symbols, e.g. ("q", "t", "l").
                  May be triliteral, quadriliteral, or any length.
            pattern: list of slots as defined above.

        Returns:
            List of abstract units, e.g.
            [("C","q","a"), ("C","t","a"), ("C","l","a")].

        Raises:
            ValueError: if a root slot indexes beyond the root length.
        """
        out: List[AbstractUnit] = []
        n = len(root)
        for slot in pattern:
            if slot[0] == "R":
                _, index, vowel = slot
                if not (0 <= index < n):
                    raise ValueError(
                        f"Root slot index {index} out of range for "
                        f"root of length {n}: {root}"
                    )
                out.append(("C", root[index], vowel))
            elif slot[0] == "F":
                _, text = slot
                out.append(("F", text))
            else:
                raise ValueError(f"Unknown slot type: {slot[0]}")
        return out

    def render(self, abstract: List[AbstractUnit]) -> str:
        """
        Convert abstract units to a surface string using the configured
        renderer. Fixed material passes through (optionally transformed
        by fixed_renderer).
        """
        if self.renderer is None:
            raise ValueError("No renderer configured for this engine")
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
        """Interdigitate then render: the full generation pipeline."""
        return self.render(self.interdigitate(root, pattern))

    # ------------------------------------------------------------------
    # Analysis (inverse) helpers
    # ------------------------------------------------------------------

    def pattern_signature(self, pattern: Pattern) -> str:
        """
        A canonical string signature for a pattern, for comparison and
        deduplication. Root slots render as C{i}{vowel}, fixed as-is.
        """
        parts = []
        for slot in pattern:
            if slot[0] == "R":
                parts.append(f"C{slot[1]}{slot[2]}")
            else:
                parts.append(f"[{slot[1]}]")
        return "-".join(parts)


# ---------------------------------------------------------------------------
# Pattern utilities (also language-agnostic)
# ---------------------------------------------------------------------------

def slots_from_spec(spec: str) -> Pattern:
    """
    Parse a compact pattern specification string into a Pattern.

    Spec mini-language:
      - "C0a" means root consonant 0 with vowel "a"
      - "C1u" means root consonant 1 with vowel "u"
      - "[text]" means fixed material "text"
      - Segments are separated by "-".

    Example:
        slots_from_spec("C0a-C1a-C2a")
        -> [("R",0,"a"), ("R",1,"a"), ("R",2,"a")]

        slots_from_spec("[ya]-C0a-C1a-C2a")
        -> [("F","ya"), ("R",0,"a"), ("R",1,"a"), ("R",2,"a")]
    """
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
    """Inverse of slots_from_spec: canonical compact string."""
    parts = []
    for slot in pattern:
        if slot[0] == "R":
            parts.append(f"C{slot[1]}{slot[2]}")
        else:
            parts.append(f"[{slot[1]}]")
    return "-".join(parts)
