"""
Pattern learner: induce abstract root-and-pattern templates from data.

This module is LANGUAGE-AGNOSTIC. It operates on:
  - roots as tuples of abstract consonant symbols
  - surface forms as strings
  - a decomposer function: char -> (consonant_symbol, vowel_label) or (None, None)

Given (root, surface_form) pairs, it aligns root consonants to the
surface string and extracts the pattern: which vowel each root consonant
takes, and what fixed material surrounds them.

The induced patterns for Ge'ez are DISCOVERED, not hardcoded. The same
learner works for Arabic or Hebrew given an appropriate decomposer.

Alignment algorithm (sequential, order-preserving):
  1. Decompose surface form into per-character (consonant, vowel) pairs.
  2. Walk the surface left to right, matching root consonants in order.
     A character "matches" root[i] if its decomposed consonant equals
     root[i]'s symbol.
  3. Matched characters become ("R", i, vowel) slots.
  4. Unmatched characters between matches become ("F", text) fixed runs.
  5. If a root consonant cannot be matched (weak verb elision, etc.),
     the alignment fails and the pair is flagged for review.

This is deliberately simple and transparent: every induced pattern can
be traced back to the alignment that produced it.
"""

from typing import List, Tuple, Dict, Optional, Callable
from dataclasses import dataclass, field
import sys
sys.path.insert(0, '/home/hatch/workspace/thesis-data/geez-morphology-model')
from engine import Pattern, make_root_slot, make_fixed, spec_from_slots


# Type aliases
Decomposer = Callable[[str], Tuple[Optional[str], Optional[str]]]
RootSpec = Tuple[str, ...]  # e.g. ("q", "t", "l")


@dataclass
class LearnedPattern:
    """A pattern induced from one or more (root, form) observations."""
    pattern: Pattern
    observations: int = 0
    # features -> example surface forms, for audit
    examples: Dict[str, List[str]] = field(default_factory=dict)

    @property
    def spec(self) -> str:
        return spec_from_slots(self.pattern)


@dataclass
class AlignmentResult:
    success: bool
    pattern: Optional[Pattern] = None
    reason: Optional[str] = None
    # Debug trace
    trace: Optional[List[str]] = None


def align_root_to_form(root: RootSpec,
                       surface: str,
                       decomposer: Decomposer,
                       debug: bool = False) -> AlignmentResult:
    """
    Align root consonants to a surface form string.

    Args:
        root: tuple of abstract consonant symbols, e.g. ("q","t","l")
        surface: surface string, e.g. "ይቀትል"
        decomposer: char -> (consonant|None, vowel|None)

    Returns:
        AlignmentResult with the induced Pattern on success.
    """
    trace = [] if debug else None
    def log(msg):
        if debug:
            trace.append(msg)

    # Decompose surface into units
    units = []  # list of (char, consonant|None, vowel|None)
    for ch in surface:
        c, v = decomposer(ch)
        units.append((ch, c, v))

    log(f"surface={surface!r} units={[(u[1],u[2]) for u in units]}")
    log(f"root={root}")

    pattern: Pattern = []
    fixed_buf = []  # accumulate fixed characters
    ri = 0  # root index pointer
    n = len(root)

    def flush_fixed():
        if fixed_buf:
            pattern.append(make_fixed("".join(fixed_buf)))
            fixed_buf.clear()

    for ch, c, v in units:
        if ri < n and c == root[ri]:
            # This character realizes root consonant ri
            flush_fixed()
            pattern.append(make_root_slot(ri, v))
            log(f"  char {ch!r} -> R{ri} vowel {v}")
            ri += 1
        else:
            # Fixed material (affix, or non-root character)
            fixed_buf.append(ch)
            log(f"  char {ch!r} -> fixed (c={c}, v={v})")

    flush_fixed()

    if ri < n:
        return AlignmentResult(
            success=False,
            reason=(f"Only matched {ri}/{n} root consonants; "
                    f"unmatched: {root[ri:]}. "
                    f"Likely weak-verb elision or irregular form."),
            trace=trace,
        )
    return AlignmentResult(success=True, pattern=pattern, trace=trace)


class PatternLearner:
    """
    Learn pattern templates from (root, surface_form, features) triples.

    Usage:
        learner = PatternLearner(decomposer)
        for root, form, features in training_data:
            learner.observe(root, form, features)
        patterns = learner.patterns_for(features)  # induced templates

    Patterns are keyed by a feature signature (e.g. tense/person/stem),
    so the same morphological category learned from multiple roots can
    be checked for consistency.
    """

    def __init__(self, decomposer: Decomposer):
        self.decomposer = decomposer
        # feature_key -> spec_string -> LearnedPattern
        self._table: Dict[str, Dict[str, LearnedPattern]] = {}
        self.failures: List[Dict] = []

    @staticmethod
    def feature_key(features: Dict) -> str:
        return "|".join(f"{k}={features[k]}" for k in sorted(features))

    def observe(self, root: RootSpec, surface: str, features: Dict):
        """Incorporate one (root, surface, features) observation."""
        key = self.feature_key(features)
        res = align_root_to_form(root, surface, self.decomposer)
        if not res.success:
            self.failures.append({
                "root": root, "surface": surface,
                "features": features, "reason": res.reason,
            })
            return res
        spec = spec_from_slots(res.pattern)
        bucket = self._table.setdefault(key, {})
        if spec not in bucket:
            bucket[spec] = LearnedPattern(pattern=res.pattern)
        lp = bucket[spec]
        lp.observations += 1
        ex_key = self.feature_key(features)
        lp.examples.setdefault(ex_key, []).append(surface)
        return res

    def patterns_for(self, features: Dict) -> List[LearnedPattern]:
        """All induced patterns for a feature combination, most frequent first."""
        key = self.feature_key(features)
        bucket = self._table.get(key, {})
        return sorted(bucket.values(), key=lambda lp: -lp.observations)

    def consensus_pattern(self, features: Dict) -> Optional[LearnedPattern]:
        """The most frequently observed pattern for these features, if any."""
        ps = self.patterns_for(features)
        return ps[0] if ps else None

    def all_features(self) -> List[str]:
        return sorted(self._table.keys())

    def consistency_report(self) -> Dict[str, Dict]:
        """
        For each feature combination, report whether all observations
        induced the SAME pattern (consistent) or multiple patterns
        (inconsistent -> allomorphy, data error, or alignment issue).
        """
        report = {}
        for key, bucket in self._table.items():
            total = sum(lp.observations for lp in bucket.values())
            report[key] = {
                "n_patterns": len(bucket),
                "n_observations": total,
                "consistent": len(bucket) == 1,
                "patterns": [
                    {"spec": lp.spec, "observations": lp.observations,
                     "examples": lp.examples.get(key, [])[:3]}
                    for lp in sorted(bucket.values(),
                                     key=lambda x: -x.observations)
                ],
            }
        return report
