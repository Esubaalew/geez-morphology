"""
geez_morphology
Canonical Classical Ge'ez Computational Morphology Package.
"""

from .engine import InterdigitationEngine, slots_from_spec, spec_from_slots
from .analyzer import GeezMorphologicalAnalyzer, HybridAnalyzer, MorphologicalAnalysis, normalize_inferred_root
