"""scripts/summarize_runs.py: McNemar test and Wilson interval."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import summarize_runs as sr  # noqa: E402


def test_mcnemar_exact():
    assert sr.mcnemar_exact(0, 0) == 1.0
    assert abs(sr.mcnemar_exact(5, 5) - 1.0) < 1e-9
    assert abs(sr.mcnemar_exact(0, 10) - 2 / 1024) < 1e-12


def test_wilson_matches_the_evaluator():
    import evaluate_gold as ev
    assert all(abs(a - b) < 1e-12 for a, b in zip(sr.wilson(37, 90), ev.wilson(37, 90)))
