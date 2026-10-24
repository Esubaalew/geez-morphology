"""generate_synthetic_data: engine failures are itemized, not only counted."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import generate_synthetic_data as gen  # noqa: E402


def test_record_and_format_engine_failures():
    failures = []
    gen.record_engine_failure(failures, "triliteral", ("q", "t", "l"), "[ይ]-C0a-C1I-C2I", ValueError("bad slot"))
    gen.record_engine_failure(failures, "quadriliteral", ["d", "n", "g", "s"], "C0a-C1a-C2a-C3a", KeyError("stem"))
    text = gen.format_engine_failures(failures)
    assert "triliteral root q-t-l" in text
    assert "ValueError: bad slot" in text
    assert "quadriliteral root d-n-g-s" in text
    assert "KeyError: 'stem'" in text
    assert "By error type: KeyError=1, ValueError=1" in text


def test_format_engine_failures_caps_and_counts():
    failures = []
    for i in range(25):
        gen.record_engine_failure(failures, "triliteral", ("q", "t", "l"), f"spec-{i}", RuntimeError(f"n={i}"))
    text = gen.format_engine_failures(failures, limit=5)
    assert text.count("[") == 5
    assert "(20 more not shown)" in text
    assert "RuntimeError=25" in text


def test_format_engine_failures_empty():
    assert gen.format_engine_failures([]) == ""
