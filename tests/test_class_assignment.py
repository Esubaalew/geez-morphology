"""Classes read off the written Kidane Wold triple: the written shape types (ጦመረ, ሴሰየ, ማህረከ shapes)."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import geez_script as gc  # noqa: E402
import verb_class_paradigms as vcp  # noqa: E402
from engine import InterdigitationEngine, slots_from_spec  # noqa: E402

ENGINE = InterdigitationEngine(renderer=gc.render_unit)
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def three(cls, lemma):
    t = vcp.CLASSES[cls](list(lemma))
    return t["perfect"]["3ms"], t["imperfect"]["3ms"], t["subjunctive"]["3ms"]


def test_type_c_o_keeps_o():          # ጦመረ ፡ ይጦምር ፡ ይጦምር
    assert three("type_c_o", "ጦመረ") == ("ጦመረ", "ይጦምር", "ይጦምር")


def test_type_c_e_keeps_e():
    assert three("type_c_e", "ሴሰረ") == ("ሴሰረ", "ይሴስር", "ይሴስር")


def test_quad_c_mahraka():            # ማህረከ ፡ ይማህርክ ፡ ይማህርክ
    assert three("quad_c", "ማህረከ") == ("ማህረከ", "ይማህርክ", "ይማህርክ")


def quad(tense, person="3", gender="m", number="s"):
    for t in json.load(open(os.path.join(HERE, "patterns_quadriliteral.json"), encoding="utf-8")):
        f = t["features"]
        if (f["tense"], f["person"], f["gender"], f["number"]) == (tense, person, gender, number):
            return ENGINE.generate(("d", "n", "g", "S2"), slots_from_spec(t["pattern_spec"]))


def test_quad_perfect_is_dangase_not_danagase():   # ደንገፀ, not ደነገፀ
    assert quad("past") == "ደንገፀ"


def test_quad_imperfect_and_subjunctive():          # ይደነግፅ ፡ ይደንግፅ (ተንበለ ፡ ይተነብል ፡ ይተንብል)
    assert quad("present") == "ይደነግፅ"
    assert quad("subjunctive") == "ይደንግፅ"


def test_type_a_a_subjunctive_spec():               # ገብረ -> ይግበር, ግበር
    root = ("g", "b", "r")
    assert ENGINE.generate(root, slots_from_spec("[ይ]-C0I-C1I-C2I".replace("C1I", "C1a"))) == "ይግበር"
    assert ENGINE.generate(root, slots_from_spec("C0I-C1I-C2I".replace("C1I", "C1a"))) == "ግበር"


# ---- OCR gate and class corrections
def test_ocr_gate():
    import kidane_wold_triples as kw
    assert kw.ocr_gate("ቀተለ", "ይቀትል", "ይቅትል") is None
    assert kw.ocr_gate("ቆመ", "ይቀውም", "ይቁም") is None          # w may be missing
    assert kw.ocr_gate("ኀንጽጽ", "የኀንጽጽ", "የኀንጽጽ") is not None   # perfect not 1st order at the end
    assert kw.ocr_gate("ኀትመ", "የኀትስም", "ይኀትስም") is not None    # radicals differ


def test_type_d_from_triple():                       # ጽሕቀ ፡ ይጽሕቅ ፡ ይጽሐቅ
    t = vcp.type_d(list("ጽሕቀ"), {"perfect": (6, 6), "imperfect": (6, 6), "subjunctive": (6, 1)})
    assert (t["perfect"]["3ms"], t["imperfect"]["3ms"], t["subjunctive"]["3ms"]) == ("ጽሕቀ", "ይጽሕቅ", "ይጽሐቅ")


def test_guttural_present_prefix():                  # የኀፍር, ተኀፍር, አኀፍር, ነኀፍር; subjunctive keeps ይ
    from generate_synthetic_data import guttural_prefix
    root = ("kh", "f", "r")
    assert ENGINE.generate(root, slots_from_spec(guttural_prefix("[ይ]-C0a-C1I-C2I"))) == "የኀፍር"
    assert ENGINE.generate(root, slots_from_spec(guttural_prefix("[ን]-C0a-C1I-C2I"))) == "ነኀፍር"
    assert guttural_prefix("C0a-C1a-C2a") == "C0a-C1a-C2a"
