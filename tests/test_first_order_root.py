"""geez_script.first_order_root."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import geez_script as gc


def test_first_order_examples():
    assert gc.first_order_root("ን-ስ-ሕ-ለ-ል") == "ነ-ሰ-ሐ-ለ-ለ"
    assert gc.first_order_root("ር-እ-የ") == "ረ-አ-የ"
    assert gc.first_order_root("ጼ-ው-ው") == "ጸ-ወ-ወ"
    assert gc.first_order_root("ሦ-ይ-ም") == "ሠ-የ-መ"


def test_labialized_stay_labialized():
    assert gc.first_order_root("ቍ-ር-ስ") == "ቈ-ረ-ሰ"
    assert gc.first_order_root("ኵ-ን-ተ-ተ") == "ኰ-ነ-ተ-ተ"
    assert gc.first_order_root("ጸ-ገ-ጒ") == "ጸ-ገ-ጐ"
    assert gc.first_order_root("ኍ-ለ-ቈ") == "ኈ-ለ-ቈ"


def test_seventh_order_taken_as_written():
    assert gc.first_order_root("ቆ-ም-ሐ") == "ቀ-መ-ሐ"
    assert gc.first_order_root("ጎ-ን-ጸ-ጸ") == "ገ-ነ-ጸ-ጸ"


def test_idempotent_latin_and_unknown():
    assert gc.first_order_root("ቀ-ተ-ለ") == "ቀ-ተ-ለ"
    assert gc.first_order_root(gc.first_order_root("ዐ-ጽ-ም")) == "ዐ-ጸ-መ"
    assert gc.first_order_root("q-t-l") == "ቀ-ተ-ለ"
    assert gc.first_order_root("በ-ጘ-በ-ዘ") == "በ-ጘ-በ-ዘ"     # not in the table: kept
    assert gc.first_order_root("") == ""

