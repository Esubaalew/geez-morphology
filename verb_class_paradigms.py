"""
Checked paradigms for the verb classes missing from the synthetic data:
Type B (ቀደሰ), Type C (ባረከ), final-weak (ሰትየ, ተለወ) and the
initial-weak subjunctive (ወለደ, ወረደ).

Forms follow the standard classical paradigms (Lambdin, Introduction to
Classical Ethiopic; Dillmann). Each class is a template over radical
letters + vowel order, so the same code works for any root of that class.
Running this file rebuilds every form from the templates and checks it
against the hand-written EXPECTED tables.

Vowel orders: 1=ä 2=u 3=i 4=a 5=e 6=ə(zero) 7=o
"""

PERSONS = ["3ms", "3fs", "2ms", "2fs", "1s", "3mp", "3fp", "2mp", "2fp", "1p"]
IMP_PERSONS = ["2ms", "2fs", "2mp", "2fp"]

# Ethiopic syllable for a base letter (any order) + order 1..7
def syl(ch, order):
    base = ord(ch) - ((ord(ch) - 0x1200) % 8)
    return chr(base + order - 1)

# prefixes for imperfect / subjunctive: (prefix, ending-order or suffix)
# ending: None = stem as is, "i"/"u"/"a" = last radical takes order 3/2/4
PREFIX = {
    "3ms": ("ይ", None), "3fs": ("ት", None), "2ms": ("ት", None),
    "2fs": ("ት", "i"), "1s": ("እ", None), "3mp": ("ይ", "u"),
    "3fp": ("ይ", "a"), "2mp": ("ት", "u"), "2fp": ("ት", "a"), "1p": ("ን", None),
}
IMP_END = {"2ms": None, "2fs": "i", "2mp": "u", "2fp": "a"}
END_ORDER = {"i": 3, "u": 2, "a": 4}

# perfect suffixes after a closed stem (last radical order 6)
PERF_CLOSED = {"2ms": "ከ", "2fs": "ኪ", "1s": "ኩ", "2mp": "ክሙ", "2fp": "ክን", "1p": "ነ"}


def strong_perfect(r, v1):
    """Perfect for Type A/B/C: v1 = order of radical 1 (1 for A/B, 4 for C)."""
    c1 = syl(r[0], v1)
    out = {
        "3ms": c1 + syl(r[1], 1) + syl(r[2], 1),
        "3fs": c1 + syl(r[1], 1) + syl(r[2], 1) + "ት",
        "3mp": c1 + syl(r[1], 1) + syl(r[2], 2),
        "3fp": c1 + syl(r[1], 1) + syl(r[2], 4),
    }
    for p, s in PERF_CLOSED.items():
        out[p] = c1 + syl(r[1], 1) + syl(r[2], 6) + s
    return out


def prefixed(r, v1, v2):
    """Imperfect/subjunctive with radical orders v1, v2, last radical 6 (or ending)."""
    out = {}
    for p, (pre, end) in PREFIX.items():
        last = syl(r[2], END_ORDER[end]) if end else syl(r[2], 6)
        out[p] = pre + syl(r[0], v1) + syl(r[1], v2) + last
    return out


def imperative(r, v1, v2):
    out = {}
    for p, end in IMP_END.items():
        last = syl(r[2], END_ORDER[end]) if end else syl(r[2], 6)
        out[p] = syl(r[0], v1) + syl(r[1], v2) + last
    return out


def type_b(r):  # ቀደሰ qaddasa / ይቄድስ yəqeddəs / ይቀድስ yəqaddəs / ቀድስ
    return {"perfect": strong_perfect(r, 1), "imperfect": prefixed(r, 5, 6),
            "subjunctive": prefixed(r, 1, 6), "imperative": imperative(r, 1, 6)}


def type_c(r):  # ባረከ bāraka / ይባርክ / ይባርክ / ባርክ  (imperfect == subjunctive)
    return {"perfect": strong_perfect(r, 4), "imperfect": prefixed(r, 4, 6),
            "subjunctive": prefixed(r, 4, 6), "imperative": imperative(r, 4, 6)}


def type_c_o(r):  # ጦመረ ṭōmara / ይጦምር / ይጦምር / ጦምር: Type C with ō kept on radical 1
    return {"perfect": strong_perfect(r, 7), "imperfect": prefixed(r, 7, 6),
            "subjunctive": prefixed(r, 7, 6), "imperative": imperative(r, 7, 6)}


def type_c_e(r):  # Type C with ē kept on radical 1 (an earlier prototype ሴሰየ type, strong radicals only)
    return {"perfect": strong_perfect(r, 5), "imperfect": prefixed(r, 5, 6),
            "subjunctive": prefixed(r, 5, 6), "imperative": imperative(r, 5, 6)}


def quad_c(r):  # ማህረከ māhraka / ይማህርክ / ይማህርክ / ማህርክ: 4-radical Type C (ā on radical 1)
    head = syl(r[0], 4) + syl(r[1], 6)
    perfect = {
        "3ms": head + syl(r[2], 1) + syl(r[3], 1),
        "3fs": head + syl(r[2], 1) + syl(r[3], 1) + "ት",
        "3mp": head + syl(r[2], 1) + syl(r[3], 2),
        "3fp": head + syl(r[2], 1) + syl(r[3], 4),
    }
    for p, s in PERF_CLOSED.items():
        perfect[p] = head + syl(r[2], 1) + syl(r[3], 6) + s
    imperfect = {}
    for p, (pre, end) in PREFIX.items():
        last = syl(r[3], END_ORDER[end]) if end else syl(r[3], 6)
        imperfect[p] = pre + head + syl(r[2], 6) + last
    imper = {}
    for p, end in IMP_END.items():
        last = syl(r[3], END_ORDER[end]) if end else syl(r[3], 6)
        imper[p] = head + syl(r[2], 6) + last
    return {"perfect": perfect, "imperfect": imperfect,
            "subjunctive": dict(imperfect), "imperative": imper}


GUTTURAL_LETTERS = set("አዐሀሐኀ")
GUTTURAL_PREFIX = {"ይ": "የ", "ት": "ተ", "እ": "አ", "ን": "ነ"}   # የኀፍር, ተኀፍር, አኀፍር, ነኀፍር


def type_d(r, orders):
    """II-guttural verbs with C1 6th order in the perfect (ጽሕቀ ፡ ይጽሕቅ ፡ ይጽሐቅ).
    orders = {"perfect"|"imperfect"|"subjunctive": (order of radical 1, order of
    radical 2)}, copied from the verb's own Kidane Wold triple, because the
    subjunctive differs from verb to verb (ይጽሐቅ, ይድኅን)."""
    po, io, so = orders["perfect"], orders["imperfect"], orders["subjunctive"]
    head = syl(r[0], po[0]) + syl(r[1], po[1])
    perfect = {"3ms": head + syl(r[2], 1), "3fs": head + syl(r[2], 1) + "ት",
               "3mp": head + syl(r[2], 2), "3fp": head + syl(r[2], 4)}
    for p, s in PERF_CLOSED.items():
        perfect[p] = head + syl(r[2], 6) + s
    gutt = syl(r[0], 1) in GUTTURAL_LETTERS

    def pre_table(o, imperfect):
        out = {}
        for p, (pre, end) in PREFIX.items():
            if imperfect and gutt:
                pre = GUTTURAL_PREFIX[pre]
            last = syl(r[2], END_ORDER[end]) if end else syl(r[2], 6)
            out[p] = pre + syl(r[0], o[0]) + syl(r[1], o[1]) + last
        return out

    imper = {}
    for p, end in IMP_END.items():
        last = syl(r[2], END_ORDER[end]) if end else syl(r[2], 6)
        imper[p] = syl(r[0], so[0]) + syl(r[1], so[1]) + last
    return {"perfect": perfect, "imperfect": pre_table(io, True),
            "subjunctive": pre_table(so, False), "imperative": imper}


def final_y(r):
    """III-y, a-type like ሰትየ satya: perfect satya/satayka, imperf yəsatti,
    subj yəstay, impv stay. r[2] must be የ."""
    c1, c2 = r[0], r[1]
    perf = {
        "3ms": syl(c1, 1) + syl(c2, 6) + "የ", "3fs": syl(c1, 1) + syl(c2, 6) + "የት",
        "3mp": syl(c1, 1) + syl(c2, 6) + "ዩ", "3fp": syl(c1, 1) + syl(c2, 6) + "ያ",
    }
    for p, s in PERF_CLOSED.items():
        perf[p] = syl(c1, 1) + syl(c2, 1) + "ይ" + s
    imperf, subj = {}, {}
    for p, (pre, end) in PREFIX.items():
        if end is None:
            imperf[p] = pre + syl(c1, 1) + syl(c2, 3)            # ይሰቲ
            subj[p] = pre + syl(c1, 6) + syl(c2, 1) + "ይ"        # ይስተይ
        elif end == "i":
            imperf[p] = pre + syl(c1, 1) + syl(c2, 3)            # ትሰቲ (2fs same as 2ms)
            subj[p] = pre + syl(c1, 6) + syl(c2, 1) + "ዪ"        # ትስተዪ
        else:
            y = syl("የ", END_ORDER[end])
            imperf[p] = pre + syl(c1, 1) + syl(c2, 6) + y        # ይሰትዩ
            subj[p] = pre + syl(c1, 6) + syl(c2, 1) + y          # ይስተዩ
    impv = {"2ms": syl(c1, 6) + syl(c2, 1) + "ይ", "2fs": syl(c1, 6) + syl(c2, 1) + "ዪ",
            "2mp": syl(c1, 6) + syl(c2, 1) + "ዩ", "2fp": syl(c1, 6) + syl(c2, 1) + "ያ"}
    return {"perfect": perf, "imperfect": imperf, "subjunctive": subj, "imperative": impv}


def final_w(r):
    """III-w like ተለወ talawa: perfect talawa/talawka, imperf yətallu,
    subj yətlu, impv tlu. r[2] must be ወ."""
    c1, c2 = r[0], r[1]
    perf = {
        "3ms": syl(c1, 1) + syl(c2, 1) + "ወ", "3fs": syl(c1, 1) + syl(c2, 1) + "ወት",
        "3mp": syl(c1, 1) + syl(c2, 1) + "ዉ", "3fp": syl(c1, 1) + syl(c2, 1) + "ዋ",
    }
    for p, s in PERF_CLOSED.items():
        perf[p] = syl(c1, 1) + syl(c2, 1) + "ው" + s
    imperf, subj = {}, {}
    for p, (pre, end) in PREFIX.items():
        if end is None:
            imperf[p] = pre + syl(c1, 1) + syl(c2, 2)            # ይተሉ
            subj[p] = pre + syl(c1, 6) + syl(c2, 2)              # ይትሉ
        else:
            w = syl("ወ", END_ORDER[end])
            imperf[p] = pre + syl(c1, 1) + syl(c2, 6) + w        # ይተልዉ / ትተልዊ
            subj[p] = pre + syl(c1, 6) + syl(c2, 6) + w          # ይትልዉ / ትትልዊ
    impv = {"2ms": syl(c1, 6) + syl(c2, 2)}
    for p in ("2fs", "2mp", "2fp"):
        impv[p] = syl(c1, 6) + syl(c2, 6) + syl("ወ", END_ORDER[IMP_END[p]])
    return {"perfect": perf, "imperfect": imperf, "subjunctive": subj, "imperative": impv}


def initial_w_drop(r):
    """Subjunctive/imperative of the I-w verbs that DROP w: ወለደ -> ይለድ, ለድ.
    Closed lexical list only (see W_DROP). Perfect and imperfect are Type A."""
    return {"subjunctive": prefixed_two(r[1], r[2]),
            "imperative": {p: syl(r[1], 1) + (syl(r[2], END_ORDER[e]) if e else syl(r[2], 6))
                           for p, e in IMP_END.items()}}


def prefixed_two(c2, c3):
    out = {}
    for p, (pre, end) in PREFIX.items():
        last = syl(c3, END_ORDER[end]) if end else syl(c3, 6)
        out[p] = pre + syl(c2, 1) + last
    return out


# I-w roots known to drop w in subjunctive/imperative. Do NOT apply to other
# I-w roots (many keep w or have their own forms); skip their subjunctive
# until checked against the dictionary.
W_DROP = {"w-l-d": "ወለደ", "w-r-d": "ወረደ"}

EXPECTED = {
    ("type_b", "ቀደሰ"): {
        "perfect": "ቀደሰ ቀደሰት ቀደስከ ቀደስኪ ቀደስኩ ቀደሱ ቀደሳ ቀደስክሙ ቀደስክን ቀደስነ",
        "imperfect": "ይቄድስ ትቄድስ ትቄድስ ትቄድሲ እቄድስ ይቄድሱ ይቄድሳ ትቄድሱ ትቄድሳ ንቄድስ",
        "subjunctive": "ይቀድስ ትቀድስ ትቀድስ ትቀድሲ እቀድስ ይቀድሱ ይቀድሳ ትቀድሱ ትቀድሳ ንቀድስ",
        "imperative": "ቀድስ ቀድሲ ቀድሱ ቀድሳ",
    },
    ("type_c", "ባረከ"): {
        "perfect": "ባረከ ባረከት ባረክከ ባረክኪ ባረክኩ ባረኩ ባረካ ባረክክሙ ባረክክን ባረክነ",
        "imperfect": "ይባርክ ትባርክ ትባርክ ትባርኪ እባርክ ይባርኩ ይባርካ ትባርኩ ትባርካ ንባርክ",
        "subjunctive": "ይባርክ ትባርክ ትባርክ ትባርኪ እባርክ ይባርኩ ይባርካ ትባርኩ ትባርካ ንባርክ",
        "imperative": "ባርክ ባርኪ ባርኩ ባርካ",
    },
    ("final_y", "ሰትየ"): {
        "perfect": "ሰትየ ሰትየት ሰተይከ ሰተይኪ ሰተይኩ ሰትዩ ሰትያ ሰተይክሙ ሰተይክን ሰተይነ",
        "imperfect": "ይሰቲ ትሰቲ ትሰቲ ትሰቲ እሰቲ ይሰትዩ ይሰትያ ትሰትዩ ትሰትያ ንሰቲ",
        "subjunctive": "ይስተይ ትስተይ ትስተይ ትስተዪ እስተይ ይስተዩ ይስተያ ትስተዩ ትስተያ ንስተይ",
        "imperative": "ስተይ ስተዪ ስተዩ ስተያ",
    },
    ("final_w", "ተለወ"): {
        "perfect": "ተለወ ተለወት ተለውከ ተለውኪ ተለውኩ ተለዉ ተለዋ ተለውክሙ ተለውክን ተለውነ",
        "imperfect": "ይተሉ ትተሉ ትተሉ ትተልዊ እተሉ ይተልዉ ይተልዋ ትተልዉ ትተልዋ ንተሉ",
        "subjunctive": "ይትሉ ትትሉ ትትሉ ትትልዊ እትሉ ይትልዉ ይትልዋ ትትልዉ ትትልዋ ንትሉ",
        "imperative": "ትሉ ትልዊ ትልዉ ትልዋ",
    },
    ("initial_w_drop", "ወለደ"): {
        "subjunctive": "ይለድ ትለድ ትለድ ትለዲ እለድ ይለዱ ይለዳ ትለዱ ትለዳ ንለድ",
        "imperative": "ለድ ለዲ ለዱ ለዳ",
    },
    ("initial_w_drop", "ወረደ"): {
        "subjunctive": "ይረድ ትረድ ትረድ ትረዲ እረድ ይረዱ ይረዳ ትረዱ ትረዳ ንረድ",
        "imperative": "ረድ ረዲ ረዱ ረዳ",
    },
}

CLASSES = {"type_b": type_b, "type_c": type_c, "final_y": final_y,
           "final_w": final_w, "initial_w_drop": initial_w_drop,
           # From the grammar check of the written shape types (2026-10-05). Used only
           # for roots whose Kidane Wold triple is written this way.
           "type_c_o": type_c_o, "type_c_e": type_c_e, "quad_c": quad_c}

# Ambiguities the analyzer and label merge must respect
NOTES = """
- Type C: imperfect == subjunctive in writing (ይባርክ). Keep both labels on one surface.
- Type B perfect is written exactly like Type A (ቀደሰ); class must come from the lexicon,
  not the surface. Only the imperfect (ይቄድስ, 5th order on C1) shows it.
- ተለወ imperfect 3ms ይተሉ looks like a strong 3mp (ይቀትሉ pattern). Analyzer must try
  final-w root t-l-w before reading -ሉ as a 3mp ending.
- ሰትየ 2fs imperfect ትሰቲ == 2ms ትሰቲ (no separate -i in writing).
- Not every final-y verb is a-type: verbs like በከየ (imperf ይበኪ, subj ይብኪ) differ in the
  subjunctive. Only use final_y() for roots tagged like ሰትየ.
- initial_w_drop applies only to W_DROP. Skip other I-w subjunctives until checked.
"""


def check():
    bad = 0
    for (cls, lemma), tenses in EXPECTED.items():
        r = list(lemma)
        built = CLASSES[cls](r)
        for tense, exp in tenses.items():
            persons = IMP_PERSONS if tense == "imperative" else PERSONS
            got = " ".join(built[tense][p] for p in persons)
            if got != exp:
                bad += 1
                print(f"FAIL {cls} {lemma} {tense}\n  exp {exp}\n  got {got}")
    n = sum(len(t) for t in EXPECTED.values())
    print(f"{n - bad}/{n} tense tables match")
    return bad == 0


def test_paradigms():
    assert check()


if __name__ == "__main__":
    check()
