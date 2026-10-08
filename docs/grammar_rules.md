# Grammar rules used by the generator and the analyzer

These are the language decisions the generator (`generate_synthetic_fixed.py`, `verb_classes.py`,
`verb_class_paradigms.py`) and the analyzer (`analyzer.py`) follow. Page numbers are the
printed pages of the grammar books they were read from. "ፍሬ ግእዝ page_014" is the
image file name in that scan, not a file in this repository.

Root letters are written in the consonant symbols of `geez_config.py` (h1 ሀ, h2 ሐ, kh ኀ, a አ,
a2 ዐ, s2 ሠ, S ጸ, S2 ፀ, T ጠ, qh ቈ).

## 1. Guttural first radical: የ- in the present

When the first radical is a guttural (ሀ ሐ ኀ አ ዐ), the present prefix takes the 1st order:
የ/ተ/አ/ነ, as in የኀፍር, ተኀፍር, ነኀፍር. The subjunctive keeps ይ: ይኀፍር.

- Source: ፍሬ ግእዝ p. 18 (page_014), ሆከ ፡ የሀውክ ፡ ይሁክ.
- Applied to the basic-stem present of Type A, hollow and quadriliteral roots.
- In a Kidane Wold triple, the imperfect is the የ-form and the subjunctive is the ይ-form. Where
  the dictionary lists them the other way round, they are swapped.

## 2. Middle guttural: ይምሕል, ይምሐል, ምሐል

In the ቀተለ class, when the second radical is a guttural (ሀ ሐ ኀ አ ዐ):
- the present is all 6th order, with no doubling: ይምሕል, not ይመሕል (a guttural cannot be doubled);
- the subjunctive and the jussive put the 1st order on the letter before the last: ይምሐል;
- the imperative is built on that stem: ምሐል, ምሐሉ;
- the perfect is unchanged: መሐለ.

Sources:
- ፍሬ ግእዝ p. 63 (page_037), rules ፲፪.፪–፲፪.፫: መሐለ ፡ ይምሕል ፡ ይምሐል and ሰአለ ፡ ይስእል ፡ ይስአል.
- ፍሬ ግእዝ p. 18 (page_014): ድሕነ ፡ ይድሕን ፡ ይድሐን and ሰዐረ ፡ ይስዕር ፡ ይስዓር.
- The book prints ሐ, አ and ዐ. ኀ is included because the rule names the
  ሀ letters as a set.

Scope:
- The rule applies to roots that would otherwise get the default Type A templates.
- It is not applied to derived stems: passive ይትመሐል and causative ያመሕል stay as they are.
- It is not applied to initial-weak roots: ይወሕን stays.
- A root's own Kidane Wold triple wins over the rule (see section 4).
- A root whose first and second radicals are both gutturals (a-a-l, a-a-n, a-kh-s, a-kh-z,
  h1-kh-t, h2-kh-t) keeps the የ- present, with the middle letter in the 6th order (የአኅዝ). No
  source has been found for its subjunctive, so it is left as generated.

## 3. 1st and 4th order after a guttural are the same spelling

ዐ/ዓ, ሐ/ሓ, ኀ/ኃ, ሀ/ሃ and አ/ኣ are spelling variants in these positions; for example, p. 18 prints
ይስዓር.
- The generator writes the 1st order.
- The analyzer and the gold evaluation treat each pair as equal: they map 4th to 1st order for
  these five letters only, and only when comparing. The data and the gold files are not rewritten.
- A word-final 4th-order guttural can also be a real ending: ሰምዓ is the 3fp perfect as well as a
  spelling of ሰምዐ. The analyzer returns both readings.

## 4. type_d roots and Kidane Wold subjunctives copied from the imperfect

- ፍሬ ግእዝ p. 63, rule ፲፪.፫: the perfect of these verbs may take a 6th-order first radical. So
  Kidane Wold's ስእለ is a real variant of ሰአለ. Its subjunctive is still ይስአል.
- Some Kidane Wold triples write the subjunctive exactly like the imperfect, with a 6th-order
  middle guttural (ድኅነ ፡ ይድኅን ፡ ይድኅን). This is an OCR merge of the 1st and 6th orders, not a
  real form. For every such root:
  - the subjunctive, jussive and imperative get the middle guttural in the 1st order;
  - the Kidane Wold perfect and present are kept.

| root | perfect | present | subjunctive | imperative |
|---|---|---|---|---|
| s-a-l | ስእለ | ይስእል | ይስአል | ስአል |
| s-a-n | ስእነ | ይስእን | ይስአን | ስአን |
| d-kh-n | ድኅነ | ይድኅን | ይድኀን | ድኀን |
| d-kh-l | ድኅለ | ይድኅል | ይድኀል | ድኀል |
| d-kh-r | ድኅረ | ይድኅር | ይድኀር | ድኀር |
| qh-h2-m | ቍሕመ | ይቍሕም | ይቍሐም | ቍሐም |
| S-h2-f | ጸሐፈ | ይጽሕፍ | ይጽሐፍ | ጽሐፍ |

For ጸ-ሐ-ፈ, the generated perfect ጸሐፈ is kept. The triple writes ጸሕፈ.

## 5. aste is አስተ-

The causative-reflexive stem is spelled with አስተ- (አስተቅተለ, ያስተቀትል). Plain አስ- is Amharic. The
stem label is `aste`.

## 6. Hollow, final-weak and initial-w roots

- **Hollow** (w/y as second radical): these use the Type A basic-stem templates, and the engine
  contracts the glide: ቆመ, ይቀውም, ይቁም. The hand-written weak_middle templates are not used.
- **Final-w (ተለወ type):** generated only for a root with a ይተሉ/ይትሉ triple or a class tag.
  ሀለወ (ይሄሉ) is Type B with a final w, not ተለወ type.
- **Final-y (ሰትየ type):** generated only for a tagged root (ይሰቲ / ይስተይ). The በከየ type (subjunctive
  ይብኪ) is different and has no checked paradigm.
- Final-weak roots without a tag are skipped.
- The checked paradigm tables are in `verb_class_paradigms.py`, and
  `tests/test_verb_class_paradigms.py` checks them.
- **Initial-w:** these get the Type A perfect and present only. The subjunctive and imperative are
  generated only for the ወለደ type (ይለድ, ለድ): roots on the W_DROP list or with a ይለድ-shaped
  triple.

## 7. Roots that are not generated

- Roots with an Amharic-only letter: ሸ ቸ ኘ ኸ ዠ ጀ ጨ. Classical Ge'ez has none of these.
- Doubly weak roots, with two of w/y, such as ሐይወ (ሐይወ / የሐዩ, not hollow).
- Quadriliteral roots ending in w or y. No checked template exists for them.
- ማህረከ (four-radical Type C). geezgram-2 page_008 prints ማህረከ ፡ ይማህርክ ፡ ይማህርክ and no
  imperative. The root is in neither dictionary file, so it has no rows.
- ጦመረ and ሴሰየ types (Type C with ō/ē). No conjugation table has been found, and no Kidane Wold
  triple has these shapes. Their builders exist and are tested, but they produce no rows.

## 8. Where verb classes come from

- A root's class comes only from lexical evidence, never from its spelling (a Type B perfect is
  spelled like Type A). The evidence is:
  - its accepted Kidane Wold triple;
  - the Merho Ges training tables' class names (ባረከ → Type C, ቀደሰ → Type B, ሰትየ → final-y);
  - the class lemmas.
- A root with no evidence gets the default for its shape.
- **an earlier prototype is a hint only.** Its classes come from earlier work that could not be confirmed. A
  an earlier prototype lemma is linked only by exact spelling, and no class is taken from it.

## 9. Kidane Wold triple check

`kw_verb_triples.py` reads the bracketed perfect ፡ imperfect ፡ subjunctive triples from the
dictionary. A triple is used only if both of these hold:
- all three forms show the same radicals in the same order (w/y may be missing in hollow,
  initial-w and final-weak verbs);
- the perfect ends in a 1st-order letter.

Accepted triples are in `geez-dataset/data/lexicon/kidane_wold_verb_triples.jsonl`. The
rejected ones, with the reason for each, are in `kidane_wold_verb_triples_rejected.jsonl`.
