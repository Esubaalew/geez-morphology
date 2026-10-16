# Grammar rules used by the generator and the analyzer

These are the language decisions the generator (`generate_synthetic_data.py`, `verb_classes.py`,
`verb_class_paradigms.py`) and the analyzer (`analyzer.py`) follow. Page numbers are the
printed pages of the grammar books they were read from.

Root letters are written in the consonant symbols of `geez_script.py` (h1 ሀ, h2 ሐ, kh ኀ, a አ,
a2 ዐ, s2 ሠ, S ጸ, S2 ፀ, T ጠ, qh ቈ).

Tenses: perfect (past) and imperfect (present), as the data labels them, then the subjunctive
and the imperative. Vowel orders are the 1st to 7th orders (ግዕዝ ... ሳብዕ).

## 1. Guttural first radical: የ- in the imperfect

When the first radical is a guttural (ሀ ሐ ኀ አ ዐ), the imperfect prefix takes the 1st order:
የ/ተ/አ/ነ, as in የኀፍር, ተኀፍር, ነኀፍር. The subjunctive keeps ይ: ይኀፍር.

- Source: ፍሬ ግእዝ p. 18, ሆከ ፡ የሀውክ ፡ ይሁክ.
- Applied to the basic-stem imperfect of Type A, hollow and quadriliteral roots.
- In a Kidane Wold triple, the imperfect is the የ-form and the subjunctive is the ይ-form. Where
  the dictionary lists them the other way round, they are swapped.
- Ge'ez has one form here, the subjunctive (ይቅትል). Some older templates in `patterns_triliteral.json` are
  labelled "jussive"; they are read as subjunctive.

## 2. Middle guttural: ይምሕል, ይምሐል, ምሐል

In the ቀተለ class, when the second radical is a guttural (ሀ ሐ ኀ አ ዐ):
- the imperfect is all 6th order, with no doubling: ይምሕል, not ይመሕል (a guttural cannot be doubled);
- the subjunctive puts the 1st order on the letter before the last: ይምሐል;
- the imperative is built on that stem: ምሐል, ምሐሉ;
- the perfect is unchanged: መሐለ.

Sources:
- ፍሬ ግእዝ p. 63, rules ፲፪.፪–፲፪.፫: መሐለ ፡ ይምሕል ፡ ይምሐል and ሰአለ ፡ ይስእል ፡ ይስአል.
- ፍሬ ግእዝ p. 18: ድሕነ ፡ ይድሕን ፡ ይድሐን and ሰዐረ ፡ ይስዕር ፡ ይስዓር.
- The book prints ሐ, አ and ዐ. ኀ is included because the rule names the
  ሀ letters as a set.

Scope:
- The rule applies to roots that would otherwise get the default Type A templates.
- It is not applied to derived stems: passive ይትመሐል and causative ያመሕል stay as they are.
- It is not applied to initial-weak roots: ይወሕን stays.
- A root's own Kidane Wold triple wins over the rule (see section 4).
- A root whose first and second radicals are both gutturals (a-a-l, a-a-n, a-kh-s, a-kh-z,
  h1-kh-t, h2-kh-t) keeps the የ- imperfect, with the middle letter in the 6th order (የአኅዝ). No
  source has been found for its subjunctive, so it is left as generated.

## 3. 1st and 4th order after a guttural are the same spelling

ዐ/ዓ, ሐ/ሓ, ኀ/ኃ, ሀ/ሃ and አ/ኣ are spelling variants in these positions; for example, p. 18 prints
ይስዓር.
- The generator writes the 1st order.
- The analyzer and the gold evaluation treat each pair as equal: they map 4th to 1st order for
  these five letters only, and only when comparing. The data and the gold files are not rewritten.
- A word-final 4th-order guttural can also be a real ending: ሰምዓ is the 3fp perfect as well as a
  spelling of ሰምዐ. The analyzer returns both readings.

## 4. Kidane Wold triples, type_d roots and checked exceptions

- ፍሬ ግእዝ p. 63, rule ፲፪.፫: the perfect of these verbs may take a 6th-order first radical. So
  Type D (ንእደ) verbs have that perfect: n-a-d, Kidane Wold 8035, ንእደ ፡ ይንእድ ፡ ይንአድ. s-a-l is Type A (ሰአለ ፡ ይስእል ፡ ይስአል).
- Some Kidane Wold triples write the subjunctive exactly like the imperfect, with a 6th-order
  middle guttural (ድኅነ ፡ ይድኅን ፡ ይድኅን). This is an OCR merge of the 1st and 6th orders, not a
  real form. For every such root:
  - the subjunctive, jussive and imperative get the middle guttural in the 1st order;
  - the Kidane Wold perfect and present are kept.

On the corrected Kidane Wold transcription these are the roots with such a triple (7):

| root | perfect | present | subjunctive | imperative |
|---|---|---|---|---|
| T-a2-y | ጣዕየ | ይጣዕይ | (off the allow-list) | (off the allow-list) |
| d-kh-n | ድኅነ | ይድኅን | ይድኀን | ድኀን |
| d-kh-r | ድኅረ | ይድኅር | ይድኀር | ድኀር |
| f-kh-r | ፈኅረ | ይፍኅር | ይፍኀር | ፍኀር |
| n-kh-r | ንኅረ | ይንኅር | ይንኀር | ንኀር |
| z-kh-n | ዝኅነ | ይዝኅን | ይዝኀን | ዝኀን |
| z-kh-r | ዘኀረ | ይዝኅር | ይዝኀር | ዝኀር |

S-h2-f and s-a-l are not in that list: their corrected triples write the section 2 forms, and
s-a-l is a regular middle-guttural verb. d-kh-l is off the allow-list.
qh-h2-m has no triple (an OCR misread of entry 360), and the root is off the allow-list (section 4.3).
A root whose triple is not classified (class "unclear") but writes the section 2 shape (ሰአለ ፡ ይስእል ፡ ይስአል) gets the
section 2 rule (`verb_classes.kw_middle_guttural_pattern`).

### 4.1 Exceptions from the triple check

The exceptions are listed in `data/triple_check_exceptions.tsv`. A root's own triple still wins over the
default rules below.

- **Final guttural (Type A).** When the third radical is a guttural and the first two are not, the
  subjunctive, jussive and imperative put the second radical in the 4th order only where the guttural ends the
  word (ይስማዕ, ስማዕ). Before any vowel suffix, -ā included, the second radical stays 6th
  (ይስምዑ, ይስምዒ, ይስምዓ, ስምዓ), and so it does before an object suffix (gold train ምርሐኒ). The rule covers roots
  with no triple and roots whose triple has this shape (`verb_classes.kw_final_guttural_subjunctive`); 14 roots are
  listed by name. n-s2-a's imperative ንሣእ follows the rule; gold train spells that cell in the 6th order.
- **Initial guttural with final w/y:** the present takes የ- as in section 1 (የአቢ).
- **Initial w with a final guttural:** w-S2-a, w-k-a2 and w-q-a2 drop the w and put the second radical in the
  4th order (ይፃእ / ይካዕ / ይቃዕ); other roots whose triple has that shape get the same table
  (`verb_classes.kw_w_drop_final_guttural`). w-k-k (doubled final) keeps the w: ይውክክ.
- **Hollow with a final guttural:** q-w-a2 has the subjunctive ይቃዕ
  (`verb_classes.kw_hollow_final_guttural_subjunctive`).
- **Four radicals:** d-w-r-h1 is ይደወርህ; the w does not contract to ዶ
  (`verb_classes.kw_quad_uncontracted`).
- **type_b tags:** T-y-r, r-y-S2 and w-S2-f (`verb_classes.CLASS_TAG_EXCEPTIONS`). w-S2-f takes its present from
  the book; its subjunctive follows the type_b rule.
- **S-a2-l:** a guttural middle radical cannot double, so the type_b subjunctive is ይጽዐል, as in the
  book. The override (`verb_classes.type_b_subjunctive_orders`) is limited to type_b roots with a guttural
  middle radical; n-z-r, s-b-h2, s-d-s, f-S2-m, g-r-m and n-S2-r keep the type_b subjunctive ይቀድስ.
- **The generator stands** for 24 roots whose book form is a misprint or misread
  (a-l-d, a-l-l, a-m-t, a2-q-m, a2-s2-r, d-g-n, d-l-l, f-S2-m, g-r-m, h2-m-s, kh-S-d, kh-S-r, kh-T-T, kh-l-q, kh-n-q, kh-s-r, kh-w-s2, m-a2-z, n-S2-r, n-z-r, s-b-h2, s-d-s, w-q-r, w-s-d), and for S-b-r (Type B, subjunctive ይጸብር), kh-r-s (የኀርስ / ይኅርስ; the
  triple's letter is a misread) and m-l-a (present ይመልእ; m is not a guttural). The tests treat them as listed
  exceptions.
- **q-n-S and g-n-h2:** the generator makes the expected forms (q-n-S present ይቀንጽ), but a line whose
  form is a gold val or test form is left out, so those lines are not in the synthetic data.
- **w-kh-T** is off the allow-list until its triple is read against the printed page.

### 4.2 Final-y past stem

For l-f-y, l-q-y, m-l-y, s-t-y and a-b-y the 3rd-person forms of the basic perfect put the second radical in the
1st order, the way the book writes them: ሰተየ, not ሰትየ, and አበየ. Forms before a consonant ending are
unchanged (ሰተይኩ). g-l-y stays ገልየ, as the book writes it. m-s-y and s-b-y are not confirmed yet and are listed in
`verb_classes.Y_PAST_C2_FIRST_PENDING`; moving a root to `Y_PAST_C2_FIRST` is the whole change.
The list is in `data/final_y_past_stem_roots.tsv`.

### 4.3 Class changes

Every root whose class changes with the rebuilt triples has a row in `data/class_changes.tsv`:
27 roots become Type B and 4 Type D because their triples say so; kh-y-r is hollow; a-l-l and s-a-l are covered above.
q-l-s stays Type A. qh-h2-m is a phantom root: its Type D triple came from an OCR misread, and it has no other book
source, so it is off the allow-list. Entry 360 is
ቅሒም / ቀሐመ ፡ ይቅሕም ፡ ይቅሐም. For d-r-g, n-z-z and q-r-h2 the Merho Ges class (Type C) outranks the new triple, which belongs
to another verb on the same root code, so they keep Type C (`verb_classes.CLASS_TAG_EXCEPTIONS`).
Passive, causative and aste forms are generated for Type A and quadriliteral roots and for Type B roots with strong
radicals (`verb_class_paradigms.type_b_derived`); a Type B root with a guttural middle radical (S-a2-l) has none, because a
guttural cannot double. The other classes have none yet.

## 5. aste is አስተ-

The causative-reflexive stem is spelled with አስተ- (አስተቅተለ, ያስተቀትል). Plain አስ- is Amharic. The
stem label is `aste`.

## 6. Hollow, final-weak and initial-w roots

- **Hollow** (w/y as second radical): these use the Type A basic-stem templates, and the engine
  contracts the glide: ቆመ, ይቀውም, ይቁም. The generator does not use the hand-written weak_middle
  templates; the analyzer and `cli.py` still use them to read hollow forms.
- **Final-w (ተለወ type):** generated only for a root with a ይተሉ/ይትሉ triple or a class tag.
  ሀለወ (ይሄሉ) is Type B with a final w, not ተለወ type.
- **Final-y (ሰተየ type):** generated only for a tagged root (ይሰቲ / ይስተይ). The በከየ type (subjunctive
  ይብኪ) is different and has no checked paradigm.
  Four roots take the perfect stem of section 4.2.
- Final-weak roots without a tag are skipped.
- The checked paradigm tables are in `verb_class_paradigms.py`, and
  `tests/test_verb_class_paradigms.py` checks them.
- **Initial-w:** these get the Type A perfect and imperfect only. The subjunctive and imperative are
  generated only for the ወለደ type (ይለድ, ለድ): roots on the W_DROP list or with a ይለድ-shaped
  triple.
  Section 4.1 adds the initial-w roots with a final guttural and w-k-k.

## 7. Roots that are not generated

- Roots with an Amharic-only letter: ሸ ቸ ኘ ኸ ዠ ጀ ጨ. Classical Ge'ez has none of these.
- Doubly weak roots, with two of w/y, such as ሐይወ (ሐይወ / የሐዩ, not hollow).
- Quadriliteral roots ending in w or y. No checked template exists for them.
- ማህረከ (four-radical Type C). geezgram-2 prints ማህረከ ፡ ይማህርክ ፡ ይማህርክ and no
  imperative. The root is in neither dictionary file, so it has no rows.
- ጦመረ and ሴሰየ types (Type C with ō/ē). No conjugation table has been found, and no Kidane Wold
  triple has these shapes. Their builders exist and are tested, but they produce no rows.

## 8. Where verb classes come from

- A root's class comes only from lexical evidence, never from its spelling (a Type B perfect is
  spelled like Type A). The evidence is:
  - its accepted Kidane Wold triple;
  - the Merho Ges training tables' class names (ባረከ → Type C, ቀደሰ → Type B, ሰተየ → final-y);
  - the class lemmas.
- A root with no evidence gets the default for its shape.
- No class is taken from any other analyzer's labels; a class comes only from the evidence above.

## 9. Kidane Wold triple check

`kidane_wold_triples.py` reads the bracketed perfect ፡ imperfect ፡ subjunctive triples from the
dictionary. A triple is used only if both of these hold:
- all three forms show the same radicals in the same order (w/y may be missing in hollow,
  initial-w and final-weak verbs);
- the perfect ends in a 1st-order letter.

Accepted triples are in `geez-dataset/data/lexicon/kidane_wold_verb_triples.jsonl`. The
rejected ones, with the reason for each, are in `kidane_wold_build/` (written by `kidane_wold_triples.py`).
