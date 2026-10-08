#!/usr/bin/env python3
"""
Command line for Ge'ez morphological analysis, generation and dictionary lookup.
"""

import sys
import argparse
import json

from analyzer import GeezMorphologicalAnalyzer
from engine import InterdigitationEngine, slots_from_spec
import geez_config
import dataset_loader


def cmd_analyze(args):
    analyzer = GeezMorphologicalAnalyzer(dataset_dir=args.dataset_dir)
    results = analyzer.analyze(args.word)
    if not results:
        print(f"No morphological pattern matched for: {args.word}")
        return

    print(f"\nMorphological Analyses for '{args.word}':")
    print("=" * 60)
    for i, a in enumerate(results, 1):
        print(f"[{i}] Lemma: {a.canonical_lemma}  (Root: {a.root_string})")
        print(f"    Tense/Aspect: {a.tense}")
        print(f"    Agreement:    Person {a.person}, {a.gender}, {a.number}")
        print(f"    Stem/Class:   {a.stem} ({a.verb_type})")
        print(f"    Pattern Spec: {a.pattern_spec}")
        if a.lexicon_entry:
            lex = a.lexicon_entry
            pos = lex.get("pos", "unknown")
            defn = lex.get("amharic_definition", "")
            print(f"    POS:          {pos}")
            print(f"    Definition:   {defn[:100]}{'...' if len(defn) > 100 else ''}")
            if lex.get("citations"):
                print(f"    Citations:    {', '.join(lex['citations'][:3])}")
        print("-" * 60)


def cmd_lemmatize(args):
    analyzer = GeezMorphologicalAnalyzer(dataset_dir=args.dataset_dir)
    tokens = analyzer.lemmatize_text(args.text)
    print(f"\nLemmatizing sentence: \"{args.text}\"")
    print("=" * 60)
    for t in tokens:
        clitic_str = f" [Proclitic: {', '.join(t.proclitics)}]" if t.proclitics else ""
        if t.analyses:
            best = t.analyses[0]
            print(f"{t.token:<15} -> {best.canonical_lemma:<12} (Root: {best.root_string}, {best.tense} {best.person}{best.gender}{best.number}){clitic_str}")
        else:
            print(f"{t.token:<15} -> (unmatched){clitic_str}")


# --verb_type takes the class names the generator uses.
#   type_a, hollow   the triliteral templates (Type A; hollow roots use the Type A basic stem,
#                    and the engine contracts the glide, as in the generator)
#   quadriliteral    patterns_quadriliteral.json
#   the others       the checked class tables (verb_classes.paradigm_rows), basic stem only
TEMPLATE_CLASSES = ("type_a", "hollow")
TABLE_CLASSES = ("type_b", "type_c", "type_d", "final_w", "final_y", "initial_w_drop")
VERB_TYPE_CHOICES = TEMPLATE_CLASSES + ("quadriliteral",) + TABLE_CLASSES


def cmd_generate(args):
    engine = InterdigitationEngine(renderer=geez_config.render_unit)
    
    # Load patterns
    import os
    base_dir = os.path.dirname(os.path.abspath(__file__))
    vtype = getattr(args, "verb_type", None)
    patterns_file = os.path.join(base_dir, "patterns_quadriliteral.json" if vtype == "quadriliteral"
                                 else "patterns_triliteral.json")
    with open(patterns_file, "r", encoding="utf-8") as f:
        patterns = json.load(f)
    if vtype in TEMPLATE_CLASSES:
        patterns = [p for p in patterns if p["features"].get("verb_type", "strong_A") == "strong_A"
                    and (vtype == "type_a" or p["features"].get("stem") == "basic")]

    # Parse root
    if "-" in args.root:
        cons = tuple(args.root.split("-"))
    elif len(args.root) in (3, 4):
        # Decompose if Ethiopic characters
        units = geez_config.decompose_string(args.root)
        cons = tuple(c for c, v in units if c is not None)
    else:
        cons = tuple(args.root)

    print(f"\nGenerating forms for root: {'-'.join(cons)}")
    print("=" * 60)

    def wanted(feat):
        return not ((args.tense and feat.get("tense") != args.tense)
                    or (args.person and feat.get("person") != args.person)
                    or (args.number and feat.get("number") != args.number)
                    or (args.gender and feat.get("gender") != args.gender)
                    or (args.stem and feat.get("stem") != args.stem))

    matches = []
    if vtype in TABLE_CLASSES:
        import verb_classes
        if not verb_classes.class_fits_root(vtype, cons):
            print(f"The {vtype} table does not take the root {'-'.join(cons)}.")
            return
        orders = None
        if vtype == "type_d":
            # type_d vowels come from the root's own Kidane Wold triple, as in the generator
            orders = verb_classes.type_d_orders(args.dataset_dir).get(cons)
            if orders is None:
                print(f"No Kidane Wold type_d triple gives the vowels for {'-'.join(cons)}.")
                return
        for feat, form in verb_classes.paradigm_rows(cons, vtype, orders):
            if wanted(feat):
                matches.append((form, feat, f"verb_class_paradigms.{vtype}"))
    for p in ([] if vtype in TABLE_CLASSES else patterns):
        feat = p["features"]
        if not wanted(feat):
            continue
        
        try:
            form = engine.generate(cons, slots_from_spec(p["pattern_spec"]))
            matches.append((form, feat, p["pattern_spec"]))
        except Exception:
            pass

    if not matches:
        print("No matching patterns found for the specified features.")
        return

    for form, feat, spec in matches:
        print(f"{form:<15} | {feat.get('tense', ''):<12} | {feat.get('person', '')}{feat.get('gender', '')}{feat.get('number', ''):<4} | {feat.get('stem', ''):<10} | {feat.get('verb_type', ''):<12} | {spec}")


def cmd_lookup(args):
    analyzer = GeezMorphologicalAnalyzer(dataset_dir=args.dataset_dir)
    entries = analyzer.lookup_lexicon(args.word)
    if not entries:
        print(f"'{args.word}' was not found in the dictionary.")
        return

    print(f"\nLexicon entries for '{args.word}' ({len(entries)} matches):")
    print("=" * 60)
    for i, e in enumerate(entries, 1):
        print(f"[{i}] Headword:    {e.get('headword_geez')} (Page {e.get('page_number')})")
        print(f"    POS:         {e.get('pos')}")
        print(f"    Root:        {e.get('root') or e.get('inferred_root')}")
        print(f"    Definition:  {e.get('amharic_definition')}")
        if e.get("forms"):
            print(f"    Forms:       {', '.join(e['forms'])}")
        if e.get("citations"):
            print(f"    Citations:   {', '.join(e['citations'])}")
        print("-" * 60)


def main():
    parser = argparse.ArgumentParser(description="Analyze, generate and look up Classical Ge'ez verbs.")
    parser.add_argument("--dataset_dir", default=dataset_loader.default_dataset_dir(),
                        help="Checkout of the geez-dataset repo, for the dictionary "
                             "(default: $GEEZ_DATASET_DIR, else ../geez-dataset)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # analyze
    p_analyze = subparsers.add_parser("analyze", help="Analyze an inflected Ge'ez word")
    p_analyze.add_argument("word", help="Ge'ez surface form (e.g. ይቀትል)")

    # lemmatize
    p_lemma = subparsers.add_parser("lemmatize", help="Lemmatize a Ge'ez sentence")
    p_lemma.add_argument("text", help="Ge'ez sentence or phrase")

    # generate
    p_gen = subparsers.add_parser("generate", help="Generate inflected forms from a root")
    p_gen.add_argument("root", help="Consonantal root (e.g. q-t-l or ቀተለ)")
    p_gen.add_argument("--tense", choices=["past", "present", "subjunctive", "imperative"],
                       help="past = perfect, present = imperfect")
    p_gen.add_argument("--person", choices=["1", "2", "3"])
    p_gen.add_argument("--number", choices=["s", "p"])
    p_gen.add_argument("--gender", choices=["m", "f"])
    p_gen.add_argument("--stem", choices=["basic", "causative", "passive", "aste"],
                       help="aste is the causative-reflexive stem (አስተ-)")
    p_gen.add_argument("--verb_type", choices=VERB_TYPE_CHOICES,
                       help="Verb class, as the generator names it. Without it every triliteral template is used.")

    # lookup
    p_lookup = subparsers.add_parser("lookup", help="Look up a lemma in the dictionary")
    p_lookup.add_argument("word", help="Ge'ez lemma or word")

    args = parser.parse_args()
    if args.command == "analyze":
        cmd_analyze(args)
    elif args.command == "lemmatize":
        cmd_lemmatize(args)
    elif args.command == "generate":
        cmd_generate(args)
    elif args.command == "lookup":
        cmd_lookup(args)


if __name__ == "__main__":
    main()
