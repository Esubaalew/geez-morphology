#!/usr/bin/env python3
"""the page-checked gold corrections: install them, check them, show what they change.

  install  <her folder>   Copy her three files (gold_page_table.tsv, gold_extra_fixes.tsv,
                          proposal_page_outcomes.tsv) verbatim into gold_corrections/sources/, the
                          folder the trainer and evaluate_gold.py read (gold_filters.PageCorrections).
  validate [--her_folder F] [--report problems.tsv]
                          Problems for the maintainers (her files are never changed). Exit 1 if any row
                          cannot be applied or two sources disagree.
  counts   [--json out.json]
                          Gold pairs per split as the trainer / evaluate_gold load them: kept, changed
                          by the page, restored, still out (and why).

Keys: every source row is resolved to one gold pair (split, record_id, row, form); see gold_filters.
Shared options: --dataset_dir ../geez-dataset-v1  --page_dir gold_corrections
"""
import argparse
import collections
import json
import os
import re
import shutil
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import geez_config  # noqa: E402
import gold_filters as G  # noqa: E402

SPLITS = ("train", "val", "test")
LATIN = re.compile(r"[A-Za-z]")


def gold_file_of(dataset_dir, split):
    """data/paradigms/merho_ges_gold_<split>.jsonl."""
    p = os.path.join(dataset_dir, "data", "paradigms", G.gold_file_name(split))
    if os.path.exists(p):
        return p
    raise FileNotFoundError(f"no gold file for {split}: {p}")


def load_records(dataset_dir):
    return {s: [json.loads(l) for l in open(gold_file_of(dataset_dir, s), encoding="utf-8")] for s in SPLITS}


class GoldPairs:
    """Every "/"-variant of every gold form cell (clean words and rejected text) with its key."""

    def __init__(self, records, dataset_dir):
        self.pairs, self.copies = {}, collections.Counter()
        self.rows = collections.defaultdict(list)
        for split, recs in records.items():
            gold_file = os.path.basename(gold_file_of(dataset_dir, split))
            for line, rec in enumerate(recs, 1):
                rid = G.record_id(rec)
                for j, row in enumerate(rec.get("table_data", [])):
                    kept, rejected = G.clean_variants(row.get("geez_form"))
                    for n, v in enumerate(kept + rejected):
                        key = (split, rid, j, v)
                        self.copies[key] += 1
                        self.rows[key[:3]].append(v)
                        self.pairs.setdefault(key, {"rec": rec, "row": row, "gold_file": gold_file,
                                                    "file_line": line, "variant": n, "clean": v in kept})


def install(folder, page_dir):
    os.makedirs(os.path.join(page_dir, "sources"), exist_ok=True)
    for name in G.PAGE_SOURCES + (G.FLAGGED_SOURCE,):
        src = os.path.join(folder, name)
        if os.path.exists(src):
            shutil.copyfile(src, os.path.join(page_dir, "sources", name))
            print(f"installed {name} ({sum(1 for _ in open(src, encoding='utf-8')) - 1} rows)")
        else:
            print(f"[!] {name} not in {folder}: the installed copy (if any) is kept")


# ------------------------------------------------------------------ validate

def old_label(row):
    return f"{row.get('tense', '')} {row.get('person', '')}{row.get('gender', '')}{row.get('number', '')}".strip()


def consonants(form):
    return {geez_config.decompose_char(c)[0] for c in form if geez_config.decompose_char(c)}


def missing_letters(form, root, old_root):
    """Letters of `root` that replace a letter of `old_root` but do not occur in `form`
    (ጠ-ቀ-ሰ -> ጠ-ቀ-ለ on the form ጠቀሰ: ለ). Weak letters (ወ, የ, አ, ዐ) can vanish, so these are checks."""
    old = old_root.split("-") if old_root else []
    new = [x for x in root.split("-") if x not in old]
    have = consonants(form)
    return [x for x in new if geez_config.decompose_char(x)[0] not in have]


# Labiovelar series: ቈ = ቀ+ወ, ኈ = ኀ+ወ, ኰ = ከ+ወ, ጐ = ገ+ወ (first letter of each 8-letter block).
LABIOVELAR = {"ቀ": 0x1248, "ኀ": 0x1288, "ከ": 0x12B0, "ገ": 0x1310}
WEAK_RADICALS = ("ወ", "የ")


def one_letter_change(old_root, new_root):
    """True if the two roots have the same number of radicals and differ in exactly one."""
    a, b = (old_root or "").split("-"), (new_root or "").split("-")
    return bool(old_root) and len(a) == len(b) and sum(x != y for x, y in zip(a, b)) == 1


def explain_missing(form, root, gone):
    """Why letters of `root` missing from `form` are expected, or None.
    - labialized letter: ጐ writes ገ+ወ in one letter (also ቈ ኰ ኈ), so ገ (and ወ) are in ጐ;
    - weak radical: ወ/የ of the root is not written in a contracted or weak form (ቆመ, አብሐ, ተከሰ)."""
    labial = {b for b, start in LABIOVELAR.items() if any(start <= ord(c) < start + 8 for c in form)}
    reasons = []
    for x in gone:
        if x in labial or (x == "ወ" and labial & set(root.split("-"))):
            reasons.append(f"{x}: labialized letter (ጐ = ገ+ወ) in the form")
        elif x in WEAK_RADICALS:
            reasons.append(f"{x}: weak radical, not written in this form")
        else:
            return None
    return "; ".join(reasons)


def shared_roots(corr, records):
    """Roots (as loaded) that occur in more than one split: {root: {split: (pairs, [headwords])}}."""
    where = collections.defaultdict(lambda: collections.defaultdict(set))
    n = collections.Counter()
    for s in SPLITS:
        for p in corr.gold_pairs(records[s], s):
            where[p["root"]][s].add(p["record"].get("title_or_lemma") or "")
            n[(p["root"], s)] += 1
    return {r: {s: (n[(r, s)], sorted(h)) for s, h in d.items()} for r, d in where.items() if len(d) > 1}


def synthetic_root_counts(corr, dataset_dir):
    """Synthetic train pairs per root, as the trainer writes the root (1st order)."""
    out = collections.Counter()
    with open(os.path.join(dataset_dir, "data", "paradigms", "synthetic_morphology_train.jsonl"), encoding="utf-8") as f:
        for line in f:
            d, _ = corr.synthetic_record(json.loads(line))
            if d is not None:
                root = (d.get("features") or {}).get("root", "")
                out[geez_config.first_order_root(geez_config.to_ethiopic_root(root))] += 1
    return out


def validate(dataset_dir, page_dir, her_folder=None, report=None):
    """-> [(where, form, level, problem)]. level "error": not applied / stays out; "check": applied, have a look;
    "ok": accepted by a rule (the reason says which); listed, not counted as to check."""
    records = load_records(dataset_dir)
    gold = GoldPairs(records, dataset_dir)
    corr = G.Corrections.from_dataset(dataset_dir, required=True, page_dir=page_dir)
    base = G.Corrections.from_dataset(dataset_dir, required=True, page_dir=None)
    page = corr.page
    problems = []
    add = lambda where, form, level, msg: problems.append((where, form, level, msg))
    if page is None:
        add("-", "", "error", f"no sources in {page_dir}/sources")
        return problems

    for name, line, form, probs in page.problems:
        for m in probs:
            add(f"{name}:{line}", form, "error", m)

    loaded = {}                                   # pairs loaded without the page corrections
    for s in SPLITS:
        for p in base.gold_pairs(records[s], s):
            loaded[(s, p["record_id"], p["row"], p["source_form"])] = p

    # source rows: key resolves to exactly one gold pair; source-specific cross-checks
    for name in G.PAGE_SOURCES:
        path = os.path.join(page_dir, "sources", name)
        if not os.path.exists(path):
            continue
        cols = open(path, encoding="utf-8").readline().rstrip("\n").split("\t")
        if name == "gold_page_table.tsv" and not {"stem", "stem_fix"} & set(cols):
            add(name, "", "check", "no stem_fix column: verb stems stay as the gold record has them")
        for line, e in enumerate(G._read_tsv(path), 2):
            e, where = G._clean(e), f"{name}:{line}"
            key, _, probs = G.NORMALIZERS[name](e, page.key_maps[name])
            if key is None:
                continue                                    # already listed with page.problems
            form = e.get("form") or key[3]
            if key not in gold.pairs:
                forms = gold.rows.get(key[:3])
                add(where, form, "error", f"matches no gold pair: {key[0]} record {key[1]} row {key[2]} "
                    + (f"has {' / '.join(forms)}, not {key[3]!r}" if forms else "does not exist"))
                continue
            info = gold.pairs[key]
            if gold.copies[key] > 1:
                add(where, form, "check", f"the gold file has this pair {gold.copies[key]} times (same record, row "
                    "and form); the correction applies to every copy")
            if name == "gold_extra_fixes.tsv":
                for col, want in (("gold_file", info["gold_file"]), ("file_line", str(info["file_line"])),
                                  ("variant", str(info["variant"]))):
                    got = G.canonical_gold_file(e.get(col, "")) if col == "gold_file" else e.get(col)
                    if e.get(col) and got != want:
                        add(where, form, "error", f"{col} {e[col]!r} does not match record {key[1]} row {key[2]} "
                            f"({col} is {want!r}): which pair is meant?")
            if name == "gold_page_table.tsv":
                was = loaded.get(key)
                raw_root = info["rec"].get("root") or ""
                if e.get("root") and e["root"] not in (raw_root, geez_config.first_order_root(raw_root),
                                                       was["root"] if was else None):
                    add(where, form, "check", f"root column {e['root']!r} is not the gold root {raw_root!r}"
                        + (f" or the loaded root {was['root']!r}" if was else ""))
                if (e.get("old_label") or "none") != (old_label(info["row"]) or "none"):
                    add(where, form, "check", f"old_label {e['old_label']!r} is not the gold label {old_label(info['row'])!r}")
                if e.get("root") and e["root"] != geez_config.first_order_root(e["root"]):
                    add(where, form, "check", f"root {e['root']!r} is not in 1st order ({geez_config.first_order_root(e['root'])})")
                if "stem" in e.get("note", "").lower() and not (e.get("stem_fix") or e.get("stem")):
                    add(where, form, "check", f"note mentions a stem but no stem is given: {e['note']!r}")
            for col in ("form", "root"):
                if LATIN.search(e.get(col, "")):
                    add(where, form, "check", f"Latin letters in {col} {e[col]!r}")
            if e.get("form_fix") and e["form_fix"] == key[3]:
                add(where, form, "check", "form_fix is the same as the form")

    # flagged_rows.tsv: these pairs stay out
    src_rows = {}
    for name in G.PAGE_SOURCES:
        path = os.path.join(page_dir, "sources", name)
        if os.path.exists(path):
            for line, e in enumerate(G._read_tsv(path), 2):
                src_rows[f"{name}:{line}"] = G._clean(e)
    value_cols = G.FIX_COLUMNS + G.LABEL_COLUMNS + ("stem_fix", "new_label", "outcome_label")

    def gives_values(w):
        e = src_rows.get(w, {})
        return any((e.get(c) or "").strip() for c in value_cols if c != "root_fix_scope")
    for key, where in sorted(page.flagged.items(), key=lambda kv: kv[1]):
        listed = [x for x in page.entries[key]["sources"] if x != where]
        others = [x for x in listed if gives_values(x)]
        if listed and not others:
            add(where, key[3], "ok", f"accepted: flagged ({key[0]} record {key[1]} row {key[2]} stays out); also listed "
                f"in {', '.join(listed)} with a note only (no values)")
        elif others:
            add(where, key[3], "check", f"conflict for checking: flagged ({key[0]} record {key[1]} row {key[2]} "
                f"stays out) but also given values in {', '.join(others)} (not applied, the flag wins): "
                "drop the other row, or unflag")
        else:
            add(where, key[3], "check", f"flagged: {key[0]} record {key[1]} row {key[2]} stays out"
                + (" (it loaded before)" if key in loaded else ""))

    # merged pairs: conflicts, forms that cannot load, pairs that stay out
    for key, e in page.entries.items():
        where, form = ", ".join(e["sources"]), key[3]
        for field, values in page.conflicts.get(key, ()):
            add(where, form, "error", f"sources disagree on {field}: "
                + "; ".join(f"{v or '(empty)'} in {', '.join(src)}" for v, src in values.items()) + " - pair stays out")
        info = gold.pairs.get(key)
        if info is None or e["status"] != "confirmed":
            continue
        if not info["clean"] and not (e["checked"] and e["form_fix"]):
            add(where, form, "error", "the gold cell is not one Ethiopic word here; it loads only with a page-checked form_fix")
        if key not in loaded and info["clean"]:
            excluded = G.form_key(key[3]) in base.excluded.get(key[1:3], ())
            if excluded and not e["checked"]:
                add(where, form, "check", "pair is excluded (relabel proposal); it comes back only when page-checked")
            if not excluded and not (e["checked"] and (e["root_fix"] or page.record_fixes.get(key[:2]))):
                add(where, form, "check", "the root tables skip this row; it comes back only with a page-checked root_fix")

    # record roots: one value -> record-wide (if record_root_applies) ; several -> each pair its own
    for (split, rid), fixes in sorted(page.record_fixes.items()):
        rec = next(r for r in records[split] if G.record_id(r) == rid)
        current, usable = corr._current_root(rec, rid)
        pairs = [k for k in gold.pairs if k[:2] == (split, rid) and gold.pairs[k]["clean"]]
        covered = [k for k in pairs if k in page.entries]
        where = "; ".join(f"{r}: {', '.join(src)}" for r, src in fixes.items())
        if len(fixes) > 1:
            no_root = [k[3] for k in pairs if not (k in page.entries and page.entries[k]["root_fix"])]
            add(where, rid, "check", f"record {rid} ({split}) is split by the page into {' / '.join(fixes)}: each pair "
                f"takes its own root_fix" + (f"; {len(no_root)} pairs have none and keep the root tables' decision: "
                                             f"{' '.join(no_root)}" if no_root else ""))
            continue
        root = next(iter(fixes))
        applied = page.record_root(split, rid, current, usable)
        uncovered = [k[3] for k in pairs if k not in covered]
        if applied is None:
            if uncovered:
                add(where, rid, "check", f"root_fix {root} is not a letter correction of record {rid}'s root "
                    f"{current or '(none)'}: applied to its own pair(s) only; {len(uncovered)} other pairs keep "
                    f"{current or 'the root tables decision'} ({' '.join(uncovered)}). Add root_fix_scope=record if "
                    f"the whole record is {root}, or root_fix_scope=row to confirm")
            continue
        state = (f"already {root} in the root tables" if current == root else
                 f"{current} -> {root}" if usable else f"{root} (the root tables skip this record)")
        if not usable and page.record_root_explicit(split, rid):
            add(where, rid, "check", f"record {rid} ({split}) root {state}: root_fix_scope=record, so all "
                f"{len(pairs)} pairs come back ({len(uncovered)} with the gold labels)")
        elif not usable:
            add(where, rid, "check", f"record {rid} ({split}) root {state}: {len(pairs) - len(uncovered)} page-checked "
                "pairs come back" + (f"; {len(uncovered)} not page-checked stay out: {' '.join(uncovered)}" if uncovered else ""))
        elif current != root and not uncovered and one_letter_change(current, root):
            add(where, rid, "ok", f"accepted: record {rid} ({split}) root {state} for all {len(pairs)} pairs: "
                "a one-letter page-checked correction, every pair listed in her sources")
        elif current != root:
            add(where, rid, "check", f"record {rid} ({split}) root {state} for all {len(pairs)} pairs"
                + (f", including {len(uncovered)} no source lists: {' '.join(uncovered)}" if uncovered else ""))
        for k in pairs:
            e = page.entries.get(k)
            form = e["form_fix"] if e and e["form_fix"] else k[3]
            whole = page.record_root_explicit(split, rid)
            if (e and e["status"] != "confirmed") or (not usable and not whole and k not in covered):
                continue
            gone = missing_letters(form, root, current if usable else "")
            own = e["root_fix"] if e and e["checked"] and e["root_fix"] and e["root_fix_scope"] == "row" else ""
            why = explain_missing(form, root, gone) if gone else None
            if gone and own and own != root:
                add(where, form, "ok", f"accepted: record root {root} does not reach {form!r}: the pair has its own "
                    f"page-checked root {own} (root_fix_scope=row)")
            elif gone and why:
                add(where, form, "ok", f"accepted: record root {root} on {form!r}: {why}")
            elif gone:
                add(where, form, "check", f"record root {root} reaches {form!r}, which has no {'/'.join(gone)}: "
                    "is the form misread (form_fix), or is this pair's root different (root_fix_scope=row)?")

    # self-check: a record-wide root really reaches every loaded pair of the record
    for s in SPLITS:
        for p in corr.gold_pairs(records[s], s):
            fixes = page.record_fixes.get((s, p["record_id"]))
            e = page.entries.get((s, p["record_id"], p["row"], p["source_form"]))
            if fixes and len(fixes) == 1 and p["root_from_page"] and not (e and e["root_fix"]):
                if p["root"] != next(iter(fixes)):
                    add(s, p["form"], "error", f"loader bug: record root {next(iter(fixes))} not applied ({p['root']})")

    # split leakage: roots shared by train / val / test, before and after the page corrections
    before, after = shared_roots(base, records), shared_roots(corr, records)
    add("splits", "", "check", f"roots in more than one split: {len(before)} without the page corrections, "
        f"{len(after)} with them (the splits are not changed)")
    synthetic = synthetic_root_counts(corr, dataset_dir) if (before or after) else {}
    for root in sorted(set(before) | set(after)):
        d = after.get(root) or before[root]
        state = "new" if root not in before else "gone" if root not in after else "as before"
        add("splits", root, "check", f"{state}: " + "; ".join(f"{s} {n} pairs ({' / '.join(h)})"
                                                            for s, (n, h) in sorted(d.items()))
            + f"; synthetic {synthetic.get(root, 0)} pairs")

    if her_folder and os.path.isdir(her_folder):
        for name in sorted(os.listdir(her_folder)):
            if name.endswith(".tsv") and name not in G.PAGE_SOURCES + (G.FLAGGED_SOURCE, "page_check_log.tsv"):
                add(name, "", "check", f"{name} is in her folder but is not read (the loader reads "
                    f"{', '.join(G.PAGE_SOURCES + (G.FLAGGED_SOURCE,))})")

    if report:
        with open(report, "w", encoding="utf-8") as f:
            f.write("where\tform\tlevel\tproblem\n")
            for p in problems:
                f.write("\t".join(str(x).replace("\t", " ") for x in p) + "\n")
    errors = sum(p[2] == "error" for p in problems)
    accepted = sum(p[2] == "ok" for p in problems)
    print(f"{page_dir}/sources: {dict(page.rows)} -> {len(page.entries)} gold pairs; "
          f"{errors} errors, {len(problems) - errors - accepted} to check, {accepted} accepted (rule given)")
    for p in problems:
        print(f"  {p[2]:5s}  {p[0]}  {p[1]}  {p[3]}")
    return problems


# ------------------------------------------------------------------ counts

def counts(dataset_dir, page_dir):
    records = load_records(dataset_dir)
    gold = GoldPairs(records, dataset_dir)
    with_page = G.Corrections.from_dataset(dataset_dir, required=True, page_dir=page_dir)
    without = G.Corrections.from_dataset(dataset_dir, required=True, page_dir=None)
    page = with_page.page
    if page is None:
        sys.exit(f"[!] no page corrections in {page_dir}/sources")
    print(f"page corrections: {dict(page.rows)} -> {len(page.entries)} pairs "
          f"({len(page.problems)} rows not applied, {len(page.conflicts)} pairs with disagreeing sources)")
    result = {}
    for s in SPLITS:
        groups = [("all records (trainer)", records[s])]
        if s != "train":
            groups.append(("verb_conjugation (evaluate_gold)", [r for r in records[s] if r.get("category") == "verb_conjugation"]))
        for label, recs in groups:
            st = collections.Counter()
            new = with_page.gold_pairs(recs, s, st)
            old = without.gold_pairs(recs, s)
            ids = {G.record_id(r) for r in recs}
            in_file = collections.Counter({k: n for k, n in gold.copies.items() if k[0] == s and k[1] in ids})
            got = collections.Counter((s, p["record_id"], p["row"], p["source_form"]) for p in new)
            loaded_rows = {(p["record_id"], p["row"]) for p in old}
            out = collections.Counter()
            for k in (in_file - got).elements():
                e, t = page.entries.get(k), with_page.gold.get(k[1])
                rec = gold.pairs[k]["rec"]
                current, usable = with_page._current_root(rec, k[1])
                page_root = page.record_root(s, k[1], current, usable)
                if e is not None and e["status"] in ("flag", "conflict"):
                    out["flagged on the page" if e["status"] == "flag" else "page sources disagree"] += 1
                elif not gold.pairs[k]["clean"]:
                    out["not one Ethiopic word (notes, phrase): needs a page-checked form_fix"] += 1
                elif page_root is not None and not usable and not page.record_root_explicit(s, k[1]):
                    out["record root fixed on the page, pair not page-checked yet"] += 1
                elif t and t["action"] == "skip":
                    out[f"root table skip: {t['group']}"] += 1
                elif t and t["action"] == "split" and k[1:3] not in loaded_rows:
                    out["root table split: form not settled"] += 1
                elif not usable:
                    out[G.root_problem(current) or "guard"] += 1
                elif G.form_key(k[3]) in with_page.excluded.get(k[1:3], ()):
                    out["relabel proposal pending (excluded), not page-checked"] += 1
                else:
                    out["other"] += 1
            changed = sum(1 for p in new if p["page"] is not None or p["root_from_page"])
            by_type = collections.Counter(p["type"] for p in new)
            print(f"\n[{s}: {label}] {sum(in_file.values())} gold pairs in the file (\"/\"-variants of the form cells)")
            print(f"  loaded without the page corrections {len(old):5d}")
            print(f"  loaded with the page corrections    {len(new):5d}   types {dict(by_type)}")
            print(f"  of these, changed by the page       {changed:5d}")
            for k, v in sorted(st.items()):
                if k.startswith("page ") or k.endswith("page_root_fix"):
                    print(f"    {k:66s} {v:5d}")
            print(f"  still out                           {sum(out.values()):5d}")
            for k, v in out.most_common():
                print(f"    {k:66s} {v:5d}")
            result[f"{s} {label}"] = {"file": sum(in_file.values()), "without": len(old), "with": len(new),
                                      "changed": changed, "types": dict(by_type), "still_out": dict(out),
                                      "page_stats": {k: v for k, v in st.items() if k.startswith("page ")}}
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["install", "validate", "counts"])
    ap.add_argument("folder", nargs="?", help="install: her folder with the three files")
    ap.add_argument("--dataset_dir", default=os.path.join(REPO, "..", "geez-dataset-v1"))
    ap.add_argument("--page_dir", default=G.PAGE_DIR)
    ap.add_argument("--her_folder", default=None, help="validate: also list .tsv files there that are not read")
    ap.add_argument("--report", default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    if a.command == "install":
        if not a.folder:
            ap.error("install needs her folder")
        install(a.folder, a.page_dir)
    elif a.command == "validate":
        problems = validate(a.dataset_dir, a.page_dir, a.her_folder, a.report)
        return 1 if any(p[2] == "error" for p in problems) else 0
    else:
        res = counts(a.dataset_dir, a.page_dir)
        if a.json:
            with open(a.json, "w", encoding="utf-8") as f:
                json.dump(res, f, ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
