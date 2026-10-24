"""Checkpoint-lineage exposure must not leak into the dev set used for selection."""
import json

import pytest

import training_mixture as TM


def _manifest(tmp_path, forms=(), roots=()):
    p = tmp_path / 'seen.json'
    p.write_text(json.dumps(dict(schema_version=1, forms=list(forms), roots=list(roots),
                                coverage='conservative eligible exposure')), encoding='utf-8')
    return p


def test_excludes_folded_forms_and_roots_but_keeps_unseen_rootless_words(tmp_path):
    manifest = _manifest(tmp_path, forms=['ሠላም'], roots=['ሠ-ለ-መ'])
    pairs = [dict(form='ሰላም', root=None), dict(form='ሌላ', root='ሰ-ለ-መ'),
             dict(form='ዝንቱ', root=None), dict(form='ቀተለ', root='ቀ-ተ-ለ')]
    stats = {}
    assert TM.exclude_checkpoint_exposure(pairs, manifest, stats) == pairs[2:]
    assert stats['pairs_before_exposure_filter'] == 4
    assert stats['pairs_excluded_seen_form'] == stats['pairs_excluded_seen_root'] == 1
    assert stats['pairs_after_exposure_filter'] == 2


def test_dev_filters_before_cap_and_removes_both_tasks(tmp_path, monkeypatch):
    manifest = _manifest(tmp_path, forms=['ሠላም'])
    pairs = [dict(form='ሰላም', root=None, type='particle'),
             dict(form='ዝንቱ', root=None, type='demonstrative')]
    monkeypatch.setattr(TM.dataset_loader, 'load_gold_split', lambda *args: pairs)
    monkeypatch.setattr(TM.dataset_loader, 'load_unified', lambda *args: [])
    stats = {}
    samples = TM.dev_samples('unused', cap=1, seen_manifest=manifest, stats=stats)
    assert len(samples) == 2
    assert {s['form'] for s in samples} == {'ዝንቱ'}
    assert {s['task'] for s in samples} == {'analyze', 'pos'}
    assert stats['pairs_after_cap'] == 1 and stats['dev_samples'] == 2


def test_invalid_manifest_fails_instead_of_silently_keeping_exposed_rows(tmp_path):
    path = tmp_path / 'invalid.json'
    path.write_text('{}')
    with pytest.raises(ValueError, match='schema_version'):
        TM.exclude_checkpoint_exposure([], path)


def test_missing_manifest_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        TM.dev_samples('unused', seen_manifest=tmp_path / 'missing.json', gold_only=True)
