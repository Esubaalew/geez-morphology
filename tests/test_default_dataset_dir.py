"""default_dataset_dir / require_dataset_dir: sibling names geez-dataset and geez-dataset-v1."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import dataset_loader as G  # noqa: E402


def _script_dir(tmp_path):
    return str(tmp_path / "geez-morphology")


def test_prefers_github_sibling_name_when_both_exist(monkeypatch, tmp_path):
    monkeypatch.delenv(G.DATASET_DIR_ENV, raising=False)
    monkeypatch.setattr(G, "SCRIPT_DIR", _script_dir(tmp_path))
    github = tmp_path / "geez-dataset"
    v1 = tmp_path / "geez-dataset-v1"
    github.mkdir()
    v1.mkdir()
    assert G.default_dataset_dir() == str(github)
    assert G.require_dataset_dir() == str(github)


def test_falls_back_to_v1_sibling_when_github_name_missing(monkeypatch, tmp_path):
    monkeypatch.delenv(G.DATASET_DIR_ENV, raising=False)
    monkeypatch.setattr(G, "SCRIPT_DIR", _script_dir(tmp_path))
    v1 = tmp_path / "geez-dataset-v1"
    v1.mkdir()
    assert G.default_dataset_dir() == str(v1)
    assert G.require_dataset_dir() == str(v1)


def test_env_overrides_siblings(monkeypatch, tmp_path):
    monkeypatch.setattr(G, "SCRIPT_DIR", _script_dir(tmp_path))
    (tmp_path / "geez-dataset").mkdir()
    other = tmp_path / "custom-dataset"
    other.mkdir()
    monkeypatch.setenv(G.DATASET_DIR_ENV, str(other))
    assert G.default_dataset_dir() == str(other)
    assert G.require_dataset_dir() == str(other)


def test_neither_sibling_returns_github_name_without_raising(monkeypatch, tmp_path):
    monkeypatch.delenv(G.DATASET_DIR_ENV, raising=False)
    monkeypatch.setattr(G, "SCRIPT_DIR", _script_dir(tmp_path))
    path = G.default_dataset_dir()
    assert os.path.basename(path) == "geez-dataset"
    assert not os.path.exists(path)


def test_require_names_both_tried_paths_and_how_to_set(monkeypatch, tmp_path):
    monkeypatch.delenv(G.DATASET_DIR_ENV, raising=False)
    monkeypatch.setattr(G, "SCRIPT_DIR", _script_dir(tmp_path))
    tried = G.sibling_dataset_dirs()
    assert [os.path.basename(p) for p in tried] == ["geez-dataset", "geez-dataset-v1"]
    with pytest.raises(FileNotFoundError, match="geez-dataset") as exc:
        G.require_dataset_dir()
    msg = str(exc.value)
    assert tried[0] in msg and tried[1] in msg
    assert G.DATASET_DIR_ENV in msg
    assert "--dataset_dir" in msg


def test_require_explicit_missing_path(monkeypatch, tmp_path):
    monkeypatch.delenv(G.DATASET_DIR_ENV, raising=False)
    missing = str(tmp_path / "not-a-dataset")
    with pytest.raises(FileNotFoundError, match="not-a-dataset"):
        G.require_dataset_dir(missing)


def test_require_github_default_path_lists_both_siblings(monkeypatch, tmp_path):
    monkeypatch.delenv(G.DATASET_DIR_ENV, raising=False)
    monkeypatch.setattr(G, "SCRIPT_DIR", _script_dir(tmp_path))
    github = G.sibling_dataset_dirs()[0]
    with pytest.raises(FileNotFoundError) as exc:
        G.require_dataset_dir(github)
    msg = str(exc.value)
    assert github in msg
    assert G.sibling_dataset_dirs()[1] in msg
    assert G.DATASET_DIR_ENV in msg
    assert "--dataset_dir" in msg
