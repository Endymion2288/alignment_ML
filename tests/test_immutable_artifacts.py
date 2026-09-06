"""T01 immutable artifact-store tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from evaluation.artifact_store import ImmutableArtifactStore, write_json_atomic


def test_second_write_raises_file_exists(tmp_path: Path):
    dest = tmp_path / "once.json"
    write_json_atomic(dest, {"ok": True})
    with pytest.raises(FileExistsError, match="overwrite"):
        write_json_atomic(dest, {"ok": False})
    assert dest.read_text(encoding="utf-8").startswith('{\n  "ok": true')


def test_unique_run_ids_and_exclusive_directory(tmp_path: Path):
    first = ImmutableArtifactStore.begin(tmp_path, "t01_smoke")
    second = ImmutableArtifactStore.begin(tmp_path, "t01_smoke")
    assert first.run_id != second.run_id
    assert first.run_dir != second.run_dir
    assert first.run_dir.is_dir() and second.run_dir.is_dir()


def test_interrupted_run_is_not_complete(tmp_path: Path):
    store = ImmutableArtifactStore.begin(tmp_path, "t01_interrupt")
    store.write_json("partial.json", {"stage": "partial"})
    assert store.is_complete() is False
    assert not (store.run_dir / "COMPLETE.json").exists()


def test_finalize_then_rewrite_fails(tmp_path: Path):
    store = ImmutableArtifactStore.begin(tmp_path, "t01_final")
    store.write_json("payload.json", {"n": 1})
    store.finalize({"pass": True})
    assert store.is_complete() is True
    with pytest.raises(FileExistsError):
        store.finalize({"pass": False})
    with pytest.raises((FileExistsError, Exception)):
        store.write_json("payload.json", {"n": 2})


def test_same_relative_name_cannot_be_written_twice(tmp_path: Path):
    store = ImmutableArtifactStore.begin(tmp_path, "t01_dup")
    store.write_json("access_policy_audit.json", {"ok": True})
    with pytest.raises(FileExistsError):
        store.write_json("access_policy_audit.json", {"ok": False})
