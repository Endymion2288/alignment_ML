"""T01 access-policy tests.  Fake sealed paths are never opened."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from datasets.access_policy import (
    AccessPolicyError,
    AccessScope,
    FrozenEvaluationCapability,
    authorize_path,
    authorize_split,
    load_curriculum_for_scope,
    refuse_allow_sealed_test_flag,
)
from datasets.physical_curriculum import SYNTHETIC_CORPUS_SCHEMA
from scripts.run_global_assignment_mlp_baseline import _load_manifest_for_scope
from scripts.run_station_pair_threshold_baseline import (
    _load_manifest_for_scope as _station_pair_load,
)


def _boom(*_args, **_kwargs):
    raise AssertionError("opener/resolver must not be called for sealed paths")


def _sample(tmp_path: Path, split: str, source_id: str, *, existing: bool) -> dict:
    paths = {}
    for name in (
        "physical_tracklets",
        "physical_propagations",
        "physical_payload_manifest",
        "synthetic_tracklets",
        "field_candidates",
    ):
        path = tmp_path / f"{source_id}_{name}.root"
        if existing:
            path.touch()
        paths[name] = str(path)
    return {
        "source_id": source_id,
        "source_ids": [source_id],
        "split": split,
        "payload_id": f"mag_0_{split}",
        "magnitude_mm": 0.0,
        "direction_trial": split,
        "injected_offsets_xy_mm": {"0": [0.0, 0.0]},
        "source_event_uids": [f"{source_id}:100012:0"],
        "physical_event_uids": [f"{source_id}:100012:0"],
        "physical_geometry_repropagation": True,
        **paths,
    }


def test_fake_sealed_path_never_opens_or_resolves():
    with pytest.raises(AccessPolicyError, match="sealed/test"):
        authorize_path(
            "/does/not/exist/sealed_test/events.root",
            AccessScope.TRAIN,
            opener=_boom,
            resolver=_boom,
        )
    with pytest.raises(AccessPolicyError, match="sealed"):
        authorize_path(
            "/tmp/missing/test/tracklets.root",
            AccessScope.DEVELOPMENT_VALIDATION,
            split="test",
            opener=_boom,
            resolver=_boom,
        )


def test_train_and_validation_paths_may_resolve(tmp_path: Path):
    path = tmp_path / "train" / "tracklets.root"
    path.parent.mkdir()
    path.touch()
    called = {}

    def resolver(item):
        called["resolver"] = True
        return Path(item)

    resolved = authorize_path(path, AccessScope.TRAIN, split="train", resolver=resolver)
    assert called["resolver"] is True
    assert resolved == path


def test_all_scopes_refuse_test_split():
    capability = FrozenEvaluationCapability(
        plan_sha256="a" * 64,
        checkpoint_sha256="b" * 64,
        calibration_sha256="c" * 64,
        source_set_sha256="d" * 64,
        unseal_allowed=True,
    )
    for scope in AccessScope:
        with pytest.raises(AccessPolicyError, match="sealed"):
            authorize_split("test", scope, capability=capability)


def test_allow_sealed_test_flag_is_not_a_license():
    with pytest.raises(AccessPolicyError, match="not a development license"):
        refuse_allow_sealed_test_flag()


def test_curriculum_scope_skips_missing_test_assets(tmp_path: Path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": SYNTHETIC_CORPUS_SCHEMA,
                "physical_geometry_repropagation": True,
                "q_over_p_mode": 0,
                "samples": [
                    _sample(tmp_path, "train", "source_train", existing=True),
                    _sample(tmp_path, "validation", "source_validation", existing=True),
                    _sample(tmp_path, "test", "sealed_test", existing=False),
                ],
            }
        ),
        encoding="utf-8",
    )
    _, samples, _ = load_curriculum_for_scope(
        manifest, AccessScope.DEVELOPMENT_VALIDATION
    )
    assert {sample.split for sample in samples} == {"train", "validation"}


def test_legacy_train_validation_loaders_still_fail_closed(tmp_path: Path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": SYNTHETIC_CORPUS_SCHEMA,
                "physical_geometry_repropagation": True,
                "q_over_p_mode": 0,
                "samples": [
                    _sample(tmp_path, "train", "source_train", existing=True),
                    _sample(tmp_path, "validation", "source_validation", existing=True),
                    _sample(tmp_path, "test", "sealed_test", existing=False),
                ],
            }
        ),
        encoding="utf-8",
    )
    _, mlp_samples, _ = _load_manifest_for_scope(manifest, validation_only=True)
    _, pair_samples, _ = _station_pair_load(manifest, evaluate_test=False)
    assert {sample.split for sample in mlp_samples} == {"train", "validation"}
    assert {sample.split for sample in pair_samples} == {"train", "validation"}
    with pytest.raises(AccessPolicyError):
        _load_manifest_for_scope(manifest, validation_only=False)
    with pytest.raises(AccessPolicyError):
        _station_pair_load(manifest, evaluate_test=True)
