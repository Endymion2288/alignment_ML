from __future__ import annotations

import json
import sys

import pytest

from datasets.physical_curriculum import SYNTHETIC_CORPUS_SCHEMA
from scripts.run_global_assignment_mlp_baseline import _load_manifest_for_scope
from scripts.run_station_pair_threshold_baseline import (
    _load_manifest_for_scope as _station_pair_load_manifest_for_scope,
)
from scripts.refreeze_transformer_validation_operating_point import (
    DEFAULT_ROUTE_SELECTION_POLICY,
    _validate_manifest_contract,
)
from scripts.train_geometry_aware_transformer_v1 import _reference_artifacts
from scripts.train_geometry_aware_transformer_v1 import (
    DEFAULT_ROUTE_SELECTION_POLICY as TRAIN_V1_DEFAULT_ROUTE_SELECTION_POLICY,
)


def _sample(
    tmp_path,
    split: str,
    source_id: str,
    *,
    existing: bool,
    magnitude_mm: float = 0.0,
) -> dict[str, object]:
    paths: dict[str, str] = {}
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
        "payload_id": f"mag_{magnitude_mm:g}_{split}",
        "magnitude_mm": magnitude_mm,
        "direction_trial": split,
        "injected_offsets_xy_mm": {"0": [0.0, 0.0]},
        "source_event_uids": [f"{source_id}:100012:0"],
        "physical_event_uids": [f"{source_id}:100012:0"],
        "physical_geometry_repropagation": True,
        **paths,
    }


def test_validation_only_global_mlp_loader_does_not_resolve_sealed_test_assets(tmp_path):
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

    _, samples, _ = _load_manifest_for_scope(manifest, validation_only=True)

    assert {sample.split for sample in samples} == {"train", "validation"}


def test_v1_reference_audit_never_resolves_a_sealed_test_path():
    result = _reference_artifacts({"frozen_route_test": "does/not/exist"})

    assert result == {"frozen_route_test": {"excluded": True, "reason": "sealed_test_boundary"}}


def _three_split_manifest(tmp_path) -> object:
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
    return manifest


def test_station_pair_loader_defaults_to_the_sealed_test_boundary(tmp_path):
    manifest = _three_split_manifest(tmp_path)

    _, samples, _ = _station_pair_load_manifest_for_scope(manifest, evaluate_test=False)

    assert {sample.split for sample in samples} == {"train", "validation"}


def test_station_pair_loader_resolves_test_only_with_evaluate_test(tmp_path):
    manifest = _three_split_manifest(tmp_path)

    with pytest.raises(FileNotFoundError, match="sealed_test"):
        _station_pair_load_manifest_for_scope(manifest, evaluate_test=True)


def test_global_assignment_loader_forbids_test_unless_explicitly_permitted(tmp_path):
    manifest = _three_split_manifest(tmp_path)

    _, samples, _ = _load_manifest_for_scope(manifest, validation_only=True)

    assert {sample.split for sample in samples} == {"train", "validation"}
    with pytest.raises(FileNotFoundError, match="sealed_test"):
        _load_manifest_for_scope(manifest, validation_only=False)


def test_curriculum_mlp_baseline_refuses_the_sealed_test_by_default(tmp_path, monkeypatch):
    from scripts.run_curriculum_mlp_baseline import main as curriculum_main

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_curriculum_mlp_baseline.py",
            "--synthetic-manifest",
            str(tmp_path / "manifest.json"),
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )

    with pytest.raises(ValueError, match="sealed test"):
        curriculum_main()


def test_validation_refreeze_contract_accepts_a_train_validation_only_corpus(tmp_path):
    """The refreeze path remains usable before a new test bank exists."""
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": SYNTHETIC_CORPUS_SCHEMA,
                "physical_geometry_repropagation": True,
                "q_over_p_mode": 0,
                "samples": [
                    _sample(
                        tmp_path,
                        split,
                        source_id,
                        existing=True,
                        magnitude_mm=magnitude,
                    )
                    for split, source_id in (
                        ("train", "source_train"),
                        ("validation", "source_validation"),
                    )
                    for magnitude in (0.0, 0.1, 1.0, 5.0, 10.0, 50.0)
                ],
            }
        ),
        encoding="utf-8",
    )

    from datasets.physical_curriculum import load_synthetic_curriculum_manifest

    _, samples, loaded_manifest = load_synthetic_curriculum_manifest(
        manifest,
        require_all_splits=False,
        allowed_splits=("train", "validation"),
    )
    root = {"refit": {"q_over_p_mode": 0}}
    transformer = {"q_over_p_mode": 0, "candidate_chi2_gate": None}

    _validate_manifest_contract(root, transformer, loaded_manifest, list(samples))


def test_validation_refreeze_has_a_documented_route_policy_default():
    """Expanded controls may omit the legacy metadata-only policy field."""
    assert DEFAULT_ROUTE_SELECTION_POLICY == (
        "nominal_primary_then_validation_capture_count_then_maximum_magnitude_"
        "then_mean_route_quality"
    )


def test_v1_training_has_the_same_documented_route_policy_default():
    assert TRAIN_V1_DEFAULT_ROUTE_SELECTION_POLICY == DEFAULT_ROUTE_SELECTION_POLICY
