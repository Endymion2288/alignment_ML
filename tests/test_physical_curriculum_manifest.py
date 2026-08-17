from __future__ import annotations

import json

from datasets.physical_curriculum import (
    SYNTHETIC_CORPUS_SCHEMA,
    load_synthetic_curriculum_manifest,
)


def test_curriculum_manifest_enforces_source_disjoint_physical_assets(tmp_path):
    samples = []
    for split, source_id in (("train", "source_train"), ("validation", "source_val"), ("test", "source_test")):
        paths = {}
        for name in (
            "physical_tracklets",
            "physical_propagations",
            "physical_payload_manifest",
            "synthetic_tracklets",
            "field_candidates",
        ):
            path = tmp_path / f"{source_id}_{name}.root"
            path.touch()
            paths[name] = str(path)
        samples.append(
            {
                "source_id": source_id,
                "split": split,
                "payload_id": "mag_0",
                "magnitude_mm": 0.0,
                "direction_trial": "unit",
                "injected_offsets_xy_mm": {"0": [0.0, 0.0]},
                "source_event_uids": [f"{source_id}:100012:0"],
                "physical_event_uids": [f"{source_id}:100012:0"],
                "physical_geometry_repropagation": True,
                **paths,
            }
        )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": SYNTHETIC_CORPUS_SCHEMA,
                "physical_geometry_repropagation": True,
                "samples": samples,
            }
        ),
        encoding="utf-8",
    )

    _, loaded, _ = load_synthetic_curriculum_manifest(manifest)

    assert [sample.source_id for sample in loaded] == ["source_train", "source_val", "source_test"]
    assert loaded[0].source_event_uids == ("source_train:100012:0",)
    assert loaded[0].source_ids == ("source_train",)
    assert loaded[0].physical_event_uids == ("source_train:100012:0",)


def test_test_only_manifest_can_be_loaded_only_with_explicit_opt_out(tmp_path):
    paths = {}
    for name in (
        "physical_tracklets",
        "physical_propagations",
        "physical_payload_manifest",
        "synthetic_tracklets",
        "field_candidates",
    ):
        path = tmp_path / f"source_test_{name}.root"
        path.touch()
        paths[name] = str(path)
    manifest = tmp_path / "test_only_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": SYNTHETIC_CORPUS_SCHEMA,
                "physical_geometry_repropagation": True,
                "samples": [
                    {
                        "source_id": "source_test",
                        "split": "test",
                        "payload_id": "mag_0_test_00",
                        "magnitude_mm": 0.0,
                        "direction_trial": "test_00",
                        "injected_offsets_xy_mm": {"0": [0.0, 0.0]},
                        "source_event_uids": ["source_test:100012:0"],
                        "physical_event_uids": ["source_test:100012:0"],
                        "physical_geometry_repropagation": True,
                        **paths,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    try:
        load_synthetic_curriculum_manifest(manifest)
    except ValueError as error:
        assert "split" in str(error)
    else:  # pragma: no cover - default isolation contract must stay strict
        raise AssertionError("default manifest loader accepted a partial split manifest")

    _, loaded, _ = load_synthetic_curriculum_manifest(manifest, require_all_splits=False)
    assert len(loaded) == 1
    assert loaded[0].split == "test"


def test_allowed_split_boundary_does_not_resolve_excluded_test_assets(tmp_path):
    """A post-test study can read validation without opening sealed test paths."""
    validation_paths = {}
    for name in (
        "physical_tracklets",
        "physical_propagations",
        "physical_payload_manifest",
        "synthetic_tracklets",
        "field_candidates",
    ):
        path = tmp_path / f"source_validation_{name}.root"
        path.touch()
        validation_paths[name] = str(path)

    def sample(split: str, source_id: str, paths: dict[str, str]) -> dict[str, object]:
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

    # Deliberately nonexistent test assets model a sealed source which must
    # never be touched by a train/validation-only post-test study.
    test_paths = {
        name: str(tmp_path / f"sealed_test_{name}.root")
        for name in validation_paths
    }
    manifest = tmp_path / "split_boundary.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": SYNTHETIC_CORPUS_SCHEMA,
                "physical_geometry_repropagation": True,
                "samples": [
                    sample("validation", "source_validation", validation_paths),
                    sample("test", "source_test", test_paths),
                ],
            }
        ),
        encoding="utf-8",
    )

    _, loaded, _ = load_synthetic_curriculum_manifest(
        manifest,
        require_all_splits=False,
        allowed_splits=("validation",),
    )
    assert [entry.split for entry in loaded] == ["validation"]


def test_manifest_loader_accepts_explicit_ift_ry_condition_without_mm_relabelling(tmp_path):
    assets = {}
    for name in (
        "physical_tracklets",
        "physical_propagations",
        "physical_payload_manifest",
        "synthetic_tracklets",
        "field_candidates",
    ):
        path = tmp_path / f"{name}.root"
        path.touch()
        assets[name] = str(path)
    manifest = tmp_path / "rotation_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": SYNTHETIC_CORPUS_SCHEMA,
                "physical_geometry_repropagation": True,
                "samples": [
                    {
                        "source_id": "pooled_train",
                        "source_ids": ["train_source"],
                        "split": "train",
                        "payload_id": "ry_p40",
                        "condition_axis": "ift_ry_mrad",
                        "condition_value": 40.0,
                        "condition_magnitude": 40.0,
                        "direction_trial": "positive",
                        "injected_station_transforms": {
                            "0": [0.0, 0.0, 0.0, 0.0, 0.04, 0.0],
                            "1": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            "2": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            "3": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        },
                        "source_event_uids": ["train_source:1:2"],
                        "physical_event_uids": ["train_source:1:2"],
                        "physical_geometry_repropagation": True,
                        **assets,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    _, samples, _ = load_synthetic_curriculum_manifest(
        manifest,
        require_all_splits=False,
        allowed_splits=("train",),
    )

    assert samples[0].condition_axis == "ift_ry_mrad"
    assert samples[0].condition_value == 40.0
    assert samples[0].curriculum_magnitude == 40.0
    assert samples[0].injected_offsets_xy_mm == {}
