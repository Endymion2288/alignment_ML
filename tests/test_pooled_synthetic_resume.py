from __future__ import annotations

import json

import pytest

from datasets.physical_curriculum import SYNTHETIC_CORPUS_SCHEMA
from scripts.materialize_pooled_curriculum_synthetics import (
    _condition_metadata,
    _load_resumable_synthetic_manifest,
    _merge_resumed_samples,
    _stable_seed,
    _sources_by_split_from_samples,
)


def _sample(split: str, payload_id: str, source_id: str) -> dict[str, object]:
    return {
        "split": split,
        "payload_id": payload_id,
        "source_id": f"pooled_{split}",
        "source_ids": [source_id],
    }


def test_split_resume_preserves_previously_materialized_train_samples():
    merged = _merge_resumed_samples(
        [_sample("train", "mag_0_train_00", "train_a")],
        [_sample("validation", "mag_0_validation_00", "validation_a")],
        {"validation"},
    )

    assert [(sample["split"], sample["payload_id"]) for sample in merged] == [
        ("train", "mag_0_train_00"),
        ("validation", "mag_0_validation_00"),
    ]
    assert _sources_by_split_from_samples(merged) == {
        "train": ["train_a"],
        "validation": ["validation_a"],
        "test": [],
    }


def test_split_resume_replaces_only_the_requested_split_and_rejects_duplicates():
    merged = _merge_resumed_samples(
        [_sample("train", "mag_0_train_00", "old_train")],
        [_sample("train", "mag_0_train_00", "new_train")],
        {"train"},
    )
    assert merged[0]["source_ids"] == ["new_train"]

    with pytest.raises(ValueError, match="duplicate sample identity"):
        _merge_resumed_samples(
            [],
            [
                _sample("train", "mag_0_train_00", "a"),
                _sample("train", "mag_0_train_00", "b"),
            ],
            {"train"},
        )


def test_resume_manifest_must_reference_the_same_physical_corpus(tmp_path):
    physical = tmp_path / "physical.json"
    physical.touch()
    other = tmp_path / "other_physical.json"
    other.touch()
    manifest = tmp_path / "synthetic.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": SYNTHETIC_CORPUS_SCHEMA,
                "physical_corpus_manifest": str(other),
                "physical_geometry_repropagation": True,
                "q_over_p_mode": 0,
                "samples": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="different physical corpus"):
        _load_resumable_synthetic_manifest(manifest, physical)


def test_rotation_condition_metadata_and_seed_keep_mrad_axis_explicit():
    condition = _condition_metadata(
        {
            "name": "ry_p40",
            "condition_axis": "ift_ry_mrad",
            "condition_value": 40.0,
            "condition_magnitude": 40.0,
            "injected_station_transforms": {"0": [0.0, 0.0, 0.0, 0.0, 0.04, 0.0]},
        }
    )

    assert condition["condition_axis"] == "ift_ry_mrad"
    assert condition["condition_magnitude"] == 40.0
    rotation_seed = _stable_seed(
        7,
        "train",
        "ry_p40",
        "ift_ry_mrad",
        40.0,
        "condition_magnitude_shared_across_direction_trials",
    )
    translation_seed = _stable_seed(
        7,
        "train",
        "mag_40",
        "translation_xy_mm",
        40.0,
        "condition_magnitude_shared_across_direction_trials",
    )
    assert rotation_seed != translation_seed
