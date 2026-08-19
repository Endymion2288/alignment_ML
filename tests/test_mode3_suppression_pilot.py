from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from datasets.propagation_loader import PropagationRecords
from datasets.root_loader import EventTracklets
from baselines.field_chi2_matching import build_field_candidates
from scripts.build_mode3_suppression_pilot import build_pilot_manifest


def _event() -> EventTracklets:
    return EventTracklets(
        run_id=4,
        event_id=9,
        station_id=np.asarray([0, 1], dtype=np.int16),
        tracklet_id=np.asarray([3, 8], dtype=np.int32),
        z_mm=np.asarray([0.0, 10.0]),
        state=np.asarray([[0.0, 0.0, 0.1, -0.1], [1.0, -1.0, 0.1, -0.1]]),
        covariance=np.tile(np.diag([0.2, 0.3, 0.01, 0.02]), (2, 1, 1)),
        chi2=np.asarray([1.0, 1.0]),
        ndof=np.asarray([2.0, 2.0]),
        n_hit=np.asarray([3, 3], dtype=np.int16),
        hit_pattern=np.asarray([7, 7], dtype=np.uint64),
        truth_particle_id=np.asarray([42, 42], dtype=np.int64),
        truth_pdg=np.asarray([13, 13], dtype=np.int32),
        truth_match_fraction=np.asarray([1.0, 1.0]),
    )


def _records(modes: list[int]) -> PropagationRecords:
    target_state = np.asarray([1.0, -1.0, 0.1, -0.1])
    count = len(modes)
    return PropagationRecords(
        run_id=np.full(count, 4, dtype=np.int64),
        event_id=np.full(count, 9, dtype=np.int64),
        source_tracklet_id=np.full(count, 3, dtype=np.int32),
        target_tracklet_id=np.full(count, 8, dtype=np.int32),
        source_station_id=np.zeros(count, dtype=np.int16),
        target_station_id=np.ones(count, dtype=np.int16),
        truth_particle_id=np.full(count, 42, dtype=np.int64),
        target_z_mm=np.full(count, 10.0),
        prediction=np.tile(target_state, (count, 1)),
        covariance=np.tile(np.diag([0.4, 0.5, 0.03, 0.04]), (count, 1, 1)),
        success=np.ones(count, dtype=bool),
        has_covariance=np.ones(count, dtype=bool),
        q_over_p_mode=np.asarray(modes, dtype=np.int8),
    )


def test_build_field_candidates_selects_mode3_records():
    event = _event()
    records = _records([0, 3])
    mode0 = build_field_candidates(event, records, 0, 1, q_over_p_mode=0)
    mode3 = build_field_candidates(event, records, 0, 1, q_over_p_mode=3)
    assert len(mode0) == 1
    assert len(mode3) == 1
    assert mode0[0].source_index == mode3[0].source_index == 0
    assert mode0[0].target_index == mode3[0].target_index == 1


def test_build_field_candidates_rejects_unknown_mode():
    with pytest.raises(ValueError, match="q_over_p_mode"):
        build_field_candidates(_event(), _records([0]), 0, 1, q_over_p_mode=7)


def _production_manifest(root: Path) -> dict:
    scan_root = root / "sources" / "mc24_a" / "physical_scan"
    point_dir = scan_root / "points" / "iteration_01_reference"
    for relative in (
        "payload/alignment_payload.json",
        "refit/tracklets.root",
        "refit/propagations.root",
        "refit/content_audit.json",
    ):
        path = point_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
    return {
        "allowed_splits": ["train", "validation"],
        "source_event_uid_convention": "source_id:run_id:event_id",
        "q_over_p_mode": 0,
        "physical_geometry_repropagation": True,
        "sources": [
            {
                "source_id": "mc24_a",
                "split": "train",
                "input_xaod": "/eos/example.root",
                "physical_scan_config": str(root / "sources" / "mc24_a" / "physical_scan_config.yaml"),
                "physical_scan_root": str(scan_root),
                "source_event_uids": ["mc24_a:1:1"],
                "points": [
                    {
                        "name": "iteration_01_reference",
                        "payload_id": "iteration_01_reference",
                        "point_role": "nominal",
                        "direction_trial": "iteration_01_reference",
                        "condition_axis": "ift_dx_dy_ry_joint_l2",
                        "condition_value": 0.0,
                        "condition_magnitude": 0.0,
                        "injected_station_transforms": {"0": [0.0] * 6},
                        "alignment_parameter_values": {"ift_dx_mm": 0.0},
                        "completed": True,
                        "completion_status": "accepted",
                        "physical_tracklets": str(point_dir / "refit" / "tracklets.root"),
                        "physical_propagations": str(point_dir / "refit" / "propagations.root"),
                        "physical_payload_manifest": str(
                            point_dir / "payload" / "alignment_payload.json"
                        ),
                        "physical_content_audit": str(point_dir / "refit" / "content_audit.json"),
                    }
                ],
            },
            {
                "source_id": "mc24_b",
                "split": "validation",
                "input_xaod": "/eos/example_b.root",
                "physical_scan_config": str(root / "sources" / "mc24_b" / "physical_scan_config.yaml"),
                "physical_scan_root": str(root / "sources" / "mc24_b" / "physical_scan"),
                "source_event_uids": ["mc24_b:1:1"],
                "points": [],
            },
        ],
    }


def test_pilot_manifest_rewrites_paths_and_checks_completion(tmp_path: Path):
    production_root = tmp_path / "production"
    production_root.mkdir()
    manifest_path = production_root / "physical_corpus_manifest.json"
    manifest_path.write_text(json.dumps(_production_manifest(production_root)))
    pilot_root = tmp_path / "pilot"
    # Only the reference point exists in the pilot root; it must be complete.
    point_dir = pilot_root / "sources" / "mc24_a" / "physical_scan" / "points" / "iteration_01_reference"
    for relative in (
        "payload/alignment_payload.json",
        "refit/tracklets.root",
        "refit/propagations.root",
        "refit/content_audit.json",
    ):
        path = point_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")

    output = tmp_path / "pilot_manifest.json"
    manifest = build_pilot_manifest(
        manifest_path, pilot_root, ["mc24_a"], output
    )

    assert output.is_file()
    assert len(manifest["sources"]) == 1
    source = manifest["sources"][0]
    assert source["physical_scan_root"].startswith(str(pilot_root))
    point = source["points"][0]
    assert point["completed"] is True
    assert point["physical_tracklets"].startswith(str(pilot_root))
    assert manifest["pilot_provenance"]["pilot_sources"] == ["mc24_a"]


def test_pilot_manifest_flags_incomplete_points(tmp_path: Path):
    production_root = tmp_path / "production"
    production_root.mkdir()
    manifest_path = production_root / "physical_corpus_manifest.json"
    manifest_path.write_text(json.dumps(_production_manifest(production_root)))
    pilot_root = tmp_path / "pilot"  # nothing produced yet

    manifest = build_pilot_manifest(
        manifest_path, pilot_root, ["mc24_a"], tmp_path / "pilot_manifest.json"
    )

    point = manifest["sources"][0]["points"][0]
    assert point["completed"] is False
    assert point["completion_status"] == "incomplete"


def test_pilot_manifest_rejects_validation_source(tmp_path: Path):
    production_root = tmp_path / "production"
    production_root.mkdir()
    manifest_path = production_root / "physical_corpus_manifest.json"
    manifest_path.write_text(json.dumps(_production_manifest(production_root)))

    with pytest.raises(ValueError, match="not a train source"):
        build_pilot_manifest(
            manifest_path, tmp_path / "pilot", ["mc24_b"], tmp_path / "out.json"
        )
