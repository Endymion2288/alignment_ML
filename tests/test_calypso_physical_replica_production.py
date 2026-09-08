"""Hermetic checks for physical replica production contracts."""

from __future__ import annotations

import pytest

from alignment.calypso_physical_replica_production import (
    N_GEOMETRY_CONDITIONS,
    REQUIRED_INPUT_INDEPENDENT_EVENTS,
    TARGET_REPLICAS_PER_CONDITION,
    PhysicalReplicaProductionError,
    allocation_plan,
    audit_covariance,
    frozen_conditions,
    independence_audit,
    locked_qualification_flags,
    qa_gate,
    refuse_forbidden_path,
    replica_uid,
)


def test_allocation_is_3200_unique_and_balanced():
    plan = allocation_plan()
    assert plan["n_events"] == REQUIRED_INPUT_INDEPENDENT_EVENTS == 3200
    assert plan["n_shards"] == 48
    assert plan["trained_on_wb83_failure_numbers"] is False
    assert all(count == 400 for count in plan["per_condition_input_events"].values())
    assert len(plan["per_condition_input_events"]) == N_GEOMETRY_CONDITIONS
    pairs = {(row["source_id"], row["xaod_entry_index"]) for row in plan["events"]}
    assert len(pairs) == 3200
    conditions = {row["condition_id"] for row in plan["events"]}
    assert len(conditions) == 8


def test_conditions_are_not_the_wb83_matrix():
    cells = frozen_conditions()
    ids = [cell["condition_id"] for cell in cells]
    assert "C3_weak_0.5x_fixed_dz" not in ids
    assert all(cell["survey_mode_is_reconstruction_difference"] is False for cell in cells)
    shas = {cell["requested_payload_sha256"] for cell in cells}
    assert len(shas) == 4


def test_forbidden_assets_are_strings_only():
    with pytest.raises(PhysicalReplicaProductionError):
        refuse_forbidden_path("/eos/experiment/faser/data0/sim/mc24/particle_gun/100047/00350_00399/x.root")
    with pytest.raises(PhysicalReplicaProductionError):
        refuse_forbidden_path("mc24_100116_anything")
    refuse_forbidden_path(
        "/eos/experiment/faser/data0/sim/mc24/particle_gun/100047/rec/s0013-r0022/"
        "FaserMC-MC24_PG_mumi_fasernu_5mrad_flukaE-100047-00100-00149-s0013-r0022-xAOD.root"
    )


def test_indefinite_covariance_is_rejected_without_repair():
    report = audit_covariance(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, -1.0],
        ]
    )
    assert report["accepted"] is False
    assert report["silently_repaired"] is False
    assert report["spd_ok"] is False
    assert report["reject_reason"] == "invalid_covariance"


def test_independence_audit_does_not_rename_reuse():
    rows = [
        {"source_id": "a", "run_id": 1, "event_id": 2, "condition_id": "c0"},
        {"source_id": "a", "run_id": 1, "event_id": 2, "condition_id": "c1"},
    ]
    audit = independence_audit(rows)
    assert audit["independence_violation"] is True
    assert audit["n_violations"] == 1
    assert audit["violations"][0]["event_uid"] == replica_uid("a", 1, 2)
    assert audit["violations"][0]["hidden_by_renaming"] is False


def test_qa_gate_cannot_qualify_the_alignment_oracle():
    flags = locked_qualification_flags()
    assert flags["alignment_oracle_qualified_for_physical_FASER"] is False
    assert flags["ml_alignment_eval_authorized"] is False
    qa = qa_gate(
        {
            "unique_physical_events": True,
            "zero_independence_violations": True,
            "zero_forbidden_assets": True,
            "geometry_roundtrip": True,
            "software_provenance_complete": True,
            "field_material_hashes_consistent": True,
            "truth_label_completeness": True,
            "measurement_schema": True,
            "covariance_contract": True,
            "condition_counts": True,
            "production_complete": True,
        }
    )
    assert qa["calypso_physical_corpus_qualified"] is True
    assert qa["alignment_oracle_qualified_for_physical_FASER"] is False
    assert TARGET_REPLICAS_PER_CONDITION == 200
