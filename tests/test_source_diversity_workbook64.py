from __future__ import annotations

from pathlib import Path

from scripts.audit_four_station_reserved_blind_unused import loaded_sources_from_artifact
from scripts.evaluate_four_station_source_diversity_blind import classify_blind_decision
from scripts.merge_four_station_source_diversity_train_corpus import common_scan_contract
from scripts.prepare_multisource_multidof_iteration import iteration_split_mode
from training.source_diversity_audit import AUTHORIZED_NEW_TRAIN_SOURCES, RESERVED_BLIND_SOURCES


def test_new_train_sources_are_the_authorized_four():
    assert AUTHORIZED_NEW_TRAIN_SOURCES == (
        "mc24_100043_00300_00399",
        "mc24_100044_00200_00299",
        "mc24_100047_00100_00149",
        "mc24_100048_00100_00149",
    )


def test_reserved_listing_is_not_loaded_use():
    path = Path("outputs/fake/decision.json")
    payload = {"reserved_blind_sources": list(RESERVED_BLIND_SOURCES)}
    assert loaded_sources_from_artifact(path, payload) == set()
    used = {
        "sources": [
            {"source_id": RESERVED_BLIND_SOURCES[0], "split": "validation"},
        ]
    }
    assert loaded_sources_from_artifact(Path("outputs/fake/iteration_manifest.json"), used) == {
        RESERVED_BLIND_SOURCES[0]
    }


def test_identical_scan_contracts_compare_equal():
    plan = {
        "scan_mode": "station_rigid_multidof",
        "q_over_p_mode": 0,
        "station_ids": [0, 1, 2, 3],
        "reference_station_ids": [],
        "movable_station_ids": [0, 1, 2, 3],
        "condition_axis": "four_station_relative_l2",
        "alignment_parameter_specs": [{"name": "s0_dx_mm"}],
        "points": [{"name": "iteration_00_reference"}],
    }
    other = dict(plan)
    other["created_utc"] = "ignore-me"
    assert common_scan_contract(plan) == common_scan_contract(other)


def test_blind_pass_authorizes_wls_and_common_drop_stops_v2():
    passed = classify_blind_decision(
        passed=True, layer1_ok=True, same_common_se3_truth_utility_drop=False
    )
    assert passed["coverage_class"] == "training_domain_coverage_limitation_resolved_by_source_diversity"
    assert passed["continue_to_15d_relative_wls"] is True
    failed = classify_blind_decision(
        passed=False, layer1_ok=True, same_common_se3_truth_utility_drop=True
    )
    assert failed["stop_v2_mainline"] is True
    assert failed["continue_to_15d_relative_wls"] is False
    assert failed["do_not_add_more_same_family_sources"] is True
    assert failed["coverage_class"] == "source_diversity_insufficient_for_gauge_transfer"


def test_iteration_split_mode_workbook64_contracts():
    assert (
        iteration_split_mode(
            {
                "allowed_splits": ["train"],
                "forbidden_splits": ["validation", "test"],
                "test_data_accessed": False,
            },
            label="train",
        )
        == "train_only"
    )
    assert (
        iteration_split_mode(
            {
                "allowed_splits": ["validation"],
                "forbidden_splits": ["train", "test"],
                "reserved_blind_validation_only": True,
                "test_data_accessed": False,
            },
            label="blind",
        )
        == "reserved_blind_validation"
    )
