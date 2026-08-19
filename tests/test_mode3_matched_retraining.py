from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.evaluate_mode3_matched_retraining import (
    _bad_edge_summary,
    _closure_parameters,
    _edge_metrics,
    _route_metrics,
    _source_wise_routes,
)
from scripts.train_geometry_aware_transformer_v1 import _validate_manifest_contract as v1_contract
from scripts.train_route_aware_transformer_v2 import _validate_input_contract as v2_contract


def test_edge_metrics_mlp():
    # Real MLP metrics.json layout: edge AP/AUC/ECE live under
    # candidate_gate_selection; the operating point carries association metrics.
    payload = {
        "candidate_gate_selection": {
            "average_precision": 0.9,
            "roc_auc": 0.95,
            "expected_calibration_error": 0.02,
        },
        "primary_validation_operating_point": {
            "average_precision": 0.9,
            "association_efficiency": 0.96,
            "association_purity": 0.99,
            "fake_rate": 0.01,
        },
    }
    metrics = _edge_metrics(payload, family="mlp")
    assert metrics["average_precision"] == pytest.approx(0.9)
    assert metrics["roc_auc"] == pytest.approx(0.95)
    assert metrics["expected_calibration_error"] == pytest.approx(0.02)
    assert metrics["association_efficiency"] == pytest.approx(0.96)


def test_edge_metrics_v1_reads_nominal_route():
    payload = {
        "calibrated_candidate": {
            "average_precision": 0.91,
            "roc_auc": 0.97,
            "expected_calibration_error": 0.03,
        },
        "route_selection": {
            "evaluation_by_magnitude": {
                "0.0": {
                    "route": {
                        "complete_track_efficiency": 0.93,
                        "complete_track_purity": 0.95,
                        "track_fake_rate": 0.05,
                    }
                }
            }
        },
    }
    metrics = _edge_metrics(payload, family="v1")
    assert metrics["nominal_complete_track_efficiency"] == pytest.approx(0.93)
    assert metrics["roc_auc"] == pytest.approx(0.97)


def test_edge_metrics_v2_reads_calibrated_stream():
    payload = {
        "score_streams": {
            "route_aware_v2": {
                "score_stream_calibration": {
                    "after": {
                        "average_precision": 0.87,
                        "roc_auc": 0.94,
                        "expected_calibration_error": 0.05,
                    }
                }
            }
        }
    }
    metrics = _edge_metrics(payload, family="v2")
    assert metrics["average_precision"] == pytest.approx(0.87)


def test_route_metrics_from_backbone_summary():
    summary = {
        "truth_labelled_mc_evaluation": {
            "route": {
                "complete_track_efficiency": 0.74,
                "complete_track_purity": 0.97,
                "track_fake_rate": 0.04,
                "candidate_complete_truth_chain_recall": 1.0,
            }
        },
        "raw_candidate_score_metrics": {"average_precision": 0.98, "roc_auc": 0.99},
        "frozen_calibrated_score_metrics": {
            "average_precision": 0.96,
            "roc_auc": 0.99,
            "expected_calibration_error": 0.08,
        },
    }
    metrics = _route_metrics(summary)
    assert metrics["complete_track_efficiency"] == pytest.approx(0.74)
    assert metrics["calibrated_expected_calibration_error"] == pytest.approx(0.08)


def test_closure_parameters_accept_legacy_error_key():
    payload = {
        "capture_success": True,
        "normal_matrix_condition_number": 108.0,
        "parameters": [
            {"name": "ift_dx_mm", "error": -0.05, "expected_delta_to_target": 0.14, "capture_success": True, "capture_tolerance": 0.1},
            {"name": "ift_dy_mm", "local_delta_error": 0.04, "expected_delta_to_target": -0.11, "capture_success": True, "capture_tolerance": 0.1},
        ],
    }
    parameters = _closure_parameters(payload)
    assert parameters["ift_dx_mm"]["local_delta"] == pytest.approx(-0.05)
    assert parameters["ift_dy_mm"]["local_delta"] == pytest.approx(0.04)
    assert parameters["capture_success"] is True


def test_bad_edge_summary_median():
    payload = {
        "copies_found": 2,
        "copies": [
            {"calibrated_score": 0.8, "rank_among_event_0_to_1_candidates": 2},
            {"calibrated_score": 0.6, "rank_among_event_0_to_1_candidates": 4},
        ],
    }
    summary = _bad_edge_summary(payload)
    assert summary["copies_found"] == 2
    assert summary["median_calibrated_score"] == pytest.approx(0.7)
    assert summary["median_rank"] == pytest.approx(3.0)


def test_source_wise_routes_counts_consistency(tmp_path: Path):
    def signature(pairs):
        return json.dumps(
            [
                {"origin_run_id": run, "origin_event_id": event, "origin_tracklet_id": idx, "station_id": idx}
                for idx, (run, event) in enumerate(pairs)
            ]
        )

    rows = [
        # truth-consistent complete route from source 9001
        {"route_origin_signature": signature([(9001, 7)] * 4), "is_complete_four_station_route": True},
        # mixed route combining sources 9001 and 9002
        {
            "route_origin_signature": signature([(9001, 7), (9002, 3), (9001, 7), (9001, 7)]),
            "is_complete_four_station_route": True,
        },
        # incomplete truth-consistent route from source 9002
        {"route_origin_signature": signature([(9002, 3)] * 3), "is_complete_four_station_route": False},
    ]
    anchor = tmp_path / "anchor"
    anchor.mkdir()
    with (anchor / "selected_routes.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    result = _source_wise_routes(anchor)
    assert result["available"] is True
    assert result["sources"] == 2
    per_source = {entry["origin_run_id"]: entry for entry in result["per_source"]}
    # source 9001 appears in the consistent route and the mixed route
    assert per_source[9001]["routes"] == 2
    assert per_source[9001]["truth_consistent_fraction"] == pytest.approx(0.5)
    assert per_source[9001]["complete_route_fraction"] == pytest.approx(1.0)
    # source 9002 appears in the mixed route and its own incomplete route
    assert per_source[9002]["routes"] == 2
    assert per_source[9002]["truth_consistent_fraction"] == pytest.approx(0.5)
    assert per_source[9002]["complete_route_fraction"] == pytest.approx(0.5)
    assert result["min_truth_consistent_fraction"] == pytest.approx(0.5)


def test_source_wise_routes_missing_file(tmp_path: Path):
    assert _source_wise_routes(tmp_path) == {"available": False}


def _v1_config(mode: int) -> tuple[dict, dict, dict, list]:
    root = {
        "allowed_splits": ["train", "validation"],
        "forbidden_splits": ["test"],
        "refit": {"q_over_p_mode": mode},
        "payload_bank": {"magnitudes_mm": [0.0, 0.1]},
    }
    transformer = {"q_over_p_mode": mode, "candidate_chi2_gate": None}
    manifest = {"physical_geometry_repropagation": True, "q_over_p_mode": mode}
    return root, transformer, manifest, []


def test_v1_contract_accepts_matching_mode3():
    root, transformer, manifest, samples = _v1_config(3)
    # Fails later on the empty sample curriculum; the mode checks must pass.
    with pytest.raises(ValueError, match="split|curriculum"):
        v1_contract(root, transformer, manifest, samples, q_over_p_mode=3)


def test_v1_contract_rejects_mode_mismatch():
    root, transformer, manifest, samples = _v1_config(0)
    with pytest.raises(ValueError, match="q_over_p_mode"):
        v1_contract(root, transformer, manifest, samples, q_over_p_mode=3)


def _v2_contract_payloads(mode: int) -> tuple[dict, dict, list]:
    contract = {
        "allowed_splits": ["train", "validation"],
        "forbidden_splits": ["test"],
        "physical_geometry_repropagation": True,
        "q_over_p_mode": mode,
        "candidate_chi2_gate": None,
        "feature_set": "residual_v1",
        "context_mode": "full_event",
        "station_path": [0, 1, 2, 3],
        "curriculum_magnitudes_mm": [0.0, 0.1],
    }
    manifest = {"physical_geometry_repropagation": True, "q_over_p_mode": mode}
    return contract, manifest, []


def test_v2_contract_accepts_matching_mode3():
    contract, manifest, samples = _v2_contract_payloads(3)
    # Fails later on the empty sample set; the mode check itself must pass.
    with pytest.raises(ValueError, match="train and validation"):
        v2_contract(contract, manifest, samples, q_over_p_mode=3)


def test_v2_contract_rejects_mode_mismatch():
    contract, manifest, samples = _v2_contract_payloads(0)
    with pytest.raises(ValueError, match="mode-3"):
        v2_contract(contract, manifest, samples, q_over_p_mode=3)
