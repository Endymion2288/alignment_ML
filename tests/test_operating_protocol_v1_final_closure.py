from __future__ import annotations

from pathlib import Path

import pytest

from alignment.calibration_modes import OPERATING_BAND_UM
from alignment.operating_protocol_v1_final_closure import (
    EVIDENCE_CHAIN,
    FROZEN_V2_CHECKPOINT_SHA256,
    OPERATING_MODE,
    RESIDUAL_DECREASE_LABEL,
    assert_no_alignment_payload,
    build_evidence_matrix,
    build_final_report,
    campaign_allows_alignment_payload,
    campaign_allows_cdx_mode,
    campaign_allows_geometry_write,
    campaign_allows_station_calibration_mode,
    evaluate_unlock,
    evidence_chain_text,
    extract_reduced_mode_inconsistency,
    extract_reference_scale,
    extract_selected_route_scaling,
    frozen_operating_state,
    label_dq_observable,
    load_closure_config,
    occupancy_provenance,
    unlock_criteria,
    walk_forbidden,
)
from alignment.real_data_residual_dq_monitoring import DECISION_MONITORING_ONLY


def _config():
    return load_closure_config(
        Path("configs/operating_protocol_v1_final_real_data_closure_v1.yaml")
    )


def test_config_freezes_monitoring_only_and_forbids_payload():
    config = _config()
    assert config["real_data_operating_mode"] == OPERATING_MODE
    assert config["geometry_write_allowed"] is False
    assert config["station_calibration_mode_available"] is False
    assert config["cdx_mode_allowed"] is False
    assert config["do_not_construct_reduced_station_mode"] is True
    assert config["do_not_generate_fd_probes"] is True
    assert config["do_not_run_newton"] is True
    assert config["do_not_write_payload"] is True
    assert config["do_not_extract_alignment_payload_from_self_nulling"] is True
    assert config["alignment_payload_from_self_nulling_residuals"] is False
    assert config["frozen_v2_checkpoint_sha256"] == FROZEN_V2_CHECKPOINT_SHA256
    assert config["residual_decrease_label"] == RESIDUAL_DECREASE_LABEL
    assert list(config["calibration_reference_runs"]) == [14973, 14974]
    assert campaign_allows_cdx_mode(DECISION_MONITORING_ONLY) is False
    assert campaign_allows_geometry_write(DECISION_MONITORING_ONLY) is False
    assert campaign_allows_station_calibration_mode(DECISION_MONITORING_ONLY) is False
    assert campaign_allows_alignment_payload(DECISION_MONITORING_ONLY) is False


def test_frozen_state_and_evidence_chain_are_immutable():
    state = frozen_operating_state()
    assert state["real_data_operating_mode"] == "residual_dq_monitoring_only"
    assert state["geometry_write_allowed"] is False
    assert state["station_calibration_mode_available"] is False
    assert state["cdx_mode_allowed"] is False
    assert state["alignment_drift_candidate"] is False
    assert state["code_path"] == [
        "current_official_geometry",
        "frozen_v2",
        "residual_dq_monitoring",
    ]
    assert [item["id"] for item in EVIDENCE_CHAIN] == [
        "mc_transfer",
        "real_data_frozen_v2_acceptance",
        "full_segment_station_calibration",
        "reduced_mode_calibration",
        "current_geometry_residual_dq_monitoring",
    ]
    text = evidence_chain_text()
    assert "MC transfer PASS" in text
    assert "statistics-limited but recovered at scale" in text
    assert "physical_nonidentifiability/cross_level_contamination" in text
    assert "cross-run non-transferability" in text
    assert "residual/DQ monitoring PASS on independent runs" in text


def test_unlock_currently_unmet_and_survey_plus_cdx_would_open_v2():
    criteria = unlock_criteria()
    assert criteria["currently_met"] is False
    assert criteria["opens"] == "operating_protocol_v2"
    assert criteria["until_then"]["alignment_payload_from_self_nulling_residuals"] is False
    ids = [item["id"] for item in criteria["required_any_of"]]
    assert ids == [
        "external_station_constraints_and_cdx_budget",
        "new_independent_real_data_topology",
    ]
    current = evaluate_unlock()
    assert current["currently_met"] is False
    assert current["opens_operating_protocol_v2"] is False
    assert current["geometry_write_allowed"] is False
    survey = evaluate_unlock(
        {
            "independent_survey_or_external_station_constraints": True,
            "fixed_station_parameters": ["ift_dx_mm", "ift_ry_mrad", "ift_dz_mm"],
            "independent_real_C_dx_abs_um": 1.6,
        }
    )
    assert survey["opens_operating_protocol_v2"] is True
    assert survey["geometry_write_allowed"] is False
    assert survey["alignment_payload_from_self_nulling_residuals"] is False
    topology = evaluate_unlock(
        {
            "new_independent_real_data_topology": True,
            "cross_run_transferable_calibration_subspace": True,
        }
    )
    assert topology["opens_operating_protocol_v2"] is True
    assert topology["geometry_write_allowed"] is False
    incomplete = evaluate_unlock(
        {
            "independent_survey_or_external_station_constraints": True,
            "fixed_station_parameters": ["ift_dx_mm"],
            "independent_real_C_dx_abs_um": 1.6,
        }
    )
    assert incomplete["opens_operating_protocol_v2"] is False
    outside_band = evaluate_unlock(
        {
            "independent_survey_or_external_station_constraints": True,
            "fixed_station_parameters": ["ift_dx_mm", "ift_ry_mrad", "ift_dz_mm"],
            "independent_real_C_dx_abs_um": 6.9,
        }
    )
    assert outside_band["opens_operating_protocol_v2"] is False
    assert list(OPERATING_BAND_UM) == [1.5, 1.7]


def test_residual_decrease_is_dq_observable_only():
    labeled = label_dq_observable({"chi2_before": 10.0, "chi2_after": 8.0})
    assert labeled["residual_decrease_label"] == "DQ observable"
    assert labeled["residual_reduction_is_not_alignment_success"] is True
    assert labeled["not_a_geometry_correction"] is True
    reduced = extract_reduced_mode_inconsistency(
        {
            "predeclared_modes": {
                "three_dof_dy_rx_rz": {
                    "floated_parameters": ["ift_dy_mm", "ift_rx_mrad", "ift_rz_mrad"],
                    "per_run": {
                        "14973": {"delta": {"ift_dy_mm": 1.08}, "condition_number": 73.0},
                        "14974": {"delta": {"ift_dy_mm": -8.93}, "condition_number": 86.0},
                    },
                    "run_to_run_consistency": {"max_nsigma": 258.0, "consistent": False},
                }
            }
        },
        {
            "linearized_calibration_transfer": {
                "three_dof_dy_rx_rz": [
                    {
                        "from_run": 14974,
                        "to_run": 14973,
                        "chi2_before": 1.8e6,
                        "chi2_after": 2.9e7,
                        "chi2_ratio": 16.3,
                    }
                ]
            }
        },
    )
    assert reduced["residual_decrease_label"] == "DQ observable"
    assert reduced["linearized_calibration_transfer"][0]["residual_decrease_label"] == (
        "DQ observable"
    )


def test_forbidden_keys_and_payload_guards():
    walk_forbidden({"selected_routes": 12, "truth_free_complete_route_fraction": 0.0}, where="ok")
    with pytest.raises(ValueError, match="forbidden truth-metric"):
        walk_forbidden({"route_purity": 0.9}, where="bad")
    with pytest.raises(ValueError, match="must not allow geometry write"):
        assert_no_alignment_payload({"geometry_write_allowed": True})
    assert_no_alignment_payload({"selected_routes": 12})
    with pytest.raises(ValueError, match="alignment payload"):
        assert_no_alignment_payload(
            {
                "geometry_write_allowed": False,
                "alignment_payload_from_self_nulling_residuals": False,
                "station_calibration_mode_available": False,
                "cdx_mode_allowed": False,
                "alignment_payload": {"ift_dx_mm": 0.1},
            }
        )


def test_evidence_matrix_and_final_report_keep_monitoring_only():
    matrix = build_evidence_matrix(
        {
            "transfer_report": {"both_modes_independent_closure": True},
            "scaling_report": {
                "decision": "real_data_route_acceptance_statistics_limited",
                "scale_table": [
                    {"run": 14973, "scale": "full", "selected_routes": 121},
                ],
            },
            "fullscale_decision": {"decision": "cross_level_contaminated"},
            "failure_classification": {"decision": "physical_nonidentifiability"},
            "reduced_decision": {
                "decision": "real_data_residual_dq_monitoring_only",
                "unique_class": "real_data_residual_dq_monitoring_only",
                "v2_candidate_mode": None,
            },
            "monitoring_decision": {"decision": "real_data_residual_dq_monitoring_only"},
            "expansion_decision": {
                "decision": "real_data_residual_dq_monitoring_only",
                "alignment_drift_candidate": False,
            },
            "expansion_run_level": {
                "blocks": [
                    {"run": 14971, "expansion_run": True, "status": "nominal_monitoring"}
                ]
            },
        }
    )
    assert matrix["geometry_write_allowed"] is False
    assert matrix["chain"][0]["status"] == "PASS"
    assert matrix["chain"][2]["rejection_class"] == [
        "physical_nonidentifiability",
        "cross_level_contamination",
    ]
    report = build_final_report(
        config=_config(),
        evidence_matrix=matrix,
        manifest={
            "occupancy_selection_rule": {"minima": {"total_tracklets": 32}},
            "reference_residual_scale": {"dy": {"robust_scale": 6.712}},
            "alarm_thresholds": {"robust_z_detector_condition": 5.0},
            "A_operator": {"A_dx": -59.213},
            "input_run_provenance": [],
        },
        expansion_decision={"alignment_drift_candidate": False},
        created_utc="2026-08-23T00:00:00+00:00",
    )
    assert report["real_data_operating_mode"] == "residual_dq_monitoring_only"
    assert report["geometry_write_allowed"] is False
    assert report["station_calibration_mode_available"] is False
    assert report["cdx_mode_allowed"] is False
    assert report["alignment_drift_candidate"] is False
    assert report["unlock_evaluation"]["currently_met"] is False


def test_scaling_and_occupancy_helpers():
    scaling = extract_selected_route_scaling(
        {
            "scale_table": [
                {"run": 14973, "role": "calibration", "scale": "n100", "selected_routes": 0}
            ]
        },
        [
            {
                "run": 14971,
                "role": "monitoring",
                "n_events": 112357,
                "selected_routes": 233,
                "complete_four_station_routes": 4,
            }
        ],
    )
    assert scaling["rows"][0]["selected_routes"] == 0
    assert scaling["rows"][1]["campaign"] == "entry_53_expansion"
    assert scaling["residual_decrease_label"] == "DQ observable"
    rows = occupancy_provenance(
        {
            "runs": {
                "14973": {
                    "role": "calibration",
                    "segment": "00007",
                    "skip_events": 49500,
                    "input_xaod": "/xaod/14973.root",
                    "status": "frozen_window",
                }
            }
        }
    )
    assert rows[0]["run"] == 14973
    reference = extract_reference_scale(
        {
            "calibration_reference": {
                "runs": [14973, 14974],
                "n_reference_field_edges": [204, 186],
                "channels": {
                    "dy": {"median": -3.81, "robust_scale": 6.712, "n": 390},
                    "rx": {"median": -0.00178, "robust_scale": 0.00524, "n": 390},
                },
            }
        }
    )
    assert reference["reestimated"] is False
    assert reference["used_for_geometry"] is False
