"""Task B11: CKF fit-covariance semantics audit.

Explains why WB105 Cin is already overwide before propagation.
Traces the pinned Calypso/ACTS path.  Does not rescale, clip,
re-diagonalize, or empirically calibrate Cin.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from alignment.faseracts_propagated_covariance_validation_v2 import FROZEN_WB81_GATES
from alignment.operating_protocol_v1_final_closure import (
    project_root,
    resolve_under_root,
    sha256_file,
)
from datasets.leave_target_out_prediction_contract import catalog_calypso_interfaces
from datasets.transport_uncertainty_shape_diagnosis import (
    CASE_E as WB105_CASE,
    audit_cin_shape,
    inherit_frozen_stage as inherit_through_wb104,
    load_contracted_sample,
)

SCHEMA_VERSION = "ckf-fit-covariance-semantics-v1"
DEFAULT_CONFIG = "configs/ckf_fit_covariance_semantics_v1.yaml"
TASK = "SB-B11"
WORKBOOK = 107

DECISION = "ckf_fit_covariance_semantics_audited"
PRIMARY = "official_cin_is_global_kf_refit_front_state"


class CovarianceSemanticsError(ValueError):
    """Raised when the B11 contract is illegal."""


def refuse_cin_repair() -> None:
    raise CovarianceSemanticsError("Cin must not be rescaled, clipped, or recalibrated")


def refuse_measurement_model_v2() -> None:
    raise CovarianceSemanticsError("Measurement Model V2 is not entered in Task B11")


def refuse_raw_ckf() -> None:
    raise CovarianceSemanticsError("raw CKF must not re-enter the official B11 input")


def refuse_focus_drop() -> None:
    raise CovarianceSemanticsError("focus identity 100043/37 must be retained")


def refuse_truth_qoverp() -> None:
    raise CovarianceSemanticsError("truth q/p is not a real-data solution")


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = resolve_under_root(project_root(), str(path or DEFAULT_CONFIG))
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != SCHEMA_VERSION:
        raise CovarianceSemanticsError(f"schema_version must be {SCHEMA_VERSION}")
    if config.get("task") != TASK:
        raise CovarianceSemanticsError(f"task must be {TASK}")
    if int(config.get("workbook", -1)) != WORKBOOK:
        raise CovarianceSemanticsError(f"workbook must be {WORKBOOK}")
    if str(config.get("official_input_scope")) != "contract_eligible":
        raise CovarianceSemanticsError("official input must be contract_eligible")
    for key in (
        "geometry_write_allowed",
        "held_out_accessed",
        "real_data_alignment_authorized",
        "measurement_model_validated",
    ):
        if bool(config.get(key, True)):
            raise CovarianceSemanticsError(f"{key} must be false")
    for key in (
        "do_not_rescale_covariance",
        "do_not_rescale_cin",
        "do_not_clip_eigenvalues",
        "do_not_rediagonalize_cin",
        "do_not_empirically_calibrate_cin",
        "do_not_use_truth_q_over_p_as_real_data_solution",
        "do_not_use_raw_ckf_as_official_input",
        "do_not_drop_focus_identity",
        "do_not_enter_measurement_model_v2",
        "eligibility_independent_of_closure",
    ):
        if bool(config.get(key, False)) is not True:
            raise CovarianceSemanticsError(f"{key} must be true")
    gates = config.get("closure_gates", {})
    for key, expected in FROZEN_WB81_GATES.items():
        if gates.get(key) != expected:
            raise CovarianceSemanticsError(f"WB81/WB87/WB98 gate must stay frozen: {key}")
    wb103 = yaml.safe_load(
        resolve_under_root(
            project_root(), config["inheritance"]["workbook_103"]["config_path"]
        ).read_text(encoding="utf-8")
    )
    if dict(config["eligibility"]) != dict(wb103["eligibility"]):
        raise CovarianceSemanticsError("B11 eligibility must stay identical to WB103")
    return dict(config)


def inherit_frozen_stage(config: Mapping[str, Any]) -> dict[str, Any]:
    inherited = inherit_through_wb104(config)
    spec = config["inheritance"]["workbook_105"]
    decision = json.loads(
        resolve_under_root(project_root(), spec["decision_path"]).read_text(encoding="utf-8")
    )
    digest = sha256_file(resolve_under_root(project_root(), spec["config_path"]))
    if digest != spec["config_sha256"]:
        raise CovarianceSemanticsError(f"workbook_105 config hash mismatch: {digest}")
    digest = sha256_file(resolve_under_root(project_root(), spec["decision_path"]))
    if digest != spec["decision_sha256"]:
        raise CovarianceSemanticsError(f"workbook_105 decision hash mismatch: {digest}")
    if decision.get("decision") != spec["frozen_decision"]:
        raise CovarianceSemanticsError("workbook_105 decision must stay frozen")
    if decision.get("primary_case") != spec["frozen_primary_case"]:
        raise CovarianceSemanticsError("workbook_105 primary case must stay frozen")
    if spec["frozen_primary_case"] != WB105_CASE:
        raise CovarianceSemanticsError("WB105 primary case token mismatch")
    inherited["workbook_105"] = {
        "decision": spec["frozen_decision"],
        "primary_case": spec["frozen_primary_case"],
        "config_sha256": spec["config_sha256"],
        "decision_sha256": spec["decision_sha256"],
    }
    return inherited


def audit_code_path(config: Mapping[str, Any]) -> dict[str, Any]:
    catalog = catalog_calypso_interfaces(config)
    return {
        "calypso_git_sha": catalog["calypso_git_sha"],
        "athena_release": catalog["athena_release"],
        "acts_version": catalog["acts_version"],
        "sources": catalog["sources"],
        "production_chain": [
            {
                "step": 1,
                "component": "CKF2 CombinatorialKalmanFilter",
                "file": "Tracking/Acts/FaserActsKalmanFilter/src/CKF2.cxx",
                "material_and_energy_loss": True,
                "smoother": "Acts::GainMatrixSmoother",
                "reference_surface": "seed-tool target plane",
                "add_fitted_params_to_track_default": True,
            },
            {
                "step": 2,
                "component": "CreateTrkTrackTool.createTrack",
                "file": "Tracking/Acts/FaserActsKalmanFilter/src/CreateTrkTrackTool.cxx",
                "measurement_tsos_parameters": "smoothed + smoothedCovariance",
                "outlier_tsos_parameters": "filtered + filteredCovariance",
                "hole_tsos_parameters": "predicted + predictedCovariance",
                "optional_front_hole": "fittedParams at CKF reference surface",
            },
            {
                "step": 3,
                "component": "KalmanFitterTool.fit refit into CKFTrackCollection",
                "file": "Tracking/Acts/FaserActsKalmanFilter/src/KalmanFitterTool.cxx",
                "uses_all_measurements_on_track": True,
                "outlier_finder_cluster_z": -1_000_000.0,
                "ift_z_lt_minus_100mm_marked_outlier": True,
                "seed_from": "inputTrack->trackParameters()->front()",
                "seed_covariance_scale_default": 100.0,
                "additional_seed_matrix_scale": 10.0,
                "create_track_after_refit_includes_fitted_params": False,
                "measurement_covariance_hardcoded": "0.08*0.08/12",
                "ms_and_eloss": True,
            },
            {
                "step": 4,
                "component": "CkfActsTransportDumpAlg",
                "file": "alignment/acts_transport_dump/CkfActsTransportDumpAlg.cxx",
                "exported_state": "track->trackParameters()->front()",
                "exported_covariance": "native 5x5 then ACTS export 5x5 Cin",
            },
        ],
        "official_front_after_successful_refit": {
            "ordering": (
                "CreateTrkTrackTool inserts reversed measurement TSOS at begin, "
                "so front() is the most upstream measurement surface."
            ),
            "if_upstream_is_ift_outlier": "filtered / predicted seed, not a measurement update",
            "if_upstream_is_measurement": "smoothed using the full refit trajectory",
            "suitable_independent_propagation_seed": False,
        },
        "filter_vs_smoother": {
            "acts_fitter_runs_gain_matrix_smoother": True,
            "persisted_measurement_tsos": "smoothed",
            "persisted_outlier_tsos": "filtered",
            "edm_flag_on_dumped_jsonl": "unavailable",
            "cannot_recover_per_tsos_flag_from_ntuple": True,
        },
        "shared_measurement": {
            "target_stations_1_2_3_used_in_official_refit": True,
            "station_0_ift_marked_outlier_in_official_refit": True,
            "leave_target_out": False,
            "leave_source_ift_out": True,
        },
        "cin_modified": False,
    }


def audit_track_parameter_provenance(code_path: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "collection": "CKFTrackCollection",
        "preferred_object": "KalmanFitterTool.fit output",
        "fallback_object": "CKF2 CreateTrkTrackTool track",
        "dumped_parameter": "trackParameters().front()",
        "reference_surface_strategy": "KalmanFitterTargetSurfaceStrategy::first",
        "units": {
            "create_trk_track_qoverp": "ATLAS MeV conversion (* 1_MeV on q/p row/col)",
            "dump_qoverp": "native Trk q/p per MeV",
        },
        "not_a_perigee_on_a_dedicated_beamline_surface": True,
        "not_a_leave_target_out_prediction": True,
        "code_path": code_path["official_front_after_successful_refit"],
    }


def audit_filtered_smoothed(code_path: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "acts_smoother_connected": True,
        "persisted_flags": code_path["filter_vs_smoother"],
        "edm_dump_available": False,
        "provenance_dump_required_for_per_track_confirmation": True,
        "code_path_conclusion": (
            "After a successful official refit, IFT (z < -100 mm) is an outlier "
            "TSOS persisted as filtered.  Stations 1-3 are measurement TSOS "
            "persisted as smoothed.  The dumped front() is therefore not a "
            "source-only filtered state and not a leave-target-out state."
        ),
    }


def decide_case(cin_shape: Mapping[str, Any]) -> dict[str, Any]:
    if bool(cin_shape.get("cin_rescaled") or cin_shape.get("eigenvalues_clipped")):
        refuse_cin_repair()
    holds = cin_shape.get("source_cin_shape_holds")
    return {
        "verdict": "DIAGNOSED",
        "decision": DECISION,
        "primary_case": PRIMARY,
        "secondary_cases": [
            "official_refit_marks_ift_as_outlier",
            "seed_covariance_inflated_before_refit",
            "target_stations_used_in_refit",
        ],
        "next_step": "transport_covariance_v4_falsification",
        "source_cin_shape_holds": holds,
        "overcoverage_present_before_propagation": bool(
            cin_shape.get("overcoverage_present_before_propagation")
        ),
        "cin_modified": False,
        "suitable_independent_propagation_seed": False,
    }


def inventory_and_audit(config: Mapping[str, Any]) -> dict[str, Any]:
    sample = load_contracted_sample(config)
    code_path = audit_code_path(config)
    provenance = audit_track_parameter_provenance(code_path)
    filtered = audit_filtered_smoothed(code_path)
    cin_shape = audit_cin_shape(config, sample)
    mechanism = decide_case(cin_shape)
    return {
        "present_sources": sample["present_sources"],
        "missing_sources": sample["missing_sources"],
        "n_contracted": len(sample["contracted"]),
        "n_official_pairs": len(sample["official_events"]),
        "provenance_hashes": sample["provenance_hashes"],
        "code_path": code_path,
        "track_parameter_provenance": provenance,
        "filtered_smoothed": filtered,
        "cin_shape": cin_shape,
        "mechanism": mechanism,
    }


def decide(inventory: Mapping[str, Any], inherited: Mapping[str, Any]) -> dict[str, Any]:
    mechanism = inventory["mechanism"]
    if bool(inventory["cin_shape"].get("cin_rescaled")):
        refuse_cin_repair()
    return {
        "verdict": mechanism["verdict"],
        "decision": mechanism["decision"],
        "primary_case": mechanism["primary_case"],
        "secondary_cases": mechanism["secondary_cases"],
        "next_step": mechanism["next_step"],
        "measurement_model_v2_authorized": False,
        "measurement_model_v2_entered": False,
        "geometry_write_allowed": False,
        "official_input_scope": "contract_eligible",
        "raw_ckf_used": False,
        "focus_identity_retained": True,
        "cin_modified": False,
        "suitable_independent_propagation_seed": False,
        "inherited_wb105_decision": inherited["workbook_105"]["decision"],
        "inherited_wb105_decision_sha256": inherited["workbook_105"]["decision_sha256"],
        "inherited_wb104_decision_sha256": inherited["workbook_104"]["decision_sha256"],
        "inherited_wb103_contract_sha256": inherited["workbook_103"]["contract_sha256"],
        "wb96_through_wb105_rewritten": False,
        "overcoverage_present_before_propagation": mechanism[
            "overcoverage_present_before_propagation"
        ],
        "source_cin_shape_holds": mechanism["source_cin_shape_holds"],
    }
