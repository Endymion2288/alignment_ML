"""Yasu Stage 1: independent 3ST→IFT prediction chain.

Official CKFTrackCollectionWithoutIFT is the only prediction source.
IFT residuals come from SCT_ClusterContainer.  LTO, 4-station CKF,
dummy SegmentFit q/p, and truth q/p cannot substitute.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from alignment.numerical_contract import is_spd
from alignment.operating_protocol_v1_final_closure import (
    project_root,
    resolve_under_root,
    sha256_file,
)
from datasets.access_policy import AccessScope, authorize_path
from datasets.ckf_without_ift_definition import (
    SOURCE_COLLECTION,
    access_split as without_ift_access_split,
)
from datasets.qoverp_semantics import SEGMENTFIT_DUMMY_QOVERP_PER_MEV
from datasets.transport_contract import MEV_PER_GEV

SCHEMA_VERSION = "three-st-to-ift-prediction-v1"
DEFAULT_CONFIG = "configs/three_st_to_ift_prediction_v1.yaml"
TASK = "YASU-S1"
WORKBOOK = 118

SOURCE_COLLECTION_NAME = SOURCE_COLLECTION
FOUR_STATION_COLLECTION = "CKFTrackCollection"
INDEPENDENT_MEASUREMENT_CONTAINER = "SCT_ClusterContainer"
IFT_STATION_ID = 0
THREE_STATION_IDS = (1, 2, 3)
QOVERP_INDEX = 4
STATION_Z_MM = (-1860.15, 47.4, 1237.4, 2427.4)

DECISION_ESTABLISHED = "three_st_to_ift_prediction_established"
DECISION_NOT_ESTABLISHED = "three_st_to_ift_prediction_not_established"
MECHANISM_STAGE0 = "without_ift_definition_not_inherited"
MECHANISM_DUMP_MISSING = "prediction_dump_not_materialized"
MECHANISM_IFT_LEAK = "ift_measurement_or_outlier_present"
MECHANISM_DUMMY = "without_ift_qoverp_matches_segmentfit_dummy"
MECHANISM_TRUTH = "truth_qoverp_forbidden"
MECHANISM_LTO_SUBSTITUTE = "lto_used_as_without_ift_substitute"
MECHANISM_FOUR_STATION = "four_station_ckf_used_as_prediction_source"
MECHANISM_SOURCE_PIN = "pinned_calypso_source_hash_mismatch"
MECHANISM_NO_5X5 = "without_ift_5x5_not_materialized"
MECHANISM_NO_IFT = "independent_ift_measurement_absent"
MECHANISM_START = "acts_start_unavailable"
MECHANISM_PROP = "acts_s1_to_ift_propagation_failed"
MECHANISM_NO_TRACKS = "no_without_ift_candidate"

FAILURE_CLASSES = (
    "missing_track_payload",
    "ift_measurement_leak",
    "acts_start_unavailable",
    "acts_s1_to_ift_propagation_failed",
    "independent_ift_measurement_absent",
    "no_ift_cluster_in_event",
    "ift_prediction_available_residual_unavailable",
    "no_without_ift_candidate",
    "surface_not_in_identifier_map",
    "surface_pointer_null",
    "propagate_surface_failed",
)


class ThreeStToIftPredictionError(ValueError):
    """Raised when the 3ST→IFT prediction contract is illegal."""


def residual_definition() -> dict[str, Any]:
    return {
        "kind": "independent_ift_cluster_local_loc0",
        "measurement": "FaserSCT_Cluster.localPosition Trk::locX",
        "prediction": "Acts BoundTrackParameters loc0 on the IFT wafer surface",
        "formula": "r = loc0_cluster - loc0_predicted",
        "unit": "mm",
        "source": INDEPENDENT_MEASUREMENT_CONTAINER,
        "station_decoder": "FaserSCT_ID",
        "prediction_uses_ift_measurement": False,
        "inside_bounds_required_for_residual": False,
    }


def state_definition() -> dict[str, Any]:
    return {
        "native_athena": ["loc1_mm", "loc2_mm", "phi", "theta", "q_over_p_per_mev"],
        "native_parameter_type": "Trk::CurvilinearParameters",
        "export_parameters": ["x_mm", "y_mm", "tx", "ty", "q_over_p_per_mev"],
        "export_frame": "global_cartesian_slopes_at_surface_z",
        "q_over_p_native_unit": "per_MeV",
        "q_over_p_acts_unit": "per_GeV",
        "q_over_p_signed": True,
        "charge_sign": "sign_of_q_over_p",
        "loc_unit": "mm",
        "angle_unit": "rad",
        "mev_per_gev": MEV_PER_GEV,
        "covariance_dimension": 5,
        "prediction_uses_ift_measurement": False,
    }


def refuse_truth_qoverp() -> None:
    raise ThreeStToIftPredictionError("truth q/p is not a real-data solution")


def refuse_dummy_segmentfit() -> None:
    raise ThreeStToIftPredictionError(
        "dummy SegmentFit covariance is not a physical prior"
    )


def refuse_lto_substitute() -> None:
    raise ThreeStToIftPredictionError(
        "WB109 LTO is not CKFTrackCollectionWithoutIFT; LTO keeps IFT"
    )


def refuse_four_station_substitute() -> None:
    raise ThreeStToIftPredictionError(
        "CKFTrackCollection is the 4-station collection, not the 3ST prediction source"
    )


def access_split(split: str) -> str:
    return without_ift_access_split(split)


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = resolve_under_root(project_root(), str(path or DEFAULT_CONFIG))
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ThreeStToIftPredictionError(f"schema_version must be {SCHEMA_VERSION}")
    if config.get("task") != TASK:
        raise ThreeStToIftPredictionError(f"task must be {TASK}")
    if int(config.get("workbook", -1)) != WORKBOOK:
        raise ThreeStToIftPredictionError(f"workbook must be {WORKBOOK}")
    if str(config.get("source_collection")) != SOURCE_COLLECTION_NAME:
        raise ThreeStToIftPredictionError(
            "source_collection must be CKFTrackCollectionWithoutIFT"
        )
    if str(config.get("independent_measurement_container")) != (
        INDEPENDENT_MEASUREMENT_CONTAINER
    ):
        raise ThreeStToIftPredictionError(
            "independent_measurement_container must be SCT_ClusterContainer"
        )
    if int(config.get("ift_station_id", -1)) != IFT_STATION_ID:
        raise ThreeStToIftPredictionError("ift_station_id must be 0")
    if int(config.get("target_station", -1)) != IFT_STATION_ID:
        raise ThreeStToIftPredictionError("target_station must be 0")
    if str(config.get("propagation_direction")) != "backward":
        raise ThreeStToIftPredictionError("propagation_direction must be backward")
    station_z = config.get("station_z_mm") or {}
    for idx, expected in enumerate(STATION_Z_MM):
        if abs(float(station_z[idx]) - expected) > 1.0e-9:
            raise ThreeStToIftPredictionError(
                f"station_z_mm[{idx}] must stay at the WB87 value {expected}"
            )
    for key in (
        "geometry_write_allowed",
        "held_out_accessed",
        "real_data_alignment_authorized",
        "measurement_model_validated",
        "mc_qp_calibration_authorized",
        "residual_conditional_authorized",
    ):
        if bool(config.get(key, True)):
            raise ThreeStToIftPredictionError(f"{key} must be false")
    for key in (
        "do_not_rescale_covariance",
        "do_not_use_truth_q_over_p_as_real_data_solution",
        "do_not_use_dummy_segmentfit_covariance",
        "do_not_treat_lto_as_without_ift",
        "do_not_treat_four_station_ckf_as_without_ift",
        "do_not_enter_alignment",
        "do_not_enter_measurement_model_v2",
        "do_not_start_new_source_campaign",
        "do_not_equate_warning_count_with_reconstruction_loss",
        "do_not_scan_acts_boundary_in_this_stage",
        "unconstrained_tracker_only_stopped",
    ):
        if bool(config.get(key, False)) is not True:
            raise ThreeStToIftPredictionError(f"{key} must be true")
    return dict(config)


def _expect_sha(path: Path, expected: str, label: str) -> None:
    digest = sha256_file(path)
    if digest != expected:
        raise ThreeStToIftPredictionError(f"{label} hash mismatch: {digest}")


def _frozen_root(config: Mapping[str, Any]) -> Path:
    root = Path(str(config.get("frozen_artifact_root") or project_root()))
    if not root.is_dir():
        raise ThreeStToIftPredictionError(f"frozen_artifact_root is missing: {root}")
    return root


def inherit_frozen_stage(config: Mapping[str, Any]) -> dict[str, Any]:
    inherited: dict[str, Any] = {}
    frozen_root = _frozen_root(config)
    for workbook, extra in (
        ("workbook_87", {"must_not_reopen_transport": True}),
        ("workbook_95", {"dummy_still_rejected": True}),
        ("workbook_96", {"four_station_5x5_is_not_without_ift": True}),
        ("workbook_109", {"lto_is_not_without_ift": True}),
        ("workbook_117", {"authorizes_this_chain_only": True}),
    ):
        spec = config["inheritance"][workbook]
        root = project_root() if spec.get("artifact_root") == "local" else frozen_root
        decision = json.loads(
            resolve_under_root(root, spec["decision_path"]).read_text(encoding="utf-8")
        )
        _expect_sha(
            resolve_under_root(project_root(), spec["config_path"]),
            spec["config_sha256"],
            f"{workbook} config",
        )
        _expect_sha(
            resolve_under_root(root, spec["decision_path"]),
            spec["decision_sha256"],
            f"{workbook} decision",
        )
        if decision.get("decision") != spec["frozen_decision"]:
            raise ThreeStToIftPredictionError(f"{workbook} decision must stay frozen")
        inherited[workbook] = {
            "decision": decision["decision"],
            "config_sha256": spec["config_sha256"],
            "decision_sha256": spec["decision_sha256"],
            "artifact_root": str(root),
            **extra,
        }
    if inherited["workbook_117"]["decision"] != "ckf_without_ift_definition_established":
        raise ThreeStToIftPredictionError("Stage 0 must PASS before Stage 1")
    return inherited


def verify_pinned_calypso_sources(config: Mapping[str, Any]) -> dict[str, Any]:
    spec = config["software_provenance"]
    root = Path(spec["calypso_root"])
    report: dict[str, Any] = {
        "calypso_root": str(root),
        "expected_git_sha": spec["calypso_git_sha"],
        "files": {},
        "all_match": True,
    }
    for name, item in config["pinned_calypso_sources"].items():
        path = root / str(item["path"])
        if not path.is_file():
            report["files"][name] = {"path": str(path), "present": False}
            report["all_match"] = False
            continue
        digest = sha256_file(path)
        match = digest == str(item["sha256"])
        report["files"][name] = {
            "path": str(path),
            "present": True,
            "sha256": digest,
            "expected": item["sha256"],
            "match": match,
        }
        if not match:
            report["all_match"] = False
    return report


def dump_path_for_source(config: Mapping[str, Any], source_id: str, kind: str) -> Path:
    root = resolve_under_root(project_root(), str(config["dump_root"]))
    filename = (
        str(config["dump_filename"]) if kind == "tracks" else str(config["event_filename"])
    )
    return root / source_id / filename


def load_jsonl(path: Path, *, split: str) -> list[dict[str, Any]]:
    authorized = authorize_path(
        path, AccessScope.DEVELOPMENT_VALIDATION, split=access_split(split)
    )
    if not authorized.is_file():
        return []
    rows = []
    for line in authorized.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _as_matrix(payload: Any) -> np.ndarray | None:
    if payload is None:
        return None
    matrix = np.asarray(payload, dtype=np.float64)
    if matrix.shape != (5, 5):
        return None
    return matrix


def _finite_stats(values: list[float]) -> dict[str, int | float | None]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = array[np.isfinite(array)]
    result: dict[str, int | float | None] = {
        "count": int(array.size),
        "finite_count": int(finite.size),
        "min": None,
        "median": None,
        "max": None,
    }
    if finite.size:
        result.update(
            {
                "min": float(np.min(finite)),
                "median": float(np.median(finite)),
                "max": float(np.max(finite)),
            }
        )
    return result


def _model_success(row: Mapping[str, Any], key: str) -> bool:
    payload = row.get(key)
    return isinstance(payload, Mapping) and bool(payload.get("success"))


def evaluate_events(events: list[Mapping[str, Any]]) -> dict[str, Any]:
    n_with_ift = 0
    n_selection_loss = 0
    n_cluster_unavailable = 0
    n_decoded = 0
    for row in events:
        if bool(row.get("cluster_stations_unavailable")):
            n_cluster_unavailable += 1
        if int(row.get("n_ift_clusters") or 0) > 0:
            n_with_ift += 1
        if bool(row.get("selection_loss")):
            n_selection_loss += 1
        if bool(row.get("station_decoded_by_fasersct_id")):
            n_decoded += 1
    return {
        "n_events": len(events),
        "n_events_with_ift_clusters": n_with_ift,
        "n_events_cluster_unavailable": n_cluster_unavailable,
        "n_events_selection_loss": n_selection_loss,
        "n_events_station_decoded": n_decoded,
    }


def evaluate_tracks(
    records: list[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    dummy_q = float(config["acceptance"]["dummy_qoverp_per_mev"])
    dummy_v = float(config["acceptance"]["dummy_variance_per_mev2"])
    tol = float(config["acceptance"]["dummy_match_tolerance"])
    selected = [
        row
        for row in records
        if str(row.get("collection")) == SOURCE_COLLECTION_NAME
        and str(row.get("kind", "track")) == "track"
    ]
    qoverp: list[float] = []
    residuals: list[float] = []
    associated_residuals: list[float] = []
    dummy_q_hits = 0
    dummy_v_hits = 0
    truth_hits = 0
    missing_cov = 0
    spd_ok = 0
    cross_nonzero = 0
    ift_mot = 0
    ift_tsos = 0
    decoded = 0
    undecoded = 0
    plane0 = 0
    plane1 = 0
    start_fail = 0
    leak = 0
    used_ift = 0
    n_residual_rows = 0
    n_residual_available = 0
    n_surface_failed = 0
    n_surface_missing = 0
    n_tracks_with_residual = 0
    failure_hist = {name: 0 for name in FAILURE_CLASSES}
    for row in selected:
        if bool(row.get("is_truth", False)):
            truth_hits += 1
            continue
        if bool(row.get("prediction_uses_ift_measurement")):
            used_ift += 1
        if bool(row.get("station_decoded_by_fasersct_id")):
            decoded += 1
        else:
            undecoded += 1
        if int(row.get("n_ift_mot") or 0) > 0:
            ift_mot += 1
        if int(row.get("n_ift_tsos") or 0) > 0:
            ift_tsos += 1
        if bool(row.get("ift_leak")):
            leak += 1
        klass = row.get("primary_failure_class")
        if klass in failure_hist:
            failure_hist[str(klass)] += 1
        if str(klass) == "acts_start_unavailable":
            start_fail += 1
        if _model_success(row, "model0_no_process_noise"):
            plane0 += 1
        if _model_success(row, "model1_acts_process_noise"):
            plane1 += 1
        state = np.asarray(row.get("native_state"), dtype=np.float64).reshape(-1)
        if state.size == 5 and np.isfinite(state).all():
            q_value = float(state[QOVERP_INDEX])
            qoverp.append(q_value)
            if abs(q_value - dummy_q) <= tol:
                dummy_q_hits += 1
        cov = _as_matrix(row.get("native_covariance"))
        if cov is None or not bool(row.get("has_covariance", False)):
            missing_cov += 1
        else:
            if abs(float(cov[QOVERP_INDEX, QOVERP_INDEX]) - dummy_v) <= tol:
                dummy_v_hits += 1
            if is_spd(cov):
                spd_ok += 1
            if np.any(np.abs(cov[QOVERP_INDEX, :QOVERP_INDEX]) > 0.0):
                cross_nonzero += 1
        track_residual = False
        for item in row.get("residuals") or []:
            n_residual_rows += 1
            if bool(item.get("residual_available")):
                n_residual_available += 1
                track_residual = True
                value = item.get("residual_loc0_mm")
                if value is not None and np.isfinite(value):
                    residuals.append(float(value))
                    if bool(item.get("reconstruction_associated")):
                        associated_residuals.append(float(value))
            elif str(item.get("failure_class")) == "propagate_surface_failed":
                n_surface_failed += 1
            elif str(item.get("failure_class")) in {
                "surface_not_in_identifier_map",
                "surface_pointer_null",
            }:
                n_surface_missing += 1
        if track_residual:
            n_tracks_with_residual += 1
    return {
        "collection": SOURCE_COLLECTION_NAME,
        "n_records": len(selected),
        "n_truth_rejected": truth_hits,
        "n_station_decoded": decoded,
        "n_station_undecoded": undecoded,
        "n_tracks_with_ift_measurements_on_track": ift_mot,
        "n_tracks_with_ift_tsos": ift_tsos,
        "n_ift_leak": leak,
        "n_prediction_used_ift_measurement": used_ift,
        "n_missing_covariance": missing_cov,
        "n_spd_ok": spd_ok,
        "n_dummy_qoverp": dummy_q_hits,
        "n_dummy_variance": dummy_v_hits,
        "n_nonzero_qoverp_cross_term": cross_nonzero,
        "n_plane_model0_success": plane0,
        "n_plane_model1_success": plane1,
        "n_acts_start_unavailable": start_fail,
        "n_residual_rows": n_residual_rows,
        "n_residual_available": n_residual_available,
        "n_tracks_with_residual": n_tracks_with_residual,
        "n_surface_propagate_failed": n_surface_failed,
        "n_surface_missing": n_surface_missing,
        "failure_class_counts": failure_hist,
        "q_over_p_per_mev": _finite_stats(qoverp),
        "residual_loc0_mm": _finite_stats(residuals),
        "associated_residual_loc0_mm": _finite_stats(associated_residuals),
        "warning_count_is_not_reconstruction_loss": True,
    }


def inventory_split(config: Mapping[str, Any], split: str) -> dict[str, Any]:
    sources = []
    tracks: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    dumps_present = True
    for spec in config["mc_data"][f"{split}_sources"]:
        xaod = authorize_path(
            spec["input_xaod"],
            AccessScope.DEVELOPMENT_VALIDATION,
            split=access_split(split),
        )
        track_path = dump_path_for_source(config, spec["source_id"], "tracks")
        event_path = dump_path_for_source(config, spec["source_id"], "events")
        track_rows = load_jsonl(track_path, split=split)
        event_rows = load_jsonl(event_path, split=split)
        if not track_rows or not event_rows:
            dumps_present = False
        tracks.extend(track_rows)
        events.extend(event_rows)
        sources.append(
            {
                "source_id": spec["source_id"],
                "input_xaod": str(xaod),
                "n_track_rows": len(track_rows),
                "n_event_rows": len(event_rows),
            }
        )
    return {
        "split": split,
        "sources": sources,
        "dumps_present": dumps_present and bool(sources),
        "events": evaluate_events(events),
        "tracks": evaluate_tracks(tracks, config),
    }


def decide(
    construction: Mapping[str, Any],
    validation: Mapping[str, Any],
    pins: Mapping[str, Any],
    inherited: Mapping[str, Any],
    *,
    dumps_materialized: bool,
) -> dict[str, Any]:
    min_tracks = 3
    min_ift_events = 1
    min_plane = 1
    if inherited.get("workbook_117", {}).get("decision") != (
        "ckf_without_ift_definition_established"
    ):
        mechanism = MECHANISM_STAGE0
        verdict = "FAIL"
    elif not pins.get("all_match", False):
        mechanism = MECHANISM_SOURCE_PIN
        verdict = "FAIL"
    elif not dumps_materialized:
        mechanism = MECHANISM_DUMP_MISSING
        verdict = "FAIL"
    elif int(construction["tracks"]["n_truth_rejected"]) or int(
        validation["tracks"]["n_truth_rejected"]
    ):
        mechanism = MECHANISM_TRUTH
        verdict = "FAIL"
    elif int(construction["tracks"]["n_prediction_used_ift_measurement"]) or int(
        validation["tracks"]["n_prediction_used_ift_measurement"]
    ):
        mechanism = MECHANISM_IFT_LEAK
        verdict = "FAIL"
    elif int(construction["tracks"]["n_ift_leak"]) or int(
        validation["tracks"]["n_ift_leak"]
    ):
        mechanism = MECHANISM_IFT_LEAK
        verdict = "FAIL"
    elif int(construction["tracks"]["n_tracks_with_ift_measurements_on_track"]) or int(
        construction["tracks"]["n_tracks_with_ift_tsos"]
    ) or int(validation["tracks"]["n_tracks_with_ift_measurements_on_track"]) or int(
        validation["tracks"]["n_tracks_with_ift_tsos"]
    ):
        mechanism = MECHANISM_IFT_LEAK
        verdict = "FAIL"
    elif int(construction["events"]["n_events_with_ift_clusters"]) < min_ift_events or int(
        validation["events"]["n_events_with_ift_clusters"]
    ) < min_ift_events:
        mechanism = MECHANISM_NO_IFT
        verdict = "FAIL"
    elif int(construction["tracks"]["n_records"]) < min_tracks or int(
        validation["tracks"]["n_records"]
    ) < min_tracks:
        mechanism = MECHANISM_NO_TRACKS
        verdict = "FAIL"
    elif int(construction["tracks"]["n_dummy_qoverp"]) == int(
        construction["tracks"]["n_records"]
    ) and int(validation["tracks"]["n_dummy_qoverp"]) == int(
        validation["tracks"]["n_records"]
    ):
        mechanism = MECHANISM_DUMMY
        verdict = "FAIL"
    elif int(construction["tracks"]["n_missing_covariance"]) or int(
        validation["tracks"]["n_missing_covariance"]
    ):
        mechanism = MECHANISM_NO_5X5
        verdict = "FAIL"
    elif int(construction["tracks"]["n_acts_start_unavailable"]) == int(
        construction["tracks"]["n_records"]
    ) and int(validation["tracks"]["n_acts_start_unavailable"]) == int(
        validation["tracks"]["n_records"]
    ):
        mechanism = MECHANISM_START
        verdict = "FAIL"
    elif int(construction["tracks"]["n_plane_model1_success"]) < min_plane or int(
        validation["tracks"]["n_plane_model1_success"]
    ) < min_plane:
        mechanism = MECHANISM_PROP
        verdict = "FAIL"
    else:
        mechanism = None
        verdict = "PASS"
    established = verdict == "PASS"
    residual_ok = established and int(
        construction["tracks"]["n_residual_available"]
    ) >= 1 and int(validation["tracks"]["n_residual_available"]) >= 1
    return {
        "kind": "three_st_to_ift_prediction_contract",
        "task": TASK,
        "workbook": WORKBOOK,
        "verdict": verdict,
        "decision": DECISION_ESTABLISHED if established else DECISION_NOT_ESTABLISHED,
        "mechanism": mechanism,
        "source_collection": SOURCE_COLLECTION_NAME,
        "independent_measurement_container": INDEPENDENT_MEASUREMENT_CONTAINER,
        "contrast_collection": FOUR_STATION_COLLECTION,
        "state_definition": state_definition(),
        "residual_definition": residual_definition(),
        "lto_is_not_this_collection": True,
        "four_station_ckf_is_not_prediction_source": True,
        "prediction_uses_ift_measurement": False,
        "warning_count_is_not_reconstruction_loss": True,
        "boundary_scan_deferred_to_stage_6": True,
        "mc_qp_calibration_authorized": established,
        "residual_conditional_authorized": residual_ok,
        "measurement_model_v2_entered": False,
        "geometry_write_allowed": False,
        "held_out_accessed": False,
        "real_data_alignment_authorized": False,
        "b14m_reopen_authorized": False,
        "b15_authorized": False,
        "dummy_qoverp_per_mev": SEGMENTFIT_DUMMY_QOVERP_PER_MEV,
        "construction": {
            "n_plane_model1_success": construction["tracks"]["n_plane_model1_success"],
            "n_residual_available": construction["tracks"]["n_residual_available"],
            "n_events_selection_loss": construction["events"]["n_events_selection_loss"],
            "failure_class_counts": construction["tracks"]["failure_class_counts"],
        },
        "validation": {
            "n_plane_model1_success": validation["tracks"]["n_plane_model1_success"],
            "n_residual_available": validation["tracks"]["n_residual_available"],
            "n_events_selection_loss": validation["events"]["n_events_selection_loss"],
            "failure_class_counts": validation["tracks"]["failure_class_counts"],
        },
    }


def provenance_hashes(config: Mapping[str, Any]) -> dict[str, str]:
    spec = config["software_provenance"]
    tags = config["tags"]
    calypso = spec["calypso_git_sha"]
    material = Path(str(config["material_map"]["path"]))
    material_sha = sha256_file(material) if material.is_file() else "absent"
    return {
        "geometry_hash": hashlib.sha256(
            f"{tags['geometry']}|{calypso}".encode("utf-8")
        ).hexdigest(),
        "field_hash": hashlib.sha256(
            f"{tags['field']}|{calypso}".encode("utf-8")
        ).hexdigest(),
        "conditions_hash": hashlib.sha256(
            f"{tags['conditions']}|{tags['database_instance']}|{calypso}".encode("utf-8")
        ).hexdigest(),
        "material_hash": hashlib.sha256(
            f"{material}|{material_sha}|{calypso}".encode("utf-8")
        ).hexdigest(),
        "calypso_git_sha": calypso,
        "athena_release": spec["athena_release"],
        "acts_version": spec["acts_version"],
    }
