"""Nov-2022 metrology provenance and Calypso station-ry transform contract.

Read-only inventory of original Nov-2022 measurements, plus a software
finite-difference validation of /Tracker/Align/Stations ry.  Does not
retrain, write geometry, map Kabsch/dz-vs-x to station ry, construct
covariance from population scatter, or enter Fisher.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from alignment.cad_survey_nov22 import (
    DECISION as ENTRY66_DECISION,
    git_head_sha,
    parse_cad_survey_nov22_file,
    refuse_unconfirmed_ry_mapping,
    reproduce_quoted_summaries,
)
from alignment.operating_protocol_v1_final_closure import (
    FROZEN_V2_CHECKPOINT_SHA256,
    OPERATING_MODE,
    RESIDUAL_DECREASE_LABEL,
    project_root,
    resolve_under_root,
    sha256_file,
)
from alignment.survey_metrology_iov_infrastructure import (
    empty_constraint_catalog,
    load_infrastructure_config,
    validate_constraint,
)
from alignment.true_cluster_local_residual import RESIDUAL_KIND, common_operating_state

SCHEMA_VERSION = "faser-nov22-metrology-provenance-station-ry-contract-v1"
DEFAULT_CONFIG_RELATIVE = Path("configs") / "nov22_metrology_provenance_station_ry_contract_v1.yaml"
DECISION = (
    "nov22_raw_covariance_and_iov_unresolved_"
    "station_ry_is_global_left_multiply_about_faser_origin"
)
CONFIG_MUST_BE_FALSE = (
    "geometry_write_allowed",
    "station_calibration_mode_available",
    "cdx_mode_allowed",
    "alignment_payload_from_self_nulling_residuals",
    "geometry_candidate_from_cad_survey",
    "geometry_candidate_from_software_fd",
)
CONFIG_MUST_BE_TRUE = (
    "frozen",
    "do_not_retrain_v2",
    "do_not_modify_frozen_v2_checkpoint",
    "do_not_construct_station_calibration_mode",
    "do_not_construct_reduced_station_mode",
    "do_not_enter_cdx_mode",
    "do_not_run_newton",
    "do_not_solve_alignment_correction",
    "do_not_write_official_conditions",
    "do_not_use_tracklet_intercept_as_cluster_residual",
    "do_not_enter_full_module_identifiability_map",
    "do_not_invent_new_cosine_cut",
    "do_not_select_events_from_residual_or_cosine",
    "do_not_select_prior_from_residual",
    "do_not_restack_2024_r0022_collision_like",
    "do_not_rebuild_2024_r0022_jacobian",
    "do_not_mix_cross_year_residuals_or_alignment_constants",
    "do_not_average_across_iovs",
    "do_not_invent_survey_numbers",
    "do_not_use_design_as_survey",
    "do_not_use_software_gauge_as_survey",
    "do_not_treat_conditions_as_independent_survey",
    "do_not_upgrade_numeric_agreement_to_survey_derived",
    "do_not_treat_ppt_as_alignment_prior",
    "do_not_treat_ppt_as_geometry",
    "do_not_map_unconfirmed_tilt_to_ry",
    "do_not_use_station_sd_as_measurement_sigma",
    "do_not_equate_layer_trend_to_calypso_station_ry",
    "do_not_map_stereo_or_layerpitch_to_station_ry",
    "do_not_map_planes_rotations_to_station_ry",
    "do_not_map_kabsch_or_dz_vs_x_to_station_ry",
    "do_not_guess_index_tuple_meanings",
    "do_not_modify_original_survey_file",
    "do_not_expand_ml_or_module_map",
    "do_not_expand_collision_track_statistics",
    "do_not_infer_physics_from_filename",
    "do_not_construct_covariance_from_population_scatter",
    "synthetic_prior_feasibility_only",
    "slide_summary_feasibility_only",
    "software_fd_sensitivity_only",
)


class LiveCalypsoSensorDumpMissingError(ValueError):
    """Raised when a live GeoModelTest aligned dump is treated as completed."""


def load_contract_config(path: str | Path | None = None) -> dict[str, Any]:
    source = Path(path).expanduser().resolve() if path is not None else (
        project_root() / DEFAULT_CONFIG_RELATIVE
    )
    with source.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"nov22 contract config must be a mapping: {source}")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unexpected nov22 contract schema: {source}")
    for key in CONFIG_MUST_BE_TRUE:
        if payload.get(key) is not True:
            raise ValueError(f"nov22 contract config must set {key}=true")
    for key in CONFIG_MUST_BE_FALSE:
        if payload.get(key) is not False:
            raise ValueError(f"nov22 contract config must set {key}=false")
    if payload.get("real_data_operating_mode") != OPERATING_MODE:
        raise ValueError("audit must remain residual_dq_monitoring_only")
    if payload.get("frozen_v2_checkpoint_sha256") != FROZEN_V2_CHECKPOINT_SHA256:
        raise ValueError("audit must not replace the frozen V2 checkpoint SHA256")
    if payload.get("residual_kind") != RESIDUAL_KIND:
        raise ValueError("audit must keep the true cluster-local residual")
    if payload.get("residual_decrease_label") != RESIDUAL_DECREASE_LABEL:
        raise ValueError("residual decrease must stay labeled DQ observable")
    frozen = payload.get("entry_61_frozen") or {}
    if float(frozen.get("degeneracy_breaking_sigma_ry_mrad") or 0.0) != 20.0:
        raise ValueError("entry-61 degeneracy-breaking σ(ry) must stay 20 mrad")
    if float(frozen.get("degeneracy_breaking_sigma_cdx_mm") or 0.0) != 0.315:
        raise ValueError("entry-61 degeneracy-breaking σ(C_dx) must stay 0.315 mm")
    config = dict(payload)
    config["config_path"] = str(source)
    return config


def common_audit_state() -> dict[str, Any]:
    state = common_operating_state()
    state.update(
        {
            "schema_version": SCHEMA_VERSION,
            "emits_alignment_payload": False,
            "geometry_candidate": False,
            "did_train_model": False,
            "did_run_alignment_fit": False,
            "did_write_geometry": False,
            "did_rebuild_2024_r0022_jacobian": False,
            "did_expand_ml_or_module_map": False,
            "did_expand_collision_track_statistics": False,
            "ppt_used_as_alignment_prior": False,
            "ppt_used_as_geometry": False,
            "unconfirmed_tilt_mapped_to_ry": False,
            "conditions_treated_as_independent_survey": False,
            "survey_numbers_were_invented": False,
            "station_sd_used_as_sigma": False,
            "slide_summary_feasibility_only": True,
            "software_fd_sensitivity_only": True,
            "original_survey_file_modified": False,
            "did_enter_fisher": False,
            "did_run_newton": False,
            "did_write_official_conditions": False,
            "live_calypso_sensor_dump_ran": False,
        }
    )
    return state


def _try_listdir(path: Path) -> dict[str, Any]:
    record = {
        "path": str(path),
        "exists": False,
        "readable": False,
        "entries": None,
        "error": None,
    }
    try:
        if not path.exists():
            record["error"] = "does_not_exist"
            return record
        record["exists"] = True
        names = sorted(item.name for item in path.iterdir())
        record["readable"] = True
        record["entries"] = names[:40]
        record["n_entries"] = len(names)
    except OSError as error:
        record["error"] = f"{type(error).__name__}: {error}"
    return record


def station_alignment_transform(
    constants: Sequence[float],
) -> np.ndarray:
    """Return the 4x4 global station transform of TrackerAlignDBTool.

    constants = [dx_mm, dy_mm, dz_mm, rx_rad, ry_rad, rz_rad]
    composition = T(dx,dy,dz) * Rz(rz) * Ry(ry) * Rx(rx)
    Eigen left-multiply: r' = g * r, with homogeneous r = (x,y,z,1).
    """
    if len(constants) != 6:
        raise ValueError("station constants must be [dx, dy, dz, rx, ry, rz]")
    if not all(math.isfinite(float(value)) for value in constants):
        raise ValueError("station constants must be finite")
    dx, dy, dz, rx, ry, rz = (float(value) for value in constants)
    translation = np.eye(4, dtype=np.float64)
    translation[:3, 3] = (dx, dy, dz)
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    rot_x = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, cx, -sx, 0.0],
            [0.0, sx, cx, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    rot_y = np.array(
        [
            [cy, 0.0, sy, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [-sy, 0.0, cy, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    rot_z = np.array(
        [
            [cz, -sz, 0.0, 0.0],
            [sz, cz, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return translation @ rot_z @ rot_y @ rot_x


def apply_station_alignment(
    points_mm: np.ndarray,
    constants: Sequence[float],
) -> np.ndarray:
    """Apply the global station transform to Cartesian points in millimetres."""
    array = np.asarray(points_mm, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    transform = station_alignment_transform(constants)
    homogeneous = np.concatenate(
        [array, np.ones((array.shape[0], 1), dtype=np.float64)], axis=1
    )
    return (homogeneous @ transform.T)[:, :3]


def extract_alpha_beta_gamma(transform: np.ndarray) -> tuple[float, float, float]:
    """TrackerAlignDBTool::extractAlphaBetaGamma on a 4x4 or 3x3 matrix."""
    matrix = np.asarray(transform, dtype=np.float64)
    siny = max(-1.0, min(1.0, float(matrix[0, 2])))
    beta = math.asin(siny)
    if matrix[1, 2] == 0.0 and matrix[2, 2] == 0.0:
        gamma = 0.0
        alpha = math.atan2(float(matrix[1, 1]), float(matrix[2, 1]))
    else:
        alpha = math.atan2(-float(matrix[1, 2]), float(matrix[2, 2]))
        gamma = math.atan2(-float(matrix[0, 1]), float(matrix[0, 0]))
    return alpha, beta, gamma


def rotation_about_point(
    points_mm: np.ndarray,
    *,
    ry_rad: float,
    pivot_mm: Sequence[float],
) -> np.ndarray:
    """Rotate about an arbitrary pivot with the same Ry matrix as Calypso."""
    array = np.asarray(points_mm, dtype=np.float64)
    pivot = np.asarray(pivot_mm, dtype=np.float64)
    relative = array - pivot
    rotated = apply_station_alignment(relative, (0.0, 0.0, 0.0, 0.0, ry_rad, 0.0))
    return rotated + pivot


def jacobian_global_ry(points_mm: np.ndarray) -> np.ndarray:
    """Analytic small-angle ∂(x,y,z)/∂ry for r' = Ry(ry) * r about the origin.

    Ry = [[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]] so at ry=0:
      dx/dry = +z, dy/dry = 0, dz/dry = -x.
    """
    array = np.asarray(points_mm, dtype=np.float64)
    jacobian = np.zeros_like(array)
    jacobian[:, 0] = array[:, 2]
    jacobian[:, 1] = 0.0
    jacobian[:, 2] = -array[:, 0]
    return jacobian


def jacobian_ry_about_pivot(
    points_mm: np.ndarray, pivot_mm: Sequence[float]
) -> np.ndarray:
    array = np.asarray(points_mm, dtype=np.float64)
    pivot = np.asarray(pivot_mm, dtype=np.float64)
    return jacobian_global_ry(array - pivot)


def _rms(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(np.sum(array**2, axis=-1))))


def software_fd_station_ry(
    parsed: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Geometry-only finite difference of /Tracker/Align/Stations ry.

    Uses implied GeoModel nominal coordinates from cad_survey_nov22
    (corrected - offset).  This is the source-traced T*c = g*T contract,
    not a live GeoModelTest dump and not an alignment solve.
    """
    ry = float(config["software_fd"]["ry_rad"])
    evidence = config["calypso_evidence"]
    origin_z = float(evidence["fasernu04_station0_origin_z_mm"])
    parent_z = float(evidence["fasernu04_sct_parent_z_mm"])
    interface_posz = float(evidence["fasernu04_interface_posz_mm"])
    sensors = [
        row
        for row in parsed["sensors"]
        if int(row["station_id"]) == 0 and int(row["side"]) == 0
    ]
    nominal = np.asarray(
        [row["implied_nominal_mm"] for row in sensors], dtype=np.float64
    )
    plus = apply_station_alignment(nominal, (0.0, 0.0, 0.0, 0.0, ry, 0.0))
    minus = apply_station_alignment(nominal, (0.0, 0.0, 0.0, 0.0, -ry, 0.0))
    numeric = (plus - minus) / (2.0 * ry)
    analytic = jacobian_global_ry(nominal)
    centroid = np.mean(nominal, axis=0)
    station_origin = np.array([0.0, 0.0, origin_z], dtype=np.float64)
    support_beam = np.asarray(parsed["delta_faser_mm"], dtype=np.float64)
    about_centroid = jacobian_ry_about_pivot(nominal, centroid)
    about_station = jacobian_ry_about_pivot(nominal, station_origin)
    about_beam = jacobian_ry_about_pivot(nominal, support_beam)
    rms_origin = _rms(numeric - analytic)
    rms_centroid = _rms(numeric - about_centroid)
    rms_station = _rms(numeric - about_station)
    rms_beam = _rms(numeric - about_beam)
    mean_z = float(np.mean(nominal[:, 2]))
    mean_numeric_dx_dry = float(np.mean(numeric[:, 0]))
    mean_analytic_dx_dry = float(np.mean(analytic[:, 0]))
    wb17 = config["workbook_17_frozen_physical_refit"]
    wb17_dx_dry = float(wb17["mean_delta_x_mm_plus"]) / float(wb17["ry_rad"])
    composition = station_alignment_transform((0.0, 0.0, 0.0, 0.0, ry, 0.0))
    extracted = extract_alpha_beta_gamma(composition)
    identity = station_alignment_transform((0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    if not np.allclose(identity, np.eye(4), atol=1.0e-15):
        raise ValueError("identity station transform is not the 4x4 identity")
    if abs(extracted[1] - ry) > 1.0e-12:
        raise ValueError("extractAlphaBetaGamma did not recover the injected ry")
    if rms_origin >= min(rms_centroid, rms_station, rms_beam):
        raise ValueError(
            "numeric Jacobian is not closest to rotation about the FASER origin"
        )
    return {
        **common_audit_state(),
        "kind": "software_fd_sensitivity_only",
        "not_an_alignment_solve": True,
        "not_a_geometry_candidate": True,
        "not_a_survey_prior": True,
        "live_calypso_sensor_dump_ran": False,
        "live_calypso_sensor_dump_reason": (
            "GeoModelTestAlg reads SCT_DetectorManager::getDetectorElementCollection(), "
            "the unaligned DetStore collection. Aligned centers live in the conditions "
            "object SCT_DetectorElementCollection written by FaserSCT_DetectorElementCondAlg. "
            "A live GeoModelTest ±ry dump would therefore reprint as-built positions and "
            "cannot validate Stations ry. The source-traced T*c=g*T contract is validated "
            "numerically here instead. A later aligned dump must read "
            "SCT_DetectorElementCollection, not the manager collection."
        ),
        "ry_rad": ry,
        "n_station0_side0_sensors": int(nominal.shape[0]),
        "point_cloud": "cad_survey_nov22 implied GeoModel nominal = corrected - offset",
        "composition": "T(dx,dy,dz) * Rz(rz) * Ry(ry) * Rx(rx)",
        "application": "r_prime = g * r  (homogeneous left multiply, FASER global origin)",
        "conjugation": "c = T.inverse() * g * T  so T*c = g*T and global points transform as g*r",
        "small_angle_jacobian": {
            "dx_dry": "+z",
            "dy_dry": "0",
            "dz_dry": "-x",
            "pivot": "FASER / Calypso global origin (0,0,0), not the station GeoModel origin",
        },
        "numeric_mean_dxyz_dry_mm_per_rad": [
            float(np.mean(numeric[:, axis])) for axis in range(3)
        ],
        "analytic_mean_dxyz_dry_mm_per_rad": [
            float(np.mean(analytic[:, axis])) for axis in range(3)
        ],
        "rms_numeric_minus_analytic_mm_per_rad": {
            "about_faser_origin": rms_origin,
            "about_sensor_centroid": rms_centroid,
            "about_station_geomodel_origin": rms_station,
            "about_casper_support_beam_delta_faser": rms_beam,
        },
        "closest_pivot": "faser_global_origin",
        "station0_implied_nominal_mean_mm": [float(item) for item in centroid],
        "station0_geomodel_origin_mm": [0.0, 0.0, origin_z],
        "station0_geomodel_origin_construction": (
            f"SCT parent POSZ {parent_z} mm + Interface POSZ {interface_posz} mm "
            f"= {origin_z} mm from geomDB SCTTopLevel-02 / HVS 107793"
        ),
        "support_beam_delta_faser_mm": [float(item) for item in support_beam],
        "mean_z_mm": mean_z,
        "mean_numeric_dx_dry_mm_per_rad": mean_numeric_dx_dry,
        "mean_analytic_dx_dry_mm_per_rad": mean_analytic_dx_dry,
        "analytic_dx_dry_equals_mean_z": bool(abs(mean_analytic_dx_dry - mean_z) < 1.0e-12),
        "numeric_matches_analytic_sinc_factor": bool(
            abs(mean_numeric_dx_dry - mean_analytic_dx_dry * math.sin(ry) / ry) < 1.0e-9
        ),
        "extracted_ry_rad": extracted[1],
        "workbook_17_physical_refit": {
            "observable": wb17["observable"],
            "ry_rad": float(wb17["ry_rad"]),
            "mean_delta_x_mm_plus": float(wb17["mean_delta_x_mm_plus"]),
            "implied_dx_dry_mm_per_rad": wb17_dx_dry,
            "matches_sensor_center_mean_z_sign": bool(wb17_dx_dry < 0.0),
            "not_a_live_sensor_dump": True,
            "not_a_survey_prior": True,
            "note": (
                "Workbook 17 Δx/ry ≈ -1860 mm has the same sign and magnitude as "
                "IFT mean z under r'=Ry*r about the FASER origin. It is a "
                "truth-matched track-state response, not a sensor-center dump, "
                "and is recorded only as independent physical-chain consistency."
            ),
        },
        "forbidden_identifications": {
            "kabsch_centroid_ry": "not /Tracker/Align/Stations ry",
            "dz_vs_x_ry": "not /Tracker/Align/Stations ry",
            "layer_x_slope": "not /Tracker/Align/Stations ry",
            "STEREOANGLE": "not /Tracker/Align/Stations ry",
            "LAYERPITCH": "not /Tracker/Align/Stations ry",
            "Tracker_Align_Planes_ry": "different conjugation, not Stations ry",
            "station_geomodel_origin_ry": (
                "would give dx/dry = +(z - z_station) ≈ layer pitch, not z_global"
            ),
        },
        "evidence": [
            evidence["align_db_tool"] + " stationAlignment T*Rz*Ry*Rx",
            evidence["global_delta"] + " setAlignableTransformGlobalDelta c=T^{-1}*g*T",
            evidence["sct_manager_delta"] + " Stations level 3 is higher-level global",
            evidence["detector_factory"] + " addChannel(/Tracker/Align/Stations, 3, global)",
            evidence["align_cond_alg"] + " FaserSCT_AlignCondAlg applies the container",
            evidence["detector_element"] + " center() from transformHit() * GeoModel origin",
            evidence["geomdb_sql"] + " SCTTopLevel-02 Interface+SCT parent z",
        ],
        "may_enter_geometry_candidate": False,
        "may_fill_measured_ry_slot": False,
    }


def station_ry_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    evidence = config["calypso_evidence"]
    origin_z = float(evidence["fasernu04_station0_origin_z_mm"])
    return {
        **common_audit_state(),
        "parameter": "station_ry",
        "cool_folder": evidence["stations_channel"],
        "cool_level": int(evidence["stations_level"]),
        "declared_frame": evidence["stations_frame"],
        "constants_order": list(evidence["station_constants_order"]),
        "units": {"translation": "mm", "rotation": "rad"},
        "composition": evidence["rotation_convention"],
        "eigen_multiply": "left: transform *= AngleAxis(rz,z); *= AngleAxis(ry,y); *= AngleAxis(rx,x)",
        "global_application": (
            "SiDetectorManager::setAlignableTransformGlobalDelta: "
            "c = T.inverse() * g * T, T = child fullPhysVol getDefAbsoluteTransform. "
            "Therefore T*c = g*T and a global point transforms as r' = g * r."
        ),
        "pivot": {
            "name": "FASER / Calypso global origin",
            "coordinates_mm": [0.0, 0.0, 0.0],
            "not_station_geomodel_origin": True,
            "not_sensor_centroid": True,
            "not_casper_support_beam": True,
            "station0_geomodel_origin_mm": [0.0, 0.0, origin_z],
            "why_not_station_origin": (
                "The conjugation puts the global g to the left of T. "
                "A pure Ry therefore rotates every sensor about the FASER origin, "
                "not about the station volume origin at z=-1860.15 mm. "
                "Small-angle dx/dry = +z_global, not +(z - z_station)."
            ),
        },
        "local_vs_global_axes": {
            "Stations": "global, level 3",
            "Planes": (
                "global, level 2, but conjugated about Translate(0,0, element z) "
                "in TrackerAlignDBTool; not the same object as Stations ry"
            ),
            "Interface/Upstream/Central/Downstream modules": "local, level 1",
            "dirkey_inversion": (
                "TrackerAlignDBTool::dirkey uses level 1 = Stations, "
                "level 2 = Planes, level 3 = modules; "
                "SCT_DetectorFactory addChannel uses the opposite numbering "
                "(Stations=3, Planes=2, modules=1). Both describe the same folders."
            ),
        },
        "small_angle_derivatives": {
            "rx": {"dx": "0", "dy": "-z", "dz": "+y"},
            "ry": {"dx": "+z", "dy": "0", "dz": "-x"},
            "rz": {"dx": "-y", "dy": "+x", "dz": "0"},
        },
        "aligned_center_path": (
            "FaserSCT_AlignCondAlg writes SCTAlignmentStore; "
            "FaserSCT_DetectorElementCondAlg copies SiDetectorElement objects "
            "bound to that store into SCT_DetectorElementCollection; "
            "SiDetectorElement::center() uses transformHit() which reads the store. "
            "GeoModelTestAlg currently dumps the unaligned manager collection."
        ),
        "validated_by": "numeric T*c=g*T finite difference on cad_survey implied nominals",
        "live_calypso_sensor_dump": "not_run_because_geomodeltest_dumps_unaligned_manager_collection",
        "maps_survey_rigid_estimator_to_stations_ry": False,
        "reason_survey_still_unmapped": (
            "The software contract of Stations ry is now unique: global left-multiply "
            "about the FASER origin. A survey rigid-body estimator still has to be "
            "the same object (same pivot, same axis, independent covariance, IOV) "
            "before it can fill ift_station0_ry. Kabsch about the sensor centroid "
            "and dz-vs-x about mean x are different estimators."
        ),
        "evidence": [
            evidence["align_db_tool"],
            evidence["global_delta"],
            evidence["sct_manager_delta"],
            evidence["detector_factory"],
            evidence["align_cond_alg"],
            evidence["detector_element_cond_alg"],
            evidence["detector_element"],
            evidence["geomdb_sql"],
            evidence["toplevel_placements"],
            evidence["geomodel_test"],
        ],
        "may_enter_geometry_candidate": False,
    }


def provenance_inventory(config: Mapping[str, Any]) -> dict[str, Any]:
    root = project_root()
    source = resolve_under_root(root, str(config["source"]["relative"]))
    pdf_2022 = resolve_under_root(root, str(config["adjacent_sources"]["survey_2022_pdf"]))
    pdf_2021 = resolve_under_root(root, str(config["adjacent_sources"]["survey_2021_pdf"]))
    searches = [
        {
            "location": str(root),
            "what": "workspace survey/metrology/CAD/xlsx/csv/step besides cad_survey_nov22 and the two PDFs",
            "found": [
                "docs/cad_survey_nov22.txt",
                "docs/04Nov2022_Survey.pdf",
                "docs/FAS_IFT_Dec10-2021.pdf",
            ],
            "raw_instrument_table": False,
        },
        {
            "location": "/eos/home-x/xcheng/FASER/calypso",
            "what": "Calypso tree filenames or source strings Delta FASER / cad_survey_nov22 / 04Nov2022",
            "found": [],
            "raw_instrument_table": False,
            "producing_script_identified": False,
        },
        {
            "location": "/eos/experiment/faser",
            "what": "depth-2 survey/metrology/align/cad/geo names",
            "found": [],
            "raw_instrument_table": False,
            "readme": (
                "30 April 2024: reconstruction outputs live under "
                "/eos/experiment/faser/data0; top-level dirs are "
                "data0/filter/gen/phys/raw/rec/runlist/sim/staged/Test"
            ),
        },
        {
            "location": "/cvmfs/faser.cern.ch/repo/sw/database/DBRelease/current/poolcond",
            "what": "survey-named files",
            "found": [
                "FASER-04_2022_Align.pool.root",
                "FASER-05_2023_Align.pool.root",
                "FASER-05_2024_Align.pool.root",
                "FASER-06_2023_Align.pool.root",
                "FASER-06_2024_Align.pool.root",
                "FASER-06_2025_Align.pool.root",
            ],
            "raw_instrument_table": False,
            "note": "yearly Align POOL files are reconstruction conditions, not survey",
        },
    ]
    author_homes = {
        "fcadoux_home": _try_listdir(Path("/eos/home-f/fcadoux")),
        "fcadoux_user": _try_listdir(Path("/eos/user/f/fcadoux")),
        "dcasper_home": _try_listdir(Path("/eos/home-d/dcasper")),
        "dcasper_user": _try_listdir(Path("/eos/user/d/dcasper")),
        "casper_user": _try_listdir(Path("/eos/user/c/casper")),
        "casper_home": _try_listdir(Path("/eos/home-c/casper")),
    }
    return {
        **common_audit_state(),
        "source_file": {
            "relative": config["source"]["relative"],
            "path": str(source),
            "exists": source.is_file(),
            "sha256": sha256_file(source) if source.is_file() else None,
            "expected_sha256": config["source"]["expected_sha256"],
            "size_bytes": source.stat().st_size if source.is_file() else None,
            "role": "Casper comparison dump of Franck/Cadoux CAD vs GeoModelTest, not the raw instrument table",
        },
        "adjacent_pdfs": {
            "2022": {
                "relative": config["adjacent_sources"]["survey_2022_pdf"],
                "path": str(pdf_2022),
                "exists": pdf_2022.is_file(),
                "sha256": sha256_file(pdf_2022) if pdf_2022.is_file() else None,
                "author": "D. Casper, survey data from Franck/Cadoux",
                "date": "2022-11-04",
                "mechanical_state_recorded": False,
                "opening_closing_thermal_recorded": False,
                "instrument_precision_recorded": False,
                "per_point_covariance_recorded": False,
            },
            "2021": {
                "relative": config["adjacent_sources"]["survey_2021_pdf"],
                "path": str(pdf_2021),
                "exists": pdf_2021.is_file(),
                "sha256": sha256_file(pdf_2021) if pdf_2021.is_file() else None,
                "role": "Cadoux 2021-12-10 CAD-frame I/F survey; not this dump",
            },
        },
        "producing_script": {
            "identified": False,
            "searched": [
                str(root),
                "/eos/home-x/xcheng/FASER/calypso",
                "/eos/experiment/faser",
            ],
            "pdf_says": (
                "Use CalypsoExamples/GeoModelTest to dump as-built sensor positions, "
                "no alignment applied; wafer IDs assigned manually; Stations 1-3 "
                "front-sensor averages fix the support-beam translation."
            ),
        },
        "raw_measurement_objects": {
            "wafer_sensor_csv_xlsx": False,
            "laser_tracker_theodolite": False,
            "unige_cmm_nov22_table": False,
            "cern_metrology_nov22_table": False,
            "repeat_survey": False,
            "instrument_precision_for_this_dump": False,
            "fit_covariance_for_this_dump": False,
            "literature_unige_cmm_nima_1034": {
                "present_as_paper": True,
                "usable_as_nov22_covariance": False,
                "reason": (
                    "arXiv:2112.01116 / NIMA 1034 (2022) 166825 describes UNIGE "
                    "plane-assembly CMM (~5 µm in-plane, 10-15 µm out-of-plane). "
                    "That is not the Nov-2022 CAD-vs-GeoModel wafer table."
                ),
            },
        },
        "filename_physics_inference": False,
        "author_homes": author_homes,
        "author_homes_readable": False,
        "searches": searches,
        "ingestion_layer_built": False,
        "ingestion_layer_reason": (
            "No original measurement table was found. Building a machine-readable "
            "ingestion layer from cad_survey_nov22 quoted means would invent "
            "instrument/operator/uncertainty fields that are not in the file."
        ),
        "git_sha": git_head_sha(root),
        "config_sha256": sha256_file(Path(config["config_path"])),
        "may_enter_geometry_candidate": False,
    }


def iov_provenance(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        **common_audit_state(),
        "survey_talk_date": "2022-11-04",
        "survey_data_from": "Franck / Cadoux CAD+survey, presented by D. Casper",
        "mechanical_state_recorded_in_pdf_or_dump": False,
        "opening_closing_recorded": False,
        "thermal_state_recorded": False,
        "installation_state_recorded": False,
        "may_constrain_year": [2022],
        "may_constrain_conditions_tags": [],
        "may_be_used_as_2024_or_2025_station_rigid_body": False,
        "independent_stability_evidence_2022_to_2024": False,
        "independent_stability_evidence_2022_to_2025": False,
        "yearly_align_pool_is_survey": False,
        "yearly_align_pool_files": [
            "FASER-04_2022_Align.pool.root",
            "FASER-05_2023_Align.pool.root",
            "FASER-05_2024_Align.pool.root",
            "FASER-06_2023_Align.pool.root",
            "FASER-06_2024_Align.pool.root",
            "FASER-06_2025_Align.pool.root",
        ],
        "station_rigid_motion_policy": "IOV-specific by default",
        "C_dx_policy": "common_static_candidate_until_opening_or_thermal_evidence",
        "C_dx_upgraded_to_common_static": False,
        "do_not_average_across_years_or_tags": True,
        "reason": (
            "The talk date is 2022-11-04. Neither the PDF nor cad_survey_nov22 "
            "records opening/closing/thermal/mechanical state. CERN backbone "
            "survey and UNIGE plane CMM predate Nov-22 and do not prove "
            "2022→2024/2025 station rigid stability. Yearly Align POOL files "
            "are reconstruction conditions, not survey."
        ),
        "required_evidence_to_use_in_2024_or_2025": [
            "named mechanical/installation/opening/closing/thermal state of the Nov-2022 survey",
            "independent stability measurement covering 2022 to the target year",
            "explicit statement of which conditions tag the survey may constrain",
        ],
        "may_enter_geometry_candidate": False,
    }


def official_constraint_slots(
    config: Mapping[str, Any],
    *,
    reproduced: Mapping[str, Any],
    covariance_ok: bool,
    ry_mapping_ok: bool,
    iov_ok: bool,
) -> list[dict[str, Any]]:
    infrastructure = load_infrastructure_config(
        resolve_under_root(project_root(), str(config["adjacent_sources"]["infrastructure_config"]))
    )
    slots = empty_constraint_catalog(infrastructure)
    filled = []
    for slot in slots:
        item = dict(slot)
        item["may_enter_geometry_candidate"] = False
        if item["constraint_id"] in {"ift_C_dx", "ift_l0_minus_l2_dx"}:
            value = (
                reproduced["quoted_C_dx_mm"]
                if item["constraint_id"] == "ift_C_dx"
                else reproduced["quoted_delta_x_L0_minus_L2_mm"]
            )
            item.update(
                {
                    "value": [float(value)],
                    "sigma": None,
                    "covariance": None,
                    "availability": "feasibility_only",
                    "source_file": config["source"]["relative"],
                    "provenance": (
                        "cad_survey_nov22 quoted layer means from entry 66. "
                        "Central value only; Sigma is population spread."
                    ),
                    "year": 2022,
                    "iov_id": "2022_survey_casper_nov04_not_a_2024_conditions_iov",
                    "ingest_gates": {
                        "parameter_mapping_validated": True,
                        "measurement_covariance_has_independent_provenance": covariance_ok,
                        "measurement_year_conditions_iov_identified": iov_ok,
                    },
                    "note": (
                        "Not promoted to availability=measured: covariance and "
                        "cross-year IOV gates fail."
                    ),
                }
            )
        if item["constraint_id"] == "ift_station0_ry":
            item.update(
                {
                    "availability": "unavailable",
                    "value": None,
                    "sigma": None,
                    "ingest_gates": {
                        "parameter_mapping_validated": ry_mapping_ok,
                        "measurement_covariance_has_independent_provenance": covariance_ok,
                        "measurement_year_conditions_iov_identified": iov_ok,
                    },
                    "note": (
                        "Software Stations-ry contract is now unique (global "
                        "left-multiply about the FASER origin). No survey "
                        "estimator with that pivot, an independent covariance, "
                        "and a 2024/2025 IOV was found."
                    ),
                }
            )
        validate_constraint(item)
        filled.append(item)
    return filled


def decide_next_stage(
    *,
    provenance: Mapping[str, Any],
    contract: Mapping[str, Any],
    fd_result: Mapping[str, Any],
    iov: Mapping[str, Any],
    slots: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    covariance_ok = bool(provenance["raw_measurement_objects"]["fit_covariance_for_this_dump"])
    ry_mapping_ok = False
    iov_ok = bool(iov["may_be_used_as_2024_or_2025_station_rigid_body"])
    blockers = [
        {
            "id": "measurement_covariance",
            "status": "unresolved",
            "what": "per-point or fit measurement covariance with independent provenance",
            "required_evidence": [
                "original Nov-2022 wafer/sensor table (CSV/XLSX/instrument export)",
                "sensor identifier (station, layer, phi_module, eta_module, side)",
                "measured (x,y,z) or offset from a named nominal",
                "3x3 or at least diagonal measurement covariance in that frame",
                "instrument, date, operator, repeat count, source SHA256",
            ],
            "do_not_send": "station/module population standard deviations as sigma",
            "found": False,
        },
        {
            "id": "cad_to_calypso_station_ry_mapping",
            "status": "software_contract_resolved_survey_mapping_unresolved",
            "what": "survey estimator that is the same object as /Tracker/Align/Stations ry",
            "software_contract": {
                "resolved": True,
                "pivot": "FASER global origin (0,0,0)",
                "composition": contract["composition"],
                "r_prime": "g * r",
                "dx_dry": "+z_global",
            },
            "survey_mapping_resolved": False,
            "required_evidence": [
                "a survey rigid rotation about the FASER origin, not the sensor centroid",
                "independent measurement covariance of that rotation",
                "explicit statement that this is /Tracker/Align/Stations ry",
            ],
            "do_not_send": "Kabsch -7.75 mrad, dz-vs-x, LAYERPITCH, STEREOANGLE, Planes ry",
        },
        {
            "id": "iov_provenance",
            "status": "unresolved",
            "what": "year/conditions IOV that this 2022 dump may constrain",
            "resolved_as": "2022_measurement_only",
            "required_evidence_to_use_in_2024_or_2025": list(
                iov["required_evidence_to_use_in_2024_or_2025"]
            ),
            "C_dx_policy": iov["C_dx_policy"],
            "station_rigid_motion_policy": iov["station_rigid_motion_policy"],
        },
    ]
    return {
        **common_audit_state(),
        "decision": DECISION,
        "entry66_decision": ENTRY66_DECISION,
        "three_gates": {
            "validated_parameter_mapping": ry_mapping_ok,
            "independent_measurement_covariance": covariance_ok,
            "year_conditions_iov": iov_ok,
        },
        "blocker_status": {
            "measurement_covariance": "unresolved",
            "station_ry_software_contract": "resolved",
            "station_ry_survey_mapping": "unresolved",
            "iov_provenance": "unresolved_2022_measurement_only",
        },
        "have_validated_ry_mapping": ry_mapping_ok,
        "have_measurement_covariance": covariance_ok,
        "have_matching_year_conditions_iov": iov_ok,
        "enough_to_fill_measured": False,
        "enough_to_enter_fisher": False,
        "enough_to_write_alignment_prior": False,
        "enough_to_write_geometry": False,
        "did_enter_fisher": False,
        "did_run_newton": False,
        "did_fill_measured_constraint_slot": False,
        "official_ry_slot_availability": next(
            item["availability"] for item in slots if item["constraint_id"] == "ift_station0_ry"
        ),
        "official_cdx_slot_availability": next(
            item["availability"] for item in slots if item["constraint_id"] == "ift_C_dx"
        ),
        "closest_pivot_of_stations_ry": fd_result["closest_pivot"],
        "ingestion_layer_built": provenance["ingestion_layer_built"],
        "author_homes_readable": provenance["author_homes_readable"],
        "live_calypso_sensor_dump_ran": fd_result["live_calypso_sensor_dump_ran"],
        "blockers": blockers,
        "minimum_request_to_hardware_alignment_survey": [
            {
                "priority": 1,
                "what": "Original Nov-2022 measurement table with covariance",
                "fields": blockers[0]["required_evidence"],
                "do_not_send": blockers[0]["do_not_send"],
            },
            {
                "priority": 1,
                "what": "Survey rigid rotation about the FASER origin, identified as Stations ry",
                "fields": blockers[1]["required_evidence"],
                "do_not_send": blockers[1]["do_not_send"],
            },
            {
                "priority": 1,
                "what": "IOV / mechanical-stability statement",
                "fields": blockers[2]["required_evidence_to_use_in_2024_or_2025"],
            },
        ],
        "next_allowed_step": (
            "Keep residual_dq_monitoring_only. Do not Newton-solve and do not "
            "enter Fisher. Ask Casper/Cadoux for the original Nov-2022 table, "
            "and keep Stations ry defined as global left-multiply about the "
            "FASER origin. Frozen V2 and the association policy stay unchanged."
        ),
        "reason": (
            "A read-only inventory did not find the producing script of "
            "cad_survey_nov22, the original wafer/sensor table, or a "
            "measurement covariance. The Calypso Stations-ry contract is now "
            "source-traced and numerically validated as r'=g*r about the FASER "
            "origin, which is not Kabsch about the sensor centroid and not a "
            "rotation about the station GeoModel origin. The Nov-2022 survey "
            "remains a 2022 measurement. All three ingest gates still fail, so "
            "the official slots stay feasibility_only/unavailable."
        ),
    }


def refuse_live_geomodeltest_as_aligned_dump() -> None:
    raise LiveCalypsoSensorDumpMissingError(
        "GeoModelTestAlg dumps the unaligned SCT_DetectorManager collection; "
        "it is not an aligned sensor-center finite-difference of Stations ry"
    )


def refuse_kabsch_as_station_ry() -> None:
    refuse_unconfirmed_ry_mapping("ift_kabsch_ry")


def build_all_reports(config: Mapping[str, Any]) -> dict[str, Any]:
    root = project_root()
    source = resolve_under_root(root, str(config["source"]["relative"]))
    if not source.is_file():
        raise FileNotFoundError(f"cad_survey_nov22 is missing: {source}")
    digest = sha256_file(source)
    if digest != str(config["source"]["expected_sha256"]):
        raise ValueError("cad_survey_nov22 sha256 does not match the frozen digest")
    parsed = parse_cad_survey_nov22_file(source)
    reproduced = reproduce_quoted_summaries(parsed)
    provenance = provenance_inventory(config)
    contract = station_ry_contract(config)
    fd_result = software_fd_station_ry(parsed, config)
    iov = iov_provenance(config)
    slots = official_constraint_slots(
        config,
        reproduced=reproduced,
        covariance_ok=False,
        ry_mapping_ok=False,
        iov_ok=False,
    )
    decision = decide_next_stage(
        provenance=provenance,
        contract=contract,
        fd_result=fd_result,
        iov=iov,
        slots=slots,
    )
    return {
        "nov22_raw_metrology_provenance": provenance,
        "calypso_station_ry_transform_contract": contract,
        "station_ry_software_fd": fd_result,
        "nov22_iov_provenance": iov,
        "official_constraint_slots": {
            **common_audit_state(),
            "slots": [
                {
                    "constraint_id": item["constraint_id"],
                    "availability": item["availability"],
                    "value": item.get("value"),
                    "sigma": item.get("sigma"),
                    "year": item.get("year"),
                    "iov_id": item.get("iov_id"),
                    "ingest_gates": item.get("ingest_gates"),
                    "may_enter_geometry_candidate": item.get("may_enter_geometry_candidate"),
                }
                for item in slots
            ],
        },
        "next_stage_decision": decision,
    }
