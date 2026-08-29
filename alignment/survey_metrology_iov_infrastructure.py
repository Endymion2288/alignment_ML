"""Survey/metrology ingestion and IOV-aware alignment infrastructure.

Defines the external-constraint schema, IOV state θ_static+Δθ_IOV, a
track+prior Fisher combiner, and a collaboration metrology-requirements
table.  Does not train, rebuild the 2024 r0022 Jacobian, invent survey
numbers, treat design/gauge as measurements, or emit an alignment payload.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from alignment.external_constraint_iov_feasibility import (
    collinear_toy_jacobian,
    column_normalized_spectrum,
    combined_identifiability,
    correlation_ry_cdx,
    fisher_from_augmented,
    jacobian_in_scan_units,
    unlocked,
)
from alignment.layer_hierarchy import IFT_LAYER_IDS, IFT_STATION_ID
from alignment.operating_protocol_v1_final_closure import (
    FROZEN_V2_CHECKPOINT_SHA256,
    OPERATING_MODE,
    RESIDUAL_DECREASE_LABEL,
    project_root,
)
from alignment.true_cluster_local_residual import RESIDUAL_KIND, common_operating_state

SCHEMA_VERSION = "faser-survey-metrology-iov-alignment-infrastructure-v1"
DEFAULT_CONFIG_RELATIVE = Path("configs") / "survey_metrology_iov_alignment_infrastructure_v1.yaml"
DECISION_INGESTION_READY = "survey_metrology_ingestion_ready_awaiting_independent_measurements"
PARAMETER_NAMES = ("station_dx", "station_ry", "C_dx")
SCAN_PARAMETER_UNITS = ("station_dx_mm", "station_ry_mrad", "C_dx_mm")
STATION_NAMES = {0: "IFT_Interface", 1: "Upstream", 2: "Central", 3: "Downstream"}
STATION_COOL_MODULE_PREFIX = {0: "Interface", 1: "Upstream", 2: "Central", 3: "Downstream"}
CONSTRAINT_KINDS = (
    "station_rigid_transform",
    "layer_relative_transform",
    "C_dx",
    "linear_equality",
)
AVAILABILITY = ("unavailable", "feasibility_only", "measured")
FORBIDDEN_AS_MEASUREMENT = (
    "geomdb_layerpitch",
    "design_stereo",
    "write_alignment_neutral_pool",
    "software_dz_gauge_5mm",
    "nominal_station_z",
)
CONFIG_MUST_BE_FALSE = (
    "geometry_write_allowed",
    "station_calibration_mode_available",
    "cdx_mode_allowed",
    "alignment_payload_from_self_nulling_residuals",
    "geometry_candidate_from_synthetic",
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
    "synthetic_prior_feasibility_only",
    "software_fd_sensitivity_only",
)


def load_infrastructure_config(path: str | Path | None = None) -> dict[str, Any]:
    source = Path(path).expanduser().resolve() if path is not None else (
        project_root() / DEFAULT_CONFIG_RELATIVE
    )
    with source.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"infrastructure config must be a mapping: {source}")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unexpected infrastructure schema: {source}")
    for key in CONFIG_MUST_BE_TRUE:
        if payload.get(key) is not True:
            raise ValueError(f"infrastructure config must set {key}=true")
    for key in CONFIG_MUST_BE_FALSE:
        if payload.get(key) is not False:
            raise ValueError(f"infrastructure config must set {key}=false")
    if payload.get("real_data_operating_mode") != OPERATING_MODE:
        raise ValueError("audit must remain residual_dq_monitoring_only")
    if payload.get("frozen_v2_checkpoint_sha256") != FROZEN_V2_CHECKPOINT_SHA256:
        raise ValueError("audit must not replace the frozen V2 checkpoint SHA256")
    if payload.get("residual_kind") != RESIDUAL_KIND:
        raise ValueError("audit must keep the true cluster-local residual")
    if payload.get("residual_decrease_label") != RESIDUAL_DECREASE_LABEL:
        raise ValueError("residual decrease must stay labeled DQ observable")
    physics = payload.get("physics_scales") or {}
    if float(physics.get("measurement_sigma_mm") or 0.0) != float(physics.get("strip_pitch_mm") or -1.0):
        raise ValueError("measurement sigma must be the strip pitch, not a residual RMS")
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
            "do_not_enter_full_module_identifiability_map": True,
            "do_not_invent_new_cosine_cut": True,
            "do_not_select_events_from_residual_or_cosine": True,
            "do_not_select_prior_from_residual": True,
            "do_not_invent_survey_numbers": True,
            "do_not_use_design_as_survey": True,
            "do_not_rebuild_2024_r0022_jacobian": True,
            "do_not_mix_cross_year_residuals_or_alignment_constants": True,
            "do_not_average_across_iovs": True,
            "schema_version": SCHEMA_VERSION,
            "emits_alignment_payload": False,
            "geometry_candidate": False,
        }
    )
    return state


def empty_measurement_fields() -> dict[str, Any]:
    return {
        "value": None,
        "sigma": None,
        "covariance": None,
        "availability": "unavailable",
        "independent_of_track_residual": True,
        "source_file": None,
        "provenance": None,
        "may_enter_geometry_candidate": False,
    }


def _base_constraint(
    *,
    constraint_id: str,
    kind: str,
    detector_hierarchy: Mapping[str, Any],
    frame: str,
    reference_object: str,
    parameter_names: Sequence[str],
    units: Sequence[str],
    year: int | None = None,
    iov_id: str | None = None,
    geometry_tag: str | None = None,
    conditions_tag: str | None = None,
    mapping_to_scan_parameters: Mapping[str, Any] | None = None,
    note: str = "",
) -> dict[str, Any]:
    if kind not in CONSTRAINT_KINDS:
        raise ValueError(f"unsupported constraint kind: {kind}")
    payload = {
        "constraint_id": constraint_id,
        "kind": kind,
        "detector_hierarchy": dict(detector_hierarchy),
        "frame": frame,
        "reference_object": reference_object,
        "parameter_names": list(parameter_names),
        "units": list(units),
        "year": year,
        "iov_id": iov_id,
        "geometry_tag": geometry_tag,
        "conditions_tag": conditions_tag,
        "mapping_to_scan_parameters": dict(mapping_to_scan_parameters or {}),
        "note": note,
        **empty_measurement_fields(),
    }
    validate_constraint(payload)
    return payload


def validate_constraint(constraint: Mapping[str, Any]) -> None:
    availability = constraint.get("availability")
    if availability not in AVAILABILITY:
        raise ValueError(f"availability must be one of {AVAILABILITY}")
    if constraint.get("kind") not in CONSTRAINT_KINDS:
        raise ValueError(f"kind must be one of {CONSTRAINT_KINDS}")
    provenance = str(constraint.get("provenance") or "")
    if provenance in FORBIDDEN_AS_MEASUREMENT and availability == "measured":
        raise ValueError(f"design or software gauge cannot be a measurement: {provenance}")
    if availability == "unavailable":
        if constraint.get("value") is not None or constraint.get("sigma") is not None:
            raise ValueError("unavailable constraints must keep value=null and sigma=null")
        if constraint.get("may_enter_geometry_candidate"):
            raise ValueError("unavailable constraints cannot be geometry candidates")
    if availability == "feasibility_only" and constraint.get("may_enter_geometry_candidate"):
        raise ValueError("synthetic/feasibility priors cannot enter a geometry candidate")
    if availability == "measured":
        if constraint.get("value") is None or constraint.get("sigma") is None:
            raise ValueError("measured constraints need value and sigma")
        if constraint.get("independent_of_track_residual") is not True:
            raise ValueError("ingestible survey must be independent of track residual")
        if not constraint.get("source_file"):
            raise ValueError("measured constraints need a source file")


def assert_not_geometry_candidate(constraint: Mapping[str, Any]) -> None:
    if constraint.get("may_enter_geometry_candidate"):
        raise ValueError("constraint is marked as a geometry candidate")
    if constraint.get("availability") == "feasibility_only" and constraint.get("geometry_candidate"):
        raise ValueError("feasibility_only synthetic prior leaked into geometry_candidate")


def calypso_detector_hierarchy(config: Mapping[str, Any]) -> dict[str, Any]:
    physics = config["physics_scales"]
    calypso_root = project_root().parent / "calypso"
    factory = calypso_root / str(config["calypso"]["detector_factory"])
    return {
        "align_folder": config["calypso"]["align_folder"],
        "sct_geometry_loads": "/Tracker/Align from SCT_OFL (FaserSCT_GeoModelConfig)",
        "detector_factory_channels": {
            "/Tracker/Align/Stations": {"level": 3, "frame": "global", "meaning": "stations in world"},
            "/Tracker/Align/Planes": {"level": 2, "frame": "global", "meaning": "planes in world"},
            "/Tracker/Align/Interface{1,2,3}": {"level": 1, "frame": "local", "station": 0, "meaning": "IFT modules in planes"},
            "/Tracker/Align/Upstream{1,2,3}": {"level": 1, "frame": "local", "station": 1},
            "/Tracker/Align/Central{1,2,3}": {"level": 1, "frame": "local", "station": 2},
            "/Tracker/Align/Downstream{1,2,3}": {"level": 1, "frame": "local", "station": 3},
        },
        "detector_factory_source": str(factory) if factory.is_file() else config["calypso"]["detector_factory"],
        "align_db_tool_dirkey_note": (
            "TrackerAlignDBTool::dirkey uses inverted level labels "
            "(level 1 → Stations, level 2 → Planes, level 3 → Interface/Upstream/"
            "Central/Downstream). Schema frames follow SCT_DetectorFactory addChannel."
        ),
        "stations": [
            {
                "station_id": station,
                "name": STATION_NAMES[station],
                "cool_module_prefix": STATION_COOL_MODULE_PREFIX[station],
                "nominal_z_mm": float(physics["station_z_nominal_mm"][station]),
                "nominal_z_is_survey": False,
                "is_ift": station == IFT_STATION_ID,
            }
            for station in sorted(STATION_NAMES)
        ],
        "ift_layers": list(IFT_LAYER_IDS),
        "stereo_full_mrad": float(physics["stereo_full_mrad"]),
        "stereo_per_side_mrad": float(physics["stereo_per_side_mrad"]),
        "stereo_is_survey": False,
        "layer_pitch_mm": float(physics["ift_layer_pitch_mm"]),
        "layer_pitch_is_survey": False,
        "no_ift_meaning": config["calypso"]["no_ift_meaning"],
        "geom_flag_to_tags": config["calypso"]["geom_flag_to_tags"],
        "write_alignment_is_neutral": True,
    }


def empty_constraint_catalog(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    slots = []
    rigid = ("dx", "dy", "dz", "rx", "ry", "rz")
    rigid_units = ("mm", "mm", "mm", "mrad", "mrad", "mrad")
    for station, name in STATION_NAMES.items():
        slots.append(
            _base_constraint(
                constraint_id=f"station_{station}_{name}_rigid_6dof",
                kind="station_rigid_transform",
                detector_hierarchy={
                    "system": "Tracker",
                    "station_id": station,
                    "station_name": name,
                    "cool_channel": "/Tracker/Align/Stations",
                    "layer_id": None,
                },
                frame="global",
                reference_object="FASER global / GeoModel world",
                parameter_names=list(rigid),
                units=list(rigid_units),
                mapping_to_scan_parameters={"ry": "station_ry", "dx": "station_dx"} if station == IFT_STATION_ID else {"dx": "station_dx"},
                note="As-built station pose. Empty until a survey file is ingested. Nominal z is not this slot.",
            )
        )
    slots.append(
        _base_constraint(
            constraint_id="ift_station0_ry",
            kind="station_rigid_transform",
            detector_hierarchy={
                "system": "Tracker",
                "station_id": IFT_STATION_ID,
                "station_name": STATION_NAMES[IFT_STATION_ID],
                "cool_channel": "/Tracker/Align/Stations",
                "dof": "ry",
            },
            frame="global",
            reference_object="FASER global y-axis rotation of IFT/station 0",
            parameter_names=["ry"],
            units=["mrad"],
            mapping_to_scan_parameters={"ry": "station_ry", "scan_column": 1},
            note="Priority request: independent IFT station orientation.",
        )
    )
    slots.append(
        _base_constraint(
            constraint_id="ift_l0_minus_l2_dx",
            kind="layer_relative_transform",
            detector_hierarchy={
                "system": "Tracker",
                "station_id": IFT_STATION_ID,
                "layer_ids": [0, 2],
                "cool_channel": "/Tracker/Align/Planes",
            },
            frame="global",
            reference_object="IFT cassette; layer 0 minus layer 2 transverse x",
            parameter_names=["dx_L0_minus_dx_L2"],
            units=["mm"],
            mapping_to_scan_parameters={
                "C_dx": "(dx_L0-dx_L2)/2",
                "sigma_C_dx": "sigma(dx_L0-dx_L2)/2",
                "scan_column": 2,
            },
            note="Priority request: L0/L2 relative transverse displacement maps to C_dx.",
        )
    )
    slots.append(
        _base_constraint(
            constraint_id="ift_C_dx",
            kind="C_dx",
            detector_hierarchy={
                "system": "Tracker",
                "station_id": IFT_STATION_ID,
                "layers": list(IFT_LAYER_IDS),
                "definition": "C_dx=(dx_L0-dx_L2)/2",
            },
            frame="IFT plane local dx, zero common mode",
            reference_object="IFT cassette outer-contrast gauge",
            parameter_names=["C_dx"],
            units=["mm"],
            mapping_to_scan_parameters={"C_dx": "C_dx", "scan_column": 2},
            note="Direct C_dx metrology if the hardware group quotes the contrast, not LAYERPITCH.",
        )
    )
    slots.append(
        _base_constraint(
            constraint_id="general_linear_C_theta_equals_b",
            kind="linear_equality",
            detector_hierarchy={"system": "Tracker", "station_id": None},
            frame="scan parameter basis (dx mm, ry mrad, C_dx mm)",
            reference_object="user-supplied linear combination of alignment parameters",
            parameter_names=list(PARAMETER_NAMES),
            units=list(SCAN_PARAMETER_UNITS),
            mapping_to_scan_parameters={"C": "3-vector row", "b": "scalar"},
            note="Generic Cθ=b slot. Empty until a real constraint matrix arrives.",
        )
    )
    return slots


def synthetic_feasibility_constraint(
    *,
    constraint_id: str,
    kind: str,
    parameter_name: str,
    sigma: float,
    unit: str,
    scan_column: int,
    frozen_requirement: str,
) -> dict[str, Any]:
    constraint = _base_constraint(
        constraint_id=constraint_id,
        kind=kind,
        detector_hierarchy={"system": "Tracker", "station_id": IFT_STATION_ID, "synthetic": True},
        frame="scan parameter basis",
        reference_object="entry-61 frozen prior-strength unit test (not a survey)",
        parameter_names=[parameter_name],
        units=[unit],
        mapping_to_scan_parameters={"scan_column": scan_column, "sigma": sigma},
        note=frozen_requirement,
    )
    constraint.update(
        {
            "value": [0.0],
            "sigma": [float(sigma)],
            "covariance": [[float(sigma) ** 2]],
            "availability": "feasibility_only",
            "source_file": "synthetic_entry61_prior_strength",
            "provenance": "feasibility_only_not_survey",
            "independent_of_track_residual": True,
            "may_enter_geometry_candidate": False,
            "central_value_is_survey": False,
        }
    )
    validate_constraint(constraint)
    assert_not_geometry_candidate(constraint)
    return constraint


def constraint_prior_row(constraint: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return a Gaussian prior row in scan units, or None if unavailable."""
    validate_constraint(constraint)
    if constraint.get("availability") == "unavailable":
        return None
    sigma = constraint.get("sigma")
    if sigma is None:
        return None
    sigma_value = float(sigma[0] if isinstance(sigma, Sequence) else sigma)
    if not math.isfinite(sigma_value) or sigma_value <= 0.0:
        return None
    mapping = constraint.get("mapping_to_scan_parameters") or {}
    column = mapping.get("scan_column")
    kind = constraint.get("kind")
    row = [0.0, 0.0, 0.0]
    if column is not None:
        row[int(column)] = 1.0 / sigma_value
        effective_sigma = {"column": int(column), "sigma": sigma_value, "unit": constraint["units"][0]}
    elif kind == "layer_relative_transform" and "C_dx" in mapping:
        row[2] = 2.0 / sigma_value
        effective_sigma = {"column": 2, "sigma_C_dx_mm": sigma_value / 2.0, "from": "sigma(dx_L0-dx_L2)/2"}
    elif kind == "linear_equality":
        matrix = mapping.get("C")
        if matrix is None:
            return None
        row = [float(item) / sigma_value for item in matrix]
        effective_sigma = {"C": list(matrix), "sigma": sigma_value}
    else:
        return None
    return {
        "constraint_id": constraint["constraint_id"],
        "availability": constraint["availability"],
        "row": row,
        "effective_sigma": effective_sigma,
        "may_enter_geometry_candidate": False,
    }


def covariance_matched_collinear_jacobian(config: Mapping[str, Any]) -> np.ndarray:
    """Analysis-only Jacobian with the frozen entry-61 collinear covariance.

    Does not read 2024 r0022 tracks.  Column 1 is stored in rad so
    jacobian_in_scan_units converts it to mrad.
    """
    frozen = config["entry_61_frozen"]
    physics = config["physics_scales"]
    n = int(frozen["collinear_track_only_n_rows"])
    sigma_r = float(physics["measurement_sigma_mm"])
    sigma_ry = float(frozen["collinear_track_only_marginal_sigma_ry_mrad"])
    sigma_cdx = float(frozen["collinear_track_only_marginal_sigma_cdx_mm"])
    rho = float(frozen["collinear_track_only_ry_cdx_correlation"])
    covariance = np.asarray(
        [
            [sigma_ry ** 2, rho * sigma_ry * sigma_cdx],
            [rho * sigma_ry * sigma_cdx, sigma_cdx ** 2],
        ],
        dtype=np.float64,
    )
    factor = np.linalg.cholesky(np.linalg.inv(covariance))
    rng = np.random.default_rng(14973)
    q_matrix, _ = np.linalg.qr(rng.normal(size=(n, 2)))
    block = q_matrix @ factor.T * sigma_r
    j_dx = rng.normal(size=(n,)).astype(np.float64)
    j_ry_rad = block[:, 0] / 0.001
    j_cdx = block[:, 1]
    return np.column_stack([j_dx, j_ry_rad, j_cdx])


def track_fisher(jacobian: np.ndarray, *, sigma_r_mm: float) -> np.ndarray:
    scan = jacobian_in_scan_units(jacobian) / float(sigma_r_mm)
    return fisher_from_augmented(scan)


def combine_track_and_external_fisher(
    jacobian: np.ndarray,
    constraints: Sequence[Mapping[str, Any]],
    *,
    sigma_r_mm: float,
    rank_tolerance: float,
) -> dict[str, Any]:
    """F_tot = J_track^T W J_track + Σ C_i^T Σ_i^{-1} C_i.  No geometry write."""
    used = []
    rows = [jacobian_in_scan_units(jacobian) / float(sigma_r_mm)]
    for constraint in constraints:
        prior = constraint_prior_row(constraint)
        if prior is None:
            continue
        used.append(prior)
        rows.append(np.asarray([prior["row"]], dtype=np.float64))
    augmented = np.vstack(rows)
    fisher = fisher_from_augmented(augmented)
    spectrum = column_normalized_spectrum(augmented, rank_tolerance=float(rank_tolerance))
    rho = correlation_ry_cdx(fisher)
    try:
        covariance = np.linalg.inv(fisher)
        finite = True
    except np.linalg.LinAlgError:
        covariance = np.linalg.pinv(fisher)
        finite = False
    metrics = {
        "rank": spectrum["rank"],
        "sigma3_over_sigma1": spectrum["sigma3_over_sigma1"],
        "ry_cdx_correlation": rho,
        "singular_values": spectrum["singular_values"],
        "finite_covariance": finite,
        "marginal_sigma_dx_mm": math.sqrt(abs(float(covariance[0, 0]))) if finite else None,
        "marginal_sigma_ry_mrad": math.sqrt(abs(float(covariance[1, 1]))) if finite else None,
        "marginal_sigma_cdx_mm": math.sqrt(abs(float(covariance[2, 2]))) if finite else None,
        "n_track_rows": int(np.asarray(jacobian).shape[0]),
        "n_external_rows": int(len(used)),
        "external_constraints_used": [row["constraint_id"] for row in used],
        "synthetic_only": bool(used) and all(
            constraint.get("availability") == "feasibility_only"
            for constraint in constraints
            if constraint_prior_row(constraint) is not None
        ),
        "geometry_candidate": False,
        "emits_alignment_payload": False,
    }
    return metrics


def summarize_grl(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        return {"path": str(source), "exists": False, "n_runs": None, "first_run": None, "last_run": None, "n_fills": None}
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or not payload:
        return {"path": str(source), "exists": True, "n_runs": 0, "first_run": None, "last_run": None, "n_fills": 0}
    runs = sorted(int(key) for key in payload)
    fills: set[int] = set()
    for row in payload.values():
        if not isinstance(row, Mapping):
            continue
        for interval in row.get("stable_list") or []:
            fill = (interval or {}).get("lhc_fill")
            if fill is not None:
                fills.add(int(fill))
    return {
        "path": str(source),
        "exists": True,
        "n_runs": len(runs),
        "first_run": runs[0],
        "last_run": runs[-1],
        "n_fills": len(fills),
        "fill_range": [min(fills), max(fills)] if fills else None,
        "do_not_average_across_this_and_other_iovs": True,
    }


def alignment_parameter_blocks(ift: bool) -> dict[str, Any]:
    static = [
        {"block": "strip_pitch", "status": "immutable_static_geometry", "source": "sensor manufacturing"},
        {"block": "stereo_angle", "status": "immutable_static_geometry", "source": "SCTBRLMODULE STEREOANGLE, not station ry"},
        {"block": "ift_layer_pitch_and_cassette_drawing", "status": "immutable_static_geometry", "source": "LAYERPITCH design"},
        {"block": "nominal_station_z", "status": "immutable_static_geometry", "source": "FASERNU-04 reconstruction z, not survey"},
    ]
    iov_specific = [
        {"block": "station_dx", "status": "iov_specific", "float_from_tracks": True},
        {"block": "station_dy", "status": "iov_specific", "float_from_tracks": True},
        {"block": "station_rx", "status": "iov_specific", "float_from_tracks": True},
        {"block": "station_ry", "status": "iov_specific", "float_from_tracks": False, "needs_external": True,
         "reason": "Degenerate with C_dx on collision-like tracks; wait for survey or keep IOV-specific with external prior."},
        {"block": "station_rz", "status": "iov_specific", "float_from_tracks": True},
    ]
    gauge = [
        {"block": "station_dz", "status": "survey_or_gauge_constrained", "float_from_tracks": False,
         "reason": "Track Jacobian dz is gauge-like. Do not hand dz back to tracks. Software 5 mm is not metrology."}
    ]
    c_dx = {
        "block": "C_dx",
        "status": "common_static_candidate_until_opening_or_thermal_evidence",
        "float_from_tracks": False,
        "external_metrology_may_override": True,
        "present": bool(ift),
        "reason": "Cassette-internal. Keep common-static until evidence; a real L0/L2 metrology constraint overrides.",
    }
    if not ift:
        c_dx = {
            "block": "C_dx",
            "status": "not_a_parameter",
            "present": False,
            "reason": "No IFT cassette in this IOV (2022 r0021 / TestBeam). Do not copy 2024 C_dx.",
        }
    return {
        "form": "theta^(y) = theta_static + Delta_theta_IOV^(y)",
        "immutable_static": static,
        "iov_specific": iov_specific,
        "survey_or_gauge_constrained": gauge,
        "C_dx": c_dx,
        "do_not_free_fill_or_run_initially": True,
    }


def iov_records(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    grl = {int(year): summarize_grl(path) for year, path in (config.get("grl") or {}).items()}
    pool = Path(str(config["official_alignment_pool_dir"]))
    records = [
        {
            "iov_id": "2022_r0021_OFLCOND-FASER-04",
            "year": 2022,
            "rec_tag": "r0021",
            "geometry_tag": "FASERNU-04",
            "geom_flag": "TI12Data04",
            "conditions_tag": "OFLCOND-FASER-04",
            "alignment_pool": "FASER-04_2022_Align.pool.root",
            "ift_in_ckf": False,
            "c_dx_parameter": False,
            "track_topology": "collision_like",
            "note": "--noIFT three-station CKF. IFT clusters may exist; C_dx is not a parameter of this IOV.",
        },
        {
            "iov_id": "2022_r0022_OFLCOND-FASER-05",
            "year": 2022,
            "rec_tag": "r0022",
            "geometry_tag": "FASERNU-04",
            "geom_flag": "TI12Data04",
            "conditions_tag": "OFLCOND-FASER-05",
            "alignment_pool": "FASER-04_2022_Align.pool.root",
            "ift_in_ckf": True,
            "c_dx_parameter": True,
            "track_topology": "collision_like",
            "note": "IFT-era 4-station CKF. Different conditions tag from 2022 r0021; do not merge.",
        },
        {
            "iov_id": "2023_r0021_OFLCOND-FASER-04",
            "year": 2023,
            "rec_tag": "r0021",
            "geometry_tag": "FASERNU-04",
            "geom_flag": "TI12Data04",
            "conditions_tag": "OFLCOND-FASER-04",
            "alignment_pool": "FASER-05_2023_Align.pool.root",
            "ift_in_ckf": False,
            "c_dx_parameter": False,
            "track_topology": "collision_like",
        },
        {
            "iov_id": "2023_cos_FASERNU-03_OFLCOND-FASER-04",
            "year": 2023,
            "rec_tag": "r0019",
            "geometry_tag": "FASERNU-03",
            "geom_flag": "TI12Data03",
            "conditions_tag": "OFLCOND-FASER-04",
            "alignment_pool": "FASER-05_2023_Align.pool.root",
            "ift_in_ckf": False,
            "c_dx_parameter": False,
            "track_topology": "named_cosmic_stream",
            "xaod_retained": False,
            "note": "Different geometry tag from 2023 collision FASERNU-04. xAOD not retained.",
        },
        {
            "iov_id": "2023_ift_OFLCOND-FASER-05",
            "year": 2023,
            "rec_tag": "phys_2023_ift",
            "geometry_tag": "FASERNU-04",
            "geom_flag": "TI12Data04",
            "conditions_tag": "OFLCOND-FASER-05",
            "alignment_pool": "FASER-06_2023_Align.pool.root",
            "ift_in_ckf": True,
            "c_dx_parameter": True,
            "track_topology": "collision_like",
            "note": "Still collision-like topology. Do not mix residuals with 2024.",
        },
        {
            "iov_id": "2024_r0022_production_OFLCOND-FASER-05",
            "year": 2024,
            "rec_tag": "r0022",
            "geometry_tag": "FASERNU-04",
            "geom_flag": "TI12Data04",
            "conditions_tag": "OFLCOND-FASER-05",
            "alignment_pool": "FASER-05_2024_Align.pool.root",
            "ift_in_ckf": False,
            "c_dx_parameter": True,
            "track_topology": "collision_like_frozen_v2_windows",
            "frozen_v2_runs": [14973, 14974],
            "note": "Production reconstruction tag. Distinct from the current analysis protocol -06.",
        },
        {
            "iov_id": "2024_protocol_OFLCOND-FASER-06",
            "year": 2024,
            "rec_tag": "analysis_protocol",
            "geometry_tag": "FASERNU-04",
            "geom_flag": "TI12Data04",
            "conditions_tag": "OFLCOND-FASER-06",
            "alignment_pool": "FASER-06_2024_Align.pool.root",
            "ift_in_ckf": False,
            "c_dx_parameter": True,
            "track_topology": "same_collision_like_sample_different_conditions_provenance",
            "note": "faser_reco TI12Data04 currently sets OFLCOND-FASER-06. Not the same IOV as production -05. Do not average residuals or corrections with -05.",
        },
        {
            "iov_id": "2025_r0023_production_OFLCOND-FASER-05",
            "year": 2025,
            "rec_tag": "r0023",
            "geometry_tag": "FASERNU-04",
            "geom_flag": "TI12Data04",
            "conditions_tag": "OFLCOND-FASER-05",
            "alignment_pool": "FASER-06_2025_Align.pool.root",
            "ift_in_ckf": False,
            "c_dx_parameter": True,
            "track_topology": "collision_like_few_r0023_runs",
            "note": "Production rec used -05 while the year-tagged pool file is FASER-06_2025. Treat rec conditions and pool filename as provenance, not as a shared constant set with 2024.",
        },
    ]
    enriched = []
    for row in records:
        pool_file = pool / str(row["alignment_pool"])
        year = int(row["year"])
        enriched.append(
            {
                **row,
                "alignment_pool_exists": pool_file.is_file(),
                "alignment_pool_interpreted_as": "neutral_WriteAlignment_payload_not_survey",
                "run_fill_from_year_grl": grl.get(year),
                "parameter_blocks": alignment_parameter_blocks(bool(row["c_dx_parameter"])),
                "do_not_merge_residuals_or_corrections_with_other_iovs": True,
            }
        )
    return enriched


def conditions_provenance_matrix(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    pairs = []
    for index, left in enumerate(records):
        for right in records[index + 1 :]:
            same_year = left["year"] == right["year"]
            same_geo = left["geometry_tag"] == right["geometry_tag"]
            same_cond = left["conditions_tag"] == right["conditions_tag"]
            pairs.append(
                {
                    "left": left["iov_id"],
                    "right": right["iov_id"],
                    "same_year": same_year,
                    "same_geometry_tag": same_geo,
                    "same_conditions_tag": same_cond,
                    "may_average_residuals": False,
                    "may_average_corrections": False,
                    "may_share_alignment_constants": False,
                    "reason": (
                        "Distinct IOV. Even 2024 OFLCOND-FASER-05 vs -06 is a conditions-provenance split."
                        if left["year"] == 2024 and right["year"] == 2024 and not same_cond
                        else "Cross-IOV residual/correction average is forbidden."
                    ),
                }
            )
    production_vs_protocol = next(
        pair
        for pair in pairs
        if pair["left"] == "2024_r0022_production_OFLCOND-FASER-05"
        and pair["right"] == "2024_protocol_OFLCOND-FASER-06"
    )
    return {
        **common_audit_state(),
        "n_iovs": len(records),
        "iov_ids": [row["iov_id"] for row in records],
        "pairs": pairs,
        "production_2024_05_versus_protocol_2024_06": production_vs_protocol,
        "do_not_average_across_iovs": True,
    }


def metrology_requirement_table(config: Mapping[str, Any]) -> dict[str, Any]:
    frozen = config["entry_61_frozen"]
    useful = config["useful_precision"]
    stretch = config["stretch_precision"]
    rows = [
        {
            "priority": 1,
            "request": "Independent IFT / station-0 orientation about global y (station ry)",
            "what_to_measure": "Angle of the IFT cassette / station 0 relative to the FASER global frame, not the module stereo angle.",
            "frame": "FASER global; maps to /Tracker/Align/Stations ry",
            "degeneracy_breaking": {
                "sigma": float(frozen["degeneracy_breaking_sigma_ry_mrad"]),
                "unit": "mrad",
                "label": "degeneracy_breaking_only",
                "not_the_final_measurement_target": True,
            },
            "physically_useful": {
                "sigma": float(useful["ry_mrad"]),
                "unit": "mrad",
                "label": "physically_useful_alignment_precision",
                "rationale": useful["ry_rationale"],
            },
            "stretch": {
                "sigma": float(stretch["ry_mrad"]),
                "unit": "mrad",
                "rationale": stretch["ry_rationale"],
            },
            "independent_of_track_residual": True,
        },
        {
            "priority": 1,
            "request": "IFT layer-0 vs layer-2 relative transverse displacement",
            "what_to_measure": "dx(L0) − dx(L2) on the cassette. C_dx = (dx_L0 − dx_L2)/2.",
            "frame": "IFT planes; maps to /Tracker/Align/Planes and C_dx",
            "degeneracy_breaking": {
                "sigma_C_dx_mm": float(frozen["degeneracy_breaking_sigma_cdx_mm"]),
                "sigma_dx_L0_minus_L2_mm": 2.0 * float(frozen["degeneracy_breaking_sigma_cdx_mm"]),
                "label": "degeneracy_breaking_only",
                "not_the_final_measurement_target": True,
            },
            "physically_useful": {
                "sigma_C_dx_mm": float(useful["cdx_mm"]),
                "sigma_dx_L0_minus_L2_mm": float(useful["l0_minus_l2_dx_mm"]),
                "label": "physically_useful_alignment_precision",
                "rationale": useful["cdx_rationale"],
            },
            "stretch": {
                "sigma_C_dx_mm": float(stretch["cdx_mm"]),
                "sigma_dx_L0_minus_L2_mm": float(stretch["l0_minus_l2_dx_mm"]),
                "rationale": stretch["cdx_rationale"],
            },
            "independent_of_track_residual": True,
        },
    ]
    return {
        **common_audit_state(),
        "audience": "FASER hardware / alignment / metrology",
        "do_not_quote_20_mrad_or_0_315_mm_as_the_final_target": True,
        "why_two_tiers": (
            "On 2024 r0022 collinear tracks, σ(ry)≲20 mrad or σ(C_dx)≲0.315 mm restores "
            "rank-3 and |ρ|≤0.90. After a tight ry prior the leftover σ(C_dx) saturates "
            f"at {frozen['leftover_sigma_cdx_mm_after_tight_ry']:.3f} mm, coarser than "
            "strip pitch 0.080 mm. Useful C_dx therefore requires direct L0/L2 metrology, "
            "not a coarse ry survey plus these tracks."
        ),
        "not_requested_as_measurements": list(FORBIDDEN_AS_MEASUREMENT),
        "requests": rows,
        "either_priority_one_constraint_is_sufficient_to_start": True,
        "both_needed_for_useful_precision_on_current_tracks": True,
    }


def fisher_validation(config: Mapping[str, Any]) -> dict[str, Any]:
    frozen = config["entry_61_frozen"]
    physics = config["physics_scales"]
    rules = config["unlock"]
    sigma_r = float(physics["measurement_sigma_mm"])
    toy = collinear_toy_jacobian(alpha_mm_per_rad=float(physics["ift_layer_pitch_mm"]))
    leaky = covariance_matched_collinear_jacobian(config)
    ry_break = synthetic_feasibility_constraint(
        constraint_id="synthetic_ry_degeneracy_break",
        kind="station_rigid_transform",
        parameter_name="ry",
        sigma=float(frozen["degeneracy_breaking_sigma_ry_mrad"]),
        unit="mrad",
        scan_column=1,
        frozen_requirement="entry-61 degeneracy-breaking σ(ry)=20 mrad, feasibility_only",
    )
    cdx_break = synthetic_feasibility_constraint(
        constraint_id="synthetic_cdx_degeneracy_break",
        kind="C_dx",
        parameter_name="C_dx",
        sigma=float(frozen["degeneracy_breaking_sigma_cdx_mm"]),
        unit="mm",
        scan_column=2,
        frozen_requirement="entry-61 degeneracy-breaking σ(C_dx)=0.315 mm, feasibility_only",
    )
    ry_useful = synthetic_feasibility_constraint(
        constraint_id="synthetic_ry_useful",
        kind="station_rigid_transform",
        parameter_name="ry",
        sigma=float(config["useful_precision"]["ry_mrad"]),
        unit="mrad",
        scan_column=1,
        frozen_requirement="useful-precision σ(ry)=0.5 mrad, feasibility_only, not a survey",
    )
    cases = {
        "perfect_toy_track_only": combine_track_and_external_fisher(toy, [], sigma_r_mm=sigma_r, rank_tolerance=rules["rank_tolerance"]),
        "perfect_toy_plus_20mrad_ry": combine_track_and_external_fisher(toy, [ry_break], sigma_r_mm=sigma_r, rank_tolerance=rules["rank_tolerance"]),
        "perfect_toy_plus_0p5mrad_ry": combine_track_and_external_fisher(toy, [ry_useful], sigma_r_mm=sigma_r, rank_tolerance=rules["rank_tolerance"]),
        "leaky_toy_track_only": combine_track_and_external_fisher(leaky, [], sigma_r_mm=sigma_r, rank_tolerance=rules["rank_tolerance"]),
        "leaky_toy_plus_20mrad_ry": combine_track_and_external_fisher(leaky, [ry_break], sigma_r_mm=sigma_r, rank_tolerance=rules["rank_tolerance"]),
        "leaky_toy_plus_0p315mm_cdx": combine_track_and_external_fisher(leaky, [cdx_break], sigma_r_mm=sigma_r, rank_tolerance=rules["rank_tolerance"]),
    }
    for name, metrics in cases.items():
        metrics["unlocked"] = unlocked(metrics, rules)
        metrics["case"] = name
    reference = combined_identifiability(
        leaky,
        sigma_r_mm=sigma_r,
        sigma_ry_mrad=float(frozen["degeneracy_breaking_sigma_ry_mrad"]),
        rank_tolerance=float(rules["rank_tolerance"]),
    )
    return {
        **common_audit_state(),
        "did_rebuild_2024_r0022_jacobian": False,
        "entry_61_frozen_requirement": {
            "sigma_ry_mrad": frozen["degeneracy_breaking_sigma_ry_mrad"],
            "sigma_cdx_mm": frozen["degeneracy_breaking_sigma_cdx_mm"],
            "not_the_final_measurement_target": True,
        },
        "synthetic_constraints": [ry_break, cdx_break, ry_useful],
        "cases": cases,
        "combiner_matches_entry61_identifiability_on_leaky_ry": bool(
            abs(float(cases["leaky_toy_plus_20mrad_ry"]["ry_cdx_correlation"]) - float(reference["ry_cdx_correlation"])) < 1.0e-9
            and int(cases["leaky_toy_plus_20mrad_ry"]["rank"]) == int(reference["rank"])
        ),
        "perfect_toy_20mrad_does_not_unlock": not bool(cases["perfect_toy_plus_20mrad_ry"]["unlocked"]),
        "leaky_toy_20mrad_unlocks": bool(cases["leaky_toy_plus_20mrad_ry"]["unlocked"]),
        "leaky_toy_0p315mm_cdx_unlocks": bool(cases["leaky_toy_plus_0p315mm_cdx"]["unlocked"]),
        "synthetic_may_enter_geometry_candidate": False,
    }


def decide_next_stage(
    *,
    catalog: Sequence[Mapping[str, Any]],
    validation: Mapping[str, Any],
    matrix: Mapping[str, Any],
) -> dict[str, Any]:
    n_measured = sum(1 for row in catalog if row.get("availability") == "measured")
    return {
        **common_audit_state(),
        "decision": DECISION_INGESTION_READY,
        "n_measured_survey_constraints": n_measured,
        "as_built_survey_available": False,
        "go_to_full_module_identifiability_map": False,
        "still_no_new_network": True,
        "geometry_write_allowed": False,
        "emits_alignment_payload": False,
        "survey_numbers_were_invented": False,
        "synthetic_used_as_geometry_candidate": False,
        "did_rebuild_2024_r0022_jacobian": False,
        "combiner_ready": bool(validation.get("leaky_toy_20mrad_unlocks")) and bool(validation.get("leaky_toy_0p315mm_cdx_unlocks")),
        "conditions_05_and_06_are_distinct_iovs": bool(
            (matrix.get("production_2024_05_versus_protocol_2024_06") or {}).get("may_average_corrections") is False
        ),
        "next_allowed_step": "ingest_real_independent_survey_into_matching_iov_and_test_ry_cdx_separation",
        "reason": (
            "The schema, IOV state, and track+prior Fisher combiner are ready. "
            "No as-built survey is present. A feasibility-only synthetic prior at the "
            "frozen entry-61 degeneracy-breaking strength restores rank-3 on a leaky "
            "collinear toy and is forbidden from geometry candidates. When a real "
            "independent ry or L0/L2 C_dx measurement with uncertainty arrives, ingest "
            "it into the matching year/conditions IOV and test identifiability without "
            "changing ML, track selection, or parameterization."
        ),
    }
