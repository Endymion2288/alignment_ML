"""External-constraint and year/IOV alignment-framework feasibility.

Inventory of independent mechanical/metrology information, prior-strength
Fisher scan on the frozen {dx, ry, C_dx} Jacobian, and an IOV-aware
parameterization.  Does not train, write geometry, invent survey numbers,
select a prior from residuals, or emit an alignment payload.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from alignment.operating_protocol_v1_final_closure import (
    FROZEN_V2_CHECKPOINT_SHA256,
    OPERATING_MODE,
    RESIDUAL_DECREASE_LABEL,
    project_root,
)
from alignment.true_cluster_local_residual import (
    LEAKAGE_RANK_RELATIVE_TOLERANCE,
    RESIDUAL_KIND,
    common_operating_state,
)
from alignment.true_cluster_local_stability_transfer import summarize_jacobian

SCHEMA_VERSION = "faser-external-constraint-iov-alignment-feasibility-v1"
DEFAULT_CONFIG_RELATIVE = Path("configs") / "external_constraint_iov_alignment_feasibility_v1.yaml"
DECISION_SURVEY_AND_IOV = "external_survey_or_metrology_required_and_year_iov_parameterization_required"
PARAMETER_NAMES = ("station_dx", "station_ry", "C_dx")
SCAN_PARAMETER_UNITS = ("station_dx_mm", "station_ry_mrad", "C_dx_mm")
RAD_TO_MRAD = 0.001
SURVEY_NAME_RE = re.compile(
    r"survey|metrology|as[-_]?built|installation|theodolite|laser.?tracker",
    re.IGNORECASE,
)
ALIGN_POOL_RE = re.compile(r"FASER-.*Align\.pool\.root$")
GO_NO_GO_QUESTIONS = (
    "What type and precision of independent mechanical/metrology constraint is needed to break ry↔C_dx?",
    "Given those constraints, should 2022/2023/2024/2025 use common static geometry plus year/IOV-specific corrections rather than one shared constant set?",
)
CONFIG_MUST_BE_FALSE = (
    "geometry_write_allowed",
    "station_calibration_mode_available",
    "cdx_mode_allowed",
    "alignment_payload_from_self_nulling_residuals",
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
    "do_not_mix_cross_year_residuals_or_alignment_constants",
    "do_not_invent_survey_numbers",
    "software_fd_sensitivity_only",
)


def load_framework_config(path: str | Path | None = None) -> dict[str, Any]:
    source = Path(path).expanduser().resolve() if path is not None else (
        project_root() / DEFAULT_CONFIG_RELATIVE
    )
    with source.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"external-constraint config must be a mapping: {source}")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unexpected external-constraint schema: {source}")
    for key in CONFIG_MUST_BE_TRUE:
        if payload.get(key) is not True:
            raise ValueError(f"external-constraint config must set {key}=true")
    for key in CONFIG_MUST_BE_FALSE:
        if payload.get(key) is not False:
            raise ValueError(f"external-constraint config must set {key}=false")
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
    if physics.get("measurement_sigma_source") != "strip_pitch_not_residual_rms":
        raise ValueError("do not select the measurement sigma from residuals")
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
            "do_not_mix_cross_year_residuals_or_alignment_constants": True,
            "schema_version": SCHEMA_VERSION,
            "go_no_go_questions": list(GO_NO_GO_QUESTIONS),
            "emits_alignment_payload": False,
        }
    )
    return state


def collinear_toy_jacobian(
    *,
    n: int = 80,
    alpha_mm_per_rad: float = 31.5,
    dx_scale: float = 1.0,
    seed: int = 14973,
) -> np.ndarray:
    """Rank-2 toy: independent dx, perfectly collinear ry and C_dx.

    Columns are the same units as the real FD Jacobian: (mm, rad, mm).
    Design map C_dx ≈ ry * LAYERPITCH so J_ry = LAYERPITCH_mm_per_rad * J_Cdx.
    """
    rng = np.random.default_rng(int(seed))
    base = rng.normal(size=(int(n),)).astype(np.float64)
    j_dx = dx_scale * rng.normal(size=(int(n),)).astype(np.float64)
    j_cdx = base
    j_ry = float(alpha_mm_per_rad) * base
    return np.column_stack([j_dx, j_ry, j_cdx])


def jacobian_in_scan_units(jacobian: np.ndarray) -> np.ndarray:
    """Convert FD columns (mm, rad, mm) to scan units (mm, mrad, mm)."""
    values = np.asarray(jacobian, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("jacobian must be n by 3")
    scaled = values.copy()
    scaled[:, 1] *= RAD_TO_MRAD
    return scaled


def column_normalized(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    norms = np.linalg.norm(values, axis=0)
    norms = np.where(norms > 0.0, norms, 1.0)
    return values / norms


def column_normalized_spectrum(matrix: np.ndarray, *, rank_tolerance: float) -> dict[str, Any]:
    normalized = column_normalized(matrix)
    singular = np.linalg.svd(normalized, compute_uv=False)
    if singular.size == 0 or float(singular[0]) <= 0.0:
        return {
            "rank": 0,
            "singular_values": [],
            "sigma3_over_sigma1": None,
        }
    rank = int(np.sum(singular > float(rank_tolerance) * float(singular[0])))
    ratio = float(singular[2] / singular[0]) if singular.size > 2 else None
    return {
        "rank": rank,
        "singular_values": [float(value) for value in singular],
        "sigma3_over_sigma1": ratio,
    }


def augment_with_priors(
    jacobian_scan_units: np.ndarray,
    *,
    sigma_r_mm: float,
    sigma_dx_mm: float | None = None,
    sigma_ry_mrad: float | None = None,
    sigma_cdx_mm: float | None = None,
) -> np.ndarray:
    """Stack Gaussian prior rows onto a scan-unit Jacobian (mm, mrad, mm)."""
    values = np.asarray(jacobian_scan_units, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("jacobian must be n by 3")
    rows = [values / float(sigma_r_mm)]
    if sigma_dx_mm is not None and math.isfinite(float(sigma_dx_mm)) and float(sigma_dx_mm) > 0.0:
        rows.append(np.asarray([[1.0 / float(sigma_dx_mm), 0.0, 0.0]], dtype=np.float64))
    if sigma_ry_mrad is not None and math.isfinite(float(sigma_ry_mrad)) and float(sigma_ry_mrad) > 0.0:
        rows.append(np.asarray([[0.0, 1.0 / float(sigma_ry_mrad), 0.0]], dtype=np.float64))
    if sigma_cdx_mm is not None and math.isfinite(float(sigma_cdx_mm)) and float(sigma_cdx_mm) > 0.0:
        rows.append(np.asarray([[0.0, 0.0, 1.0 / float(sigma_cdx_mm)]], dtype=np.float64))
    return np.vstack(rows)


def fisher_from_augmented(jacobian_aug: np.ndarray) -> np.ndarray:
    values = np.asarray(jacobian_aug, dtype=np.float64)
    return values.T @ values


def correlation_ry_cdx(fisher: np.ndarray) -> float:
    matrix = np.asarray(fisher, dtype=np.float64)
    try:
        covariance = np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        covariance = np.linalg.pinv(matrix)
    denom = math.sqrt(abs(float(covariance[1, 1]) * float(covariance[2, 2])))
    if denom <= 0.0 or not math.isfinite(denom):
        return float("nan")
    return float(covariance[1, 2] / denom)


def combined_identifiability(
    jacobian: np.ndarray,
    *,
    sigma_r_mm: float,
    sigma_ry_mrad: float | None = None,
    sigma_cdx_mm: float | None = None,
    rank_tolerance: float = LEAKAGE_RANK_RELATIVE_TOLERANCE,
) -> dict[str, Any]:
    """Track Jacobian plus optional Gaussian priors in (mm, mrad, mm) scan units.

    Rank and σ3/σ1 are taken from the column-normalized augmented Jacobian so a
    strong prior cannot inflate σ1 and hide the other two directions.  Parameter
    correlation comes from the Fisher covariance, which is scale-invariant.
    """
    scan = jacobian_in_scan_units(jacobian)
    augmented = augment_with_priors(
        scan,
        sigma_r_mm=float(sigma_r_mm),
        sigma_ry_mrad=sigma_ry_mrad,
        sigma_cdx_mm=sigma_cdx_mm,
    )
    spectrum = column_normalized_spectrum(augmented, rank_tolerance=float(rank_tolerance))
    fisher = fisher_from_augmented(augmented)
    rho = correlation_ry_cdx(fisher)
    eigenvalues = np.sort(np.linalg.eigvalsh(fisher))[::-1]
    try:
        covariance = np.linalg.inv(fisher)
        finite_cov = True
    except np.linalg.LinAlgError:
        covariance = np.linalg.pinv(fisher)
        finite_cov = False
    point = summarize_jacobian(augmented)
    return {
        "n_track_rows": int(np.asarray(jacobian).shape[0]),
        "n_augmented_rows": int(augmented.shape[0]),
        "parameter_units": list(SCAN_PARAMETER_UNITS),
        "spectrum_kind": "column_normalized_scan_units",
        "sigma_ry_mrad": sigma_ry_mrad,
        "sigma_cdx_mm": sigma_cdx_mm,
        "rank": spectrum["rank"],
        "station_dx_vs_C_dx": point.get("station_dx_vs_C_dx"),
        "station_ry_vs_C_dx": point.get("station_ry_vs_C_dx"),
        "sigma3_over_sigma1": spectrum["sigma3_over_sigma1"],
        "singular_values": spectrum["singular_values"],
        "unnormalized_singular_values": point.get("singular_values"),
        "ry_cdx_correlation": rho,
        "fisher_eigenvalues": [float(value) for value in eigenvalues],
        "finite_covariance": finite_cov,
        "marginal_sigma_dx_mm": math.sqrt(abs(float(covariance[0, 0]))) if finite_cov else None,
        "marginal_sigma_ry_mrad": (
            math.sqrt(abs(float(covariance[1, 1]))) if finite_cov else None
        ),
        "marginal_sigma_cdx_mm": math.sqrt(abs(float(covariance[2, 2]))) if finite_cov else None,
        "prior_form": _prior_form(sigma_ry_mrad, sigma_cdx_mm),
    }


def _prior_form(sigma_ry_mrad: float | None, sigma_cdx_mm: float | None) -> dict[str, Any]:
    rows = []
    if sigma_ry_mrad is not None:
        rows.append(
            {
                "parameter": "station_ry",
                "form": "theta_ry = theta_ry_ext ± sigma_ry",
                "constraint_row": [0.0, 1.0, 0.0],
                "sigma_mrad": float(sigma_ry_mrad),
                "central_value": None,
                "central_value_note": "not invented; scan uses strength only",
            }
        )
    if sigma_cdx_mm is not None:
        rows.append(
            {
                "parameter": "C_dx",
                "form": "C_dx = C_dx_ext ± sigma_C_dx",
                "constraint_row": [0.0, 0.0, 1.0],
                "sigma_mm": float(sigma_cdx_mm),
                "central_value": None,
                "central_value_note": "not invented; scan uses strength only",
            }
        )
    return {"C_theta_equals_b": rows, "independent_of_track_residual": True}


def unlocked(metrics: Mapping[str, Any], rules: Mapping[str, Any]) -> bool:
    rank_ok = int(metrics.get("rank") or 0) >= 3
    rho = metrics.get("ry_cdx_correlation")
    rho_ok = rho is not None and math.isfinite(float(rho)) and abs(float(rho)) <= float(rules["max_ry_cdx_correlation"])
    ratio = metrics.get("sigma3_over_sigma1")
    ratio_ok = ratio is not None and math.isfinite(float(ratio)) and float(ratio) >= float(rules["min_sigma3_over_sigma1"])
    return bool(rank_ok and rho_ok and ratio_ok)


def prior_grid(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    scan = config["prior_scan"]
    points: list[dict[str, Any]] = [{"family": "none", "sigma_ry_mrad": None, "sigma_cdx_mm": None}]
    for sigma_ry in scan["sigma_ry_mrad"]:
        points.append({"family": "ry_only", "sigma_ry_mrad": float(sigma_ry), "sigma_cdx_mm": None})
    for sigma_c in scan["sigma_cdx_mm"]:
        points.append({"family": "cdx_only", "sigma_ry_mrad": None, "sigma_cdx_mm": float(sigma_c)})
    for sigma_ry in scan["sigma_ry_mrad"]:
        for sigma_c in scan["sigma_cdx_mm"]:
            points.append(
                {
                    "family": "both",
                    "sigma_ry_mrad": float(sigma_ry),
                    "sigma_cdx_mm": float(sigma_c),
                }
            )
    return points


def _tightest_unlocked(
    results: Sequence[Mapping[str, Any]],
    *,
    family: str,
    key: str,
) -> dict[str, Any] | None:
    unlocked_rows = [
        row
        for row in results
        if row.get("family") == family and row.get("unlocked") and row.get(key) is not None
    ]
    if not unlocked_rows:
        return None
    return min(unlocked_rows, key=lambda row: float(row[key]))


def weakest_unlocking_prior(
    results: Sequence[Mapping[str, Any]],
    *,
    family: str,
    key: str,
) -> dict[str, Any] | None:
    unlocked_rows = [
        row
        for row in results
        if row.get("family") == family and row.get("unlocked") and row.get(key) is not None
    ]
    if not unlocked_rows:
        return None
    return min(unlocked_rows, key=lambda row: -float(row[key]))


def plateau_not_overly_sensitive(
    results: Sequence[Mapping[str, Any]],
    *,
    family: str,
    key: str,
) -> bool:
    """True when unlocking is a plateau, not a single grid-edge flicker.

    Family rows are ordered from weak prior (large σ) to tight prior (small σ).
    After the first unlocking point, every tighter grid point must also unlock,
    and at least two points must unlock.
    """
    rows = [row for row in results if row.get("family") == family and row.get(key) is not None]
    if len(rows) < 2:
        return False
    ordered = sorted(rows, key=lambda row: -float(row[key]))
    flags = [bool(row.get("unlocked")) for row in ordered]
    if flags.count(True) < 2:
        return False
    first = flags.index(True)
    return all(flags[first:])


def scan_priors(
    jacobian: np.ndarray,
    config: Mapping[str, Any],
    *,
    sample_id: str,
) -> dict[str, Any]:
    physics = config["physics_scales"]
    rules = config["unlock"]
    sigma_r = float(physics["measurement_sigma_mm"])
    rows = []
    for point in prior_grid(config):
        metrics = combined_identifiability(
            jacobian,
            sigma_r_mm=sigma_r,
            sigma_ry_mrad=point["sigma_ry_mrad"],
            sigma_cdx_mm=point["sigma_cdx_mm"],
            rank_tolerance=float(rules["rank_tolerance"]),
        )
        metrics["family"] = point["family"]
        metrics["unlocked"] = unlocked(metrics, rules)
        metrics["sample_id"] = sample_id
        rows.append(metrics)
    ry_only = weakest_unlocking_prior(rows, family="ry_only", key="sigma_ry_mrad")
    cdx_only = weakest_unlocking_prior(rows, family="cdx_only", key="sigma_cdx_mm")
    factor = float(rules["sensitivity_sigma_factor"])
    strip = float(physics["strip_pitch_mm"])
    ry_tight = _tightest_unlocked(rows, family="ry_only", key="sigma_ry_mrad")
    cdx_tight = _tightest_unlocked(rows, family="cdx_only", key="sigma_cdx_mm")
    leftover_cdx = None if ry_tight is None else ry_tight.get("marginal_sigma_cdx_mm")
    leftover_ry = None if cdx_tight is None else cdx_tight.get("marginal_sigma_ry_mrad")
    return {
        "sample_id": sample_id,
        "n_track_rows": int(np.asarray(jacobian).shape[0]),
        "measurement_sigma_mm": sigma_r,
        "measurement_sigma_source": physics["measurement_sigma_source"],
        "points": rows,
        "weakest_ry_only_unlock_mrad": None if ry_only is None else ry_only["sigma_ry_mrad"],
        "weakest_cdx_only_unlock_mm": None if cdx_only is None else cdx_only["sigma_cdx_mm"],
        "ry_only_not_overly_sensitive": bool(
            ry_only is not None and plateau_not_overly_sensitive(rows, family="ry_only", key="sigma_ry_mrad")
        ),
        "cdx_only_not_overly_sensitive": bool(
            cdx_only is not None and plateau_not_overly_sensitive(rows, family="cdx_only", key="sigma_cdx_mm")
        ),
        "ry_only_leftover_sigma_cdx_mm_at_tightest": leftover_cdx,
        "cdx_only_leftover_sigma_ry_mrad_at_tightest": leftover_ry,
        "ry_only_leftover_cdx_finer_than_strip_pitch": bool(
            leftover_cdx is not None and math.isfinite(float(leftover_cdx)) and float(leftover_cdx) <= strip
        ),
        "sensitivity_sigma_factor_documented": factor,
        "useful_cdx_scale_mm": strip,
        "useful_cdx_scale_source": "strip_pitch_not_residual_rms",
    }


def _existing_dir(path: str | Path) -> Path | None:
    candidate = Path(path)
    return candidate if candidate.is_dir() else None


def _list_matching_names(directory: Path, pattern: re.Pattern[str], *, limit: int = 40) -> list[str]:
    hits: list[str] = []
    try:
        for child in sorted(directory.iterdir(), key=lambda item: item.name):
            if pattern.search(child.name):
                hits.append(str(child))
            if len(hits) >= limit:
                break
    except OSError:
        return hits
    return hits


def probe_filesystem_sources(config: Mapping[str, Any]) -> dict[str, Any]:
    """Shallow filesystem probe.  Does not invent numerical survey values."""
    pool_dir = Path(str(config["official_alignment_pool_dir"]))
    root = project_root()
    workspace_calypso = root.parent / "calypso"
    geomdb_sql = workspace_calypso / "DetectorDescription/GeoModel/FaserGeoModel/data/geomDB.sql"
    write_alignment = workspace_calypso / "Control/CalypsoExample/WriteAlignment"
    detector_factory = (
        workspace_calypso / "Tracker/TrackerDetDescr/FaserSCT_GeoModel/src/SCT_DetectorFactory.cxx"
    )
    paper_limitations = (
        root / "docs/operating_protocol_v1_paper_ready/main_claims_and_limitations.json"
    )
    candidate_dirs = [
        str(pool_dir),
        "/cvmfs/faser.cern.ch/repo/sw/database/DBRelease/current",
        "/cvmfs/faser.cern.ch/repo/sw/database/DBRelease/current/geomDB",
        "/cvmfs/faser.cern.ch/repo/sw/database/DBRelease/current/sqlite200",
        "/eos/experiment/faser/condb",
        "/eos/experiment/faser/geo",
        "/eos/experiment/faser/survey",
        "/eos/experiment/faser/alignment",
        "/eos/experiment/faser/metrology",
        "/eos/experiment/faser",
        str(workspace_calypso),
    ]
    directory_status = []
    survey_name_hits: list[str] = []
    for raw in candidate_dirs:
        path = _existing_dir(raw)
        directory_status.append({"path": raw, "exists": path is not None})
        if path is None:
            continue
        survey_name_hits.extend(_list_matching_names(path, SURVEY_NAME_RE))
    pool_files = []
    if pool_dir.is_dir():
        pool_files = sorted(path.name for path in pool_dir.iterdir() if ALIGN_POOL_RE.search(path.name))
    eos_top = _existing_dir("/eos/experiment/faser")
    eos_children = []
    if eos_top is not None:
        try:
            eos_children = sorted(child.name for child in eos_top.iterdir() if child.is_dir())
        except OSError:
            eos_children = []
    return {
        "pool_dir": str(pool_dir),
        "pool_dir_exists": pool_dir.is_dir(),
        "official_year_tagged_align_pool_files": pool_files,
        "candidate_directories": directory_status,
        "eos_experiment_faser_top_level_dirs": eos_children,
        "survey_or_metrology_filename_hits": survey_name_hits,
        "geomdb_sql_exists": geomdb_sql.is_file(),
        "geomdb_sql_path": str(geomdb_sql) if geomdb_sql.is_file() else None,
        "write_alignment_readme": str(write_alignment / "README.md") if (write_alignment / "README.md").is_file() else None,
        "sct_detector_factory": str(detector_factory) if detector_factory.is_file() else None,
        "paper_ready_limitations": str(paper_limitations) if paper_limitations.is_file() else None,
        "as_built_survey_table_found": False,
    }


def external_constraint_inventory(config: Mapping[str, Any]) -> dict[str, Any]:
    physics = config["physics_scales"]
    probe = probe_filesystem_sources(config)
    sources = [
        {
            "id": "geomdb_layerpitch",
            "source": "calypso DetectorDescription/GeoModel/FaserGeoModel/data/geomDB.sql SCTFASERGENERAL_DATA LAYERPITCH via SCT_BarrelParameters::layerPitch",
            "quantity": "IFT plane spacing 31.5 mm",
            "numerical_value": float(physics["ift_layer_pitch_mm"]),
            "unit": "mm",
            "coordinate_system": "GeoModel tracker local z",
            "reference_frame": "nominal IFT cassette design",
            "year_iov": "geometry tag (FASERNU-0X / FASER-TB00), not a data IOV",
            "precision_sigma": None,
            "independent_of_track_residual": True,
            "constrains": [],
            "is_as_built_survey": False,
            "prior_form": None,
            "path": probe.get("geomdb_sql_path"),
            "note": "Design pitch INSERT VALUES (..., 31.5, ...). It maps ry↔C_dx (C_dx ≈ ry * LAYERPITCH) but does not measure either.",
        },
        {
            "id": "geomdb_station_z",
            "source": "nominal station z used in FASERNU-04 reconstruction / topology config",
            "quantity": "station z",
            "numerical_value": {"0": -1860.15, "1": 47.4, "2": 1237.4, "3": 2427.4},
            "unit": "mm",
            "coordinate_system": "FASER global z",
            "reference_frame": "nominal GeoModel",
            "year_iov": "FASERNU-04; FASERNU-03 and FASER-TB00 differ",
            "precision_sigma": None,
            "independent_of_track_residual": True,
            "constrains": [],
            "is_as_built_survey": False,
            "prior_form": None,
            "note": "Design positions, not an as-built station ry survey.",
        },
        {
            "id": "design_stereo",
            "source": "SCT stereo design",
            "quantity": "stereo angle",
            "numerical_value": float(physics["stereo_rad"]),
            "unit": "rad",
            "coordinate_system": "module local",
            "reference_frame": "sensor design",
            "year_iov": "all geometry tags",
            "precision_sigma": None,
            "independent_of_track_residual": True,
            "constrains": [],
            "is_as_built_survey": False,
            "prior_form": None,
            "note": "Sets the u-measurement axis.  Not a station orientation survey.",
        },
        {
            "id": "tracker_align_cool_folders",
            "source": "SCT_DetectorFactory addChannel /Tracker/Align",
            "quantity": "AlignableTransform slots",
            "numerical_value": None,
            "channels": [
                "/Tracker/Align/Stations (level 3, global)",
                "/Tracker/Align/Planes (level 2, global)",
                "/Tracker/Align/Interface{1,2,3} (level 1, local)",
                "/Tracker/Align/Upstream{1,2,3} (level 1, local)",
                "/Tracker/Align/Central{1,2,3} (level 1, local)",
                "/Tracker/Align/Downstream{1,2,3} (level 1, local)",
            ],
            "coordinate_system": "Stations/Planes global; module channels local",
            "reference_frame": "Calypso AlignableTransformContainer",
            "year_iov": "conditions tag + year-tagged POOL files",
            "precision_sigma": None,
            "independent_of_track_residual": True,
            "constrains": [],
            "is_as_built_survey": False,
            "prior_form": None,
            "path": probe.get("sct_detector_factory"),
            "note": "Software interface for constants.  TrackerAlignDBTool::createDB writes a null/identity set.",
        },
        {
            "id": "official_year_tagged_align_pool",
            "source": probe["pool_dir"],
            "quantity": "TRACKER-ALIGN conditions payloads",
            "numerical_value": None,
            "files": probe["official_year_tagged_align_pool_files"],
            "coordinate_system": "/Tracker/Align",
            "reference_frame": "conditions POOL",
            "year_iov": "explicit year in filename (2022/2023/2024/2025/TB00)",
            "precision_sigma": None,
            "independent_of_track_residual": True,
            "constrains": [],
            "is_as_built_survey": False,
            "prior_form": None,
            "note": (
                "Official DBRelease already splits alignment POOL files by year. "
                "WriteAlignment README states these are generated as neutral constants, "
                "not as-built survey numbers.  Do not read them as metrology."
            ),
        },
        {
            "id": "write_alignment_neutral_constants",
            "source": "calypso Control/CalypsoExample/WriteAlignment/README.md",
            "quantity": "neutral alignment constants",
            "numerical_value": 0.0,
            "unit": "identity transform",
            "coordinate_system": "/Tracker/Align",
            "reference_frame": "WriteAlignment / TrackerAlignDBTool",
            "year_iov": "generated per geometry tag, then copied into year-tagged POOL files",
            "precision_sigma": None,
            "independent_of_track_residual": True,
            "constrains": [],
            "is_as_built_survey": False,
            "prior_form": None,
            "path": probe.get("write_alignment_readme"),
            "note": "README: 'generates a set of neutral alignment constants'. WriteAlignmentConfig default AlignmentConstants = {}.",
        },
        {
            "id": "hierarchical_cdx_definition",
            "source": "alignment.layer_hierarchy C_dx = (dx_L0 - dx_L2)/2",
            "quantity": "IFT internal outer-contrast dx",
            "numerical_value": None,
            "coordinate_system": "IFT plane local dx, zero common mode",
            "reference_frame": "station 0 layer contrast",
            "year_iov": "parameter definition, not an IOV",
            "precision_sigma": None,
            "independent_of_track_residual": True,
            "constrains": ["C_dx"],
            "is_as_built_survey": False,
            "prior_form": "gauge choice, not C θ = b",
            "note": "Parameter definition / gauge choice, not a measurement of C_dx.",
        },
        {
            "id": "software_dz_gauge_prior",
            "source": "operating-protocol Station Mode 5-DoF + dz=5 mm survey prior, written dz=0",
            "quantity": "station dz gauge",
            "numerical_value": 0.0,
            "sigma": 5.0,
            "unit": "mm",
            "coordinate_system": "FASER global z",
            "reference_frame": "software gauge, not mechanical survey",
            "year_iov": "analysis convention",
            "precision_sigma": 5.0,
            "independent_of_track_residual": True,
            "constrains": ["station_dz"],
            "is_as_built_survey": False,
            "prior_form": "theta_dz = 0 ± 5 mm (software; do not treat as metrology)",
            "note": "Chosen because the track Jacobian dz is gauge-like.  Not an as-built z survey, and it does not constrain ry or C_dx.",
        },
        {
            "id": "ift_cassette_rigidity",
            "source": "mechanical design: three IFT layers on one support",
            "quantity": "possible C_dx ≈ 0 if the cassette is rigid",
            "numerical_value": None,
            "sigma": None,
            "coordinate_system": "IFT cassette",
            "reference_frame": "common support",
            "year_iov": "static cassette, unless opened or thermally deformed",
            "precision_sigma": None,
            "independent_of_track_residual": True,
            "constrains": ["C_dx"],
            "is_as_built_survey": False,
            "prior_form": None,
            "note": "Qualitative rigidity.  No documented deformation sigma; do not invent one.",
        },
        {
            "id": "independent_optical_or_laser_survey",
            "source": "workspace Calypso/geometry/conditions + CVMFS DBRelease + /eos/experiment/faser top-level",
            "quantity": "station ry or IFT layer relative dx",
            "numerical_value": None,
            "sigma": None,
            "independent_of_track_residual": True,
            "constrains": ["station_ry", "C_dx"],
            "is_as_built_survey": False,
            "found": False,
            "filename_hits": probe["survey_or_metrology_filename_hits"],
            "eos_top_level_dirs": probe["eos_experiment_faser_top_level_dirs"],
            "prior_form": "theta_ext ± sigma_ext or C θ = b, if a table arrives",
            "note": (
                "No as-built survey/metrology table with uncertainties was found. "
                "EOS top-level has data0/filter/gen/phys/raw/rec/runlist/sim/staged, not survey/. "
                "Paper-ready limitations already record this."
            ),
        },
        {
            "id": "paper_ready_limitation_no_survey",
            "source": "docs/operating_protocol_v1_paper_ready/main_claims_and_limitations.json",
            "quantity": "documented absence of independent survey",
            "numerical_value": None,
            "independent_of_track_residual": True,
            "constrains": [],
            "is_as_built_survey": False,
            "path": probe.get("paper_ready_limitations"),
            "note": "Limitation text: No independent survey or external station constraint is available in this corpus to fix dx/ry/dz.",
        },
    ]
    n_real = sum(1 for row in sources if row.get("is_as_built_survey") and row.get("numerical_value") is not None)
    return {
        **common_audit_state(),
        "n_as_built_survey_numbers": n_real,
        "as_built_survey_available": False,
        "do_not_invent_survey_numbers": True,
        "filesystem_probe": probe,
        "sources": sources,
        "usable_prior_today": (
            "None with a documented sigma.  Design LAYERPITCH supplies the "
            "ry↔C_dx unit map for the information-requirement scan only."
        ),
        "prior_conversion_if_survey_arrives": {
            "station_ry": "theta_ry_ext ± sigma_ry in the FASER global frame, independent of track r_u",
            "C_dx": "C_dx_ext ± sigma_C_dx from layer-0 vs layer-2 relative dx / 2 on the IFT cassette",
            "constraint_matrix": "C θ = b with C selecting ry and/or C_dx",
        },
    }


def _year_mode_policy(year_row: Mapping[str, Any]) -> dict[str, Any]:
    year = int(year_row["year"])
    independent = year_row.get("independent_of_ti12") is True
    ift = year_row.get("ift_in_production")
    common_static = [
        "module_design_stereo_and_strip_pitch",
        "IFT_LAYERPITCH_and_cassette_drawing",
    ]
    year_specific = ["station_ry", "station_dx_dy_rx_rz"]
    survey_fixed_if_available = ["station_dz"]
    if independent:
        common_static = ["module_design_stereo_and_strip_pitch"]
        year_specific = ["TestBeam2021_TB00", "station_ry", "station_dx_dy_rx_rz"]
        survey_fixed_if_available = ["station_dz"]
        c_dx = "not_a_parameter_no_IFT"
    elif ift is True or ift == "mixed":
        survey_fixed_if_available.append("C_dx_internal_deformation")
        c_dx = "common_static_until_evidence_of_opening_or_thermal_change"
        if year == 2022:
            year_specific.append("2022_r0021_pre_IFT")
    else:
        c_dx = "not_a_parameter"
    return {
        "share_across_years": common_static,
        "must_be_independent": year_specific,
        "survey_fixed_if_available": survey_fixed_if_available,
        "C_dx_policy": c_dx,
        "do_not_merge_with_other_years": True,
    }


def year_iov_manifest(config: Mapping[str, Any]) -> dict[str, Any]:
    pool_dir = Path(str(config["official_alignment_pool_dir"]))
    years = []
    for row in config["year_iovs"]:
        files = []
        for name in row.get("alignment_pool_files") or []:
            path = pool_dir / name
            files.append(
                {
                    "name": name,
                    "exists": path.is_file(),
                    "size_bytes": int(path.stat().st_size) if path.is_file() else None,
                    "content_interpreted_as": "neutral_WriteAlignment_payload_not_survey",
                }
            )
        years.append(
            {
                **dict(row),
                "alignment_pool": files,
                "mode_policy": _year_mode_policy(row),
                "do_not_merge_residuals_or_corrections_with_other_years": True,
            }
        )
    return {
        **common_audit_state(),
        "years": years,
        "current_protocol_conditions_tag": "OFLCOND-FASER-06",
        "production_2024_r0022_conditions_tag": "OFLCOND-FASER-05",
        "do_not_merge_residuals_or_corrections_across_years": True,
        "cluster_local_jacobian_this_stage": {
            "years": [2024],
            "runs": list(config["track_jacobians"]["runs"]),
            "note": "Only 2024 r0022 dumps 14973/14974 are Jacobianed here. Other years are provenance/track-sample inventory, not a mixed residual fit.",
        },
        "note": (
            "Official POOL filenames already encode year.  Production 2024 used "
            "OFLCOND-FASER-05 while the current protocol reads OFLCOND-FASER-06. "
            "That is an IOV difference even inside 2024."
        ),
    }


def iov_parameterization_report(config: Mapping[str, Any]) -> dict[str, Any]:
    modes = [
        {
            "mode": "module_design_stereo_and_strip_pitch",
            "share_across_years": "common_static",
            "year_specific": False,
            "survey_fixed_if_available": False,
            "reason": "sensor manufacturing; not an alignment IOV",
        },
        {
            "mode": "IFT_LAYERPITCH_and_cassette_drawing",
            "share_across_years": "common_static",
            "year_specific": False,
            "survey_fixed_if_available": True,
            "reason": "One IFT cassette.  As-built C_dx metrology, if it exists, is a static constraint.",
        },
        {
            "mode": "C_dx_internal_deformation",
            "share_across_years": "common_static_until_evidence_of_opening_or_thermal_change",
            "year_specific": False,
            "survey_fixed_if_available": True,
            "reason": "Internal to the cassette.  Do not float a new C_dx in every year without evidence.",
        },
        {
            "mode": "station_ry",
            "share_across_years": False,
            "year_specific": True,
            "survey_fixed_if_available": True,
            "reason": "Station orientation can change with installation, IFT insertion (2022), and TI12 access.  Δθ_year^(y).",
        },
        {
            "mode": "station_dx_dy_rx_rz",
            "share_across_years": False,
            "year_specific": True,
            "survey_fixed_if_available": False,
            "reason": "Rigid station pose is the year-level data term inside one IOV.  Tracks already separate dx from C_dx.",
        },
        {
            "mode": "station_dz",
            "share_across_years": False,
            "year_specific": False,
            "survey_fixed_if_available": True,
            "reason": "Station Jacobian dz is gauge-like.  Keep survey-constrained; do not float from these tracks.",
        },
        {
            "mode": "TestBeam2021_TB00",
            "share_across_years": False,
            "year_specific": True,
            "survey_fixed_if_available": False,
            "reason": "Different geometry tag and no IFT/C_dx.  Fully independent of TI12 years.",
        },
        {
            "mode": "2022_r0021_pre_IFT",
            "share_across_years": False,
            "year_specific": True,
            "survey_fixed_if_available": False,
            "reason": "No C_dx parameter.  Do not copy 2024 IFT internals onto 2022 r0021.",
        },
    ]
    return {
        **common_audit_state(),
        "form": config["parameterization"]["form"],
        "do_not_free_fill_or_run_initially": True,
        "do_not_share_one_constant_set_across_years": True,
        "static_component": (
            "Manufacturing and the IFT cassette drawing, plus any future as-built "
            "layer metrology that does not change by year."
        ),
        "year_component": (
            "Station rigid pose Δθ_year^(y), constrained only by data from that "
            "year's geometry/conditions IOV.  Subdivide to fill/run only after "
            "year-level residuals show time structure."
        ),
        "modes": modes,
        "official_db_already_year_split": True,
        "shared_constants_hypothesis": "rejected_by_provenance",
    }


def decide_next_stage(
    *,
    inventory: Mapping[str, Any],
    collinear_scan: Mapping[str, Any],
    mixed_scan: Mapping[str, Any] | None,
    parameterization: Mapping[str, Any],
) -> dict[str, Any]:
    ry_need = collinear_scan.get("weakest_ry_only_unlock_mrad")
    cdx_need = collinear_scan.get("weakest_cdx_only_unlock_mm")
    leftover_cdx = collinear_scan.get("ry_only_leftover_sigma_cdx_mm_at_tightest")
    leftover_ry = collinear_scan.get("cdx_only_leftover_sigma_ry_mrad_at_tightest")
    leftover_reaches_pitch = bool(collinear_scan.get("ry_only_leftover_cdx_finer_than_strip_pitch"))
    survey_missing = not bool(inventory.get("as_built_survey_available"))
    iov_required = bool(parameterization.get("do_not_share_one_constant_set_across_years"))
    answers = {
        "q1_type_and_precision": {
            "question": GO_NO_GO_QUESTIONS[0],
            "answer": (
                "Type: independent station-ry in the FASER global frame, or IFT "
                "layer-relative C_dx=(dx_L0-dx_L2)/2 on the cassette, or both. "
                "Design LAYERPITCH, WriteAlignment POOL files, and the software "
                "dz=5 mm gauge are not as-built constraints. Degeneracy-breaking "
                f"precision on the collinear 2024 r0022 Jacobian: σ(ry) ≲ {ry_need} mrad "
                f"or σ(C_dx) ≲ {cdx_need} mm (plateau, not a knife-edge). That restores "
                "rank-3 and |ρ(ry,C_dx)|≤0.90. Leftover σ(C_dx) after a tight ry prior "
                f"saturates at {leftover_cdx} mm, "
                f"{'already' if leftover_reaches_pitch else 'not'} finer than strip pitch; "
                f"a tight C_dx prior leaves σ(ry)≈{leftover_ry} mrad. No as-built number "
                "exists today."
            ),
            "required_sigma_ry_mrad": ry_need,
            "required_sigma_cdx_mm": cdx_need,
            "leftover_sigma_cdx_mm_after_tight_ry": leftover_cdx,
            "leftover_sigma_ry_mrad_after_tight_cdx": leftover_ry,
            "leftover_cdx_finer_than_strip_pitch": leftover_reaches_pitch,
            "as_built_survey_available": False,
        },
        "q2_year_iov_parameterization": {
            "question": GO_NO_GO_QUESTIONS[1],
            "answer": "Yes.  Use θ^(y)=θ_static+Δθ_year^(y).  Do not share one alignment-constant set across 2022/2023/2024/2025.",
            "use_common_static_plus_year_delta": True,
            "share_one_constant_set": False,
            "official_pool_files_already_year_split": True,
        },
    }
    return {
        **common_audit_state(),
        "decision": DECISION_SURVEY_AND_IOV,
        "answers": answers,
        "go_to_full_module_identifiability_map": False,
        "still_no_new_network": True,
        "geometry_write_allowed": False,
        "emits_alignment_payload": False,
        "survey_numbers_were_invented": False,
        "prior_was_selected_from_residuals": False,
        "as_built_survey_missing": survey_missing,
        "year_iov_parameterization_required": iov_required,
        "collinear_prior_restores_rank3": bool(ry_need is not None or cdx_need is not None),
        "mixed_sample_already_rank3_without_prior": bool(
            mixed_scan and mixed_scan.get("track_only_rank") == 3
        ),
        "next_allowed_step": (
            "obtain_independent_station_ry_or_ift_layer_metrology_then_fit_"
            "common_static_plus_year_delta_inside_each_iov"
        ),
        "reason": (
            "Track topology cannot portably split ry and C_dx.  A Gaussian prior "
            "on either parameter restores a rank-3 combined Fisher on the "
            "collinear collision-like Jacobian once the prior is tighter than "
            f"σ(ry)={ry_need} mrad or σ(C_dx)={cdx_need} mm.  That is degeneracy "
            "breaking, not strip-pitch C_dx: leftover σ(C_dx) after a tight ry "
            f"prior saturates at {leftover_cdx} mm.  Official alignment POOL files "
            "are already year-tagged and production conditions tags differ "
            "(OFLCOND-FASER-04/05 vs protocol -06).  Share manufacturing/cassette "
            "static terms; constrain year deltas only inside one IOV."
        ),
    }
