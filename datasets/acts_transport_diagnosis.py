"""Task B3: ACTS transport covariance physical diagnosis.

Explains why C1 = F Cin F^T + Q_ACTS fails e = x_prop - x_target closure.
Does not repair closure, enter Measurement Model V2, tune Q, or drop tails.
WB87–WB98 tokens stay frozen.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from alignment.operating_protocol_v1_final_closure import (
    project_root,
    resolve_under_root,
    sha256_file,
)
from alignment.propagated_covariance_closure import _load_truth_table
from datasets.access_policy import AccessScope, authorize_path
from datasets.acts_process_noise_contract import (
    DECISION_NOT_ESTABLISHED as WB97_DECISION,
    FAILURE_MEASUREMENT_MODEL,
    MECHANISM_CLOSURE_FAILED,
    MODEL0,
    MODEL1,
    _as_matrix,
    _pick_truth,
    _qoverp_bin_name,
    _truth_residual,
)
from datasets.acts_transport_dump import (
    DECISION_NOT_ESTABLISHED as WB98_DECISION,
    SITUATION_B,
)
from datasets.qoverp_covariance_export import (
    DECISION_ESTABLISHED as WB96_DECISION,
    provenance_hashes as wb96_provenance_hashes,
)
from datasets.qoverp_semantics import (
    DECISION_NOT_ESTABLISHED as WB95_DECISION,
    MECHANISM_NOT_EXPORTED as WB95_MECHANISM,
)

SCHEMA_VERSION = "acts-transport-diagnosis-v1"
DEFAULT_CONFIG = "configs/acts_transport_diagnosis_v1.yaml"
TASK = "SB-B3"
WORKBOOK = 99

STATE_NAMES = ("x", "y", "tx", "ty", "q_over_p")

CASE_A = "transport_contract_state_surface_or_jacobian"
CASE_B = "acts_material_process_noise_mismatch"
CASE_C = "high_chi2_tail_dominated"
CASE_D = "measurement_uncertainty_model_insufficient"

NEXT_STEP = {
    CASE_A: "fix_transport_contract",
    CASE_B: "acts_material_diagnosis",
    CASE_C: "uncertainty_model_analysis_no_covariance_tuning",
    CASE_D: "reassess_measurement_model_v2_input_model",
}

DECISION_DIAGNOSED = "acts_transport_covariance_failure_diagnosed"


class TransportDiagnosisError(ValueError):
    """Raised when the B3 diagnosis contract is illegal."""


def refuse_truth_qoverp() -> None:
    raise TransportDiagnosisError("truth q/p is not a real-data solution")


def refuse_dummy_segmentfit() -> None:
    raise TransportDiagnosisError("dummy SegmentFit covariance is not a physical prior")


def refuse_covariance_rescale() -> None:
    raise TransportDiagnosisError("covariance rescale is forbidden")


def refuse_process_noise_tuning() -> None:
    raise TransportDiagnosisError("process noise must not be adjusted to chi2")


def refuse_outlier_rejection() -> None:
    raise TransportDiagnosisError("outlier rejection and chi2 clipping are forbidden")


def refuse_measurement_model_v2() -> None:
    raise TransportDiagnosisError("Measurement Model V2 is not entered in Task B3")


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = resolve_under_root(project_root(), str(path or DEFAULT_CONFIG))
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != SCHEMA_VERSION:
        raise TransportDiagnosisError(f"schema_version must be {SCHEMA_VERSION}")
    if config.get("task") != TASK:
        raise TransportDiagnosisError(f"task must be {TASK}")
    if int(config.get("workbook", -1)) != WORKBOOK:
        raise TransportDiagnosisError(f"workbook must be {WORKBOOK}")
    for key in (
        "geometry_write_allowed",
        "held_out_accessed",
        "real_data_alignment_authorized",
        "measurement_model_validated",
        "do_not_enter_measurement_model_v2",
    ):
        if key == "do_not_enter_measurement_model_v2":
            if bool(config.get(key, False)) is not True:
                raise TransportDiagnosisError(f"{key} must be true")
            continue
        if bool(config.get(key, True)):
            raise TransportDiagnosisError(f"{key} must be false")
    for key in (
        "do_not_rescale_covariance",
        "do_not_use_truth_q_over_p_as_real_data_solution",
        "do_not_use_dummy_segmentfit_covariance",
        "do_not_tune_covariance_to_chi2",
        "do_not_tune_process_noise",
        "do_not_delete_state_variables",
        "do_not_reject_outliers",
        "do_not_clip_chi2_tails",
        "do_not_enter_alignment",
        "do_not_reread_sealed_test",
        "do_not_start_new_source_campaign",
        "do_not_write_geometry_or_conditions_payload",
        "do_not_modify_faseracts_extrapolation_tool_source",
        "do_not_construct_acts_objects_in_python",
        "unconstrained_tracker_only_stopped",
    ):
        if bool(config.get(key, False)) is not True:
            raise TransportDiagnosisError(f"{key} must be true")
    bins = config.get("qoverp_bins_abs_per_mev") or {}
    expected = {
        "high_momentum": [0.0, 1.0e-6],
        "medium_momentum": [1.0e-6, 5.0e-6],
        "low_momentum": [5.0e-6, 1.0],
    }
    for name, edges in expected.items():
        got = [float(v) for v in bins.get(name, [])]
        if got != edges:
            raise TransportDiagnosisError(f"q/p bin {name} must stay frozen")
    if Path(str(config.get("output_root"))).as_posix() == Path(
        str(config.get("dump_root"))
    ).as_posix():
        raise TransportDiagnosisError("B3 must not write into the WB98 dump root")
    return dict(config)


def _expect_sha(path: Path, expected: str, label: str) -> None:
    digest = sha256_file(path)
    if digest != expected:
        raise TransportDiagnosisError(f"{label} hash mismatch: {digest}")


def inherit_frozen_stage(config: Mapping[str, Any]) -> dict[str, Any]:
    inherited: dict[str, Any] = {}
    checks = (
        ("workbook_87", None, None, None),
        ("workbook_95", "mechanism", None, None),
        ("workbook_96", None, None, None),
        ("workbook_97", "mechanism", "failure_type", None),
        ("workbook_98", "mechanism", "failure_type", "frozen_situation"),
    )
    for name, mechanism_key, failure_key, situation_key in checks:
        spec = config["inheritance"][name]
        decision = json.loads(
            resolve_under_root(project_root(), spec["decision_path"]).read_text(
                encoding="utf-8"
            )
        )
        _expect_sha(
            resolve_under_root(project_root(), spec["config_path"]),
            spec["config_sha256"],
            f"{name} config",
        )
        _expect_sha(
            resolve_under_root(project_root(), spec["decision_path"]),
            spec["decision_sha256"],
            f"{name} decision",
        )
        if decision.get("decision") != spec["frozen_decision"]:
            raise TransportDiagnosisError(f"{name} decision must stay frozen")
        if mechanism_key and spec.get("frozen_mechanism"):
            if decision.get("mechanism") != spec.get("frozen_mechanism"):
                raise TransportDiagnosisError(f"{name} mechanism must stay frozen")
        if failure_key and spec.get("frozen_failure_type"):
            if decision.get(failure_key) != spec.get("frozen_failure_type"):
                raise TransportDiagnosisError(f"{name} failure_type must stay frozen")
        if situation_key and spec.get(situation_key):
            if decision.get("situation") != spec.get(situation_key):
                raise TransportDiagnosisError(f"{name} situation must stay frozen")
        inherited[name] = {
            "decision": spec["frozen_decision"],
            "config_sha256": spec["config_sha256"],
            "decision_sha256": spec["decision_sha256"],
        }
        if spec.get("frozen_mechanism"):
            inherited[name]["mechanism"] = spec["frozen_mechanism"]
        if spec.get("frozen_failure_type"):
            inherited[name]["failure_type"] = spec["frozen_failure_type"]
        if spec.get("frozen_situation"):
            inherited[name]["situation"] = spec["frozen_situation"]
    if inherited["workbook_95"]["decision"] != WB95_DECISION:
        raise TransportDiagnosisError("WB95 decision token mismatch")
    if inherited["workbook_95"].get("mechanism") != WB95_MECHANISM:
        raise TransportDiagnosisError("WB95 mechanism token mismatch")
    if inherited["workbook_96"]["decision"] != WB96_DECISION:
        raise TransportDiagnosisError("WB96 must keep the CKF export established")
    if inherited["workbook_97"]["decision"] != WB97_DECISION:
        raise TransportDiagnosisError("WB97 decision must stay not-established")
    if inherited["workbook_98"]["decision"] != WB98_DECISION:
        raise TransportDiagnosisError("WB98 decision must stay not-established")
    if inherited["workbook_98"].get("mechanism") != MECHANISM_CLOSURE_FAILED:
        raise TransportDiagnosisError("WB98 mechanism must stay closure-failed")
    if inherited["workbook_98"].get("failure_type") != FAILURE_MEASUREMENT_MODEL:
        raise TransportDiagnosisError("WB98 failure_type must stay frozen")
    if inherited["workbook_98"].get("situation") != SITUATION_B:
        raise TransportDiagnosisError("WB98 situation must stay acts_q_materialized_closure_failed")
    return inherited


def dump_path_for_source(config: Mapping[str, Any], source_id: str) -> Path:
    root = resolve_under_root(project_root(), str(config["dump_root"]))
    return root / source_id / str(config.get("dump_filename", "ckf_acts_transport.jsonl"))


def load_dump_records(path: Path, *, split: str) -> list[dict[str, Any]]:
    authorized = authorize_path(path, AccessScope.DEVELOPMENT_VALIDATION, split=split)
    if not authorized.is_file():
        return []
    rows = []
    for line in authorized.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _finite(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    return array[np.isfinite(array)]


def _stats(values: Sequence[float]) -> dict[str, int | float | None]:
    finite = _finite(values)
    result: dict[str, int | float | None] = {
        "count": int(np.asarray(values).size),
        "finite_count": int(finite.size),
        "mean": None,
        "rms": None,
        "median": None,
        "min": None,
        "max": None,
        "p95": None,
    }
    if finite.size:
        result.update(
            {
                "mean": float(np.mean(finite)),
                "rms": float(np.sqrt(np.mean(finite * finite))),
                "median": float(np.median(finite)),
                "min": float(np.min(finite)),
                "max": float(np.max(finite)),
                "p95": float(np.quantile(finite, 0.95)),
            }
        )
    return result


def _bin_name(value: float, bins: Mapping[str, Sequence]) -> str | None:
    number = float(value)
    for name, edges in bins.items():
        lo, hi = float(edges[0]), float(edges[1])
        if lo <= number < hi:
            return str(name)
    return None


def _safe_corr(left: Sequence[float], right: Sequence[float]) -> float | None:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    mask = np.isfinite(a) & np.isfinite(b)
    if int(np.count_nonzero(mask)) < 3:
        return None
    if float(np.std(a[mask])) == 0.0 or float(np.std(b[mask])) == 0.0:
        return None
    return float(np.corrcoef(a[mask], b[mask])[0, 1])


def _eigen_spectrum(matrix: np.ndarray) -> dict[str, Any]:
    values = np.linalg.eigvalsh(matrix)
    return {
        "eigenvalues": [float(v) for v in values],
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "n_negative": int(np.sum(values < 0.0)),
        "psd": bool(np.all(values >= -1.0e-12)),
    }


def compare_c0_to_jacobian(
    transport_jacobian: np.ndarray,
    input_covariance: np.ndarray,
    output_covariance_no_material: np.ndarray,
) -> dict[str, Any]:
    """Direct C0 vs F Cin F^T.  Does not infer F from covariances."""
    predicted = transport_jacobian @ input_covariance @ transport_jacobian.T
    delta = output_covariance_no_material - predicted
    denom = float(np.linalg.norm(output_covariance_no_material))
    rel = float(np.linalg.norm(delta) / denom) if denom > 0.0 else None
    diag_rel = []
    row_rel = []
    for index, name in enumerate(STATE_NAMES):
        c0_ii = float(output_covariance_no_material[index, index])
        pred_ii = float(predicted[index, index])
        row_denom = float(np.linalg.norm(output_covariance_no_material[index]))
        diag_rel.append(
            {
                "state": name,
                "c0_diag": c0_ii,
                "f_cin_ft_diag": pred_ii,
                "abs_delta_diag": abs(c0_ii - pred_ii),
                "rel_diag": abs(c0_ii - pred_ii) / abs(c0_ii) if abs(c0_ii) > 0.0 else None,
            }
        )
        row_rel.append(
            {
                "state": name,
                "row_frobenius_rel": (
                    float(np.linalg.norm(delta[index]) / row_denom) if row_denom > 0.0 else None
                ),
            }
        )
    return {
        "frobenius_relative_error": rel,
        "per_state_diagonal": diag_rel,
        "per_state_row": row_rel,
        "inferred_f_from_covariance": False,
    }


def _read_helper_source(config: Mapping[str, Any]) -> str:
    relative = str(config["software_provenance"]["dump_helper"])
    path = resolve_under_root(project_root(), relative)
    if not path.is_file():
        raise TransportDiagnosisError(f"dump helper is absent: {relative}")
    return path.read_text(encoding="utf-8", errors="replace")


def audit_state_surface_contract(
    config: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """B3.1: bound → curvilinear → export and surface convention."""
    source = _read_helper_source(config)
    provenance = wb96_provenance_hashes(config)
    checks = {
        "plane_at_z_origin_normal_z": "PlaneSurface" in source
        and bool(
            re.search(
                r"Vector3\(0\.0,\s*0\.0,\s*zMm\),\s*Acts::Vector3\(0\.0,\s*0\.0,\s*1\.0\)",
                source,
            )
        ),
        "export_uses_global_position_not_loc": "position.x()" in source
        and "position.y()" in source
        and "momentum.x() / momentum.z()" in source,
        "export_does_not_assign_loc_to_xy": not bool(
            re.search(r"values\[0\]\s*=\s*parameters\.parameters\(\)\[Acts::eBoundLoc0\]", source)
        ),
        "forward_if_target_z_ge_source_z": "targetZ >= sourceZ ? Acts::Direction::Forward"
        in source,
        "native_qoverp_divided_by_mev": "nativeQOverP / 1_MeV" in source,
        "export_qoverp_times_mev": "eBoundQOverP] * 1_MeV" in source,
        "particle_hypothesis_muon": "ParticleHypothesis::muon()" in source,
        "no_truth_qoverp_override": "truth" not in source.lower()
        or "is_truth\"] = false" in source,
        "q_defined_as_c1_minus_c0": "subtract5(*cov1, *cov0)" in source,
        "f_from_no_material_export_perturbation": "propagateToZ(*m_toolNoNoise" in source
        and "actsFromExport" in source,
        "f_not_inferred_from_c0": "transport_jacobian" in source
        and "output_covariance_no_material_from_jacobian" in source,
    }
    n_rows = 0
    n_forward = 0
    n_qop_sign = 0
    n_hash = 0
    n_surface = 0
    n_native_loc_zero = 0
    fx_qop = []
    fy_qop = []
    sample_f = None
    sample_surface = None
    sample_state = None
    expected_geo = provenance["geometry_hash"]
    for row in records:
        n_rows += 1
        source_z = float(row.get("source_z_mm", 0.0))
        target_z = float(row.get("target_z_mm", 0.0))
        if target_z >= source_z:
            n_forward += 1
        qop = float(row.get("q_over_p_per_mev", 0.0))
        charge = float(row.get("charge", 0.0))
        if qop * charge > 0.0:
            n_qop_sign += 1
        if str(row.get("geometry_hash")) == expected_geo:
            n_hash += 1
        surface = row.get("surface") or {}
        normal = surface.get("normal")
        origin = surface.get("origin_mm")
        if (
            surface.get("type") == "plane"
            and isinstance(normal, list)
            and normal == [0.0, 0.0, 1.0]
            and isinstance(origin, list)
            and len(origin) == 3
            and float(origin[2]) == float(surface.get("z_mm", target_z))
        ):
            n_surface += 1
        native = np.asarray(row.get("native_state"), dtype=np.float64).reshape(-1)
        if native.size >= 2 and abs(float(native[0])) <= 1.0e-12 and abs(float(native[1])) <= 1.0e-12:
            n_native_loc_zero += 1
        jacobian = _as_matrix(row.get("transport_jacobian"), 5)
        if jacobian is not None:
            fx_qop.append(float(jacobian[0, 4]))
            fy_qop.append(float(jacobian[1, 4]))
            if sample_f is None:
                sample_f = jacobian.tolist()
                sample_surface = surface
                sample_state = {
                    "native_state": row.get("native_state"),
                    "derived_state": row.get("derived_state"),
                    "export_state_model1": (row.get(MODEL1) or {}).get(
                        "state_xy_tx_ty_qoverp"
                    ),
                    "source_z_mm": source_z,
                    "target_z_mm": target_z,
                    "q_over_p_per_mev": qop,
                    "charge": charge,
                    "phi": float(native[2]) if native.size >= 3 else None,
                    "theta": float(native[3]) if native.size >= 4 else None,
                }
    frame_holds = (
        all(checks.values())
        and n_rows > 0
        and n_forward == n_rows
        and n_qop_sign == n_rows
        and n_hash == n_rows
        and n_surface == n_rows
    )
    mean_abs_fx = float(np.mean(np.abs(fx_qop))) if fx_qop else None
    mean_abs_fy = float(np.mean(np.abs(fy_qop))) if fy_qop else None
    bending_axis = None
    if mean_abs_fx is not None and mean_abs_fy is not None:
        bending_axis = "y" if mean_abs_fy > mean_abs_fx else "x"
    return {
        "input_state_definition": {
            "native_athena": ["loc1_mm", "loc2_mm", "phi", "theta", "q_over_p_per_mev"],
            "frame": "athena_trackparameters_curvilinear_or_bound",
            "units": ["mm", "mm", "rad", "rad", "1/MeV"],
            "q_over_p_signed": True,
            "loc_is_global_xy": False,
            "note": "CKF front parameters; loc1=loc2=0 is the curvilinear origin, not a missing x/y.",
        },
        "output_state_definition": {
            "parameters": ["x_mm", "y_mm", "tx", "ty", "q_over_p_per_mev"],
            "frame": "global_cartesian_slopes_at_surface_z",
            "units": ["mm", "mm", "1", "1", "1/MeV"],
            "q_over_p_signed": True,
            "source": "Acts BoundTrackParameters.position() and momentum()",
        },
        "acts_bound_state_definition": {
            "parameters": ["loc0", "loc1", "phi", "theta", "q_over_p_per_GeV", "time"],
            "q_over_p_unit": "1/GeV",
            "conversion": "native_q_over_p_per_MeV / 1_MeV",
        },
        "frame": "global_cartesian_slopes_at_surface_z",
        "units": ["mm", "mm", "1", "1", "1/MeV"],
        "surface_identifier": {
            "type": "Acts::PlaneSurface",
            "constructor": "PlaneSurface((0,0,z_mm), normal=(0,0,1))",
            "source_z": "track position z",
            "target_z": "frozen WB87 station z",
            "local_axes": "plane u/v in the global x/y plane; export does not use loc as x/y",
        },
        "propagation_direction": {
            "rule": "Forward iff target_z >= source_z, else Backward",
            "n_rows": n_rows,
            "n_forward": n_forward,
            "all_forward_on_registered_pairs": n_forward == n_rows and n_rows > 0,
        },
        "phi_theta_convention": {
            "native_phi": "atan2(py, px) on the Athena TrackParameters",
            "native_theta": "polar angle of momentum, beam tracks ~ 0",
            "export_slopes": "tx = px/pz, ty = py/pz",
        },
        "q_over_p_sign": {
            "native_unit": "1/MeV",
            "acts_unit": "1/GeV",
            "export_unit": "1/MeV",
            "signed": True,
            "n_rows": n_rows,
            "n_sign_matches_charge": n_qop_sign,
            "all_match": n_qop_sign == n_rows and n_rows > 0,
        },
        "transformation_jacobian": {
            "native_to_acts_bound": "numerical 6x5 from nativeSurface.createUniqueTrackParameters plus transformFreeToBoundParameters",
            "acts_bound_to_export": "numerical 5x6 of exportFromBound",
            "transport_F": "numerical 5x5 of no-material propagate in export space; not inferred from C0",
            "sample_transport_F": sample_f,
            "mean_abs_F_x_qoverp": mean_abs_fx,
            "mean_abs_F_y_qoverp": mean_abs_fy,
            "magnetic_bending_axis_from_F": bending_axis,
            "expected_faser_bending_axis": "y",
            "bending_axis_consistent_with_Bx": bending_axis == "y",
        },
        "geometry_hash": expected_geo,
        "field_hash": provenance["field_hash"],
        "conditions_hash": provenance["conditions_hash"],
        "n_geometry_hash_match": n_hash,
        "n_surface_contract_match": n_surface,
        "n_native_loc_zero": n_native_loc_zero,
        "sample": sample_state,
        "sample_surface": sample_surface,
        "source_checks": checks,
        "frame_contract_holds": frame_holds,
        "principal_failure_direction_from_wb98": "x",
        "x_as_export_global_not_native_loc": True,
    }


def audit_jacobian_consistency(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """B3.2: C0 vs F Cin F^T on dumped matrices only."""
    rels = []
    per_state = {name: {"diag_rel": [], "row_rel": []} for name in STATE_NAMES}
    dumped_rel = []
    n_compared = 0
    n_mismatch_dump = 0
    q_min_eigs = []
    n_q_not_psd = 0
    n_q_xx_negative = 0
    n_c1_xx_lt_c0 = 0
    for row in records:
        f_mat = _as_matrix(row.get("transport_jacobian"), 5)
        cin = _as_matrix(row.get("input_covariance"), 5)
        c0 = _as_matrix(row.get("output_covariance_no_material"), 5)
        if f_mat is None or cin is None or c0 is None:
            continue
        report = compare_c0_to_jacobian(f_mat, cin, c0)
        n_compared += 1
        if report["frobenius_relative_error"] is not None:
            rels.append(float(report["frobenius_relative_error"]))
        dumped = row.get("c0_jacobian_frobenius_rel")
        if dumped is not None and np.isfinite(float(dumped)):
            dumped_rel.append(float(dumped))
            if report["frobenius_relative_error"] is not None:
                if abs(float(dumped) - float(report["frobenius_relative_error"])) > 1.0e-9:
                    n_mismatch_dump += 1
        for item in report["per_state_diagonal"]:
            if item["rel_diag"] is not None:
                per_state[item["state"]]["diag_rel"].append(float(item["rel_diag"]))
        for item in report["per_state_row"]:
            if item["row_frobenius_rel"] is not None:
                per_state[item["state"]]["row_rel"].append(float(item["row_frobenius_rel"]))
        q_mat = _as_matrix(row.get("process_noise"), 5)
        c1 = _as_matrix(row.get("output_covariance_with_material"), 5)
        if q_mat is not None:
            spectrum = _eigen_spectrum(q_mat)
            q_min_eigs.append(spectrum["min"])
            if not spectrum["psd"]:
                n_q_not_psd += 1
            if float(q_mat[0, 0]) < 0.0:
                n_q_xx_negative += 1
        if c0 is not None and c1 is not None and float(c1[0, 0]) < float(c0[0, 0]):
            n_c1_xx_lt_c0 += 1
    finite = _finite(rels)
    n_above_median_gate = int(np.sum(finite > 0.01)) if finite.size else 0
    n_above_p95_gate = int(np.sum(finite > 0.05)) if finite.size else 0
    systematic = bool(
        finite.size
        and float(np.median(finite)) < 0.01
    )
    return {
        "n_rows": len(records),
        "n_compared": n_compared,
        "inferred_f_from_covariance": False,
        "frobenius_relative_error": _stats(rels),
        "dumped_c0_jacobian_frobenius_rel": _stats(dumped_rel),
        "n_dumped_rel_mismatch": n_mismatch_dump,
        "per_state": {
            name: {
                "diag_rel": _stats(payload["diag_rel"]),
                "row_frobenius_rel": _stats(payload["row_rel"]),
            }
            for name, payload in per_state.items()
        },
        "q_as_c1_minus_c0": {
            "n_not_psd": n_q_not_psd,
            "n_q_xx_negative": n_q_xx_negative,
            "n_c1_xx_lt_c0": n_c1_xx_lt_c0,
            "min_eigenvalue": _stats(q_min_eigs),
            "note": "Q_ACTS := C1-C0 is the honest increment, not a PSD process-noise matrix.",
        },
        "n_rel_above_0_01": n_above_median_gate,
        "n_rel_above_0_05": n_above_p95_gate,
        "fraction_rel_above_0_05": (
            float(n_above_p95_gate / finite.size) if finite.size else None
        ),
        "systematic_jacobian_holds": systematic,
        "jacobian_tail_present": bool(
            finite.size and float(np.quantile(finite, 0.95)) >= 0.05
        ),
        "jacobian_contract_holds": systematic,
    }


def _chi2_ndof(residual: np.ndarray, covariance: np.ndarray) -> float | None:
    try:
        return float(residual @ np.linalg.solve(covariance, residual)) / float(residual.size)
    except np.linalg.LinAlgError:
        return None


def collect_matched_events(
    config: Mapping[str, Any],
    *,
    records: Sequence[Mapping[str, Any]],
    truth_tables: Mapping[str, Mapping[tuple[int, int], Any]],
    split: str,
) -> list[dict[str, Any]]:
    """Join Model-1 predictions to truth.  Does not drop high-χ² tails."""
    tr = config["truth_reference"]
    dummy = float(config["acceptance"]["dummy_qoverp_per_mev"])
    frozen_pairs = {f"({int(a)},{int(b)})" for a, b in config["station_pairs"]}
    events: list[dict[str, Any]] = []
    for row in records:
        if bool(row.get("is_truth", False)):
            continue
        block = row.get(MODEL1) or {}
        if not bool(block.get("success", False)):
            continue
        cov4 = _as_matrix(block.get("covariance_4x4"), 4)
        cov5 = _as_matrix(block.get("covariance_5x5") or row.get("output_covariance_with_material"), 5)
        pred4 = np.asarray(block.get("state_xy_tx_ty"), dtype=np.float64).reshape(-1)
        pred5 = np.asarray(
            block.get("state_xy_tx_ty_qoverp") or list(pred4) + [row.get("q_over_p_per_mev")],
            dtype=np.float64,
        ).reshape(-1)
        if cov4 is None or pred4.size != 4:
            continue
        table = truth_tables.get(str(row.get("source_id")))
        if table is None:
            continue
        event_truth = table.get((int(row["run_id"]), int(row["event_id"])))
        if not event_truth:
            continue
        target_station = int(row["target_station"])
        _, tgt = _pick_truth(event_truth, target_station)
        if tgt is None:
            continue
        residual4 = _truth_residual(
            pred4,
            tgt["pos"],
            tgt["mom"],
            float(row["target_z_mm"]),
            straight_line=bool(tr["straight_line_z_correction"]),
        )
        if residual4 is None:
            continue
        dummy_q = abs(float(row.get("q_over_p_per_mev", 0.0)) - dummy) <= 1.0e-18
        if dummy_q:
            continue
        p_truth = float(np.linalg.norm(tgt["mom"]))
        charge = float(row.get("charge", np.nan))
        qop_pred = float(pred5[4]) if pred5.size == 5 else float(row.get("q_over_p_per_mev"))
        qop_truth = (charge / p_truth) if p_truth > 0.0 and np.isfinite(charge) else float("nan")
        residual_qop = qop_pred - qop_truth
        chi2 = _chi2_ndof(residual4, cov4)
        c0 = _as_matrix(row.get("output_covariance_no_material"), 5)
        q_mat = _as_matrix(row.get("process_noise"), 5)
        q_frob = float(row["q_frobenius"]) if row.get("q_frobenius") is not None else (
            float(np.linalg.norm(q_mat)) if q_mat is not None else float("nan")
        )
        c0_frob = float(np.linalg.norm(c0)) if c0 is not None else float("nan")
        q_over_c0 = q_frob / c0_frob if c0_frob and np.isfinite(c0_frob) and c0_frob > 0.0 else float("nan")
        tx, ty = float(pred4[2]), float(pred4[3])
        incidence = float(np.arctan(np.hypot(tx, ty)))
        pulls = []
        chi2_parts = []
        for index in range(4):
            var = float(cov4[index, index])
            pull = float(residual4[index] / np.sqrt(var)) if var > 0.0 else float("nan")
            pulls.append(pull)
            chi2_parts.append(float(residual4[index] * residual4[index] / var) if var > 0.0 else float("nan"))
        pull_qop = float("nan")
        if cov5 is not None and float(cov5[4, 4]) > 0.0 and np.isfinite(residual_qop):
            pull_qop = float(residual_qop / np.sqrt(float(cov5[4, 4])))
        events.append(
            {
                "split": split,
                "source_id": str(row.get("source_id")),
                "run_id": int(row["run_id"]),
                "event_id": int(row["event_id"]),
                "track_index": int(row.get("track_index", -1)),
                "source_station": int(row["source_station"]),
                "target_station": target_station,
                "station_pair": f"({int(row['source_station'])},{target_station})",
                "on_frozen_station_pair": f"({int(row['source_station'])},{target_station})"
                in frozen_pairs,
                "q_over_p_per_mev": qop_pred,
                "abs_q_over_p": abs(qop_pred),
                "p_mev": float(row.get("p_mev", float("nan"))),
                "p_truth_mev": p_truth,
                "charge": charge,
                "material_thickness_x0": None,
                "material_thickness_available": False,
                "q_frobenius": q_frob,
                "q_over_c0": q_over_c0,
                "incidence_angle_rad": incidence,
                "abs_slope": float(np.hypot(tx, ty)),
                "residual": {
                    "x": float(residual4[0]),
                    "y": float(residual4[1]),
                    "tx": float(residual4[2]),
                    "ty": float(residual4[3]),
                    "q_over_p": float(residual_qop) if np.isfinite(residual_qop) else None,
                },
                "pull": {
                    "x": pulls[0],
                    "y": pulls[1],
                    "tx": pulls[2],
                    "ty": pulls[3],
                    "q_over_p": pull_qop if np.isfinite(pull_qop) else None,
                },
                "chi2_contribution": {
                    "x": chi2_parts[0],
                    "y": chi2_parts[1],
                    "tx": chi2_parts[2],
                    "ty": chi2_parts[3],
                },
                "chi2_per_ndof": chi2,
                "q_contribution": {
                    "frobenius": q_frob,
                    "over_c0": q_over_c0,
                    "diag": [float(q_mat[i, i]) for i in range(5)] if q_mat is not None else None,
                },
                "c_eigen_spectrum": _eigen_spectrum(cov5) if cov5 is not None else _eigen_spectrum(cov4),
                "c4_eigen_spectrum": _eigen_spectrum(cov4),
                "truth_qoverp_used_as_cin": False,
                "dummy_rejected": False,
            }
        )
    return events


def _group_stats(events: Sequence[Mapping[str, Any]], key: str, bins: Mapping[str, Sequence] | None = None) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    if bins is None:
        for event in events:
            grouped.setdefault(str(event.get(key)), []).append(dict(event))
    else:
        for name in bins:
            grouped[name] = []
        for event in events:
            name = _bin_name(float(event[key]), bins)
            if name is not None:
                grouped[name].append(dict(event))
    return {name: residual_decomposition(items) for name, items in grouped.items()}


def residual_decomposition(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not events:
        return {"n": 0}
    chi2 = [event["chi2_per_ndof"] for event in events if event.get("chi2_per_ndof") is not None]
    components = {}
    for name in STATE_NAMES:
        residual = [event["residual"][name] for event in events if event["residual"].get(name) is not None]
        pull = [event["pull"][name] for event in events if event["pull"].get(name) is not None]
        contrib = [
            event["chi2_contribution"][name]
            for event in events
            if name in event["chi2_contribution"] and event["chi2_contribution"][name] is not None
        ]
        components[name] = {
            "residual": _stats(residual),
            "pull": _stats(pull),
            "chi2_contribution": _stats(contrib),
        }
    charges = [float(event["charge"]) for event in events]
    xs = [float(event["residual"]["x"]) for event in events]
    qops = [float(event["abs_q_over_p"]) for event in events]
    qfrobs = [float(event["q_frobenius"]) for event in events]
    return {
        "n": len(events),
        "chi2_per_ndof": _stats(chi2),
        "components": components,
        "x_vs_charge_correlation": _safe_corr(xs, charges),
        "x_vs_abs_qoverp_correlation": _safe_corr(xs, qops),
        "x_vs_q_frobenius_correlation": _safe_corr(xs, qfrobs),
        "x_abs_vs_abs_qoverp_correlation": _safe_corr(np.abs(xs), qops),
        "x_abs_vs_q_frobenius_correlation": _safe_corr(np.abs(xs), qfrobs),
    }


def _split_decomposition(
    events: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    by_split = {}
    for split in ("construction", "validation"):
        selected = [event for event in events if event["split"] == split]
        by_split[split] = {
            "overall": residual_decomposition(selected),
            "by_qoverp_bin": _group_stats(selected, "abs_q_over_p", config["qoverp_bins_abs_per_mev"]),
            "by_charge_bin": _group_stats(selected, "charge", config["charge_bins"]),
            "by_q_frobenius_bin": _group_stats(selected, "q_frobenius", config["q_frobenius_bins"]),
            "by_incidence_bin": _group_stats(
                selected, "abs_slope", config["incidence_abs_slope_bins"]
            ),
            "by_station_pair": _group_stats(selected, "station_pair"),
        }
    return by_split


def decompose_residuals(
    events: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """B3.3: per-component residual, pull, χ², and pre-registered dependences."""
    official = [event for event in events if event.get("on_frozen_station_pair")]
    extra = [event for event in events if not event.get("on_frozen_station_pair")]
    return {
        "bins_retuned": False,
        "truth_qoverp_used_as_cin": False,
        "material_thickness_x0": None,
        "material_proxy": ["q_frobenius", "q_over_c0"],
        "official_station_pairs": [f"({int(a)},{int(b)})" for a, b in config["station_pairs"]],
        "n_all_pairs": len(events),
        "n_official_pairs": len(official),
        "n_off_contract_pairs": len(extra),
        "off_contract_events_retained": True,
        "overall": residual_decomposition(official),
        "all_pairs_overall": residual_decomposition(events),
        "off_contract_overall": residual_decomposition(extra),
        "splits": _split_decomposition(official, config),
        "all_pairs_splits": _split_decomposition(events, config),
    }


def _tail_catalog(
    events: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    ranked = [
        event
        for event in events
        if event.get("chi2_per_ndof") is not None and np.isfinite(float(event["chi2_per_ndof"]))
    ]
    ranked.sort(key=lambda item: float(item["chi2_per_ndof"]), reverse=True)
    total = float(sum(float(event["chi2_per_ndof"]) for event in ranked)) if ranked else 0.0
    catalog = {}
    for quantile in config["tail_quantiles"]:
        if not ranked:
            catalog[str(quantile)] = {"n": 0, "events": [], "chi2_share": None}
            continue
        threshold = float(np.quantile([float(event["chi2_per_ndof"]) for event in ranked], float(quantile)))
        selected = [
            event for event in ranked if float(event["chi2_per_ndof"]) >= threshold
        ]
        share = (
            float(sum(float(event["chi2_per_ndof"]) for event in selected) / total)
            if total > 0.0
            else None
        )
        catalog[str(quantile)] = {
            "quantile": float(quantile),
            "threshold_chi2_per_ndof": threshold,
            "n": len(selected),
            "n_retained": len(selected),
            "rejected": 0,
            "clipped": 0,
            "chi2_share": share,
            "events": selected,
        }
    chi2_values = [float(event["chi2_per_ndof"]) for event in ranked]
    chi2_stats = _stats(chi2_values)
    return {
        "n_events": len(ranked),
        "n_dropped": 0,
        "outlier_rejection": False,
        "chi2_per_ndof": chi2_stats,
        "mean_over_median": (
            float(chi2_stats["mean"] / chi2_stats["median"])
            if chi2_stats["median"]
            else None
        ),
        "tails": catalog,
    }


def tail_provenance(
    events: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """B3.4: keep every high-χ² event.  No clip, no rejection."""
    official = [event for event in events if event.get("on_frozen_station_pair", True)]
    extra = [event for event in events if not event.get("on_frozen_station_pair", True)]
    report = _tail_catalog(official, config)
    report["scope"] = "frozen_station_pairs"
    report["all_pairs"] = _tail_catalog(events, config)
    report["off_contract_pairs"] = {
        "n_retained": len(extra),
        "rejected": 0,
        "chi2_per_ndof": residual_decomposition(extra).get("chi2_per_ndof"),
    }
    return report


def classify_failure(
    contract: Mapping[str, Any],
    jacobian: Mapping[str, Any],
    residuals: Mapping[str, Any],
    tails: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    gates = config["diagnosis_gates"]
    jacobian_ok = bool(jacobian.get("jacobian_contract_holds"))
    if jacobian.get("frobenius_relative_error", {}).get("median") is not None:
        jacobian_ok = jacobian_ok and (
            float(jacobian["frobenius_relative_error"]["median"])
            <= float(gates["jacobian_frobenius_rel_median_max"])
        )
    jacobian_tail = bool(jacobian.get("jacobian_tail_present"))
    if jacobian.get("frobenius_relative_error", {}).get("p95") is not None:
        jacobian_tail = jacobian_tail or (
            float(jacobian["frobenius_relative_error"]["p95"])
            > float(gates["jacobian_frobenius_rel_p95_max"])
        )
    frame_ok = bool(contract.get("frame_contract_holds"))
    bending_ok = bool(
        (contract.get("transformation_jacobian") or {}).get("bending_axis_consistent_with_Bx")
    )
    x_charge = residuals.get("overall", {}).get("x_vs_charge_correlation")
    charge_flag = (
        x_charge is not None
        and abs(float(x_charge)) >= float(gates["charge_mean_x_correlation_abs_case_a_min"])
        and not bending_ok
    )
    case_a = (not jacobian_ok) or (not frame_ok) or charge_flag

    q_info = jacobian.get("q_as_c1_minus_c0") or {}
    n_q = int(jacobian.get("n_compared") or 0)
    q_not_psd_frac = float(q_info.get("n_not_psd") or 0) / n_q if n_q else 0.0
    x_abs_q = residuals.get("overall", {}).get("x_abs_vs_q_frobenius_correlation")
    case_b = q_not_psd_frac >= 0.20 or (
        x_abs_q is not None and abs(float(x_abs_q)) >= 0.3 and q_not_psd_frac >= 0.05
    )

    tail_1 = (tails.get("tails") or {}).get("0.99") or {}
    tail_share = tail_1.get("chi2_share")
    mean_over_median = tails.get("mean_over_median")
    case_c = bool(
        (tail_share is not None and float(tail_share) >= float(gates["tail_chi2_share_case_c_min"]))
        or (
            mean_over_median is not None
            and float(mean_over_median) >= float(gates["mean_over_median_chi2_case_c_min"])
        )
    )

    median_chi2 = (residuals.get("overall", {}).get("chi2_per_ndof") or {}).get("median")
    bulk_overcover = median_chi2 is not None and float(median_chi2) < 1.0
    case_d = (not case_a) and (jacobian_ok and frame_ok)

    if case_a:
        primary = CASE_A
    elif case_c:
        primary = CASE_C
    elif case_b:
        primary = CASE_B
    else:
        primary = CASE_D
    secondaries = []
    if primary != CASE_C and case_c:
        secondaries.append(CASE_C)
    if primary != CASE_B and case_b:
        secondaries.append(CASE_B)
    if primary != CASE_D and case_d and bulk_overcover:
        secondaries.append(CASE_D)
    return {
        "primary_case": primary,
        "secondary_cases": secondaries,
        "next_step": NEXT_STEP[primary],
        "evidence": {
            "jacobian_contract_holds": jacobian_ok,
            "jacobian_tail_present": jacobian_tail,
            "frame_contract_holds": frame_ok,
            "bending_axis_consistent_with_Bx": bending_ok,
            "x_vs_charge_correlation": x_charge,
            "q_not_psd_fraction": q_not_psd_frac,
            "tail_1pct_chi2_share": tail_share,
            "mean_over_median_chi2": mean_over_median,
            "bulk_median_chi2_below_one": bulk_overcover,
        },
        "case_flags": {"A": case_a, "B": case_b, "C": case_c, "D": case_d},
        "measurement_model_v2_entered": False,
        "closure_repaired": False,
        "covariance_retuned": False,
        "tails_rejected": False,
    }


def _all_sources(config: Mapping[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    items = []
    for spec in config["mc_data"]["construction_sources"]:
        items.append(("construction", spec))
    for spec in config["mc_data"]["validation_sources"]:
        items.append(("validation", spec))
    return items


def inventory_and_diagnose(config: Mapping[str, Any]) -> dict[str, Any]:
    records_by_split: dict[str, list[dict[str, Any]]] = {
        "construction": [],
        "validation": [],
    }
    truth_tables: dict[str, Any] = {}
    present = []
    missing = []
    for split, spec in _all_sources(config):
        source_id = spec["source_id"]
        access = "train" if split == "construction" else "validation"
        authorize_path(spec["input_xaod"], AccessScope.DEVELOPMENT_VALIDATION, split=access)
        path = dump_path_for_source(config, source_id)
        rows = load_dump_records(path, split=access)
        if not rows:
            missing.append(source_id)
            continue
        present.append(source_id)
        records_by_split[split].extend(rows)
        refit = resolve_under_root(
            project_root(),
            str(config["mc_data"]["source_root_template"]).format(source_id=source_id),
        )
        truth_tables[source_id] = _load_truth_table(
            refit / config["mc_data"]["enhanced_file"],
            config["mc_data"]["enhanced_tree"],
            config,
        )
    all_rows = records_by_split["construction"] + records_by_split["validation"]
    contract = audit_state_surface_contract(config, all_rows)
    jacobian = audit_jacobian_consistency(all_rows)
    events: list[dict[str, Any]] = []
    for split, rows in records_by_split.items():
        events.extend(
            collect_matched_events(
                config, records=rows, truth_tables=truth_tables, split=split
            )
        )
    residuals = decompose_residuals(events, config)
    tails = tail_provenance(events, config)
    classification = classify_failure(contract, jacobian, residuals, tails, config)
    return {
        "n_present_sources": len(present),
        "present_sources": present,
        "missing_sources": missing,
        "n_dump_rows": len(all_rows),
        "n_matched_events": len(events),
        "n_official_pair_events": int(residuals["n_official_pairs"]),
        "n_off_contract_pair_events": int(residuals["n_off_contract_pairs"]),
        "n_matched_dropped": 0,
        "state_surface_contract": contract,
        "jacobian_covariance_consistency": jacobian,
        "residual_decomposition": residuals,
        "tail_provenance": tails,
        "classification": classification,
        "dump_root": str(config["dump_root"]),
        "wb98_dumps_read_only": True,
    }


def jsonable(payload: Any) -> Any:
    """Replace NaN/Inf so the immutable artifact store can serialize."""
    if isinstance(payload, dict):
        return {str(key): jsonable(value) for key, value in payload.items()}
    if isinstance(payload, list):
        return [jsonable(item) for item in payload]
    if isinstance(payload, tuple):
        return [jsonable(item) for item in payload]
    if isinstance(payload, np.ndarray):
        return jsonable(payload.tolist())
    if isinstance(payload, (np.floating, float)):
        value = float(payload)
        return value if np.isfinite(value) else None
    if isinstance(payload, (np.integer,)):
        return int(payload)
    if isinstance(payload, (np.bool_,)):
        return bool(payload)
    return payload


def decide(inventory: Mapping[str, Any], inherited: Mapping[str, Any]) -> dict[str, Any]:
    classification = inventory["classification"]
    if bool(classification.get("measurement_model_v2_entered")):
        refuse_measurement_model_v2()
    return {
        "verdict": "DIAGNOSED",
        "decision": DECISION_DIAGNOSED,
        "primary_case": classification["primary_case"],
        "secondary_cases": classification["secondary_cases"],
        "next_step": classification["next_step"],
        "closure_pass": False,
        "closure_repaired": False,
        "measurement_model_v2_entered": False,
        "geometry_write_allowed": False,
        "unconstrained_tracker_only_stopped": True,
        "inherited_wb98_decision": inherited["workbook_98"]["decision"],
        "inherited_wb98_mechanism": inherited["workbook_98"]["mechanism"],
        "inherited_wb98_situation": inherited["workbook_98"]["situation"],
        "wb87_through_wb98_rewritten": False,
        "evidence": classification["evidence"],
        "case_flags": classification["case_flags"],
    }
