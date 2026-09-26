"""Yasu-S3A: fixed-measurement curvature/geometry response Jacobian.

Answers whether a reasonable track-curvature q/p error can produce an
IFT residual response similar to R_y, d_x, or t_y, without requiring a
trusted reconstructed momentum.  Does not use qp_bending_proxy.  Does
not weight by the miscalibrated native 5x5.  Mean-response only.

The 20-track focus list is frozen in the config before any Jacobian
number is inspected.  Association is truth-SDO on IFT clusters, never
4ST survival.  Sensor transforms come from runtime Calypso dumps.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import yaml

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
from datasets.three_st_qp_calibration import (
    SOURCE_COLLECTION_NAME,
    iter_jsonl,
    verify_pinned_calypso_sources,
)

SCHEMA_VERSION = "yasu-s3a-jacobian-closure-v1"
DEFAULT_CONFIG = "configs/yasu_s3a_jacobian_closure_v1.yaml"
TASK = "YASU-S3A"
WORKBOOK = 128

DECISION_CONTRACT = "yasu_s3a_jacobian_contract_established"
DECISION_RECORDED = "yasu_s3a_jacobian_recorded"
DECISION_NOT = "yasu_s3a_jacobian_not_established"

STATUS_PASS = "PASS"
STATUS_FAIL = "FAIL"
STATUS_ISOLATED = "isolated"
STATUS_UNRESOLVED = "unresolved"
STATUS_SUPPORTED = "supported"
STATUS_REJECTED = "rejected"

PARAM_QP = "q_over_p"
PARAM_RY = "R_y"
PARAM_DX = "d_x"
PARAM_TY = "t_y"
PARAM_ORDER = (PARAM_QP, PARAM_RY, PARAM_DX, PARAM_TY)

INTERVENTION_FIXED = "fixed_state"
INTERVENTION_PROFILED = "profiled_track"

# Dirty / missing-station identities are frozen coverage.  They must not
# fail the Jacobian stage when truth-SDO IFT association is empty.
COVERAGE_ONLY_ROLES = frozenset(
    {
        "construction_s2_front_dirty",
        "validation_s2_front_dirty",
    }
)
JACOBIAN_ELIGIBLE_ROLES = frozenset(
    {
        "construction_s1_front_control",
        "validation_s1_front_control",
        "frozen_wb119",
    }
)

FAILURE_SURFACE = "surface_identity_changed"
FAILURE_ASSOCIATION = "association_changed"
FAILURE_NAVIGATION = "navigation_path_changed"
FAILURE_FD = "finite_difference_not_closed"
FAILURE_LINEAR = "odd_even_linear_not_closed"
FAILURE_IDENTITY = "track_identity_incomplete"
FAILURE_PROPAGATE = "propagate_surface_failed"
CONTRACT_FAILURES = frozenset(
    {
        FAILURE_SURFACE,
        FAILURE_ASSOCIATION,
        FAILURE_NAVIGATION,
        FAILURE_FD,
        FAILURE_LINEAR,
        FAILURE_IDENTITY,
        FAILURE_PROPAGATE,
    }
)

# Numerical tolerances.  Not physical effect gates.
REL_CONVERGENCE = 1.0e-2
ABS_RESIDUAL_TOL_MM = 1.0e-3
LINEAR_REL_TOL = 1.0e-2
LINEAR_ABS_TOL_MM = 1.0e-3
NEAR_ZERO_RESPONSE_MM = 1.0e-3
TARGET_RESIDUAL_MM = (0.10, 1.00)
COLLINEAR_COSINE = 0.98

# Frozen FD amplitudes.  Chosen from SCT loc0 scale (~0.08 mm pitch),
# typical station alignment (0.1 mm, 1 mrad), and a curvature step that
# moves IFT loc0 by a few tenths of a millimetre at B~0.5 T, L~2 m.
FROZEN_DELTAS = {
    PARAM_QP: 1.0e-6,   # 1/MeV
    PARAM_RY: 1.0e-3,   # rad
    PARAM_DX: 0.10,     # mm
    PARAM_TY: 1.0e-4,   # slope
}
FD_RUNG_FACTORS = (1.0, 0.5, 0.25)
K_GEV_PER_TM = 0.299792458
MEV_PER_GEV = 1000.0
PIVOT_GLOBAL_MM = (0.0, 0.0, 0.0)
RY_MATRIX_CONVENTION = "stations_global_origin_TRzRyRx_active_left_multiply"

REQUIRED_INHERITANCE = (
    "workbook_117",
    "workbook_118",
    "workbook_119",
    "workbook_124",
    "workbook_126",
)

P1_BLOCKERS = (
    {
        "id": "P1-1",
        "title": "TrackTruthMatchingTool is plurality, not strict majority",
        "contaminates_s3a_focus": False,
        "action": "record; S3A IFT association uses explicit SDO barcode equality and reports plurality/majority/tie",
    },
    {
        "id": "P1-2",
        "title": "S1 truth momentum is not Geant local momentum on the same reference surface",
        "contaminates_s3a_focus": False,
        "action": "record; S3A does not use S1 truth momentum as a Jacobian input",
    },
    {
        "id": "P1-3",
        "title": "bound→curvilinear covariance / actual KF seed surface semantics are unclosed",
        "contaminates_s3a_focus": False,
        "action": "record; S3A is mean-response only and does not weight by native 5x5",
    },
    {
        "id": "P1-4",
        "title": "CircleFitTrackSeedTool static s_spacePointMap may retain stale pointers",
        "contaminates_s3a_focus": False,
        "action": "record; S3A does not reseeds and does not change the fitter",
    },
    {
        "id": "P1-5",
        "title": "measurement-bending space-point fallback / dedup is not strict",
        "contaminates_s3a_focus": False,
        "action": "record; S3A does not use bending centroids",
    },
    {
        "id": "P1-6",
        "title": "WB118 associated residual is conditioned on 4ST association",
        "contaminates_s3a_focus": True,
        "action": "isolate; S3A IFT association is truth-SDO, frozen, never 4ST",
    },
)


def jsonable(payload: Any) -> Any:
    """Replace ndarray / NaN / Inf so the immutable artifact store can serialize."""
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


class YasuS3AError(ValueError):
    """Raised when the S3A contract is illegal."""


def refuse_s2k_proxy() -> None:
    raise YasuS3AError("S3A must not use qp_bending_proxy")


def refuse_times_two_patch() -> None:
    raise YasuS3AError("S3A must not multiply the isolated S2K kernel by 2")


def refuse_free_scale() -> None:
    raise YasuS3AError("S3A must not fit a free scale from truth")


def refuse_native_covariance_weight() -> None:
    raise YasuS3AError("S3A must not weight the Jacobian by native 5x5")


def refuse_four_station_association() -> None:
    raise YasuS3AError("S3A IFT association must not use 4ST survival")


def refuse_handwritten_station_response() -> None:
    raise YasuS3AError("S3A must not use a handwritten station residual response")


def refuse_htcondor() -> None:
    raise YasuS3AError("S3A must not submit HTCondor")


def refuse_change_fitter() -> None:
    raise YasuS3AError("S3A must not change fitter, seed, hits, geometry payload, or covariance scale")


def refuse_b14m() -> None:
    raise YasuS3AError("S3A must not enter B14M / B15 / MM V2")


def refuse_held_out() -> None:
    raise YasuS3AError("S3A must not touch held-out or sealed data")


def refuse_weak_mode_claim() -> None:
    raise YasuS3AError(
        "S3A must not claim a natural-data q/p-R_y weak mode"
    )


def refuse_replace_focus() -> None:
    raise YasuS3AError("S3A must not replace the focus list after seeing Jacobian numbers")


def refuse_s2k_batch() -> None:
    raise YasuS3AError("S2K batch is paused; qp_bending_proxy is isolated after P0")


def access_split(split: str) -> str:
    return without_ift_access_split(split)


def residual_definition() -> dict[str, Any]:
    return {
        "kind": "truth_associated_ift_sensor_local_loc0",
        "measurement": "FaserSCT_Cluster.localPosition Trk::locX on the runtime SiDetectorElement",
        "prediction": "Acts BoundTrackParameters loc0 on the same IFT wafer surface",
        "formula": "r = loc0_cluster - loc0_predicted",
        "unit": "mm",
        "frame": "actual_sensor_local_measurement_axis",
        "not_yz_bending_geometry": True,
        "inside_bounds_required_for_residual": False,
        "association": "truth_sdo_barcode_on_ift_cluster",
        "four_station_association_forbidden": True,
    }


def state_definition() -> dict[str, Any]:
    return {
        "native_athena": ["loc1_mm", "loc2_mm", "phi", "theta", "q_over_p_per_mev"],
        "export_parameters": ["x_mm", "y_mm", "tx", "ty", "q_over_p_per_mev"],
        "export_frame": "global_cartesian_slopes_at_source_surface",
        "q_over_p_native_unit": "per_MeV",
        "mean_response_only": True,
        "native_5x5_not_a_jacobian_weight": True,
        "perturbation_ty_is_export_ty": True,
        "perturbation_qp_is_native_q_over_p": True,
    }


def geometry_response_definition() -> dict[str, Any]:
    return {
        "pivot_mm": list(PIVOT_GLOBAL_MM),
        "composition": "G = T * Rz * Ry * Rx, active left-multiply",
        "convention": RY_MATRIX_CONVENTION,
        "R_y_matrix": "[[c,0,s],[0,1,0],[-s,0,c]] with c=cos(Ry), s=sin(Ry)",
        "d_x": "global translation of the IFT station, millimetres",
        "applied_to": "runtime sensor transform read from Calypso, then left-composed",
        "handwritten_station_response": False,
        "payload_write": False,
    }


def identity_contract() -> dict[str, Any]:
    return {
        "keys": [
            "file_sha256",
            "source_id",
            "run_id",
            "event_id",
            "skip_index",
            "collection",
            "track_index",
            "measurement_digest",
        ],
        "file_sha256_is_content_hash_of_input_xaod": True,
        "event_guid_recorded_when_available": True,
        "entry_is_athena_skip_index": True,
        "collection": SOURCE_COLLECTION_NAME,
        "measurement_digest": "sha256 of sorted MOT compact identifiers plus frozen IFT association identifiers",
    }


def association_contract() -> dict[str, Any]:
    return {
        "method": "truth_sdo_barcode",
        "ift_source": "SCT_ClusterContainer station==0",
        "truth_source": "SCT_SDO_Map deposits on the IFT cluster RDO list",
        "match": "any deposit barcode equals the track plurality barcode",
        "plurality_is_not_majority": True,
        "majority_and_tie_flags_recorded": True,
        "four_station_forbidden": True,
        "frozen_before_jacobian": True,
        "ambiguous_and_unmatched_kept": True,
    }


def ry_matrix(angle_rad: float) -> np.ndarray:
    cosine = math.cos(float(angle_rad))
    sine = math.sin(float(angle_rad))
    return np.array(
        [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]],
        dtype=np.float64,
    )


def dx_matrix(dx_mm: float) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[0, 3] = float(dx_mm)
    return transform


def se3_ry_dx(*, ry_rad: float = 0.0, dx_mm: float = 0.0) -> np.ndarray:
    """Active left-multiply increment G = T(dx) * Ry about the FASER origin."""
    rotation = np.eye(4, dtype=np.float64)
    rotation[:3, :3] = ry_matrix(ry_rad)
    return dx_matrix(dx_mm) @ rotation


def apply_station_increment(
    center_mm: Sequence[float],
    axes: Mapping[str, Sequence[float]],
    *,
    ry_rad: float = 0.0,
    dx_mm: float = 0.0,
) -> dict[str, np.ndarray]:
    increment = se3_ry_dx(ry_rad=ry_rad, dx_mm=dx_mm)
    center = np.array([*center_mm, 1.0], dtype=np.float64)
    new_center = increment @ center
    rotation = increment[:3, :3]
    return {
        "center_mm": new_center[:3],
        "u": rotation @ np.asarray(axes["u"], dtype=np.float64),
        "v": rotation @ np.asarray(axes["v"], dtype=np.float64),
        "n": rotation @ np.asarray(axes["n"], dtype=np.float64),
    }


def line_plane_intersection(
    origin_mm: Sequence[float],
    direction: Sequence[float],
    *,
    center_mm: Sequence[float],
    u: Sequence[float],
    v: Sequence[float],
    n: Sequence[float],
) -> dict[str, Any]:
    origin = np.asarray(origin_mm, dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64)
    center = np.asarray(center_mm, dtype=np.float64)
    axis_u = np.asarray(u, dtype=np.float64)
    axis_n = np.asarray(n, dtype=np.float64)
    denom = float(direction @ axis_n)
    if abs(denom) < 1.0e-18:
        return {"success": False, "failure_class": "parallel_to_plane"}
    path = float((center - origin) @ axis_n) / denom
    point = origin + path * direction
    loc0 = float((point - center) @ axis_u)
    loc1 = float((point - center) @ np.asarray(v, dtype=np.float64))
    return {
        "success": True,
        "failure_class": None,
        "point_mm": point,
        "loc0_mm": loc0,
        "loc1_mm": loc1,
        "path": path,
    }


def zero_field_analytic_residual(
    origin_mm: Sequence[float],
    direction: Sequence[float],
    cluster_global_mm: Sequence[float],
    surface: Mapping[str, Sequence[float]],
    *,
    ry_rad: float = 0.0,
    dx_mm: float = 0.0,
) -> dict[str, Any]:
    perturbed = apply_station_increment(
        surface["center_mm"],
        {"u": surface["u"], "v": surface["v"], "n": surface["n"]},
        ry_rad=ry_rad,
        dx_mm=dx_mm,
    )
    predicted = line_plane_intersection(
        origin_mm,
        direction,
        center_mm=perturbed["center_mm"],
        u=perturbed["u"],
        v=perturbed["v"],
        n=perturbed["n"],
    )
    if not predicted["success"]:
        return predicted
    cluster = np.asarray(cluster_global_mm, dtype=np.float64)
    # loc0_cluster is the frozen local measurement.  Geometry increments
    # move the prediction surface, not the recorded cluster loc0.
    loc0_cluster = float(
        (cluster - np.asarray(surface["center_mm"], dtype=np.float64))
        @ np.asarray(surface["u"], dtype=np.float64)
    )
    return {
        "success": True,
        "failure_class": None,
        "loc0_cluster_mm": loc0_cluster,
        "loc0_predicted_mm": predicted["loc0_mm"],
        "residual_loc0_mm": loc0_cluster - predicted["loc0_mm"],
        "predicted_point_mm": predicted["point_mm"],
    }


def circular_orbit_yz(
    *,
    z0_mm: float,
    y0_mm: float,
    ty0: float,
    q_over_p_per_mev: float,
    bx_tesla: float,
    z_mm: float,
) -> dict[str, float]:
    """Uniform Bx circle in the YZ plane.  Small-tx analytic control."""
    q_over_p_per_gev = float(q_over_p_per_mev) * MEV_PER_GEV
    kappa = K_GEV_PER_TM * q_over_p_per_gev * float(bx_tesla)
    alpha0 = math.atan(float(ty0))
    delta_z_m = (float(z_mm) - float(z0_mm)) * 1.0e-3
    alpha = alpha0 + kappa * delta_z_m
    if abs(kappa) < 1.0e-18:
        y_mm = float(y0_mm) + float(ty0) * (float(z_mm) - float(z0_mm))
        return {"y_mm": y_mm, "ty": float(ty0), "z_mm": float(z_mm)}
    y_m = float(y0_mm) * 1.0e-3 + (math.sin(alpha) - math.sin(alpha0)) / kappa
    return {"y_mm": y_m * 1.0e3, "ty": math.tan(alpha), "z_mm": float(z_mm)}


def measurement_digest(identifiers: Sequence[str]) -> str:
    payload = "\n".join(sorted(str(item) for item in identifiers))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def track_identity_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        str(row.get("file_sha256") or ""),
        str(row.get("source_id") or ""),
        int(row.get("skip_index") if row.get("skip_index") is not None else row.get("event_id") or -1),
        str(row.get("collection") or SOURCE_COLLECTION_NAME),
        int(row.get("track_index") if row.get("track_index") is not None else 0),
        str(row.get("measurement_digest") or ""),
    )


def require_identity(row: Mapping[str, Any]) -> None:
    missing = [
        key
        for key in (
            "file_sha256",
            "source_id",
            "run_id",
            "event_id",
            "skip_index",
            "collection",
            "track_index",
            "measurement_digest",
        )
        if row.get(key) in (None, "")
    ]
    if missing:
        raise YasuS3AError(f"track identity incomplete: {missing}")
    if str(row.get("collection")) != SOURCE_COLLECTION_NAME:
        raise YasuS3AError("collection must be CKFTrackCollectionWithoutIFT")


def frozen_deltas(config: Mapping[str, Any] | None = None) -> dict[str, float]:
    configured = ((config or {}).get("finite_difference") or {}).get("deltas") or {}
    out = dict(FROZEN_DELTAS)
    for name, value in configured.items():
        if name not in out:
            raise YasuS3AError(f"unknown FD parameter {name}")
        if abs(float(value) - float(out[name])) > 0.0:
            raise YasuS3AError(f"FD amplitude for {name} is frozen and must not change")
    return out


def fd_rungs(delta: float) -> tuple[float, ...]:
    return tuple(float(delta) * float(factor) for factor in FD_RUNG_FACTORS)


def central_derivative(plus: float, minus: float, step: float) -> float:
    return (float(plus) - float(minus)) / (2.0 * float(step))


def relative_change(new: float, old: float, *, abs_floor: float) -> float:
    scale = max(abs(float(new)), abs(float(old)), float(abs_floor))
    return abs(float(new) - float(old)) / scale


def cosine_similarity(left, right) -> float | None:
    a = np.asarray(left, dtype=np.float64).reshape(-1)
    b = np.asarray(right, dtype=np.float64).reshape(-1)
    if a.size == 0 or b.size == 0 or a.size != b.size:
        return None
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1.0e-18 or nb < 1.0e-18:
        return None
    return float(np.dot(a, b) / (na * nb))


def layer_pattern(residuals: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_layer: dict[str, list[float]] = {}
    for row in residuals:
        if not bool(row.get("residual_available")):
            continue
        key = f"layer{int(row['layer'])}_side{int(row['side'])}"
        by_layer.setdefault(key, []).append(float(row["residual_loc0_mm"]))
    return {
        name: {
            "n": len(values),
            "mean_mm": float(np.mean(values)),
            "rms_mm": float(np.sqrt(np.mean(np.square(values)))),
        }
        for name, values in sorted(by_layer.items())
    }


def vector_from_residuals(
    residuals: Sequence[Mapping[str, Any]],
    *,
    association_digest: str | None = None,
) -> np.ndarray:
    values = []
    digest_parts = []
    for row in residuals:
        digest_parts.append(str(row.get("identifier")))
        if not bool(row.get("residual_available")):
            continue
        values.append(float(row["residual_loc0_mm"]))
    if association_digest is not None:
        got = measurement_digest(digest_parts)
        if got != association_digest:
            raise YasuS3AError("association digest changed during Jacobian evaluation")
    return np.asarray(values, dtype=np.float64)


def rung_arrays(
    payload: Mapping[str, Any],
    name: str,
    *,
    deltas: Mapping[str, float],
) -> tuple[list[np.ndarray], list[np.ndarray], tuple[float, ...]]:
    """Return plus/minus arrays ordered as δ, δ/2, δ/4."""
    steps = fd_rungs(float(deltas[name]))
    block = payload[name]
    plus: list[np.ndarray] = []
    minus: list[np.ndarray] = []
    if "plus" in block and "minus" in block:
        plus_list = block["plus"]
        minus_list = block["minus"]
        if len(plus_list) != len(steps) or len(minus_list) != len(steps):
            raise YasuS3AError(f"{name} rungs must have {len(steps)} entries")
        plus = [np.asarray(item, dtype=np.float64) for item in plus_list]
        minus = [np.asarray(item, dtype=np.float64) for item in minus_list]
        return plus, minus, steps
    for step in steps:
        plus.append(np.asarray(block[f"+{step}"], dtype=np.float64))
        minus.append(np.asarray(block[f"-{step}"], dtype=np.float64))
    return plus, minus, steps


def jacobian_from_rungs(
    nominal: np.ndarray,
    rungs: Mapping[str, Mapping[str, Any]],
    *,
    deltas: Mapping[str, float],
    abs_tol_mm: float = ABS_RESIDUAL_TOL_MM,
    rel_tol: float = REL_CONVERGENCE,
) -> dict[str, Any]:
    """Build mean-response columns from ±δ, ±δ/2, ±δ/4 rungs."""
    columns: dict[str, Any] = {}
    stop = False
    failures: list[str] = []
    names = [name for name in PARAM_ORDER if name in rungs]
    for name in names:
        plus_list, minus_list, steps = rung_arrays(rungs, name, deltas=deltas)
        derivatives = []
        odd_even = []
        for plus, minus, step in zip(plus_list, minus_list, steps):
            if plus.shape != nominal.shape or minus.shape != nominal.shape:
                stop = True
                failures.append(FAILURE_NAVIGATION)
                break
            derivatives.append((plus - minus) / (2.0 * step))
            odd_even.append(0.5 * (plus + minus) - nominal)
        if stop:
            break
        last = derivatives[-1]
        prev = derivatives[-2]
        rel = np.array(
            [
                relative_change(a, b, abs_floor=abs_tol_mm / max(abs(steps[-1]), 1.0e-18))
                for a, b in zip(last, prev)
            ],
            dtype=np.float64,
        )
        even_last = odd_even[-1]
        linear_scale = np.maximum(
            LINEAR_ABS_TOL_MM,
            LINEAR_REL_TOL * np.abs(last) * float(steps[-1]),
        )
        linear_ok = bool(np.all(np.abs(even_last) <= linear_scale))
        conv_ok = bool(
            np.all(rel <= rel_tol)
            or np.all(np.abs(last) * float(steps[-1]) <= abs_tol_mm)
        )
        if not conv_ok:
            stop = True
            failures.append(FAILURE_FD)
        if not linear_ok:
            stop = True
            failures.append(FAILURE_LINEAR)
        columns[name] = {
            "derivative": last.tolist(),
            "derivative_rungs": [item.tolist() for item in derivatives],
            "steps": list(steps),
            "relative_change_last_two": rel.tolist(),
            "even_residual_last_rung_mm": even_last.tolist(),
            "convergence_ok": conv_ok,
            "linear_ok": linear_ok,
            "near_zero": bool(np.max(np.abs(last) * steps[-1]) <= NEAR_ZERO_RESPONSE_MM),
            "rms_mm_per_unit": float(np.sqrt(np.mean(np.square(last)))) if last.size else 0.0,
            "sign_pattern": [int(np.sign(v)) for v in last],
        }
    return {
        "columns": columns,
        "stop_physical_interpretation": stop,
        "failure_classes": failures,
        "parameter_order": list(names),
    }


def pairwise_similarity(columns: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    names = [name for name in PARAM_ORDER if name in columns]
    table: dict[str, Any] = {}
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            cosine = cosine_similarity(columns[left]["derivative"], columns[right]["derivative"])
            table[f"{left}__{right}"] = {
                "cosine": cosine,
                "near_collinear": bool(cosine is not None and abs(float(cosine)) >= COLLINEAR_COSINE),
                "high_correlation_is_not_a_weak_mode": True,
            }
    return table


def qp_needed_for_residual(
    qp_column: Mapping[str, Any] | None,
    *,
    targets_mm: Sequence[float] = TARGET_RESIDUAL_MM,
) -> dict[str, Any]:
    if qp_column is None:
        return {"available": False}
    rms = float(qp_column.get("rms_mm_per_unit") or 0.0)
    out: dict[str, Any] = {"available": rms > 0.0, "rms_mm_per_mev_inv": rms}
    for target in targets_mm:
        key = f"delta_qp_for_{target:g}mm"
        out[key] = None if rms <= 0.0 else float(target) / rms
    return out


def collinearity_verdict(similarity: Mapping[str, Any], qp_needed: Mapping[str, Any]) -> dict[str, Any]:
    qp_pairs = {
        name: row
        for name, row in similarity.items()
        if name.startswith(f"{PARAM_QP}__")
    }
    near = [name for name, row in qp_pairs.items() if row.get("near_collinear")]
    amplitude_ok = bool(qp_needed.get("available"))
    return {
        "qp_near_collinear_with": near,
        "high_correlation_is_not_a_weak_mode": True,
        "curvature_can_mimic_geometry_direction": bool(near) and amplitude_ok,
        "authorize_profiled_schur_next": False,
        "legal_claim_only": (
            "under a frozen and verified measurement/surface contract, "
            "a curvature error can/cannot produce a given IFT residual "
            "response and is/is not similar to listed geometry/slope responses"
        ),
    }


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = resolve_under_root(project_root(), str(path or DEFAULT_CONFIG))
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != SCHEMA_VERSION:
        raise YasuS3AError(f"schema_version must be {SCHEMA_VERSION}")
    if config.get("task") != TASK:
        raise YasuS3AError(f"task must be {TASK}")
    if int(config.get("workbook", -1)) != WORKBOOK:
        raise YasuS3AError("workbook must be 128")
    if str(config.get("source_collection")) != SOURCE_COLLECTION:
        raise YasuS3AError("source_collection must be CKFTrackCollectionWithoutIFT")
    for key in (
        "geometry_write_allowed",
        "held_out_accessed",
        "real_data_alignment_authorized",
        "residual_conditional_authorized",
        "b14m_reopen_authorized",
        "b15_authorized",
        "three_st_qp_trusted_observable",
        "use_qp_bending_proxy",
        "use_native_5x5_jacobian_weight",
        "four_station_association_authorized",
        "s2k_batch_authorized",
        "times_two_patch_authorized",
        "fit_free_scale_from_truth",
        "htcondor_authorized",
        "weak_mode_claim_authorized",
    ):
        if bool(config.get(key, True)):
            raise YasuS3AError(f"{key} must be false")
    if bool(config.get("do_not_replace_focus_after_seeing_results")) is not True:
        raise YasuS3AError("focus list is frozen")
    identities = config.get("focus_identities") or []
    if len(identities) != 20:
        raise YasuS3AError("focus list must contain exactly 20 tracks")
    keys = {
        (row["source_id"], int(row["skip_index"]), int(row["track_index"]))
        for row in identities
    }
    if len(keys) != 20:
        raise YasuS3AError("focus identities must be unique")
    frozen_deltas(config)
    if list(config.get("finite_difference", {}).get("rung_factors") or []) != list(FD_RUNG_FACTORS):
        raise YasuS3AError("FD rung factors are frozen at ±δ, ±δ/2, ±δ/4")
    return dict(config)


def _expect_sha(path: Path, expected: str, label: str) -> None:
    digest = sha256_file(path)
    if digest != expected:
        raise YasuS3AError(f"{label} hash mismatch: {digest}")


def inherit_frozen_stage(config: Mapping[str, Any]) -> dict[str, Any]:
    inherited: dict[str, Any] = {}
    frozen_root = Path(str(config.get("frozen_artifact_root") or project_root()))
    for workbook in REQUIRED_INHERITANCE:
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
            raise YasuS3AError(f"{workbook} decision must stay frozen")
        inherited[workbook] = {
            "decision": decision["decision"],
            "config_sha256": spec["config_sha256"],
            "decision_sha256": spec["decision_sha256"],
        }
    inherited["s2k_isolated"] = True
    inherited["qp_bending_proxy_isolated"] = True
    inherited["pinned_calypso_sources"] = verify_pinned_calypso_sources(config)
    return inherited


def focus_manifest(config: Mapping[str, Any]) -> dict[str, Any]:
    rows = []
    for item in config["focus_identities"]:
        rows.append(
            {
                "source_id": item["source_id"],
                "run_id": int(item["run_id"]),
                "event_id": int(item["event_id"]),
                "skip_index": int(item["skip_index"]),
                "track_index": int(item["track_index"]),
                "collection": SOURCE_COLLECTION_NAME,
                "role": item["role"],
                "reason": item["reason"],
                "file_sha256": item["file_sha256"],
                "input_xaod": item["input_xaod"],
            }
        )
    return {
        "kind": "yasu_s3a_focus_manifest",
        "n": len(rows),
        "frozen_before_jacobian": True,
        "replaced_after_results": False,
        "identities": rows,
        "coverage": {
            "complete_three_st": sum(1 for row in rows if "s1_front_control" in row["role"]),
            "wb119_sign_flip": sum(1 for row in rows if row["reason"] == "sign_flip"),
            "wb119_large_pull": sum(1 for row in rows if row["reason"] == "large_pull"),
            "s2_front_dirty": sum(1 for row in rows if "s2_front_dirty" in row["role"]),
        },
    }


def _rung_pair(
    *,
    origin: Sequence[float],
    direction: Sequence[float],
    cluster: Sequence[float],
    surface: Mapping[str, Sequence[float]],
    name: str,
    amplitude: float,
) -> dict[str, list[np.ndarray]]:
    plus: list[np.ndarray] = []
    minus: list[np.ndarray] = []
    for factor in FD_RUNG_FACTORS:
        step = float(amplitude) * float(factor)
        for sign, bucket in ((1.0, plus), (-1.0, minus)):
            ry = sign * step if name == PARAM_RY else 0.0
            dx = sign * step if name == PARAM_DX else 0.0
            ty = sign * step if name == PARAM_TY else 0.0
            direction_p = (direction[0], direction[1] + ty, direction[2])
            payload = zero_field_analytic_residual(
                origin, direction_p, cluster, surface, ry_rad=ry, dx_mm=dx
            )
            bucket.append(np.array([payload["residual_loc0_mm"]]))
    return {"plus": plus, "minus": minus}


def analytic_zero_field_control() -> dict[str, Any]:
    """Straight line onto a rotated/translated plane.  Closed in loc0."""
    origin = (0.0, 0.0, 50.0)
    direction = (0.002, 0.001, 1.0)
    surface = {
        "center_mm": np.array([10.0, -4.0, -1860.15]),
        "u": np.array([1.0, 0.0, 0.0]),
        "v": np.array([0.0, 1.0, 0.0]),
        "n": np.array([0.0, 0.0, 1.0]),
    }
    cluster = np.array([10.4, -3.7, -1860.15])
    nominal = zero_field_analytic_residual(origin, direction, cluster, surface)
    deltas = frozen_deltas()
    nominal_vec = np.array([nominal["residual_loc0_mm"]])
    columns = {}
    stop = False
    for name in (PARAM_RY, PARAM_DX, PARAM_TY):
        rungs = _rung_pair(
            origin=origin,
            direction=direction,
            cluster=cluster,
            surface=surface,
            name=name,
            amplitude=deltas[name],
        )
        report = jacobian_from_rungs(nominal_vec, {name: rungs}, deltas={name: deltas[name]})
        columns[name] = report["columns"][name]
        columns[name]["stop"] = report["stop_physical_interpretation"]
        stop = stop or report["stop_physical_interpretation"]
    qp_plus = [nominal_vec.copy() for _ in FD_RUNG_FACTORS]
    qp_minus = [nominal_vec.copy() for _ in FD_RUNG_FACTORS]
    qp_report = jacobian_from_rungs(
        nominal_vec,
        {PARAM_QP: {"plus": qp_plus, "minus": qp_minus}},
        deltas={PARAM_QP: FROZEN_DELTAS[PARAM_QP]},
    )
    return {
        "kind": "zero_field_analytic_line_plane",
        "nominal_residual_mm": nominal["residual_loc0_mm"],
        "q_over_p_column_near_zero": qp_report["columns"][PARAM_QP]["near_zero"],
        "q_over_p_convergence_ok": qp_report["columns"][PARAM_QP]["convergence_ok"],
        "geometry_columns": {
            name: {
                "convergence_ok": columns[name]["convergence_ok"],
                "linear_ok": columns[name]["linear_ok"],
                "rms_mm_per_unit": columns[name]["rms_mm_per_unit"],
                "near_zero": columns[name]["near_zero"],
            }
            for name in (PARAM_RY, PARAM_DX, PARAM_TY)
        },
        "stop_physical_interpretation": stop or qp_report["stop_physical_interpretation"],
    }


def analytic_uniform_field_circle_control() -> dict[str, Any]:
    """Uniform Bx circle: q/p and ty move y at a downstream plane."""
    z0, z1 = 47.4, -1860.15
    bx = -0.55
    qp0 = 5.0e-6
    ty0 = 0.0
    y0 = 0.0
    plane_y = circular_orbit_yz(
        z0_mm=z0, y0_mm=y0, ty0=ty0, q_over_p_per_mev=qp0, bx_tesla=bx, z_mm=z1
    )["y_mm"]
    cluster_y = plane_y + 0.05
    deltas = frozen_deltas()
    nominal = np.array([cluster_y - plane_y])
    rungs: dict[str, dict[str, list[np.ndarray]]] = {
        PARAM_QP: {"plus": [], "minus": []},
        PARAM_TY: {"plus": [], "minus": []},
    }
    for factor in FD_RUNG_FACTORS:
        dqp = deltas[PARAM_QP] * factor
        dty = deltas[PARAM_TY] * factor
        for sign, side in ((1.0, "plus"), (-1.0, "minus")):
            y_qp = circular_orbit_yz(
                z0_mm=z0,
                y0_mm=y0,
                ty0=ty0,
                q_over_p_per_mev=qp0 + sign * dqp,
                bx_tesla=bx,
                z_mm=z1,
            )["y_mm"]
            y_ty = circular_orbit_yz(
                z0_mm=z0,
                y0_mm=y0,
                ty0=ty0 + sign * dty,
                q_over_p_per_mev=qp0,
                bx_tesla=bx,
                z_mm=z1,
            )["y_mm"]
            rungs[PARAM_QP][side].append(np.array([cluster_y - y_qp]))
            rungs[PARAM_TY][side].append(np.array([cluster_y - y_ty]))
    report = jacobian_from_rungs(
        nominal,
        rungs,
        deltas={PARAM_QP: deltas[PARAM_QP], PARAM_TY: deltas[PARAM_TY]},
    )
    # Independent truth: d y / d(q/p) from kappa scaling.
    kappa0 = K_GEV_PER_TM * (qp0 * MEV_PER_GEV) * bx
    dz_m = (z1 - z0) * 1.0e-3
    # y_m = (sin(kappa dz)-0)/kappa ; dy/d(q/p_per_mev)
    analytic = []
    for qp in (qp0 + deltas[PARAM_QP], qp0 - deltas[PARAM_QP]):
        analytic.append(
            circular_orbit_yz(
                z0_mm=z0, y0_mm=y0, ty0=ty0, q_over_p_per_mev=qp, bx_tesla=bx, z_mm=z1
            )["y_mm"]
        )
    truth_dydqp = (analytic[0] - analytic[1]) / (2.0 * deltas[PARAM_QP])
    fd_dydqp = -float(report["columns"][PARAM_QP]["derivative"][0])
    return {
        "kind": "uniform_field_analytic_circle",
        "bx_tesla": bx,
        "convergence_ok": report["columns"][PARAM_QP]["convergence_ok"]
        and report["columns"][PARAM_TY]["convergence_ok"],
        "linear_ok": report["columns"][PARAM_QP]["linear_ok"]
        and report["columns"][PARAM_TY]["linear_ok"],
        "fd_dy_dqp_mm_per_mev_inv": fd_dydqp,
        "truth_dy_dqp_mm_per_mev_inv": truth_dydqp,
        "relative_fd_vs_truth": relative_change(fd_dydqp, truth_dydqp, abs_floor=1.0e-9),
        "stop_physical_interpretation": report["stop_physical_interpretation"],
        "kappa0": kappa0,
        "dz_m": dz_m,
    }


def evaluate_controls() -> dict[str, Any]:
    zero = analytic_zero_field_control()
    circle = analytic_uniform_field_circle_control()
    ok = (
        not zero["stop_physical_interpretation"]
        and not circle["stop_physical_interpretation"]
        and bool(zero["q_over_p_column_near_zero"])
        and float(circle["relative_fd_vs_truth"]) <= REL_CONVERGENCE
    )
    return {
        "zero_field_line_plane": zero,
        "uniform_field_circle": circle,
        "energy_loss_control": "runtime_dump_required",
        "fixed_surface_sequence": "runtime_dump_required",
        "analytic_controls_closed": ok,
        "stop_physical_interpretation": not ok,
    }


def coverage_only_role(role: str | None) -> bool:
    return str(role or "") in COVERAGE_ONLY_ROLES


def jacobian_eligible_role(role: str | None) -> bool:
    return str(role or "") in JACOBIAN_ELIGIBLE_ROLES


def evaluate_track_jacobian(track: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    require_identity(track)
    if bool(track.get("four_station_association_used")):
        refuse_four_station_association()
    if bool(track.get("used_qp_bending_proxy")):
        refuse_s2k_proxy()
    if bool(track.get("native_5x5_weighted")):
        refuse_native_covariance_weight()
    residuals = track.get("nominal_residuals") or []
    digest = str(track.get("association_digest") or track.get("measurement_digest"))
    role = track.get("role")
    coverage_only = coverage_only_role(role)
    eligible = jacobian_eligible_role(role) and not coverage_only
    identity = {key: track.get(key) for key in (
        "file_sha256", "source_id", "skip_index", "track_index", "measurement_digest"
    )}
    identity["role"] = role
    association = track.get("ift_association") or {}
    if str(association.get("method")) != "truth_sdo_barcode":
        return {
            "identity": identity,
            "association": association,
            "failure_class": FAILURE_ASSOCIATION,
            "failure_classes": [FAILURE_ASSOCIATION],
            "stop_physical_interpretation": True,
            "coverage_only": coverage_only,
            "jacobian_eligible": False,
            "n_residuals": 0,
            "mean_response_only": True,
            "native_5x5_weighted": False,
            "used_qp_bending_proxy": False,
        }
    nominal = vector_from_residuals(residuals, association_digest=None)
    n_available = int(sum(1 for row in residuals if bool(row.get("residual_available"))))
    if not residuals or n_available == 0:
        failure = str(
            track.get("primary_failure_class")
            or (
                "propagate_surface_failed"
                if residuals
                else "independent_ift_measurement_absent"
            )
        )
        return {
            "identity": identity,
            "association": association,
            "failure_class": failure,
            "failure_classes": [failure],
            "stop_physical_interpretation": True,
            "coverage_only": coverage_only,
            "jacobian_eligible": False,
            "n_residuals": n_available,
            "n_associated_ift": len(residuals),
            "mean_response_only": True,
            "native_5x5_weighted": False,
            "used_qp_bending_proxy": False,
        }
    rungs_payload = track.get("fixed_state_rungs") or {}
    packed: dict[str, dict[str, list[np.ndarray]]] = {}
    deltas = frozen_deltas(config)
    surfaces = [row.get("surface_geo_id") for row in residuals]
    nom_ids = [row.get("identifier") for row in residuals]

    def _pack(block: Mapping[str, Any], label: str) -> dict[str, dict[str, list[np.ndarray]]] | dict[str, Any]:
        packed_local: dict[str, dict[str, list[np.ndarray]]] = {}
        for name in PARAM_ORDER:
            entry = block.get(name) or {}
            plus_rows = entry.get("plus")
            minus_rows = entry.get("minus")
            if plus_rows is None or minus_rows is None or len(plus_rows) != 3 or len(minus_rows) != 3:
                return {
                    "identity": {k: track.get(k) for k in ("source_id", "skip_index", "track_index")},
                    "failure_class": FAILURE_FD,
                    "stop_physical_interpretation": True,
                    "missing_rung": f"{label}:{name}",
                }
            packed_local[name] = {"plus": [], "minus": []}
            for side, rows_list in (("plus", plus_rows), ("minus", minus_rows)):
                for rows in rows_list:
                    if [row.get("surface_geo_id") for row in rows] != surfaces:
                        return {"failure_class": FAILURE_SURFACE, "stop_physical_interpretation": True}
                    if [row.get("identifier") for row in rows] != nom_ids:
                        return {"failure_class": FAILURE_ASSOCIATION, "stop_physical_interpretation": True}
                    packed_local[name][side].append(vector_from_residuals(rows))
        return packed_local

    packed_or_fail = _pack(rungs_payload, INTERVENTION_FIXED)
    if "failure_class" in packed_or_fail:
        packed_or_fail = dict(packed_or_fail)
        packed_or_fail.setdefault("identity", identity)
        packed_or_fail["coverage_only"] = coverage_only
        packed_or_fail["jacobian_eligible"] = bool(eligible)
        packed_or_fail["association"] = association
        packed_or_fail.setdefault("failure_classes", [packed_or_fail["failure_class"]])
        return packed_or_fail
    packed = packed_or_fail  # type: ignore[assignment]
    report = jacobian_from_rungs(nominal, packed, deltas=deltas)
    similarity = pairwise_similarity(report["columns"])
    qp_needed = qp_needed_for_residual(report["columns"].get(PARAM_QP))
    profiled_report = None
    if track.get("profiled_rungs"):
        packed_p = _pack(track["profiled_rungs"], INTERVENTION_PROFILED)
        if "failure_class" not in packed_p:
            profiled_report = jacobian_from_rungs(nominal, packed_p, deltas=deltas)
    stop = bool(report["stop_physical_interpretation"])
    return {
        "identity": {
            "file_sha256": track.get("file_sha256"),
            "source_id": track.get("source_id"),
            "run_id": track.get("run_id"),
            "event_id": track.get("event_id"),
            "skip_index": track.get("skip_index"),
            "track_index": track.get("track_index"),
            "measurement_digest": track.get("measurement_digest"),
            "association_digest": digest,
            "role": role,
        },
        "association": association,
        "n_residuals": int(nominal.size),
        "layer_pattern": layer_pattern(residuals),
        "fixed_state": report,
        "profiled_track": profiled_report,
        "pairwise_similarity": similarity,
        "qp_needed_for_target_residual": qp_needed,
        "collinearity": collinearity_verdict(similarity, qp_needed),
        "eloss_on_off_max_abs_delta_mm": track.get("eloss_on_off_max_abs_delta_mm"),
        "stop_physical_interpretation": stop,
        "failure_classes": report["failure_classes"],
        "coverage_only": coverage_only,
        "jacobian_eligible": bool(eligible) and not stop,
        "mean_response_only": True,
        "native_5x5_weighted": False,
        "used_qp_bending_proxy": False,
    }


def summarize_tracks(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    n = len(results)
    n_coverage = sum(1 for row in results if row.get("coverage_only"))
    eligible_rows = [
        row
        for row in results
        if jacobian_eligible_role((row.get("identity") or {}).get("role"))
        and not row.get("coverage_only")
    ]
    n_eligible = len(eligible_rows)
    n_eligible_stop = sum(1 for row in eligible_rows if row.get("stop_physical_interpretation"))
    n_eligible_closed = n_eligible - n_eligible_stop
    n_stop = sum(1 for row in results if row.get("stop_physical_interpretation"))
    n_closed = n - n_stop
    contract_breaks = []
    for row in eligible_rows:
        classes = list(row.get("failure_classes") or [])
        if row.get("failure_class"):
            classes.append(str(row["failure_class"]))
        for item in classes:
            if item in CONTRACT_FAILURES:
                contract_breaks.append(item)
    collinear = []
    qp_needed = []
    for row in eligible_rows:
        if row.get("stop_physical_interpretation"):
            continue
        collinear.extend(row.get("collinearity", {}).get("qp_near_collinear_with") or [])
        needed = row.get("qp_needed_for_target_residual") or {}
        if needed.get("delta_qp_for_0.1mm") is not None:
            qp_needed.append(float(needed["delta_qp_for_0.1mm"]))
    authorize_next = (
        n_eligible_closed > 0
        and not contract_breaks
        and bool(collinear)
    )
    return {
        "n_tracks": n,
        "n_closed": n_closed,
        "n_stop_physical_interpretation": n_stop,
        "n_coverage_only": n_coverage,
        "n_jacobian_eligible": n_eligible,
        "n_jacobian_eligible_closed": n_eligible_closed,
        "n_jacobian_eligible_stop": n_eligible_stop,
        "eligible_contract_breaks": contract_breaks,
        "qp_near_collinear_counts": {name: collinear.count(name) for name in sorted(set(collinear))},
        "median_abs_delta_qp_for_0.1mm": (
            float(np.median(np.abs(qp_needed))) if qp_needed else None
        ),
        "authorize_profiled_schur_next": authorize_next,
        "high_correlation_is_not_a_weak_mode": True,
        "legal_final_claim": (
            "Under a frozen and verified measurement/surface contract, "
            "a curvature error can or cannot produce a given IFT residual "
            "response, and is or is not similar to listed geometry/slope "
            "responses.  This does not establish a natural-data q/p-R_y weak mode."
        ),
    }


def decide(
    *,
    inherited: Mapping[str, Any],
    controls: Mapping[str, Any],
    track_results: Sequence[Mapping[str, Any]],
    dumps_materialized: bool,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    if bool(config.get("three_st_qp_trusted_observable")):
        raise YasuS3AError("three_st_qp_trusted_observable must stay false")
    if bool(config.get("use_qp_bending_proxy")):
        refuse_s2k_proxy()
    if bool(config.get("weak_mode_claim_authorized")):
        refuse_weak_mode_claim()
    pins_ok = bool((inherited.get("pinned_calypso_sources") or {}).get("all_match", True))
    inherited_ok = inherited.get("workbook_119", {}).get("decision") == "three_st_qp_calibration_not_established"
    controls_ok = bool(controls.get("analytic_controls_closed"))
    summary = summarize_tracks(track_results)
    if not pins_ok or not inherited_ok:
        decision = DECISION_NOT
        verdict = STATUS_FAIL
    elif not dumps_materialized:
        decision = DECISION_CONTRACT
        verdict = STATUS_PASS if controls_ok else STATUS_FAIL
    elif (
        not controls_ok
        or summary["n_jacobian_eligible_closed"] <= 0
        or bool(summary["eligible_contract_breaks"])
    ):
        decision = DECISION_RECORDED
        verdict = STATUS_FAIL
    else:
        decision = DECISION_RECORDED
        verdict = STATUS_PASS
    return {
        "kind": "yasu_s3a_jacobian_contract",
        "task": TASK,
        "workbook": WORKBOOK,
        "verdict": verdict,
        "decision": decision,
        "three_st_qp_trusted_observable": False,
        "residual_conditional_authorized": False,
        "s2_flipped_to_pass": False,
        "qp_bending_proxy_used": False,
        "native_5x5_weighted": False,
        "four_station_association_used": False,
        "geometry_write_allowed": False,
        "held_out_accessed": False,
        "b14m_reopen_authorized": False,
        "b15_authorized": False,
        "htcondor_submitted": False,
        "weak_mode_claimed": False,
        "mean_response_only": True,
        "focus_replaced_after_results": False,
        "dumps_materialized": dumps_materialized,
        "analytic_controls_closed": controls_ok,
        "inherited_workbook_119": inherited.get("workbook_119", {}).get("decision"),
        "s2k_isolated": True,
        "p1_blockers": list(P1_BLOCKERS),
        "controls": controls,
        "summary": summary,
        "n_tracks": summary["n_tracks"],
        "authorize_profiled_schur_next": bool(
            verdict == STATUS_PASS
            and dumps_materialized
            and summary["authorize_profiled_schur_next"]
        ),
        "legal_final_claim": summary["legal_final_claim"],
        "forbidden_claim": "natural-data q/p-R_y weak mode is not established",
        "residual_definition": residual_definition(),
        "state_definition": state_definition(),
        "geometry_response_definition": geometry_response_definition(),
        "identity_contract": identity_contract(),
        "association_contract": association_contract(),
        "finite_difference": {
            "deltas": frozen_deltas(config),
            "rung_factors": list(FD_RUNG_FACTORS),
            "relative_convergence": REL_CONVERGENCE,
            "absolute_residual_tolerance_mm": ABS_RESIDUAL_TOL_MM,
        },
    }


def dump_path_for_source(config: Mapping[str, Any], source_id: str) -> Path:
    return (
        resolve_under_root(project_root(), str(config["dump_root"]))
        / source_id
        / str(config["dump_filename"])
    )


def inventory_dumps(config: Mapping[str, Any]) -> dict[str, Any]:
    present = {}
    for spec in config["mc_data"]["construction_sources"] + config["mc_data"]["validation_sources"]:
        path = dump_path_for_source(config, spec["source_id"])
        present[spec["source_id"]] = path.is_file()
    return {
        "sources_present": present,
        "dumps_present": bool(present) and all(present.values()),
    }


def attach_digests(row: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute SHA-256 digests from frozen identifier lists in the dump."""
    merged = dict(row)
    mot = [str(item) for item in (row.get("mot_identifiers") or [])]
    associated = [str(item) for item in (row.get("associated_ift_identifiers") or [])]
    inputs = row.get("measurement_digest_inputs")
    if inputs is None:
        inputs = sorted(mot + associated)
    merged["measurement_digest"] = measurement_digest([str(item) for item in inputs])
    merged["association_digest"] = measurement_digest(associated)
    return merged


def load_focus_tracks(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    wanted = {
        (row["source_id"], int(row["skip_index"]), int(row["track_index"])): row
        for row in config["focus_identities"]
    }
    tracks: list[dict[str, Any]] = []
    for spec in config["mc_data"]["construction_sources"] + config["mc_data"]["validation_sources"]:
        split = "construction" if spec in config["mc_data"]["construction_sources"] else "validation"
        path = dump_path_for_source(config, spec["source_id"])
        if not path.is_file():
            continue
        authorize_path(path, AccessScope.DEVELOPMENT_VALIDATION, split=access_split(split))
        for row in iter_jsonl(path, split=split):
            if row.get("kind") != "track":
                continue
            key = (row.get("source_id"), int(row.get("skip_index")), int(row.get("track_index")))
            if key in wanted:
                merged = attach_digests(row)
                merged["role"] = wanted[key]["role"]
                merged["file_sha256"] = merged.get("file_sha256") or wanted[key]["file_sha256"]
                tracks.append(merged)
    return tracks


def provenance_hashes(config: Mapping[str, Any]) -> dict[str, str]:
    from datasets.three_st_qp_calibration import provenance_hashes as _hashes

    return _hashes(config)
