"""Workbook 86: Full CurvilinearUVT Branch Covariance Contract Repair Validation.

Replaces the WB85 beam-track slot map ``loc1=-y, loc2=+x`` with the full
Athena ``CurvilinearUVT`` map on both ``|t.z|`` branches.  Re-runs the frozen
WB83 Stage A (same split, same gates, same truth) and compares baseline /
WB85 beam repair / WB86 full repair.  No scale, chi2 tuning, outlier drop,
or angle-based rejection.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from alignment.operating_protocol_v1_final_closure import (
    project_root,
    resolve_under_root,
    sha256_file,
)
from alignment import segmentfit_covariance_coordinate_contract as sccc
from alignment import segmentfit_getstate_repair_validation as sgrv
from alignment.segmentfit_getstate_repair_validation import (
    FROZEN_WB83_GATES,
    RepairRecords,
    _evaluate_records,
    _load_split_repair_records,
    _split_calibrated,
    curvilinear_slot_map as beam_slot_map,
    repair_exported_covariance as beam_repair_exported_covariance,
)

SCHEMA_VERSION = "full-curvilinear-uvt-covariance-contract-repair-validation-v1"
WORKBOOK = 86
DEFAULT_CONFIG = (
    "configs/full_curvilinear_uvt_covariance_contract_repair_validation_v1.yaml"
)

DECISION_VALIDATED = "segmentfit_full_curvilinear_covariance_contract_validated"
DECISION_NOT_CLOSED = "segmentfit_covariance_model_not_closed_after_coordinate_repair"

BEAM_ABS_T_DOT_Z = 0.99


class ConfigError(ValueError):
    """Raised when the WB86 config or its frozen inheritance is inconsistent."""


def _expect_false(config: Mapping[str, Any], key: str) -> None:
    if bool(config.get(key, False)) is not False:
        raise ConfigError(f"frozen flag must be false: {key}")


def _expect_true(config: Mapping[str, Any], key: str) -> None:
    if bool(config.get(key, False)) is not True:
        raise ConfigError(f"frozen prohibition must be true: {key}")


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the WB86 config and verify the frozen WB81–WB85 inheritance."""
    config_path = resolve_under_root(project_root(), str(path or DEFAULT_CONFIG))
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, Mapping):
        raise ConfigError(f"WB86 config must be a mapping: {config_path}")
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ConfigError(f"schema_version must be {SCHEMA_VERSION}")
    if int(config.get("workbook", -1)) != WORKBOOK:
        raise ConfigError(f"workbook must be {WORKBOOK}")

    for key in (
        "geometry_write_allowed",
        "official_conditions_write_allowed",
        "real_data_candidate_alignment_authorized",
        "real_data_alignment_authorized",
        "measurement_model_validated",
        "held_out_accessed",
        "external_constraint_ingest_authorized",
    ):
        _expect_false(config, key)
    for key in (
        "do_not_modify_covariance_with_scale_factors",
        "do_not_add_scale_factors",
        "do_not_tune_parameters_to_chi2",
        "do_not_enter_alignment",
        "do_not_read_real_data_residuals",
        "do_not_write_geometry_or_conditions_payload",
        "do_not_enter_faseracts_propagation",
        "do_not_enter_stage_b",
        "do_not_open_held_out",
        "do_not_reject_by_angle",
        "do_not_drop_outliers",
        "do_not_change_central_prediction_in_covariance_repair",
    ):
        _expect_true(config, key)

    gates = config.get("source_closure_gates", {})
    for key, expected in FROZEN_WB83_GATES.items():
        if gates.get(key) != expected:
            raise ConfigError(f"WB83 Stage-A gate must stay frozen: {key}")

    inheritance = config.get("inheritance", {})
    def _parent(*args, **kwargs):
        try:
            sgrv._verify_parent_inheritance(*args, **kwargs)
        except sgrv.ConfigError as exc:
            raise ConfigError(str(exc)) from exc

    _parent(
        inheritance,
        "workbook_81",
        decision_filename="propagated_covariance_decision.json",
        require_mechanism=True,
    )
    _parent(
        inheritance,
        "workbook_82",
        decision_filename="wide_ty_mc_support_decision.json",
        require_mechanism=False,
    )
    _parent(
        inheritance,
        "workbook_83",
        decision_filename="propagated_covariance_upstream_repair_decision.json",
        require_mechanism=True,
    )
    _parent(
        inheritance,
        "workbook_84",
        decision_filename="segmentfit_covariance_coordinate_contract_decision.json",
        require_mechanism=True,
    )
    # WB85 stores the remaining-structure category, not failure_classification.
    _parent(
        inheritance,
        "workbook_85",
        decision_filename="covariance_repair_decision.json",
        require_mechanism=False,
    )
    wb85_root = resolve_under_root(
        project_root(), str(inheritance["workbook_85_output_root"])
    )
    wb85 = json.loads(
        (wb85_root / "covariance_repair_decision.json").read_text(encoding="utf-8")
    )
    if wb85.get("decision") != str(inheritance["workbook_85_frozen_decision"]):
        raise ConfigError("WB85 frozen decision mismatch")
    remaining = (wb85.get("remaining_structure") or {}).get("category")
    if remaining != str(inheritance["workbook_85_frozen_mechanism"]):
        raise ConfigError("WB85 frozen remaining-structure mismatch")

    mc = config["mc_data"]
    if set(mc["construction_source_ids"]) & set(mc["validation_source_ids"]):
        raise ConfigError("MC construction/validation sources are not disjoint")
    return dict(config)


# ---------------------------------------------------------------------------
# Full Athena CurvilinearUVT slot map (both |t.z| branches)
# ---------------------------------------------------------------------------


def abs_t_dot_z(tx: float, ty: float) -> float:
    direction = np.array([tx, ty, 1.0], dtype=float)
    t = direction / np.linalg.norm(direction)
    return float(abs(t[2]))


def full_curvilinear_slot_map(curv_u: np.ndarray, curv_v: np.ndarray) -> np.ndarray:
    """Map ``(x, y, phi, theta) -> (loc1, loc2, phi, theta)`` from CurvilinearUVT.

    Solve ``loc1 * curvU_xy + loc2 * curvV_xy = (x, y)``.  On the beam branch
    (``|t.z| >= 0.99``) this is only approximately ``loc1 ≈ -y``, ``loc2 ≈ +x``
    (``O(tx)`` corrections, because ``curvU`` is not exactly ``-ŷ``).
    """
    a = np.array(
        [[float(curv_u[0]), float(curv_v[0])], [float(curv_u[1]), float(curv_v[1])]],
        dtype=float,
    )
    p_xy = np.linalg.inv(a)
    p = np.eye(4)
    p[:2, :2] = p_xy
    return p


def repair_exported_covariance_full(
    c_exported: np.ndarray,
    tx: float,
    ty: float,
    ref_pos: np.ndarray | None = None,
    dz: float = 0.0,
) -> np.ndarray | None:
    """Invert the code-as-written GetState transform; re-export with full UVT."""
    if ref_pos is None:
        ref_pos = np.zeros(3)
    phi = float(np.arctan2(ty, tx))
    theta = float(np.arctan(np.sqrt(tx * tx + ty * ty)))
    direction = np.array([tx, ty, 1.0], dtype=float)
    curv_u, curv_v, _ = sccc.curvilinear_uvt(direction)
    j_wrong = sccc.segment_fit_jacobian(tx, ty, dz=dz, sign_error=True)
    j_correct = sccc.segment_fit_jacobian(tx, ty, dz=dz, sign_error=False)
    j_exp = sccc.exporter_jacobian(phi, theta, ref_pos, curv_u, curv_v)
    slot = full_curvilinear_slot_map(curv_u, curv_v)
    j_buggy = j_exp @ j_wrong
    j_repaired = j_exp @ slot @ j_correct
    try:
        inv = np.linalg.inv(j_buggy)
    except np.linalg.LinAlgError:
        return None
    c_fit = inv @ c_exported @ inv.T
    return j_repaired @ c_fit @ j_repaired.T


def repair_covariance_batch_full(
    c_exported: np.ndarray,
    tx: np.ndarray,
    ty: np.ndarray,
    ref_pos: np.ndarray | None = None,
    dz: float = 0.0,
) -> tuple[np.ndarray, int]:
    n = c_exported.shape[0]
    out = np.array(c_exported, copy=True)
    n_fail = 0
    for i in range(n):
        repaired = repair_exported_covariance_full(
            c_exported[i], float(tx[i]), float(ty[i]), ref_pos=ref_pos, dz=dz
        )
        if repaired is None or not np.all(np.isfinite(repaired)):
            n_fail += 1
            continue
        out[i] = repaired
    return out, n_fail


def branch_mask(tx: np.ndarray, ty: np.ndarray, *, large_angle: bool) -> np.ndarray:
    tz = np.array([abs_t_dot_z(float(a), float(b)) for a, b in zip(tx, ty)])
    return tz < BEAM_ABS_T_DOT_Z if large_angle else tz >= BEAM_ABS_T_DOT_Z


def _slice_records(recs: RepairRecords, mask: np.ndarray) -> RepairRecords:
    return RepairRecords(
        station_id=recs.station_id,
        e_source=recs.e_source[mask],
        c_source=recs.c_source[mask],
        tx=recs.tx[mask],
        ty=recs.ty[mask],
        abs_tx=recs.abs_tx[mask],
        abs_ty=recs.abs_ty[mask],
        abs_q_over_p=recs.abs_q_over_p[mask],
        truth_pdg=recs.truth_pdg[mask],
        n_matched=int(mask.sum()),
    )


def _arm_metrics(
    recs: RepairRecords, config: Mapping[str, Any], c_source: np.ndarray
) -> dict[str, Any]:
    return _evaluate_records(recs, config, c_source)


def _track_chi2(residual: np.ndarray, cov: np.ndarray) -> float:
    try:
        inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        return float("nan")
    return float(residual @ inv @ residual)


def per_track_covariance_diagnostic(
    recs: RepairRecords, c_map: Mapping[str, np.ndarray]
) -> list[dict[str, Any]]:
    """Residual-blind per-track dump used when a subset is below the n=30 gate."""
    tracks: list[dict[str, Any]] = []
    for i in range(recs.size):
        entry: dict[str, Any] = {
            "tx": float(recs.tx[i]),
            "ty": float(recs.ty[i]),
            "abs_t_dot_z": abs_t_dot_z(float(recs.tx[i]), float(recs.ty[i])),
            "Cxx": {},
            "Cyy": {},
            "Ctyty": {},
            "chi2": {},
        }
        for arm, cov in c_map.items():
            entry["Cxx"][arm] = float(cov[i, 0, 0])
            entry["Cyy"][arm] = float(cov[i, 1, 1])
            entry["Ctyty"][arm] = float(cov[i, 3, 3])
            entry["chi2"][arm] = _track_chi2(recs.e_source[i], cov[i])
        tracks.append(entry)
    return tracks


def _subset_report(
    recs: RepairRecords,
    config: Mapping[str, Any],
    c_map: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    min_n = int(config["source_closure_gates"]["min_tracklets_per_station"])
    out: dict[str, Any] = {}
    for name, large in (("beam", False), ("large_angle", True)):
        mask = branch_mask(recs.tx, recs.ty, large_angle=large)
        n = int(mask.sum())
        entry: dict[str, Any] = {
            "n_tracklets": n,
            "large_angle": large,
            "wb83_population_gates_applicable": n >= min_n,
        }
        if n == 0:
            out[name] = entry
            continue
        sliced = _slice_records(recs, mask)
        sliced_map = {arm: cov[mask] for arm, cov in c_map.items()}
        for arm, cov in sliced_map.items():
            entry[arm] = _arm_metrics(sliced, config, cov)
        if n < min_n:
            entry["per_track_diagnostic"] = per_track_covariance_diagnostic(
                sliced, sliced_map
            )
            entry["reason"] = "n_lt_min_tracklets_diagnostic_only"
        out[name] = entry
    return out


def run_three_way(config: Mapping[str, Any]) -> dict[str, Any]:
    """Compare baseline / WB85 beam repair / WB86 full CurvilinearUVT repair."""
    ref_pos = np.array(config["repair"]["reference_position"], dtype=float)
    dz = float(config["repair"]["dz"])
    before_after: dict[str, Any] = {}
    validation: dict[str, Any] = {}
    subsets: dict[str, Any] = {}
    n_fail_beam = 0
    n_fail_full = 0
    n_total = 0
    n_large = 0
    n_beam = 0
    for split in ("construction", "validation"):
        pooled = _load_split_repair_records(config, split)
        per_base: dict[str, Any] = {}
        per_beam: dict[str, Any] = {}
        per_full: dict[str, Any] = {}
        per_subset: dict[str, Any] = {}
        stats: dict[str, Any] = {}
        for station_id, recs in pooled.items():
            n_total += recs.size
            large = branch_mask(recs.tx, recs.ty, large_angle=True)
            n_large += int(large.sum())
            n_beam += int((~large).sum())
            c_beam, fail_b = sgrv.repair_covariance_batch(
                recs.c_source, recs.tx, recs.ty, ref_pos=ref_pos, dz=dz
            )
            c_full, fail_f = repair_covariance_batch_full(
                recs.c_source, recs.tx, recs.ty, ref_pos=ref_pos, dz=dz
            )
            n_fail_beam += fail_b
            n_fail_full += fail_f
            per_base[str(station_id)] = _arm_metrics(recs, config, recs.c_source)
            per_beam[str(station_id)] = _arm_metrics(recs, config, c_beam)
            per_full[str(station_id)] = _arm_metrics(recs, config, c_full)
            per_subset[str(station_id)] = _subset_report(
                recs, config, {"baseline": recs.c_source, "wb85_beam": c_beam, "wb86_full": c_full}
            )
            stats[str(station_id)] = {
                "n_tracklets": recs.size,
                "n_beam": int((~large).sum()),
                "n_large_angle": int(large.sum()),
                "n_beam_repair_failed": int(fail_b),
                "n_full_repair_failed": int(fail_f),
                "central_prediction_changed": False,
            }
        def _pack(per):
            return {
                "split": split,
                "source_ids": list(config["mc_data"][f"{split}_source_ids"]),
                "per_station": per,
            }

        baseline = _pack(per_base)
        beam = _pack(per_beam)
        full = _pack(per_full)
        before_after[split] = {
            "baseline": baseline,
            "wb85_beam": beam,
            "wb86_full": full,
            "repair_stats": stats,
        }
        subsets[split] = per_subset
        validation[split] = {
            "baseline_all_stations_calibrated": _split_calibrated(baseline),
            "wb85_beam_all_stations_calibrated": _split_calibrated(beam),
            "wb86_full_all_stations_calibrated": _split_calibrated(full),
        }
    validation["source_disjoint_agreement"] = bool(
        validation["construction"]["wb86_full_all_stations_calibrated"]
        == validation["validation"]["wb86_full_all_stations_calibrated"]
    )
    validation["whitening_closure_recovered"] = bool(
        validation["construction"]["wb86_full_all_stations_calibrated"]
        and validation["validation"]["wb86_full_all_stations_calibrated"]
    )
    validation["n_tracklets_total"] = int(n_total)
    validation["n_beam_total"] = int(n_beam)
    validation["n_large_angle_total"] = int(n_large)
    validation["n_beam_repair_failed_total"] = int(n_fail_beam)
    validation["n_full_repair_failed_total"] = int(n_fail_full)
    return {
        "source_covariance_closure_three_way": {
            "kind": "source_covariance_closure_three_way",
            "stage": "A",
            "arms": ["baseline", "wb85_beam", "wb86_full"],
            "repair": {
                "allowed": list(config["repair"]["allowed"]),
                "forbidden": list(config["repair"]["forbidden"]),
                "central_prediction_changed": False,
            },
            "construction": before_after["construction"],
            "validation": before_after["validation"],
            "diagnostic_only": True,
            "alignment_authorized": False,
        },
        "curvilinear_branch_subset_closure": {
            "kind": "curvilinear_branch_subset_closure",
            "beam_abs_t_dot_z_min": BEAM_ABS_T_DOT_Z,
            "construction": subsets["construction"],
            "validation": subsets["validation"],
        },
        "full_curvilinear_uvt_repair_validation": {
            "kind": "full_curvilinear_uvt_repair_validation",
            "gates": dict(config["source_closure_gates"]),
            "gates_identical_to_wb83": True,
            "validation": validation,
        },
    }


def decide(
    config: Mapping[str, Any], validation: Mapping[str, Any]
) -> dict[str, Any]:
    recovered = bool(validation.get("whitening_closure_recovered", False))
    agreed = bool(validation.get("source_disjoint_agreement", False))
    decision = DECISION_VALIDATED if recovered and agreed else DECISION_NOT_CLOSED
    return {
        "kind": "covariance_repair_decision",
        "decision": decision,
        "whitening_closure_recovered": recovered,
        "source_disjoint_agreement": agreed,
        "construction_full_calibrated": bool(
            validation["construction"]["wb86_full_all_stations_calibrated"]
        ),
        "validation_full_calibrated": bool(
            validation["validation"]["wb86_full_all_stations_calibrated"]
        ),
        "wb85_beam_still_insufficient": bool(
            not validation["construction"]["wb85_beam_all_stations_calibrated"]
            or not validation["validation"]["wb85_beam_all_stations_calibrated"]
        ),
        "measurement_model_validated": False,
        "measurement_model_v2_discussion_allowed": bool(decision == DECISION_VALIDATED),
        "propagated_covariance_validation_authorized": bool(
            decision == DECISION_VALIDATED
        ),
        "geometry_write_allowed": False,
        "real_data_alignment_authorized": False,
        "held_out_accessed": False,
        "faseracts_propagation_entered": False,
        "stage_b_entered": False,
        "frozen_v2_alignment_authorized": False,
    }
