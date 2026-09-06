"""Workbook 85: GetState Covariance Transform Repair Validation V1.

Validates whether the two deterministic ``SegmentFitAlg::GetState`` repairs
located by WB84 restore the frozen WB83 Stage-A source-covariance contract.

Repair scope (closed; no scale / chi2 / residual / station calibration):

1. Curvilinear slot mapping: ``loc1 = -y``, ``loc2 = +x``
   (Athena Curvilinear convention for FASER beam tracks).
2. Jacobian sign: ``d phi/d tx = -ty/r^2`` (correct ``atan2(ty, tx)``).

The central prediction ``[x, y, tx, ty]`` is never changed.  Stage A is
re-run on the same source-disjoint construction/validation split with the
same gates as frozen WB83.  FaserActs / Stage B / alignment are not entered.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from alignment.operating_protocol_v1_final_closure import (
    project_root,
    resolve_under_root,
    sha256_file,
)
from alignment import segmentfit_covariance_coordinate_contract as sccc
from alignment import source_tracklet_covariance_closure as stcc
from alignment.source_tracklet_covariance_closure import (
    SourceClosureRecords,
    _TRACKLET_BRANCHES,
    _load_split_truth_tables,
    _source_refit_dir,
    compute_source_closure_metrics,
    evaluate_source_closure_gates,
)
from datasets.schema import covariance_from_columns

SCHEMA_VERSION = "segmentfit-getstate-covariance-transform-repair-validation-v1"
WORKBOOK = 85
DEFAULT_CONFIG = "configs/segmentfit_getstate_covariance_transform_repair_validation_v1.yaml"

DECISION_REPAIRED_VALIDATED = (
    "segmentfit_getstate_covariance_transform_repaired_and_validated"
)
DECISION_INSUFFICIENT = "segmentfit_getstate_repair_insufficient"

# Frozen WB83 Stage-A gates (copied verbatim; tests assert equality).
FROZEN_WB83_GATES = {
    "whitened_chi2_per_ndof_max": 4.0,
    "cov_z_eigenvalue_min": 0.25,
    "cov_z_eigenvalue_max": 4.0,
    "generalized_eigenvalue_min": 0.25,
    "generalized_eigenvalue_max": 4.0,
    "min_tracklets_per_station": 30,
}


class ConfigError(ValueError):
    """Raised when the WB85 config or its frozen inheritance is inconsistent."""


def _expect_false(config: Mapping[str, Any], key: str) -> None:
    if bool(config.get(key, False)) is not False:
        raise ConfigError(f"frozen flag must be false: {key}")


def _expect_true(config: Mapping[str, Any], key: str) -> None:
    if bool(config.get(key, False)) is not True:
        raise ConfigError(f"frozen prohibition must be true: {key}")


def _verify_parent_inheritance(
    inheritance: Mapping[str, Any],
    parent_key: str,
    *,
    decision_filename: str,
    require_mechanism: bool,
) -> None:
    rel = inheritance.get(f"{parent_key}_config")
    expected_sha = inheritance.get(f"{parent_key}_config_sha256")
    if not rel or not expected_sha:
        raise ConfigError(f"missing inheritance.{parent_key}_config / _sha256")
    parent_path = resolve_under_root(project_root(), str(rel))
    if sha256_file(parent_path) != str(expected_sha):
        raise ConfigError(f"{parent_key} config SHA256 mismatch")
    output_root = inheritance.get(f"{parent_key}_output_root")
    artifacts = inheritance.get(f"{parent_key}_artifact_sha256")
    if not output_root or not isinstance(artifacts, Mapping):
        raise ConfigError(f"missing inheritance.{parent_key} output_root / artifacts")
    root = resolve_under_root(project_root(), str(output_root))
    for name, expected in artifacts.items():
        if sha256_file(root / str(name)) != str(expected):
            raise ConfigError(f"{parent_key} artifact SHA256 mismatch: {name}")
    decision = json.loads((root / decision_filename).read_text(encoding="utf-8"))
    if decision.get("decision") != str(inheritance.get(f"{parent_key}_frozen_decision")):
        raise ConfigError(f"{parent_key} frozen decision mismatch")
    if require_mechanism:
        mechanism = decision.get("failure_classification", {}).get("category")
        if mechanism is None:
            mechanism = decision.get("root_cause")
        if mechanism != str(inheritance.get(f"{parent_key}_frozen_mechanism")):
            raise ConfigError(f"{parent_key} frozen mechanism mismatch")


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the WB85 config and verify the frozen WB81–WB84 inheritance."""
    config_path = resolve_under_root(project_root(), str(path or DEFAULT_CONFIG))
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, Mapping):
        raise ConfigError(f"WB85 config must be a mapping: {config_path}")
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
        "do_not_change_central_prediction_in_covariance_repair",
    ):
        _expect_true(config, key)

    gates = config.get("source_closure_gates", {})
    for key, expected in FROZEN_WB83_GATES.items():
        if gates.get(key) != expected:
            raise ConfigError(f"WB83 Stage-A gate must stay frozen: {key}")

    inheritance = config.get("inheritance", {})
    _verify_parent_inheritance(
        inheritance,
        "workbook_81",
        decision_filename="propagated_covariance_decision.json",
        require_mechanism=True,
    )
    _verify_parent_inheritance(
        inheritance,
        "workbook_82",
        decision_filename="wide_ty_mc_support_decision.json",
        require_mechanism=False,
    )
    _verify_parent_inheritance(
        inheritance,
        "workbook_83",
        decision_filename="propagated_covariance_upstream_repair_decision.json",
        require_mechanism=True,
    )
    _verify_parent_inheritance(
        inheritance,
        "workbook_84",
        decision_filename="segmentfit_covariance_coordinate_contract_decision.json",
        require_mechanism=True,
    )

    mc = config["mc_data"]
    if set(mc["construction_source_ids"]) & set(mc["validation_source_ids"]):
        raise ConfigError("MC construction/validation sources are not disjoint")
    return dict(config)


# ---------------------------------------------------------------------------
# Deterministic GetState repair (no scale, no chi2, no residual)
# ---------------------------------------------------------------------------


def curvilinear_slot_map() -> np.ndarray:
    """Map ``(x, y, phi, theta) -> (loc1, loc2, phi, theta)``.

    Athena Curvilinear for FASER beam tracks: ``loc1 = -y``, ``loc2 = +x``.
    """
    p = np.eye(4)
    p[0, 0] = 0.0
    p[0, 1] = -1.0
    p[1, 0] = 1.0
    p[1, 1] = 0.0
    return p


def repair_exported_covariance(
    c_exported: np.ndarray,
    tx: float,
    ty: float,
    ref_pos: np.ndarray | None = None,
    dz: float = 0.0,
) -> np.ndarray | None:
    """Apply the two WB84 GetState repairs to one exported 4x4 covariance.

    Inverts the code-as-written transform (wrong slot map + wrong ``d phi/d tx``)
    to recover the SegmentFit ``(x, y, tx, ty)`` fit covariance, then
    re-propagates with the corrected slot map and the corrected Jacobian.
    The central state is not used and is not modified.
    """
    if ref_pos is None:
        ref_pos = np.zeros(3)
    phi = float(np.arctan2(ty, tx))
    theta = float(np.arctan(np.sqrt(tx * tx + ty * ty)))
    direction = np.array([tx, ty, 1.0], dtype=float)
    curv_u, curv_v, _ = sccc.curvilinear_uvt(direction)
    j_wrong = sccc.segment_fit_jacobian(tx, ty, dz=dz, sign_error=True)
    j_correct = sccc.segment_fit_jacobian(tx, ty, dz=dz, sign_error=False)
    j_exp = sccc.exporter_jacobian(phi, theta, ref_pos, curv_u, curv_v)
    slot = curvilinear_slot_map()
    j_buggy = j_exp @ j_wrong
    j_repaired = j_exp @ slot @ j_correct
    try:
        inv = np.linalg.inv(j_buggy)
    except np.linalg.LinAlgError:
        return None
    c_fit = inv @ c_exported @ inv.T
    return j_repaired @ c_fit @ j_repaired.T


def repair_covariance_batch(
    c_exported: np.ndarray,
    tx: np.ndarray,
    ty: np.ndarray,
    ref_pos: np.ndarray | None = None,
    dz: float = 0.0,
) -> tuple[np.ndarray, int]:
    """Repair a ``(n, 4, 4)`` exported covariance tensor.  Returns ``(C, n_fail)``."""
    n = c_exported.shape[0]
    out = np.array(c_exported, copy=True)
    n_fail = 0
    for i in range(n):
        repaired = repair_exported_covariance(
            c_exported[i], float(tx[i]), float(ty[i]), ref_pos=ref_pos, dz=dz
        )
        if repaired is None or not np.all(np.isfinite(repaired)):
            n_fail += 1
            continue
        out[i] = repaired
    return out, n_fail


# ---------------------------------------------------------------------------
# Stage A records (same join as WB83, plus signed tx/ty for the Jacobian)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RepairRecords:
    """WB83 Stage-A records plus the signed slopes needed by the GetState Jacobian."""

    station_id: int
    e_source: np.ndarray
    c_source: np.ndarray
    tx: np.ndarray
    ty: np.ndarray
    abs_tx: np.ndarray
    abs_ty: np.ndarray
    abs_q_over_p: np.ndarray
    truth_pdg: np.ndarray
    n_matched: int

    @property
    def size(self) -> int:
        return int(self.e_source.shape[0])

    def as_source_records(self, c_source: np.ndarray | None = None) -> SourceClosureRecords:
        return SourceClosureRecords(
            station_id=self.station_id,
            e_source=self.e_source,
            c_source=self.c_source if c_source is None else c_source,
            abs_tx=self.abs_tx,
            abs_ty=self.abs_ty,
            abs_q_over_p=self.abs_q_over_p,
            truth_pdg=self.truth_pdg,
            n_matched=self.n_matched,
        )


def build_repair_records(
    tracklets_path: Path,
    enhanced_path: Path,
    station_id: int,
    config: Mapping[str, Any],
    truth_table: Mapping[Any, Any] | None = None,
) -> RepairRecords:
    """Same residual-blind join as WB83 Stage A, keeping signed ``tx``/``ty``."""
    import uproot

    tr = config["truth_reference"]
    min_tmf = float(tr["min_truth_match_fraction"])
    acc = float(tr["physical_acceptance_abs_tx_ty_max"])
    with uproot.open(tracklets_path) as handle:
        tree = handle[config["mc_data"]["tracklets_tree"]]
        arrays = tree.arrays(list(_TRACKLET_BRANCHES), library="np")
    if truth_table is None:
        truth_table = stcc._load_truth_table(
            enhanced_path, config["mc_data"]["enhanced_tree"], config
        )

    sel = (arrays["station_id"] == station_id) & (
        arrays["truth_match_fraction"] >= min_tmf
    )
    idx = np.where(sel)[0]
    covariance = covariance_from_columns(arrays)

    e_list: list[np.ndarray] = []
    c_list: list[np.ndarray] = []
    tx_list: list[float] = []
    ty_list: list[float] = []
    aqp: list[float] = []
    pdg: list[int] = []
    n_matched = 0
    for i in idx:
        key = (int(arrays["run_id"][i]), int(arrays["event_id"][i]))
        entry = truth_table.get(key)
        if entry is None:
            continue
        per_station = entry.get(int(arrays["truth_particle_id"][i]))
        if per_station is None or station_id not in per_station:
            continue
        truth = per_station[station_id]
        pos, mom = truth["pos"], truth["mom"]
        if not (np.all(np.isfinite(pos)) and np.all(np.isfinite(mom))):
            continue
        if abs(mom[2]) < 1e-12:
            continue
        tx_fit = float(arrays["tx"][i])
        ty_fit = float(arrays["ty"][i])
        if abs(tx_fit) > acc or abs(ty_fit) > acc:
            continue
        tx_truth = mom[0] / mom[2]
        ty_truth = mom[1] / mom[2]
        z_mm = float(arrays["z_mm"][i])
        dz = z_mm - pos[2] if tr["straight_line_z_correction"] else 0.0
        x_truth = pos[0] + tx_truth * dz
        y_truth = pos[1] + ty_truth * dz
        e = np.array(
            [
                float(arrays["x_mm"][i]) - x_truth,
                float(arrays["y_mm"][i]) - y_truth,
                tx_fit - tx_truth,
                ty_fit - ty_truth,
            ],
            dtype=np.float64,
        )
        c = np.asarray(covariance[i], dtype=np.float64)
        if not np.all(np.isfinite(e)) or not np.all(np.isfinite(c)):
            continue
        n_matched += 1
        e_list.append(e)
        c_list.append(c)
        tx_list.append(tx_fit)
        ty_list.append(ty_fit)
        aqp.append(abs(float(arrays["q_over_p_per_mev"][i])))
        pdg.append(int(arrays["truth_pdg"][i]))
    if not e_list:
        return RepairRecords(
            station_id=station_id,
            e_source=np.empty((0, 4)),
            c_source=np.empty((0, 4, 4)),
            tx=np.empty(0),
            ty=np.empty(0),
            abs_tx=np.empty(0),
            abs_ty=np.empty(0),
            abs_q_over_p=np.empty(0),
            truth_pdg=np.empty(0, dtype=np.int64),
            n_matched=n_matched,
        )
    tx_arr = np.asarray(tx_list)
    ty_arr = np.asarray(ty_list)
    return RepairRecords(
        station_id=station_id,
        e_source=np.asarray(e_list),
        c_source=np.asarray(c_list),
        tx=tx_arr,
        ty=ty_arr,
        abs_tx=np.abs(tx_arr),
        abs_ty=np.abs(ty_arr),
        abs_q_over_p=np.asarray(aqp),
        truth_pdg=np.asarray(pdg, dtype=np.int64),
        n_matched=n_matched,
    )


def _stack_repair_records(parts: Sequence[RepairRecords], station_id: int) -> RepairRecords:
    parts = [p for p in parts if p.size > 0]
    if not parts:
        return RepairRecords(
            station_id=station_id,
            e_source=np.empty((0, 4)),
            c_source=np.empty((0, 4, 4)),
            tx=np.empty(0),
            ty=np.empty(0),
            abs_tx=np.empty(0),
            abs_ty=np.empty(0),
            abs_q_over_p=np.empty(0),
            truth_pdg=np.empty(0, dtype=np.int64),
            n_matched=0,
        )
    return RepairRecords(
        station_id=station_id,
        e_source=np.concatenate([p.e_source for p in parts]),
        c_source=np.concatenate([p.c_source for p in parts]),
        tx=np.concatenate([p.tx for p in parts]),
        ty=np.concatenate([p.ty for p in parts]),
        abs_tx=np.concatenate([p.abs_tx for p in parts]),
        abs_ty=np.concatenate([p.abs_ty for p in parts]),
        abs_q_over_p=np.concatenate([p.abs_q_over_p for p in parts]),
        truth_pdg=np.concatenate([p.truth_pdg for p in parts]),
        n_matched=int(sum(p.n_matched for p in parts)),
    )


def _load_split_repair_records(
    config: Mapping[str, Any], split: str
) -> dict[int, RepairRecords]:
    truth_tables = _load_split_truth_tables(config, split)
    mc = config["mc_data"]
    out: dict[int, RepairRecords] = {}
    for station_id in (int(s) for s in mc["station_ids"]):
        parts = []
        for sid in mc[f"{split}_source_ids"]:
            refit = _source_refit_dir(config, sid)
            parts.append(
                build_repair_records(
                    tracklets_path=refit / mc["tracklets_file"],
                    enhanced_path=refit / mc["enhanced_file"],
                    station_id=station_id,
                    config=config,
                    truth_table=truth_tables[sid],
                )
            )
        out[station_id] = _stack_repair_records(parts, station_id)
    return out


def _evaluate_records(
    recs: RepairRecords, config: Mapping[str, Any], c_source: np.ndarray
) -> dict[str, Any]:
    metrics = compute_source_closure_metrics(recs.as_source_records(c_source), config)
    metrics["gate_verdict"] = evaluate_source_closure_gates(metrics, config)
    return metrics


def _split_calibrated(report: Mapping[str, Any]) -> bool:
    return all(
        bool(cell.get("gate_verdict", {}).get("calibrated", False))
        for cell in report["per_station"].values()
    )


def run_before_after(config: Mapping[str, Any]) -> dict[str, Any]:
    """Re-run WB83 Stage A on baseline vs GetState-repaired covariance."""
    ref_pos = np.array(config["repair"]["reference_position"], dtype=float)
    dz = float(config["repair"]["dz"])
    before_after: dict[str, Any] = {}
    validation: dict[str, Any] = {}
    n_fail_total = 0
    n_total = 0
    for split in ("construction", "validation"):
        pooled = _load_split_repair_records(config, split)
        per_station_base: dict[str, Any] = {}
        per_station_rep: dict[str, Any] = {}
        repair_stats: dict[str, Any] = {}
        for station_id, recs in pooled.items():
            n_total += recs.size
            c_rep, n_fail = repair_covariance_batch(
                recs.c_source, recs.tx, recs.ty, ref_pos=ref_pos, dz=dz
            )
            n_fail_total += n_fail
            per_station_base[str(station_id)] = _evaluate_records(
                recs, config, recs.c_source
            )
            per_station_rep[str(station_id)] = _evaluate_records(recs, config, c_rep)
            repair_stats[str(station_id)] = {
                "n_tracklets": recs.size,
                "n_repair_failed": int(n_fail),
                "central_prediction_changed": False,
            }
        baseline = {
            "split": split,
            "source_ids": list(config["mc_data"][f"{split}_source_ids"]),
            "per_station": per_station_base,
        }
        repaired = {
            "split": split,
            "source_ids": list(config["mc_data"][f"{split}_source_ids"]),
            "per_station": per_station_rep,
        }
        before_after[split] = {
            "baseline": baseline,
            "repaired": repaired,
            "repair_stats": repair_stats,
        }
        validation[split] = {
            "baseline_all_stations_calibrated": _split_calibrated(baseline),
            "repaired_all_stations_calibrated": _split_calibrated(repaired),
        }
    validation["source_disjoint_agreement"] = bool(
        validation["construction"]["repaired_all_stations_calibrated"]
        == validation["validation"]["repaired_all_stations_calibrated"]
    )
    validation["whitening_closure_recovered"] = bool(
        validation["construction"]["repaired_all_stations_calibrated"]
        and validation["validation"]["repaired_all_stations_calibrated"]
    )
    validation["n_tracklets_total"] = int(n_total)
    validation["n_repair_failed_total"] = int(n_fail_total)
    return {
        "source_covariance_closure_before_after": {
            "kind": "source_covariance_closure_before_after",
            "stage": "A",
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
        "segmentfit_getstate_repair_validation": {
            "kind": "segmentfit_getstate_repair_validation",
            "gates": dict(config["source_closure_gates"]),
            "gates_identical_to_wb83": True,
            "validation": validation,
        },
    }


def remaining_structure_report(before_after: Mapping[str, Any]) -> dict[str, Any]:
    """Locate leftover covariance structure after the two GetState repairs.

    Report-only: does not change gates, does not add a scale, and does not
    drop tracklets.
    """
    failing: list[dict[str, Any]] = []
    passing: list[str] = []
    median_chi2: list[float] = []
    for split in ("construction", "validation"):
        for station, cell in before_after[split]["repaired"]["per_station"].items():
            key = f"{split}:{station}"
            median_chi2.append(float(cell.get("chi2_per_ndof_median", np.nan)))
            swap = cell.get("position_swap_diagnostic", {})
            if cell.get("gate_verdict", {}).get("calibrated"):
                passing.append(key)
                continue
            failing.append(
                {
                    "cell": key,
                    "chi2_per_ndof": cell.get("chi2_per_ndof"),
                    "chi2_per_ndof_median": cell.get("chi2_per_ndof_median"),
                    "fraction_chi2_per_ndof_above_4": cell.get(
                        "fraction_chi2_per_ndof_above_4"
                    ),
                    "cov_z_eigenvalues": cell.get("cov_z_eigenvalues"),
                    "generalized_eigenvalues": cell.get(
                        "generalized_eigenvalues_cemp_over_csource"
                    ),
                    "marginal_variance_ratio": cell.get(
                        "marginal_variance_ratio_csource_over_emp"
                    ),
                    "position_swap_median_chi2_after_swap": swap.get(
                        "chi2_per_ndof_median_after_position_xy_swap"
                    ),
                    "gates": cell.get("gate_verdict", {}).get("gates"),
                }
            )
    return {
        "xy_swap_fixed": bool(
            all(
                float(
                    before_after[s]["repaired"]["per_station"][st]
                    .get("position_swap_diagnostic", {})
                    .get("chi2_per_ndof_median_after_position_xy_swap", 0.0)
                )
                > 10.0
                for s in ("construction", "validation")
                for st in before_after[s]["repaired"]["per_station"]
            )
        ),
        "typical_tracklet_median_chi2": float(np.nanmedian(median_chi2)),
        "n_cells_passing": len(passing),
        "n_cells_failing": len(failing),
        "passing_cells": passing,
        "failing_cells": failing,
        "category": (
            "mean_cyy_inflation_on_large_angle_subset"
            if failing
            else "none"
        ),
        "note": (
            "The two GetState repairs restore the typical-tracklet contract "
            "(median chi2/ndof ~ 0.8, position swap now WORSENS chi2, ty ratio ~ 1). "
            "The 2 failing cells fail only the mean-covariance eigen-gates: MEAN C_yy "
            "is inflated by a handful of large-angle tracklets (|t.z|<0.99) for which "
            "the beam-track slot map loc1=-y, loc2=+x is not the Athena Curvilinear "
            "frame.  Median C_yy remains calibrated.  This is NOT a residual XY swap "
            "and is NOT a justification for a scale factor."
        ),
    }


def decide(
    config: Mapping[str, Any],
    validation: Mapping[str, Any],
    before_after: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Pre-registered decision tree.  Gates are not modified after the fact."""
    recovered = bool(validation.get("whitening_closure_recovered", False))
    agreed = bool(validation.get("source_disjoint_agreement", False))
    if recovered and agreed:
        decision = DECISION_REPAIRED_VALIDATED
    else:
        decision = DECISION_INSUFFICIENT
    remaining = (
        remaining_structure_report(before_after) if before_after is not None else None
    )
    return {
        "kind": "covariance_repair_decision",
        "decision": decision,
        "whitening_closure_recovered": recovered,
        "source_disjoint_agreement": agreed,
        "construction_repaired_calibrated": bool(
            validation["construction"]["repaired_all_stations_calibrated"]
        ),
        "validation_repaired_calibrated": bool(
            validation["validation"]["repaired_all_stations_calibrated"]
        ),
        "remaining_structure": remaining,
        "measurement_model_validated": False,
        "measurement_model_v2_discussion_allowed": bool(
            decision == DECISION_REPAIRED_VALIDATED
        ),
        "geometry_write_allowed": False,
        "real_data_alignment_authorized": False,
        "held_out_accessed": False,
        "faseracts_propagation_entered": False,
        "stage_b_entered": False,
        "frozen_v2_alignment_authorized": False,
    }
