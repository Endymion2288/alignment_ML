"""Yasu-S2H: CKF / KalmanFitter-refit provenance on ≤20 focus events.

Reads the WB123-authorized diagnostic dump.  Truth is joined only after
the dump and never enters the fit.  Does not flip S2 or open S3.
"""

from __future__ import annotations

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
from datasets.three_st_qp_calibration import (
    SOURCE_COLLECTION_NAME,
    access_split,
    iter_jsonl,
    verify_pinned_calypso_sources,
)

SCHEMA_VERSION = "three-st-qp-refit-provenance-v1"
DEFAULT_CONFIG = "configs/three_st_qp_refit_provenance_v1.yaml"
TASK = "YASU-S2H"
WORKBOOK = 124

DECISION_CONTRACT = "three_st_qp_refit_provenance_contract_established"
DECISION_RECORDED = "three_st_qp_refit_provenance_recorded"
DECISION_NOT = "three_st_qp_refit_provenance_not_established"

STATUS_SUPPORTED = "supported"
STATUS_REJECTED = "rejected"
STATUS_UNRESOLVED = "unresolved"

MECH_FALLBACK = "ckf_only_fallback_drives_s2_front"
MECH_QP_SHIFT = "kf_refit_introduces_qp_shift"
MECH_FRONT = "persisted_front_state_selection_artifact"
MECH_COV = "covariance_changed_in_refit"
MECH_MIXED = "mixed/inconclusive"

REQUIRED_INHERITANCE = (
    "workbook_117",
    "workbook_118",
    "workbook_119",
    "workbook_120",
    "workbook_121",
    "workbook_122",
    "workbook_123",
)

KIND_HOLE = "ckf_target_hole"
KIND_MOT = "first_mot"
KIND_OTHER = "other"


class ThreeStQpRefitProvenanceError(ValueError):
    """Raised when the S2H contract is illegal."""


def refuse_flip_s2() -> None:
    raise ThreeStQpRefitProvenanceError("S2H cannot set three_st_qp_trusted_observable true")


def refuse_residual_conditional() -> None:
    raise ThreeStQpRefitProvenanceError("E[r_IFT|q/p] is Stage 3; S2H does not open it")


def refuse_change_fitter() -> None:
    raise ThreeStQpRefitProvenanceError(
        "S2H must not change fitter, seed, hits, geometry, or covariance scale"
    )


def refuse_replace_focus() -> None:
    raise ThreeStQpRefitProvenanceError("focus identities are frozen; do not replace after seeing results")


def conversion_chain() -> dict[str, Any]:
    return {
        "calypso_git_sha": "40892527e9c65409afd2378a2abfc25ddbddac03",
        "ckf2_cxx": "Tracking/Acts/FaserActsKalmanFilter/src/CKF2.cxx",
        "ckf2_persist_loop": "CKF2.cxx:268-297",
        "add_fitted_params": "CKF2.cxx:278-280 and CKF2Config.py:137 addFittedParamsToTrack=True",
        "create_track_ckf": "CKF2.cxx:285 createTrack(..., fittedParams, backward=false)",
        "kf_refit_call": "CKF2.cxx:288-289 KalmanFitterTool::fit(trk, Zero(), isMC)",
        "refit_success_keeps_trk2": "CKF2.cxx:290-292",
        "refit_failure_keeps_ckf_trk": "CKF2.cxx:293-296",
        "create_track_mot_order": "CreateTrkTrackTool.cxx:25-78 insert-at-begin when !backward",
        "create_track_target_hole": "CreateTrkTrackTool.cxx:82-93 Hole + FitQualityOnSurface(-99, 0)",
        "kf_min_measurements": "KalmanFitterTool.cxx:348-351 default MinMeasurements=12",
        "kf_origin": "KalmanFitterTool.cxx:360 front()->z() - 10",
        "kf_inflate_cov": "KalmanFitterTool.cxx:372-374 covariance *= 10",
        "kf_success_create_without_fitted_params": "KalmanFitterTool.cxx:423 createTrack(gctx, track)",
        "kf_fail_returns_nullptr": "KalmanFitterTool.cxx:424-426",
        "without_ift_output": "faser_reco.py:317-319 CKF_woIFT OutputCollection=CKFTrackCollectionWithoutIFT",
        "backward_propagation_default": False,
        "persisted_front_if_fallback": "CKF target-plane Hole (chi2=-99); z ≈ seed minZ-10, usually near S1",
        "persisted_front_if_refit_success": "first MOT smoothed state; S2 if S1 is missing",
    }


def official_quantities() -> dict[str, str]:
    return {
        "kf_refit_attempted": "original path always attempts KalmanFitterTool::fit after createTrack",
        "kf_refit_succeeded": "true iff persisted front is not the CKF target Hole(-99,0)",
        "persisted_front_kind": "ckf_target_hole | first_mot | other",
        "pre_refit": "persisted native state if front is CKF target Hole(-99,0); otherwise discarded from xAOD",
        "post_refit": "persisted native state if front is first MOT; otherwise original KF failed and post is absent",
        "original_complementary_state_in_xaod": "always false: CKF2 keeps only trk2 or trk",
        "truth_join": "offline WB119 identity join after dump; never a fit input",
    }


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = resolve_under_root(project_root(), str(path or DEFAULT_CONFIG))
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ThreeStQpRefitProvenanceError(f"schema_version must be {SCHEMA_VERSION}")
    if config.get("task") != TASK:
        raise ThreeStQpRefitProvenanceError(f"task must be {TASK}")
    if int(config.get("workbook", -1)) != WORKBOOK:
        raise ThreeStQpRefitProvenanceError("workbook must be 124")
    if bool(config.get("three_st_qp_trusted_observable", True)):
        raise ThreeStQpRefitProvenanceError("three_st_qp_trusted_observable must stay false")
    if bool(config.get("residual_conditional_authorized", True)):
        raise ThreeStQpRefitProvenanceError("residual_conditional_authorized must stay false")
    identities = config.get("focus_identities") or []
    if len(identities) == 0 or len(identities) > int(config.get("max_focus_events", 20)):
        raise ThreeStQpRefitProvenanceError("focus_identities must be 1–20 frozen rows")
    if len({(r["source_id"], r["skip_index"], r["track_index"]) for r in identities}) != len(identities):
        raise ThreeStQpRefitProvenanceError("focus identities must be unique in skip_index")
    frozen_roles = [r for r in identities if r.get("role") == "frozen_wb119"]
    if len(frozen_roles) != 6:
        raise ThreeStQpRefitProvenanceError("the six WB119 identities must stay in the focus list")
    return dict(config)


def _expect_sha(path: Path, expected: str, label: str) -> None:
    digest = sha256_file(path)
    if digest != expected:
        raise ThreeStQpRefitProvenanceError(f"{label} hash mismatch: {digest}")


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
            raise ThreeStQpRefitProvenanceError(f"{workbook} decision must stay frozen")
        inherited[workbook] = {
            "decision": decision["decision"],
            "config_sha256": spec["config_sha256"],
            "decision_sha256": spec["decision_sha256"],
        }
    if inherited["workbook_119"]["decision"] != "three_st_qp_calibration_not_established":
        raise ThreeStQpRefitProvenanceError("WB119 must remain not_established")
    if inherited["workbook_123"]["decision"] != "three_st_qp_tail_source_recorded":
        raise ThreeStQpRefitProvenanceError("WB123 tail/source audit must remain recorded")
    pins = verify_pinned_calypso_sources(config)
    if not pins.get("all_match"):
        raise ThreeStQpRefitProvenanceError("pinned Calypso sources must match WB121 hashes")
    inherited["pinned_calypso_sources"] = pins
    return inherited


def _finite(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if np.isfinite(number) else None


def _front_label(z_mm: float | None, config: Mapping[str, Any]) -> str:
    if z_mm is None:
        return "front_unknown"
    station = config.get("station_z_mm") or {}
    s1 = float(station.get(1, station.get("1", 47.4)))
    s2 = float(station.get(2, station.get("2", 1237.4)))
    if abs(z_mm - s1) <= float(config["binning"]["front_near_s1_abs_mm"]):
        return "front_near_s1"
    if abs(z_mm - s2) <= float(config["binning"]["front_near_s2_abs_mm"]):
        return "front_near_s2"
    return "front_other"


def classify_persisted_front(row: Mapping[str, Any], config: Mapping[str, Any]) -> str:
    kind = row.get("persisted_front_kind")
    if kind in {KIND_HOLE, KIND_MOT, KIND_OTHER}:
        return str(kind)
    chi2 = _finite(row.get("front_fit_quality_chi2"))
    ndof = _finite(row.get("front_fit_quality_ndof"))
    is_hole = bool(row.get("front_is_hole"))
    hole_chi2 = float(config["binning"]["ckf_target_hole_chi2"])
    hole_ndof = float(config["binning"]["ckf_target_hole_ndof"])
    if is_hole and chi2 is not None and ndof is not None:
        if abs(chi2 - hole_chi2) <= 1.0e-6 and abs(ndof - hole_ndof) <= 1.0e-6:
            return KIND_HOLE
    if bool(row.get("front_is_measurement")):
        return KIND_MOT
    return KIND_OTHER


def _status(supported: bool, rejected: bool) -> str:
    if supported and not rejected:
        return STATUS_SUPPORTED
    if rejected and not supported:
        return STATUS_REJECTED
    return STATUS_UNRESOLVED


def evaluate_tracks(rows: Iterable[Mapping[str, Any]], config: Mapping[str, Any]) -> dict[str, Any]:
    tracks: list[dict[str, Any]] = []
    for raw in rows:
        if str(raw.get("kind", "track")) != "track":
            continue
        if str(raw.get("collection", SOURCE_COLLECTION_NAME)) != SOURCE_COLLECTION_NAME:
            continue
        kind = classify_persisted_front(raw, config)
        z_mm = _finite(raw.get("z_mm") if raw.get("z_mm") is not None else raw.get("front_z_mm"))
        q_fit = _finite(raw.get("q_over_p_fit_per_mev"))
        q_truth = _finite(raw.get("q_over_p_truth_s1_per_mev"))
        sigma = _finite(raw.get("sigma_q_over_p_per_mev"))
        q_post = _finite(raw.get("diagnostic_q_over_p_per_mev"))
        sigma_post = _finite(raw.get("diagnostic_sigma_q_over_p_per_mev"))
        delta = None if q_fit is None or q_truth is None else q_fit - q_truth
        pull = None if delta is None or sigma is None or sigma <= 0.0 else delta / sigma
        q_pre = q_fit
        if kind == KIND_HOLE:
            original_succeeded = False
        elif kind == KIND_MOT:
            original_succeeded = True
        else:
            original_succeeded = None
        missing = raw.get("missing") or raw.get("missing_pattern")
        if not missing:
            stations = {
                int(station)
                for station in (raw.get("measurements_on_track_stations") or [])
                if station in (1, 2, 3)
            }
            absent = {1, 2, 3} - stations
            if not absent:
                missing = "complete"
            elif absent == {1}:
                missing = "missing_s1"
            elif len(absent) >= 2:
                missing = "missing_two_or_more"
            elif absent == {2}:
                missing = "missing_s2"
            elif absent == {3}:
                missing = "missing_s3"
            else:
                missing = "missing_other"
        tracks.append(
            {
                "source_id": raw.get("source_id"),
                "run_id": raw.get("run_id"),
                "event_id": raw.get("event_id"),
                "track_index": raw.get("track_index"),
                "skip_index": raw.get("skip_index"),
                "role": raw.get("role"),
                "z_mm": z_mm,
                "front_label": _front_label(z_mm, config),
                "persisted_front_kind": kind,
                "front_is_hole": bool(raw.get("front_is_hole")),
                "front_is_measurement": bool(raw.get("front_is_measurement")),
                "n_mot": int(raw.get("n_mot") or 0),
                "n_tsos": int(raw.get("n_tsos") or 0),
                "missing": missing,
                "q_fit": q_fit,
                "q_truth": q_truth,
                "sigma": sigma,
                "delta": delta,
                "pull": pull,
                "kf_refit_attempted": bool(raw.get("kf_refit_attempted", True)),
                "kf_refit_succeeded": original_succeeded,
                "diagnostic_refit_attempted": bool(raw.get("diagnostic_refit_attempted", False)),
                "diagnostic_refit_succeeded": raw.get("diagnostic_refit_succeeded"),
                "diagnostic_fallback_reason": raw.get("diagnostic_fallback_reason"),
                "q_pre": q_pre,
                "q_post": q_post if q_post is not None else (q_fit if original_succeeded else None),
                "sigma_pre": sigma,
                "sigma_post": sigma_post if sigma_post is not None else (sigma if original_succeeded else None),
                "post_z_mm": _finite(raw.get("diagnostic_z_mm")),
                "truth_used_as_fit_seed": bool(raw.get("truth_used_as_fit_seed")),
                "truth_used_as_solution": bool(raw.get("truth_used_as_solution")),
                "tsos": list(raw.get("tsos") or []),
            }
        )
    return {
        "n_tracks": len(tracks),
        "n_truth_as_solution": sum(1 for row in tracks if row["truth_used_as_fit_seed"] or row["truth_used_as_solution"]),
        "tracks": tracks,
    }


def _assoc(tracks: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> dict[str, Any]:
    gates = config["gates"]
    s2 = [row for row in tracks if row["front_label"] == "front_near_s2"]
    n = len(s2)
    n_hole = sum(1 for row in s2 if row["persisted_front_kind"] == KIND_HOLE)
    n_mot = sum(1 for row in s2 if row["persisted_front_kind"] == KIND_MOT)
    frac = None if n <= 0 else n_hole / n
    missing_s1 = sum(1 for row in s2 if "missing_s1" in str(row.get("missing") or "") or str(row.get("missing")) == "missing_two_or_more")
    if n < int(gates["min_s2_front_association"]) or frac is None:
        status = STATUS_UNRESOLVED
    elif frac >= float(gates["fallback_s2_fraction_min"]):
        status = STATUS_SUPPORTED
    elif frac <= float(gates["fallback_s2_fraction_max_reject"]):
        status = STATUS_REJECTED
    else:
        status = STATUS_UNRESOLVED
    return {
        "status": status,
        "n_s2_front": n,
        "n_s2_ckf_target_hole": n_hole,
        "n_s2_first_mot": n_mot,
        "fallback_fraction": frac,
        "n_s2_missing_s1_like": missing_s1,
        "n_s2_hole_n_mot_lt_12": sum(
            1 for row in s2 if row["persisted_front_kind"] == KIND_HOLE and int(row.get("n_mot") or 0) < 12
        ),
        "n_s2_first_mot_n_mot_ge_12": sum(
            1 for row in s2 if row["persisted_front_kind"] == KIND_MOT and int(row.get("n_mot") or 0) >= 12
        ),
        "legal_successful_refit_first_mot": bool(n >= int(gates["min_s2_front_association"]) and n_mot == n and n_hole == 0),
        "one_to_one_s2_front_equals_fallback": bool(n >= int(gates["min_s2_front_association"]) and n_hole == n),
    }


def _qp_shift(tracks: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> dict[str, Any]:
    gates = config["gates"]
    diffs: list[float] = []
    for row in tracks:
        pre, post = row.get("q_pre"), row.get("q_post")
        if pre is None or post is None:
            continue
        if row.get("diagnostic_refit_succeeded") is not True and row.get("kf_refit_succeeded") is not False:
            # Need a real pre/post pair: fallback persisted + diagnostic post, or exported pair.
            if row.get("persisted_front_kind") != KIND_HOLE:
                continue
        diffs.append(abs(float(post) - float(pre)))
    if len(diffs) < 2:
        return {"status": STATUS_UNRESOLVED, "n_pairs": len(diffs), "median_abs_shift": None}
    med = float(np.median(diffs))
    supported = med >= float(gates["qp_shift_abs_min"])
    rejected = med < float(gates["qp_shift_abs_min"]) * 0.1
    return {
        "status": _status(supported, rejected and not supported),
        "n_pairs": len(diffs),
        "median_abs_shift": med,
    }


def _front_artifact(tracks: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> dict[str, Any]:
    known = [row for row in tracks if row["persisted_front_kind"] in {KIND_HOLE, KIND_MOT}]
    if len(known) < int(config["gates"]["min_tracks_contract"]):
        return {"status": STATUS_UNRESOLVED, "n_known": len(known)}
    inconsistent = 0
    for row in known:
        post_z = row.get("post_z_mm")
        z = row.get("z_mm")
        if post_z is not None and z is not None:
            if abs(float(post_z) - float(z)) > float(config["gates"]["front_z_match_abs_mm"]):
                if row["persisted_front_kind"] == KIND_MOT:
                    inconsistent += 1
    if inconsistent > 0:
        return {"status": STATUS_SUPPORTED, "n_known": len(known), "n_inconsistent": inconsistent}
    return {"status": STATUS_REJECTED, "n_known": len(known), "n_inconsistent": 0}


def _cov_change(tracks: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> dict[str, Any]:
    gates = config["gates"]
    ratios: list[float] = []
    for row in tracks:
        pre, post = row.get("sigma_pre"), row.get("sigma_post")
        if pre is None or post is None or pre <= 0.0 or post <= 0.0:
            continue
        if row.get("persisted_front_kind") != KIND_HOLE and row.get("diagnostic_refit_succeeded") is not True:
            continue
        ratios.append(abs(math.log(post / pre)))
    if len(ratios) < 2:
        return {"status": STATUS_UNRESOLVED, "n_pairs": len(ratios), "median_abs_log_ratio": None}
    med = float(np.median(ratios))
    supported = med >= float(gates["sigma_log_ratio_min"])
    rejected = med < 0.5 * float(gates["sigma_log_ratio_min"])
    return {
        "status": _status(supported, rejected and not supported),
        "n_pairs": len(ratios),
        "median_abs_log_ratio": med,
    }


def evaluate_provenance(acc: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    tracks = acc["tracks"]
    assoc = _assoc(tracks, config)
    qp = _qp_shift(tracks, config)
    front = _front_artifact(tracks, config)
    cov = _cov_change(tracks, config)
    verdicts = {
        MECH_FALLBACK: assoc,
        MECH_QP_SHIFT: qp,
        MECH_FRONT: front,
        MECH_COV: cov,
    }
    supported = [name for name, item in verdicts.items() if item.get("status") == STATUS_SUPPORTED]
    if assoc.get("legal_successful_refit_first_mot") and assoc.get("status") == STATUS_REJECTED:
        note = (
            "S2-front tracks are first-MOT states of a successful refit "
            "(missing S1); fallback hypothesis is rejected"
        )
    else:
        note = None
    official = supported[0] if len(supported) == 1 else MECH_MIXED
    return {
        "n_tracks": acc["n_tracks"],
        "n_s2_front": assoc["n_s2_front"],
        "n_s1_front": sum(1 for row in tracks if row["front_label"] == "front_near_s1"),
        "n_ckf_target_hole": sum(1 for row in tracks if row["persisted_front_kind"] == KIND_HOLE),
        "n_first_mot": sum(1 for row in tracks if row["persisted_front_kind"] == KIND_MOT),
        "n_truth_as_solution": acc["n_truth_as_solution"],
        "tracks": tracks,
        "structured_verdicts": verdicts,
        "official_mechanism": official,
        "fallback_note": note,
        "group_comparison": _group_comparison(tracks),
        "conversion_chain": conversion_chain(),
    }


def decide(
    report: Mapping[str, Any],
    inherited: Mapping[str, Any],
    *,
    dumps_materialized: bool,
    campaign: str,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    contract = "FAIL"
    mechanism = None
    if inherited.get("workbook_119", {}).get("decision") != "three_st_qp_calibration_not_established":
        mechanism = "s2_not_inherited_as_fail"
    elif inherited.get("workbook_123", {}).get("decision") != "three_st_qp_tail_source_recorded":
        mechanism = "s2g_not_inherited"
    elif int(report.get("n_truth_as_solution") or 0):
        mechanism = "truth_used_as_fit_or_solution"
    elif not dumps_materialized:
        mechanism = "provenance_dump_not_materialized"
    elif int(report.get("n_tracks") or 0) > int(config.get("max_focus_events", 20)):
        mechanism = "focus_list_exceeded"
    else:
        contract = "PASS"
    if contract != "PASS":
        decision = DECISION_NOT
        verdict = "FAIL"
        diagnosis = "FAIL"
        official = None
    elif int(report.get("n_tracks") or 0) < int(config["gates"]["min_tracks_contract"]):
        decision = DECISION_CONTRACT
        verdict = "PASS"
        diagnosis = "INCONCLUSIVE"
        official = None
    elif campaign != "batch" and int(report.get("n_s2_front") or 0) < int(
        config["gates"]["min_s2_front_association"]
    ):
        decision = DECISION_CONTRACT
        verdict = "PASS"
        diagnosis = "INCONCLUSIVE"
        official = None
    else:
        decision = DECISION_RECORDED
        verdict = "PASS"
        diagnosis = "RECORDED"
        official = report["official_mechanism"]
        mechanism = official
    return {
        "kind": "three_st_qp_refit_provenance_contract",
        "task": TASK,
        "workbook": WORKBOOK,
        "campaign": campaign,
        "verdict": verdict,
        "decision": decision,
        "mechanism": mechanism,
        "contract_verdict": contract,
        "diagnosis_verdict": diagnosis,
        "official_mechanism": official,
        "structured_verdicts": report.get("structured_verdicts"),
        "fallback_note": report.get("fallback_note"),
        "group_comparison": report.get("group_comparison"),
        "wb119_join": report.get("wb119_join"),
        "n_tracks": report.get("n_tracks"),
        "n_s2_front": report.get("n_s2_front"),
        "n_s1_front": report.get("n_s1_front"),
        "n_ckf_target_hole": report.get("n_ckf_target_hole"),
        "n_first_mot": report.get("n_first_mot"),
        "tracks": report.get("tracks"),
        "conversion_chain": conversion_chain(),
        "official_quantities": official_quantities(),
        "three_st_qp_trusted_observable": False,
        "residual_conditional_authorized": False,
        "s2_flipped_to_pass": False,
        "fitter_or_geometry_changed": False,
        "focus_replaced_after_results": False,
        "large_dump_submitted": False,
    }


def _median(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(np.median(values))


def _group_comparison(tracks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def _pack(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        return {
            "n": len(rows),
            "n_ckf_target_hole": sum(1 for row in rows if row["persisted_front_kind"] == KIND_HOLE),
            "n_first_mot": sum(1 for row in rows if row["persisted_front_kind"] == KIND_MOT),
            "median_q_fit": _median([row["q_fit"] for row in rows if row.get("q_fit") is not None]),
            "median_sigma": _median([row["sigma"] for row in rows if row.get("sigma") is not None]),
            "median_delta": _median([row["delta"] for row in rows if row.get("delta") is not None]),
            "median_pull": _median([row["pull"] for row in rows if row.get("pull") is not None]),
        }

    s2 = [row for row in tracks if row["front_label"] == "front_near_s2"]
    s1 = [row for row in tracks if row["front_label"] == "front_near_s1"]
    dirty = [row for row in tracks if "s2_front_dirty" in str(row.get("role") or "")]
    control = [row for row in tracks if "s1_front_control" in str(row.get("role") or "")]
    return {
        "s2_front": _pack(s2),
        "s1_front": _pack(s1),
        "s2_front_dirty": _pack(dirty),
        "s1_front_control": _pack(control),
    }


def dump_path_for_source(
    config: Mapping[str, Any],
    source_id: str,
    kind: str,
    campaign: str,
) -> Path:
    root = resolve_under_root(project_root(), str(config["dump_root"]))
    filename = (
        str(config["dump_filename"]) if kind == "tracks" else str(config["event_filename"])
    )
    return root / campaign / source_id / filename


def focus_pairs_for_source(config: Mapping[str, Any], source_id: str) -> str:
    pairs = [
        f"{int(row['event_id'])}:{int(row['track_index'])}"
        for row in config["focus_identities"]
        if row["source_id"] == source_id
    ]
    return ",".join(pairs)


def _source_split(config: Mapping[str, Any], source_id: str) -> str:
    for row in config["mc_data"]["construction_sources"]:
        if row["source_id"] == source_id:
            return "construction"
    return "validation"


def attach_focus_metadata(
    rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    index = {
        (row["source_id"], int(row["skip_index"]), int(row["track_index"])): row
        for row in config["focus_identities"]
    }
    attached: list[dict[str, Any]] = []
    for raw in rows:
        item = dict(raw)
        skip = item.get("skip_index")
        if skip is None:
            skip = item.get("event_id")
            item["skip_index"] = skip
        key = (item.get("source_id"), int(skip), int(item.get("track_index") or 0))
        meta = index.get(key)
        if meta is not None:
            item["role"] = meta.get("role")
            item["focus_reason"] = meta.get("reason")
            item["skip_index"] = meta.get("skip_index")
        attached.append(item)
    return attached


def join_wb119_calibration(
    rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Offline identity join.  Truth never rewrites a reconstructed state."""
    needed: dict[str, set[tuple[int, int]]] = {}
    for row in config["focus_identities"]:
        needed.setdefault(str(row["source_id"]), set()).add(
            (int(row["skip_index"]), int(row["track_index"]))
        )
    found: dict[tuple[str, int, int], dict[str, Any]] = {}
    campaign = str(config.get("wb119_join_campaign") or "batch")
    root = resolve_under_root(project_root(), str(config["wb119_dump_root"]))
    for source_id, wanted in needed.items():
        split = _source_split(config, source_id)
        track_path = root / campaign / source_id / str(config["wb119_dump_filename"])
        event_path = root / campaign / source_id / str(config["wb119_event_filename"])
        if not track_path.is_file() or not event_path.is_file():
            continue
        max_skip = max(item[0] for item in wanted)
        event_ids: list[int] = []
        for i, row in enumerate(iter_jsonl(event_path, split=split)):
            event_ids.append(int(row["event_id"]))
            if i >= max_skip:
                break
        ptr = 0
        prev = None
        n_this = 0
        for raw in iter_jsonl(track_path, split=split):
            event_id = raw.get("event_id")
            if event_id != prev:
                while ptr < len(event_ids) and event_ids[ptr] != event_id:
                    ptr += 1
                prev = event_id
            if ptr >= len(event_ids):
                break
            key = (ptr, int(raw.get("track_index") or 0))
            if key in wanted:
                found[(source_id, key[0], key[1])] = raw
                n_this += 1
                if n_this >= len(wanted):
                    break
    joined: list[dict[str, Any]] = []
    n_joined = 0
    for raw in rows:
        item = dict(raw)
        skip = int(item.get("skip_index") if item.get("skip_index") is not None else item.get("event_id"))
        match = found.get((str(item.get("source_id")), skip, int(item.get("track_index") or 0)))
        item["wb119_joined"] = match is not None
        if match is not None:
            n_joined += 1
            q_truth = _finite(match.get("q_over_p_truth_s1_per_mev"))
            q_fit = _finite(item.get("q_over_p_fit_per_mev"))
            sigma = _finite(item.get("sigma_q_over_p_per_mev"))
            if item.get("q_over_p_truth_s1_per_mev") is None and q_truth is not None:
                item["q_over_p_truth_s1_per_mev"] = q_truth
            item["wb119_q_over_p_truth_s1_per_mev"] = q_truth
            item["wb119_q_over_p_fit_per_mev"] = _finite(match.get("q_over_p_fit_per_mev"))
            item["wb119_sigma_q_over_p_per_mev"] = _finite(match.get("sigma_q_over_p_per_mev"))
            delta = None if q_fit is None or q_truth is None else q_fit - q_truth
            pull = None if delta is None or sigma is None or sigma <= 0.0 else delta / sigma
            item["wb119_delta"] = delta
            item["wb119_pull"] = pull
            item["q_over_p_fit_not_rewritten"] = True
        joined.append(item)
    return joined, {"n_joined": n_joined, "n_rows": len(rows), "truth_rewrote_fit": False}


def inventory_campaign(config: Mapping[str, Any], campaign: str) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    sources_present: list[str] = []
    for split_name in ("construction", "validation"):
        for spec in config["mc_data"][f"{split_name}_sources"]:
            source_id = spec["source_id"]
            path = dump_path_for_source(config, source_id, "tracks", campaign)
            if not path.is_file():
                continue
            sources_present.append(source_id)
            rows.extend(iter_jsonl(path, split=split_name))
    rows = attach_focus_metadata(rows, config)
    rows, join_info = join_wb119_calibration(rows, config)
    acc = evaluate_tracks(rows, config)
    report = evaluate_provenance(acc, config)
    report["wb119_join"] = join_info
    report["n_truth_as_solution"] = acc["n_truth_as_solution"]
    report["sources_present"] = sources_present
    report["dumps_present"] = len(sources_present) == (
        len(config["mc_data"]["construction_sources"])
        + len(config["mc_data"]["validation_sources"])
    )
    return report
