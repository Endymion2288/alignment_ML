"""Yasu Stage 2: 3ST CKF q/p calibration contract.

Official CKFTrackCollectionWithoutIFT signed q/p is the reconstructed
observable.  Truth q/p is a calibration reference only.  LTO, 4-station
CKF, dummy SegmentFit q/p, and IFT residual conditionals cannot
substitute.  Binning and gates are frozen before batch inspection.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

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

SCHEMA_VERSION = "three-st-qp-calibration-v1"
DEFAULT_CONFIG = "configs/three_st_qp_calibration_v1.yaml"
TASK = "YASU-S2"
WORKBOOK = 119

SOURCE_COLLECTION_NAME = SOURCE_COLLECTION
FOUR_STATION_COLLECTION = "CKFTrackCollection"
IFT_STATION_ID = 0
THREE_STATION_IDS = (1, 2, 3)
OFFICIAL_TRUTH_STATION = 1
QOVERP_INDEX = 4
STATION_Z_MM = (-1860.15, 47.4, 1237.4, 2427.4)

DECISION_ESTABLISHED = "three_st_qp_calibration_established"
DECISION_CONTRACT = "three_st_qp_calibration_contract_established"
DECISION_NOT_ESTABLISHED = "three_st_qp_calibration_not_established"

MECHANISM_STAGE0 = "without_ift_definition_not_inherited"
MECHANISM_STAGE1 = "three_st_to_ift_prediction_not_inherited"
MECHANISM_DUMP_MISSING = "calibration_dump_not_materialized"
MECHANISM_IFT_LEAK = "ift_measurement_or_outlier_present"
MECHANISM_DUMMY = "without_ift_qoverp_matches_segmentfit_dummy"
MECHANISM_TRUTH = "truth_qoverp_used_as_fit_or_solution"
MECHANISM_LTO_SUBSTITUTE = "lto_used_as_without_ift_substitute"
MECHANISM_FOUR_STATION = "four_station_ckf_used_as_qp_source"
MECHANISM_SOURCE_PIN = "pinned_calypso_source_hash_mismatch"
MECHANISM_NO_5X5 = "without_ift_5x5_not_materialized"
MECHANISM_NO_TRACKS = "no_without_ift_candidate"
MECHANISM_NO_MATCH = "truth_match_unavailable"
MECHANISM_NO_REF = "truth_reference_unavailable"
MECHANISM_MEAN_BIAS = "mean_bias"
MECHANISM_SIGN = "charge_sign_failure"
MECHANISM_KINEMATIC = "kinematic_dependent_bias"
MECHANISM_COV = "covariance_miscalibration"
MECHANISM_SELECTION = "candidate_selection_loss"
MECHANISM_FIT = "fit_failure"

PHYSICS_FAILURE_ORDER = (
    MECHANISM_SELECTION,
    MECHANISM_FIT,
    MECHANISM_SIGN,
    MECHANISM_MEAN_BIAS,
    MECHANISM_KINEMATIC,
    MECHANISM_COV,
)

ABS_QOVERP_EDGES = (0.0, 2.0e-7, 1.0e-6, 5.0e-6, 1.0e5)
ABS_QOVERP_LABELS = (
    "qp_near_zero_p_ge_5tev",
    "high_p_1_to_5tev",
    "mid_200gev_to_1tev",
    "high_curvature_p_lt_200gev",
)
P_TRUTH_EDGES_MEV = (0.0, 2.0e5, 5.0e5, 1.0e6, 2.0e6, 1.0e12)
P_TRUTH_LABELS = (
    "p_lt_200gev",
    "p_200_500gev",
    "p_500_1000gev",
    "p_1000_2000gev",
    "p_ge_2000gev",
)
SLOPE_EDGES = (-1.0, -0.010, -0.005, 0.0, 0.005, 0.010, 1.0)
TX_LABELS = (
    "tx_lt_m0p010",
    "tx_m0p010_to_m0p005",
    "tx_m0p005_to_0",
    "tx_0_to_0p005",
    "tx_0p005_to_0p010",
    "tx_ge_0p010",
)
TY_LABELS = (
    "ty_lt_m0p010",
    "ty_m0p010_to_m0p005",
    "ty_m0p005_to_0",
    "ty_0_to_0p005",
    "ty_0p005_to_0p010",
    "ty_ge_0p010",
)
HIGH_P_ABS_LABELS = frozenset({"qp_near_zero_p_ge_5tev"})
HIGH_P_P_LABELS = frozenset({"p_ge_2000gev"})


class ThreeStQpCalibrationError(ValueError):
    """Raised when the 3ST q/p calibration contract is illegal."""


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
        "tx_ty_definition": "px/pz, py/pz from TrackParameters.momentum()",
        "truth_is_not_a_fit_input": True,
    }


def truth_definition() -> dict[str, Any]:
    return {
        "matching_tool": "TrackTruthMatchingTool",
        "matching_method": "majority_barcode_from_SCT_SDO_Map_deposits_on_MOT",
        "momentum_tool": "FiducialParticleTool",
        "momentum_method": "average_FaserSiHit_momentum_per_station",
        "official_truth_station": OFFICIAL_TRUTH_STATION,
        "official_q_over_p": (
            "truthParticle.charge() / |p_truth| at station "
            f"{OFFICIAL_TRUTH_STATION} (S1, WithoutIFT front)"
        ),
        "official_formula_source": "NtupleDumperAlg.cxx tracklet path",
        "production_q_over_p": "diagnostic only; not the official reference",
        "tx_ty_truth": "px/pz, py/pz from station-1 FiducialParticleTool momentum",
        "role": "calibration_reference_only",
        "not_a_fit_input": True,
        "not_a_prior": True,
        "not_a_geometry_solve": True,
    }


def primary_quantities() -> dict[str, str]:
    return {
        "delta_q_over_p": "q_over_p_fit - q_over_p_truth_s1",
        "pull_z_q_over_p": "delta_q_over_p / sigma_q_over_p",
        "sign_agreement": "sign(q_over_p_fit) == sign(q_over_p_truth_s1)",
        "coverage_68": "fraction |z| < 1",
        "coverage_95": "fraction |z| < 1.96",
        "pull_mean": "mean z",
        "pull_width": "rms z",
        "charge_even_bias": "0.5 * (mean_delta_mu_plus + mean_delta_mu_minus)",
        "charge_odd_bias": "0.5 * (mean_delta_mu_plus - mean_delta_mu_minus)",
    }


def refuse_truth_qoverp() -> None:
    raise ThreeStQpCalibrationError(
        "truth q/p is a calibration reference only; it is not a fit seed, "
        "prior, prediction, or real-data solution"
    )


def refuse_dummy_segmentfit() -> None:
    raise ThreeStQpCalibrationError(
        "dummy SegmentFit covariance is not a physical prior"
    )


def refuse_lto_substitute() -> None:
    raise ThreeStQpCalibrationError(
        "WB109 LTO is not CKFTrackCollectionWithoutIFT; LTO keeps IFT"
    )


def refuse_four_station_substitute() -> None:
    raise ThreeStQpCalibrationError(
        "CKFTrackCollection is the 4-station collection, not the 3ST q/p source"
    )


def refuse_residual_conditional() -> None:
    raise ThreeStQpCalibrationError(
        "E[r_IFT|q/p] is Stage 3; Stage 2 does not open residual conditionals"
    )


def refuse_drop_sign_flip() -> None:
    raise ThreeStQpCalibrationError(
        "sign-flip tracks stay in the sample; they are a failure class"
    )


def refuse_covariance_rescale() -> None:
    raise ThreeStQpCalibrationError(
        "covariance rescale is forbidden; miscalibration is a failure class"
    )


def access_split(split: str) -> str:
    return without_ift_access_split(split)


def _require_edges(config: Mapping[str, Any], key: str, expected: Sequence[float]) -> None:
    got = [float(x) for x in config["binning"][key]]
    if len(got) != len(expected) or any(abs(a - b) > 0.0 for a, b in zip(got, expected)):
        raise ThreeStQpCalibrationError(f"binning.{key} is frozen and must not change")


def _require_labels(config: Mapping[str, Any], key: str, expected: Sequence[str]) -> None:
    got = [str(x) for x in config["binning"][key]]
    if got != list(expected):
        raise ThreeStQpCalibrationError(f"binning.{key} is frozen and must not change")


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = resolve_under_root(project_root(), str(path or DEFAULT_CONFIG))
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ThreeStQpCalibrationError(f"schema_version must be {SCHEMA_VERSION}")
    if config.get("task") != TASK:
        raise ThreeStQpCalibrationError(f"task must be {TASK}")
    if int(config.get("workbook", -1)) != WORKBOOK:
        raise ThreeStQpCalibrationError(f"workbook must be {WORKBOOK}")
    if str(config.get("source_collection")) != SOURCE_COLLECTION_NAME:
        raise ThreeStQpCalibrationError(
            "source_collection must be CKFTrackCollectionWithoutIFT"
        )
    if int(config.get("official_truth_station", -1)) != OFFICIAL_TRUTH_STATION:
        raise ThreeStQpCalibrationError("official_truth_station must be 1 (S1)")
    if int(config.get("ift_station_id", -1)) != IFT_STATION_ID:
        raise ThreeStQpCalibrationError("ift_station_id must be 0")
    station_z = config.get("station_z_mm") or {}
    for idx, expected in enumerate(STATION_Z_MM):
        if abs(float(station_z[idx]) - expected) > 1.0e-9:
            raise ThreeStQpCalibrationError(
                f"station_z_mm[{idx}] must stay at the WB87 value {expected}"
            )
    for key in (
        "geometry_write_allowed",
        "held_out_accessed",
        "real_data_alignment_authorized",
        "measurement_model_validated",
        "residual_conditional_authorized",
        "b14m_reopen_authorized",
        "b15_authorized",
    ):
        if bool(config.get(key, True)):
            raise ThreeStQpCalibrationError(f"{key} must be false")
    for key in (
        "do_not_rescale_covariance",
        "do_not_use_truth_q_over_p_as_real_data_solution",
        "do_not_use_truth_as_fit_seed_or_prior",
        "do_not_replace_fit_qoverp_with_truth",
        "do_not_use_dummy_segmentfit_covariance",
        "do_not_treat_lto_as_without_ift",
        "do_not_treat_four_station_ckf_as_without_ift",
        "do_not_enter_alignment",
        "do_not_enter_measurement_model_v2",
        "do_not_enter_residual_conditional",
        "do_not_start_new_source_campaign",
        "do_not_drop_sign_flip_tracks",
        "do_not_drop_high_momentum_tail",
        "do_not_drop_large_pull",
        "do_not_pick_favorable_momentum_range",
        "do_not_introduce_ad_hoc_prior",
        "unconstrained_tracker_only_stopped",
    ):
        if bool(config.get(key, False)) is not True:
            raise ThreeStQpCalibrationError(f"{key} must be true")
    _require_edges(config, "abs_q_over_p_per_mev_edges", ABS_QOVERP_EDGES)
    _require_labels(config, "abs_q_over_p_labels", ABS_QOVERP_LABELS)
    _require_edges(config, "p_truth_mev_edges", P_TRUTH_EDGES_MEV)
    _require_labels(config, "p_truth_labels", P_TRUTH_LABELS)
    _require_edges(config, "tx_edges", SLOPE_EDGES)
    _require_labels(config, "tx_labels", TX_LABELS)
    _require_edges(config, "ty_edges", SLOPE_EDGES)
    _require_labels(config, "ty_labels", TY_LABELS)
    return dict(config)


def _expect_sha(path: Path, expected: str, label: str) -> None:
    digest = sha256_file(path)
    if digest != expected:
        raise ThreeStQpCalibrationError(f"{label} hash mismatch: {digest}")


def _frozen_root(config: Mapping[str, Any]) -> Path:
    root = Path(str(config.get("frozen_artifact_root") or project_root()))
    if not root.is_dir():
        raise ThreeStQpCalibrationError(f"frozen_artifact_root is missing: {root}")
    return root


def inherit_frozen_stage(config: Mapping[str, Any]) -> dict[str, Any]:
    inherited: dict[str, Any] = {}
    frozen_root = _frozen_root(config)
    for workbook, extra in (
        ("workbook_87", {"must_not_reopen_transport": True}),
        ("workbook_95", {"dummy_still_rejected": True}),
        ("workbook_96", {"four_station_5x5_is_not_without_ift": True}),
        ("workbook_109", {"lto_is_not_without_ift": True}),
        ("workbook_117", {"authorizes_without_ift_definition_only": True}),
        ("workbook_118", {"authorizes_this_calibration_only": True}),
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
            raise ThreeStQpCalibrationError(f"{workbook} decision must stay frozen")
        inherited[workbook] = {
            "decision": decision["decision"],
            "config_sha256": spec["config_sha256"],
            "decision_sha256": spec["decision_sha256"],
            "artifact_root": str(root),
            **extra,
        }
    if inherited["workbook_117"]["decision"] != "ckf_without_ift_definition_established":
        raise ThreeStQpCalibrationError("Stage 0 must PASS before Stage 2")
    if inherited["workbook_118"]["decision"] != "three_st_to_ift_prediction_established":
        raise ThreeStQpCalibrationError("Stage 1 must PASS before Stage 2")
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


def iter_jsonl(path: Path, *, split: str) -> Iterator[dict[str, Any]]:
    authorized = authorize_path(
        path, AccessScope.DEVELOPMENT_VALIDATION, split=access_split(split)
    )
    if not authorized.is_file():
        return
    with authorized.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def load_jsonl(path: Path, *, split: str) -> list[dict[str, Any]]:
    return list(iter_jsonl(path, split=split))


def _as_matrix(payload: Any) -> np.ndarray | None:
    if payload is None:
        return None
    matrix = np.asarray(payload, dtype=np.float64)
    if matrix.shape != (5, 5):
        return None
    return matrix


def _finite(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    if not np.isfinite(number):
        return None
    return number


def _sign(value: float) -> int:
    if value > 0.0:
        return 1
    if value < 0.0:
        return -1
    return 0


def assign_bin(value: float, edges: Sequence[float], labels: Sequence[str]) -> str:
    for idx, label in enumerate(labels):
        if edges[idx] <= value < edges[idx + 1]:
            return label
    if value == edges[-1]:
        return labels[-1]
    return "out_of_range"


def _finite_stats(values: Sequence[float]) -> dict[str, int | float | None]:
    array = np.asarray(list(values), dtype=np.float64).reshape(-1)
    finite = array[np.isfinite(array)]
    result: dict[str, int | float | None] = {
        "count": int(array.size),
        "finite_count": int(finite.size),
        "mean": None,
        "median": None,
        "std": None,
        "sem": None,
        "rms": None,
        "min": None,
        "max": None,
    }
    if finite.size:
        result["mean"] = float(np.mean(finite))
        result["median"] = float(np.median(finite))
        result["rms"] = float(np.sqrt(np.mean(np.square(finite))))
        result["min"] = float(np.min(finite))
        result["max"] = float(np.max(finite))
        if finite.size > 1:
            std = float(np.std(finite, ddof=1))
            result["std"] = std
            result["sem"] = std / math.sqrt(float(finite.size))
        else:
            result["std"] = 0.0
            result["sem"] = None
    return result


def evaluate_events(events: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    n_events = 0
    n_selection_loss = 0
    n_decoded = 0
    n_sdo = 0
    n_truth = 0
    n_hits = 0
    n_tracks = 0
    for row in events:
        n_events += 1
        n_tracks += int(row.get("n_without_ift_tracks") or 0)
        if bool(row.get("selection_loss")) or int(row.get("n_without_ift_tracks") or 0) == 0:
            n_selection_loss += 1
        if bool(row.get("station_decoded_by_fasersct_id")):
            n_decoded += 1
        if bool(row.get("sdo_map_present")):
            n_sdo += 1
        if bool(row.get("truth_particles_present")):
            n_truth += 1
        if bool(row.get("sct_hits_present")):
            n_hits += 1
    return {
        "n_events": n_events,
        "n_events_selection_loss": n_selection_loss,
        "n_events_station_decoded": n_decoded,
        "n_events_sdo_map_present": n_sdo,
        "n_events_truth_particles_present": n_truth,
        "n_events_sct_hits_present": n_hits,
        "n_without_ift_tracks_from_events": n_tracks,
        "tracks_per_event": (
            float(n_tracks) / float(n_events) if n_events else 0.0
        ),
    }


def _primary_row(row: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any] | None:
    if str(row.get("collection")) != SOURCE_COLLECTION_NAME:
        return None
    if str(row.get("kind", "track")) != "track":
        return None
    if bool(row.get("is_truth", False)):
        return None
    q_fit = _finite(row.get("q_over_p_fit_per_mev"))
    if q_fit is None:
        state = np.asarray(row.get("native_state"), dtype=np.float64).reshape(-1)
        if state.size == 5 and np.isfinite(state).all():
            q_fit = float(state[QOVERP_INDEX])
    q_truth = _finite(row.get("q_over_p_truth_s1_per_mev"))
    sigma = _finite(row.get("sigma_q_over_p_per_mev"))
    if sigma is None:
        cov = _as_matrix(row.get("native_covariance"))
        if cov is not None and float(cov[QOVERP_INDEX, QOVERP_INDEX]) > 0.0:
            sigma = float(math.sqrt(float(cov[QOVERP_INDEX, QOVERP_INDEX])))
    if (
        q_fit is None
        or q_truth is None
        or sigma is None
        or sigma <= 0.0
        or not bool(row.get("truth_matched"))
        or not bool(row.get("truth_reference_available"))
    ):
        return None
    delta = q_fit - q_truth
    pull = delta / sigma
    p_truth = _finite(row.get("p_truth_s1_mev"))
    if p_truth is None and abs(q_truth) > 0.0:
        p_truth = abs(1.0 / q_truth)
    tx = _finite(row.get("tx_truth"))
    if tx is None:
        tx = _finite(row.get("tx"))
    ty = _finite(row.get("ty_truth"))
    if ty is None:
        ty = _finite(row.get("ty"))
    truth_charge = _finite(row.get("truth_charge"))
    if truth_charge is None:
        truth_charge = float(_sign(q_truth))
    return {
        "q_fit": q_fit,
        "q_truth": q_truth,
        "sigma": sigma,
        "delta": delta,
        "pull": pull,
        "sign_agree": _sign(q_fit) != 0 and _sign(q_fit) == _sign(q_truth),
        "p_truth": p_truth,
        "tx": tx,
        "ty": ty,
        "truth_charge": truth_charge,
        "abs_q_bin": assign_bin(abs(q_truth), ABS_QOVERP_EDGES, ABS_QOVERP_LABELS),
        "p_bin": assign_bin(p_truth if p_truth is not None else -1.0, P_TRUTH_EDGES_MEV, P_TRUTH_LABELS)
        if p_truth is not None
        else "unavailable",
        "tx_bin": assign_bin(tx, SLOPE_EDGES, TX_LABELS) if tx is not None else "unavailable",
        "ty_bin": assign_bin(ty, SLOPE_EDGES, TY_LABELS) if ty is not None else "unavailable",
        "high_momentum": (
            assign_bin(abs(q_truth), ABS_QOVERP_EDGES, ABS_QOVERP_LABELS) in HIGH_P_ABS_LABELS
            or (
                p_truth is not None
                and assign_bin(p_truth, P_TRUTH_EDGES_MEV, P_TRUTH_LABELS) in HIGH_P_P_LABELS
            )
        ),
    }


def _stratum_report(
    rows: Sequence[Mapping[str, Any]],
    key: str,
    labels: Sequence[str],
    gates: Mapping[str, Any],
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    min_n = int(gates["min_n_stratum"])
    limit = float(gates["stratum_abs_mean_pull_max"])
    for label in labels:
        selected = [row for row in rows if row.get(key) == label]
        pulls = [float(row["pull"]) for row in selected]
        deltas = [float(row["delta"]) for row in selected]
        stats = _finite_stats(pulls)
        delta_stats = _finite_stats(deltas)
        mean_z = stats["mean"]
        fail = (
            int(stats["finite_count"]) >= min_n
            and mean_z is not None
            and abs(float(mean_z)) > limit
        )
        report[label] = {
            "n": len(selected),
            "delta": delta_stats,
            "pull": stats,
            "sign_agreement": (
                float(np.mean([1.0 if row["sign_agree"] else 0.0 for row in selected]))
                if selected
                else None
            ),
            "kinematic_gate_applicable": int(stats["finite_count"]) >= min_n,
            "kinematic_gate_failed": bool(fail),
        }
    return report


def evaluate_physics(
    rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    gates = config["gates"]
    z68 = float(gates["coverage_z68"])
    z95 = float(gates["coverage_z95"])
    deltas = [float(row["delta"]) for row in rows]
    pulls = [float(row["pull"]) for row in rows]
    plus = [row for row in rows if float(row["truth_charge"]) > 0.0]
    minus = [row for row in rows if float(row["truth_charge"]) < 0.0]
    high = [row for row in rows if bool(row["high_momentum"])]
    delta_plus = _finite_stats([float(row["delta"]) for row in plus])
    delta_minus = _finite_stats([float(row["delta"]) for row in minus])
    even = None
    odd = None
    if delta_plus["mean"] is not None and delta_minus["mean"] is not None:
        even = 0.5 * (float(delta_plus["mean"]) + float(delta_minus["mean"]))
        odd = 0.5 * (float(delta_plus["mean"]) - float(delta_minus["mean"]))
    pull_plus = _finite_stats([float(row["pull"]) for row in plus])
    pull_minus = _finite_stats([float(row["pull"]) for row in minus])
    even_z = None
    odd_z = None
    if pull_plus["mean"] is not None and pull_minus["mean"] is not None:
        even_z = 0.5 * (float(pull_plus["mean"]) + float(pull_minus["mean"]))
        odd_z = 0.5 * (float(pull_plus["mean"]) - float(pull_minus["mean"]))
    coverage_68 = (
        float(np.mean([abs(float(row["pull"])) < z68 for row in rows])) if rows else None
    )
    coverage_95 = (
        float(np.mean([abs(float(row["pull"])) < z95 for row in rows])) if rows else None
    )
    sign_agreement = (
        float(np.mean([1.0 if row["sign_agree"] else 0.0 for row in rows])) if rows else None
    )
    high_sign = (
        float(np.mean([1.0 if row["sign_agree"] else 0.0 for row in high])) if high else None
    )
    return {
        "n_primary": len(rows),
        "n_mu_plus": len(plus),
        "n_mu_minus": len(minus),
        "n_high_momentum": len(high),
        "n_sign_flip": int(sum(0 if row["sign_agree"] else 1 for row in rows)),
        "delta_q_over_p": _finite_stats(deltas),
        "pull": _finite_stats(pulls),
        "coverage_68": coverage_68,
        "coverage_95": coverage_95,
        "sign_agreement": sign_agreement,
        "charge_plus": {"delta": delta_plus, "pull": pull_plus},
        "charge_minus": {"delta": delta_minus, "pull": pull_minus},
        "charge_even_delta": even,
        "charge_odd_delta": odd,
        "charge_even_pull": even_z,
        "charge_odd_pull": odd_z,
        "high_momentum": {
            "n": len(high),
            "delta": _finite_stats([float(row["delta"]) for row in high]),
            "pull": _finite_stats([float(row["pull"]) for row in high]),
            "sign_agreement": high_sign,
        },
        "given_q_over_p_truth": _stratum_report(rows, "abs_q_bin", ABS_QOVERP_LABELS, gates),
        "given_p_truth": _stratum_report(rows, "p_bin", P_TRUTH_LABELS, gates),
        "given_tx": _stratum_report(rows, "tx_bin", TX_LABELS, gates),
        "given_ty": _stratum_report(rows, "ty_bin", TY_LABELS, gates),
        "looked_at_mean_qoverp_only": False,
    }


def evaluate_tracks(
    records: Iterable[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    dummy_q = float(config["acceptance"]["dummy_qoverp_per_mev"])
    dummy_v = float(config["acceptance"]["dummy_variance_per_mev2"])
    tol = float(config["acceptance"]["dummy_match_tolerance"])
    dummy_q_hits = 0
    dummy_v_hits = 0
    truth_hits = 0
    truth_as_solution = 0
    missing_cov = 0
    spd_ok = 0
    cross_nonzero = 0
    ift_mot = 0
    ift_tsos = 0
    leak = 0
    decoded = 0
    unmatched = 0
    ref_unavailable = 0
    fit_failure = 0
    n = 0
    primary_rows: list[dict[str, Any]] = []
    for row in records:
        if str(row.get("collection")) != SOURCE_COLLECTION_NAME:
            continue
        if str(row.get("kind", "track")) != "track":
            continue
        n += 1
        if n % 250000 == 0:
            print(
                f"three_st_qp_calibration: evaluated {n} tracks",
                file=sys.stderr,
                flush=True,
            )
        if bool(row.get("is_truth", False)):
            truth_hits += 1
        if bool(row.get("truth_used_as_fit_seed")) or bool(row.get("truth_used_as_solution")):
            truth_as_solution += 1
        if bool(row.get("station_decoded_by_fasersct_id")):
            decoded += 1
        if int(row.get("n_ift_mot") or 0) > 0:
            ift_mot += 1
        if int(row.get("n_ift_tsos") or 0) > 0:
            ift_tsos += 1
        if bool(row.get("ift_leak")):
            leak += 1
        state = np.asarray(row.get("native_state"), dtype=np.float64).reshape(-1)
        q_value = _finite(row.get("q_over_p_fit_per_mev"))
        if q_value is None and state.size == 5 and np.isfinite(state).all():
            q_value = float(state[QOVERP_INDEX])
        if q_value is not None and abs(q_value - dummy_q) <= tol:
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
        if str(row.get("primary_failure_class")) == "fit_failure" or not bool(
            row.get("has_parameters", True)
        ):
            fit_failure += 1
        if not bool(row.get("truth_matched")):
            unmatched += 1
        elif not bool(row.get("truth_reference_available")):
            ref_unavailable += 1
        primary = _primary_row(row, config)
        if primary is not None:
            primary_rows.append(primary)
    physics = evaluate_physics(primary_rows, config)
    return {
        "collection": SOURCE_COLLECTION_NAME,
        "n_records": n,
        "n_truth_rejected": truth_hits,
        "n_truth_used_as_solution": truth_as_solution,
        "n_station_decoded": decoded,
        "n_tracks_with_ift_measurements_on_track": ift_mot,
        "n_tracks_with_ift_tsos": ift_tsos,
        "n_ift_leak": leak,
        "n_missing_covariance": missing_cov,
        "n_spd_ok": spd_ok,
        "n_dummy_qoverp": dummy_q_hits,
        "n_dummy_variance": dummy_v_hits,
        "n_nonzero_qoverp_cross_term": cross_nonzero,
        "n_unmatched": unmatched,
        "n_truth_reference_unavailable": ref_unavailable,
        "n_fit_failure": fit_failure,
        "n_primary": physics["n_primary"],
        "unmatched_rate": float(unmatched) / float(n) if n else 1.0,
        "fit_failure_rate": float(fit_failure + missing_cov) / float(n) if n else 1.0,
        "physics": physics,
        "primary_rows": primary_rows,
    }


def source_list_key(campaign: str, split: str) -> str:
    if campaign == "batch":
        return f"batch_{split}_sources"
    return f"{split}_sources"


def inventory_split(
    config: Mapping[str, Any],
    split: str,
    campaign: str,
) -> dict[str, Any]:
    sources: list[dict[str, Any]] = []
    dumps_present = True
    key = source_list_key(campaign, split)
    specs: list[tuple[Mapping[str, Any], Path, Path, Path]] = []
    for spec in config["mc_data"][key]:
        xaod = authorize_path(
            spec["input_xaod"],
            AccessScope.DEVELOPMENT_VALIDATION,
            split=access_split(split),
        )
        track_path = dump_path_for_source(config, spec["source_id"], "tracks", campaign)
        event_path = dump_path_for_source(config, spec["source_id"], "events", campaign)
        if not track_path.is_file() or not event_path.is_file():
            dumps_present = False
        specs.append((spec, xaod, track_path, event_path))

    event_counts: list[int] = []

    def _event_iter() -> Iterator[dict[str, Any]]:
        for spec, _xaod, _track_path, event_path in specs:
            n_event_rows = 0
            for row in iter_jsonl(event_path, split=split):
                n_event_rows += 1
                yield row
            event_counts.append(n_event_rows)

    def _track_iter() -> Iterator[dict[str, Any]]:
        for index, (spec, xaod, track_path, _event_path) in enumerate(specs):
            n_track_rows = 0
            for row in iter_jsonl(track_path, split=split):
                n_track_rows += 1
                yield row
            sources.append(
                {
                    "source_id": spec["source_id"],
                    "input_xaod": str(xaod),
                    "n_track_rows": n_track_rows,
                    "n_event_rows": event_counts[index] if index < len(event_counts) else 0,
                }
            )

    events = evaluate_events(_event_iter())
    tracks = evaluate_tracks(_track_iter(), config)
    return {
        "split": split,
        "campaign": campaign,
        "sources": sources,
        "dumps_present": dumps_present and bool(sources),
        "events": events,
        "tracks": tracks,
    }


def _physics_failures(
    pooled: Mapping[str, Any],
    construction: Mapping[str, Any],
    validation: Mapping[str, Any],
    config: Mapping[str, Any],
) -> list[str]:
    gates = config["gates"]
    failures: list[str] = []
    n_events = int(construction["events"]["n_events"]) + int(validation["events"]["n_events"])
    n_tracks = int(construction["tracks"]["n_records"]) + int(validation["tracks"]["n_records"])
    tracks_per_event = float(n_tracks) / float(n_events) if n_events else 0.0
    unmatched_rate = (
        float(construction["tracks"]["n_unmatched"] + validation["tracks"]["n_unmatched"])
        / float(n_tracks)
        if n_tracks
        else 1.0
    )
    fit_rate = (
        float(
            construction["tracks"]["n_fit_failure"]
            + construction["tracks"]["n_missing_covariance"]
            + validation["tracks"]["n_fit_failure"]
            + validation["tracks"]["n_missing_covariance"]
        )
        / float(n_tracks)
        if n_tracks
        else 1.0
    )
    if tracks_per_event < float(gates["min_tracks_per_event"]) or unmatched_rate > float(
        gates["max_unmatched_rate"]
    ):
        failures.append(MECHANISM_SELECTION)
    if fit_rate > float(gates["max_fit_failure_rate"]):
        failures.append(MECHANISM_FIT)
    physics = pooled
    n = int(physics["n_primary"])
    sign = physics["sign_agreement"]
    if sign is not None:
        if n >= 20 and float(sign) < float(gates["sign_agreement_hard_min"]):
            failures.append(MECHANISM_SIGN)
        elif n >= 50 and float(sign) < float(gates["sign_agreement_min"]):
            failures.append(MECHANISM_SIGN)
    high = physics["high_momentum"]
    if (
        int(high["n"]) >= int(gates["min_n_high_momentum"])
        and high["sign_agreement"] is not None
        and float(high["sign_agreement"]) < float(gates["high_momentum_sign_agreement_min"])
    ):
        if MECHANISM_SIGN not in failures:
            failures.append(MECHANISM_SIGN)
    mean_z = physics["pull"]["mean"]
    if (
        n >= int(gates["min_tracks_for_physics_verdict"])
        and mean_z is not None
        and abs(float(mean_z)) > float(gates["abs_mean_pull_max"])
    ):
        failures.append(MECHANISM_MEAN_BIAS)
    kinematic = False
    for family in ("given_q_over_p_truth", "given_p_truth", "given_tx", "given_ty"):
        for item in physics[family].values():
            if bool(item.get("kinematic_gate_failed")):
                kinematic = True
    high_mean = high["pull"]["mean"]
    if (
        int(high["n"]) >= int(gates["min_n_high_momentum"])
        and high_mean is not None
        and abs(float(high_mean)) > float(gates["high_momentum_abs_mean_pull_max"])
    ):
        kinematic = True
    if kinematic:
        failures.append(MECHANISM_KINEMATIC)
    if n >= int(gates["min_n_coverage"]):
        cov68 = physics["coverage_68"]
        cov95 = physics["coverage_95"]
        rms = physics["pull"]["rms"]
        bad_cov = False
        if cov68 is None or not (
            float(gates["coverage_68_min"]) <= float(cov68) <= float(gates["coverage_68_max"])
        ):
            bad_cov = True
        if cov95 is None or not (
            float(gates["coverage_95_min"]) <= float(cov95) <= float(gates["coverage_95_max"])
        ):
            bad_cov = True
        if rms is None or not (
            float(gates["pull_rms_min"]) <= float(rms) <= float(gates["pull_rms_max"])
        ):
            bad_cov = True
        if bad_cov:
            failures.append(MECHANISM_COV)
    return failures


def decide(
    construction: Mapping[str, Any],
    validation: Mapping[str, Any],
    pins: Mapping[str, Any],
    inherited: Mapping[str, Any],
    *,
    dumps_materialized: bool,
    campaign: str,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    gates = config["gates"]
    min_tracks = int(gates["min_tracks_per_split_contract"])
    min_matched = int(gates["min_matched_per_split_contract"])
    mechanism = None
    contract = "FAIL"
    if inherited.get("workbook_117", {}).get("decision") != (
        "ckf_without_ift_definition_established"
    ):
        mechanism = MECHANISM_STAGE0
    elif inherited.get("workbook_118", {}).get("decision") != (
        "three_st_to_ift_prediction_established"
    ):
        mechanism = MECHANISM_STAGE1
    elif not pins.get("all_match", False):
        mechanism = MECHANISM_SOURCE_PIN
    elif not dumps_materialized:
        mechanism = MECHANISM_DUMP_MISSING
    elif int(construction["tracks"]["n_truth_rejected"]) or int(
        validation["tracks"]["n_truth_rejected"]
    ) or int(construction["tracks"]["n_truth_used_as_solution"]) or int(
        validation["tracks"]["n_truth_used_as_solution"]
    ):
        mechanism = MECHANISM_TRUTH
    elif int(construction["tracks"]["n_ift_leak"]) or int(validation["tracks"]["n_ift_leak"]):
        mechanism = MECHANISM_IFT_LEAK
    elif int(construction["tracks"]["n_tracks_with_ift_measurements_on_track"]) or int(
        construction["tracks"]["n_tracks_with_ift_tsos"]
    ) or int(validation["tracks"]["n_tracks_with_ift_measurements_on_track"]) or int(
        validation["tracks"]["n_tracks_with_ift_tsos"]
    ):
        mechanism = MECHANISM_IFT_LEAK
    elif int(construction["tracks"]["n_records"]) < min_tracks or int(
        validation["tracks"]["n_records"]
    ) < min_tracks:
        mechanism = MECHANISM_NO_TRACKS
    elif int(construction["tracks"]["n_dummy_qoverp"]) == int(
        construction["tracks"]["n_records"]
    ) and int(validation["tracks"]["n_dummy_qoverp"]) == int(
        validation["tracks"]["n_records"]
    ):
        mechanism = MECHANISM_DUMMY
    elif int(construction["tracks"]["n_missing_covariance"]) or int(
        validation["tracks"]["n_missing_covariance"]
    ):
        mechanism = MECHANISM_NO_5X5
    elif int(construction["tracks"]["n_primary"]) < min_matched or int(
        validation["tracks"]["n_primary"]
    ) < min_matched:
        if int(construction["tracks"]["n_unmatched"]) == int(
            construction["tracks"]["n_records"]
        ) and int(validation["tracks"]["n_unmatched"]) == int(
            validation["tracks"]["n_records"]
        ):
            mechanism = MECHANISM_NO_MATCH
        else:
            mechanism = MECHANISM_NO_REF
    else:
        contract = "PASS"
        mechanism = None

    pooled_rows = []
    # Physics is recomputed from already-evaluated summaries when possible;
    # tests pass precomputed track evaluations, so re-pool from those.
    pooled = {
        "n_primary": int(construction["tracks"]["n_primary"])
        + int(validation["tracks"]["n_primary"]),
        "n_mu_plus": int(construction["tracks"]["physics"]["n_mu_plus"])
        + int(validation["tracks"]["physics"]["n_mu_plus"]),
        "n_mu_minus": int(construction["tracks"]["physics"]["n_mu_minus"])
        + int(validation["tracks"]["physics"]["n_mu_minus"]),
        "n_high_momentum": int(construction["tracks"]["physics"]["n_high_momentum"])
        + int(validation["tracks"]["physics"]["n_high_momentum"]),
        "n_sign_flip": int(construction["tracks"]["physics"]["n_sign_flip"])
        + int(validation["tracks"]["physics"]["n_sign_flip"]),
        "delta_q_over_p": construction["tracks"]["physics"]["delta_q_over_p"],
        "pull": construction["tracks"]["physics"]["pull"],
        "coverage_68": construction["tracks"]["physics"]["coverage_68"],
        "coverage_95": construction["tracks"]["physics"]["coverage_95"],
        "sign_agreement": construction["tracks"]["physics"]["sign_agreement"],
        "charge_plus": construction["tracks"]["physics"]["charge_plus"],
        "charge_minus": validation["tracks"]["physics"]["charge_minus"]
        if validation["tracks"]["physics"]["n_mu_minus"]
        else construction["tracks"]["physics"]["charge_minus"],
        "charge_even_delta": None,
        "charge_odd_delta": None,
        "charge_even_pull": None,
        "charge_odd_pull": None,
        "high_momentum": construction["tracks"]["physics"]["high_momentum"],
        "given_q_over_p_truth": construction["tracks"]["physics"]["given_q_over_p_truth"],
        "given_p_truth": construction["tracks"]["physics"]["given_p_truth"],
        "given_tx": construction["tracks"]["physics"]["given_tx"],
        "given_ty": construction["tracks"]["physics"]["given_ty"],
        "looked_at_mean_qoverp_only": False,
    }
    # Re-evaluate physics on the union when the caller supplied raw-capable
    # summaries.  The audit path always goes through inventory_split, which
    # already computed per-split physics; the official pooled numbers are
    # recomputed below if both splits expose a hidden row cache.
    if "primary_rows" in construction["tracks"] and "primary_rows" in validation["tracks"]:
        pooled_rows = list(construction["tracks"]["primary_rows"]) + list(
            validation["tracks"]["primary_rows"]
        )
        pooled = evaluate_physics(pooled_rows, config)
    else:
        # Merge charge-even/odd from the two splits when they are charge-separated.
        plus_mean = construction["tracks"]["physics"]["charge_plus"]["delta"]["mean"]
        minus_mean = validation["tracks"]["physics"]["charge_minus"]["delta"]["mean"]
        if plus_mean is None:
            plus_mean = validation["tracks"]["physics"]["charge_plus"]["delta"]["mean"]
        if minus_mean is None:
            minus_mean = construction["tracks"]["physics"]["charge_minus"]["delta"]["mean"]
        if plus_mean is not None and minus_mean is not None:
            pooled["charge_even_delta"] = 0.5 * (float(plus_mean) + float(minus_mean))
            pooled["charge_odd_delta"] = 0.5 * (float(plus_mean) - float(minus_mean))
        plus_z = construction["tracks"]["physics"]["charge_plus"]["pull"]["mean"]
        minus_z = validation["tracks"]["physics"]["charge_minus"]["pull"]["mean"]
        if plus_z is None:
            plus_z = validation["tracks"]["physics"]["charge_plus"]["pull"]["mean"]
        if minus_z is None:
            minus_z = construction["tracks"]["physics"]["charge_minus"]["pull"]["mean"]
        if plus_z is not None and minus_z is not None:
            pooled["charge_even_pull"] = 0.5 * (float(plus_z) + float(minus_z))
            pooled["charge_odd_pull"] = 0.5 * (float(plus_z) - float(minus_z))
        # Weighted coverage / sign from the two splits.
        n_c = int(construction["tracks"]["physics"]["n_primary"])
        n_v = int(validation["tracks"]["physics"]["n_primary"])
        n_tot = n_c + n_v
        if n_tot:
            def _wavg(a: Any, b: Any) -> float | None:
                if a is None and b is None:
                    return None
                va = 0.0 if a is None else float(a) * n_c
                vb = 0.0 if b is None else float(b) * n_v
                den = (n_c if a is not None else 0) + (n_v if b is not None else 0)
                return va / den + vb / den if den else None

            pooled["coverage_68"] = _wavg(
                construction["tracks"]["physics"]["coverage_68"],
                validation["tracks"]["physics"]["coverage_68"],
            )
            pooled["coverage_95"] = _wavg(
                construction["tracks"]["physics"]["coverage_95"],
                validation["tracks"]["physics"]["coverage_95"],
            )
            pooled["sign_agreement"] = _wavg(
                construction["tracks"]["physics"]["sign_agreement"],
                validation["tracks"]["physics"]["sign_agreement"],
            )
            # Merge pull mean/rms approximately by concatenating is not possible
            # here; prefer the larger split's pull stats only as a fallback.
            # Audit always supplies primary_rows after inventory.  Tests that
            # need exact pooled physics attach primary_rows.
            c_pull = construction["tracks"]["physics"]["pull"]
            v_pull = validation["tracks"]["physics"]["pull"]
            if c_pull["mean"] is not None and v_pull["mean"] is not None:
                pooled["pull"] = {
                    **c_pull,
                    "mean": (float(c_pull["mean"]) * n_c + float(v_pull["mean"]) * n_v)
                    / float(n_tot),
                    "rms": math.sqrt(
                        (
                            (float(c_pull["rms"] or 0.0) ** 2) * n_c
                            + (float(v_pull["rms"] or 0.0) ** 2) * n_v
                        )
                        / float(n_tot)
                    ),
                    "finite_count": n_tot,
                    "count": n_tot,
                }
            # Merge strata kinematic flags (OR).
            for family in ("given_q_over_p_truth", "given_p_truth", "given_tx", "given_ty"):
                merged = {}
                keys = set(construction["tracks"]["physics"][family]) | set(
                    validation["tracks"]["physics"][family]
                )
                for label in keys:
                    left = construction["tracks"]["physics"][family].get(label, {})
                    right = validation["tracks"]["physics"][family].get(label, {})
                    n_left = int(left.get("n") or 0)
                    n_right = int(right.get("n") or 0)
                    n_s = n_left + n_right
                    mean_z = None
                    if left.get("pull", {}).get("mean") is not None or right.get("pull", {}).get(
                        "mean"
                    ) is not None:
                        num = 0.0
                        den = 0
                        for part, nn in ((left, n_left), (right, n_right)):
                            mz = part.get("pull", {}).get("mean")
                            if mz is not None and nn:
                                num += float(mz) * nn
                                den += nn
                        mean_z = num / den if den else None
                    failed = bool(left.get("kinematic_gate_failed")) or bool(
                        right.get("kinematic_gate_failed")
                    )
                    if (
                        n_s >= int(gates["min_n_stratum"])
                        and mean_z is not None
                        and abs(float(mean_z)) > float(gates["stratum_abs_mean_pull_max"])
                    ):
                        failed = True
                    merged[label] = {
                        "n": n_s,
                        "pull": {"mean": mean_z, "finite_count": n_s},
                        "kinematic_gate_applicable": n_s >= int(gates["min_n_stratum"]),
                        "kinematic_gate_failed": failed,
                    }
                pooled[family] = merged
            high_c = construction["tracks"]["physics"]["high_momentum"]
            high_v = validation["tracks"]["physics"]["high_momentum"]
            n_h = int(high_c["n"]) + int(high_v["n"])
            pooled["high_momentum"] = {
                "n": n_h,
                "pull": {
                    "mean": (
                        (
                            (float(high_c["pull"]["mean"]) * int(high_c["n"]))
                            if high_c["pull"]["mean"] is not None
                            else 0.0
                        )
                        + (
                            (float(high_v["pull"]["mean"]) * int(high_v["n"]))
                            if high_v["pull"]["mean"] is not None
                            else 0.0
                        )
                    )
                    / float(n_h)
                    if n_h
                    and (
                        high_c["pull"]["mean"] is not None or high_v["pull"]["mean"] is not None
                    )
                    else None
                },
                "sign_agreement": (
                    (
                        (float(high_c["sign_agreement"]) * int(high_c["n"]))
                        if high_c["sign_agreement"] is not None
                        else 0.0
                    )
                    + (
                        (float(high_v["sign_agreement"]) * int(high_v["n"]))
                        if high_v["sign_agreement"] is not None
                        else 0.0
                    )
                )
                / float(
                    (int(high_c["n"]) if high_c["sign_agreement"] is not None else 0)
                    + (int(high_v["n"]) if high_v["sign_agreement"] is not None else 0)
                )
                if (
                    (high_c["sign_agreement"] is not None and int(high_c["n"]))
                    or (high_v["sign_agreement"] is not None and int(high_v["n"]))
                )
                else None,
            }

    n_primary = int(pooled["n_primary"])
    physics_failures: list[str] = []
    physics_verdict = "INCONCLUSIVE"
    if contract == "PASS":
        if n_primary < int(gates["min_tracks_for_physics_verdict"]):
            physics_verdict = "INCONCLUSIVE"
        else:
            physics_failures = _physics_failures(pooled, construction, validation, config)
            physics_verdict = "FAIL" if physics_failures else "PASS"

    if contract != "PASS":
        decision = DECISION_NOT_ESTABLISHED
        verdict = "FAIL"
    elif physics_verdict == "PASS":
        decision = DECISION_ESTABLISHED
        verdict = "PASS"
        mechanism = None
    elif physics_verdict == "INCONCLUSIVE":
        decision = DECISION_CONTRACT
        verdict = "PASS"
        mechanism = None
    else:
        decision = DECISION_NOT_ESTABLISHED
        verdict = "FAIL"
        mechanism = physics_failures[0] if physics_failures else MECHANISM_MEAN_BIAS

    usable = decision == DECISION_ESTABLISHED
    return {
        "kind": "three_st_qp_calibration_contract",
        "task": TASK,
        "workbook": WORKBOOK,
        "campaign": campaign,
        "verdict": verdict,
        "decision": decision,
        "mechanism": mechanism,
        "contract_verdict": contract,
        "physics_verdict": physics_verdict,
        "failure_classes": physics_failures if contract == "PASS" else [mechanism],
        "source_collection": SOURCE_COLLECTION_NAME,
        "official_truth_station": OFFICIAL_TRUTH_STATION,
        "state_definition": state_definition(),
        "truth_definition": truth_definition(),
        "primary_quantities": primary_quantities(),
        "lto_is_not_this_collection": True,
        "four_station_ckf_is_not_qp_source": True,
        "truth_is_calibration_reference_only": True,
        "truth_used_as_fit_seed": False,
        "covariance_rescaled": False,
        "sign_flip_tracks_dropped": False,
        "high_momentum_tail_dropped": False,
        "favorable_momentum_range_picked": False,
        "ad_hoc_prior_introduced": False,
        "looked_at_mean_qoverp_only": False,
        "batch_htcondor_authorized": contract == "PASS",
        "residual_conditional_authorized": usable,
        "three_st_qp_trusted_observable": usable,
        "measurement_model_v2_entered": False,
        "geometry_write_allowed": False,
        "held_out_accessed": False,
        "real_data_alignment_authorized": False,
        "b14m_reopen_authorized": False,
        "b15_authorized": False,
        "dummy_qoverp_per_mev": SEGMENTFIT_DUMMY_QOVERP_PER_MEV,
        "physics": pooled,
        "construction": {
            "n_records": construction["tracks"]["n_records"],
            "n_primary": construction["tracks"]["n_primary"],
            "n_unmatched": construction["tracks"]["n_unmatched"],
            "n_events_selection_loss": construction["events"]["n_events_selection_loss"],
            "physics": construction["tracks"]["physics"],
        },
        "validation": {
            "n_records": validation["tracks"]["n_records"],
            "n_primary": validation["tracks"]["n_primary"],
            "n_unmatched": validation["tracks"]["n_unmatched"],
            "n_events_selection_loss": validation["events"]["n_events_selection_loss"],
            "physics": validation["tracks"]["physics"],
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
