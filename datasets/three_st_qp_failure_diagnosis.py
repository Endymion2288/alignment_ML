"""Yasu-S2D: diagnose why WB119 3ST q/p calibration failed.

Reuses the frozen WB119 dumps and denominator.  Does not flip S2 to
PASS, does not authorize S3, and does not drop sign-flips or tails.
Truth remains a calibration reference only.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
import yaml

from alignment.operating_protocol_v1_final_closure import (
    project_root,
    resolve_under_root,
    sha256_file,
)
from datasets.access_policy import AccessScope, authorize_path
from datasets.three_st_qp_calibration import (
    ABS_QOVERP_EDGES,
    ABS_QOVERP_LABELS,
    OFFICIAL_TRUTH_STATION,
    P_TRUTH_EDGES_MEV,
    P_TRUTH_LABELS,
    QOVERP_INDEX,
    SLOPE_EDGES,
    SOURCE_COLLECTION_NAME,
    STATION_Z_MM,
    TX_LABELS,
    TY_LABELS,
    access_split,
    assign_bin,
    dump_path_for_source,
    evaluate_events,
    iter_jsonl,
    provenance_hashes,
    source_list_key,
    verify_pinned_calypso_sources,
)

SCHEMA_VERSION = "three-st-qp-failure-diagnosis-v1"
DEFAULT_CONFIG = "configs/three_st_qp_failure_diagnosis_v1.yaml"
TASK = "YASU-S2D"
WORKBOOK = 120

DECISION_CONTRACT = "three_st_qp_failure_diagnosis_contract_established"
DECISION_RECORDED = "three_st_qp_failure_diagnosis_recorded"
DECISION_NOT_ESTABLISHED = "three_st_qp_failure_diagnosis_not_established"

MECHANISM_CURVATURE = "curvature_information_loss"
MECHANISM_COVARIANCE = "covariance_semantics_failure"
MECHANISM_CHARGE = "charge_asymmetric_reconstruction"
MECHANISM_REFERENCE = "reference_definition_error"
MECHANISM_MIXED = "mixed/inconclusive"

MECHANISM_STAGE2 = "s2_not_inherited_as_fail"
MECHANISM_DUMP_MISSING = "calibration_dump_not_materialized"
MECHANISM_DENOMINATOR = "wb119_denominator_drift"
MECHANISM_SOURCE_PIN = "pinned_calypso_source_hash_mismatch"
MECHANISM_NO_TRACKS = "no_without_ift_candidate"
MECHANISM_S2_FLIP = "s2_trusted_observable_reopened"

NATIVE_NAMES = ("loc1", "loc2", "phi", "theta", "q_over_p")
SIGNIFICANCE_EDGES = (0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 1.0e12)
SIGNIFICANCE_LABELS = (
    "sig_lt_0p5",
    "sig_0p5_to_1",
    "sig_1_to_2",
    "sig_2_to_3",
    "sig_3_to_5",
    "sig_5_to_10",
    "sig_ge_10",
)
N_MOT_EDGES = (0.0, 6.0, 12.0, 15.0, 18.0, 19.0)
N_MOT_LABELS = ("mot_lt_6", "mot_6_11", "mot_12_14", "mot_15_17", "mot_18")
CHI2_EDGES = (0.0, 0.5, 1.0, 2.0, 5.0, 1.0e6)
CHI2_LABELS = (
    "chi2ndof_lt_0p5",
    "chi2ndof_0p5_to_1",
    "chi2ndof_1_to_2",
    "chi2ndof_2_to_5",
    "chi2ndof_ge_5",
)
OUTLIER_EDGES = (0.0, 1.0, 3.0, 100.0)
OUTLIER_LABELS = ("outlier_0", "outlier_1_2", "outlier_ge_3")
MISSING_LABELS = (
    "complete_s1s2s3",
    "missing_s1",
    "missing_s2",
    "missing_s3",
    "missing_two_or_more",
    "other",
)


class ThreeStQpFailureDiagnosisError(ValueError):
    """Raised when the S2D diagnosis contract is illegal."""


def refuse_truth_qoverp() -> None:
    raise ThreeStQpFailureDiagnosisError(
        "truth q/p is a calibration reference only; it is not a fit seed, "
        "prior, prediction, or real-data solution"
    )


def refuse_drop_sign_flip() -> None:
    raise ThreeStQpFailureDiagnosisError(
        "sign-flip tracks stay in the sample; they are a failure class"
    )


def refuse_covariance_rescale() -> None:
    raise ThreeStQpFailureDiagnosisError(
        "covariance rescale is forbidden; miscalibration is a failure class"
    )


def refuse_residual_conditional() -> None:
    raise ThreeStQpFailureDiagnosisError(
        "E[r_IFT|q/p] is Stage 3; S2D does not open residual conditionals"
    )


def refuse_flip_s2() -> None:
    raise ThreeStQpFailureDiagnosisError(
        "S2D cannot set three_st_qp_trusted_observable true or flip WB119"
    )


def primary_quantities() -> dict[str, str]:
    return {
        "sign_flip_rate_vs_p": "P(sign flip | p_truth bin); keep all flips",
        "sign_flip_rate_vs_significance": "P(sign flip | |q_truth|/sigma)",
        "mean_sigma_vs_p": "E[sigma(q/p) | p_truth bin]",
        "mean_significance_vs_p": "E[|q_truth|/sigma | p_truth bin]",
        "pull_rms_and_coverage": "distinguish mean-wrong from sigma-wrong",
        "native_correlation_qp": "corr(q/p, loc1/loc2/phi/theta)",
        "covariance_condition": "cond(native 5x5) and min eigenvalue",
        "fisher_qq": "I_qq = (C^{-1})_{44} when invertible",
        "charge_even_odd": "0.5*(mu+ +/- mu-) of delta and pull",
        "reference_identities": "q*p, S1 recompute, front z, units",
        "topology_conditionals": "P(flip | n_mot, missing station, chi2/ndof)",
        "seed_qp": "not materialized in WB119 dump",
    }


def _require_edges(config: Mapping[str, Any], key: str, expected: Sequence[float]) -> None:
    got = [float(x) for x in config["binning"][key]]
    if len(got) != len(expected) or any(abs(a - b) > 0.0 for a, b in zip(got, expected)):
        raise ThreeStQpFailureDiagnosisError(f"binning.{key} is frozen and must not change")


def _require_labels(config: Mapping[str, Any], key: str, expected: Sequence[str]) -> None:
    got = [str(x) for x in config["binning"][key]]
    if got != list(expected):
        raise ThreeStQpFailureDiagnosisError(f"binning.{key} is frozen and must not change")


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = resolve_under_root(project_root(), str(path or DEFAULT_CONFIG))
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ThreeStQpFailureDiagnosisError(f"schema_version must be {SCHEMA_VERSION}")
    if config.get("task") != TASK:
        raise ThreeStQpFailureDiagnosisError(f"task must be {TASK}")
    if int(config.get("workbook", -1)) != WORKBOOK:
        raise ThreeStQpFailureDiagnosisError(f"workbook must be {WORKBOOK}")
    if str(config.get("source_collection")) != SOURCE_COLLECTION_NAME:
        raise ThreeStQpFailureDiagnosisError(
            "source_collection must be CKFTrackCollectionWithoutIFT"
        )
    if int(config.get("official_truth_station", -1)) != OFFICIAL_TRUTH_STATION:
        raise ThreeStQpFailureDiagnosisError("official_truth_station must be 1 (S1)")
    if bool(config.get("three_st_qp_trusted_observable", True)):
        raise ThreeStQpFailureDiagnosisError("three_st_qp_trusted_observable must stay false")
    if bool(config.get("residual_conditional_authorized", True)):
        raise ThreeStQpFailureDiagnosisError("residual_conditional_authorized must stay false")
    for key in (
        "do_not_drop_sign_flip_tracks",
        "do_not_drop_high_momentum_tail",
        "do_not_rescale_covariance",
        "do_not_flip_s2_to_pass",
        "do_not_set_trusted_observable_true",
        "do_not_enter_residual_conditional",
        "do_not_enter_alignment",
        "do_not_pick_favorable_momentum_range",
        "do_not_introduce_ad_hoc_prior",
        "do_not_replace_fit_qoverp_with_truth",
    ):
        if bool(config.get(key, False)) is not True:
            raise ThreeStQpFailureDiagnosisError(f"{key} must be true")
    _require_edges(config, "abs_q_over_p_per_mev_edges", ABS_QOVERP_EDGES)
    _require_labels(config, "abs_q_over_p_labels", ABS_QOVERP_LABELS)
    _require_edges(config, "p_truth_mev_edges", P_TRUTH_EDGES_MEV)
    _require_labels(config, "p_truth_labels", P_TRUTH_LABELS)
    _require_edges(config, "tx_edges", SLOPE_EDGES)
    _require_labels(config, "tx_labels", TX_LABELS)
    _require_edges(config, "ty_edges", SLOPE_EDGES)
    _require_labels(config, "ty_labels", TY_LABELS)
    _require_edges(config, "significance_edges", SIGNIFICANCE_EDGES)
    _require_labels(config, "significance_labels", SIGNIFICANCE_LABELS)
    if len(config.get("focus_identities") or []) != 6:
        raise ThreeStQpFailureDiagnosisError("focus_identities must stay the six WB119 smoke rows")
    return dict(config)


def _expect_sha(path: Path, expected: str, label: str) -> None:
    digest = sha256_file(path)
    if digest != expected:
        raise ThreeStQpFailureDiagnosisError(f"{label} hash mismatch: {digest}")


def inherit_frozen_stage(config: Mapping[str, Any]) -> dict[str, Any]:
    inherited: dict[str, Any] = {}
    frozen_root = Path(str(config.get("frozen_artifact_root") or project_root()))
    for workbook, extra in (
        ("workbook_87", {"must_not_reopen_transport": True}),
        ("workbook_95", {"dummy_still_rejected": True}),
        ("workbook_96", {"four_station_5x5_is_not_without_ift": True}),
        ("workbook_109", {"lto_is_not_without_ift": True}),
        ("workbook_117", {"authorizes_without_ift_definition_only": True}),
        ("workbook_118", {"authorizes_prediction_chain_only": True}),
        ("workbook_119", {"s2_fail_is_frozen": True}),
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
            raise ThreeStQpFailureDiagnosisError(f"{workbook} decision must stay frozen")
        inherited[workbook] = {
            "decision": decision["decision"],
            "config_sha256": spec["config_sha256"],
            "decision_sha256": spec["decision_sha256"],
            "artifact_root": str(root),
            **extra,
        }
    s2 = inherited["workbook_119"]
    if s2["decision"] != "three_st_qp_calibration_not_established":
        raise ThreeStQpFailureDiagnosisError("WB119 must remain not_established")
    return inherited


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


def _empty_bin() -> dict[str, Any]:
    return {
        "n": 0,
        "n_flip": 0,
        "n_cov68": 0,
        "n_cov95": 0,
        "n_median_abs_z_le_0p5": 0,
        "sum_delta": 0.0,
        "sumsq_delta": 0.0,
        "sum_pull": 0.0,
        "sumsq_pull": 0.0,
        "sum_sigma": 0.0,
        "sum_significance": 0.0,
        "sum_abs_corr_max": 0.0,
        "sum_log_cond": 0.0,
        "n_cond": 0,
        "n_min_eig_le_0": 0,
        "n_plus": 0,
        "n_minus": 0,
        "n_flip_plus": 0,
        "n_flip_minus": 0,
        "sum_delta_plus": 0.0,
        "sum_delta_minus": 0.0,
        "sum_pull_plus": 0.0,
        "sum_pull_minus": 0.0,
    }


def _add_to_bin(bin_acc: dict[str, Any], row: Mapping[str, Any]) -> None:
    bin_acc["n"] += 1
    if row["flip"]:
        bin_acc["n_flip"] += 1
    if row["cov68"]:
        bin_acc["n_cov68"] += 1
    if row["cov95"]:
        bin_acc["n_cov95"] += 1
    if row["median_abs_z_le_0p5"]:
        bin_acc["n_median_abs_z_le_0p5"] += 1
    bin_acc["sum_delta"] += row["delta"]
    bin_acc["sumsq_delta"] += row["delta"] * row["delta"]
    bin_acc["sum_pull"] += row["pull"]
    bin_acc["sumsq_pull"] += row["pull"] * row["pull"]
    bin_acc["sum_sigma"] += row["sigma"]
    bin_acc["sum_significance"] += row["significance"]
    bin_acc["sum_abs_corr_max"] += row["abs_corr_max"]
    if row["log_cond"] is not None:
        bin_acc["sum_log_cond"] += row["log_cond"]
        bin_acc["n_cond"] += 1
    if row["min_eig_le_0"]:
        bin_acc["n_min_eig_le_0"] += 1
    if row["charge"] > 0.0:
        bin_acc["n_plus"] += 1
        bin_acc["sum_delta_plus"] += row["delta"]
        bin_acc["sum_pull_plus"] += row["pull"]
        if row["flip"]:
            bin_acc["n_flip_plus"] += 1
    elif row["charge"] < 0.0:
        bin_acc["n_minus"] += 1
        bin_acc["sum_delta_minus"] += row["delta"]
        bin_acc["sum_pull_minus"] += row["pull"]
        if row["flip"]:
            bin_acc["n_flip_minus"] += 1


def _merge_bin(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    out = dict(left)
    for key, value in right.items():
        if key == "n":
            out["n"] = int(left["n"]) + int(right["n"])
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            out[key] = left[key] + value
    return out


def _summarize_bin(bin_acc: Mapping[str, Any]) -> dict[str, Any]:
    n = int(bin_acc["n"])
    if n <= 0:
        return {
            "n": 0,
            "flip_rate": None,
            "coverage_68": None,
            "coverage_95": None,
            "mean_delta": None,
            "rms_delta": None,
            "mean_pull": None,
            "rms_pull": None,
            "mean_sigma": None,
            "mean_significance": None,
            "mean_abs_corr_max": None,
            "mean_log_cond": None,
            "n_min_eig_le_0": 0,
            "median_abs_z_le_0p5_fraction": None,
            "n_plus": 0,
            "n_minus": 0,
            "flip_rate_plus": None,
            "flip_rate_minus": None,
            "mean_delta_plus": None,
            "mean_delta_minus": None,
            "mean_pull_plus": None,
            "mean_pull_minus": None,
        }
    def _rate(num: int, den: int) -> float | None:
        return float(num) / float(den) if den else None

    return {
        "n": n,
        "n_flip": int(bin_acc["n_flip"]),
        "flip_rate": _rate(int(bin_acc["n_flip"]), n),
        "coverage_68": _rate(int(bin_acc["n_cov68"]), n),
        "coverage_95": _rate(int(bin_acc["n_cov95"]), n),
        "mean_delta": float(bin_acc["sum_delta"]) / n,
        "rms_delta": math.sqrt(float(bin_acc["sumsq_delta"]) / n),
        "mean_pull": float(bin_acc["sum_pull"]) / n,
        "rms_pull": math.sqrt(float(bin_acc["sumsq_pull"]) / n),
        "mean_sigma": float(bin_acc["sum_sigma"]) / n,
        "mean_significance": float(bin_acc["sum_significance"]) / n,
        "mean_abs_corr_max": float(bin_acc["sum_abs_corr_max"]) / n,
        "mean_log_cond": (
            float(bin_acc["sum_log_cond"]) / float(bin_acc["n_cond"])
            if int(bin_acc["n_cond"])
            else None
        ),
        "n_min_eig_le_0": int(bin_acc["n_min_eig_le_0"]),
        "median_abs_z_le_0p5_fraction": _rate(int(bin_acc["n_median_abs_z_le_0p5"]), n),
        "n_plus": int(bin_acc["n_plus"]),
        "n_minus": int(bin_acc["n_minus"]),
        "flip_rate_plus": _rate(int(bin_acc["n_flip_plus"]), int(bin_acc["n_plus"])),
        "flip_rate_minus": _rate(int(bin_acc["n_flip_minus"]), int(bin_acc["n_minus"])),
        "mean_delta_plus": (
            float(bin_acc["sum_delta_plus"]) / float(bin_acc["n_plus"])
            if int(bin_acc["n_plus"])
            else None
        ),
        "mean_delta_minus": (
            float(bin_acc["sum_delta_minus"]) / float(bin_acc["n_minus"])
            if int(bin_acc["n_minus"])
            else None
        ),
        "mean_pull_plus": (
            float(bin_acc["sum_pull_plus"]) / float(bin_acc["n_plus"])
            if int(bin_acc["n_plus"])
            else None
        ),
        "mean_pull_minus": (
            float(bin_acc["sum_pull_minus"]) / float(bin_acc["n_minus"])
            if int(bin_acc["n_minus"])
            else None
        ),
    }


def _family(labels: Sequence[str]) -> dict[str, dict[str, Any]]:
    return {label: _empty_bin() for label in labels}


def empty_accumulator() -> dict[str, Any]:
    return {
        "n_records": 0,
        "n_primary": 0,
        "n_unmatched": 0,
        "n_flip": 0,
        "n_truth_as_solution": 0,
        "n_dummy_qoverp": 0,
        "n_seed_present": 0,
        "n_identity_fit_fail": 0,
        "n_identity_truth_fail": 0,
        "n_identity_recompute_fail": 0,
        "n_official_station_fail": 0,
        "n_ift_front": 0,
        "n_production_cov68": 0,
        "n_production_cov95": 0,
        "n_production_primary": 0,
        "sumsq_pull_production": 0.0,
        "n_gev_rescue_cov68": 0,
        "global": _empty_bin(),
        "given_p": _family(P_TRUTH_LABELS),
        "given_q": _family(ABS_QOVERP_LABELS),
        "given_tx": _family(TX_LABELS),
        "given_ty": _family(TY_LABELS),
        "given_significance": _family(SIGNIFICANCE_LABELS),
        "given_n_mot": _family(N_MOT_LABELS),
        "given_chi2ndof": _family((*CHI2_LABELS, "ndof_zero")),
        "given_outlier": _family(OUTLIER_LABELS),
        "given_missing": _family(MISSING_LABELS),
        "given_source": {},
        "focus_hits": [],
    }


def _merge_family(
    left: Mapping[str, dict[str, Any]], right: Mapping[str, dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    keys = set(left) | set(right)
    return {
        key: _merge_bin(left.get(key, _empty_bin()), right.get(key, _empty_bin()))
        for key in keys
    }


def merge_accumulators(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    out = empty_accumulator()
    for key in (
        "n_records",
        "n_primary",
        "n_unmatched",
        "n_flip",
        "n_truth_as_solution",
        "n_dummy_qoverp",
        "n_seed_present",
        "n_identity_fit_fail",
        "n_identity_truth_fail",
        "n_identity_recompute_fail",
        "n_official_station_fail",
        "n_ift_front",
        "n_production_cov68",
        "n_production_cov95",
        "n_production_primary",
        "sumsq_pull_production",
        "n_gev_rescue_cov68",
    ):
        out[key] = left[key] + right[key]
    out["global"] = _merge_bin(left["global"], right["global"])
    for family in (
        "given_p",
        "given_q",
        "given_tx",
        "given_ty",
        "given_significance",
        "given_n_mot",
        "given_chi2ndof",
        "given_outlier",
        "given_missing",
        "given_source",
    ):
        out[family] = _merge_family(left[family], right[family])
    out["focus_hits"] = list(left["focus_hits"]) + list(right["focus_hits"])
    return out


def _missing_label(stations: Sequence[int]) -> str:
    have = {int(s) for s in stations if int(s) in (1, 2, 3)}
    missing = {1, 2, 3} - have
    if not missing:
        return "complete_s1s2s3"
    if len(missing) >= 2:
        return "missing_two_or_more"
    only = next(iter(missing))
    return f"missing_s{only}"


def _native_diagnostics(cov_payload: Any, sigma: float) -> tuple[float, float | None, bool, float | None]:
    abs_corr_max = 0.0
    log_cond = None
    min_eig_le_0 = False
    fisher_qq = None
    if cov_payload is None:
        return abs_corr_max, log_cond, min_eig_le_0, fisher_qq
    cov = np.asarray(cov_payload, dtype=np.float64)
    if cov.shape != (5, 5) or not np.isfinite(cov).all():
        return abs_corr_max, log_cond, min_eig_le_0, fisher_qq
    diag = np.diag(cov)
    if sigma > 0.0:
        for idx in range(4):
            if diag[idx] > 0.0:
                corr = abs(float(cov[idx, QOVERP_INDEX]) / math.sqrt(float(diag[idx]) * sigma * sigma))
                if corr > abs_corr_max:
                    abs_corr_max = corr
    symmetric = 0.5 * (cov + cov.T)
    try:
        eig = np.linalg.eigvalsh(symmetric)
    except np.linalg.LinAlgError:
        return abs_corr_max, log_cond, True, fisher_qq
    if not np.isfinite(eig).all():
        return abs_corr_max, log_cond, True, fisher_qq
    min_eig = float(np.min(eig))
    max_eig = float(np.max(eig))
    min_eig_le_0 = min_eig <= 0.0
    if min_eig > 0.0:
        log_cond = math.log(max_eig / min_eig)
        try:
            inv = np.linalg.inv(symmetric)
            fisher_qq = float(inv[QOVERP_INDEX, QOVERP_INDEX])
        except np.linalg.LinAlgError:
            fisher_qq = None
    return abs_corr_max, log_cond, min_eig_le_0, fisher_qq


def _recomputed_q_truth(row: Mapping[str, Any]) -> float | None:
    charge = _finite(row.get("truth_charge"))
    stations = row.get("truth_station_momenta")
    if charge is None or not isinstance(stations, list) or len(stations) < 2:
        return None
    s1 = stations[1]
    if not isinstance(s1, Mapping):
        return None
    p_s1 = _finite(s1.get("p"))
    if p_s1 is None or p_s1 <= 0.0:
        return None
    return charge / p_s1


def _focus_key(row: Mapping[str, Any]) -> tuple[str, int, int, int]:
    return (
        str(row.get("source_id") or ""),
        int(row.get("run_id") or 0),
        int(row.get("event_id") or 0),
        int(row.get("track_index") or 0),
    )


def _focus_set(config: Mapping[str, Any]) -> set[tuple[str, int, int, int]]:
    keys = set()
    for item in config["focus_identities"]:
        keys.add(
            (
                str(item["source_id"]),
                int(item["run_id"]),
                int(item["event_id"]),
                int(item["track_index"]),
            )
        )
    return keys


def process_track(
    row: Mapping[str, Any],
    acc: dict[str, Any],
    config: Mapping[str, Any],
    focus_keys: set[tuple[str, int, int, int]],
) -> None:
    gates = config["gates"]
    if str(row.get("collection")) != SOURCE_COLLECTION_NAME:
        return
    if str(row.get("kind", "track")) != "track":
        return
    acc["n_records"] += 1
    if bool(row.get("truth_used_as_fit_seed")) or bool(row.get("truth_used_as_solution")):
        acc["n_truth_as_solution"] += 1
    if row.get("q_over_p_seed_per_mev") is not None:
        acc["n_seed_present"] += 1
    dummy = float(config["acceptance"]["dummy_qoverp_per_mev"])
    q_fit = _finite(row.get("q_over_p_fit_per_mev"))
    if q_fit is not None and abs(q_fit - dummy) <= float(config["acceptance"]["dummy_match_tolerance"]):
        acc["n_dummy_qoverp"] += 1
    if not bool(row.get("truth_matched")):
        acc["n_unmatched"] += 1
        return
    q_truth = _finite(row.get("q_over_p_truth_s1_per_mev"))
    sigma = _finite(row.get("sigma_q_over_p_per_mev"))
    if q_fit is None or q_truth is None or sigma is None or sigma <= 0.0:
        return
    if not bool(row.get("truth_reference_available")):
        return
    p_truth = _finite(row.get("p_truth_s1_mev"))
    p_fit = _finite(row.get("p_fit_mev"))
    charge = _finite(row.get("truth_charge"))
    if charge is None:
        charge = float(_sign(q_truth))
    if p_fit is not None and abs(abs(q_fit) * p_fit - 1.0) > float(gates["identity_qp_p_tolerance"]):
        acc["n_identity_fit_fail"] += 1
    if p_truth is not None and abs(abs(q_truth) * p_truth - 1.0) > float(gates["identity_qp_p_tolerance"]):
        acc["n_identity_truth_fail"] += 1
    recomputed = _recomputed_q_truth(row)
    if recomputed is None or abs(recomputed - q_truth) > float(gates["identity_truth_recompute_tolerance"]):
        acc["n_identity_recompute_fail"] += 1
    if int(row.get("official_truth_station") or -1) != OFFICIAL_TRUTH_STATION:
        acc["n_official_station_fail"] += 1
    z_mm = _finite(row.get("z_mm"))
    if z_mm is not None and z_mm < float(gates["ift_z_mm_max"]):
        acc["n_ift_front"] += 1
    q_prod = _finite(row.get("q_over_p_truth_production_per_mev"))
    if q_prod is not None:
        z_prod = (q_fit - q_prod) / sigma
        acc["n_production_primary"] += 1
        acc["sumsq_pull_production"] += z_prod * z_prod
        if abs(z_prod) < float(gates["coverage_z68"]):
            acc["n_production_cov68"] += 1
        if abs(z_prod) < float(gates["coverage_z95"]):
            acc["n_production_cov95"] += 1
    z_gev = (q_fit - q_truth / 1000.0) / sigma
    if abs(z_gev) < float(gates["coverage_z68"]):
        acc["n_gev_rescue_cov68"] += 1

    delta = q_fit - q_truth
    pull = delta / sigma
    flip = _sign(q_fit) == 0 or _sign(q_truth) == 0 or _sign(q_fit) != _sign(q_truth)
    significance = abs(q_truth) / sigma
    tx = _finite(row.get("tx_truth"))
    if tx is None:
        tx = _finite(row.get("tx"))
    ty = _finite(row.get("ty_truth"))
    if ty is None:
        ty = _finite(row.get("ty"))
    stations = [int(s) for s in (row.get("measurements_on_track_stations") or [])]
    abs_corr_max, log_cond, min_eig_le_0, _fisher = _native_diagnostics(
        row.get("native_covariance"), sigma
    )
    chi2 = _finite(row.get("chi2"))
    ndof = _finite(row.get("ndof"))
    if ndof is None or ndof <= 0.0:
        chi_label = "ndof_zero"
    else:
        chi_label = assign_bin(float(chi2 or 0.0) / ndof, CHI2_EDGES, CHI2_LABELS)
    payload = {
        "delta": delta,
        "pull": pull,
        "sigma": sigma,
        "significance": significance,
        "flip": flip,
        "cov68": abs(pull) < float(gates["coverage_z68"]),
        "cov95": abs(pull) < float(gates["coverage_z95"]),
        "median_abs_z_le_0p5": abs(pull) <= float(gates["median_abs_z_near_zero"]),
        "charge": charge,
        "abs_corr_max": abs_corr_max,
        "log_cond": log_cond,
        "min_eig_le_0": min_eig_le_0,
    }
    acc["n_primary"] += 1
    if flip:
        acc["n_flip"] += 1
    _add_to_bin(acc["global"], payload)

    def _put(family: str, label: str) -> None:
        acc[family].setdefault(label, _empty_bin())
        _add_to_bin(acc[family][label], payload)

    if p_truth is not None:
        _put("given_p", assign_bin(p_truth, P_TRUTH_EDGES_MEV, P_TRUTH_LABELS))
    _put("given_q", assign_bin(abs(q_truth), ABS_QOVERP_EDGES, ABS_QOVERP_LABELS))
    if tx is not None:
        _put("given_tx", assign_bin(tx, SLOPE_EDGES, TX_LABELS))
    if ty is not None:
        _put("given_ty", assign_bin(ty, SLOPE_EDGES, TY_LABELS))
    _put("given_significance", assign_bin(significance, SIGNIFICANCE_EDGES, SIGNIFICANCE_LABELS))
    _put("given_n_mot", assign_bin(float(row.get("n_mot") or 0), N_MOT_EDGES, N_MOT_LABELS))
    _put("given_chi2ndof", chi_label)
    _put(
        "given_outlier",
        assign_bin(float(row.get("n_outlier_hits") or 0), OUTLIER_EDGES, OUTLIER_LABELS),
    )
    _add_to_bin(acc["given_missing"][_missing_label(stations)], payload)
    source_id = str(row.get("source_id") or "unknown")
    acc["given_source"].setdefault(source_id, _empty_bin())
    _add_to_bin(acc["given_source"][source_id], payload)
    key = _focus_key(row)
    if key in focus_keys:
        acc["focus_hits"].append(
            {
                "source_id": key[0],
                "run_id": key[1],
                "event_id": key[2],
                "track_index": key[3],
                "q_fit": q_fit,
                "q_truth": q_truth,
                "sigma": sigma,
                "pull": pull,
                "significance": significance,
                "flip": flip,
                "p_truth_s1_mev": p_truth,
                "p_fit_mev": p_fit,
                "n_mot": int(row.get("n_mot") or 0),
                "chi2": chi2,
                "ndof": ndof,
                "z_mm": z_mm,
                "missing": _missing_label(stations),
                "abs_corr_max": abs_corr_max,
                "log_cond": log_cond,
            }
        )


def evaluate_tracks(
    records: Iterable[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    acc = empty_accumulator()
    focus_keys = _focus_set(config)
    n = 0
    for row in records:
        n += 1
        if n % 250000 == 0:
            print(
                f"three_st_qp_failure_diagnosis: evaluated {n} rows",
                file=sys.stderr,
                flush=True,
            )
        process_track(row, acc, config, focus_keys)
    return acc


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
        for _spec, _xaod, _track_path, event_path in specs:
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


def _flip(summary: Mapping[str, Any] | None) -> float | None:
    if not summary:
        return None
    return summary.get("flip_rate")


def _n(summary: Mapping[str, Any] | None) -> int:
    return int(summary["n"]) if summary else 0


def evaluate_mechanisms(acc: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    gates = config["gates"]
    overall = _summarize_bin(acc["global"])
    given_p = {key: _summarize_bin(val) for key, val in acc["given_p"].items()}
    given_sig = {key: _summarize_bin(val) for key, val in acc["given_significance"].items()}
    given_tx = {key: _summarize_bin(val) for key, val in acc["given_tx"].items()}
    given_ty = {key: _summarize_bin(val) for key, val in acc["given_ty"].items()}
    given_q = {key: _summarize_bin(val) for key, val in acc["given_q"].items()}
    given_mot = {key: _summarize_bin(val) for key, val in acc["given_n_mot"].items()}
    given_chi = {key: _summarize_bin(val) for key, val in acc["given_chi2ndof"].items()}
    given_out = {key: _summarize_bin(val) for key, val in acc["given_outlier"].items()}
    given_miss = {key: _summarize_bin(val) for key, val in acc["given_missing"].items()}
    given_src = {key: _summarize_bin(val) for key, val in acc["given_source"].items()}

    high = given_p.get("p_ge_2000gev") or {}
    low = given_p.get("p_lt_200gev") or {}
    mid = given_p.get("p_200_500gev") or {}
    sig_low = _summarize_bin(_merge_bin(acc["given_significance"]["sig_lt_0p5"], acc["given_significance"]["sig_0p5_to_1"]))
    sig_high = given_sig.get("sig_ge_10") or {}
    n_high = _n(high)
    n_low = _n(low)
    curvature_applicable = n_high >= int(gates["min_n_high_momentum"]) and n_low >= int(
        gates["min_n_stratum"]
    )
    flip_gap = None
    if high.get("flip_rate") is not None and low.get("flip_rate") is not None:
        flip_gap = float(high["flip_rate"]) - float(low["flip_rate"])
    sig_gap = None
    if sig_low.get("flip_rate") is not None and sig_high.get("flip_rate") is not None:
        sig_gap = float(sig_low["flip_rate"]) - float(sig_high["flip_rate"])
    surprising = given_sig.get("sig_ge_5") or given_sig.get("sig_5_to_10") or {}
    # Frozen union of sig >= 5.
    surprising_acc = _merge_bin(acc["given_significance"]["sig_5_to_10"], acc["given_significance"]["sig_ge_10"])
    surprising = _summarize_bin(surprising_acc)
    curvature_support = False
    curvature_reject = False
    if curvature_applicable:
        if (
            flip_gap is not None
            and flip_gap >= float(gates["flip_rate_high_minus_low_min"])
            and sig_gap is not None
            and sig_gap >= float(gates["flip_rate_low_sig_minus_high_sig_min"])
            and high.get("mean_significance") is not None
            and float(high["mean_significance"]) <= float(gates["high_p_mean_significance_max"])
        ):
            curvature_support = True
        if surprising.get("flip_rate") is not None and _n(surprising) >= int(gates["min_n_stratum"]):
            if float(surprising["flip_rate"]) >= float(gates["surprising_flip_rate_reject"]):
                curvature_reject = True
        if flip_gap is not None and flip_gap < 0.0:
            curvature_reject = True
    if curvature_reject:
        curvature_support = False

    n = int(overall["n"])
    cov_applicable = n >= int(gates["min_n_mechanism_verdict"])
    rms = overall.get("rms_pull")
    cov68 = overall.get("coverage_68")
    cov95 = overall.get("coverage_95")
    median_near = overall.get("median_abs_z_le_0p5_fraction")
    mean_z = overall.get("mean_pull")
    miscal = False
    if rms is not None and float(rms) > float(gates["pull_rms_calibrated_max"]):
        miscal = True
    if cov68 is not None and float(cov68) < float(gates["coverage_68_calibrated_min"]):
        miscal = True
    if cov95 is not None and float(cov95) < float(gates["coverage_95_calibrated_min"]):
        miscal = True
    sigma_wrong = bool(
        miscal and median_near is not None and float(median_near) >= 0.50
    )
    mean_wrong = bool(
        mean_z is not None
        and abs(float(mean_z)) > float(gates["median_abs_z_near_zero"])
        and rms is not None
        and float(rms) <= float(gates["pull_rms_calibrated_max"])
    )
    both_wrong = bool(
        mean_z is not None
        and abs(float(mean_z)) > float(gates["median_abs_z_near_zero"])
        and miscal
    )
    covariance_support = bool(cov_applicable and (sigma_wrong or mean_wrong or both_wrong or miscal))
    covariance_kind = (
        "both"
        if both_wrong
        else "sigma_wrong"
        if sigma_wrong
        else "mean_wrong"
        if mean_wrong
        else "miscalibrated"
        if miscal
        else "not_supported"
    )

    n_plus = int(acc["global"]["n_plus"])
    n_minus = int(acc["global"]["n_minus"])
    charge_applicable = n_plus >= int(gates["min_n_charge"]) and n_minus >= int(
        gates["min_n_charge"]
    )
    even_delta = None
    odd_delta = None
    even_z = None
    odd_z = None
    if overall.get("mean_delta_plus") is not None and overall.get("mean_delta_minus") is not None:
        even_delta = 0.5 * (float(overall["mean_delta_plus"]) + float(overall["mean_delta_minus"]))
        odd_delta = 0.5 * (float(overall["mean_delta_plus"]) - float(overall["mean_delta_minus"]))
    if overall.get("mean_pull_plus") is not None and overall.get("mean_pull_minus") is not None:
        even_z = 0.5 * (float(overall["mean_pull_plus"]) + float(overall["mean_pull_minus"]))
        odd_z = 0.5 * (float(overall["mean_pull_plus"]) - float(overall["mean_pull_minus"]))
    charge_sign_gap = None
    if overall.get("flip_rate_plus") is not None and overall.get("flip_rate_minus") is not None:
        charge_sign_gap = abs(float(overall["flip_rate_plus"]) - float(overall["flip_rate_minus"]))
    p_bin_gap = False
    for label, item in given_p.items():
        if item["n_plus"] >= 50 and item["n_minus"] >= 50:
            if item["flip_rate_plus"] is not None and item["flip_rate_minus"] is not None:
                if abs(float(item["flip_rate_plus"]) - float(item["flip_rate_minus"])) >= float(
                    gates["charge_sign_rate_gap_min"]
                ):
                    p_bin_gap = True
    charge_support = False
    if charge_applicable and odd_z is not None and even_z is not None:
        if abs(float(odd_z)) >= float(gates["odd_abs_pull_min"]) and abs(float(odd_z)) > abs(
            float(even_z)
        ):
            if p_bin_gap or (
                charge_sign_gap is not None
                and charge_sign_gap >= float(gates["charge_sign_rate_gap_min"])
            ):
                charge_support = True
    if charge_applicable and odd_z is not None and even_z is not None:
        if abs(float(odd_z)) <= abs(float(even_z)):
            charge_support = False

    n_id = int(acc["n_primary"])
    id_tol_frac = float(gates["identity_fail_fraction_max"])
    identity_fail = False
    if n_id:
        if float(acc["n_identity_fit_fail"]) / n_id > id_tol_frac:
            identity_fail = True
        if float(acc["n_identity_truth_fail"]) / n_id > id_tol_frac:
            identity_fail = True
        if float(acc["n_identity_recompute_fail"]) / n_id > id_tol_frac:
            identity_fail = True
        if acc["n_official_station_fail"] or acc["n_ift_front"]:
            identity_fail = True
    production_cov68 = (
        float(acc["n_production_cov68"]) / float(acc["n_production_primary"])
        if acc["n_production_primary"]
        else None
    )
    production_rms = (
        math.sqrt(float(acc["sumsq_pull_production"]) / float(acc["n_production_primary"]))
        if acc["n_production_primary"]
        else None
    )
    production_rescues = bool(
        production_cov68 is not None
        and production_rms is not None
        and float(production_cov68) >= float(gates["coverage_68_calibrated_min"])
        and float(production_rms) <= float(gates["pull_rms_calibrated_max"])
        and miscal
    )
    gev_rescue = bool(
        n_id
        and float(acc["n_gev_rescue_cov68"]) / n_id >= float(gates["coverage_68_calibrated_min"])
        and miscal
    )
    reference_support = bool(identity_fail or production_rescues or gev_rescue)

    topology = {**given_mot, **given_chi, **given_out, **given_miss}
    concentrated = []
    if overall.get("flip_rate") is not None:
        for label, item in topology.items():
            if item["n"] >= int(gates["min_n_stratum"]) and item["flip_rate"] is not None:
                excess = float(item["flip_rate"]) - float(overall["flip_rate"])
                if excess >= float(gates["topology_flip_excess_min"]):
                    concentrated.append({"label": label, "n": item["n"], "flip_rate": item["flip_rate"], "excess": excess})
    topology_support = bool(concentrated)

    supported = []
    if curvature_support:
        supported.append(MECHANISM_CURVATURE)
    if covariance_support:
        supported.append(MECHANISM_COVARIANCE)
    if charge_support:
        supported.append(MECHANISM_CHARGE)
    if reference_support:
        supported.append(MECHANISM_REFERENCE)

    if reference_support:
        verdict = MECHANISM_REFERENCE
    elif len([m for m in supported if m != MECHANISM_REFERENCE]) >= 2:
        verdict = MECHANISM_MIXED
    elif supported == [MECHANISM_CURVATURE]:
        verdict = MECHANISM_CURVATURE
    elif supported == [MECHANISM_COVARIANCE]:
        verdict = MECHANISM_COVARIANCE
    elif supported == [MECHANISM_CHARGE]:
        verdict = MECHANISM_CHARGE
    else:
        verdict = MECHANISM_MIXED

    excluded = []
    if not curvature_support and curvature_reject:
        excluded.append(MECHANISM_CURVATURE)
    if not covariance_support and cov_applicable:
        excluded.append(MECHANISM_COVARIANCE)
    if not charge_support and charge_applicable:
        excluded.append(MECHANISM_CHARGE)
    if not reference_support and n_id >= int(gates["min_n_mechanism_verdict"]):
        excluded.append(MECHANISM_REFERENCE)

    seed_present = int(acc["n_seed_present"]) > 0
    new_dump = bool(
        verdict == MECHANISM_MIXED
        and not curvature_support
        and not covariance_support
        and not reference_support
        and not seed_present
    )

    return {
        "mechanism_verdict": verdict,
        "supported_mechanisms": supported,
        "excluded_mechanisms": excluded,
        "curvature": {
            "applicable": curvature_applicable,
            "supported": curvature_support,
            "rejected": curvature_reject,
            "flip_rate_high_minus_low": flip_gap,
            "flip_rate_low_sig_minus_high_sig": sig_gap,
            "high_p_mean_significance": high.get("mean_significance"),
            "high_p_flip_rate": high.get("flip_rate"),
            "low_p_flip_rate": low.get("flip_rate"),
            "mid_p_flip_rate": mid.get("flip_rate"),
            "surprising_flip_rate_sig_ge_5": surprising.get("flip_rate"),
            "surprising_n": surprising.get("n"),
        },
        "covariance": {
            "applicable": cov_applicable,
            "supported": covariance_support,
            "kind": covariance_kind,
            "mean_pull": mean_z,
            "rms_pull": rms,
            "coverage_68": cov68,
            "coverage_95": cov95,
            "median_abs_z_le_0p5_fraction": median_near,
            "mean_abs_corr_max": overall.get("mean_abs_corr_max"),
            "mean_log_cond": overall.get("mean_log_cond"),
            "n_min_eig_le_0": overall.get("n_min_eig_le_0"),
        },
        "charge": {
            "applicable": charge_applicable,
            "supported": charge_support,
            "n_plus": n_plus,
            "n_minus": n_minus,
            "even_delta": even_delta,
            "odd_delta": odd_delta,
            "even_pull": even_z,
            "odd_pull": odd_z,
            "flip_rate_plus": overall.get("flip_rate_plus"),
            "flip_rate_minus": overall.get("flip_rate_minus"),
            "charge_sign_gap": charge_sign_gap,
            "p_bin_sign_gap": p_bin_gap,
            "not_a_magnetic_weak_mode": True,
        },
        "reference": {
            "supported": reference_support,
            "n_identity_fit_fail": int(acc["n_identity_fit_fail"]),
            "n_identity_truth_fail": int(acc["n_identity_truth_fail"]),
            "n_identity_recompute_fail": int(acc["n_identity_recompute_fail"]),
            "n_official_station_fail": int(acc["n_official_station_fail"]),
            "n_ift_front": int(acc["n_ift_front"]),
            "production_coverage_68": production_cov68,
            "production_pull_rms": production_rms,
            "production_rescues_s1": production_rescues,
            "gev_unit_rescues": gev_rescue,
        },
        "topology": {
            "supported_concentration": topology_support,
            "concentrated_bins": concentrated,
            "not_a_post_hoc_cut": True,
        },
        "seed_leave_one_station": {
            "seed_qp_materialized": seed_present,
            "leave_one_station_materialized": False,
            "new_reconstruction_dump_authorized": new_dump,
        },
        "overall": overall,
        "given_p": given_p,
        "given_q": given_q,
        "given_tx": given_tx,
        "given_ty": given_ty,
        "given_significance": given_sig,
        "given_n_mot": given_mot,
        "given_chi2ndof": given_chi,
        "given_outlier": given_out,
        "given_missing": given_miss,
        "given_source": given_src,
        "focus_hits": list(acc["focus_hits"]),
        "n_primary": int(acc["n_primary"]),
        "n_records": int(acc["n_records"]),
        "n_flip": int(acc["n_flip"]),
        "n_unmatched": int(acc["n_unmatched"]),
        "n_truth_as_solution": int(acc["n_truth_as_solution"]),
    }


def _denominator_ok(acc: Mapping[str, Any], campaign: str, config: Mapping[str, Any]) -> bool:
    if campaign == "batch":
        frozen = config["frozen_wb119_batch_denominator"]
        return (
            int(acc["n_primary"]) == int(frozen["n_primary"])
            and int(acc["n_flip"]) == int(frozen["n_sign_flip"])
            and int(acc["n_records"]) == int(frozen["n_without_ift_tracks"])
        )
    frozen = config["frozen_wb119_smoke_denominator"]
    return int(acc["n_primary"]) == int(frozen["n_primary"]) and int(acc["n_flip"]) == int(
        frozen["n_sign_flip"]
    )


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
    mechanism = None
    contract = "FAIL"
    if inherited.get("workbook_119", {}).get("decision") != (
        "three_st_qp_calibration_not_established"
    ):
        mechanism = MECHANISM_STAGE2
    elif not pins.get("all_match", False):
        mechanism = MECHANISM_SOURCE_PIN
    elif not dumps_materialized:
        mechanism = MECHANISM_DUMP_MISSING
    elif int(construction["tracks"]["n_truth_as_solution"]) or int(
        validation["tracks"]["n_truth_as_solution"]
    ):
        mechanism = MECHANISM_S2_FLIP
    else:
        pooled_acc = merge_accumulators(construction["tracks"], validation["tracks"])
        if int(pooled_acc["n_records"]) < int(gates["min_tracks_per_split_contract"]):
            mechanism = MECHANISM_NO_TRACKS
        elif not _denominator_ok(pooled_acc, campaign, config):
            mechanism = MECHANISM_DENOMINATOR
        else:
            contract = "PASS"
            mechanism = None

    if contract == "PASS":
        pooled_acc = merge_accumulators(construction["tracks"], validation["tracks"])
    else:
        pooled_acc = merge_accumulators(construction["tracks"], validation["tracks"])
    diagnosis = evaluate_mechanisms(pooled_acc, config)
    n_primary = int(diagnosis["n_primary"])
    if contract != "PASS":
        decision = DECISION_NOT_ESTABLISHED
        verdict = "FAIL"
        mechanism_verdict = None
        diagnosis_verdict = "FAIL"
    elif n_primary < int(gates["min_n_mechanism_verdict"]):
        decision = DECISION_CONTRACT
        verdict = "PASS"
        mechanism_verdict = None
        diagnosis_verdict = "INCONCLUSIVE"
        mechanism = None
    else:
        decision = DECISION_RECORDED
        verdict = "PASS"
        mechanism_verdict = diagnosis["mechanism_verdict"]
        diagnosis_verdict = "RECORDED"
        mechanism = mechanism_verdict

    return {
        "kind": "three_st_qp_failure_diagnosis_contract",
        "task": TASK,
        "workbook": WORKBOOK,
        "campaign": campaign,
        "verdict": verdict,
        "decision": decision,
        "mechanism": mechanism,
        "contract_verdict": contract,
        "diagnosis_verdict": diagnosis_verdict,
        "mechanism_verdict": mechanism_verdict,
        "supported_mechanisms": diagnosis["supported_mechanisms"] if contract == "PASS" else [],
        "excluded_mechanisms": diagnosis["excluded_mechanisms"] if contract == "PASS" else [],
        "failure_classes": [mechanism] if contract != "PASS" and mechanism else [],
        "source_collection": SOURCE_COLLECTION_NAME,
        "official_truth_station": OFFICIAL_TRUTH_STATION,
        "primary_quantities": primary_quantities(),
        "truth_is_calibration_reference_only": True,
        "wb119_decision_remains": "three_st_qp_calibration_not_established",
        "three_st_qp_trusted_observable": False,
        "residual_conditional_authorized": False,
        "s2_flipped_to_pass": False,
        "sign_flip_tracks_dropped": False,
        "high_momentum_tail_dropped": False,
        "covariance_rescaled": False,
        "favorable_momentum_range_picked": False,
        "ad_hoc_prior_introduced": False,
        "new_reconstruction_dump_authorized": bool(
            diagnosis["seed_leave_one_station"]["new_reconstruction_dump_authorized"]
            and contract == "PASS"
            and diagnosis_verdict == "RECORDED"
        ),
        "measurement_model_v2_entered": False,
        "geometry_write_allowed": False,
        "held_out_accessed": False,
        "real_data_alignment_authorized": False,
        "b14m_reopen_authorized": False,
        "b15_authorized": False,
        "diagnosis": diagnosis,
        "construction": {
            "n_records": construction["tracks"]["n_records"],
            "n_primary": construction["tracks"]["n_primary"],
            "n_flip": construction["tracks"]["n_flip"],
            "n_events": construction["events"]["n_events"],
        },
        "validation": {
            "n_records": validation["tracks"]["n_records"],
            "n_primary": validation["tracks"]["n_primary"],
            "n_flip": validation["tracks"]["n_flip"],
            "n_events": validation["events"]["n_events"],
        },
    }
