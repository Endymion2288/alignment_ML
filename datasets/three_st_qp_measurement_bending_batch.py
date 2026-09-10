"""Yasu-S2J: batch calibration of WB125 bending_raw and CKF mapping audit.

Reuses the frozen YZ bending_raw constructor.  Fitted q/p and truth
never enter construction.  Does not flip S2 or open Stage 3.
"""

from __future__ import annotations

import json
from collections import defaultdict
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
    iter_jsonl,
    verify_pinned_calypso_sources,
)
from datasets.three_st_qp_measurement_bending import (
    CLASS_A,
    CLASS_B,
    CLASS_C,
    CLASS_D,
    THREE_STATION_IDS,
    centroids_from_row,
    classify_topology,
    construct_bending,
    conversion_chain as s2i_conversion_chain,
    official_quantities as s2i_official_quantities,
)

SCHEMA_VERSION = "three-st-qp-measurement-bending-batch-v1"
DEFAULT_CONFIG = "configs/three_st_qp_measurement_bending_batch_v1.yaml"
TASK = "YASU-S2J"
WORKBOOK = 126

DECISION_CONTRACT = "three_st_qp_measurement_bending_batch_contract_established"
DECISION_RECORDED = "three_st_qp_measurement_bending_batch_recorded"
DECISION_NOT = "three_st_qp_measurement_bending_batch_not_established"

STATUS_SUPPORTED = "supported"
STATUS_REJECTED = "rejected"
STATUS_UNRESOLVED = "unresolved"

MECH_PRESENT = "raw_bending_information_present"
MECH_LIMITED = "raw_bending_high_p_limit"
MECH_FIT = "fit_additional_sign_failure"
MECH_SOURCE = "raw_measurement_source_dependence"
MECH_MAP = "mapping_nontransferability"
MECH_MIXED = "mixed/inconclusive"

REQUIRED_INHERITANCE = (
    "workbook_117",
    "workbook_118",
    "workbook_119",
    "workbook_120",
    "workbook_121",
    "workbook_122",
    "workbook_123",
    "workbook_124",
    "workbook_125",
)

CONSTRUCTION_RUNS = frozenset({"100043", "100044"})
VALIDATION_RUNS = frozenset({"100047", "100048"})

SLIM_TRACK_KEYS = (
    "kind",
    "collection",
    "source_id",
    "n_mot",
    "n_ift_mot",
    "ift_leak",
    "measurements_on_track_stations",
    "station_centroids",
    "q_over_p_fit_per_mev",
    "q_over_p_truth_s1_per_mev",
    "p_truth_s1_mev",
    "truth_charge",
    "sigma_q_over_p_per_mev",
    "skip_index",
    "event_id",
    "run_id",
    "track_index",
    "centroid_source",
    "truth_matched",
    "truth_reference_available",
)


class ThreeStQpMeasurementBendingBatchError(ValueError):
    """Raised when the S2J contract is illegal."""


def refuse_flip_s2() -> None:
    raise ThreeStQpMeasurementBendingBatchError(
        "S2J cannot set three_st_qp_trusted_observable true"
    )


def refuse_residual_conditional() -> None:
    raise ThreeStQpMeasurementBendingBatchError(
        "E[r_IFT|q/p] and E[r_IFT|bending] are Stage 3; S2J does not open them"
    )


def refuse_change_bending() -> None:
    raise ThreeStQpMeasurementBendingBatchError(
        "S2J must not change the frozen WB125 YZ bending_raw definition"
    )


def refuse_qp_like() -> None:
    raise ThreeStQpMeasurementBendingBatchError(
        "S2J must not convert bending_raw into an official q/p-like observable"
    )


def refuse_invent_sigma_b() -> None:
    raise ThreeStQpMeasurementBendingBatchError(
        "S2J must not invent sigma_b; point-estimator calibration only"
    )


def conversion_chain() -> dict[str, Any]:
    chain = s2i_conversion_chain()
    chain["inherited_observable"] = "workbook_125 bending_raw unchanged"
    chain["tx_ty_matching"] = "measurement chord (x3-x1)/(z3-z1), (y3-y1)/(z3-z1)"
    chain["qp_like"] = "still unauthorized; field-contract is read-only"
    return chain


def official_quantities() -> dict[str, str]:
    quantities = s2i_official_quantities()
    quantities["p_fit_flip_given_bending_correct"] = (
        "Pr(fitted q/p flip | sign(bending_raw)==sign(truth))"
    )
    quantities["p_raw_bending_flip"] = "Pr(sign(bending_raw)!=sign(truth))"
    quantities["wb119_class_a_fraction"] = (
        "share of fitted sign-flips that already have correct raw bending"
    )
    quantities["wb119_class_c_fraction"] = (
        "share of fitted sign-flips that already flip at measurement level"
    )
    return quantities


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = resolve_under_root(project_root(), str(path or DEFAULT_CONFIG))
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ThreeStQpMeasurementBendingBatchError(f"schema_version must be {SCHEMA_VERSION}")
    if config.get("task") != TASK:
        raise ThreeStQpMeasurementBendingBatchError(f"task must be {TASK}")
    if int(config.get("workbook", -1)) != WORKBOOK:
        raise ThreeStQpMeasurementBendingBatchError("workbook must be 126")
    if bool(config.get("three_st_qp_trusted_observable", True)):
        raise ThreeStQpMeasurementBendingBatchError("three_st_qp_trusted_observable must stay false")
    if bool(config.get("residual_conditional_authorized", True)):
        raise ThreeStQpMeasurementBendingBatchError("residual_conditional_authorized must stay false")
    if str(config.get("bending_plane")) != "YZ":
        raise ThreeStQpMeasurementBendingBatchError("bending_plane must stay YZ")
    if str(config.get("observable", {}).get("name")) != "bending_raw":
        raise ThreeStQpMeasurementBendingBatchError("primary observable must stay bending_raw")
    if bool(config.get("official_qp_like_jacobian_authorized", True)):
        raise ThreeStQpMeasurementBendingBatchError("q/p-like Jacobian must stay unauthorized")
    if str(config.get("observable", {}).get("formula")) != (
        "atan((y2-y1)/(z2-z1)) - atan((y3-y2)/(z3-z2))"
    ):
        refuse_change_bending()
    return dict(config)


def _expect_sha(path: Path, expected: str, label: str) -> None:
    digest = sha256_file(path)
    if digest != expected:
        raise ThreeStQpMeasurementBendingBatchError(f"{label} hash mismatch: {digest}")


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
            raise ThreeStQpMeasurementBendingBatchError(f"{workbook} decision must stay frozen")
        inherited[workbook] = {
            "decision": decision["decision"],
            "config_sha256": spec["config_sha256"],
            "decision_sha256": spec["decision_sha256"],
        }
    if inherited["workbook_119"]["decision"] != "three_st_qp_calibration_not_established":
        raise ThreeStQpMeasurementBendingBatchError("WB119 must remain not_established")
    if inherited["workbook_125"]["decision"] != "three_st_qp_measurement_bending_contract_established":
        raise ThreeStQpMeasurementBendingBatchError("WB125 bending contract must remain established")
    pins = verify_pinned_calypso_sources(config)
    if not pins.get("all_match"):
        raise ThreeStQpMeasurementBendingBatchError("pinned Calypso sources must match config hashes")
    inherited["pinned_calypso_sources"] = pins
    return inherited


def _finite(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if np.isfinite(number) else None


def _sign(value: float | None) -> int:
    if value is None:
        return 0
    if value > 0.0:
        return 1
    if value < 0.0:
        return -1
    return 0


def _spearman(x: Sequence[float], y: Sequence[float]) -> float | None:
    if len(x) < 3:
        return None
    left = np.asarray(x, dtype=np.float64)
    right = np.asarray(y, dtype=np.float64)
    if left.size != right.size or left.size == 0:
        return None
    if float(np.std(left)) == 0.0 or float(np.std(right)) == 0.0:
        return None
    rx = np.argsort(np.argsort(left)).astype(np.float64)
    ry = np.argsort(np.argsort(right)).astype(np.float64)
    if float(np.std(rx)) == 0.0 or float(np.std(ry)) == 0.0:
        return None
    value = float(np.corrcoef(rx, ry)[0, 1])
    return value if np.isfinite(value) else None


def assign_bin(value: float | None, edges: Sequence[float], labels: Sequence[str]) -> str:
    if value is None:
        return "unknown"
    numeric = [float(edge) for edge in edges]
    for idx, label in enumerate(labels):
        if numeric[idx] <= value < numeric[idx + 1]:
            return label
    if value == numeric[-1]:
        return labels[-1]
    return "out_of_range"


def _truth_charge(row: Mapping[str, Any]) -> float | None:
    charge = _finite(row.get("truth_charge"))
    if charge is not None:
        return charge
    qp = _finite(row.get("q_over_p_truth_s1_per_mev"))
    return float(_sign(qp)) if qp is not None and qp != 0.0 else None


def _truth_tx_ty(row: Mapping[str, Any]) -> tuple[float | None, float | None]:
    stations = row.get("truth_station_momenta")
    if isinstance(stations, Sequence) and len(stations) > 1 and isinstance(stations[1], Mapping):
        return _finite(stations[1].get("tx")), _finite(stations[1].get("ty"))
    return _finite(row.get("tx_truth")), _finite(row.get("ty_truth"))


def _split_of(source_id: str) -> str:
    for run in CONSTRUCTION_RUNS:
        if run in source_id:
            return "construction"
    for run in VALIDATION_RUNS:
        if run in source_id:
            return "validation"
    return "unknown"


def _fit_pull(row: Mapping[str, Any]) -> float | None:
    fit = _finite(row.get("q_over_p_fit_per_mev"))
    truth = _finite(row.get("q_over_p_truth_s1_per_mev"))
    sigma = _finite(row.get("sigma_q_over_p_per_mev"))
    if fit is None or truth is None or sigma is None or sigma <= 0.0:
        return None
    return (fit - truth) / sigma


def _has_complete_centroids(row: Mapping[str, Any]) -> bool:
    raw = row.get("station_centroids")
    if not isinstance(raw, Mapping):
        return False
    for station in THREE_STATION_IDS:
        item = raw.get(str(station), raw.get(station))
        if not isinstance(item, Mapping):
            return False
        x_mm = _finite(item.get("x_mm", item.get("x")))
        y_mm = _finite(item.get("y_mm", item.get("y")))
        z_mm = _finite(item.get("z_mm", item.get("z")))
        if x_mm is None or y_mm is None or z_mm is None:
            return False
    return True


def slim_track_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Drop mot_hits when station centroids are already complete."""
    out = {key: row[key] for key in SLIM_TRACK_KEYS if key in row}
    if not _has_complete_centroids(row):
        hits = row.get("mot_hits")
        if hits is not None:
            out["mot_hits"] = hits
    return out


def decorate_track(row: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(row)
    topology = classify_topology(row)
    out.update(topology)
    centroids = centroids_from_row(row)
    out["has_three_station_centroids"] = centroids is not None
    if centroids is None:
        out["bending_raw"] = None
        out["bending_x"] = None
        out["constructable"] = False
        return out
    bending = construct_bending(centroids, config)
    out.update(bending)
    dz13 = float(centroids[3]["z"]) - float(centroids[1]["z"])
    if abs(dz13) >= float(config["gates"]["min_station_dz_mm"]):
        out["t_x_chord"] = (float(centroids[3]["x"]) - float(centroids[1]["x"])) / dz13
        out["t_y_chord"] = (float(centroids[3]["y"]) - float(centroids[1]["y"])) / dz13
    else:
        out["t_x_chord"] = None
        out["t_y_chord"] = None
    return out


def _empty_acc() -> dict[str, Any]:
    return {
        "n": 0,
        "n_constructable": 0,
        "n_with_truth": 0,
        "n_sign": 0,
        "n_sign_agree": 0,
        "n_bending_flip": 0,
        "n_fit": 0,
        "n_fit_flip": 0,
        "n_fit_all": 0,
        "n_fit_flip_all": 0,
        "n_same_sign_bending_truth": 0,
        "n_fit_flip_given_same_sign": 0,
        "n_fit_flip_raw_correct": 0,
        "n_fit_flip_raw_flip": 0,
        "n_fit_flip_unresolvable": 0,
        "n_fit_flip_other": 0,
        "class_counts": {CLASS_A: 0, CLASS_B: 0, CLASS_C: 0, CLASS_D: 0},
        "charge_odd_sum": 0.0,
        "charge_odd_n": 0,
        "charge_even_sum": 0.0,
        "charge_even_n": 0,
        "bending": [],
        "qp_truth": [],
        "bending_x": [],
        "qp_truth_x": [],
        "p_bins": defaultdict(lambda: {"n_sign": 0, "n_agree": 0, "n_fit": 0, "n_fit_flip": 0}),
        "tx_bins": defaultdict(lambda: {"n_sign": 0, "n_agree": 0}),
        "ty_bins": defaultdict(lambda: {"n_sign": 0, "n_agree": 0}),
        "abs_b_bins": defaultdict(lambda: {"n": 0, "n_fit_flip": 0, "n_sign": 0, "n_agree": 0}),
    }


def _observe(acc: dict[str, Any], row: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    acc["n"] += 1
    qp_truth = _finite(row.get("q_over_p_truth_s1_per_mev"))
    qp_fit = _finite(row.get("q_over_p_fit_per_mev"))
    if qp_fit is not None and qp_truth is not None and _sign(qp_fit) != 0 and _sign(qp_truth) != 0:
        acc["n_fit_all"] += 1
        if _sign(qp_fit) != _sign(qp_truth):
            acc["n_fit_flip_all"] += 1
    if not row.get("constructable") or _finite(row.get("bending_raw")) is None:
        return
    acc["n_constructable"] += 1
    bending = float(row["bending_raw"])
    charge = _truth_charge(row)
    p_truth = _finite(row.get("p_truth_s1_mev"))
    bending_x = _finite(row.get("bending_x"))
    t_x = _finite(row.get("t_x_chord"))
    t_y = _finite(row.get("t_y_chord"))
    if charge is None:
        return
    acc["n_with_truth"] += 1
    if qp_truth is not None:
        acc["bending"].append(bending)
        acc["qp_truth"].append(qp_truth)
        if bending_x is not None:
            acc["bending_x"].append(bending_x)
            acc["qp_truth_x"].append(qp_truth)
    acc["charge_odd_sum"] += bending * _sign(charge)
    acc["charge_odd_n"] += 1
    acc["charge_even_sum"] += bending
    acc["charge_even_n"] += 1
    bins = config["binning"]
    p_bin = assign_bin(p_truth, bins["p_truth_edges_mev"], bins["p_truth_labels"])
    tx_bin = assign_bin(t_x, bins["tx_edges"], bins["tx_labels"])
    ty_bin = assign_bin(t_y, bins["ty_edges"], bins["ty_labels"])
    abs_bin = assign_bin(abs(bending), bins["abs_bending_edges"], bins["abs_bending_labels"])
    unresolvable = abs(bending) < float(config["gates"]["unresolvable_abs_bending"])
    fit_is_flip = False
    if qp_fit is not None and qp_truth is not None and _sign(qp_fit) != 0 and _sign(qp_truth) != 0:
        acc["n_fit"] += 1
        fit_is_flip = _sign(qp_fit) != _sign(qp_truth)
        if fit_is_flip:
            acc["n_fit_flip"] += 1
            if unresolvable:
                acc["n_fit_flip_unresolvable"] += 1
            elif _sign(bending) != 0 and _sign(charge) != 0 and _sign(bending) == _sign(charge):
                acc["n_fit_flip_raw_correct"] += 1
            elif _sign(bending) != 0 and _sign(charge) != 0 and _sign(bending) != _sign(charge):
                acc["n_fit_flip_raw_flip"] += 1
            else:
                acc["n_fit_flip_other"] += 1
        acc["p_bins"][p_bin]["n_fit"] += 1
        acc["p_bins"][p_bin]["n_fit_flip"] += int(fit_is_flip)
        acc["abs_b_bins"][abs_bin]["n"] += 1
        acc["abs_b_bins"][abs_bin]["n_fit_flip"] += int(fit_is_flip)
    if _sign(bending) != 0 and _sign(charge) != 0:
        acc["n_sign"] += 1
        agree = _sign(bending) == _sign(charge)
        if agree:
            acc["n_sign_agree"] += 1
        else:
            acc["n_bending_flip"] += 1
        acc["p_bins"][p_bin]["n_sign"] += 1
        acc["p_bins"][p_bin]["n_agree"] += int(agree)
        acc["tx_bins"][tx_bin]["n_sign"] += 1
        acc["tx_bins"][tx_bin]["n_agree"] += int(agree)
        acc["ty_bins"][ty_bin]["n_sign"] += 1
        acc["ty_bins"][ty_bin]["n_agree"] += int(agree)
        acc["abs_b_bins"][abs_bin]["n_sign"] += 1
        acc["abs_b_bins"][abs_bin]["n_agree"] += int(agree)
        if agree and qp_fit is not None and qp_truth is not None and _sign(qp_fit) != 0:
            acc["n_same_sign_bending_truth"] += 1
            if fit_is_flip:
                acc["n_fit_flip_given_same_sign"] += 1
                acc["class_counts"][CLASS_A] += 1
    if unresolvable:
        acc["class_counts"][CLASS_B] += 1
    if (
        not unresolvable
        and _sign(bending) != 0
        and _sign(charge) != 0
        and _sign(bending) != _sign(charge)
    ):
        acc["class_counts"][CLASS_C] += 1
    pull = _fit_pull(row)
    if pull is not None and abs(pull) >= float(config["gates"]["extreme_pull_abs"]):
        acc["class_counts"][CLASS_D] += 1


def _finalize_acc(acc: dict[str, Any]) -> dict[str, Any]:
    def rate(num: int, den: int) -> float | None:
        return (num / den) if den else None

    p_bins = {}
    for label, item in acc["p_bins"].items():
        p_bins[label] = {
            "n_sign": item["n_sign"],
            "sign_agree": rate(item["n_agree"], item["n_sign"]),
            "n_fit": item["n_fit"],
            "n_fit_flip": item["n_fit_flip"],
            "fit_sign_agree": rate(item["n_fit"] - item["n_fit_flip"], item["n_fit"]),
        }
    tx_bins = {
        label: {"n_sign": item["n_sign"], "sign_agree": rate(item["n_agree"], item["n_sign"])}
        for label, item in acc["tx_bins"].items()
    }
    ty_bins = {
        label: {"n_sign": item["n_sign"], "sign_agree": rate(item["n_agree"], item["n_sign"])}
        for label, item in acc["ty_bins"].items()
    }
    abs_bins = {
        label: {
            "n": item["n"],
            "n_fit_flip": item["n_fit_flip"],
            "p_fit_flip": rate(item["n_fit_flip"], item["n"]),
            "n_sign": item["n_sign"],
            "sign_agree": rate(item["n_agree"], item["n_sign"]),
        }
        for label, item in acc["abs_b_bins"].items()
    }
    return {
        "n": acc["n"],
        "n_constructable": acc["n_constructable"],
        "n_with_truth": acc["n_with_truth"],
        "n_sign": acc["n_sign"],
        "n_sign_agree": acc["n_sign_agree"],
        "sign_agree": rate(acc["n_sign_agree"], acc["n_sign"]),
        "n_bending_flip": acc["n_bending_flip"],
        "p_raw_bending_flip": rate(acc["n_bending_flip"], acc["n_sign"]),
        "n_fit": acc["n_fit"],
        "n_fit_flip": acc["n_fit_flip"],
        "n_fit_all": acc["n_fit_all"],
        "n_fit_flip_all": acc["n_fit_flip_all"],
        "fit_sign_agree": rate(acc["n_fit"] - acc["n_fit_flip"], acc["n_fit"]),
        "fit_sign_agree_all": rate(acc["n_fit_all"] - acc["n_fit_flip_all"], acc["n_fit_all"]),
        "n_same_sign_bending_truth": acc["n_same_sign_bending_truth"],
        "n_fit_flip_given_same_sign": acc["n_fit_flip_given_same_sign"],
        "p_fit_flip_given_bending_correct": rate(
            acc["n_fit_flip_given_same_sign"], acc["n_same_sign_bending_truth"]
        ),
        "n_fit_flip_raw_correct": acc["n_fit_flip_raw_correct"],
        "n_fit_flip_raw_flip": acc["n_fit_flip_raw_flip"],
        "n_fit_flip_unresolvable": acc["n_fit_flip_unresolvable"],
        "n_fit_flip_other": acc["n_fit_flip_other"],
        "frac_fit_flip_raw_correct": rate(acc["n_fit_flip_raw_correct"], acc["n_fit_flip"]),
        "frac_fit_flip_raw_flip": rate(acc["n_fit_flip_raw_flip"], acc["n_fit_flip"]),
        "frac_fit_flip_unresolvable": rate(acc["n_fit_flip_unresolvable"], acc["n_fit_flip"]),
        "spearman_bending_vs_qp_truth": _spearman(acc["bending"], acc["qp_truth"])
        if len(acc["bending"]) == len(acc["qp_truth"])
        else None,
        "spearman_bending_x_vs_qp_truth": _spearman(acc["bending_x"], acc["qp_truth_x"]),
        "charge_odd_mean": (acc["charge_odd_sum"] / acc["charge_odd_n"]) if acc["charge_odd_n"] else None,
        "charge_even_mean": (acc["charge_even_sum"] / acc["charge_even_n"]) if acc["charge_even_n"] else None,
        "class_counts": dict(acc["class_counts"]),
        "p_bins": p_bins,
        "tx_bins": tx_bins,
        "ty_bins": ty_bins,
        "abs_bending_bins": abs_bins,
        "sigma_b_used": False,
        "point_estimator_only": True,
    }


def match_source_stability(
    cells: Mapping[tuple[str, ...], Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    gates = config["gates"]
    min_cell = int(gates["min_matched_cell"])
    floor = float(gates["abs_mean_floor"])
    matched_c = 0.0
    matched_v = 0.0
    weight = 0.0
    weighted_c = 0.0
    weighted_v = 0.0
    n_cells = 0
    n_tracks_c = 0
    n_tracks_v = 0
    for key, item in cells.items():
        n_c = int(item.get("n_construction") or 0)
        n_v = int(item.get("n_validation") or 0)
        n_tracks_c += n_c
        n_tracks_v += n_v
        if n_c < min_cell or n_v < min_cell:
            continue
        mean_c = item["sum_construction"] / n_c
        mean_v = item["sum_validation"] / n_v
        w = float(min(n_c, n_v))
        weighted_c += mean_c * w
        weighted_v += mean_v * w
        weight += w
        matched_c += n_c
        matched_v += n_v
        n_cells += 1
    total = n_tracks_c + n_tracks_v
    matched_frac = ((matched_c + matched_v) / total) if total else None
    if weight <= 0.0:
        return {
            "n_matched_cells": 0,
            "matched_fraction": matched_frac,
            "construction_charge_odd_mean": None,
            "validation_charge_odd_mean": None,
            "rel_range": None,
            "sign_disagree": None,
            "enough": False,
        }
    mean_c = weighted_c / weight
    mean_v = weighted_v / weight
    rel = abs(mean_c - mean_v) / max(abs(mean_c), abs(mean_v), floor)
    return {
        "n_matched_cells": n_cells,
        "matched_fraction": matched_frac,
        "n_matched_construction": int(matched_c),
        "n_matched_validation": int(matched_v),
        "construction_charge_odd_mean": mean_c,
        "validation_charge_odd_mean": mean_v,
        "rel_range": rel,
        "sign_disagree": _sign(mean_c) != 0 and _sign(mean_v) != 0 and _sign(mean_c) != _sign(mean_v),
        "enough": (matched_frac or 0.0) >= float(gates["min_matched_fraction"]),
    }


def _empty_source() -> dict[str, Any]:
    return {
        "n": 0,
        "n_sign": 0,
        "n_sign_agree": 0,
        "n_bending_flip": 0,
        "n_fit": 0,
        "n_fit_flip": 0,
        "n_fit_flip_raw_correct": 0,
        "n_fit_flip_raw_flip": 0,
        "charge_odd_sum": 0.0,
        "charge_odd_n": 0,
    }


def evaluate_rows(
    rows: Iterable[Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    keep_identities: bool = False,
) -> dict[str, Any]:
    groups = {
        "all": _empty_acc(),
        "clean_18hit": _empty_acc(),
        "complete_three_st": _empty_acc(),
        "dirty": _empty_acc(),
    }
    match_cells: dict[tuple[str, ...], dict[str, float]] = defaultdict(
        lambda: {
            "n_construction": 0.0,
            "n_validation": 0.0,
            "sum_construction": 0.0,
            "sum_validation": 0.0,
        }
    )
    unmatched = {
        "construction": {"sum": 0.0, "n": 0},
        "validation": {"sum": 0.0, "n": 0},
    }
    source_stats: dict[str, dict[str, Any]] = defaultdict(_empty_source)
    identities: list[dict[str, Any]] = []
    dump_diag = {
        "n_complete_centroids": 0,
        "n_mot_hits_fallback": 0,
        "n_truth_matched": 0,
        "n_truth_reference": 0,
        "centroid_sources": defaultdict(int),
    }
    n_tracks = 0
    for raw in rows:
        slim = slim_track_row(raw) if "station_centroids" in raw or "mot_hits" in raw else dict(raw)
        row = decorate_track(slim, config)
        n_tracks += 1
        if _has_complete_centroids(slim):
            dump_diag["n_complete_centroids"] += 1
        elif slim.get("mot_hits"):
            dump_diag["n_mot_hits_fallback"] += 1
        if row.get("truth_matched"):
            dump_diag["n_truth_matched"] += 1
        if row.get("truth_reference_available"):
            dump_diag["n_truth_reference"] += 1
        dump_diag["centroid_sources"][str(row.get("centroid_source") or "absent")] += 1
        _observe(groups["all"], row, config)
        if row.get("clean_18hit"):
            _observe(groups["clean_18hit"], row, config)
        if row.get("complete_three_st"):
            _observe(groups["complete_three_st"], row, config)
        if row.get("dirty"):
            _observe(groups["dirty"], row, config)
        source_id = str(row.get("source_id") or "unknown")
        src = source_stats[source_id]
        src["n"] += 1
        charge = _truth_charge(row)
        bending = _finite(row.get("bending_raw"))
        qp_fit = _finite(row.get("q_over_p_fit_per_mev"))
        qp_truth = _finite(row.get("q_over_p_truth_s1_per_mev"))
        if bending is not None and charge is not None and row.get("constructable"):
            src["charge_odd_sum"] += bending * _sign(charge)
            src["charge_odd_n"] += 1
            if _sign(bending) != 0 and _sign(charge) != 0:
                src["n_sign"] += 1
                if _sign(bending) == _sign(charge):
                    src["n_sign_agree"] += 1
                else:
                    src["n_bending_flip"] += 1
        if qp_fit is not None and qp_truth is not None and _sign(qp_fit) != 0 and _sign(qp_truth) != 0:
            src["n_fit"] += 1
            if _sign(qp_fit) != _sign(qp_truth):
                src["n_fit_flip"] += 1
                if bending is not None and charge is not None:
                    if abs(bending) < float(config["gates"]["unresolvable_abs_bending"]):
                        pass
                    elif _sign(bending) == _sign(charge):
                        src["n_fit_flip_raw_correct"] += 1
                    elif _sign(bending) != 0:
                        src["n_fit_flip_raw_flip"] += 1
        if keep_identities and row.get("constructable") and charge is not None and bending is not None:
            qp_fit_s = _sign(qp_fit)
            qp_truth_s = _sign(qp_truth)
            if qp_fit_s != 0 and qp_truth_s != 0 and qp_fit_s != qp_truth_s:
                label = None
                if abs(bending) < float(config["gates"]["unresolvable_abs_bending"]):
                    label = CLASS_B
                elif _sign(bending) == _sign(charge):
                    label = CLASS_A
                elif _sign(bending) != 0:
                    label = CLASS_C
                if label is not None:
                    identities.append(
                        {
                            "source_id": source_id,
                            "skip_index": row.get("skip_index"),
                            "track_index": row.get("track_index"),
                            "class": label,
                            "bending_raw": bending,
                            "q_over_p_fit_per_mev": qp_fit,
                            "q_over_p_truth_s1_per_mev": qp_truth,
                        }
                    )
        if not (row.get("clean_18hit") and row.get("constructable")):
            continue
        if charge is None or bending is None:
            continue
        bins = config["binning"]
        p_bin = assign_bin(
            _finite(row.get("p_truth_s1_mev")),
            bins["p_truth_edges_mev"],
            bins["p_truth_labels"],
        )
        tx_bin = assign_bin(_finite(row.get("t_x_chord")), bins["tx_edges"], bins["tx_labels"])
        ty_bin = assign_bin(_finite(row.get("t_y_chord")), bins["ty_edges"], bins["ty_labels"])
        charge_label = "plus" if _sign(charge) > 0 else "minus"
        split = _split_of(source_id)
        key = (charge_label, p_bin, tx_bin, ty_bin, "clean_18hit")
        odd = bending * _sign(charge)
        if split == "construction":
            match_cells[key]["n_construction"] += 1
            match_cells[key]["sum_construction"] += odd
            unmatched["construction"]["sum"] += odd
            unmatched["construction"]["n"] += 1
        elif split == "validation":
            match_cells[key]["n_validation"] += 1
            match_cells[key]["sum_validation"] += odd
            unmatched["validation"]["sum"] += odd
            unmatched["validation"]["n"] += 1
    clean = _finalize_acc(groups["clean_18hit"])
    all_rows = _finalize_acc(groups["all"])
    matched = match_source_stability(match_cells, config)
    floor = float(config["gates"]["abs_mean_floor"])
    unmatched_c = (
        unmatched["construction"]["sum"] / unmatched["construction"]["n"]
        if unmatched["construction"]["n"]
        else None
    )
    unmatched_v = (
        unmatched["validation"]["sum"] / unmatched["validation"]["n"]
        if unmatched["validation"]["n"]
        else None
    )
    unmatched_rel = None
    if unmatched_c is not None and unmatched_v is not None:
        unmatched_rel = abs(unmatched_c - unmatched_v) / max(
            abs(unmatched_c), abs(unmatched_v), floor
        )
    sources_out = {}
    for source_id, item in source_stats.items():
        n_sign = int(item["n_sign"])
        sources_out[source_id] = {
            "n": int(item["n"]),
            "n_sign": n_sign,
            "sign_agree": (item["n_sign_agree"] / n_sign) if n_sign else None,
            "n_bending_flip": int(item["n_bending_flip"]),
            "n_fit": int(item["n_fit"]),
            "n_fit_flip": int(item["n_fit_flip"]),
            "n_fit_flip_raw_correct": int(item["n_fit_flip_raw_correct"]),
            "n_fit_flip_raw_flip": int(item["n_fit_flip_raw_flip"]),
            "charge_odd_mean": (item["charge_odd_sum"] / item["charge_odd_n"])
            if item["charge_odd_n"]
            else None,
        }
    fit_flips = int(all_rows["n_fit_flip"])
    fit_flips_all = int(all_rows["n_fit_flip_all"])
    class_counts = all_rows["class_counts"]
    expected = float(config["gates"]["wb119_n_fit_flip_expected"])
    rel_tol = float(config["gates"]["wb119_n_fit_flip_rel_tol"])

    def _share(num: int, den: float | int | None) -> float | None:
        if not den:
            return None
        return num / float(den)

    wb119 = {
        "n_expected": int(config["gates"]["wb119_n_fit_flip_expected"]),
        "denominator": "all_truth_matched_primary",
        "n_fit_all": int(all_rows["n_fit_all"]),
        "n_fit_flip_all": fit_flips_all,
        "n_fit_flip_constructable": fit_flips,
        "n_fit_flip_no_bending": fit_flips_all - fit_flips,
        "n_fit_flip_observed": fit_flips,
        "n_class_a": class_counts[CLASS_A],
        "n_class_b": class_counts[CLASS_B],
        "n_class_c": class_counts[CLASS_C],
        "n_fit_flip_raw_correct": all_rows["n_fit_flip_raw_correct"],
        "n_fit_flip_raw_flip": all_rows["n_fit_flip_raw_flip"],
        "n_fit_flip_unresolvable": all_rows["n_fit_flip_unresolvable"],
        "frac_class_a_of_fit_flip": all_rows["frac_fit_flip_raw_correct"],
        "frac_class_c_of_fit_flip": all_rows["frac_fit_flip_raw_flip"],
        "frac_class_a_of_wb119": _share(int(all_rows["n_fit_flip_raw_correct"]), expected),
        "frac_class_c_of_wb119": _share(int(all_rows["n_fit_flip_raw_flip"]), expected),
        "frac_unresolvable_of_wb119": _share(int(all_rows["n_fit_flip_unresolvable"]), expected),
        "frac_no_bending_of_wb119": _share(fit_flips_all - fit_flips, expected),
        "clean_18hit": {
            "n_fit_flip": clean["n_fit_flip"],
            "n_fit_flip_raw_correct": clean["n_fit_flip_raw_correct"],
            "n_fit_flip_raw_flip": clean["n_fit_flip_raw_flip"],
            "frac_class_a_of_fit_flip": clean["frac_fit_flip_raw_correct"],
            "frac_class_c_of_fit_flip": clean["frac_fit_flip_raw_flip"],
        },
        "quantified": False,
        "reason": "batch_dumps_required_for_243758",
    }
    count_for_match = fit_flips_all if fit_flips_all else fit_flips
    if count_for_match >= int(config["gates"]["min_fit_flip_recorded"]) and abs(count_for_match - expected) <= rel_tol * expected:
        wb119["quantified"] = True
        wb119["reason"] = "observed_fit_flips_within_tolerance_of_WB119"
    elif count_for_match >= int(config["gates"]["min_fit_flip_recorded"]):
        wb119["quantified"] = True
        wb119["reason"] = "observed_fit_flips_on_same_nine_sources_not_count_matched"
    return {
        "n_tracks": n_tracks,
        "n_clean_18hit": clean["n"],
        "n_complete_three_st": groups["complete_three_st"]["n"],
        "n_dirty": groups["dirty"]["n"],
        "n_constructable": all_rows["n_constructable"],
        "clean_18hit": clean,
        "complete_three_st": _finalize_acc(groups["complete_three_st"]),
        "dirty": _finalize_acc(groups["dirty"]),
        "all": all_rows,
        "source_stability_matched": matched,
        "source_stability_unmatched": {
            "construction_charge_odd_mean": unmatched_c,
            "validation_charge_odd_mean": unmatched_v,
            "rel_range": unmatched_rel,
            "n_construction": unmatched["construction"]["n"],
            "n_validation": unmatched["validation"]["n"],
            "note": "unmatched is a spectrum diagnostic; official gate uses matched cells",
        },
        "by_source": sources_out,
        "dump_diagnostics": {
            "n_complete_centroids": dump_diag["n_complete_centroids"],
            "n_mot_hits_fallback": dump_diag["n_mot_hits_fallback"],
            "n_truth_matched": dump_diag["n_truth_matched"],
            "n_truth_reference": dump_diag["n_truth_reference"],
            "centroid_sources": dict(dump_diag["centroid_sources"]),
        },
        "identities": identities if keep_identities else [],
        "wb119_sign_flip": wb119,
    }


def evaluate_mechanisms(acc: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    gates = config["gates"]
    clean = acc["clean_18hit"]
    all_rows = acc["all"]
    n_clean = int(acc["n_clean_18hit"])
    enough = n_clean >= int(gates["min_clean_contract"])
    present = STATUS_UNRESOLVED
    sign_agree = clean.get("sign_agree")
    spearman = clean.get("spearman_bending_vs_qp_truth")
    spearman_x = clean.get("spearman_bending_x_vs_qp_truth")
    if enough and sign_agree is not None and spearman is not None:
        ortho_ok = spearman_x is None or abs(float(spearman_x)) <= float(gates["orthogonal_spearman_abs_max"])
        if (
            float(sign_agree) >= float(gates["sign_agree_present"])
            and abs(float(spearman)) >= float(gates["spearman_abs_present"])
            and ortho_ok
        ):
            present = STATUS_SUPPORTED
        else:
            present = STATUS_REJECTED
    p_bins = clean.get("p_bins") or {}
    high = p_bins.get(str(config["binning"]["high_p_limit_label"])) or {}
    low = p_bins.get(str(config["binning"]["low_p_label"])) or {}
    limited = STATUS_UNRESOLVED
    high_n = int(high.get("n_sign") or 0)
    low_n = int(low.get("n_sign") or 0)
    if (
        n_clean >= int(gates["min_clean_recorded"])
        and high_n >= int(gates["min_high_p_2tev"])
        and low_n >= int(gates["min_low_p"])
        and high.get("sign_agree") is not None
        and low.get("sign_agree") is not None
    ):
        if float(low["sign_agree"]) - float(high["sign_agree"]) >= float(gates["high_p_sign_drop"]):
            limited = STATUS_SUPPORTED
        else:
            limited = STATUS_REJECTED
    elif enough and high_n < int(gates["min_high_p_2tev"]):
        limited = STATUS_UNRESOLVED
    fit_extra = STATUS_UNRESOLVED
    frac = clean.get("p_fit_flip_given_bending_correct")
    high_raw = high.get("sign_agree")
    high_fit = high.get("fit_sign_agree")
    if enough and clean.get("n_same_sign_bending_truth"):
        extra_rate = frac is not None and float(frac) >= float(gates["fit_extra_flip_frac"])
        extra_high = (
            high_n >= int(gates["min_high_p_2tev"])
            and high_raw is not None
            and high_fit is not None
            and float(high_raw) >= float(gates["high_p_raw_fidelity_min"])
            and float(high_raw) - float(high_fit) >= float(gates["high_p_sign_drop"])
        )
        if extra_rate or extra_high:
            fit_extra = STATUS_SUPPORTED
        elif frac is not None:
            fit_extra = STATUS_REJECTED
    source = STATUS_UNRESOLVED
    matched = acc.get("source_stability_matched") or {}
    if matched.get("enough") and matched.get("rel_range") is not None:
        if bool(matched.get("sign_disagree")) or float(matched["rel_range"]) >= float(
            gates["source_rel_range_matched"]
        ):
            source = STATUS_SUPPORTED
        else:
            source = STATUS_REJECTED
    mapping = STATUS_UNRESOLVED
    wb119 = acc.get("wb119_sign_flip") or {}
    n_flip = int(wb119.get("n_fit_flip_observed") or 0)
    frac_a = wb119.get("frac_class_a_of_fit_flip")
    frac_c = wb119.get("frac_class_c_of_fit_flip")
    if n_flip >= int(gates["min_fit_flip_recorded"]) and frac_a is not None and frac_c is not None:
        if float(frac_a) >= float(gates["mapping_class_a_frac"]):
            mapping = STATUS_SUPPORTED
        elif float(frac_c) >= float(gates["mapping_class_c_frac"]):
            mapping = STATUS_REJECTED
        else:
            mapping = STATUS_UNRESOLVED
    supported = [
        name
        for name, status in (
            (MECH_PRESENT, present),
            (MECH_LIMITED, limited),
            (MECH_FIT, fit_extra),
            (MECH_SOURCE, source),
            (MECH_MAP, mapping),
        )
        if status == STATUS_SUPPORTED
    ]
    official = supported[0] if len(supported) == 1 else MECH_MIXED
    return {
        MECH_PRESENT: present,
        MECH_LIMITED: limited,
        MECH_FIT: fit_extra,
        MECH_SOURCE: source,
        MECH_MAP: mapping,
        "official_mechanism": official,
        "supported_mechanisms": supported,
        "enough_clean_for_contract": enough,
    }


def dump_path_for_source(
    config: Mapping[str, Any],
    source_id: str,
    kind: str,
    campaign: str,
) -> Path:
    filename = (
        str(config["dump_filename"]) if kind == "tracks" else str(config["event_filename"])
    )
    return resolve_under_root(project_root(), str(config["dump_root"])) / campaign / source_id / filename


def _campaign_source_specs(config: Mapping[str, Any], campaign: str) -> list[tuple[str, dict[str, Any], Path]]:
    specs: list[tuple[str, dict[str, Any], Path]] = []
    if campaign == "smoke" and bool(config.get("smoke", {}).get("reuse_wb125_contract")):
        root = resolve_under_root(project_root(), str(config["smoke"]["wb125_dump_root"]))
        keys = (("construction", "construction_sources"), ("validation", "validation_sources"))
        for split, key in keys:
            for spec in config["mc_data"][key]:
                path = root / spec["source_id"] / str(config["dump_filename"])
                specs.append((split, spec, path))
        return specs
    source_key = (
        ("construction", "batch_construction_sources"),
        ("validation", "batch_validation_sources"),
    ) if campaign == "batch" else (
        ("construction", "construction_sources"),
        ("validation", "validation_sources"),
    )
    for split, key in source_key:
        for spec in config["mc_data"][key]:
            path = dump_path_for_source(config, spec["source_id"], "tracks", campaign)
            specs.append((split, spec, path))
    return specs


def iter_campaign_tracks(config: Mapping[str, Any], campaign: str):
    for split, spec, path in _campaign_source_specs(config, campaign):
        for row in iter_jsonl(path, split=split):
            if row.get("kind") != "track":
                continue
            if campaign != "smoke" and row.get("collection") not in (None, SOURCE_COLLECTION_NAME):
                continue
            yield slim_track_row(row)


def campaign_source_presence(config: Mapping[str, Any], campaign: str) -> dict[str, bool]:
    return {
        spec["source_id"]: path.is_file()
        for _split, spec, path in _campaign_source_specs(config, campaign)
    }


def inventory_campaign(config: Mapping[str, Any], campaign: str) -> dict[str, Any]:
    sources_present = campaign_source_presence(config, campaign)
    acc = evaluate_rows(
        iter_campaign_tracks(config, campaign),
        config,
        keep_identities=campaign == "smoke",
    )
    acc["sources_present"] = sources_present
    acc["dumps_present"] = bool(sources_present) and all(sources_present.values())
    acc["campaign"] = campaign
    return acc


def decide(
    report: Mapping[str, Any],
    inherited: Mapping[str, Any],
    *,
    dumps_materialized: bool,
    campaign: str,
    config: Mapping[str, Any],
    htcondor_submitted: bool = False,
) -> dict[str, Any]:
    if bool(config.get("three_st_qp_trusted_observable")):
        refuse_flip_s2()
    if bool(config.get("residual_conditional_authorized")):
        refuse_residual_conditional()
    mechanisms = evaluate_mechanisms(report, config)
    n_clean = int(report.get("n_clean_18hit") or 0)
    min_recorded = int(config["gates"]["min_clean_recorded"])
    dumps_ok = bool(dumps_materialized)
    pins_ok = bool((inherited.get("pinned_calypso_sources") or {}).get("all_match", True))
    inherited_ok = inherited.get("workbook_119", {}).get("decision") == "three_st_qp_calibration_not_established"
    if not dumps_ok or not pins_ok or not inherited_ok:
        decision = DECISION_NOT
        verdict = "FAIL"
        diagnosis = "FAIL"
    elif campaign == "batch" and n_clean >= min_recorded:
        decision = DECISION_RECORDED
        verdict = "PASS"
        diagnosis = "RECORDED"
    else:
        decision = DECISION_CONTRACT
        verdict = "PASS"
        diagnosis = "INCONCLUSIVE" if not mechanisms["enough_clean_for_contract"] else "RECORDED"
    return {
        "kind": "three_st_qp_measurement_bending_batch_contract",
        "task": TASK,
        "workbook": WORKBOOK,
        "campaign": campaign,
        "verdict": verdict,
        "diagnosis_verdict": diagnosis,
        "decision": decision,
        "official_mechanism": mechanisms["official_mechanism"],
        "mechanisms": mechanisms,
        "n_tracks": report.get("n_tracks"),
        "n_clean_18hit": n_clean,
        "n_complete_three_st": report.get("n_complete_three_st"),
        "n_dirty": report.get("n_dirty"),
        "n_constructable": report.get("n_constructable"),
        "clean_18hit": report.get("clean_18hit"),
        "complete_three_st": report.get("complete_three_st"),
        "dirty": report.get("dirty"),
        "all": report.get("all"),
        "source_stability_matched": report.get("source_stability_matched"),
        "source_stability_unmatched": report.get("source_stability_unmatched"),
        "by_source": report.get("by_source"),
        "dump_diagnostics": report.get("dump_diagnostics"),
        "identities": report.get("identities") if campaign == "smoke" else [],
        "wb119_sign_flip": report.get("wb119_sign_flip"),
        "three_st_qp_trusted_observable": False,
        "residual_conditional_authorized": False,
        "s2_flipped_to_pass": False,
        "official_qp_like_jacobian_authorized": False,
        "measurement_uncertainty_propagated": False,
        "sigma_b_invented": False,
        "e_r_ift_given_qp_computed": False,
        "e_r_ift_given_bending_computed": False,
        "bending_definition_changed": False,
        "dumps_materialized": dumps_ok,
        "htcondor_submitted": bool(htcondor_submitted),
        "inherited_workbook_119": inherited.get("workbook_119", {}).get("decision"),
        "inherited_workbook_125": inherited.get("workbook_125", {}).get("decision"),
        "conversion_chain": conversion_chain(),
        "official_quantities": official_quantities(),
    }
