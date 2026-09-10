"""Yasu-S2I: measurement-level 3ST bending / fit-independent curvature proxy.

Constructs signed bending_raw from S1/S2/S3 MOT cluster centroids only.
Fitted q/p and its 5x5 never enter construction, selection, or prior.
Truth q/p is a source-disjoint MC calibration reference only.
Does not flip S2 or open Stage 3.
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
    iter_jsonl,
    verify_pinned_calypso_sources,
)

SCHEMA_VERSION = "three-st-qp-measurement-bending-v1"
DEFAULT_CONFIG = "configs/three_st_qp_measurement_bending_v1.yaml"
TASK = "YASU-S2I"
WORKBOOK = 125

DECISION_CONTRACT = "three_st_qp_measurement_bending_contract_established"
DECISION_RECORDED = "three_st_qp_measurement_bending_recorded"
DECISION_NOT = "three_st_qp_measurement_bending_not_established"

STATUS_SUPPORTED = "supported"
STATUS_REJECTED = "rejected"
STATUS_UNRESOLVED = "unresolved"

MECH_PRESENT = "raw_bending_information_present"
MECH_LIMITED = "raw_bending_information_limited_at_high_p"
MECH_FIT = "fit_additional_sign_failure"
MECH_SOURCE = "raw_measurement_bias/source_dependence"
MECH_MIXED = "mixed/inconclusive"

CLASS_A = "truth_bending_same_fitted_flip"
CLASS_B = "truth_and_bending_unresolvable"
CLASS_C = "bending_sign_flip"
CLASS_D = "fitted_qp_extreme_tail"

REQUIRED_INHERITANCE = (
    "workbook_117",
    "workbook_118",
    "workbook_119",
    "workbook_120",
    "workbook_121",
    "workbook_122",
    "workbook_123",
    "workbook_124",
)

THREE_STATION_IDS = (1, 2, 3)


class ThreeStQpMeasurementBendingError(ValueError):
    """Raised when the S2I contract is illegal."""


def refuse_flip_s2() -> None:
    raise ThreeStQpMeasurementBendingError(
        "S2I cannot set three_st_qp_trusted_observable true"
    )


def refuse_residual_conditional() -> None:
    raise ThreeStQpMeasurementBendingError(
        "E[r_IFT|q/p] and E[r_IFT|bending] are Stage 3; S2I does not open them"
    )


def refuse_change_fitter() -> None:
    raise ThreeStQpMeasurementBendingError(
        "S2I must not change fitter, seed, hits, geometry, or covariance scale"
    )


def refuse_fitted_qp_in_construction() -> None:
    raise ThreeStQpMeasurementBendingError(
        "fitted q/p must not enter bending_raw construction"
    )


def refuse_invent_sigma_b() -> None:
    raise ThreeStQpMeasurementBendingError(
        "S2I must not invent sigma_b; point-estimator calibration only"
    )


def refuse_seed_jacobian() -> None:
    raise ThreeStQpMeasurementBendingError(
        "CircleFit 0.55 T is seed-only; not an official q/p Jacobian"
    )


def refuse_htcondor() -> None:
    raise ThreeStQpMeasurementBendingError(
        "S2I must not submit a large HTCondor reconstruction dump"
    )


def conversion_chain() -> dict[str, Any]:
    return {
        "calypso_git_sha": "40892527e9c65409afd2378a2abfc25ddbddac03",
        "bending_plane": "YZ",
        "orthogonal_plane": "XZ",
        "circle_fit_plane": "CircleFit.cxx:6-8 space-point (z,y)",
        "circle_fit_charge": "CircleFitTrackSeedTool.cxx:335 cy<0 => charge +1",
        "non_bending_plane": "CircleFitTrackSeedTool.cxx:310-314 linear (z,x)",
        "seed_B_tesla_unofficial": 0.55,
        "seed_momentum": "CircleFitTrackSeedTool.cxx:334 r*0.001*0.3*0.55",
        "lorentz": "v~z-hat, B~x-hat => v x B in y",
        "centroid": "unique MOT-matched SCT space-point mean per station; else det-element localToGlobal of MOT cluster",
        "not_centroid": "persisted front() xyz, fitted q/p, 5x5, truth q/p, seed circle",
        "primary_observable": "bending_raw = atan(ty12)-atan(ty23) [radian]",
        "companion": "sagitta_y = y2 - chord(S1,S3) at z2 [mm]",
        "sigma_b": "not invented; cluster local cov is 1D strip without stereo Jacobian",
        "qp_like": "unauthorized until machine-confirmed int B_perp dl",
        "without_ift_output": "faser_reco.py:317-319 CKFTrackCollectionWithoutIFT",
    }


def official_quantities() -> dict[str, str]:
    return {
        "bending_raw": "signed YZ bending angle from MOT station centroids; primary",
        "sagitta_y_mm": "signed Y sagitta; same-sign companion",
        "bending_x": "XZ control; must not carry charge information",
        "q_over_p_fit_per_mev": "comparison only; never a construction input",
        "q_over_p_truth_s1_per_mev": "source-disjoint MC calibration reference only",
        "sigma_b": "absent; point estimator only",
        "truth_join": "dumped after construction fields; never a fit or bending input",
    }


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = resolve_under_root(project_root(), str(path or DEFAULT_CONFIG))
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ThreeStQpMeasurementBendingError(f"schema_version must be {SCHEMA_VERSION}")
    if config.get("task") != TASK:
        raise ThreeStQpMeasurementBendingError(f"task must be {TASK}")
    if int(config.get("workbook", -1)) != WORKBOOK:
        raise ThreeStQpMeasurementBendingError("workbook must be 125")
    if bool(config.get("three_st_qp_trusted_observable", True)):
        raise ThreeStQpMeasurementBendingError("three_st_qp_trusted_observable must stay false")
    if bool(config.get("residual_conditional_authorized", True)):
        raise ThreeStQpMeasurementBendingError("residual_conditional_authorized must stay false")
    if str(config.get("bending_plane")) != "YZ":
        raise ThreeStQpMeasurementBendingError("bending_plane must be YZ from CircleFit")
    if str(config.get("observable", {}).get("name")) != "bending_raw":
        raise ThreeStQpMeasurementBendingError("primary observable must be bending_raw")
    if bool(config.get("official_qp_like_jacobian_authorized", True)):
        raise ThreeStQpMeasurementBendingError("official q/p-like Jacobian must stay unauthorized")
    if bool(config.get("measurement_uncertainty_propagated", True)):
        raise ThreeStQpMeasurementBendingError("do not claim a propagated sigma_b")
    if bool(config.get("do_not_submit_htcondor")) is not True:
        raise ThreeStQpMeasurementBendingError("do_not_submit_htcondor must stay true")
    return dict(config)


def _expect_sha(path: Path, expected: str, label: str) -> None:
    digest = sha256_file(path)
    if digest != expected:
        raise ThreeStQpMeasurementBendingError(f"{label} hash mismatch: {digest}")


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
            raise ThreeStQpMeasurementBendingError(f"{workbook} decision must stay frozen")
        inherited[workbook] = {
            "decision": decision["decision"],
            "config_sha256": spec["config_sha256"],
            "decision_sha256": spec["decision_sha256"],
        }
    if inherited["workbook_119"]["decision"] != "three_st_qp_calibration_not_established":
        raise ThreeStQpMeasurementBendingError("WB119 must remain not_established")
    if inherited["workbook_124"]["decision"] != "three_st_qp_refit_provenance_recorded":
        raise ThreeStQpMeasurementBendingError("WB124 provenance must remain recorded")
    pins = verify_pinned_calypso_sources(config)
    if not pins.get("all_match"):
        raise ThreeStQpMeasurementBendingError("pinned Calypso sources must match config hashes")
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
    if left.size != right.size:
        return None
    rx = np.argsort(np.argsort(left)).astype(np.float64)
    ry = np.argsort(np.argsort(right)).astype(np.float64)
    if float(np.std(rx)) == 0.0 or float(np.std(ry)) == 0.0:
        return None
    value = float(np.corrcoef(rx, ry)[0, 1])
    return value if np.isfinite(value) else None


def _ols_slope(x: Sequence[float], y: Sequence[float]) -> float | None:
    if len(x) < 2:
        return None
    left = np.asarray(x, dtype=np.float64)
    right = np.asarray(y, dtype=np.float64)
    if float(np.std(left)) == 0.0:
        return None
    matrix = np.corrcoef(left, right)
    if not np.isfinite(matrix[0, 1]):
        return None
    return float(matrix[0, 1] * np.std(right) / np.std(left))


def _mean(values: Sequence[float]) -> float | None:
    array = np.asarray(list(values), dtype=np.float64)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return None
    return float(np.mean(finite))


def _station_key(raw: Any) -> int | None:
    try:
        station = int(raw)
    except (TypeError, ValueError):
        return None
    return station if station in THREE_STATION_IDS else None


def centroids_from_row(row: Mapping[str, Any]) -> dict[int, dict[str, float]] | None:
    raw = row.get("station_centroids")
    out: dict[int, dict[str, float]] = {}
    if isinstance(raw, Mapping):
        for key, item in raw.items():
            station = _station_key(key)
            if station is None or not isinstance(item, Mapping):
                continue
            x_mm = _finite(item.get("x_mm", item.get("x")))
            y_mm = _finite(item.get("y_mm", item.get("y")))
            z_mm = _finite(item.get("z_mm", item.get("z")))
            n_hits = item.get("n")
            if x_mm is None or y_mm is None or z_mm is None:
                continue
            out[station] = {
                "x": x_mm,
                "y": y_mm,
                "z": z_mm,
                "n": float(n_hits) if n_hits is not None else 1.0,
            }
    if all(station in out for station in THREE_STATION_IDS):
        return out
    hits = row.get("mot_hits")
    if not isinstance(hits, Sequence):
        return None
    sums: dict[int, list[float]] = {1: [0.0, 0.0, 0.0, 0.0], 2: [0.0, 0.0, 0.0, 0.0], 3: [0.0, 0.0, 0.0, 0.0]}
    for hit in hits:
        if not isinstance(hit, Mapping):
            continue
        station = _station_key(hit.get("station"))
        x_mm = _finite(hit.get("x_mm", hit.get("x")))
        y_mm = _finite(hit.get("y_mm", hit.get("y")))
        z_mm = _finite(hit.get("z_mm", hit.get("z")))
        if station is None or x_mm is None or y_mm is None or z_mm is None:
            continue
        sums[station][0] += x_mm
        sums[station][1] += y_mm
        sums[station][2] += z_mm
        sums[station][3] += 1.0
    built: dict[int, dict[str, float]] = {}
    for station, (sx, sy, sz, n_hits) in sums.items():
        if n_hits <= 0.0:
            continue
        built[station] = {"x": sx / n_hits, "y": sy / n_hits, "z": sz / n_hits, "n": n_hits}
    complete = built if all(station in built for station in THREE_STATION_IDS) else None
    return complete


def construct_bending(
    centroids: Mapping[int, Mapping[str, float]],
    config: Mapping[str, Any],
    *,
    q_over_p_fit: Any = None,
    q_over_p_truth: Any = None,
) -> dict[str, Any]:
    if q_over_p_fit is not None:
        refuse_fitted_qp_in_construction()
    if q_over_p_truth is not None:
        raise ThreeStQpMeasurementBendingError(
            "truth q/p must not enter bending_raw construction"
        )
    min_dz = float(config["gates"]["min_station_dz_mm"])
    try:
        c1 = centroids[1]
        c2 = centroids[2]
        c3 = centroids[3]
    except KeyError as exc:
        raise ThreeStQpMeasurementBendingError("S1/S2/S3 centroids are required") from exc
    dz12 = float(c2["z"]) - float(c1["z"])
    dz23 = float(c3["z"]) - float(c2["z"])
    dz13 = float(c3["z"]) - float(c1["z"])
    if abs(dz12) < min_dz or abs(dz23) < min_dz or abs(dz13) < min_dz:
        return {
            "bending_raw": None,
            "bending_x": None,
            "sagitta_y_mm": None,
            "sagitta_x_mm": None,
            "t_y12": None,
            "t_y23": None,
            "constructable": False,
            "reason": "station_dz_below_floor",
        }
    t_x12 = (float(c2["x"]) - float(c1["x"])) / dz12
    t_y12 = (float(c2["y"]) - float(c1["y"])) / dz12
    t_x23 = (float(c3["x"]) - float(c2["x"])) / dz23
    t_y23 = (float(c3["y"]) - float(c2["y"])) / dz23
    bending_raw = math.atan(t_y12) - math.atan(t_y23)
    bending_x = math.atan(t_x12) - math.atan(t_x23)
    frac = dz12 / dz13
    sagitta_y = float(c2["y"]) - (float(c1["y"]) + (float(c3["y"]) - float(c1["y"])) * frac)
    sagitta_x = float(c2["x"]) - (float(c1["x"]) + (float(c3["x"]) - float(c1["x"])) * frac)
    return {
        "bending_raw": float(bending_raw),
        "bending_x": float(bending_x),
        "sagitta_y_mm": float(sagitta_y),
        "sagitta_x_mm": float(sagitta_x),
        "t_y12": float(t_y12),
        "t_y23": float(t_y23),
        "t_x12": float(t_x12),
        "t_x23": float(t_x23),
        "constructable": True,
        "reason": None,
        "used_fitted_q_over_p": False,
        "used_truth_q_over_p": False,
        "used_front_xyz": False,
        "plane": "YZ",
    }


def classify_topology(row: Mapping[str, Any]) -> dict[str, Any]:
    stations = {
        int(station)
        for station in (row.get("measurements_on_track_stations") or [])
        if station is not None
    }
    n_mot = int(row.get("n_mot") or 0)
    n_ift = int(row.get("n_ift_mot") or 0)
    complete = THREE_STATION_IDS[0] in stations and THREE_STATION_IDS[1] in stations and THREE_STATION_IDS[2] in stations
    ift_leak = n_ift > 0 or bool(row.get("ift_leak"))
    clean_18hit = n_mot == 18 and complete and not ift_leak
    dirty = not clean_18hit
    return {
        "stations": sorted(stations),
        "n_mot": n_mot,
        "complete_three_st": complete and not ift_leak,
        "clean_18hit": clean_18hit,
        "dirty": dirty,
        "ift_leak": ift_leak,
    }


def assign_p_bin(p_mev: float | None, config: Mapping[str, Any]) -> str:
    if p_mev is None:
        return "p_unknown"
    edges = [float(v) for v in config["binning"]["p_truth_edges_mev"]]
    labels = list(config["binning"]["p_truth_labels"])
    for idx, label in enumerate(labels):
        if edges[idx] <= p_mev < edges[idx + 1]:
            return label
    if p_mev == edges[-1]:
        return labels[-1]
    return "out_of_range"


def _truth_charge(row: Mapping[str, Any]) -> float | None:
    charge = _finite(row.get("truth_charge"))
    if charge is not None:
        return charge
    qp = _finite(row.get("q_over_p_truth_s1_per_mev"))
    return float(_sign(qp)) if qp is not None and qp != 0.0 else None


def _fit_pull(row: Mapping[str, Any]) -> float | None:
    fit = _finite(row.get("q_over_p_fit_per_mev"))
    truth = _finite(row.get("q_over_p_truth_s1_per_mev"))
    sigma = _finite(row.get("sigma_q_over_p_per_mev"))
    if fit is None or truth is None or sigma is None or sigma <= 0.0:
        return None
    return (fit - truth) / sigma


def decorate_track(row: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(row)
    topology = classify_topology(row)
    out.update(topology)
    centroids = centroids_from_row(row)
    out["has_three_station_centroids"] = centroids is not None
    if centroids is None:
        out["bending_raw"] = None
        out["bending_x"] = None
        out["sagitta_y_mm"] = None
        out["constructable"] = False
        return out
    bending = construct_bending(centroids, config)
    out.update(bending)
    return out


def _subset_metrics(rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> dict[str, Any]:
    gates = config["gates"]
    floor = float(gates["unresolvable_abs_bending"])
    high_p_edge = float(gates["high_p_edge_mev"])
    constructable = [row for row in rows if row.get("constructable")]
    with_truth = [
        row
        for row in constructable
        if _finite(row.get("bending_raw")) is not None and _truth_charge(row) is not None
    ]
    sign_ok = 0
    sign_n = 0
    bending_flip = 0
    fit_flip = 0
    fit_n = 0
    same_sign_fit_flip = 0
    same_sign_n = 0
    class_counts = {CLASS_A: 0, CLASS_B: 0, CLASS_C: 0, CLASS_D: 0}
    bending_vals: list[float] = []
    qp_truth_vals: list[float] = []
    bending_x_vals: list[float] = []
    charge_odd: list[float] = []
    charge_even: list[float] = []
    high_sign_ok = 0
    high_sign_n = 0
    low_sign_ok = 0
    low_sign_n = 0
    quartile_flips = [0, 0, 0, 0]
    quartile_n = [0, 0, 0, 0]
    abs_bending: list[float] = []
    for row in with_truth:
        bending = float(row["bending_raw"])
        charge = float(_truth_charge(row))
        qp_truth = _finite(row.get("q_over_p_truth_s1_per_mev"))
        qp_fit = _finite(row.get("q_over_p_fit_per_mev"))
        p_truth = _finite(row.get("p_truth_s1_mev"))
        bending_x = _finite(row.get("bending_x"))
        bending_vals.append(bending)
        if qp_truth is not None:
            qp_truth_vals.append(qp_truth)
        if bending_x is not None:
            bending_x_vals.append(bending_x)
        charge_odd.append(bending * _sign(charge))
        charge_even.append(bending)
        if _sign(bending) != 0 and _sign(charge) != 0:
            sign_n += 1
            agree = _sign(bending) == _sign(charge)
            if agree:
                sign_ok += 1
            else:
                bending_flip += 1
            if p_truth is not None and p_truth >= high_p_edge:
                high_sign_n += 1
                high_sign_ok += int(agree)
            else:
                low_sign_n += 1
                low_sign_ok += int(agree)
        unresolvable = abs(bending) < floor
        if unresolvable:
            class_counts[CLASS_B] += 1
        if (
            not unresolvable
            and _sign(bending) != 0
            and _sign(charge) != 0
            and _sign(bending) != _sign(charge)
        ):
            class_counts[CLASS_C] += 1
        fit_is_flip = False
        if qp_fit is not None and qp_truth is not None and _sign(qp_fit) != 0 and _sign(qp_truth) != 0:
            fit_n += 1
            fit_is_flip = _sign(qp_fit) != _sign(qp_truth)
            if fit_is_flip:
                fit_flip += 1
            if _sign(bending) == _sign(charge) and _sign(bending) != 0:
                same_sign_n += 1
                if fit_is_flip:
                    same_sign_fit_flip += 1
                    class_counts[CLASS_A] += 1
        pull = _fit_pull(row)
        if pull is not None and abs(pull) >= float(gates["extreme_pull_abs"]):
            class_counts[CLASS_D] += 1
        abs_bending.append(abs(bending))
    if abs_bending:
        edges = np.quantile(np.asarray(abs_bending, dtype=np.float64), [0.25, 0.50, 0.75])
        for row in with_truth:
            bending = abs(float(row["bending_raw"]))
            qp_fit = _finite(row.get("q_over_p_fit_per_mev"))
            qp_truth = _finite(row.get("q_over_p_truth_s1_per_mev"))
            if qp_fit is None or qp_truth is None or _sign(qp_fit) == 0 or _sign(qp_truth) == 0:
                continue
            if bending <= edges[0]:
                idx = 0
            elif bending <= edges[1]:
                idx = 1
            elif bending <= edges[2]:
                idx = 2
            else:
                idx = 3
            quartile_n[idx] += 1
            if _sign(qp_fit) != _sign(qp_truth):
                quartile_flips[idx] += 1
    spearman = (
        _spearman(bending_vals, qp_truth_vals)
        if len(bending_vals) == len(qp_truth_vals) and qp_truth_vals
        else None
    )
    spearman_x = (
        _spearman(bending_x_vals, qp_truth_vals)
        if len(bending_x_vals) == len(qp_truth_vals) and qp_truth_vals
        else None
    )
    slope = (
        _ols_slope(qp_truth_vals, bending_vals)
        if len(bending_vals) == len(qp_truth_vals) and qp_truth_vals
        else None
    )
    return {
        "n": len(rows),
        "n_constructable": len(constructable),
        "n_with_truth": len(with_truth),
        "n_sign": sign_n,
        "n_sign_agree": sign_ok,
        "sign_agree": (sign_ok / sign_n) if sign_n else None,
        "n_bending_flip": bending_flip,
        "n_fit": fit_n,
        "n_fit_flip": fit_flip,
        "fit_sign_agree": ((fit_n - fit_flip) / fit_n) if fit_n else None,
        "n_same_sign_bending_truth": same_sign_n,
        "n_fit_flip_given_same_sign": same_sign_fit_flip,
        "fit_flip_given_same_sign": (same_sign_fit_flip / same_sign_n) if same_sign_n else None,
        "spearman_bending_vs_qp_truth": spearman,
        "spearman_bending_x_vs_qp_truth": spearman_x,
        "ols_slope_bending_vs_qp_truth": slope,
        "charge_odd_mean": _mean(charge_odd),
        "charge_even_mean": _mean(charge_even),
        "high_p_n_sign": high_sign_n,
        "high_p_sign_agree": (high_sign_ok / high_sign_n) if high_sign_n else None,
        "low_p_n_sign": low_sign_n,
        "low_p_sign_agree": (low_sign_ok / low_sign_n) if low_sign_n else None,
        "class_counts": class_counts,
        "fit_flip_by_abs_bending_quartile": [
            {
                "quartile": idx + 1,
                "n": quartile_n[idx],
                "n_fit_flip": quartile_flips[idx],
                "p_fit_flip": (quartile_flips[idx] / quartile_n[idx]) if quartile_n[idx] else None,
            }
            for idx in range(4)
        ],
        "sigma_b_used": False,
        "point_estimator_only": True,
    }


def evaluate_tracks(rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> dict[str, Any]:
    decorated = [decorate_track(row, config) for row in rows]
    clean = [row for row in decorated if row.get("clean_18hit")]
    complete = [row for row in decorated if row.get("complete_three_st")]
    dirty = [row for row in decorated if row.get("dirty")]
    construction = [row for row in clean if "100043" in str(row.get("source_id", ""))]
    validation = [row for row in clean if "100048" in str(row.get("source_id", ""))]
    clean_metrics = _subset_metrics(clean, config)
    source_means = {
        "construction_charge_odd_mean": _subset_metrics(construction, config)["charge_odd_mean"],
        "validation_charge_odd_mean": _subset_metrics(validation, config)["charge_odd_mean"],
        "n_construction_clean": len(construction),
        "n_validation_clean": len(validation),
    }
    left = source_means["construction_charge_odd_mean"]
    right = source_means["validation_charge_odd_mean"]
    floor = float(config["gates"]["abs_mean_floor"])
    if left is None or right is None:
        source_rel_range = None
        source_sign_disagree = None
    else:
        denom = max(abs(left), abs(right), floor)
        source_rel_range = abs(left - right) / denom
        source_sign_disagree = _sign(left) != 0 and _sign(right) != 0 and _sign(left) != _sign(right)
    return {
        "n_tracks": len(decorated),
        "n_clean_18hit": len(clean),
        "n_complete_three_st": len(complete),
        "n_dirty": len(dirty),
        "n_constructable": sum(1 for row in decorated if row.get("constructable")),
        "clean_18hit": clean_metrics,
        "complete_three_st": _subset_metrics(complete, config),
        "dirty": _subset_metrics(dirty, config),
        "all": _subset_metrics(decorated, config),
        "source_stability": {**source_means, "rel_range": source_rel_range, "sign_disagree": source_sign_disagree},
        "tracks": decorated,
    }


def evaluate_mechanisms(acc: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    gates = config["gates"]
    clean = acc["clean_18hit"]
    n_clean = int(acc["n_clean_18hit"])
    enough = n_clean >= int(gates["min_clean_contract"])
    sign_agree = clean.get("sign_agree")
    spearman = clean.get("spearman_bending_vs_qp_truth")
    spearman_x = clean.get("spearman_bending_x_vs_qp_truth")
    present = STATUS_UNRESOLVED
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
    limited = STATUS_UNRESOLVED
    high_n = int(clean.get("high_p_n_sign") or 0)
    high_agree = clean.get("high_p_sign_agree")
    low_agree = clean.get("low_p_sign_agree")
    if enough and high_n >= int(gates["min_high_p"]) and high_agree is not None and low_agree is not None:
        if float(low_agree) - float(high_agree) >= float(gates["high_p_sign_drop"]):
            limited = STATUS_SUPPORTED
        else:
            limited = STATUS_REJECTED
    fit_extra = STATUS_UNRESOLVED
    frac = clean.get("fit_flip_given_same_sign")
    if enough and clean.get("n_same_sign_bending_truth"):
        if frac is not None and float(frac) >= float(gates["fit_extra_flip_frac"]):
            fit_extra = STATUS_SUPPORTED
        else:
            fit_extra = STATUS_REJECTED
    source = STATUS_UNRESOLVED
    stability = acc.get("source_stability") or {}
    if (
        enough
        and int(stability.get("n_construction_clean") or 0) > 0
        and int(stability.get("n_validation_clean") or 0) > 0
        and stability.get("rel_range") is not None
    ):
        if bool(stability.get("sign_disagree")) or float(stability["rel_range"]) >= float(gates["source_rel_range"]):
            source = STATUS_SUPPORTED
        else:
            source = STATUS_REJECTED
    supported = [
        name
        for name, status in (
            (MECH_PRESENT, present),
            (MECH_LIMITED, limited),
            (MECH_FIT, fit_extra),
            (MECH_SOURCE, source),
        )
        if status == STATUS_SUPPORTED
    ]
    official = supported[0] if len(supported) == 1 else MECH_MIXED
    return {
        "raw_bending_information_present": present,
        "raw_bending_information_limited_at_high_p": limited,
        "fit_additional_sign_failure": fit_extra,
        "raw_measurement_bias/source_dependence": source,
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


def inventory_campaign(config: Mapping[str, Any], campaign: str) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    sources_present: dict[str, bool] = {}
    for split, key in (("construction", "construction_sources"), ("validation", "validation_sources")):
        for spec in config["mc_data"][key]:
            source_id = spec["source_id"]
            path = dump_path_for_source(config, source_id, "tracks", campaign)
            present = path.is_file()
            sources_present[source_id] = present
            for row in iter_jsonl(path, split=split):
                if row.get("kind") != "track":
                    continue
                if row.get("collection") not in (None, SOURCE_COLLECTION_NAME):
                    continue
                rows.append(row)
    acc = evaluate_tracks(rows, config)
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
    elif campaign not in {"smoke", "contract"}:
        raise ThreeStQpMeasurementBendingError("S2I batch/HTCondor is not authorized")
    elif campaign == "batch" and n_clean >= min_recorded:
        decision = DECISION_RECORDED
        verdict = "PASS"
        diagnosis = "RECORDED"
    else:
        decision = DECISION_CONTRACT
        verdict = "PASS"
        diagnosis = "INCONCLUSIVE" if not mechanisms["enough_clean_for_contract"] else "RECORDED"
    clean = report.get("clean_18hit") or {}
    return {
        "kind": "three_st_qp_measurement_bending_contract",
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
        "clean_18hit": {k: v for k, v in clean.items() if k != "class_counts"} | {
            "class_counts": clean.get("class_counts")
        },
        "complete_three_st": report.get("complete_three_st"),
        "dirty": report.get("dirty"),
        "source_stability": report.get("source_stability"),
        "wb119_sign_flip_full_sample": {
            "n_expected": 243758,
            "quantified": False,
            "reason": "smoke_or_small_sample_cannot_cover_WB119_batch",
        },
        "three_st_qp_trusted_observable": False,
        "residual_conditional_authorized": False,
        "s2_flipped_to_pass": False,
        "official_qp_like_jacobian_authorized": False,
        "measurement_uncertainty_propagated": False,
        "sigma_b_invented": False,
        "e_r_ift_given_qp_computed": False,
        "e_r_ift_given_bending_computed": False,
        "large_dump_submitted": False,
        "htcondor_submitted": False,
        "front_provenance_chased": False,
        "dumps_materialized": dumps_ok,
        "inherited_workbook_119": inherited.get("workbook_119", {}).get("decision"),
        "inherited_workbook_124": inherited.get("workbook_124", {}).get("decision"),
        "conversion_chain": conversion_chain(),
        "official_quantities": official_quantities(),
    }
