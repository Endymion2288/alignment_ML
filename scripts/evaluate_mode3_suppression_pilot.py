#!/usr/bin/env python3
"""Evaluate the fixed-q/p-seed covariance-suppression (mode-3) pilot.

Assembles the joint pilot verdict from already produced artifacts:

* same-truth-edge mode 0/3/1 comparison on the pilot physical propagations
  (residual identity, propagated/combined sigma, covariance eigenvalues and
  condition number, pulls, chi2 tails) per station pair and payload point;
* candidate-coverage audits of the mode-0 and mode-3 pilot overlays
  (0->1 truth-edge recall, complete-chain candidate coverage);
* frozen V2 backbone outputs on the identical overlay events under both
  covariance variants (truth/fake separation, route efficiency/purity/fake);
* route-selected dx/dy/Ry closure under both variants with the frozen
  physical_edge_deduplicated observation semantics.

The promotion criteria are the frozen pilot gate: residuals unchanged,
q/p-induced covariance inflation gone, truth/fake separation or candidate
coverage actually improved, route metrics not degraded, and the multi-DoF
closure still inside the frozen capture tolerances.  The script never touches
test data, never re-trains, and never modifies production artifacts.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from datasets.propagation_loader import load_propagation_records
from datasets.root_loader import load_events
from geometry.propagation import mahalanobis_chi2

ADJACENT_PAIRS: tuple[tuple[int, int], ...] = ((0, 1), (1, 2), (2, 3))
MODES: tuple[int, ...] = (0, 3, 1)
SCHEMA_VERSION = "faser-mode3-suppression-pilot-eval-v1"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _median(values: Sequence[float]) -> float | None:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return None
    return float(np.median(array))


def _robust_sigma(values: Sequence[float]) -> float | None:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return None
    median = np.median(array)
    return float(1.4826 * np.median(np.abs(array - median)))


def _quantile(values: Sequence[float], q: float) -> float | None:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return None
    return float(np.quantile(array, q))


def collect_same_edge_budget(
    tracklets: Path, propagations: Path
) -> dict[tuple[str, int], dict[str, list[float]]]:
    """Per (pair, mode) same-edge ingredients for truth-matched edges."""
    events = load_events(tracklets, require_mc_labels=True)
    records = load_propagation_records(propagations)
    rows_by_key: dict[tuple[int, int, int, int, int], int] = {}
    for row in range(records.size):
        if not (bool(records.success[row]) and bool(records.has_covariance[row])):
            continue
        mode = int(records.q_over_p_mode[row]) if records.q_over_p_mode is not None else 0
        key = (
            int(records.run_id[row]),
            int(records.event_id[row]),
            int(records.source_tracklet_id[row]),
            int(records.target_tracklet_id[row]),
            mode,
        )
        rows_by_key.setdefault(key, row)

    out: dict[tuple[str, int], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for event in events:
        if event.truth_particle_id is None:
            continue
        for source_station, target_station in ADJACENT_PAIRS:
            pair = f"{source_station}->{target_station}"
            per_station: dict[int, dict[int, int]] = {}
            for index in range(event.size):
                station = int(event.station_id[index])
                truth = int(event.truth_particle_id[index])
                per_station.setdefault(station, {})[truth] = index
            shared = sorted(
                set(per_station.get(source_station, {})) & set(per_station.get(target_station, {}))
            )
            for truth_id in shared:
                source_index = per_station[source_station][truth_id]
                target_index = per_station[target_station][truth_id]
                base = (
                    int(event.run_id),
                    int(event.event_id),
                    int(event.tracklet_id[source_index]),
                    int(event.tracklet_id[target_index]),
                )
                for mode in MODES:
                    row = rows_by_key.get((*base, mode))
                    if row is None:
                        continue
                    propagated = np.asarray(records.covariance[row], dtype=np.float64)
                    target_cov = np.asarray(event.covariance[target_index], dtype=np.float64)
                    combined = propagated + target_cov
                    residual = np.asarray(
                        event.state[target_index], dtype=np.float64
                    ) - np.asarray(records.prediction[row], dtype=np.float64)
                    try:
                        chi2 = mahalanobis_chi2(residual, combined)
                    except ValueError:
                        chi2 = float("nan")
                    eig = np.linalg.eigvalsh(combined)
                    budget = out[(pair, mode)]
                    budget["count"].append(1.0)
                    budget["residual_x"].append(float(residual[0]))
                    budget["residual_y"].append(float(residual[1]))
                    budget["prop_sigma_x"].append(float(math.sqrt(max(propagated[0, 0], 0.0))))
                    budget["prop_sigma_y"].append(float(math.sqrt(max(propagated[1, 1], 0.0))))
                    budget["combined_sigma_x"].append(float(math.sqrt(max(combined[0, 0], 0.0))))
                    budget["combined_sigma_y"].append(float(math.sqrt(max(combined[1, 1], 0.0))))
                    budget["chi2"].append(chi2)
                    budget["min_eigenvalue"].append(float(eig[0]))
                    budget["max_eigenvalue"].append(float(eig[-1]))
                    budget["condition_number"].append(
                        float(eig[-1] / eig[0]) if eig[0] > 0.0 else float("inf")
                    )
                    for i, component in enumerate(("x", "y")):
                        sigma = math.sqrt(max(combined[i, i], 0.0))
                        budget[f"pull_{component}"].append(
                            float(residual[i] / sigma) if sigma > 0.0 else float("nan")
                        )
    return out


def summarize_budget(budget: dict[str, list[float]]) -> dict[str, Any]:
    chi2 = np.asarray(budget["chi2"], dtype=np.float64)
    chi2 = chi2[np.isfinite(chi2)]
    return {
        "count": int(len(budget["count"])),
        "residual_x_robust_sigma_mm": _robust_sigma(budget["residual_x"]),
        "residual_y_robust_sigma_mm": _robust_sigma(budget["residual_y"]),
        "propagated_sigma_x_median_mm": _median(budget["prop_sigma_x"]),
        "propagated_sigma_y_median_mm": _median(budget["prop_sigma_y"]),
        "combined_sigma_x_median_mm": _median(budget["combined_sigma_x"]),
        "combined_sigma_y_median_mm": _median(budget["combined_sigma_y"]),
        "pull_x_robust_sigma": _robust_sigma(budget["pull_x"]),
        "pull_y_robust_sigma": _robust_sigma(budget["pull_y"]),
        "chi2_median": float(np.median(chi2)) if chi2.size else None,
        "chi2_q99": _quantile(chi2, 0.99),
        "chi2_frac_above_100": float(np.mean(chi2 > 100.0)) if chi2.size else None,
        "min_eigenvalue_median": _median(budget["min_eigenvalue"]),
        "condition_number_median": _median(budget["condition_number"]),
        "condition_number_q99": _quantile(budget["condition_number"], 0.99),
    }


def _coverage_recall(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    overall = payload["overall"]
    by_pair = payload["by_station_pair"]
    return {
        "raw_physical_candidate_truth_recall": overall["raw_physical_candidate_truth_recall"],
        "recall_0_to_1": by_pair.get("0->1", {}).get("raw_physical_candidate_truth_recall"),
        "recall_1_to_2": by_pair.get("1->2", {}).get("raw_physical_candidate_truth_recall"),
        "recall_2_to_3": by_pair.get("2->3", {}).get("raw_physical_candidate_truth_recall"),
        "eligibility_reasons": overall["eligibility_reasons"],
    }


def _backbone_metrics(path: Path) -> dict[str, Any]:
    summary = _read_json(path / "iteration_01_anchor" / "association_summary.json")
    evaluation = summary["truth_labelled_mc_evaluation"]
    route = evaluation["route"]
    scores = summary["raw_candidate_score_metrics"]
    per_pair = {}
    for pair, payload in sorted(evaluation["by_station_pair"].items()):
        association = payload["association"]
        per_pair[pair] = {
            "association_efficiency": association["association_efficiency"],
            "association_purity": association["association_purity"],
            "fake_rate": association["fake_rate"],
        }
    score_per_pair = {
        pair: {
            "roc_auc": payload.get("roc_auc"),
            "average_precision": payload.get("average_precision"),
        }
        for pair, payload in sorted(scores["by_station_pair"].items())
    }
    return {
        "complete_track_efficiency": route["complete_track_efficiency"],
        "complete_track_purity": route["complete_track_purity"],
        "fake_endpoint_route_rate": route["fake_endpoint_route_rate"],
        "candidate_complete_truth_chain_recall": route["candidate_complete_truth_chain_recall"],
        "per_pair_association": per_pair,
        "per_pair_score_separation": score_per_pair,
        "overall_roc_auc": scores.get("roc_auc"),
        "overall_average_precision": scores.get("average_precision"),
    }


def _closure_metrics(path: Path) -> dict[str, Any]:
    payload = _read_json(path / "route_selected_update.json")
    return {
        "capture_success": bool(payload["capture_success"]),
        "observation_statistics": payload.get("observation_statistics"),
        "q_over_p_mode": payload.get("q_over_p_mode"),
        "parameters": {
            str(parameter["name"]): {
                "local_delta_error": parameter.get("local_delta_error"),
                "recovered_local_delta": parameter.get("recovered_local_delta"),
                "recovered_sigma": parameter.get("recovered_sigma"),
                "capture_success": bool(parameter.get("capture_success")),
                "capture_tolerance": parameter.get("capture_tolerance"),
            }
            for parameter in payload["parameters"]
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-physical-manifest", required=True)
    parser.add_argument("--points", nargs="+", required=True)
    parser.add_argument("--coverage-mode0", required=True)
    parser.add_argument("--coverage-mode3", required=True)
    parser.add_argument("--backbone-mode0", required=True)
    parser.add_argument("--backbone-mode3", required=True)
    parser.add_argument("--closure-mode0", required=True)
    parser.add_argument("--closure-mode3", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)

    manifest = _read_json(Path(args.pilot_physical_manifest).expanduser().resolve())

    same_edge: dict[str, dict[str, Any]] = {}
    for point_name in args.points:
        pooled: dict[tuple[str, int], dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for entry in manifest["sources"]:
            root = Path(str(entry["physical_scan_root"])).expanduser().resolve()
            plan = _read_json(root / "scan_plan.json")
            points = {str(point["name"]): point for point in plan["points"]}
            point = points.get(point_name)
            if point is None:
                raise ValueError(f"point '{point_name}' absent from {root}/scan_plan.json")
            point_root = root / str(point["relative_point_dir"])
            source_budget = collect_same_edge_budget(
                point_root / "refit" / "tracklets.root",
                point_root / "refit" / "propagations.root",
            )
            for key, payload in source_budget.items():
                for name, values in payload.items():
                    pooled[key][name].extend(values)
        same_edge[point_name] = {
            f"{pair}/mode{mode}": summarize_budget(payload)
            for (pair, mode), payload in sorted(pooled.items())
        }

    coverage = {
        "mode0": _coverage_recall(Path(args.coverage_mode0).expanduser().resolve() / "coverage_audit.json"),
        "mode3": _coverage_recall(Path(args.coverage_mode3).expanduser().resolve() / "coverage_audit.json"),
    }
    backbone = {
        "mode0": _backbone_metrics(Path(args.backbone_mode0).expanduser().resolve()),
        "mode3": _backbone_metrics(Path(args.backbone_mode3).expanduser().resolve()),
    }
    closure = {
        "mode0": _closure_metrics(Path(args.closure_mode0).expanduser().resolve()),
        "mode3": _closure_metrics(Path(args.closure_mode3).expanduser().resolve()),
    }

    # Joint pilot gate.  Each criterion is evaluated on numbers, not on the
    # narrative: residual identity, covariance inflation removal, separation or
    # coverage gain, route-metric non-degradation, closure within tolerances.
    criteria: dict[str, Any] = {}
    residual_shifts = []
    sigma_y_ratios = []
    for point_payload in same_edge.values():
        for pair in ("0->1", "1->2", "2->3"):
            mode0 = point_payload.get(f"{pair}/mode0")
            mode3 = point_payload.get(f"{pair}/mode3")
            if not mode0 or not mode3 or not mode0["residual_x_robust_sigma_mm"]:
                continue
            residual_shifts.append(
                abs(mode3["residual_x_robust_sigma_mm"] - mode0["residual_x_robust_sigma_mm"])
                / mode0["residual_x_robust_sigma_mm"]
            )
            if mode0["combined_sigma_y_median_mm"]:
                sigma_y_ratios.append(
                    mode0["combined_sigma_y_median_mm"] / mode3["combined_sigma_y_median_mm"]
                )
    criteria["residual_unchanged"] = {
        "max_relative_robust_sigma_shift": max(residual_shifts) if residual_shifts else None,
        "pass": bool(residual_shifts) and max(residual_shifts) < 0.05,
    }
    criteria["covariance_inflation_removed"] = {
        "min_sigma_y_collapse_factor": min(sigma_y_ratios) if sigma_y_ratios else None,
        "pass": bool(sigma_y_ratios) and min(sigma_y_ratios) > 10.0,
    }
    separation_gain = (
        backbone["mode3"]["overall_roc_auc"] is not None
        and backbone["mode0"]["overall_roc_auc"] is not None
        and backbone["mode3"]["overall_roc_auc"] > backbone["mode0"]["overall_roc_auc"] + 0.005
    )
    coverage_gain = (
        coverage["mode3"]["raw_physical_candidate_truth_recall"] is not None
        and coverage["mode0"]["raw_physical_candidate_truth_recall"] is not None
        and coverage["mode3"]["raw_physical_candidate_truth_recall"]
        > coverage["mode0"]["raw_physical_candidate_truth_recall"] + 0.001
    )
    criteria["separation_or_coverage_improved"] = {
        "roc_auc_mode0": backbone["mode0"]["overall_roc_auc"],
        "roc_auc_mode3": backbone["mode3"]["overall_roc_auc"],
        "coverage_mode0": coverage["mode0"]["raw_physical_candidate_truth_recall"],
        "coverage_mode3": coverage["mode3"]["raw_physical_candidate_truth_recall"],
        "separation_gain": bool(separation_gain),
        "coverage_gain": bool(coverage_gain),
        "pass": bool(separation_gain or coverage_gain),
    }
    route_degradation = {
        "complete_track_efficiency_drop": (
            backbone["mode0"]["complete_track_efficiency"] - backbone["mode3"]["complete_track_efficiency"]
        ),
        "complete_track_purity_drop": (
            backbone["mode0"]["complete_track_purity"] - backbone["mode3"]["complete_track_purity"]
        ),
        "fake_endpoint_route_rate_increase": (
            backbone["mode3"]["fake_endpoint_route_rate"] - backbone["mode0"]["fake_endpoint_route_rate"]
        ),
    }
    criteria["route_metrics_not_degraded"] = {
        **route_degradation,
        "pass": bool(
            route_degradation["complete_track_efficiency_drop"] <= 0.005
            and route_degradation["complete_track_purity_drop"] <= 0.005
            and route_degradation["fake_endpoint_route_rate_increase"] <= 0.005
        ),
    }
    criteria["closure_within_frozen_tolerances"] = {
        "mode0_capture_success": closure["mode0"]["capture_success"],
        "mode3_capture_success": closure["mode3"]["capture_success"],
        "pass": bool(closure["mode0"]["capture_success"] and closure["mode3"]["capture_success"]),
    }
    verdict = {
        "schema_version": SCHEMA_VERSION,
        "pilot_physical_manifest": str(args.pilot_physical_manifest),
        "points": list(args.points),
        "same_edge_budget": same_edge,
        "candidate_coverage": coverage,
        "backbone_metrics": backbone,
        "closure_metrics": closure,
        "criteria": criteria,
        "all_criteria_pass": all(bool(item["pass"]) for item in criteria.values()),
        "test_data_accessed": False,
    }
    _write_json(output / "mode3_suppression_pilot_verdict.json", verdict)
    print(json.dumps({"all_criteria_pass": verdict["all_criteria_pass"], "criteria": {
        name: item["pass"] for name, item in criteria.items()
    }}, indent=2))


if __name__ == "__main__":
    main()
