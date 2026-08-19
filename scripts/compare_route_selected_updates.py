#!/usr/bin/env python3
"""Compare route-selected multi-DoF alignment updates across splits.

Reads two or more ``run_route_selected_multidof_update.py`` output directories
(typically the train-source and validation-source closures of one alignment
iteration) and reports, side by side, the frozen closure metrics: recovered
local delta, held-out error against the known MC target, normal-matrix rank
and condition number, parameter correlation, common-route fraction, response
chi2/ndof, per-payload anchor-edge availability, and a per-source WLS
re-solve of the same frozen observations (no refit of the pooled update, no
hyperparameter selection).  Optionally folds in the anchor backbone's
association summary for candidate/route coverage.

The comparison is read-only and never opens test data.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

PARAMETER_ORDER = ("ift_dx_mm", "ift_dy_mm", "ift_ry_mrad")
FROZEN_TOLERANCES = {"ift_dx_mm": 0.1, "ift_dy_mm": 0.1, "ift_ry_mrad": 1.0}
_NAMESPACE_ORIGIN = 9_000_000_000


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_update(update_dir: Path, label: str) -> dict[str, Any]:
    summary_path = update_dir / "route_selected_update.json"
    arrays_path = update_dir / "route_selected_update_arrays.npz"
    if not summary_path.is_file() or not arrays_path.is_file():
        raise FileNotFoundError(f"incomplete route-selected update output: {update_dir}")
    summary = _read_json(summary_path)
    arrays = np.load(arrays_path, allow_pickle=False)
    parameters = {str(row["name"]): row for row in summary["parameters"]}
    names = [str(value) for value in summary["probe_points"].keys()]
    tolerances = {
        name: (
            float(parameters[name]["capture_tolerance"])
            if parameters[name].get("capture_tolerance") is not None
            else FROZEN_TOLERANCES[name]
        )
        for name in names
    }
    contract = summary["association_contract"]
    availability = {"target": contract["target"].get("anchor_edge_audit", {}).get("anchor_edge_availability")}
    for sign, group in (("plus", contract["positive"]), ("minus", contract["negative"])):
        for name, entry in group.items():
            availability[f"{name}_{sign}"] = entry.get("anchor_edge_audit", {}).get("anchor_edge_availability")
    return {
        "label": label,
        "update_dir": str(update_dir),
        "summary": summary,
        "arrays": arrays,
        "parameter_names": names,
        "capture_tolerances": tolerances,
        "capture_success": summary.get("capture_success"),
        "normal_matrix_rank": int(summary["normal_matrix_rank"]),
        "normal_matrix_condition_number": float(summary["normal_matrix_condition_number"]),
        "parameter_correlation": np.asarray(summary["parameter_correlation"], dtype=np.float64),
        "response_chi2": float(summary["response_chi2"]),
        "response_ndof": int(summary["response_ndof"]),
        "common_observations": int(contract["selected_route_overlap"]["common_selected_observations"]),
        "anchor_common_fraction": contract["selected_route_overlap"]["anchor_common_fraction"],
        "anchor_edge_availability_by_payload": availability,
    }


def _source_index_from_key(key_json: str) -> int:
    key = json.loads(key_json)
    signature = json.loads(key[3])
    origin_run = int(signature[0]["origin_run_id"])
    return (origin_run - _NAMESPACE_ORIGIN) // 100


def _per_source_solve(item: Mapping[str, Any], source_ids: Sequence[str]) -> dict[str, Any]:
    """Re-solve the frozen FD system per source; no pooled refit occurs here."""
    arrays = item["arrays"]
    keys = [str(value) for value in arrays["observation_keys"]]
    response = np.asarray(arrays["response"], dtype=np.float64)
    derivative = np.asarray(arrays["derivative_native"], dtype=np.float64)
    covariance = np.asarray(arrays["covariance"], dtype=np.float64)
    names = [str(value) for value in arrays["parameter_names"]]
    by_source: dict[int, list[int]] = {}
    for row, key in enumerate(keys):
        by_source.setdefault(_source_index_from_key(key), []).append(row)
    per_source: dict[str, Any] = {}
    for source_index, rows in sorted(by_source.items()):
        rows_array = np.asarray(rows, dtype=int)
        design = derivative[rows_array].reshape(len(rows) * 4, len(names))
        target = response[rows_array].reshape(len(rows) * 4)
        covariance_blocks = np.zeros((len(rows) * 4, len(rows) * 4), dtype=np.float64)
        for block_index, row in enumerate(rows):
            covariance_blocks[
                block_index * 4 : block_index * 4 + 4, block_index * 4 : block_index * 4 + 4
            ] = covariance[row]
        weight = np.linalg.inv(covariance_blocks)
        normal = design.T @ weight @ design
        try:
            delta = np.linalg.solve(normal, design.T @ weight @ target)
        except np.linalg.LinAlgError:
            delta = np.full(len(names), np.nan)
        source_label = (
            str(source_ids[source_index]) if source_index < len(source_ids) else f"index_{source_index}"
        )
        per_source[source_label] = {
            "observations": int(len(rows)),
            "recovered_local_delta": {name: float(delta[i]) for i, name in enumerate(names)},
        }
    spread = {}
    for i, name in enumerate(names):
        values = np.asarray(
            [entry["recovered_local_delta"][name] for entry in per_source.values()], dtype=np.float64
        )
        finite = values[np.isfinite(values)]
        spread[name] = {
            "std": float(np.std(finite)) if finite.size else None,
            "min": float(np.min(finite)) if finite.size else None,
            "max": float(np.max(finite)) if finite.size else None,
        }
    return {"per_source": per_source, "spread": spread}


def _anchor_coverage(anchor_association_dir: Path) -> dict[str, Any]:
    summary = _read_json(anchor_association_dir / "association_summary.json")
    truth = summary.get("truth_labelled_mc_evaluation", {})
    by_pair = {
        pair: {
            "association_efficiency": values["association"]["association_efficiency"],
            "association_purity": values["association"]["association_purity"],
        }
        for pair, values in truth.get("by_station_pair", {}).items()
    }
    selection = summary.get("truth_free_selection", {})
    return {
        "selected_routes": truth.get("assignment", {}).get("selected_routes"),
        "complete_four_station_routes": selection.get("complete_four_station_routes"),
        "selected_field_aware_edges": summary.get("field_aware_route_consistency", {}).get(
            "selected_field_aware_edges"
        ),
        "by_station_pair": by_pair,
    }


def _parameter_row(item: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    rows = {}
    for name, parameter in ((p["name"], p) for p in item["summary"]["parameters"]):
        rows[str(name)] = {
            "anchor_value": float(parameter["anchor_value"]),
            "recovered_local_delta": float(parameter["recovered_local_delta"]),
            "local_delta_error": float(parameter["local_delta_error"]),
            "recovered_sigma": parameter["recovered_sigma"],
            "capture_tolerance": item["capture_tolerances"][str(name)],
            "capture_success": bool(parameter["capture_success"])
            if parameter.get("capture_success") is not None
            else abs(float(parameter["local_delta_error"])) <= item["capture_tolerances"][str(name)],
        }
    return rows


def _workbook_fragment(report: Mapping[str, Any]) -> str:
    lines = [
        "## route-selected update train/validation 并排比较（自动生成）",
        "",
        "| 指标 | " + " | ".join(item["label"] for item in report["iterations"]) + " |",
        "| --- | " + " | ".join("---" for _ in report["iterations"]) + " |",
    ]
    for name in report["parameter_names"]:
        tol = report["iterations"][0]["parameters"][name]["capture_tolerance"]
        cells = []
        for item in report["iterations"]:
            parameter = item["parameters"][name]
            cells.append(
                f"Δ{parameter['recovered_local_delta']:+.4f} 误差{parameter['local_delta_error']:+.4f} "
                f"(σ={parameter['recovered_sigma']:.4f})"
            )
        lines.append(f"| {name}（容差 {tol}） | " + " | ".join(cells) + " |")
    for label, key in (
        ("rank", "normal_matrix_rank"),
        ("条件数", "normal_matrix_condition_number"),
        ("公共路由边数", "common_observations"),
        ("anchor 边保留率", "anchor_common_fraction"),
        ("response χ²/ndof", "response_chi2_per_ndof"),
        ("dx-Ry 相关", "corr_dx_ry"),
        ("逐源散布 dx", "spread_ift_dx_mm"),
        ("逐源散布 dy", "spread_ift_dy_mm"),
        ("逐源散布 Ry", "spread_ift_ry_mrad"),
    ):
        cells = []
        for item in report["iterations"]:
            value = item[key]
            cells.append("—" if value is None else (f"{value:.4g}" if isinstance(value, float) else str(value)))
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines.append(f"| 冻结判定 | " + " | ".join(
        "通过" if item["capture_success"] else "未通过" for item in report["iterations"]
    ) + " |")
    lines.append("")
    lines.append(f"冻结判定结果：`{report['decision']}` — {report['decision_reason']}")
    return "\n".join(lines) + "\n"


def frozen_decision(iterations: Sequence[Mapping[str, Any]]) -> tuple[str, str]:
    last = iterations[-1]
    failures = [
        name
        for name in last["parameter_names"]
        if abs(float(last["parameters"][name]["local_delta_error"])) > float(last["capture_tolerances"][name])
    ]
    if not failures:
        return (
            "validation_closure_confirmed",
            "validation source-disjoint closure passes all frozen tolerances "
            "(|dx|<=0.1 mm, |dy|<=0.1 mm, |Ry|<=1 mrad); record as the first "
            "unknown-association source-disjoint multi-DoF iterative alignment closure",
        )
    return (
        "hold_and_diagnose",
        "validation closure fails frozen tolerances for "
        + ", ".join(failures)
        + "; do not retune thresholds or retrain — compare selected-route composition, "
        "source-wise Jacobians, common-route fraction, and covariance-weighted residuals first",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--update", action="append", required=True, metavar="DIR")
    parser.add_argument("--label", action="append", required=True)
    parser.add_argument(
        "--anchor-association",
        action="append",
        default=None,
        metavar="LABEL:DIR",
        help="Anchor backbone output dir per update label for coverage metrics.",
    )
    parser.add_argument(
        "--physical-manifest",
        default=None,
        help="Assembled physical corpus manifest providing split source ordering.",
    )
    parser.add_argument("--split", default=None, help="Split of the last update (for source naming).")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if len(args.update) != len(args.label):
        raise ValueError("--update and --label must be supplied the same number of times")

    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)

    source_ids: list[str] = []
    if args.physical_manifest is not None and args.split is not None:
        manifest = _read_json(Path(args.physical_manifest).expanduser().resolve())
        source_ids = sorted(
            str(source["source_id"])
            for source in manifest["sources"]
            if str(source["split"]) == args.split
        )

    coverage_by_label = {}
    for entry in args.anchor_association or ():
        label, _, directory = entry.partition(":")
        coverage_by_label[label] = _anchor_coverage(Path(directory).expanduser().resolve())

    iterations = []
    for directory, label in zip(args.update, args.label):
        item = _load_update(Path(directory).expanduser().resolve(), label)
        item["parameters"] = _parameter_row(item)
        if source_ids:
            item["per_source"] = _per_source_solve(item, source_ids)
        if label in coverage_by_label:
            item["anchor_coverage"] = coverage_by_label[label]
        iterations.append(item)

    report_iterations = []
    for item in iterations:
        correlation = item["parameter_correlation"]
        names = item["parameter_names"]
        dx_ry = None
        if "ift_dx_mm" in names and "ift_ry_mrad" in names:
            dx_ry = float(correlation[names.index("ift_dx_mm"), names.index("ift_ry_mrad")])
        spread = item.get("per_source", {}).get("spread", {})
        report_iterations.append(
            {
                "label": item["label"],
                "parameter_names": item["parameter_names"],
                "parameters": item["parameters"],
                "capture_tolerances": item["capture_tolerances"],
                "capture_success": item["capture_success"],
                "normal_matrix_rank": item["normal_matrix_rank"],
                "normal_matrix_condition_number": item["normal_matrix_condition_number"],
                "parameter_correlation": correlation.tolist(),
                "corr_dx_ry": dx_ry,
                "response_chi2": item["response_chi2"],
                "response_ndof": item["response_ndof"],
                "response_chi2_per_ndof": item["response_chi2"] / max(item["response_ndof"], 1),
                "common_observations": item["common_observations"],
                "anchor_common_fraction": item["anchor_common_fraction"],
                "anchor_edge_availability_by_payload": item["anchor_edge_availability_by_payload"],
                "spread_ift_dx_mm": spread.get("ift_dx_mm", {}).get("std"),
                "spread_ift_dy_mm": spread.get("ift_dy_mm", {}).get("std"),
                "spread_ift_ry_mrad": spread.get("ift_ry_mrad", {}).get("std"),
                "per_source": item.get("per_source", {}).get("per_source"),
                "anchor_coverage": item.get("anchor_coverage"),
            }
        )
    decision, reason = frozen_decision(report_iterations)
    report = {
        "method": "route_selected_update_split_comparison",
        "test_data_accessed": False,
        "parameter_names": report_iterations[-1]["parameter_names"],
        "iterations": report_iterations,
        "decision": decision,
        "decision_reason": reason,
    }
    (output / "route_selected_update_comparison.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output / "route_selected_update_comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["label", "parameter", "anchor", "recovered_delta", "error", "sigma", "tolerance", "pass"]
        )
        for item in report_iterations:
            for name in item["parameter_names"]:
                parameter = item["parameters"][name]
                writer.writerow(
                    [
                        item["label"],
                        name,
                        parameter["anchor_value"],
                        parameter["recovered_local_delta"],
                        parameter["local_delta_error"],
                        parameter["recovered_sigma"],
                        parameter["capture_tolerance"],
                        parameter["capture_success"],
                    ]
                )
    (output / "workbook_fragment_cn.md").write_text(
        _workbook_fragment(report), encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(output), "decision": decision}, indent=2))


if __name__ == "__main__":
    main()
