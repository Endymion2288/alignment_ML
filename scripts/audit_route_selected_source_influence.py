#!/usr/bin/env python3
"""Per-source leverage/influence diagnosis of a frozen route-selected update.

Read-only re-analysis of an existing route-selected multi-DoF update: from the
stored observation arrays (anchor/target residuals, anchor covariance, native
FD derivatives) rebuild the pooled normal equation, then decompose it by
source file.  Per source reports normal-matrix share (leverage), projection
onto the weakest constrained eigenmode, anchor-residual chi2 tail structure,
the tail's contribution to the weak-mode right-hand side, per-source Jacobian
orientation versus the pooled weak mode, and the exact leave-one-source-out
update shift.  No source is dropped or reweighted in any stored product; the
LOSO solves are diagnostics only.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from scripts.compare_route_selected_updates import _source_index_from_key
from scripts.run_multisource_refit_multidof_local_step import _read_json

SCHEMA_VERSION = "faser-route-selected-source-influence-v1"
TAIL_GATES = (25.0, 100.0, 500.0)


def _inverse(blocks: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.inv(blocks)
    except np.linalg.LinAlgError:
        return np.asarray([np.linalg.pinv(block) for block in blocks], dtype=np.float64)


def _ordered_source_ids(manifest: Mapping[str, Any], split: str) -> list[str]:
    return [
        str(entry["source_id"])
        for entry in manifest["sources"]
        if str(entry["split"]) == split
    ]


def _eigensystem(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    symmetric = 0.5 * (normal + normal.T)
    values, vectors = np.linalg.eigh(symmetric)
    order = np.argsort(values)
    return values[order], vectors[:, order]


def analyze_update(update_dir: Path, source_ids: Sequence[str]) -> dict[str, Any]:
    arrays = np.load(update_dir / "route_selected_update_arrays.npz")
    summary = _read_json(update_dir / "route_selected_update.json")
    keys = [str(value) for value in arrays["observation_keys"]]
    names = [str(value) for value in arrays["parameter_names"]]
    response = np.asarray(arrays["response"], dtype=np.float64)
    anchor_residual = np.asarray(arrays["anchor_residual"], dtype=np.float64)
    derivative = np.asarray(arrays["derivative_native"], dtype=np.float64)
    covariance = np.asarray(arrays["covariance"], dtype=np.float64)
    native_normal = np.asarray(arrays["normal_matrix_native"], dtype=np.float64)
    scaled_normal = np.asarray(arrays["normal_matrix_scaled"], dtype=np.float64)
    scales = np.sqrt(np.diag(scaled_normal) / np.diag(native_normal))

    weights = _inverse(covariance)
    n_edges = len(keys)
    per_edge_normal = np.einsum("nij,njk->nik", np.einsum("npi,nij->npj", derivative.transpose(0, 2, 1), weights), derivative)
    per_edge_rhs = np.einsum("npi,nij,nj->np", derivative.transpose(0, 2, 1), weights, response)
    normal = per_edge_normal.sum(axis=0)
    rhs = per_edge_rhs.sum(axis=0)
    pooled_delta = np.linalg.solve(normal, rhs)

    # Eigen-analysis in the solver's scaled coordinates.
    scale_matrix = np.diag(scales)
    normal_scaled = scale_matrix @ normal @ scale_matrix
    eigenvalues, eigenvectors_scaled = _eigensystem(normal_scaled)
    weakest_scaled = eigenvectors_scaled[:, 0]
    weakest_native = scale_matrix @ weakest_scaled
    weakest_native /= np.linalg.norm(weakest_native)

    edge_chi2 = np.einsum("ni,nij,nj->n", anchor_residual, weights, anchor_residual)

    by_source: dict[int, list[int]] = {}
    for row, key in enumerate(keys):
        by_source.setdefault(_source_index_from_key(key), []).append(row)

    weak_mode_total_information = float(weakest_native @ normal @ weakest_native)
    weak_mode_total_rhs = float(weakest_native @ rhs)

    per_source: list[dict[str, Any]] = []
    for source_index, rows in sorted(by_source.items()):
        rows_array = np.asarray(rows, dtype=int)
        source_normal = per_edge_normal[rows_array].sum(axis=0)
        source_rhs = per_edge_rhs[rows_array].sum(axis=0)
        label = (
            str(source_ids[source_index])
            if source_index < len(source_ids)
            else f"index_{source_index}"
        )
        leverage = float(np.trace(np.linalg.solve(normal, source_normal)))
        weak_info = float(weakest_native @ source_normal @ weakest_native)
        weak_rhs = float(weakest_native @ source_rhs)
        # Exact leave-one-source-out re-solve (linear WLS: closed form).
        reduced_normal = normal - source_normal
        reduced_rhs = rhs - source_rhs
        try:
            loso_delta = np.linalg.solve(reduced_normal, reduced_rhs)
            loso_shift = loso_delta - pooled_delta
        except np.linalg.LinAlgError:
            loso_delta = np.full(len(names), np.nan)
            loso_shift = np.full(len(names), np.nan)
        # Per-source Jacobian orientation: weakest direction of the source's
        # own normal matrix in scaled coordinates.
        source_normal_scaled = scale_matrix @ source_normal @ scale_matrix
        source_eigenvalues, source_eigenvectors = _eigensystem(source_normal_scaled)
        source_weakest_native = scale_matrix @ source_eigenvectors[:, 0]
        source_weakest_native /= np.linalg.norm(source_weakest_native)
        weak_alignment = float(abs(source_weakest_native @ weakest_native))
        chi2_values = edge_chi2[rows_array]
        rhs_projection_edges = (per_edge_rhs[rows_array] @ weakest_native)
        tail_contribution = {}
        for gate in TAIL_GATES:
            mask = chi2_values > gate
            tail_contribution[f"chi2_gt_{int(gate)}"] = {
                "edges": int(np.sum(mask)),
                "weak_rhs_share_of_source": (
                    None
                    if abs(weak_rhs) < 1e-300
                    else float(np.sum(rhs_projection_edges[mask]) / weak_rhs)
                ),
            }
        per_source.append(
            {
                "source_id": label,
                "observations": int(len(rows)),
                "leverage_trace_share": leverage,
                "weak_mode_information_share": (
                    None if weak_mode_total_information == 0.0 else weak_info / weak_mode_total_information
                ),
                "weak_mode_rhs": weak_rhs,
                "weak_mode_rhs_share_of_pooled": (
                    None if abs(weak_mode_total_rhs) < 1e-300 else weak_rhs / weak_mode_total_rhs
                ),
                "anchor_chi2_median": float(np.median(chi2_values)),
                "anchor_chi2_q95": float(np.quantile(chi2_values, 0.95)),
                "anchor_chi2_q99": float(np.quantile(chi2_values, 0.99)),
                "anchor_chi2_max": float(np.max(chi2_values)),
                "tail_contribution": tail_contribution,
                "jacobian_weakest_alignment_with_pooled": weak_alignment,
                "jacobian_weakest_eigenvalue_share": float(
                    source_eigenvalues[0] / max(float(np.sum(source_eigenvalues)), 1e-300)
                ),
                "loso_shift": {name: float(loso_shift[i]) for i, name in enumerate(names)},
                "loso_shift_along_weak_mode": float(weakest_native @ loso_shift),
            }
        )

    return {
        "update_dir": str(update_dir),
        "parameter_names": names,
        "parameter_scales": scales.tolist(),
        "edges": n_edges,
        "pooled_delta_from_arrays": {
            name: float(pooled_delta[i]) for i, name in enumerate(names)
        },
        "pooled_delta_from_summary": {
            str(p["name"]): float(p["recovered_local_delta"]) for p in summary["parameters"]
        },
        "expected_delta": {
            str(p["name"]): float(p["expected_delta_to_target"]) for p in summary["parameters"]
        },
        "scaled_eigenvalues": eigenvalues.tolist(),
        "weakest_mode_native_direction": {
            name: float(weakest_native[i]) for i, name in enumerate(names)
        },
        "weakest_mode_eigenvalue_share": float(eigenvalues[0] / np.sum(eigenvalues)),
        "per_source": per_source,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--update", action="append", required=True, metavar="LABEL:DIR")
    parser.add_argument("--physical-manifest", required=True)
    parser.add_argument("--split", default="validation", choices=("train", "validation"))
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = _read_json(Path(args.physical_manifest).expanduser().resolve())
    source_ids = _ordered_source_ids(manifest, args.split)

    analyses = []
    for spec in args.update:
        label, _, path = spec.partition(":")
        analyses.append({"label": label, **analyze_update(Path(path).expanduser().resolve(), source_ids)})

    (output / "source_influence.json").write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "split": args.split,
                "test_data_accessed": False,
                "analyses": analyses,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    rows: list[dict[str, Any]] = []
    for analysis in analyses:
        for entry in analysis["per_source"]:
            row = {
                "label": analysis["label"],
                "source_id": entry["source_id"],
                "observations": entry["observations"],
                "leverage_trace_share": entry["leverage_trace_share"],
                "weak_mode_information_share": entry["weak_mode_information_share"],
                "weak_mode_rhs": entry["weak_mode_rhs"],
                "anchor_chi2_median": entry["anchor_chi2_median"],
                "anchor_chi2_q99": entry["anchor_chi2_q99"],
                "jacobian_weakest_alignment_with_pooled": entry[
                    "jacobian_weakest_alignment_with_pooled"
                ],
                "loso_shift_along_weak_mode": entry["loso_shift_along_weak_mode"],
            }
            for name in analysis["parameter_names"]:
                row[f"loso_shift_{name}"] = entry["loso_shift"][name]
            rows.append(row)
    with (output / "source_influence.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"output_dir": str(output), "analyses": len(analyses)}, indent=2))


if __name__ == "__main__":
    main()
