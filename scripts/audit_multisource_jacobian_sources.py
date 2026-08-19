#!/usr/bin/env python3
"""Source-wise Jacobian diagnosis for a completed multi-DoF iteration bank.

Read-only audit over an existing multi-source physical iteration bank.  For
every source it recomputes the local finite-difference Jacobian, normal
matrix, parameter covariance/correlation and recovered update with exactly
the same solver as the pooled closure, then asks three mechanism questions
about the observed dx/dy residual bias:

1. Finite-difference nonlinearity: an axis-wise quadratic model built from
   the existing +/- probes predicts a deterministic bias; a corrected solve
   shows how much of the observed bias it explains (mixed second derivatives
   remain out of reach of the existing probe plan).
2. Phase-space-dependent response: per-source recovered biases and Jacobian
   directions are correlated with per-source track kinematics (x/y/tx/ty).
3. Source weighting bias: leave-one-source-out / leave-one-group-out pooled
   solutions and an equal-per-source reweighting show how much the pooled
   answer moves when the source composition changes.

No new physical probe is produced; the tool only recombines existing refit
artifacts.  Sealed test sources are never touched: the iteration manifest
schema itself only admits train/validation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from alignment.physical_jacobian import solve_physical_finite_difference
from datasets.root_loader import load_events
from scripts.run_multisource_refit_multidof_local_step import (
    RESIDUAL_LABELS,
    _load_bank,
    _read_json,
)


SCHEMA_VERSION = "faser-multisource-jacobian-source-diagnosis-v1"
PRODUCTION_GROUPS = ("100043", "100044", "100047", "100048")


def _production_group(source_id: str) -> str:
    for group in PRODUCTION_GROUPS:
        if f"_{group}_" in source_id:
            return group
    return "other"


def _row_mask(bank: Any, source_id: str) -> np.ndarray:
    return np.asarray([str(value) == source_id for value in bank.source_ids], dtype=bool)


def _fit_rows(bank: Any, mask: np.ndarray, *, rcond: float) -> Any:
    return solve_physical_finite_difference(
        bank.anchor_residual[mask],
        bank.positive_residual[:, mask, :],
        bank.negative_residual[:, mask, :],
        bank.target_residual[mask],
        bank.covariance[mask],
        parameter_names=bank.parameter_names,
        positive_values=bank.positive_values,
        negative_values=bank.negative_values,
        parameter_scales=bank.parameter_scales,
        rcond=rcond,
    )


def _axis_quadratic_coefficients(
    anchor: np.ndarray,
    positive: np.ndarray,
    negative: np.ndarray,
    plus: np.ndarray,
    minus: np.ndarray,
    anchor_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-pair axis-wise central slope c1 and curvature c2 of the response."""
    if not np.allclose(plus - anchor_values, anchor_values - minus, rtol=0.0, atol=1.0e-12):
        raise ValueError("axis quadratic requires probes symmetric around the anchor")
    half = plus - minus  # = 2h
    c1 = (positive - negative) / half[:, None, None]
    c2 = (positive + negative - 2.0 * anchor[None, :, :]) / (2.0 * (0.5 * half) ** 2)[:, None, None]
    return c1, c2


def _quadratic_corrected_solve(
    bank: Any,
    mask: np.ndarray,
    *,
    rcond: float,
) -> dict[str, Any]:
    """One Gauss-Newton correction of the linear solve for axis curvature.

    The per-pair response along parameter p is modelled as
    r(u) = r0 + c1_p u + c2_p u^2 from the existing +/- probes.  The target
    residual is then fitted with sum_p [c1_p d_p + c2_p d_p^2]; linearising
    that model around the purely linear solution gives one correction step.
    """
    anchor = bank.anchor_residual[mask]
    positive = bank.positive_residual[:, mask, :]
    negative = bank.negative_residual[:, mask, :]
    response = bank.target_residual[mask] - anchor
    names = tuple(bank.parameter_names)
    anchor_values = np.asarray(bank.anchor_values, dtype=np.float64)
    target_values = np.asarray(bank.target_values, dtype=np.float64)
    true_delta = target_values - anchor_values
    c1, c2 = _axis_quadratic_coefficients(
        anchor, positive, negative, bank.positive_values, bank.negative_values, anchor_values
    )
    linear = _fit_rows(bank, mask, rcond=rcond)
    delta_lin = np.asarray(linear.recovered_parameters, dtype=np.float64)

    # Predicted per-pair response from the quadratic model at the linear solution.
    predicted = np.einsum("pnr,p->nr", c1, delta_lin) + np.einsum("pnr,p->nr", c2, delta_lin**2)
    residual_correction = response - predicted
    # Linearise the quadratic model at delta_lin: d/d(d_p) = c1_p + 2 c2_p d_p.
    effective = c1 + 2.0 * c2 * delta_lin[:, None, None]
    inverse_covariance = np.linalg.inv(bank.covariance[mask])
    parameters = len(names)
    normal = np.zeros((parameters, parameters), dtype=np.float64)
    rhs = np.zeros(parameters, dtype=np.float64)
    for row in range(anchor.shape[0]):
        jacobian = effective[:, row, :].T
        normal += jacobian.T @ inverse_covariance[row] @ jacobian
        rhs += jacobian.T @ inverse_covariance[row] @ residual_correction[row]
    normal = 0.5 * (normal + normal.T)
    correction = np.linalg.solve(normal, rhs)
    delta_quad = delta_lin + correction

    # Deterministic per-pair bias prediction under a pure axis-quadratic model:
    # delta_rec = sum_p (c1 d + c2 d^2) projected through the linear normal equation.
    predicted_response_at_truth = np.einsum("pnr,p->nr", c1, true_delta) + np.einsum(
        "pnr,p->nr", c2, true_delta**2
    )
    normal_lin = np.zeros((parameters, parameters), dtype=np.float64)
    rhs_truth = np.zeros(parameters, dtype=np.float64)
    jacobian_lin = np.moveaxis(c1, 0, -1)
    for row in range(anchor.shape[0]):
        normal_lin += jacobian_lin[row].T @ inverse_covariance[row] @ jacobian_lin[row]
        rhs_truth += jacobian_lin[row].T @ inverse_covariance[row] @ predicted_response_at_truth[row]
    normal_lin = 0.5 * (normal_lin + normal_lin.T)
    predicted_recovered_at_truth = np.linalg.solve(normal_lin, rhs_truth)

    return {
        "linear_delta": {name: float(delta_lin[i]) for i, name in enumerate(names)},
        "quadratic_corrected_delta": {name: float(delta_quad[i]) for i, name in enumerate(names)},
        "true_delta": {name: float(true_delta[i]) for i, name in enumerate(names)},
        "linear_bias": {name: float(delta_lin[i] - true_delta[i]) for i, name in enumerate(names)},
        "quadratic_corrected_bias": {
            name: float(delta_quad[i] - true_delta[i]) for i, name in enumerate(names)
        },
        "axis_quadratic_predicted_bias": {
            name: float(predicted_recovered_at_truth[i] - true_delta[i])
            for i, name in enumerate(names)
        },
    }


def _source_kinematics(bank: Any, mask: np.ndarray, tracklet_states: Mapping[tuple, tuple]) -> dict[str, float]:
    xs: list[float] = []
    ys: list[float] = []
    txs: list[float] = []
    tys: list[float] = []
    runs = bank.run_ids[mask]
    events = bank.event_ids[mask]
    tracklets = bank.source_tracklet_ids[mask]
    missing = 0
    for run, event, tracklet in zip(runs, events, tracklets):
        state = tracklet_states.get((int(run), int(event), int(tracklet)))
        if state is None:
            missing += 1
            continue
        xs.append(state[0])
        ys.append(state[1])
        txs.append(state[2])
        tys.append(state[3])
    if not xs:
        raise ValueError("no bank row could be joined to the source tracklet states")
    out: dict[str, float] = {"kinematics_join_missing_rows": float(missing)}
    for label, values in (("x_mm", xs), ("y_mm", ys), ("tx", txs), ("ty", tys)):
        array = np.asarray(values, dtype=np.float64)
        out[f"mean_{label}"] = float(array.mean())
        out[f"std_{label}"] = float(array.std())
        out[f"mean_abs_{label}"] = float(np.abs(array).mean())
    return out


def _load_anchor_tracklet_states(manifest: Mapping[str, object], anchor_point: str) -> dict[str, dict[tuple, tuple]]:
    states: dict[str, dict[tuple, tuple]] = {}
    for entry in manifest["sources"]:
        source_id = str(entry["source_id"])
        root = Path(str(entry["physical_scan_root"])).expanduser().resolve()
        plan = _read_json(root / "scan_plan.json")
        point = next(
            (item for item in plan["points"] if str(item.get("name")) == anchor_point),
            None,
        )
        if point is None:
            raise ValueError(f"source '{source_id}' plan lacks anchor point '{anchor_point}'")
        tracklets = root / str(point["relative_point_dir"]) / "refit" / "tracklets.root"
        table: dict[tuple, tuple] = {}
        for event in load_events(tracklets, require_mc_labels=True):
            for row in range(event.station_id.shape[0]):
                key = (int(event.run_id), int(event.event_id), int(event.tracklet_id[row]))
                table[key] = (
                    float(event.state[row, 0]),
                    float(event.state[row, 1]),
                    float(event.state[row, 2]),
                    float(event.state[row, 3]),
                )
        states[source_id] = table
    return states


def _correlation_summary(pairs: Sequence[tuple[float, float]]) -> dict[str, float | int | None]:
    x = np.asarray([pair[0] for pair in pairs], dtype=np.float64)
    y = np.asarray([pair[1] for pair in pairs], dtype=np.float64)
    if x.size < 3 or np.allclose(x.std(), 0.0) or np.allclose(y.std(), 0.0):
        return {"n": int(x.size), "pearson": None, "spearman": None}
    pearson = float(np.corrcoef(x, y)[0, 1])
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    spearman = float(np.corrcoef(rx, ry)[0, 1])
    return {"n": int(x.size), "pearson": pearson, "spearman": spearman}


def _fit_summary(fit: Any, names: Sequence[str]) -> dict[str, Any]:
    return {
        "observations": int(fit.used_pairs),
        "normal_matrix_rank": int(fit.normal_matrix_rank),
        "normal_matrix_condition_number": float(fit.normal_matrix_condition_number),
        "recovered_local_delta": {
            name: float(fit.recovered_parameters[index]) for index, name in enumerate(names)
        },
        "recovered_sigma": {
            name: float(math.sqrt(max(float(fit.covariance_native[index, index]), 0.0)))
            for index, name in enumerate(names)
        },
        "parameter_correlation": {
            f"{names[i]}__{names[j]}": float(fit.correlation_native[i, j])
            for i in range(len(names))
            for j in range(i + 1, len(names))
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iteration-manifest", required=True)
    parser.add_argument("--anchor-point", required=True)
    parser.add_argument("--target-point", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-truth-match-fraction", type=float, default=0.99)
    parser.add_argument("--rcond", type=float, default=1.0e-10)
    args = parser.parse_args()

    manifest_path = Path(args.iteration_manifest).expanduser().resolve()
    manifest = _read_json(manifest_path)
    output_root = Path(args.output_dir).expanduser().resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError("refusing to overwrite a non-empty diagnosis output directory")
    output_root.mkdir(parents=True, exist_ok=False)

    banks = {
        split: _load_bank(
            manifest,
            split=split,
            anchor_point=str(args.anchor_point),
            target_point=str(args.target_point),
            min_truth_match_fraction=float(args.min_truth_match_fraction),
        )
        for split in ("train", "validation")
    }
    names = tuple(banks["train"].parameter_names)
    anchor_states = _load_anchor_tracklet_states(manifest, str(args.anchor_point))

    per_source_rows: list[dict[str, Any]] = []
    fits_by_source: dict[str, Any] = {}
    kinematics_by_source: dict[str, dict[str, float]] = {}
    quad_by_source: dict[str, Any] = {}
    for entry in manifest["sources"]:
        source_id = str(entry["source_id"])
        split = str(entry["split"])
        bank = banks[split]
        mask = _row_mask(bank, source_id)
        fit = _fit_rows(bank, mask, rcond=float(args.rcond))
        fits_by_source[source_id] = fit
        kinematics = _source_kinematics(bank, mask, anchor_states[source_id])
        kinematics_by_source[source_id] = kinematics
        quad_by_source[source_id] = _quadratic_corrected_solve(bank, mask, rcond=float(args.rcond))
        summary = _fit_summary(fit, names)
        per_source_rows.append(
            {
                "source_id": source_id,
                "split": split,
                "production_group": _production_group(source_id),
                **summary,
                **kinematics,
            }
        )

    pooled_fits = {
        split: _fit_rows(
            banks[split],
            np.ones(banks[split].observations, dtype=bool),
            rcond=float(args.rcond),
        )
        for split in ("train", "validation")
    }

    # Leave-one-source-out and leave-one-group-out on the fit (train) split.
    fit_bank = banks["train"]
    loso_rows: list[dict[str, Any]] = []
    train_sources = [str(e["source_id"]) for e in manifest["sources"] if str(e["split"]) == "train"]
    for excluded in train_sources:
        mask = np.asarray([str(value) != excluded for value in fit_bank.source_ids], dtype=bool)
        fit = _fit_rows(fit_bank, mask, rcond=float(args.rcond))
        loso_rows.append(
            {
                "excluded_source": excluded,
                "excluded_group": _production_group(excluded),
                "recovered_local_delta": {
                    name: float(fit.recovered_parameters[i]) for i, name in enumerate(names)
                },
            }
        )
    logo_rows: list[dict[str, Any]] = []
    for group in sorted({_production_group(source) for source in train_sources}):
        mask = np.asarray(
            [_production_group(str(value)) != group for value in fit_bank.source_ids], dtype=bool
        )
        fit = _fit_rows(fit_bank, mask, rcond=float(args.rcond))
        logo_rows.append(
            {
                "excluded_group": group,
                "recovered_local_delta": {
                    name: float(fit.recovered_parameters[i]) for i, name in enumerate(names)
                },
            }
        )

    # Equal-per-source reweighting: scale each source's normal equation so that
    # every source contributes the same total weight, then re-solve.
    reweighted_normal = np.zeros((len(names), len(names)), dtype=np.float64)
    reweighted_rhs = np.zeros(len(names), dtype=np.float64)
    for source_id in train_sources:
        mask = _row_mask(fit_bank, source_id)
        anchor = fit_bank.anchor_residual[mask]
        inverse_covariance = np.linalg.inv(fit_bank.covariance[mask])
        derivative = np.moveaxis(
            (fit_bank.positive_residual[:, mask, :] - fit_bank.negative_residual[:, mask, :])
            / (fit_bank.positive_values - fit_bank.negative_values)[:, None, None],
            0,
            -1,
        )
        response = fit_bank.target_residual[mask] - anchor
        normal = np.zeros((len(names), len(names)), dtype=np.float64)
        rhs = np.zeros(len(names), dtype=np.float64)
        for row in range(anchor.shape[0]):
            normal += derivative[row].T @ inverse_covariance[row] @ derivative[row]
            rhs += derivative[row].T @ inverse_covariance[row] @ response[row]
        weight = float(np.trace(normal))
        if weight <= 0.0:
            raise ValueError(f"source '{source_id}' contributes no normal-equation weight")
        reweighted_normal += normal / weight
        reweighted_rhs += rhs / weight
    reweighted_normal = 0.5 * (reweighted_normal + reweighted_normal.T)
    reweighted_solution = np.linalg.solve(reweighted_normal, reweighted_rhs)

    # Cross-source mechanism correlations.
    dx_ry_key = "ift_dx_mm__ift_ry_mrad"
    kinematic_keys = (
        "mean_x_mm",
        "mean_y_mm",
        "mean_tx",
        "mean_ty",
        "mean_abs_x_mm",
        "mean_abs_y_mm",
        "mean_abs_tx",
        "mean_abs_ty",
        "std_tx",
        "std_ty",
    )
    true_delta_by_split = {
        split: np.asarray(banks[split].target_values, dtype=np.float64)
        - np.asarray(banks[split].anchor_values, dtype=np.float64)
        for split in banks
    }
    bias_by_source = {
        row["source_id"]: {
            name: row["recovered_local_delta"][name] - float(true_delta_by_split[row["split"]][index])
            for index, name in enumerate(names)
        }
        for row in per_source_rows
    }
    correlation_rows: list[dict[str, Any]] = []
    for observable in ("bias_ift_dx_mm", "bias_ift_dy_mm", "bias_ift_ry_mrad", "corr_dx_ry"):
        for key in kinematic_keys:
            pairs = []
            for row in per_source_rows:
                if observable == "corr_dx_ry":
                    value = row["parameter_correlation"][dx_ry_key]
                else:
                    value = bias_by_source[row["source_id"]][observable[len("bias_"):]]
                pairs.append((float(row[key]), float(value)))
            summary = _correlation_summary(pairs)
            correlation_rows.append({"observable": observable, "kinematic": key, **summary})

    # Consistency of the per-source scatter with pure statistics.
    scatter: dict[str, Any] = {}
    for index, name in enumerate(names):
        values = np.asarray(
            [fits_by_source[row["source_id"]].recovered_parameters[index] for row in per_source_rows],
            dtype=np.float64,
        )
        sigmas = np.asarray(
            [
                math.sqrt(
                    max(float(fits_by_source[row["source_id"]].covariance_native[index, index]), 0.0)
                )
                for row in per_source_rows
            ],
            dtype=np.float64,
        )
        weighted_mean = float(np.sum(values / sigmas**2) / np.sum(1.0 / sigmas**2))
        chi2 = float(np.sum(((values - weighted_mean) / sigmas) ** 2))
        scatter[name] = {
            "n_sources": int(values.size),
            "weighted_mean": weighted_mean,
            "std_across_sources": float(values.std()),
            "mean_statistical_sigma": float(sigmas.mean()),
            "scatter_chi2": chi2,
            "scatter_chi2_ndof": chi2 / float(values.size - 1),
        }

    report = {
        "schema_version": SCHEMA_VERSION,
        "iteration_manifest": str(manifest_path),
        "anchor_point": str(args.anchor_point),
        "target_point": str(args.target_point),
        "parameter_names": list(names),
        "residual_labels": list(RESIDUAL_LABELS),
        "pooled_fit": {split: _fit_summary(fit, names) for split, fit in pooled_fits.items()},
        "per_source": per_source_rows,
        "quadratic_axis_diagnosis": quad_by_source,
        "leave_one_source_out": loso_rows,
        "leave_one_group_out": logo_rows,
        "equal_source_weight_solution": {
            name: float(reweighted_solution[i]) for i, name in enumerate(names)
        },
        "cross_source_correlations": correlation_rows,
        "per_source_scatter_consistency": scatter,
    }
    (output_root / "source_jacobian_diagnosis.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    with (output_root / "per_source_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["source_id", "split", "production_group", "observations"]
        for name in names:
            fieldnames += [f"delta_{name}", f"sigma_{name}", f"bias_{name}"]
        fieldnames += ["corr_dx_ry", "normal_matrix_condition_number"] + list(kinematic_keys)
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in per_source_rows:
            record: dict[str, Any] = {
                "source_id": row["source_id"],
                "split": row["split"],
                "production_group": row["production_group"],
                "observations": row["observations"],
                "corr_dx_ry": row["parameter_correlation"][dx_ry_key],
                "normal_matrix_condition_number": row["normal_matrix_condition_number"],
            }
            for name in names:
                record[f"delta_{name}"] = row["recovered_local_delta"][name]
                record[f"sigma_{name}"] = row["recovered_sigma"][name]
                record[f"bias_{name}"] = bias_by_source[row["source_id"]][name]
            for key in kinematic_keys:
                record[key] = row[key]
            writer.writerow(record)

    with (output_root / "loso_solutions.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["excluded_source", "excluded_group"] + [f"delta_{name}" for name in names]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in loso_rows:
            writer.writerow(
                {
                    "excluded_source": row["excluded_source"],
                    "excluded_group": row["excluded_group"],
                    **{f"delta_{name}": row["recovered_local_delta"][name] for name in names},
                }
            )

    _plot(output_root, per_source_rows, loso_rows, names, bias_by_source, dx_ry_key)
    print(
        json.dumps(
            {
                "output_dir": str(output_root),
                "sources": len(per_source_rows),
                "scatter_chi2_ndof": {
                    name: scatter[name]["scatter_chi2_ndof"] for name in names
                },
            },
            indent=2,
        )
    )


def _plot(
    output_root: Path,
    per_source_rows: Sequence[Mapping[str, Any]],
    loso_rows: Sequence[Mapping[str, Any]],
    names: Sequence[str],
    bias_by_source: Mapping[str, Mapping[str, float]],
    dx_ry_key: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [row["source_id"].replace("mc24_", "") for row in per_source_rows]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, len(names), figsize=(6 * len(names), 5), sharex=True)
    for axis, name in zip(axes, names):
        deltas = [row["recovered_local_delta"][name] for row in per_source_rows]
        sigmas = [row["recovered_sigma"][name] for row in per_source_rows]
        axis.errorbar(x, deltas, yerr=sigmas, fmt="o", capsize=3)
        axis.axhline(0.0, color="gray", linewidth=0.8)
        axis.set_title(f"per-source recovered delta: {name}")
        axis.set_xticks(x)
        axis.set_xticklabels(labels, rotation=90, fontsize=6)
    fig.tight_layout()
    fig.savefig(output_root / "per_source_recovered_delta.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    mean_abs_tx = [row["mean_abs_tx"] for row in per_source_rows]
    mean_abs_ty = [row["mean_abs_ty"] for row in per_source_rows]
    corr = [row["parameter_correlation"][dx_ry_key] for row in per_source_rows]
    dx_bias = [bias_by_source[row["source_id"]]["ift_dx_mm"] for row in per_source_rows]
    axes[0].scatter(mean_abs_tx, corr)
    axes[0].set_xlabel("source mean |tx|")
    axes[0].set_ylabel("dx-ry parameter correlation")
    axes[1].scatter(mean_abs_ty, dx_bias)
    axes[1].set_xlabel("source mean |ty|")
    axes[1].set_ylabel("dx bias [mm]")
    for axis in axes:
        axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_root / "dxry_correlation_vs_kinematics.png", dpi=150)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7, 4.5))
    excluded = [row["excluded_source"].replace("mc24_", "") for row in loso_rows]
    xx = np.arange(len(excluded))
    for name in names:
        axis.plot(
            xx,
            [row["recovered_local_delta"][name] for row in loso_rows],
            "o-",
            label=name,
            markersize=4,
        )
    axis.set_xticks(xx)
    axis.set_xticklabels(excluded, rotation=90, fontsize=6)
    axis.set_title("leave-one-source-out recovered deltas (train)")
    axis.legend()
    axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_root / "loso_spread.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
