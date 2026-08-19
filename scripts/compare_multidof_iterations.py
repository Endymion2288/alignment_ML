#!/usr/bin/env python3
"""Compare completed multi-DoF alignment iterations and freeze the next decision.

Reads the ``multisource_local_step.json`` + ``source_fit_diagnostics.csv`` of
two or more completed iteration closures and produces one JSON, one CSV, trend
plots and a Chinese workbook fragment.  The decision rule is frozen *before*
iteration-1 results exist:

- if the last iteration's held-out validation recovers every parameter within
  its frozen tolerance (dx <= 0.1 mm, dy <= 0.1 mm, Ry <= 1 mrad), the next
  step is the route-selected update with the frozen association backbone;
- otherwise NO iteration-2 is submitted.  The follow-up is chosen from the
  source-wise Jacobian diagnosis: FD step-size study, source reweighting, or
  a nonlinear local solver.

The tool never opens sealed test assets; it only reads closure outputs that
were themselves produced under the train/validation boundary.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "faser-multidof-iteration-comparison-v1"
FROZEN_TOLERANCES = {"ift_dx_mm": 0.1, "ift_dy_mm": 0.1, "ift_ry_mrad": 1.0}
DECISION_ROUTE_UPDATE = "proceed_to_route_selected_update"
DECISION_NO_ITERATION2 = "hold_iterations_diagnose_mechanism_first"


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_iteration(closure_dir: Path, label: str) -> dict[str, Any]:
    step_path = closure_dir / "multisource_local_step.json"
    diagnostics_path = closure_dir / "source_fit_diagnostics.csv"
    if not step_path.is_file():
        raise FileNotFoundError(f"{closure_dir} lacks multisource_local_step.json")
    step = _read_json(step_path)
    names = [str(p["name"]) for p in step["parameters"]]
    anchor = {str(p["name"]): float(p["anchor_value"]) for p in step["parameters"]}
    held_out_error = {
        str(p["name"]): float(p["held_out_independent_local_delta_error"])
        for p in step["parameters"]
    }
    fit_error = {
        str(p["name"]): float(p["fit_split_local_delta_error"]) for p in step["parameters"]
    }
    tolerances = {
        str(p["name"]): (
            float(p["capture_tolerance"])
            if p.get("capture_tolerance") is not None
            else FROZEN_TOLERANCES[str(p["name"])]
        )
        for p in step["parameters"]
    }
    held_out_pass = {
        name: abs(held_out_error[name]) <= tolerances[name] for name in names
    }
    correlation = step.get("parameter_correlation")
    dx_ry = None
    if correlation is not None and "ift_dx_mm" in names and "ift_ry_mrad" in names:
        dx_ry = float(correlation[names.index("ift_dx_mm")][names.index("ift_ry_mrad")])
    application = step["held_out_split"]["frozen_fit_split_update_application"]
    candidate_metrics = step.get("candidate_metrics", [])
    anchor_metrics = next(
        (m for m in candidate_metrics if str(m.get("point_role")) == "anchor"),
        candidate_metrics[0] if candidate_metrics else {},
    )
    reference_metrics = next(
        (m for m in candidate_metrics if str(m.get("point_role")) == "reference"),
        {},
    )
    source_spread: dict[str, float | None] = {}
    source_count = 0
    if diagnostics_path.is_file():
        train_rows = []
        with diagnostics_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("split") == "train" and row.get("status") == "ok":
                    train_rows.append(row)
        source_count = len(train_rows)
        for name in names:
            values = [
                float(ast.literal_eval(row["recovered_local_delta"])[name]) for row in train_rows
            ]
            if len(values) >= 2:
                source_spread[name] = float(np_std(values))
            else:
                source_spread[name] = None
    return {
        "label": label,
        "closure_dir": str(closure_dir),
        "parameter_names": names,
        "anchor_values": anchor,
        "fit_split_recovered_local_delta": step["fit_split"]["fit"]["recovered_local_delta"],
        "fit_split_local_delta_error": fit_error,
        "held_out_recovered_local_delta": step["held_out_split"][
            "independent_fit_diagnostic_only"
        ]["recovered_local_delta"],
        "held_out_local_delta_error": held_out_error,
        "capture_tolerances": tolerances,
        "held_out_capture_pass": held_out_pass,
        "held_out_capture_pass_all": all(held_out_pass.values()),
        "held_out_chi2_baseline": float(application["baseline_response_chi2"]),
        "held_out_chi2_post_update": float(application["post_update_response_chi2"]),
        "held_out_chi2_reduction_fraction": float(
            application["response_chi2_reduction_fraction"]
        ),
        "normal_matrix_rank": int(step["fit_split"]["fit"]["normal_matrix_rank"]),
        "normal_matrix_condition_number": float(
            step["fit_split"]["fit"]["normal_matrix_condition_number"]
        ),
        "dx_ry_parameter_correlation": dx_ry,
        "source_to_source_spread": source_spread,
        "train_source_count": source_count,
        "anchor_chain_recall": anchor_metrics.get("candidate_complete_truth_chain_recall"),
        "anchor_edge_recall": {
            pair: anchor_metrics.get(f"{pair}_candidate_truth_edge_recall")
            for pair in ("0->1", "1->2", "2->3")
        },
        "reference_chain_recall": reference_metrics.get(
            "candidate_complete_truth_chain_recall"
        ),
        "proposed_next_parameter_values": step.get("proposed_next_parameter_values"),
    }


def np_std(values: Sequence[float]) -> float:
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def frozen_decision(iterations: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Frozen next-step rule; must not be edited after iteration-1 starts."""
    last = iterations[-1]
    if last["held_out_capture_pass_all"]:
        return {
            "decision": DECISION_ROUTE_UPDATE,
            "rationale": (
                "last iteration passes all frozen held-out tolerances "
                "(dx<=0.1 mm, dy<=0.1 mm, Ry<=1 mrad); proceed to the "
                "route-selected update with the frozen association backbone"
            ),
        }
    failing = [
        name for name, passed in last["held_out_capture_pass"].items() if not passed
    ]
    return {
        "decision": DECISION_NO_ITERATION2,
        "failing_parameters": failing,
        "rationale": (
            "held-out validation still fails frozen tolerances for "
            + ", ".join(failing)
            + "; do NOT submit iteration-2. Choose between FD step-size study, "
            "source reweighting, or a nonlinear local solver based on the "
            "source-wise Jacobian diagnosis"
        ),
    }


def _workbook_fragment(iterations: Sequence[Mapping[str, Any]], decision: Mapping[str, Any]) -> str:
    lines = [
        "## multi-DoF 迭代比较（自动生成）",
        "",
        "| 迭代 | anchor dx/dy/Ry | held-out 偏差 dx/dy/Ry | χ² 下降 | 条件数 | dx-Ry 相关 | 链召回(anchor) | 冻结判定 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for item in iterations:
        anchor = item["anchor_values"]
        errors = item["held_out_local_delta_error"]
        lines.append(
            "| {label} | {adx:.2f}/{ady:.2f}/{ary:.1f} | {edx:.3f}/{edy:.3f}/{ery:.2f} | "
            "{chi2:.1%} | {cond:.0f} | {corr} | {chain} | {verdict} |".format(
                label=item["label"],
                adx=anchor.get("ift_dx_mm", float("nan")),
                ady=anchor.get("ift_dy_mm", float("nan")),
                ary=anchor.get("ift_ry_mrad", float("nan")),
                edx=errors.get("ift_dx_mm", float("nan")),
                edy=errors.get("ift_dy_mm", float("nan")),
                ery=errors.get("ift_ry_mrad", float("nan")),
                chi2=item["held_out_chi2_reduction_fraction"],
                cond=item["normal_matrix_condition_number"],
                corr=(
                    "n/a"
                    if item["dx_ry_parameter_correlation"] is None
                    else f"{item['dx_ry_parameter_correlation']:.2f}"
                ),
                chain=(
                    "n/a"
                    if item["anchor_chain_recall"] is None
                    else f"{item['anchor_chain_recall']:.2f}"
                ),
                verdict="通过" if item["held_out_capture_pass_all"] else "未通过",
            )
        )
    lines += [
        "",
        f"冻结判定结果：`{decision['decision']}` — {decision['rationale']}",
        "",
    ]
    return "\n".join(lines)


def _plot(output_root: Path, iterations: Sequence[Mapping[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [item["label"] for item in iterations]
    x = list(range(len(iterations)))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for name, color in (("ift_dx_mm", "C0"), ("ift_dy_mm", "C1"), ("ift_ry_mrad", "C2")):
        axes[0].plot(
            x,
            [item["held_out_local_delta_error"][name] for item in iterations],
            "o-",
            label=name,
            color=color,
        )
        axes[0].axhline(
            FROZEN_TOLERANCES[name], color=color, linestyle="--", linewidth=0.8
        )
        axes[0].axhline(
            -FROZEN_TOLERANCES[name], color=color, linestyle="--", linewidth=0.8
        )
    axes[0].set_title("held-out residual bias vs frozen tolerance")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].legend(fontsize=8)
    axes[1].plot(
        x, [item["held_out_chi2_reduction_fraction"] for item in iterations], "o-"
    )
    axes[1].set_title("held-out chi2 reduction fraction")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    chains = [item["anchor_chain_recall"] for item in iterations]
    axes[2].plot(x, chains, "o-")
    axes[2].set_title("anchor complete truth-chain recall")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels)
    for axis in axes:
        axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_root / "iteration_trends.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--closure",
        action="append",
        required=True,
        help="closure output directory, one per iteration, in iteration order",
    )
    parser.add_argument(
        "--label",
        action="append",
        default=None,
        help="optional label per --closure (same order)",
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    closures = [Path(value).expanduser().resolve() for value in args.closure]
    labels = list(args.label) if args.label else [f"iteration-{index}" for index in range(len(closures))]
    if len(labels) != len(closures):
        raise ValueError("--label must be given exactly once per --closure")
    output_root = Path(args.output_dir).expanduser().resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError("refusing to overwrite a non-empty comparison output directory")
    output_root.mkdir(parents=True, exist_ok=False)

    iterations = [_load_iteration(path, label) for path, label in zip(closures, labels)]
    decision = frozen_decision(iterations)
    report = {
        "schema_version": SCHEMA_VERSION,
        "frozen_tolerances": FROZEN_TOLERANCES,
        "iterations": iterations,
        "decision": decision,
    }
    (output_root / "iteration_comparison.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    names = iterations[0]["parameter_names"]
    with (output_root / "iteration_comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["label"]
        for name in names:
            fieldnames += [f"anchor_{name}", f"held_out_error_{name}", f"spread_{name}"]
        fieldnames += [
            "held_out_chi2_reduction_fraction",
            "normal_matrix_condition_number",
            "dx_ry_parameter_correlation",
            "anchor_chain_recall",
            "held_out_capture_pass_all",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in iterations:
            row: dict[str, Any] = {"label": item["label"]}
            for name in names:
                row[f"anchor_{name}"] = item["anchor_values"][name]
                row[f"held_out_error_{name}"] = item["held_out_local_delta_error"][name]
                row[f"spread_{name}"] = item["source_to_source_spread"].get(name)
            row["held_out_chi2_reduction_fraction"] = item["held_out_chi2_reduction_fraction"]
            row["normal_matrix_condition_number"] = item["normal_matrix_condition_number"]
            row["dx_ry_parameter_correlation"] = item["dx_ry_parameter_correlation"]
            row["anchor_chain_recall"] = item["anchor_chain_recall"]
            row["held_out_capture_pass_all"] = item["held_out_capture_pass_all"]
            writer.writerow(row)
    (output_root / "workbook_fragment_cn.md").write_text(
        _workbook_fragment(iterations, decision), encoding="utf-8"
    )
    _plot(output_root, iterations)
    print(
        json.dumps(
            {"output_dir": str(output_root), "decision": decision["decision"]}, indent=2
        )
    )


if __name__ == "__main__":
    main()
