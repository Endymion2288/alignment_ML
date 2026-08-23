#!/usr/bin/env python3
"""Write the Operating Protocol V1 paper-ready validation package.

Reads the frozen entry-54 JSON/CSV only.  Does not reopen any alignment
mode, retune V2, or search a subset that would allow geometry write.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

from alignment.operating_protocol_v1_final_closure import (
    RESIDUAL_DECREASE_LABEL,
    project_root,
    walk_forbidden,
)
from alignment.operating_protocol_v1_paper_ready import (
    DECISION_FLOW_TEXT,
    SCHEMA_VERSION,
    build_main_claims,
    figure_specs,
    load_frozen_package,
    load_paper_config,
    render_all_documents,
)


PROJECT_ROOT = project_root()
PASS = "#0072B2"
REJECT = "#D55E00"
MONITOR = "#009E73"
GRAY = "#4D4D4D"
ORANGE = "#E69F00"
INK = "#222222"
PALETTE = (PASS, REJECT, MONITOR, ORANGE, "#CC79A7", "#56B4E9", GRAY)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "legend.fontsize": 8,
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": ":",
            "axes.unicode_minus": False,
        }
    )


def _save(fig: plt.Figure, directory: Path, stem: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(directory / f"{stem}.png", bbox_inches="tight")
    plt.close(fig)


def _f(row: dict, key: str) -> float:
    return float(row[key])


def plot_decision_flow(directory: Path) -> None:
    nodes = [
        ("MC transfer\nPASS", PASS),
        ("real-data\nassociation PASS", PASS),
        ("Station\ncalibration REJECT", REJECT),
        ("reduced\ncalibration REJECT", REJECT),
        ("residual/DQ\nmonitoring PASS", MONITOR),
    ]
    fig, axis = plt.subplots(figsize=(11.2, 2.6))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    width, height = 0.15, 0.46
    xs = np.linspace(0.08, 0.77, len(nodes))
    for x, (label, color) in zip(xs, nodes):
        box = FancyBboxPatch(
            (x, 0.27),
            width,
            height,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            facecolor=color,
            edgecolor=INK,
            linewidth=0.8,
            alpha=0.18,
        )
        axis.add_patch(box)
        axis.text(x + width / 2, 0.50, label, ha="center", va="center", fontsize=8.5, color=INK)
    for left, right in zip(xs[:-1], xs[1:]):
        axis.annotate(
            "",
            xy=(right - 0.004, 0.50),
            xytext=(left + width + 0.004, 0.50),
            arrowprops={"arrowstyle": "->", "color": INK, "lw": 1.1},
        )
    axis.set_title(DECISION_FLOW_TEXT, fontsize=9, pad=8)
    fig.text(
        0.5,
        0.06,
        "Residual/χ² decrease is a DQ observable only.  Implied C_dx is not a measurement.",
        ha="center",
        fontsize=8,
        color=GRAY,
    )
    _save(fig, directory, "fig00_decision_flow")


def plot_scaling(frozen: dict, directory: Path) -> None:
    fig, axis = plt.subplots(figsize=(7.2, 4.4))
    runs = sorted({int(row["run"]) for row in frozen["scaling_rows"]})
    for index, run in enumerate(runs):
        rows = [row for row in frozen["scaling_rows"] if int(row["run"]) == run]
        rows = sorted(rows, key=lambda item: _f(item, "n_events"))
        marker = "D" if run == 14977 else "o"
        axis.plot(
            [_f(row, "n_events") for row in rows],
            [_f(row, "selected_routes") for row in rows],
            marker=marker,
            linewidth=1.4,
            markersize=5 if run != 14977 else 6,
            color=PALETTE[index % len(PALETTE)],
            label=str(run),
        )
    axis.set_xscale("log")
    axis.set_xlabel("Reconstructed events in the analysis window")
    axis.set_ylabel("Frozen-V2 selected routes")
    axis.axhline(10, color=GRAY, linestyle="--", linewidth=0.9, label="alignment-DQ minimum (10)")
    axis.legend(title="run", ncol=4, frameon=False, loc="upper left")
    axis.set_title("Statistics-limited at 100 events; recovered at full segment")
    fig.text(
        0.5,
        -0.02,
        "Association transfer only.  Not a geometry update.  Residual decrease is a DQ observable.",
        ha="center",
        fontsize=8,
        color=GRAY,
    )
    _save(fig, directory, "fig01_route_acceptance_scaling")


def plot_composition(frozen: dict, directory: Path) -> None:
    rows = sorted(frozen["composition_rows"], key=lambda item: int(item["run"]))
    runs = [str(row["run"]) for row in rows]
    two = np.array([_f(row, "n_2_station") for row in rows])
    three = np.array([_f(row, "n_3_station") for row in rows])
    four = np.array([_f(row, "n_4_station") for row in rows])
    fig, axis = plt.subplots(figsize=(8.0, 4.4))
    x = np.arange(len(runs))
    axis.bar(x, two, color=PASS, label="2-station", width=0.72)
    axis.bar(x, three, bottom=two, color=ORANGE, label="3-station", width=0.72)
    axis.bar(x, four, bottom=two + three, color=MONITOR, label="4-station", width=0.72)
    for index, row in enumerate(rows):
        if row["status"] == "insufficient_statistics_for_alignment_dq":
            axis.text(index, two[index] + three[index] + four[index] + 6, "insuff.\nstats", ha="center", fontsize=7, color=GRAY)
    axis.set_xticks(x, runs, rotation=40)
    axis.set_ylabel("Selected routes")
    axis.set_xlabel("Run")
    axis.legend(frameon=False, loc="upper right")
    axis.set_title("Selected-route composition (DQ observable)")
    _save(fig, directory, "fig02_route_composition")


def plot_robust_z(frozen: dict, directory: Path) -> None:
    rows = sorted(frozen["timeseries_rows"], key=lambda item: (_f(item, "lhc_fill"), int(item["run"])))
    fills = [_f(row, "lhc_fill") for row in rows]
    dy = [_f(row, "dy_robust_z") for row in rows]
    rx = [_f(row, "rx_robust_z") for row in rows]
    labels = [str(row["run"]) for row in rows]
    insufficient = [row["status"] == "insufficient_statistics_for_alignment_dq" for row in rows]
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 5.6), sharex=True)
    for axis, values, name in ((axes[0], dy, r"$dy$ robust-$z$"), (axes[1], rx, r"$rx$ robust-$z$")):
        sufficient_x = [fill for fill, flag in zip(fills, insufficient) if not flag]
        sufficient_y = [value for value, flag in zip(values, insufficient) if not flag]
        axis.plot(sufficient_x, sufficient_y, color=PASS, marker="o", linewidth=1.4, markersize=5)
        for fill, value, flag, label in zip(fills, values, insufficient, labels):
            if flag:
                axis.scatter([fill], [value], facecolors="none", edgecolors=GRAY, s=36, zorder=3)
                axis.annotate(label, (fill, value), textcoords="offset points", xytext=(4, 6), fontsize=7, color=GRAY)
            else:
                axis.annotate(label, (fill, value), textcoords="offset points", xytext=(4, 5), fontsize=6.5, color=INK)
        axis.axhline(0.0, color=INK, linewidth=0.6)
        axis.axhline(3.0, color=ORANGE, linestyle="--", linewidth=0.9)
        axis.axhline(-3.0, color=ORANGE, linestyle="--", linewidth=0.9)
        axis.axhline(5.0, color=REJECT, linestyle=":", linewidth=0.9)
        axis.axhline(-5.0, color=REJECT, linestyle=":", linewidth=0.9)
        axis.set_ylabel(name)
        axis.set_ylim(-5.8, 5.8)
    axes[1].set_xlabel("LHC fill")
    axes[0].set_title("Isolation residuals vs frozen 14973/14974 scale (DQ observable)")
    fig.text(0.5, 0.01, "Dashed |z|=3 drift tag; dotted |z|=5 detector-condition alarm.  Hollow: 14977 insufficient statistics.", ha="center", fontsize=8, color=GRAY)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    _save(fig, directory, "fig03_dy_rx_robust_z_timeseries")


def plot_spectrum(frozen: dict, directory: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 4.2), sharey=False)
    titles = {
        "six_dof": "Six-DoF (includes dz)",
        "five_dof_excluding_dz": "Five-DoF (survey dz removed)",
    }
    for axis, dof in zip(axes, ("six_dof", "five_dof_excluding_dz")):
        for color, run in ((PASS, 14973), (REJECT, 14974)):
            rows = [
                row
                for row in frozen["spectrum_rows"]
                if int(row["run"]) == run and row["dof"] == dof
            ]
            rows = sorted(rows, key=lambda item: int(item["index"]))
            axis.semilogy(
                [int(row["index"]) for row in rows],
                [_f(row, "singular_value") for row in rows],
                marker="o",
                color=color,
                linewidth=1.5,
                label=str(run),
            )
        axis.set_title(titles[dof])
        axis.set_xlabel("Singular-value index")
        axis.set_xticks([1, 2, 3, 4, 5, 6] if dof == "six_dof" else [1, 2, 3, 4, 5])
    axes[0].set_ylabel("Singular value (identifiability diagnostic)")
    axes[0].legend(frameon=False)
    fig.suptitle("Station Jacobian spectrum: dz is gauge-like; 14974 remains weak along dx", fontsize=10)
    fig.text(0.5, -0.02, "Not a closure test.  Residual decrease is a DQ observable.  Implied C_dx is not shown as a measurement.", ha="center", fontsize=8, color=GRAY)
    fig.tight_layout()
    _save(fig, directory, "fig04_station_jacobian_singular_spectrum")


def plot_reduced(frozen: dict, directory: Path) -> None:
    reduced = frozen["chart_data"]["reduced_mode_cross_run_inconsistency"]
    names = ["ift_dy_mm", "ift_rx_mrad", "ift_rz_mrad"]
    labels = [r"$dy$ [mm]", r"$rx$ [mrad]", r"$rz$ [mrad]"]
    by_run = {int(row["run"]): row["delta"] for row in reduced["per_run"]}
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 4.0))
    x = np.array([0, 1])
    for axis, name, label in zip(axes, names, labels):
        values = [float(by_run[14973][name]), float(by_run[14974][name])]
        axis.bar(x, values, color=[PASS, REJECT], width=0.62)
        axis.axhline(0.0, color=INK, linewidth=0.6)
        axis.set_xticks(x, ["14973", "14974"])
        axis.set_ylabel(label)
        axis.set_title(f"{label.split()[0]}  {reduced['nsigma'][name]:.0f}σ", fontsize=10)
    fig.suptitle("Reduced {dy,rx,rz} self-nulling updates do not transfer", fontsize=10)
    fig.text(
        0.5,
        -0.03,
        "Fisher-identifiable, not writable.  14974→14973 linearized χ² ×16 is a DQ observable, not closure.",
        ha="center",
        fontsize=8,
        color=GRAY,
    )
    fig.tight_layout()
    _save(fig, directory, "fig05_reduced_mode_cross_run_inconsistency")


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        raise ValueError("cannot normalize a zero vector")
    return vector / norm


def plot_a_versus_weak(frozen: dict, directory: Path) -> None:
    payload = frozen["chart_data"]["frozen_A_versus_station_weak_direction"]
    names_5 = ["ift_dx_mm", "ift_dy_mm", "ift_rx_mrad", "ift_ry_mrad", "ift_rz_mrad"]
    labels_5 = [r"$dx$", r"$dy$", r"$rx$", r"$ry$", r"$rz$"]
    a5 = _unit(np.array([float(payload["frozen_A_native_per_mm_C_dx"][name]) for name in names_5]))
    weak74 = None
    small6 = {}
    for row in payload["per_run"]:
        if int(row["run"]) == 14974:
            weak74 = _unit(np.array([float(row["five_dof_smallest_direction"][name]) for name in names_5]))
        small6[int(row["run"])] = row["six_dof_smallest_direction"]
    if weak74 is None:
        raise ValueError("missing 14974 five-DoF weak direction")
    if float(np.dot(a5, weak74)) < 0.0:
        weak74 = -weak74
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.2))
    x = np.arange(len(labels_5))
    width = 0.38
    axes[0].bar(x - width / 2, a5, width=width, color=PASS, label="unit frozen A (5-DoF)")
    axes[0].bar(x + width / 2, weak74, width=width, color=REJECT, label="14974 weak direction")
    axes[0].axhline(0.0, color=INK, linewidth=0.6)
    axes[0].set_xticks(x, labels_5)
    axes[0].set_ylabel("Component of unit vector")
    axes[0].set_title("cosine(A, 14974 weak) = 0.994")
    axes[0].legend(frameon=False, loc="lower left")
    names_6 = ["ift_dx_mm", "ift_dy_mm", "ift_dz_mm", "ift_rx_mrad", "ift_ry_mrad", "ift_rz_mrad"]
    labels_6 = [r"$dx$", r"$dy$", r"$dz$", r"$rx$", r"$ry$", r"$rz$"]
    x6 = np.arange(len(labels_6))
    for color, run, offset in ((PASS, 14973, -width / 2), (REJECT, 14974, width / 2)):
        vec = np.array([float(small6[run][name]) for name in names_6])
        if vec[2] < 0.0:
            vec = -vec
        axes[1].bar(x6 + offset, vec, width=width, color=color, label=f"{run} 6-DoF weakest")
    axes[1].axhline(0.0, color=INK, linewidth=0.6)
    axes[1].set_xticks(x6, labels_6)
    axes[1].set_title("Six-DoF weakest direction is gauge-like $dz$")
    axes[1].legend(frameon=False)
    fig.suptitle("Cross-level leakage geometry, not a C_dx measurement", fontsize=10)
    fig.text(0.5, -0.03, "Identifiability diagnostic only.  A is not retuned.", ha="center", fontsize=8, color=GRAY)
    fig.tight_layout()
    _save(fig, directory, "fig06_a_versus_station_weak_direction")


def write_documents(output_root: Path, docs_root: Path, claims: dict) -> None:
    documents = render_all_documents(claims)
    for name, text in documents.items():
        _write_text(output_root / name, text)
        _write_text(docs_root / name, text)
    _write_json(output_root / "main_claims_and_limitations.json", claims)
    _write_json(docs_root / "main_claims_and_limitations.json", claims)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs/operating_protocol_v1_paper_ready_validation_v1.yaml"),
    )
    parser.add_argument(
        "--output-root",
        default=str(PROJECT_ROOT / "outputs/operating_protocol_v1_paper_ready_validation_v1"),
    )
    parser.add_argument(
        "--docs-root",
        default=str(PROJECT_ROOT / "docs/operating_protocol_v1_paper_ready"),
    )
    args = parser.parse_args()

    config = load_paper_config(args.config)
    frozen = load_frozen_package(config)
    claims = build_main_claims(frozen)
    claims["created_utc"] = datetime.now(timezone.utc).isoformat()
    claims["residual_decrease_label"] = RESIDUAL_DECREASE_LABEL
    walk_forbidden(claims, where="main_claims_and_limitations.json")

    output_root = Path(args.output_root).expanduser().resolve()
    docs_root = Path(args.docs_root).expanduser().resolve()
    figure_dir = output_root / "figures"
    docs_figure_dir = docs_root / "figures"
    output_root.mkdir(parents=True, exist_ok=True)
    docs_root.mkdir(parents=True, exist_ok=True)

    _style()
    plot_decision_flow(figure_dir)
    plot_scaling(frozen, figure_dir)
    plot_composition(frozen, figure_dir)
    plot_robust_z(frozen, figure_dir)
    plot_spectrum(frozen, figure_dir)
    plot_reduced(frozen, figure_dir)
    plot_a_versus_weak(frozen, figure_dir)
    docs_figure_dir.mkdir(parents=True, exist_ok=True)
    for spec in figure_specs():
        for suffix in (".pdf", ".png"):
            source = figure_dir / f"{spec['file_stem']}{suffix}"
            shutil.copy2(source, docs_figure_dir / source.name)

    write_documents(output_root, docs_root, claims)
    print(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "output_root": str(output_root),
                "docs_root": str(docs_root),
                "decision_flow": DECISION_FLOW_TEXT,
                "real_data_operating_mode": claims["real_data_operating_mode"],
                "geometry_write_allowed": False,
                "unlock_currently_met": False,
                "residual_decrease_label": RESIDUAL_DECREASE_LABEL,
                "figures": [item["file_stem"] for item in figure_specs()],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
