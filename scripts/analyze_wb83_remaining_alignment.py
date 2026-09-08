#!/usr/bin/env python3
"""Diagnose the two remaining WB83-v2 alignment failures.

Does not change frozen gates, does not add official replicas, and does not
re-interpret v1.  Writes a diagnostic JSON beside the v2 artifacts.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from alignment.wb83_qualification import refuse_wb83_path, write_json
from alignment.wb83_remaining_mechanism import diagnose_replica, summarize_diagnostics

DEFAULT_ROOT = PROJECT_ROOT / "outputs/mc24_four_station_wb83_truth_only_replicas_v2"
FAILING = ("C3_weak_0.5x_fixed_dz", "C3_weak_1x_finite_survey_prior")


def _load_matrix(root: Path) -> dict:
    path = root / "geometry_qualification_matrix.json"
    refuse_wb83_path(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _cell(matrix: dict, cell_id: str) -> dict:
    for item in matrix["cells"]:
        if item["cell_id"] == cell_id:
            return item
    raise SystemExit(f"missing cell {cell_id}")


def _run_cell(
    cell: dict,
    n: int,
    *,
    inject_amplitude: float | None = None,
    iterate: bool = True,
) -> list[dict]:
    rows = []
    for replica_id in range(n):
        rows.append(
            diagnose_replica(
                cell,
                replica_id,
                inject_amplitude=inject_amplitude,
                iterate=iterate,
            )
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(DEFAULT_ROOT))
    parser.add_argument("--n-replicas", type=int, default=100)
    args = parser.parse_args()
    root = Path(args.output_root)
    refuse_wb83_path(root)
    matrix = _load_matrix(root)
    n = int(args.n_replicas)
    report: dict[str, object] = {
        "workbook": "83",
        "label": "remaining_alignment_mechanism_v2",
        "diagnostic_only": True,
        "not_a_qualification_gate": True,
        "does_not_change_frozen_thresholds": True,
        "matrix_sha256": matrix["matrix_sha256"],
        "n_replicas": n,
        "official_failing_cells": list(FAILING),
        "cells": {},
    }
    half = _cell(matrix, "C3_weak_0.5x_fixed_dz")
    finite = _cell(matrix, "C3_weak_1x_finite_survey_prior")
    finite_half = _cell(matrix, "C3_weak_0.5x_finite_survey_prior")
    report["cells"]["C3_weak_0.5x_fixed_dz"] = {
        "role": "official_0.5x_fixed_first_vs_iterate",
        "summary": summarize_diagnostics(_run_cell(half, n, iterate=True)),
    }
    report["cells"]["C3_weak_0.5x_fixed_dz_amp0_same_seeds"] = {
        "role": "same_0.5x_seeds_zero_injection_first_step",
        "summary": summarize_diagnostics(_run_cell(half, n, inject_amplitude=0.0, iterate=False)),
    }
    report["cells"]["C3_weak_0.5x_fixed_dz_amp1_same_seeds"] = {
        "role": "same_0.5x_seeds_1x_injection_first_step",
        "summary": summarize_diagnostics(_run_cell(half, n, inject_amplitude=1.0, iterate=False)),
    }
    report["cells"]["C3_weak_1x_finite_survey_prior"] = {
        "role": "official_1x_finite_last_step_sigma",
        "summary": summarize_diagnostics(_run_cell(finite, n, iterate=True)),
    }
    report["cells"]["C3_weak_0.5x_finite_survey_prior"] = {
        "role": "compare_0.5x_finite_last_step_sigma",
        "summary": summarize_diagnostics(_run_cell(finite_half, n, iterate=True)),
    }
    write_json(root / "remaining_alignment_mechanism.json", report)
    print(json.dumps({key: value["summary"] for key, value in report["cells"].items()}, indent=2))


if __name__ == "__main__":
    main()
