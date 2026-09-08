#!/usr/bin/env python3
"""Fail-closed WB83 coverage summary and oracle gate.

Does not change pre-registered criteria after seeing numbers.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from alignment.wb83_qualification import (
    EXPERIMENT,
    FD_REL_MAX,
    N_TARGET_REPLICAS,
    qualify,
    refuse_wb83_path,
    summarize_cell,
    write_json,
)


def _load_jsonl(root: Path) -> list[dict]:
    rows = []
    replica_root = root / "replicas"
    if not replica_root.exists():
        return rows
    for path in sorted(replica_root.glob("*/*.jsonl")):
        refuse_wb83_path(path)
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(PROJECT_ROOT / f"outputs/mc24_four_station_{EXPERIMENT}"))
    args = parser.parse_args()
    output = Path(args.output_root)
    refuse_wb83_path(output)
    matrix = json.loads((output / "geometry_qualification_matrix.json").read_text(encoding="utf-8"))
    weak = json.loads((output / "weak_mode_report.json").read_text(encoding="utf-8"))
    rows = _load_jsonl(output)
    combined = output / "per_replica_results.jsonl"
    refuse_wb83_path(combined)
    with combined.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    by_cell: dict[str, list] = {}
    for row in rows:
        by_cell.setdefault(str(row["cell_id"]), []).append(row)
    cell_reports = []
    for cell in matrix["cells"]:
        cell_reports.append(summarize_cell(by_cell.get(str(cell["cell_id"]), []), cell))
    n_independent = 0
    if rows:
        n_independent = min(len(by_cell.get(str(cell["cell_id"]), [])) for cell in matrix["cells"])
    fd = weak.get("fd_stability") or {}
    fd_ok = float(fd.get("rel_h_over_2", 1.0)) < FD_REL_MAX and float(fd.get("rel_2h", 1.0)) < FD_REL_MAX
    forbidden = any(row.get("overlay_used") or row.get("w64_used") or row.get("ml_association_used") for row in rows)
    gate = qualify(cell_reports, fd_ok=fd_ok, n_independent=n_independent, forbidden_accessed=forbidden)
    coverage = {
        "workbook": "83",
        "n_rows": len(rows),
        "n_independent_replicas_per_cell_min": n_independent,
        "n_target_replicas": N_TARGET_REPLICAS,
        "cells": cell_reports,
        "fd_stability": fd,
        "fd_stability_qualified": fd_ok,
        "wb82_smoke_is_not_qualification": True,
    }
    convergence = {
        "workbook": "83",
        "cells": [
            {
                "cell_id": report["cell_id"],
                "n_converged": report.get("n_converged"),
                "n_nonconverged": report.get("n_nonconverged"),
            }
            for report in cell_reports
        ],
        "automatically_pass_at_max_iterations": False,
    }
    write_json(output / "coverage_report.json", coverage)
    write_json(output / "convergence_report.json", convergence)
    write_json(
        output / "truth_oracle_qualification.json",
        {
            **gate,
            "association_default_system": "frozen_W64_raw_energy_plus_exact_solver",
            "continue_v5a_frozen_head": False,
            "continue_route_energy_rewrite": False,
            "continue_hybrid_expansion": False,
            "continue_residual_calibration": False,
            "continue_to_15d_relative_wls": False,
            "final_blind_eval_authorized": False,
            "sealed_test_accessed": False,
            "development_00350_used": False,
            "ml_alignment_eval_authorized": False,
            "matrix_sha256": matrix["matrix_sha256"],
            "v1_official_fail_preserved": True,
            "v1_dag_cluster": 1109787,
            "v2_dag_cluster": 1109806,
            "v2_label": "WB83-v2 solver-fix rerun",
        },
    )
    print(
        json.dumps(
            {
                "solver_implementation_fixed": gate.get("solver_implementation_fixed"),
                "common_track_solver_qualified_under_toy_model": gate.get(
                    "common_track_solver_qualified_under_toy_model"
                ),
                "alignment_oracle_qualified_for_physical_FASER": gate.get(
                    "alignment_oracle_qualified_for_physical_FASER"
                ),
                "qualification": gate["qualification"],
                "alignment_oracle_qualified": gate["alignment_oracle_qualified"],
                "n_independent": n_independent,
                "n_rows": len(rows),
                "reason": gate["reason"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
