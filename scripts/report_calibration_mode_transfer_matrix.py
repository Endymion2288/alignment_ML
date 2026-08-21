#!/usr/bin/env python3
"""Compare current-corpus vs large-statistics calibration-mode transfer reports.

Does not retune V2, capture, A, or add DoF.  Residual reduction is a DQ
observable only.  A source- or pooled-transfer failure is recorded as a
transfer limitation of that exclusive mode.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping


def _read(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected a JSON object at {path}")
    return dict(payload)


def _finite(value: object) -> float | None:
    if value is None:
        return None
    number = float(value)
    return None if not math.isfinite(number) else number


def _rel_change(current: float | None, transfer: float | None) -> float | None:
    if current is None or transfer is None:
        return None
    if current == 0.0:
        return None if transfer == 0.0 else float("inf")
    return (transfer - current) / abs(current)


def _parameter_map(rows: object) -> dict[str, dict[str, float | None]]:
    result: dict[str, dict[str, float | None]] = {}
    if not isinstance(rows, list):
        return result
    for row in rows:
        if not isinstance(row, Mapping) or not row.get("name"):
            continue
        result[str(row["name"])] = {
            "error": _finite(row.get("error")),
            "fit_sigma": _finite(row.get("fit_sigma")),
            "pull": _finite(row.get("pull")),
        }
    return result


def _score_cell(current: Mapping[str, Any] | None, transfer: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(current, Mapping) or not isinstance(transfer, Mapping):
        return {"present": False}
    current_params = _parameter_map(current.get("parameters"))
    transfer_params = _parameter_map(transfer.get("parameters"))
    names = sorted(set(current_params) | set(transfer_params))
    return {
        "present": True,
        "capture_success": {
            "current": current.get("capture_success"),
            "transfer": transfer.get("capture_success"),
            "unchanged": current.get("capture_success") == transfer.get("capture_success"),
        },
        "geometry_write_allowed": {
            "current": (current.get("mode_validity") or {}).get("geometry_write_allowed")
            if isinstance(current.get("mode_validity"), Mapping)
            else None,
            "transfer": (transfer.get("mode_validity") or {}).get("geometry_write_allowed")
            if isinstance(transfer.get("mode_validity"), Mapping)
            else None,
        },
        "unique_physical_edges": {
            "current": current.get("unique_physical_edges"),
            "transfer": transfer.get("unique_physical_edges"),
        },
        "normal_matrix_condition_number": {
            "current": current.get("normal_matrix_condition_number"),
            "transfer": transfer.get("normal_matrix_condition_number"),
            "relative_change": _rel_change(
                _finite(current.get("normal_matrix_condition_number")),
                _finite(transfer.get("normal_matrix_condition_number")),
            ),
        },
        "post_over_pre_rms": {
            "current": current.get("post_over_pre_rms"),
            "transfer": transfer.get("post_over_pre_rms"),
            "residual_reduction_is_not_alignment_success": True,
        },
        "parameters": {
            name: {
                "current": current_params.get(name),
                "transfer": transfer_params.get(name),
                "pull_change": _rel_change(
                    (current_params.get(name) or {}).get("pull"),
                    (transfer_params.get(name) or {}).get("pull"),
                ),
            }
            for name in names
        },
        "reject_indicators": {
            "current": list((current.get("mode_validity") or {}).get("reject_indicators") or [])
            if isinstance(current.get("mode_validity"), Mapping)
            else [],
            "transfer": list((transfer.get("mode_validity") or {}).get("reject_indicators") or [])
            if isinstance(transfer.get("mode_validity"), Mapping)
            else [],
        },
    }


def _mode_block(report: Mapping[str, Any], mode: str) -> Mapping[str, Any]:
    payload = report.get(mode)
    return payload if isinstance(payload, Mapping) else {}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current-report", required=True)
    parser.add_argument("--transfer-report", required=True)
    parser.add_argument("--a-stability", default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    current = _read(Path(args.current_report).expanduser().resolve())
    transfer = _read(Path(args.transfer_report).expanduser().resolve())
    a_stability = None
    if args.a_stability:
        a_stability = _read(Path(args.a_stability).expanduser().resolve())
    modes = ("station_mode", "ift_internal_mode")
    splits = ("train", "validation")
    matrix: dict[str, Any] = {}
    mode_failures: list[str] = []
    for mode in modes:
        current_mode = _mode_block(current, mode)
        transfer_mode = _mode_block(transfer, mode)
        cells = {}
        for split in splits:
            cell = _score_cell(current_mode.get(split), transfer_mode.get(split))
            cells[split] = cell
            allowed = (cell.get("geometry_write_allowed") or {}).get("transfer")
            capture = (cell.get("capture_success") or {}).get("transfer")
            if allowed is False or capture is False:
                mode_failures.append(f"{mode}/{split}")
        matrix[mode] = cells
    a_stable = None if not isinstance(a_stability, Mapping) else a_stability.get("all_stable")
    if a_stable is False:
        mode_failures.append("A_stability")
    gate_passed = not mode_failures
    report = {
        "schema_version": "faser-ift-calibration-mode-transfer-matrix-v1",
        "current_report": str(Path(args.current_report).expanduser().resolve()),
        "transfer_report": str(Path(args.transfer_report).expanduser().resolve()),
        "a_stability": None if args.a_stability is None else str(Path(args.a_stability).expanduser().resolve()),
        "a_all_stable": a_stable,
        "do_not_retune_V2_capture_or_A": True,
        "do_not_add_layer_or_module_dof": True,
        "residual_reduction_is_not_alignment_success": True,
        "real_data_dry_run_allowed": bool(gate_passed),
        "mc_transfer_gate_passed": bool(gate_passed),
        "failed_cells": mode_failures,
        "if_failed": (
            "localize the failing exclusive mode's transfer limitation; "
            "do not retune V2, capture, A, or add DoF"
        ),
        "if_passed": "Operating Protocol V1 real FASER data dry-run is the next stage",
        "matrix": matrix,
        "test_data_accessed": False,
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "mc_transfer_gate_passed": gate_passed, "failed_cells": mode_failures}, indent=2))


if __name__ == "__main__":
    main()
