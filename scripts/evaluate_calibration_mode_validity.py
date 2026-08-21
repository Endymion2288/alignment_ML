#!/usr/bin/env python3
"""Evaluate one exclusive calibration mode against the frozen validity contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from alignment.calibration_modes import (
    DEFAULT_CONTRACT_RELATIVE,
    evaluate_A_stability,
    evaluate_mode_validity,
    load_mode_validity_contract,
    read_station_framework_capture_success,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", default=None)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--floated-parameters", nargs="+", required=True)
    parser.add_argument("--declared-unmodeled-cdx-um", type=float, default=None)
    parser.add_argument(
        "--cdx-fixed-by",
        choices=("external_geometry", "dedicated_calibration", "isolation_zero"),
        default=None,
    )
    parser.add_argument("--station-capture-artifact", default=None)
    parser.add_argument("--station-framework-capture-success", choices=("true", "false"), default=None)
    parser.add_argument("--same-data-stage-as-other-mode", action="store_true")
    parser.add_argument("--measured-A-dx", type=float, default=None)
    parser.add_argument("--measured-A-ry", type=float, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--require-mode-valid",
        action="store_true",
        help=(
            "Exit non-zero unless geometry precondition, capture, A stability, "
            "and exclusive-mode forbids all allow a geometry-write candidate. "
            "Residual reduction is never sufficient."
        ),
    )
    args = parser.parse_args()
    contract_path = (
        Path(args.contract).expanduser().resolve()
        if args.contract
        else Path(__file__).resolve().parents[1] / DEFAULT_CONTRACT_RELATIVE
    )
    contract = load_mode_validity_contract(contract_path)
    station_capture = None
    if args.station_framework_capture_success is not None:
        station_capture = args.station_framework_capture_success == "true"
    elif args.station_capture_artifact is not None:
        station_capture = read_station_framework_capture_success(Path(args.station_capture_artifact).expanduser().resolve())
    report = {
        "contract": str(contract_path),
        "mode_validity": evaluate_mode_validity(
            contract,
            mode=args.mode,
            floated_parameters=args.floated_parameters,
            unmodeled_abs_C_dx_um=args.declared_unmodeled_cdx_um,
            cdx_fixed_by=args.cdx_fixed_by,
            station_framework_capture_success=station_capture,
            same_data_stage_as_other_mode=bool(args.same_data_stage_as_other_mode),
        ),
        "residual_reduction_is_not_alignment_success": True,
    }
    if args.measured_A_dx is not None:
        report["A_stability"] = evaluate_A_stability(
            contract,
            measured_A_dx=float(args.measured_A_dx),
            measured_A_ry=args.measured_A_ry,
        )
        if not report["A_stability"]["stable"]:
            report["mode_validity"]["reject_indicators"] = list(
                dict.fromkeys([*report["mode_validity"]["reject_indicators"], "A_unstable"])
            )
            report["mode_validity"]["geometry_write_allowed"] = False
            report["mode_validity"]["cross_level_contaminated"] = True
    text = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        Path(args.output).expanduser().resolve().write_text(text, encoding="utf-8")
    print(text, end="")
    if args.require_mode_valid and not bool(report["mode_validity"].get("geometry_write_allowed")):
        raise SystemExit("mode is not a geometry-write candidate; residual reduction is not alignment success")


if __name__ == "__main__":
    main()
