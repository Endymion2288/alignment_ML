#!/usr/bin/env python3
"""Compare a measured leakage operator A against the frozen mode-validity contract.

Does not retune A.  If relative deviation exceeds the freeze envelope, the new
sample must not inherit the 1.5–1.7 µm C_dx budget.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from alignment.calibration_modes import DEFAULT_CONTRACT_RELATIVE, evaluate_A_stability, load_mode_validity_contract


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", default=None)
    parser.add_argument(
        "--leakage-table",
        default=str(
            Path(__file__).resolve().parents[1]
            / "outputs/mc24_ift_hierarchical_v1_leakage_identifiability_audit_v1/leakage_operator_table.csv"
        ),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--observation-kind", default="route_selected")
    parser.add_argument("--geometry", default="iteration_00")
    args = parser.parse_args()
    contract_path = (
        Path(args.contract).expanduser().resolve()
        if args.contract
        else Path(__file__).resolve().parents[1] / DEFAULT_CONTRACT_RELATIVE
    )
    contract = load_mode_validity_contract(contract_path)
    rows: list[dict[str, Any]] = []
    with Path(args.leakage_table).expanduser().resolve().open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            if raw.get("observation_kind") != args.observation_kind:
                continue
            if raw.get("geometry") != args.geometry:
                continue
            scored = evaluate_A_stability(
                contract,
                measured_A_dx=float(raw["A6_dx"]),
                measured_A_ry=float(raw["A6_ry"]),
            )
            rows.append(
                {
                    "split": raw.get("split"),
                    "target": raw.get("target"),
                    "geometry": raw.get("geometry"),
                    "observation_kind": raw.get("observation_kind"),
                    **scored,
                }
            )
    if not rows:
        raise ValueError("leakage table has no rows matching the requested observation/geometry")
    all_stable = all(bool(row["stable"]) for row in rows)
    report = {
        "contract": str(contract_path),
        "leakage_table": str(Path(args.leakage_table).expanduser().resolve()),
        "observation_kind": args.observation_kind,
        "geometry": args.geometry,
        "n_corners": len(rows),
        "all_stable": all_stable,
        "do_not_retune": True,
        "corners": rows,
        "test_data_accessed": False,
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "n_corners": len(rows), "all_stable": all_stable}, indent=2))


if __name__ == "__main__":
    main()
