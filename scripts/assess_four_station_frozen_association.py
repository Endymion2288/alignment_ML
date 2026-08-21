#!/usr/bin/env python3
"""Compare frozen V2 association diagnostics against the pre-registered gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from scripts.run_refit_multidof_closure import _json_ready, _read_json
from scripts.summarize_four_station_frozen_association import assess


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operating-point", default="configs/physical_refit_four_station_unknown_association.yaml")
    parser.add_argument("--diagnostic-json", action="append", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()
    operating = yaml.safe_load(Path(args.operating_point).expanduser().resolve().read_text(encoding="utf-8"))
    if operating.get("alignment_formulation") != "four_station_v1":
        raise SystemExit("operating point is not four_station_v1")
    reports = {}
    for raw in args.diagnostic_json:
        report = _read_json(Path(raw).expanduser().resolve())
        reports[str(report["payload_id"])] = report
    expected = list(operating["payloads"])
    missing = [name for name in expected if name not in reports]
    if missing:
        raise SystemExit("missing association diagnostics for: " + ", ".join(missing))
    decision = assess({name: reports[name] for name in expected}, operating["association_gates"])
    output = Path(args.output_json).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_json_ready(decision), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(_json_ready({
        "output_json": str(output),
        "continue_to_15d_relative_wls": decision["continue_to_15d_relative_wls"],
        "failure_class": decision["failure_class"],
        "retrain_transformer": decision["retrain_transformer"],
    }), indent=2))


if __name__ == "__main__":
    main()
