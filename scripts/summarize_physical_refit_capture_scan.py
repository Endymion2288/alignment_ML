#!/usr/bin/env python3
"""Collect and plot results from a real-conditions physical refit scan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from evaluation.physical_capture_scan import (
    collect_physical_capture_scan,
    write_capture_scan_artifacts,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-root", required=True)
    parser.add_argument("--scan-plan", default=None)
    parser.add_argument("--capture-tolerance-mm", type=float, required=True)
    args = parser.parse_args()
    if args.capture_tolerance_mm < 0.0:
        parser.error("--capture-tolerance-mm must be non-negative")
    root = Path(args.scan_root).expanduser().resolve()
    plan_path = (
        Path(args.scan_plan).expanduser().resolve()
        if args.scan_plan
        else root / "scan_plan.json"
    )
    with plan_path.open(encoding="utf-8") as handle:
        plan = json.load(handle)
    points, diagnostics = collect_physical_capture_scan(
        root, plan, args.capture_tolerance_mm
    )
    summary = write_capture_scan_artifacts(
        root, points, diagnostics, args.capture_tolerance_mm
    )
    (root / "capture_scan_resolved_config.yaml").write_text(
        yaml.safe_dump(
            {
                "scan_root": str(root),
                "scan_plan": str(plan_path),
                "capture_tolerance_mm": args.capture_tolerance_mm,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary["by_magnitude"], indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
