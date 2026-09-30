#!/usr/bin/env python3
"""Record one frozen WB87 trajectory.  Does not write WB86 artifacts."""

from __future__ import annotations

import argparse
import os
import traceback
from pathlib import Path

from alignment.physical_common_track_execution import CalypsoActsPhysicalBackend
from alignment.wb85_physical_qualification_protocol import chart_parameter_names
from alignment.wb86_physical_qualification import WB86Error, load_catalog_event
from alignment.common_track_solver import MAX_ITERATIONS
from alignment.wb87_convergence_autopsy import (
    FIXED_EVENT_INDICES,
    WB86_CATALOG_ROOT,
    WB87_OUTPUT_ROOT,
    record_catalog_event,
    write_trajectory,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--output-root", default=str(WB87_OUTPUT_ROOT))
    parser.add_argument("--catalog-root", default=str(WB86_CATALOG_ROOT))
    parser.add_argument("--work-dir", default="")
    args = parser.parse_args()
    index = int(args.index)
    if index not in FIXED_EVENT_INDICES:
        raise SystemExit(f"index {index} is outside the frozen 13-event list")
    planner = CalypsoActsPhysicalBackend(execute=False)
    if planner.execute is not False:
        raise WB86Error("CalypsoActsPhysicalBackend.execute must stay False")
    output = Path(args.output_root)
    if "wb86_physical_qualification" in str(output):
        raise SystemExit("WB87 must not write into the WB86 output root")
    row = load_catalog_event(index, Path(args.catalog_root))
    names = chart_parameter_names(str(row["chart_kind"]), str(row["survey_mode"]))
    budget = int(MAX_ITERATIONS) * len(names) + 11
    work = Path(args.work_dir) if args.work_dir else Path(os.environ.get("TMPDIR") or "/tmp") / f"wb87_event_{index:04d}_{os.getpid()}"
    target = output / "trajectories" / f"event_{index:04d}.json"
    try:
        result = record_catalog_event(index, work_dir=work, catalog_root=Path(args.catalog_root))
        result["physical_refit_budget"] = budget
        result["within_physical_refit_budget"] = int(result["n_official_refits"]) <= budget
        result["execution_class"] = "trajectory"
        write_trajectory(target, result)
        print(
            f"event {index} trajectory reason={result['reason']} "
            f"refits={result['n_official_refits']} budget={budget}"
        )
    except Exception as error:
        text = str(error)
        reproduced = "missing truth-associated hit" in text
        payload = {
            "schema": "wb87_iteration_trajectory_v1",
            "qualifies_wb86": False,
            "index": index,
            "event_uid": row["event_uid"],
            "condition_id": row["condition_id"],
            "family": row["family"],
            "survey_mode": row["survey_mode"],
            "chart_kind": row["chart_kind"],
            "execution_class": "reproduced_missing_hit" if reproduced else "execution_mismatch",
            "error_type": type(error).__name__,
            "error": text,
            "traceback": traceback.format_exc()[-4000:],
            "history": [],
            "n_official_refits": None,
            "physical_refit_budget": budget,
            "within_physical_refit_budget": None,
        }
        write_trajectory(target, payload)
        print(f"event {index} {payload['execution_class']} {type(error).__name__}: {text}")


if __name__ == "__main__":
    main()
