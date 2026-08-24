#!/usr/bin/env python3
"""Run the frozen-V2 module-level residual identifiability PoC.

Read-only on the frozen association artifacts.  Software FD only.  No
Athena payload, no Newton, no V2 change, no official geometry write.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from alignment.module_level_residual_poc import (
    SCHEMA_VERSION,
    analyze_identifiability,
    assert_no_alignment_payload,
    build_mapping,
    build_measurements,
    common_operating_state,
    compare_leakage,
    decide_next_stage,
    frozen_a_from_report,
    index_events,
    json_ready,
    load_poc_config,
    load_selected_routes,
    load_tracklet_hits,
    measurement_records,
    parameter_columns,
    run_fd_smoke,
    select_smoke_modules,
    validate_residuals,
)
from alignment.operating_protocol_v1_final_closure import project_root, resolve_under_root
from datasets.root_loader import load_events


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    assert_no_alignment_payload(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(project_root() / "configs/module_level_residual_identifiability_poc_v1.yaml"),
    )
    args = parser.parse_args()
    config = load_poc_config(args.config)
    root = project_root()
    output = resolve_under_root(root, str(config["output_dir"]))
    output.mkdir(parents=True, exist_ok=True)

    selected_path = resolve_under_root(root, str(config["selected_routes"]))
    enhanced_path = resolve_under_root(root, str(config["enhanced_tracklets"]))
    tracklets_path = resolve_under_root(root, str(config["physical_tracklets"]))
    frozen_a_path = resolve_under_root(root, str(config["frozen_a_report"]))
    routes = load_selected_routes(selected_path)
    wanted = {(int(row["run_id"]), int(row["event_id"])) for row in routes}
    events = index_events(load_events(tracklets_path, require_mc_labels=False))
    events = {key: events[key] for key in wanted if key in events}
    hits = load_tracklet_hits(enhanced_path, wanted)
    mapping = build_mapping(routes, events, hits)
    representative = config["representative"]
    station_id = int(representative["station_id"])
    layer_id = int(representative["layer_id"])
    smoke_modules = select_smoke_modules(
        mapping,
        station_id=station_id,
        layer_id=layer_id,
        n_modules=int(representative["n_modules"]),
    )
    measurements = build_measurements(
        routes,
        events,
        hits,
        representative_station=station_id,
    )
    residual_report = validate_residuals(measurements)
    fd_cfg = config["finite_difference"]
    fd_report = run_fd_smoke(
        measurements,
        smoke_modules,
        station_id=station_id,
        translation_steps_mm=fd_cfg["translation_steps_mm"],
        rotation_steps_mrad=fd_cfg["rotation_steps_mrad"],
        c_dx_steps_mm=fd_cfg["c_dx_steps_mm"],
        linearity_max_relative_deviation=float(fd_cfg["linearity_max_relative_deviation"]),
        targeting_max_offmodule_fraction=float(fd_cfg["targeting_max_offmodule_fraction"]),
    )
    names, jacobian = parameter_columns(
        measurements,
        smoke_modules,
        station_id=station_id,
        translation_step_mm=float(fd_cfg["translation_steps_mm"][0]),
        rotation_step_rad=float(fd_cfg["rotation_steps_mrad"][0]) / 1000.0,
        c_dx_step_mm=float(fd_cfg["c_dx_steps_mm"][0]),
    )
    ident = analyze_identifiability(measurements, jacobian, names)
    leakage = compare_leakage(
        measurements,
        names,
        jacobian,
        station_id=station_id,
        translation_step_mm=float(fd_cfg["translation_steps_mm"][0]),
        rotation_step_rad=float(fd_cfg["rotation_steps_mrad"][0]) / 1000.0,
        c_dx_step_mm=float(fd_cfg["c_dx_steps_mm"][0]),
        thresholds=config["leakage_decision"],
        frozen_a=frozen_a_from_report(frozen_a_path),
    )
    decision = decide_next_stage(
        leakage,
        ident,
        fd_report,
        worse_null_condition_ratio=float(config["leakage_decision"]["worse_null_condition_ratio"]),
    )

    created = datetime.now(timezone.utc).isoformat()
    state = common_operating_state()
    mapping_report = {
        **state,
        "schema_version": SCHEMA_VERSION,
        "created_utc": created,
        "source_id": config["source_id"],
        "run": int(config["run"]),
        "selected_routes": str(selected_path),
        "enhanced_tracklets": str(enhanced_path),
        "physical_tracklets": str(tracklets_path),
        "representative": dict(representative),
        "smoke_modules": smoke_modules,
        **mapping,
    }
    residual_out = {
        **state,
        "schema_version": SCHEMA_VERSION,
        "created_utc": created,
        "source_id": config["source_id"],
        **residual_report,
        "n_hits_on_representative_station": int(len(measurements)),
        "measurements": measurement_records(measurements),
    }
    fd_out = {
        **state,
        "schema_version": SCHEMA_VERSION,
        "created_utc": created,
        **fd_report,
    }
    ident_out = {
        **state,
        "schema_version": SCHEMA_VERSION,
        "created_utc": created,
        "representative": dict(representative),
        "smoke_modules": smoke_modules,
        **ident,
    }
    leakage_out = {
        **state,
        "schema_version": SCHEMA_VERSION,
        "created_utc": created,
        "representative_station": station_id,
        **leakage,
    }
    decision_out = {
        **state,
        "schema_version": SCHEMA_VERSION,
        "created_utc": created,
        **decision,
    }
    for payload in (mapping_report, residual_out, fd_out, ident_out, leakage_out, decision_out):
        assert_no_alignment_payload(payload)

    _write_json(output / "module_measurement_mapping_report.json", mapping_report)
    _write_json(output / "module_residual_validation_report.json", residual_out)
    _write_json(output / "module_fd_jacobian_report.json", fd_out)
    _write_json(output / "module_identifiability_report.json", ident_out)
    _write_json(output / "station_vs_module_leakage_comparison.json", leakage_out)
    _write_json(output / "next_stage_decision.json", decision_out)
    print(
        json.dumps(
            {
                "output_dir": str(output),
                "n_measurements": residual_report["n_measurements"],
                "smoke_modules": smoke_modules,
                "fd_passed": fd_report["passed"],
                "answer": decision["answer"],
                "decision": decision["decision"],
                "go_to_full_module_identifiability_map": decision["go_to_full_module_identifiability_map"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
