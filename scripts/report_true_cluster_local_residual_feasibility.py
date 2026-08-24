#!/usr/bin/env python3
"""Stage 1.5: true cluster-local residual feasibility on frozen V2 routes.

Optional Athena dump of SCT_ClusterContainer local u and SiDetectorElement
surfaces, then unbiased r_u Jacobians.  No new network, no Station / C_dx
Mode, no official geometry write.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from alignment.module_level_residual_poc import (
    build_mapping,
    index_events,
    load_selected_routes,
    load_tracklet_hits,
    select_smoke_modules,
)
from alignment.operating_protocol_v1_final_closure import project_root, resolve_under_root
from alignment.true_cluster_local_residual import (
    SCHEMA_VERSION,
    assert_no_alignment_payload,
    build_cluster_residuals,
    common_operating_state,
    compare_leakage,
    decide_next_stage,
    frozen_a_from_report,
    json_ready,
    load_cluster_local_hits,
    load_json,
    load_stage_config,
    measurement_records,
    now_utc,
    parameter_columns,
    run_fd_smoke,
    summarize_cluster_export,
    validate_residuals,
    write_event_list,
)
from datasets.root_loader import load_events


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    assert_no_alignment_payload(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(project_root() / "configs/true_cluster_local_residual_feasibility_v1.yaml"),
    )
    parser.add_argument(
        "--cluster-dump",
        default="",
        help="Existing cluster_local ROOT dump; default is output_dir/cluster_local.root",
    )
    args = parser.parse_args()
    config = load_stage_config(args.config)
    root = project_root()
    output = resolve_under_root(root, str(config["output_dir"]))
    output.mkdir(parents=True, exist_ok=True)

    selected_path = resolve_under_root(root, str(config["selected_routes"]))
    enhanced_path = resolve_under_root(root, str(config["enhanced_tracklets"]))
    tracklets_path = resolve_under_root(root, str(config["physical_tracklets"]))
    frozen_a_path = resolve_under_root(root, str(config["frozen_a_report"]))
    previous_path = resolve_under_root(root, str(config["previous_poc_comparison"]))
    dump_path = (
        Path(args.cluster_dump).expanduser().resolve()
        if args.cluster_dump
        else output / "cluster_local.root"
    )
    if not dump_path.is_file():
        raise FileNotFoundError(
            f"cluster-local dump is missing: {dump_path}. "
            "Run scripts/run_true_cluster_local_dump.py first."
        )

    routes = load_selected_routes(selected_path)
    wanted = {(int(row["run_id"]), int(row["event_id"])) for row in routes}
    events = index_events(load_events(tracklets_path, require_mc_labels=False))
    events = {key: events[key] for key in wanted if key in events}
    hits = load_tracklet_hits(enhanced_path, wanted)
    mapping = build_mapping(routes, events, hits)
    event_list = write_event_list(routes, output / "selected_events.txt")
    clusters = load_cluster_local_hits(dump_path)
    export_report = summarize_cluster_export(
        clusters,
        routes,
        hits,
        dump_path=dump_path,
        event_list=event_list,
        edm=config["edm"],
    )
    if not export_report["join_complete"]:
        raise RuntimeError(
            "selected-route cluster identifiers did not join to the SCT dump; "
            f"missing {export_report['n_selected_route_clusters_missing']}"
        )

    representative = config["representative"]
    station_id = int(representative["station_id"])
    smoke_modules = select_smoke_modules(
        mapping,
        station_id=station_id,
        layer_id=int(representative["layer_id"]),
        n_modules=int(representative["n_modules"]),
    )
    measurements = build_cluster_residuals(
        routes,
        events,
        hits,
        clusters,
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
        l1_locality_max_fraction=float(fd_cfg["l1_locality_max_fraction"]),
    )
    names, jacobian = parameter_columns(
        measurements,
        station_id=station_id,
        translation_step_mm=float(fd_cfg["translation_steps_mm"][0]),
        rotation_step_rad=float(fd_cfg["rotation_steps_mrad"][0]) / 1000.0,
        c_dx_step_mm=float(fd_cfg["c_dx_steps_mm"][0]),
    )
    previous = load_json(previous_path)
    leakage = compare_leakage(
        measurements,
        jacobian,
        station_id=station_id,
        fd_report=fd_report,
        previous_poc=previous,
        thresholds=config["leakage_decision"],
        frozen_a=frozen_a_from_report(frozen_a_path),
        steps={
            "translation_step_mm": float(fd_cfg["translation_steps_mm"][0]),
            "rotation_step_rad": float(fd_cfg["rotation_steps_mrad"][0]) / 1000.0,
            "c_dx_step_mm": float(fd_cfg["c_dx_steps_mm"][0]),
        },
    )
    decision = decide_next_stage(leakage, fd_report)
    created = now_utc()
    state = common_operating_state()

    _write_json(
        output / "cluster_local_measurement_export_report.json",
        {
            **state,
            "schema_version": SCHEMA_VERSION,
            "created_utc": created,
            "source_id": config["source_id"],
            "run": int(config["run"]),
            "selected_routes": str(selected_path),
            "enhanced_tracklets": str(enhanced_path),
            "input_xaod": str(config["input_xaod"]),
            "ntuple_lacked_local_u_and_surface": True,
            "minimal_export": "ClusterLocalDumpAlg from SCT_ClusterContainer + SiDetectorElement",
            **mapping,
            **export_report,
            "position_source": "FaserSCT_Cluster.localPosition Trk::locX on SiDetectorElement",
            "calypso_unbiased_cluster_residual_available": False,
            "constructed_unbiased_surface_residual": True,
        },
    )
    _write_json(
        output / "unbiased_cluster_residual_validation.json",
        {
            **state,
            "schema_version": SCHEMA_VERSION,
            "created_utc": created,
            "source_id": config["source_id"],
            **residual_report,
            "n_hits_on_representative_station": int(len(measurements)),
            "measurements": measurement_records(measurements),
        },
    )
    _write_json(
        output / "true_cluster_fd_jacobian_report.json",
        {
            **state,
            "schema_version": SCHEMA_VERSION,
            "created_utc": created,
            "source_id": config["source_id"],
            "parameter_names": names,
            "jacobian_column_norms": {
                name: float(np.linalg.norm(jacobian[:, index]))
                for index, name in enumerate(names)
            },
            **fd_report,
        },
    )
    _write_json(
        output / "station_dx_ry_cdx_leakage_comparison.json",
        {
            **state,
            "schema_version": SCHEMA_VERSION,
            "created_utc": created,
            "source_id": config["source_id"],
            "representative_station": station_id,
            "previous_poc_comparison": str(previous_path),
            **leakage,
        },
    )
    _write_json(
        output / "next_stage_decision.json",
        {
            **state,
            "schema_version": SCHEMA_VERSION,
            "created_utc": created,
            "source_id": config["source_id"],
            **decision,
        },
    )
    print(
        json.dumps(
            {
                "output_dir": str(output),
                "n_cluster_residuals": int(len(measurements)),
                "join_complete": export_report["join_complete"],
                "fd_passed": fd_report["passed"],
                "true_cluster_local_abs_cosine": leakage["true_cluster_local_abs_cosine"],
                "decision": decision["decision"],
                "answer": decision["answer"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
