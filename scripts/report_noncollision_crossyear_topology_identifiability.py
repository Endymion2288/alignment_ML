#!/usr/bin/env python3
"""Non-collision / cross-year topology inventory and identifiability gate.

Residual-blind provenance, PHYS topology, and Jacobian admission.  Reuses
entries 57-59 cluster-local Jacobian only if a predeclared real topology
beats the frozen r0022 collision baseline.  No new network, no cosine cut.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from alignment.noncollision_crossyear_topology import (
    SCHEMA_VERSION,
    common_audit_state,
    decide_next_stage,
    inventory_eos_layout,
    jacobian_admission_for_probe,
    load_inventory_config,
    parse_job_log,
    predeclared_topology_report,
    probe_xaod,
    summarize_phys_topology,
)
from alignment.operating_protocol_v1_final_closure import project_root, resolve_under_root
from alignment.true_cluster_local_residual import assert_no_alignment_payload
from alignment.module_level_residual_poc import json_ready


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    assert_no_alignment_payload(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(project_root() / "configs/noncollision_crossyear_topology_identifiability_v1.yaml"),
    )
    args = parser.parse_args()
    config = load_inventory_config(args.config)
    root = project_root()
    output = resolve_under_root(root, str(config["output_dir"]))
    output.mkdir(parents=True, exist_ok=True)
    created = datetime.now(timezone.utc).isoformat()
    physics = config["physics_scales"]
    baseline = config["r0022_collision_baseline"]
    rules = config["jacobian_admission"]
    state = common_audit_state()

    layout = inventory_eos_layout(config)
    log_probes = []
    for row in config.get("log_probes") or []:
        parsed = parse_job_log(row["path"])
        parsed["id"] = row["id"]
        log_probes.append(parsed)
    xaod_probes = []
    for row in config.get("xaod_probes") or []:
        probed = probe_xaod(row["path"], year=row.get("year"), rec_tag=row.get("rec_tag"))
        probed["id"] = row["id"]
        xaod_probes.append(probed)

    data_source_inventory = {
        **state,
        "created_utc": created,
        "eos_layout": layout,
        "xaod_probes": xaod_probes,
        "log_probes": [{"id": row["id"], "path": row["path"], "exists": row["exists"]} for row in log_probes],
        "years_are_topology_sources_only": True,
        "do_not_mix_cross_year_residuals_or_alignment_constants": True,
    }
    _write_json(output / "data_source_inventory.json", data_source_inventory)

    iov_rows = []
    for row in log_probes:
        iov_rows.append(
            {
                "id": row["id"],
                "path": row["path"],
                "exists": row["exists"],
                "geometry_tag": row.get("geometry_tag"),
                "conditions_tag": row.get("conditions_tag"),
                "geom_flag": row.get("geom_flag"),
                "no_ift_four_station_ckf": row.get("no_ift_four_station_ckf"),
                "use_ift": row.get("use_ift"),
                "cosmics_only": row.get("cosmics_only"),
                "stable_beams_flag": row.get("stable_beams_flag"),
                "backward": row.get("backward"),
                "trigger_mask": row.get("trigger_mask"),
                "remaining_args": row.get("remaining_args"),
            }
        )
    xaod_meta = {
        row["id"]: {
            "n_events": row.get("n_events"),
            "usable_as_cluster_local_jacobian_input": row.get("usable_as_cluster_local_jacobian_input"),
            "missing_required_for_cdx_jacobian": row.get("missing_required_for_cdx_jacobian"),
            "has_ift_ckf_collection": row.get("has_ift_ckf_collection"),
            "has_three_station_ckf": row.get("has_three_station_ckf"),
            "has_lhc_data": row.get("has_lhc_data"),
            "file_metadata": row.get("file_metadata"),
            "year": row.get("year"),
            "rec_tag": row.get("rec_tag"),
        }
        for row in xaod_probes
    }
    geometry_manifest = {
        **state,
        "created_utc": created,
        "production_vs_current_protocol": {
            "2024_r0022_production_conditions": "OFLCOND-FASER-05",
            "operating_protocol_current_conditions": "OFLCOND-FASER-06",
            "2024_r0022_production_geometry": "FASERNU-04",
            "do_not_treat_other_years_as_the_same_alignment_iov": True,
        },
        "log_derived_iovs": iov_rows,
        "xaod_collections": xaod_meta,
        "cdx_jacobian_requires": {
            "ift_station_0": True,
            "sct_cluster_container": True,
            "segment_fit": True,
            "same_period_geometry_and_conditions": True,
            "no_cross_year_constant_mix": True,
        },
        "testbeam": {
            "geometry_tag": "FASER-TB00",
            "conditions_tag": "OFLCOND-FASER-TB00",
            "ift_cdx_applicable": False,
        },
        "note": (
            "--noIFT in faser_reco.py disables 4-station CKF, not SegmentFit. "
            "IFT clusters can still exist. C_dx still requires station-0 measurements "
            "and must be evaluated inside one geometry/conditions IOV."
        ),
    }
    _write_json(output / "geometry_conditions_iov_manifest.json", geometry_manifest)

    phys_cfg = config["phys_topology_probes"]
    max_events = int(phys_cfg.get("max_events") or 20000)
    topology_rows = []
    for row in phys_cfg.get("files") or []:
        stats = summarize_phys_topology(row["path"], physics, max_events=max_events)
        topology_rows.append(
            {
                "id": row["id"],
                "topology_id": row["topology_id"],
                "role": row.get("role"),
                "path": row["path"],
                "topology": stats,
            }
        )
    topology_inventory = {
        **state,
        "created_utc": created,
        "r0022_collision_baseline": dict(baseline),
        "slope_used_for_gate": "ckf_single_track_endpoint_xyz",
        "slope_rejected_for_gate": [
            "local_segmentfit_tx_ty",
            "unassociated_event_segment_endpoints",
        ],
        "probes": topology_rows,
        "note": (
            "Inventory is residual-blind. Category membership was declared from "
            "metadata before these rates were computed. Unassociated PHYS segments "
            "are not treated as spectrometer routes."
        ),
    }
    _write_json(output / "residual_blind_topology_inventory.json", topology_inventory)

    candidate_report = predeclared_topology_report(config)
    admissions = [
        jacobian_admission_for_probe(row, baseline=baseline, rules=rules)
        for row in topology_rows
    ]
    admitted = [row for row in admissions if row.get("admitted_to_jacobian")]
    candidate_report.update(
        {
            "created_utc": created,
            "measured_topology_attached": True,
            "jacobian_admissions": admissions,
            "n_admitted": len(admitted),
            "testbeam_xaod": xaod_meta.get("xaod_testbeam_003395"),
            "2023_cos_xaod_in_rec_tree": False,
            "2023_cos_xaod_note": (
                "2023_cos runs 011780-013815 have rec logs under rec/2023/r0019_log "
                "but no retained xAOD run directories in rec/2023. PHYS-only; "
                "cluster-local Jacobian cannot run."
            ),
        }
    )
    _write_json(output / "candidate_noncollision_topologies.json", candidate_report)

    jacobian_executed = False
    portable_within_period = False
    identifiability = {
        **state,
        "created_utc": created,
        "jacobian_executed": False,
        "reason_jacobian_skipped": (
            "No predeclared topology passed the residual-blind admission gate "
            "(wide-dominated CKF >=90%, or 3x occupancy four-station for cosmic-like "
            "streams, with IFT, xAOD, and >=2 same-period runs). Entries 57-59 "
            "Jacobian is not applied to a failed gate, and cross-year correction "
            "values are not compared."
        ),
        "admissions": admissions,
        "within_period_transfer": None,
        "cross_year_identifiability_structure": "not_compared_gate_failed",
        "cross_year_correction_values": "forbidden",
        "reused_method_if_admitted": {
            "residual": "true cluster-local r_u on SiDetectorElement",
            "unbiased": "leave-one-station-out surface projection",
            "parameters": "{station dx, station ry, C_dx}",
            "fd_steps": dict(config["finite_difference"]),
        },
        "portable_high_lever_arm_topology_found": False,
    }
    if admitted:
        identifiability["reason_jacobian_skipped"] = (
            "Admission list is non-empty in config evaluation but this audit "
            "still requires xAOD+IFT+same-period transfer; check admissions."
        )
    _write_json(output / "crossrun_identifiability_report.json", identifiability)

    decision = decide_next_stage(
        admissions=admissions,
        jacobian_executed=jacobian_executed,
        portable_within_period=portable_within_period,
    )
    decision["created_utc"] = created
    decision["schema_version"] = SCHEMA_VERSION
    decision["entry_59_decision"] = config["entry_59"]["decision"]
    decision["do_not_enter_full_module_identifiability_map"] = True
    _write_json(output / "next_stage_decision.json", decision)
    print(json.dumps({"output_dir": str(output), "decision": decision["decision"]}, indent=2))


if __name__ == "__main__":
    main()
