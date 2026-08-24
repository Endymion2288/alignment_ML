#!/usr/bin/env python3
"""Residual-blind topology inventory and predeclared-category Jacobian audit.

Same true cluster-local r_u, surface, leave-one-station-out, FD steps, and
frozen V2 routes as entries 57-58.  No new network, no cosine retuning.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from alignment.alternative_track_topology_feasibility import (
    SCHEMA_VERSION,
    bootstrap_pack,
    category_keys,
    common_audit_state,
    decide_next_stage,
    evaluate_category_portability,
    fd_steps,
    inventory_selected_features,
    load_feasibility_config,
    load_route_features,
    occupancy_four_station_snapshot,
    parameter_subspace_transfer,
    predeclared_category_report,
    r0022_shallow_dominated,
    stacked_rows,
    subset_measurements,
    summarize_jacobian,
    survey_alternative_samples,
)
from alignment.operating_protocol_v1_final_closure import project_root, resolve_under_root
from alignment.true_cluster_local_residual import assert_no_alignment_payload, json_ready
from alignment.true_cluster_local_stability_transfer import (
    build_jacobian,
    group_indices,
    load_run_measurements,
)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    assert_no_alignment_payload(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _compact_point(point: Mapping[str, Any]) -> dict[str, Any]:
    ratios = point.get("singular_value_ratios") or {}
    return {
        "n_rows": point.get("n_rows"),
        "n_routes": point.get("n_routes"),
        "station_dx_vs_C_dx": point.get("station_dx_vs_C_dx"),
        "station_ry_vs_C_dx": point.get("station_ry_vs_C_dx"),
        "rank": point.get("rank"),
        "singular_values": point.get("singular_values"),
        "sigma3_over_sigma1": ratios.get("sigma3_over_sigma1"),
        "sigma2_over_sigma1": ratios.get("sigma2_over_sigma1"),
        "right_singular_vectors": point.get("right_singular_vectors"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(project_root() / "configs/alternative_track_topology_identifiability_feasibility_v1.yaml"),
    )
    parser.add_argument("--skip-parent-tracklets", action="store_true")
    args = parser.parse_args()
    config = load_feasibility_config(args.config)
    root = project_root()
    output = resolve_under_root(root, str(config["output_dir"]))
    output.mkdir(parents=True, exist_ok=True)
    physics = config["physics_scales"]
    categories = list(config["predeclared_categories"])
    station_id = int(config["representative"]["station_id"])
    steps = fd_steps(config)
    boot_cfg = config["bootstrap"]
    proxy_ry = float(config["entry_57"]["module_proxy_ry_cdx_abs_cosine"])
    created = datetime.now(timezone.utc).isoformat()
    state = common_audit_state()

    category_doc = predeclared_category_report(config)
    _write_json(output / "predeclared_topology_categories.json", {**state, "created_utc": created, **category_doc})

    inventory_specs = list(config["runs"]["jacobian"]) + list(config["runs"].get("inventory_only") or [])
    run_inventories = []
    features_by_run: dict[int, list] = {}
    for spec in inventory_specs:
        loaded = load_route_features(spec, root=root, physics=physics)
        if args.skip_parent_tracklets:
            loaded["parent_tracklet_inventory"] = {"skipped": True}
        selected = inventory_selected_features(
            loaded["selected_features"],
            categories,
            n_processed_events=int(spec.get("nevents") or 0),
        )
        features_by_run[int(spec["run"])] = loaded["selected_features"]
        run_inventories.append(
            {
                "run": loaded["run"],
                "source_id": loaded["source_id"],
                "role": loaded["role"],
                "corpus": loaded["corpus"],
                "skip_events": loaded["skip_events"],
                "nevents": loaded["nevents"],
                "participates_in_method_selection": False,
                "selected": selected,
                "parent_tracklets": loaded["parent_tracklet_inventory"],
            }
        )

    occupancy = occupancy_four_station_snapshot(config.get("occupancy_preflight_root"), root)
    alternative = survey_alternative_samples(config)
    shallow = r0022_shallow_dominated(run_inventories, physics)
    totals = {
        "n_runs": int(len(run_inventories)),
        "n_selected_routes": int(sum(int(row["selected"]["n_selected_routes"]) for row in run_inventories)),
        "n_jacobian_eligible": int(sum(int(row["selected"]["n_jacobian_eligible"]) for row in run_inventories)),
        "n_complete_four_station": int(sum(int(row["selected"]["n_complete_four_station"]) for row in run_inventories)),
        "n_wide_incident_angle_eligible": int(
            sum(int(row["selected"]["n_wide_incident_angle_eligible"]) for row in run_inventories)
        ),
    }
    _write_json(
        output / "topology_inventory_report.json",
        {
            **state,
            "created_utc": created,
            "schema_version": SCHEMA_VERSION,
            "residual_blind": True,
            "feature_source": category_doc["feature_source"],
            "r0022_still_shallow_dominated": shallow,
            "totals": totals,
            "runs": run_inventories,
            "occupancy_preflight_four_station": occupancy,
            "alternative_sample_survey": alternative,
        },
    )

    jacobian_results: dict[int, dict[str, Any]] = {}
    for spec in config["runs"]["jacobian"]:
        loaded = load_run_measurements(
            spec,
            root=root,
            cluster_dump=spec["cluster_dump"],
            representative_station=station_id,
        )
        if not loaded["measurements"]:
            jacobian_results[int(spec["run"])] = {
                "run": int(spec["run"]),
                "role": spec.get("role"),
                "skipped": True,
                "reason": "no IFT cluster-local residuals",
            }
            continue
        jacobian = build_jacobian(loaded["measurements"], station_id=station_id, steps=steps)
        groups = group_indices(loaded["measurements"])
        features = features_by_run[int(spec["run"])]
        by_category = {}
        for spec_cat in categories:
            keys = category_keys(features, spec_cat)
            measurements = subset_measurements(loaded["measurements"], keys)
            n_routes = int(len({route for route in keys if route in groups}))
            if len(measurements) < 3 or n_routes < 1:
                by_category[str(spec_cat["id"])] = {
                    "insufficient": True,
                    "n_routes": n_routes,
                    "n_measurements": int(len(measurements)),
                    "reason": "too few Jacobian rows",
                }
                continue
            index_groups = [groups[key] for key in keys if key in groups]
            subset_j = stacked_rows(jacobian, index_groups)
            point = summarize_jacobian(subset_j)
            point["n_routes"] = n_routes
            boot = bootstrap_pack(
                subset_j,
                measurements,
                n_event=int(boot_cfg["n_event_replicates"]),
                n_half=int(boot_cfg["n_half_splits"]),
                seed=int(boot_cfg["seed"]) + int(spec["run"]),
                min_routes=int(boot_cfg["min_routes"]),
                reference_ry=float(point["station_ry_vs_C_dx"]),
            )
            by_category[str(spec_cat["id"])] = {
                "insufficient": bool(boot.get("insufficient")),
                "n_routes": n_routes,
                "n_measurements": int(len(measurements)),
                "point": point,
                "bootstrap": boot,
                "can_unlock_module_map": bool(spec_cat.get("can_unlock_module_map")),
                "role": spec_cat.get("role"),
                "participates_in_method_selection": False,
            }
        jacobian_results[int(spec["run"])] = {
            "run": int(spec["run"]),
            "role": spec.get("role"),
            "source_id": spec.get("source_id"),
            "n_measurements": loaded["n_measurements"],
            "n_routes_with_residuals": loaded["n_routes_with_residuals"],
            "join_complete": loaded["join_complete"],
            "fd_steps": steps,
            "categories": by_category,
        }

    stability_rows = []
    for spec_cat in categories:
        cid = str(spec_cat["id"])
        per_run = {}
        for run_id, block in jacobian_results.items():
            per_run[str(run_id)] = (block.get("categories") or {}).get(cid)
        stability_rows.append({"category": cid, "role": spec_cat.get("role"), "runs": per_run})
    _write_json(
        output / "topology_jacobian_stability_report.json",
        {
            **state,
            "created_utc": created,
            "schema_version": SCHEMA_VERSION,
            "identical_construction": {
                "residual": "r_u = u_cluster - u_track^unbiased on SiDetectorElement",
                "unbiased_method": "leave_one_station_out_cluster_globals_projected_to_detector_surface",
                "fd_steps": steps,
                "frozen_v2_routes": True,
                "no_retuning": True,
                "categories_frozen_before_jacobian": True,
            },
            "categories": stability_rows,
        },
    )

    transfer_rows = []
    evaluations = []
    ref_run = jacobian_results.get(14973, {})
    tr_run = jacobian_results.get(14974, {})
    for spec_cat in categories:
        cid = str(spec_cat["id"])
        ref_cat = (ref_run.get("categories") or {}).get(cid)
        tr_cat = (tr_run.get("categories") or {}).get(cid)
        subspace = None
        if (
            ref_cat
            and tr_cat
            and not ref_cat.get("insufficient")
            and not tr_cat.get("insufficient")
            and ref_cat.get("point")
            and tr_cat.get("point")
        ):
            subspace = parameter_subspace_transfer(ref_cat["point"], tr_cat["point"])
        evaluation = evaluate_category_portability(
            spec=spec_cat,
            reference=ref_cat,
            transfer=tr_cat,
            proxy_ry=proxy_ry,
            min_routes=int(boot_cfg["min_routes"]),
        )
        evaluations.append(evaluation)
        read_only = {}
        for run_id in (14975, 14976):
            block = ((jacobian_results.get(run_id) or {}).get("categories") or {}).get(cid)
            if not block or block.get("insufficient") or not block.get("point"):
                read_only[str(run_id)] = {"skipped": True, "block": block}
                continue
            angles = None
            if ref_cat and ref_cat.get("point"):
                angles = parameter_subspace_transfer(ref_cat["point"], block["point"])
            read_only[str(run_id)] = {
                "point": _compact_point(block["point"]),
                "n_routes": block.get("n_routes"),
                "subspace_vs_14973": angles,
                "participates_in_method_selection": False,
            }
        transfer_rows.append(
            {
                "category": cid,
                "evaluation": evaluation,
                "reference_14973": None if not ref_cat else _compact_point(ref_cat.get("point") or {}),
                "transfer_14974": None if not tr_cat else _compact_point(tr_cat.get("point") or {}),
                "parameter_space_principal_angles": subspace,
                "read_only_runs": read_only,
            }
        )
    _write_json(
        output / "cross_run_topology_transfer_report.json",
        {
            **state,
            "created_utc": created,
            "schema_version": SCHEMA_VERSION,
            "module_proxy_ry_cdx_abs_cosine": proxy_ry,
            "categories": transfer_rows,
        },
    )

    decision = decide_next_stage(
        evaluations=evaluations,
        inventory_summary={"r0022_still_shallow_dominated": shallow},
        alternative_survey=alternative,
    )
    _write_json(
        output / "next_stage_decision.json",
        {
            **state,
            "created_utc": created,
            "schema_version": SCHEMA_VERSION,
            **decision,
        },
    )
    print(
        json.dumps(
            {
                "output_dir": str(output),
                "r0022_still_shallow_dominated": shallow,
                "decision": decision["decision"],
                "answer": decision["answer"],
                "portable_categories": decision.get("portable_categories"),
                "n_inventory_runs": int(len(run_inventories)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
