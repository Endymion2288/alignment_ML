#!/usr/bin/env python3
"""External-constraint inventory, prior-strength Fisher scan, and IOV framework.

Reuses entries 57-59 true cluster-local {dx, ry, C_dx} Jacobian.  Does not
train, write geometry, invent survey numbers, or emit an alignment payload.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from alignment.alternative_track_topology_feasibility import (
    category_keys,
    fd_steps,
    load_feasibility_config,
    load_route_features,
    subset_measurements,
)
from alignment.external_constraint_iov_feasibility import (
    SCHEMA_VERSION,
    collinear_toy_jacobian,
    combined_identifiability,
    common_audit_state,
    decide_next_stage,
    external_constraint_inventory,
    iov_parameterization_report,
    load_framework_config,
    scan_priors,
    year_iov_manifest,
)
from alignment.module_level_residual_poc import json_ready
from alignment.operating_protocol_v1_final_closure import project_root, resolve_under_root
from alignment.true_cluster_local_residual import assert_no_alignment_payload
from alignment.true_cluster_local_stability_transfer import (
    build_jacobian,
    load_run_measurements,
    summarize_jacobian,
)


def _same_order(left: float, right: float, *, factor: float) -> bool:
    if left <= 0.0 or right <= 0.0:
        return False
    ratio = max(left, right) / min(left, right)
    return bool(ratio <= float(factor))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    assert_no_alignment_payload(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _compact_scan(scan: Mapping[str, Any]) -> dict[str, Any]:
    compact_points = []
    for row in scan.get("points") or []:
        compact_points.append(
            {
                "family": row.get("family"),
                "sigma_ry_mrad": row.get("sigma_ry_mrad"),
                "sigma_cdx_mm": row.get("sigma_cdx_mm"),
                "rank": row.get("rank"),
                "sigma3_over_sigma1": row.get("sigma3_over_sigma1"),
                "ry_cdx_correlation": row.get("ry_cdx_correlation"),
                "station_ry_vs_C_dx": row.get("station_ry_vs_C_dx"),
                "unlocked": row.get("unlocked"),
                "marginal_sigma_ry_mrad": row.get("marginal_sigma_ry_mrad"),
                "marginal_sigma_cdx_mm": row.get("marginal_sigma_cdx_mm"),
            }
        )
    payload = dict(scan)
    payload["points"] = compact_points
    return payload


def _load_category_jacobian(
    *,
    run_spec: Mapping[str, Any],
    topology_config: Mapping[str, Any],
    category_id: str,
    root: Path,
    station_id: int,
    steps: Mapping[str, float],
    min_routes: int,
) -> dict[str, Any]:
    physics = topology_config["physics_scales"]
    categories = list(topology_config["predeclared_categories"])
    spec = next(row for row in categories if row["id"] == category_id)
    loaded = load_run_measurements(
        run_spec,
        root=root,
        cluster_dump=run_spec["cluster_dump"],
        representative_station=int(station_id),
    )
    features = load_route_features(run_spec, root=root, physics=physics)
    keys = category_keys(features["selected_features"], spec, jacobian_eligible_only=True)
    subset = subset_measurements(loaded["measurements"], keys)
    n_routes = len({(int(item.run_id), int(item.event_id), int(item.route_index)) for item in subset})
    if n_routes < int(min_routes) or len(subset) < 3:
        return {
            "run": int(run_spec["run"]),
            "category": category_id,
            "insufficient": True,
            "n_routes": n_routes,
            "n_measurements": int(len(subset)),
        }
    jacobian = build_jacobian(subset, station_id=int(station_id), steps=steps)
    point = summarize_jacobian(jacobian)
    return {
        "run": int(run_spec["run"]),
        "category": category_id,
        "insufficient": False,
        "n_routes": n_routes,
        "n_measurements": int(len(subset)),
        "track_only": {
            "rank": point.get("rank"),
            "station_ry_vs_C_dx": point.get("station_ry_vs_C_dx"),
            "station_dx_vs_C_dx": point.get("station_dx_vs_C_dx"),
            "sigma3_over_sigma1": (point.get("singular_value_ratios") or {}).get("sigma3_over_sigma1"),
            "singular_values": point.get("singular_values"),
        },
        "jacobian": jacobian,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(project_root() / "configs/external_constraint_iov_alignment_feasibility_v1.yaml"),
    )
    args = parser.parse_args()
    config = load_framework_config(args.config)
    root = project_root()
    output = resolve_under_root(root, str(config["output_dir"]))
    output.mkdir(parents=True, exist_ok=True)
    created = datetime.now(timezone.utc).isoformat()
    state = common_audit_state()

    inventory = external_constraint_inventory(config)
    inventory["created_utc"] = created
    _write_json(output / "external_constraint_inventory.json", inventory)

    year_manifest = year_iov_manifest(config)
    year_manifest["created_utc"] = created
    _write_json(output / "year_iov_geometry_manifest.json", year_manifest)

    parameterization = iov_parameterization_report(config)
    parameterization["created_utc"] = created
    _write_json(output / "iov_parameterization_report.json", parameterization)

    topology = load_feasibility_config(root / str(config["track_jacobians"]["topology_config"]))
    steps = fd_steps(topology)
    station_id = int(config["representative"]["station_id"])
    min_routes = int(config["unlock"]["min_routes"])
    run_by_id = {int(row["run"]): row for row in topology["runs"]["jacobian"]}
    loaded_j = {}
    for run in config["track_jacobians"]["runs"]:
        for category in (
            config["track_jacobians"]["collinear_category"],
            config["track_jacobians"]["mixed_category"],
        ):
            key = f"{run}_{category}"
            loaded_j[key] = _load_category_jacobian(
                run_spec=run_by_id[int(run)],
                topology_config=topology,
                category_id=str(category),
                root=root,
                station_id=station_id,
                steps=steps,
                min_routes=min_routes,
            )

    physics = config["physics_scales"]
    toy = collinear_toy_jacobian(alpha_mm_per_rad=float(physics["ift_layer_pitch_mm"]))
    toy_scan = scan_priors(toy, config, sample_id="analytic_collinear_toy")
    toy_none = combined_identifiability(
        toy,
        sigma_r_mm=float(physics["measurement_sigma_mm"]),
        rank_tolerance=float(config["unlock"]["rank_tolerance"]),
    )

    collinear_key = f"{config['track_jacobians']['runs'][0]}_{config['track_jacobians']['collinear_category']}"
    mixed_key = f"{config['track_jacobians']['runs'][0]}_{config['track_jacobians']['mixed_category']}"
    collinear_pack = loaded_j[collinear_key]
    mixed_pack = loaded_j[mixed_key]
    if collinear_pack.get("insufficient") or "jacobian" not in collinear_pack:
        raise RuntimeError(f"collinear requirement Jacobian missing: {collinear_key}")
    if mixed_pack.get("insufficient") or "jacobian" not in mixed_pack:
        raise RuntimeError(f"mixed comparison Jacobian missing: {mixed_key}")
    collinear_scan = scan_priors(collinear_pack["jacobian"], config, sample_id=collinear_key)
    mixed_scan = scan_priors(mixed_pack["jacobian"], config, sample_id=mixed_key)
    mixed_scan["track_only_rank"] = (mixed_pack.get("track_only") or {}).get("rank")

    transfer_key = f"{config['track_jacobians']['runs'][1]}_{config['track_jacobians']['collinear_category']}"
    transfer_pack = loaded_j[transfer_key]
    transfer_scan = None
    if not transfer_pack.get("insufficient"):
        transfer_scan = scan_priors(transfer_pack["jacobian"], config, sample_id=transfer_key)

    prior_report = {
        **state,
        "created_utc": created,
        "prior_grid_predeclared": True,
        "prior_selected_from_residual": False,
        "survey_numbers_invented": False,
        "analytic_toy_track_only": {
            "rank": toy_none["rank"],
            "ry_cdx_correlation": toy_none["ry_cdx_correlation"],
            "sigma3_over_sigma1": toy_none["sigma3_over_sigma1"],
            "station_ry_vs_C_dx": toy_none["station_ry_vs_C_dx"],
        },
        "analytic_toy": _compact_scan(toy_scan),
        "collinear_real_track_only": collinear_pack.get("track_only"),
        "collinear_real": _compact_scan(collinear_scan),
        "mixed_real_track_only": mixed_pack.get("track_only"),
        "mixed_real": _compact_scan(mixed_scan),
        "collinear_transfer_run": None if transfer_scan is None else _compact_scan(transfer_scan),
        "information_requirement": {
            "on_collinear_r0022_intermediate_tracks": {
                "sigma_ry_mrad": collinear_scan.get("weakest_ry_only_unlock_mrad"),
                "sigma_cdx_mm": collinear_scan.get("weakest_cdx_only_unlock_mm"),
                "ry_plateau": collinear_scan.get("ry_only_not_overly_sensitive"),
                "cdx_plateau": collinear_scan.get("cdx_only_not_overly_sensitive"),
                "leftover_sigma_cdx_mm_after_tight_ry": collinear_scan.get(
                    "ry_only_leftover_sigma_cdx_mm_at_tightest"
                ),
                "leftover_sigma_ry_mrad_after_tight_cdx": collinear_scan.get(
                    "cdx_only_leftover_sigma_ry_mrad_at_tightest"
                ),
                "leftover_cdx_finer_than_strip_pitch": collinear_scan.get(
                    "ry_only_leftover_cdx_finer_than_strip_pitch"
                ),
            },
            "on_analytic_collinear_toy": {
                "sigma_ry_mrad": toy_scan.get("weakest_ry_only_unlock_mrad"),
                "sigma_cdx_mm": toy_scan.get("weakest_cdx_only_unlock_mm"),
                "leftover_sigma_cdx_mm_after_tight_ry": toy_scan.get(
                    "ry_only_leftover_sigma_cdx_mm_at_tightest"
                ),
            },
            "design_map_mm_per_mrad": physics["c_dx_ry_design_map_mm_per_mrad"],
            "note": (
                "Requirement is read from the rank-2 collision-like Jacobian, not from "
                "the mixed sample that already looks rank-3 at finite sample size."
            ),
        },
    }
    _write_json(output / "survey_prior_requirement_scan.json", prior_report)

    same_ry = None
    same_cdx = None
    if transfer_scan is not None:
        same_ry = transfer_scan.get("weakest_ry_only_unlock_mrad")
        same_cdx = transfer_scan.get("weakest_cdx_only_unlock_mm")
    ident = {
        **state,
        "created_utc": created,
        "jacobian_source": "true cluster-local r_u finite difference {station_dx, station_ry, C_dx}",
        "external_prior_source": "predeclared Gaussian strength scan; no survey central value",
        "collinear_category": config["track_jacobians"]["collinear_category"],
        "mixed_category": config["track_jacobians"]["mixed_category"],
        "runs": list(config["track_jacobians"]["runs"]),
        "track_only": {
            "14973_intermediate": collinear_pack.get("track_only"),
            "14973_mixed": mixed_pack.get("track_only"),
            "14974_intermediate": transfer_pack.get("track_only"),
        },
        "combined_restores_rank3_on_collinear": bool(
            collinear_scan.get("weakest_ry_only_unlock_mrad") is not None
            or collinear_scan.get("weakest_cdx_only_unlock_mm") is not None
        ),
        "analytic_toy_restores_rank3": bool(
            toy_scan.get("weakest_ry_only_unlock_mrad") is not None
            or toy_scan.get("weakest_cdx_only_unlock_mm") is not None
        ),
        "requirement_sample": "collinear_intermediate_angle_not_mixed_ge3",
        "not_overly_sensitive": {
            "collinear_ry_only": collinear_scan.get("ry_only_not_overly_sensitive"),
            "collinear_cdx_only": collinear_scan.get("cdx_only_not_overly_sensitive"),
        },
        "leftover_after_tight_prior": {
            "sigma_cdx_mm_after_tight_ry": collinear_scan.get(
                "ry_only_leftover_sigma_cdx_mm_at_tightest"
            ),
            "sigma_ry_mrad_after_tight_cdx": collinear_scan.get(
                "cdx_only_leftover_sigma_ry_mrad_at_tightest"
            ),
            "cdx_finer_than_strip_pitch": collinear_scan.get(
                "ry_only_leftover_cdx_finer_than_strip_pitch"
            ),
        },
        "within_2024_r0022_transfer_of_requirement": {
            "14974_sigma_ry_mrad": same_ry,
            "14974_sigma_cdx_mm": same_cdx,
            "same_order_as_14973": bool(
                collinear_scan.get("weakest_ry_only_unlock_mrad") is not None
                and same_ry is not None
                and _same_order(
                    float(collinear_scan["weakest_ry_only_unlock_mrad"]),
                    float(same_ry),
                    factor=float(config["unlock"]["sensitivity_sigma_factor"]),
                )
            ),
            "note": "Same year/IOV only.  Not a cross-year transfer of correction values.",
        },
        "cross_year_correction_values": "forbidden",
        "emits_alignment_payload": False,
    }
    _write_json(output / "combined_track_external_identifiability_report.json", ident)

    decision = decide_next_stage(
        inventory=inventory,
        collinear_scan=collinear_scan,
        mixed_scan=mixed_scan,
        parameterization=parameterization,
    )
    decision["created_utc"] = created
    decision["schema_version"] = SCHEMA_VERSION
    _write_json(output / "next_stage_decision.json", decision)
    print(
        json.dumps(
            {
                "output_dir": str(output),
                "decision": decision["decision"],
                "sigma_ry_mrad": collinear_scan.get("weakest_ry_only_unlock_mrad"),
                "sigma_cdx_mm": collinear_scan.get("weakest_cdx_only_unlock_mm"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
