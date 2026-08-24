#!/usr/bin/env python3
"""Stability and 14974 transfer audit of the entry-57 true cluster-local Jacobian.

Same r_u, surface, leave-one-station-out prediction, FD steps, and frozen V2
routes.  No new network, no cosine retuning, no geometry write.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from alignment.module_level_residual_poc import load_selected_routes
from alignment.operating_protocol_v1_final_closure import project_root, resolve_under_root
from alignment.true_cluster_local_residual import (
    assert_no_alignment_payload,
    json_ready,
)
from alignment.true_cluster_local_stability_transfer import (
    SCHEMA_VERSION,
    attach_route_multiplicity,
    bootstrap_event_summaries,
    build_jacobian,
    collect_metric,
    common_audit_state,
    coverage_matched_draws,
    coverage_tables,
    decide_next_stage,
    event_groups,
    fd_steps,
    group_indices,
    half_split_summaries,
    load_audit_config,
    load_run_measurements,
    parameter_subspace_transfer,
    route_metadata,
    summarize_distribution,
    summarize_jacobian,
)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    assert_no_alignment_payload(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _bootstrap_pack(
    jacobian,
    measurements,
    *,
    n_event: int,
    n_half: int,
    seed: int,
    min_routes: int,
    reference_ry: float | None = None,
    reference_dx: float | None = None,
) -> dict[str, Any]:
    event_rows = bootstrap_event_summaries(
        jacobian,
        measurements,
        n_replicates=n_event,
        seed=seed,
        min_routes=min_routes,
    )
    halves = half_split_summaries(
        jacobian,
        measurements,
        n_splits=n_half,
        seed=seed,
        min_routes=min_routes,
    )
    half_ry = [abs(left["station_ry_vs_C_dx"] - right["station_ry_vs_C_dx"]) for left, right in halves]
    return {
        "n_events": int(len(event_groups(measurements))),
        "n_routes": int(len(group_indices(measurements))),
        "n_measurements": int(len(measurements)),
        "event_bootstrap": {
            "n_replicates": int(n_event),
            "station_dx_vs_C_dx": summarize_distribution(
                collect_metric(event_rows, "station_dx_vs_C_dx"),
                reference=reference_dx,
            ),
            "station_ry_vs_C_dx": summarize_distribution(
                collect_metric(event_rows, "station_ry_vs_C_dx"),
                reference=reference_ry,
            ),
            "rank": summarize_distribution(collect_metric(event_rows, "rank")),
            "sigma2_over_sigma1": summarize_distribution(
                [
                    float((row.get("singular_value_ratios") or {}).get("sigma2_over_sigma1") or float("nan"))
                    for row in event_rows
                ]
            ),
            "sigma3_over_sigma1": summarize_distribution(
                [
                    float((row.get("singular_value_ratios") or {}).get("sigma3_over_sigma1") or float("nan"))
                    for row in event_rows
                ]
            ),
        },
        "random_half_split": {
            "n_splits": int(n_half),
            "station_ry_vs_C_dx": summarize_distribution(
                [float(left["station_ry_vs_C_dx"]) for left, right in halves]
                + [float(right["station_ry_vs_C_dx"]) for left, right in halves]
            ),
            "abs_half_difference_ry_cdx": summarize_distribution(half_ry),
        },
    }


def _run_point(loaded: Mapping[str, Any], jacobian, station_id: int, steps: Mapping[str, float]) -> dict[str, Any]:
    summary = summarize_jacobian(jacobian)
    return {
        "run": loaded["run"],
        "source_id": loaded["source_id"],
        "role": loaded["role"],
        "participates_in_method_selection": loaded["participates_in_method_selection"],
        "n_selected_routes": loaded["n_selected_routes"],
        "n_routes_with_residuals": loaded["n_routes_with_residuals"],
        "n_measurements": loaded["n_measurements"],
        "join_complete": loaded["join_complete"],
        "representative_station": int(station_id),
        "fd_steps": dict(steps),
        "unbiased_method": loaded["unbiased_method"],
        **summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(project_root() / "configs/true_cluster_local_ry_cdx_stability_transfer_v1.yaml"),
    )
    parser.add_argument("--skip-read-only", action="store_true")
    args = parser.parse_args()
    config = load_audit_config(args.config)
    root = project_root()
    output = resolve_under_root(root, str(config["output_dir"]))
    output.mkdir(parents=True, exist_ok=True)
    station_id = int(config["representative"]["station_id"])
    steps = fd_steps(config)
    boot_cfg = config["bootstrap"]
    cov_cfg = config["coverage"]
    entry57 = config["entry_57"]
    proxy_ry = float(entry57["module_proxy_ry_cdx_abs_cosine"])
    state = common_audit_state()
    created = datetime.now(timezone.utc).isoformat()

    reference_spec = config["runs"]["reference"]
    transfer_spec = config["runs"]["transfer"]
    reference = load_run_measurements(
        reference_spec,
        root=root,
        cluster_dump=reference_spec.get("cluster_dump") or entry57["cluster_dump"],
        representative_station=station_id,
    )
    transfer_dump = resolve_under_root(root, str(config["output_dir"])) / "dumps" / "cluster_local_r14974.root"
    if transfer_spec.get("cluster_dump"):
        transfer_dump = resolve_under_root(root, str(transfer_spec["cluster_dump"]))
    transfer = load_run_measurements(
        transfer_spec,
        root=root,
        cluster_dump=transfer_dump,
        representative_station=station_id,
    )
    if not reference["join_complete"] or not reference["measurements"]:
        raise RuntimeError("14973 cluster-local residuals failed to rebuild")
    if not transfer["join_complete"] or not transfer["measurements"]:
        raise RuntimeError("14974 cluster-local residuals failed to rebuild")

    ref_j = build_jacobian(reference["measurements"], station_id=station_id, steps=steps)
    tr_j = build_jacobian(transfer["measurements"], station_id=station_id, steps=steps)
    ref_point = _run_point(reference, ref_j, station_id, steps)
    tr_point = _run_point(transfer, tr_j, station_id, steps)

    ref_boot = _bootstrap_pack(
        ref_j,
        reference["measurements"],
        n_event=int(boot_cfg["n_event_replicates"]),
        n_half=int(boot_cfg["n_half_splits"]),
        seed=int(boot_cfg["seed"]),
        min_routes=int(boot_cfg["min_routes"]),
        reference_ry=float(entry57["observed_ry_cdx_abs_cosine"]),
        reference_dx=float(entry57["observed_dx_cdx_abs_cosine"]),
    )
    tr_boot = _bootstrap_pack(
        tr_j,
        transfer["measurements"],
        n_event=int(boot_cfg["n_event_replicates"]),
        n_half=int(boot_cfg["n_half_splits"]),
        seed=int(boot_cfg["seed"]) + 14974,
        min_routes=int(boot_cfg["min_routes"]),
        reference_ry=float(tr_point["station_ry_vs_C_dx"]),
        reference_dx=float(tr_point["station_dx_vs_C_dx"]),
    )

    ref_routes = load_selected_routes(resolve_under_root(root, str(reference_spec["selected_routes"])))
    tr_routes = load_selected_routes(resolve_under_root(root, str(transfer_spec["selected_routes"])))
    ref_meta = route_metadata(reference["measurements"])
    tr_meta = route_metadata(transfer["measurements"])
    attach_route_multiplicity(ref_meta, ref_routes)
    attach_route_multiplicity(tr_meta, tr_routes)
    ref_groups = group_indices(reference["measurements"])
    tr_groups = group_indices(transfer["measurements"])
    ref_coverage = coverage_tables(
        ref_j,
        reference["measurements"],
        ref_meta,
        ref_groups,
        n_slope_bins=int(cov_cfg["n_slope_bins"]),
        min_subset_routes=int(cov_cfg["min_subset_routes"]),
    )
    tr_coverage = coverage_tables(
        tr_j,
        transfer["measurements"],
        tr_meta,
        tr_groups,
        n_slope_bins=int(cov_cfg["n_slope_bins"]),
        min_subset_routes=int(cov_cfg["min_subset_routes"]),
    )
    matched = coverage_matched_draws(
        ref_meta,
        tr_j,
        tr_meta,
        tr_groups,
        n_draws=int(cov_cfg["n_coverage_matched_draws"]),
        seed=int(boot_cfg["seed"]),
        n_slope_bins=int(cov_cfg["n_slope_bins"]),
        min_subset_routes=int(cov_cfg["min_subset_routes"]),
    )
    subspace = parameter_subspace_transfer(ref_point, tr_point)

    read_only_points = []
    if not args.skip_read_only:
        for spec in config["runs"].get("read_only") or []:
            dump = resolve_under_root(root, str(config["output_dir"])) / "dumps" / f"cluster_local_r{int(spec['run'])}.root"
            if spec.get("cluster_dump"):
                dump = resolve_under_root(root, str(spec["cluster_dump"]))
            if not dump.is_file():
                read_only_points.append(
                    {
                        "run": int(spec["run"]),
                        "role": spec.get("role"),
                        "skipped": True,
                        "reason": f"cluster dump missing: {dump}",
                        "participates_in_method_selection": False,
                    }
                )
                continue
            loaded = load_run_measurements(
                spec,
                root=root,
                cluster_dump=dump,
                representative_station=station_id,
            )
            if not loaded["measurements"]:
                read_only_points.append(
                    {
                        "run": int(spec["run"]),
                        "role": spec.get("role"),
                        "skipped": True,
                        "reason": "no IFT cluster-local residuals",
                        "participates_in_method_selection": False,
                    }
                )
                continue
            jacobian = build_jacobian(loaded["measurements"], station_id=station_id, steps=steps)
            point = _run_point(loaded, jacobian, station_id, steps)
            point["subspace_vs_14973"] = parameter_subspace_transfer(ref_point, point)
            point["participates_in_method_selection"] = False
            read_only_points.append(point)

    decision = decide_next_stage(
        reference_point=ref_point,
        bootstrap=ref_boot,
        coverage=ref_coverage,
        transfer_point=tr_point,
        transfer_bootstrap=tr_boot,
        coverage_matched=matched,
        proxy_ry=proxy_ry,
    )

    _write_json(
        output / "bootstrap_stability_report.json",
        {
            **state,
            "schema_version": SCHEMA_VERSION,
            "created_utc": created,
            "run": 14973,
            "entry_57_point": {
                "station_dx_vs_C_dx": float(entry57["observed_dx_cdx_abs_cosine"]),
                "station_ry_vs_C_dx": float(entry57["observed_ry_cdx_abs_cosine"]),
                "leakage_rank": int(entry57["observed_leakage_rank"]),
            },
            "recomputed_point": ref_point,
            "reproduces_entry_57_ry_cdx": abs(
                float(ref_point["station_ry_vs_C_dx"]) - float(entry57["observed_ry_cdx_abs_cosine"])
            )
            < 1.0e-6,
            **ref_boot,
        },
    )
    _write_json(
        output / "coverage_dependence_report.json",
        {
            **state,
            "schema_version": SCHEMA_VERSION,
            "created_utc": created,
            "run": 14973,
            "transfer_run": 14974,
            "reference_coverage": ref_coverage,
            "transfer_coverage": {
                "run": 14974,
                **tr_coverage,
            },
            "coverage_matched_14974_to_14973_slope_histogram": matched,
        },
    )
    _write_json(
        output / "cross_run_cluster_local_jacobian_report.json",
        {
            **state,
            "schema_version": SCHEMA_VERSION,
            "created_utc": created,
            "identical_construction": {
                "residual": "r_u = u_cluster - u_track^unbiased on SiDetectorElement",
                "unbiased_method": "leave_one_station_out_cluster_globals_projected_to_detector_surface",
                "fd_steps": steps,
                "frozen_v2_routes": True,
                "no_retuning": True,
            },
            "reference_14973": ref_point,
            "transfer_14974": tr_point,
            "transfer_14974_bootstrap": tr_boot,
            "read_only_runs": read_only_points,
        },
    )
    _write_json(
        output / "leakage_subspace_transfer_report.json",
        {
            **state,
            "schema_version": SCHEMA_VERSION,
            "created_utc": created,
            "module_proxy_ry_cdx_abs_cosine": proxy_ry,
            "reference_14973": {
                "abs_cosines": {
                    "station_dx_vs_C_dx": ref_point["station_dx_vs_C_dx"],
                    "station_ry_vs_C_dx": ref_point["station_ry_vs_C_dx"],
                },
                "singular_values": ref_point["singular_values"],
                "singular_value_ratios": ref_point["singular_value_ratios"],
                "rank": ref_point["rank"],
            },
            "transfer_14974": {
                "abs_cosines": {
                    "station_dx_vs_C_dx": tr_point["station_dx_vs_C_dx"],
                    "station_ry_vs_C_dx": tr_point["station_ry_vs_C_dx"],
                },
                "singular_values": tr_point["singular_values"],
                "singular_value_ratios": tr_point["singular_value_ratios"],
                "rank": tr_point["rank"],
            },
            "parameter_space_principal_angles": subspace,
            "read_only_runs": [
                {
                    "run": row.get("run"),
                    "role": row.get("role"),
                    "skipped": row.get("skipped", False),
                    "station_ry_vs_C_dx": row.get("station_ry_vs_C_dx"),
                    "station_dx_vs_C_dx": row.get("station_dx_vs_C_dx"),
                    "rank": row.get("rank"),
                    "singular_value_ratios": row.get("singular_value_ratios"),
                    "subspace_vs_14973": row.get("subspace_vs_14973"),
                    "participates_in_method_selection": False,
                }
                for row in read_only_points
            ],
        },
    )
    _write_json(
        output / "next_stage_decision.json",
        {
            **state,
            "schema_version": SCHEMA_VERSION,
            "created_utc": created,
            **decision,
        },
    )
    print(
        json.dumps(
            {
                "output_dir": str(output),
                "recomputed_14973_ry_cdx": ref_point["station_ry_vs_C_dx"],
                "bootstrap_14973_ry_cdx": ref_boot["event_bootstrap"]["station_ry_vs_C_dx"],
                "transfer_14974_ry_cdx": tr_point["station_ry_vs_C_dx"],
                "decision": decision["decision"],
                "answer": decision["answer"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
