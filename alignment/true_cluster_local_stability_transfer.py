"""True cluster-local ry↔C_dx stability and cross-run transfer audit.

Keeps the entry-57 residual, surface, leave-one-station-out prediction, and
FD steps unchanged.  Does not train, write geometry, reopen Station / C_dx
Mode, invent a new cosine cut, or enter a full module identifiability map.
"""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from alignment.module_level_residual_poc import (
    index_events,
    load_selected_routes,
    load_tracklet_hits,
)
from alignment.operating_protocol_v1_final_closure import (
    FROZEN_V2_CHECKPOINT_SHA256,
    OPERATING_MODE,
    RESIDUAL_DECREASE_LABEL,
    project_root,
    resolve_under_root,
)
from alignment.true_cluster_local_residual import (
    ClusterResidual,
    LEAKAGE_RANK_RELATIVE_TOLERANCE,
    MEASUREMENT_SOURCE,
    RESIDUAL_KIND,
    UNBIASED_METHOD,
    assert_no_alignment_payload,
    build_cluster_residuals,
    common_operating_state,
    cosine,
    json_ready,
    leakage_spectrum,
    load_cluster_local_hits,
    load_json,
    parameter_columns,
    summarize_cluster_export,
)
from datasets.root_loader import load_events

SCHEMA_VERSION = "faser-true-cluster-local-ry-cdx-stability-transfer-v1"
DEFAULT_CONFIG_RELATIVE = Path("configs") / "true_cluster_local_ry_cdx_stability_transfer_v1.yaml"
DECISION_CANDIDATE = "true_cluster_local_observable_restores_remaining_identifiability_candidate"
DECISION_NOT_TRANSFERABLE = "cluster_local_identifiability_not_transferable"
GO_NO_GO_QUESTION = (
    "Is the entry-57 ry↔C_dx separation a repeatable, cross-run identifiability "
    "gain, or only a finite-sample / coverage fluctuation?"
)
CONFIG_MUST_BE_FALSE = (
    "geometry_write_allowed",
    "station_calibration_mode_available",
    "cdx_mode_allowed",
    "alignment_payload_from_self_nulling_residuals",
)
CONFIG_MUST_BE_TRUE = (
    "frozen",
    "do_not_retrain_v2",
    "do_not_modify_frozen_v2_checkpoint",
    "do_not_construct_station_calibration_mode",
    "do_not_construct_reduced_station_mode",
    "do_not_enter_cdx_mode",
    "do_not_run_newton",
    "do_not_solve_alignment_correction",
    "do_not_write_official_conditions",
    "do_not_use_tracklet_intercept_as_cluster_residual",
    "do_not_enter_full_module_identifiability_map",
    "do_not_invent_new_cosine_cut",
    "software_fd_sensitivity_only",
)


def load_audit_config(path: str | Path | None = None) -> dict[str, Any]:
    source = Path(path).expanduser().resolve() if path is not None else (
        project_root() / DEFAULT_CONFIG_RELATIVE
    )
    with source.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"stability-transfer config must be a mapping: {source}")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unexpected stability-transfer schema: {source}")
    for key in CONFIG_MUST_BE_TRUE:
        if payload.get(key) is not True:
            raise ValueError(f"stability-transfer config must set {key}=true")
    for key in CONFIG_MUST_BE_FALSE:
        if payload.get(key) is not False:
            raise ValueError(f"stability-transfer config must set {key}=false")
    if payload.get("real_data_operating_mode") != OPERATING_MODE:
        raise ValueError("audit must remain residual_dq_monitoring_only")
    if payload.get("frozen_v2_checkpoint_sha256") != FROZEN_V2_CHECKPOINT_SHA256:
        raise ValueError("audit must not replace the frozen V2 checkpoint SHA256")
    if payload.get("residual_kind") != RESIDUAL_KIND:
        raise ValueError("audit must keep the true cluster-local residual")
    if payload.get("residual_decrease_label") != RESIDUAL_DECREASE_LABEL:
        raise ValueError("residual decrease must stay labeled DQ observable")
    config = dict(payload)
    config["config_path"] = str(source)
    return config


def fd_steps(config: Mapping[str, Any]) -> dict[str, float]:
    fd = config["finite_difference"]
    return {
        "translation_step_mm": float(fd["translation_step_mm"]),
        "rotation_step_rad": float(fd["rotation_step_mrad"]) / 1000.0,
        "c_dx_step_mm": float(fd["c_dx_step_mm"]),
    }


def track_tx_ty(measurement: ClusterResidual) -> tuple[float, float]:
    direction = np.asarray(measurement.track_direction, dtype=np.float64)
    if abs(float(direction[2])) <= 1.0e-12:
        return float("nan"), float("nan")
    return float(direction[0] / direction[2]), float(direction[1] / direction[2])


def track_slope(measurement: ClusterResidual) -> float:
    tx, ty = track_tx_ty(measurement)
    if not math.isfinite(tx) or not math.isfinite(ty):
        return float("nan")
    return float(math.hypot(tx, ty))


def route_key(measurement: ClusterResidual) -> tuple[int, int, int]:
    return (int(measurement.run_id), int(measurement.event_id), int(measurement.route_index))


def group_indices(measurements: Sequence[ClusterResidual]) -> dict[tuple[int, int, int], list[int]]:
    groups: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for index, item in enumerate(measurements):
        groups[route_key(item)].append(int(index))
    return dict(groups)


def event_groups(measurements: Sequence[ClusterResidual]) -> dict[tuple[int, int], list[int]]:
    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, item in enumerate(measurements):
        groups[(int(item.run_id), int(item.event_id))].append(int(index))
    return dict(groups)


def build_jacobian(
    measurements: Sequence[ClusterResidual],
    *,
    station_id: int,
    steps: Mapping[str, float],
) -> np.ndarray:
    _names, jacobian = parameter_columns(
        measurements,
        station_id=int(station_id),
        translation_step_mm=float(steps["translation_step_mm"]),
        rotation_step_rad=float(steps["rotation_step_rad"]),
        c_dx_step_mm=float(steps["c_dx_step_mm"]),
    )
    return np.asarray(jacobian, dtype=np.float64)


def summarize_jacobian(jacobian: np.ndarray) -> dict[str, Any]:
    values = np.asarray(jacobian, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or values.shape[0] < 3:
        return {
            "n_rows": int(values.shape[0]) if values.ndim == 2 else 0,
            "station_dx_vs_C_dx": float("nan"),
            "station_ry_vs_C_dx": float("nan"),
            "station_dx_vs_station_ry": float("nan"),
            "rank": 0,
            "singular_values": [],
            "singular_value_ratios": {},
            "right_singular_vectors": None,
        }
    spectrum = leakage_spectrum(values)
    _u, singular, vt = np.linalg.svd(values, full_matrices=False)
    ratios = {}
    if singular.size and float(singular[0]) > 0.0:
        ratios = {
            "sigma2_over_sigma1": float(singular[1] / singular[0]) if singular.size > 1 else None,
            "sigma3_over_sigma1": float(singular[2] / singular[0]) if singular.size > 2 else None,
        }
    return {
        "n_rows": int(values.shape[0]),
        "station_dx_vs_C_dx": abs(cosine(values[:, 0], values[:, 2])),
        "station_ry_vs_C_dx": abs(cosine(values[:, 1], values[:, 2])),
        "station_dx_vs_station_ry": abs(cosine(values[:, 0], values[:, 1])),
        "rank": spectrum["rank"],
        "singular_values": [float(value) for value in spectrum["singular_values"]],
        "singular_value_ratios": ratios,
        "right_singular_vectors": [[float(item) for item in row] for row in vt],
        "column_norms": [float(np.linalg.norm(values[:, index])) for index in range(3)],
    }


def stacked_rows(jacobian: np.ndarray, index_groups: Sequence[Sequence[int]]) -> np.ndarray:
    rows = [np.asarray(jacobian[list(group)], dtype=np.float64) for group in index_groups if len(group)]
    if not rows:
        return np.zeros((0, jacobian.shape[1]), dtype=np.float64)
    return np.vstack(rows)


def percentile_interval(values: Sequence[float], coverage: float) -> tuple[float, float]:
    array = np.asarray([float(value) for value in values if math.isfinite(float(value))], dtype=np.float64)
    if array.size == 0:
        return float("nan"), float("nan")
    tail = 0.5 * (1.0 - float(coverage))
    return float(np.quantile(array, tail)), float(np.quantile(array, 1.0 - tail))


def summarize_distribution(values: Sequence[float], *, reference: float | None = None) -> dict[str, Any]:
    array = np.asarray([float(value) for value in values if math.isfinite(float(value))], dtype=np.float64)
    if array.size == 0:
        return {"n": 0}
    lo68, hi68 = percentile_interval(array, 0.68)
    lo95, hi95 = percentile_interval(array, 0.95)
    payload = {
        "n": int(array.size),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=1)) if array.size > 1 else 0.0,
        "interval_68": [lo68, hi68],
        "interval_95": [lo95, hi95],
        "width_68": float(hi68 - lo68),
        "width_95": float(hi95 - lo95),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }
    if reference is not None and math.isfinite(float(reference)):
        payload["reference"] = float(reference)
        payload["reference_inside_68"] = bool(lo68 <= float(reference) <= hi68)
        payload["reference_inside_95"] = bool(lo95 <= float(reference) <= hi95)
    return payload


def intervals_overlap(left: Sequence[float], right: Sequence[float]) -> bool:
    if len(left) != 2 or len(right) != 2:
        return False
    if not all(math.isfinite(float(value)) for value in list(left) + list(right)):
        return False
    return max(float(left[0]), float(right[0])) <= min(float(left[1]), float(right[1]))


def principal_angles(left: np.ndarray, right: np.ndarray) -> list[float]:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.size == 0 or b.size == 0:
        return []
    qa, _ = np.linalg.qr(a)
    qb, _ = np.linalg.qr(b)
    singular = np.linalg.svd(qa.T @ qb, compute_uv=False)
    return [float(math.acos(min(1.0, max(0.0, float(value))))) for value in singular]


def parameter_subspace_transfer(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    left_v = np.asarray(left.get("right_singular_vectors") or [], dtype=np.float64)
    right_v = np.asarray(right.get("right_singular_vectors") or [], dtype=np.float64)
    if left_v.shape != (3, 3) or right_v.shape != (3, 3):
        return {"comparable": False}
    # Rows of V^T are right singular vectors.  Weak {ry, C_dx} plane ≈ last two.
    weak_left = left_v[1:, :].T
    weak_right = right_v[1:, :].T
    full = principal_angles(left_v.T, right_v.T)
    weak = principal_angles(weak_left, weak_right)
    return {
        "comparable": True,
        "full_parameter_space_principal_angles_rad": full,
        "weak_two_mode_principal_angles_rad": weak,
        "max_weak_principal_angle_rad": max(weak) if weak else float("nan"),
        "max_weak_principal_angle_deg": (
            float(math.degrees(max(weak))) if weak else float("nan")
        ),
    }


def load_run_measurements(
    spec: Mapping[str, Any],
    *,
    root: Path,
    cluster_dump: str | Path,
    representative_station: int,
) -> dict[str, Any]:
    selected_path = resolve_under_root(root, str(spec["selected_routes"]))
    enhanced_path = resolve_under_root(root, str(spec["enhanced_tracklets"]))
    tracklets_path = resolve_under_root(root, str(spec["physical_tracklets"]))
    dump_path = resolve_under_root(root, str(cluster_dump))
    routes = load_selected_routes(selected_path)
    wanted = {(int(row["run_id"]), int(row["event_id"])) for row in routes}
    events = index_events(load_events(tracklets_path, require_mc_labels=False))
    events = {key: events[key] for key in wanted if key in events}
    hits = load_tracklet_hits(enhanced_path, wanted)
    clusters = load_cluster_local_hits(dump_path)
    export = summarize_cluster_export(
        clusters,
        routes,
        hits,
        dump_path=dump_path,
        event_list={"n_selected_routes": int(len(routes)), "n_unique_events": int(len(wanted))},
        edm={
            "cluster_container": "SCT_ClusterContainer",
            "cluster_class": "Tracker::FaserSCT_Cluster",
            "surface_class": "TrackerDD::SiDetectorElement",
        },
    )
    measurements = build_cluster_residuals(
        routes,
        events,
        hits,
        clusters,
        representative_station=int(representative_station),
    )
    return {
        "run": int(spec["run"]),
        "source_id": str(spec["source_id"]),
        "role": str(spec.get("role", "")),
        "participates_in_method_selection": bool(spec.get("participates_in_method_selection", False)),
        "n_selected_routes": int(len(routes)),
        "n_measurements": int(len(measurements)),
        "n_routes_with_residuals": int(len({route_key(item) for item in measurements})),
        "export": export,
        "measurements": measurements,
        "join_complete": bool(export.get("join_complete")),
        "unbiased_method": UNBIASED_METHOD,
        "measurement_source": MEASUREMENT_SOURCE,
    }


def route_metadata(measurements: Sequence[ClusterResidual]) -> dict[tuple[int, int, int], dict[str, Any]]:
    by_route: dict[tuple[int, int, int], list[ClusterResidual]] = defaultdict(list)
    for item in measurements:
        by_route[route_key(item)].append(item)
    meta = {}
    for key, items in by_route.items():
        slopes = [track_slope(item) for item in items]
        layers = tuple(sorted({int(item.layer_id) for item in items}))
        modules = tuple(sorted({item.module_id for item in items}))
        meta[key] = {
            "n_hits": int(len(items)),
            "mean_slope": float(np.nanmean(slopes)) if slopes else float("nan"),
            "layers": layers,
            "layer_topology": "L" + "+L".join(str(layer) for layer in layers) if layers else "none",
            "modules": modules,
            "n_stations_on_route": int(len({int(item.station_id) for item in items})),
            "event_id": int(items[0].event_id),
        }
    return meta


def attach_route_multiplicity(
    meta: dict[tuple[int, int, int], dict[str, Any]],
    routes: Sequence[Mapping[str, Any]],
) -> None:
    lookup = {
        (int(row["run_id"]), int(row["event_id"]), int(row.get("route_index", 0))): int(
            len(row.get("endpoint_provenance") or [])
        )
        for row in routes
    }
    for key, row in meta.items():
        row["endpoint_count"] = int(lookup.get(key, row["n_stations_on_route"]))


def bootstrap_event_summaries(
    jacobian: np.ndarray,
    measurements: Sequence[ClusterResidual],
    *,
    n_replicates: int,
    seed: int,
    min_routes: int,
) -> list[dict[str, Any]]:
    events = list(event_groups(measurements).items())
    if len(events) < int(min_routes):
        raise ValueError("not enough events for bootstrap")
    rng = np.random.default_rng(int(seed))
    rows = []
    for _ in range(int(n_replicates)):
        choice = rng.integers(0, len(events), size=len(events))
        groups = [events[int(index)][1] for index in choice]
        summary = summarize_jacobian(stacked_rows(jacobian, groups))
        rows.append(summary)
    return rows


def half_split_summaries(
    jacobian: np.ndarray,
    measurements: Sequence[ClusterResidual],
    *,
    n_splits: int,
    seed: int,
    min_routes: int,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    events = list(event_groups(measurements).items())
    if len(events) < 2 * int(min_routes):
        raise ValueError("not enough events for half-splits")
    rng = np.random.default_rng(int(seed) + 17)
    pairs = []
    half = len(events) // 2
    for _ in range(int(n_splits)):
        order = rng.permutation(len(events))
        left = [events[int(index)][1] for index in order[:half]]
        right = [events[int(index)][1] for index in order[half : 2 * half]]
        pairs.append(
            (
                summarize_jacobian(stacked_rows(jacobian, left)),
                summarize_jacobian(stacked_rows(jacobian, right)),
            )
        )
    return pairs


def coverage_tables(
    jacobian: np.ndarray,
    measurements: Sequence[ClusterResidual],
    meta: Mapping[tuple[int, int, int], Mapping[str, Any]],
    groups: Mapping[tuple[int, int, int], Sequence[int]],
    *,
    n_slope_bins: int,
    min_subset_routes: int,
) -> dict[str, Any]:
    keys = list(groups)
    slope_values = np.asarray([float(meta[key]["mean_slope"]) for key in keys], dtype=np.float64)
    finite = np.isfinite(slope_values)
    edges = None
    slope_bins: dict[str, list[tuple[int, int, int]]] = {}
    if int(np.sum(finite)) >= int(n_slope_bins) * int(min_subset_routes):
        edges = np.quantile(slope_values[finite], np.linspace(0.0, 1.0, int(n_slope_bins) + 1))
        edges[-1] = edges[-1] + 1.0e-12
        for index in range(int(n_slope_bins)):
            selected = [
                key
                for key, slope in zip(keys, slope_values)
                if math.isfinite(float(slope)) and edges[index] <= float(slope) < edges[index + 1]
            ]
            slope_bins[f"slope_bin_{index}"] = selected

    def _subset_report(label: str, selected_keys: Sequence[tuple[int, int, int]]) -> dict[str, Any]:
        if len(selected_keys) < int(min_subset_routes):
            return {
                "label": label,
                "n_routes": int(len(selected_keys)),
                "too_small": True,
            }
        summary = summarize_jacobian(stacked_rows(jacobian, [groups[key] for key in selected_keys]))
        summary.update({"label": label, "n_routes": int(len(selected_keys)), "too_small": False})
        return summary

    slope_reports = [_subset_report(name, selected) for name, selected in slope_bins.items()]
    topologies: dict[str, list[tuple[int, int, int]]] = defaultdict(list)
    modules: dict[str, list[tuple[int, int, int]]] = defaultdict(list)
    multiplicities: dict[str, list[tuple[int, int, int]]] = defaultdict(list)
    for key in keys:
        topologies[str(meta[key]["layer_topology"])].append(key)
        multiplicities[f"endpoints_{meta[key].get('endpoint_count', meta[key]['n_stations_on_route'])}"].append(key)
        for module_id in meta[key]["modules"]:
            modules[str(module_id)].append(key)

    leave_module = []
    for module_id, owned in sorted(modules.items(), key=lambda item: -len(item[1])):
        remainder = [key for key in keys if key not in set(owned)]
        report = _subset_report(f"leave_out_{module_id}", remainder)
        report["n_routes_with_module"] = int(len(owned))
        leave_module.append(report)

    leave_topology = []
    for name, owned in sorted(topologies.items(), key=lambda item: -len(item[1])):
        remainder = [key for key in keys if key not in set(owned)]
        report = _subset_report(f"leave_out_{name}", remainder)
        report["n_routes_in_group"] = int(len(owned))
        leave_topology.append(report)

    module_only = [
        _subset_report(f"only_{name}", owned)
        for name, owned in sorted(modules.items(), key=lambda item: -len(item[1]))[:8]
    ]
    topology_only = [
        _subset_report(name, owned)
        for name, owned in sorted(topologies.items(), key=lambda item: -len(item[1]))
    ]
    multiplicity_only = [
        _subset_report(name, owned)
        for name, owned in sorted(multiplicities.items())
    ]

    full = summarize_jacobian(jacobian)
    full_ry = float(full["station_ry_vs_C_dx"])
    dominance = []
    for report in leave_module + leave_topology:
        if report.get("too_small"):
            continue
        delta = abs(float(report["station_ry_vs_C_dx"]) - full_ry)
        dominance.append({"label": report["label"], "abs_delta_ry_cdx": delta, **report})
    dominance.sort(key=lambda item: -float(item["abs_delta_ry_cdx"]))
    return {
        "full_sample": full,
        "slope_bin_edges": None if edges is None else [float(value) for value in edges],
        "slope_bins": slope_reports,
        "module_only": module_only,
        "layer_topology_only": topology_only,
        "route_multiplicity_only": multiplicity_only,
        "leave_one_module_family_out": leave_module,
        "leave_one_topology_out": leave_topology,
        "largest_ry_cdx_shifts_when_leaving_a_group_out": dominance[:8],
        "n_modules_touched": int(len(modules)),
        "n_layer_topologies": int(len(topologies)),
        "module_route_counts": {key: int(len(value)) for key, value in modules.items()},
        "topology_route_counts": {key: int(len(value)) for key, value in topologies.items()},
        "multiplicity_route_counts": {key: int(len(value)) for key, value in multiplicities.items()},
    }


def coverage_matched_draws(
    reference_meta: Mapping[tuple[int, int, int], Mapping[str, Any]],
    transfer_jacobian: np.ndarray,
    transfer_meta: Mapping[tuple[int, int, int], Mapping[str, Any]],
    transfer_groups: Mapping[tuple[int, int, int], Sequence[int]],
    *,
    n_draws: int,
    seed: int,
    n_slope_bins: int,
    min_subset_routes: int,
) -> dict[str, Any]:
    ref_slopes = np.asarray(
        [float(row["mean_slope"]) for row in reference_meta.values() if math.isfinite(float(row["mean_slope"]))],
        dtype=np.float64,
    )
    if ref_slopes.size < int(n_slope_bins) * int(min_subset_routes):
        return {"possible": False, "reason": "reference slope sample too small"}
    edges = np.quantile(ref_slopes, np.linspace(0.0, 1.0, int(n_slope_bins) + 1))
    edges[-1] = edges[-1] + 1.0e-12
    ref_counts = []
    for index in range(int(n_slope_bins)):
        ref_counts.append(
            int(np.sum((ref_slopes >= edges[index]) & (ref_slopes < edges[index + 1])))
        )
    transfer_by_bin: list[list[tuple[int, int, int]]] = [[] for _ in range(int(n_slope_bins))]
    for key, row in transfer_meta.items():
        slope = float(row["mean_slope"])
        if not math.isfinite(slope):
            continue
        for index in range(int(n_slope_bins)):
            if edges[index] <= slope < edges[index + 1]:
                transfer_by_bin[index].append(key)
                break
    if any(len(bucket) < max(1, count) for bucket, count in zip(transfer_by_bin, ref_counts) if count):
        return {
            "possible": False,
            "reason": "transfer run cannot fill the reference slope histogram",
            "reference_bin_counts": ref_counts,
            "transfer_bin_counts": [int(len(bucket)) for bucket in transfer_by_bin],
            "slope_bin_edges": [float(value) for value in edges],
        }
    rng = np.random.default_rng(int(seed) + 101)
    summaries = []
    for _ in range(int(n_draws)):
        selected: list[tuple[int, int, int]] = []
        for bucket, count in zip(transfer_by_bin, ref_counts):
            if count <= 0:
                continue
            indexes = rng.choice(
                np.arange(len(bucket)),
                size=int(count),
                replace=len(bucket) < int(count),
            )
            selected.extend(bucket[int(index)] for index in indexes)
        summaries.append(
            summarize_jacobian(stacked_rows(transfer_jacobian, [transfer_groups[key] for key in selected]))
        )
    return {
        "possible": True,
        "n_draws": int(n_draws),
        "reference_bin_counts": ref_counts,
        "transfer_bin_counts": [int(len(bucket)) for bucket in transfer_by_bin],
        "slope_bin_edges": [float(value) for value in edges],
        "station_ry_vs_C_dx": summarize_distribution(
            [row["station_ry_vs_C_dx"] for row in summaries]
        ),
        "station_dx_vs_C_dx": summarize_distribution(
            [row["station_dx_vs_C_dx"] for row in summaries]
        ),
        "sigma3_over_sigma1": summarize_distribution(
            [
                float((row.get("singular_value_ratios") or {}).get("sigma3_over_sigma1") or float("nan"))
                for row in summaries
            ]
        ),
    }


def collect_metric(rows: Sequence[Mapping[str, Any]], key: str) -> list[float]:
    values = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            values.append(float(value))
    return values


def decide_next_stage(
    *,
    reference_point: Mapping[str, Any],
    bootstrap: Mapping[str, Any],
    coverage: Mapping[str, Any],
    transfer_point: Mapping[str, Any] | None,
    transfer_bootstrap: Mapping[str, Any] | None,
    coverage_matched: Mapping[str, Any] | None,
    proxy_ry: float,
) -> dict[str, Any]:
    ry_boot = bootstrap.get("event_bootstrap", {}).get("station_ry_vs_C_dx") or {}
    half = bootstrap.get("random_half_split", {}).get("station_ry_vs_C_dx") or {}
    point = float(reference_point.get("station_ry_vs_C_dx", float("nan")))
    median = float(ry_boot.get("median", float("nan")))
    width95 = float(ry_boot.get("width_95", float("nan")))
    hi95 = float((ry_boot.get("interval_95") or [float("nan"), float("nan")])[1])
    lo95 = float((ry_boot.get("interval_95") or [float("nan"), float("nan")])[0])
    claimed_drop = float(proxy_ry) - point if math.isfinite(point) else float("nan")
    sampling_fluctuation = bool(
        math.isfinite(width95) and math.isfinite(claimed_drop) and width95 >= abs(claimed_drop)
    )
    closer_to_proxy_than_point = bool(
        math.isfinite(hi95)
        and math.isfinite(point)
        and abs(float(proxy_ry) - hi95) < abs(hi95 - point)
    )
    half_width = float(half.get("width_95", float("nan")))
    shifts = coverage.get("largest_ry_cdx_shifts_when_leaving_a_group_out") or []
    max_shift = max((float(item.get("abs_delta_ry_cdx", 0.0)) for item in shifts), default=float("nan"))
    dominated = bool(
        math.isfinite(max_shift)
        and math.isfinite(claimed_drop)
        and abs(claimed_drop) > 0.0
        and max_shift >= 0.5 * abs(claimed_drop)
    )
    median_recovers_point = bool(
        math.isfinite(median)
        and math.isfinite(point)
        and math.isfinite(claimed_drop)
        and abs(claimed_drop) > 0.0
        and abs(median - point) <= 0.25 * abs(claimed_drop)
    )
    concentrated = bool(
        median_recovers_point
        and not sampling_fluctuation
        and not closer_to_proxy_than_point
    )
    transfer_inside_95 = False
    overlap_68 = False
    transfer_ry = float("nan")
    if transfer_point is not None:
        transfer_ry = float(transfer_point.get("station_ry_vs_C_dx", float("nan")))
        transfer_inside_95 = bool(
            math.isfinite(transfer_ry) and math.isfinite(lo95) and lo95 <= transfer_ry <= hi95
        )
    if transfer_bootstrap is not None:
        t_int = (transfer_bootstrap.get("event_bootstrap", {}).get("station_ry_vs_C_dx") or {}).get("interval_68")
        r_int = ry_boot.get("interval_68")
        overlap_68 = bool(t_int and r_int and intervals_overlap(r_int, t_int))
    matched_ok = False
    if coverage_matched and coverage_matched.get("possible"):
        matched_median = float((coverage_matched.get("station_ry_vs_C_dx") or {}).get("median", float("nan")))
        matched_ok = bool(math.isfinite(matched_median) and math.isfinite(lo95) and lo95 <= matched_median <= hi95)
    reproduced = bool(transfer_inside_95 or overlap_68)
    if concentrated and reproduced and not dominated:
        label = DECISION_CANDIDATE
        answer = "Yes"
        go = True
        reason = (
            "The 14973 bootstrap distribution of |cos(ry,C_dx)| stays concentrated "
            "around the entry-57 point, does not re-expand to the module-proxy "
            "collinear value, and 14974 independently falls in the same interval."
        )
        next_step = "full_module_level_identifiability_map_no_new_network"
    else:
        label = DECISION_NOT_TRANSFERABLE
        answer = "No"
        go = False
        parts = []
        if sampling_fluctuation:
            parts.append("14973 bootstrap 95% width is as large as the claimed drop from the module-proxy cosine")
        if closer_to_proxy_than_point:
            parts.append("the 14973 95% upper edge is closer to the module-proxy collinear value than to 0.531")
        if dominated:
            parts.append("leaving out one module family or layer topology shifts |cos(ry,C_dx)| by a large fraction of the claimed drop")
        if not reproduced:
            parts.append("14974 does not land inside the 14973 bootstrap interval")
        if not concentrated and not parts:
            parts.append("14973 resampling is not concentrated around the entry-57 separation")
        reason = (
            "Entry 57 opened an extra leakage dimension, but the current real "
            "track sample does not make that separation a portable alignment basis: "
            + "; ".join(parts)
            + "."
        )
        next_step = "keep_ml_frozen_no_cluster_architecture_descent"
    return {
        "go_no_go_question": GO_NO_GO_QUESTION,
        "answer": answer,
        "decision": label,
        "reason": reason,
        "next_allowed_step": next_step,
        "go_to_full_module_identifiability_map": go,
        "still_no_new_network": True,
        "do_not_sink_ml_to_more_complex_cluster_architecture": not go,
        "geometry_write_allowed": False,
        "station_calibration_mode_available": False,
        "cdx_mode_allowed": False,
        "alignment_correction": False,
        "do_not_invent_new_cosine_cut": True,
        "reference_ry_cdx": point,
        "bootstrap_ry_cdx_median": median,
        "bootstrap_ry_cdx_width_95": width95,
        "claimed_drop_from_module_proxy": claimed_drop,
        "sampling_fluctuation": sampling_fluctuation,
        "upper_95_closer_to_proxy_than_to_point": closer_to_proxy_than_point,
        "coverage_group_dominates": dominated,
        "largest_leave_one_group_abs_delta": max_shift,
        "half_split_width_95": half_width,
        "14973_bootstrap_concentrated": concentrated,
        "14973_bootstrap_median_recovers_point": median_recovers_point,
        "14974_inside_14973_bootstrap_95": transfer_inside_95,
        "14974_68_interval_overlaps_14973": overlap_68,
        "coverage_matched_14974_inside_14973_95": matched_ok,
        "14974_ry_cdx": transfer_ry,
        "read_only_runs_did_not_select_the_method": True,
    }


def common_audit_state() -> dict[str, Any]:
    state = common_operating_state()
    state.update(
        {
            "do_not_enter_full_module_identifiability_map": True,
            "do_not_invent_new_cosine_cut": True,
            "unbiased_method": UNBIASED_METHOD,
            "measurement_source": MEASUREMENT_SOURCE,
        }
    )
    return state
