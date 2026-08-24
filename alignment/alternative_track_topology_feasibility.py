"""Alternative track-topology identifiability feasibility.

Residual-blind inventory of frozen-V2 2024 r0022 routes, then Jacobian
stability on predeclared physics topologies.  Does not train, write
geometry, invent a cosine cut, or pick events from residuals.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from alignment.module_level_residual_poc import (
    STATION_Z_MM,
    decode_hit_pattern,
    index_events,
    load_selected_routes,
)
from alignment.operating_protocol_v1_final_closure import (
    FROZEN_V2_CHECKPOINT_SHA256,
    OPERATING_MODE,
    RESIDUAL_DECREASE_LABEL,
    project_root,
    resolve_under_root,
)
from alignment.true_cluster_local_residual import (
    MEASUREMENT_SOURCE,
    RESIDUAL_KIND,
    UNBIASED_METHOD,
    common_operating_state,
)
from alignment.true_cluster_local_stability_transfer import (
    bootstrap_event_summaries,
    build_jacobian,
    collect_metric,
    event_groups,
    fd_steps as _fd_steps_from_config,
    group_indices,
    half_split_summaries,
    intervals_overlap,
    load_run_measurements,
    parameter_subspace_transfer,
    route_key,
    stacked_rows,
    summarize_distribution,
    summarize_jacobian,
)
from datasets.root_loader import load_events

SCHEMA_VERSION = "faser-alternative-track-topology-identifiability-feasibility-v1"
DEFAULT_CONFIG_RELATIVE = Path("configs") / "alternative_track_topology_identifiability_feasibility_v1.yaml"
DECISION_PORTABLE = "predeclared_topology_restores_portable_identifiability_candidate"
DECISION_NONE = "no_portable_alternative_topology_in_current_r0022"
GO_NO_GO_QUESTION = (
    "Is there an independent, repeatable real track topology that supplies "
    "the angular lever arm missing from ordinary r0022 selected routes, so "
    "that ry/C_dx becomes portable alignment information?"
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
    "do_not_select_events_from_residual_or_cosine",
    "software_fd_sensitivity_only",
)


def load_feasibility_config(path: str | Path | None = None) -> dict[str, Any]:
    source = Path(path).expanduser().resolve() if path is not None else (
        project_root() / DEFAULT_CONFIG_RELATIVE
    )
    with source.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"topology-feasibility config must be a mapping: {source}")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unexpected topology-feasibility schema: {source}")
    for key in CONFIG_MUST_BE_TRUE:
        if payload.get(key) is not True:
            raise ValueError(f"topology-feasibility config must set {key}=true")
    for key in CONFIG_MUST_BE_FALSE:
        if payload.get(key) is not False:
            raise ValueError(f"topology-feasibility config must set {key}=false")
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
    return _fd_steps_from_config(config)


def station_z(station_id: int, physics: Mapping[str, Any]) -> float:
    table = physics.get("station_z_mm") or STATION_Z_MM
    return float(table[int(station_id)])


@dataclass(frozen=True)
class TopologyFeatures:
    run_id: int
    event_id: int
    route_index: int
    n_stations: int
    stations: tuple[int, ...]
    has_ift: bool
    is_complete_four_station: bool
    lever_arm_mm: float
    slope: float
    local_ift_slope: float
    incident_angle_rad: float
    ift_layers: tuple[int, ...]
    n_ift_modules: int
    jacobian_eligible: bool


def _slope_from_tx_ty(tx: float, ty: float) -> float:
    if not math.isfinite(tx) or not math.isfinite(ty):
        return float("nan")
    return float(math.hypot(tx, ty))


def _global_slope_from_points(points: Sequence[Sequence[float]]) -> float:
    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] < 2 or array.shape[1] != 3:
        return float("nan")
    order = np.argsort(array[:, 2])
    first = array[int(order[0])]
    last = array[int(order[-1])]
    dz = float(last[2] - first[2])
    if abs(dz) <= 1.0e-6:
        return float("nan")
    return _slope_from_tx_ty((last[0] - first[0]) / dz, (last[1] - first[1]) / dz)


def route_topology_features(
    route: Mapping[str, Any],
    event,
    *,
    physics: Mapping[str, Any],
) -> TopologyFeatures:
    provenance = list(route.get("endpoint_provenance") or [])
    stations = tuple(sorted({int(item["station_id"]) for item in provenance}))
    n_stations = int(len(stations))
    has_ift = 0 in stations
    four = bool(route.get("is_complete_four_station_route")) or stations == (0, 1, 2, 3)
    zs = [station_z(station, physics) for station in stations]
    lever = float(max(zs) - min(zs)) if zs else float("nan")
    ift_tx: list[float] = []
    ift_ty: list[float] = []
    points: list[list[float]] = []
    ift_layers: set[int] = set()
    ift_modules: set[tuple[int, int]] = set()
    if event is not None:
        for item in provenance:
            station_id = int(item["station_id"])
            tracklet_id = int(item["origin_tracklet_id"])
            matches = np.flatnonzero(
                (np.asarray(event.station_id) == station_id)
                & (np.asarray(event.tracklet_id) == tracklet_id)
            )
            if matches.size != 1:
                continue
            row = int(matches[0])
            tx = float(event.state[row, 2])
            ty = float(event.state[row, 3])
            points.append(
                [float(event.state[row, 0]), float(event.state[row, 1]), float(event.z_mm[row])]
            )
            if station_id == 0:
                ift_tx.append(tx)
                ift_ty.append(ty)
                pattern = decode_hit_pattern(int(event.hit_pattern[row]))
                for layer, _side in pattern:
                    ift_layers.add(int(layer))
                    ift_modules.add((int(layer), int(_side)))
    local_ift = (
        _slope_from_tx_ty(float(np.mean(ift_tx)), float(np.mean(ift_ty))) if ift_tx else float("nan")
    )
    slope = _global_slope_from_points(points)
    angle = float(math.atan(slope)) if math.isfinite(slope) else float("nan")
    return TopologyFeatures(
        run_id=int(route["run_id"]),
        event_id=int(route["event_id"]),
        route_index=int(route.get("route_index", 0)),
        n_stations=n_stations,
        stations=stations,
        has_ift=has_ift,
        is_complete_four_station=four,
        lever_arm_mm=lever,
        slope=slope,
        local_ift_slope=local_ift,
        incident_angle_rad=angle,
        ift_layers=tuple(sorted(ift_layers)),
        n_ift_modules=int(len(ift_modules)),
        jacobian_eligible=bool(n_stations >= 3 and has_ift),
    )


def matches_category(features: TopologyFeatures, spec: Mapping[str, Any]) -> bool:
    if spec.get("min_n_stations") is not None and features.n_stations < int(spec["min_n_stations"]):
        return False
    if spec.get("requires_ift") and not features.has_ift:
        return False
    if spec.get("requires_complete_four_station") and not features.is_complete_four_station:
        return False
    required = spec.get("requires_stations")
    if required and not set(int(value) for value in required).issubset(set(features.stations)):
        return False
    if spec.get("requires_full_ift_layers") and tuple(features.ift_layers) != (0, 1, 2):
        return False
    if spec.get("max_slope") is not None:
        if not math.isfinite(features.slope) or features.slope >= float(spec["max_slope"]):
            return False
    if spec.get("min_slope") is not None:
        if not math.isfinite(features.slope) or features.slope < float(spec["min_slope"]):
            return False
    return True


def assign_categories(
    features: TopologyFeatures,
    categories: Sequence[Mapping[str, Any]],
) -> list[str]:
    return [str(spec["id"]) for spec in categories if matches_category(features, spec)]


def features_payload(features: TopologyFeatures) -> dict[str, Any]:
    payload = asdict(features)
    payload["stations"] = list(features.stations)
    payload["ift_layers"] = list(features.ift_layers)
    return payload


def inventory_parent_tracklets(event_map: Mapping[tuple[int, int], Any], physics: Mapping[str, Any]) -> dict[str, Any]:
    ip_max = float(physics["ip_like_max_slope"])
    wide_min = float(physics["wide_angle_min_slope"])
    n_events = int(len(event_map))
    n_four = 0
    n_ift = 0
    n_ge3_stations = 0
    n_has_0_and_3 = 0
    n_ip_like_ift = 0
    n_wide_ift = 0
    n_full_ift_layers = 0
    n_ip_like_global = 0
    n_wide_global = 0
    n_global = 0
    local_slopes = []
    global_slopes = []
    for event in event_map.values():
        stations = set(int(value) for value in np.unique(event.station_id))
        if len(stations) >= 3:
            n_ge3_stations += 1
        if stations >= {0, 1, 2, 3}:
            n_four += 1
        if 0 in stations and 3 in stations:
            n_has_0_and_3 += 1
        by_station = {
            int(station): np.flatnonzero(np.asarray(event.station_id) == int(station))
            for station in stations
        }
        ift = by_station.get(0, np.asarray([], dtype=int))
        if ift.size:
            n_ift += 1
            tx = float(np.mean(event.state[ift, 2]))
            ty = float(np.mean(event.state[ift, 3]))
            local = _slope_from_tx_ty(tx, ty)
            if math.isfinite(local):
                local_slopes.append(local)
                if local < ip_max:
                    n_ip_like_ift += 1
                if local >= wide_min:
                    n_wide_ift += 1
            layers = set()
            for row in ift:
                for layer, _side in decode_hit_pattern(int(event.hit_pattern[row])):
                    layers.add(int(layer))
            if layers == {0, 1, 2}:
                n_full_ift_layers += 1
        if 0 in by_station and len(by_station) >= 2:
            points = []
            for station, rows in by_station.items():
                points.append(
                    [
                        float(np.mean(event.state[rows, 0])),
                        float(np.mean(event.state[rows, 1])),
                        float(np.mean(event.z_mm[rows])),
                    ]
                )
            global_slope = _global_slope_from_points(points)
            if math.isfinite(global_slope):
                n_global += 1
                global_slopes.append(global_slope)
                if global_slope < ip_max:
                    n_ip_like_global += 1
                if global_slope >= wide_min:
                    n_wide_global += 1
    local_arr = np.asarray(local_slopes, dtype=np.float64) if local_slopes else np.asarray([], dtype=np.float64)
    global_arr = np.asarray(global_slopes, dtype=np.float64) if global_slopes else np.asarray([], dtype=np.float64)
    return {
        "n_events_with_tracklets": n_events,
        "n_events_with_ift_tracklets": n_ift,
        "n_events_with_ge3_stations": n_ge3_stations,
        "n_events_with_four_stations": n_four,
        "n_events_with_stations_0_and_3": n_has_0_and_3,
        "n_events_with_global_slope": n_global,
        "n_local_ift_ip_like": n_ip_like_ift,
        "n_local_ift_wide_angle": n_wide_ift,
        "n_global_ip_like": n_ip_like_global,
        "n_global_wide_angle": n_wide_global,
        "n_full_ift_three_layers": n_full_ift_layers,
        "four_station_rate_per_tracklet_event": (n_four / n_events) if n_events else None,
        "global_wide_angle_rate": (n_wide_global / n_global) if n_global else None,
        "global_ip_like_rate": (n_ip_like_global / n_global) if n_global else None,
        "local_ift_slope": summarize_distribution(local_arr.tolist()) if local_arr.size else {"n": 0},
        "global_station_slope": summarize_distribution(global_arr.tolist()) if global_arr.size else {"n": 0},
        "category_slope_definition": "global_route_from_endpoint_tracklet_xyz",
    }


def inventory_selected_features(
    features: Sequence[TopologyFeatures],
    categories: Sequence[Mapping[str, Any]],
    *,
    n_processed_events: int,
) -> dict[str, Any]:
    n_routes = int(len(features))
    counts = Counter()
    jacobian_counts = Counter()
    for item in features:
        labels = assign_categories(item, categories)
        for label in labels:
            counts[label] += 1
            if item.jacobian_eligible:
                jacobian_counts[label] += 1
    n_ge3_ift = sum(1 for item in features if item.jacobian_eligible)
    n_four = sum(1 for item in features if item.is_complete_four_station)
    n_wide = sum(
        1
        for item in features
        if item.jacobian_eligible and math.isfinite(item.slope) and item.slope >= float(
            next(spec["min_slope"] for spec in categories if spec["id"] == "wide_incident_angle")
        )
    )
    slopes = [item.slope for item in features if item.jacobian_eligible and math.isfinite(item.slope)]
    denom = max(int(n_processed_events), 1)
    return {
        "n_selected_routes": n_routes,
        "n_jacobian_eligible": n_ge3_ift,
        "n_complete_four_station": n_four,
        "n_wide_incident_angle_eligible": n_wide,
        "processed_events": int(n_processed_events),
        "selected_route_rate": n_routes / denom,
        "jacobian_eligible_rate": n_ge3_ift / denom,
        "four_station_selected_rate": n_four / denom,
        "wide_angle_selected_rate": n_wide / denom,
        "category_counts": dict(counts),
        "jacobian_eligible_category_counts": dict(jacobian_counts),
        "ge3_ift_slope": summarize_distribution(slopes) if slopes else {"n": 0},
        "station_multiplicity": dict(Counter(item.n_stations for item in features)),
    }


def load_route_features(spec: Mapping[str, Any], *, root: Path, physics: Mapping[str, Any]) -> dict[str, Any]:
    selected_path = resolve_under_root(root, str(spec["selected_routes"]))
    tracklets_path = resolve_under_root(root, str(spec["physical_tracklets"]))
    routes = load_selected_routes(selected_path)
    wanted = {(int(row["run_id"]), int(row["event_id"])) for row in routes}
    events = index_events(load_events(tracklets_path, require_mc_labels=False))
    selected_features = [
        route_topology_features(route, events.get((int(route["run_id"]), int(route["event_id"]))), physics=physics)
        for route in routes
    ]
    parent = inventory_parent_tracklets(events, physics)
    return {
        "run": int(spec["run"]),
        "source_id": str(spec["source_id"]),
        "role": str(spec.get("role", "")),
        "corpus": str(spec.get("corpus", "")),
        "skip_events": int(spec.get("skip_events", 0)),
        "nevents": int(spec.get("nevents", 0)),
        "n_selected_routes": int(len(routes)),
        "selected_features": selected_features,
        "parent_tracklet_inventory": parent,
        "n_tracklet_events_loaded": int(len(events)),
        "n_selected_events_matched": int(sum(1 for key in wanted if key in events)),
    }


def occupancy_four_station_snapshot(path: str | Path | None, root: Path) -> dict[str, Any]:
    if not path:
        return {"available": False}
    source = resolve_under_root(root, str(path))
    if not source.is_file():
        return {"available": False, "path": str(source)}
    import json as json_lib

    payload = json_lib.loads(source.read_text(encoding="utf-8"))
    rows = []
    for run, block in (payload.get("runs") or {}).items():
        summary = (block or {}).get("occupancy_summary") or {}
        n_events = int(summary.get("n_events") or 0)
        n_four = int(summary.get("n_events_with_four_station_tracklets") or 0)
        rows.append(
            {
                "run": int(run),
                "n_events_in_occupancy_window": n_events,
                "n_events_with_four_station_tracklets": n_four,
                "four_station_rate_in_100_event_window": (n_four / n_events) if n_events else None,
                "residual_blind": bool((block or {}).get("residual_blind", True)),
            }
        )
    return {"available": True, "path": str(source), "windows": rows}


def survey_alternative_samples(config: Mapping[str, Any]) -> dict[str, Any]:
    survey = config["alternative_sample_survey"]
    rec_root = Path(str(survey["eos_rec_root"]))
    keywords = [str(item).lower() for item in survey.get("name_keywords") or []]
    years = {}
    keyword_hits = []
    for year in survey.get("years") or []:
        path = rec_root / str(year)
        tags = sorted(child.name for child in path.iterdir()) if path.is_dir() else []
        years[str(year)] = {"path": str(path), "exists": path.is_dir(), "tags": tags}
        for tag in tags:
            if any(key in tag.lower() for key in keywords):
                keyword_hits.append(f"{year}/{tag}")
    extras = {}
    for name in survey.get("extra_trees") or []:
        path = rec_root / str(name)
        extras[str(name)] = {
            "path": str(path),
            "exists": path.is_dir(),
            "tags": sorted(child.name for child in path.iterdir()) if path.is_dir() else [],
        }
        for tag in extras[str(name)]["tags"]:
            if any(key in tag.lower() for key in keywords):
                keyword_hits.append(f"{name}/{tag}")
    physics_2024 = rec_root / "2024" / "r0022"
    n_physics_runs = len(list(physics_2024.iterdir())) if physics_2024.is_dir() else 0
    return {
        "eos_rec_root": str(rec_root),
        "years": years,
        "extra_trees": extras,
        "keyword_directory_hits": keyword_hits,
        "n_2024_r0022_run_directories": n_physics_runs,
        "dedicated_cosmic_or_halo_stream_in_rec_tree": bool(keyword_hits),
        "testbeam_2021_present": bool((rec_root / "TestBeam2021").is_dir()),
        "independent_survey_available": bool(survey.get("independent_survey_available")),
        "mc_wide_angle_note": survey.get("mc_wide_angle_note"),
        "conclusion": (
            "No dedicated cosmic/halo/calib stream is named under /eos/experiment/faser/rec. "
            "Available alternatives are other collision years (2022 r0022, 2023 r0019-r0021, "
            "2025 r0023) and TestBeam2021; none are loaded in this audit."
        ),
    }


def category_keys(
    features: Sequence[TopologyFeatures],
    spec: Mapping[str, Any],
    *,
    jacobian_eligible_only: bool = True,
) -> list[tuple[int, int, int]]:
    keys = []
    for item in features:
        if jacobian_eligible_only and not item.jacobian_eligible:
            continue
        if matches_category(item, spec):
            keys.append((item.run_id, item.event_id, item.route_index))
    return keys


def bootstrap_pack(
    jacobian,
    measurements,
    *,
    n_event: int,
    n_half: int,
    seed: int,
    min_routes: int,
    reference_ry: float | None = None,
) -> dict[str, Any]:
    n_events = int(len(event_groups(measurements)))
    if n_events < int(min_routes):
        return {
            "n_events": n_events,
            "n_measurements": int(len(measurements)),
            "insufficient": True,
            "reason": f"n_events={n_events} < min_routes={min_routes}",
        }
    event_rows = bootstrap_event_summaries(
        jacobian,
        measurements,
        n_replicates=int(n_event),
        seed=int(seed),
        min_routes=int(min_routes),
    )
    pack = {
        "n_events": n_events,
        "n_measurements": int(len(measurements)),
        "insufficient": False,
        "event_bootstrap": {
            "n_replicates": int(n_event),
            "station_dx_vs_C_dx": summarize_distribution(collect_metric(event_rows, "station_dx_vs_C_dx")),
            "station_ry_vs_C_dx": summarize_distribution(
                collect_metric(event_rows, "station_ry_vs_C_dx"),
                reference=reference_ry,
            ),
            "rank": summarize_distribution(collect_metric(event_rows, "rank")),
            "sigma3_over_sigma1": summarize_distribution(
                [
                    float((row.get("singular_value_ratios") or {}).get("sigma3_over_sigma1") or float("nan"))
                    for row in event_rows
                ]
            ),
        },
    }
    if n_events >= 2 * int(min_routes):
        halves = half_split_summaries(
            jacobian,
            measurements,
            n_splits=int(n_half),
            seed=int(seed),
            min_routes=int(min_routes),
        )
        pack["random_half_split"] = {
            "n_splits": int(n_half),
            "station_ry_vs_C_dx": summarize_distribution(
                [float(left["station_ry_vs_C_dx"]) for left, right in halves]
                + [float(right["station_ry_vs_C_dx"]) for left, right in halves]
            ),
        }
    else:
        pack["random_half_split"] = {"skipped": True, "reason": "not enough events for half-splits"}
    return pack


def subset_measurements(measurements, keys: Sequence[tuple[int, int, int]]):
    wanted = set(keys)
    return [item for item in measurements if route_key(item) in wanted]


def concentrated_bootstrap(point_ry: float, boot: Mapping[str, Any], proxy_ry: float) -> dict[str, Any]:
    ry = (boot.get("event_bootstrap") or {}).get("station_ry_vs_C_dx") or {}
    median = float(ry.get("median", float("nan")))
    width95 = float(ry.get("width_95", float("nan")))
    hi95 = float((ry.get("interval_95") or [float("nan"), float("nan")])[1])
    claimed = float(proxy_ry) - float(point_ry) if math.isfinite(point_ry) else float("nan")
    sampling_fluctuation = bool(
        math.isfinite(width95) and math.isfinite(claimed) and width95 >= abs(claimed)
    )
    closer_to_proxy = bool(
        math.isfinite(hi95)
        and math.isfinite(point_ry)
        and abs(float(proxy_ry) - hi95) < abs(hi95 - point_ry)
    )
    median_recovers = bool(
        math.isfinite(median)
        and math.isfinite(point_ry)
        and math.isfinite(claimed)
        and abs(claimed) > 0.0
        and abs(median - point_ry) <= 0.25 * abs(claimed)
    )
    rank_boot = (boot.get("event_bootstrap") or {}).get("rank") or {}
    rank_stable = bool(float(rank_boot.get("median", 0.0) or 0.0) >= 3.0)
    concentrated = bool(median_recovers and not sampling_fluctuation and not closer_to_proxy and rank_stable)
    return {
        "median": median,
        "width_95": width95,
        "interval_95": ry.get("interval_95"),
        "claimed_drop_from_module_proxy": claimed,
        "sampling_fluctuation": sampling_fluctuation,
        "upper_95_closer_to_proxy_than_to_point": closer_to_proxy,
        "median_recovers_point": median_recovers,
        "rank_stable_at_3": rank_stable,
        "concentrated": concentrated,
    }


def transfer_ok(reference_boot: Mapping[str, Any], transfer_point: Mapping[str, Any], transfer_boot: Mapping[str, Any] | None) -> dict[str, Any]:
    ry = (reference_boot.get("event_bootstrap") or {}).get("station_ry_vs_C_dx") or {}
    lo95, hi95 = (ry.get("interval_95") or [float("nan"), float("nan")])[:2]
    transfer_ry = float(transfer_point.get("station_ry_vs_C_dx", float("nan")))
    inside = bool(math.isfinite(transfer_ry) and math.isfinite(float(lo95)) and float(lo95) <= transfer_ry <= float(hi95))
    overlap = False
    if transfer_boot and not transfer_boot.get("insufficient"):
        t_int = (transfer_boot.get("event_bootstrap") or {}).get("station_ry_vs_C_dx", {}).get("interval_68")
        r_int = ry.get("interval_68")
        overlap = bool(t_int and r_int and intervals_overlap(r_int, t_int))
    return {
        "transfer_ry_cdx": transfer_ry,
        "inside_reference_95": inside,
        "overlap_68": overlap,
        "reproduced": bool(inside or overlap),
    }


def evaluate_category_portability(
    *,
    spec: Mapping[str, Any],
    reference: Mapping[str, Any] | None,
    transfer: Mapping[str, Any] | None,
    proxy_ry: float,
    min_routes: int,
) -> dict[str, Any]:
    can_unlock = bool(spec.get("can_unlock_module_map"))
    if reference is None or transfer is None:
        return {
            "category": spec["id"],
            "can_unlock_module_map": can_unlock,
            "portable": False,
            "reason": "missing 14973 or 14974 Jacobian for this category",
        }
    if reference.get("insufficient") or transfer.get("insufficient"):
        return {
            "category": spec["id"],
            "can_unlock_module_map": can_unlock,
            "portable": False,
            "reason": "category is below the predeclared min_routes on 14973 or 14974",
            "n_routes_14973": reference.get("n_routes"),
            "n_routes_14974": transfer.get("n_routes"),
            "min_routes": int(min_routes),
        }
    conc = concentrated_bootstrap(
        float(reference["point"]["station_ry_vs_C_dx"]),
        reference.get("bootstrap") or {},
        proxy_ry,
    )
    xfer = transfer_ok(reference.get("bootstrap") or {}, transfer["point"], transfer.get("bootstrap"))
    rank_ok = int(reference["point"].get("rank") or 0) >= 3 and int(transfer["point"].get("rank") or 0) >= 3
    portable = bool(can_unlock and conc["concentrated"] and xfer["reproduced"] and rank_ok)
    reasons = []
    if not can_unlock:
        reasons.append("category is a baseline/control and is not allowed to unlock the module map")
    if not rank_ok:
        reasons.append("rank is not 3 on both 14973 and 14974")
    if not conc["concentrated"]:
        reasons.append("14973 bootstrap is not concentrated")
    if not xfer["reproduced"]:
        reasons.append("14974 does not reproduce the 14973 interval")
    return {
        "category": spec["id"],
        "role": spec.get("role"),
        "can_unlock_module_map": can_unlock,
        "portable": portable,
        "rank_3_on_both_calibration_runs": rank_ok,
        "bootstrap": conc,
        "transfer": xfer,
        "n_routes_14973": reference.get("n_routes"),
        "n_routes_14974": transfer.get("n_routes"),
        "reason": "portable predeclared topology" if portable else "; ".join(reasons),
    }


def decide_next_stage(
    *,
    evaluations: Sequence[Mapping[str, Any]],
    inventory_summary: Mapping[str, Any],
    alternative_survey: Mapping[str, Any],
) -> dict[str, Any]:
    portable = [row for row in evaluations if row.get("portable")]
    shallow = bool(inventory_summary.get("r0022_still_shallow_dominated"))
    if portable:
        winners = [str(row["category"]) for row in portable]
        return {
            "go_no_go_question": GO_NO_GO_QUESTION,
            "answer": "Yes",
            "decision": DECISION_PORTABLE,
            "portable_categories": winners,
            "go_to_full_module_identifiability_map": True,
            "still_no_new_network": True,
            "do_not_sink_ml_to_more_complex_cluster_architecture": False,
            "geometry_write_allowed": False,
            "station_calibration_mode_available": False,
            "cdx_mode_allowed": False,
            "alignment_correction": False,
            "do_not_invent_new_cosine_cut": True,
            "read_only_runs_did_not_select_the_method": True,
            "next_allowed_step": "full_module_level_identifiability_map_no_new_network",
            "reason": (
                "Predeclared topology "
                + ", ".join(winners)
                + " is rank-3, bootstrap-concentrated on 14973, and transfers to 14974."
            ),
            "evaluations": list(evaluations),
        }
    next_step = "keep_ml_frozen_seek_alternative_real_samples_and_external_survey"
    if shallow:
        reason = (
            "Ordinary 2024 r0022 selected routes remain shallow-track dominated. "
            "No predeclared high-lever-arm topology is both statistically sufficient "
            "and transferable. Do not stack more collision-like events. "
            "Next constraints are an independent external survey and a real non-collision "
            "sample (cosmic-like, beam-halo, test-beam, or another year/beam condition) "
            "that is not present as a named stream in the current rec tree"
            + (
                ""
                if alternative_survey.get("dedicated_cosmic_or_halo_stream_in_rec_tree")
                else "."
            )
        )
    else:
        reason = (
            "High-angle or long-lever categories exist in r0022 but do not form a "
            "bootstrap-stable, cross-run portable ry/C_dx basis."
        )
    return {
        "go_no_go_question": GO_NO_GO_QUESTION,
        "answer": "No",
        "decision": DECISION_NONE,
        "portable_categories": [],
        "go_to_full_module_identifiability_map": False,
        "still_no_new_network": True,
        "do_not_sink_ml_to_more_complex_cluster_architecture": True,
        "geometry_write_allowed": False,
        "station_calibration_mode_available": False,
        "cdx_mode_allowed": False,
        "alignment_correction": False,
        "do_not_invent_new_cosine_cut": True,
        "read_only_runs_did_not_select_the_method": True,
        "r0022_still_shallow_dominated": shallow,
        "independent_survey_available": bool(alternative_survey.get("independent_survey_available")),
        "next_allowed_step": next_step,
        "reason": reason,
        "evaluations": list(evaluations),
    }


def r0022_shallow_dominated(run_inventories: Sequence[Mapping[str, Any]], physics: Mapping[str, Any]) -> bool:
    n_eligible = 0
    n_wide = 0
    for row in run_inventories:
        ge3 = int((row.get("selected") or {}).get("n_jacobian_eligible") or 0)
        counts = (row.get("selected") or {}).get("jacobian_eligible_category_counts") or {}
        n_eligible += ge3
        n_wide += int(counts.get("wide_incident_angle") or 0)
    if n_eligible <= 0:
        return True
    return bool((n_wide / n_eligible) < 0.5)


def common_audit_state() -> dict[str, Any]:
    state = common_operating_state()
    state.update(
        {
            "do_not_enter_full_module_identifiability_map": True,
            "do_not_invent_new_cosine_cut": True,
            "do_not_select_events_from_residual_or_cosine": True,
            "unbiased_method": UNBIASED_METHOD,
            "measurement_source": MEASUREMENT_SOURCE,
            "schema_version": SCHEMA_VERSION,
        }
    )
    return state


def predeclared_category_report(config: Mapping[str, Any]) -> dict[str, Any]:
    physics = config["physics_scales"]
    return {
        **common_audit_state(),
        "frozen_before_jacobian": True,
        "selection_forbidden": [
            "unbiased_residual_u_mm",
            "station_ry_vs_C_dx",
            "station_dx_vs_C_dx",
            "cosine",
        ],
        "feature_source": "physical tracklet xyz (global route slope), hit_pattern layers, selected-route endpoint stations",
        "physics_scales": dict(physics),
        "categories": list(config["predeclared_categories"]),
        "note": (
            "Slope edges are stereo/10 and IP acceptance, not the entry-58 "
            "data-driven tertiles 0.000681 / 0.001544.  The slope itself is the "
            "spectrometer Δx/Δz from endpoint tracklet positions, not local SegmentFit tx/ty."
        ),
    }
