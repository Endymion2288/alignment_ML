"""True cluster-local residual feasibility on frozen V2 selected routes.

Exports and uses FaserSCT_Cluster local coordinates on the real
SiDetectorElement surface.  Does not train a new model, reopen Station /
reduced Station / C_dx Mode, write official geometry, or solve a correction.

The residual is r_u = u_cluster - u_track^unbiased on that surface.
Tracklet intercepts at nominal layer z are never the final observable.
Software finite differences perturb the detector surface, not conditions.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from alignment.layer_hierarchy import IFT_STATION_ID
from alignment.module_level_residual_poc import (
    STATION_Z_MM,
    TrackletHit,
    _find_tracklet_row,
    _loo_fit_for_station,
    _route_endpoints,
    assert_no_alignment_payload as _assert_poc_payload,
    c_dx_layer_weight,
    cosine,
    frozen_a_from_report,
    json_ready,
    load_json,
    load_selected_routes,
    load_tracklet_hits,
    module_key,
    predict_state_at_z,
    select_smoke_modules,
)
from alignment.operating_protocol_v1_final_closure import (
    FROZEN_V2_CHECKPOINT_SHA256,
    OPERATING_MODE,
    RESIDUAL_DECREASE_LABEL,
    project_root,
    resolve_under_root,
)
from datasets.root_loader import EventTracklets
from models.track_fitter import GlobalTrackFit

SCHEMA_VERSION = "faser-true-cluster-local-residual-feasibility-v1"
DEFAULT_CONFIG_RELATIVE = Path("configs") / "true_cluster_local_residual_feasibility_v1.yaml"
DECISION_RESTORES = "true_cluster_local_observable_restores_remaining_identifiability"
DECISION_TOPOLOGY = "ry_cdx_track_topology_degeneracy"
DECISION_INCONCLUSIVE = "true_cluster_local_ry_cdx_inconclusive"
MEASUREMENT_SOURCE = "FaserSCT_Cluster.localPosition Trk::locX on SiDetectorElement"
UNBIASED_METHOD = "leave_one_station_out_cluster_globals_projected_to_detector_surface"
UNBIASED_METHOD_FALLBACK = "leave_one_station_out_tracklets_projected_to_detector_surface"
RESIDUAL_KIND = "true cluster-local unbiased residual"
GO_NO_GO_QUESTION = (
    "In true silicon local measurement space, can the remaining "
    "station ry ↔ C_dx degeneracy be lifted?"
)
LEAKAGE_RANK_RELATIVE_TOLERANCE = 1.0e-2
FORBIDDEN_PAYLOAD_KEYS = {
    "alignment_payload",
    "station_payload",
    "cdx_payload",
    "newton_delta",
    "conditions_payload",
}
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
    "software_fd_sensitivity_only",
    "residual_reduction_is_not_alignment_success",
    "implied_cdx_is_not_a_measurement",
)


def load_stage_config(path: str | Path | None = None) -> dict[str, Any]:
    source = Path(path).expanduser().resolve() if path is not None else (
        project_root() / DEFAULT_CONFIG_RELATIVE
    )
    with source.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"true-cluster-local config must be a mapping: {source}")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unexpected true-cluster-local schema: {source}")
    for key in CONFIG_MUST_BE_TRUE:
        if payload.get(key) is not True:
            raise ValueError(f"true-cluster-local config must set {key}=true")
    for key in CONFIG_MUST_BE_FALSE:
        if payload.get(key) is not False:
            raise ValueError(f"true-cluster-local config must set {key}=false")
    if payload.get("residual_decrease_label") != RESIDUAL_DECREASE_LABEL:
        raise ValueError("residual decrease must stay labeled DQ observable")
    if payload.get("real_data_operating_mode") != OPERATING_MODE:
        raise ValueError("stage must remain residual_dq_monitoring_only")
    if payload.get("frozen_v2_checkpoint_sha256") != FROZEN_V2_CHECKPOINT_SHA256:
        raise ValueError("stage must not replace the frozen V2 checkpoint SHA256")
    if payload.get("residual_kind") != RESIDUAL_KIND:
        raise ValueError("residual_kind must be the true cluster-local residual")
    config = dict(payload)
    config["config_path"] = str(source)
    return config


def assert_no_alignment_payload(payload: Mapping[str, Any]) -> None:
    _assert_poc_payload(payload)
    extra = FORBIDDEN_PAYLOAD_KEYS.intersection(payload)
    if extra:
        raise ValueError(f"true-cluster-local stage contains alignment payload keys {sorted(extra)}")


def common_operating_state() -> dict[str, Any]:
    return {
        "real_data_operating_mode": OPERATING_MODE,
        "geometry_write_allowed": False,
        "station_calibration_mode_available": False,
        "cdx_mode_allowed": False,
        "alignment_payload_from_self_nulling_residuals": False,
        "alignment_correction": False,
        "frozen_v2_checkpoint_sha256": FROZEN_V2_CHECKPOINT_SHA256,
        "residual_decrease_label": RESIDUAL_DECREASE_LABEL,
        "residual_kind": RESIDUAL_KIND,
        "uses_nominal_layer_z_tracklet_intercept_as_residual": False,
        "uses_true_cluster_local_u": True,
        "uses_true_detector_surface": True,
    }


def write_event_list(routes: Sequence[Mapping[str, Any]], path: str | Path) -> dict[str, Any]:
    pairs = sorted({(int(row["run_id"]), int(row["event_id"])) for row in routes})
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "# run event\n" + "".join(f"{run} {event}\n" for run, event in pairs),
        encoding="utf-8",
    )
    return {
        "event_list": str(destination),
        "n_selected_routes": int(len(routes)),
        "n_unique_events": int(len(pairs)),
    }


def rotation_about_y_matching_se3(angle_rad: float) -> np.ndarray:
    cosine_a = math.cos(float(angle_rad))
    sine_a = math.sin(float(angle_rad))
    return np.asarray(
        [[cosine_a, 0.0, -sine_a], [0.0, 1.0, 0.0], [sine_a, 0.0, cosine_a]],
        dtype=np.float64,
    )


def rotate_about_y(
    vector: Sequence[float],
    angle_rad: float,
    origin: Sequence[float] | None = None,
) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64).reshape(3)
    matrix = rotation_about_y_matching_se3(angle_rad)
    if origin is None:
        return matrix @ value
    pivot = np.asarray(origin, dtype=np.float64).reshape(3)
    return pivot + matrix @ (value - pivot)


def _unit(vector: Sequence[float]) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(value))
    if norm <= 0.0:
        raise ValueError("cannot normalize a zero vector")
    return value / norm


@dataclass(frozen=True)
class DetectorSurface:
    center_mm: np.ndarray
    phi_axis: np.ndarray
    eta_axis: np.ndarray
    normal: np.ndarray
    sin_stereo: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "center_mm", np.asarray(self.center_mm, dtype=np.float64).reshape(3))
        object.__setattr__(self, "phi_axis", _unit(self.phi_axis))
        object.__setattr__(self, "eta_axis", _unit(self.eta_axis))
        object.__setattr__(self, "normal", _unit(self.normal))


def local_uv(point_mm: Sequence[float], surface: DetectorSurface) -> tuple[float, float]:
    delta = np.asarray(point_mm, dtype=np.float64).reshape(3) - surface.center_mm
    return float(np.dot(delta, surface.phi_axis)), float(np.dot(delta, surface.eta_axis))


def intersect_line_plane(
    origin_mm: Sequence[float],
    direction: Sequence[float],
    surface: DetectorSurface,
) -> np.ndarray | None:
    origin = np.asarray(origin_mm, dtype=np.float64).reshape(3)
    direction_v = np.asarray(direction, dtype=np.float64).reshape(3)
    denom = float(np.dot(direction_v, surface.normal))
    if abs(denom) <= 1.0e-14:
        return None
    scale = float(np.dot(surface.center_mm - origin, surface.normal) / denom)
    return origin + scale * direction_v


def predicted_local_u(
    origin_mm: Sequence[float],
    direction: Sequence[float],
    surface: DetectorSurface,
) -> float:
    point = intersect_line_plane(origin_mm, direction, surface)
    if point is None:
        return float("nan")
    return local_uv(point, surface)[0]


def perturb_surface(
    surface: DetectorSurface,
    *,
    station_id: int,
    layer_id: int,
    dx_mm: float = 0.0,
    ry_rad: float = 0.0,
    c_dx_mm: float = 0.0,
    c_dx_station: int = IFT_STATION_ID,
) -> DetectorSurface:
    center = np.array(surface.center_mm, dtype=np.float64)
    phi = np.array(surface.phi_axis, dtype=np.float64)
    eta = np.array(surface.eta_axis, dtype=np.float64)
    normal = np.array(surface.normal, dtype=np.float64)
    if abs(float(dx_mm)) > 0.0:
        center = center + np.asarray([float(dx_mm), 0.0, 0.0], dtype=np.float64)
    if int(station_id) == int(c_dx_station) and abs(float(c_dx_mm)) > 0.0:
        center = center + np.asarray(
            [float(c_dx_mm) * c_dx_layer_weight(layer_id), 0.0, 0.0],
            dtype=np.float64,
        )
    if abs(float(ry_rad)) > 0.0:
        origin = np.asarray([0.0, 0.0, STATION_Z_MM[int(station_id)]], dtype=np.float64)
        center = rotate_about_y(center, ry_rad, origin)
        phi = rotate_about_y(phi, ry_rad)
        eta = rotate_about_y(eta, ry_rad)
        normal = rotate_about_y(normal, ry_rad)
    return DetectorSurface(
        center_mm=center,
        phi_axis=phi,
        eta_axis=eta,
        normal=normal,
        sin_stereo=float(surface.sin_stereo),
    )


def fit_line_3d(points_mm: Sequence[Sequence[float]]) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points_mm, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] < 2:
        raise ValueError("3D line fit needs at least two xyz points")
    centroid = points.mean(axis=0)
    centered = points - centroid
    if float(np.max(np.ptp(points, axis=0))) <= 1.0e-9:
        raise ValueError("3D line fit needs spatially distinct points")
    _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
    direction = np.asarray(vh[0], dtype=np.float64)
    if direction[2] < 0.0:
        direction = -direction
    return centroid, _unit(direction)


def track_line_from_fit(fit: GlobalTrackFit) -> tuple[np.ndarray, np.ndarray]:
    x0, y0, tx, ty = (float(value) for value in fit.parameters)
    origin = np.asarray([x0, y0, float(fit.z_reference_mm)], dtype=np.float64)
    return origin, np.asarray([tx, ty, 1.0], dtype=np.float64)


@dataclass(frozen=True)
class ClusterLocalHit:
    run_id: int
    event_id: int
    cluster_identifier: int
    module_identifier: int
    station_id: int
    layer_id: int
    phi_module: int
    eta_module: int
    side: int
    local_u_mm: float
    local_v_mm: float
    local_u_var_mm2: float
    global_x_mm: float
    global_y_mm: float
    global_z_mm: float
    surface: DetectorSurface
    ontrack_local_u_mm: float
    ontrack_matched: bool
    detel_valid: bool


@dataclass(frozen=True)
class ClusterResidual:
    event_id: int
    route_index: int
    run_id: int
    tracklet_id: int
    station_id: int
    layer_id: int
    module_id: str
    cluster_id: int
    module_identifier: int
    phi_module: int
    eta_module: int
    side: int
    local_u_mm: float
    local_u_var_mm2: float
    sigma_u_mm: float
    predicted_u_mm: float
    unbiased_residual_u_mm: float
    global_x_mm: float
    global_y_mm: float
    global_z_mm: float
    surface: DetectorSurface
    track_origin_mm: np.ndarray
    track_direction: np.ndarray
    unbiased_method: str
    measurement_source: str = MEASUREMENT_SOURCE
    residual_kind: str = RESIDUAL_KIND
    residual_label: str = RESIDUAL_DECREASE_LABEL
    uses_nominal_layer_z_tracklet_intercept_as_residual: bool = False


@dataclass
class SoftwarePayload:
    dx_mm: float = 0.0
    ry_rad: float = 0.0
    c_dx_mm: float = 0.0
    station_id: int = IFT_STATION_ID
    module_dx_mm: dict[str, float] = field(default_factory=dict)

    def copy(self) -> "SoftwarePayload":
        return SoftwarePayload(
            dx_mm=float(self.dx_mm),
            ry_rad=float(self.ry_rad),
            c_dx_mm=float(self.c_dx_mm),
            station_id=int(self.station_id),
            module_dx_mm={key: float(value) for key, value in self.module_dx_mm.items()},
        )


def load_cluster_local_hits(path: str | Path) -> dict[tuple[int, int, int], ClusterLocalHit]:
    import uproot

    source = Path(path).expanduser().resolve()
    with uproot.open(source) as root_file:
        tree_name = None
        for key in root_file.keys():
            name = str(key).split(";")[0]
            if name.endswith("cluster_local") or name == "cluster_local":
                tree_name = str(key)
                break
        if tree_name is None:
            raise ValueError(f"cluster_local tree is absent from {source}: {list(root_file.keys())}")
        arrays = root_file[tree_name].arrays(library="np")
    hits: dict[tuple[int, int, int], ClusterLocalHit] = {}
    n_rows = int(np.asarray(arrays["cluster_identifier"]).shape[0])
    for index in range(n_rows):
        if int(arrays["detel_valid"][index]) != 1:
            continue
        surface = DetectorSurface(
            center_mm=[
                float(arrays["surface_cx_mm"][index]),
                float(arrays["surface_cy_mm"][index]),
                float(arrays["surface_cz_mm"][index]),
            ],
            phi_axis=[
                float(arrays["surface_ux"][index]),
                float(arrays["surface_uy"][index]),
                float(arrays["surface_uz"][index]),
            ],
            eta_axis=[
                float(arrays["surface_vx"][index]),
                float(arrays["surface_vy"][index]),
                float(arrays["surface_vz"][index]),
            ],
            normal=[
                float(arrays["surface_nx"][index]),
                float(arrays["surface_ny"][index]),
                float(arrays["surface_nz"][index]),
            ],
            sin_stereo=float(arrays["sin_stereo"][index]),
        )
        hit = ClusterLocalHit(
            run_id=int(arrays["run"][index]),
            event_id=int(arrays["event"][index]),
            cluster_identifier=int(arrays["cluster_identifier"][index]),
            module_identifier=int(arrays["module_identifier"][index]),
            station_id=int(arrays["station"][index]),
            layer_id=int(arrays["layer"][index]),
            phi_module=int(arrays["phi_module"][index]),
            eta_module=int(arrays["eta_module"][index]),
            side=int(arrays["side"][index]),
            local_u_mm=float(arrays["local_u_mm"][index]),
            local_v_mm=float(arrays["local_v_mm"][index]),
            local_u_var_mm2=float(arrays["local_u_var_mm2"][index]),
            global_x_mm=float(arrays["global_x_mm"][index]),
            global_y_mm=float(arrays["global_y_mm"][index]),
            global_z_mm=float(arrays["global_z_mm"][index]),
            surface=surface,
            ontrack_local_u_mm=float(arrays["ontrack_local_u_mm"][index]),
            ontrack_matched=bool(int(arrays["ontrack_matched"][index])),
            detel_valid=True,
        )
        hits[(hit.run_id, hit.event_id, hit.cluster_identifier)] = hit
    return hits


def summarize_cluster_export(
    hits: Mapping[tuple[int, int, int], ClusterLocalHit],
    routes: Sequence[Mapping[str, Any]],
    tracklet_hits: Mapping[tuple[int, int], Sequence[TrackletHit]],
    *,
    dump_path: str | Path,
    event_list: Mapping[str, Any],
    edm: Mapping[str, Any],
) -> dict[str, Any]:
    wanted: list[tuple[int, int, int]] = []
    for route in routes:
        key = (int(route["run_id"]), int(route["event_id"]))
        for hit in tracklet_hits.get(key, ()):
            wanted.append((hit.run_id, hit.event_id, int(hit.cluster_identifier)))
    unique_wanted = set(wanted)
    matched = [key for key in unique_wanted if key in hits]
    missing = [key for key in unique_wanted if key not in hits]
    reconstruction = []
    ontrack_delta = []
    for key in matched:
        hit = hits[key]
        recon_u, _recon_v = local_uv(
            [hit.global_x_mm, hit.global_y_mm, hit.global_z_mm],
            hit.surface,
        )
        reconstruction.append(recon_u - hit.local_u_mm)
        if hit.ontrack_matched and math.isfinite(hit.ontrack_local_u_mm):
            ontrack_delta.append(hit.ontrack_local_u_mm - hit.local_u_mm)
    recon_arr = np.asarray(reconstruction, dtype=np.float64) if reconstruction else np.asarray([])
    ontrack_arr = np.asarray(ontrack_delta, dtype=np.float64) if ontrack_delta else np.asarray([])
    return {
        "dump_path": str(dump_path),
        "edm": dict(edm),
        "n_dumped_clusters_with_surface": int(len(hits)),
        "n_unique_events_dumped": int(len({(hit.run_id, hit.event_id) for hit in hits.values()})),
        "n_selected_route_cluster_ids": int(len(unique_wanted)),
        "n_selected_route_clusters_joined": int(len(matched)),
        "n_selected_route_clusters_missing": int(len(missing)),
        "join_complete": bool(unique_wanted and not missing),
        "join_key": "run + event + TrackletHit_cluster_identifier = FaserSCT_Cluster.identify().get_compact()",
        "measurement_source": MEASUREMENT_SOURCE,
        "not_using_tracklet_intercept_as_cluster_residual": True,
        "surface_reconstruction_u_minus_measured_mm": _array_stats(recon_arr),
        "surface_frame_consistent": bool(
            recon_arr.size > 0 and float(np.max(np.abs(recon_arr))) < 0.05
        ),
        "prd_vs_ontrack_local_u_mm": _array_stats(ontrack_arr),
        "n_ontrack_cross_checks": int(ontrack_arr.size),
        **dict(event_list),
    }


def _array_stats(values: np.ndarray) -> dict[str, Any]:
    if values.size == 0:
        return {"n": 0}
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"n": 0}
    return {
        "n": int(finite.size),
        "mean": float(np.mean(finite)),
        "rms": float(np.sqrt(np.mean(finite * finite))),
        "max_abs": float(np.max(np.abs(finite))),
    }


def build_cluster_residuals(
    routes: Sequence[Mapping[str, Any]],
    events: Mapping[tuple[int, int], EventTracklets],
    tracklet_hits: Mapping[tuple[int, int], Sequence[TrackletHit]],
    cluster_hits: Mapping[tuple[int, int, int], ClusterLocalHit],
    *,
    representative_station: int,
) -> list[ClusterResidual]:
    residuals: list[ClusterResidual] = []
    for route in routes:
        endpoints = _route_endpoints(route)
        if len(endpoints) < 3:
            continue
        stations = [int(item["station_id"]) for item in endpoints]
        if int(representative_station) not in stations:
            continue
        key = (int(route["run_id"]), int(route["event_id"]))
        event = events.get(key)
        event_hits = tracklet_hits.get(key)
        if event is None or event_hits is None:
            continue
        rows = [
            _find_tracklet_row(event, int(item["station_id"]), int(item["origin_tracklet_id"]))
            for item in endpoints
        ]
        target_items = [
            (item, row)
            for item, row in zip(endpoints, rows)
            if int(item["station_id"]) == int(representative_station)
        ]
        if not target_items:
            continue
        other_points = []
        for item in endpoints:
            if int(item["station_id"]) == int(representative_station):
                continue
            for hit in event_hits:
                if int(hit.tracklet_id) != int(item["origin_tracklet_id"]):
                    continue
                if int(hit.station_id) != int(item["station_id"]):
                    continue
                cluster = cluster_hits.get((hit.run_id, hit.event_id, int(hit.cluster_identifier)))
                if cluster is None:
                    continue
                other_points.append(
                    [cluster.global_x_mm, cluster.global_y_mm, cluster.global_z_mm]
                )
        origin: np.ndarray | None = None
        direction: np.ndarray | None = None
        method = UNBIASED_METHOD
        if len(other_points) >= 2:
            try:
                origin, direction = fit_line_3d(other_points)
            except ValueError:
                origin = None
        if origin is None:
            method = UNBIASED_METHOD_FALLBACK
            left_out_row = target_items[0][1]
            fit = _loo_fit_for_station(event, rows, left_out_row)
            origin, direction = track_line_from_fit(fit)
        for item, _row in target_items:
            matched = [
                hit
                for hit in event_hits
                if int(hit.tracklet_id) == int(item["origin_tracklet_id"])
                and int(hit.station_id) == int(item["station_id"])
            ]
            for hit in matched:
                cluster = cluster_hits.get((hit.run_id, hit.event_id, int(hit.cluster_identifier)))
                if cluster is None:
                    continue
                predicted = predicted_local_u(origin, direction, cluster.surface)
                if not math.isfinite(predicted):
                    continue
                variance = float(cluster.local_u_var_mm2)
                if not math.isfinite(variance) or variance <= 0.0:
                    variance = 1.0e-4
                residuals.append(
                    ClusterResidual(
                        event_id=int(route["event_id"]),
                        route_index=int(route.get("route_index", 0)),
                        run_id=int(route["run_id"]),
                        tracklet_id=int(item["origin_tracklet_id"]),
                        station_id=int(cluster.station_id),
                        layer_id=int(cluster.layer_id),
                        module_id=module_key(
                            cluster.station_id,
                            cluster.layer_id,
                            cluster.eta_module,
                            cluster.phi_module,
                        ),
                        cluster_id=int(cluster.cluster_identifier),
                        module_identifier=int(cluster.module_identifier),
                        phi_module=int(cluster.phi_module),
                        eta_module=int(cluster.eta_module),
                        side=int(cluster.side),
                        local_u_mm=float(cluster.local_u_mm),
                        local_u_var_mm2=variance,
                        sigma_u_mm=float(math.sqrt(variance)),
                        predicted_u_mm=float(predicted),
                        unbiased_residual_u_mm=float(cluster.local_u_mm - predicted),
                        global_x_mm=float(cluster.global_x_mm),
                        global_y_mm=float(cluster.global_y_mm),
                        global_z_mm=float(cluster.global_z_mm),
                        surface=cluster.surface,
                        track_origin_mm=np.asarray(origin, dtype=np.float64),
                        track_direction=np.asarray(direction, dtype=np.float64),
                        unbiased_method=method,
                    )
                )
    return residuals


def residual_u(measurement: ClusterResidual, payload: SoftwarePayload | None = None) -> float:
    if payload is None:
        return float(measurement.unbiased_residual_u_mm)
    extra_dx = float(payload.module_dx_mm.get(measurement.module_id, 0.0))
    surface = perturb_surface(
        measurement.surface,
        station_id=measurement.station_id,
        layer_id=measurement.layer_id,
        dx_mm=float(payload.dx_mm) + extra_dx,
        ry_rad=float(payload.ry_rad),
        c_dx_mm=float(payload.c_dx_mm),
        c_dx_station=int(payload.station_id),
    )
    predicted = predicted_local_u(
        measurement.track_origin_mm,
        measurement.track_direction,
        surface,
    )
    return float(measurement.local_u_mm - predicted)


def residual_vector(measurements: Sequence[ClusterResidual], payload: SoftwarePayload | None = None) -> np.ndarray:
    return np.asarray([residual_u(item, payload) for item in measurements], dtype=np.float64)


def validate_residuals(measurements: Sequence[ClusterResidual]) -> dict[str, Any]:
    if not measurements:
        raise ValueError("no true cluster-local residuals")
    values = residual_vector(measurements)
    methods = Counter(item.unbiased_method for item in measurements)
    recon = []
    for item in measurements:
        recon.append(local_uv([item.global_x_mm, item.global_y_mm, item.global_z_mm], item.surface)[0] - item.local_u_mm)
    recon_arr = np.asarray(recon, dtype=np.float64)
    return {
        "n_measurements": int(len(measurements)),
        "n_routes": int(len({(item.run_id, item.event_id, item.route_index) for item in measurements})),
        "unbiased_method": UNBIASED_METHOD,
        "unbiased_method_counts": {key: int(value) for key, value in methods.items()},
        "measurement_source": MEASUREMENT_SOURCE,
        "residual_kind": RESIDUAL_KIND,
        "uses_nominal_layer_z_tracklet_intercept_as_residual": False,
        "biased_residual_used_as_alignment_observable": False,
        "official_acts_unbiased_residual_tool": False,
        "leave_one_station_out": True,
        "projected_to_real_detector_surface": True,
        "residual_label": RESIDUAL_DECREASE_LABEL,
        "finite": bool(np.isfinite(values).all()),
        "residual_u_mm": {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "rms": float(np.sqrt(np.mean(values * values))),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        },
        "surface_reconstruction_u_minus_measured_mm": _array_stats(recon_arr),
        "not_alignment_closure": True,
    }


def _central_column(
    measurements: Sequence[ClusterResidual],
    *,
    plus: SoftwarePayload,
    minus: SoftwarePayload,
    step: float,
) -> np.ndarray:
    if abs(float(step)) <= 0.0:
        raise ValueError("FD step must be non-zero")
    return (residual_vector(measurements, plus) - residual_vector(measurements, minus)) / (
        2.0 * float(step)
    )


def parameter_columns(
    measurements: Sequence[ClusterResidual],
    *,
    station_id: int,
    translation_step_mm: float,
    rotation_step_rad: float,
    c_dx_step_mm: float,
) -> tuple[list[str], np.ndarray]:
    names = ["station_dx", "station_ry", "C_dx"]
    columns = [
        _central_column(
            measurements,
            plus=SoftwarePayload(dx_mm=translation_step_mm, station_id=station_id),
            minus=SoftwarePayload(dx_mm=-translation_step_mm, station_id=station_id),
            step=translation_step_mm,
        ),
        _central_column(
            measurements,
            plus=SoftwarePayload(ry_rad=rotation_step_rad, station_id=station_id),
            minus=SoftwarePayload(ry_rad=-rotation_step_rad, station_id=station_id),
            step=rotation_step_rad,
        ),
        _central_column(
            measurements,
            plus=SoftwarePayload(c_dx_mm=c_dx_step_mm, station_id=station_id),
            minus=SoftwarePayload(c_dx_mm=-c_dx_step_mm, station_id=station_id),
            step=c_dx_step_mm,
        ),
    ]
    return names, np.column_stack(columns)


def run_fd_smoke(
    measurements: Sequence[ClusterResidual],
    smoke_modules: Sequence[str],
    *,
    station_id: int,
    translation_steps_mm: Sequence[float],
    rotation_steps_mrad: Sequence[float],
    c_dx_steps_mm: Sequence[float],
    linearity_max_relative_deviation: float,
    targeting_max_offmodule_fraction: float,
    l1_locality_max_fraction: float,
) -> dict[str, Any]:
    if len(translation_steps_mm) < 2 or len(rotation_steps_mrad) < 2 or len(c_dx_steps_mm) < 2:
        raise ValueError("FD smoke test needs two steps per family")
    t_small, t_large = (float(value) for value in translation_steps_mm[:2])
    r_small, r_large = (float(value) / 1000.0 for value in rotation_steps_mrad[:2])
    c_small, c_large = (float(value) for value in c_dx_steps_mm[:2])
    probes: list[dict[str, Any]] = []
    all_linear = True
    all_signed = True
    all_targeted = True

    def _record(name: str, column_a: np.ndarray, column_b: np.ndarray, *, step_a: float, step_b: float) -> None:
        nonlocal all_linear
        norm_a = float(np.linalg.norm(column_a))
        rel = float("nan")
        if norm_a > 0.0:
            rel = float(np.linalg.norm(column_a - column_b) / norm_a)
        elif float(np.linalg.norm(column_b)) == 0.0:
            rel = 0.0
        linear = bool(math.isfinite(rel) and rel <= float(linearity_max_relative_deviation))
        all_linear = all_linear and linear
        probes.append(
            {
                "name": name,
                "step_a": float(step_a),
                "step_b": float(step_b),
                "jacobian_norm_step_a": norm_a,
                "linearity_relative_deviation": rel,
                "linear": linear,
                "official_conditions_write": False,
            }
        )

    names_a, jac_a = parameter_columns(
        measurements,
        station_id=station_id,
        translation_step_mm=t_small,
        rotation_step_rad=r_small,
        c_dx_step_mm=c_small,
    )
    _names_b, jac_b = parameter_columns(
        measurements,
        station_id=station_id,
        translation_step_mm=t_large,
        rotation_step_rad=r_large,
        c_dx_step_mm=c_large,
    )
    for index, name in enumerate(names_a):
        step_a = t_small if name != "station_ry" else r_small
        step_b = t_large if name != "station_ry" else r_large
        if name == "C_dx":
            step_a, step_b = c_small, c_large
        _record(name, jac_a[:, index], jac_b[:, index], step_a=step_a, step_b=step_b)

    phi_x = np.asarray([item.surface.phi_axis[0] for item in measurements], dtype=np.float64)
    l0 = np.asarray([item.layer_id == 0 for item in measurements], dtype=bool)
    l1 = np.asarray([item.layer_id == 1 for item in measurements], dtype=bool)
    l2 = np.asarray([item.layer_id == 2 for item in measurements], dtype=bool)
    j_dx = jac_a[:, 0]
    j_cdx = jac_a[:, 2]
    stereo_ok = np.abs(phi_x) > 1.0e-6
    dx_sign = True
    if stereo_ok.any():
        dx_sign = bool(np.mean(j_dx[stereo_ok] / phi_x[stereo_ok]) > 0.0)
    c_dx_l0 = True
    c_dx_l2 = True
    if l0.any() and np.any(stereo_ok & l0):
        c_dx_l0 = bool(np.mean(j_cdx[stereo_ok & l0] / phi_x[stereo_ok & l0]) > 0.0)
    if l2.any() and np.any(stereo_ok & l2):
        c_dx_l2 = bool(np.mean(j_cdx[stereo_ok & l2] / phi_x[stereo_ok & l2]) < 0.0)
    all_signed = bool(dx_sign and c_dx_l0 and c_dx_l2)
    on_cdx = float(np.max(np.abs(j_cdx[l0 | l2]))) if (l0 | l2).any() else 0.0
    off_cdx = float(np.max(np.abs(j_cdx[l1]))) if l1.any() else 0.0
    c_dx_local = bool(on_cdx > 0.0 and off_cdx <= float(l1_locality_max_fraction) * on_cdx) if l1.any() else True

    for module_id in smoke_modules:
        mask = np.asarray([item.module_id == module_id for item in measurements], dtype=bool)
        plus = SoftwarePayload(module_dx_mm={module_id: t_small}, station_id=station_id)
        minus = SoftwarePayload(module_dx_mm={module_id: -t_small}, station_id=station_id)
        column = _central_column(measurements, plus=plus, minus=minus, step=t_small)
        targeted = True
        if mask.any() and (~mask).any():
            on = float(np.max(np.abs(column[mask])))
            off = float(np.max(np.abs(column[~mask])))
            targeted = bool(on > 0.0 and off <= float(targeting_max_offmodule_fraction) * on) if on > 0.0 or off > 0.0 else True
        all_targeted = all_targeted and targeted
        probes.append(
            {
                "name": f"{module_id}_dx",
                "target_ok": targeted,
                "official_conditions_write": False,
            }
        )

    return {
        "representative_station": int(station_id),
        "smoke_modules": list(smoke_modules),
        "n_measurements": int(len(measurements)),
        "probes": probes,
        "station_dx_sign_ok": dx_sign,
        "c_dx_layer0_positive_after_stereo": c_dx_l0,
        "c_dx_layer2_negative_after_stereo": c_dx_l2,
        "c_dx_layer1_local": c_dx_local,
        "geometry_payload_acts_on_target_module": all_targeted,
        "residual_derivative_signs_ok": all_signed,
        "jacobian_linear_region": all_linear,
        "official_conditions_write": False,
        "athena_fd_probes": False,
        "alignment_correction": False,
        "passed": bool(all_targeted and all_signed and all_linear and c_dx_local),
        "two_step_abs_cosines": {
            "step_a": _pair_cosines(jac_a),
            "step_b": _pair_cosines(jac_b),
        },
    }


def _pair_cosines(jacobian: np.ndarray) -> dict[str, float]:
    return {
        "station_dx_vs_C_dx": abs(cosine(jacobian[:, 0], jacobian[:, 2])),
        "station_ry_vs_C_dx": abs(cosine(jacobian[:, 1], jacobian[:, 2])),
        "station_dx_vs_station_ry": abs(cosine(jacobian[:, 0], jacobian[:, 1])),
    }


def leakage_spectrum(jacobian: np.ndarray) -> dict[str, Any]:
    values = np.asarray(jacobian, dtype=np.float64)
    singular = np.linalg.svd(values, compute_uv=False)
    rank = (
        int(np.sum(singular > LEAKAGE_RANK_RELATIVE_TOLERANCE * float(singular[0])))
        if singular.size
        else 0
    )
    weak = []
    if singular.size:
        _u, _s, vt = np.linalg.svd(values, full_matrices=False)
        names = ["station_dx", "station_ry", "C_dx"]
        for index, value in enumerate(singular):
            if value <= LEAKAGE_RANK_RELATIVE_TOLERANCE * float(singular[0]):
                direction = {names[col]: float(vt[index, col]) for col in range(min(3, vt.shape[1]))}
                weak.append(
                    {
                        "singular_value": float(value),
                        "fraction_of_largest": float(value / singular[0]) if singular[0] else None,
                        "leading_components": sorted(direction.items(), key=lambda item: -abs(item[1])),
                    }
                )
    return {
        "rank": rank,
        "singular_values": [float(value) for value in singular],
        "null_or_weak_modes": weak,
    }


def _subset_cosines(measurements: Sequence[ClusterResidual], station_id: int, steps: Mapping[str, float]) -> dict[str, float]:
    if len(measurements) < 6:
        return {"station_dx_vs_C_dx": float("nan"), "station_ry_vs_C_dx": float("nan")}
    _names, jacobian = parameter_columns(
        measurements,
        station_id=station_id,
        translation_step_mm=float(steps["translation_step_mm"]),
        rotation_step_rad=float(steps["rotation_step_rad"]),
        c_dx_step_mm=float(steps["c_dx_step_mm"]),
    )
    return _pair_cosines(jacobian)


def compare_leakage(
    measurements: Sequence[ClusterResidual],
    jacobian: np.ndarray,
    *,
    station_id: int,
    fd_report: Mapping[str, Any],
    previous_poc: Mapping[str, Any] | None,
    thresholds: Mapping[str, float],
    frozen_a: Mapping[str, Any] | None = None,
    steps: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    cluster_cos = _pair_cosines(jacobian)
    spectrum = leakage_spectrum(jacobian)
    previous_station = (previous_poc or {}).get("station_level_abs_cosine") or {}
    previous_module = (previous_poc or {}).get("module_level_abs_cosine") or {}
    station_dx = float(previous_station.get("station_dx_vs_C_dx", float("nan")))
    station_ry = float(previous_station.get("station_ry_vs_C_dx", float("nan")))
    module_dx = float(previous_module.get("station_dx_vs_C_dx", float("nan")))
    module_ry = float(previous_module.get("station_ry_vs_C_dx", float("nan")))
    cluster_dx = float(cluster_cos["station_dx_vs_C_dx"])
    cluster_ry = float(cluster_cos["station_ry_vs_C_dx"])
    drop_dx = (
        float(station_dx - cluster_dx)
        if math.isfinite(station_dx) and math.isfinite(cluster_dx)
        else float("nan")
    )
    drop_ry = (
        float(station_ry - cluster_ry)
        if math.isfinite(station_ry) and math.isfinite(cluster_ry)
        else float("nan")
    )
    drop_ry_from_proxy = (
        float(module_ry - cluster_ry)
        if math.isfinite(module_ry) and math.isfinite(cluster_ry)
        else float("nan")
    )
    high = float(thresholds["high_cosine"])
    restored = float(thresholds["restored_cosine"])
    topology = float(thresholds["topology_cosine"])
    min_drop = float(thresholds["minimum_drop"])
    stability_tol = float(thresholds["stability_max_abs_delta"])
    two_step = fd_report.get("two_step_abs_cosines") or {}
    step_a = two_step.get("step_a") or {}
    step_b = two_step.get("step_b") or {}
    two_step_delta = abs(
        float(step_a.get("station_ry_vs_C_dx", float("nan")))
        - float(step_b.get("station_ry_vs_C_dx", float("nan")))
    )
    subset_reports = {}
    if steps is not None and measurements:
        even = [item for item in measurements if int(item.event_id) % 2 == 0]
        odd = [item for item in measurements if int(item.event_id) % 2 == 1]
        subset_reports["even_event"] = _subset_cosines(even, station_id, steps)
        subset_reports["odd_event"] = _subset_cosines(odd, station_id, steps)
        module_ids = sorted({item.module_id for item in measurements if item.layer_id == 0})
        for module_id in module_ids[:4]:
            subset = [item for item in measurements if item.module_id == module_id or item.layer_id != 0]
            subset_reports[module_id] = _subset_cosines(subset, station_id, steps)
        tight = [item for item in measurements if abs(item.unbiased_residual_u_mm) <= 5.0]
        subset_reports["abs_r_u_le_5mm"] = _subset_cosines(tight, station_id, steps)
    subset_ry = [
        float(row["station_ry_vs_C_dx"])
        for row in subset_reports.values()
        if math.isfinite(float(row.get("station_ry_vs_C_dx", float("nan"))))
    ]
    subset_span = float(max(subset_ry) - min(subset_ry)) if subset_ry else float("nan")
    stable = bool(
        math.isfinite(two_step_delta)
        and two_step_delta <= stability_tol
        and (not math.isfinite(subset_span) or subset_span <= stability_tol)
    )
    dx_reproduced = bool(math.isfinite(cluster_dx) and cluster_dx <= restored)
    ry_separated = bool(math.isfinite(cluster_ry) and cluster_ry <= restored)
    ry_topology = bool(math.isfinite(cluster_ry) and cluster_ry > topology)
    ry_dropped = bool(math.isfinite(drop_ry) and drop_ry >= min_drop and ry_separated)
    return {
        "observation_spaces": {
            "station_level": (
                "previous PoC station-recompressed tracklet propagated to other-station z"
            ),
            "module_proxy": "previous PoC module-proxy unbiased residual_x at nominal layer z",
            "true_cluster_local": "unbiased r_u on the real SiDetectorElement surface",
        },
        "station_level_abs_cosine": {
            "station_dx_vs_C_dx": station_dx,
            "station_ry_vs_C_dx": station_ry,
        },
        "module_proxy_abs_cosine": {
            "station_dx_vs_C_dx": module_dx,
            "station_ry_vs_C_dx": module_ry,
        },
        "true_cluster_local_abs_cosine": cluster_cos,
        "cosine_drop_station_minus_cluster": {
            "station_dx_vs_C_dx": drop_dx,
            "station_ry_vs_C_dx": drop_ry,
        },
        "cosine_drop_module_proxy_minus_cluster": {
            "station_ry_vs_C_dx": drop_ry_from_proxy,
        },
        "dx_cdx_reproduction_check": {
            "previous_module_proxy": module_dx,
            "true_cluster_local": cluster_dx,
            "still_separated": dx_reproduced,
        },
        "dx_separated": dx_reproduced,
        "ry_separated": ry_separated,
        "ry_still_collinear": bool(math.isfinite(cluster_ry) and cluster_ry >= high),
        "ry_topology_collinear": ry_topology,
        "ry_cosine_dropped": ry_dropped,
        "cluster_leakage_subspace_rank": spectrum["rank"],
        "cluster_leakage_singular_values": spectrum["singular_values"],
        "cluster_null_or_weak_modes": spectrum["null_or_weak_modes"],
        "station_leakage_subspace_rank": (previous_poc or {}).get("station_leakage_subspace_rank"),
        "module_proxy_leakage_subspace_rank": (previous_poc or {}).get("module_leakage_subspace_rank"),
        "leakage_subspace_resolvable": spectrum["rank"] >= 3,
        "two_step_fd_ry_cdx_abs_delta": two_step_delta,
        "subset_abs_cosines": subset_reports,
        "subset_ry_cdx_span": subset_span,
        "stable_across_fd_and_subsets": stable,
        "thresholds": dict(thresholds),
        "frozen_A_native_per_mm_C_dx": None if frozen_a is None else dict(frozen_a),
        "implied_cdx_is_not_a_measurement": True,
        "residual_label": RESIDUAL_DECREASE_LABEL,
        "uses_nominal_layer_z_tracklet_intercept_as_residual": False,
    }


def decide_next_stage(
    leakage: Mapping[str, Any],
    fd_report: Mapping[str, Any],
) -> dict[str, Any]:
    ry_cos = float((leakage.get("true_cluster_local_abs_cosine") or {}).get("station_ry_vs_C_dx", float("nan")))
    ry_dropped = bool(leakage.get("ry_cosine_dropped"))
    ry_separated = bool(leakage.get("ry_separated"))
    ry_topology = bool(leakage.get("ry_topology_collinear"))
    resolvable = bool(leakage.get("leakage_subspace_resolvable"))
    stable = bool(leakage.get("stable_across_fd_and_subsets"))
    if not fd_report.get("passed"):
        label = DECISION_INCONCLUSIVE
        answer = "Not yet"
        go = False
        reason = "FD smoke test did not confirm a usable true cluster-local Jacobian."
        next_step = "fix_true_cluster_local_fd_before_any_further_ml_descent"
    elif ry_dropped and ry_separated and resolvable:
        label = DECISION_RESTORES
        answer = "Yes"
        go = True
        reason = (
            "True silicon local residuals drop |cos(station ry, C_dx)| enough "
            "to form a stable independent direction.  The remaining degeneracy "
            "was a module-proxy / intercept limitation, not track topology."
        )
        next_step = "full_module_level_identifiability_map_no_new_network"
    elif ry_topology and stable:
        label = DECISION_TOPOLOGY
        answer = "No"
        go = False
        reason = (
            "True detector-surface local residuals leave station ry and C_dx "
            "highly collinear.  The remaining degeneracy is the three-plane "
            "near-straight-track topology, not station-tracklet compression "
            "and not the previous module-proxy intercept."
        )
        next_step = "stop_ml_descent_prefer_survey_or_new_topology_or_extra_constraint"
    else:
        label = DECISION_INCONCLUSIVE
        answer = "Not yet"
        go = False
        reason = (
            f"|cos(station ry, C_dx)| dropped from the module-proxy ~0.995 to "
            f"{ry_cos:.3f} and the leakage subspace rank is 3, so the remaining "
            "degeneracy is not station-tracklet compression and not a locked "
            ">0.9 topology collinearity.  It is also not a stable independent "
            "direction across track subsets, so this does not yet restore "
            "alignment identifiability."
        )
        next_step = "do_not_sink_ml_until_ry_cdx_is_classified"
    return {
        "go_no_go_question": GO_NO_GO_QUESTION,
        "answer": answer,
        "decision": label,
        "reason": reason,
        "true_cluster_local_ry_cdx_abs_cosine": ry_cos,
        "ry_cdx_restored": bool(label == DECISION_RESTORES),
        "ry_cdx_track_topology_degeneracy": bool(label == DECISION_TOPOLOGY),
        "go_to_full_module_identifiability_map": go,
        "go_to_full_detector_module_level_alignment_basis_study": False,
        "still_no_new_network": True,
        "do_not_sink_ml_to_more_complex_cluster_architecture": not go,
        "geometry_write_allowed": False,
        "station_calibration_mode_available": False,
        "cdx_mode_allowed": False,
        "alignment_payload_from_self_nulling_residuals": False,
        "alignment_correction": False,
        "if_yes": (
            "Expand to a full-detector module-level identifiability map.  "
            "Do not train a new network."
        ),
        "if_no": (
            "Stop sinking ML for alignment identifiability.  Prefer external "
            "survey, a different track topology, or an extra physical constraint."
        ),
        "next_allowed_step": next_step,
        "stable_across_fd_and_subsets": stable,
        "leakage_subspace_resolvable": resolvable,
    }


def measurement_records(measurements: Sequence[ClusterResidual]) -> list[dict[str, Any]]:
    return [
        {
            "event_id": item.event_id,
            "route_index": item.route_index,
            "run_id": item.run_id,
            "tracklet_id": item.tracklet_id,
            "station_id": item.station_id,
            "layer_id": item.layer_id,
            "module_id": item.module_id,
            "cluster_id": item.cluster_id,
            "local_u_mm": item.local_u_mm,
            "predicted_u_mm": item.predicted_u_mm,
            "unbiased_residual_u_mm": item.unbiased_residual_u_mm,
            "sigma_u_mm": item.sigma_u_mm,
            "global_position_mm": [item.global_x_mm, item.global_y_mm, item.global_z_mm],
            "surface_center_mm": item.surface.center_mm.tolist(),
            "phi_axis": item.surface.phi_axis.tolist(),
            "unbiased_method": item.unbiased_method,
            "measurement_source": item.measurement_source,
            "residual_kind": item.residual_kind,
            "uses_nominal_layer_z_tracklet_intercept_as_residual": False,
        }
        for item in measurements
    ]


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()
