"""Module-level residual and identifiability proof-of-concept.

Uses frozen V2 selected routes and existing TrackletHit identifiers.
Does not train a new model, reopen Station / reduced Station / C_dx Mode,
write official geometry, or solve an alignment correction.

Measurements are tracklet intercepts at nominal layer z, assigned to the
real module/cluster identifiers from the enhanced ntuple.  Residuals are
leave-one-station-out predictions, never in-sample biased fit residuals.
Software finite differences are sensitivity probes, not conditions writes.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import yaml

from alignment.gauge_equivalence import IFT_LAYER_PITCH_MM
from alignment.layer_hierarchy import IFT_LAYER_IDS, IFT_STATION_ID
from alignment.operating_protocol_v1_final_closure import (
    FROZEN_V2_CHECKPOINT_SHA256,
    OPERATING_MODE,
    RESIDUAL_DECREASE_LABEL,
    project_root,
    resolve_under_root,
)
from datasets.root_loader import EventTracklets, load_events
from models.track_fitter import GlobalTrackFit, fit_global_straight_track

SCHEMA_VERSION = "faser-module-level-residual-identifiability-poc-v1"
DEFAULT_CONFIG_RELATIVE = Path("configs") / "module_level_residual_identifiability_poc_v1.yaml"
DECISION_RESTORES = "module_level_observable_restores_identifiability_candidate"
DECISION_TOPOLOGY = "track_topology_limited"
DECISION_MIXED = "partial_separation_ry_still_topology_limited"
SIX_DOF = ("dx", "dy", "dz", "rx", "ry", "rz")
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
    "software_fd_sensitivity_only",
    "residual_reduction_is_not_alignment_success",
    "implied_cdx_is_not_a_measurement",
    "do_not_claim_acts_cluster_residual",
)

STATION_Z_MM = {
    0: -1860.15,
    1: 47.4,
    2: 1237.4,
    3: 2427.4,
}
PHI_MODULE_Y_MM = (-48.0, -16.0, 16.0, 48.0)
ETA_MODULE_X_MM = {-1: -30.0, 1: 30.0}
STEREO_RAD = 0.020
STRIP_PITCH_MM = 0.080
MEASUREMENT_SOURCE = "tracklet_intercept_at_nominal_layer_z"
UNBIASED_METHOD = "leave_one_station_out_projected_to_layer"
RESIDUAL_KIND = "module-proxy unbiased residual"
GO_NO_GO_QUESTION = (
    "Does the module-level observable actually lift the current station "
    "dx/ry versus internal C_dx degeneracy, so that a full-detector "
    "module-level alignment basis study is warranted?"
)
MIXED_REASON = (
    "Measurement compression explains the translational degeneracy, but "
    "ry/C_dx remains limited by near-straight three-plane track topology."
)
LEAKAGE_RANK_RELATIVE_TOLERANCE = 1.0e-2


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, np.ndarray):
        return [json_ready(item) for item in value.tolist()]
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def load_poc_config(path: str | Path | None = None) -> dict[str, Any]:
    source = Path(path).expanduser().resolve() if path is not None else (
        project_root() / DEFAULT_CONFIG_RELATIVE
    )
    with source.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"module-level PoC config must be a mapping: {source}")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unexpected module-level PoC schema: {source}")
    for key in CONFIG_MUST_BE_TRUE:
        if payload.get(key) is not True:
            raise ValueError(f"module-level PoC config must set {key}=true")
    for key in CONFIG_MUST_BE_FALSE:
        if payload.get(key) is not False:
            raise ValueError(f"module-level PoC config must set {key}=false")
    if payload.get("residual_decrease_label") != RESIDUAL_DECREASE_LABEL:
        raise ValueError("residual decrease must stay labeled DQ observable")
    if payload.get("real_data_operating_mode") != OPERATING_MODE:
        raise ValueError("PoC must remain residual_dq_monitoring_only")
    if payload.get("frozen_v2_checkpoint_sha256") != FROZEN_V2_CHECKPOINT_SHA256:
        raise ValueError("PoC must not replace the frozen V2 checkpoint SHA256")
    config = dict(payload)
    config["config_path"] = str(source)
    return config


def assert_no_alignment_payload(payload: Mapping[str, Any]) -> None:
    if payload.get("geometry_write_allowed") is True:
        raise ValueError("module-level PoC must not allow geometry write")
    if payload.get("alignment_payload_from_self_nulling_residuals") is True:
        raise ValueError("module-level PoC must not emit a self-nulling payload")
    if payload.get("station_calibration_mode_available") is True:
        raise ValueError("module-level PoC must not reopen Station Mode")
    if payload.get("cdx_mode_allowed") is True:
        raise ValueError("module-level PoC must not reopen C_dx Mode")
    extra = FORBIDDEN_PAYLOAD_KEYS.intersection(payload)
    if extra:
        raise ValueError(f"module-level PoC contains alignment payload keys {sorted(extra)}")
    if payload.get("alignment_correction") not in (None, False, {}, []):
        raise ValueError("module-level PoC must not compute an alignment correction")


def layer_z_mm(station_id: int, layer_id: int) -> float:
    if int(layer_id) not in IFT_LAYER_IDS:
        raise ValueError(f"layer must be 0, 1 or 2: {layer_id}")
    return float(STATION_Z_MM[int(station_id)]) + (int(layer_id) - 1) * float(IFT_LAYER_PITCH_MM)


def module_center_mm(station_id: int, layer_id: int, eta_module: int, phi_module: int) -> np.ndarray:
    if int(eta_module) not in ETA_MODULE_X_MM:
        raise ValueError(f"eta_module must be ±1: {eta_module}")
    if int(phi_module) not in (0, 1, 2, 3):
        raise ValueError(f"phi_module must be 0..3: {phi_module}")
    return np.asarray(
        [
            float(ETA_MODULE_X_MM[int(eta_module)]),
            float(PHI_MODULE_Y_MM[int(phi_module)]),
            layer_z_mm(station_id, layer_id),
        ],
        dtype=np.float64,
    )


def module_key(station_id: int, layer_id: int, eta_module: int, phi_module: int) -> str:
    return f"s{int(station_id)}_l{int(layer_id)}_e{int(eta_module)}_p{int(phi_module)}"


def parse_module_key(key: str) -> tuple[int, int, int, int]:
    parts = str(key).split("_")
    if len(parts) != 4 or not parts[0].startswith("s"):
        raise ValueError(f"not a module key: {key}")
    return (
        int(parts[0][1:]),
        int(parts[1][1:]),
        int(parts[2][1:]),
        int(parts[3][1:]),
    )


def decode_hit_pattern(hit_pattern: int) -> tuple[tuple[int, int], ...]:
    hits = []
    pattern = int(hit_pattern)
    for layer in IFT_LAYER_IDS:
        for side in (0, 1):
            if pattern & (1 << (2 * layer + side)):
                hits.append((int(layer), int(side)))
    return tuple(hits)


def c_dx_layer_weight(layer_id: int) -> float:
    if int(layer_id) == 0:
        return 1.0
    if int(layer_id) == 2:
        return -1.0
    return 0.0


def se3_delta_xyz(
    position_mm: Sequence[float],
    *,
    center_mm: Sequence[float],
    dx_mm: float = 0.0,
    dy_mm: float = 0.0,
    dz_mm: float = 0.0,
    rx_rad: float = 0.0,
    ry_rad: float = 0.0,
    rz_rad: float = 0.0,
) -> np.ndarray:
    x, y, z = (float(value) for value in position_mm)
    cx, cy, cz = (float(value) for value in center_mm)
    lx, ly, lz = x - cx, y - cy, z - cz
    return np.asarray(
        [
            float(dx_mm) + float(rz_rad) * ly - float(ry_rad) * lz,
            float(dy_mm) - float(rz_rad) * lx + float(rx_rad) * lz,
            float(dz_mm) + float(ry_rad) * lx - float(rx_rad) * ly,
        ],
        dtype=np.float64,
    )


def intercept_shift_xy(
    delta_xyz_mm: Sequence[float],
    *,
    tx: float,
    ty: float,
) -> np.ndarray:
    dx, dy, dz = (float(value) for value in delta_xyz_mm)
    return np.asarray([dx - float(tx) * dz, dy - float(ty) * dz], dtype=np.float64)


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    a = np.asarray(left, dtype=np.float64).reshape(-1)
    b = np.asarray(right, dtype=np.float64).reshape(-1)
    if a.size != b.size:
        raise ValueError("cosine vectors must have the same length")
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a <= 0.0 or norm_b <= 0.0:
        return float("nan")
    return float(np.dot(a, b) / (norm_a * norm_b))


def predict_state_at_z(fit: GlobalTrackFit, z_mm: float) -> np.ndarray:
    delta = float(z_mm) - float(fit.z_reference_mm)
    x0, y0, tx, ty = (float(value) for value in fit.parameters)
    return np.asarray([x0 + tx * delta, y0 + ty * delta, tx, ty], dtype=np.float64)


def propagate_xy_variance(covariance: Sequence[Sequence[float]], delta_z_mm: float) -> tuple[float, float]:
    cov = np.asarray(covariance, dtype=np.float64)
    dz = float(delta_z_mm)
    var_x = float(cov[0, 0] + 2.0 * dz * cov[0, 2] + dz * dz * cov[2, 2])
    var_y = float(cov[1, 1] + 2.0 * dz * cov[1, 3] + dz * dz * cov[3, 3])
    return max(var_x, 1.0e-8), max(var_y, 1.0e-8)


@dataclass(frozen=True)
class TrackletHit:
    run_id: int
    event_id: int
    tracklet_id: int
    station_id: int
    layer_id: int
    phi_module: int
    eta_module: int
    side: int
    module_identifier: int
    cluster_identifier: int


@dataclass(frozen=True)
class ModuleMeasurement:
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
    global_x_mm: float
    global_y_mm: float
    global_z_mm: float
    local_u_mm: float
    local_v_mm: float
    sigma_x_mm: float
    sigma_y_mm: float
    tx: float
    ty: float
    predicted_x_mm: float
    predicted_y_mm: float
    unbiased_residual_x_mm: float
    unbiased_residual_y_mm: float
    neighbor_z_mm: tuple[float, ...] = (47.4,)
    measurement_source: str = MEASUREMENT_SOURCE
    unbiased_method: str = UNBIASED_METHOD
    residual_kind: str = RESIDUAL_KIND
    residual_label: str = RESIDUAL_DECREASE_LABEL
    not_a_complete_acts_cluster_residual: bool = True


@dataclass
class SoftwarePayload:
    module_six: dict[str, np.ndarray] = field(default_factory=dict)
    station_six: dict[int, np.ndarray] = field(default_factory=dict)
    c_dx_mm: float = 0.0
    c_dx_station: int = IFT_STATION_ID

    def copy(self) -> "SoftwarePayload":
        return SoftwarePayload(
            module_six={key: np.array(value, dtype=np.float64) for key, value in self.module_six.items()},
            station_six={int(key): np.array(value, dtype=np.float64) for key, value in self.station_six.items()},
            c_dx_mm=float(self.c_dx_mm),
            c_dx_station=int(self.c_dx_station),
        )


def zero_six() -> np.ndarray:
    return np.zeros(6, dtype=np.float64)


def load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected JSON mapping: {path}")
    return dict(payload)


def load_selected_routes(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            row = json.loads(text)
            if not isinstance(row, Mapping):
                raise ValueError(f"selected route is not a mapping in {path}")
            rows.append(dict(row))
    return rows


def load_tracklet_hits(enhanced_path: str | Path, wanted: set[tuple[int, int]]) -> dict[tuple[int, int], tuple[TrackletHit, ...]]:
    import awkward as ak
    import uproot

    path = Path(enhanced_path).expanduser().resolve()
    with uproot.open(path) as root_file:
        tree = root_file["nt"]
        arrays = tree.arrays(
            [
                "run",
                "eventID",
                "TrackletHit_tracklet_id",
                "TrackletHit_station_id",
                "TrackletHit_layer",
                "TrackletHit_phi_module",
                "TrackletHit_eta_module",
                "TrackletHit_side",
                "TrackletHit_module_identifier",
                "TrackletHit_cluster_identifier",
            ],
            library="ak",
        )
    runs = np.asarray(arrays["run"])
    events = np.asarray(arrays["eventID"])
    out: dict[tuple[int, int], list[TrackletHit]] = {}
    for index, (run_id, event_id) in enumerate(zip(runs, events)):
        key = (int(run_id), int(event_id))
        if key not in wanted:
            continue
        hits = []
        for row in zip(
            ak.to_numpy(arrays["TrackletHit_tracklet_id"][index]),
            ak.to_numpy(arrays["TrackletHit_station_id"][index]),
            ak.to_numpy(arrays["TrackletHit_layer"][index]),
            ak.to_numpy(arrays["TrackletHit_phi_module"][index]),
            ak.to_numpy(arrays["TrackletHit_eta_module"][index]),
            ak.to_numpy(arrays["TrackletHit_side"][index]),
            ak.to_numpy(arrays["TrackletHit_module_identifier"][index]),
            ak.to_numpy(arrays["TrackletHit_cluster_identifier"][index]),
        ):
            hits.append(
                TrackletHit(
                    run_id=key[0],
                    event_id=key[1],
                    tracklet_id=int(row[0]),
                    station_id=int(row[1]),
                    layer_id=int(row[2]),
                    phi_module=int(row[3]),
                    eta_module=int(row[4]),
                    side=int(row[5]),
                    module_identifier=int(row[6]),
                    cluster_identifier=int(row[7]),
                )
            )
        out[key] = tuple(hits)
    return {key: tuple(value) for key, value in out.items()}


def index_events(events: Sequence[EventTracklets]) -> dict[tuple[int, int], EventTracklets]:
    return {(int(event.run_id), int(event.event_id)): event for event in events}


def _route_endpoints(route: Mapping[str, Any]) -> list[dict[str, Any]]:
    provenance = route.get("endpoint_provenance")
    if not isinstance(provenance, list) or not provenance:
        raise ValueError("selected route lacks endpoint_provenance")
    return [dict(item) for item in provenance]


def _find_tracklet_row(event: EventTracklets, station_id: int, tracklet_id: int) -> int:
    matches = np.flatnonzero(
        (np.asarray(event.station_id) == int(station_id))
        & (np.asarray(event.tracklet_id) == int(tracklet_id))
    )
    if matches.size != 1:
        raise ValueError(
            f"event {event.run_id}/{event.event_id} has {matches.size} "
            f"rows for station {station_id} tracklet {tracklet_id}"
        )
    return int(matches[0])


def _loo_fit_for_station(
    event: EventTracklets,
    endpoint_rows: Sequence[int],
    left_out_row: int,
) -> GlobalTrackFit:
    reference = [int(row) for row in endpoint_rows if int(row) != int(left_out_row)]
    if len(reference) < 2:
        raise ValueError("leave-one-station-out requires at least two remaining stations")
    return fit_global_straight_track(
        event.state[reference],
        event.covariance[reference],
        event.z_mm[reference],
        endpoint_indices=reference,
        station_ids=event.station_id[reference],
    )


def build_mapping(
    routes: Sequence[Mapping[str, Any]],
    events: Mapping[tuple[int, int], EventTracklets],
    hits_by_event: Mapping[tuple[int, int], Sequence[TrackletHit]],
) -> dict[str, Any]:
    joined = 0
    missing_event = 0
    missing_hits = 0
    modules: Counter[str] = Counter()
    layers: Counter[tuple[int, int]] = Counter()
    stations: Counter[int] = Counter()
    cluster_ids = 0
    unique_modules: set[int] = set()
    unique_clusters: set[int] = set()
    for route in routes:
        key = (int(route["run_id"]), int(route["event_id"]))
        if key not in events:
            missing_event += 1
            continue
        event_hits = hits_by_event.get(key, ())
        for endpoint in _route_endpoints(route):
            station_id = int(endpoint["station_id"])
            tracklet_id = int(endpoint["origin_tracklet_id"])
            matched = [
                hit
                for hit in event_hits
                if int(hit.station_id) == station_id and int(hit.tracklet_id) == tracklet_id
            ]
            if not matched:
                missing_hits += 1
                continue
            joined += 1
            stations[station_id] += len(matched)
            for hit in matched:
                key_m = module_key(hit.station_id, hit.layer_id, hit.eta_module, hit.phi_module)
                modules[key_m] += 1
                layers[(hit.station_id, hit.layer_id)] += 1
                unique_modules.add(int(hit.module_identifier))
                unique_clusters.add(int(hit.cluster_identifier))
                cluster_ids += 1
    return {
        "n_selected_routes": int(len(routes)),
        "n_selected_events": int(len({(int(row["run_id"]), int(row["event_id"])) for row in routes})),
        "n_endpoints_joined_to_hits": int(joined),
        "n_endpoints_missing_event": int(missing_event),
        "n_endpoints_missing_hits": int(missing_hits),
        "n_hits": int(cluster_ids),
        "n_unique_modules": int(len(unique_modules)),
        "n_unique_clusters": int(len(unique_clusters)),
        "hits_by_station": {str(key): int(value) for key, value in sorted(stations.items())},
        "hits_by_station_layer": {
            f"s{station}_l{layer}": int(count)
            for (station, layer), count in sorted(layers.items())
        },
        "hits_by_module": {key: int(value) for key, value in modules.most_common()},
        "traceable_mapping": "V2_selected_route → station_tracklet → SCT_cluster → layer → module",
        "cluster_identifier_source": "TrackletHit_cluster_identifier",
        "module_identifier_source": "TrackletHit_module_identifier",
        "position_source": MEASUREMENT_SOURCE,
        "calypso_unbiased_cluster_residual_available": False,
        "complete": joined > 0 and missing_hits == 0,
    }


def select_smoke_modules(
    mapping: Mapping[str, Any],
    *,
    station_id: int,
    layer_id: int,
    n_modules: int,
) -> list[str]:
    prefix = f"s{int(station_id)}_l{int(layer_id)}_"
    ranked = [
        (key, int(count))
        for key, count in (mapping.get("hits_by_module") or {}).items()
        if str(key).startswith(prefix)
    ]
    ranked.sort(key=lambda item: (-item[1], item[0]))
    if not ranked:
        raise ValueError(f"no TrackletHit modules on station {station_id} layer {layer_id}")
    return [key for key, _ in ranked[: int(n_modules)]]


def build_measurements(
    routes: Sequence[Mapping[str, Any]],
    events: Mapping[tuple[int, int], EventTracklets],
    hits_by_event: Mapping[tuple[int, int], Sequence[TrackletHit]],
    *,
    representative_station: int,
) -> list[ModuleMeasurement]:
    measurements: list[ModuleMeasurement] = []
    for route in routes:
        endpoints = _route_endpoints(route)
        if len(endpoints) < 3:
            continue
        stations = [int(item["station_id"]) for item in endpoints]
        if int(representative_station) not in stations:
            continue
        key = (int(route["run_id"]), int(route["event_id"]))
        event = events.get(key)
        event_hits = hits_by_event.get(key)
        if event is None or event_hits is None:
            continue
        rows = [_find_tracklet_row(event, int(item["station_id"]), int(item["origin_tracklet_id"])) for item in endpoints]
        target_items = [
            (item, row)
            for item, row in zip(endpoints, rows)
            if int(item["station_id"]) == int(representative_station)
        ]
        if not target_items:
            continue
        neighbor_z = tuple(
            float(event.z_mm[row])
            for endpoint, row in zip(endpoints, rows)
            if int(endpoint["station_id"]) != int(representative_station)
        )
        if not neighbor_z:
            continue
        for item, row in target_items:
            loo = _loo_fit_for_station(event, rows, row)
            state = event.state[row]
            covariance = event.covariance[row]
            z_tracklet = float(event.z_mm[row])
            matched = [
                hit
                for hit in event_hits
                if int(hit.tracklet_id) == int(item["origin_tracklet_id"])
                and int(hit.station_id) == int(item["station_id"])
            ]
            for hit in matched:
                z_layer = layer_z_mm(hit.station_id, hit.layer_id)
                delta_z = z_layer - z_tracklet
                x_m = float(state[0] + state[2] * delta_z)
                y_m = float(state[1] + state[3] * delta_z)
                var_x, var_y = propagate_xy_variance(covariance, delta_z)
                predicted = predict_state_at_z(loo, z_layer)
                center = module_center_mm(hit.station_id, hit.layer_id, hit.eta_module, hit.phi_module)
                stereo = STEREO_RAD if int(hit.side) == 0 else -STEREO_RAD
                local_u = (y_m - center[1]) * math.cos(stereo) + (x_m - center[0]) * math.sin(stereo)
                local_v = (x_m - center[0]) * math.cos(stereo) - (y_m - center[1]) * math.sin(stereo)
                measurements.append(
                    ModuleMeasurement(
                        event_id=int(route["event_id"]),
                        route_index=int(route.get("route_index", 0)),
                        run_id=int(route["run_id"]),
                        tracklet_id=int(item["origin_tracklet_id"]),
                        station_id=int(hit.station_id),
                        layer_id=int(hit.layer_id),
                        module_id=module_key(hit.station_id, hit.layer_id, hit.eta_module, hit.phi_module),
                        cluster_id=int(hit.cluster_identifier),
                        module_identifier=int(hit.module_identifier),
                        phi_module=int(hit.phi_module),
                        eta_module=int(hit.eta_module),
                        side=int(hit.side),
                        global_x_mm=x_m,
                        global_y_mm=y_m,
                        global_z_mm=z_layer,
                        local_u_mm=float(local_u),
                        local_v_mm=float(local_v),
                        sigma_x_mm=float(math.sqrt(var_x)),
                        sigma_y_mm=float(math.sqrt(var_y)),
                        tx=float(state[2]),
                        ty=float(state[3]),
                        predicted_x_mm=float(predicted[0]),
                        predicted_y_mm=float(predicted[1]),
                        unbiased_residual_x_mm=float(x_m - predicted[0]),
                        unbiased_residual_y_mm=float(y_m - predicted[1]),
                        neighbor_z_mm=neighbor_z,
                    )
                )
    return measurements


def measurement_shift_xy(measurement: ModuleMeasurement, payload: SoftwarePayload) -> np.ndarray:
    position = np.asarray(
        [measurement.global_x_mm, measurement.global_y_mm, measurement.global_z_mm],
        dtype=np.float64,
    )
    total = np.zeros(3, dtype=np.float64)
    station_six = payload.station_six.get(int(measurement.station_id))
    if station_six is not None:
        center = np.asarray(
            [0.0, 0.0, STATION_Z_MM[int(measurement.station_id)]],
            dtype=np.float64,
        )
        total = total + se3_delta_xyz(
            position,
            center_mm=center,
            dx_mm=float(station_six[0]),
            dy_mm=float(station_six[1]),
            dz_mm=float(station_six[2]),
            rx_rad=float(station_six[3]),
            ry_rad=float(station_six[4]),
            rz_rad=float(station_six[5]),
        )
    if int(measurement.station_id) == int(payload.c_dx_station) and abs(payload.c_dx_mm) > 0.0:
        total = total + np.asarray(
            [float(payload.c_dx_mm) * c_dx_layer_weight(measurement.layer_id), 0.0, 0.0],
            dtype=np.float64,
        )
    module_six = payload.module_six.get(measurement.module_id)
    if module_six is not None:
        center = module_center_mm(
            measurement.station_id,
            measurement.layer_id,
            measurement.eta_module,
            measurement.phi_module,
        )
        total = total + se3_delta_xyz(
            position,
            center_mm=center,
            dx_mm=float(module_six[0]),
            dy_mm=float(module_six[1]),
            dz_mm=float(module_six[2]),
            rx_rad=float(module_six[3]),
            ry_rad=float(module_six[4]),
            rz_rad=float(module_six[5]),
        )
    return intercept_shift_xy(total, tx=measurement.tx, ty=measurement.ty)


def shifted_residuals(measurements: Sequence[ModuleMeasurement], payload: SoftwarePayload) -> np.ndarray:
    rows = []
    for measurement in measurements:
        shift = measurement_shift_xy(measurement, payload)
        rows.append(
            [
                float(measurement.unbiased_residual_x_mm + shift[0]),
                float(measurement.unbiased_residual_y_mm + shift[1]),
            ]
        )
    return np.asarray(rows, dtype=np.float64)


def residual_vector_x(measurements: Sequence[ModuleMeasurement], payload: SoftwarePayload | None = None) -> np.ndarray:
    if payload is None:
        return np.asarray([item.unbiased_residual_x_mm for item in measurements], dtype=np.float64)
    return shifted_residuals(measurements, payload)[:, 0]


def validate_residuals(measurements: Sequence[ModuleMeasurement]) -> dict[str, Any]:
    if not measurements:
        raise ValueError("no module-level measurements for residual validation")
    residuals_x = np.asarray([item.unbiased_residual_x_mm for item in measurements], dtype=np.float64)
    residuals_y = np.asarray([item.unbiased_residual_y_mm for item in measurements], dtype=np.float64)
    finite = bool(np.isfinite(residuals_x).all() and np.isfinite(residuals_y).all())
    return {
        "n_measurements": int(len(measurements)),
        "n_routes": int(len({(item.run_id, item.event_id, item.route_index) for item in measurements})),
        "unbiased_method": UNBIASED_METHOD,
        "measurement_source": MEASUREMENT_SOURCE,
        "residual_kind": RESIDUAL_KIND,
        "not_a_complete_acts_cluster_residual": True,
        "calypso_acts_unbiased_residual_available": False,
        "biased_residual_used_as_alignment_observable": False,
        "residual_label": RESIDUAL_DECREASE_LABEL,
        "leave_one_station_out": True,
        "finite": finite,
        "residual_x_mm": {
            "mean": float(np.mean(residuals_x)),
            "median": float(np.median(residuals_x)),
            "rms": float(np.sqrt(np.mean(residuals_x * residuals_x))),
            "min": float(np.min(residuals_x)),
            "max": float(np.max(residuals_x)),
        },
        "residual_y_mm": {
            "mean": float(np.mean(residuals_y)),
            "median": float(np.median(residuals_y)),
            "rms": float(np.sqrt(np.mean(residuals_y * residuals_y))),
            "min": float(np.min(residuals_y)),
            "max": float(np.max(residuals_y)),
        },
        "not_alignment_closure": True,
    }


def _six_with(index: int, value: float) -> np.ndarray:
    six = zero_six()
    six[int(index)] = float(value)
    return six


def _central_jacobian_column(
    measurements: Sequence[ModuleMeasurement],
    *,
    plus: SoftwarePayload,
    minus: SoftwarePayload,
    step: float,
) -> np.ndarray:
    if abs(float(step)) <= 0.0:
        raise ValueError("FD step must be non-zero")
    return (residual_vector_x(measurements, plus) - residual_vector_x(measurements, minus)) / (
        2.0 * float(step)
    )


def _payload_module(module_id: str, index: int, value: float) -> SoftwarePayload:
    return SoftwarePayload(module_six={module_id: _six_with(index, value)})


def _payload_station(station_id: int, index: int, value: float) -> SoftwarePayload:
    return SoftwarePayload(station_six={int(station_id): _six_with(index, value)})


def _payload_cdx(value: float, station_id: int = IFT_STATION_ID) -> SoftwarePayload:
    return SoftwarePayload(c_dx_mm=float(value), c_dx_station=int(station_id))


def run_fd_smoke(
    measurements: Sequence[ModuleMeasurement],
    smoke_modules: Sequence[str],
    *,
    station_id: int,
    translation_steps_mm: Sequence[float],
    rotation_steps_mrad: Sequence[float],
    c_dx_steps_mm: Sequence[float],
    linearity_max_relative_deviation: float,
    targeting_max_offmodule_fraction: float,
) -> dict[str, Any]:
    if not 2 <= len(smoke_modules) <= 4:
        raise ValueError("smoke test must use 2-4 modules")
    if len(translation_steps_mm) < 2 or len(rotation_steps_mrad) < 2 or len(c_dx_steps_mm) < 2:
        raise ValueError("FD smoke test needs two steps per family")
    probes: list[dict[str, Any]] = []
    all_linear = True
    all_signed = True
    all_targeted = True

    def _record(
        name: str,
        column_a: np.ndarray,
        column_b: np.ndarray,
        *,
        expected_sign: int | None,
        target_mask: np.ndarray | None,
        step_a: float,
        step_b: float,
    ) -> None:
        nonlocal all_linear, all_signed, all_targeted
        norm_a = float(np.linalg.norm(column_a))
        norm_b = float(np.linalg.norm(column_b))
        rel = float("nan")
        if norm_a > 0.0:
            rel = float(np.linalg.norm(column_a - column_b) / norm_a)
        elif norm_b == 0.0:
            rel = 0.0
        linear = bool(math.isfinite(rel) and rel <= float(linearity_max_relative_deviation))
        mean_deriv = float(np.mean(column_a)) if expected_sign is None else float(
            np.mean(column_a[target_mask]) if target_mask is not None and target_mask.any() else np.mean(column_a)
        )
        signed = True
        if expected_sign is not None:
            signed = bool(mean_deriv * float(expected_sign) > 0.0)
        targeted = True
        if target_mask is not None and target_mask.any() and (~target_mask).any():
            on = float(np.max(np.abs(column_a[target_mask])))
            off = float(np.max(np.abs(column_a[~target_mask])))
            if on == 0.0 and off == 0.0:
                targeted = True
            else:
                targeted = bool(on > 0.0 and off <= float(targeting_max_offmodule_fraction) * on)
        all_linear = all_linear and linear
        all_signed = all_signed and signed
        all_targeted = all_targeted and targeted
        probes.append(
            {
                "name": name,
                "step_a": float(step_a),
                "step_b": float(step_b),
                "jacobian_norm_step_a": norm_a,
                "linearity_relative_deviation": rel,
                "linear": linear,
                "mean_derivative": mean_deriv,
                "expected_sign": expected_sign,
                "sign_ok": signed,
                "target_ok": targeted,
                "geometry_payload_backend": "software_module_transform",
                "official_conditions_write": False,
            }
        )

    t_small, t_large = (float(value) for value in translation_steps_mm[:2])
    r_small, r_large = (float(value) / 1000.0 for value in rotation_steps_mrad[:2])
    c_small, c_large = (float(value) for value in c_dx_steps_mm[:2])
    for module_id in smoke_modules:
        mask = np.asarray([item.module_id == module_id for item in measurements], dtype=bool)
        for index, name in enumerate(SIX_DOF):
            if name in ("dx", "dy", "dz"):
                step_a, step_b = t_small, t_large
            else:
                step_a, step_b = r_small, r_large
            column_a = _central_jacobian_column(
                measurements,
                plus=_payload_module(module_id, index, step_a),
                minus=_payload_module(module_id, index, -step_a),
                step=step_a,
            )
            column_b = _central_jacobian_column(
                measurements,
                plus=_payload_module(module_id, index, step_b),
                minus=_payload_module(module_id, index, -step_b),
                step=step_b,
            )
            expected = 1 if name == "dx" else None
            _record(
                f"{module_id}_{name}",
                column_a,
                column_b,
                expected_sign=expected,
                target_mask=mask,
                step_a=step_a,
                step_b=step_b,
            )
    l0 = np.asarray([item.layer_id == 0 for item in measurements], dtype=bool)
    l2 = np.asarray([item.layer_id == 2 for item in measurements], dtype=bool)
    column_c_a = _central_jacobian_column(
        measurements,
        plus=_payload_cdx(c_small, station_id),
        minus=_payload_cdx(-c_small, station_id),
        step=c_small,
    )
    column_c_b = _central_jacobian_column(
        measurements,
        plus=_payload_cdx(c_large, station_id),
        minus=_payload_cdx(-c_large, station_id),
        step=c_large,
    )
    _record(
        "C_dx",
        column_c_a,
        column_c_b,
        expected_sign=None,
        target_mask=None,
        step_a=c_small,
        step_b=c_large,
    )
    c_dx_l0_sign = bool(l0.any() and float(np.mean(column_c_a[l0])) > 0.0)
    c_dx_l2_sign = bool(l2.any() and float(np.mean(column_c_a[l2])) < 0.0)
    all_signed = all_signed and c_dx_l0_sign and c_dx_l2_sign
    return {
        "representative_station": int(station_id),
        "smoke_modules": list(smoke_modules),
        "n_measurements": int(len(measurements)),
        "probes": probes,
        "c_dx_layer0_positive": c_dx_l0_sign,
        "c_dx_layer2_negative": c_dx_l2_sign,
        "geometry_payload_acts_on_target_module": all_targeted,
        "residual_derivative_signs_ok": all_signed,
        "jacobian_linear_region": all_linear,
        "official_conditions_write": False,
        "athena_fd_probes": False,
        "alignment_correction": False,
        "passed": bool(all_targeted and all_signed and all_linear),
    }


def parameter_columns(
    measurements: Sequence[ModuleMeasurement],
    smoke_modules: Sequence[str],
    *,
    station_id: int,
    translation_step_mm: float,
    rotation_step_rad: float,
    c_dx_step_mm: float,
) -> tuple[list[str], np.ndarray]:
    names: list[str] = []
    columns: list[np.ndarray] = []
    for module_id in smoke_modules:
        for index, dof in enumerate(SIX_DOF):
            step = float(translation_step_mm) if dof in ("dx", "dy", "dz") else float(rotation_step_rad)
            names.append(f"{module_id}_{dof}")
            columns.append(
                _central_jacobian_column(
                    measurements,
                    plus=_payload_module(module_id, index, step),
                    minus=_payload_module(module_id, index, -step),
                    step=step,
                )
            )
    names.append("C_dx")
    columns.append(
        _central_jacobian_column(
            measurements,
            plus=_payload_cdx(c_dx_step_mm, station_id),
            minus=_payload_cdx(-c_dx_step_mm, station_id),
            step=c_dx_step_mm,
        )
    )
    names.append("station_dx")
    columns.append(
        _central_jacobian_column(
            measurements,
            plus=_payload_station(station_id, 0, translation_step_mm),
            minus=_payload_station(station_id, 0, -translation_step_mm),
            step=translation_step_mm,
        )
    )
    names.append("station_ry")
    columns.append(
        _central_jacobian_column(
            measurements,
            plus=_payload_station(station_id, 4, rotation_step_rad),
            minus=_payload_station(station_id, 4, -rotation_step_rad),
            step=rotation_step_rad,
        )
    )
    return names, np.column_stack(columns)


def analyze_identifiability(
    measurements: Sequence[ModuleMeasurement],
    jacobian: np.ndarray,
    parameter_names: Sequence[str],
) -> dict[str, Any]:
    values = np.asarray(jacobian, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(parameter_names):
        raise ValueError("Jacobian shape does not match parameter names")
    weights = np.asarray(
        [1.0 / max(item.sigma_x_mm ** 2, 1.0e-8) for item in measurements],
        dtype=np.float64,
    )
    weighted = values * np.sqrt(weights)[:, None]
    singular = np.linalg.svd(weighted, compute_uv=False)
    rank = int(np.sum(singular > 1.0e-8 * float(singular[0]) if singular.size else 0.0))
    condition = float(singular[0] / singular[-1]) if singular.size and singular[-1] > 0.0 else float("inf")
    fisher = weighted.T @ weighted
    fisher = 0.5 * (fisher + fisher.T)
    correlation = np.full(fisher.shape, float("nan"), dtype=np.float64)
    diagonal = np.diag(fisher)
    for i in range(fisher.shape[0]):
        for j in range(fisher.shape[1]):
            denom = math.sqrt(abs(diagonal[i] * diagonal[j]))
            if denom > 0.0:
                correlation[i, j] = float(fisher[i, j] / denom)
    weak = []
    if singular.size:
        _u, _s, vt = np.linalg.svd(weighted, full_matrices=False)
        for index, value in enumerate(singular):
            if value <= 1.0e-3 * float(singular[0]):
                direction = {
                    name: float(vt[index, col])
                    for col, name in enumerate(parameter_names)
                }
                weak.append(
                    {
                        "singular_value": float(value),
                        "fraction_of_largest": float(value / singular[0]) if singular[0] else None,
                        "leading_components": sorted(
                            direction.items(),
                            key=lambda item: -abs(item[1]),
                        )[:4],
                    }
                )
    return {
        "n_observations": int(values.shape[0]),
        "n_parameters": int(values.shape[1]),
        "parameter_names": list(parameter_names),
        "rank": rank,
        "numerical_rank_tolerance": "1e-8 * sigma_max",
        "singular_values": [float(value) for value in singular],
        "condition_number": condition,
        "fisher_diagonal": [float(value) for value in diagonal],
        "parameter_correlation": json_ready(correlation),
        "null_or_weak_modes": weak,
        "alignment_correction": False,
        "residual_label": RESIDUAL_DECREASE_LABEL,
    }


def station_space_columns(
    measurements: Sequence[ModuleMeasurement],
    *,
    station_id: int,
    translation_step_mm: float,
    rotation_step_rad: float,
    c_dx_step_mm: float,
) -> dict[str, np.ndarray]:
    groups: dict[tuple[int, int, int], list[ModuleMeasurement]] = defaultdict(list)
    for item in measurements:
        groups[(item.run_id, item.event_id, item.route_index)].append(item)

    def _compress(payload: SoftwarePayload) -> np.ndarray:
        residuals = []
        for items in groups.values():
            station_hits = [hit for hit in items if int(hit.station_id) == int(station_id)]
            if len(station_hits) < 2:
                continue
            z = np.asarray([hit.global_z_mm for hit in station_hits], dtype=np.float64)
            x = []
            for hit in station_hits:
                shift = measurement_shift_xy(hit, payload)
                x.append(hit.global_x_mm + float(shift[0]))
            x = np.asarray(x, dtype=np.float64)
            zref = float(np.mean(z))
            design = np.column_stack([np.ones(z.size), z - zref])
            params, *_ = np.linalg.lstsq(design, x, rcond=None)
            neighbor_z = tuple(station_hits[0].neighbor_z_mm)
            if not neighbor_z:
                neighbor_z = (STATION_Z_MM[1],)
            for z_far in neighbor_z:
                residuals.append(float(params[0] + params[1] * (float(z_far) - zref)))
        return np.asarray(residuals, dtype=np.float64)

    def _column(plus: SoftwarePayload, minus: SoftwarePayload, step: float) -> np.ndarray:
        return (_compress(plus) - _compress(minus)) / (2.0 * float(step))

    return {
        "station_dx": _column(
            _payload_station(station_id, 0, translation_step_mm),
            _payload_station(station_id, 0, -translation_step_mm),
            translation_step_mm,
        ),
        "station_ry": _column(
            _payload_station(station_id, 4, rotation_step_rad),
            _payload_station(station_id, 4, -rotation_step_rad),
            rotation_step_rad,
        ),
        "C_dx": _column(
            _payload_cdx(c_dx_step_mm, station_id),
            _payload_cdx(-c_dx_step_mm, station_id),
            c_dx_step_mm,
        ),
    }


def compare_leakage(
    measurements: Sequence[ModuleMeasurement],
    parameter_names: Sequence[str],
    module_jacobian: np.ndarray,
    *,
    station_id: int,
    translation_step_mm: float,
    rotation_step_rad: float,
    c_dx_step_mm: float,
    thresholds: Mapping[str, float],
    frozen_a: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    by_name = {name: module_jacobian[:, index] for index, name in enumerate(parameter_names)}
    required = ("station_dx", "station_ry", "C_dx")
    missing = [name for name in required if name not in by_name]
    if missing:
        raise ValueError(f"module Jacobian lacks leakage columns {missing}")
    module_dx = abs(cosine(by_name["station_dx"], by_name["C_dx"]))
    module_ry = abs(cosine(by_name["station_ry"], by_name["C_dx"]))
    station_cols = station_space_columns(
        measurements,
        station_id=station_id,
        translation_step_mm=translation_step_mm,
        rotation_step_rad=rotation_step_rad,
        c_dx_step_mm=c_dx_step_mm,
    )
    station_dx = abs(cosine(station_cols["station_dx"], station_cols["C_dx"]))
    station_ry = abs(cosine(station_cols["station_ry"], station_cols["C_dx"]))
    drop_dx = float(station_dx - module_dx) if math.isfinite(station_dx) and math.isfinite(module_dx) else float("nan")
    drop_ry = float(station_ry - module_ry) if math.isfinite(station_ry) and math.isfinite(module_ry) else float("nan")
    high = float(thresholds["high_cosine"])
    restored = float(thresholds["restored_cosine"])
    leakage_matrix = np.column_stack([by_name["station_dx"], by_name["station_ry"], by_name["C_dx"]])
    leakage_singular = np.linalg.svd(leakage_matrix, compute_uv=False)
    leakage_rank = (
        int(np.sum(leakage_singular > LEAKAGE_RANK_RELATIVE_TOLERANCE * float(leakage_singular[0])))
        if leakage_singular.size
        else 0
    )
    min_drop = float(thresholds["minimum_drop"])
    dx_separated = bool(math.isfinite(module_dx) and module_dx <= restored)
    ry_separated = bool(math.isfinite(module_ry) and module_ry <= restored)
    dx_still_collinear = bool(math.isfinite(module_dx) and module_dx >= high)
    ry_still_collinear = bool(math.isfinite(module_ry) and module_ry >= high)
    dx_dropped = bool(math.isfinite(drop_dx) and drop_dx >= min_drop and dx_separated)
    ry_dropped = bool(math.isfinite(drop_ry) and drop_ry >= min_drop and ry_separated)
    station_leakage = np.column_stack(
        [station_cols["station_dx"], station_cols["station_ry"], station_cols["C_dx"]]
    )
    station_leakage_singular = np.linalg.svd(station_leakage, compute_uv=False)
    station_leakage_rank = (
        int(np.sum(station_leakage_singular > LEAKAGE_RANK_RELATIVE_TOLERANCE * float(station_leakage_singular[0])))
        if station_leakage_singular.size
        else 0
    )
    return {
        "observation_spaces": {
            "station_level": (
                "station-recompressed tracklet propagated to other-station z "
                "(field-edge equivalent of cluster → tracklet compression)"
            ),
            "module_level": "per-module module-proxy unbiased residual_x",
        },
        "station_level_abs_cosine": {
            "station_dx_vs_C_dx": station_dx,
            "station_ry_vs_C_dx": station_ry,
        },
        "module_level_abs_cosine": {
            "station_dx_vs_C_dx": module_dx,
            "station_ry_vs_C_dx": module_ry,
        },
        "cosine_drop_station_minus_module": {
            "station_dx_vs_C_dx": drop_dx,
            "station_ry_vs_C_dx": drop_ry,
        },
        "dx_separated": dx_separated,
        "ry_separated": ry_separated,
        "dx_still_collinear": dx_still_collinear,
        "ry_still_collinear": ry_still_collinear,
        "dx_cosine_dropped": dx_dropped,
        "ry_cosine_dropped": ry_dropped,
        "both_cosines_dropped": bool(dx_dropped and ry_dropped),
        "module_leakage_subspace_rank": leakage_rank,
        "module_leakage_singular_values": [float(value) for value in leakage_singular],
        "station_leakage_subspace_rank": station_leakage_rank,
        "station_leakage_singular_values": [float(value) for value in station_leakage_singular],
        "leakage_subspace_resolvable": leakage_rank >= 3,
        "worse_null_than_two_physical_directions": leakage_rank < 2,
        "thresholds": dict(thresholds),
        "frozen_A_native_per_mm_C_dx": None if frozen_a is None else dict(frozen_a),
        "implied_cdx_is_not_a_measurement": True,
        "residual_label": RESIDUAL_DECREASE_LABEL,
    }


def decide_next_stage(
    leakage: Mapping[str, Any],
    identifiability: Mapping[str, Any],
    fd_report: Mapping[str, Any],
    *,
    worse_null_condition_ratio: float,
) -> dict[str, Any]:
    del identifiability, worse_null_condition_ratio
    dx_ok = bool(leakage.get("dx_cosine_dropped") if "dx_cosine_dropped" in leakage else leakage.get("dx_separated"))
    ry_ok = bool(leakage.get("ry_cosine_dropped") if "ry_cosine_dropped" in leakage else leakage.get("ry_separated"))
    resolvable = bool(leakage.get("leakage_subspace_resolvable"))
    worse_null = bool(leakage.get("worse_null_than_two_physical_directions"))
    if not fd_report.get("passed"):
        label = DECISION_TOPOLOGY
        answer = "No"
        go = False
        reason = "FD smoke test did not confirm a usable module Jacobian."
    elif worse_null:
        label = DECISION_TOPOLOGY
        answer = "No"
        go = False
        reason = (
            "The {station dx, station ry, C_dx} leakage subspace has rank < 2, "
            "or the module Jacobian is more singular than the compressed station space."
        )
    elif dx_ok and ry_ok and resolvable:
        label = DECISION_RESTORES
        answer = "Yes"
        go = True
        reason = (
            "Both station dx↔C_dx and station ry↔C_dx absolute cosines drop "
            "from the station-recompressed space into the module-proxy residual "
            "space, and the leakage subspace remains resolvable.  The current "
            "station-level degeneracy is therefore measurement compression."
        )
    elif dx_ok and not ry_ok:
        label = DECISION_MIXED
        answer = "No"
        go = False
        reason = MIXED_REASON
    else:
        label = DECISION_TOPOLOGY
        answer = "No"
        go = False
        reason = (
            "Station dx/ry versus C_dx remain highly collinear in module residual "
            "space, or the module Jacobian introduces a worse null space."
        )
    return {
        "go_no_go_question": GO_NO_GO_QUESTION,
        "answer": answer,
        "go_to_full_detector_module_level_alignment_basis_study": go,
        "go_to_full_module_identifiability_map": go,
        "decision": label,
        "reason": reason,
        "dx_cdx_restored": dx_ok,
        "ry_cdx_restored": ry_ok,
        "worse_null_space": worse_null,
        "still_no_new_network": True,
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
            "Do not sink into more complex ML.  Prefer external survey or a "
            "new independent track topology before any geometry write."
        ),
        "next_allowed_step": (
            "full_module_level_identifiability_map_no_new_network"
            if go
            else "stop_ml_descent_prefer_survey_or_new_topology"
        ),
    }


def frozen_a_from_report(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    report = load_json(path)
    operator = report.get("A_operator") or {}
    native = operator.get("A_native_per_mm_C_dx")
    if not isinstance(native, Mapping):
        return None
    return {
        "source": operator.get("source"),
        "do_not_retune": True,
        "A_native_per_mm_C_dx": dict(native),
        "not_a_measurement": True,
    }


def measurement_records(measurements: Sequence[ModuleMeasurement]) -> list[dict[str, Any]]:
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
            "module_identifier": item.module_identifier,
            "phi_module": item.phi_module,
            "eta_module": item.eta_module,
            "side": item.side,
            "global_position_mm": [item.global_x_mm, item.global_y_mm, item.global_z_mm],
            "module_local_uv_mm": [item.local_u_mm, item.local_v_mm],
            "measurement_uncertainty_mm": [item.sigma_x_mm, item.sigma_y_mm],
            "track_direction_tx_ty": [item.tx, item.ty],
            "predicted_intercept_mm": [item.predicted_x_mm, item.predicted_y_mm],
            "unbiased_residual_mm": [item.unbiased_residual_x_mm, item.unbiased_residual_y_mm],
            "measurement_source": item.measurement_source,
            "unbiased_method": item.unbiased_method,
            "residual_kind": item.residual_kind,
            "not_a_complete_acts_cluster_residual": item.not_a_complete_acts_cluster_residual,
            "residual_label": item.residual_label,
        }
        for item in measurements
    ]


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
        "not_a_complete_acts_cluster_residual": True,
    }
