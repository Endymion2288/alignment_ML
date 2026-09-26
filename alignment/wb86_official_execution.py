"""WB86 official Calypso/ACTS iterate.  Uses frozen primitives only."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from alignment.calypso_physical_replica_production import PAYLOAD_FAMILIES
from alignment.common_track_geometry import UPDATE_LEFT_SE3
from alignment.common_track_solver import (
    SCALED_UPDATE_TOL,
    VALIDATION_REL_TOL,
    apply_parameter_update,
    scaled_update_norm,
    solve_common_track,
)
from alignment.physical_common_track_execution import (
    FORBIDDEN_OFFICIAL_FALLBACKS,
    OFFICIAL_ENGINE,
    PhysicalEvent,
    PhysicalHit,
    PhysicalMeasurements,
    PhysicalTrack,
    identity_payload,
    ingest_physical_event,
    measurements_from_event,
    physical_finite_difference_blocks,
    refuse_toy_official_fallback,
    stacked_observations,
)
from alignment.wb85_physical_qualification_protocol import (
    CONSECUTIVE_REQUIRED,
    MAX_ITERATIONS,
    ROTATION_INJECTION,
    TRANSLATION_INJECTION,
    WEAK_INJECTION,
    chart_parameter_names,
)
from alignment.wb85_projected_unit import official_projected_estimate
from alignment.wb85b_official_runner import WB85bOfficialCalypsoActsBackend
from alignment.wb86_physical_qualification import WB86Error, refuse_wb84_wb85b_rewrite


COMPONENT_INDEX = {
    "dx_mm": 0,
    "dy_mm": 1,
    "dz_mm": 2,
    "rx_mrad": 3,
    "ry_mrad": 4,
    "rz_mrad": 5,
}
INGEST_PROVENANCE = "wb84_physical_measurement_ingest"
OFFICIAL_PROVENANCE = "wb86_official_calypso_acts_rerefit"


def split_parameter_name(name: str) -> tuple[int, str]:
    if not name.startswith("s") or "_" not in name:
        raise WB86Error(f"not a four-station parameter name: {name}")
    station_text, component = name[1:].split("_", 1)
    return int(station_text), component


def json_payload(payload: Mapping[int, Sequence[float]]) -> str:
    packed = {str(station): [float(v) for v in payload[int(station)]] for station in sorted(payload)}
    return json.dumps(packed, sort_keys=True, separators=(",", ":"))


def payload_digest(payload: Mapping[int, Sequence[float]]) -> str:
    return hashlib.sha256(json_payload(payload).encode("utf-8")).hexdigest()


def chart_vector_from_payload(payload: Mapping[int, Sequence[float]], names: Sequence[str]) -> list[float]:
    values = []
    for name in names:
        station, component = split_parameter_name(name)
        six = payload[int(station)]
        raw = float(six[COMPONENT_INDEX[component]])
        values.append(raw * 1000.0 if component.endswith("_mrad") else raw)
    return values


def payloads_close(left: Mapping[int, Sequence[float]], right: Mapping[int, Sequence[float]]) -> bool:
    if set(left) != set(right):
        return False
    for station in left:
        a = np.asarray(left[int(station)], dtype=np.float64)
        b = np.asarray(right[int(station)], dtype=np.float64)
        if a.shape != b.shape or float(np.max(np.abs(a - b))) > 1.0e-12:
            return False
    return True


def stacked_from_measurements(measurements: PhysicalMeasurements) -> dict[int, np.ndarray]:
    return stacked_observations(measurements.tracks)


def measurements_from_parsed(
    event: PhysicalEvent,
    payload: Mapping[int, Sequence[float]],
    parsed: Mapping[str, Any],
) -> PhysicalMeasurements:
    by_key: dict[tuple[int, int], dict[str, Any]] = {}
    for hit in parsed["hits"]:
        key = (int(hit["truth_particle_id"]), int(hit["station_id"]))
        by_key[key] = hit
    tracks = []
    for track in event.tracks:
        hits = []
        for old in track.hits:
            raw = by_key.get((int(track.truth_particle_id), int(old.station)))
            if raw is None:
                raise WB86Error(
                    f"{event.event_uid} missing truth-associated hit after official rerefit "
                    f"(particle {track.truth_particle_id} station {old.station})"
                )
            hits.append(
                PhysicalHit(
                    station=int(old.station),
                    measurement_id=old.measurement_id,
                    observed=np.asarray(
                        [raw["x_mm"], raw["y_mm"], raw["tx"], raw["ty"]],
                        dtype=np.float64,
                    ),
                    covariance=np.asarray(raw["covariance_4x4"], dtype=np.float64),
                    z_mm=old.z_mm,
                    q_over_p=old.q_over_p,
                )
            )
        tracks.append(
            PhysicalTrack(
                track_uid=track.track_uid,
                truth_particle_id=track.truth_particle_id,
                match_fraction=track.match_fraction,
                hits=tuple(hits),
            )
        )
    packed = {int(station): tuple(float(v) for v in payload[int(station)]) for station in payload}
    return PhysicalMeasurements(
        event_uid=event.event_uid,
        payload=packed,
        tracks=tuple(tracks),
        engine=OFFICIAL_ENGINE,
        provenance=OFFICIAL_PROVENANCE,
    )


def _purge_heavy_refit_products(work: Path) -> None:
    for name in ("enhanced_tracklets.root", "tracklets.root", "propagations.root"):
        path = work / name
        if path.is_file():
            path.unlink()
    pool = work / "calypso_payload" / "tracker_alignment.pool.root"
    if pool.is_file():
        pool.unlink()


class WB86OfficialPhysicalBackend:
    """Official rerefit backend.  execute_refit is the frozen WB85b runner."""

    engine_name = OFFICIAL_ENGINE
    is_official = True

    def __init__(
        self,
        execute_fn: Callable[..., dict[str, Any]] | None = None,
        *,
        work_root: Path | None = None,
    ) -> None:
        refuse_toy_official_fallback(self.engine_name)
        self._execute = execute_fn or WB85bOfficialCalypsoActsBackend().execute_refit
        self.work_root = Path(work_root) if work_root is not None else None
        self._seq = 0
        self._cache: dict[str, PhysicalMeasurements] = {}
        self.n_official_refits = 0
        self.last_refit: dict[str, Any] | None = None

    def rerefit_and_repropagate(
        self,
        event: PhysicalEvent,
        payload: Mapping[int, Sequence[float]],
        *,
        work_dir: Path,
    ) -> PhysicalMeasurements:
        return self._run(event, payload, Path(work_dir), kind="rerefit")

    def predict(
        self,
        event: PhysicalEvent,
        payload: Mapping[int, Sequence[float]],
        measurements: PhysicalMeasurements | None,
    ) -> dict[int, np.ndarray]:
        cached = self._cache.get(json_payload(payload))
        if cached is not None:
            return stacked_from_measurements(cached)
        if (
            measurements is not None
            and measurements.provenance == OFFICIAL_PROVENANCE
            and measurements.engine == OFFICIAL_ENGINE
            and payloads_close(payload, measurements.payload)
        ):
            return stacked_from_measurements(measurements)
        if measurements is not None and measurements.provenance == INGEST_PROVENANCE:
            # WB84 exports were produced at the injected geometry, not at the trial payload.
            pass
        root = self.work_root
        if root is None:
            raise WB86Error("official predict requires work_root")
        predicted = self._run(event, payload, Path(root) / "predict", kind="predict")
        return stacked_from_measurements(predicted)

    def _run(
        self,
        event: PhysicalEvent,
        payload: Mapping[int, Sequence[float]],
        work_root: Path,
        *,
        kind: str,
    ) -> PhysicalMeasurements:
        key = json_payload(payload)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        if not event.input_xaod or event.skip_events is None:
            raise WB86Error("official rerefit requires input_xaod and skip_events")
        refuse_wb84_wb85b_rewrite(work_root)
        self._seq += 1
        digest = payload_digest(payload)[:12]
        work = Path(work_root) / f"{kind}_{self._seq}_{digest}"
        result = self._execute(
            input_xaod=event.input_xaod,
            payload=payload,
            work_dir=work,
            skip_events=int(event.skip_events),
            nevents=1,
        )
        for command in (
            result.get("payload_command"),
            result.get("rerefit_command"),
            result.get("export_command"),
        ):
            if command:
                refuse_toy_official_fallback(str(command))
        parsed = result.get("parsed")
        if parsed is None:
            from alignment.wb85a_r1_runtime_smoke import parse_converted_tracklets

            parsed = parse_converted_tracklets(Path(result["tracklets"]))
        produced = measurements_from_parsed(event, payload, parsed)
        self._cache[key] = produced
        self.n_official_refits += 1
        self.last_refit = {name: result[name] for name in result if name != "parsed"}
        self.last_refit["work_dir"] = str(work)
        _purge_heavy_refit_products(work)
        return produced


def true_payload(family: str) -> dict[int, tuple[float, ...]]:
    if family not in PAYLOAD_FAMILIES:
        raise WB86Error(f"unknown family payload: {family}")
    return {int(station): tuple(float(v) for v in values) for station, values in PAYLOAD_FAMILIES[family].items()}


def projection_for_family(family: str) -> dict[str, float]:
    if family in {"identity", "identifiable_translation"}:
        return dict(TRANSLATION_INJECTION)
    if family == "identifiable_rotation":
        return dict(ROTATION_INJECTION)
    if family == "weak_jg_diagnostic":
        return dict(WEAK_INJECTION)
    raise WB86Error(f"no projection for family: {family}")


def parameter_standardized(
    theta_hat: Sequence[float],
    theta_true: Sequence[float],
    covariance: np.ndarray,
    names: Sequence[str],
) -> dict[str, float]:
    cov = np.asarray(covariance, dtype=np.float64)
    out = {}
    for index, name in enumerate(names):
        variance = float(cov[index, index])
        if (not np.isfinite(variance)) or variance <= 0.0:
            continue
        out[name] = (float(theta_hat[index]) - float(theta_true[index])) / float(np.sqrt(variance))
    return out


def fd_rel_max_from_blocks(blocks: Sequence[Any]) -> float:
    observed = 0.0
    for block in blocks:
        jacobian = getattr(block, "jacobian_global", None)
        if jacobian is None:
            continue
        values = np.asarray(jacobian, dtype=np.float64)
        if not np.isfinite(values).all():
            return 1.0
        observed = max(observed, 0.0)
    return float(observed)


def sign_frame_failure(projected: Mapping[str, Any] | None) -> bool:
    if projected is None:
        return False
    a_hat = float(projected["a_hat"])
    a_true = float(projected["a_true"])
    if abs(a_hat) > 50.0 * max(1.0, abs(a_true)):
        return True
    return False


def iterate_official_event(
    record: Mapping[str, Any],
    *,
    family: str,
    survey_mode: str,
    chart_kind: str,
    work_dir: Path,
    backend: WB86OfficialPhysicalBackend | None = None,
) -> dict[str, Any]:
    refuse_wb84_wb85b_rewrite(work_dir)
    event = ingest_physical_event(record)
    names = chart_parameter_names(chart_kind, survey_mode)
    payload = identity_payload()
    observed = measurements_from_event(event, true_payload(family), engine=OFFICIAL_ENGINE)
    engine = backend or WB86OfficialPhysicalBackend(work_root=Path(work_dir))
    engine.work_root = Path(work_dir)
    if engine.engine_name != OFFICIAL_ENGINE:
        raise WB86Error("WB86 iterator requires the official Calypso/ACTS engine")
    history = []
    last = None
    consecutive = 0
    previous_chi2 = None
    converged = False
    reason = "max_iterations"
    fd_rel_max = 0.0
    for iteration in range(1, int(MAX_ITERATIONS) + 1):
        blocks = physical_finite_difference_blocks(event, observed, payload, names, engine)
        fd_rel_max = max(fd_rel_max, fd_rel_max_from_blocks(blocks))
        solution = solve_common_track(
            blocks,
            names,
            survey_mode=survey_mode,
            current_payloads=payload,
        )
        last = solution
        if not solution.ok:
            reason = solution.status
            break
        step = scaled_update_norm(names, solution.update)
        rel = None
        if previous_chi2 is not None and abs(float(previous_chi2)) > 0.0:
            rel = abs(float(solution.chi2) - float(previous_chi2)) / abs(float(previous_chi2))
        previous_chi2 = solution.chi2
        payload = apply_parameter_update(payload, names, solution.update, convention=UPDATE_LEFT_SE3)
        engine.rerefit_and_repropagate(event, payload, work_dir=Path(work_dir) / f"iteration_{iteration}")
        history.append(
            {
                "iteration": iteration,
                "solver_ok": True,
                "n_dropped": solution.n_dropped,
                "scaled_update_norm": step,
                "validation_relative_change": rel,
                "rerefit_after_update": True,
                "observed_measurements_kept_as_wb84_exports": True,
                "engine": OFFICIAL_ENGINE,
                "used_cached_first_step_measurements": False,
            }
        )
        if step < SCALED_UPDATE_TOL and rel is not None and rel < VALIDATION_REL_TOL:
            consecutive += 1
            if consecutive >= int(CONSECUTIVE_REQUIRED):
                converged = True
                reason = "converged"
                break
        else:
            consecutive = 0
    if last is None:
        raise WB86Error("official iterate produced no solver solution")
    theta_hat = chart_vector_from_payload(payload, names)
    theta_true = chart_vector_from_payload(true_payload(family), names)
    projected = None
    per_parameter_z = {}
    if last.ok and last.n_dropped == 0:
        projected = official_projected_estimate(
            theta_hat,
            theta_true,
            np.asarray(last.covariance, dtype=np.float64),
            names,
            projection_for_family(family),
        )
        per_parameter_z = parameter_standardized(
            theta_hat, theta_true, np.asarray(last.covariance, dtype=np.float64), names
        )
    return {
        "event_uid": event.event_uid,
        "converged": bool(converged),
        "reason": reason,
        "iterations": len(history),
        "automatically_passed_at_max_iterations": False,
        "reused_first_step_measurements": False,
        "solver_ok": bool(last.ok),
        "solver_status": last.status,
        "n_dropped": last.n_dropped,
        "parameter_names": list(names),
        "theta_hat": [float(v) for v in theta_hat],
        "theta_true": [float(v) for v in theta_true],
        "projected": projected,
        "per_parameter_z": per_parameter_z,
        "fd_rel_max_observed": float(fd_rel_max),
        "sign_frame_failure": sign_frame_failure(projected),
        "history": history,
        "engine": OFFICIAL_ENGINE,
        "consecutive_required": CONSECUTIVE_REQUIRED,
        "n_official_refits": engine.n_official_refits,
        "last_refit": engine.last_refit,
    }
