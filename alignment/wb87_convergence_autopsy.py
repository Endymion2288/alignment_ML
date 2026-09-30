"""WB87 diagnostic recorder for one frozen WB86 official iteration.

The numerical step is the WB86 loop copied so that per-iteration fields can
be kept.  It calls the same frozen primitives:

- physical_finite_difference_blocks
- solve_common_track
- scaled_update_norm
- apply_parameter_update(left SE(3))
- WB85bOfficialCalypsoActsBackend through WB86OfficialPhysicalBackend

It does not change those primitives, the rank cut, the damping, the
convergence tolerances, or any WB86 artifact.  A trajectory is not a
qualification result.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from alignment.common_track_geometry import UPDATE_LEFT_SE3
from alignment.common_track_solver import (
    SCALED_UPDATE_TOL,
    VALIDATION_REL_TOL,
    apply_parameter_update,
    scaled_update_norm,
    solve_common_track,
)
from alignment.physical_common_track_execution import (
    OFFICIAL_ENGINE,
    identity_payload,
    ingest_physical_event,
    measurements_from_event,
    physical_finite_difference_blocks,
    stacked_observations,
)
from alignment.wb85_physical_qualification_protocol import (
    CONSECUTIVE_REQUIRED,
    MAX_ITERATIONS,
    chart_parameter_names,
)
from alignment.wb85_projected_unit import official_projected_estimate
from alignment.wb86_official_execution import (
    WB86OfficialPhysicalBackend,
    chart_vector_from_payload,
    json_payload,
    payload_digest,
    projection_for_family,
    true_payload,
)
from alignment.wb86_physical_qualification import (
    WB86Error,
    chart_kind_for_family,
    load_catalog_event,
    load_replica_record,
    refuse_wb84_wb85b_rewrite,
)


# Frozen one-sided chart step inside physical_finite_difference_blocks.
FD_CHART_STEP = 1.0e-2
RHO_EPSILON = 1.0e-12
TRAJECTORY_SCHEMA = "wb87_iteration_trajectory_v1"


class RecordingOfficialBackend(WB86OfficialPhysicalBackend):
    """Same official backend, plus a copy of each accepted measurement stack."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.accepted: list[dict[str, Any]] = []

    def _run(self, event, payload, work_root, *, kind: str):  # type: ignore[no-untyped-def]
        produced = super()._run(event, payload, work_root, kind=kind)
        self.accepted.append(
            {
                "kind": str(kind),
                "payload_sha256": payload_digest(payload),
                "stacked": {
                    int(key): np.asarray(value, dtype=np.float64).copy()
                    for key, value in stacked_observations(produced.tracks).items()
                },
            }
        )
        return produced


def _stack_gap(left: Mapping[int, np.ndarray], right: Mapping[int, np.ndarray]) -> np.ndarray:
    pieces = []
    for key in sorted(left):
        pieces.append(
            np.asarray(right[int(key)], dtype=np.float64) - np.asarray(left[int(key)], dtype=np.float64)
        )
    if not pieces:
        return np.zeros(0, dtype=np.float64)
    return np.concatenate(pieces)


def diagnostic_fd_chart_response(
    payload: Mapping[int, Sequence[float]],
    names: Sequence[str],
) -> dict[str, list[float]]:
    """Chart change of one frozen FD shift.  Does not call Calypso or the solver.

    ``physical_finite_difference_blocks`` already applies this same
    ``UPDATE_LEFT_SE3`` shift.  The dict is a record of that geometry map.
    """
    packed = {int(station): tuple(float(value) for value in payload[int(station)]) for station in payload}
    base = chart_vector_from_payload(packed, names)
    columns: dict[str, list[float]] = {}
    for name in names:
        shifted = apply_parameter_update(packed, (name,), (FD_CHART_STEP,), convention=UPDATE_LEFT_SE3)
        changed = chart_vector_from_payload(shifted, names)
        columns[str(name)] = [float(after - before) for after, before in zip(changed, base)]
    return columns


def solver_scaled_distance(names: Sequence[str], left: Sequence[float], right: Sequence[float]) -> float:
    """Distance in the frozen ``PARAMETER_SCALES`` units of ``scaled_update_norm``.

    Each component is divided by the same scale the solver uses before the
    Euclidean norm.  This is not a new convention.
    """
    delta = [float(a) - float(b) for a, b in zip(left, right)]
    return scaled_update_norm(names, delta)


def raw_chart_l2(left: Sequence[float], right: Sequence[float]) -> float:
    delta = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    return float(np.linalg.norm(delta))


def _rho(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Dimensionless local-linearization disagreement.

    ``predicted = J delta``.  ``J`` is the one-sided chart Jacobian already
    computed by ``physical_finite_difference_blocks`` at step 0.01.
    ``actual`` is the change in the official stacked prediction across the
    same left-SE(3) update.  Both vectors use the measurement stack
    ``(x_mm, y_mm, tx, ty)`` repeated over the event's stations and truth
    tracks.  ``rho = 0`` is exact local agreement.  ``rho`` near 1 means
    the physical change differs from the linear prediction by about the
    size of that prediction.
    """
    gap = np.asarray(actual, dtype=np.float64) - np.asarray(predicted, dtype=np.float64)
    predicted_norm = float(np.linalg.norm(predicted))
    disagreement = float(np.linalg.norm(gap))
    return {
        "J_delta_norm": predicted_norm,
        "actual_prediction_change_norm": float(np.linalg.norm(actual)),
        "linearization_error_norm": disagreement,
        "predicted_change_norm": predicted_norm,
        "actual_change_norm": float(np.linalg.norm(actual)),
        "disagreement_norm": disagreement,
        "rho": disagreement / max(predicted_norm, RHO_EPSILON),
    }


def _response_metrics(
    base: Mapping[str, Any],
    after: Mapping[str, Any],
    blocks: Sequence[Any],
    update: np.ndarray,
    observed: Mapping[int, np.ndarray],
    *,
    status: str,
) -> dict[str, Any]:
    order = sorted(base["stacked"])
    if len(blocks) != len(order):
        return {"trajectory_status": "track_count_mismatch"}
    # physical_finite_difference_blocks walks measurements.tracks in event
    # order.  The stacked prediction is keyed by truth particle id.  Rebuild
    # J in sorted-particle order from the blocks' track ids.
    by_track = {}
    for block in blocks:
        by_track[int(block.track_id)] = np.asarray(block.jacobian_global, dtype=np.float64)
    try:
        ordered = [by_track[int(track_id)] for track_id in order]
    except KeyError:
        return {"trajectory_status": "track_id_mismatch"}
    jacobian = np.vstack(ordered)
    predicted = jacobian @ np.asarray(update, dtype=np.float64)
    actual = _stack_gap(base["stacked"], after["stacked"])
    if actual.shape != predicted.shape:
        return {
            "trajectory_status": "shape_mismatch",
            "actual_shape": list(actual.shape),
            "predicted_shape": list(predicted.shape),
        }
    metrics = _rho(actual, predicted)
    metrics.update(
        {
            "trajectory_status": status,
            "residual_norm_before_update": float(np.linalg.norm(_stack_gap(base["stacked"], observed))),
            "residual_norm_after_update": float(np.linalg.norm(_stack_gap(after["stacked"], observed))),
            "geometry_payload_sha256_before": base["payload_sha256"],
            "geometry_payload_sha256_after": after["payload_sha256"],
        }
    )
    return metrics


def _stack_for_payload(accepted: Sequence[Mapping[str, Any]], digest: str) -> Mapping[str, Any] | None:
    for row in reversed(accepted):
        if str(row.get("payload_sha256")) == digest:
            return row
    return None


def _linear_metrics(
    accepted: Sequence[Mapping[str, Any]],
    *,
    iteration: int,
    parameter_count: int,
    blocks: Sequence[Any],
    update: np.ndarray,
    observed: Mapping[int, np.ndarray],
    update_cache_hit: bool,
    payload_sha256_before: str,
    payload_sha256_after: str,
) -> dict[str, Any]:
    """Match accepted official products to one solver step.

    The rerefit for iteration ``k`` is accepted record
    ``k * (P + 1)``.  The ``P`` records immediately before it are the
    forward-difference shifts.  Iteration 1 has an explicit base predict at
    record 0.  Later iterations reuse the previous rerefit as that base,
    and that reuse does not append a new accepted record.

    A later iteration whose update payload is already cached appends no
    rerefit.  That is ``cached_repeat``, including every iteration after the
    first repeat.  ``accepted_record_missing`` is reserved for a step whose
    official refit count says a new geometry was produced and the record is
    absent.
    """
    rerefit_index = int(iteration) * (parameter_count + 1)
    if int(iteration) == 1:
        base_index = 0
    else:
        base_index = rerefit_index - (parameter_count + 1)
    indexed = (
        rerefit_index < len(accepted)
        and base_index < len(accepted)
        and base_index >= 0
    )
    if indexed:
        base = accepted[base_index]
        after = accepted[rerefit_index]
        shift_start = rerefit_index - parameter_count
        shifts = accepted[shift_start:rerefit_index]
        kinds_ok = after["kind"] == "rerefit" and all(row["kind"] == "predict" for row in shifts)
        if int(iteration) == 1:
            kinds_ok = kinds_ok and base["kind"] == "predict"
        else:
            kinds_ok = kinds_ok and base["kind"] == "rerefit"
        if not kinds_ok:
            return {"trajectory_status": "accepted_record_kind_mismatch"}
        return _response_metrics(base, after, blocks, update, observed, status="recorded")
    if update_cache_hit and int(iteration) > 1:
        # Both endpoints were computed on an earlier visit.  Reuse those
        # stacks when the cache key is the payload digest.  A test double
        # that keys the cache differently still reports cached_repeat.
        base = _stack_for_payload(accepted, payload_sha256_before)
        after = _stack_for_payload(accepted, payload_sha256_after)
        if base is not None and after is not None:
            metrics = _response_metrics(base, after, blocks, update, observed, status="cached_repeat")
            if metrics.get("trajectory_status") == "cached_repeat":
                return metrics
        return {
            "trajectory_status": "cached_repeat",
            "geometry_payload_sha256_before": payload_sha256_before,
            "geometry_payload_sha256_after": payload_sha256_after,
        }
    return {"trajectory_status": "accepted_record_missing"}


def record_frozen_trajectory(
    record: Mapping[str, Any],
    *,
    family: str,
    survey_mode: str,
    chart_kind: str,
    work_dir: Path,
) -> dict[str, Any]:
    """One frozen official iteration with the per-step record kept."""
    refuse_wb84_wb85b_rewrite(work_dir)
    event = ingest_physical_event(record)
    names = chart_parameter_names(chart_kind, survey_mode)
    payload = identity_payload()
    observed_measurements = measurements_from_event(event, true_payload(family), engine=OFFICIAL_ENGINE)
    observed = stacked_observations(observed_measurements.tracks)
    engine = RecordingOfficialBackend(work_root=Path(work_dir))
    engine.work_root = Path(work_dir)
    if engine.engine_name != OFFICIAL_ENGINE:
        raise WB86Error("WB87 recorder requires the official Calypso/ACTS engine")

    history: list[dict[str, Any]] = []
    last = None
    consecutive = 0
    previous_chi2 = None
    converged = False
    reason = "max_iterations"
    theta_rows: list[list[float]] = [chart_vector_from_payload(payload, names)]

    for iteration in range(1, int(MAX_ITERATIONS) + 1):
        blocks = physical_finite_difference_blocks(event, observed_measurements, payload, names, engine)
        solution = solve_common_track(blocks, names, survey_mode=survey_mode, current_payloads=payload)
        last = solution
        if not solution.ok:
            reason = solution.status
            break
        theta_before = chart_vector_from_payload(payload, names)
        payload_before = {
            int(station): tuple(float(value) for value in payload[int(station)]) for station in payload
        }
        step = scaled_update_norm(names, solution.update)
        rel = None
        if previous_chi2 is not None and abs(float(previous_chi2)) > 0.0:
            rel = abs(float(solution.chi2) - float(previous_chi2)) / abs(float(previous_chi2))
        spectrum = np.asarray(solution.condition_spectrum, dtype=np.float64)
        update = np.asarray(solution.update, dtype=np.float64)
        refits_before = int(engine.n_official_refits)
        payload = apply_parameter_update(payload, names, update, convention=UPDATE_LEFT_SE3)
        engine.rerefit_and_repropagate(event, payload, work_dir=Path(work_dir) / f"iteration_{iteration}")
        refits_after = int(engine.n_official_refits)
        theta = chart_vector_from_payload(payload, names)
        theta_rows.append(theta)
        if step < SCALED_UPDATE_TOL and rel is not None and rel < VALIDATION_REL_TOL:
            consecutive += 1
            gate_met = True
        else:
            consecutive = 0
            gate_met = False
        payload_after = {
            int(station): tuple(float(value) for value in payload[int(station)]) for station in payload
        }
        row = {
            "iteration": int(iteration),
            "solver_status": solution.status,
            "solver_ok": True,
            "chi2": None if solution.chi2 is None else float(solution.chi2),
            "previous_chi2": None if previous_chi2 is None else float(previous_chi2),
            "scaled_update_norm": float(step),
            "validation_relative_change": None if rel is None else float(rel),
            "consecutive_convergence_count": int(consecutive),
            "gate_met_this_iteration": bool(gate_met),
            "scaled_update_tol": float(SCALED_UPDATE_TOL),
            "validation_rel_tol": float(VALIDATION_REL_TOL),
            "consecutive_required": int(CONSECUTIVE_REQUIRED),
            "n_dropped": int(solution.n_dropped),
            "n_retained": int(solution.n_retained),
            "rank": int(solution.n_retained),
            "condition_spectrum": [float(value) for value in spectrum],
            "singular_values": [float(value) for value in spectrum],
            "condition_number": (
                float(spectrum[0] / spectrum[-1]) if spectrum.size and float(spectrum[-1]) > 0.0 else None
            ),
            "right_singular_vectors": "spectrum_unavailable",
            "dropped_parameter_names": "spectrum_unavailable",
            "rank_threshold": "spectrum_unavailable",
            "delta_theta": [float(value) for value in update],
            "theta_before": [float(value) for value in theta_before],
            "theta_after": [float(value) for value in theta],
            "raw_chart_L2": raw_chart_l2(theta, theta_before),
            "solver_scaled_chart_distance": solver_scaled_distance(names, theta, theta_before),
            "parameter_names": list(names),
            "payload_before": {str(station): list(values) for station, values in sorted(payload_before.items())},
            "payload_after": {str(station): list(values) for station, values in sorted(payload_after.items())},
            "payload_sha256_before": payload_digest(payload_before),
            "payload_sha256_after": payload_digest(payload_after),
            "official_refits_before": refits_before,
            "official_refits_after": refits_after,
            "update_rerefit_cache": "hit" if refits_after == refits_before else "miss",
        }
        projected = None
        projected_status = "n_dropped_nonzero"
        if int(solution.n_dropped) == 0:
            try:
                projected_raw = official_projected_estimate(
                    theta,
                    chart_vector_from_payload(true_payload(family), names),
                    np.asarray(solution.covariance, dtype=np.float64),
                    names,
                    projection_for_family(family),
                )
                projected = {
                    key: (bool(value) if isinstance(value, (bool, np.bool_)) else value)
                    for key, value in projected_raw.items()
                }
                for key, value in list(projected.items()):
                    if isinstance(value, float):
                        projected[key] = float(value)
                projected_status = "official_projected_estimate"
            except Exception as error:  # projection can refuse a non-positive variance
                projected = None
                projected_status = f"unavailable:{type(error).__name__}"
        row["projected"] = projected
        row["projected_status"] = projected_status
        row["diagnostic_fd_chart_response"] = diagnostic_fd_chart_response(payload_before, names)
        previous_chi2 = solution.chi2
        row.update(
            _linear_metrics(
                engine.accepted,
                iteration=iteration,
                parameter_count=len(names),
                blocks=blocks,
                update=update,
                observed=observed,
                update_cache_hit=refits_after == refits_before,
                payload_sha256_before=payload_digest(payload_before),
                payload_sha256_after=payload_digest(payload_after),
            )
        )
        history.append(row)
        if gate_met and consecutive >= int(CONSECUTIVE_REQUIRED):
            converged = True
            reason = "converged"
            break

    if last is None:
        raise WB86Error("WB87 recorder produced no solver solution")
    theta_hat = chart_vector_from_payload(payload, names)
    theta_true = chart_vector_from_payload(true_payload(family), names)
    return {
        "schema": TRAJECTORY_SCHEMA,
        "qualifies_wb86": False,
        "event_uid": event.event_uid,
        "family": family,
        "survey_mode": survey_mode,
        "chart_kind": chart_kind,
        "parameter_names": list(names),
        "theta_true": [float(value) for value in theta_true],
        "theta_hat": [float(value) for value in theta_hat],
        "theta_initial": theta_rows[0],
        "converged": bool(converged),
        "reason": reason,
        "iterations": len(history),
        "n_dropped_last": int(last.n_dropped),
        "solver_status_last": last.status,
        "n_official_refits": int(engine.n_official_refits),
        "engine": OFFICIAL_ENGINE,
        "executor": "WB85bOfficialCalypsoActsBackend.execute_refit",
        "fd_chart_step": FD_CHART_STEP,
        "rho_epsilon": RHO_EPSILON,
        "rho_definition": (
            "rho = ||actual_prediction_change - J delta|| / max(||J delta||, 1e-12); "
            "dimensionless; J and the prediction change are in the stacked "
            "(x_mm, y_mm, tx, ty) measurement coordinates"
        ),
        "history": history,
        "artifact_sha256": None,
    }


def freeze_artifact_hash(payload: dict[str, Any]) -> dict[str, Any]:
    body = dict(payload)
    body["artifact_sha256"] = None
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    body["artifact_sha256"] = hashlib.sha256(encoded).hexdigest()
    return body


def write_trajectory(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


# Fixed before any physical trajectory.  Do not add or drop indices.
FIXED_EVENT_INDICES = (0, 510, 105, 151, 239, 247, 657, 1104, 2285, 1586, 358, 489, 1745)
WB86_CATALOG_ROOT = Path("outputs/mc24_four_station_wb86_physical_qualification_v1")
WB87_OUTPUT_ROOT = Path("outputs/mc24_four_station_wb87_convergence_autopsy_v1")


def record_catalog_event(index: int, *, work_dir: Path, catalog_root: Path = WB86_CATALOG_ROOT) -> dict[str, Any]:
    """One frozen trajectory for a catalog index.  Reads WB86 catalog, writes nothing there."""
    row = load_catalog_event(int(index), catalog_root)
    if chart_kind_for_family(str(row["family"])) != str(row["chart_kind"]):
        raise WB86Error(f"catalog chart_kind disagrees with family for index {index}")
    record = load_replica_record(row)
    result = record_frozen_trajectory(
        record,
        family=str(row["family"]),
        survey_mode=str(row["survey_mode"]),
        chart_kind=str(row["chart_kind"]),
        work_dir=Path(work_dir),
    )
    result["index"] = int(index)
    result["condition_id"] = str(row["condition_id"])
    result["event_uid"] = str(row["event_uid"])
    result["source_id"] = str(row["source_id"])
    result["input_xaod"] = str(row["input_xaod"])
    result["xaod_entry_index"] = int(row["xaod_entry_index"])
    return freeze_artifact_hash(result)
