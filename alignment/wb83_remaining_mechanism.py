"""WB83 remaining-alignment diagnostics.  Does not change frozen gates.

Reads the same seeds / matrix as the official v2 replicas.  Amplitude-swap
and last-step covariance are research diagnostics, not new qualification
cells.  Do not use this module to loosen |pull_mean|<0.2 or coverage.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

from alignment.common_track_geometry import identity_station_map
from alignment.common_track_solver import (
    DEFAULT_DAMPING,
    MAX_ITERATIONS,
    MOMENTUM_PRIOR,
    apply_parameter_update,
    iterate_common_track,
    linearize_track,
    solve_common_track,
)
from alignment.wb83_qualification import (
    CONSECUTIVE_REQUIRED,
    FIELD_Y,
    N_FIT,
    N_VAL,
    RECONSTRUCTION_PROVENANCE,
    Z_MM,
    apply_direction,
    chart_names,
    hits_for_track,
    lie_chart_vector,
    make_state,
    project_scalar,
    projected_sigma,
    replica_seed,
    resolved_chart,
)

# Partner of the injected C3 dx z-slope.  Same (1,2,3)/sqrt(14) on dy.
# Orthogonal to the frozen weak_direction at the 1e-3 dy contamination level.
DY_SLOPE_DIRECTION = {
    "s1_dy_mm": 1.0 / math.sqrt(14.0),
    "s2_dy_mm": 2.0 / math.sqrt(14.0),
    "s3_dy_mm": 3.0 / math.sqrt(14.0),
}


def generate_replica(
    cell: Mapping[str, Any],
    replica_id: int,
    *,
    inject_amplitude: float | None = None,
) -> dict[str, Any]:
    """Same seed as the official cell; optional different injected amplitude."""
    seed = replica_seed(
        condition_id=str(cell["condition_id"]),
        amplitude=float(cell["amplitude"]),
        survey_mode=str(cell["survey_mode"]),
        replica_id=int(replica_id),
    )
    rng = np.random.default_rng(seed)
    names = chart_names(str(cell["survey_mode"]), cell["chart"])
    start = identity_station_map()
    amplitude = float(cell["amplitude"]) if inject_amplitude is None else float(inject_amplitude)
    truth = apply_direction(start, names, cell["direction"], amplitude, float(cell["envelope"]))
    tracks = []
    states = {}
    for track_id in range(N_FIT + N_VAL):
        state = make_state(rng, track_id)
        states[track_id] = state
        tracks.append(hits_for_track(track_id, state, truth, rng))
    return {
        "seed": seed,
        "names": names,
        "start": start,
        "truth": truth,
        "tracks": tracks,
        "states": states,
        "amplitude_injected": amplitude,
        "true_coefficient": amplitude * float(cell["envelope"]),
    }


def diagnose_replica(
    cell: Mapping[str, Any],
    replica_id: int,
    *,
    inject_amplitude: float | None = None,
    iterate: bool = True,
) -> dict[str, Any]:
    """First-step GLS vs iterate, last-step σ, and partner-mode leakage."""
    generated = generate_replica(cell, replica_id, inject_amplitude=inject_amplitude)
    names = generated["names"]
    start = generated["start"]
    truth = generated["truth"]
    tracks = generated["tracks"]
    states = generated["states"]
    direction = cell["direction"]
    val_ids = tuple(range(N_FIT, N_FIT + N_VAL))
    fit_blocks = [
        linearize_track(
            tracks[track_id],
            states[track_id],
            start,
            names,
            z_mm=Z_MM,
            field_y=FIELD_Y,
            momentum_mode=MOMENTUM_PRIOR,
            qp_prior_mean=float(states[track_id][4]),
        )
        for track_id in range(N_FIT)
    ]
    linear = solve_common_track(fit_blocks, names, survey_mode=str(cell["survey_mode"]))
    first_payloads = start
    first_residual = None
    first_pull = None
    identity_sigma = None
    if linear.ok:
        first_payloads = apply_parameter_update(start, names, linear.update)
        first_remaining = lie_chart_vector(first_payloads, truth, names)
        first_residual = project_scalar(first_remaining, names, direction)
        identity_sigma = projected_sigma(linear.covariance, names, direction)
        first_pull = first_residual / identity_sigma
    if not iterate:
        return {
            "workbook": "83",
            "diagnostic_only": True,
            "reconstruction_provenance": RECONSTRUCTION_PROVENANCE,
            "cell_id": cell["cell_id"],
            "replica_id": int(replica_id),
            "seed": generated["seed"],
            "amplitude_official": float(cell["amplitude"]),
            "amplitude_injected": generated["amplitude_injected"],
            "true_coefficient": generated["true_coefficient"],
            "converged": None,
            "iterations": 1,
            "identity_sigma": identity_sigma,
            "last_step_sigma": None,
            "first_step_residual": first_residual,
            "first_step_pull": first_pull,
            "iterate_residual": first_residual,
            "iterate_pull_identity_sigma": first_pull,
            "iterate_pull_last_sigma": None,
            "dy_slope_residual": project_scalar(
                lie_chart_vector(first_payloads, truth, names), names, DY_SLOPE_DIRECTION
            ),
            "s3_dz_remaining_mm": 0.0,
            "covered_95_identity_sigma": bool(
                identity_sigma is not None and abs(first_residual) <= 1.959963984540054 * identity_sigma
            ),
            "covered_95_last_step_sigma": None,
            "not_a_qualification_row": True,
            "iterate_ran": False,
        }
    result = iterate_common_track(
        tracks,
        {key: value.copy() for key, value in states.items()},
        start,
        z_mm=Z_MM,
        field_y=FIELD_Y,
        survey_mode=str(cell["survey_mode"]),
        reference_station=0,
        validation_track_ids=val_ids,
        max_iterations=MAX_ITERATIONS,
        damping=DEFAULT_DAMPING,
        momentum_mode=MOMENTUM_PRIOR,
        components=resolved_chart(str(cell["survey_mode"]), cell["chart"])[0],
        stations=resolved_chart(str(cell["survey_mode"]), cell["chart"])[1],
        consecutive_required=CONSECUTIVE_REQUIRED,
    )
    remaining = lie_chart_vector(result["payloads"], truth, names)
    residual = project_scalar(remaining, names, direction)
    iterate_pull = None if identity_sigma is None else residual / identity_sigma
    last_sigma = None
    last_pull = None
    last = result.get("solution")
    if last is not None and getattr(last, "ok", False):
        last_sigma = projected_sigma(last.covariance, names, direction)
        last_pull = residual / last_sigma
    dy_residual = project_scalar(remaining, names, DY_SLOPE_DIRECTION)
    dz = 0.0
    if "s3_dz_mm" in names:
        dz = float(remaining[list(names).index("s3_dz_mm")])
    covered_identity = bool(
        identity_sigma is not None and abs(residual) <= 1.959963984540054 * identity_sigma
    )
    covered_last = bool(last_sigma is not None and abs(residual) <= 1.959963984540054 * last_sigma)
    return {
        "workbook": "83",
        "diagnostic_only": True,
        "reconstruction_provenance": RECONSTRUCTION_PROVENANCE,
        "cell_id": cell["cell_id"],
        "replica_id": int(replica_id),
        "seed": generated["seed"],
        "amplitude_official": float(cell["amplitude"]),
        "amplitude_injected": generated["amplitude_injected"],
        "true_coefficient": generated["true_coefficient"],
        "converged": bool(result["converged"]),
        "iterations": result["iterations"],
        "identity_sigma": identity_sigma,
        "last_step_sigma": last_sigma,
        "first_step_residual": first_residual,
        "first_step_pull": first_pull,
        "iterate_residual": residual,
        "iterate_pull_identity_sigma": iterate_pull,
        "iterate_pull_last_sigma": last_pull,
        "dy_slope_residual": dy_residual,
        "s3_dz_remaining_mm": dz,
        "covered_95_identity_sigma": covered_identity,
        "covered_95_last_step_sigma": covered_last,
        "not_a_qualification_row": True,
        "iterate_ran": True,
    }


def summarize_diagnostics(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0}
    def _col(name: str) -> np.ndarray:
        return np.asarray([float(row[name]) for row in rows if row.get(name) is not None], dtype=np.float64)

    iterate = _col("iterate_pull_identity_sigma")
    first = _col("first_step_pull")
    last_pull = _col("iterate_pull_last_sigma")
    identity_sigma = _col("identity_sigma")
    last_sigma = _col("last_step_sigma")
    residual = _col("iterate_residual")
    first_residual = _col("first_step_residual")
    dy = _col("dy_slope_residual")
    dz = _col("s3_dz_remaining_mm")
    n = len(rows)
    covered_id = int(sum(1 for row in rows if row.get("covered_95_identity_sigma")))
    covered_last = int(sum(1 for row in rows if row.get("covered_95_last_step_sigma")))
    corr_rdz = None
    if n > 2 and float(np.std(dz)) > 0.0 and float(np.std(residual)) > 0.0:
        corr_rdz = float(np.corrcoef(residual, dz)[0, 1])
    return {
        "n": n,
        "n_converged": int(sum(1 for row in rows if row.get("converged"))),
        "iterate_pull_mean": float(np.mean(iterate)) if iterate.size else None,
        "iterate_pull_width": float(np.std(iterate, ddof=1)) if iterate.size > 1 else None,
        "first_step_pull_mean": float(np.mean(first)) if first.size else None,
        "first_step_pull_width": float(np.std(first, ddof=1)) if first.size > 1 else None,
        "iterate_minus_first_pull_mean": float(np.mean(iterate - first)) if iterate.size and first.size == iterate.size else None,
        "last_step_pull_mean": float(np.mean(last_pull)) if last_pull.size else None,
        "last_step_pull_width": float(np.std(last_pull, ddof=1)) if last_pull.size > 1 else None,
        "identity_sigma_mean": float(np.mean(identity_sigma)) if identity_sigma.size else None,
        "last_step_sigma_mean": float(np.mean(last_sigma)) if last_sigma.size else None,
        "residual_mean": float(np.mean(residual)) if residual.size else None,
        "residual_std": float(np.std(residual, ddof=1)) if residual.size > 1 else None,
        "first_step_residual_mean": float(np.mean(first_residual)) if first_residual.size else None,
        "dy_slope_residual_mean": float(np.mean(dy)) if dy.size else None,
        "dy_slope_residual_std": float(np.std(dy, ddof=1)) if dy.size > 1 else None,
        "s3_dz_remaining_mean": float(np.mean(dz)) if dz.size else None,
        "s3_dz_remaining_std": float(np.std(dz, ddof=1)) if dz.size > 1 else None,
        "corr_weak_residual_s3_dz": corr_rdz,
        "coverage_identity_sigma": covered_id / n if n else None,
        "coverage_last_step_sigma": covered_last / n if n else None,
        "n_covered_identity": covered_id,
        "n_covered_last_step": covered_last,
    }
