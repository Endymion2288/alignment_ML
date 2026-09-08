"""Workbook 83: independent truth-only replica qualification (fail-closed).

Does not read W64, overlays, 00350, Final Blind, or sealed sets.
WB82 smoke_pass is not an oracle.  Calypso physical-event replicas are not
available under the current data contract without overlays; this module
qualifies the frozen toy-field common-track solver on independently drawn
track ensembles and measurement noise.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from alignment.common_track_geometry import (
    SOLVER_CONTRACT,
    identity_station_map,
    left_update_payload,
)
from alignment.common_track_solver import (
    ASSOCIATION_DEFAULT_SYSTEM,
    DEFAULT_DAMPING,
    FORBIDDEN_PATH_NEEDLES,
    MAX_ITERATIONS,
    MEASUREMENT_DIM,
    MOMENTUM_PRIOR,
    PARAMETER_SCALES,
    QP_PRIOR_SIGMA,
    SURVEY_FINITE_PRIOR,
    SURVEY_FIXED_DZ,
    AlignmentSolverError,
    StationHit,
    assemble_unique_hits,
    chi2_of_blocks,
    fd_column_stability,
    iterate_common_track,
    lab_transport,
    linearize_track,
    pairwise_independent_chi2,
    parameter_chart,
    predict_local_measurement,
    relative_lie_table,
    solve_common_track,
    solver_implementation_contract,
    solver_implementation_fixed,
)
from alignment.four_station import IDENTITY_SIX, STATION_IDS, SURVEY_DZ_PRIOR_SIGMA_MM, parameter_name


WORKBOOK = "83"
EXPERIMENT = "wb83_truth_only_replicas_v1"
RECONSTRUCTION_PROVENANCE = "toy_uniform_By_independent_v1"
SOURCE_ID = "toy_independent_measurement_v1"
MASTER_SEED = 20260907
DESIGN_SEED = 202609070
N_TARGET_REPLICAS = 100
N_FIT = 24
N_VAL = 8
FIELD_Y = 0.35
HIT_SIGMA_MM = 0.02
Z_MM = {0: 0.0, 1: 1000.0, 2: 2000.0, 3: 3000.0}
TRANSLATION_ENVELOPE_MM = 0.5
ROTATION_ENVELOPE_MRAD = 1.0
WEAK_ENVELOPE_MM = 0.5
CONSECUTIVE_REQUIRED = 2
PULL_MEAN_MAX = 0.2
PULL_WIDTH_MIN = 0.8
PULL_WIDTH_MAX = 1.2
COVERAGE_NOMINAL = 0.95
COVERAGE_CI_MAX_WIDTH = 0.15
SCREEN_DXDY_MM = 0.1
SCREEN_RXRY_MRAD = 0.1
SCREEN_RZ_MRAD = 1.0
MIN_REPLICAS_FOR_PASS = 100
FD_REL_MAX = 0.01
OVERLAY_NEEDLES = ("overlay_synthetic", "synthetic_multitrack")

# Mixed S1/S2/S3 relative dx, Gram-Schmidt orthogonal to (z1,z2,z3).
# v = (4, 1, -2) / sqrt(21).  Frozen a priori, not from replica outcomes.
_SQRT21 = math.sqrt(21.0)
TRANSLATION_DIRECTION = {
    "s1_dx_mm": 4.0 / _SQRT21,
    "s2_dx_mm": 1.0 / _SQRT21,
    "s3_dx_mm": -2.0 / _SQRT21,
}
ROTATION_DIRECTION = {
    "s1_rz_mrad": -1.0 / math.sqrt(2.0),
    "s3_rz_mrad": 1.0 / math.sqrt(2.0),
}
TRANSLATION_CHART = {
    "components": ("dx_mm", "dy_mm"),
    "stations": (1, 2, 3),
}
ROTATION_CHART = {
    "components": ("rx_mrad", "rz_mrad"),
    "stations": (1, 3),
}

FORBIDDEN_NEEDLES = FORBIDDEN_PATH_NEEDLES + OVERLAY_NEEDLES


class WB83Error(AlignmentSolverError):
    """Fail-closed WB83 contract violation."""


def refuse_wb83_path(path: object) -> None:
    text = str(path)
    for needle in FORBIDDEN_NEEDLES:
        if needle in text:
            raise WB83Error(f"WB83 refuses development / blind / sealed / overlay path: {text}")


def write_json(path: Path, payload: object) -> None:
    refuse_wb83_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def sha256_payload(payload: object) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def replica_seed(*, condition_id: str, amplitude: float, survey_mode: str, replica_id: int) -> int:
    blob = f"{MASTER_SEED}|{condition_id}|{amplitude:.6f}|{survey_mode}|{int(replica_id)}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(blob).digest()[:8], "little") % (2**31 - 1)


def _direction_vector(names: Sequence[str], direction: Mapping[str, float]) -> np.ndarray:
    values = np.zeros(len(names), dtype=np.float64)
    for index, name in enumerate(names):
        values[index] = float(direction.get(name, 0.0))
    norm = float(np.linalg.norm(values))
    if norm <= 0.0:
        raise WB83Error("geometry direction has vanishing support on the chart")
    return values / norm


def resolved_chart(survey_mode: str, chart: Mapping[str, object]) -> tuple[tuple[str, ...], tuple[int, ...]]:
    components = [str(item) for item in chart["components"]]
    stations = [int(station) for station in chart["stations"]]
    if survey_mode == SURVEY_FINITE_PRIOR:
        if "dz_mm" not in components:
            components.append("dz_mm")
        if 3 not in stations:
            stations.append(3)
    return tuple(components), tuple(stations)


def chart_names(survey_mode: str, chart: Mapping[str, object]) -> tuple[str, ...]:
    components, stations = resolved_chart(survey_mode, chart)
    return parameter_chart(survey_mode=survey_mode, reference_station=0, components=components, stations=stations)


def apply_direction(
    payloads: Mapping[int, Sequence[float]],
    names: Sequence[str],
    direction: Mapping[str, float],
    amplitude: float,
    envelope: float,
) -> dict[int, tuple[float, ...]]:
    from alignment.common_track_solver import apply_parameter_update

    vector = _direction_vector(names, direction) * float(amplitude) * float(envelope)
    return apply_parameter_update(payloads, names, vector)


def lie_chart_vector(predicted: Mapping[int, Sequence[float]], truth: Mapping[int, Sequence[float]], names: Sequence[str]) -> np.ndarray:
    """Newton-chart remaining error from Lie-log ``log(T_pred^{-1} T_truth)``."""
    table = relative_lie_table(predicted, truth)["per_station"]
    values = np.zeros(len(names), dtype=np.float64)
    index_of = {"dx_mm": 0, "dy_mm": 1, "dz_mm": 2, "rx_mrad": 3, "ry_mrad": 4, "rz_mrad": 5}
    for index, name in enumerate(names):
        station = int(name[1])
        component = name.split("_", 1)[1]
        raw = float(table[str(station)][index_of[component]])
        values[index] = raw * (1000.0 if component.endswith("mrad") else 1.0)
    return values


def project_scalar(vector: np.ndarray, names: Sequence[str], direction: Mapping[str, float]) -> float:
    unit = _direction_vector(names, direction)
    return float(unit @ np.asarray(vector, dtype=np.float64))


def projected_sigma(covariance: np.ndarray, names: Sequence[str], direction: Mapping[str, float]) -> float:
    unit = _direction_vector(names, direction)
    variance = float(unit @ covariance @ unit)
    if variance <= 0.0 or not math.isfinite(variance):
        raise WB83Error("projected covariance is not a positive finite variance")
    return math.sqrt(variance)


def make_state(rng: np.random.Generator, track_id: int) -> np.ndarray:
    return np.asarray(
        [
            rng.normal(scale=2.0),
            rng.normal(scale=2.0),
            rng.normal(scale=0.003),
            rng.normal(scale=0.003),
            0.08 + 0.02 * ((track_id % 5) - 2),
        ],
        dtype=np.float64,
    )


def hits_for_track(
    track_id: int,
    state: np.ndarray,
    truth_payloads: Mapping[int, Sequence[float]],
    rng: np.random.Generator,
) -> list[StationHit]:
    hits = []
    noise = rng.normal(scale=HIT_SIGMA_MM, size=(4, 4))
    for station in STATION_IDS:
        predicted = predict_local_measurement(
            state,
            truth_payloads[station],
            z_ref_mm=Z_MM[0],
            z_station_mm=Z_MM[station],
            field_y=FIELD_Y,
        )
        hits.append(
            StationHit(
                track_id=track_id,
                station=station,
                measurement_id=f"{SOURCE_ID}-r{track_id}-s{station}",
                observed=predicted + noise[station],
                covariance=np.eye(4) * HIT_SIGMA_MM**2,
            )
        )
    return hits


def measurement_audit(tracks: Sequence[Sequence[StationHit]]) -> dict[str, float | int]:
    raw = [hit for hits in tracks for hit in hits]
    unique = assemble_unique_hits(raw) if raw else ()
    pairwise = pairwise_independent_chi2(raw) if raw else 0.0
    unique_chi2 = pairwise_independent_chi2(unique) if unique else 0.0
    return {
        "n_raw_pairwise_observations": int(len(raw)),
        "n_unique_measurements": int(len(unique)),
        "deduplication_count": int(len(raw) - len(unique)),
        "pairwise_chi2": float(pairwise),
        "unique_chi2": float(unique_chi2),
        "covariance_block_structure": f"block_diag_{MEASUREMENT_DIM}x{MEASUREMENT_DIM}_per_unique_hit",
    }


def clopper_pearson(successes: int, n: int, *, alpha: float = 0.05) -> tuple[float, float]:
    if n <= 0:
        raise WB83Error("Clopper-Pearson requires a positive replica count")
    from scipy.stats import beta

    k = int(successes)
    if k < 0 or k > n:
        raise WB83Error("coverage count is outside [0, n]")
    lower = 0.0 if k == 0 else float(beta.ppf(alpha / 2.0, k, n - k + 1))
    upper = 1.0 if k == n else float(beta.ppf(1.0 - alpha / 2.0, k + 1, n - k))
    return lower, upper


def frozen_criteria() -> dict[str, object]:
    return {
        "pull_mean_max": PULL_MEAN_MAX,
        "pull_width_min": PULL_WIDTH_MIN,
        "pull_width_max": PULL_WIDTH_MAX,
        "coverage_nominal": COVERAGE_NOMINAL,
        "coverage_ci_max_width": COVERAGE_CI_MAX_WIDTH,
        "min_replicas_for_pass": MIN_REPLICAS_FOR_PASS,
        "screen_dxdy_mm": SCREEN_DXDY_MM,
        "screen_rxry_mrad": SCREEN_RXRY_MRAD,
        "screen_rz_mrad": SCREEN_RZ_MRAD,
        "screening_note": "research-screening only; not a collaboration physics requirement",
        "bias_screening_conditions": ["C0_nominal", "C1_translation", "C2_rotation"],
        "fd_rel_max": FD_REL_MAX,
        "consecutive_required": CONSECUTIVE_REQUIRED,
        "scaled_update_tol": 1.0e-3,
        "validation_rel_tol": 1.0e-3,
        "max_iterations": MAX_ITERATIONS,
        "automatically_pass_at_max_iterations": False,
        "robust_authorized": False,
    }


def measurement_likelihood_contract_json() -> dict[str, object]:
    return {
        "workbook": WORKBOOK,
        "parent_workbook": "82",
        "solver_contract": SOLVER_CONTRACT,
        "association_default_system": ASSOCIATION_DEFAULT_SYSTEM,
        "association": "truth_only",
        "measurement": ["x_mm", "y_mm", "tx", "ty"],
        "nuisance": ["x_mm", "y_mm", "tx", "ty", "q_over_p"],
        "update_convention": "left_se3",
        "finite_error": "se3_log",
        "units": {"translation": "mm", "chart_rotation": "mrad", "payload_rotation": "rad"},
        "survey_modes": [SURVEY_FIXED_DZ, SURVEY_FINITE_PRIOR],
        "damping": DEFAULT_DAMPING,
        "qp_prior_sigma": QP_PRIOR_SIGMA,
        "momentum_mode": MOMENTUM_PRIOR,
        "unique_hit_rule": "measurement_id once",
        "reconstruction_provenance": RECONSTRUCTION_PROVENANCE,
        "not_calypso_physical_refit": True,
        "not_wb82_smoke_oracle": True,
        "alignment_oracle_qualified": False,
        "ml_alignment_eval_authorized": False,
    }


def geometry_cells() -> list[dict[str, object]]:
    cells = [
        {
            "cell_id": "C0_nominal_1p0_fixed_dz",
            "condition_id": "C0_nominal",
            "label": "nominal / no injected misalignment",
            "family": "nominal",
            "amplitude": 1.0,
            "envelope": 0.0,
            "direction": dict(TRANSLATION_DIRECTION),
            "chart": dict(TRANSLATION_CHART),
            "survey_mode": SURVEY_FIXED_DZ,
            "apply_bias_screening": True,
        }
    ]
    for amplitude in (0.5, 1.0):
        cells.append(
            {
                "cell_id": f"C1_translation_{amplitude:g}x_fixed_dz",
                "condition_id": "C1_translation",
                "label": "identifiable relative S1/S2/S3 dx",
                "family": "translation",
                "amplitude": amplitude,
                "envelope": TRANSLATION_ENVELOPE_MM,
                "direction": dict(TRANSLATION_DIRECTION),
                "chart": dict(TRANSLATION_CHART),
                "survey_mode": SURVEY_FIXED_DZ,
                "apply_bias_screening": True,
            }
        )
        cells.append(
            {
                "cell_id": f"C2_rotation_{amplitude:g}x_fixed_dz",
                "condition_id": "C2_rotation",
                "label": "mixed S1/S3 rz, no ry",
                "family": "rotation",
                "amplitude": amplitude,
                "envelope": ROTATION_ENVELOPE_MRAD,
                "direction": dict(ROTATION_DIRECTION),
                "chart": dict(ROTATION_CHART),
                "survey_mode": SURVEY_FIXED_DZ,
                "apply_bias_screening": True,
            }
        )
        cells.append(
            {
                "cell_id": f"C3_weak_{amplitude:g}x_fixed_dz",
                "condition_id": "C3_weak",
                "label": "frozen whitened weak mode",
                "family": "weak",
                "amplitude": amplitude,
                "envelope": WEAK_ENVELOPE_MM,
                "direction": "WEAK_MODE_PLACEHOLDER",
                "chart": dict(TRANSLATION_CHART),
                "survey_mode": SURVEY_FIXED_DZ,
                "apply_bias_screening": False,
            }
        )
    extras = []
    for cell in cells:
        prior = dict(cell)
        prior["cell_id"] = str(cell["cell_id"]).replace("fixed_dz", "finite_survey_prior")
        prior["survey_mode"] = SURVEY_FINITE_PRIOR
        extras.append(prior)
    return cells + extras


def freeze_weak_mode() -> dict[str, object]:
    """SVD of the whitened reduced normal on a dedicated design draw.  Not a replica."""
    rng = np.random.default_rng(DESIGN_SEED)
    start = identity_station_map()
    names = chart_names(SURVEY_FIXED_DZ, TRANSLATION_CHART)
    tracks = []
    states = {}
    for track_id in range(N_FIT):
        state = make_state(rng, track_id)
        states[track_id] = state
        tracks.append(hits_for_track(track_id, state, start, rng))
    blocks = [
        linearize_track(
            hits,
            states[int(hits[0].track_id)],
            start,
            names,
            z_mm=Z_MM,
            field_y=FIELD_Y,
            momentum_mode=MOMENTUM_PRIOR,
            qp_prior_mean=float(states[int(hits[0].track_id)][4]),
        )
        for hits in tracks
    ]
    solution = solve_common_track(blocks, names, survey_mode=SURVEY_FIXED_DZ)
    if not solution.ok:
        raise WB83Error(f"weak-mode design solve failed: {solution.message}")
    spectrum = np.asarray(solution.condition_spectrum, dtype=np.float64)
    if spectrum.size < 2 or float(spectrum[0]) <= 0.0:
        raise WB83Error("weak-mode design spectrum is unusable")
    # Whitened SVD is already the scaled chart used by the solver.
    reduced = np.zeros((len(names), len(names)), dtype=np.float64)
    for block in blocks:
        from alignment.common_track_solver import schur_reduce

        piece, _rhs, *_rest = schur_reduce(block)
        reduced += piece
    scales = np.asarray([PARAMETER_SCALES[name.split("_", 1)[1]] for name in names], dtype=np.float64)
    scaled = reduced * np.outer(scales, scales)
    _left, values, right = np.linalg.svd(scaled, full_matrices=False)
    keep = values > 1.0e-12 * float(values[0])
    if int(np.count_nonzero(keep)) < 2:
        raise WB83Error("design chart has fewer than two retained modes")
    weak_index = int(np.where(keep)[0][-1])
    weak_u = right[weak_index]
    weak_native = weak_u * scales
    weak_native = weak_native / float(np.linalg.norm(weak_native))
    direction = {name: float(value) for name, value in zip(names, weak_native)}
    fd = fd_column_stability(
        lambda values: lab_transport(values, 0.0, 2000.0, field_y=FIELD_Y)[:4],
        np.asarray([0.2, -0.1, 0.001, -0.0005, 0.1], dtype=np.float64),
        1.0e-3,
        index=0,
    )
    return {
        "frozen_before_replicas": True,
        "design_seed": DESIGN_SEED,
        "parameter_names": list(names),
        "whitened_spectrum": [float(value) for value in values],
        "n_retained": int(np.count_nonzero(keep)),
        "n_dropped": int(len(names) - np.count_nonzero(keep)),
        "condition_number": float(values[0] / values[keep][-1]),
        "weak_mode_index": weak_index,
        "weak_direction": direction,
        "not_algebraic_relative_gauge": True,
        "note": "Physical JG partner of reference-chart slope / field; not g_i^{-1} g_j.",
        "fd_stability": fd,
        "full_rank_claim": bool(int(len(names) - np.count_nonzero(keep)) == 0),
    }


def build_geometry_matrix(weak: Mapping[str, object]) -> dict[str, object]:
    cells = []
    weak_direction = dict(weak["weak_direction"])
    for cell in geometry_cells():
        item = dict(cell)
        if item["direction"] == "WEAK_MODE_PLACEHOLDER":
            item["direction"] = weak_direction
        item["chart"] = {
            "components": list(item["chart"]["components"]),
            "stations": list(item["chart"]["stations"]),
        }
        cells.append(item)
    matrix = {
        "workbook": WORKBOOK,
        "experiment": EXPERIMENT,
        "immutable": True,
        "master_seed": MASTER_SEED,
        "design_seed": DESIGN_SEED,
        "n_target_replicas": N_TARGET_REPLICAS,
        "reconstruction_provenance": RECONSTRUCTION_PROVENANCE,
        "source_id": SOURCE_ID,
        "field_y": FIELD_Y,
        "z_mm": Z_MM,
        "translation_envelope_mm": TRANSLATION_ENVELOPE_MM,
        "rotation_envelope_mrad": ROTATION_ENVELOPE_MRAD,
        "weak_envelope_mm": WEAK_ENVELOPE_MM,
        "translation_direction": TRANSLATION_DIRECTION,
        "rotation_direction": ROTATION_DIRECTION,
        "criteria": frozen_criteria(),
        "weak_mode": dict(weak),
        "cells": cells,
        "wb82_smoke_is_not_qualification": True,
        "overlay_replicas_forbidden": True,
        "calypso_physical_replicas_available": False,
    }
    matrix["matrix_sha256"] = sha256_payload({key: value for key, value in matrix.items() if key != "matrix_sha256"})
    return matrix


def build_replica_manifest(matrix: Mapping[str, object]) -> dict[str, object]:
    rows = []
    for cell in matrix["cells"]:
        for replica_id in range(N_TARGET_REPLICAS):
            seed = replica_seed(
                condition_id=str(cell["condition_id"]),
                amplitude=float(cell["amplitude"]),
                survey_mode=str(cell["survey_mode"]),
                replica_id=replica_id,
            )
            if seed == DESIGN_SEED:
                raise WB83Error("replica seed collided with the design seed")
            rows.append(
                {
                    "replica_key": f"{cell['cell_id']}:{replica_id:04d}",
                    "cell_id": cell["cell_id"],
                    "condition_id": cell["condition_id"],
                    "survey_mode": cell["survey_mode"],
                    "amplitude": cell["amplitude"],
                    "replica_id": replica_id,
                    "source_id": SOURCE_ID,
                    "run": f"wb83_{cell['cell_id']}",
                    "event": f"replica_{replica_id:04d}",
                    "seed": seed,
                    "reconstruction_provenance": RECONSTRUCTION_PROVENANCE,
                }
            )
    keys = [row["replica_key"] for row in rows]
    if len(keys) != len(set(keys)):
        raise WB83Error("replica keys are not unique")
    seeds = [row["seed"] for row in rows]
    if len(seeds) != len(set(seeds)):
        raise WB83Error("replica seeds are not unique")
    return {
        "workbook": WORKBOOK,
        "n_target_replicas": N_TARGET_REPLICAS,
        "n_cells": len(matrix["cells"]),
        "n_declared_rows": len(rows),
        "independence": "independent_track_draw_and_hit_noise",
        "not_overlay": True,
        "not_same_event_payload_reuse": True,
        "not_repeated_track_sampling": True,
        "matrix_sha256": matrix["matrix_sha256"],
        "rows": rows,
    }


def run_one_replica(cell: Mapping[str, object], replica_id: int, *, matrix_sha256: str) -> dict[str, object]:
    seed = replica_seed(
        condition_id=str(cell["condition_id"]),
        amplitude=float(cell["amplitude"]),
        survey_mode=str(cell["survey_mode"]),
        replica_id=int(replica_id),
    )
    rng = np.random.default_rng(seed)
    names = chart_names(str(cell["survey_mode"]), cell["chart"])
    start = identity_station_map()
    truth = apply_direction(start, names, cell["direction"], float(cell["amplitude"]), float(cell["envelope"]))
    tracks = []
    states = {}
    for track_id in range(N_FIT + N_VAL):
        state = make_state(rng, track_id)
        states[track_id] = state
        tracks.append(hits_for_track(track_id, state, truth, rng))
    val_ids = tuple(range(N_FIT, N_FIT + N_VAL))
    audit = measurement_audit(tracks)
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
    after_blocks = [
        linearize_track(
            tracks[track_id],
            result["states"][track_id],
            result["payloads"],
            names,
            z_mm=Z_MM,
            field_y=FIELD_Y,
            momentum_mode=MOMENTUM_PRIOR,
            qp_prior_mean=float(states[track_id][4]),
        )
        for track_id in val_ids
    ]
    remaining = lie_chart_vector(result["payloads"], truth, names)
    estimate_minus_truth = -remaining
    true_scale = float(cell["amplitude"]) * float(cell["envelope"])
    true_coeff = true_scale
    est_coeff = project_scalar(estimate_minus_truth, names, cell["direction"]) + true_coeff
    residual_coeff = project_scalar(remaining, names, cell["direction"])
    sigma = None
    pull = None
    if linear.ok:
        sigma = projected_sigma(linear.covariance, names, cell["direction"])
        pull = residual_coeff / sigma
    covered = bool(sigma is not None and abs(residual_coeff) <= 1.959963984540054 * sigma)
    geometry_hash = sha256_payload({str(station): list(truth[station]) for station in STATION_IDS})
    return {
        "workbook": WORKBOOK,
        "replica_key": f"{cell['cell_id']}:{int(replica_id):04d}",
        "cell_id": cell["cell_id"],
        "condition_id": cell["condition_id"],
        "survey_mode": cell["survey_mode"],
        "amplitude": cell["amplitude"],
        "envelope": cell["envelope"],
        "replica_id": int(replica_id),
        "source_id": SOURCE_ID,
        "run": f"wb83_{cell['cell_id']}",
        "event": f"replica_{int(replica_id):04d}",
        "seed": seed,
        "reconstruction_provenance": RECONSTRUCTION_PROVENANCE,
        "geometry_payload_sha256": geometry_hash,
        "matrix_sha256": matrix_sha256,
        "parameter_names": list(names),
        "true_coefficient": true_coeff,
        "estimated_coefficient": est_coeff,
        "residual_coefficient": residual_coeff,
        "sigma": sigma,
        "pull": pull,
        "covered_95": covered,
        "converged": bool(result["converged"]),
        "reason": result["reason"],
        "iterations": result["iterations"],
        "automatically_passed_at_max_iterations": False,
        "nonconverged": not bool(result["converged"]),
        "validation_chi2_after": chi2_of_blocks(after_blocks),
        "n_retained": None if result.get("solution") is None else getattr(result["solution"], "n_retained", None),
        "n_dropped": None if result.get("solution") is None else getattr(result["solution"], "n_dropped", None),
        "full_rank_claim": False if result.get("solution") is None else getattr(result["solution"], "n_dropped", 1) == 0,
        "measurement_audit": audit,
        "history": result["history"],
        "message": result.get("message"),
        "w64_used": False,
        "ml_association_used": False,
        "overlay_used": False,
    }


def summarize_cell(rows: Sequence[Mapping[str, Any]], cell: Mapping[str, object]) -> dict[str, object]:
    usable = [row for row in rows if row.get("pull") is not None and math.isfinite(float(row["pull"]))]
    n = len(usable)
    if n == 0:
        return {
            "cell_id": cell["cell_id"],
            "condition_id": cell["condition_id"],
            "survey_mode": cell["survey_mode"],
            "n_replicas": 0,
            "qualification": "UNKNOWN",
            "reason": "no_finite_pulls",
        }
    pulls = np.asarray([float(row["pull"]) for row in usable], dtype=np.float64)
    residuals = np.asarray([float(row["residual_coefficient"]) for row in usable], dtype=np.float64)
    sigmas = np.asarray([float(row["sigma"]) for row in usable], dtype=np.float64)
    covered = int(sum(1 for row in usable if row.get("covered_95")))
    converged = int(sum(1 for row in usable if row.get("converged")))
    dropped = int(sum(1 for row in usable if int(row.get("n_dropped") or 0) > 0))
    lower, upper = clopper_pearson(covered, n)
    pull_mean = float(np.mean(pulls))
    pull_width = float(np.std(pulls, ddof=1)) if n > 1 else None
    bias = float(np.mean(residuals))
    rms = float(np.sqrt(np.mean(residuals**2)))
    mean_sigma = float(np.mean(sigmas))
    pulls_ok = (
        abs(pull_mean) < PULL_MEAN_MAX
        and pull_width is not None
        and PULL_WIDTH_MIN <= pull_width <= PULL_WIDTH_MAX
    )
    coverage_compatible = lower <= COVERAGE_NOMINAL <= upper
    coverage_narrow = (upper - lower) <= COVERAGE_CI_MAX_WIDTH
    if n < MIN_REPLICAS_FOR_PASS:
        coverage_status = "UNKNOWN"
    elif coverage_compatible and coverage_narrow:
        coverage_status = "PASS"
    else:
        coverage_status = "FAIL"
    screening = None
    screening_ok = True
    if cell.get("apply_bias_screening"):
        family = str(cell.get("family") or cell["condition_id"])
        if family in {"nominal", "translation", "C0_nominal", "C1_translation"}:
            screening = {"kind": "dxdy_mm", "limit": SCREEN_DXDY_MM, "abs_bias": abs(bias)}
            screening_ok = abs(bias) <= SCREEN_DXDY_MM
        elif family in {"rotation", "C2_rotation"}:
            screening = {"kind": "rz_mrad", "limit": SCREEN_RZ_MRAD, "abs_bias": abs(bias)}
            screening_ok = abs(bias) <= SCREEN_RZ_MRAD
    if dropped:
        rank_status = "FAIL_not_full_rank"
    else:
        rank_status = "ok_no_dropped_mode"
    if n < MIN_REPLICAS_FOR_PASS:
        qualification = "UNKNOWN"
        reason = "insufficient_independent_statistics"
    elif not pulls_ok:
        qualification = "FAIL"
        reason = "pulls_failed"
    elif coverage_status != "PASS":
        qualification = "FAIL"
        reason = "coverage_failed"
    elif not screening_ok:
        qualification = "FAIL"
        reason = "bias_screening_failed"
    elif dropped:
        qualification = "FAIL"
        reason = "pseudoinverse_full_rank_forbidden"
    elif converged < n:
        qualification = "FAIL"
        reason = "nonconverged_replicas"
    else:
        qualification = "PASS"
        reason = "pre_registered_criteria"
    return {
        "cell_id": cell["cell_id"],
        "condition_id": cell["condition_id"],
        "survey_mode": cell["survey_mode"],
        "amplitude": cell["amplitude"],
        "n_replicas": n,
        "n_converged": converged,
        "n_nonconverged": n - converged,
        "n_dropped_mode": dropped,
        "bias": bias,
        "rms": rms,
        "mean_sigma": mean_sigma,
        "pull_mean": pull_mean,
        "pull_width": pull_width,
        "n_covered_95": covered,
        "coverage": covered / n,
        "coverage_clopper_pearson": [lower, upper],
        "pulls_qualified": pulls_ok,
        "coverage_qualified": coverage_status,
        "bias_screening": screening,
        "bias_screening_pass": screening_ok,
        "rank_status": rank_status,
        "qualification": qualification,
        "reason": reason,
    }


def _three_layer_fields(*, toy_qualified: bool, qualification: str) -> dict[str, object]:
    """Physical FASER oracle stays false under toy_uniform_By_independent_v1."""
    return {
        "solver_implementation_fixed": bool(solver_implementation_fixed()),
        "solver_implementation_contract": solver_implementation_contract(),
        "common_track_solver_qualified_under_toy_model": bool(toy_qualified) and qualification == "PASS",
        "alignment_oracle_qualified_for_physical_FASER": False,
        "reconstruction_provenance": RECONSTRUCTION_PROVENANCE,
        "calypso_physical_replicas_available": False,
    }


def qualify(cell_reports: Sequence[Mapping[str, Any]], *, fd_ok: bool, n_independent: int, forbidden_accessed: bool) -> dict[str, object]:
    if forbidden_accessed:
        return {
            **_three_layer_fields(toy_qualified=False, qualification="FAIL"),
            "alignment_oracle_qualified": False,
            "qualification": "FAIL",
            "reason": "forbidden_asset_accessed",
        }
    statuses = [str(report["qualification"]) for report in cell_reports]
    if not cell_reports or n_independent < MIN_REPLICAS_FOR_PASS or any(status == "UNKNOWN" for status in statuses):
        return {
            **_three_layer_fields(toy_qualified=False, qualification="UNKNOWN"),
            "alignment_oracle_qualified": "unknown",
            "qualification": "UNKNOWN",
            "reason": "insufficient_independent_statistics",
            "n_independent_replicas": n_independent,
        }
    required = {
        "all_cells_pass": all(status == "PASS" for status in statuses),
        "coverage_qualified": all(report.get("coverage_qualified") == "PASS" for report in cell_reports),
        "pulls_qualified": all(bool(report.get("pulls_qualified")) for report in cell_reports),
        "bias_qualified": all(bool(report.get("bias_screening_pass", True)) for report in cell_reports),
        "fd_stability_qualified": bool(fd_ok),
        "convergence_qualified": all(int(report.get("n_nonconverged", 1)) == 0 for report in cell_reports),
        "no_forbidden_assets_accessed": True,
    }
    passed = all(required.values())
    qualification = "PASS" if passed else "FAIL"
    return {
        **_three_layer_fields(toy_qualified=passed, qualification=qualification),
        "alignment_oracle_qualified": False,
        "qualification": qualification,
        "reason": "all_pre_registered_gates" if passed else "one_or_more_gates_failed",
        "n_independent_replicas": n_independent,
        "gates": required,
        "ml_alignment_eval_authorized": False,
        "wb82_smoke_used_as_qualification": False,
    }
