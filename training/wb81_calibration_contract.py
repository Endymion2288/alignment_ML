"""Workbook-81a contract: bounded residual on frozen W64 energy.  No training."""

from __future__ import annotations

import inspect
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from models.route_energy import (
    CANONICAL_ENERGY_VERSION,
    RouteEnergyRecord,
    RouteEnergyTable,
    assign_from_energy_table,
    route_kind,
)
from training.explicit_route_energy import PhysicalRoute, refuse_forbidden_experiment_path


DELTA_MAX = 0.25
UNMATCHED_PENALTY = -1.0
IDENTITY_METRIC_KEYS = (
    "complete_track_efficiency",
    "complete_fake_rate",
    "complete_track_purity",
    "all_route_purity",
)
MARGIN_BINS = (
    ("negative", -math.inf, 0.0),
    ("near_zero", 0.0, 1.0),
    ("comfortable", 1.0, math.inf),
)
WB81A_FORBIDDEN_NEEDLES = (
    "00800_00849",
    "mc24_100116",
    "mc24_100117",
    "00350_00399",
    "source_diversity_blind",
    "arm_b_checkpoint",
    "arm_c_checkpoint",
    "hybrid_route_energy",
)
EXPECTED_FLIPPABLE = {
    "family1_ds100043_100044": 919,
    "family2_ds100047_100048": 458,
}
BIN_WEIGHTS = {
    "negative": 2.0,
    "near_zero": 1.0,
    "comfortable": 0.0,
}
ACCEPT_EFF_GAIN = 0.01
EXPECTED_HOLDOUT_COUNTS = {
    "family1_ds100043_100044": {
        "n_scored_events": 3360,
        "complete_truth_chains": 6789,
        "t_neg": 1071,
        "t_near": 5718,
        "t_comf": 0,
        "f_a": 77,
        "correct_complete_routes": 5718,
        "fake_complete_routes": 77,
        "selected_complete_routes": 5795,
        "selected_fragment_routes": 0,
    },
    "family2_ds100047_100048": {
        "n_scored_events": 1680,
        "complete_truth_chains": 3248,
        "t_neg": 463,
        "t_near": 2785,
        "t_comf": 0,
        "f_a": 56,
        "correct_complete_routes": 2785,
        "fake_complete_routes": 56,
        "selected_complete_routes": 2841,
        "selected_fragment_routes": 0,
    },
}


def refuse_wb81a_path(path: object) -> None:
    text = str(path)
    refuse_forbidden_experiment_path(text)
    for needle in WB81A_FORBIDDEN_NEEDLES:
        if needle in text:
            raise ValueError(f"WB81a refuses development / blind / B / C / hybrid path: {text}")


def margin_bin(inclusion_gap: float) -> str:
    number = float(inclusion_gap)
    for name, low, high in MARGIN_BINS:
        if number >= low and number < high:
            return name
    return "comfortable"


def zero_theta_delta(n_routes: int) -> np.ndarray:
    """θ = 0 ⇒ ΔU = 0 for every 2/3/4 route."""
    return np.zeros(int(n_routes), dtype=np.float64)


def bounded_physics_delta(raw_score: Sequence[float]) -> np.ndarray:
    """Same bound for every route length.  Not a complete-only head."""
    values = np.asarray(raw_score, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("raw residual scores must be a vector")
    if values.size and not np.isfinite(values).all():
        raise ValueError("raw residual scores must be finite")
    return (DELTA_MAX * np.tanh(values)).astype(np.float64, copy=False)


def apply_calibrated_utilities(u_w64: Sequence[float], delta: Sequence[float]) -> np.ndarray:
    """U_new = U_W64 + ΔU.  Length-agnostic.  No sigmoid / clip / logit."""
    base = np.asarray(u_w64, dtype=np.float64)
    correction = np.asarray(delta, dtype=np.float64)
    if base.shape != correction.shape or base.ndim != 1:
        raise ValueError("U_W64 and ΔU must be aligned vectors")
    if base.size and (not np.isfinite(base).all() or not np.isfinite(correction).all()):
        raise ValueError("U_W64 and ΔU must be finite")
    if correction.size and float(np.max(np.abs(correction))) > DELTA_MAX + 1.0e-12:
        raise ValueError(f"ΔU exceeds frozen bound {DELTA_MAX}")
    return (base + correction).astype(np.float64, copy=False)


def calibrated_energy_table(
    routes: Sequence[PhysicalRoute],
    u_w64: Sequence[float],
    delta: Sequence[float],
    unmatched_penalty: float = UNMATCHED_PENALTY,
) -> RouteEnergyTable:
    utilities = apply_calibrated_utilities(u_w64, delta)
    corrections = np.asarray(delta, dtype=np.float64)
    records = []
    for route_id, (route, energy, correction) in enumerate(zip(routes, utilities, corrections)):
        records.append(
            RouteEnergyRecord(
                route_id=route_id,
                endpoints=tuple((int(station), int(index)) for station, index in zip(route.stations, route.node_indices)),
                energy=float(energy),
                n_stations=len(route.stations),
                kind=route_kind(len(route.stations)),
                contract=CANONICAL_ENERGY_VERSION,
                correction=float(correction),
            )
        )
    return RouteEnergyTable(
        version=CANONICAL_ENERGY_VERSION,
        unmatched_penalty=float(unmatched_penalty),
        records=tuple(records),
    )


def selected_mask(table: RouteEnergyTable) -> tuple[int, ...]:
    flags = assign_from_energy_table(table).selected
    return tuple(index for index, flag in enumerate(flags.tolist()) if flag)


def compare_identity(
    selected_a: Sequence[int],
    selected_new: Sequence[int],
    metrics_a: Mapping[str, object],
    metrics_new: Mapping[str, object],
) -> dict[str, object]:
    same_selected = tuple(int(value) for value in selected_a) == tuple(int(value) for value in selected_new)
    metric_pairs = {}
    metrics_equal = True
    for key in IDENTITY_METRIC_KEYS:
        left = metrics_a.get(key)
        right = metrics_new.get(key)
        metric_pairs[key] = {"A": left, "new": right}
        if left is None and right is None:
            continue
        if left is None or right is None or float(left) != float(right):
            metrics_equal = False
    passed = bool(same_selected and metrics_equal)
    return {
        "passed": passed,
        "selected_identical": same_selected,
        "metrics_identical": metrics_equal,
        "n_selected_A": len(selected_a),
        "n_selected_new": len(selected_new),
        "metrics": metric_pairs,
    }


def contract_source_guards() -> dict[str, bool]:
    """Static guards: one correction map, bound, no probability decode, no complete-only branch."""
    apply_src = inspect.getsource(apply_calibrated_utilities)
    bound_src = inspect.getsource(bounded_physics_delta)
    zero_src = inspect.getsource(zero_theta_delta)
    combined = apply_src + bound_src + zero_src
    return {
        "apply_has_no_n_stations_branch": "n_stations" not in apply_src and "complete" not in apply_src,
        "bound_has_no_complete_only_branch": "n_stations" not in bound_src and "== 4" not in bound_src,
        "no_sigmoid": "sigmoid(" not in combined,
        "no_clip": "clip(" not in combined,
        "no_logit": "logit(" not in combined,
        "delta_max_frozen": DELTA_MAX == 0.25,
    }


def checksum_holdout(family: str, observed: Mapping[str, int]) -> dict[str, object]:
    expected = EXPECTED_HOLDOUT_COUNTS[family]
    mismatches = {key: {"expected": expected[key], "observed": observed.get(key)} for key in expected if observed.get(key) != expected[key]}
    return {"passed": not mismatches, "expected": expected, "observed": dict(observed), "mismatches": mismatches}


def verify_wb81a_checksum_file(path: object, family: str) -> dict[str, object]:
    """Read a WB81a checksum JSON.  Do not recompute or rewrite the counts."""
    refuse_wb81a_path(path)
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    counts = payload.get("boundary_counts") or {}
    observed = {
        "t_neg": counts.get("t_neg"),
        "t_near": counts.get("t_near"),
        "t_comf": counts.get("t_comf"),
        "f_a": counts.get("f_a"),
        "flippable_neg_under_delta_max": counts.get("flippable_neg_under_delta_max"),
    }
    expected = {
        "t_neg": EXPECTED_HOLDOUT_COUNTS[family]["t_neg"],
        "t_near": EXPECTED_HOLDOUT_COUNTS[family]["t_near"],
        "t_comf": EXPECTED_HOLDOUT_COUNTS[family]["t_comf"],
        "f_a": EXPECTED_HOLDOUT_COUNTS[family]["f_a"],
        "flippable_neg_under_delta_max": EXPECTED_FLIPPABLE[family],
    }
    mismatches = {key: {"expected": expected[key], "observed": observed[key]} for key in expected if observed[key] != expected[key]}
    if mismatches:
        raise ValueError(f"WB81a checksum mismatch for {family}: {mismatches}")
    return {"passed": True, "family": family, "expected": expected, "observed": observed}


def fake_hard_gate(fake_new: float | None, fake_a: float | None) -> bool:
    """Zero-slack complete-fake gate.  Missing rates fail closed."""
    if fake_new is None or fake_a is None:
        return False
    return float(fake_new) <= float(fake_a)


def calibration_acceptable(eff_new: float | None, fake_new: float | None, eff_a: float | None, fake_a: float | None) -> bool:
    if None in (eff_new, fake_new, eff_a, fake_a):
        return False
    return float(fake_new) <= float(fake_a) and float(eff_new) >= float(eff_a) + ACCEPT_EFF_GAIN
