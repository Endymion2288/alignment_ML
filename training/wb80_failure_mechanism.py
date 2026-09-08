"""Workbook-80 route-level audit and hybrid features.  No V5A / Arm C / 00350."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from evaluation.route_counterfactuals import inclusion_gap
from models.hybrid_route_energy import HYBRID_FEATURE_DIM, hybrid_feature_names
from models.route_energy import assign_from_energy_table
from training.explicit_route_energy import (
    PhysicalRoute,
    energy_table_from_utilities,
    refuse_forbidden_experiment_path,
    route_feature_matrix,
    w64_edge_logit_matrix,
)
from training.geometry_aware_transformer import TransformerGraph


NEAR_DUSTBIN_ABS = 1.0
W64_MARGIN_BINS = (
    ("negative", -np.inf, 0.0),
    ("near_zero", 0.0, 1.0),
    ("comfortable", 1.0, np.inf),
)


def hybrid_feature_matrix(
    graph: TransformerGraph,
    routes: Sequence[PhysicalRoute],
    base_logits_by_owner_row: Mapping[tuple[int, int], float],
) -> np.ndarray:
    """Physical 54-D row plus per-hop raw W64 logits, their sum, and hop count."""
    physical = route_feature_matrix(graph, routes)
    logits = w64_edge_logit_matrix(graph, routes, base_logits_by_owner_row)
    hop_count = np.asarray([len(route.score_edge_indices) for route in routes], dtype=np.float64).reshape(-1, 1)
    extra = np.concatenate(
        (logits, np.sum(logits, axis=1, keepdims=True), hop_count),
        axis=1,
    )
    matrix = np.concatenate((physical, extra), axis=1)
    if matrix.shape[1] != HYBRID_FEATURE_DIM:
        raise RuntimeError("hybrid feature width drifted")
    if not np.isfinite(matrix).all():
        raise ValueError("hybrid features must be finite")
    return matrix.astype(np.float64, copy=False)


def _bin_name(value: float) -> str:
    number = float(value)
    for name, low, high in W64_MARGIN_BINS:
        if number >= low and number < high:
            return name
    return "comfortable"


def _safe_corr(left: np.ndarray, right: np.ndarray) -> float | None:
    if left.size < 3 or right.size != left.size:
        return None
    if np.std(left) < 1.0e-12 or np.std(right) < 1.0e-12:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _safe_spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    if left.size < 3 or right.size != left.size:
        return None
    left_rank = np.argsort(np.argsort(left))
    right_rank = np.argsort(np.argsort(right))
    return _safe_corr(left_rank.astype(np.float64), right_rank.astype(np.float64))


def _ratio(numerator: int, denominator: int) -> float | None:
    if int(denominator) <= 0:
        return None
    return float(numerator) / float(denominator)


def event_assignment_masks(
    routes: Sequence[PhysicalRoute],
    utilities: Mapping[str, np.ndarray],
    unmatched_penalty: float,
) -> dict[str, np.ndarray]:
    selected = {}
    for name, values in utilities.items():
        table = energy_table_from_utilities(routes, values, unmatched_penalty=unmatched_penalty)
        selected[name] = np.asarray(assign_from_energy_table(table).selected, dtype=bool)
    return selected


def truth_margin_rows(
    routes: Sequence[PhysicalRoute],
    utilities: Mapping[str, np.ndarray],
    selected: Mapping[str, np.ndarray],
    unmatched_penalty: float,
) -> list[dict[str, object]]:
    """Inclusion-gap / rival-margin rows for complete truth routes only."""
    tables = {
        name: energy_table_from_utilities(routes, values, unmatched_penalty=unmatched_penalty)
        for name, values in utilities.items()
    }
    rows = []
    for route_id, route in enumerate(routes):
        if not (route.truth_consistent and len(route.stations) == 4):
            continue
        row: dict[str, object] = {
            "route_id": int(route_id),
            "n_stations": 4,
            "truth_consistent": True,
        }
        for name, table in tables.items():
            energy = float(utilities[name][route_id])
            gap = inclusion_gap(table, route_id)
            row[f"U_{name}"] = energy
            row[f"inclusion_gap_{name}"] = float(gap.inclusion_gap)
            row[f"single_rival_margin_{name}"] = gap.single_rival_margin
            row[f"near_dustbin_{name}"] = bool(abs(energy) < NEAR_DUSTBIN_ABS)
            row[f"selected_{name}"] = bool(selected[name][route_id])
        row["w64_margin_bin"] = _bin_name(float(row["inclusion_gap_A"]))
        rows.append(row)
    return rows


def selected_route_rows(
    routes: Sequence[PhysicalRoute],
    utilities: Mapping[str, np.ndarray],
    selected: Mapping[str, np.ndarray],
) -> list[dict[str, object]]:
    rows = []
    for route_id, route in enumerate(routes):
        flags = {name: bool(mask[route_id]) for name, mask in selected.items()}
        if not any(flags.values()):
            continue
        rows.append(
            {
                "route_id": int(route_id),
                "n_stations": int(len(route.stations)),
                "truth_consistent": bool(route.truth_consistent),
                "selected": flags,
                "utilities": {name: float(values[route_id]) for name, values in utilities.items()},
                "near_dustbin": {
                    name: bool(abs(float(values[route_id])) < NEAR_DUSTBIN_ABS) for name, values in utilities.items()
                },
            }
        )
    return rows


def _empty_length_bucket() -> dict[str, int]:
    return {
        "selected": 0,
        "truth_consistent_selected": 0,
        "fake_selected": 0,
        "near_dustbin_selected": 0,
    }


def stratify_holdout(
    length_rows: Sequence[Mapping[str, object]],
    truth_rows: Sequence[Mapping[str, object]],
    utilities_holdout: Mapping[str, list[float]],
) -> dict[str, object]:
    by_length = {arm: {2: _empty_length_bucket(), 3: _empty_length_bucket(), 4: _empty_length_bucket()} for arm in ("A", "B", "C")}
    for row in length_rows:
        n_stations = int(row["n_stations"])
        if n_stations not in (2, 3, 4):
            continue
        truth = bool(row["truth_consistent"])
        selected = row["selected"]
        near = row["near_dustbin"]
        for arm in ("A", "B", "C"):
            if not selected.get(arm):
                continue
            bucket = by_length[arm][n_stations]
            bucket["selected"] += 1
            bucket["truth_consistent_selected"] += int(truth)
            bucket["fake_selected"] += int(not truth)
            bucket["near_dustbin_selected"] += int(bool(near.get(arm)))

    margin = {
        name: {
            "n_complete_truth": 0,
            "selected_A": 0,
            "selected_B": 0,
            "selected_C": 0,
            "mean_inclusion_gap_A": None,
            "mean_inclusion_gap_B": None,
            "mean_inclusion_gap_C": None,
        }
        for name, _, _ in W64_MARGIN_BINS
    }
    gap_lists = {name: {"A": [], "B": [], "C": []} for name, _, _ in W64_MARGIN_BINS}
    for row in truth_rows:
        bin_name = str(row["w64_margin_bin"])
        block = margin[bin_name]
        block["n_complete_truth"] += 1
        for arm in ("A", "B", "C"):
            if bool(row.get(f"selected_{arm}")):
                block[f"selected_{arm}"] += 1
            gap_lists[bin_name][arm].append(float(row[f"inclusion_gap_{arm}"]))
    for bin_name, block in margin.items():
        for arm in ("A", "B", "C"):
            values = gap_lists[bin_name][arm]
            block[f"mean_inclusion_gap_{arm}"] = None if not values else float(np.mean(values))
            block[f"recall_{arm}"] = _ratio(int(block[f"selected_{arm}"]), int(block["n_complete_truth"]))

    truth_u = {arm: np.asarray(utilities_holdout[arm], dtype=np.float64) for arm in utilities_holdout}
    # Caller passes only complete-truth utilities for correlation.
    correlation = {
        "pearson_A_B": _safe_corr(truth_u.get("A", np.array([])), truth_u.get("B", np.array([]))),
        "pearson_A_C": _safe_corr(truth_u.get("A", np.array([])), truth_u.get("C", np.array([]))),
        "spearman_A_B": _safe_spearman(truth_u.get("A", np.array([])), truth_u.get("B", np.array([]))),
        "spearman_A_C": _safe_spearman(truth_u.get("A", np.array([])), truth_u.get("C", np.array([]))),
        "n_complete_truth": int(truth_u["A"].size) if "A" in truth_u else 0,
    }
    return {
        "by_length": by_length,
        "by_w64_margin": margin,
        "near_dustbin_abs": NEAR_DUSTBIN_ABS,
        "complete_truth_energy_correlation": correlation,
        "hybrid_feature_names": list(hybrid_feature_names()),
    }


def interpret_ablation(
    holdout_eval: Mapping[str, Mapping[str, object]],
    same_family_eval: Mapping[str, Mapping[str, object]] | None,
    correlation: Mapping[str, object],
) -> dict[str, object]:
    """Pre-registered reading rules.  Missing rates stay None and fail closed."""

    def _metric(block: Mapping[str, object], arm: str, name: str) -> float | None:
        value = (block.get(arm) or {}).get(name)
        return None if value is None else float(value)

    def _close(left: float | None, right: float | None, tol: float) -> bool | None:
        if left is None or right is None:
            return None
        return abs(left - right) <= tol

    def _le(left: float | None, right: float | None) -> bool | None:
        if left is None or right is None:
            return None
        return left <= right

    a_eff = _metric(holdout_eval, "A", "complete_track_efficiency")
    b_eff = _metric(holdout_eval, "B", "complete_track_efficiency")
    c_eff = _metric(holdout_eval, "C", "complete_track_efficiency")
    a_fake = _metric(holdout_eval, "A", "complete_fake_rate")
    b_fake = _metric(holdout_eval, "B", "complete_fake_rate")
    c_fake = _metric(holdout_eval, "C", "complete_fake_rate")
    c_recovers_a = _close(c_eff, a_eff, 0.02) and _close(c_fake, a_fake, 0.02)
    c_beats_a = (
        c_eff is not None
        and a_eff is not None
        and c_fake is not None
        and a_fake is not None
        and c_eff >= a_eff + 0.02
        and c_fake <= a_fake
    )
    b_tracks_a_ranking = (
        correlation.get("spearman_A_B") is not None and float(correlation["spearman_A_B"]) >= 0.80
    )
    in_sample = None
    if same_family_eval is not None:
        in_sample = {
            "delta_eff_b_minus_a": (
                None
                if _metric(same_family_eval, "A", "complete_track_efficiency") is None
                or _metric(same_family_eval, "B", "complete_track_efficiency") is None
                else float(
                    _metric(same_family_eval, "B", "complete_track_efficiency")
                    - _metric(same_family_eval, "A", "complete_track_efficiency")
                )
            )
        }
    physical_lacks_in_sample = (
        in_sample is not None
        and in_sample["delta_eff_b_minus_a"] is not None
        and float(in_sample["delta_eff_b_minus_a"]) <= -0.03
    )
    return {
        "w64_superiority_from_edge_information": c_recovers_a is True and b_tracks_a_ranking is False,
        "w64_superiority_from_solver_aggregation": b_tracks_a_ranking is True and c_recovers_a is not True,
        "physical_features_lack_information": physical_lacks_in_sample is True and c_recovers_a is True,
        "hybrid_worth_entering": c_beats_a is True,
        "hybrid_not_justified_beyond_w64_sum": c_recovers_a is True and c_beats_a is not True,
        "scorer_cannot_use_w64_logits": c_recovers_a is False
        and _close(c_eff, b_eff, 0.02) is True
        and _le(c_eff, a_eff) is True,
        "pre_registered_tolerances": {
            "recover_eff_abs": 0.02,
            "recover_fake_abs": 0.02,
            "beat_eff": 0.02,
            "ranking_spearman": 0.80,
            "in_sample_eff_gap": -0.03,
        },
        "observed": {
            "holdout_eff": {"A": a_eff, "B": b_eff, "C": c_eff},
            "holdout_fake": {"A": a_fake, "B": b_fake, "C": c_fake},
            "c_recovers_a": c_recovers_a,
            "c_beats_a": c_beats_a,
            "b_tracks_a_ranking": b_tracks_a_ranking,
            "same_family": in_sample,
        },
    }

