"""Workbook-62 post-training comparison for the frozen max-reduction control.

Does not pick a new reduction, weight, or operating point.  Train movement
uses the workbook-61 frozen margin bins.  Transfer numbers are recorded
only after the single pre-registered evaluation.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from training.dustbin_aware_route_margin import PACKING_MARGIN
from training.route_utility_identifiability import _median


TWIN_FAMILIES = {
    "draw_00": ("iteration_00_draw_00", "iteration_00_draw_00_plus_common"),
    "draw_01": ("iteration_00_draw_01", "iteration_00_draw_01_plus_common"),
    "hard_s3_ry": ("iteration_00_hard_s3_ry", "iteration_00_hard_s3_ry_plus_common"),
}


def route_identity(row: Mapping[str, Any]) -> tuple[object, ...]:
    if row.get("truth_id") is None:
        raise ValueError("matched comparison requires truth_id")
    return (
        str(row.get("payload_id")),
        int(row["run_id"]),
        int(row["event_id"]),
        int(row["truth_id"]),
    )


def _apply_event_max(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, object]]:
    from collections import defaultdict

    grouped: dict[tuple[object, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row.get("payload_id"), row.get("run_id"), row.get("event_id"))].append(row)
    enriched: list[dict[str, object]] = []
    for event_rows in grouped.values():
        losses = [float(row.get("dustbin_aware_margin_loss") or 0.0) for row in event_rows]
        max_index = max(
            range(len(event_rows)),
            key=lambda index: (losses[index], -float(event_rows[index].get("production_margin") or 0.0)),
        )
        for index, row in enumerate(event_rows):
            enriched.append({**dict(row), "is_event_max_dustbin": bool(index == max_index)})
    return enriched


def match_control_candidate(
    control_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
) -> dict[str, object]:
    """Join workbook-59 and the new control on the same train overlay identities."""
    control = _apply_event_max(control_rows)
    candidate = {route_identity(row): row for row in candidate_rows}
    paired: list[dict[str, object]] = []
    missing = 0
    for row in control:
        key = route_identity(row)
        other = candidate.get(key)
        if other is None:
            missing += 1
            continue
        paired.append(
            {
                "payload_id": row.get("payload_id"),
                "run_id": row.get("run_id"),
                "event_id": row.get("event_id"),
                "truth_id": row.get("truth_id"),
                "control_bin": row.get("margin_bin"),
                "candidate_bin": other.get("margin_bin"),
                "control_short_bin": row.get("short_station_bin"),
                "candidate_short_bin": other.get("short_station_bin"),
                "control_margin": row.get("production_margin"),
                "candidate_margin": other.get("production_margin"),
                "control_u_truth": row.get("u_truth"),
                "candidate_u_truth": other.get("u_truth"),
                "control_u_fragment": row.get("u_best_solver_fragment"),
                "candidate_u_fragment": other.get("u_best_solver_fragment"),
                "control_competitor_n_stations": row.get("competitor_n_stations"),
                "candidate_competitor_n_stations": other.get("competitor_n_stations"),
                "control_is_event_max": bool(row.get("is_event_max_dustbin")),
                "control_production_winner": bool(row.get("production_fragment_winner")),
            }
        )
    short_boundary = [
        row
        for row in paired
        if row["control_short_bin"] in {"hard", "near_boundary"}
    ]
    event_max_short = [
        row
        for row in short_boundary
        if row["control_is_event_max"]
    ]
    four_hard = [row for row in paired if row["control_bin"] == "hard" and row["control_short_bin"] not in {"hard", "near_boundary"}]
    return {
        "paired": int(len(paired)),
        "missing_on_candidate": int(missing),
        "control_short_boundary": int(len(short_boundary)),
        "control_event_max_short": int(len(event_max_short)),
        "short_boundary_now_easy": int(sum(1 for row in short_boundary if row["candidate_bin"] == "easy")),
        "short_boundary_still_near": int(sum(1 for row in short_boundary if row["candidate_bin"] == "near_boundary")),
        "short_boundary_now_hard": int(sum(1 for row in short_boundary if row["candidate_bin"] == "hard")),
        "event_max_short_now_easy": int(sum(1 for row in event_max_short if row["candidate_bin"] == "easy")),
        "event_max_short_still_near": int(sum(1 for row in event_max_short if row["candidate_bin"] == "near_boundary")),
        "event_max_short_now_hard": int(sum(1 for row in event_max_short if row["candidate_bin"] == "hard")),
        "event_max_short_candidate_margin_median": _median_or_none(
            [row["candidate_margin"] for row in event_max_short]
        ),
        "short_boundary_candidate_margin_median": _median_or_none(
            [row["candidate_margin"] for row in short_boundary]
        ),
        "four_station_hard_now_easy": int(sum(1 for row in four_hard if row["candidate_bin"] == "easy")),
        "four_station_hard_control": int(len(four_hard)),
        "rows": paired,
    }


def _median_or_none(values: Sequence[object]) -> float | None:
    kept = [float(value) for value in values if value is not None]
    return None if not kept else float(_median(kept))


def classify_train_reduction_effect(matched: Mapping[str, Any], *, margin: float = PACKING_MARGIN) -> str:
    """Train-only: did the frozen event-max short cases leave the margin?"""
    n = int(matched.get("control_event_max_short") or 0)
    if n <= 0:
        return "no_control_event_max_short"
    median_delta = matched.get("event_max_short_candidate_margin_median")
    if median_delta is None:
        return "undefined"
    if float(median_delta) >= float(margin):
        return "short_event_max_pushed_off_boundary"
    return "short_event_max_still_on_boundary"


def classify_stop_reason(
    *,
    gates_passed: bool,
    train_effect: str,
) -> dict[str, object]:
    if gates_passed:
        return {
            "continue_to_15d_relative_wls": True,
            "stop_weighting_and_operating_point_rescue": False,
            "failure_class": None,
            "next_step": "open_15d_route_selected_delta_t_ij_wls",
        }
    if train_effect == "short_event_max_pushed_off_boundary":
        failure = "training_domain_coverage_limitation"
    else:
        failure = "objective_reduction_failure"
    return {
        "continue_to_15d_relative_wls": False,
        "stop_weighting_and_operating_point_rescue": True,
        "failure_class": failure,
        "next_step": "stop_weighting_and_operating_point_rescue",
        "train_reduction_effect": train_effect,
    }


def compact_reduction_row(row: Mapping[str, Any]) -> dict[str, object]:
    return {
        "payload_id": row.get("payload_id"),
        "payload_family": row.get("payload_family"),
        "run_id": row.get("run_id"),
        "event_id": row.get("event_id"),
        "truth_id": row.get("truth_id"),
        "origin_run_id": row.get("origin_run_id"),
        "selected": row.get("selected"),
        "u_truth": row.get("u_truth"),
        "u_best_solver_fragment": row.get("u_best_solver_fragment"),
        "production_margin": row.get("production_margin"),
        "margin_bin": row.get("margin_bin"),
        "short_station_bin": row.get("short_station_bin"),
        "competitor_n_stations": row.get("competitor_n_stations"),
        "n_complete_truth_in_event": row.get("n_complete_truth_in_event"),
        "dustbin_aware_margin_loss": row.get("dustbin_aware_margin_loss"),
        "production_fragment_winner": row.get("production_fragment_winner"),
        "production_by_length": row.get("production_by_length"),
    }


def bin_transition_counts(matched_rows: Sequence[Mapping[str, Any]], key_control: str, key_candidate: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in matched_rows:
        label = f"{row.get(key_control)}->{row.get(key_candidate)}"
        counts[label] = counts.get(label, 0) + 1
    return counts

