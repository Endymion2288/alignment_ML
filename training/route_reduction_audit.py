"""Workbook-61 train-only weighting / reduction feasibility.

Does not train, does not retune the operating point, and does not read
transfer numbers to pick a weight or a reduction.  Production competitors
stay in ``_route_hypotheses`` (``U > 0``).  Margin bins use the frozen
production convention ``Δ = U_truth - max(U_fragment, 0)`` and
``margin = 1``.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence

from baselines.route_assignment import Route
from training.dustbin_aware_route_margin import PACKING_MARGIN, relu_gap
from training.route_operating_audit import DUSTBIN_UTILITY
from training.route_utility_identifiability import _median, classify_route_topology
from training.solver_hard_negative_audit import (
    UNMATCHED_PENALTY,
    _endpoints,
    attach_solver_hard_negative,
    overlapping_production_hypotheses,
)


MARGIN_BINS = ("hard", "near_boundary", "easy")
LENGTHS = (2, 3, 4)
# Pre-registered definition of "almost no short-fragment boundary cases".
# Frozen before the audit runs; not taken from transfer.
SHORT_SPARSE_ROUTE_FRACTION = 0.01
WB59_EDGE_WEIGHT = 1.0
WB59_PACKING_WEIGHT = 0.07061055340401011
WB59_DUSTBIN_WEIGHT = 0.05
WB59_GAUGE_WEIGHT = 1.0


def production_margin(u_truth: float | None, u_fragment: float | None) -> float | None:
    """``U_truth - max(U_fragment, 0)``.  Missing truth is undefined."""
    if u_truth is None:
        return None
    competitor = DUSTBIN_UTILITY if u_fragment is None else max(float(u_fragment), float(DUSTBIN_UTILITY))
    return float(u_truth) - float(competitor)


def classify_margin_bin(delta: float | None, *, margin: float = PACKING_MARGIN) -> str:
    if delta is None:
        return "undefined"
    if float(delta) <= 0.0:
        return "hard"
    if float(delta) < float(margin):
        return "near_boundary"
    return "easy"


def competitor_length_class(n_stations: int | None) -> int | None:
    if n_stations is None:
        return None
    value = int(n_stations)
    return value if value in LENGTHS else None


def best_production_by_length(
    hypotheses: Sequence[Route],
    truth_endpoints: Sequence[tuple[int, int]],
) -> dict[int, dict[str, object]]:
    """Strongest overlapping production hypothesis in each 2/3/4-station bin."""
    grouped: dict[int, list[Route]] = {length: [] for length in LENGTHS}
    for route in overlapping_production_hypotheses(hypotheses, truth_endpoints):
        length = competitor_length_class(len(route.endpoints))
        if length is None:
            continue
        grouped[length].append(route)
    result: dict[int, dict[str, object]] = {}
    for length, routes in grouped.items():
        if not routes:
            continue
        winner = max(routes, key=lambda route: float(route.utility))
        endpoints = tuple((int(station), int(index)) for station, index in winner.endpoints)
        result[length] = {
            "n_stations": length,
            "utility": float(winner.utility),
            "topology": classify_route_topology(endpoints),
            "n_overlapping": int(len(routes)),
        }
    return result


def best_miner_by_length(rivals: Sequence[Mapping[str, Any]]) -> dict[int, dict[str, object]]:
    grouped: dict[int, list[Mapping[str, Any]]] = {length: [] for length in LENGTHS}
    for rival in rivals:
        if not rival.get("threshold_feasible"):
            continue
        length = competitor_length_class(rival.get("n_stations"))
        if length is None:
            continue
        grouped[length].append(rival)
    result: dict[int, dict[str, object]] = {}
    for length, items in grouped.items():
        if not items:
            continue
        winner = max(items, key=lambda row: float(row["utility"]))
        result[length] = {
            "n_stations": length,
            "utility": float(winner["utility"]),
            "topology": winner.get("topology"),
            "n_overlapping": int(len(items)),
        }
    return result


def logit_margin_gradient_norm(n_truth_edges: int, n_competitor_edges: int, *, shared_edges: int = 0) -> float:
    """L2 of ``∂relu(U_comp + m - U_truth) / ∂logit`` when the ReLU is active.

    Identity Platt makes packing utility a sum of logits plus ``n * penalty``,
    so each exclusive edge contributes ±1.  Shared edges cancel.
    """
    exclusive = max(int(n_truth_edges) + int(n_competitor_edges) - 2 * int(shared_edges), 0)
    return math.sqrt(float(exclusive))


def _length_payload(
    by_length: Mapping[int, Mapping[str, Any]],
    u_truth: float | None,
    *,
    margin: float,
) -> dict[str, object]:
    payload: dict[str, object] = {}
    for length in LENGTHS:
        record = by_length.get(length)
        if record is None:
            payload[str(length)] = {
                "present": False,
                "utility": None,
                "delta": None,
                "bin": "absent",
                "topology": None,
                "loss": 0.0,
            }
            continue
        utility = float(record["utility"])
        delta = production_margin(u_truth, utility)
        payload[str(length)] = {
            "present": True,
            "utility": utility,
            "delta": delta,
            "bin": classify_margin_bin(delta, margin=margin),
            "topology": record.get("topology"),
            "n_overlapping": record.get("n_overlapping"),
            "loss": 0.0 if u_truth is None else relu_gap(max(utility, DUSTBIN_UTILITY), float(u_truth), margin),
        }
    return payload


def attach_reduction_audit(
    row: Mapping[str, Any],
    **kwargs: Any,
) -> dict[str, object]:
    """Workbook-60 production attach plus length bins and reduction shares."""
    attached = attach_solver_hard_negative(row, **kwargs)
    u_truth = attached.get("u_truth")
    margin = float(kwargs.get("margin", PACKING_MARGIN))
    production_u = attached.get("u_best_solver_fragment")
    miner_u = attached.get("u_best_miner_rival")
    delta = production_margin(None if u_truth is None else float(u_truth), production_u)
    miner_delta = production_margin(None if u_truth is None else float(u_truth), miner_u)
    production_by_length = best_production_by_length(
        kwargs["hypotheses"],
        _endpoints(row["endpoints"]),
    )
    miner_rivals = []
    strongest = attached.get("miner_strongest")
    # Rebuild length table from the same miner the attach used.
    from training.solver_hard_negative_audit import miner_rivals as _miner_rivals

    miner_rivals = _miner_rivals(
        kwargs["pair_tables"],
        _endpoints(row["endpoints"]),
        threshold=float(kwargs.get("threshold", 0.001)),
        unmatched_penalty=float(kwargs.get("unmatched_penalty", UNMATCHED_PENALTY)),
    )
    miner_by_length = best_miner_by_length(miner_rivals)
    competitor = attached.get("solver_competitor") or {}
    n_comp = competitor.get("n_stations")
    n_shared = len(competitor.get("shared_endpoints") or [])
    n_comp_edges = 0 if n_comp is None else max(int(n_comp) - 1, 0)
    active = float(attached["dustbin_aware_margin_loss"]) > 0.0
    grad = 0.0 if not active else logit_margin_gradient_norm(3, n_comp_edges, shared_edges=n_shared)
    n_event = max(int(attached["n_complete_truth_in_event"]), 1)
    origin = row.get("origin_signature") or attached.get("origin_signature")
    origin_run = None if not origin else int(origin[0][0])
    return {
        **attached,
        "production_margin": delta,
        "miner_production_margin": miner_delta,
        "margin_bin": classify_margin_bin(delta, margin=margin),
        "miner_margin_bin": classify_margin_bin(miner_delta, margin=margin),
        "competitor_n_stations": n_comp,
        "production_by_length": _length_payload(production_by_length, None if u_truth is None else float(u_truth), margin=margin),
        "miner_by_length": _length_payload(miner_by_length, None if u_truth is None else float(u_truth), margin=margin),
        "short_station_bin": _short_bin(_length_payload(production_by_length, None if u_truth is None else float(u_truth), margin=margin)),
        "dustbin_mean_share": float(attached["dustbin_aware_margin_loss"]) / float(n_event),
        "packing_mean_share": float(attached["workbook56_packing_loss"]) / float(n_event),
        "dustbin_logit_grad_norm": grad,
        "dustbin_mean_grad_share": grad / float(n_event),
        "origin_run_id": origin_run,
        "payload_family": _payload_family(str(attached.get("payload_id") or row.get("payload_id") or "")),
        "miner_strongest_n_stations": None if strongest is None else strongest.get("n_stations"),
    }


def _short_bin(by_length: Mapping[str, Mapping[str, Any]]) -> str:
    ranks = {"hard": 0, "near_boundary": 1, "easy": 2, "absent": 3}
    two = str(by_length["2"]["bin"])
    three = str(by_length["3"]["bin"])
    return two if ranks[two] <= ranks[three] else three


def _payload_family(payload_id: str) -> str:
    name = str(payload_id)
    if name.endswith("_plus_common"):
        return name[: -len("_plus_common")]
    return name


def _apply_event_reduction(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                row.get("payload_id"),
                row.get("run_id"),
                row.get("event_id"),
            )
        ].append(row)
    enriched: list[dict[str, object]] = []
    for event_rows in grouped.values():
        dustbin = [float(row["dustbin_aware_margin_loss"]) for row in event_rows]
        event_mean = float(sum(dustbin) / max(len(dustbin), 1))
        event_max = 0.0 if not dustbin else float(max(dustbin))
        max_index = max(range(len(event_rows)), key=lambda index: (dustbin[index], -float(event_rows[index].get("production_margin") or 0.0)))
        for index, row in enumerate(event_rows):
            enriched.append(
                {
                    **dict(row),
                    "event_dustbin_mean": event_mean,
                    "event_dustbin_max": event_max,
                    "is_event_max_dustbin": bool(index == max_index),
                    "event_max_would_keep": bool(index == max_index),
                }
            )
    return enriched


def _count_bins(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, int]:
    return dict(Counter(str(row.get(key) or "undefined") for row in rows))


def _sum(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    return float(sum(float(row.get(key) or 0.0) for row in rows))


def summarize_reduction_audit(rows: Sequence[Mapping[str, Any]]) -> dict[str, object]:
    enriched = _apply_event_reduction(rows)
    n = len(enriched)
    events = {
        (row.get("payload_id"), row.get("run_id"), row.get("event_id"))
        for row in enriched
    }
    hard = [row for row in enriched if row.get("margin_bin") == "hard"]
    near = [row for row in enriched if row.get("margin_bin") == "near_boundary"]
    easy = [row for row in enriched if row.get("margin_bin") == "easy"]
    boundary = hard + near
    short_hard = [
        row
        for row in enriched
        if row.get("short_station_bin") == "hard"
    ]
    short_near = [
        row
        for row in enriched
        if row.get("short_station_bin") == "near_boundary"
    ]
    short_boundary = short_hard + short_near
    long_boundary = [
        row
        for row in enriched
        if str((row.get("production_by_length") or {}).get("4", {}).get("bin")) in {"hard", "near_boundary"}
    ]
    event_max = [row for row in enriched if row.get("is_event_max_dustbin")]
    short_event_max = [
        row
        for row in event_max
        if competitor_length_class(row.get("competitor_n_stations")) in {2, 3}
        or row.get("short_station_bin") in {"hard", "near_boundary"}
        and row.get("is_event_max_dustbin")
    ]
    short_near_is_max = [row for row in short_boundary if row.get("is_event_max_dustbin")]
    mean_mass = _sum(enriched, "dustbin_mean_share")
    max_mass = float(sum(float(row["event_dustbin_max"]) for row in event_max))
    boundary_mean = _sum(boundary, "dustbin_mean_share")
    boundary_max = float(sum(float(row["dustbin_aware_margin_loss"]) for row in event_max if row.get("margin_bin") in {"hard", "near_boundary"}))
    hard_mean = _sum(hard, "dustbin_mean_share")
    near_mean = _sum(near, "dustbin_mean_share")
    short_mean = _sum(short_boundary, "dustbin_mean_share")
    short_max = float(sum(float(row["dustbin_aware_margin_loss"]) for row in short_near_is_max))
    mean_grad = _sum(enriched, "dustbin_mean_grad_share")
    boundary_grad = _sum(boundary, "dustbin_mean_grad_share")
    hard_grad = _sum(hard, "dustbin_mean_grad_share")
    near_grad = _sum(near, "dustbin_mean_grad_share")
    multiplicity = [
        int(row["n_complete_truth_in_event"])
        for row in enriched
        if row.get("margin_bin") in {"hard", "near_boundary"}
    ]
    by_length_bin: dict[str, dict[str, int]] = {}
    for length in LENGTHS:
        by_length_bin[str(length)] = dict(
            Counter(str((row.get("production_by_length") or {}).get(str(length), {}).get("bin") or "absent") for row in enriched)
        )
    deltas_by_length: dict[str, list[float]] = {str(length): [] for length in LENGTHS}
    loss_by_length: dict[str, list[float]] = {str(length): [] for length in LENGTHS}
    for row in enriched:
        for length in LENGTHS:
            item = (row.get("production_by_length") or {}).get(str(length)) or {}
            if item.get("delta") is not None:
                deltas_by_length[str(length)].append(float(item["delta"]))
            if item.get("present"):
                loss_by_length[str(length)].append(float(item.get("loss") or 0.0))
    source_counts = dict(Counter(str(row.get("origin_run_id")) for row in short_boundary))
    payload_counts = dict(Counter(str(row.get("payload_id")) for row in short_boundary))
    family_counts = dict(Counter(str(row.get("payload_family")) for row in short_boundary))
    strongest_length = dict(Counter(str(row.get("competitor_n_stations") or "none") for row in boundary))
    event_max_length = dict(Counter(str(row.get("competitor_n_stations") or "none") for row in event_max if float(row.get("dustbin_aware_margin_loss") or 0.0) > 0.0))
    return {
        "complete_truth_chains": int(n),
        "n_events": int(len(events)),
        "margin_bin_counts": _count_bins(enriched, "margin_bin"),
        "hard": int(len(hard)),
        "near_boundary": int(len(near)),
        "easy": int(len(easy)),
        "production_margin_median": None if not enriched else _median(
            [float(row["production_margin"]) for row in enriched if row.get("production_margin") is not None]
        ),
        "hard_production_winners": int(sum(1 for row in hard if row.get("production_fragment_winner"))),
        "short_station_hard": int(len(short_hard)),
        "short_station_near_boundary": int(len(short_near)),
        "short_station_boundary": int(len(short_boundary)),
        "four_station_boundary": int(len(long_boundary)),
        "short_boundary_route_fraction": None if not n else float(len(short_boundary) / n),
        "short_to_long_boundary_ratio": (
            None if not long_boundary else float(len(short_boundary) / len(long_boundary))
        ),
        "production_bin_by_competitor_length": by_length_bin,
        "strongest_competitor_length_on_boundary": strongest_length,
        "event_max_competitor_length_when_loss_positive": event_max_length,
        "n_event_max_is_short_boundary": int(len(short_near_is_max)),
        "median_event_multiplicity_on_boundary": None if not multiplicity else float(_median(multiplicity)),
        "mean_dustbin_loss": None if not n else float(_sum(enriched, "dustbin_aware_margin_loss") / n),
        "mean_packing_loss": None if not n else float(_sum(enriched, "workbook56_packing_loss") / n),
        "boundary_share_of_mean_dustbin_loss": None if mean_mass <= 0.0 else float(boundary_mean / mean_mass),
        "boundary_share_of_max_dustbin_loss": None if max_mass <= 0.0 else float(boundary_max / max_mass),
        "hard_share_of_mean_dustbin_loss": None if mean_mass <= 0.0 else float(hard_mean / mean_mass),
        "near_boundary_share_of_mean_dustbin_loss": None if mean_mass <= 0.0 else float(near_mean / mean_mass),
        "hard_share_of_mean_logit_grad": None if mean_grad <= 0.0 else float(hard_grad / mean_grad),
        "near_boundary_share_of_mean_logit_grad": None if mean_grad <= 0.0 else float(near_grad / mean_grad),
        "short_boundary_share_of_mean_dustbin_loss": None if mean_mass <= 0.0 else float(short_mean / mean_mass),
        "short_boundary_share_of_max_dustbin_loss": None if max_mass <= 0.0 else float(short_max / max_mass),
        "boundary_share_of_mean_logit_grad": None if mean_grad <= 0.0 else float(boundary_grad / mean_grad),
        "mean_dilutes_boundary": bool(
            (boundary_mean + 1.0e-12) < boundary_max
            and (None if not multiplicity else float(_median(multiplicity))) is not None
            and float(_median(multiplicity) or 0.0) >= 2.0
        ),
        "length_margin_median": {
            length: None if not values else float(_median(values))
            for length, values in deltas_by_length.items()
        },
        "length_dustbin_loss_mean": {
            length: None if not values else float(sum(values) / len(values))
            for length, values in loss_by_length.items()
        },
        "length_dustbin_loss_sum": {
            length: float(sum(values)) for length, values in loss_by_length.items()
        },
        "short_boundary_by_origin_run": source_counts,
        "short_boundary_by_payload": payload_counts,
        "short_boundary_by_family": family_counts,
        "frozen_aux_weights": {
            "edge": WB59_EDGE_WEIGHT,
            "packing_route_competition": WB59_PACKING_WEIGHT,
            "dustbin_aware_route_margin": WB59_DUSTBIN_WEIGHT,
            "gauge_twin": WB59_GAUGE_WEIGHT,
        },
        "weighted_mean_contributions": {
            "packing_route_competition": WB59_PACKING_WEIGHT * (0.0 if not n else _sum(enriched, "workbook56_packing_loss") / n),
            "dustbin_aware_route_margin": WB59_DUSTBIN_WEIGHT * (0.0 if not n else _sum(enriched, "dustbin_aware_margin_loss") / n),
            "dustbin_aware_from_hard_near_after_mean": WB59_DUSTBIN_WEIGHT * (0.0 if not n else boundary_mean / max(len(events), 1)),
        },
        "payloads": {},
    }


def recommend_next_objective(train_summary: Mapping[str, Any]) -> dict[str, object]:
    """Train-only rule.  Transfer counts are not inputs."""
    n = int(train_summary.get("complete_truth_chains") or 0)
    short_hard = int(train_summary.get("short_station_hard") or 0)
    short_near = int(train_summary.get("short_station_near_boundary") or 0)
    short = short_hard + short_near
    long_b = int(train_summary.get("four_station_boundary") or 0)
    short_frac = 0.0 if n <= 0 else short / float(n)
    short_max = int(train_summary.get("n_event_max_is_short_boundary") or 0)
    diluted = bool(train_summary.get("mean_dilutes_boundary"))
    absent = short == 0
    sparse = (not absent) and short_hard == 0 and short_frac < float(SHORT_SPARSE_ROUTE_FRACTION)
    max_would_hit_short = short_max > 0
    if absent or sparse:
        next_step = "stop_curriculum_domain_coverage_limitation"
        designed = None
        reason = (
            "train_has_no_2_or_3_station_production_competitor_inside_frozen_margin"
            if absent
            else "train_short_fragment_boundary_cases_are_sparse_under_pre_registered_1pct_rule"
        )
    elif not max_would_hit_short:
        next_step = "stop_curriculum_domain_coverage_limitation"
        designed = None
        reason = "short_boundary_exists_but_is_never_the_per_event_max_so_max_reduction_would_not_train_it"
    elif diluted:
        next_step = "design_hard_aware_max_reduction_control"
        designed = "per_event_max_route_competition_reduction"
        reason = "short_boundary_is_present_and_mean_dilutes_the_event_max_dangerous_route"
    else:
        next_step = "stop_reduction_is_not_the_limiter"
        designed = None
        reason = "short_boundary_present_but_mean_reduction_already_keeps_their_mass"
    return {
        "continue_to_15d_relative_wls": False,
        "open_15d_wls_authorized_by_this_audit": False,
        "new_checkpoint_authorized": False,
        "transfer_used_to_pick_reduction_or_weight": False,
        "train_complete_truth_chains": n,
        "train_short_station_hard": short_hard,
        "train_short_station_near_boundary": short_near,
        "train_four_station_boundary": long_b,
        "train_short_boundary_route_fraction": short_frac,
        "train_short_boundary_is_absent": absent,
        "train_short_boundary_is_sparse": sparse,
        "train_short_boundary_is_event_max": bool(max_would_hit_short),
        "train_mean_dilutes_boundary": diluted,
        "pre_register_hard_aware_reduction_control": designed == "per_event_max_route_competition_reduction",
        "next_step": next_step,
        "next_objective_if_a_later_control_is_opened": designed,
        "reason": reason,
        "reduction_if_opened": "max_over_complete_truth_routes_in_event",
        "do_not_change_dustbin_aware_target": True,
        "do_not_retune_operating_point": True,
        "do_not_increase_dustbin_weight": True,
        "architecture_change_allowed": False,
        "short_sparse_route_fraction_threshold": SHORT_SPARSE_ROUTE_FRACTION,
    }
