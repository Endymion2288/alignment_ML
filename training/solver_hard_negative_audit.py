"""Workbook-60 solver-generated hard-negative mining audit.

Does not train, does not change the operating point, and does not unroll a
new solver.  Production competitors come from
``baselines.route_assignment._route_hypotheses`` — the admitted hypothesis
set of ``adjacent_contiguous_unit_capacity_set_packing`` (``U > 0`` only).
Workbook-59 training-miner rivals are reconstructed from the same physical
adjacent edges the miner enumerates; they are never used as a substitute
for the production hypothesis set.
"""

from __future__ import annotations

import inspect
from collections import Counter
from typing import Any, Mapping, Sequence

from baselines.route_assignment import (
    Route,
    RouteAssignmentConfig,
    _matrix_edge_lookup,
    _route_hypotheses,
    adjacent_station_pairs,
)
from datasets.root_loader import EventTracklets
from evaluation.route_metrics import _unique_truth_by_station
from training.dustbin_aware_route_margin import PACKING_MARGIN, relu_gap
from training.gauge_consistent_route import packing_route_competition_loss
from training.route_operating_audit import (
    ADJACENT_PAIRS,
    DUSTBIN_UTILITY,
    STATION_PATH,
    classify_selected_route,
    complete_route_packing_utility,
)
from training.route_utility_identifiability import (
    classify_route_topology,
    _median,
)


THRESHOLD = 0.001
UNMATCHED_PENALTY = -1.0
WB59_CHECKPOINT_SHA256 = "6565d028e01e561105d4ce467d10d5c82e5a2435daad6a69c47a07920defb5dc"

TOPOLOGY_FAMILIES = (
    "two_station_01",
    "two_station_12",
    "two_station_23",
    "three_station_prefix",
    "three_station_suffix",
    "four_station_mixed",
    "four_station_other_truth",
    "other",
    "none",
)
COMPOSITION_FAMILIES = (
    "own_truth_partial",
    "other_truth_consistent_partial",
    "other_truth_complete",
    "mixed_route",
    "cross_truth_endpoint_conflict",
    "fake_endpoint_conflict",
    "none",
)


def _endpoints(items: Sequence[Mapping[str, Any] | tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    result: list[tuple[int, int]] = []
    for item in items:
        if isinstance(item, Mapping):
            result.append((int(item["station"]), int(item["index"])))
        else:
            result.append((int(item[0]), int(item[1])))
    return tuple(result)


def _endpoint_payload(endpoints: Sequence[tuple[int, int]]) -> list[dict[str, int]]:
    return [{"station": int(station), "index": int(index)} for station, index in endpoints]


def physical_utility(records: Sequence[Mapping[str, Any]], unmatched_penalty: float) -> float:
    probabilities = [float(record["calibrated_probability"]) for record in records]
    return complete_route_packing_utility(
        probabilities, unmatched_penalty, n_stations=len(records) + 1
    )


def classify_hard_negative(
    endpoints: Sequence[tuple[int, int]],
    event: EventTracklets,
    unique_by_index: Mapping[int, Mapping[int, int]],
    *,
    truth_endpoints: Sequence[tuple[int, int]],
    truth_id: int | None,
) -> dict[str, object]:
    """Topology + origin/truth composition of one solver-feasible competitor."""
    topology = classify_route_topology(endpoints)
    selected = classify_selected_route(event, endpoints, unique_by_index)
    if topology == "four_station":
        topology = "four_station_other_truth" if selected["truth_consistent"] else "four_station_mixed"
    truth_set = {(int(station), int(index)) for station, index in truth_endpoints}
    competitor_set = {(int(station), int(index)) for station, index in endpoints}
    shared = sorted(truth_set.intersection(competitor_set))
    competitor_truth_id = selected.get("truth_id")
    own_partial = bool(
        selected["truth_consistent"]
        and truth_id is not None
        and competitor_truth_id == int(truth_id)
        and len(endpoints) < 4
    )
    other_consistent = bool(
        selected["truth_consistent"]
        and (truth_id is None or competitor_truth_id != int(truth_id))
    )
    if own_partial:
        composition = "own_truth_partial"
    elif other_consistent and len(endpoints) < 4:
        composition = "other_truth_consistent_partial"
    elif other_consistent:
        composition = "other_truth_complete"
    elif selected["has_fake_endpoint"] and shared:
        composition = "fake_endpoint_conflict"
    elif not selected["truth_consistent"] and topology == "four_station_mixed":
        composition = "mixed_route"
    elif shared and not selected["truth_consistent"]:
        composition = "cross_truth_endpoint_conflict"
    else:
        composition = str(selected["kind"])
    station_pairs = [
        f"{int(left[0])}->{int(right[0])}"
        for left, right in zip(endpoints, endpoints[1:])
    ]
    return {
        **selected,
        "topology": topology,
        "composition": composition,
        "shares_endpoint": bool(shared),
        "shared_endpoints": _endpoint_payload(shared),
        "shared_stations": [int(station) for station, _ in shared],
        "station_pairs": station_pairs,
        "involves_2_3": "2->3" in station_pairs,
        "involves_s3": any(int(station) == 3 for station, _ in endpoints),
        "n_stations": int(len(endpoints)),
        "cross_truth_endpoint_conflict": bool(
            shared
            and (
                composition
                in {
                    "cross_truth_endpoint_conflict",
                    "fake_endpoint_conflict",
                    "mixed_route",
                    "other_truth_consistent_partial",
                    "other_truth_complete",
                }
            )
        ),
    }


def enumerate_miner_routes(
    pair_tables: Mapping[tuple[int, int], Mapping[tuple[int, int], Mapping[str, Any]]],
) -> list[tuple[tuple[tuple[int, int], ...], list[Mapping[str, Any]]]]:
    """Same 2/3/4-station contiguous enumerator the workbook-59 miner uses."""
    tables = {pair: dict(pair_tables.get(pair) or {}) for pair in ADJACENT_PAIRS}
    routes: list[tuple[tuple[tuple[int, int], ...], list[Mapping[str, Any]]]] = []
    for pair, table in tables.items():
        for (source, target), record in table.items():
            routes.append((((int(pair[0]), int(source)), (int(pair[1]), int(target))), [record]))
    for (source_zero, target_one), first in tables[(0, 1)].items():
        for (source_one, target_two), second in tables[(1, 2)].items():
            if int(source_one) != int(target_one):
                continue
            prefix = (
                ((0, int(source_zero)), (1, int(target_one)), (2, int(target_two))),
                [first, second],
            )
            routes.append(prefix)
            for (source_two, target_three), third in tables[(2, 3)].items():
                if int(source_two) != int(target_two):
                    continue
                routes.append(
                    (
                        (
                            (0, int(source_zero)),
                            (1, int(target_one)),
                            (2, int(target_two)),
                            (3, int(target_three)),
                        ),
                        [first, second, third],
                    )
                )
    for (source_one, target_two), second in tables[(1, 2)].items():
        for (source_two, target_three), third in tables[(2, 3)].items():
            if int(source_two) != int(target_two):
                continue
            routes.append(
                (
                    ((1, int(source_one)), (2, int(target_two)), (3, int(target_three))),
                    [second, third],
                )
            )
    return routes


def miner_rivals(
    pair_tables: Mapping[tuple[int, int], Mapping[tuple[int, int], Mapping[str, Any]]],
    truth_endpoints: Sequence[tuple[int, int]],
    *,
    threshold: float = THRESHOLD,
    unmatched_penalty: float = UNMATCHED_PENALTY,
) -> list[dict[str, object]]:
    """Workbook-59 mined negatives: overlapping threshold-feasible physical routes."""
    truth = _endpoints(truth_endpoints)
    truth_set = set(truth)
    rivals: list[dict[str, object]] = []
    for endpoints, records in enumerate_miner_routes(pair_tables):
        if endpoints == truth:
            continue
        if not truth_set.intersection(endpoints):
            continue
        feasible = all(float(record["calibrated_probability"]) >= float(threshold) for record in records)
        utility = physical_utility(records, unmatched_penalty)
        rivals.append(
            {
                "endpoints": _endpoint_payload(endpoints),
                "utility": float(utility),
                "n_stations": int(len(endpoints)),
                "threshold_feasible": bool(feasible),
                "topology": classify_route_topology(endpoints),
            }
        )
    return rivals


def workbook59_miner_strongest(rivals: Sequence[Mapping[str, Any]]) -> dict[str, object] | None:
    feasible = [row for row in rivals if row.get("threshold_feasible")]
    if not feasible:
        return None
    return max(feasible, key=lambda row: float(row["utility"]))


def production_hypotheses(
    event: EventTracklets,
    station_matrices: Mapping[tuple[int, int], tuple[object, Sequence[object]]],
    config: RouteAssignmentConfig,
    complete_route_scores: Mapping[tuple[int, ...], float] | None = None,
) -> list[Route]:
    """Call the production enumerator.  Only ``U > 0`` hypotheses are returned."""
    pairs = adjacent_station_pairs(tuple(int(station) for station in config.station_path))
    lookups = {
        pair: _matrix_edge_lookup(
            station_matrices[pair][0],
            station_matrices[pair][1],
            float(config.score_threshold_by_pair[pair]),
        )
        for pair in pairs
    }
    return _route_hypotheses(
        event,
        tuple(int(station) for station in config.station_path),
        lookups,
        float(config.unmatched_penalty),
        int(config.maximum_hypotheses),
        complete_route_scores,
        config.complete_route_score_threshold,
        config.complete_route_score_composition,
        config.complete_route_context_weight,
    )


def overlapping_production_hypotheses(
    hypotheses: Sequence[Route],
    truth_endpoints: Sequence[tuple[int, int]],
) -> list[Route]:
    truth = _endpoints(truth_endpoints)
    truth_set = set(truth)
    result: list[Route] = []
    for route in hypotheses:
        endpoints = tuple((int(station), int(index)) for station, index in route.endpoints)
        if endpoints == truth:
            continue
        if not truth_set.intersection(endpoints):
            continue
        result.append(route)
    return result


def _route_key(endpoints: Sequence[tuple[int, int]] | Sequence[Mapping[str, Any]]) -> tuple[tuple[int, int], ...]:
    return _endpoints(endpoints)


def attach_solver_hard_negative(
    row: Mapping[str, Any],
    *,
    event: EventTracklets,
    unique_by_index: Mapping[int, Mapping[int, int]],
    pair_tables: Mapping[tuple[int, int], Mapping[tuple[int, int], Mapping[str, Any]]],
    hypotheses: Sequence[Route],
    selected_routes: Sequence[Route],
    n_complete_truth_in_event: int,
    margin: float = PACKING_MARGIN,
    unmatched_penalty: float = UNMATCHED_PENALTY,
    threshold: float = THRESHOLD,
) -> dict[str, object]:
    """Attach production-solver competitors and workbook-59 miner coverage."""
    truth_endpoints = _endpoints(row["endpoints"])
    truth_id = None if row.get("truth_id") is None else int(row["truth_id"])
    u_truth = None if row.get("complete_truth_route_utility") is None else float(row["complete_truth_route_utility"])
    production = overlapping_production_hypotheses(hypotheses, truth_endpoints)
    selected_blockers = [
        route
        for route in selected_routes
        if set((int(station), int(index)) for station, index in route.endpoints).intersection(truth_endpoints)
        and tuple((int(station), int(index)) for station, index in route.endpoints) != truth_endpoints
    ]
    best_production = None if not production else max(production, key=lambda route: float(route.utility))
    best_selected = None if not selected_blockers else max(selected_blockers, key=lambda route: float(route.utility))
    winner = best_production
    miner = miner_rivals(
        pair_tables,
        truth_endpoints,
        threshold=threshold,
        unmatched_penalty=unmatched_penalty,
    )
    strongest_miner = workbook59_miner_strongest(miner)
    miner_utilities = [float(item["utility"]) for item in miner if item["threshold_feasible"]]
    miner_u = None if strongest_miner is None else float(strongest_miner["utility"])
    winner_payload = None
    winner_in_miner = False
    winner_is_strongest = False
    winner_physical = None
    if winner is not None:
        winner_endpoints = tuple((int(station), int(index)) for station, index in winner.endpoints)
        hops: list[Mapping[str, Any]] = []
        complete = True
        for left, right in zip(winner_endpoints, winner_endpoints[1:]):
            pair = (int(left[0]), int(right[0]))
            record = (pair_tables.get(pair) or {}).get((int(left[1]), int(right[1])))
            if record is None:
                complete = False
                break
            hops.append(record)
        if getattr(winner, "complete_route_score", None) is not None:
            # Production packing consumed L_corrected for this 4-station
            # hypothesis.  Do not reconstruct utility from adjacent edges.
            winner_physical = float(winner.utility) - 1.0e-9 * (len(winner_endpoints) - 1)
        elif complete:
            winner_physical = physical_utility(hops, unmatched_penalty)
        else:
            winner_physical = float(winner.utility) - 1.0e-9 * (len(winner_endpoints) - 1)
        classified = classify_hard_negative(
            winner_endpoints,
            event,
            unique_by_index,
            truth_endpoints=truth_endpoints,
            truth_id=truth_id,
        )
        winner_payload = {
            **classified,
            "endpoints": _endpoint_payload(winner_endpoints),
            "solver_utility": float(winner.utility),
            "physical_utility": winner_physical,
            "was_selected_by_packing": bool(
                best_selected is not None
                and tuple((int(station), int(index)) for station, index in best_selected.endpoints)
                == winner_endpoints
            ),
            "selected_blocker_n_stations": None if best_selected is None else int(len(best_selected.endpoints)),
            "selected_blocker_utility": None if best_selected is None else float(best_selected.utility),
        }
        feasible_keys = {
            _route_key(item["endpoints"])
            for item in miner
            if item["threshold_feasible"]
        }
        winner_in_miner = winner_endpoints in feasible_keys
        winner_is_strongest = bool(
            strongest_miner is not None
            and _route_key(strongest_miner["endpoints"]) == winner_endpoints
        )
    miner_loss = 0.0 if u_truth is None else relu_gap(
        DUSTBIN_UTILITY if miner_u is None else miner_u,
        float(u_truth),
        margin,
    )
    dustbin_aware_loss = 0.0 if u_truth is None else relu_gap(
        DUSTBIN_UTILITY if miner_u is None else max(float(miner_u), float(DUSTBIN_UTILITY)),
        float(u_truth),
        margin,
    )
    oracle_u = DUSTBIN_UTILITY if winner_physical is None else max(float(winner_physical), float(DUSTBIN_UTILITY))
    oracle_loss = 0.0 if u_truth is None else relu_gap(oracle_u, float(u_truth), margin)
    solver_beats_truth = bool(
        winner_physical is not None
        and u_truth is not None
        and float(winner_physical) >= float(u_truth)
        and float(winner_physical) > float(DUSTBIN_UTILITY)
    )
    enumerated_best = None if not miner_utilities else float(max(miner_utilities))
    return {
        **dict(row),
        "u_truth": u_truth,
        "n_complete_truth_in_event": int(n_complete_truth_in_event),
        "n_production_overlapping_hypotheses": int(len(production)),
        "n_selected_overlapping_blockers": int(len(selected_blockers)),
        "n_miner_threshold_feasible_rivals": int(len(miner_utilities)),
        "u_best_solver_fragment": None if winner_physical is None else float(winner_physical),
        "u_best_solver_fragment_solver_scale": None if winner is None else float(winner.utility),
        "u_best_miner_rival": miner_u,
        "u_best_enumerated_including_nonadmitted": enumerated_best,
        "delta_u_solver_minus_truth": (
            None if winner_physical is None or u_truth is None else float(winner_physical) - float(u_truth)
        ),
        "production_fragment_winner": bool(solver_beats_truth and not row.get("selected")),
        "workbook59_truth_lt_fragment": bool(
            enumerated_best is not None and u_truth is not None and float(u_truth) < float(enumerated_best)
        ),
        "solver_competitor": winner_payload,
        "miner_strongest": strongest_miner,
        "winner_in_mined_negative_set": bool(winner_in_miner),
        "winner_selected_as_strongest_miner": bool(winner_is_strongest),
        "workbook56_packing_loss": float(miner_loss),
        "dustbin_aware_margin_loss": float(dustbin_aware_loss),
        "solver_winner_oracle_loss": float(oracle_loss),
        "missed_training_signal": bool(oracle_loss > 0.0 and dustbin_aware_loss <= 0.0),
        "oracle_loss_exceeds_miner": bool(oracle_loss > dustbin_aware_loss + 1.0e-12),
        "source_id": row.get("sample_id"),
    }


def summarize_hard_negatives(rows: Sequence[Mapping[str, Any]]) -> dict[str, object]:
    unselected = [row for row in rows if not row.get("selected")]
    fragment_wins = [row for row in rows if row.get("production_fragment_winner")]
    truth_lt = [row for row in rows if row.get("workbook59_truth_lt_fragment")]
    selected = [row for row in rows if row.get("selected")]

    def _count(subset: Sequence[Mapping[str, Any]], key: str) -> dict[str, int]:
        return dict(Counter(str((row.get("solver_competitor") or {}).get(key) or "none") for row in subset))

    def _bool_frac(subset: Sequence[Mapping[str, Any]], key: str) -> float | None:
        if not subset:
            return None
        return float(sum(1 for row in subset if row.get(key)) / len(subset))

    utilities = [float(row["u_truth"]) for row in rows if row.get("u_truth") is not None]
    solver_u = [float(row["u_best_solver_fragment"]) for row in fragment_wins if row.get("u_best_solver_fragment") is not None]
    miner_losses = [float(row["dustbin_aware_margin_loss"]) for row in fragment_wins]
    oracle_losses = [float(row["solver_winner_oracle_loss"]) for row in fragment_wins]
    by_source = Counter(str(row.get("source_id")) for row in fragment_wins)
    by_multiplicity = Counter(int(row.get("n_complete_truth_in_event") or 0) for row in fragment_wins)
    s3 = sum(1 for row in fragment_wins if (row.get("solver_competitor") or {}).get("involves_2_3"))
    return {
        "complete_truth_chains": int(len(rows)),
        "selected": int(len(selected)),
        "unselected": int(len(unselected)),
        "production_fragment_winners": int(len(fragment_wins)),
        "workbook59_truth_lt_fragment": int(len(truth_lt)),
        "u_truth_median": None if not utilities else float(_median(utilities)),
        "u_truth_fraction_nonpositive": (
            None if not utilities else float(sum(1 for value in utilities if value <= 0.0) / len(utilities))
        ),
        "fragment_winner_u_median": None if not solver_u else float(_median(solver_u)),
        "fragment_winner_topology": _count(fragment_wins, "topology"),
        "fragment_winner_composition": _count(fragment_wins, "composition"),
        "truth_lt_fragment_topology": _count(truth_lt, "topology"),
        "truth_lt_fragment_composition": _count(truth_lt, "composition"),
        "fragment_winner_involves_2_3": int(s3),
        "fragment_winner_by_source": dict(by_source),
        "fragment_winner_by_event_multiplicity": {str(key): int(value) for key, value in sorted(by_multiplicity.items())},
        "winner_in_mined_negative_set": int(sum(1 for row in fragment_wins if row.get("winner_in_mined_negative_set"))),
        "winner_selected_as_strongest_miner": int(
            sum(1 for row in fragment_wins if row.get("winner_selected_as_strongest_miner"))
        ),
        "missed_training_signal": int(sum(1 for row in fragment_wins if row.get("missed_training_signal"))),
        "oracle_loss_exceeds_miner": int(sum(1 for row in fragment_wins if row.get("oracle_loss_exceeds_miner"))),
        "fraction_winner_in_miner": _bool_frac(fragment_wins, "winner_in_mined_negative_set"),
        "fraction_winner_is_strongest": _bool_frac(fragment_wins, "winner_selected_as_strongest_miner"),
        "fraction_missed_training_signal": _bool_frac(fragment_wins, "missed_training_signal"),
        "mean_dustbin_aware_loss_on_fragment_wins": None if not miner_losses else float(sum(miner_losses) / len(miner_losses)),
        "mean_oracle_loss_on_fragment_wins": None if not oracle_losses else float(sum(oracle_losses) / len(oracle_losses)),
        "payloads": {},
    }


def _origin_signature(row: Mapping[str, Any]) -> tuple[tuple[int, ...], ...]:
    origin = row.get("origin_signature")
    if origin is None:
        raise ValueError("twin alignment requires origin signatures")
    return tuple(tuple(int(value) for value in item) for item in origin)


def twin_alignment_key(row: Mapping[str, Any]) -> tuple[object, ...]:
    """Same physical track across a gauge twin.

    Overlay ``origin_tracklet_id`` is almost always the station index
    ``(0, 1, 2, 3)``, so ``origin_signature`` only identifies the original
    ``(run, event)``.  Multiple complete truth routes share that signature
    inside one payload; ``truth_id`` is the MC particle that survives the
    common SE(3) transform.  Synthetic ``run_id`` / ``event_id`` are not
    part of the key: chart and twin pack the same origin events differently.
    """
    if row.get("truth_id") is None:
        raise ValueError("twin alignment requires truth_id to split multi-track origin events")
    return (_origin_signature(row), int(row["truth_id"]))


def align_origin_twins(
    chart_rows: Sequence[Mapping[str, Any]],
    twin_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, object]], dict[str, int]]:
    """Same-origin chart/twin join for one relative family."""
    chart = {twin_alignment_key(row): row for row in chart_rows}
    twin = {twin_alignment_key(row): row for row in twin_rows}
    if len(chart) != len(chart_rows) or len(twin) != len(twin_rows):
        raise ValueError("duplicate origin+truth_id inside a gauge twin payload")
    shared = set(chart) & set(twin)
    aligned: list[dict[str, object]] = []
    for key in sorted(shared):
        left = chart[key]
        right = twin[key]
        u_left = left.get("u_truth")
        u_right = right.get("u_truth")
        f_left = left.get("u_best_solver_fragment")
        f_right = right.get("u_best_solver_fragment")
        aligned.append(
            {
                "source_id": left.get("sample_id") or left.get("source_id"),
                "run_id": int(left["run_id"]),
                "event_id": int(left["event_id"]),
                "truth_id": int(left["truth_id"]),
                "origin_signature": left.get("origin_signature"),
                "chart_selected": bool(left.get("selected")),
                "twin_selected": bool(right.get("selected")),
                "chart_u_truth": u_left,
                "twin_u_truth": u_right,
                "delta_u_truth_twin_minus_chart": (
                    None if u_left is None or u_right is None else float(u_right) - float(u_left)
                ),
                "chart_u_fragment": f_left,
                "twin_u_fragment": f_right,
                "delta_u_fragment_twin_minus_chart": (
                    None if f_left is None or f_right is None else float(f_right) - float(f_left)
                ),
                "chart_competitor_topology": (left.get("solver_competitor") or {}).get("topology"),
                "twin_competitor_topology": (right.get("solver_competitor") or {}).get("topology"),
                "chart_competitor_composition": (left.get("solver_competitor") or {}).get("composition"),
                "twin_competitor_composition": (right.get("solver_competitor") or {}).get("composition"),
                "chart_winner_is_strongest": bool(left.get("winner_selected_as_strongest_miner")),
                "twin_winner_is_strongest": bool(right.get("winner_selected_as_strongest_miner")),
            }
        )
    stats = {
        "aligned": int(len(aligned)),
        "unmatched_chart": int(len(set(chart) - set(twin))),
        "unmatched_twin": int(len(set(twin) - set(chart))),
        "chart_rows": int(len(chart_rows)),
        "twin_rows": int(len(twin_rows)),
        "alignment_key": "origin_signature+truth_id",
    }
    return aligned, stats


def summarize_twin_efficiency_gap(aligned: Sequence[Mapping[str, Any]]) -> dict[str, object]:
    chart_only = [row for row in aligned if row["chart_selected"] and not row["twin_selected"]]
    twin_only = [row for row in aligned if row["twin_selected"] and not row["chart_selected"]]
    both = [row for row in aligned if row["chart_selected"] and row["twin_selected"]]
    lost = chart_only

    def _median_key(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        return None if not values else float(_median(values))

    truth_drop = [row for row in lost if row.get("delta_u_truth_twin_minus_chart") is not None and float(row["delta_u_truth_twin_minus_chart"]) < 0.0]
    fragment_rise = [
        row
        for row in lost
        if row.get("delta_u_fragment_twin_minus_chart") is not None
        and float(row["delta_u_fragment_twin_minus_chart"]) > 0.0
    ]
    truth_drop_dominates = 0
    fragment_rise_dominates = 0
    for row in lost:
        d_truth = row.get("delta_u_truth_twin_minus_chart")
        d_frag = row.get("delta_u_fragment_twin_minus_chart")
        if d_truth is None:
            continue
        truth_mag = abs(min(0.0, float(d_truth)))
        frag_mag = 0.0 if d_frag is None else max(0.0, float(d_frag))
        if truth_mag > frag_mag:
            truth_drop_dominates += 1
        elif frag_mag > truth_mag:
            fragment_rise_dominates += 1
    return {
        "aligned_origin_routes": int(len(aligned)),
        "both_selected": int(len(both)),
        "chart_selected_twin_lost": int(len(chart_only)),
        "twin_selected_chart_lost": int(len(twin_only)),
        "lost_delta_u_truth_median": _median_key(lost, "delta_u_truth_twin_minus_chart"),
        "lost_delta_u_fragment_median": _median_key(lost, "delta_u_fragment_twin_minus_chart"),
        "lost_truth_utility_decreased": int(len(truth_drop)),
        "lost_fragment_utility_increased": int(len(fragment_rise)),
        "lost_truth_drop_dominates": int(truth_drop_dominates),
        "lost_fragment_rise_dominates": int(fragment_rise_dominates),
        "attribution": (
            "truth_utility_drop"
            if truth_drop_dominates > fragment_rise_dominates
            else "fragment_utility_rise"
            if fragment_rise_dominates > truth_drop_dominates
            else "mixed_or_undefined"
        ),
    }


def recommend_next_objective(train_summary: Mapping[str, Any]) -> dict[str, object]:
    """Train-only mechanism rule.  Transfer numbers are not used to pick weights."""
    n_winner = int(train_summary.get("production_fragment_winners") or 0)
    in_miner = int(train_summary.get("winner_in_mined_negative_set") or 0)
    strongest = int(train_summary.get("winner_selected_as_strongest_miner") or 0)
    missed = int(train_summary.get("missed_training_signal") or 0)
    if n_winner <= 0:
        next_step = "diagnose_remaining_association_after_hard_negative_audit"
        designed = None
    elif strongest < n_winner or missed > 0:
        next_step = "design_solver_in_the_loop_hard_negative_control"
        designed = "solver_in_the_loop_hard_negative"
    else:
        next_step = "diagnose_loss_weighting_per_event_reduction_and_route_length_bias"
        designed = None
    return {
        "continue_to_15d_relative_wls": False,
        "open_15d_wls_authorized_by_this_audit": False,
        "new_checkpoint_authorized": False,
        "train_production_fragment_winners": n_winner,
        "train_winner_in_miner": in_miner,
        "train_winner_selected_as_strongest": strongest,
        "train_missed_training_signal": missed,
        "production_hard_winners_not_selected_as_strongest": bool(n_winner > 0 and strongest < n_winner),
        "production_hard_winners_fully_covered_as_strongest": bool(n_winner > 0 and strongest == n_winner and missed == 0),
        "next_step": next_step,
        "next_objective_if_a_later_control_is_opened": designed,
        "pre_register_solver_in_the_loop_control": designed == "solver_in_the_loop_hard_negative",
        "do_not_increase_dustbin_weight": True,
        "do_not_retune_operating_point": True,
        "architecture_change_allowed": False,
    }


def audit_workbook59_miner_does_not_unroll_solver() -> dict[str, object]:
    source = inspect.getsource(packing_route_competition_loss)
    return {
        "function": "training.gauge_consistent_route.packing_route_competition_loss",
        "calls_production_solver": "adjacent_route_assignment" in source or "_select_disjoint_routes" in source,
        "calls_route_hypotheses": "_route_hypotheses" in source,
        "uses_max_of_enumerated_rivals": "torch.stack(competitor_utilities).max()" in source,
        "default_include_dustbin": "include_dustbin: bool = False" in source,
        "note": (
            "Workbook-59 keeps the local max-rival miner and does not unroll "
            "the unit-capacity packer.  Production winners must be taken from "
            "_route_hypotheses, not from this miner."
        ),
    }
