"""Frozen production-utility identifiability for four-station packing.

The helpers do not train, do not retune the operating point, and do not change
the unit-capacity solver.  They reconstruct every threshold-feasible contiguous
route the production enumerator can see — including hypotheses with
``utility <= 0`` that the solver refuses to admit — and compare that set with
the training packing-competition negative miner.

Dustbin utility is identically zero.  A production hypothesis is admitted only
when its packing utility is strictly positive.
"""

from __future__ import annotations

from collections import Counter
from itertools import product
from typing import Any, Mapping, Sequence

from baselines.route_assignment import ScoredMatch
from datasets.root_loader import EventTracklets
from training.route_operating_audit import (
    ADJACENT_PAIRS,
    DUSTBIN_UTILITY,
    STATION_PATH,
    classify_selected_route,
    complete_route_packing_utility,
    solver_log_odds,
)
from evaluation.route_metrics import _unique_truth_by_station


TRAINING_ENUMERATED_TOPOLOGIES = frozenset(
    {
        "two_station_01",
        "two_station_12",
        "two_station_23",
        "three_station_prefix",
        "three_station_suffix",
        "four_station_mixed",
        "four_station_other_truth",
    }
)
DECISION_CLASSES = (
    "selected",
    "below_threshold_or_missing_candidate",
    "truth_beats_wrong_routes_but_loses_to_dustbin",
    "truth_loses_to_fragment",
    "margin_satisfied_solver_does_not_select",
)


def classify_route_topology(endpoints: Sequence[tuple[int, int]]) -> str:
    stations = tuple(int(station) for station, _ in endpoints)
    if stations == (0, 1):
        return "two_station_01"
    if stations == (1, 2):
        return "two_station_12"
    if stations == (2, 3):
        return "two_station_23"
    if stations == (0, 1, 2):
        return "three_station_prefix"
    if stations == (1, 2, 3):
        return "three_station_suffix"
    if stations == STATION_PATH:
        return "four_station"
    return "other"


def classify_competitor(
    endpoints: Sequence[tuple[int, int]],
    event: EventTracklets,
    unique_by_index: Mapping[int, Mapping[int, int]],
    *,
    truth_endpoints: Sequence[tuple[int, int]] | None = None,
) -> dict[str, object]:
    topology = classify_route_topology(endpoints)
    selected = classify_selected_route(event, endpoints, unique_by_index)
    if topology == "four_station":
        topology = "four_station_other_truth" if selected["truth_consistent"] else "four_station_mixed"
    truth_set = {(int(station), int(index)) for station, index in (truth_endpoints or ())}
    competitor_set = {(int(station), int(index)) for station, index in endpoints}
    return {
        **selected,
        "topology": topology,
        "shares_endpoint": bool(truth_set and competitor_set.intersection(truth_set)),
        "truth_consistent_fragment": bool(
            selected["truth_consistent"] and topology.startswith("three_station")
        ),
    }


def training_negative_mining_coverage(
    topology: str,
    n_threshold_feasible_overlapping_competitors: int,
) -> dict[str, object]:
    """Whether the workbook-56 miner can see this production hard competitor.

    The miner enumerates the same contiguous 2/3/4-station routes and keeps
    endpoint-overlapping threshold-feasible rivals.  Dustbin is substituted
    only when that rival set is empty.  It never takes
    ``max(rival_utility, dustbin)`` when a negative-utility fragment exists.
    """
    n_rivals = int(n_threshold_feasible_overlapping_competitors)
    return {
        "topology": topology,
        "topology_in_training_enumerator": topology in TRAINING_ENUMERATED_TOPOLOGIES,
        "dustbin_in_training_negative_set": n_rivals == 0,
        "dustbin_winner_missing_from_training_negatives": n_rivals > 0,
        "training_uses_max_of_feasible_rivals_not_max_with_dustbin": True,
    }


def classify_utility_decision(
    *,
    selected: bool,
    score_retained: bool,
    candidate_retained: bool,
    u_truth: float | None,
    u_best_competitor: float | None,
    u_dustbin: float = DUSTBIN_UTILITY,
) -> str:
    if selected:
        return "selected"
    if not candidate_retained or not score_retained or u_truth is None:
        return "below_threshold_or_missing_candidate"
    rival = float(u_dustbin) if u_best_competitor is None else max(float(u_best_competitor), float(u_dustbin))
    if float(u_truth) > rival:
        return "margin_satisfied_solver_does_not_select"
    if u_best_competitor is not None and float(u_best_competitor) > float(u_dustbin) and float(u_best_competitor) >= float(u_truth):
        return "truth_loses_to_fragment"
    return "truth_beats_wrong_routes_but_loses_to_dustbin"


def enumerate_threshold_feasible_routes(
    event: EventTracklets,
    edge_lookups: Mapping[tuple[int, int], Mapping[tuple[int, int], ScoredMatch]],
    unmatched_penalty: float,
    station_path: Sequence[int] = STATION_PATH,
) -> list[dict[str, object]]:
    """All contiguous routes whose edges pass the pair thresholds, including U<=0."""
    path = tuple(int(station) for station in station_path)
    indices_by_station = {
        station: tuple(int(index) for index in event.indices_for_station(station).tolist())
        for station in path
    }
    _, unique_by_index = _unique_truth_by_station(event, path)
    routes: list[dict[str, object]] = []
    for start in range(len(path) - 1):
        for stop in range(start + 2, len(path) + 1):
            route_stations = path[start:stop]
            if any(not indices_by_station[station] for station in route_stations):
                continue
            for route_indices in product(*(indices_by_station[station] for station in route_stations)):
                probabilities: list[float] = []
                endpoints: list[tuple[int, int]] = []
                complete = True
                for source_station, target_station, source_index, target_index in zip(
                    route_stations,
                    route_stations[1:],
                    route_indices,
                    route_indices[1:],
                ):
                    match = edge_lookups[(int(source_station), int(target_station))].get(
                        (int(source_index), int(target_index))
                    )
                    if match is None:
                        complete = False
                        break
                    probabilities.append(float(match.score))
                    endpoints.append((int(source_station), int(source_index)))
                if not complete:
                    continue
                endpoints.append((int(route_stations[-1]), int(route_indices[-1])))
                utility = complete_route_packing_utility(
                    probabilities, unmatched_penalty, n_stations=len(route_stations)
                )
                classification = classify_competitor(endpoints, event, unique_by_index)
                routes.append(
                    {
                        **classification,
                        "endpoints": [
                            {"station": int(station), "index": int(index)} for station, index in endpoints
                        ],
                        "utility": float(utility),
                        "n_stations": int(len(route_stations)),
                        "production_admitted": bool(utility > DUSTBIN_UTILITY),
                        "edge_logits": [solver_log_odds(value) for value in probabilities],
                    }
                )
    return routes


def attach_truth_route_identifiability(
    row: Mapping[str, Any],
    feasible_routes: Sequence[Mapping[str, Any]],
    unmatched_penalty: float,
) -> dict[str, object]:
    """Add U_truth / U_best_competitor / U_dustbin / ΔU and mining coverage."""
    truth_endpoints = tuple(
        (int(item["station"]), int(item["index"])) for item in row["endpoints"]
    )
    u_truth = None if row.get("complete_truth_route_utility") is None else float(row["complete_truth_route_utility"])
    overlapping: list[Mapping[str, Any]] = []
    for route in feasible_routes:
        endpoints = tuple((int(item["station"]), int(item["index"])) for item in route["endpoints"])
        if endpoints == truth_endpoints:
            continue
        if not set(endpoints).intersection(truth_endpoints):
            continue
        overlapping.append(route)
    best = None
    if overlapping:
        best = max(overlapping, key=lambda item: float(item["utility"]))
    u_best = None if best is None else float(best["utility"])
    u_dustbin = float(DUSTBIN_UTILITY)
    delta = None if u_truth is None else float(u_truth - max(u_best if u_best is not None else u_dustbin, u_dustbin))
    decision = classify_utility_decision(
        selected=bool(row.get("selected")),
        score_retained=bool(row.get("score_retained")),
        candidate_retained=bool(row.get("candidate_retained")),
        u_truth=u_truth,
        u_best_competitor=u_best,
        u_dustbin=u_dustbin,
    )
    best_topology = "none" if best is None else str(best["topology"])
    mining = training_negative_mining_coverage(best_topology, len(overlapping))
    production_winner = "selected_truth" if row.get("selected") else (
        "dustbin" if not any(bool(item["production_admitted"]) for item in overlapping) else "fragment"
    )
    family_counts = Counter(_competitor_family(item) for item in overlapping)
    return {
        **dict(row),
        "u_truth": u_truth,
        "u_best_competitor": u_best,
        "u_dustbin": u_dustbin,
        "delta_u": delta,
        "u_truth_minus_best_competitor": None if u_truth is None or u_best is None else float(u_truth - u_best),
        "truth_outranks_overlapping_competitors": (
            None if u_truth is None else bool(u_best is None or float(u_truth) > float(u_best))
        ),
        "unmatched_penalty": float(unmatched_penalty),
        "n_threshold_feasible_overlapping_competitors": int(len(overlapping)),
        "n_production_admitted_overlapping_competitors": int(
            sum(1 for item in overlapping if item["production_admitted"])
        ),
        "overlapping_competitor_families": {name: int(count) for name, count in sorted(family_counts.items())},
        "best_competitor_topology": best_topology,
        "best_competitor_family": "none" if best is None else _competitor_family(best),
        "best_competitor_truth_consistent_fragment": (
            None if best is None else bool(best.get("truth_consistent_fragment"))
        ),
        "best_competitor_n_stations": None if best is None else int(best["n_stations"]),
        "best_competitor_production_admitted": None if best is None else bool(best["production_admitted"]),
        "decision_class": decision,
        "production_winner": production_winner,
        "training_negative_mining": mining,
        "dustbin_is_production_winner": production_winner == "dustbin",
        "dustbin_winner_missed_by_training_miner": bool(
            production_winner == "dustbin" and mining["dustbin_winner_missing_from_training_negatives"]
        ),
    }


def summarize_identifiability(rows: Sequence[Mapping[str, Any]]) -> dict[str, object]:
    decisions = Counter(str(row["decision_class"]) for row in rows)
    winners = Counter(str(row["production_winner"]) for row in rows)
    topologies = Counter(
        str(row["best_competitor_topology"])
        for row in rows
        if row["decision_class"] in {"truth_loses_to_fragment", "truth_beats_wrong_routes_but_loses_to_dustbin"}
    )
    utilities = [float(row["u_truth"]) for row in rows if row.get("u_truth") is not None]
    deltas = [float(row["delta_u"]) for row in rows if row.get("delta_u") is not None]
    missed_dustbin = sum(1 for row in rows if row.get("dustbin_winner_missed_by_training_miner"))
    families = Counter(str(row.get("best_competitor_family") or "none") for row in rows)
    outranks = [row for row in rows if row.get("truth_outranks_overlapping_competitors") is True]
    ranking_but_dustbin = [
        row
        for row in rows
        if row.get("decision_class") == "truth_beats_wrong_routes_but_loses_to_dustbin"
        and row.get("truth_outranks_overlapping_competitors") is True
    ]
    ranking_loses_fragment_but_dustbin = [
        row
        for row in rows
        if row.get("decision_class") == "truth_beats_wrong_routes_but_loses_to_dustbin"
        and row.get("truth_outranks_overlapping_competitors") is False
    ]
    vs_best = [
        float(row["u_truth_minus_best_competitor"])
        for row in rows
        if row.get("u_truth_minus_best_competitor") is not None
    ]
    return {
        "complete_truth_chains": int(len(rows)),
        "decision_class_counts": {name: int(decisions.get(name, 0)) for name in DECISION_CLASSES},
        "production_winner_counts": dict(winners),
        "hard_competitor_topologies": dict(topologies),
        "best_competitor_families": dict(families),
        "selected": int(sum(1 for row in rows if row.get("selected"))),
        "score_retained": int(sum(1 for row in rows if row.get("score_retained"))),
        "u_truth_median": None if not utilities else float(_median(utilities)),
        "u_truth_fraction_nonpositive": None if not utilities else float(sum(1 for value in utilities if value <= 0.0) / len(utilities)),
        "delta_u_median": None if not deltas else float(_median(deltas)),
        "delta_u_fraction_nonpositive": None if not deltas else float(sum(1 for value in deltas if value <= 0.0) / len(deltas)),
        "u_truth_minus_best_competitor_median": None if not vs_best else float(_median(vs_best)),
        "truth_outranks_overlapping_competitors": int(len(outranks)),
        "ranking_correct_but_dustbin_wins": int(len(ranking_but_dustbin)),
        "loses_to_nonadmitted_fragment_and_dustbin_wins": int(len(ranking_loses_fragment_but_dustbin)),
        "dustbin_winner_missed_by_training_miner": int(missed_dustbin),
    }


def compare_aligned_truth_routes(
    candidate_rows: Sequence[Mapping[str, Any]],
    control_rows: Sequence[Mapping[str, Any]],
) -> dict[str, object]:
    """Compare gauge-consistent vs workbook-54 utilities on the same origin route."""

    def _key(row: Mapping[str, Any]) -> tuple[object, ...]:
        origin = row.get("origin_signature")
        if origin is None:
            raise ValueError("model comparison requires origin signatures")
        return (
            str(row["payload_id"]),
            int(row["run_id"]),
            int(row["event_id"]),
            tuple(tuple(item) for item in origin),
        )

    candidate = {_key(row): row for row in candidate_rows}
    control = {_key(row): row for row in control_rows}
    if set(candidate) != set(control):
        raise ValueError(
            "candidate and control truth routes are not origin-aligned: "
            f"only_candidate={len(set(candidate) - set(control))}, "
            f"only_control={len(set(control) - set(candidate))}"
        )
    logit_abs: list[float] = []
    utility_delta: list[float] = []
    margin_delta: list[float] = []
    decision_changed = 0
    consistency_improved_margin_not = 0
    for key in candidate:
        left = candidate[key]
        right = control[key]
        left_logits = _truth_edge_logits(left)
        right_logits = _truth_edge_logits(right)
        if left_logits is not None and right_logits is not None:
            logit_abs.extend(abs(a - b) for a, b in zip(left_logits, right_logits))
        if left.get("u_truth") is not None and right.get("u_truth") is not None:
            utility_delta.append(float(left["u_truth"]) - float(right["u_truth"]))
        if left.get("delta_u") is not None and right.get("delta_u") is not None:
            margin_delta.append(float(left["delta_u"]) - float(right["delta_u"]))
        if left["decision_class"] != right["decision_class"]:
            decision_changed += 1
        if (
            left.get("delta_u") is not None
            and right.get("delta_u") is not None
            and float(left["delta_u"]) <= float(right["delta_u"])
            and left["decision_class"] != "selected"
            and right["decision_class"] == "selected"
        ):
            consistency_improved_margin_not += 1
    return {
        "aligned_truth_routes": int(len(candidate)),
        "decision_class_changed": int(decision_changed),
        "candidate_selected": int(sum(1 for row in candidate.values() if row.get("selected"))),
        "control_selected": int(sum(1 for row in control.values() if row.get("selected"))),
        "abs_edge_logit_shift_median": None if not logit_abs else float(_median(logit_abs)),
        "u_truth_candidate_minus_control_median": None if not utility_delta else float(_median(utility_delta)),
        "delta_u_candidate_minus_control_median": None if not margin_delta else float(_median(margin_delta)),
        "control_selected_but_candidate_margin_not_improved": int(consistency_improved_margin_not),
    }


def _competitor_family(route: Mapping[str, Any]) -> str:
    topology = str(route.get("topology") or "other")
    if bool(route.get("truth_consistent_fragment")):
        return f"truth_consistent_{topology}"
    if topology.startswith("two_station"):
        return "two_station_fragment"
    if topology == "four_station_mixed":
        return "mixed_route"
    if topology == "four_station_other_truth":
        return "endpoint_conflict_other_truth"
    if bool(route.get("shares_endpoint")) and not bool(route.get("truth_consistent")):
        return "endpoint_conflict_route"
    return topology


def _truth_edge_logits(row: Mapping[str, Any]) -> list[float] | None:
    edges = row.get("edges")
    if not isinstance(edges, list) or len(edges) != 3:
        return None
    logits: list[float] = []
    for edge in edges:
        if not isinstance(edge, Mapping) or "calibrated_logit" not in edge:
            return None
        logits.append(float(edge["calibrated_logit"]))
    return logits


def _median(values: Sequence[float]) -> float:
    ordered = sorted(float(value) for value in values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return 0.5 * (ordered[mid - 1] + ordered[mid])
