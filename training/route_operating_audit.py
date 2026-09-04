"""Truth-only operating-layer audit for frozen adjacent four-station packing.

The helpers never feed labels into the solver.  They reconstruct the same
contiguous-route utilities the unit-capacity packer uses — clipped log-odds of
calibrated adjacent probabilities plus ``n_stations * unmatched_penalty`` —
and then classify why a complete truth chain was or was not selected.

Dustbin utility is identically zero: the empty assignment is the reference,
and a hypothesis is admitted only when its utility is strictly positive.
"""

from __future__ import annotations

from collections import Counter
from typing import Mapping, Sequence

import numpy as np

from baselines.field_chi2_matching import FieldCandidate
from baselines.global_assignment import ScoreMatrix
from baselines.route_assignment import (
    Route,
    RouteAssignmentConfig,
    RouteAssignmentResult,
    _matrix_edge_lookup,
    _route_hypotheses,
    adjacent_route_assignment,
    adjacent_station_pairs,
)
from datasets.root_loader import EventTracklets
from evaluation.pairwise_metrics import probability_to_logit
from evaluation.route_metrics import _complete_truth_chains, _unique_truth_by_station


STATION_PATH = (0, 1, 2, 3)
ADJACENT_PAIRS = adjacent_station_pairs(STATION_PATH)
DUSTBIN_UTILITY = 0.0
SOLVER_PROBABILITY_FLOOR = 1.0e-6
HISTORICAL_UNMATCHED_PENALTY = -1.0
LOSS_STAGES = (
    "selected",
    "candidate_missing",
    "below_station_pair_threshold",
    "utility_nonpositive",
    "packing_competition",
)


def pair_label(pair: tuple[int, int]) -> str:
    return f"{int(pair[0])}->{int(pair[1])}"


def solver_log_odds(probability: float) -> float:
    """Match ``baselines.route_assignment._route_hypotheses`` edge log-odds."""
    value = float(probability)
    if not np.isfinite(value):
        raise ValueError("route probability must be finite")
    clipped = min(max(value, SOLVER_PROBABILITY_FLOOR), 1.0 - SOLVER_PROBABILITY_FLOOR)
    return float(np.log(clipped) - np.log1p(-clipped))


def complete_route_packing_utility(
    probabilities: Sequence[float],
    unmatched_penalty: float,
    *,
    n_stations: int = 4,
) -> float:
    """Packing utility of a contiguous route relative to the empty dustbin."""
    values = [float(item) for item in probabilities]
    if len(values) != int(n_stations) - 1:
        raise ValueError("complete-route packing utility expects one probability per adjacent hop")
    return float(sum(solver_log_odds(value) for value in values) + int(n_stations) * float(unmatched_penalty))


def origin_key(event: EventTracklets, index: int) -> tuple[int, int, int] | None:
    if event.origin_run_id is None or event.origin_event_id is None or event.origin_tracklet_id is None:
        return None
    return (
        int(event.origin_run_id[int(index)]),
        int(event.origin_event_id[int(index)]),
        int(event.origin_tracklet_id[int(index)]),
    )


def chain_origin_signature(
    event: EventTracklets, endpoints: Sequence[tuple[int, int]]
) -> tuple[tuple[int, int, int], ...] | None:
    keys = [origin_key(event, int(index)) for _, index in endpoints]
    if any(item is None for item in keys):
        return None
    return tuple(keys)  # type: ignore[return-value]


def _rank_descending(values: Sequence[float], query: float) -> int:
    """1-based rank; ties receive the average-free dense rank of strictly greater scores plus one."""
    return 1 + sum(1 for value in values if float(value) > float(query))


def _edge_index(
    candidates: Sequence[FieldCandidate],
    raw_scores: np.ndarray,
    calibrated_scores: np.ndarray,
    labels: np.ndarray,
) -> dict[tuple[int, int], dict[str, object]]:
    raw = np.asarray(raw_scores, dtype=np.float64)
    calibrated = np.asarray(calibrated_scores, dtype=np.float64)
    target = np.asarray(labels, dtype=bool)
    if raw.shape != (len(candidates),) or calibrated.shape != raw.shape or target.shape != raw.shape:
        raise ValueError("candidate scores and labels must align with the physical candidate list")
    table: dict[tuple[int, int], dict[str, object]] = {}
    for candidate, raw_value, calibrated_value, label in zip(candidates, raw, calibrated, target):
        key = (int(candidate.source_index), int(candidate.target_index))
        if key in table:
            raise ValueError("duplicate physical candidate edge")
        table[key] = {
            "source_index": int(candidate.source_index),
            "target_index": int(candidate.target_index),
            "chi2": float(candidate.chi2),
            "raw_probability": float(raw_value),
            "calibrated_probability": float(calibrated_value),
            "raw_logit": float(probability_to_logit(np.asarray([raw_value]))[0]),
            "calibrated_logit": float(solver_log_odds(calibrated_value)),
            "label": bool(label),
        }
    return table


def _endpoint_tuple(route: Route) -> tuple[tuple[int, int], ...]:
    return tuple((int(station), int(index)) for station, index in route.endpoints)


def classify_selected_route(
    event: EventTracklets,
    endpoints: Sequence[tuple[int, int]],
    unique_by_index: Mapping[int, Mapping[int, int]],
) -> dict[str, object]:
    labels = [unique_by_index.get(int(station), {}).get(int(index)) for station, index in endpoints]
    raw_labels = [int(event.truth_particle_id[int(index)]) for _, index in endpoints]
    has_fake = any(label < 0 for label in raw_labels)
    truth_consistent = (
        not has_fake and all(label is not None for label in labels) and len(set(labels)) == 1
    )
    if has_fake:
        kind = "fake_endpoint"
    elif truth_consistent:
        kind = "truth_consistent"
    else:
        kind = "mixed_or_ambiguous"
    return {
        "kind": kind,
        "truth_consistent": bool(truth_consistent),
        "has_fake_endpoint": bool(has_fake),
        "n_stations": int(len(tuple(endpoints))),
        "complete_four_station": tuple(int(station) for station, _ in endpoints) == STATION_PATH,
        "truth_id": None if not truth_consistent else int(labels[0]),
    }


def classify_truth_chain_loss(
    *,
    candidate_retained: bool,
    score_retained: bool,
    failed_pairs: Sequence[str],
    packing_utility: float | None,
    selected: bool,
) -> str:
    if selected:
        return "selected"
    if not candidate_retained:
        return "candidate_missing"
    if not score_retained:
        if not failed_pairs:
            raise ValueError("score-retention failure must name at least one station pair")
        return "below_station_pair_threshold"
    if packing_utility is None or packing_utility <= DUSTBIN_UTILITY:
        return "utility_nonpositive"
    return "packing_competition"


def _pair_edge_audit(
    pair: tuple[int, int],
    source_index: int,
    target_index: int,
    table: Mapping[tuple[int, int], Mapping[str, object]],
    threshold: float,
) -> dict[str, object]:
    source_scores_raw: list[float] = []
    source_scores_cal: list[float] = []
    pair_scores_raw: list[float] = []
    pair_scores_cal: list[float] = []
    for (src, _tgt), record in table.items():
        pair_scores_raw.append(float(record["raw_probability"]))
        pair_scores_cal.append(float(record["calibrated_probability"]))
        if int(src) == int(source_index):
            source_scores_raw.append(float(record["raw_probability"]))
            source_scores_cal.append(float(record["calibrated_probability"]))
    record = table.get((int(source_index), int(target_index)))
    if record is None:
        return {
            "pair": pair_label(pair),
            "in_candidate_graph": False,
            "above_threshold": False,
            "threshold": float(threshold),
        }
    raw_p = float(record["raw_probability"])
    cal_p = float(record["calibrated_probability"])
    return {
        "pair": pair_label(pair),
        "in_candidate_graph": True,
        "above_threshold": bool(cal_p >= float(threshold)),
        "threshold": float(threshold),
        "chi2": float(record["chi2"]),
        "raw_probability": raw_p,
        "calibrated_probability": cal_p,
        "raw_logit": float(record["raw_logit"]),
        "calibrated_logit": float(record["calibrated_logit"]),
        "truth_rank_among_source_raw": _rank_descending(source_scores_raw, raw_p),
        "truth_rank_among_source_calibrated": _rank_descending(source_scores_cal, cal_p),
        "truth_rank_among_pair_raw": _rank_descending(pair_scores_raw, raw_p),
        "truth_rank_among_pair_calibrated": _rank_descending(pair_scores_cal, cal_p),
        "source_candidate_count": int(len(source_scores_cal)),
        "pair_candidate_count": int(len(pair_scores_cal)),
        "platt_preserves_source_rank": _rank_descending(source_scores_raw, raw_p)
        == _rank_descending(source_scores_cal, cal_p),
    }


def _strongest_competitor(
    truth_endpoints: Sequence[tuple[int, int]],
    hypotheses: Sequence[Route],
    unique_by_index: Mapping[int, Mapping[int, int]],
    event: EventTracklets,
) -> dict[str, object] | None:
    truth_set = set(_normalize_endpoints(truth_endpoints))
    truth_ids = {int(station): int(index) for station, index in truth_set}
    best: Route | None = None
    for route in hypotheses:
        endpoints = set(_endpoint_tuple(route))
        if endpoints == truth_set:
            continue
        if not endpoints.intersection(truth_set):
            continue
        if best is None or float(route.utility) > float(best.utility):
            best = route
    if best is None:
        return None
    classification = classify_selected_route(event, _endpoint_tuple(best), unique_by_index)
    shares = [
        {"station": int(station), "index": int(index)}
        for station, index in _endpoint_tuple(best)
        if truth_ids.get(int(station)) == int(index)
    ]
    return {
        "utility": float(best.utility),
        "n_stations": int(len(best.endpoints)),
        "complete_four_station": classification["complete_four_station"],
        "kind": classification["kind"],
        "shared_endpoints": shares,
        "endpoints": [
            {"station": int(station), "index": int(index)} for station, index in _endpoint_tuple(best)
        ],
    }


def _normalize_endpoints(endpoints: Sequence[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    return tuple((int(station), int(index)) for station, index in endpoints)


def _raw_complete_utility(
    pairs: Sequence[tuple[int, int]],
    by_station: Mapping[int, int],
    tables: Mapping[tuple[int, int], Mapping[tuple[int, int], Mapping[str, object]]],
    unmatched_penalty: float,
) -> float | None:
    probabilities: list[float] = []
    for pair in pairs:
        record = tables[pair].get((int(by_station[pair[0]]), int(by_station[pair[1]])))
        if record is None:
            return None
        probabilities.append(float(record["raw_probability"]))
    return complete_route_packing_utility(probabilities, unmatched_penalty)


def audit_event_truth_chains(
    event: EventTracklets,
    station_matrices: Mapping[tuple[int, int], tuple[ScoreMatrix, Sequence[FieldCandidate]]],
    pair_tables: Mapping[tuple[int, int], Mapping[tuple[int, int], Mapping[str, object]]],
    config: RouteAssignmentConfig,
    result: RouteAssignmentResult,
    *,
    complete_route_scores: Mapping[tuple[int, ...], float] | None = None,
) -> list[dict[str, object]]:
    """Classify every complete truth chain in one physical synthetic event."""
    if event.truth_particle_id is None:
        return []
    pairs = adjacent_station_pairs(tuple(int(station) for station in config.station_path))
    unique_by_truth, unique_by_index = _unique_truth_by_station(event, config.station_path)
    complete = _complete_truth_chains(unique_by_truth, config.station_path)
    edge_lookups = {
        pair: _matrix_edge_lookup(
            station_matrices[pair][0],
            station_matrices[pair][1],
            float(config.score_threshold_by_pair[pair]),
        )
        for pair in pairs
    }
    hypotheses = _route_hypotheses(
        event,
        tuple(int(station) for station in config.station_path),
        edge_lookups,
        float(config.unmatched_penalty),
        int(config.maximum_hypotheses),
        complete_route_scores,
        config.complete_route_score_threshold,
        config.complete_route_score_composition,
        config.complete_route_context_weight,
    )
    selected = {_endpoint_tuple(route) for route in result.routes}
    selected_utility = {_endpoint_tuple(route): float(route.utility) for route in result.routes}
    rows: list[dict[str, object]] = []
    for truth_id, endpoints in complete.items():
        by_station = {int(station): int(index) for station, index in endpoints}
        pair_rows = []
        failed_pairs: list[str] = []
        calibrated_probabilities: list[float] = []
        candidate_retained = True
        score_retained = True
        for pair in pairs:
            row = _pair_edge_audit(
                pair,
                by_station[pair[0]],
                by_station[pair[1]],
                pair_tables[pair],
                float(config.score_threshold_by_pair[pair]),
            )
            pair_rows.append(row)
            if not row["in_candidate_graph"]:
                candidate_retained = False
                score_retained = False
                failed_pairs.append(pair_label(pair))
                continue
            calibrated_probabilities.append(float(row["calibrated_probability"]))
            if not row["above_threshold"]:
                score_retained = False
                failed_pairs.append(pair_label(pair))
        packing_utility = (
            complete_route_packing_utility(calibrated_probabilities, float(config.unmatched_penalty))
            if candidate_retained
            else None
        )
        raw_utility = _raw_complete_utility(
            pairs, by_station, pair_tables, float(config.unmatched_penalty)
        )
        endpoint_tuple = _normalize_endpoints(endpoints)
        query_score = None
        if complete_route_scores is not None:
            query_key = tuple(int(index) for _, index in endpoint_tuple)
            if query_key not in complete_route_scores:
                raise RuntimeError(
                    "complete physical truth route has no route-query score; "
                    "the V4 score map and adjacent candidate graph disagree"
                )
            query_score = float(complete_route_scores[query_key])
            if candidate_retained:
                packing_utility = solver_log_odds(query_score) + 4.0 * float(config.unmatched_penalty)
        is_selected = endpoint_tuple in selected
        stage = classify_truth_chain_loss(
            candidate_retained=candidate_retained,
            score_retained=score_retained,
            failed_pairs=failed_pairs,
            packing_utility=packing_utility if score_retained else None,
            selected=is_selected,
        )
        competitor = _strongest_competitor(endpoint_tuple, hypotheses, unique_by_index, event)
        selected_blockers = [
            {
                "utility": selected_utility[block],
                "n_stations": len(block),
                "endpoints": [
                    {"station": int(station), "index": int(index)} for station, index in block
                ],
                **classify_selected_route(event, block, unique_by_index),
            }
            for block in selected
            if block != endpoint_tuple and set(block).intersection(endpoint_tuple)
        ]
        ranking_stable = all(
            row.get("platt_preserves_source_rank", True) for row in pair_rows if row.get("in_candidate_graph")
        )
        source_rank_one = all(
            int(row.get("truth_rank_among_source_calibrated", 0)) == 1
            for row in pair_rows
            if row.get("in_candidate_graph")
        )
        rows.append(
            {
                "run_id": int(event.run_id),
                "event_id": int(event.event_id),
                "truth_id": int(truth_id),
                "origin_signature": chain_origin_signature(event, endpoint_tuple),
                "endpoints": [
                    {"station": int(station), "index": int(index)} for station, index in endpoint_tuple
                ],
                "candidate_retained": bool(candidate_retained),
                "score_retained": bool(score_retained),
                "selected": bool(is_selected),
                "loss_stage": stage,
                "failed_pairs": failed_pairs,
                "edges": pair_rows,
                "complete_truth_route_utility": packing_utility,
                "complete_truth_route_utility_raw_probabilities": raw_utility,
                "dustbin_utility": DUSTBIN_UTILITY,
                "utility_minus_dustbin": None if packing_utility is None else float(packing_utility - DUSTBIN_UTILITY),
                "strongest_conflicting_hypothesis": competitor,
                "selected_endpoint_blockers": selected_blockers,
                "complete_route_query_score": None if query_score is None else float(query_score),
                "within_pair_ranking_stable_under_platt": bool(ranking_stable),
                "truth_edges_all_source_rank_one": bool(source_rank_one),
                "raw_beats_calibrated_competitor": _raw_ranking_survives_calibration(
                    packing_utility, raw_utility, competitor, pair_tables, by_station, pairs, config
                ),
            }
        )
    return rows


def _raw_ranking_survives_calibration(
    packing_utility: float | None,
    raw_utility: float | None,
    competitor: Mapping[str, object] | None,
    pair_tables: Mapping[tuple[int, int], Mapping[tuple[int, int], Mapping[str, object]]],
    by_station: Mapping[int, int],
    pairs: Sequence[tuple[int, int]],
    config: RouteAssignmentConfig,
) -> bool | None:
    """True when the calibrated competitor also beats truth on raw probabilities.

    Used to separate a scale/operating-point failure from a raw-ranking failure.
    Only defined for packing-competition losses of complete four-station competitors.
    """
    if packing_utility is None or competitor is None:
        return None
    if not bool(competitor.get("complete_four_station")):
        return None
    if raw_utility is None:
        return None
    competitor_endpoints = {
        int(item["station"]): int(item["index"]) for item in competitor["endpoints"]
    }
    probabilities: list[float] = []
    for pair in pairs:
        record = pair_tables[pair].get((competitor_endpoints[pair[0]], competitor_endpoints[pair[1]]))
        if record is None:
            return None
        probabilities.append(float(record["raw_probability"]))
    competitor_raw = complete_route_packing_utility(probabilities, float(config.unmatched_penalty))
    calibrated_competitor_wins = float(competitor["utility"]) > float(packing_utility)
    raw_competitor_wins = float(competitor_raw) > float(raw_utility)
    if not calibrated_competitor_wins:
        return None
    return bool(raw_competitor_wins)


def audit_selected_routes(
    event: EventTracklets,
    result: RouteAssignmentResult,
    pair_tables: Mapping[tuple[int, int], Mapping[tuple[int, int], Mapping[str, object]]],
    config: RouteAssignmentConfig,
) -> list[dict[str, object]]:
    if event.truth_particle_id is None:
        return []
    _, unique_by_index = _unique_truth_by_station(event, config.station_path)
    pairs = adjacent_station_pairs(tuple(int(station) for station in config.station_path))
    rows: list[dict[str, object]] = []
    for route in result.routes:
        endpoints = _endpoint_tuple(route)
        classification = classify_selected_route(event, endpoints, unique_by_index)
        probabilities: list[float] = []
        raw_probabilities: list[float] = []
        complete_edges = True
        for left, right in zip(endpoints, endpoints[1:]):
            pair = (int(left[0]), int(right[0]))
            if pair not in pairs:
                complete_edges = False
                break
            record = pair_tables[pair].get((int(left[1]), int(right[1])))
            if record is None:
                complete_edges = False
                break
            probabilities.append(float(record["calibrated_probability"]))
            raw_probabilities.append(float(record["raw_probability"]))
        historical_utility = None
        if complete_edges:
            historical_utility = complete_route_packing_utility(
                probabilities,
                HISTORICAL_UNMATCHED_PENALTY,
                n_stations=len(endpoints),
            )
        rows.append(
            {
                **classification,
                "utility": float(route.utility),
                "historical_penalty_counterfactual_utility": historical_utility,
                "historical_penalty_would_be_nonpositive": (
                    None if historical_utility is None else bool(historical_utility <= DUSTBIN_UTILITY)
                ),
                "origin_signature": chain_origin_signature(event, endpoints),
            }
        )
    return rows


def summarize_truth_rows(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    stages = Counter(str(row["loss_stage"]) for row in rows)
    failed_pairs = Counter(
        pair for row in rows if row["loss_stage"] == "below_station_pair_threshold" for pair in row["failed_pairs"]
    )
    competition_kinds = Counter(
        str((row.get("strongest_conflicting_hypothesis") or {}).get("kind") or "none")
        for row in rows
        if row["loss_stage"] == "packing_competition"
    )
    rank_one = sum(1 for row in rows if row.get("truth_edges_all_source_rank_one"))
    ranking_stable = sum(1 for row in rows if row.get("within_pair_ranking_stable_under_platt"))
    raw_beats = [
        row.get("raw_beats_calibrated_competitor")
        for row in rows
        if row["loss_stage"] == "packing_competition"
        and row.get("raw_beats_calibrated_competitor") is not None
    ]
    return {
        "complete_truth_chains": int(len(rows)),
        "loss_stage_counts": {stage: int(stages.get(stage, 0)) for stage in LOSS_STAGES},
        "below_threshold_failed_pair_counts": dict(failed_pairs),
        "packing_competition_competitor_kinds": dict(competition_kinds),
        "truth_edges_all_source_rank_one": int(rank_one),
        "within_pair_ranking_stable_under_platt": int(ranking_stable),
        "packing_competition_with_defined_raw_comparison": int(len(raw_beats)),
        "packing_competition_raw_also_beats_truth": int(sum(1 for value in raw_beats if value)),
        "packing_competition_raw_does_not_beat_truth": int(sum(1 for value in raw_beats if not value)),
        "score_retained": int(sum(1 for row in rows if row["score_retained"])),
        "selected": int(sum(1 for row in rows if row["selected"])),
    }


def summarize_selected_rows(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    kinds = Counter(str(row["kind"]) for row in rows)
    lengths = Counter(int(row["n_stations"]) for row in rows)
    fake_or_mixed = [row for row in rows if not row["truth_consistent"]]
    historical_killed = [
        row
        for row in fake_or_mixed
        if row.get("historical_penalty_would_be_nonpositive") is True
    ]
    complete = [row for row in rows if row["complete_four_station"]]
    complete_fake = [row for row in complete if not row["truth_consistent"]]
    return {
        "selected_routes": int(len(rows)),
        "by_kind": dict(kinds),
        "by_n_stations": {str(key): int(value) for key, value in sorted(lengths.items())},
        "track_fake_rate": None if not rows else float(len(fake_or_mixed) / len(rows)),
        "complete_track_purity": (
            None if not complete else float((len(complete) - len(complete_fake)) / len(complete))
        ),
        "fake_or_mixed_selected": int(len(fake_or_mixed)),
        "fake_or_mixed_historical_penalty_would_drop": int(len(historical_killed)),
        "fake_or_mixed_partial_routes": int(sum(1 for row in fake_or_mixed if not row["complete_four_station"])),
        "diagnostic_only_historical_penalty_is_not_a_new_operating_point": True,
    }


def pair_tables_from_sets(
    candidate_sets: Sequence[object],
    raw_scores: Sequence[np.ndarray],
    calibrated_scores: Sequence[np.ndarray],
) -> dict[tuple[str, str, int, int], dict[tuple[int, int], dict[tuple[int, int], dict[str, object]]]]:
    """Group adjacent candidate tables by physical event key."""
    grouped: dict[tuple[str, str, int, int], dict[tuple[int, int], dict[tuple[int, int], dict[str, object]]]] = {}
    for candidate_set, raw, calibrated in zip(candidate_sets, raw_scores, calibrated_scores):
        pair = tuple(int(value) for value in candidate_set.station_pair)
        if pair not in ADJACENT_PAIRS:
            continue
        event = candidate_set.event
        key = (
            str(candidate_set.sample.source_id),
            str(candidate_set.sample.payload_id),
            int(event.run_id),
            int(event.event_id),
        )
        payload = grouped.setdefault(key, {})
        if pair in payload:
            raise ValueError("duplicate adjacent pair table for one event")
        payload[pair] = _edge_index(candidate_set.candidates, raw, calibrated, candidate_set.labels)
    return grouped


def assign_and_audit_payload(
    candidate_sets: Sequence[object],
    raw_scores: Sequence[np.ndarray],
    calibrated_scores: Sequence[np.ndarray],
    config: RouteAssignmentConfig,
    *,
    complete_route_scores_by_event: Mapping[tuple[str, str, int, int], Mapping[tuple[int, ...], float]] | None = None,
    calibration_bins: int = 15,
) -> dict[str, object]:
    """Run the frozen solver once and attach the operating-layer audit."""
    from training.route_assignment import assign_adjacent_route_sets, prepare_route_assignment_context

    context = prepare_route_assignment_context(
        candidate_sets, calibrated_scores, calibration_bins, config.station_path
    )
    assigned = assign_adjacent_route_sets(
        candidate_sets,
        calibrated_scores,
        config,
        calibration_bins,
        context=context,
        complete_route_scores_by_event=complete_route_scores_by_event,
    )
    tables = pair_tables_from_sets(candidate_sets, raw_scores, calibrated_scores)
    truth_rows: list[dict[str, object]] = []
    selected_rows: list[dict[str, object]] = []
    for item in assigned:
        pair_tables = tables[item.key]
        query_map = None if complete_route_scores_by_event is None else complete_route_scores_by_event.get(item.key)
        event_truth = audit_event_truth_chains(
            item.event,
            next(group.station_matrices for group in context.groups if group.key == item.key),
            pair_tables,
            config,
            item.result,
            complete_route_scores=query_map,
        )
        for row in event_truth:
            row["sample_id"] = item.key[0]
            row["payload_id"] = item.key[1]
            row["run_id"] = item.key[2]
            row["event_id"] = item.key[3]
        truth_rows.extend(event_truth)
        event_selected = audit_selected_routes(item.event, item.result, pair_tables, config)
        for row in event_selected:
            row["sample_id"] = item.key[0]
            row["payload_id"] = item.key[1]
            row["run_id"] = item.key[2]
            row["event_id"] = item.key[3]
        selected_rows.extend(event_selected)
    return {
        "truth_chains": truth_rows,
        "selected_routes": selected_rows,
        "truth_summary": summarize_truth_rows(truth_rows),
        "selected_summary": summarize_selected_rows(selected_rows),
        "assigned_events": int(len(assigned)),
    }


def align_gauge_twins(
    chart_rows: Sequence[Mapping[str, object]],
    twin_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Join complete truth chains that share overlay origin and (run, event)."""

    def _key(row: Mapping[str, object]) -> tuple[object, ...]:
        origin = row.get("origin_signature")
        if origin is None:
            raise ValueError("gauge-twin alignment requires origin signatures")
        return (int(row["run_id"]), int(row["event_id"]), tuple(tuple(item) for item in origin))

    chart = {_key(row): row for row in chart_rows}
    twin = {_key(row): row for row in twin_rows}
    if len(chart) != len(chart_rows) or len(twin) != len(twin_rows):
        raise ValueError("duplicate origin signature inside a gauge twin payload")
    if set(chart) != set(twin):
        missing_in_twin = sorted(set(chart) - set(twin))
        missing_in_chart = sorted(set(twin) - set(chart))
        raise ValueError(
            "gauge twins are not event-aligned: "
            f"missing_in_twin={len(missing_in_twin)}, missing_in_chart={len(missing_in_chart)}"
        )
    aligned: list[dict[str, object]] = []
    for key in sorted(chart):
        left = chart[key]
        right = twin[key]
        aligned.append(
            {
                "run_id": int(key[0]),
                "event_id": int(key[1]),
                "origin_signature": left["origin_signature"],
                "chart_loss_stage": left["loss_stage"],
                "twin_loss_stage": right["loss_stage"],
                "chart_selected": bool(left["selected"]),
                "twin_selected": bool(right["selected"]),
                "stage_changed": left["loss_stage"] != right["loss_stage"],
                "chart_failed_pairs": list(left["failed_pairs"]),
                "twin_failed_pairs": list(right["failed_pairs"]),
                "chart_utility": left["complete_truth_route_utility"],
                "twin_utility": right["complete_truth_route_utility"],
                "chart_edges": left["edges"],
                "twin_edges": right["edges"],
                "chart_competitor": left["strongest_conflicting_hypothesis"],
                "twin_competitor": right["strongest_conflicting_hypothesis"],
                "chart_source_rank_one": bool(left["truth_edges_all_source_rank_one"]),
                "twin_source_rank_one": bool(right["truth_edges_all_source_rank_one"]),
            }
        )
    return aligned


def summarize_twin_alignment(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    changed = [row for row in rows if row["stage_changed"]]
    chart_only = [row for row in changed if row["chart_selected"] and not row["twin_selected"]]
    twin_only = [row for row in changed if row["twin_selected"] and not row["chart_selected"]]
    both_lost = Counter(
        (str(row["chart_loss_stage"]), str(row["twin_loss_stage"]))
        for row in rows
        if not row["chart_selected"] and not row["twin_selected"]
    )
    chart_win_twin_loss = Counter(str(row["twin_loss_stage"]) for row in chart_only)
    return {
        "aligned_complete_truth_chains": int(len(rows)),
        "stage_changed": int(len(changed)),
        "chart_selected_twin_lost": int(len(chart_only)),
        "twin_selected_chart_lost": int(len(twin_only)),
        "both_selected": int(sum(1 for row in rows if row["chart_selected"] and row["twin_selected"])),
        "chart_selected_twin_lost_by_twin_stage": dict(chart_win_twin_loss),
        "both_lost_stage_pairs": {f"{left}->{right}": int(count) for (left, right), count in both_lost.items()},
    }


def json_ready(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value
