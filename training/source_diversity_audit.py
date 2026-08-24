"""Workbook-63 source-disjoint four-station training-diversity audit.

Does not train, does not retune the workbook-62 objective or operating
point, and does not treat the already-opened workbook-56 transfer set as a
final independent gate.  The question is whether the two current train
sources cover the track phase space and source characteristics that produce
the stable ``draw_01 + common`` truth-utility drop.

Workbook-56 transfer numbers may be used only as a development diagnostic.
Reserved blind sources must never be loaded.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Iterable, Mapping, Sequence

from datasets.root_loader import EventTracklets
from training.dustbin_aware_route_margin import PACKING_MARGIN
from training.route_reduction_audit import classify_margin_bin, production_margin
from training.route_utility_identifiability import _median


FAILURE_PAYLOAD = "iteration_00_draw_01_plus_common"
CHART_PAYLOAD = "iteration_00_draw_01"
STATIONS = (0, 1, 2, 3)
STATE_KEYS = ("x_mm", "y_mm", "tx", "ty")
PAIR_23 = (2, 3)
SEALED_PREFIXES = ("mc24_100116_", "mc24_100117_")
OUTLIER_SOURCES = frozenset({"mc24_100047_00150_00199"})
CURRENT_TRAIN_SOURCES = frozenset(
    {"mc24_100043_00200_00299", "mc24_100044_00300_00399"}
)
DEVELOPMENT_SOURCES = frozenset(
    {"mc24_100047_00050_00099", "mc24_100048_00050_00099"}
)
TRANSFER_DIAGNOSTIC_SOURCES = frozenset(
    {"mc24_100047_00300_00349", "mc24_100048_00300_00349"}
)
CANDIDATE_DIAGNOSTIC_SOURCES = (
    "mc24_100043_00300_00399",
    "mc24_100044_00200_00299",
    "mc24_100047_00100_00149",
    "mc24_100048_00100_00149",
)
PROPOSED_NEW_TRAIN_SOURCES = (
    "mc24_100043_00200_00299",
    "mc24_100044_00300_00399",
    "mc24_100043_00300_00399",
    "mc24_100044_00200_00299",
    "mc24_100047_00100_00149",
    "mc24_100048_00100_00149",
)
RESERVED_BLIND_SOURCES = (
    "mc24_100047_00350_00399",
    "mc24_100048_00350_00399",
)
UNUSED_RESERVE_SOURCES = (
    "mc24_100047_00800_00849",
    "mc24_100048_00800_00849",
)
WB62_CHECKPOINT_SHA256 = "a46a35bd28eb294fe307590d4f12595f6d3bfaa8dc64aea0bf7418543605e1ef"
WB62_OBJECTIVE = "dustbin_aware_plus_gauge_consistent_plus_max_reduction"

# Frozen before the audit numbers are opened.
COVERAGE_QUANTILES = (0.05, 0.95)
MIN_KINEMATIC_INSIDE = 0.90
MIN_S3_KINEMATIC_INSIDE = 0.90
MIN_OCCUPANCY_INSIDE = 0.90
MIN_CHARGE_FRACTION = 0.05
MAX_DELTA_HARD_RATE_SHIFT = 0.10
MAX_WINNER_RATE_SHIFT = 0.05
MAX_MEDIAN_DELTA_SHIFT = 0.20
MAX_MEDIAN_LOGIT23_SHIFT = 0.50

SOURCE_CHARGE = {
    "100043": "mu_minus",
    "100044": "mu_plus",
    "100047": "mu_minus",
    "100048": "mu_plus",
}


def source_dsid(source_id: str) -> str:
    parts = str(source_id).split("_")
    if len(parts) < 2:
        raise ValueError(f"unrecognized source id: {source_id}")
    return parts[1]


def charge_from_pdg(pdg: int | None) -> str | None:
    if pdg is None:
        return None
    value = int(pdg)
    if value == 13:
        return "mu_minus"
    if value == -13:
        return "mu_plus"
    return f"pdg_{value}"


def charge_from_source_id(source_id: str | None) -> str | None:
    if not source_id:
        return None
    return SOURCE_CHARGE.get(source_dsid(str(source_id)))


def is_sealed_source(source_id: str) -> bool:
    return str(source_id).startswith(SEALED_PREFIXES)


def assert_sources_allowed(source_ids: Iterable[str], *, allow_reserved_blind: bool = False) -> None:
    seen = {str(source) for source in source_ids}
    sealed = sorted(source for source in seen if is_sealed_source(source))
    if sealed:
        raise ValueError("refusing sealed sources: " + ", ".join(sealed))
    outliers = sorted(seen & OUTLIER_SOURCES)
    if outliers:
        raise ValueError("refusing historical outlier sources: " + ", ".join(outliers))
    if not allow_reserved_blind:
        reserved = sorted(seen & (set(RESERVED_BLIND_SOURCES) | set(UNUSED_RESERVE_SOURCES)))
        if reserved:
            raise ValueError("refusing reserved blind/unused-reserve sources: " + ", ".join(reserved))


def quantile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("quantile of an empty sample")
    if fraction <= 0.0:
        return float(ordered[0])
    if fraction >= 1.0:
        return float(ordered[-1])
    position = (len(ordered) - 1) * float(fraction)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def fraction_inside_quantiles(
    values: Sequence[float],
    reference: Sequence[float],
    *,
    lo: float = COVERAGE_QUANTILES[0],
    hi: float = COVERAGE_QUANTILES[1],
) -> float | None:
    if not values or not reference:
        return None
    low = quantile(reference, lo)
    high = quantile(reference, hi)
    return float(sum(1 for value in values if low <= float(value) <= high) / len(values))


def truth_edge_logit(row: Mapping[str, Any], pair: tuple[int, int]) -> float | None:
    edges = row.get("edges")
    if not isinstance(edges, list):
        stored = row.get("truth_edge_logits")
        if isinstance(stored, Mapping) and f"{pair[0]}->{pair[1]}" in stored:
            value = stored[f"{pair[0]}->{pair[1]}"]
            return None if value is None else float(value)
        return None
    for edge in edges:
        if not isinstance(edge, Mapping):
            continue
        left = edge.get("source_station")
        right = edge.get("target_station")
        pair_name = edge.get("pair")
        if left is None or right is None:
            if pair_name == f"{pair[0]}->{pair[1]}" or pair_name == pair:
                left, right = pair
            else:
                continue
        if (int(left), int(right)) != pair:
            continue
        if "calibrated_logit" not in edge:
            return None
        return float(edge["calibrated_logit"])
    if len(edges) == 3 and pair == PAIR_23 and isinstance(edges[2], Mapping):
        if "calibrated_logit" not in edges[2]:
            return None
        return float(edges[2]["calibrated_logit"])
    return None


def _edge_logit_table(row: Mapping[str, Any]) -> dict[str, float | None]:
    names = ("0->1", "1->2", "2->3")
    edges = row.get("edges")
    table: dict[str, float | None] = {name: None for name in names}
    if isinstance(edges, list) and len(edges) == 3:
        for name, edge in zip(names, edges):
            if isinstance(edge, Mapping) and "calibrated_logit" in edge:
                table[name] = float(edge["calibrated_logit"])
        return table
    stored = row.get("truth_edge_logits")
    if isinstance(stored, Mapping):
        for name in names:
            value = stored.get(name)
            table[name] = None if value is None else float(value)
    return table


def track_state_by_station(
    event: EventTracklets,
    endpoints: Sequence[Mapping[str, Any]],
) -> dict[int, dict[str, float]]:
    states: dict[int, dict[str, float]] = {}
    for item in endpoints:
        station = int(item["station"])
        index = int(item["index"])
        x_mm, y_mm, tx, ty = (float(value) for value in event.state[index])
        states[station] = {"x_mm": x_mm, "y_mm": y_mm, "tx": tx, "ty": ty}
    return states


def station_occupancy(event: EventTracklets) -> dict[str, int]:
    return {f"s{station}": int((event.station_id == station).sum()) for station in STATIONS}


def attach_route_phase_space(
    row: Mapping[str, Any],
    event: EventTracklets,
    *,
    source_by_namespaced_run: Mapping[int, str] | None = None,
) -> dict[str, object]:
    """Add overlay-visible kinematics, charge, occupancy, and 2→3 logits."""
    endpoints = row.get("endpoints") or []
    states = track_state_by_station(event, endpoints) if endpoints else {}
    first_index = None if not endpoints else int(endpoints[0]["index"])
    pdg = None
    if first_index is not None and event.truth_pdg is not None:
        pdg = int(event.truth_pdg[first_index])
    origin_run = row.get("origin_run_id")
    if origin_run is None and event.origin_run_id is not None and first_index is not None:
        origin_run = int(event.origin_run_id[first_index])
    source_id = None
    if origin_run is not None and source_by_namespaced_run:
        source_id = source_by_namespaced_run.get(int(origin_run))
    logits = _edge_logit_table(row)
    delta = row.get("production_margin")
    if delta is None:
        delta = production_margin(row.get("u_truth"), row.get("u_best_solver_fragment"))
    payload = {
        **dict(row),
        "origin_run_id": None if origin_run is None else int(origin_run),
        "origin_source_id": source_id,
        "charge": charge_from_pdg(pdg) or charge_from_source_id(source_id),
        "truth_pdg": pdg,
        "n_complete_truth_in_event": int(row.get("n_complete_truth_in_event") or 0),
        "station_occupancy": station_occupancy(event),
        "track_state_by_station": {str(station): states[station] for station in states},
        "truth_edge_logits": logits,
        "logit_2to3": logits["2->3"],
        "s3_state": states.get(3),
        "production_margin": None if delta is None else float(delta),
        "margin_bin": classify_margin_bin(None if delta is None else float(delta), margin=PACKING_MARGIN),
    }
    payload.pop("edges", None)
    payload.pop("selected_endpoint_blockers", None)
    payload.pop("strongest_conflicting_hypothesis", None)
    payload.pop("production_by_length", None)
    payload.pop("miner_by_length", None)
    payload.pop("solver_competitor", None)
    payload.pop("miner_strongest", None)
    return payload


def select_failure_population(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, object]]:
    return [dict(row) for row in rows if str(row.get("payload_id")) == FAILURE_PAYLOAD]


def align_draw01_twins(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[object, ...], dict[str, Mapping[str, Any]]]:
    paired: dict[tuple[object, ...], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        payload = str(row.get("payload_id") or "")
        if payload not in {CHART_PAYLOAD, FAILURE_PAYLOAD}:
            continue
        origin = row.get("origin_signature")
        if origin is None or row.get("truth_id") is None:
            continue
        key = (tuple(tuple(item) for item in origin), int(row["truth_id"]))
        role = "twin" if payload == FAILURE_PAYLOAD else "chart"
        paired[key][role] = row
    return {key: value for key, value in paired.items() if "chart" in value and "twin" in value}


def select_failure_core(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, object]]:
    """Unselected, near/hard, or common-SE(3) truth-utility drop on draw_01+common."""
    twins = align_draw01_twins(rows)
    drop_keys = set()
    for key, pair in twins.items():
        chart_u = pair["chart"].get("u_truth")
        twin_u = pair["twin"].get("u_truth")
        if chart_u is None or twin_u is None:
            continue
        if float(twin_u) < float(chart_u):
            drop_keys.add(key)
    core: list[dict[str, object]] = []
    for row in select_failure_population(rows):
        origin = row.get("origin_signature")
        key = None
        if origin is not None and row.get("truth_id") is None:
            key = None
        elif origin is not None:
            key = (tuple(tuple(item) for item in origin), int(row["truth_id"]))
        delta = row.get("production_margin")
        near_or_hard = delta is not None and float(delta) < float(PACKING_MARGIN)
        if (not row.get("selected")) or near_or_hard or (key in drop_keys):
            item = dict(row)
            item["truth_utility_drop"] = bool(key in drop_keys)
            core.append(item)
    return core


def _collect(rows: Sequence[Mapping[str, Any]], getter) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = getter(row)
        if value is None:
            continue
        values.append(float(value))
    return values


def _state_getter(station: int, key: str):
    def _get(row: Mapping[str, Any]) -> float | None:
        block = (row.get("track_state_by_station") or {}).get(str(station))
        if not isinstance(block, Mapping) or key not in block:
            return None
        return float(block[key])

    return _get


def _occupancy_getter(station: int):
    def _get(row: Mapping[str, Any]) -> float | None:
        block = row.get("station_occupancy")
        if not isinstance(block, Mapping):
            return None
        value = block.get(f"s{station}")
        return None if value is None else float(value)

    return _get


def summarize_numeric(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "median": None, "q05": None, "q95": None, "mean": None}
    return {
        "n": int(len(values)),
        "median": float(_median(values)),
        "q05": float(quantile(values, 0.05)),
        "q95": float(quantile(values, 0.95)),
        "mean": float(sum(values) / len(values)),
    }


def summarize_overlay_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, object]:
    population = select_failure_population(rows)
    core = select_failure_core(rows)
    twins = align_draw01_twins(rows)
    delta_all = _collect(rows, lambda row: row.get("production_margin"))
    delta_fail = _collect(population, lambda row: row.get("production_margin"))
    logit_all = _collect(rows, lambda row: row.get("logit_2to3"))
    logit_fail = _collect(population, lambda row: row.get("logit_2to3"))
    charges = Counter(str(row.get("charge") or "unknown") for row in rows)
    sources = Counter(str(row.get("origin_source_id") or "unknown") for row in rows)
    drop = 0
    for pair in twins.values():
        chart_u = pair["chart"].get("u_truth")
        twin_u = pair["twin"].get("u_truth")
        if chart_u is not None and twin_u is not None and float(twin_u) < float(chart_u):
            drop += 1
    return {
        "n_complete": int(len(rows)),
        "n_events": int(
            len({(row.get("payload_id"), row.get("run_id"), row.get("event_id")) for row in rows})
        ),
        "n_draw01_plus_common": int(len(population)),
        "n_failure_core": int(len(core)),
        "selected": int(sum(1 for row in rows if row.get("selected"))),
        "fragment_winners": int(sum(1 for row in rows if row.get("production_fragment_winner"))),
        "hard": int(sum(1 for row in rows if row.get("margin_bin") == "hard")),
        "near_boundary": int(sum(1 for row in rows if row.get("margin_bin") == "near_boundary")),
        "short_station_hard_or_near": int(
            sum(1 for row in rows if row.get("short_station_bin") in {"hard", "near_boundary"})
        ),
        "multiplicity_median": None
        if not rows
        else float(_median([float(row.get("n_complete_truth_in_event") or 0) for row in rows])),
        "production_margin": summarize_numeric(delta_all),
        "production_margin_draw01_plus_common": summarize_numeric(delta_fail),
        "logit_2to3": summarize_numeric(logit_all),
        "logit_2to3_draw01_plus_common": summarize_numeric(logit_fail),
        "hard_rate_draw01_plus_common": None
        if not population
        else float(sum(1 for row in population if (row.get("production_margin") is not None and float(row["production_margin"]) < PACKING_MARGIN)) / len(population)),
        "winner_rate_draw01_plus_common": None
        if not population
        else float(sum(1 for row in population if row.get("production_fragment_winner")) / len(population)),
        "aligned_draw01_twins": int(len(twins)),
        "truth_utility_drop_twins": int(drop),
        "charge": dict(charges),
        "origin_source": dict(sources),
        "kinematics": {
            f"s{station}_{key}": summarize_numeric(_collect(rows, _state_getter(station, key)))
            for station in STATIONS
            for key in STATE_KEYS
        },
        "occupancy": {
            f"s{station}": summarize_numeric(_collect(rows, _occupancy_getter(station)))
            for station in STATIONS
        },
    }


def summarize_identity_events(
    events: Sequence[EventTracklets],
    *,
    source_id: str,
) -> dict[str, object]:
    assert_sources_allowed([source_id])
    states: dict[str, list[float]] = {
        f"s{station}_{key}": [] for station in STATIONS for key in STATE_KEYS
    }
    occupancy = {f"s{station}": [] for station in STATIONS}
    charges: Counter[str] = Counter()
    n_truth = 0
    for event in events:
        occ = station_occupancy(event)
        for station in STATIONS:
            occupancy[f"s{station}"].append(float(occ[f"s{station}"]))
        if event.truth_particle_id is None:
            continue
        counted: set[int] = set()
        for index, truth_id in enumerate(event.truth_particle_id):
            if int(truth_id) < 0:
                continue
            if int(truth_id) not in counted:
                counted.add(int(truth_id))
                n_truth += 1
                pdg = None if event.truth_pdg is None else int(event.truth_pdg[index])
                charges[charge_from_pdg(pdg) or charge_from_source_id(source_id) or "unknown"] += 1
            station = int(event.station_id[index])
            x_mm, y_mm, tx, ty = (float(value) for value in event.state[index])
            if station in STATIONS:
                states[f"s{station}_x_mm"].append(x_mm)
                states[f"s{station}_y_mm"].append(y_mm)
                states[f"s{station}_tx"].append(tx)
                states[f"s{station}_ty"].append(ty)
    return {
        "source_id": source_id,
        "charge_label": charge_from_source_id(source_id),
        "n_events": int(len(events)),
        "n_truth_tracks": int(n_truth),
        "charge": dict(charges),
        "kinematics": {name: summarize_numeric(values) for name, values in states.items()},
        "occupancy": {name: summarize_numeric(values) for name, values in occupancy.items()},
        "identity_bank": True,
    }


def _inside_table(
    query_rows: Sequence[Mapping[str, Any]],
    reference_rows: Sequence[Mapping[str, Any]],
) -> dict[str, float | None]:
    table: dict[str, float | None] = {}
    for station in STATIONS:
        for key in STATE_KEYS:
            name = f"s{station}_{key}"
            table[name] = fraction_inside_quantiles(
                _collect(query_rows, _state_getter(station, key)),
                _collect(reference_rows, _state_getter(station, key)),
            )
        table[f"occupancy_s{station}"] = fraction_inside_quantiles(
            _collect(query_rows, _occupancy_getter(station)),
            _collect(reference_rows, _occupancy_getter(station)),
        )
    return table


def _mean(values: Iterable[float | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    if not present:
        return None
    return float(sum(present) / len(present))


def charge_coverage(
    query_rows: Sequence[Mapping[str, Any]],
    reference_rows: Sequence[Mapping[str, Any]],
    *,
    minimum: float = MIN_CHARGE_FRACTION,
) -> dict[str, object]:
    query = Counter(str(row.get("charge") or "unknown") for row in query_rows)
    reference = Counter(str(row.get("charge") or "unknown") for row in reference_rows)
    n_query = max(sum(query.values()), 1)
    n_ref = max(sum(reference.values()), 1)
    required = [
        name
        for name, count in query.items()
        if name in {"mu_minus", "mu_plus"} and count / n_query >= minimum
    ]
    missing = [
        name
        for name in required
        if reference[name] / n_ref < minimum
    ]
    return {
        "query_charge": dict(query),
        "reference_charge": dict(reference),
        "required_charges": required,
        "missing_charges": missing,
        "covered": not missing,
    }


def coverage_report(
    train_rows: Sequence[Mapping[str, Any]],
    transfer_rows: Sequence[Mapping[str, Any]],
    *,
    development_rows: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, object]:
    failure = select_failure_core(transfer_rows)
    if not failure:
        failure = select_failure_population(transfer_rows)
    inside = _inside_table(failure, train_rows)
    kinematic_keys = [f"s{station}_{key}" for station in STATIONS for key in STATE_KEYS]
    s3_keys = [f"s3_{key}" for key in STATE_KEYS] + [f"s2_{key}" for key in STATE_KEYS]
    occupancy_keys = [f"occupancy_s{station}" for station in STATIONS]
    train_fail = select_failure_population(train_rows)
    transfer_fail = select_failure_population(transfer_rows)
    train_hard = None if not train_fail else float(
        sum(1 for row in train_fail if row.get("production_margin") is not None and float(row["production_margin"]) < PACKING_MARGIN)
        / len(train_fail)
    )
    transfer_hard = None if not transfer_fail else float(
        sum(1 for row in transfer_fail if row.get("production_margin") is not None and float(row["production_margin"]) < PACKING_MARGIN)
        / len(transfer_fail)
    )
    train_winner = None if not train_fail else float(
        sum(1 for row in train_fail if row.get("production_fragment_winner")) / len(train_fail)
    )
    transfer_winner = None if not transfer_fail else float(
        sum(1 for row in transfer_fail if row.get("production_fragment_winner")) / len(transfer_fail)
    )
    train_delta = _collect(train_fail, lambda row: row.get("production_margin"))
    transfer_delta = _collect(transfer_fail, lambda row: row.get("production_margin"))
    train_logit = _collect(train_fail, lambda row: row.get("logit_2to3"))
    transfer_logit = _collect(transfer_fail, lambda row: row.get("logit_2to3"))
    charges = charge_coverage(failure, train_rows)
    phase_space = {
        "failure_core_n": int(len(failure)),
        "kinematic_inside": {key: inside[key] for key in kinematic_keys},
        "kinematic_mean_inside": _mean(inside[key] for key in kinematic_keys),
        "s3_kinematic_mean_inside": _mean(inside[key] for key in s3_keys),
        "occupancy_inside": {key: inside[key] for key in occupancy_keys},
        "occupancy_mean_inside": _mean(inside[key] for key in occupancy_keys),
        "charge": charges,
    }
    characteristic = {
        "train_hard_rate_draw01_plus_common": train_hard,
        "transfer_hard_rate_draw01_plus_common": transfer_hard,
        "hard_rate_transfer_minus_train": None
        if train_hard is None or transfer_hard is None
        else float(transfer_hard - train_hard),
        "train_winner_rate_draw01_plus_common": train_winner,
        "transfer_winner_rate_draw01_plus_common": transfer_winner,
        "winner_rate_transfer_minus_train": None
        if train_winner is None or transfer_winner is None
        else float(transfer_winner - train_winner),
        "train_median_delta_draw01_plus_common": None if not train_delta else float(_median(train_delta)),
        "transfer_median_delta_draw01_plus_common": None if not transfer_delta else float(_median(transfer_delta)),
        "median_delta_train_minus_transfer": None
        if not train_delta or not transfer_delta
        else float(_median(train_delta) - _median(transfer_delta)),
        "train_median_logit_2to3_draw01_plus_common": None if not train_logit else float(_median(train_logit)),
        "transfer_median_logit_2to3_draw01_plus_common": None if not transfer_logit else float(_median(transfer_logit)),
        "median_logit_2to3_train_minus_transfer": None
        if not train_logit or not transfer_logit
        else float(_median(train_logit) - _median(transfer_logit)),
    }
    if development_rows is not None:
        phase_space["development_failure_core_n"] = int(len(select_failure_core(development_rows)))
    return {
        "failure_payload": FAILURE_PAYLOAD,
        "transfer_role": "development_diagnostic_only",
        "phase_space": phase_space,
        "source_characteristic": characteristic,
    }


def recommend_diversity_next(report: Mapping[str, Any]) -> dict[str, object]:
    """Frozen rule: written before the audit numbers are opened."""
    phase = report["phase_space"]
    char = report["source_characteristic"]
    kinematic = phase.get("kinematic_mean_inside")
    s3 = phase.get("s3_kinematic_mean_inside")
    occupancy = phase.get("occupancy_mean_inside")
    charge_ok = bool((phase.get("charge") or {}).get("covered"))
    phase_fail = []
    if kinematic is None or float(kinematic) < MIN_KINEMATIC_INSIDE:
        phase_fail.append("kinematic_mean_inside")
    if s3 is None or float(s3) < MIN_S3_KINEMATIC_INSIDE:
        phase_fail.append("s3_kinematic_mean_inside")
    if occupancy is None or float(occupancy) < MIN_OCCUPANCY_INSIDE:
        phase_fail.append("occupancy_mean_inside")
    if not charge_ok:
        phase_fail.append("charge")

    char_fail = []
    hard_shift = char.get("hard_rate_transfer_minus_train")
    winner_shift = char.get("winner_rate_transfer_minus_train")
    delta_shift = char.get("median_delta_train_minus_transfer")
    logit_shift = char.get("median_logit_2to3_train_minus_transfer")
    if hard_shift is None or float(hard_shift) > MAX_DELTA_HARD_RATE_SHIFT:
        char_fail.append("hard_rate_transfer_minus_train")
    if winner_shift is None or float(winner_shift) > MAX_WINNER_RATE_SHIFT:
        char_fail.append("winner_rate_transfer_minus_train")
    if delta_shift is None or float(delta_shift) > MAX_MEDIAN_DELTA_SHIFT:
        char_fail.append("median_delta_train_minus_transfer")
    if logit_shift is None or float(logit_shift) > MAX_MEDIAN_LOGIT23_SHIFT:
        char_fail.append("median_logit_2to3_train_minus_transfer")

    if phase_fail:
        coverage_class = "source_phase_space_undercoverage"
        authorize = True
        next_step = "add_source_disjoint_four_station_training_sources"
    elif char_fail:
        coverage_class = "source_characteristic_undercoverage"
        authorize = True
        next_step = "add_source_disjoint_four_station_training_sources"
    else:
        coverage_class = "current_train_covers_failure_region"
        authorize = False
        next_step = "discuss_architecture_level_relative_gauge_equivariant_representation"

    return {
        "coverage_class": coverage_class,
        "phase_space_failures": phase_fail,
        "source_characteristic_failures": char_fail,
        "authorize_new_training_sources": bool(authorize),
        "freeze_workbook62_objective": True,
        "frozen_objective": WB62_OBJECTIVE,
        "workbook56_transfer_is_final_gate": False,
        "retune_loss_weight_reduction_or_operating_point": False,
        "enlarge_misalignment_envelope": False,
        "add_new_dof": False,
        "continue_to_15d_relative_wls": False,
        "reserved_blind_sources": list(RESERVED_BLIND_SOURCES),
        "proposed_new_train_sources": list(PROPOSED_NEW_TRAIN_SOURCES) if authorize else list(CURRENT_TRAIN_SOURCES),
        "next_step": next_step,
        "if_expanded_diversity_still_fails_same_way": "discuss_architecture_level_relative_gauge_equivariant_representation",
    }
