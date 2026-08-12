"""Deterministic overlays of single-particle events into multi-track samples."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import uproot

from .root_loader import EventTracklets
from .schema import CANONICAL_TREE_NAME, SCHEMA_VERSION, covariance_columns
from .propagation_loader import PropagationRecords, load_propagation_records
from geometry.propagation import mahalanobis_chi2


Q_OVER_P_FIELDS = (
    "q_over_p_per_mev",
    "q_over_p_from_momentum_per_mev",
    "q_over_p_variance_per_mev2",
    "has_q_over_p_covariance",
)


# These values are provenance only.  They are deliberately not returned by
# pair_feature_matrix, so the MLP cannot exploit how a synthetic row was made.
SYNTHETIC_ROLE_TRUE = 0
SYNTHETIC_ROLE_RANDOM_EASY_FAKE = 1
SYNTHETIC_ROLE_FIELD_HARD_FAKE = 2

SYNTHETIC_ROLE_NAMES = {
    SYNTHETIC_ROLE_TRUE: "true_tracklet",
    SYNTHETIC_ROLE_RANDOM_EASY_FAKE: "random_easy_fake",
    SYNTHETIC_ROLE_FIELD_HARD_FAKE: "field_aware_hard_negative",
}

SyntheticPayloadRow = tuple[EventTracklets, int, int, int, float, int, int, float]
TruthCounterpartKey = tuple[int, int, int, int]


@dataclass(frozen=True)
class SourceTrack:
    """One truth particle represented by one selected tracklet per station."""

    event: EventTracklets
    truth_particle_id: int
    rows_by_station: dict[int, int]


@dataclass(frozen=True)
class SyntheticOverlaySummary:
    source_events: int
    eligible_source_tracks: int
    output_events: int
    requested_tracks_per_event: int
    true_tracklets_written: int
    missing_true_tracklets: int
    fake_tracklets_written: int
    random_easy_fake_tracklets_written: int
    field_hard_fake_tracklets_written: int
    random_easy_fake_selection_failures: int
    field_hard_fake_selection_failures: int
    duplicate_source_tracks_prevented: bool
    optional_q_over_p_fields: tuple[str, ...]
    station_ids: tuple[int, ...]
    seed: int


def select_complete_truth_tracks(
    events: Sequence[EventTracklets],
    station_ids: Sequence[int],
    minimum_truth_match_fraction: float = 0.99,
) -> list[SourceTrack]:
    """Select complete truth tracks with one deterministic segment per station.

    When reconstruction supplies multiple local segments for the same truth
    particle and station, the one with highest truth-match fraction is used;
    ties are broken by lower local chi-square and then tracklet ID.  This is a
    source-sample selection convention, not an association target.
    """
    stations = tuple(sorted({int(station) for station in station_ids}))
    if not stations:
        raise ValueError("at least one station is required")
    if not np.isfinite(minimum_truth_match_fraction) or not 0.0 <= minimum_truth_match_fraction <= 1.0:
        raise ValueError("minimum_truth_match_fraction must be between zero and one")
    selected: list[SourceTrack] = []
    for event in events:
        if event.truth_particle_id is None or event.truth_match_fraction is None:
            raise ValueError("synthetic overlays require MC truth labels and match fractions")
        available_by_station: dict[int, set[int]] = {}
        for station in stations:
            rows = event.indices_for_station(station)
            eligible_rows = rows[
                (event.truth_particle_id[rows] >= 0)
                & np.isfinite(event.truth_match_fraction[rows])
                & (event.truth_match_fraction[rows] >= minimum_truth_match_fraction)
            ]
            available_by_station[station] = {
                int(event.truth_particle_id[row]) for row in eligible_rows
            }
        common_truth_ids = set.intersection(*available_by_station.values())
        for truth_particle_id in sorted(common_truth_ids):
            rows_by_station: dict[int, int] = {}
            for station in stations:
                rows = event.indices_for_station(station)
                candidates = [
                    int(row)
                    for row in rows
                    if int(event.truth_particle_id[row]) == truth_particle_id
                    and np.isfinite(event.truth_match_fraction[row])
                    and event.truth_match_fraction[row] >= minimum_truth_match_fraction
                ]
                if not candidates:
                    raise RuntimeError("internal complete-track selection inconsistency")
                rows_by_station[station] = min(
                    candidates,
                    key=lambda row: (
                        -float(event.truth_match_fraction[row]),
                        float(event.chi2[row]) if np.isfinite(event.chi2[row]) else np.inf,
                        int(event.tracklet_id[row]),
                    ),
                )
            selected.append(
                SourceTrack(
                    event=event,
                    truth_particle_id=truth_particle_id,
                    rows_by_station=rows_by_station,
                )
            )
    return selected


def _q_over_p_fields_available(events: Sequence[EventTracklets]) -> tuple[str, ...]:
    return tuple(
        field
        for field in Q_OVER_P_FIELDS
        if all(getattr(event, field) is not None for event in events)
    )


def _append_tracklet(
    event: EventTracklets,
    row: int,
    tracklet_id: int,
    run_id: int,
    event_id: int,
    truth_particle_id: int,
    truth_pdg: int,
    truth_match_fraction: float,
    synthetic_role: int,
    synthetic_hard_anchor_station: int,
    synthetic_hard_anchor_chi2: float,
    q_over_p_fields: Sequence[str],
    columns: dict[str, list[int | float | bool]],
    states: list[np.ndarray],
    covariances: list[np.ndarray],
) -> None:
    columns["run_id"].append(run_id)
    columns["event_id"].append(event_id)
    columns["station_id"].append(int(event.station_id[row]))
    columns["tracklet_id"].append(tracklet_id)
    columns["z_mm"].append(float(event.z_mm[row]))
    columns["chi2"].append(float(event.chi2[row]))
    columns["ndof"].append(float(event.ndof[row]))
    columns["n_hit"].append(int(event.n_hit[row]))
    columns["hit_pattern"].append(int(event.hit_pattern[row]))
    columns["truth_particle_id"].append(truth_particle_id)
    columns["truth_pdg"].append(truth_pdg)
    columns["truth_match_fraction"].append(truth_match_fraction)
    columns["origin_run_id"].append(event.run_id)
    columns["origin_event_id"].append(event.event_id)
    columns["origin_tracklet_id"].append(int(event.tracklet_id[row]))
    columns["synthetic_role"].append(int(synthetic_role))
    columns["synthetic_hard_anchor_station"].append(int(synthetic_hard_anchor_station))
    columns["synthetic_hard_anchor_chi2"].append(float(synthetic_hard_anchor_chi2))
    for field in q_over_p_fields:
        value = getattr(event, field)
        assert value is not None
        columns[field].append(value[row].item())
    states.append(np.asarray(event.state[row], dtype=np.float64))
    covariances.append(np.asarray(event.covariance[row], dtype=np.float64))


def _origin_key(event: EventTracklets, row: int) -> tuple[int, int, int]:
    """Return immutable pre-overlay provenance for one tracklet row."""
    return (event.run_id, event.event_id, int(event.tracklet_id[row]))


def _truth_counterpart_index(
    payload: Sequence[SyntheticPayloadRow],
) -> dict[TruthCounterpartKey, tuple[EventTracklets, int]]:
    """Index retained same-truth endpoints for relative hard-negative selection."""
    true_rows = [
        (event, row, truth_id)
        for event, row, truth_id, _, _, role, _, _ in payload
        if truth_id >= 0 and role == SYNTHETIC_ROLE_TRUE
    ]
    result: dict[TruthCounterpartKey, tuple[EventTracklets, int]] = {}
    for source_event, source_row, truth_id in true_rows:
        source_station = int(source_event.station_id[source_row])
        source_key = _origin_key(source_event, source_row)
        for target_event, target_row, target_truth_id in true_rows:
            if target_truth_id != truth_id or int(target_event.station_id[target_row]) <= source_station:
                continue
            key = (*source_key, int(target_event.station_id[target_row]))
            previous = result.get(key)
            if previous is None:
                result[key] = (target_event, target_row)
            elif previous[0] is not target_event or previous[1] != target_row:
                raise RuntimeError("ambiguous retained truth counterpart for a synthetic source")
    return result


def _passes_truth_relative_band(
    candidate_chi2: float,
    truth_chi2: float,
    minimum_ratio: float | None,
    maximum_ratio: float | None,
) -> bool:
    """Require a hard fake to be near but not more compatible than truth."""
    if minimum_ratio is None and maximum_ratio is None:
        return True
    if not np.isfinite(candidate_chi2) or not np.isfinite(truth_chi2) or truth_chi2 <= 0.0:
        return False
    ratio = candidate_chi2 / truth_chi2
    return bool(
        (minimum_ratio is None or ratio >= minimum_ratio)
        and (maximum_ratio is None or ratio <= maximum_ratio)
    )


def _valid_covariance(covariance: np.ndarray) -> bool:
    values = np.asarray(covariance, dtype=np.float64)
    return bool(
        values.shape == (4, 4)
        and np.isfinite(values).all()
        and np.allclose(values, values.T, rtol=1.0e-7, atol=1.0e-12)
        and np.all(np.diag(values) > 0.0)
    )


def _prediction_index(records: PropagationRecords) -> dict[tuple[int, int, int, int], int]:
    """Index usable mode-0 Acts states by source provenance and target station."""
    result: dict[tuple[int, int, int, int], int] = {}
    modes = records.q_over_p_mode
    for row in range(records.size):
        if (
            (modes is not None and int(modes[row]) != 0)
            or not bool(records.success[row])
            or not bool(records.has_covariance[row])
            or not _valid_covariance(records.covariance[row])
        ):
            continue
        key = (
            int(records.run_id[row]),
            int(records.event_id[row]),
            int(records.source_tracklet_id[row]),
            int(records.target_station_id[row]),
        )
        previous = result.setdefault(key, row)
        if previous != row:
            raise ValueError(f"duplicate usable Acts prediction for source/target key {key}")
    return result


def _field_chi2(
    source_event: EventTracklets,
    source_row: int,
    target_event: EventTracklets,
    target_row: int,
    records: PropagationRecords,
    predictions: Mapping[tuple[int, int, int, int], int],
    target_z_tolerance_mm: float,
) -> float | None:
    """Evaluate a cross-event candidate using a real same-payload Acts state."""
    source_station = int(source_event.station_id[source_row])
    target_station = int(target_event.station_id[target_row])
    if source_station >= target_station:
        return None
    row = predictions.get((*_origin_key(source_event, source_row), target_station))
    if row is None:
        return None
    if not np.isclose(
        float(records.target_z_mm[row]),
        float(target_event.z_mm[target_row]),
        rtol=0.0,
        atol=target_z_tolerance_mm,
    ):
        return None
    combined = records.covariance[row] + target_event.covariance[target_row]
    if not _valid_covariance(combined):
        return None
    try:
        value = mahalanobis_chi2(target_event.state[target_row] - records.prediction[row], combined)
    except ValueError:
        return None
    return float(value) if np.isfinite(value) else None


def _minimum_anchor_chi2(
    event: EventTracklets,
    row: int,
    anchors: Sequence[tuple[EventTracklets, int]],
    records: PropagationRecords,
    predictions: Mapping[tuple[int, int, int, int], int],
    target_z_tolerance_mm: float,
) -> float | None:
    """Minimum physically propagated distance between one fake and true rows."""
    values: list[float] = []
    station = int(event.station_id[row])
    for anchor_event, anchor_row in anchors:
        anchor_station = int(anchor_event.station_id[anchor_row])
        if anchor_station < station:
            chi2 = _field_chi2(
                anchor_event,
                anchor_row,
                event,
                row,
                records,
                predictions,
                target_z_tolerance_mm,
            )
        elif station < anchor_station:
            chi2 = _field_chi2(
                event,
                row,
                anchor_event,
                anchor_row,
                records,
                predictions,
                target_z_tolerance_mm,
            )
        else:
            chi2 = None
        if chi2 is not None:
            values.append(chi2)
    return min(values) if values else None


def _draw_random_easy_fake(
    pool: Sequence[tuple[EventTracklets, int]],
    anchors: Sequence[tuple[EventTracklets, int]],
    used_origins: set[tuple[int, int, int]],
    rng: np.random.Generator,
    records: PropagationRecords | None,
    predictions: Mapping[tuple[int, int, int, int], int] | None,
    minimum_chi2: float | None,
    max_trials: int,
    target_z_tolerance_mm: float,
) -> tuple[EventTracklets, int, float] | None:
    """Draw an unconditioned fake, optionally rejecting geometry-near rows."""
    if not pool:
        return None
    for candidate_index in rng.permutation(len(pool))[:max_trials]:
        event, row = pool[int(candidate_index)]
        if _origin_key(event, row) in used_origins:
            continue
        minimum = None
        if records is not None and predictions is not None:
            minimum = _minimum_anchor_chi2(
                event,
                row,
                anchors,
                records,
                predictions,
                target_z_tolerance_mm,
            )
        if minimum_chi2 is not None and (minimum is None or minimum < minimum_chi2):
            continue
        return event, row, float("nan") if minimum is None else float(minimum)
    return None


def _draw_field_hard_fake(
    pool: Sequence[tuple[EventTracklets, int]],
    target_station: int,
    anchors: Sequence[tuple[EventTracklets, int]],
    used_origins: set[tuple[int, int, int]],
    rng: np.random.Generator,
    records: PropagationRecords,
    predictions: Mapping[tuple[int, int, int, int], int],
    minimum_chi2: float,
    maximum_chi2: float,
    max_trials: int,
    target_z_tolerance_mm: float,
    truth_counterparts: Mapping[TruthCounterpartKey, tuple[EventTracklets, int]],
    minimum_truth_chi2_ratio: float | None,
    maximum_truth_chi2_ratio: float | None,
) -> tuple[EventTracklets, int, int, float] | None:
    """Draw a same-payload near fake, optionally constrained relative to truth."""
    source_anchors = [
        (event, row)
        for event, row in anchors
        if int(event.station_id[row]) < target_station
    ]
    if not source_anchors or not pool:
        return None
    candidates = rng.permutation(len(pool))
    anchors_order = rng.permutation(len(source_anchors))
    attempts = 0
    for anchor_index in anchors_order:
        source_event, source_row = source_anchors[int(anchor_index)]
        truth_counterpart = truth_counterparts.get((*_origin_key(source_event, source_row), target_station))
        truth_chi2: float | None = None
        if minimum_truth_chi2_ratio is not None or maximum_truth_chi2_ratio is not None:
            if truth_counterpart is None:
                continue
            truth_chi2 = _field_chi2(
                source_event,
                source_row,
                truth_counterpart[0],
                truth_counterpart[1],
                records,
                predictions,
                target_z_tolerance_mm,
            )
        for candidate_index in candidates:
            if attempts >= max_trials:
                return None
            attempts += 1
            event, row = pool[int(candidate_index)]
            if _origin_key(event, row) in used_origins:
                continue
            chi2 = _field_chi2(
                source_event,
                source_row,
                event,
                row,
                records,
                predictions,
                target_z_tolerance_mm,
            )
            if chi2 is None or chi2 < minimum_chi2 or chi2 > maximum_chi2:
                continue
            if not _passes_truth_relative_band(
                chi2,
                float("nan") if truth_chi2 is None else truth_chi2,
                minimum_truth_chi2_ratio,
                maximum_truth_chi2_ratio,
            ):
                continue
            return event, row, int(source_event.station_id[source_row]), float(chi2)
    return None


def write_synthetic_multitrack_root(
    events: Sequence[EventTracklets],
    output_path: str | Path,
    output_events: int,
    tracks_per_event: int,
    station_ids: Sequence[int] = (0, 1, 2, 3),
    missing_tracklet_probability: float = 0.0,
    fake_mean_per_station: float = 0.0,
    random_easy_fake_mean_per_station: float | None = None,
    random_easy_min_chi2: float | None = None,
    hard_negative_mean_per_target_station: float = 0.0,
    hard_negative_chi2_min: float = 1.0,
    hard_negative_chi2_max: float = 100.0,
    hard_negative_max_trials: int = 256,
    hard_negative_min_truth_chi2_ratio: float | None = None,
    hard_negative_max_truth_chi2_ratio: float | None = None,
    physical_propagations: str | Path | None = None,
    target_z_tolerance_mm: float = 1.0e-6,
    allow_duplicate_source_tracks: bool = False,
    seed: int = 12345,
    synthetic_run_id: int = 990000,
    minimum_truth_match_fraction: float = 0.99,
) -> SyntheticOverlaySummary:
    """Write deterministic multi-track overlays in the canonical ROOT schema.

    Source truth IDs are namespaced per synthetic event and track slot, so
    truth-based association labels remain unambiguous after event overlay.
    All rows are copied from physical refit outputs.  A random fake can be
    required to be field-distant from the true rows, while a hard fake is
    selected only when an actual mode-0 Acts prediction puts it inside a
    declared chi2 band of a true source.  Neither provenance label is exposed
    to the MLP feature matrix.
    """
    if output_events < 1 or tracks_per_event < 1:
        raise ValueError("output_events and tracks_per_event must both be positive")
    if not np.isfinite(missing_tracklet_probability) or not 0.0 <= missing_tracklet_probability <= 1.0:
        raise ValueError("missing_tracklet_probability must be between zero and one")
    if not np.isfinite(fake_mean_per_station) or fake_mean_per_station < 0.0:
        raise ValueError("fake_mean_per_station must be finite and non-negative")
    random_mean = (
        float(fake_mean_per_station)
        if random_easy_fake_mean_per_station is None
        else float(random_easy_fake_mean_per_station)
    )
    if not np.isfinite(random_mean) or random_mean < 0.0:
        raise ValueError("random_easy_fake_mean_per_station must be finite and non-negative")
    if random_easy_fake_mean_per_station is not None and fake_mean_per_station != 0.0:
        raise ValueError(
            "use either legacy fake_mean_per_station or random_easy_fake_mean_per_station"
        )
    if random_easy_min_chi2 is not None and (
        not np.isfinite(random_easy_min_chi2) or random_easy_min_chi2 < 0.0
    ):
        raise ValueError("random_easy_min_chi2 must be finite and non-negative when supplied")
    if not np.isfinite(hard_negative_mean_per_target_station) or hard_negative_mean_per_target_station < 0.0:
        raise ValueError("hard_negative_mean_per_target_station must be finite and non-negative")
    if (
        not np.isfinite(hard_negative_chi2_min)
        or not np.isfinite(hard_negative_chi2_max)
        or hard_negative_chi2_min < 0.0
        or hard_negative_chi2_max < hard_negative_chi2_min
    ):
        raise ValueError("hard-negative chi2 bounds must be finite and ordered")
    if hard_negative_max_trials < 1:
        raise ValueError("hard_negative_max_trials must be positive")
    for name, value in (
        ("hard_negative_min_truth_chi2_ratio", hard_negative_min_truth_chi2_ratio),
        ("hard_negative_max_truth_chi2_ratio", hard_negative_max_truth_chi2_ratio),
    ):
        if value is not None and (not np.isfinite(value) or value < 1.0):
            raise ValueError(f"{name} must be finite and at least one when supplied")
    if (
        hard_negative_min_truth_chi2_ratio is not None
        and hard_negative_max_truth_chi2_ratio is not None
        and hard_negative_max_truth_chi2_ratio < hard_negative_min_truth_chi2_ratio
    ):
        raise ValueError(
            "hard_negative_max_truth_chi2_ratio must not be below the minimum ratio"
        )
    if not np.isfinite(target_z_tolerance_mm) or target_z_tolerance_mm < 0.0:
        raise ValueError("target_z_tolerance_mm must be finite and non-negative")
    stations = tuple(sorted({int(station) for station in station_ids}))
    source_tracks = select_complete_truth_tracks(
        events, stations, minimum_truth_match_fraction=minimum_truth_match_fraction
    )
    if not source_tracks:
        raise ValueError("no complete source tracks satisfy the selection")
    if tracks_per_event > len(source_tracks) and not allow_duplicate_source_tracks:
        raise ValueError(
            "tracks_per_event exceeds independent complete source tracks; "
            "refusing to duplicate a physical trajectory"
        )
    fake_pool: dict[int, list[tuple[EventTracklets, int]]] = {station: [] for station in stations}
    for event in events:
        for station in stations:
            fake_pool[station].extend((event, int(row)) for row in event.indices_for_station(station))
    unavailable_fake_stations = [station for station, pool in fake_pool.items() if not pool]
    if unavailable_fake_stations and (random_mean > 0.0 or hard_negative_mean_per_target_station > 0.0):
        raise ValueError(f"no fake source tracklets for stations {unavailable_fake_stations}")

    records: PropagationRecords | None = None
    predictions: Mapping[tuple[int, int, int, int], int] | None = None
    needs_geometry = hard_negative_mean_per_target_station > 0.0 or random_easy_min_chi2 is not None
    if needs_geometry:
        if physical_propagations is None:
            raise ValueError(
                "physical_propagations is required for geometry-conditioned synthetic fakes"
            )
        records = load_propagation_records(physical_propagations)
        predictions = _prediction_index(records)

    q_over_p_fields = _q_over_p_fields_available(events)
    columns: dict[str, list[int | float | bool]] = {
        "run_id": [],
        "event_id": [],
        "station_id": [],
        "tracklet_id": [],
        "z_mm": [],
        "chi2": [],
        "ndof": [],
        "n_hit": [],
        "hit_pattern": [],
        "truth_particle_id": [],
        "truth_pdg": [],
        "truth_match_fraction": [],
        "origin_run_id": [],
        "origin_event_id": [],
        "origin_tracklet_id": [],
        "synthetic_role": [],
        "synthetic_hard_anchor_station": [],
        "synthetic_hard_anchor_chi2": [],
        **{field: [] for field in q_over_p_fields},
    }
    states: list[np.ndarray] = []
    covariances: list[np.ndarray] = []
    rng = np.random.default_rng(seed)
    true_tracklets_written = 0
    missing_true_tracklets = 0
    fake_tracklets_written = 0
    random_easy_fake_tracklets_written = 0
    field_hard_fake_tracklets_written = 0
    random_easy_fake_selection_failures = 0
    field_hard_fake_selection_failures = 0

    for synthetic_event_id in range(output_events):
        source_indices = rng.choice(
            len(source_tracks),
            size=tracks_per_event,
            replace=allow_duplicate_source_tracks and tracks_per_event > len(source_tracks),
        )
        payload: list[SyntheticPayloadRow] = []
        for slot, source_index in enumerate(source_indices):
            source_track = source_tracks[int(source_index)]
            synthetic_truth_id = synthetic_event_id * tracks_per_event + slot + 1
            first_row = source_track.rows_by_station[stations[0]]
            truth_pdg = int(source_track.event.truth_pdg[first_row]) if source_track.event.truth_pdg is not None else 0
            for station in stations:
                if rng.random() < missing_tracklet_probability:
                    missing_true_tracklets += 1
                    continue
                payload.append(
                    (
                        source_track.event,
                        source_track.rows_by_station[station],
                        synthetic_truth_id,
                        truth_pdg,
                        1.0,
                        SYNTHETIC_ROLE_TRUE,
                        -1,
                        float("nan"),
                    )
                )
                true_tracklets_written += 1
        anchors = [(event, row) for event, row, truth_id, *_ in payload if truth_id >= 0]
        used_origins = {_origin_key(event, row) for event, row, *_ in payload}
        truth_counterparts = _truth_counterpart_index(payload)
        if hard_negative_mean_per_target_station > 0.0:
            assert records is not None and predictions is not None
            for station in stations:
                if station == stations[0]:
                    continue
                for _ in range(int(rng.poisson(hard_negative_mean_per_target_station))):
                    hard = _draw_field_hard_fake(
                        fake_pool[station],
                        station,
                        anchors,
                        used_origins,
                        rng,
                        records,
                        predictions,
                        float(hard_negative_chi2_min),
                        float(hard_negative_chi2_max),
                        hard_negative_max_trials,
                        target_z_tolerance_mm,
                        truth_counterparts,
                        hard_negative_min_truth_chi2_ratio,
                        hard_negative_max_truth_chi2_ratio,
                    )
                    if hard is None:
                        field_hard_fake_selection_failures += 1
                        continue
                    event, row, anchor_station, anchor_chi2 = hard
                    used_origins.add(_origin_key(event, row))
                    payload.append(
                        (
                            event,
                            row,
                            -1,
                            0,
                            float("nan"),
                            SYNTHETIC_ROLE_FIELD_HARD_FAKE,
                            anchor_station,
                            anchor_chi2,
                        )
                    )
                    fake_tracklets_written += 1
                    field_hard_fake_tracklets_written += 1
        for station in stations:
            for _ in range(int(rng.poisson(random_mean))):
                random_fake = _draw_random_easy_fake(
                    fake_pool[station],
                    anchors,
                    used_origins,
                    rng,
                    records,
                    predictions,
                    random_easy_min_chi2,
                    hard_negative_max_trials,
                    target_z_tolerance_mm,
                )
                if random_fake is None:
                    random_easy_fake_selection_failures += 1
                    continue
                event, row, _ = random_fake
                used_origins.add(_origin_key(event, row))
                payload.append(
                    (
                        event,
                        row,
                        -1,
                        0,
                        float("nan"),
                        SYNTHETIC_ROLE_RANDOM_EASY_FAKE,
                        -1,
                        float("nan"),
                    )
                )
                fake_tracklets_written += 1
                random_easy_fake_tracklets_written += 1
        if payload:
            order = rng.permutation(len(payload))
            for tracklet_id, payload_index in enumerate(order):
                (
                    event,
                    row,
                    truth_id,
                    truth_pdg,
                    fraction,
                    role,
                    hard_anchor_station,
                    hard_anchor_chi2,
                ) = payload[int(payload_index)]
                _append_tracklet(
                    event,
                    row,
                    tracklet_id=tracklet_id,
                    run_id=synthetic_run_id,
                    event_id=synthetic_event_id,
                    truth_particle_id=truth_id,
                    truth_pdg=truth_pdg,
                    truth_match_fraction=fraction,
                    synthetic_role=role,
                    synthetic_hard_anchor_station=hard_anchor_station,
                    synthetic_hard_anchor_chi2=hard_anchor_chi2,
                    q_over_p_fields=q_over_p_fields,
                    columns=columns,
                    states=states,
                    covariances=covariances,
                )

    root_columns: dict[str, np.ndarray] = {
        "run_id": np.asarray(columns["run_id"], dtype=np.int64),
        "event_id": np.asarray(columns["event_id"], dtype=np.int64),
        "station_id": np.asarray(columns["station_id"], dtype=np.int8),
        "tracklet_id": np.asarray(columns["tracklet_id"], dtype=np.int32),
        "z_mm": np.asarray(columns["z_mm"], dtype=np.float64),
        "chi2": np.asarray(columns["chi2"], dtype=np.float64),
        "ndof": np.asarray(columns["ndof"], dtype=np.float64),
        "n_hit": np.asarray(columns["n_hit"], dtype=np.int16),
        "hit_pattern": np.asarray(columns["hit_pattern"], dtype=np.uint64),
        "truth_particle_id": np.asarray(columns["truth_particle_id"], dtype=np.int64),
        "truth_pdg": np.asarray(columns["truth_pdg"], dtype=np.int32),
        "truth_match_fraction": np.asarray(columns["truth_match_fraction"], dtype=np.float64),
        "origin_run_id": np.asarray(columns["origin_run_id"], dtype=np.int64),
        "origin_event_id": np.asarray(columns["origin_event_id"], dtype=np.int64),
        "origin_tracklet_id": np.asarray(columns["origin_tracklet_id"], dtype=np.int32),
        "synthetic_role": np.asarray(columns["synthetic_role"], dtype=np.int8),
        "synthetic_hard_anchor_station": np.asarray(
            columns["synthetic_hard_anchor_station"], dtype=np.int8
        ),
        "synthetic_hard_anchor_chi2": np.asarray(
            columns["synthetic_hard_anchor_chi2"], dtype=np.float64
        ),
    }
    if states:
        state_array = np.asarray(states, dtype=np.float64)
        covariance_array = np.asarray(covariances, dtype=np.float64)
    else:
        state_array = np.empty((0, 4), dtype=np.float64)
        covariance_array = np.empty((0, 4, 4), dtype=np.float64)
    root_columns.update(
        {
            "x_mm": state_array[:, 0],
            "y_mm": state_array[:, 1],
            "tx": state_array[:, 2],
            "ty": state_array[:, 3],
        }
    )
    root_columns.update(covariance_columns(covariance_array))
    for field in q_over_p_fields:
        dtype = np.bool_ if field == "has_q_over_p_covariance" else np.float64
        root_columns[field] = np.asarray(columns[field], dtype=dtype)

    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    summary = SyntheticOverlaySummary(
        source_events=len(events),
        eligible_source_tracks=len(source_tracks),
        output_events=output_events,
        requested_tracks_per_event=tracks_per_event,
        true_tracklets_written=true_tracklets_written,
        missing_true_tracklets=missing_true_tracklets,
        fake_tracklets_written=fake_tracklets_written,
        random_easy_fake_tracklets_written=random_easy_fake_tracklets_written,
        field_hard_fake_tracklets_written=field_hard_fake_tracklets_written,
        random_easy_fake_selection_failures=random_easy_fake_selection_failures,
        field_hard_fake_selection_failures=field_hard_fake_selection_failures,
        duplicate_source_tracks_prevented=not allow_duplicate_source_tracks,
        optional_q_over_p_fields=tuple(q_over_p_fields),
        station_ids=stations,
        seed=seed,
    )
    with uproot.recreate(destination) as root_file:
        root_file[CANONICAL_TREE_NAME] = root_columns
        root_file["metadata"] = {
            "schema_version": np.asarray([SCHEMA_VERSION]),
            "coordinate_unit": np.asarray(["mm"]),
            "generator": np.asarray(["datasets.synthetic_overlay.write_synthetic_multitrack_root"]),
            "overlay_config": np.asarray(
                [
                    json.dumps(
                        {
                            "station_ids": stations,
                            "output_events": output_events,
                            "tracks_per_event": tracks_per_event,
                            "missing_tracklet_probability": missing_tracklet_probability,
                            "fake_mean_per_station": fake_mean_per_station,
                            "random_easy_fake_mean_per_station": random_easy_fake_mean_per_station,
                            "random_easy_min_chi2": random_easy_min_chi2,
                            "hard_negative_mean_per_target_station": hard_negative_mean_per_target_station,
                            "hard_negative_chi2_min": hard_negative_chi2_min,
                            "hard_negative_chi2_max": hard_negative_chi2_max,
                            "hard_negative_max_trials": hard_negative_max_trials,
                            "hard_negative_min_truth_chi2_ratio": hard_negative_min_truth_chi2_ratio,
                            "hard_negative_max_truth_chi2_ratio": hard_negative_max_truth_chi2_ratio,
                            "physical_propagations": (
                                None
                                if physical_propagations is None
                                else str(Path(physical_propagations).expanduser().resolve())
                            ),
                            "target_z_tolerance_mm": target_z_tolerance_mm,
                            "allow_duplicate_source_tracks": allow_duplicate_source_tracks,
                            "synthetic_role_codes": SYNTHETIC_ROLE_NAMES,
                            "seed": seed,
                            "synthetic_run_id": synthetic_run_id,
                            "minimum_truth_match_fraction": minimum_truth_match_fraction,
                        },
                        sort_keys=True,
                    )
                ]
            ),
            "overlay_summary": np.asarray([json.dumps(asdict(summary), sort_keys=True)]),
        }
    return summary
