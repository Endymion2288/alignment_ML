from __future__ import annotations

from dataclasses import replace

import numpy as np

from datasets.pooled_physical import (
    merge_propagation_records,
    namespace_physical_source,
    write_pooled_propagations_root,
    write_pooled_tracklets_root,
)
from datasets.propagation_loader import PropagationRecords, load_propagation_records
from datasets.root_loader import load_events
from datasets.synthetic import make_synthetic_tracklet_root


def _records(events):
    rows = []
    for event in events:
        source = int(event.indices_for_station(0)[0])
        target = int(event.indices_for_station(1)[0])
        rows.append((event, source, target))
    return PropagationRecords(
        run_id=np.asarray([event.run_id for event, _, _ in rows], dtype=np.int64),
        event_id=np.asarray([event.event_id for event, _, _ in rows], dtype=np.int64),
        source_tracklet_id=np.asarray(
            [event.tracklet_id[source] for event, source, _ in rows], dtype=np.int32
        ),
        target_tracklet_id=np.asarray(
            [event.tracklet_id[target] for event, _, target in rows], dtype=np.int32
        ),
        source_station_id=np.zeros(len(rows), dtype=np.int16),
        target_station_id=np.ones(len(rows), dtype=np.int16),
        truth_particle_id=np.asarray(
            [event.truth_particle_id[source] for event, source, _ in rows], dtype=np.int64
        ),
        target_z_mm=np.asarray([event.z_mm[target] for event, _, target in rows], dtype=np.float64),
        prediction=np.asarray([event.state[target] for event, _, target in rows], dtype=np.float64),
        covariance=np.asarray(
            [event.covariance[target] for event, _, target in rows], dtype=np.float64
        ),
        success=np.ones(len(rows), dtype=bool),
        has_covariance=np.ones(len(rows), dtype=bool),
        q_over_p_mode=np.zeros(len(rows), dtype=np.int8),
        source_q_over_p_per_mev=np.full(len(rows), np.nan, dtype=np.float64),
    )


def test_pooled_namespace_preserves_states_and_disambiguates_provenance(tmp_path):
    source = make_synthetic_tracklet_root(tmp_path / "source.root")
    events = load_events(source, require_mc_labels=True)[:2]
    # Deliberately reuse identical original run/event IDs in a second source.
    duplicate_events = [replace(event) for event in events]
    left_events, left_records, left_map = namespace_physical_source(
        events, _records(events), namespace_base=9_000_000_000
    )
    right_events, right_records, right_map = namespace_physical_source(
        duplicate_events, _records(duplicate_events), namespace_base=9_000_000_100
    )
    assert left_map[0]["original_run_id"] == right_map[0]["original_run_id"] == 1
    assert left_map[0]["namespaced_run_id"] != right_map[0]["namespaced_run_id"]
    merged_records = merge_propagation_records((left_records, right_records))
    tracklets_path = tmp_path / "pooled_tracklets.root"
    propagations_path = tmp_path / "pooled_propagations.root"
    provenance = {"source_pooling": "unit_test"}
    write_pooled_tracklets_root([*left_events, *right_events], tracklets_path, provenance)
    write_pooled_propagations_root(merged_records, propagations_path, provenance)

    loaded_events = load_events(tracklets_path, require_mc_labels=True)
    loaded_records = load_propagation_records(propagations_path)
    assert len(loaded_events) == 4
    assert {event.run_id for event in loaded_events} == {9_000_000_000, 9_000_000_100}
    assert loaded_records.size == 4
    assert set(loaded_records.run_id.tolist()) == {9_000_000_000, 9_000_000_100}
    np.testing.assert_allclose(loaded_events[0].state, events[0].state)
    np.testing.assert_allclose(loaded_records.prediction[0], events[0].state[1])
