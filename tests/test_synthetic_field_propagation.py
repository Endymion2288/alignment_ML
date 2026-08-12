from __future__ import annotations

import numpy as np
import uproot

from baselines.field_chi2_matching import (
    build_field_candidates,
    candidate_labels,
    greedy_field_chi2_match,
)
from datasets.propagation_loader import load_propagation_records
from datasets.root_loader import load_events
from datasets.schema import covariance_columns
from datasets.synthetic import make_synthetic_tracklet_root
from datasets.synthetic_field_propagation import write_synthetic_field_candidate_root
from datasets.synthetic_overlay import write_synthetic_multitrack_root
from evaluation.metrics import evaluate_event_matches


def _write_source_propagations(path, source_events) -> None:
    rows: dict[str, list[int | float | bool]] = {
        "run_id": [],
        "event_id": [],
        "source_tracklet_id": [],
        "target_tracklet_id": [],
        "source_station_id": [],
        "target_station_id": [],
        "truth_particle_id": [],
        "target_z_mm": [],
        "pred_x_mm": [],
        "pred_y_mm": [],
        "pred_tx": [],
        "pred_ty": [],
        "success": [],
        "has_covariance": [],
        "q_over_p_mode": [],
        "source_q_over_p_per_mev": [],
    }
    prediction_covariance: list[np.ndarray] = []
    for event in source_events:
        source = int(event.indices_for_station(0)[0])
        targets = event.indices_for_station(1)
        target = next(
            int(row)
            for row in targets
            if event.truth_particle_id is not None and event.truth_particle_id[row] >= 0
        )
        rows["run_id"].append(event.run_id)
        rows["event_id"].append(event.event_id)
        rows["source_tracklet_id"].append(int(event.tracklet_id[source]))
        rows["target_tracklet_id"].append(int(event.tracklet_id[target]))
        rows["source_station_id"].append(0)
        rows["target_station_id"].append(1)
        rows["truth_particle_id"].append(int(event.truth_particle_id[target]))
        rows["target_z_mm"].append(float(event.z_mm[target]))
        rows["pred_x_mm"].append(float(event.state[target, 0]))
        rows["pred_y_mm"].append(float(event.state[target, 1]))
        rows["pred_tx"].append(float(event.state[target, 2]))
        rows["pred_ty"].append(float(event.state[target, 3]))
        rows["success"].append(True)
        rows["has_covariance"].append(True)
        rows["q_over_p_mode"].append(0)
        rows["source_q_over_p_per_mev"].append(1.0e-5)
        prediction_covariance.append(np.diag([0.01, 0.01, 1.0e-8, 1.0e-8]))
    columns = {
        "run_id": np.asarray(rows["run_id"], dtype=np.int64),
        "event_id": np.asarray(rows["event_id"], dtype=np.int64),
        "source_tracklet_id": np.asarray(rows["source_tracklet_id"], dtype=np.int32),
        "target_tracklet_id": np.asarray(rows["target_tracklet_id"], dtype=np.int32),
        "source_station_id": np.asarray(rows["source_station_id"], dtype=np.int16),
        "target_station_id": np.asarray(rows["target_station_id"], dtype=np.int16),
        "truth_particle_id": np.asarray(rows["truth_particle_id"], dtype=np.int64),
        "target_z_mm": np.asarray(rows["target_z_mm"], dtype=np.float64),
        "pred_x_mm": np.asarray(rows["pred_x_mm"], dtype=np.float64),
        "pred_y_mm": np.asarray(rows["pred_y_mm"], dtype=np.float64),
        "pred_tx": np.asarray(rows["pred_tx"], dtype=np.float64),
        "pred_ty": np.asarray(rows["pred_ty"], dtype=np.float64),
        "success": np.asarray(rows["success"], dtype=np.bool_),
        "has_covariance": np.asarray(rows["has_covariance"], dtype=np.bool_),
        "q_over_p_mode": np.asarray(rows["q_over_p_mode"], dtype=np.int8),
        "source_q_over_p_per_mev": np.asarray(
            rows["source_q_over_p_per_mev"], dtype=np.float64
        ),
    }
    columns.update(
        {f"pred_{field}": values for field, values in covariance_columns(np.asarray(prediction_covariance)).items()}
    )
    with uproot.recreate(path) as root_file:
        root_file["propagations"] = columns


def test_overlay_provenance_drives_exact_field_candidate_fanout(tmp_path):
    source_path = make_synthetic_tracklet_root(tmp_path / "single.root")
    source_events = load_events(source_path, require_mc_labels=True)
    source_propagations = tmp_path / "source_propagations.root"
    _write_source_propagations(source_propagations, source_events)
    overlay_path = tmp_path / "overlay.root"
    write_synthetic_multitrack_root(
        source_events,
        overlay_path,
        output_events=2,
        tracks_per_event=2,
        station_ids=(0, 1),
        missing_tracklet_probability=0.0,
        fake_mean_per_station=0.0,
        seed=19,
    )
    synthetic_events = load_events(overlay_path, require_mc_labels=True)
    assert synthetic_events[0].origin_run_id is not None
    assert synthetic_events[0].origin_event_id is not None
    assert synthetic_events[0].origin_tracklet_id is not None

    candidate_path = tmp_path / "field_candidates.root"
    summary = write_synthetic_field_candidate_root(
        overlay_path,
        source_propagations,
        candidate_path,
    )
    assert summary.candidate_records == 8
    assert summary.target_z_mismatch == 0
    records = load_propagation_records(candidate_path)
    event = synthetic_events[0]
    candidates = build_field_candidates(event, records, 0, 1)
    assert len(candidates) == 4
    assert int(np.count_nonzero(candidate_labels(event, candidates))) == 2
    matches = greedy_field_chi2_match(candidates)
    metrics = evaluate_event_matches(event, matches, 0, 1)
    assert metrics.efficiency == 1.0
    assert metrics.inclusive_purity == 1.0
