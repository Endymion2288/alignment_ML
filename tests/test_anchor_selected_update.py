from __future__ import annotations

import csv
import json

import numpy as np
import uproot

from alignment.route_selected_update import (
    align_route_selected_observations,
    read_anchor_selected_field_edge_observations,
)
from baselines.field_chi2_matching import build_field_candidates
from datasets.propagation_loader import load_propagation_records
from datasets.root_loader import load_events
from datasets.schema import covariance_columns
from datasets.synthetic import make_synthetic_tracklet_root
from datasets.synthetic_field_propagation import write_synthetic_field_candidate_root
from datasets.synthetic_overlay import write_synthetic_multitrack_root


def _write_source_propagations(path, source_events) -> None:
    rows: dict[str, list[int | float | bool]] = {
        key: []
        for key in (
            "run_id",
            "event_id",
            "source_tracklet_id",
            "target_tracklet_id",
            "source_station_id",
            "target_station_id",
            "truth_particle_id",
            "target_z_mm",
            "pred_x_mm",
            "pred_y_mm",
            "pred_tx",
            "pred_ty",
            "success",
            "has_covariance",
            "q_over_p_mode",
            "source_q_over_p_per_mev",
        )
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
        {
            f"pred_{field}": values
            for field, values in covariance_columns(np.asarray(prediction_covariance)).items()
        }
    )
    with uproot.recreate(path) as root_file:
        root_file["propagations"] = columns


def _fixture(tmp_path):
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
    candidate_path = tmp_path / "field_candidates.root"
    write_synthetic_field_candidate_root(overlay_path, source_propagations, candidate_path)
    return overlay_path, candidate_path


def _anchor_csv(tmp_path, overlay_path, rows):
    table = tmp_path / "anchor.csv"
    fieldnames = [
        "sample_id",
        "run_id",
        "event_id",
        "route_origin_signature",
        "route_endpoint_count",
        "source_station_id",
        "target_station_id",
        "source_origin_run_id",
        "source_origin_event_id",
        "source_origin_tracklet_id",
        "target_origin_run_id",
        "target_origin_event_id",
        "target_origin_tracklet_id",
        "source_synthetic_role",
        "target_synthetic_role",
    ]
    with table.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return table


def _true_edge_row(overlay_path, event_index=0):
    event = load_events(overlay_path, require_mc_labels=True)[event_index]
    source_row = int(event.indices_for_station(0)[0])
    target_row = next(
        int(row)
        for row in event.indices_for_station(1)
        if event.truth_particle_id is not None
        and int(event.truth_particle_id[row]) >= 0
        and int(event.truth_particle_id[row]) == int(event.truth_particle_id[source_row])
    )
    signature = json.dumps(
        [
            {
                "station_id": int(event.station_id[row]),
                "origin_run_id": int(event.origin_run_id[row]),
                "origin_event_id": int(event.origin_event_id[row]),
                "origin_tracklet_id": int(event.origin_tracklet_id[row]),
            }
            for row in (source_row, target_row)
        ]
    )
    return {
        "sample_id": "pooled_train",
        "run_id": int(event.run_id),
        "event_id": int(event.event_id),
        "route_origin_signature": signature,
        "route_endpoint_count": 2,
        "source_station_id": 0,
        "target_station_id": 1,
        "source_origin_run_id": int(event.origin_run_id[source_row]),
        "source_origin_event_id": int(event.origin_event_id[source_row]),
        "source_origin_tracklet_id": int(event.origin_tracklet_id[source_row]),
        "target_origin_run_id": int(event.origin_run_id[target_row]),
        "target_origin_event_id": int(event.origin_event_id[target_row]),
        "target_origin_tracklet_id": int(event.origin_tracklet_id[target_row]),
        "source_synthetic_role": 0,
        "target_synthetic_role": 0,
    }


def test_anchor_selected_reader_recovers_canonical_candidate_residuals(tmp_path):
    overlay_path, candidate_path = _fixture(tmp_path)
    row = _true_edge_row(overlay_path)
    table = _anchor_csv(tmp_path, overlay_path, [row])

    bank, audit = read_anchor_selected_field_edge_observations(
        table, overlay_path, candidate_path, movable_station_ids=(0, 1)
    )
    assert len(bank) == 1
    assert audit["anchor_selected_edges"] == 1
    assert audit["payload_observations"] == 1
    assert audit["anchor_edge_availability"] == 1.0

    key = ("pooled_train", row["run_id"], row["event_id"], row["route_origin_signature"], 0, 1)
    observation = bank[key]
    assert observation.observation_kind == "anchor_selected_field_aware_acts_edge"
    assert observation.route_endpoint_count == 2
    assert observation.target_synthetic_role == 0

    event = load_events(overlay_path, require_mc_labels=True)[0]
    records = load_propagation_records(candidate_path)
    origin_rows = {
        (
            int(event.origin_run_id[row]),
            int(event.origin_event_id[row]),
            int(event.origin_tracklet_id[row]),
        ): row
        for row in range(int(event.station_id.shape[0]))
    }
    source_row = origin_rows[
        (row["source_origin_run_id"], row["source_origin_event_id"], row["source_origin_tracklet_id"])
    ]
    target_row = origin_rows[
        (row["target_origin_run_id"], row["target_origin_event_id"], row["target_origin_tracklet_id"])
    ]
    canonical = {
        (int(candidate.source_index), int(candidate.target_index)): candidate
        for candidate in build_field_candidates(event, records, 0, 1)
    }
    expected = canonical[(source_row, target_row)]
    np.testing.assert_allclose(observation.residual, expected.residual)
    np.testing.assert_allclose(observation.covariance, expected.combined_covariance)
    assert observation.chi2 == expected.chi2


def test_anchor_selected_reader_drops_and_counts_missing_origins(tmp_path):
    overlay_path, candidate_path = _fixture(tmp_path)
    row = _true_edge_row(overlay_path)
    missing = dict(row)
    missing["route_origin_signature"] = row["route_origin_signature"] + "-absent"
    missing["source_origin_tracklet_id"] = 999
    table = _anchor_csv(tmp_path, overlay_path, [row, missing])

    bank, audit = read_anchor_selected_field_edge_observations(
        table, overlay_path, candidate_path, movable_station_ids=(0, 1)
    )
    assert len(bank) == 1
    assert audit["anchor_selected_edges"] == 2
    assert audit["missing_origin_in_payload"] == 1


def test_anchor_selected_reader_applies_covariance_calibration(tmp_path):
    from alignment.covariance_calibration import CovarianceCalibration

    overlay_path, candidate_path = _fixture(tmp_path)
    row = _true_edge_row(overlay_path)
    table = _anchor_csv(tmp_path, overlay_path, [row])

    plain, _ = read_anchor_selected_field_edge_observations(
        table, overlay_path, candidate_path, movable_station_ids=(0, 1)
    )
    calibration = CovarianceCalibration(
        factors={(0, 1): {"x_mm": 0.25, "y_mm": 4.0}}, provenance={"split": "train"}
    )
    scaled, _ = read_anchor_selected_field_edge_observations(
        table,
        overlay_path,
        candidate_path,
        movable_station_ids=(0, 1),
        covariance_calibration=calibration,
    )
    key = ("pooled_train", row["run_id"], row["event_id"], row["route_origin_signature"], 0, 1)
    assert len(plain) == len(scaled) == 1
    np.testing.assert_allclose(plain[key].residual, scaled[key].residual)
    assert not np.allclose(plain[key].covariance, scaled[key].covariance)
    # The tracklet covariance is untouched; only the propagated block scaled.
    difference = scaled[key].covariance - plain[key].covariance
    propagated = difference + np.zeros((4, 4))
    assert propagated[0, 0] <= 0.0  # x variance shrank
    assert difference[1, 1] >= 0.0  # y variance grew


def test_anchor_selected_banks_intersect_across_payloads(tmp_path):
    overlay_path, candidate_path = _fixture(tmp_path)
    rows = [_true_edge_row(overlay_path, event_index=0), _true_edge_row(overlay_path, event_index=1)]
    table = _anchor_csv(tmp_path, overlay_path, rows)

    anchor_like, _ = read_anchor_selected_field_edge_observations(
        table, overlay_path, candidate_path, movable_station_ids=(0, 1)
    )
    probe_like, _ = read_anchor_selected_field_edge_observations(
        table, overlay_path, candidate_path, movable_station_ids=(0, 1)
    )
    keys, residuals, covariances, overlap = align_route_selected_observations(
        [anchor_like, probe_like]
    )
    assert len(keys) == 2
    assert overlap["common_selected_observations"] == 2
    np.testing.assert_allclose(residuals[0], residuals[1])
