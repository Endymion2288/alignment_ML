from __future__ import annotations

import csv

import numpy as np

from alignment.route_selected_update import (
    align_route_selected_observations,
    read_route_selected_field_edge_observations,
    read_route_selected_observations,
)


def _row(*, payload_id: str, residual_x: float, signature: str = "route-a"):
    return {
        "sample_id": "pooled_validation",
        "payload_id": payload_id,
        "run_id": "996001",
        "event_id": "12",
        "route_origin_signature": signature,
        "target_station_id": "0",
        "route_endpoint_count": "4",
        "target_synthetic_role": "0",
        "chi2": "1.0",
        "residual_x_mm": str(residual_x),
        "residual_y_mm": "-0.2",
        "residual_tx": "0.003",
        "residual_ty": "-0.004",
        "combined_cov_xx_mm2": "0.04",
        "combined_cov_xy_mm2": "0.0",
        "combined_cov_xtx_mm": "0.0",
        "combined_cov_xty_mm": "0.0",
        "combined_cov_yy_mm2": "0.04",
        "combined_cov_ytx_mm": "0.0",
        "combined_cov_yty_mm": "0.0",
        "combined_cov_txtx": "0.000001",
        "combined_cov_txty": "0.0",
        "combined_cov_tyty": "0.000001",
    }


def _write(path, rows):
    fields = sorted({field for row in rows for field in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_route_selected_observations_align_only_by_truth_free_provenance(tmp_path):
    anchor_path = tmp_path / "anchor.csv"
    probe_path = tmp_path / "probe.csv"
    _write(anchor_path, [_row(payload_id="anchor", residual_x=1.0), _row(payload_id="anchor", residual_x=2.0, signature="route-b")])
    _write(probe_path, [_row(payload_id="probe", residual_x=1.5), _row(payload_id="probe", residual_x=3.0, signature="route-c")])

    anchor = read_route_selected_observations(anchor_path, movable_station_ids=(0,))
    probe = read_route_selected_observations(probe_path, movable_station_ids=(0,))
    keys, residuals, covariances, overlap = align_route_selected_observations((anchor, probe))

    assert len(keys) == 1
    assert residuals[0].shape == (1, 4)
    assert np.allclose(residuals[0][0], [1.0, -0.2, 0.003, -0.004])
    assert np.allclose(residuals[1][0], [1.5, -0.2, 0.003, -0.004])
    assert covariances[0].shape == (1, 4, 4)
    assert overlap["anchor_common_fraction"] == 0.5


def _field_row(*, payload_id: str, residual_x: float, signature: str = "route-a"):
    row = _row(payload_id=payload_id, residual_x=residual_x, signature=signature)
    row.update(
        {
            "source_station_id": "0",
            "target_station_id": "1",
            "source_synthetic_role": "0",
            "target_synthetic_role": "0",
        }
    )
    return row


def test_field_edge_observations_keep_movable_source_and_align_by_edge_provenance(tmp_path):
    anchor_path = tmp_path / "anchor_field.csv"
    probe_path = tmp_path / "probe_field.csv"
    # The 1->2 row is intentionally absent from the movable-station block.
    excluded = _field_row(payload_id="anchor", residual_x=9.0, signature="route-excluded")
    excluded["source_station_id"] = "1"
    excluded["target_station_id"] = "2"
    _write(anchor_path, [_field_row(payload_id="anchor", residual_x=1.0), excluded])
    _write(probe_path, [_field_row(payload_id="probe", residual_x=1.5), _field_row(payload_id="probe", residual_x=3.0, signature="route-other")])

    anchor = read_route_selected_field_edge_observations(anchor_path, movable_station_ids=(0,))
    probe = read_route_selected_field_edge_observations(probe_path, movable_station_ids=(0,))
    keys, residuals, covariances, overlap = align_route_selected_observations((anchor, probe))

    assert len(keys) == 1
    assert keys[0][-2:] == (0, 1)
    assert anchor[keys[0]].source_station_id == 0
    assert anchor[keys[0]].target_station_id == 1
    assert anchor[keys[0]].observation_kind == "selected_field_aware_acts_edge"
    assert np.allclose(residuals[0][0], [1.0, -0.2, 0.003, -0.004])
    assert covariances[0].shape == (1, 4, 4)
    assert overlap["anchor_common_fraction"] == 1.0
