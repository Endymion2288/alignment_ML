from __future__ import annotations

import json

import numpy as np
import pytest

from alignment.closure import AlignmentMeasurements, select_measurements
from alignment.payload import (
    apply_station_coordinate_offsets,
    load_station_alignment_payload,
    load_station_rigid_alignment_payload,
)


def _manifest(tmp_path, constants: dict[str, list[float]]):
    sqlite = tmp_path / "tracker_alignment.sqlite"
    pool = tmp_path / "tracker_alignment.pool.root"
    catalog = tmp_path / "PoolFileCatalog.xml"
    for path in (sqlite, pool, catalog):
        path.write_bytes(b"")

    manifest = tmp_path / "alignment_payload.json"
    manifest.write_text(
        json.dumps(
            {
                "folder": "/Tracker/Align",
                "station_transform_convention": {
                    "frame": "global",
                    "components": "[dx_mm, dy_mm, dz_mm, rx_rad, ry_rad, rz_rad]",
                },
                "alignment_constants": constants,
                "sqlite": str(sqlite),
                "pool": str(pool),
                "pool_catalog": str(catalog),
            }
        ),
        encoding="utf-8",
    )
    return manifest


def test_payload_manifest_applies_global_station_xy_offsets(tmp_path):
    manifest = _manifest(
        tmp_path,
        {
            "station:0": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "station:3": [1.25, -0.5, 0.0, 0.0, 0.0, 0.0],
        },
    )

    payload = load_station_alignment_payload(manifest)
    shifted_x, shifted_y = apply_station_coordinate_offsets(
        np.asarray([0, 1, 3], dtype=np.int16),
        np.asarray([10.0, 20.0, 30.0]),
        np.asarray([-1.0, -2.0, -3.0]),
        payload,
    )

    assert payload.offset_for_station(1) == (0.0, 0.0)
    np.testing.assert_allclose(shifted_x, [10.0, 20.0, 31.25])
    np.testing.assert_allclose(shifted_y, [-1.0, -2.0, -3.5])


def test_payload_manifest_rejects_non_v1_transform_components(tmp_path):
    manifest = _manifest(
        tmp_path,
        {"station:2": [0.1, 0.2, 0.0, 0.0, 0.0, 1.0e-3]},
    )

    with pytest.raises(ValueError, match="only station-level dx/dy"):
        load_station_alignment_payload(manifest)


def test_rigid_payload_loader_preserves_ry_for_physical_refit_only(tmp_path):
    manifest = _manifest(
        tmp_path,
        {
            "station:0": [1.0, -2.0, 0.0, 0.0, 0.060, 0.0],
            "station:1": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        },
    )

    rigid = load_station_rigid_alignment_payload(manifest)

    assert rigid.transform_for_station(0) == (1.0, -2.0, 0.0, 0.0, 0.06, 0.0)
    assert rigid.rotation_for_station(0) == (0.0, 0.06, 0.0)
    assert rigid.transform_for_station(3) == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="only station-level dx/dy"):
        load_station_alignment_payload(manifest)


def test_measurement_selection_keeps_pair_arrays_aligned():
    measurements = AlignmentMeasurements(
        source_station_id=np.asarray([0, 0, 1], dtype=np.int16),
        target_station_id=np.asarray([1, 2, 2], dtype=np.int16),
        nominal_residual_xy_mm=np.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        covariance_xy_mm2=np.tile(np.eye(2), (3, 1, 1)),
    )

    selected = select_measurements(measurements, np.asarray([True, False, True]))

    assert selected.size == 2
    np.testing.assert_array_equal(selected.source_station_id, [0, 1])
    np.testing.assert_array_equal(selected.target_station_id, [1, 2])
    np.testing.assert_allclose(selected.nominal_residual_xy_mm, [[1.0, 2.0], [5.0, 6.0]])
