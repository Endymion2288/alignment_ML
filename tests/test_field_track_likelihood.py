"""T12 field-likelihood schema and T11 gate.  No production C_prop."""

from __future__ import annotations

import numpy as np
import pytest

from datasets.alignment_measurements import (
    AlignmentMeasurement,
    MeasurementSchemaError,
    require_measurement_ids,
)
from models.field_track_likelihood import (
    FieldLikelihoodError,
    TrackBlock,
    accumulate_alignment_normal,
    refuse_physical_run,
)


def _measurement(**overrides):
    payload = dict(
        original_source_uid="mc24_fixture",
        run_id=1,
        event_id=2,
        physical_track_id="t0",
        cluster_id="c0",
        sensor_id="s0",
        local_u_mm=0.1,
        residual_dimension=1,
        covariance=np.array([[0.01]]),
        surface_frame="sensor_local_u",
        units="mm",
        geometry_hash="a" * 64,
        field_hash="b" * 64,
        material_hash="c" * 64,
        iov="fixture",
        uses_production_c_prop=False,
    )
    payload.update(overrides)
    return AlignmentMeasurement(**payload)


def test_schema_requires_ids_and_spd_r():
    require_measurement_ids(_measurement())
    with pytest.raises(MeasurementSchemaError, match="C_prop"):
        require_measurement_ids(_measurement(uses_production_c_prop=True))
    with pytest.raises(Exception):
        require_measurement_ids(_measurement(covariance=np.array([[-1.0]])))


def test_physical_run_blocked_when_t11_failed():
    with pytest.raises(FieldLikelihoodError, match="T11"):
        refuse_physical_run(t11_contract_established=False)
    refuse_physical_run(t11_contract_established=True)


def test_accumulate_two_tracks_without_c_prop():
    rng = np.random.default_rng(2)
    g = rng.normal(size=(6, 2))
    h = rng.normal(size=(6, 5))
    residual = rng.normal(size=6)
    measurements = tuple(_measurement(cluster_id=f"c{i}") for i in range(6))
    block = TrackBlock(
        measurements=measurements,
        residual=residual,
        g=g,
        h=h,
        covariance=np.eye(6),
    )
    stacked = accumulate_alignment_normal([block, block])
    assert stacked["n_tracks"] == 2
    assert stacked["uses_production_c_prop"] is False
    assert stacked["normal"].shape == (2, 2)
