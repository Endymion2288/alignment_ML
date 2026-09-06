"""Typed measurement records for a field-aware alignment likelihood.

Production FaserActs C_prop is not a default R.  T11 must pass before a
physical adapter may attach transported covariances.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from alignment.numerical_contract import require_spd


class MeasurementSchemaError(ValueError):
    """Raised when an alignment measurement is missing required provenance."""


REQUIRED_IDS = (
    "original_source_uid",
    "run_id",
    "event_id",
    "physical_track_id",
    "cluster_id",
    "sensor_id",
)


@dataclass(frozen=True)
class AlignmentMeasurement:
    original_source_uid: str
    run_id: int
    event_id: int
    physical_track_id: str
    cluster_id: str
    sensor_id: str
    local_u_mm: float
    residual_dimension: int
    covariance: np.ndarray
    surface_frame: str
    units: str
    geometry_hash: str
    field_hash: str
    material_hash: str
    iov: str
    uses_production_c_prop: bool = False

    def as_json(self) -> dict[str, Any]:
        return {
            "original_source_uid": self.original_source_uid,
            "run_id": int(self.run_id),
            "event_id": int(self.event_id),
            "physical_track_id": self.physical_track_id,
            "cluster_id": self.cluster_id,
            "sensor_id": self.sensor_id,
            "local_u_mm": float(self.local_u_mm),
            "residual_dimension": int(self.residual_dimension),
            "surface_frame": self.surface_frame,
            "units": self.units,
            "geometry_hash": self.geometry_hash,
            "field_hash": self.field_hash,
            "material_hash": self.material_hash,
            "iov": self.iov,
            "uses_production_c_prop": bool(self.uses_production_c_prop),
        }


def require_measurement_ids(record: AlignmentMeasurement) -> None:
    for name in REQUIRED_IDS:
        value = getattr(record, name)
        if value is None or value == "":
            raise MeasurementSchemaError(f"measurement missing {name}")
    require_spd(record.covariance, name="measurement R")
    if record.uses_production_c_prop:
        raise MeasurementSchemaError(
            "production C_prop is not an allowed measurement covariance while T11 is failed"
        )
