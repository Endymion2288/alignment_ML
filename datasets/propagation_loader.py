"""Loader and contract for flat field-aware tracklet propagation records."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import uproot

from .schema import COVARIANCE_FIELDS, DatasetSchemaError, covariance_from_columns


PROPAGATION_TREE_NAME = "propagations"
PROPAGATION_SCHEMA_VERSION = "faser-tracklet-propagations-v1"

OPTIONAL_PROPAGATION_FIELDS = (
    "q_over_p_mode",
    "source_q_over_p_per_mev",
)

REQUIRED_PROPAGATION_FIELDS = (
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
    *(f"pred_{field}" for field in COVARIANCE_FIELDS),
    "success",
    "has_covariance",
)


@dataclass(frozen=True)
class PropagationRecords:
    """Flat truth-matched field propagation records from NtupleDumper."""

    run_id: np.ndarray
    event_id: np.ndarray
    source_tracklet_id: np.ndarray
    target_tracklet_id: np.ndarray
    source_station_id: np.ndarray
    target_station_id: np.ndarray
    truth_particle_id: np.ndarray
    target_z_mm: np.ndarray
    prediction: np.ndarray
    covariance: np.ndarray
    success: np.ndarray
    has_covariance: np.ndarray
    q_over_p_mode: np.ndarray | None = None
    source_q_over_p_per_mev: np.ndarray | None = None

    @property
    def size(self) -> int:
        return int(self.run_id.size)


def _field_names(tree: object) -> set[str]:
    return {str(name).split(";")[0] for name in getattr(tree, "keys")()}


def _columns_from_uproot(raw_columns: object, fields: tuple[str, ...]) -> dict[str, np.ndarray]:
    if isinstance(raw_columns, dict):
        return {field: np.asarray(raw_columns[field]) for field in fields}
    if isinstance(raw_columns, np.ndarray) and raw_columns.dtype.names is not None:
        return {field: np.asarray(raw_columns[field]) for field in fields}
    raise DatasetSchemaError("uproot returned unsupported propagation column data")


def load_propagation_records(
    root_path: str | Path,
    tree_name: str = PROPAGATION_TREE_NAME,
) -> PropagationRecords:
    """Load the canonical flat propagation tree without dropping failed rows."""
    path = Path(root_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    with uproot.open(path) as root_file:
        if tree_name not in root_file:
            raise DatasetSchemaError(f"tree '{tree_name}' is absent from {path}")
        tree = root_file[tree_name]
        available = _field_names(tree)
        missing = [field for field in REQUIRED_PROPAGATION_FIELDS if field not in available]
        if missing:
            raise DatasetSchemaError(
                "missing required propagation fields: " + ", ".join(missing)
            )
        optional = tuple(field for field in OPTIONAL_PROPAGATION_FIELDS if field in available)
        fields = (*REQUIRED_PROPAGATION_FIELDS, *optional)
        raw_columns = tree.arrays(list(fields), library="np")

    columns = _columns_from_uproot(raw_columns, fields)
    lengths = {field: int(values.size) for field, values in columns.items()}
    if len(set(lengths.values())) != 1:
        raise DatasetSchemaError(f"propagation field length mismatch: {lengths}")
    prediction = np.column_stack(
        [
            columns["pred_x_mm"],
            columns["pred_y_mm"],
            columns["pred_tx"],
            columns["pred_ty"],
        ]
    ).astype(np.float64, copy=False)
    covariance = covariance_from_columns(
        {field: columns[f"pred_{field}"] for field in COVARIANCE_FIELDS}
    )
    return PropagationRecords(
        run_id=np.asarray(columns["run_id"], dtype=np.int64),
        event_id=np.asarray(columns["event_id"], dtype=np.int64),
        source_tracklet_id=np.asarray(columns["source_tracklet_id"], dtype=np.int32),
        target_tracklet_id=np.asarray(columns["target_tracklet_id"], dtype=np.int32),
        source_station_id=np.asarray(columns["source_station_id"], dtype=np.int16),
        target_station_id=np.asarray(columns["target_station_id"], dtype=np.int16),
        truth_particle_id=np.asarray(columns["truth_particle_id"], dtype=np.int64),
        target_z_mm=np.asarray(columns["target_z_mm"], dtype=np.float64),
        prediction=prediction,
        covariance=np.asarray(covariance, dtype=np.float64),
        success=np.asarray(columns["success"], dtype=bool),
        has_covariance=np.asarray(columns["has_covariance"], dtype=bool),
        q_over_p_mode=(
            np.asarray(columns["q_over_p_mode"], dtype=np.int8)
            if "q_over_p_mode" in columns
            else np.zeros(prediction.shape[0], dtype=np.int8)
        ),
        source_q_over_p_per_mev=(
            np.asarray(columns["source_q_over_p_per_mev"], dtype=np.float64)
            if "source_q_over_p_per_mev" in columns
            else np.full(prediction.shape[0], np.nan, dtype=np.float64)
        ),
    )
