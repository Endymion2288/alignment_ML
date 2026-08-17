"""Validated station-level alignment payload conventions for physical studies.

The Calypso payload writer records global station transforms in
``/Tracker/Align/Stations``.  V1 limits these to translations in x and y.
This module reads the writer manifest and applies the translation-only V1
coordinate convention to already reconstructed local-tracklet states.  It
also exposes a read-only rigid-transform manifest loader for physical
cluster-to-segment refits.  The latter must never be passed to the coordinate
surrogate: a rotation changes detector surfaces, local measurements, and
tracklet covariance in ways that a coordinate shift cannot represent.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np


_COMPONENTS = "[dx_mm, dy_mm, dz_mm, rx_rad, ry_rad, rz_rad]"


@dataclass(frozen=True)
class StationAlignmentPayload:
    """Global station translations read from a payload writer manifest."""

    manifest_path: Path
    offsets_xy_mm: Mapping[int, tuple[float, float]]
    sqlite_path: Path
    pool_path: Path
    pool_catalog_path: Path

    def offset_for_station(self, station_id: int) -> tuple[float, float]:
        """Return a station translation, using identity for absent entries."""
        return self.offsets_xy_mm.get(int(station_id), (0.0, 0.0))

    def offsets_for_stations(self, station_ids: np.ndarray) -> np.ndarray:
        """Return one [dx, dy] row per supplied station identifier."""
        stations = np.asarray(station_ids, dtype=np.int64)
        offsets = np.zeros((stations.size, 2), dtype=np.float64)
        for row, station in enumerate(stations):
            offsets[row] = self.offset_for_station(int(station))
        return offsets


@dataclass(frozen=True)
class StationRigidAlignmentPayload:
    """Full station transforms declared by a physical conditions payload.

    The values are the exact six constants passed to Calypso's
    ``TrackerAlignDBTool``.  They are metadata for provenance and physical
    scan validation only; this class intentionally provides no operation that
    mutates persisted tracklet coordinates.
    """

    manifest_path: Path
    transforms: Mapping[int, tuple[float, float, float, float, float, float]]
    sqlite_path: Path
    pool_path: Path
    pool_catalog_path: Path

    def transform_for_station(self, station_id: int) -> tuple[float, float, float, float, float, float]:
        """Return a six-component identity transform for an absent station."""
        return self.transforms.get(int(station_id), (0.0, 0.0, 0.0, 0.0, 0.0, 0.0))

    def offset_for_station(self, station_id: int) -> tuple[float, float]:
        """Return the translation part when a report needs a compact view."""
        transform = self.transform_for_station(station_id)
        return transform[0], transform[1]

    def rotation_for_station(self, station_id: int) -> tuple[float, float, float]:
        """Return ``[rx, ry, rz]`` in radians."""
        transform = self.transform_for_station(station_id)
        return transform[3], transform[4], transform[5]


def _load_manifest_common(path: str | Path) -> tuple[Path, dict[str, object], dict[str, Path]]:
    """Validate the writer-level manifest contract shared by both loaders."""
    manifest_path = Path(path).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise ValueError("payload manifest must be a JSON mapping")
    if manifest.get("folder") != "/Tracker/Align":
        raise ValueError("payload manifest does not describe /Tracker/Align")
    convention = manifest.get("station_transform_convention")
    if not isinstance(convention, dict) or convention.get("frame") != "global":
        raise ValueError("payload manifest must declare global station transforms")
    if convention.get("components") != _COMPONENTS:
        raise ValueError("payload manifest has an incompatible station-transform convention")

    required_paths = {
        "sqlite": manifest.get("sqlite"),
        "pool": manifest.get("pool"),
        "pool_catalog": manifest.get("pool_catalog"),
    }
    resolved_paths: dict[str, Path] = {}
    for name, value in required_paths.items():
        if not isinstance(value, str) or not value:
            raise ValueError(f"payload manifest lacks a valid {name} path")
        artifact = Path(value).expanduser().resolve()
        if not artifact.is_file():
            raise FileNotFoundError(f"payload {name} artifact is missing: {artifact}")
        resolved_paths[name] = artifact
    return manifest_path, manifest, resolved_paths


def load_station_rigid_alignment_payload(path: str | Path) -> StationRigidAlignmentPayload:
    """Load all six station-transform components from a writer manifest.

    This loader is for true physical refit workflows.  Unlike
    :func:`load_station_alignment_payload`, it accepts rotations and therefore
    cannot be used as evidence for a coordinate-level injection.
    """
    manifest_path, manifest, resolved_paths = _load_manifest_common(path)
    constants = manifest.get("alignment_constants")
    if not isinstance(constants, dict):
        raise ValueError("payload manifest lacks alignment_constants")
    transforms: dict[int, tuple[float, float, float, float, float, float]] = {}
    for key, value in constants.items():
        if not isinstance(key, str) or not key.startswith("station:"):
            raise ValueError(f"unsupported alignment-constant key: {key!r}")
        try:
            station = int(key.removeprefix("station:"))
        except ValueError as error:
            raise ValueError(f"invalid station key: {key!r}") from error
        if station in transforms:
            raise ValueError(f"duplicate station transform for station {station}")
        values = np.asarray(value, dtype=np.float64)
        if values.shape != (6,) or not np.isfinite(values).all():
            raise ValueError(f"station {station} requires six finite transform components")
        transforms[station] = tuple(float(component) for component in values)
    return StationRigidAlignmentPayload(
        manifest_path=manifest_path,
        transforms=transforms,
        sqlite_path=resolved_paths["sqlite"],
        pool_path=resolved_paths["pool"],
        pool_catalog_path=resolved_paths["pool_catalog"],
    )


def load_station_alignment_payload(path: str | Path) -> StationAlignmentPayload:
    """Load a translation-only manifest emitted by the Calypso payload writer.

    The manifest is part of the reproducibility contract: the COOL SQLite and
    POOL files are verified to exist before their transform constants are used
    to build a controlled coordinate-level injection.
    """
    rigid = load_station_rigid_alignment_payload(path)
    offsets: dict[int, tuple[float, float]] = {}
    for station, transform in rigid.transforms.items():
        values = np.asarray(transform, dtype=np.float64)
        if not np.allclose(values[2:], 0.0, rtol=0.0, atol=1.0e-15):
            raise ValueError(
                "V1 coordinate injection supports only station-level dx/dy; "
                f"station {station} contains dz or rotations"
            )
        offsets[station] = (float(values[0]), float(values[1]))

    return StationAlignmentPayload(
        manifest_path=rigid.manifest_path,
        offsets_xy_mm=offsets,
        sqlite_path=rigid.sqlite_path,
        pool_path=rigid.pool_path,
        pool_catalog_path=rigid.pool_catalog_path,
    )


def apply_station_coordinate_offsets(
    station_id: np.ndarray,
    x_mm: np.ndarray,
    y_mm: np.ndarray,
    payload: StationAlignmentPayload,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the payload's global dx/dy transform to local-tracklet positions.

    For the translation-only V1 model, positions transform as ``x' = x + dx``
    and ``y' = y + dy``.  Slopes and covariance are unchanged by this rigid
    translation.  This is a controlled coordinate-level surrogate for a
    local segment reconstructed in the displaced station frame.
    """
    station = np.asarray(station_id)
    x = np.asarray(x_mm, dtype=np.float64)
    y = np.asarray(y_mm, dtype=np.float64)
    if station.ndim != 1 or x.shape != station.shape or y.shape != station.shape:
        raise ValueError("station_id, x_mm, and y_mm must be one-dimensional arrays of equal shape")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("tracklet coordinates must be finite")
    offsets = payload.offsets_for_stations(station)
    return x + offsets[:, 0], y + offsets[:, 1]
