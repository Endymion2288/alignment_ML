"""Event bootstrap that keeps with-replacement multiplicity.

Historical workbook 68/69/73 reports stay frozen.  New draws must not convert
a boolean mask into a deployment pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Hashable, Mapping, Sequence

import numpy as np

BANK_ROW_KEYS = (
    "anchor_residual",
    "reference_residual",
    "covariance",
    "run_id",
    "event_id",
    "truth_particle_id",
    "source_station_id",
    "target_station_id",
    "source_tracklet_id",
    "target_tracklet_id",
    "source_tx",
    "source_ty",
    "source_slope",
    "source_uid",
    "original_source_uid",
)


EventKey = tuple[str, int, int]


def event_identity_keys(bank: Mapping[str, Any]) -> list[EventKey]:
    """Immutable source UID + run/event.  Overlay event numbers do not merge sources."""
    n_pairs = int(np.asarray(bank["anchor_residual"]).shape[0])
    runs = np.asarray(bank["run_id"]).reshape(-1)
    events = np.asarray(bank["event_id"]).reshape(-1)
    if runs.shape[0] != n_pairs or events.shape[0] != n_pairs:
        raise ValueError("run_id/event_id length must match the pair bank")
    source = None
    for name in ("original_source_uid", "source_uid", "source_id"):
        if name in bank and bank[name] is not None:
            source = bank[name]
            break
    if source is None:
        raise ValueError("bank must carry an immutable source UID")
    raw = np.asarray(source, dtype=object)
    if raw.ndim == 0 or raw.size == 1:
        sources = np.full(n_pairs, str(raw.reshape(-1)[0]), dtype=object)
    else:
        if raw.shape[0] != n_pairs:
            raise ValueError("source uid length must match the pair bank")
        sources = np.asarray([str(item) for item in raw.tolist()], dtype=object)
    return [
        (str(sources[index]), int(runs[index]), int(events[index]))
        for index in range(n_pairs)
    ]


def group_rows_by_event(keys: Sequence[EventKey]) -> dict[EventKey, list[int]]:
    groups: dict[EventKey, list[int]] = {}
    for index, key in enumerate(keys):
        groups.setdefault(key, []).append(int(index))
    return groups


def rows_for_event_draw(
    groups: Mapping[EventKey, Sequence[int]],
    drawn_keys: Sequence[EventKey],
) -> np.ndarray:
    """Concatenate row indices, repeating a group's rows for each draw of that event."""
    rows: list[int] = []
    for key in drawn_keys:
        if key not in groups:
            raise KeyError(f"unknown event key: {key}")
        rows.extend(int(index) for index in groups[key])
    return np.asarray(rows, dtype=np.int64)


def sample_event_keys(
    keys: Sequence[EventKey],
    *,
    seed: int,
    n_draws: int | None = None,
) -> list[EventKey]:
    unique = list(dict.fromkeys(keys))
    if len(unique) < 2:
        raise ValueError("event bootstrap needs at least two events")
    rng = np.random.default_rng(int(seed))
    size = int(n_draws if n_draws is not None else len(unique))
    choice = rng.integers(0, len(unique), size=size)
    return [unique[int(index)] for index in choice]


def index_physical_bank(bank: Mapping[str, Any], indices: object) -> dict[str, Any]:
    """Repeat-aware bank subset.  Integer indices may contain duplicates."""
    idx = np.asarray(indices, dtype=np.int64)
    if idx.ndim != 1:
        raise ValueError("pair indices must be a 1-D integer array")
    n_pairs = int(np.asarray(bank["anchor_residual"]).shape[0])
    if idx.size and (int(idx.min()) < 0 or int(idx.max()) >= n_pairs):
        raise ValueError("pair indices are out of range")
    result = dict(bank)
    for key in BANK_ROW_KEYS:
        if key not in bank:
            continue
        array = np.asarray(bank[key])
        if array.shape[:1] == (n_pairs,):
            result[key] = array[idx]
    result["anchor_residual"] = np.asarray(bank["anchor_residual"])[idx]
    result["positive_residual"] = np.asarray(bank["positive_residual"])[:, idx]
    result["negative_residual"] = np.asarray(bank["negative_residual"])[:, idx]
    if "reference_residual" in bank:
        result["reference_residual"] = np.asarray(bank["reference_residual"])[idx]
    if "covariance" in bank:
        result["covariance"] = np.asarray(bank["covariance"])[idx]
    return result


@dataclass(frozen=True)
class BootstrapDraw:
    drawn_keys: tuple[EventKey, ...]
    row_indices: np.ndarray
    n_draws: int
    n_unique: int
    effective_multiplicity: float
    n_rows: int

    def as_json(self) -> dict[str, Any]:
        return {
            "n_draws": int(self.n_draws),
            "n_unique": int(self.n_unique),
            "effective_multiplicity": float(self.effective_multiplicity),
            "n_rows": int(self.n_rows),
            "drawn_keys": [list(key) for key in self.drawn_keys],
        }


def describe_event_draw(
    groups: Mapping[EventKey, Sequence[int]],
    drawn_keys: Sequence[EventKey],
) -> BootstrapDraw:
    rows = rows_for_event_draw(groups, drawn_keys)
    n_draws = len(drawn_keys)
    n_unique = len(set(drawn_keys))
    multiplicity = (float(n_draws) / float(n_unique)) if n_unique else 0.0
    return BootstrapDraw(
        drawn_keys=tuple(drawn_keys),
        row_indices=rows,
        n_draws=n_draws,
        n_unique=n_unique,
        effective_multiplicity=multiplicity,
        n_rows=int(rows.size),
    )
