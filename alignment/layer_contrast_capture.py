"""Pre-registered dual capture criteria for the IFT ``C_dx+C_rx`` subspace.

The registered sigmas come from the already-closed 3-source train
route-selected 1-D solves.  Validation never participates in the freeze.
Engineering numbers are informational; framework capture is statistical
AND this-fit coverage, matching the station 5-DoF dual gate.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from alignment.capture_criteria import DEFAULT_COVERAGE_K, DEFAULT_STATISTICAL_K
from alignment.contrast_sampling import CONTRAST_PARAMETERS


SCHEMA_VERSION = "faser-ift-layer-contrast-2d-capture-criteria-v1"
FREE_PARAMETERS: tuple[str, ...] = CONTRAST_PARAMETERS
DEFAULT_ENGINEERING: dict[str, float] = {
    "C_dx": 0.03,
    "C_rx": 0.15,
}


def validate_layer_contrast_capture_criteria(payload: Mapping[str, Any], path: Path) -> dict[str, Any]:
    """Return a validated contrast-2D capture contract."""
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"not a C_dx+C_rx capture-criteria contract: {path}")
    if payload.get("validation_used_in_registration") is not False:
        raise ValueError("capture criteria must declare validation_used_in_registration=false")
    if payload.get("test_data_accessed") is not False:
        raise ValueError("capture criteria must not access test data")
    per = payload.get("per_parameter")
    if not isinstance(per, Mapping):
        raise ValueError(f"{path} lacks per_parameter rows")
    missing = [name for name in FREE_PARAMETERS if name not in per]
    if missing:
        raise ValueError("contrast capture criteria missing " + ", ".join(missing))
    for name in FREE_PARAMETERS:
        row = per[name]
        if not isinstance(row, Mapping) or row.get("role") != "free":
            raise ValueError(f"contrast capture row '{name}' must be a free parameter")
        sigma = row.get("registered_sigma")
        if sigma is None or float(sigma) <= 0.0:
            raise ValueError(f"contrast capture row '{name}' has no usable registered_sigma")
    return dict(payload)


def load_layer_contrast_capture_criteria(path: Path) -> dict[str, Any]:
    import json

    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected a JSON object at {path}")
    return validate_layer_contrast_capture_criteria(payload, path)


def default_statistical_k() -> float:
    return DEFAULT_STATISTICAL_K


def default_coverage_k() -> float:
    return DEFAULT_COVERAGE_K
