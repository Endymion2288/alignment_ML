#!/usr/bin/env python3
"""Collect hierarchical V1 remaining payloads from train route-selected updates.

The remaining geometry is never ``proposed_next_parameter_values`` from a
reference-linearized Newton step: that chart would zero the unfloated level.
Validation updates may be listed for the record but do not choose the shared
remaining payload.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from alignment.hierarchical_v1 import (
    C_DX,
    C_DX_ENVELOPE_MM,
    HIERARCHICAL_V1_PARAMETERS,
    LAYER_LEVEL,
    STATION_LEVEL,
    complete_hierarchical_values,
    inside_cdx_envelope,
    remaining_after_level_step,
)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object at {path}")
    return payload


def _floated_level(payload: Mapping[str, Any]) -> str:
    names = [str(item) for item in payload.get("only_parameters", ())]
    if names == [C_DX]:
        return LAYER_LEVEL
    if C_DX in names:
        raise ValueError(f"{payload} mixed station and C_dx in one Newton step")
    if not names:
        raise ValueError("route-selected update lacks only_parameters")
    return STATION_LEVEL


def _remaining_from_update(path: Path, *, name: str, split: str) -> dict[str, Any]:
    payload = _read_json(path)
    if payload.get("test_opened") is True or payload.get("test_data_accessed") is True:
        raise ValueError(f"{path} accessed test data")
    if int(payload.get("q_over_p_mode", -1)) != 0:
        raise ValueError(f"{path} is not mode-0")
    if payload.get("joint_station_cdx_newton") is True:
        raise ValueError(f"{path} mixed station and C_dx in one Newton step")
    floated_level = _floated_level(payload)
    audit = payload.get("hierarchy_internal_audit")
    remaining = None
    injected = None
    recovered = None
    if isinstance(audit, Mapping):
        remaining = audit.get("remaining_hierarchical_v1")
        injected = audit.get("injected_hierarchical_v1")
        recovered = audit.get("recovered")
    if remaining is None:
        if not isinstance(injected, Mapping) or not isinstance(recovered, Mapping):
            raise ValueError(f"{path} lacks remaining_hierarchical_v1 and cannot reconstruct it")
        remaining = remaining_after_level_step(
            injected=injected,
            recovered=recovered,
            floated_level=floated_level,
        )
    remaining = complete_hierarchical_values(remaining)
    if not inside_cdx_envelope(remaining[C_DX]):
        raise ValueError(f"{path} remaining leaves |C_dx|<={C_DX_ENVELOPE_MM} mm")
    proposed = payload.get("proposed_next_parameter_values")
    return {
        "name": name,
        "floated_level": floated_level,
        "parent_target": str(payload.get("target_point")),
        "split": split,
        "alignment_parameter_values": remaining,
        "injected": None if not isinstance(injected, Mapping) else complete_hierarchical_values(injected),
        "recovered": None if not isinstance(recovered, Mapping) else dict(recovered),
        "source_update": str(path),
        "allow_identically_zero": all(abs(remaining[name_]) <= 1.0e-15 for name_ in HIERARCHICAL_V1_PARAMETERS),
        "proposed_next_is_not_remaining": True,
        "joint_station_cdx_newton": False,
        "proposed_next_parameter_values_ignored": proposed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--order", required=True, help="station_then_C_dx or C_dx_then_station")
    parser.add_argument(
        "--train-update",
        action="append",
        nargs=2,
        metavar=("NAME", "PATH"),
        required=True,
        help="train route-selected update that chooses the shared remaining payload",
    )
    parser.add_argument(
        "--validation-update",
        action="append",
        nargs=2,
        metavar=("NAME", "PATH"),
        default=None,
        help="validation route-selected update recorded only, never used to choose remaining",
    )
    args = parser.parse_args()
    if args.order not in {"station_then_C_dx", "C_dx_then_station"}:
        parser.error("--order must be station_then_C_dx or C_dx_then_station")
    points = [_remaining_from_update(Path(path), name=name, split="train") for name, path in args.train_update]
    validation = []
    if args.validation_update:
        validation = [
            _remaining_from_update(Path(path), name=name, split="validation")
            for name, path in args.validation_update
        ]
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite remaining JSON: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "order": str(args.order),
        "train_chooses_remaining": True,
        "validation_used_in_registration": False,
        "test_data_accessed": False,
        "joint_station_cdx_newton": False,
        "proposed_next_is_not_remaining": True,
        "points": points,
        "validation_record_only": validation,
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"remaining_json": str(output), "points": [item["name"] for item in points]}, indent=2))


if __name__ == "__main__":
    main()
