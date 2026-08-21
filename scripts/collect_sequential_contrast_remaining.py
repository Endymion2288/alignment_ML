#!/usr/bin/env python3
"""Collect sequential 1-D remaining payloads from train route-selected updates.

The remaining geometry is never ``proposed_next_parameter_values`` from a 1-D
Newton step linearized at the zero reference: that chart would zero the
unfloated contrast.  Validation updates may be listed for the record but do
not choose the shared remaining payload.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from alignment.contrast_sampling import CONTRAST_PARAMETERS, inside_contrast_envelope
from alignment.sequential_contrast import remaining_after_block_step


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object at {path}")
    return payload


def _remaining_from_update(path: Path, *, name: str, split: str) -> dict[str, Any]:
    payload = _read_json(path)
    if payload.get("test_opened") is True or payload.get("test_data_accessed") is True:
        raise ValueError(f"{path} accessed test data")
    if int(payload.get("q_over_p_mode", -1)) != 0:
        raise ValueError(f"{path} is not mode-0")
    names = [str(item) for item in payload.get("only_parameters", ())]
    if len(names) != 1 or names[0] not in CONTRAST_PARAMETERS:
        raise ValueError(f"{path} is not a sequential 1-D C_dx/C_rx update")
    floated = names[0]
    audit = payload.get("hierarchy_internal_audit")
    remaining = None
    injected = None
    recovered_floated = None
    if isinstance(audit, Mapping):
        remaining = audit.get("remaining_contrast")
        injected = audit.get("injected_contrast")
        recovered = audit.get("recovered")
        if isinstance(recovered, Mapping) and floated in recovered:
            recovered_floated = float(recovered[floated])
    if remaining is None:
        if not isinstance(injected, Mapping) or recovered_floated is None:
            raise ValueError(f"{path} lacks remaining_contrast and cannot reconstruct it")
        remaining = remaining_after_block_step(
            injected=injected,
            recovered_floated=recovered_floated,
            floated=floated,
        )
    remaining = {name_: float(remaining[name_]) for name_ in CONTRAST_PARAMETERS}
    if not inside_contrast_envelope(remaining):
        raise ValueError(f"{path} remaining leaves the verified contrast envelope")
    return {
        "name": name,
        "floated": floated,
        "parent_target": str(payload.get("target_point")),
        "split": split,
        "alignment_parameter_values": remaining,
        "injected_contrast": None if not isinstance(injected, Mapping) else {
            name_: float(injected[name_]) for name_ in CONTRAST_PARAMETERS if name_ in injected
        },
        "recovered_floated": recovered_floated,
        "source_update": str(path),
        "allow_identically_zero": all(
            abs(remaining[name_]) <= 1.0e-15 for name_ in CONTRAST_PARAMETERS
        ),
        "proposed_next_is_not_remaining": True,
        "joint_2d_newton": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--train-update",
        action="append",
        required=True,
        metavar="NAME:PATH",
        help="Remaining point name and train route-selected JSON that writes the shared payload.",
    )
    parser.add_argument(
        "--validation-update",
        action="append",
        default=None,
        metavar="NAME:PATH",
        help="Optional validation 1-D updates recorded only; they never choose the remaining payload.",
    )
    args = parser.parse_args()
    points = []
    for item in args.train_update:
        if ":" not in item:
            raise ValueError(f"expected NAME:PATH, got {item!r}")
        name, raw_path = item.split(":", 1)
        points.append(_remaining_from_update(Path(raw_path).expanduser().resolve(), name=name, split="train"))
    validation_record = []
    for item in args.validation_update or []:
        if ":" not in item:
            raise ValueError(f"expected NAME:PATH, got {item!r}")
        name, raw_path = item.split(":", 1)
        validation_record.append(
            _remaining_from_update(Path(raw_path).expanduser().resolve(), name=name, split="validation")
        )
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "faser-ift-layer-sequential-contrast-remaining-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "test_data_accessed": False,
        "validation_used_to_choose_remaining": False,
        "joint_2d_newton": False,
        "station_six_vector_forced_zero": True,
        "payload_expansion": "L0=(+C_dx,+C_rx), L1=0, L2=(-C_dx,-C_rx); station six-vector identically 0",
        "remaining_chosen_from": "train_route_selected_1d",
        "points": points,
        "validation_record": validation_record,
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "points": [item["name"] for item in points]}, indent=2))


if __name__ == "__main__":
    main()
