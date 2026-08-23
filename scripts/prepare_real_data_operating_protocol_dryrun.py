#!/usr/bin/env python3
"""Select 2024 r0022 xAOD blocks and write Station Mode dry-run scan configs.

Workbook 03 still blocks the 2022 data0 IFT PHYS/xAOD re-export.  This command
admits only reconstructed 2024 r0022 xAOD with persisted clusters, records
run / reconstruction / geometry / conditions / trigger / DQ metadata, and
freezes the calibration / holdout / held-out-DQ split before residuals exist.
It does not write the official conditions database.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import uproot
import yaml

from alignment.real_data_operating_protocol import (
    ROLE_CALIBRATION,
    ROLE_HELD_OUT_DQ,
    ROLE_HOLDOUT,
    WORKBOOK_03_BLOCKED_REASON,
    compile_real_data_current_only,
    compile_real_data_station_linearization,
)
from scripts.config_loader import load_yaml_with_base
from scripts.run_physical_refit_capture_scan import _build_plan


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BLOCKED_2022_XAOD = Path(
    "/eos/experiment/faser/data0/rec/2022/r0022/008090/Faser-Physics-008090-00000-r0022-xAOD.root"
)


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"YAML mapping required: {path}")
    return dict(payload)


def _protocol(config: Mapping[str, Any]) -> dict[str, Any]:
    raw = config.get("operating_protocol_v1_real_data_dryrun", config)
    if not isinstance(raw, Mapping):
        raise ValueError("operating_protocol_v1_real_data_dryrun must be a mapping")
    return dict(raw)


def _assert_workbook_03_gate(xaod: Path) -> None:
    resolved = xaod.expanduser().resolve()
    if "data0/rec/2022" in str(resolved) or resolved == BLOCKED_2022_XAOD.resolve():
        raise ValueError(WORKBOOK_03_BLOCKED_REASON)
    if "rec/2024/r0022" not in str(resolved):
        raise ValueError(f"real-data dry-run admits only 2024 r0022 xAOD, got {resolved}")
    if not resolved.is_file():
        raise FileNotFoundError(resolved)


def _xaod_provenance(path: Path) -> dict[str, Any]:
    with uproot.open(path) as handle:
        tree = handle["CollectionTree"]
        keys = {str(name).split(";")[0] for name in tree.keys()}
        n_events = int(tree.num_entries)
    required = {"SCT_ClusterContainer", "SegmentFit", "Segments", "EventInfo"}
    missing = sorted(required - keys)
    if missing:
        raise ValueError(f"{path} is missing persisted keys: {', '.join(missing)}")
    if "TruthParticles" in keys or "TruthParticlesAux." in keys:
        raise ValueError(f"{path} looks like MC; refuse mixing MC labels into real data")
    return {
        "n_events": n_events,
        "persisted_keys": sorted(keys),
        "has_sct_cluster_container": True,
        "has_segment_fit": True,
        "has_segments": True,
        "mc_truth_collections_absent": "TruthParticles" not in keys,
    }


def _phys_auxiliary(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    with uproot.open(path) as handle:
        tree = handle["nt"]
        n = int(tree.num_entries)
        runs = tree["run"].array(library="np", entry_stop=min(n, 8))
        events = tree["eventID"].array(library="np", entry_stop=min(n, 8))
        fills = tree["fillNumber"].array(library="np", entry_stop=min(n, 8)) if "fillNumber" in tree else None
    sequential = list(events) == list(range(1, len(events) + 1))
    return {
        "path": str(path),
        "n_rows": n,
        "first_run_ids": [int(value) for value in runs],
        "first_event_ids": [int(value) for value in events],
        "first_fill_numbers": None if fills is None else [int(value) for value in fills],
        "event_ids_look_sequential_1_n": sequential,
        "legacy_phys_not_alignment_input": True,
        "note": "PHYS is provenance auxiliary.  Alignment uses the r0022 xAOD cluster-refit chain.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol-config",
        default=str(PROJECT_ROOT / "configs" / "operating_protocol_v1_real_data_dryrun.yaml"),
    )
    parser.add_argument(
        "--station-template",
        default=str(PROJECT_ROOT / "configs" / "physical_refit_station_mode_real_data_dryrun.yaml"),
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--occupancy-provenance",
        required=True,
        help="Frozen residual-blind occupancy windows; wave-1 is refused until all five runs have a window",
    )
    args = parser.parse_args()
    protocol_path = Path(args.protocol_config).expanduser().resolve()
    protocol = _protocol(_read_yaml(protocol_path))
    if protocol.get("official_conditions_db_write") is not False:
        raise ValueError("protocol must declare official_conditions_db_write=false")
    if protocol.get("mc_truth_used") is not False:
        raise ValueError("protocol must declare mc_truth_used=false")
    provenance_path = Path(args.occupancy_provenance).expanduser().resolve()
    with provenance_path.open(encoding="utf-8") as handle:
        occupancy = json.load(handle)
    if occupancy.get("residual_blind") is not True:
        raise ValueError("occupancy provenance must be residual-blind")
    if occupancy.get("all_windows_frozen") is not True:
        raise ValueError(
            "wave-1 physical plan is refused until all five runs have frozen occupancy windows"
        )
    occupancy_runs = occupancy.get("runs")
    if not isinstance(occupancy_runs, Mapping):
        raise ValueError("occupancy provenance missing runs")
    template = _read_yaml(Path(args.station_template).expanduser().resolve())
    output_root = Path(args.output_root).expanduser().resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty dry-run root: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    roles = {str(block["role"]) for block in protocol["blocks"]}
    if roles != {ROLE_CALIBRATION, ROLE_HOLDOUT, ROLE_HELD_OUT_DQ}:
        raise ValueError("blind split must include calibration, holdout, and held_out_dq")
    if sum(1 for block in protocol["blocks"] if block["role"] == ROLE_HELD_OUT_DQ) < 1:
        raise ValueError("at least one held_out_dq block is required")
    if sum(1 for block in protocol["blocks"] if block["role"] == ROLE_CALIBRATION) < 1:
        raise ValueError("at least one calibration block is required")

    sources: list[dict[str, Any]] = []
    for raw in protocol["blocks"]:
        run = int(raw["run"])
        role = str(raw["role"])
        occupancy_run = occupancy_runs.get(str(run))
        if not isinstance(occupancy_run, Mapping) or occupancy_run.get("window_accepted") is not True:
            raise ValueError(f"run {run} has no frozen occupancy window")
        if occupancy_run.get("role") != role:
            raise ValueError(f"occupancy provenance role drift for run {run}")
        xaod = Path(str(occupancy_run.get("input_xaod") or raw["input_xaod"])).expanduser().resolve()
        _assert_workbook_03_gate(xaod)
        xaod_meta = _xaod_provenance(xaod)
        phys_path = Path(str(raw["phys_auxiliary"])).expanduser() if raw.get("phys_auxiliary") else None
        phys_meta = _phys_auxiliary(phys_path)
        if phys_meta is not None and phys_meta["event_ids_look_sequential_1_n"]:
            raise ValueError(f"auxiliary PHYS for run {run} looks like the workbook-03 sequential rewrite")
        segment = str(occupancy_run["segment"])
        skip_events = int(occupancy_run["skip_events"])
        nevents = int(occupancy_run["nevents"])
        source_id = f"data24_r{run:05d}_{segment}_skip{skip_events:05d}"
        compiled, contract = (
            compile_real_data_station_linearization(template)
            if role == ROLE_CALIBRATION
            else compile_real_data_current_only(template)
        )
        scan = dict(compiled["physical_refit_capture_scan"])
        scan["input_xaod"] = str(xaod)
        scan["nevents"] = nevents
        scan["skip_events"] = skip_events
        scan["is_mc"] = False
        scan["include_truth"] = False
        scan["require_mc_labels"] = False
        scan["cdx_fixed_by"] = "external_geometry"
        plan = _build_plan(scan)
        source_root = output_root / "sources" / source_id
        source_root.mkdir(parents=True, exist_ok=False)
        config_path = source_root / "physical_scan_config.yaml"
        config_path.write_text(
            yaml.safe_dump({"physical_refit_capture_scan": scan}, sort_keys=False),
            encoding="utf-8",
        )
        (source_root / "linearization_contract.json").write_text(
            json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (source_root / "scan_plan.json").write_text(
            json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        sources.append(
            {
                "source_id": source_id,
                "run": run,
                "role": role,
                "split": str(raw["split"]),
                "input_xaod": str(xaod),
                "xaod_segment": segment,
                "skip_events": skip_events,
                "nevents": nevents,
                "occupancy_summary": occupancy_run.get("occupancy_summary"),
                "occupancy_window_status": occupancy_run.get("status"),
                "geometry_tag": protocol["geometry_tag"],
                "conditions_tag": protocol["conditions_tag"],
                "reconstruction_tag": protocol["reconstruction_tag"],
                "geom_selector": protocol["geom_selector"],
                "lumiblock": {
                    "ntuple_column": "absent_from_current_ntuple",
                    "time_block": "frozen_occupancy_window_segment_skip_events_nevents",
                },
                "trigger_and_dq": protocol["data_filters"],
                "cdx_fixed_by": "external_geometry",
                "geometry_estimation_allowed": role == ROLE_CALIBRATION,
                "xaod": xaod_meta,
                "phys_auxiliary": phys_meta,
                "physical_scan_config": str(config_path),
                "physical_scan_root": str(source_root / "physical_scan"),
                "n_planned_points": len(plan["points"]),
            }
        )

    catalog = {
        "schema_version": "faser-operating-protocol-v1-real-data-provenance",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "workbook_03_gate": protocol["workbook_03_gate"],
        "workbook_03_blocked_reason": WORKBOOK_03_BLOCKED_REASON,
        "official_conditions_db_write": False,
        "mc_truth_used": False,
        "blind_split_frozen_before_residuals": True,
        "occupancy_provenance": str(provenance_path),
        "occupancy_windows_frozen_before_wave1": True,
        "protocol_config": str(protocol_path),
        "blocks": sources,
    }
    (output_root / "real_data_provenance_catalog.json").write_text(
        json.dumps(catalog, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_root / "iteration_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "faser-operating-protocol-v1-real-data-station-wave1",
                "created_utc": catalog["created_utc"],
                "physical_geometry_repropagation": True,
                "q_over_p_mode": 0,
                "is_mc": False,
                "official_conditions_db_write": False,
                "sources": sources,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output_root": str(output_root), "n_blocks": len(sources)}, indent=2))


if __name__ == "__main__":
    main()
