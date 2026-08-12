#!/usr/bin/env python3
"""Create source-disjoint, physically refitted payload banks for MLP training.

The driver never alters canonical tracklet coordinates.  It writes a real
/Tracker/Align payload for every bank point and delegates the actual work to
``run_physical_refit_capture_scan.py``, which reruns SCT clusters through the
segment refit and Acts export chain.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from datasets.physical_curriculum import PHYSICAL_CORPUS_SCHEMA, SPLITS
from datasets.root_loader import load_events
from scripts.config_loader import load_yaml_with_base
from scripts.run_physical_refit_capture_scan import _build_plan


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCAN_DRIVER = PROJECT_ROOT / "scripts" / "run_physical_refit_capture_scan.py"


def _load_config(path: Path) -> dict[str, Any]:
    supplied = load_yaml_with_base(path)
    config = supplied.get("physical_curriculum_mlp", supplied)
    if not isinstance(config, Mapping):
        raise ValueError("physical_curriculum_mlp must be a YAML mapping")
    return dict(config)


def _normalised_direction(rng: np.random.Generator) -> list[float]:
    values = rng.normal(size=2)
    norm = float(np.linalg.norm(values))
    if not np.isfinite(norm) or norm <= 0.0:
        raise RuntimeError("failed to draw a non-zero random alignment direction")
    return [float(values[0] / norm), float(values[1] / norm)]


def _payload_directions(config: Mapping[str, Any]) -> dict[str, list[dict[str, object]]]:
    bank = config["payload_bank"]
    if not isinstance(bank, Mapping):
        raise ValueError("payload_bank must be a mapping")
    counts = bank["direction_trials_per_split"]
    if not isinstance(counts, Mapping):
        raise ValueError("direction_trials_per_split must be a mapping")
    stations = tuple(sorted(int(value) for value in config["refit"]["station_ids"]))
    reference = int(config["refit"]["reference_station"])
    movable = tuple(station for station in stations if station != reference)
    if not movable:
        raise ValueError("at least one non-reference station is required")
    rng = np.random.default_rng(int(bank["seed"]))
    result: dict[str, list[dict[str, object]]] = {}
    for split in SPLITS:
        count = int(counts.get(split, 0))
        if count < 1:
            raise ValueError(f"payload_bank requires at least one direction for {split}")
        trials: list[dict[str, object]] = []
        for index in range(count):
            trials.append(
                {
                    "name": f"{split}_{index:02d}",
                    "unit_offsets_xy": {
                        int(station): _normalised_direction(rng) for station in movable
                    },
                }
            )
        result[split] = trials
    return result


def _validate_sources(config: Mapping[str, Any]) -> list[dict[str, str]]:
    supplied = config.get("sources")
    if not isinstance(supplied, list) or not supplied:
        raise ValueError("sources must be a non-empty list")
    result: list[dict[str, str]] = []
    seen: set[str] = set()
    present: set[str] = set()
    for raw in supplied:
        if not isinstance(raw, Mapping):
            raise ValueError("each source must be a mapping")
        source_id = str(raw.get("id", ""))
        split = str(raw.get("split", ""))
        input_xaod = Path(str(raw.get("input_xaod", ""))).expanduser().resolve()
        if not source_id or not source_id.replace("_", "").isalnum() or source_id in seen:
            raise ValueError("source IDs must be unique alphanumeric identifiers")
        if split not in SPLITS:
            raise ValueError(f"source '{source_id}' has invalid split '{split}'")
        if not input_xaod.is_file():
            raise FileNotFoundError(f"source xAOD is unavailable: {input_xaod}")
        seen.add(source_id)
        present.add(split)
        result.append({"source_id": source_id, "split": split, "input_xaod": str(input_xaod)})
    missing = set(SPLITS) - present
    if missing:
        raise ValueError("sources omit split(s): " + ", ".join(sorted(missing)))
    return result


def _source_scan_config(
    config: Mapping[str, Any],
    source: Mapping[str, str],
    directions: Mapping[str, list[dict[str, object]]],
) -> dict[str, object]:
    refit = dict(config["refit"])
    bank = dict(config["payload_bank"])
    scan = {
        "input_xaod": source["input_xaod"],
        "nevents": int(refit["nevents"]),
        "station_ids": [int(value) for value in refit["station_ids"]],
        "reference_station": int(refit["reference_station"]),
        "q_over_p_mode": int(refit.get("q_over_p_mode", 0)),
        "min_truth_match_fraction": float(refit.get("min_truth_match_fraction", 0.99)),
        "chi2_gate": float(refit.get("chi2_gate", 25.0)),
        "refinement_iterations": int(refit.get("refinement_iterations", 1)),
        "prior_sigma_mm": refit.get("prior_sigma_mm"),
        "capture_tolerance_mm": float(refit.get("capture_tolerance_mm", 0.01)),
        "magnitudes_mm": [float(value) for value in bank["magnitudes_mm"]],
        "direction_trials": directions[source["split"]],
        # Fixed-truth closure was already established.  The curriculum corpus
        # requires the expensive physical refit/export, not a duplicate WLS
        # closure at every training payload.
        "run_alignment_closure": False,
    }
    return {"physical_refit_capture_scan": scan}


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _source_event_uids(source_id: str, tracklets: Path) -> list[str]:
    if not tracklets.is_file():
        return []
    events = load_events(tracklets, require_mc_labels=True)
    return [f"{source_id}:{event.run_id}:{event.event_id}" for event in events]


def _manifest(
    output_root: Path,
    config_path: Path,
    config: Mapping[str, Any],
    sources: list[dict[str, str]],
    directions: Mapping[str, list[dict[str, object]]],
) -> dict[str, object]:
    entries: list[dict[str, object]] = []
    for source in sources:
        source_root = output_root / "sources" / source["source_id"]
        source_config_path = source_root / "physical_scan_config.yaml"
        scan_root = source_root / "physical_scan"
        if source_config_path.is_file():
            with source_config_path.open(encoding="utf-8") as handle:
                source_scan = yaml.safe_load(handle)
        else:
            source_scan = _source_scan_config(config, source, directions)
        scan_section = source_scan["physical_refit_capture_scan"]
        plan = _build_plan(scan_section)
        points: list[dict[str, object]] = []
        for point in plan["points"]:
            point_root = scan_root / str(point["relative_point_dir"])
            tracklets = point_root / "refit" / "tracklets.root"
            propagations = point_root / "refit" / "propagations.root"
            payload_manifest = point_root / "payload" / "alignment_payload.json"
            points.append(
                {
                    **point,
                    "payload_id": str(point["name"]),
                    "physical_tracklets": str(tracklets),
                    "physical_propagations": str(propagations),
                    "physical_payload_manifest": str(payload_manifest),
                    "completed": bool(
                        tracklets.is_file() and propagations.is_file() and payload_manifest.is_file()
                    ),
                }
            )
        zero_tracklets = scan_root / "points" / f"mag_0_{directions[source['split']][0]['name']}" / "refit" / "tracklets.root"
        entries.append(
            {
                **source,
                "physical_scan_config": str(source_config_path),
                "physical_scan_root": str(scan_root),
                "source_event_uids": _source_event_uids(source["source_id"], zero_tracklets),
                "points": points,
            }
        )
    return {
        "schema_version": PHYSICAL_CORPUS_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_source": str(config_path),
        "output_root": str(output_root),
        "source_split_unit": "original_xAOD_file",
        "source_event_uid_convention": "source_id:run_id:event_id",
        "physical_geometry_repropagation": True,
        "refit_chain": (
            "persisted SCT_ClusterContainer -> SegmentFitRefit -> SegmentsRefit -> "
            "NtupleDumper -> FaserActsExtrapolationTool"
        ),
        "q_over_p_mode": int(config["refit"].get("q_over_p_mode", 0)),
        "payload_bank": {
            **dict(config["payload_bank"]),
            "generated_direction_trials": directions,
        },
        "sources": entries,
    }


def _run_source(source_root: Path, max_points: int | None, resume: bool) -> None:
    config_path = source_root / "physical_scan_config.yaml"
    scan_root = source_root / "physical_scan"
    command = [
        sys.executable,
        str(SCAN_DRIVER),
        "--config",
        str(config_path),
        "--output-dir",
        str(scan_root),
    ]
    if resume or scan_root.exists():
        command.append("--resume")
    if max_points is not None:
        command.extend(["--max-points", str(max_points)])
    log_path = source_root / "physical_scan_driver.log"
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("# command\n" + " ".join(command) + "\n\n# output\n")
        result = subprocess.run(command, cwd=PROJECT_ROOT, stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode:
        raise subprocess.CalledProcessError(result.returncode, command)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "physical_curriculum_mlp_muon.yaml"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--source-id",
        action="append",
        default=None,
        help="Process only this configured source ID; repeat for a controlled shard",
    )
    parser.add_argument("--max-sources", type=int, default=None)
    parser.add_argument("--max-points-per-source", type=int, default=None)
    args = parser.parse_args()
    if args.max_sources is not None and args.max_sources < 1:
        parser.error("--max-sources must be positive")
    if args.max_points_per_source is not None and args.max_points_per_source < 1:
        parser.error("--max-points-per-source must be positive")
    config_path = Path(args.config).expanduser().resolve()
    config = _load_config(config_path)
    if int(config["refit"].get("q_over_p_mode", 0)) != 0:
        raise ValueError("physical curriculum V1 must use q_over_p_mode=0")
    sources = _validate_sources(config)
    directions = _payload_directions(config)
    output_root = Path(args.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    for source in sources:
        source_root = output_root / "sources" / source["source_id"]
        source_root.mkdir(parents=True, exist_ok=True)
        source_config = _source_scan_config(config, source, directions)
        config_destination = source_root / "physical_scan_config.yaml"
        if config_destination.is_file():
            existing = yaml.safe_load(config_destination.read_text(encoding="utf-8"))
            if existing != source_config:
                raise ValueError(f"existing source config differs for {source['source_id']}")
        else:
            config_destination.write_text(yaml.safe_dump(source_config, sort_keys=True), encoding="utf-8")
    manifest_path = output_root / "physical_corpus_manifest.json"
    _write_json(manifest_path, _manifest(output_root, config_path, config, sources, directions))
    if args.prepare_only:
        print(json.dumps({"manifest": str(manifest_path), "status": "prepared"}, indent=2))
        return
    if args.source_id:
        requested = set(args.source_id)
        known = {source["source_id"] for source in sources}
        unknown = requested - known
        if unknown:
            parser.error("unknown --source-id: " + ", ".join(sorted(unknown)))
        selected = [source for source in sources if source["source_id"] in requested]
    else:
        selected = sources if args.max_sources is None else sources[: args.max_sources]
    for source in selected:
        source_root = output_root / "sources" / source["source_id"]
        _run_source(source_root, args.max_points_per_source, resume=args.resume)
        refreshed = _manifest(output_root, config_path, config, sources, directions)
        _write_json(manifest_path, refreshed)
        if args.max_points_per_source is None:
            source_entry = next(
                entry for entry in refreshed["sources"] if entry["source_id"] == source["source_id"]
            )
            incomplete = [point["payload_id"] for point in source_entry["points"] if not point["completed"]]
            if incomplete:
                raise RuntimeError(
                    f"physical corpus source '{source['source_id']}' has incomplete payloads: "
                    + ", ".join(incomplete)
                )
    print(json.dumps({"manifest": str(manifest_path), "sources_processed": len(selected)}, indent=2))


if __name__ == "__main__":
    main()
