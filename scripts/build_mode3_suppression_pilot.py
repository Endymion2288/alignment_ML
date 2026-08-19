#!/usr/bin/env python3
"""Build the fixed-q/p-seed covariance-suppression (mode-3) pilot artifacts.

Two stages:

* ``manifest`` derives a pilot physical corpus manifest from the frozen
  iteration-1 production manifest by selecting a small set of train sources
  whose scan was re-produced with the mode-3 diagnostic dumper variant.  Every
  point is completion-checked against the pilot scan root so a partial
  production can never enter the overlay pool.
* ``mode3-overlay`` takes an already materialized pilot overlay (canonical
  mode-0 candidates) and exports the mode-3 candidate graph for the identical
  synthetic events, writing parallel per-payload sample directories and a
  mode-3 synthetic corpus manifest.  The frozen backbone/update tooling can
  then run the exact same events under either covariance variant.

Both stages are read-only with respect to the frozen production artifacts.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from datasets.synthetic_field_propagation import write_synthetic_field_candidate_root

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODE3 = 3


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _rewrite_path(value: str, production_root: Path, pilot_root: Path) -> str:
    path = Path(value)
    try:
        relative = path.relative_to(production_root)
    except ValueError as error:
        raise ValueError(f"manifest path escapes the production root: {value}") from error
    return str(pilot_root / relative)


def build_pilot_manifest(
    production_manifest: Path,
    pilot_physical_root: Path,
    source_ids: Sequence[str],
    output: Path,
) -> dict[str, Any]:
    """Select pilot train sources and re-anchor every path at the pilot root."""
    production = _read_json(production_manifest)
    production_root = production_manifest.parent
    selected = []
    for source in production["sources"]:
        source_id = str(source["source_id"])
        if source_id not in set(source_ids):
            continue
        if str(source["split"]) != "train":
            raise ValueError(f"pilot source {source_id} is not a train source")
        pilot_source = dict(source)
        pilot_source["physical_scan_config"] = _rewrite_path(
            str(source["physical_scan_config"]), production_root, pilot_physical_root
        )
        pilot_source["physical_scan_root"] = _rewrite_path(
            str(source["physical_scan_root"]), production_root, pilot_physical_root
        )
        points = []
        for point in source["points"]:
            pilot_point = dict(point)
            for key in (
                "physical_tracklets",
                "physical_propagations",
                "physical_payload_manifest",
                "physical_content_audit",
            ):
                if key in pilot_point and pilot_point[key] is not None:
                    pilot_point[key] = _rewrite_path(
                        str(pilot_point[key]), production_root, pilot_physical_root
                    )
            required = [
                pilot_point.get("physical_tracklets"),
                pilot_point.get("physical_propagations"),
                pilot_point.get("physical_payload_manifest"),
                pilot_point.get("physical_content_audit"),
            ]
            complete = all(value is not None and Path(str(value)).is_file() for value in required)
            pilot_point["completed"] = complete
            pilot_point["completion_status"] = "accepted" if complete else "incomplete"
            points.append(pilot_point)
        pilot_source["points"] = points
        selected.append(pilot_source)
    missing = set(source_ids) - {str(source["source_id"]) for source in selected}
    if missing:
        raise ValueError("pilot sources absent from the production manifest: " + ", ".join(sorted(missing)))

    manifest = {
        key: value
        for key, value in production.items()
        if key not in {"sources", "created_utc", "config_source", "source_split_audit"}
    }
    manifest["created_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["config_source"] = "scripts/build_mode3_suppression_pilot.py"
    manifest["sources"] = selected
    manifest["pilot_provenance"] = {
        "production_manifest": str(production_manifest),
        "pilot_physical_root": str(pilot_physical_root),
        "pilot_sources": sorted(source_ids),
        "purpose": (
            "fixed-q/p-seed covariance-suppression (mode-3) diagnostic pilot; "
            "mode-0/1/2 production records are unchanged and compared side by side"
        ),
    }
    _write_json(output, manifest)
    return manifest


def build_mode3_overlay(overlay_root: Path, output_root: Path) -> dict[str, Any]:
    """Export mode-3 candidates for the identical pilot overlay events."""
    manifest_path = overlay_root / "synthetic_corpus_manifest.json"
    manifest = _read_json(manifest_path)
    if int(manifest.get("q_over_p_mode", -1)) != 0:
        raise ValueError("pilot overlay must be the canonical mode-0 materialization")
    samples_out = []
    for sample in manifest["samples"]:
        split = str(sample["split"])
        payload_id = str(sample["payload_id"])
        sample_root = overlay_root / "samples" / split / payload_id
        synthetic = sample_root / "synthetic_tracklets.root"
        pooled = sample_root / "pooled_physical_propagations.root"
        if not synthetic.is_file() or not pooled.is_file():
            raise FileNotFoundError(f"pilot overlay sample is incomplete: {split}/{payload_id}")
        destination_dir = output_root / "samples" / split / payload_id
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination_synthetic = destination_dir / "synthetic_tracklets.root"
        if not destination_synthetic.is_file():
            shutil.copyfile(synthetic, destination_synthetic)
        destination_candidates = destination_dir / "field_candidates.root"
        summary = write_synthetic_field_candidate_root(
            synthetic_tracklets=destination_synthetic,
            source_propagations=pooled,
            destination=destination_candidates,
            q_over_p_mode=MODE3,
            target_z_tolerance_mm=float(
                manifest.get("field_candidate_export", {}).get("target_z_tolerance_mm", 1.0e-6)
            ),
        )
        if summary.target_z_mismatch:
            raise RuntimeError(f"mode-3 candidate export has a target-z mismatch: {split}/{payload_id}")
        resolved = _read_json(sample_root / "resolved_config.json")
        resolved["q_over_p_mode"] = MODE3
        resolved["synthetic_tracklets"] = str(destination_synthetic)
        resolved["field_candidates"] = str(destination_candidates)
        resolved["mode3_provenance"] = {
            "mode0_sample": str(sample_root),
            "candidate_export": "write_synthetic_field_candidate_root(q_over_p_mode=3)",
            "candidate_summary": {
                "candidate_records": int(summary.candidate_records),
                "missing_source_prediction": int(summary.missing_source_prediction),
                "records_by_station_pair": dict(summary.records_by_station_pair),
            },
        }
        _write_json(destination_dir / "resolved_config.json", resolved)
        sample_out = dict(sample)
        sample_out["synthetic_tracklets"] = str(destination_synthetic)
        sample_out["field_candidates"] = str(destination_candidates)
        samples_out.append(sample_out)
    manifest_out = dict(manifest)
    manifest_out["q_over_p_mode"] = MODE3
    manifest_out["field_candidate_export"] = {
        **dict(manifest.get("field_candidate_export", {})),
        "q_over_p_mode": MODE3,
    }
    manifest_out["samples"] = samples_out
    manifest_out["mode3_provenance"] = {
        "mode0_overlay": str(overlay_root),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(output_root / "synthetic_corpus_manifest.json", manifest_out)
    return manifest_out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="stage", required=True)

    manifest_parser = subparsers.add_parser("manifest", help="build the pilot physical corpus manifest")
    manifest_parser.add_argument("--production-manifest", required=True)
    manifest_parser.add_argument("--pilot-physical-root", required=True)
    manifest_parser.add_argument("--source-id", action="append", required=True)
    manifest_parser.add_argument("--output", required=True)

    overlay_parser = subparsers.add_parser("mode3-overlay", help="export mode-3 candidates for a pilot overlay")
    overlay_parser.add_argument("--overlay-root", required=True)
    overlay_parser.add_argument("--output-root", required=True)

    args = parser.parse_args()
    if args.stage == "manifest":
        manifest = build_pilot_manifest(
            Path(args.production_manifest).expanduser().resolve(),
            Path(args.pilot_physical_root).expanduser().resolve(),
            [str(value) for value in args.source_id],
            Path(args.output).expanduser().resolve(),
        )
        print(json.dumps({"sources": len(manifest["sources"])}, indent=2))
    else:
        manifest = build_mode3_overlay(
            Path(args.overlay_root).expanduser().resolve(),
            Path(args.output_root).expanduser().resolve(),
        )
        print(json.dumps({"samples": len(manifest["samples"])}, indent=2))


if __name__ == "__main__":
    main()
