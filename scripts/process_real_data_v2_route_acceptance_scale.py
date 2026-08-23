#!/usr/bin/env python3
"""Materialize identity + frozen V2 association for completed scaling Athena jobs.

Current-geometry only.  No FD probes, no residuals in the association summary
beyond what the frozen V2 runner already writes for provenance, no Station
Mode, and no official conditions write.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from datasets.physical_curriculum import SYNTHETIC_CORPUS_SCHEMA
from datasets.real_data_identity import write_real_data_identity_tracklets
from datasets.root_loader import load_events
from datasets.synthetic_field_propagation import write_synthetic_field_candidate_root
from scripts.submit_real_data_operating_protocol_condor import _complete_source


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CURRENT_POINT = "iteration_00_current"


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected JSON mapping: {path}")
    return dict(payload)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _materialize_source(source: Mapping[str, Any], *, identity_root: Path) -> Path:
    source_id = str(source["source_id"])
    sample_root = identity_root / "samples" / source_id / CURRENT_POINT
    physical_root = Path(str(source["physical_scan_root"]))
    point_root = physical_root / "points" / CURRENT_POINT
    tracklets = point_root / "refit" / "tracklets.root"
    propagations = point_root / "refit" / "propagations.root"
    if not tracklets.is_file() or not propagations.is_file():
        raise FileNotFoundError(f"{source_id} physical current point is incomplete")
    sample_root.mkdir(parents=True, exist_ok=True)
    synthetic = sample_root / "synthetic_tracklets.root"
    if not synthetic.is_file():
        write_real_data_identity_tracklets(tracklets, synthetic)
    events = load_events(synthetic, require_mc_labels=False)
    if any(event.truth_particle_id is not None for event in events):
        raise ValueError(f"{source_id} identity sample carries MC labels")
    candidates = sample_root / "field_candidates.root"
    if not candidates.is_file():
        summary = write_synthetic_field_candidate_root(
            synthetic_tracklets=synthetic,
            source_propagations=propagations,
            destination=candidates,
            q_over_p_mode=0,
            require_mc_labels=False,
        )
        if summary.target_z_mismatch:
            raise RuntimeError(f"target-z mismatch in {source_id}")
    else:
        summary = None
    uids = [f"{source_id}:{int(event.run_id)}:{int(event.event_id)}" for event in events]
    sample = {
        "source_id": source_id,
        "source_ids": [source_id],
        "split": str(source["split"]),
        "payload_id": CURRENT_POINT,
        "point_name": CURRENT_POINT,
        "direction_trial": CURRENT_POINT,
        "synthetic_tracklets": str(synthetic),
        "field_candidates": str(candidates),
        "physical_tracklets": str(tracklets),
        "physical_propagations": str(propagations),
        "physical_payload_manifest": str(point_root / "payload" / "alignment_payload.json"),
        "physical_geometry_repropagation": True,
        "q_over_p_mode": 0,
        "physical_event_uids": uids,
        "source_event_uids": uids,
        "mc_labels": False,
        "overlay": "identity",
        "real_data": True,
        "condition_axis": "ift_station0_five_dof_survey_dz_l2",
        "condition_value": 0.0,
        "condition_magnitude": 0.0,
        "magnitude_mm": 0.0,
        "injected_offsets_xy_mm": {},
        "injected_station_transforms": {
            "0": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "1": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "2": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "3": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        },
        "alignment_parameter_values": {
            "ift_dx_mm": 0.0,
            "ift_dy_mm": 0.0,
            "ift_dz_mm": 0.0,
            "ift_rx_mrad": 0.0,
            "ift_ry_mrad": 0.0,
            "ift_rz_mrad": 0.0,
        },
        "role": str(source["role"]),
        "run": int(source["run"]),
        "candidate_records": None if summary is None else int(summary.candidate_records),
    }
    manifest = {
        "schema_version": SYNTHETIC_CORPUS_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "physical_geometry_repropagation": True,
        "q_over_p_mode": 0,
        "overlay": "identity",
        "mc_labels": False,
        "real_data": True,
        "architecture_or_threshold_tuning": False,
        "test_opened": False,
        "samples": [sample],
    }
    destination = identity_root / "manifests" / f"{source_id}.json"
    _write_json(destination, manifest)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scaling-root", required=True)
    parser.add_argument("--frozen-v2", default=str(PROJECT_ROOT / "outputs" / "mc24_v3_expanded_trainval_v2_bce_control_v1"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--source-id", action="append", default=None)
    parser.add_argument("--run-association", action="store_true")
    args = parser.parse_args()
    scaling_root = Path(args.scaling_root).expanduser().resolve()
    iteration = _read_json(scaling_root / "iteration_manifest.json")
    identity_root = scaling_root / "identity"
    association_root = scaling_root / "association"
    selected = list(iteration["sources"])
    if args.source_id:
        wanted = set(args.source_id)
        selected = [source for source in selected if source["source_id"] in wanted]
    processed: list[str] = []
    for source in selected:
        source_id = str(source["source_id"])
        source_root = scaling_root / "sources" / source_id
        if not _complete_source(source_root):
            continue
        manifest = _materialize_source(source, identity_root=identity_root)
        processed.append(source_id)
        if not args.run_association:
            continue
        output = association_root / source_id
        if output.exists() and any(output.iterdir()):
            continue
        import subprocess

        command = [
            "python",
            "scripts/run_frozen_association_backbone.py",
            "--backbone",
            "v2",
            "--allow-real-data",
            "--q-over-p-mode",
            "0",
            "--split",
            str(source["split"]),
            "--synthetic-manifest",
            str(manifest),
            "--frozen-output",
            str(Path(args.frozen_v2).expanduser().resolve()),
            "--output-dir",
            str(output),
            "--device",
            args.device,
        ]
        result = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
        if result.returncode:
            raise RuntimeError(f"frozen V2 association failed for {source_id}")
    print(json.dumps({"processed": processed, "n": len(processed)}, indent=2))


if __name__ == "__main__":
    main()
