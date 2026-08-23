#!/usr/bin/env python3
"""Materialize identity synthetic-corpus samples for real-data V2 inference.

Copies existing identity tracklets (or writes them from a physical payload),
fans out exact mode-0 Acts predictions onto those endpoints, and emits a
``faser-curriculum-synthetic-corpus-v1`` manifest.  There is no MC overlay,
no threshold change, and no official conditions write.

``--current-only`` materializes the frozen current-geometry windows used for
candidate-graph DQ.  ``--include-fd`` adds the already-produced station FD
probes for calibration-only Station Mode.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import asdict
from typing import Any, Mapping

from datasets.physical_curriculum import SYNTHETIC_CORPUS_SCHEMA
from datasets.real_data_identity import write_real_data_identity_tracklets
from datasets.root_loader import load_events
from datasets.synthetic_field_propagation import write_synthetic_field_candidate_root


CURRENT_POINT = "iteration_00_current"
Q_OVER_P_MODE = 0


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected JSON mapping: {path}")
    return dict(payload)


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _event_uids(source_id: str, tracklets: Path) -> list[str]:
    events = load_events(tracklets, require_mc_labels=False)
    if not events:
        raise ValueError(f"{tracklets} has no events")
    if any(event.truth_particle_id is not None for event in events):
        raise ValueError(f"{tracklets} carries MC truth labels")
    return [f"{source_id}:{int(event.run_id)}:{int(event.event_id)}" for event in events]


def _identity_for_point(
    *,
    source_id: str,
    point: Mapping[str, Any],
    physical_root: Path,
    identity_by_source: Mapping[str, Path],
    sample_root: Path,
) -> Path:
    payload_id = str(point["name"])
    destination = sample_root / "synthetic_tracklets.root"
    if payload_id == CURRENT_POINT and source_id in identity_by_source:
        source = identity_by_source[source_id]
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, destination)
        return destination
    physical_tracklets = (
        physical_root / "sources" / source_id / "physical_scan" / str(point["relative_point_dir"]) / "refit" / "tracklets.root"
    )
    write_real_data_identity_tracklets(physical_tracklets, destination)
    return destination


def _materialize_sample(
    *,
    source: Mapping[str, Any],
    point: Mapping[str, Any],
    physical_root: Path,
    identity_by_source: Mapping[str, Path],
    output_root: Path,
) -> dict[str, Any]:
    source_id = str(source["source_id"])
    payload_id = str(point["name"])
    sample_root = output_root / "samples" / source_id / payload_id
    if sample_root.exists():
        raise FileExistsError(sample_root)
    sample_root.mkdir(parents=True, exist_ok=False)
    relative = str(point["relative_point_dir"])
    point_root = physical_root / "sources" / source_id / "physical_scan" / relative
    physical_tracklets = point_root / "refit" / "tracklets.root"
    physical_propagations = point_root / "refit" / "propagations.root"
    physical_payload = point_root / "payload" / "alignment_payload.json"
    for path in (physical_tracklets, physical_propagations, physical_payload):
        if not path.is_file():
            raise FileNotFoundError(path)
    synthetic = _identity_for_point(
        source_id=source_id,
        point=point,
        physical_root=physical_root,
        identity_by_source=identity_by_source,
        sample_root=sample_root,
    )
    candidates = sample_root / "field_candidates.root"
    summary = write_synthetic_field_candidate_root(
        synthetic_tracklets=synthetic,
        source_propagations=physical_propagations,
        destination=candidates,
        q_over_p_mode=Q_OVER_P_MODE,
        require_mc_labels=False,
    )
    if summary.target_z_mismatch:
        raise RuntimeError(f"target-z mismatch in {source_id}/{payload_id}")
    uids = _event_uids(source_id, synthetic)
    resolved = {
        "source_id": source_id,
        "source_ids": [source_id],
        "split": str(source["split"]),
        "payload_id": payload_id,
        "physical_tracklets": str(physical_tracklets),
        "physical_propagations": str(physical_propagations),
        "physical_payload_manifest": str(physical_payload),
        "synthetic_tracklets": str(synthetic),
        "field_candidates": str(candidates),
        "physical_geometry_repropagation": True,
        "q_over_p_mode": Q_OVER_P_MODE,
        "physical_event_uids": uids,
        "mc_labels": False,
        "overlay": "identity",
        "real_data": True,
    }
    _write_json(sample_root / "resolved_config.json", resolved)
    _write_json(sample_root / "field_candidate_summary.json", asdict(summary))
    sample = {
        **resolved,
        "source_event_uids": uids,
        "condition_axis": str(point.get("condition_axis", "ift_station0_five_dof_survey_dz_l2")),
        "condition_value": float(point.get("condition_value", 0.0)),
        "condition_magnitude": float(point.get("condition_magnitude", 0.0)),
        "magnitude_mm": float(point.get("condition_magnitude", 0.0)),
        "direction_trial": str(point.get("direction_trial", payload_id)),
        "injected_offsets_xy_mm": {},
        "injected_station_transforms": dict(point.get("injected_station_transforms") or {}),
        "alignment_parameter_values": dict(point.get("alignment_parameter_values") or {}),
        "point_name": payload_id,
        "role": str(source["role"]),
        "run": int(source["run"]),
        "candidate_records": int(summary.candidate_records),
        "missing_source_prediction": int(summary.missing_source_prediction),
        "records_by_station_pair": dict(summary.records_by_station_pair),
    }
    return sample


def _selected_points(plan: Mapping[str, Any], *, include_fd: bool) -> list[dict[str, Any]]:
    points = plan.get("points")
    if not isinstance(points, list) or not points:
        raise ValueError("scan plan has no points")
    selected: list[dict[str, Any]] = []
    for raw in points:
        if not isinstance(raw, Mapping):
            raise ValueError("scan plan point is not a mapping")
        name = str(raw.get("name", ""))
        if name == CURRENT_POINT or (include_fd and name.startswith(f"{CURRENT_POINT}_fd_")):
            selected.append(dict(raw))
    if not any(str(point["name"]) == CURRENT_POINT for point in selected):
        raise ValueError("scan plan lacks iteration_00_current")
    if include_fd and len(selected) != 1 + 12:
        raise ValueError(f"expected current + 12 FD probes, got {len(selected)}")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical-output-dir", required=True)
    parser.add_argument("--identity-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--include-fd", action="store_true")
    parser.add_argument(
        "--fd-roles",
        nargs="+",
        default=("calibration",),
        help="Roles allowed to materialize FD probes (default: calibration only).",
    )
    args = parser.parse_args()
    physical = Path(args.physical_output_dir).expanduser().resolve()
    identity_manifest = _read_json(Path(args.identity_manifest).expanduser().resolve())
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    iteration = _read_json(physical / "iteration_manifest.json")
    identity_by_source = {
        str(sample["source_id"]): Path(str(sample["destination"])).expanduser().resolve()
        for sample in identity_manifest["samples"]
        if isinstance(sample, Mapping)
    }
    samples: list[dict[str, Any]] = []
    fd_roles = {str(role) for role in args.fd_roles}
    for source in iteration["sources"]:
        source_id = str(source["source_id"])
        role = str(source["role"])
        include_fd = bool(args.include_fd) and role in fd_roles
        plan = _read_json(Path(str(source["physical_scan_root"])) / "scan_plan.json")
        for point in _selected_points(plan, include_fd=include_fd):
            samples.append(
                _materialize_sample(
                    source=source,
                    point=point,
                    physical_root=physical,
                    identity_by_source=identity_by_source,
                    output_root=output,
                )
            )
    manifest = {
        "schema_version": SYNTHETIC_CORPUS_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "physical_corpus_manifest": str(physical / "iteration_manifest.json"),
        "physical_geometry_repropagation": True,
        "source_split_unit": "original_xAOD_file",
        "source_event_uid_convention": "source_id:run_id:event_id",
        "q_over_p_mode": Q_OVER_P_MODE,
        "overlay": "identity",
        "mc_labels": False,
        "real_data": True,
        "include_fd": bool(args.include_fd),
        "architecture_or_threshold_tuning": False,
        "test_opened": False,
        "samples": samples,
    }
    manifest_path = output / "synthetic_corpus_manifest.json"
    _write_json(manifest_path, manifest)
    by_source: dict[str, list[dict[str, Any]]] = {}
    for sample in samples:
        by_source.setdefault(str(sample["source_id"]), []).append(sample)
    for source_id, source_samples in by_source.items():
        per_source = dict(manifest)
        per_source["samples"] = source_samples
        _write_json(output / "manifests" / f"{source_id}.json", per_source)
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "samples": len(samples),
                "include_fd": bool(args.include_fd),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
