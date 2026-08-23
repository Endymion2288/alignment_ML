#!/usr/bin/env python3
"""Materialize FD identity samples and run calibration-only self-nulling Newton.

Does not retune V2, does not start C_dx Mode, and does not write official
conditions.  Residual reduction is recorded as DQ only.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from datasets.real_data_identity import write_real_data_identity_tracklets
from datasets.root_loader import load_events
from datasets.synthetic_field_propagation import write_synthetic_field_candidate_root


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CURRENT_POINT = "iteration_00_current"
PARAMETERS = (
    "ift_dx_mm",
    "ift_dy_mm",
    "ift_dz_mm",
    "ift_rx_mrad",
    "ift_ry_mrad",
    "ift_rz_mrad",
)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected JSON mapping: {path}")
    return dict(payload)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _symlink(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        return
    destination.symlink_to(source)


def _materialize_point(
    *,
    source_id: str,
    point: Mapping[str, Any],
    physical_scan_root: Path,
    sample_root: Path,
    reuse_current: Path | None,
) -> None:
    payload_id = str(point["name"])
    sample_root.mkdir(parents=True, exist_ok=True)
    point_root = physical_scan_root / str(point["relative_point_dir"])
    physical_tracklets = point_root / "refit" / "tracklets.root"
    physical_propagations = point_root / "refit" / "propagations.root"
    physical_payload = point_root / "payload" / "alignment_payload.json"
    synthetic = sample_root / "synthetic_tracklets.root"
    candidates = sample_root / "field_candidates.root"
    if payload_id == CURRENT_POINT and reuse_current is not None:
        _symlink(reuse_current / "synthetic_tracklets.root", synthetic)
        _symlink(reuse_current / "field_candidates.root", candidates)
    else:
        if not synthetic.is_file():
            write_real_data_identity_tracklets(physical_tracklets, synthetic)
        if not candidates.is_file():
            summary = write_synthetic_field_candidate_root(
                synthetic_tracklets=synthetic,
                source_propagations=physical_propagations,
                destination=candidates,
                q_over_p_mode=0,
                require_mc_labels=False,
            )
            if summary.target_z_mismatch:
                raise RuntimeError(f"target-z mismatch in {source_id}/{payload_id}")
    events = load_events(synthetic, require_mc_labels=False)
    uids = [f"{source_id}:{int(event.run_id)}:{int(event.event_id)}" for event in events]
    _write_json(
        sample_root / "resolved_config.json",
        {
            "source_id": source_id,
            "payload_id": payload_id,
            "synthetic_tracklets": str(synthetic),
            "field_candidates": str(candidates),
            "physical_tracklets": str(physical_tracklets),
            "physical_propagations": str(physical_propagations),
            "physical_payload_manifest": str(physical_payload),
            "physical_geometry_repropagation": True,
            "q_over_p_mode": 0,
            "physical_event_uids": uids,
            "mc_labels": False,
            "overlay": "identity",
            "real_data": True,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True)
    parser.add_argument(
        "--scaling-root",
        default=str(PROJECT_ROOT / "outputs" / "operating_protocol_v1_real_data_v2_route_acceptance_scaling_v1"),
    )
    parser.add_argument("--source-id", action="append", default=None)
    parser.add_argument("--skip-newton", action="store_true")
    args = parser.parse_args()
    campaign = Path(args.campaign_root).expanduser().resolve()
    physical = campaign / "physical"
    identity_root = campaign / "identity"
    newton_root = campaign / "newton"
    iteration = _read_json(physical / "iteration_manifest.json")
    selected = [row for row in iteration["sources"] if row["role"] == "calibration"]
    if args.source_id:
        wanted = set(args.source_id)
        selected = [row for row in selected if row["source_id"] in wanted]
    processed: list[str] = []
    for source in selected:
        source_id = str(source["source_id"])
        scan_root = Path(str(source["physical_scan_root"]))
        plan = _read_json(scan_root / "scan_plan.json")
        reuse_current = (
            Path(args.scaling_root).expanduser().resolve()
            / "identity"
            / "samples"
            / source_id
            / CURRENT_POINT
        )
        for point in plan["points"]:
            _materialize_point(
                source_id=source_id,
                point=point,
                physical_scan_root=scan_root,
                sample_root=identity_root / "samples" / source_id / str(point["name"]),
                reuse_current=reuse_current if reuse_current.is_dir() else None,
            )
        processed.append(source_id)
        if args.skip_newton:
            continue
        output = newton_root / source_id
        if output.exists() and (output / "route_selected_update.json").is_file():
            continue
        command = [
            "python",
            "scripts/run_route_selected_multidof_update.py",
            "--scan-root",
            str(scan_root),
            "--anchor-point",
            CURRENT_POINT,
            "--anchor-association-output",
            str(Path(args.scaling_root).expanduser().resolve() / "association" / source_id),
            "--anchor-payload-sample",
            str(identity_root / "samples" / source_id / CURRENT_POINT),
            "--self-nulling-update",
            "--allow-real-data",
            "--observation-kind",
            "anchor_selected_field_edge",
            "--observation-statistics",
            "physical_edge_deduplicated",
            "--q-over-p-mode",
            "0",
            "--calibration-mode",
            "station",
            "--mode-validity-contract",
            str(PROJECT_ROOT / "outputs" / "mc24_ift_calibration_mode_validity_contract_v1" / "mode_validity_contract.json"),
            "--cdx-fixed-by",
            "external_geometry",
            "--capture-criteria",
            str(PROJECT_ROOT / "outputs" / "mc24_ift_5dof_survey_dz_capture_criteria_train_v1" / "capture_criteria.json"),
            "--covariance-calibration",
            str(
                PROJECT_ROOT
                / "outputs"
                / "mc24_multidof_ift_pull_calibration_train_v1"
                / "mode0_covariance_calibration_train_frozen.json"
            ),
            "--prior-sigma",
            "ift_dz_mm:5.0",
            "--output-dir",
            str(output),
        ]
        for name in PARAMETERS:
            command.extend(
                [
                    "--positive-association-output",
                    f"{name}:{identity_root / 'samples' / source_id / f'{CURRENT_POINT}_fd_{name}_p'}",
                    "--negative-association-output",
                    f"{name}:{identity_root / 'samples' / source_id / f'{CURRENT_POINT}_fd_{name}_m'}",
                ]
            )
        result = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
        if result.returncode:
            raise RuntimeError(f"self-nulling Newton failed for {source_id}")
        update = _read_json(output / "route_selected_update.json")
        values = dict(update.get("proposed_next_parameter_values") or {})
        _write_json(
            output / "update_summary.json",
            {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "run": int(source["run"]),
                "source_id": source_id,
                "self_nulling_update": True,
                "geometry_write_allowed": False,
                "residual_reduction_is_not_alignment_success": True,
                "parameter_update": values,
                "alignment_parameter_values": values,
                "capture": {
                    "passed": False,
                    "applicable": False,
                    "reason": "real_data_self_nulling_has_no_injected_target",
                },
            },
        )
    print(json.dumps({"processed": processed, "n": len(processed)}, indent=2))


if __name__ == "__main__":
    main()
