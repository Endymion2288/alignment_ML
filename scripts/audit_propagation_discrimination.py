#!/usr/bin/env python3
"""Edge-level truth/fake dataset for the mode-0 propagation chi2 mechanism study.

Collects every ungated field-aware candidate edge (same frozen endpoint
definition as the association backbone, no chi2 gate) for adjacent station
pairs, labelled by MC truth, from

- physical sources of an iteration bank (truth edges; single-track events),
- multi-track synthetic overlay samples (truth and fake edges),

at multiple payloads (e.g. anchor and reference) so misalignment dependence
can be assessed.  Each edge records the residual, the full combined
covariance, marginal pulls, the Mahalanobis chi2, the source tracklet state,
and whether the edge was route-selected by the frozen backbone (joined by
synthetic event/tracklet provenance).  Validation edges are collected for the
single frozen transfer evaluation only; no model parameter may be derived
from them.  Read-only: nothing is refit, reselected, or retrained.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from baselines.field_chi2_matching import build_field_candidates
from datasets.propagation_loader import load_propagation_records
from datasets.root_loader import load_events

ADJACENT_PAIRS = ((0, 1), (1, 2), (2, 3))


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))

SCHEMA_VERSION = "faser-propagation-discrimination-edges-v1"
STATE_LABELS = ("x_mm", "y_mm", "tx", "ty")
COV_LABELS = (
    "xx_mm2", "xy_mm2", "xtx_mm", "xty_mm",
    "yy_mm2", "ytx_mm", "yty_mm", "txtx", "txty", "tyty",
)

FIELDNAMES = [
    "split", "sample_kind", "origin_id", "payload", "station_pair",
    "run_id", "event_id", "source_tracklet_id", "target_tracklet_id",
    "is_truth", "route_selected",
    "source_state_x_mm", "source_state_y_mm", "source_state_tx", "source_state_ty",
    "residual_x_mm", "residual_y_mm", "residual_tx", "residual_ty",
    "pull_x_mm", "pull_y_mm", "pull_tx", "pull_ty",
    "chi2",
] + [f"combined_cov_{label}" for label in COV_LABELS]


def _upper_triangular(covariance: np.ndarray) -> list[float]:
    return [float(covariance[i, j]) for i in range(4) for j in range(i, 4)]


def _edge_rows(
    events: Any,
    records: Any,
    *,
    split: str,
    sample_kind: str,
    origin_id: str,
    payload: str,
    selected_keys: set[tuple[int, int, int, int, int, int]] | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for event in events:
        if event.truth_particle_id is None:
            continue
        for source_station, target_station in ADJACENT_PAIRS:
            candidates = build_field_candidates(
                event, records, source_station, target_station, chi2_gate=None
            )
            for candidate in candidates:
                source_index = int(candidate.source_index)
                target_index = int(candidate.target_index)
                is_truth = int(event.truth_particle_id[source_index]) == int(
                    event.truth_particle_id[target_index]
                )
                source_tracklet = int(event.tracklet_id[source_index])
                target_tracklet = int(event.tracklet_id[target_index])
                route_selected = False
                if selected_keys is not None:
                    route_selected = (
                        int(event.run_id),
                        int(event.event_id),
                        source_tracklet,
                        target_tracklet,
                        source_station,
                        target_station,
                    ) in selected_keys
                row: dict[str, Any] = {
                    "split": split,
                    "sample_kind": sample_kind,
                    "origin_id": origin_id,
                    "payload": payload,
                    "station_pair": f"{source_station}->{target_station}",
                    "run_id": int(event.run_id),
                    "event_id": int(event.event_id),
                    "source_tracklet_id": source_tracklet,
                    "target_tracklet_id": target_tracklet,
                    "is_truth": int(is_truth),
                    "route_selected": int(route_selected),
                    "source_state_x_mm": float(event.state[source_index][0]),
                    "source_state_y_mm": float(event.state[source_index][1]),
                    "source_state_tx": float(event.state[source_index][2]),
                    "source_state_ty": float(event.state[source_index][3]),
                    "chi2": float(candidate.chi2),
                }
                for label, value in zip(STATE_LABELS, candidate.residual):
                    row[f"residual_{label}"] = float(value)
                for label, value in zip(STATE_LABELS, candidate.pull):
                    row[f"pull_{label}"] = float(value)
                for label, value in zip(COV_LABELS, _upper_triangular(candidate.combined_covariance)):
                    row[f"combined_cov_{label}"] = value
                rows.append(row)
    return rows


def _load_selected_keys(anchor_table: Path) -> set[tuple[int, int, int, int, int, int]]:
    keys: set[tuple[int, int, int, int, int, int]] = set()
    with anchor_table.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            keys.add(
                (
                    int(row["run_id"]),
                    int(row["event_id"]),
                    int(row["source_synthetic_tracklet_id"]),
                    int(row["target_synthetic_tracklet_id"]),
                    int(row["source_station_id"]),
                    int(row["target_station_id"]),
                )
            )
    return keys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iteration-manifest", required=True)
    parser.add_argument("--physical-point", action="append", default=[])
    parser.add_argument("--synthetic-root", default=None)
    parser.add_argument("--synthetic-payload", action="append", default=[])
    parser.add_argument(
        "--backbone-anchor",
        action="append",
        default=[],
        help="PAYLOAD:DIR of a frozen backbone anchor payload for route-selected flags",
    )
    parser.add_argument("--split", required=True, choices=("train", "validation"))
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)

    selected_by_payload: dict[str, set[tuple[int, int, int, int, int, int]]] = {}
    for entry in args.backbone_anchor:
        payload, _, directory = entry.partition(":")
        selected_by_payload[payload] = _load_selected_keys(
            Path(directory).expanduser().resolve() / "selected_route_field_edge_residuals.csv"
        )

    rows: list[dict[str, Any]] = []
    manifest = _read_json(Path(args.iteration_manifest).expanduser().resolve())
    for entry in manifest["sources"]:
        if str(entry["split"]) != args.split:
            continue
        source_id = str(entry["source_id"])
        root = Path(str(entry["physical_scan_root"])).expanduser().resolve()
        plan = _read_json(root / "scan_plan.json")
        points = {str(point["name"]): point for point in plan["points"]}
        for point_name in args.physical_point:
            point = points.get(point_name)
            if point is None:
                raise ValueError(f"point '{point_name}' absent from {root}/scan_plan.json")
            point_root = root / str(point["relative_point_dir"])
            events = load_events(point_root / "refit" / "tracklets.root", require_mc_labels=True)
            records = load_propagation_records(point_root / "refit" / "propagations.root")
            rows.extend(
                _edge_rows(
                    events,
                    records,
                    split=args.split,
                    sample_kind="physical",
                    origin_id=source_id,
                    payload=point_name,
                    selected_keys=None,
                )
            )

    if args.synthetic_root is not None:
        synthetic_root = Path(args.synthetic_root).expanduser().resolve()
        for payload in args.synthetic_payload:
            sample_dir = synthetic_root / args.split / payload
            events = load_events(sample_dir / "synthetic_tracklets.root", require_mc_labels=True)
            records = load_propagation_records(sample_dir / "field_candidates.root")
            rows.extend(
                _edge_rows(
                    events,
                    records,
                    split=args.split,
                    sample_kind="synthetic_overlay",
                    origin_id=f"{args.split}/{payload}",
                    payload=payload,
                    selected_keys=selected_by_payload.get(payload),
                )
            )

    table_path = output / "edges.csv"
    with table_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "split": args.split,
        "rows": len(rows),
        "truth_rows": int(sum(row["is_truth"] for row in rows)),
        "fake_rows": int(sum(1 - row["is_truth"] for row in rows)),
        "route_selected_rows": int(sum(row["route_selected"] for row in rows)),
        "physical_points": list(args.physical_point),
        "synthetic_payloads": list(args.synthetic_payload),
        "test_data_accessed": False,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output), "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
