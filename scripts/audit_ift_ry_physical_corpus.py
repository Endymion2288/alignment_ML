#!/usr/bin/env python3
"""Audit a completed source-disjoint physical IFT-R_y corpus.

This is a read-only post-refit audit.  It consumes only per-point outputs
produced by the real /Tracker/Align -> SCT cluster -> segment refit -> Acts
chain and rejects a corpus that is incomplete or contains test sources.  It
does not construct coordinate-shift surrogates, synthetic overlays, scores, or
assignments.  Truth labels are used after physical candidate construction to
measure raw truth-edge and IFT -> S1 -> S2 -> S3 route retention.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from alignment.payload import load_station_rigid_alignment_payload
from datasets.propagation_loader import load_propagation_records
from datasets.root_loader import EventTracklets, load_events
from evaluation.field_propagation import evaluate_field_propagation
from scripts.audit_ift_ry_refit_response import (
    PREDICTOR_LABELS,
    RESIDUAL_LABELS,
    STATE_LABELS,
    _linear_response,
    _propagation_response,
    _tracklet_response,
)
from scripts.audit_physical_route_candidate_graph import (
    _acts_surface_summary,
    _condition_chain_checks,
    audit_candidate_graph,
)


STATION_PATH = (0, 1, 2, 3)
ADJACENT_PAIRS = tuple(zip(STATION_PATH[:-1], STATION_PATH[1:]))


def _load_manifest(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError("physical corpus manifest must be a mapping")
    if payload.get("physical_geometry_repropagation") is not True:
        raise ValueError("physical corpus must declare physical_geometry_repropagation=true")
    if int(payload.get("q_over_p_mode", -1)) != 0:
        raise ValueError("IFT R_y physical corpus audit requires q_over_p_mode=0")
    if "test" not in {str(value) for value in payload.get("forbidden_splits", ())}:
        raise ValueError("physical corpus must explicitly forbid test")
    return dict(payload)


def _scalar(row: Mapping[str, object], name: str) -> float:
    value = row.get(name)
    if value is None:
        return float("nan")
    return float(value)


def _point_metadata(source: Mapping[str, object], point: Mapping[str, object]) -> dict[str, object]:
    axis = str(point.get("condition_axis", ""))
    if axis != "ift_ry_mrad":
        raise ValueError(f"physical rotation corpus has unsupported condition axis: {axis!r}")
    transforms = point.get("injected_station_transforms")
    if not isinstance(transforms, Mapping):
        raise ValueError("rotation point lacks injected_station_transforms")
    payload = load_station_rigid_alignment_payload(Path(str(point["physical_payload_manifest"])))
    for station in (1, 2, 3):
        if any(abs(value) > 1.0e-15 for value in payload.transform_for_station(station)):
            raise ValueError("IFT R_y corpus requires downstream stations 1--3 to remain fixed")
    return {
        "source_id": str(source["source_id"]),
        "split": str(source["split"]),
        "payload_id": str(point["payload_id"]),
        "condition_axis": axis,
        "condition_value": float(point["condition_value"]),
        "condition_magnitude": float(point["condition_magnitude"]),
        "direction_trial": str(point["direction_trial"]),
        "joint_dx_mm": float(point.get("joint_translation_xy_mm", [0.0, 0.0])[0]),
        "joint_dy_mm": float(point.get("joint_translation_xy_mm", [0.0, 0.0])[1]),
        "ift_ry_mrad_payload": float(payload.transform_for_station(0)[4] * 1.0e3),
    }


def _completed_points(source: Mapping[str, object]) -> list[dict[str, object]]:
    raw_points = source.get("points")
    if not isinstance(raw_points, list) or not raw_points:
        raise ValueError(f"source {source.get('source_id')} lacks physical points")
    points = [dict(point) for point in raw_points if isinstance(point, Mapping)]
    if len(points) != len(raw_points):
        raise ValueError("physical point entry must be a mapping")
    incomplete = [str(point.get("payload_id", point.get("name", "?"))) for point in points if not point.get("completed")]
    if incomplete:
        raise RuntimeError(
            f"physical corpus source {source.get('source_id')} is incomplete: " + ", ".join(incomplete)
        )
    return points


def _nominal_point(points: Sequence[Mapping[str, object]]) -> Mapping[str, object]:
    nominal = [point for point in points if abs(float(point["condition_value"])) <= 1.0e-15]
    if len(nominal) != 1:
        raise ValueError("each source must contain exactly one nominal IFT R_y point")
    return nominal[0]


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty audit table: {path.name}")
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _finite_ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else float(numerator / denominator)


def _condition_key(row: Mapping[str, object]) -> tuple[object, ...]:
    return (
        row["split"],
        row["condition_axis"],
        float(row["condition_value"]),
        float(row["condition_magnitude"]),
        row["direction_trial"],
        float(row["joint_dx_mm"]),
        float(row["joint_dy_mm"]),
    )


def _condition_metadata(key: tuple[object, ...]) -> dict[str, object]:
    return {
        "split": key[0],
        "condition_axis": key[1],
        "condition_value": key[2],
        "condition_magnitude": key[3],
        "direction_trial": key[4],
        "joint_dx_mm": key[5],
        "joint_dy_mm": key[6],
    }


def _candidate_summary(point_rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], dict[str, Any]] = {}
    for row in point_rows:
        key = _condition_key(row)
        entry = grouped.setdefault(
            key,
            {
                "sources": 0,
                "complete_truth_chains": 0,
                "retained_truth_chains": 0,
                "surface_errors": 0,
                "layer_overlaps": 0,
                "all_condition_checks": True,
                "pairs": defaultdict(lambda: {"truth": 0, "retained": 0, "edges": 0}),
            },
        )
        entry["sources"] += 1
        entry["complete_truth_chains"] += int(row["complete_truth_chains"])
        entry["retained_truth_chains"] += int(row["candidate_retained_complete_truth_chains"])
        entry["surface_errors"] += int(row["acts_surface_error_count"])
        entry["layer_overlaps"] += int(row["acts_layer_overlap_error_count"])
        entry["all_condition_checks"] = bool(entry["all_condition_checks"]) and bool(
            row["condition_chain_complete"]
        )
        for source, target in ADJACENT_PAIRS:
            label = f"{source}->{target}"
            entry["pairs"][label]["truth"] += int(row[f"{label}_truth_pairs"])
            entry["pairs"][label]["retained"] += int(row[f"{label}_retained_truth_pairs"])
            entry["pairs"][label]["edges"] += int(row[f"{label}_physical_candidate_edges"])
    summary: list[dict[str, object]] = []
    for key, entry in sorted(grouped.items()):
        row = {
            **_condition_metadata(key),
            "sources": int(entry["sources"]),
            "complete_truth_chains": int(entry["complete_truth_chains"]),
            "candidate_retained_complete_truth_chains": int(entry["retained_truth_chains"]),
            "candidate_complete_truth_chain_recall": _finite_ratio(
                int(entry["retained_truth_chains"]), int(entry["complete_truth_chains"])
            ),
            "acts_surface_error_count": int(entry["surface_errors"]),
            "acts_layer_overlap_error_count": int(entry["layer_overlaps"]),
            "condition_chain_complete": bool(entry["all_condition_checks"]),
        }
        for label, counts in sorted(entry["pairs"].items()):
            row[f"{label}_truth_pairs"] = int(counts["truth"])
            row[f"{label}_retained_truth_pairs"] = int(counts["retained"])
            row[f"{label}_candidate_truth_edge_recall"] = _finite_ratio(
                int(counts["retained"]), int(counts["truth"])
            )
            row[f"{label}_physical_candidate_edges"] = int(counts["edges"])
        summary.append(row)
    return summary


def _regressions(
    rows: Sequence[Mapping[str, object]],
    *,
    domain: str,
    response_labels: Sequence[str],
    predictor_prefix: str,
    group_fields: Sequence[str],
) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[field] for field in group_fields)].append(row)
    output: list[dict[str, object]] = []
    for key, group in sorted(grouped.items()):
        base = dict(zip(group_fields, key))
        for response in response_labels:
            values = np.asarray([_scalar(row, response) for row in group], dtype=np.float64)
            for predictor in PREDICTOR_LABELS:
                fit = _linear_response(
                    np.asarray([_scalar(row, f"{predictor_prefix}{predictor}") for row in group]),
                    values,
                )
                output.append(
                    {
                        **base,
                        "domain": domain,
                        "response": response,
                        "predictor": predictor,
                        **fit,
                    }
                )
    return output


def _point_row(
    metadata: Mapping[str, object],
    graph: Mapping[str, object],
    surface: Mapping[str, object],
    chain: Mapping[str, object] | None,
) -> dict[str, object]:
    row: dict[str, object] = {
        **metadata,
        "events": int(graph["events"]),
        "complete_truth_chains": int(graph["complete_truth_chains"]),
        "candidate_retained_complete_truth_chains": int(graph["candidate_retained_complete_truth_chains"]),
        "candidate_complete_truth_chain_recall": graph["candidate_complete_truth_chain_recall"],
        "acts_surface_error_count": 0 if chain is None else int(chain["acts_surface_error_count"]),
        "acts_layer_overlap_error_count": 0 if chain is None else int(chain["acts_layer_overlap_error_count"]),
        "condition_chain_complete": False if chain is None else all(bool(value) for value in chain["checks"].values()),
    }
    pairs = graph["by_station_pair"]
    if not isinstance(pairs, Mapping):
        raise ValueError("candidate graph pair summary is invalid")
    for source, target in ADJACENT_PAIRS:
        label = f"{source}->{target}"
        pair = pairs[label]
        if not isinstance(pair, Mapping):
            raise ValueError(f"candidate graph lacks pair {label}")
        row[f"{label}_truth_pairs"] = int(pair["truth_pairs_with_unique_endpoints"])
        row[f"{label}_retained_truth_pairs"] = int(pair["candidate_retained_truth_pairs"])
        row[f"{label}_candidate_truth_edge_recall"] = pair["candidate_truth_edge_recall"]
        row[f"{label}_physical_candidate_edges"] = int(pair["physical_candidate_edges"])
        observed = surface.get(label, {})
        if not isinstance(observed, Mapping):
            raise ValueError(f"surface response lacks pair {label}")
        row[f"{label}_acts_records"] = int(observed.get("records", 0))
        row[f"{label}_acts_success"] = int(observed.get("success", 0))
        row[f"{label}_acts_finite_covariance"] = int(observed.get("finite_covariance", 0))
    return row


def audit_corpus(
    manifest: Mapping[str, object], *, splits: set[str], chi2_gate: float | None
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    point_rows: list[dict[str, object]] = []
    state_rows: list[dict[str, object]] = []
    propagation_rows: list[dict[str, object]] = []
    raw_sources = manifest.get("sources")
    if not isinstance(raw_sources, list):
        raise ValueError("physical corpus manifest lacks sources")
    for raw_source in raw_sources:
        if not isinstance(raw_source, Mapping):
            raise ValueError("physical corpus source must be a mapping")
        source = dict(raw_source)
        split = str(source["split"])
        if split == "test":
            raise ValueError("sealed test source appears in physical corpus")
        if split not in splits:
            continue
        points = _completed_points(source)
        nominal = _nominal_point(points)
        nominal_events = load_events(Path(str(nominal["physical_tracklets"])), require_mc_labels=True)
        nominal_records = load_propagation_records(Path(str(nominal["physical_propagations"])))
        nominal_evaluation = evaluate_field_propagation(
            nominal_events,
            nominal_records,
            require_truth_match=True,
            q_over_p_mode=0,
            min_truth_match_fraction=0.99,
        )
        for point in points:
            metadata = _point_metadata(source, point)
            events = load_events(Path(str(point["physical_tracklets"])), require_mc_labels=True)
            records = load_propagation_records(Path(str(point["physical_propagations"])))
            graph = audit_candidate_graph(events, records, chi2_gate=chi2_gate)
            refit_log = (
                Path(str(source["physical_scan_root"]))
                / str(point["relative_point_dir"])
                / "logs"
                / "refit.log"
            )
            point_rows.append(
                _point_row(metadata, graph, _acts_surface_summary(records), _condition_chain_checks(refit_log))
            )
            if abs(float(metadata["condition_value"])) <= 1.0e-15:
                continue
            tracklet_rows, _ = _tracklet_response(nominal_events, events)
            displaced_evaluation = evaluate_field_propagation(
                events,
                records,
                require_truth_match=True,
                q_over_p_mode=0,
                min_truth_match_fraction=0.99,
            )
            propagation_delta_rows, _ = _propagation_response(nominal_evaluation, displaced_evaluation)
            predictors = {
                (int(row["run_id"]), int(row["event_id"]), int(row["tracklet_id"])): row
                for row in tracklet_rows
            }
            for row in tracklet_rows:
                state_rows.append({**metadata, **row})
            for row in propagation_delta_rows:
                source_predictor = predictors.get(
                    (int(row["run_id"]), int(row["event_id"]), int(row["source_tracklet_id"]))
                )
                enriched = {**metadata, **row}
                for predictor in PREDICTOR_LABELS:
                    enriched[f"source_nominal_{predictor}"] = (
                        None
                        if source_predictor is None
                        else source_predictor[f"nominal_{predictor}"]
                    )
                propagation_rows.append(enriched)
    if not point_rows:
        raise ValueError("no completed train/validation physical points selected")
    return point_rows, state_rows, propagation_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", action="append", choices=("train", "validation"), default=None)
    parser.add_argument("--chi2-gate", type=float, default=None)
    args = parser.parse_args()
    if args.chi2_gate is not None and (not np.isfinite(args.chi2_gate) or args.chi2_gate <= 0.0):
        parser.error("--chi2-gate must be finite and positive")
    manifest_path = Path(args.physical_manifest).expanduser().resolve()
    manifest = _load_manifest(manifest_path)
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    splits = set(args.split or ("train", "validation"))
    point_rows, state_rows, propagation_rows = audit_corpus(
        manifest, splits=splits, chi2_gate=args.chi2_gate
    )
    candidate_summary = _candidate_summary(point_rows)
    state_regressions = _regressions(
        state_rows,
        domain="tracklet_state",
        response_labels=tuple(f"delta_{label}" for label in STATE_LABELS),
        predictor_prefix="nominal_",
        group_fields=(
            "split",
            "condition_axis",
            "condition_value",
            "condition_magnitude",
            "direction_trial",
            "joint_dx_mm",
            "joint_dy_mm",
            "station_id",
        ),
    )
    propagation_regressions = _regressions(
        propagation_rows,
        domain="mode0_propagation",
        response_labels=tuple(f"delta_{label}" for label in RESIDUAL_LABELS)
        + tuple(f"delta_pull_{label}" for label in RESIDUAL_LABELS)
        + ("delta_chi2",),
        predictor_prefix="source_nominal_",
        group_fields=(
            "split",
            "condition_axis",
            "condition_value",
            "condition_magnitude",
            "direction_trial",
            "joint_dx_mm",
            "joint_dy_mm",
            "source_station_id",
            "target_station_id",
        ),
    )
    _write_csv(output / "point_metrics.csv", point_rows)
    _write_csv(output / "candidate_summary.csv", candidate_summary)
    _write_csv(output / "position_dependence.csv", state_regressions + propagation_regressions)
    summary = {
        "method": "read_only_source_disjoint_physical_ift_ry_corpus_audit",
        "physical_geometry_repropagation": True,
        "coordinate_surrogate": False,
        "q_over_p_mode": 0,
        "physical_manifest": str(manifest_path),
        "selected_splits": sorted(splits),
        "chi2_gate": args.chi2_gate,
        "points": len(point_rows),
        "non_nominal_tracklet_response_rows": len(state_rows),
        "non_nominal_propagation_response_rows": len(propagation_rows),
        "candidate_summary": candidate_summary,
        "position_dependence_rows": len(state_regressions) + len(propagation_regressions),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output_dir": str(output),
                "points": len(point_rows),
                "candidate_conditions": len(candidate_summary),
                "position_dependence_rows": summary["position_dependence_rows"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
