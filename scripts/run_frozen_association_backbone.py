#!/usr/bin/env python3
"""Apply a sealed V1/V2 association backbone without re-tuning it.

The command rebuilds only the existing mode-0 physical candidate graph from a
new train/validation synthetic manifest.  Checkpoint, calibration, score
thresholds, dustbin penalty, and route solver are loaded from the named frozen
validation artifact and are never fitted or selected here.  The emitted route
selection table is truth-free and can feed the global track-fit/alignment
iteration; truth-labelled route metrics are written separately for MC audit.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from baselines.route_assignment import RouteAssignmentConfig, adjacent_station_pairs
from datasets.physical_curriculum import load_synthetic_curriculum_manifest, uniform_condition_axis
from models.field_route_fitter import (
    candidate_lookup,
    field_aware_route_summary,
    selected_field_aware_route,
)
from models.track_fitter import (
    fit_event_route,
    global_track_fit_summary,
    leave_one_out_event_route_residuals,
)
from training.curriculum_mlp import build_candidate_sets
from training.geometry_aware_transformer import (
    ALL_STATION_PAIRS,
    apply_frozen_transformer_calibration,
    build_transformer_graph_bundle,
    candidate_score_metrics,
    load_transformer_artifact,
    predict_transformer_scores,
)
from training.route_assignment import assign_adjacent_route_sets, evaluate_adjacent_route_assignment_sets
from training.route_aware_transformer import (
    load_route_aware_transformer_artifact,
    predict_route_aware_scores,
)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected JSON mapping: {path}")
    return dict(payload)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_ready(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(_json_ready(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields = sorted({str(key) for row in rows for key in row}) or ["empty"]
    values = list(rows) if rows else [{"empty": ""}]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in values:
            writer.writerow(_json_ready(dict(row)))


_STATE_LABELS = ("x_mm", "y_mm", "tx", "ty")
_COVARIANCE_LABELS = (
    (0, 0, "xx_mm2"),
    (0, 1, "xy_mm2"),
    (0, 2, "xtx_mm"),
    (0, 3, "xty_mm"),
    (1, 1, "yy_mm2"),
    (1, 2, "ytx_mm"),
    (1, 3, "yty_mm"),
    (2, 2, "txtx"),
    (2, 3, "txty"),
    (3, 3, "tyty"),
)


def _route_endpoint_metadata(event: object, endpoints: Sequence[tuple[int, int]]) -> list[dict[str, int]]:
    """Return truth-free physical provenance for selected synthetic endpoints."""
    try:
        station_ids = np.asarray(event.station_id, dtype=np.int64)
        tracklet_ids = np.asarray(event.tracklet_id, dtype=np.int64)
        origin_run = getattr(event, "origin_run_id")
        origin_event = getattr(event, "origin_event_id")
        origin_tracklet = getattr(event, "origin_tracklet_id")
        role = getattr(event, "synthetic_role", None)
    except AttributeError as error:
        raise ValueError("route event lacks station/tracklet provenance arrays") from error
    has_origin = origin_run is not None and origin_event is not None and origin_tracklet is not None
    result: list[dict[str, int]] = []
    for expected_station, row in endpoints:
        index = int(row)
        station = int(station_ids[index])
        if station != int(expected_station):
            raise ValueError("route endpoint station does not agree with the event row")
        result.append(
            {
                "event_row": index,
                "station_id": station,
                "synthetic_tracklet_id": int(tracklet_ids[index]),
                "origin_run_id": int(origin_run[index]) if has_origin else int(event.run_id),
                "origin_event_id": int(origin_event[index]) if has_origin else int(event.event_id),
                "origin_tracklet_id": int(origin_tracklet[index]) if has_origin else int(tracklet_ids[index]),
                "synthetic_role": int(role[index]) if role is not None else -1,
            }
        )
    return result


def _route_origin_signature(metadata: Sequence[Mapping[str, int]]) -> str:
    """Stable route identity across refits; it contains no MC truth label."""
    return json.dumps(
        [
            {
                "station_id": int(item["station_id"]),
                "origin_run_id": int(item["origin_run_id"]),
                "origin_event_id": int(item["origin_event_id"]),
                "origin_tracklet_id": int(item["origin_tracklet_id"]),
            }
            for item in metadata
        ],
        sort_keys=True,
        separators=(",", ":"),
    )


def _state_row(prefix: str, value: np.ndarray) -> dict[str, float]:
    values = np.asarray(value, dtype=np.float64)
    if values.shape != (4,) or not np.isfinite(values).all():
        raise ValueError(f"{prefix} state is not a finite four-vector")
    return {f"{prefix}_{label}": float(values[index]) for index, label in enumerate(_STATE_LABELS)}


def _covariance_row(prefix: str, value: np.ndarray) -> dict[str, float]:
    covariance = np.asarray(value, dtype=np.float64)
    if covariance.shape != (4, 4) or not np.isfinite(covariance).all():
        raise ValueError(f"{prefix} covariance is not a finite 4x4 matrix")
    if not np.allclose(covariance, covariance.T, rtol=1.0e-7, atol=1.0e-12):
        raise ValueError(f"{prefix} covariance is not symmetric")
    return {
        f"{prefix}_{label}": float(covariance[row, column])
        for row, column, label in _COVARIANCE_LABELS
    }


def _field_candidates_by_event(candidate_sets: Sequence[object]) -> dict[tuple[str, str, int, int], dict[tuple[int, int, int, int], object]]:
    """Index the exact adjacent ACTS candidates used by the frozen route solver."""
    grouped: dict[tuple[str, str, int, int], dict[tuple[int, int], Sequence[object]]] = {}
    for candidate_set in candidate_sets:
        try:
            sample = candidate_set.sample
            event = candidate_set.event
            pair = tuple(int(value) for value in candidate_set.station_pair)
            candidates = candidate_set.candidates
        except AttributeError as error:
            raise ValueError("candidate set lacks physical route-candidate fields") from error
        key = (str(sample.source_id), str(sample.payload_id), int(event.run_id), int(event.event_id))
        destination = grouped.setdefault(key, {})
        if pair in destination:
            raise ValueError("duplicate adjacent candidate set for one route event")
        destination[pair] = candidates
    return {key: candidate_lookup(value) for key, value in grouped.items()}


def _assert_sealed_contract(contract: Mapping[str, object], *, label: str) -> None:
    if contract.get("loaded_event_splits") != ["train", "validation"]:
        raise ValueError(f"{label} did not declare a train/validation-only model contract")
    if contract.get("test_events_loaded") is not False or contract.get("test_artifacts_opened") is not False:
        raise ValueError(f"{label} is not sealed from test data")
    if "test" not in {str(value) for value in contract.get("forbidden_splits", ())}:
        raise ValueError(f"{label} does not explicitly forbid test")
    if contract.get("physical_geometry_repropagation") is not True or int(contract.get("q_over_p_mode", -1)) not in (0, 3):
        raise ValueError(f"{label} lacks a physical mode-0/mode-3 candidate contract")


def _route_config(operating: Mapping[str, object], maximum_hypotheses: int) -> RouteAssignmentConfig:
    if operating.get("method") != "adjacent_contiguous_unit_capacity_set_packing":
        raise ValueError("frozen operating point uses an unsupported route solver")
    if operating.get("selection_split") != "validation_only" or operating.get("test_opened") is not False:
        raise ValueError("frozen operating point is not validation-only")
    raw_thresholds = operating.get("thresholds")
    if not isinstance(raw_thresholds, Mapping):
        raise ValueError("frozen operating point lacks route thresholds")
    thresholds: dict[tuple[int, int], float] = {}
    for source, target in adjacent_station_pairs((0, 1, 2, 3)):
        value = float(raw_thresholds[f"{source}->{target}"])
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("frozen route threshold is invalid")
        thresholds[(source, target)] = value
    penalty = float(operating["unmatched_penalty"])
    if not np.isfinite(penalty):
        raise ValueError("frozen route unmatched penalty is invalid")
    return RouteAssignmentConfig(
        score_threshold_by_pair=thresholds,
        unmatched_penalty=penalty,
        station_path=(0, 1, 2, 3),
        maximum_hypotheses=int(maximum_hypotheses),
    )


def _maximum_hypotheses(root: Path, section: str) -> int:
    payload = yaml.safe_load((root / "resolved_config.yaml").read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("frozen resolved configuration is malformed")
    settings = payload.get(section)
    if not isinstance(settings, Mapping):
        raise ValueError(f"frozen resolved configuration lacks {section}")
    selection = settings.get("route_selection")
    if not isinstance(selection, Mapping):
        raise ValueError("frozen resolved configuration lacks route_selection")
    maximum = int(selection.get("maximum_hypotheses", 100_000))
    if maximum < 1:
        raise ValueError("frozen maximum_hypotheses must be positive")
    return maximum


def _load_calibration(root: Path) -> Mapping[str, object]:
    wrapper = _read_json(root / "calibration.json")
    calibration = wrapper.get("calibration")
    if (
        wrapper.get("fit_split") != "validation_only"
        or wrapper.get("test_opened") is not False
        or not isinstance(calibration, Mapping)
        or calibration.get("fit_split") != "validation_only"
    ):
        raise ValueError("frozen score calibration is not validation-only")
    return dict(calibration)


def _load_v1(root: Path, ablation: str, device: str):
    contract = _read_json(root / "validation_run_contract.json")
    _assert_sealed_contract(contract, label="V1 frozen artifact")
    ablation_root = root / ablation
    checkpoint = ablation_root / "checkpoint.pt"
    selection = _read_json(ablation_root / "model_selection.json")
    if selection.get("selection_split") != "validation_only" or selection.get("test_opened") is not False:
        raise ValueError("V1 checkpoint selection is not validation-only")
    if selection.get("selected_checkpoint_sha256") != _sha256(checkpoint):
        raise ValueError("V1 selected checkpoint hash differs from frozen artifact")
    calibration = _load_calibration(ablation_root)
    operating = _read_json(ablation_root / "validation_selected_operating_point.json")
    route = _route_config(operating, _maximum_hypotheses(root, "geometry_aware_transformer_v1"))
    model, artifact = load_transformer_artifact(checkpoint, device=device)
    return {
        "model": model,
        "artifact": artifact,
        "calibration": calibration,
        "route": route,
        "metadata": {
            "backbone": "v1",
            "ablation": ablation,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "frozen_condition_axis": contract.get("condition_axis"),
            "frozen_q_over_p_mode": int(contract.get("q_over_p_mode", -1)),
            "thresholds": operating.get("thresholds"),
            "unmatched_penalty": operating.get("unmatched_penalty"),
        },
    }


def _load_v2(root: Path, device: str):
    contract = _read_json(root / "validation_run_contract.json")
    _assert_sealed_contract(contract, label="V2 frozen artifact")
    checkpoint = root / "route_aware_transformer_v2.pt"
    calibration = _load_calibration(root)
    operating = _read_json(root / "validation_selected_operating_point.json")
    route = _route_config(operating, _maximum_hypotheses(root, "route_aware_transformer_v2"))
    model, artifact = load_route_aware_transformer_artifact(checkpoint, device=device)
    return {
        "model": model,
        "artifact": artifact,
        "calibration": calibration,
        "route": route,
        "metadata": {
            "backbone": "v2_bce_route_query",
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "frozen_condition_axis": contract.get("condition_axis"),
            "frozen_q_over_p_mode": int(contract.get("q_over_p_mode", -1)),
            "thresholds": operating.get("thresholds"),
            "unmatched_penalty": operating.get("unmatched_penalty"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-manifest", required=True)
    parser.add_argument("--backbone", choices=("v1", "v2"), required=True)
    parser.add_argument("--frozen-output", required=True)
    parser.add_argument("--v1-ablation", default="geometry_aware_full_context")
    parser.add_argument("--split", choices=("train", "validation"), default="validation")
    parser.add_argument(
        "--payload-id",
        action="append",
        default=None,
        help=(
            "Run only the named real physical payload sample(s). This is an inference filter only: it does not "
            "change the frozen model, calibration, thresholds, dustbin penalty, or route solver."
        ),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--calibration-bins", type=int, default=15)
    parser.add_argument(
        "--q-over-p-mode",
        type=int,
        default=0,
        choices=(0, 3),
        help=(
            "Propagation record variant used to build the physical candidate graph. "
            "Mode 0 is the canonical production baseline; mode 3 is the fixed-q/p-seed "
            "covariance-suppression diagnostic variant. The frozen checkpoint, "
            "calibration, route thresholds, and solver are identical either way."
        ),
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help=(
            "Bound a train/validation inference smoke by physical payload sample. "
            "This never changes the loaded split, frozen model, calibration, or operating point."
        ),
    )
    parser.add_argument(
        "--unmatched-penalty-override",
        type=float,
        default=None,
        help=(
            "Validation-only operating-point scan: replace the frozen route unmatched "
            "penalty for this inference run. The frozen checkpoint, feature "
            "standardizers, calibration, candidate graph, and route solver are "
            "unchanged; the override is recorded in the output metadata."
        ),
    )
    parser.add_argument(
        "--threshold-scale-override",
        type=float,
        default=None,
        help=(
            "Validation-only operating-point scan: multiply every frozen per-pair "
            "score threshold by this factor for this inference run, preserving the "
            "frozen per-pair proportions. Recorded in the output metadata."
        ),
    )
    parser.add_argument(
        "--max-events-per-sample",
        type=int,
        default=None,
        help=(
            "Bound a deterministic smoke to the first physical synthetic events in each selected payload. "
            "It is forbidden for final evaluation but useful for pipeline-contract checks."
        ),
    )
    parser.add_argument(
        "--candidate-chi2-gate",
        type=float,
        default=None,
        help=(
            "Prune the physical candidate graph at this Mahalanobis chi2 before "
            "frozen scoring.  This is a candidate-generation policy knob only: the "
            "frozen checkpoint, calibration, route thresholds, unmatched penalty, "
            "and route solver are unchanged.  Default None reproduces the frozen "
            "ungated candidate graph."
        ),
    )
    args = parser.parse_args()
    if (
        args.batch_size < 1
        or args.calibration_bins < 2
        or (args.max_samples is not None and args.max_samples < 1)
        or (args.max_events_per_sample is not None and args.max_events_per_sample < 1)
    ):
        parser.error("batch size/sample/event limits must be positive and calibration bins must be at least two")
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path, samples, manifest = load_synthetic_curriculum_manifest(
        args.synthetic_manifest,
        require_all_splits=False,
        allowed_splits=(args.split,),
    )
    if {sample.split for sample in samples} != {args.split}:
        raise ValueError("frozen backbone inference loaded an unexpected split")
    full_split_sample_count = len(samples)
    if args.payload_id is not None:
        requested_payloads = {str(value) for value in args.payload_id}
        available_payloads = {str(sample.payload_id) for sample in samples}
        unknown_payloads = requested_payloads - available_payloads
        if unknown_payloads:
            raise ValueError("--payload-id is absent from the selected split: " + ", ".join(sorted(unknown_payloads)))
        samples = [sample for sample in samples if str(sample.payload_id) in requested_payloads]
        if not samples:  # pragma: no cover - covered by unknown-payload validation
            raise ValueError("--payload-id excludes every selected physical payload")
    if args.max_samples is not None:
        samples = samples[: args.max_samples]
        if not samples:  # pragma: no cover - checked positive and manifest non-empty
            raise ValueError("--max-samples excludes every requested physical payload")
    if manifest.get("physical_geometry_repropagation") is not True or int(
        manifest.get("q_over_p_mode", -1)
    ) != int(args.q_over_p_mode):
        raise ValueError(
            f"synthetic manifest lacks a physical mode-{args.q_over_p_mode} candidate contract"
        )
    root = Path(args.frozen_output).expanduser().resolve()
    frozen = _load_v1(root, args.v1_ablation, args.device) if args.backbone == "v1" else _load_v2(root, args.device)
    if args.unmatched_penalty_override is not None or args.threshold_scale_override is not None:
        route = frozen["route"]
        penalty = route.unmatched_penalty if args.unmatched_penalty_override is None else float(args.unmatched_penalty_override)
        scale = 1.0 if args.threshold_scale_override is None else float(args.threshold_scale_override)
        if not np.isfinite(penalty):
            raise ValueError("unmatched penalty override must be finite")
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("threshold scale override must be positive and finite")
        thresholds = {
            pair: min(1.0, max(0.0, float(value) * scale))
            for pair, value in route.score_threshold_by_pair.items()
        }
        frozen["route"] = dataclasses.replace(
            route,
            score_threshold_by_pair=thresholds,
            unmatched_penalty=penalty,
        )
        frozen["metadata"]["operating_point_override"] = {
            "unmatched_penalty": penalty,
            "threshold_scale": scale,
            "thresholds": {f"{source}->{target}": value for (source, target), value in thresholds.items()},
            "selection_split": "validation_only",
        }
    candidate_sets = build_candidate_sets(
        samples,
        ALL_STATION_PAIRS,
        chi2_gate=args.candidate_chi2_gate,
        feature_set="residual_v1",
        max_events_per_sample=args.max_events_per_sample,
        q_over_p_mode=int(args.q_over_p_mode),
    )
    artifact = frozen["artifact"]
    bundle = build_transformer_graph_bundle(candidate_sets, context_mode=artifact.context_mode)
    field_candidates = _field_candidates_by_event(bundle.adjacent_sets)
    if args.backbone == "v1":
        raw_scores = predict_transformer_scores(
            frozen["model"],
            bundle,
            artifact.node_standardizer,
            artifact.edge_standardizer,
            device=args.device,
            batch_size=args.batch_size,
        )
    else:
        prediction = predict_route_aware_scores(
            frozen["model"],
            bundle,
            artifact.node_standardizer,
            artifact.edge_standardizer,
            device=args.device,
            batch_size=args.batch_size,
        )
        raw_scores = list(prediction.edge_scores)
    calibrated = apply_frozen_transformer_calibration(bundle.adjacent_sets, raw_scores, frozen["calibration"])
    selections = assign_adjacent_route_sets(
        bundle.adjacent_sets,
        calibrated,
        frozen["route"],
        calibration_bins=args.calibration_bins,
    )
    # Evaluation remains MC truth-labelled, but route selection and global
    # track fitting above are deliberately truth-free.
    route_evaluation = evaluate_adjacent_route_assignment_sets(
        bundle.adjacent_sets,
        calibrated,
        frozen["route"],
        calibration_bins=args.calibration_bins,
    )
    route_rows: list[dict[str, object]] = []
    fit_rows: list[dict[str, object]] = []
    leave_one_out_rows: list[dict[str, object]] = []
    field_edge_rows: list[dict[str, object]] = []
    all_fits = []
    straight_fit_rejected_routes = 0
    all_field_routes = []
    for assigned in selections:
        event_candidates = field_candidates.get(tuple(assigned.key))
        if event_candidates is None:
            raise RuntimeError("selected route event is absent from the physical candidate lookup")
        for route_index, route in enumerate(assigned.result.routes):
            endpoints = tuple(int(index) for _, index in route.endpoints)
            # This straight fit is retained only to expose its incompatibility
            # with magnetic-field trajectories.  It must never block the
            # exact field-aware selected-edge alignment observation below.
            try:
                fit = fit_event_route(assigned.event, endpoints)
            except ValueError:
                fit = None
                straight_fit_rejected_routes += 1
            else:
                all_fits.append(fit)
            endpoint_metadata = _route_endpoint_metadata(assigned.event, route.endpoints)
            metadata_by_event_row = {
                int(item["event_row"]): item for item in endpoint_metadata
            }
            origin_signature = _route_origin_signature(endpoint_metadata)
            field_route = selected_field_aware_route(route, event_candidates)
            all_field_routes.append(field_route)
            route_rows.append(
                {
                    "sample_id": assigned.key[0],
                    "payload_id": assigned.key[1],
                    "run_id": assigned.key[2],
                    "event_id": assigned.key[3],
                    "route_index": route_index,
                    "endpoint_stations": [int(station) for station, _ in route.endpoints],
                    "endpoint_indices": list(endpoints),
                    "endpoint_provenance": endpoint_metadata,
                    "route_origin_signature": origin_signature,
                    "utility": float(route.utility),
                    "complete_route_score": route.complete_route_score,
                    "is_complete_four_station_route": len(endpoints) == 4,
                    "global_track_chi2": None if fit is None else fit.chi2,
                    "global_track_ndof": None if fit is None else fit.ndof,
                    "global_track_reduced_chi2": None if fit is None else fit.reduced_chi2,
                    "field_aware_edge_count": field_route.edge_count,
                    "field_aware_edge_chi2_sum": field_route.edge_chi2_sum,
                }
            )
            if fit is not None:
                fit_rows.append(
                    {
                        "sample_id": assigned.key[0],
                        "payload_id": assigned.key[1],
                        "run_id": assigned.key[2],
                        "event_id": assigned.key[3],
                        "route_index": route_index,
                        "endpoint_indices": list(fit.endpoint_indices),
                        "station_ids": list(fit.station_ids),
                        "route_origin_signature": origin_signature,
                        "chi2": fit.chi2,
                        "ndof": fit.ndof,
                        "reduced_chi2": fit.reduced_chi2,
                        "normal_matrix_rank": fit.normal_matrix_rank,
                    }
                )
            # Retain the old straight-line leave-one-out rows as a diagnostic
            # comparison.  The physical alignment update consumes the exact
            # field-aware route edges emitted below instead.
            if len(endpoints) >= 3:
                for leave_one_out in leave_one_out_event_route_residuals(assigned.event, endpoints):
                    target = next(
                        item
                        for item in endpoint_metadata
                        if int(item["station_id"]) == int(leave_one_out.station_id)
                        and int(item["synthetic_tracklet_id"])
                        == int(assigned.event.tracklet_id[leave_one_out.endpoint_index])
                    )
                    row: dict[str, object] = {
                        "sample_id": assigned.key[0],
                        "payload_id": assigned.key[1],
                        "run_id": assigned.key[2],
                        "event_id": assigned.key[3],
                        "route_index": route_index,
                        "route_origin_signature": origin_signature,
                        "route_endpoint_count": len(endpoints),
                        "target_station_id": int(leave_one_out.station_id),
                        "target_event_row": int(leave_one_out.endpoint_index),
                        "target_synthetic_tracklet_id": int(target["synthetic_tracklet_id"]),
                        "target_origin_run_id": int(target["origin_run_id"]),
                        "target_origin_event_id": int(target["origin_event_id"]),
                        "target_origin_tracklet_id": int(target["origin_tracklet_id"]),
                        "target_synthetic_role": int(target["synthetic_role"]),
                        "z_mm": float(leave_one_out.z_mm),
                        "chi2": float(leave_one_out.chi2),
                        "reference_fit_chi2": float(leave_one_out.reference_fit.chi2),
                        "reference_fit_ndof": int(leave_one_out.reference_fit.ndof),
                    }
                    row.update(_state_row("prediction", leave_one_out.prediction))
                    row.update(_state_row("residual", leave_one_out.residual))
                    row.update(_covariance_row("prediction_cov", leave_one_out.prediction_covariance))
                    row.update(_covariance_row("combined_cov", leave_one_out.combined_covariance))
                    leave_one_out_rows.append(row)
            for edge in field_route.edges:
                source = metadata_by_event_row.get(int(edge.source_index))
                target = metadata_by_event_row.get(int(edge.target_index))
                if source is None or target is None:
                    raise RuntimeError("selected physical edge endpoint has no route provenance")
                row = {
                    "sample_id": assigned.key[0],
                    "payload_id": assigned.key[1],
                    "run_id": assigned.key[2],
                    "event_id": assigned.key[3],
                    "route_index": route_index,
                    "route_origin_signature": origin_signature,
                    "route_endpoint_count": len(endpoints),
                    "source_station_id": int(edge.source_station_id),
                    "target_station_id": int(edge.target_station_id),
                    "source_event_row": int(edge.source_index),
                    "target_event_row": int(edge.target_index),
                    "source_synthetic_tracklet_id": int(source["synthetic_tracklet_id"]),
                    "target_synthetic_tracklet_id": int(target["synthetic_tracklet_id"]),
                    "source_origin_run_id": int(source["origin_run_id"]),
                    "source_origin_event_id": int(source["origin_event_id"]),
                    "source_origin_tracklet_id": int(source["origin_tracklet_id"]),
                    "target_origin_run_id": int(target["origin_run_id"]),
                    "target_origin_event_id": int(target["origin_event_id"]),
                    "target_origin_tracklet_id": int(target["origin_tracklet_id"]),
                    "source_synthetic_role": int(source["synthetic_role"]),
                    "target_synthetic_role": int(target["synthetic_role"]),
                    "score": float(edge.score),
                    "chi2": float(edge.chi2),
                }
                row.update(_state_row("residual", edge.residual))
                row.update(_state_row("pull", edge.pull))
                row.update(_covariance_row("combined_cov", edge.combined_covariance))
                field_edge_rows.append(row)
    with (output / "selected_routes.jsonl").open("w", encoding="utf-8") as handle:
        for row in route_rows:
            handle.write(json.dumps(_json_ready(row), sort_keys=True, allow_nan=False) + "\n")
    _write_csv(output / "selected_route_global_track_fits.csv", fit_rows)
    _write_csv(output / "selected_route_leave_one_out_residuals.csv", leave_one_out_rows)
    _write_csv(output / "selected_route_field_edge_residuals.csv", field_edge_rows)
    summary = {
        "method": "frozen_physical_association_then_field_aware_route_consistency",
        "synthetic_manifest": str(manifest_path),
        "split": args.split,
        "full_split_physical_payload_samples": full_split_sample_count,
        "inference_physical_payload_samples": len(samples),
        "inference_payload_filter": None if args.payload_id is None else sorted({str(value) for value in args.payload_id}),
        "inference_smoke_limit": args.max_samples,
        "inference_event_limit_per_payload": args.max_events_per_sample,
        "synthetic_condition_axis": uniform_condition_axis(samples),
        "physical_geometry_repropagation": True,
        "q_over_p_mode": int(args.q_over_p_mode),
        "test_opened": False,
        "architecture_or_threshold_tuning": False,
        "candidate_chi2_gate": (
            None if args.candidate_chi2_gate is None else float(args.candidate_chi2_gate)
        ),
        "frozen_backbone": frozen["metadata"],
        "route_solver": {
            "method": "adjacent_contiguous_unit_capacity_set_packing",
            "thresholds": {
                f"{source}->{target}": float(value)
                for (source, target), value in frozen["route"].score_threshold_by_pair.items()
            },
            "unmatched_penalty": float(frozen["route"].unmatched_penalty),
        },
        "truth_free_selection": {
            "events": len(selections),
            "selected_routes": len(route_rows),
            "complete_four_station_routes": int(sum(bool(row["is_complete_four_station_route"]) for row in route_rows)),
            "leave_one_out_observations": len(leave_one_out_rows),
            "leave_one_out_observation_contract": (
                "selected-route endpoint residual against a global fit of the other selected stations; "
                "origin provenance is emitted without MC truth labels"
            ),
            "field_aware_edge_observations": len(field_edge_rows),
            "field_aware_edge_observation_contract": (
                f"Each selected adjacent route edge retains the exact mode-{int(args.q_over_p_mode)} ACTS "
                "residual, pull, and combined covariance from the existing physical candidate graph. "
                "This is the alignment-update observation."
            ),
        },
        "field_aware_route_consistency": field_aware_route_summary(all_field_routes),
        "straight_line_global_track_diagnostic": {
            **global_track_fit_summary(all_fits),
            "rejected_routes": straight_fit_rejected_routes,
            "alignment_objective": False,
        },
        "truth_labelled_mc_evaluation": route_evaluation,
        "raw_candidate_score_metrics": candidate_score_metrics(
            bundle.adjacent_sets, raw_scores, args.calibration_bins
        ),
        "frozen_calibrated_score_metrics": candidate_score_metrics(
            bundle.adjacent_sets, calibrated, args.calibration_bins
        ),
    }
    _write_json(output / "association_summary.json", summary)
    print(
        json.dumps(
            _json_ready(
                {
                    "output_dir": str(output),
                    "field_aware_route_consistency": summary["field_aware_route_consistency"],
                    "straight_line_global_track_diagnostic": summary["straight_line_global_track_diagnostic"],
                }
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
