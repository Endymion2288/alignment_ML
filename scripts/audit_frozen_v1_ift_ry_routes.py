#!/usr/bin/env python3
"""Audit frozen V1 route controls by signed IFT R_y condition on validation.

The program is intentionally post-selection only.  It rebuilds the existing
physical candidate graph for validation samples, applies each ablation's saved
checkpoint, calibration, thresholds, and dustbin penalty exactly as frozen,
then reports pure positive/negative rotations and joint dx/dy trials
separately.  It never opens a test asset or performs model selection.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from baselines.route_assignment import RouteAssignmentConfig, adjacent_station_pairs
from datasets.physical_curriculum import (
    CurriculumSample,
    load_synthetic_curriculum_manifest,
    uniform_condition_axis,
)
from training.curriculum_mlp import CandidateSet, build_candidate_sets
from training.geometry_aware_transformer import (
    ALL_STATION_PAIRS,
    apply_frozen_transformer_calibration,
    build_transformer_graph_bundle,
    candidate_score_metrics,
    load_transformer_artifact,
    predict_transformer_scores,
)
from training.route_assignment import evaluate_adjacent_route_assignment_sets


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ABLATIONS = (
    "geometry_aware_full_context",
    "geometry_aware_no_multistation_context",
)
ROUTE_METRICS = (
    "candidate_complete_truth_chain_recall",
    "score_threshold_complete_truth_chain_recall",
    "complete_track_efficiency",
    "complete_track_purity",
    "track_fake_rate",
    "missing_station_recovery",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected a JSON mapping: {path}")
    return dict(payload)


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(_json_value(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields = sorted({str(key) for row in rows for key in row})
    if not fields:
        fields = ["empty"]
        rows = [{"empty": ""}]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(_json_value(dict(row)))


def _assert_sealed(contract: Mapping[str, object], label: str) -> None:
    if contract.get("loaded_event_splits") != ["train", "validation"]:
        raise ValueError(f"{label} did not train on exactly train and validation")
    if contract.get("test_events_loaded") is not False:
        raise ValueError(f"{label} is not sealed from test events")
    if contract.get("test_opened") not in (False, None):
        raise ValueError(f"{label} opened test data")
    if contract.get("test_artifacts_opened") not in (False, None):
        raise ValueError(f"{label} opened test artifacts")
    if contract.get("condition_axis") != "ift_ry_mrad":
        raise ValueError(f"{label} is not an explicit IFT R_y validation artifact")


def _criteria(payload: Mapping[str, object], label: str) -> dict[str, float]:
    raw = payload.get("capture_success_criteria")
    if not isinstance(raw, Mapping):
        raise ValueError(f"{label} lacks capture-success criteria")
    names = (
        "minimum_complete_track_efficiency",
        "minimum_complete_track_purity",
        "maximum_track_fake_rate",
    )
    if any(name not in raw for name in names):
        raise ValueError(f"{label} has incomplete capture-success criteria")
    return {name: float(raw[name]) for name in names}


def _capture_success(route: Mapping[str, object], criteria: Mapping[str, float]) -> bool:
    try:
        return bool(
            float(route["complete_track_efficiency"])
            >= criteria["minimum_complete_track_efficiency"]
            and float(route["complete_track_purity"])
            >= criteria["minimum_complete_track_purity"]
            and float(route["track_fake_rate"])
            <= criteria["maximum_track_fake_rate"]
        )
    except (KeyError, TypeError, ValueError):
        return False


def _station0_translation(sample: CurriculumSample) -> tuple[float, float]:
    raw = sample.injected_station_transforms
    value = raw.get("0", raw.get(0)) if isinstance(raw, Mapping) else None
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) == 6:
        return float(value[0]), float(value[1])
    offsets = sample.injected_offsets_xy_mm
    fallback = offsets.get("0", offsets.get(0)) if isinstance(offsets, Mapping) else None
    if isinstance(fallback, Sequence) and not isinstance(fallback, (str, bytes)) and len(fallback) == 2:
        return float(fallback[0]), float(fallback[1])
    return 0.0, 0.0


def _condition_fields(sample: CurriculumSample) -> dict[str, object]:
    dx, dy = _station0_translation(sample)
    return {
        "condition_axis": sample.condition_axis,
        "condition_value": float(
            sample.curriculum_magnitude
            if sample.condition_value is None
            else sample.condition_value
        ),
        "condition_magnitude": float(sample.curriculum_magnitude),
        "direction_trial": sample.direction_trial,
        "joint_dx_mm": dx,
        "joint_dy_mm": dy,
        "pure_ry_trial": bool(np.isclose(dx, 0.0) and np.isclose(dy, 0.0)),
    }


def _sample_key(sample: CurriculumSample) -> tuple[str, str]:
    return str(sample.source_id), str(sample.payload_id)


def _sets_for_sample(
    samples: Sequence[CurriculumSample], candidate_sets: Sequence[CandidateSet]
) -> dict[tuple[str, str], list[int]]:
    expected = {_sample_key(sample) for sample in samples}
    grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, candidate_set in enumerate(candidate_sets):
        grouped[_sample_key(candidate_set.sample)].append(index)
    if set(grouped) != expected:
        raise ValueError("candidate sets do not cover exactly the requested validation samples")
    return grouped


def _parse_route_config(
    operating: Mapping[str, object], maximum_hypotheses: int
) -> RouteAssignmentConfig:
    if operating.get("method") != "adjacent_contiguous_unit_capacity_set_packing":
        raise ValueError("frozen V1 operating point uses an unexpected route solver")
    raw_thresholds = operating.get("thresholds")
    if not isinstance(raw_thresholds, Mapping):
        raise ValueError("frozen V1 operating point has no thresholds")
    station_path = (0, 1, 2, 3)
    thresholds: dict[tuple[int, int], float] = {}
    for left, right in adjacent_station_pairs(station_path):
        value = float(raw_thresholds[f"{left}->{right}"])
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("frozen V1 score threshold is invalid")
        thresholds[(left, right)] = value
    penalty = float(operating["unmatched_penalty"])
    if not np.isfinite(penalty):
        raise ValueError("frozen V1 unmatched penalty is invalid")
    return RouteAssignmentConfig(
        score_threshold_by_pair=thresholds,
        unmatched_penalty=penalty,
        station_path=station_path,
        maximum_hypotheses=maximum_hypotheses,
    )


def _route_trial_rows(
    ablation: str,
    samples: Sequence[CurriculumSample],
    candidate_sets: Sequence[CandidateSet],
    scores: Sequence[np.ndarray],
    config: RouteAssignmentConfig,
    criteria: Mapping[str, float],
    calibration_bins: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    grouped = _sets_for_sample(samples, candidate_sets)
    by_sample = {_sample_key(sample): sample for sample in samples}
    rows: list[dict[str, object]] = []
    details: list[dict[str, object]] = []
    for key in sorted(grouped):
        indices = grouped[key]
        evaluation = evaluate_adjacent_route_assignment_sets(
            [candidate_sets[index] for index in indices],
            [scores[index] for index in indices],
            config,
            calibration_bins=calibration_bins,
        )
        route = evaluation.get("route")
        if not isinstance(route, Mapping):
            raise RuntimeError("frozen V1 route evaluation is malformed")
        sample = by_sample[key]
        success = _capture_success(route, criteria)
        rows.append(
            {
                "ablation": ablation,
                "source_id": sample.source_id,
                "payload_id": sample.payload_id,
                **_condition_fields(sample),
                "capture_success": success,
                **dict(route),
            }
        )
        details.append(
            {
                "source_id": sample.source_id,
                "payload_id": sample.payload_id,
                **_condition_fields(sample),
                "capture_success": success,
                "evaluation": evaluation,
            }
        )
    return rows, details


def _condition_summary(
    ablation: str,
    trial_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    grouped: dict[tuple[object, ...], list[Mapping[str, object]]] = defaultdict(list)
    fields = (
        "condition_axis",
        "condition_value",
        "condition_magnitude",
        "direction_trial",
        "joint_dx_mm",
        "joint_dy_mm",
        "pure_ry_trial",
    )
    for row in trial_rows:
        grouped[tuple(row[field] for field in fields)].append(row)
    for key, rows in sorted(grouped.items()):
        output: dict[str, object] = {
            "ablation": ablation,
            **dict(zip(fields, key)),
            "samples": len(rows),
            "capture_successes": int(sum(bool(row["capture_success"]) for row in rows)),
            "capture_fraction": float(np.mean([bool(row["capture_success"]) for row in rows])),
        }
        for metric in ROUTE_METRICS:
            values = np.asarray(
                [float(row[metric]) for row in rows if row.get(metric) is not None],
                dtype=np.float64,
            )
            output[f"mean_{metric}"] = None if not values.size else float(np.mean(values))
            output[f"std_{metric}"] = None if not values.size else float(np.std(values))
        result.append(output)
    return result


def _magnitude_summary(
    ablation: str,
    trial_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    grouped: dict[float, list[Mapping[str, object]]] = defaultdict(list)
    for row in trial_rows:
        grouped[float(row["condition_magnitude"])].append(row)
    for magnitude, rows in sorted(grouped.items()):
        output: dict[str, object] = {
            "ablation": ablation,
            "condition_axis": "ift_ry_mrad",
            "condition_magnitude": magnitude,
            "direction_trials": len(rows),
            "capture_successes": int(sum(bool(row["capture_success"]) for row in rows)),
            "capture_fraction": float(np.mean([bool(row["capture_success"]) for row in rows])),
        }
        for metric in ROUTE_METRICS:
            values = np.asarray(
                [float(row[metric]) for row in rows if row.get(metric) is not None],
                dtype=np.float64,
            )
            output[f"direction_mean_{metric}"] = None if not values.size else float(np.mean(values))
            output[f"direction_std_{metric}"] = None if not values.size else float(np.std(values))
        result.append(output)
    return result


def _load_maximum_hypotheses(root: Path) -> int:
    config_path = root / "resolved_config.yaml"
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("frozen V1 resolved configuration is malformed")
    settings = payload.get("geometry_aware_transformer_v1")
    if not isinstance(settings, Mapping):
        raise ValueError("frozen V1 resolved configuration lacks settings")
    route = settings.get("route_selection")
    if not isinstance(route, Mapping):
        raise ValueError("frozen V1 resolved configuration lacks route selection")
    maximum = int(route.get("maximum_hypotheses", 100_000))
    if maximum < 1:
        raise ValueError("frozen V1 maximum_hypotheses must be positive")
    return maximum


def _load_ablation_artifacts(
    root: Path, ablation: str, maximum_hypotheses: int
) -> tuple[Path, Mapping[str, object], Mapping[str, object], RouteAssignmentConfig]:
    ablation_root = root / ablation
    checkpoint = ablation_root / "checkpoint.pt"
    selected = _read_json(ablation_root / "model_selection.json")
    calibration_wrapper = _read_json(ablation_root / "calibration.json")
    operating = _read_json(ablation_root / "validation_selected_operating_point.json")
    if selected.get("selection_split") != "validation_only" or selected.get("test_opened") is not False:
        raise ValueError(f"V1 {ablation} model selection is not validation-only")
    if selected.get("selected_checkpoint_sha256") != _sha256(checkpoint):
        raise ValueError(f"V1 {ablation} checkpoint differs from its frozen model selection")
    calibration = calibration_wrapper.get("calibration")
    if (
        calibration_wrapper.get("fit_split") != "validation_only"
        or calibration_wrapper.get("test_opened") is not False
        or not isinstance(calibration, Mapping)
        or calibration.get("fit_split") != "validation_only"
    ):
        raise ValueError(f"V1 {ablation} calibration is not validation-only")
    if operating.get("selection_split") != "validation_only" or operating.get("test_opened") is not False:
        raise ValueError(f"V1 {ablation} route selection is not validation-only")
    return checkpoint, dict(calibration), operating, _parse_route_config(operating, maximum_hypotheses)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-manifest", required=True)
    parser.add_argument("--v1-output", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--ablation", action="append", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--calibration-bins", type=int, default=15)
    args = parser.parse_args()
    if args.batch_size < 1 or args.calibration_bins < 2:
        parser.error("batch size must be positive and calibration bins must be at least two")

    output_root = Path(args.output_dir).expanduser().resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError("refusing to overwrite a non-empty frozen V1 audit output")
    output_root.mkdir(parents=True, exist_ok=True)
    v1_root = Path(args.v1_output).expanduser().resolve()
    contract = _read_json(v1_root / "validation_run_contract.json")
    _assert_sealed(contract, "V1 validation output")
    maximum_hypotheses = _load_maximum_hypotheses(v1_root)

    manifest_path, samples, manifest = load_synthetic_curriculum_manifest(
        args.synthetic_manifest,
        require_all_splits=False,
        allowed_splits=("validation",),
    )
    if {sample.split for sample in samples} != {"validation"}:
        raise RuntimeError("frozen V1 audit loaded a non-validation sample")
    if uniform_condition_axis(samples) != "ift_ry_mrad":
        raise ValueError("frozen V1 audit requires an explicit IFT R_y synthetic corpus")
    if manifest.get("physical_geometry_repropagation") is not True or int(manifest.get("q_over_p_mode", -1)) != 0:
        raise ValueError("synthetic manifest does not certify physical mode-0 propagation")
    requested = tuple(args.ablation or DEFAULT_ABLATIONS)
    if not requested or len(requested) != len(set(requested)):
        raise ValueError("requested V1 ablations must be non-empty and unique")

    candidate_sets = build_candidate_sets(
        samples, ALL_STATION_PAIRS, chi2_gate=None, feature_set="residual_v1"
    )
    bundles: dict[str, object] = {}
    trial_rows: list[dict[str, object]] = []
    condition_rows: list[dict[str, object]] = []
    magnitude_rows: list[dict[str, object]] = []
    metadata: dict[str, object] = {}
    for ablation in requested:
        checkpoint, calibration, operating, route_config = _load_ablation_artifacts(
            v1_root, ablation, maximum_hypotheses
        )
        model, artifact = load_transformer_artifact(checkpoint, device=args.device)
        bundle = bundles.get(artifact.context_mode)
        if bundle is None:
            bundle = build_transformer_graph_bundle(candidate_sets, context_mode=artifact.context_mode)
            bundles[artifact.context_mode] = bundle
        raw_scores = predict_transformer_scores(
            model,
            bundle,
            artifact.node_standardizer,
            artifact.edge_standardizer,
            device=args.device,
            batch_size=args.batch_size,
        )
        calibrated = apply_frozen_transformer_calibration(
            bundle.adjacent_sets, raw_scores, calibration
        )
        criteria = _criteria(operating, f"V1 {ablation}")
        rows, details = _route_trial_rows(
            ablation,
            samples,
            list(bundle.adjacent_sets),
            list(calibrated),
            route_config,
            criteria,
            args.calibration_bins,
        )
        trial_rows.extend(rows)
        condition_rows.extend(_condition_summary(ablation, rows))
        magnitude_rows.extend(_magnitude_summary(ablation, rows))
        metadata[ablation] = {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "context_mode": artifact.context_mode,
            "criteria": criteria,
            "thresholds": operating.get("thresholds"),
            "unmatched_penalty": operating.get("unmatched_penalty"),
            "raw_candidate_metrics": candidate_score_metrics(
                bundle.adjacent_sets, raw_scores, args.calibration_bins
            ),
            "frozen_calibrated_candidate_metrics": candidate_score_metrics(
                bundle.adjacent_sets, calibrated, args.calibration_bins
            ),
            "route_trial_evaluations": details,
        }

    _write_csv(output_root / "frozen_v1_route_trial_metrics.csv", trial_rows)
    _write_csv(output_root / "frozen_v1_route_condition_summary.csv", condition_rows)
    _write_csv(output_root / "frozen_v1_route_magnitude_summary.csv", magnitude_rows)
    _write_json(
        output_root / "audit_contract.json",
        {
            "schema_version": "faser-frozen-v1-ift-ry-route-audit-v1",
            "v1_output": str(v1_root),
            "synthetic_manifest": str(manifest_path),
            "loaded_event_splits": ["validation"],
            "forbidden_splits": ["test"],
            "test_events_loaded": False,
            "test_artifacts_opened": False,
            "physical_geometry_repropagation": True,
            "q_over_p_mode": 0,
            "condition_axis": "ift_ry_mrad",
            "candidate_graph": "existing_mode0_acts_physical_candidates_all_six_station_pairs",
            "calibration_refit": False,
            "route_threshold_selection": False,
            "model_training": False,
            "maximum_hypotheses": maximum_hypotheses,
            "ablations": metadata,
        },
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_root),
                "ablations": list(requested),
                "validation_samples": len(samples),
                "test_events_loaded": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
