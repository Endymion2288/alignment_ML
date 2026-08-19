#!/usr/bin/env python3
"""Consolidated verdict for the mode-3 matched-retraining validation.

Compares the strict matched control (mode0-data + mode0-model versus
mode3-data + mode3-model) plus the domain-shift control
(mode3-data + frozen-mode0-model) across every predeclared metric family:

* candidate truth-chain recall (coverage audits on the training corpus);
* edge AP/AUC/ECE per model family (MLP + route, V1 full-context,
  V1 no-context, V2 BCE route-query);
* complete-track efficiency/purity/fake from the frozen-backbone route
  evaluation on the iteration-1 validation overlay;
* score/rank of the known bad 0->1 mismatch copies;
* source-wise stability of the route metrics;
* dx/dy/Ry route-selected closure under physical_edge_deduplicated semantics;
* intrinsic chi2 separation and same-edge pull sanity (diagnostic arm).

Promotion requires all of: (1) mode-3 retrained models recover nominal
primary validation metrics; (2) mode-3 route metrics are not worse than
mode-0; (3) mode-3 keeps the more physical covariance/pull behaviour and
multi-DoF closure; (4) the known mis-associated edge or source-dependent
outliers actually improve.  Otherwise mode-3 remains a diagnostic variant.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _read_json(path: Path) -> dict[str, Any]:
    with Path(path).expanduser().resolve().open(encoding="utf-8") as handle:
        return json.load(handle)


def _edge_metrics(payload: Mapping[str, Any], *, family: str) -> dict[str, float | None]:
    """Pull overall edge AP/AUC/ECE from a model family's validation record."""
    if family == "mlp":
        gate = payload.get("candidate_gate_selection", {})
        point = payload.get("primary_validation_operating_point", {})
        return {
            "average_precision": _float(gate.get("average_precision")),
            "roc_auc": _float(gate.get("roc_auc")),
            "expected_calibration_error": _float(gate.get("expected_calibration_error")),
            "association_efficiency": _float(point.get("association_efficiency")),
            "association_purity": _float(point.get("association_purity")),
            "fake_rate": _float(point.get("fake_rate")),
        }
    if family in ("v1", "v1full", "v1nocontext"):
        calibrated = payload.get("calibrated_candidate", {})
        by_magnitude = payload.get("route_selection", {}).get("evaluation_by_magnitude", {})
        nominal = by_magnitude.get("0.0", {})
        route = nominal.get("route", {})
        return {
            "average_precision": _float(calibrated.get("average_precision")),
            "roc_auc": _float(calibrated.get("roc_auc")),
            "expected_calibration_error": _float(calibrated.get("expected_calibration_error")),
            "nominal_complete_track_efficiency": _float(route.get("complete_track_efficiency")),
            "nominal_complete_track_purity": _float(route.get("complete_track_purity")),
            "nominal_track_fake_rate": _float(route.get("track_fake_rate")),
        }
    if family == "v2":
        streams = payload.get("score_streams", {}).get("route_aware_v2", {})
        after = streams.get("score_stream_calibration", {}).get("after", {})
        return {
            "average_precision": _float(after.get("average_precision")),
            "roc_auc": _float(after.get("roc_auc")),
            "expected_calibration_error": _float(after.get("expected_calibration_error")),
        }
    raise ValueError(f"unknown model family: {family}")


def _float(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _route_metrics(summary: Mapping[str, Any]) -> dict[str, float | None]:
    route = summary.get("truth_labelled_mc_evaluation", {}).get("route", {})
    raw = summary.get("raw_candidate_score_metrics", {})
    calibrated = summary.get("frozen_calibrated_score_metrics", {})
    return {
        "complete_track_efficiency": _float(route.get("complete_track_efficiency")),
        "complete_track_purity": _float(route.get("complete_track_purity")),
        "track_fake_rate": _float(route.get("track_fake_rate")),
        "candidate_complete_truth_chain_recall": _float(route.get("candidate_complete_truth_chain_recall")),
        "raw_average_precision": _float(raw.get("average_precision")),
        "raw_roc_auc": _float(raw.get("roc_auc")),
        "calibrated_average_precision": _float(calibrated.get("average_precision")),
        "calibrated_roc_auc": _float(calibrated.get("roc_auc")),
        "calibrated_expected_calibration_error": _float(calibrated.get("expected_calibration_error")),
    }


def _closure_parameters(payload: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for parameter in payload.get("parameters", []):
        name = str(parameter.get("name"))
        raw_delta = parameter.get("local_delta", parameter.get("local_delta_error", parameter.get("error")))
        out[name] = {
            "local_delta": _float(raw_delta),
            "expected_delta_to_target": _float(parameter.get("expected_delta_to_target")),
            "capture_success": bool(parameter.get("capture_success")),
            "capture_tolerance": _float(parameter.get("capture_tolerance")),
        }
    out["normal_matrix_condition_number"] = _float(payload.get("normal_matrix_condition_number"))
    out["capture_success"] = bool(payload.get("capture_success"))
    return out


def _bad_edge_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    copies = payload.get("copies", [])
    scores = [_float(copy.get("calibrated_score")) for copy in copies]
    ranks = [_float(copy.get("rank_among_event_0_to_1_candidates")) for copy in copies]
    return {
        "copies_found": int(payload.get("copies_found", 0)),
        "median_calibrated_score": _median(scores),
        "median_rank": _median(ranks),
        "copies": copies,
    }


def _median(values: list[float | None]) -> float | None:
    clean = sorted(value for value in values if value is not None)
    if not clean:
        return None
    middle = len(clean) // 2
    if len(clean) % 2:
        return float(clean[middle])
    return float(0.5 * (clean[middle - 1] + clean[middle]))


def _source_wise_routes(anchor_dir: Path) -> dict[str, Any]:
    """Per-origin-run route statistics from a backbone anchor output.

    A route is truth-consistent when every endpoint shares one physical origin
    (origin_run_id, origin_event_id); mixed routes combine endpoints from
    different origins.  This is the per-source stability view of the frozen
    route metrics.
    """
    path = anchor_dir / "selected_routes.jsonl"
    if not path.is_file():
        return {"available": False}
    by_source: dict[str, dict[str, int]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            signature = json.loads(row.get("route_origin_signature", "[]"))
            origins = {
                (int(item.get("origin_run_id", -1)), int(item.get("origin_event_id", -1)))
                for item in signature
            }
            run_ids = {origin[0] for origin in origins}
            consistent = len(origins) == 1 and -1 not in run_ids
            complete = bool(row.get("is_complete_four_station_route"))
            for run_id in run_ids:
                stats = by_source.setdefault(
                    str(run_id), {"routes": 0, "truth_consistent": 0, "complete": 0}
                )
                stats["routes"] += 1
                stats["truth_consistent"] += int(consistent)
                stats["complete"] += int(complete)
    per_source = []
    for run_id, stats in sorted(by_source.items()):
        routes = max(stats["routes"], 1)
        per_source.append(
            {
                "origin_run_id": int(run_id),
                "routes": stats["routes"],
                "truth_consistent_fraction": stats["truth_consistent"] / routes,
                "complete_route_fraction": stats["complete"] / routes,
            }
        )
    consistency = [source["truth_consistent_fraction"] for source in per_source]
    return {
        "available": True,
        "sources": len(per_source),
        "min_truth_consistent_fraction": min(consistency) if consistency else None,
        "max_truth_consistent_fraction": max(consistency) if consistency else None,
        "per_source": per_source,
    }


def _delta(mode0: float | None, mode3: float | None) -> float | None:
    if mode0 is None or mode3 is None:
        return None
    return mode3 - mode0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coverage-mode0", required=True)
    parser.add_argument("--coverage-mode3", required=True)
    parser.add_argument("--model-metrics", action="append", required=True, metavar="FAMILY:MODE:PATH")
    parser.add_argument("--backbone-anchor", action="append", required=True, metavar="ARM:PATH")
    parser.add_argument("--bad-edge", action="append", required=True, metavar="ARM:PATH")
    parser.add_argument("--closure-mode0", required=True)
    parser.add_argument("--closure-mode3", required=True)
    parser.add_argument("--intrinsic-separation", default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    coverage0 = _read_json(args.coverage_mode0)
    coverage3 = _read_json(args.coverage_mode3)

    model_metrics: dict[str, dict[str, Any]] = {}
    for spec in args.model_metrics:
        family, mode, path = spec.split(":", 2)
        model_metrics.setdefault(family, {})[mode] = _edge_metrics(_read_json(Path(path)), family=family)

    backbone: dict[str, Any] = {}
    for spec in args.backbone_anchor:
        arm, path = spec.split(":", 1)
        summary_path = Path(path)
        metrics = _route_metrics(_read_json(summary_path))
        if metrics is not None:
            metrics["source_wise"] = _source_wise_routes(summary_path.parent)
        backbone[arm] = metrics

    bad_edge: dict[str, Any] = {}
    for spec in args.bad_edge:
        arm, path = spec.split(":", 1)
        bad_edge[arm] = _bad_edge_summary(_read_json(Path(path)))

    closure0 = _closure_parameters(_read_json(args.closure_mode0))
    closure3 = _closure_parameters(_read_json(args.closure_mode3))

    intrinsic = None if args.intrinsic_separation is None else _read_json(args.intrinsic_separation)

    # ---- promotion criteria -------------------------------------------------
    criteria: dict[str, Any] = {}

    primary_recovery: dict[str, Any] = {}
    for family, arms in sorted(model_metrics.items()):
        mode0 = arms.get("mode0", {})
        mode3 = arms.get("mode3", {})
        primary_recovery[family] = {
            "mode0": mode0,
            "mode3": mode3,
            "delta_average_precision": _delta(mode0.get("average_precision"), mode3.get("average_precision")),
            "delta_roc_auc": _delta(mode0.get("roc_auc"), mode3.get("roc_auc")),
            "delta_expected_calibration_error": _delta(
                mode0.get("expected_calibration_error"), mode3.get("expected_calibration_error")
            ),
        }
    criteria["primary_metric_recovery"] = primary_recovery

    route_arm0 = backbone.get("mode0_data_mode0_model", {})
    route_arm3 = backbone.get("mode3_data_mode3_model", {})
    route_shift = backbone.get("mode3_data_frozen_mode0_model", {})
    criteria["route_metrics"] = {
        "mode0_data_mode0_model": route_arm0,
        "mode3_data_mode3_model": route_arm3,
        "mode3_data_frozen_mode0_model": route_shift,
        "delta_complete_track_efficiency": _delta(
            route_arm0.get("complete_track_efficiency"), route_arm3.get("complete_track_efficiency")
        ),
        "delta_complete_track_purity": _delta(
            route_arm0.get("complete_track_purity"), route_arm3.get("complete_track_purity")
        ),
        "delta_track_fake_rate": _delta(route_arm0.get("track_fake_rate"), route_arm3.get("track_fake_rate")),
        "domain_shift_efficiency_collapse": _delta(
            route_arm0.get("complete_track_efficiency"), route_shift.get("complete_track_efficiency")
        ),
    }

    criteria["candidate_coverage"] = {
        "mode0_overall_truth_pair_recall": coverage0.get("overall", {}).get("raw_physical_candidate_truth_recall"),
        "mode3_overall_truth_pair_recall": coverage3.get("overall", {}).get("raw_physical_candidate_truth_recall"),
        "mode0_candidate_eligible_truth_pairs": coverage0.get("overall", {}).get("field_candidate_eligible_truth_pairs"),
        "mode3_candidate_eligible_truth_pairs": coverage3.get("overall", {}).get("field_candidate_eligible_truth_pairs"),
        "mode0_acts_propagation_failed": coverage0.get("overall", {}).get("eligibility_reasons", {}).get("acts_propagation_failed"),
        "mode3_acts_propagation_failed": coverage3.get("overall", {}).get("eligibility_reasons", {}).get("acts_propagation_failed"),
    }

    mode0_source_min = (route_arm0.get("source_wise") or {}).get("min_truth_consistent_fraction")
    mode3_source_min = (route_arm3.get("source_wise") or {}).get("min_truth_consistent_fraction")
    criteria["bad_edge"] = {
        "mode0_data_mode0_model": bad_edge.get("mode0_data_mode0_model"),
        "mode3_data_mode3_model": bad_edge.get("mode3_data_mode3_model"),
        "mode3_data_frozen_mode0_model": bad_edge.get("mode3_data_frozen_mode0_model"),
        "mode0_min_source_truth_consistent_fraction": mode0_source_min,
        "mode3_min_source_truth_consistent_fraction": mode3_source_min,
        "source_wise_min_consistency_gain": _delta(mode0_source_min, mode3_source_min),
        "semantics": (
            "The six audited copies ARE the pilot-identified bad edge itself "
            "(origin event 9000000300/12, tracklet 0@st0 -> 1@st1, residual_x "
            "-150 mm, chi2 12.8 under mode-0, route-selected 6/6 in both "
            "retrained arms). The is_truth=true field is the MC label "
            "(truth_particle_id 10001 on both endpoints); the pilot mechanism "
            "analysis established this edge is kinematically poison and must "
            "be demoted. A HIGHER calibrated score or BETTER rank therefore "
            "means the bad edge is MORE favoured (regression), not an "
            "improvement of mis-association handling."
        ),
    }

    criteria["closure"] = {"mode0": closure0, "mode3": closure3}
    source_wise = {
        arm: (metrics or {}).get("source_wise") for arm, metrics in backbone.items()
    }
    if any(value is not None and value.get("available") for value in source_wise.values()):
        criteria["source_wise_stability"] = source_wise
    if intrinsic is not None:
        criteria["intrinsic_separation"] = intrinsic

    verdict = {
        "schema_version": "faser-mode3-matched-retraining-verdict-v1",
        "criteria": criteria,
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(verdict, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
