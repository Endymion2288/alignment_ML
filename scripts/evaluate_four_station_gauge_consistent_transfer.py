#!/usr/bin/env python3
"""Three-layer transfer-validation gate for workbook-56 gauge-consistent V2.

Only the new source-disjoint transfer overlay may claim the production
association gate.  After these numbers are opened, the objective, margin,
loss weights, threshold, unmatched penalty, and calibration stay frozen.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from scripts.evaluate_four_station_matched_association import (
    FORBIDDEN_PAYLOADS,
    _adjacent_recovery,
    _families_from_plan,
    _gauge_audit,
    _gates,
    _load_reports,
    _pair_efficiency,
    _refuse_forbidden_inputs,
)
from scripts.run_refit_multidof_closure import _json_ready, _read_json
from scripts.summarize_four_station_frozen_association import assess


NOMINAL = "iteration_00_reference"
PAIRS = ("0->1", "1->2", "2->3")


def _absolute_nominal(report: Mapping[str, Any], purity_min: float, fake_max: float) -> dict[str, object]:
    selected = report["selected_route"]
    purity = selected.get("complete_track_purity")
    fake = selected.get("track_fake_rate")
    if purity is None or fake is None:
        return {
            "complete_track_purity": purity,
            "track_fake_rate": fake,
            "purity_ok": False,
            "fake_ok": False,
            "ok": False,
            "no_selected_tracks": True,
        }
    purity_value = float(purity)
    fake_value = float(fake)
    return {
        "complete_track_purity": purity_value,
        "track_fake_rate": fake_value,
        "purity_ok": purity_value >= purity_min,
        "fake_ok": fake_value <= fake_max,
        "ok": purity_value >= purity_min and fake_value <= fake_max,
        "no_selected_tracks": False,
    }


def _pair_and_s3(reports: Mapping[str, Mapping[str, Any]]) -> dict[str, object]:
    rows = {}
    for name, report in reports.items():
        selected = report["selected_physical_edges"]
        rows[name] = {
            "efficiency_0_1": _pair_efficiency(report, "0->1"),
            "efficiency_1_2": _pair_efficiency(report, "1->2"),
            "efficiency_2_3": _pair_efficiency(report, "2->3"),
            "s3_selected_edge_endpoints": int(
                (selected.get("station_participation_edge_endpoints") or {}).get("3") or 0
            ),
        }
    return rows


def _ratio_ok(new: Mapping[str, Any], control: Mapping[str, Any], key: str, limit: float) -> dict[str, object]:
    new_median = new.get(key)
    control_median = control.get(key)
    if new_median is None or control_median is None:
        return {"ok": False, "ratio": None, "new_median": new_median, "control_median": control_median}
    control_value = float(control_median)
    new_value = float(new_median)
    if control_value <= 0.0:
        ok = new_value == 0.0
        ratio = 0.0 if new_value == 0.0 else None
    else:
        ratio = new_value / control_value
        ok = ratio <= float(limit) and new_value < control_value
    return {
        "ok": ok,
        "ratio": ratio,
        "new_median": new_value,
        "control_median": control_value,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gates", default="configs/physical_four_station_gauge_consistent_route_training.yaml")
    parser.add_argument("--iteration-manifest", required=True)
    parser.add_argument("--candidate-diagnostic-json", action="append", required=True)
    parser.add_argument("--control-diagnostic-json", action="append", required=True)
    parser.add_argument("--candidate-score-scale-json", required=True)
    parser.add_argument("--control-score-scale-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--split", default="validation")
    args = parser.parse_args()
    if args.split == "test":
        raise SystemExit("refusing to open the sealed test split")
    gates_path = Path(args.gates).expanduser().resolve()
    contract = yaml.safe_load(gates_path.read_text(encoding="utf-8"))
    if contract.get("control_id") != "retrained_v2_gauge_consistent_route_v1":
        raise SystemExit("gates file is not the workbook-56 objective contract")
    if contract.get("test_data_accessed") is not False:
        raise SystemExit("contract opened sealed test")
    candidate_paths = [Path(item).expanduser().resolve() for item in args.candidate_diagnostic_json]
    control_paths = [Path(item).expanduser().resolve() for item in args.control_diagnostic_json]
    _refuse_forbidden_inputs(candidate_paths + control_paths + [gates_path], list(FORBIDDEN_PAYLOADS))
    candidate_reports = _load_reports(candidate_paths)
    control_reports = _load_reports(control_paths)
    association_gates = _gates(contract["gates"])
    association_gates["raw_complete_truth_chain_recall_min"] = float(
        contract["gates"]["raw_complete_truth_chain_recall_min"]
    )
    candidate = assess(candidate_reports, association_gates)
    control = assess(control_reports, association_gates)
    recovery = _adjacent_recovery(
        candidate_reports, float(association_gates["vs_nominal_efficiency_drop_max"])
    )
    plan = _read_json(Path(args.iteration_manifest).expanduser().resolve())["common_scan_plan"]
    gauge = _gauge_audit(candidate_reports, _families_from_plan(plan), contract["gates"]["gauge_invariance_audit"])
    nominal = _absolute_nominal(
        candidate_reports[NOMINAL],
        float(contract["gates"]["nominal_complete_track_purity_min"]),
        float(contract["gates"]["nominal_track_fake_rate_max"]),
    )
    candidate_scale = _read_json(Path(args.candidate_score_scale_json))
    control_scale = _read_json(Path(args.control_score_scale_json))
    limits = contract["gates"]["score_scale"]
    logit_shift = _ratio_ok(
        candidate_scale["pooled_matched_origin_raw_logit_abs_shift"],
        control_scale["pooled_matched_origin_raw_logit_abs_shift"],
        "median",
        float(limits["matched_origin_raw_logit_median_abs_shift_ratio_max"]),
    )
    utility_shift = _ratio_ok(
        candidate_scale["pooled_matched_complete_route_utility_abs_shift"],
        control_scale["pooled_matched_complete_route_utility_abs_shift"],
        "median",
        float(limits["matched_origin_truth_route_utility_median_abs_shift_ratio_max"]),
    )
    layer1 = bool(candidate["raw_candidate_graph_complete"])
    layer2 = bool(candidate["frozen_v2_association_stable_vs_nominal"] and nominal["ok"] and recovery["ok"])
    layer3 = bool(gauge["ok"] and logit_shift["ok"] and utility_shift["ok"])
    passed = layer1 and layer2 and layer3
    if not layer1:
        failure = "candidate_or_propagation"
        next_step = "diagnose_candidate_graph"
    elif not logit_shift["ok"]:
        failure = "gauge_score_scale_drift_remains"
        next_step = contract["if_gauge_twin_raw_logit_drift_remains"]["next"]
    elif not layer2 and gauge["ok"] and logit_shift["ok"]:
        failure = "association_domain_shift"
        next_step = contract["if_association_fails_and_gauge_score_drift_gone"]["next"]
    elif not layer2:
        failure = "association_domain_shift"
        next_step = "diagnose_packing_route_competition_objective"
    elif not gauge["ok"]:
        failure = "gauge_invariance_failure"
        next_step = "diagnose_gauge_route_metrics"
    else:
        failure = None
        next_step = "open_15d_route_selected_delta_t_ij_wls" if passed else None
    decision = {
        "split": args.split,
        "test_data_accessed": False,
        "development_validation_used": False,
        "continue_to_15d_relative_wls": passed,
        "failure_class": failure,
        "next_step": next_step,
        "layer1_raw_candidate_recall": {"ok": layer1, "payloads": candidate["payloads"]},
        "layer2_association": {
            "ok": layer2,
            "nominal_absolute": nominal,
            "vs_nominal": candidate["frozen_v2_association_stable_vs_nominal"],
            "adjacent_pairs_and_s3": _pair_and_s3(candidate_reports),
            "adjacent_2_3_and_s3_recovered": recovery,
        },
        "layer3_gauge": {
            "ok": layer3,
            "route_metrics": gauge,
            "raw_logit_shift_vs_workbook_54": logit_shift,
            "truth_route_utility_shift_vs_workbook_54": utility_shift,
        },
        "workbook_54_control_on_same_transfer_overlay": {
            "raw_candidate_graph_complete": control["raw_candidate_graph_complete"],
            "association_stable_vs_nominal": control["frozen_v2_association_stable_vs_nominal"],
            "failure_class": control["failure_class"],
        },
        "unmatched_penalty_rescue_used": False,
        "second_platt_after_training_used": False,
        "architecture_change_used": False,
        "capture_contract_if_opened": "workbook_49_50_frozen",
    }
    output = Path(args.output_json).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_json_ready(decision), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            _json_ready(
                {
                    "output_json": str(output),
                    "continue_to_15d_relative_wls": passed,
                    "failure_class": failure,
                    "next_step": next_step,
                    "layer1_ok": layer1,
                    "layer2_ok": layer2,
                    "layer3_ok": layer3,
                }
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
