#!/usr/bin/env python3
"""Transfer gate for workbook-59 dustbin-aware route-margin V2.

After these numbers are opened, objective / margin / weights / threshold /
penalty / calibration stay frozen.  Sealed test is never opened.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from scripts.evaluate_four_station_gauge_consistent_transfer import (
    NOMINAL,
    _absolute_nominal,
    _pair_and_s3,
)
from scripts.evaluate_four_station_matched_association import (
    FORBIDDEN_PAYLOADS,
    _adjacent_recovery,
    _families_from_plan,
    _gauge_audit,
    _gates,
    _load_reports,
    _refuse_forbidden_inputs,
)
from scripts.run_refit_multidof_closure import _json_ready, _read_json
from scripts.summarize_four_station_frozen_association import assess


def _not_worse(new: Mapping[str, Any], control: Mapping[str, Any], key: str) -> dict[str, object]:
    new_median = new.get(key)
    control_median = control.get(key)
    if new_median is None or control_median is None:
        return {"ok": False, "ratio": None, "new_median": new_median, "control_median": control_median}
    new_value = float(new_median)
    control_value = float(control_median)
    ratio = None if control_value <= 0.0 else new_value / control_value
    return {
        "ok": new_value <= control_value,
        "ratio": ratio,
        "new_median": new_value,
        "control_median": control_value,
    }


def _mechanism_next(mechanism: Mapping[str, Any] | None, layer2_ok: bool) -> tuple[str | None, str | None]:
    if mechanism is None or layer2_ok:
        return None, None
    u_frac = mechanism.get("u_truth_fraction_nonpositive")
    admitted = int(mechanism.get("production_admitted_fragment_wins") or 0)
    selected = int(mechanism.get("selected") or 0)
    if u_frac is not None and float(u_frac) <= 0.50 and admitted > selected:
        return (
            "association_loses_to_fragment_after_dustbin_crossed",
            "diagnose_solver_generated_hard_negative_mining",
        )
    if u_frac is not None and float(u_frac) >= 0.50:
        return (
            "utility_margin_still_fails",
            "route_objective_still_misses_production_decision_boundary",
        )
    return None, None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gates", default="configs/physical_four_station_dustbin_aware_route_training.yaml")
    parser.add_argument("--iteration-manifest", required=True)
    parser.add_argument("--candidate-diagnostic-json", action="append", required=True)
    parser.add_argument("--control-diagnostic-json", action="append", required=True)
    parser.add_argument("--candidate-score-scale-json", required=True)
    parser.add_argument("--control-score-scale-json", required=True)
    parser.add_argument("--mechanism-json", default=None)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--split", default="validation")
    args = parser.parse_args()
    if args.split == "test":
        raise SystemExit("refusing to open the sealed test split")
    gates_path = Path(args.gates).expanduser().resolve()
    contract = yaml.safe_load(gates_path.read_text(encoding="utf-8"))
    control_id = contract.get("control_id")
    if control_id not in {
        "retrained_v2_dustbin_aware_route_margin_v1",
        "retrained_v2_hard_aware_max_reduction_v1",
    }:
        raise SystemExit("gates file is not a pre-registered dustbin-aware / max-reduction contract")
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
    logit_shift = _not_worse(
        candidate_scale["pooled_matched_origin_raw_logit_abs_shift"],
        control_scale["pooled_matched_origin_raw_logit_abs_shift"],
        "median",
    )
    utility_shift = _not_worse(
        candidate_scale["pooled_matched_complete_route_utility_abs_shift"],
        control_scale["pooled_matched_complete_route_utility_abs_shift"],
        "median",
    )
    mechanism = None if args.mechanism_json is None else _read_json(Path(args.mechanism_json))
    layer1 = bool(candidate["raw_candidate_graph_complete"])
    layer2 = bool(candidate["frozen_v2_association_stable_vs_nominal"] and nominal["ok"] and recovery["ok"])
    layer3 = bool(gauge["ok"] and logit_shift["ok"] and utility_shift["ok"])
    passed = layer1 and layer2 and layer3
    if control_id == "retrained_v2_hard_aware_max_reduction_v1":
        if not layer1:
            failure = "candidate_or_propagation"
            next_step = "diagnose_candidate_graph"
        elif passed:
            failure = None
            next_step = "open_15d_route_selected_delta_t_ij_wls"
        else:
            failure = "association_or_gauge_after_hard_aware_max_reduction"
            next_step = "stop_weighting_and_operating_point_rescue"
    else:
        mechanism_failure, mechanism_next = _mechanism_next(mechanism, layer2)
        if not layer1:
            failure = "candidate_or_propagation"
            next_step = "diagnose_candidate_graph"
        elif not layer2 and mechanism_next is not None:
            failure = mechanism_failure
            next_step = mechanism_next
        elif not layer2:
            failure = "association_domain_shift"
            next_step = "diagnose_association_after_dustbin_aware_objective"
        elif not layer3:
            failure = "gauge_invariance_or_score_scale_worse_than_workbook_54"
            next_step = "diagnose_gauge_twin_after_dustbin_aware_objective"
        else:
            failure = None
            next_step = "open_15d_route_selected_delta_t_ij_wls"
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
            "require_not_worse_than_workbook_54": True,
        },
        "mechanism": mechanism,
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
    print(json.dumps(_json_ready({
        "output_json": str(output),
        "continue_to_15d_relative_wls": passed,
        "failure_class": failure,
        "next_step": next_step,
        "layer1_ok": layer1,
        "layer2_ok": layer2,
        "layer3_ok": layer3,
    }), indent=2))


if __name__ == "__main__":
    main()
