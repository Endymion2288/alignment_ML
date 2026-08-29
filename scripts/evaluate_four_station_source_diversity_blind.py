#!/usr/bin/env python3
"""Workbook 64 reserved-blind gate for source-disjoint training diversity.

After these numbers are opened, training, objective, and operating
convention stay frozen.  Workbook-56 transfer and workbook 53–55
development are not this gate.
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
from training.source_diversity_audit import RESERVED_BLIND_SOURCES


def classify_blind_decision(
    *,
    passed: bool,
    layer1_ok: bool,
    same_common_se3_truth_utility_drop: bool,
) -> dict[str, object]:
    if passed:
        return {
            "passed": True,
            "coverage_class": "training_domain_coverage_limitation_resolved_by_source_diversity",
            "continue_to_15d_relative_wls": True,
            "stop_v2_mainline": False,
            "do_not_add_more_same_family_sources": False,
            "do_not_reweight_or_retune_operating_point": True,
            "next": "open_15d_route_selected_delta_t_ij_wls",
        }
    if layer1_ok and same_common_se3_truth_utility_drop:
        return {
            "passed": False,
            "coverage_class": "source_diversity_insufficient_for_gauge_transfer",
            "continue_to_15d_relative_wls": False,
            "stop_v2_mainline": True,
            "do_not_add_more_same_family_sources": True,
            "do_not_reweight_or_retune_operating_point": True,
            "next": "stop_v2_mainline_and_discuss_architecture_level_relative_gauge_equivariant_representation",
        }
    return {
        "passed": False,
        "coverage_class": "source_diversity_blind_failed_other",
        "continue_to_15d_relative_wls": False,
        "stop_v2_mainline": True,
        "do_not_add_more_same_family_sources": True,
        "do_not_reweight_or_retune_operating_point": True,
        "next": "stop_v2_mainline_and_discuss_architecture_level_relative_gauge_equivariant_representation",
    }


def _twin_truth_utility_drop(mechanism: Mapping[str, Any] | None, families: Mapping[str, tuple[str, str]]) -> dict[str, object]:
    if mechanism is None:
        return {"ok_to_call_same_mode": False, "families": {}, "systematic": False}
    rows = {}
    drops = 0
    for family, (chart, control) in families.items():
        chart_block = ((mechanism.get("payloads") or {}).get(chart) or {}).get("feasibility") or {}
        twin_block = ((mechanism.get("payloads") or {}).get(control) or {}).get("feasibility") or {}
        chart_u = chart_block.get("u_truth_median")
        if chart_u is None:
            chart_u = ((chart_block.get("quantiles") or {}).get("u_truth") or {}).get("p50")
        twin_u = twin_block.get("u_truth_median")
        if twin_u is None:
            twin_u = ((twin_block.get("quantiles") or {}).get("u_truth") or {}).get("p50")
        drop = None if chart_u is None or twin_u is None else float(twin_u) < float(chart_u)
        if drop:
            drops += 1
        rows[family] = {
            "chart": chart,
            "control": control,
            "chart_u_truth_median": None if chart_u is None else float(chart_u),
            "twin_u_truth_median": None if twin_u is None else float(twin_u),
            "truth_utility_drop": drop,
        }
    return {
        "families": rows,
        "n_families": int(len(families)),
        "n_drops": int(drops),
        "systematic": bool(families) and drops >= max(1, len(families) - 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gates", default="configs/physical_four_station_diversity_training.yaml")
    parser.add_argument("--iteration-manifest", required=True)
    parser.add_argument("--candidate-diagnostic-json", action="append", required=True)
    parser.add_argument("--mechanism-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--split", default="validation")
    args = parser.parse_args()
    if args.split == "test":
        raise SystemExit("refusing to open the sealed test split")
    gates_path = Path(args.gates).expanduser().resolve()
    contract = yaml.safe_load(gates_path.read_text(encoding="utf-8"))
    if contract.get("control_id") != "retrained_v2_source_disjoint_diversity_v1":
        raise SystemExit("gates file is not the workbook-64 source-diversity contract")
    if contract.get("test_data_accessed") is not False:
        raise SystemExit("contract opened sealed test")
    if contract.get("workbook56_transfer_is_final_gate") is not False:
        raise SystemExit("workbook-56 transfer must not be the final gate")
    candidate_paths = [Path(item).expanduser().resolve() for item in args.candidate_diagnostic_json]
    _refuse_forbidden_inputs(candidate_paths + [gates_path], list(FORBIDDEN_PAYLOADS))
    candidate_reports = _load_reports(candidate_paths)
    association_gates = _gates(contract["gates"])
    association_gates["raw_complete_truth_chain_recall_min"] = float(
        contract["gates"]["raw_complete_truth_chain_recall_min"]
    )
    candidate = assess(candidate_reports, association_gates)
    recovery = _adjacent_recovery(
        candidate_reports, float(association_gates["vs_nominal_efficiency_drop_max"])
    )
    iteration = _read_json(Path(args.iteration_manifest).expanduser().resolve())
    sources = {str(item.get("source_id")) for item in iteration.get("sources") or []}
    if sources != set(RESERVED_BLIND_SOURCES):
        raise SystemExit("blind gate iteration is not the reserved pair")
    if iteration.get("reserved_blind_validation_only") is not True:
        raise SystemExit("iteration manifest is not reserved-blind-only")
    plan = iteration["common_scan_plan"]
    families = _families_from_plan(plan)
    gauge = _gauge_audit(candidate_reports, families, contract["gates"]["gauge_invariance_audit"])
    nominal = _absolute_nominal(
        candidate_reports[NOMINAL],
        float(contract["gates"]["nominal_complete_track_purity_min"]),
        float(contract["gates"]["nominal_track_fake_rate_max"]),
    )
    pairs = _pair_and_s3(candidate_reports)
    mechanism = _read_json(Path(args.mechanism_json))
    utility = _twin_truth_utility_drop(mechanism, families)
    layer1 = bool(candidate["raw_candidate_graph_complete"])
    layer2 = bool(candidate["frozen_v2_association_stable_vs_nominal"] and nominal["ok"] and recovery["ok"])
    layer3 = bool(gauge["ok"])
    passed = layer1 and layer2 and layer3
    plus_common_fail = any(
        (candidate.get("payloads") or {}).get(name, {}).get("association_vs_nominal_ok") is False
        for name in candidate_reports
        if str(name).endswith("_plus_common")
    )
    same_mode = bool(layer1 and (not layer2 or not layer3) and (utility["systematic"] or plus_common_fail))
    decision = classify_blind_decision(
        passed=passed,
        layer1_ok=layer1,
        same_common_se3_truth_utility_drop=same_mode,
    )
    payload = {
        "control_id": contract["control_id"],
        "workbook": 64,
        "split": args.split,
        "reserved_blind_sources": sorted(sources),
        "compare_against": "this_bank_nominal",
        "workbook56_transfer_is_final_gate": False,
        "test_data_accessed": False,
        "do_not_retune_after_seeing_blind": True,
        "layer1_raw_candidate": layer1,
        "layer2_association_vs_own_nominal": layer2,
        "layer3_gauge": layer3,
        "passed": passed,
        "candidate": candidate,
        "nominal": nominal,
        "adjacent_23_and_s3": recovery,
        "pair_and_s3": pairs,
        "gauge": gauge,
        "mechanism": {
            "u_truth_median": mechanism.get("u_truth_median"),
            "u_best_fragment_median": mechanism.get("u_best_fragment_median"),
            "production_admitted_fragment_wins": mechanism.get("production_admitted_fragment_wins"),
            "dustbin_winner_missed_by_training_miner": mechanism.get("dustbin_winner_missed_by_training_miner"),
            "feasibility_family_counts": mechanism.get("feasibility_family_counts"),
            "twin_truth_utility": utility,
        },
        **decision,
    }
    output = Path(args.output_json).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(_json_ready({"output": str(output), **decision, "passed": passed}), indent=2))


if __name__ == "__main__":
    main()
