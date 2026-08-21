#!/usr/bin/env python3
"""Matched association comparison for the four-station relative retraining pilot.

Validation-only.  Workbook-52 held-out overlays are refused as operating-point
inputs.  15-DoF unknown-association WLS is not opened here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from scripts.run_refit_multidof_closure import _json_ready, _read_json
from scripts.summarize_four_station_frozen_association import assess


NOMINAL = "iteration_00_reference"
FORBIDDEN_PAYLOADS = {
    "iteration_00_closure_relative",
    "iteration_00_closure_relative_plus_common",
}


def _gates(raw: Mapping[str, Any]) -> dict[str, float]:
    def _require(*keys: str) -> float:
        for key in keys:
            if key in raw:
                return float(raw[key])
        raise KeyError("missing association gate " + " / ".join(keys))

    return {
        "raw_complete_truth_chain_recall_min": _require("raw_complete_truth_chain_recall_min"),
        "raw_adjacent_truth_edge_recall_min": _require(
            "adjacent_truth_edge_recall_min", "raw_adjacent_truth_edge_recall_min"
        ),
        "vs_nominal_efficiency_drop_max": _require(
            "complete_track_efficiency_drop_vs_nominal_max", "vs_nominal_efficiency_drop_max"
        ),
        "vs_nominal_purity_drop_max": _require(
            "complete_track_purity_drop_vs_nominal_max", "vs_nominal_purity_drop_max"
        ),
        "vs_nominal_fake_rate_increase_max": _require(
            "track_fake_rate_increase_vs_nominal_max", "vs_nominal_fake_rate_increase_max"
        ),
    }


def _pair_efficiency(report: Mapping[str, Any], pair: str) -> float | None:
    block = (report.get("by_adjacent_station_pair") or {}).get(pair) or {}
    value = (block.get("association") or {}).get("association_efficiency")
    return None if value is None else float(value)


def _refuse_forbidden_inputs(paths: list[Path], forbidden: list[str]) -> None:
    resolved = [str(path.resolve()) for path in paths]
    for item in forbidden:
        needle = str(Path(item))
        for observed in resolved:
            if needle in observed:
                raise SystemExit(f"refusing workbook-52 operating-point input: {observed}")


def _load_reports(paths: list[Path]) -> dict[str, dict[str, Any]]:
    reports: dict[str, dict[str, Any]] = {}
    for path in paths:
        report = _read_json(path)
        payload = str(report["payload_id"])
        if payload in FORBIDDEN_PAYLOADS:
            raise SystemExit(f"refusing workbook-52 held-out payload '{payload}' for operating-point selection")
        reports[payload] = report
    return reports


def _adjacent_recovery(reports: Mapping[str, Mapping[str, Any]], max_drop: float) -> dict[str, object]:
    nominal_23 = _pair_efficiency(reports[NOMINAL], "2->3")
    rows = {}
    ok = nominal_23 is not None
    for name, report in reports.items():
        selected = report["selected_physical_edges"]
        efficiency = _pair_efficiency(report, "2->3")
        drop = None if name == NOMINAL or efficiency is None or nominal_23 is None else float(nominal_23) - float(efficiency)
        s3 = int((selected.get("station_participation_edge_endpoints") or {}).get("3") or 0)
        payload_ok = bool(selected.get("includes_adjacent_2_3")) and s3 > 0
        if name != NOMINAL:
            payload_ok = payload_ok and drop is not None and drop <= max_drop
        ok = ok and payload_ok
        rows[name] = {
            "association_efficiency_2_3": efficiency,
            "efficiency_2_3_drop_vs_nominal": drop,
            "includes_adjacent_2_3": bool(selected.get("includes_adjacent_2_3")),
            "s3_selected_edge_endpoints": s3,
            "recovered": payload_ok,
        }
    return {"ok": ok, "payloads": rows, "nominal_association_efficiency_2_3": nominal_23}


def _gauge_audit(
    reports: Mapping[str, Mapping[str, Any]],
    families: Mapping[str, tuple[str, str]],
    limits: Mapping[str, Any],
) -> dict[str, object]:
    rows = {}
    ok = True
    for family, (chart, control) in families.items():
        left = reports[chart]["selected_route"]
        right = reports[control]["selected_route"]
        efficiency_diff = abs(float(left["complete_track_efficiency"]) - float(right["complete_track_efficiency"]))
        purity_diff = abs(float(left["complete_track_purity"]) - float(right["complete_track_purity"]))
        fake_diff = abs(float(left["track_fake_rate"]) - float(right["track_fake_rate"]))
        left_23 = _pair_efficiency(reports[chart], "2->3")
        right_23 = _pair_efficiency(reports[control], "2->3")
        pair_diff = None if left_23 is None or right_23 is None else abs(left_23 - right_23)
        payload_ok = (
            efficiency_diff <= float(limits["complete_track_efficiency_twin_abs_diff_max"])
            and purity_diff <= float(limits["complete_track_purity_twin_abs_diff_max"])
            and fake_diff <= float(limits["track_fake_rate_twin_abs_diff_max"])
            and pair_diff is not None
            and pair_diff <= float(limits["adjacent_23_efficiency_twin_abs_diff_max"])
        )
        ok = ok and payload_ok
        rows[family] = {
            "chart": chart,
            "control": control,
            "complete_track_efficiency_abs_diff": efficiency_diff,
            "complete_track_purity_abs_diff": purity_diff,
            "track_fake_rate_abs_diff": fake_diff,
            "adjacent_23_efficiency_abs_diff": pair_diff,
            "ok": payload_ok,
        }
    return {"ok": ok, "families": rows}


def _families_from_plan(plan: Mapping[str, Any]) -> dict[str, tuple[str, str]]:
    grouped: dict[str, dict[str, str]] = {}
    for point in plan["points"]:
        family = str(point.get("relative_family") or "")
        role = str(point.get("gauge_role") or "")
        name = str(point["name"])
        if family in {"", "nominal"}:
            continue
        grouped.setdefault(family, {})[role] = name
    families = {}
    for family, roles in grouped.items():
        if "s0_sampling_chart" not in roles or "left_se3_control" not in roles:
            raise ValueError(f"relative family '{family}' lacks a gauge-control twin")
        families[family] = (roles["s0_sampling_chart"], roles["left_se3_control"])
    if not families:
        raise ValueError("scan plan has no relative families")
    return families


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gates", default="configs/physical_four_station_association_retraining_gates.yaml")
    parser.add_argument("--iteration-manifest", required=True)
    parser.add_argument("--frozen-diagnostic-json", action="append", required=True)
    parser.add_argument("--retrained-diagnostic-json", action="append", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--split", default="validation")
    args = parser.parse_args()
    if args.split == "test":
        raise SystemExit("refusing to open the sealed test split")
    gates_path = Path(args.gates).expanduser().resolve()
    contract = yaml.safe_load(gates_path.read_text(encoding="utf-8"))
    if contract.get("alignment_formulation") != "four_station_v1":
        raise SystemExit("gates file is not four_station_v1")
    frozen_paths = [Path(item).expanduser().resolve() for item in args.frozen_diagnostic_json]
    retrained_paths = [Path(item).expanduser().resolve() for item in args.retrained_diagnostic_json]
    _refuse_forbidden_inputs(
        frozen_paths + retrained_paths + [gates_path],
        list(contract.get("forbidden_operating_point_inputs") or []),
    )
    frozen_reports = _load_reports(frozen_paths)
    retrained_reports = _load_reports(retrained_paths)
    association_gates = _gates(contract["association_gates"])
    frozen = assess(frozen_reports, association_gates)
    retrained = assess(retrained_reports, association_gates)
    recovery = _adjacent_recovery(
        retrained_reports,
        float(association_gates["vs_nominal_efficiency_drop_max"]),
    )
    plan = _read_json(Path(args.iteration_manifest).expanduser().resolve())["common_scan_plan"]
    gauge = _gauge_audit(retrained_reports, _families_from_plan(plan), contract["gauge_invariance_audit"])
    require_23 = bool(contract["association_gates"].get("require_23_and_s3_recovery", True))
    retrained_ok = bool(
        retrained["raw_candidate_graph_complete"]
        and retrained["frozen_v2_association_stable_vs_nominal"]
        and (recovery["ok"] if require_23 else True)
        and gauge["ok"]
    )
    if not retrained["raw_candidate_graph_complete"]:
        failure = "candidate_or_propagation"
    elif not retrained["frozen_v2_association_stable_vs_nominal"]:
        failure = "association_domain_shift"
    elif require_23 and not recovery["ok"]:
        failure = "adjacent_2_3_or_s3_not_recovered"
    elif not gauge["ok"]:
        failure = "gauge_invariance_failure"
    else:
        failure = None
    representation_gap = bool(
        retrained["raw_candidate_graph_complete"]
        and not retrained_ok
        and recovery["ok"] is False
        and all(
            (report["raw_candidate_graph"]["candidate_complete_truth_chain_recall"] or 0.0) >= 0.999
            for report in retrained_reports.values()
        )
    )
    decision = {
        "split": args.split,
        "test_data_accessed": False,
        "architecture_or_threshold_tuning_on_workbook_52_held_outs": False,
        "continue_to_15d_relative_wls": retrained_ok,
        "failure_class": failure,
        "historical_frozen_v2_control": {
            "raw_candidate_graph_complete": frozen["raw_candidate_graph_complete"],
            "association_stable_vs_nominal": frozen["frozen_v2_association_stable_vs_nominal"],
            "failure_class": frozen["failure_class"],
            "payloads": frozen["payloads"],
        },
        "retrained_v2": {
            "raw_candidate_graph_complete": retrained["raw_candidate_graph_complete"],
            "association_stable_vs_nominal": retrained["frozen_v2_association_stable_vs_nominal"],
            "adjacent_2_3_and_s3_recovered": recovery,
            "gauge_invariance": gauge,
            "payloads": retrained["payloads"],
            "score_threshold_retention": retrained["score_threshold_retention"],
        },
        "scientific_question": (
            "four_station_joint_misalignment_is_historical_v2_domain_shift_if_retrained_passes; "
            "representation_lacks_relative_four_station_inductive_bias_if_2_3_still_drops_at_candidate_recall_one"
        ),
        "representation_gap_suspect": representation_gap,
        "unmatched_penalty_rescue_used": False,
        "gates": contract["association_gates"],
        "gauge_invariance_audit": contract["gauge_invariance_audit"],
    }
    output = Path(args.output_json).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_json_ready(decision), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            _json_ready(
                {
                    "output_json": str(output),
                    "continue_to_15d_relative_wls": retrained_ok,
                    "failure_class": failure,
                    "frozen_control_association_ok": frozen["frozen_v2_association_stable_vs_nominal"],
                    "retrained_association_ok": retrained["frozen_v2_association_stable_vs_nominal"],
                    "adjacent_2_3_and_s3_recovered": recovery["ok"],
                    "gauge_invariance_ok": gauge["ok"],
                    "representation_gap_suspect": representation_gap,
                }
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
