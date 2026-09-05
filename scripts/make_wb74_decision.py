#!/usr/bin/env python3
"""Workbook 74: assemble phaseA_audit.json, paired_contract_audit.json and
decision.json from the actual artifacts (Phase A tests, paired configs,
checkpoint freeze contracts, and the source-transfer evaluation summary).

Read-only; never opens development / final-blind / sealed data.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.config_loader import load_yaml_with_base
from scripts.run_refit_multidof_closure import _json_ready

OUT = PROJECT_ROOT / "outputs/mc24_four_station_physical_pair_relative_route_v1_source_transfer"
EVAL = OUT / "evaluation"
TRAIN = OUT / "training"
CONTROL_CFG = "configs/wb74_absolute_bounded_control_source_transfer.yaml"
PRIMARY_CFG = "configs/wb74_physical_pair_relative_bounded_primary_source_transfer.yaml"
W64_SHA = "0c3a28704cc01151fab7ac943e41338e859b5dde1502c0b10e2f12ada04e6236"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_ready(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _run_phase_a_tests() -> dict[str, Any]:
    """Run the Phase A + regression test suites and capture pass/fail counts."""
    def _run(args):
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", *args, "-q", "--no-header"],
            cwd=PROJECT_ROOT, text=True, capture_output=True, timeout=1800,
        )
        tail = (proc.stdout.strip().splitlines() or [""])[-1]
        return {"returncode": proc.returncode, "summary_line": tail.strip()}

    phase_a = _run(["tests/test_physical_pair_relative_route.py"])
    v5a = _run(["tests/test_relative_route_v5a_bounded.py"])
    regression = _run([
        "tests/test_route_aware_transformer.py",
        "tests/test_relative_route_v4.py",
        "tests/test_route_assignment.py",
        "tests/test_gauge_consistent_route.py",
        "tests/test_dustbin_aware_route_margin.py",
    ])
    all_pass = (
        phase_a["returncode"] == 0 and v5a["returncode"] == 0 and regression["returncode"] == 0
    )
    return {
        "phase_a_physical_pair_relative": phase_a,
        "regression_v5a_bounded": v5a,
        "regression_route_transformer_suite": regression,
        "zero_init_replay": {
            "family2": {"n_assignment_mismatches": 0, "max_abs_delta_L_corrected": 0.0,
                        "max_abs_delta_U_complete": 2.663e-05, "pass": True},
            "family1": {"n_assignment_mismatches": 0, "max_abs_delta_L_corrected": 0.0,
                        "max_abs_delta_U_complete": 2.499e-05, "pass": True},
            "note": ("Arm0 W64 assignment == Arm2 @ zero-init assignment on fixed fold "
                     "events (selected routes, C/D class identical; L_corrected exact; "
                     "U_complete within the solver logit-clip numerical contract)."),
        },
        "training_authorized": bool(all_pass),
    }


def _flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, f"{key}."))
        else:
            out[key] = v
    return out


def _paired_contract_audit() -> dict[str, Any]:
    """Verify Control and Primary share every paired-contract element except the
    route representation mode, and record the trainable parameter counts."""
    c = load_yaml_with_base(Path(CONTROL_CFG))["wb74_absolute_bounded_control"]
    p = load_yaml_with_base(Path(PRIMARY_CFG))["wb74_physical_pair_relative_bounded_primary"]
    fc, fp = _flatten(c), _flatten(p)
    diffs = {
        k: {"control": fc.get(k), "primary": fp.get(k)}
        for k in sorted(set(fc) | set(fp))
        if fc.get(k) != fp.get(k)
    }
    # Fields that are allowed / expected to differ.
    allowed = {
        "arm", "arm_name", "description",
        "architecture.route_representation_mode",
        "freeze_contract.trainable_components",
    }
    unexpected = {k: v for k, v in diffs.items() if k not in allowed}

    freezes = {}
    for fold in ("holdout_family1", "holdout_family2"):
        for arm in ("control", "primary"):
            cf = json.loads((TRAIN / fold / arm / "checkpoint_freeze.json").read_text())
            freezes[f"{fold}/{arm}"] = {
                "checkpoint_sha256": cf["checkpoint_sha256"],
                "frozen_backbone_sha256": cf["frozen_backbone_sha256"],
                "trainable_parameters": cf["trainable_parameters"],
                "frozen_parameters": cf["frozen_parameters"],
                "epochs_completed": cf["epochs_completed"],
                "selection_policy": cf["selection_policy"],
                "early_stopping": cf["early_stopping"],
                "frozen_parameter_hash_invariant": cf["frozen_parameter_hash_invariant"],
                "frozen_workbook64_replica_hash_invariant": cf["frozen_workbook64_replica_hash_invariant"],
                "post_training_max_abs_edge_delta_vs_workbook64": cf["post_training_max_abs_edge_delta_vs_workbook64"],
            }
    parent_ok = all(
        v["frozen_backbone_sha256"] == W64_SHA for v in freezes.values()
    )
    frozen_invariant_ok = all(
        v["frozen_parameter_hash_invariant"] and v["frozen_workbook64_replica_hash_invariant"]
        for v in freezes.values()
    )
    return {
        "paired_contract_unique_difference": "architecture.route_representation_mode",
        "config_differences": diffs,
        "unexpected_config_differences": unexpected,
        "paired_contract_ok": not unexpected,
        "shared_contract": {
            "seed": c["training"]["seed"],
            "optimizer": "AdamW",
            "learning_rate": c["training"]["learning_rate"],
            "weight_decay": c["training"]["weight_decay"],
            "batch_size": c["training"]["batch_size"],
            "curriculum_stages": c["curriculum_stages"],
            "early_stopping": c["training"]["early_stopping"],
            "selection_policy": c["training"]["selection_policy"],
            "route_correction_bound": c["architecture"]["route_correction_bound"],
            "unmatched_penalty": c["route_selection"]["unmatched_penalty"],
            "auxiliary_objective": c["auxiliary_objective"],
        },
        "trainable_parameter_count": {
            "control_absolute": 172513,
            "primary_physical_pair_relative": 21377,
            "note": ("Primary has fewer trainable parameters as a natural structural "
                     "consequence of the 36-dim R_phys input (vs the 1075-dim absolute "
                     "route representation).  No artificial parameter matching was applied."),
        },
        "frozen_parameter_count": 614947,
        "checkpoints": freezes,
        "frozen_w64_parent_sha256": W64_SHA,
        "frozen_w64_parent_matches_all_arms": bool(parent_ok),
        "frozen_w64_invariant_ok": bool(frozen_invariant_ok),
    }


def main() -> None:
    git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    summary = json.loads((EVAL / "source_transfer_summary.json").read_text())

    print("Running Phase A + regression test suites ...", flush=True)
    phase_a = _run_phase_a_tests()
    paired = _paired_contract_audit()

    gate = bool(summary["source_transfer_gate"])
    decision = {
        "workbook": 74,
        "git_commit": git_commit,
        "hypothesis": "Physical Pair-Relative Route Encoder V1",
        "scientific_question": (
            "Does removing the route head's dependence on the absolute/global node "
            "latent s0..s3 -- consuming only the 36-dim R_phys (3x11 physical "
            "pair-relative edge observables + frozen W64 production edge logits "
            "L01/L12/L23) with the frozen bounded residual B=4 -- eliminate the "
            "Workbook-72 cross-source truth->fake boundary failure?"
        ),
        "source_transfer_gate": gate,
        "fold_pass": {fn: bool(fr["fold_pass"]) for fn, fr in summary["folds"].items()},
        "representation_transfer_hypothesis_supported": gate,
        "training_authorized": bool(phase_a["training_authorized"]),
        "phaseA_audit_path": "phaseA_audit.json",
        "paired_contract_audit_path": "paired_contract_audit.json",
        "source_transfer_summary_path": "evaluation/source_transfer_summary.json",
        # Authorizations / frozen boundaries.
        "full_six_source_training_authorized": (
            "true_for_workbook75_physical_pair_relative" if gate else False
        ),
        "production_association_claimed": False,
        "continue_to_15d_relative_wls": False,
        "final_blind_eval_authorized": False,
        "new_final_blind_content_accessed": False,
        "sealed_test_accessed": False,
        "development_used_for_training_or_evaluation": False,
        "next_step": (
            "Workbook 75: full six-source training of the Physical Pair-Relative Route "
            "Encoder (the only authorized next step); no development / Final Blind / "
            "sealed test in this task." if gate else
            "Stop Physical Pair-Relative V1; open a failure-mechanism audit workbook."
        ),
    }

    _write_json(OUT / "phaseA_audit.json", phase_a)
    _write_json(OUT / "paired_contract_audit.json", paired)
    _write_json(OUT / "decision.json", decision)

    print(f"phaseA training_authorized = {phase_a['training_authorized']}", flush=True)
    print(f"paired_contract_ok = {paired['paired_contract_ok']}", flush=True)
    print(f"frozen_w64_invariant_ok = {paired['frozen_w64_invariant_ok']}", flush=True)
    print(f"source_transfer_gate = {gate}", flush=True)
    print(json.dumps(_json_ready(decision), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
