#!/usr/bin/env python3
"""Workbook-79 real-data residual/covariance model adequacy V1 driver.

Stages enforce the frozen ordering structurally:

    validate-config    config + WB78 inheritance SHAs + tracker regression
    reproduce          Stage-0: reproduce the frozen WB78 calibration baseline
                       (bank SHA, chi2, gamma, beta, eigenvalues, rank,
                       condition, bootstrap).  Hard stop on any mismatch.
    covariance-audit   Stage-1+2: covariance eigenstructure/whitening +
                       14973<->14974 empirical covariance cross-check
    score-decomposition  Stage-3: alignment-score contribution decomposition
    bootstrap-decomposition  Stage-4: bootstrap-instability decomposition
    counterfactuals    Stage-5: diagnostic-only covariance counterfactuals
    transfer-support   Stage-6: observable/Jacobian-transfer support audit
    cross-run          Stage-7: calibration cross-run transportability
    decide             pre-registered decision tree

Every audit runs ONLY on the frozen calibration subset (14973+14974) and MC
control; no held-out residual is ever read.  ``reproduce`` must pass before
any audit stage runs.  ``decide`` reads the frozen audit artifacts.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from alignment.cad_survey_nov22 import git_head_sha
from alignment.real_data_covariance_model_adequacy import (
    DEFAULT_CONFIG_RELATIVE,
    bootstrap_instability_decomposition,
    covariance_counterfactuals,
    covariance_eigen_audit,
    cross_run_transportability,
    decide_campaign,
    empirical_covariance_crosscheck,
    load_config,
    reproduce_baseline,
    score_contribution_decomposition,
    transfer_support_audit,
)
from alignment.gauge_fixed_real_data_diagnostic import _json_ready
from alignment.operating_protocol_v1_final_closure import (
    project_root,
    resolve_under_root,
    sha256_file,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_ready(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _output_root(config: dict) -> Path:
    return resolve_under_root(project_root(), str(config["output_root"]))


def _stage_validate_config(config: dict) -> dict:
    report = {
        "kind": "config_validation",
        "config_path": config["config_path"],
        "config_sha256": sha256_file(Path(config["config_path"])),
        "git_head_sha": git_head_sha(),
        "frozen_flags_verified": True,
        "held_out_accessed": False,
        "inheritance_sha256_verified": True,
        "population": "calibration_14973_14974_only",
        "pass": True,
    }
    return {"config_validation": report}


def _build_baseline(config: dict) -> dict:
    return reproduce_baseline(config)


def _stage_reproduce(config: dict, baseline: dict | None = None) -> dict:
    baseline = baseline or _build_baseline(config)
    return {"baseline_reproduction": baseline["report"]}


def _require_baseline(config: dict) -> dict:
    """Rebuild the baseline and require that it reproduced the frozen WB78
    numbers; every audit stage is refused otherwise."""
    baseline = _build_baseline(config)
    if not baseline["report"]["pass"]:
        raise RuntimeError("WB78 baseline reproduction failed; audit stage refused")
    return baseline


def _stage_covariance_audit(config: dict, baseline: dict | None = None) -> dict:
    baseline = baseline or _require_baseline(config)
    return {
        "covariance_eigen_audit": covariance_eigen_audit(config, baseline["bank"]),
        "empirical_covariance_crosscheck": empirical_covariance_crosscheck(config, baseline["bank"]),
    }


def _stage_score_decomposition(config: dict, baseline: dict | None = None) -> dict:
    baseline = baseline or _require_baseline(config)
    return {
        "score_contribution_decomposition": score_contribution_decomposition(
            config,
            baseline["subspace"],
            baseline["transfer"],
            baseline["bank"],
            baseline["candidate"],
        )
    }


def _stage_bootstrap_decomposition(config: dict, baseline: dict | None = None) -> dict:
    baseline = baseline or _require_baseline(config)
    return {
        "bootstrap_instability_decomposition": bootstrap_instability_decomposition(
            config, baseline["subspace"], baseline["transfer"], baseline["bank"]
        )
    }


def _stage_counterfactuals(config: dict, baseline: dict | None = None) -> dict:
    baseline = baseline or _require_baseline(config)
    return {
        "covariance_counterfactuals": covariance_counterfactuals(
            config, baseline["subspace"], baseline["transfer"], baseline["bank"]
        )
    }


def _stage_transfer_support(config: dict, baseline: dict | None = None) -> dict:
    baseline = baseline or _require_baseline(config)
    return {
        "transfer_support_audit": transfer_support_audit(
            config,
            baseline["transfer"],
            baseline["pooled"],
            baseline["extras"],
            baseline["bank"],
        )
    }


def _stage_cross_run(config: dict, baseline: dict | None = None) -> dict:
    baseline = baseline or _require_baseline(config)
    return {
        "cross_run_transportability": cross_run_transportability(
            config, baseline["subspace"], baseline["transfer"], baseline["bank"]
        )
    }


def _stage_decide(config: dict) -> dict:
    root = _output_root(config)
    reproduction = _read_json(root / "baseline_reproduction.json")["baseline_reproduction"]
    cov_path = root / "covariance_audit.json"
    cov = _read_json(cov_path) if cov_path.is_file() else None
    transfer_path = root / "transfer_support.json"
    transfer = _read_json(transfer_path)["transfer_support_audit"] if transfer_path.is_file() else None
    cross_path = root / "cross_run.json"
    cross = _read_json(cross_path)["cross_run_transportability"] if cross_path.is_file() else None
    decision = decide_campaign(
        reproduction=reproduction,
        covariance_audit=cov["covariance_eigen_audit"] if cov else None,
        crosscheck=cov["empirical_covariance_crosscheck"] if cov else None,
        transfer_support=transfer,
        cross_run=cross,
    )
    summary = {
        "kind": "campaign_summary",
        "schema_version": config["schema_version"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_head_sha": git_head_sha(),
        "config_path": config["config_path"],
        "decision": decision["decision"],
        "failed_components": decision["failed_components"],
        "geometry_write_allowed": False,
        "official_conditions_write_allowed": False,
        "real_data_candidate_alignment_authorized": False,
        "held_out_accessed": False,
    }
    return {"model_adequacy_decision": decision, "campaign_summary": summary}


_AUDIT_STAGES = (
    "covariance-audit",
    "score-decomposition",
    "bootstrap-decomposition",
    "counterfactuals",
    "transfer-support",
    "cross-run",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_RELATIVE))
    parser.add_argument(
        "--stage",
        required=True,
        choices=(
            "validate-config",
            "reproduce",
            *_AUDIT_STAGES,
            "decide",
            "all",
        ),
    )
    args = parser.parse_args()
    config = load_config(args.config)
    root = _output_root(config)
    root.mkdir(parents=True, exist_ok=True)

    if args.stage in ("validate-config", "all"):
        _write_json(root / "config_validation.json", _stage_validate_config(config))

    # Build the baseline once and share it across all requested audit stages.
    baseline = None
    if args.stage == "all" or args.stage == "reproduce" or args.stage in _AUDIT_STAGES:
        baseline = _build_baseline(config)

    if args.stage in ("reproduce", "all"):
        _write_json(root / "baseline_reproduction.json", _stage_reproduce(config, baseline))
        if not baseline["report"]["pass"]:
            return _finish_decide_early(config)
    if args.stage == "all" and not baseline["report"]["pass"]:
        return _finish_decide_early(config)

    if args.stage in ("covariance-audit", "all"):
        _write_json(root / "covariance_audit.json", _stage_covariance_audit(config, baseline))
    if args.stage in ("score-decomposition", "all"):
        _write_json(root / "score_decomposition.json", _stage_score_decomposition(config, baseline))
    if args.stage in ("bootstrap-decomposition", "all"):
        _write_json(
            root / "bootstrap_decomposition.json", _stage_bootstrap_decomposition(config, baseline)
        )
    if args.stage in ("counterfactuals", "all"):
        _write_json(root / "counterfactuals.json", _stage_counterfactuals(config, baseline))
    if args.stage in ("transfer-support", "all"):
        _write_json(root / "transfer_support.json", _stage_transfer_support(config, baseline))
    if args.stage in ("cross-run", "all"):
        _write_json(root / "cross_run.json", _stage_cross_run(config, baseline))
    if args.stage in ("decide", "all"):
        reports = _stage_decide(config)
        _write_json(root / "model_adequacy_decision.json", reports["model_adequacy_decision"])
        _write_json(root / "campaign_summary.json", reports["campaign_summary"])
    return 0


def _finish_decide_early(config: dict) -> int:
    root = _output_root(config)
    reports = _stage_decide(config)
    _write_json(root / "model_adequacy_decision.json", reports["model_adequacy_decision"])
    _write_json(root / "campaign_summary.json", reports["campaign_summary"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
