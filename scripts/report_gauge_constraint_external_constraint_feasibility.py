#!/usr/bin/env python3
"""Workbook-77 gauge-constrained / external-constraint feasibility driver.

Stages:
  validate-config   load the pre-registered config and verify every
                    inherited SHA256 (no tracker bank required);
  all               full feasibility campaign: rebuild the frozen
                    workbook-68 tracker information with regression,
                    build the candidate gauge constraints, run the
                    pre-registered gauge-invariant observable closure,
                    evaluate the external-constraint eligibility table,
                    and freeze the decision.

This driver never runs reconstruction, never touches real data, never
writes geometry or official conditions, and never consults the
workbook-76 K-short FD spectrum.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from alignment.gauge_constraint_feasibility import (
    DEFAULT_CONFIG_RELATIVE,
    build_all_reports,
    load_config,
)
from alignment.module_level_residual_poc import json_ready
from alignment.operating_protocol_v1_final_closure import (
    project_root,
    resolve_under_root,
    sha256_file,
)


def _strip_private(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _strip_private(item)
            for key, item in value.items()
            if not str(key).startswith("_")
        }
    if isinstance(value, (list, tuple)):
        return [_strip_private(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    ready = json_ready(_strip_private(payload))
    path.write_text(json.dumps(ready, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return sha256_file(path)


def stage_validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "stage": "validate-config",
        "config_path": config["config_path"],
        "config_sha256": sha256_file(Path(config["config_path"])),
        "schema_version": config["schema_version"],
        "inheritance_sha256_verified": True,
        "n_gauge_candidates": len(config["gauge_candidates"]),
        "n_eligibility_audit_rows": len(
            config["external_constraint_eligibility"]["audit_rows"]
        ),
        "parameter_names": list(config["parameter_space"]["parameter_names"]),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_RELATIVE),
        help="pre-registered campaign config",
    )
    parser.add_argument("--stage", choices=("validate-config", "all"), default="validate-config")
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = resolve_under_root(project_root(), str(config["output_dir"]))

    if args.stage == "validate-config":
        payload = stage_validate_config(config)
        path = output_dir / "config_validation.json"
        sha = _write_json(path, payload)
        print(json.dumps({"config_validation": str(path), "sha256": sha}, indent=2))
        return

    reports = build_all_reports(config)
    outputs = {
        "tracker_information_regression": reports["tracker_information_regression"],
        "gauge_candidates": {"candidates": reports["gauge_candidates"]},
        "gauge_invariant_closure": reports["gauge_invariant_closure"],
        "external_constraint_eligibility": reports["external_constraint_eligibility"],
        "next_stage_decision": reports["next_stage_decision"],
    }
    summary = {
        "schema_version": reports["schema_version"],
        "created_utc": reports["created_utc"],
        "git_head_sha": reports["git_head_sha"],
        "config_path": reports["config_path"],
    }
    written = {}
    for name, payload in outputs.items():
        path = output_dir / f"{name}.json"
        written[name] = {"path": str(path), "sha256": _write_json(path, payload)}
    summary["artifacts"] = written
    summary["decision"] = reports["next_stage_decision"]["decision"]
    summary_path = output_dir / "campaign_summary.json"
    summary["campaign_summary_sha256"] = _write_json(summary_path, summary)
    print(
        json.dumps(
            {"campaign_summary": str(summary_path), "decision": summary["decision"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
