#!/usr/bin/env python3
"""Workbook-76 K-short 5DoF FD complementarity report driver.

Stages:
  validate-config       load the pre-registered config and verify every
                        inherited SHA256 (no FD bank required);
  canonical-regression  rebuild the canonical-only analysis from the frozen
                        hierarchical V1 bank and check it reproduces the
                        workbook-73 frozen artifacts (negative control);
  all                   full campaign: physical closure, canonical
                        regression, K-short-only, joint complementarity,
                        and the pre-registered pass/fail decision.

The long K-short physical FD reconstruction runs on HTCondor before the
``all`` stage; this driver never submits reconstruction jobs.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from alignment.kshort_rigid_station_5dof_complementarity import (
    DEFAULT_CONFIG_RELATIVE,
    build_all_reports,
    canonical_regression,
    load_canonical_banks,
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
        "n_canonical_sources": len(config["canonical_corpus"]["train_source_ids"])
        + len(config["canonical_corpus"]["validation_source_ids"]),
        "n_kshort_sources": len(config["kshort_corpus"]["train_source_ids"])
        + len(config["kshort_corpus"]["validation_source_ids"]),
        "physical_event_identity": dict(config["physical_event_identity"]),
        "complementarity_contract": dict(config["complementarity_contract"]),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_RELATIVE),
        help="pre-registered campaign config",
    )
    parser.add_argument(
        "--stage",
        choices=("validate-config", "canonical-regression", "all"),
        default="validate-config",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = resolve_under_root(project_root(), str(config["output_dir"]))

    if args.stage == "validate-config":
        payload = stage_validate_config(config)
        path = output_dir / "config_validation.json"
        sha = _write_json(path, payload)
        print(json.dumps({"config_validation": str(path), "sha256": sha}, indent=2))
        return

    if args.stage == "canonical-regression":
        canonical_config, banks = load_canonical_banks(config)
        regression = canonical_regression(
            config, canonical_config=canonical_config, banks=banks
        )
        payload = {key: value for key, value in regression.items() if key != "reports"}
        path = output_dir / "canonical_regression.json"
        sha = _write_json(path, payload)
        print(
            json.dumps(
                {
                    "canonical_regression": str(path),
                    "sha256": sha,
                    "pass": bool(regression["pass"]),
                },
                indent=2,
            )
        )
        return

    reports = build_all_reports(config)
    outputs = {
        "physical_closure": reports["physical_closure"],
        "canonical_regression": reports["canonical_regression"],
        "canonical_only": reports["canonical_only"],
        "kshort_only": reports["kshort_only"],
        "joint": reports["joint"],
        "complementarity": reports["complementarity"],
        "next_stage_decision": reports["next_stage_decision"],
    }
    summary = {"schema_version": reports["schema_version"], "created_utc": reports["created_utc"]}
    written = {}
    for name, payload in outputs.items():
        path = output_dir / f"{name}.json"
        written[name] = {"path": str(path), "sha256": _write_json(path, payload)}
    summary["artifacts"] = written
    summary["decision"] = reports["next_stage_decision"]["decision"]
    summary["complementarity_pass"] = bool(
        reports["next_stage_decision"].get("complementarity_pass")
    )
    summary_path = output_dir / "campaign_summary.json"
    summary["campaign_summary_sha256"] = _write_json(summary_path, summary)
    print(json.dumps({"campaign_summary": str(summary_path), "decision": summary["decision"]}, indent=2))


if __name__ == "__main__":
    main()
