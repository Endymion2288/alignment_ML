#!/usr/bin/env python3
"""Task B4 audit.  Analyze frozen WB99 tails without repairing closure."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

from alignment.cad_survey_nov22 import git_head_sha
from alignment.operating_protocol_v1_final_closure import (
    project_root,
    resolve_under_root,
    sha256_file,
)
from datasets.acts_transport_diagnosis import jsonable
from datasets.acts_transport_tail_analysis import (
    decide,
    inherit_frozen_stage,
    inventory_and_analyze,
    load_config,
)
from evaluation.artifact_store import ImmutableArtifactStore


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/acts_transport_tail_analysis_v1.yaml")
    args = parser.parse_args()
    config_path = resolve_under_root(project_root(), args.config)
    config = load_config(config_path)
    inherited = inherit_frozen_stage(config)
    inventory = inventory_and_analyze(config)
    decision = decide(inventory, inherited)
    store = ImmutableArtifactStore.begin(
        resolve_under_root(project_root(), str(config["output_root"])),
        "sbb4_acts_transport_tail_analysis",
    )
    created = datetime.now(timezone.utc).isoformat()
    config_sha = sha256_file(config_path)
    store.write_json(
        "tail_failure_classification.json",
        jsonable(
            {
                "kind": "tail_failure_classification",
                "created_utc": created,
                "config_sha256": config_sha,
                **inventory["tail_failure_classification"],
            }
        ),
    )
    store.write_json(
        "reconstruction_quality.json",
        jsonable(
            {
                "kind": "reconstruction_quality",
                "created_utc": created,
                "config_sha256": config_sha,
                **inventory["reconstruction_quality"],
            }
        ),
    )
    store.write_json(
        "pull_distribution.json",
        jsonable(
            {
                "kind": "pull_distribution",
                "created_utc": created,
                "config_sha256": config_sha,
                **inventory["pull_distribution"],
            }
        ),
    )
    store.write_json(
        "q_eigenvalue_diagnostic.json",
        jsonable(
            {
                "kind": "q_eigenvalue_diagnostic",
                "created_utc": created,
                "config_sha256": config_sha,
                **inventory["q_eigenvalue_diagnostic"],
            }
        ),
    )
    store.write_json("inherited_stage.json", {"kind": "inherited_stage", **inherited})
    store.write_json(
        "acts_transport_tail_analysis_contract.json",
        jsonable(
            {
                "created_utc": created,
                "config_sha256": config_sha,
                "git_head_sha": git_head_sha(),
                "n_dump_rows": inventory["n_dump_rows"],
                "n_matched_events": inventory["n_matched_events"],
                "n_official_pair_events": inventory["n_official_pair_events"],
                "present_sources": inventory["present_sources"],
                "missing_sources": inventory["missing_sources"],
                **decision,
            }
        ),
    )
    store.finalize(
        {
            "verdict": decision["verdict"],
            "task": "SB-B4",
            "workbook": 100,
            "decision": decision["decision"],
            "primary_case": decision["primary_case"],
            "next_step": decision["next_step"],
            "config_sha256": config_sha,
            "inherited_wb99_decision_sha256": decision["inherited_wb99_decision_sha256"],
            "inherited_wb99_tail_sha256": decision["inherited_wb99_tail_sha256"],
        }
    )
    print(
        f"SB-B4 verdict={decision['verdict']} case={decision['primary_case']} "
        f"next={decision['next_step']} run_id={store.run_id}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
