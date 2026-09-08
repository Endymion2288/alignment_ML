#!/usr/bin/env python3
"""Task B5 audit.  CKF tail provenance without repairing closure."""

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
from datasets.ckf_tail_provenance import (
    decide,
    inherit_frozen_stage,
    inventory_and_audit,
    load_config,
)
from evaluation.artifact_store import ImmutableArtifactStore


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/ckf_tail_provenance_v1.yaml")
    args = parser.parse_args()
    config_path = resolve_under_root(project_root(), args.config)
    config = load_config(config_path)
    inherited = inherit_frozen_stage(config)
    inventory = inventory_and_audit(config)
    decision = decide(inventory, inherited)
    store = ImmutableArtifactStore.begin(
        resolve_under_root(project_root(), str(config["output_root"])),
        "sbb5_ckf_tail_provenance",
    )
    created = datetime.now(timezone.utc).isoformat()
    config_sha = sha256_file(config_path)
    store.write_json(
        "ckf_tail_provenance.json",
        jsonable(
            {
                "kind": "ckf_tail_provenance",
                "created_utc": created,
                "config_sha256": config_sha,
                **inventory["ckf_tail_provenance"],
            }
        ),
    )
    store.write_json(
        "qoverp_failure.json",
        jsonable(
            {
                "kind": "qoverp_failure",
                "created_utc": created,
                "config_sha256": config_sha,
                **inventory["qoverp_failure"],
            }
        ),
    )
    store.write_json(
        "association_consistency.json",
        jsonable(
            {
                "kind": "association_consistency",
                "created_utc": created,
                "config_sha256": config_sha,
                **inventory["association"],
            }
        ),
    )
    store.write_json("inherited_stage.json", {"kind": "inherited_stage", **inherited})
    store.write_json(
        "ckf_tail_provenance_contract.json",
        jsonable(
            {
                "created_utc": created,
                "config_sha256": config_sha,
                "git_head_sha": git_head_sha(),
                "present_sources": inventory["present_sources"],
                "missing_sources": inventory["missing_sources"],
                **decision,
            }
        ),
    )
    store.finalize(
        {
            "verdict": decision["verdict"],
            "task": "SB-B5",
            "workbook": 101,
            "decision": decision["decision"],
            "primary_case": decision["primary_case"],
            "next_step": decision["next_step"],
            "config_sha256": config_sha,
            "inherited_wb100_decision_sha256": decision["inherited_wb100_decision_sha256"],
            "inherited_wb100_classification_sha256": decision[
                "inherited_wb100_classification_sha256"
            ],
        }
    )
    print(
        f"SB-B5 verdict={decision['verdict']} case={decision['primary_case']} "
        f"next={decision['next_step']} run_id={store.run_id}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
