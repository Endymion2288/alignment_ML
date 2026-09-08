#!/usr/bin/env python3
"""Task B6 audit.  CKF reconstruction input contract without repairing closure."""

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
from datasets.ckf_reconstruction_contract import (
    decide,
    inherit_frozen_stage,
    inventory_and_audit,
    load_config,
)
from evaluation.artifact_store import ImmutableArtifactStore


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/ckf_reconstruction_contract_audit_v1.yaml")
    args = parser.parse_args()
    config_path = resolve_under_root(project_root(), args.config)
    config = load_config(config_path)
    inherited = inherit_frozen_stage(config)
    inventory = inventory_and_audit(config)
    decision = decide(inventory, inherited)
    store = ImmutableArtifactStore.begin(
        resolve_under_root(project_root(), str(config["output_root"])),
        "sbb6_ckf_reconstruction_contract",
    )
    created = datetime.now(timezone.utc).isoformat()
    config_sha = sha256_file(config_path)
    store.write_json(
        "ckf_input_scope_audit.json",
        jsonable(
            {
                "kind": "ckf_input_scope_audit",
                "created_utc": created,
                "config_sha256": config_sha,
                **inventory["ckf_input_scope_audit"],
                "tracks": inventory["ckf_input_scope_tracks"],
            }
        ),
    )
    store.write_json(
        "ckf_reconstruction_contract.json",
        jsonable(
            {
                "kind": "ckf_reconstruction_contract",
                "created_utc": created,
                "config_sha256": config_sha,
                **inventory["ckf_reconstruction_contract"],
            }
        ),
    )
    store.write_json(
        "tail_reproduction_audit.json",
        jsonable(
            {
                "kind": "tail_reproduction_audit",
                "created_utc": created,
                "config_sha256": config_sha,
                **inventory["tail_reproduction"],
            }
        ),
    )
    store.write_json(
        "short_track_failure_analysis.json",
        jsonable(
            {
                "kind": "short_track_failure_analysis",
                "created_utc": created,
                "config_sha256": config_sha,
                **inventory["short_track_failure_analysis"],
            }
        ),
    )
    store.write_json("inherited_stage.json", {"kind": "inherited_stage", **inherited})
    store.write_json(
        "ckf_reconstruction_contract_decision.json",
        jsonable(
            {
                "created_utc": created,
                "config_sha256": config_sha,
                "git_head_sha": git_head_sha(),
                "present_sources": inventory["present_sources"],
                "missing_sources": inventory["missing_sources"],
                "n_dump_rows": inventory["n_dump_rows"],
                "n_official_pair_events": inventory["n_official_pair_events"],
                "frozen_1pct_n": inventory["frozen_1pct_n"],
                "frozen_1pct_rescreened": inventory["frozen_1pct_rescreened"],
                **decision,
            }
        ),
    )
    store.finalize(
        {
            "verdict": decision["verdict"],
            "task": "SB-B6",
            "workbook": 102,
            "decision": decision["decision"],
            "primary_case": decision["primary_case"],
            "next_step": decision["next_step"],
            "config_sha256": config_sha,
            "inherited_wb101_decision_sha256": decision["inherited_wb101_decision_sha256"],
            "inherited_wb101_provenance_sha256": decision[
                "inherited_wb101_provenance_sha256"
            ],
            "closure_pass": False,
            "input_filter_implemented": False,
        }
    )
    print(
        f"SB-B6 verdict={decision['verdict']} case={decision['primary_case']} "
        f"next={decision['next_step']} run_id={store.run_id}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
