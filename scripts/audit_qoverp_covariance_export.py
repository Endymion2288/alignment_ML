#!/usr/bin/env python3
"""Task A2 audit.  Validates dumped CKF 5x5; fails closed if dumps are absent."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

from alignment.cad_survey_nov22 import git_head_sha
from alignment.operating_protocol_v1_final_closure import (
    project_root,
    resolve_under_root,
    sha256_file,
)
from datasets.qoverp_covariance_export import (
    decide,
    inherit_frozen_stage,
    inventory_sources,
    load_config,
    provenance_hashes,
    state_definition,
)
from evaluation.artifact_store import ImmutableArtifactStore


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/qoverp_covariance_export_v1.yaml")
    args = parser.parse_args()
    config_path = resolve_under_root(project_root(), args.config)
    config = load_config(config_path)
    inherited = inherit_frozen_stage(config)
    inventory = inventory_sources(config)
    construction = inventory["splits"]["construction"]
    validation = inventory["splits"]["validation"]
    decision = decide(
        construction,
        validation,
        dumps_materialized=bool(inventory["dumps_materialized"]),
    )
    decision.update(provenance_hashes(config))
    decision["dumps_materialized"] = bool(inventory["dumps_materialized"])
    decision["construction"] = construction
    decision["validation"] = validation
    decision["condor"] = {
        "new_source_campaign": False,
        "nevents_per_source": int(config["nevents"]),
        "n_sources": int(construction["n_sources"]) + int(validation["n_sources"]),
        "dump_root": str(config["dump_root"]),
        "long_jobs_required": "HTCondor",
    }
    store = ImmutableArtifactStore.begin(
        resolve_under_root(project_root(), str(config["output_root"])),
        "sba2_qoverp_covariance",
    )
    created = datetime.now(timezone.utc).isoformat()
    config_sha = sha256_file(config_path)
    store.write_json(
        "state_definition.json",
        {
            "kind": "qoverp_covariance_state_definition",
            "created_utc": created,
            "config_sha256": config_sha,
            "git_head_sha": git_head_sha(),
            **state_definition(),
        },
    )
    store.write_json("source_inventory.json", {"kind": "ckf_covariance_inventory", **inventory})
    store.write_json("inherited_stage.json", {"kind": "inherited_stage", **inherited})
    store.write_json("qoverp_covariance_contract.json", decision)
    store.finalize(
        {
            "verdict": decision["verdict"],
            "task": "SB-A2",
            "workbook": 96,
            "decision": decision["decision"],
            "config_sha256": config_sha,
        }
    )
    print(
        f"SB-A2 verdict={decision['verdict']} decision={decision['decision']} "
        f"run_id={store.run_id}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
