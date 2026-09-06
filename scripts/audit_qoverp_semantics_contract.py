#!/usr/bin/env python3
"""Task A: inventory existing q/p exports.  No new campaign, no alignment."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

from alignment.cad_survey_nov22 import git_head_sha
from alignment.operating_protocol_v1_final_closure import (
    project_root,
    resolve_under_root,
    sha256_file,
)
from datasets.qoverp_semantics import (
    build_candidates,
    calypso_provenance,
    decide,
    inherit_frozen_stage,
    inventory_split,
    load_config,
    state_definition,
)
from evaluation.artifact_store import ImmutableArtifactStore


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/qoverp_semantics_contract_v1.yaml")
    args = parser.parse_args()
    config_path = resolve_under_root(project_root(), args.config)
    config = load_config(config_path)
    inherited = inherit_frozen_stage(config)
    construction = inventory_split(config, "construction")
    validation = inventory_split(config, "validation")
    candidates = build_candidates(construction, validation)
    decision = decide(candidates)
    provenance = calypso_provenance(config)
    store = ImmutableArtifactStore.begin(
        resolve_under_root(project_root(), str(config["output_root"])),
        "sba_qoverp_semantics",
    )
    created = datetime.now(timezone.utc).isoformat()
    config_sha = sha256_file(config_path)
    store.write_json(
        "state_definition.json",
        {
            "kind": "qoverp_state_definition",
            "created_utc": created,
            "config_sha256": config_sha,
            "git_head_sha": git_head_sha(),
            **state_definition(),
        },
    )
    store.write_json(
        "source_inventory.json",
        {
            "kind": "qoverp_source_inventory",
            "validation_split": "WB87 construction / validation, file-level disjoint",
            "new_source_campaign_started": False,
            "eighteen_source_jobs_submitted": False,
            "construction": construction,
            "validation": validation,
        },
    )
    store.write_json(
        "dummy_segmentfit_audit.json",
        {
            "kind": "dummy_segmentfit_audit",
            "physical": False,
            "qoverp_per_mev": 1.0e-5,
            "variance_per_mev2": 5.0e-6,
            "correlation_with_loc_and_angles": 0.0,
            "construction_all_dummy": construction["all_tracklet_qoverp_dummy"],
            "validation_all_dummy": validation["all_tracklet_qoverp_dummy"],
            "source": "SegmentFitAlg::GetState",
        },
    )
    store.write_json("calypso_provenance.json", provenance)
    store.write_json("inherited_stage.json", {"kind": "inherited_stage", **inherited})
    store.write_json("qoverp_semantics_contract.json", decision)
    store.finalize(
        {
            "verdict": decision["verdict"],
            "task": "SB-A",
            "workbook": 95,
            "decision": decision["decision"],
            "config_sha256": config_sha,
        }
    )
    print(
        f"SB-A verdict={decision['verdict']} decision={decision['decision']} "
        f"run_id={store.run_id}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
