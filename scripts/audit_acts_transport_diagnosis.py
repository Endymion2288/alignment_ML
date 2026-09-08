#!/usr/bin/env python3
"""Task B3 audit.  Diagnose WB98 closure failure without repairing it."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

from alignment.cad_survey_nov22 import git_head_sha
from alignment.operating_protocol_v1_final_closure import (
    project_root,
    resolve_under_root,
    sha256_file,
)
from datasets.acts_transport_diagnosis import (
    decide,
    inherit_frozen_stage,
    inventory_and_diagnose,
    jsonable,
    load_config,
)
from evaluation.artifact_store import ImmutableArtifactStore


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/acts_transport_diagnosis_v1.yaml")
    args = parser.parse_args()
    config_path = resolve_under_root(project_root(), args.config)
    config = load_config(config_path)
    inherited = inherit_frozen_stage(config)
    inventory = inventory_and_diagnose(config)
    decision = decide(inventory, inherited)
    store = ImmutableArtifactStore.begin(
        resolve_under_root(project_root(), str(config["output_root"])),
        "sbb3_acts_transport_diagnosis",
    )
    created = datetime.now(timezone.utc).isoformat()
    config_sha = sha256_file(config_path)
    store.write_json(
        "state_surface_contract.json",
        jsonable(
            {
                "kind": "state_surface_contract",
                "created_utc": created,
                "config_sha256": config_sha,
                **inventory["state_surface_contract"],
            }
        ),
    )
    store.write_json(
        "jacobian_covariance_consistency.json",
        jsonable(
            {
                "kind": "jacobian_covariance_consistency",
                "created_utc": created,
                "config_sha256": config_sha,
                **inventory["jacobian_covariance_consistency"],
            }
        ),
    )
    store.write_json(
        "residual_decomposition.json",
        jsonable(
            {
                "kind": "residual_decomposition",
                "created_utc": created,
                "config_sha256": config_sha,
                **inventory["residual_decomposition"],
            }
        ),
    )
    store.write_json(
        "tail_provenance.json",
        jsonable(
            {
                "kind": "tail_provenance",
                "created_utc": created,
                "config_sha256": config_sha,
                **inventory["tail_provenance"],
            }
        ),
    )
    store.write_json("inherited_stage.json", {"kind": "inherited_stage", **inherited})
    store.write_json(
        "acts_transport_diagnosis_contract.json",
        jsonable(
            {
                "created_utc": created,
                "config_sha256": config_sha,
                "git_head_sha": git_head_sha(),
                "n_dump_rows": inventory["n_dump_rows"],
                "n_matched_events": inventory["n_matched_events"],
                "n_official_pair_events": inventory["n_official_pair_events"],
                "n_off_contract_pair_events": inventory["n_off_contract_pair_events"],
                "present_sources": inventory["present_sources"],
                "missing_sources": inventory["missing_sources"],
                **decision,
            }
        ),
    )
    store.finalize(
        {
            "verdict": decision["verdict"],
            "task": "SB-B3",
            "workbook": 99,
            "decision": decision["decision"],
            "primary_case": decision["primary_case"],
            "next_step": decision["next_step"],
            "config_sha256": config_sha,
        }
    )
    print(
        f"SB-B3 verdict={decision['verdict']} case={decision['primary_case']} "
        f"next={decision['next_step']} run_id={store.run_id}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
