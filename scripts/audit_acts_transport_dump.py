#!/usr/bin/env python3
"""Task B0 audit.  B1 provenance plus B2 closure on C++ ACTS dumps."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

from alignment.cad_survey_nov22 import git_head_sha
from alignment.operating_protocol_v1_final_closure import (
    project_root,
    resolve_under_root,
    sha256_file,
)
from datasets.acts_process_noise_contract import audit_process_noise_configuration
from datasets.acts_transport_dump import (
    decide,
    inherit_frozen_stage,
    inventory_sources,
    load_config,
    plugin_library_path,
)
from evaluation.artifact_store import ImmutableArtifactStore


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/acts_transport_dump_v1.yaml")
    args = parser.parse_args()
    config_path = resolve_under_root(project_root(), args.config)
    config = load_config(config_path)
    inherited = inherit_frozen_stage(config)
    configuration = audit_process_noise_configuration(config)
    inventory = inventory_sources(config)
    decision = decide(configuration, inventory)
    store = ImmutableArtifactStore.begin(
        resolve_under_root(project_root(), str(config["output_root"])),
        "sbb0_acts_transport_dump",
    )
    created = datetime.now(timezone.utc).isoformat()
    config_sha = sha256_file(config_path)
    library = plugin_library_path(config)
    store.write_json(
        "acts_transport_dump_configuration.json",
        {
            "created_utc": created,
            "config_sha256": config_sha,
            "git_head_sha": git_head_sha(),
            "plugin_path": str(library),
            "plugin_sha256": sha256_file(library) if library.is_file() else None,
            **configuration,
        },
    )
    store.write_json("source_inventory.json", {"kind": "acts_transport_dump_inventory", **inventory})
    store.write_json("inherited_stage.json", {"kind": "inherited_stage", **inherited})
    store.write_json("acts_transport_dump_contract.json", decision)
    store.finalize(
        {
            "verdict": decision["verdict"],
            "task": "SB-B0",
            "workbook": 98,
            "decision": decision["decision"],
            "mechanism": decision["mechanism"],
            "situation": decision["situation"],
            "config_sha256": config_sha,
        }
    )
    print(
        f"SB-B0 verdict={decision['verdict']} decision={decision['decision']} "
        f"mechanism={decision['mechanism']} situation={decision['situation']} "
        f"run_id={store.run_id}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
