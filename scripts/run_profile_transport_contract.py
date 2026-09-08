#!/usr/bin/env python3
"""Login-safe B14T entry: inheritance and contract printout.

Does not launch Athena, does not enter B15, and does not submit the
1989-row campaign.  Official writes go through
audit_profile_transport_contract.py.  ACTS smoke is
run_profile_transport_contract_smoke.sh.
"""

from __future__ import annotations

import argparse
import json

from alignment.operating_protocol_v1_final_closure import project_root, resolve_under_root
from datasets.profile_transport_contract import (
    inherit_frozen_stage,
    load_config,
    standalone_likelihood_transport_contract,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/profile_transport_contract_v1.yaml")
    args = parser.parse_args()
    config = load_config(resolve_under_root(project_root(), args.config))
    inherited = inherit_frozen_stage(config)
    payload = {
        "task": "SB-B14T",
        "workbook": 116,
        "inherited_wb115": inherited["workbook_115"]["decision"],
        "statistical_model_unchanged": True,
        "b15_authorized": False,
        "b14m_reopen_authorized": False,
        "full_sample_authorized": False,
        "login_node_full_sample_forbidden": True,
        "likelihood_contract": standalone_likelihood_transport_contract(),
    }
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
