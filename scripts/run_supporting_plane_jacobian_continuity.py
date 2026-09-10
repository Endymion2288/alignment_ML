#!/usr/bin/env python3
"""Login-safe B14J entry: inheritance and contract printout.

Does not launch Athena, does not enter B15, and does not submit the
1989-row campaign.  Official writes go through
audit_supporting_plane_jacobian_continuity.py.  ACTS smoke is
run_supporting_plane_jacobian_continuity_smoke.sh.
"""

from __future__ import annotations

import argparse
import json

from alignment.operating_protocol_v1_final_closure import project_root, resolve_under_root
from datasets.supporting_plane_jacobian_continuity import (
    inherit_frozen_stage,
    likelihood_contract,
    load_config,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/supporting_plane_jacobian_continuity_v1.yaml"
    )
    args = parser.parse_args()
    config = load_config(resolve_under_root(project_root(), args.config))
    inherited = inherit_frozen_stage(config)
    payload = {
        "task": "SB-B14J",
        "workbook": 117,
        "inherited_wb116": inherited["workbook_116"]["decision"],
        "inherited_wb116_jacobian_contract_established": inherited["workbook_116"][
            "jacobian_contract_established"
        ],
        "statistical_model_unchanged": True,
        "b15_authorized": False,
        "b14m_reopen_authorized": False,
        "full_sample_authorized": False,
        "login_node_full_sample_forbidden": True,
        "do_not_select_best_step": True,
        "do_not_switch_to_direct": True,
        "five_d_cin_not_the_objective": True,
        "likelihood_contract": likelihood_contract(),
    }
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
