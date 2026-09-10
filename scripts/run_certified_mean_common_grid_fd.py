#!/usr/bin/env python3
"""Login-safe B14ZC entry: inheritance and contract printout.

Does not launch Athena, does not enter B15, and does not submit 1989.
Official writes go through audit_certified_mean_common_grid_fd.py.
"""

from __future__ import annotations

import argparse
import json

from alignment.operating_protocol_v1_final_closure import project_root, resolve_under_root
from datasets.certified_mean_common_grid_fd import (
    inherit_frozen_stage,
    likelihood_contract,
    load_config,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/certified_mean_common_grid_fd_v1.yaml"
    )
    args = parser.parse_args()
    config = load_config(resolve_under_root(project_root(), args.config))
    inherited = inherit_frozen_stage(config)
    payload = {
        "task": "SB-B14ZC",
        "workbook": 128,
        "inherited_wb127": inherited["workbook_127"]["decision"],
        "inherited_wb126": inherited["workbook_126"]["decision"],
        "inherited_wb125": inherited["workbook_125"]["decision"],
        "inherited_wb123": inherited["workbook_123"]["decision"],
        "inherited_wb127_shadow_mean_contract_established": inherited[
            "workbook_127"
        ]["shadow_mean_contract_established"],
        "inherited_wb127_jacobian_contract_established": inherited[
            "workbook_127"
        ]["jacobian_contract_established"],
        "b14m_reopen_authorized": False,
        "full_sample_authorized": False,
        "do_not_change_mean_semantics": True,
        "do_not_change_official_mean_path": True,
        "do_not_relax_five_percent_gate": True,
        "do_not_add_fd_rung": True,
        "do_not_reuse_wb124_adaptive_dopri5": True,
        "do_not_submit_1989": True,
        "production_eloss_quantity": "computeEnergyLossBethe",
        "required_first_unstable": {"1": 6, "2": 11, "3": 11},
        "likelihood_contract": likelihood_contract(),
        "focus_event": "100048/86",
    }
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
