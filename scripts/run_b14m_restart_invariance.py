#!/usr/bin/env python3
"""Login-safe B14M-R entry: inheritance and recovered optimizer printout.

Does not launch Athena, does not enter B15/WB130, and does not submit 1989.
Official writes go through audit_b14m_restart_invariance.py.
"""

from __future__ import annotations

import argparse
import json

from alignment.operating_protocol_v1_final_closure import project_root, resolve_under_root
from datasets.b14m_restart_invariance import (
    inherit_frozen_stage,
    likelihood_contract,
    load_config,
    recover_optimizer_contract,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/b14m_restart_invariance_v2.yaml"
    )
    args = parser.parse_args()
    config = load_config(resolve_under_root(project_root(), args.config))
    inherited = inherit_frozen_stage(config)
    contract = recover_optimizer_contract(config)
    payload = {
        "task": "SB-B14MR",
        "workbook": 129,
        "inherited_wb128": inherited["workbook_128"]["decision"],
        "inherited_wb127": inherited["workbook_127"]["decision"],
        "shadow_mean_contract_established": inherited["workbook_127"][
            "shadow_mean_contract_established"
        ],
        "focus_independent_reference_established": inherited["workbook_128"][
            "focus_independent_reference_established"
        ],
        "jacobian_contract_established": inherited["workbook_128"][
            "jacobian_contract_established"
        ],
        "b14m_reopen_authorized": inherited["workbook_128"]["b14m_reopen_authorized"],
        "restart_invariance_authorized": False,
        "full_sample_authorized": False,
        "contracted_denominator": inherited["workbook_103"]["contracted_denominator"],
        "optimizer_contract_not_recoverable": contract[
            "optimizer_contract_not_recoverable"
        ],
        "recovered_chi2_rel_tolerance": contract.get("chi2_rel_tolerance"),
        "recovered_prediction_abs_tolerance_mm": contract.get(
            "prediction_abs_tolerance_mm"
        ),
        "do_not_submit_1989": True,
        "do_not_enter_b15": True,
        "do_not_enter_wb130": True,
        "likelihood_contract": likelihood_contract(),
        "smoke_events": ["100043/0", "100043/1", "100043/37", "100048/86"],
    }
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
