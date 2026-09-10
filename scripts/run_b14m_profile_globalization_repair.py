#!/usr/bin/env python3
"""Login-safe B14M-T entry: inheritance and preregistered trust region.

Does not launch Athena and does not submit 1989.
"""

from __future__ import annotations

import argparse
import json

from alignment.operating_protocol_v1_final_closure import project_root, resolve_under_root
from datasets.b14m_profile_globalization_repair import (
    inherit_frozen_stage,
    likelihood_contract,
    load_config,
    preregistered_trust_region,
)
from datasets.b14m_restart_invariance import recover_optimizer_contract


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/b14m_profile_globalization_repair_v1.yaml"
    )
    args = parser.parse_args()
    config = load_config(resolve_under_root(project_root(), args.config))
    inherited = inherit_frozen_stage(config)
    contract = recover_optimizer_contract(config)
    payload = {
        "task": "SB-B14MT",
        "workbook": 131,
        "inherited_wb130": inherited["workbook_130"]["decision"],
        "inherited_wb129": inherited["workbook_129"]["decision"],
        "jacobian_contract_established": inherited["workbook_128"][
            "jacobian_contract_established"
        ],
        "shadow_mean_contract_established": inherited["workbook_127"][
            "shadow_mean_contract_established"
        ],
        "restart_invariance_authorized": False,
        "full_sample_authorized": False,
        "contracted_denominator": inherited["workbook_103"]["contracted_denominator"],
        "optimizer_contract_not_recoverable": contract[
            "optimizer_contract_not_recoverable"
        ],
        "trust_region": preregistered_trust_region(config),
        "likelihood_contract": likelihood_contract(),
        "do_not_submit_1989": True,
        "do_not_enter_b15": True,
        "smoke_events": ["100043/0", "100043/1", "100043/37", "100048/86"],
        "gate_identities": ["100043/37 T1", "100043/37 T2", "100048/86 T1"],
    }
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
