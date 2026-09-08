#!/usr/bin/env python3
"""Login-safe B14M entry: synthetic profile validation and contract printout.

Does not launch Athena, does not enter B15, and does not write official
artifacts.  Official writes go through audit_profiled_weak_nuisance_likelihood.py.
"""

from __future__ import annotations

import argparse
import json

from alignment.operating_protocol_v1_final_closure import project_root, resolve_under_root
from alignment.profiled_measurement_likelihood import validate_linear_profile_agreement
from datasets.profiled_weak_nuisance_likelihood import (
    inherit_frozen_stage,
    load_config,
    measurement_likelihood_contract,
    profiled_state_partition,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/profiled_weak_nuisance_likelihood_v1.yaml"
    )
    args = parser.parse_args()
    config = load_config(resolve_under_root(project_root(), args.config))
    inherited = inherit_frozen_stage(config)
    numerical = validate_linear_profile_agreement(
        relative=float(config["profile_numerical"]["pinv_relative"])
    )
    payload = {
        "task": "SB-B14M",
        "workbook": 114,
        "synthetic_passed": numerical["passed"],
        "inherited_wb113": inherited["workbook_113"]["decision"],
        "likelihood_contract": measurement_likelihood_contract(config),
        "partition": profiled_state_partition(config),
        "numerical_validation": numerical,
        "marginalization_executed": False,
        "b15_authorized": False,
    }
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0 if numerical["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
