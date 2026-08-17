#!/usr/bin/env python3
"""Audit whether a propagated-tracklet ROOT product supports a global field fit.

The audit reads branch metadata only.  It distinguishes the current
field-aware pairwise candidate contract from the additional state-transition
information required to form a defensible multi-station field-aware global
track likelihood.  It does not infer an unexported Acts Jacobian.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import uproot

from datasets.propagation_loader import PROPAGATION_TREE_NAME, REQUIRED_PROPAGATION_FIELDS


JACOBIAN_TOKENS = ("jacob", "transport", "transition", "derivative")
SOURCE_STATE_TOKENS = ("source_x", "source_y", "source_tx", "source_ty", "source_cov")


def _branches(path: Path, tree_name: str) -> set[str]:
    with uproot.open(path) as root_file:
        if tree_name not in root_file:
            raise ValueError(f"tree '{tree_name}' is absent from {path}")
        return {str(name).split(";")[0] for name in root_file[tree_name].keys()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--propagations", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tree", default=PROPAGATION_TREE_NAME)
    args = parser.parse_args()
    path = Path(args.propagations).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    fields = _branches(path, str(args.tree))
    required_missing = sorted(set(REQUIRED_PROPAGATION_FIELDS) - fields)
    jacobian_fields = sorted(field for field in fields if any(token in field.lower() for token in JACOBIAN_TOKENS))
    source_state_fields = sorted(field for field in fields if any(token in field.lower() for token in SOURCE_STATE_TOKENS))
    has_transport_jacobian = bool(jacobian_fields)
    supports_global_field_fit = not required_missing and has_transport_jacobian
    payload = {
        "method": "read_only_field_global_track_fit_input_contract_audit",
        "propagations": str(path),
        "tree": str(args.tree),
        "branch_count": len(fields),
        "required_pairwise_propagation_fields_missing": required_missing,
        "transport_jacobian_candidate_fields": jacobian_fields,
        "source_state_candidate_fields": source_state_fields,
        "supports_pairwise_field_aware_candidate_residual": not required_missing,
        "supports_field_aware_global_track_fit": supports_global_field_fit,
        "interpretation": (
            "A global field-aware fit requires an explicitly exported transport/state-transition Jacobian or an "
            "equivalent validated Acts repropagation interface. Pairwise target predictions and covariances alone "
            "support route-edge consistency, but their endpoint residuals are correlated and are not an independent "
            "global likelihood."
        ),
        "next_required_interface": (
            None
            if supports_global_field_fit
            else "Export a documented Acts transport Jacobian/state representation, or provide a validated refit API "
            "that can propagate a common global parameter vector through the current alignment context."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
