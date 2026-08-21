#!/usr/bin/env python3
"""Write the payload-level gauge-equivalence audit for IFT outer contrast.

This is an analytic/numeric comparison of Calypso six-vectors and composed
detector-element transforms.  It does not open a physical bank or the sealed
test split.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from alignment.gauge_equivalence import (
    IFT_LAYER_PITCH_MM,
    IFT_PLANE_Z_MM,
    IFT_STATION_Z_MM,
    audit_outer_contrast_equivalence,
)
from scripts.run_layer_linear_internal_closure import _json_ready


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="outputs/mc24_ift_layer_gauge_equivalence_audit_v1",
        help="Empty directory that receives gauge_equivalence_audit.json",
    )
    args = parser.parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    dx = audit_outer_contrast_equivalence("dx_mm", 0.12)
    rx = audit_outer_contrast_equivalence("rx_mrad", 0.7)
    report = {
        "schema_version": "faser-ift-layer-gauge-equivalence-audit-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "test_data_accessed": False,
        "station_5dof_frozen": True,
        "geometry": {
            "tag": "FASERNU-04",
            "ift_station_z_mm": IFT_STATION_Z_MM,
            "ift_layer_pitch_mm": IFT_LAYER_PITCH_MM,
            "ift_plane_z_mm": list(IFT_PLANE_Z_MM),
            "plane_z_source": (
                "station-0 reconstructed tracklet z from the relative-rx held-out "
                "content_audit, plus geomDB SCTFASERGENERAL LAYERPITCH=31.5 mm"
            ),
            "composition": (
                "g_element = g_station * Ad_{T(z_plane)}(g_layer), with "
                "g = T(dx,dy,dz) * Rz * Ry * Rx as in TrackerAlignDBTool"
            ),
        },
        "injections": {
            "relative_dx": dx,
            "relative_rx": rx,
        },
        "decision": {
            "frozen_station_reference_layer_is_gauge_cross_check": False,
            "frozen_station_reference_layer_role": "negative_control_different_physical_family",
            "canonical_internal_basis": "outer_contrast",
            "c_dx_unchanged": True,
            "c_rx_truth_selected_may_be_admitted_in_contrast_basis": True,
            "additive_equivalence_requires_station_compensation": True,
            "rx_se3_still_differs_after_station_compensation": True,
            "rx_mismatch_cause": "station_rx_global_origin_versus_layer_rx_conjugated_to_plane_z",
        },
    }
    path = output_dir / "gauge_equivalence_audit.json"
    path.write_text(json.dumps(_json_ready(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"wrote": str(path), "decision": report["decision"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
