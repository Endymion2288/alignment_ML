#!/usr/bin/env python3
"""T03 alignment-chart contract smoke.  Geometry only; no Athena, no payload write."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

import numpy as np

from alignment.cad_survey_nov22 import git_head_sha
from alignment.nov22_metrology_provenance_station_ry import (
    extract_alpha_beta_gamma,
    station_alignment_transform,
)
from alignment.operating_protocol_v1_final_closure import (
    project_root,
    resolve_under_root,
    sha256_file,
)
from evaluation.artifact_store import ImmutableArtifactStore
from geometry.alignment_charts import (
    ROUND_TRIP_TOLERANCE,
    STATIONS_GLOBAL_ORIGIN,
    origin_translation_from_center,
    se3_from_sixvector,
    sixvector_from_se3,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/alignment_chart_contract_v1.yaml")
    args = parser.parse_args()
    config_path = resolve_under_root(project_root(), args.config)
    store = ImmutableArtifactStore.begin(
        resolve_under_root(project_root(), "outputs/alignment_chart_contract_v1"),
        "t03_alignment_chart",
    )
    mixed = (1.0, -2.0, 0.5, 0.03, -0.04, 0.05)
    built = se3_from_sixvector(mixed, STATIONS_GLOBAL_ORIGIN)
    extracted = sixvector_from_se3(built, STATIONS_GLOBAL_ORIGIN)
    rebuilt = se3_from_sixvector(extracted, STATIONS_GLOBAL_ORIGIN)
    round_trip = float(np.linalg.norm(rebuilt - built))
    survey = extract_alpha_beta_gamma(station_alignment_transform(mixed))
    survey_rebuild = station_alignment_transform((0.0, 0.0, 0.0, *survey))
    survey_frobenius = float(
        np.linalg.norm(
            survey_rebuild[:3, :3] - station_alignment_transform(mixed)[:3, :3]
        )
    )
    center = np.array([10.0, -4.0, -1860.0], dtype=np.float64)
    rotation = built[:3, :3]
    t_origin = origin_translation_from_center((0.2, -0.1, 0.05), rotation, center)
    payload = {
        "kind": "alignment_chart_contract",
        "task": "T03",
        "config_sha256": sha256_file(config_path),
        "git_head_sha": git_head_sha(),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "round_trip_norm": round_trip,
        "round_trip_tolerance": ROUND_TRIP_TOLERANCE,
        "round_trip_pass": round_trip < ROUND_TRIP_TOLERANCE,
        "legacy_survey_extract_frobenius": survey_frobenius,
        "legacy_survey_is_not_general_inverse": survey_frobenius > 1.0e-4,
        "origin_from_center_mm": [float(item) for item in t_origin],
        "geometry_write_allowed": False,
        "sqlite_or_pool_written": False,
        "charts_silently_converted": False,
    }
    fd = {
        "kind": "finite_difference_checks",
        "center_to_origin_identity_linearization": "see tests/test_alignment_chart_contract.py",
        "relative_tolerance": 1.0e-6,
        "geometry_only": True,
    }
    store.write_json("alignment_chart_contract.json", payload)
    store.write_json("finite_difference_checks.json", fd)
    verdict = (
        "PASS"
        if payload["round_trip_pass"] and payload["legacy_survey_is_not_general_inverse"]
        else "FAIL"
    )
    store.finalize({"verdict": verdict, "task": "T03"})
    print(f"T03 verdict={verdict} run_id={store.run_id}")
    return 0 if verdict == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
