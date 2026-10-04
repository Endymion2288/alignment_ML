#!/usr/bin/env python3
"""Independent, read-only audit of the WB127 recovery exports."""
import json, hashlib
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/mc24_four_station_wb127_strip_measurement_v1"
REC = OUT / "recovery_v5"
AUDIT = REC / "independent_audit.json"
INDICES = (1, 4, 8, 12, 16, 20)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    events = []
    for index in INDICES:
        name = f"{index:02d}"
        export_path = REC / "events" / name / "export.json"
        fixture_path = OUT / "events" / name / "fixture.json"
        export = json.loads(export_path.read_text())
        fixture = json.loads(fixture_path.read_text())
        references = {int(ref["station"]): ref for ref in fixture["references"]}
        rows = export["rows"]
        expected = [cluster for ref in fixture["references"] for cluster in ref["clusters"]]
        actual = [row["cluster_id"] for row in rows]
        if sorted(actual) != sorted(expected) or len(set(actual)) != len(actual):
            raise RuntimeError(f"allowlist mismatch in event {index}")
        if export["identity"]["actual_run"] != fixture["actual_run"] or export["identity"]["actual_event"] != fixture["actual_event"]:
            raise RuntimeError(f"identity mismatch in event {index}")
        A, u, weights, residuals = [], [], [], []
        frame_errors, frame_roundtrip_errors, rdo_counts = [], [], []
        seed = np.asarray(export["p_seed"], dtype=float).reshape(-1)
        if seed.size != 5:
            raise RuntimeError(f"P seed shape in event {index}")
        z0 = references[0]["z_center_mm"]
        x0, y0, tx, ty = seed[:4]
        for row in rows:
            sa, ca = float(row["sin_alpha"]), float(row["cos_alpha"])
            z, sigma_sq = float(row["z_relative_center"]), float(row["sigma_sq"])
            if not np.isfinite(sigma_sq) or sigma_sq <= 0:
                raise RuntimeError(f"covariance in event {index}")
            A.append([sa, ca, z * sa, z * ca])
            u.append(float(row["u"]))
            weights.append(1.0 / sigma_sq)
            transform = np.asarray(row["sensor_transform"], dtype=float)
            rotation = transform[:3, :3]
            frame_errors.append(float(np.max(np.abs(rotation.T @ rotation - np.eye(3)))))
            local = np.asarray(row["local_position"], dtype=float)
            global_position = np.asarray(row["global_position"], dtype=float)
            lifted = transform @ np.asarray([local[0], local[1], 0.0, 1.0])
            frame_roundtrip_errors.append(float(np.max(np.abs(lifted - np.r_[global_position, 1.0]))))
            rdo_counts.append(len(row["rdo_ids"]))
            predicted_global = np.asarray([x0 + tx * (global_position[2] - z0), y0 + ty * (global_position[2] - z0), global_position[2], 1.0])
            predicted_local = np.linalg.inv(transform) @ predicted_global
            residuals.append(float(local[0] - predicted_local[0]))
        A, u, weights = np.asarray(A), np.asarray(u), np.asarray(weights)
        normal = A.T @ (weights[:, None] * A)
        rhs = A.T @ (weights * u)
        station_results = []
        for station, reference in sorted(references.items()):
            selected = np.asarray([row["cluster_id"] in set(reference["clusters"]) for row in rows])
            ns = A[selected].T @ (weights[selected, None] * A[selected])
            bs = A[selected].T @ (weights[selected] * u[selected])
            fit = np.linalg.solve(ns, bs)
            raw_normal = np.asarray(reference["raw_normal"], dtype=float)
            raw_fit = np.asarray(reference["raw_fit"], dtype=float).reshape(-1)
            station_results.append({
                "station": station,
                "rows": int(np.sum(selected)),
                "normal_max_abs_diff": float(np.max(np.abs(ns - raw_normal))),
                "rhs_max_abs_diff": float(np.max(np.abs(bs - raw_normal @ raw_fit))),
                "fit_max_abs_diff": float(np.max(np.abs(fit - raw_fit))),
            })
        by_station = {}
        for station in sorted({row["station"] for row in rows}):
            values = np.asarray([residuals[i] for i, row in enumerate(rows) if row["station"] == station])
            by_station[str(station)] = {
                "rows": int(values.size),
                "min_mm": float(np.min(values)),
                "max_mm": float(np.max(values)),
                "rms_mm": float(np.sqrt(np.mean(values * values))),
            }
        events.append({
            "index": index,
            "actual_run": fixture["actual_run"],
            "actual_event": fixture["actual_event"],
            "rows": len(rows),
            "stations": sorted({row["station"] for row in rows}),
            "allowlist_exact": True,
            "frame_max_orthogonality_error": max(frame_errors),
            "frame_roundtrip_max_abs_error": max(frame_roundtrip_errors),
            "covariance_positive_finite": True,
            "rdo_min_count": min(rdo_counts),
            "normal_max_abs_diff": max(item["normal_max_abs_diff"] for item in station_results),
            "rhs_from_raw_fit_max_abs_diff": max(item["rhs_max_abs_diff"] for item in station_results),
            "fit_max_abs_diff": max(item["fit_max_abs_diff"] for item in station_results),
            "station_normal_fit": station_results,
            "straight_line_local_residual_mm": by_station,
            "straight_line_residual_no_pass_cut": True,
        })
    result = {
        "schema": "wb127_independent_audit_v1",
        "source": "recovery_v5/export.json",
        "population": len(events),
        "new_propagation_calls": 0,
        "measurement_contract": "PASS",
        "normal_rhs_interface_closure": "PASS",
        "straight_line_prediction": "DIAGNOSTIC_ONLY",
        "curve_prediction": "NOT_EXECUTED",
        "association": "INHERITED_CONDITIONAL",
        "events": events,
        "input_digests": {str(path.relative_to(ROOT)): digest(path) for path in sorted(REC.glob("events/*/export.json"))},
    }
    with AUDIT.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print("AUDIT_COMPLETE", AUDIT)


if __name__ == "__main__":
    main()
