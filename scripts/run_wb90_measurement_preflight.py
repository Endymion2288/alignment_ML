#!/usr/bin/env python3
"""Freeze/run WB90 G0. Stops before check/Calypso jobs if covariance is invalid."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from alignment.wb90_measurement_contract import (
    ROOT as PROJECT, OUTPUT, PROTOCOL, CATALOG, ALLOWLIST, digest, read_public,
    write_new, selection, getstate_fragments, development_slopes,
    numerical_angle_jacobian, correct_angle_jacobian,
)

TEMPLATE = PROJECT / "scripts/wb90_trk_probe.cpp.in"
SELF = Path(__file__).resolve()


def freeze(output: Path) -> None:
    if output.exists():
        raise FileExistsError("new output directory required; historical artifacts are immutable")
    protocol = read_public(PROTOCOL)
    rows = selection(read_public(CATALOG), read_public(ALLOWLIST))
    fragments = getstate_fragments(PROJECT.parent / "calypso")
    source_paths = [PROTOCOL, CATALOG, ALLOWLIST, SELF, TEMPLATE,
                    PROJECT / "alignment/wb90_measurement_contract.py",
                    PROJECT / "scripts/setup_environment.sh"]
    # Pin the actual parameter definition, not a guessed coordinate convention.
    release = Path("/cvmfs/atlas.cern.ch/repo/sw/software/24.0/Athena/24.0.41/InstallArea/x86_64-el9-gcc13-opt")
    source_paths += [release / "include/TrkParametersBase/CurvilinearParametersT.h",
                     release / "include/TrkParametersBase/CurvilinearParametersT.icc",
                     release / "include/TrkEventPrimitives/CurvilinearUVT.h"]
    source_hashes = {str(p.resolve()): digest(p) for p in source_paths}
    write_new(output / "protocol.json", protocol)
    write_new(output / "selection.json", {"schema": "wb90_selection_v1", "events": rows,
              "n_events": len(rows), "selection_used_outcomes": False,
              "data_role": "historically seen development; not independent qualification"})
    write_new(output / "source_freeze.json", {
        "utc": datetime.now(timezone.utc).isoformat(), "source_hashes": source_hashes,
        "fragments": fragments,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--short"], cwd=PROJECT, text=True),
        "check_measurements_opened": False, "physical_calls": 0})
    print(f"Frozen {len(rows)} selection entries and source fragments at {output}")


def probe(output: Path) -> None:
    frozen = read_public(output / "source_freeze.json")
    protocol = read_public(output / "protocol.json")
    if (output / "preflight_summary.json").exists():
        raise FileExistsError("no overwriting a completed preflight")
    for name, expected in frozen["source_hashes"].items():
        if digest(Path(name)) != expected:
            raise ValueError(f"frozen source changed: {name}")
    fragments = frozen["fragments"]
    for key in ("fitter", "exporter"):
        if digest(Path(fragments[key])) != fragments[f"{key}_sha256"]:
            raise ValueError("Calypso source changed after freeze")
    import ROOT
    libraries = {}
    for name in ("libTrkParameters", "libTrkSurfaces"):
        if ROOT.gSystem.Load(name) < 0:
            raise RuntimeError(f"cannot load actual Athena library: {name}")
        path = Path(str(ROOT.gSystem.DynamicPathName(name + ".so", True))).resolve()
        libraries[name] = {"path": str(path), "sha256": digest(path)}
    for prefix in os.environ.get("CMAKE_PREFIX_PATH", "").split(":"):
        for suffix in ("include", "include/eigen3"):
            p = Path(prefix) / suffix
            if p.is_dir():
                ROOT.gInterpreter.AddIncludePath(str(p))
    code = TEMPLATE.read_text().replace("@@EXPORT_SOURCE@@", fragments["export_cpp"]).replace(
        "@@GETSTATE_SOURCE@@", fragments["covariance_cpp"])
    # A local static constexpr member is illegal in this ROOT C++ standard;
    # preserve source expression but supply zCenter via a namespace struct.
    code = code.replace("namespace WB90Probe {", "namespace WB90Probe {\nstruct fitInfo { static constexpr double zCenter = 0.; };")
    code = code.replace("  struct fitInfo { static constexpr double zCenter = 0.; };\n", "")
    cpp_path = output / "compiled_probe_source.cpp"
    if cpp_path.exists():
        raise FileExistsError(cpp_path)
    cpp_path.write_text(code)
    if not ROOT.gInterpreter.Declare(code):
        raise RuntimeError("actual Trk diagnostic source failed to compile")
    controls = [{"kind": "analytic_control", "tx": tx, "ty": ty} for tx, ty in protocol["cases"]]
    rows = read_public(output / "selection.json")["events"]
    cases = controls + [{"kind": "development_slope_kernel", **r} for r in development_slopes(rows)]
    results = []
    for case in cases:
        tx, ty = case["tx"], case["ty"]
        values = np.array(list(ROOT.WB90Probe.evaluate(tx, ty, protocol["frame_derivative_step_mm"])))
        if values.size != 71 or not np.isfinite(values).all():
            raise ValueError("invalid actual kernel output")
        source_j = values[:16].reshape(4, 4)[2:, 2:]
        native_cov = values[16:41].reshape(5, 5)
        frame = values[41:50].reshape(3, 3)
        export_cov = values[50:60]
        position_j = values[60:66].reshape(2, 3).T
        fd = numerical_angle_jacobian(tx, ty, protocol["angle_fd_step"])
        independent = correct_angle_jacobian(tx, ty)
        correct_fd_error = float(np.linalg.norm(independent-fd)/np.linalg.norm(fd))
        source_fd_error = float(np.linalg.norm(source_j-fd)/np.linalg.norm(fd))
        # The fitted position lies at fixed lab z. Transform its uncertainty
        # onto the actual curvilinear UV axes; basis comes from Trk, not a guess.
        fit_cov = np.diag(protocol["fit_covariance_control_diagonal"][:2])
        projection = frame[:, :2].T[:, :2]
        correct_uv_cov = projection @ fit_cov @ projection.T
        frame_cov_error = float(np.linalg.norm(native_cov[:2, :2]-correct_uv_cov)/np.linalg.norm(correct_uv_cov))
        factory_error = float(np.max(np.abs(position_j-frame[:, :2])))
        if correct_fd_error > protocol["derivative_relative_tolerance"]:
            raise ValueError("independent analytic positive control disagrees with FD")
        results.append({**case, "source_angle_jacobian": source_j.tolist(), "central_fd": fd.tolist(),
            "correct_analytic_relative_error": correct_fd_error,
            "source_jacobian_relative_error": source_fd_error,
            "actual_curvilinear_frame": frame.tolist(), "native_parameters": values[66:].tolist(),
            "source_native_covariance_control": native_cov.tolist(),
            "proper_uv_position_covariance_control": correct_uv_cov.tolist(),
            "native_position_covariance_relative_error": frame_cov_error,
            "surface_factory_basis_absolute_error": factory_error,
            "exported_covariance_upper_triangle_control": export_cov.tolist(),
            "angle_contract_pass": source_fd_error <= protocol["derivative_relative_tolerance"],
            "native_position_covariance_contract_pass": frame_cov_error <= protocol["derivative_relative_tolerance"],
            "actual_surface_factory_pass": factory_error <= protocol["frame_absolute_tolerance"]})
    write_new(output / "kernel_results.json", {"results": results, "libraries": libraries,
        "compiled_source_sha256": digest(cpp_path), "root_version": ROOT.gROOT.GetVersion(),
        "control_covariance_is_actual_event_covariance": False})
    angle_fail = sum(not r["angle_contract_pass"] for r in results)
    frame_fail = sum(not r["native_position_covariance_contract_pass"] for r in results)
    factory_fail = sum(not r["actual_surface_factory_pass"] for r in results)
    write_new(output / "preflight_summary.json", {
        "schema": "wb90_preflight_summary_v1", "preflight": "FAIL" if angle_fail or frame_fail or factory_fail else "SUPPORTED",
        "n_analytic_controls": len(controls), "n_development_slope_kernels": len(cases)-len(controls),
        "development_events_opened": len({r["event_uid"] for r in cases if "event_uid" in r}),
        "angle_contract_failures": angle_fail, "native_covariance_contract_failures": frame_fail,
        "surface_factory_failures": factory_fail, "check_measurements_opened": False,
        "physical_calls": 0, "calypso_condor_submitted": False,
        "actual_acts_propagation_evaluated": False, "common_state_jacobian_evaluated": False,
        "physical_covariance_calibration": "UNKNOWN", "historical_binary_causal_attribution": "UNKNOWN",
        "qualification": "NOT_EVALUATED", "ml_alignment_eval_authorized": False,
        "reason": "Covariance/frame prerequisite contradicted; fail-fast, do not consume check/geometry budget" if angle_fail or frame_fail or factory_fail else "Preflight only; no physical qualification",
        "source_freeze_sha256": digest(output / "source_freeze.json"),
        "protocol_sha256": digest(output / "protocol.json"), "selection_sha256": digest(output / "selection.json"),
        "kernel_results_sha256": digest(output / "kernel_results.json")})
    print(json.dumps(read_public(output / "preflight_summary.json"), indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "probe"))
    parser.add_argument("--output-root", type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if not output.is_relative_to(PROJECT / "outputs") or not output.name.startswith("mc24_four_station_wb90_"):
        parser.error("WB90 new output root required")
    (freeze if args.action == "freeze" else probe)(output)


if __name__ == "__main__":
    main()
