"""Independent scaled checks of immutable common-seed ACTS diagnostic samples."""
from __future__ import annotations

import numpy as np

from alignment.common_track_geometry import se3_exp


def array(value, shape):
    a = np.asarray(value, dtype=float).reshape(shape)
    if not np.isfinite(a).all():
        raise ValueError("nonfinite scientific value")
    return a


def peak(value):
    return float(np.max(np.abs(value)))


def derivative_check(samples, steps, scales, protocol):
    if len(samples) != len(steps):
        raise ValueError("incomplete derivative columns")
    full, half = [], []
    for s in samples:
        full.append((array(s["plus"], 4) - array(s["minus"], 4)) / 2)
        half.append(array(s["half_plus"], 4) - array(s["half_minus"], 4))
    full = np.asarray(full).T / scales[:, None]
    half = np.asarray(half).T / scales[:, None]
    difference = peak(full - half)
    tolerance = protocol["step_halving_absolute_floor"] + protocol["step_halving_relative_tolerance"] * peak(half)
    return {"error": difference, "tolerance": tolerance, "pass": difference <= tolerance,
            "scaled_effect": half.tolist(), "jacobian": (half * scales[:, None] / np.asarray(steps)[None, :]).tolist()}


def taylor_check(sample, nominal, jacobian, scales, protocol):
    delta = array(sample["delta"], jacobian.shape[1])
    effect = jacobian @ delta
    if not np.allclose(effect, array(sample["linear_effect"], 4), rtol=1e-10, atol=1e-12):
        raise ValueError("stored Taylor effect not from independent FD reconstruction")
    full = peak((array(sample["full"], 4) - nominal - effect) / scales)
    half = peak((array(sample["half"], 4) - nominal - 0.5 * effect) / scales)
    floor = protocol["taylor_absolute_floor"]
    tolerance = floor + protocol["taylor_relative_tolerance"] * peak(effect / scales)
    contraction = half / full if full > floor else None
    passed = full <= tolerance and (contraction is None or contraction <= protocol["taylor_contraction_maximum"])
    return {"full_error": full, "half_error": half, "tolerance": tolerance,
            "contraction": contraction, "pass": passed}


def verify_identity(result, fixture):
    for name in ("input_xaod", "ordinal", "actual_run", "actual_event", "variant"):
        if result[name] != fixture[name]:
            raise ValueError(f"identity mismatch: {name}")
    references = {r["station"]: r for r in fixture["references"]}
    targets = {r["station"]: r for r in result["targets"]}
    if set(references) != set(range(4)) or set(targets) != set(range(4)) or len(result["targets"]) != 4:
        raise ValueError("missing/duplicate station")
    expected = np.r_[array(references[0]["fixed_z_state"], 4), references[0]["q_over_p_per_MeV"]]
    if not np.array_equal(array(result["seed"], 5), expected):
        raise ValueError("common seed changed")
    if result["seed_z_mm"] != references[0]["z_state_mm"]:
        raise ValueError("seed reference plane changed")
    for station, target in targets.items():
        if not np.array_equal(array(target["y"], 4), array(references[station]["fixed_z_state"], 4)):
            raise ValueError("measurement y changed under model perturbation")
        if [r["strip"] for r in target["sensors"]] != references[station]["clusters"]:
            raise ValueError("sensor provenance/ordering changed")
    return references, targets


def validate(result, fixture):
    p = fixture["protocol"]
    references, targets = verify_identity(result, fixture)
    scales = np.asarray(p["output_scales"])
    seed_scale = np.r_[scales, 1e-5]
    roundtrip = peak((array(result["seed_roundtrip"], 5) - array(result["seed"], 5)) / seed_scale)
    if result["acts_q_over_p"] * result["acts_MeV_unit"] != references[0]["q_over_p_per_MeV"]:
        raise ValueError("ACTS q/p unit contract violated")
    twist = np.asarray(fixture["twist"], dtype=float)
    expected = se3_exp(twist)
    checks = []
    for station, target in targets.items():
        A = expected if station == fixture["perturbed_station"] else np.eye(4)
        for sensor in target["sensors"]:
            D = array(sensor["delta"], (4, 4))
            tr = peak(D[:3, 3] - A[:3, 3])
            rot = peak(D[:3, :3] - A[:3, :3])
            checks.append({"name": "actual_sensitive_surface_transform", "station": station,
                           "translation_error_mm": tr, "rotation_error": rot,
                           "pass": tr <= p["frame_translation_tolerance_mm"] and rot <= p["frame_rotation_tolerance"]})
        frame = array(target["frame"], (4, 4))
        base = np.eye(4); base[2, 3] = references[station]["z_state_mm"]
        frame_error = peak(frame - A @ base)
        checks.append({"name": "actual_reference_frame", "station": station, "error": frame_error,
                       "pass": frame_error <= p["frame_translation_tolerance_mm"]})
        h = array(target["h"], 4)
        local = np.linalg.solve(frame, np.r_[array(target["global_position"], 3), 1])
        direction = frame[:3, :3].T @ array(target["global_direction"], 3)
        back = np.r_[local[:2], direction[:2] / direction[2]]
        error = max(peak((back - h) / scales), abs(local[2]))
        checks.append({"name": "independent_frame_roundtrip", "station": station, "error": error,
                       "pass": error <= p["frame_roundtrip_tolerance"]})
        repeat = peak((array(target["repeat_h"], 4) - h) / scales)
        checks.append({"name": "repeated_official_prediction", "station": station, "error": repeat, "pass": repeat <= 1e-9})
        if target["boundary_not_propagated"] != (station == 0):
            raise ValueError("common boundary misreported")
        for sample in target["field_samples"]:
            array(sample["position_mm"], 3); array(sample["field_T"], 3)
        if fixture["variant"] == "baseline":
            xi = derivative_check(target["xi"], p["seed_steps"], scales, p)
            checks.append({"name": "Hxi_step_halving", "station": station, **xi})
            checks.append({"name": "Hxi_Taylor", "station": station,
                           **taylor_check(target["xi_direction"], h, np.asarray(xi["jacobian"]), scales, p)})
            # This detects cached/fixed predictions without imposing a physical association.
            response = abs(xi["scaled_effect"][0][0])
            checks.append({"name": "common_seed_x_response", "station": station, "response": response, "pass": response > 1e-5})
            if station:
                theta = derivative_check(target["theta_plane"], p["geometry_steps"], scales, p)
                checks.append({"name": "Htheta_plane_step_halving", "station": station, **theta})
                checks.append({"name": "Htheta_plane_Taylor", "station": station,
                               **taylor_check(target["theta_direction"], h, np.asarray(theta["jacobian"]), scales, p)})
    checks.append({"name": "seed_ACTS_roundtrip", "error": roundtrip,
                   "pass": roundtrip <= p["frame_roundtrip_tolerance"]})
    return {"gate": "PASS" if all(c["pass"] for c in checks) else "FAIL", "checks": checks,
            "qualification": "NOT_EVALUATED", "association": "NOT_EVALUATED", "covariance": "NOT_EVALUATED"}


def conditions_comparison(baseline, variants, fixture):
    p = fixture["protocol"]; scales = np.asarray(p["output_scales"])
    station = p["sqlite_pilot_station"]
    target = next(t for t in baseline["targets"] if t["station"] == station)
    checks = []
    for axis in p["sqlite_pilot_axes"]:
        samples = {multiplier: next(t for t in variants[(axis, multiplier)]["targets"] if t["station"] == station)
                   for multiplier in p["sqlite_multipliers"]}
        plane = target["theta_plane"][axis]
        for multiplier, key in ((1., "plus"), (-1., "minus"), (.5, "half_plus"), (-.5, "half_minus")):
            effect = (array(plane[key], 4) - array(target["h"], 4)) / scales
            error = peak((array(samples[multiplier]["h"], 4) - array(plane[key], 4)) / scales)
            tolerance = p["step_halving_absolute_floor"] + p["step_halving_relative_tolerance"] * peak(effect)
            checks.append({"name": "conditions_vs_target_plane", "axis": axis, "multiplier": multiplier,
                           "error": error, "tolerance": tolerance, "pass": error <= tolerance})
        s = {"plus": samples[1.]["h"], "minus": samples[-1.]["h"],
             "half_plus": samples[.5]["h"], "half_minus": samples[-.5]["h"]}
        checks.append({"name": "Htheta_conditions_step_halving", "axis": axis,
                       **derivative_check([s], [p["geometry_steps"][axis]], scales, p)})
    return {"gate": "PASS" if all(c["pass"] for c in checks) else "FAIL", "checks": checks,
            "scope": "pilot_station_1_vx_wx_wz_only", "full_Htheta": "UNKNOWN", "qualification": "NOT_EVALUATED"}
