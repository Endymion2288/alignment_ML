"""Task B14P: target-independent weak-parameter prior admissibility.

Asks whether y, tx/φ, and q/p have a real-data, target-independent
prior with defined uncertainty semantics.  Does not introduce a prior,
delete q/p, pick a seed scale, enter B14M, enter B15, or enter V2.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import yaml

from alignment.faseracts_propagated_covariance_validation_v2 import FROZEN_WB81_GATES
from alignment.operating_protocol_v1_final_closure import (
    project_root,
    resolve_under_root,
    sha256_file,
)
from datasets.acts_transport_tail_analysis import identity_key
from datasets.leave_target_out_state_materialization import (
    FROZEN_N_CONTRACTED,
    FROZEN_N_INELIGIBLE,
    FROZEN_N_OFFICIAL_PAIRS,
    FROZEN_N_RAW,
    _row_key,
    load_lto_dumps,
)
from datasets.lto_supported_state_contract import (
    CASE_B as WB112_DECISION,
    CLASS_MEASUREMENT,
    CLASS_PRIOR,
    CLASS_WEAK,
    inherit_frozen_stage as inherit_through_wb111,
)
from datasets.transport_uncertainty_shape_diagnosis import load_contracted_sample

SCHEMA_VERSION = "target-independent-weak-prior-contract-v1"
DEFAULT_CONFIG = "configs/target_independent_weak_prior_contract_v1.yaml"
TASK = "SB-B14P"
WORKBOOK = 113

CASE_A = "target_independent_prior_contract_established"
CASE_B = "target_independent_prior_not_available"
CASE_C = "prior_available_for_subset_only"
CASE_D = "prior_bias_or_independence_not_established"
CASE_E = "mixed_or_inconclusive"

FROZEN_DIRECTION_CLASS = {
    "x": CLASS_MEASUREMENT,
    "y": CLASS_WEAK,
    "tx": CLASS_PRIOR,
    "ty": CLASS_MEASUREMENT,
    "q_over_p": CLASS_WEAK,
}
WEAK_DIRECTIONS = ("y", "tx", "q_over_p")
SUPPORTED_DIRECTIONS = ("x", "ty")
EVENT_LEVEL = "event_level_measurement_prior"
POPULATION_LEVEL = "population_level_physics_prior"


class TargetIndependentPriorError(ValueError):
    """Raised when the B14P contract is illegal."""


def refuse_truth_prior() -> None:
    raise TargetIndependentPriorError("truth q/p is not a real-data prior")


def refuse_covariance_rescale() -> None:
    raise TargetIndependentPriorError("covariance rescale is forbidden")


def refuse_seed_scale_choice() -> None:
    raise TargetIndependentPriorError("seed scale must not be chosen from truth or chi2")


def refuse_prior_introduction() -> None:
    raise TargetIndependentPriorError("B14P audits priors but does not introduce one")


def refuse_qoverp_deletion() -> None:
    raise TargetIndependentPriorError("q/p must remain an explicit nuisance, not be deleted")


def refuse_qoverp_fix() -> None:
    raise TargetIndependentPriorError("q/p must not be fixed to a seed or constant")


def refuse_focus_drop() -> None:
    raise TargetIndependentPriorError("focus identity 100043/37 must be retained")


def refuse_measurement_model_v2() -> None:
    raise TargetIndependentPriorError("Measurement Model V2 is not entered in Task B14P")


def refuse_b15() -> None:
    raise TargetIndependentPriorError("Task B15 is not entered in Task B14P")


def refuse_b14m() -> None:
    raise TargetIndependentPriorError("Task B14M is not entered until B14P Case B is official")


def refuse_full_track_prior() -> None:
    raise TargetIndependentPriorError("full-track CKF q/p or WB107 Cin cannot be an LTO prior")


def refuse_arbitrary_gaussian() -> None:
    raise TargetIndependentPriorError("arbitrary Gaussian is not an admissible prior")


def refuse_cin_injection() -> None:
    raise TargetIndependentPriorError("prior covariance must not be added into WB109 Cin")


def _expect_sha(path: Path, expected: str, label: str) -> None:
    digest = sha256_file(path)
    if digest != expected:
        raise TargetIndependentPriorError(f"{label} hash mismatch: {digest}")


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = resolve_under_root(project_root(), str(path or DEFAULT_CONFIG))
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != SCHEMA_VERSION:
        raise TargetIndependentPriorError(f"schema_version must be {SCHEMA_VERSION}")
    if config.get("task") != TASK:
        raise TargetIndependentPriorError(f"task must be {TASK}")
    if int(config.get("workbook", -1)) != WORKBOOK:
        raise TargetIndependentPriorError(f"workbook must be {WORKBOOK}")
    if str(config.get("official_input_scope")) != "contract_eligible":
        raise TargetIndependentPriorError("official input must be contract_eligible")
    for key in (
        "geometry_write_allowed",
        "held_out_accessed",
        "real_data_alignment_authorized",
        "measurement_model_validated",
    ):
        if bool(config.get(key, True)):
            raise TargetIndependentPriorError(f"{key} must be false")
    for key in (
        "do_not_rescale_covariance",
        "do_not_use_truth_q_over_p_as_real_data_solution",
        "do_not_use_raw_ckf_as_official_input",
        "do_not_tighten_reconstruction_contract",
        "do_not_drop_focus_identity",
        "do_not_enter_measurement_model_v2",
        "do_not_enter_b15",
        "do_not_enter_b14m",
        "do_not_choose_seed_scale_from_closure",
        "do_not_introduce_target_independent_prior",
        "do_not_instantiate_prior_in_production",
        "do_not_add_prior_into_wb109_cin",
        "do_not_delete_qoverp",
        "do_not_fix_qoverp",
        "do_not_use_full_track_ckf_as_prior",
        "do_not_use_wb107_cin_as_prior",
        "do_not_use_truth_as_prior",
        "do_not_use_closure_derived_prior",
        "do_not_use_seed_scale_as_prior",
        "do_not_use_arbitrary_gaussian_prior",
        "eligibility_independent_of_closure",
        "do_not_select_states_from_closure",
    ):
        if bool(config.get(key, False)) is not True:
            raise TargetIndependentPriorError(f"{key} must be true")
    if bool(config.get("do_not_enter_b14p", True)):
        raise TargetIndependentPriorError("B14P config must allow entering B14P")
    gates = config.get("closure_gates", {})
    for key, expected in FROZEN_WB81_GATES.items():
        if gates.get(key) != expected:
            raise TargetIndependentPriorError(f"frozen gate changed: {key}")
    wb103 = yaml.safe_load(
        resolve_under_root(
            project_root(), config["inheritance"]["workbook_103"]["config_path"]
        ).read_text(encoding="utf-8")
    )
    if dict(config["eligibility"]) != dict(wb103["eligibility"]):
        raise TargetIndependentPriorError("B14P eligibility must stay identical to WB103")
    if list(config["weak_directions"]) != list(WEAK_DIRECTIONS):
        raise TargetIndependentPriorError("weak directions must stay y, tx, q_over_p")
    if list(config["measurement_supported_directions"]) != list(SUPPORTED_DIRECTIONS):
        raise TargetIndependentPriorError("supported directions must stay x, ty")
    return dict(config)


def inherit_frozen_stage(config: Mapping[str, Any]) -> dict[str, Any]:
    inherited = inherit_through_wb111(config)
    spec = config["inheritance"]["workbook_112"]
    decision = json.loads(
        resolve_under_root(project_root(), spec["decision_path"]).read_text(encoding="utf-8")
    )
    contract = json.loads(
        resolve_under_root(project_root(), spec["contract_path"]).read_text(encoding="utf-8")
    )
    _expect_sha(resolve_under_root(project_root(), spec["config_path"]), spec["config_sha256"], "workbook_112 config")
    _expect_sha(resolve_under_root(project_root(), spec["decision_path"]), spec["decision_sha256"], "workbook_112 decision")
    _expect_sha(resolve_under_root(project_root(), spec["contract_path"]), spec["contract_sha256"], "workbook_112 contract")
    if decision.get("decision") != spec["frozen_decision"]:
        raise TargetIndependentPriorError("workbook_112 decision must stay frozen")
    if spec["frozen_decision"] != WB112_DECISION:
        raise TargetIndependentPriorError("WB112 decision token mismatch")
    if decision.get("b15_authorized"):
        raise TargetIndependentPriorError("WB112 must not have authorized B15")
    if decision.get("prior_introduced"):
        raise TargetIndependentPriorError("WB112 must not have introduced a prior")
    if not decision.get("b14p_authorized"):
        raise TargetIndependentPriorError("WB112 must have authorized B14P")
    classes = dict(contract.get("typical_direction_class") or {})
    expected = dict(spec["frozen_direction_class"])
    if classes != expected or classes != dict(FROZEN_DIRECTION_CLASS):
        raise TargetIndependentPriorError("WB112 direction classes must stay frozen")
    if contract.get("five_d_state_physically_supported"):
        raise TargetIndependentPriorError("WB112 five_d_state_physically_supported must stay false")
    inherited["workbook_112"] = {
        "decision": spec["frozen_decision"],
        "primary_case": spec["frozen_primary_case"],
        "active_mechanisms": list(spec.get("frozen_active_mechanisms") or []),
        "config_sha256": spec["config_sha256"],
        "decision_sha256": spec["decision_sha256"],
        "contract_sha256": spec["contract_sha256"],
        "typical_direction_class": expected,
        "five_d_state_physically_supported": False,
        "b14p_authorized": True,
        "b15_authorized": False,
        "prior_introduced": False,
    }
    return inherited


def _load_wb112_json(config: Mapping[str, Any], key: str) -> dict[str, Any]:
    spec = config["inheritance"]["workbook_112"]
    path = resolve_under_root(project_root(), spec[key])
    _expect_sha(path, spec[f"{key.replace('_path', '_sha256')}"], f"workbook_112 {key}")
    return json.loads(path.read_text(encoding="utf-8"))


def _focus_keys(config: Mapping[str, Any]) -> set[tuple[Any, ...]]:
    wanted = {
        (str(item["source_id"]), int(item["run_id"]), int(item["event_id"]))
        for item in config["focus_identities"]
    }
    for item in config.get("catastrophic_identities") or []:
        wanted.add((str(item["source_id"]), int(item["run_id"]), int(item["event_id"])))
    return wanted


def _identity_triple(row: Mapping[str, Any]) -> tuple[str, int, int]:
    return (str(row.get("source_id")), int(row.get("run_id", -1)), int(row.get("event_id", -1)))


def _admissible(flags: Mapping[str, Any]) -> bool:
    return bool(
        flags.get("target_independent")
        and flags.get("admissible_for_real_data")
        and flags.get("uncertainty_semantics_defined")
        and flags.get("independence_from_surviving_measurements")
        and flags.get("independence_from_alignment_geometry")
        and not flags.get("uses_truth")
        and not flags.get("uses_target_measurement")
        and not flags.get("closure_derived")
        and not flags.get("seed_scale_selected")
        and not flags.get("arbitrary_gaussian")
    )


def _candidate(**payload: Any) -> dict[str, Any]:
    flags = {
        "target_independent": bool(payload.get("target_independent", False)),
        "admissible_for_real_data": bool(payload.get("admissible_for_real_data", False)),
        "uncertainty_semantics_defined": bool(payload.get("uncertainty_semantics_defined", False)),
        "independence_from_surviving_measurements": bool(
            payload.get("independence_from_surviving_measurements", False)
        ),
        "independence_from_alignment_geometry": bool(
            payload.get("independence_from_alignment_geometry", False)
        ),
        "uses_truth": bool(payload.get("uses_truth", False)),
        "uses_target_measurement": bool(payload.get("uses_target_measurement", False)),
        "closure_derived": bool(payload.get("closure_derived", False)),
        "seed_scale_selected": bool(payload.get("seed_scale_selected", False)),
        "arbitrary_gaussian": bool(payload.get("arbitrary_gaussian", False)),
    }
    return {
        **payload,
        **flags,
        "admissible": _admissible(flags),
    }


def audit_requirements(config: Mapping[str, Any], wb112: Mapping[str, Any]) -> dict[str, Any]:
    contract = wb112["contract"]
    gain = wb112["information_gain"]
    directional = wb112["directional"]
    classes = dict(contract["typical_direction_class"])
    typical_gain = gain.get("typical_diagonal_gain_ratio") or {}
    typical_seed = directional.get("typical_direction_prior_dominated") or {}
    native = {"x": "loc0", "y": "loc1", "tx": "phi", "ty": "theta", "q_over_p": "q_over_p"}
    by_direction = {}
    for name in ("y", "tx", "q_over_p"):
        gain_med = (typical_gain.get(native[name]) or {}).get("median")
        by_direction[name] = {
            "wb112_class": classes[name],
            "why_prior_needed": (
                "weakly measured: surviving hits shrink the uninformative seed, "
                "but fitted width still follows the directional seed"
                if classes[name] == CLASS_WEAK
                else "prior-dominated: native information gain versus the uninformative seed is ~1"
            ),
            "measurement_only_information": {
                "typical_diagonal_gain_ratio_median": gain_med,
                "native_chart": native[name],
            },
            "seed_sensitivity": {
                "typical_direction_prior_dominated": bool(typical_seed.get(name)),
            },
            "prior_would_affect_state_mean": name == "q_over_p",
            "prior_would_affect_uncertainty": True,
            "must_be_available_on_real_data": True,
            "propagation_critical": name == "q_over_p",
            "must_not_delete_parameter": True,
        }
    return {
        "frozen_direction_class": classes,
        "measurement_supported_directions": list(SUPPORTED_DIRECTIONS),
        "weak_directions": list(WEAK_DIRECTIONS),
        "do_not_add_strong_prior_to_supported_directions": True,
        "by_direction": by_direction,
        "five_d_state_physically_supported": False,
    }


def audit_dependency_graph(
    sample: Mapping[str, Any],
    lto_by_key: Mapping[tuple[Any, ...], Mapping[str, Any]],
) -> dict[str, Any]:
    by_target: dict[str, dict[str, Any]] = {}
    n_source_in_likelihood = 0
    n_target_leaked = 0
    n_compared = 0
    for official in sample["contracted"]:
        lto = lto_by_key.get(_row_key(official))
        if lto is None:
            continue
        n_compared += 1
        target = int(official.get("target_station", -1))
        used_stations = [int(v) for v in (lto.get("used_station_ids") or [])]
        if 0 in used_stations:
            n_source_in_likelihood += 1
        if target in used_stations or int(lto.get("target_station_measurements_used") or 0) > 0:
            n_target_leaked += 1
        bucket = by_target.setdefault(
            str(target),
            {
                "n": 0,
                "n_source_used": 0,
                "n_target_used": 0,
                "used_station_sets": defaultdict(int),
            },
        )
        bucket["n"] += 1
        if 0 in used_stations:
            bucket["n_source_used"] += 1
        if target in used_stations:
            bucket["n_target_used"] += 1
        bucket["used_station_sets"][tuple(sorted(set(used_stations)))] += 1

    packed = {}
    for target, bucket in by_target.items():
        surviving = [station for station in (0, 1, 2, 3) if str(station) != target]
        packed[target] = {
            "excluded_target": int(target),
            "surviving_stations": surviving,
            "independent_unused_tracker_stations": [],
            "source_hits_already_in_surviving_likelihood": bucket["n_source_used"] == bucket["n"],
            "n": bucket["n"],
            "n_source_used": bucket["n_source_used"],
            "n_target_used": bucket["n_target_used"],
            "dominant_used_station_set": (
                max(bucket["used_station_sets"].items(), key=lambda item: item[1])[0]
                if bucket["used_station_sets"]
                else None
            ),
        }
    return {
        "n_compared": n_compared,
        "n_source_in_likelihood": n_source_in_likelihood,
        "n_target_measurement_leaked": n_target_leaked,
        "source_only_is_not_independent_of_lto_likelihood": n_source_in_likelihood == n_compared and n_compared > 0,
        "excluded_target_is_only_held_out_tracker": True,
        "systems": {
            "tracker_station_0": {
                "role": "source_ift",
                "crosses_spectrometer_magnet": False,
                "in_surviving_likelihood_for_targets": [1, 2, 3],
                "can_measure_q_over_p": False,
            },
            "tracker_station_1": {
                "in_surviving_likelihood_for_targets": [2, 3],
                "excluded_when_target": [1],
            },
            "tracker_station_2": {
                "in_surviving_likelihood_for_targets": [1, 3],
                "excluded_when_target": [2],
            },
            "tracker_station_3": {
                "in_surviving_likelihood_for_targets": [1, 2],
                "excluded_when_target": [3],
            },
            "spectrometer_magnet": {
                "between_stations": [0, 1],
                "is_measurement": False,
            },
            "calorimeter": {
                "downstream_of_tracker": True,
                "in_lto_measurement_set": False,
                "provides_event_level_q_over_p": False,
                "reason": "FASER muons are MIPs; calorimeter energy is not a momentum spectrometer",
            },
            "timing_scintillator_veto": {
                "in_lto_measurement_set": False,
                "provides_event_level_q_over_p": False,
            },
            "full_track_ckf": {
                "shares_target_measurements": True,
                "admissible": False,
            },
        },
        "by_target": packed,
        "independence_judged_by_measurement_identity": True,
    }


def audit_inventory_v2(graph: Mapping[str, Any]) -> dict[str, Any]:
    candidates = [
        _candidate(
            family="A_beam_or_production",
            name="mc_particle_gun_energy",
            directions=["q_over_p"],
            prior_level=POPULATION_LEVEL,
            target_independent=True,
            admissible_for_real_data=False,
            uncertainty_semantics_defined=False,
            independence_from_surviving_measurements=True,
            independence_from_alignment_geometry=False,
            uses_truth=True,
            uses_target_measurement=False,
            event_or_population="population",
            depends_on_current_mc_particle_gun_truth=True,
            simulation_to_data_circularity=True,
            reason="current campaign is particle-gun; gun energy is truth, not a real-data beam prior",
        ),
        _candidate(
            family="A_beam_or_production",
            name="lhc_collision_or_fluka_spectrum",
            directions=["q_over_p"],
            prior_level=POPULATION_LEVEL,
            target_independent=True,
            admissible_for_real_data=False,
            uncertainty_semantics_defined=False,
            independence_from_surviving_measurements=True,
            independence_from_alignment_geometry=False,
            uses_truth=True,
            uses_target_measurement=False,
            event_or_population="population",
            depends_on_current_mc_particle_gun_truth=False,
            simulation_to_data_circularity=True,
            could_bias_alignment_estimator=True,
            reason=(
                "no registered real-data p(q/p) contract exists; a FLUKA/LHC spectrum "
                "would be simulation-derived, population-level only, and acceptance "
                "still depends on geometry"
            ),
        ),
        _candidate(
            family="A_beam_or_production",
            name="real_data_fixed_energy_beam",
            directions=["q_over_p"],
            prior_level=EVENT_LEVEL,
            target_independent=True,
            admissible_for_real_data=False,
            uncertainty_semantics_defined=False,
            independence_from_surviving_measurements=True,
            independence_from_alignment_geometry=True,
            uses_truth=False,
            uses_target_measurement=False,
            event_or_population="event",
            reason="FASER collision/neutrino muons are not a fixed-energy test beam",
        ),
        _candidate(
            family="B_source_only",
            name="source_only_ift_tracking",
            directions=["x", "y", "tx", "ty"],
            prior_level=EVENT_LEVEL,
            target_independent=True,
            admissible_for_real_data=True,
            uncertainty_semantics_defined=False,
            independence_from_surviving_measurements=False,
            independence_from_alignment_geometry=False,
            uses_truth=False,
            uses_target_measurement=False,
            can_measure_q_over_p=False,
            already_in_surviving_likelihood=True,
            reason=(
                "station 0 never crosses the magnet, so it cannot provide q/p; "
                "its hits are already inside L_surviving_measurements for every LTO target"
            ),
        ),
        _candidate(
            family="C_independent_downstream",
            name="other_tracker_stations_not_in_surviving_set",
            directions=["y", "tx", "q_over_p"],
            prior_level=EVENT_LEVEL,
            target_independent=False,
            admissible_for_real_data=False,
            uncertainty_semantics_defined=False,
            independence_from_surviving_measurements=False,
            independence_from_alignment_geometry=False,
            uses_truth=False,
            uses_target_measurement=False,
            unused_tracker_stations=graph.get("by_target"),
            reason="the only tracker station not in a given LTO likelihood is the excluded target itself",
        ),
        _candidate(
            family="C_independent_downstream",
            name="calorimeter_or_timing_or_veto",
            directions=["q_over_p"],
            prior_level=EVENT_LEVEL,
            target_independent=True,
            admissible_for_real_data=False,
            uncertainty_semantics_defined=False,
            independence_from_surviving_measurements=True,
            independence_from_alignment_geometry=False,
            uses_truth=False,
            uses_target_measurement=False,
            provides_event_level_q_over_p=False,
            reason="no external FASER detector in this campaign is a momentum spectrometer for MIP muons",
        ),
        _candidate(
            family="C_independent_downstream",
            name="independent_spectrometer_reconstruction",
            directions=["q_over_p"],
            prior_level=EVENT_LEVEL,
            target_independent=False,
            admissible_for_real_data=False,
            uncertainty_semantics_defined=False,
            independence_from_surviving_measurements=False,
            independence_from_alignment_geometry=False,
            uses_truth=False,
            uses_target_measurement=False,
            reason="the only spectrometer is tracker plus the 0–1 magnet, which is the LTO measurement system",
        ),
        _candidate(
            family="D_full_track_ckf",
            name="full_track_ckf_qoverp",
            directions=["q_over_p"],
            prior_level=EVENT_LEVEL,
            target_independent=False,
            admissible_for_real_data=False,
            uncertainty_semantics_defined=False,
            independence_from_surviving_measurements=False,
            independence_from_alignment_geometry=False,
            uses_truth=False,
            uses_target_measurement=True,
            forbidden=True,
            reason="contains target-station measurements",
        ),
        _candidate(
            family="D_full_track_ckf",
            name="official_wb107_cin",
            directions=["x", "y", "tx", "ty", "q_over_p"],
            prior_level=EVENT_LEVEL,
            target_independent=False,
            admissible_for_real_data=False,
            uncertainty_semantics_defined=False,
            independence_from_surviving_measurements=False,
            independence_from_alignment_geometry=False,
            uses_truth=False,
            uses_target_measurement=True,
            forbidden=True,
            reason="full-track KF front state; frozen inadmissible",
        ),
        _candidate(
            family="E_empirical_or_closure",
            name="truth_residual_or_chi2_or_seed_scale",
            directions=["y", "tx", "q_over_p"],
            prior_level=POPULATION_LEVEL,
            target_independent=False,
            admissible_for_real_data=False,
            uncertainty_semantics_defined=False,
            independence_from_surviving_measurements=False,
            independence_from_alignment_geometry=False,
            uses_truth=True,
            uses_target_measurement=False,
            closure_derived=True,
            seed_scale_selected=True,
            forbidden=True,
            reason="truth residual, chi2 closure, 0.1/1/10 scan, and gate-tuned widths are forbidden",
        ),
        _candidate(
            family="E_empirical_or_closure",
            name="arbitrary_gaussian_or_current_seed_covariance",
            directions=["y", "tx", "q_over_p"],
            prior_level=POPULATION_LEVEL,
            target_independent=False,
            admissible_for_real_data=False,
            uncertainty_semantics_defined=False,
            independence_from_surviving_measurements=False,
            independence_from_alignment_geometry=False,
            uses_truth=False,
            uses_target_measurement=False,
            arbitrary_gaussian=True,
            forbidden=True,
            reason="an uninformative or hand-chosen Gaussian is not a physical prior contract",
        ),
    ]
    admissible = [item for item in candidates if item["admissible"]]
    by_direction = {name: [] for name in WEAK_DIRECTIONS}
    for item in candidates:
        for direction in item.get("directions") or []:
            if direction in by_direction and item["admissible"]:
                by_direction[direction].append(item["name"])
    return {
        "prior_introduced": False,
        "authorized_as_production_prior": False,
        "full_track_CKF_qoverp": "inadmissible",
        "WB107_Cin": "inadmissible",
        "n_candidates": len(candidates),
        "n_admissible": len(admissible),
        "candidates": candidates,
        "admissible_names": [item["name"] for item in admissible],
        "admissible_by_weak_direction": {name: bool(by_direction[name]) for name in WEAK_DIRECTIONS},
        "admissible_candidate_names_by_weak_direction": by_direction,
    }


def audit_level_semantics(inventory: Mapping[str, Any]) -> dict[str, Any]:
    event_level = []
    population_level = []
    for item in inventory["candidates"]:
        record = {
            "name": item["name"],
            "prior_level": item.get("prior_level"),
            "admissible": item["admissible"],
            "admissible_for_real_data": item["admissible_for_real_data"],
        }
        if item.get("prior_level") == EVENT_LEVEL:
            event_level.append(record)
        else:
            population_level.append(record)
    return {
        "event_level_measurement_prior": event_level,
        "population_level_physics_prior": population_level,
        "levels_must_not_be_mixed": True,
        "population_prior_is_not_an_event_measurement": True,
        "if_population_prior_were_later_allowed": {
            "source": "not_established",
            "applicable_data_domain": "not_established",
            "needs_nuisance_hyperparameters": True,
            "could_bias_alignment_estimator": True,
            "independent_of_geometry_to_be_estimated": False,
            "tuning_performed": False,
        },
        "n_admissible_event_level": sum(1 for item in event_level if item["admissible"]),
        "n_admissible_population_level": sum(1 for item in population_level if item["admissible"]),
    }


def audit_likelihood_contract(
    inventory: Mapping[str, Any],
    requirements: Mapping[str, Any],
) -> dict[str, Any]:
    any_admissible = bool(inventory.get("n_admissible"))
    return {
        "instantiated": False,
        "prior_covariance_added_to_wb109_cin": False,
        "formula": "L(theta) = L_surviving_measurements(theta) * pi_independent(theta_weak)",
        "theta": ["x", "y", "tx", "ty", "q_over_p"],
        "measurement_supported_directions": list(SUPPORTED_DIRECTIONS),
        "weak_directions": list(WEAK_DIRECTIONS),
        "weak_directions_deleted": False,
        "prior_distribution_source": None if not any_admissible else "see_admissible_candidates",
        "independence_assumptions": (
            "pi_independent must be disjoint in measurement identity from "
            "L_surviving_measurements and independent of the geometry being estimated"
        ),
        "normalization": "not_instantiated",
        "chart": "native_bound_mev_at_source_plane",
        "units": ["mm", "mm", "1", "1", "1/MeV"],
        "reference_surface": "source_station_plane_z=-1860.15mm",
        "admissible_prior_exists": any_admissible,
        "q_over_p_remains_explicit_nuisance": True,
        "wb112_classes": requirements["frozen_direction_class"],
    }


def audit_prior_influence(inventory: Mapping[str, Any]) -> dict[str, Any]:
    if inventory.get("n_admissible"):
        raise TargetIndependentPriorError("admissible prior found but influence smoke is not authorized here")
    return {
        "executed": False,
        "reason": "no_admissible_prior",
        "posterior_mean_sensitivity": "unavailable",
        "posterior_covariance_sensitivity": "unavailable",
        "information_contribution_from_likelihood": "unavailable",
        "information_contribution_from_prior": "unavailable",
        "truth_used_to_choose_prior": False,
        "arbitrary_gaussian_smoke_forbidden": True,
    }


def audit_focus(
    config: Mapping[str, Any],
    sample: Mapping[str, Any],
    lto_by_key: Mapping[tuple[Any, ...], Mapping[str, Any]],
    wb112: Mapping[str, Any],
) -> dict[str, Any]:
    wanted = _focus_keys(config)
    rows = []
    for official in sample["contracted"]:
        identity = _identity_triple(official)
        if identity not in wanted:
            continue
        lto = lto_by_key.get(_row_key(official))
        if lto is None:
            continue
        cat_row = None
        for item in (wb112["catastrophic"].get("rows") or []):
            if (
                str(item.get("source_id")) == identity[0]
                and int(item.get("event_id", -1)) == identity[2]
                and int(item.get("target_station", -1)) == int(official.get("target_station", -1))
            ):
                cat_row = item
                break
        q_gain = None if cat_row is None else cat_row.get("qoverp_information_gain_ratio")
        rows.append(
            {
                "source_id": official.get("source_id"),
                "run_id": official.get("run_id"),
                "event_id": official.get("event_id"),
                "target_station": official.get("target_station"),
                "n_measurements_in_fit": lto.get("n_measurements_in_fit"),
                "used_station_ids": lto.get("used_station_ids"),
                "q_over_p_per_mev": lto.get("q_over_p_per_mev"),
                "qoverp_information_gain_ratio": q_gain,
                "measurement_likelihood_has_almost_no_qoverp_information": bool(
                    q_gain is not None and float(q_gain) <= 1.25
                ),
                "admissible_prior_would_only_take_over_qoverp": True,
                "posterior_would_look_stable_but_prior_dominated": True,
                "special_cased": False,
                "deleted": False,
                "downweighted": False,
                "truth_replaced": False,
            }
        )
    return {
        "retained_all_events": True,
        "special_prior_used": False,
        "n_reported": len(rows),
        "rows": rows,
        "qoverp_is_prior_dominated_latent_if_any_prior_were_forced": True,
        "do_not_write_as_measured_momentum": True,
    }


def decide_case(inventory: Mapping[str, Any]) -> dict[str, Any]:
    if inventory.get("seed_scale_selected"):
        refuse_seed_scale_choice()
    if inventory.get("prior_introduced"):
        refuse_prior_introduction()
    denom = inventory.get("denominator") or {}
    admissible = inventory.get("admissible_by_weak_direction") or {}
    n_ok = sum(1 for name in WEAK_DIRECTIONS if admissible.get(name))
    q_ok = bool(admissible.get("q_over_p"))
    independence_unresolved = bool(inventory.get("independence_unresolved"))
    if denom.get("frozen_denominator_holds") is False:
        primary = "wb103_denominator_redefined"
        verdict = "FAIL"
        active: list[str] = []
    elif independence_unresolved and n_ok > 0:
        primary = CASE_D
        verdict = "FAIL"
        active = ["D"]
    elif n_ok == 3 and q_ok:
        primary = CASE_A
        verdict = "PASS"
        active = ["A"]
    elif n_ok == 0:
        primary = CASE_B
        verdict = "FAIL"
        active = ["B"]
    elif 0 < n_ok < 3:
        primary = CASE_C
        verdict = "DIAGNOSED"
        active = ["C"]
    else:
        primary = CASE_E
        verdict = "DIAGNOSED"
        active = ["E"]
    return {
        "verdict": verdict,
        "decision": primary,
        "primary_case": primary,
        "active_mechanisms": active,
        "next_step": (
            "reconstruct_measurement_times_prior_likelihood_and_redo_b14"
            if primary == CASE_A
            else "b14m_profiled_marginalized_weak_nuisance_likelihood"
            if primary == CASE_B
            else "partial_prior_plus_remaining_explicit_nuisance"
            if primary == CASE_C
            else "refuse_prior_until_independence_established"
            if primary == CASE_D
            else "keep_mixed_prior_admissibility"
        ),
        "b14m_authorized": primary == CASE_B,
        "b15_authorized": False,
        "transport_covariance_validated": False,
        "measurement_model_v2_authorized": False,
        "measurement_model_v2_entered": False,
        "lto_cin_contract_established": False,
        "prior_contract_established": primary == CASE_A,
        "prior_introduced": False,
        "seed_scale_selected_from_closure": False,
        "q_over_p_deleted": False,
        "q_over_p_fixed": False,
        "focus_identity_retained": True,
        "do_not_force_5d_lto_covariance": primary in (CASE_B, CASE_C, CASE_D, CASE_E),
    }


def inventory_and_audit(config: Mapping[str, Any]) -> dict[str, Any]:
    dumps = load_lto_dumps(config)
    sample = load_contracted_sample(config)
    if not any(identity_key(row) in _focus_keys(config) for row in sample["contracted"]):
        refuse_focus_drop()
    frozen_ok = bool(
        int(sample["n_raw"]) == FROZEN_N_RAW
        and int(sample["n_ineligible"]) == FROZEN_N_INELIGIBLE
        and len(sample["contracted"]) == FROZEN_N_CONTRACTED
        and len(sample["official_events"]) == FROZEN_N_OFFICIAL_PAIRS
    )
    lto_by_key = {_row_key(row): row for row in dumps["rows"]}
    wb112 = {
        "contract": _load_wb112_json(config, "contract_path"),
        "information_gain": _load_wb112_json(config, "information_gain_path"),
        "directional": _load_wb112_json(config, "directional_path"),
        "catastrophic": _load_wb112_json(config, "catastrophic_path"),
    }
    requirements = audit_requirements(config, wb112)
    graph = audit_dependency_graph(sample, lto_by_key)
    inventory = audit_inventory_v2(graph)
    levels = audit_level_semantics(inventory)
    contract = audit_likelihood_contract(inventory, requirements)
    influence = audit_prior_influence(inventory)
    focus = audit_focus(config, sample, lto_by_key, wb112)
    return {
        "dumps": {
            "present_sources": dumps["present_sources"],
            "missing_sources": dumps["missing_sources"],
            "dumps_present": dumps["dumps_present"],
        },
        "denominator": {
            "n_raw": sample["n_raw"],
            "n_ineligible": sample["n_ineligible"],
            "n_contracted": len(sample["contracted"]),
            "n_official_pairs": len(sample["official_events"]),
            "frozen_denominator_holds": frozen_ok,
        },
        "requirements": requirements,
        "dependency_graph": graph,
        "inventory": inventory,
        "levels": levels,
        "likelihood_contract": contract,
        "influence": influence,
        "focus": focus,
        "admissible_by_weak_direction": inventory["admissible_by_weak_direction"],
        "independence_unresolved": False,
        "present_sources": sample["present_sources"],
        "missing_sources": dumps["missing_sources"],
        "n_contracted": len(sample["contracted"]),
        "n_official_pairs": len(sample["official_events"]),
        "n_raw": sample["n_raw"],
        "n_ineligible": sample["n_ineligible"],
        "seed_scale_selected": False,
        "prior_introduced": False,
    }


def decide(inventory: Mapping[str, Any], inherited: Mapping[str, Any]) -> dict[str, Any]:
    mechanism = decide_case(inventory)
    if mechanism["measurement_model_v2_entered"]:
        refuse_measurement_model_v2()
    if mechanism["b15_authorized"]:
        refuse_b15()
    if mechanism["prior_introduced"]:
        refuse_prior_introduction()
    if mechanism["q_over_p_deleted"]:
        refuse_qoverp_deletion()
    if mechanism["q_over_p_fixed"]:
        refuse_qoverp_fix()
    return {
        **mechanism,
        "geometry_write_allowed": False,
        "official_input_scope": "contract_eligible",
        "raw_ckf_used": False,
        "inherited_wb112_decision_sha256": inherited["workbook_112"]["decision_sha256"],
        "inherited_wb111_decision_sha256": inherited["workbook_111"]["decision_sha256"],
        "inherited_wb110_decision_sha256": inherited["workbook_110"]["decision_sha256"],
        "inherited_wb109_decision_sha256": inherited["workbook_109"]["decision_sha256"],
        "inherited_wb103_contract_sha256": inherited["workbook_103"]["contract_sha256"],
        "wb96_through_wb112_rewritten": False,
    }
