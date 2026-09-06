"""Stage B / Task A: physical q/p uncertainty semantics.

Inventories reconstruction-chain q/p sources already exported on the frozen
WB87 construction/validation split.  Dummy SegmentFit q/p is not physical.
Truth q/p is not a real-data solution.  A mean CKF momentum without
exported covariance is not a complete prior.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from alignment.operating_protocol_v1_final_closure import (
    project_root,
    resolve_under_root,
    sha256_file,
)
from datasets.access_policy import AccessPolicyError, AccessScope, authorize_path
from datasets.transport_contract import MEV_PER_GEV

SCHEMA_VERSION = "qoverp-semantics-contract-v1"
DEFAULT_CONFIG = "configs/qoverp_semantics_contract_v1.yaml"
TASK = "SB-A"
WORKBOOK = 95

DECISION_ESTABLISHED = "physical_qoverp_semantics_established"
DECISION_NOT_ESTABLISHED = "physical_qoverp_semantics_not_established"
MECHANISM_NOT_EXPORTED = "reconstruction_chain_qoverp_uncertainty_not_exported"

SEGMENTFIT_DUMMY_QOVERP_PER_MEV = 1.0e-5
SEGMENTFIT_DUMMY_VARIANCE_SCALE = 5.0e4
SEGMENTFIT_DUMMY_MOMENTUM_MEV = 1.0e5

NATIVE_ATHENA_STATE = ("loc1", "loc2", "phi", "theta", "q_over_p")
EXPORT_TRACKLET_STATE = ("x_mm", "y_mm", "tx", "ty")
QOVERP_NATIVE_UNIT = "per_MeV"

TRACKLET_QOP_BRANCHES = (
    "q_over_p_per_mev",
    "q_over_p_from_momentum_per_mev",
    "q_over_p_variance_per_mev2",
    "has_q_over_p_covariance",
    "truth_q_over_p_per_mev",
    "station_id",
)
CKF_MEAN_BRANCHES = ("Track_p0", "Track_charge")
CKF_FORBIDDEN_COV_MARKERS = ("cov", "variance", "q_over_p", "qoverp")


class QoverPSemanticsError(ValueError):
    """Raised when the q/p contract is illegal or a forbidden repair is requested."""


def dummy_segmentfit_variance_per_mev2(
    q_over_p_per_mev: float = SEGMENTFIT_DUMMY_QOVERP_PER_MEV,
) -> float:
    return float(SEGMENTFIT_DUMMY_VARIANCE_SCALE) * float(q_over_p_per_mev) ** 2


def q_over_p_per_mev_from_charge_and_p(
    charge: float, p_mev: float
) -> float:
    if p_mev == 0.0:
        raise QoverPSemanticsError("momentum must be non-zero")
    return float(charge) / float(p_mev)


def state_definition() -> dict[str, Any]:
    return {
        "native_athena": list(NATIVE_ATHENA_STATE),
        "q_over_p_native_unit": QOVERP_NATIVE_UNIT,
        "q_over_p_signed": True,
        "export_tracklet_state": list(EXPORT_TRACKLET_STATE),
        "magnetic_transport_requires_physical_qoverp": True,
        "mev_per_gev": MEV_PER_GEV,
    }


def refuse_truth_as_real_data_solution() -> None:
    raise QoverPSemanticsError("truth q/p is not a real-data solution")


def refuse_dummy_as_physical() -> None:
    raise QoverPSemanticsError("SegmentFit dummy q/p is not a physical momentum uncertainty")


def refuse_delete_qoverp_column() -> None:
    raise QoverPSemanticsError("deleting the q/p column is not the final scheme")


def refuse_covariance_rescale() -> None:
    raise QoverPSemanticsError("covariance rescale is not a q/p semantics repair")


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = resolve_under_root(project_root(), str(path or DEFAULT_CONFIG))
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != SCHEMA_VERSION:
        raise QoverPSemanticsError(f"schema_version must be {SCHEMA_VERSION}")
    if config.get("task") != TASK:
        raise QoverPSemanticsError(f"task must be {TASK}")
    if int(config.get("workbook", -1)) != WORKBOOK:
        raise QoverPSemanticsError(f"workbook must be {WORKBOOK}")
    required_false = (
        "geometry_write_allowed",
        "held_out_accessed",
        "real_data_alignment_authorized",
        "measurement_model_validated",
        "do_not_enter_alignment",
        "do_not_enter_measurement_model_v2",
        "do_not_start_new_source_campaign",
        "do_not_submit_eighteen_source_jobs",
    )
    for key in required_false:
        if key.startswith("do_not_"):
            if bool(config.get(key, False)) is not True:
                raise QoverPSemanticsError(f"{key} must be true")
        elif bool(config.get(key, True)):
            raise QoverPSemanticsError(f"{key} must be false")
    if bool(config.get("do_not_rescale_covariance", False)) is not True:
        raise QoverPSemanticsError("covariance rescale is forbidden")
    if bool(config.get("do_not_use_truth_q_over_p_as_real_data_solution", False)) is not True:
        raise QoverPSemanticsError("truth q/p is not a real-data solution")
    if bool(config.get("do_not_delete_qoverp_column_as_final_scheme", False)) is not True:
        raise QoverPSemanticsError("deleting the q/p column is forbidden")
    if bool(config.get("unconstrained_tracker_only_stopped", False)) is not True:
        raise QoverPSemanticsError("unconstrained tracker-only must stay stopped")
    dummy = config["segmentfit_dummy"]
    if abs(float(dummy["qoverp_per_mev"]) - SEGMENTFIT_DUMMY_QOVERP_PER_MEV) > 0.0:
        raise QoverPSemanticsError("config dummy q/p must match SegmentFit GetState")
    if abs(float(dummy["variance_scale"]) - SEGMENTFIT_DUMMY_VARIANCE_SCALE) > 0.0:
        raise QoverPSemanticsError("config dummy variance scale must match SegmentFit GetState")
    return dict(config)


def _expect_sha(path: Path, expected: str, label: str) -> str:
    digest = sha256_file(path)
    if digest != expected:
        raise QoverPSemanticsError(f"{label} hash mismatch")
    return digest


def inherit_frozen_stage(config: Mapping[str, Any]) -> dict[str, Any]:
    inherited: dict[str, Any] = {}
    spec87 = config["inheritance"]["workbook_87"]
    decision87 = json.loads(
        resolve_under_root(project_root(), spec87["decision_path"]).read_text(encoding="utf-8")
    )
    _expect_sha(
        resolve_under_root(project_root(), spec87["config_path"]),
        spec87["config_sha256"],
        "WB87 config",
    )
    _expect_sha(
        resolve_under_root(project_root(), spec87["decision_path"]),
        spec87["decision_sha256"],
        "WB87 decision",
    )
    if decision87.get("decision") != spec87["frozen_decision"]:
        raise QoverPSemanticsError("WB87 decision must stay frozen")
    inherited["workbook_87"] = {
        "decision": decision87["decision"],
        "mechanism": spec87["frozen_mechanism"],
        "config_sha256": spec87["config_sha256"],
        "decision_sha256": spec87["decision_sha256"],
    }

    spec93 = config["inheritance"]["workbook_93"]
    decision93 = json.loads(
        resolve_under_root(project_root(), spec93["decision_path"]).read_text(encoding="utf-8")
    )
    _expect_sha(
        resolve_under_root(project_root(), spec93["config_path"]),
        spec93["config_sha256"],
        "WB93 config",
    )
    _expect_sha(
        resolve_under_root(project_root(), spec93["decision_path"]),
        spec93["decision_sha256"],
        "WB93 decision",
    )
    if decision93.get("verdict") != spec93["frozen_verdict"]:
        raise QoverPSemanticsError("WB93 verdict must stay frozen")
    if bool(decision93.get("contract_established_for_t12", True)):
        raise QoverPSemanticsError("T11 contract must remain unestablished")
    inherited["workbook_93"] = {
        "verdict": decision93["verdict"],
        "contract_established_for_t12": False,
        "config_sha256": spec93["config_sha256"],
        "decision_sha256": spec93["decision_sha256"],
    }

    spec94 = config["inheritance"]["workbook_94"]
    decision94 = json.loads(
        resolve_under_root(project_root(), spec94["decision_path"]).read_text(encoding="utf-8")
    )
    _expect_sha(
        resolve_under_root(project_root(), spec94["config_path"]),
        spec94["config_sha256"],
        "WB94 config",
    )
    _expect_sha(
        resolve_under_root(project_root(), spec94["decision_path"]),
        spec94["decision_sha256"],
        "WB94 decision",
    )
    if decision94.get("verdict") != spec94["frozen_verdict"]:
        raise QoverPSemanticsError("WB94 verdict must stay frozen")
    if bool(decision94.get("conditional_go", True)):
        raise QoverPSemanticsError("T12 conditional_go must stay false")
    inherited["workbook_94"] = {
        "verdict": decision94["verdict"],
        "conditional_go": False,
        "unconstrained_tracker_only_stopped": True,
        "config_sha256": spec94["config_sha256"],
        "decision_sha256": spec94["decision_sha256"],
    }
    return inherited


def classify_candidate(record: Mapping[str, Any]) -> dict[str, Any]:
    """Classify one q/p source.  Truth and dummy cannot become physical."""
    if bool(record.get("is_truth", False)):
        return {
            "role": "rejected",
            "reason": "truth_qoverp_forbidden_as_real_data_solution",
            "complete_physical_prior": False,
        }
    if bool(record.get("is_dummy_segmentfit", False)):
        return {
            "role": "rejected",
            "reason": "dummy_segmentfit_not_physical",
            "complete_physical_prior": False,
        }
    has_estimate = bool(record.get("has_estimate", False))
    has_uncertainty = bool(record.get("has_uncertainty", False))
    has_correlation = bool(record.get("has_correlation", False))
    reconstruction = bool(record.get("reconstruction_chain", False))
    provenance = bool(record.get("provenance_complete", False))
    if not reconstruction:
        return {
            "role": "rejected",
            "reason": "not_reconstruction_chain",
            "complete_physical_prior": False,
        }
    if has_estimate and not has_uncertainty:
        return {
            "role": "incomplete",
            "reason": "mean_state_only_uncertainty_not_exported",
            "complete_physical_prior": False,
        }
    if has_estimate and has_uncertainty and not has_correlation:
        return {
            "role": "incomplete",
            "reason": "qoverp_correlation_not_exported",
            "complete_physical_prior": False,
        }
    complete = (
        has_estimate
        and has_uncertainty
        and has_correlation
        and provenance
        and not bool(record.get("is_truth", False))
        and not bool(record.get("is_dummy_segmentfit", False))
    )
    if complete:
        return {
            "role": "physical_prior_candidate",
            "reason": "estimate_uncertainty_and_correlation_from_reconstruction",
            "complete_physical_prior": True,
        }
    return {
        "role": "incomplete",
        "reason": "missing_required_qoverp_fields",
        "complete_physical_prior": False,
    }


def decide(candidates: list[Mapping[str, Any]]) -> dict[str, Any]:
    classified = [dict(item) | classify_candidate(item) for item in candidates]
    physical = [item for item in classified if item.get("complete_physical_prior")]
    established = bool(physical)
    return {
        "kind": "qoverp_semantics_contract",
        "task": TASK,
        "workbook": WORKBOOK,
        "verdict": "PASS" if established else "FAIL",
        "decision": DECISION_ESTABLISHED if established else DECISION_NOT_ESTABLISHED,
        "mechanism": None if established else MECHANISM_NOT_EXPORTED,
        "state_definition": state_definition(),
        "units": {
            "q_over_p": QOVERP_NATIVE_UNIT,
            "q_over_p_variance": "per_MeV2",
            "momentum": "MeV",
            "charge": "e",
        },
        "covariance_source": {
            "segmentfit_tracklet": "dummy_diagonal_qoverp_from_GetState",
            "ckf_track": "mean_p_only_covariance_not_exported",
            "truth": "forbidden_as_real_data_solution",
            "physical_prior": None if not established else "reconstruction_chain",
        },
        "uncertainty_propagation": {
            "native_to_export": "SegmentFit 4D fit -> GetState 5x5 with dummy q/p",
            "ckf_to_ntuple": "CKFTrackCollection mean p dumped; 5x5 discarded",
            "magnetic_transport": "requires physical q/p uncertainty; not supplied",
            "rescale_allowed": False,
        },
        "validation_split": {
            "construction": "WB87 construction_source_ids",
            "validation": "WB87 validation_source_ids",
            "file_level_disjoint": True,
            "new_source_campaign": False,
        },
        "n_candidates": len(classified),
        "n_complete_physical_priors": len(physical),
        "candidates": classified,
        "contract_established_for_process_noise": established,
        "measurement_model_v2_entered": False,
        "geometry_write_allowed": False,
        "held_out_accessed": False,
        "truth_qoverp_used_as_real_data_solution": False,
        "qoverp_column_deleted_as_final_scheme": False,
        "covariance_tuned_to_chi2": False,
        "new_source_campaign_started": False,
        "unconstrained_tracker_only_stopped": True,
    }


def _source_refit_dir(config: Mapping[str, Any], source_id: str) -> Path:
    template = config["mc_data"]["source_root_template"]
    return resolve_under_root(project_root(), template.format(source_id=source_id))


def _flatten_jagged(values: Any) -> np.ndarray:
    flat: list[float] = []
    for item in values:
        array = np.asarray(item, dtype=np.float64).reshape(-1)
        flat.extend(float(x) for x in array.tolist())
    return np.asarray(flat, dtype=np.float64)


def _finite_stats(values: np.ndarray) -> dict[str, int | float | None]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = array[np.isfinite(array)]
    result: dict[str, int | float | None] = {
        "count": int(array.size),
        "finite_count": int(finite.size),
        "min": None,
        "median": None,
        "max": None,
    }
    if finite.size:
        result.update(
            {
                "min": float(np.min(finite)),
                "median": float(np.median(finite)),
                "max": float(np.max(finite)),
            }
        )
    return result


def _authorize_existing(path: Path, split: str) -> Path:
    authorized = authorize_path(path, AccessScope.DEVELOPMENT_VALIDATION, split=split)
    if not authorized.is_file():
        raise AccessPolicyError(f"required reconstruction export is missing: {authorized}")
    return authorized


def inventory_tracklets(path: Path, tree: str, config: Mapping[str, Any]) -> dict[str, Any]:
    import uproot

    dummy_q = float(config["segmentfit_dummy"]["qoverp_per_mev"])
    dummy_var = dummy_segmentfit_variance_per_mev2(dummy_q)
    q_tol = float(config["acceptance"]["dummy_qoverp_match_tolerance_per_mev"])
    v_tol = float(config["acceptance"]["dummy_variance_match_tolerance_per_mev2"])
    with uproot.open(path) as handle:
        arrays = handle[tree].arrays(list(TRACKLET_QOP_BRANCHES), library="np")
    native = np.asarray(arrays["q_over_p_per_mev"], dtype=np.float64)
    variance = np.asarray(arrays["q_over_p_variance_per_mev2"], dtype=np.float64)
    from_momentum = np.asarray(arrays["q_over_p_from_momentum_per_mev"], dtype=np.float64)
    truth = np.asarray(arrays["truth_q_over_p_per_mev"], dtype=np.float64)
    n = int(native.size)
    dummy_q_match = int(np.count_nonzero(np.isfinite(native) & (np.abs(native - dummy_q) <= q_tol)))
    dummy_var_match = int(
        np.count_nonzero(np.isfinite(variance) & (np.abs(variance - dummy_var) <= v_tol))
    )
    return {
        "kind": "segmentfit_tracklet_qoverp",
        "path": str(path),
        "n_tracklets": n,
        "q_over_p_per_mev": _finite_stats(native),
        "q_over_p_variance_per_mev2": _finite_stats(variance),
        "q_over_p_from_momentum_per_mev": _finite_stats(from_momentum),
        "truth_q_over_p_per_mev": _finite_stats(truth),
        "n_exact_dummy_qoverp": dummy_q_match,
        "n_exact_dummy_variance": dummy_var_match,
        "all_qoverp_are_dummy": n > 0 and dummy_q_match == n,
        "all_variance_are_dummy": n > 0 and dummy_var_match == n,
        "n_unique_native_qoverp": int(np.unique(np.round(native[np.isfinite(native)], 18)).size),
        "has_qoverp_correlation_export": False,
        "is_dummy_segmentfit": True,
        "is_truth": False,
        "has_estimate": True,
        "has_uncertainty": True,
        "has_correlation": False,
        "reconstruction_chain": True,
        "provenance_complete": True,
    }


def inventory_ckf_mean(path: Path, tree: str) -> dict[str, Any]:
    import uproot

    with uproot.open(path) as handle:
        names = [key.split(";")[0] for key in handle[tree].keys()]
        missing = [name for name in CKF_MEAN_BRANCHES if name not in names]
        cov_like = [
            name
            for name in names
            if name.startswith("Track_")
            and any(marker in name.lower() for marker in CKF_FORBIDDEN_COV_MARKERS)
        ]
        if missing:
            return {
                "kind": "ckf_track_mean_qoverp",
                "path": str(path),
                "available": False,
                "missing_branches": missing,
                "is_dummy_segmentfit": False,
                "is_truth": False,
                "has_estimate": False,
                "has_uncertainty": False,
                "has_correlation": False,
                "reconstruction_chain": True,
                "provenance_complete": False,
            }
        arrays = handle[tree].arrays(list(CKF_MEAN_BRANCHES), library="np")
    momentum = _flatten_jagged(arrays["Track_p0"])
    charge = _flatten_jagged(arrays["Track_charge"])
    n = int(min(momentum.size, charge.size))
    momentum = momentum[:n]
    charge = charge[:n]
    finite = np.isfinite(momentum) & np.isfinite(charge) & (np.abs(momentum) > 0.0)
    qoverp = np.full(n, np.nan, dtype=np.float64)
    qoverp[finite] = charge[finite] / momentum[finite]
    dummy_p = np.abs(momentum - SEGMENTFIT_DUMMY_MOMENTUM_MEV) <= 1.0
    return {
        "kind": "ckf_track_mean_qoverp",
        "path": str(path),
        "available": True,
        "collection": "CKFTrackCollection",
        "n_tracks": n,
        "n_finite_charge_over_p": int(np.count_nonzero(finite)),
        "p0_mev": _finite_stats(momentum),
        "charge": _finite_stats(charge),
        "q_over_p_from_mean_p_per_mev": _finite_stats(qoverp),
        "n_unique_p0": int(np.unique(np.round(momentum[np.isfinite(momentum)], 6)).size),
        "n_near_dummy_100gev": int(np.count_nonzero(dummy_p)),
        "covariance_like_track_branches": cov_like,
        "has_qoverp_correlation_export": False,
        "is_dummy_segmentfit": False,
        "is_truth": False,
        "has_estimate": n > 0 and int(np.count_nonzero(finite)) > 0,
        "has_uncertainty": False,
        "has_correlation": False,
        "reconstruction_chain": True,
        "provenance_complete": True,
    }


def inventory_split(config: Mapping[str, Any], split: str) -> dict[str, Any]:
    if split == "construction":
        source_ids = list(config["mc_data"]["construction_source_ids"])
        access_split = "train"
    elif split == "validation":
        source_ids = list(config["mc_data"]["validation_source_ids"])
        access_split = "validation"
    else:
        raise QoverPSemanticsError(f"unknown split: {split}")
    sources = []
    for source_id in source_ids:
        refit = _source_refit_dir(config, source_id)
        tracklets = _authorize_existing(
            refit / str(config["mc_data"]["tracklets_file"]), access_split
        )
        enhanced = _authorize_existing(
            refit / str(config["mc_data"]["enhanced_file"]), access_split
        )
        tracklet_inv = inventory_tracklets(
            tracklets, str(config["mc_data"]["tracklets_tree"]), config
        )
        ckf_inv = inventory_ckf_mean(enhanced, str(config["mc_data"]["enhanced_tree"]))
        sources.append(
            {
                "source_id": source_id,
                "split": split,
                "tracklets": tracklet_inv,
                "ckf_mean": ckf_inv,
            }
        )
    n_tracklets = sum(int(item["tracklets"]["n_tracklets"]) for item in sources)
    n_dummy = sum(int(item["tracklets"]["n_exact_dummy_qoverp"]) for item in sources)
    n_ckf = sum(int(item["ckf_mean"].get("n_tracks", 0)) for item in sources)
    return {
        "split": split,
        "n_sources": len(sources),
        "n_tracklets": n_tracklets,
        "n_dummy_tracklet_qoverp": n_dummy,
        "all_tracklet_qoverp_dummy": n_tracklets > 0 and n_dummy == n_tracklets,
        "n_ckf_mean_tracks": n_ckf,
        "sources": sources,
    }


def build_candidates(construction: Mapping[str, Any], validation: Mapping[str, Any]) -> list[dict[str, Any]]:
    dummy_complete = bool(construction["all_tracklet_qoverp_dummy"]) and bool(
        validation["all_tracklet_qoverp_dummy"]
    )
    ckf_tracks = int(construction["n_ckf_mean_tracks"]) + int(validation["n_ckf_mean_tracks"])
    return [
        {
            "name": "segmentfit_dummy_qoverp",
            "reconstruction_chain": True,
            "is_dummy_segmentfit": True,
            "is_truth": False,
            "has_estimate": True,
            "has_uncertainty": True,
            "has_correlation": False,
            "provenance_complete": True,
            "observed_on_both_splits": dummy_complete,
        },
        {
            "name": "ntuple_mode_1_or_2_truth_qoverp",
            "reconstruction_chain": False,
            "is_dummy_segmentfit": False,
            "is_truth": True,
            "has_estimate": True,
            "has_uncertainty": False,
            "has_correlation": False,
            "provenance_complete": True,
            "observed_on_both_splits": True,
        },
        {
            "name": "ckf_track_mean_momentum",
            "reconstruction_chain": True,
            "is_dummy_segmentfit": False,
            "is_truth": False,
            "has_estimate": ckf_tracks > 0,
            "has_uncertainty": False,
            "has_correlation": False,
            "provenance_complete": True,
            "n_tracks_construction_plus_validation": ckf_tracks,
        },
    ]


def calypso_provenance(config: Mapping[str, Any]) -> dict[str, Any]:
    spec = config["software_provenance"]
    root = Path(spec["calypso_root"])
    files = {
        "segmentfit_source": spec["segmentfit_source"],
        "ntuple_dumper_source": spec["ntuple_dumper_source"],
        "ntuple_dumper_header": spec["ntuple_dumper_header"],
    }
    hashed = {}
    for key, relative in files.items():
        path = root / relative
        hashed[key] = {
            "path": str(path),
            "exists": path.is_file(),
            "sha256": sha256_file(path) if path.is_file() else None,
        }
    return {
        "kind": "calypso_qoverp_provenance",
        "calypso_git_sha": spec["calypso_git_sha"],
        "athena_release": spec["athena_release"],
        "acts_version": spec["acts_version"],
        "historical_runtime_unverified": True,
        "segmentfit_dummy_qoverp_per_mev": SEGMENTFIT_DUMMY_QOVERP_PER_MEV,
        "segmentfit_dummy_variance_per_mev2": dummy_segmentfit_variance_per_mev2(),
        "segmentfit_dummy_correlations": 0.0,
        "ckf_collection": "CKFTrackCollection",
        "ckf_mean_exported": True,
        "ckf_covariance_exported": False,
        "files": hashed,
    }
