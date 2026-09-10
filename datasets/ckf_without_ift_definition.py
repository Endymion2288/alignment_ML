"""Yasu Stage 0: official CKFTrackCollectionWithoutIFT definition contract.

Pins the Calypso source path that writes the collection, the FaserSCT_ID
station map, and the machine checks that IFT measurements did not enter
the persisted 3-station fit.  LTO, 4-station CKF, dummy SegmentFit q/p,
and truth q/p cannot substitute for this collection.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from alignment.numerical_contract import is_spd
from alignment.operating_protocol_v1_final_closure import (
    project_root,
    resolve_under_root,
    sha256_file,
)
from datasets.access_policy import AccessScope, authorize_path
from datasets.qoverp_semantics import SEGMENTFIT_DUMMY_QOVERP_PER_MEV
from datasets.transport_contract import MEV_PER_GEV

SCHEMA_VERSION = "ckf-without-ift-definition-v1"
DEFAULT_CONFIG = "configs/ckf_without_ift_definition_v1.yaml"
TASK = "YASU-S0"
WORKBOOK = 117

SOURCE_COLLECTION = "CKFTrackCollectionWithoutIFT"
FOUR_STATION_COLLECTION = "CKFTrackCollection"
IFT_STATION_ID = 0
THREE_STATION_IDS = (1, 2, 3)
MASKED_LAYERS = (0, 1, 2, 3, 4, 5)
QOVERP_INDEX = 4

DECISION_ESTABLISHED = "ckf_without_ift_definition_established"
DECISION_NOT_ESTABLISHED = "ckf_without_ift_definition_not_established"
MECHANISM_COLLECTION_ABSENT = "collection_absent_from_xaod"
MECHANISM_DUMP_MISSING = "reconstruction_output_dump_not_materialized"
MECHANISM_IFT_LEAK = "ift_measurement_or_outlier_present"
MECHANISM_STATION_UNDECODED = "station_id_not_machine_decoded"
MECHANISM_NO_IFT_CLUSTERS = "ift_clusters_absent_so_exclusion_uninformative"
MECHANISM_DUMMY = "without_ift_qoverp_matches_segmentfit_dummy"
MECHANISM_TRUTH = "truth_qoverp_forbidden"
MECHANISM_LTO_SUBSTITUTE = "lto_used_as_without_ift_substitute"
MECHANISM_SOURCE_PIN_MISMATCH = "pinned_calypso_source_hash_mismatch"
MECHANISM_NO_5X5 = "without_ift_5x5_not_materialized"

STATION_MAP = {
    0: "interface_IFT",
    1: "upstream_S1",
    2: "central_S2",
    3: "downstream_S3",
}


class WithoutIftDefinitionError(ValueError):
    """Raised when the WithoutIFT definition contract is illegal."""


def layer_code(station: int, layer: int, side: int) -> int:
    return 6 * int(station) + 2 * int(layer) + int(side)


def ift_layer_codes() -> tuple[int, ...]:
    return tuple(layer_code(0, layer, side) for layer in (0, 1, 2) for side in (0, 1))


def calypso_source_contract() -> dict[str, Any]:
    return {
        "writer": "faser_reco.py CKF2Cfg instance CKF_woIFT",
        "output_collection": SOURCE_COLLECTION,
        "masked_layers": list(MASKED_LAYERS),
        "masked_layer_encoding": "6 * FaserSCT_ID.station + 2 * layer + side",
        "ift_layer_codes": list(ift_layer_codes()),
        "station_map": dict(STATION_MAP),
        "si_detector_element": {
            "isInterface": "station == 0",
            "isUpstream": "station == 1",
            "isCentral": "station == 2",
            "isDownstream": "station == 3",
        },
        "exclusion_path": [
            "CircleFitTrackSeedTool skips masked clusters when building sourceLinks and measurements",
            "CKF2 finder only sees those sourceLinks/measurements",
            "CreateTrkTrackTool writes only CKF sourceLink clusters onto the Trk::Track",
            "KalmanFitterTool.fit getMeasurementsFromTrack reads only measurementsOnTrack",
        ],
        "not_the_same_as": [
            "CKFTrackCollection four-station CKF",
            "WB109 LTO leaving out stations 1/2/3 while keeping IFT",
            "SegmentFit dummy q/p",
        ],
        "removeIFT_seed_flag_default": False,
        "exclusion_mechanism": "maskedLayers, not m_removeIFT",
        "native_state_after_create_trk_track": {
            "type": "Trk::CurvilinearParameters",
            "q_over_p_unit": "per_MeV",
            "acts_to_athena": "q/p and covariance row/col 4 multiplied by 1_MeV",
            "charge": "sign of converted q/p",
        },
    }


def state_definition() -> dict[str, Any]:
    return {
        "native_athena": ["loc1_mm", "loc2_mm", "phi", "theta", "q_over_p_per_mev"],
        "native_parameter_type": "Trk::CurvilinearParameters",
        "native_frame": "curvilinear_trackparameters",
        "q_over_p_native_unit": "per_MeV",
        "q_over_p_acts_unit": "per_GeV",
        "q_over_p_signed": True,
        "charge_sign": "sign_of_q_over_p",
        "loc_unit": "mm",
        "angle_unit": "rad",
        "mev_per_gev": MEV_PER_GEV,
        "covariance_dimension": 5,
    }


def refuse_truth_qoverp() -> None:
    raise WithoutIftDefinitionError("truth q/p is not a real-data solution")


def refuse_dummy_segmentfit() -> None:
    raise WithoutIftDefinitionError("dummy SegmentFit covariance is not a physical prior")


def refuse_lto_substitute() -> None:
    raise WithoutIftDefinitionError(
        "WB109 LTO is not CKFTrackCollectionWithoutIFT; LTO keeps IFT"
    )


def refuse_four_station_substitute() -> None:
    raise WithoutIftDefinitionError(
        "CKFTrackCollection is the 4-station collection, not WithoutIFT"
    )


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = resolve_under_root(project_root(), str(path or DEFAULT_CONFIG))
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != SCHEMA_VERSION:
        raise WithoutIftDefinitionError(f"schema_version must be {SCHEMA_VERSION}")
    if config.get("task") != TASK:
        raise WithoutIftDefinitionError(f"task must be {TASK}")
    if int(config.get("workbook", -1)) != WORKBOOK:
        raise WithoutIftDefinitionError(f"workbook must be {WORKBOOK}")
    if str(config.get("source_collection")) != SOURCE_COLLECTION:
        raise WithoutIftDefinitionError("source_collection must be CKFTrackCollectionWithoutIFT")
    if int(config.get("ift_station_id", -1)) != IFT_STATION_ID:
        raise WithoutIftDefinitionError("ift_station_id must be 0")
    if tuple(config.get("masked_layers") or ()) != MASKED_LAYERS:
        raise WithoutIftDefinitionError("masked_layers must be [0, 1, 2, 3, 4, 5]")
    for key in (
        "geometry_write_allowed",
        "held_out_accessed",
        "real_data_alignment_authorized",
        "measurement_model_validated",
        "three_st_to_ift_chain_authorized",
    ):
        if bool(config.get(key, True)):
            raise WithoutIftDefinitionError(f"{key} must be false")
    for key in (
        "do_not_rescale_covariance",
        "do_not_use_truth_q_over_p_as_real_data_solution",
        "do_not_use_dummy_segmentfit_covariance",
        "do_not_treat_lto_as_without_ift",
        "do_not_treat_four_station_ckf_as_without_ift",
        "do_not_enter_alignment",
        "do_not_enter_measurement_model_v2",
        "do_not_start_new_source_campaign",
        "unconstrained_tracker_only_stopped",
    ):
        if bool(config.get(key, False)) is not True:
            raise WithoutIftDefinitionError(f"{key} must be true")
    return dict(config)


def _expect_sha(path: Path, expected: str, label: str) -> None:
    digest = sha256_file(path)
    if digest != expected:
        raise WithoutIftDefinitionError(f"{label} hash mismatch: {digest}")


def _frozen_root(config: Mapping[str, Any]) -> Path:
    root = Path(str(config.get("frozen_artifact_root") or project_root()))
    if not root.is_dir():
        raise WithoutIftDefinitionError(f"frozen_artifact_root is missing: {root}")
    return root


def inherit_frozen_stage(config: Mapping[str, Any]) -> dict[str, Any]:
    inherited: dict[str, Any] = {}
    frozen_root = _frozen_root(config)
    for workbook, extra in (
        ("workbook_87", {"must_not_reopen_transport": True}),
        ("workbook_95", {"dummy_still_rejected": True}),
        ("workbook_96", {"four_station_5x5_is_not_without_ift": True}),
        ("workbook_109", {"lto_is_not_without_ift": True}),
    ):
        spec = config["inheritance"][workbook]
        decision = json.loads(
            resolve_under_root(frozen_root, spec["decision_path"]).read_text(
                encoding="utf-8"
            )
        )
        _expect_sha(
            resolve_under_root(project_root(), spec["config_path"]),
            spec["config_sha256"],
            f"{workbook} config",
        )
        _expect_sha(
            resolve_under_root(frozen_root, spec["decision_path"]),
            spec["decision_sha256"],
            f"{workbook} decision",
        )
        if decision.get("decision") != spec["frozen_decision"]:
            raise WithoutIftDefinitionError(f"{workbook} decision must stay frozen")
        inherited[workbook] = {
            "decision": decision["decision"],
            "config_sha256": spec["config_sha256"],
            "decision_sha256": spec["decision_sha256"],
            "frozen_artifact_root": str(frozen_root),
            **extra,
        }
    return inherited


def access_split(split: str) -> str:
    if split == "construction":
        return "train"
    if split == "validation":
        return "validation"
    raise WithoutIftDefinitionError(f"unsupported scientific split: {split}")


def verify_pinned_calypso_sources(config: Mapping[str, Any]) -> dict[str, Any]:
    spec = config["software_provenance"]
    root = Path(spec["calypso_root"])
    report: dict[str, Any] = {
        "calypso_root": str(root),
        "expected_git_sha": spec["calypso_git_sha"],
        "files": {},
        "all_match": True,
    }
    for name, item in config["pinned_calypso_sources"].items():
        path = root / str(item["path"])
        if not path.is_file():
            report["files"][name] = {"path": str(path), "present": False}
            report["all_match"] = False
            continue
        digest = sha256_file(path)
        match = digest == str(item["sha256"])
        report["files"][name] = {
            "path": str(path),
            "present": True,
            "sha256": digest,
            "expected": item["sha256"],
            "match": match,
        }
        if not match:
            report["all_match"] = False
    return report


def probe_xaod_keys(path: str | Path, *, split: str, config: Mapping[str, Any]) -> dict[str, Any]:
    import uproot

    authorized = authorize_path(
        path, AccessScope.DEVELOPMENT_VALIDATION, split=access_split(split)
    )
    report: dict[str, Any] = {
        "path": str(authorized),
        "exists": authorized.is_file(),
        "n_events": None,
        "collection_keys": {},
        "has_without_ift": False,
        "has_four_station": False,
        "file_metadata": {},
    }
    if not authorized.is_file():
        return report
    with uproot.open(authorized) as handle:
        if "CollectionTree" not in handle:
            report["missing"] = "CollectionTree"
            return report
        tree = handle["CollectionTree"]
        keys = {str(name).split(";")[0] for name in tree.keys()}
        report["n_events"] = int(tree.num_entries)
        wanted = list(config["root_keys_of_interest"])
        report["collection_keys"] = {key: (key in keys) for key in wanted}
        report["has_without_ift"] = bool(report["collection_keys"].get(SOURCE_COLLECTION))
        report["has_four_station"] = bool(
            report["collection_keys"].get(FOUR_STATION_COLLECTION)
        )
        if "MetaData" in handle:
            meta = handle["MetaData"]
            for key in meta.keys():
                name = str(key).split(";")[0]
                if name.endswith(("conditionsTag", "productionRelease", "dataType")):
                    try:
                        values = meta[key].array(library="np")
                        report["file_metadata"][name] = [
                            str(item)
                            for item in (values.tolist() if hasattr(values, "tolist") else [values])
                        ][:4]
                    except Exception as exc:  # noqa: BLE001 — inventory must continue
                        report["file_metadata"][name] = f"unreadable:{type(exc).__name__}"
    return report


def dump_path_for_source(config: Mapping[str, Any], source_id: str, kind: str) -> Path:
    root = resolve_under_root(project_root(), str(config["dump_root"]))
    filename = (
        str(config["dump_filename"])
        if kind == "tracks"
        else str(config["event_filename"])
    )
    return root / source_id / filename


def load_jsonl(path: Path, *, split: str) -> list[dict[str, Any]]:
    authorized = authorize_path(
        path, AccessScope.DEVELOPMENT_VALIDATION, split=access_split(split)
    )
    if not authorized.is_file():
        return []
    rows = []
    for line in authorized.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _as_matrix(payload: Any) -> np.ndarray | None:
    if payload is None:
        return None
    matrix = np.asarray(payload, dtype=np.float64)
    if matrix.shape != (5, 5):
        return None
    return matrix


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


def evaluate_events(events: list[Mapping[str, Any]]) -> dict[str, Any]:
    n_with_ift = 0
    n_station_decoded = 0
    n_cluster_unavailable = 0
    for row in events:
        if row.get("cluster_stations_unavailable"):
            n_cluster_unavailable += 1
            continue
        counts = row.get("cluster_station_counts") or {}
        if int(counts.get("0", 0) or 0) > 0:
            n_with_ift += 1
        if bool(row.get("station_decoded_by_fasersct_id")):
            n_station_decoded += 1
    return {
        "n_events": len(events),
        "n_events_with_ift_clusters": n_with_ift,
        "n_events_cluster_unavailable": n_cluster_unavailable,
        "n_events_station_decoded": n_station_decoded,
    }


def evaluate_tracks(
    records: list[Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    collection: str,
) -> dict[str, Any]:
    dummy_q = float(config["acceptance"]["dummy_qoverp_per_mev"])
    dummy_v = float(config["acceptance"]["dummy_variance_per_mev2"])
    tol = float(config["acceptance"]["dummy_match_tolerance"])
    selected = [row for row in records if str(row.get("collection")) == collection]
    qoverp = []
    dummy_q_hits = 0
    dummy_v_hits = 0
    truth_hits = 0
    missing_cov = 0
    spd_ok = 0
    cross_nonzero = 0
    ift_mot = 0
    ift_tsos = 0
    decoded = 0
    undecoded = 0
    station_hist = {str(i): 0 for i in range(4)}
    three_st_tracks = 0
    for row in selected:
        if bool(row.get("is_truth", False)):
            truth_hits += 1
            continue
        if bool(row.get("station_decoded_by_fasersct_id")):
            decoded += 1
        else:
            undecoded += 1
        mot = [int(s) for s in (row.get("measurements_on_track_stations") or [])]
        tsos = [int(s) for s in (row.get("tsos_stations") or [])]
        if IFT_STATION_ID in mot:
            ift_mot += 1
        if IFT_STATION_ID in tsos:
            ift_tsos += 1
        for station in set(mot + tsos):
            if str(station) in station_hist:
                station_hist[str(station)] += 1
        if set(mot).issubset(set(THREE_STATION_IDS)) and mot:
            three_st_tracks += 1
        state = np.asarray(row.get("native_state"), dtype=np.float64).reshape(-1)
        if state.size == 5 and np.isfinite(state).all():
            q_value = float(state[QOVERP_INDEX])
            qoverp.append(q_value)
            if abs(q_value - dummy_q) <= tol:
                dummy_q_hits += 1
        cov = _as_matrix(row.get("native_covariance"))
        if cov is None or not bool(row.get("has_covariance", False)):
            missing_cov += 1
            continue
        if abs(float(cov[QOVERP_INDEX, QOVERP_INDEX]) - dummy_v) <= tol:
            dummy_v_hits += 1
        if is_spd(cov):
            spd_ok += 1
        if np.any(np.abs(cov[QOVERP_INDEX, :QOVERP_INDEX]) > 0.0):
            cross_nonzero += 1
    return {
        "collection": collection,
        "n_records": len(selected),
        "n_truth_rejected": truth_hits,
        "n_station_decoded": decoded,
        "n_station_undecoded": undecoded,
        "n_tracks_with_ift_measurements_on_track": ift_mot,
        "n_tracks_with_ift_tsos": ift_tsos,
        "n_tracks_only_three_stations": three_st_tracks,
        "station_track_counts": station_hist,
        "n_missing_covariance": missing_cov,
        "n_spd_ok": spd_ok,
        "n_dummy_qoverp": dummy_q_hits,
        "n_dummy_variance": dummy_v_hits,
        "n_nonzero_qoverp_cross_term": cross_nonzero,
        "q_over_p_per_mev": _finite_stats(np.asarray(qoverp, dtype=np.float64)),
    }


def inventory_split(config: Mapping[str, Any], split: str) -> dict[str, Any]:
    sources = []
    tracks: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    dumps_present = True
    for spec in config["mc_data"][f"{split}_sources"]:
        xaod = authorize_path(
            spec["input_xaod"],
            AccessScope.DEVELOPMENT_VALIDATION,
            split=access_split(split),
        )
        keys = probe_xaod_keys(xaod, split=split, config=config)
        track_path = dump_path_for_source(config, spec["source_id"], "tracks")
        event_path = dump_path_for_source(config, spec["source_id"], "events")
        track_rows = load_jsonl(track_path, split=split)
        event_rows = load_jsonl(event_path, split=split)
        if not track_rows or not event_rows:
            dumps_present = False
        tracks.extend(track_rows)
        events.extend(event_rows)
        sources.append(
            {
                "source_id": spec["source_id"],
                "input_xaod": str(xaod),
                "root_keys": keys,
                "n_track_rows": len(track_rows),
                "n_event_rows": len(event_rows),
            }
        )
    return {
        "split": split,
        "sources": sources,
        "dumps_present": dumps_present and bool(sources),
        "root_has_without_ift": all(
            bool(item["root_keys"].get("has_without_ift")) for item in sources
        )
        if sources
        else False,
        "events": evaluate_events(events),
        "without_ift": evaluate_tracks(tracks, config, collection=SOURCE_COLLECTION),
        "four_station": evaluate_tracks(
            tracks, config, collection=FOUR_STATION_COLLECTION
        ),
    }


def decide(
    construction: Mapping[str, Any],
    validation: Mapping[str, Any],
    pins: Mapping[str, Any],
    *,
    dumps_materialized: bool,
) -> dict[str, Any]:
    min_tracks = 3
    min_ift_events = 1
    if not pins.get("all_match", False):
        mechanism = MECHANISM_SOURCE_PIN_MISMATCH
        verdict = "FAIL"
    elif not dumps_materialized:
        mechanism = MECHANISM_DUMP_MISSING
        verdict = "FAIL"
    elif not (
        construction.get("root_has_without_ift") and validation.get("root_has_without_ift")
    ):
        mechanism = MECHANISM_COLLECTION_ABSENT
        verdict = "FAIL"
    elif int(construction["without_ift"]["n_truth_rejected"]) or int(
        validation["without_ift"]["n_truth_rejected"]
    ):
        mechanism = MECHANISM_TRUTH
        verdict = "FAIL"
    elif int(construction["without_ift"]["n_station_undecoded"]) or int(
        validation["without_ift"]["n_station_undecoded"]
    ):
        mechanism = MECHANISM_STATION_UNDECODED
        verdict = "FAIL"
    elif int(construction["events"]["n_events_with_ift_clusters"]) < min_ift_events or int(
        validation["events"]["n_events_with_ift_clusters"]
    ) < min_ift_events:
        mechanism = MECHANISM_NO_IFT_CLUSTERS
        verdict = "FAIL"
    elif int(construction["without_ift"]["n_records"]) < min_tracks or int(
        validation["without_ift"]["n_records"]
    ) < min_tracks:
        mechanism = MECHANISM_DUMP_MISSING
        verdict = "FAIL"
    elif (
        int(construction["without_ift"]["n_tracks_with_ift_measurements_on_track"])
        or int(construction["without_ift"]["n_tracks_with_ift_tsos"])
        or int(validation["without_ift"]["n_tracks_with_ift_measurements_on_track"])
        or int(validation["without_ift"]["n_tracks_with_ift_tsos"])
    ):
        mechanism = MECHANISM_IFT_LEAK
        verdict = "FAIL"
    elif int(construction["without_ift"]["n_dummy_qoverp"]) == int(
        construction["without_ift"]["n_records"]
    ) and int(validation["without_ift"]["n_dummy_qoverp"]) == int(
        validation["without_ift"]["n_records"]
    ):
        mechanism = MECHANISM_DUMMY
        verdict = "FAIL"
    elif int(construction["without_ift"]["n_missing_covariance"]) or int(
        validation["without_ift"]["n_missing_covariance"]
    ):
        mechanism = MECHANISM_NO_5X5
        verdict = "FAIL"
    else:
        mechanism = None
        verdict = "PASS"
    established = verdict == "PASS"
    return {
        "kind": "ckf_without_ift_definition_contract",
        "task": TASK,
        "workbook": WORKBOOK,
        "verdict": verdict,
        "decision": DECISION_ESTABLISHED if established else DECISION_NOT_ESTABLISHED,
        "mechanism": mechanism,
        "source_collection": SOURCE_COLLECTION,
        "contrast_collection": FOUR_STATION_COLLECTION,
        "calypso_source_contract": calypso_source_contract(),
        "state_definition": state_definition(),
        "lto_is_not_this_collection": True,
        "four_station_ckf_is_not_this_collection": True,
        "three_st_to_ift_chain_authorized": established,
        "measurement_model_v2_entered": False,
        "geometry_write_allowed": False,
        "held_out_accessed": False,
        "real_data_alignment_authorized": False,
        "b14m_reopen_authorized": False,
        "b15_authorized": False,
        "dummy_qoverp_per_mev": SEGMENTFIT_DUMMY_QOVERP_PER_MEV,
    }


def provenance_hashes(config: Mapping[str, Any]) -> dict[str, str]:
    spec = config["software_provenance"]
    tags = config["tags"]
    calypso = spec["calypso_git_sha"]
    return {
        "geometry_hash": hashlib.sha256(
            f"{tags['geometry']}|{calypso}".encode("utf-8")
        ).hexdigest(),
        "field_hash": hashlib.sha256(
            f"{tags['field']}|{calypso}".encode("utf-8")
        ).hexdigest(),
        "conditions_hash": hashlib.sha256(
            f"{tags['conditions']}|{tags['database_instance']}|{calypso}".encode("utf-8")
        ).hexdigest(),
        "calypso_git_sha": calypso,
        "athena_release": spec["athena_release"],
        "acts_version": spec["acts_version"],
    }
