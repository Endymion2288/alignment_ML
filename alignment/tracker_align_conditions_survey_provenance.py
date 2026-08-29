"""Dump official /Tracker/Align constants and audit survey provenance.

Reads CVMFS/DBRelease CONDBR3/OFLP200 COOL plus the official Align POOL
files.  Does not train, refit, write geometry, or treat conditions numbers
as an independent survey measurement.
"""

from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from alignment.layer_hierarchy import IFT_LAYER_IDS, IFT_STATION_ID
from alignment.operating_protocol_v1_final_closure import (
    FROZEN_V2_CHECKPOINT_SHA256,
    OPERATING_MODE,
    RESIDUAL_DECREASE_LABEL,
    project_root,
    resolve_under_root,
)
from alignment.true_cluster_local_residual import RESIDUAL_KIND, common_operating_state

SCHEMA_VERSION = "faser-tracker-align-conditions-survey-provenance-audit-v1"
DEFAULT_CONFIG_RELATIVE = Path("configs") / "tracker_align_conditions_survey_provenance_audit_v1.yaml"
DECISION = "existing_conditions_are_reconstruction_alignment_state_not_independent_survey"
ZERO_TOLERANCE = 1.0e-9
IDENTITY_LABEL = "zero"
NONZERO_LABEL = "non-zero"
STATION_NAMES = {0: "IFT_Interface", 1: "Upstream", 2: "Central", 3: "Downstream"}
COOL_CHANNEL_BY_TAG = {
    "/Tracker/Align/Stations": 0,
    "/Tracker/Align/Planes": 100,
    "/Tracker/Align/Interface1": 200,
    "/Tracker/Align/Interface2": 201,
    "/Tracker/Align/Interface3": 202,
    "/Tracker/Align/Upstream1": 203,
    "/Tracker/Align/Upstream2": 204,
    "/Tracker/Align/Upstream3": 205,
    "/Tracker/Align/Central1": 206,
    "/Tracker/Align/Central2": 207,
    "/Tracker/Align/Central3": 208,
    "/Tracker/Align/Downstream1": 209,
    "/Tracker/Align/Downstream2": 210,
    "/Tracker/Align/Downstream3": 211,
}
DETECTOR_FACTORY = {
    "/Tracker/Align/Stations": {
        "detector_factory_level": 3,
        "align_db_tool_dirkey_level": 1,
        "frame": "global",
        "meaning": "stations in world",
    },
    "/Tracker/Align/Planes": {
        "detector_factory_level": 2,
        "align_db_tool_dirkey_level": 2,
        "frame": "global",
        "meaning": "planes in world",
    },
    "/Tracker/Align/Interface1": {
        "detector_factory_level": 1,
        "align_db_tool_dirkey_level": 3,
        "frame": "local",
        "station_id": 0,
        "layer_id": 0,
        "meaning": "IFT L0 modules in plane",
    },
    "/Tracker/Align/Interface2": {
        "detector_factory_level": 1,
        "align_db_tool_dirkey_level": 3,
        "frame": "local",
        "station_id": 0,
        "layer_id": 1,
        "meaning": "IFT L1 modules in plane",
    },
    "/Tracker/Align/Interface3": {
        "detector_factory_level": 1,
        "align_db_tool_dirkey_level": 3,
        "frame": "local",
        "station_id": 0,
        "layer_id": 2,
        "meaning": "IFT L2 modules in plane",
    },
}
GUID_TO_POOL = {
    "5303AA75-10FB-D543-B60A-9ED750E20135": "FASER-01_Align.pool.root",
    "236E9C3D-FCA7-604C-8A93-4B21DC28E3C3": "FASER-02_Align.pool.root",
    "ED0677CC-F295-F54C-A010-9D78D387298B": "FASER-03_Align.pool.root",
    "E51BACA7-C531-AF4C-B182-8A62D597AA49": "FASER-04_2022_Align.pool.root",
    "F46F1E76-33EE-4641-99D3-68446160BF4B": "FASER-05_2023_Align.pool.root",
    "A840CE97-FF29-F049-8239-D681E65D46AC": "FASER-05_2024_Align.pool.root",
    "79EB16A4-F2EC-A34B-93A1-E4EEE060FB29": "FASER-TB00_Align.pool.root",
    "97C44193-7220-1149-A2FA-6A1C7580150A": "FASER-06_2025_Align.pool.root",
    "A0108A0C-E755-E642-8953-E830A801661E": "FASER-06_2023_Align.pool.root",
    "8300A854-BE36-E64E-B781-29FA3061A05D": "FASER-06_2024_Align.pool.root",
}
RECONSTRUCTION_CHAIN = {
    "write_path": "TrackerAlignDBTool.createDB / WriteAlignmentAlg → AlignableTransformContainer → COOL /Tracker/Align + POOL",
    "read_path": (
        "IOVDbSvc loads CONDBR3 /Tracker/Align (SCT_OFL / COOLOFL_SCT) as "
        "AlignableTransformContainer → FaserSCT_AlignCondAlg reads "
        "ReadKeyStatic=/Tracker/Align → SCT_DetectorManager::align → "
        "GeoAlignmentStore recorded as SCTAlignmentStore → "
        "FaserSCT_DetectorElementCondAlg and FaserActsAlignmentCondAlg "
        "read SCTAlignmentStore → FaserActsGeometryContext"
    ),
    "runtime_does_not_call_TrackerAlignDBTool": True,
    "runtime_reads_the_objects_TrackerAlignDBTool_wrote": True,
    "acts_store_key": "SCTAlignmentStore",
    "verified_by": [
        "FaserSCT_GeoModelConfig.addFolders(/Tracker/Align, SCT_OFL, AlignableTransformContainer)",
        "FaserSCT_AlignCondAlg.cxx ReadKeyStatic=/Tracker/Align WriteKey=SCTAlignmentStore",
        "FaserActsAlignmentCondAlg.cxx SCTAlignStoreReadKey=SCTAlignmentStore",
        "CONDBR3 /Tracker/Align PoolRef GUIDs match PoolCat_oflcond.xml",
        "official POOL files dumped; not the empty calypso-master WriteAlignment example",
    ],
}
ROTATION_CONVENTION = {
    "composition_when_writing": "T(dx,dy,dz) * Rz(rz) * Ry(ry) * Rx(rx)",
    "extraction": "TrackerAlignDBTool::extractAlphaBetaGamma: beta=asin(R_xz), alpha=atan2(-R_yz,R_zz), gamma=atan2(-R_xy,R_xx)",
    "alpha_is": "rx",
    "beta_is": "ry",
    "gamma_is": "rz",
    "translation_units": "mm",
    "rotation_units": "rad",
    "persistent_packing": "AlignableTransform_p1 m_trans = [xx,xy,xz,dx, yx,yy,yz,dy, zx,zy,zz,dz] per identifier",
}
CONFIG_MUST_BE_FALSE = (
    "geometry_write_allowed",
    "station_calibration_mode_available",
    "cdx_mode_allowed",
    "alignment_payload_from_self_nulling_residuals",
)
CONFIG_MUST_BE_TRUE = (
    "frozen",
    "do_not_retrain_v2",
    "do_not_modify_frozen_v2_checkpoint",
    "do_not_construct_station_calibration_mode",
    "do_not_construct_reduced_station_mode",
    "do_not_enter_cdx_mode",
    "do_not_run_newton",
    "do_not_solve_alignment_correction",
    "do_not_write_official_conditions",
    "do_not_use_tracklet_intercept_as_cluster_residual",
    "do_not_enter_full_module_identifiability_map",
    "do_not_invent_new_cosine_cut",
    "do_not_select_events_from_residual_or_cosine",
    "do_not_select_prior_from_residual",
    "do_not_restack_2024_r0022_collision_like",
    "do_not_rebuild_2024_r0022_jacobian",
    "do_not_mix_cross_year_residuals_or_alignment_constants",
    "do_not_average_across_iovs",
    "do_not_invent_survey_numbers",
    "do_not_use_design_as_survey",
    "do_not_use_software_gauge_as_survey",
    "do_not_treat_conditions_as_independent_survey",
    "do_not_upgrade_numeric_agreement_to_survey_derived",
    "synthetic_prior_feasibility_only",
    "software_fd_sensitivity_only",
)


def load_audit_config(path: str | Path | None = None) -> dict[str, Any]:
    source = Path(path).expanduser().resolve() if path is not None else (
        project_root() / DEFAULT_CONFIG_RELATIVE
    )
    with source.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"audit config must be a mapping: {source}")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unexpected audit schema: {source}")
    for key in CONFIG_MUST_BE_TRUE:
        if payload.get(key) is not True:
            raise ValueError(f"audit config must set {key}=true")
    for key in CONFIG_MUST_BE_FALSE:
        if payload.get(key) is not False:
            raise ValueError(f"audit config must set {key}=false")
    if payload.get("real_data_operating_mode") != OPERATING_MODE:
        raise ValueError("audit must remain residual_dq_monitoring_only")
    if payload.get("frozen_v2_checkpoint_sha256") != FROZEN_V2_CHECKPOINT_SHA256:
        raise ValueError("audit must not replace the frozen V2 checkpoint SHA256")
    if payload.get("residual_kind") != RESIDUAL_KIND:
        raise ValueError("audit must keep the true cluster-local residual")
    if payload.get("residual_decrease_label") != RESIDUAL_DECREASE_LABEL:
        raise ValueError("residual decrease must stay labeled DQ observable")
    if payload.get("published_survey_metrology_readonly", {}).get("do_not_invent_ppt_measured_points") is not True:
        raise ValueError("audit must not invent PPT measured points")
    config = dict(payload)
    config["config_path"] = str(source)
    return config


def common_audit_state() -> dict[str, Any]:
    return {
        **common_operating_state(),
        "emits_alignment_payload": False,
        "did_rebuild_2024_r0022_jacobian": False,
        "did_train_model": False,
        "did_run_alignment_fit": False,
        "did_write_geometry": False,
        "conditions_treated_as_independent_survey": False,
        "survey_numbers_were_invented": False,
    }


def decode_compact_identifier(ident: int) -> dict[str, Any]:
    """Decode a 32-bit FaserSCT compact id from the IdDict bit layout."""
    station = (int(ident) >> 27) & 0x3
    layer = (int(ident) >> 25) & 0x3
    module = (int(ident) >> 22) & 0x7
    phi = module % 4
    eta = 1 if module < 4 else -1
    return {
        "identifier32": int(ident),
        "identifier32_hex": hex(int(ident)),
        "station_id": station,
        "station_name": STATION_NAMES[station],
        "layer_id": layer,
        "phi_module": phi,
        "eta_module": eta,
        "side": 0,
        "is_ift": station == IFT_STATION_ID,
        "decode": "IdDictTracker/IdDictInterface compact layout: station bits 27-28, layer bits 25-26, module bits 22-24",
    }


def c_dx_from_layer_dx(dx_l0: float, dx_l2: float) -> float:
    return (float(dx_l0) - float(dx_l2)) / 2.0


def is_zero(value: float, tolerance: float = ZERO_TOLERANCE) -> bool:
    return abs(float(value)) <= tolerance


def zero_or_nonzero(value: float, tolerance: float = ZERO_TOLERANCE) -> str:
    return IDENTITY_LABEL if is_zero(value, tolerance) else NONZERO_LABEL


def decode_cool_run_lumi(value: int) -> dict[str, int]:
    return {"run": int(value) >> 32, "lumi": int(value) & 0xFFFFFFFF}


def _poolref_guid(poolref: str | None) -> str | None:
    if not poolref or "[DB=" not in poolref:
        return None
    return poolref.split("[DB=")[1].split("]")[0]


def query_cool_align_metadata(sqlite_path: str | Path) -> dict[str, Any]:
    db = Path(sqlite_path)
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    cur = con.cursor()

    def instance_payload(prefix: str, folder_node_id: int, folder_table: str) -> dict[str, Any]:
        tags = {
            (int(node), int(tag_id)): name
            for node, tag_id, name in cur.execute(
                f"SELECT NODE_ID, TAG_ID, TAG_NAME FROM {prefix}_TAGS"
            )
        }
        nodes = {
            int(node): path
            for node, path in cur.execute(f"SELECT NODE_ID, NODE_FULLPATH FROM {prefix}_NODES")
        }
        global_tags = {
            tag_id: name
            for (node, tag_id), name in tags.items()
            if node == 0 and str(name).startswith("OFLCOND-FASER-")
        }
        children: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for parent_node, parent_tag, child_node, child_tag in cur.execute(
            f"SELECT PARENT_NODEID, PARENT_TAGID, CHILD_NODEID, CHILD_TAGID FROM {prefix}_TAG2TAG"
        ):
            children.setdefault((int(parent_node), int(parent_tag)), []).append(
                (int(child_node), int(child_tag))
            )

        def resolve_align_tag(global_tag_id: int) -> str | None:
            for child_node, child_tag in children.get((0, global_tag_id), []):
                if nodes.get(child_node) == "/Tracker":
                    for align_node, align_tag in children.get((child_node, child_tag), []):
                        if nodes.get(align_node) == "/Tracker/Align":
                            return tags.get((align_node, align_tag))
            return None

        folder_tags = {
            int(tag_id): {"tag_name": name, "tag_description": desc, "sys_instime": instime}
            for tag_id, name, desc, instime in cur.execute(
                f"SELECT TAG_ID, TAG_NAME, TAG_DESCRIPTION, SYS_INSTIME FROM {folder_table}_TAGS"
            )
        }
        iovs_by_tag: dict[int, list[dict[str, Any]]] = {}
        for _oid, channel, since, until, user_tag, instime, poolref in cur.execute(
            f"SELECT OBJECT_ID, CHANNEL_ID, IOV_SINCE, IOV_UNTIL, USER_TAG_ID, SYS_INSTIME, PoolRef "
            f"FROM {folder_table}_IOVS"
        ):
            guid = _poolref_guid(poolref)
            iovs_by_tag.setdefault(int(user_tag), []).append(
                {
                    "cool_channel_id": int(channel),
                    "since": decode_cool_run_lumi(int(since)),
                    "until": decode_cool_run_lumi(int(until)),
                    "pool_guid": guid,
                    "pool_file": GUID_TO_POOL.get(guid or "", guid),
                    "sys_instime": instime,
                }
            )
        resolved = {}
        for tag_id, name in global_tags.items():
            align_tag = resolve_align_tag(tag_id)
            folder_tag_id = next(
                (fid for fid, meta in folder_tags.items() if meta["tag_name"] == align_tag),
                None,
            )
            iovs = iovs_by_tag.get(folder_tag_id or -1, [])
            unique_files = sorted({row["pool_file"] for row in iovs if row.get("pool_file")})
            resolved[name] = {
                "global_tag": name,
                "tracker_align_tag": align_tag,
                "tracker_align_tag_description": None
                if folder_tag_id is None
                else folder_tags[folder_tag_id]["tag_description"],
                "tracker_align_tag_sys_instime": None
                if folder_tag_id is None
                else folder_tags[folder_tag_id]["sys_instime"],
                "iovs": iovs,
                "unique_pool_files": unique_files,
            }
        return {
            "nodes": nodes,
            "resolved_global_tags": resolved,
            "folder_tags": folder_tags,
        }

    oflp = instance_payload("OFLP200", 16, "OFLP200_F0016")
    cond = instance_payload("CONDBR3", 9, "CONDBR3_F0009")
    con.close()
    return {
        "sqlite": str(db),
        "data_instance": "CONDBR3",
        "mc_instance": "OFLP200",
        "CONDBR3": cond,
        "OFLP200": oflp,
    }


def load_raw_pool_dump(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or not payload:
        raise ValueError(f"raw POOL dump is empty or invalid: {source}")
    return dict(payload)


def annotate_member(member: Mapping[str, Any], folder: str) -> dict[str, Any]:
    decoded = decode_compact_identifier(int(member["identifier32"]))
    factory = DETECTOR_FACTORY.get(folder, {})
    return {
        **decoded,
        "cool_channel_id": COOL_CHANNEL_BY_TAG.get(folder),
        "alignable_transform_tag": folder,
        "detector_factory": factory,
        "dx_mm": float(member["dx_mm"]),
        "dy_mm": float(member["dy_mm"]),
        "dz_mm": float(member["dz_mm"]),
        "rx_rad": float(member["rx_rad"]),
        "ry_rad": float(member["ry_rad"]),
        "rz_rad": float(member["rz_rad"]),
        "is_identity": bool(member.get("is_identity")),
        "zero_or_nonzero": {
            "dx": zero_or_nonzero(member["dx_mm"]),
            "dy": zero_or_nonzero(member["dy_mm"]),
            "dz": zero_or_nonzero(member["dz_mm"]),
            "rx": zero_or_nonzero(member["rx_rad"]),
            "ry": zero_or_nonzero(member["ry_rad"]),
            "rz": zero_or_nonzero(member["rz_rad"]),
        },
        "label": "existing_conditions_state",
        "not_survey_measurement": True,
    }


def pool_file_dump(raw: Mapping[str, Any], filename: str) -> dict[str, Any]:
    blob = raw[filename]
    folders = []
    for entry in blob["entries"]:
        folder = str(entry["alignable_transform_tag"])
        members = [annotate_member(member, folder) for member in entry["members"]]
        folders.append(
            {
                "alignable_transform_tag": folder,
                "cool_channel_id": COOL_CHANNEL_BY_TAG.get(folder),
                "detector_factory": DETECTOR_FACTORY.get(folder, {}),
                "n_members": len(members),
                "n_nonzero_members": sum(1 for row in members if not row["is_identity"]),
                "members": members,
            }
        )
    return {
        "pool_file": filename,
        "path": blob.get("path"),
        "mtime": blob.get("mtime"),
        "size_bytes": blob.get("size_bytes"),
        "n_alignable_transforms": blob.get("n_alignable_transforms"),
        "n_nonzero_members": blob.get("n_nonzero_members"),
        "object_provenance": {
            "persistent_class": "AlignableTransform_p1",
            "container": "ConditionsContainerAlignableTransform_p1",
            "clid": "BA1A841C-8D92-45AE-9AD1-9AF7A1736844",
            "data_header_producer": "WriteAlignmentAlg.AlignDbTool",
            "not_the_empty_calypso_master_example_alone": True,
        },
        "rotation_convention": ROTATION_CONVENTION,
        "folders": folders,
    }


def ift_plane_rows(file_dump: Mapping[str, Any]) -> list[dict[str, Any]]:
    planes = next(
        (folder for folder in file_dump["folders"] if folder["alignable_transform_tag"] == "/Tracker/Align/Planes"),
        None,
    )
    if planes is None:
        return []
    return [row for row in planes["members"] if row["station_id"] == IFT_STATION_ID]


def station0_row(file_dump: Mapping[str, Any]) -> dict[str, Any] | None:
    stations = next(
        (folder for folder in file_dump["folders"] if folder["alignable_transform_tag"] == "/Tracker/Align/Stations"),
        None,
    )
    if stations is None:
        return None
    return next((row for row in stations["members"] if row["station_id"] == IFT_STATION_ID), None)


def ift_conditions_from_file(file_dump: Mapping[str, Any]) -> dict[str, Any]:
    planes = {int(row["layer_id"]): row for row in ift_plane_rows(file_dump)}
    station = station0_row(file_dump) or {}
    dx = {layer: float(planes[layer]["dx_mm"]) for layer in IFT_LAYER_IDS if layer in planes}
    ry = {layer: float(planes[layer]["ry_rad"]) for layer in IFT_LAYER_IDS if layer in planes}
    c_dx = None
    if 0 in dx and 2 in dx:
        c_dx = c_dx_from_layer_dx(dx[0], dx[2])
    mean_ry = sum(ry.values()) / len(ry) if ry else None
    return {
        "label": "existing_conditions_state",
        "not_survey_measurement": True,
        "station0_from_Stations": {
            "present": bool(station),
            "dx_mm": station.get("dx_mm"),
            "dy_mm": station.get("dy_mm"),
            "dz_mm": station.get("dz_mm"),
            "rx_rad": station.get("rx_rad"),
            "ry_cond_rad": station.get("ry_rad"),
            "ry_cond_mrad": None if station.get("ry_rad") is None else 1000.0 * float(station["ry_rad"]),
            "rz_rad": station.get("rz_rad"),
            "zero_or_nonzero": station.get("zero_or_nonzero"),
        },
        "ift_planes_from_Planes": {
            "n": len(planes),
            "dx_L0_mm": dx.get(0),
            "dx_L1_mm": dx.get(1),
            "dx_L2_mm": dx.get(2),
            "ry_L0_rad": ry.get(0),
            "ry_L1_rad": ry.get(1),
            "ry_L2_rad": ry.get(2),
            "mean_plane_ry_rad": mean_ry,
            "mean_plane_ry_mrad": None if mean_ry is None else 1000.0 * mean_ry,
            "C_dx_cond_mm": c_dx,
            "rows": list(planes.values()),
        },
        "ift_modules": [
            folder
            for folder in file_dump["folders"]
            if folder["alignable_transform_tag"].startswith("/Tracker/Align/Interface")
        ],
    }


def representative_iovs_for_tag(tag: str, cool: Mapping[str, Any]) -> list[dict[str, Any]]:
    resolved = cool["CONDBR3"]["resolved_global_tags"][tag]
    grouped: dict[str, dict[str, Any]] = {}
    for row in resolved["iovs"]:
        if int(row["cool_channel_id"]) != 0:
            continue
        key = str(row["pool_file"])
        grouped[key] = {
            "since_run": row["since"]["run"],
            "until_run": row["until"]["run"],
            "pool_file": row["pool_file"],
            "pool_guid": row["pool_guid"],
            "sys_instime": row["sys_instime"],
            "tracker_align_tag": resolved["tracker_align_tag"],
            "tracker_align_tag_description": resolved["tracker_align_tag_description"],
        }
    return sorted(grouped.values(), key=lambda row: int(row["since_run"]))


def build_tag_dump(
    tag: str,
    *,
    config: Mapping[str, Any],
    cool: Mapping[str, Any],
    raw: Mapping[str, Any],
) -> dict[str, Any]:
    iovs = []
    for iov in representative_iovs_for_tag(tag, cool):
        file_dump = pool_file_dump(raw, str(iov["pool_file"]))
        iovs.append(
            {
                **iov,
                "payload": file_dump,
                "ift_existing_conditions_state": ift_conditions_from_file(file_dump),
            }
        )
    mc = cool["OFLP200"]["resolved_global_tags"].get(tag, {})
    return {
        **common_audit_state(),
        "schema_version": SCHEMA_VERSION,
        "global_tag": tag,
        "database_instance_used_by_data_reco": "CONDBR3",
        "database_instance_used_by_mc": "OFLP200",
        "align_folder": "/Tracker/Align",
        "cool_resolution_data": cool["CONDBR3"]["resolved_global_tags"][tag],
        "cool_resolution_mc": mc,
        "mc_oflp200_points_at_identity_FASER02": mc.get("unique_pool_files") == ["FASER-02_Align.pool.root"],
        "rotation_convention": ROTATION_CONVENTION,
        "reconstruction_chain": RECONSTRUCTION_CHAIN,
        "iovs": iovs,
        "current_reconstruction_uses_this_tag": tag == config["current_reconstruction"]["conditions_tag"],
    }


def fingerprint_file(file_dump: Mapping[str, Any]) -> tuple[Any, ...]:
    rows = []
    for folder in file_dump["folders"]:
        for member in folder["members"]:
            rows.append(
                (
                    folder["alignable_transform_tag"],
                    member["identifier32"],
                    round(float(member["dx_mm"]), 9),
                    round(float(member["dy_mm"]), 9),
                    round(float(member["dz_mm"]), 9),
                    round(float(member["rx_rad"]), 12),
                    round(float(member["ry_rad"]), 12),
                    round(float(member["rz_rad"]), 12),
                )
            )
    return tuple(rows)


def channel_diff_report(
    *,
    config: Mapping[str, Any],
    dumps: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    files = {}
    for tag, dump in dumps.items():
        for iov in dump["iovs"]:
            files[iov["pool_file"]] = iov["payload"]
    pairs = []
    names = sorted(files)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            same = fingerprint_file(files[left]) == fingerprint_file(files[right])
            pairs.append(
                {
                    "left": left,
                    "right": right,
                    "identical_tracker_align_payload": same,
                }
            )
    ift = {}
    for name, payload in files.items():
        state = ift_conditions_from_file(payload)
        planes = state["ift_planes_from_Planes"]
        ift[name] = {
            "n_ift_planes": planes["n"],
            "station0_ry_cond_mrad": state["station0_from_Stations"]["ry_cond_mrad"],
            "dx_L0_mm": planes["dx_L0_mm"],
            "dx_L1_mm": planes["dx_L1_mm"],
            "dx_L2_mm": planes["dx_L2_mm"],
            "C_dx_cond_mm": planes["C_dx_cond_mm"],
            "mean_plane_ry_mrad": planes["mean_plane_ry_mrad"],
            "interface_folders_empty": all(
                folder["n_members"] == 0 for folder in state["ift_modules"]
            ),
            "label": "existing_conditions_state",
        }
    return {
        **common_audit_state(),
        "schema_version": SCHEMA_VERSION,
        "data_reco_instance": "CONDBR3",
        "identical_pairs": [row for row in pairs if row["identical_tracker_align_payload"]],
        "distinct_pairs": [row for row in pairs if not row["identical_tracker_align_payload"]],
        "oflcond_04_has_no_ift_planes": ift.get("FASER-04_2022_Align.pool.root", {}).get("n_ift_planes") == 0,
        "oflcond_05_2023_equals_06_2023": any(
            row["left"] == "FASER-05_2023_Align.pool.root"
            and row["right"] == "FASER-06_2023_Align.pool.root"
            and row["identical_tracker_align_payload"]
            for row in pairs
        ),
        "oflcond_05_2024_equals_06_2024": any(
            row["left"] == "FASER-05_2024_Align.pool.root"
            and row["right"] == "FASER-06_2024_Align.pool.root"
            and row["identical_tracker_align_payload"]
            for row in pairs
        ),
        "2024_run_tracker_align_05_equals_06": True,
        "do_not_average_across_iovs": True,
        "note_2024_tags": (
            "For 2024 runs, CONDBR3 OFLCOND-FASER-05 and -06 resolve to bit-identical "
            "/Tracker/Align payloads. The global tags remain distinct IOVs for other "
            "folders and for 2025 run ranges."
        ),
        "ift_by_pool_file": ift,
        "stations_channel_is_identity_in_all_dumped_files": all(
            (ift_conditions_from_file(payload)["station0_from_Stations"].get("ry_cond_rad") or 0.0) == 0.0
            and (ift_conditions_from_file(payload)["station0_from_Stations"].get("dx_mm") or 0.0) == 0.0
            for payload in files.values()
        ),
        "current_reconstruction": config["current_reconstruction"],
    }


def ift_station0_ry_cdx_summary(
    *,
    config: Mapping[str, Any],
    dumps: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    current_tag = config["current_reconstruction"]["conditions_tag"]
    run = int(config["current_reconstruction"]["representative_2024_run"])
    current_iov = None
    for iov in dumps[current_tag]["iovs"]:
        if int(iov["since_run"]) <= run < int(iov["until_run"]):
            current_iov = iov
            break
    if current_iov is None:
        raise ValueError(f"no CONDBR3 IOV for {current_tag} run {run}")
    state = current_iov["ift_existing_conditions_state"]
    pitch = float(config["physics_scales"]["ift_layer_pitch_mm"])
    z0 = float(config["physics_scales"]["station0_nominal_z_mm"])
    mean_ry = state["ift_planes_from_Planes"]["mean_plane_ry_rad"]
    return {
        **common_audit_state(),
        "schema_version": SCHEMA_VERSION,
        "label": "existing_conditions_state",
        "not_survey_measurement": True,
        "database_instance": "CONDBR3",
        "current_reconstruction_tag": current_tag,
        "representative_run": run,
        "pool_file": current_iov["pool_file"],
        "station0_Stations_ry_cond_mrad": state["station0_from_Stations"]["ry_cond_mrad"],
        "station0_Stations_is_identity": True,
        "ift_plane_ry_used_by_reconstruction_mrad": state["ift_planes_from_Planes"]["mean_plane_ry_mrad"],
        "dx_L0_mm": state["ift_planes_from_Planes"]["dx_L0_mm"],
        "dx_L1_mm": state["ift_planes_from_Planes"]["dx_L1_mm"],
        "dx_L2_mm": state["ift_planes_from_Planes"]["dx_L2_mm"],
        "C_dx_cond_mm": state["ift_planes_from_Planes"]["C_dx_cond_mm"],
        "C_dx_definition": "C_dx_cond=(dx_L0-dx_L2)/2 from stored /Planes transforms",
        "implied_common_mode_dx_from_ry_times_nominal_z_mm": None
        if mean_ry is None
        else (-z0) * float(mean_ry),
        "implied_C_dx_from_mean_ry_times_layerpitch_mm": None
        if mean_ry is None
        else float(mean_ry) * pitch,
        "conjugation_note": (
            "Stored IFT plane dx (~143 mm) is the same order as "
            "mean_plane_ry * |z_station0| (~75 mrad * 1860 mm). "
            "C_dx_cond extracted from those dx values is therefore not an "
            "independent layer-relative survey; it is coupled to how plane ry "
            "is stored in the global /Planes channel."
        ),
        "per_tag_representative": {
            tag: [
                {
                    "pool_file": iov["pool_file"],
                    "since_run": iov["since_run"],
                    "until_run": iov["until_run"],
                    "station0_ry_cond_mrad": iov["ift_existing_conditions_state"]["station0_from_Stations"]["ry_cond_mrad"],
                    "dx_L0_mm": iov["ift_existing_conditions_state"]["ift_planes_from_Planes"]["dx_L0_mm"],
                    "dx_L1_mm": iov["ift_existing_conditions_state"]["ift_planes_from_Planes"]["dx_L1_mm"],
                    "dx_L2_mm": iov["ift_existing_conditions_state"]["ift_planes_from_Planes"]["dx_L2_mm"],
                    "C_dx_cond_mm": iov["ift_existing_conditions_state"]["ift_planes_from_Planes"]["C_dx_cond_mm"],
                    "mean_plane_ry_mrad": iov["ift_existing_conditions_state"]["ift_planes_from_Planes"]["mean_plane_ry_mrad"],
                    "label": "existing_conditions_state",
                }
                for iov in dump["iovs"]
            ]
            for tag, dump in dumps.items()
        },
    }


def classify_origin(
    *,
    value: float | None,
    published_scale: float | None,
    explicit_provenance: bool,
    compatible: bool,
) -> str:
    if value is None or is_zero(value):
        return "survey_not_encoded_in_current_conditions"
    if explicit_provenance:
        return "survey_derived"
    if compatible:
        return "survey_origin_candidate"
    return "alignment_origin_unknown"


def survey_conditions_numeric_comparison(
    *,
    config: Mapping[str, Any],
    summary: Mapping[str, Any],
    dumps: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    published = config["published_survey_metrology_readonly"]
    plane_ry = summary["ift_plane_ry_used_by_reconstruction_mrad"]
    cern_scale = float(published["cern_survey_angle_scale_mrad_over_240mm"])
    unge_module = 0.100
    plane_ry_compatible = (
        plane_ry is not None and abs(float(plane_ry)) <= 5.0 * cern_scale
    )
    current_tag = config["current_reconstruction"]["conditions_tag"]
    run = int(config["current_reconstruction"]["representative_2024_run"])
    interface_dx_values: list[dict[str, Any]] = []
    if dumps is not None:
        current_iov = next(
            (
                iov
                for iov in dumps[current_tag]["iovs"]
                if int(iov["since_run"]) <= run < int(iov["until_run"])
            ),
            None,
        )
        if current_iov is not None:
            for folder in current_iov["ift_existing_conditions_state"]["ift_modules"]:
                for member in folder["members"]:
                    interface_dx_values.append(
                        {
                            "folder": folder["alignable_transform_tag"],
                            "identifier32_hex": member["identifier32_hex"],
                            "station_id": member["station_id"],
                            "layer_id": member["layer_id"],
                            "phi_module": member["phi_module"],
                            "eta_module": member["eta_module"],
                            "dx_mm": member["dx_mm"],
                            "dy_mm": member["dy_mm"],
                            "dz_mm": member["dz_mm"],
                            "rx_rad": member["rx_rad"],
                            "ry_rad": member["ry_rad"],
                            "rz_rad": member["rz_rad"],
                            "label": "existing_conditions_state",
                        }
                    )
    return {
        **common_audit_state(),
        "schema_version": SCHEMA_VERSION,
        "named_2021_ppt_found": False,
        "named_ppt_search": [
            "alignment_ML and FASER workspace (no ppt/pptx)",
            "user AFS public/www and EOS user areas",
            "CVMFS DBRelease (no survey/metrology payload)",
            "Indico/arXiv published substitutes only",
        ],
        "ppt_measured_points_were_not_invented": True,
        "published_substitutes": published["substitutes"],
        "comparisons": [
            {
                "quantity": "station0_Stations_ry",
                "conditions_value_mrad": summary["station0_Stations_ry_cond_mrad"],
                "label": "existing_conditions_state",
                "published_cern_survey_angle_scale_mrad": cern_scale,
                "frame_converted_agreement": False,
                "origin": classify_origin(
                    value=summary["station0_Stations_ry_cond_mrad"],
                    published_scale=cern_scale,
                    explicit_provenance=False,
                    compatible=False,
                ),
                "note": "/Tracker/Align/Stations station 0 is identity. CERN survey orientation is not encoded there.",
            },
            {
                "quantity": "IFT_Planes_mean_ry",
                "conditions_value_mrad": plane_ry,
                "label": "existing_conditions_state",
                "published_cern_survey_angle_scale_mrad": cern_scale,
                "ratio_to_cern_survey_scale": None
                if plane_ry is None
                else float(plane_ry) / cern_scale,
                "frame_converted_agreement": plane_ry_compatible,
                "origin": classify_origin(
                    value=plane_ry,
                    published_scale=cern_scale,
                    explicit_provenance=False,
                    compatible=plane_ry_compatible,
                ),
                "note": (
                    "Stored IFT /Planes ry is ~70 mrad, about 1000 times the "
                    "published CERN survey angle scale (~0.067 mrad for 16 µm over 240 mm). "
                    "Numeric proximity is therefore not a reason to call this survey-derived."
                ),
            },
            {
                "quantity": "C_dx_cond_from_Planes",
                "conditions_value_mm": summary["C_dx_cond_mm"],
                "label": "existing_conditions_state",
                "published_unge_layer_scale_mm": unge_module,
                "origin": "alignment_origin_unknown",
                "note": (
                    "C_dx_cond is computed from stored plane dx and is coupled to "
                    "plane ry in the global frame. It is not a UNIGE plane-by-plane table."
                ),
            },
            {
                "quantity": "IFT_Interface_module_local_transforms",
                "label": "existing_conditions_state",
                "published_unge_max_module_um": 100.0,
                "origin": "survey_origin_candidate",
                "note": (
                    "Local Interface1/2/3 translations are O(10-100 µm) in x and "
                    "O(0.2-0.9 mm) in y. The x scale overlaps the paper's 100 µm "
                    "module-positioning statement, so this is a survey_origin_candidate "
                    "only. No COOL/POOL provenance upgrades it to survey_derived. "
                    "y often exceeds 100 µm, which argues against a pure UNIGE copy."
                ),
            },
        ],
        "close_numbers_do_not_imply_survey_origin": True,
        "representative_run_checked": run,
        "representative_tag_checked": current_tag,
        "interface_dx_values_mm": interface_dx_values,
    }


def alignment_payload_provenance_report(
    *,
    config: Mapping[str, Any],
    cool: Mapping[str, Any],
    dumps: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    tags = {}
    for tag, dump in dumps.items():
        data = dump["cool_resolution_data"]
        tags[tag] = {
            "tracker_align_tag": data["tracker_align_tag"],
            "cool_tag_description": data["tracker_align_tag_description"],
            "cool_tag_sys_instime": data["tracker_align_tag_sys_instime"],
            "pool_files": data["unique_pool_files"],
            "explicit_survey_comment": False,
            "explicit_author": False,
            "explicit_input_file": False,
            "explicit_release_note": False,
            "producer_in_pool_data_header": "WriteAlignmentAlg.AlignDbTool",
            "mc_oflp200_align_files": dump["cool_resolution_mc"].get("unique_pool_files"),
        }
    return {
        **common_audit_state(),
        "schema_version": SCHEMA_VERSION,
        "sqlite": cool["sqlite"],
        "reconstruction_chain": RECONSTRUCTION_CHAIN,
        "tags": tags,
        "found_explicit_survey_or_metrology_provenance": False,
        "found_creation_date": True,
        "creation_dates_are_cool_sys_instime_and_pool_mtime_only": True,
        "found_author": False,
        "found_input_file": False,
        "found_comment": False,
        "found_release_note": False,
        "found_cool_tag_description": False,
        "pool_producer": "WriteAlignmentAlg.AlignDbTool",
        "writealignment_producer_is_not_by_itself_survey_provenance": True,
        "mc_oflp200_04_05_06_still_point_at_FASER02_identity": all(
            dump.get("mc_oflp200_points_at_identity_FASER02") for dump in dumps.values()
        ),
        "data_reco_uses_condbr3_not_oflp200": True,
        "upgrade_to_survey_derived": False,
        "origin_summary": "alignment_origin_unknown for non-zero plane/module constants; survey_not_encoded_in_current_conditions for Stations station 0",
    }


def decide_next_stage(
    *,
    summary: Mapping[str, Any],
    comparison: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    survey_derived = any(
        row.get("origin") == "survey_derived" for row in comparison["comparisons"]
    )
    return {
        **common_audit_state(),
        "schema_version": SCHEMA_VERSION,
        "decision": DECISION,
        "current_reco_station0_Stations_ry_mrad": summary["station0_Stations_ry_cond_mrad"],
        "current_reco_ift_plane_ry_mrad": summary["ift_plane_ry_used_by_reconstruction_mrad"],
        "current_reco_C_dx_cond_mm": summary["C_dx_cond_mm"],
        "enough_provenance_for_independent_survey_constraint": False,
        "survey_derived_found": survey_derived,
        "named_2021_ppt_found": False,
        "next_allowed_step": (
            "Keep residual_dq_monitoring_only. Ingest a real independent 2021 "
            "CERN/UNIGE survey table into the entry-62 schema when the slides or "
            "metrology files are available. Do not use the dumped /Tracker/Align "
            "numbers as that independent constraint."
        ),
        "reason": (
            "CONDBR3 /Tracker/Align for current TI12Data04 reconstruction "
            "(OFLCOND-FASER-06, 2024 run 14973 → FASER-06_2024) has identity "
            "Stations station-0 ry and non-zero IFT /Planes ry (~74.6 mrad) and "
            "C_dx_cond (~3.58 mm). Those numbers are existing reconstruction "
            "alignment state. COOL tag descriptions are empty, POOL objects were "
            "written by WriteAlignmentAlg.AlignDbTool, and no author/input/"
            "comment links them to the 2021 CERN survey or UNIGE metrology. "
            "The named 2021 PPT was not found; published survey scales are "
            "incompatible with the ~70 mrad plane ry. Therefore the constants "
            "cannot be treated as a truly independent survey/metrology constraint."
        ),
        "found_explicit_provenance": provenance["found_explicit_survey_or_metrology_provenance"],
    }


def build_all_reports(config: Mapping[str, Any], raw: Mapping[str, Any], cool: Mapping[str, Any]) -> dict[str, Any]:
    dumps = {
        tag: build_tag_dump(tag, config=config, cool=cool, raw=raw)
        for tag in config["conditions_tags"]
    }
    diff = channel_diff_report(config=config, dumps=dumps)
    summary = ift_station0_ry_cdx_summary(config=config, dumps=dumps)
    comparison = survey_conditions_numeric_comparison(config=config, summary=summary, dumps=dumps)
    provenance = alignment_payload_provenance_report(config=config, cool=cool, dumps=dumps)
    decision = decide_next_stage(summary=summary, comparison=comparison, provenance=provenance)
    return {
        "dumps": dumps,
        "diff": diff,
        "summary": summary,
        "comparison": comparison,
        "provenance": provenance,
        "decision": decision,
    }
