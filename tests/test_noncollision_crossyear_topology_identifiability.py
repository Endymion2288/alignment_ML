from __future__ import annotations

from pathlib import Path

import pytest

from alignment.noncollision_crossyear_topology import (
    DECISION_FOUND,
    DECISION_INSUFFICIENT,
    GO_NO_GO_QUESTION,
    SCHEMA_VERSION,
    assign_station_from_z,
    decide_next_stage,
    jacobian_admission_for_probe,
    load_inventory_config,
    parse_job_log,
    predeclared_topology_report,
)
from alignment.operating_protocol_v1_final_closure import (
    FROZEN_V2_CHECKPOINT_SHA256,
    OPERATING_MODE,
)
from alignment.true_cluster_local_residual import assert_no_alignment_payload


def _config():
    return load_inventory_config(
        Path("configs/noncollision_crossyear_topology_identifiability_v1.yaml")
    )


def test_config_freezes_protocol_and_forbids_cosine_and_year_mix():
    config = _config()
    assert config["schema_version"] == SCHEMA_VERSION
    assert config["real_data_operating_mode"] == OPERATING_MODE
    assert config["do_not_select_events_from_residual_or_cosine"] is True
    assert config["do_not_invent_new_cosine_cut"] is True
    assert config["do_not_restack_2024_r0022_collision_like"] is True
    assert config["do_not_mix_cross_year_residuals_or_alignment_constants"] is True
    assert config["do_not_enter_full_module_identifiability_map"] is True
    assert config["frozen_v2_checkpoint_sha256"] == FROZEN_V2_CHECKPOINT_SHA256
    physics = config["physics_scales"]
    assert physics["wide_angle_min_slope"] == pytest.approx(0.002)
    assert physics["ip_like_max_slope"] == pytest.approx(0.0005)
    assert physics["slope_not_local_segmentfit_tx_ty"] is True
    assert physics["slope_not_unassociated_event_segments"] is True
    rules = config["jacobian_admission"]
    assert rules["min_wide_dominated_ckf_fraction"] == pytest.approx(0.90)
    assert rules["collision_like_metadata_cannot_admit"] is True
    assert rules["forbid_cosine_event_selection"] is True
    ids = [row["id"] for row in config["predeclared_topologies"]]
    assert "phys_2024_cosmic_beam_mode" in ids
    assert "testbeam_2021" in ids
    assert all(row["can_mix_residuals_with_2024_r0022"] is False for row in config["predeclared_topologies"])


def test_station_assignment_and_predeclared_report_are_residual_blind():
    config = _config()
    physics = config["physics_scales"]
    assert assign_station_from_z(-1860.15, physics) == 0
    assert assign_station_from_z(47.4, physics) == 1
    assert assign_station_from_z(2427.4, physics) == 3
    assert assign_station_from_z(9000.0, physics) is None
    report = predeclared_topology_report(config)
    assert report["frozen_before_jacobian"] is True
    assert "cosine" in report["selection_forbidden"]
    assert_no_alignment_payload(report)


def test_parse_job_log_extracts_geom_and_cosmics(tmp_path: Path):
    log = tmp_path / "job.log"
    log.write_text(
        "Remaining: --geom TI12Data04 --noIFT --cosmics\n"
        "IOVDbSvc                                   INFO Global tag: OFLCOND-FASER-05 set from joboptions\n"
        "GeoModelSvc.VetoNuDetectorTool             INFO Building VetoNu with Version Tag: FASERNU-04 at Node: FASER\n"
        "Cosmics = True\n"
        "Stable Beams = False\n",
        encoding="utf-8",
    )
    parsed = parse_job_log(log)
    assert parsed["geometry_tag"] == "FASERNU-04"
    assert parsed["conditions_tag"] == "OFLCOND-FASER-05"
    assert parsed["geom_flag"] == "TI12Data04"
    assert parsed["no_ift_four_station_ckf"] is True
    assert parsed["cosmics_only"] is True
    tb = tmp_path / "tb.log"
    tb.write_text(
        "Geom: TI12Data03\n"
        "Starting reconstruction of Faser-Physics-011780-00000.raw with type TI12Data03\n"
        "IOVDbSvc                                   INFO Global tag: OFLCOND-FASER-04 set from joboptions\n"
        "GeoModelSvc.VetoNuDetectorTool             INFO Building VetoNu with Version Tag: FASERNU-03 at Node: FASER\n",
        encoding="utf-8",
    )
    cosmic_rec = parse_job_log(tb)
    assert cosmic_rec["geom_flag"] == "TI12Data03"
    assert cosmic_rec["geometry_tag"] == "FASERNU-03"


def test_jacobian_gate_rejects_collision_like_and_empty_cosmics():
    config = _config()
    baseline = config["r0022_collision_baseline"]
    rules = config["jacobian_admission"]
    collision = jacobian_admission_for_probe(
        {
            "id": "phys_2022_ift",
            "topology_id": "phys_ift_collision",
            "role": "candidate",
            "topology": {
                "n_file_events": 18000,
                "n_used": 18000,
                "ckf_single_track_slope": {"wide_fraction": 0.70, "n": 18000},
                "cluster_occupancy": {
                    "n_ge3_ift_events": 17000,
                    "four_station_coincidence": 0.93,
                },
            },
        },
        baseline=baseline,
        rules=rules,
    )
    assert collision["admitted_to_jacobian"] is False
    assert "collision-like" in collision["reason"]
    cosmic = jacobian_admission_for_probe(
        {
            "id": "phys_2024_cos",
            "topology_id": "phys_2024_cosmic_beam_mode",
            "role": "candidate",
            "topology": {
                "n_file_events": 164,
                "n_used": 164,
                "ckf_single_track_slope": {"n": 0, "wide_fraction": None},
                "cluster_occupancy": {
                    "n_ge3_ift_events": 0,
                    "four_station_coincidence": 0.0,
                },
            },
        },
        baseline=baseline,
        rules=rules,
    )
    assert cosmic["admitted_to_jacobian"] is False
    assert "tiny" in cosmic["reason"] or "ge3" in cosmic["reason"]
    # A hypothetical 2x-wide cosmic sample with IFT would still need metadata class + stats.
    hypothetical = jacobian_admission_for_probe(
        {
            "id": "phys_2024_cos_rich",
            "topology_id": "phys_2024_cosmic_beam_mode",
            "role": "candidate",
            "topology": {
                "n_file_events": 50000,
                "n_used": 20000,
                "ckf_single_track_slope": {"wide_fraction": 0.99, "n": 15000},
                "cluster_occupancy": {
                    "n_ge3_ift_events": 8000,
                    "four_station_coincidence": 0.40,
                },
            },
        },
        baseline=baseline,
        rules=rules,
    )
    assert hypothetical["admitted_to_jacobian"] is True


def test_decision_labels_and_no_module_map_without_portable_topology():
    no = decide_next_stage(admissions=[{"admitted_to_jacobian": False}], jacobian_executed=False, portable_within_period=False)
    assert no["answer"] == "No"
    assert no["decision"] == DECISION_INSUFFICIENT
    assert no["go_to_full_module_identifiability_map"] is False
    assert no["still_no_new_network"] is True
    assert no["go_no_go_question"] == GO_NO_GO_QUESTION
    assert "external_survey" in no["next_allowed_step"]
    assert_no_alignment_payload(no)
    yes = decide_next_stage(
        admissions=[{"id": "phys_2024_cos_rich", "admitted_to_jacobian": True}],
        jacobian_executed=True,
        portable_within_period=True,
    )
    assert yes["decision"] == DECISION_FOUND
    assert yes["go_to_full_module_identifiability_map"] is True
    assert yes["still_no_new_network"] is True
