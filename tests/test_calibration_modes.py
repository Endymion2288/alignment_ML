from __future__ import annotations

import pytest

from pathlib import Path

from alignment.calibration_modes import (
    IFT_INTERNAL_MODE,
    STATION_MODE,
    STATUS_CONTAMINATED,
    STATUS_EXCLUSIVE_VIOLATION,
    STATUS_PREREQUISITE_UNMET,
    STATUS_VALID,
    assert_exclusive_parameters,
    build_mode_validity_contract,
    evaluate_A_stability,
    evaluate_mode_validity,
    implied_unmodeled_cdx_um,
    load_mode_validity_contract,
    normalize_mode,
)
from alignment.hierarchical_v1 import C_DX, STATION_SOLVE_PARAMETERS


def _contract():
    a_dx = -59.21304180539863
    return build_mode_validity_contract(
        leakage_operator={
            "A_native_per_mm_C_dx": {
                "ift_dx_mm": a_dx,
                "ift_dy_mm": 1.702908121992806,
                "ift_dz_mm": 10.118594792888189,
                "ift_rx_mrad": -2.029981711910054,
                "ift_ry_mrad": -31.90780811244653,
                "ift_rz_mrad": -0.36665518323206925,
            },
            "disguised_um_or_urad_per_um_C_dx": {
                "ift_dx_mm": a_dx,
                "ift_dy_mm": 1.702908121992806,
                "ift_dz_mm": 10.118594792888189,
                "ift_rx_mrad": -2.029981711910054,
                "ift_ry_mrad": -31.90780811244653,
                "ift_rz_mrad": -0.36665518323206925,
            },
            "subspace_r2": 0.9996588932083365,
            "subspace_cosine": 0.9998294320574567,
            "normal_cc_native": 5633517.766004098,
            "used_prior_sigma_native": {"ift_dz_mm": 5.0},
        },
        reverse_B_native={
            "ift_dx_mm": -0.016498401374776463,
            "ift_dy_mm": -0.00029992371612075266,
            "ift_dz_mm": 0.00010974130641351963,
            "ift_rx_mrad": -0.00021903134857596354,
            "ift_ry_mrad": -0.0006916339378901584,
            "ift_rz_mrad": 0.0010213274524762013,
        },
        station_capture_path=__import__("pathlib").Path("outputs/mc24_ift_5dof_survey_dz_capture_criteria_train_v1/capture_criteria.json"),
        cdx_capture_path=__import__("pathlib").Path("outputs/mc24_ift_layer_contrast_2d_capture_criteria_train_v1/capture_criteria.json"),
        station_registered_sigma={
            "ift_dx_mm": 0.028859249028587053,
            "ift_dy_mm": 0.3132778835465969,
            "ift_dz_mm": 4.248025032585663,
            "ift_rx_mrad": 0.38413678173076177,
            "ift_ry_mrad": 0.11138846324360457,
            "ift_rz_mrad": 0.5017709511755527,
        },
        cdx_registered_sigma_mm=0.006908845941386675,
        statistical_max_abs_C_dx_um=1.4621398334896487,
        engineering_max_abs_C_dx_um=1.6888171414778206,
        A_stability={"transfer_max_rel_deviation": 0.10, "do_not_retune": True},
        provenance={"workbook": 45},
        created_utc="2026-08-21T00:00:00+00:00",
    )


def test_mode_aliases_and_exclusive_parameter_sets():
    assert normalize_mode("C_dx") == IFT_INTERNAL_MODE
    assert assert_exclusive_parameters("station", STATION_SOLVE_PARAMETERS) == STATION_MODE
    assert assert_exclusive_parameters("ift_internal", [C_DX]) == IFT_INTERNAL_MODE
    with pytest.raises(ValueError, match="joint"):
        assert_exclusive_parameters("station", list(STATION_SOLVE_PARAMETERS) + [C_DX])
    with pytest.raises(ValueError, match="only C_dx"):
        assert_exclusive_parameters("ift_internal", ["C_dx", "C_rx"])


def test_station_mode_requires_unmodeled_cdx_below_statistical_budget():
    contract = _contract()
    ok = evaluate_mode_validity(
        contract,
        mode=STATION_MODE,
        floated_parameters=STATION_SOLVE_PARAMETERS,
        unmodeled_abs_C_dx_um=1.0,
        cdx_fixed_by="isolation_zero",
    )
    assert ok["status"] == STATUS_VALID
    assert ok["geometry_write_allowed"] is True
    assert ok["cross_level_contaminated"] is False

    mid = evaluate_mode_validity(
        contract,
        mode=STATION_MODE,
        floated_parameters=STATION_SOLVE_PARAMETERS,
        unmodeled_abs_C_dx_um=1.6,
        cdx_fixed_by="dedicated_calibration",
    )
    assert mid["status"] == STATUS_CONTAMINATED
    assert mid["unmodeled_C_dx"]["within_engineering_budget"] is True
    assert mid["geometry_write_allowed"] is False

    high = evaluate_mode_validity(
        contract,
        mode=STATION_MODE,
        floated_parameters=STATION_SOLVE_PARAMETERS,
        unmodeled_abs_C_dx_um=6.91,
        cdx_fixed_by="dedicated_calibration",
    )
    assert high["cross_level_contaminated"] is True
    assert "unmodeled_C_dx_exceeds_statistical_budget" in high["reject_indicators"]
    assert high["unmodeled_C_dx"]["within_engineering_budget"] is False

    missing = evaluate_mode_validity(
        contract,
        mode=STATION_MODE,
        floated_parameters=STATION_SOLVE_PARAMETERS,
    )
    assert missing["status"] == STATUS_PREREQUISITE_UNMET
    assert "C_dx_not_declared_fixed" in missing["reject_indicators"]


def test_registered_cdx_sigma_cannot_satisfy_station_budget():
    contract = _contract()
    sigma_um = contract["station_mode"]["prerequisite"]["registered_C_dx_sigma_um"]
    budget = contract["station_mode"]["unmodeled_C_dx"]["statistical_max_abs_um"]
    assert sigma_um > budget


def test_ift_internal_mode_requires_frozen_station_capture_and_records_systematic():
    contract = _contract()
    ok = evaluate_mode_validity(
        contract,
        mode=IFT_INTERNAL_MODE,
        floated_parameters=[C_DX],
        station_framework_capture_success=True,
    )
    assert ok["status"] == STATUS_VALID
    assert ok["geometry_write_allowed"] is True
    rss = ok["station_to_C_dx_propagation"]["rss_1sigma_um"]
    assert 0.5 < rss < 1.0

    failed = evaluate_mode_validity(
        contract,
        mode=IFT_INTERNAL_MODE,
        floated_parameters=[C_DX],
        station_framework_capture_success=False,
    )
    assert failed["status"] == STATUS_PREREQUISITE_UNMET
    assert failed["geometry_write_allowed"] is False

    same_stage = evaluate_mode_validity(
        contract,
        mode=IFT_INTERNAL_MODE,
        floated_parameters=[C_DX],
        station_framework_capture_success=True,
        same_data_stage_as_other_mode=True,
    )
    assert same_stage["status"] == STATUS_EXCLUSIVE_VIOLATION
    assert "same_data_stage_cross_mode_iteration" in same_stage["reject_indicators"]


def test_A_stability_gate_does_not_retune():
    contract = _contract()
    frozen = contract["leakage_operator"]["A_native_per_mm_C_dx"]
    stable = evaluate_A_stability(contract, measured_A_dx=frozen["ift_dx_mm"], measured_A_ry=frozen["ift_ry_mrad"])
    assert stable["stable"] is True
    assert stable["do_not_retune"] is True
    shifted = evaluate_A_stability(contract, measured_A_dx=-40.0)
    assert shifted["stable"] is False


def test_implied_cdx_from_station_dx_uses_frozen_A():
    a_dx = -59.21304180539863
    implied = implied_unmodeled_cdx_um(station_dx_mm=5.431, A_dx=a_dx)
    assert implied == pytest.approx(1.0e3 * 5.431 / a_dx)
    contract = _contract()
    scored = evaluate_mode_validity(
        contract,
        mode=STATION_MODE,
        floated_parameters=STATION_SOLVE_PARAMETERS,
        unmodeled_abs_C_dx_um=0.0,
        cdx_fixed_by="isolation_zero",
        implied_C_dx_um_from_station_dx=abs(implied),
    )
    assert scored["cross_level_contaminated"] is True
    assert "implied_C_dx_from_station_dx_exceeds_budget" in scored["reject_indicators"]


def test_residual_reduction_is_never_success():
    contract = _contract()
    scored = evaluate_mode_validity(
        contract,
        mode=STATION_MODE,
        floated_parameters=STATION_SOLVE_PARAMETERS,
        unmodeled_abs_C_dx_um=0.0,
        cdx_fixed_by="isolation_zero",
        residual_reduction_used_as_success=True,
    )
    assert scored["geometry_write_allowed"] is False
    assert scored["residual_reduction_is_not_alignment_success"] is True


def test_registered_disk_contract_matches_workbook_45():
    path = Path("outputs/mc24_ift_calibration_mode_validity_contract_v1/mode_validity_contract.json")
    contract = load_mode_validity_contract(path)
    assert contract["hierarchical_v1_rescue_closed"] is True
    assert contract["schur_projection_production"] is False
    assert contract["leakage_operator"]["A_native_per_mm_C_dx"]["ift_dx_mm"] == pytest.approx(-59.21304180539863)
    budget = contract["station_mode"]["unmodeled_C_dx"]
    assert budget["statistical_max_abs_um"] == pytest.approx(1.4621398334896487)
    assert budget["engineering_max_abs_um"] == pytest.approx(1.6888171414778206)
    rss = contract["ift_internal_mode"]["station_to_C_dx_propagation"]["free_station_rss"]["rss_1sigma_um"]
    assert rss == pytest.approx(0.7149628952946077, rel=1.0e-9)


def test_evaluate_require_mode_valid_rejects_A_instability(tmp_path, monkeypatch):
    import json
    import scripts.evaluate_calibration_mode_validity as evaluate

    contract = _contract()
    contract_path = tmp_path / "mode_validity_contract.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    output = tmp_path / "report.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluate_calibration_mode_validity.py",
            "--contract",
            str(contract_path),
            "--mode",
            "station",
            "--floated-parameters",
            *STATION_SOLVE_PARAMETERS,
            "--cdx-fixed-by",
            "isolation_zero",
            "--declared-unmodeled-cdx-um",
            "0.0",
            "--measured-A-dx",
            "-40.0",
            "--measured-A-ry",
            "-31.90780811244653",
            "--output",
            str(output),
            "--require-mode-valid",
        ],
    )
    with pytest.raises(SystemExit, match="geometry-write candidate"):
        evaluate.main()
    report = json.loads(output.read_text())
    assert report["mode_validity"]["geometry_write_allowed"] is False
    assert report["A_stability"]["stable"] is False
    assert report["residual_reduction_is_not_alignment_success"] is True


def test_transfer_matrix_does_not_retune_on_failure(tmp_path, monkeypatch):
    import json
    from scripts.report_calibration_mode_transfer_matrix import main as matrix_main

    def _cell(*, allowed: bool, capture: bool, error: float) -> dict:
        return {
            "capture_success": capture,
            "unique_physical_edges": 10,
            "normal_matrix_condition_number": 2.0,
            "post_over_pre_rms": 0.1,
            "mode_validity": {
                "geometry_write_allowed": allowed,
                "reject_indicators": [] if allowed else ["transfer_failure"],
            },
            "parameters": [{"name": "ift_dx_mm", "error": error, "fit_sigma": 0.01, "pull": error / 0.01}],
        }

    current = {
        "station_mode": {"train": _cell(allowed=True, capture=True, error=0.01), "validation": _cell(allowed=True, capture=True, error=0.02)},
        "ift_internal_mode": {"train": _cell(allowed=True, capture=True, error=0.001), "validation": _cell(allowed=True, capture=True, error=0.002)},
    }
    transfer = {
        "station_mode": {"train": _cell(allowed=True, capture=True, error=0.011), "validation": _cell(allowed=False, capture=False, error=0.20)},
        "ift_internal_mode": {"train": _cell(allowed=True, capture=True, error=0.0011), "validation": _cell(allowed=True, capture=True, error=0.0021)},
    }
    current_path = tmp_path / "current.json"
    transfer_path = tmp_path / "transfer.json"
    output = tmp_path / "matrix.json"
    current_path.write_text(json.dumps(current), encoding="utf-8")
    transfer_path.write_text(json.dumps(transfer), encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        [
            "report_calibration_mode_transfer_matrix.py",
            "--current-report",
            str(current_path),
            "--transfer-report",
            str(transfer_path),
            "--output",
            str(output),
        ],
    )
    matrix_main()
    payload = json.loads(output.read_text())
    assert payload["mc_transfer_gate_passed"] is False
    assert payload["real_data_dry_run_allowed"] is False
    assert "station_mode/validation" in payload["failed_cells"]
    assert payload["do_not_retune_V2_capture_or_A"] is True
    assert payload["residual_reduction_is_not_alignment_success"] is True
