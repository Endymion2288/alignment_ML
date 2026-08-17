from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from scripts.prepare_multisource_multidof_iteration import _values_from_update, prepare_iteration
from scripts.run_multisource_refit_multidof_local_step import ResponseBank, _apply_update
from scripts.run_physical_refit_capture_scan import _build_plan


def _template(path: Path) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "physical_refit_capture_scan": {
                    "scan_mode": "station_rigid_multidof",
                    "input_xaod": "/unused/template.root",
                    "nevents": 1,
                    "station_ids": [0, 1, 2, 3],
                    "reference_station_ids": [1, 2, 3],
                    "movable_station_ids": [0],
                    "condition_axis": "ift_dx_dy_ry_joint_l2",
                    "q_over_p_mode": 0,
                    "min_truth_match_fraction": 0.99,
                    "chi2_gate": 25.0,
                    "refinement_iterations": 1,
                    "alignment_parameter_specs": [
                        {
                            "name": "ift_dx_mm",
                            "station_id": 0,
                            "component": "dx_mm",
                            "unit": "mm",
                            "finite_difference_step": 0.5,
                            "severity_scale": 5.0,
                        },
                        {
                            "name": "ift_dy_mm",
                            "station_id": 0,
                            "component": "dy_mm",
                            "unit": "mm",
                            "finite_difference_step": 0.5,
                            "severity_scale": 5.0,
                        },
                        {
                            "name": "ift_ry_mrad",
                            "station_id": 0,
                            "component": "ry_mrad",
                            "unit": "mrad",
                            "finite_difference_step": 10.0,
                            "severity_scale": 60.0,
                        },
                    ],
                    "rigid_points": [
                        {
                            "name": "nominal",
                            "point_role": "nominal",
                            "direction_trial": "nominal",
                            "alignment_parameter_values": {
                                "ift_dx_mm": 0.0,
                                "ift_dy_mm": 0.0,
                                "ift_ry_mrad": 0.0,
                            },
                            "station_transforms": {station: [0.0] * 6 for station in range(4)},
                        }
                    ],
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def test_prepare_multisource_iteration_writes_common_anchor_probe_plan_without_test(tmp_path):
    train = tmp_path / "train.root"
    validation = tmp_path / "validation.root"
    train.touch()
    validation.touch()
    source_config = tmp_path / "sources.yaml"
    source_config.write_text(
        yaml.safe_dump(
            {
                "physical_curriculum_mlp": {
                    "allowed_splits": ["train", "validation"],
                    "forbidden_splits": ["test"],
                    "sources": [
                        {"id": "train_source", "split": "train", "input_xaod": str(train)},
                        {"id": "validation_source", "split": "validation", "input_xaod": str(validation)},
                    ],
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    template = tmp_path / "iteration.yaml"
    _template(template)
    output = tmp_path / "iteration"

    manifest = prepare_iteration(
        source_config_path=source_config,
        iteration_template_path=template,
        output_root=output,
        iteration=2,
        current_values={"ift_dx_mm": 2.0, "ift_dy_mm": -1.5, "ift_ry_mrad": 35.0},
        nevents=17,
    )

    assert manifest["test_data_accessed"] is False
    assert manifest["forbidden_splits"] == ["test"]
    assert {source["split"] for source in manifest["sources"]} == {"train", "validation"}
    assert {Path(source["input_xaod"]).name for source in manifest["sources"]} == {
        "train.root",
        "validation.root",
    }
    assert len(manifest["common_scan_plan"]["points"]) == 8
    for source in manifest["sources"]:
        config = yaml.safe_load(Path(source["physical_scan_config"]).read_text(encoding="utf-8"))
        plan = _build_plan(config["physical_refit_capture_scan"])
        assert [point["name"] for point in plan["points"][:2]] == [
            "iteration_02_reference",
            "iteration_02_anchor",
        ]
        assert plan["points"][1]["alignment_parameter_values"]["ift_ry_mrad"] == 35.0

    persisted = json.loads((output / "iteration_manifest.json").read_text(encoding="utf-8"))
    assert persisted["test_data_accessed"] is False


def test_prepare_multisource_iteration_requires_explicit_test_seal(tmp_path):
    source = tmp_path / "train.root"
    source.touch()
    source_config = tmp_path / "sources.yaml"
    source_config.write_text(
        yaml.safe_dump(
            {
                "physical_curriculum_mlp": {
                    "allowed_splits": ["train", "validation"],
                    "forbidden_splits": [],
                    "sources": [
                        {"id": "train_source", "split": "train", "input_xaod": str(source)},
                        {"id": "validation_source", "split": "validation", "input_xaod": str(source)},
                    ],
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    template = tmp_path / "iteration.yaml"
    _template(template)

    with pytest.raises(ValueError, match="explicitly forbid test"):
        prepare_iteration(
            source_config_path=source_config,
            iteration_template_path=template,
            output_root=tmp_path / "iteration",
            iteration=0,
            current_values={"ift_dx_mm": 0.0, "ift_dy_mm": 0.0, "ift_ry_mrad": 1.0},
            nevents=1,
        )


def test_held_out_application_only_evaluates_the_given_update():
    bank = ResponseBank(
        split="validation",
        parameter_names=("ift_dx_mm",),
        specs=({"name": "ift_dx_mm", "station_id": 0, "component": "dx_mm", "unit": "mm"},),
        parameter_scales=np.asarray([1.0]),
        anchor_values=np.asarray([2.0]),
        target_values=np.asarray([0.0]),
        positive_values=np.asarray([2.5]),
        negative_values=np.asarray([1.5]),
        anchor_transforms={"0": [2.0, 0.0, 0.0, 0.0, 0.0, 0.0]},
        anchor_residual=np.zeros((1, 4)),
        positive_residual=np.zeros((1, 1, 4)),
        negative_residual=np.zeros((1, 1, 4)),
        target_residual=np.asarray([[-2.0, 0.0, 0.0, 0.0]]),
        covariance=np.tile(np.eye(4), (1, 1, 1)),
        source_ids=np.asarray(["validation_source"], dtype=object),
        run_ids=np.asarray([1]),
        event_ids=np.asarray([1]),
        source_station_ids=np.asarray([0]),
        target_station_ids=np.asarray([1]),
        source_tracklet_ids=np.asarray([0]),
        target_tracklet_ids=np.asarray([0]),
        source_overlap=(),
    )
    derivative = np.asarray([[[1.0], [0.0], [0.0], [0.0]]])

    result = _apply_update(bank, derivative, np.asarray([-2.0]))

    assert result["baseline_response_chi2"] == 4.0
    assert result["post_update_response_chi2"] == 0.0
    assert result["response_chi2_reduction_fraction"] == 1.0


def test_iteration_update_json_requires_an_explicit_success_gate(tmp_path):
    update = tmp_path / "update.json"
    update.write_text(
        json.dumps(
            {
                "capture_success": False,
                "proposed_next_parameter_values": {
                    "ift_dx_mm": 0.1,
                    "ift_dy_mm": -0.2,
                    "ift_ry_mrad": 3.0,
                },
            }
        ),
        encoding="utf-8",
    )
    names = ("ift_dx_mm", "ift_dy_mm", "ift_ry_mrad")

    with pytest.raises(ValueError, match="capture_success=true"):
        _values_from_update(update, names, allow_unverified=False)

    values = _values_from_update(update, names, allow_unverified=True)
    assert values == {"ift_dx_mm": 0.1, "ift_dy_mm": -0.2, "ift_ry_mrad": 3.0}
