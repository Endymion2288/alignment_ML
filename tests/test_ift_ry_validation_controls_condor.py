from __future__ import annotations

import pytest

from scripts.submit_ift_ry_validation_controls_condor import (
    PROJECT_ROOT,
    _parse_control,
    _validate_config,
    _write_submit,
)


def test_control_parser_allows_only_frozen_control_kinds(tmp_path):
    kind, config, output = _parse_control(f"mlp:configs/control.yaml:{tmp_path / 'output'}")

    assert kind == "mlp"
    assert config == (PROJECT_ROOT / "configs" / "control.yaml").resolve()
    assert output == (tmp_path / "output").resolve()
    route_kind, _, _ = _parse_control(f"mlp_route:configs/control.yaml:{tmp_path / 'route-output'}")
    assert route_kind == "mlp_route"
    with pytest.raises(ValueError, match="unsupported control"):
        _parse_control("v3:configs/control.yaml:outputs/control")


@pytest.mark.parametrize(
    ("kind", "config"),
    (
        ("v1", "geometry_aware_transformer_v1_ift_ry_validation.yaml"),
        ("v2", "geometry_aware_transformer_v2_ift_ry_validation.yaml"),
        ("mlp_route", "geometry_aware_transformer_v2_ift_ry_validation.yaml"),
    ),
)
def test_transformer_control_configs_explicitly_forbid_test(kind, config):
    _validate_config(kind, PROJECT_ROOT / "configs" / config)


def test_gpu_submit_file_declares_one_gpu(tmp_path):
    controls = tmp_path / "controls.tsv"
    controls.write_text("mlp\tconfig.yaml\toutput\n", encoding="utf-8")
    submit = tmp_path / "controls.sub"

    _write_submit(
        submit,
        controls_path=controls,
        synthetic_manifest=tmp_path / "synthetic.json",
        log_dir=tmp_path / "logs",
        request_memory_mb=16000,
        request_cpus=4,
        job_flavour="tomorrow",
        mlp_checkpoint=None,
        mlp_calibration=None,
    )

    text = submit.read_text(encoding="utf-8")
    assert "request_gpus = 1" in text
    assert "TARGET.GPUs_Capability >= 7.5" in text
    assert "H100|A100|L40|RTX" in text
    assert "__none__" in text
    assert "queue kind, config, output_dir" in text


def test_gpu_submit_file_passes_frozen_route_artifacts(tmp_path):
    controls = tmp_path / "controls.tsv"
    controls.write_text("mlp_route\tconfig.yaml\toutput\n", encoding="utf-8")
    submit = tmp_path / "controls.sub"
    checkpoint = tmp_path / "model.pt"
    calibration = tmp_path / "route_calibration.json"

    _write_submit(
        submit,
        controls_path=controls,
        synthetic_manifest=tmp_path / "synthetic.json",
        log_dir=tmp_path / "logs",
        request_memory_mb=16000,
        request_cpus=4,
        job_flavour="tomorrow",
        mlp_checkpoint=checkpoint,
        mlp_calibration=calibration,
    )

    text = submit.read_text(encoding="utf-8")
    assert str(checkpoint) in text
    assert str(calibration) in text
