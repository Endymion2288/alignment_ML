from __future__ import annotations

from pathlib import Path

import pytest

import json

import scripts.submit_physical_curriculum_condor as submit
from scripts.submit_physical_curriculum_condor import PROJECT_ROOT, WORKER, _require_eos_path, _write_submit


def test_eossubmit_descriptor_passes_explicit_eos_project_root_to_staged_worker(tmp_path):
    target = tmp_path / "physical_curriculum.sub"
    _write_submit(
        target,
        output_root=Path("/eos/home-x/xcheng/FASER/alignment_ML/outputs/physical"),
        source_ids_path=Path("/eos/home-x/xcheng/FASER/alignment_ML/outputs/submit/source_ids.txt"),
        log_root=Path("/eos/home-x/xcheng/FASER/alignment_ML/outputs/submit/logs"),
        request_memory_mb=6000,
        job_flavour="tomorrow",
        schedd_mode="eossubmit",
    )

    text = target.read_text(encoding="utf-8")
    assert f"executable = {WORKER}" in text
    assert f"arguments = /eos/home-x/xcheng/FASER/alignment_ML/outputs/physical $(source_id) {PROJECT_ROOT}" in text
    assert "log = /eos/home-x/xcheng/FASER/alignment_ML/outputs/submit/logs/physical_curriculum.$(ClusterId).log" in text
    assert "$(ProcId).log" not in text


def test_eossubmit_rejects_non_eos_submit_paths(tmp_path):
    with pytest.raises(ValueError, match="under /eos"):
        _require_eos_path(tmp_path / "not_eos", label="worker")


def test_skip_complete_uses_the_same_physical_acceptance_gate(tmp_path, monkeypatch):
    source_root = tmp_path / "source"
    scan = source_root / "physical_scan"
    point_root = scan / "points" / "mag_0_train_00"
    point_root.mkdir(parents=True)
    (scan / "scan_plan.json").write_text(
        json.dumps(
            {
                "station_ids": [0, 1, 2, 3],
                "points": [
                    {
                        "relative_point_dir": "points/mag_0_train_00",
                        "injected_offsets_xy_mm": {
                            "0": [0.0, 0.0],
                            "1": [0.0, 0.0],
                            "2": [0.0, 0.0],
                            "3": [0.0, 0.0],
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    calls = []

    def fake_completion(**kwargs):
        calls.append(kwargs)
        return True, "accepted"

    monkeypatch.setattr(submit, "_physical_point_completion", fake_completion)

    assert submit._complete_source(source_root) is True
    assert len(calls) == 1
    assert calls[0]["station_ids"] == (0, 1, 2, 3)
    assert calls[0]["expected_offsets_xy_mm"]["3"] == [0.0, 0.0]
