"""T00 evidence-registry contract tests.  No sealed paths are opened."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml

from datasets.evidence_registry import (
    AUDIT_BASELINE_SHA,
    EvidenceRegistryError,
    build_registry,
    load_config,
    verify_records,
)

OFFICIAL = Path("configs/audit_evidence_registry_v1.yaml")


def _write(path: Path, text: str) -> str:
    path.write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _minimal_config(tmp_path: Path, *, extra_evidence=None) -> dict:
    artifact = tmp_path / "e01.json"
    digest = _write(artifact, '{"ok": true}\n')
    config = {
        "schema_version": "master-audit-reconciliation-v1",
        "task": "T00",
        "audit_baseline_sha": AUDIT_BASELINE_SHA,
        "do_not_backfill_post_audit_into_master_audit": True,
        "do_not_reread_sealed_test": True,
        "do_not_rewrite_frozen_negatives": True,
        "geometry_write_allowed": False,
        "pipeline_entries": {
            "inference": "control",
            "paired_response_analysis": "diagnostic",
            "real_data_monitoring": "dq",
        },
        "evidence": [
            {
                "id": "E01",
                "kind": "paired_response_analysis",
                "relative_path": str(artifact),
                "expected_sha256": digest,
                "lineage": "audit_baseline",
                "claim": "fixture",
            }
        ],
        "findings": {"F01_bootstrap_boolean_mask": "still_open"},
        "post_audit_lineage": {
            "note": "not audit evidence",
            "entries": [{"workbook": 87, "backfill_into_master_audit": False}],
        },
        "software_provenance_pins": {
            "calypso_git_sha": "40892527e9c65409afd2378a2abfc25ddbddac03",
            "historical_runtime_unverified": True,
        },
        "output_root": str(tmp_path / "out"),
    }
    if extra_evidence:
        config["evidence"].extend(extra_evidence)
    return config


def test_official_config_loads():
    config = load_config(OFFICIAL)
    assert config["task"] == "T00"
    assert config["audit_baseline_sha"] == AUDIT_BASELINE_SHA
    ids = [row["id"] for row in config["evidence"]]
    assert ids[:9] == [
        "E01",
        "E02",
        "E02_arrays",
        "E03",
        "E04",
        "E05",
        "E06",
        "E07",
        "E08",
    ]
    assert "E09" in ids
    kinds = {row["kind"] for row in config["evidence"]}
    assert kinds <= {
        "inference",
        "paired_response_analysis",
        "real_data_monitoring",
    }
    assert config["findings"]["F01_bootstrap_boolean_mask"] == "still_open"
    assert config["findings"]["F05_covariance_transport"] == (
        "post_audit_investigated_not_a_pass"
    )
    assert all(
        item["backfill_into_master_audit"] is False
        for item in config["post_audit_lineage"]["entries"]
    )


def test_rejects_duplicate_id(tmp_path: Path):
    config = _minimal_config(tmp_path)
    config["evidence"].append(dict(config["evidence"][0], id="E01", relative_path="other.json"))
    with pytest.raises(EvidenceRegistryError, match="duplicate evidence id"):
        verify_records(config)


def test_rejects_duplicate_path(tmp_path: Path):
    config = _minimal_config(tmp_path)
    path = config["evidence"][0]["relative_path"]
    config["evidence"].append(
        {
            "id": "E99",
            "kind": "inference",
            "relative_path": path,
            "expected_sha256": "a" * 64,
        }
    )
    with pytest.raises(EvidenceRegistryError, match="duplicate evidence path"):
        verify_records(config)


def test_rejects_missing_hash(tmp_path: Path):
    config = _minimal_config(tmp_path)
    config["evidence"][0]["expected_sha256"] = "deadbeef"
    with pytest.raises(EvidenceRegistryError, match="missing/invalid sha256"):
        verify_records(config)


def test_rejects_unknown_kind(tmp_path: Path):
    config = _minimal_config(tmp_path)
    config["evidence"][0]["kind"] = "new_payload"
    with pytest.raises(EvidenceRegistryError, match="unknown evidence kind"):
        verify_records(config)


def test_rejects_sealed_test_path(tmp_path: Path):
    config = _minimal_config(tmp_path)
    config["evidence"][0]["relative_path"] = "outputs/sealed/test/events.root"
    config["evidence"][0]["expected_sha256"] = "a" * 64
    with pytest.raises(EvidenceRegistryError, match="sealed test"):
        verify_records(config)


def test_missing_artifact_is_missing_not_pass(tmp_path: Path):
    config = _minimal_config(tmp_path)
    config["evidence"][0]["relative_path"] = str(tmp_path / "absent.json")
    config["evidence"][0]["expected_sha256"] = "a" * 64
    rows = verify_records(config)
    assert rows[0]["status"] == "MISSING"
    assert rows[0]["status"] != "PASS"


def test_hash_mismatch_is_not_pass(tmp_path: Path):
    config = _minimal_config(tmp_path)
    config["evidence"][0]["expected_sha256"] = "b" * 64
    rows = verify_records(config)
    assert rows[0]["status"] == "HASH_MISMATCH"
    assert rows[0]["exists"] is True


def test_matching_hash_passes(tmp_path: Path):
    config = _minimal_config(tmp_path)
    rows = verify_records(config)
    assert rows[0]["status"] == "PASS"
    payload = build_registry(config)
    assert payload["claim_reconciliation"]["verdict"] == "PASS"
    assert payload["claim_reconciliation"]["do_not_backfill_post_audit_into_master_audit"]
    assert payload["claim_reconciliation"]["post_audit_lineage"]["entries"][0][
        "backfill_into_master_audit"
    ] is False


def test_rejects_geometry_write(tmp_path: Path):
    config = _minimal_config(tmp_path)
    config["geometry_write_allowed"] = True
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(EvidenceRegistryError, match="geometry_write_allowed"):
        load_config(path)


def test_rejects_backfill_disabled_false(tmp_path: Path):
    config = _minimal_config(tmp_path)
    config["do_not_backfill_post_audit_into_master_audit"] = False
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(EvidenceRegistryError, match="backfill"):
        load_config(path)


def test_config_input_kind_mismatch_rejected(tmp_path: Path):
    config = _minimal_config(tmp_path)
    config["evidence"][0]["kind"] = "paired_response_analysis"
    config["evidence"].append(
        {
            "id": "E09",
            "kind": "not_a_kind",
            "relative_path": str(tmp_path / "other.json"),
            "expected_sha256": "c" * 64,
        }
    )
    with pytest.raises(EvidenceRegistryError, match="unknown evidence kind"):
        verify_records(config)
