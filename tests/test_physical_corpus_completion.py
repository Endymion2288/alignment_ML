from __future__ import annotations

import json

import scripts.build_physical_curriculum_corpus as corpus


def _audit(tracklets, *, stations=(0, 1, 2, 3)):
    return {
        "input": str(tracklets.resolve()),
        "events": 2,
        "tracklets": 8,
        "has_mc_labels": True,
        "station_counts": {str(station): 2 for station in stations},
        "covariance": {"rows": 8, "positive_definite_rows": 8},
    }


def test_physical_point_requires_audited_complete_refit_and_acts_assets(tmp_path, monkeypatch):
    tracklets = tmp_path / "tracklets.root"
    propagations = tmp_path / "propagations.root"
    payload = tmp_path / "alignment_payload.json"
    audit = tmp_path / "content_audit.json"
    failure = tmp_path / "failure.json"
    for path in (tracklets, propagations, payload):
        path.touch()
    audit.write_text(json.dumps(_audit(tracklets)), encoding="utf-8")

    monkeypatch.setattr(corpus, "_payload_matches_injection", lambda *_: (True, "accepted"))
    monkeypatch.setattr(corpus, "_mode0_propagation_completion", lambda *_: (True, "accepted"))
    accepted, status = corpus._physical_point_completion(
        tracklets=tracklets,
        propagations=propagations,
        payload_manifest=payload,
        content_audit=audit,
        failure=failure,
        station_ids=(0, 1, 2, 3),
        expected_offsets_xy_mm={str(station): [0.0, 0.0] for station in range(4)},
    )

    assert accepted is True
    assert status == "accepted"


def test_physical_point_rejects_missing_or_invalid_content_audit(tmp_path, monkeypatch):
    tracklets = tmp_path / "tracklets.root"
    propagations = tmp_path / "propagations.root"
    payload = tmp_path / "alignment_payload.json"
    audit = tmp_path / "content_audit.json"
    failure = tmp_path / "failure.json"
    for path in (tracklets, propagations, payload):
        path.touch()

    monkeypatch.setattr(corpus, "_payload_matches_injection", lambda *_: (True, "accepted"))
    monkeypatch.setattr(corpus, "_mode0_propagation_completion", lambda *_: (True, "accepted"))
    accepted, status = corpus._physical_point_completion(
        tracklets=tracklets,
        propagations=propagations,
        payload_manifest=payload,
        content_audit=audit,
        failure=failure,
        station_ids=(0, 1, 2, 3),
        expected_offsets_xy_mm={str(station): [0.0, 0.0] for station in range(4)},
    )
    assert accepted is False
    assert status == "missing:content_audit"

    invalid = _audit(tracklets, stations=(0, 1, 2))
    audit.write_text(json.dumps(invalid), encoding="utf-8")
    accepted, status = corpus._physical_point_completion(
        tracklets=tracklets,
        propagations=propagations,
        payload_manifest=payload,
        content_audit=audit,
        failure=failure,
        station_ids=(0, 1, 2, 3),
        expected_offsets_xy_mm={str(station): [0.0, 0.0] for station in range(4)},
    )
    assert accepted is False
    assert status == "content_audit_missing_station:3"


def test_physical_point_rejects_propagation_without_mode0_graph_coverage(tmp_path, monkeypatch):
    tracklets = tmp_path / "tracklets.root"
    propagations = tmp_path / "propagations.root"
    payload = tmp_path / "alignment_payload.json"
    audit = tmp_path / "content_audit.json"
    failure = tmp_path / "failure.json"
    for path in (tracklets, propagations, payload):
        path.touch()
    audit.write_text(json.dumps(_audit(tracklets)), encoding="utf-8")
    monkeypatch.setattr(corpus, "_payload_matches_injection", lambda *_: (True, "accepted"))
    monkeypatch.setattr(
        corpus,
        "_mode0_propagation_completion",
        lambda *_: (False, "propagations_missing_mode0_pair:1->3"),
    )

    accepted, status = corpus._physical_point_completion(
        tracklets=tracklets,
        propagations=propagations,
        payload_manifest=payload,
        content_audit=audit,
        failure=failure,
        station_ids=(0, 1, 2, 3),
        expected_offsets_xy_mm={str(station): [0.0, 0.0] for station in range(4)},
    )

    assert accepted is False
    assert status == "propagations_missing_mode0_pair:1->3"


def test_physical_point_rejects_payload_that_does_not_match_the_scan_plan(tmp_path, monkeypatch):
    tracklets = tmp_path / "tracklets.root"
    propagations = tmp_path / "propagations.root"
    payload = tmp_path / "alignment_payload.json"
    audit = tmp_path / "content_audit.json"
    failure = tmp_path / "failure.json"
    for path in (tracklets, propagations, payload):
        path.touch()
    audit.write_text(json.dumps(_audit(tracklets)), encoding="utf-8")
    monkeypatch.setattr(
        corpus,
        "_payload_matches_injection",
        lambda *_: (False, "alignment_payload_offset_mismatch:2"),
    )

    accepted, status = corpus._physical_point_completion(
        tracklets=tracklets,
        propagations=propagations,
        payload_manifest=payload,
        content_audit=audit,
        failure=failure,
        station_ids=(0, 1, 2, 3),
        expected_offsets_xy_mm={str(station): [0.0, 0.0] for station in range(4)},
    )

    assert accepted is False
    assert status == "alignment_payload_offset_mismatch:2"
