from __future__ import annotations

import pytest

from scripts.assemble_physical_curriculum_manifest import _validate_entry


def test_assembled_manifest_requires_complete_physical_assets(tmp_path):
    assets = {}
    for field in ("physical_tracklets", "physical_propagations", "physical_payload_manifest"):
        path = tmp_path / f"{field}.root"
        path.touch()
        assets[field] = str(path)
    entry = {
        "source_id": "source_a",
        "split": "train",
        "input_xaod": str(tmp_path / "source.root"),
        "source_event_uids": ["source_a:1:2"],
        "points": [{"completed": True, **assets}],
    }
    _validate_entry(entry, ("train", str(tmp_path / "source.root")))
    entry["points"][0]["completed"] = False
    with pytest.raises(ValueError, match="incomplete"):
        _validate_entry(entry, ("train", str(tmp_path / "source.root")))
