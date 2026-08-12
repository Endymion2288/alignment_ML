from __future__ import annotations

import pytest

from scripts.config_loader import load_yaml_with_base


def test_relative_base_config_merges_mappings_and_replaces_lists(tmp_path):
    base = tmp_path / "base.yaml"
    child = tmp_path / "child.yaml"
    base.write_text(
        "root:\n  nested:\n    retained: 1\n    overridden: old\n  sources: [base_source]\n",
        encoding="utf-8",
    )
    child.write_text(
        "base_config: base.yaml\nroot:\n  nested:\n    overridden: new\n  sources: [child_source]\n",
        encoding="utf-8",
    )

    loaded = load_yaml_with_base(child)

    assert loaded == {
        "root": {
            "nested": {"retained": 1, "overridden": "new"},
            "sources": ["child_source"],
        }
    }


def test_base_config_cycle_is_rejected(tmp_path):
    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"
    first.write_text("base_config: second.yaml\n", encoding="utf-8")
    second.write_text("base_config: first.yaml\n", encoding="utf-8")

    with pytest.raises(ValueError, match="cyclic base_config"):
        load_yaml_with_base(first)
