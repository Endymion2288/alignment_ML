from __future__ import annotations

import ast
from pathlib import Path


def test_background_audit_exposes_an_explicit_split_filter():
    path = Path("scripts/audit_synthetic_background.py")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    source = ast.unparse(tree)

    assert "'--split'" in source
    assert "selected_samples" in source
    assert "test_opened" in source
