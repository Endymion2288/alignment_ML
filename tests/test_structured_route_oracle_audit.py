from __future__ import annotations

from scripts.audit_structured_route_oracle import _quantiles


def test_structured_oracle_audit_preserves_float_timing_maximum():
    summary = _quantiles([0.001, 0.004, 0.002])

    assert summary["count"] == 3
    assert summary["max"] == 0.004


def test_structured_oracle_audit_keeps_integer_route_maximum_integral():
    summary = _quantiles([3, 9, 5])

    assert summary["count"] == 3
    assert summary["max"] == 9
