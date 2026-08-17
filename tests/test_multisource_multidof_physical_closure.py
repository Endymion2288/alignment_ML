from __future__ import annotations

import numpy as np
import pytest

from scripts.run_multisource_multidof_physical_closure import _parse_tolerances, _probe


def test_nominal_reference_closure_accepts_untagged_central_probes():
    points = {
        "nominal": {"name": "nominal", "point_role": "nominal"},
        "dx_p": {
            "name": "dx_p",
            "finite_difference_for": "ift_dx_mm",
            "probe_sign": "positive",
        },
        "dx_m": {
            "name": "dx_m",
            "finite_difference_for": "ift_dx_mm",
            "probe_sign": "negative",
        },
    }

    assert _probe(points, "ift_dx_mm", "positive", "nominal")["name"] == "dx_p"
    assert _probe(points, "ift_dx_mm", "negative", "nominal")["name"] == "dx_m"


def test_anchor_tagged_probe_requires_the_requested_reference():
    points = {
        "correct": {
            "name": "correct",
            "finite_difference_for": "ift_dx_mm",
            "probe_sign": "positive",
            "finite_difference_anchor": "anchor_a",
        },
        "other": {
            "name": "other",
            "finite_difference_for": "ift_dx_mm",
            "probe_sign": "positive",
            "finite_difference_anchor": "anchor_b",
        },
    }

    assert _probe(points, "ift_dx_mm", "positive", "anchor_a")["name"] == "correct"
    with pytest.raises(ValueError, match="exactly one"):
        _probe(points, "ift_dx_mm", "positive", "nominal")


def test_named_capture_tolerances_preserve_native_parameter_order():
    values = _parse_tolerances(
        ["ift_ry_mrad:1.0", "ift_dx_mm:0.1", "ift_dy_mm:0.2"],
        ("ift_dx_mm", "ift_dy_mm", "ift_ry_mrad"),
    )

    assert np.allclose(values, [0.1, 0.2, 1.0])
