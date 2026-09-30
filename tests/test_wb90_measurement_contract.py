import json
from pathlib import Path

import numpy as np
import pytest

from alignment.wb90_measurement_contract import (
    selection, correct_angle_jacobian, numerical_angle_jacobian, write_new, read_public, selected_records,
)


def corpus():
    families = ("identity", "identifiable_translation", "identifiable_rotation", "weak_jg_diagnostic")
    allowed = {f"source{i}": f"/allowed/input{i}.root" for i in range(6)}
    events = []
    for source, path in allowed.items():
        for family in families:
            for _ in range(3):
                entry = 2000 + len(events)
                events.append({"source_id": source, "input_xaod": path, "xaod_entry_index": entry,
                               "survey_mode": "fixed_dz", "family": family, "index": len(events)})
    return {"events": events}, {"sources": [{"source_id": k, "path": v} for k, v in allowed.items()]}


def test_selection_is_order_invariant_and_entry_disjoint():
    data, allow = corpus()
    picked = selection(data, allow)
    assert selection({"events": list(reversed(data["events"]))}, allow) == picked
    assert len(picked) == 48
    sets = [{(r["input_xaod"], r["xaod_entry_index"]) for r in picked if r["role"] == role}
            for role in ("development", "check")]
    assert len(sets[0]) == len(sets[1]) == 24
    assert not sets[0] & sets[1]


@pytest.mark.parametrize("mutation", ["source", "duplicate", "cell"])
def test_corrupt_selection_fails_closed(mutation):
    data, allow = corpus()
    if mutation == "source":
        data["events"][0]["input_xaod"] = "/wrong/input.root"
    elif mutation == "duplicate":
        data["events"].append(dict(data["events"][0]))
    else:
        data["events"] = [r for r in data["events"] if (r["source_id"], r["family"]) != ("source0", "identity")]
    with pytest.raises(ValueError):
        selection(data, allow)


def test_phi_derivative_against_independent_finite_difference_all_quadrants():
    for tx in (-0.03, -0.002, 0.002, 0.03):
        for ty in (-0.004, -0.001, 0.001, 0.004):
            analytic = correct_angle_jacobian(tx, ty)
            numerical = numerical_angle_jacobian(tx, ty, 1e-7)
            np.testing.assert_allclose(analytic, numerical, rtol=1e-7, atol=1e-7)
            assert np.sign(analytic[0, 0]) == -np.sign(ty)


def test_zero_direction_is_invalid():
    with pytest.raises(ValueError):
        correct_angle_jacobian(0., 0.)


def test_write_never_overwrites_and_forbidden_read_fails_before_access(tmp_path):
    target = tmp_path / "report.json"
    write_new(target, {"value": 1})
    with pytest.raises(FileExistsError):
        write_new(target, {"value": 2})
    assert json.loads(target.read_text()) == {"value": 1}
    with pytest.raises(ValueError):
        read_public(tmp_path / "00800_00849" / "missing.json")


def test_nonselected_measurements_are_not_decoded(tmp_path):
    # An invalid nonselected JSON payload would fail if json.loads saw it.
    # This verifies isolation before decoding, not just result filtering.
    shard = tmp_path / "released.jsonl"
    shard.write_text('{"event_uid":"other", "measurement": invalid}\n' +
                     '{"event_uid":"wanted", "measurement":42}\n')
    assert selected_records(shard, "wanted") == [{"event_uid": "wanted", "measurement": 42}]
