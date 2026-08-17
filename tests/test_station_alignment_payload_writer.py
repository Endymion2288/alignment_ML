from __future__ import annotations

import argparse

import pytest

from scripts.write_station_alignment_payload import parse_transform


def test_parse_rigid_station_transform_uses_mm_and_rad_components():
    station, transform = parse_transform("0:1.5:-2.0:0:0:0.06:0")

    assert station == 0
    assert transform == (1.5, -2.0, 0.0, 0.0, 0.06, 0.0)


@pytest.mark.parametrize(
    "value",
    (
        "0:0:0:0:0:0",
        "4:0:0:0:0:0:0",
        "0:0:0:0:0:not-a-number:0",
    ),
)
def test_parse_rigid_station_transform_rejects_invalid_input(value):
    with pytest.raises(argparse.ArgumentTypeError):
        parse_transform(value)
