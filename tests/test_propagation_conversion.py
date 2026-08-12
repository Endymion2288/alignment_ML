from __future__ import annotations

import awkward as ak
import numpy as np
import uproot

from datasets.propagation_loader import load_propagation_records
from scripts.convert_ntuple_tracklet_propagations import (
    PROPAGATION_BRANCHES,
    convert_ntuple_tracklet_propagations,
)


def test_propagation_converter_flattens_event_vectors(tmp_path):
    source = tmp_path / "enhanced.root"
    destination = tmp_path / "propagations.root"
    branches = {
        "run": np.asarray([4], dtype=np.int32),
        "eventID": np.asarray([9], dtype=np.int32),
    }
    for canonical_name, branch_name in PROPAGATION_BRANCHES.items():
        if canonical_name in {"success", "has_covariance"}:
            values = [True, False]
        elif canonical_name in {"source_tracklet_id", "target_tracklet_id", "source_station_id", "target_station_id", "truth_particle_id"}:
            values = [1, 2]
        elif canonical_name.startswith("pred_cov_"):
            values = [1.0 if canonical_name in {"pred_cov_xx_mm2", "pred_cov_yy_mm2", "pred_cov_txtx", "pred_cov_tyty"} else 0.0, 0.0]
        else:
            values = [3.0, 4.0]
        branches[branch_name] = ak.Array([values])
    with uproot.recreate(source) as root_file:
        root_file["nt"] = branches

    summary = convert_ntuple_tracklet_propagations(source, destination)
    records = load_propagation_records(destination)

    assert summary.events_read == 1
    assert summary.records_written == 2
    assert summary.successful_records == 1
    assert records.size == 2
    assert records.success.tolist() == [True, False]
    assert records.prediction[:, 0].tolist() == [3.0, 4.0]
