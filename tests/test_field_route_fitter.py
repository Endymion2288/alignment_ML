from __future__ import annotations

import numpy as np

from baselines.field_chi2_matching import FieldCandidate, ScoredMatch
from baselines.route_assignment import Route
from models.field_route_fitter import (
    candidate_lookup,
    field_aware_route_summary,
    selected_field_aware_route,
)


def _candidate(source: int, target: int, source_station: int, target_station: int, chi2: float) -> FieldCandidate:
    covariance = np.diag([0.04, 0.09, 1.0e-6, 2.0e-6])
    residual = np.asarray([float(source - target), 0.2, -0.003, 0.004])
    return FieldCandidate(
        source_index=source,
        target_index=target,
        source_station=source_station,
        target_station=target_station,
        chi2=chi2,
        residual=residual,
        pull=residual / np.sqrt(np.diag(covariance)),
        combined_covariance=covariance,
    )


def test_selected_field_aware_route_preserves_existing_candidate_residuals():
    first = _candidate(4, 8, 0, 1, 2.5)
    second = _candidate(8, 13, 1, 2, 3.5)
    route = Route(
        endpoints=((0, 4), (1, 8), (2, 13)),
        matches=(
            ((0, 1), ScoredMatch(source_index=4, target_index=8, chi2=2.5, score=0.9)),
            ((1, 2), ScoredMatch(source_index=8, target_index=13, chi2=3.5, score=0.8)),
        ),
        utility=1.0,
    )

    selected = selected_field_aware_route(
        route,
        candidate_lookup({(0, 1): (first,), (1, 2): (second,)}),
    )

    assert selected.endpoint_indices == (4, 8, 13)
    assert selected.endpoint_stations == (0, 1, 2)
    assert selected.edge_count == 2
    assert selected.edge_chi2_sum == 6.0
    assert np.array_equal(selected.edges[0].residual, first.residual)
    assert np.array_equal(selected.edges[1].combined_covariance, second.combined_covariance)
    summary = field_aware_route_summary((selected,))
    assert summary["selected_field_aware_edges"] == 2
    assert summary["edge_chi2_mean"] == 3.0
    assert "not an independent global likelihood" in str(summary["statistical_interpretation"])
