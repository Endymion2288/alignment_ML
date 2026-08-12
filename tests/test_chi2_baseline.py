from __future__ import annotations

from baselines.chi2_matching import build_candidates, greedy_one_to_one_match
from datasets.root_loader import load_events
from datasets.synthetic import make_synthetic_tracklet_root
from evaluation.metrics import AssociationMetrics, evaluate_event_matches


def test_synthetic_loader_and_chi2_baseline(tmp_path):
    root_path = make_synthetic_tracklet_root(tmp_path / "synthetic.root")
    events = load_events(root_path, require_mc_labels=True)
    assert len(events) == 4

    metrics = AssociationMetrics()
    for event in events:
        candidates = build_candidates(event, 0, 1, chi2_gate=50.0)
        matches = greedy_one_to_one_match(candidates)
        metrics.add(evaluate_event_matches(event, matches, 0, 1))

    assert metrics.possible_matches == 4
    assert metrics.correct_matches == 4
    assert metrics.efficiency == 1.0
    assert metrics.purity == 1.0

