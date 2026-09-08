"""Workbook-81b calibration loss, fake gate, and holdout audit.  Frozen objective."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import torch

from evaluation.route_counterfactuals import inclusion_gap
from training.explicit_route_energy import (
    PhysicalRoute,
    energy_table_from_utilities,
    route_feature_matrix,
    w64_route_utilities,
)
from training.wb81_calibration_contract import (
    ACCEPT_EFF_GAIN,
    BIN_WEIGHTS,
    DELTA_MAX,
    UNMATCHED_PENALTY,
    calibrated_energy_table,
    compare_identity,
    fake_hard_gate,
    margin_bin,
    selected_mask,
)
from training.geometry_aware_transformer import TransformerGraph


def differentiable_margin(energies: torch.Tensor, table, route_id: int) -> torch.Tensor:
    gap = inclusion_gap(table, int(route_id))
    in_idx = [int(item) for item in gap.forced_in_selected]
    out_idx = [int(item) for item in gap.forced_out_selected]
    forced_in = energies[in_idx].sum() if in_idx else energies.sum() * 0.0
    forced_out = energies[out_idx].sum() if out_idx else energies.sum() * 0.0
    return forced_in - forced_out


def calibration_event_loss(
    model,
    standardizer,
    graph: TransformerGraph,
    logits: Mapping[tuple[int, int], float],
    device: torch.device,
) -> dict[str, torch.Tensor] | None:
    """L = L_gap + L_keep.  No LAI.  bin_A and selected_A come from frozen U_W64."""
    from training.explicit_route_energy import enumerate_contiguous_physical_routes

    routes = enumerate_contiguous_physical_routes(graph)
    complete_truth = [index for index, route in enumerate(routes) if route.truth_consistent and len(route.stations) == 4]
    if not complete_truth:
        return None
    u_w64 = w64_route_utilities(graph, routes, logits, UNMATCHED_PENALTY)
    features = standardizer.transform(route_feature_matrix(graph, routes))
    delta = model(torch.as_tensor(features, dtype=torch.float32, device=device))
    u_new = torch.as_tensor(u_w64, dtype=torch.float32, device=device) + delta
    table_a = energy_table_from_utilities(routes, u_w64, unmatched_penalty=UNMATCHED_PENALTY)
    selected_a = set(selected_mask(table_a))
    bins = {}
    for route_id in complete_truth:
        bins[route_id] = margin_bin(float(inclusion_gap(table_a, route_id).inclusion_gap))
    table_new = energy_table_from_utilities(routes, u_new.detach().cpu().tolist(), unmatched_penalty=UNMATCHED_PENALTY)
    gap_terms = []
    keep_terms = []
    for route_id in complete_truth:
        margin = differentiable_margin(u_new, table_new, route_id)
        gap_terms.append(float(BIN_WEIGHTS[bins[route_id]]) * torch.relu(-margin))
        if bins[route_id] == "near_zero" or route_id in selected_a:
            keep_terms.append(torch.relu(-margin))
    gap_loss = torch.stack(gap_terms).mean()
    keep_loss = torch.stack(keep_terms).mean() if keep_terms else u_new.sum() * 0.0
    return {"loss": gap_loss + keep_loss, "l_gap": gap_loss, "l_keep": keep_loss}


def score_routes(model, standardizer, graph, routes, logits, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    u_w64 = w64_route_utilities(graph, routes, logits, UNMATCHED_PENALTY)
    features = standardizer.transform(route_feature_matrix(graph, routes))
    with torch.no_grad():
        delta = model(torch.as_tensor(features, dtype=torch.float32, device=device)).detach().cpu().numpy()
    return u_w64, delta.astype(np.float64, copy=False)


def identity_on_graphs(graphs, adjacent_sets, logits, account_event) -> dict[str, object]:
    from training.explicit_route_energy import enumerate_contiguous_physical_routes

    selected_a_all = []
    selected_new_all = []
    accounting_a = None
    accounting_new = None
    per_event_selected_equal = True
    for graph in graphs:
        routes = enumerate_contiguous_physical_routes(graph)
        if not routes:
            continue
        u_w64 = w64_route_utilities(graph, routes, logits, UNMATCHED_PENALTY)
        delta = np.zeros(len(routes), dtype=np.float64)
        table_a = energy_table_from_utilities(routes, u_w64, unmatched_penalty=UNMATCHED_PENALTY)
        table_new = calibrated_energy_table(routes, u_w64, delta, UNMATCHED_PENALTY)
        selected_a = selected_mask(table_a)
        selected_new = selected_mask(table_new)
        per_event_selected_equal = per_event_selected_equal and selected_a == selected_new
        selected_a_all.extend(selected_a)
        selected_new_all.extend(selected_new)
        block_a = account_event(graph, routes, table_a, adjacent_sets)
        block_new = account_event(graph, routes, table_new, adjacent_sets)
        if accounting_a is None:
            accounting_a = block_a
            accounting_new = block_new
        else:
            accounting_a.add(block_a)
            accounting_new.add(block_new)
    if accounting_a is None:
        raise RuntimeError("identity replay saw no evaluable events")
    report = compare_identity(selected_a_all, selected_new_all, accounting_a.as_dict(), accounting_new.as_dict())
    report["passed"] = bool(report["passed"] and per_event_selected_equal)
    report["selected_identical"] = bool(report["selected_identical"] and per_event_selected_equal)
    report["metrics_A"] = accounting_a.as_dict()
    report["metrics_new"] = accounting_new.as_dict()
    return report


def identity_with_model(graphs, adjacent_sets, logits, model, standardizer, device, account_event) -> dict[str, object]:
    """θ=0 linear map must reproduce Arm A on the official scoring path."""
    from training.explicit_route_energy import enumerate_contiguous_physical_routes

    selected_a_all = []
    selected_new_all = []
    accounting_a = None
    accounting_new = None
    per_event_selected_equal = True
    max_abs_delta = 0.0
    for graph in graphs:
        routes = enumerate_contiguous_physical_routes(graph)
        if not routes:
            continue
        u_w64, delta = score_routes(model, standardizer, graph, routes, logits, device)
        max_abs_delta = max(max_abs_delta, float(np.max(np.abs(delta))) if delta.size else 0.0)
        table_a = energy_table_from_utilities(routes, u_w64, unmatched_penalty=UNMATCHED_PENALTY)
        table_new = calibrated_energy_table(routes, u_w64, delta, UNMATCHED_PENALTY)
        selected_a = selected_mask(table_a)
        selected_new = selected_mask(table_new)
        per_event_selected_equal = per_event_selected_equal and selected_a == selected_new
        selected_a_all.extend(selected_a)
        selected_new_all.extend(selected_new)
        block_a = account_event(graph, routes, table_a, adjacent_sets)
        block_new = account_event(graph, routes, table_new, adjacent_sets)
        if accounting_a is None:
            accounting_a = block_a
            accounting_new = block_new
        else:
            accounting_a.add(block_a)
            accounting_new.add(block_new)
    if accounting_a is None:
        raise RuntimeError("model identity replay saw no evaluable events")
    report = compare_identity(selected_a_all, selected_new_all, accounting_a.as_dict(), accounting_new.as_dict())
    report["passed"] = bool(report["passed"] and per_event_selected_equal and max_abs_delta <= 1.0e-7)
    report["selected_identical"] = bool(report["selected_identical"] and per_event_selected_equal)
    report["max_abs_delta"] = max_abs_delta
    report["metrics_A"] = accounting_a.as_dict()
    report["metrics_new"] = accounting_new.as_dict()
    return report


def fold_decision(
    *,
    identity_passed: bool,
    admitted: bool,
    holdout_a: Mapping[str, object],
    holdout_d: Mapping[str, object],
) -> dict[str, object]:
    if not admitted:
        return {
            "decision": "reject_calibration_under_zero_fake_slack",
            "calibration_acceptable": False,
            "calibration_has_no_accepted_increment": False,
            "calibration_unacceptable_under_zero_fake_slack": True,
            "physics_constrained_calibration_supported": False,
            "default_system": "frozen_W64_raw_energy_plus_exact_solver",
            "restore": "DeltaU = 0",
            "identity_passed": identity_passed,
            "accept_eff_gain": ACCEPT_EFF_GAIN,
            "note": "two-fold conjunction is written only by summarize_wb81b_calibration.py",
        }
    acceptable = bool(
        holdout_d.get("complete_fake_rate") is not None
        and holdout_a.get("complete_fake_rate") is not None
        and holdout_d.get("complete_track_efficiency") is not None
        and holdout_a.get("complete_track_efficiency") is not None
        and float(holdout_d["complete_fake_rate"]) <= float(holdout_a["complete_fake_rate"])
        and float(holdout_d["complete_track_efficiency"]) >= float(holdout_a["complete_track_efficiency"]) + ACCEPT_EFF_GAIN
    )
    return {
        "decision": "accept_calibration" if acceptable else "no_accepted_increment",
        "calibration_acceptable": acceptable,
        "calibration_has_no_accepted_increment": bool(identity_passed and not acceptable),
        "calibration_unacceptable_under_zero_fake_slack": False,
        "physics_constrained_calibration_supported": False,
        "default_system": (
            "physics_constrained_calibration_d_v1" if acceptable else "frozen_W64_raw_energy_plus_exact_solver"
        ),
        "restore": None,
        "identity_passed": identity_passed,
        "accept_eff_gain": ACCEPT_EFF_GAIN,
        "note": "two-fold conjunction is written only by summarize_wb81b_calibration.py",
    }


def audit_holdout(
    graphs,
    logits,
    model,
    standardizer,
    device: torch.device,
    wb81a_truths: Sequence[Mapping[str, object]],
    wb81a_fakes: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    from training.explicit_route_energy import enumerate_contiguous_physical_routes

    truths_by_event: dict[int, list[dict[str, object]]] = {}
    for row in wb81a_truths:
        truths_by_event.setdefault(int(row["event_index"]), []).append(dict(row))
    fakes_by_event: dict[int, list[dict[str, object]]] = {}
    for row in wb81a_fakes:
        fakes_by_event.setdefault(int(row["event_index"]), []).append(dict(row))

    recovered_from_t_neg = 0
    lost_from_t_near = 0
    a_selected_truth_lost = 0
    a_fake_retained = 0
    a_fake_removed = 0
    new_complete_fake = 0
    t_neg = t_near = t_comf = 0
    t_neg_selected_a = t_neg_selected_d = 0
    t_near_selected_a = t_near_selected_d = 0
    deltas_truth = []
    deltas_fake = []
    gaps_a = []
    gaps_d = []
    saturated = 0
    n_delta = 0
    length = {arm: {2: 0, 3: 0, 4: 0} for arm in ("A", "D")}
    for event_index, graph in enumerate(graphs):
        routes = enumerate_contiguous_physical_routes(graph)
        if not routes:
            continue
        u_w64, delta = score_routes(model, standardizer, graph, routes, logits, device)
        table_a = energy_table_from_utilities(routes, u_w64, unmatched_penalty=UNMATCHED_PENALTY)
        table_d = calibrated_energy_table(routes, u_w64, delta, UNMATCHED_PENALTY)
        selected_a = set(selected_mask(table_a))
        selected_d = set(selected_mask(table_d))
        for route_id, route in enumerate(routes):
            n_stations = int(len(route.stations))
            if n_stations in (2, 3, 4):
                if route_id in selected_a:
                    length["A"][n_stations] += 1
                if route_id in selected_d:
                    length["D"][n_stations] += 1
            n_delta += 1
            if abs(float(delta[route_id])) >= DELTA_MAX - 1.0e-3:
                saturated += 1
            if route.truth_consistent:
                deltas_truth.append(float(delta[route_id]))
            elif route_id in selected_d or route_id in selected_a:
                deltas_fake.append(float(delta[route_id]))
            if route_id in selected_d and n_stations == 4 and not route.truth_consistent and route_id not in selected_a:
                new_complete_fake += 1
        for row in truths_by_event.get(event_index, ()):
            route_id = int(row["route_id"])
            bin_name = str(row["bin"])
            if bin_name == "negative":
                t_neg += 1
                t_neg_selected_a += int(route_id in selected_a)
                t_neg_selected_d += int(route_id in selected_d)
                if route_id in selected_d and route_id not in selected_a:
                    recovered_from_t_neg += 1
            elif bin_name == "near_zero":
                t_near += 1
                t_near_selected_a += int(route_id in selected_a)
                t_near_selected_d += int(route_id in selected_d)
                if route_id in selected_a and route_id not in selected_d:
                    lost_from_t_near += 1
            else:
                t_comf += 1
            if bool(row.get("selected_A")) and route_id not in selected_d:
                a_selected_truth_lost += 1
            if route_id < len(routes):
                gaps_a.append(float(row["inclusion_gap_A"]))
                gaps_d.append(float(inclusion_gap(table_d, route_id).inclusion_gap))
        for row in fakes_by_event.get(event_index, ()):
            route_id = int(row["route_id"])
            if route_id in selected_d:
                a_fake_retained += 1
            else:
                a_fake_removed += 1

    def _mean(values: list[float]) -> float | None:
        return None if not values else float(np.mean(values))

    def _ratio(num: int, den: int) -> float | None:
        return None if den <= 0 else float(num) / float(den)

    return {
        "recovered_from_T_neg": recovered_from_t_neg,
        "lost_from_T_near": lost_from_t_near,
        "new_complete_fake": new_complete_fake,
        "a_selected_truth_lost": a_selected_truth_lost,
        "a_selected_fake_retained": a_fake_retained,
        "a_selected_fake_removed": a_fake_removed,
        "t_neg": t_neg,
        "t_near": t_near,
        "t_comf": t_comf,
        "t_neg_recall_A": _ratio(t_neg_selected_a, t_neg),
        "t_neg_recall_D": _ratio(t_neg_selected_d, t_neg),
        "t_near_preservation_A": _ratio(t_near_selected_a, t_near),
        "t_near_preservation_D": _ratio(t_near_selected_d, t_near),
        "selected_route_length": length,
        "delta_u_truth_mean": _mean(deltas_truth),
        "delta_u_fake_mean": _mean(deltas_fake),
        "delta_u_truth_abs_mean": _mean([abs(value) for value in deltas_truth]),
        "delta_u_fake_abs_mean": _mean([abs(value) for value in deltas_fake]),
        "fraction_abs_delta_near_0_25": _ratio(saturated, n_delta),
        "inclusion_gap_A_mean": _mean(gaps_a),
        "inclusion_gap_D_mean": _mean(gaps_d),
        "delta_max": DELTA_MAX,
    }
