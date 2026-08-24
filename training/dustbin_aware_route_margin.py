"""Diagnostic dustbin-aware route-margin helpers.  Not a training objective.

Workbook 58 only measures the lift a frozen checkpoint would need for the
production decision boundary ``U > dustbin = 0``.  Nothing here is wired
into an optimizer, and it does not change inference.
"""

from __future__ import annotations

import inspect
from collections import Counter
from typing import Any, Mapping, Sequence

from training.gauge_consistent_route import packing_route_competition_loss
from training.route_operating_audit import DUSTBIN_UTILITY
from training.route_utility_identifiability import _median


PACKING_MARGIN = 1.0
FAILURE_FAMILIES = (
    "below_threshold_or_missing_candidate",
    "truth_gt_fragment_lt_dustbin",
    "truth_lt_fragment",
    "production_margin_satisfied",
    "training_margin_satisfied_inference_fails",
)


def workbook56_miner_strongest(fragment_utilities: Sequence[float] | None) -> float | None:
    """Replicate workbook-56 ``max(feasible rival)``, dustbin only if empty."""
    if not fragment_utilities:
        return float(DUSTBIN_UTILITY)
    return float(max(float(value) for value in fragment_utilities))


def dustbin_aware_strongest(fragment_utilities: Sequence[float] | None) -> float:
    """Production competitor: ``max(feasible rival, dustbin=0)``."""
    if not fragment_utilities:
        return float(DUSTBIN_UTILITY)
    return float(max(max(float(value) for value in fragment_utilities), float(DUSTBIN_UTILITY)))


def required_margin_to_dustbin(u_truth: float, margin: float = PACKING_MARGIN) -> float:
    """``-U_truth + margin``: lift needed so ``U_truth >= dustbin + margin``."""
    return float(-float(u_truth) + float(margin))


def required_margin_to_fragment(
    u_truth: float,
    u_best_fragment: float,
    margin: float = PACKING_MARGIN,
) -> float:
    """``U_best_fragment - U_truth + margin``."""
    return float(float(u_best_fragment) - float(u_truth) + float(margin))


def relu_gap(strongest: float, u_truth: float, margin: float = PACKING_MARGIN) -> float:
    return float(max(0.0, float(strongest) + float(margin) - float(u_truth)))


def classify_feasibility_family(
    *,
    score_retained: bool,
    candidate_retained: bool,
    u_truth: float | None,
    u_best_fragment: float | None,
    margin: float = PACKING_MARGIN,
    u_dustbin: float = DUSTBIN_UTILITY,
) -> str:
    if not candidate_retained or not score_retained or u_truth is None:
        return "below_threshold_or_missing_candidate"
    fragments = None if u_best_fragment is None else [float(u_best_fragment)]
    miner_gap = relu_gap(workbook56_miner_strongest(fragments), float(u_truth), margin)
    production = dustbin_aware_strongest(fragments)
    if float(u_truth) > production:
        return "production_margin_satisfied"
    if miner_gap <= 0.0 and float(u_truth) <= float(u_dustbin):
        return "training_margin_satisfied_inference_fails"
    if u_best_fragment is not None and float(u_truth) <= float(u_best_fragment):
        return "truth_lt_fragment"
    return "truth_gt_fragment_lt_dustbin"


def attach_dustbin_aware_margin(
    row: Mapping[str, Any],
    *,
    margin: float = PACKING_MARGIN,
) -> dict[str, object]:
    """Add diagnostic required-margin fields to an identifiability row."""
    u_truth = None if row.get("u_truth") is None else float(row["u_truth"])
    u_fragment = None if row.get("u_best_competitor") is None else float(row["u_best_competitor"])
    u_dustbin = float(row.get("u_dustbin", DUSTBIN_UTILITY))
    fragments = None if u_fragment is None else [u_fragment]
    miner_strongest = None if u_truth is None else workbook56_miner_strongest(fragments)
    production_strongest = None if u_truth is None else dustbin_aware_strongest(fragments)
    family = classify_feasibility_family(
        score_retained=bool(row.get("score_retained")),
        candidate_retained=bool(row.get("candidate_retained")),
        u_truth=u_truth,
        u_best_fragment=u_fragment,
        margin=margin,
        u_dustbin=u_dustbin,
    )
    miner_gap = None if u_truth is None or miner_strongest is None else relu_gap(miner_strongest, u_truth, margin)
    dustbin_gap = (
        None if u_truth is None or production_strongest is None else relu_gap(production_strongest, u_truth, margin)
    )
    return {
        **dict(row),
        "packing_margin": float(margin),
        "u_best_fragment": u_fragment,
        "u_dustbin": u_dustbin,
        "required_margin_to_dustbin": None if u_truth is None else required_margin_to_dustbin(u_truth, margin),
        "required_margin_to_fragment": (
            None if u_truth is None or u_fragment is None else required_margin_to_fragment(u_truth, u_fragment, margin)
        ),
        "workbook56_miner_strongest": miner_strongest,
        "dustbin_aware_strongest": production_strongest,
        "workbook56_miner_gap": miner_gap,
        "dustbin_aware_gap": dustbin_gap,
        "miner_understates_production_gap": (
            None if miner_gap is None or dustbin_gap is None else bool(miner_gap + 1.0e-12 < dustbin_gap)
        ),
        "miner_satisfied_production_fails": bool(
            miner_gap is not None and miner_gap <= 0.0 and u_truth is not None and float(u_truth) <= u_dustbin
        ),
        "feasibility_family": family,
    }


def summarize_feasibility(rows: Sequence[Mapping[str, Any]]) -> dict[str, object]:
    families = Counter(str(row.get("feasibility_family")) for row in rows)
    to_dustbin = [float(row["required_margin_to_dustbin"]) for row in rows if row.get("required_margin_to_dustbin") is not None]
    to_fragment = [
        float(row["required_margin_to_fragment"]) for row in rows if row.get("required_margin_to_fragment") is not None
    ]
    miner_gaps = [float(row["workbook56_miner_gap"]) for row in rows if row.get("workbook56_miner_gap") is not None]
    aware_gaps = [float(row["dustbin_aware_gap"]) for row in rows if row.get("dustbin_aware_gap") is not None]
    utilities = [float(row["u_truth"]) for row in rows if row.get("u_truth") is not None]
    fragments = [float(row["u_best_fragment"]) for row in rows if row.get("u_best_fragment") is not None]
    logits = _collect_edge_logits(rows)
    understates = sum(1 for row in rows if row.get("miner_understates_production_gap"))
    miner_ok_prod_fail = sum(1 for row in rows if row.get("miner_satisfied_production_fails"))
    return {
        "complete_truth_chains": int(len(rows)),
        "feasibility_family_counts": {name: int(families.get(name, 0)) for name in FAILURE_FAMILIES},
        "u_truth_median": None if not utilities else float(_median(utilities)),
        "u_truth_fraction_nonpositive": None if not utilities else float(sum(1 for value in utilities if value <= 0.0) / len(utilities)),
        "u_best_fragment_median": None if not fragments else float(_median(fragments)),
        "required_margin_to_dustbin_median": None if not to_dustbin else float(_median(to_dustbin)),
        "required_margin_to_fragment_median": None if not to_fragment else float(_median(to_fragment)),
        "workbook56_miner_gap_median": None if not miner_gaps else float(_median(miner_gaps)),
        "dustbin_aware_gap_median": None if not aware_gaps else float(_median(aware_gaps)),
        "miner_understates_production_gap": int(understates),
        "miner_satisfied_production_fails": int(miner_ok_prod_fail),
        "edge_logit_median": None if not logits else float(_median(logits)),
        "quantiles": {
            "u_truth": _quantiles(utilities),
            "u_best_fragment": _quantiles(fragments),
            "required_margin_to_dustbin": _quantiles(to_dustbin),
            "required_margin_to_fragment": _quantiles(to_fragment),
            "workbook56_miner_gap": _quantiles(miner_gaps),
            "dustbin_aware_gap": _quantiles(aware_gaps),
            "edge_logit": _quantiles(logits),
        },
    }


def compare_scale_distributions(
    candidate_rows: Sequence[Mapping[str, Any]],
    control_rows: Sequence[Mapping[str, Any]],
) -> dict[str, object]:
    candidate_logits = _collect_edge_logits(candidate_rows)
    control_logits = _collect_edge_logits(control_rows)
    cand_u = [float(row["u_truth"]) for row in candidate_rows if row.get("u_truth") is not None]
    ctrl_u = [float(row["u_truth"]) for row in control_rows if row.get("u_truth") is not None]
    cand_dust = [
        float(row["required_margin_to_dustbin"])
        for row in candidate_rows
        if row.get("required_margin_to_dustbin") is not None
    ]
    ctrl_dust = [
        float(row["required_margin_to_dustbin"])
        for row in control_rows
        if row.get("required_margin_to_dustbin") is not None
    ]
    return {
        "candidate_edge_logit_median": None if not candidate_logits else float(_median(candidate_logits)),
        "control_edge_logit_median": None if not control_logits else float(_median(control_logits)),
        "edge_logit_candidate_minus_control_median": (
            None
            if not candidate_logits or not control_logits
            else float(_median(candidate_logits) - _median(control_logits))
        ),
        "candidate_u_truth_median": None if not cand_u else float(_median(cand_u)),
        "control_u_truth_median": None if not ctrl_u else float(_median(ctrl_u)),
        "u_truth_candidate_minus_control_median": (
            None if not cand_u or not ctrl_u else float(_median(cand_u) - _median(ctrl_u))
        ),
        "candidate_required_margin_to_dustbin_median": None if not cand_dust else float(_median(cand_dust)),
        "control_required_margin_to_dustbin_median": None if not ctrl_dust else float(_median(ctrl_dust)),
        "absolute_scale_compressed_vs_control": bool(
            cand_u and ctrl_u and float(_median(cand_u)) < float(_median(ctrl_u))
        ),
    }


def audit_workbook56_packing_loss_source() -> dict[str, object]:
    """Static confirmation that the miner omits dustbin when a rival exists."""
    source = inspect.getsource(packing_route_competition_loss)
    uses_stack_max = "torch.stack(competitor_utilities).max()" in source
    substitutes_dustbin_if_nonfinite = "torch.where(torch.isfinite(strongest), strongest, dustbin)" in source
    default_omits_dustbin = "include_dustbin: bool = False" in source
    dustbin_aware_branch = "if include_dustbin:" in source and "torch.maximum(strongest, dustbin)" in source
    return {
        "function": "training.gauge_consistent_route.packing_route_competition_loss",
        "uses_max_of_feasible_rival_utilities": uses_stack_max,
        "dustbin_substituted_only_when_no_finite_rival": substitutes_dustbin_if_nonfinite,
        "default_include_dustbin": False,
        "explicitly_takes_max_rival_and_dustbin": not default_omits_dustbin,
        "dustbin_aware_branch_uses_max_with_dustbin": dustbin_aware_branch,
        "omits_dustbin_when_negative_fragment_exists": uses_stack_max and default_omits_dustbin,
        "registered_margin": float(PACKING_MARGIN),
        "production_dustbin_utility": float(DUSTBIN_UTILITY),
        "note": (
            "Workbook-56 packing competition uses max(feasible rival utilities) "
            "and substitutes dustbin only if that set is empty or fully masked.  "
            "Production always compares against dustbin 0 because U<=0 is not admitted."
        ),
    }


def recommend_next_objective(
    pooled: Mapping[str, Any],
    identifiability: Mapping[str, Any] | None = None,
) -> dict[str, object]:
    counts = pooled.get("feasibility_family_counts") or {}
    dustbin_scale = int(counts.get("truth_gt_fragment_lt_dustbin") or 0)
    fragment = int(counts.get("truth_lt_fragment") or 0)
    solver_mismatch = int(counts.get("training_margin_satisfied_inference_fails") or 0)
    satisfied = int(counts.get("production_margin_satisfied") or 0)
    total = int(pooled.get("complete_truth_chains") or 0)
    u_frac = pooled.get("u_truth_fraction_nonpositive")
    admitted_fragment = 0
    if identifiability is not None:
        admitted_fragment = int(
            (identifiability.get("decision_class_counts") or {}).get("truth_loses_to_fragment") or 0
        )
    if solver_mismatch > dustbin_scale and solver_mismatch > fragment and solver_mismatch > 0:
        next_step = "audit_solver_implementation_consistency"
        designed = None
    elif admitted_fragment > 0 and admitted_fragment >= dustbin_scale and admitted_fragment >= int(total * 0.5):
        next_step = "enhance_solver_generated_hard_negative_mining_only"
        designed = "solver_generated_hard_negatives"
    elif u_frac is not None and float(u_frac) >= 0.90 and admitted_fragment == 0:
        next_step = "design_dustbin_aware_route_margin_objective"
        designed = "dustbin_aware_route_margin"
    elif fragment > dustbin_scale:
        next_step = "enhance_solver_generated_hard_negative_mining_only"
        designed = "solver_generated_hard_negatives"
    else:
        next_step = "design_dustbin_aware_route_margin_objective"
        designed = "dustbin_aware_route_margin"
    return {
        "continue_to_15d_relative_wls": False,
        "open_15d_wls_authorized_by_this_audit": False,
        "new_checkpoint_authorized": False,
        "association_gate_satisfied": satisfied == total and total > 0,
        "truth_route_utility_margin_satisfied": bool(u_frac == 0.0 and satisfied == total and total > 0),
        "dominant_feasibility_family": max(counts, key=counts.get) if counts else None,
        "production_admitted_fragment_wins": int(admitted_fragment),
        "u_truth_fraction_nonpositive": u_frac,
        "next_step": next_step,
        "next_objective_if_a_later_control_is_opened": designed,
        "stay_on_route_competition_objective_diagnosis": True,
    }


def _collect_edge_logits(rows: Sequence[Mapping[str, Any]]) -> list[float]:
    values: list[float] = []
    for row in rows:
        edges = row.get("edges")
        if not isinstance(edges, list):
            continue
        for edge in edges:
            if isinstance(edge, Mapping) and "calibrated_logit" in edge:
                values.append(float(edge["calibrated_logit"]))
    return values


def _quantiles(values: Sequence[float]) -> dict[str, float] | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)

    def _at(fraction: float) -> float:
        index = int(round((len(ordered) - 1) * float(fraction)))
        return float(ordered[index])

    return {"p05": _at(0.05), "p25": _at(0.25), "p50": _at(0.50), "p75": _at(0.75), "p95": _at(0.95)}
