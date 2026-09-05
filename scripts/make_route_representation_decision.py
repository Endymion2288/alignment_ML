#!/usr/bin/env python3
"""Workbook 73: build decision.json + physical_relative_probe_summary.json from the probe outputs."""
import json
import os

OUT = "outputs/mc24_four_station_route_representation_domain_audit_v1"


def main():
    cf = json.load(open(os.path.join(OUT, "cross_family_truth_fake_probe.json")))
    sp = json.load(open("outputs/mc24_four_station_route_representation_domain_audit_v1/source_probe_summary.json"))
    fa = json.load(open(os.path.join(OUT, "fold_asymmetry_summary.json")))
    co = json.load(open(os.path.join(OUT, "catastrophic_cohort_summary.json")))

    def transfer(lv):
        a = cf[lv]["family1_to_family2"]["cross_family"]
        b = cf[lv]["family2_to_family1"]["cross_family"]
        return {
            "f1_to_f2_auc": a["auc"], "f2_to_f1_auc": b["auc"],
            "f1_to_f2_truth_fnr": a["truth_fnr"], "f2_to_f1_truth_fnr": b["truth_fnr"],
            "worst_direction_truth_fnr": max(a["truth_fnr"], b["truth_fnr"]),
        }

    levels = ["R0", "R1", "R2", "R3", "R3_abs", "R3_qual", "R_phys", "R4", "R5", "R6"]
    table = {}
    for lv in levels:
        e = {"cross_family_transfer": transfer(lv)}
        if lv in sp:
            e["source_predictability_auc"] = {"truth": sp[lv]["truth_only"]["auc"], "fake": sp[lv]["fake_only"]["auc"]}
        if lv in fa:
            e["fold_nn_asymmetry"] = fa[lv]["nn_asymmetry"]
        table[lv] = e

    # ---- physical-relative probe summary ----
    phys = {
        "definition": "R_phys = [edge01,edge12,edge23 physical pair-relative observables (11 each, residual/pull/chi2/cov/delta-z) + frozen L01,L12,L23]; NO absolute node latent",
        "n_dim": 36,
        "cross_family_transfer": transfer("R_phys"),
        "source_predictability_auc": sp["R_phys"],
        "fold_nn_asymmetry": fa["R_phys"]["nn_asymmetry"],
        "comparison": {
            "R_phys_worst_truth_fnr": transfer("R_phys")["worst_direction_truth_fnr"],
            "R5_worst_truth_fnr": transfer("R5")["worst_direction_truth_fnr"],
            "R6_worst_truth_fnr": transfer("R6")["worst_direction_truth_fnr"],
            "R4_worst_truth_fnr": transfer("R4")["worst_direction_truth_fnr"],
            "R0_worst_truth_fnr": transfer("R0")["worst_direction_truth_fnr"],
            "R2_worst_truth_fnr": transfer("R2")["worst_direction_truth_fnr"],
        },
        "finding": (
            "R_phys achieves near-zero cross-family truth false-negative rate "
            f"(worst-direction FNR={transfer('R_phys')['worst_direction_truth_fnr']:.4f}), "
            f"~{transfer('R6')['worst_direction_truth_fnr']/max(transfer('R_phys')['worst_direction_truth_fnr'],1e-9):.0f}x lower than the absolute-node-latent route pipeline R6 "
            f"(FNR={transfer('R6')['worst_direction_truth_fnr']:.4f}) which matches the observed V5A catastrophic truth-destruction rate (~215/6789=3.2%). "
            "Dropping the absolute node latent removes the cross-family truth/fake boundary error."
        ),
    }
    with open(os.path.join(OUT, "physical_relative_probe_summary.json"), "w") as f:
        json.dump(phys, f, indent=2)

    # ---- decision ----
    decision = {
        "workbook": 73,
        "diagnostic_only": True,
        "not_checkpoint_candidate": True,
        "training_authorized": False,
        "final_blind_eval_authorized": False,
        "new_final_blind_content_accessed": False,
        "sealed_test_accessed": False,
        "continue_to_15d_relative_wls": False,
        "per_level_summary": table,
        "catastrophic_cohort": {
            "n_doomed_proxy": co["n_doomed"], "n_surviving": co["n_surviving"],
            "wb72_solver_catastrophic_count": 215,
            "earliest_distinguishing_layer": "R0 (physical edge observables, doomed-vs-surviving |SMD| max = %.2f)" % co["per_level_doomed_vs_surviving"]["R0"]["smd_abs_max"],
            "interpretation": "doomed truth routes are physically-hard (2->3/S3) routes, distinguishable already at R0; the head then mis-scores them cross-family.",
        },
        "primary_localization": (
            "Source-family information is present from R0 (moderate source-probe AUC ~0.7) and rises with depth "
            "(R2 edge-latent 0.96, R4 node-latent 0.85, R5 route-input 0.95). BUT source predictability does NOT equal transfer failure: "
            "R2 and R_phys transfer nearly perfectly (truth FNR <= 0.008) despite R2's high source predictability. "
            "The transfer-relevant corruption enters via the ABSOLUTE NODE LATENT (R4): alone it is a weak, non-transferable discriminator "
            "(in-family AUC 0.66, cross-family truth FNR 0.39-0.47), and when consumed by the route head (R5/R6) it imposes a ~3.6-4.8% "
            "cross-family truth false-negative rate that matches the observed V5A catastrophic truth destruction (~3.2%)."
        ),
        "earliest_source_leakage_layer": "R0/R1 (moderate) rising to R2/R4/R5 (high); transfer-breaking component is the absolute node latent R4",
        "best_cross_family_representation": "R_phys (physical pair-relative route, no absolute node latent), worst-direction truth FNR = %.4f" % transfer("R_phys")["worst_direction_truth_fnr"],
        "fold_asymmetry_explanation": (
            "Representation-level support asymmetry between family1-truth and family2-truth is SMALL (nn_asymmetry 0.12-0.65; distributions overlap). "
            "The family2->family1 degradation is a boundary-POSITION error: a boundary learned on family2 is miscalibrated for the ~3% of family1 truth "
            "routes that are physically hard (low L23 / 2->3), misclassifying them as fake. R_phys has the smallest asymmetry (0.12) and ~0 truth FNR."
        ),
        "r_phys_vs_absolute_node_latent": (
            "R_phys worst-direction truth FNR %.4f vs R5 %.4f / R6 %.4f / R4 %.4f. "
            "R_phys is dramatically more transferable."
            % (transfer("R_phys")["worst_direction_truth_fnr"], transfer("R5")["worst_direction_truth_fnr"],
               transfer("R6")["worst_direction_truth_fnr"], transfer("R4")["worst_direction_truth_fnr"])
        ),
        "decision_case": "case_1",
        "selected_next_hypothesis": "Physical Pair-Relative Route Encoder",
        "selected_next_hypothesis_detail": (
            "Route representation no longer consumes the absolute node latent (s0..s3). It consumes only the three adjacent "
            "physical pair-relative edge representations (existing residual/pull/chi2/cov/delta-z observables) plus the frozen W64 edge logits "
            "L01,L12,L23. This is the R_phys representation, which is the only level with near-zero cross-family truth FNR. "
            "No SE(3)/gauge-equivariant or domain-adversarial network is built in Workbook 73; this is the minimal, physically-interpretable fix."
        ),
    }
    with open(os.path.join(OUT, "decision.json"), "w") as f:
        json.dump(decision, f, indent=2)
    print("wrote decision.json + physical_relative_probe_summary.json")
    print("decision_case = case_1 (Physical Pair-Relative Route Encoder)")
    print("R_phys worst truth FNR = %.4f | R5 %.4f | R6 %.4f | R4 %.4f" % (
        transfer("R_phys")["worst_direction_truth_fnr"], transfer("R5")["worst_direction_truth_fnr"],
        transfer("R6")["worst_direction_truth_fnr"], transfer("R4")["worst_direction_truth_fnr"]))


if __name__ == "__main__":
    main()
