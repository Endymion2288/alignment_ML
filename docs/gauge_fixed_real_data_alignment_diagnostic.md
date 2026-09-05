# Gauge-Fixed Real-Data Alignment Diagnostic V1 (Workbook 78)

Status: **pre-registered** (design frozen before any real-data candidate is computed; results backfilled after execution and frozen in a separate commit).

## Scientific question

Without claiming to recover a unique mechanical station pose, can a pre-frozen reconstruction gauge representative be stably determined from a real-data calibration subset and improve gauge-invariant tracker observables on completely held-out real data?

This is a diagnostic, not a geometry correction deployment. For the whole workbook: `geometry_write_allowed = false`, `official_conditions_write_allowed = false`, `real_data_candidate_alignment_authorized = false`, `external_constraint_ingest_authorized = false`.

## Frozen inheritance

- Workbook 77 freeze commit `9996765`; all seven WB77 artifact SHA256 verified; WB77 config SHA256 `d44876589d6bfed742714b957de80119024d7c93f747e45a1ed9357a5c0527e6`. WB77 closed the gauge/external-constraint feasibility campaign with no eligible external physical constraint.
- Tracker information: frozen WB68 7D parameter order, severity scales S = (5, 5, 5, 60, 60, 60, 0.12), `rank_tolerance = 0.01`, frozen identifiable/null basis; SHA + amended regression (basis-independent projector Frobenius distance ≤ 1e-8; the arccos-quantized principal-angle gate is NOT reinstated).
- Real-data population: frozen Operating Protocol V1 / Frozen-V2 selected-route contract (12 runs, 2024 r0022). Residual-blind role map inherited from the WB48-52 blind corpus: calibration = 14973+14974; held-out = 14975+14976 (primary), 7 expansion monitoring runs, 14977 report-only. No re-splitting, no event selection, no restacking.

## Key structural findings (MC-only design pilots, before pre-registration)

1. Frozen selected routes are dominated by 3-station (0,1,2) routes: anchor-pair composition across all 12 runs is (0,1):1074 / (0,2):763 / (0,3):18.
2. Cropping the frozen MC bank to the real composition, the information matrix in the frozen 5D identifiable space has rank 2 at the frozen tolerance even with exact per-pair Jacobians: the missing three dimensions need (0,3) leverage that essentially does not exist in the real population.
3. The two informed directions are track-driven {rx+dy, ry+dx}-like combinations.
4. The only legal observable-model transfer (station-pair mean native Jacobian from the frozen MC bank; real-data FD probes remain forbidden) recovers informed-subspace injections with 8-27% deviation in MC half-split pilots; pre-registered MC-control gates are set from these pilots with margin (0.35) before held-out is opened.
5. Real (0,1) covariances are ~300x (x rows) / ~800x (tx rows) larger than MC - a frozen property of the collision-data population, mirrored in the MC-control `real_scale_covariance` scenario.
6. The real-data bank builder (frozen identity files + frozen `evaluate_field_propagation` chain) reproduces all frozen DQ adjacent-edge CSV rows bit-exactly; this is a hard regression test.

## Pre-registered design

- **Gauges**: primary `minimum_norm_scaled_gauge` (inherits the WB68/77 severity-scaled minimum-norm representative and the `solver_restricted_to_identifiable_subspace` doctrine - chosen by contract, not by condition number); secondary control `named_parameter_zero_gauge`; report-only `minimum_norm_native_gauge`. No gauge switching after seeing real data. Outputs are named *reconstruction gauge representative* / *gauge-fixed candidate diagnostic*, never mechanical measurements.
- **Transfer**: station-pair mean native Jacobian from the frozen MC pooled bank; model error quantified by the MC control before any real-data solve.
- **Solve** (one-shot, no iteration): calibration residuals -> transfer model -> projection into the frozen identifiable basis -> real-information rank K at the frozen tolerance -> informed-subspace WLS -> gauge representatives via the null-space formulation of the WB77 KKT contract -> candidate artifact frozen with SHA -> only then held-out evaluation. Uninformed identifiable modes receive a zero update explicitly labelled `not_informed_by_real_data` (never "measured zero").
- **Applicability/linearity audit** before the solve (report-first, no event selection); linear envelope gate `max |gamma_k| <= 0.15` inherited from the WB33-35/68 injection design; exceeding it freezes `real_data_linearized_model_out_of_support`.
- **MC control** (before any real-data solve): two covariance scenarios, injection recovery/leakage gates (0.35), 200-replicate null ensemble producing the spurious-improvement floor for held-out gate A, gauge-invariance closure (1e-8).
- **Held-out gates**: A improvement beyond the null floor; B run-level consistency (both primary runs improve; leave-one-run-out stays negative); C per-cell worsening within the 2*sqrt(2n) chi-square fluctuation bound plus the exactly-zero non-IFT null control; D gauge-invariant held-out predictions (1e-8); E bootstrap direction stability (15 deg, WB68 inheritance); F no new systematic shift in frozen DQ slices (entry-52 alarm inheritance); G amplitude within the linear envelope.
- **Decision tree** (exactly one terminal string): `gauge_fixed_real_data_candidate_diagnostic_pass` or one of the nine pre-registered failure strings (`tracker_information_regression_failed`, `real_data_population_provenance_mismatch`, `real_data_observable_model_transfer_failed`, `real_data_diagnostic_inconclusive`, `real_data_identifiable_information_rank_zero`, `real_data_linearized_model_out_of_support`, `real_data_gauge_fixed_candidate_not_source_stable`, `real_data_gauge_invariance_failed`, `real_data_heldout_observable_not_improved`). On failure: freeze the type, continue `residual_dq_monitoring_only`, no gauge/split/selection/S/rank changes, no identifiability rescue.

## Engineering

- New: `configs/gauge_fixed_real_data_alignment_diagnostic_v1.yaml`, `alignment/gauge_fixed_real_data_diagnostic.py`, `scripts/report_gauge_fixed_real_data_alignment_diagnostic.py`, `tests/test_gauge_fixed_real_data_diagnostic.py`, bilingual docs, `outputs/gauge_fixed_real_data_alignment_diagnostic_v1/`.
- No new reconstruction: existing real-data outputs + linear algebra only, interactive.
- Driver stages structurally enforce the freeze ordering: `validate-config -> freeze-split -> mc-control -> calibrate -> evaluate -> decide`; `calibrate` requires a passing MC control; `evaluate` requires the frozen candidate artifact with matching SHA before reading any held-out residual.
- 20 regression tests cover: frozen primary gauge, deterministic split, held-out inaccessibility before candidate freeze, gauge-equivalent predictions, no external prior ingestion, no geometry/conditions writes, candidate-artifact reproducibility, hard failure on wrong split/provenance, bit-exact bank/CSV reproduction, MC-control machinery, and every decision-tree branch.

## Results

**Frozen decision: `real_data_linearized_model_out_of_support`.**

The staged campaign executed in freeze order. Config validation, tracker-information regression (amended projector-Frobenius contract), split freeze (SHA256 `c9c359793c41b59df1296e3ec2075af652c3f585746ca832a9c2e37f4a4f61d3`), and MC control all passed (injection recovery/leakage ≤ 0.35 in both covariance scenarios; exact gauge invariance; null improvement floor 49.72). The one-shot calibration solve produced a frozen candidate (artifact SHA256 `347bfbb5f41d5f19cf497938d48e75c6fa77cae458e8ba7c78b6129a1d81ddc7`) whose dominant informed mode requires |γ|max = 2.46 in scaled severity units — 16× the pre-registered linear envelope of 0.15. The bootstrap stability gate also failed (17/50 rank changes, max direction angle 158°). Per the pre-registered decision tree the held-out evaluation was structurally refused and never read held-out residuals. The candidate is frozen but is **not** a valid correction and must not be interpreted as a mechanical station displacement measurement.

The real calibration information has informed rank 3 inside the frozen identifiable 5D space; the frozen population contains only 2 (0,3) anchor pairs, so the campaign is a diagnostic of this population, not of an idealized one. No gauge switch, re-split, selection tuning, S/rank change, or source removal was performed. The campaign continues `residual_dq_monitoring_only`; `geometry_write_allowed` and `official_conditions_write_allowed` remain `false`.

Full execution record and artifact SHAs: workbook 78 §6.
