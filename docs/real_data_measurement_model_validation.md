# Real-Data Measurement-Model Reconstruction & Cross-Run Validation V1

Workbook 80. **Status: completed and frozen** — executed in freeze order within a
single interactive session; every validation gate was written into the config and
frozen before any confirmatory computation, and no gate was modified after
execution (only two code bugs that do not affect any gate or the science were
fixed).

**Final decision: `measurement_model_multiple_components_not_validated`.** Both
independent key gates fail: (A) the residual covariance model fails cross-fit
validation, and (B) the MC→real Jacobian-transfer model has insufficient
real-data support. Hence `covariance_model_validated=false` and
`jacobian_transfer_model_validated=false`; the cross-run information check is
**structurally skipped** per the pre-registered rule, and
`real_data_alignment_v2_preregistration_allowed=false`. Real data continues
`residual_dq_monitoring_only`; **no Workbook 81 is opened**.

This campaign only rebuilds and validates the measurement model (covariance +
Jacobian transfer). It does **not** solve for a final alignment correction, open
held-out data, write geometry, run nonlinear iteration, change the gauge,
redefine V_id/V_null, modify S/rank_tolerance, or use any external prior. For the
whole workbook:

- `geometry_write_allowed = false`
- `official_conditions_write_allowed = false`
- `real_data_candidate_alignment_authorized = false`
- `external_constraint_ingest_authorized = false`
- `held_out_accessed = false` (held-out remains completely closed, code-level guard)

## Scientific question (the only one)

Workbook 79 froze `real_data_statistical_model_multiple_failures`. The question
here: can a trustworthy measurement model — (A) a residual covariance model and
(B) an MC→real Jacobian-transfer model — be rebuilt from first principles and
validated, each passing an independent, residual-blind cross-run /
MC-source-disjoint validation? Only if both pass and the cross-run information is
stable may a new Workbook 81 (Gauge-Fixed Real-Data Alignment Diagnostic V2) be
pre-registered. This campaign itself produces **no** deployable candidate.

## Frozen inputs (no population re-selection)

- **Calibration (only real data used)**: runs 14973 + 14974, with the frozen
  route/event/source identity, Frozen-V2 selected-route contract and split SHA.
- **Held-out (remains closed)**: 14975/14976 + 7 monitoring runs
  (14971/14972/14980/14981/14985/14989/15007) + 14977 report-only. A code-level
  guard (`assert_no_held_out_access`) raises on any held-out run id.
- Tracker information / gauge / identifiable basis / S / rank_tolerance are
  inherited from WB78 (loaded via the WB79 chain with SHA verification); the
  frozen V_id/V_null are injected at `load_config` and never re-derived on real
  data.

## Stages and results

1. **Reproduce WB79 baseline (hard stop)** — PASS. The WB79 covariance
   eigensystem/whitening baseline reproduces exactly (6/6 checks): bank SHA
   `6e4eaae0...`, χ²_total = 1999613.163574398, whitened χ²/ndof =
   1058.4585330827601, smallest-eigenmode χ² fraction 0.99741, condition median
   2.347e12, n_pairs = 378.

2. **Covariance semantics audit** — the combined 4×4 covariance is
   `C_combined = C_propagated_source + C_target` (plain sum, verified bit-exactly,
   max abs diff 0.0), assuming source/target independence. `C_prop` is computed
   upstream by the external Calypso/ACTS extrapolation and is empirically
   near-singular (median condition ~4.4e14); whether that tool already includes
   multiple-scattering process noise is **not confirmable from this repo**
   (unresolved external provenance). Marginal per-pair pull RMS (x=1.68, y=0.13,
   tx=0.42, ty=0.12) is roughly calibrated — the problem is the off-diagonal
   correlation structure, not the marginals.

3. **Numerical-inversion audit** — all four methods (direct inv / Cholesky /
   eigendecomposition / SVD) agree on the total χ² to ~2.2e-7 with zero
   factorization failures. The huge χ² is **not** a numerical artifact; it is a
   genuine statistical-model mismatch. This stage is strictly a method-A
   comparison (stable evaluation of the same C⁻¹); it does not modify C.

4. **Covariance model cross-fit validation — FAIL.** Candidates were built from
   first principles only (never tuned on γ/rank/condition/χ²): `C0_frozen`,
   `C1_ms_leverarm` (multiple-scattering random-walk process noise
   `C = C_combined + θ²·G_MS(L)`), `C2_scaled_diagonal_floor`,
   `C3_variance_inflation`. Nuisance scales were estimated by Gaussian NLL with
   strict 14973↔14974 cross-fitting (no same-run self-proof). C1 and C2 bring the
   whitened χ²/ndof from ~1000 down to ~1 (a ~1000×, physically sensible
   improvement) with a healthy largest Cov(z) eigenvalue, **but the smallest
   Cov(z) eigenvalue stays ~0.001–0.009, far below the 0.25 gate**. Root cause:
   `C_prop` over-estimates the real residual variance along the pencil direction
   by ~100–2000× (the claimed strong x↔tx correlation is not present in the real
   residual); additive corrections cannot remove that over-correlation. No
   candidate reproduces the residual correlation structure, so
   `covariance_model_validated=false`.

5. **MC-only conditional Jacobian model — MC validation PASS.** Three candidates
   (`J0_station_pair_mean`, `J1_kinematic_binned`, `J2_linear_regression`) fit on
   3 construction MC sources and validated on 4 disjoint validation MC sources all
   pass the MC gates (per-pair Frobenius median ~0.5–1.6%, injection recovery
   ~5–6%, null leakage ~1%). Features are residual-blind (pred_tx, pred_ty only).

6. **Real-data transfer-support validation — FAIL.** Applied to the frozen
   14973/14974 kinematics (residual-blind), only **80.6%** of (0,1) and **87.2%**
   of (0,2) real pairs fall within the MC 99% Mahalanobis (tx,ty) envelope
   (gate ≥ 90%); the real ty distribution is ~2.7× wider than MC. Out-of-support
   pairs were never extrapolated into a solve.
   `jacobian_transfer_model_validated=false`.

7. **Cross-run information sanity check — structurally skipped** (both component
   models failed, so the check is not meaningful and was not run).

## Decision and physical interpretation

`measurement_model_multiple_components_not_validated`. Under the frozen inputs, a
trustworthy measurement model **cannot** be rebuilt and validated: the covariance
model cannot reproduce the real residual correlation structure (the propagated
covariance's pencil direction is over-correlated), and the Jacobian-transfer model
has insufficient real-data kinematic support. All freeze flags remain;
`residual_dq_monitoring_only` continues; no Workbook 81 is opened.

Physically-motivated future directions (each requiring a separate pre-registered
campaign): (1) confirm the external propagation tool's covariance transformation
and multiple-scattering process-noise handling directly from its provenance, and
fix any systematic pencil-direction variance over-estimate upstream rather than
patching it in this repo; (2) obtain MC/control samples covering the wide-ty real
topology so the conditional-J model's applicability domain extends to the real
support.

Full execution record and artifact SHAs: workbook 80 §6.
