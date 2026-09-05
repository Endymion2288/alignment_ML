# Real-Data Residual/Covariance Model Adequacy & Nonlinear-Response Prerequisite V1

Workbook 79. **Status: completed and frozen** — pre-registered as `0257c9e`,
executed in freeze order, results backfilled and frozen.

**Final decision: `real_data_statistical_model_multiple_failures`.** All three
key model-adequacy audits fail: covariance/statistical model (H2),
Jacobian-transfer support (H3), and calibration cross-run transportability.
The WB78 out-of-support candidate is a **spurious alignment amplitude produced
by projecting the residuals through a mis-specified near-singular combined
covariance**, not a genuine large geometry displacement. Real-data nonlinear
alignment is **not** authorized; continue `residual_dq_monitoring_only`; no
Workbook 80 is opened.

This is a **model-adequacy audit**, not a geometry correction and not a
nonlinear alignment. For the whole workbook:

- `geometry_write_allowed = false`
- `official_conditions_write_allowed = false`
- `real_data_candidate_alignment_authorized = false`
- `external_constraint_ingest_authorized = false`
- `held_out_accessed = false` (held-out remains completely closed)

## Scientific question (the only one)

Is the Workbook-78 out-of-support candidate (|γ|max = 2.46, 16× the frozen
0.15 linear envelope) caused by:

- **H1** — a genuine large nonlinear geometry displacement (the
  residual/covariance statistical model itself is adequate), or
- **H2** — a covariance/statistical-model mismatch: the combined 4×4
  covariance's correlation structure mis-weights near-singular eigen-directions
  on the real collision population, inflating χ² and the inferred γ, or
- **H3** — an observable/Jacobian-transfer mismatch: the station-pair mean MC
  Jacobian transfer error on the real collision topology significantly biases
  the inferred alignment direction?

Until this is answered, real-data nonlinear alignment is forbidden.

## Frozen inputs (no population re-selection)

- **Calibration**: runs 14973 + 14974 and the frozen route/event/source
  identity and split SHA
  `c9c359793c41b59df1296e3ec2075af652c3f585746ca832a9c2e37f4a4f61d3`. No new
  run, no event removal, no route re-selection.
- **Held-out**: 14975/14976 + 7 monitoring runs + 14977 report-only remain
  completely closed. Workbook 79 never reads a held-out residual.
- All model-adequacy studies use only MC/control and the already-open
  calibration 14973/14974.
- Tracker information / gauge / transfer contracts are inherited from the
  frozen WB78 config (loaded and SHA-verified, never duplicated).

## Pre-registered design

### Stage 0 — exact reproduction of the WB78 baseline (hard stop)

Before any audit quantity, rebuild the calibration bank and one-shot solve
with the same deterministic WB78 chain and verify the frozen baseline
(rtol/atol = 1e-9; exact bank SHA match): bank SHA256 `6e4eaae0...`, χ²_zero =
1999613.6054909483 (378 pairs), informed rank 3, information eigenvalues
(8.04e-5, 1.70, 68.5, 117.3, 2608.0), restricted condition 38.06, γ =
(−2.460, +0.857, −0.403), bootstrap 17/50 rank changes and 158.133°. Any
mismatch freezes `real_data_model_adequacy_inconclusive` immediately.

### Stage 1 — covariance eigenstructure / whitening audit (H2 primary)

WB78 observed a zero-candidate χ² ≈ 2.00e6 (1512 dof, χ²/dof ≈ 1322) while the
diagonal marginal pulls are all small (< 0.1) — strong evidence that χ² is
dominated by the combined covariance's near-singular correlation directions.

Read-only per-pair audit of the combined 4×4 covariance: eigenvalues,
condition number, correlation matrix, smallest-eigenvalue direction, residual
projection onto covariance eigenvectors, Mahalanobis contribution by
eigenmode, marginal pulls, by pair type and by run.

**Whitening test (primary gate)**: z = L⁻¹(r − m_cell), where m_cell is the
per-(run, station-pair-type) coherent mean of the raw residual (a
diagnostic-only alignment-bias proxy, never a new residual definition or
alignment input). If the frozen covariance describes the fluctuations around
the coherent mean, the whitened χ²/ndof is O(1).

- **Pre-registered gate (generous to the model)**: `whitened_chi2_resid / ndof
  ≤ 4.0`, with ndof reduced by the estimated cell means.
- Report-only: full eigenstructure, correlations, χ² concentration by
  covariance eigenmode (especially the smallest-mode fraction), marginal
  pulls, condition distributions, empirical Cov(z) eigenvalues/condition.

### Stage 2 — empirical covariance cross-check (14973 ↔ 14974)

Build a diagnostic empirical residual covariance (never promoted to a weight).
Per station-pair type on (0,1) and (0,2) (sufficient statistics), compare the
empirical covariance structure between runs 14973 and 14974; (0,3) has ≤1 pair
per run and is report-only.

- **Primary metric (basis-independent, robust to near-degenerate spectra)**:
  generalized-eigenvalue RMS log-deviation between the two runs' empirical
  covariances. **Pre-registered gate: ≤ ln(3) (factor-3).**
- Report-only: principal-eigenvector angle, per-eigenvalue log-ratios,
  marginal scales, correlation signs, empirical-vs-propagated comparison.
- Cross-checks 14973→14974 and 14974→14973 to avoid self-proof.
- A severe empirical-vs-propagated mismatch freezes
  `real_data_covariance_model_not_adequate`; a separate pre-registered
  covariance-model campaign is then required — Workbook 79 never swaps in an
  empirical W to re-solve a candidate.

### Stage 3 — alignment-score contribution decomposition (report-only)

Decompose the frozen transfer model's normal-equation score b = JᵀWr per
informed direction γ_k by run, station-pair, observable (x/y/tx/ty), and
covariance eigenmode. Report cumulative contribution (top 1%/5%/10%/50% of
pairs to |b_k|) to answer whether γ=−2.46 is driven coherently by many pairs
or amplified by a few near-singular covariance directions. No pair is dropped.

### Stage 4 — bootstrap-instability decomposition (report-only + diagnostic counterfactual)

WB78 bootstrap: 50 replicates, 17/50 rank changes, max direction angle 158°.
Decompose: rank distribution, eigenvalue distribution around the rank cut,
mode persistence, which pair type controls the unstable mode, and whether the
instability is covariance-weight driven (via the diagonal-covariance
diagnostic counterfactual). The aim is to attribute the instability to
coverage/statistics vs the covariance correlation model — not to find a more
stable W.

### Stage 5 — diagnostic-only covariance counterfactuals (A/B/C/D)

All marked `diagnostic_only=true, alignment_authorized=false`:

- **A** `full_frozen`: official baseline (= WB78).
- **B** `diagonal_marginal`: same marginal variances, off-diagonal = 0.
- **C** `condition_capped`: eigenvalue floor at λ_max/1e3.
- **D** `unit_weight`: equal weighting of all four observables.

Compare only: candidate direction, magnitude, rank, bootstrap stability, χ²
concentration. B/C/D are never promoted to a new alignment model, never
produce a deployable candidate, and a smaller γ under B/C/D is not a fix. If
the conclusion is extremely sensitive to the covariance treatment, that is
model-mismatch evidence, not authorization to pick a best W.

### Stage 6 — observable/Jacobian-transfer support audit (H3)

No new real-data FD probes. Check the mean-J transfer error structure:

- **(a) MC per-pair J dispersion (frozen MC property, gated)**: relative
  Frobenius deviation of each per-pair J from the station-pair mean J.
  **Pre-registered gate: RMS relative deviation ≤ 0.35** (same scale as the
  WB78 injection-recovery gate the transfer already passed).
- **(b) real-vs-MC kinematic support overlap (residual-blind, gated)**: in the
  track-slope plane (pred_tx, pred_ty) per station pair, a real pair is
  "within support" if its robust Mahalanobis² to the MC cloud is within the
  99% contour (χ²₂(0.99) = 9.210). **Pre-registered gate: ≥90% of real pairs
  within support for each of (0,1) and (0,2)**; (0,3) is report-only (2
  pairs).
- No residual-based track selection; no nearest-neighbour J substitution.

### Stage 7 — calibration cross-run transportability (report-only solves, gated)

Report-only one-shot diagnostic solves on 14973-only and 14974-only (not new
corrections; no held-out). Compare rank, informed-subspace projector Frobenius
distance (report-only), γ direction, γ magnitude, predicted observable
direction.

- **Pre-registered gates (inheriting the WB68/77 15° contract)**: equal ranks;
  dominant informed eigenvector angle ≤ 15°; γ-hat direction angle ≤ 15°.

### Decision tree (pre-registered, exactly one terminal string)

Hard gates (WB78 baseline reproduction, finite data, bank/CSV bit-exact
reproduction) failing → `real_data_model_adequacy_inconclusive`.

Otherwise count soft failures among {covariance_adequacy (H2),
transfer_support (H3), cross_run_transportability}:

- 0 failures →
  `real_data_model_adequacy_pass_nonlinear_response_preregistration_allowed`
- 1 failure → the specific string:
  - covariance → `real_data_covariance_model_not_adequate`
  - transfer → `real_data_jacobian_transfer_support_not_adequate`
  - cross-run → `real_data_calibration_information_not_cross_run_stable`
- ≥2 failures → `real_data_statistical_model_multiple_failures`

If any of covariance/J/support/transportability fails, no nonlinear geometry
campaign is opened; continue `residual_dq_monitoring_only`. Only a full pass
allows a separately pre-registered Workbook 80.

## Future nonlinear response (not in this campaign)

Workbook 79 is model-adequacy only. Only if it shows (1) no severe
covariance/statistical mismatch explaining the large γ, (2) acceptable
transfer support, (3) stable cross-run direction, and (4) remaining evidence
favouring a genuine nonlinear response, may a separate Workbook 80
pre-register `Calibration-Only Nonlinear / Trust-Region Reconstruction
Response Feasibility V1`. Workbook 80 would still use only calibration
14973/14974, keep held-out closed, use temporary/non-official geometry only,
write no COOL/POOL, pre-freeze trust radius/probe grid/stop rule, never tune
on held-out, run full reconstruction per step, and first validate a response
curve (e.g. γ = 0, ±0.05, ±0.10, ±0.15) for monotonicity, local smoothness,
and linear-prediction consistency — with probe values inherited from frozen
support contracts, never expanded to ±2.5 because γ=2.46.

## Engineering

- New: `configs/real_data_residual_covariance_model_adequacy_v1.yaml`,
  `alignment/real_data_covariance_model_adequacy.py`,
  `scripts/report_real_data_covariance_model_adequacy.py`,
  `tests/test_real_data_covariance_model_adequacy.py`, bilingual docs,
  `outputs/real_data_residual_covariance_model_adequacy_v1/`.
- No new reconstruction: existing real-data outputs + MC reference
  propagations + linear algebra, interactive.
- Driver stages structurally enforce the freeze ordering: `validate-config →
  reproduce → covariance-audit → score-decomposition →
  bootstrap-decomposition → counterfactuals → transfer-support → cross-run →
  decide`. Every audit stage requires the Stage-0 baseline reproduction to
  pass; `decide` reads the frozen audit artifacts.
- Every diagnostic counterfactual carries `diagnostic_only=true,
  alignment_authorized=false` in JSON.
- Regression tests cover: held-out path inaccessibility, frozen calibration
  population, no event dropping, counterfactual W never entering the
  production solver, empirical covariance never auto-promoted to a weight, no
  geometry write, no external prior, deterministic contribution decomposition,
  independent 14973/14974 reporting, exact WB78 baseline χ²/candidate-score
  reproduction, and every decision-tree branch.

## Results

**Frozen decision: `real_data_statistical_model_multiple_failures`** (failed:
covariance_adequacy, transfer_support, cross_run_transportability).

The campaign executed in freeze order. Stage 0 reproduced the frozen WB78
calibration baseline exactly (11/11 checks: bank SHA256 `6e4eaae0...`,
χ²_zero = 1999613.6054909483, informed rank 3, eigenvalues, γ, β, condition
38.06, bootstrap 17/50 & 158.133°).

### H2 — covariance/statistical-model adequacy: decisive failure

- Zero-candidate χ² = 2.00×10⁶ (1512 dof, χ²/ndof = 1322.5); **99.74% of χ² is
  concentrated in the smallest covariance eigenmode**. The combined covariance
  is numerically near-singular (condition median 2.3×10¹², p95 4.1×10¹³;
  **99.7% of pairs have condition > 10⁸**; median smallest eigenvalue
  6.2×10⁻⁷).
- **Whitening test (primary gate)**: after removing the per-(run, pair-type)
  coherent mean, the whitened residual χ²/ndof = **1058.46 ≫ gate 4.0**. Even
  with the most generous coherent-mean removal, the frozen covariance does not
  describe the residual fluctuations (empirical Cov(z) eigenvalues [1.08,
  4.18, 21.27, 4159.09], condition 3841, should be ≈1).
- **Mechanism (verified)**: the full and diagonal covariances give a
  bit-identical information matrix, eigenvalues and β (‖H_full−H_diag‖/‖H_full‖
  = 0.0) — the near-singular direction is **alignment-blind** (in the design
  matrix's left null space), inflating χ² by ~1750× without changing the
  alignment solution.
- **Diagonal counterfactual control**: with the same marginal variances but no
  off-diagonal correlation, χ² = 1140.2 (χ²/ndof = **0.75**, healthy). The
  problem is isolated to the off-diagonal correlation model.
- `inv(C)` is numerically unreliable (worst pair condition 1.6×10¹⁴,
  ‖C·C⁻¹−I‖max = 5.7×10⁻⁶).
- **Cross-run empirical covariance not reproducible**: (0,2)
  generalized-eigenvalue RMS log-deviation = 1.274 > gate ln(3) = 1.099
  ((0,1) = 1.058 passes).

### H3 — observable/Jacobian-transfer support: failure

- (a) MC per-pair J dispersion **passes** (rms 0.244 / 0.078 / 0.111 for
  (0,1)/(0,2)/(0,3), all ≤ gate 0.35) — the mean-J is a reasonable summary on
  MC.
- (b) Real-vs-MC kinematic support overlap **fails**: only **86.34% < gate
  90%** of real (0,1) pairs lie within the MC 99% Mahalanobis (tx, ty)
  envelope ((0,2) 96.64% passes; (0,3) report-only). The real (0,1)
  track-slope support is significantly wider than MC (tx p95 0.065 vs 0.037;
  **ty p95 0.026 vs 0.0087, ~3×**), so the mean-J transfer extrapolates for
  ~14% of (0,1) pairs.

### Score decomposition, bootstrap, counterfactuals (report-only)

- The score b = JᵀWr is driven entirely by the smallest covariance eigenmode
  (mode 0 contributes (−87443, 36486, −793957); modes 1–3 are ~1000× smaller)
  and is highly concentrated (top 1% of pairs contribute 44–58% of |score|).
  This is the signature of a few near-singular directions amplifying the
  residual, not many pairs pushing coherently (which would be the H1 signal).
  Split by run, 14973 and 14974 push γ_0 in **opposite** directions (−155031
  vs +67242), further confirming a non-coherent, non-physical signal.
- Bootstrap: 17/50 replicates drop to rank 2; the fragile rank-3 mode
  (information eigenvalue ~68.5, near the 26.08 cut) fluctuates 3.5–149.6; the
  scarce (0,3) pairs (only 2) contribute a high but fragile eigenvalue. The
  diagonal counterfactual is equally unstable (17 changes, 158°), while
  condition-capped/unit are stable but give smaller γ — the conclusion is
  sensitive to the covariance treatment (model-mismatch evidence, not a
  license to pick a best W).

| counterfactual | rank | max|γ| | χ² | smallest-mode χ² frac | bootstrap rank changes | angle vs full |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A full_frozen | 3 | 2.460 | 2.00×10⁶ | 0.997 | 17 | — |
| B diagonal_marginal | 3 | 2.460 | 1140.2 | 0.062 | 17 | 0.0° |
| C condition_capped | 3 | 0.574 | 370.5 | 0.124 | 0 | 15.2° |
| D unit_weight | 3 | 0.772 | 1.82×10⁶ | 0.934 | 0 | 17.4° |

All counterfactuals are `diagnostic_only=true, alignment_authorized=false`;
none entered the production solver or produced a deployable candidate.

### Cross-run transportability: failure

- Ranks differ (14973 = 3, 14974 = 2). Dominant informed eigenvector angle
  5.85° ≤ 15° (OK), but the **γ-hat direction angle is 88.16° ≫ 15°** and
  max|γ| differs ~15× (2.439 vs 0.160). A genuine coherent geometry
  displacement (H1) would be transportable across runs; the near-orthogonal
  inferred directions decisively reject H1.

### Consequence

The WB78 out-of-support candidate is a spurious artifact of the mis-specified,
numerically near-singular combined covariance (with a contributing transfer
support shortfall for wide-ty (0,1) tracks), not a genuine geometry
displacement. `geometry_write_allowed=false`,
`official_conditions_write_allowed=false`,
`real_data_candidate_alignment_authorized=false`,
`external_constraint_ingest_authorized=false`, `held_out_accessed=false` all
remain. Continue `residual_dq_monitoring_only`. Before any alignment solve can
be trusted, a **separate pre-registered campaign** must re-derive and
re-validate the combined-covariance correlation model (and the transfer
support for wide-ty tracks); Workbook 80 nonlinear response is not opened
until the covariance model is repaired and re-frozen.

Full execution record and artifact SHAs: workbook 79 §7.
