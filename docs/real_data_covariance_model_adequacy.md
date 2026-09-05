# Real-Data Residual/Covariance Model Adequacy & Nonlinear-Response Prerequisite V1

Workbook 79. **Status: pre-registered** — all gates frozen before any audit
quantity is computed; results backfilled after execution and frozen in a
separate commit.

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

(to be backfilled after execution)
