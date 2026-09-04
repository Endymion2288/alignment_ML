# Gauge-Constrained / External-Constraint Alignment Feasibility V1

Workbook 77 (2026-09-05).  Status: **pre-registered** (frozen before any
gauge-fixed solve was run).

## Scientific question

After the pre-registered failures of every tracker-only route - the 7D
tracker-only identifiable subspace (workbook 68: unstable basis), the
cross-source stable core (workbook 69: independent validation 8/11 < 0.80),
cluster-local observables (workbooks 70-71), rigid-station-only 5DoF
(workbook 73: pooled rank 5 but not portable), and physically-distinct
K-short coverage (workbooks 74-76: physical-closure failure, 6-19
movable-station pairs per source vs the frozen `min_pairs=20`) - this
campaign answers two strictly separated questions:

**Under incomplete tracker information, what is a legal reconstruction
gauge representative, and does any existing external measurement qualify to
promote some gauge/null directions into genuine physical constraints?**

This stage is constraint-space design / feasibility only.  It does not
solve real-data corrections.

## The two concepts are kept strictly separate

- **A) software/reconstruction gauge constraint**: a linear constraint
  `G theta = 0` that selects one definite mathematical representative
  inside the tracker-unidentifiable directions.  It is NOT a physical
  measurement; gauge-fixed parameters are never reported as "measured to
  be 0".
- **B) external physical constraint**: only a truly independent
  survey/metrology quantity with a validated frame mapping, a known
  measurement covariance, and identified IOV/mechanical-stability
  provenance.  The two never share one conclusion.

## Frozen inheritance (SHA256-pinned in the config)

- Workbook 68: the 7D parameter space `(ift_dx_mm, ift_dy_mm, ift_dz_mm,
  ift_rx_mrad, ift_ry_mrad, ift_rz_mrad, C_dx)`, `S = (5,5,5,60,60,60,
  0.12)`, `rank_tolerance = 0.01`, `A = W^{1/2} J S`; pooled rank 5, null
  dimension 2 (near-null singular values 28.16 and 2.52);
  `solver_restricted_to_identifiable_subspace`; "V_null^T u_hat = 0 is the
  minimum-norm/gauge representative in scaled coordinates, not a
  measurement that null modes are physically zero".
- Workbooks 69/73/76: all identifiability failures stay frozen.  No
  post-hoc `min_pairs` lowering, no reinterpretation of the report-only
  K-short pooled rank 5 as success, no use of the workbook-76 FD spectrum
  to design or tune any gauge criterion, no rescue re-openings, no more
  same-kind tracker data to chase rank.
- Workbooks 61-67: no existing survey/metrology evidence is an ingestable
  physical constraint.  `ift_C_dx` / `ift_l0_minus_l2_dx` stay
  `feasibility_only, sigma=null`; `ift_station0_ry` stays `unavailable`;
  Nov-2022/cad_survey numbers, population Sigma, existing
  `/Tracker/Align` conditions, and Kabsch / dz-vs-x tilts remain forbidden
  as physical priors.  The workbook-67 Stations-ry software contract
  (global left-multiply about the FASER origin, `T.Rz.Ry.Rx`) stays
  frozen.

## Tracker information contract

The pooled 7D normal matrix is rebuilt from the frozen hierarchical V1
iteration-00 bank through the frozen loader with the identical
workbook-68 seven-source corpus, and regressed against the frozen
workbook-68 `identifiable_basis.json` (singular values rtol=1e-6, exact
rank/null dimensions, identifiable/null basis principal angles
<= 1e-6 deg).  A regression failure terminates the campaign as
`tracker_information_regression_failed` before any gauge use.  No new
reconstruction is run; this stage needs no HTCondor jobs.

## Candidate mathematical gauges (three, all pre-registered)

Each candidate documents: the explicit linear form `G theta = 0` (native
7-parameter order), the remaining degrees of freedom (always 5), whether
the tracker-observable prediction changes (never, verified by closure),
the relation to the Calypso alignment convention, and which physical
parameters lose independent interpretation.

1. **`named_parameter_zero_gauge`** (fixed reference rigid mode):
   `theta_dz = 0` and `theta_C_dx = 0`.  Matches the existing Calypso
   software-gauge language (workbooks 61-62: dz is not returned to
   tracks; `dz=0+/-5 mm` is a software gauge; `C_dx` is a gauge
   definition, not a measurement).  `ift_dz_mm` and `C_dx` lose
   independent interpretation; `ift_dx_mm` then means "dx given dz=0 and
   C_dx=0".
2. **`minimum_norm_scaled_gauge`** (minimum-norm representative,
   severity-scaled metric): `V_null^T S^{-1} theta = 0` - the
   representative already named in the frozen workbook-68 assumptions.
3. **`minimum_norm_native_gauge`** (minimum-norm representative, native
   metric): `(S V_null)^T theta = 0`; differs from the scaled one wherever
   S is non-uniform on the null block (expected, report-only).

Legality criterion (pre-registered): `rank(G) = 2` and
`det(G . S . V_null) != 0`, so the constraint intersects every gauge
orbit exactly once.

## Pure-gauge closure (pre-registered gates)

Following the frozen `solver_restricted_to_identifiable_subspace`
doctrine, the solve uses the identifiable restriction `A_eff = A P_id`:
the near-null directions below the frozen rank cut are excluded from the
solve, the gauge space is exactly `null(A_eff)`, and gauge invariance is
an exact linear-algebra statement.  Measurement model
`r_w = A_eff u + eps`, `eps ~ N(0, I)` in the weighted space.

Per replicate: draw a truth `u*` with identifiable and null content,
build observable-equivalent representatives `u*_k` differing only along
the null directions (identical measurements), and solve under each legal
gauge via KKT (full-parameter unconstrained Newton is forbidden).

Pre-registered gates (all must pass):

- `gauge_constraint_exactly_satisfied`: `|G u_hat|max <= 1e-9`
- `representative_unique_per_gauge`: identical representatives (<= 1e-8)
  from different equivalent truths - no drift along unconstrained null
  directions
- `observable_prediction_gauge_invariant`: cross-gauge observable
  prediction relative difference <= 1e-8
- `identifiable_projection_gauge_invariant`: <= 1e-8 absolute
- `held_out_observables_gauge_invariant`: pair half-split (frozen seed),
  solve on half 1, predict half 2, cross-gauge relative difference
  <= 1e-8
- `kkt_solve_residual_within_gate` <= 1e-8; `kkt_numerically_stable`
  (full rank, condition <= 1e12)

Explicitly NOT a gate: recovering the full injected theta truth
(gauge-equivalent truths are non-unique).  Gauge-dependent absolute
components are expected to differ across gauges (report-only).  The
near-null information discarded by the frozen rank cut is reported as a
diagnostic and never used.

## External-constraint eligibility table

Fifteen candidate quantities are audited row by row: measurement value,
real measurement uncertainty/covariance, coordinate frame,
pivot/rotation convention, Calypso parameter mapping, measurement date,
IOV/mechanical-stability provenance, and independence from tracks and
conditions.  A row is `eligible_physical_constraint=true` only if ALL
four pre-registered gates pass: validated parameter mapping, independent
measurement covariance, identified measurement-year/conditions IOV, and
independence.  The three official slots (`ift_C_dx`,
`ift_l0_minus_l2_dx`, `ift_station0_ry`) are additionally cross-checked
mechanically against the frozen workbook 66/67 slot artifacts.

Pre-registered expectation (from the frozen workbook 61-67 chain): no
existing quantity passes all four gates, freezing
`no_ingestable_external_physical_constraint_available`.  This does not
block pure-gauge feasibility.  If an eligible candidate were
nevertheless found, the decision becomes
`external_physical_constraint_candidate_found_separate_subcampaign_required`
and a separate sub-campaign must write the measurement as
`y = H theta + eps, eps ~ N(0, C)` with a code/geometry-validated `H`
and a real measurement covariance `C`; without covariance no formal
Fisher/posterior ingest may run, only a feasibility report.  This stage
never authorizes ingest.

## Pre-registered decision chain (short-circuit order)

1. Tracker-information regression failure ->
   `tracker_information_regression_failed`
2. Any illegal gauge contract -> `gauge_constraint_contract_invalid`
3. Ill-posed KKT solve / drift -> `gauge_fixed_solver_not_well_posed`
4. Any closure gate failure -> `gauge_invariant_observable_closure_failed`
5. Eligible external candidate found ->
   `external_physical_constraint_candidate_found_separate_subcampaign_required`
6. Otherwise ->
   `gauge_feasibility_closed_no_eligible_external_physical_constraint`

Frozen this stage: `gauge_constraint_contract_valid`,
`gauge_fixed_solver_well_posed`, `gauge_invariant_observable_closure`,
`eligible_external_physical_constraints`,
`external_constraint_ingest_authorized = false` (constant),
`real_data_candidate_alignment_authorized = false` (constant),
`geometry_write_allowed = false` (constant),
`official_conditions_write_allowed = false` (constant).

## Conditional branches

- Pure-gauge closure passes: no real geometry is written.  The next
  stage must separately pre-register a **Gauge-Fixed Real-Data Alignment
  Diagnostic V1** whose output is called a reconstruction gauge
  representative / candidate diagnostic, never the unique true
  mechanical station pose; it uses a calibration/held-out split and
  evaluates gauge-invariant observable improvement only on held-out
  residual/DQ populations; official COOL/POOL writes stay closed.
- A genuinely eligible external physical measurement: a separate
  sub-campaign, never mixed with the software-gauge conclusion; only
  then are tracker-only and external information compared on the
  null/near-null directions with source/IOV robustness.

## Engineering

- `configs/gauge_constraint_external_constraint_feasibility_v1.yaml` -
  campaign config with every inherited SHA256 (this pre-registration).
- `alignment/gauge_constraint_feasibility.py` - analysis module
  (tracker-information regression, gauge construction, KKT solver,
  closure, eligibility table, decision).
- `scripts/report_gauge_constraint_external_constraint_feasibility.py` -
  `validate-config` / `all` stages.
- `tests/test_gauge_constraint_feasibility.py` - 16 tests.
- `outputs/gauge_constraint_external_constraint_feasibility_v1/` -
  artifacts with config SHA, git SHA, input provenance, constraint
  matrices `G`, parameter ordering, units, null/identifiable basis
  provenance, solver method, condition metrics, gauge-invariant metrics,
  external-evidence eligibility, and unresolved assumptions.

## Results (backfilled after the analysis run)

- [ ] Tracker-information regression (negative control)
- [ ] Gauge-contract legality (three candidates)
- [ ] Gauge-invariant observable closure
- [ ] External-constraint eligibility table
- [ ] Final frozen decision and artifact SHAs
