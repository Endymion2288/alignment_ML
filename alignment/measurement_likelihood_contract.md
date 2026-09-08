# Measurement likelihood contract — common-track alignment v1

Workbook 82.  This file freezes the statistical model.  It does not certify
an alignment oracle and does not reopen association training.

## Association interface (frozen, unused here)

```text
association_default_system = frozen_W64_raw_energy_plus_exact_solver
WB82 phase 1 association   = truth_only
```

W64, `raw_energy_v1`, the exact solver, route accounting, and WB74–WB81
artifacts are not inputs to this likelihood.  `continue_to_15d_relative_wls`
stays false.  `alignment/physical_jacobian.py` and route-selected relative
closure are **response / regression diagnostics**, not this solver.

## Measurement frame

- Station measurement is the 4-vector `(x_mm, y_mm, tx, ty)` in the
  **aligned station local frame**.
- Nominal placement is `N_i = T(0, 0, z_i)`.
- Alignment payload `g_i` is Calypso `T * Rz * Ry * Rx` in mm / rad.
- Composed pose: `G_i = g_i N_i` (left / GeoModel: `T_new = g * T_nominal`).
- Transport uses a uniform lab `B_y` to the **lab z of the station origin**
  `G_i(0)`, then the lab state is expressed in `G_i^{-1}`.
- Units: translations mm, slopes dimensionless, rotations rad in the payload
  and mrad in the Newton chart, `q/p` in the local nuisance.

## Local nuisance and global parameters

Each truth track has one local state at station-0 reference z:

```text
ξ = (x_mm, y_mm, tx, ty, q/p)
```

`q/p` is never a global alignment coordinate.  It is Schur-eliminated.

Two local-momentum treatments are declared:

| mode | meaning |
|---|---|
| `q_over_p_free` | no beam prior; absolute `dx` vs `ry` is only weakly constrained |
| `q_over_p_prior` | Gaussian prior, default `σ(q/p) = 1e-3`, mean = generating / beam `q/p` |

`q_over_p_free` together with free station `ry` is a physical `JG` under a
fixed lab field.  It is not algebraic `g_i^{-1} g_j` invariance.  Phase-1
smoke does **not** free `ry`; the Newton chart is translations only.  The
`q/p` prior (percent-level, `σ = 1e-3`) is the default local treatment and
is not a slack on the S3 residual.

Global Newton coordinates depend on the survey mode, the reference-station
chart (default: hold station 0 at identity), and the admitted components.
Phase-1 smoke admits `(dx_mm, dy_mm)` on **station 3 only**, matching the
injected S3 translation.  Freeing S1–S3 translations together leaves a
near-null `a + b z` slope (reference-chart `ty` partner).  That slope is a
physical `JG`, not algebraic `g_i^{-1} g_j` invariance, and is not in this
smoke.  Free `rx/ry/rz` remain in the SE(3) implementation and tests.
Results are compared with the **se(3) logarithm** of `T_pred^{-1} T_truth`
and of relative `ΔT_ij = T_i^{-1} T_j`.  Euler subtraction is not a finite
rotation residual.

## Likelihood

Do **not** add pairwise adjacent residuals as independent observations.

```text
r_t = stack_i [ m_{t,i} - h_i(ξ_t, {g}) ]
Σ_t = block-diag of unique-hit covariances
ℓ_t = r_t^T Σ_t^{-1} r_t
```

`measurement_id` is the physical tracklet identity.  The same hit cannot
enter `Σ_t` twice.  Shared-edge pairwise chi² is a diagnostic and
double-counts.

Joint GLS on `(ξ_1,…,ξ_T, θ)` is reduced by the Schur complement

```text
C_red = Σ_t ( C_t - B_t^T A_t^{-1} B_t )
δθ    = C_red^{-1} b_red
```

plus a numerical damping `λ I`, `λ = 1e-8`.  The Schur step must match the
dense joint solve to numerical tolerance.  A near-null of the reduced
normal is solved on the retained SVD subspace and reported
(`n_retained`, `n_dropped`); it is not deleted and called an observable
gauge.

## Two survey modes (not interchangeable)

| mode | dz in Newton chart | meaning |
|---|---|---|
| `fixed_dz` | no | dz held at the survey value (hard constraint) |
| `finite_survey_prior` | yes | Gaussian prior, default `σ = 5 mm` |

They are different statistical models.  A `fixed_dz` result must not be
relabelled as a finite prior, or the reverse.

## SE(3) and field

- Production update is **left** SE(3): `T ← Exp(ξ) T`.
- Right update exists only as a sign / convention control.
- Finite pose error is Lie-log, including at nonzero nominal pose.
- Changing the reference station is a chart rewrite; relatives are compared
  after `T_i' = T_ref^{-1} T_i`.
- If the lab field is held fixed, a common detector motion is a physical
  `JG`, not an observable gauge.
- A true coordinate change rotates field, material, and surfaces together.
- Algebraic invariance of `g_i^{-1} g_j` is a relative-chart identity, not
  that field-fixed symmetry.

## Iteration (pre-registered)

```text
max iterations        <= 10
damping               = 1e-8  (fixed)
relinearize           at the current geometry
stop if               scaled ||u|| < 1e-3
                      AND |Δχ²_val|/χ²_val < 1e-3
```

Reaching iteration 10 is `max_iterations`, not PASS.
`u_k = θ_k / scale_k` with scales `(5 mm, 5 mm, 5 mm, 60 mrad, 60 mrad, 60 mrad)`.

## Failure handling

Fail closed (no finite “success” update) on:

- singular / ill-conditioned reduced normal
- NaN / Inf in residual, Jacobian, covariance, or update
- non-SPD measurement covariance

## Independent closure (not this phase’s bulk job)

Scientific validation is

```text
inject → reconstruct/refit → truth association → solve →
apply geometry → refit / repropagate → held-out residual
```

Same-event `observed − nominal` counterfactual is not that chain.
Phase 1 only runs a single-condition smoke of this chain in the declared
toy field model.  `alignment_oracle_qualified` stays false until
independent replica coverage exists.

## Forbidden

00350 model selection, `00800-00849`, `100116_*`, `100117_*`, Final Blind,
sealed test, ML association inside the solver, robust-loss masking of a
truth-only failure, and any sweep of `δ_max` / fake slack / association loss.
