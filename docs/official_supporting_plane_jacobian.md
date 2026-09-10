# Official Supporting-Plane Transport Jacobian Contract (Stage B / Task B14U)

Workbook 120.  WB119 froze `analytic_chain_or_chart_contract_broken`
after proving that the dummy-covariance bound-to-surface
`Propagator::Result::transportJacobian` is the **wrong Jacobian** for
official `h_i(theta)`.  This task asks one question: can ACTS 32.0.2
expose a free-state / stepping variational Jacobian on the official
Mode-B path itself?

```
source bound
  → unbounded navigator/stepper (nullopt cov, BoundaryCheck false)
  → final free state
  → supporting-plane intersection
  → local loc0
  → residual = m_loc0 - loc0
```

It does not change the WB114 statistical model, retune Gauss–Newton,
change production `stepTolerance`, pick a best FD step or tolerance,
switch the official sequential likelihood to direct-from-source,
replace the official likelihood with any ACTS Jacobian, introduce a
prior or ridge, delete 37/86, use truth q/p, repair 5D Cin, or enter
B14M / B15 / Measurement Model V2.  Dummy-cov bounded
`transportJacobian` is forbidden as the official derivative.  No
hand-written magnetic-field model is introduced.

```
theta = (loc0, loc1, phi, theta, q/p)
alpha = (loc0, theta)
nu    = (loc1, phi, q/p)
chi2(theta) = Σ r_i(theta)^T R_i^{-1} r_i(theta)
R_i = (0.08 mm)² / 12
r_i = m_loc0 - predicted_loc0_on_supporting_plane
```

Official `h_i` remains one deterministic sequential supporting-plane
trajectory.  Production `PropagatorPlainOptions::stepTolerance`
remains the ACTS 32.0.2 default `1e-4`.  The FD ladder is the frozen
WB117 set `h, h/2, h/4, h/8`.

On the official `nullopt` hop, EigenStepper leaves `jacTransport = I`
because `covTransport` is filled only when a start covariance is
present.  The allowed diagnostic is to flip that existing stepper
switch on the **same** unbounded supporting-plane hop, read the
already-maintained `jacTransport` segments, and collect their product
before `MaterialInteractor` reinitializes them to curvilinear.  That
is not dummy-cov propagate-to-bounded-surface.

The dumped official-path chain is

```
source bound
  → boundToFree at the hop start
  → product of RK free-transport D matrices
  → supporting-plane intersection / local loc0
  → residual = m_loc0 - loc0
```

`acts_composed = jacTransport * jacToGlobal * jacobian` is recorded
only as a contrast.  It includes material curvilinear resets that are
not part of official `h_i`.  The primary family is `rk_free_chain`.

## Allowed decisions

- `official_supporting_plane_jacobian_established`
- `acts_free_state_jacobian_unavailable`
- `official_path_jacobian_inconsistent_with_fd`

`jacobian_contract_established` and `b14m_reopen_authorized` require
the first case **and** control 0/1/37 PASS **and** 86 PASS, no branch
switching, target leakage = 0, and an unchanged statistical model.
Even then the next step is B14M smoke restart invariance, not the
1989-row campaign.

Official run `sbb14u_official_jacobian_20260908T184910Z_d5148ffc`:

```
decision = official_path_jacobian_inconsistent_with_fd
smoke_gate_passed = true
jacobian_contract_established = false
b14m_reopen_authorized = false
restart_invariance_authorized = false
full_sample_authorized = false
free_state_jacobian_available = true
official_loc0_matches_diagnostic = true
control_fd_converged = true
control_official_path_agrees_fd = false
focus_fd_converged = false
focus_official_path_agrees_fd = false
branch_switching = false
target_exclusion_holds = true
statistical_model_unchanged = true
next_step = keep_official_path_jacobian_diagnosis
```

This is not `acts_free_state_jacobian_unavailable`.  The official
path now has a same-path free-state Jacobian, function values match
official predicted loc0 to 0, and the long-hop `q/p` column is the
same order as official FD.  The frozen 5% column contract still
fails: control `100043/1` `loc1` on targets 1 and 2, and all focus
86 columns except some `loc0` comparisons.

## What this run showed

Official `nullopt` hops keep `covTransport = false` and
`jacTransport = I`.  The diagnostic dummy covariance is only the
variational switch.  Diagnostic supporting-plane loc0 matches official
Mode-B loc0 at 0 on every hop of all 12 rows.  All hops, including
the long magnetic hops on 37 and 86, have a free-state Jacobian
(`chain_complete = true`).

Primary `rk_free` versus frozen FD:

- `100043/0` and `100043/37`: all five columns, all three targets,
  last-pair FD rel ≪ 0.05 and `rk_free` vs FD rel ≤ 0.002.
- `100043/1`: `loc0/phi/theta/q/p` pass; `loc1` fails the 5% contract
  on targets 1 and 2 (rel 0.0616 and 0.0564).  Target 3 `loc1` passes
  (0.027).  The column itself is small (norm ≈ 0.085).
- `100048/86`: FD ladder still does not converge.  `rk_free` `q/p`
  norms are O(10³), the same order as official FD, unlike WB119's
  dummy-cov bound-to-surface `q/p` column of O(10⁻³)–O(10⁻¹).
  Column-by-column 5% agreement still fails.

The contrast family `acts_composed` still disagrees with FD on the
controls (typical rel 0.9–28).  That is the WB119 object: material
curvilinear covariance transport is not official `h_i`.

Energy-loss mean updates are not inside the RK `D` product.  Control
`q/p` columns nevertheless agree at the 10⁻³ level, so missing
`d(EL)/dθ` is not the control failure mode.

This is not a B14M PASS and does not authorize restart invariance or
the 1989-row campaign.
