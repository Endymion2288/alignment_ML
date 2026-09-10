# ACTS Transport Jacobian vs Fixed Multi-Scale FD Derivative Contract (Stage B / Task B14L)

Workbook 119.  WB118 froze `mixed_or_inconclusive` with
`acts_transport_jacobian_available = true` and
`analytic_vs_fd_not_yet_contracted = true`.  This task asks one
question: can the ACTS 32.0.2 `Result::transportJacobian` establish a
stable derivative contract for the frozen official
source→measurement map `h_i(theta)`?

It does not change the WB114 statistical model, retune Gauss–Newton,
change production `stepTolerance`, pick a best FD step or tolerance,
switch the official sequential likelihood to direct-from-source,
replace the official likelihood with the ACTS Jacobian, introduce a
prior or ridge, delete 37/86, use truth q/p, repair 5D Cin, or enter
B14M / B15 / Measurement Model V2.

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
WB117/WB118 set `h, h/2, h/4, h/8`.  The `1e-5 / 1e-6` rungs are
WB118 pre-registered falsification only.

The ACTS 5×5 (actually 6×6, including time) bound-to-bound matrix is
never compared raw to a residual FD column.  The dumped chain is

```
source bound
  → ACTS bound-to-bound transportJacobian
  → boundToFree at ACTS end
  → supporting-plane intersection / local chart
  → predicted loc0
  → residual = m_loc0 - loc0
```

Continuation is `transform_free_to_bound(intersection, time=0, freeDir, q/p)`,
not the raw ACTS end bound.  Contract columns are the first five
bound parameters.  ACTS q/p and the FD `h(q/p)=1e-6` are both `1/GeV`.

## Allowed decisions

- `finite_difference_not_reliable_for_focus_transport`
- `acts_transport_jacobian_not_numerically_converged`
- `analytic_chain_or_chart_contract_broken`
- `mixed_or_inconclusive`

`jacobian_contract_established` and `b14m_reopen_authorized` require
control 0/1/37 PASS, 86 derivative-contract PASS, no branch
switching, target leakage = 0, and an unchanged statistical model.
Even then the next step is B14M smoke restart invariance, not the
1989-row campaign.

Official run `sbb14l_derivative_contract_20260908T180541Z_33bda85a`:

```
decision = analytic_chain_or_chart_contract_broken
smoke_gate_passed = true
jacobian_contract_established = false
b14m_reopen_authorized = false
restart_invariance_authorized = false
full_sample_authorized = false
control_fd_converged = true
control_acts_chain_agrees_fd = false
focus_acts_chain_stable = true
focus_fd_converged = false
acts_end_loc0_matches_official = true
chain_complete = false
branch_switching = false
target_exclusion_holds = true
statistical_model_unchanged = true
next_step = diagnose_jacobian_chain_chart_units_or_projection
```

WB117/WB118 are not physical-nonsmoothness conclusions.  This run
does not authorize treating 86 FD failure as
`finite_difference_not_reliable_for_focus_transport`, because the
same ACTS chain already disagrees with the frozen FD on the known
FD-PASS controls 0/1/37.

## What this run showed

Zero-step geometric hops that start from the source surface agree
with official FD at the 1e-9–1e-13 level on `loc0 / loc1 / phi / theta`.
That checks residual sign (`J_r = -dh/dθ`), the first-five-column
contract, q/p units, the supporting-plane loc0 chart, and
`J_proj * J_b2f * J_acts`.  `projection_composed` equals
`acts_bound` at ~1e-16 once the ACTS end is on the surface.

The first official hop that actually integrates (`steps ≥ 1`) already
disagrees, including first hops from the source where continuation
is the identity.  Later zero-step stereo hops then inherit that
broken integrated Jacobian.

On successful hops, ACTS end loc0 still matches official supporting-plane
loc0.  The functions agree at `θ0`; the ACTS dummy-covariance
bound-to-surface `transportJacobian` is not the derivative of official
`h_i`.  After the ~3 m magnetic hop, official FD `q/p` columns are
O(10²–10³) while the ACTS-chain `q/p` column stays O(10⁻³)–O(10⁻¹).

Dummy-covariance propagate-to-surface is not the official unbounded
`SurfaceReached` + project path.  On 37 the long hop hits the 4000-step
limit and later hops fail `position not on surface`.  On 86 some hops
fail the same way.  Official Mode B still evaluates.

86 ACTS-chain columns that exist are stable across the pre-registered
`1e-4 / 1e-5 / 1e-6` rungs (last-pair relative errors ~1e-5).  That
stability is not a derivative contract with official `h_i`.

This is not a B14M PASS and does not authorize restart invariance or
the 1989-row campaign.
