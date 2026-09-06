# T02 Bootstrap Multiplicity and Numerical Contract

Repairs the event-bootstrap boolean-mask bug and the underdetermined SVD
null basis.  Historical workbook 68/69/73 ranks stay frozen and are not
reinterpreted as a deployment pass.

## Bootstrap

Event key = `(original_source_uid, run, event)`.  Draws keep with-replacement
row multiplicity.  Reports `n_draws`, `n_unique`, `effective_multiplicity`,
seed, and invalid-replicate reasons.

`official_event_bootstrap` already used row lists and is unchanged.

## SVD

Tall/square `A` still uses the economy right basis.  When `m < n`, the
complete right-null complement is kept so `dim(V_null) = n - rank`.

## Covariance

`require_spd` uses a finite / symmetric / Cholesky check.  The 2×2 block
`[[1,2],[2,1]]` is rejected and cannot return `chi2 = -2`.  Eigenvalues are
not clipped.

## Frozen

`rank_tolerance = 0.01`.  Scale matrix `S` is unchanged.
