# Yasu 3ST q/p ↔ IFT geometry weak-coupling study

Preregistered research line on the `yasuCheck` branch.  It reuses the
WB95–116 physical CKF q/p, 5×5, LTO, ACTS transport, profile-likelihood
and identifiability infrastructure.  It does **not** start from the
SegmentFit dummy `q/p = 10^{-5}/MeV`.

## Scientific question

Does the signed `q/p` from the official S1+S2+S3 CKF
(`CKFTrackCollectionWithoutIFT`) have a bias or non-zero mean, and does
it couple to IFT alignment DoFs (`R_y`, `d_x`, slope-like `dy/dz`) as a
true weak mode that can produce a non-zero IFT unbiased-residual mean or
unstable four-station alignment?

A residual correlation is not enough.  The study must separate

- q/p reconstruction bias
- geometry weak mode
- ACTS propagation / navigation failure
- covariance miscalibration
- candidate / selection loss

## Stages

| Stage | Gate | Authorization |
| --- | --- | --- |
| 0 | Official WithoutIFT definition from source + ROOT + dump | 3ST→IFT chain — **PASS, WB117** |
| 1 | Independent 3ST→IFT prediction; IFT hits do not enter the fit | MC q/p calibration — **PASS, WB118** |
| 2 | Truth-known, source-disjoint MC: sign, scale, uncertainty, 5×5 | residual conditionals — **FAIL, WB119; Stage 3 closed** |
| 2D | Diagnose WB119 FAIL (curvature vs covariance vs reference vs charge) | residual conditionals — **recorded mixed/inconclusive, WB120; Stage 3 still closed** |
| 2E | Split curvature-information limit from covariance failure (A/B/C/D) | residual conditionals — **recorded A+D supported, C rejected, B unresolved, WB121; Stage 3 still closed** |
| 2F | Population-mean / reconstruction-bias / tail decomposition of \(E[q/p]\) | residual conditionals — **recorded mixed/inconclusive, WB122; Stage 3 still closed** |
| 2G | Extreme-tail and split/source instability of 3ST \(q/p\) | residual conditionals — **recorded mixed/inconclusive, WB123; Stage 3 still closed; ≤20-event refit diagnostic dump authorized** |
| 2H | CKF / KF-refit provenance on the ≤20 frozen identities | residual conditionals — **recorded mixed/inconclusive, WB124; Stage 3 still closed; S2-front is not a one-to-one fallback tag** |
| 2I | Fit-independent 3ST measurement-level bending / curvature proxy | residual conditionals — **contract established, mixed/inconclusive, WB125; Stage 3 still closed; YZ `bending_raw` present on clean 18-hit; two WB119 sign-flips are fit-additional; 243 758 quantified in 2J** |
| 2J | Batch calibration of WB125 `bending_raw` and CKF mapping audit | residual conditionals — **recorded mixed/inconclusive, WB126; Stage 3 still closed; raw bending present (sign 0.967, Spearman 0.954); 46.4% of 243 758 flips are fit-additional, 11.2% already flip at measurement, 41.5% have no constructable 3ST bending; high-\(p\) raw limit and matched source dependence rejected; trusted/S3 stay false** |
| 3 | `E[r_IFT|q/p]`, `E[r_IFT|t_y]`, charge / p / slope dependence | Jacobian / profile |
| 4 | `∂(d_x,R_y)/∂(q/p)`, profile likelihood, Fisher singular vectors | payload only if needed |
| 5 | Controlled physical payload after pivot/convention confirmation | optional |
| 6 | ACTS boundary scan on a fixed event/geometry batch, if required | optional |
| 7 | Real-data residual diagnostic only after MC closure | monitoring only |

Small sample → independent MC closure → batch statistics → real-data
residual diagnostic.  No large jobs before Stage 0–1.

## Inherited rules

Fail-closed access policy.  Sealed / final-blind paths stay closed.
Existing artifacts are never overwritten.  Construction / validation
sources stay file-level disjoint.  Negative results are kept.  Do not
force PASS by changing thresholds, dropping outliers, using truth q/p,
rescaling covariance, adding ridge, or inventing a prior.

LTO is reusable infrastructure, not a substitute for the official
3-station collection.  WB109 LTO keeps IFT.

## Forbidden claims until the corresponding stage PASSes

- that Yasu’s observation is explained by a q/p–geometry weak mode
- that 3ST q/p is calibrated
- that IFT residual bias is caused by `R_y` or `d_x`
- that four-station alignment instability is thereby explained
- that ACTS warning counts equal reconstruction loss
