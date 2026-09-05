# Wide-ty Real-Support-Matched MC Coverage Feasibility V1

Workbook 82. **Status: completed and frozen** — executed in freeze order within a
single interactive session; every gate was written into the config and frozen
before any confirmatory computation, and no gate was modified after execution.

**Final decision: `existing_mc_real_wide_ty_support_validated`.**
**Validated candidate: `floor_muon_100120` (floor-origin muon gun).**

> **Naming:** this Workbook 82 is the independent side-branch that Workbook 81
> explicitly reserved under the consecutive number 82. It is **not** a covariance
> repair — covariance repair is Workbook 83 (`Propagated-Covariance Upstream
> Repair & MC Validation V1`), which may only start after WB82 is frozen.

This campaign is **fully residual-blind**. For the whole workbook:
`held_out_accessed=false`, `real_data_alignment_authorized=false`,
`geometry_write_allowed=false`, `official_conditions_write_allowed=false`,
`external_constraint_ingest_authorized=false`, `measurement_model_validated=false`
(untouched), and `residual_blind=true` — it reads **no** alignment residual, FD
derivative, Jacobian prediction error, singular value, rank, or final-correction
performance.

## Scientific question (the only one)

Does any existing non-sealed MC/control sample have enough track kinematic
coverage to cover the WB80/81 real calibration population's (0,1)/(0,2)
`pred_tx,pred_ty` applicability domain, so that a conditional-J model could in
future be trained/validated without extrapolating to the real tracks?

Workbook 80 showed the canonical 3-source construction subset fails the transfer
support ((0,1)=0.806, (0,2)=0.872; gate ≥0.90) and froze
`jacobian_transfer_model_validated=false`. That was a **specific 3-source
subset** result. This campaign asks the broader question: across **all** existing
non-sealed MC/control, does **any** candidate cover the real calibration support?

## Real support target (`real_support_target.json`, residual-blind)

Inherited from the frozen WB80 calibration population (runs 14973, 14974); only
residual-independent track/pair metadata is read (no residual values). Primary
kinematic is the source-tracklet frame `source_tx/source_ty` (uniform across all
candidates and the real target); `pred_tx/pred_ty` is a cross-check.

| pair | n_pairs | source_tx_p95 | source_ty_p95 | pred_tx_p95 | pred_ty_p95 | gated |
| --- | --- | --- | --- | --- | --- | --- |
| (0,1) | 227 | 0.0635 | 0.0237 | 0.0653 | 0.0262 | yes |
| (0,2) | 149 | 0.0417 | 0.0128 | 0.0448 | 0.0133 | yes |
| (0,3) | 2 | 0.0293 | 0.0130 | 0.0247 | 0.0154 | report-only |

The real (0,1) population is markedly wider in ty than the canonical MC (real
source_ty_p95≈0.024 vs canonical≈0.0097) and is +ty-asymmetric (~87% of tracks
have ty>0). This is the WB80 "wide-ty" gap.

## Candidate inventory (kinematic metadata only)

Five existing non-sealed candidates were inventoried; admission used only physics
metadata/kinematics. Sealed sources (`mc24_100116_00030_00039`,
`mc24_100117_00030_00039`) and the historical 100012 test split
(`mc24_00020_00024`) are structurally excluded. Prior campaign failures (100120
physically-distinct admission, 100130 movable-station min_pairs) do **not**
auto-exclude a residual-blind coverage control, and a good coverage result does
**not** reopen those identifiability conclusions.

## Coverage results (`coverage.json`)

**Primary gate: per gated pair type, Mahalanobis 99%-envelope fraction ≥ 0.90 AND
density bin-occupancy (h=0.01) ≥ 0.90 AND sufficient statistics (≥30 pairs, ≥2
sources).** The two coverage metrics are redundant: Mahalanobis is the
WB80-consistent envelope; bin-occupancy is a robust density measure guarding
against "wide-but-sparse" / outlier inflation. A pre-registered physical
acceptance filter (|tx|,|ty|≤0.2) removes unphysical vertical-track outliers so
the Mahalanobis envelope is not inflated.

| candidate | (0,1) maha | (0,1) dens | (0,2) maha | (0,2) dens | n(0,1) | species(p_med) | coverage | particle |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| canonical_hierarchical_v1 | 0.841 | 0.881 | 0.940 | 0.960 | 1431 | muon(486 GeV) | **fail (0,1)** | ok |
| gaussian_theta_100012 | 0.749 | 0.529 | 0.893 | 0.651 | 17 | muon(100 GeV) | **fail (stats)** | ok |
| fluka2d_100116_100117 | 0.877 | 0.767 | 0.980 | 0.906 | 65 | muon(54 GeV) | **fail (0,1)** | ok |
| **floor_muon_100120** | **0.943** | **0.965** | **0.993** | **1.000** | **36123** | **muon(265 GeV)** | **pass** | **ok** |
| kshort_100130 | 0.885 | 0.700 | 0.980 | 0.832 | 43 | pion 211(113 GeV) | **fail (density)** | **mismatch** |

**Frame cross-check:** for all muon candidates the tracklet-frame and pred-frame
coverage agree (Δ≤0.03; e.g. canonical (0,1) is 0.841 in both), confirming the
source-slope frame is a good proxy for the pred frame for muons — so 100120's
tracklet-frame coverage (no propagations available) is trustworthy. The pion
100130 shows a large frame difference (0.93 vs 0.78), confirming low-momentum
pion frames cannot be mixed, but its species is already incompatible.

**Lever arm / origin audit:** all candidates and the real target share the same
source-z (−1860.15 mm, station 0) and lever arm ((0,1) 1907.6 mm) — floor origin
only changes the (tx,ty) distribution, not the station-pair geometry, so it does
not affect the conditional-J geometric dependence.

## Decision (`wide_ty_mc_support_decision.json`)

`floor_muon_100120` passes the Mahalanobis + density + statistics gates on both
gated pair types and is particle-domain compatible (muon, 265 GeV, matched lever
arm). All other candidates fail (0,1); `kshort_100130` is additionally
species-incompatible (pion).

**Decision: `existing_mc_real_wide_ty_support_validated`**
- `validated_candidates = ["floor_muon_100120"]`
- `real_kinematic_jacobian_support_validated = true`
- `conditional_j_retrain_permitted = true` (only the kinematic-support premise
  for a **future, separately pre-registered** J-retrain campaign)
- `new_mc_generation_required = false`
- `measurement_model_validated = false` (unchanged; still needs the WB83
  covariance gate)
- `real_data_alignment_v2_preregistration_allowed = false` (unchanged)

## Conclusions and boundaries

1. An existing non-sealed MC (floor-origin muon 100120) covers the real wide-ty
   calibration support, passing the 0.90 gate on both (0,1) and (0,2), with
   abundant statistics (36k pairs, 10 sources — enough for a future
   construction/validation split) and compatible species/momentum/lever-arm.
   **No new MC generation is required.**
2. The canonical hierarchical V1 (18 sources) still fails (0,1) (maha 0.841,
   density 0.881), consistent with the WB80 3-source direction — the canonical
   ty support is genuinely too narrow.
3. `kshort_100130` is pion; even where it covers part of the real angular support
   it is flagged `particle_domain_mismatch` and cannot train a muon conditional-J.
4. This campaign rewrites **no** prior conclusion: 100120's physically-distinct
   admission failure, 100130's movable-station min_pairs failure, WB80's
   measurement-model failure, and WB81's not-calibrated covariance all remain
   frozen. WB82 answers only the kinematic-coverage question.
5. **Convergence condition:** Measurement Model Reconstruction & Validation V2 may
   only open once `propagated_covariance_model_validated == true` (WB83) **and**
   `real_kinematic_jacobian_support_validated == true` (established here). The
   covariance gate is not yet met, so a direct return to real-data alignment is
   not allowed.

## Next steps

- **Workbook 83** (`Propagated-Covariance Upstream Repair & MC Validation V1`):
  source-tracklet covariance truth closure (Stage A) first, then propagation
  covariance construction (Stage B); the q/p covariance semantic repair is a
  separate contract. WB82 modifies no FaserActs covariance.
- A future conditional-J retrain campaign (separately pre-registered) may use
  `floor_muon_100120` as the kinematic-support-matched training/validation
  sample; its propagations (pred frame) must be produced and the frame
  consistency re-verified in that campaign.
