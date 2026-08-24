# Four-Station Alignment (S0/IFT + S1 + S2 + S3)

## Scientific question

When IFT/S0, S1, S2, and S3 all carry real `/Tracker/Align` misalignment, can
the existing source-disjoint physical chain

```text
/Tracker/Align → SegmentFitRefit → SegmentsRefit → NtupleDumper
  → FaserActsExtrapolationTool(mode 0)
```

recover the **relative** four-station geometry by trajectory association plus
alignment?  Global common mode / gauge and survey `dz` priors must be stated
explicitly.  No station is assumed correct, and station 0 is not a default
reference.

This branch does **not** copy the historical “fix S0, fit only IFT”
assumption.  The IFT-only 5-DoF + survey-`dz` line remains frozen on the old
mainline; this branch builds a new contract.

## First-version state

| Role | Components | Count |
| --- | --- | ---: |
| Free, track-constrained | `dx, dy, rx, ry, rz` on each of S0–S3 | 20 |
| Survey-constrained | `dz` on each station, 5 mm prior | 4 |

`dz` is not promoted to a free Newton coordinate.  The 20-D space is **not**
assumed solvable; SVD of a physical Jacobian decides the admitted relative
subspace.

## Schema and injection

Machine-readable names are `s{station}_{component}`, e.g. `s2_ry_mrad`.
Every physical point writes **all four** station six-vectors.  Calypso
composition remains `T(dx,dy,dz) * Rz * Ry * Rx` (mm, rad).  GeoModel applies
a stored station delta by left multiplication `T_new = g * T_nominal`.
Coordinate or residual surrogates are forbidden.

`alignment_formulation: four_station_v1` is opt-in.  Configs that omit it keep
the historical non-empty reference set and the “reference stations stay at
identity” check.

## Gauges versus physical constraints

Solve-time charts, after the Jacobian exists:

1. **`reference_station`**: rewrite `T_i' = T_ref^{-1} T_i` so one station is
   identity.  The reference is a coordinate choice, not a true geometry.
2. **`common_mode_constraint`**: keep all four stations writable and remove a
   left SE(3) common mode `T_i' = G^{-1} T_i`, with `G` built from the mean
   payload six-vector.  Subtracting that mean componentwise is **not** the
   same map once finite rotations are present; it is only a linearized
   Jacobian diagnostic.

The FD / identifiability chart is **`unconstrained_full`**: all four stations
are probed and no station is held at identity.

Results are compared only after conversion to

```text
ΔT_ij = T_i^{-1} T_j
```

A left-common SE(3) element leaves `ΔT_ij` invariant (true gauge).  Adding
the same six-vector to every station is a true gauge for translations.  A
finite *additive* common rotation is **not** automatically the same map as a
left SE(3) rotation; that difference is a transform-semantics issue, not a
gauge label.  Forcing one station to identity without rewriting the others
is a different physical constraint.

## Phase 1 — identifiability, no Transformer

1. Independent central-FD probes of all 20 free components and the 4 survey
   `dz` columns, on 1–3 train sources, around nominal.
2. Parameter-wise sensitivity, station-pair residual response, rank, scaled
   condition, SVD vectors, covariance/correlation, source spread.
3. Explicit common translation/rotation modes, adjacent relative modes, and
   long-baseline weak modes.
4. Gauge-invariant `ΔT_ij` comparison of the two solve charts.

Do not retrain V2 and do not build a 20-D curriculum until this map exists.
If the frozen V2 association remains stable under four-station misalignment,
keep it.  Retrain only if truth edges still exist but frozen scores/routes
degrade systematically.  Train/validation stay source-file disjoint.  Sealed
test stays closed.  GPU-only if a model is ever trained.

## Phase 1 result — admitted relative subspace

Two train sources (50 events each, μ−/μ+) produced a complete 51-point
physical FD bank.  Condor cluster 1000360: 51/51 points per source, zero
`failure.json`, all content audits present.

The unconstrained 24-D chart is numerically full rank and unusable
(scaled condition ~1.6×10⁹).  Survey `dz` data σ is ~43–49 mm.  Dropping
`dz` leaves a 20-D free chart whose condition (~8.5×10⁵) still hides five
global common modes (last singular values ~6–12 versus relative `rx`/`ry`
~10⁶; station `dx`–`dx` correlation 0.96–0.98).  **Neither 24-D nor 20-D
is admitted as a Newton chart.**

A reference-station 15-DoF chart (five track-constrained coordinates of
one station dropped at solve time) is full rank with scaled condition
~2×10⁴.  Pooled native σ is ~0.2 mm / 0.15 mrad for `dx,dy,rx,ry` and
~1.2 mrad for `rz`.  Adding relative `dz` returns condition ~1.5×10⁸.
`dz` stays a 5 mm survey prior.  `rz` is the first coordinate to drop if
15-D relative closure fails.

Column-dropping the Jacobian linearized at the identity is a solve-time
gauge, not a claim that the reference station is physically correct, and
it is not the same map as a finite-rotation `T_i' = T_ref^{-1} T_i`
re-linearization.  Compare recovered geometry only through
`ΔT_ij = T_i^{-1} T_j`.  Left SE(3) common mode preserves that table;
componentwise mean subtraction does not.

The next driver is `scripts/run_four_station_relative_closure.py` on the
existing held-out points, with capture pre-registered in
`configs/physical_refit_four_station_relative_closure.yaml`.  Do not
retrain V2 unless truth edges remain while frozen scores/routes degrade.

Truth-selected 15-DoF closure on this identifiability bank passed that
pre-registered `ΔT_ij` contract for both held-out points and both
reference-station charts (workbook 50).  A common dx does not enter the
relatives.  The S3 column-dropped chart spends about half of the `rz`
tolerance on a 5 mrad finite rotation; that is linearization/chart
mismatch, not a reason to treat S0 as physically true.

Unknown-association closure uses the same bank and the same `ΔT_ij`
contract.  It does **not** retrain V2, expand Athena production, or retune
thresholds on the two held-outs.  Observation filtering must keep
`movable_station_ids = [0, 1, 2, 3]`; dropping to station 0 would silently
discard 1→2 and 2→3 edges.  The 15-DoF reduction is solve-time only: it is
not a physically true reference station.  Unconstrained 20-D remains
forbidden.  Association is scored on the identity physical ROOT files
(no overlay).  Pre-registered gates live in
`configs/physical_refit_four_station_unknown_association.yaml`
(workbook 51).  Residual reduction is DQ only.

Frozen V2 unknown-association diagnostics on this bank (workbook 52) did
**not** pass the pre-registered vs-nominal association gates.  The identity
physical graph keeps truth chains and frozen scores, but overlay is the
frozen operating-point domain.  On that overlay, raw candidate-chain recall
is 1.0 and 1→2 / 2→3 selected edges are present, yet complete-track
efficiency falls from 0.90 at nominal to 0.71 on both held-outs, concentrated
on the 2→3 / S3 edge under the injected 5 mrad `ry`.  Score thresholds still
retain every complete truth chain (473/473).  15-DoF relative WLS was not
opened.  The failure class is association domain shift; four-station-aware
GPU retraining may now be planned, without retuning these held-outs.

Workbook 53 starts that matched retraining pilot.  It does **not** redesign
the Transformer.  Relative misalignments are sampled in the admitted 15-DoF
S0 chart, then left-multiplied by a common SE(3) gauge control.  Training
reuses the two completed identifiability-pilot xAODs; validation uses two
expanded-contract sources that never entered that pilot
(`mc24_100047_00050_00099`, `mc24_100048_00050_00099`).  Architecture,
`residual_v1` features, unit-capacity packing, and the 30-epoch budget stay
with historical V2.  Threshold / unmatched-penalty / calibration selection
on the workbook-52 held-out overlays is forbidden.  15-DoF unknown-
association WLS stays closed until the source-disjoint association gate
passes.

Workbook 54 ran that pilot.  Condor cluster 1000434 finished 28/28 physical
points.  On the new source-disjoint validation overlay, frozen historical V2
still fails vs-nominal (hard `s3_ry` efficiency drop 0.13; gauge twins of the
same `ΔT_ij` disagree).  Matched retraining restores 2→3 / S3 (hard 2→3
0.51 → 0.86) and passes the gauge-invariance audit.  The pre-registered
complete-track efficiency drop ≤0.10 still fails on one payload
(`draw_00_plus_common`, drop 0.104).  15-DoF unknown-association WLS stays
closed.  The 2→3 collapse was mainly historical domain shift, not a missing
relative-geometry inductive bias; the checkpoint is not frozen for WLS.

Workbook 55 is a validation-only operating-layer audit of that retrained V2,
plus one pre-registered low-capacity control.  Event-aligned `draw_00` gauge
twins show that truth-edge ranking is stable (Platt preserves pair order;
failed 1→2 / 2→3 edges are still mostly source rank-1) while absolute scores
drop on the left-SE(3) twin.  The 0.104 efficiency miss is almost entirely
the 0.5 station-pair thresholds, not a missing candidate graph.  Nominal fake
~0.10 comes from the validation-selected `unmatched_penalty=+0.5`, which
admits short fragments.  The single control (train-only 3-pair Platt, frozen
logits, historical packing 0.001 / −1.0) restores nominal fake/purity and
clears `draw_00_plus_common`, but drops nominal efficiency to 0.45 and fails
`draw_01_plus_common` (drop 0.117).  Record: representation is basically
sufficient; the association operating layer is not at the production gate.
Do not loosen 0.10.  Do not open 15-DoF WLS.

Workbook 56 is one same-architecture V2 training-objective control:
gauge-twin consistency plus a local packing-utility margin, with the
historical body kept.  Operating convention is frozen before training
(identity Platt, threshold 0.001, unmatched_penalty −1.0).  Workbook 53–55
validation sources are `development_validation_only` and cannot claim the
production gate.  The gate moves to a new source-disjoint transfer bank
(`mc24_100047_00300_00349`, `mc24_100048_00300_00349`).  Sealed test stays
closed.  15-DoF WLS opens only if that transfer set passes, using the
already frozen workbook 49/50 capture.

## Commands

```bash
source scripts/setup_environment.sh ml
python scripts/prepare_four_station_identifiability_pilot.py \
  --source-config configs/physical_curriculum_four_station_identifiability_sources.yaml \
  --iteration-template configs/physical_refit_four_station_identifiability_pilot.yaml \
  --output-root outputs/mc24_four_station_identifiability_pilot_v1 \
  --iteration 0 \
  --nevents 50 \
  --source-id mc24_100043_00200_00299 \
  --source-id mc24_100044_00300_00399
```

Physical production uses the existing Condor wrappers.  After completion,
audit `failure.json`, ROOT readability, content audit, and manifest
completion before any scientific claim.  Then:

```bash
python scripts/audit_four_station_identifiability.py \
  --iteration-manifest outputs/mc24_four_station_identifiability_pilot_v1/iteration_manifest.json \
  --output-json outputs/mc24_four_station_identifiability_pilot_v1/identifiability_audit.json \
  --split train
```

Admit the relative subspace (train only) and then, after capture is
pre-registered, run truth-selected 15-DoF closure on the existing bank:

```bash
python scripts/admit_four_station_relative_subspace.py \
  --iteration-manifest outputs/mc24_four_station_identifiability_pilot_v1/iteration_manifest.json \
  --output-json outputs/mc24_four_station_identifiability_pilot_v1/admitted_subspace.json \
  --split train

python scripts/run_four_station_relative_closure.py \
  --iteration-manifest outputs/mc24_four_station_identifiability_pilot_v1/iteration_manifest.json \
  --observed-point iteration_00_closure_relative \
  --output-json outputs/mc24_four_station_identifiability_pilot_v1/relative_closure_relative.json \
  --operating-point configs/physical_refit_four_station_relative_closure.yaml \
  --split train
```

Unknown-association (frozen V2, identity physical bank, 15-DoF relative WLS):

```bash
python scripts/prepare_four_station_identity_association_manifest.py \
  --iteration-manifest outputs/mc24_four_station_identifiability_pilot_v1/iteration_manifest.json \
  --output-json outputs/mc24_four_station_identifiability_pilot_v1/identity_association_manifest.json

python scripts/run_frozen_association_backbone.py --backbone v2 --q-over-p-mode 0 \
  --split train --device auto \
  --synthetic-manifest outputs/mc24_four_station_identifiability_pilot_v1/identity_association_manifest.json \
  --frozen-output /eos/home-x/xcheng/FASER/alignment_ML/outputs/mc24_v3_expanded_trainval_v2_bce_control_v1 \
  --payload-id iteration_00_reference \
  --output-dir outputs/mc24_four_station_identifiability_pilot_v1/frozen_v2/iteration_00_reference

python scripts/run_four_station_route_selected_relative_closure.py \
  --iteration-manifest outputs/mc24_four_station_identifiability_pilot_v1/iteration_manifest.json \
  --observed-point iteration_00_closure_relative \
  --anchor-association-output outputs/mc24_four_station_identifiability_pilot_v1/frozen_v2/iteration_00_reference \
  --output-json outputs/mc24_four_station_identifiability_pilot_v1/unknown_association_relative.json \
  --operating-point configs/physical_refit_four_station_unknown_association.yaml \
  --split train
```

Matched association retraining (workbook 53).  Do not open 15-DoF WLS from
this path until the validation association gate passes:

```bash
bash scripts/run_four_station_association_retraining.sh prepare
bash scripts/run_four_station_association_retraining.sh submit
```

Workbook 55 operating-layer audit and the single pre-registered control
(does not open 15-DoF WLS):

```bash
python scripts/audit_four_station_route_operating_layer.py \
  --synthetic-manifest outputs/mc24_four_station_relative_association_retrain_v1/overlay_synthetic_v1/synthetic_corpus_manifest.json \
  --frozen-output outputs/mc24_four_station_relative_association_retrain_v1/retrained_v2 \
  --output-dir outputs/mc24_four_station_relative_association_retrain_v1/operating_layer_audit_v1 \
  --split validation --device auto

python scripts/run_four_station_operating_layer_control.py \
  --control-config configs/physical_four_station_operating_layer_control.yaml \
  --synthetic-manifest outputs/mc24_four_station_relative_association_retrain_v1/overlay_synthetic_v1/synthetic_corpus_manifest.json \
  --frozen-output outputs/mc24_four_station_relative_association_retrain_v1/retrained_v2 \
  --iteration-manifest outputs/mc24_four_station_relative_association_retrain_v1/iteration_manifest.json \
  --output-dir outputs/mc24_four_station_relative_association_retrain_v1/operating_layer_control_v1 \
  --device auto
```

Workbook 56 ran that pre-registered objective on the new transfer bank and
failed the production association gate: raw-chain recall passed and
origin-matched score-scale contracted versus workbook 54, but packing
selected zero routes.  Workbooks 57–58 showed the missing competitor was
dustbin 0, not a missing fragment topology.  Workbook 59 trained one
pre-registered dustbin-aware route-margin auxiliary under the same frozen
packing convention.  Transfer layer 1 passed and most payloads now select
routes (`U_truth` median +0.87), but `draw_01` plus common SE(3) missed
the vs-nominal efficiency gate and the twin route-metric gate.  15-DoF
WLS stays closed.  The new checkpoint is a control artifact, not a frozen
production V2.  Workbook 60 called the production `_route_hypotheses`
set on that frozen checkpoint: every production fragment winner was
already the workbook-59 `max(rival)` (train 14/14, transfer 513/513).
Do not open solver-in-the-loop mining.  Workbook 61 scored every train
complete truth route, not only the 14 winners: 740 / 3331 have a
2/3-station production competitor inside the frozen margin, 584 of
those cases are the per-event max dustbin-aware loss, and mean
reduction dilutes that max by the 2- or 3-fold complete-route
multiplicity.  Train is not missing short near-boundary coverage, so
do not stop as a curriculum/domain-coverage limitation and do not
reweight a nonexistent short-hard distribution.  One reduction-only
control was trained under that frozen contract (workbook 62,
`a46a35bd28eb294fe307590d4f12595f6d3bfaa8dc64aea0bf7418543605e1ef`).
Layer 1 passed and score-scale contracted versus workbook 54, but
`draw_01` plus common SE(3) still missed vs-nominal efficiency
(`Δeff=0.104`) and the twin efficiency gate (`|Δ|=0.070`).  The 584
train event-max short cases stayed inside the frozen margin (median
`Δ=0.992`).  Stop weighting / reduction / operating-point rescue.
Failure class: `objective_reduction_failure`.  15-DoF WLS stays closed.

Workbook 63 is the source-disjoint training-diversity audit.  It does
not retune workbook 59/62 weighting, reduction, margin, or the
operating point.  The already-opened workbook-56 transfer set is a
development diagnostic only and is no longer the next model's final
independent gate.  Frozen workbook-62 scoring of train / development /
transfer, plus unused non-sealed identity banks, shows that occupancy,
charge, and route multiplicity already overlap, but the
`draw_01+common` failure core sits outside the two current train
sources in `ty` / S3 `tx` (kinematic inside 0.821 / 0.809 < 0.90).
The same overlay recipe also has a large source-characteristic shift:
transfer hard rate 0.799 vs train 0.234, fragment-winner rate 0.170 vs
0.010, median `Δ` 0.413 vs 1.104.  Development `100047/100048`
00050–00099 already repeats that pattern.  Coverage class:
`source_phase_space_undercoverage`.  New source-disjoint training
files are authorized (keep the current μ± pair; add
`100043_00300`, `100044_00200`, `100047_00100`, `100048_00100`).
The workbook-62 objective is frozen as a whole.  The new final gate is
the reserved blind pair `100047_00350` / `100048_00350`, never loaded
here and never used in workbooks 48–62.  Gates stay
nominal purity ≥ 0.95, fake ≤ 0.05, all non-nominal `Δeff≤0.10`,
gauge-twin and 2→3 / S3 stability.  15-DoF WLS stays closed until that
new blind set passes.  If the frozen objective still fails there with
the same common-SE(3) truth-utility drop, the next discussion is
architecture-level relative / gauge-equivariant representation.

Workbook 56 gauge-consistent route training (one pre-registered objective;
from workbook 63 onward the opened transfer set is a development
diagnostic only):

```bash
bash scripts/run_four_station_gauge_consistent_training.sh audit-sources
bash scripts/run_four_station_gauge_consistent_training.sh prepare-transfer
bash scripts/run_four_station_gauge_consistent_training.sh submit-transfer
bash scripts/run_four_station_gauge_consistent_training.sh train
```

Workbook 59 dustbin-aware route-margin control (same frozen inference
convention; do not retune after transfer results):

```bash
bash scripts/run_four_station_dustbin_aware_training.sh train
bash scripts/run_four_station_dustbin_aware_training.sh infer-transfer
bash scripts/run_four_station_dustbin_aware_training.sh score-scale
bash scripts/run_four_station_dustbin_aware_training.sh mechanism
bash scripts/run_four_station_dustbin_aware_training.sh assess
```

Workbook 60 solver-generated hard-negative mining audit (frozen
workbook-59 checkpoint; train-only decision; do not retune):

```bash
bash scripts/run_four_station_solver_hard_negative_audit.sh all
```

Workbook 61 train-only weighting / reduction feasibility (frozen
workbook-59 checkpoint; do not load transfer to pick a reduction):

```bash
bash scripts/run_four_station_weighting_reduction_audit.sh
```

Workbook 62 hard-aware max-reduction control (frozen workbook-61
contract; one evaluation on the already-opened transfer set):

```bash
bash scripts/run_four_station_hard_aware_reduction_training.sh train
bash scripts/run_four_station_hard_aware_reduction_training.sh infer-transfer
bash scripts/run_four_station_hard_aware_reduction_training.sh score-scale
bash scripts/run_four_station_hard_aware_reduction_training.sh mechanism
bash scripts/run_four_station_hard_aware_reduction_training.sh assess
bash scripts/run_four_station_hard_aware_reduction_training.sh compare
```

Workbook 63 source-disjoint training-diversity audit (frozen
workbook-62 checkpoint; do not retune; do not load reserved blind
sources; workbook-56 transfer is diagnostic only):

```bash
bash scripts/run_four_station_training_diversity_audit.sh
```
