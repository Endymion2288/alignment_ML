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
