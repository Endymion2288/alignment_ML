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
