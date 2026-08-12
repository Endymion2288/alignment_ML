# Multi-Direction Physical Scan and Adjacent Route Baseline

## Scope

This study remains before any Transformer implementation.  It evaluates a
frozen low-capacity pairwise MLP with a truth-free four-station route solver
on newly generated physical geometry trials.

Every physical point follows:

```text
persisted SCT_ClusterContainer -> SegmentFitRefit -> SegmentsRefit
-> NtupleDumper -> FaserActsExtrapolationTool (mode 0)
```

No tracklet coordinate shift, cached residual, residual-level surrogate,
raw-hit retracking, test-time calibration, or test-time threshold selection is
used.  Local segment `q/p` remains a propagation seed only; all candidate
construction uses `q_over_p_mode=0`.

## Physical Trial Bank

The sealed source-file test split is unchanged:

| Source xAOD ID | Charge production | Source events |
| --- | --- | ---: |
| `mc24_100116_00030_00039` | MC24 FASERnu 2D-FLUKA mu+ | 10 |
| `mc24_100117_00030_00039` | MC24 FASERnu 2D-FLUKA mu- | 9 |

For station 0 (IFT) the injected translation is fixed to `(0, 0)`.  Stations
1--3 receive independent normalized `(dx, dy)` directions for every nonzero
magnitude.  The fixed configuration seed is `20260812`.

| Injected magnitude [mm] | Independent direction trials | Fresh physical points/source |
| ---: | ---: | ---: |
| 0 | 1 | 1 |
| 0.1, 1, 5, 10, 50 | 3 each | 15 |

`0 mm` has one fresh refit because a zero vector has no physical direction.
The result is 16 points for each xAOD, or 32 newly written
`/Tracker/Align` SQLite/POOL payloads and independent refit/Acts exports.

`outputs/mc24_muon_2dfluka_multidirection_test_physical_audit_v2/` verifies
that all 32 payload points have complete assets, MC truth labels, and
positive-definite tracklet covariance.  At every nonzero magnitude, the three
directions together contain 57 refitted events and 213 tracklets across the
two source files.

An explicit response audit for source `100116`, `5 mm`, direction `test_02`
shows the requested physical behavior.  The median refitted station shifts
for stations 1--3 are `(4.8633, 1.1610)`, `(-3.3355, -3.7249)`, and
`(1.8542, 4.6435) mm`, matching the written payload translations.  The
mode-0 truth-pair residual-response-error p95 is `0.00064 mm` in x and
`0.00832 mm` in y.  This validates displaced-geometry refitting rather than a
coordinate surrogate.

## Controlled Synthetic Overlay

Each physical payload is pooled only within the two sealed test source files
and materialized into 240 synthetic events with three distinct source-muon
tracks per event, missing tracklets, random-easy fakes, and field-aware hard
negatives.  Source-file membership is checked before evaluation.

For different direction trials at a fixed magnitude, the overlay RNG seed is
identical (`overlay_seed_scope: magnitude_shared_across_direction_trials`).
Thus the source-track and random-fake draw is controlled across directions.
Geometry-dependent hard-negative acceptance can still differ, because it is
evaluated from the corresponding real mode-0 Acts records; this is physical,
not an artificial coordinate perturbation.

The background audit contains 4,595 hard-negative candidates with a retained
truth counterpart; zero have `hard_chi2 <= truth_chi2`.  Synthetic role
provenance is audit-only and never enters MLP features or route assignment.

## Frozen Pairwise Contract

| Item | Frozen value |
| --- | --- |
| MLP checkpoint | `...full_train_aug_mlp_validation_v1/mlp_pair_classifier.pt` |
| Calibration | per-station-pair temperature, fit on validation only |
| Pair threshold source | validation-only exact threshold-vector selection |
| Adjacent thresholds | `0->1: 0.05`, `1->2: 0.30`, `2->3: 0.35` |
| Dustbin penalty | `1.0` |
| Test-time calibration/selection | disabled |

The route solver consumes only calibrated adjacent scores for
`IFT -> S1 -> S2 -> S3`.  It enumerates contiguous paths of length at least
two and selects endpoint-disjoint paths with an exact small-event binary
set-packing solve.  This is equivalent to a unit-capacity route flow with an
explicit dustbin for every unused endpoint.  The solver never reads truth.
There is no direct `0->3` edge requirement.

The fixed capture criterion is:

```text
complete-track efficiency >= 0.70
complete-track purity     >= 0.95
route fake rate           <= 0.05
```

Here route fake rate is inclusive: every selected route that is not
truth-consistent counts as false, including incompatible genuine endpoints as
well as synthetic fake endpoints. The latter is reported separately as
`fake_endpoint_route_rate`. Duplicate rate is a route-fragmentation
diagnostic, but is not a newly tuned capture requirement.

## Results

Values below are means across direction trials; `+/-` is the population
standard deviation across directions.  Raw candidate-chain recall is measured
before MLP thresholds.  Score-chain recall requires all three truth adjacent
edges to survive the frozen thresholds.

| Magnitude [mm] | Capture trials | Raw candidate-chain recall | Score-chain recall | Complete-track efficiency | Complete purity | Route fake rate | Duplicate rate | Missing-station recovery |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1 / 1 | 1.000 | 0.932 | 0.778 | 0.977 | 0.037 | 0.081 | 0.970 |
| 0.1 | 3 / 3 | 1.000 | 0.958 +/- 0.030 | 0.790 +/- 0.042 | 0.984 +/- 0.000 | 0.037 | 0.089 | 0.970 |
| 1 | 2 / 3 | 1.000 | 0.954 +/- 0.033 | 0.802 +/- 0.040 | 0.987 +/- 0.004 | 0.048 | 0.069 | 0.965 |
| 5 | 0 / 3 | 1.000 | 0.793 +/- 0.210 | 0.529 +/- 0.251 | 0.962 +/- 0.028 | 0.090 | 0.096 | 0.965 |
| 10 | 0 / 3 | 1.000 | 0.620 +/- 0.246 | 0.260 +/- 0.289 | 0.946 +/- 0.042 | 0.090 | 0.141 | 0.942 |
| 50 | 0 / 3 | 1.000 | 0.000 | 0.000 | -- | 0.425 | 0.000 | 0.947 |

At `50 mm`, complete purity is undefined because no complete route is
selected.  The candidate graph nevertheless retains every complete truth
chain in all three directions.  The failure is therefore downstream of
physical candidate generation: frozen MLP score separation and thresholded
global route selection fail before the raw graph does.

The same conclusion already appears at `5 mm`: all directions retain the raw
truth chains, but none passes the fixed capture criterion.  Per-pair results,
including candidate recall, AUC/AP, association efficiency/purity/fake rate,
and unmatched recall, are in
`station_pair_trial_metrics.csv`; all raw adjacent-pair candidate recalls are
1.0 through `50 mm`.

Plots:

- `capture_fraction_vs_misalignment.png`
- `route_quality_vs_misalignment.png`

## Decision

The requested Transformer trigger condition is met formally: for large real
misalignments (`5`, `10`, and `50 mm`), the physical candidate graph retains
the truth chains while the frozen pairwise MLP plus unit-capacity route
assignment has zero capture fraction.  This establishes a controlled failure
of pairwise local scoring, not a loss caused by the chi2 candidate gate or a
single payload direction.

It does not demonstrate that a Transformer improves capture range.  It
establishes the necessary baseline condition for the next experiment:
Geometry-Aware Transformer versus this frozen, source-disjoint, physical
route baseline.  No Transformer is implemented in this study.

## Reproduction

```bash
cd /eos/home-x/xcheng/FASER/alignment_ML
source scripts/setup_environment.sh ml

# Prepare then run every test-source physical point.  Each point invokes the
# Calypso cluster -> segment refit -> Acts chain, not a residual surrogate.
python scripts/build_physical_curriculum_corpus.py \
  --config configs/physical_global_assignment_muon_2dfluka_curriculum_external_multidirection_test.yaml \
  --output-dir outputs/mc24_muon_2dfluka_multidirection_test_physical_v1 \
  --prepare-only

python scripts/build_physical_curriculum_corpus.py \
  --config configs/physical_global_assignment_muon_2dfluka_curriculum_external_multidirection_test.yaml \
  --output-dir outputs/mc24_muon_2dfluka_multidirection_test_physical_v1 \
  --resume \
  --source-id mc24_100116_00030_00039 \
  --source-id mc24_100117_00030_00039

# Materialize only the sealed test source pool; direction trials share an
# overlay RNG stream at the same magnitude.
python scripts/materialize_pooled_curriculum_synthetics.py \
  --physical-manifest outputs/mc24_muon_2dfluka_multidirection_test_physical_v1/physical_corpus_manifest.json \
  --config configs/physical_global_assignment_muon_2dfluka_curriculum_external_multidirection_test.yaml \
  --output-dir outputs/mc24_muon_2dfluka_multidirection_test_synthetic_controlled_v2 \
  --split test

python scripts/run_frozen_route_level_scan.py \
  --config configs/frozen_route_level_multidirection_test.yaml \
  --synthetic-manifest outputs/mc24_muon_2dfluka_multidirection_test_synthetic_controlled_v2/synthetic_corpus_manifest.json \
  --checkpoint outputs/mc24_muon_2dfluka_curriculum_external_full_train_aug_mlp_validation_v1/mlp_pair_classifier.pt \
  --frozen-calibration outputs/mc24_muon_2dfluka_curriculum_external_full_train_aug_shared_station_pair_thresholds_exact_dense_validation_v1/calibration.json \
  --frozen-operating-point outputs/mc24_muon_2dfluka_curriculum_external_full_train_aug_shared_station_pair_thresholds_exact_dense_validation_v1/validation_selected_operating_point.json \
  --output-dir outputs/mc24_muon_2dfluka_multidirection_test_route_level_controlled_v4
```

The output contract records SHA-256 hashes for the configuration, manifest,
checkpoint, calibration, and validation operating point in
`frozen_evaluation_contract.json`.
