# Pairwise MLP with Global Assignment Baseline

## Scope

This study deliberately remains pre-Transformer. It establishes a low-capacity
pairwise MLP plus explicit one-to-one station-pair assignment baseline before
considering a multi-station context model.

Every association input is produced by the physical chain:

```text
SCT_ClusterContainer -> SegmentFitRefit -> SegmentsRefit -> NtupleDumper
-> FaserActsExtrapolationTool (mode 0)
```

Each geometry point uses an independently generated SQLite conditions payload
and a new segment refit. Coordinate shifts and residual-level surrogates are
not accepted. The local segment `q/p` is a fixed propagation seed rather than
a measurement, so V1 fixes `q_over_p_mode=0`.

## Source-Disjoint Nominal Pilot

The nominal pilot uses eight independent MC24 2D-FLUKA xAOD files: four train,
two validation, and two test files, from muon-positive source `100116` and
muon-negative source `100117`. The original xAOD file is the split unit;
source provenance is checked before synthetic materialisation and at MLP load
time.

| Split | Physical source events | Synthetic events | True tracklets | Random easy fakes | Omitted true tracklets |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 39 | 480 | 5,178 | 443 | 582 |
| Validation | 19 | 240 | 2,577 | 223 | 303 |
| Test | 19 | 240 | 2,600 | 242 | 280 |

Each synthetic event overlays three distinct physical single-muon trajectories.
The nominal pilot deliberately contains random easy fakes and missing
tracklets, but no field-aware hard-negative injection (`0` hard negatives).
This limitation is recorded rather than treating the random fake distribution
as realistic; hard-negative construction is a required follow-up for the
misalignment curriculum corpus.

Validation candidate coverage is `3,461 / 3,461 = 1.000` for every station
pair. Test coverage is lower because of physical exporter/refit availability,
not a coordinate surrogate: the selected ungated test graph has candidate
recall `0.98493`.

## Field-Aware Hard-Negative Smoke Test

An initial absolute-band construction used
`hard_negative_mean_per_target_station = 0.50` and real same-payload mode-0
Acts chi2 in `[1, 1000]`. It produced 731 train and 360 validation hard
negatives, but the audit found that roughly 40--60% were more compatible with
their source than the retained truth endpoint. That is an adversarial
truth-dominating construction, not the intended ordinary near-background
population, so it is not the current curriculum definition.

The replacement keeps the same real field propagation and absolute chi2 band,
but additionally requires `1.10 <= hard_chi2 / truth_chi2 <= 10.0` whenever
the truth endpoint is retained. Its validation-only smoke produced 715 train
and 304 validation hard negatives; selection failures (24 and 48) are retained
in provenance rather than filled with a surrogate. Every hard candidate with a
comparable truth endpoint satisfied `hard_chi2 > truth_chi2`. Random easy
fakes still require an anchor chi2 of at least 1000.

The current shared `state_augmented_v2` MLP did not meet the primary point on
this deliberately hard nominal stress sample: its best quality configuration
has efficiency about 0.45 at the 250 gate, where candidate recall is only
about 0.52; the ungated graph reaches about 0.83 efficiency but 0.091 inclusive
fake rate. A station-pair ensemble was also tested and was worse under the
same validation-only protocol. No hard-negative-smoke test file was opened.
The next repair is more source-disjoint physical training data, not a
Transformer.

An exact multi-station flow screen was also run on the same pairwise MLP scores
with a compact validation-only threshold/penalty grid. Its best quality point
uses the ungated graph and unmatched penalty 0.5: efficiency `0.57835`,
inclusive purity `0.95520`, fake rate `0.04480`, and candidate recall `1.0`.
This is materially better than pairwise dustbin assignment at the same stress
quality, but remains below the declared 0.70 efficiency floor. It confirms that
global consistency helps, while also showing that it does not yet remove the
need for a better source-disjoint curriculum baseline.

## Models and Assignment

The first residual/pull-only pair MLP was insufficient: validation AUC was
`0.9171`, AP `0.8302`, but the best Sinkhorn assignment reached only
`18.98%` efficiency at `95.36%` purity. The diagnostic found high-score
different-truth pairs, particularly for `S0 -> S2` and `S0 -> S3`.

The repaired, still low-capacity `state_augmented_v2` MLP adds the persisted
refitted source and target state `(x, y, tx, ty)` to the physical residual/pull
features. It uses neither truth IDs, synthetic-role labels, a Transformer, nor
`q/p` as an association measurement. Its ungated validation candidate metrics
are AUC `0.96307`, AP `0.89888`, and calibrated ECE `0.05165`.

For every ordered station pair, the MLP scores populate the full candidate
score matrix. Four truth-blind methods are evaluated:

1. `greedy`: descending-score one-to-one reference.
2. `hungarian`: thresholded global bipartite assignment.
3. `dustbin_hungarian`: source- and target-specific dummy nodes, equivalent
   to a unit-capacity min-cost-flow assignment with an unmatched penalty.
4. `sinkhorn_hungarian`: log-domain Sinkhorn normalisation of the dustbin
   matrix followed by Hungarian rounding.

Validation alone calibrates scores and scans the candidate chi2 gate, decision
threshold, unmatched penalty, and Sinkhorn temperature. All five candidate
gates (`25`, `250`, `1000`, `5000`, and ungated) are scanned. Test files
are opened only after selected operating points are fixed. The predeclared
primary condition is inclusive fake rate <= `0.05` and inclusive purity >=
`0.95`, maximising validation efficiency within those limits.

## Sealed Nominal Test Result

| Method | Selected gate | Test efficiency | Inclusive purity | Fake rate | Missing unmatched recall |
| --- | --- | ---: | ---: | ---: | ---: |
| Greedy | `5000` | 0.82514 | 0.98540 | 0.01460 | 0.99347 |
| Hungarian | ungated | 0.82940 | 0.98781 | 0.01219 | 0.99217 |
| Dustbin Hungarian | ungated | 0.93261 | 0.97590 | 0.02410 | 0.96997 |
| Sinkhorn -> Hungarian | ungated | 0.93261 | 0.97590 | 0.02410 | 0.96997 |

The core nominal primary operating point is satisfied. The hard-negative stress
point remains unresolved, so its test split remains sealed. The completed
physical hard-background update below documents the expanded source bank and
the strictly physical `0, 0.1, 1, 5, 10, 50 mm` conditions/refit chain. None
of this justifies starting a Transformer.

## Physical Hard-Background Update

The full curriculum bank is complete with six 2D-FLUKA train files, two
validation files, and two sealed test files. Its source-event audit is
`59 / 19 / 19` and has zero cross-split event UID overlap. Two independently
produced MC24 `100043` muon-minus 5 mrad FLUKA-E xAOD files were then run
through all six real conditions payloads and added to training only. The
expanded bank has eight / two / two source files and `79 / 19 / 19` source
events. Both additions passed the same
`SCT_ClusterContainer -> SegmentFitRefit -> SegmentsRefit -> NtupleDumper ->
FaserActsExtrapolationTool(mode 0)` chain at every magnitude.

The synthetic audit is deliberately physical: three distinct source tracks are
overlaid per event, missing tracklets are retained, random easy fakes require
large Acts chi2, and field-aware hard fakes are selected from the same payload
with `1.10 <= hard_chi2 / truth_chi2 <= 10.0`. No hard fake with a retained
truth counterpart is more compatible than that truth counterpart. The full
hard-background reference uses mean `0.50` hard fakes per non-reference target
station. Controlled `0.10` and `0.25` occupancy points also remained below the
primary point, with best quality efficiencies `0.3790` and `0.3851`.

All results below use the expanded training bank, the same sealed validation
sources, and validation-only control selection. The declared primary remains
efficiency >= `0.70`, inclusive purity >= `0.95`, and inclusive fake rate <=
`0.05` at nominal geometry.

| Baseline | Best nominal quality point | Efficiency | Inclusive purity | Fake rate | Candidate recall | Test |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Shared `state_augmented_v2` | Sinkhorn + Hungarian, chi2 <= 250 | 0.4476 | 0.9580 | 0.0420 | 0.5198 | sealed |
| Shared v2, per-pair temperature | Sinkhorn + Hungarian, chi2 <= 250 | 0.4722 | 0.9523 | 0.0477 | 0.5198 | sealed |
| Exact four-station flow | ungated, 10,000 hypothesis cap | 0.4205 | 0.9517 | 0.0483 | 1.0000 | sealed |
| Six independent pair MLPs | per-pair temperature, chi2 <= 250 | 0.4361 | 0.9521 | 0.0479 | 0.5198 | sealed |
| Shared v2, threshold vector | per-pair Sinkhorn thresholds | 0.5986 | 0.9509 | 0.0491 | 0.6032 score-edge retention | sealed |
| Six independent pair MLPs, exact dense threshold vector | per-pair Sinkhorn thresholds | 0.5209 | 0.9500 | 0.0500 | 1.0000; 0.5590 score-edge retention | sealed |
| `state_hitpattern_v4` | persisted layer-side occupancy bits | 0.3576 | 0.9597 | 0.0403 | 1.0000 | sealed |

The threshold-vector result is an explicitly bounded validation-only search:
five thresholds, two initial values, two coordinate sweeps, and three unmatched
penalties. Its selected vector is `0.35, 0.60, 0.60, 0.20, 0.20, 0.20` for
`0->1, 0->2, 0->3, 1->2, 1->3, 2->3`, respectively. It improves the shared
threshold baseline substantially but still fails the predeclared efficiency
floor. At its nominal point, score-edge truth retention is only `0.068` for
`0->2` and `0.142` for `0->3`, while the shorter pairs retain `0.70--0.93`.
The long-baseline pairwise score separation is therefore the limiting effect.

The independent-pair ensemble was also rerun with an exact finite
station-pair threshold selector, rather than relying on coordinate descent.
For each fixed dustbin penalty and Sinkhorn temperature, truth-blind
bipartite assignments are evaluated separately for every pair/threshold.
Their additive correct/predicted counts are then combined through a Pareto
frontier on validation only. A dense audit of 16 thresholds, three penalties,
and temperatures `0.2, 0.5, 1.0` found the best quality-feasible nominal point
at efficiency `0.5209`, inclusive purity `0.9500`, and fake rate `0.0500`.
It uses penalty `1.0`, temperature `0.2`, and thresholds
`0.10, 0.90, 0.90, 0.30, 0.60, 0.20` in the same pair order. The long `0->2`
and `0->3` pairs retain no truth edges at this operating point. Thus the
failure to reach `0.70` is not a coordinate-descent local optimum or a
five-point threshold-grid artifact.

`Tracklet_hit_pattern` was checked in the NtupleDumper source before use: it is
a six-bit SCT layer-side pattern, with bit `2 * layer + side`, and the audited
train/validation values are in `[15, 63]`. Adding its twelve source/target
bits worsened the validation result, so it is retained as a documented
negative ablation rather than adopted as the baseline.

No Transformer, attention layer, or raw-hit retracking was started in this
pre-source-expansion result. The next baseline repair was additional physical
source coverage under the same source-disjoint and test-sealing protocol; its
outcome is recorded below.

## Expanded-Source Exact Threshold-Vector Result

The next source-coverage repair is now complete. Two independent MC24
`100044` muon-positive 5 mrad FLUKA-E xAOD files were processed through all
six real conditions payloads and added to training only. The resulting
physical corpus contains `10 / 2 / 2` source files and `99 / 19 / 19` source
events for train/validation/test. All source-event UID intersections remain
empty. The train/validation synthetic audit contains `77,797 / 15,574` true
tracklets, `7,254 / 1,394` random-easy fakes, and `9,590 / 1,701` field-aware
hard fakes. Every comparable hard fake is less compatible than its retained
truth counterpart.

The shared `state_augmented_v2` MLP was retrained on this enlarged physical
bank on CUDA. A standard single-threshold global scan still missed the primary
point (best quality diagnostic efficiency `0.44675`). Therefore the fixed
checkpoint was subjected to a validation-only, exact finite station-pair
threshold-vector search. For each of 16 score thresholds, three unmatched
penalties, and three Sinkhorn temperatures, the solver makes truth-blind
one-to-one assignments separately for every station pair. Its additive
correct/predicted counts are combined through a Pareto frontier; truth labels
are used only after assignment to select the validation point.

The selected validation configuration is Sinkhorn followed by Hungarian
rounding, penalty `1.0`, Sinkhorn temperature `1.0`, and thresholds
`0.05, 0.20, 0.90, 0.30, 0.35, 0.35` for
`0->1, 0->2, 0->3, 1->2, 1->3, 2->3`. It reaches efficiency `0.71313`,
inclusive purity `0.95075`, and inclusive fake rate `0.04925`; this is the
first physical hard-background validation point satisfying the predeclared
inclusive primary. The configuration was then frozen and evaluated once on
the sealed test sources.

| Injected magnitude [mm] | Test efficiency | Inclusive purity | Inclusive fake rate | Primary-quality status |
| ---: | ---: | ---: | ---: | --- |
| 0.0 | 0.77114 | 0.95758 | 0.04242 | pass |
| 0.1 | 0.77702 | 0.95715 | 0.04285 | pass |
| 1.0 | 0.76236 | 0.94686 | 0.05314 | fail |
| 5.0 | 0.75257 | 0.95257 | 0.04743 | pass |
| 10.0 | 0.69620 | 0.93333 | 0.06667 | fail |
| 50.0 | 0.07546 | 0.45392 | 0.54608 | fail |

This is not yet a capture-fraction curve: each magnitude currently has one
real payload direction in the sealed test sample, so the non-monotonic `1`
and `5 mm` rows must not be interpreted as a deterministic range. It does
show a clear large-offset failure while physical candidate recall remains high
for most pairs. At `50 mm`, raw candidate recall is `1.0` for five pairs and
`0.929` for `0->3`, whereas score-threshold truth recall is only `0.253`
globally.

There is an important topology caveat. The validation-selected `0->3`
threshold of `0.90` retains no direct `S0->S3` truth edge. The result therefore
satisfies the declared *inclusive pairwise* primary, but it does not establish
a complete direct four-station association chain. The next non-Transformer
study should repeat the physical payload directions and evaluate an
adjacent-station chain or a route-level unit-capacity flow with an explicit
four-station completeness metric. No Transformer was started.

## Reproduction

```bash
cd /eos/home-x/xcheng/FASER/alignment_ML
source scripts/setup_environment.sh ml

# Full physical conditions/refit bank, one source at a time.
python scripts/build_physical_curriculum_corpus.py \
  --config configs/physical_global_assignment_muon_2dfluka_curriculum.yaml \
  --output-dir outputs/mc24_muon_2dfluka_curriculum_physical_v1

# Seal source provenance and audit each refitted payload before materialising.
python scripts/assemble_physical_curriculum_manifest.py \
  --config configs/physical_global_assignment_muon_2dfluka_curriculum.yaml \
  --input-manifest outputs/mc24_muon_2dfluka_curriculum_physical_v1/physical_corpus_manifest.json \
  --output outputs/mc24_muon_2dfluka_curriculum_physical_v1/combined_physical_corpus_manifest.json

python scripts/audit_physical_curriculum_bank.py \
  --physical-manifest outputs/mc24_muon_2dfluka_curriculum_physical_v1/combined_physical_corpus_manifest.json \
  --output-dir outputs/mc24_muon_2dfluka_curriculum_physical_bank_audit_v1

# After the audit, materialise source-disjoint synthetics.
python scripts/materialize_pooled_curriculum_synthetics.py \
  --physical-manifest outputs/mc24_muon_2dfluka_curriculum_physical_v1/combined_physical_corpus_manifest.json \
  --config configs/physical_global_assignment_muon_2dfluka_curriculum.yaml \
  --output-dir outputs/mc24_muon_2dfluka_curriculum_synthetic_v1

python scripts/run_global_assignment_mlp_baseline.py \
  --config configs/physical_global_assignment_muon_2dfluka_curriculum.yaml \
  --synthetic-manifest outputs/mc24_muon_2dfluka_curriculum_synthetic_v1/synthetic_corpus_manifest.json \
  --output-dir outputs/mc24_muon_2dfluka_curriculum_mlp_v1

# Bounded per-station-pair score-threshold search using a fixed shared MLP.
# This is validation-only unless --evaluate-test is supplied after a primary
# validation point has been established.
python scripts/run_station_pair_threshold_baseline.py \
  --config configs/physical_global_assignment_muon_2dfluka_curriculum_external_mumi_station_pair_thresholds.yaml \
  --synthetic-manifest outputs/mc24_muon_2dfluka_curriculum_external_mumi_train_aug_synthetic_v1/synthetic_corpus_manifest.json \
  --checkpoint outputs/mc24_muon_2dfluka_curriculum_external_mumi_train_aug_mlp_validation_v1/mlp_pair_classifier.pt \
  --output-dir outputs/mc24_muon_2dfluka_curriculum_external_mumi_station_pair_thresholds_validation_v1

# Exact dense validation-only station-pair grid for the already-trained
# independent pair-MLP ensemble. The sealed test is not opened here.
python scripts/run_station_pair_threshold_baseline.py \
  --config configs/physical_global_assignment_muon_2dfluka_curriculum_external_mumi_pair_ensemble_station_pair_thresholds_exact_dense.yaml \
  --synthetic-manifest outputs/mc24_muon_2dfluka_curriculum_external_mumi_train_aug_synthetic_v1/synthetic_corpus_manifest.json \
  --checkpoint-dir outputs/mc24_muon_2dfluka_curriculum_external_mumi_pair_ensemble_validation_v1 \
  --output-dir outputs/mc24_muon_2dfluka_curriculum_external_mumi_pair_ensemble_station_pair_thresholds_exact_dense_validation_v1

# Expanded 100043 + 100044 training bank. Train/select on validation only.
python scripts/run_global_assignment_mlp_baseline.py \
  --config configs/physical_global_assignment_muon_2dfluka_curriculum_external_train_aug.yaml \
  --synthetic-manifest outputs/mc24_muon_2dfluka_curriculum_external_full_train_aug_synthetic_v1/synthetic_corpus_manifest.json \
  --output-dir outputs/mc24_muon_2dfluka_curriculum_external_full_train_aug_mlp_validation_v1 \
  --validation-only

# Freeze that checkpoint and perform the exact validation search.
python scripts/run_station_pair_threshold_baseline.py \
  --config configs/physical_global_assignment_muon_2dfluka_curriculum_external_full_train_aug_shared_station_pair_thresholds_exact_dense.yaml \
  --synthetic-manifest outputs/mc24_muon_2dfluka_curriculum_external_full_train_aug_synthetic_v1/synthetic_corpus_manifest.json \
  --checkpoint outputs/mc24_muon_2dfluka_curriculum_external_full_train_aug_mlp_validation_v1/mlp_pair_classifier.pt \
  --output-dir outputs/mc24_muon_2dfluka_curriculum_external_full_train_aug_shared_station_pair_thresholds_exact_dense_validation_v1

# The same command opens test only after the frozen validation primary passes.
python scripts/run_station_pair_threshold_baseline.py \
  --config configs/physical_global_assignment_muon_2dfluka_curriculum_external_full_train_aug_shared_station_pair_thresholds_exact_dense.yaml \
  --synthetic-manifest outputs/mc24_muon_2dfluka_curriculum_external_full_train_aug_synthetic_v1/synthetic_corpus_manifest.json \
  --checkpoint outputs/mc24_muon_2dfluka_curriculum_external_full_train_aug_mlp_validation_v1/mlp_pair_classifier.pt \
  --output-dir outputs/mc24_muon_2dfluka_curriculum_external_full_train_aug_shared_station_pair_thresholds_exact_dense_test_v1 \
  --evaluate-test
```

The run writes candidate recall, global matching efficiency, purity/fake rate,
unmatched recall, score calibration, and station-pair/misalignment breakdowns.
