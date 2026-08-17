# Geometry-Aware Transformer V3: Structured Global Assignment Learning

## Status and Boundary

V3 is an implemented train/validation-only hypothesis. The expanded physical corpus is complete, but V3 has not yet been trained or assessed on it; this record therefore contains control results where available, not a V3 performance claim.

The V1 test sources remain permanently sealed. V3 configuration, training, calibration, threshold selection, and physical-source production explicitly exclude `test`; no V1 test event, artifact, or new test bank is opened or created.

The physical input contract is unchanged:

```text
SCT_ClusterContainer -> SegmentFitRefit -> SegmentsRefit -> NtupleDumper
-> FaserActsExtrapolationTool (mode 0)
```

Every curriculum point is a real `/Tracker/Align` conditions payload followed by a new segment refit and Acts propagation. V3 does not make coordinate surrogates, modify persisted tracklet states, or use local segment `q/p` as a measurement.

## Fixed Model and Solver

V3 retains the V1/V2 route encoder: `d_model=128`, four sparse blocks, eight heads, and FFN `128 -> 256 -> 128`. It consumes the existing ungated mode-0 physical candidate graph and its residual/covariance features. Complete candidates are only chains of already-existing adjacent physical edges:

```text
IFT -> S1 -> S2 -> S3
```

The established unit-capacity route solver remains the inference backend. V3 adds a generic exact unit-capacity packing interface for loss-augmented training only; it does not change the candidate graph, matching data definition, or final solver.

The learned complete-route utility is:

```text
U_theta(r) = sum_{e in r} edge_logit_theta(e) + route_residual_theta(r).
```

Partial/missing-station routes retain the existing edge-based utility. Truth IDs, fake labels, hard-negative labels, and source provenance are supervision or audit fields, never model features.

## Structured Objective

For each event, V3 constructs `Y` as a maximum-cardinality truth-consistent unit-capacity assignment of complete routes. It uses the same exact solver to find the strongest feasible competing assignment:

```text
Y_hat = argmax_A [ U_theta(A) + m * Delta(A, Y) ]
L = max(0, U_theta(Y_hat) + m * Delta(Y_hat, Y) - U_theta(Y)).
```

`Delta` penalizes mixed-truth routes, fake-endpoint routes, duplicate truth alternatives, and endpoint-conflicting alternatives. Hard negatives add an explicit severity term. This directly trains against the event-level failure modes observed in V2 instead of independent row-wise BCE.

Independent edge BCE and independent route BCE are disabled in the primary V3 run. Early stopping ranks checkpoints by validation structured hinge, then validation violation fraction; calibration and route thresholds are not used for checkpoint selection.

Complete routes are four-endpoint hyperedges, so ordinary matrix Sinkhorn is not an exact relaxation. A separately configured capacity-normalized differentiable soft-assignment hyperedge surrogate is available as a labelled control. Its default weight is zero and it does not alter the primary structured-margin result.

## Validation Protocol

After a checkpoint is fixed, `evaluate_structured_assignment_v3.py` opens only validation samples. It fits station-pair edge calibration and selects edge thresholds, unmatched utility, and V3 route-utility temperature only on validation. It reports both the V3 route-utility stream and a same-checkpoint edge-only route-solver control.

The frozen validation gate is:

```text
nominal complete efficiency >= 0.70
nominal complete purity     >= 0.95
nominal fake rate           <= 0.05
```

It also requires a clear 5/10 mm improvement over retrained MLP and V1 controls on the same expanded corpus. A same-architecture expanded V2 BCE/focal-loss control is also predeclared: it isolates the structured objective from route-encoder capacity and data volume. The frozen V2 result remains supplementary historical context. Only a passed validation hypothesis permits a new source-disjoint multidirection test bank.

## Matched Control Retraining

The expanded comparison has explicit train/validation-only controls. `run_global_assignment_mlp_baseline.py --validation-only` now filters the manifest before any test asset is resolved and emits `route_calibration.json` for the ungated station-pair route control. `train_geometry_aware_transformer_v1.py` also filters before loading and never resolves a legacy `frozen_route_test` reference.

Use `physical_global_assignment_mlp_v3_expanded_control.yaml` for the shared pairwise MLP and `geometry_aware_transformer_v1_expanded_control.yaml` for the four V1 ablations. Both declare `allowed_splits=[train, validation]`, `forbidden_splits=[test]`, the unchanged mode-0 physical graph, and GPU training. `geometry_aware_transformer_v2_expanded_bce_control.yaml` is the matched route-encoder BCE/focal-loss control; it has the same 128-dimensional four-block backbone as V3 and no historical or test reference.

## Expanded MLP Control Result

The retrained pairwise MLP has completed its train/validation-only run in `outputs/mc24_v3_expanded_trainval_mlp_control_v1`. Its contract records 994 train and 796 validation original source events, zero UID overlap, `test_events_loaded=false`, and no test artifact access. Station-pair temperature calibration reduces the aggregate ungated validation ECE from `0.0992` to `0.0344` (373,325 candidate rows).

At its validation-selected station-pair multistation-flow point (score threshold `0.10`, unmatched penalty `0.5`), the nominal result is association efficiency `0.9578`, purity `0.9938`, fake rate `0.00621`, and candidate truth-edge recall `0.99815`. This pairwise result is not the four-station primary operating point: at `5 mm` it falls to efficiency `0.4447`, purity `0.9273`, and fake rate `0.0727`, despite candidate truth-edge recall `0.99775`.

The separately evaluated four-station route control in `outputs/mc24_v3_expanded_trainval_mlp_route_validation_v1` uses the frozen MLP and its station-pair calibration, then selects only route thresholds and dustbin utility on validation. Its chosen controls are threshold `0.001` for each adjacent pair and unmatched penalty `0.0`. Nominal complete-track efficiency is `0.9094` and complete-track purity is `0.9724`, but route fake rate is `0.06286`, above the frozen `0.05` limit, so even nominal capture is false. At `5/10/50 mm`, complete-track efficiency is `0.0`; meanwhile the physical candidate complete-truth-chain recall remains `1.0` at every scanned magnitude. This is evidence that the current bottleneck is scoring/structured competition rather than candidate retention. It is a validation-only control result, does not authorize a new test bank, and is the explicit baseline for V1/V2/V3.

The validation-refreeze utility now also loads only `train` and `validation` entries from a partial corpus manifest; it never requires a future test split. Its legacy optional `selection_policy` metadata has a documented default, so an expanded control cannot fail after completing its computation merely because that label is absent.

## Expanded V1 Controls

All four V1 controls now have source-disjoint validation operating points. The ordinary sparse and full-context geometry-aware checkpoints were trained once and then validation-refrozen without changing weights; the no-chi2 and no-context controls were independently trained. Every result below uses the same physical mode-0 candidate graph, station-pair Platt calibration, the validation-only route grid, and no test asset.

| V1 control | candidate AP | nominal eff / purity / fake | 5 mm eff / purity / fake | 10 mm eff / purity / fake | capture magnitudes |
| --- | ---: | --- | --- | --- | --- |
| ordinary sparse | 0.8858 | 0.9637 / 0.9708 / 0.0350 | 0.9125 / 0.9464 / 0.0586 | 0.2980 / 0.7934 / 0.1217 | 0, 0.1, 1 mm |
| geometry-aware | 0.8908 | 0.9547 / 0.9664 / 0.0432 | 0.8833 / 0.9484 / 0.0593 | 0.2802 / 0.7761 / 0.1200 | 0, 0.1, 1 mm |
| geometry-aware, no explicit chi2 term | 0.8938 | 0.9552 / 0.9659 / 0.0439 | 0.9079 / 0.9502 / 0.0569 | 0.3297 / 0.7907 / 0.1170 | 0, 0.1, 1 mm |
| geometry-aware, no multi-station context | 0.9157 | 0.9358 / 0.9554 / 0.0481 | 0.8690 / 0.9460 / 0.0586 | 0.5760 / 0.8879 / 0.0959 | 0, 1 mm |

The frozen primary criterion is not met at 5 mm by any V1 control, mainly because route purity/fake rate fail even where candidate truth-chain recall remains intact. Full-event geometric context did not improve this matched V1 comparison. These results motivate testing the V2 route objective and V3 structured assignment objective; they do not authorize test creation.

## Exact Oracle Throughput

`scripts/audit_structured_route_oracle.py` audits only declared train or validation samples. On the 7,200 expanded train graphs, it finds 116.8 complete physical route candidates per graph on average (median 108, p99 336; maximum 720); the endpoint-conflict component is usually the whole event rather than independent tracks. A generic MILP benchmark averaged 24.2 ms/event over a deterministic 64-event sample.

The exact unit-capacity solver now uses a proven fixed-order multipartite dynamic-programming fast path for complete IFT/S1/S2/S3 tuple components, with fallback to the existing MILP for mixed route lengths, non-partitioned endpoint IDs, or large states. Unit tests compare it with brute-force optimum and retain the MILP fallback coverage. The current corrected 64-event audit averages 4.61 ms/event (median 2.99 ms) while preserving the candidate graph and mathematical objective. This removes a practical oracle bottleneck without changing V3 supervision.

## Expanded Physical Corpus

`configs/physical_curriculum_v3_expanded_trainval.yaml` defines ten MC24 train source files and eight source-file-disjoint validation files. Both splits contain `mumi` and `mupl` production chunks. The requested target is up to 1,000 train and 800 validation source-event slots before reconstruction acceptance; exact accepted counts are recorded only after each physical chain completes.

The production is now complete. The single-process physical-manifest refresh accepted all `108/108` planned points: 60 from ten train source files and 48 from eight validation source files. Calypso-exported provenance contains 994 distinct train and 796 distinct validation `source_event_uids`, with zero train/validation overlap; no test source or artifact is in this corpus. Each accepted point has a payload matching its scan-plan dx/dy injection, refitted four-station tracklets with positive-definite covariance, and finite successful mode-0 Acts coverage of all six forward station pairs. The train-then-validation pooled materialization produced six samples per split; each train/validation payload pools exactly 994/796 physical source events, respectively, and all candidate exports report `q_over_p_mode=0` with zero target-z mismatch.

The full nominal refitted-state audit is stored in `outputs/mc24_v3_source_audit_v1/refitted_source_diversity_expanded_trainval.json`. Across the 18 sources it observes 1,790 events, 6,690 tracklets, and 6,488 truth-matched muon tracklets, with both truth PDG signs present. The observed state envelope is `x=[-123.13, 123.94] mm`, `y=[-121.99, 125.97] mm`, `tx=[-0.1905, 0.2244]`, and `ty=[-0.0991, 0.1617]`. This establishes charge and refitted state diversity for the admitted corpus, but does not by itself claim exhaustive physics-production coverage.

One independent train and one validation source completed the real nominal smoke test. They exported 100/99 events and 373/371 tracklets, respectively, with positive-definite covariance for every exported tracklet and mode-0 propagation records for every station pair. `audit_refitted_source_diversity.py` measures this from refitted tracklets rather than filenames: the first two sources contain 360/364 truth-matched muon tracklets and show observed `x/y` coverage of roughly `[-115, 120] mm` and `[-104, 123] mm`, with nonzero `tx/ty` spreads. Remaining sources and payload points are not admitted to training until their own output audits pass.

The additional nominal train source `mc24_100043_00200_00299` confirms the mode choice with a paired propagation-component audit. Across 505 truth-matched records present in both modes, the median chi2 changes from `5.9670` (mode 0) to `19.9836` (mode 1). Holding the mode-0 combined covariance fixed while replacing only the residual has median effect `-2.56e-05`; replacing the covariance at the mode-1 residual contributes `6.4121`, while the median combined-covariance log-determinant change is `-14.0282`. Thus this increase is principally uncertainty narrowing rather than raw-residual degradation. V3 continues to use only mode 0; the mode-1 fixed q/p seed is not treated as a reliable local momentum measurement.

Production filenames alone still do not establish all angular, position, or conditions diversity. The refitted-state audit is the acceptance evidence for observed coverage; a separate Calypso/truth audit remains required before claiming broader production coverage.

The first EOSSubmit cluster `990949` exited before any payload work because the staged worker inferred its project root from Condor scratch space. The next cluster `990951` exposed a second startup-only issue: its strict nounset shell option was inherited by the external LCG setup script. The worker now receives the EOS project root explicitly and disables nounset only while sourcing that setup script. The active corrected submission is Condor cluster `990952`, with 18 workers and six real payload magnitudes per source. Workers never write the shared corpus manifest; a single post-job aggregation refreshes provenance after output audits pass. A physical point is now admitted only when its persisted `/Tracker/Align` payload matches the planned per-station injection, it has refitted tracklets and no `failure.json`, and its `content_audit.json` confirms MC labels, all four stations, and positive-definite covariance for every exported tracklet. Its propagation ROOT must also contain finite, successful, covariance-bearing mode-0 records for every configured forward station pair.

## Reproduction

```bash
cd /eos/home-x/xcheng/FASER/alignment_ML
source scripts/setup_environment.sh ml

# Prepare or refresh source-local configs and provenance. No test source is used.
python scripts/build_physical_curriculum_corpus.py \
  --config configs/physical_curriculum_v3_expanded_trainval.yaml \
  --output-dir outputs/mc24_v3_expanded_trainval_physical_v1 \
  --prepare-only --resume

# Generate EOSSubmit-compatible workers. Add --submit to submit the bank.
python scripts/submit_physical_curriculum_condor.py \
  --config configs/physical_curriculum_v3_expanded_trainval.yaml \
  --physical-output-dir outputs/mc24_v3_expanded_trainval_physical_v1 \
  --submit-dir outputs/NEW_CONDOR_SUBMISSION \
  --skip-complete

# After every required physical point passes, refresh then materialize each split.
python scripts/materialize_pooled_curriculum_synthetics.py \
  --physical-manifest outputs/mc24_v3_expanded_trainval_physical_v1/physical_corpus_manifest.json \
  --config configs/physical_curriculum_v3_expanded_trainval.yaml \
  --output-dir outputs/NEW_V3_SYNTHETIC --split train
python scripts/materialize_pooled_curriculum_synthetics.py \
  --physical-manifest outputs/mc24_v3_expanded_trainval_physical_v1/physical_corpus_manifest.json \
  --config configs/physical_curriculum_v3_expanded_trainval.yaml \
  --output-dir outputs/NEW_V3_SYNTHETIC --split validation --resume

The resume contract replaces only the explicitly requested split and preserves
the already materialized split entries. It rejects a manifest from another
physical corpus and duplicate `(split, payload_id)` entries, so the second
validation-only invocation cannot erase train provenance.

python scripts/train_structured_assignment_v3.py \
  --synthetic-manifest outputs/NEW_V3_SYNTHETIC/synthetic_corpus_manifest.json \
  --config configs/geometry_aware_transformer_v3.yaml \
  --output-dir outputs/NEW_V3_TRAIN_VALIDATION
python scripts/evaluate_structured_assignment_v3.py \
  --synthetic-manifest outputs/NEW_V3_SYNTHETIC/synthetic_corpus_manifest.json \
  --training-dir outputs/NEW_V3_TRAIN_VALIDATION \
  --config configs/geometry_aware_transformer_v3.yaml \
  --output-dir outputs/NEW_V3_VALIDATION
```

For matched MLP/V1 controls after the same synthetic manifest is complete:

```bash
python scripts/run_global_assignment_mlp_baseline.py \
  --synthetic-manifest outputs/NEW_V3_SYNTHETIC/synthetic_corpus_manifest.json \
  --config configs/physical_global_assignment_mlp_v3_expanded_control.yaml \
  --output-dir outputs/NEW_EXPANDED_MLP_VALIDATION --validation-only

python scripts/evaluate_pairwise_mlp_route_validation.py \
  --synthetic-manifest outputs/NEW_V3_SYNTHETIC/synthetic_corpus_manifest.json \
  --checkpoint outputs/NEW_EXPANDED_MLP_VALIDATION/mlp_pair_classifier.pt \
  --frozen-calibration outputs/NEW_EXPANDED_MLP_VALIDATION/route_calibration.json \
  --config configs/pairwise_mlp_route_expanded_control.yaml \
  --output-dir outputs/NEW_EXPANDED_MLP_ROUTE_VALIDATION --device cuda

python scripts/train_geometry_aware_transformer_v1.py \
  --synthetic-manifest outputs/NEW_V3_SYNTHETIC/synthetic_corpus_manifest.json \
  --config configs/geometry_aware_transformer_v1_expanded_control.yaml \
  --output-dir outputs/NEW_EXPANDED_V1_VALIDATION

python scripts/train_route_aware_transformer_v2.py \
  --synthetic-manifest outputs/NEW_V3_SYNTHETIC/synthetic_corpus_manifest.json \
  --config configs/geometry_aware_transformer_v2_expanded_bce_control.yaml \
  --output-dir outputs/NEW_EXPANDED_V2_BCE_VALIDATION
python scripts/evaluate_route_aware_transformer_v2_direct_routes.py \
  --synthetic-manifest outputs/NEW_V3_SYNTHETIC/synthetic_corpus_manifest.json \
  --v2-validation-dir outputs/NEW_EXPANDED_V2_BCE_VALIDATION \
  --config configs/geometry_aware_transformer_v2_expanded_direct_routes.yaml \
  --output-dir outputs/NEW_EXPANDED_V2_BCE_ROUTE_VALIDATION --device cuda
```

The train/evaluation commands require a completed manifest containing only the declared train and validation inputs. They reject test loading by contract.

## Completed Expanded-Corpus Validation

All results below use the same physical mode-0 candidate graph, original-xAOD-file-disjoint source split, and validation-only calibration/control selection. No result opened a test event or artifact. The primary capture gate is `efficiency >= 0.70`, `purity >= 0.95`, and `fake rate <= 0.05`.

| control | nominal eff / purity / fake | 5 mm eff / purity / fake | 10 mm eff / purity / fake | captured magnitudes [mm] |
| --- | --- | --- | --- | --- |
| pairwise MLP + route solver | 0.9094 / 0.9724 / 0.0629 | 0.0000 / 0.0000 / 0.7481 | 0.0000 / 0.0000 / 0.9772 | none |
| V1 ordinary sparse | 0.9637 / 0.9708 / 0.0350 | 0.9125 / 0.9464 / 0.0586 | 0.2980 / 0.7934 / 0.1217 | 0, 0.1, 1 |
| V1 geometry-aware | 0.9547 / 0.9664 / 0.0432 | 0.8833 / 0.9484 / 0.0593 | 0.2802 / 0.7761 / 0.1200 | 0, 0.1, 1 |
| V2 BCE route query | 0.9163 / 0.9667 / 0.0443 | 0.7559 / 0.9462 / 0.0725 | 0.2360 / 0.7935 / 0.1446 | 0, 0.1, 1 |
| V3 exact structured margin | 0.7988 / 0.8639 / 0.2523 | 0.5450 / 0.8294 / 0.2638 | 0.2188 / 0.5953 / 0.3218 | none |
| V3 exact margin + soft-assignment control | 0.7335 / 0.8509 / 0.2779 | 0.3884 / 0.7256 / 0.3141 | 0.1546 / 0.5405 / 0.3810 | none |

The expanded V2 control passes nominal validation but fails at 5 mm because both purity and fake rate miss the fixed gate. Neither V3 variant passes nominal validation. For the exact-margin V3, raw physical candidate complete-truth-chain recall remains `1.0` at every magnitude, while its score-threshold retention is `0.8952` at nominal, `0.9110` at 5 mm, and `0.9014` at 10 mm. Its route utility increases nominal complete-track efficiency from the same-checkpoint edge-only control's `0.4460` to `0.7988`, but introduces too many fake routes. The soft control is worse on both retention and route quality. This identifies score calibration/selectivity and structured-loss optimization, rather than loss of physical candidate truth chains, as the current V3 limitation.

The primary exact-margin checkpoint selected validation epoch 2; the soft-control checkpoint selected epoch 1. The primary run skips the optional soft surrogate entirely at zero weight, while the labelled control enables it. Batch size is 16 for throughput on the available GPU; the encoder and event-level objective are otherwise unchanged. The exact oracle audit now reports a correctly serialized float maximum (`29.93 ms`) and a mean of `4.61 ms/event` on its deterministic 64-event train-only sample.

`scripts/plot_expanded_validation_route_controls.py` and `configs/expanded_validation_route_controls.yaml` reproduce the validation-only MLP/V2/V3 comparison figure and CSV. The generated artifact is `outputs/mc24_v3_expanded_trainval_validation_control_plots_v3/route_validation_control_comparison.png`.

The V3 validation hypothesis is therefore not frozen as successful, and no new multidirection test bank has been generated or opened. The next mechanism work must remain train/validation-only and target route-score selectivity or the structured objective before any test decision.
