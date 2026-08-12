# Geometry-Aware Sparse Transformer V1

## Scope and frozen input contract

This is the first Transformer study after the physical four-station route
baseline. It does not change the data definition, propagation, candidate
graph, route solver, or source split used by that baseline.

Every tracklet and candidate originates from the already materialized physical
chain:

```text
SCT_ClusterContainer -> SegmentFitRefit -> SegmentsRefit -> NtupleDumper
-> FaserActsExtrapolationTool (mode 0)
```

The candidate graph is the ungated set of existing mode-0 Acts candidates for
all six forward station pairs. Local segment `q/p` remains only a propagation
seed. No coordinate shift, cached residual, residual-level surrogate,
retracking, or test-time candidate construction is used.

The original-xAOD-file split is fixed at 10 train, 2 validation, and 2 test
sources (99, 19, and 19 source events). The three split source-event UID sets
are disjoint. The sealed test scan uses the same controlled physical
multi-direction manifest as the frozen MLP route baseline: one `0 mm` trial
and three independent `dx/dy` direction trials at each nonzero magnitude.

## V1 model

`TrackletStateEncoder` consumes the persisted state, covariance diagonal, fit
quality, hit count, six layer-side occupancy bits, and a learnable station
embedding. Its output dimension is 128. The sparse network has exactly four
pre-normalized blocks, eight heads, and an FFN `128 -> 256 -> 128`.

Attention is evaluated only on existing physical candidate edges (both message
directions of an already existing forward candidate):

```text
QK / sqrt(d) + MLP(g_ij) - lambda * chi2_ij / (2 * tau)
```

The geometric vector is `[rx, ry, rtx, rty, pull_x, pull_y, pull_tx, pull_ty,
log1p(chi2), logdet(S), dz]`, supplemented in the attention MLP by
station-pair and message-direction embeddings. The raw chi-square term is
independently switchable. The edge scorer emits only `0->1`, `1->2`, and
`2->3` probabilities; the existing adjacent unit-capacity route assignment
with dustbins remains the only matching backend.

For the three geometry-aware configurations, a small local edge residual head
was trained first on the same physical features and train split, then retained
while the sparse context stack was trained. It never receives truth, a prior
MLP score, or synthetic-role information. This makes the `no multi-station
context` variant the controlled test of cross-station messages. The ordinary
sparse control deliberately disables both geometric attention bias and this
local residual decoder, so it is a non-geometric reference rather than a
one-variable-only comparison to the full model.

The loss is weighted BCE (`focal_gamma=0`) with event-graph-uniform sampling,
class-imbalance positive weighting capped at 30, and a factor-two hard-negative
weight. The local stage follows the physical `0/0.1 -> 5 -> 50 mm` curriculum
for 25/30/35 epochs; sparse-context training follows the same curriculum for
8/10/12 epochs. CUDA was used for model training.

## Validation-only model selection

All feature standardizers are fit on train graphs only. The two geometry
chi-square candidates (`lambda=1`, `tau=0.5e6` and `2.0e6`) were selected by
validation candidate AP. Per-station-pair monotonic Platt calibration and the
route thresholds/dustbin penalty were fitted on validation only. A
deterministic event subset proposed finite controls, and the top 24 proposals
were reranked over every validation event before the operating point was
frozen.

| Variant | Raw validation AP | Raw validation AUC | Selected nominal efficiency / purity / fake rate |
| --- | ---: | ---: | ---: |
| Ordinary sparse Transformer | 0.8153 | 0.9304 | 0.570 / 0.910 / 0.161 |
| Geometry-aware | 0.8726 | 0.9429 | 0.753 / 0.940 / 0.088 |
| Geometry-aware, no raw chi2 term | 0.8726 | 0.9430 | 0.744 / 0.940 / 0.087 |
| Geometry-aware, no multi-station context | 0.8716 | 0.9432 | 0.655 / 0.891 / 0.118 |

The predeclared primary point is efficiency `>= 0.70`, purity `>= 0.95`, and
inclusive fake rate `<= 0.05`. No Transformer variant passed it on validation.
This result was retained; test was not used to alter the choice.

## One sealed multi-direction test scan

The scan output is
`outputs/geometry_aware_transformer_v1f_final_multidirection_test_v1/`.
Its frozen contract records the test-manifest hash, every checkpoint and
validation-artifact hash, strict test-source membership, and
`no_test_time_model_selection`, `no_test_time_calibration`, and
`no_test_time_threshold_selection` all set to `true`.

Raw complete truth-chain candidate recall is 1.0 at every magnitude for every
method because all methods consume the same physical candidate graph. The
following table gives capture successes over direction trials:

| Method | 0 mm | 0.1 mm | 1 mm | 5 mm | 10 mm | 50 mm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Frozen pairwise MLP route baseline | 1/1 | 3/3 | 2/3 | 0/3 | 0/3 | 0/3 |
| Ordinary sparse Transformer | 0/1 | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 |
| Geometry-aware Transformer | 1/1 | 1/3 | 2/3 | 0/3 | 0/3 | 0/3 |
| Geometry-aware, no raw chi2 term | 1/1 | 1/3 | 2/3 | 0/3 | 0/3 | 0/3 |
| Geometry-aware, no multi-station context | 1/1 | 3/3 | 2/3 | 0/3 | 0/3 | 0/3 |

For the full geometry-aware model, the direction-mean route metrics at `5 mm`
are threshold-chain recall `0.901`, complete-track efficiency `0.705 +/- 0.021`,
purity `0.985 +/- 0.003`, fake rate `0.065 +/- 0.005`, and missing recovery
`0.973 +/- 0.004`. At `10 mm` they are `0.884`, `0.632 +/- 0.085`,
`0.974 +/- 0.005`, `0.069 +/- 0.017`, and `0.961 +/- 0.004`, respectively.
Thus its score retention and efficiency improve over the MLP at these points,
but the fixed fake-rate requirement still fails; no capture-range extension is
established. At `50 mm` no complete route is selected by the full model.

The no-multi-station-context ablation is stronger than the full-context model
on the sealed test at small and intermediate offsets. Consequently this V1 run
does not support the claim that the present sparse multi-station context is the
mechanism extending association capture range. Removing the explicit raw
chi-square term changes the results only marginally at the validation-selected
`tau=2.0e6` scale.

Detailed route and station-pair breakdowns are retained without aggregation
loss in each model directory:

- `trial_metrics.csv`: raw/threshold truth-chain recall, complete efficiency,
  purity, fake/duplicate rate, and missing recovery for each direction.
- `station_pair_trial_metrics.csv`: candidate recall, AUC/AP, calibration,
  efficiency/purity/fake rate, and unmatched metrics for `0->1`, `1->2`, and
  `2->3`.
- `magnitude_summary.csv`: direction mean, population spread, pooled counts,
  and capture fraction.
- `comparison_magnitude_summary.csv` and the two PNG figures: common-method
  comparison against the frozen MLP baseline.

## Conclusion and next experiment boundary

V1 is a valid negative result. It improves candidate-level AP relative to the
ordinary sparse control and preserves all raw physical truth edges, but it does
not retain the predeclared route operating point at `5`, `10`, or `50 mm`. It
therefore does not demonstrate a larger association or alignment capture range
than the frozen pairwise MLP baseline.

The sealed test output must not be reused for threshold changes, calibration,
architecture selection, or additional ablation selection. Any V2 work should
be designed and selected on train/validation first, then use a newly reserved
test source or an explicitly independent test bank. The immediate diagnostic
question is why the full message-passing context underperforms its local
no-context ablation, rather than adding a larger Transformer.

## Reproduction

```bash
cd /eos/home-x/xcheng/FASER/alignment_ML
source scripts/setup_environment.sh ml

# Train and freeze only train/validation artifacts. Use a new output path.
python scripts/train_geometry_aware_transformer_v1.py \
  --synthetic-manifest outputs/mc24_muon_2dfluka_curriculum_external_full_train_aug_synthetic_v1/synthetic_corpus_manifest.json \
  --config configs/geometry_aware_transformer_v1f_final_ablations.yaml \
  --output-dir outputs/NEW_TRANSFORMER_VALIDATION_RUN

# Only after validation is frozen, run the test-only evaluator once.
python scripts/run_frozen_geometry_aware_transformer_v1_scan.py \
  --synthetic-manifest outputs/mc24_muon_2dfluka_multidirection_test_synthetic_controlled_v2/synthetic_corpus_manifest.json \
  --validation-output-dir outputs/NEW_TRANSFORMER_VALIDATION_RUN \
  --config configs/frozen_geometry_aware_transformer_v1_multidirection_test.yaml \
  --output-dir outputs/NEW_TRANSFORMER_TEST_RUN
```

