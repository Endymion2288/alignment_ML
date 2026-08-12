# Geometry-Aware Transformer V2: Route-Aware Validation Study

## Decision

V1's existing test sources are sealed permanently. V2 architecture, losses,
calibration, thresholds, and route-utility controls used only the source-file
disjoint train and validation splits. No new test bank was generated and no
test event or test artifact was opened.

The V2 validation hypothesis did not pass. The selected route-context weight
is `0.0`, which is exactly the frozen V2 edge-level route-assignment control.
Therefore this study does not establish a route-aware gain, a larger
association capture range, or a justification to run a new final test.

## Immutable physical input

All V2 inputs continue to originate from the real displaced-geometry chain:

```text
SCT_ClusterContainer -> SegmentFitRefit -> SegmentsRefit -> NtupleDumper
-> FaserActsExtrapolationTool (mode 0)
```

The model uses the ungated existing mode-0 Acts candidate graph. It does not
make coordinate-level surrogates, regenerate residuals, create a new edge, or
use local segment `q/p` as a measurement. Train and validation source files
remain strictly disjoint; the manifest loader is called with explicit
`allowed_splits`, so excluded test asset paths are not resolved.

The V2 run contracts record `test_events_loaded=false` and
`test_artifacts_opened=false` in:

- `outputs/geometry_aware_transformer_v2_validation_v1/`
- `outputs/geometry_aware_transformer_v2_route_residual_utility_validation_v1/`
- `outputs/geometry_aware_transformer_v2_validation_assessment_v1/`

## Why V1 full context was not better

The preceding validation-only mechanism diagnosis is documented in
[the V2 diagnostic note](geometry_aware_transformer_v2_diagnostics.md). It
used 1,440 validation events, 18,669 nodes, 175,084 directed physical message
edges, and 44,921 adjacent output edges.

The message-depth, hop-direction, adjacent-only, and leave-one-station-out
interventions show that the frozen V1 edge ranking barely changes with global
messages. Depth-zero to four-layer full-message score drift is `0.01350` MAE
with rank correlation `0.99937`; depth-three to full is `0.00435` with
correlation `0.99989`. Edge AP changes only slightly:

| Station pair | Local depth 0 AP | Full depth 4 AP |
| --- | ---: | ---: |
| `0->1` | 0.8433 | 0.8405 |
| `1->2` | 0.8657 | 0.8666 |
| `2->3` | 0.8962 | 0.8983 |

In contrast, the node-state mean within-station cosine similarity rises from
`0.319` to `0.549`, and mean norm rises from `8.01` to `21.25`. This supports
representation mixing/over-smoothing while the local residual edge decoder
still determines most of the ranking. At 10 mm, selected false routes contain
both fake endpoints and mixed-truth chains; removing or restricting message
directions trades efficiency against fake rate rather than reaching the fixed
operating point.

## V2 route-aware model

`models/route_transformer.py` retains the fixed V1 backbone: model width 128,
four sparse blocks, eight heads, and FFN `128 -> 256 -> 128`. A route query is
constructed only for an existing complete physical chain
`IFT -> S1 -> S2 -> S3`. It aggregates:

- the ordered four encoded node states;
- the three existing adjacent physical edge feature vectors;
- station-pair embeddings; and
- the three backbone edge logits.

The candidate bank is physical and truth-blind. It contains 800,228 complete
routes in 7,200 train graphs and 148,551 in 1,440 validation graphs. The
validation bank has 2,832 truth-consistent routes, 71,756 fake-endpoint routes,
and 45,142 hard-negative routes. Truth particle IDs and synthetic roles are
used only as supervision/audit labels.

The declared training loss is the sum of weighted focal edge BCE, route-truth
consistency focal BCE, endpoint-incidence log-sum-exp competition, and a
fake-endpoint softplus penalty. The route query does not replace the
unit-capacity route assignment backend.

## Direct query diagnosis and residual utility

The first direct-query integration replaced the sum of three edge log-odds by
the calibrated complete-route log-odds. It was structurally inconsistent with
the solver: a complete route then competes with partial routes that retain the
three strong individual edge utilities. Across its validation grid, no complete
route was selected. This is a route-utility formulation failure, not a loss of
physical truth candidates.

The route query has nontrivial validation discrimination before this solver
integration: raw complete-route AP is `0.4095` and ROC AUC is `0.9449` over
148,551 physical routes. Validation-only Platt calibration reduces ECE from
`0.0177` to `0.00312`. It does not by itself make the score compatible with
the partial-route objective.

`baselines/route_assignment.py` now supports a residual composition for a
complete route:

```text
L = L_edge + w * (L_route - L_edge)
L_edge = sum(logit(p_edge))
L_route = logit(p_route)
```

`w=0` exactly recovers the edge-only route solver and `w=1` recovers direct
replacement. Partial routes always retain their existing physical edge
utility, so missing-station/dustbin recovery remains available. The residual
scan leaves every existing complete physical route eligible
(`complete_route_score_threshold=0`) and selects only `w` and the shared
dustbin utility on validation. The frozen V2 adjacent edge calibration and
thresholds (`0->1=0.001`, `1->2=0.05`, `2->3=0.05`) are not refit.

## Validation result

The residual grid is
`w in {0, 0.025, 0.05, 0.10, 0.20, 0.40, 0.70, 1.0}` and dustbin penalty in
`{-0.5, 0, 0.5, 1.0}`. Its validation-selected point is `w=0`, dustbin penalty
`1.0`; every nonzero route-context weight reduces the route efficiency. The
same-checkpoint edge-only control is numerically identical at `w=0`.

| Injected offset | Raw truth-chain recall | Edge-threshold chain recall | Complete efficiency | Complete purity | Fake rate |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 mm | 1.0000 | 0.9978 | 0.5293 | 0.8561 | 0.2183 |
| 5 mm | 1.0000 | 0.9872 | 0.5556 | 0.8754 | 0.2245 |
| 10 mm | 1.0000 | 0.9782 | 0.4858 | 0.8610 | 0.2356 |
| 50 mm | 1.0000 | 0.6253 | 0.0683 | 0.6600 | 0.3435 |

Thus the primary failure is downstream of the raw candidate graph. At 5 and
10 mm, most physical truth chains remain candidate and edge-score reachable,
but route-level false/mixed/fake competition prevents the required operating
point.

The predeclared validation gate is nominal efficiency `>=0.70`, purity
`>=0.95`, fake rate `<=0.05`, plus a material 5/10 mm gain over the pairwise
MLP, V1 full-context, V1 no-context, and the same-checkpoint edge control.
The post-selection assessment records all gate flags as false. In particular,
the route-aware stream has no gain over its exact same-checkpoint control.

## Consequence

No test action is allowed from this result. The existing V1 test is not used
to select a route weight, threshold, calibration, loss, or architecture, and
no new source-disjoint test bank is created.

A subsequent V2 proposal must be selected anew on train/validation. A
well-motivated next change is to train a route logit as an explicit residual
around the edge-route utility and to include an event-level assignment
surrogate in training, rather than learning an absolute rare-route probability
and blending it only after training. That is a new validation hypothesis, not
a reason to reopen the sealed test.

## Reproduction

```bash
cd /eos/home-x/xcheng/FASER/alignment_ML
source scripts/setup_environment.sh ml

# Train V2 only on train/validation physical payloads.
python scripts/train_route_aware_transformer_v2.py \
  --synthetic-manifest outputs/mc24_muon_2dfluka_curriculum_external_full_train_aug_synthetic_v1/synthetic_corpus_manifest.json \
  --config configs/geometry_aware_transformer_v2.yaml \
  --output-dir outputs/NEW_V2_VALIDATION

# Evaluate the route-query residual only on validation, reusing frozen V2 edge controls.
python scripts/evaluate_route_aware_transformer_v2_direct_routes.py \
  --synthetic-manifest outputs/mc24_muon_2dfluka_curriculum_external_full_train_aug_synthetic_v1/synthetic_corpus_manifest.json \
  --v2-validation-dir outputs/NEW_V2_VALIDATION \
  --config configs/geometry_aware_transformer_v2_route_residual_utility.yaml \
  --output-dir outputs/NEW_V2_ROUTE_RESIDUAL_VALIDATION \
  --device cuda

# Compare validation artifacts only. This command opens no event data.
python scripts/assess_route_aware_transformer_v2_validation.py \
  --v2-output outputs/NEW_V2_VALIDATION \
  --direct-route-output outputs/NEW_V2_ROUTE_RESIDUAL_VALIDATION \
  --mlp-route-output outputs/pairwise_mlp_route_validation_v2_control_v1 \
  --v1-validation-dir outputs/geometry_aware_transformer_v1f_final_validation_v1 \
  --output-dir outputs/NEW_V2_VALIDATION_ASSESSMENT
```
