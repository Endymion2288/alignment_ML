# Geometry-Aware Transformer V2: V1 Mechanism Diagnosis

## Sealed boundary

The V1 test sources remain permanently sealed. This diagnosis opened only the
two validation source files in the existing physical curriculum:
`mc24_100116_00010_00019` and `mc24_100117_00010_00019`.

`diagnostic_contract.json` records `loaded_splits=["validation"]`,
`forbidden_splits=["test"]`, `test_events_loaded=false`, and
`test_artifacts_opened=false`. It uses the unchanged physical chain and
candidate graph:

```text
SCT_ClusterContainer -> SegmentFitRefit -> SegmentsRefit -> NtupleDumper
-> FaserActsExtrapolationTool (mode 0)
```

The validation graph contains 1,440 full events, 18,669 nodes, 175,084
directed physical message edges, and 44,921 adjacent output edges (10,511
truth-positive). The frozen V1 full-context checkpoint, Platt calibration, and
route controls (`0->1=0.001`, `1->2=0.5`, `2->3=0.05`, dustbin penalty `1.0`)
were reused without refitting.

## Interventions

The following validation-only interventions change only the messages executed
by the frozen V1 model. Output candidate edges, physical residual/covariance
features, calibration, thresholds, and the route assignment backend do not
change:

- message depths 0, 1, 2, 3, and 4;
- four-layer adjacent-only, forward-only, and backward-only messages;
- four leave-one-station-out message interventions.

The detailed evidence is retained in
`outputs/geometry_aware_transformer_v2_mechanism_diagnostics_validation_v1/`:
`layer_station_pair_edge_metrics.csv`, `node_state_depth_summary.csv`,
`attention_by_layer_station_pair.csv`, `score_drift_from_full_context.csv`,
`fake_route_contributions.csv`, and frozen route metric tables.

## Findings

The score ranking is nearly unchanged by message depth. Relative to the
four-layer full graph, depth-zero local scoring has mean absolute probability
drift `0.0135` and rank correlation `0.99937`; depth three has drift `0.00435`
and correlation `0.99989`. Adjacent edge AP changes only marginally: at depth
zero, `0->1/1->2/2->3` AP is `0.8433/0.8657/0.8962`; at full depth it is
`0.8405/0.8666/0.8983`.

Meanwhile, representation mixing is substantial. Across all nodes, the mean
within-station cosine similarity rises from `0.319` at depth zero to `0.549`
at depth four, and mean state norm rises from `8.01` to `21.25`. This is
consistent with over-smoothing/scale growth while the local residual edge
decoder continues to dominate the ranking.

The route-level failure is not explained by raw candidate loss: thresholded
truth-chain recall remains around `0.825` at 5 mm for local and full messages.
Instead, full context does not suppress false routes enough. With the frozen
V1 controls at 10 mm, full depth has complete efficiency `0.573`, purity
`0.913`, and fake rate `0.129`; selected false routes include both fake
endpoints and mixed-truth chains. Direction-only message policies can change
efficiency, but they trade it against fake rate rather than reaching the fixed
primary point.

Thus the evidence supports a targeted V2 hypothesis: retain the physical
candidate graph and V1 backbone, but supervise complete route consistency and
endpoint competition directly rather than expecting independent edge scores
and generic message passing to solve four-station combinatorics.

## V2 pre-test design

V2 creates an explicit route candidate only when all three existing adjacent
physical edges form `IFT -> S1 -> S2 -> S3`. A route query aggregates the four
node states, three physical edge descriptors, station-pair embeddings, and
base edge evidence. It produces a route logit and a correction projected back
onto the same adjacent edge rows consumed by the unchanged unit-capacity route
assignment.

The training loss is the declared sum of weighted edge BCE/focal loss,
route-truth consistency BCE/focal loss, endpoint-incidence log-sum-exp
one-to-one competition, and a fake-endpoint route softplus penalty. Truth IDs
and synthetic roles are labels/loss audit fields only, never network inputs.

V2 will use train/validation only. Its same-checkpoint control disables only
the route-derived edge correction while reusing the V2 calibration and route
operating point; it receives no independent threshold or calibration search.
No new test bank will be generated unless the declared validation criterion is
met first.

## Reproduction

```bash
cd /eos/home-x/xcheng/FASER/alignment_ML
source scripts/setup_environment.sh ml

python scripts/diagnose_geometry_aware_transformer_v1_context.py \
  --synthetic-manifest outputs/mc24_muon_2dfluka_curriculum_external_full_train_aug_synthetic_v1/synthetic_corpus_manifest.json \
  --v1-validation-dir outputs/geometry_aware_transformer_v1f_final_validation_v1 \
  --output-dir outputs/geometry_aware_transformer_v2_mechanism_diagnostics_validation_v1
```
