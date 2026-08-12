# Physical-Payload Synthetic Unknown-Association Baselines

This controlled study begins only after the complete truth-fixed physical capture scan. It provides the two required non-Transformer association references: conventional field-aware chi-square matching and a small MLP pair classifier trained at nominal geometry.

## Physical Input Contract

For each point in the physical scan, the source is that point's independently refitted canonical tracklet file and its mode-0 `FaserActsExtrapolationTool` propagation export:

```text
point-specific /Tracker/Align payload
  -> SCT_ClusterContainer -> SegmentFitRefit -> SegmentsRefit
  -> tracklets.root + mode-0 propagations.root
  -> deterministic synthetic multi-track overlay
  -> field-aware candidate records
  -> unknown-association baseline evaluation
```

The overlay stores `origin_run_id`, `origin_event_id`, and `origin_tracklet_id` for every synthetic tracklet, including copied fake rows. A candidate prediction is copied only from the exact mode-0 Acts record whose source provenance and target station agree, and only after the exported target z matches the synthetic target z. Event overlay therefore does not replace propagation with a coordinate shift or a straight-line surrogate. The candidate ROOT tree deliberately sets its propagation truth field to `-1`; truth is unavailable to candidate construction and matching.

The current MC24 control has common mode-0 station reference planes, so this fan-out is valid. A target-z mismatch is counted and omitted rather than being silently approximated.

## Synthetic Sample And Split

`configs/synthetic_unknown_association_muon.yaml` specifies 100 deterministic synthetic events with four overlaid source tracks per event, a 0.10 true tracklet-missing probability, and a Poisson mean of 0.50 fake rows per station. The identical RNG seed and overlay layout are used for every physical payload point.

The MLP is trained once on the nominal-payload synthetic sample. Its deterministic 70/30 event split is written to `heldout_event_ids.json`. Both the field-chi2 baseline and the frozen MLP are evaluated only on that same held-out event subset for every injected geometry. This avoids using nominal MLP training events as its reported baseline performance.

The source control currently contains only four eligible complete muon tracks from five xAOD events. Overlay reuses those states by design, so these curves measure controlled geometry robustness, not independent particle-level generalization.

## Baselines

The field-chi2 reference considers all ordered forward station pairs and applies conventional ascending-chi2 greedy one-to-one assignment per station pair. Its fixed `1500` gate is configured before displaced-point evaluation; the nominal truth-chi2 audit is persisted with the output.

The MLP has two ReLU hidden layers of width 64. Inputs are the field-aware residual and pull components, `log1p(chi2)`, combined-covariance log determinant, delta-z, local fit quality, hit counts, and station-pair one-hot features. It does not consume truth, q/p, or injected alignment values. A validation-candidate F1 sweep fixes its output threshold, which functions as the unmatched decision for this baseline.

`q_over_p_mode=0` is mandatory throughout. Segment q/p is a seed rather than a dependable local measurement, so no q/p feature is used.

## Metrics

For every geometry point and station pair, outputs retain candidate counts, truth-possible matches, predicted matches, and correct matches. The scan plots and compares:

- association efficiency: correct / possible truth matches;
- inclusive association purity: correct / all predictions;
- inclusive fake rate: non-correct / all predictions.

The inclusive quantities count endpoints labelled as synthetic fake or unknown as false predictions. Legacy scorable-only purity and fake rate are also preserved in each point's metrics JSON for diagnostic continuity.

## Current Held-Out Result

The completed result is in
`outputs/mc24_muon_fasernu_physical_unknown_association_v2_heldout/`. Its
30 held-out synthetic events contain 572 truth-possible pair matches. At zero
offset, field-chi2 has efficiency/purity/inclusive-fake-rate of
`0.834/0.796/0.204`; the nominal MLP has `0.892/0.875/0.125`.

At `10 mm`, mean efficiency is `0.812` for field-chi2 and `0.449` for the
frozen MLP. The MLP's apparently stable purity at this scale reflects its
nominally tuned threshold rejecting many candidates, not robust recall. At
`50 mm`, the mean efficiencies are `0.280` and `0.134`; at `250 mm` they are
`0.075` and `0.017`. At `500 mm`, MLP efficiency is zero in all directions;
at `1000 mm`, it produces no matches. The chi-square purity at extreme offsets
is survivor-biased because its fixed gate retains very few candidates.

This evidence does not justify starting a Transformer. The next association
step is misalignment augmentation and threshold/calibration studies using a
larger, source-event-independent MC sample, followed by an alignment-aware
candidate-gating comparison.

## Reproduction

Run only after every physical point has a completed `closure.json`:

```bash
cd /eos/home-x/xcheng/FASER
source alignment_ML/scripts/setup_environment.sh ml
python3 alignment_ML/scripts/run_synthetic_unknown_association_scan.py \
  --physical-scan-root alignment_ML/outputs/mc24_muon_fasernu_physical_capture_scan_v1 \
  --output-dir alignment_ML/outputs/mc24_muon_fasernu_physical_unknown_association_v1 \
  --config alignment_ML/configs/synthetic_unknown_association_muon.yaml
```

Use `--resume` only with the same physical scan and configuration. The output contains each point's synthetic ROOT files, provenance and target-z audit, baseline logs and metrics, the nominal MLP checkpoint, held-out IDs, `unknown_association_scan_points.csv`, and `unknown_association_scan.png`.
