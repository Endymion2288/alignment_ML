# Physical Misalignment-Augmentation MLP Baseline

## Scope

This study remains deliberately pre-Transformer.  It measures the robustness
ceiling of a small pairwise MLP after training on real displaced-geometry
refits.  `q_over_p_mode=0` is mandatory: local segment `q/p` is a seed, not a
trusted measurement.

Every payload point follows this chain:

```text
persisted SCT_ClusterContainer
  -> SegmentFitRefit -> SegmentsRefit -> NtupleDumper
  -> canonical tracklets + FaserActs mode-0 propagations
  -> synthetic overlay + physical candidate fanout
```

The candidate exporter copies an Acts prediction only from the same payload
and only after exact target-reference-z validation.  Coordinate shifts and
cached residual substitutions are prohibited.

## Source Split Contract

`configs/physical_curriculum_mlp_muon.yaml` assigns MC24 `s0012-r0019` xAOD
chunks to train, validation and test.  Splitting occurs by original input
file before overlays are generated.  The manifest records every original event
as `source_id:run_id:event_id`; source IDs protect against accidental reuse or
run/event-number collisions across xAOD chunks.  Each payload also records
its actually refitted event membership. A payload may lose an event through
physical reconstruction, but it may never introduce one from another split.

The initial bank uses five input chunks: three train, one validation and one
test.  Geometry direction trials are seeded and held out by split, in addition
to the strict source-event split.

## Payload Bank and Curriculum

Physical station-level `dx/dy` payload magnitudes are
`0, 0.1, 1, 5, 10, 50 mm`.  Nonzero directions are deterministic draws saved
in the source scan configurations.  At training time the MLP sees only actual
refit samples and progresses through:

1. `0` to `0.1 mm`
2. through `5 mm`
3. through `50 mm`

All stages retain the same feature standardizer fitted on train-split rows
only, and continue the prior stage weights.  Within each stage, sampling is
uniform over physical `(source_id, payload_id)` groups rather than candidate
rows, so wide-gate or fake-rich payloads cannot dominate exposure.  The
synthetic overlays add missing tracklets and fake/background tracklets but do
not change their underlying physical refit state.

## Commands

```bash
source alignment_ML/scripts/setup_environment.sh ml
cd alignment_ML

python scripts/build_physical_curriculum_corpus.py \
  --config configs/physical_curriculum_mlp_muon.yaml \
  --output-dir outputs/mc24_muon_curriculum_physical_v1 --resume

python scripts/materialize_curriculum_synthetics.py \
  --physical-manifest outputs/mc24_muon_curriculum_physical_v1/physical_corpus_manifest.json \
  --output-dir outputs/mc24_muon_curriculum_synthetic_v1 --resume

python scripts/run_curriculum_mlp_baseline.py \
  --synthetic-manifest outputs/mc24_muon_curriculum_synthetic_v1/synthetic_corpus_manifest.json \
  --config configs/physical_curriculum_mlp_muon.yaml \
  --output-dir outputs/mc24_muon_curriculum_mlp_v1
```

## Evaluation Contract

For every candidate `chi2` gate, the runner trains one MLP, fits a scalar
temperature on validation candidates only, and scans decision thresholds on
validation greedy one-to-one associations.  It writes two complementary
operating-point definitions:

- one global threshold selected on all validation payload magnitudes, which
  tests a single deployment threshold;
- one threshold per injected magnitude, selected only from validation payloads
  at that same magnitude, which provides the controlled fixed-fake-rate and
  fixed-purity scan curves.

Neither selection reads test scores. Test outputs include:

- candidate truth recall before MLP selection;
- ROC AUC and average precision;
- Brier score, NLL and ECE before/after validation calibration;
- association efficiency, inclusive purity and inclusive fake rate by
  misalignment magnitude;
- the same metrics by station pair.

`candidate_gate_test_by_magnitude.csv` is written even if a learned operating
point cannot satisfy a requested constraint, keeping candidate-recall loss
separate from classifier/assignment loss.
`test_candidate_metrics_by_magnitude.csv` and
`test_candidate_metrics_by_station_pair.csv` are also always written: they
contain AUC, PR and calibrated/raw reliability metrics without requiring a
non-empty assignment operating point.

Rows carry `threshold_scope` and `validation_magnitude_mm`, so a global
deployment result cannot be confused with a magnitude-conditioned comparison.
The scan always retains the configured threshold grid and supplements it with
validation-score quantiles after temperature scaling.  This prevents a narrow
calibrated score range from hiding an otherwise valid high-score operating
point.

To reassess thresholds and calibration without retraining a completed model,
run the same command with a fresh output directory and
`--checkpoint-root outputs/mc24_muon_curriculum_mlp_v1`.

## Controlled-Pilot Result

The completed MC24 muon pilot uses 45 physical payloads from 25 original MC
events: 15 train, 5 validation and 5 test events, split by five mutually
exclusive xAOD chunks.  It produces 2,640/480/480 synthetic multi-track
events for train/validation/test.  These numbers make this a reproducibility
and failure-mode study, not a final statistical precision result.

At 50 mm, candidate truth recall is `0.298` for `chi2=250`, `0.567` for
`chi2=1000`, `0.891` for `chi2=5000`, and `1.000` without a gate.  The
ungated MLP test ROC AUC changes from `0.846` at 0 mm to `0.804` at 50 mm;
its average precision changes from `0.466` to `0.468`.  Validation-only
temperature scaling reduces the ungated test ECE from `0.146` to `0.063` at
0 mm and from `0.154` to `0.096` at 50 mm.

The predeclared primary operating targets, inclusive fake rate <= 0.05 and
inclusive purity >= 0.95, have no non-empty validation solution for any gate,
including nominal geometry.  Therefore no primary fixed-operating-point
association-efficiency curve is reported; fabricating one from an unmet
constraint would be misleading.  The ungated candidate recall is sufficient,
but the pairwise MLP is not usable at the primary association requirement even
before a high-misalignment degradation can be isolated.  This does not yet
meet the predeclared evidence condition for beginning a Transformer.

For context, the lowest non-empty validation fake rates are `7.1%`, `11.2%`,
`42.9%`, and `38.5%` for gates `250`, `1000`, `5000`, and ungated,
respectively.  The tight gate is closest to the requirement but loses 50 mm
candidate recall; widening the gate recovers candidates while exposing the
limitation of greedy pairwise assignment.

Results are under
`outputs/mc24_muon_curriculum_mlp_batch8192_reassessed_v1/`, including
`candidate_metrics_vs_misalignment.png`, candidate metrics by magnitude and
station pair, calibration, and the saved source-disjoint audit.

The test efficiency curve is evidence for a Transformer only when the
candidate recall remains high at a displaced point while the calibrated MLP
efficiency remains materially lower than its small-offset value.
`metrics.json` applies this declared screening criterion but never starts a
Transformer automatically.
