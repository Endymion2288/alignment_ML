# Physical IFT R_y Rotation Study

## Scope and isolation

This study pauses the V3 structured-margin objective. It does not open,
modify, calibrate on, or otherwise use the sealed V1/V2/V3 test sources.
The rotation curriculum inherits only the existing source-file-disjoint train
and validation source list. The downstream three-station frame is fixed:
stations 1, 2, and 3 have identity transforms in every payload, while IFT is
station 0.

The physical corpus configuration is
`configs/physical_curriculum_ift_ry_expanded_trainval.yaml`. It declares pure
IFT rotations at `0, +/-10, +/-25, +/-40, +/-60 mrad` and four controlled
`dx/dy + R_y` trials at `|R_y| = 40 mrad`. No coordinate-level injection is
accepted for this study.

## Verified conditions semantics

Calypso's `TrackerAlignDBTool::stationAlignment` validates six constants per
station in the following order:

```text
[dx_mm, dy_mm, dz_mm, rx_rad, ry_rad, rz_rad]
```

It applies the global transform as

```text
T(dx, dy, dz) * Rz(rz) * Ry(ry) * Rx(rx)
```

with translations in mm and rotations in rad. The tracker conditions
algorithm creates `SCTAlignmentStore`; `FaserActsAlignmentCondAlg` copies that
store and caches each aligned detector-element transform in
`FaserActsAlignmentStore`. The ACTS detector element reads that store from the
geometry context, so a physical payload changes both segment refit geometry
and the ACTS surface transform used by mode-0 propagation.

## Reproducible smoke

The 10-event source-isolated smoke plan is
`configs/physical_refit_ift_ry_smoke_mc24_100043.yaml`. Every selected point
runs:

```text
/Tracker/Align SQLite/POOL
  -> SCT_ClusterContainer
  -> SegmentFitRefit
  -> SegmentsRefit
  -> NtupleDumper
  -> FaserActsExtrapolationTool, q/p mode 0
```

All 13 frozen smoke points have completed in
`outputs/mc24_ift_ry_physical_smoke_v1`: pure rotations at
`0, +/-10, +/-25, +/-40, +/-60 mrad`, plus four `+/-40 mrad + dx/dy` joint
trials.

For `+10 mrad`, the truth-matched IFT state changed by mean
`delta x = -18.5981 mm` and `delta tx = +0.0100209`; for `-10 mrad`, the
corresponding values were `+18.5970 mm` and `-0.0100143`. This observed
near-antisymmetry is a physical refit response, not an imposed coordinate
offset. At every one of the 13 points, each adjacent truth edge and each
complete IFT -> S1 -> S2 -> S3 truth chain was retained for all 10 events;
raw candidate truth-chain recall is `1.0`, including `+/-60 mrad`. At
`+60 mrad`, the mean IFT state response is
`delta x=-111.5404 mm, delta tx=+0.0602876`; at `-60 mrad`, it is
`delta x=+111.4999 mm, delta tx=-0.0600532`.

The logs contain ACTS layer-overlap and surface-error messages even for the
nominal geometry. The audit therefore records those log counts separately
from the exact mode-0 candidate edges: a job-level log error is not equated
with failure of a truth-matched propagated state or covariance. Each completed
point now has state, residual, pull, chi2, combined covariance/logdet, and
observed `x/y/tx/ty` slope/correlation outputs. The 10-event smoke is a
mechanism check only, not a statistically frozen position-dependence result.

## Tools

- `scripts/write_station_alignment_payload.py` supports exact six-component
  `--transform` arguments while retaining legacy `--offset` support.
- `scripts/run_physical_refit_capture_scan.py` has an explicit
  `ift_ry_rotation` scan mode and never routes a rotation through the
  coordinate surrogate.
- `scripts/audit_ift_ry_refit_response.py` compares truth-matched tracklet
  state, covariance, residual, pull, and chi2 response versus nominal.
- `scripts/audit_physical_route_candidate_graph.py` measures raw mode-0
  physical candidate and complete truth-chain retention and logs ACTS surface
  response separately.
- `scripts/run_refit_ift_ry_closure.py` performs a one-parameter, central
  finite-difference WLS closure from independently refitted `+R_y` and `-R_y`
  probes. It fixes stations 1--3 and rejects joint translation trials for the
  one-parameter closure.
- `scripts/run_refit_ift_ry_scan_closure.py` evaluates a held-out scalar
  closure using several independently physical response points; it is not a
  coordinate/residual surrogate.
- `scripts/run_physical_refit_scan_condor.sh` and
  `scripts/submit_physical_refit_scan_condor.py` run an arbitrary frozen scan
  as one Condor job while preserving the same real payload/refit/Acts driver.
- `scripts/audit_ift_ry_physical_corpus.py` performs a read-only aggregate
  audit over a completed source-disjoint train/validation corpus, including
  actual mode-0 candidate/route/surface metrics and nominal `x/y/tx/ty`
  position-dependence regressions; it rejects incomplete points and test
  sources.

The station-0 parameter is identifiable but current one-step closure has not
reached 1 mrad: the `+/-10 mrad` central response gives `5.775/5.191 mrad`
absolute error at held-out `+/-60 mrad`, and a quadratic held-out physical
curve gives `4.233/7.425 mrad`. Independent physical refinement probes at
`0,+/-45,+/-50,+/-55 mrad` have therefore been submitted while `+/-60 mrad`
remains held out. In parallel, the expanded train/validation-only 13-point
rotation bank is being produced on Condor. It must finish and pass physical
manifest auditing before synthetic multi-track materialization and
validation-only MLP/V1/V2 retraining; the sealed test sources will not be
opened or recalibrated.

The local refinement has now completed and passed: using a `+50 mrad` anchor
with `+45/+55 mrad` physical probes recovers the held-out `+60 mrad` point as
`+60.06898 mrad` (absolute error `0.06898 mrad`); a `-50 mrad` anchor with
`-55/-45 mrad` probes recovers held-out `-60 mrad` as `-60.41250 mrad`
(absolute error `0.41250 mrad`). Both normal matrices have rank 1 and
condition number 1. Thus real physical refit/Acts closure works through a
coarse-to-local iteration; the earlier failure was a large-range one-step
linearization, not non-identifiability of the rotation.

The expanded physical bank has completed and passed manifest auditing: 10
train sources/994 events, 8 validation sources/796 events, 234/234 completed
points, and no test source. On the larger validation corpus, raw candidate
complete truth-chain recall is `0.9493` at nominal, `0.9542` at `-60 mrad`,
and `0.9678` at `+60 mrad`; rotation has not removed truth routes from the
candidate graph, although `0->1` remains the principal loss pair. A read-only
source `x/y/tx/ty` position-dependence audit is available.

## Frozen validation controls

The MLP, V1, and V2 studies are now complete. All controls use the same
source-disjoint physical train/validation corpus, mode-0 Acts candidates, and
the exact adjacent unit-capacity route solver. The historical test source was
not opened. V1 and V2 optimize on train sources and use validation only for
early stopping, calibration, and route-control selection. The MLP route
postprocessor (`outputs/ift_ry_mlp_route_validation_v1`)
loaded only validation events, reused the frozen MLP checkpoint and
station-pair calibration, and recorded `test_events_loaded=false`,
`test_artifacts_opened=false`, and `calibration_refit=false`. Its checkpoint
and calibration SHA-256 values agree with the declared frozen artifacts.

The predeclared primary point is complete-track efficiency >= 0.70, purity >=
0.95, and fake rate <= 0.05. Pooled validation values at nominal / `|R_y|=60`
mrad are:

| Control | Efficiency | Purity | Fake rate | Capture |
| --- | --- | --- | --- | --- |
| Pairwise MLP + route assignment | 0.9489 / 0.9370 | 0.9888 / 0.9878 | 0.0312 / 0.0270 | 5/5 points |
| V1 geometry-aware full context | 0.9651 / 0.9626 | 0.9819 / 0.9859 | 0.0295 / 0.0253 | 5/5 points |
| V1 without multi-station context | 0.9382 / 0.9429 | 0.9848 / 0.9870 | 0.0210 / 0.0192 | 5/5 points |
| V2 BCE route-query | 0.9355 / 0.9381 | 0.9797 / 0.9811 | 0.0404 / 0.0368 | 5/5 points |

The same-checkpoint V2 edge-only comparison does not meet the primary point:
its nominal fake rate is `0.0628`. Thus the V2 route correction reduces false
routes enough to pass its own criterion, but it does not exceed V1 full-context
efficiency in this rotation bank. The pairwise MLP selects validation-only
thresholds `0->1=0.20`, `1->2=0.05`, `2->3=0.05`, with unmatched penalty `1.0`;
its frozen calibrated candidate AP is `0.9738` and ECE is `0.00520`.

Every synthetic-overlay control has candidate complete-truth-chain recall
`1.0`. This is deliberately not compared as an equal denominator to the raw
physical value above: multi-track overlays select complete physical truth
tracks, whereas the raw audit includes every refitted physical truth segment.
At pure `+/-60 mrad`, the complete-track efficiency ranges are `0.9366--0.9374`
for MLP, `0.9620--0.9632` for V1 full context, and `0.9342--0.9420` for V2;
all sign trials capture. The four `|R_y|=40 mrad` joint `dx/dy` trials also
capture for every listed control.

The rotation response is not merely a mean shift. In the validation raw
physical audit at `-60 mrad`, IFT `delta t_x` has correlations `-0.415` with
source `t_x` and `-0.414` with source `t_y`; IFT `delta t_y` has correlation
`-0.627` with source `t_y`. For mode-0 `0->1`, the `delta r_{t_x}` correlation
with source `y` is `+0.576` at `-60 mrad` and `-0.337` at `+60 mrad`.
These are observed refit/Acts responses, not features injected through a
coordinate surrogate.

Consequently, this bank does not satisfy the trigger for a new hard-route
contrastive objective: physical truth candidates remain, and the current
pairwise and route models do not show a primary-point failure through 60 mrad.
No new test bank is generated, and V3 structured-margin remains paused. The
next useful study is a more challenging real-background/association corpus or
an expanded physical closure campaign, not a deeper Transformer or test-time
retuning. The read-only combined plot and table are in
`outputs/ift_ry_validation_control_assessment_v2`.
