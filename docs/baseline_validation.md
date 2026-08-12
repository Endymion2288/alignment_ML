# Baseline Validation: 2026-08-08

## What Was Validated

The following chain has been executed on lxplus, not only unit-tested:

1. Calypso reads an xAOD and writes opt-in event-wise `Tracklet_*` branches
   from the ghost-busted `Segments` collection.
2. `scripts.convert_ntuple_tracklets` validates vector lengths and writes the
   canonical flat `tracklets` tree.
3. `scripts.audit_tracklets` checks observed station occupancy, covariance
   definiteness, hit/fit summaries, and MC-label completeness.
4. `scripts.run_chi2_baseline` runs a straight-line propagated covariance
   gate followed by greedy one-to-one assignment.

The baseline CSV now carries the source/target truth barcodes and a
`truth_relation`. A predicted edge with an unknown or duplicate truth label is
reported as `unscorable`; it is excluded from purity and fake-rate denominators
instead of being labelled as a fake.

## MC Samples and Observed Payload

| Sample | Input | Geometry used | Events | Canonical tracklets | Observed station IDs |
| --- | --- | --- | ---: | ---: | --- |
| MC22 100 GeV electron particle gun | `/eos/experiment/faser/sim/mc22/particle_gun/100022/rec/s0012-r0019/FaserMC-MC22_PG_elec_100GeV-100022-00000-00004-s0012-r0019-xAOD.root` | `TI12MC03` | 10 | 28 | `1, 2, 3` |
| MC24 100 GeV FASERnu muon particle gun, development sample | `/eos/experiment/faser/sim/mc24/particle_gun/100012/rec/dev/FaserMC-MC24_PG_muon_fasernu_100GeV-100012-00000-00004-xAOD.root` | auto-configured `FASERNU-04` | 5 | 24 | `0, 1, 2, 3` |

The MC22 electron payload has positive-definite covariance for all 28 rows and
known per-tracklet truth labels for all rows. It is a three-station sample;
station `0` is absent. The MC24 FASERnu payload has positive-definite
covariance for all 24 rows; station `0` is observed in every event, and its
runtime geometry loaded `/Tracker/Align` with `TRACKER-ALIGN-02`.

The identifiers themselves establish the station labels. The reported z values
are only an audit cross-check, not the station-label source.

## Straight-Line Baseline Results

`possible` denotes truth IDs unique in both stations. `raw/scored/unscorable`
denotes all greedy assignments, the subset with two unambiguous known endpoint
labels, and the remaining subset. Wide gate `1e6` is a diagnostic value for
these tiny samples, not a production gate choice.

| Sample | Pair / gate | possible / correct | raw / scored / unscorable | efficiency | purity | fake rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| MC22 electron | `1 -> 2`, 50 | 9 / 3 | 3 / 3 / 0 | 0.333 | 1.000 | 0.000 |
| MC22 electron | `1 -> 2`, `1e6` | 9 / 9 | 10 / 9 / 1 | 1.000 | 1.000 | 0.000 |
| MC22 electron | `2 -> 3`, `1e6` | 4 / 4 | 5 / 4 / 1 | 1.000 | 1.000 | 0.000 |
| MC24 FASERnu muon | `0 -> 1`, `1e6` | 4 / 4 | 5 / 4 / 1 | 1.000 | 1.000 | 0.000 |
| MC24 FASERnu muon | `1 -> 2`, `1e6` | 3 / 3 | 5 / 3 / 2 | 1.000 | 1.000 | 0.000 |
| MC24 FASERnu muon | `2 -> 3`, `1e6` | 4 / 4 | 5 / 4 / 1 | 1.000 | 1.000 | 0.000 |

The default gate of 50 produces no `0 -> 1` candidate in the MC24 FASERnu
sample. With the temporary straight-line propagator, the same-truth `0 -> 1`
candidates have chi-square from about 118 to 1,617. This is expected evidence
that a straight line is not an adequate field-aware gate across that span; it
is not evidence of a missing station or invalid covariance.

These small particle-gun samples do not provide a meaningful false-pair
population: the candidate audit for MC22 `1 -> 2` found 13 wide-gate pairs and
all have the same known truth barcode. The table therefore validates the data
contract and baseline execution, not a final association-performance claim.

## Negative Check

The first 10 events of
`/eos/experiment/faser/sim/mc24/fluka/210010/rec/s0013-r0019/FaserMC-MC24_Fluka_2023_exp001_z448p6_d31p4_zsim3p99-210010-00000-00007-s0013-xAOD.root`
had both `SegmentFit` and ghost-busted `Segments` empty. The exporter wrote a
valid empty canonical tree rather than silently fabricating rows. That FLUKA
slice is unsuitable for baseline validation without a selection that contains
local segments.

## Reproduction

```bash
cd /eos/home-x/xcheng/FASER
alignment_ML/scripts/export_mc_tracklets.sh \
  --input /eos/experiment/faser/sim/mc24/particle_gun/100012/rec/dev/FaserMC-MC24_PG_muon_fasernu_100GeV-100012-00000-00004-xAOD.root \
  --output-dir alignment_ML/outputs/mc24_muon_fasernu_check \
  --nevents 5 \
  --source-station 0 --target-station 1 --chi2-gate 1e6

source alignment_ML/scripts/setup_environment.sh ml
cd alignment_ML
python -m scripts.audit_tracklets \
  outputs/mc22_elec_100gev_10events/tracklets.root \
  --require-mc-labels --source-station 1 --target-station 2 \
  --chi2-gate 1e6
```

The first command makes a new directory and refuses to overwrite existing
artifacts. It saves `content_audit.json`, `metrics.json`, `matches.csv`, the
resolved configuration, and the candidate-chi-square plot.

## Blocking Inputs Before Transformer Work

1. A nominated or newly produced MC24 electron xAOD with confirmed station
   `0,1,2,3` local segments is required for the stated electron study. A
   filename search under the current MC24 particle-gun directory found no
   xAOD names containing `elec` or `electron`; that does not prove that no
   suitable campaign exists.
2. A versioned numerical geometry sidecar is still required: nominal station
   transforms, geometry/conditions tags, reference station, and the exact
   injected `dx_mm`, `dy_mm` for every episode. The runtime conditions tag is
   known, but no numerical transform is inferred from tracklet z values.
3. `TrackletHit_cluster_identifier` is available in the enhanced output, but
   it is not a raw RDO ID. Raw hit IDs remain unimplemented until their
   persistence semantics are verified.
4. The candidate gate needs a field-aware first-order propagation or a
   validated Calypso/Acts propagation payload before gate thresholds can be
   interpreted physically. The current straight-line implementation remains a
   deliberately simple control baseline.
