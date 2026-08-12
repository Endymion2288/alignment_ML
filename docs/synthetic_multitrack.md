# Controlled Synthetic Multi-Track Overlay

## Purpose

This dataset supplies the first controlled association input after propagation
and alignment closure. It overlays complete single-muon source tracks into one
synthetic event, then adds missing tracklets and fake/background tracklets. It
now supports the documented MLP pair-classifier baseline, but no Transformer.

## Construction

- A source track must have one truth-known segment at every requested station
  and truth-match fraction at least 0.99.
- If a station has multiple segments for the same truth particle, choose the
  highest truth-match fraction, then lowest local chi-square, then smallest
  original tracklet ID. This is a deterministic source selection rule.
- Source truth IDs are re-namespaced per synthetic event and track slot, so
  truth association labels remain unambiguous after overlay.
- Each true segment is independently removed with
  `missing_tracklet_probability`.
- For each station, a Poisson number of fake rows is drawn. Their local state
  and covariance are sampled from the original same-station pool, but their
  truth ID is set to `-1`.
- Rows are shuffled and receive new event-local `tracklet_id` values.
- Every row retains `origin_run_id`, `origin_event_id`, and
  `origin_tracklet_id`, allowing a mode-0 Acts source prediction from the
  same physical payload to be mapped into a synthetic event without a
  coordinate-level propagation approximation.

## Current Artifact

`outputs/mc24_muon_fasernu_5events_fieldaware_v4_truthcontrol/synthetic_multitrack_100events.root`
was built from five eligible MC24 source tracks using four tracks per event,
10% true-tracklet removal, and a fake mean of 0.5 per station. With seed
`20260808`, it contains 100 events and 1,642 rows: 1,449 known-truth rows,
151 removed true rows, and 193 fake/unknown rows. All four station IDs are
present; optional q/p fields are retained for schema compatibility, not as a
validated local q/p measurement.

Only five physical source trajectories are reused. This is suitable for
pipeline tests and controlled association ablations, not an unbiased final
training/evaluation split. The MLP scan's held-out synthetic-event split
controls overlay-event leakage, but it is not an independent source-event
split.

## Reproduction

```bash
cd /eos/home-x/xcheng/FASER
source alignment_ML/scripts/setup_environment.sh ml
cd alignment_ML

python scripts/make_synthetic_multitrack.py \
  --config configs/synthetic_multitrack_muon.yaml \
  --input outputs/mc24_muon_fasernu_5events_fieldaware_v4_truthcontrol/tracklets.root \
  --output outputs/mc24_muon_fasernu_5events_fieldaware_v4_truthcontrol/synthetic_multitrack_100events.root
```

The generator writes ROOT metadata and a sibling
`synthetic_multitrack_100events.resolved_config.yaml` file. Vary only the YAML
parameters to construct missing-tracklet, fake-rate, and multiplicity scans.
