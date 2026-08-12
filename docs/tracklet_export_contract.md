# Tracklet Export Contract

## Scope

The V1 exporter reads the existing ghost-busted `Segments` local-track
collection. It does not rerun hit finding or global tracking. Calypso first
writes event-wise `Tracklet_*` vectors in the normal `nt` tree; the Python
converter then writes one flat row per local tracklet to `tracklets`.

## Required Columns

| Field group | Required columns | Source requirement |
| --- | --- | --- |
| identity | `run_id`, `event_id`, `station_id`, `tracklet_id` | `station_id` must come from `FaserSCT_ID::station()` on the associated cluster, not from a z-bin heuristic. |
| state | `x_mm`, `y_mm`, `z_mm`, `tx`, `ty` | Position and direction must use the same fitted reference surface. |
| covariance | the ten `cov_*` columns in `[x, y, tx, ty]` order | The exporter must record the transformation method and reject invalid matrices. |
| quality | `chi2`, `ndof`, `n_hit`, `hit_pattern` | `hit_pattern` is a station-local layer bit mask. |
| provenance | optional `module_ids`, `raw_hit_ids`, `source_file_id`; synthetic-only `origin_run_id`, `origin_event_id`, `origin_tracklet_id` | The V1 detector export does not yet carry module/raw-hit columns. Synthetic overlays retain origin identity so same-payload Acts predictions can be mapped without coordinate-level propagation. Enhanced output retains module/cluster identifiers per hit; raw RDO IDs are not claimed. |
| MC labels | `truth_particle_id`, `truth_pdg`, `truth_match_fraction` | Match each local segment with `ITrackTruthMatchingTool`; use `-1` for unmatched data/fakes. |

## Implemented Export Path

The Calypso changes are opt-in through `faser_ntuple_maker.py
--export-tracklets`. They leave the default PHYS branches unchanged and add
the following event-wise vectors only when enabled:

- `Tracklet_station_id`, state, quality, and `Tracklet_id` are emitted from
  the ghost-busted segment itself. The station is obtained from every
  `FaserSCT_ClusterOnTrack` identifier and a segment spanning more than one
  station is rejected.
- Native track-parameter covariance is transformed from
  `(loc1, loc2, phi, theta, q/p)` to global `[x, y, tx, ty]` by a central
  numerical Jacobian. Each perturbation is constructed with the associated
  `Trk::Surface`, so the measured module orientation is retained. The `q/p`
  Jacobian column is zero because this V1 state contains no momentum-magnitude
  coordinate. `Tracklet_has_covariance=false` marks a segment that cannot be
  transformed safely; its covariance entries are NaN and the converter drops
  it by default.
- Optional q/p audit branches retain native `q/p`, the equivalent value from
  reconstructed momentum, and the q/p covariance validity flag. They are not
  required by the original MC22 electron smoke test. When
  `--export-tracklet-propagation` is enabled, truth-matched source/target
  segment pairs are also propagated with `FaserActsExtrapolationTool` and
  written as separate event-wise vectors for the field-aware validation path.
- `TrackletHit_*` vectors retain per-cluster station/layer/module metadata.
  `TrackletHit_cluster_identifier` is a reconstruction-level cluster
  identifier, not a raw RDO identifier. Raw RDO lists are intentionally not
  claimed until their exact persistence semantics are validated.
- For MC only, `Tracklet_truth_*` comes from one
  `ITrackTruthMatchingTool::getTruthParticle(segment)` call per local
  segment. It is independent of the existing global-track `t_barcode` branch.

The flat conversion is reproducible with:

```bash
source alignment_ML/scripts/setup_environment.sh ml
cd alignment_ML
python -m scripts.convert_ntuple_tracklets INPUT-PHYS.root \
  --output data/tracklets.root --include-truth
python -m scripts.inspect_root_schema data/tracklets.root
python -m scripts.audit_tracklets data/tracklets.root --require-mc-labels
python -m scripts.run_chi2_baseline \
  --input data/tracklets.root --config configs/baseline_chi2.yaml \
  --output-dir outputs/chi2_mc
```

`--include-truth` is appropriate only for an MC export. For data, omit it.
The converter rejects event-wise branch length mismatches and, by default,
drops only rows explicitly marked as lacking covariance.

## Geometry Contract

A run also requires a versioned geometry YAML/JSON recording the nominal
transform of each station, source geometry tag, conditions tag, and the
reference station. V1 alignment parameters are only station-level `dx_mm`
and `dy_mm`; one station is fixed exactly to `(0, 0)`.

## Current Status

The existing PHYS tree has local segment position, momentum, chi-square, and
ndof, but not the explicit columns above. The opt-in Calypso exporter is now
built and installed, and small xAOD exports have been validated through the
canonical loader and chi-square baseline. It never fabricates missing
covariance or station labels.

Two distinct validation samples are deliberately kept separate: MC22 100 GeV
electrons validate per-segment truth and the electron export path but expose
only station IDs `1,2,3`; MC24 FASERnu muons validate station IDs `0,1,2,3`
and the IFT path but are not electron training data. The MC24 FLUKA slice used
for the first check has no local segments in its first 10 events. See
[baseline validation](baseline_validation.md) for commands and exact results.

The numerical station-transform sidecar, raw RDO IDs, and an IFT-plus-three
station electron MC source are still open inputs. A controlled truth-fixed,
baseline-subtracted residual-level x/y closure is now allowed on the MC24
muon sample; it is not a replacement for a deformed-geometry ACTS closure.
Do not begin Transformer association training until covariance calibration,
physical geometry injection, and the controlled multi-track association
baseline have been validated.
