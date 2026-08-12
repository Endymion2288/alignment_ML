# Data Audit: 2026-08-08

## Confirmed inputs

- Real IFT PHYS sample:
  `/eos/experiment/faser/phys/2024_ift/dev/014975/Faser-Physics-014975-00550-00553-PHYS.root`
  contains tree `nt` with 73,083 events.
- MC24 sample:
  `/eos/experiment/faser/sim/mc24/fluka/210010/phy/s0013-r0019/FaserMC-MC24_Fluka_2023_exp001_z448p6_d31p4_zsim3p99-210010-00000-00007-s0013-s0012-PHYS.root`
  contains 200,000 events.
- Corresponding MC xAOD retains `Segments`, `SegmentFit`,
  `SCT_ClusterContainer`, `SCT_SDO_Map`, `TruthParticles`, and `SCT_Hits`.
- Electron PHYS audit sample:
  `/eos/experiment/faser/sim/mc22/particle_gun/100020/phy/r0013/FaserMC-MC22_PG_elec_logE-100020-00000-00004-s0008-r0013-PHYS.root`
  contains 2,500 events. In its first 200 events, 193 contain local segments
  and global-track truth contains `pdg=+/-11`. It is a legacy three-station
  PHYS output, not an IFT-plus-three-station training dataset.
- Electron xAOD validation sample:
  `/eos/experiment/faser/sim/mc22/particle_gun/100022/rec/s0012-r0019/FaserMC-MC22_PG_elec_100GeV-100022-00000-00004-s0012-r0019-xAOD.root`.
  It is an MC22 three-station sample and must be run with `TI12MC03`.
- IFT-plus-three-station structural validation sample:
  `/eos/experiment/faser/sim/mc24/particle_gun/100012/rec/dev/FaserMC-MC24_PG_muon_fasernu_100GeV-100012-00000-00004-xAOD.root`.
  Its runtime metadata selects `FASERNU-04`; it is a muon control sample, not
  the final electron training source.

## Executed Export Checks

- The first 10 events of the selected MC24 FLUKA xAOD have empty `SegmentFit`
  and `Segments` collections. The exporter therefore produces a valid empty
  canonical tree; this is a sample-content result, not an exporter failure.
- The MC22 electron sample exported 28 canonical tracklets in 10 events with
  observed station IDs `1,2,3`, valid covariance on every row, and per-segment
  MC labels. It validates the electron export path but does not cover IFT.
- The MC24 FASERnu muon sample exported 24 canonical tracklets in 5 events
  with observed station IDs `0,1,2,3`, including station `0` in every event.
  It validates the IFT-plus-three-station schema path but does not substitute
  for electron training.

Detailed commands, content summaries, and chi-square baseline results are in
[baseline validation](baseline_validation.md).

## data0 Follow-up

- The MC22 100 GeV electron xAOD is also available at
  `/eos/experiment/faser/data0/sim/mc22/particle_gun/100022/rec/s0012-r0019/`
  and is the same EOS inode as the already validated path above. Its 10-event
  canonical export has 28 tracklets in stations `1,2,3`, positive-definite
  covariance on every row, and truth PDG values `+/-11`. It is suitable only
  for a three-station electron smoke test.
- No MC22 reconstructed xAOD with an `r0022` tag or an `IFT` filename was
  found under `data0/sim/mc22`. This filename/path audit is not proof that no
  suitable production exists elsewhere, but it rules out treating the located
  MC22 particle-gun sample as an IFT-plus-three-station dataset.
- `data0/phys/2022_ift/r0022/008090` contains an existing PHYS ntuple with
  18,363 rows, of which 18,116 have `TrackSegments>0`. The paired visible
  xAOD path, `data0/rec/2022/r0022/008090`, has a 44,639-entry
  `CollectionTree` and persists the keys `SegmentFit` and `Segments`.
- Re-exporting its first 100 xAOD events with `--useIFT --export-tracklets`, no
  global-track or stable-beam filter, default blinding, and
  `--skip-ghostbusters` produced 100 ntuple rows but zero `TrackSegments`,
  `GhostBustedTrackSegments`, and detailed tracklets. Those exported rows use
  event IDs `1...100`, whereas the existing PHYS file begins at event ID
  `767688`. The provenance/content relation therefore remains unverified; do
  not use this xAOD/PHYS pairing for training or alignment until the exact
  production input is identified.
- Some MC22 FASER-FORESEE `Aee` PHYS files contain multi-lepton truth content,
  but the inspected `110052/rec` directory has no xAOD input. Its legacy PHYS
  fields still lack station ID, covariance, hit information, and per-tracklet
  truth labels, so it is not an alternative V1 input without re-reconstruction.

The NtupleDumper command now has an opt-in `--skip-ghostbusters` switch for
inputs that already persist `Segments`; it prevents scheduling a duplicate
writer for that key. The Calypso setup helper puts the workspace CLI ahead of
the installed copy so this Python-only option is usable without rebuilding the
runtime libraries.

## Existing PHYS Field Mapping

| Requested quantity | Existing PHYS field | Status |
| --- | --- | --- |
| run/event ID | `run`, `eventID` | available |
| tracklet position | `TrackSegment_x/y/z` | available |
| slopes | derivable from `TrackSegment_px/pz`, `py/pz` | not stored explicitly |
| local chi2/ndf | `TrackSegment_Chi2`, `TrackSegment_nDoF` | available |
| station ID | none | missing |
| covariance or errors | none | missing |
| tracklet hit count/pattern | none | missing |
| module/raw-hit IDs | none | missing |
| per-segment MC truth ID | none | missing |
| nominal station transforms | none | missing from PHYS; available only through Calypso geometry/conditions |
| injected misalignment | none | must be generated and recorded by the study |

`TrueTrackSegments` in older real PHYS output is the historical name of the
ghost-busted segment collection, not a truth label. MC `t_barcode` belongs to
global CKF tracks, so it must not be used as a local-segment association label.

## Geometry Semantics Confirmed in Source

For the current FASER geometry factory, `SCT_DetectorFactory.cxx` builds
stations in the ordered list `Interface`, `StationA`, `StationB`, `StationC`
and assigns their identifiers `0`, `1`, `2`, `3`. Exported station IDs will
come from each cluster identifier, not from the observed segment z position.
The runtime conditions tag has been observed, but the audit has not yet
produced an authoritative numerical station-level nominal transform payload.
That transform and the injected alignment payload remain required geometry
sidecar metadata for alignment experiments.
