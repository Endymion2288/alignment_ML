# Physical Capture-Range Scan

This scan measures the capture range of the truth-fixed station-level
alignment solve using reconstructed detector data, not a coordinate-level
surrogate.  It is intentionally performed before any learned association
model is introduced.

## Reconstructed chain

For every planned point the driver creates a new `/Tracker/Align`
SQLite/POOL payload and reruns:

```text
persisted SCT_ClusterContainer
  -> SegmentFitRefit
  -> SegmentsRefit
  -> NtupleDumper tracklet and propagation export
  -> FaserActsExtrapolationTool (q/p mode 0)
  -> truth-fixed WLS alignment closure
```

The source xAOD is reused, but neither segment coordinates nor propagation
residuals are reused between payload points.  Mode 0 is compulsory because
the local segment `q/p` is a fixed seed from `SegmentFitAlg`, rather than a
reliable measured parameter.

The separately refitted zero-payload control is retained only as the common
physical WLS baseline when forming a residual increment.  It is never shifted
or otherwise transformed to stand in for a nonzero injection: every observed
nonzero residual is exported from that point's own Calypso refit.

## Scan definition

`configs/physical_refit_capture_scan_muon.yaml` defines the MC24 100 GeV
FASERnu muon source, five events, station 0 as reference, three deterministic
directions in the station-1-to-3 `dx/dy` space, and increasing offset
magnitudes.  The zero-offset point is run once; all nonzero magnitudes are run
for all three directions.  This gives a direction-sampled capture fraction
instead of treating one injection as a statistical fraction.

The current success condition is:

1. physical refit and propagation completed;
2. at least one truth-fixed pair is active after the WLS gate;
3. normal-matrix rank equals six for the three movable stations;
4. maximum movable-station recovery-norm error is at most `0.01 mm`.

## Current Control Result

The completed MC24 five-event control is stored in
`outputs/mc24_muon_fasernu_physical_capture_scan_v1/`. It has 25 physical
points: one zero-offset point and three direction trials at each nonzero
magnitude. Every point completed its own refit/export/propagation chain.

With the configured `0.01 mm` criterion, the zero point and all three
`0.1 mm` trials capture. Their maximum station-norm recovery errors are
`0`, `0.00493`, `0.00330`, and `0.00132 mm`, respectively. All three
`1 mm` trials retain rank six and truth pairs but fail the strict tolerance
(`0.0487`, `0.0334`, and `0.0133 mm`). Higher magnitudes do not capture;
some `500` and `1000 mm` directions also lose normal-matrix rank after the
WLS gate. The resulting capture range is therefore at most `0.1 mm` under
this exact control configuration and tolerance.

This is not a detector-wide performance claim. It depends on five source
events, the selected three displacement directions, the fixed truth
association, mode-0 seed q/p, WLS gate, and refit configuration.

## Reproduction

Run the orchestration process from a clean Python environment.  It invokes
the Calypso and ML environments independently for the appropriate child
steps, preventing LCG Python-library cross-contamination.

```bash
cd /eos/home-x/xcheng/FASER
env -i HOME="$HOME" USER="$USER" LOGNAME="$USER" \
  PATH=/usr/local/bin:/usr/bin:/bin SHELL=/bin/bash \
  /usr/bin/python3 alignment_ML/scripts/run_physical_refit_capture_scan.py \
  --config alignment_ML/configs/physical_refit_capture_scan_muon.yaml \
  --output-dir alignment_ML/outputs/mc24_muon_fasernu_physical_capture_scan_v1
```

An interrupted identical scan can continue with `--resume`.  The driver
refuses a nonempty output directory without that flag and compares the saved
scan plan before resuming.

## Artifacts

Each `points/<point>/` directory contains its conditions payload, Athena
logs, refitted ROOT output, canonical tracklets, mode-0 propagation records,
content audit, and closure artifacts.  The scan root contains:

- `scan_plan.json` and `resolved_config.yaml`
- `capture_scan_summary.json` and `capture_scan_points.csv`
- `capture_scan_station_pair_diagnostics.csv`, with field-aware `rx`, `ry`,
  `rtx`, `rty`, pulls, and chi-square statistics for every station pair
- `capture_scan.png`, showing capture fraction and median recovery error as a
  function of injected offset magnitude

Each point-level `closure.json` also preserves the available and active
truth-pair counts, WLS normal-matrix rank and condition number, injected and
recovered offsets, per-station absolute recovery errors, covariance-response
diagnostics, and raw field-aware residual/pull summaries.
