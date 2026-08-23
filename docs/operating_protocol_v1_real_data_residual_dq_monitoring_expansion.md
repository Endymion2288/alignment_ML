# Operating Protocol V1 real-data residual/DQ monitoring expansion

Workbook 53 / 2026-08-23. Entry 52 remains frozen as
`real_data_residual_dq_monitoring_only`. This stage does **not** open a
Station or C_dx solve. It applies the already-frozen current-geometry
monitoring chain to the pre-listed independent 2024 r0022 runs
14971, 14972, 14980, 14981, 14985, 14989, and 15007.

## Frozen invariants

- Propagation: mode-0
- Association: frozen V2 (`0c85a001…766a27`)
- Threshold `0.001`, `unmatched_penalty=-1.0`
- Observation: `anchor_selected_field_edge` + `physical_edge_deduplicated`
- Occupancy rule: first 100-event window meeting 16/16/32/8, then the
  next r0022 segment if needed. Occupancy only finds a start.
  Official monitoring uses the full remaining segment.
- Calibration reference: the entry-52 14973/14974 scale, not re-estimated
- Alarms: the entry-52 priority, not retuned
- `geometry_write_allowed=false`
- `station_calibration_mode_available=false`
- `cdx_mode_allowed=false`

Forbidden: FD probes; Newton; alignment payload writes; V2 retraining;
threshold or penalty changes; residual-based window re-picks; converting
an `alignment_drift_candidate` into a geometry update; reopening
self-nulling calibration.

## Occupancy

Condor **1001193** scanned segments `00000–00007` for all seven runs.
14972/`00003` hit an EOS mkdir race and was retried with 14985
`00008–00015` on **1001194**. All seven windows froze without lowering
minima.

| run | LHC fill | segment | skip | remaining events |
| ---: | ---: | --- | ---: | ---: |
| 14971 | 9564 | 00003 | 20900 | 112357 |
| 14972 | 9565 | 00004 | 23300 | 110071 |
| 14980 | 9573 | 00005 | 57300 | 77621 |
| 14981 | 9574 | 00006 | 95600 | 40646 |
| 14985 | 9575 | 00012 | 95300 | 40837 |
| 14989 | 9579 | 00005 | 71900 | 63559 |
| 15007 | 9585 | 00005 | 111800 | 24905 |

## Current-geometry Athena and frozen V2

Condor **1001195** reconstructed all seven full remaining segments at
official geometry (`--NoTrackFilt --no_stable`, one current point, no
FD). Frozen V2 association ran locally on GPU.

| run | events | tracklets | all-pairs | selected | 2/3/4-st | `dy` z | `rx` z | status |
| ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |
| 14971 | 112357 | 88562 | 99021 | 233 | 78/151/4 | −0.055 | −0.008 | nominal |
| 14972 | 110071 | 84697 | 94951 | 193 | 72/119/2 | 0.048 | 0.034 | nominal |
| 14980 | 77621 | 55978 | 62395 | 114 | 43/69/2 | −0.175 | −0.046 | nominal |
| 14981 | 40646 | 28678 | 31787 | 77 | 31/46/0 | −0.049 | 0.056 | nominal |
| 14985 | 40837 | 29762 | 33246 | 74 | 20/53/1 | −0.055 | 0.035 | nominal |
| 14989 | 63559 | 44454 | 49757 | 91 | 20/66/5 | 0.089 | 0.046 | nominal |
| 15007 | 24905 | 18387 | 20570 | 32 | 13/19/0 | 0.050 | 0.014 | nominal |

The parent 14973–14976 blocks stay nominal. 14977 stays
`insufficient_statistics_for_alignment_dq`. Time order is LHC fill, then
run, then skip — not a run-number proxy. In this corpus fill and run
increase together. No adjacent statistically sufficient pair has a
same-sign isolation `|z| ≥ 3`. `alignment_drift_candidate=false`.
`dx`/`ry` remain cross-level-sensitive auxiliaries. `dz` is not a
track-driven observable. `rz` is not fabricated.

## Unique decision

**`real_data_residual_dq_monitoring_only`** stays frozen.

The seven new runs are all `nominal_monitoring`. Current official
geometry plus frozen V2 can carry long-term DQ monitoring. There is no
consecutive detector-condition change and no alignment-drift candidate,
so this is not a survey/external-alignment trigger and not a
self-nulling reopen.

`geometry_write_allowed=false`,
`station_calibration_mode_available=false`,
`cdx_mode_allowed=false`.

## Outputs

Under `outputs/operating_protocol_v1_real_data_residual_dq_monitoring_expansion_v1/`:

- `expansion_provenance.json`
- `run_level_dq_report.json`
- `time_stability_report.json`
- `alarm_summary.json`
- `operating_protocol_monitoring_decision.json`
