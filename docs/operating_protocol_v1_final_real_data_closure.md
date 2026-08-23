# Operating Protocol V1 final real-data closure and reproducibility freeze

Workbook 54 / 2026-08-23. Entries 47–53 are assembled into an immutable
evidence package. This stage does **not** reopen Station Mode, reduced
Station Mode, or `C_dx` Mode. It does not retrain V2, retune occupancy,
alarms, `A`, or the 14973/14974 residual scale, and it does not extract
an alignment payload from self-nulling residuals.

## Evidence chain

`MC transfer PASS → real-data frozen-V2 acceptance statistics-limited but recovered at scale → full-segment Station calibration REJECTED by physical_nonidentifiability/cross_level_contamination → reduced-mode calibration REJECTED by cross-run non-transferability → current-geometry residual/DQ monitoring PASS on independent runs`

| step | entries | verdict |
| ---: | --- | --- |
| 1 | 47 | MC transfer **PASS**. Source-disjoint Station and IFT-Internal both close. `A` is not updated. |
| 2 | 48–49 | Frozen-V2 acceptance is statistics-limited at 100 events and recovered at full-segment scale (121 / 109 / 156 / 63 / 2 selected routes). |
| 3 | 49–50 | Full-segment Station calibration **REJECTED** (`cross_level_contaminated`, unique class `physical_nonidentifiability`). |
| 4 | 51 | Reduced Station calibration **REJECTED** by cross-run non-transferability. `{dy,rx,rz}` is Fisher-identifiable and nearly orthogonal to `A`, but the self-nulling correction does not transfer (max ~258σ; 14974→14973 linearized χ² ×16). |
| 5 | 52–53 | Current-geometry residual/DQ monitoring **PASS** on seven independent r0022 runs, all `nominal_monitoring`. `alignment_drift_candidate=false`. |

Any residual or χ² decrease in this package is a **DQ observable**
only. It is never alignment success.

## Frozen invariants

These are not retuned:

- Propagation: mode-0
- Association: frozen V2
  (`outputs/mc24_v3_expanded_trainval_v2_bce_control_v1`, SHA256
  `0c85a001677678163512c59bd5ceb6d08bdefaab79c0f4628659a4a8ad766a27`)
- Threshold `0.001`, `unmatched_penalty=-1.0`
- Observation: `anchor_selected_field_edge` + `physical_edge_deduplicated`
- Occupancy: first 100-event window meeting 16/16/32/8; next r0022
  segment if needed. Occupancy only finds a start. Official monitoring
  uses the full remaining segment.
- Geometry / conditions / reconstruction: `FASERNU-04` /
  `OFLCOND-FASER-06` / `r0022`
- Athena real-data flags: `--NoTrackFilt --no_stable`
- Leakage operator `A`: `A_dx=-59.213`, `A_ry=-31.908` µm per µm `C_dx`
- Isolation budget: 1.5–1.7 µm
- Reference residual scale from 14973/14974 field-edge residuals:
  `dy` median −3.810 mm, robust scale 6.712 mm; `rx` median −0.00178,
  scale 0.00524. Do not re-estimate.
- Alarms (entry 52, not retuned):
  1. `selected==0` and `all_pairs>=50` → `association_or_reconstruction_degradation`
  2. `0<selected<10` → `insufficient_statistics_for_alignment_dq`
  3. selected ≥ 10 and event share > 0.50 → association/reconstruction degradation
  4. isolation `|robust z|>5` vs 14973/14974 → `detector_condition_change`
  5. same-sign isolation `|z|≥3` on ≥ 2 non-reference, statistically
     sufficient runs → `alignment_drift_candidate` only, never a correction

Forbidden: FD probes; Newton; alignment payload writes; V2 retraining;
threshold or penalty changes; residual-based window re-picks; converting
an `alignment_drift_candidate` into a geometry update; reopening
self-nulling calibration; opening the sealed test.

## Final machine-readable state

```
real_data_operating_mode=residual_dq_monitoring_only
geometry_write_allowed=false
station_calibration_mode_available=false
cdx_mode_allowed=false
alignment_drift_candidate=false
```

The only allowed code path is

`current official geometry → frozen V2 → residual/DQ monitoring`.

Do not extract any alignment payload from real-data self-nulling
residuals.

## Unlock criteria for Operating Protocol V2

Machine-readable file:
`outputs/operating_protocol_v1_final_real_data_closure_v1/operating_protocol_v2_unlock_criteria.json`

`currently_met=false`. Either of the following may open V2; neither is
met now:

1. Independent survey or external station constraints fix `dx/ry/dz`,
   **and** an independent measurement proves that the true `|C_dx|`
   lies in the 1.5–1.7 µm isolation budget.
2. A new independent real-data track topology empirically demonstrates
   a calibration subspace that transfers across runs.

A met criterion only permits opening Operating Protocol V2. This
package still does not write geometry.

## Chart data for papers and group meetings

Under `outputs/operating_protocol_v1_final_real_data_closure_v1/chart_data/`:

| file | content |
| --- | --- |
| `selected_route_scaling.csv` | selected-route count vs event statistics (n100 / n1000 / n10000 / full, plus seven expansion full segments) |
| `route_composition.csv` | 2/3/4-station selected-route composition for all 12 runs |
| `dy_rx_robust_z_timeseries.csv` | `dy/rx` robust-z versus LHC fill |
| `station_jacobian_singular_spectrum.csv` | six-DoF and five-DoF singular values on 14973/14974 |
| `reduced_mode_cross_run_inconsistency.csv` | `{dy,rx,rz}` self-nulling deltas on 14973 vs 14974 |
| `frozen_A_versus_station_weak_direction.csv` | frozen `A` versus the station weak direction (14974 five-DoF cosine 0.994) |

Label every residual or χ² decrease **DQ observable**.

## Outputs

`outputs/operating_protocol_v1_final_real_data_closure_v1/`

- `operating_protocol_v1_final_report.json`
- `real_data_evidence_matrix.json`
- `reproducibility_manifest.json`
- `operating_protocol_v2_unlock_criteria.json`
- `chart_data/`
