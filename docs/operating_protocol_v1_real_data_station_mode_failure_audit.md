# Operating Protocol V1 real-data Station Mode failure audit

Workbook 50 / 2026-08-22. This stage characterises the already observed
`cross_level_contaminated` result. It does **not** write official
conditions and does **not** start C_dx Mode.

The empty 100-event selected graph was statistics-limited. Full-segment
selected routes exist. The remaining failure is that the current
selected graph cannot satisfy station identifiability and cross-level
isolation at the same time.

Frozen V2, route policy, score threshold `0.001`, and
`unmatched_penalty=-1.0` are unchanged.

## Frozen invariants

- Canonical propagation: mode-0
- Association: frozen V2
  (`outputs/mc24_v3_expanded_trainval_v2_bce_control_v1`, SHA256
  `0c85a001…766a27`)
- Observation: `anchor_selected_field_edge` + `physical_edge_deduplicated`
- Leakage operator `A` and the 1.5–1.7 µm isolation budget remain the
  registered MC contract
- Blind roles: `14973/14974` calibration only for the verdict

Forbidden: retrain V2; retune thresholds or unmatched penalty; joint
station+`C_dx` Newton; new layer/module DoF; C_dx Mode; Schur production
estimator; treating residual reduction as closure; inverting implied
`|C_dx|` into a new payload; official conditions writes; opening the
sealed test.

This stage is local analysis of already-captured finite-difference
probes. No Condor jobs were required.

## DQ versus alignment correctness

| Product | Role |
| --- | --- |
| Selected-route counts, complete-four-station fraction, edge reuse, event concentration | Data quality only |
| `J_s` rank, condition number, singular spectrum | Identifiability diagnostic, **not** closure |
| Implied `|C_dx|` from frozen `A` and the self-nulling `dx/ry` | Rejection diagnostic, **not** a `C_dx` measurement |
| Residual drop after self-nulling | Never alignment success |

`geometry_write_allowed` stays false.

## 1. Identifiability audit

`J_s` is rebuilt from the 12 axial station finite-difference probes
already captured on the frozen selected-route edges. Residual
improvement is not scored.

| run | edges | 6-DoF rank | 6-DoF cond. | near-null | 5-DoF cond. (drop survey `dz`) |
| ---: | ---: | ---: | ---: | --- | ---: |
| 14973 | 118 | 6 | 4.90e5 | `dz` (\|v\|=1.000) | 114 |
| 14974 | 104 | 6 | 2.75e6 | `dz` (\|v\|=1.000) | 1.64e4 |

The 6-DoF near-null is survey `dz` on both calibration runs. `dx`,
`dy`, and `ry` are not the smallest 6-DoF direction. After dropping
`dz`, run 14973 is already below the MC admission gate `1e4`. Run
14974's remaining weak 5-DoF direction is almost pure `dx`
(\|v\|=0.993) and has cosine 0.994 with frozen `A`.

Two-station and three-station subsets are also `dz`-null. The two
complete four-station observations on 14974 are *more* degenerate in
`dz` (6-DoF condition ~3.4e9) than the mixed graph.

The self-nulling `dz` proposals (658 mm / 1250 mm) show that this
direction is not track-constrained. They are not a geometry to write.

## 2. Frozen-A back-projection

The registered MC operator (`A_dx=-59.21`, `A_ry=-31.91`, subspace
`r2=0.99966`) is applied to the self-nulling station correction. `A`
is not remeasured. No new `C_dx` payload is emitted
(`new_cdx_payload=null`).

| run | dx mm | ry mrad | implied `C_dx` from dx (µm) | implied `C_dx` from ry (µm) |
| ---: | ---: | ---: | ---: | ---: |
| 14973 | −1.30 | 20.0 | −22 | +628 |
| 14974 | +6.57 | 33.5 | +111 | +1049 |

Both exceed the 1.5–1.7 µm isolation budget. The dx and ry projections
disagree (opposite sign on 14973), so the numbers are a rejection
diagnostic, not a physical `C_dx` value.

## 3. Selected-route statistics

Current V2 association is retained.

| run | selected | complete 4-station | fraction | edge reuse | max event share |
| ---: | ---: | ---: | ---: | --- | ---: |
| 14973 | 121 | 0 | 0 | none | 0.008 |
| 14974 | 109 | 2 | 0.018 | none | 0.009 |

Complete four-station routes are rare. The station solve is dominated
by local two- and three-station field topology. That is a concurrent
statistics limit, not a V2 association failure: selected graphs are
non-empty and all-pairs graphs were already non-empty.

## 4. Analysis-only statistics control

No geometry is written. The Schur production estimator is not used.

- Same-topology scaling leaves the condition number invariant.
- Adding copies of the observed complete four-station normal block:
  - 6-DoF does **not** recover. The observed complete block is more
    degenerate in `dz` than the mixed graph.
  - 5-DoF: 14973 is already below `1e4`; 14974 would need ~758 copies
    of that block (~1515 complete routes, ~4.6e7 events at the
    observed rate). That is not practical, and the remaining weak
    direction stays aligned with `A`.
- `station_covariance_can_recover_by_more_complete_tracks=false`
- `practical_path_to_geometry_write=false`

More real data with the same V2 / occupancy mix cannot restore both
station 6-DoF covariance and cross-level isolation.

## Unique decision

**B. `physical_nonidentifiability`**

- Concurrent: A. `reconstruction_statistics_limitation` (complete
  four-station occupancy is also too low)
- Ruled out: C. `v2_association_failure`

`geometry_write_allowed=false`. C_dx Mode stays closed. The sealed
test stays closed. No new DoF. V2 is not retrained. A production
conditions path requires a redefined calibration mode, not this
Newton step.

## Outputs

Under `outputs/operating_protocol_v1_real_data_station_mode_failure_audit_v1/`:

- `station_identifiability_real_data_audit.json`
- `cdx_cross_level_contamination_audit.json`
- `route_statistics_limitation_audit.json`
- `operating_protocol_failure_classification.json`
