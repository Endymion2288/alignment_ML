# Operating Protocol V1 reduced Station calibration mode feasibility

Workbook 51 / 2026-08-23. This stage does **not** patch the original
Station Mode. It asks which station degrees of freedom real data can
safely constrain. It is analysis only.

`geometry_write_allowed` stays false. C_dx Mode is not started. Frozen
V2, route policy, threshold `0.001`, and `unmatched_penalty=-1.0` are
unchanged. `dz` is removed from the track-driven solve and fixed at
survey 0.

The 12 station finite-difference probes already captured on 14973/14974
are reused. No Condor jobs were required.

## Frozen invariants

- Propagation: mode-0
- Association: frozen V2 (`0c85a001…766a27`)
- Observation: `anchor_selected_field_edge` + `physical_edge_deduplicated`
- Leakage operator `A` and the 1.5–1.7 µm isolation budget
- Verdict roles: 14973/14974 only. 14977 is never used for selection

Forbidden: official conditions writes; any self-nulling correction;
C_dx Mode; joint Newton; new layer/module DoF; opening the sealed test;
treating residual reduction as success; rescuing a `dx`/`ry` isolation
failure with more events or a looser threshold.

## DQ versus alignment correctness

| Product | Role |
| --- | --- |
| Reduced-mode rank, condition, σ | Identifiability diagnostic, not closure |
| Implied \|C_dx\| from floated `dx`/`ry` | Rejection diagnostic, not a measurement |
| Linearized χ² change | Transfer DQ only; a drop is never success |
| Blind-block residual IQR at current geometry | Data quality only |

## Pre-declared modes

`dz` is never floated.

| mode | floated | 14973 cond. | 14974 cond. | A projection | implied \|C_dx\| | run-to-run | admitted |
| --- | --- | ---: | ---: | ---: | --- | --- | --- |
| `{dy,rx,rz}` | 3 | 73 | 86 | 0.040 | no `dx`/`ry` | inconsistent (max 258σ) | no |
| `{dy,rx,ry,rz}` (`dx` fixed) | 4 | 91 | 1624 | 0.476 | 1126 / 479 µm | inconsistent | hard reject |
| `{dx,dy,rx,rz}` (`ry` fixed) | 4 | 89 | 90 | 0.881 | 60 / 268 µm | inconsistent | hard reject |
| `{dx,dy,rx,ry,rz}` | 5 | 114 | 1.64e4 | 1.00 | 20–1100 µm | inconsistent | hard reject |

Any mode that still floats `dx` or `ry` and exceeds the 1.5–1.7 µm
band is rejected immediately. That isolation failure cannot be rescued
by more events or a looser threshold.

`{dy,rx,rz}` is full rank, condition-stable, and clearly orthogonal to
`A` on both calibration runs. That is a Fisher-space result only. The
self-nulling point estimates do not travel (`dy` +1.08 vs −8.93 mm,
`rz` −4.3 vs +95 mrad). Applying the 14974 correction to 14973
worsens linearized χ² by a factor of ~16. It is **not** a V2
candidate.

Smaller isolation-safe subsets (`{dy}`, `{rx}`, `{rz}`, and the three
pairs) are also full rank and A-orthogonal. None are run-to-run
consistent.

## Common identifiable subspace

Axis cosines with frozen `A`:

- `dx` 0.880, `ry` 0.474 — leakage directions
- `dy` 0.025, `rx` 0.030, `rz` 0.005 — isolation-safe

The recommended Fisher subspace is `{dy,rx,rz}`. `dx`, `ry`, and `dz`
would go to survey/external alignment **if** that subspace also passed
run-to-run consistency and transfer. It does not, so no writable
reduced mode is defined.

## Read-only transfer DQ

14975/14976/14977 have no finite-difference `J_s`, so no candidate
geometry is applied. At current geometry, residual IQRs on 14975 and
14976 overlap the calibration blocks. 14977 has only two selected
routes and is insufficient to judge. 14977 is not used for the
verdict. Residual drop is never alignment success.

## Unique decision

**`real_data_residual_dq_monitoring_only`**

- No Real-Data Station Calibration Mode V2 candidate exists
- Every station DoF, including the isolation-safe `{dy,rx,rz}` Fisher
  axes, is assigned to survey/external alignment
- `geometry_write_allowed=false`
- No self-nulling correction is written
- C_dx Mode stays closed

Current real data can only support dedicated residual/DQ monitoring.
It cannot produce a station geometry update.

## Outputs

Under `outputs/operating_protocol_v1_real_data_reduced_station_mode_feasibility_v1/`:

- `reduced_mode_identifiability_audit.json`
- `reduced_mode_leakage_audit.json`
- `reduced_mode_transfer_dq_report.json`
- `reduced_station_mode_feasibility_report.json`
- `operating_protocol_next_decision.json`
