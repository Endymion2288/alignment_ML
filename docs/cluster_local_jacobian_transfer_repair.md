# Cluster-Local Jacobian Transfer Exact-Join Repair V1

Workbook 70 / 2026-09-03. Workbook 68 skipped the cluster-local
diagnostic as `cluster_local_reference_or_transfer_unavailable`
because Frozen-V2 selected routes from transfer run 14974 were
joined against the workbook-57 r14973 feasibility dump. That dump
contains only run 14973, so the exact join produced zero
measurements.

This is a pure provenance repair. It is not an identifiability
campaign and is **not** a workbook-69 stable-core gate. The only
question is whether the cluster-local transfer join can be restored
without relaxing provenance.

It does not retrain V2/V3/Transformer, change the frozen
pairwise/route policy, restack 2024 r0022 collision-like tracks,
invent a cosine cut, run Newton, write geometry, or emit an
alignment payload. Real data stays `residual_dq_monitoring_only`.
A restored join does not reopen workbook 58
`cluster_local_identifiability_not_transferable`.

## Frozen contract

- Real data remains `residual_dq_monitoring_only`.
- `geometry_write_allowed=false`.
- Frozen V2 checkpoint SHA256
  `0c85a001677678163512c59bd5ceb6d08bdefaab79c0f4628659a4a8ad766a27`.
- Join key remains
  `run + event + TrackletHit_cluster_identifier =
  FaserSCT_Cluster.identify().get_compact()`.
- Fuzzy matching, residual/position nearest-neighbour join, and
  population reselection to force a non-zero match are forbidden.
- Wire the already-produced
  `outputs/true_cluster_local_ry_cdx_stability_transfer_v1/dumps/cluster_local_r14974.root`.
  Do not invent a new key.
- Keep the workbook-68 failure mode as a negative control: r14974
  routes × r14973 dump must stay zero.
- Restoring the exact join does not authorize a new cluster-local
  observable identifiability campaign and must not be merged into
  the stable-core gate.

## Decision

`cluster_local_transfer_exact_join_restored_not_an_identifiability_campaign`

| gate | value |
| --- | --- |
| exact join restored | **true** |
| fuzzy / nearest-neighbour join | **false** |
| wrong-dump control still zero | **true** |
| workbook-58 identifiability reopened | **false** |
| cluster-local identifiability campaign authorized | **false** |
| merged into stable-core gate | **false** |
| Frozen-V2 unknown-association | **not authorized** |
| real-data alignment correction | **not authorized** |
| geometry write | **false** |

Workbook 68’s zero match is reproduced as a run-id mismatch, not a
broken join key. Wiring the existing r14974 dump restores a
complete exact join. That restores provenance only.

## Provenance layers

Config
`configs/cluster_local_jacobian_transfer_repair_v1.yaml`, SHA256
`c761af1fa3cba38ee88c8c5e80cf819557483b65fe9d397d21be0d340e433e97`.

| layer | reference r14973 | transfer r14974 |
| --- | --- | --- |
| role | calibration_reference | independent_calibration_transfer |
| source_id | `data24_r14973_00007_skip49500_n84988` | `data24_r14974_00005_skip74400_n60886` |
| reconstruction | r0022 | r0022 |
| cluster dump | `true_cluster_local_residual_feasibility_v1/cluster_local.root` | `true_cluster_local_ry_cdx_stability_transfer_v1/dumps/cluster_local_r14974.root` |
| dump runs | `{14973: 4073}` | `{14974: 3843}` |
| selected routes | 121 | 109 |
| wanted cluster keys | 2473 | 2123 |
| exact joined keys | **2473 / 2473** | **2123 / 2123** |
| measurements | **454** | **403** |
| missing keys | 0 | 0 |

Negative control (workbook-68 failure mode): r14974 routes ×
r14973 dump gives run overlap 0, joined keys 0, measurements 0,
`mismatch_reason = run_id_disjoint_dump_does_not_contain_route_run`.

Join key unchanged. `fuzzy_join_used=false`,
`nearest_neighbour_join_used=false`,
`population_reselected=false`. Population remains the Frozen-V2
selected-route set from entries 57–58.

## Commands

```bash
source scripts/setup_environment.sh ml
pytest -q tests/test_cluster_local_jacobian_transfer_repair.py
python scripts/report_cluster_local_jacobian_transfer_repair.py
```

CUDA `True` (Tesla T4). Related tests included in the 26 passed.
No Condor job.

Reports:
`outputs/cluster_local_jacobian_transfer_repair_v1/`.
Recorded HEAD `a1fd01bb5c2c5daf432c13db3b021feb42bdf6d1`.

## Next stage

Keep residual DQ monitoring. The exact join can be restored without
relaxing provenance. That is not identifiability success and is not
stable-core success. A future cluster-local observable
identifiability study requires a new pre-registered config and
workbook entry, still using exact join only.
