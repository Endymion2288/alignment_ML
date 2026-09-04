# Cluster-Local Observable Cross-Run Identifiability and Transfer Validation V1

Workbook 71 / 2026-09-03. Workbooks 68 and 69 remain frozen
tracklet-level failures. Workbook 70 restored cluster-local transfer
exact join only. Entry 58 remains
`cluster_local_identifiability_not_transferable` until this campaign
classifies that failure under unified exact-join wiring.

The only question is whether, with strict exact join, the same
population, the same parameter definition, and pre-registered
weighting/scaling, the true cluster-local residual supplies more
stable, cross-run-portable alignment information than the earlier
tracklet-level `[rx, ry, rtx, rty]` observable.

Phase 1 uses r14973 as reference and r14974 as transfer only. It
does not solve a correction, run three-arm, or open the Frozen-V2
alignment loop.

It does not retrain V2/V3/Transformer, change the frozen
pairwise/route policy, restack 2024 r0022 collision-like tracks,
invent a cosine cut, tune selection from residual or cosine, run
full-parameter Newton, write geometry, or emit an alignment payload.
Survey/metrology stays an external cross-check. Real data stays
`residual_dq_monitoring_only`. Tracklet-level rescue is forbidden.

## Frozen contract

- Real data remains `residual_dq_monitoring_only`.
- `geometry_write_allowed=false`.
- Frozen V2 checkpoint SHA256
  `0c85a001677678163512c59bd5ceb6d08bdefaab79c0f4628659a4a8ad766a27`.
- Inherited decisions: entry 58
  `cluster_local_identifiability_not_transferable`; workbook 68
  `tracker_only_identifiable_basis_unstable_solve_stopped`; workbook
  69 `cross_source_stable_core_independent_validation_fail`; workbook
  70 `cluster_local_transfer_exact_join_restored_not_an_identifiability_campaign`.
- Official SVD is `A = W^{1/2} J S` with native millimetre /
  milliradian columns and frozen
  `S = (5, 60, 0.12)`, `rank_tolerance = 0.01`. Do not SVD a naked
  mixed-unit Jacobian as the official rank.
- The `ry` column of `parameter_columns` is per radian. Convert to
  1/mrad before forming `A`.
- Join key remains
  `run + event + TrackletHit_cluster_identifier =
  FaserSCT_Cluster.identify().get_compact()`.
- Residual is `r_u = u_cluster - u_track^unbiased` on
  `SiDetectorElement`. Tracklet intercepts at nominal layer z are
  not the observable.
- Runs are the statistical unit. Do not pool measurements and treat
  pooled rank as transferability.
- If a slope tertile with at least eight routes has identifiable
  rank below the full-sample rank, the extra dimension is
  coverage-conditioned and is not portable.
- Entry 58 naked SVD / `|cos(ry, C_dx)|` is a negative-control
  reproduction only.
- Workbook-70 join repair does not automatically overturn entry 58.
- Until all four gates pass: no three-arm, no Frozen-V2 alignment
  loop, no real-data correction, no geometry write, no extra runs.
- If the campaign fails, freeze the negative conclusion. Do not
  chase by retuning scale, rank, selection, or population.

## Decision

`cluster_local_observable_not_cross_run_portable`

| gate | value |
| --- | --- |
| `cluster_local_reference_jacobian_valid` | **true** |
| `cluster_local_transfer_jacobian_valid` | **true** |
| `cross_run_rank_consistent` | **true** (both official ranks 3) |
| `cross_run_subspace_transferable` | **false** (slope-tertile rank drop) |
| entry-58 negative control | still `cluster_local_identifiability_not_transferable` |
| workbook-70 wrong-dump still zero | **true** |
| Frozen-V2 unknown-association | **not authorized** |
| three-arm | **not authorized** |
| real-data alignment correction | **not authorized** |
| geometry write | **false** |

Exact join is complete and both Jacobians are valid. The full-sample
official subspaces almost coincide. The failure is the pre-registered
coverage gate: slope tertiles with ≥8 routes drop official rank from
3 to 2. The third dimension is coverage-conditioned, not a portable
identifiable subspace.

## Official spectrum

Config
`configs/cluster_local_observable_cross_run_identifiability_v1.yaml`,
SHA256
`dff4d132aed358dc37c0d76d95f67823de4844491147e3efec6bc598ff8d5577`.

| quantity | r14973 | r14974 |
| --- | ---: | ---: |
| exact joined keys | 2473 / 2473 | 2123 / 2123 |
| measurements | 454 | 403 |
| IFT routes | 82 | 74 |
| official rank | 3 | 3 |
| singular values | 100.85, 55.72, 1.667 | 97.30, 55.12, 1.604 |
| native \|cos(ry, C_dx)\| | 0.531 | 0.571 |
| slope-tertile ranks | **2 / 2 / 3** | **2 / 2 / 3** |

Full-sample max principal angle `8.5e-7` deg; projector Frobenius
`2.5e-16`; min mode persistence 1.0. Modes are reconstruction-
observable linear combinations and must not be relabelled as
measured `ry` or `C_dx`.

Entry 58 under the same exact join still fails: 14973 bootstrap of
`|cos(ry, C_dx)|` is not concentrated, coverage-matched resampling
cannot fill 27 / 27 / 28 (actual 26 / 24 / 22), and the decision
remains `cluster_local_identifiability_not_transferable`. Official
rank matches the naked rank (both 3), so the failure is not a
Jacobian-construction artefact.

## Commands

```bash
source scripts/setup_environment.sh ml
pytest -q tests/test_cluster_local_observable_cross_run.py
python scripts/report_cluster_local_observable_cross_run.py
```

CUDA `True` (Tesla T4). Related tests: 40 passed. No Condor job.

Reports:
`outputs/cluster_local_observable_cross_run_identifiability_v1/`.
Recorded HEAD `a1fd01bb5c2c5daf432c13db3b021feb42bdf6d1`.

## Next stage

Keep residual DQ monitoring. Do not chase tracker-only cluster-local
or tracklet observables by retuning thresholds. The parallel
workbook-69 failure-source audit (workbook 72) is read-only and is
not a confirmatory set.
