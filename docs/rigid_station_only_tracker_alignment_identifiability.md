# Rigid-Station-Only Tracker Alignment Identifiability V1

Workbook 73 / 2026-09-03. Workbooks 68–72 remain frozen. This
campaign does not rescue the 7D tracklet-level observable, the
cluster-local observable, or the cross-source stable core. It does
not retune `rank_tolerance=0.01`, retune `S` from the spectrum, drop
failed sources, change the residual, redefine a stable core, or
reselect population from slope / residual / cosine.

The physical model was frozen before inspecting any 5DoF SVD.
Tracker fit may float only station rigid-body `dx, dy, rx, ry, rz`.
`dz` stays track-unidentifiable / survey-or-gauge-constrained and is
not given to tracks. Internal plane/module geometry and `C_dx` are
not free parameters; they stay fixed at the current reconstruction
geometry / metrology state. Fixing `C_dx` is a model boundary, not a
measurement that `C_dx=0`, not a survey verification of the current
`C_dx`, and not a licence to force-write old survey central values.
Survey/metrology remains an external cross-check only.

Native five-column `A = W^{1/2} J S` is reconstructed from
hierarchical V1 physical FD banks via `only_parameters`. Deleting
`dz` / `C_dx` from a previous 7D SVD object is refused. Pooled
rank=5 is not portability. Every pre-declared source-disjoint
validation and slope/topology coverage slice must pass.

It does not retrain V2/V3/Transformer, change Frozen V2, run Newton,
write geometry, or emit an alignment payload. Real data stays
`residual_dq_monitoring_only`. Three-arm, Frozen-V2 unknown
association, and real-data correction stay closed until
identifiability passes.

## Frozen contract

- Real data remains `residual_dq_monitoring_only`.
- `geometry_write_allowed=false`.
- Frozen V2 checkpoint SHA256
  `0c85a001677678163512c59bd5ceb6d08bdefaab79c0f4628659a4a8ad766a27`.
- Inherited decisions: workbook 68
  `tracker_only_identifiable_basis_unstable_solve_stopped`; workbook
  69 `cross_source_stable_core_independent_validation_fail`; workbook
  70 `cluster_local_transfer_exact_join_restored_not_an_identifiability_campaign`;
  workbook 71 `cluster_local_observable_not_cross_run_portable`;
  workbook 72
  `tracklet_independent_failure_provenance_audit_complete_sources_not_dropped`.
- Parameter order
  `ift_dx_mm, ift_dy_mm, ift_rx_mrad, ift_ry_mrad, ift_rz_mrad`.
- Units `mm, mm, mrad, mrad, mrad`.
- FD steps `0.5 mm, 0.5 mm, 10 mrad, 10 mrad, 10 mrad`.
- `S = (5, 5, 60, 60, 60)` mm/mrad, declared before SVD.
- `rank_tolerance = 0.01`.
- Residual weight `W` is the nominal 4×4 pair-covariance WLS already
  used by `solve_physical_finite_difference`.
- All 18 hierarchical V1 train/validation sources are included,
  including the workbook-69 rank 2/3/4 sources.
- Slope is `hypot(source_tx, source_ty)` of the truth-matched
  source-station tracklet local state at the FD anchor, not leftover
  residual `(rtx, rty)`.
- Topology is the complete four-station truth-route mask.
- This campaign is not workbook 36 and is not 7D column deletion.

## Decision

`rigid_station_five_dof_not_source_or_coverage_portable`

| gate | value |
| --- | --- |
| Jacobian validity | **true** |
| pooled identifiable rank | **5** (not portability) |
| every pre-declared source rank 5 | **false** (14/18) |
| source-disjoint subspace stable | **false** |
| train vs validation subspace | **true** |
| no slope-tertile rank drop | **false** |
| complete-route topology rank 5 | **false** |
| event bootstrap stable | **false** |
| failure class | **normal_physics_or_coverage_loss** |
| sources dropped | **false** |
| 7D column deletion | **false** |
| three-arm / Frozen-V2 / real-data correction | **not authorized** |
| geometry write | **false** |

With internal detector geometry fixed, current tracker data do
**not** determine station rigid-body 5DoF stably across sources and
slope coverage. The failure is not a Jacobian-construction artifact:
plus/minus probes are complete, populations align, residuals are
finite, no column is near zero, source slopes have zero missing
pairs, held-out points were not loaded, and sealed test was not
opened.

## Official spectrum

Config
`configs/rigid_station_only_tracker_alignment_identifiability_v1.yaml`,
SHA256
`80b5ebebe71e6d2bb20604c8593b97540cd998cbafee86e94d201dd6b00719ba`.

Pooled 4141 pairs: singular values
`16327, 2824, 527, 424, 265`, official rank **5**.

Native source ranks below five:

| source | pairs | official rank | σ₅/σ₁ or first sub-threshold |
| --- | ---: | ---: | --- |
| `mc24_100043_00400_00499` | 240 | **2** | σ₃/σ₁ = 0.0074 |
| `mc24_100043_00500_00599` | 234 | **3** | σ₄/σ₁ = 0.0053 |
| `mc24_100044_00200_00299` | 234 | **4** | σ₅/σ₁ = 0.0051 |
| `mc24_100044_00400_00499` | 235 | **4** | σ₅/σ₁ = 0.0089 |

The last of those four was a rank-5 7D sibling in workbook 72. Its
native 5DoF reconstruction is rank 4. That is expected: this
campaign did not delete columns from the 7D SVD. The source is not
dropped.

Slope tertiles (≥8 events/bin, 54 usable bins) drop rank on 9
sources, including several full-sample rank-5 sources. Angular
coverage remains a real failure mechanism, as in workbook 71.
Complete-route topology reproduces ranks 2/3/4 on the same four
low-rank sources.

Train versus validation pooled subspaces coincide. Instability is
rank change across pre-declared sources, tertiles, topology,
bootstrap (1/40), and half-splits (1/24), not large same-rank
principal angles.

## Commands

```bash
source scripts/setup_environment.sh ml
pytest -q tests/test_rigid_station_only_identifiability.py
python scripts/report_rigid_station_only_identifiability.py
```

CUDA `True` (Tesla T4). Campaign tests 8 passed; 35 passed with the
related identifiability suite. No Condor job.

Reports:
`outputs/rigid_station_only_tracker_alignment_identifiability_v1/`.
Recorded HEAD `a1fd01bb5c2c5daf432c13db3b021feb42bdf6d1`.

## Next stage

Do not recover rank 5 by dropping sources, deleting further DoF, or
retuning thresholds. Do not reopen 7D / cluster-local / stable-core
rescue. Keep residual DQ monitoring. Workbook 74 Stage 1 opened the
physically distinct track-coverage inventory and froze
`residual_blind_export_authorized_fd_not_opened`; it did not skip to
gauge / external constraint. The next algorithm change is still
forbidden.
