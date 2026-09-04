# Physically-Distinct Track-Coverage Identifiability Feasibility V1

Workbook 74 / 2026-09-03. Workbooks 59–73 remain frozen. This
campaign does not rescue the 7D tracklet-level observable, the
cluster-local observable, the cross-source stable core, or
rigid-station 5DoF. It does not retune `rank_tolerance=0.01`, retune
`S` from the spectrum, drop failed sources, delete `rz` or further
DoF, change the residual, redefine a stable core, retrain
V2/V3/Transformer, change pairwise/route policy, or restack more
2024 r0022 collision-like statistics.

The question is not whether more tracks raise rank. It is whether a
physically different track-angle, origin, or topology population
supplies the independent alignment information that current
collision-like and canonical 5 mrad FLUKA-E tracks lack, so that
rigid-station 5DoF becomes stably identifiable under source-disjoint
and coverage-disjoint conditions.

Stage 1 is a fully residual-blind coverage inventory. Candidates are
defined only from pre-existing physical metadata: provenance, run
type, generator/process, detector configuration, track angular phase
space, and station coverage. Residual, Jacobian singular values,
cosine, alignment response, and final rank are not selection
variables. Uncertain branch/run meaning is confirmed from Calypso,
ROOT metadata, GRL/run documentation, and existing reconstruction
outputs — never from a filename token such as `5mrad`. Local
SegmentFit `hypot(tx, ty)` is not a spectrometer wide-angle gate.
Observed distinctness is overlap with the canonical `(tx, ty)`
envelope only. Statistical gates were frozen before looking at
coverage numbers; a candidate below gate is `insufficient`. Gates
are not lowered.

This stage does not construct `A = W^{1/2} J S`, does not SVD, and
does not inspect rank. A later native 5DoF FD campaign is authorized
only after residual-blind inventory proves a candidate is
metadata-distinct, observed-`(tx, ty)`-distinct, and above frozen
statistical gates. Mixed pooled rank is not complementarity. The
gauge / external-constraint branch is not opened until inventory
strictly proves that no sufficient physically distinct 5DoF coverage
population exists in available reconstructions.

It does not retrain V2/V3/Transformer, change Frozen V2, run Newton,
write geometry, or emit an alignment payload. Real data stays
`residual_dq_monitoring_only`. Three-arm, Frozen-V2 unknown
association, and real-data correction stay closed.

## Frozen contract

- Real data remains `residual_dq_monitoring_only`.
- `geometry_write_allowed=false`.
- Frozen V2 checkpoint SHA256
  `0c85a001677678163512c59bd5ceb6d08bdefaab79c0f4628659a4a8ad766a27`.
- Inherited decisions: workbook 59
  `no_portable_alternative_topology_in_current_r0022`; workbook 60
  `real_track_topology_insufficient_for_ry_cdx_separation`; workbook
  68 `tracker_only_identifiable_basis_unstable_solve_stopped`;
  workbook 69 `cross_source_stable_core_independent_validation_fail`;
  workbook 70
  `cluster_local_transfer_exact_join_restored_not_an_identifiability_campaign`;
  workbook 71 `cluster_local_observable_not_cross_run_portable`;
  workbook 72
  `tracklet_independent_failure_provenance_audit_complete_sources_not_dropped`;
  workbook 73 `rigid_station_five_dof_not_source_or_coverage_portable`.
- Rigid-station 5DoF (not reconstructed this stage):
  `ift_dx_mm, ift_dy_mm, ift_rx_mrad, ift_ry_mrad, ift_rz_mrad`;
  units `mm, mm, mrad, mrad, mrad`; `S = (5, 5, 60, 60, 60)`;
  `rank_tolerance = 0.01`; `A = W^{1/2} J S`.
- Slope, if reported, is `hypot(source_tx, source_ty)` of the
  truth-matched source-station tracklet local state. It is not
  leftover residual `(rtx, rty)` and not spectrometer `Δx/Δz`. It
  is not an admission gate.
- Observed distinctness: `outside_canonical_quantile_box_fraction ≥
  0.20` **or** histogram intersection `≤ 0.80`.
- Statistical gates: `min_events=200`, `min_ift_events=200`,
  `min_complete_four_station_events=80`,
  `min_independent_sources_or_runs=2`, IFT + stations 0–3 required.
- Collision-like metadata, same production as canonical, sealed
  test, and coverage-unmeasured cannot admit FD.
- Unused 100043/044/047/048 files of the same 5 mrad FLUKA-E
  production are not a new population.
- Sealed `mc24_100116_00030_00039` and `mc24_100117_00030_00039`
  stay closed.
- This campaign is not workbook 36, 59, or 60 rank rescue.

## Decision

`residual_blind_export_authorized_fd_not_opened`

| gate | value |
| --- | --- |
| residual-blind Stage 1 | **true** |
| SVD / rank this stage | **false** |
| FD identifiability this stage | **false** |
| populations admitted to a separate FD campaign | **0** |
| residual-blind HTCondor export authorized | **2** (`mc24_100120_muon_floor`, `mc24_100130_kshort_end_fasernu`) |
| statistical gates lowered | **false** |
| local `hypot(tx,ty)` used as spectrometer wide-angle gate | **false** |
| filename `5mrad` treated as phase-space proof | **false** |
| 2024 r0022 collision-like restacked | **false** |
| unused 5 mrad files restacked | **false** |
| sealed test opened | **false** |
| freeze `current_track_coverage_insufficient_for_unconstrained_rigid_station_5dof` | **false** (inventory not closed) |
| open gauge / external-constraint branch | **false** |
| three-arm / Frozen-V2 / real-data correction | **not authorized** |
| geometry write | **false** |

No already-exported population is metadata-distinct,
observed-`(tx, ty)`-distinct, above frozen statistical gates, and
not metadata-forbidden. A later native 5DoF FD campaign is therefore
not opened. Two metadata-distinct xAODs without tracklets may be
exported residual-blind on HTCondor; that is not rank rescue and not
FD. Inventory has not yet proven that available reconstructions
contain no sufficient physically distinct population, so the gauge /
external-constraint branch stays closed. Workbook 61 remains frozen
for the older `ry↔C_dx` question.

## Canonical envelope

Canonical is hierarchical V1 iteration-00 train/validation FD-anchor
tracklets, not real 2024 r0022. Generator logs confirm
SingleParticle, `z = -3990 mm`, `theta = None`, `phi = 0..2π`,
energy `TH2Sampler`, FASERNU-04 / TI12MC04.

Pooled residual-blind truth-matched SegmentFit:

| quantity | value |
| --- | ---: |
| sources | 18 |
| events | 1790 |
| IFT events | 1615 |
| complete four-station events | 1433 |
| tracklets | 6690 |
| angular tracklets | 1695 |
| station events 0/1/2/3 | 1615 / 1705 / 1696 / 1674 |
| quantile box `tx` | `[-0.05600, 0.05311]` |
| quantile box `ty` | `[-0.01911, 0.01874]` |
| observed extrema `tx` | `[-0.19049, 0.16983]` |
| observed extrema `ty` | `[-0.09908, 0.16064]` |
| self histogram intersection | 1.0 |

## Candidate admissions

Config
`configs/physically_distinct_track_coverage_identifiability_feasibility_v1.yaml`,
SHA256
`2a8b17a362fa38c4fb9ad16fdb379bf4c7f8046fb38f1bc1f345776e21936caf`.
Schema
`faser-physically-distinct-track-coverage-identifiability-feasibility-v1`.
Created `2026-09-03T19:12:10.950820+00:00`. Recorded HEAD
`a1fd01bb5c2c5daf432c13db3b021feb42bdf6d1`.

| id | hypothesis | observed distinct | events / IFT / complete4 / sources | hist ∩ | outside box | verdict |
| --- | --- | --- | ---: | ---: | ---: | --- |
| `mc24_100012_100gev_gaussian_theta` | true | true (∩=0.764) | 20 / 18 / 17 / 4 | 0.764 | 0.050 | **insufficient** (below gates; gates not lowered) |
| `mc24_100116_117_2d_fluka_nonsealed` | true | **false** (∩=0.860) | 78 / 75 / 58 / 8 | 0.860 | 0.107 | **insufficient** (overlap + below gates) |
| `mc24_100049_050_calonu_5mrad` | **false** | true (∩=0.674, outside=0) | 5 / 5 / 5 / 1 | 0.674 | 0.0 | **insufficient** (hypothesis false; CaloNu export not authorized) |
| `mc24_100120_muon_floor` | true | unmeasured | 0 / 0 / 0 / 0 | — | — | **authorize residual-blind export** (not FD) |
| `mc24_100123_124_year_labeled_muon` | **false** | unmeasured | 0 / 0 / 0 / 0 | — | — | **insufficient** (year tag is not phase-space proof; export not authorized) |
| `mc24_100130_kshort_end_fasernu` | true | unmeasured | 0 / 0 / 0 / 0 | — | — | **authorize residual-blind export** (not FD) |
| `mc24_fluka_210010_full_shower` | true | unmeasured | 0 / 0 / 0 / 1 | — | — | **insufficient** (known-empty SegmentFit; export forbidden) |
| `real_2024_cosmic_beam_mode` | true | inherit 60 | — | — | — | **insufficient** (occupancy-empty; not restacked) |
| `real_2023_cosmic_like` | true | inherit 60 | — | — | — | **insufficient** (not high lever-arm; not restacked) |
| `real_backward_alps_ift_collision` | **false** | inherit 60 | — | — | — | **insufficient** (still collision-like) |
| `real_testbeam_2021` | true | inherit 60 | — | — | — | **insufficient** (no IFT) |
| `real_2024_r0022_wide_angle_selected_routes` | **false** | inherit 59 | complete4 selected routes = 17 | — | — | **insufficient** (`do_not_restack`) |

100012 is the closest already-exported population: ConstSampler
[100 GeV] + GaussianSampler theta, histogram intersection 0.764
≤ 0.80, therefore observed-distinct. Counts 20/18/17 are far below
200/200/80. Wide local slope fraction 0.70 is **not** an admission
reason. Gates are not lowered.

100116/117 non-sealed mag_0 train/val was hypothesized as a 2D FLUKA
sampler, but observed `(tx, ty)` still overlaps the canonical
envelope (∩=0.860). Sealed `00030_00039` was never opened.

100049 `--geom TI12MCCaloNu` is a detector-configuration hypothesis,
pre-registered as `physically_distinct_hypothesis: false`. The
filename token `5mrad` is not evidence of a wider phase space.
Remaining rec files exist; export is not authorized because the
hypothesis is false.

100120 origin `x=[-100,100], y=-120, z=[-2000,1000] mm` is
metadata-distinct floor/edge topology versus IP-like `z=-3990 mm`.
100130 is `pid=310`, `z=[-2000,-1951] mm` neutral-kaon gun. Neither
has tracklets, so coverage is unmeasured and FD is not admitted;
both satisfy the residual-blind export contract.

100123/124 year tags are not phase-space proof; the generator remains
`z=-3990 mm`, `theta=None`, TH2Sampler; hypothesis is false; export
is not authorized. FLUKA 210010 SegmentFit/Segments are empty in the
10-event and 100-event smokes (`known_empty_segmentfit: true`);
export is forbidden.

## Commands

```bash
source scripts/setup_environment.sh ml
pytest -q tests/test_physically_distinct_track_coverage.py
python scripts/report_physically_distinct_track_coverage.py
```

CUDA `True` (Tesla T4). Campaign tests 7 passed. No Condor job. No
`A = W^{1/2} J S` this stage.

Reports:
`outputs/physically_distinct_track_coverage_identifiability_feasibility_v1/`.
Recorded HEAD `a1fd01bb5c2c5daf432c13db3b021feb42bdf6d1`.

## Next stage

Do not open FD. Do not skip to gauge / external constraint. Do not
lower gates. Do not restack r0022 or unused 5 mrad files. The next
allowed technical step is HTCondor residual-blind tracklet export of
`mc24_100120_muon_floor` and `mc24_100130_kshort_end_fasernu`, then
repeat the coverage inventory. If that inventory still finds no
statistically sufficient physically distinct population, freeze
`current_track_coverage_insufficient_for_unconstrained_rigid_station_5dof`
and open Gauge-Constrained / External-Constraint Alignment
Feasibility. That step is not authorized yet.
