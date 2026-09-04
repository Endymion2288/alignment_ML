# Cross-Source Stable-Core Identifiable Subspace Definition and Independent Validation V1

Workbook 69 / 2026-09-03. Workbook 68 remains frozen as
`tracker_only_identifiable_basis_unstable_solve_stopped`. This
campaign does not reopen that V1 solve, does not retune
`rank_tolerance=0.01`, does not retune physical scale `S`, does not
drop the rank-6 source, and does not change selection, route policy,
or Frozen V2.

The question is not “force rank 5”. It is whether a
tracker-visible core subspace persists across sources and can be
moved, while the source-dependent sixth mode is isolated.

It does not retrain V2/V3/Transformer, change the frozen
pairwise/route policy, restack 2024 r0022 collision-like tracks,
invent a cosine cut, tune selection from residual or cosine, run
full-parameter Newton, write geometry, or emit an alignment payload.
Survey/metrology stays an external cross-check. Real data stays
`residual_dq_monitoring_only`. Cluster-local transfer repair is a
separate campaign and is not a gate here.

## Frozen contract

- Real data remains `residual_dq_monitoring_only`.
- `geometry_write_allowed=false`.
- Frozen V2 checkpoint SHA256
  `0c85a001677678163512c59bd5ceb6d08bdefaab79c0f4628659a4a8ad766a27`.
- Inherited workbook-68 decision
  `tracker_only_identifiable_basis_unstable_solve_stopped`.
- Form `A = W^{1/2} J S` with frozen
  `S = (5, 5, 5, 60, 60, 60, 0.12)` mm/mrad and frozen
  `rank_tolerance=0.01`. Do not SVD a naked mixed-unit Jacobian.
- Each source projector uses the **native frozen-rank SVD**,
  including rank-6 `mc24_100047_00150_00199`. That source is not
  truncated to 5-D and is not dropped.
- The seven V1 sources are **hypothesis construction only**. They
  are not confirmatory evidence.
- Core dimension is automatic from frozen eigenvalue and
  persistence cuts. It is not forced to 5.
- Independent confirmation uses already-produced source-disjoint
  hierarchical iteration-00 FD banks that were not inspected for the
  workbook-68 rank flip. Sealed test is not opened. Athena is not
  rerun. Criteria are not changed after seeing those sources.
- Three-arm, if opened, uses frozen `V_core`, not workbook-68
  pooled `V_id`. Arm-B injects the core-orthogonal complement.
- `dz` stays in the 7-parameter model. Removing it requires a new
  config citing the frozen 6-DoF unidentifiability map, not this
  SVD. `dx`–`C_dx` remains a possible observable or null
  combination.
- If independent validation fails, freeze the negative conclusion.
  Do not chase by retuning scale, rank, selection, source set, or
  model.

## Decision

`cross_source_stable_core_independent_validation_fail`

| gate | value |
| --- | --- |
| hypothesis `core_dimension` | **5** (automatic, not forced) |
| hypothesis LOSO / source-bootstrap stable | **true** |
| independent source-support fraction | **8/11 = 0.727 < 0.80** |
| `independent_validation_pass` | **false** |
| `null_injection_leakage_gate` | **null** (three-arm not opened) |
| `mixed_injection_projected_closure` | **null** (three-arm not opened) |
| Frozen-V2 unknown-association | **not authorized** |
| real-data alignment correction | **not authorized** |
| geometry write | **false** |

A 5-D consensus core can be constructed on the seven hypothesis
sources, and the sixth mode is isolated to the rank-6 source. That
core is LOSO- and source-bootstrap-stable. Independent confirmation
fails: three unseen FD sources have native ranks 2, 3 and 4, so
pre-registered persistence / principal-angle / projector gates fail.
Do not retune.

## Pre-registered algorithm

Config
`configs/cross_source_stable_core_identifiable_subspace_v1.yaml`,
SHA256
`4b3b3fba9d46fc8925682566ace4c56ba8b05420359594db4eb34e914973ca50`.

- Per-source projector `P_s = V_id,s V_id,s^T` from the native
  frozen-rank SVD of `A = W^{1/2} J S`.
- Equal-weight consensus `M = (1/N) Σ_s P_s`.
- Symmetric eigendecomposition, descending eigenvalues, sign
  convention `largest_abs_right_vector_entry_positive`.
- Prefix core: keep leading modes while eigenvalue ≥ 0.70 **and**
  source-support fraction of `q^T P_s q` ≥ 0.85 is ≥ 0.80. Stop at
  the first failure. Do not cherry-pick non-consecutive
  eigenvectors.
- LOSO: same core dimension, max principal angle ≤ 15 deg,
  projector Frobenius ≤ 1.0.
- Source-with-replacement bootstrap 40: max angle ≤ 20 deg,
  projector Frobenius ≤ 1.2.
- Independent gate: support fraction ≥ 0.80; per source,
  persistence ≥ 0.85, angle ≤ 15 deg, missing projector Frobenius
  ≤ 0.75.

Hypothesis sources (workbook-68 rank-flip set, construction only):

| split | source | pairs | native rank |
| --- | --- | ---: | ---: |
| train | `mc24_100043_00200_00299` | 232 | 5 |
| train | `mc24_100043_00600_00699` | 221 | 5 |
| train | `mc24_100044_00300_00399` | 240 | 5 |
| validation | `mc24_100047_00000_00049` | 231 | 5 |
| validation | `mc24_100047_00150_00199` | 240 | **6** |
| validation | `mc24_100048_00000_00049` | 216 | 5 |
| validation | `mc24_100048_00150_00199` | 223 | 5 |

Independent confirmation sources (same iteration-00 FD corpus,
file-disjoint, not used in the workbook-68 rank-flip inspection):
`mc24_100043_00300_00399`, `00400_00499`, `00500_00599`,
`mc24_100044_00200_00299`, `00400_00499`, `00500_00599`,
`00600_00699`, `mc24_100047_00050_00099`, `00100_00149`,
`mc24_100048_00050_00099`, `00100_00149`.

## Hypothesis consensus

Eigenvalues of `M`:
`[1.000000, 1.000000, 0.999995, 0.999964, 0.999558, 0.143304, 3.6e-5]`.

The first five eigenvalues are ≈ 1 with source-support fraction
1.0 and are labelled **core**. Mode 5 has eigenvalue 0.143 < 0.70
and support 1/7 = 0.143; it is labelled
**source_specific_or_null**. Mode 6 is ≈ 0. Automatic core
dimension is **5**. Persistence of mode 5 is ≈ 1 only on
`mc24_100047_00150_00199` and ≈ 0 on the other six sources. That
is the workbook-68 source-dependent sixth singular value, now
isolated from the core.

Leading scaled core compositions (reconstruction-observable linear
combinations, **not** lone mechanical parameters):

- 0: `ift_ry_mrad`
- 1: `ift_rx_mrad`
- 2: `ift_dy_mm`
- 3: `ift_rz_mrad`
- 4: `C_dx` mixed with `ift_dx_mm`

Core-orthogonal:

- `ift_dx_mm` 0.815 / `C_dx` 0.578 (recorded `dx`–`C_dx`
  degeneracy)
- `ift_dz_mm` 1.000 (track-unidentifiable `dz` from the 6-DoF map)

`V_core^⊥ u_hat = 0` remains the minimum-norm / gauge
representative, not a measurement that orthogonal modes are
physically zero.

## Hypothesis stability

Comparisons are subspaces, not signed vector elements.

| audit | stable | max angle | projector Frobenius |
| --- | --- | ---: | ---: |
| LOSO 7 | **yes** | 0.433 deg | 0.0107 |
| source-with-replacement 40 | **yes** | 1.049 deg | 0.0259 |
| event-bootstrap core persistence 56 | 56/56 pass | — | — |

Leaving out the rank-6 source still yields a 5-D core (0.032 deg).
All 40 source-bootstrap replicates stay 5-D. Event bootstrap of
`mc24_100047_00150_00199` still rank-flips 5↔6 (5 of 8 replicates
rank 6), but `V_core` persistence against the native projector
passes: the flip is isolated in mode 5. This is hypothesis
construction, not confirmation.

## Independent validation

Pre-registered support-fraction gate is 0.80. Result **8/11 =
0.727**.

| split | source | native rank | pass | max angle | min persistence | missing F |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| train | `mc24_100043_00300_00399` | 5 | yes | 0.773 | 0.9998 | 0.019 |
| train | `mc24_100043_00400_00499` | **2** | **no** | 71.0 | 0.0002 | 1.732 |
| train | `mc24_100043_00500_00599` | **3** | **no** | 88.3 | 0.0015 | 1.414 |
| train | `mc24_100044_00200_00299` | **4** | **no** | 18.7 | 0.106 | 1.000 |
| train | `mc24_100044_00400_00499` | 5 | yes | 1.135 | 0.9997 | 0.028 |
| train | `mc24_100044_00500_00599` | 5 | yes | 2.932 | 0.9981 | 0.072 |
| train | `mc24_100044_00600_00699` | 5 | yes | 0.195 | 1.0000 | 0.005 |
| validation | `mc24_100047_00050_00099` | 5 | yes | 3.005 | 0.9973 | 0.074 |
| validation | `mc24_100047_00100_00149` | 5 | yes | 2.491 | 0.9991 | 0.062 |
| validation | `mc24_100048_00050_00099` | 5 | yes | 0.514 | 0.9999 | 0.015 |
| validation | `mc24_100048_00100_00149` | 5 | yes | 0.365 | 1.0000 | 0.010 |

Failed sources are not truncated to 5-D. Whole core directions lie
outside those native projectors. Do not retune `rank_tolerance`,
drop those sources, or reuse the hypothesis seven as confirmation.

Three-arm **not opened**. A later opening must use this campaign’s
frozen `V_core`, not workbook-68 pooled `V_id`.

## Commands

```bash
source scripts/setup_environment.sh ml
python -c "import torch; print(torch.cuda.is_available())"
pytest -q tests/test_cross_source_stable_core.py tests/test_tracker_only_identifiable_subspace.py
python scripts/report_cross_source_stable_core.py
```

CUDA `True` (Tesla T4). 26 related tests passed. No Condor job:
independent sources already exist as FD banks.

Reports:
`outputs/cross_source_stable_core_identifiable_subspace_v1/`.
Recorded HEAD `a1fd01bb5c2c5daf432c13db3b021feb42bdf6d1`.

## Next stage

Keep residual DQ monitoring. Do not authorize Frozen-V2
unknown-association. The hypothesis 5-D core is **not** an
independently portable tracker-visible subspace under the
pre-registered gates. Do not chase this campaign by moving
`rank_tolerance`, retuning `S`, dropping sources, or forcing rank
5. A later Jacobian, residual, or association control requires a
new pre-registered config and workbook entry.
