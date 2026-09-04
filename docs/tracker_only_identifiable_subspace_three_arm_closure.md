# Tracker-Only Identifiable-Subspace Definition and Three-Arm Closure V1

Workbook 68 / 2026-09-02. Entries 60–67 freeze survey/metrology as an
external cross-check, not an alignment input. This stage turns the
mainline to tracker-only identifiable-mode alignment: define the
identifiable subspace of the already-validated physical finite-difference
Jacobian, audit source-disjoint / bootstrap subspace stability, and open
three-arm linear MC closure only if that basis is stable.

It does not retrain V2/V3/Transformer, change the frozen pairwise/route
policy, restack 2024 r0022 collision-like tracks, invent a cosine cut,
tune selection from residual or cosine, run full-parameter Newton, write
geometry, or emit an alignment payload. Survey priors stay
`feasibility_only` / `unavailable`. Real data stays
`residual_dq_monitoring_only`. The first-round solver control is
truth-selected association. Frozen-V2 unknown-association closure is
not opened here.

## Frozen contract

- Real data remains `residual_dq_monitoring_only`.
- `geometry_write_allowed=false`.
- Frozen V2 checkpoint SHA256
  `0c85a001677678163512c59bd5ceb6d08bdefaab79c0f4628659a4a8ad766a27`.
- Survey/metrology is an external cross-check only (workbook 60–67).
- Do not SVD a naked mixed-unit Jacobian. Form
  `A = W^{1/2} J S` with the pre-declared scale matrix `S` and the
  existing WLS residual weight `W`.
- Do not retune `S` or the rank cut from singular values.
- Rank tolerance remains the frozen leakage/Fisher relative cut
  `1e-2`.
- Identifiable modes are reconstruction-observable linear combinations.
  They must not be relabelled as a lone mechanical `ry` or `C_dx`.
- `V_null^T u_hat = 0` is the minimum-norm / gauge representative in
  scaled coordinates, not a measurement that null modes are physically
  zero.
- Per-physical-parameter truth error is not a campaign gate.
- If the identifiable basis is unstable, stop the solve and freeze the
  negative conclusion. Do not chase by retuning scale, rank, selection,
  route policy, or model.

## Decision

`tracker_only_identifiable_basis_unstable_solve_stopped`

| gate | value |
| --- | --- |
| `identifiable_rank` | **5** |
| `identifiable_basis_stable` | **false** |
| `null_injection_leakage_gate` | **null** (three-arm not opened) |
| `mixed_injection_projected_closure` | **null** (three-arm not opened) |
| Frozen-V2 unknown-association | **not authorized** |
| real-data alignment correction | **not authorized** |
| geometry write | **false** |

The failure is a source-dependent rank change at the frozen relative
cut, not a large principal angle among same-rank sources. Do not retune
the cut.

## Jacobian, units, and scales

The definition Jacobian is the hierarchical V1 physical central
finite-difference bank already produced for workbook 44–45:

- corpus
  `outputs/mc24_ift_hierarchical_v1_iteration00_trainval_physical_v1/`
- residual 4-vector `[rx, ry, rtx, rty]` on truth-selected mode-0
  physical edges (`q_over_p_mode=0`, `min_truth_match_fraction=0.99`)
- `W` is the existing 4×4 pair covariance WLS weight, not an
  independent same-cluster covariance
- native units: millimetres for translations / `C_dx`, milliradians
  for rotations, matching
  `physical_jacobian.COMPONENT_INDEX_AND_PAYLOAD_SCALE`
- first-round loader is FD-only (reference + axial probes); held-out
  physical points are not mixed into the SVD
- Athena is not rerun

Seven source-disjoint files (not the full 18-source reload) are used:

| split | source | pairs | rank | `σ6/σ1` |
| --- | --- | ---: | ---: | ---: |
| train | `mc24_100043_00200_00299` | 232 | 5 | 0.001785 |
| train | `mc24_100043_00600_00699` | 221 | 5 | 0.003005 |
| train | `mc24_100044_00300_00399` | 240 | 5 | 0.002265 |
| validation | `mc24_100047_00000_00049` | 231 | 5 | 0.004125 |
| validation | `mc24_100047_00150_00199` | 240 | **6** | **0.012897** |
| validation | `mc24_100048_00000_00049` | 216 | 5 | 0.001695 |
| validation | `mc24_100048_00150_00199` | 223 | 5 | 0.003179 |

Pooled: 1603 pairs, 6412 observations, 7 parameters.

Pre-declared `S` (frozen `severity_scale`, never retuned from `Σ`):

| parameter | unit | `S` |
| --- | --- | ---: |
| `ift_dx_mm` | mm | 5 |
| `ift_dy_mm` | mm | 5 |
| `ift_dz_mm` | mm | 5 |
| `ift_rx_mrad` | mrad | 60 |
| `ift_ry_mrad` | mrad | 60 |
| `ift_rz_mrad` | mrad | 60 |
| `C_dx` | mm | 0.12 |

Right singular vectors live in dimensionless `u = S^{-1} θ`. Mode sign
convention: the largest-magnitude right-vector entry is positive.

## Pooled SVD of `A = W^{1/2} J S`

Pooled singular values

```text
[3729.82, 1747.30, 292.11, 170.81, 164.27, 28.16, 2.52]
```

Relative to `σ1`:

```text
[1.000, 0.4685, 0.0783, 0.0458, 0.0440, 0.00755, 0.00067]
```

Frozen cut `σ_k > 0.01 σ1` ⇒ pooled identifiable rank **5**, null
dimension **2**. The WLS normal matrix at `rcond=1e-10` is rank 7; that
is not the campaign rank. The campaign rank is the frozen relative SVD
cut on `A`.

Identifiable modes (scaled composition, `|comp| > 0.15`):

| mode | leading | composition | `σ` | `σ/σ1` |
| ---: | --- | --- | ---: | ---: |
| 0 | `ift_ry_mrad` | `ift_ry_mrad` 0.999 | 3729.82 | 1.000 |
| 1 | `ift_rx_mrad` | `ift_rx_mrad` 0.998 | 1747.30 | 0.468 |
| 2 | mix | `C_dx` 0.796, `ift_dx_mm` −0.571, `ift_rz_mrad` 0.164 | 292.11 | 0.078 |
| 3 | `ift_dy_mm` | `ift_dy_mm` 0.988 | 170.81 | 0.046 |
| 4 | `ift_rz_mrad` | `ift_rz_mrad` 0.984 | 164.27 | 0.044 |

Null modes:

| mode | leading | composition | `σ` | `σ/σ1` |
| ---: | --- | --- | ---: | ---: |
| 0 | `ift_dx_mm` | `ift_dx_mm` 0.815, `C_dx` 0.575 | 28.16 | 0.00755 |
| 1 | `ift_dz_mm` | `ift_dz_mm` 1.000 | 2.52 | 0.00067 |

These are reconstruction-observable linear combinations. Mode 0 is not
“the station `ry`”. Mode 2 is not “`C_dx`”. Null mode 0 is the
`dx`–`C_dx` degeneracy already recorded in the hierarchical leakage
work; null mode 1 is the unidentifiable `dz` of the 6-DoF map. Neither
null direction is a measurement of zero.

## Subspace stability

Stability compares identifiable **subspaces** by principal angles and
projector Frobenius distance. Signed vector-element equality is not
required. Rank change fails the pre-declared same-rank gate.

| audit | stable | n | max same-rank angle | notes |
| --- | --- | ---: | ---: | --- |
| source-disjoint vs pooled | **false** | 7 | 4.28 deg | `validation:mc24_100047_00150_00199` rank 6 (`σ6/σ1=0.0129`) |
| train vs validation (pooled) | true | 1 | 5.89 deg | projector 0.145 |
| complete-truth-route topology | true | 1 | 0.07 deg | projector 0.0017 |
| event bootstrap | **false** | 40 | 3.88 deg | 7/40 rank-flip to 6 |
| event half-split | **false** | 24 halves | 3.90 deg | 6/24 rank-flip to 6 |

Same-rank sources sit at 3.5–4.3 deg / projector 0.09–0.11. The
instability is the frozen `1e-2` cut sitting next to a weakly
source-dependent sixth singular value (~1.3 % of `σ1` on one
validation source, ≪1 % elsewhere). That is a rank-definition
instability, not a large rotation of a five-dimensional subspace.

Because the basis is unstable, the three-arm solve is **not opened**.
Arm-A / Arm-B / Arm-C remain un-evaluated on this Jacobian. Synthetic
unit tests still prove the solver math: amplitudes `a` then
`Δθ = S V_id a`; mixed injection recovers `P_id q_truth` only;
`V_null^T u_hat = 0` is gauge.

## Cluster-local diagnostic (not a gate)

Software FD of true cluster-local `r_u` on `{station_dx, station_ry,
C_dx}` is recorded as a diagnostic only. The ry column is converted
from per-radian to per-mrad, then `A = W^{1/2} J S` is formed with
the frozen cluster-local scale map. Naked mixed-unit SVD is refused.

The live diagnostic was skipped:
`cluster_local_reference_or_transfer_unavailable`. Reference run 14973
loads; transfer run 14974 has zero measurements after joining the
entry-57 cluster dump (that dump is the r14973 feasibility file). This
does not enter the four campaign gates and is not a real-data
alignment solve.

## What was not done

- No full-parameter Newton.
- No Frozen-V2 unknown-association closure.
- No real-data alignment correction.
- No geometry / POOL / COOL / alignment payload.
- No retune of `S`, rank tolerance, selection, route policy, or model.
- No sealed-test access.
- No 2024 r0022 collision-like restack.
- Survey numbers from workbook 66–67 are not alignment inputs.

## Reports

`outputs/tracker_only_identifiable_subspace_three_arm_closure_v1/`

Config SHA256
`1657019a2c0746835d490fcf58ce91b70b1cb846f73bc29e2ad69e0b5a2f7237`.
Recorded HEAD `a1fd01bb5c2c5daf432c13db3b021feb42bdf6d1`.
12 unit/regression tests passed. CUDA `True` (Tesla T4). Interactive
SVD / bootstrap / small closure; no Condor job.

## Next stage

Keep residual DQ monitoring. Do not authorize Frozen-V2
unknown-association until a **stable** identifiable subspace is
defined without retuning this campaign’s `S` or rank cut. If a later
stage changes the Jacobian (different residual, different association
control, or a genuinely new observation), it must pre-register a new
config and a new workbook entry. It must not move `rank_tolerance`
from `0.01` to `0.013` to absorb `mc24_100047_00150_00199`.
